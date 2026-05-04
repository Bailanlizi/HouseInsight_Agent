from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from server.core.config import Settings, get_settings
from server.core.paths import ProjectPaths
from server.core.session_store import SessionState, SessionStore, utc_now_iso
from server.tools.analysis import analyze_second_hand_listings
from server.tools.analysis_narrative import enrich_analysis_markdown
from server.tools.analysis_plan import plan_analysis_with_llm, run_planned_analysis
from server.tools.cleaning import apply_default_cleaning_pipeline
from server.tools.data_quality import assess_clean_quality, coach_clean_retry_hints
from server.tools.listing_numeric_parse import finalize_listing_dataframe
from server.tools.export import maybe_render_pdf, render_report_html, write_excel_report
from server.tools.io import canonicalize_known_aliases, iter_tabular_paths, load_raw_directory
from server.tools.registry import build_cleaning_tools_for_session
from server.tools.viz import figures_from_analysis


class PipelineState(TypedDict, total=False):
    session_id: str
    error: str | None


_CLEANING_BEHAVIOR_GUIDE = """\
【成都二手房数据范例 — 你应按类似逻辑处理】
• 房屋信息「3室2厅|89平米|南|精装|中楼层(共26层)|2016年建|板楼」→ 拆成户型、面积、朝向、装修、楼层、建成年份、结构；过渡列务必使用标准键名：layout_str、area_m2_str、decoration_str、building_type_str、orientation_str、floor_text、build_year 等，再 apply_column_rename 到正式列；勿发明未注册键。
• 关注信息「135人关注/6个月以前发布」→ followers、发布时间相关列。
• 「153万」「17190元/平米」「89㎡」→ 写入 total_price(万元)、unit_price(元/㎡)、area_m2；解析会在收尾自动加固。
• 区域「青羊」等 → district。
分析侧偏好：区域均价用横向条形更易读；户型对比均价；面积段统计 + 占比饼图。"""


_MAX_PROGRESS_EVENTS = 200


def _emit(
    store: SessionStore,
    session_id: str,
    stage: str,
    pct: int,
    msg: str,
    *,
    phase: str | None = None,
    step_id: str | None = None,
    event: str | None = None,
) -> None:
    st = store.require(session_id)
    st.touch(stage, pct, msg)
    payload: dict[str, object] = {"stage": stage, "pct": pct, "msg": msg, "ts": utc_now_iso()}
    if phase is not None:
        payload["phase"] = phase
    if step_id is not None:
        payload["step_id"] = step_id
    if event is not None:
        payload["event"] = event
    st.progress_events.append(dict(payload))
    if len(st.progress_events) > _MAX_PROGRESS_EVENTS:
        del st.progress_events[: len(st.progress_events) - _MAX_PROGRESS_EVENTS]
    store.schedule_emit(session_id, payload)


def pipeline_event(
    store: SessionStore,
    session_id: str,
    stage: str,
    pct: int,
    msg: str,
    *,
    phase: str | None = None,
    step_id: str | None = None,
    event: str | None = None,
) -> None:
    """供 API 层（如上传完成）写入与流水线一致的进度事件。"""
    _emit(store, session_id, stage, pct, msg, phase=phase, step_id=step_id, event=event)


def _chat_columns_mentioned_in_question(user_text: str, columns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        if c and c in user_text:
            out.append(c)
    return out[:16]


def _build_chat_provenance(
    st: SessionState, user_text: str
) -> tuple[str, list[dict[str, str]]]:
    n = len(st.df_clean) if st.df_clean is not None else 0
    cols = list(st.df_clean.columns) if st.df_clean is not None else []
    hinted = _chat_columns_mentioned_in_question(user_text, cols)
    col_preview = ", ".join(cols[:12]) + ("…" if len(cols) > 12 else "")
    footer_lines = [
        "",
        "---",
        "【数据来源】",
        f"· 会话清洗表 df_clean：约 {n} 行，{len(cols)} 列（列名示例：{col_preview or '无'}）。",
        "· 统计与图表：基于本会话流水线生成的 analysis / figures，非实时重算全表。",
        "· 未对原始上传文件做逐行人工核对；若问题超出摘要范围，结论可能不完整。",
    ]
    if hinted:
        footer_lines.append(f"· 本问与列名显式相关：{', '.join(hinted)}。")
    footer = "\n".join(footer_lines)
    sources: list[dict[str, str]] = [
        {"label": "df_clean", "detail": f"{n} 行 × {len(cols)} 列"},
        {"label": "analysis", "detail": "会话内结构化分析结果"},
    ]
    if hinted:
        sources.append({"label": "提及列", "detail": ", ".join(hinted)})
    return footer, sources


def _build_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.houseinsight_llm_model,
        api_key=settings.dashscope_api_key or None,
        base_url=settings.dashscope_base_url,
        temperature=0.2,
    )


def run_chat_turn(
    store: SessionStore, session_id: str, user_text: str, settings: Settings | None = None
) -> tuple[str, list[dict[str, str]]]:
    settings = settings or get_settings()
    st = store.require(session_id)
    llm = _build_llm(settings)
    history = st.chat_messages[-12:]
    msgs: list[Any] = [
        SystemMessage(
            content=(
                "你是二手房数据分析助手。只能基于对话中给出的【数据集摘要】与已知的 analysis 结论作答；"
                "禁止编造表中不存在的列名、行号或具体房源；信息不足时请直接说明。"
            )
        ),
    ]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    summary = ""
    if st.df_clean is not None:
        aq = st.analysis.get("unit_price_quantiles") if isinstance(st.analysis, dict) else None
        extra = f"；单价分位数摘要: {aq}" if aq else ""
        summary = (
            f"\n\n[数据集摘要] 行数={len(st.df_clean)}, 列={list(st.df_clean.columns)}{extra}"
        )
    msgs.append(HumanMessage(content=user_text + summary))
    ai = llm.invoke(msgs)
    text = getattr(ai, "content", str(ai))
    footer, sources = _build_chat_provenance(st, user_text)
    full_reply = text + footer
    st.chat_messages.append({"role": "user", "content": user_text})
    st.chat_messages.append({"role": "assistant", "content": full_reply})
    return full_reply, sources


def build_pipeline_graph(store: SessionStore, paths: ProjectPaths, settings: Settings):
    def node_ingest(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        try:
            _emit(store, sid, "ingest", 10, "读取并合并原始表格…", phase="ingest.read", step_id="ingest_start")
            raw_dir = paths.raw_dir(sid)
            df, ingest_warnings = load_raw_directory(raw_dir)
            df = canonicalize_known_aliases(df)
            st = store.require(sid)
            st.progress_events.clear()
            st.df_raw = df
            st.df_clean = df.copy()
            st.clean_attempt_count = 0
            st.quality_report = {}
            st.quality_coach_hint = ""
            st.cleaning_notes = ""
            msg = f"合并完成，行数={len(df)}"
            if ingest_warnings:
                tail = "; ".join(ingest_warnings[:6])
                extra = f" …共 {len(ingest_warnings)} 条" if len(ingest_warnings) > 6 else ""
                msg += f"。下列文件未读取（已跳过）: {tail}{extra}"
                skipped_names = [w.split(":", 1)[0].strip() for w in ingest_warnings if ":" in w]
                if skipped_names and all(n.startswith("._") for n in skipped_names):
                    msg += "（提示：未读文件多为 ._ 附属占位；真实表通常为同名无前缀文件。）"
            _emit(store, sid, "ingest", 25, msg, phase="ingest.merge", step_id="ingest_done")
            return {"session_id": sid, "error": None}
        except Exception as e:
            err_text = str(e)
            store.require(sid).error = err_text
            extra = ""
            try:
                raw_dir = paths.raw_dir(sid)
                names = [p.name for p in iter_tabular_paths(raw_dir)]
                if len(names) > 0 and all(n.startswith("._") for n in names):
                    extra = " 提示：目录内均为 ._ 前缀文件时，请改用不带 ._ 的表格文件。"
            except Exception:
                pass
            _emit(store, sid, "error", 0, err_text + extra)
            return {"session_id": sid, "error": str(e)}

    def node_clean(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        if st.df_clean is None:
            err = "清洗失败：无数据"
            st.error = err
            return {"session_id": sid, "error": err}

        st.clean_attempt_count += 1
        if st.clean_attempt_count > 1 and st.df_raw is not None:
            st.df_clean = st.df_raw.copy()
            st.cleaning_notes = (
                st.cleaning_notes + "\n\n--- 质检未通过：已从原始表重置并开始第 "
                f"{st.clean_attempt_count} 轮清洗 ---\n"
            ).strip()

        st.cleaning_trace.clear()
        _emit(
            store,
            sid,
            "clean",
            36,
            f"智能清洗（第 {st.clean_attempt_count}/{settings.houseinsight_max_clean_attempts} 轮）…",
            phase="clean.start",
            step_id="clean_round",
        )
        tools = build_cleaning_tools_for_session(store, sid)

        base_user = (
            "请清洗当前会话中的二手房挂牌表（成都等 Excel 常见格式）：\n"
            + _CLEANING_BEHAVIOR_GUIDE
            + "\n执行：先 get_dataset_profile，再按需拆分/映射/数值化，保证 district、layout、area_m2、"
            "total_price、unit_price 可用于统计。"
        )
        if st.clean_attempt_count > 1 and (st.quality_coach_hint or "").strip():
            user_content = (
                base_user
                + "\n\n【上一轮质检反馈】请优先解决下列问题后再收尾：\n"
                + st.quality_coach_hint.strip()
            )
        else:
            user_content = base_user

        if settings.dashscope_api_key:
            _emit(
                store,
                sid,
                "clean",
                38,
                "调用 LLM 与领域工具进行缺失值/拆分等处理…",
                phase="clean.llm_agent",
                step_id="clean_llm",
            )
            try:
                llm = _build_llm(settings)
                agent = create_agent(
                    model=llm,
                    tools=tools,
                    system_prompt=(
                        "你是二手房挂牌数据清洗专家（ReAct：观察画像→调用工具→再观察），"
                        "只能通过工具修改会话中的 df_clean。"
                        "流程：先 get_dataset_profile；再按需 split_delimited（|）、split_slash（/）、"
                        "derive_floor_band、normalize_decoration、coerce_followers、apply_column_rename。"
                        "拆分出的过渡列必须使用 house_schema 已定义的标准键（含 layout_str、area_m2_str、"
                        "decoration_str、building_type_str、orientation_str、floor_text、publish_time_raw、"
                        "followers_str 等），禁止臆造未注册键名。"
                        "若收到「质检反馈」，优先按要求修正；复杂表可先领域拆分再 run_full_default_clean。"
                        "完成后简短中文总结步骤。"
                    ),
                )
                result = agent.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_content),
                        ]
                    }
                )
                msgs = result.get("messages", [])
                last = msgs[-1] if msgs else None
                note = getattr(last, "content", str(last)) if last else ""
                st.cleaning_notes = (st.cleaning_notes + "\n" + str(note)).strip()
            except Exception as e:
                _emit(
                    store,
                    sid,
                    "clean",
                    40,
                    "LLM 清洗异常，切换默认规则（去重、数值化、IQR 等）…",
                    phase="clean.fallback_rules",
                    step_id="clean_default",
                )
                st.df_clean, note = apply_default_cleaning_pipeline(st.df_raw.copy())
                fallback_head = (
                    "【清洗说明】智能体在某次工具调用中出错，已自动改用默认规则并完成数值解析加固；"
                    "下方数据仍可用于分析与图表。\n"
                    f"（报错摘要）{e}"
                )
                st.cleaning_notes = (st.cleaning_notes + "\n\n" + fallback_head + "\n\n" + note).strip()
                st.cleaning_trace.append("apply_default_cleaning_pipeline(fallback_after_llm_error)")
        else:
            _emit(
                store,
                sid,
                "clean",
                38,
                "无 API Key：使用默认规则（数值化、单价推算、去重、IQR）…",
                phase="clean.default_rules",
                step_id="clean_default",
            )
            st.df_clean, note = apply_default_cleaning_pipeline(st.df_raw.copy())
            st.cleaning_notes = note
            st.cleaning_trace.append("apply_default_cleaning_pipeline(no_api_key)")

        _emit(
            store,
            sid,
            "clean",
            50,
            "解析「万/元/㎡」文本、过渡列合并、缺失单价填补…",
            phase="clean.numeric_finalize",
            step_id="finalize_listing",
        )
        st.df_clean = finalize_listing_dataframe(st.df_clean)
        _emit(store, sid, "clean", 52, f"第 {st.clean_attempt_count} 轮清洗完成（已文本数值加固）", phase="clean.done")
        return {"session_id": sid, "error": None}

    def node_quality_gate(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        _emit(store, sid, "quality", 56, "数据质检（守门）…", phase="quality.rules", step_id="quality_start")
        report = assess_clean_quality(st.df_raw, st.df_clean, settings)
        st.quality_report = report
        if report.get("passed"):
            st.quality_coach_hint = ""
            _emit(store, sid, "quality", 58, "质检通过，进入分析与可视化")
        else:
            st.quality_coach_hint = coach_clean_retry_hints(settings, report)
            fails = ",".join(report.get("failures") or [])
            _emit(
                store,
                sid,
                "quality",
                58,
                f"质检未通过（{fails or '未知'}），"
                f"行保留率={report.get('metrics', {}).get('row_retention_ratio', '?')}",
            )
        return {"session_id": sid, "error": None}

    def route_after_quality(state: PipelineState) -> str:
        if state.get("error"):
            return "analyze"
        st = store.require(state["session_id"])
        if st.quality_report.get("passed"):
            return "analyze"
        if st.clean_attempt_count < settings.houseinsight_max_clean_attempts:
            return "clean"
        return "analyze"

    def node_analyze(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        base = st.df_clean if st.df_clean is not None else st.df_raw
        base = base if base is not None else pd.DataFrame()
        base = finalize_listing_dataframe(base)

        _emit(store, sid, "analyze", 62, "生成分析计划…", phase="analyze.plan", step_id="plan_llm")
        planned_tasks, raw_plan = plan_analysis_with_llm(base, settings)
        st.analysis_plan = [t.model_dump(mode="json") for t in planned_tasks]
        st.analysis_plan_raw = raw_plan or json.dumps(st.analysis_plan, ensure_ascii=False)

        _emit(
            store,
            sid,
            "analyze",
            68,
            f"执行分析任务（{len(planned_tasks)} 项）…",
            phase="analyze.execute",
            step_id="execute_tasks",
        )
        legacy = analyze_second_hand_listings(base)
        planned_out = run_planned_analysis(base, planned_tasks)
        st.analysis = {**legacy, **planned_out}
        summary = enrich_analysis_markdown(settings, base, {**st.analysis})
        if st.quality_report.get("passed") is False:
            summary = (
                "【质检提示】已达到最大清洗次数或本轮质检未通过，以下为降级分析，请结合 quality_report 排查。\n"
                + summary
            )
        st.analysis["analysis_summary_markdown"] = summary
        st.analysis_summary_markdown = summary
        _emit(store, sid, "analyze", 75, "分析完成")
        return {"session_id": sid, "error": None}

    def node_viz(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        df = st.df_clean if st.df_clean is not None else pd.DataFrame()
        df = finalize_listing_dataframe(df) if not df.empty else df
        _emit(store, sid, "viz", 80, "生成交互图表…", phase="viz.build", step_id="plotly")
        st.figures = figures_from_analysis(df, st.analysis)
        _emit(store, sid, "viz", 88, "图表完成", phase="viz.done")
        return {"session_id": sid, "error": None}

    def node_export(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        df = st.df_clean if st.df_clean is not None else st.df_raw
        out_dir = paths.output_dir(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_md = st.analysis_summary_markdown or st.analysis.get("analysis_summary_markdown") or ""

        _emit(store, sid, "export", 90, "写出轻量 Excel 样本…", phase="export.xlsx", step_id="report_xlsx")
        xlsx_path = write_excel_report(out_dir, df if df is not None else pd.DataFrame(), st.analysis)
        st.artifacts["report.xlsx"] = str(xlsx_path)

        if not st.skip_full_report_export:
            _emit(store, sid, "export", 93, "生成 HTML 报告…", phase="export.html", step_id="report_html")
            html_path = render_report_html(
                paths, sid, st.analysis, st.figures, st.cleaning_notes, narrative=summary_md
            )
            st.artifacts["report.html"] = str(html_path)
            pdf_path = out_dir / "report.pdf"
            pdf = maybe_render_pdf(html_path, pdf_path)
            if pdf:
                st.artifacts["report.pdf"] = str(pdf)
            else:
                st.artifacts.pop("report.pdf", None)
        else:
            st.artifacts.pop("report.html", None)
            st.artifacts.pop("report.pdf", None)

        if st.return_cleaned_file and df is not None and not df.empty:
            _emit(store, sid, "export", 96, "写出清洗结果 CSV…", phase="export.cleaned_csv", step_id="cleaned_csv")
            clean_path = out_dir / "cleaned.csv"
            df.to_csv(clean_path, index=False, encoding="utf-8-sig")
            st.artifacts["cleaned.csv"] = str(clean_path)
        else:
            st.artifacts.pop("cleaned.csv", None)

        st.stage = "done"
        st.progress_pct = 100
        _emit(
            store,
            sid,
            "done",
            100,
            "流水线完成（分析结果已就绪，可与 Agent 对话）",
            phase="pipeline.done",
            step_id="run_complete",
            event="run_complete",
        )
        return {"session_id": sid, "error": None}

    g = StateGraph(PipelineState)
    g.add_node("ingest", node_ingest)
    g.add_node("clean", node_clean)
    g.add_node("quality_gate", node_quality_gate)
    g.add_node("analyze", node_analyze)
    g.add_node("viz", node_viz)
    g.add_node("export", node_export)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "clean")
    g.add_edge("clean", "quality_gate")
    g.add_conditional_edges(
        "quality_gate",
        route_after_quality,
        {"clean": "clean", "analyze": "analyze"},
    )
    g.add_edge("analyze", "viz")
    g.add_edge("viz", "export")
    g.add_edge("export", END)
    return g.compile()
