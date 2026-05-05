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
from server.pipeline.listing_etl import run_listing_etl
from server.pipeline.rule_feature_engineering import apply_rule_text_features
from server.tools.analysis_plain_summary import build_analysis_plain_summary
from server.tools.cleaning_housing import derive_floor_band_column
from server.tools.composite_field_parse import expand_composite_listing_columns
from server.tools.data_quality import assess_clean_quality, coach_clean_retry_hints
from server.tools.listing_numeric_parse import finalize_listing_dataframe, slim_cleaned_export_dataframe
from server.tools.export import write_excel_report
from server.tools.io import canonicalize_known_aliases, iter_tabular_paths, load_raw_directory
from server.tools.registry import build_cleaning_tools_for_session


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


def _prior_user_messages_for_intent(chat_messages: list[dict[str, Any]], *, max_segments: int = 8) -> list[str]:
    """提取近期用户原话，供房源查询意图继承城区/户型等多轮约束。"""
    out: list[str] = []
    for m in chat_messages:
        if m.get("role") != "user":
            continue
        t = (m.get("content") or "").strip()
        if t:
            out.append(t)
    return out[-max_segments:]


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


def _build_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.houseinsight_llm_model,
        api_key=settings.dashscope_api_key or None,
        base_url=settings.dashscope_base_url,
        temperature=0.2,
    )


def _build_llm_intent_parser(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.houseinsight_llm_model,
        api_key=settings.dashscope_api_key or None,
        base_url=settings.dashscope_base_url,
        temperature=0.0,
    )


def _aggregated_chat_context(st: SessionState) -> str:
    """对话只附聚合信息，不塞入全量列名或宽表原文。"""
    lines: list[str] = ["[会话聚合信息]"]
    if st.df_clean is not None:
        lines.append(f"清洗表行数约 {len(st.df_clean)}")
    if (st.analysis_summary_plain or "").strip():
        excerpt = st.analysis_summary_plain.strip()[:700]
        lines.append("分析总结摘录：\n" + excerpt)
    an = st.analysis if isinstance(st.analysis, dict) else {}
    uq = an.get("unit_price_quantiles")
    if uq:
        lines.append(f"单价分位数：{json.dumps(uq, ensure_ascii=False)}")
    ds = an.get("district_summary")
    if isinstance(ds, list) and ds:
        lines.append("城区摘要（前若干条）：" + json.dumps(ds[:8], ensure_ascii=False))
    tr = an.get("task_results")
    if isinstance(tr, dict) and tr:
        ok_n = sum(1 for v in tr.values() if isinstance(v, dict) and v.get("ok"))
        lines.append(f"规划分析任务成功数：{ok_n}/{len(tr)}")
    if st.df_clean is not None:
        keys = [
            c
            for c in (
                "district",
                "community",
                "layout",
                "layout_normalized",
                "area_m2",
                "unit_price",
                "build_year",
                "tag_near_subway",
                "tag_subway_station_hint",
                "description_hint_subway",
                "description_hint_school",
            )
            if c in st.df_clean.columns
        ]
        if keys:
            lines.append("关键列（仅列名，无逐行数据）：" + ", ".join(keys))
    return "\n".join(lines)


def run_chat_turn(
    store: SessionStore, session_id: str, user_text: str, settings: Settings | None = None
) -> str:
    settings = settings or get_settings()
    st = store.require(session_id)
    llm = _build_llm(settings)
    history = st.chat_messages[-12:]
    listing_block = ""
    sample_count = 0
    sample_max_rows = 0
    if settings.dashscope_api_key and st.df_clean is not None and len(st.df_clean) > 0:
        try:
            from server.tools.chat_listing_query import (
                apply_listing_search_intent,
                listings_to_llm_block,
                parse_listing_search_intent,
            )

            intent_llm = _build_llm_intent_parser(settings)
            prior_msgs = _prior_user_messages_for_intent(st.chat_messages)
            intent = parse_listing_search_intent(
                user_text,
                intent_llm,
                prior_user_messages=prior_msgs if prior_msgs else None,
            )
            if intent and intent.needs_row_samples:
                sub, query_relax_note = apply_listing_search_intent(st.df_clean, intent)
                sample_count = len(sub)
                sample_max_rows = int(intent.max_rows)
                if sample_count == 0:
                    note_prefix = f"\n\n{query_relax_note}" if query_relax_note else ""
                    listing_block = (
                        note_prefix
                        + "\n\n[查询结果] pandas 对会话清洗表的查询结果为 **0 条**（非摘要里的总套数）。"
                        "如实告知用户未找到；说明可能是城区关键词、必选标签或噪声剔除导致候选为空；"
                        "给一条放宽建议（如去掉某一硬条件）。禁止用聚合信息臆造房源列表。"
                    )
                else:
                    relax_prefix = (
                        f"\n\n[查询说明] {query_relax_note}\n" if query_relax_note else ""
                    )
                    listing_block = (
                        relax_prefix
                        + f"\n\n[查询结果] 以下为 pandas 查询命中行（共 {sample_count} 条，单次最多 {sample_max_rows}）。"
                        "仅可引用 JSON 中出现的行与字段；禁止用会话聚合里的总套数、均价等编造具体房源或「虚构套数」。\n"
                        + listings_to_llm_block(sub)
                    )
        except Exception:
            pass

    msgs: list[Any] = [
        SystemMessage(
            content=(
                "你是二手房数据分析助手。只能基于对话中给出的【会话聚合信息】、analysis 结论、"
                "以及（若有）【pandas 查询 JSON 样本】作答，必须遵守以下规则：\n"
                "1) 行级房源列表**仅**来自 JSON；聚合信息里的「总套数、分区统计」不得当成本次筛选结果，不得用来编造具体房源或套数。\n"
                "2) 如实告知数量：JSON 有 N 条就报 N 条（含 0）；不要扩展、不要凑齐到任何上限。\n"
                "3) 严禁编造：不得引用 JSON 之外的房源、小区、价格、字段；字段为空（NaN/null/空串）视为不满足，不要猜测。\n"
                "4) 命中较少或为零时：说明 pandas 硬条件/城区收缩可能导致候选为空；给一条可操作的放宽建议，不得为凑数把不符样本说成符合。\n"
                "5) 输出风格：有 JSON 时用短条目或要点；只问统计/概念时可用聚合信息；信息不足时直接说明。"
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
    summary = "\n\n" + _aggregated_chat_context(st)
    msgs.append(HumanMessage(content=user_text + summary + listing_block))
    ai = llm.invoke(msgs)
    text = getattr(ai, "content", str(ai))
    full_reply = text
    st.chat_messages.append({"role": "user", "content": user_text})
    st.chat_messages.append({"role": "assistant", "content": full_reply})
    return full_reply


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
            f"清洗（第 {st.clean_attempt_count}/{settings.houseinsight_max_clean_attempts} 轮）…",
            phase="clean.start",
            step_id="clean_round",
        )

        use_legacy = settings.houseinsight_legacy_agent_clean and bool(settings.dashscope_api_key)

        if use_legacy:
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
            st.df_clean = st.df_raw.copy()
            _emit(
                store,
                sid,
                "clean",
                38,
                "旧版：LLM ReAct 清洗 Agent（仅对照/排障，生产请关 HOUSEINSIGHT_LEGACY_AGENT_CLEAN）…",
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
                result = agent.invoke({"messages": [HumanMessage(content=user_content)]})
                msgs = result.get("messages", [])
                last = msgs[-1] if msgs else None
                note = getattr(last, "content", str(last)) if last else ""
                st.cleaning_notes = (st.cleaning_notes + "\n" + str(note)).strip()
                _emit(
                    store,
                    sid,
                    "clean",
                    46,
                    "旧版路径：Agent 后对表做规则加固（复合列+finalize）…",
                    phase="clean.legacy_finalize",
                )
                st.df_clean = expand_composite_listing_columns(st.df_clean)
                st.df_clean = finalize_listing_dataframe(st.df_clean)
                if "floor" in st.df_clean.columns:
                    st.df_clean = derive_floor_band_column(st.df_clean, "floor")
                st.cleaning_trace.append("legacy_agent_clean_then_finalize")
            except Exception as e:
                _emit(
                    store,
                    sid,
                    "clean",
                    40,
                    "旧版 LLM 清洗异常，改为一键确定性 ETL…",
                    phase="clean.fallback_rules",
                    step_id="clean_default",
                )
                st.df_clean, note = run_listing_etl(st.df_raw.copy())
                fallback_head = (
                    "【清洗说明】旧版智能体出错，已改用确定性 ETL。\n"
                    f"（报错摘要）{e}"
                )
                st.cleaning_notes = (st.cleaning_notes + "\n\n" + fallback_head + "\n\n" + note).strip()
                st.cleaning_trace.append("run_listing_etl(after_legacy_agent_error)")
        else:
            _emit(
                store,
                sid,
                "clean",
                38,
                "一键确定性 ETL（复合列规则 + 数值化 + 去重 + IQR + 楼层档），不经 LLM 改表…",
                phase="clean.etl",
                step_id="run_listing_etl",
            )
            st.df_clean, note = run_listing_etl(st.df_raw.copy())
            st.cleaning_notes = (
                (st.cleaning_notes + "\n" + note).strip() if st.cleaning_notes else note
            ).strip()
            st.cleaning_trace.append("run_listing_etl")

        _emit(store, sid, "clean", 52, f"第 {st.clean_attempt_count} 轮清洗完成", phase="clean.done")
        return {"session_id": sid, "error": None}

    def node_feature_engineering(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        if st.df_clean is None or st.df_clean.empty:
            return {"session_id": sid, "error": None}
        _emit(
            store,
            sid,
            "clean",
            50,
            "规则文本特征（无 LLM）+ 列 finalize…",
            phase="clean.rule_features",
            step_id="rule_features",
        )
        try:
            st.df_clean, feat_note = apply_rule_text_features(st.df_clean)
            st.cleaning_notes = (
                (st.cleaning_notes + "\n" + feat_note).strip() if st.cleaning_notes else feat_note
            ).strip()
            st.cleaning_trace.append("apply_rule_text_features")
        except Exception as e:
            _emit(
                store,
                sid,
                "clean",
                51,
                f"规则特征跳过：{e}",
                phase="clean.rule_features",
                step_id="features_skip",
            )
        return {"session_id": sid, "error": None}

    def node_enrich_description(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        if st.df_clean is None or st.df_clean.empty:
            return {"session_id": sid, "error": None}
        _emit(
            store,
            sid,
            "clean",
            54,
            "可选 L3：描述文本弱特征（抽样 LLM，失败则跳过）…",
            phase="clean.description_enrich",
            step_id="l3_description",
        )
        try:
            from server.tools.description_enrich import enrich_description_columns

            st.df_clean = enrich_description_columns(st.df_clean, settings)
        except Exception as e:
            _emit(
                store,
                sid,
                "clean",
                55,
                f"L3 描述增强跳过：{e}",
                phase="clean.description_enrich",
                step_id="l3_skip",
            )
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
            _emit(store, sid, "quality", 58, "质检通过，进入分析与导出")
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
        # df_clean 已在 feature_engineering 节点 finalize；此处直接使用会话主表
        base = st.df_clean if st.df_clean is not None else st.df_raw
        base = base if base is not None else pd.DataFrame()

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
        tr = st.analysis.get("task_results")
        plain = build_analysis_plain_summary(
            base,
            st.analysis,
            tr if isinstance(tr, dict) else None,
        )
        st.analysis_summary_plain = plain
        st.analysis["analysis_summary_plain"] = plain
        _emit(store, sid, "analyze", 75, "分析完成")
        return {"session_id": sid, "error": None}

    def node_export(state: PipelineState) -> PipelineState:
        sid = state["session_id"]
        if state.get("error"):
            return state
        st = store.require(sid)
        df = st.df_clean if st.df_clean is not None else st.df_raw
        out_dir = paths.output_dir(sid)
        out_dir.mkdir(parents=True, exist_ok=True)

        _emit(store, sid, "export", 90, "写出轻量 Excel 样本…", phase="export.xlsx", step_id="report_xlsx")
        xlsx_path = write_excel_report(out_dir, df if df is not None else pd.DataFrame(), st.analysis)
        st.artifacts["report.xlsx"] = str(xlsx_path)
        st.artifacts.pop("report.html", None)
        st.artifacts.pop("report.pdf", None)

        if st.return_cleaned_file and df is not None and not df.empty:
            _emit(store, sid, "export", 96, "写出清洗结果 CSV…", phase="export.cleaned_csv", step_id="cleaned_csv")
            clean_path = out_dir / "cleaned.csv"
            slim_cleaned_export_dataframe(df).to_csv(clean_path, index=False, encoding="utf-8-sig")
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
    g.add_node("feature_engineering", node_feature_engineering)
    g.add_node("enrich_description", node_enrich_description)
    g.add_node("quality_gate", node_quality_gate)
    g.add_node("analyze", node_analyze)
    g.add_node("export", node_export)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "clean")
    g.add_edge("clean", "feature_engineering")
    g.add_edge("feature_engineering", "enrich_description")
    g.add_edge("enrich_description", "quality_gate")
    g.add_conditional_edges(
        "quality_gate",
        route_after_quality,
        {"clean": "clean", "analyze": "analyze"},
    )
    g.add_edge("analyze", "export")
    g.add_edge("export", END)
    return g.compile()
