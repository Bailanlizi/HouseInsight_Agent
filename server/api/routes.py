from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from server.agent.house_agent import build_pipeline_graph, pipeline_event, run_chat_turn
from server.api.deps import get_paths, get_store
from server.core.config import get_settings
from server.core.paths import ProjectPaths
from server.core.session_store import SessionStore

router = APIRouter(tags=["houseinsight"])


class SessionCreated(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class ChatOut(BaseModel):
    reply: str
    sources: list[dict[str, str]] = Field(default_factory=list)


class RunPipelineBody(BaseModel):
    return_cleaned_file: bool = False
    skip_full_report_export: bool = True


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/sessions", response_model=SessionCreated)
async def create_session(store: SessionStore = Depends(get_store)) -> SessionCreated:
    st = store.create_session()
    return SessionCreated(session_id=st.session_id)


@router.post("/sessions/{session_id}/upload")
async def upload_files(
    session_id: str,
    files: list[UploadFile] = File(...),
    store: SessionStore = Depends(get_store),
    paths: ProjectPaths = Depends(get_paths),
) -> dict[str, object]:
    try:
        store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    if not files:
        raise HTTPException(status_code=400, detail="no files")
    raw_dir = paths.raw_dir(session_id)
    saved: list[str] = []
    for uf in files:
        name = Path(uf.filename or "upload.bin").name
        dest = raw_dir / name
        content = await uf.read()
        dest.write_bytes(content)
        saved.append(name)
    preview = ", ".join(saved[:6]) + ("…" if len(saved) > 6 else "")
    pipeline_event(
        store,
        session_id,
        "upload",
        8,
        f"已保存 {len(saved)} 个文件到 raw" + (f"（{preview}）" if preview else ""),
        phase="upload.files",
        step_id="upload.saved",
    )
    return {"session_id": session_id, "saved": saved}


@router.post("/sessions/{session_id}/run")
async def run_pipeline(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
) -> dict[str, str]:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None

    body = RunPipelineBody()
    try:
        raw = await request.body()
        if raw.strip():
            body = RunPipelineBody.model_validate_json(raw.decode("utf-8"))
    except Exception:
        body = RunPipelineBody()
    st.return_cleaned_file = body.return_cleaned_file
    st.skip_full_report_export = body.skip_full_report_export

    paths: ProjectPaths = request.app.state.paths
    settings = get_settings()
    graph = build_pipeline_graph(store, paths, settings)

    def _invoke() -> None:
        try:
            graph.invoke({"session_id": session_id})
        except Exception as e:
            st = store.get(session_id)
            if st:
                st.error = str(e)
                st.touch("error", 0, str(e))
            store.schedule_emit(session_id, {"stage": "error", "pct": 0, "msg": str(e)})

    async def _job() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _invoke)

    asyncio.create_task(_job())
    return {"status": "started", "session_id": session_id}


@router.get("/sessions/{session_id}/status")
async def session_status(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, object]:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    return {
        "session_id": session_id,
        "stage": st.stage,
        "progress_pct": st.progress_pct,
        "last_message": st.last_message,
        "error": st.error,
        "artifacts": st.artifacts,
    }


@router.get("/sessions/{session_id}/analysis")
async def get_analysis(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, object]:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    return {
        "session_id": session_id,
        "analysis": st.analysis,
        "analysis_plan": st.analysis_plan,
        "analysis_plan_raw": st.analysis_plan_raw,
        "analysis_summary_markdown": st.analysis_summary_markdown,
        "cleaning_trace": st.cleaning_trace,
        "quality_report": st.quality_report,
        "clean_attempt_count": st.clean_attempt_count,
        "quality_coach_hint": st.quality_coach_hint,
    }


@router.get("/sessions/{session_id}/run_result")
async def get_run_result(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, object]:
    """聚合首屏：分析摘要、图表键、产物键、质检与进度时间线尾部。"""
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    fig_keys = list(st.figures.keys())
    total_chars = sum(len(v) for v in st.figures.values())
    qr = st.quality_report
    quality_brief: dict[str, object] = {}
    if isinstance(qr, dict):
        quality_brief = {
            "passed": qr.get("passed"),
            "failures": qr.get("failures"),
            "metrics": qr.get("metrics"),
        }
    return {
        "session_id": session_id,
        "stage": st.stage,
        "progress_pct": st.progress_pct,
        "last_message": st.last_message,
        "error": st.error,
        "analysis": st.analysis,
        "analysis_summary_markdown": st.analysis_summary_markdown,
        "figures_keys": fig_keys,
        "figures_payload_chars": total_chars,
        "figures_too_large_for_inline": total_chars > 500_000,
        "artifacts": st.artifacts,
        "quality_report": st.quality_report,
        "quality_brief": quality_brief,
        "progress_events": st.progress_events[-120:],
        "options": {
            "return_cleaned_file": st.return_cleaned_file,
            "skip_full_report_export": st.skip_full_report_export,
        },
    }


@router.get("/sessions/{session_id}/figures")
async def get_figures(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, object]:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    return {"session_id": session_id, "figures": st.figures}


@router.get("/sessions/{session_id}/artifacts")
async def list_artifacts(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, object]:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    return {"session_id": session_id, "artifacts": st.artifacts}


@router.get("/sessions/{session_id}/artifacts/download")
async def download_artifact(
    session_id: str,
    name: str,
    store: SessionStore = Depends(get_store),
) -> FileResponse:
    try:
        st = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    path_str = st.artifacts.get(name)
    if not path_str:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(path, filename=path.name)


@router.post("/sessions/{session_id}/chat", response_model=ChatOut)
async def chat(session_id: str, body: ChatIn, store: SessionStore = Depends(get_store)) -> ChatOut:
    try:
        store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found") from None
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail="DASHSCOPE_API_KEY 未配置，无法对话")
    reply, sources = run_chat_turn(store, session_id, body.message, settings)
    return ChatOut(reply=reply, sources=sources)
