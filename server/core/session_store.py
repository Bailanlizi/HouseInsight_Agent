from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionState:
    session_id: str
    created_at: str = field(default_factory=utc_now_iso)
    stage: str = "idle"
    progress_pct: int = 0
    last_message: str = ""
    df_raw: pd.DataFrame | None = None
    df_clean: pd.DataFrame | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    chat_messages: list[dict[str, str]] = field(default_factory=list)
    cleaning_notes: str = ""
    clean_attempt_count: int = 0
    quality_report: dict[str, Any] = field(default_factory=dict)
    quality_coach_hint: str = ""
    cleaning_trace: list[str] = field(default_factory=list)
    analysis_plan: list[dict[str, Any]] = field(default_factory=list)
    analysis_plan_raw: str = ""
    analysis_summary_markdown: str = ""
    error: str | None = None

    def touch(self, stage: str, pct: int, msg: str) -> None:
        self.stage = stage
        self.progress_pct = pct
        self.last_message = msg


class SessionStore:
    """内存会话存储 + 每会话 WebSocket 订阅队列。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._ws_queues: dict[str, list[asyncio.Queue[str]]] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

    def bind_main_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._main_loop = loop

    def schedule_emit(self, session_id: str, payload: dict[str, Any]) -> None:
        """供线程池中的流水线节点回调进度（主线程事件循环上执行 emit）。"""
        loop = self._main_loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.emit(session_id, payload), loop)

    def create_session(self) -> SessionState:
        sid = uuid.uuid4().hex
        st = SessionState(session_id=sid)
        self._sessions[sid] = st
        self._ws_queues.setdefault(sid, [])
        return st

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> SessionState:
        st = self.get(session_id)
        if st is None:
            raise KeyError(session_id)
        return st

    def subscribe_ws(self, session_id: str) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self._ws_queues.setdefault(session_id, []).append(q)
        return q

    def unsubscribe_ws(self, session_id: str, q: asyncio.Queue[str]) -> None:
        queues = self._ws_queues.get(session_id)
        if not queues:
            return
        try:
            queues.remove(q)
        except ValueError:
            pass

    async def emit(self, session_id: str, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        for q in list(self._ws_queues.get(session_id, [])):
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass
