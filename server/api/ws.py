from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.core.session_store import SessionStore

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(
    websocket: WebSocket,
    session_id: str,
) -> None:
    store: SessionStore = websocket.app.state.store
    try:
        store.require(session_id)
    except KeyError:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    q = store.subscribe_ws(session_id)
    try:
        await websocket.send_json(
            {"stage": store.require(session_id).stage, "pct": store.require(session_id).progress_pct, "msg": "connected"}
        )
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=25.0)
                await websocket.send_text(line)
            except asyncio.TimeoutError:
                await websocket.send_json({"stage": "ping", "pct": -1, "msg": "keepalive"})
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe_ws(session_id, q)
