from fastapi import Request

from server.core.paths import ProjectPaths
from server.core.session_store import SessionStore


def get_store(request: Request) -> SessionStore:
    return request.app.state.store


def get_paths(request: Request) -> ProjectPaths:
    return request.app.state.paths
