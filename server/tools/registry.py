"""汇总 Agent 可用工具（清洗）及模块导出接口。"""

from __future__ import annotations

from server.core.session_store import SessionStore
from server.tools.cleaning import make_cleaning_tools
from server.tools.cleaning_housing import make_housing_cleaning_tools

__all__ = ["build_cleaning_tools_for_session"]


def build_cleaning_tools_for_session(store: SessionStore, session_id: str):
    """画像与二手房原语在前，通用数值清洗在后。"""
    return [*make_housing_cleaning_tools(store, session_id), *make_cleaning_tools(store, session_id)]
