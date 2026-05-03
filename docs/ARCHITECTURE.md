# HouseInsight Agent 架构说明

## 目录职责

| 目录 | 说明 |
|------|------|
| `server/core` | 配置（`Settings`）、`SessionStore`、工程路径解析 |
| `server/tools` | 房源读写、清洗（LangChain Tool）、分析、Plotly、`registry.py` 汇总 |
| `server/agent` | `house_agent.py`：DashScope 兼容 LLM、`create_agent`、LangGraph 流水线编译 |
| `server/api` | REST 路由、`ws.py` WebSocket 进度推送 |
| `templates` | `report_template.html` Jinja2 报告 |
| `data/raw/{session_id}` | 用户上传原始文件 |
| `data/output/{session_id}` | 清洗中间结果、报告 HTML/XLSX/PDF |
| `frontend` | Web UI（上传、会话、图表、报告预览、对话） |
| `scripts` | `generate_sample_house_data.py` 样本生成 |

## 请求与流水线

1. `POST /api/v1/sessions` 创建会话，得到 `session_id`。
2. `POST /api/v1/sessions/{id}/upload` 上传多个 CSV/XLSX，写入 `data/raw/{id}/`。
3. `POST /api/v1/sessions/{id}/run` 异步执行 LangGraph：`ingest → clean（create_agent+清洗工具）→ analyze → viz → export`。
4. `WS /api/v1/ws/sessions/{id}`（浏览器 WebSocket）订阅 `{stage,pct,msg}` 进度事件。
5. 产物：`GET /api/v1/sessions/{id}/artifacts` 列表；`GET .../artifacts/download?name=report.html` 下载。
6. `GET /api/v1/sessions/{id}/analysis`：结构化分析结果（JSON）。
7. `GET /api/v1/sessions/{id}/figures`：Plotly 生成的 HTML 片段映射（供前端 `dangerouslySetInnerHTML` 或 iframe）。

## 多轮对话

`POST /api/v1/sessions/{id}/chat`：在同一会话内保留近期消息，可选附带当前数据集摘要（字段与行数），使用同一 DashScope 兼容 Chat 模型回复。

## 数据不落库

首版使用内存 `SessionStore` + 磁盘 `data/*`；后续可替换为 Redis/对象存储而不改 API 形状。
