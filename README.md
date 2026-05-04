# HouseInsight Agent

面向**二手房挂牌表**的端到端数据分析 Agent：**多文件合并 → 画像与领域清洗（LLM + 白名单工具）→ 数据质检与重试回路 → 分析任务规划与确定性执行 → Plotly 交互图表 → HTML / Excel（可选 PDF）报告**，并支持基于会话数据的**多轮对话**。

技术栈：**FastAPI**、**LangChain Agents**、**LangGraph**、**pandas**、**Plotly**、阿里云 **DashScope**（OpenAI 兼容接口）。

---

## 项目综述

| 模块 | 路径 | 职责 |
|------|------|------|
| 流水线编排 | `server/agent/house_agent.py` | LangGraph：`ingest` → `clean` → `quality_gate` →（未通过则回到 `clean`，最多 N 轮）→ `analyze` → `viz` → `export` |
| 标准字段 | `server/core/house_schema.py` | 标准列名、中文别名、拆分用过渡列（如 `layout_str`、`area_m2_str`、`decoration_str`） |
| 合并与列名归一 | `server/tools/io.py` | 多 CSV/XLS/XLSX 合并、`COLUMN_ALIASES` → 标准键 |
| 领域清洗工具 | `server/tools/cleaning_housing.py` | `get_dataset_profile`、按 `\|`/`/` 拆分、楼层档、装修规范化、关注人数、`apply_column_rename` |
| 通用数值清洗 | `server/tools/cleaning.py` | 数值化、单价推算、安全去重子集、IQR 异常过滤；默认一键流水线 |
| 文本数值解析 | `server/tools/listing_numeric_parse.py` | 「153万」「17190元/平米」「89平米」等解析；过渡列晋升；每轮清洗/分析前 **`finalize_listing_dataframe`** |
| 数据质检 | `server/tools/data_quality.py` | 行保留率、单价/地理覆盖等规则；可选 LLM「教练」生成重试建议 |
| 分析规划与执行 | `server/tools/analysis_plan.py` | 任务白名单枚举、LLM 输出 JSON 任务列表、`execute_task` 确定性聚合 |
| 分析师叙述 | `server/tools/analysis_narrative.py` | 基于结构化结果追加「分析师观点」（需 API Key） |
| 可视化 | `server/tools/viz.py` | 按 `task_results` 动态出图（横向城区均价条、饼图等）；首个图嵌入 Plotly CDN |
| 报告导出 | `server/tools/export.py`、`templates/` | Jinja2 HTML、Excel（含 `plan_tasks` 等） |
| REST / WS | `server/api/routes.py`、`server/api/ws.py` | 会话、上传、跑流水线、状态、分析结果、图表、产物下载；进度 WebSocket |

**设计原则**：智能主要体现在**规划与工具选型**；统计与出图均为**白名单原语**，便于测试与审计。任意步骤**禁止**执行用户或模型注入的任意代码。

---

## 数据流水线（简图）

```
上传表格 → ingest（合并 + 别名归一）
       → clean（画像 → LLM 工具链；异常则默认规则 + finalize）
       → quality_gate（规则质检；未通过且未达最大轮次 → 重置 raw 再 clean）
       → analyze（任务规划 + run_planned_analysis + legacy 指标 + 可选分析师叙述）
       → viz（按 task_results 出图）
       → export（默认可写 report.xlsx；`skip_full_report_export=true` 时跳过 HTML/PDF；可选写 cleaned.csv）
```

---

## 环境变量

复制 `.env.example` 为 `.env` 后配置：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key；为空时清洗走默认规则、分析规划走 fallback、无「分析师观点」LLM |
| `HOUSEINSIGHT_LLM_MODEL` | 默认对话与清洗等模型，如 `qwen-plus` |
| `HOUSEINSIGHT_PLAN_MODEL` | 可选；分析任务规划专用模型，为空则与上面一致 |
| `HOUSEINSIGHT_MAX_CLEAN_ATTEMPTS` | 清洗最大轮次（含首轮），默认 `3` |
| `QUALITY_MIN_ROWS` | 质检：清洗后最少行数期望，默认 `25` |
| `QUALITY_MIN_RETENTION_RATIO` | 质检：相对原始表最小行保留比例（大行数表），默认 `0.02` |
| `QUALITY_MIN_UNIT_PRICE_COVERAGE` | 质检：有效单价占比下限等，默认 `0.15` |
| `QUALITY_MIN_GEO_COVERAGE` | 质检：`district`/`community` 有效文本占比下限，默认 `0.15` |

（布尔与数值型配置见 `server/core/config.py`，支持 `.env` 覆盖。）

---

## 快速开始

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

2. 复制环境变量：

```bash
copy .env.example .env
```

编辑 `.env`，填入 `DASHSCOPE_API_KEY`（可选，但开启完整「智能清洗 + 规划 + 叙述」能力建议填写）。

3. 启动后端：

```bash
start.bat
```

或：`uvicorn server.main:app --reload --host 0.0.0.0 --port 8000`

4. 启动前端（可选）：

```bash
cd frontend
npm install
npm run dev
```

- 前端开发地址一般为 `http://localhost:5173`
- OpenAPI：`http://localhost:8000/docs`

前端开发模式下默认直连后端 `http://127.0.0.1:8000`。若会话接口异常或 WebSocket 失败，请先确认后端已在 8000 端口运行。

---

## API 摘要

- `POST /sessions` — 创建会话  
- `POST /sessions/{id}/upload` — 上传原始表格到会话 `raw` 目录；进度事件写入会话时间线并经 WS 推送（与流水线 `_emit` 字段一致）  
- `POST /sessions/{id}/run` — 异步执行整条流水线；**JSON 请求体（可选）**：`return_cleaned_file`（默认 `false`）是否在 `output` 写出 `cleaned.csv` 并登记到 `artifacts`；`skip_full_report_export`（默认 `true`）为 `true` 时不生成 `report.html` / `report.pdf`（仍会写 `report.xlsx`）  
- `GET /sessions/{id}/status` — 阶段与进度  
- `GET /sessions/{id}/run_result` — 首屏聚合：`analysis`、`analysis_summary_markdown`、`figures_keys`、`figures_payload_chars`、`figures_too_large_for_inline`、`artifacts`、`quality_report` / `quality_brief`、`progress_events` 尾部、`options` 回显  
- `GET /sessions/{id}/analysis` — `analysis`、`analysis_plan`、`quality_report`、`cleaning_trace`、`clean_attempt_count` 等  
- `GET /sessions/{id}/figures` — Plotly HTML 片段字典  
- `GET /sessions/{id}/artifacts` / `.../download` — 报告与导出文件  
- `POST /sessions/{id}/chat` — 多轮对话（需配置 API Key）；响应体含 `reply` 与 `sources`（结构化数据来源说明；正文末尾另有中文脚注）  
- WebSocket：订阅会话进度；负载除 `stage` / `pct` / `msg` 外可含 `ts`、`phase`、`step_id`、`event`（流水线完成时为 `event=run_complete`）

---

## 生成样本数据

```bash
python scripts/generate_sample_house_data.py
```

默认写入 `data/raw/demo/`，便于本地演示。

---

## 测试

```bash
pytest
```

覆盖 IO 合并、清洗与去重、`listing_numeric_parse`、质检、`analysis_plan`、端到端流水线（无 Key）、健康检查等。

---

## PDF 导出（可选）

```bash
pip install -e ".[pdf]"
```

Windows 下 WeasyPrint 依赖较重；不安装时仍可使用 HTML + Excel。

---

## 版本

当前包版本见 `pyproject.toml`（`houseinsight-agent`）。
