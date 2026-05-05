# HouseInsight Agent

面向**二手房挂牌表**的端到端数据分析：**多文件合并 → 确定性 pandas ETL（复合列规则 + 数值化 + 去重 + IQR）→ 可选 L3 描述弱特征（抽样 LLM）→ 质检回路 → 分析任务规划（LLM 仅见聚合画像）与确定性执行 → Plotly 图表 → 报告导出**，并支持基于**聚合摘要**的多轮对话。清洗默认**不经 LLM 改全表**；仅当 `HOUSEINSIGHT_LEGACY_AGENT_CLEAN=true` 时保留旧版 ReAct 清洗 Agent 作对照。

技术栈：**FastAPI**、**LangChain Agents**、**LangGraph**、**pandas**、**Plotly**、阿里云 **DashScope**（OpenAI 兼容接口）。

---

## 项目综述

| 模块 | 路径 | 职责 |
|------|------|------|
| 流水线编排 | `server/agent/house_agent.py` | LangGraph：`ingest` → `clean`（`run_listing_etl`）→ **`feature_engineering`**（规则标签 + `finalize`）→ `enrich_description`（可选 L3）→ `quality_gate` → `analyze` → `viz` → `export` |
| 确定性 ETL | `server/pipeline/listing_etl.py` | 一键：`expand_composite` → `promote` → 文本数值解析 → 数值化 → 单价推算 → 去重 → IQR → `floor_band`（**不含**规则文本标签） |
| 规则特征节点 | `server/pipeline/rule_feature_engineering.py` | 调用 `text_label_features` 后接 `finalize_listing_dataframe`，与会话主表 `df_clean` 对齐 |
| 文本规则标签 | `server/tools/text_label_features.py` | **全量、无 LLM**：从 `description_raw` / `location_raw` / `listing_title` / `house_info_raw` 拼文本后写 `tag_*`、`layout_normalized`、`build_year` 文本回填等 |
| L3 描述弱特征 | `server/tools/description_enrich.py` | 可选：`HOUSEINSIGHT_DESCRIPTION_ENRICH=true` 且含 `description_raw` 时**抽样** LLM，写入 `description_hint_subway` / `description_hint_school`（与上列规则标签分工：规则标签供筛选与展示；LLM 分数为补充弱信号） |
| 标准字段 | `server/core/house_schema.py` | 标准列名、中文别名、拆分用过渡列（如 `layout_str`、`area_m2_str`、`decoration_str`） |
| 合并与列名归一 | `server/tools/io.py` | 多 CSV/XLS/XLSX 纵向合并、`COLUMN_ALIASES` → 标准键；为每行写入 **`ingest_file`**（源表文件名不含后缀），与 `listing_id` 一起参与去重，避免多区文件合并后编号撞车被误删 |
| 复合列解析 | `server/tools/composite_field_parse.py` | 无 LLM 时从 `house_info_raw`（`|`）、`follow_info_raw`（`/`）确定性抽取户型、面积、装修、楼层、年代、关注人数与时间文本等 |
| 领域清洗工具 | `server/tools/cleaning_housing.py` | `get_dataset_profile`、按 `\|`/`/` 拆分、楼层档、装修规范化、关注人数、`apply_column_rename` |
| 通用数值清洗 | `server/tools/cleaning.py` | 工具闭包（旧版 Agent 用）；`apply_default_cleaning_pipeline` = `run_listing_etl` + `apply_rule_text_features`（与线上一致） |
| 文本数值解析 | `server/tools/listing_numeric_parse.py` | 「153万」「17190元/平米」「89平米」等解析；**`finalize_listing_dataframe`** 在规则特征节点内对会话表执行，保证对话与下游列类型一致 |
| 数据质检 | `server/tools/data_quality.py` | **阻塞**：表空、或原始行数≥50 且清洗后保留率&lt;50%；**警告**（不阻塞、不触发重洗）：行数偏少、单价/地理弱、`tag_near_subway` 占比过低等；可选 LLM「教练」仅针对阻塞项 |
| 分析规划与执行 | `server/tools/analysis_plan.py` | 任务白名单枚举、LLM 输出 JSON 任务列表、`execute_task` 确定性聚合 |
| 分析师叙述 | `server/tools/analysis_narrative.py` | 基于结构化结果追加「分析师观点」（需 API Key） |
| 可视化 | `server/tools/viz.py` | 按 `task_results` 动态出图（横向城区均价条、饼图等）；首个图嵌入 Plotly CDN |
| 报告导出 | `server/tools/export.py`、`templates/` | Jinja2 HTML、Excel（含 `plan_tasks` 等） |
| REST / WS | `server/api/routes.py`、`server/api/ws.py` | 会话、上传、跑流水线、状态、分析结果、图表、产物下载；进度 WebSocket |

**设计原则**：**ETL 与统计为确定性代码**；LLM 用于**分析任务规划**、**对话（仅聚合上下文）**及**可选的描述抽样标注**；统计与出图均为**白名单原语**。任意步骤**禁止**执行用户或模型注入的任意代码。

---

## 数据流水线（简图）

```
上传表格 → ingest（合并 + 别名归一）
       → clean（run_listing_etl：复合列 + 数值化 + 去重 + IQR + floor_band；不经 LLM 改表）
       → feature_engineering（规则文本标签 + finalize）
       → enrich_description（可选：抽样 LLM 写描述弱特征列；可关）
       → quality_gate（阻塞项未通过且未达最大轮次 → 重置 raw 再 clean；其余为警告写入 `warnings_zh`）
       → analyze（LLM 仅根据裁剪后的列画像规划任务；run_planned_analysis 确定性执行）
       → viz → export（…）
```

---

## 环境变量

复制 `.env.example` 为 `.env` 后配置：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key；为空时清洗走默认规则、分析规划走 fallback、无「分析师观点」LLM |
| `HOUSEINSIGHT_LLM_MODEL` | 默认对话、意图解析、描述增强、质检教练等模型，如 `qwen3.6-flash` |
| `HOUSEINSIGHT_PLAN_MODEL` | 可选；分析任务规划专用模型，为空则与上面一致 |
| `HOUSEINSIGHT_MAX_CLEAN_ATTEMPTS` | 清洗最大轮次（含首轮），默认 `3` |
| `HOUSEINSIGHT_LEGACY_AGENT_CLEAN` | `true` 时启用旧版「清洗 ReAct Agent 改表」；**默认 false（推荐）** |
| `HOUSEINSIGHT_DESCRIPTION_ENRICH` | `true` 时对 `description_raw` 抽样 LLM 写入 `description_hint_*`；默认 false |
| `HOUSEINSIGHT_DESCRIPTION_SAMPLE_N` | L3 抽样行数上限，默认 `200` |
| `HOUSEINSIGHT_PLAN_PROFILE_MAX_COLS` | 分析规划请求中列画像条数上限，默认 `48` |
| `QUALITY_MIN_ROWS` | 质检：**警告**用（不阻塞）；清洗后行数低于该期望时提示 |
| `QUALITY_MIN_RETENTION_RATIO` | 质检：**警告**用；保留率低于该比例且仍≥50% 时提示（低于 50% 才阻塞重洗） |
| `QUALITY_MIN_UNIT_PRICE_COVERAGE` | 质检：**警告**用；有效单价占比参考 |
| `QUALITY_MIN_GEO_COVERAGE` | 质检：**警告**用；地理字段有效占比参考 |

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
- `GET /sessions/{id}/run_result` — 首屏聚合：`analysis`、`analysis_summary_markdown`、**`analysis_summary_plain`**（约 300～500 字中文纯文本）、`figures_keys`、`figures_payload_chars`、`figures_too_large_for_inline`、`artifacts`、`quality_report` / `quality_brief`、`progress_events` 尾部、`options` 回显  
- `GET /sessions/{id}/analysis` — `analysis`、`analysis_plan`、`quality_report`、`cleaning_trace`、`clean_attempt_count` 等  
- `GET /sessions/{id}/figures` — Plotly HTML 片段字典  
- `GET /sessions/{id}/figures/embed?name=...` — 返回**完整 HTML 文档**（供前端 iframe `src` 加载，解决内联脚本在 SPA 中不执行的问题）  
- `GET /sessions/{id}/artifacts` / `.../download` — 报告与导出文件  
- `POST /sessions/{id}/chat` — 多轮对话（需配置 API Key）；响应体含 `reply` 与 `sources`。若模型判定用户需要**具体房源行**，会先经结构化意图解析再对 `df_clean` 做**白名单列、白名单条件**的 pandas 筛选，将 JSON 样本附在上下文中回答（见 `server/tools/chat_listing_query.py`）  
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
