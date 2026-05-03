# HouseInsight Agent

二手房专属数据分析 Agent：多文件上传 → 智能清洗 → 分析 → Plotly 可视化 → HTML/Excel（可选 PDF）报告。

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

编辑 `.env`，填入 `DASHSCOPE_API_KEY`。

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

浏览器访问前端开发地址（一般为 `http://localhost:5173`），API 文档：`http://localhost:8000/docs`。

前端在开发模式下默认直连后端 `http://127.0.0.1:8000`（避免仅依赖 Vite 代理时出现会话接口返回非 JSON、WebSocket 连接失败）。若会话创建日志里出现 `undefined`，多半是后端未启动或请求未打到 FastAPI；请先确认 `start.bat` / uvicorn 已在 8000 端口运行。

## 生成样本数据

```bash
python scripts/generate_sample_house_data.py
```

默认写入 `data/raw/demo/`（便于本地演示）。

## PDF 导出（可选）

```bash
pip install -e ".[pdf]"
```

Windows 下 WeasyPrint 依赖较重，可不装；项目默认提供 HTML + Excel。
