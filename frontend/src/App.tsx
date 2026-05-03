import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiUrl, wsUrl } from "./api";

type WsMsg = { stage?: string; pct?: number; msg?: string };

const api = (path: string, init?: RequestInit) => fetch(apiUrl(path), init);

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [log, setLog] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [figures, setFigures] = useState<Record<string, string>>({});
  const [analysisText, setAnalysisText] = useState<string>("");
  const [chatIn, setChatIn] = useState("");
  const [chatOut, setChatOut] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);

  const appendLog = useCallback((line: string) => {
    setLog((s) => (s ? `${s}\n${line}` : line));
  }, []);

  const refreshArtifacts = useCallback(async (sid: string) => {
    const [fa, ff] = await Promise.all([
      api(`/api/v1/sessions/${sid}/analysis`),
      api(`/api/v1/sessions/${sid}/figures`),
    ]);
    if (fa.ok) {
      const j = (await fa.json()) as { analysis?: unknown };
      setAnalysisText(JSON.stringify(j.analysis ?? {}, null, 2));
    }
    if (ff.ok) {
      const j = (await ff.json()) as { figures?: Record<string, string> };
      setFigures(j.figures ?? {});
    }
    const st = await api(`/api/v1/sessions/${sid}/status`);
    if (st.ok) {
      const j = (await st.json()) as { stage?: string; progress_pct?: number; last_message?: string };
      setStatus(`${j.stage ?? "?"} (${j.progress_pct ?? 0}%) — ${j.last_message ?? ""}`);
    }
  }, []);

  const connectWs = useCallback(
    (sid: string) => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      const url = wsUrl(`/api/v1/ws/sessions/${sid}`);
      appendLog(`WebSocket 连接: ${url}`);
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => appendLog("WebSocket 已连接");
      ws.onclose = (ev) => appendLog(`WebSocket 关闭 code=${ev.code}`);
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data as string) as WsMsg;
          if (data.stage === "ping") return;
          appendLog(`[${data.stage ?? "?"} ${data.pct ?? "-"}%] ${data.msg ?? ""}`);
          if (data.stage === "done") void refreshArtifacts(sid);
        } catch {
          appendLog(String(ev.data));
        }
      };
      ws.onerror = () => appendLog("WebSocket error（请确认后端已启动在 8000 端口，且未被防火墙拦截）");
    },
    [appendLog, refreshArtifacts]
  );

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const createSession = async () => {
    try {
      const r = await api("/api/v1/sessions", { method: "POST" });
      const text = await r.text();
      let j: { session_id?: string } = {};
      try {
        j = JSON.parse(text) as { session_id?: string };
      } catch {
        appendLog(`创建会话失败：响应不是 JSON（HTTP ${r.status}）。 body 前 200 字：${text.slice(0, 200)}`);
        return;
      }
      if (!r.ok) {
        appendLog(`创建会话失败 HTTP ${r.status}: ${text.slice(0, 400)}`);
        return;
      }
      const sid = j.session_id;
      if (!sid) {
        appendLog(`创建会话失败：返回 JSON 缺少 session_id。body=${text.slice(0, 400)}`);
        return;
      }
      setSessionId(sid);
      appendLog(`会话已创建: ${sid}`);
      connectWs(sid);
    } catch (e) {
      appendLog(`创建会话异常: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    if (!sessionId) return;
    const files = e.target.files;
    if (!files?.length) return;
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    try {
      const r = await api(`/api/v1/sessions/${sessionId}/upload`, { method: "POST", body: fd });
      const text = await r.text();
      if (!r.ok) {
        appendLog(`上传失败 HTTP ${r.status}: ${text.slice(0, 400)}`);
        return;
      }
      const j = JSON.parse(text) as { saved?: string[] };
      appendLog(`上传完成: ${JSON.stringify(j.saved ?? [])}`);
    } catch (err) {
      appendLog(`上传异常: ${err instanceof Error ? err.message : String(err)}`);
    }
    e.target.value = "";
  };

  const runPipeline = async () => {
    if (!sessionId) return;
    appendLog("启动流水线…");
    try {
      const r = await api(`/api/v1/sessions/${sessionId}/run`, { method: "POST" });
      const t = await r.text();
      if (!r.ok) appendLog(`启动失败 HTTP ${r.status}: ${t.slice(0, 400)}`);
    } catch (e) {
      appendLog(`启动异常: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const sendChat = async () => {
    if (!sessionId || !chatIn.trim()) return;
    const r = await api(`/api/v1/sessions/${sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: chatIn }),
    });
    if (!r.ok) {
      const t = await r.text();
      setChatOut(`错误: ${t}`);
      return;
    }
    const j = (await r.json()) as { reply?: string };
    setChatOut(j.reply ?? "");
    setChatIn("");
  };

  const figEntries = useMemo(() => Object.entries(figures), [figures]);

  const reportUrl = sessionId ? apiUrl(`/api/v1/sessions/${sessionId}/artifacts/download?name=report.html`) : "";

  const uploadDisabled = !sessionId;

  return (
    <div className="layout">
      <h1>HouseInsight 二手房分析</h1>
      <p className="muted">
        开发模式默认请求 <code>http://127.0.0.1:8000</code>（请先启动后端）。也可用环境变量{" "}
        <code>VITE_API_BASE</code> 自定义。
      </p>

      <div className="card">
        <div className="row">
          <button className="primary" type="button" onClick={() => void createSession()}>
            新建会话
          </button>
          <span className="muted">session_id:</span>
          <code>{sessionId || "（未创建）"}</code>
        </div>
      </div>

      <div className="card">
        <h3>多文件上传</h3>
        {uploadDisabled ? (
          <p className="muted">请先点击「新建会话」。会话创建成功后，下面的文件按钮才会启用（禁用时点文件按钮不会有反应）。</p>
        ) : null}
        <input type="file" multiple accept=".csv,.xlsx,.xls" onChange={(e) => void onUpload(e)} disabled={uploadDisabled} />
        <div className="row" style={{ marginTop: 12 }}>
          <button className="primary" type="button" onClick={() => void runPipeline()} disabled={!sessionId}>
            运行分析流水线
          </button>
          <button type="button" onClick={() => sessionId && void refreshArtifacts(sessionId)} disabled={!sessionId}>
            刷新状态与图表
          </button>
          {reportUrl ? (
            <a href={reportUrl} target="_blank" rel="noreferrer">
              打开 HTML 报告
            </a>
          ) : null}
        </div>
        <p className="muted">
          <strong>刷新状态与图表</strong>
          ：不向服务器重新跑流水线，只是再拉取当前会话的「阶段/进度文案」「分析 JSON」「Plotly 图表 HTML」。适用于 WebSocket
          漏了进度、你晚开了页面、或跑完后想手动对齐界面与服务端状态。
        </p>
        <p className="muted">{status}</p>
      </div>

      <div className="card">
        <h3>进度（WebSocket）</h3>
        <pre className="log">{log || "（暂无）"}</pre>
      </div>

      <div className="card">
        <h3>交互图表（Plotly HTML）</h3>
        {figEntries.length === 0 ? <p className="muted">运行完成后会自动刷新；也可手动点「刷新状态与图表」。</p> : null}
        {figEntries.map(([name, html]) => (
          <div key={name} className="figure">
            <div className="muted">{name}</div>
            <div dangerouslySetInnerHTML={{ __html: html }} />
          </div>
        ))}
      </div>

      <div className="card">
        <h3>分析摘要（JSON）</h3>
        <pre className="log">{analysisText || "（暂无）"}</pre>
      </div>

      <div className="card">
        <h3>多轮对话（需要配置 DASHSCOPE_API_KEY）</h3>
        <textarea rows={3} value={chatIn} onChange={(e) => setChatIn(e.target.value)} placeholder="问一问当前数据集…" />
        <div className="row" style={{ marginTop: 8 }}>
          <button type="button" className="primary" onClick={() => void sendChat()} disabled={!sessionId}>
            发送
          </button>
        </div>
        <pre className="log" style={{ marginTop: 12 }}>
          {chatOut || "（暂无回复）"}
        </pre>
      </div>
    </div>
  );
}
