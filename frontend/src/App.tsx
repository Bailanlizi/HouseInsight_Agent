import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiUrl, wsUrl } from "./api";
import { PlotlyFigure } from "./PlotlyFigure";

type ProgressEvt = {
  stage?: string;
  pct?: number;
  msg?: string;
  ts?: string;
  phase?: string;
  step_id?: string;
  event?: string;
};

type RunResult = {
  stage?: string;
  progress_pct?: number;
  last_message?: string;
  analysis?: unknown;
  analysis_summary_markdown?: string;
  analysis_summary_plain?: string;
  figures_keys?: string[];
  figures_too_large_for_inline?: boolean;
  artifacts?: Record<string, string>;
  progress_events?: ProgressEvt[];
};

const api = (path: string, init?: RequestInit) => fetch(apiUrl(path), init);

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [log, setLog] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [figures, setFigures] = useState<Record<string, string>>({});
  const [analysisSummaryPlain, setAnalysisSummaryPlain] = useState<string>("");
  const [artifactNames, setArtifactNames] = useState<string[]>([]);
  const [progressEvents, setProgressEvents] = useState<ProgressEvt[]>([]);
  const [returnCleanedFile, setReturnCleanedFile] = useState(false);
  const [chatIn, setChatIn] = useState("");
  const [chatOut, setChatOut] = useState<string>("");
  const [chatSources, setChatSources] = useState<{ label: string; detail: string }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const appendLog = useCallback((line: string) => {
    setLog((s) => (s ? `${s}\n${line}` : line));
  }, []);

  const refreshRunResult = useCallback(
    async (sid: string) => {
      const rr = await api(`/api/v1/sessions/${sid}/run_result`);
      if (!rr.ok) {
        appendLog(`run_result HTTP ${rr.status}`);
        return;
      }
      const j = (await rr.json()) as RunResult;
      setAnalysisSummaryPlain(j.analysis_summary_plain ?? "");
      setArtifactNames(Object.keys(j.artifacts ?? {}));
      if (Array.isArray(j.progress_events) && j.progress_events.length) {
        setProgressEvents(j.progress_events);
      }
      setStatus(`${j.stage ?? "?"} (${j.progress_pct ?? 0}%) — ${j.last_message ?? ""}`);

      if (j.figures_too_large_for_inline) {
        appendLog(
          `图表 HTML 合计约 ${(j as { figures_payload_chars?: number }).figures_payload_chars ?? "?"} 字符，前端跳过内联加载；可单独请求 GET .../figures。`,
        );
        setFigures({});
      } else {
        const ff = await api(`/api/v1/sessions/${sid}/figures`);
        if (ff.ok) {
          const fj = (await ff.json()) as { figures?: Record<string, string> };
          setFigures(fj.figures ?? {});
        }
      }
    },
    [appendLog],
  );

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
          const data = JSON.parse(ev.data as string) as ProgressEvt;
          if (data.stage === "ping") return;
          appendLog(`[${data.stage ?? "?"} ${data.pct ?? "-"}%] ${data.msg ?? ""}`);
          setProgressEvents((prev) => {
            const next = [...prev, data];
            return next.length > 200 ? next.slice(-200) : next;
          });
          if (data.event === "run_complete" || data.stage === "done") void refreshRunResult(sid);
        } catch {
          appendLog(String(ev.data));
        }
      };
      ws.onerror = () => appendLog("WebSocket error（请确认后端已启动在 8000 端口，且未被防火墙拦截）");
    },
    [appendLog, refreshRunResult],
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
      setProgressEvents([]);
      setArtifactNames([]);
      setAnalysisSummaryPlain("");
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
      const r = await api(`/api/v1/sessions/${sessionId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          return_cleaned_file: returnCleanedFile,
          skip_full_report_export: true,
        }),
      });
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
      setChatSources([]);
      return;
    }
    const j = (await r.json()) as { reply?: string; sources?: { label: string; detail: string }[] };
    setChatOut(j.reply ?? "");
    setChatSources(j.sources ?? []);
    setChatIn("");
  };

  const figEntries = useMemo(() => Object.entries(figures), [figures]);

  const reportUrl =
    sessionId && artifactNames.includes("report.html")
      ? apiUrl(`/api/v1/sessions/${sessionId}/artifacts/download?name=report.html`)
      : "";

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
          <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={returnCleanedFile}
              onChange={(e) => setReturnCleanedFile(e.target.checked)}
            />
            运行结束后写出并登记 <code>cleaned.csv</code>（可下载）
          </label>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button className="primary" type="button" onClick={() => void runPipeline()} disabled={!sessionId}>
            运行分析流水线
          </button>
          <button type="button" onClick={() => sessionId && void refreshRunResult(sessionId)} disabled={!sessionId}>
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
          ：拉取 <code>run_result</code> 聚合接口与图表；适用于 WebSocket 漏消息或跑完后对齐界面。默认不生成 HTML 报告（仅 Excel
          等轻量产物），有 <code>report.html</code> 时才会显示「打开 HTML 报告」链接。
        </p>
        <p className="muted">{status}</p>
        {artifactNames.length > 0 ? (
          <div className="row" style={{ marginTop: 8 }}>
            <span className="muted">下载产物：</span>
            {artifactNames.map((name) => (
              <a
                key={name}
                href={apiUrl(`/api/v1/sessions/${sessionId}/artifacts/download?name=${encodeURIComponent(name)}`)}
              >
                {name}
              </a>
            ))}
          </div>
        ) : null}
      </div>

      <div className="card">
        <h3>进度</h3>
        <details>
          <summary>
            展开时间线（{progressEvents.length} 条）— WebSocket 实时追加；完成后与 <code>run_result</code> 对齐
          </summary>
          <ol className="progress-list">
            {progressEvents.map((ev, i) => (
              <li key={`${ev.ts ?? ""}-${i}`}>
                <span className="muted">{ev.ts ?? ""}</span>{" "}
                <strong>{ev.stage ?? "?"}</strong> {ev.pct ?? "-"}% — {ev.msg ?? ""}
                {ev.phase ? (
                  <span className="muted">
                    {" "}
                    <code>{ev.phase}</code>
                  </span>
                ) : null}
              </li>
            ))}
          </ol>
        </details>
        <p className="muted" style={{ marginTop: 8 }}>
          原始日志
        </p>
        <pre className="log">{log || "（暂无）"}</pre>
      </div>

      <div className="card">
        <h3>交互图表（Plotly HTML）</h3>
        {figEntries.length === 0 ? <p className="muted">运行完成后会自动刷新；也可手动点「刷新状态与图表」。</p> : null}
        {figEntries.map(([name]) => (
          <div key={name} className="figure">
            <div className="muted">{name}</div>
            <PlotlyFigure sessionId={sessionId} figureName={name} title={name} />
          </div>
        ))}
      </div>

      <div className="card">
        <h3>分析总结</h3>
        <p className="muted" style={{ marginBottom: 8 }}>
          约 300～500 字纯文本结论（数据概览、价格、供应与关键发现）；跑完流水线后由 <code>run_result</code> 填充。
        </p>
        {analysisSummaryPlain ? (
          <pre className="log log-tall log-plain">{analysisSummaryPlain}</pre>
        ) : (
          <p className="muted">（暂无）</p>
        )}
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
        {chatSources.length > 0 ? (
          <div className="sources-muted">
            <div>数据来源（结构化）</div>
            <ul>
              {chatSources.map((s) => (
                <li key={s.label}>
                  <strong>{s.label}</strong>：{s.detail}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}
