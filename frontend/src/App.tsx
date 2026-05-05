import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Database,
  Download,
  FileText,
  Home,
  Loader2,
  MessageSquare,
  RefreshCw,
  Send,
  Sparkles,
  Terminal,
  Upload,
  X,
} from "lucide-react";

import { apiUrl, wsUrl } from "./api";

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
  artifacts?: Record<string, string>;
  progress_events?: ProgressEvt[];
};

type BusyKey = "session" | "upload" | "run" | "refresh" | "chat" | null;
type ChatMsg = { role: "user" | "assistant"; text: string; isError?: boolean };

const api = (path: string, init?: RequestInit) => fetch(apiUrl(path), init);

async function parseHttpError(res: Response): Promise<string> {
  const t = await res.text();
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail
        .map((x: { msg?: string }) => x.msg || JSON.stringify(x))
        .join("；");
    }
  } catch {
    /* ignore */
  }
  return t.slice(0, 400) || `请求失败（HTTP ${res.status}）`;
}

function stageIcon(stage?: string) {
  switch (stage) {
    case "ingest":
    case "upload":
      return Database;
    case "clean":
    case "quality":
      return Sparkles;
    case "analyze":
      return FileText;
    case "export":
      return Download;
    case "done":
      return CheckCircle2;
    case "error":
      return AlertCircle;
    default:
      return Clock;
  }
}

const stageLabelMap: Record<string, string> = {
  upload: "上传",
  ingest: "数据合并",
  clean: "清洗",
  quality: "质检",
  analyze: "分析",
  export: "导出",
  done: "完成",
  error: "错误",
};

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [sessionStage, setSessionStage] = useState<string>("");
  const [progressPct, setProgressPct] = useState<number>(0);
  const [lastMessage, setLastMessage] = useState<string>("");
  const [log, setLog] = useState("");
  const [analysisSummaryPlain, setAnalysisSummaryPlain] = useState("");
  const [artifactNames, setArtifactNames] = useState<string[]>([]);
  const [progressEvents, setProgressEvents] = useState<ProgressEvt[]>([]);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [returnCleanedFile, setReturnCleanedFile] = useState(false);
  const [hasUploaded, setHasUploaded] = useState(false);
  const [chatIn, setChatIn] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMsg[]>([]);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [busy, setBusy] = useState<BusyKey>(null);
  const [wsOnline, setWsOnline] = useState(false);
  const [logExpanded, setLogExpanded] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const autoCreatedRef = useRef(false);

  const busySession = busy === "session";
  const busyUpload = busy === "upload";
  const busyRun = busy === "run";
  const busyRefresh = busy === "refresh";
  const busyChat = busy === "chat";

  const appendLog = useCallback((line: string) => {
    setLog((s) => (s ? `${s}\n${line}` : line));
  }, []);

  const refreshRunResult = useCallback(
    async (sid: string, silent = false): Promise<boolean> => {
      if (!silent) setBusy("refresh");
      try {
        const rr = await api(`/api/v1/sessions/${sid}/run_result`);
        if (!rr.ok) {
          setErrorBanner(await parseHttpError(rr));
          appendLog(`run_result HTTP ${rr.status}`);
          return false;
        }
        setErrorBanner(null);
        const j = (await rr.json()) as RunResult;
        setAnalysisSummaryPlain(j.analysis_summary_plain ?? "");
        setArtifactNames(Object.keys(j.artifacts ?? {}));
        setSessionStage(j.stage ?? "");
        setProgressPct(j.progress_pct ?? 0);
        setLastMessage(j.last_message ?? "");
        if (Array.isArray(j.progress_events) && j.progress_events.length) {
          setProgressEvents(j.progress_events);
        }
        return true;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setErrorBanner(msg);
        appendLog(`刷新失败: ${msg}`);
        return false;
      } finally {
        if (!silent) setBusy(null);
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
      ws.onopen = () => {
        setWsOnline(true);
        appendLog("WebSocket 已连接");
      };
      ws.onclose = (ev) => {
        setWsOnline(false);
        appendLog(`WebSocket 关闭 code=${ev.code}`);
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data as string) as ProgressEvt;
          if (data.stage === "ping") return;
          appendLog(`[${data.stage ?? "?"} ${data.pct ?? "-"}%] ${data.msg ?? ""}`);
          setProgressEvents((prev) => {
            const next = [...prev, data];
            return next.length > 200 ? next.slice(-200) : next;
          });
          if (typeof data.pct === "number") setProgressPct(data.pct);
          if (data.stage) setSessionStage(data.stage);
          if (data.msg) setLastMessage(data.msg);
          if (data.event === "run_complete" || data.stage === "done") void refreshRunResult(sid, true);
        } catch {
          appendLog(String(ev.data));
        }
      };
      ws.onerror = () => {
        setWsOnline(false);
        appendLog("WebSocket 异常（请确认后端已启动且端口可达）");
      };
    },
    [appendLog, refreshRunResult],
  );

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const createSession = async () => {
    setBusy("session");
    setErrorBanner(null);
    try {
      const r = await api("/api/v1/sessions", { method: "POST" });
      const text = await r.text();
      let j: { session_id?: string } = {};
      try {
        j = JSON.parse(text) as { session_id?: string };
      } catch {
        setErrorBanner(`服务器返回非 JSON（HTTP ${r.status}）`);
        return;
      }
      if (!r.ok) {
        setErrorBanner(await parseHttpError(r));
        return;
      }
      const sid = j.session_id;
      if (!sid) {
        setErrorBanner("响应缺少 session_id");
        return;
      }
      setSessionId(sid);
      setSessionStage("");
      setProgressPct(0);
      setLastMessage("");
      setProgressEvents([]);
      setArtifactNames([]);
      setAnalysisSummaryPlain("");
      setHasUploaded(false);
      setPendingFiles([]);
      setChatMessages([]);
      appendLog(`会话已创建: ${sid}`);
      connectWs(sid);
    } catch (e) {
      setErrorBanner(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  useEffect(() => {
    if (autoCreatedRef.current) return;
    if (sessionId) return;
    autoCreatedRef.current = true;
    void createSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const addFiles = (list: FileList | File[]) => {
    const arr = Array.from(list);
    if (!arr.length) return;
    setPendingFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}-${f.size}`));
      const merged = [...prev];
      for (const f of arr) {
        const key = `${f.name}-${f.size}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(f);
        }
      }
      return merged;
    });
  };

  const removeFile = (idx: number) => {
    setPendingFiles((p) => p.filter((_, i) => i !== idx));
  };

  const onFilePick: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = "";
  };

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (!sessionId || busyUpload) return;
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
  };

  const uploadFiles = async (sid: string): Promise<boolean> => {
    if (!pendingFiles.length) return true;
    setBusy("upload");
    const fd = new FormData();
    for (const f of pendingFiles) fd.append("files", f);
    try {
      const r = await api(`/api/v1/sessions/${sid}/upload`, { method: "POST", body: fd });
      if (!r.ok) {
        setErrorBanner(await parseHttpError(r));
        appendLog(`上传失败 HTTP ${r.status}`);
        return false;
      }
      const j = (await r.json()) as { saved?: string[] };
      setHasUploaded(true);
      setPendingFiles([]);
      appendLog(`上传完成: ${JSON.stringify(j.saved ?? [])}`);
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorBanner(msg);
      appendLog(`上传异常: ${msg}`);
      return false;
    } finally {
      setBusy(null);
    }
  };

  const runPipeline = async () => {
    if (!sessionId) return;
    setErrorBanner(null);

    if (pendingFiles.length > 0) {
      const ok = await uploadFiles(sessionId);
      if (!ok) return;
    } else if (!hasUploaded) {
      setErrorBanner("请先选择并上传至少一个文件再运行清洗");
      return;
    }

    setBusy("run");
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
      if (!r.ok) {
        setErrorBanner(await parseHttpError(r));
        appendLog(`启动失败 HTTP ${r.status}`);
        return;
      }
      appendLog("已提交运行（进度见下方时间线或 WebSocket）");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErrorBanner(msg);
      appendLog(`启动异常: ${msg}`);
    } finally {
      setBusy(null);
    }
  };

  const sendChat = async () => {
    if (!sessionId || !chatIn.trim() || busyChat) return;
    const userText = chatIn.trim();
    setChatIn("");
    setChatMessages((m) => [...m, { role: "user", text: userText }]);
    setBusy("chat");
    setErrorBanner(null);
    try {
      const r = await api(`/api/v1/sessions/${sessionId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      if (!r.ok) {
        const errText = await parseHttpError(r);
        setChatMessages((m) => [...m, { role: "assistant", text: errText, isError: true }]);
        return;
      }
      const j = (await r.json()) as { reply?: string };
      setChatMessages((m) => [...m, { role: "assistant", text: j.reply ?? "（无内容）" }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setChatMessages((m) => [...m, { role: "assistant", text: msg, isError: true }]);
    } finally {
      setBusy(null);
    }
  };

  const onChatKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendChat();
    }
  };

  const pipelineDone = useMemo(
    () =>
      sessionStage === "done" ||
      artifactNames.some((n) => n.endsWith(".xlsx") || n.endsWith(".csv")),
    [sessionStage, artifactNames],
  );

  const showProgress =
    Boolean(sessionStage) || progressEvents.length > 0 || busyRun || busyUpload;

  const stageStatus: "idle" | "running" | "done" =
    sessionStage === "done" || pipelineDone
      ? "done"
      : sessionStage && sessionStage !== "idle"
        ? "running"
        : "idle";

  return (
    <div className="min-h-screen flex flex-col">
      {/* Sticky Header */}
      <header className="sticky top-0 z-20 bg-white/70 backdrop-blur-md border-b border-rose-100">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-3 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-orange-500 text-white flex items-center justify-center shadow-sm shadow-rose-500/25 shrink-0">
            <Home size={18} strokeWidth={2} />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-lg sm:text-xl font-bold leading-tight bg-gradient-to-r from-rose-500 to-orange-500 bg-clip-text text-transparent">
              HouseInsight
            </h1>
            <p className="text-xs text-slate-500 leading-tight">二手房智能查询助手</p>
          </div>

          <button
            type="button"
            className="btn-pill-ghost shrink-0"
            disabled={!sessionId || busyRefresh}
            onClick={() => sessionId && void refreshRunResult(sessionId)}
            title="从服务端拉取最新阶段、摘要与产物"
          >
            {busyRefresh ? (
              <Loader2 size={12} className="spin" />
            ) : (
              <RefreshCw size={12} />
            )}
            刷新
          </button>

          <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500 shrink-0">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                wsOnline ? "bg-emerald-500" : "bg-amber-500"
              }`}
              aria-hidden="true"
              title={wsOnline ? "WebSocket 在线" : "WebSocket 未连接"}
            />
            <span>Session:</span>
            <code className="px-1.5 py-0.5 rounded bg-slate-100/80 text-slate-600 max-w-[120px] truncate">
              {sessionId ? `${sessionId.slice(0, 8)}…` : "…"}
            </code>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-5">
        {errorBanner ? (
          <div
            className="flex items-start gap-3 p-3.5 rounded-xl border border-rose-200 bg-rose-50/80 text-rose-800 text-sm"
            role="alert"
          >
            <AlertCircle size={16} strokeWidth={1.75} className="mt-0.5 shrink-0" />
            <div className="flex-1">{errorBanner}</div>
            <button
              type="button"
              onClick={() => setErrorBanner(null)}
              className="text-xs text-rose-600/80 hover:text-rose-800"
            >
              关闭
            </button>
          </div>
        ) : null}

        {/* Upload Card */}
        <section className="card">
          <div className="flex items-center gap-2 mb-4">
            <Upload size={18} strokeWidth={1.75} className="text-rose-500" />
            <h2 className="text-base font-semibold text-slate-800">上传房源数据</h2>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (sessionId && !busyUpload) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => sessionId && !busyUpload && fileInputRef.current?.click()}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed px-6 py-8 sm:py-10 text-center transition-all ${
              !sessionId || busyUpload
                ? "border-slate-200 bg-slate-50/50 cursor-not-allowed"
                : dragOver
                  ? "border-rose-400 bg-rose-50/60"
                  : "border-rose-200 bg-rose-50/30 hover:border-rose-300 hover:bg-rose-50/50"
            }`}
            role="button"
            tabIndex={0}
            aria-disabled={!sessionId || busyUpload}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".csv,.xlsx,.xls"
              onChange={onFilePick}
              disabled={!sessionId || busyUpload}
              className="hidden"
            />
            <div className="flex flex-col items-center gap-2">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-rose-500/10 to-orange-500/10 flex items-center justify-center text-rose-500">
                <Upload size={22} strokeWidth={1.75} />
              </div>
              <div className="text-sm text-slate-700 font-medium">
                {dragOver ? "松开以添加文件" : "拖拽文件到此处"}
              </div>
              <div className="text-xs text-slate-500">支持 CSV / Excel 格式</div>
            </div>
          </div>

          {pendingFiles.length > 0 ? (
            <div className="mt-4 flex flex-wrap gap-2">
              {pendingFiles.map((f, idx) => (
                <span key={`${f.name}-${idx}`} className="chip">
                  <FileText size={12} />
                  <span className="truncate max-w-[160px]">{f.name}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(idx);
                    }}
                    className="text-rose-400 hover:text-rose-600"
                    aria-label={`移除 ${f.name}`}
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          ) : null}

          <label className="mt-4 flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={returnCleanedFile}
              onChange={(e) => setReturnCleanedFile(e.target.checked)}
              className="accent-rose-500"
            />
            <span>
              下载 <code className="px-1.5 py-0.5 rounded bg-slate-100 text-xs">cleaned.csv</code>
            </span>
          </label>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              className="btn-primary"
              disabled={!sessionId || busyRun || busyUpload}
              onClick={() => void runPipeline()}
            >
              {busyRun || busyUpload ? (
                <Loader2 size={16} className="spin" />
              ) : (
                <Sparkles size={16} strokeWidth={2} />
              )}
              {busyUpload ? "上传中…" : busyRun ? "运行中…" : "运行清洗"}
            </button>
            <button
              type="button"
              className="text-link"
              disabled={busySession}
              onClick={() => void createSession()}
            >
              {busySession ? "创建中…" : "新建会话"}
            </button>
          </div>
        </section>

        {/* Progress Card */}
        {showProgress ? (
          <section className="card">
            <div className="flex items-center gap-2 mb-4">
              <Clock size={18} strokeWidth={1.75} className="text-rose-500" />
              <h2 className="text-base font-semibold text-slate-800">处理进度</h2>
              <StageBadge status={stageStatus} />
              <button
                type="button"
                onClick={() => setLogExpanded((v) => !v)}
                className="ml-auto inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700"
              >
                <Terminal size={12} />
                {logExpanded ? "收起日志" : "查看日志"}
                {logExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            </div>

            <ProgressBar pct={progressPct} done={stageStatus === "done"} />
            {lastMessage ? (
              <p className="mt-2 text-xs text-slate-500 truncate">{lastMessage}</p>
            ) : null}

            {/* Simplified timeline */}
            <ol className="mt-4 space-y-1.5 max-h-60 overflow-auto scroll-soft pr-1">
              {progressEvents.slice(-30).map((ev, i) => {
                const Icon = stageIcon(ev.stage);
                const label = stageLabelMap[ev.stage ?? ""] ?? ev.stage ?? "?";
                return (
                  <li
                    key={`${ev.ts ?? ""}-${i}`}
                    className="flex items-center gap-3 text-xs text-slate-600"
                  >
                    <Icon size={14} className="text-rose-400 shrink-0" />
                    <span className="font-medium text-slate-700 w-14 shrink-0">{label}</span>
                    <span className="flex-1 truncate">{ev.msg ?? ""}</span>
                    <span className="text-slate-400 tabular-nums shrink-0">{ev.pct ?? "-"}%</span>
                  </li>
                );
              })}
              {progressEvents.length === 0 ? (
                <li className="text-xs text-slate-400">尚无事件，等待后端推送…</li>
              ) : null}
            </ol>

            {logExpanded ? (
              <div className="mt-4 rounded-xl bg-slate-900 text-slate-100 p-4 font-mono text-xs leading-relaxed max-h-64 overflow-auto scroll-soft whitespace-pre-wrap break-words">
                {log || "（暂无原始日志）"}
              </div>
            ) : null}
          </section>
        ) : null}

        {/* Summary Card */}
        {analysisSummaryPlain ? (
          <section className="card">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={18} strokeWidth={1.75} className="text-rose-500" />
              <h2 className="text-base font-semibold text-slate-800">分析总结</h2>
            </div>
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap break-words max-w-prose">
              {analysisSummaryPlain}
            </p>
          </section>
        ) : null}

        {/* Artifacts Card */}
        {artifactNames.length > 0 ? (
          <section className="card">
            <div className="flex items-center gap-2 mb-3">
              <Download size={18} strokeWidth={1.75} className="text-rose-500" />
              <h2 className="text-base font-semibold text-slate-800">下载产物</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {artifactNames.map((name) => (
                <a
                  key={name}
                  className="inline-flex items-center gap-2 px-4 h-9 rounded-xl bg-gradient-to-r from-rose-500 to-orange-500 text-white text-sm font-medium shadow-sm shadow-rose-500/20 hover:from-rose-600 hover:to-orange-600 transition-colors"
                  href={apiUrl(
                    `/api/v1/sessions/${sessionId}/artifacts/download?name=${encodeURIComponent(name)}`,
                  )}
                  download
                >
                  <Download size={14} />
                  {name}
                </a>
              ))}
            </div>
          </section>
        ) : null}

        {/* Chat Card */}
        <section className="card">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare size={18} strokeWidth={1.75} className="text-rose-500" />
            <h2 className="text-base font-semibold text-slate-800">智能对话</h2>
            {!pipelineDone ? (
              <span className="ml-auto text-xs text-amber-600">请先完成数据清洗</span>
            ) : null}
          </div>

          <div className="rounded-xl border border-rose-100 bg-white/60 max-h-[420px] overflow-auto scroll-soft p-3 sm:p-4">
            {chatMessages.length === 0 ? (
              <div className="flex flex-col items-center gap-2 py-10 text-slate-400">
                <MessageSquare size={28} strokeWidth={1.25} />
                <span className="text-sm text-slate-600 font-medium">开始对话，查询房源信息</span>
                <span className="text-xs text-slate-400">
                  例如：温江区地铁附近的二手房有哪些？
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                {chatMessages.map((m, i) => (
                  <div
                    key={`chat-${i}`}
                    className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words shadow-sm ${
                        m.role === "user"
                          ? "bg-gradient-to-r from-rose-500 to-orange-500 text-white rounded-br-sm"
                          : m.isError
                            ? "bg-rose-50 border border-rose-200 text-rose-700 rounded-bl-sm"
                            : "bg-slate-100/80 text-slate-800 rounded-bl-sm"
                      }`}
                    >
                      {m.text}
                    </div>
                  </div>
                ))}
                {busyChat ? (
                  <div className="flex justify-start">
                    <div className="rounded-2xl rounded-bl-sm bg-slate-100/80 text-slate-500 px-4 py-2.5 text-sm inline-flex items-center gap-2">
                      <Loader2 size={14} className="spin" />
                      正在思考…
                    </div>
                  </div>
                ) : null}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          <div className="mt-3 flex items-center gap-2">
            <input
              type="text"
              value={chatIn}
              onChange={(e) => setChatIn(e.target.value)}
              onKeyDown={onChatKeyDown}
              placeholder={
                pipelineDone ? "问一问当前数据集…" : "完成清洗后即可开始对话"
              }
              disabled={!pipelineDone || busyChat}
              className="flex-1 h-10 rounded-full border border-rose-100 bg-white/80 px-4 text-sm text-slate-700
                focus:outline-none focus:ring-2 focus:ring-rose-200 focus:border-rose-300
                disabled:bg-slate-50/80 disabled:text-slate-400 disabled:cursor-not-allowed"
              aria-label="对话输入"
            />
            <button
              type="button"
              onClick={() => void sendChat()}
              disabled={!pipelineDone || !chatIn.trim() || busyChat}
              className="w-10 h-10 shrink-0 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 text-white flex items-center justify-center shadow-sm shadow-rose-500/25 hover:from-rose-600 hover:to-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              aria-label="发送"
            >
              {busyChat ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
            </button>
          </div>
        </section>
      </main>

      <footer className="max-w-4xl mx-auto w-full px-4 sm:px-6 py-6 text-center text-xs text-slate-400">
        HouseInsight · 二手房分析 · 开发模式默认连接 <code>http://127.0.0.1:8000</code>
      </footer>
    </div>
  );
}

function StageBadge({ status }: { status: "idle" | "running" | "done" }) {
  if (status === "done") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
        <CheckCircle2 size={12} />
        完成
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200">
        <Loader2 size={12} className="spin" />
        运行中
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-slate-50 text-slate-500 border border-slate-200">
      待开始
    </span>
  );
}

function ProgressBar({ pct, done }: { pct: number; done: boolean }) {
  const pctClamped = Math.max(0, Math.min(100, pct || 0));
  return (
    <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
      <div
        className={`h-full transition-[width] duration-500 rounded-full ${
          done
            ? "bg-gradient-to-r from-emerald-400 to-emerald-500"
            : "bg-gradient-to-r from-rose-400 to-orange-500"
        }`}
        style={{ width: `${done ? 100 : pctClamped}%` }}
      />
    </div>
  );
}
