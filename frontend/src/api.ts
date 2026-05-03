/**
 * 开发环境默认直连后端，避免 Vite 代理下偶发的 JSON 非 API 响应 / WebSocket 升级失败。
 * 生产可通过 VITE_API_BASE 覆盖（例如与前端同域则留空）。
 */
export function getApiBase(): string {
  const v = import.meta.env.VITE_API_BASE?.trim();
  if (v) return v.replace(/\/$/, "");
  if (import.meta.env.DEV) return "http://127.0.0.1:8000";
  return "";
}

export function apiUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBase()}${p}`;
}

export function wsUrl(path: string): string {
  const base = getApiBase();
  if (base.startsWith("https://")) {
    return `wss://${base.slice("https://".length)}${path}`;
  }
  if (base.startsWith("http://")) {
    return `ws://${base.slice("http://".length)}${path}`;
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
}
