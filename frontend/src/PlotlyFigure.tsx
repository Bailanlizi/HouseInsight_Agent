import { apiUrl } from "./api";

type Props = {
  sessionId: string;
  figureName: string;
  title: string;
};

/**
 * 通过 iframe 直接加载后端返回的完整 HTML 文档（含 Plotly CDN + 内联脚本），
 * 避免在 React 内用 Blob/srcDoc 与 sandbox 导致的脚本不执行问题。
 */
export function PlotlyFigure({ sessionId, figureName, title }: Props) {
  if (!sessionId || !figureName) return null;
  const src = apiUrl(
    `/api/v1/sessions/${sessionId}/figures/embed?name=${encodeURIComponent(figureName)}`,
  );
  return (
    <iframe
      title={title}
      src={src}
      style={{
        width: "100%",
        height: 480,
        border: "1px solid #e8e8ef",
        borderRadius: 8,
        background: "#fff",
      }}
      referrerPolicy="no-referrer-when-downgrade"
    />
  );
}
