from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.core.paths import ProjectPaths


def render_report_html(
    paths: ProjectPaths,
    session_id: str,
    analysis: dict[str, Any],
    figures: dict[str, str],
    cleaning_notes: str,
    narrative: str | None = None,
) -> Path:
    out_dir = paths.output_dir(session_id)
    env = Environment(
        loader=FileSystemLoader(str(paths.templates)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("report_template.html")
    html = tpl.render(
        session_id=session_id,
        analysis=analysis,
        figures=figures,
        cleaning_notes=cleaning_notes,
        narrative=narrative or "",
    )
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def write_excel_report(output_dir: Path, df: pd.DataFrame, analysis: dict[str, Any]) -> Path:
    xlsx_path = output_dir / "report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.head(5000).to_excel(writer, sheet_name="clean_sample", index=False)
        summary = pd.DataFrame([{"metric": "rows", "value": len(df)}])
        summary.to_excel(writer, sheet_name="summary", index=False)
        if "district_summary" in analysis and analysis["district_summary"]:
            pd.DataFrame(analysis["district_summary"]).to_excel(writer, sheet_name="district", index=False)
        tr = analysis.get("task_results")
        if isinstance(tr, dict) and tr:
            rows = []
            for key, spec in tr.items():
                if isinstance(spec, dict):
                    rows.append(
                        {
                            "task_key": key,
                            "ok": spec.get("ok"),
                            "title": spec.get("title"),
                            "chart_kind": spec.get("chart_kind"),
                            "reason": spec.get("reason"),
                        }
                    )
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name="plan_tasks", index=False)
    return xlsx_path


def maybe_render_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    try:
        from weasyprint import HTML
    except Exception:
        return None
    try:
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception:
        return None
