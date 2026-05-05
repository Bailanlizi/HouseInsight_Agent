from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
