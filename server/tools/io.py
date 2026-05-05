from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from server.core.house_schema import STANDARD_COLUMN_KEYS, normalize_header

_TEXT_ENCODINGS_CSV = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936")

_SMALL_XLSX_BYTES = 500


def _read_csv_with_encoding_fallback(p: Path) -> pd.DataFrame:
    """先按字节严格解码（避免 UTF-8+replace 误吞 GBK），最后再宽松尝试。"""
    raw = p.read_bytes()
    last_err: Exception | None = None
    for enc in _TEXT_ENCODINGS_CSV:
        try:
            text = raw.decode(enc)
            return pd.read_csv(StringIO(text))
        except Exception as e:
            last_err = e
            continue
    try:
        text = raw.decode(_TEXT_ENCODINGS_CSV[-1], errors="replace")
        return pd.read_csv(StringIO(text))
    except Exception:
        pass
    if last_err is not None:
        raise last_err
    raise ValueError(f"无法用常见编码读取 CSV: {p}")


def _read_xlsx_with_hints(p: Path) -> pd.DataFrame:
    try:
        sz = p.stat().st_size
    except OSError:
        sz = -1
    if sz >= 0 and sz < _SMALL_XLSX_BYTES:
        try:
            return pd.read_excel(p, engine="openpyxl")
        except Exception as e:
            raise ValueError(
                f"{p.name}: 文件过小({sz}B)，可能不是真实 xlsx（例如误改后缀或系统附属文件）。原始错误: {e}"
            ) from e
    return pd.read_excel(p, engine="openpyxl")


def _read_xls(p: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(p, engine="xlrd")
    except ImportError as e:
        raise ValueError(f"{p.name}: 读取 .xls 需要安装 xlrd（项目依赖已声明）。{e}") from e


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return _read_csv_with_encoding_fallback(path)
    if suf == ".xlsx":
        return _read_xlsx_with_hints(path)
    if suf == ".xls":
        return _read_xls(path)
    raise ValueError(f"不支持的文件类型: {path}")


def iter_tabular_paths(raw_dir: Path) -> list[Path]:
    """枚举目录内 csv/xlsx/xls（多后缀大小写 glob，同一文件 resolve 去重）。"""
    patterns = ("*.csv", "*.CSV", "*.xlsx", "*.XLSX", "*.xls", "*.XLS")
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in raw_dir.glob(pat):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    out.sort(key=lambda x: x.name.lower())
    return out


def _total_failure_hint(paths: list[Path]) -> str:
    hints: list[str] = []
    if paths and all(p.name.startswith("._") for p in paths):
        hints.append(
            "当前目录下文件名均以 ._ 开头时，多为随盘附带的占位文件而非表格本体；"
            "请在来源处查找同名但不带 ._ 前缀的 xlsx/csv/xls 再上传。"
        )
    try:
        if paths and all(p.suffix.lower() == ".xlsx" for p in paths):
            small = [p for p in paths if p.stat().st_size < _SMALL_XLSX_BYTES]
            if len(small) == len(paths):
                hints.append("所有 xlsx 体积均过小，可能并非真实 Excel 工作簿。")
    except OSError:
        pass
    return " ".join(hints)


def load_raw_directory(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    逐个尝试读取表格并纵向合并。
    无法解析的文件记入 warnings，不因文件名拒绝任何上传文件。
    """
    paths = iter_tabular_paths(raw_dir)
    if not paths:
        raise FileNotFoundError(f"目录中未找到 csv/xlsx/xls: {raw_dir}")

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for p in paths:
        try:
            tbl = read_table(p)
            # 多文件合并时各区房源编号常在文件内唯一、跨文件重复；下游按 ingest_file+listing_id 去重
            tbl["ingest_file"] = p.stem
            frames.append(tbl)
        except Exception as e:
            warnings.append(f"{p.name}: {type(e).__name__}: {e}")

    if not frames:
        detail = "\n".join(warnings) if warnings else "未知错误"
        hint = _total_failure_hint(paths)
        tail = f"\n{hint}" if hint else ""
        raise ValueError(
            f"未能读取任何表格（共 {len(paths)} 个文件）。若扩展名为 .xlsx 但内容不是有效 Excel，也会失败。"
            f"{tail}\n详情:\n{detail}"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged.columns = [normalize_header(c) for c in merged.columns]
    return merged, warnings


def canonicalize_known_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """将别名列进一步映射为标准英文键（若列名已是中文别名映射结果）。"""
    out = df.copy()
    rename: dict[str, str] = {}
    for col in list(out.columns):
        key = normalize_header(str(col))
        if key != col and key in STANDARD_COLUMN_KEYS:
            rename[col] = key
    if rename:
        out = out.rename(columns=rename)
    return out
