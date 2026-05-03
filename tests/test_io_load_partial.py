"""ingest：按文件尝试读取，失败项记入 warnings；编码/xls/glob 回归。"""

from pathlib import Path

import pytest

from server.tools.io import iter_tabular_paths, load_raw_directory


def test_load_raw_directory_skips_unreadable_keeps_good(tmp_path: Path) -> None:
    d = tmp_path
    (d / "good.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (d / "bad.xlsx").write_bytes(b"not-a-valid-xlsx-zip")
    df, warnings = load_raw_directory(d)
    assert len(df) == 1
    assert any("bad.xlsx" in w for w in warnings)


def test_load_raw_directory_all_fail_raises(tmp_path: Path) -> None:
    d = tmp_path
    (d / "only_bad.xlsx").write_bytes(b"nope")
    try:
        load_raw_directory(d)
    except ValueError as e:
        assert "未能读取" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_load_raw_gbk_csv(tmp_path: Path) -> None:
    d = tmp_path
    content = "城区,总价\n海淀,100\n".encode("gbk")
    (d / "房源.csv").write_bytes(content)
    df, warnings = load_raw_directory(d)
    assert not warnings
    assert len(df) == 1
    assert str(df.iloc[0, 0]) == "海淀"


def test_iter_tabular_paths_includes_xls(tmp_path: Path) -> None:
    d = tmp_path
    (d / "data.xls").write_bytes(b"\xd0\xcf\x11\xe0")
    (d / "x.csv").write_text("a\n1\n", encoding="utf-8")
    paths = iter_tabular_paths(d)
    names = {p.name.lower() for p in paths}
    assert "data.xls" in names
    assert "x.csv" in names


def test_load_raw_xls_with_xlwt_fixture(tmp_path: Path) -> None:
    pytest.importorskip("xlwt")
    import xlwt

    d = tmp_path
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")
    ws.write(0, 0, "城区")
    ws.write(0, 1, "总价")
    ws.write(1, 0, "海淀")
    ws.write(1, 1, 100)
    xls_path = d / "房源.xls"
    wb.save(str(xls_path))
    df, warnings = load_raw_directory(d)
    assert not warnings
    assert len(df) == 1
