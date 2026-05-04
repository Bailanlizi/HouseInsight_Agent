"""横向条形图：Y 轴应展示全部类目标签（不依赖 Plotly 自动抽稀）。"""

from server.tools.viz import _bar_from_records


def test_horizontal_bar_includes_all_district_labels() -> None:
    records = [{"district": f"测试区{n}", "unit_price": 10_000.0 + n * 50} for n in range(23)]
    html = _bar_from_records(
        records,
        "district",
        "unit_price",
        "均价",
        False,
        horizontal=True,
    )
    for n in range(23):
        assert f"测试区{n}" in html
