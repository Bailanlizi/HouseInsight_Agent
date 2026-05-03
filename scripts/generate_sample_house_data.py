"""
生成贴近真实字段的二手房样本 CSV，默认写入 data/raw/demo/（需在 ARCHITECTURE 中约定的演示路径）。
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

DISTRICTS = ["海淀区", "朝阳区", "丰台区", "石景山区", "通州区"]
COMMUNITIES = ["梧桐苑", "晨光小区", "滨河家园", "望京花园", "莲花小区"]
LAYOUTS = ["2室1厅", "3室2厅", "1室1厅", "3室1厅", "4室2厅"]
ORIENTATIONS = ["南", "南北", "东", "东南", "西南"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    rows = []
    for i in range(120):
        district = random.choice(DISTRICTS)
        area = round(random.uniform(45, 145), 1)
        unit_price = random.randint(35_000, 95_000)
        total_wan = round(unit_price * area / 10_000, 2)  # 总价（万元）
        build_year = random.randint(1998, 2022)
        rows.append(
            {
                "城区": district,
                "小区": random.choice(COMMUNITIES) + str(random.randint(1, 3)),
                "户型": random.choice(LAYOUTS),
                "建筑面积": area,
                "总价": total_wan,
                "单价": unit_price if random.random() > 0.15 else None,
                "楼层": f"{random.randint(1, 18)}/{random.randint(18, 33)}",
                "朝向": random.choice(ORIENTATIONS),
                "建筑年代": build_year,
                "挂牌日期": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "房源编号": f"DEMO-{i:04d}",
            }
        )

    df = pd.DataFrame(rows)
    p1 = out_dir / "demo_batch_a.csv"
    p2 = out_dir / "demo_batch_b.csv"
    df.iloc[:60].to_csv(p1, index=False, encoding="utf-8-sig")
    df.iloc[60:].to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"写入: {p1}")
    print(f"写入: {p2}")


if __name__ == "__main__":
    main()
