import pandas as pd

from server.tools.cleaning_housing import (
    coerce_followers_column,
    derive_floor_band_column,
    normalize_decoration_column,
    split_delimited_column,
    split_slash_part,
)


def test_split_pipe_into_standard_columns() -> None:
    df = pd.DataFrame({"blob": ["朝阳区|2室1厅|精装", "海淀区|3室|毛坯"]})
    out = split_delimited_column(df, "blob", "|", ["district", "layout", "decoration"])
    assert list(out["district"]) == ["朝阳区", "海淀区"]
    assert "精装" in str(out.loc[0, "decoration"])


def test_split_slash_followers_segment() -> None:
    df = pd.DataFrame({"关注信息": ["挂牌30天/关注1.2万/浏览99", "x/y/z"]})
    out = split_slash_part(df, "关注信息", 1, "followers")
    assert pd.notna(out.loc[0, "followers"])
    assert "1.2万" in str(out.loc[0, "followers"])


def test_coerce_followers_wan() -> None:
    df = pd.DataFrame({"raw": ["1.2万", "3千", "100"]})
    out = coerce_followers_column(df, "raw", "followers")
    assert abs(float(out.loc[0, "followers"]) - 12000.0) < 1
    assert abs(float(out.loc[1, "followers"]) - 3000.0) < 1


def test_floor_band_from_slash_ratio() -> None:
    df = pd.DataFrame({"floor": ["5/30", "高层", "2/10"]})
    out = derive_floor_band_column(df, "floor")
    assert out.loc[0, "floor_band"] == "低层"
    assert out.loc[1, "floor_band"] == "高层"


def test_normalize_decoration_keywords() -> None:
    df = pd.DataFrame({"txt": ["精装修 南北通透", "清水房"]})
    out = normalize_decoration_column(df, "txt", "decoration")
    assert out.loc[0, "decoration"] == "精装"
    assert out.loc[1, "decoration"] == "毛坯"
