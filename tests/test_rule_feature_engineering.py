import pandas as pd

from server.pipeline.rule_feature_engineering import apply_rule_text_features


def test_apply_rule_text_features_adds_tags_and_finalize_smoke() -> None:
    df = pd.DataFrame(
        {
            "description_raw": ["\u5730\u94c1\u623f\u8fd1\u5357\u718f\u5927\u9053\u7ad9"],
            "district": ["\u6e29\u6c5f\u533a"],
            "unit_price": [10000.0],
        }
    )
    out, note = apply_rule_text_features(df)
    assert "tag_near_subway" in out.columns
    assert bool(out.loc[0, "tag_near_subway"])
    assert "\u89c4\u5219" in note or "finalize" in note
