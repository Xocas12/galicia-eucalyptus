"""Landsat back-cast helpers: cloud mask, scene choice, quantile matching."""

import numpy as np
import pandas as pd

from eucalyptus_impact.real import landsat as LS


def test_qa_mask_flags_cloud_shadow_and_fill():
    qa = np.array(
        [
            0b0,  # clear
            0b1,  # fill
            1 << 4,  # cloud
            3 << 5,  # high-confidence cloud
            3 << 7,  # high-confidence shadow
            1 << 5,  # low-confidence cloud: kept
        ],
        dtype=np.uint16,
    )
    assert LS._qa_bad(qa, "LANDSAT_5").tolist() == [False, True, True, True, True, False]
    cirrus = np.array([3 << 11], dtype=np.uint16)
    assert LS._qa_bad(cirrus, "LANDSAT_8")[0] and not LS._qa_bad(cirrus, "LANDSAT_5")[0]


def test_choose_scenes_prefers_clear_and_avoids_slc_off():
    rows = []
    for i, (date, sat, cc) in enumerate(
        [
            ("2009-07-10", "LANDSAT_5", 20.0),
            ("2009-08-11", "LANDSAT_7", 1.0),  # SLC-off: ranked after any SLC-on scene
            ("2010-07-13", "LANDSAT_5", 5.0),
            ("2012-07-01", "LANDSAT_5", 0.0),  # outside the 2010 epoch
            ("2009-12-20", "LANDSAT_5", 10.0),  # winter 2010
        ]
    ):
        rows.append(
            {
                "PRODUCT_ID": f"p{i}",
                "SPACECRAFT_ID": sat,
                "COLLECTION_CATEGORY": "T1",
                "WRS_PATH": 204,
                "WRS_ROW": 30,
                "CLOUD_COVER": cc,
                "date": pd.Timestamp(date),
            }
        )
    idx = pd.DataFrame(rows)
    idx["season_year"] = idx["date"].dt.year + (idx["date"].dt.month >= 11).astype(int)
    sc = LS.choose_scenes(idx, "2010", per_season=2)
    summer = sc[sc["season"] == "summer"]["PRODUCT_ID"].tolist()
    assert summer == ["p2", "p0"]
    assert sc[sc["season"] == "winter"]["PRODUCT_ID"].tolist() == ["p4"]


def test_quantile_match_maps_distribution_onto_reference():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.6, 0.1, (1, 60, 60)).astype("float16")
    src = (ref.astype("float32") * 0.8 - 0.05).astype("float16")  # older sensor: gain and bias
    mask = np.ones((60, 60), bool)
    out = LS.quantile_match(src, ref, mask, n=3600)
    assert np.nanmax(np.abs(out.astype("float32") - ref.astype("float32"))) < 0.01
