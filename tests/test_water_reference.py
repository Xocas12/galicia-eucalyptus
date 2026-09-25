"""Catchment routing, gauge ingestion, the runoff power study and GBIF reference scoring."""

import numpy as np
import pandas as pd

from eucalyptus_impact.real import reference as R
from eucalyptus_impact.real import water as W


def _valley(ny=40, nx=30):
    """A valley draining south with a pit in it; sea (NaN) along the bottom row."""
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    z = 2.0 * (ny - y) + 0.5 * np.abs(x - nx // 2)
    z[20, 15] -= 30  # a depression that must be filled, not trap flow
    z[-1, :] = np.nan
    return z


def test_priority_flood_routes_everything_to_the_sea():
    z = _valley()
    rec, order, _edge = W.priority_flood(z)
    land = np.isfinite(z).ravel()
    assert len(order) == land.sum()
    # every cell comes after its receiver in the flooding order
    pos = np.empty(z.size, int)
    pos[order] = np.arange(len(order))
    has = rec[order] >= 0
    assert (pos[rec[order][has]] < pos[order][has]).all()
    acc = W.accumulation(rec, order, 1.0)
    outlet = int(np.argmax(acc))
    r, c = divmod(outlet, z.shape[1])
    # the largest catchment exits near the valley floor at the coast
    assert r == z.shape[0] - 2 and abs(c - 15) <= 1
    # nothing drains into the pit's interior: the pit cell passes flow on
    assert rec[20 * z.shape[1] + 15] >= 0


def test_nested_catchments_contain_their_tributaries():
    z = _valley()
    rec, order, _ = W.priority_flood(z)
    acc = W.accumulation(rec, order, 1.0)
    down = int(np.argmax(acc))
    up = int(np.argmax(np.where(np.arange(z.size) // z.shape[1] == 15, acc, 0)))
    mem = W.catchment_members(rec, order, np.array([down, up]))
    assert set(mem[1]).issubset(set(mem[0]))
    assert len(mem[0]) == acc[down]
    assert len(mem[1]) == acc[up]


def test_generic_gauge_files_and_runoff(tmp_path):
    pd.DataFrame({"id": ["g1"], "lon": [-8.0], "lat": [42.8], "area_km2": [100.0]}).to_csv(
        tmp_path / "stations.csv", index=False
    )
    dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
    pd.DataFrame({"id": "g1", "date": dates, "q_m3s": 2.0}).to_csv(
        tmp_path / "flows.csv", index=False
    )
    st, fl = W.load_gauges(tmp_path)
    assert {"x", "y"}.issubset(st.columns)
    q = W.annual_runoff(fl, st.set_index("id")["area_km2"])
    # 2 m3/s over 100 km2 = 2 * 86400 * 365.25 / 1e8 m = 631 mm
    assert q["year"].tolist() == [2020]
    assert np.isclose(q["runoff"].iloc[0], 2 * 86400 * 365.25 / 1e8 * 1000)


def test_cedex_gauge_files(tmp_path):
    (tmp_path / "estaf.csv").write_text(
        "indroea;lugar;xetrs89;yetrs89;suprest\n1401;Ponte;550000;4750000;250,5\n",
        encoding="latin-1",
    )
    (tmp_path / "afliq.csv").write_text(
        "indroea;fecha;altura;caudal\n1401;01/10/2019;0,5;3,25\n1401;02/10/2019;0,5;3,5\n",
        encoding="latin-1",
    )
    st, fl = W.load_gauges(tmp_path)
    assert st["area_km2"].iloc[0] == 250.5
    assert fl["q_m3s"].tolist() == [3.25, 3.5]
    assert fl["date"].iloc[0] == pd.Timestamp("2019-10-01")


def test_thornthwaite_is_plausible_for_galicia():
    t = np.array([8, 9, 11, 12, 15, 18, 20, 20, 18, 15, 11, 9], float)
    hi = float(((t / 5) ** 1.514).sum())
    pet = W.thornthwaite(t, hi)
    assert 600 < pet.sum() < 850
    assert pet.argmax() in (6, 7)


YEARS = tuple(range(2001, 2025))


def _panel(k=40, years=YEARS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(k):
        f0, jump = rng.uniform(0.05, 0.5), rng.uniform(-0.3, 0.3)
        for y in years:
            rows.append(
                {
                    "catchment": c,
                    "year": y,
                    "precip": rng.uniform(900, 2000),
                    "pet": rng.uniform(680, 800),
                    "f_eucalyptus": np.clip(f0 + (y >= 2012) * jump, 0, 1),
                    "f_pine": 0.2,
                    "f_native_broadleaf": 0.15,
                }
            )
    return pd.DataFrame(rows)


def test_power_study_recovers_a_known_effect():
    ps = W.power_study(
        _panel(), n_catchments=(40,), effects=(0.0, -20.0), reps=30, noise_cv=0.03, seed=1
    )
    for _, r in ps.iterrows():
        assert abs(r["bias"]) < 3, r
    strong = ps[ps["true_mm_per_10pts"] == -20]
    assert (strong["power"] > 0.9).all()
    null = ps[ps["true_mm_per_10pts"] == 0]
    assert (null["power"] < 0.2).all()
    mde = W.minimum_detectable(ps)
    assert (mde["mde_mm_per_10pts"] > 0).all()


def test_reference_agreement_and_scores():
    pred = np.array([[0, 0, 1], [2, 2, 0], [255, 1, 1]], np.uint8)
    pts = pd.DataFrame(
        {
            "row": [0, 0, 1, 1, 1, 2, 2],
            "col": [0, 1, 2, 0, 1, 1, 0],
            "ref_class": [0, 0, 0, 2, 0, 1, 1],
        }
    )
    tab = R.agreement_table(pts, pred)
    assert np.isclose(tab.loc[0, 0], 0.75)  # 3 of 4 euc points mapped euc
    assert tab.loc[1, "n"] == 1  # the 255 (no data) point is dropped
    s = R.eucalyptus_scores(pts, pred, prior_euc=0.5)
    assert s["n_euc"] == 4 and np.isclose(s["recall"], 0.75)
    assert s["false_euc_rate"] == 0.0 and np.isclose(s["precision_at_prior"], 1.0)
    lo, hi = R.wilson(3, 4)
    assert 0 < lo < 0.75 < hi <= 1


def test_filter_records_keeps_precise_recent_observations():
    df = pd.DataFrame(
        {
            "gbifid": [1, 2, 3, 4],
            "genus": ["Eucalyptus", "Eucalyptus", "Pinus", "Quercus"],
            "decimallatitude": [42.1, 42.2, 42.3, 42.4],
            "decimallongitude": [-8.1, -8.2, -8.3, -8.4],
            "coordinateuncertaintyinmeters": [10, 500, 20, 30],
            "year": [2022, 2022, 2010, 2023],
            "basisofrecord": ["HUMAN_OBSERVATION"] * 3 + ["PRESERVED_SPECIMEN"],
        }
    )
    d = R.filter_records(df)
    assert d["gbifid"].tolist() == [1]
    assert d["ref_class"].tolist() == [0]
