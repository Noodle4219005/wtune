"""Unit tests for post/fetch/igra.py parsing and post/sounding.py PBL-height / comparison helpers
(small hand-written fixtures; real-data checks are in the scc runs recorded in the wiki)."""
import datetime as dt

import numpy as np
import pytest

from wtune.post.fetch import igra

HDR = "#USM00072233 2005 08 27 12 1115  3 ncdc6301 ncdc6301  303369  -898250"


def _lv(t1, t2, press, gph, temp, rh, dpdp, wdir, wspd):
    """One IGRA2 data record in the documented fixed columns (PRESS 10-15, GPH 17-21, ...)."""
    return "%d%d %5d %6d%s%5d%s%5d%s%5d %5d %5d %5d" % (t1, t2, 0, press, " ", gph, " ", temp, " ", rh, dpdp,
                                                        wdir, wspd)


SFC = _lv(2, 1, 101000, 99, 250, 900, 30, 70, 21)
LV2 = _lv(1, 0, 92500, 784, 240, 600, 90, 189, 26)
LV3 = _lv(1, 0, 85000, 1528, -9999, -8888, 200, 150, 10)
HDR2 = "#USM00072233 2005 08 28 00 2305  1 ncdc6301 ncdc6301  303369  -898250"
LATE = "#USM00072233 2005 09 20 12 1115  1 ncdc6301 ncdc6301  303369  -898250"


def _lines():
    return [HDR, SFC, LV2, LV3, HDR2, SFC, LATE, SFC]


def test_parse_level_units_and_missing():
    lv = igra.parse_level(LV3)
    assert lv["pressure_hPa"] == 850.0 and lv["height_m"] == 1528.0
    assert lv["temp_C"] is None and lv["dewpoint_C"] is None      # -9999 / QA-removed
    lv = igra.parse_level(LV2)
    assert lv["temp_C"] == pytest.approx(24.0)
    assert lv["dewpoint_C"] == pytest.approx(24.0 - 9.0)          # T - dewpoint depression
    assert lv["wspd_ms"] == pytest.approx(2.6)


def test_iter_soundings_window_hours_and_early_stop():
    u = dt.timezone.utc
    start, end = dt.datetime(2005, 8, 27, 0, tzinfo=u), dt.datetime(2005, 8, 29, 0, tzinfo=u)
    got = list(igra.iter_soundings(_lines(), start, end, hours=(0, 12)))
    assert [t.day * 100 + t.hour for t, _ in got] == [2712, 2800]
    assert len(got[0][1]) == 3
    only12 = list(igra.iter_soundings(_lines(), start, end, hours=(12,)))
    assert len(only12) == 1


def test_csv_roundtrip(tmp_path):
    u = dt.timezone.utc
    got = list(igra.iter_soundings(_lines(), dt.datetime(2005, 8, 27, tzinfo=u),
                                   dt.datetime(2005, 8, 28, tzinfo=u)))
    path = str(tmp_path / "S.csv")
    sha = igra.write_station_csv(path, got)
    assert sha == igra.sha256_file(path)
    back = igra.read_soundings(path)
    t = dt.datetime(2005, 8, 27, 12, tzinfo=u)
    assert back[t]["pressure_hPa"].tolist() == [1010.0, 925.0, 850.0]
    assert np.isnan(back[t]["temp_C"][2])


def test_select_stations_spreads_and_filters():
    u = dt.datetime(2005, 8, 27, tzinfo=dt.timezone.utc)
    st = [dict(id="S%d" % k, lat=30 + 0.1 * k, lon=-90 + 0.1 * k, elev_m=1.0, name="n", first=1990, last=2026)
          for k in range(10)]
    st.append(dict(id="OUT", lat=50, lon=-90, elev_m=0, name="n", first=1990, last=2026))
    st.append(dict(id="OLD", lat=30, lon=-90, elev_m=0, name="n", first=1946, last=1990))
    chosen = igra.select_stations(st, (29, 32, -91, -89), u, u, max_n=3)
    ids = [s["id"] for s in chosen]
    assert len(ids) == 3 and "OUT" not in ids and "OLD" not in ids
    assert "S0" in ids and "S9" in ids            # spread to the extremes, not the first three


def test_station_list_parse():
    line = "%-11s %8.4f %9.4f %6.1f %2s %-30s %4d %4d %6d" % ("USM00072233", 30.3369, -89.825, 9.9, "LA",
                                                          "SLIDELL/MUN. LA", 1957, 2026, 44000)
    s = igra.parse_station_list(line)[0]
    assert s["id"] == "USM00072233" and s["lat"] == pytest.approx(30.3369) and s["last"] == 2026


metpy = pytest.importorskip("metpy")
import wtune.post.sounding as sounding  # noqa: E402


def _column(z_top=3000.0, mixed_depth=1000.0, n=60):
    """Well-mixed layer (constant theta) below mixed_depth, stable above, light constant wind."""
    z = np.linspace(0.0, z_top, n)
    theta = np.where(z < mixed_depth, 300.0, 300.0 + 0.006 * (z - mixed_depth))
    p = 1000.0 * np.exp(-z / 8000.0)
    t = theta * (p / 1000.0) ** 0.2854 - 273.15
    return dict(p=p, t=t, td=t - 15.0, u=np.full(n, 3.0), v=np.zeros(n), z=z, ter=0.0, pblh_model=np.nan)


def test_pblh_bulk_ri_finds_inversion_base():
    h = sounding.pblh_bulk_ri(_column(mixed_depth=1000.0))
    assert 1000.0 <= h <= 1600.0          # Ri_c is crossed a little above the mixed-layer top


def test_pblh_bulk_ri_missing_wind_levels_are_interpolated():
    col = _column()
    col["u"] = col["u"].copy()
    col["u"][5:40] = np.nan               # sondes report wind on a subset of levels
    col["v"] = col["v"].copy()
    col["v"][5:40] = np.nan
    assert sounding.pblh_bulk_ri(col) == pytest.approx(sounding.pblh_bulk_ri(_column()), abs=1.0)


def test_pblh_bulk_ri_floor_and_none():
    stable = _column(mixed_depth=0.0)
    assert sounding.pblh_bulk_ri(stable) == pytest.approx(sounding.ZMIN_M, abs=60.0) or \
        sounding.pblh_bulk_ri(stable) >= sounding.ZMIN_M
    unstable = _column(mixed_depth=1e9)   # constant theta everywhere: Ri never exceeds critical
    assert np.isnan(sounding.pblh_bulk_ri(unstable))


def test_profile_stats_zero_for_identical_and_bias_sign():
    col = _column()
    assert sounding.profile_stats(col, col)["T"]["rmse"] == pytest.approx(0.0, abs=1e-9)
    warm = dict(col, t=col["t"] + 2.0)
    st = sounding.profile_stats(warm, col)
    assert st["T"]["bias"] == pytest.approx(2.0) and st["T"]["n"] > 10


def test_fill_below_valid_leaves_no_nan_wedge():
    from wtune.post.recipes.pbl import _fill_below_valid
    a = np.array([[np.nan, 1.0], [np.nan, 2.0], [5.0, 3.0]])
    out = _fill_below_valid(a)
    assert not np.isnan(out).any() and out[0, 0] == 5.0
