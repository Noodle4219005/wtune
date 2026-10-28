"""Fixed, comparable colour scales for the time-series / change-map figures (Katrina m1 review,
2026-09-29): one scale per field across all times, change maps symmetric about a white zero bin,
rain 'change' sequential from 0."""
import numpy as np

from wtune.post import changemap
from wtune.post.recipes import mapfig


def test_pooled_levels_span_all_frames_not_one():
    f0 = np.linspace(987, 1011, 100)
    f1 = np.linspace(965, 1010, 100)
    lev = mapfig.pooled_levels([f0, f1])
    assert lev[0] <= 968 and lev[-1] >= 1008
    lev_one = mapfig.pooled_levels([f0])
    assert lev_one[0] > lev[0]           # a single frame would have given a narrower range


def test_pooled_levels_floor_clamps_lower_bound():
    lev = mapfig.pooled_levels([np.linspace(0.3, 30, 200)], floor=0.0)
    assert lev[0] >= 0.0


def test_change_scale_symmetric_with_no_zero_boundary():
    diffs = [np.random.RandomState(0).randn(50, 50) * 2, np.random.RandomState(1).randn(50, 50)]
    levels, cmap, extend = changemap._change_scale("t2", diffs)
    assert cmap == "RdBu_r" and extend == "both"
    assert np.allclose(levels, -levels[::-1])
    assert not np.any(np.isclose(levels, 0.0))
    assert len(levels) % 2 == 0          # odd number of bins -> the central bin is the white one


def test_change_scale_rain_is_sequential_on_wpc_thresholds():
    # decision 0064: interval rainfall on the WPC isohyets; below 0.01 in stays unshaded
    from wtune.post.obs_figs import PRECIP_LEVELS
    levels, cmap, extend = changemap._change_scale("rain_acc", [np.array([0.0, 5.0, 80.0])])
    assert list(levels) == PRECIP_LEVELS[:len(levels)] and levels[0] == 0.25
    assert cmap != "RdBu_r" and extend == "max"


def test_change_maps_share_one_scale_per_field_and_mode(tmp_path, monkeypatch):
    times = ["2005-08-27_%02d:00:00" % h for h in (0, 6, 12, 18)]
    amp = {"2005-08-27_06:00:00": 1.0, "2005-08-27_12:00:00": 3.0, "2005-08-27_18:00:00": 2.0}
    monkeypatch.setattr(changemap, "_compute_diff",
                        lambda rd, f, dom, od, a, b: (np.zeros((4, 4)), np.zeros((4, 4)),
                                                     amp[b] * np.linspace(-1, 1, 16).reshape(4, 4)))
    seen = []
    monkeypatch.setattr(changemap, "_draw_change",
                        lambda rd, f, dom, op, a, b, h, la, lo, d, scale: seen.append((f, a, scale)))
    recs = changemap.figures_for_change_maps("/fake", "r", 2, str(tmp_path), times, fields=("t2",))
    assert all(not r["missing"] for r in recs)
    t2 = [s for s in seen if s[0] == "t2"]
    # order: consec(0-6), [rain], consec(6-12), vs_t0(0-12), ... -> split by pair start
    vs_t0 = [s[2][0] for s in t2 if s[1] == times[0]][1:]     # drop the first consecutive pair
    consec = [s[2][0] for s in t2 if s[1] != times[0]] + [[s[2][0] for s in t2 if s[1] == times[0]][0]]
    assert len(vs_t0) == 2 and np.array_equal(vs_t0[0], vs_t0[1])
    assert all(np.array_equal(consec[0], c) for c in consec)


def test_minmax_levels_keep_the_extremes():
    f0 = np.full((50, 50), 1005.0)
    f1 = f0.copy()
    f1[25, 25] = 944.0                   # one-cell eye: a 2nd-98th percentile clip would drop it
    lev = mapfig.pooled_levels([f0, f1], lo_pct=0.0, hi_pct=100.0, nbins=18)
    assert lev[0] <= 944.0 and lev[-1] >= 1005.0
    assert mapfig.pooled_levels([f0, f1])[0] > 944.0


def test_time_series_fields_include_pw_and_rules_documented():
    from wtune.post import timeseries
    assert "pw" in timeseries.TIME_SERIES_FIELDS
    assert timeseries.SCALE_RULE["slp"].startswith("true min/max")
    assert "percentile" in timeseries.SCALE_RULE["mcape"]


def test_change_scale_rain_over_taiwan_uses_cwa_thresholds():
    # haitang2017 run72: rain_acc-chg6h over Taiwan was drawn on the WPC inch scale
    from wtune.post.rainscale import CWA_LEVELS
    lat, lon = np.meshgrid(np.linspace(21, 26, 5), np.linspace(118, 123, 5), indexing="ij")
    levels, cmap, extend = changemap._change_scale("rain_acc", [np.array([0.0, 5.0, 80.0])],
                                                   lat, lon)
    assert list(levels) == [float(v) for v in CWA_LEVELS[:len(levels)]] and levels[-1] == 90.0
    assert cmap.name == "cwa_rain" and extend == "max"
