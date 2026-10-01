"""Decision 0064: figure conventions from the WRF/NWP verification literature."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from post import obs_figs  # noqa: E402
from post.recipes import mapfig, synoptic  # noqa: E402


def test_wpc_precip_levels_in_mm():
    assert obs_figs.PRECIP_LEVELS[:4] == [0.25, 2.54, 6.35, 12.7]
    assert obs_figs.PRECIP_LEVELS[4] == 25.4
    assert len(obs_figs.PRECIP_LEVELS) == len(obs_figs.PRECIP_LEVELS_IN)


def test_knots_factor():
    assert synoptic.KT_PER_MS == pytest.approx(1.943844, rel=1e-6)


def test_celsius_levels_symmetric_about_zero():
    t = np.linspace(-31.0, 7.0, 200)
    lv = synoptic.celsius_levels(t)
    assert lv[0] == -lv[-1]
    assert 0.0 in lv
    assert np.allclose(np.diff(lv), lv[1] - lv[0])


def test_performance_diagram_success_ratio():
    rows = [dict(thresh=1.0, pod=0.99, far=0.32, csi=0.67, fbias=1.47, total=3973),
            dict(thresh=10.0, pod=0.14, far=0.70, csi=0.10, fbias=0.46, total=3973),
            dict(thresh=25.4, pod=float("nan"), far=float("nan"), csi=float("nan"),
                 fbias=float("nan"), total=3973)]
    pts = obs_figs.performance_diagram_points(rows)
    assert [p[0] for p in pts] == [1.0, 10.0]            # NaN threshold dropped
    assert pts[0][1] == pytest.approx(0.68)              # SR = 1 - FAR
    assert pts[1][2] == pytest.approx(0.14)


def test_performance_diagram_renders(tmp_path):
    rows = [dict(thresh=1.0, pod=0.99, far=0.32, csi=0.67, fbias=1.47, total=10)]
    out = obs_figs.scores_vs_threshold(rows, "t", str(tmp_path / "pd.png"), "mm")
    assert os.path.getsize(out) > 1000


def test_highs_lows_finds_one_low_and_one_high():
    y, x = np.mgrid[0:60, 0:80]
    slp = 1012 + 12 * np.exp(-((x - 60) ** 2 + (y - 30) ** 2) / 120.0) \
        - 15 * np.exp(-((x - 20) ** 2 + (y - 30) ** 2) / 120.0)
    hl = mapfig.highs_lows(slp, window=21)
    kinds = sorted(k for _, _, k in hl)
    assert kinds == ["H", "L"]
    low = [(j, i) for j, i, k in hl if k == "L"][0]
    assert abs(low[1] - 20) <= 1 and abs(low[0] - 30) <= 1


def test_bias_rmse_series_with_ci_and_counts(tmp_path):
    t = [datetime(2019, 1, 28, 6) + timedelta(hours=2 * k) for k in range(6)]
    ci = dict(bias_lo=[-2] * 6, bias_hi=[0] * 6, rmse_lo=[1] * 6, rmse_hi=[3] * 6)
    out = obs_figs.bias_rmse_series({"T2": t}, {"T2": ("°C", [-1] * 6, [2] * 6, [30] * 6, ci)},
                                    "t", str(tmp_path / "s.png"))
    assert os.path.getsize(out) > 1000


def test_station_temperature_units_are_celsius():
    from post import met_pointstat
    assert met_pointstat.VARS["TMP"]["unit"] == "°C"
    assert met_pointstat.VARS["DPT"]["unit"] == "°C"


def test_highs_lows_skips_excluded_cells():
    y, x = np.mgrid[0:60, 0:80]
    slp = 1012 - 15 * np.exp(-((x - 20) ** 2 + (y - 30) ** 2) / 120.0)
    mask = np.zeros(slp.shape, bool)
    mask[20:40, 10:30] = True                      # "mountains" over the low
    assert [k for _, _, k in mapfig.highs_lows(slp, window=21, exclude=mask)].count("L") == 0


def test_reflectivity_table_fits_boundarynorm_with_overflow():
    pytest.importorskip("metpy")
    from matplotlib.colors import BoundaryNorm
    from post.conventions import DBZ_LEVELS, reflectivity_cmap
    cm = reflectivity_cmap()
    BoundaryNorm(DBZ_LEVELS, cm.N, extend="max")      # raises if the table is one colour short
    assert DBZ_LEVELS[0] == 5 and DBZ_LEVELS[-1] == 75 and DBZ_LEVELS[1] - DBZ_LEVELS[0] == 5
