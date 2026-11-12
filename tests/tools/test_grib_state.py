#!/usr/bin/env python3
"""Offline tests for tools/grib_state.py.

Source selection, URL construction, .idx parsing, availability checking, download, GRIB decode,
and bbox crop all moved to tools/metdata.py -- see tools/test_metdata.py for those. This suite
covers what's still grib_state.py's own: nearest-neighbour regrid, feature math (matching the
WRF-side thresholds in tools/state_timeline.py), the RSL-lite patch-imbalance helpers, and
sunlit_frac against wtune.solar directly. No network access, no eccodes/cfgrib required.

Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q tools/test_grib_state.py
"""
import datetime
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtune.tools import grib_state as gs          # noqa: E402


# ---------------------------------------------------------------------------
# Nearest-neighbour regrid
# ---------------------------------------------------------------------------

class TestRegridNearest(unittest.TestCase):


    def test_out_of_bounds_clips(self):
        src_lats = np.array([0.0, 1.0, 2.0])
        src_lons = np.array([0.0, 1.0, 2.0])
        src_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        tgt_lat = np.array([[99.0], [-99.0]])
        tgt_lon = np.array([[99.0], [-99.0]])
        out = gs.regrid_nearest_regular(src_lats, src_lons, src_vals, tgt_lat, tgt_lon)
        self.assertEqual(out[0, 0], 9.0)   # clipped to top-right corner
        self.assertEqual(out[1, 0], 1.0)   # clipped to bottom-left corner

    def test_0_360_source_with_pm180_target(self):
        # Monotonic 0..360 source grid (GFS convention, e.g. columns near 255..257 deg == WRF's
        # -105..-103 pm180); target given in pm180 must land on the matching 0..360 column.
        src_lats = np.array([0.0, 1.0])
        src_lons = np.array([255.0, 256.0, 257.0])
        src_vals = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        tgt_lat = np.array([[0.0]])
        tgt_lon = np.array([[-104.0]])   # pm180 -104.0 == 256.0 in 0..360
        out = gs.regrid_nearest_regular(src_lats, src_lons, src_vals, tgt_lat, tgt_lon)
        self.assertEqual(out[0, 0], 20.0)


# ---------------------------------------------------------------------------
# Feature math on synthetic arrays
# ---------------------------------------------------------------------------

class TestFeatureMath(unittest.TestCase):
    def test_precip_frac_and_mean(self):
        rate = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 5.0])   # 3 of 6 strictly > 1.0 (1.5, 2.0, 5.0)
        self.assertAlmostEqual(gs.precip_frac(rate), 3.0 / 6.0)
        self.assertAlmostEqual(gs.precip_mean(rate), rate.mean())

    def test_precip_rate_from_apcp_window(self):
        apcp = np.array([3.0, 6.0, 9.0])
        rate = gs.precip_rate_from_apcp_window(apcp, 3.0)
        np.testing.assert_allclose(rate, [1.0, 2.0, 3.0])

    def test_precip_rate_rejects_nonpositive_window(self):
        with self.assertRaises(ValueError):
            gs.precip_rate_from_apcp_window(np.array([1.0]), 0.0)

    def test_pblh_mean_and_p90(self):
        pblh = np.arange(1, 101, dtype=float)   # 1..100
        self.assertAlmostEqual(gs.pblh_mean(pblh), 50.5)
        self.assertAlmostEqual(gs.pblh_p90(pblh), np.percentile(pblh, 90))

    def test_wind10_max(self):
        u = np.array([3.0, 0.0])
        v = np.array([4.0, 1.0])
        self.assertAlmostEqual(gs.wind10_max(u, v), 5.0)

    def test_hfx_mean(self):
        shtfl = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(gs.hfx_mean(shtfl), 20.0)


# ---------------------------------------------------------------------------
# sunlit_frac against wtune.solar directly
# ---------------------------------------------------------------------------

class TestSunlitFrac(unittest.TestCase):


    def test_sunlit_frac_derived_from_cos_zenith_grid(self):
        lat = np.array([[0.0, 0.0]])
        lon = np.array([[0.0, 180.0]])
        dt = datetime.datetime(2021, 3, 20, 12, 0, 0)
        cz = gs.cos_zenith_grid(lat, lon, dt)
        self.assertAlmostEqual(gs.sunlit_frac(lat, lon, dt), float(np.mean(cz > 0)))


if __name__ == '__main__':
    unittest.main()
