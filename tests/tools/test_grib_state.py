#!/usr/bin/env python3
"""Offline tests for tools/grib_state.py.

Source selection, URL construction, .idx parsing, availability checking, download, GRIB decode,
and bbox crop all moved to tools/metdata.py -- see tools/test_metdata.py for those. This suite
covers what's still grib_state.py's own: nearest-neighbour regrid, feature math (matching the
WRF-side thresholds in tools/state_timeline.py), the RSL-lite patch-imbalance helpers, and
sunlit_frac against wtune_solar directly. No network access, no eccodes/cfgrib required.

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
from wtune import solar as wsolar     # noqa: E402


# ---------------------------------------------------------------------------
# Nearest-neighbour regrid
# ---------------------------------------------------------------------------

class TestRegridNearest(unittest.TestCase):
    def test_exact_grid_points_roundtrip(self):
        src_lats = np.array([10.0, 10.25, 10.5, 10.75])
        src_lons = np.array([100.0, 100.25, 100.5])
        src_vals = np.arange(12.0).reshape(4, 3)
        # target grid exactly = row 1 of the source grid, replicated
        tgt_lat = np.full((1, 3), src_lats[1])
        tgt_lon = src_lons.reshape(1, 3)
        out = gs.regrid_nearest_regular(src_lats, src_lons, src_vals, tgt_lat, tgt_lon)
        np.testing.assert_array_equal(out, src_vals[1:2, :])

    def test_off_grid_point_snaps_to_nearest(self):
        src_lats = np.array([0.0, 1.0, 2.0])
        src_lons = np.array([0.0, 1.0, 2.0])
        src_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        tgt_lat = np.array([[0.9]])
        tgt_lon = np.array([[1.4]])
        out = gs.regrid_nearest_regular(src_lats, src_lons, src_vals, tgt_lat, tgt_lon)
        # nearest to (0.9, 1.4) is source index (lat=1 -> row1, lon=1 -> col1) = 5.0
        self.assertEqual(out[0, 0], 5.0)

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
# sunlit_frac against wtune_solar directly
# ---------------------------------------------------------------------------

class TestSunlitFrac(unittest.TestCase):
    def test_matches_cos_zenith_at_known_point(self):
        # Equator, noon local solar time (lon=0, UTC=12:00) on the equinox-ish date (doy s.t.
        # declination ~0): cos_zenith should be ~1 (sun overhead) -> definitely sunlit.
        lat = np.array([[0.0]])
        lon = np.array([[0.0]])
        dt = datetime.datetime(2021, 3, 20, 12, 0, 0)  # near vernal equinox
        frac = gs.sunlit_frac(lat, lon, dt)
        self.assertEqual(frac, 1.0)

        doy = wsolar.day_of_year(dt.month, dt.day)
        decl = wsolar.declination_deg(doy)
        cz = wsolar.cos_zenith(0.0, 0.0, 12.0, decl)
        self.assertGreater(cz, 0.9)   # sun nearly overhead

    def test_matches_cos_zenith_at_local_midnight(self):
        lat = np.array([[0.0]])
        lon = np.array([[0.0]])
        dt = datetime.datetime(2021, 3, 20, 0, 0, 0)   # UTC midnight == local midnight at lon=0
        frac = gs.sunlit_frac(lat, lon, dt)
        self.assertEqual(frac, 0.0)

        doy = wsolar.day_of_year(dt.month, dt.day)
        decl = wsolar.declination_deg(doy)
        cz = wsolar.cos_zenith(0.0, 0.0, 0.0, decl)
        self.assertLess(cz, 0.0)

    def test_mixed_grid_gives_fractional_result(self):
        # Two points 12h of local time apart (lon 0 and lon 180) at the equator: exactly one is
        # sunlit at any given UTC instant (away from the terminator itself).
        lat = np.array([[0.0, 0.0]])
        lon = np.array([[0.0, 180.0]])
        dt = datetime.datetime(2021, 3, 20, 12, 0, 0)
        frac = gs.sunlit_frac(lat, lon, dt)
        self.assertAlmostEqual(frac, 0.5)

    def test_vectorized_matches_pointwise_loop(self):
        rng = np.random.RandomState(0)
        lat = rng.uniform(-60, 60, size=(5, 5))
        lon = rng.uniform(-179, 179, size=(5, 5))
        dt = datetime.datetime(2020, 7, 15, 6, 30, 0)
        got = gs.sunlit_frac(lat, lon, dt)
        doy = wsolar.day_of_year(dt.month, dt.day)
        decl = wsolar.declination_deg(doy)
        utc_hours = 6 + 30.0 / 60.0
        expect_count = sum(
            1 for la, lo in zip(lat.flat, lon.flat)
            if wsolar.cos_zenith(la, lo, utc_hours, decl) > 0
        )
        self.assertAlmostEqual(got, expect_count / lat.size)

    def test_cos_zenith_grid_matches_wtune_solar_pointwise(self):
        lat = np.array([[0.0, 30.0], [-30.0, 60.0]])
        lon = np.array([[0.0, 90.0], [-90.0, 179.0]])
        dt = datetime.datetime(2021, 6, 21, 4, 15, 0)
        got = gs.cos_zenith_grid(lat, lon, dt)
        doy = wsolar.day_of_year(dt.month, dt.day)
        decl = wsolar.declination_deg(doy)
        utc_hours = 4 + 15.0 / 60.0
        expect = np.array([[wsolar.cos_zenith(la, lo, utc_hours, decl) for la, lo in row]
                            for row in zip(lat, lon)])
        # zip(lat, lon) above pairs whole rows; rebuild elementwise properly instead.
        expect = np.array([[wsolar.cos_zenith(lat[j, i], lon[j, i], utc_hours, decl)
                             for i in range(lat.shape[1])] for j in range(lat.shape[0])])
        np.testing.assert_allclose(got, expect)

    def test_sunlit_frac_derived_from_cos_zenith_grid(self):
        lat = np.array([[0.0, 0.0]])
        lon = np.array([[0.0, 180.0]])
        dt = datetime.datetime(2021, 3, 20, 12, 0, 0)
        cz = gs.cos_zenith_grid(lat, lon, dt)
        self.assertAlmostEqual(gs.sunlit_frac(lat, lon, dt), float(np.mean(cz > 0)))


if __name__ == '__main__':
    unittest.main()
