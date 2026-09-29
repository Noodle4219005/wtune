"""Tests for post/tc_metrics.py against an analytic Rankine vortex -- the only surviving
piece is quadrant_radii() (decision 0056, 2026-09-29 update deleted vmax_rmw/azimuthal_
mean_profile/core_ring_means/choose_finest_domain/choose_fine_domain: Vmax/MSLP now come
straight from TempestExtremes' own track, RMW/azimuthal structure from MET tc_rmw).

A Rankine vortex has a closed-form wind profile: linear rise from the centre to Rmw, then
1/r decay beyond it (speed(r) = Vmax*r/Rmw for r<=Rmw, else Vmax*Rmw/r). Built on a real
lat/lon grid (post/tc_metrics.dist_km itself computes the radius), so quadrant radii have
a known closed-form answer to check quadrant_radii() against, independent of any wrfout
fixture.
"""
import numpy as np
import pytest

import post.tc_metrics as tm

LAT0, LON0 = 20.0, 260.0
VMAX, RMW = 40.0, 50.0
HALF_DEG = 6.0
N = 241                         # grid spacing ~5.56 km, fine enough for tight tolerances
GRID_KM = 5.56


def _grid():
    lat1d = np.linspace(LAT0 - HALF_DEG, LAT0 + HALF_DEG, N)
    lon1d = np.linspace(LON0 - HALF_DEG, LON0 + HALF_DEG, N)
    lon, lat = np.meshgrid(lon1d, lat1d)
    rr = tm.dist_km(LAT0, LON0, lat, lon)
    return lat, lon, rr


def _rankine(rr, vmax=VMAX, rmw=RMW):
    return np.where(rr <= rmw, vmax * rr / rmw, vmax * rmw / np.maximum(rr, 1e-9))


def test_dist_km_symmetric_and_zero_at_centre():
    assert tm.dist_km(LAT0, LON0, LAT0, LON0) == pytest.approx(0.0, abs=1e-9)
    d1 = tm.dist_km(LAT0, LON0, LAT0 + 1.0, LON0)
    d2 = tm.dist_km(LAT0 + 1.0, LON0, LAT0, LON0)
    assert d1 == pytest.approx(d2, abs=1e-9)


def test_quadrant_radii_symmetric_vortex_all_quadrants_equal():
    lat, lon, rr = _grid()
    speed = _rankine(rr)
    qr = tm.quadrant_radii(LAT0, LON0, lat, lon, rr, speed)
    for name in ("r34", "r50", "r64"):
        vals = [qr["%s_%s_km" % (name, qn)] for qn in tm.QUADRANT_NAMES]
        assert max(vals) - min(vals) < 2 * GRID_KM, (name, vals)
        th = tm.THRESHOLDS_MS[name]
        expected = VMAX * RMW / th
        assert np.mean(vals) == pytest.approx(expected, abs=2 * GRID_KM)


def test_quadrant_radii_asymmetric_vortex_differ_by_quadrant():
    lat, lon, rr = _grid()
    q = tm.quadrant_index(LAT0, LON0, lat, lon)
    factor = np.select([q == 0, q == 1, q == 2, q == 3], [1.0, 0.8, 0.6, 0.9])
    speed = _rankine(rr) * factor
    qr = tm.quadrant_radii(LAT0, LON0, lat, lon, rr, speed)

    for name in ("r34", "r50", "r64"):
        th = tm.THRESHOLDS_MS[name]
        for qn, f in zip(tm.QUADRANT_NAMES, (1.0, 0.8, 0.6, 0.9)):
            local_vmax = VMAX * f
            got = qr["%s_%s_km" % (name, qn)]
            if local_vmax < th:
                assert got == 0.0
            else:
                expected = min(tm.QUADRANT_RADIUS_MAX_KM, local_vmax * RMW / th)
                assert got == pytest.approx(expected, abs=2 * GRID_KM)

    # NE (factor=1.0) must have the largest radius at every threshold: quadrant asymmetry
    # is actually resolved, not averaged away.
    for name in ("r34", "r50"):
        assert qr["%s_ne_km" % name] > qr["%s_sw_km" % name]
