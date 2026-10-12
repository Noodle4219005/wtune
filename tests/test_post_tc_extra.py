"""Tests for post/recipes/tc_extra.py's tangential_wind(): sign convention (cyclonic-positive
in the NH) against an idealized solid-body rotating vortex, where the analytic answer is known
exactly. radius_time_plot()'s full wrfout-reading path needs real multi-frame data and an
a-deck -- exercised manually against the real m1 run (see the delegation report).
"""
import numpy as np
import pytest

from wtune.post.recipes import tc_extra


def test_tangential_wind_matches_solid_body_rotation_east_of_centre():
    # A point due east of the centre (dlon>0, dlat=0): for cyclonic (counterclockwise) flow,
    # the wind there points due north (v>0, u=0) -- tangential_wind should read that as fully
    # tangential and positive.
    lat0, lon0 = 25.0, -80.0
    lat, lon = 25.0, -79.9   # east of centre
    u, v = 0.0, 10.0         # due north, i.e. purely cyclonic here
    vt = tc_extra.tangential_wind(lat0, lon0, lat, lon, u, v)
    assert vt == pytest.approx(10.0, rel=1e-6)


def test_tangential_wind_is_negative_for_anticyclonic_flow():
    lat0, lon0 = 25.0, -80.0
    lat, lon = 25.0, -79.9
    u, v = 0.0, -10.0        # due south at a point east of centre -> anticyclonic
    vt = tc_extra.tangential_wind(lat0, lon0, lat, lon, u, v)
    assert vt < 0


def test_tangential_wind_zero_for_purely_radial_flow():
    lat0, lon0 = 25.0, -80.0
    lat, lon = 25.0, -79.9   # east of centre
    u, v = 5.0, 0.0          # purely radial (outward), no tangential component
    vt = tc_extra.tangential_wind(lat0, lon0, lat, lon, u, v)
    assert abs(vt) < 1e-9


def test_tangential_wind_vectorized_over_arrays():
    lat0, lon0 = 25.0, -80.0
    lat = np.array([25.0, 25.1])
    lon = np.array([-79.9, -80.0])
    u = np.array([0.0, 5.0])
    v = np.array([10.0, 0.0])
    vt = tc_extra.tangential_wind(lat0, lon0, lat, lon, u, v)
    assert vt.shape == (2,)
