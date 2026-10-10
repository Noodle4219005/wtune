"""Smoke test for post/sounding.py; importorskip'd when metpy/wrf-python are missing."""
import os

import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")
matplotlib = pytest.importorskip("matplotlib")
metpy = pytest.importorskip("metpy")

from test_post_extract import _write_wrfout, LAT0, DLAT, NY, LON0, TIMES  # noqa: E402
import wtune.post.sounding as sounding


def test_make_sounding_writes_png(tmp_path):
    t = TIMES[0]
    _write_wrfout(tmp_path / ("wrfout_d01_%s" % t), [t])
    out = tmp_path / "skewt.png"
    result = sounding.make_sounding(str(tmp_path), LAT0 + DLAT * NY / 2.0, LON0 + 1.0, t, str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    assert "cape_j_kg" in result and "cin_j_kg" in result


def test_add_parcel_shades_cape_and_returns_profile():
    import numpy as np
    from metpy.plots import SkewT
    from metpy.units import units
    import matplotlib.pyplot as plt
    p = np.array([1000., 900, 800, 700, 600, 500, 400, 300, 200])
    t = np.array([30., 22, 15, 8, 0, -10, -25, -40, -55])
    td = np.array([24., 18, 10, 0, -10, -20, -35, -50, -65])
    fig = plt.figure()
    skew = SkewT(fig)
    prof = sounding.add_parcel(skew, p * units.hPa, t * units.degC, td * units.degC)
    assert prof is not None and len(prof) == len(p)
    assert abs(prof[0].magnitude - 30.0) < 1e-6          # parcel starts at the surface T
    assert len(skew.ax.collections) >= 1                 # shaded CAPE and/or CIN polygons
    assert sounding.add_parcel(skew, p[::-1] * units.hPa, t * units.degC, td * units.degC) is None
    plt.close(fig)
