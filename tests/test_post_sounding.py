"""Smoke test for post/sounding.py; importorskip'd when metpy/wrf-python are missing."""
import os

import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")
matplotlib = pytest.importorskip("matplotlib")
metpy = pytest.importorskip("metpy")

from test_post_extract import _write_wrfout, LAT0, DLAT, NY, LON0, TIMES  # noqa: E402
import post.sounding as sounding


def test_make_sounding_writes_png(tmp_path):
    t = TIMES[0]
    _write_wrfout(tmp_path / ("wrfout_d01_%s" % t), [t])
    out = tmp_path / "skewt.png"
    result = sounding.make_sounding(str(tmp_path), LAT0 + DLAT * NY / 2.0, LON0 + 1.0, t, str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    assert "cape_j_kg" in result and "cin_j_kg" in result
