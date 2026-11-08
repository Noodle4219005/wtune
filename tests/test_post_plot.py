"""Smoke test for post/plotting.py: figure gets written, importorskip'd when matplotlib
or wrf-python are not installed."""

import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")
matplotlib = pytest.importorskip("matplotlib")

import numpy as np

from test_post_extract import _write_wrfout, NY, NX, TIMES  # noqa: E402
import wtune.post.plotting as plotting


@pytest.fixture
def run_dir(tmp_path):
    for mm, t in zip((0.0, 2.0), TIMES):
        _write_wrfout(tmp_path / ("wrfout_d01_%s" % t), [t], rainc_mm=mm)
        with netCDF4.Dataset(tmp_path / ("wrfout_d01_%s" % t), "a") as ds:
            ds.createVariable("LANDMASK", "f4", ("Time", "south_north", "west_east"))[:] = \
                np.zeros((1, NY, NX), dtype=np.float32)
    return tmp_path


def test_plot_field_with_wind_overlay(run_dir, tmp_path):
    out_dir = tmp_path / "figs2"
    written = plotting.plot_field(str(run_dir), "slp", str(out_dir), overlay_wind=True,
                                  times=[TIMES[0]])
    assert len(written) == 1


def test_plot_field_cartopy_basemap_if_available(run_dir, tmp_path):
    pytest.importorskip("cartopy")
    out_dir = tmp_path / "figs3"
    written = plotting.plot_field(str(run_dir), "slp", str(out_dir), basemap="cartopy",
                                  times=[TIMES[0]])
    assert len(written) == 1
