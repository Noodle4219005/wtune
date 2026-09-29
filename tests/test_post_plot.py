"""Smoke test for post/plotting.py: figure gets written, importorskip'd when matplotlib
or wrf-python are not installed."""
import os

import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")
matplotlib = pytest.importorskip("matplotlib")

import numpy as np

from test_post_extract import _write_wrfout, LAT0, DLAT, NY, NX, LON0, DLON, TIMES  # noqa: E402
import post.plotting as plotting


@pytest.fixture
def run_dir(tmp_path):
    for mm, t in zip((0.0, 2.0), TIMES):
        _write_wrfout(tmp_path / ("wrfout_d01_%s" % t), [t], rainc_mm=mm)
        with netCDF4.Dataset(tmp_path / ("wrfout_d01_%s" % t), "a") as ds:
            ds.createVariable("LANDMASK", "f4", ("Time", "south_north", "west_east"))[:] = \
                np.zeros((1, NY, NX), dtype=np.float32)
    return tmp_path


def test_plot_field_writes_one_png_per_time(run_dir, tmp_path):
    out_dir = tmp_path / "figs"
    written = plotting.plot_field(str(run_dir), "slp", str(out_dir))
    assert len(written) == len(TIMES)
    for p in written:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


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


def test_plot_field_falls_back_to_landmask_when_natural_earth_missing(run_dir, tmp_path,
                                                                       monkeypatch, capsys):
    pytest.importorskip("cartopy")
    monkeypatch.setattr(plotting, "_natural_earth_ready", lambda: False)
    out_dir = tmp_path / "figs4"
    written = plotting.plot_field(str(run_dir), "slp", str(out_dir), basemap="cartopy",
                                  times=[TIMES[0]])
    assert len(written) == 1
    assert "falling back to LANDMASK" in capsys.readouterr().out
