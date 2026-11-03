"""post/fetch/gridsat.py: GridSat-B1 IR -> wtune-obs/1 satellite_ir that obs-check accepts."""
import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from wtune.post import agent_obs, sources  # noqa: E402
from wtune.post.fetch import gridsat  # noqa: E402


def _gridsat_file(path, tb_k):
    """A GridSat-B1-shaped file: packed int16 with valid_range in unpacked kelvin."""
    lat = np.arange(-70.0, 70.0, 0.5)
    lon = np.arange(-180.0, 180.0, 0.5)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        v = ds.createVariable("irwin_cdr", "i2", ("time", "lat", "lon"), fill_value=np.int16(-31999))
        v.scale_factor, v.add_offset = np.float32(0.01), np.float32(200.0)
        v.valid_range = np.array([140.0, 375.0], "f4")
        v.units = "Kelvin"
        ds.set_auto_maskandscale(False)
        raw = np.full((1, len(lat), len(lon)), -31999, "i2")
        jj = (lat >= 15) & (lat <= 35)
        raw[0, jj, :] = np.round((tb_k - 200.0) / 0.01).astype("i2")
        v[:] = raw


def test_valid_times_are_3_hourly_inside_the_window():
    got = gridsat.valid_times(datetime(2017, 7, 31, 1), datetime(2017, 7, 31, 10))
    assert got == [datetime(2017, 7, 31, h) for h in (3, 6, 9)]


def test_fetch_writes_satellite_ir_that_obs_check_accepts(tmp_path, monkeypatch):
    src = str(tmp_path / "global.nc")
    _gridsat_file(src, 230.5)
    got = []

    def fake_download(url, dest, known_sha=None):
        got.append(url)
        shutil.copy(src, dest)
        return sources.sha256_file(dest), True
    monkeypatch.setattr(sources, "download", fake_download)
    proj = str(tmp_path / "proj")
    args = argparse.Namespace(project=proj, run_dir=None, start="2017-08-01_20", end="2017-08-02_01",
                              bbox="20,27,116,123", jobs=2)
    paths = gridsat.run(args)
    assert [os.path.basename(p) for p in paths] == ["irtb_201708012100.nc", "irtb_201708020000.nc"]
    assert got[0].endswith("2017/GRIDSAT-B1.2017.08.01.21.v02r01.nc")
    assert not [f for f in os.listdir(os.path.dirname(paths[0])) if f.endswith(".global.nc")]
    lat, lon, tb = agent_obs.read_grid(paths[0], "tb")
    # automatic masking would compare the K valid_range with packed ints and mask it all
    inside = (lat >= 20) & (lat <= 27)
    assert np.allclose(tb[inside], 230.5) and np.isnan(tb[lat > 35]).all()
    assert lon.min() >= 115.0 and lon.max() <= 124.0
    errors, _warn = agent_obs.check(os.path.dirname(paths[0]))
    assert errors == []
    assert gridsat.run(args) == paths and len(got) == 2         # second run: nothing downloaded
