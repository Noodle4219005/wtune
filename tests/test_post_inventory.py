"""Tests for post/inventory.py: grid/variable summary (ncdump -h + cdo sinfon, mocked with
the real captures in tests/fixtures/post/) and missing-frame detection (pure, no tool call).
"""
from conftest import swap_wtune_module
import os

import post.inventory as inventory
from post import tools

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")
NCDUMP_H_TEXT = open(os.path.join(FIXTURES, "ncdump_h_sample.txt")).read()
CDO_SINFON_TEXT = open(os.path.join(FIXTURES, "cdo_sinfon_sample.txt")).read()


def _touch_wrfout(path):
    open(path, "w").close()   # ncdump/cdo are mocked -- the file just needs to exist


def test_inventory_grid_and_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ncdump_header", lambda path: NCDUMP_H_TEXT)
    monkeypatch.setattr(tools, "cdo_sinfon", lambda path: CDO_SINFON_TEXT)
    for t in ("2005-08-27_00:00:00", "2005-08-27_06:00:00", "2005-08-27_12:00:00"):
        _touch_wrfout(tmp_path / ("wrfout_d01_%s" % t))

    inv = inventory.inventory(str(tmp_path))
    assert inv["format"] == "netcdf"
    d01 = inv["domains"]["d01"]
    assert d01["n_times"] == 3
    assert d01["grid"]["dx"] == 12000.0
    assert d01["grid"]["e_we"] == 170        # west_east_stag fallback (WEST-EAST_GRID_DIMENSION absent in this capture)
    assert d01["grid"]["lon_range"] == (-99.00387, -78.99613)
    assert "T2" in d01["variables"]
    assert d01["missing"] == []
    summary = inventory.render_summary(inv)
    assert "d01" in summary


def test_inventory_reports_missing_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ncdump_header", lambda path: NCDUMP_H_TEXT)
    monkeypatch.setattr(tools, "cdo_sinfon", lambda path: CDO_SINFON_TEXT)
    # 00, 06, 12 present but 18 missing before the 24h frame -> one 6h gap doubled to 12h
    for t in ("2005-08-27_00:00:00", "2005-08-27_06:00:00", "2005-08-27_12:00:00",
              "2005-08-28_00:00:00"):
        _touch_wrfout(tmp_path / ("wrfout_d01_%s" % t))

    inv = inventory.inventory(str(tmp_path))
    d01 = inv["domains"]["d01"]
    assert len(d01["missing"]) == 1
    m = d01["missing"][0]
    assert m["after"] == "2005-08-27_12:00:00"
    assert m["before"] == "2005-08-28_00:00:00"
    assert m["n_missing_frames"] == 1
    summary = inventory.render_summary(inv)
    assert "MISSING" in summary


def _no_bp_converter(monkeypatch):
    """A wtune_bp whose adios2 interpreter is missing (as on a machine without envs/adios2)."""
    import sys
    import types
    from post import extract

    def unavailable():
        raise RuntimeError("no Python that can `import adios2` was found: run `pixi install` "
                           "in envs/adios2")
    swap_wtune_module(monkeypatch, "bp", types.SimpleNamespace(
        require_adios2_python=unavailable, bp2nc=None))
    monkeypatch.setattr(extract, "_BP_MEMO", {})


def test_inventory_detects_adios2_bp_dirs(tmp_path, monkeypatch):
    _no_bp_converter(monkeypatch)
    (tmp_path / "wrfout_d01_2005-08-27_00:00:00.bp").mkdir()
    (tmp_path / "wrfout_d01_2005-08-27_00:00:00.bp").joinpath("md.idx").touch()        # an empty dir is not a frame
    inv = inventory.inventory(str(tmp_path))
    assert inv["format"] == "adios2_bp"
    assert inv["bp_dirs"]
    assert "wrf_adios2_to_netcdf4" in inv["note"] and "pixi install" in inv["note"]


def test_missing_frames_pure():
    times = ["2005-08-27_00:00:00", "2005-08-27_06:00:00", "2005-08-27_12:00:00",
             "2005-08-28_00:00:00"]
    expected, missing = inventory._missing_frames(times)
    assert expected == 6 * 3600
    assert len(missing) == 1
