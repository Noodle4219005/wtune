"""Tests for post/met_tc.py: config template rendering, the generic MET table parser
(against a hand-built representative .tcst-like table -- MET is not installed, so this is
not a captured real sample; see the module docstring for what IS/isn't verified), tool-
missing error path, and the cdo-based tc_rmw regrid step with cdo mocked.
"""
import os

import pytest

from post import met_tc, tools

# Representative of a MET .tcst/.stat table: header line of column names, then
# whitespace-padded rows -- the one MET convention this module actually depends on.
FAKE_TCST = (
    "AMODEL BMODEL STORM_ID LEAD    VALID           TK_ERR AMAX_WIND BMAX_WIND\n"
    "WTUN   BEST    AL122005  0 20050827_000000       12.3        88        90\n"
    "WTUN   BEST    AL122005  6 20050827_060000       25.6        90        95\n"
)


def test_parse_met_table():
    rows = met_tc.parse_met_table(FAKE_TCST)
    assert len(rows) == 2
    assert rows[0]["AMODEL"] == "WTUN"
    assert rows[0]["TK_ERR"] == "12.3"
    assert rows[1]["LEAD"] == "6"


def test_parse_met_table_empty():
    assert met_tc.parse_met_table("") == []
    assert met_tc.parse_met_table("   \n  \n") == []


def test_run_tc_pairs_missing_tool_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_POSTPROC_VIEW", raising=False)
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools, "REPO_ROOT", str(tmp_path))  # no real repo postproc view / pixi env
    with pytest.raises(tools.ToolNotFoundError, match="tc_pairs"):
        met_tc.run_tc_pairs("a.dat", "b.dat", str(tmp_path), "WTUN")


def test_run_tc_pairs_builds_config_and_invokes(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "find_postproc_tool", lambda name: "/fake/bin/" + name)
    calls = []

    def _fake_run(cmd, cwd=None, check=True):
        calls.append(cmd)
        open(cmd[cmd.index("-out") + 1] + ".tcst", "w").write(FAKE_TCST)
        return 0, "", ""

    monkeypatch.setattr(tools, "run", _fake_run)
    tcst = met_tc.run_tc_pairs("a.dat", "b.dat", str(tmp_path), "WTUN")
    assert tcst.endswith("tc_pairs.tcst")
    assert os.path.exists(tcst)
    assert calls[0][0] == "/fake/bin/tc_pairs"
    config_path = os.path.join(str(tmp_path), "tc_pairs.conf")
    assert os.path.exists(config_path)
    assert 'model          = "WTUN";' in open(config_path).read()


def test_run_tc_pairs_raises_when_output_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "find_postproc_tool", lambda name: "/fake/bin/" + name)
    monkeypatch.setattr(tools, "run", lambda cmd, cwd=None, check=True: (0, "", ""))
    with pytest.raises(met_tc.METError):
        met_tc.run_tc_pairs("a.dat", "b.dat", str(tmp_path), "WTUN")


def test_regrid_for_tc_rmw_remaps_each_frame_then_merges_to_grib1(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "find_on_path", lambda name, install_hint=None: "/fake/cdo")
    monkeypatch.setattr(tools, "cdo_sinfon", lambda path: "fake")
    monkeypatch.setattr(tools, "parse_cdo_sinfon_grid", lambda text: dict(
        nx=10, ny=8, lon_range=(-90.0, -80.0), lat_range=(20.0, 28.0)))
    fields = []
    monkeypatch.setattr(met_tc, "_wrf_fields_on_pressure_levels",
                        lambda src, dst, timeidx=0: fields.append((src, dst)))
    calls = []
    monkeypatch.setattr(tools, "run", lambda cmd, cwd=None, check=True: calls.append(cmd))

    out_grb = str(tmp_path / "rmw_input.grb")
    frames = ["wrfout_d01_2005-08-27_00:00:00", "wrfout_d01_2005-08-27_06:00:00"]
    assert met_tc.regrid_for_tc_rmw(frames, out_grb) == out_grb

    griddes = open(out_grb + ".griddes").read()
    assert "gridtype = lonlat" in griddes and "xsize = 10" in griddes and "ysize = 8" in griddes
    assert [src for src, _ in fields] == frames              # wrf-python fields per frame
    remaps = [c for c in calls if any(str(a).startswith("remapbil,") for a in c)]
    assert len(remaps) == 2                                    # one remap per frame
    assert any("mergetime" in c for c in calls)
    final = calls[-1]
    assert final[-1] == out_grb and "-f" in final and "grb" in final
    for code in ("-setparam,33.2", "-setparam,34.2"):   # UGRD, VGRD
        assert code in final


def test_regrid_for_tc_rmw_raises_without_grid_extent(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "find_on_path", lambda name, install_hint=None: "/fake/cdo")
    monkeypatch.setattr(tools, "cdo_sinfon", lambda path: "fake")
    monkeypatch.setattr(tools, "parse_cdo_sinfon_grid", lambda text: None)
    with pytest.raises(met_tc.METError):
        met_tc.regrid_for_tc_rmw(["wrfout_d01_2005-08-27_00:00:00"], str(tmp_path / "out.grb"))


def test_storm_relative_winds_cyclone_on_clockwise_from_east_grid():
    import numpy as np
    # a pure counterclockwise vortex of 50 m/s sampled on tc_rmw's azimuth grid
    az = np.arange(0, 360, 2.0)                    # degrees clockwise from east
    th = np.radians(-az)                           # math angle of each sample point
    u, v = -50 * np.sin(th), 50 * np.cos(th)       # counterclockwise tangent
    vt, vr = met_tc.storm_relative_winds(u, v, az)
    assert np.allclose(vt, 50) and np.allclose(vr, 0, atol=1e-9)
    # the Katrina sample that exposed tc_rmw's own VT/VR (850 hPa, 50 km, azimuth 80)
    vt80, vr80 = met_tc.storm_relative_winds(42.4, 21.4, 80.0)
    assert abs(vt80 - 45.5) < 0.2 and abs(vr80 - -13.7) < 0.2
