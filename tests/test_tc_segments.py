"""Segment-aware TC post-processing (2026-10-02): post/tc.py frame lookups, project
discover_runs, and analysis.tc_track._find_binary's pixi-bin fallback.

A segmented `wtune run --out OUT` holds OUT/seg_<k>/wrfout_*; the shared boundary frame exists
in two segments and must resolve to the later one (post.extract.frame_index).
"""
import os

import netCDF4
import numpy as np
import pytest

from wtune import project as proj
from analysis import tc_track
from post import tc, tools

T = ["2005-08-27_00:00:00", "2005-08-27_06:00:00", "2005-08-27_12:00:00", "2005-08-27_18:00:00"]


def _nc(path, dx=12000.0, lat0=20.0):
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        ds.setncattr("DX", dx)
        for name, base in (("XLAT", lat0), ("XLONG", -90.0)):
            v = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[:] = base + np.zeros((1, 3, 4))


def _segmented(tmp_path, frames=True):
    """OUT/seg_0: T0,T1  OUT/seg_1: T1,T2  OUT/seg_10: T3 (seg_10 sorts after seg_1 by number)."""
    out = tmp_path / "runs" / "o2"
    layout = {"seg_0": T[0:2], "seg_1": T[1:3], "seg_10": T[3:4]}
    for seg, times in layout.items():
        d = out / seg
        d.mkdir(parents=True)
        if frames:
            for t in times:
                _nc(d / ("wrfout_d01_%s" % t))
            (d / ("wrfout_d02_%s" % times[0])).write_text("")
    return out


# ------------------------------------------------------------------ post/tc.py
def test_run_start_and_max_dom_on_segments(tmp_path):
    out = _segmented(tmp_path)
    assert tc._run_start(str(out)).strftime(tc.WRF_TIME_FMT) == T[0]
    assert tc._detect_max_dom(str(out)) == 2


def test_tracks_for_domain_gets_one_frame_per_time_from_segments(tmp_path, monkeypatch):
    out = _segmented(tmp_path)
    from analysis import diagnostics
    seen = {}

    class Stop(Exception):
        pass

    def fake(paths, out_nc, *a, **k):
        seen["paths"] = paths
        raise Stop

    monkeypatch.setattr(diagnostics, "wrfout_to_tc_fields", fake)
    with pytest.raises(Stop):
        tc._tracks_for_domain(str(out), 1, str(tmp_path / "work"))
    assert seen["paths"] == [str(out / "seg_0" / ("wrfout_d01_%s" % T[0])),
                             str(out / "seg_1" / ("wrfout_d01_%s" % T[1])),   # later segment wins
                             str(out / "seg_1" / ("wrfout_d01_%s" % T[2])),
                             str(out / "seg_10" / ("wrfout_d01_%s" % T[3]))]
    # the list form is accepted by diagnostics' own expander unchanged
    assert diagnostics.expand_wrfout(seen["paths"]) == sorted(seen["paths"])


def test_run_domains_and_dx_on_segments(tmp_path):
    out = _segmented(tmp_path)
    _nc(out / "seg_10" / ("wrfout_d01_%s" % T[3]), lat0=21.0)   # d01 "moved" between first/last
    doms = tc.run_domains(str(out), max_dom=1)
    assert list(doms) == [1] and sorted(doms[1]["frames"]) == T
    assert doms[1]["frames"][T[1]].endswith(os.path.join("seg_1", "wrfout_d01_%s" % T[1]))
    assert doms[1]["dx_km"] == 12.0 and (doms[1]["ny"], doms[1]["nx"]) == (3, 4)
    assert doms[1]["moving"] is True
    assert tc._domain_dx_km(str(out), 1) == 12.0


def test_flat_run_dir_unchanged(tmp_path):
    d = tmp_path / "flat"
    d.mkdir()
    for t in T[:2]:
        _nc(d / ("wrfout_d01_%s" % t), dx=4000.0)
    assert tc._run_start(str(d)).strftime(tc.WRF_TIME_FMT) == T[0]
    assert tc._detect_max_dom(str(d)) == 1
    doms = tc.run_domains(str(d))
    assert doms[1]["frames"] == {t: str(d / ("wrfout_d01_%s" % t)) for t in T[:2]}
    assert doms[1]["moving"] is False and tc._domain_dx_km(str(d), 1) == 4.0


def test_no_frames_raises(tmp_path):
    out = _segmented(tmp_path, frames=False)
    with pytest.raises(tc.TCError, match="no wrfout_d01_"):
        tc._run_start(str(out))
    with pytest.raises(tc.TCError, match="no wrfout_dNN_"):
        tc._detect_max_dom(str(out))


# ------------------------------------------------------------------ discover_runs
def test_discover_runs_lists_segmented_run_once(tmp_path):
    p = proj.init_project(str(tmp_path / "proj"))
    _segmented(tmp_path / "proj")                       # proj/runs/o2/seg_*
    flat = tmp_path / "proj" / "runs" / "flat"
    flat.mkdir()
    (flat / ("wrfout_d01_%s" % T[0])).write_text("")
    empty = tmp_path / "proj" / "runs" / "empty" / "seg_0"
    empty.mkdir(parents=True)                           # segments without wrfout: not a run
    assert proj.discover_runs(p) == [("flat", str(flat)),
                                     ("o2", str(tmp_path / "proj" / "runs" / "o2"))]


# ------------------------------------------------------------------ _find_binary
def test_find_binary_falls_back_to_pixi_env_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools, "REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("WTUNE_POSTPROC_VIEW", raising=False)
    with pytest.raises(tc_track.TCTrackError) as e:
        tc_track._find_binary("DetectNodes")
    msg = str(e.value)
    assert "PATH" in msg and os.path.join("envs", "wtune", ".pixi", "envs", "default", "bin",
                                          "DetectNodes") in msg
    assert "envs/wtune/pixi.toml" in msg
    bindir = tmp_path / "envs" / "wtune" / ".pixi" / "envs" / "default" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "DetectNodes").write_text("#!/bin/sh\n")
    assert tc_track._find_binary("DetectNodes") == str(bindir / "DetectNodes")


def test_find_binary_prefers_path(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    assert tc_track._find_binary("StitchNodes") == "/usr/bin/StitchNodes"
