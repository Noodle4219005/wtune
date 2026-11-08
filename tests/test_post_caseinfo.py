"""Tests for post/caseinfo.py: multi-label case-type detection (decision 0057 section 1)
against P2 Katrina m1's real namelist.input (tests/fixtures/post/m1-namelist.input, copied
verbatim from ~/wtune-runs/p2-nest/m1/namelist.input on scc, 2026-09-29: 12->4 km nest,
d02 cu_physics=0, run_hours=48) and a HURDAT2 excerpt for AL122005 covering that run's window
(tests/fixtures/post/al122005-hurdat2-excerpt.txt, extracted from the same run's
~/wtune-runs/p2-verify/hurdat2.txt -- the full storm's 34-fix HURDAT2 block, header included).

detect_tc()'s own _d01_bbox_and_window() needs a real wrfout to read XLAT/XLONG from; it is
monkeypatched here to the real m1 d01 bbox/window (read once, by hand, from the actual copied
m1 wrfout_d01_2005-08-27_00:00:00 via netCDF4 -- not re-derived by this test), so these tests
stay netCDF-free and fast.
"""
import os
import shutil
from datetime import datetime

import pytest

from wtune.post import caseinfo

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "post")
NAMELIST_SRC = os.path.join(FIXTURE_DIR, "m1-namelist.input")
HURDAT2_PATH = os.path.join(FIXTURE_DIR, "al122005-hurdat2-excerpt.txt")

# m1's real d01 bbox (netCDF4 XLAT/XLONG min/max of wrfout_d01_2005-08-27_00:00:00) and its
# full 48 h run window (namelist.input: start 2005-08-27_00, run_hours=48).
M1_D01_BBOX_WINDOW = (17.912153244018555, 31.701738357543945,
                     -99.00386810302734, -78.99613189697266,
                     datetime(2005, 8, 27, 0, 0, 0), datetime(2005, 8, 29, 0, 0, 0))


@pytest.fixture
def m1_run_dir(tmp_path):
    d = tmp_path / "m1"
    d.mkdir()
    shutil.copy(NAMELIST_SRC, d / "namelist.input")
    return str(d)


def _patch_bbox(monkeypatch, bbox_window=M1_D01_BBOX_WINDOW):
    monkeypatch.setattr(caseinfo, "_d01_bbox_and_window", lambda run_dir: bbox_window)


def test_types_override_rejects_unknown_type(m1_run_dir, monkeypatch):
    _patch_bbox(monkeypatch)
    with pytest.raises(caseinfo.CaseInfoError, match="unknown"):
        caseinfo.detect_types([m1_run_dir], types_override="not_a_real_type")


def test_ensemble_requires_flag_and_multiple_run_dirs(m1_run_dir, monkeypatch):
    _patch_bbox(monkeypatch)
    info_single = caseinfo.detect_types([m1_run_dir], ensemble=True)
    assert "ensemble" not in info_single["types"]
    info_multi = caseinfo.detect_types([m1_run_dir, m1_run_dir], ensemble=True)
    assert "ensemble" in info_multi["types"]
    info_multi_no_flag = caseinfo.detect_types([m1_run_dir, m1_run_dir], ensemble=False)
    assert "ensemble" not in info_multi_no_flag["types"]


def test_types_always_ordered_per_all_types_order(m1_run_dir, monkeypatch):
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir], types_override="synoptic,tc,precip",
                                 hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert info["types"] == ["common", "tc", "precip", "synoptic"]


def test_detect_terrain_two_domain_only_d01_qualifies(tmp_path):
    """2026-09-29 third lead review: katrina m1's real case is exactly this shape -- d01
    (the coarse parent) covers real relief (e.g. Mexico), d02 (a nest entirely over the Gulf
    of Mexico) does not. qualifying_domains must name only d01, not both, so callers can
    target the finest domain that ITSELF has the terrain (post/recipes/terrain.py's
    _terrain_domain()), not blindly the run's overall finest domain."""
    import netCDF4
    import numpy as np
    run_dir = tmp_path / "two_domain_run"
    run_dir.mkdir()

    with netCDF4.Dataset(str(run_dir / "wrfout_d01_2005-08-27_00:00:00"), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 3)
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        hgt[0] = np.array([[0.0, 0.0, 0.0], [0.0, 1800.0, 0.0], [0.0, 0.0, 0.0]], dtype="f4")

    with netCDF4.Dataset(str(run_dir / "wrfout_d02_2005-08-27_00:00:00"), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 3)
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        hgt[0] = np.full((3, 3), 5.0, dtype="f4")   # flat -- the Gulf-of-Mexico nest

    ok, evidence, qualifying = caseinfo.detect_terrain(str(run_dir), 2, bdy_width=0)
    assert ok is True
    assert qualifying == [1]
    assert "d01" in evidence and "d02" not in evidence


# -- boundary-zone exclusion (2026-09-29 fourth lead review): the crest in WRF's own lateral
# boundary zone (spec_bdy_width cells on every edge, where the solution is specified/relaxed
# toward the driving data, not the model's own dynamics) must never be picked as "the
# terrain" -- katrina m1's real d01 "peak" (3411 m) turned out to sit exactly there. -------


def test_interior_bounds_excludes_bdy_width_on_every_edge():
    assert caseinfo._interior_bounds(15, 15, 5) == (5, 10, 5, 10)
    assert caseinfo._interior_bounds(9, 9, 5) is None   # nothing left after exclusion
