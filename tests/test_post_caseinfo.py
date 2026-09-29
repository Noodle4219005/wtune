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

from post import caseinfo

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


def test_domain_physics_reads_m1_real_namelist(m1_run_dir):
    max_dom, domains = caseinfo.domain_physics(m1_run_dir)
    assert max_dom == 2
    assert domains[0]["dx_km"] == 12.0
    assert domains[1]["dx_km"] == 4.0
    assert domains[0]["cu_physics"] == 1
    assert domains[1]["cu_physics"] == 0
    assert domains[0]["bl_pbl_physics"] == 1
    assert domains[1]["bl_pbl_physics"] == 1
    assert domains[0]["chem_opt"] == 0 and domains[1]["chem_opt"] == 0
    assert domains[0]["e_we"] == 170 and domains[1]["e_we"] == 244


def test_d01_extent_km_matches_r1_about_2000km(m1_run_dir):
    _, domains = caseinfo.domain_physics(m1_run_dir)
    we_km, sn_km = caseinfo.d01_extent_km(domains)
    assert 1900 < we_km < 2100          # decision 0057 R1: "d01 約 2000 km"
    assert 1500 < sn_km < 1600


def test_run_length_hours_from_namelist(m1_run_dir):
    assert caseinfo.run_length_hours(m1_run_dir) == 48.0


def test_r1_auto_detect_is_common_tc_convection_precip_synoptic(m1_run_dir, monkeypatch):
    """Decision 0057 R1: --types auto on m1 with --hurdat2/--storm given must detect exactly
    common+tc+convection+precip+synoptic -- not pbl/chem/climate/ensemble. The 2026-09-29
    update adds "moisture" to this list (decision 0057 update T2: "Katrina m1 判定出
    moisture") since moisture triggers on precip/convection/tc, all three of which this case
    already has; m1's namelist-only fixture (no real wrfout on disk) has no HGT to read, so
    terrain is correctly NOT detected here -- see test_post_recipe.py for a terrain-detected
    case built from a synthetic wrfout."""
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir], hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert info["types"] == ["common", "tc", "convection", "precip", "synoptic", "moisture"]
    assert "d02" in info["evidence"]["convection"] and "cu_physics=0" in info["evidence"]["convection"]
    assert "precip" in info["evidence"]["moisture"] or "tc" in info["evidence"]["moisture"]
    assert "1500" in info["evidence"]["synoptic"] or ">=" in info["evidence"]["synoptic"]


def test_tc_not_detected_without_hurdat2_or_bdeck(m1_run_dir, monkeypatch):
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir])
    assert "tc" not in info["types"]
    assert "not auto-detected" in info["evidence"]["tc"]


def test_tc_not_detected_when_track_outside_window(m1_run_dir, monkeypatch):
    """A best track that never falls inside d01 during the run window must not set tc, even
    with --hurdat2/--storm given (overlap rule cuts both ways)."""
    outside_window = (M1_D01_BBOX_WINDOW[0], M1_D01_BBOX_WINDOW[1], M1_D01_BBOX_WINDOW[2],
                      M1_D01_BBOX_WINDOW[3], datetime(1999, 1, 1), datetime(1999, 1, 2))
    _patch_bbox(monkeypatch, outside_window)
    info = caseinfo.detect_types([m1_run_dir], hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert "tc" not in info["types"]
    assert "no fix inside d01" in info["evidence"]["tc"]


def test_r4_types_override_is_exactly_common_and_tc(m1_run_dir, monkeypatch):
    """Decision 0057 R4: `--types tc` must yield only common+tc, even though convection/
    precip/synoptic would otherwise also be detected on this same case."""
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir], types_override="tc",
                                 hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert info["types"] == ["common", "tc"]


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


def test_pbl_chem_climate_not_detected_on_m1(m1_run_dir, monkeypatch):
    """m1 is bl_pbl_physics=1 on both domains (no LES), dx>=4km (no <=1km trigger),
    chem_opt=0, and 48h (<< 30 days) -- none of pbl/chem/climate should fire."""
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir], hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert "pbl" not in info["types"]
    assert "chem" not in info["types"]
    assert "climate" not in info["types"]


def test_types_always_ordered_per_all_types_order(m1_run_dir, monkeypatch):
    _patch_bbox(monkeypatch)
    info = caseinfo.detect_types([m1_run_dir], types_override="synoptic,tc,precip",
                                 hurdat2_path=HURDAT2_PATH, storm_id="AL122005")
    assert info["types"] == ["common", "tc", "precip", "synoptic"]


def test_detect_terrain_true_when_relief_exceeds_threshold(tmp_path):
    """decision 0057 update: relief >= 1000 m OR peak >= 1500 m -> terrain. A synthetic
    wrfout_d01_* with a real HGT variable (netCDF4) exercises detect_terrain()/
    _domain_hgt_stats() end to end without needing a real WRF case."""
    import netCDF4
    import numpy as np
    run_dir = tmp_path / "terrain_run"
    run_dir.mkdir()
    path = run_dir / "wrfout_d01_2005-08-27_00:00:00"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 3)
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        hgt[0] = np.array([[0.0, 0.0, 0.0], [0.0, 1800.0, 0.0], [0.0, 0.0, 0.0]], dtype="f4")

    # bdy_width=0: this test is about the relief/threshold rule itself, not the fourth lead
    # review's boundary-zone exclusion (which has its own dedicated tests below and needs a
    # namelist.input/output on disk that detect_terrain()'s default bdy_width lookup would
    # otherwise try, and fail, to read).
    ok, evidence, qualifying = caseinfo.detect_terrain(str(run_dir), 1, bdy_width=0)
    assert ok is True
    assert "d01" in evidence and "1800" in evidence
    assert qualifying == [1]


def test_detect_terrain_false_for_flat_domain(tmp_path):
    import netCDF4
    import numpy as np
    run_dir = tmp_path / "flat_run"
    run_dir.mkdir()
    path = run_dir / "wrfout_d01_2005-08-27_00:00:00"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        hgt[0] = np.full((2, 2), 10.0, dtype="f4")

    ok, evidence, qualifying = caseinfo.detect_terrain(str(run_dir), 1, bdy_width=0)
    assert ok is False
    assert qualifying == []
    assert "no domain" in evidence


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

def _make_wrfout_hgt(path, hgt_array):
    import netCDF4
    ny, nx = hgt_array.shape
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        var = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        var[0] = hgt_array.astype("f4")


def test_spec_bdy_width_reads_real_m1_namelist(m1_run_dir):
    # tests/fixtures/post/m1-namelist.input has "spec_bdy_width=5, spec_zone=1, relax_zone=4,"
    assert caseinfo.spec_bdy_width(m1_run_dir) == 5


def test_spec_bdy_width_defaults_to_5_when_bdy_control_is_absent(tmp_path):
    run_dir = tmp_path / "no_bdy_control"
    run_dir.mkdir()
    (run_dir / "namelist.input").write_text("&domains\n max_dom = 1,\n/\n")
    assert caseinfo.spec_bdy_width(str(run_dir)) == 5


def test_interior_bounds_excludes_bdy_width_on_every_edge():
    assert caseinfo._interior_bounds(15, 15, 5) == (5, 10, 5, 10)
    assert caseinfo._interior_bounds(9, 9, 5) is None   # nothing left after exclusion


def test_detect_terrain_ignores_relief_that_lives_only_in_the_boundary_zone():
    """A 15x15 domain where the only high point (3000 m, well past both thresholds) sits at
    row/col 1 -- inside the bdy_width=5 boundary zone -- while the interior (rows/cols 5..9)
    is flat. Must NOT be detected: the boundary-zone relief must never count."""
    import numpy as np
    hgt = np.zeros((15, 15))
    hgt[1, 1] = 3000.0   # inside the 5-cell boundary zone on every edge

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _make_wrfout_hgt(os.path.join(tmp, "wrfout_d01_2005-08-27_00:00:00"), hgt)
        ok, evidence, qualifying = caseinfo.detect_terrain(tmp, 1, bdy_width=5)
    assert ok is False
    assert qualifying == []
    assert "no domain" in evidence


def test_detect_terrain_uses_interior_relief_when_boundary_also_has_relief():
    """Same shape, but the INTERIOR also has real relief (1800 m at (7,7), well inside the
    boundary-excluded region) -- must be detected, and the reported max must be the
    interior's own 1800 m, not the boundary-zone's 3000 m (proving the exclusion actually
    changes the computed statistic, not just the pass/fail outcome)."""
    import numpy as np
    hgt = np.zeros((15, 15))
    hgt[1, 1] = 3000.0    # boundary zone -- must be excluded from the reported max
    hgt[7, 7] = 1800.0    # interior -- this is the real terrain

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _make_wrfout_hgt(os.path.join(tmp, "wrfout_d01_2005-08-27_00:00:00"), hgt)
        ok, evidence, qualifying = caseinfo.detect_terrain(tmp, 1, bdy_width=5)
    assert ok is True
    assert qualifying == [1]
    assert "1800" in evidence and "3000" not in evidence


def test_parse_namelist_keeps_group_past_unquoted_path_value(tmp_path):
    # namelist.output writes string defaults unquoted; a slash inside the value must not end
    # the group (it used to truncate &physics before cu_physics / bl_pbl_physics)
    import wtune_cli
    nl = tmp_path / "namelist.output"
    nl.write_text(" &physics\n MP_PHYSICS = 8, 8,\n PATH_TO_FILES = ~/WRFV3/,\n"
                  " CU_PHYSICS = 1, 0,\n BL_PBL_PHYSICS = 1, 1,\n /\n &domains\n MAX_DOM = 2, /\n")
    groups = wtune_cli.parse_namelist(str(nl))
    phys = {k.lower(): v for k, v in groups["physics"].items()}
    assert "cu_physics" in phys and "bl_pbl_physics" in phys
    assert "max_dom" in {k.lower() for k in groups["domains"]}
