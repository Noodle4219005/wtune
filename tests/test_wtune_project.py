"""Tests for wtune/project.py (decision 0058): `wtune project init` layout/symlink contract
(P1), project discovery (--project/$WTUNE_PROJECT/walk-up), manifest.jsonl, and the
post/<id>/-style output-directory naming.
"""
import hashlib
import os

import pytest

from wtune import project as proj


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


@pytest.fixture
def case_dir(tmp_path):
    d = tmp_path / "case-src"
    d.mkdir()
    (d / "namelist.input").write_text("&time_control\nrun_hours = 6,\n/\n")
    (d / "namelist.output").write_text("&time_control\nrun_hours = 6,\n/\n")
    (d / "wrfinput_d01").write_bytes(os.urandom(4096))
    sub = d / "met_em"
    sub.mkdir()
    (sub / "met_em.d01.nc").write_bytes(os.urandom(64))
    return str(d)


# -- P1: init layout, symlinks vs. copies, source untouched --------------------------------


def test_init_skips_outputs_of_a_previous_run(tmp_path, capsys):
    """climate-ea50: --case was a former run dir; 397 old wrfout files got linked into case/."""
    src = tmp_path / "former-run"
    src.mkdir()
    (src / "namelist.input").write_text("&time_control\n/\n")
    (src / "namelist.output").write_text("&time_control\n/\n")
    inputs = ["wrfinput_d01", "wrfbdy_d01", "wrflowinp_d01", "wrffdda_d01", "iofields_d01.txt",
              "met_em.d01.2020-01-01_00:00:00.nc", "LANDUSE.TBL", "namelist.wps", "tslist"]
    outputs = ["wrfout_d01_2020-01-01_00:00:00", "wrfout_d01_2020-01-02_00:00:00",
               "wrfrst_d01_2020-01-02_00:00:00", "rsl.out.0000", "rsl.error.0000",
               "rsl.error.0001", "wrf.log", "real.log", "wrf.out", "sd1.d02.TS",
               "rap.d02.all.TS", "sd1.d01.UU"]
    executables = ["wrf.exe", "real.exe"]         # the WRF that former run used
    for n in inputs + outputs + executables:
        (src / n).write_text("x")
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir, case_dir=str(src))
    dest = os.path.join(project_dir, "case")
    assert sorted(os.listdir(dest)) == sorted(inputs + ["namelist.input"])
    out = capsys.readouterr().out
    assert "skipped %d output file(s) of a previous run in %s" % (len(outputs) + 1, src) in out
    assert "skipped 2 executable(s) in %s: the project runs the machine's WRF" % src in out


def test_init_keeps_namelist_output_when_it_is_the_only_namelist(tmp_path, capsys):
    src = tmp_path / "nml-out-only"
    src.mkdir()
    (src / "namelist.output").write_text("&time_control\n/\n")
    (src / "wrfinput_d01").write_text("x")
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir, case_dir=str(src))
    assert sorted(os.listdir(os.path.join(project_dir, "case"))) == ["namelist.output",
                                                                      "wrfinput_d01"]
    assert "skipped" not in capsys.readouterr().out


def test_discover_runs_falls_back_to_case_source_dir_wrfout(tmp_path):
    """init no longer links a former run's wrfout into case/: 'case' then points at the source."""
    src = tmp_path / "former-run"
    src.mkdir()
    (src / "namelist.input").write_text("&time_control\n/\n")
    (src / "wrfout_d01_2020-01-01_00:00:00").write_text("x")
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir, case_dir=str(src))
    assert proj.discover_runs(project_dir) == [("case", str(src))]


def test_init_never_modifies_the_source_case_dir(tmp_path, case_dir):
    before_sha = _sha256(os.path.join(case_dir, "wrfinput_d01"))
    before_mtime = os.stat(os.path.join(case_dir, "namelist.input")).st_mtime
    before_entries = sorted(os.listdir(case_dir))

    proj.init_project(str(tmp_path / "proj3"), case_dir=case_dir)

    assert _sha256(os.path.join(case_dir, "wrfinput_d01")) == before_sha
    assert os.stat(os.path.join(case_dir, "namelist.input")).st_mtime == before_mtime
    assert sorted(os.listdir(case_dir)) == before_entries


def test_init_refuses_a_nonempty_existing_dir(tmp_path):
    project_dir = tmp_path / "proj5"
    project_dir.mkdir()
    (project_dir / "already_here.txt").write_text("x")
    with pytest.raises(proj.ProjectError):
        proj.init_project(str(project_dir))


# -- discovery -------------------------------------------------------------------------------


# -- manifest.jsonl --------------------------------------------------------------------------


def test_manifest_skips_sha256_for_large_files(tmp_path, monkeypatch):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    monkeypatch.setattr(proj, "MANIFEST_HASH_LIMIT", 10)   # force "large" for a tiny file
    big_file = tmp_path / "big.bin"
    big_file.write_bytes(b"x" * 100)
    entry = proj.manifest_append(project_dir, ["wtune"], key_paths=[str(big_file)])
    assert "sha256" not in entry["inputs"][0]
    assert entry["inputs"][0]["size_bytes"] == 100


# -- output-directory naming -----------------------------------------------------------------


def test_discover_runs_falls_back_to_case_dir(tmp_path):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    case_dir = os.path.join(project_dir, "case")
    open(os.path.join(case_dir, "wrfout_d01_2005-08-27_00:00:00"), "w").close()

    runs = proj.discover_runs(project_dir)
    assert runs == [("case", case_dir)]


# -- --project wiring in tune / run / predict-rebuild (decision 0058 item 2: "only add
# --project default output directories ... where they already take an --out; do not change
# their logic") ---------------------------------------------------------------------------

_NAMELIST_1DOM = """&time_control
 run_days = 0,
 run_hours = 6,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2020,
 start_month = 06,
 start_day = 01,
 start_hour = 00,
 end_year = 2020,
 end_month = 06,
 end_day = 01,
 end_hour = 06,
 interval_seconds = 21600
 input_from_file = .true.,
 history_interval = 60,
 frames_per_outfile = 1,
 restart = .false.,
 restart_interval = 7200,
 io_form_history = 2
 io_form_restart = 2
 io_form_input = 2
 io_form_boundary = 2
/
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 45,
 parent_id = 1,
 parent_grid_ratio = 1,
 parent_time_step_ratio = 1,
/
&physics
 radt = 30,
/
"""


def _make_run_case(tmp_path, name="case"):
    case = tmp_path / name
    case.mkdir()
    (case / "namelist.input").write_text(_NAMELIST_1DOM)
    for f in ("wrfinput_d01", "wrfbdy_d01", "namelist.output"):
        (case / f).write_text("x")
    return case


def test_wtune_run_defaults_out_to_project_runs_dir_when_out_omitted(tmp_path):
    from wtune import run as wr
    case = _make_run_case(tmp_path)
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir)

    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "1", "--ny", "1", "--project", project_dir,
                 "--dry-run"], log=lines.append)
    # rc==0 (not the "--out is required" RunError) is the actual coverage here: --project
    # alone (no --out) resolved successfully. --dry-run itself never creates --out on disk
    # (see test_wtune_run.py's own test_dry_run_submits_nothing), so this cannot also assert
    # the runs/<id>/ directory exists.
    assert rc == 0


def test_project_set_post_flags_round_trip_and_show_in_info(tmp_path):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    doc = proj.set_project_defaults(
        project_dir, post_options=dict(post_times="2019-01-30T06", post_map_domains="all",
                                       post_station_method="nearest", post_calm_kt="2.5"))
    post = proj.project_post_defaults(doc)
    assert post["times"] == ["2019-01-30T06"] and post["map_domains"] == ["all"]
    assert post["station_method"] == "nearest" and post["calm_kt"] == 2.5
    assert "times=['2019-01-30T06']" in proj.info_project(project_dir)
    with pytest.raises(proj.ProjectError, match="--post-times"):
        proj.set_project_defaults(project_dir, post_options=dict(post_times="noon"))
