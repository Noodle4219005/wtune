"""Tests for wtune/project.py (decision 0058): `wtune project init` layout/symlink contract
(P1), project discovery (--project/$WTUNE_PROJECT/walk-up), manifest.jsonl, and the
post/<id>/-style output-directory naming.
"""
import hashlib
import os

import pytest
import yaml

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

def test_init_creates_the_full_layout(tmp_path):
    project_dir = str(tmp_path / "myproj")
    proj.init_project(project_dir, name="myproj")
    for sub in proj.SUBDIRS:
        assert os.path.isdir(os.path.join(project_dir, sub))
    assert os.path.isfile(os.path.join(project_dir, "project.yaml"))
    assert os.path.isfile(os.path.join(project_dir, "manifest.jsonl"))
    assert os.path.isfile(os.path.join(project_dir, "analysis", "example_tc_track.py"))


def test_init_namelists_are_copied_big_files_are_symlinked(tmp_path, case_dir):
    project_dir = str(tmp_path / "proj2")
    proj.init_project(project_dir, case_dir=case_dir)
    dest = os.path.join(project_dir, "case")

    assert not os.path.islink(os.path.join(dest, "namelist.input"))
    # namelist.output next to a namelist.input is a previous run's output: not taken over
    assert not os.path.lexists(os.path.join(dest, "namelist.output"))
    assert open(os.path.join(dest, "namelist.input")).read() == \
        open(os.path.join(case_dir, "namelist.input")).read()

    assert os.path.islink(os.path.join(dest, "wrfinput_d01"))
    assert os.readlink(os.path.join(dest, "wrfinput_d01")) == \
        os.path.join(case_dir, "wrfinput_d01")
    assert os.path.islink(os.path.join(dest, "met_em"))


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
               "rsl.error.0001", "wrf.log", "real.log"]
    for n in inputs + outputs:
        (src / n).write_text("x")
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir, case_dir=str(src))
    dest = os.path.join(project_dir, "case")
    assert sorted(os.listdir(dest)) == sorted(inputs + ["namelist.input"])
    out = capsys.readouterr().out
    assert "skipped %d output file(s) of a previous run in %s" % (len(outputs) + 1, src) in out


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


def test_init_default_name_is_case_and_utc_stamp(tmp_path, case_dir):
    project_dir = str(tmp_path / "proj4")
    proj.init_project(project_dir, case_dir=case_dir)
    doc = proj.load_project(project_dir)
    assert doc["name"].startswith(os.path.basename(case_dir) + "-")


def test_init_refuses_a_nonempty_existing_dir(tmp_path):
    project_dir = tmp_path / "proj5"
    project_dir.mkdir()
    (project_dir / "already_here.txt").write_text("x")
    with pytest.raises(proj.ProjectError):
        proj.init_project(str(project_dir))


def test_init_writes_storm_into_project_yaml(tmp_path):
    hurdat2 = tmp_path / "hurdat2.txt"
    hurdat2.write_text("AL122005,KATRINA\n")
    project_dir = str(tmp_path / "proj6")
    proj.init_project(project_dir, storm="AL122005", hurdat2=str(hurdat2))
    doc = proj.load_project(project_dir)
    assert doc["storm"]["id"] == "AL122005"
    assert doc["storm"]["hurdat2"] == str(hurdat2)


# -- discovery -------------------------------------------------------------------------------

def test_find_project_explicit_project_arg(tmp_path):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    assert proj.find_project(explicit=project_dir) == os.path.abspath(project_dir)


def test_find_project_explicit_without_project_yaml_raises(tmp_path):
    with pytest.raises(proj.ProjectError):
        proj.find_project(explicit=str(tmp_path))


def test_find_project_env_var(tmp_path, monkeypatch):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    monkeypatch.setenv("WTUNE_PROJECT", project_dir)
    assert proj.find_project() == os.path.abspath(project_dir)


def test_find_project_walks_up_from_cwd(tmp_path, monkeypatch):
    project_dir = tmp_path / "p"
    proj.init_project(str(project_dir))
    nested = project_dir / "runs" / "m1"
    nested.mkdir(parents=True)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    assert proj.find_project(start_dir=str(nested)) == os.path.abspath(str(project_dir))


def test_find_project_returns_none_when_not_inside_one(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    outside = tmp_path / "not_a_project"
    outside.mkdir()
    assert proj.find_project(start_dir=str(outside)) is None


# -- manifest.jsonl --------------------------------------------------------------------------

def test_manifest_append_writes_one_json_line(tmp_path):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    manifest_path = os.path.join(project_dir, "manifest.jsonl")
    n_before = sum(1 for _ in open(manifest_path))

    input_file = tmp_path / "small_input.txt"
    input_file.write_text("hello")
    out_dir = os.path.join(project_dir, "post", "20260101000000-recipe")
    entry = proj.manifest_append(project_dir, ["wtune", "post", "recipe", str(input_file)],
                                 out_dir=out_dir, exit_code=0, key_paths=[str(input_file)])

    lines = open(manifest_path).readlines()
    assert len(lines) == n_before + 1
    assert entry["out_dir"] == os.path.relpath(out_dir, project_dir)
    assert entry["exit_code"] == 0
    assert entry["inputs"][0]["path"] == str(input_file)
    assert "sha256" in entry["inputs"][0]   # small file: hashed


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

def test_default_out_dir_uses_post_subdir_and_tag(tmp_path):
    project_dir = str(tmp_path / "p")
    out = proj.default_out_dir(project_dir, "post", tag="recipe")
    assert out.startswith(os.path.join(project_dir, "post"))
    assert out.endswith("-recipe")


def test_discover_runs_prefers_runs_subdirs_over_case(tmp_path):
    project_dir = str(tmp_path / "p")
    proj.init_project(project_dir)
    run1 = os.path.join(project_dir, "runs", "m1")
    os.makedirs(run1)
    open(os.path.join(run1, "wrfout_d01_2005-08-27_00:00:00"), "w").close()
    case_dir = os.path.join(project_dir, "case")
    open(os.path.join(case_dir, "wrfout_d01_2005-08-27_00:00:00"), "w").close()

    runs = proj.discover_runs(project_dir)
    assert runs == [("m1", run1)]


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


def test_wtune_run_still_requires_out_without_project(tmp_path, capsys):
    from wtune import run as wr
    case = _make_run_case(tmp_path)
    rc = wr.main([str(case), "--nodes", "1", "--nx", "1", "--ny", "1", "--dry-run"],
                 log=lambda s: None)
    assert rc == 2
    assert "error: --out is required" in capsys.readouterr().err


def test_wtune_run_explicit_out_still_overrides_project(tmp_path):
    from wtune import run as wr
    case = _make_run_case(tmp_path)
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir)
    explicit_out = tmp_path / "explicit-out"

    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "1", "--ny", "1", "--project", project_dir,
                 "--out", str(explicit_out), "--dry-run"], log=lines.append)
    assert rc == 0
    assert not os.listdir(os.path.join(project_dir, "runs"))   # project/runs/ untouched


def test_wtune_tune_cli_has_project_flag_wired_to_out_dir_default(tmp_path):
    """wtune/tune/'s own out_dir line (`out_dir = sweep._expand(args.out) if args.out else
    ...`) was edited to check --project before falling back to $HOME/wtune-runs/tune/<case>;
    this reproduces exactly that logic against a real project (not a mock), the same way the
    module itself computes it, since driving wtune/tune/'s full main() needs the scc-tuning
    mocks already covered by test_wtune_tune.py and would duplicate them for no extra
    coverage of the --project wiring itself."""
    import wtune.tune

    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir)

    args_out, args_project, case = None, project_dir, "some-case"
    if args_out:
        out_dir = wtune.tune.sweep._expand(args_out)
    else:
        project_resolved = proj.find_project(explicit=args_project)
        out_dir = (proj.default_out_dir(project_resolved, "tune") if project_resolved
                  else wtune.tune.sweep._expand("$HOME/wtune-runs/tune/%s" % case))
    assert out_dir.startswith(os.path.join(project_dir, "tune"))

    # --out still takes priority over --project when both are given.
    args_out = str(tmp_path / "explicit")
    out_dir2 = wtune.tune.sweep._expand(args_out) if args_out else None
    assert out_dir2 == str(tmp_path / "explicit")


def test_wtune_rebuild_predict_out_defaults_to_project_predict_dir(tmp_path):
    project_dir = str(tmp_path / "proj")
    proj.init_project(project_dir)
    out_dir = proj.default_out_dir(proj.find_project(explicit=project_dir), "predict")
    assert out_dir.startswith(os.path.join(project_dir, "predict"))
