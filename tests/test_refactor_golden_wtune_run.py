#!/usr/bin/env python3
"""Golden snapshots of `wtune run` (wtune_run), pinned before the round-3 package split: what
main() prints (log, stdout, stderr, exit code), the OUT tree, every segment namelist /
segment.sbatch / adios2.xml and run_report.json (key order included), plus the pure planners'
results and error texts. Fakes are test_wtune_run's (no job, no adios2); the tmp path, the repo
path and the UTC stamps are normalised.

Regenerate (only with the UNCHANGED code): WTUNE_GOLDEN_REGEN=1 pytest tests/<this file>.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys

import pytest

import test_wtune_run as T          # first: its wtune_run import restores SITE/configs
import test_wtune_extensions as TE
import test_wtune_run_resume as TR
from wtune import run as wr
from test_wtune_run import fake_bp  # noqa: F401  (autouse: the fake src/wtune/bp.py)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
GOLDEN_DIR = os.path.join(HERE, "fixtures", "golden", "wtune_run")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
_UTC_RE = re.compile(r'("(?:submitted|started|finished)_utc": )"[^"]*"')
_TEXT_FILES = ("namelist.input", "segment.sbatch", "adios2.xml", "segment.advice.namelist.input")
LAYOUT = ["--nodes", "1", "--nx", "8", "--ny", "9"]
ALIGN = {"steps": 360, "seconds": 21600.0, "label": "interval_seconds 06:00:00"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No node exclusions, no WRF tree / project / post cache from the environment."""
    import scc_sweep
    for var in ("WTUNE_SBATCH_EXCLUDE", scc_sweep.PREFLIGHT_EXCLUDE_ENV, "WRF_TREE",
                "WTUNE_NC_CACHE", "SLURM_JOB_ID"):   # inside a Slurm job the copy guard warns
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.chdir(tmp_path)


def _check(name, text):
    path = os.path.join(GOLDEN_DIR, name)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "output differs from golden %s" % path


def _norm(text, tmp_path):
    return _UTC_RE.sub(r'\1"<UTC>"', text.replace(str(tmp_path), "<TMP>").replace(REPO, "<REPO>"))


def _plain(x):
    """A record (dict today, a dataclass with to_dict() after the split) as plain JSON data."""
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _dump(x):
    return json.dumps(_plain(x), indent=1, default=str)


def _tree(root):
    """One line per entry below root: 'd name/', 'f name', 'l name -> target'."""
    out = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for n in sorted(dns + fns):
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out.append("l %s -> %s" % (rel, os.readlink(p)))
            elif os.path.isdir(p):
                out.append("d %s/" % rel)
            else:
                out.append("f %s" % rel)
    return sorted(out)


def _snapshot(tmp_path, rc, lines, capsys, out):
    cap = capsys.readouterr()
    parts = ["rc=%s" % rc, "--- log"] + list(lines) + ["--- stdout", cap.out, "--- stderr", cap.err]
    if os.path.isdir(out):
        parts += ["--- tree"] + _tree(out)
        for d in sorted(os.listdir(out)):
            for name in _TEXT_FILES:
                p = os.path.join(out, d, name)
                if os.path.isfile(p) and not os.path.islink(p):
                    with open(p) as fh:
                        parts += ["--- %s/%s" % (d, name), fh.read()]
        rep = os.path.join(out, "run_report.json")
        if os.path.isfile(rep):
            with open(rep) as fh:
                parts += ["--- run_report.json", fh.read()]
    return _norm("\n".join(parts) + "\n", tmp_path)


def _main(tmp_path, argv, fakes=None, layout=LAYOUT, **kw):
    f = fakes or T.Fakes()
    lines = []
    try:
        rc = wr.main([str(tmp_path / "case")] + list(layout) + ["--out", str(tmp_path / "out"),
                                                                 "--poll-interval", "0"]
                     + list(argv), sbatch_runner=f.sbatch, sacct_runner=f.sacct,
                     snapshot_fn=f.snapshot, sleep=lambda s: None, log=lines.append, **kw)
    except Exception as e:      # noqa: BLE001  an uncaught error is part of the behaviour
        rc = "%s: %s" % (type(e).__name__, e)
    return rc, lines


# ---------------------------------------------------------------------------
# `wtune run` end to end
# ---------------------------------------------------------------------------
PRED = {"hardware": {"nx": 8, "ny": 9, "omp": 1}, "nodes": 1,
        "result": {"points": [{"at": "2020-06-01T03:00:00", "s_normal": 0.5},
                              {"at": "2020-06-01T09:00:00", "s_normal": 0.7}],
                   "r_night": 2.0, "r_day": 4.0, "init_s": 60.0},
        "io": {"history": [{"domain": 1, "median_s": 5.0}, {"domain": 2, "median_s": 3.0},
                           {"domain": 3, "median_s": 2.0}],
               "restart": {"per_write_s": 40.0}}}
WINNER = {"cid": "ANCHOR", "cores_per_node": 72, "median": 0.5, "nodes": 1, "numtiles": 2,
          "nx": 8, "ny": 9, "omp": 1, "ranks": 72, "reps": 3, "source": "t0 sweep ANCHOR"}
T0S, T1S = "2020-06-01T00:00:00", "2020-06-01T12:00:00"
RANKING = [
    {"id": "A", "nproc_x": 8, "nproc_y": 9, "numtiles": 1, "omp": 1, "dist": "block",
     "points": [{"at": T0S, "s_normal": 1.0, "r": 2.0, "n_rad": 2},
                {"at": T1S, "s_normal": 3.0, "r": None, "n_rad": 0}]},
    {"id": "B", "nproc_x": 6, "nproc_y": 12, "numtiles": 2, "omp": 1, "dist": "cyclic",
     "points": [{"at": T0S, "s_normal": 2.0, "r": None, "n_rad": 0}]},
    {"id": "C", "nproc_x": 4, "nproc_y": 4, "numtiles": 1, "omp": 1, "points": []}]
SEGFAULT = "forrtl: severe (174): SIGSEGV, segmentation fault occurred\n"
AUX_FILES = ("wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03", "wrffdda_d01")
IOFIELDS = T.NAMELIST_3DOM.replace(" io_form_history                     = 2",
                                   ' iofields_filename = "iofields_d01.txt",\n'
                                   " io_form_history                     = 2")
STAGGERED = T.NAMELIST_3DOM.replace(
    " start_hour                          = 00,   00,   00,",
    " start_hour                          = 00,   03,   00,")


def _json_file(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def _tune_report(tmp_path, **kw):
    rep = {"nodes": 1, "ranking": [], "winner": dict(WINNER),
           "node_selection": {"chosen": 1, "chosen_note": "x", "layouts": [dict(WINNER)]}}
    rep.update(kw)
    return _json_file(tmp_path, "tune_report.json", rep)


class NoSacct(T.Fakes):
    """sacct never lists the job: wait_for_job gives up with UNKNOWN and no elapsed time."""
    def sacct(self, cmd):
        return ""


# name -> dict(text: case namelist, files: extra case files, argv, argv_fn(tmp_path) -> argv,
#              fakes() -> the fake cluster, layout: the --nodes/--nx/--ny flags, pre(tmp_path,
#              fake_bp): extra setup)
RUNS = {
    "one_segment_auto_checkpoint": dict(argv=[]),
    "one_segment_resume_0": dict(argv=["--resume", "0"]),
    "one_segment_case_restart_kept": dict(text=T._with_restart(180), argv=[]),
    "two_segments": dict(argv=["--segment-minutes", "360"]),
    "two_segments_netcdf": dict(argv=["--segment-minutes", "360", "--io", "netcdf"]),
    "two_segments_netcdf_restart_form_2": dict(
        argv=["--segment-minutes", "360", "--io", "netcdf", "--restart-io-form", "2"]),
    "two_segments_restart_form_2": dict(argv=["--segment-minutes", "360",
                                              "--restart-io-form", "2"]),
    "two_segments_case_restart_divides": dict(text=T._with_restart(120),
                                              argv=["--segment-minutes", "360"]),
    "three_segments_stop_at_netcdf": dict(
        argv=["--segment-minutes", "240", "--io", "netcdf", "--stop-at", "2020-06-01_10:00:00"]),
    "wall_limit_netcdf": dict(argv=["--wall-limit", "06:00:00", "--io", "netcdf",
                                    "--omp", "2", "--numtiles", "3", "--nx", "4"]),
    "aux_streams": dict(text=T.NAMELIST_AUX, files=AUX_FILES, argv=["--segment-minutes", "360"]),
    "aux_streams_missing_files_dry": dict(text=T.NAMELIST_AUX, argv=["--dry-run"]),
    "iofields_missing_dry": dict(text=IOFIELDS, argv=["--dry-run"]),
    "dry_run": dict(argv=["--wall-limit", "02:00:00", "--segment-minutes", "480", "--dry-run"]),
    "dry_run_netcdf": dict(argv=["--segment-minutes", "180", "--io", "netcdf", "--dry-run"]),
    "dry_run_checkpoint_0": dict(argv=["--checkpoint-minutes", "0", "--dry-run"]),
    "dry_run_prediction_wall_limit": dict(
        argv_fn=lambda p: ["--wall-limit", "00:10:00", "--prediction",
                           _json_file(p, "pred_rebuild.json", PRED), "--dry-run"]),
    "prediction_other_config": dict(
        argv_fn=lambda p: ["--segment-minutes", "360", "--nx", "4", "--prediction",
                           _json_file(p, "pred_rebuild.json", PRED)]),
    "checkpoint_explicit_netcdf": dict(argv=["--io", "netcdf", "--checkpoint-minutes", "120"]),
    "checkpoint_explicit_segments": dict(argv=["--io", "netcdf", "--segment-minutes", "360",
                                               "--checkpoint-minutes", "90"]),
    "tune_report_rate_only": dict(
        layout=[], argv_fn=lambda p: ["--tune-report", _tune_report(p),
                                      "--segment-minutes", "360"]),
    "tune_report_cli_override_dry": dict(
        layout=["--nx", "6", "--ny", "12"],
        argv_fn=lambda p: ["--tune-report", _tune_report(p), "--dry-run"]),
    "tune_report_ranking": dict(
        layout=[], argv_fn=lambda p: ["--tune-report", _tune_report(p, ranking=RANKING),
                                      "--segment-minutes", "360", "--io", "netcdf"]),
    "tune_report_ignored_for_prediction_dry": dict(
        argv_fn=lambda p: ["--tune-report", _tune_report(p), "--prediction",
                           _json_file(p, "pred_rebuild.json", PRED), "--dry-run"]),
    "resume_node_fail": dict(text=T._with_restart(120), argv=["--segment-minutes", "720"],
                             fakes=lambda: TR.Dying([("died", "NODE_FAIL", 9)])),
    "resume_auto_checkpoint": dict(argv=[], fakes=lambda: TR.Dying([("died", "TIMEOUT", 9)])),
    "resume_then_next_segment_netcdf": dict(
        text=T._with_restart(120), argv=["--segment-minutes", "360", "--io", "netcdf"],
        fakes=lambda: TR.Dying([("died", "NODE_FAIL", 5)])),
    "resume_twice_netcdf": dict(
        text=T._with_restart(120),
        argv=["--segment-minutes", "720", "--io", "netcdf", "--resume", "2"],
        fakes=lambda: TR.Dying([("died", "NODE_FAIL", 9), ("died", "COMPLETED", 11)])),
    "resume_budget_used_netcdf": dict(
        text=T._with_restart(120), argv=["--segment-minutes", "720", "--io", "netcdf"],
        fakes=lambda: TR.Dying([("died", "NODE_FAIL", 9), ("died", "NODE_FAIL", 11)])),
    "resume_0_disables": dict(text=T._with_restart(120),
                              argv=["--segment-minutes", "720", "--resume", "0"],
                              fakes=lambda: TR.Dying([("died", "NODE_FAIL", 9)])),
    "no_valid_restart": dict(text=T._with_restart(120), argv=["--segment-minutes", "720"],
                             fakes=lambda: TR.Dying([("died", "NODE_FAIL", 1)])),
    "model_failure": dict(text=T._with_restart(120), argv=["--segment-minutes", "720"],
                          fakes=lambda: TR.Dying([("died", "FAILED", 9)],
                                                 {"rsl.error.0003": SEGFAULT})),
    "cancelled": dict(text=T._with_restart(120), argv=["--segment-minutes", "720"],
                      fakes=lambda: TR.Dying([("died", "CANCELLED", 9)])),
    "out_of_memory": dict(text=T._with_restart(120), argv=["--segment-minutes", "720"],
                          fakes=lambda: TR.Dying([("died", "OUT_OF_MEMORY", 9)])),
    "failed_no_cfl": dict(argv=["--segment-minutes", "360"],
                          fakes=lambda: T.Fakes(outcomes=["boom"])),
    "cfl_advice": dict(argv=["--segment-minutes", "360"], fakes=lambda: T.Fakes(outcomes=["cfl"])),
    "cfl_advice_chem": dict(text=T.NAMELIST_3DOM + TE.CHEM_GROUP,
                            argv=["--segment-minutes", "360", "--io", "netcdf"],
                            fakes=lambda: T.Fakes(outcomes=["cfl"])),
    "cfl_rescue": dict(argv=["--segment-minutes", "360", "--cfl-rescue"],
                       fakes=lambda: T.Fakes(outcomes=["cfl", "done", "done"])),
    "cfl_rescue_second_segment_netcdf": dict(
        argv=["--segment-minutes", "360", "--cfl-rescue", "--io", "netcdf"],
        fakes=lambda: T.Fakes(outcomes=["done", "cfl", "done"])),
    "cfl_rescue_used_twice": dict(argv=["--segment-minutes", "360", "--cfl-rescue"],
                                  fakes=lambda: T.Fakes(outcomes=["cfl", "cfl"])),
    "sacct_unknown": dict(argv=["--segment-minutes", "360", "--io", "netcdf"], fakes=NoSacct),
    "error_out_inside_case": dict(
        argv_fn=lambda p: ["--out", str(p / "case" / "out"), "--dry-run"]),
    "error_no_layout": dict(layout=[], argv=["--dry-run"]),
    "error_no_out": dict(argv_fn=lambda p: ["--out", ""]),
    "error_segment_shorter_than_records": dict(argv=["--segment-minutes", "180", "--dry-run"]),
    "error_checkpoint_off_record_step": dict(argv=["--checkpoint-minutes", "300"]),
    "error_checkpoint_not_a_number": dict(argv=["--checkpoint-minutes", "soon"]),
    "error_checkpoint_does_not_divide": dict(
        argv=["--io", "netcdf", "--segment-minutes", "360", "--checkpoint-minutes", "240"]),
    "error_wall_limit_too_small": dict(argv=["--wall-limit", "00:00:30", "--dry-run"]),
    "error_wall_limit_too_small_netcdf": dict(
        argv=["--wall-limit", "00:00:30", "--io", "netcdf", "--dry-run"]),
    "error_stop_at_off_step": dict(argv=["--stop-at", "2020-06-01_06:00:30", "--dry-run"]),
    "error_stop_at_before_start": dict(argv=["--stop-at", "2020-06-01_00:00:00", "--dry-run"]),
    "error_too_many_ranks": dict(layout=["--nodes", "1", "--nx", "9", "--ny", "9"],
                                 argv=["--dry-run"]),
    "error_case_is_restart_run": dict(
        text=T.NAMELIST_3DOM.replace("restart                             = .false.,",
                                     "restart                             = .true.,"),
        argv=["--dry-run"]),
    "error_rst_outname": dict(
        text=T.NAMELIST_3DOM.replace(" io_form_restart                     = 2",
                                     ' rst_outname = "r_d<domain>_<date>",\n'
                                     " io_form_restart                     = 2"),
        argv=["--dry-run"]),
    "error_staggered_nest": dict(text=STAGGERED, argv=["--dry-run"]),
    "error_existing_run": dict(argv=["--segment-minutes", "360"],
                               pre=lambda p, fb: (p / "out" / "seg_0").mkdir(parents=True)),
    "error_converter_fails": dict(
        argv=["--segment-minutes", "360"],
        pre=lambda p, fb: setattr(fb, "fail", "no Python that can `import adios2`")),
    "error_record_missing_at_segment_start": dict(
        argv=["--segment-minutes", "360"],
        pre=lambda p, fb: fb.records.update(
            wrfbdy_d01=["2020-06-01_00:00:00", "2020-06-01_12:00:00"])),
    "error_wrfda_refused": dict(text=T.NAMELIST_3DOM + TE.WRFVAR, argv=["--dry-run"]),
    "error_cfl_rescue_refused_on_chem": dict(text=T.NAMELIST_3DOM + TE.CHEM_GROUP,
                                             argv=["--cfl-rescue", "--dry-run"]),
    "wrfda_forecast_only_dry": dict(text=T.NAMELIST_3DOM + TE.WRFVAR,
                                    argv=["--wrfda-forecast-only", "--dry-run"]),
    "override_restart_timers_and_hist_at_0h_dry": dict(
        text=T.NAMELIST_3DOM.replace(
            " restart                             = .false.,",
            " restart                             = .false.,\n"
            " override_restart_timers = .true.,\n write_hist_at_0h_rst = .true.,").replace(
            " history_interval                    = 60,  60,   60,",
            " history_interval                    = 0,  60,   60,"),
        argv=["--segment-minutes", "360", "--dry-run"]),
    "single_multirecord_bp_dry": dict(
        argv=["--dry-run"],
        pre=lambda p, fb: ((p / "case" / "wrfbdy_d01").unlink(),
                           T._mk_bp(str(p / "case" / "wrfbdy_d01.bp")))),
}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_golden_run(name, tmp_path, capsys, fake_bp):  # noqa: F811
    spec = RUNS[name]
    case = T._make_case(tmp_path, spec.get("text", T.NAMELIST_3DOM))
    for n in spec.get("files", ()):
        (case / n).write_text("x")
    if "pre" in spec:
        spec["pre"](tmp_path, fake_bp)
    argv = spec["argv_fn"](tmp_path) if "argv_fn" in spec else spec["argv"]
    rc, lines = _main(tmp_path, argv, spec.get("fakes", T.Fakes)(),
                      layout=spec.get("layout", LAYOUT))
    _check("run_%s.txt" % name, _snapshot(tmp_path, rc, lines, capsys, str(tmp_path / "out")))


def test_golden_run_prints_with_the_default_log(tmp_path, capsys):
    T._make_case(tmp_path)
    rc = wr.main([str(tmp_path / "case")] + LAYOUT + ["--out", str(tmp_path / "out"),
                                                      "--segment-minutes", "360", "--dry-run"])
    _check("run_default_log.txt", _snapshot(tmp_path, rc, [], capsys, str(tmp_path / "out")))


def test_golden_run_project_default_out(tmp_path, capsys):
    from wtune import project as wtune_project
    T._make_case(tmp_path)
    proj = tmp_path / "proj"
    wtune_project.init_project(str(proj))
    lines = []
    rc = wr.main([str(tmp_path / "case")] + LAYOUT + ["--project", str(proj), "--dry-run"],
                 log=lines.append)
    _check("run_project_default_out.txt", _snapshot(tmp_path, rc, lines, capsys, "/nonexistent"))


def test_golden_run_sidecar_idle(tmp_path, capsys):
    """--sidecar with nothing to convert: its report section and its closing line."""
    pytest.importorskip("netCDF4")

    def no_srun(argv):
        raise AssertionError("srun started: %r" % (argv,))
    T._make_case(tmp_path)
    rc, lines = _main(tmp_path, ["--segment-minutes", "360", "--io", "netcdf", "--sidecar"],
                      sidecar_runner=no_srun)
    _check("run_sidecar_idle.txt", _snapshot(tmp_path, rc, lines, capsys, str(tmp_path / "out")))


def test_golden_run_report_while_running(tmp_path, capsys):
    """run_report.json as each sacct poll finds it: attempts submitted but not yet finished."""
    class Peek(TR.Dying):
        seen = []

        def sacct(self, cmd):
            with open(str(tmp_path / "out" / "run_report.json")) as fh:
                self.seen.append(fh.read())
            return super().sacct(cmd)
    T._make_case(tmp_path, T._with_restart(120))
    f = Peek([("died", "NODE_FAIL", 5), "cfl", "done", "done"])
    rc, _lines = _main(tmp_path, ["--segment-minutes", "360", "--io", "netcdf", "--cfl-rescue"], f)
    capsys.readouterr()
    _check("run_report_while_running.txt", _norm(
        "rc=%s\n" % rc + "".join("--- poll %d\n%s\n" % (i, t) for i, t in enumerate(f.seen)),
        tmp_path))


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def test_golden_cli_help():
    _check("cli_help.txt", wr.build_arg_parser().format_help().replace(REPO, "<REPO>"))


@pytest.mark.parametrize("entry", ["script", "module"])
def test_golden_entry_points(entry, tmp_path):
    env = dict(os.environ, COLUMNS="100", PYTHONPATH=SRC)
    argv = ([sys.executable, os.path.join(SRC, "wtune", "run")] if entry == "script"
            else [sys.executable, "-m", "wtune.run"])
    T._make_case(tmp_path)
    out = []
    for args in (["--help"], [str(tmp_path / "case")] + LAYOUT + [
            "--out", str(tmp_path / "out"), "--segment-minutes", "360", "--dry-run"]):
        r = subprocess.run(argv + args, capture_output=True, text=True, cwd=str(tmp_path),
                           env=env, timeout=120)
        out.append("rc=%d\n%s--- stderr\n%s" % (r.returncode, r.stdout, r.stderr))
    _check("entry_%s.txt" % entry, _norm("\n".join(out), tmp_path))


# ---------------------------------------------------------------------------
# pure planners and renderers
# ---------------------------------------------------------------------------
def _try(fn):
    try:
        return fn()
    except Exception as e:      # noqa: BLE001  the error text is what is pinned
        return "%s: %s" % (type(e).__name__, e)


def _text36():
    return T.NAMELIST_3DOM.replace("time_step                           = 60,",
                                   "time_step                           = 36,")


def test_golden_geometry_and_wall_model(tmp_path):
    g = T._geo(tmp_path)
    g36 = T._geo(tmp_path, _text36())
    over = wr.overheads(init_s=100.0, restart_read_s=50.0, restart_write_s=40.0,
                        history_frame_s=[10.0, 5.0], source="test")
    comp = wr.compute_fn_from_points(g, [(g["start"], 1.0), (g["end"], 3.0)], 2.0, 4.0, None)
    fb = wr.fallback_overheads(g, 2)
    pred = {"result": {"points": [{"at": g["start"].isoformat(), "s_normal": 1.0}],
                       "r_night": 1.0, "r_day": 1.0, "init_s": 50.0}}
    pred_io = dict(pred, io={"restart": {"per_write_s": 7.0}, "history": [{"median_s": 1.5}]})
    out = {
        "geometry": g, "geometry_dt36": g36, "overheads_default": wr.overheads(),
        "fallback_overheads": {"%s_%s" % (f, io): wr.fallback_overheads(g, f, io)
                               for f in (14, 2) for io in ("adios2", "netcdf")},
        "model_from_prediction": [wr.model_from_prediction(p, g, None, fb)[1]
                                  for p in (pred, pred_io)],
        "compute": [comp(0, 300), comp(300, 720), wr.constant_compute_fn(0.5)(10, 30)],
        "history_frames": [[wr.history_frames(g, d, a, b) for d in range(4)]
                           for a, b in ((0, 360), (360, 720), (0, 59), (59, 61))],
        "history_wall": [wr.history_wall(g, over, 0, 360), wr.history_wall(g, fb, 360, 720),
                         wr.history_wall(g, wr.overheads(), 0, 720)],
        "segment_wall": [wr.segment_wall(g, comp, over, a, b, L, cad)
                         for a, b, L, cad in ((0, 720, None, None), (0, 360, 360, None),
                                              (360, 720, 360, None), (360, 700, 360, None),
                                              (0, 720, None, 10800.0), (360, 720, 360, 7200.0))],
        "restart_write_steps": [wr.restart_write_steps(g, a, b, c) for a, b, c in (
            (0, 720, 10800.0), (0, 10, 90.0), (3, 9, 90.0), (0, 10, None))],
        "restart_times": wr.restart_times(g, 360, 720, 7200.0),
        "restart_interval_minutes": [_try(lambda L=L: wr.restart_interval_minutes(g36, L))
                                     for L in (None, 5, 300, 7)],
        "time_limit_s": [wr.time_limit_s(1100.0), wr.time_limit_s(1000.0),
                         wr.time_limit_s(1100.0, 1800)],
        "parse_hms": [_try(lambda s=s: wr.parse_hms(s))
                      for s in ("02:00:00", "30:00:00", "1-02:03:04", "10:00")],
        "step_times": [str(wr.step_time(g36, 5)), wr.step_of(g36, "2020-06-01_00:03:00"),
                       wr.step_of(g36, "2020-06-01_00:01:00")],
        "resolve_end_step": [_try(lambda s=s: wr.resolve_end_step(g, s)) for s in (
            None, "2020-06-01_06:00:00", "2020-06-02_00:00:00", "2020-06-01_06:00:30",
            "2020-06-01_00:00:00")],
        "segment_steps_for_minutes": [_try(lambda m=m: wr.segment_steps_for_minutes(g36, m))
                                      for m in (3, 1, 0)],
        "plan_segments_fixed": [_try(lambda n=n: wr.plan_segments_fixed(720, n, 60))
                                for n in (300, 0)],
        "rescue_time_step": [wr.rescue_time_step(*a) for a in (
            (60, 21600, (3600, 1800)), (100, 6000, (3600,)), (60, 3600, (3600, 7)), (1, 60, ()),
            (60, 3607, ()))],
        "attempt_dir_name": [wr.attempt_dir_name(k, wr.Attempt(a=0, **kw)) for k, kw in (
            (0, {}), (1, {"time_step": 45}), (2, {"resume": 1}),
            (3, {"time_step": 45, "resume": 2}))],
    }
    _check("pure_geometry_and_wall_model.json", _dump(out) + "\n")


def test_golden_layouts_and_candidates(tmp_path):
    g = T._geo(tmp_path)
    rows = [dict(WINNER, nodes=2, median=0.3), dict(WINNER, nodes=1, median=0.6, nx=4, ny=18),
            dict(WINNER, nodes=1, median=None, nx=6, ny=12)]
    cands = wr.tune_candidates({"nodes": 1, "ranking": RANKING}, g, None, {}, 1)
    out = {
        "tune_report_layout": [wr.tune_report_layout(r) for r in (
            {"winner": WINNER}, {"node_selection": {"winner": dict(WINNER, nx=6, cid=None)}},
            {"node_selection": {"chosen": 1, "layouts": rows}},
            {"node_selection": {"chosen": 1, "final": 2, "layouts": rows}}, {"ranking": []},
            {"winner": {"nodes": 1, "nx": 2}})],
        "make_config": [_try(lambda a=a: wr.make_config(*a)) for a in (
            (8, 9, 1, 1, 1), ("6", "12", "2", "2", 2, "G7", "cyclic"), (9, 9, 1, 1, 1))],
        "rank_mapping_notes": [wr.rank_mapping_notes(wr.make_config(*a), c) for a, c in (
            ((8, 10, 1, 3, 5), 48), ((8, 10, 1, 3, 6), None), ((8, 10, 1, 3, 5), 72),
            ((3, 1, 1, 40, 2), None))],
        "tune_candidates": [c["config"] for c in cands],
        "tune_candidates_compute": [[c["compute"](0, 360), c["compute"](360, 720)] for c in cands],
        "tune_candidates_errors": [
            _try(lambda: wr.tune_candidates({"nodes": 2, "ranking": RANKING}, g, None, {}, 1)),
            _try(lambda: wr.tune_candidates({"ranking": RANKING[2:]}, g, None, {}, 1))],
    }
    _check("pure_layouts_and_candidates.json", _dump(out) + "\n")


def _plans(g):
    one = [{"config": wr.make_config(8, 9, 1, 1, 1), "compute": wr.constant_compute_fn(1.0)}]
    two = one + [{"config": wr.make_config(6, 12, 2, 1, 1, cid="B"),
                  "compute": lambda a, b: (b - a) * (0.5 if a >= 360 else 2.0)}]
    over = wr.overheads(init_s=100.0, restart_read_s=50.0, restart_write_s=50.0,
                        history_frame_s=[10.0, 10.0, 10.0], source="test")
    return {
        "one_segment": (720, one, over, {}),
        "segment_minutes": (700, one, over, dict(segment_minutes=300)),
        "segment_minutes_two_candidates": (720, two, over, dict(segment_minutes=360)),
        "wall_limit": (720, one, over, dict(wall_limit_s=600.0)),
        "wall_limit_too_small": (720, one, over, dict(wall_limit_s=100.0)),
        "wall_limit_aligned": (720, one, over, dict(wall_limit_s=1000.0, align=dict(
            ALIGN, steps=240, seconds=14400.0, label="interval_seconds 04:00:00"))),
        "wall_limit_aligned_too_small": (720, one, over, dict(wall_limit_s=10.0, align=ALIGN)),
        "wall_limit_explicit_checkpoint": (720, one, over, dict(wall_limit_s=600.0,
                                                                checkpoint=45)),
        "segment_minutes_rounded_to_records": (720, one, over, dict(segment_minutes=500,
                                                                    align=ALIGN)),
        "segment_minutes_shorter_than_records": (720, one, over, dict(segment_minutes=180,
                                                                      align=ALIGN)),
        "segment_minutes_whole_run": (720, one, over, dict(segment_minutes=720, align=ALIGN)),
        "case_restart_kept": (720, one, over, dict(case_restart_s=180 * 60.0)),
        "case_restart_divides_segments": (720, one, over, dict(segment_minutes=360,
                                                               case_restart_s=120 * 60.0)),
        "auto_checkpoint": (720, one, over, dict(align=ALIGN, checkpoint="auto")),
    }


def test_golden_plan_run(tmp_path):
    g = T._geo(tmp_path)
    out = {}
    for name, (n_end, cands, over, kw) in _plans(g).items():
        notes = []
        res = _try(lambda: wr.plan_run(g, n_end, cands, over, notes=notes, **kw))
        out[name] = res if isinstance(res, str) else {"L": res[0], "segments": res[1],
                                                      "notes": notes}
    _check("pure_plan_run.json", _dump(out) + "\n")


def test_golden_restart_and_checkpoint_settings(tmp_path):
    g = T._geo(tmp_path)
    out = {}
    for L in (None, 360):
        for case_min in (0, 120, 180, 250, 360, 7200):
            for ckpt in (None, "auto", 90, 240, 360):
                for align in (None, ALIGN):
                    key = "L=%s case=%s checkpoint=%s align=%s" % (L, case_min, ckpt,
                                                                   bool(align))
                    out[key] = _try(lambda: wr.restart_setting(g, L, 720, case_min * 60.0, ckpt,
                                                               align))
    out["72h auto"] = wr.restart_setting(g, None, 72 * 60, 100000 * 60.0, "auto", ALIGN)
    out["checkpoint_request"] = [_try(lambda v=v, r=r: wr.checkpoint_request(v, r)) for v, r in (
        ("auto", 1), ("auto", 0), ("0", 1), ("90", 0), ("soon", 1), ("-5", 1))]
    out["checkpoint_steps_for_minutes"] = [
        _try(lambda m=m, al=al: wr.checkpoint_steps_for_minutes(g, m, al))
        for m, al in ((720, ALIGN), (45, None), (300, ALIGN))]
    out["auto_checkpoint_steps"] = [wr.auto_checkpoint_steps(*a) for a in (
        (4320, None, 360), (4320, None, 1), (720, None, 360), (360, None, 360),
        (4320, 1440, 360), (4320, 1080, 360), (720, 360, 360))]
    _check("pure_restart_settings.json", _dump(out) + "\n")


def test_golden_record_alignment_and_io_plan(tmp_path):
    g = T._geo(tmp_path)
    nl = T._nl_from_text(tmp_path, T.NAMELIST_AUX.replace("auxinput4_interval = 360, 360, 360",
                                                          "auxinput4_interval = 240, 120, 240"))
    no_iv = T._nl_from_text(tmp_path, T.NAMELIST_AUX.replace(
        " interval_seconds                    = 21600\n", ""), "noiv.nml")
    out = {"alignment": [_try(lambda n=n, x=x: wr.input_record_alignment(x, g, n)) for x, n in (
        (nl, ["wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02", "wrffdda_d01"]),
        (nl, ["wrfbdy_d01"]), (nl, ["wrflowinp_d02"]), (nl, ["wrfinput_d01"]), (nl, []),
        (no_iv, ["wrfbdy_d01"]))]}
    cand = [{"config": wr.make_config(8, 9, 1, 1, 1), "compute": wr.constant_compute_fn(1.0)}]
    _L, segs = wr.plan_run(g, 720, cand, wr.overheads(), segment_minutes=360)
    links = {"wrfbdy_d01_2020-06-01_00:00:00.bp": "x", "wrfinput_d01.bp": "y"}
    out["missing_record_files"] = [
        wr.missing_record_files(links, ["wrfbdy_d01", "wrflowinp_d01"], segs),
        wr.missing_record_files({}, ["wrfbdy_d01"], segs)]
    case = T._make_case(tmp_path, T.NAMELIST_AUX)
    cc = wr.sweep.campaign_configs
    plan = cc.resolve_bp_inputs(T.NAMELIST_AUX, str(case), str(tmp_path / "cache"), convert=False)
    notes = []
    streams = wr.adios2_streams_or_raise(plan, str(case), notes)
    out["adios2_streams"] = [st["stream"] for st in streams]
    out["adios2_notes"] = notes
    out["io_plan_lines"] = [
        wr.io_plan_line("adios2", streams, 14, plan, str(tmp_path / "cache"), str(case)),
        wr.io_plan_line("adios2", streams, 2, plan, str(tmp_path / "cache"), str(case)),
        wr.io_plan_line("netcdf", None, 2, None, None, str(case))]
    (case / "wrfbdy_d01").unlink()
    plan = cc.resolve_bp_inputs(T.NAMELIST_AUX, str(case), str(tmp_path / "cache"), convert=False)
    out["adios2_missing_boundary"] = _try(lambda: wr.adios2_streams_or_raise(plan, str(case), []))
    _check("pure_record_alignment_and_io_plan.json", _norm(_dump(out), tmp_path) + "\n")


def test_golden_segment_namelists():
    t = T._dt
    cfg = wr.make_config(12, 6, 4, 2, 2)
    aux = [st for st in wr.sweep.campaign_configs.adios2_input_streams(T.NAMELIST_AUX)]
    edits = T.REAL_BP.per_record_namelist_edits(["wrfbdy_d01", "wrflowinp_d01", "wrffdda_d01"], 3)
    many = T.NAMELIST_3DOM.replace(
        T._RST_LINE, T._RST_LINE + "\n restart_interval_h = 5,\n restart_interval_m = 30,\n"
        " override_restart_timers = .true.,").replace(
        " time_step                           = 60,",
        " time_step                           = 60,\n time_step_fract_num = 1,\n"
        " time_step_fract_den = 2,")
    cases = {
        "k0": (T.NAMELIST_3DOM, "00", "06", 0, 360, {}),
        "k1_netcdf_restart": (T.NAMELIST_3DOM, "06", "12", 1, 360, dict(restart_io_form=2)),
        "k0_resumed": (T.NAMELIST_3DOM, "04", "12", 0, 120, dict(resumed=True)),
        "k1_case_restart_keys_kept": (many, "06", "12", 1, None, {}),
        "k0_restart_variants_time_step": (many, "00", "06", 0, 360, dict(time_step=45)),
        "k0_adios2_aux_streams": (T.NAMELIST_AUX, "00", "06", 0, 360, dict(io_streams=aux)),
        "k1_adios2_per_record": (T.NAMELIST_AUX, "06", "12", 1, wr.RESTART_OFF_MIN,
                                 dict(io_streams=aux, per_record_edits=edits)),
        "k0_adios2_numaggregators_of_case": (
            T.NAMELIST_3DOM + " &namelist_adios2\n adios2_numaggregators = 4,\n /\n", "00", "06",
            0, 360, {}),
    }
    parts = []
    for name, (text, h0, h1, k, rst, kw) in cases.items():
        new = wr.render_segment_namelist(text, t("2020-06-01_%s:00:00" % h0),
                                         t("2020-06-01_%s:00:00" % h1), k, cfg, rst, **kw)
        parts += ["--- %s" % name, new, "--- %s diff" % name] + wr.namelist_diff_summary(text, new)
    parts += ["--- errors", _try(lambda: wr.render_segment_namelist(
        T.NAMELIST_3DOM, t("2020-06-01_06:00:00"), t("2020-06-01_06:00:00"), 0, cfg, 360))]
    _check("pure_segment_namelists.txt", "\n".join(parts) + "\n")


def test_golden_segment_sbatch_and_xml(tmp_path):
    parts = []
    for name, cfg, k, kw in (
            ("omp2_two_nodes", wr.make_config(6, 12, 2, 2, 2), 0,
             dict(campaign_dir="/repo/campaign")),
            ("omp1_cyclic_default_campaign", wr.make_config(10, 13, 1, 1, 2, dist="cyclic"), 3,
             dict(env_source="/opt/env.sh"))):
        parts += ["--- sbatch %s" % name, wr.render_segment_sbatch(
            str(tmp_path / "seg dir"), k, cfg, "01:30:00", "wtrun-x-s%d" % k, **kw)]
    base = ('<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfrst_d01_X">\n'
            '    <engine type="BP5"/>\n  </io>\n</adios-config>\n')
    extra = ('<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfrst_d01_X">\n'
             '    <engine type="BP4"/>\n  </io>\n  <io name="a">\n  </io></adios-config>\n')
    merged = wr.merge_adios2_xml(base, extra)
    parts += ["--- merge", merged, "--- merge no base", wr.merge_adios2_xml(None, extra),
              "--- merge not xml", wr.merge_adios2_xml("x", extra),
              "--- merge no extra", wr.merge_adios2_xml(base, ""),
              "--- restart xml", wr.render_adios2_restart_xml(["X", "Y"], 2),
              "--- restart xml on base", wr.render_adios2_restart_xml(["X", "Y"], 2, merged),
              "--- restart xml on text", wr.render_adios2_restart_xml(["X"], 1, "x")]
    _check("pure_segment_sbatch_and_xml.txt", _norm("\n".join(parts) + "\n", tmp_path))


def test_golden_output_dirs_and_staging(tmp_path):
    t0 = dt.datetime(2020, 6, 1, 0)
    nl = T._nl(tmp_path, history_interval="180, 60, 60", restart_interval="360",
               frames_per_outfile="1, 3, 1")
    out = {"output_dir_names": [wr.output_dir_names(nl, a, b, n) for a, b, n in (
        (t0, t0 + dt.timedelta(hours=6), 2), (t0 + dt.timedelta(hours=6),
                                              t0 + dt.timedelta(hours=12), 3))]}
    case = T._make_case(tmp_path, IOFIELDS)
    out["missing_namelist_files"] = wr._missing_namelist_files(IOFIELDS, str(case))
    seg = tmp_path / "out" / "seg_0"
    out["staged"] = wr.setup_segment_dir(str(case), str(seg), IOFIELDS, adios2_xml="<x/>")
    out["tree"] = _tree(str(tmp_path / "out"))
    out["inside_case"] = _try(lambda: wr.setup_segment_dir(str(case), str(case / "s"), "NML"))
    out["restart_files"] = wr.restart_files(str(case), "2020-06-01_06:00:00", 2)
    out["fresh_bp_links"] = wr.fresh_bp_links(
        {"a_1.bp": "/cache/a/a_1.bp", "b.bp": "/cache/b"}, {"a": "/cache/a"}, {"a": "/seg/_bp/a"})
    geo = T._geo(tmp_path)
    out["restart_links_missing"] = _try(lambda: wr._restart_links(
        geo, 1, wr.Attempt(a=360, src_dir=str(case)), "2020-06-01_06:00:00"))
    _check("pure_output_dirs_and_staging.json", _norm(_dump(out), tmp_path) + "\n")


def test_golden_wait_for_job():
    """The poll loop: one log line per state change, on_poll before every sleep."""
    def run(states, **kw):
        seq, log, polls, sleeps = list(states), [], [], []
        res = wr.wait_for_job("77", lambda cmd: seq.pop(0) if seq else "", poll_s=3.0,
                              sleep=sleeps.append, log=log.append, on_poll=polls.append, **kw)
        return {"result": res, "log": log, "on_poll": polls, "sleeps": sleeps}
    out = {"completed": run(["77|PENDING|00:00:00\n", "77|RUNNING|00:01:00\n",
                             "77|RUNNING|00:02:00\n", "77|COMPLETED|01:02:03\n"]),
           "failed_at_once": run(["77|NODE_FAIL|1-00:00:10\n"]),
           "unknown_then_listed": run(["", "", "77|TIMEOUT|00:30:00\n"]),
           "never_listed": run([])}
    _check("pure_wait_for_job.json", _dump(out) + "\n")



# ---------------------------------------------------------------------------
# package structure
# ---------------------------------------------------------------------------
LAYERS = ["_deps", "common", "geometry", "estimate", "layout", "plan", "namelist", "staging",
          "sbatch", "inputs", "context", "report", "attempts", "controller", "cli"]


def test_package_layers_import_only_lower_layers():
    """Each wtune_run module imports (from .x) only modules listed before it in LAYERS."""
    import ast
    pkg = os.path.join(SRC, "wtune", "run")
    if not os.path.isdir(pkg):
        pytest.skip("single-module wtune_run")
    found = sorted(n[:-3] for n in os.listdir(pkg)
                   if n.endswith(".py") and n not in ("__init__.py", "__main__.py"))
    assert found == sorted(LAYERS)
    for i, name in enumerate(LAYERS):
        with open(os.path.join(pkg, name + ".py")) as fh:
            tree = ast.parse(fh.read())
        deps = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}
        assert deps <= set(LAYERS[:i]), "%s imports upwards: %s" % (name, deps - set(LAYERS[:i]))
