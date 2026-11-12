#!/usr/bin/env python3
"""pytest suite for wtune/run/ (`wtune run`: segmented WRF executor with restart hand-offs).
No job is ever submitted: sbatch/sacct/snapshot are fakes, every directory is a tmp dir."""
from conftest import swap_wtune_module
import datetime
import json
import glob
import os
import re
import subprocess

import sys
import types

import pytest

# Importing wtune.run -> scc_sweep sets SITE=scc and loads campaign `configs` with it; undo both so
# campaign/test_configs.py (collected later in the same process) gets its own f1 import.
_SITE_BEFORE = os.environ.get("SITE")
_CONFIGS_BEFORE = sys.modules.get("configs")

from wtune import predict as wp
from wtune import run as wr
import wtune.run.inputs
import wtune.watch

if _SITE_BEFORE is None:
    os.environ.pop("SITE", None)
if _CONFIGS_BEFORE is None:
    sys.modules.pop("configs", None)


def _load_real_bp():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wtune_bp_real", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "src", "wtune", "bp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_BP = _load_real_bp()        # its per-record helpers (stdlib only, no adios2) are used as-is


def _mk_bp(path):
    """A directory wtune.bp.is_bp recognizes (BP4: md.idx + data.0)."""
    os.makedirs(path, exist_ok=True)
    for f in ("md.idx", "data.0"):
        open(os.path.join(path, f), "w").close()


class FakeBP:
    """Stand-in for src/wtune/bp.py (the converter module): convert_inputs makes cache_dir/<name>
    BP directories (its real output naming; wrfbdy/wrflowinp/wrffdda as per-record containers of
    <name>_<date>.bp, dates from `records`) and records the call; no adios2 anywhere. The
    per-record helpers are the real module's."""
    XML = '<?xml version="1.0"?>\n<!-- wtune -->\n<adios-config>\n</adios-config>\n'
    DEFAULT_RECORDS = ["2020-06-01_00:00:00", "2020-06-01_06:00:00", "2020-06-01_12:00:00"]

    def __init__(self):
        self.calls = []
        self.fail = None
        self.records = {}
        self.cp_calls = []
        self.no_reflink = False

    def module(self):
        m = types.ModuleType("wtune_bp")
        m.ADIOS2_IO_FORM = 14
        m.require_adios2_python = self.require_adios2_python
        m.convert_inputs = self.convert_inputs
        m.adios2_xml = lambda engine_by_stream=None: self.XML
        m.is_bp = os.path.isdir
        m.cached_inputs = lambda d, cache, names, engine="BP4": (
            {n: os.path.join(cache, n) for n in names}
            if all(os.path.exists(os.path.join(cache, n)) for n in names) else None)
        for name in ("PER_RECORD_NAME_RE", "is_per_record_dir", "record_files",
                     "bp_input_links", "per_record_namelist_edits"):
            setattr(m, name, getattr(REAL_BP, name))
        m.fresh_input_copy = lambda src, dest, stats=None: REAL_BP.fresh_input_copy(
            src, dest, runner=self.cp, stats=stats)
        return m

    def cp(self, cmd, **kw):
        """Fake `cp`: reflink succeeds unless `no_reflink`; copies with shutil."""
        import shutil
        self.cp_calls.append(list(cmd))
        if "--reflink=always" in cmd and self.no_reflink:
            return subprocess.CompletedProcess(cmd, 1, b"", b"reflink not supported")
        shutil.copytree(cmd[-2], cmd[-1], symlinks=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def require_adios2_python(self):
        if self.fail:
            raise RuntimeError(self.fail)
        return "/fake/adios2/python"

    def convert_inputs(self, case_dir, cache_dir, names=None, engine="BP4"):
        self.calls.append((case_dir, cache_dir, list(names), engine))
        out = {}
        for n in names:
            p = os.path.join(cache_dir, n)
            if REAL_BP.PER_RECORD_NAME_RE.match(n):
                for t in self.records.get(n, self.DEFAULT_RECORDS):
                    _mk_bp(os.path.join(p, "%s_%s.bp" % (n, t)))
            else:
                _mk_bp(p)
            out[n] = p
        return out


@pytest.fixture(autouse=True)
def fake_bp(monkeypatch):
    fb = FakeBP()
    swap_wtune_module(monkeypatch, "bp", fb.module())
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    return fb

NAMELIST_3DOM = """ &time_control
 run_days                            = 0,
 run_hours                           = 12,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2020, 2020, 2020,
 start_month                         = 06,   06,   06,
 start_day                           = 01,   01,   01,
 start_hour                          = 00,   00,   00,
 end_year                            = 2020, 2020, 2020,
 end_month                           = 06,   06,   06,
 end_day                             = 01,   01,   01,
 end_hour                            = 12,   12,   12,
 interval_seconds                    = 21600
 input_from_file                     = .true.,.true.,.true.,
 history_interval                    = 60,  60,   60,
 frames_per_outfile                  = 1, 1, 1,
 restart                             = .false.,
 restart_interval                    = 7200,
 io_form_history                     = 2
 io_form_restart                     = 2
 io_form_input                       = 2
 io_form_boundary                    = 2
 /

 &domains
 time_step                           = 60,
 max_dom                             = 3,
 e_we                                = 100, 100, 91,
 e_sn                                = 100, 91,  91,
 e_vert                              = 45,  45,  45,
 parent_id                           = 1,   1,   2,
 parent_grid_ratio                   = 1,   3,   3,
 parent_time_step_ratio              = 1,   3,   3,
 /

 &physics
 radt                                = 30, 10, 10,
 /
"""


def _dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")


def _nl_from_text(tmp_path, text, name="namelist.input"):
    p = tmp_path / name
    p.write_text(text)
    return wp.parse_namelist(str(p))


def _geo(tmp_path, text=NAMELIST_3DOM):
    return wr.case_geometry(_nl_from_text(tmp_path, text, "geo.nml"))


def _cfg(nx=8, ny=9, numtiles=1, omp=1, nodes=1, cid="cli"):
    return wr.make_config(nx, ny, numtiles, omp, nodes, cid=cid)


# ---------------------------------------------------------------------------
# run dir
# ---------------------------------------------------------------------------
def _make_case(tmp_path, text=NAMELIST_3DOM):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(text)
    for name in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "rsl.error.0000", "wrf.out",
                 "namelist.output", "wrfout_d01_2020-06-01_00:00:00",
                 "wrfrst_d01_2020-06-01_06:00:00", "adios2.xml", "myiofields.txt"):
        (case / name).write_text("x")
    (case / "wrfinput_d03.bp").mkdir()
    (case / "junkdir").mkdir()
    return case


def _tree_state(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            st = os.lstat(p)
            out[p] = (st.st_mtime_ns, st.st_size)
    return out


def test_setup_segment_dir_links_and_never_writes_case(tmp_path):
    case = _make_case(tmp_path)
    wrf = tmp_path / "WRF"
    (wrf / "run").mkdir(parents=True)
    for name in ("LANDUSE.TBL", "wrf.exe", "namelist.input", "myiofields.txt"):
        (wrf / "run" / name).write_text("w")
    prev = tmp_path / "out" / "seg_0"
    prev.mkdir(parents=True)
    rst = [prev / ("wrfrst_d%02d_2020-06-01_06:00:00" % d) for d in (1, 2, 3)]
    for p in rst:
        p.mkdir()                                   # BP4 restarts are directories
    before = _tree_state(case)
    seg = tmp_path / "out" / "seg_1"
    xml = wr.render_adios2_restart_xml(["2020-06-01_06:00:00", "2020-06-01_09:00:00"], 3)
    found = wr.restart_files(str(prev), "2020-06-01_06:00:00", 3)
    assert all(len(v) == 1 for v in found.values())
    wr.setup_segment_dir(str(case), str(seg), "NML", [p for v in found.values() for p in v], xml,
                         str(wrf))
    assert _tree_state(case) == before
    names = set(os.listdir(seg))
    assert {"wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrfinput_d03.bp", "myiofields.txt",
            "LANDUSE.TBL", "wrf.exe", "namelist.input", "adios2.xml"} <= names
    assert not names & {"rsl.error.0000", "wrf.out", "namelist.output", "junkdir",
                        "wrfout_d01_2020-06-01_00:00:00"}
    assert os.readlink(seg / "myiofields.txt") == str(case / "myiofields.txt")   # case wins
    for p in rst:
        assert os.readlink(seg / p.name) == str(p)
    assert not os.path.islink(seg / "namelist.input")
    assert (seg / "namelist.input").read_text() == "NML"
    assert not os.path.islink(seg / "adios2.xml")
    assert (seg / "adios2.xml").read_text() == (
        '<?xml version="1.0"?>\n<adios-config>\n'
        '  <io name="wrfrst_d01_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d02_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d03_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d01_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d02_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d03_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '</adios-config>\n')


@pytest.mark.parametrize("site, nodes, args, expected", [
    ("scc", 2, "72 36 2 block", "mpirun -np 72 --map-by ppr:36:node:PE=2 --bind-to core ./wrf.exe"),
    ("scc", 2, "130 65 1 cyclic",
     "mpirun -np 130 --map-by ppr:65:node:PE=1 --bind-to core --rank-by node ./wrf.exe"),
    # two NUMA domains per node on f1-arm: the ranks of a node are split over them
    ("f1arm", 1, "90 90 1 block", "mpirun -np 90 --map-by ppr:45:numa:PE=1 --bind-to core ./wrf.exe"),
    ("f1arm", 1, "81 81 1 block", "mpirun -np 81 --map-by ppr:81:node:PE=1 --bind-to core ./wrf.exe"),
])
def test_a_segment_starts_wrf_with_the_site_files_launcher(site, nodes, args, expected):
    campaign = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "campaign")
    script = ('CAMPAIGN=%s; source "$CAMPAIGN/sites/%s.sh"; site_launch %s; '
              'echo "${SITE_LAUNCH_CMD[*]}"' % (campaign, site, args))
    env = dict(HOME=os.environ.get("HOME", "/tmp"), PATH=os.environ["PATH"],
               SLURM_JOB_NUM_NODES=str(nodes))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert out.stdout.strip() == expected, out.stderr


# ---------------------------------------------------------------------------
# controller (fakes)
# ---------------------------------------------------------------------------
class Fakes:
    """sbatch creates the restart files the segment would write; sacct says COMPLETED; the
    snapshot outcome per submission comes from `outcomes` (default done)."""

    def __init__(self, outcomes=None):
        self.submitted = []
        self.outcomes = list(outcomes or [])
        self.snap_by_dir = {}

    def sbatch(self, cmd):
        path = cmd.split()[-1].strip("'")
        seg = os.path.dirname(path)
        nl = wp.parse_namelist(os.path.join(seg, "namelist.input"))
        info = wtune.watch.run_info(nl)
        outcome = self.outcomes.pop(0) if self.outcomes else "done"
        self.submitted.append((seg, outcome))
        if outcome == "done":
            for d in range(1, len(info["domains"]) + 1):
                rst = os.path.join(seg, "wrfrst_d%02d_%s" % (d, info["end"].strftime(
                    "%Y-%m-%d_%H:%M:%S")))
                os.makedirs(rst, exist_ok=True)
                open(os.path.join(rst, "md.idx"), "w").close()
        self.snap_by_dir[seg] = outcome
        return "Submitted batch job %d\n" % (1000 + len(self.submitted))

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        return "%s|COMPLETED|00:10:00\n" % jid

    def snapshot(self, rundir):
        outcome = self.snap_by_dir[rundir]
        if outcome == "done":
            return {"status": "done", "warnings": [], "sum_d01_elapsed_s": 300.0,
                    "n_d01_steps": 360}
        if outcome == "cfl":
            return {"status": "failed", "warnings": [
                {"code": "FAILED", "text": "NaN in a d01 Timing line"},
                {"code": "CFL", "text": "3 cfl line(s) in rsl.error.0000"}]}
        return {"status": "failed", "warnings": [{"code": "FAILED", "text": "forrtl: severe"}]}


def _run(tmp_path, fakes, extra=()):
    case = tmp_path / "case" if (tmp_path / "case").exists() else _make_case(tmp_path)
    out = tmp_path / "out"
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes", "360",
                  "--out", str(out), "--poll-interval", "0"] + list(extra),
                 sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                 snapshot_fn=fakes.snapshot, sleep=lambda s: None, log=lines.append)
    report = None
    if (out / "run_report.json").exists():
        report = json.loads((out / "run_report.json").read_text())
    return rc, out, report, lines


# ---------------------------------------------------------------------------
# 0.8.2 (climate-ea50): the case's restart cadence is kept; namelist-named files are staged
# ---------------------------------------------------------------------------
_RST_LINE = " restart_interval                    = 7200,"


def _with_restart(minutes, text=NAMELIST_3DOM):
    return text.replace(_RST_LINE, " restart_interval                    = %d," % minutes)


# ---------------------------------------------------------------------------
# ADIOS2 I/O (default since 2026-10-03; --io netcdf = legacy escape)
# ---------------------------------------------------------------------------
NAMELIST_AUX = NAMELIST_3DOM.replace(
    " &physics\n", " &physics\n sst_update = 1,\n").replace(
    " io_form_boundary                    = 2\n",
    " io_form_boundary                    = 2\n auxinput4_inname = \"wrflowinp_d<domain>\",\n"
    " io_form_auxinput4 = 2,\n auxinput4_interval = 360, 360, 360,\n") + (
    " &fdda\n grid_fdda = 1, 0, 0,\n gfdda_inname = \"wrffdda_d<domain>\",\n"
    " io_form_gfdda = 2,\n /\n")


def _streams(text):
    return [st for st in wr.sweep.campaign_configs.adios2_input_streams(text)]


def test_adios2_streams_of_case_with_sst_update_and_fdda():
    sts = {st["stream"]: st for st in _streams(NAMELIST_AUX)}
    assert list(sts) == ["input", "boundary", "auxinput4", "gfdda"]
    assert sts["input"]["names"] == ["wrfinput_d01", "wrfinput_d02", "wrfinput_d03"]
    assert sts["input"]["required"] == ["wrfinput_d01"]
    assert sts["boundary"]["names"] == ["wrfbdy_d01"]
    assert sts["auxinput4"]["inname"] == "wrflowinp_d<domain>.bp"
    assert sts["gfdda"]["names"] == ["wrffdda_d01"] and sts["gfdda"]["group"] == "fdda"
    assert [st["stream"] for st in _streams(NAMELIST_3DOM)] == ["input", "boundary"]


NAMELIST_AUXHIST = NAMELIST_3DOM.replace(
    " io_form_boundary                    = 2\n",
    " io_form_boundary                    = 2\n io_form_auxhist24 = 2\n"
    " auxhist24_interval_s = 60, 60, 60,\n io_form_auxhist3 = 0,\n")


def test_namelist_adios2_switches_aux_outputs_the_case_writes():
    txt = wr.render_segment_namelist(NAMELIST_AUXHIST, _dt("2020-06-01_00:00:00"),
                                     _dt("2020-06-01_06:00:00"), 0, _cfg(), 360,
                                     io_streams=_streams(NAMELIST_AUXHIST))
    assert re.search(r"(?m)^\s*io_form_auxhist24\s*=\s*14\b", txt)
    assert re.search(r"(?m)^\s*io_form_auxhist3\s*=\s*0\b", txt)
    legacy = wr.render_segment_namelist(NAMELIST_AUXHIST, _dt("2020-06-01_00:00:00"),
                                        _dt("2020-06-01_06:00:00"), 0, _cfg(), 360)
    assert re.search(r"(?m)^\s*io_form_auxhist24\s*=\s*2\b", legacy)
    line = wr.io_plan_line("adios2", _streams(NAMELIST_AUXHIST), 14,
                           {"convert": [], "existing": []}, None, "/case",
                           ("io_form_auxhist24",))
    assert line.startswith("I/O: ADIOS2 (history/restart/input/boundary/auxhist24)")


def test_a_rejected_conversion_job_is_an_error_not_a_traceback(tmp_path, fake_bp, capsys):
    case = _make_case(tmp_path)

    def sbatch(cmd):
        raise RuntimeError("sbatch: error: Invalid account or account/partition combination")
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--out",
                  str(tmp_path / "out"), "--poll-interval", "0"], sbatch_runner=sbatch,
                 sacct_runner=lambda c: "", sleep=lambda s: None, log=lambda m: None,
                 scancel_runner=lambda j: None, convert_in_job=True)
    assert rc == 2 and "Invalid account" in capsys.readouterr().err and fake_bp.calls == []


def test_bp_cache_is_the_tunes_in_a_project_with_a_case_id(tmp_path, monkeypatch):
    """f1-arm marshall 2026-10-21: tune's input/bp-cache and run's input/bp each held the same
    3 GB conversion."""
    import hashlib
    proj, case = tmp_path / "proj", tmp_path / "proj" / "case"
    case.mkdir(parents=True)
    (proj / "project.yaml").write_text("name: p\ncase:\n  id: marshall\n")
    monkeypatch.setenv("WTUNE_PROJECT", str(proj))
    monkeypatch.delenv("WTUNE_BP_CACHE", raising=False)
    key = hashlib.sha1(os.path.realpath(str(case)).encode()).hexdigest()[:12]
    shared = proj / "input" / "bp-cache" / "marshall" / key
    out = str(tmp_path / "out")
    assert wr.bp_cache_dir(out, None, str(case)) == str(shared)
    monkeypatch.setenv("WTUNE_BP_CACHE", str(tmp_path / "env"))    # as `wtune tune` honours it
    assert wr.bp_cache_dir(out, None, str(case)) == str(tmp_path / "env" / "marshall" / key)
    monkeypatch.delenv("WTUNE_BP_CACHE")
    legacy = proj / "input" / "bp"
    legacy.mkdir(parents=True)
    (legacy / "bp_manifest.json").write_text("{}")
    assert wr.bp_cache_dir(out, None, str(case)) == str(legacy)    # an older project's run
    shared.mkdir(parents=True)
    (shared / "bp_manifest.json").write_text("{}")
    assert wr.bp_cache_dir(out, None, str(case)) == str(shared)


def test_fresh_copy_falls_back_to_cp_a_without_reflink(tmp_path, fake_bp, capsys):
    fake_bp.no_reflink = True
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0
    assert [c[:3] for c in fake_bp.cp_calls[:2]] == [["cp", "-a", "--reflink=always"],
                                                      ["cp", "-a", fake_bp.cp_calls[1][2]]]
    assert "WARNING: no reflink copy of" in capsys.readouterr().err
    assert any("fresh per-record inputs" in ln and "cp -a" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pre-created output dirs (NFS CREATE stalls on rank 0, scc jobs 2962-2977)
# ---------------------------------------------------------------------------
def _nl(tmp_path, **kv):
    text = NAMELIST_3DOM
    for key, val in kv.items():
        text = re.sub(r"(?m)^ %s\s*=.*$" % key, " %s = %s" % (key, val), text)
    p = tmp_path / "nl.input"
    p.write_text(text)
    return wp.parse_namelist(str(p))


T0 = datetime.datetime(2020, 6, 1, 0)


class _Spy(Fakes):
    """Records the empty dirs present at submit and what restart_files sees after the job."""
    def sbatch(self, cmd):
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        self.seen = getattr(self, "seen", {})
        self.seen[os.path.basename(seg)] = sorted(
            n for n in os.listdir(seg) if n.startswith(("wrfout", "wrfrst")))
        return super().sbatch(cmd)


def test_a_segment_runs_wrf_tree_not_what_a_former_run_dir_brought(tmp_path):
    """atlas-b cases 2026-10-14: the case was a former run directory holding wrf.exe and table
    links of the WRF it ran with; linked first, they shadowed WRF_TREE's."""
    from wtune.run import staging

    def tree(name):
        root = tmp_path / name
        for d in ("main", "Registry", "run"):
            (root / d).mkdir(parents=True)
        (root / "main" / "wrf.exe").write_text(name)
        (root / "run" / "wrf.exe").symlink_to(root / "main" / "wrf.exe")
        (root / "run" / "LANDUSE.TBL").write_text(name)
        return root
    new, old = tree("new"), tree("old")
    (old / "run" / "ONLY_OLD.TBL").write_text("old")
    case, seg = tmp_path / "case", tmp_path / "seg"
    case.mkdir()
    seg.mkdir()
    (case / "wrfinput_d01").write_text("in")
    (case / "wrf.exe").symlink_to(old / "main" / "wrf.exe")
    (case / "real.exe").write_text("a copied executable")
    (case / "LANDUSE.TBL").symlink_to(old / "run" / "LANDUSE.TBL")     # left over: the tree's wins
    (case / "ONLY_OLD.TBL").symlink_to(old / "run" / "ONLY_OLD.TBL")   # WRF_TREE has none: kept
    (case / "VEGPARM.TBL").write_text("the case's own table")
    staging._link_case_files(str(case), str(seg), [], False, str(new))
    staging._link_wrf_run_files(str(new), str(seg))
    assert (seg / "wrf.exe").read_text() == "new" and not (seg / "real.exe").exists()
    assert (seg / "LANDUSE.TBL").read_text() == "new"
    assert (seg / "ONLY_OLD.TBL").read_text() == "old"
    assert (seg / "VEGPARM.TBL").read_text() == "the case's own table"
    assert (seg / "wrfinput_d01").read_text() == "in"


# ---------------------------------------------------------------------------
# stall detection
# ---------------------------------------------------------------------------
def _stall(tmp_path, stall_s=600.0):
    rsl = tmp_path / "rsl.error.0000"
    rsl.write_text("starting\n")
    clock, cancelled, logs = [0.0], [], []
    w = wr.attempts.StallWatch(str(rsl), "123", stall_s, lambda: clock[0], cancelled.append,
                               logs.append)
    return w, rsl, clock, cancelled


def test_stall_watch_cancels_once_when_rsl_stops_changing(tmp_path):
    w, rsl, clock, cancelled = _stall(tmp_path)
    for t in (0, 300, 599):
        clock[0] = t
        w("RUNNING")
    assert cancelled == []
    clock[0] = 600
    w("RUNNING")
    w("RUNNING")
    assert cancelled == ["123"] and "unchanged for 10 min" in w.why


def test_a_stalled_attempt_stops_the_run_without_resuming(tmp_path):
    rec = wr.attempts.AttemptRecord(dir=str(tmp_path), job_id="7", time_step=None,
                                    submitted_utc="x")
    rec.stalled = "rsl.error.0000 unchanged for 20 min"
    out = wr.attempts.Outcome(str(tmp_path), rec, "CANCELLED", 1200.0, {}, False)
    with pytest.raises(wr.RunError, match="stalled and was cancelled"):
        wr.controller._next_attempt(None, 0, None, None, out)
    assert rec.failure_class == "stalled"


def test_compress_history_cleanup_with_a_restart_link_between_segments(tmp_path, monkeypatch):
    """seg_1 starts from a link to seg_0's restart (f1-arm severe2023 2026-10-21: the removed
    link was then sized as a file -> FileNotFoundError after a run whose segments all passed)."""
    import types
    real = tmp_path / "seg_0" / "wrfrst_d01_2020-06-01_06:00:00"
    real.mkdir(parents=True)
    (real / "data.0").write_bytes(b"x" * 100)
    (tmp_path / "seg_1").mkdir()
    (tmp_path / "seg_1" / "wrfrst_d01_2020-06-01_06:00:00").symlink_to(real)
    ctx = types.SimpleNamespace(plan=types.SimpleNamespace(out=str(tmp_path)), report={},
                                log=lambda s: None)
    monkeypatch.setattr(wr.controller, "_save_report", lambda c: None)
    wr.controller._delete_own_restarts(ctx)
    assert ctx.report["restarts_deleted"] == {"files": 1, "bytes": 100}
    assert not list(tmp_path.glob("seg_*/wrfrst_*"))


def test_a_restart_case_starts_segment_0_from_its_own_wrfrst(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(NAMELIST_3DOM.replace(".false.,", ".true.,", 1))
    for name in ["wrfbdy_d01"] + ["wrfrst_d%02d_2020-06-01_00:00:00" % d for d in (1, 2, 3)]:
        (case / name).write_text("x")
    f = Fakes()
    rc, out, rep, lines = _run(tmp_path, f, ["--restart-io-form", "14"])
    assert rc == 0 and rep["status"] == "completed"
    seg0 = out / "seg_0"
    assert os.path.realpath(seg0 / "wrfrst_d02_2020-06-01_00:00:00") == str(
        case / "wrfrst_d02_2020-06-01_00:00:00")
    tc = wp.parse_namelist(str(seg0 / "namelist.input"))["time_control"]
    assert tc["restart"] == [True] and tc["io_form_restart"] == [2]   # netCDF wrfrst: io_form 2
    assert any("io_form_restart 2" in l for l in lines)


def test_output_dir_names_only_on_the_domains_time_step(tmp_path):
    # haitang2017 (2026-10-26): d01 dt 270 s, history 60 min -> WRF wrote wrfout_d01_..._21:04:30
    # while the pre-created on-hour directories stayed empty (left behind by a stopped run)
    nl = _nl(tmp_path, time_step="270,", history_interval="60, 60, 60", restart_interval="180",
             io_form_history="14", io_form_restart="14")
    got = wr.output_dir_names(nl, T0, T0 + datetime.timedelta(hours=6), 1)
    assert [n for n in got if n.startswith("wrfout")] == [
        "wrfout_d01_2020-06-01_00:00:00", "wrfout_d01_2020-06-01_03:00:00",
        "wrfout_d01_2020-06-01_06:00:00"]
    assert [n for n in got if n.startswith("wrfrst")] == [
        "wrfrst_d01_2020-06-01_03:00:00", "wrfrst_d01_2020-06-01_06:00:00"]
    nl["domains"]["use_adaptive_time_step"] = [True]
    assert wr.output_dir_names(nl, T0, T0 + datetime.timedelta(hours=6), 1) == []


def test_an_out_outside_the_project_is_linked_into_runs(tmp_path, monkeypatch):
    # mock competition 2026-10-26: `wtune run --out /work2/...` from the project; `wtune stop`
    # (project runs/*/run.state.json) then found nothing to stop
    import wtune.run.cli as run_cli
    import wtune.stop
    project = tmp_path / "proj"
    project.mkdir()
    (project / "project.yaml").write_text("name: p\n")
    out = tmp_path / "work2" / "restart-run"
    out.mkdir(parents=True)
    (out / "run.state.json").write_text("{}")
    monkeypatch.chdir(project)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    opts = type("O", (), dict(project=None, out=str(out)))()
    link = run_cli.link_into_project(opts, str(out), lambda *a: None)
    assert link == str(project / "runs" / "restart-run")
    assert os.path.realpath(link) == str(out)
    assert run_cli.link_into_project(opts, str(out), lambda *a: None) == link
    found = [p for pat in wtune.stop.STATES for p in glob.glob(str(project / pat))]
    assert found == [str(project / "runs" / "restart-run" / "run.state.json")]
    inside = project / "runs" / "20261026"
    inside.mkdir()
    assert run_cli.link_into_project(type("O", (), dict(project=None, out=str(inside)))(),
                                     str(inside), lambda *a: None) is None
