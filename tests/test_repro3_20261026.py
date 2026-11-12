"""Independent reproductions of eight WTune defects (2026-10-26). Behaviour only: real entry
points (wtune run / tune / submit main, the preflight, the report and predict tables), the
generated segment script executed by bash, fakes only at the process boundary (sbatch, sacct,
srun, the adios2 interpreter, the quota reader). No Slurm job, no ~/wtune-runs."""
import getpass
import os
import shlex
import shutil
import subprocess
import textwrap

import pytest

import wtune.machine
import wtune.project
import wtune.run as wr
import wtune.submit
import wtune.tune.cli as tcli
import wtune.tune.preflight as pf
from wtune.run import cli as run_cli
from wtune.tools import scc_sweep as sweep

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_NL = os.path.join(REPO, "tests", "fixtures", "golden", "configs", "prep_conus12",
                          "namelist.input")
TABLES = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat")
CC = sweep.campaign_configs


# --------------------------------------------------------------------------- shared fixtures
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("WTUNE_BP_CACHE", raising=False)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.delenv("WTUNE_TABLE_CACHE", raising=False)
    monkeypatch.chdir(tmp_path)
    saved = dict(os.environ)          # the tune exports WTUNE_BP_CACHE, SITE, ... into os.environ
    yield h
    os.environ.clear()
    os.environ.update(saved)


def _write(path, text, mode=None):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w") as f:
        f.write(text)
    if mode:
        os.chmod(str(path), mode)
    return str(path)


def _src_case(tmp_path, name="case_src"):
    """A one-domain 6 h case (conus12 fixture namelist) with netCDF-named inputs."""
    d = tmp_path / name
    d.mkdir()
    shutil.copy(FIXTURE_NL, str(d / "namelist.input"))
    for n in ("wrfinput_d01", "wrfbdy_d01"):
        with open(str(d / n), "wb") as f:
            f.write(b"CDF\x01" + b"\0" * 4096)
    return str(d)


FAKE_WRF = r"""#!/usr/bin/env bash
# stand-in for wrf.exe: records the Thompson table links it starts with, computes and writes
# the tables that do not exist (as Thompson does at init), then writes rsl of 4 ranks
for t in __TABLES__; do
  [ -L "$t" ] && echo "$t -> $(readlink "$t")" >> tables_at_start.txt
done
for t in __TABLES__; do
  if [ ! -e "$t" ]; then
    { echo "computed $t" > "$t"; } 2>/dev/null || {
      echo "FATAL CALLED FROM FILE: module_mp_thompson.F  Error writing $t" >> rsl.error.0000
      exit 1; }
  fi
done
for r in 0000 0001 0002 0003; do
  yes "rank $r d01 2020-01-01_00:01:12 timing line for a compressible rsl" | head -n 3000 > rsl.out.$r
  yes "rank $r error-stream line" | head -n 500 > rsl.error.$r
done
if [ "${FAKE_WRF_MODE:-ok}" = fail ]; then
  echo "FATAL CALLED FROM FILE: <stdin>  LINE: 1  fake failure" >> rsl.error.0000
  exit 1
fi
echo "d01 2020-01-01_06:00:00 wrf: SUCCESS COMPLETE WRF" >> rsl.error.0000
""".replace("__TABLES__", " ".join(TABLES))


class Site:
    """A fake campaign dir (sites/<site>.sh launching FAKE_WRF), env source and WRF tree."""

    def __init__(self, root):
        self.root = str(root)
        self.tree = os.path.join(self.root, "WRF")
        self.wrf = _write(os.path.join(self.tree, "main", "wrf.exe"), FAKE_WRF, 0o755)
        _write(os.path.join(self.tree, "run", "LANDUSE.TBL"), "landuse\n")
        self.campaign = os.path.join(self.root, "campaign")
        _write(os.path.join(self.campaign, "sites", wtune.machine.site() + ".sh"), textwrap.dedent("""\
            W=%s
            site_env () { :; }
            site_hygiene () { SITE_HYGIENE_CMD=(true); }
            site_launch () { SITE_LAUNCH_CMD=(bash "$W/main/wrf.exe"); }
            """ % shlex.quote(self.tree)))
        self.env_source = _write(os.path.join(self.root, "env.sh"), ": fake env\n")
        self.args = ["--campaign-dir", self.campaign, "--env-source", self.env_source,
                     "--wrf-tree", self.tree]


class BashSlurm:
    """sbatch runs the segment script with bash right away; sacct reports its exit."""

    def __init__(self, env=None):
        self.env, self.rc, self.scripts, self.next = dict(env or {}), {}, [], 500

    def sbatch(self, cmd):
        path = shlex.split(cmd)[-1]
        self.scripts.append(path)
        self.next += 1
        env = {k: v for k, v in os.environ.items() if k not in ("WRF_TREE", "CAMPAIGN")}
        env.update(self.env, SLURM_JOB_ID=str(self.next), SLURM_JOB_NUM_NODES="1")
        r = subprocess.run(["bash", path], cwd=os.path.dirname(path), env=env,
                           capture_output=True, text=True, timeout=120)
        with open(os.path.join(os.path.dirname(path), "segment-%d.out" % self.next), "w") as f:
            f.write(r.stdout + r.stderr)
        self.rc[str(self.next)] = r.returncode
        return "Submitted batch job %d\n" % self.next

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1] if isinstance(cmd, list) else cmd.split("-j")[1].split()[0]
        state = "COMPLETED" if self.rc.get(jid) == 0 else "FAILED"
        return "%s|%s|00:01:00\n" % (jid, state)

    @staticmethod
    def snapshot(rundir):
        try:
            with open(os.path.join(rundir, "rsl.error.0000")) as f:
                done = "SUCCESS COMPLETE WRF" in f.read()
        except OSError:
            done = False
        return {"status": "done" if done else "failed", "warnings": []}


def _run_main(argv, slurm):
    lines = []
    rc = wr.main(argv + ["--poll-interval", "0", "--stall-minutes", "0"],
                 sbatch_runner=slurm.sbatch, sacct_runner=slurm.sacct,
                 snapshot_fn=slurm.snapshot, sleep=lambda s: None, log=lines.append,
                 scancel_runner=lambda j: None)
    return rc, lines


# --------------------------------------------------------------------------- fake adios2
FAKE_ADIOS2_PY = r"""#!/usr/bin/env bash
# stand-in for the envs/adios2 interpreter: `-c "import adios2"` succeeds; nc2bp.py SRC DST
# [--engine E] [--split-prefix P] writes a BP4-shaped directory (per record with a prefix)
[ "$1" = "-c" ] && exit 0
src="$2"; dst="$3"; prefix=""
shift 3
while [ $# -gt 0 ]; do [ "$1" = "--split-prefix" ] && prefix="$2"; shift; done
echo "$src -> $dst" >> "$FAKE_NC2BP_LOG"
if [ -n "$prefix" ]; then
  mkdir -p "$dst/${prefix}_2020-01-01_00:00:00.bp"
  : > "$dst/${prefix}_2020-01-01_00:00:00.bp/md.idx"; : > "$dst/${prefix}_2020-01-01_00:00:00.bp/data.0"
else
  mkdir -p "$dst"; : > "$dst/md.idx"; : > "$dst/data.0"
fi
"""


@pytest.fixture
def fake_adios2(tmp_path, monkeypatch):
    log = tmp_path / "nc2bp.log"
    log.write_text("")
    py = _write(tmp_path / "fake-adios2-python", FAKE_ADIOS2_PY, 0o755)
    monkeypatch.setenv("WTUNE_ADIOS2_PYTHON", py)
    monkeypatch.setenv("FAKE_NC2BP_LOG", str(log))
    return log


def _conversions(log):
    return [ln for ln in log.read_text().splitlines() if ln.strip()]


def _project(tmp_path, monkeypatch):
    """A real `wtune project init --case` project; the process sees its case table as the
    `wtune` launcher arranges (CASES_JSON with the project's case)."""
    src = _src_case(tmp_path)
    p = str(tmp_path / "proj")
    wtune.project.init_project(p, case_dir=src)
    doc = wtune.project.load_project(p)
    case_id = wtune.project.project_case_id(doc)
    assert case_id
    merged = wtune.project.merged_case_table(p, case_id)
    monkeypatch.setattr(CC, "CASES_JSON_PATH", merged)
    monkeypatch.setitem(CC.CASES, case_id, CC.load_cases(merged)[case_id])
    return p, case_id, merged


def _tune_converts(case_id, merged, project_dir, monkeypatch, root=None):
    """What the tune does for this project: WTUNE_BP_CACHE=<project>/input/bp-cache (or the
    user's), the case's plan, its conversion into that cache. -> the cache dir."""
    with monkeypatch.context() as m:
        if root is None:
            m.setenv("WTUNE_BP_CACHE", os.path.join(project_dir, "input", "bp-cache"))
        _res, cache, _stale = CC._case_bp_plan(case_id, merged)
        case = CC.load_cases(merged)[case_id]
        with open(os.path.join(case["prep_dir"], "namelist.input")) as f:
            text = f.read()
        CC.resolve_bp_inputs(text, case["prep_dir"], cache, convert=True)
    assert os.path.isfile(os.path.join(cache, "bp_manifest.json"))
    return cache


# =========================================================================== defect 1
def test_d1_run_reuses_the_tunes_bp_cache(tmp_path, monkeypatch, home, fake_adios2):
    p, case_id, merged = _project(tmp_path, monkeypatch)
    cache = _tune_converts(case_id, merged, p, monkeypatch)
    assert cache.startswith(os.path.join(p, "input", "bp-cache"))
    tune_conversions = _conversions(fake_adios2)
    assert len(tune_conversions) == 2
    site = Site(tmp_path / "site")
    rc, lines = _run_main(["--project", p, "--out", str(tmp_path / "out"), "--nodes", "1",
                           "--nx", "2", "--ny", "2", "--resume", "0"] + site.args,
                          BashSlurm())
    assert rc == 0, "\n".join(lines)
    assert _conversions(fake_adios2) == tune_conversions, "wtune run converted the inputs again"
    assert not os.path.exists(os.path.join(p, "input", "bp", "bp_manifest.json"))
    link = os.path.join(str(tmp_path / "out"), "seg_0", "wrfinput_d01.bp")
    assert os.path.realpath(link).startswith(os.path.realpath(cache) + os.sep)


# =========================================================================== defect 2
class Submitted(Exception):
    pass


def _tune(p, case_id, tmp_path, extra=()):
    args = tcli.build_arg_parser().parse_args(
        [case_id, "-O1", "--no-node-preflight", "--project", p,
         "--out", str(tmp_path / "tune-out")] + list(extra))
    args.site = args.site or wtune.machine.site()
    sbatch_calls = []

    def sbatch(cmd):
        sbatch_calls.append(cmd)
        raise Submitted(cmd)
    try:
        tcli.run(args, sbatch_runner=sbatch, sacct_runner=lambda c: "",
                 scancel_runner=lambda j: None)
        outcome = "returned"
    except SystemExit as e:
        outcome = "exit: %s" % e
    except Submitted:
        outcome = "submitted"
    return outcome, sbatch_calls


@pytest.fixture
def tiny_quota(monkeypatch):
    """The cache filesystem has 1 MB writable (a GPFS soft quota nearly reached)."""
    import wtune.run.space as space
    monkeypatch.setattr(space, "writable_bytes", lambda path, runner=None: (1.0e6, "GPFS quota"))
    monkeypatch.setattr(run_cli, "writable_bytes", lambda path, runner=None: (1.0e6, "GPFS quota"))


def _big_inputs(p):
    """Real netCDF inputs holding 4 MB of variables each (8 MB of BP to write)."""
    import netCDF4
    import numpy as np
    for n in ("wrfinput_d01", "wrfbdy_d01"):
        path = os.path.join(p, "case", n)
        os.remove(path)
        with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
            ds.createDimension("x", 1000000)
            ds.createVariable("T", "f4", ("x",))[:] = np.zeros(1000000, "f4")


def test_d2_tune_refuses_a_conversion_that_does_not_fit(tmp_path, monkeypatch, home,
                                                       fake_adios2, tiny_quota):
    p, case_id, _merged = _project(tmp_path, monkeypatch)
    _big_inputs(p)
    outcome, calls = _tune(p, case_id, tmp_path)
    assert outcome.startswith("exit:") and "--no-space-check" in outcome, outcome
    assert calls == [] and _conversions(fake_adios2) == []


def test_d2_tune_already_converted_needs_no_space(tmp_path, monkeypatch, home, fake_adios2,
                                                 tiny_quota):
    p, case_id, merged = _project(tmp_path, monkeypatch)
    _big_inputs(p)
    _tune_converts(case_id, merged, p, monkeypatch)
    outcome, calls = _tune(p, case_id, tmp_path)
    assert outcome == "submitted" and calls, outcome


# =========================================================================== defect 4
def test_d4_preflight_keeps_unallocated_and_cancels_its_timed_out_probe(monkeypatch, tmp_path):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    calls = []

    def runner(argv, timeout_s):
        calls.append(list(argv))
        if argv[0] == "sinfo":
            return 0, "n1 idle\nn2 reserved\nn3 idle\nn4 idle\n"
        if argv[0] == "scancel":
            return 0, ""
        node = argv[argv.index("-w") + 1]
        if node == "n1":
            return 0, "ok"
        if node == "n2":
            return 1, ("srun: error: Unable to allocate resources: Requested node configuration "
                       "is not available")
        if node == "n3":
            return 1, ("srun: job 4921 queued and waiting for resources\nsrun: error: Unable to "
                       "allocate resources: Immediate execution impossible")
        raise subprocess.TimeoutExpired(argv, timeout_s)
    h = pf.node_preflight(runner, case_dir=str(tmp_path), src_dir=str(tmp_path),
                          wrf_exe="/bin/true", requested=4)
    nodes = h["nodes"]
    assert nodes["n1"]["ok"] and nodes["n1"]["probed"]
    for n in ("n2", "n3"):
        assert nodes[n]["ok"] and not nodes[n]["probed"], (n, nodes[n])
    assert "n2" not in h["failed"] and "n3" not in h["failed"]
    srun_n4 = [a for a in calls if a[0] == "srun" and a[a.index("-w") + 1] == "n4"][0]
    names = [a.split("=", 1)[1] for a in srun_n4 if a.startswith("--job-name=")]
    assert names, "the probe job has no name to cancel it by: %s" % srun_n4
    cancels = [a for a in calls if a[0] == "scancel"]
    # n4 timed out twice (the second probe is the retry of an idle node): each one is cancelled
    assert cancels == [["scancel", "--user=" + getpass.getuser(), "--name=" + names[0]]] * 2, cancels


def test_f1arm_numa_per_node_1_maps_by_node_so_two_nodes_fit_their_slots():
    script = ('CAMPAIGN=%s; source "$CAMPAIGN/sites/f1arm.sh" >/dev/null 2>&1; '
              'site_launch 144 72 1 block; echo "${SITE_LAUNCH_CMD[*]}"'
              % shlex.quote(os.path.join(REPO, "campaign")))
    run = lambda numa: subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env=dict(os.environ, SLURM_JOB_NUM_NODES="2", WTUNE_NUMA_PER_NODE=numa)).stdout.strip()
    assert "--map-by ppr:72:node:PE=1" in run("1")      # was ppr:72:numa: 144 ranks on one node
    assert "--map-by ppr:36:numa:PE=1" in run("2")      # the two-die mapping is unchanged


# =========================================================================== defect 7
@pytest.mark.parametrize("resume,ckpt,stall", [("3", "30", "12.5"), ("0", "0", "0")])
def test_d7_submit_passes_run_options_through(tmp_path, monkeypatch, home, resume, ckpt, stall):
    p, _case_id, _merged = _project(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(wtune.submit, "_call", lambda cmd, cwd: called.append(cmd) or 0)
    rc = wtune.submit.main(["--project", p, "--no-post", "--resume", resume,
                            "--checkpoint-minutes", ckpt, "--stall-minutes", stall,
                            "--no-space-check"])
    assert rc == 0
    by_step = {cmd[1]: cmd[2:] for cmd in called}
    run = run_cli.parse_options(by_step["run"])
    assert (run.resume, str(run.checkpoint_minutes), run.stall_minutes, run.no_space_check) == (
        int(resume), ckpt, float(stall), True)
    tune = tcli.build_arg_parser().parse_args(by_step["tune"])
    assert tune.no_space_check is True
