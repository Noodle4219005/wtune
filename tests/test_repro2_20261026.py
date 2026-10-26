"""Independent reproductions of the 2026-10-26 defect reports (4df97ca -> 1bc6509).

Each test drives the real entry point (wtune.tune.run, the round/prediction functions, the
generated bash of wtune calibrate plan, strict_campaign.sbatch's harvest function, wtune run's
main) with fakes only at the Slurm / WRF boundary. Nothing is submitted; HOME is a tmp dir."""
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import types

import pytest

_SITE_BEFORE = os.environ.get("SITE")
_CONFIGS_BEFORE = sys.modules.get("configs")

import wtune.tune
import wtune.tune.cli
import wtune.tune.context
import wtune.tune.prediction
import wtune.tune.rounds
import wtune.tools.scc_sweep as sweep
import wtune.calibrate
from wtune import predict as wp
from wtune import run as wr
import wtune.watch

if _SITE_BEFORE is None:
    os.environ.pop("SITE", None)
if _CONFIGS_BEFORE is None:
    sys.modules.pop("configs", None)

# captured before tests/conftest.py's autouse fixture stubs it (absent on OLD)
_REAL_BUILD_ID = getattr(wtune.tune.cli, "_build_id", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN = os.path.join(REPO, "campaign")
TABLES3 = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat")


def _sha16(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _exe(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture(autouse=True)
def _restore_campaign_out():
    old = os.environ.get("WTUNE_CAMPAIGN_OUT")
    yield
    if old is None:
        os.environ.pop("WTUNE_CAMPAIGN_OUT", None)
    else:
        os.environ["WTUNE_CAMPAIGN_OUT"] = old


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


# ---------------------------------------------------------------------------------------------
# wtune tune -O0 harness (defects 1 and 3)
# ---------------------------------------------------------------------------------------------
BIG_GRID = dict(e_we=400, e_sn=400, e_vert=45)
ALL_NODES_PROFILE = {"core_count_priors": {"efficiency_table": [
    {"nodes": n, "legal": True, "efficiency_E": 0.9} for n in (1, 2, 4, 8, 13)]}}


def _o0_harness(monkeypatch, tmp_path):
    seen = []
    case_json = dict(BIG_GRID, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"),
                                           ALL_NODES_PROFILE))
    monkeypatch.setattr(wtune.tune.context, "_total_steps", lambda nl: 1000)

    def cold_start(ctx, nodes, cfg, steps):
        seen.append(dict(build_id=ctx.build_id, ledger_dir=ctx.ledger_dir,
                         campaign_out=os.environ.get("WTUNE_CAMPAIGN_OUT")))
        return 1000.0, 0.05, "stub cold start"
    monkeypatch.setattr(wtune.tune.prediction, "cold_start_prediction", cold_start)

    def no_sbatch(cmd):
        raise AssertionError("-O0 must not submit")

    def tune(argv):
        args = wtune.tune.build_arg_parser().parse_args(["matthew", "-O0"] + argv)
        wtune.tune.run(args, sbatch_runner=no_sbatch, clock=FakeClock(),
                       preflight_runner=lambda argv, timeout: None)
        return seen[-1]
    return tune


# --- defect 1 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("site", ["scc", "f1"])
def test_tune_build_id_is_the_wrf_exe_the_probe_jobs_run(monkeypatch, tmp_path, site):
    """machine.yaml's wrf_tree (exported as WRF_TREE) names tree A; the submit shell's env.sh
    exports WRF_TREE=B (scc: W=${WRF_TREE:-...} -> B); f1.sh hard-codes W under $HOME."""
    if _REAL_BUILD_ID is not None:
        monkeypatch.setattr(wtune.tune.cli, "_build_id", _REAL_BUILD_ID)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    tree_a, tree_b = tmp_path / "treeA", tmp_path / "treeB"
    _exe(str(tree_a / "main" / "wrf.exe"), "machine.yaml build\n")
    _exe(str(tree_b / "main" / "wrf.exe"), "env.sh build\n")
    f1_tree = home / "wrf-stack" / "wrf" / "intel-adios2-dmsm" / "WRF"
    _exe(str(f1_tree / "main" / "wrf.exe"), "f1 site build\n")
    env_sh = tmp_path / "env.sh"
    env_sh.write_text("export WRF_TREE=%s\n" % tree_b)
    monkeypatch.setenv("WRF_TREE", str(tree_a))
    monkeypatch.setenv("WTUNE_SITE", site)
    monkeypatch.chdir(tmp_path)
    tune = _o0_harness(monkeypatch, tmp_path)
    got = tune(["--site", site, "--out", str(tmp_path / "out"), "--ledger-dir",
                str(tmp_path / "ledger"), "--env-source", str(env_sh)])
    # what the jobs hash: the site file's $W after sourcing env.sh, resolved by bash itself
    w = subprocess.run(["bash", "-c", 'source "$1" >/dev/null 2>&1; CAMPAIGN="$2"; '
                        'source "$CAMPAIGN/sites/$3.sh" >/dev/null 2>&1; printf %s "$W"',
                        "x", str(env_sh), CAMPAIGN, site],
                       capture_output=True, text=True, env=dict(os.environ)).stdout
    with open(os.path.join(w, "main", "wrf.exe"), "rb") as f:
        job_build = _sha16(f.read())
    with open(tree_a / "main" / "wrf.exe", "rb") as f:
        machine_build = _sha16(f.read())
    assert job_build != machine_build          # the scenario: the two trees differ
    assert got["build_id"] == job_build


# --- defect 3 ---------------------------------------------------------------------------------
def test_second_tune_in_process_does_not_inherit_first_tunes_ledger_dir(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.delenv("WTUNE_CAMPAIGN_OUT", raising=False)
    monkeypatch.chdir(tmp_path)
    tune = _o0_harness(monkeypatch, tmp_path)
    first_dir = str(tmp_path / "first-ledger")
    first = tune(["--site", "scc", "--out", str(tmp_path / "out1"), "--ledger-dir", first_dir])
    assert first["ledger_dir"] == first_dir
    assert first["campaign_out"] == first_dir          # the probe jobs write where it reads
    assert "WTUNE_CAMPAIGN_OUT" not in os.environ      # nothing left behind after the tune
    second = tune(["--site", "scc", "--out", str(tmp_path / "out2")])
    assert second["ledger_dir"] == str(home / "wtune-runs" / "campaign")


def test_tune_restores_a_preexisting_campaign_out(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("WTUNE_CAMPAIGN_OUT", str(tmp_path / "users-own"))
    monkeypatch.chdir(tmp_path)
    tune = _o0_harness(monkeypatch, tmp_path)
    got = tune(["--site", "scc", "--out", str(tmp_path / "out"), "--ledger-dir",
                str(tmp_path / "explicit")])
    assert got["campaign_out"] == str(tmp_path / "explicit")
    assert os.environ.get("WTUNE_CAMPAIGN_OUT") == str(tmp_path / "users-own")


# ---------------------------------------------------------------------------------------------
# defect 2: s/step estimates keep to the build and read the history ledgers
# ---------------------------------------------------------------------------------------------
BUILD_A, BUILD_B = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"
FIELDS = ["case_base", "hardware_id", "n_nodes", "s_per_step_median", "notes"]


def _ledger(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s, build in rows:
            w.writerow(dict(case_base="matthew", hardware_id="scc-gh200", n_nodes=2,
                            s_per_step_median=s, notes=("build=%s" % build) if build else "x=1"))


# (ledger rows, history rows, expected s/step at 2 nodes for build A)
SCENARIOS = {
    "same-build-plus-history": ([(4.0, BUILD_A), (40.0, BUILD_B), (40.0, BUILD_B)],
                                [(6.0, BUILD_A)], 5.0),
    "unknown-build-fallback": ([(40.0, BUILD_B), (40.0, BUILD_B), (8.0, "")], [(10.0, "")], 9.0),
}


def _ctx(tmp_path, scenario):
    rows, hist, expected = SCENARIOS[scenario]
    ledger, history = tmp_path / "ledger", tmp_path / "history"
    _ledger(str(ledger / "ledger_matthew_2001.csv"), rows)
    _ledger(str(history / "ledger_matthew_1001.csv"), hist)
    ctx = wtune.tune.Ctx(
        case="matthew", case_json=dict(BIG_GRID, prep_dir=str(tmp_path / "case")),
        namelist_path=str(tmp_path / "case" / "namelist.input"), nodes=2, cores_per_node=72,
        hardware_id="scc-gh200", steps=20, seed=1, kappa=0.5, budget_s=1800.0, max_nodes=13,
        out_dir=str(tmp_path / "out"), ledger_dir=str(ledger), env_source=str(tmp_path / "env.sh"),
        sbatch_script=os.path.join(CAMPAIGN, "strict_campaign.sbatch"), sacct_cmd="sacct",
        poll_interval=0.0, sbatch_runner=lambda c: pytest.fail("no submission expected"),
        sacct_runner=lambda c: "", scancel_runner=lambda j: None,
        ledger_reader=lambda case, jobs: [], proposer=None, clock=FakeClock(),
        build_id=BUILD_A, history_dirs=[str(history)])
    return ctx, expected


class _Seen(Exception):
    pass


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_round0_sizes_off_this_builds_rows_and_history(monkeypatch, tmp_path, scenario):
    ctx, expected = _ctx(tmp_path, scenario)
    seen = []

    def spy(ctx_, s_est, slots):
        seen.append(s_est)
        raise _Seen()
    monkeypatch.setattr(wtune.tune.rounds, "_round0_size", spy)
    with pytest.raises(_Seen):
        wtune.tune.rounds.run_round0(ctx)
    assert seen == [pytest.approx(expected)]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_tuning_at_without_an_estimate_uses_this_builds_rows_and_history(monkeypatch, tmp_path,
                                                                         scenario):
    """The node-search finalist path: run_tuning_at(ctx) with no s_est carries its own
    estimate into the later stages (update_s_est(results, s_est))."""
    ctx, expected = _ctx(tmp_path, scenario)
    seen = []
    monkeypatch.setattr(wtune.tune.rounds, "run_round0", lambda ctx_, fill=False: ([], {}, []))

    def spy(results, s_est):
        seen.append(s_est)
        raise _Seen()
    monkeypatch.setattr(wtune.tune.rounds, "update_s_est", spy)
    with pytest.raises(_Seen):
        wtune.tune.rounds.run_tuning_at(ctx)
    assert seen == [pytest.approx(expected)]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_o0_cold_start_ledger_estimate_uses_this_builds_rows_and_history(tmp_path, scenario):
    ctx, expected = _ctx(tmp_path, scenario)
    total, s_est, src = wtune.tune.prediction.cold_start_prediction(ctx, 2, None, 1000)
    assert s_est == pytest.approx(expected)
    assert total == pytest.approx(1000 * expected + ctx.startup_s)


# ---------------------------------------------------------------------------------------------
# defect 4: strict_campaign.sbatch harvest_thompson_tables
# ---------------------------------------------------------------------------------------------
def _harvest_source():
    with open(os.path.join(CAMPAIGN, "strict_campaign.sbatch")) as f:
        lines = f.read().splitlines()
    names = next(ln for ln in lines if ln.startswith("THOMPSON_TABLE_NAMES="))
    start = lines.index("harvest_thompson_tables () {")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return names + "\n" + "\n".join(lines[start:end + 1]) + "\n"


RSL_DIED_IN_INIT = ("taskid: 0 hostname: work1\n"
                    " Ntasks in X 1 , ntasks in Y 1\n"
                    " THOMPSON_INIT: computing qr_acr_qg\n")
RSL_PAST_INIT = (RSL_DIED_IN_INIT +
                 "Timing for main: time 2020-06-01_00:01:00 on domain   1:    0.51000 elapsed "
                 "seconds\n")


def _harvest(run_dir, cache, path_prefix=None, job="77"):
    env = dict(os.environ)
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env["PATH"]
    script = ("set -uo pipefail\n" + _harvest_source() +
              'TABLE_CACHE="$1"; J="$2"; cd "$3" || exit 9; harvest_thompson_tables\n')
    return subprocess.run(["bash", "-c", script, "harvest", str(cache), job, str(run_dir)],
                          env=env, capture_output=True, text=True)


def _run_dir(path, rsl, tables=("qr_acr_qg_V4.dat", "freezeH2O.dat")):
    path.mkdir(parents=True)
    (path / "rsl.error.0000").write_text(rsl)
    content = {}
    for t in tables:
        data = ("%s computed by this run\n" % t) * 4096
        (path / t).write_text(data)
        content[t] = data
    return content


def test_harvest_caches_only_from_a_run_that_got_past_wrf_init(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    died = tmp_path / "died"
    _run_dir(died, RSL_DIED_IN_INIT)
    r = _harvest(died, cache)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(cache)) == []             # a possibly partial table: not cached

    done = tmp_path / "done"
    content = _run_dir(done, RSL_PAST_INIT)
    r = _harvest(done, cache, job="78")
    assert r.returncode == 0, r.stderr
    for t, data in content.items():                    # a run past init: cached, complete
        assert (cache / t).read_text() == data


def test_harvest_never_exposes_a_partial_table_under_the_cache_name(tmp_path):
    """The run dir and the cache are on different filesystems: there mv(1) and cp(1) copy the
    bytes into the destination. The shims below make that copy die half way (job killed, quota),
    which is what a reader of the cache name could see at any moment of a slow copy."""
    run_root, cache_root = tmp_path / "scratch", tmp_path / "home-fs"
    cache = cache_root / "thompson_tables" / "b1d"
    cache.mkdir(parents=True)
    done = run_root / "run_1"
    content = _run_dir(done, RSL_PAST_INIT)
    shims = tmp_path / "shims"
    shims.mkdir()
    for tool in ("mv", "cp"):
        real = shutil.which(tool)
        _exe(str(shims / tool), """#!/bin/bash
ops=(); for a in "$@"; do case "$a" in -*) ;; *) ops+=("$a");; esac; done
src=$(realpath -m "${ops[0]}"); dst=$(realpath -m "${ops[1]}")
case "$src" in %(run)s/*) case "$dst" in %(cache)s/*)
  # crosses filesystems: copy the data into dst ... and die half way
  head -c $(( $(stat -c %%s "$src") / 2 )) "$src" > "$dst"; exit 1;; esac;; esac
exec %(real)s "$@"
""" % dict(run=run_root, cache=cache_root, real=real))
    _harvest(done, cache, path_prefix=str(shims))
    for t, data in content.items():
        p = cache / t
        assert not p.exists() or p.read_text() == data, "partial %s under the cache name" % t
        assert (done / t).exists() or p.read_text() == data   # and the table was not lost


# ---------------------------------------------------------------------------------------------
# defect 5: wtune calibrate plan's script caches what a completed run computed
# ---------------------------------------------------------------------------------------------
CAL_NAMELIST = """ &time_control
 run_hours = 1,
 start_year = 2020, start_month = 06, start_day = 01, start_hour = 00,
 end_year = 2020, end_month = 06, end_day = 01, end_hour = 01,
 interval_seconds = 21600
 history_interval = 60,
 restart = .false.,
 /
 &domains
 time_step = 60,
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 45,
 /
"""

FAKE_WRF = """#!/bin/bash
# stands in for wrf.exe: Thompson init computes each table that is not already present
for t in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do
  if [ ! -e "$t" ]; then
    for i in $(seq 200); do echo "$t table row $i"; done > "$t"
    echo "$SLURM_JOB_ID $t" >> "$COMPUTE_LOG"
  fi
done
printf 'Timing for main: time 2020-06-01_00:01:00 on domain   1:    0.5 elapsed seconds\\n' > rsl.error.0000
printf 'wrf: SUCCESS COMPLETE WRF\\n' >> rsl.error.0000
"""


def test_calibrate_plan_script_caches_the_tables_a_completed_run_computed(tmp_path, monkeypatch):
    home, case, out = tmp_path / "home", tmp_path / "case", tmp_path / "plan"
    home.mkdir()
    case.mkdir()
    (case / "namelist.input").write_text(CAL_NAMELIST)
    tree = tmp_path / "WRF"
    _exe(str(tree / "main" / "wrf.exe"), FAKE_WRF)
    (tree / "run").mkdir()
    (tree / "run" / "LANDUSE.TBL").write_text("lu\n")
    env_sh = tmp_path / "env.sh"
    env_sh.write_text("export OMP_STACKSIZE=1G\n")
    shims = tmp_path / "shims"
    fakepy = shims / "fakepy"
    _exe(str(fakepy), """#!/bin/bash
case "$2" in
  stage-bp-inputs) echo "BP_DIR=%s"; echo "WTUNE_PER_RECORD_BP=''"; echo "WTUNE_AUX_BP_STREAMS=''";;
  edit-namelist) cp "$3" "$4";;
  *) exit 2;;
esac
""" % (tmp_path / "bp"))
    _exe(str(shims / "pixi"), "#!/bin/bash\necho %s\n" % fakepy)
    _exe(str(shims / "mpirun"), '#!/bin/bash\n[ "${@: -1}" = ./wrf.exe ] && exec ./wrf.exe\n'
                                'exit 0\n')
    monkeypatch.setenv("HOME", str(home))
    assert wtune.calibrate.main(["plan", "--site", "scc", "--case-dir", str(case), "--out",
                                 str(out), "--ranks-list", "1", "--cores-per-node", "72"]) == 0
    script = out / "plan_1.sbatch"
    cache_root = tmp_path / "tblcache"
    log = tmp_path / "computed.log"
    env = dict(PATH="%s:%s" % (shims, os.environ["PATH"]), HOME=str(home),
               WTUNE_ENV_SOURCE=str(env_sh), WTUNE_TABLE_CACHE=str(cache_root), WRF_TREE=str(tree),
               COMPUTE_LOG=str(log))
    for job in ("101", "102"):
        r = subprocess.run(["bash", str(script)], env=dict(env, SLURM_JOB_ID=job),
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        wall = (out / ("run_1_%s" % job) / "wall.txt").read_text()
        assert "OK 1" in wall, wall
    build = _sha16((tree / "main" / "wrf.exe").read_bytes())
    first = out / "run_1_101"
    for t in TABLES3:                              # the first run's tables are in the cache
        assert (cache_root / build / t).read_text() == (first / t).read_text()
    computed = log.read_text().split("\n")
    assert sorted(c for c in computed if c) == sorted("101 %s" % t for t in TABLES3)


# ---------------------------------------------------------------------------------------------
# defect 6: wtune run deletes the tables WRF computed into a segment (no --thompson-cache)
# ---------------------------------------------------------------------------------------------
def _load_real_bp():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wtune_bp_real_repro2",
                                                  os.path.join(REPO, "src", "wtune", "bp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_BP = _load_real_bp()


def _fake_bp_module():
    m = types.ModuleType("wtune_bp")
    m.ADIOS2_IO_FORM = 14
    m.require_adios2_python = lambda: "/fake/adios2/python"

    def convert_inputs(case_dir, cache_dir, names=None, engine="BP4"):
        out = {}
        for n in names:
            p = os.path.join(cache_dir, n)
            if REAL_BP.PER_RECORD_NAME_RE.match(n):
                for t in ("2020-06-01_00:00:00", "2020-06-01_06:00:00", "2020-06-01_12:00:00"):
                    q = os.path.join(p, "%s_%s.bp" % (n, t))
                    os.makedirs(q, exist_ok=True)
                    for f in ("md.idx", "data.0"):
                        open(os.path.join(q, f), "w").close()
            else:
                os.makedirs(p, exist_ok=True)
                for f in ("md.idx", "data.0"):
                    open(os.path.join(p, f), "w").close()
            out[n] = p
        return out
    m.convert_inputs = convert_inputs
    m.adios2_xml = lambda engine_by_stream=None: (
        '<?xml version="1.0"?>\n<adios-config>\n</adios-config>\n')
    m.is_bp = os.path.isdir
    for name in ("PER_RECORD_NAME_RE", "is_per_record_dir", "record_files", "bp_input_links",
                 "per_record_namelist_edits"):
        setattr(m, name, getattr(REAL_BP, name))

    def cp(cmd, **kw):
        shutil.copytree(cmd[-2], cmd[-1], symlinks=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    m.fresh_input_copy = lambda src, dest, stats=None: REAL_BP.fresh_input_copy(
        src, dest, runner=cp, stats=stats)
    return m


RUN_NAMELIST = """ &time_control
 run_days = 0,
 run_hours = 12,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2020,
 start_month = 06,
 start_day = 01,
 start_hour = 00,
 end_year = 2020,
 end_month = 06,
 end_day = 01,
 end_hour = 12,
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
 /

 &physics
 radt = 30,
 /
"""

COMPUTED = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qg_mp38V1.dat")   # WRF writes these
KEPT = ("LANDUSE.TBL", "my_notes.dat", "qr_acr_qg_V4.dat.orig")              # not Thompson tables


class SegmentJob:
    """sbatch: WRF of the segment computes the Thompson tables into the seg dir (and, when the
    outcome is done, writes the restart); sacct reports `state`; snapshot reports the outcome."""

    def __init__(self, state="COMPLETED", outcome="done"):
        self.state, self.outcome, self.segs = state, outcome, []

    def sbatch(self, cmd):
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        self.segs.append(seg)
        for t in COMPUTED + KEPT:
            with open(os.path.join(seg, t), "w") as f:
                f.write("%s written in the job\n" % t)
        if self.outcome == "done":
            info = wtune.watch.run_info(wp.parse_namelist(os.path.join(seg, "namelist.input")))
            for d in range(1, len(info["domains"]) + 1):
                rst = os.path.join(seg, "wrfrst_d%02d_%s" % (d, info["end"].strftime(
                    "%Y-%m-%d_%H:%M:%S")))
                os.makedirs(rst, exist_ok=True)
                open(os.path.join(rst, "md.idx"), "w").close()
        return "Submitted batch job %d\n" % (5000 + len(self.segs))

    def sacct(self, cmd):
        return "%s|%s|00:10:00\n" % (cmd[cmd.index("-j") + 1], self.state)

    def snapshot(self, seg):
        if self.outcome == "done":
            return {"status": "done", "warnings": [], "sum_d01_elapsed_s": 300.0,
                    "n_d01_steps": 720}
        return {"status": "failed", "warnings": [{"code": "FAILED", "text": "forrtl: severe"}]}


def _wtune_run(monkeypatch, tmp_path, job, extra=(), **kw):
    monkeypatch.setitem(sys.modules, "wtune.bp", _fake_bp_module())
    import wtune
    monkeypatch.setattr(wtune, "bp", sys.modules["wtune.bp"], raising=False)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(RUN_NAMELIST)
    for n in ("wrfinput_d01", "wrfbdy_d01"):
        (case / n).write_text("x")
    (case / "qr_acr_qsV2.dat").write_text("the case's own table\n")
    out = tmp_path / "out"
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes",
                  "720", "--out", str(out), "--poll-interval", "0"] + list(extra),
                 sbatch_runner=job.sbatch, sacct_runner=kw.get("sacct", job.sacct),
                 snapshot_fn=job.snapshot, sleep=kw.get("sleep", lambda s: None),
                 log=lines.append, scancel_runner=kw.get("scancel"))
    assert job.segs, lines
    return rc, job.segs[0], case


def _assert_tables_removed(seg, case):
    for t in COMPUTED:
        assert not os.path.lexists(os.path.join(seg, t)), "%s left in %s" % (t, seg)
    own = os.path.join(seg, "qr_acr_qsV2.dat")
    assert os.path.islink(own) and open(own).read() == "the case's own table\n"
    assert (case / "qr_acr_qsV2.dat").exists()
    for t in KEPT:
        assert os.path.isfile(os.path.join(seg, t)), "%s deleted" % t


@pytest.mark.parametrize("state,outcome", [("COMPLETED", "done"), ("FAILED", "failed"),
                                           ("TIMEOUT", "failed"), ("CANCELLED", "failed"),
                                           ("NODE_FAIL", "failed")])
def test_segment_tables_deleted_whatever_the_job_did(monkeypatch, tmp_path, state, outcome):
    rc, seg, case = _wtune_run(monkeypatch, tmp_path, SegmentJob(state, outcome))
    _assert_tables_removed(seg, case)


def test_segment_tables_deleted_when_the_run_is_stopped_during_the_segment(monkeypatch, tmp_path):
    """Ctrl-C / `wtune stop` while the segment's job runs: the controller cancels the job and
    stops. The segment directory stays behind with whatever WRF had computed."""
    import wtune.stopping
    monkeypatch.setitem(wtune.stopping.STOP, "reason", None)
    job, cancelled = SegmentJob("RUNNING", "failed"), []

    def sleep(_s):
        wtune.stopping.STOP["reason"] = "stopped by Ctrl-C"
    rc, seg, case = _wtune_run(monkeypatch, tmp_path, job, sleep=sleep,
                               scancel=cancelled.append)
    assert rc == 130 and cancelled
    _assert_tables_removed(seg, case)


def test_thompson_cache_keeps_the_segment_tables_for_the_harvest(monkeypatch, tmp_path):
    """control: with --thompson-cache the job's own harvest takes them (they stay)."""
    rc, seg, case = _wtune_run(monkeypatch, tmp_path, SegmentJob(), extra=["--thompson-cache"])
    for t in COMPUTED + KEPT:
        assert os.path.isfile(os.path.join(seg, t))
