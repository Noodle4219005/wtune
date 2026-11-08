"""Independent reproductions of the 2026-10-26 defect reports (Thompson table cache, build id
in probe ledgers, probe ledger directory, predict cores per node). Each test drives the
generated bash or the Python entry point and checks what it does, not how it is written."""
import csv
import hashlib
import os
import subprocess
import types

import pytest

from wtune.tune.cli import _build_id as REAL_BUILD_ID   # before conftest's stub
from wtune.tune.cli import _probe_tree as REAL_PROBE_TREE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN = os.path.join(REPO, "campaign")
TABLES = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat")


# ---------------------------------------------------------------- helpers

def _bash(script, env, cwd=None, check=True):
    p = subprocess.run(["bash", "-c", script], env=env, cwd=cwd, capture_output=True,
                       text=True, timeout=120)
    if check:
        assert p.returncode == 0, "bash rc=%d\nstdout:\n%s\nstderr:\n%s" % (
            p.returncode, p.stdout, p.stderr)
    return p


def _env(tmp_path, **extra):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
           "USER": os.environ.get("USER", "nobody"),
           "WTUNE_TABLE_CACHE": str(tmp_path / "tblcache")}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _make_tree(root, contents):
    """A WRF tree whose main/wrf.exe holds `contents` (rewritten in place on a rebuild)."""
    os.makedirs(os.path.join(root, "main"), exist_ok=True)
    os.makedirs(os.path.join(root, "run"), exist_ok=True)
    with open(os.path.join(root, "run", "LANDUSE.TBL"), "w") as f:
        f.write("x\n")
    with open(os.path.join(root, "main", "wrf.exe"), "w") as f:
        f.write(contents)
    return str(root)


def _config():
    from wtune.run.layout import Config
    return Config(id="c", nx=2, ny=2, ranks=4, numtiles=1, omp=1, nodes=1)


def _sha16(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- defect 1


# ---------------------------------------------------------------- defect 2


# ---------------------------------------------------------------- defect 3


# ---------------------------------------------------------------- defect 4

def _ledger(path, rows):
    cols = ["case_base", "hardware_id", "s_per_step_median", "n_nodes", "notes",
            "e_we", "e_sn", "e_vert", "n_ranks", "omp_threads", "nproc_x", "nproc_y",
            "numtiles"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _row(case, s, build=None):
    notes = "config=C1 mode=timing io=adios2" + (" build=%s" % build if build else "")
    return {"case_base": case, "hardware_id": "hw1", "s_per_step_median": s, "n_nodes": 1,
            "notes": notes, "e_we": 100, "e_sn": 100, "e_vert": 40, "n_ranks": 4,
            "omp_threads": 1, "nproc_x": 2, "nproc_y": 2, "numtiles": 1}


# ---------------------------------------------------------------- defect 5

def _tune_ctx_kwargs(tmp_path, monkeypatch, argv):
    """What `wtune tune` builds its Ctx from (the real _build_ctx, Ctx itself captured)."""
    from wtune.tune import cli
    captured = {}

    def fake_ctx(**kw):
        captured.update(kw)
        return types.SimpleNamespace(**kw)
    monkeypatch.setattr(cli, "Ctx", fake_ctx)
    args = cli.build_arg_parser().parse_args(["c1"] + argv)
    o = types.SimpleNamespace(case="c1", minutes=10.0, confirm_reps=1, restart_io_form=14,
                              cfg={"search": {}, "exclude": [], "prediction": {}, "probe": {}})
    ns = types.SimpleNamespace(nodes=1, node_seeds=None, hotspot_fill=None)
    nl = tmp_path / "namelist.input"
    nl.write_text("&domains\n/\n")
    cli._build_ctx(args, o, ns, {}, str(nl), False, 10, None, 4, str(tmp_path / "tout"),
                   None, None, None, None, None, None)
    return captured


def test_d4_tune_filters_on_the_machine_wrf_build(tmp_path, monkeypatch):
    from wtune.tools import scc_sweep as sweep
    tree = _make_tree(tmp_path / "WRF", "build-1")
    monkeypatch.setenv("WRF_TREE", tree)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    from wtune.tune import cli
    monkeypatch.setattr(cli, "_build_id", REAL_BUILD_ID)    # conftest stubs it for hermeticity
    monkeypatch.setattr(cli, "_probe_tree", REAL_PROBE_TREE)
    kw = _tune_ctx_kwargs(tmp_path, monkeypatch, ["--ledger-dir", str(tmp_path / "l"),
                                                  "--env-source", str(tmp_path / "no-env.sh")])
    assert kw.get("build_id"), "the tune does not know which WRF build it measures"
    assert kw["build_id"] == sweep.wrf_build_id(tree)


# ---------------------------------------------------------------- defect 6


def _launch_retry_block():
    """The probe launch loop of campaign/strict_campaign.sbatch (`for ATTEMPT in 1 2` .. done)."""
    lines = open(os.path.join(CAMPAIGN, "strict_campaign.sbatch")).read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "for ATTEMPT in 1 2; do")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "  done")
    return "\n".join(lines[start:end + 1])


@pytest.mark.parametrize("crashes,rc,retries", [(1, 0, 1), (2, 139, 1), (0, 0, 0)])
def test_a_probe_killed_by_sigsegv_runs_once_more(tmp_path, crashes, rc, retries):
    # haitang2017 (2026-10-27): probes crashed intermittently on SIGSEGV at the first step; the same
    # configuration ran on the next try, but tune had dropped every node-sweep anchor
    fake = tmp_path / "fake_mpirun.sh"
    fake.write_text("n=$(cat count 2>/dev/null || echo 0); n=$((n+1)); echo $n > count\n"
                    "if [ $n -le %d ]; then echo 'rank 54 exited on signal 11'; exit 139; fi\n"
                    "echo 'd01 SUCCESS COMPLETE WRF' > rsl.error.0000\n" % crashes)
    script = ("site_launch () { SITE_LAUNCH_CMD=(bash %s); }\n"
              "ranks=4 ntasks_per_node=4 omp=1 dist=block cid=c rep=1 CASE=x J=1 TIMEOUT_EFF=60\n"
              "%s\necho RC=$RC\n" % (fake, _launch_retry_block()))
    p = _bash(script, _env(tmp_path), cwd=str(tmp_path))
    assert "RC=%d" % rc in p.stdout
    assert p.stdout.count("RETRY cfg=c") == retries
    assert os.path.exists(tmp_path / "wrf.out.attempt1") == bool(retries)
