"""Independent reproductions of the 2026-10-26 defect reports (Thompson table cache, build id
in probe ledgers, probe ledger directory, predict cores per node). Each test drives the
generated bash or the Python entry point and checks what it does, not how it is written."""
import csv
import hashlib
import inspect
import os
import re
import string
import subprocess
import types

import pytest

from wtune.tune.cli import _build_id as REAL_BUILD_ID   # before conftest's stub

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


def _fake_wrf(tmp_path):
    """Stands in for wrf.exe's Thompson init: logs which tables are symlinks when it starts,
    computes (writes) every absent table -- a write through a dangling link into a missing
    directory fails, as WRF's did -- and reports success in rsl.error.0000."""
    path = tmp_path / "fakewrf.sh"
    path.write_text(
        "links=''\n"
        "for t in %s; do [ -L \"$t\" ] && links=\"$links $t\"; done\n"
        "echo \"links=$links\" > wrf_start.log\n"
        "for t in %s; do\n"
        "  if [ ! -e \"$t\" ]; then printf 'computed\\n' > \"$t\" || { echo \"cannot write $t\" >&2; exit 9; }; fi\n"
        "done\n"
        "echo 'd01 SUCCESS COMPLETE WRF' > rsl.error.0000\n" % (" ".join(TABLES), " ".join(TABLES)))
    return str(path)


def _cache_files(root):
    """{relative dir: sorted table names} of every regular table file under the cache root."""
    out = {}
    for d, _sub, files in os.walk(root):
        names = sorted(f for f in files if f in TABLES)
        if names:
            out[os.path.relpath(d, root)] = names
    return out


def _start_links(run_dir):
    return open(os.path.join(run_dir, "wrf_start.log")).read().split("=", 1)[1].split()


def _fake_campaign(tmp_path, tree, fakewrf):
    import wtune.machine
    camp = tmp_path / "campaign"
    (camp / "sites").mkdir(parents=True, exist_ok=True)
    (camp / "sites" / ("%s.sh" % wtune.machine.site())).write_text(
        "W=%s\n"
        "site_env () { :; }\n"
        "site_hygiene () { SITE_HYGIENE_CMD=(true); }\n"
        "site_launch () { SITE_LAUNCH_CMD=(bash %s); }\n" % (tree, fakewrf))
    envsrc = tmp_path / "env.sh"
    envsrc.write_text(":\n")
    return str(camp), str(envsrc)


def _config():
    from wtune.run.layout import Config
    return Config(id="c", nx=2, ny=2, ranks=4, numtiles=1, omp=1, nodes=1)


def _render_segment(seg_dir, camp, envsrc, thompson_cache):
    """render_segment_sbatch with the cache on/off where the option exists (before the fix
    there is no option: the cache is always on)."""
    from wtune.run.sbatch import render_segment_sbatch
    kw = dict(env_source=envsrc, campaign_dir=camp)
    if "thompson_cache" in inspect.signature(render_segment_sbatch).parameters:
        kw["thompson_cache"] = thompson_cache
    return render_segment_sbatch(seg_dir, 1, _config(), "01:00:00", "t", **kw)


def _run_script(script_path, env, cwd):
    p = _bash("bash %s" % script_path, env, cwd=cwd, check=False)
    assert "SEGMENT rc=0" in p.stdout, (p.stdout, p.stderr)
    return p


def _run_segment(tmp_path, name, tree, thompson_cache=True):
    seg = tmp_path / name
    seg.mkdir()
    camp, envsrc = _fake_campaign(tmp_path, tree, _fake_wrf(tmp_path))
    spath = seg / "segment.sbatch"
    spath.write_text(_render_segment(str(seg), camp, envsrc, thompson_cache))
    _run_script(str(spath), _env(tmp_path), str(seg))
    return str(seg)


def _submit_segment_via_run_options(tmp_path, monkeypatch, name, tree, argv):
    """`wtune run <argv>` option parsing -> the attempt submission that writes segment.sbatch
    -> the written script executed with a fake site and wrf.exe. Returns the segment dir."""
    from wtune.run import attempts, cli
    camp, envsrc = _fake_campaign(tmp_path, tree, _fake_wrf(tmp_path))
    opts = cli.parse_options(["--env-source", envsrc, "--campaign-dir", camp] + argv)
    monkeypatch.setattr(attempts, "step_time", lambda geo, a: 0)
    monkeypatch.setattr(attempts, "_fmt", lambda t: "2000-01-01_00:00:00")
    seg = types.SimpleNamespace(config=_config(), time_limit="01:00:00", end="E")
    att = types.SimpleNamespace(time_limit=None, a=0, time_step=None)
    ctx = types.SimpleNamespace(
        opts=opts, plan=types.SimpleNamespace(case_dir=str(tmp_path / "case")),
        runners=types.SimpleNamespace(sbatch_runner=lambda cmd: "Submitted batch job 42\n"),
        log=lambda s: None, segments=[seg], geo=None)
    seg_dir = tmp_path / name
    seg_dir.mkdir()
    assert attempts._submit_attempt(ctx, 0, seg, att, str(seg_dir)) == "42"
    _run_script(str(seg_dir / "segment.sbatch"), _env(tmp_path), str(seg_dir))
    return str(seg_dir)


def _coldstart_wrf_step(tmp_path, name, tree, populate=None):
    """Runs the coldstart probe script's wrf.exe step (run_window's tail) in a fake PLAN."""
    from wtune.tools import coldstart_sample as cs
    plan = tmp_path / name
    (plan / "win_1" / "wrf").mkdir(parents=True)
    (plan / "win_1" / "real").mkdir(parents=True)
    for f in ("wrfinput_d01", "wrfbdy_d01"):
        (plan / "win_1" / "real" / f).write_text("x")
    tool = tmp_path / "tool.py"
    tool.write_text("pass\n")
    fakewrf = _fake_wrf(tmp_path)
    body = "\n".join(cs._run_window_wrf_lines(4, 4, 1, 2, 2, 1, 60))
    script = ("set -uo pipefail\n"
              "pkill () { :; }\n"
              "site_launch () { SITE_LAUNCH_CMD=(bash %s); }\n"
              "wrf_step () {\n%s\n}\n"
              "k=1; PLAN=%s; TOOL=%s\n"
              "wrf_step; echo STEP_RC=$?\n" % (fakewrf, body, plan, tool))
    p = _bash(script, _env(tmp_path, WRF_TREE=tree), check=False)
    return str(plan / "win_1" / "wrf"), p


def _strict_lines(pattern):
    with open(os.path.join(CAMPAIGN, "strict_campaign.sbatch")) as f:
        return [ln for ln in f.read().splitlines() if re.match(pattern, ln)]


def _strict_table_cache(tmp_path, tree):
    """TABLE_CACHE as campaign/strict_campaign.sbatch computes it for wrf.exe under W."""
    lines = _strict_lines(r"^\s*(BUILD_ID|TABLE_CACHE)=")
    script = "\n".join(lines + ['echo "$TABLE_CACHE"'])
    out = _bash(script, _env(tmp_path, W=tree, OUT=tmp_path / "out")).stdout.strip()
    return out


def _calibrate_table_block():
    """The Thompson table linking block of `wtune calibrate plan`'s generated job script."""
    from wtune import calibrate
    tpl = calibrate.PLAN_SBATCH_TEMPLATE
    fields = {f for _lit, f, _spec, _conv in string.Formatter().parse(tpl) if f}
    text = tpl.format(**{f: "X" for f in fields})
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^(TBL_CACHE=|for tbl in )", ln))
    end = next(i for i in range(start, len(lines)) if lines[i] == "done")
    return "\n".join(lines[start:end + 1])


def _sha16(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- defect 1

def test_d1_run_segment_cache_off_by_default(tmp_path, monkeypatch):
    """Without --thompson-cache a segment neither harvests its tables into the cache nor links
    tables that are already cached for this build."""
    tree = _make_tree(tmp_path / "WRF", "build-1")
    seg = _submit_segment_via_run_options(tmp_path, monkeypatch, "seg_a", tree, [])
    assert _cache_files(str(tmp_path / "tblcache")) == {}, \
        "a default segment harvested Thompson tables into the cache"
    assert _start_links(seg) == []
    # now fill the cache the way an opted-in segment does; a default one still must not use it
    _submit_segment_via_run_options(tmp_path, monkeypatch, "seg_b", tree, ["--thompson-cache"])
    assert _cache_files(str(tmp_path / "tblcache"))
    seg_c = _submit_segment_via_run_options(tmp_path, monkeypatch, "seg_c", tree, [])
    assert _start_links(seg_c) == [], "a default segment linked cached tables"


def test_d1_run_thompson_cache_option_links_and_harvests(tmp_path, monkeypatch):
    tree = _make_tree(tmp_path / "WRF", "build-1")
    seg1 = _submit_segment_via_run_options(tmp_path, monkeypatch, "seg_1", tree,
                                           ["--thompson-cache"])
    assert _start_links(seg1) == []          # empty cache: nothing linked, WRF computes
    cached = _cache_files(str(tmp_path / "tblcache"))
    assert len(cached) == 1 and list(cached.values())[0] == sorted(TABLES)
    seg2 = _submit_segment_via_run_options(tmp_path, monkeypatch, "seg_2", tree,
                                           ["--thompson-cache"])
    assert sorted(_start_links(seg2)) == sorted(TABLES)


def test_d1_submit_forwards_thompson_cache(tmp_path):
    from wtune import submit
    from wtune.run import cli as run_cli

    def run_opts(argv):
        args = submit.build_parser().parse_args(argv)
        steps = submit.plan_steps(str(tmp_path), "id1", "2", True, True,
                                  run_args=tuple(submit.run_args(args)))
        run_argv = dict(steps)["run"]
        assert run_argv[1] == "run"
        return run_cli.parse_options(run_argv[2:])

    assert run_opts(["--thompson-cache"]).thompson_cache is True
    assert run_opts([]).thompson_cache is False


# ---------------------------------------------------------------- defect 2

def test_d2_segment_rebuild_in_place_does_not_reuse_tables(tmp_path):
    tree = _make_tree(tmp_path / "WRF", "build-1")
    _run_segment(tmp_path, "seg_1", tree)
    first = _cache_files(str(tmp_path / "tblcache"))
    assert len(first) == 1
    _make_tree(tmp_path / "WRF", "build-2 (same path, new binary)")
    seg2 = _run_segment(tmp_path, "seg_2", tree)
    assert _start_links(seg2) == [], "the rebuilt wrf.exe was given the old build's tables"
    second = _cache_files(str(tmp_path / "tblcache"))
    assert len(second) == 2, "the rebuilt wrf.exe's tables were not cached under a new key"


def test_d2_all_scripts_resolve_one_cache_dir_per_build(tmp_path):
    tree = _make_tree(tmp_path / "WRF", "build-1")
    root = str(tmp_path / "tblcache")
    _run_segment(tmp_path, "seg_1", tree)
    (seg_key,) = _cache_files(root)
    seg_dir = os.path.join(root, seg_key)
    assert os.path.realpath(_strict_table_cache(tmp_path, tree)) == os.path.realpath(seg_dir), \
        "strict_campaign.sbatch uses another table cache directory"

    # calibrate plan only links: tables in the per-build directory must be found
    run = tmp_path / "calrun"
    run.mkdir()
    _bash(_calibrate_table_block(), _env(tmp_path, W=tree, OUT=tmp_path / "out"), cwd=str(run))
    linked = {t: os.path.realpath(run / t) for t in TABLES if os.path.islink(run / t)}
    assert set(linked) == set(TABLES), "calibrate did not link the per-build cached tables"
    assert {os.path.dirname(v) for v in linked.values()} == {os.path.realpath(seg_dir)}

    __import__("shutil").rmtree(root)
    _wrfdir, p = _coldstart_wrf_step(tmp_path, "plan_1", tree)
    assert "STEP_RC=0" in p.stdout, (p.stdout, p.stderr)
    (cold_key,) = _cache_files(root)
    assert os.path.join(root, cold_key) == seg_dir, "coldstart probes and run segments disagree"

    # and the key follows the contents for every script
    _make_tree(tmp_path / "WRF", "build-2")
    assert os.path.realpath(_strict_table_cache(tmp_path, tree)) != os.path.realpath(seg_dir)
    _wrfdir, p = _coldstart_wrf_step(tmp_path, "plan_2", tree)
    assert "STEP_RC=0" in p.stdout, (p.stdout, p.stderr)
    assert len(_cache_files(root)) == 2


# ---------------------------------------------------------------- defect 3

@pytest.mark.parametrize("cache_exists", [True, False])
def test_d3_coldstart_empty_cache_links_nothing(tmp_path, cache_exists):
    tree = _make_tree(tmp_path / "WRF", "build-1")
    if cache_exists:   # the per-build directory exists but holds nothing (any key)
        (tmp_path / "tblcache").mkdir()
        (tmp_path / "tblcache" / _sha16(os.path.join(tree, "main", "wrf.exe"))).mkdir()
    wrfdir, p = _coldstart_wrf_step(tmp_path, "plan", tree)
    assert _start_links(wrfdir) == [], "table links made from an empty cache"
    assert "STEP_RC=0" in p.stdout, (p.stdout, p.stderr)


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


def test_d4_probe_ledger_rows_identify_the_build(tmp_path):
    from wtune.tools import scc_sweep as sweep
    tree = _make_tree(tmp_path / "WRF", "build-1")
    lines = _strict_lines(r"^\s*BUILD_ID=") + [
        ln.strip() for ln in _strict_lines(r'^\s*NOTES="config=\$cid')]
    assert len(lines) >= 1

    def notes():
        script = "\n".join(["cid=C1 NEXCL=0 REF_CONFIG=C1 MODE=timing"] + lines + ['echo "$NOTES"'])
        return _bash(script, _env(tmp_path, W=tree)).stdout.strip()

    tok = [t for t in notes().split() if t.startswith("build=")]
    assert tok, "probe ledger notes carry no build id: %r" % notes()
    assert hasattr(sweep, "wrf_build_id"), "the tune has no way to compute the build id"
    assert tok[0] == "build=" + sweep.wrf_build_id(tree), \
        "the probes' build id is not the one the tune filters on"
    _make_tree(tmp_path / "WRF", "build-2")
    assert [t for t in notes().split() if t.startswith("build=")] != tok


def test_d4_estimate_s_per_step_never_mixes_builds(tmp_path):
    from wtune.tools import scc_sweep as sweep
    d = tmp_path / "ledgers"
    d.mkdir()
    _ledger(d / "ledger_cx_1.csv", [_row("cx", 10.0, "aaaa")])
    _ledger(d / "ledger_cx_2.csv", [_row("cx", 30.0, "bbbb")])
    s, src, n = sweep.estimate_s_per_step("cx", 1, str(d), "hw1", fallback=99.0, build_id="aaaa")
    assert (s, n) == (10.0, 1), "another build's rows were averaged in"
    s, src, n = sweep.estimate_s_per_step("cx", 1, str(d), "hw1", fallback=99.0, build_id="cccc")
    assert (s, n) == (99.0, 0), "another build's rows were used for an unmeasured build"
    _ledger(d / "ledger_cx_3.csv", [_row("cx", 50.0)])          # written before build ids
    s, src, n = sweep.estimate_s_per_step("cx", 1, str(d), "hw1", fallback=99.0, build_id="cccc")
    assert (s, n) == (50.0, 1), "rows without a build id are the fallback"
    s, src, n = sweep.estimate_s_per_step("cx", 1, str(d), "hw1", fallback=99.0, build_id="aaaa")
    assert (s, n) == (10.0, 1), "the build's own rows must win over unknown-build rows"


def test_d4_default_proposer_prior_excludes_other_builds(tmp_path, monkeypatch):
    from wtune.predictor import decomp_search
    from wtune.tune import rounds
    d = tmp_path / "ledgers"
    d.mkdir()
    _ledger(d / "ledger_other_1.csv", [_row("other", 10.0, "aaaa"), _row("other", 20.0, "bbbb"),
                                       _row("other", 40.0)])
    seen = {}

    def fake_propose(case, case_json, case_fields, prior_rows, state, batch_size, **kw):
        seen["prior"] = prior_rows
        return []
    monkeypatch.setattr(decomp_search, "propose_for_case", fake_propose)
    monkeypatch.setattr(decomp_search, "case_row_fields", lambda cj: {})
    monkeypatch.setattr(rounds, "cores_used", lambda ctx: 1, raising=False)
    ctx = types.SimpleNamespace(case="mine", case_json={}, ledger_dir=str(d), history_dirs=[],
                                build_id="aaaa", nodes=1, hardware_id="hw1", kappa=1.0)
    rounds._default_proposer(ctx)(None, 1)
    got = sorted(r["s_per_step_median"] for r in seen["prior"])
    assert got == [10.0, 40.0], "cross-case prior rows: %r" % got


def _two_build_ledgers(tmp_path):
    d = tmp_path / "ledgers"
    d.mkdir()
    _ledger(d / "ledger_cx_1.csv", [_row("cx", 10.0, "aaaa")])     # this build
    _ledger(d / "ledger_cx_2.csv", [_row("cx", 30.0, "bbbb")])     # a tuned-flag build
    from test_wtune_tune import _ctx
    ctx = _ctx(case="cx", nodes=1, ledger_dir=str(d), hardware_id="hw1", build_id="aaaa",
               history_dirs=[])
    ctx.startup_s = 0.0
    return ctx


def test_d4_cold_start_prediction_uses_only_this_build(tmp_path):
    """-O0's ledger estimate (tune/prediction.py cold_start_prediction without a config)."""
    from wtune.tune import prediction
    ctx = _two_build_ledgers(tmp_path)
    _T, s_est, _src = prediction.cold_start_prediction(ctx, 1, None, 100)
    assert s_est == 10.0, "s/step %r averaged another build's rows" % s_est


def test_d4_round0_estimate_uses_only_this_build(tmp_path, monkeypatch):
    """The s/step run_round0 sizes round 0 with (tune/rounds.py)."""
    from wtune.tools import scc_sweep as sweep
    from wtune.tune import rounds
    ctx = _two_build_ledgers(tmp_path)
    real = sweep.estimate_s_per_step
    got = []

    class _Stop(Exception):
        pass

    def spy(*a, **kw):
        got.append(real(*a, **kw)[0])
        raise _Stop
    monkeypatch.setattr(sweep, "estimate_s_per_step", spy)
    with pytest.raises(_Stop):
        rounds.run_round0(ctx)
    assert got == [10.0], "round 0 s/step %r averaged another build's rows" % got


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


def _site_out(tmp_path, site_file):
    env = _env(tmp_path, CAMPAIGN=CAMPAIGN)
    if os.environ.get("WTUNE_CAMPAIGN_OUT"):
        env["WTUNE_CAMPAIGN_OUT"] = os.environ["WTUNE_CAMPAIGN_OUT"]
    return _bash('source "%s" >/dev/null 2>&1; echo "$OUT"' % site_file, env).stdout.strip()


@pytest.mark.parametrize("where", ["explicit", "project", "env", "default"])
def test_d5_probe_ledger_dir_is_where_the_tune_reads(tmp_path, monkeypatch, where):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.delenv("WTUNE_CAMPAIGN_OUT", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    argv = []
    if where in ("explicit", "project"):
        (work / "project.yaml").write_text("name: p\n")
    if where in ("explicit", "project", "env"):
        monkeypatch.setenv("WTUNE_CAMPAIGN_OUT", str(tmp_path / "envout"))
    if where == "explicit":
        argv = ["--ledger-dir", str(tmp_path / "explicit")]
    expected = {"explicit": tmp_path / "explicit", "project": work / "probes",
                "env": tmp_path / "envout",
                "default": home / "wtune-runs" / "campaign"}[where]
    kw = _tune_ctx_kwargs(tmp_path, monkeypatch, argv)
    assert os.path.realpath(kw["ledger_dir"]) == os.path.realpath(str(expected)), \
        "tune reads ledgers from %s" % kw["ledger_dir"]
    # the probe jobs (sbatch --export=ALL) see this environment; every site's $OUT must agree
    sites = sorted(os.path.join(CAMPAIGN, "sites", f)
                   for f in os.listdir(os.path.join(CAMPAIGN, "sites")) if f.endswith(".sh"))
    assert sites
    from wtune.tune import cli
    with cli._campaign_out(kw["ledger_dir"]):        # what the tune sets while its jobs run
        for site in sites:
            assert os.path.realpath(_site_out(tmp_path, site)) == os.path.realpath(str(expected)), \
                "%s: probes write to %s, the tune reads %s" % (
                    os.path.basename(site), _site_out(tmp_path, site), expected)


def test_d4_tune_filters_on_the_machine_wrf_build(tmp_path, monkeypatch):
    from wtune.tools import scc_sweep as sweep
    tree = _make_tree(tmp_path / "WRF", "build-1")
    monkeypatch.setenv("WRF_TREE", tree)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    from wtune.tune import cli
    monkeypatch.setattr(cli, "_build_id", REAL_BUILD_ID)    # conftest stubs it for hermeticity
    kw = _tune_ctx_kwargs(tmp_path, monkeypatch, ["--ledger-dir", str(tmp_path / "l"),
                                                  "--env-source", str(tmp_path / "no-env.sh")])
    assert kw.get("build_id"), "the tune does not know which WRF build it measures"
    assert kw["build_id"] == sweep.wrf_build_id(tree)


# ---------------------------------------------------------------- defect 6

def test_d6_predict_env_cores_per_node_beats_calibration(monkeypatch):
    from wtune import predict
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", "144")
    assert predict.cores_per_node({"cores_per_node": 72}) == 144


def test_d6_predict_machine_yaml_cores_per_node_beats_calibration(tmp_path, monkeypatch):
    import wtune.machine
    from wtune import predict
    f = tmp_path / "machine.yaml"
    f.write_text("cores_per_node: 144\n")
    monkeypatch.setenv("WTUNE_MACHINE_FILE", str(f))
    monkeypatch.delenv("WTUNE_CORES_PER_NODE", raising=False)
    # what the wtune dispatcher does before every command (wtune/cli.py _run_forwarded)
    wtune.machine.export(wtune.machine.load_user())
    try:
        assert predict.cores_per_node({"cores_per_node": 72}) == 144
    finally:
        os.environ.pop("WTUNE_CORES_PER_NODE", None)


def test_d6_predict_calibration_used_when_machine_unset(monkeypatch):
    from wtune import predict
    monkeypatch.delenv("WTUNE_CORES_PER_NODE", raising=False)
    assert predict.cores_per_node({"cores_per_node": 72}) == 72
    assert predict.cores_per_node({}) == 72


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
