"""The probe ledgers carry the WRF build, live where the tune says, and the Thompson table cache is
keyed by the wrf.exe contents (2026-10-26: a vanilla and a tuned-flag build of one case were
averaged together; a rebuild in place kept the old tables; f1.sh's $OUT was not the directory the
tune read)."""
import argparse
import csv
import os
import re
import subprocess

from wtune.tools import coldstart_sample as cs
from wtune.tools import scc_sweep as sweep
from wtune.tune import cli as tune_cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["case_base", "hardware_id", "n_nodes", "s_per_step_median", "notes"]


def _tree(tmp_path, content="wrf"):
    (tmp_path / "W" / "main").mkdir(parents=True, exist_ok=True)
    (tmp_path / "W" / "main" / "wrf.exe").write_text(content)
    return str(tmp_path / "W")


def _bash(body, env, cwd):
    return subprocess.run(["bash", "-c", body], env=dict(PATH=os.environ["PATH"], **env), cwd=cwd,
                          check=True, capture_output=True, text=True).stdout


def _ledger(d, case, job, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ledger_%s_%s.csv" % (case, job)), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for nodes, s, notes in rows:
            w.writerow(dict(case_base=case, hardware_id="hw", n_nodes=nodes,
                            s_per_step_median=s, notes=notes))


def test_probe_and_tune_compute_the_same_build_id(tmp_path):
    tree = _tree(tmp_path)
    text = open(os.path.join(REPO, "campaign", "strict_campaign.sbatch")).read()
    lines = [ln for ln in text.splitlines() if ln.startswith(("BUILD_ID=", "TABLE_CACHE="))]
    out = _bash("\n".join(lines) + '\necho "$BUILD_ID $TABLE_CACHE"',
                dict(W=tree, HOME=str(tmp_path)), str(tmp_path))
    build, cache = out.split()
    assert build == sweep.wrf_build_id(tree) and re.fullmatch("[0-9a-f]{16}", build)
    assert cache == str(tmp_path / "wtune-runs" / "thompson_tables" / build)
    assert "build=$BUILD_ID" in text
    assert _bash("echo " + cs.build_id_expr("$W"), dict(W=tree), str(tmp_path)).strip() == build
    assert sweep.wrf_build_id(str(tmp_path / "none")) == ""
    _tree(tmp_path, "rebuilt")
    assert sweep.wrf_build_id(tree) != build


def test_estimate_keeps_to_the_build(tmp_path):
    d = str(tmp_path / "ledgers")
    _ledger(d, "c", 1, [(1, 1.0, "config=a build=aaaa"), (2, 0.5, "config=b build=aaaa")])
    _ledger(d, "c", 2, [(1, 5.0, "config=a build=bbbb")])
    _ledger(d, "c", 3, [(1, 3.0, "config=a")])
    est = sweep.estimate_s_per_step
    assert est("c", 1, d, "hw", fallback=9.0, build_id="aaaa") == (1.0, "ledger", 2)
    assert est("c", 1, d, "hw", fallback=9.0, build_id="cccc") == (3.0, "ledger_unknown_build", 1)
    assert est("c", 1, d, "hw", fallback=9.0) == (2.0, "ledger", 4)
    assert est("c", 1, str(tmp_path / "empty"), "hw", fallback=9.0, build_id="aaaa",
               extra_dirs=[d]) == (1.0, "ledger", 2)


def test_tune_ledger_dir_is_the_project_and_reaches_the_jobs(tmp_path, monkeypatch):
    import wtune.project
    args = argparse.Namespace(ledger_dir=None, project=None)
    monkeypatch.setattr(wtune.project, "find_project", lambda explicit=None: str(tmp_path / "proj"))
    probes = str(tmp_path / "proj" / "probes")
    assert tune_cli._ledger_dir(args) == probes
    with tune_cli._campaign_out(probes):
        assert os.environ[sweep.CAMPAIGN_OUT_ENV] == probes            # what the jobs inherit
    assert sweep.CAMPAIGN_OUT_ENV not in os.environ                    # not the next tune's default
    assert sweep.history_ledger_dirs(probes) == [sweep.HISTORY_LEDGER_DIR]
    assert sweep.history_ledger_dirs(sweep.HISTORY_LEDGER_DIR) == []
    args.ledger_dir = str(tmp_path / "mine")
    assert tune_cli._ledger_dir(args) == str(tmp_path / "mine")
    monkeypatch.setattr(wtune.project, "find_project", lambda explicit=None: None)
    args.ledger_dir = None
    assert tune_cli._ledger_dir(args) == sweep._expand(sweep.DEFAULT_LEDGER_DIR)
    monkeypatch.setenv(sweep.CAMPAIGN_OUT_ENV, str(tmp_path / "user"))
    assert tune_cli._ledger_dir(args) == str(tmp_path / "user")


def test_build_id_comes_from_the_tree_the_jobs_run(tmp_path):
    """4df97ca review: the id came from machine.wrf_tree(), the jobs hash the site file's $W
    after `source env.sh` (f1-arm: env.sh sets WRF_TREE, machine.yaml has no wrf_tree)."""
    tree = _tree(tmp_path)
    (tmp_path / "campaign" / "sites").mkdir(parents=True)
    (tmp_path / "campaign" / "sites" / "s.sh").write_text('W=${WRF_TREE:-/nonexistent/WRF}\n')
    (tmp_path / "env.sh").write_text("export WRF_TREE=%s\necho chatter\n" % tree)
    assert sweep.job_wrf_tree(str(tmp_path / "env.sh"), str(tmp_path / "campaign"), "s") == tree
    assert sweep.job_wrf_tree(str(tmp_path / "none.sh"), str(tmp_path / "campaign"), "s") == \
        "/nonexistent/WRF"
    assert sweep.job_wrf_tree(str(tmp_path / "env.sh"), None, "s") is None


def test_site_files_take_the_tunes_out(tmp_path):
    for site in ("scc", "f1arm", "f1"):
        text = open(os.path.join(REPO, "campaign", "sites", site + ".sh")).read()
        line = next(ln for ln in text.splitlines() if ln.startswith("OUT="))
        env = dict(HOME=str(tmp_path), WTUNE_CAMPAIGN_OUT=str(tmp_path / "p"))
        assert _bash(line + '\necho "$OUT"', env, str(tmp_path)).strip() == str(tmp_path / "p")


def test_coldstart_window_links_no_table_the_cache_lacks(tmp_path):
    """The run segment's 2026-10-25 failure in the coldstart probes' copy: a link to the absent
    freezeH2O.dat made WRF fail writing it."""
    lines = [ln.strip() for ln in cs._run_window_wrf_lines(2, 2, 1, 1, 2, 1, 60)]
    link = [ln for ln in lines if ln.startswith("TBL_CACHE=") or '"$TBL_CACHE"/qr_acr_q' in ln]
    harvest = [ln for ln in lines if ln.startswith(("shopt -s nullglob; TBLS=", 'for f in "${TBLS'))]
    assert len(link) == 2 and len(harvest) == 2
    tree = _tree(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    env = dict(WRF_TREE=tree, HOME=str(tmp_path))
    _bash("\n".join(link), env, str(run))
    assert os.listdir(run) == []
    (run / "freezeH2O.dat").write_text("table")
    _bash("\n".join(link + harvest), env, str(run))
    cache = tmp_path / "wtune-runs" / "thompson_tables" / sweep.wrf_build_id(tree)
    assert os.listdir(cache) == ["freezeH2O.dat"]


def test_o3_prior_leaves_out_other_builds(tmp_path, monkeypatch):
    import types
    from wtune.predictor import decomp_search
    from wtune.tune import rounds
    d, hist = str(tmp_path / "probes"), str(tmp_path / "shared")
    _ledger(d, "me", 1, [(1, 1.0, "build=aaaa")])
    _ledger(d, "other", 2, [(1, 2.0, "build=aaaa"), (1, 3.0, "build=bbbb")])
    _ledger(hist, "old", 3, [(1, 4.0, "config=a")])
    seen = {}
    monkeypatch.setattr(decomp_search, "case_row_fields", lambda cj: {})
    monkeypatch.setattr(decomp_search, "propose_for_case",
                        lambda case, cj, cf, prior, *a, **k: seen.setdefault("prior", prior))
    ctx = types.SimpleNamespace(case="me", case_json={}, ledger_dir=d, history_dirs=[hist],
                                build_id="aaaa", nodes=1, hardware_id="hw", kappa=1.0)
    monkeypatch.setattr(rounds, "cores_used", lambda c: 1)
    rounds._default_proposer(ctx)({}, 1)
    assert sorted(r["s_per_step_median"] for r in seen["prior"]) == [2.0, 4.0]


def test_every_tune_estimate_keeps_to_the_build(tmp_path, monkeypatch):
    """4df97ca review: round 0 (which sizes the first measurement), the node-search finalists and
    the cold-start prediction each called the estimate without the build or the older ledgers."""
    import pytest
    from wtune.tune import prediction, rounds
    from test_wtune_tune import _ctx
    probes, hist = str(tmp_path / "probes"), str(tmp_path / "shared")
    _ledger(hist, "c", 1, [(1, 2.0, "config=a")])
    _ledger(probes, "c", 2, [(1, 1.0, "build=aaaa"), (1, 9.0, "build=bbbb")])
    ctx = _ctx(case="c", hardware_id="hw", ledger_dir=probes, build_id="aaaa", history_dirs=[hist])
    assert ctx.estimate_s_per_step() == (1.0, "ledger", 1)
    ctx.build_id = "cccc"
    assert ctx.estimate_s_per_step(2) == (1.0, "ledger_unknown_build", 1)
    ctx.build_id = "aaaa"
    got = []
    real = sweep.estimate_s_per_step

    def spy(*a, **k):
        got.append(real(*a, **k))
        raise RuntimeError("stop")
    monkeypatch.setattr(sweep, "estimate_s_per_step", spy)
    for call in (lambda: rounds.run_round0(ctx), lambda: rounds._run_tuning_at(ctx, None, False, False),
                 lambda: prediction.cold_start_prediction(ctx, 1, None, 10)):
        with pytest.raises(RuntimeError):
            call()
    assert got == [(1.0, "ledger", 1)] * 3


def _block(text, start, end):
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(start))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith(end))
    return "\n".join(lines[i:j + 1])


def test_probe_harvest_caches_only_after_init_and_through_a_temporary_name(tmp_path):
    """4df97ca review: the harvest ran whatever the run did and mv'd straight to the cache name,
    a copy when <project>/probes is on another filesystem than the cache."""
    text = open(os.path.join(REPO, "campaign", "strict_campaign.sbatch")).read()
    func = _block(text, "harvest_thompson_tables () {", "}")
    cache, run = tmp_path / "cache", tmp_path / "run"
    cache.mkdir(), run.mkdir()
    body = ("THOMPSON_TABLE_NAMES=(freezeH2O.dat qr_acr_qsV2.dat); TABLE_CACHE=%s; J=7\n%s\n"
            "harvest_thompson_tables" % (cache, func))
    (run / "freezeH2O.dat").write_text("partial")
    _bash(body, {}, str(run))                       # died before 'Timing for main'
    assert os.listdir(cache) == [] and (run / "freezeH2O.dat").exists()
    (run / "rsl.error.0000").write_text("Timing for main: time 2020-06-01_00:00:10 on domain 1\n")
    _bash(body, {}, str(run))
    assert os.listdir(cache) == ["freezeH2O.dat"] and not (run / "freezeH2O.dat").exists()


def test_calibrate_run_caches_the_tables_it_computed(tmp_path):
    import test_wtune_calibrate as tc
    from wtune import calibrate as wc
    case_dir, out = tmp_path / "case", tmp_path / "plan_out"
    case_dir.mkdir()
    tc._write(str(case_dir / "namelist.input"), tc.SEVERE_LIKE_NAMELIST)
    assert wc.plan_cmd(tc._plan_args(case_dir, out)) == 0
    text = next(out.glob("plan_*.sbatch")).read_text()
    harvest = _block(text, 'if [ "$OK" = 1 ]; then', "fi")
    cache, run = tmp_path / "cache", tmp_path / "run"
    run.mkdir()
    (run / "qr_acr_qsV2.dat").write_text("table")
    _bash("TBL_CACHE=%s; OK=0\n%s" % (cache, harvest), {}, str(run))
    assert not cache.exists()
    _bash("TBL_CACHE=%s; OK=1\n%s" % (cache, harvest), {}, str(run))
    assert os.listdir(cache) == ["qr_acr_qsV2.dat"]
