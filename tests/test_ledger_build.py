"""The probe ledgers carry the WRF build, live where the tune says, and the Thompson table cache is
keyed by the wrf.exe contents (2026-10-26: a vanilla and a tuned-flag build of one case were
averaged together; a rebuild in place kept the old tables; f1.sh's $OUT was not the directory the
tune read)."""
import argparse
import csv
import os
import subprocess


from wtune.tools import scc_sweep as sweep
from wtune.tune import cli as tune_cli

REAL_BUILD_ID = tune_cli._build_id          # at collection, before conftest stubs it
REAL_PROBE_TREE = tune_cli._probe_tree

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


def test_tune_build_id_wiring(tmp_path, monkeypatch, capsys):
    """tune/cli._probe_tree and _build_id (conftest stubs them elsewhere): the env.sh the jobs source, the site
    file of the campaign the sbatch script belongs to, the fallback and what it prints."""
    import wtune.machine
    tree = _tree(tmp_path)
    (tmp_path / "env.sh").write_text("export WRF_TREE=%s\n" % tree)
    monkeypatch.delenv("WRF_TREE", raising=False)
    monkeypatch.setattr(wtune.machine, "site", lambda: "f1arm")
    script = os.path.join(REPO, "campaign", "strict_campaign.sbatch")
    args = argparse.Namespace(env_source=str(tmp_path / "env.sh"), sbatch_script=script)
    assert REAL_PROBE_TREE(args) == tree
    assert REAL_BUILD_ID(REAL_PROBE_TREE(args)) == sweep.wrf_build_id(tree) != ""
    assert "build %s (%s/main/wrf.exe)" % (sweep.wrf_build_id(tree), tree) in capsys.readouterr().out
    (tmp_path / "env.sh").write_text("export WRF_TREE=%s\n" % (tmp_path / "gone"))
    assert REAL_BUILD_ID(REAL_PROBE_TREE(args)) == ""
    assert "no wrf.exe under %s" % (tmp_path / "gone") in capsys.readouterr().out
