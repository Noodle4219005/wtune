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
    assert tune_cli._ledger_dir(args) == str(tmp_path / "proj" / "probes")
    assert os.environ[sweep.CAMPAIGN_OUT_ENV] == str(tmp_path / "proj" / "probes")
    assert sweep.history_ledger_dirs(str(tmp_path / "proj" / "probes")) == [
        sweep._expand(sweep.DEFAULT_LEDGER_DIR)]
    assert sweep.history_ledger_dirs(sweep.DEFAULT_LEDGER_DIR) == []
    args.ledger_dir = str(tmp_path / "mine")
    assert tune_cli._ledger_dir(args) == str(tmp_path / "mine")
    monkeypatch.setattr(wtune.project, "find_project", lambda explicit=None: None)
    args.ledger_dir = None
    assert tune_cli._ledger_dir(args) == str(tmp_path / "mine")      # the exported one
    os.environ.pop(sweep.CAMPAIGN_OUT_ENV)
    assert tune_cli._ledger_dir(args) == sweep._expand(sweep.DEFAULT_LEDGER_DIR)


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
