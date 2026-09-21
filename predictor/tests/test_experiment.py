#!/usr/bin/env python3
"""predictor/tests/test_experiment.py -- pytest suite for predictor/experiment.py (WTune
decision-0050 step 5: minimal experiment engine).

Behaviour-focused: synthetic ledgers under tmp_path, a fake `recommend` module injected into
sys.modules (predictor/recommend.py itself is NOT importable in a plain stdlib environment --
its dependency chain needs scikit-learn, see experiment.py's own module docstring), and a
fake `sacct` via a `python -c` subprocess command instead of a real binary.

Run with:
    /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q predictor/tests/test_experiment.py
"""
import csv
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "campaign"))

import experiment as ex          # noqa: E402
import configs as campaign_configs  # noqa: E402  -- campaign/configs.py, for the round-trip test


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def _row(**kw):
    base = {k: "" for k in ex.LEDGER_HEADER}
    base.update(
        source_dataset="strict_campaign", job_id="9001", case_raw=kw.get("case_base", "karina"),
        hardware_id="f1-spr-icpnq", binary_kind="pristine", compliance_class="STRICT",
        e_we="901", e_sn="901", e_vert="45", distribution="cyclic", kmp_blocktime="0", n_nodes="4",
    )
    base.update(kw)
    return base


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ex.LEDGER_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _karina_rows(case="karina", c01=(0.40, 0.41), c06=(0.255, 0.257), c05=(0.262, 0.260)):
    """C01 (baseline, 448x1), C06 (numtiles=12, 224x2), C05 (numtiles=13, 224x2) -- the real
    campaign/configs.py TEMPLATES shapes for karina (901x901x45), so campaign_configs.
    resolve_case() and these rows describe the SAME decompositions (nx=8,ny=56 for 448 ranks;
    nx=7,ny=32 for 224 ranks -- see campaign/configs.py legal_shapes/ranked_shapes)."""
    rows = []
    specs = [("C01", "448", "1", "8", "56", "1", c01),
             ("C06", "224", "2", "7", "32", "12", c06),
             ("C05", "224", "2", "7", "32", "13", c05)]
    for cid, ranks, omp, nx, ny, numtiles, meds in specs:
        for rep, med in enumerate(meds, 1):
            rows.append(_row(
                run_id="%s_%s_r%d_9001" % (case, cid, rep), rep_index=str(rep),
                case_base=case, case_raw=case, n_ranks=ranks, omp_threads=omp,
                nproc_x=nx, nproc_y=ny, numtiles=numtiles, s_per_step_median=str(med),
                s_per_step_mean=str(med), n_steps_measured="30", notes="config=%s rc=0" % cid,
            ))
    return rows


def _plan_args(tmp_path, case="karina", ledger_paths=None, loco=False, configs=None,
               out=None, reps=2, nodes=4, top_k=3, namelist=None):
    return SimpleNamespace(
        case=case, ledger=ledger_paths or [], namelist=namelist or os.devnull,
        hardware_id="f1-spr-icpnq", nodes=nodes, top_k=top_k, loco=loco, configs=configs,
        reps=reps, out=str(out or (tmp_path / "plan")),
    )


def _fake_sacct_cmd(line):
    """A `sacct`-shaped subprocess command that just prints one fixed line, ignoring whatever
    -j/-X/-n/-P/--format args experiment.py appends -- no real sacct binary needed."""
    return [sys.executable, "-c", "print(%r)" % line]


def _install_fake_recommend(monkeypatch, shortlist, capture=None):
    """Inject a fake `recommend` module into sys.modules so experiment.py's lazy `import
    recommend as recommend_mod` picks it up instead of the real predictor/recommend.py (which
    needs scikit-learn, unavailable in this test environment)."""
    mod = types.ModuleType("recommend")

    def recommend(ledger_path, namelist_path, case_name, hardware_id, nodes, top_k, **kw):
        if capture is not None:
            with open(ledger_path, newline="") as f:
                ledger_rows = list(csv.DictReader(f))
            capture.append(dict(ledger_path=ledger_path, ledger_rows=ledger_rows,
                                 namelist_path=namelist_path, case_name=case_name,
                                 hardware_id=hardware_id, nodes=nodes, top_k=top_k, kwargs=kw))
        return dict(model_used="fake_model", magnitude_model_used="fake_point",
                    warnings=["fake warning"], shortlist=shortlist[:top_k])

    mod.recommend = recommend
    monkeypatch.setitem(sys.modules, "recommend", mod)
    return mod


_SHORTLIST_C06_TOP1 = [
    dict(config_id="C06", rank=1, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32,
         numtiles=12, distribution="cyclic", kmp_blocktime=0,
         predicted_gain=0.06, gain_lower_95=0.02, label="SAFE"),
    dict(config_id="C05", rank=2, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32,
         numtiles=13, distribution="cyclic", kmp_blocktime=0,
         predicted_gain=0.03, gain_lower_95=-0.01, label="EXPERIMENTAL"),
]


# ---------------------------------------------------------------------------
# LOCO
# ---------------------------------------------------------------------------
def test_loco_drops_target_case_rows_before_recommend_but_best_known_uses_full_ledger(
        tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    rows = _karina_rows() + _karina_rows(case="conus25", c01=(1.0, 1.01), c06=(0.9, 0.91),
                                          c05=(0.95, 0.96))
    _write_csv(ledger, rows)

    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1, capture=captured)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], loco=True,
                       out=tmp_path / "plan_loco")
    plan = ex.cmd_plan(args)

    assert len(captured) == 1
    seen_cases = {r["case_base"] for r in captured[0]["ledger_rows"]}
    assert "karina" not in seen_cases, "LOCO must drop every karina row before recommend() runs"
    assert "conus25" in seen_cases, "LOCO must not touch other cases' rows"

    # best_known must still reflect karina's OWN full (unfiltered) history: C06 (0.256 median)
    # beats C05 (0.261) and C01 (0.405) in _karina_rows()'s fixture medians.
    assert plan["best_known"]["config_id"] == "C06"
    assert plan["best_known"]["measured_median_s_per_step"] == pytest.approx(0.256, abs=1e-9)
    assert plan["recommend"]["loco"] is True
    assert plan["recommend"]["n_ledger_rows_used"] < plan["recommend"]["n_ledger_rows_total"]


def test_best_known_ignores_rows_from_other_node_counts():
    # a faster 8-node run must not become the comparator for a 4-node job
    rows = _karina_rows() + [_row(
        run_id="karina_C99_r1_9002", case_base="karina", n_ranks="896", omp_threads="1",
        nproc_x="16", nproc_y="56", numtiles="1", s_per_step_median="0.10",
        n_nodes="8", notes="config=C99 rc=0")]
    assert ex._best_known(rows, "karina")[0] == "C99"            # unfiltered: wrong comparator
    assert ex._best_known(ex._rows_on_nodes(rows, 4), "karina")[0] == "C06"


def test_no_loco_keeps_target_case_rows_in_recommend_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1, capture=captured)

    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], loco=False, out=tmp_path / "plan_full")
    ex.cmd_plan(args)

    seen_cases = {r["case_base"] for r in captured[0]["ledger_rows"]}
    assert "karina" in seen_cases


# ---------------------------------------------------------------------------
# configs.json round-trip through campaign/configs.py's CONFIG_FILE loader
# ---------------------------------------------------------------------------
def test_configs_json_round_trips_through_config_file_loader(tmp_path):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C06", "C05"]))

    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit),
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    loaded = campaign_configs.load_generated_configs(plan["configs_json_path"])
    assert {c["id"] for c in loaded} == {"C01", "C05", "C06"}  # + baseline/best_known

    ids, by_id, reason = campaign_configs.available_decomps(
        "karina", config_file=plan["configs_json_path"])
    assert reason is None
    assert set(ids) == {"C01", "C05", "C06"}
    assert by_id["C06"]["ranks"] == 224 and by_id["C06"]["numtiles"] == 12


# ---------------------------------------------------------------------------
# cost estimate + --time formatting (hand-computed)
# ---------------------------------------------------------------------------
def test_cost_line_hand_computed():
    case_json = dict(steps=10)
    candidate_ids = ["C01", "C05"]
    s_est = {"C01": 1.0, "C05": 0.8}
    cost = ex._cost_estimate(case_json, candidate_ids, s_est, startup_s=60.0, reps=2, nodes=4)

    # total_seconds = 2*(10*1.0+60) + 2*(10*0.8+60) = 140 + 136 = 276
    # job_hours = 276/3600 + 5/60 = 0.16 exactly
    assert cost["job_hours"] == pytest.approx(0.16, abs=1e-9)
    assert cost["node_hours"] == pytest.approx(0.64, abs=1e-9)
    assert cost["su"] == pytest.approx(14.336, abs=1e-9)          # 0.64 * 112 * 0.2
    assert cost["cost_line"] == (
        "1 jobs x 4 nodes x 0.16 h = 0.64 node-hours = 14.34 SU, wall-clock ~0.16 h")


def test_time_limit_formatting():
    # job_minutes = 0.16*60 = 9.6; ceil(1.5*9.6)=ceil(14.4)=15; +10 = 25 -> 00:25:00
    assert ex._format_hms(1.5 * 9.6 + 10) == "00:25:00"
    # exact boundary: 125 minutes -> 02:05:00
    assert ex._format_hms(125.0) == "02:05:00"
    # fractional minutes always round UP (never under-request wall-clock)
    assert ex._format_hms(59.01) == "01:00:00"


def test_startup_assumed_true_when_no_ledger_column():
    rows = _karina_rows()
    startup_s, assumed = ex._startup_s(rows)
    assert assumed is True
    assert startup_s == ex.DEFAULT_STARTUP_S


# ---------------------------------------------------------------------------
# dedupe with multiple roles
# ---------------------------------------------------------------------------
def test_dedupe_collapses_multiple_roles_onto_one_candidate(tmp_path):
    # best_known (C01, artificially made the fastest) and baseline are the SAME id, and the
    # --configs list ALSO explicitly asks for C01 -- must collapse to ONE candidate entry.
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows(c01=(0.10, 0.11), c06=(0.9, 0.91), c05=(0.95, 0.96)))
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C01"]))

    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit),
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    ids = [c["config_id"] for c in plan["candidates"]]
    assert ids.count("C01") == 1, "duplicate candidate entries for the same config_id"
    c01 = next(c for c in plan["candidates"] if c["config_id"] == "C01")
    assert set(c01["roles"]) == {"explicit:1", "baseline", "best_known"}
    assert plan["best_known"]["config_id"] == "C01"


# ---------------------------------------------------------------------------
# missing best_known -> null, verdict INCONCLUSIVE
# ---------------------------------------------------------------------------
def test_missing_best_known_is_null_and_reconcile_is_inconclusive(tmp_path):
    # karina has a campaign/configs.py COST_GUESS entry for its grid (901,901,45), so the cost
    # estimate still succeeds with ZERO ledger rows for this case -- best_known must be null.
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [])   # no measured rows for ANY case
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C06"]))

    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit), out=plan_dir)
    plan = ex.cmd_plan(args)
    assert plan["best_known"] is None
    assert not any("best_known" in c["roles"] for c in plan["candidates"])

    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="7001"))

    job_ledger_dir = tmp_path / "job_ledgers"
    job_ledger_dir.mkdir()
    _write_csv(job_ledger_dir / "ledger_karina_7001.csv", _karina_rows())  # a real run happened
    block = ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                          sacct_cmd=_fake_sacct_cmd("7001|strictcamp|COMPLETED|00:10:00|4|448"),
                          results_dir=str(tmp_path / "results"))
    assert block["verdict"] == "INCONCLUSIVE"
    assert block["best_known"] is None


# ---------------------------------------------------------------------------
# reconcile: PASS / FAIL (regret 5%) / sacct non-COMPLETED -> INCONCLUSIVE / est vs actual SU
# ---------------------------------------------------------------------------
def _planned(tmp_path, name, ledger_rows, explicit_ids=("C06", "C05")):
    ledger = tmp_path / ("%s_ledger.csv" % name)
    _write_csv(ledger, ledger_rows)
    explicit = tmp_path / ("%s_explicit.json" % name)
    explicit.write_text(json.dumps(list(explicit_ids)))
    plan_dir = tmp_path / ("%s_plan" % name)
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit), out=plan_dir)
    ex.cmd_plan(args)
    return plan_dir


def test_reconcile_pass(tmp_path):
    plan_dir = _planned(tmp_path, "pass_case", _karina_rows(c06=(0.256, 0.258)))
    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="8001"))
    job_ledger_dir = tmp_path / "pass_job_ledgers"
    job_ledger_dir.mkdir()
    # best_rec (C06) measures 0.250 this job, best_known (also C06, from the plan ledger) was
    # 0.257 -- 0.250 <= 1.02*0.257 -> PASS regardless, and here it is literally the same config.
    _write_csv(job_ledger_dir / "ledger_karina_8001.csv",
               _karina_rows(c01=(0.40, 0.41), c06=(0.249, 0.251), c05=(0.262, 0.259)))
    block = ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                          sacct_cmd=_fake_sacct_cmd("8001|strictcamp|COMPLETED|00:12:00|4|448"),
                          results_dir=str(tmp_path / "pass_results"))
    assert block["verdict"] == "PASS"
    assert block["regret"] == pytest.approx(0.0, abs=1e-6)
    assert os.path.exists(block["csv_path"])


def test_reconcile_fail_regret_5_percent(tmp_path):
    # Planning ledger: C01 is the historical fastest (0.10), so best_known == C01 (baseline
    # role only -- NOT one of the "explicit" recommended-like candidates C06/C05). Job ledger:
    # best_known's (C01) OWN median in this job is 1.000; the best recommended candidate (C06,
    # min of the two explicit roles) measures 1.050 -- regret = 1.050/1.000 - 1 = 0.05, which
    # exceeds the 1.02x PASS threshold -> FAIL.
    plan_dir = _planned(tmp_path, "fail_case",
                        _karina_rows(c01=(0.10, 0.10), c06=(1.2, 1.2), c05=(1.3, 1.3)))
    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="8002"))
    job_ledger_dir = tmp_path / "fail_job_ledgers"
    job_ledger_dir.mkdir()
    _write_csv(job_ledger_dir / "ledger_karina_8002.csv",
               _karina_rows(c01=(1.000, 1.000), c06=(1.050, 1.050), c05=(1.100, 1.100)))
    block = ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                          sacct_cmd=_fake_sacct_cmd("8002|strictcamp|COMPLETED|00:12:00|4|448"),
                          results_dir=str(tmp_path / "fail_results"))
    assert block["best_known"]["config_id"] == "C01"
    assert block["verdict"] == "FAIL"
    assert block["regret"] == pytest.approx(0.05, abs=1e-6)
    assert block["speedup_vs_c01"] == pytest.approx(-0.05, abs=1e-6)


def test_reconcile_sacct_failed_state_is_inconclusive(tmp_path):
    plan_dir = _planned(tmp_path, "sacct_fail_case", _karina_rows())
    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="8003"))
    job_ledger_dir = tmp_path / "sacct_fail_job_ledgers"
    job_ledger_dir.mkdir()
    # a fully successful ledger, but the SLURM job itself failed -- must still be INCONCLUSIVE.
    _write_csv(job_ledger_dir / "ledger_karina_8003.csv", _karina_rows())
    block = ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                          sacct_cmd=_fake_sacct_cmd("8003|strictcamp|FAILED|00:03:00|4|448"),
                          results_dir=str(tmp_path / "sacct_fail_results"))
    assert block["verdict"] == "INCONCLUSIVE"


def test_reconcile_est_vs_actual_su(tmp_path):
    plan_dir = _planned(tmp_path, "su_case", _karina_rows())
    plan = json.loads((plan_dir / "plan.json").read_text())
    est_su = plan["cost"]["su"]

    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="8004"))
    job_ledger_dir = tmp_path / "su_job_ledgers"
    job_ledger_dir.mkdir()
    _write_csv(job_ledger_dir / "ledger_karina_8004.csv", _karina_rows())
    # Elapsed=00:30:00 (0.5h), NNodes=4, NCPUS=448 -> actual_su = 448*0.5*0.2 = 44.8
    block = ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                          sacct_cmd=_fake_sacct_cmd("8004|strictcamp|COMPLETED|00:30:00|4|448"),
                          results_dir=str(tmp_path / "su_results"))
    assert block["est_su"] == pytest.approx(est_su, abs=1e-9)
    assert block["actual_su"] == pytest.approx(44.8, abs=1e-9)
    assert block["actual_node_hours"] == pytest.approx(2.0, abs=1e-9)   # 4 nodes * 0.5h


# ---------------------------------------------------------------------------
# explicit --configs bypasses recommend()
# ---------------------------------------------------------------------------
def test_explicit_configs_bypasses_recommend(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("recommend() must not be called when --configs is given")
    fake = types.ModuleType("recommend")
    fake.recommend = _boom
    monkeypatch.setitem(sys.modules, "recommend", fake)

    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C06", "C05"]))

    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit),
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)   # must not raise
    assert plan["recommend"] is None
    ids = {c["config_id"] for c in plan["candidates"]}
    assert ids == {"C01", "C05", "C06"}


def test_explicit_configs_unknown_id_raises(tmp_path):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C99"]))
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], configs=str(explicit),
                       out=tmp_path / "plan")
    with pytest.raises(SystemExit):
        ex.cmd_plan(args)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_reports_totals(tmp_path, capsys):
    plan_dir = _planned(tmp_path, "sum_case", _karina_rows())
    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="8005"))
    job_ledger_dir = tmp_path / "sum_job_ledgers"
    job_ledger_dir.mkdir()
    _write_csv(job_ledger_dir / "ledger_karina_8005.csv", _karina_rows())
    ex.reconcile(str(plan_dir), ledger_dir=str(job_ledger_dir),
                 sacct_cmd=_fake_sacct_cmd("8005|strictcamp|COMPLETED|00:12:00|4|448"),
                 results_dir=str(tmp_path / "sum_results"))
    rows = ex.cmd_summarize(SimpleNamespace(plans=[str(plan_dir)]))
    assert rows[0]["case"] == "karina"
    assert rows[0]["verdict"] in ("PASS", "FAIL")
    out = capsys.readouterr().out
    assert "TOTAL" in out
