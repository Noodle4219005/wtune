#!/usr/bin/env python3
# RETIRED 2026-10-10: tests of the archived experiment engine, cut from
# tests/predictor/test_experiment.py as at 35c1c10. Not runnable as is.
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
import math
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
import decomp_space              # noqa: E402  -- predictor/decomp_space.py: real (not faked) in
                                  # every generated-space test below, stdlib-only, no sklearn
                                  # needed (unlike decomp_search, which every generated-space
                                  # test DOES fake -- see _install_fake_decomp_search)


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
               out=None, reps=2, nodes=4, top_k=3, namelist=None, space="auto",
               screen_spares=0):
    return SimpleNamespace(
        case=case, ledger=ledger_paths or [], namelist=namelist or os.devnull,
        hardware_id="f1-spr-icpnq", nodes=nodes, top_k=top_k, loco=loco, configs=configs,
        reps=reps, out=str(out or (tmp_path / "plan")), space=space,
        screen_spares=screen_spares,
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


# ---------------------------------------------------------------------------
# generated-space fakes: predictor/benchmark.py and predictor/decomp_search.py both need
# scikit-learn/catboost transitively (benchmark.py -> models.py -> sklearn, same as
# decomp_search.py -> models.py), so both are faked the same way _install_fake_recommend fakes
# predictor/recommend.py.
# ---------------------------------------------------------------------------
_NUMERIC_LEDGER_FIELDS = (
    "e_we", "e_sn", "e_vert", "dx", "dy", "time_step", "radt", "cudt",
    "history_interval", "n_nodes", "n_ranks", "omp_threads", "nproc_x",
    "nproc_y", "numtiles", "kmp_blocktime", "s_per_step_median",
    "s_per_step_mean", "n_steps_measured", "diffwrf_differing_fields",
    "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
    "sf_surface_physics", "ra_lw_physics", "ra_sw_physics",
    "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
    "profile_mpi_share", "profile_io_share", "profile_rank_imbalance",
    "profile_thread_imbalance",
)
_NULLABLE_LEDGER_FIELDS = (
    "job_id", "rep_index", "case_raw", "case_base", "case_family",
    "hardware_id", "binary_kind", "compliance_class", "distribution",
    "fidelity_label", "notes", "run_id", "source_dataset",
)


def _install_fake_benchmark(monkeypatch):
    """Fake predictor/benchmark.py: only load_ledger() is needed by
    experiment._call_decomp_search_propose, and its real implementation is plain stdlib csv +
    float coercion (the ModuleNotFoundError comes only from benchmark.py's OWN unconditional
    `import models as modelmod` at module scope, unrelated to load_ledger's own logic) --
    reproduced verbatim here rather than importing the real module."""
    mod = types.ModuleType("benchmark")

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def load_ledger(path):
        with open(path, newline="") as fh:
            rows = []
            for r in csv.DictReader(fh):
                row = dict(r)
                for key in _NUMERIC_LEDGER_FIELDS:
                    row[key] = _f(row.get(key))
                for key in _NULLABLE_LEDGER_FIELDS:
                    if row.get(key) == "":
                        row[key] = None
                rows.append(row)
        return rows

    mod.load_ledger = load_ledger
    monkeypatch.setitem(sys.modules, "benchmark", mod)
    return mod


def _install_fake_decomp_search(monkeypatch, chosen=(), capture=None, stopped=False, reason=None):
    """Inject a fake `decomp_search` module into sys.modules so experiment.py's lazy `import
    decomp_search` picks it up instead of the real predictor/decomp_search.py (which needs
    scikit-learn/catboost via predictor/models.py, unavailable in this test environment). Stands
    in for both case_row_fields() and propose_for_case() (decomp_search.py:505, which itself
    calls propose_core at decomp_search.py:426 in the real module)."""
    mod = types.ModuleType("decomp_search")

    def case_row_fields(case_json):
        return dict(e_we=case_json.get("e_we"), e_sn=case_json.get("e_sn"))

    def propose_for_case(case_name, case_json, case_fields, prior_rows, state, batch_size,
                          total_cores=448, nodes=4, hardware_id=None, kappa=1.5, **kw):
        if capture is not None:
            capture.append(dict(case_name=case_name, case_json=case_json,
                                 case_fields=case_fields, prior_rows=prior_rows, state=state,
                                 batch_size=batch_size, total_cores=total_cores, nodes=nodes,
                                 hardware_id=hardware_id, kappa=kappa, kwargs=kw))
        result = dict(stopped=stopped, round=1, reason=reason,
                      chosen=[] if stopped else list(chosen[:batch_size]))
        return result, []

    mod.case_row_fields = case_row_fields
    mod.propose_for_case = propose_for_case
    monkeypatch.setitem(sys.modules, "decomp_search", mod)
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
# case["max_omp"] (c34/marshall in campaign/cases.json: WRF-Fire crashes rc=137/139 at the
# first d01 step with any omp>1) -- defense-in-depth: recommend()'s own shortlist is already
# built from campaign_configs.resolve_case()'s filtered pool (predictor/recommend.py
# build_candidates), but this fake recommend() bypasses resolve_case entirely, so this test
# exercises _build_candidates's OWN filter, not resolve_case's.
# ---------------------------------------------------------------------------
_SHORTLIST_C08_OMP2 = [
    dict(config_id="C08", rank=1, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32,
         numtiles=2, distribution="cyclic", kmp_blocktime=0,
         predicted_gain=0.05, gain_lower_95=0.01, label="SAFE"),
]


def test_template_space_excludes_recommended_candidate_above_case_max_omp(tmp_path, monkeypatch):
    monkeypatch.setitem(campaign_configs.CASES, "karina",
                         dict(campaign_configs.CASES["karina"], max_omp=1))
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_C08_OMP2, capture=captured)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    ids = {c["config_id"] for c in plan["candidates"]}
    assert "C08" not in ids
    assert "C01" in ids   # baseline, omp=1, unaffected by the cap

    excluded = plan["recommend"]["excluded"]
    assert excluded == [dict(label="C08", omp=2, reason="omp(2) exceeds case max_omp=1")]


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
# --site (decision 0050 step 8, minimal hardware abstraction)
# ---------------------------------------------------------------------------
def test_plan_site_f1_default_is_byte_identical(tmp_path, monkeypatch):
    """No --site (or --site f1 explicitly) must reproduce today's f1-only sbatch command and
    cost line -- this is the regression the whole --site plumbing must never break."""
    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1, capture=captured)
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())

    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], out=tmp_path / "plan")
    plan_no_site = ex.cmd_plan(args)

    args2 = _plan_args(tmp_path, ledger_paths=[str(ledger)], out=tmp_path / "plan2")
    args2.site = "f1"
    plan_explicit_f1 = ex.cmd_plan(args2)

    assert plan_no_site["sbatch"]["command"] == plan_explicit_f1["sbatch"]["command"]
    assert plan_no_site["space"] == "template"
    assert " SU," in plan_no_site["cost"]["cost_line"]
    assert plan_no_site["cost"]["su"] is not None
    assert "--exclusive" not in plan_no_site["sbatch"]["command"]
    assert "SITE=" not in plan_no_site["sbatch"]["command"]


def test_plan_site_scc_forces_generated_space_and_emits_scc_sbatch_args(tmp_path, monkeypatch):
    # karina resolves C01 on f1 (test_plan_site_f1_default_is_byte_identical above confirms
    # --space auto picks 'template' for it there) -- this documents that --site scc overrides
    # that pick to 'generated' regardless, since scc's 72 cores/node resolves none of the
    # C01-C16 STRICT templates (anchored at f1's fixed 448/224/112 rank counts).
    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=[])
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], out=tmp_path / "plan")
    args.site = "scc"
    plan = ex.cmd_plan(args)

    assert plan["space"] == "generated"
    assert plan["site"] == "scc"
    assert plan["partition"] == "compute"

    cmd = plan["sbatch"]["command"]
    assert cmd.startswith("sbatch -p compute -N %d --time " % args.nodes)
    assert "--exclusive" in cmd
    assert "--exclude=head" in cmd
    assert "--ntasks-per-node=72" in cmd
    assert "SITE=scc" in cmd
    assert "WTUNE_TOTAL_CORES=%d" % (args.nodes * 72) in cmd
    assert " -A " not in cmd and "--account" not in cmd

    assert plan["cost"]["su"] is None
    assert "SU" not in plan["cost"]["cost_line"]
    assert "node-hours" in plan["cost"]["cost_line"]


def test_site_table_rejects_unknown_site():
    with pytest.raises(SystemExit):
        ex._site_cfg("nosuchsite")


# ---------------------------------------------------------------------------
# generated space (--space): matthew (91x100, campaign/cases.json) resolves no C01 STRICT
# template at any of the 448/224/112-rank anchors (every legal (nx,ny) at ranks=112 needs
# nx<=9 (91//nx>=10) and ny<=10 (100//ny>=10), so the max legal ranks is 9*10=90, not 112) --
# real motivating case from the problem statement's 6 small cluster cases.
# ---------------------------------------------------------------------------
_HW = "f1-spr-icpnq"


def _matthew_case():
    return campaign_configs.CASES["matthew"]


def _matthew_space(total_cores=112, nodes=1):
    return decomp_space.enumerate_space(_matthew_case(), total_cores=total_cores, nodes=nodes,
                                         hardware_id=_HW)


def _matthew_cost_row(s=1.0):
    """A minimal STRICT row for matthew at n_nodes=1 so _s_per_step_estimates() (unchanged,
    pre-existing logic -- not part of this change) has a case_median fallback for cost
    estimation; matthew has no campaign/configs.py COST_GUESS entry for its grid. Its
    decomposition identity matches none of GBASE/GREC*, so it never becomes a best_known
    candidate -- purely a cost-estimate fallback row."""
    return _row(case_base="matthew", case_raw="matthew", n_nodes="1",
                s_per_step_median=str(s), s_per_step_mean=str(s), n_steps_measured="30")


def test_generated_space_auto_selected_with_max_legal_baseline_and_full_membership(
        tmp_path, monkeypatch):
    case_json = _matthew_case()
    assert campaign_configs.resolve_case(case_json)[0] == [], \
        "matthew must resolve ZERO STRICT templates for this test to be meaningful"

    candidates, axis = _matthew_space()
    # omp=1 has ZERO legal shapes at total_cores=112 (axis["shapes_count"][1] == 0 -- this IS
    # why the baseline needs its own scan down to ranks=90, see _generated_baseline); the
    # recommender searches this same (total_cores=112, nodes=1) pool, whose legal omp values are
    # {2, 7, 14} instead (axis["omp_options"]). A real, legal 2-candidate shortlist (not
    # fabricated), so "every candidate is in the legal space" is exercised on genuine
    # decomp_space output, not hand-rolled numbers.
    assert axis["shapes_count"][1] == 0
    pool = [c for c in candidates if c["tile_shape"] == "1d_y" and c["numtiles"] == 1
            and c["dist"] == "cyclic" and c["blocktime"] == "0"]
    assert pool, "expected at least one legal 1d_y/numtiles=1/cyclic/blocktime=0 candidate"
    captured = []
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [_matthew_cost_row()])
    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=pool[:2], capture=captured)

    args = _plan_args(tmp_path, case="matthew", ledger_paths=[str(ledger)], nodes=1, top_k=2,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    assert plan["space"] == "generated"
    assert plan["baseline_rule"] == ex.GENERATED_BASELINE_RULE

    base = next(c for c in plan["candidates"] if c["config_id"] == "GBASE")
    assert base["roles"] == ["baseline"]
    assert base["ranks"] == 90 and base["omp"] == 1 and base["numtiles"] == 1
    assert base["dist"] == "cyclic" and base["blocktime"] == "0"
    # WRF's real legality gate (share/module_check_a_mundo.F:3014-3029), not the ceil
    # approximation: floor(91/9)=10>=10, floor(100/10)=10>=10 -- both at the MIN_PATCH floor.
    assert (base["nx"], base["ny"]) == (9, 10)
    assert base["patch_x"] is not None and base["patch_y"] is not None

    # ranks=90 (not total_cores=112) is the max legal rank count at omp=1 -- 91,92,...,112 all
    # have zero legal (nx,ny) shapes (nx<=9, ny<=10 caps the product at 90).
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    for ranks in range(91, 113):
        assert decomp_space.legal_shapes_exact(we_list, sn_list, ranks) == []
    assert decomp_space.legal_shapes_exact(we_list, sn_list, 90) == [(9, 10)]

    # every GBASE/GREC* candidate must be a member of the legal space: its own (ranks, nx, ny)
    # passes legal_shapes_exact for that ranks value (checked directly, the same predicate
    # enumerate_space itself is built on).
    for c in plan["candidates"]:
        if c["config_id"] in ("GBASE",) or c["config_id"].startswith("GREC"):
            assert (c["nx"], c["ny"]) in decomp_space.legal_shapes_exact(
                we_list, sn_list, c["ranks"])

    rec_ids = [c["config_id"] for c in plan["candidates"] if c["config_id"].startswith("GREC")]
    assert rec_ids == ["GREC1", "GREC2"]
    for i, cid in enumerate(rec_ids, start=1):
        c = next(x for x in plan["candidates"] if x["config_id"] == cid)
        assert c["roles"] == ["recommended:%d" % i]

    assert len(captured) == 1
    assert captured[0]["kappa"] == 0.0            # pure exploitation, no LCB exploration bonus
    assert captured[0]["batch_size"] == 2          # top_k
    assert captured[0]["state"] == dict(configs={}, measured={}, rounds=[])   # fresh state
    assert plan["recommend"]["space"] == "generated"
    assert plan["recommend"]["kappa"] == 0.0


def test_generated_space_loco_drops_target_rows_before_propose_core_but_best_known_uses_full_ledger(
        tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    rows = (_karina_rows(case="matthew")
            + _karina_rows(case="conus25", c01=(1.0, 1.01), c06=(0.9, 0.91), c05=(0.95, 0.96))
            + [_matthew_cost_row()])
    _write_csv(ledger, rows)

    _install_fake_benchmark(monkeypatch)
    captured = []
    _install_fake_decomp_search(monkeypatch, chosen=[], capture=captured)

    args = _plan_args(tmp_path, case="matthew", ledger_paths=[str(ledger)], loco=True, nodes=1,
                       out=tmp_path / "plan_loco")
    plan = ex.cmd_plan(args)

    assert plan["space"] == "generated"
    assert len(captured) == 1
    seen_cases = {r.get("case_base") for r in captured[0]["prior_rows"]}
    assert "matthew" not in seen_cases, "LOCO must drop every matthew row before propose_core runs"
    assert "conus25" in seen_cases, "LOCO must not touch other cases' rows"

    # best_known must still reflect matthew's OWN full (unfiltered) history: C06 (0.256 median)
    # beats C05 (0.261) and C01 (0.405) in _karina_rows()'s fixture medians (n_nodes="4" there,
    # but this plan is 1-node -- best_known is filtered to nodes=1 rows, so it must be null).
    assert plan["best_known"] is None
    assert plan["recommend"]["loco"] is True
    assert plan["recommend"]["n_ledger_rows_used"] < plan["recommend"]["n_ledger_rows_total"]


def test_generated_space_best_known_uses_full_ledger_on_matching_node_count(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    rows = [dict(r, n_nodes="1") for r in _karina_rows(case="matthew")]
    _write_csv(ledger, rows)
    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=[])

    args = _plan_args(tmp_path, case="matthew", ledger_paths=[str(ledger)], nodes=1,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)
    assert plan["best_known"]["config_id"] == "C06"
    assert plan["best_known"]["measured_median_s_per_step"] == pytest.approx(0.256, abs=1e-9)


def test_generated_space_configs_json_round_trips_and_sbatch_uses_config_file_small_partition(
        tmp_path, monkeypatch):
    candidates, _axis = _matthew_space()
    pool = [c for c in candidates if c["tile_shape"] == "1d_y" and c["numtiles"] == 1
            and c["dist"] == "cyclic" and c["blocktime"] == "0"]
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [_matthew_cost_row()])
    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=pool[:1])

    args = _plan_args(tmp_path, case="matthew", ledger_paths=[str(ledger)], nodes=1, top_k=1,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    assert plan["template_ids"] == []
    assert set(plan["nontemplate_ids"]) == {"GBASE", "GREC1"}

    loaded = campaign_configs.load_generated_configs(plan["configs_json_path"])
    assert {c["id"] for c in loaded} == {"GBASE", "GREC1"}
    for cfg in loaded:
        assert cfg["patch_x"] not in (None, "")
        assert cfg["patch_y"] not in (None, "")

    ids, by_id, reason = campaign_configs.available_decomps(
        "matthew", config_file=plan["configs_json_path"])
    assert reason is None
    assert set(ids) == {"GBASE", "GREC1"}

    assert "CONFIG_FILE=%s" % plan["configs_json_path"] in plan["sbatch"]["command"]
    assert " -p cf112 -N 1 " in plan["sbatch"]["command"]
    assert plan["partition"] == "cf112"


def test_space_generated_forces_it_for_a_template_resolving_case(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    kept, _skipped, _aliases = campaign_configs.resolve_case(campaign_configs.CASES["karina"])
    assert "C01" in {c["id"] for c in kept}, "karina must resolve C01 for this test to be meaningful"

    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=[])

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], space="generated",
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)
    assert plan["space"] == "generated"
    ids = {c["config_id"] for c in plan["candidates"]}
    assert "GBASE" in ids
    assert "C01" not in ids


# ---------------------------------------------------------------------------
# case["max_omp"] on the generated-space path: c34 (campaign/cases.json) carries max_omp=1 for
# the same WRF-Fire omp>1 crash as the template-path tests above -- must (a) reach
# decomp_search.propose_for_case's own max_omp kwarg and (b) drop any omp>max_omp candidate
# still returned, via _build_generated_candidates's own filter (this fake propose_for_case does
# not itself implement the real module's max_omp filtering, so only (b) is what actually keeps
# the omp=2 entry below out of the plan).
# ---------------------------------------------------------------------------
def test_generated_space_excludes_chosen_candidate_above_case_max_omp(tmp_path, monkeypatch):
    case_json = campaign_configs.CASES["c34"]
    assert case_json.get("max_omp") == 1, "c34 must carry max_omp=1 in cases.json for this test"

    chosen = [
        dict(ranks=110, omp=1, nx=10, ny=11, numtiles=1, dist="cyclic", blocktime="0"),
        dict(ranks=56, omp=2, nx=7, ny=8, numtiles=1, dist="block", blocktime="0"),
    ]
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [_row(case_base="c34", case_raw="c34", n_nodes="1",
                              s_per_step_median="0.41339", s_per_step_mean="0.41820",
                              n_steps_measured="30")])
    _install_fake_benchmark(monkeypatch)
    captured = []
    _install_fake_decomp_search(monkeypatch, chosen=chosen, capture=captured)

    args = _plan_args(tmp_path, case="c34", ledger_paths=[str(ledger)], nodes=1, top_k=2,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    assert plan["space"] == "generated"
    assert captured[0]["kwargs"]["max_omp"] == 1   # the cap must reach propose_for_case

    rec = [c for c in plan["candidates"] if c["config_id"].startswith("GREC")]
    assert len(rec) == 1
    assert rec[0]["config_id"] == "GREC1"
    assert rec[0]["omp"] == 1

    excluded = plan["recommend"]["excluded"]
    assert len(excluded) == 1
    assert excluded[0]["omp"] == 2
    assert excluded[0]["reason"] == "omp(2) exceeds case max_omp=1"


def test_configs_flag_rejected_with_space_generated(tmp_path):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(["C06"]))
    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], configs=str(explicit),
                       space="generated", out=tmp_path / "plan")
    with pytest.raises(SystemExit):
        ex.cmd_plan(args)


def test_template_space_unaffected_when_case_resolves_c01(tmp_path, monkeypatch):
    """Template path behaviour must be unchanged for cases that resolve C01 -- auto-selects
    'template' exactly as before this change, and decomp_search is never even imported."""
    # sys.modules[name] = None makes `import name` raise ImportError (a documented CPython
    # import-system convention) -- if the template path ever imports decomp_search, cmd_plan
    # raises SystemExit here instead of returning a plan, failing this test.
    monkeypatch.setitem(sys.modules, "decomp_search", None)
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1, capture=captured)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], out=tmp_path / "plan")
    plan = ex.cmd_plan(args)
    assert plan["space"] == "template"
    assert plan["baseline_rule"] is None
    assert len(captured) == 1


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


# ---------------------------------------------------------------------------
# PART A: feasibility ledger (_failure_index) -- results/derived-data/2026-09-22-atlas-loco-
# phase1-verdicts.csv: 3/10 LOCO cases FAILED-infeasible because a failed row (empty median,
# non-zero rc= in notes) is silently invisible to every candidate-building path above. This index
# reads that dropped half of the ledger directly.
# ---------------------------------------------------------------------------
def _fail_row(case, cid, rc, ranks, omp, nx, ny, numtiles=1, nodes=4, run_suffix="f1", notes=None):
    if notes is None:
        notes = "config=%s rc=%d" % (cid, rc)
    return _row(run_id="%s_%s_%s_9003" % (case, cid, run_suffix), case_base=case, case_raw=case,
                n_ranks=str(ranks), omp_threads=str(omp), nproc_x=str(nx), nproc_y=str(ny),
                numtiles=str(numtiles), s_per_step_median="", s_per_step_mean="",
                n_steps_measured="0", n_nodes=str(nodes), notes=notes)


_C08_IDENTITY = (224, 2, 7, 32, 2, "cyclic", "0", 0, 0, 0, 0, 0)
_C06_IDENTITY = (224, 2, 7, 32, 12, "cyclic", "0", 0, 0, 0, 0, 0)


def test_failure_index_counts_ok_and_fail_separately_by_case_nodes_identity():
    rows = _karina_rows() + [
        _fail_row("karina", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2)]
    idx = ex._failure_index(rows)
    assert idx[("karina", 4, _C08_IDENTITY)] == dict(fail=1, ok=0, rcs=[137], kinds={"crash"})
    # C06's identity (2 successful rows from _karina_rows(), no failures) must be unaffected.
    assert idx[("karina", 4, _C06_IDENTITY)]["ok"] == 2
    assert idx[("karina", 4, _C06_IDENTITY)]["fail"] == 0


def test_failure_index_rc124_is_timeout_other_nonzero_rc_is_crash():
    rows = [_fail_row("elevconv", "C09", 124, ranks=224, omp=2, nx=7, ny=32, numtiles=12),
            _fail_row("marshall", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2,
                      run_suffix="f2")]
    idx = ex._failure_index(rows)
    assert idx[("elevconv", 4, (224, 2, 7, 32, 12, "cyclic", "0", 0, 0, 0, 0, 0))]["kinds"] \
        == {"timeout"}
    assert idx[("marshall", 4, _C08_IDENTITY)]["kinds"] == {"crash"}


def test_failure_index_excludes_physics_edit_failure_but_includes_p00_reference():
    # same decomposition (56 ranks, omp=2, 7x8) under TWO physics rows: one an edited template
    # (P07, a real physics-knob change) that crashed, one the P00 REFERENCE (no edits) that also
    # crashed -- only the P00 row is a decomposition-feasibility observation (PART A's rule).
    physics_edit_fail = _row(
        case_base="c34", case_raw="c34", n_ranks="56", omp_threads="2", nproc_x="7", nproc_y="8",
        numtiles="1", s_per_step_median="", s_per_step_mean="", n_steps_measured="0", n_nodes="1",
        notes="mode=physics template=P07 family=microphysics edits=mp_physics=8 decomp=C01 rc=139")
    p00_fail = _row(
        case_base="c34", case_raw="c34", n_ranks="56", omp_threads="2", nproc_x="7", nproc_y="8",
        numtiles="1", s_per_step_median="", s_per_step_mean="", n_steps_measured="0", n_nodes="1",
        notes="mode=physics template=P00 family=- edits= decomp=C01 rc=139")
    idx = ex._failure_index([physics_edit_fail, p00_fail])
    key = ("c34", 1, (56, 2, 7, 8, 1, "cyclic", "0", 0, 0, 0, 0, 0))
    assert idx[key] == dict(fail=1, ok=0, rcs=[139], kinds={"crash"})


def test_failure_index_skips_non_strict_rows():
    row = _fail_row("karina", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2)
    row["compliance_class"] = "RESEARCH"
    assert ex._failure_index([row]) == {}


# ---------------------------------------------------------------------------
# PART B: same-case known-infeasible exclusion (in plan, non-LOCO) / no leakage under --loco
# ---------------------------------------------------------------------------
def test_template_space_excludes_recommended_candidate_known_infeasible_on_case(
        tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    rows = _karina_rows() + [
        _fail_row("karina", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2)]
    _write_csv(ledger, rows)
    _install_fake_recommend(monkeypatch, _SHORTLIST_C08_OMP2)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    ids = {c["config_id"] for c in plan["candidates"]}
    assert "C08" not in ids
    assert "C01" in ids   # baseline unaffected
    excluded = plan["recommend"]["excluded"]
    assert dict(label="C08", reason="known_infeasible_on_case", rcs=[137],
                kinds=["crash"]) in excluded


def test_template_space_loco_does_not_leak_target_case_own_failure_history(tmp_path, monkeypatch):
    # WITHOUT --loco this exact fixture excludes C08 (proved above). WITH --loco for karina, the
    # failure index must be built from train_rows (karina's own rows dropped BEFORE the index is
    # built), so C08 must survive here -- excluding it WOULD be leakage: the target case's own
    # measured failure informing what a supposedly-blind recommendation run is allowed to keep.
    ledger = tmp_path / "ledger.csv"
    rows = _karina_rows() + [
        _fail_row("karina", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2)]
    _write_csv(ledger, rows)
    _install_fake_recommend(monkeypatch, _SHORTLIST_C08_OMP2)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], loco=True,
                       out=tmp_path / "plan_loco")
    plan = ex.cmd_plan(args)

    ids = {c["config_id"] for c in plan["candidates"]}
    assert "C08" in ids
    excluded = plan["recommend"]["excluded"]
    assert not any(e.get("reason") == "known_infeasible_on_case" for e in excluded)


def test_template_space_keeps_flaky_candidate_with_mixed_evidence(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    rows = _karina_rows() + [
        _row(run_id="karina_C08_ok_9004", case_base="karina", case_raw="karina", n_ranks="224",
             omp_threads="2", nproc_x="7", nproc_y="32", numtiles="2", s_per_step_median="0.30",
             s_per_step_mean="0.30", n_steps_measured="30", notes="config=C08 rc=0"),
        _fail_row("karina", "C08", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2),
    ]
    _write_csv(ledger, rows)
    _install_fake_recommend(monkeypatch, _SHORTLIST_C08_OMP2)

    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    c08 = next(c for c in plan["candidates"] if c["config_id"] == "C08")
    assert c08["flaky"] is True
    assert not any(e.get("label") == "C08" for e in plan["recommend"]["excluded"])
    other_ids = {c["config_id"] for c in plan["candidates"] if c["config_id"] != "C08"}
    assert all(not c.get("flaky") for c in plan["candidates"] if c["config_id"] in other_ids)


def test_generated_space_excludes_chosen_candidate_known_infeasible_on_case(tmp_path, monkeypatch):
    # generated-space equivalent, c34-shaped: the SAME identity (110 ranks, omp=1, 10x11) that
    # crashed before must be excluded from a fresh plan even though its omp already clears
    # max_omp=1 (proving this is PART B's OWN filter, not PART D's/PART A's max_omp filter).
    case_json = campaign_configs.CASES["c34"]
    assert case_json.get("max_omp") == 1
    chosen = [dict(ranks=110, omp=1, nx=10, ny=11, numtiles=1, dist="cyclic", blocktime="0")]
    ledger = tmp_path / "ledger.csv"
    rows = [_fail_row("c34", "R01D0001", 139, ranks=110, omp=1, nx=10, ny=11, numtiles=1,
                       nodes=1),
            # a successful row too, purely so _s_per_step_estimates has a case_median fallback
            # for GBASE's own cost estimate (c34 has no campaign/configs.py COST_GUESS entry).
            _row(case_base="c34", case_raw="c34", n_nodes="1", s_per_step_median="0.41339",
                 s_per_step_mean="0.41820", n_steps_measured="30")]
    _write_csv(ledger, rows)
    _install_fake_benchmark(monkeypatch)
    _install_fake_decomp_search(monkeypatch, chosen=chosen)

    args = _plan_args(tmp_path, case="c34", ledger_paths=[str(ledger)], nodes=1, top_k=1,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    rec = [c for c in plan["candidates"] if c["config_id"].startswith("GREC")]
    assert rec == []
    excluded = plan["recommend"]["excluded"]
    assert any(e.get("reason") == "known_infeasible_on_case" for e in excluded)


# ---------------------------------------------------------------------------
# PART D: c34's generated-space max_omp=1 caused ZERO recommended candidates (job 1090690,
# "after fix propose returns 0 recs") -- diagnosed as decomp_search.propose_for_case's exact-fit
# ranks=total_cores//omp having no legal shape at omp=1/total_cores=112 for c34's 118x118 grid
# (every factor pair of 112 needs a side >11; floor(118/x)>=10 caps x<=11). Fix:
# _call_decomp_search_propose now downscales total_cores to the largest under-subscribed legal
# ranks at max_omp (the SAME rule _generated_baseline already used for GBASE) before calling
# decomp_search.propose_for_case, which is NOT owned by this change.
# ---------------------------------------------------------------------------
def test_generated_space_c34_downscales_total_cores_so_propose_gets_a_legal_ranks_value(
        tmp_path, monkeypatch):
    case_json = campaign_configs.CASES["c34"]
    assert case_json.get("max_omp") == 1
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [_row(case_base="c34", case_raw="c34", n_nodes="1",
                              s_per_step_median="0.41339", s_per_step_mean="0.41820",
                              n_steps_measured="30")])
    _install_fake_benchmark(monkeypatch)
    captured = []
    # a REAL (not fabricated) omp=1 candidate at the downscaled ranks -- proves the fake
    # decomp_search actually received a total_cores whose exact-fit division (110 // 1) is a
    # legal ranks value, by construction of this assertion against the real decomp_space module.
    chosen = [dict(ranks=110, omp=1, nx=10, ny=11, numtiles=1, dist="cyclic", blocktime="0")]
    _install_fake_decomp_search(monkeypatch, chosen=chosen, capture=captured)

    args = _plan_args(tmp_path, case="c34", ledger_paths=[str(ledger)], nodes=1, top_k=1,
                       out=tmp_path / "plan")
    plan = ex.cmd_plan(args)

    assert len(captured) == 1
    effective_total_cores = captured[0]["total_cores"]
    assert effective_total_cores == 110          # NOT the naive nodes*112 == 112
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    assert decomp_space.legal_shapes_exact(we_list, sn_list, effective_total_cores // 1) != []

    rec = [c for c in plan["candidates"] if c["config_id"].startswith("GREC")]
    assert len(rec) == 1 and rec[0]["omp"] == 1   # PART D: >=1 omp=1 candidate, not zero


def test_generated_space_matthew_total_cores_unchanged_no_max_omp(tmp_path, monkeypatch):
    # matthew has no max_omp at all -- the PART D downscale must never trigger for it, so
    # enumerate_space keeps seeing the FULL nodes*112 total_cores exactly as before PART D
    # (predictor/tests/test_decomp_space.py's own shapes_count[1]==0 assertion for a tiny grid
    # depends on this NOT changing).
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, [_matthew_cost_row()])
    _install_fake_benchmark(monkeypatch)
    captured = []
    _install_fake_decomp_search(monkeypatch, chosen=[], capture=captured)

    args = _plan_args(tmp_path, case="matthew", ledger_paths=[str(ledger)], nodes=1, top_k=1,
                       out=tmp_path / "plan")
    ex.cmd_plan(args)
    assert captured[0]["total_cores"] == 112


# ---------------------------------------------------------------------------
# PART E: recommend.build_candidates caps omp=1 when the namelist's &fire ifire>0 on any active
# domain, independent of whether the case is even registered in campaign/cases.json. Tested
# against the REAL predictor/recommend.py (not the fakes _install_fake_recommend/_install_fake_
# decomp_search inject elsewhere in this file): its sklearn-only dependency modules (benchmark/
# decomp_search/models/physics/validate -- see predictor/models.py's `from sklearn...` import,
# unavailable in this test environment, same reason experiment.py's own module docstring gives
# for never importing predictor/recommend.py directly) are stubbed as attribute-permissive dummy
# modules; parse_case_from_namelist/build_candidates (this test's only targets) never touch them,
# but recommend.py's own module-scope code (argparse defaults sourced from e.g.
# physics.DEFAULT_MAX_TEMPLATE_LARGE_RATE) evaluates at import time and needs SOME attribute
# value there, not a real one.
# ---------------------------------------------------------------------------
def _import_real_recommend(monkeypatch):
    class _AnyAttrModule(types.ModuleType):
        def __getattr__(self, name):
            return 0

    for name in ("benchmark", "decomp_search", "models", "physics", "validate"):
        monkeypatch.setitem(sys.modules, name, _AnyAttrModule(name))
    monkeypatch.delitem(sys.modules, "recommend", raising=False)
    import recommend as recommend_mod
    monkeypatch.delitem(sys.modules, "recommend", raising=False)  # don't leak the real module
    return recommend_mod


_KARINA_GRID_NAMELIST = (
    "&domains\n"
    " max_dom = 1,\n"
    " e_we = 901,\n"
    " e_sn = 901,\n"
    " e_vert = 45,\n"
    " dx = 1000,\n"
    " dy = 1000,\n"
    " time_step = 6,\n"
    "/\n"
    "&physics\n"
    " mp_physics = 6,\n"
    "/\n"
    "&fire\n"
    " ifire = %s,\n"
    "/\n"
)


_TWO_DOMAIN_FIRE_NAMELIST = (
    "&domains\n"
    " max_dom = 2,\n"
    " e_we = 118, 118,\n"
    " e_sn = 118, 118,\n"
    " e_vert = 45, 45,\n"
    " dx = 1000,\n"
    " dy = 1000,\n"
    " time_step = 6,\n"
    "/\n"
    "&physics\n"
    " mp_physics = 6, 6,\n"
    "/\n"
    "&fire\n"
    " ifire = 0, 2,\n"
    "/\n"
)


def test_recommend_parses_ifire_list_from_namelist_fire_group(tmp_path, monkeypatch):
    # c34-shaped 2-domain namelist (d01 ifire=0, d02 ifire=2, matching cases.json's own c34/
    # marshall notes) -- proves ifire is read per-domain, like e_we/e_sn, not just d01's value.
    recommend_mod = _import_real_recommend(monkeypatch)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(_TWO_DOMAIN_FIRE_NAMELIST)
    case_fields = recommend_mod.parse_case_from_namelist(str(namelist))
    assert case_fields["ifire_list"] == [0, 2]


def test_recommend_build_candidates_caps_omp_when_namelist_ifire_positive(tmp_path, monkeypatch):
    recommend_mod = _import_real_recommend(monkeypatch)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(_KARINA_GRID_NAMELIST % "2")
    case_fields = recommend_mod.parse_case_from_namelist(str(namelist))
    assert case_fields["ifire_list"] == [2]

    # case_name="karina" is registered in cases.json but carries NO max_omp of its own -- proves
    # the cap comes from ifire, via min(1, case_dict.get("max_omp", 1)), not from cases.json.
    assert campaign_configs.CASES["karina"].get("max_omp") is None
    kept, _skipped, _aliases, _dn, _nn = recommend_mod.build_candidates(
        case_fields, nodes=4, max_dom=1, case_name="karina")
    assert kept, "expected at least one legal STRICT template for this 901x901 grid"
    assert all(c["omp"] == 1 for c in kept)


def test_recommend_build_candidates_no_ifire_allows_omp_above_1(tmp_path, monkeypatch):
    # control: the SAME grid, ifire=0 everywhere -- omp>1 templates must NOT be capped.
    recommend_mod = _import_real_recommend(monkeypatch)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(_KARINA_GRID_NAMELIST % "0")
    case_fields = recommend_mod.parse_case_from_namelist(str(namelist))
    assert case_fields["ifire_list"] == [0]

    kept, _skipped, _aliases, _dn, _nn = recommend_mod.build_candidates(
        case_fields, nodes=4, max_dom=1, case_name="karina")
    assert any(c["omp"] > 1 for c in kept)


# ---------------------------------------------------------------------------
# PART C: feasibility screen -- --screen-spares S (default 0 = old behaviour, exercised by every
# test above this section) writes a cheap MODE=timing/REPS=1/STEPS=SCREEN_STEPS screen job
# instead of the direct timing plan; `screen` reads its ledger back and rewrites the plan for the
# real timing job with only the candidates the screen job proved feasible.
# ---------------------------------------------------------------------------
_SHORTLIST_4 = [
    dict(config_id="C20", rank=1, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32, numtiles=2,
         distribution="cyclic", kmp_blocktime=0, predicted_gain=0.06, gain_lower_95=0.02,
         label="SAFE"),
    dict(config_id="C21", rank=2, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32, numtiles=3,
         distribution="cyclic", kmp_blocktime=0, predicted_gain=0.05, gain_lower_95=0.01,
         label="SAFE"),
    dict(config_id="C22", rank=3, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32, numtiles=4,
         distribution="cyclic", kmp_blocktime=0, predicted_gain=0.04, gain_lower_95=-0.01,
         label="EXPERIMENTAL"),
    dict(config_id="C23", rank=4, n_ranks=224, omp_threads=2, nproc_x=7, nproc_y=32, numtiles=5,
         distribution="cyclic", kmp_blocktime=0, predicted_gain=0.03, gain_lower_95=-0.02,
         label="EXPERIMENTAL"),
]


def test_screen_spares_zero_is_old_behaviour(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1)
    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], out=plan_dir)
    plan = ex.cmd_plan(args)
    assert plan["screen_spares"] == 0
    assert plan["screen"] is None
    assert not os.path.isdir(str(plan_dir / "screen"))
    assert plan["cost"]["reps"] == 2   # args.reps, unchanged behaviour


def test_screen_plan_writes_screen_job_params(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    captured = []
    _install_fake_recommend(monkeypatch, _SHORTLIST_4, capture=captured)

    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], top_k=2, out=plan_dir,
                       screen_spares=2)
    plan = ex.cmd_plan(args)

    assert captured[0]["top_k"] == 4     # top_k(2) + screen_spares(2) reached the recommender
    rec_ids = sorted(c["config_id"] for c in plan["candidates"]
                      if any(r.startswith("recommended:") for r in c["roles"]))
    assert rec_ids == ["C20", "C21", "C22", "C23"]     # all 4 screened, not just top_k=2

    sblock = plan["screen"]
    assert sblock["spares"] == 2 and sblock["top_k"] == 2 and sblock["steps"] == ex.SCREEN_STEPS
    assert "STEPS=%d" % ex.SCREEN_STEPS in sblock["sbatch"]["command"]
    assert "REPS=1" in sblock["sbatch"]["command"]
    assert "MODE=timing" in sblock["sbatch"]["command"]
    assert "TIMEOUT_S=%d" % sblock["timeout_s"] in sblock["sbatch"]["command"]
    assert plan["sbatch"]["command"] == sblock["sbatch"]["command"]   # top-level == screen job

    startup_s = sblock["cost"]["startup_s"]
    s_est_max = max(sblock["cost"]["s_per_step_est"].values())
    expected_timeout = max(
        ex.SCREEN_TIMEOUT_MIN_S,
        int(math.ceil(ex.SCREEN_TIMEOUT_MULTIPLIER * (startup_s + ex.SCREEN_STEPS * s_est_max))))
    assert sblock["timeout_s"] == expected_timeout

    assert os.path.exists(sblock["configs_json_path"])
    assert sblock["configs_json_path"] == plan["configs_json_path"]
    assert plan["cost"]["reps"] == 1
    assert plan["cost"]["steps"] == ex.SCREEN_STEPS


def test_screen_to_timing_rewrite_keeps_first_k_feasible_in_rank_order_plus_controls(
        tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    _install_fake_recommend(monkeypatch, _SHORTLIST_4)

    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], top_k=2,
                       out=plan_dir, screen_spares=2)
    plan = ex.cmd_plan(args)
    assert set(plan["screen"]["candidate_ids"]) == {"C01", "C06", "C20", "C21", "C22", "C23"}

    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="9101", stage="screen"))

    # screen ledger: rank-1 C20 crashes (rc=137, infeasible); C21/C22/C23 all measure fine
    # (feasible); baseline C01 ALSO crashes in the screen -- it must survive anyway ("always").
    screen_ledger_dir = tmp_path / "screen_job_ledgers"
    screen_ledger_dir.mkdir()
    screen_rows = [
        _fail_row("karina", "C01", 99, ranks=448, omp=1, nx=8, ny=56, numtiles=1, run_suffix="s"),
        _fail_row("karina", "C20", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=2, run_suffix="s"),
        _row(run_id="karina_C21_s_9101", case_base="karina", case_raw="karina", n_ranks="224",
             omp_threads="2", nproc_x="7", nproc_y="32", numtiles="3", s_per_step_median="0.20",
             s_per_step_mean="0.20", n_steps_measured=str(ex.SCREEN_STEPS),
             notes="config=C21 rc=0"),
        _row(run_id="karina_C22_s_9101", case_base="karina", case_raw="karina", n_ranks="224",
             omp_threads="2", nproc_x="7", nproc_y="32", numtiles="4", s_per_step_median="0.21",
             s_per_step_mean="0.21", n_steps_measured=str(ex.SCREEN_STEPS),
             notes="config=C22 rc=0"),
        _row(run_id="karina_C23_s_9101", case_base="karina", case_raw="karina", n_ranks="224",
             omp_threads="2", nproc_x="7", nproc_y="32", numtiles="5", s_per_step_median="0.22",
             s_per_step_mean="0.22", n_steps_measured=str(ex.SCREEN_STEPS),
             notes="config=C23 rc=0"),
        # best_known C06 is untouched by this screen job at all (no row for it) -- must still
        # survive into the final timing plan via its best_known role.
    ]
    _write_csv(screen_ledger_dir / "ledger_karina_9101.csv", screen_rows)

    plan = ex.screen(str(plan_dir), ledger_dir=str(screen_ledger_dir))

    assert plan["screen"]["kept_recommended_ids"] == ["C21", "C22"]   # first 2 feasible, in rank order
    assert set(plan["screen"]["dropped_ids"]) == {"C20", "C23"}       # infeasible + excess feasible
    assert plan["screen"]["underfilled"] is False

    final_ids = {c["config_id"] for c in plan["candidates"]}
    assert final_ids == {"C01", "C06", "C21", "C22"}   # baseline + best_known ALWAYS kept
    assert "C20" not in final_ids and "C23" not in final_ids

    # rewritten for the REAL timing job: REPS back to plan["reps"] (2), full case steps (not
    # SCREEN_STEPS), MODE=timing, and record()/reconcile() need no --stage afterward.
    assert plan["cost"]["reps"] == 2
    assert plan["cost"]["steps"] != ex.SCREEN_STEPS
    assert "REPS=2" in plan["sbatch"]["command"]
    assert "STEPS=" not in plan["sbatch"]["command"]
    assert plan["state"] == "draft"

    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="9102"))   # no --stage: timing
    reloaded = json.loads((plan_dir / "plan.json").read_text())
    assert reloaded["job_id"] == "9102"
    assert reloaded["screen"]["job_id"] == "9101"   # screen's own job_id is untouched


def test_screen_fewer_than_top_k_feasible_keeps_what_exists_and_notes_underfilled(
        tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    _install_fake_recommend(monkeypatch, _SHORTLIST_4)

    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, case="karina", ledger_paths=[str(ledger)], top_k=3,
                       out=plan_dir, screen_spares=1)
    ex.cmd_plan(args)
    ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="9201", stage="screen"))

    screen_ledger_dir = tmp_path / "screen_job_ledgers"
    screen_ledger_dir.mkdir()
    # only ONE of the 4 screened recommended candidates (top_k=3, spares=1) comes back feasible.
    screen_rows = [
        _fail_row("karina", "C20", 124, ranks=224, omp=2, nx=7, ny=32, numtiles=2, run_suffix="u"),
        _fail_row("karina", "C22", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=4, run_suffix="u"),
        _fail_row("karina", "C23", 137, ranks=224, omp=2, nx=7, ny=32, numtiles=5, run_suffix="u"),
        _row(run_id="karina_C21_u_9201", case_base="karina", case_raw="karina", n_ranks="224",
             omp_threads="2", nproc_x="7", nproc_y="32", numtiles="3", s_per_step_median="0.20",
             s_per_step_mean="0.20", n_steps_measured=str(ex.SCREEN_STEPS),
             notes="config=C21 rc=0"),
    ]
    _write_csv(screen_ledger_dir / "ledger_karina_9201.csv", screen_rows)

    plan = ex.screen(str(plan_dir), ledger_dir=str(screen_ledger_dir))

    assert plan["screen"]["kept_recommended_ids"] == ["C21"]
    assert plan["screen"]["underfilled"] is True
    final_rec_ids = {c["config_id"] for c in plan["candidates"]
                     if any(r.startswith("recommended:") for r in c["roles"])}
    assert final_rec_ids == {"C21"}


def test_record_stage_screen_requires_screen_block(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.csv"
    _write_csv(ledger, _karina_rows())
    _install_fake_recommend(monkeypatch, _SHORTLIST_C06_TOP1)
    plan_dir = tmp_path / "plan"
    args = _plan_args(tmp_path, ledger_paths=[str(ledger)], out=plan_dir)   # screen_spares=0
    ex.cmd_plan(args)
    with pytest.raises(SystemExit):
        ex.cmd_record(SimpleNamespace(plan=str(plan_dir), job_id="1", stage="screen"))
