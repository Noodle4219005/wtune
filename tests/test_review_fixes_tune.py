"""Regression tests for the 8bdcd3c source review (tune side): config-id reuse across rounds,
sacct misses, run ids of cases with '_', re-proposal of failed configs, numtiles == omp under
the cap, rebuild-job ownership, the rebuild-prep wait in stage1=t0 and WINNER vs BEST."""
import datetime as dt
import json
import types

import pytest

from wtune import tune as wt
from wtune.predictor import decomp_search            # predictor/ is on sys.path once wtune.tune is imported
from wtune.predictor import decomp_space
from test_wtune_tune import (FakeClock, MATTHEW_LIKE, _ctx, _pipe_run, make_fake_sacct,
                             make_fake_sbatch, make_fake_scancel)


def _cfg(cid, ranks=72, omp=1, nx=8, ny=9, numtiles=1, **kw):
    return dict(dict(id=cid, ranks=ranks, omp=omp, nx=nx, ny=ny, numtiles=numtiles, dist="block",
                     blocktime="0", tile_shape="1d_y"), **kw)


# ---------------------------------------------------------------------------
# 1: fallback / top-up config ids are unique within one tune
# ---------------------------------------------------------------------------
def test_diverse_sample_ids_continue_after_the_ids_already_used():
    ctx = _ctx()
    first = wt.sample_more_diverse_configs(ctx, [], 3, seed=1)
    assert [c["id"] for c in first] == ["D0001", "D0002", "D0003"]      # first call unchanged
    second = wt.sample_more_diverse_configs(ctx, first, 3, seed=2)
    assert [c["id"] for c in second] == ["D0004", "D0005", "D0006"]
    idents = [wt.sweep._identity(c) for c in first + second]
    assert len(set(idents)) == 6                                        # six different layouts


def test_adaptive_fallback_rounds_never_overwrite_an_earlier_rounds_config():
    sbatch_runner, _calls, paths = make_fake_sbatch(start=2000)
    submitted = []

    def ledger_reader(case, job_ids):
        rows = []
        for j in job_ids:
            with open(paths[j]) as f:
                for c in json.load(f)["configs"]:
                    submitted.append(c)
                    rows.append(dict(run_id="%s_%s_r1_%s" % (case, c["id"], j), job_id=j,
                                     s_per_step_median="1.0"))
        return rows

    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, budget_s=3600.0,
               proposer=lambda state, batch_size, kappa=None: (dict(stopped=True, reason="x"), []))
    all_configs_by_cid, all_results = {}, {}
    wt.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)
    assert len(ctx.owned_jobs) >= 2                                     # several fallback rounds
    assert len(all_configs_by_cid) == len(submitted) == len(all_results)
    for c in submitted:                 # every id still names the layout that was measured
        assert wt.sweep._identity(all_configs_by_cid[c["id"]]) == wt.sweep._identity(c)


# ---------------------------------------------------------------------------
# 2: a job sacct does not list is still in flight
# ---------------------------------------------------------------------------
def _scripted_sacct(script):
    """sacct runner answering poll i with script[i] (the last entry repeats); "" = no row."""
    polls = []

    def runner(cmd):
        polls.append(cmd)
        return script[min(len(polls) - 1, len(script) - 1)]

    return runner, polls


@pytest.mark.parametrize("budget_mode", ["wall", "running"])
def test_wait_for_jobs_keeps_waiting_while_sacct_does_not_list_the_job(budget_mode):
    # a transient sacct failure / a job slurmdbd has not recorded yet: "" twice, then the job
    runner, polls = _scripted_sacct(["", "", "9|RUNNING|00:00:10", "", "9|COMPLETED|00:01:00"])
    scancel, cancelled = make_fake_scancel()
    ctx = _ctx(sacct_runner=runner, scancel_runner=scancel, poll_interval=10.0, budget_s=3600.0)
    ctx.budget_mode = budget_mode
    states = wt.wait_for_jobs(ctx, ["9"], "t")
    assert states == {"9": ("COMPLETED", "00:01:00")}
    assert len(polls) == 5 and not cancelled


def test_wait_for_jobs_gives_up_after_consecutive_sacct_misses(capsys):
    runner, polls = _scripted_sacct([""])
    ctx = _ctx(sacct_runner=runner, poll_interval=10.0, budget_s=36000.0)
    assert wt.wait_for_jobs(ctx, ["9"], "t") == {}
    assert len(polls) == wt.jobs.SACCT_MISS_MAX
    assert "job 9 not reported by sacct" in capsys.readouterr().out


def test_wait_for_jobs_cancels_an_unlisted_job_at_the_deadline():
    runner, _polls = _scripted_sacct([""])
    scancel, cancelled = make_fake_scancel()
    ctx = _ctx(sacct_runner=runner, scancel_runner=scancel, poll_interval=600.0, budget_s=900.0)
    wt.wait_for_jobs(ctx, ["9"], "t")
    assert cancelled == [["9"]]                 # it may be running: never left behind uncancelled


def test_default_sacct_runner_reports_nothing_when_sacct_fails():
    assert wt.sweep.default_sacct_runner(["sh", "-c", "echo '9|COMPLETED|00:01:00'; exit 1"]) == ""
    ok = wt.sweep.default_sacct_runner(["sh", "-c", "echo '9|COMPLETED|00:01:00'"])
    assert wt.sweep.sacct_states(["9"], sacct_runner=lambda cmd: ok) == {
        "9": ("COMPLETED", "00:01:00")}


# ---------------------------------------------------------------------------
# 3: run ids of a case id containing '_'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("parse", [wt.parse_run_id, decomp_search.parse_run_id])
def test_parse_run_id_allows_underscores_in_the_case(parse):
    assert parse("my_case_dir_D0001_r2_1234") == dict(case="my_case_dir", cid="D0001", rep="2",
                                                      job="1234")
    assert parse("karina_R01D0003_r1_1078123") == dict(case="karina", cid="R01D0003", rep="1",
                                                       job="1078123")
    assert parse("matthewp0_ANCHOREND_r1_7")["cid"] == "ANCHOREND"
    assert parse("not-a-run-id") is None


def test_aggregate_by_cid_finds_the_configs_of_a_case_with_underscores():
    cfg = _cfg("G0001")
    rows = [dict(run_id="sc26_p1_G0001_r1_55", s_per_step_median="0.5"),
            dict(run_id="sc26_p1_ANCHOR_r1_55", s_per_step_median="0.6")]
    res = wt.aggregate_by_cid(rows, {"G0001": cfg, "ANCHOR": _cfg("ANCHOR")})
    assert set(res) == {"G0001", "ANCHOR"} and res["G0001"]["config"] is cfg


def test_report_text_survives_a_best_row_without_a_config():
    ctx = _ctx()
    res = {"X1": dict(cid="X1", config=None, s_values=[0.5], median=0.5, lo=0.5, hi=0.5, n=1)}
    report = wt.build_report(ctx, 1, "note", res, None, None)
    assert report["best"]["id"] == "X1" and report["param_table"] is None
    text = wt.render_report_text(report)                    # raised TypeError before
    assert "X1" in text and "BEST:" not in text


# ---------------------------------------------------------------------------
# 4: a submitted config that failed is not proposed again
# ---------------------------------------------------------------------------
def test_measured_candidate_keys_cover_submitted_and_proposed_configs():
    a, b, c, d = (_cfg("A"), _cfg("B", nx=9, ny=8), _cfg("C", nx=6, ny=12),
                  _cfg("D", nx=12, ny=6))
    key = decomp_search.candidate_key
    state = dict(configs=dict(A=a, B=b, C=c, D=d), measured=dict(A=[{}]), submitted=["A", "B"],
                 rounds=[dict(round=1, proposed_ids=["C"])])
    assert decomp_search.measured_candidate_keys(state) == {key(a), key(b), key(c)}
    # a state that registers the whole pool up front (simulate) still excludes the measured only
    assert decomp_search.measured_candidate_keys(dict(configs=dict(A=a, B=b),
                                                      measured=dict(A=[{}]))) == {key(a)}


def test_adaptive_rounds_tell_the_proposer_which_configs_failed():
    failed0, ok0, failed1 = _cfg("G0001", nx=9, ny=8), _cfg("ANCHOR"), _cfg("R01D0001", nx=6, ny=12)
    seen = []

    def proposer(state, batch_size, kappa=None):
        seen.append(decomp_search.measured_candidate_keys(state))
        if len(seen) == 1:
            return dict(stopped=False, round=1), [failed1]
        return dict(stopped=False, round=len(seen)), [_cfg("R%02dD0001" % len(seen), nx=4,
                                                           ny=18 + len(seen))]

    ctx = _ctx(proposer=proposer, budget_s=3600.0, ledger_reader=lambda case, job_ids: [])
    all_configs_by_cid = {"ANCHOR": ok0, "G0001": failed0}        # G0001 ran in round 0, no row
    all_results = {"ANCHOR": dict(cid="ANCHOR", config=ok0, s_values=[1.0], median=1.0, lo=1.0,
                                  hi=1.0, n=1)}
    wt.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)
    key = decomp_search.candidate_key
    assert {key(ok0), key(failed0)} <= seen[0]
    assert key(failed1) not in seen[0] and key(failed1) in seen[1]     # round 1's run failed


# ---------------------------------------------------------------------------
# 5: numtiles == omp survives the max_numtiles cap
# ---------------------------------------------------------------------------
def test_numtiles_candidates_always_keep_numtiles_equal_omp():
    opts = decomp_space.DEFAULT_TILE_OPTIONS
    cap = opts["max_numtiles"]
    at72 = decomp_space.numtiles_candidates(72, opts)
    assert 72 in at72 and 144 not in at72                   # only k=1 is exempt from the cap
    assert [n for n in at72 if n != 72] == [n for n in at72 if n <= cap and n != 72]
    for omp in (1, 2, 7, 14, 18, 36):                       # omp <= cap: nothing changes
        got = decomp_space.numtiles_candidates(omp, opts)
        assert omp in got and all(n <= cap for n in got)
    assert 8 not in decomp_space.numtiles_candidates(4, dict(numtiles_multiples=(2,),
                                                              max_numtiles=4))


def test_enumerate_space_has_a_numtiles_72_candidate_at_omp_72():
    cands, _axis = decomp_space.enumerate_space(MATTHEW_LIKE, 4 * 72, 4, "scc-gh200")
    assert {c["numtiles"] for c in cands if c["omp"] == 72} >= {1, 72}


# ---------------------------------------------------------------------------
# 6: a rebuild job is owned as soon as it is submitted
# ---------------------------------------------------------------------------
def _rebuild_env(monkeypatch, tmp_path, sbatch_runner):
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    monkeypatch.setattr(wt.probe_plan, "plan_rebuild_segments",
                        lambda cdir, pts, cfg, pdir, **k: dict(pdir=pdir))
    monkeypatch.setattr(wt.coldstart, "resolve_and_fetch_forcing", lambda *a, **k: (0, {}))
    monkeypatch.setattr(wt.budget, "rebuild_segment_timeout_min", lambda *a, **k: 30)
    monkeypatch.setattr(wt.coldstart, "stage_plan", lambda pdir, **k: pdir + "/run.sbatch")
    monkeypatch.setattr(wt.sweep, "build_anchor", lambda cj, cores, nodes, hw: _cfg("ANCHOR"))
    ctx = _ctx(out_dir=str(tmp_path / "o"), sbatch_runner=sbatch_runner)
    pts = [types.SimpleNamespace(at=dt.datetime(2021, 7, 19, 18 + h), reason="auto")
           for h in range(3)]
    return ctx, pts, (_cfg("ANCHOR"), str(case_dir), None, {}, str(tmp_path / "rebuilt"), 20, 17,
                      36.0)


def test_submit_rebuilt_points_owns_the_jobs_submitted_before_another_point_fails(monkeypatch,
                                                                                tmp_path):
    submitted = []

    def sbatch(cmd):
        if "pt_1" in cmd:
            raise RuntimeError("sbatch failed")
        job = str(700 + int(cmd.split("pt_")[1][0]))
        submitted.append(job)
        return "Submitted batch job %s\n" % job

    ctx, pts, args = _rebuild_env(monkeypatch, tmp_path, sbatch)
    with pytest.raises(RuntimeError, match="sbatch failed"):
        wt.submit_rebuilt_points(ctx, pts, *args)
    # a point not yet started when pt_1 fails is cancelled and never submitted; whatever WAS
    # submitted must be owned (cancellable), not orphaned
    assert submitted and sorted(ctx.owned_jobs) == sorted(submitted)


def test_submit_rebuilt_points_keeps_point_order_when_all_succeed(monkeypatch, tmp_path):
    ctx, pts, args = _rebuild_env(
        monkeypatch, tmp_path,
        lambda cmd: "Submitted batch job %s\n" % (700 + int(cmd.split("pt_")[1][0])))
    ctx.owned_jobs.append("1")
    handle = wt.submit_rebuilt_points(ctx, pts, *args)
    assert handle["job_ids"] == ["700", "701", "702"]
    assert ctx.owned_jobs == ["1", "700", "701", "702"]


# ---------------------------------------------------------------------------
# 7: stage1=t0 -- the rebuild prep is awaited outside --minutes
# ---------------------------------------------------------------------------
def _prep_ctx(tmp_path):
    states = {"5": ("RUNNING", "00:10:00")}
    polls = []

    def sacct(cmd):
        polls.append(1)
        if len(polls) >= 3:
            states["5"] = ("COMPLETED", "00:12:00")
        return make_fake_sacct(states)(cmd)

    scancel, cancelled = make_fake_scancel()
    clock = FakeClock()
    ctx = _ctx(clock=clock, sacct_runner=sacct, scancel_runner=scancel, budget_s=1800.0,
               poll_interval=30.0)
    ctx.budget_mode = "running"
    clock.t = 1800.0                               # stage 1 used all of --minutes
    handle = dict(points=[], done=[], job_ids=["5"], anchor=_cfg("ANCHOR"), case_entry={},
                  out_dir=str(tmp_path), spinup_min=20, stepra=17, dt=36.0, steps=36)
    return ctx, handle, cancelled


def test_finish_rebuilt_points_exempt_handle_is_not_cancelled_at_the_budget(tmp_path):
    ctx, handle, cancelled = _prep_ctx(tmp_path)
    assert wt.finish_rebuilt_points(ctx, dict(handle, exempt_budget=True)) == []
    assert not cancelled and not ctx.cancelled_jobs


def test_finish_rebuilt_points_inside_the_budget_is_still_cancelled(tmp_path):
    ctx, handle, cancelled = _prep_ctx(tmp_path)            # pstar / pipeline: prep is budgeted
    with pytest.raises(RuntimeError, match="not COMPLETED"):
        wt.finish_rebuilt_points(ctx, handle)
    assert cancelled == [["5"]]


# ---------------------------------------------------------------------------
# 8: the layout table's WINNER is the report's BEST
# ---------------------------------------------------------------------------
def _res(cfg, vals):
    med = sorted(vals)[len(vals) // 2]
    return dict(cid=cfg["id"], config=cfg, s_values=list(vals), median=med, lo=min(vals),
                hi=max(vals), n=len(vals))


def _outcome(all_results, confirm_results):
    return wt.node_search.FinalistOutcome(report={}, best_median=0.0, ctx=None, cores=72,
                                          all_results=all_results, anchor=None,
                                          confirm_results=confirm_results)


def test_layout_winner_is_not_an_unconfirmed_one_rep_row():
    a, b, c = _cfg("ANCHOR"), _cfg("D0001", nx=9, ny=8), _cfg("D0002", nx=6, ny=12)
    outcomes = {4: _outcome({"ANCHOR": _res(a, [1.05]), "D0001": _res(b, [0.97]),
                             "D0002": _res(c, [0.90])},                 # D0002: one lucky rep
                            {"ANCHOR": _res(a, [1.00, 1.00]), "D0001": _res(b, [0.99, 0.99])})}
    sw = dict(results={}, anchors={})
    ctx = _ctx()
    tab = wt.build_layout_table(sw, {}, outcomes, ctx)
    assert tab["winner"]["cid"] == "D0001" and tab["runner_up"]["cid"] == "ANCHOR"
    assert [r["cid"] for r in tab["rows"]] == ["D0002", "D0001", "ANCHOR"]   # table: all, by median
    # the report's BEST (the ANCHOR wins a within-noise tie) is the table's winner
    best = (4, 72, wt.sweep._identity(a))
    tab = wt.build_layout_table(sw, {}, outcomes, ctx, best=best)
    assert tab["winner"]["cid"] == "ANCHOR" and tab["runner_up"]["cid"] == "D0001"
    assert tab["comparison"]["gap"] == pytest.approx(-0.01)
    assert tab["comparison"]["beyond_noise"] is False


def test_node_search_report_winner_and_best_name_the_same_layout(monkeypatch, tmp_path, capsys):
    report, _env = _pipe_run(monkeypatch, tmp_path, full_cap=None, grid=MATTHEW_LIKE,
                             physics_advice=False)
    best, w = report["best"], report["node_selection"]["winner"]
    assert (w["nodes"], w["cid"]) == (report["nodes"], best["id"])
    assert (w["ranks"], w["omp"], w["nx"], w["ny"], w["numtiles"]) == (
        best["ranks"], best["omp"], best["nproc_x"], best["nproc_y"], best["numtiles"])


def test_first_node_sweep_budget_check_does_not_use_the_10s_time_limit_estimate(monkeypatch):
    from wtune.tune import node_search
    monkeypatch.setattr(node_search.predict, "parse_namelist", lambda path: {})
    monkeypatch.setattr(node_search, "_fallback_s_per_step", lambda nl, ranks, omp: 0.1)
    ctx = _ctx()
    anchor = _cfg("ANCHOR")
    s, src = node_search._cold_budget_s_per_step(ctx, anchor, 10.0, "fallback_10s")
    assert s == pytest.approx(0.3) and "cell-step model" in src
    assert node_search._cold_budget_s_per_step(ctx, anchor, 0.2, "ledger") == (0.2, "ledger")
    monkeypatch.setattr(node_search, "_fallback_s_per_step", lambda *a: 1 / 0)
    assert node_search._cold_budget_s_per_step(ctx, anchor, 10.0, "fallback_10s")[0] == 10.0


def _row(cid, job, s, notes="node_dirty=none foreign_load=0"):
    return {"run_id": "conus25_%s_r1_%s" % (cid, job), "s_per_step_median": str(s), "notes": notes}


def test_aggregate_marks_measurements_taken_on_a_dirty_node():
    cfgs = {"D0001": _cfg("D0001"), "D0002": _cfg("D0002")}
    out = wt.aggregate_by_cid([_row("D0001", 1, 0.2, "node_dirty=work3:cpu64:0.98"),
                               _row("D0002", 1, 0.3)], cfgs)
    assert out["D0001"]["dirty"] == ["node_dirty=work3:cpu64:0.98"]
    assert "dirty" not in out["D0002"]


@pytest.mark.parametrize("second_notes,expect_dirty,expect_median", [
    ("node_dirty=none foreign_load=0", False, 0.5),       # clean re-measure replaces the number
    ("foreign_load=1 foreign=work3:1:root:code:91", True, 0.2),   # still dirty: number kept
])
def test_a_dirty_fastest_layout_is_measured_once_more(second_notes, expect_dirty, expect_median,
                                                      capsys):
    from wtune.tune import rounds
    cfgs = {"D0001": _cfg("D0001"), "D0002": _cfg("D0002")}
    results = wt.aggregate_by_cid([_row("D0001", 1, 0.2, "node_dirty=work3:cpu64:0.98"),
                                   _row("D0002", 1, 0.3)], cfgs)
    submitted = []
    ctx = _ctx(sbatch_runner=lambda cmd: submitted.append(cmd) or "Submitted batch job 900",
               ledger_reader=lambda case, ids: [_row("D0001", 900, 0.5, second_notes)],
               budget_s=36000.0)
    rounds._remeasure_dirty_best(ctx, results, 0.2)
    assert len(submitted) == 1
    assert bool(results["D0001"].get("dirty")) is expect_dirty
    assert results["D0001"]["median"] == pytest.approx(expect_median)
    assert "dirty node" in capsys.readouterr().out


def test_a_clean_fastest_layout_is_not_measured_again():
    from wtune.tune import rounds
    cfgs = {"D0001": _cfg("D0001"), "D0002": _cfg("D0002")}
    results = wt.aggregate_by_cid([_row("D0001", 1, 0.2),
                                   _row("D0002", 1, 0.3, "node_dirty=work3:cpu64:0.98")], cfgs)
    submitted = []
    ctx = _ctx(sbatch_runner=lambda cmd: submitted.append(cmd) or "Submitted batch job 900")
    rounds._remeasure_dirty_best(ctx, results, 0.2)
    assert submitted == []
