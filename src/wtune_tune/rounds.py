"""The decomposition search at one node count: round 0 (anchor + diverse sample over parallel
slots, or the full layout space), adaptive proposer rounds and the confirm job."""
from __future__ import annotations

import glob
import math
import os
from typing import Any, Callable

from ._deps import sweep
from .budget import (_job_time_est, _ledger_case, _ranking_wrap, _stage1_active, _stage1_anchor,
                     _stage1_cutoff, _stage1_may_start, describe_cost_model,
                     describe_stage2_reserve, estimate_run_time, rad_factor, stage1_reserve_s,
                     update_s_est, wrap_with_anchors)
from .candidates import (FULL_SPACE_CONFIGS_PER_JOB, _apply_search_filter, aggregate_by_cid,
                         rank_predicted_entries, sample_more_diverse_configs)
from .common import (ANCHOR_END_ID, _fmt_hms, _parse_slurm_elapsed, chunk_round_robin,
                     compute_slots, would_exceed_budget)
from .context import Ctx, cores_used
from .estimates import EST_ROUND0_CONFIGS
from .jobs import _record_stage1_gates, submit_configs_job
from .prediction import predict_candidates
from . import budget as _budget
from . import candidates as _candidates
from . import jobs as _jobs

ROUND0_BUDGET_FRACTION = 0.35          # share of the budget round 0's parallel slots may use
ADAPTIVE_STOP_FRACTION = 0.75          # spec: stop proposing when projected finish > 75% of budget
ADAPTIVE_MAX_ROUNDS = 6
KAPPA_ESCALATION_FACTOR = 2.0          # retry a round that stopped early with more exploration
KAPPA_MAX_ESCALATIONS = 2              # escalations before falling back to decomp_space diversity

Wave = tuple[list, dict, list, dict, dict]   # (job_ids, configs_by_cid, rows, job_configs, states)


def select_confirm_entries(ctx: Ctx, nodes: int, all_results: dict,
                           anchor_cid: str | None) -> list[dict]:
    """The confirm candidates: the top 3 by predicted whole-run time (rank_predicted_entries)
    plus the anchor, deduped, as aggregate_by_cid entries with their pred_* fields."""
    predicted = predict_candidates(ctx, nodes, all_results)
    ranked = rank_predicted_entries(predicted, anchor_cid)
    top = list(ranked[:3])
    cids = {r["cid"] for r in top}
    if anchor_cid in predicted and anchor_cid not in cids:
        top.append(predicted[anchor_cid])
    seen, entries = set(), []
    for r in top:
        if r["cid"] in seen or r["config"] is None:
            continue
        seen.add(r["cid"])
        entries.append(r)
    return entries


def select_confirm_configs(ctx: Ctx, nodes: int, all_results: dict,
                           anchor_cid: str | None) -> list[dict]:
    """The configs of select_confirm_entries."""
    return [e["config"] for e in select_confirm_entries(ctx, nodes, all_results, anchor_cid)]


def _submit_round0_wave(ctx: Ctx, configs: list[dict], slots: int, s_est: float,
                        name_prefix: str) -> Wave:
    """Deal `configs` over `slots` parallel jobs, submit each whose RAW expected time fits the
    budget, wait for all. Returns (job_ids, configs_by_cid, rows, job_configs, states) -- the
    last two feed measure_startup_s. Predicted-mode stage 1 uses _submit_ranking_wave."""
    if _stage1_active(ctx):
        return _submit_ranking_wave(ctx, configs, slots, s_est, name_prefix)
    chunks = chunk_round_robin(configs, slots)
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, chunk in enumerate(chunks):
        chunk = _ranking_wrap(ctx, chunk)     # predicted mode: ANCHOR ... ANCHOREND in EVERY chunk
        time_est = _job_time_est(ctx, len(chunk), 1, s_est)
        if would_exceed_budget(ctx.elapsed(), time_est["total_run_s"], ctx.budget_s):
            print("%s slot %d: SKIP -- estimated end would exceed budget" % (name_prefix, i))
            continue
        ok, why = _stage1_may_start(ctx, time_est["total_run_s"])
        if not ok:
            print("%s slot %d: SKIP -- %s" % (name_prefix, i, why))
            continue
        job_id, _path, _te = submit_configs_job(ctx, chunk, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=time_est)
        job_ids.append(job_id)
        job_configs[job_id] = chunk
        for c in chunk:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs)" % (name_prefix, i, job_id, len(chunk)))

    states = _jobs.wait_for_jobs(ctx, job_ids, name_prefix, pending_cutoff=_stage1_cutoff(ctx))
    rows = ctx.ledger_reader(_ledger_case(ctx), job_ids)
    _record_stage1_gates(ctx, job_ids, rows, name_prefix)
    return job_ids, configs_by_cid, rows, job_configs, states


def fit_stage1_job(ctx: Ctx, candidates: list[dict],
                   s_est: float) -> tuple[list[dict] | None, list[dict], dict | None]:
    """Predicted mode: the longest prefix of `candidates` (sampling priority order) whose job --
    spin-up + ANCHOR + prefix + ANCHOREND -- still fits elapsed + job + stage-2 reserve <= budget.
    Returns (wrapped, dropped, time_est), or (None, candidates, estimate of the minimum)."""
    cands = [c for c in candidates if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    est = None
    for n in range(len(cands), 0, -1):
        wrapped = wrap_with_anchors(cands[:n], _stage1_anchor(ctx))
        est = _job_time_est(ctx, len(wrapped), 1, s_est)
        ok, _why = _stage1_may_start(ctx, est["total_run_s"])
        if ok:
            return wrapped, cands[n:], est
    return None, cands, est


def _submit_ranking_wave(ctx: Ctx, configs: list[dict], slots: int, s_est: float,
                         name_prefix: str) -> Wave:
    """Predicted-mode stage-1 wave: candidates dealt over min(slots, #candidates) slots, each
    with its own ANCHOR/ANCHOREND and SHRUNK to fit (fit_stage1_job; trimmed ones recorded); the
    plan is logged before anything is submitted. Same return as _submit_round0_wave."""
    cands = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    n_slots = max(1, min(int(slots), len(cands)))
    plan = []
    for i, chunk in enumerate(chunk_round_robin(cands, n_slots)):
        wrapped, dropped, est = fit_stage1_job(ctx, chunk, s_est)
        plan.append((i, wrapped, dropped, est))
    print("%s plan: %d slot(s) (max %d concurrent = floor(max_nodes/nodes)), candidates per slot "
          "%s, expected %s per job; elapsed %s, %s"
          % (name_prefix, sum(1 for p in plan if p[1]), slots,
             [len(p[1]) - 2 if p[1] else 0 for p in plan],
             ", ".join(_fmt_hms(p[3]["total_run_s"]) if p[1] else "skip" for p in plan),
             _fmt_hms(ctx.elapsed()), describe_stage2_reserve(ctx)))
    if plan and plan[0][3] and plan[0][3].get("cost_model"):
        print("%s cost model: %s" % (name_prefix, describe_cost_model(plan[0][3]["cost_model"])))
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, wrapped, dropped, est in plan:
        for c in dropped:
            ctx.stage1_trimmed.append(dict(cid=c["id"], stage=name_prefix,
                                           reason="trimmed to fit the budget + stage-2 reserve"))
        if wrapped is None:
            print("%s slot %d: SKIP -- even ANCHOR + 1 candidate + ANCHOREND (%s) does not fit: %s"
                  % (name_prefix, i, _fmt_hms(est["total_run_s"]) if est else "-",
                     _stage1_may_start(ctx, est["total_run_s"])[1] if est else "no candidate"))
            continue
        if dropped:
            print("%s slot %d: trimmed %d candidate(s) to fit: %s"
                  % (name_prefix, i, len(dropped), ", ".join(c["id"] for c in dropped)))
        job_id, _path, _te = submit_configs_job(ctx, wrapped, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=est)
        job_ids.append(job_id)
        job_configs[job_id] = wrapped
        for c in wrapped:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs, expected %s)"
              % (name_prefix, i, job_id, len(wrapped), _fmt_hms(est["total_run_s"])))
    states = _jobs.wait_for_jobs(ctx, job_ids, name_prefix, pending_cutoff=_stage1_cutoff(ctx))
    rows = ctx.ledger_reader(_ledger_case(ctx), job_ids)
    _record_stage1_gates(ctx, job_ids, rows, name_prefix)
    return job_ids, configs_by_cid, rows, job_configs, states


def measure_ranking_overhead(ctx: Ctx, job_configs: dict, states: dict,
                             wave_results: dict) -> tuple[float | None, int]:
    """Predicted-mode measure_startup_s: per-run overhead = (elapsed - compute) / (runs + 1), the
    compute including the ANCHOR spin-up and radiation factors; it replaces startup_s + restart
    read together. (None, 0) without a usable COMPLETED job."""
    r = ctx.ranking
    spin_steps = r["spinup_minutes"] * 60.0 / float(r["dt"])
    f_run, f_spin = rad_factor(r["steps"], r.get("stepra")), rad_factor(spin_steps, r.get("stepra"))
    tot_e = tot_c = 0.0
    n = 0
    for jid, configs in job_configs.items():
        state, elapsed_str = states.get(jid, ("UNKNOWN", ""))
        el = _parse_slurm_elapsed(elapsed_str)
        a = wave_results.get("ANCHOR")
        if state != "COMPLETED" or el is None or not a:
            continue
        comp, cnt = spin_steps * a["median"] * f_spin, 0
        for c in configs:
            w = wave_results.get(c["id"])
            if w:
                comp += r["steps"] * w["median"] * f_run
                cnt += 1
        if cnt:
            tot_e += el
            tot_c += comp
            n += cnt + 1
    if not n:
        return None, 0
    return max(0.0, (tot_e - tot_c) / n), n


def _configs_for_budget(budget_s: float, per_run_s: float, slots: int) -> int:
    """Configs that fit `budget_s` at `per_run_s` each; with no time cap EST_ROUND0_CONFIGS
    spread over the slots."""
    if not math.isfinite(budget_s):
        return max(1, int(math.ceil(EST_ROUND0_CONFIGS / float(max(1, slots)))))
    return max(1, int(budget_s // per_run_s))


def _set_measured_startup(ctx: Ctx, measured: float, n_runs: int) -> None:
    ctx.startup_s = measured
    ctx.startup_source = "measured from round0 (n=%d completed runs)" % n_runs
    ctx.startup_runs = n_runs


def _run_full_space_round0(ctx: Ctx, s_est: float, slots: int) -> tuple[list, dict, list]:
    """round0 of the full layout search: every enumerated layout once (REPS=1) in waves of
    `slots` jobs of FULL_SPACE_CONFIGS_PER_JOB configs; what was not measured goes to
    ctx.layout_notes["untried"] with its reason. Returns (job_ids, configs_by_cid, rows)."""
    configs, info = _candidates.enumerate_full_space(ctx)
    notes = ctx.layout_notes if ctx.layout_notes is not None else dict(untried=[], enumerated={})
    notes["enumerated"]["N=%d c=%d" % (ctx.nodes, cores_used(ctx))] = info
    print("round0 (full layout search): N=%d cores/node=%d: %d enumerated, %d unique, %d legal "
          "(%d excluded) -> %d waves of %d job(s)"
          % (ctx.nodes, cores_used(ctx), info["raw"], info["unique"], info["legal"],
             info["excluded"], math.ceil(len(configs) / float(slots * FULL_SPACE_CONFIGS_PER_JOB)),
             slots))
    if not configs:
        return [], {}, []
    job_ids, configs_by_cid, rows = [], {}, []
    wave_size = slots * FULL_SPACE_CONFIGS_PER_JOB
    for w, start in enumerate(range(0, len(configs), wave_size)):
        chunk = configs[start:start + wave_size]
        ids, by_cid, wrows, job_configs, states = _submit_round0_wave(
            ctx, chunk, slots, s_est, "round0-w%d" % w)
        job_ids += ids
        configs_by_cid.update(by_cid)
        rows += wrows
        if w == 0 and ids:               # the first wave measures this allocation's startup
            res = aggregate_by_cid(wrows, by_cid)
            measured, n_runs = _budget.measure_startup_s(job_configs, states, res, ctx.steps)
            if measured is not None:
                print("startup_s: measured %.1fs/run from %d run(s) (was %.1fs, %s)"
                      % (measured, n_runs, ctx.startup_s, ctx.startup_source))
                _set_measured_startup(ctx, measured, n_runs)
            s_est = update_s_est(res, s_est)
    measured_cids = {r["cid"] for r in aggregate_by_cid(rows, configs_by_cid).values()}
    for c in configs:
        if c["id"] not in configs_by_cid:
            reason = "not submitted: the job would exceed the time cap"
        elif c["id"] not in measured_cids:
            reason = "no measurement (the run failed or its job was cancelled)"
        else:
            continue
        notes["untried"].append(dict(nodes=ctx.nodes, cores_per_node=cores_used(ctx), id=c["id"],
                                     ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"],
                                     numtiles=c["numtiles"], reason=reason))
    return job_ids, configs_by_cid, rows


def _round0_size(ctx: Ctx, s_est: float, slots: int) -> tuple[int, int]:
    """(configs per job, configs to build incl. the ANCHOR): round 0's share of the budget at
    `s_est`; in predicted-mode stage 1 every job carries its own ANCHOR/ANCHOREND (+1), and pstar
    sizes off what one job holds after its spin-up and two anchor runs."""
    per_run_s = ctx.startup_s + ctx.steps * s_est
    round0_budget_s = min(ctx.budget_s - ctx.elapsed() - stage1_reserve_s(ctx),
                          ctx.budget_s * ROUND0_BUDGET_FRACTION)
    configs_per_job = _configs_for_budget(round0_budget_s, per_run_s, slots)
    total_k = slots * configs_per_job
    if _stage1_active(ctx) and not ctx.ranking:
        total_k = slots * configs_per_job + 1
    elif _stage1_active(ctx):
        est1 = _job_time_est(ctx, 1, 1, s_est)
        m = est1.get("cost_model")
        per_run = m["per_run_s"] if m else est1["total_run_s"]
        spin = m["spin_s"] if m else 0.0
        avail = ctx.budget_s - ctx.elapsed() - stage1_reserve_s(ctx) - spin - 2 * per_run
        configs_per_job = max(1, int(avail // per_run))
        total_k = slots * configs_per_job + 1
    return configs_per_job, total_k


def _adopt_round0_startup(ctx: Ctx, job_configs: dict, states: dict, wave0_results: dict) -> None:
    """Refresh ctx.startup_s from wave 0's own Slurm elapsed time (the ranking overhead in pstar),
    logged either way so the run says which startup is in effect."""
    if ctx.ranking:
        measured_startup, n_runs = measure_ranking_overhead(ctx, job_configs, states,
                                                            wave0_results)
        if measured_startup is not None:
            ctx.restart_read_s = 0.0      # the measured overhead already includes it
    else:
        measured_startup, n_runs = _budget.measure_startup_s(job_configs, states, wave0_results,
                                                             ctx.steps)
    if measured_startup is not None:
        print("startup_s: measured %.1fs/run from %d completed run(s) in round0 (was %.1fs, %s)"
              % (measured_startup, n_runs, ctx.startup_s, ctx.startup_source))
        _set_measured_startup(ctx, measured_startup, n_runs)
    else:
        print("startup_s: no COMPLETED round0 run to measure from; keeping %.1fs (%s)"
              % (ctx.startup_s, ctx.startup_source))


def _round0_topup(ctx: Ctx, configs_by_cid: dict, s_est: float, refreshed_s_est: float,
                  slots: int) -> tuple[list, dict, list] | None:
    """One more wave of not-yet-measured candidates sized at the measured s/step: the pre-round0
    estimate can be the 10 s/step no-ledger fallback, 333x too slow (sc26p2 2026-09-22).
    Returns (job_ids, configs_by_cid, rows), or None when there is nothing to add."""
    topup_per_run_s = ctx.startup_s + ctx.steps * refreshed_s_est
    topup_budget_s = min(ctx.budget_s - ctx.elapsed(), ctx.budget_s * ROUND0_BUDGET_FRACTION)
    topup_configs_per_job = _configs_for_budget(topup_budget_s, topup_per_run_s, slots)
    topup_k = slots * topup_configs_per_job
    fresh = sample_more_diverse_configs(ctx, list(configs_by_cid.values()), topup_k,
                                        ctx.seed + 1)
    print("round0 top-up: s_est=%.3fs/step (measured, was %.3fs/step) startup_s=%.1fs "
          "configs_per_job=%d candidates=%d"
          % (refreshed_s_est, s_est, ctx.startup_s, topup_configs_per_job, len(fresh)))
    if not fresh:
        return None
    topup_ids, topup_by_cid, topup_rows, _jc, _st = _submit_round0_wave(
        ctx, fresh, slots, refreshed_s_est, "round0-topup")
    return topup_ids, topup_by_cid, topup_rows


def run_round0(ctx: Ctx) -> tuple[list, dict, list]:
    """Round 0 at ctx.nodes: ANCHOR + a diverse decomp_space sample over floor(max_nodes/nodes)
    parallel jobs (or the full space, ctx.full_space), then the measured startup and a top-up
    wave when the measured s/step undercuts the estimate. Returns (job_ids, configs_by_cid,
    rows)."""
    s_est, s_source, n_rows = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                          ctx.hardware_id)
    if ctx.full_space:
        return _run_full_space_round0(ctx, s_est, compute_slots(ctx.max_nodes, ctx.nodes))
    if _stage1_active(ctx) and ctx.s_prior and n_rows == 0:
        # no ledger history: the cell-step model beats the 10 s/step fallback, which would also
        # inflate the stage-2 reserve until nothing in stage 1 fits
        s_est, s_source = ctx.s_prior, "cell-step model"
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    configs_per_job, total_k = _round0_size(ctx, s_est, slots)

    built = sweep.build_configs(ctx.case, ctx.case_json, ctx.nodes, max(0, total_k - 1), ctx.seed,
                                 ctx.hardware_id, cores_used(ctx))
    if ctx.anchor_override:
        # node-search finalist: the sweep's hybrid ANCHOR replaces build_configs' omp=1 one
        ov = ctx.anchor_override
        rest = [c for c in built["configs"]
                if c.get("id") != "ANCHOR" and sweep._identity(c) != sweep._identity(ov)]
        built = dict(built, configs=[ov] + rest, skip_reason=None)
    print("round0: s_est=%.3fs/step[%s,n=%d] startup_s=%.1fs[%s] slots=%d configs_per_job=%d "
          "total=%d (%s)"
          % (s_est, s_source, n_rows, ctx.startup_s, ctx.startup_source, slots, configs_per_job,
             len(built["configs"]), built["skip_reason"] or "ok"))
    if built["skip_reason"]:
        return [], {}, []

    configs0 = _apply_search_filter(ctx, built["configs"])
    if not configs0:
        print("round0: SKIP -- every candidate (including the anchor) was excluded by --config "
              "search restrictions/exclude rules")
        return [], {}, []

    job_ids, configs_by_cid, rows, job_configs, states = _submit_round0_wave(
        ctx, configs0, slots, s_est, "round0")
    if job_ids:
        wave0_results = aggregate_by_cid(rows, configs_by_cid)
        _adopt_round0_startup(ctx, job_configs, states, wave0_results)
        refreshed_s_est = update_s_est(wave0_results, s_est)
        if refreshed_s_est < s_est and ctx.elapsed() < ctx.budget_s:
            topup = _round0_topup(ctx, configs_by_cid, s_est, refreshed_s_est, slots)
            if topup:
                topup_ids, topup_by_cid, topup_rows = topup
                job_ids = job_ids + topup_ids
                configs_by_cid.update(topup_by_cid)
                rows = rows + topup_rows
    return job_ids, configs_by_cid, rows


def _default_proposer(ctx: Ctx) -> Callable:
    """proposer(state, batch_size, kappa) over predictor/decomp_search.py, imported here so the
    tuner itself never needs numpy / scikit-learn / catboost. Other cases' ledgers are the
    cross-case prior (this case's own rows are already in `state`)."""
    import decomp_search   # predictor/decomp_search.py

    case_fields = decomp_search.case_row_fields(ctx.case_json)
    prior_rows = []
    for path in sorted(glob.glob(os.path.join(ctx.ledger_dir, "ledger_*.csv"))):
        if os.path.basename(path).startswith("ledger_%s_" % ctx.case):
            continue    # this case's own ledgers are already in `state` (case_measured_rows)
        if ctx.ranking and os.path.basename(path).startswith("ledger_%s_" % ctx.ranking["case_id"]):
            continue    # predicted mode: this run's own rebuilt-point ledgers, same reason
        try:
            prior_rows.extend(decomp_search.benchmark.load_ledger(path))
        except Exception:
            continue    # best-effort cross-case prior; a malformed sibling ledger is not fatal

    def proposer(state, batch_size, kappa=None):
        return decomp_search.propose_for_case(
            ctx.case, ctx.case_json, case_fields, prior_rows, state, batch_size,
            total_cores=ctx.nodes * cores_used(ctx), nodes=ctx.nodes,
            hardware_id=ctx.hardware_id, kappa=ctx.kappa if kappa is None else kappa)

    return proposer


def _round_batch_size(ctx: Ctx, round_idx: int, elapsed: float, s_est: float) -> int | None:
    """Configs adaptive round `round_idx` may propose, or None (printed) when the rounds stop:
    t0 at 75% of the budget, predicted-mode stage 1 when no stage-1 time is left."""
    if not _stage1_active(ctx) and elapsed >= ADAPTIVE_STOP_FRACTION * ctx.budget_s:
        print("adaptive rounds: stopping before round %d -- elapsed already >= %.0f%% of budget"
              % (round_idx, ADAPTIVE_STOP_FRACTION * 100))
        return None
    remaining_s = ctx.budget_s - elapsed - stage1_reserve_s(ctx)
    if _stage1_active(ctx) and remaining_s <= 0:
        print("adaptive rounds: stopping before round %d -- no stage-1 time left (elapsed %s, "
              "stage-1 reserve %s, budget %s)"
              % (round_idx, _fmt_hms(elapsed), _fmt_hms(stage1_reserve_s(ctx)),
                 _fmt_hms(ctx.budget_s)))
        return None
    per_run_s = ctx.startup_s + ctx.steps * s_est
    return _configs_for_budget(min(remaining_s, ctx.budget_s * ROUND0_BUDGET_FRACTION), per_run_s, 1)


def _propose_round(ctx: Ctx, proposer: Callable, state: dict, batch_size: int, round_idx: int,
                   all_configs_by_cid: dict) -> tuple[list[dict] | None, float]:
    """(configs, kappa) of one adaptive round: the proposer, retried with escalating kappa, its
    output filtered (it knows no --config restriction); when that is empty, unmeasured
    decomp_space-diverse candidates (sc26p2 2026-09-22 stopped at round 1 with 90% unused).
    (None, kappa), printed, when even that pool is exhausted."""
    result, configs_out, kappa = None, [], ctx.kappa
    for attempt in range(KAPPA_MAX_ESCALATIONS + 1):
        result, configs_out = proposer(state, batch_size, kappa=kappa)
        if not result.get("stopped") and configs_out:
            break
        if attempt < KAPPA_MAX_ESCALATIONS:
            next_kappa = kappa * KAPPA_ESCALATION_FACTOR
            print("adaptive round %d: proposer stopped at kappa=%.3g (%s); retrying with more "
                  "exploration pressure (kappa=%.3g)"
                  % (round_idx, kappa, result.get("reason", ""), next_kappa))
            kappa = next_kappa
    configs_out = _apply_search_filter(ctx, configs_out)
    if result.get("stopped") or not configs_out:
        fresh = sample_more_diverse_configs(ctx, list(all_configs_by_cid.values()), batch_size,
                                             ctx.seed + round_idx)
        if not fresh:
            print("adaptive rounds: stopped at round %d -- proposer exhausted even after "
                  "kappa escalation (%s) and the decomp_space pool has no unmeasured "
                  "candidate left" % (round_idx, result.get("reason", "no candidates proposed")))
            return None, kappa
        print("adaptive round %d: proposer exhausted (%s); falling back to %d decomp_space-"
              "diverse candidate(s) not yet measured"
              % (round_idx, result.get("reason", ""), len(fresh)))
        configs_out = fresh
    return configs_out, kappa


def _plan_round_job(ctx: Ctx, configs_out: list[dict], s_est: float, round_idx: int,
                    elapsed: float) -> tuple[list[dict], dict] | None:
    """(configs, time estimate) of the round's job -- shrunk to fit in predicted-mode stage 1 --
    or None (printed) when it may not start (t0: projected finish > 75% of the budget)."""
    time_est = _job_time_est(ctx, len(_ranking_wrap(ctx, configs_out)), 1, s_est)
    if _stage1_active(ctx):
        wrapped, dropped, fit_est = fit_stage1_job(ctx, configs_out, s_est)
        if wrapped is None:
            print("adaptive rounds: stopping before round %d -- even ANCHOR + 1 candidate + "
                  "ANCHOREND does not fit before the stage-2 reserve" % round_idx)
            return None
        for c in dropped:
            ctx.stage1_trimmed.append(dict(cid=c["id"], stage="round%d" % round_idx,
                                           reason="trimmed to fit the budget + stage-2 "
                                                  "reserve"))
        configs_out = wrapped[1:-1]
        time_est = fit_est
        print("adaptive round %d plan: %d candidate(s), expected %s%s"
              % (round_idx, len(configs_out), _fmt_hms(time_est["total_run_s"]),
                 (" (trimmed %d)" % len(dropped)) if dropped else ""))
    projected_end = elapsed + time_est["total_run_s"]        # RAW/EXPECTED -- never job_hours
    ok, why = _stage1_may_start(ctx, time_est["total_run_s"])
    if not ok:
        print("adaptive rounds: stopping before round %d -- %s" % (round_idx, why))
        return None
    if not _stage1_active(ctx) and projected_end > ADAPTIVE_STOP_FRACTION * ctx.budget_s:
        print("adaptive rounds: stopping before round %d -- projected finish %s > %.0f%% "
              "of budget" % (round_idx, _fmt_hms(projected_end), ADAPTIVE_STOP_FRACTION * 100))
        return None
    return configs_out, time_est


def run_adaptive_rounds(ctx: Ctx, all_configs_by_cid: dict, all_results: dict,
                        s_est: float) -> None:
    """Up to ADAPTIVE_MAX_ROUNDS rounds of one job each, proposed by predictor/decomp_search.py
    (ctx.proposer, else lazily imported) from everything measured so far; updates
    `all_configs_by_cid` / `all_results` in place and stops at the budget lines."""
    proposer = ctx.proposer
    if proposer is None:
        try:
            proposer = _default_proposer(ctx)
        except ImportError as e:
            print("adaptive rounds skipped: predictor/decomp_search.py not importable (%s)" % e)
            return

    state: dict[str, Any] = dict(configs={}, measured={}, rounds=[])
    for cid, r in all_results.items():
        if r["config"] is None:
            continue
        state["configs"][cid] = r["config"]
        state["measured"][cid] = [dict(run_id="%s_%s_r%d_seed" % (ctx.case, cid, i),
                                        job_id=None, rep_index=i, s_per_step_median=s)
                                   for i, s in enumerate(r["s_values"])]

    for round_idx in range(1, ADAPTIVE_MAX_ROUNDS + 1):
        elapsed = ctx.elapsed()
        batch_size = _round_batch_size(ctx, round_idx, elapsed, s_est)
        if batch_size is None:
            return
        # state["rounds"] grows once per round, not per kappa retry (propose_core's round index)
        configs_out, kappa = _propose_round(ctx, proposer, state, batch_size, round_idx,
                                            all_configs_by_cid)
        if configs_out is None:
            return
        state.setdefault("rounds", []).append(dict(round=round_idx, batch_size=batch_size,
                                                     kappa=kappa))
        planned = _plan_round_job(ctx, configs_out, s_est, round_idx, elapsed)
        if planned is None:
            return
        configs_out, time_est = planned

        job_id, _path, _te = submit_configs_job(ctx, configs_out, reps=1,
                                                  name_suffix="round%d" % round_idx, s_est=s_est,
                                                  time_est=time_est)
        print("adaptive round %d: submitted job %s (%d configs)"
              % (round_idx, job_id, len(configs_out)))
        for c in configs_out:
            all_configs_by_cid[c["id"]] = c
            state["configs"][c["id"]] = c

        _jobs.wait_for_jobs(ctx, [job_id], "round%d" % round_idx, pending_cutoff=_stage1_cutoff(ctx))
        rows = ctx.ledger_reader(_ledger_case(ctx), [job_id])
        _record_stage1_gates(ctx, [job_id], rows, "round%d" % round_idx)
        new_results = aggregate_by_cid(rows, all_configs_by_cid)
        new_results.pop(ANCHOR_END_ID, None)       # drift check only, never a candidate
        for cid, r in new_results.items():
            all_results[cid] = r
            state["measured"][cid] = [dict(run_id="%s_%s_r%d_%s" % (ctx.case, cid, i, job_id),
                                            job_id=job_id, rep_index=i, s_per_step_median=s)
                                       for i, s in enumerate(r["s_values"])]
        s_est = update_s_est(all_results, s_est)   # keep sizing off this run's freshest numbers


def _fit_confirm(ctx: Ctx, entries: list[dict]) -> tuple[list[dict], list[dict], dict | None]:
    """(kept, dropped, time_est): the cheapest candidates first, each kept while the RAW expected
    confirm job still fits -- one slow outlier must not veto the fast ones (sc26p2), and a
    padded estimate must not either (conus25 2026-09-23)."""
    entries_sorted = sorted(entries, key=lambda e: e["median"])
    kept: list[dict] = []
    dropped: list[dict] = []
    time_est = None
    for e in entries_sorted:
        trial = kept + [e]
        trial_est = estimate_run_time([(x["config"], x["median"]) for x in trial],
                                       ctx.confirm_reps, ctx.steps, startup_s=ctx.startup_s)
        if would_exceed_budget(ctx.elapsed(), trial_est["total_run_s"], ctx.budget_s):
            dropped.append(e)
            continue
        kept.append(e)
        time_est = trial_est
    return kept, dropped, time_est


def run_confirm(ctx: Ctx, all_configs_by_cid: dict, all_results: dict, anchor_cid: str | None,
                s_est: float) -> tuple[str | None, dict | None]:
    """Re-measure the confirm candidates (top 3 + anchor) in one job of ctx.confirm_reps
    replicates, keeping only those that fit the budget. Returns (job id, {cid: result}) or
    (None, None) when nothing fits."""
    entries = select_confirm_entries(ctx, ctx.nodes, all_results, anchor_cid)
    if not entries:
        print("confirm: SKIP -- no measured configs to confirm")
        return None, None

    # every candidate's numbers are logged up front, so a skipped confirm is explainable
    for e in entries:
        per_run_s = ctx.startup_s + ctx.steps * e["median"]
        print("confirm candidate %-8s median=%.4fs/step (n=%d) -> per_run~%.1fs (startup_s=%.1fs) "
              "x REPS=%d" % (e["cid"], e["median"], e["n"], per_run_s, ctx.startup_s,
                              ctx.confirm_reps))

    kept, dropped, time_est = _fit_confirm(ctx, entries)
    if not kept:
        entries_sorted = sorted(entries, key=lambda e: e["median"])
        solo_est = estimate_run_time(
            [(entries_sorted[0]["config"], entries_sorted[0]["median"])], ctx.confirm_reps,
            ctx.steps, startup_s=ctx.startup_s)
        print("confirm: SKIP -- even the cheapest candidate alone (%s, %.4fs/step) is estimated "
              "at %s (expected, REPS=%d), which would exceed the remaining budget "
              "(elapsed=%s / budget=%s)"
              % (entries_sorted[0]["cid"], entries_sorted[0]["median"],
                 _fmt_hms(solo_est["total_run_s"]), ctx.confirm_reps, _fmt_hms(ctx.elapsed()),
                 _fmt_hms(ctx.budget_s)))
        return None, None

    if dropped:
        print("confirm: dropping %d candidate(s) that would not fit the remaining budget: %s"
              % (len(dropped),
                 ", ".join("%s(%.4fs/step)" % (e["cid"], e["median"]) for e in dropped)))

    configs = [e["config"] for e in kept]
    print("confirm: estimated %s for %d config(s) (REPS=%d, expected -- not the padded --time "
          "request) -- elapsed=%s / budget=%s"
          % (_fmt_hms(time_est["total_run_s"]), len(configs), ctx.confirm_reps,
             _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))

    job_id, _path, _te = submit_configs_job(ctx, configs, reps=ctx.confirm_reps, name_suffix="confirm",
                                              s_est=s_est, time_est=time_est)
    print("confirm: submitted job %s (%d configs, REPS=%d)" % (job_id, len(configs), ctx.confirm_reps))
    _jobs.wait_for_jobs(ctx, [job_id], "confirm")
    rows = ctx.ledger_reader(ctx.case, [job_id])
    confirm_configs_by_cid = {c["id"]: c for c in configs}
    confirm_results = aggregate_by_cid(rows, confirm_configs_by_cid)
    return job_id, confirm_results


def run_tuning_at(ctx: Ctx, s_est: float | None = None, adaptive: bool = True,
                  confirm: bool = True) -> tuple[dict, dict | None, str | None]:
    """round0 -> adaptive rounds -> confirm at ctx.nodes. Returns (all_results,
    confirm_results, anchor_cid)."""
    if s_est is None:
        s_est, _src, _n = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                     ctx.hardware_id)
    job_ids0, configs_by_cid, rows0 = run_round0(ctx)
    all_results = aggregate_by_cid(rows0, configs_by_cid)
    anchor_cid = "ANCHOR" if "ANCHOR" in configs_by_cid else None
    # later stages size off what round0 measured, not the pre-round0 estimate (conus12
    # 2026-09-22: the stale 10 s/step fallback left 13.6 of 30 minutes unused)
    s_est = update_s_est(all_results, s_est)

    if job_ids0 and adaptive and not ctx.full_space:       # the full space leaves nothing to propose
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)

    confirm_job_id, confirm_results = (None, None)
    if all_results and confirm:
        confirm_job_id, confirm_results = run_confirm(ctx, configs_by_cid, all_results, anchor_cid,
                                                        s_est)

    return all_results, confirm_results, anchor_cid
