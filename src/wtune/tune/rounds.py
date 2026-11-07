"""The decomposition search at one node count: round 0 (anchor + diverse sample over parallel
slots, or the full layout space), adaptive proposer rounds and the confirm job."""
from __future__ import annotations

import dataclasses
import glob
import math
import os
from typing import Any, Callable

import wtune.machine

from ._deps import sweep
from .budget import _job_time_est, estimate_run_time, update_s_est
from .candidates import (FULL_SPACE_CONFIGS_PER_JOB, _apply_search_filter, aggregate_by_cid,
                         merge_aggs, rank_predicted_entries, sample_more_diverse_configs)
from .common import (ANCHOR_END_ID, RACE_MAX, _fmt_hms, chunk_round_robin, close_to_best,
                     compute_slots, would_exceed_budget)
from .context import Ctx, QueueCapExceeded, cores_used
from .estimates import EST_ROUND0_CONFIGS
from .jobs import submit_configs_job
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
    """The confirm candidates: the top 3 by predicted whole-run time (rank_predicted_entries),
    the fastest measured median when those miss it (f1-arm conus12 2026-10-14: the prediction
    does not see dist, and cyclic 0.132 s/step went unconfirmed against block 0.187), and the
    anchor, deduped, as aggregate_by_cid entries with their pred_* fields."""
    predicted = predict_candidates(ctx, nodes, all_results)
    ranked = rank_predicted_entries(predicted, anchor_cid)
    top = list(ranked[:3])
    cids = {r["cid"] for r in top}
    measured = [e for e in predicted.values()
                if e.get("median") is not None and e.get("config") is not None]
    if measured:
        fastest = min(measured, key=lambda e: (e["median"], e["cid"]))
        if fastest["cid"] not in cids:
            top.append(fastest)
            cids.add(fastest["cid"])
    if anchor_cid in predicted and anchor_cid not in cids:
        top.append(predicted[anchor_cid])
    seen, entries = set(), []
    for r in top:
        if r["cid"] in seen or r["config"] is None:
            continue
        seen.add(r["cid"])
        entries.append(r)
    return entries


def _submit_round0_wave(ctx: Ctx, configs: list[dict], slots: int, s_est: float,
                        name_prefix: str) -> Wave:
    """Deal `configs` over `slots` parallel jobs, submit each whose RAW expected time fits the
    budget, wait for all. Returns (job_ids, configs_by_cid, rows, job_configs, states) -- the
    last two feed measure_startup_s."""
    chunks = chunk_round_robin(configs, slots)
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, chunk in enumerate(chunks):
        time_est = _job_time_est(ctx, len(chunk), 1, s_est)
        if would_exceed_budget(ctx.elapsed(), time_est["total_run_s"], ctx.budget_s):
            print("%s slot %d: SKIP -- estimated end would exceed budget" % (name_prefix, i))
            continue
        job_id, _path, _te = submit_configs_job(ctx, chunk, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=time_est)
        job_ids.append(job_id)
        job_configs[job_id] = chunk
        for c in chunk:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs)" % (name_prefix, i, job_id, len(chunk)))

    states = _jobs.wait_for_jobs(ctx, job_ids, name_prefix)
    rows = ctx.ledger_reader(ctx.case, job_ids)
    return job_ids, configs_by_cid, rows, job_configs, states


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
    `s_est`."""
    per_run_s = ctx.startup_s + ctx.steps * s_est
    round0_budget_s = min(ctx.budget_s - ctx.elapsed(), ctx.budget_s * ROUND0_BUDGET_FRACTION)
    configs_per_job = _configs_for_budget(round0_budget_s, per_run_s, slots)
    return configs_per_job, slots * configs_per_job


def _adopt_round0_startup(ctx: Ctx, job_configs: dict, states: dict, wave0_results: dict) -> None:
    """Refresh ctx.startup_s from wave 0's own Slurm elapsed time, logged either way so the run
    says which startup is in effect."""
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
                  slots: int, wave: int = 1) -> tuple[list, dict, list] | None:
    """One more wave of not-yet-measured candidates sized at the measured s/step: the pre-round0
    estimate can be the 10 s/step no-ledger fallback, 333x too slow (sc26p2 2026-09-22).
    Returns (job_ids, configs_by_cid, rows), or None when there is nothing to add."""
    topup_per_run_s = ctx.startup_s + ctx.steps * refreshed_s_est
    topup_budget_s = min(ctx.budget_s - ctx.elapsed(), ctx.budget_s * ROUND0_BUDGET_FRACTION)
    topup_configs_per_job = _configs_for_budget(topup_budget_s, topup_per_run_s, slots)
    topup_k = slots * topup_configs_per_job
    fresh = sample_more_diverse_configs(ctx, list(configs_by_cid.values()), topup_k,
                                        ctx.seed + wave)
    print("round0 top-up %d: s_est=%.3fs/step (measured, was %.3fs/step) startup_s=%.1fs "
          "configs_per_job=%d candidates=%d"
          % (wave, refreshed_s_est, s_est, ctx.startup_s, topup_configs_per_job, len(fresh)))
    if not fresh:
        return None
    topup_ids, topup_by_cid, topup_rows, _jc, _st = _submit_round0_wave(
        ctx, fresh, slots, refreshed_s_est, "round0-topup")
    return topup_ids, topup_by_cid, topup_rows


def run_round0(ctx: Ctx, fill: bool = False) -> tuple[list, dict, list]:
    """Round 0 at ctx.nodes: ANCHOR + a diverse decomp_space sample (or the full space,
    ctx.full_space). With ctx.proxy the sample is ranked by proxy_stage.select and only its picks
    are real-probed, with no top-up waves after. Without the proxy (off, or it could not run) the
    sample is real-probed over floor(max_nodes/nodes) parallel jobs, then the measured startup and
    a top-up wave when the measured s/step undercuts the estimate; with `fill` and a time cap,
    top-up waves sized off the measured startup and s/step continue until the cap (less one race
    job) or the candidates run out. Returns (job_ids, configs_by_cid, rows)."""
    s_est, s_source, n_rows = ctx.estimate_s_per_step()
    if ctx.full_space:
        return _run_full_space_round0(ctx, s_est, compute_slots(ctx.max_nodes, ctx.nodes))
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
    if ctx.proxy:
        from . import proxy_stage
        configs0 = proxy_stage.select(ctx, configs0)

    job_ids, configs_by_cid, rows, job_configs, states = _submit_round0_wave(
        ctx, configs0, _proxy_slots(ctx, slots), s_est, "round0")
    if job_ids:
        wave0_results = aggregate_by_cid(rows, configs_by_cid)
        _adopt_round0_startup(ctx, job_configs, states, wave0_results)
        refreshed_s_est = update_s_est(wave0_results, s_est)
        fill = fill and math.isfinite(ctx.budget_s)
        wave = 1
        # the proxy already ranked a 3x pool: no unranked top-up waves after it
        while (not ctx.proxy_used and (fill or (wave == 1 and refreshed_s_est < s_est))
               and ctx.elapsed() < ctx.budget_s):
            race = _job_time_est(ctx, RACE_MAX, 1, refreshed_s_est)["total_run_s"] if fill else 0.0
            topup = _round0_topup(dataclasses.replace(ctx, budget_s=ctx.budget_s - race),
                                  configs_by_cid, s_est, refreshed_s_est, slots, wave)
            if not topup or not topup[0]:
                break
            topup_ids, topup_by_cid, topup_rows = topup
            job_ids = job_ids + topup_ids
            configs_by_cid.update(topup_by_cid)
            rows = rows + topup_rows
            refreshed_s_est = update_s_est(aggregate_by_cid(rows, configs_by_cid), refreshed_s_est)
            wave += 1
    return job_ids, configs_by_cid, rows


def _default_proposer(ctx: Ctx) -> Callable:
    """proposer(state, batch_size, kappa) over predictor/decomp_search.py, imported here so the
    tuner itself never needs numpy / scikit-learn / catboost. Other cases' ledgers are the
    cross-case prior (this case's own rows are already in `state`)."""
    from wtune.predictor import decomp_search   # predictor/decomp_search.py

    case_fields = decomp_search.case_row_fields(ctx.case_json)
    prior_rows = []
    paths = [p for d in [ctx.ledger_dir] + list(ctx.history_dirs)
             for p in sorted(glob.glob(os.path.join(d, "ledger_*.csv")))]
    for path in paths:
        if os.path.basename(path).startswith("ledger_%s_" % ctx.case):
            continue    # this case's own ledgers are already in `state` (case_measured_rows)
        try:
            rows = decomp_search.benchmark.load_ledger(path)
        except Exception:
            continue    # best-effort cross-case prior; a malformed sibling ledger is not fatal
        prior_rows.extend(r for r in rows if not ctx.build_id
                          or sweep.row_build_id(r) in ("", ctx.build_id))   # never another build

    def proposer(state, batch_size, kappa=None):
        return decomp_search.propose_for_case(
            ctx.case, ctx.case_json, case_fields, prior_rows, state, batch_size,
            total_cores=ctx.nodes * cores_used(ctx), nodes=ctx.nodes,
            hardware_id=ctx.hardware_id, kappa=ctx.kappa if kappa is None else kappa)

    return proposer


def _round_batch_size(ctx: Ctx, round_idx: int, elapsed: float, s_est: float) -> int | None:
    """Configs adaptive round `round_idx` may propose, or None (printed) when the rounds stop
    at 75% of the budget. The batch is one job's share; run_adaptive_rounds spreads it over
    the free node slots, so a round ends sooner instead of measuring more."""
    if elapsed >= ADAPTIVE_STOP_FRACTION * ctx.budget_s:
        print("adaptive rounds: stopping before round %d -- elapsed already >= %.0f%% of budget"
              % (round_idx, ADAPTIVE_STOP_FRACTION * 100))
        return None
    remaining_s = ctx.budget_s - elapsed
    per_run_s = ctx.startup_s + ctx.steps * s_est
    return _configs_for_budget(min(remaining_s, ctx.budget_s * ROUND0_BUDGET_FRACTION),
                               per_run_s, 1)


def _clear_leader(all_results: dict, new_results: dict, noise: float) -> str | None:
    """Why the adaptive rounds may stop: the leader is ahead of the runner-up by more than
    the noise and this round found nothing better (the leader is not one of its configs)."""
    ranked = sorted((r for r in all_results.values()
                     if r.get("median") is not None and not r.get("dirty")),
                    key=lambda r: r["median"])
    if len(ranked) < 2 or ranked[0]["cid"] in new_results:
        return None
    leader, runner_up = ranked[0], ranked[1]
    if runner_up["median"] <= leader["median"] * (1.0 + noise):
        return None
    return ("%s (%.4g s/step) leads %s (%.4g) by more than %.0f%% and this round found nothing "
            "faster" % (leader["cid"], leader["median"], runner_up["cid"], runner_up["median"],
                        noise * 100))


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
                    elapsed: float, slots: int = 1) -> tuple[list[dict], dict] | None:
    """(configs, time estimate of its longest job) of the round, or None (printed) when it may
    not start (projected finish > 75% of the budget)."""
    time_est = _job_time_est(ctx, int(math.ceil(len(configs_out) / float(max(1, slots)))), 1,
                             s_est)
    projected_end = elapsed + time_est["total_run_s"]        # RAW/EXPECTED -- never job_hours
    if projected_end > ADAPTIVE_STOP_FRACTION * ctx.budget_s:
        print("adaptive rounds: stopping before round %d -- projected finish %s > %.0f%% "
              "of budget" % (round_idx, _fmt_hms(projected_end), ADAPTIVE_STOP_FRACTION * 100))
        return None
    return configs_out, time_est


def run_adaptive_rounds(ctx: Ctx, all_configs_by_cid: dict, all_results: dict,
                        s_est: float) -> None:
    """Up to ADAPTIVE_MAX_ROUNDS rounds, each dealt over the free node slots as parallel jobs,
    proposed by predictor/decomp_search.py (ctx.proposer, else lazily imported) from everything
    measured so far; updates `all_configs_by_cid` / `all_results` in place and stops at the
    budget lines, or early once the leader is clearly ahead (_clear_leader)."""
    proposer = ctx.proposer
    if proposer is None:
        try:
            proposer = _default_proposer(ctx)
        except ImportError as e:
            print("adaptive rounds skipped: predictor/decomp_search.py not importable (%s)" % e)
            return

    state: dict[str, Any] = dict(configs={}, measured={}, rounds=[], submitted=[])
    for cid, r in all_results.items():
        if r["config"] is None:
            continue
        state["configs"][cid] = r["config"]
        state["measured"][cid] = [dict(run_id="%s_%s_r%d_seed" % (ctx.case, cid, i),
                                        job_id=None, rep_index=i, s_per_step_median=s)
                                   for i, s in enumerate(r["s_values"])]
    for cid, c in all_configs_by_cid.items():      # submitted but failed: never proposed again
        if c is not None:
            state["configs"].setdefault(cid, c)
            state["submitted"].append(cid)

    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    noise = wtune.machine.measured("noise_frac")
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
        planned = _plan_round_job(ctx, configs_out, s_est, round_idx, elapsed, slots)
        if planned is None:
            return
        configs_out, _time_est = planned

        for c in configs_out:
            all_configs_by_cid[c["id"]] = c
            state["configs"][c["id"]] = c
            state["submitted"].append(c["id"])
        job_ids, _by_cid, rows, job_configs, _states = _submit_round0_wave(
            ctx, configs_out, slots, s_est, "round%d" % round_idx)
        job_of = {c["id"]: j for j, chunk in job_configs.items() for c in chunk}
        new_results = aggregate_by_cid(rows, all_configs_by_cid)
        new_results.pop(ANCHOR_END_ID, None)       # drift check only, never a candidate
        for cid, r in new_results.items():
            all_results[cid] = r
            job_id = job_of.get(cid)
            state["measured"][cid] = [dict(run_id="%s_%s_r%d_%s" % (ctx.case, cid, i, job_id),
                                            job_id=job_id, rep_index=i, s_per_step_median=s)
                                       for i, s in enumerate(r["s_values"])]
        s_est = update_s_est(all_results, s_est)   # keep sizing off this run's freshest numbers
        why = _clear_leader(all_results, new_results, noise)
        if why:
            print("adaptive rounds: early stop after round %d -- %s" % (round_idx, why))
            return


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
    """Re-measure the confirm candidates (top 3, the fastest measured, the anchor) in one job of ctx.confirm_reps
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


def _remeasure_dirty_best(ctx: Ctx, all_results: dict, s_est: float) -> None:
    """A measurement taken on a dirty node (orphan wrf.exe, foreign load) is kept but marked
    `dirty`. When the fastest layout is one of them, measure it once more and use the new
    number if that run was clean; otherwise the mark stays and the report warns."""
    measured = [r for r in all_results.values() if r.get("config")]
    if not measured:
        return
    best = min(measured, key=lambda r: r["median"])
    if not best.get("dirty"):
        return
    cid = best["cid"]
    print("WARNING: the fastest layout %s (%.4f s/step) was measured on a dirty node (%s)"
          % (cid, best["median"], "; ".join(best["dirty"])))
    est = estimate_run_time([(best["config"], best["median"])], 2, ctx.steps,
                            startup_s=ctx.startup_s)
    if would_exceed_budget(ctx.elapsed(), est["total_run_s"], ctx.budget_s):
        print("recheck: SKIP -- no budget left; %s keeps its dirty-node number" % cid)
        return
    job_id, _path, _te = submit_configs_job(ctx, [best["config"]], reps=2, name_suffix="recheck",
                                            s_est=s_est, time_est=est)
    print("recheck: submitted job %s (%s, REPS=2)" % (job_id, cid))
    _jobs.wait_for_jobs(ctx, [job_id], "recheck")
    again = aggregate_by_cid(ctx.ledger_reader(ctx.case, [job_id]), {cid: best["config"]}).get(cid)
    if again is None or again.get("dirty"):
        print("recheck: %s -- %s keeps its dirty-node number"
              % ("no measurement" if again is None else "node still dirty", cid))
        return
    print("recheck: %s clean re-measure %.4f s/step (was %.4f)" % (cid, again["median"],
                                                                 best["median"]))
    all_results[cid] = again


def _proxy_slots(ctx: Ctx, slots: int) -> int:
    """Parallel jobs once the proxy ranked the layouts: one config per job, up to
    PROXY_JOB_NODES_FACTOR x max_nodes nodes at once (queue wait is not tune time; a config takes
    ~startup + probe steps, 4-5 min on an ISC-size case, so a job of 3 ran 14 min)."""
    if not ctx.proxy_used:
        return slots
    from .proxy_stage import PROXY_JOB_NODES_FACTOR
    return max(slots, PROXY_JOB_NODES_FACTOR * ctx.max_nodes // max(1, ctx.nodes))


def _race_leaders(ctx: Ctx, all_results: dict, s_est: float) -> None:
    """With no confirm stage (-O2) the pick rests on one run: the layouts that tie with the
    fastest (common.close_to_best) run once more, in one job, merged into their results."""
    measured = {cid: r for cid, r in all_results.items() if r.get("config")}
    cids = close_to_best({cid: r["median"] for cid, r in measured.items()},
                         wtune.machine.measured("noise_frac"))
    if not cids:
        return
    configs = [measured[c]["config"] for c in cids]
    est = estimate_run_time([(measured[c]["config"], measured[c]["median"]) for c in cids], 1,
                            ctx.steps, startup_s=ctx.startup_s)
    if would_exceed_budget(ctx.elapsed(), est["total_run_s"], ctx.budget_s):
        print("race: SKIP -- no budget left for a second run of %s" % cids)
        return
    groups = chunk_round_robin(configs, min(len(configs), _proxy_slots(ctx, 1)))
    job_ids = []
    for k, g in enumerate(groups):
        g_est = estimate_run_time([(c, measured[c["id"]]["median"]) for c in g], 1, ctx.steps,
                                  startup_s=ctx.startup_s)
        job_id, _path, _te = submit_configs_job(ctx, g, reps=1,
                                                name_suffix="race" if k == 0 else "race%d" % k,
                                                s_est=s_est, time_est=g_est)
        job_ids.append(job_id)
    print("race: submitted job %s (%s once more, within 2x noise of the fastest)"
          % (",".join(map(str, job_ids)), ", ".join(cids)))
    _jobs.wait_for_jobs(ctx, job_ids, "race")
    again = aggregate_by_cid(ctx.ledger_reader(ctx.case, job_ids),
                             {c: measured[c]["config"] for c in cids})
    for cid, r in again.items():
        all_results[cid] = merge_aggs(all_results[cid], r)


def run_tuning_at(ctx: Ctx, s_est: float | None = None, adaptive: bool = True,
                  confirm: bool = True) -> tuple[dict, dict | None, str | None]:
    """round0 -> adaptive rounds -> confirm at ctx.nodes. Returns (all_results,
    confirm_results, anchor_cid). A queue cap or a user stop keeps what was measured."""
    first_job = len(ctx.owned_jobs)
    try:
        return _run_tuning_at(ctx, s_est, adaptive, confirm)
    except QueueCapExceeded as e:
        jobs = [j for j in ctx.owned_jobs[first_job:] if j in ctx.job_configs]
        configs_by_cid = {c["id"]: c for j in jobs for c in ctx.job_configs[j]}
        all_results = aggregate_by_cid(ctx.ledger_reader(ctx.case, jobs) if jobs else [],
                                       configs_by_cid)
        print("layouts at N=%d: %s; keeping the %d measured" % (ctx.nodes, e, len(all_results)))
        return all_results, None, "ANCHOR" if "ANCHOR" in all_results else None


def _run_tuning_at(ctx: Ctx, s_est: float | None, adaptive: bool,
                   confirm: bool) -> tuple[dict, dict | None, str | None]:
    if s_est is None:
        s_est, _src, _n = ctx.estimate_s_per_step()
    job_ids0, configs_by_cid, rows0 = run_round0(ctx, fill=not adaptive and not confirm)
    all_results = aggregate_by_cid(rows0, configs_by_cid)
    anchor_cid = "ANCHOR" if "ANCHOR" in configs_by_cid else None
    # later stages size off what round0 measured, not the pre-round0 estimate (conus12
    # 2026-09-22: the stale 10 s/step fallback left 13.6 of 30 minutes unused)
    s_est = update_s_est(all_results, s_est)

    if job_ids0 and adaptive and not ctx.full_space:       # the full space leaves nothing to propose
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)

    _remeasure_dirty_best(ctx, all_results, s_est)

    if all_results and not confirm:
        _race_leaders(ctx, all_results, s_est)

    confirm_job_id, confirm_results = (None, None)
    if all_results and confirm:
        confirm_job_id, confirm_results = run_confirm(ctx, configs_by_cid, all_results, anchor_cid,
                                                        s_est)

    return all_results, confirm_results, anchor_cid
