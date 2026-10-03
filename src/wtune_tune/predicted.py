"""--rank-by predicted (decision 0055): rank candidates by predicted whole-run compute+init built
from measurements at rebuilt ERA5 points. Stage 1 searches (t0, or from P*'s shared restart),
stage 2 measures the shortlist + ANCHOR/ANCHOREND at every point; the budget is RUNNING time."""
from __future__ import annotations

import dataclasses
import math
import os
import statistics
from typing import Any

from ._deps import coldstart, sweep, wtune_predict, wtune_probe_plan
from .budget import (TOP_K_STAGE2, _fallback_s_per_step, _restart_read_s, _stage2_estimate,
                     describe_cost_model, describe_stage2_reserve, estimate_run_time,
                     estimate_stage2_s, ranking_context, ranking_cost_model, stage2_concurrency,
                     update_s_est)
from .candidates import aggregate_by_cid
from .common import ANCHOR_END_ID, _fmt_hms, compute_slots, parse_run_id
from .context import Ctx
from .jobs import job_timing
from .points import (_predicted_winner_text, choose_primary_point, fit_stage2,
                     predicted_spinup_min, rebuilt_steps)
from .prediction import _drift_calibration
from .report import local_param_table
from .rounds import run_adaptive_rounds, run_round0
from . import points as _points

RANK_BY_CHOICES = ("t0", "predicted")
STAGE1_CHOICES = ("t0", "pstar")       # predicted mode's stage 1 (see run_predicted)
PREDICTED_MAX_POINTS = 3               # default probe.max_points (--max-points / --config win)
MAX_WALL_FACTOR = 3.0                  # hard wall-clock cap = 3x the budget
REBUILD_PREP_EST_S = 480.0             # s: ERA5 fetch + WPS + real.exe, 1-node jobs in parallel
                                       # (conus25 e2e attempt 2, 2026-09-27: 7 min)


def stage1_normalized_ranking(all_results: dict, stage1_jobs: dict) -> tuple[list, list]:
    """(ranked, excluded) of stage 1 (cross-node bias, decision 0055 gap 1): each run divided by
    its own job's s_anchor_job, key = the median ratio over the candidate's non-excluded runs.
    ranked [{cid, config, median, norm_key, jobs}] by (norm_key, cid), never ANCHOR/ANCHOREND;
    excluded [{cid, median, reason}] for candidates with no run left."""
    ranked, excluded = [], []
    for cid, r in sorted(all_results.items()):
        if cid in ("ANCHOR", ANCHOR_END_ID) or r.get("config") is None:
            continue
        ratios, jobs, why = [], [], []
        for s, rid in zip(r["s_values"], r.get("run_ids") or []):
            job = str((parse_run_id(rid) or {}).get("job"))
            info = stage1_jobs.get(job)
            if info is None or not info.get("s_anchor_job"):
                why.append("job %s: no anchor normalizer" % job)
            elif info["excluded"]:
                why.append("job %s: %s" % (job, info["reason"]))
            else:
                ratios.append(s / info["s_anchor_job"])
                jobs.append(job)
        if ratios:
            ranked.append(dict(cid=cid, config=r["config"], median=r["median"],
                               norm_key=statistics.median(ratios), jobs=jobs))
        else:
            excluded.append(dict(cid=cid, median=r.get("median"),
                                 reason="; ".join(why) or "no run"))
    ranked.sort(key=lambda e: (e["norm_key"], e["cid"]))
    return ranked, excluded


def predicted_min_budget_s(ctx: Ctx, n_points: int, steps: int, spinup_min: float, dt: float,
                           stepra: int, s_step: float, slots: int,
                           stage1_mode: str = "pstar") -> tuple[float, str]:
    """(s, detail): the least RUNNING budget predicted mode can succeed with -- rebuild prep + a
    minimal stage 1 (ANCHOR + 1 + ANCHOREND; pstar from the spin-up, t0 at ctx.steps) + stage 2
    for that candidate + anchors at every point. Checked before anything is submitted."""
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spinup_min * 60.0 / dt,
                           stepra, s_step)
    if stage1_mode == "t0":
        stage1 = estimate_run_time(3, 1, ctx.steps, s_est=s_step,
                                   startup_s=ctx.startup_s)["total_run_s"]
        s1_note = "3 t0 runs x %d steps" % ctx.steps
    else:
        stage1 = m["spin_s"] + 3 * m["per_run_s"]
        s1_note = "spin-up + 3 runs"
    stage2 = estimate_stage2_s(n_points, 3, steps, spinup_min, dt, s_step, slots, ctx.startup_s,
                               _restart_read_s(ctx), stepra=stepra)
    total = REBUILD_PREP_EST_S + stage1 + stage2
    return total, ("minimum %s = rebuild prep %s + stage 1=%s %s (%s) + stage 2 %s "
                   "(%d point(s), %d concurrent -- nothing observed yet, conservative); %s"
                   % (_fmt_hms(total), _fmt_hms(REBUILD_PREP_EST_S), stage1_mode,
                      _fmt_hms(stage1), s1_note, _fmt_hms(stage2), n_points, slots,
                      describe_cost_model(m)))


def run_predicted(ctx: Ctx, nodes: int, nodes_note: str, s_est: float) -> dict:
    """Plan the points and P* (the most sunlit), rebuild them; stage 1 = ctx.stage1_mode: t0
    (default: the broad t0 search, --minutes is stage 1 only, rebuild prep beside it) or pstar
    (ranking from P*'s shared restart, --minutes covers both stages); stage 2 = the top 3 by
    anchor-normalized stage-1 median + anchors at ALL points; winner = min predicted whole-run
    compute+init. Returns the report dict."""
    if wtune_predict is None or wtune_probe_plan is None or coldstart is None:
        raise SystemExit("--rank-by predicted needs wtune_predict, wtune_probe_plan and "
                         "tools/coldstart_sample importable")
    pp = _predicted_plan(ctx)
    anchor, n_led = _predicted_anchor_and_prior(ctx, pp)
    stage2_est0 = _predicted_feasibility(ctx, pp)
    rebuild_args = (ctx, pp.points, anchor, pp.case_dir, ctx.namelist_path, ctx.case_json,
                    os.path.join(ctx.out_dir, "rebuilt"), pp.spinup_min, pp.stepra, pp.dt)
    rebuilt, prep_handle = None, None
    if ctx.stage1_mode == "pstar":
        rebuilt = _predicted_start_pstar(ctx, pp, anchor, n_led, rebuild_args)
    else:
        prep_handle = _predicted_start_t0(ctx, anchor, n_led, rebuild_args)
    print(describe_stage2_reserve(ctx))

    all_results, stage1, stage1_excluded, top, s_by_cid = _predicted_stage1(ctx, s_est)
    reserve_note = describe_stage2_reserve(ctx)
    stage1_running_s = ctx.elapsed()
    ctx.ranking = None                                   # stage 2 builds its own per-point contexts
    ctx.stage1 = None
    ctx.stage1_extra_reserve_s = 0.0
    if prep_handle is not None:
        ctx.background_jobs = []
        rebuilt = _points.finish_rebuilt_points(ctx, prep_handle)
    top, stage2_est, skip_reason, meas_by_cid, gates, stage2_jobs = _predicted_stage2(
        ctx, pp, stage1, top, anchor, rebuilt, s_by_cid)
    stage2_running_s = ctx.elapsed() - stage1_running_s
    calibration = _drift_calibration(ctx)
    cfg_by_cid = dict({c["id"]: c for c in top}, ANCHOR=anchor)
    ranked = _points.rank_by_predicted_total(
        meas_by_cid, pp.run_start, pp.run_end, pp.dt, pp.stepra, calibration, pp.sample_points,
        pp.case_dir, pp.nl, {cid: c["ranks"] for cid, c in cfg_by_cid.items()}, nodes)
    report = build_predicted_report(ctx, nodes, nodes_note, ranked, meas_by_cid, cfg_by_cid,
                                    all_results, rebuilt, pp.p_idx, gates, stage2_jobs, pp.notes)
    norm = {e["cid"]: e["norm_key"] for e in stage1}
    for row in report["ranking"]:
        row["stage1_norm_key"] = norm.get(row["id"])
    report.update(spinup_min=pp.spinup_min,
                  stage1_ranking=[dict(id=e["cid"], median_s_per_step=e["median"],
                                       norm_key=e["norm_key"], jobs=e["jobs"],
                                       stage2=e["cid"] in {c["id"] for c in top})
                                  for e in stage1],
                  stage1_excluded=stage1_excluded, not_measured=list(ctx.not_measured),
                  stage1_trimmed=list(ctx.stage1_trimmed), stage1_mode=ctx.stage1_mode,
                  stage2_reserve_note=reserve_note,
                  stage2_overrun_s=ctx.stage2_overrun_s,
                  stage2_concurrency=stage2_concurrency(ctx),
                  stage1_max_running=ctx.stage1_max_running,
                  stage2_estimate_s=stage2_est, stage2_skipped_reason=skip_reason,
                  stage1_running_min=stage1_running_s / 60.0,
                  stage2_running_min=stage2_running_s / 60.0,
                  stage2_estimate_min=stage2_est0 / 60.0,
                  prep_submit_min=ctx.prep_submit_s / 60.0,
                  total_running_min=(stage1_running_s + stage2_running_s
                                     + ctx.prep_submit_s) / 60.0,
                  stage2_minutes_cap=(ctx.stage2_budget_s / 60.0
                                      if ctx.stage2_budget_s is not None else None),
                  budget_semantics=("--minutes = stage-1 search only; stage 2 separate"
                                    if ctx.stage1_mode == "t0" else
                                    "--minutes = stage 1 + stage-2 reserve (pstar)"))
    return report


@dataclasses.dataclass
class _PredictedPlan:
    """Step 0 of run_predicted: the namelist's run window, time step and radiation step, the
    planned ERA5-rebuildable points (+ notes), the spin-up and P* (index p_idx)."""
    case_dir: str
    nl: Any
    run_start: Any
    run_end: Any
    dt: float
    stepra: int
    points: list
    notes: list
    spinup_min: int
    sample_points: Any
    p_idx: int


def _predicted_plan(ctx: Ctx) -> _PredictedPlan:
    """Plan the points and P* (printed); SystemExit when no point can be rebuilt. No jobs."""
    case_dir = sweep._expand(ctx.case_json["prep_dir"])
    _n, _run_s, dt0, _note = wtune_predict.get_d01_steps(
        wtune_predict.parse_namelist(ctx.namelist_path))
    spinup_min = predicted_spinup_min(dt0)
    nl, run_start, run_end, dt, stepra, points, notes = _points.plan_predicted_points(ctx, spinup_min)
    print("predicted mode: spin-up %d min (%d steps of %gs)" % (spinup_min,
                                                                 spinup_min * 60 // dt, dt))
    if not points:
        raise SystemExit("--rank-by predicted: no ERA5-rebuildable point planned (%s)"
                         % "; ".join(notes))
    sample_points, _src = wtune_predict.read_wrfinput_sample_points(case_dir)
    p_idx, p_star, p_frac = choose_primary_point(points, sample_points)
    print("predicted mode: %d point(s) %s; P* = point %d %s (sunlit fraction %.2f)"
          % (len(points), ", ".join(p.at.strftime("%m-%d_%H:%M") for p in points), p_idx,
             p_star.at.isoformat(), p_frac))
    return _PredictedPlan(case_dir=case_dir, nl=nl, run_start=run_start, run_end=run_end, dt=dt,
                          stepra=stepra, points=points, notes=notes, spinup_min=spinup_min,
                          sample_points=sample_points, p_idx=p_idx)


def _predicted_anchor_and_prior(ctx: Ctx, pp: _PredictedPlan) -> tuple[dict, int]:
    """(ANCHOR at ctx.nodes, ledger runs behind the prior); sets ctx.s_prior from the ledger,
    else the cell-step model, else the ledger fallback. SystemExit without a legal ANCHOR."""
    anchor = sweep.build_anchor(ctx.case_json, ctx.nodes * ctx.cores_per_node, ctx.nodes,
                                ctx.hardware_id)
    if anchor is None:
        raise SystemExit("--rank-by predicted: no legal ANCHOR decomposition at %d node(s)"
                         % ctx.nodes)
    s_led, _s_src, n_led = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                      ctx.hardware_id)
    if n_led:
        ctx.s_prior = s_led
    else:
        try:
            ctx.s_prior = _fallback_s_per_step(pp.nl, anchor["ranks"], anchor.get("omp", 1))
        except Exception:
            ctx.s_prior = s_led
    return anchor, n_led


def _predicted_feasibility(ctx: Ctx, pp: _PredictedPlan) -> float:
    """Set the stage-2 reserve inputs, print the feasibility and stage-2 estimate lines and
    return the stage-2 estimate (s); SystemExit when --minutes cannot hold the minimum (t0: one
    minimal stage-1 job; pstar: both stages). stage1=t0 without --max-wall-minutes: the wall
    cap covers stage 2 too."""
    points, steps = pp.points, rebuilt_steps(pp.stepra)
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    ctx.reserve = dict(n_points=len(points), n_configs=TOP_K_STAGE2 + 2, steps=steps,
                       spinup_min=pp.spinup_min, dt=pp.dt, stepra=pp.stepra, slots=slots)
    stage2_est0 = _stage2_estimate(ctx, TOP_K_STAGE2 + 2, ctx.s_prior)
    if ctx.stage1_mode == "t0":
        need_s = estimate_run_time(3, 1, ctx.steps, s_est=ctx.s_prior,
                                   startup_s=ctx.startup_s)["total_run_s"]
        detail = ("stage-1 minimum %s = one t0 job, ANCHOR + 1 + ANCHOREND x %d steps at "
                  "%.3fs/step + startup %.0fs" % (_fmt_hms(need_s), ctx.steps, ctx.s_prior,
                                                  ctx.startup_s))
    else:
        need_s, detail = predicted_min_budget_s(ctx, len(points), steps, pp.spinup_min, pp.dt,
                                                pp.stepra, ctx.s_prior, stage2_concurrency(ctx),
                                                stage1_mode=ctx.stage1_mode)
    print("predicted mode: feasibility -- %s" % detail)
    ctx.stage2_estimate_start_s = stage2_est0
    print("predicted mode: stage 2 will need about %.0f min %s (%d point(s) x (spin-up + %d "
          "runs), concurrency %d, estimated at %.3fs/step)"
          % (stage2_est0 / 60.0,
             "MORE, on top of --minutes" if ctx.stage1_mode == "t0" else "INSIDE --minutes",
             len(points), TOP_K_STAGE2 + 2, stage2_concurrency(ctx), ctx.s_prior))
    if ctx.stage1_mode == "t0" and not ctx.max_wall_explicit and ctx.max_wall_s is not None:
        ctx.max_wall_s = MAX_WALL_FACTOR * (ctx.budget_s + stage2_est0)
    if ctx.budget_s < need_s:
        raise SystemExit("predicted mode needs at least %.0f min of running time for %s at %d "
                         "node(s) (have %.0f): %s"
                         % (math.ceil(need_s / 60.0), ctx.case, ctx.nodes, ctx.budget_s / 60.0,
                            detail))
    return stage2_est0


def _predicted_start_pstar(ctx: Ctx, pp: _PredictedPlan, anchor: dict, n_led: int,
                           rebuild_args: tuple) -> list[dict]:
    """stage1=pstar: rebuild every point first (awaited), then rank at P*'s shared restart: sets
    ctx.steps / stage1 / ranking. Returns the rebuilt points."""
    rebuilt = _points.prepare_rebuilt_points(*rebuild_args)
    rp_star = rebuilt[pp.p_idx]
    ctx.steps = rp_star["steps"]
    ctx.stage1 = dict(mode="pstar", anchor=anchor)
    ctx.ranking = ranking_context(rp_star["case_id"], rp_star["cases_json"], pp.spinup_min,
                                  rp_star["steps"], anchor,
                                  restart_io_form=ctx.restart_io_form)
    ctx.ranking["dt"] = pp.dt
    ctx.ranking["stepra"] = pp.stepra
    print("stage 1 (pstar): ranking at P* (case %s, start %s, STEPS=%d, "
          "RANKING_RESTART_IO_FORM=%d); s_prior %.4fs/step [%s]"
          % (rp_star["case_id"], rp_star["T"].isoformat(), ctx.steps, ctx.restart_io_form,
             ctx.s_prior or 0.0, "ledger" if n_led else "cell-step model"))
    return rebuilt


def _predicted_start_t0(ctx: Ctx, anchor: dict, n_led: int, rebuild_args: tuple) -> dict:
    """stage1=t0: submit the rebuild prep now (it runs beside stage 1, awaited before stage 2;
    its controller fetch/submit time is not charged) and set up a t0 stage 1. Returns the prep
    handle."""
    t_prep = ctx.clock.now()
    prep_handle = _points.submit_rebuilt_points(*rebuild_args)
    ctx.prep_submit_s = ctx.clock.now() - t_prep
    ctx.uncharged_s += ctx.prep_submit_s
    ctx.background_jobs = list(prep_handle["job_ids"])
    ctx.stage1 = dict(mode="t0", anchor=anchor)
    ctx.reserve_in_stage1 = False
    ctx.stage1_extra_reserve_s = 0.0
    print("stage 1 (t0): broad t0 search on the whole --minutes (%s of running time), "
          "STEPS=%d, ANCHOR/ANCHOREND in every job; rebuild prep job(s) %s running beside "
          "it; s_prior %.4fs/step [%s]"
          % (_fmt_hms(ctx.budget_s), ctx.steps, ",".join(prep_handle["job_ids"]),
             ctx.s_prior or 0.0, "ledger" if n_led else "cell-step model"))
    return prep_handle


def _predicted_stage1(ctx: Ctx, s_est: float) -> tuple[dict, list, list, list, dict]:
    """round0 + adaptive rounds, then the anchor-normalized ranking. Returns (all_results,
    stage-1 ranking, excluded, top: the stage-2 shortlist configs, median s/step by cid);
    SystemExit when stage 1 measured nothing."""
    job_ids0, configs_by_cid, rows0 = run_round0(ctx)
    configs_by_cid.pop(ANCHOR_END_ID, None)
    all_results = aggregate_by_cid(rows0, configs_by_cid)
    all_results.pop(ANCHOR_END_ID, None)
    s_est = update_s_est(all_results, s_est)
    if job_ids0:
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)
    all_results.pop(ANCHOR_END_ID, None)
    stage1, stage1_excluded = stage1_normalized_ranking(all_results, ctx.stage1_jobs)
    for x in stage1_excluded:
        print("WARNING: stage 1: %s excluded from ranking (%s)" % (x["cid"], x["reason"]))
    top = [r["config"] for r in stage1[:TOP_K_STAGE2]]
    if not top and "ANCHOR" not in all_results:
        raise SystemExit("--rank-by predicted: stage 1 measured nothing -- see gate warnings")
    s_by_cid = {cid: r["median"] for cid, r in all_results.items()}
    for x in ctx.not_measured:
        print("stage 1: job %s never started -- not measured: %s"
              % (x["job_id"], ", ".join(x["cids"]) or "-"))
    return all_results, stage1, stage1_excluded, top, s_by_cid


def _predicted_stage2(ctx: Ctx, pp: _PredictedPlan, stage1: list, top: list, anchor: dict,
                      rebuilt: list, s_by_cid: dict) -> tuple:
    """Fit the shortlist to the stage-2 budget (t0: the optional --stage2-minutes cap; pstar:
    what is left of --minutes) and measure it + anchors at every rebuilt point. Returns (top,
    estimate, skip_reason, meas_by_cid, gates, job ids); pstar records the overrun."""
    if ctx.stage1_mode == "t0":
        top, stage2_est, skip_reason = fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, pp.nl,
                                                  left_s=ctx.stage2_budget_s)
    else:
        top, stage2_est, skip_reason = fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, pp.nl)
    if skip_reason:
        print("stage 2: SKIPPED -- %s" % skip_reason)
        return top, stage2_est, skip_reason, {}, [], []
    print("stage 2: %s + ANCHOR/ANCHOREND at all %d point(s), expected running %s "
          "(elapsed %s / budget %s)"
          % (", ".join("%s(%.4fs/step, %.4fx anchor)" % (r["cid"], r["median"], r["norm_key"])
                       for r in stage1 if r["cid"] in {c["id"] for c in top})
             or "(no candidate besides ANCHOR)", len(rebuilt), _fmt_hms(stage2_est),
             _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))
    try:
        meas_by_cid, gates, stage2_jobs = _points.measure_candidates_at_points(
            ctx, top, rebuilt, anchor, s_by_cid=s_by_cid)
    finally:
        if ctx.stage1_mode != "t0":   # stage1=t0: stage 2 is outside --minutes by design
            ctx.stage2_overrun_s = max(0.0, ctx.elapsed() - ctx.budget_s)
    if ctx.stage2_overrun_s:
        print("stage 2: ran %s of running time past the budget (exempt once started)"
              % _fmt_hms(ctx.stage2_overrun_s))
    return top, stage2_est, skip_reason, meas_by_cid, gates, stage2_jobs


def build_predicted_report(ctx: Ctx, nodes: int, nodes_note: str, ranked: list,
                           meas_by_cid: dict, cfg_by_cid: dict, stage1: dict, rebuilt: list,
                           p_idx: int, gates: list, stage2_jobs: list, plan_notes: list) -> dict:
    """tune_report.json of predicted mode: the ranking by pred_total_s (+ range) with each
    point's s_normal / r / n_rad, stage-1 medians, every gate record and per-job queue vs run
    time (sacct)."""
    anchor_pred = next((e["pred_total_s"] for e in ranked if e["cid"] == "ANCHOR"), None)
    table = []
    for e in ranked:
        cfg = cfg_by_cid.get(e["cid"]) or {}
        table.append(dict(
            id=e["cid"], ranks=cfg.get("ranks"), omp=cfg.get("omp"), nproc_x=cfg.get("nx"),
            nproc_y=cfg.get("ny"), numtiles=cfg.get("numtiles"), dist=cfg.get("dist"),
            predicted_total_s=e["pred_total_s"], predicted_total_hms=_fmt_hms(e["pred_total_s"]),
            predicted_range_s=[e["pred_low_s"], e["pred_high_s"]],
            speedup_vs_anchor_predicted=(anchor_pred / e["pred_total_s"]
                                         if anchor_pred and e["pred_total_s"] else None),
            stage1_median_s_per_step=(stage1.get(e["cid"]) or {}).get("median"),
            points=[dict(at=pm.at.isoformat(), s_normal=pm.s_normal, r=pm.r, n_rad=pm.n_rad)
                    for pm in sorted(meas_by_cid.get(e["cid"], []), key=lambda m: m.at)]))
    best_cfg = cfg_by_cid.get(ranked[0]["cid"]) if ranked else None
    timing = job_timing(ctx, ctx.owned_jobs)
    return dict(
        case=ctx.case, rank_by="predicted", nodes=nodes, nodes_note=nodes_note,
        cores_per_node=ctx.cores_per_node, budget_minutes=ctx.budget_s / 60.0,
        elapsed_minutes=ctx.elapsed() / 60.0, job_ids=list(ctx.owned_jobs),
        cancelled_job_ids=list(ctx.cancelled_jobs), anchor_id="ANCHOR", ranking=table,
        best=table[0] if table else None,
        param_table=local_param_table(best_cfg, nodes) if best_cfg else None,
        points=[dict(index=rp["index"], at=rp["point"].at.isoformat(), reason=rp["point"].reason,
                     case_id=rp["case_id"], start=rp["T"].isoformat(),
                     rebuild_job=rp["job_id"], primary=(rp["index"] == p_idx))
                for rp in rebuilt],
        plan_notes=list(plan_notes), stage2_job_ids=list(stage2_jobs),
        gates=list(ctx.ranking_gates) if ctx.ranking_gates else list(gates),
        job_timing=timing,
        # running-time budget: elapsed_minutes above is RUNNING time
        running_elapsed_minutes=ctx.elapsed() / 60.0,
        wall_elapsed_minutes=ctx.wall_elapsed() / 60.0,
        queue_only_minutes=ctx.queue_wait_s / 60.0,
        max_wall_minutes=(ctx.max_wall_s / 60.0) if ctx.max_wall_s else None,
        total_queue_s=sum(t["queue_s"] for t in timing.values() if t["queue_s"] is not None),
        total_job_run_s=sum(t["run_s"] for t in timing.values() if t["run_s"] is not None),
        full_run_prediction=_predicted_winner_text(ranked[0]) if ranked else "no ranking",
        ranking_start_note=ctx.ranking_start_note)


def _predicted_ranking_lines(report: dict) -> list[str]:
    lines = ["%-10s %12s %21s %6s %4s %5s %5s %5s %9s %10s"
             % ("id", "pred_total", "range", "ranks", "omp", "nx", "ny", "tiles",
                "pred_spd", "stage1_s")]
    for row in report["ranking"]:
        lo, hi = row["predicted_range_s"]
        lines.append("%-10s %12s %21s %6s %4s %5s %5s %5s %9s %10s"
                     % (row["id"], row["predicted_total_hms"],
                        "%s-%s" % (_fmt_hms(lo), _fmt_hms(hi)), row["ranks"], row["omp"],
                        row["nproc_x"], row["nproc_y"], row["numtiles"],
                        ("%.3fx" % row["speedup_vs_anchor_predicted"])
                        if row["speedup_vs_anchor_predicted"] else "-",
                        ("%.4f" % row["stage1_median_s_per_step"])
                        if row["stage1_median_s_per_step"] else "-"))
        for pt in row["points"]:
            lines.append("    %s s_normal=%.4f r=%s n_rad=%d"
                         % (pt["at"], pt["s_normal"],
                            ("%.3f" % pt["r"]) if pt["r"] is not None else "-", pt["n_rad"]))
    return lines


def _predicted_stage_lines(report: dict) -> list[str]:
    """Stage-1 ranking, running time per stage, stage-2 overrun / skip."""
    lines = []
    if report.get("stage1_ranking") is not None:
        lines.append("")
        lines.append("stage 1=%s (spin-up %s min; key = median / own job's mean(ANCHOR, "
                     "ANCHOREND)):" % (report.get("stage1_mode", "-"), report.get("spinup_min", "-")))
        for e in report["stage1_ranking"]:
            lines.append("  %-10s norm=%.4f raw=%.4fs/step jobs=%s%s"
                         % (e["id"], e["norm_key"], e["median_s_per_step"], ",".join(e["jobs"]),
                            "  -> stage 2" if e["stage2"] else ""))
        for x in report.get("stage1_excluded") or []:
            lines.append("  %-10s EXCLUDED: %s" % (x["cid"], x["reason"]))
        for x in report.get("not_measured") or []:
            lines.append("  not measured (job %s never started before the stage-1 cutoff): %s"
                         % (x["job_id"], ", ".join(x["cids"]) or "-"))
    if report.get("stage1_running_min") is not None:
        lines.append("")
        lines.append("running time: stage 1 %.1f min + stage 2 %.1f min (estimated %.1f at start"
                     "%s) + rebuild-prep submit %.1f min = %.1f min  [%s]"
                     % (report["stage1_running_min"], report["stage2_running_min"],
                        report["stage2_estimate_min"],
                        (", cap %.1f" % report["stage2_minutes_cap"])
                        if report.get("stage2_minutes_cap") else "",
                        report.get("prep_submit_min") or 0.0, report["total_running_min"],
                        report.get("budget_semantics", "")))
    if report.get("stage2_overrun_s"):
        lines.append("")
        lines.append("STAGE 2 OVERRUN: %s of running time past the budget (stage 2 is exempt once "
                     "started; stage-2 concurrency assumed %s, stage-1 max observed running %s)"
                     % (_fmt_hms(report["stage2_overrun_s"]), report.get("stage2_concurrency"),
                        report.get("stage1_max_running")))
    if report.get("stage2_skipped_reason"):
        lines.append("")
        lines.append("STAGE 2 SKIPPED: %s" % report["stage2_skipped_reason"])
    return lines


def _predicted_gate_job_lines(report: dict) -> list[str]:
    lines = [""]
    for g in report["gates"]:
        if g.get("excluded"):
            lines.append("gate stage1   job %s: EXCLUDED  %s" % (g["job_id"], g["exclude_reason"]))
        drift = ("%+.2f%%" % (100 * g["anchor_drift"])) if g.get("anchor_drift") is not None else "-"
        lines.append("gate %-8s job %s: %s anchor_drift=%s%s"
                     % (g.get("stage", "-"), g["job_id"], "PASS" if g["ok"] else "FAIL", drift,
                        ("  " + "; ".join(g["failures"])) if g["failures"] else ""))
    lines.append("")
    lines.append("jobs: total queue %s, total running %s (per job below)"
                 % (_fmt_hms(report.get("total_queue_s") or 0),
                    _fmt_hms(report.get("total_job_run_s") or 0)))
    for jid, t in sorted(report["job_timing"].items()):
        lines.append("job %s: queue=%s run=%s" % (
            jid, _fmt_hms(t["queue_s"]) if t["queue_s"] is not None else "-",
            _fmt_hms(t["run_s"]) if t["run_s"] is not None else "-"))
    return lines


def render_predicted_report_text(report: dict) -> str:
    """The printed predicted-mode report."""
    lines = ["=== wtune_tune (rank_by=predicted): %s ===" % report["case"],
             "nodes=%d (%s)" % (report["nodes"], report["nodes_note"]),
             "budget=%.1f min (running time)  running=%.1f min  wall=%.1f min  queue-only=%.1f min"
             "  hard wall cap=%s  jobs=%s"
             % (report["budget_minutes"], report["running_elapsed_minutes"],
                report["wall_elapsed_minutes"], report["queue_only_minutes"],
                ("%.1f min" % report["max_wall_minutes"]) if report.get("max_wall_minutes")
                else "-", ",".join(report["job_ids"]) or "-")]
    for p in report["points"]:
        lines.append("point %d %s %-8s case=%s start=%s rebuild_job=%s%s"
                     % (p["index"], p["at"], p["reason"], p["case_id"], p["start"],
                        p["rebuild_job"], "  [P*]" if p["primary"] else ""))
    lines.append("")
    lines += _predicted_ranking_lines(report)
    lines += _predicted_stage_lines(report)
    lines += _predicted_gate_job_lines(report)
    if report["best"]:
        pt = report["param_table"]
        lines += ["", "BEST: %s" % report["best"]["id"],
                  "  namelist: " + " ".join(pt["namelist_lines"]),
                  "  launch:   " + pt["launch_line"],
                  "  env:      " + " ".join("%s=%s" % kv for kv in pt["env"].items())
                  + "  " + pt.get("ulimit", "ulimit -s unlimited")]
    lines += ["", report["full_run_prediction"]]
    for n in report.get("plan_notes") or []:
        lines.append("NOTE: " + n)
    return "\n".join(lines)
