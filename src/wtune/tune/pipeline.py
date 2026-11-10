"""--pipeline full / -O levels (decision 0059): one pass of layout search -> whole-run prediction
from rebuilt points -> physics advice under a RUNNING-time cap B, plus -O0 (no jobs).

Phases 0 plan, 1 layout, 2 predict here; 3 physics in physics_phase. They share one
PipelineRun and build its PipelineSummary (report["pipeline"]).
"""
from __future__ import annotations

import dataclasses
import os
import statistics
from typing import Any, Callable

import wtune.machine
import wtune.nodes
from ._deps import sweep, predict
from .candidates import _sweep_anchor, _sweep_anchor_fitting
from .common import ANCHOR_END_ID, _fmt_hms, _fmt_sbatch_time, compute_slots, if_set, is_set
from .context import STOP, Ctx, QueueCapExceeded
from .estimates import format_level_estimates, highest_level_within, level_estimates
from .node_search import DEFAULT_SWEEP_BUDGET_FRAC, choose_nodes
from .physics_phase import (PIPELINE_PHYSICS_MAX_CANDIDATES, WINDOW_FRAME_LEVEL,
                            _delete_window_frames, _pipeline_phase3_physics, phase2_writes_frames)
from .pipeline_state import (NOT_RUN_BUDGET, ColdEstimate, LevelInfo, MeasuredLayout, Phase2Plan,
                             PipelineRun, PipelineSummary, PointPreview, PredictionBand,
                             PredictionPlanRecord, RebuildInfo, RebuildSubmission, ShortRun,
                             StepDown, _attach_phase_timing, _mark, _phase_record)
from .points import (_predicted_winner_text, choose_point_mode, fit_stage2, pick_points,
                     predicted_spinup_min, rebuilt_steps)
from .prediction import (WholeRunPrediction, plan_prediction_points, precision_label,
                         run_rebuild_points,
                         prediction_init, time_suggestion_s)
from .report import _assert_report_is_strict_only, local_param_table
from . import candidates as _candidates
from . import context as _context
from . import jobs as _jobs
from . import node_search as _node_search
from . import points as _points
from . import prediction as _prediction

# cumulative phase shares of B: plan 5, layout 50, prediction 25, physics 20 (unused rolls forward)
PIPELINE_FRACS = (("plan", 0.05), ("layout", 0.50), ("predict", 0.25), ("physics", 0.20))
PIPELINE_MAX_QUEUE_HOURS = 6.0        # --max-queue-hours: stop once queue-only wait reaches this
PIPELINE_DEFAULT_MINUTES = 30          # min: B when neither CLI nor YAML gives one
PRECISION_DEFAULT = "10%"              # --precision: whole-run time needed to +-10 % (or MIN minutes)
_PRED_CONFIG_KEYS = ("ranks", "omp", "nx", "ny", "numtiles", "dist")


def pipeline_layouts(collect: dict, k: int = 2) -> list[dict]:
    """The top-k DISTINCT phase-1 layouts [{nodes, cores, cfg, median, anchor, cid}]: the final
    N's best first, then every other finalist's best, then the winner's next-best configs."""
    outs = collect["outcomes"]
    rows = []
    for n, o in outs.items():
        res = o.confirm_results or o.all_results or {}
        for cid, r in res.items():
            if r.get("config") and r.get("median"):
                rows.append(dict(nodes=n, cores=o.cores, cfg=r["config"], median=r["median"],
                                 anchor=o.anchor, cid=cid))
    final_n = collect["final_n"]
    rows.sort(key=lambda r: (r["nodes"] != final_n, r["median"]))
    firsts: list[dict] = []
    rest: list[dict] = []
    seen_n: set = set()
    for r in rows:
        (rest if r["nodes"] in seen_n else firsts).append(r)
        seen_n.add(r["nodes"])
    out, seen = [], set()
    for r in firsts + rest:
        ident = (r["nodes"], r["cores"], sweep._identity(r["cfg"]))
        if ident in seen:
            continue
        seen.add(ident)
        out.append(r)
        if len(out) >= k:
            break
    return out


def phase2_plans(layouts: list[dict], rebuilt: list[dict], sample_points: Any,
                 winner_only: bool = False) -> list[tuple]:
    """Phase 2's degradation ladder, largest first: [(label, layouts, points)] -- both top-2
    layouts (the gate's twin) then the winner only, at all points, then t0 + the costliest,
    then t0 alone."""
    r2 = pick_points(rebuilt, 2, sample_points)
    r1 = pick_points(rebuilt, 1)
    lay_sets = [("%d layout(s)" % len(layouts), layouts)] if len(layouts) > 1 and not winner_only \
        else []
    lay_sets.append(("winner only", layouts[:1]))
    plans, seen = [], set()
    ladder = []
    for ln, ls in lay_sets:
        ladder += [(ln, ls, rebuilt, "all %d points" % len(rebuilt)),
                   (ln, ls, r2, "2 points (t0 + costliest)")]
    ladder.append(("winner only", layouts[:1], r1, "1 point (t0)"))
    for ln, ls, pts, pn in ladder:
        key = (len(ls), tuple(rp["index"] for rp in pts))
        if key in seen:
            continue
        seen.add(key)
        plans.append(("%s x %s" % (ln, pn), ls, pts))
    return plans


def run_pipeline_full(ctx: Ctx, profile: dict, *, minutes: float | None = None, level: int = 3,
                      total_steps: int | None = None, objective: str = "time",
                      deadline_s: float | None = None, min_efficiency: float | None = None,
                      k_finalists: int = 2, nodes_policy: str = "knee",
                      sweep_budget_frac: float | None = None, core_options: list | None = None,
                      extension: bool = False, physics: bool = False,
                      physics_max_candidates: int = PIPELINE_PHYSICS_MAX_CANDIDATES,
                      max_queue_hours: float | None = PIPELINE_MAX_QUEUE_HOURS,
                      keep_frames: bool = False,
                      recheck: Callable | None = None, stop_if_short: bool = False,
                      precision_min: Any = PRECISION_DEFAULT) -> dict:
    """Decision 0059 at -O`level` (1..3): 0 plan the points and submit their ERA5 rebuilds; 1
    layout search (node search); 2 predict the top layouts' whole-run time from the rebuilt
    restarts; 3 physics advice (opt-in, -O3). `minutes` is an optional cap B on RUNNING time
    (queue wait and controller work are reported, not charged). Returns the phase-1 report,
    updated to phase 2's winner, with report["pipeline"]."""
    st = _pipeline_start(ctx, minutes, level, physics, stop_if_short, max_queue_hours,
                         total_steps, extension)
    _pipeline_pre_estimate(st, profile, nodes_policy, precision_min)
    # phase 3 (physics advice) needs phase 0's shared spun-up restarts (decision 0055); without
    # it the points are planned and rebuilt after the layout search, at its winner only
    rank_at_points = physics
    if rank_at_points:
        _pipeline_phase0_plan(st, precision_min)
    else:
        _pipeline_phase0_deferred(st)
    report, layouts = _pipeline_phase1_layout(st, profile, objective, deadline_s, min_efficiency,
                                              k_finalists, nodes_policy, sweep_budget_frac,
                                              core_options)
    health_checks = _pipeline_recheck_nodes(st, recheck)
    pplan, T_dec = _pipeline_replan_points(st, precision_min)
    if rank_at_points:
        _pipeline_phase2_predict(st, layouts, pplan, precision_min)
    else:
        _pipeline_phase2_rebuild_winner(st, layouts, pplan, precision_min)
    pred, band_eff, io_run = _pipeline_prediction(st, report, layouts, pplan, T_dec)
    _pipeline_phase3_physics(st, layouts, physics_max_candidates)
    return _pipeline_finish(st, report, pred, pplan, band_eff, io_run, health_checks, keep_frames)


REBUILD_POINT_OVERHEAD_S = 90.0     # one rebuild point's WPS + real.exe + start (Sundowner: 4:10-4:25
                                    # in all for 48 + 41 steps at 2.0-2.6 s)


def _rebuild_point_s(s_per_step, dt, measure_steps=0) -> float | None:
    """One rebuild point's job time: its spin-up and measured steps at the winner's s/step plus
    the WPS / real.exe overhead; None without both inputs."""
    try:
        steps = PREDICT_REBUILD_SPINUP_MIN * 60.0 / float(dt) + int(measure_steps or 0)
        return REBUILD_POINT_OVERHEAD_S + steps * float(s_per_step)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _pipeline_shares(physics: bool, rebuild: bool = True) -> tuple[dict, dict]:
    """(fracs, cum) of B in PIPELINE_FRACS order; without physics advice its share goes to
    layout and prediction 2:1; with phase 2's rebuild unable to run, prediction's goes to layout."""
    fracs = dict(PIPELINE_FRACS)
    if not physics:
        extra = fracs.pop("physics")
        total = fracs["layout"] + fracs["predict"]
        fracs["layout"] += extra * fracs["layout"] / total
        fracs["predict"] += extra * fracs["predict"] / total
        fracs["physics"] = 0.0
        if not rebuild:
            fracs["layout"] += fracs["predict"]
            fracs["predict"] = 0.0
    cum, acc = {}, 0.0
    for name, _f in PIPELINE_FRACS:
        acc += fracs[name]
        cum[name] = acc
    return fracs, cum


def _pipeline_start(ctx: Ctx, minutes: float | None, level: int, physics: bool,
                    stop_if_short: bool, max_queue_hours: float | None, total_steps: int | None,
                    extension: bool) -> PipelineRun:
    """Switch ctx to the running-time budget with one shared queue accumulator, print the header
    and return the PipelineRun with its initial PipelineSummary."""
    ctx.budget_mode = "running"
    ctx.shared_queue = {"s": 0.0, "cap_hit": False, "run_s": 0.0}
    ctx.max_queue_s = float(max_queue_hours) * 3600.0 if max_queue_hours else None
    ctx.background_jobs = []
    t_start = ctx.elapsed()
    level_req = int(level)
    plan = dict(level=level_req)                  # the EFFECTIVE level; the guard may lower it
    physics = bool(physics and level_req >= 3)       # physics advice is opt-in (--physics-advice)
    guard = bool(stop_if_short)                      # the step-down guard is opt-in
    missing = _prediction.rebuild_env_missing()
    short = None if missing or minutes is None else _prediction.rebuild_space_short(ctx.out_dir)
    fracs, cum = _pipeline_shares(physics, rebuild=not missing and not short)
    if minutes is not None and (missing or short) and not physics:
        print("pipeline: phase 2 rebuild cannot run (%s): its time goes to the layout search"
              % (", ".join(missing) + " not set" if missing else short))
    B = float(minutes) * 60.0 if minutes is not None else float("inf")
    pipe = PipelineSummary(
        total_steps=total_steps, B_s=(B if B != float("inf") else None),
        budget_source=("--minutes hard cap (job running time; queue wait and controller "
                       "work excluded)" if minutes is not None else "no time cap"),
        T_updated_s=None, phases=[], rebuild=None, extension=bool(extension),
        max_queue_s=ctx.max_queue_s, queue_cap_hit=False,
        shares={n: fracs[n] for n, _f in PIPELINE_FRACS})
    st = PipelineRun(ctx=ctx, t_start=t_start, level_req=level_req, plan=plan, physics=physics,
                     extension=extension, guard=guard, total_steps=total_steps, B=B, cum=cum,
                     pipe=pipe)
    print("pipeline O%d: hard cap %s of RUNNING time (queue wait not charged; queue cap %s)%s"
          % (level_req, _fmt_hms(B), _fmt_hms(ctx.max_queue_s or 0),
             "; phase limits (cumulative): " + ", ".join(
                 "%s %s" % (n, _fmt_hms(cum[n] * B)) for n, _f in PIPELINE_FRACS)
             if minutes is not None else ""))
    return st


def _pipeline_preview_points(ctx: Ctx) -> PointPreview:
    """The prediction points planned from the namelist alone (no job is submitted)."""
    try:
        spinup0 = predicted_spinup_min(_context._namelist_dt(ctx))
        _n, _rs, _re, dt_p, stepra_p, points_p, notes_p = _points.plan_predicted_points(ctx, spinup0)
        return PointPreview(ok=True, spinup_min=spinup0, dt=dt_p, stepra=stepra_p, points=points_p,
                            pkw=dict(run_start=_rs, run_end=_re,
                                     window_s=rebuilt_steps(stepra_p) * dt_p),
                            notes=list(notes_p),
                            wrapped=[dict(index=i, point=p) for i, p in enumerate(points_p)])
    except Exception as e:                                  # noqa: BLE001
        return PointPreview(ok=False, error="%s: %s" % (type(e).__name__, e), points=[],
                            wrapped=[])


def _pipeline_cold_estimate(st: PipelineRun, profile: dict, nodes_policy: str,
                            precision_min: Any) -> ColdEstimate:
    """Cold-start whole-run time and per-level tuning estimate (needs the step count); a failure
    part-way keeps what was computed and prints "pre-run estimate unavailable"."""
    ctx, pre = st.ctx, st.pre
    ce = ColdEstimate()
    if not st.total_steps:
        return ce
    try:
        n0 = choose_nodes(profile, ctx.max_nodes, nodes_policy)[0]
        cfg0, _why = _sweep_anchor(ctx, n0)
        ce.T_cold, s_n0, src = _prediction.cold_start_prediction(ctx, n0, cfg0, st.total_steps)
        ce.est_info = dict(s_per_step_n=s_n0)      # T_cold's compute even if the rest fails
        pplan0 = ce.pplan0 = plan_prediction_points(ctx, pre.wrapped, precision_min, ce.T_cold,
                                                    False, **pre.pkw)
        try:
            n_full = len(_candidates.enumerate_full_space(dataclasses.replace(
                ctx, nodes=n0, anchor_override=None, core_budget=None))[0])
        except Exception:                               # noqa: BLE001
            n_full = 0
        ce.n_full = n_full
        ce.est = level_estimates(s1=s_n0 * n0, steps_probe=ctx.steps,
                                 startup_s=ctx.startup_s, max_nodes=ctx.max_nodes,
                                 n_points=len(pplan0["selected"]), n_full=n_full,
                                 proxy=_node_search.joint_proxy_usable(ctx),
                                 proxy_node_cap=ctx.proxy_max_nodes)
        ce.est_info = dict(nodes=n0, s_per_step_n=s_n0, source=src, startup_s=ctx.startup_s,
                           prediction_points=len(pplan0["selected"]),
                           points_source=pplan0["source"], precision_min=precision_min)
    except Exception as e:                              # noqa: BLE001
        print("pipeline: pre-run estimate unavailable (%s: %s)" % (type(e).__name__, e))
    return ce


def _pipeline_pre_estimate(st: PipelineRun, profile: dict, nodes_policy: str,
                           precision_min: Any) -> None:
    """Print the per-level tuning-time estimate BEFORE any job (layout(level) + the case type's
    prediction points); with the guard on, step the level down to what fits the real run.
    Sets st.pre / st.pplan0 and pipe.level / n_points_est / n_full_est / T_cold_s."""
    ctx, pipe, plan = st.ctx, st.pipe, st.plan
    st.pre = _pipeline_preview_points(ctx)
    ce = _pipeline_cold_estimate(st, profile, nodes_policy, precision_min)
    est, T_cold, pplan0 = ce.est, ce.T_cold, ce.pplan0
    st.pplan0 = pplan0
    if est is not None and st.guard and plan["level"] >= 2:
        lv = highest_level_within(est, T_cold, plan["level"])
        if lv < plan["level"]:
            print("pipeline: 模擬本身只要約 %.0f 分鐘（冷啟動估計），比 O%d 的調校時間短；"
                  "自動降為 O%d（--stop-if-short / --minutes 啟用）%s"
                  % (T_cold / 60.0, plan["level"], lv,
                     "；O%d 用聯合代理挑選，不跑真實 N 節點探測" % lv
                     if lv in (1, 2) and _node_search.joint_proxy_usable(ctx) else ""))
            plan["level"] = lv
    pipe.level = LevelInfo(requested=st.level_req, effective=plan["level"],
                           estimate_s={("O%d" % k): v for k, v in (est or {}).items()},
                           estimate_info=ce.est_info, T_cold_s=T_cold, stepped_down=[])
    pipe.n_points_est = len(pplan0["selected"]) if pplan0 else 0
    pipe.n_full_est = ce.n_full
    if plan["level"] < st.level_req:
        pipe.level.stepped_down.append(StepDown(when="pre-run", to=plan["level"], T_real_s=T_cold))
    if est is not None:
        print("pipeline: 預估調校時間 " + format_level_estimates(est, plan["level"])
              + "; whole run ~%s (cold start); prediction points for this case at %s: %d "
                "(%s)" % (_fmt_hms(T_cold), precision_label(precision_min), len(pplan0["selected"]),
                          pplan0["source"]))
    pipe.T_cold_s = T_cold
    s_n0 = ce.est_info.get("s_per_step_n")
    pipe.T_cold_compute_s = s_n0 * st.total_steps if T_cold and s_n0 else None


def _phase0_submit_rebuild(st: PipelineRun, precision_min: Any, info: RebuildInfo) -> str:
    """Plan the ERA5-rebuildable points and submit rebuild + spin-up jobs for the ones this case
    needs at `precision_min` (worst case: nothing measured yet). Fills `info` as it goes, sets
    st.sub / st.prep_handle and ctx.background_jobs (they run beside phase 1). Returns the
    status; raises on failure."""
    ctx, pre = st.ctx, st.pre
    if not pre.ok:
        raise RuntimeError("point planning failed: %s" % pre.error)
    spinup_min, dt0, stepra0, points, notes = (pre.spinup_min, pre.dt, pre.stepra,
                                               pre.points, pre.notes)
    info.points = [p.at.isoformat() for p in points]
    info.spinup_min = spinup_min
    info.notes = list(notes)
    if not points:
        raise RuntimeError("no ERA5-rebuildable point planned (%s)" % "; ".join(notes))
    anchor0 = sweep.build_anchor(ctx.case_json, ctx.cores_per_node, 1, ctx.hardware_id)
    if anchor0 is None:
        for n in range(1, ctx.max_nodes + 1):
            anchor0, _why = _sweep_anchor(ctx, n)
            if anchor0:
                break
    case_dir = sweep._expand(ctx.case_json["prep_dir"])
    try:
        sample_pts0, _src = predict.read_wrfinput_sample_points(case_dir)
    except Exception:                                   # noqa: BLE001
        sample_pts0 = None
    wrapped0 = [dict(index=i, point=p) for i, p in enumerate(points)]
    st.sub = RebuildSubmission(wrapped=wrapped0, anchor=anchor0, case_dir=case_dir,
                               spinup_min=spinup_min, stepra=stepra0, dt=dt0,
                               sample_points=sample_pts0)
    pp0 = st.pplan0 if st.pplan0 is not None else plan_prediction_points(
        ctx, wrapped0, precision_min, st.pipe.T_cold_s, False, sample_pts0, **pre.pkw)
    chosen0 = pp0["selected"]
    info.prediction_points = len(chosen0)
    info.points_source = pp0["source"]
    if not chosen0:
        return "ok (no rebuild submitted: the case needs no prediction point)"
    st.prep_handle = _points.submit_rebuilt_points(
        ctx, [w["point"] for w in chosen0], anchor0, case_dir, ctx.namelist_path,
        ctx.case_json, os.path.join(ctx.out_dir, "rebuilt"), spinup_min, stepra0, dt0)
    ctx.background_jobs[:] = list(st.prep_handle["job_ids"])   # count as "ours RUNNING"
    info.rebuild_jobs = list(st.prep_handle["job_ids"])
    return "ok"


def _pipeline_phase0_plan(st: PipelineRun, precision_min: Any) -> None:
    """Phase 0 (_phase0_submit_rebuild); a failure is reported in the phase record and
    pipe.rebuild, not raised."""
    ctx = st.ctx
    m0 = _mark(ctx)
    info = RebuildInfo()
    try:
        p0_status = _phase0_submit_rebuild(st, precision_min, info)
    except Exception as e:                                  # noqa: BLE001 -- reported, not hidden
        p0_status = "failed: %s: %s" % (type(e).__name__, e)
        print("pipeline: phase 0 rebuild submission FAILED -- %s" % p0_status)
    st.pipe.phases.append(_phase_record("plan", m0, _mark(ctx), st.limit("plan"), p0_status,
                                        **info.to_dict()))
    st.pipe.rebuild = dataclasses.replace(info, status=p0_status)
    st.cut("plan")


def _pipeline_phase0_deferred(st: PipelineRun) -> None:
    """Default phase 0: nothing submitted -- the prediction points are planned from phase 1's
    measured time and rebuilt at its winner (_pipeline_phase2_rebuild_winner), so no rebuild
    job competes with the layout probes for the nodes."""
    ctx = st.ctx
    m0 = _mark(ctx)
    status = "deferred (points planned and rebuilt after the layout search, at its winner)"
    st.pipe.phases.append(_phase_record("plan", m0, _mark(ctx), st.limit("plan"), status))
    st.pipe.rebuild = RebuildInfo(status=status)
    st.cut("plan")


PREDICT_REBUILD_FORCING = "auto"   # the case's own forcing chain (GFS for haitang2017); era5 needs envs/ai
PREDICT_REBUILD_SPINUP_MIN = 60    # wtune predict-rebuild's default for --forcing auto


def _pipeline_phase2_rebuild_winner(st: PipelineRun, layouts: list[dict], pplan: dict | None,
                                    precision_min: Any) -> None:
    """Phase 2 (default, 2026-10-27): the phase-1 winner measured at as many rebuilt points as
    the precision needs, by `wtune predict-rebuild`'s procedure (rebuild + spin-up + measure per
    point, in waves within max_nodes) and combine(). haitang2017: one 40-step t0 window missed
    the run by 19-29 %, five points by 4 % (replay of its own 960 steps)."""
    from wtune import probe_plan as pp
    from wtune import rebuild_predict as rp
    from .points import REBUILT_MIN_STEPS
    ctx = st.ctx
    m2 = _mark(ctx)
    if st.cap_hit():
        status = "not run (%s)" % (STOP["reason"] or "max queue wait exceeded")
    elif not layouts:
        status = "not run (no layout)"
    elif _prediction.rebuild_env_missing():
        status = "not run (%s not set)" % ", ".join(_prediction.rebuild_env_missing())
    elif pplan is None or not pplan["selected"]:
        status = ("not run (the t0 anchor suffices for this case at %s: %s, band ±%.0f%%)"
                  % (precision_label(precision_min), pplan["method"] if pplan else "-",
                     (pplan["band"] if pplan else 0) * 100))
    else:
        lay0 = layouts[0]
        cfg = lay0["cfg"]
        try:
            case_dir = sweep._expand(ctx.case_json["prep_dir"])
            _nl, run_start, _end, _dt, stepra, points, _notes = rp.plan(
                case_dir, PREDICT_REBUILD_FORCING, PREDICT_REBUILD_SPINUP_MIN,
                len(pplan["selected"]), namelist_path=getattr(ctx, "namelist_path", None))
            best = dict(ranks=cfg["ranks"], omp=cfg.get("omp", 1), nx=cfg["nx"], ny=cfg["ny"],
                        numtiles=cfg.get("numtiles", 1))
            meas, job_ids = run_rebuild_points(
                ctx, points, best, lay0["nodes"], case_dir, os.path.join(ctx.out_dir, "rebuilt"),
                run_start, spinup_min=PREDICT_REBUILD_SPINUP_MIN, forcing=None,
                measure_steps=max(REBUILT_MIN_STEPS, int(stepra) + 1),
                point_s=_rebuild_point_s(lay0.get("median") or cfg.get("median"), _dt,
                                         max(REBUILT_MIN_STEPS, int(stepra) + 1)))
            (st.nl, st.run_start, st.run_end, st.dt, st.stepra, st.calibration,
             st.sample_points, st.case_dir) = _prediction._prediction_context(ctx)
            st.pc_ok = True
            res = pp.combine(meas, st.run_start, st.run_end, st.dt, st.stepra,
                             calibration=st.calibration, sample_points=st.sample_points,
                             case_dir=st.case_dir, nl=st.nl, ranks=best["ranks"],
                             nodes=lay0["nodes"])
            st.rebuild_pred = dict(result=res, layout=lay0, job_ids=job_ids,
                                   points=[p.at.isoformat() for p in points])
            n_ok = len({m.at for m in meas})
            st.rebuild_pred["points"] = [m.at.isoformat() for m in meas]
            status = ("ok (%d rebuilt point(s) at the winner, jobs %s)" % (n_ok, job_ids)
                      if n_ok == len(points) else
                      "degraded (%d of %d rebuilt point(s) measured at the winner, jobs %s)"
                      % (n_ok, len(points), job_ids))
        except Exception as e:                              # noqa: BLE001 -- reported
            status = "failed: %s: %s" % (type(e).__name__, e)
            print("pipeline: phase 2 rebuild at the winner FAILED -- %s" % status)
    st.pipe.phases.append(_phase_record("predict", m2, _mark(ctx), st.limit("predict"), status))
    st.cut("predict")


def _pipeline_on_sweep(st: PipelineRun, rows: list[dict], _sw: dict) -> None:
    """Phase-1 sweep callback, informational: the whole run implied by the best measured s/step
    (an upper bound: anchor config) -> pipe.T_updated_s, printed. B is NOT changed."""
    if not st.total_steps:
        return
    pipe = st.pipe
    s_best = min(r["median"] for r in rows)
    pipe.T_updated_s = st.total_steps * s_best + st.ctx.startup_s
    pipe.T_updated_compute_s = st.total_steps * s_best
    print("pipeline: best measured %.4f s/step -> whole run ~%s" % (s_best,
                                                                   _fmt_hms(pipe.T_updated_s)))


def _pipeline_on_wave(st: PipelineRun, meas: dict) -> None:
    """Short-run guard after a measurement wave: when the whole run is shorter than the tuning
    time of the planned level, step down to the highest level that fits (never below O1; the
    sweep reads the level live). Records the step-down and pipe.short_run; returns None."""
    ctx, pipe, plan, total_steps = st.ctx, st.pipe, st.plan, st.total_steps
    if not st.guard or not total_steps or not meas or plan["level"] < 2:
        return None
    n_best, vals = min(((n, statistics.median(v)) for n, v in meas.items() if v),
                       key=lambda kv: kv[1])
    T = total_steps * vals + ctx.startup_s
    est2 = level_estimates(s1=vals * n_best, steps_probe=ctx.steps, startup_s=ctx.startup_s,
                           max_nodes=ctx.max_nodes, n_points=if_set(pipe.n_points_est, 0),
                           n_full=if_set(pipe.n_full_est, 0), proxy=_node_search.joint_proxy_usable(ctx),
                           proxy_node_cap=ctx.proxy_max_nodes)
    lv = highest_level_within(est2, T, plan["level"])
    if lv >= plan["level"]:
        return None
    msg = ("模擬本身只要約 %.0f 分鐘，比 O%d 的調校時間（約 %.0f 分鐘）短；已自動降為 O%d，"
           "建議直接以目前最佳設定執行" % (T / 60.0, plan["level"],
                                      est2[plan["level"]] / 60.0, lv))
    pipe.level.stepped_down.append(StepDown(when="after first wave", frm=plan["level"], to=lv,
                                            T_real_s=T, message=msg))
    plan["level"] = lv
    pipe.level.effective = lv
    pipe.short_run = ShortRun(message=msg, real_run_s=T)
    print("pipeline: %s" % msg)
    return None                      # the sweep itself goes on; later stages read plan


def _pipeline_adopt_phase1_startup(ctx: Ctx, collect: dict) -> None:
    """Adopt the median startup measured by the finalists' round0 (on their ctx copies): the
    default 120 s/run made phase 2 look 2-4x too expensive. No-op when none was measured."""
    meas_ctx = [o.ctx for o in collect.get("outcomes", {}).values()
                if o.ctx is not None
                and str(getattr(o.ctx, "startup_source", "")).startswith("measured")]
    meas_su = [c.startup_s for c in meas_ctx]
    if meas_su:
        ctx.startup_s = statistics.median(meas_su)
        ctx.startup_source = "measured in phase 1 (median of %d finalist(s))" % len(meas_su)
        ctx.startup_runs = sum(getattr(c, "startup_runs", None) or 0 for c in meas_ctx) or None


def _pipeline_phase1_layout(st: PipelineRun, profile: dict, objective: str,
                            deadline_s: float | None, min_efficiency: float | None,
                            k_finalists: int, nodes_policy: str, sweep_budget_frac: float | None,
                            core_options: list | None) -> tuple[dict, list[dict]]:
    """Phase 1: run_node_search under the layout limit. Returns (report -- a pipeline_aborted
    stub when it stopped before any result --, the top-2 layouts for phase 2)."""
    ctx, pipe = st.ctx, st.pipe
    m1 = _mark(ctx)
    # no prediction point to measure: the prediction share goes to the layout search
    no_points = st.pplan0 is not None and not st.pplan0["selected"] and st.prep_handle is None
    end = st.limit("predict" if no_points else "layout")
    ctx1 = dataclasses.replace(ctx, budget_s=max(end, m1.run + 1.0))
    collect: dict = {}
    report = None
    try:
        report = _node_search.run_node_search(
            ctx1, profile, objective, deadline_s,
            min_efficiency if min_efficiency is not None else wtune.nodes.DEFAULT_MIN_EFFICIENCY,
            k_finalists, st.total_steps, nodes_policy=nodes_policy,
            sweep_budget_frac=(sweep_budget_frac if sweep_budget_frac is not None
                               else DEFAULT_SWEEP_BUDGET_FRAC),
            core_options=core_options,
            on_sweep=lambda rows, _sw: _pipeline_on_sweep(st, rows, _sw), collect=collect,
            on_wave=lambda meas: _pipeline_on_wave(st, meas), plan=st.plan)
        ph1 = "ok" if not is_set(pipe.short_run) else "ok (stepped down: short run)"
    except QueueCapExceeded as e:
        ph1 = "stopped: %s" % e
        print("pipeline: phase 1 STOPPED -- %s" % e)
    pipe.phases.append(_phase_record(
        "layout", m1, _mark(ctx), end, ph1,
        **(dict(nodes=report["nodes"], cores_per_node=report.get("cores_used_per_node"))
           if report else {})))
    st.cut("layout")
    if report is None:                         # aborted before any layout result existed
        report = dict(case=ctx.case, pipeline_aborted=True, nodes=None, job_ids=[],
                      cancelled_job_ids=[])
    layouts = pipeline_layouts(collect, 2) if collect.get("outcomes") else []
    _pipeline_adopt_phase1_startup(ctx, collect)
    return report, layouts


def _pipeline_recheck_nodes(st: PipelineRun, recheck: Callable | None) -> list[dict]:
    """Node health re-check before phase 2 (controller time). Returns the checks so far; a
    failed re-check is printed and the earlier exclusions are kept."""
    ctx = st.ctx
    health_checks = [ctx.node_health] if ctx.node_health else []
    if recheck is not None and not st.cap_hit():
        try:                  # nodes can drop out during phase 1: probe again (controller time)
            h = recheck("before phase 2")
            health_checks.append(h)
            ctx.node_health = h
            ctx.max_nodes = h["max_nodes_effective"]
        except Exception as e:                              # noqa: BLE001
            print("pipeline: node re-check failed (%s: %s); keeping the earlier exclusions"
                  % (type(e).__name__, e))
    return health_checks


def _pipeline_replan_points(st: PipelineRun, precision_min: Any) -> tuple[dict | None, Any]:
    """Re-plan the prediction points with phase 1's measured whole-run time. Returns (point plan
    or None, the time it was planned for); records pipe.prediction_plan and cancels surplus
    rebuild jobs."""
    pipe = st.pipe
    T_dec = pipe.T_updated_s or if_set(pipe.T_cold_s)
    pipe.level.effective = st.plan["level"]
    wrapped = st.sub.wrapped if st.sub else (st.pre.wrapped if st.pre and st.pre.ok else None)
    pplan = (plan_prediction_points(st.ctx, wrapped, precision_min, T_dec,
                                    pipe.T_updated_s is not None,
                                    st.sub.sample_points if st.sub else None, **st.pre.pkw)
             if (wrapped is not None and T_dec) else st.pplan0)
    if pplan is not None:
        pipe.prediction_plan = PredictionPlanRecord(
            points=[w["index"] for w in pplan["selected"]], band=pplan["band"],
            method=pplan["method"], source=pplan["source"], T_est_s=T_dec,
            precision_min=precision_min, measured=pipe.T_updated_s is not None)
    if st.prep_handle and pplan is not None:
        _pipeline_cancel_surplus(st, pplan, precision_min)
    return pplan, T_dec


def _pipeline_cancel_surplus(st: PipelineRun, pplan: dict, precision_min: Any) -> None:
    """Fewer points planned than submitted in phase 0: scancel the surplus rebuild jobs, narrow
    st.prep_handle (None when nothing is kept) and ctx.background_jobs."""
    ctx, prep_handle = st.ctx, st.prep_handle
    have = len(prep_handle["points"])
    sel_at = {w["point"].at for w in pplan["selected"]}
    keep = [i for i, p in enumerate(prep_handle["points"]) if p.at in sel_at]
    if not keep and pplan["selected"]:           # the plan wants other points than submitted
        keep = list(range(have))
    if len(keep) >= have:
        return
    drop = [j for i, j in enumerate(prep_handle["job_ids"]) if i not in keep]
    print("pipeline: the case needs %d prediction point(s) at %s: cancelling %d "
          "surplus rebuild job(s) %s" % (len(keep), precision_label(precision_min), len(drop),
                                         ",".join(drop)))
    ctx.scancel_runner(drop)
    ctx.cancelled_jobs.extend(drop)
    st.pipe.prediction_plan.rebuild_cancelled = drop
    if keep:
        st.prep_handle = dict(prep_handle, points=[prep_handle["points"][i] for i in keep],
                              done=[prep_handle["done"][i] for i in keep],
                              job_ids=[prep_handle["job_ids"][i] for i in keep])
        ctx.background_jobs[:] = list(st.prep_handle["job_ids"])
    else:
        st.prep_handle = None
        ctx.background_jobs[:] = []


def _phase2_collect_rebuilt(st: PipelineRun, layouts: list[dict], pplan: dict | None,
                            precision_min: Any) -> str | None:
    """Whether phase 2 can run and the rebuilt points (collected whatever phase 2 can afford:
    phase 3 needs them too). Returns the phase-2 status when already decided, else None."""
    ctx = st.ctx
    if st.cap_hit():
        return "not run (%s)" % (STOP["reason"] or "max queue wait exceeded")
    if pplan is not None and not pplan["selected"] and layouts:
        return ("not run (the t0 anchor suffices for this case at %s: %s, band ±%.0f%%)"
                % (precision_label(precision_min), pplan["method"], pplan["band"] * 100))
    if st.prep_handle is None or not layouts:
        return "not run (no rebuilt points)" if st.prep_handle is None else "not run (no layout)"
    ph2 = None
    c2 = dataclasses.replace(ctx, budget_s=st.limit("physics"))
    try:
        st.rebuilt = _points.finish_rebuilt_points(c2, st.prep_handle)
    except QueueCapExceeded as e:
        ph2 = "stopped: %s" % e
    except Exception as e:                              # noqa: BLE001
        ph2 = "failed: rebuild jobs: %s: %s" % (type(e).__name__, e)
    ctx.background_jobs[:] = []
    return ph2


def _phase2_groups(lays_sel: list[dict], frames_wanted: bool) -> list[tuple]:
    """[(n, c, anchor, others, s_by, lays)] per (N, c) of the selected layouts; with
    `frames_wanted` the configs keep window frames (phase 3's twin). Sets each layout's pid."""
    groups: dict = {}
    for lay in lays_sel:
        groups.setdefault((lay["nodes"], lay["cores"]), []).append(lay)
    out = []
    for (n, c), lays in groups.items():
        wf = dict(window_frame=WINDOW_FRAME_LEVEL) if frames_wanted else {}
        first, anchor, others = lays[0], dict(lays[0]["cfg"], **wf), []
        for i, lay in enumerate(lays[1:], start=1):
            lay["pid"] = "L%d" % i
            others.append(dict(lay["cfg"], id=lay["pid"], config_source="generated", **wf))
        first["pid"] = "ANCHOR"
        if len(lays) == 1 and not frames_wanted:
            # winner-only: one fused wrf.exe; the ANCHOR placeholder names the spin-up layout
            first["pid"] = "W0"
            others = [dict(first["cfg"], id="W0", config_source="generated")]
        s_by = {"ANCHOR": first["median"], first["pid"]: first["median"],
                **{l["pid"]: l["median"] for l in lays[1:]}}
        out.append((n, c, anchor, others, s_by, lays))
    return out


def _phase2_gctx(st: PipelineRun, n: int, c: int) -> Ctx:
    """The ctx copy one phase-2 (N, c) group is measured with (phase-2 limit, slots for N)."""
    ctx = st.ctx
    return dataclasses.replace(ctx, nodes=n, core_budget=c, budget_s=st.limit("predict"),
                               stage1_max_running=compute_slots(ctx.max_nodes, n))


def _phase2_choose_plan(st: PipelineRun, layouts: list[dict], frames_wanted: bool, left2: float,
                        winner_only: bool) -> tuple[list[dict], tuple | None]:
    """The largest phase2_plans plan whose summed estimate fits `left2` (groups run one after
    another). Returns (tried [{plan, estimate_s}], (label, layouts, points, estimate) or None)."""
    tried = []
    short = st.pipe.short_run if is_set(st.pipe.short_run) else None
    for label, lays_sel, pts in phase2_plans(layouts, st.rebuilt, st.sample_points,
                                             winner_only=winner_only):
        est = 0.0
        for n, c, anchor, others, s_by, _l in _phase2_groups(lays_sel, frames_wanted):
            est += fit_stage2(_phase2_gctx(st, n, c), others, anchor, pts, s_by, st.nl,
                              left_s=None, mode=choose_point_mode(others, anchor))[1]
        tried.append(dict(plan=label, estimate_s=est))
        if est <= left2 and not (short and est >= short.real_run_s):
            return tried, (label, lays_sel, pts, est)
    return tried, None


def _phase2_measure_groups(st: PipelineRun, lays_sel: list[dict], pts: list[dict],
                           frames_wanted: bool) -> list[str]:
    """Measure the chosen plan one (N, c) group at a time (one running-budget wait at a time; the
    point jobs of a group run in parallel). Extends st.ranked_all / st.frame_runs; returns the
    per-group skip / failure notes."""
    ctx = st.ctx
    skipped2 = []
    for n, c, anchor, others, s_by, lays in _phase2_groups(lays_sel, frames_wanted):
        gctx = _phase2_gctx(st, n, c)
        pmode = choose_point_mode(others, anchor)
        top, _est, skip = fit_stage2(gctx, others, anchor, pts, s_by, st.nl,
                                     left_s=st.limit("predict") - ctx.elapsed(), mode=pmode)
        if skip:
            skipped2.append("N=%d c=%d: %s" % (n, c, skip))
            continue
        try:
            meas, _gates, jobs = _points.measure_candidates_at_points(gctx, top, pts, anchor,
                                                                      s_by_cid=s_by, mode=pmode)
            cfgs = dict({o["id"]: o for o in top}, ANCHOR=anchor)
            ranked = _points.rank_by_predicted_total(
                meas, st.run_start, st.run_end, st.dt, st.stepra, st.calibration,
                st.sample_points, st.case_dir, st.nl,
                {cid: cf["ranks"] for cid, cf in cfgs.items()}, n)
        except QueueCapExceeded as e:
            skipped2.append("N=%d c=%d: stopped: %s" % (n, c, e))
            break
        except Exception as e:                      # noqa: BLE001
            skipped2.append("N=%d c=%d: failed: %s: %s" % (n, c, type(e).__name__, e))
            continue
        by_pid = {lay["pid"]: lay for lay in lays}
        st.frame_runs.extend((rp, cid, jid) for cid in ["ANCHOR", ANCHOR_END_ID] + [
            o["id"] for o in top] for rp, jid in zip(pts, jobs))
        st.ranked_all.extend(
            MeasuredLayout(layout=by_pid[e["cid"]], nodes=n, cores=c, cid=e["cid"],
                           pred_total_s=e["pred_total_s"], pred_low_s=e["pred_low_s"],
                           pred_high_s=e["pred_high_s"], result=e["result"],
                           meas=meas.get(e["cid"]), jobs=list(jobs),
                           points=[rp["index"] for rp in pts]) for e in ranked if e["cid"] in by_pid)
    return skipped2


def _phase2_measure(st: PipelineRun, layouts: list[dict]) -> str:
    """Phase 2 proper: the largest plan that fits, measured and ranked by predicted whole-run
    time. Returns the status; sets pipe.predict_plan and st.winner."""
    ctx, pipe = st.ctx, st.pipe
    frames_wanted = phase2_writes_frames(st)   # phase 3 gates on these frames; they cost I/O
    left2 = st.limit("predict") - ctx.elapsed()
    tried, chosen_plan = _phase2_choose_plan(
        st, layouts, frames_wanted, left2,
        winner_only=(st.plan["level"] <= 2 or not st.physics))
    pipe.predict_plan = Phase2Plan(left_s=left2, tried=tried, startup_s=ctx.startup_s,
                                   chosen=chosen_plan[0] if chosen_plan else None,
                                   degraded=bool(chosen_plan and tried and len(tried) > 1))
    if chosen_plan is None:
        ph2 = NOT_RUN_BUDGET + ": " + (
            "the smallest plan (%s) needs ~%s of running time, %s left of the phase-2 share"
            % (tried[-1]["plan"], _fmt_hms(tried[-1]["estimate_s"]), _fmt_hms(max(0.0, left2))))
        short = pipe.short_run if is_set(pipe.short_run) else None
        if short and tried[-1]["estimate_s"] >= short.real_run_s:
            ph2 = NOT_RUN_BUDGET + (": short run: even the winner alone needs ~%s, more than "
                                    "the real run (~%s)" % (_fmt_hms(tried[-1]["estimate_s"]),
                                                            _fmt_hms(short.real_run_s)))
        print("pipeline: phase 2 %s" % ph2)
        return ph2
    label, lays_sel, pts, est = chosen_plan
    degraded = pipe.predict_plan.degraded
    if degraded:
        print("pipeline: phase 2 DEGRADED to %s (estimate %s of %s left): %s"
              % (label, _fmt_hms(est), _fmt_hms(left2),
                 "; ".join("%s ~%s" % (t["plan"], _fmt_hms(t["estimate_s"]))
                           for t in tried[:-1] if t["plan"] != label)))
    skipped2 = _phase2_measure_groups(st, lays_sel, pts, frames_wanted)
    if st.ranked_all:
        st.ranked_all.sort(key=lambda e: e.pred_total_s)
        st.winner = st.ranked_all[0]
        return ("ok" if not skipped2 and not degraded
                else "partial: " + "; ".join((["degraded to %s" % label] if degraded else [])
                                             + skipped2))
    budget_msgs = ("exceed", "needs an expected", "left")
    return (NOT_RUN_BUDGET if skipped2 and all(any(m in x for m in budget_msgs)
                                               for x in skipped2)
            else "failed: " + "; ".join(skipped2))


def _pipeline_phase2_predict(st: PipelineRun, layouts: list[dict], pplan: dict | None,
                             precision_min: Any) -> None:
    """Phase 2: rebuilt points -> prediction context -> _phase2_measure; records the phase."""
    ctx = st.ctx
    m2 = _mark(ctx)
    ph2 = _phase2_collect_rebuilt(st, layouts, pplan, precision_min)
    if st.rebuilt is not None:
        try:
            (st.nl, st.run_start, st.run_end, st.dt, st.stepra, st.calibration, st.sample_points,
             st.case_dir) = _prediction._prediction_context(ctx)
            st.pc_ok = True
        except Exception as e:                              # noqa: BLE001
            ph2 = "failed: prediction context: %s: %s" % (type(e).__name__, e)
    if ph2 is None and ctx.elapsed() >= st.limit("predict"):
        ph2 = NOT_RUN_BUDGET + ": phase-2 limit reached"
    if ph2 is None:
        ph2 = _phase2_measure(st, layouts)
    st.pipe.phases.append(_phase_record("predict", m2, _mark(ctx), st.limit("predict"), ph2))
    st.cut("predict")


def _pipeline_adopt_winner(ctx: Ctx, report: dict, winner: MeasuredLayout, cfg: dict,
                           pred_s: float) -> None:
    """Phase 2 overturned phase 1's pick (rank flip across points): switch the report's nodes,
    cores, parameter table (STRICT-guarded) and best to it."""
    pt = local_param_table(cfg, winner.nodes)
    _assert_report_is_strict_only(ctx, pt)
    report["nodes"], report["nodes_note"] = (winner.nodes,
                                             "phase 2 predicted-whole-run winner")
    report["cores_used_per_node"] = winner.cores
    report["param_table"] = pt
    report["best"] = dict(id=winner.layout.get("cid"), ranks=cfg.get("ranks"),
                          omp=cfg.get("omp"), nproc_x=cfg.get("nx"),
                          nproc_y=cfg.get("ny"), numtiles=cfg.get("numtiles"),
                          dist=cfg.get("dist"), predicted_total_s=pred_s)


def _prediction_from_winner(st: PipelineRun, report: dict, layouts: list[dict],
                            pplan: dict | None, band_eff: float,
                            io_run: dict | None) -> WholeRunPrediction:
    """Phase 2's winner as the whole-run prediction (combine() + run-length I/O, the measured
    init); also sets report["full_run_prediction"] and adopts the winner when it overturns
    phase 1's pick."""
    ctx, winner = st.ctx, st.winner
    cfg = winner.layout["cfg"]
    pred_s = winner.pred_total_s
    predict_plan = st.pipe.predict_plan
    pred = WholeRunPrediction(
        layout_id=winner.layout.get("cid"), nodes=winner.nodes, cores_per_node=winner.cores,
        config={k: cfg.get(k) for k in _PRED_CONFIG_KEYS},
        predicted_total_s=pred_s, predicted_total_hms=_fmt_hms(pred_s),
        predicted_range_s=[winner.pred_low_s, winner.pred_high_s],
        time_suggestion=_fmt_sbatch_time(
            time_suggestion_s(pred_s, pplan["band"] if pplan else wtune.machine.measured("cold_start_band"))),
        points_used=len(winner.points or []),
        degraded=bool(is_set(predict_plan) and predict_plan.degraded),
        candidates=[dict(nodes=e.nodes, cores=e.cores, pid=e.cid,
                         predicted_total_s=e.pred_total_s,
                         config={k: e.layout["cfg"].get(k)
                                 for k in ("ranks", "omp", "nx", "ny", "numtiles")})
                    for e in st.ranked_all])
    init_w = float((winner.result or {}).get("init_s") or 0.0)
    init_s, init_src = prediction_init(ctx, init_w, "wtune.predict.predict_init")
    pred.add_run_length_io(io_run, pred_s - init_w, init_s, band_eff, init_src)
    pred_s = pred.predicted_total_s
    if (winner.nodes, winner.cores, sweep._identity(cfg)) != (
            report["nodes"], report.get("cores_used_per_node"),
            sweep._identity(layouts[0]["cfg"])):
        _pipeline_adopt_winner(ctx, report, winner, cfg, pred_s)
    report["full_run_prediction"] = "\n".join(
        [_predicted_winner_text(dict(cid=pred.layout_id, result=winner.result)),
         "whole run: " + pred.breakdown
         + ("  [init: %s]" % pred.init_source if pred.init_source else "")]
        + (["NOTE: " + pred.segment_note] if pred.segment_note else []))
    return pred


def _prediction_from_rebuild(st: PipelineRun, report: dict, band_eff: float,
                             io_run: dict | None) -> WholeRunPrediction:
    """Phase 2 rebuild-winner: combine()'s compute + init at the rebuilt points, plus run-length
    I/O; the range is combine()'s, shifted by the I/O. Also sets report["full_run_prediction"]."""
    rp, lay = st.rebuild_pred, st.rebuild_pred["layout"]
    res = rp["result"]
    T = float(res["total_s"])
    pred = WholeRunPrediction(
        layout_id=lay.get("cid"), nodes=lay["nodes"], cores_per_node=lay["cores"],
        config={k: lay["cfg"].get(k) for k in _PRED_CONFIG_KEYS},
        predicted_total_s=T, predicted_total_hms=_fmt_hms(T),
        predicted_range_s=[float(res["range_low_s"]), float(res["range_high_s"])],
        time_suggestion=_fmt_sbatch_time(time_suggestion_s(T, band_eff)),
        points_used=len(rp["points"]), degraded=False,
        source="phase 2: the phase-1 winner at %d rebuilt point(s) (combine())" % len(rp["points"]))
    pred.add_run_length_io(io_run, float(res["compute_total_s"]), float(res["init_s"]), band_eff,
                           res.get("init_note"))
    report["full_run_prediction"] = "\n".join(
        ["=== full-run prediction (phase 2: %s at %d rebuilt point(s): %s) ==="
         % (lay.get("cid"), len(rp["points"]), ", ".join(rp["points"])),
         "TOTAL: %s (range %s - %s)" % (_fmt_hms(pred.predicted_total_s),
                                       _fmt_hms(pred.predicted_range_s[0]),
                                       _fmt_hms(pred.predicted_range_s[1])),
         "whole run: " + (pred.breakdown if is_set(pred.breakdown) and pred.breakdown else
                          "compute %.0fs + init %.0fs" % (pred.compute_s, pred.init_s))])
    return pred


def _prediction_from_t0_anchor(st: PipelineRun, lay0: dict, T_dec: float, band_eff: float,
                               io_run: dict | None) -> WholeRunPrediction:
    """No rebuilt point needed: phase 1's t0 anchor s/step x steps + startup (+ run-length I/O),
    its range the band around it."""
    ctx, pipe = st.ctx, st.pipe
    T = T_dec
    pred = WholeRunPrediction(
        layout_id=lay0.get("cid"), nodes=lay0["nodes"], cores_per_node=lay0["cores"],
        config={k: lay0["cfg"].get(k) for k in _PRED_CONFIG_KEYS},
        predicted_total_s=T, predicted_total_hms=_fmt_hms(T),
        predicted_range_s=[T * (1 - band_eff), T * (1 + band_eff)],
        time_suggestion=_fmt_sbatch_time(time_suggestion_s(T, band_eff)),
        points_used=0, degraded=False, band=band_eff, error_min=band_eff * T / 60.0,
        source="phase-1 t0 anchor s/step x steps + startup (no rebuilt point)")
    comp = (if_set(pipe.T_updated_compute_s) if pipe.T_updated_s
            else if_set(pipe.T_cold_compute_s))
    if comp is None:
        comp = T - ctx.startup_s
    init_s, init_src = prediction_init(ctx, ctx.startup_s, ctx.startup_source)
    pred.add_run_length_io(io_run, comp, init_s, band_eff, init_src)
    T = pred.predicted_total_s                     # this path's range is the band around T
    pred.predicted_range_s = [T * (1 - band_eff), T * (1 + band_eff)]
    return pred


def _pipeline_prediction(st: PipelineRun, report: dict, layouts: list[dict], pplan: dict | None,
                         T_dec: Any) -> tuple[WholeRunPrediction | None, float, dict | None]:
    """The whole-run prediction after phase 2 -- phase 2's winner, else (no rebuilt point
    needed) the t0 anchor, else None -- with run-length I/O added (combine() gives compute +
    init only). Returns (pred, band, run-length I/O); sets pipe.prediction."""
    ctx, pipe, winner = st.ctx, st.pipe, st.winner
    pred = None
    band_eff = pplan["band"] if pplan else wtune.machine.measured("cold_start_band")
    io_run = (_prediction.pipeline_run_length_io(getattr(ctx, "namelist_path", None))
              if (winner is not None or T_dec or pipe.T_updated_s) else None)
    if winner is not None:
        pred = _prediction_from_winner(st, report, layouts, pplan, band_eff, io_run)
    elif st.rebuild_pred is not None:
        pred = _prediction_from_rebuild(st, report, band_eff, io_run)
    if pred is not None:
        pred.band = band_eff
        pred.error_min = band_eff * pred.predicted_total_s / 60.0
    elif pplan is not None and not pplan["selected"] and layouts and T_dec:
        pred = _prediction_from_t0_anchor(st, layouts[0], T_dec, band_eff, io_run)
    pipe.prediction = pred
    return pred, band_eff, io_run


def _pipeline_finish(st: PipelineRun, report: dict, pred: WholeRunPrediction | None,
                     pplan: dict | None, band_eff: float, io_run: dict | None,
                     health_checks: list[dict], keep_frames: bool) -> dict:
    """Delete the window frames (unless --keep-frames; the gate's json/csv stay), attach the
    per-phase job timing, fill the pipeline totals and set report["pipeline"] (to_dict), the
    node health and the job / time fields. Returns `report`."""
    ctx, pipe, B, t_start = st.ctx, st.pipe, st.B, st.t_start
    pipe.frames_kept = bool(keep_frames)
    pipe.frames_deleted = 0 if keep_frames else _delete_window_frames(ctx, st.frame_runs)
    _attach_phase_timing(ctx, pipe.phases, st.slices)

    best = report.get("best") or {}
    if not pred and not pipe.T_updated_s and st.total_steps and best.get("median_s_per_step"):
        # the joint proxy search has no node-sweep callback: the winner's proxy s/step instead
        # (phase 2 needs rebuilt points; with no WPS the report said "predicted real run 0:00")
        pipe.T_updated_compute_s = st.total_steps * best["median_s_per_step"]
        pipe.T_updated_s = pipe.T_updated_compute_s + ctx.startup_s
    running = ctx.elapsed() - t_start
    predicted_real = (pred.predicted_total_s if pred else
                      (if_set(pipe.T_updated_compute_s, pipe.T_updated_s - ctx.startup_s)
                       + ctx.startup_s + (io_run["total_s"] if io_run else 0.0))
                      if pipe.T_updated_s else None)
    pipe.time_suggestion = (_fmt_sbatch_time(time_suggestion_s(predicted_real, band_eff))
                            if predicted_real else None)
    pipe.prediction_band = PredictionBand(
        method=(pplan or {}).get("method", "cold-start predict"), band=band_eff,
        error_min=(band_eff * predicted_real / 60.0 if predicted_real else None))
    pipe.level.actual_running_s = ctx.elapsed() - t_start
    pipe.level.effective = st.plan["level"]
    pipe.total_running_s = running
    pipe.total_wall_s = ctx.wall_elapsed()
    pipe.jobs_busy_s = _jobs.busy_s(ctx, ctx.owned_jobs)
    pipe.total_queue_wait_s = ctx.queue_wait_s
    pipe.queue_cap_hit = st.cap_hit()
    pipe.total_controller_s = max(0.0, ctx.wall_elapsed() - running - ctx.queue_wait_s)
    pipe.predicted_real_s = predicted_real
    joint = (report.get("node_selection") or {}).get("note") == "joint proxy search"
    pipe.predicted_real_source = (("phase 2 combine()" if pred else
                                   "winner's joint proxy s/step x steps" if joint else
                                   "phase-1 anchor s/step x steps")
                                  + (" + run-length I/O" if io_run else ""))
    pipe.hard_deadline_s = (B if B != float("inf") else None)
    pipe.over_deadline_s = (max(0.0, running - B) if B != float("inf") else 0.0)
    report["pipeline"] = pipe.to_dict()
    if ctx.node_health:
        report["node_health"] = dict(ctx.node_health, checks=[
            {k: v for k, v in c.items() if k != "nodes"} for c in health_checks])
    report["job_ids"] = list(ctx.owned_jobs)
    report["cancelled_job_ids"] = list(ctx.cancelled_jobs)
    report["elapsed_minutes"] = running / 60.0
    report["budget_minutes"] = (B / 60.0 if B != float("inf") else None)
    return report


def run_level0(ctx: Ctx, profile: dict, total_steps: int | None,
               nodes_policy: str = "knee") -> dict:
    """-O0, no jobs: the proxy's node count (choose_nodes; the first N with a legal layout), the
    hybrid anchor there on all the cores, or on fewer when the grid cannot take that many ranks
    (_sweep_anchor_fitting, as the node sweep), and wtune.predict's cold-start whole-run time
    (+-25%)."""
    n0, note, _warn = choose_nodes(profile, ctx.max_nodes, nodes_policy)
    cfg, cores, why = None, None, None
    for n in [n0] + [m for m in range(1, ctx.max_nodes + 1) if m != n0]:
        cfg, cores, why = _sweep_anchor_fitting(ctx, n)
        if cfg:
            n0 = n
            break
    if cfg is None:
        raise SystemExit("O0: no legal layout at any node count (%s)" % why)
    band = wtune.machine.measured("cold_start_band")
    T, s_n, src = (_prediction.cold_start_prediction(ctx, n0, cfg, total_steps) if total_steps
                   else (None, None, "run length unknown"))
    io_run = _prediction.pipeline_run_length_io(ctx.namelist_path) if T else None
    if T:
        T += io_run["total_s"] if io_run else 0.0
    pt = local_param_table(cfg, n0)
    _assert_report_is_strict_only(ctx, pt)
    pipe = PipelineSummary(
        total_steps=total_steps, B_s=None, budget_source="no jobs", T_updated_s=None, phases=[],
        rebuild=None, extension=False, queue_cap_hit=False,
        level=LevelInfo(requested=0, effective=0, estimate_s={"O0": 0.0}, stepped_down=[],
                        actual_running_s=0.0),
        prediction=WholeRunPrediction(
            layout_id="ANCHOR", nodes=n0, cores_per_node=cores,
            config={k: cfg.get(k) for k in _PRED_CONFIG_KEYS},
            predicted_total_s=T, predicted_total_hms=_fmt_hms(T) if T else None,
            predicted_range_s=[T * (1 - band), T * (1 + band)] if T else None,
            time_suggestion=(_fmt_sbatch_time(time_suggestion_s(T, band)) if T else None),
            band=band, error_min=(band * T / 60.0 if T else None), points_used=0,
            degraded=False, source=src + (" + run-length I/O" if io_run else "")),
        time_suggestion=(_fmt_sbatch_time(time_suggestion_s(T, band)) if T else None),
        prediction_band=PredictionBand(method="cold-start predict", band=band,
                                       error_min=(band * T / 60.0 if T else None)),
        total_running_s=0.0, total_wall_s=ctx.wall_elapsed(), total_queue_wait_s=0.0,
        total_controller_s=ctx.wall_elapsed(), predicted_real_s=T, hard_deadline_s=None,
        over_deadline_s=0.0, physics_advice=None)
    if T:
        # the same fields as the -O1..3 predictions; T already holds the I/O, ranges stay symmetric
        p0 = pipe.prediction
        io_s = io_run["total_s"] if io_run else 0.0
        comp = s_n * total_steps
        p0.add_run_length_io(io_run, comp, T - io_s - comp, band,
                             ctx.startup_source if src.startswith("ledger") else src)
        p0.predicted_range_s = [T * (1 - band), T * (1 + band)]
    print("O0: no jobs -- nodes=%d (%s), layout %s, cold-start whole run %s ±%.0f%% (%s)"
          % (n0, note, {k: cfg.get(k) for k in ("ranks", "omp", "nx", "ny")},
             _fmt_hms(T) if T else "unknown", band * 100, src))
    return dict(case=ctx.case, pipeline_only=True, nodes=n0,
                nodes_note="O0: proxy node count, no measurement", param_table=pt,
                pipeline=pipe.to_dict(), cores_per_node=ctx.cores_per_node,
                cores_used_per_node=cores, job_ids=[],
                cancelled_job_ids=[],
                best=dict(id="ANCHOR", **{k: cfg.get(k) for k in ("ranks", "omp", "numtiles",
                                                                   "dist")},
                          nproc_x=cfg.get("nx"), nproc_y=cfg.get("ny")))
