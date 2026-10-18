"""The run controller: segments in order, each attempt checked; a failed one is followed by an
auto-resume from its latest valid restart (infrastructure failures, --resume) or one CFL
rescue per run (--cfl-rescue, PHYSICS-GRAY) -- anything else stops, never a silent retry."""
from __future__ import annotations

import dataclasses
import glob
import math
import os
import shutil
from typing import Callable, Iterable, Optional

import wtune.extensions
import wtune.machine
from ._deps import resume, sweep
from .attempts import Attempt, Outcome, _attempt_namelist, _attempt_times, _run_attempt
from .common import UNSET, RunError, _fmt, _hms
from .context import RunContext
from .estimate import history_wall, time_limit_s
from .geometry import _step_fraction, step_of, step_time
from .plan import Segment
from .report import _save_report
from .staging import FRESH_BP_SUBDIR

CFL_RESCUE_FACTOR = 0.75
CFL_RESCUE_MIN_FACTOR = 0.5            # never search below half the case's dt


def rescue_time_step(dt_s: float, seg_len_s: int, multiples: Iterable = ()) -> Optional[int]:
    """Largest integer step in [ceil(0.5 dt), floor(0.75 dt)] dividing seg_len_s and every
    nonzero entry of `multiples` (history interval, radt in s); retried with seg_len_s alone
    (a tiny step that divides everything is not a rescue); None if none."""
    top = int(math.floor(CFL_RESCUE_FACTOR * dt_s))
    bottom = max(1, int(math.ceil(CFL_RESCUE_MIN_FACTOR * dt_s)))
    for extra in (tuple(int(m) for m in multiples if m and float(m) == int(m)), ()):
        for c in range(top, bottom - 1, -1):
            if seg_len_s % c == 0 and all(m % c == 0 for m in extra):
                return c
    return None


def rsl_has_cfl(rundir: str) -> bool:
    """True when any rsl.error.* / rsl.out.* line mentions cfl (every rank, not just 0000)."""
    return resume.rsl_mentions(rundir, ("cfl",))


def _interrupted(out: Outcome) -> bool:
    """The job was cut short from outside: sacct says so, or the sbatch trap recorded
    INTERRUPTED in segment_status.txt (it exits 3, so sacct says FAILED)."""
    if out.state in resume.INFRA_STATES:
        return True
    try:
        with open(os.path.join(out.dir, "segment_status.txt")) as f:
            return "INTERRUPTED" in f.read()
    except OSError:
        return False


def resume_time_limit(ctx: RunContext, seg: Segment, a: int, snap: dict) -> str:
    """--time of an attempt resumed at step a: the failed attempt's observed s/step (else the
    segment's predicted rate) x the steps left, + init + restart read."""
    n, total = snap.get("n_d01_steps"), snap.get("sum_d01_elapsed_s")
    rate = (total / n if n and total
            else seg.predicted_compute_s / (seg.end_step - seg.start_step))
    over = ctx.plan.over
    wall = rate * (seg.end_step - a) + over.init_s + over.restart_read_s
    return _hms(time_limit_s(wall, ctx.plan.wall_limit_s))


def _resume_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt,
                    out: Outcome) -> tuple[Optional[Attempt], Optional[str]]:
    """(the attempt resuming the failed `out` from its latest valid restart, None), else
    (None, why not). Logs and records the resume."""
    geo, rec, run, limit = ctx.geo, out.rec, ctx.state, ctx.opts.resume or 0
    if run.resumes >= limit:
        return None, "--resume %d %s" % (limit, "already used" if limit else "disables it")
    is_bp = sweep.campaign_configs._import_wtune_bp().is_bp
    times = resume.valid_restart_times(out.dir, geo.max_dom, is_bp)
    last = resume.last_d01_time(out.dir)
    align = ctx.plan.inputs.align
    n, why = resume.resume_point([step_of(geo, t) for t in times], att.a, seg.end_step,
                                 (align.steps if align else None) or geo.quantum,
                                 _step_fraction(geo, last) if last else None)
    if n is None:
        return None, why
    run.resumes += 1
    rec.resume_time = _fmt(step_time(geo, n))
    ctx.report["resumes"] = run.resumes
    ctx.report["decisions"].append({"segment": k, "decision": "resume",
                                    "class": rec.failure_class, "job_id": rec.job_id,
                                    "restart": rec.resume_time})
    ctx.log("  resume %d/%d: %s at job %s; restart %s"
            % (run.resumes, limit, rec.failure_class, rec.job_id, rec.resume_time))
    return dataclasses.replace(att, a=n, src_dir=out.dir, resume=run.resumes,
                               time_limit=resume_time_limit(ctx, seg, n, out.snap)), None


def _rescue_dt(ctx: RunContext, seg: Segment, att: Attempt) -> tuple[Optional[int], int]:
    """(CFL-rescue time step or None, the length in s it has to divide)."""
    geo, cadence_s = ctx.geo, ctx.plan.restart.cadence_s
    t_a, t_b = _attempt_times(geo, seg, att)
    seg_len = int(round((t_b - t_a).total_seconds()))
    if cadence_s:                                  # keep the restart alarm whole too
        seg_len = math.gcd(seg_len, int(cadence_s))
    hist = geo.hist[0].iv_s if geo.hist[0] else None
    return rescue_time_step(geo.dt, seg_len, (hist, geo.radt_s)), seg_len


def _cfl_rescue_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt) -> Attempt:
    """`att` again with the rescue time step; the run becomes PHYSICS-GRAY."""
    geo, report = ctx.geo, ctx.report
    new_dt, seg_len = _rescue_dt(ctx, seg, att)
    if new_dt is None:
        raise RunError("segment %d failed with CFL/NaN and no integer time step in "
                       "[%.2f, %.2f] s divides its %d s length" % (
                           k, CFL_RESCUE_MIN_FACTOR * geo.dt, CFL_RESCUE_FACTOR * geo.dt,
                           seg_len))
    ctx.state.rescue_used = True
    report["physics_gray"] = True
    report["physics_gray_reason"] = (
        "segment %d re-run with time_step %d s (case %g s) after CFL/NaN" % (k, new_dt, geo.dt))
    report["decisions"].append({"segment": k, "decision": "cfl_rescue", "time_step": new_dt})
    ctx.log("PHYSICS-GRAY: " + report["physics_gray_reason"])
    return dataclasses.replace(att, time_step=new_dt)


def _cfl_advice(ctx: RunContext, k: int, seg: Segment, att: Attempt, seg_dir: str) -> None:
    """Without --cfl-rescue (decision 0056: solution-changing knobs are advisory only): print
    the recommended time_step and write the would-be namelist; nothing is submitted."""
    geo, log = ctx.geo, ctx.log
    advice_dt, _len = _rescue_dt(ctx, seg, att)
    advice_path = None
    if advice_dt is not None:
        advice_path = os.path.join(seg_dir, "segment.advice.namelist.input")
        with open(advice_path, "w") as f:
            f.write(_attempt_namelist(ctx, k, seg, att, advice_dt))
    log("PHYSICS-GRAY ADVICE: segment %d failed with CFL/NaN; recommended "
        "time_step=%s s (case %g s) -- this CHANGES THE SOLUTION (PHYSICS-GRAY), "
        "so it is advisory only and was NOT applied. Re-run with --cfl-rescue "
        "(the only user-pre-authorized exception) to apply it automatically."
        % (k, advice_dt if advice_dt is not None else "none found in range", geo.dt))
    if advice_path:
        log("  advice namelist written: %s" % advice_path)
    for _w in wtune.extensions.cfl_coupling_warnings(ctx.plan.case_dir, geo.dt, advice_dt):
        log(_w)
    ctx.report["decisions"].append({"segment": k, "decision": "cfl_advice",
                                    "recommended_time_step": advice_dt,
                                    "advice_namelist": advice_path})


def _failure_reason(ctx: RunContext, out: Outcome, cfl: bool, why: Optional[str]) -> str:
    rec, snap = out.rec, out.snap
    reason = ("watch status %s, sacct %s, restart %s, warnings %s%s"
              % (snap.get("status"), out.state, "ok" if rec.restart_ok else "MISSING",
                 ",".join(rec.warnings) or "none", ", CFL/NaN seen" if cfl else ""))
    if cfl and ctx.state.rescue_used:
        reason += " (CFL rescue already used once in this run)"
    if why:
        reason += "; not resumed: " + why
    return reason


def _next_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt, out: Outcome) -> Attempt:
    """The attempt that follows a failed one: an auto-resume (infrastructure failure with a
    valid restart) or the CFL rescue; anything else raises RunError -- never a silent retry."""
    rec, snap = out.rec, out.snap
    if rec.stalled is not UNSET:
        rec.failure_class = "stalled"       # a full disk would only stall again: no resume
        raise RunError("segment %d (%s) stalled and was cancelled: %s" % (k, out.dir, rec.stalled))
    codes = rec.warnings
    nan = any(w["code"] == "FAILED" and "NaN" in w["text"] for w in snap.get("warnings", []))
    # a cfl line alone is routine output: in an interrupted job it is not the failure
    cfl = nan or (("CFL" in codes or rsl_has_cfl(out.dir)) and not _interrupted(out))
    model = cfl or "FAILED" in codes or resume.rsl_mentions(
        out.dir, [n for n in resume.MODEL_NEEDLES if n != "cfl"])
    rec.failure_class = resume.classify_failure(out.state, model)
    why = None
    if rec.failure_class == "infrastructure":
        nxt, why = _resume_attempt(ctx, k, seg, att, out)
        if nxt:
            return nxt
        rec.no_resume_reason = why
    if (ctx.opts.cfl_rescue and cfl and rec.failure_class == "model"
            and not ctx.state.rescue_used):
        return _cfl_rescue_attempt(ctx, k, seg, att)
    if cfl and not ctx.opts.cfl_rescue:
        _cfl_advice(ctx, k, seg, att, out.dir)
    raise RunError("segment %d (%s) failed: %s" % (k, out.dir,
                                                   _failure_reason(ctx, out, cfl, why)))


def run_controller(ctx: RunContext) -> int:
    """Execute the plan's segments in order -> 0; a stop raises RunError. The report is
    rewritten after every attempt."""
    try:
        _run_segments(ctx)
    except Exception:
        if ctx.sidecar:                # a failed or stopped run's frames are converted and booked too
            ctx.sidecar.finish()
        raise
    if ctx.sidecar:
        ctx.sidecar.finish()
    if ctx.opts.compress_history and ctx.plan.restart.set_min is not None:
        _delete_own_restarts(ctx)
    return 0


def _delete_own_restarts(ctx: RunContext) -> None:
    """--compress-history after a run that succeeded: the restart files WTune itself asked for
    (not the case's) only served the segment hand-offs."""
    paths = sorted(glob.glob(os.path.join(ctx.plan.out, "seg_*", "wrfrst_d*")))
    for p in [p for p in paths if os.path.islink(p)]:   # links to the case's or the last segment's
        os.remove(p)
    paths = [p for p in paths if not os.path.islink(p)]
    size = 0
    for p in paths:
        if os.path.isdir(p):            # ADIOS2 restart: a BP directory
            size += sum(os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
        else:
            size += os.path.getsize(p)
            os.remove(p)
    ctx.report["restarts_deleted"] = dict(files=len(paths), bytes=size)
    _save_report(ctx)
    ctx.log("deleted %d restart file(s) WTune wrote (%.1f GB)" % (len(paths), size / 1e9))


def _run_segments(ctx: RunContext) -> None:
    report, segs = ctx.report, ctx.segments
    report["segments"] = segs
    prev_dir = ctx.plan.start_dir
    for k, seg in enumerate(segs):
        is_last = k == len(segs) - 1
        if "attempts" not in seg:
            seg.attempts = []
        att = Attempt(a=seg.start_step, src_dir=prev_dir)
        out = _run_attempt(ctx, k, seg, att, is_last)
        while not out.ok:
            att = _next_attempt(ctx, k, seg, att, out)
            out = _run_attempt(ctx, k, seg, att, is_last)
        seg.observed_wall_s = out.elapsed
        seg.run_dir = prev_dir = out.dir
        # the copy is only for this job's reads; a failed attempt keeps it for diagnosis
        shutil.rmtree(os.path.join(out.dir, FRESH_BP_SUBDIR), ignore_errors=True)
        report["decisions"].append({"segment": k, "decision": "done" if is_last else "next"})
        if not is_last:
            _update_from_observed(ctx, k, out.snap, out.elapsed)
        _save_report(ctx)


def _update_from_observed(ctx: RunContext, k: int, snap: dict, elapsed: Optional[float]) -> None:
    """Segment k's observed overhead (job elapsed - sum d01) replaces the predicted overheads
    in every later segment's wall and --time (+ the restart read segment 0 did not do);
    without a prediction the observed mean d01 step replaces the compute model too. The L
    grid never changes."""
    geo, over = ctx.geo, ctx.plan.over
    sum_d01, n = snap.get("sum_d01_elapsed_s"), snap.get("n_d01_steps")
    if elapsed is None or not n or sum_d01 is None:
        return
    ovh = max(0.0, elapsed - sum_d01)
    read = over.restart_read_s if k == 0 else 0.0
    s_step = sum_d01 / n if ctx.plan.replan else None
    for seg in ctx.segments[k + 1:]:
        a, b = seg.start_step, seg.end_step
        if s_step is not None:
            seg.predicted_compute_s = s_step * (b - a)
            wall = seg.predicted_compute_s + ovh + read
        else:
            wall = seg.predicted_compute_s + ovh + read + history_wall(geo, over, a, b)
        seg.predicted_wall_s = wall
        seg.time_limit = _hms(time_limit_s(wall, ctx.plan.wall_limit_s))
        seg.wall_source = "observed overhead of segment %d" % k
    ctx.report["decisions"].append({
        "segment": k, "decision": "observed_overhead", "overhead_s": ovh,
        "observed_s_per_step": sum_d01 / n, "compute_replaced": s_step is not None})
    ctx.log("segment %d observed: overhead %.0f s, %.3f s/step -> later --time updated"
            % (k, ovh, sum_d01 / n))


def _make_sidecar(ctx: RunContext, nc_root: str, runner: Optional[Callable] = None):
    """The run's wtune.sidecar.Sidecar; its sruns go through `runner` (tests inject a fake)."""
    import wtune.sidecar
    out = ctx.plan.out
    return wtune.sidecar.Sidecar(
        out, nc_root, ctx.geo.max_dom, sweep.campaign_configs._import_wtune_bp().is_bp,
        ctx.report, save=lambda: _save_report(ctx), log=ctx.log,
        cores_per_node=wtune.machine.cores_per_node(), exclude_arg=sweep.exclude_arg(),
        runner=runner or wtune.sidecar.popen_runner(os.path.join(out, "sidecar.log")),
        replace=ctx.opts.compress_history,
        keep_last_restart=ctx.opts.compress_history and ctx.plan.restart.set_min is not None)
