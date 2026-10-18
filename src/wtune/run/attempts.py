"""One attempt = one Slurm job of a segment: stage its run dir, submit segment.sbatch, wait
(sacct), inspect the result (wtune.watch.snapshot) and record it in the report."""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
import shlex
import time
from typing import Any, Callable, Optional

import wtune.stopping
from ._deps import sweep, wp, tune
from .common import ADIOS2_IO_FORM, UNSET, WRF_FMT, Record, RunError, RunStopped, _fmt
from .context import RunContext
from .geometry import Geometry, step_time
from .namelist import render_segment_namelist
from .plan import Segment, restart_times
from .report import _save_report
from .sbatch import render_segment_sbatch
from .staging import (merge_adios2_xml, output_dir_names, precreate_output_dirs,
                      remove_empty_output_dirs, render_adios2_restart_xml, restart_files,
                      setup_segment_dir)

UNKNOWN_POLLS_MAX = 10                 # sacct not listing a fresh job id yet: poll this many times


@dataclasses.dataclass(frozen=True)
class Attempt:
    """One job of a segment: d01 steps (a, segment end] from the restart in src_dir."""
    a: int
    src_dir: Optional[str] = None      # None: cold start
    time_step: Optional[int] = None    # s, CFL rescue
    resume: int = 0                    # ordinal of the --resume that started it
    time_limit: Optional[str] = None   # --time when not the segment's own


@dataclasses.dataclass(eq=False)
class AttemptRecord(Record):
    """An attempt in run_report.json: written at submission, completed when the job ends."""
    dir: str
    job_id: str
    time_step: Optional[int]
    submitted_utc: str
    start: Any = UNSET                 # WRF_FMT, resumed attempts only
    state: Any = UNSET                 # sacct state
    elapsed_s: Any = UNSET
    watch_status: Any = UNSET
    warnings: Any = UNSET              # wtune.watch warning codes
    sum_d01_elapsed_s: Any = UNSET
    n_d01_steps: Any = UNSET
    restart_ok: Any = UNSET
    ok: Any = UNSET
    failure_class: Any = UNSET         # wtune.resume.classify_failure
    resume_time: Any = UNSET           # the restart the next attempt resumed from
    no_resume_reason: Any = UNSET
    stalled: Any = UNSET               # why StallWatch cancelled the job


@dataclasses.dataclass
class Outcome:
    """What one attempt left: its dir, report record, sacct state and watch snapshot."""
    dir: str
    rec: AttemptRecord
    state: str
    elapsed: float
    snap: dict
    ok: bool


def attempt_dir_name(k: int, att: Attempt) -> str:
    """seg_<k>[_rescue][_r<n>]; a later attempt sorts later (wtune.watch._SEG_DIR_RE)."""
    return "seg_%d%s%s" % (k, "_rescue" if att.time_step else "",
                           "_r%d" % att.resume if att.resume else "")


def _attempt_times(geo: Geometry, seg: Segment, att: Attempt) -> tuple[dt.datetime, dt.datetime]:
    """(start, end) datetimes of an attempt, whole seconds as WRF names them."""
    return (dt.datetime.strptime(_fmt(step_time(geo, att.a)), WRF_FMT),
            dt.datetime.strptime(seg.end, WRF_FMT))


def _attempt_namelist(ctx: RunContext, k: int, seg: Segment, att: Attempt,
                      time_step: Optional[int]) -> str:
    t_a, t_b = _attempt_times(ctx.geo, seg, att)
    inputs = ctx.plan.inputs
    return render_segment_namelist(ctx.plan.text, t_a, t_b, k, seg.config,
                                   ctx.plan.restart.set_min, ctx.opts.restart_io_form,
                                   time_step=time_step, io_streams=inputs.io_streams,
                                   per_record_edits=inputs.per_record_edits,
                                   resumed=att.src_dir is not None)


def _restart_links(geo: Geometry, k: int, att: Attempt, start: str) -> list[str]:
    """Every domain's wrfrst at `start` in the attempt's source dir; [] for a cold start."""
    if att.src_dir is None:
        return []
    found = restart_files(att.src_dir, start, geo.max_dom)
    missing = [d for d, p in found.items() if not p]
    if missing:
        raise RunError("segment %d: no wrfrst for d%s at %s in %s"
                       % (k, ",".join("%02d" % d for d in missing), start, att.src_dir))
    return [p for d in sorted(found) for p in found[d]]


def _attempt_xml(ctx: RunContext, seg: Segment, att: Attempt, start: str) -> Optional[str]:
    """The attempt's adios2.xml text; None when the case's own file is linked instead."""
    adios2_io = ctx.plan.inputs.io_streams is not None
    base = (merge_adios2_xml(ctx.case_adios2_xml, ctx.plan.inputs.bp_xml)
            if adios2_io else ctx.case_adios2_xml)
    if int(ctx.opts.restart_io_form) == ADIOS2_IO_FORM:
        times = (([start] if att.src_dir else [])
                 + restart_times(ctx.geo, att.a, seg.end_step, ctx.plan.restart.cadence_s))
        return render_adios2_restart_xml(times, ctx.geo.max_dom, base)
    return base if adios2_io and base else None


def _stage_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt) -> tuple[str, list]:
    """Create and populate the attempt's run dir -> (dir, pre-created output dir names)."""
    geo, log, plan, cc = ctx.geo, ctx.log, ctx.plan, sweep.campaign_configs
    seg_dir = os.path.join(plan.out, attempt_dir_name(k, att))
    t_a, t_b = _attempt_times(geo, seg, att)
    txt = _attempt_namelist(ctx, k, seg, att, att.time_step)
    staged = setup_segment_dir(plan.case_dir, seg_dir, txt,
                               _restart_links(geo, k, att, _fmt(t_a)),
                               _attempt_xml(ctx, seg, att, _fmt(t_a)), ctx.wrf_tree,
                               bp_links=plan.inputs.links,
                               fresh_sources=plan.inputs.fresh_sources, copier=ctx.bp_copier)
    merged = cc.merge_iofields(txt, seg_dir)
    out_names = []
    if plan.inputs.io_streams is not None:
        nl_seg = wp.parse_namelist(os.path.join(seg_dir, "namelist.input"))
        out_names = output_dir_names(nl_seg, t_a, t_b, geo.max_dom)
        log("  pre-created %d output dirs" % precreate_output_dirs(seg_dir, out_names))
    if k == 0 and not seg.attempts:
        staged = cc.format_namelist_stage(staged, [plan.case_dir])
        for ln in staged + merged:
            log(ln)
    return seg_dir, out_names


def _submit_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt, seg_dir: str) -> str:
    """Write segment.sbatch into seg_dir and submit it -> job id."""
    limit = att.time_limit or seg.time_limit
    script = render_segment_sbatch(
        seg_dir, k, seg.config, limit,
        "wtrun-%s-s%d" % (os.path.basename(ctx.plan.case_dir), k),
        env_source=ctx.opts.env_source, campaign_dir=ctx.opts.campaign_dir)
    spath = os.path.join(seg_dir, "segment.sbatch")
    with open(spath, "w") as f:
        f.write(script)
    job_id = sweep.submit_job("sbatch %s" % shlex.quote(spath),
                              sbatch_runner=ctx.runners.sbatch_runner)
    ctx.log("segment %d/%d %s -> %s  %s  --time=%s  job %s%s"
            % (k, len(ctx.segments) - 1, _fmt(step_time(ctx.geo, att.a)), seg.end,
               seg.config.id, limit, job_id,
               "  (CFL rescue dt=%ds)" % att.time_step if att.time_step else ""))
    return job_id


def wait_for_job(job_id: str, sacct_runner: Callable, sacct_cmd: str = "sacct",
                 poll_s: float = 60.0, sleep: Callable = time.sleep, log: Callable = print,
                 on_poll: Optional[Callable] = None) -> tuple[str, Optional[float]]:
    """Poll sacct until job_id leaves PENDING/RUNNING -> (state, elapsed_s or None). on_poll
    (state) runs before every sleep (--sidecar)."""
    unknown = 0
    last = None
    while True:
        state, elapsed = sweep.sacct_states([job_id], sacct_runner=sacct_runner,
                                            sacct_cmd=sacct_cmd).get(str(job_id), ("UNKNOWN", ""))
        if state != last:
            log("  job %s: %s" % (job_id, state))
            last = state
        if state == "UNKNOWN":
            unknown += 1
            if unknown > UNKNOWN_POLLS_MAX:
                return state, None
        elif state not in sweep.IN_FLIGHT_STATES:
            return state, (tune._parse_slurm_elapsed(elapsed) if tune else None)
        else:
            unknown = 0                # only consecutive misses count
        if on_poll:
            on_poll(state)
        sleep(poll_s)


class StallWatch:
    """An on_poll(state): while the job runs, cancel it once rsl.error.0000 has not changed
    (size, mtime) for stall_s. WRF+ADIOS2 waits forever on a full quota instead of failing
    (f1-arm job 15170 idled 75 min to its time limit)."""

    def __init__(self, rsl: str, job_id: str, stall_s: float, clock: Callable,
                 cancel: Callable, log: Callable) -> None:
        self.rsl, self.job_id, self.stall_s = rsl, job_id, stall_s
        self.clock, self.cancel, self.log = clock, cancel, log
        self.sig, self.since, self.why = UNSET, None, None

    def __call__(self, state: str) -> None:
        if self.why or not self.stall_s or state != "RUNNING":
            return
        try:
            st = os.stat(self.rsl)
            sig = (st.st_size, st.st_mtime)
        except OSError:
            sig = None
        now = self.clock()
        if sig != self.sig or self.since is None:
            self.sig, self.since = sig, now
            return
        if now - self.since >= self.stall_s:
            self.why = "%s unchanged for %d min" % (os.path.basename(self.rsl),
                                                   (now - self.since) // 60)
            self.log("  job %s: STALLED (%s); cancelling it" % (self.job_id, self.why))
            self.cancel(self.job_id)


def stop_before_submit() -> None:
    """No segment is submitted once the user stopped the run."""
    if wtune.stopping.STOP["reason"]:
        raise RunStopped("%s; no further segment submitted" % wtune.stopping.STOP["reason"])


def _run_attempt(ctx: RunContext, k: int, seg: Segment, att: Attempt, is_last: bool) -> Outcome:
    """Stage, submit, wait for and inspect one attempt; its record goes into the report."""
    geo, side, run = ctx.geo, ctx.sidecar, ctx.runners
    seg_dir, out_names = _stage_attempt(ctx, k, seg, att)
    stop_before_submit()
    job_id = _submit_attempt(ctx, k, seg, att, seg_dir)
    ctx.state.jobs.append(job_id)
    if ctx.state_file:
        wtune.stopping.write_state(ctx.state_file, "run", ctx.state.jobs)
    rec = AttemptRecord(dir=seg_dir, job_id=job_id, time_step=att.time_step,
                        submitted_utc=dt.datetime.utcnow().isoformat())
    if att.resume:
        rec.start = _fmt(step_time(geo, att.a))
    seg.attempts.append(rec)
    _save_report(ctx)
    stall = StallWatch(os.path.join(seg_dir, "rsl.error.0000"), job_id,
                       60.0 * (ctx.opts.stall_minutes or 0), run.clock,
                       run.scancel_runner or (lambda _j: None), run.log)
    side_poll = side.watcher(seg_dir, job_id, seg.config) if side else None

    def on_poll(state):
        if wtune.stopping.STOP["reason"]:
            if run.scancel_runner:
                run.scancel_runner(job_id)
            rec.state = "CANCELLED"
            raise RunStopped("%s; segment %d job %s cancelled"
                             % (wtune.stopping.STOP["reason"], k, job_id))
        stall(state)
        if side_poll:
            side_poll(state)
    state, elapsed = wait_for_job(
        job_id, run.sacct_runner, ctx.opts.sacct_cmd, ctx.opts.poll_interval, run.sleep, run.log,
        on_poll=on_poll)
    if stall.why:
        rec.stalled = stall.why
    remove_empty_output_dirs(seg_dir, out_names)    # before any wrfout/wrfrst check
    if side:
        side.job_ended(seg_dir)
    snap = run.snapshot_fn(seg_dir)
    rec.state, rec.elapsed_s, rec.watch_status = state, elapsed, snap.get("status")
    rec.warnings = [w["code"] for w in snap.get("warnings", [])]
    rec.sum_d01_elapsed_s, rec.n_d01_steps = snap.get("sum_d01_elapsed_s"), snap.get("n_d01_steps")
    rec.restart_ok = is_last or all(restart_files(seg_dir, seg.end, geo.max_dom).values())
    rec.ok = snap.get("status") == "done" and rec.restart_ok
    return Outcome(seg_dir, rec, state, elapsed, snap, rec.ok)
