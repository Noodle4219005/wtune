"""Slurm side of the tuner: submit a CONFIG_FILE job, poll it under the wall or running-time
budget, cancel, read sacct timing / MaxRSS, and the per-job ranking validity gates."""
from __future__ import annotations

import datetime
import os
import re
import shutil
import sys
from typing import Callable

import wtune.machine
import wtune.stopping
from ._deps import sweep
from .budget import _job_time_est
from .common import ANCHOR_END_ID, _fmt_hms, _notes_kv, parse_run_id
from .context import STOP, Ctx, QueueCapExceeded, StopRequested

PROGRESS_BAR_WIDTH = 24                # characters of the progress bar
ANCHOR_DRIFT_MAX = wtune.machine.MEASURED["anchor_drift_max"]   # |s(ANCHOREND)/s(ANCHOR) - 1|
SACCT_MISS_MAX = 5                     # consecutive polls sacct may not list a submitted job
_RSS_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)$")
_RSS_TO_KB = {"": 1.0, "K": 1.0, "M": 1024.0, "G": 1024.0 ** 2, "T": 1024.0 ** 3}
MAXRSS_NOTE = ("MaxRSS is whatever Slurm accounting reports for the job's steps; mpirun-launched "
               "ranks can bypass step accounting (f1: knowledge/hpc/2026-08-20-f1-reports-no-"
               "rapl-energy-and-no-useful-maxrss.md -- it reported the batch script's 6 MB), so "
               "a small value means 'not accounted', not 'small footprint'")


def print_progress(stage: str, elapsed_s: float, budget_s: float, note: str = "") -> None:
    frac = 0.0 if budget_s <= 0 else min(1.0, elapsed_s / budget_s)
    filled = int(round(frac * PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    line = "[%-9s] [%s] %3d%%  elapsed=%s / budget=%s" % (
        stage, bar, int(frac * 100), _fmt_hms(elapsed_s), _fmt_hms(budget_s))
    if note:
        line += "  " + note
    print(line)


def cancel_jobs(ctx: Ctx, job_ids: list) -> None:
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return
    print("cancelling %d in-flight job(s) owned by this run: %s" % (len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)


def record_job(ctx: Ctx, job_id: str) -> None:
    """An owned job; with ctx.state_file also on disk (pid + every owned job) for `wtune stop`."""
    ctx.owned_jobs.append(job_id)
    if ctx.state_file:
        wtune.stopping.write_state(ctx.state_file, "tune", ctx.owned_jobs)


def stop_before_submit() -> None:
    """Nothing is submitted once the user stopped the tune."""
    if STOP["reason"]:
        raise StopRequested("%s; nothing more submitted" % STOP["reason"])


def _stop_if_requested(ctx: Ctx, in_flight: list) -> None:
    """Ctrl-C / wtune stop: cancel `in_flight` and raise StopRequested; the later pipeline
    phases see it as cap_hit and do not run."""
    if STOP["reason"]:
        if ctx.shared_queue is not None:
            ctx.shared_queue["cap_hit"] = True
        cancel_jobs(ctx, [j for j in in_flight if j not in ctx.cancelled_jobs])
        raise StopRequested("%s; in-flight job(s) %s cancelled" % (STOP["reason"],
                                                                  ",".join(in_flight)))


def _in_flight(job_ids: list, states: dict, misses: dict, stage: str) -> list:
    """The ids still PENDING/RUNNING. One sacct does not list (the call failed, or slurmdbd has
    not recorded the fresh job yet) counts as in flight until SACCT_MISS_MAX consecutive misses
    (`misses`: {job: count}, kept by the caller across polls)."""
    out = []
    for j in job_ids:
        if j in states:
            misses.pop(j, None)
            if states[j][0] in sweep.IN_FLIGHT_STATES:
                out.append(j)
            continue
        misses[j] = misses.get(j, 0) + 1
        if misses[j] < SACCT_MISS_MAX:
            out.append(j)
        elif misses[j] == SACCT_MISS_MAX:
            print("%s: job %s not reported by sacct in %d consecutive polls -- no longer waited "
                  "for" % (stage, j, SACCT_MISS_MAX))
    return out


def wait_for_jobs(ctx: Ctx, job_ids: list, stage: str, note: str = "",
                  exempt_budget: bool = False) -> dict:
    """Poll sacct until `job_ids` leave PENDING/RUNNING, printing one progress line per poll; at
    the budget deadline cancel those ids only. Returns the last {job: (State, Elapsed)} ({} for
    no ids). budget_mode "running" uses _wait_running_budget (exempt_budget). The jobs' input
    copies are removed once they have ended, whatever their state."""
    states = _wait_for_jobs(ctx, job_ids, stage, note, exempt_budget)
    for j in job_ids:
        d = ctx.input_copies.pop(j, None)
        if d:
            shutil.rmtree(d, ignore_errors=True)
    return states


def _wait_for_jobs(ctx: Ctx, job_ids: list, stage: str, note: str,
                   exempt_budget: bool) -> dict:
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return {}
    if ctx.budget_mode == "running":
        return _wait_running_budget(ctx, job_ids, stage, note, exempt_budget)
    misses: dict = {}
    while True:
        elapsed = ctx.elapsed()
        states = sweep.sacct_states(job_ids, sacct_runner=ctx.sacct_runner, sacct_cmd=ctx.sacct_cmd)
        in_flight = _in_flight(job_ids, states, misses, stage)
        print_progress(stage, elapsed, ctx.budget_s, note)
        if not in_flight:
            return states
        _stop_if_requested(ctx, in_flight)
        if elapsed >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)
            return states
        ctx.clock.sleep(min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed)))


def _wait_running_budget(ctx: Ctx, job_ids: list, stage: str, note: str,
                         exempt_budget: bool = False) -> dict:
    """wait_for_jobs on the RUNNING-time budget (decision 0055 gap 3): a poll interval is charged
    only when one of our jobs (or ctx.background_jobs) runs, else counted as queue wait (capped by
    max_queue_s). Cancels at the budget (not when `exempt_budget`) or the wall cap."""
    misses: dict = {}
    # each poll is charged from the end of the last one: the sacct query and the progress line
    # take seconds with many jobs, and charging the sleep alone left them uncounted (isc-synth on
    # f1: 18:40 charged of 22:11 with a job in flight, the rest shown as controller time)
    t0 = ctx.clock.now()
    while True:
        elapsed = ctx.elapsed()
        wall = ctx.wall_elapsed()
        bg = [j for j in ctx.background_jobs if j not in job_ids]
        states = sweep.sacct_states(job_ids + bg, sacct_runner=ctx.sacct_runner,
                                    sacct_cmd=ctx.sacct_cmd)
        bg_running = any(states.get(j, ("", ""))[0] == "RUNNING" for j in bg)
        states = {j: v for j, v in states.items() if j not in bg}
        in_flight = [j for j in _in_flight(job_ids, states, misses, stage)
                     if j not in ctx.cancelled_jobs]
        print_progress(stage, elapsed, ctx.budget_s,
                       (note + " " if note else "") + "(running budget; wall %s, queue-only %s)"
                       % (_fmt_hms(wall), _fmt_hms(ctx.queue_wait_s)))
        if not in_flight:
            return states
        _stop_if_requested(ctx, in_flight)
        over_budget = elapsed >= ctx.budget_s and not exempt_budget
        if over_budget or (ctx.max_wall_s is not None and wall >= ctx.max_wall_s):
            if not over_budget:
                print("hard wall-clock cap reached (wall %s >= %s)"
                      % (_fmt_hms(wall), _fmt_hms(ctx.max_wall_s)))
            cancel_jobs(ctx, in_flight)
            return states
        n_running = sum(1 for j in in_flight if states.get(j, ("", ""))[0] == "RUNNING")
        running = n_running > 0 or bg_running
        step = min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed))
        if exempt_budget:
            step = ctx.poll_interval
        if ctx.max_wall_s is not None:
            step = min(step, max(0.0, ctx.max_wall_s - wall))
        if not running:            # nothing consumes budget: only the wall cap bounds the poll
            step = ctx.poll_interval if ctx.max_wall_s is None else \
                min(ctx.poll_interval, max(0.0, ctx.max_wall_s - wall))
        ctx.clock.sleep(step)
        t1 = ctx.clock.now()
        charged, t0 = t1 - t0, t1
        if running:
            ctx.add_running(charged)
        if not running:
            ctx.add_queue_wait(charged)
            if ctx.max_queue_s is not None and ctx.queue_wait_s >= ctx.max_queue_s:
                if ctx.shared_queue is not None:
                    ctx.shared_queue["cap_hit"] = True
                cancel_jobs(ctx, [j for j in in_flight if j not in ctx.cancelled_jobs])
                raise QueueCapExceeded(
                    "queue-only wait %s reached --max-queue-hours (%s); in-flight job(s) %s "
                    "cancelled" % (_fmt_hms(ctx.queue_wait_s), _fmt_hms(ctx.max_queue_s),
                                   ",".join(in_flight)))


def _submits_for_real(ctx: Ctx) -> bool:
    """Only real sbatch submission converts or copies BP inputs; fake runners (tests) never do."""
    return ctx.sbatch_runner is sweep.default_sbatch_runner


def nc2bp_command(ctx: Ctx, cases: list[str], cases_json: str | None, out_dir: str) -> str:
    """sbatch command of ONE campaign/nc2bp.sbatch job converting `cases` (CHECK=0; the
    probe exclusions apply: never head)."""
    campaign = (sweep.campaign_dir_for(ctx.sbatch_script)
                or os.path.dirname(os.path.abspath(sweep._expand(ctx.sbatch_script))))
    export = ("ALL,SITE=%s,CHECK=0,CAMPAIGN=%s,CASES=%s"
              % (wtune.machine.site(), campaign, ":".join(cases)))
    if cases_json:
        export += ",CASES_JSON=%s" % cases_json
    return ("source %s; sbatch -p %s %s --job-name=nc2bp-%s --output=%s/nc2bp-%%j.out --export=%s %s"
            % (sweep._expand(ctx.env_source), wtune.machine.partition(), sweep.scheduler_args(),
               cases[0], out_dir, export,
               os.path.join(campaign, "nc2bp.sbatch")))


_BP_READY: set = set()   # (case_id, cases_json) already converted in this process


def ensure_bp_inputs(ctx: Ctx, cases: list[tuple], out_dir: str) -> None:
    """Make the BP cache of every (case_id, cases_json) current before its probes: ONE nc2bp job
    for the stale ones, waited for and re-checked; RuntimeError otherwise. Probes converting
    in-job raced and stalled (scc jobs 2845-2855, 2884-2895)."""
    if not _submits_for_real(ctx):
        return
    cc = sweep.campaign_configs
    status = {c: cc.bp_cache_status(c, j) for c, j in cases}
    for c, (st, detail) in sorted(status.items()):
        if st == "none":
            print("bp inputs: %s not converted (%s)" % (c, detail))
    stale = [(c, j) for c, j in cases if status[c][0] == "stale"]
    if not stale:
        print("bp inputs: cache current")
        _BP_READY.update(cases)
        return
    names, cases_json = [c for c, _j in stale], stale[0][1]   # one case file per call
    os.makedirs(out_dir, exist_ok=True)
    t0 = ctx.clock.now()
    stop_before_submit()
    job_id = sweep.submit_job(nc2bp_command(ctx, names, cases_json, out_dir),
                              sbatch_runner=ctx.sbatch_runner)
    record_job(ctx, job_id)
    state = wait_for_jobs(ctx, [job_id], "bp-convert", exempt_budget=True).get(
        job_id, ("UNKNOWN", ""))[0]
    still = [c for c, j in stale if cc.bp_cache_status(c, j)[0] != "current"]
    if state != "COMPLETED" or still:
        raise RuntimeError("ERROR: BP input conversion job %s ended %s; cache not current for %s "
                           "-- no probes submitted (log: %s/nc2bp-%s.out)"
                           % (job_id, state, " ".join(still) or "-", out_dir, job_id))
    _BP_READY.update(cases)
    print("bp inputs: converted %s in %.0f s (job %s)" % (" ".join(names), ctx.clock.now() - t0,
                                                          job_id))


def _writable_at(path: str) -> tuple[float, str]:
    """wtune.run.space.writable_bytes at path's nearest existing ancestor."""
    from wtune.run.space import writable_bytes
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return writable_bytes(path)


def input_space_need(case_id: str, cases_json: str | None = None) -> tuple[float, float, str]:
    """(bytes of the BP conversion still to write, bytes of one per-record input copy, cache
    dir) of the case's probe inputs."""
    from wtune.run.space import input_bytes
    cc = sweep.campaign_configs
    res, cache, _stale = cc._case_bp_plan(case_id, cases_json)
    conv, copy = input_bytes(res, cache, cc._import_wtune_bp())
    return conv, copy, cache


def check_input_space(case_id: str, cases_json: str | None = None,
                      writable: Callable | None = None) -> None:
    """SystemExit before any job when converting the case's inputs to BP does not fit the space
    writable at the cache (f1-arm elevconv 2026-10-23: 2.0 GB of netCDF became 7.9 GB of BP and
    pushed home2 past its quota); a line when it takes half of it. A case whose plan cannot be
    read (legacy netCDF, unknown case) is not checked."""
    if os.environ.get("WTUNE_INPUT_FORMAT") == "netcdf":
        return
    try:
        conv, _copy, cache = input_space_need(case_id, cases_json)
    except (KeyError, OSError):
        return
    if conv <= 0:
        return
    free, src = (writable or _writable_at)(cache)
    if conv > free:
        raise SystemExit("wtune tune: converting the inputs of %s to BP needs %.1f GB in %s, "
                         "only %.1f GB is writable (%s) -- free space, or bypass with "
                         "--no-space-check" % (case_id, conv / 1e9, cache, free / 1e9, src))
    if conv >= 0.5 * free:
        print("input conversion %.1f GB of %.1f GB writable (%s)" % (conv / 1e9, free / 1e9, src))


def fresh_input_export(ctx: Ctx, case_id: str, job_out_dir: str, runs: int,
                       cases_json: str | None = None) -> list[tuple]:
    """[("WTUNE_BP_FRESH_DIR", dir)] after making `runs` per-record input copies here, on the
    submitting host, or []. NFS open stall, wiki 2026-10-03 adios2 gate. Only for real sbatch
    submission: a fake runner (tests) never copies."""
    if not _submits_for_real(ctx):
        return []
    if (case_id, cases_json) not in _BP_READY:   # first submission of this case converts once
        try:
            ensure_bp_inputs(ctx, [(case_id, cases_json)], os.path.join(job_out_dir, "bp_convert"))
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    if not wtune.machine.copy_probe_inputs():
        return []                               # the jobs read the shared BP cache
    try:
        _conv, one, _cache = input_space_need(case_id, cases_json)
        free, src = _writable_at(job_out_dir)
    except (KeyError, OSError):
        one, free, src = 0.0, 0.0, ""
    if one > free:                  # fresh_probe_inputs: reflink copies share extents, else one copy
        sys.stderr.write("WARNING: the per-record input copy for %s needs %.1f GB, only %.1f GB is "
                         "writable (%s): the job reads the shared BP cache\n"
                         % (case_id, one / 1e9, free / 1e9, src))
        return []
    try:
        d = sweep.campaign_configs.fresh_probe_inputs(
            case_id, os.path.join(job_out_dir, "bp_fresh"), runs, cases_json=cases_json)
    except Exception as e:                   # the job falls back to the shared cache (warned)
        sys.stderr.write("WARNING: fresh BP input copy for %s failed (%s: %s)\n"
                         % (case_id, type(e).__name__, e))
        return []
    return [("WTUNE_BP_FRESH_DIR", d)] if d else []


def submit_configs_job(ctx: Ctx, configs: list[dict], reps: int, name_suffix: str,
                       s_est: float | None, time_est: dict | None = None) -> tuple[str, str, dict]:
    """Submit one strict_campaign.sbatch CONFIG_FILE job. Returns (job id, configs.json, time
    estimate). CAMPAIGN is exported for a real campaign checkout so the spooled script never
    pairs with stale helpers."""
    stop_before_submit()
    if time_est is None:
        time_est = _job_time_est(ctx, len(configs), reps, s_est)
    job_out_dir = os.path.join(ctx.out_dir, name_suffix)
    os.makedirs(job_out_dir, exist_ok=True)
    configs_path = os.path.join(job_out_dir, "configs.json")
    sweep.write_configs_json(configs_path, configs)
    fresh = fresh_input_export(ctx, ctx.case, job_out_dir, len(configs) * reps + 2)   # + slack
    cmd_str = sweep.build_sbatch_command(
        ctx.case, ctx.nodes, ctx.cores_per_node, time_est["time_limit"], time_est["timeout_s"],
        reps, ctx.steps, os.path.abspath(configs_path),
        job_out_dir, env_source=ctx.env_source, sbatch_script=ctx.sbatch_script,
        campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
        extra_export=fresh or None)
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    record_job(ctx, job_id)
    ctx.job_configs[job_id] = configs
    if fresh:
        ctx.input_copies[job_id] = fresh[0][1]
    return job_id, configs_path, time_est


def evaluate_job_gates(job_id: str, rows: list[dict], s_by_cid: dict, expected_cids: list,
                       drift_max: float | None = None, require_restart: bool = True,
                       check_drift: bool = True) -> dict:
    """Validity gates of ONE ranking job (decision 0055), never raising: every expected cid has a
    row that restarted (restart_used=1, unless not require_restart) on a clean node, and
    |s(ANCHOREND)/s(ANCHOR) - 1| <= drift_max (when check_drift). Returns {job_id, ok, failures,
    anchor_drift, s_anchor, s_anchor_end}."""
    drift_max = wtune.machine.measured("anchor_drift_max") if drift_max is None else drift_max
    failures = []
    by_cid = {}
    for r in rows:
        parsed = parse_run_id(r.get("run_id"))
        if parsed and str(parsed["job"]) == str(job_id):
            by_cid[parsed["cid"]] = r
    for cid in expected_cids:
        row = by_cid.get(cid)
        if row is None:
            failures.append("%s: no ledger row" % cid)
            continue
        kv = _notes_kv(row.get("notes"))
        if require_restart and kv.get("restart_used") != "1":
            failures.append("%s: restart_used=%s (status=%s reason=%s) -- did not start from the "
                            "shared restart" % (cid, kv.get("restart_used", "missing"),
                                                kv.get("status", "-"), kv.get("reason", "-")))
        if kv.get("foreign_load") == "1":
            failures.append("%s: foreign_load=1 (%s)" % (cid, kv.get("foreign", "?")))
        if kv.get("node_dirty", "none") != "none":
            failures.append("%s: node_dirty=%s" % (cid, kv.get("node_dirty")))
    a, e = s_by_cid.get("ANCHOR"), s_by_cid.get(ANCHOR_END_ID)
    drift = None
    if not check_drift:                       # ANCHOR_END=0 / fused: no ANCHOREND, gate off
        pass
    elif not a or not e:
        failures.append("anchor drift: ANCHOR s=%s ANCHOREND s=%s (both required)" % (a, e))
    else:
        drift = e / a - 1.0
        if abs(drift) > drift_max:
            failures.append("anchor drift %+.2f%% exceeds +-%.1f%% (ANCHOR %.4fs, ANCHOREND %.4fs)"
                            % (100 * drift, 100 * drift_max, a, e))
    return dict(job_id=str(job_id), ok=not failures, failures=failures, anchor_drift=drift,
                s_anchor=a, s_anchor_end=e)


def job_timing(ctx: Ctx, job_ids: list) -> dict:
    """{job: {queue_s, run_s}} from sacct Submit/Start/End; None where sacct reports nothing."""
    ids = [str(j) for j in job_ids if j]
    out = {j: dict(queue_s=None, run_s=None) for j in ids}
    if not ids:
        return out
    try:
        text = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(ids), *sweep.SACCT_SINCE,
                                 "-X", "-n", "-P", "--format=JobID,Submit,Start,End"])
    except Exception:
        return out
    fmt = "%Y-%m-%dT%H:%M:%S"
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        jid = parts[0].split(".")[0]
        if jid not in out:
            continue
        try:
            sub, st = (datetime.datetime.strptime(parts[1], fmt),
                       datetime.datetime.strptime(parts[2], fmt))
        except ValueError:
            continue
        out[jid]["queue_s"] = (st - sub).total_seconds()
        try:
            out[jid]["run_s"] = (datetime.datetime.strptime(parts[3], fmt) - st).total_seconds()
        except ValueError:
            pass
    return out


def busy_s(ctx: Ctx, job_ids: list) -> float:
    """Seconds during which at least one of the jobs ran: the union of their sacct Start..End
    (the running clock samples at each poll, so jobs that start and end between two polls are
    missed by it)."""
    ids = [str(j) for j in job_ids if j]
    if not ids:
        return 0.0
    try:
        text = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(ids), *sweep.SACCT_SINCE,
                                 "-X", "-n", "-P", "--format=JobID,Start,End"])
    except Exception:
        return 0.0
    spans = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        try:
            spans.append(tuple(datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
                               for t in parts[1:3]))
        except ValueError:
            continue
    total, end = 0.0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total, end = total + (b - a).total_seconds(), b
        elif b > end:
            total, end = total + (b - end).total_seconds(), b
    return total


def cancel_in_flight(ctx: Ctx, job_ids: list) -> None:
    """Cancel those of `job_ids` that sacct still shows PENDING/RUNNING."""
    still_up = [j for j in job_ids if j not in ctx.cancelled_jobs]
    if still_up:
        states = sweep.sacct_states(still_up, sacct_runner=ctx.sacct_runner,
                                     sacct_cmd=ctx.sacct_cmd)
        cancel_jobs(ctx, [j for j in still_up
                          if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES])


def _final_cancel_safety_net(ctx: Ctx) -> None:
    """Past the budget or after a stop, cancel whatever owned job is still PENDING/RUNNING."""
    if ctx.elapsed() >= ctx.budget_s or STOP["reason"]:
        cancel_in_flight(ctx, ctx.owned_jobs)


def _maxrss_mb_by_job(ctx: Ctx, job_ids: list) -> dict:
    """{job: MaxRSS MB or None} from one sacct call over the jobs' steps; nothing is invented."""
    job_ids = [str(j) for j in job_ids]
    best: dict = {}
    try:
        out = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(job_ids), *sweep.SACCT_SINCE,
                                "-n", "-P", "--format=JobID,MaxRSS"])
    except Exception:
        out = ""
    for line in (out or "").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        m = _RSS_RE.match(parts[1].strip())
        if not m:
            continue
        kb = float(m.group(1)) * _RSS_TO_KB[m.group(2)]
        jid = parts[0].strip().split(".")[0]
        if kb > 0:
            best[jid] = max(best.get(jid, 0.0), kb)
    return {j: (best[j] / 1024.0 if j in best else None) for j in job_ids}
