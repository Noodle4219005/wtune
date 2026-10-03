"""Slurm side of the tuner: submit a CONFIG_FILE job, poll it under the wall or running-time
budget, cancel, read sacct timing / MaxRSS, and the per-job ranking validity gates."""
from __future__ import annotations

import datetime
import os
import re
from typing import Callable

from ._deps import sweep
from .budget import (_job_time_est, _ranking_wrap, _stage1_active, ranking_export,
                     stage2_concurrency, wrap_with_anchors)
from .common import ANCHOR_END_ID, _fmt_hms, _notes_kv, parse_run_id
from .context import Ctx, QueueCapExceeded

PROGRESS_BAR_WIDTH = 24                # characters of the progress bar
ANCHOR_DRIFT_MAX = 0.01                # |s(ANCHOREND)/s(ANCHOR) - 1| allowed inside one job
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
    print("budget deadline reached: cancelling %d in-flight job(s) owned by this run: %s"
          % (len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)


def wait_for_jobs(ctx: Ctx, job_ids: list, stage: str, note: str = "",
                  pending_cutoff: Callable[[], bool] | None = None,
                  exempt_budget: bool = False) -> dict:
    """Poll sacct until `job_ids` leave PENDING/RUNNING, printing one progress line per poll; at
    the budget deadline cancel those ids only. Returns the last {job: (State, Elapsed)} ({} for
    no ids). budget_mode "running" uses _wait_running_budget (pending_cutoff, exempt_budget)."""
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return {}
    if ctx.budget_mode == "running":
        return _wait_running_budget(ctx, job_ids, stage, note, pending_cutoff, exempt_budget)
    while True:
        elapsed = ctx.elapsed()
        states = sweep.sacct_states(job_ids, sacct_runner=ctx.sacct_runner, sacct_cmd=ctx.sacct_cmd)
        in_flight = [j for j in job_ids
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        print_progress(stage, elapsed, ctx.budget_s, note)
        if not in_flight:
            return states
        if elapsed >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)
            return states
        ctx.clock.sleep(min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed)))


def _wait_running_budget(ctx: Ctx, job_ids: list, stage: str, note: str,
                         pending_cutoff: Callable[[], bool] | None,
                         exempt_budget: bool = False) -> dict:
    """wait_for_jobs on the RUNNING-time budget (decision 0055 gap 3): a poll interval is charged
    only when one of our jobs (or ctx.background_jobs) runs, else counted as queue wait (capped by
    max_queue_s). Cancels at the budget (not when `exempt_budget`) or the wall cap; once
    `pending_cutoff()` is true, still-PENDING jobs are cancelled as not measured."""
    while True:
        elapsed = ctx.elapsed()
        wall = ctx.wall_elapsed()
        bg = [j for j in ctx.background_jobs if j not in job_ids]
        states = sweep.sacct_states(job_ids + bg, sacct_runner=ctx.sacct_runner,
                                    sacct_cmd=ctx.sacct_cmd)
        bg_running = any(states.get(j, ("", ""))[0] == "RUNNING" for j in bg)
        states = {j: v for j, v in states.items() if j not in bg}
        in_flight = [j for j in job_ids
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES
                     and j not in ctx.cancelled_jobs]
        print_progress(stage, elapsed, ctx.budget_s,
                       (note + " " if note else "") + "(running budget; wall %s, queue-only %s)"
                       % (_fmt_hms(wall), _fmt_hms(ctx.queue_wait_s)))
        if not in_flight:
            return states
        if pending_cutoff is not None and pending_cutoff():
            pend = [j for j in in_flight if states.get(j, ("", ""))[0] == "PENDING"]
            if pend:
                cancel_unstarted(ctx, pend, stage)
                in_flight = [j for j in in_flight if j not in pend]
                if not in_flight:
                    return states
        over_budget = elapsed >= ctx.budget_s and not exempt_budget
        if over_budget or (ctx.max_wall_s is not None and wall >= ctx.max_wall_s):
            if not over_budget:
                print("hard wall-clock cap reached (wall %s >= %s)"
                      % (_fmt_hms(wall), _fmt_hms(ctx.max_wall_s)))
            cancel_jobs(ctx, in_flight)
            return states
        n_running = sum(1 for j in in_flight if states.get(j, ("", ""))[0] == "RUNNING")
        if _stage1_active(ctx) and n_running > ctx.stage1_max_running:
            ctx.stage1_max_running = n_running
            print("%s: observed %d of this run's ranking job(s) RUNNING at once (stage-2 "
                  "concurrency now %d)" % (stage, n_running, stage2_concurrency(ctx)))
        running = n_running > 0 or bg_running
        step = min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed))
        if exempt_budget:
            step = ctx.poll_interval
        if ctx.max_wall_s is not None:
            step = min(step, max(0.0, ctx.max_wall_s - wall))
        if not running:            # nothing consumes budget: only the wall cap bounds the poll
            step = ctx.poll_interval if ctx.max_wall_s is None else \
                min(ctx.poll_interval, max(0.0, ctx.max_wall_s - wall))
        t0 = ctx.clock.now()
        ctx.clock.sleep(step)
        if running:
            ctx.add_running(ctx.clock.now() - t0)
        if not running:
            ctx.add_queue_wait(ctx.clock.now() - t0)
            if ctx.max_queue_s is not None and ctx.queue_wait_s >= ctx.max_queue_s:
                if ctx.shared_queue is not None:
                    ctx.shared_queue["cap_hit"] = True
                cancel_jobs(ctx, [j for j in in_flight if j not in ctx.cancelled_jobs])
                raise QueueCapExceeded(
                    "queue-only wait %s reached --max-queue-hours (%s); in-flight job(s) %s "
                    "cancelled" % (_fmt_hms(ctx.queue_wait_s), _fmt_hms(ctx.max_queue_s),
                                   ",".join(in_flight)))


def cancel_unstarted(ctx: Ctx, job_ids: list, stage: str) -> None:
    """Stage-1 cutoff: scancel jobs that never started; their candidates go to ctx.not_measured."""
    print("%s: stage-1 cutoff reached -- cancelling %d job(s) that never started: %s "
          "(their candidates are not measured)" % (stage, len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)
    for j in job_ids:
        cids = [c for c in ctx.job_cids.get(j, []) if c not in ("ANCHOR", ANCHOR_END_ID)]
        ctx.not_measured.append(dict(job_id=j, cids=cids, stage=stage,
                                     reason="job still PENDING at the stage-1 cutoff"))


def submit_configs_job(ctx: Ctx, configs: list[dict], reps: int, name_suffix: str,
                       s_est: float | None, time_est: dict | None = None,
                       ranking: dict | None = None) -> tuple[str, str, dict]:
    """Submit one strict_campaign.sbatch CONFIG_FILE job. Returns (job id, configs.json, time
    estimate). With a ranking context (default ctx.ranking) the job runs as the rebuilt point's
    case from its shared restart, configs wrapped in ANCHOR ... ANCHOREND; CAMPAIGN is exported
    for a real campaign checkout so the spooled script never pairs with stale helpers."""
    ranking = ctx.ranking if ranking is None else ranking
    if ranking:
        configs = wrap_with_anchors(configs, ranking["anchor"])
    elif _stage1_active(ctx):
        configs = _ranking_wrap(ctx, configs)        # stage1=t0: anchors in every job too
    if time_est is None:
        time_est = _job_time_est(ctx, len(configs), reps, s_est, ranking=ranking)
    job_out_dir = os.path.join(ctx.out_dir, name_suffix)
    os.makedirs(job_out_dir, exist_ok=True)
    configs_path = os.path.join(job_out_dir, "configs.json")
    sweep.write_configs_json(configs_path, configs)
    cmd_str = sweep.build_sbatch_command(
        ranking["case_id"] if ranking else ctx.case, ctx.nodes, ctx.cores_per_node,
        time_est["time_limit"], time_est["timeout_s"],
        reps, ranking["steps"] if ranking else ctx.steps, os.path.abspath(configs_path),
        job_out_dir, env_source=ctx.env_source, sbatch_script=ctx.sbatch_script,
        campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
        extra_export=ranking_export(ranking, time_est) if ranking else None)
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
    ctx.job_cids[job_id] = [c["id"] for c in configs]
    return job_id, configs_path, time_est


def evaluate_job_gates(job_id: str, rows: list[dict], s_by_cid: dict, expected_cids: list,
                       drift_max: float = ANCHOR_DRIFT_MAX, require_restart: bool = True,
                       check_drift: bool = True) -> dict:
    """Validity gates of ONE ranking job (decision 0055), never raising: every expected cid has a
    row that restarted (restart_used=1, unless not require_restart) on a clean node, and
    |s(ANCHOREND)/s(ANCHOR) - 1| <= drift_max (when check_drift). Returns {job_id, ok, failures,
    anchor_drift, s_anchor, s_anchor_end}."""
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


def _record_stage1_gates(ctx: Ctx, job_ids: list, rows: list[dict], label: str) -> None:
    """Predicted-mode stage 1: record (and warn about, never raise) every job's gates, plus its
    s_anchor_job = mean(ANCHOR, ANCHOREND) normalizer and whether foreign load or a missed restart
    excludes its candidates (ctx.ranking_gates, ctx.stage1_jobs). No-op outside stage 1."""
    if not _stage1_active(ctx):
        return
    require_restart = ctx.ranking is not None     # stage1=t0 jobs start at t0, not a restart
    for jid in job_ids:
        mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == str(jid)]
        cids = [parse_run_id(r["run_id"])["cid"] for r in mine]
        s_by_cid = {}
        for r in mine:
            s = sweep._to_float(r.get("s_per_step_median"))
            if s:
                s_by_cid[parse_run_id(r["run_id"])["cid"]] = s
        rec = evaluate_job_gates(jid, mine, s_by_cid, cids, require_restart=require_restart)
        reasons = []
        for r in mine:
            kv = _notes_kv(r.get("notes"))
            cid = parse_run_id(r["run_id"])["cid"]
            if kv.get("foreign_load") == "1":
                reasons.append("%s foreign_load=1 (%s)" % (cid, kv.get("foreign", "?")))
            if require_restart and kv.get("restart_used") != "1":
                reasons.append("%s restart_used=%s" % (cid, kv.get("restart_used", "missing")))
        a, e = s_by_cid.get("ANCHOR"), s_by_cid.get(ANCHOR_END_ID)
        s_anchor_job = ((a + e) / 2.0) if (a and e) else a
        if not s_anchor_job:
            reasons.append("no ANCHOR measurement to normalize by")
        rec.update(stage="stage1", label=label, s_anchor_job=s_anchor_job,
                   excluded=bool(reasons), exclude_reason="; ".join(reasons) or None)
        ctx.ranking_gates.append(rec)
        ctx.stage1_jobs[str(jid)] = dict(s_anchor_job=s_anchor_job, excluded=bool(reasons),
                                         reason=rec["exclude_reason"], label=label)
        if reasons:
            print("WARNING: %s job %s EXCLUDED from the stage-1 ranking: %s"
                  % (label, jid, rec["exclude_reason"]))
        elif not rec["ok"]:
            print("WARNING: %s job %s failed ranking gates (kept, anchor-normalized): %s"
                  % (label, jid, "; ".join(rec["failures"])))


def job_timing(ctx: Ctx, job_ids: list) -> dict:
    """{job: {queue_s, run_s}} from sacct Submit/Start/End; None where sacct reports nothing."""
    ids = [str(j) for j in job_ids if j]
    out = {j: dict(queue_s=None, run_s=None) for j in ids}
    if not ids:
        return out
    try:
        text = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(ids), "-X", "-n", "-P",
                                 "--format=JobID,Submit,Start,End"])
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


def _final_cancel_safety_net(ctx: Ctx) -> None:
    """Past the budget, cancel whatever owned job sacct still shows PENDING/RUNNING."""
    still_up = [j for j in ctx.owned_jobs if j not in ctx.cancelled_jobs]
    if still_up:
        states = sweep.sacct_states(still_up, sacct_runner=ctx.sacct_runner,
                                     sacct_cmd=ctx.sacct_cmd)
        in_flight = [j for j in still_up
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        if in_flight and ctx.elapsed() >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)


def _maxrss_mb_by_job(ctx: Ctx, job_ids: list) -> dict:
    """{job: MaxRSS MB or None} from one sacct call over the jobs' steps; nothing is invented."""
    job_ids = [str(j) for j in job_ids]
    best: dict = {}
    try:
        out = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(job_ids), "-n", "-P",
                                "--format=JobID,MaxRSS"])
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
