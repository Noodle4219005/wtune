"""Node-count selection: the uncalibrated proxy policies (--nodes auto) and the measured node
search (--nodes search, -O levels): a coarse-to-fine ANCHOR sweep over N, a cores-per-node
sweep at the finalists, the decomposition search at each finalist and the final choice."""
from __future__ import annotations

import dataclasses
import math
import os
import statistics
from typing import Any, Callable

import wtune.machine
import wtune.nodes
from ._deps import sweep, predict
from .budget import _fallback_s_per_step, _job_time_est
from .candidates import _sweep_anchor, _sweep_anchor_fitting, aggregate_by_cid, merge_aggs
from .common import (UNSET, Record, _fmt_hms, close_to_best, parse_run_id,
                     would_exceed_budget)
from .context import STOP, Ctx, QueueCapExceeded
from .jobs import MAXRSS_NOTE, _maxrss_mb_by_job, submit_configs_job
from . import jobs as _jobs
from . import memory as _memory
from . import prediction as _prediction
from . import report as _report
from . import rounds as _rounds

NODE_SWEEP_REPS = 2                   # replicates of each sweep ANCHOR job
DEFAULT_SWEEP_BUDGET_FRAC = 0.4       # share of --minutes the node sweep may use
EARLY_UNTRIED_FRAC = 0.15             # N never sent go first if they need <= this of what is left
KNEE_MARGINAL_SPEEDUP = 1.5           # policy=knee: stop doubling below this predicted speed-up


@dataclasses.dataclass
class FinalistOutcome:
    """One node-search finalist's tuned result (collect["outcomes"][N] for the pipeline)."""
    report: dict
    best_median: float
    ctx: Ctx
    all_results: dict
    confirm_results: dict | None
    cores: int
    anchor: dict | None


@dataclasses.dataclass
class NodeSelection(Record):
    """report["node_selection"]: the measured node table, the choice per objective, the cores
    sweep, the finalists' layouts and everything not tried (UNSET fields are absent: the
    proxy fallback carries only the first group plus `note`)."""
    mode: str = "search"
    objective: Any = UNSET
    chosen: Any = UNSET
    chosen_note: Any = UNSET
    final: Any = UNSET
    final_note: Any = UNSET
    final_cores_per_node: Any = UNSET
    table: Any = UNSET                    # wtune.nodes.table rows + maxrss_mb
    choices: Any = UNSET                  # {objective: {nodes, note}}
    skipped: Any = UNSET
    finalists: Any = UNSET
    finalist_best: Any = UNSET            # {"N": best tuned median s/step}
    core_tables: Any = UNSET              # {"N": cores-per-node rows}
    core_choice: Any = UNSET              # {"N": {cores, note}}
    skipped_cores: Any = UNSET
    core_options: Any = UNSET
    total_steps: Any = UNSET
    deadline_s: Any = UNSET
    min_efficiency: Any = UNSET
    overhead_s: Any = UNSET
    sweep_jobs: Any = UNSET
    maxrss_note: Any = UNSET
    layouts: Any = UNSET                  # build_layout_table rows, best first
    winner: Any = UNSET
    runner_up: Any = UNSET
    winner_vs_runner_up: Any = UNSET
    untried: Any = UNSET
    enumerated: Any = UNSET
    note: Any = UNSET


def _legal_rows(profile: dict, max_nodes: int) -> list[dict]:
    table = profile.get("core_count_priors", {}).get("efficiency_table", [])
    return sorted((r for r in table if r.get("legal") and r["nodes"] <= max_nodes),
                  key=lambda r: r["nodes"])


def choose_nodes_min(profile: dict, max_nodes: int) -> tuple[int, str]:
    """policy=min: the smallest legal profiled node count (no trust in the efficiency proxy)."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return 1, "policy=min: no legal node count in the profile table; defaulting to nodes=1"
    return rows[0]["nodes"], "policy=min: smallest legal profiled node count (cap=%d)" % max_nodes


def choose_nodes_cap(profile: dict, max_nodes: int) -> tuple[int, str]:
    """policy=cap: the largest legal profiled node count within --max-nodes."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return max_nodes, ("policy=cap: no legal node count in the profile table; defaulting to "
                            "--max-nodes=%d" % max_nodes)
    return rows[-1]["nodes"], "policy=cap: largest legal profiled node count within cap=%d" % max_nodes


def choose_nodes_knee(profile: dict, max_nodes: int) -> tuple[int, str]:
    """policy=knee (default): stop at the first legal node count whose next profile row
    (wtune.nodes.seed_counts) has a predicted speed-up n_j*E(n_j) / (n_i*E(n_i)) below 1.5x."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return 1, "policy=knee: no legal node count in the profile table; defaulting to nodes=1"
    for i in range(len(rows) - 1):
        n_i, e_i = rows[i]["nodes"], rows[i].get("efficiency_E") or 0.0
        n_j, e_j = rows[i + 1]["nodes"], rows[i + 1].get("efficiency_E") or 0.0
        speedup_i, speedup_j = n_i * e_i, n_j * e_j
        marginal = (speedup_j / speedup_i) if speedup_i > 0 else float("inf")
        if marginal < KNEE_MARGINAL_SPEEDUP:
            return n_i, ("policy=knee: doubling nodes=%d->%d has predicted marginal speedup "
                          "%.2fx < 1.5x; stopping at nodes=%d" % (n_i, n_j, marginal, n_i))
    n = rows[-1]["nodes"]
    if n < max_nodes:
        # say the cap was not reached, so nodes=n is not read as a verified stop below the cap
        return n, ("policy=knee: every doubling among the LEGAL profiled node counts (<=%d) "
                   "clears 1.5x; nodes=%d is the largest one wtune.case profiled as legal "
                   "(--max-nodes cap is %d, not reached)" % (n, n, max_nodes))
    return n, ("policy=knee: every profiled doubling step up to --max-nodes cap=%d clears 1.5x; "
               "using nodes=%d" % (max_nodes, n))


NODES_POLICIES = {"min": choose_nodes_min, "cap": choose_nodes_cap, "knee": choose_nodes_knee}


def choose_nodes(profile: dict, max_nodes: int, policy: str) -> tuple[int, str, str]:
    """(nodes, note, warning) by `policy`. The proxy (wtune.case's halo efficiency, c=1.0) is
    UNCALIBRATED, so the choice always carries that warning. SystemExit for an unknown policy."""
    fn = NODES_POLICIES.get(policy)
    if fn is None:
        raise SystemExit("unknown --nodes-policy %r (choices: %s)"
                          % (policy, ", ".join(sorted(NODES_POLICIES))))
    nodes, note = fn(profile, max_nodes)
    warning = ("scaling model is UNCALIBRATED (wtune/case.py's halo-efficiency proxy, c=1.0, "
               "never fit to a real measurement) -- nodes=%d (policy=%s) is a starting point, "
               "not a validated optimum" % (nodes, policy))
    return nodes, note, warning


def _pack_items(items: list[dict], max_nodes: int) -> list[list[dict]]:
    """First-fit ascending packing of jobs into concurrent waves of <= max_nodes nodes
    (wtune.nodes.sweep_waves' rule)."""
    waves: list[list[dict]] = []
    for it in sorted(items, key=lambda x: (x["nodes"], -x.get("cores", 0))):
        for w in waves:
            if sum(i["nodes"] for i in w) + it["nodes"] <= max_nodes:
                w.append(it)
                break
        else:
            waves.append([it])
    return waves


COLD_BUDGET_SAFETY = 3.0


def _cold_budget_s_per_step(ctx: Ctx, anchor: dict, s_est: float, src: str) -> tuple[float, str]:
    """s/step for the budget check of a first measurement. With no ledger row s_est is the
    10 s/step used for the job's time limit; charging that to a small --minutes skipped every N
    (sc26p1: 25 min estimated, 20 s real). Use the cell-step model x3 for the budget instead."""
    if src != "fallback_10s":
        return s_est, src
    try:
        nl = predict.parse_namelist(ctx.namelist_path)
        model = _fallback_s_per_step(nl, anchor["ranks"], anchor["omp"]) * COLD_BUDGET_SAFETY
    except Exception:                                   # noqa: BLE001
        return s_est, src
    return min(s_est, model), "cell-step model x%g" % COLD_BUDGET_SAFETY


def _sweep_steps(ctx: Ctx, s_est: float) -> int:
    """Steps of one sweep probe: about what one startup costs in time, within probe.sweep_steps
    (cheap steps -> the max); ctx.steps when that is unset. Sweep choices read the median only,
    so the window needs no radiation step."""
    span = (ctx.probe_cfg or {}).get("sweep_steps")
    if not span:
        return ctx.steps
    lo, hi = span
    return max(lo, min(hi, int(round(ctx.startup_s / max(s_est, 1e-6)))))


def _sweep_reps(ctx: Ctx) -> int:
    """Runs of each sweep probe (probe.sweep_reps); with one, close calls are run again."""
    return int((ctx.probe_cfg or {}).get("sweep_reps") or NODE_SWEEP_REPS)


def _measure_anchor_wave(ctx: Ctx, items: list[dict], s_est_for: Callable,
                         skipped: list, reps: int | None = None) -> dict:
    """One ANCHOR job of `reps` runs (default _sweep_reps) per item {key, nodes, anchor, suffix,
    tag} that fits the budget (the rest go to `skipped`, with the Slurm state of a failed job),
    awaited together. Returns {key: {agg, job, maxrss_mb}} for the items that produced a
    measurement."""
    reps = reps or _sweep_reps(ctx)
    jobs: dict = {}
    try:
        return _anchor_wave(ctx, items, s_est_for, skipped, reps, jobs)
    except QueueCapExceeded as e:            # keep the jobs that finished before the stop
        _jobs.cancel_in_flight(ctx, list(jobs.values()))
        states = sweep.sacct_states(list(jobs.values()), sacct_runner=ctx.sacct_runner,
                                    sacct_cmd=ctx.sacct_cmd)
        done = {k: j for k, j in jobs.items() if states.get(j, ("", ""))[0] == "COMPLETED"}
        e.partial = _anchor_results(ctx, items, done, states, [])
        raise


def _anchor_wave(ctx: Ctx, items: list[dict], s_est_for: Callable, skipped: list, reps: int,
                 jobs: dict) -> dict:
    for it in items:
        n = it["nodes"]
        s_est, src = s_est_for(n)
        cn = dataclasses.replace(ctx, nodes=n, steps=_sweep_steps(ctx, s_est))
        est = _job_time_est(cn, 1, reps, s_est)
        s_budget, src_budget = _cold_budget_s_per_step(ctx, it["anchor"], s_est, src)
        need_s = _job_time_est(cn, 1, reps, s_budget)["total_run_s"]
        if would_exceed_budget(ctx.elapsed(), need_s, ctx.budget_s):
            why = ("estimated end would exceed the budget (job ~%s at %.3fs/step[%s], elapsed %s "
                   "/ budget %s)" % (_fmt_hms(need_s), s_budget, src_budget,
                                     _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))
            skipped.append(dict(it["tag"], reason=why))
            print("node sweep: %s skipped -- %s" % (it["suffix"], why))
            continue
        job_id, _p, _te = submit_configs_job(cn, [it["anchor"]], reps=reps,
                                               name_suffix=it["suffix"], s_est=s_est, time_est=est)
        jobs[it["key"]] = job_id
        a = it["anchor"]
        print("node sweep: %s submitted job %s (ANCHOR ranks=%d omp=%d %dx%d, REPS=%d, "
              "%d steps, ~%s)"
              % (it["suffix"], job_id, a["ranks"], a["omp"], a["nx"], a["ny"], reps,
                 cn.steps, _fmt_hms(est["total_run_s"])))
    out: dict = {}
    if not jobs:
        return out
    states = _jobs.wait_for_jobs(ctx, list(jobs.values()), "node-sweep")
    return _anchor_results(ctx, items, jobs, states, skipped)


def _anchor_results(ctx: Ctx, items: list[dict], jobs: dict, states: dict,
                    skipped: list) -> dict:
    """{key: {agg, job, maxrss_mb}} of the wave's jobs with a ledger measurement; the others go
    to `skipped` with their Slurm state."""
    out: dict = {}
    ids = list(jobs.values())
    rows = ctx.ledger_reader(ctx.case, ids)
    rss = _maxrss_mb_by_job(ctx, ids)
    for it in items:
        j = jobs.get(it["key"])
        if j is None:
            continue
        mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == j]
        agg = aggregate_by_cid(mine, {"ANCHOR": it["anchor"]}).get("ANCHOR")
        if agg:
            out[it["key"]] = dict(agg=agg, job=j, maxrss_mb=rss.get(j))
        else:
            state = states.get(j, ("UNKNOWN", ""))[0]
            skipped.append(dict(it["tag"], state=state,
                                reason="no successful measurement (job %s, state %s)" % (j, state)))
    return out


def _race(ctx: Ctx, entries: dict, s_est_for: Callable, label: str) -> list:
    """Single-run sweeps: one more run of the entries {key: {nodes, anchor, agg, suffix}} that
    tie with the fastest (common.close_to_best), merged into their agg in place. Returns the
    jobs that measured."""
    if _sweep_reps(ctx) >= 2:
        return []
    keys = close_to_best({k: e["agg"]["median"] for k, e in entries.items()},
                         wtune.machine.measured("noise_frac"))
    if not keys:
        return []
    print("%s: measuring %s once more (within 2x noise of the fastest)" % (label, keys))
    items = [dict(key=k, nodes=entries[k]["nodes"], anchor=entries[k]["anchor"],
                  suffix=entries[k]["suffix"] + "-race", tag=dict(nodes=entries[k]["nodes"]))
             for k in keys]
    lost: list = []
    jobs = []
    for wave in _pack_items(items, ctx.max_nodes):
        for k, r in _measure_anchor_wave(ctx, wave, s_est_for, lost, reps=1).items():
            entries[k]["agg"] = merge_aggs(entries[k]["agg"], r["agg"])
            jobs.append(r["job"])
    for s in lost:
        print("%s: second run not measured -- %s" % (label, s["reason"]))
    return jobs


def _halt_reason(ctx: Ctx) -> str | None:
    """Why nothing more may be submitted -- the user stopped the tune, or --max-queue-hours was
    reached -- else None."""
    if STOP["reason"]:
        return STOP["reason"]
    if ctx.max_queue_s is not None and ctx.queue_wait_s >= ctx.max_queue_s:
        return "queue cap reached"
    return None


def run_node_sweep(ctx: Ctx, legal_nodes: list[int],
                   budget_frac: float = DEFAULT_SWEEP_BUDGET_FRAC,
                   chooser: Callable | None = None, on_wave: Callable | None = None,
                   refine: Callable[[], bool] = lambda: True, mem: Any = None) -> dict:
    """Coarse-to-fine sweep of one REPS=2 ANCHOR job per N with a legal ANCHOR (_sweep_anchor):
    the seeds (wtune.nodes.seed_counts) in ascending waves with an early stop upward, then
    the untested legal neighbours of the best N (`chooser`) until they are measured;
    `budget_frac` of what is left caps it. An N whose full nodes have no legal decomposition
    gets its ANCHOR on the most cores per node that have one (`cores`). `mem` is the memory
    model of an earlier sweep of the same case. Returns {measurements, anchors, cores, results,
    skipped (with reasons), jobs, legal, maxrss_mb, memory, race_jobs, untried (legal N never
    sent to a wave and not ruled out by memory), wave_s (running time per wave)}."""
    chooser = chooser or (lambda rows: wtune.nodes.choose(rows, "time")[0])
    skipped, anchors, cores = [], {}, {}
    for n in wtune.nodes.candidate_node_counts(legal_nodes, ctx.max_nodes):
        anchor, c, why = _sweep_anchor_fitting(ctx, n)
        if anchor is None:
            skipped.append(dict(nodes=n, reason=why))
            print("node sweep: N=%d skipped -- %s" % (n, why))
            continue
        anchors[n], cores[n] = anchor, c
        if c < ctx.cores_per_node:
            print("node sweep: N=%d has no legal decomposition on all %d cores/node; ANCHOR "
                  "on %d (%d ranks x omp %d)"
                  % (n, ctx.cores_per_node, c, anchor["ranks"], anchor["omp"]))
    h = ctx.node_health or {}
    for n in range(ctx.max_nodes + 1, int(h.get("max_nodes_requested") or 0) + 1):
        skipped.append(dict(nodes=n, reason="skipped (only %d healthy nodes)" % h["healthy"]))
    legal = sorted(anchors)
    meas: dict[int, list] = {}
    results: dict[int, dict] = {}
    jobs: dict[int, str] = {}
    rss: dict[int, Any] = {}
    attempted: set = set()
    sent: set = set()
    unfit: set = set()
    every_n = ctx.node_seeds == "all"
    seeds = legal if every_n else (wtune.nodes.seed_nodes(legal, ctx.max_nodes) if legal else [])
    sweep_t0 = ctx.elapsed()
    cap_s = budget_frac * max(0.0, ctx.budget_s - sweep_t0)   # share of what is left
    state: dict[str, Any] = dict(stopped=None, waves=0)
    if mem is None:
        mem = _memory.model_for(ctx.case_json, wtune.machine.node_memory_gib())
    waves = wtune.nodes.sweep_waves(seeds, ctx.max_nodes)
    descending = bool(mem and legal
                      and mem.why_not(legal[0], anchors[legal[0]]["ranks"] // legal[0]))
    if descending:                    # the largest N fits best, and its MaxRSS calibrates the rest
        waves = waves[::-1]
        print("node sweep: N=%d may not fit in memory; largest N first" % legal[0])

    def s_est_for(n):
        if meas:
            near = min(meas, key=lambda m: (abs(m - n), m))
            return statistics.median(meas[near]), "median at N=%d" % near
        s_est, src, _k = sweep.estimate_s_per_step(ctx.case, n, ctx.ledger_dir, ctx.hardware_id)
        return s_est, src

    def cap_reason():
        if ctx.elapsed() - sweep_t0 >= cap_s:
            return ("sweep budget cap: sweep used %s >= %.0f%% of the budget (%s)"
                    % (_fmt_hms(ctx.elapsed() - sweep_t0), budget_frac * 100, _fmt_hms(cap_s)))
        return None

    def job_s(n):
        s, _src = s_est_for(n)
        if meas:                      # s_est_for is the nearest N's median: scale it to n
            near = min(meas, key=lambda m: (abs(m - n), m))
            s = statistics.median(meas[near]) * near / n
        cn = dataclasses.replace(ctx, nodes=n, steps=_sweep_steps(ctx, s))
        return _job_time_est(cn, 1, _sweep_reps(ctx), s)["total_run_s"]

    def fit_memory(n):
        """True when N should fit: its ANCHOR as is, or on fewer cores per node (anchors and
        cores updated); False, with the reason in `skipped`, when no core count fits."""
        why = mem.why_not(n, anchors[n]["ranks"] // n) if mem else None
        if why is None:
            return True
        for c in wtune.nodes.default_core_options(cores[n]):
            a = _sweep_anchor(ctx, n, cores=c)[0] if c < cores[n] else None
            if a is not None and mem.why_not(n, a["ranks"] // n) is None:
                print("node sweep: N=%d on %d cores/node -- %s on %d" % (n, c, why, cores[n]))
                anchors[n], cores[n] = a, c
                return True
        skipped.append(dict(nodes=n, reason=why))
        attempted.add(n)
        unfit.add(n)
        print("node sweep: N=%d skipped -- %s" % (n, why))
        return False

    def fill(wave):
        """`wave` plus untried legal N in the nodes it leaves idle, nearest the best N first (the
        predicted fastest before any measurement), none expected to outlast the wave."""
        if not ctx.hotspot_fill or every_n:
            return wave
        free, longest = ctx.max_nodes - sum(wave), max(job_s(n) for n in wave)
        pool = [n for n in legal if n not in attempted and n not in wave and n not in seeds]
        if meas:
            best = chooser(wtune.nodes.summarize(meas))
            pool.sort(key=lambda n: (abs(n - best), n))
        else:
            pool.sort(key=lambda n: (s_est_for(n)[0], -n))
        extra = []
        for n in pool:
            if (n <= free and job_s(n) <= longest
                    and not (mem and mem.why_not(n, anchors[n]["ranks"] // n))):
                extra.append(n)
                free -= n
        if extra:
            print("node sweep: hotspot fill: N=%s in the idle nodes of wave %s" % (extra, wave))
        return wave + extra

    def run_wave(ns):
        ns = [n for n in ns if fit_memory(n)]
        items = [dict(key=n, nodes=n, anchor=anchors[n], suffix="nodes-n%d" % n, tag=dict(nodes=n))
                 for n in ns]
        attempted.update(ns)
        sent.update(ns)
        state["waves"] += 1
        first_new = len(skipped)
        halted = None
        try:
            got = _measure_anchor_wave(ctx, items, s_est_for, skipped)
        except QueueCapExceeded as e:        # the wave's jobs that finished still count
            got, halted = e.partial, e
        for n, r in got.items():
            meas[n], results[n], jobs[n], rss[n] = (r["agg"]["s_values"], r["agg"], r["job"],
                                                    r["maxrss_mb"])
            if mem:
                mem.learn(n, anchors[n]["ranks"] // n, r["maxrss_mb"])
        for s in skipped[first_new:]:
            if mem and s.get("state") == "OUT_OF_MEMORY":
                mem.learn_oom(s["nodes"], anchors[s["nodes"]]["ranks"] // s["nodes"])
        if halted:
            if not meas:
                raise halted
            print("node sweep: %s; keeping the N measured so far" % halted)
            state["stopped"] = state["halt"] = str(halted)
            skipped.extend(dict(nodes=n, reason=str(halted)) for n in ns if n not in got)

    # phase 1: geometric seeds, early stop on the upward side
    for wave in waves:
        if not state["stopped"]:
            state["stopped"] = cap_reason()
            if state["stopped"]:
                print("node sweep: %s; larger N not submitted" % state["stopped"])
        if state["stopped"]:
            attempted.update(wave)
            skipped.extend(dict(nodes=n, reason=state["stopped"]) for n in wave)
            continue
        run_wave(fill(wave))
        if state.get("halt"):
            continue
        if on_wave is not None:
            halt = on_wave(meas)
            if halt:                  # e.g. the short-run guard: no more waves, no refinement
                state["stopped"] = state["halt"] = halt
                continue
        srows = wtune.nodes.summarize({n: v for n, v in meas.items() if n in seeds})
        if not every_n and not descending and wtune.nodes.stop_early(srows):
            a, b = srows[-2], srows[-1]
            state["stopped"] = ("early stop: N=%d (%.4f s/step) did not beat N=%d (%.4f s/step) "
                                "by >%.0f%%" % (b["nodes"], b["median"], a["nodes"], a["median"],
                                                wtune.machine.measured("noise_frac") * 100))
            print("node sweep: %s; larger N not submitted" % state["stopped"])

    # phase 2: refine around the best N
    cap_hit = state.get("halt")
    while meas and not state.get("halt") and refine():
        best = chooser(wtune.nodes.summarize(meas))
        todo = wtune.nodes.refine_targets(best, set(seeds) | set(meas), legal, attempted)
        if not todo:
            break
        cap_hit = cap_reason()
        if cap_hit:
            print("node sweep: %s; refinement stopped" % cap_hit)
            break
        print("node sweep: refining around best N=%d: measuring N=%s" % (best, todo))
        for wave in wtune.nodes.sweep_waves(todo, ctx.max_nodes):
            run_wave(fill(wave))
            if state.get("halt"):
                break

    race_jobs = []
    if not state.get("halt"):
        entries = {n: dict(nodes=n, anchor=anchors[n], agg=results[n], suffix="nodes-n%d" % n)
                   for n in results}
        race_jobs = _race(ctx, entries, s_est_for, "node sweep")
        for n, e in entries.items():
            results[n], meas[n] = e["agg"], e["agg"]["s_values"]

    _explain_unmeasured(legal, meas, seeds, skipped, cap_hit or state.get("halt"), refine,
                        chooser)
    return dict(measurements=meas, anchors=anchors, cores=cores, results=results,
                skipped=skipped, jobs=jobs, legal=legal, maxrss_mb=rss, memory=mem,
                race_jobs=race_jobs, untried=[n for n in legal if n not in sent | unfit],
                wave_s=(ctx.elapsed() - sweep_t0) / max(1, state["waves"]))


def _explain_unmeasured(legal: list[int], meas: dict, seeds: list[int], skipped: list,
                        cap_hit: Any, refine: Callable[[], bool], chooser: Callable) -> None:
    """Give every legal N that was never measured an explicit reason in `skipped` (sorted)."""
    rows = wtune.nodes.summarize(meas)
    best = chooser(rows) if rows else None
    lo, hi = wtune.nodes.bracket(best, set(seeds) | set(meas)) if best else (0, None)
    already = {s["nodes"] for s in skipped}
    for n in legal:
        if n in meas or n in already:
            continue
        if cap_hit:
            why = cap_hit
        elif not refine():
            why = "not measured: refinement is off at this level (O1 measures only {1,2,4,8,max})"
        elif best is None:
            why = "not measured: no seed measurement succeeded"
        elif lo < n and (hi is None or n < hi):
            why = ("not measured: not adjacent to the best N=%d (inside the refinement bracket "
                   "%d..%s)" % (best, lo, hi))
        else:
            why = ("not measured: outside refinement bracket (%d..%s around the best N=%d)"
                   % (lo, hi, best))
        skipped.append(dict(nodes=n, reason=why))
    skipped.sort(key=lambda s: (s["nodes"], s.get("cores", 0)))


def _core_row(cores: int, nodes: int, anchor: dict, agg: dict, maxrss_mb: float | None,
              chosen: bool = False) -> dict:
    return dict(cores=cores, ranks_per_node=anchor["ranks"] // nodes, omp=anchor["omp"],
                ranks=anchor["ranks"], nx=anchor["nx"], ny=anchor["ny"], median=agg["median"],
                n=agg["n"], spread=((agg["hi"] - agg["lo"]) / agg["median"]
                                    if agg["n"] >= 2 and agg["median"] else 0.0),
                maxrss_mb=maxrss_mb, chosen=chosen)


def _sweep_cores(ctx: Ctx, sw: dict, n: int) -> int:
    """Cores per node of the sweep's ANCHOR at N: all of them unless none was legal there."""
    return (sw.get("cores") or {}).get(n, ctx.cores_per_node)


def run_core_sweep(ctx: Ctx, finals: list[int], sw: dict, core_options: list[int],
                   budget_frac: float = DEFAULT_SWEEP_BUDGET_FRAC / 2) -> tuple[dict, list]:
    """Cores per node at each finalist N: one REPS=2 hybrid-ANCHOR job per legal (N, c < C)
    (the rest of the node idles; c = C reuses the sweep's measurement; C is the sweep's own
    core count at N, and where that is not the whole node its usual fractions are tried too), c
    chosen by
    wtune.nodes.choose_cores (ties -> fuller c). Returns ({N: {anchors, results, rss, rows,
    chosen, note}}, skipped_cores)."""
    info, skipped, items = {}, [], []
    mem = sw.get("memory")
    for n in finals:
        C = _sweep_cores(ctx, sw, n)
        info[n] = dict(anchors={C: sw["anchors"][n]}, results={C: sw["results"][n]},
                       rss={C: sw["maxrss_mb"].get(n)})
        options = set(core_options)
        if C < ctx.cores_per_node:      # the node was too big for the grid: fractions of C
            options |= set(wtune.nodes.default_core_options(C))
        for c in sorted((c for c in options if c < C), reverse=True):
            a, why = _sweep_anchor(ctx, n, cores=c)
            if a is not None and mem:
                why = mem.why_not(n, a["ranks"] // n)
                a = None if why else a
            if a is None:
                skipped.append(dict(nodes=n, cores=c, reason=why))
                print("core sweep: N=%d c=%d skipped -- %s" % (n, c, why))
                continue
            info[n]["anchors"][c] = a
            items.append(dict(key=(n, c), nodes=n, cores=c, anchor=a,
                              suffix="cores-n%d-c%d" % (n, c), tag=dict(nodes=n, cores=c)))
    t0 = ctx.elapsed()
    cap_s = budget_frac * max(0.0, ctx.budget_s - t0)

    def s_est_for(n):
        return statistics.median(sw["measurements"][n]), "node-sweep median at N=%d" % n

    for wave in _pack_items(items, ctx.max_nodes):
        if ctx.elapsed() - t0 >= cap_s:
            why = "core sweep budget cap: used %s >= %.0f%% of the budget" % (
                _fmt_hms(ctx.elapsed() - t0), budget_frac * 100)
            skipped.extend(dict(it["tag"], reason=why) for it in wave)
            continue
        for key, r in _measure_anchor_wave(ctx, wave, s_est_for, skipped).items():
            info[key[0]]["results"][key[1]] = r["agg"]
            info[key[0]]["rss"][key[1]] = r["maxrss_mb"]
    for n in finals:
        entries = {c: dict(nodes=n, anchor=info[n]["anchors"][c], agg=agg,
                           suffix="cores-n%d-c%d" % (n, c))
                   for c, agg in info[n]["results"].items()}
        _race(ctx, entries, s_est_for, "core sweep N=%d" % n)
        for c, e in entries.items():
            info[n]["results"][c] = e["agg"]
    for n in finals:
        rows = [_core_row(c, n, info[n]["anchors"][c], info[n]["results"][c], info[n]["rss"][c])
                for c in sorted(info[n]["results"], reverse=True)]
        chosen, note = wtune.nodes.choose_cores(rows)
        for r in rows:
            r["chosen"] = (r["cores"] == chosen)
        info[n].update(rows=rows, chosen=chosen, note=note)
        print("core sweep: N=%d chose cores/node=%d -- %s" % (n, chosen, note))
    skipped.sort(key=lambda s: (s["nodes"], -s["cores"]))
    return info, skipped


def node_choices(rows: list[dict], *, deadline_s: float | None, total_steps: int | None,
                 overhead_s: float, min_efficiency: float) -> dict:
    """{objective: {nodes, note}} for all three objectives (cost needs total_steps)."""
    out = {}
    for obj in wtune.nodes.OBJECTIVES:
        try:
            n, note = wtune.nodes.choose(rows, obj, deadline_s=deadline_s, total_steps=total_steps,
                                         overhead_s=overhead_s, min_efficiency=min_efficiency)
            out[obj] = dict(nodes=n, note=note)
        except ValueError as e:
            out[obj] = dict(nodes=None, note="unavailable: %s" % e)
    return out


def _proxy_fallback_report(ctx: Ctx, profile: dict, objective: str, nodes_policy: str,
                           sw: dict) -> dict:
    """No sweep measurement: tune at the proxy's node count and report that."""
    nodes, nodes_note, warning = choose_nodes(profile, ctx.max_nodes, nodes_policy)
    print("WARNING: node sweep produced no measurement; falling back to the proxy. " + warning)
    fctx = dataclasses.replace(ctx, nodes=nodes)
    res = _rounds.run_tuning_at(fctx)
    report = _report.build_report(ctx, nodes, "node sweep failed; " + nodes_note, *res)
    report["node_selection"] = NodeSelection(
        objective=objective, chosen=nodes, table=[], choices={}, skipped=sw["skipped"],
        sweep_jobs=[], finalists=[nodes], note=nodes_note).to_dict()
    return report


def _tune_finalists(ctx: Ctx, finals: list[int], cores_info: dict, sw: dict, plan: dict | None,
                    lvl: Callable[[], int], layout_notes: dict) -> dict:
    """round0/adaptive/confirm at each finalist (N, chosen c), the remaining budget split evenly
    over the finalists still to run. Returns {N: FinalistOutcome} of those with a best config."""
    outcomes = {}
    for i, n in enumerate(finals):
        remaining = ctx.budget_s - ctx.elapsed()
        if _halt_reason(ctx):
            print("node search: %s; finalist N=%d not tuned" % (_halt_reason(ctx), n))
            continue
        if remaining <= 0:
            print("node search: no budget left for finalist N=%d" % n)
            continue
        share = remaining / (len(finals) - i)
        c = cores_info[n]["chosen"] if n in cores_info else _sweep_cores(ctx, sw, n)
        anchor = cores_info[n]["anchors"][c] if n in cores_info else sw["anchors"].get(n)
        fctx = dataclasses.replace(ctx, nodes=n, budget_s=ctx.elapsed() + share,
                                    out_dir=os.path.join(ctx.out_dir, "n%d" % n),
                                    anchor_override=anchor, core_budget=c,
                                    full_space=(plan is not None and lvl() >= 3), layout_notes=layout_notes)
        os.makedirs(fctx.out_dir, exist_ok=True)
        print("node search: finalist N=%d cores/node=%d, budget share %s"
              % (n, c, _fmt_hms(share)))
        try:
            all_results, confirm_results, anchor_cid = _rounds.run_tuning_at(
                fctx, adaptive=(lvl() >= 3), confirm=(lvl() >= 3))  # O3: full space + confirm reps
        except QueueCapExceeded as e:
            print("node search: %s" % e)
            all_results, confirm_results, anchor_cid = {}, None, None
        if not all_results:
            res = (cores_info[n]["results"][c] if n in cores_info else sw["results"].get(n))
            if res:
                print("node search: nothing measured at N=%d beyond the sweep; reporting the "
                      "sweep ANCHOR" % n)
                all_results, anchor_cid = {"ANCHOR": dict(res, config=anchor)}, "ANCHOR"
        if not all_results:
            continue
        rep = _report.build_report(fctx, n, "node search finalist", all_results, confirm_results,
                                   anchor_cid, full_prediction=False)   # built once, for the winner
        if rep["best"]:
            outcomes[n] = FinalistOutcome(report=rep, best_median=rep["best"]["median_s_per_step"],
                                          ctx=fctx, all_results=all_results,
                                          confirm_results=confirm_results, cores=c, anchor=anchor)
    return outcomes


def _sweep_anchor_outcome(ctx: Ctx, chosen: int, note: str, cores_info: dict,
                          sw: dict) -> FinalistOutcome:
    """No finalist produced a result (budget gone, or O1): the sweep ANCHOR at `chosen`."""
    c = cores_info[chosen]["chosen"] if chosen in cores_info else _sweep_cores(ctx, sw, chosen)
    anchor = cores_info[chosen]["anchors"][c] if chosen in cores_info else sw["anchors"][chosen]
    res = cores_info[chosen]["results"][c] if chosen in cores_info else sw["results"][chosen]
    fctx = dataclasses.replace(ctx, nodes=chosen, core_budget=c)
    rep = _report.build_report(fctx, chosen, note, {"ANCHOR": dict(res, config=anchor)}, None,
                               "ANCHOR")
    return FinalistOutcome(report=rep, best_median=res["median"], cores=c, anchor=anchor,
                           ctx=fctx, all_results={"ANCHOR": dict(res, config=anchor)},
                           confirm_results=None)


def _measure_untried(ctx: Ctx, sw: dict, chooser: Callable, budget_frac: float) -> list[int]:
    """The running time the finalists left under a time cap: an ANCHOR at every legal N the sweep
    never sent (early stop, outside the refinement bracket), merged into `sw`. Returns the N
    measured; a queue cap reached here keeps what was measured before."""
    todo = sw["untried"]
    print("node sweep: %s of %s used; measuring the N never sent: %s"
          % (_fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s), todo))
    try:
        more = run_node_sweep(dataclasses.replace(ctx, node_seeds="all"), todo,
                              budget_frac=budget_frac, chooser=chooser, refine=lambda: False,
                              mem=sw["memory"])
    except QueueCapExceeded as e:
        print("node sweep: %s; keeping the result so far" % e)
        return []
    for k in ("measurements", "results", "jobs", "maxrss_mb"):
        sw[k].update(more[k])
    sw["race_jobs"] = sw.get("race_jobs", []) + more["race_jobs"]
    sw["skipped"] = sorted([s for s in sw["skipped"] if s["nodes"] not in todo]
                           + [s for s in more["skipped"] if s["nodes"] in todo],
                           key=lambda s: (s["nodes"], s.get("cores", 0)))
    sw["untried"] = []
    return sorted(more["measurements"])


def run_node_search(ctx: Ctx, profile: dict, objective: str, deadline_s: float | None,
                    min_efficiency: float, k_finalists: int, total_steps: int | None,
                    nodes_policy: str = "knee",
                    sweep_budget_frac: float = DEFAULT_SWEEP_BUDGET_FRAC,
                    core_options: list[int] | None = None, on_sweep: Callable | None = None,
                    collect: dict | None = None, on_wave: Callable | None = None,
                    plan: dict | None = None) -> dict:
    """Node sweep -> N by `objective` -> cores-per-node sweep at the finalists -> decomposition
    search at each finalist (N, c) -> final (N, c, config) over the finalists' best tuned
    medians. `plan` ({"level"}) is read live; `collect` receives outcomes / final_n / sweep /
    cores_info for the pipeline. Falls back to the proxy (`profile`) when nothing was measured."""
    lvl = (lambda: plan["level"]) if plan else (lambda: 3)    # effort level, read live
    overhead_s = ctx.startup_s
    kw = dict(deadline_s=deadline_s, total_steps=total_steps, overhead_s=overhead_s,
              min_efficiency=min_efficiency)

    def chooser(rows):
        return wtune.nodes.choose(rows, objective, **kw)[0]

    def summary():
        rows = wtune.nodes.summarize(sw["measurements"])
        chosen, note = wtune.nodes.choose(rows, objective, **kw)
        tab = wtune.nodes.table(rows, total_steps=total_steps, overhead_s=overhead_s)
        for r in tab:
            r["maxrss_mb"] = sw["maxrss_mb"].get(r["nodes"])
        return rows, node_choices(rows, **kw), chosen, note, tab

    def print_rows(tab, only=None):
        for r in tab:
            if only is None or r["nodes"] in only:
                print("node sweep: N=%-2d median=%.4f s/step n=%d spread=%.1f%% speedup=%.2f "
                      "eff=%.2f" % (r["nodes"], r["median"], r["n"], r["spread"] * 100,
                                    r["speedup"], r["efficiency"]))

    sw = run_node_sweep(ctx, list(range(1, ctx.max_nodes + 1)), budget_frac=sweep_budget_frac,
                        chooser=chooser, on_wave=on_wave, refine=lambda: lvl() >= 2)
    if not wtune.nodes.summarize(sw["measurements"]):
        return _proxy_fallback_report(ctx, profile, objective, nodes_policy, sw)
    if (lvl() >= 2 and sw["untried"] and not _halt_reason(ctx)
            and ctx.elapsed() < ctx.budget_s < math.inf):
        need = len(wtune.nodes.sweep_waves(sw["untried"], ctx.max_nodes)) * sw["wave_s"]
        left = ctx.budget_s - ctx.elapsed()
        early = need <= EARLY_UNTRIED_FRAC * left
        print("node sweep: the N never sent need ~%s, %s %.0f%% of the %s left; %s"
              % (_fmt_hms(need), "<=" if early else ">", EARLY_UNTRIED_FRAC * 100, _fmt_hms(left),
                 "measuring them before the finalists" if early else "they wait for leftover time"))
        if early:
            _measure_untried(ctx, sw, chooser, sweep_budget_frac)

    rows, choices, chosen, note, tab = summary()
    finals = wtune.nodes.finalists(rows, objective, k=(1 if lvl() == 2 else k_finalists), **kw)
    if on_sweep is not None:        # --pipeline full: re-estimate the budget from the measurement
        on_sweep(rows, sw)
    print("node sweep: chose N=%d -- %s; finalists %s" % (chosen, note, finals))
    print_rows(tab)

    if core_options is None:
        core_options = wtune.nodes.default_core_options(ctx.cores_per_node)
    cores_info: dict = {}
    skipped_cores: list = []
    if lvl() <= 1 or _halt_reason(ctx):  # O1: the t0 wave only -- no cores sweep, no finalists
        finals = []
    if lvl() >= 2 and any(c < ctx.cores_per_node for c in core_options):
        try:
            cores_info, skipped_cores = run_core_sweep(ctx, finals, sw, core_options)
        except QueueCapExceeded as e:
            print("node search: %s" % e)

    layout_notes: dict = dict(untried=[], enumerated={})
    outcomes = _tune_finalists(ctx, finals, cores_info, sw, plan, lvl, layout_notes)
    if (lvl() >= 2 and sw["untried"] and not _halt_reason(ctx)
            and ctx.elapsed() < ctx.budget_s < math.inf):
        new = _measure_untried(ctx, sw, chooser, sweep_budget_frac)
        if new:
            rows, choices, chosen, note, tab = summary()
            print_rows(tab, only=new)
        if chosen in new:
            finals = finals + [chosen]
            print("node search: N=%d is now the fastest ANCHOR (%s); tuning it with the %s left"
                  % (chosen, note, _fmt_hms(max(0.0, ctx.budget_s - ctx.elapsed()))))
            try:
                if any(c < ctx.cores_per_node for c in core_options):
                    more, more_skipped = run_core_sweep(ctx, [chosen], sw, core_options)
                    cores_info.update(more)
                    skipped_cores += more_skipped
                outcomes.update(_tune_finalists(ctx, [chosen], cores_info, sw, plan, lvl,
                                                layout_notes))
            except QueueCapExceeded as e:
                print("node search: %s; N=%d keeps its sweep ANCHOR" % (e, chosen))
            if chosen not in outcomes:
                outcomes[chosen] = _sweep_anchor_outcome(ctx, chosen, note, cores_info, sw)
    if not outcomes:
        outcomes[chosen] = _sweep_anchor_outcome(ctx, chosen, note, cores_info, sw)
        final_n, final_note = chosen, note + (
            "; level O1: t0 sweep only -- ANCHOR config" if lvl() <= 1 else
            "; %s -- ANCHOR config" % _halt_reason(ctx) if _halt_reason(ctx) else
            "; budget exhausted after the sweep -- ANCHOR config")
    else:
        by_sweep = {r["nodes"]: r for r in rows}
        frows = wtune.nodes.summarize({n: [o.best_median] for n, o in outcomes.items()})
        for r in frows:                     # scaling figures stay the sweep's (anchor-based)
            r["speedup"], r["efficiency"] = (by_sweep[r["nodes"]]["speedup"],
                                              by_sweep[r["nodes"]]["efficiency"])
        final_n, final_note = wtune.nodes.choose(frows, objective, **kw)
        final_note = "final over tuned finalists %s: %s" % (sorted(outcomes), final_note)
    if collect is not None:
        collect.update(outcomes=outcomes, final_n=final_n, sweep=sw, cores_info=cores_info)
    report = outcomes[final_n].report
    final_c = outcomes[final_n].cores
    if report.get("full_run_prediction") is None:       # winner of a finalist search
        o = outcomes[final_n]
        best_id = report["best"]["id"]
        entry = (o.confirm_results or {}).get(best_id) or o.all_results.get(best_id)
        report["full_run_prediction"] = _prediction.build_full_run_prediction(
            o.ctx, entry["config"] if entry else None, final_n, o.all_results,
            o.confirm_results)
    report["nodes"], report["nodes_note"] = final_n, final_note
    report["cores_used_per_node"] = final_c
    report["budget_minutes"] = ctx.budget_s / 60.0 if math.isfinite(ctx.budget_s) else None
    report["elapsed_minutes"] = ctx.elapsed() / 60.0
    report["job_ids"] = list(ctx.owned_jobs)
    report["cancelled_job_ids"] = list(ctx.cancelled_jobs)
    o = outcomes[final_n]
    best = (o.confirm_results or {}).get(report["best"]["id"]) or \
        o.all_results.get(report["best"]["id"])
    layouts_tab = build_layout_table(
        sw, cores_info, outcomes, ctx,
        best=((final_n, final_c, sweep._identity(best["config"]))
              if best and best.get("config") else None))
    report["layout_table"] = layouts_tab
    report["node_selection"] = NodeSelection(
        objective=objective, chosen=chosen, chosen_note=note, final=final_n,
        final_note=final_note, final_cores_per_node=final_c, table=tab, choices=choices,
        skipped=sw["skipped"], finalists=finals,
        finalist_best={str(n): o.best_median for n, o in outcomes.items()},
        core_tables={str(n): ci["rows"] for n, ci in cores_info.items()},
        core_choice={str(n): dict(cores=ci["chosen"], note=ci["note"])
                     for n, ci in cores_info.items()},
        skipped_cores=skipped_cores, core_options=list(core_options),
        total_steps=total_steps, deadline_s=deadline_s, min_efficiency=min_efficiency,
        overhead_s=overhead_s, sweep_jobs=list(sw["jobs"].values()) + sw.get("race_jobs", []), maxrss_note=MAXRSS_NOTE,
        layouts=layouts_tab["rows"], winner=layouts_tab["winner"],
        runner_up=layouts_tab["runner_up"], winner_vs_runner_up=layouts_tab["comparison"],
        untried=(sw["skipped"] + skipped_cores + layout_notes["untried"]),
        enumerated=layout_notes["enumerated"]).to_dict()
    return report


def _layout_row(source: str, nodes: int, cores: int, cfg: dict, vals: list[float],
                dirty: list | None = None) -> dict:
    med = statistics.median(vals)
    row = dict(source=source, nodes=nodes, cores_per_node=cores, ranks=cfg.get("ranks"),
               omp=cfg.get("omp"), nx=cfg.get("nx"), ny=cfg.get("ny"),
               numtiles=cfg.get("numtiles"), dist=cfg.get("dist"), median=med, reps=len(vals),
               spread=((max(vals) - min(vals)) / med if len(vals) >= 2 and med else 0.0),
               cid=cfg.get("id"))
    if dirty:                         # measured on a node that was not clean
        row["dirty"] = dirty
    return row


def build_layout_table(sw: dict, cores_info: dict, outcomes: dict, ctx: Ctx,
                       best: tuple | None = None) -> dict:
    """Every MEASURED layout best first (sweep ANCHORs, cores/node ANCHORs, finalist layouts with
    confirm reps winning), the winner over the finalists' layouts, the runner-up (next DIFFERENT
    layout) and whether they differ beyond noise: {rows, winner, runner_up, comparison}. The
    winner pool is what the report ranks (a finalist's confirm reps when it has any, never mixed
    with its 1-rep rows); `best` = (nodes, cores, scc_sweep._identity) of the report's BEST is
    the winner, so WINNER and BEST name one layout; without it, the pool minimum."""
    seen: dict = {}

    def add(row, key):
        old = seen.get(key)
        if old is None or row["reps"] > old["reps"] or (row["reps"] == old["reps"]
                                                       and row["median"] < old["median"]):
            seen[key] = row

    for n, agg in sw["results"].items():
        a, C = sw["anchors"][n], _sweep_cores(ctx, sw, n)
        add(_layout_row("t0 sweep ANCHOR", n, C, a, agg["s_values"], agg.get("dirty")),
            (n, C, sweep._identity(a)))
    for n, ci in cores_info.items():
        for c, agg in ci["results"].items():
            a = ci["anchors"][c]
            add(_layout_row("cores/node sweep ANCHOR", n, c, a, agg["s_values"],
                            agg.get("dirty")),
                (n, c, sweep._identity(a)))
    tuned = []
    for n, o in outcomes.items():
        merged = dict(o.all_results or {})
        merged.update(o.confirm_results or {})
        ranked = o.confirm_results or merged     # what build_report ranks: confirm reps if any
        for cid, r in merged.items():
            cfg = r.get("config")
            if not cfg or not r.get("s_values"):
                continue
            row = _layout_row("finalist search", n, o.cores, cfg, r["s_values"], r.get("dirty"))
            add(row, (n, o.cores, sweep._identity(cfg)))
            if cid in ranked:
                tuned.append((n, o.cores, sweep._identity(cfg)))
    rows = sorted(seen.values(), key=lambda r: r["median"])
    pool = [seen[k] for k in dict.fromkeys(tuned)] or rows
    pool = sorted(pool, key=lambda r: r["median"])
    if best in seen:                        # the report's BEST first, the rest by median
        pool = [seen[best]] + [r for r in pool if r is not seen[best]]
    winner = pool[0] if pool else None
    runner = pool[1] if len(pool) > 1 else None
    comparison = None
    if winner and runner:
        gap = (runner["median"] - winner["median"]) / winner["median"]
        comparison = dict(gap=gap, noise=wtune.machine.measured("noise_frac"),
                          beyond_noise=abs(gap) > wtune.machine.measured("noise_frac"))
    return dict(rows=rows, winner=winner, runner_up=runner, comparison=comparison)
