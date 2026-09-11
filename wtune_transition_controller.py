#!/usr/bin/env python3
"""WTune segment transition controller.

Decide, at a segment boundary, whether to switch (node count, uniform decomposition,
tasks-per-node) for the NEXT segment. Pure, testable decision function + CLI.

Design (grounded in measured wiki knowledge -- see anchors below):

- Trigger is DEADLINE PACE + STABLE global cost-regime, NOT storm position. A storm
  moving in space cannot be chased source-free: WRF's decomposition is uniform
  Cartesian, and the optimal uniform shape is cache/halo-geometry driven and
  LOAD-INDEPENDENT (8x56 wins regardless of where the storm is). A storm's position
  therefore changes neither the optimal shape nor the total work nor the optimal node
  count. Only a change in GLOBAL cost (e.g. a large convective outbreak, or a physics
  regime that flips hydrostatic/radt eligibility) moves the optimum -- and it acts
  THROUGH the measured s/step and the pace projection.

- STABILITY is required: within-window hotspot geography is spatially persistent
  (micro slow-set Jaccard 0.708), so a transient s/step spike must NOT trigger a
  switch. The caller passes cost_regime.stable; we act on a regime slowdown only when
  it is stable.

- Every switch costs SWITCH_COST_S (ADIOS2 restart read ~48s + write ~8s, M0). A
  node-count switch is taken only when the projected wall saved over the remaining
  work exceeds that cost. tasks-per-node changes are memory/policy mandated and are
  applied regardless of the wall-benefit guard.

- When node count (hence rank count) changes, the decomposition and numtiles are
  RE-DERIVED analytically (zero cluster measurement), from two calibrated invariants:
  patch aspect ~9:1 (the 8x56 optimum) and tile ~251 cells (numtiles invariant).

Wiki anchors: knowledge/wrf/2026-09-01-wrf-real-bottleneck-synchronization-granularity.md,
knowledge/wrf/2026-08-21-the-scaling-curve-already-passes-the-gpu-target.md,
knowledge/2026-08-26-wrf-8x56-decomposition-zero-code-win.md,
knowledge/wrf/2026-08-21-cpu-side-results-and-where-to-go-next.md,
knowledge/experiments/2026-08-31-wtune-malleable-boundary-migration-probes.md.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys


# --- calibrated constants (measured; see module docstring for provenance) ---
# Parallel efficiency vs 1 node, P1 node scan (intel-fast, restart 240 steps, n=3,
# non-radiation-step median), 112 ranks/node.
EFFICIENCY = {1: 1.00, 2: 0.88, 4: 0.85, 8: 0.76, 16: 0.69, 32: 0.52}
HW_TASKS_PER_NODE = 112       # f1 cores/node
SWITCH_COST_S = 11.6          # MEASURED 2026-09-05: ADIOS2 restart read 10.69s + write 0.90s
                              # (was 56.0 from an older read ~48s + write ~8s; 4.8x lower)
TARGET_PATCH_ASPECT = 9.0     # 8x56 optimum: patch 188x21 ~= 8.75:1, cache/halo geometry
TILE_CELLS = 251              # numtiles invariant: optimal tile ~251 cells

# f1/NCHC partition CPU-per-job tiers (112 cores/node). Node-scaling is BOUNDED by the
# partition's QOS MaxCpuPerJob: ct448 rejects >4 nodes with QOSMaxCpuPerJobLimit (and
# rejects 1 node with QOSMinCpuNotSatisfied). Pick the smallest partition that fits.
# (name, max_nodes); measured 2026-09-02 from `sacctmgr show qos`.
PARTITION_TIERS = [("ct112", 1), ("ct448", 4), ("ct1k", 10),
                   ("ct2k", 20), ("ct4k", 40), ("ct8k", 80)]


def partition_for_nodes(nodes):
    """Smallest f1 partition whose CPU-per-job cap admits `nodes`, or None if beyond ct8k."""
    for name, max_nodes in PARTITION_TIERS:
        if nodes <= max_nodes:
            return name
    return None


def _throughput(nodes):
    """Relative rank-throughput at a node count (nodes * parallel efficiency)."""
    return nodes * EFFICIENCY[nodes]


def project_s_per_step(s_now, nodes_now, nodes_new):
    """Project measured s/step from nodes_now to nodes_new via measured efficiency."""
    return s_now * _throughput(nodes_now) / _throughput(nodes_new)


def choose_decomp(ranks, nx, ny):
    """Pick px*py=ranks minimizing |patch_aspect - TARGET|.

    patch_aspect = (nx/px)/(ny/py) = (nx*py)/(ny*px); target ~9:1 (wide-flat) is the
    measured 8x56 optimum. Load-independent, so a good choice for any case/window.
    """
    best = None
    px = 1
    while px <= ranks:
        if ranks % px == 0:
            py = ranks // px
            if px <= nx and py <= ny:
                aspect = (nx * py) / float(ny * px)
                score = abs(aspect - TARGET_PATCH_ASPECT)
                if best is None or score < best[0]:
                    best = (score, px, py)
        px += 1
    if best is None:
        raise ValueError("no valid px*py=%d factorization fits domain %dx%d" % (ranks, nx, ny))
    return best[1], best[2]


def choose_numtiles(px, py, nx, ny):
    """numtiles ~= patch_cells / TILE_CELLS (the tile~251 invariant)."""
    patch_cells = (nx // px) * (ny // py)
    return max(1, int(round(patch_cells / float(TILE_CELLS))))


def effective_tasks_per_node(user_cap, mem_max):
    """tasks/node is the min of the hardware limit, the user reservation, and what
    the case's per-rank memory footprint allows (fewer ranks/node => more memory each)."""
    caps = [HW_TASKS_PER_NODE]
    if user_cap:
        caps.append(int(user_cap))
    if mem_max:
        caps.append(int(mem_max))
    return max(1, min(caps))


def assess_cost_regime(history, baseline, threshold=0.10, k=2):
    """Classify the global cost regime from a per-segment s/step history (cheap proxy).

    history: recent per-segment s/step, oldest..newest (newest = current segment).
    baseline: planned/reference s/step (e.g. the plan, or a median of early segments).
    shifted: current s/step deviates from baseline by > threshold (either direction).
    stable: the last k segments ALL deviate past threshold in the SAME direction --
            i.e. a persistent regime change, not a one-segment spike. The controller
            acts on a slowdown only when stable, so a transient wiggle never churns.

    Detection is deliberately here, separate from decide(): the Oracle kernel-cost
    model can replace this with a physics-based signal without touching the decision.
    """
    if not history or not baseline:
        return {"shifted": False, "stable": False}
    current = history[-1]
    dev = (current - baseline) / float(baseline)
    tail = history[-k:]
    devs = [(s - baseline) / float(baseline) for s in tail]
    stable = (len(tail) >= k and
              (all(x > threshold for x in devs) or all(x < -threshold for x in devs)))
    return {
        "shifted": abs(dev) > threshold,
        "stable": bool(stable),
        "current_s_per_step": current,
        "baseline_s_per_step": baseline,
        "deviation": round(dev, 3),
    }


def imbalance_gate(per_rank_compute_s, migrate_cost_s, safety=1.0):
    """Decide whether per-rank compute imbalance is worth correcting via the rebalance system.

    per_rank_compute_s: MEASURED per-rank compute seconds projected over the REMAINING
        window (measured s/step-per-rank x steps remaining). rank 0 is the serial-I/O
        outlier -- neutralise (median) before calling. Truth source: blockbar wait_frac
        x wall (1-frac), or APS elapsed-mpi.
    migrate_cost_s: one-time wall cost of the rebalance mechanism considered. A full
        ADIOS2-restart rebalance costs SWITCH_COST_S (56s); cheap in-place/runtime block
        migration costs far less -- pass its own measured cost.

    Gate on ABSOLUTE recoverable seconds, NOT the spread ratio. Measured 2026-09-02
    (knowledge/experiments/2026-09-02-wtune-bigcase-perrank-works-at-scale.md): WRF
    per-rank imbalance stays a modest 1.05-1.14x RATIO across regimes at every scale,
    but the ABSOLUTE recoverable scales with the run -- ~1.2s at 12km/36-rank (never
    worth it) vs ~7-16s at 3km/448-rank/2h. A ratio threshold misfires across scales; a
    seconds threshold is scale-correct.

    recoverable_s = max - mean: the bottleneck rank sets the wall (sync-granularity: wall
    = per-phase max, not per-rank total), so a perfectly balanced decomposition recovers
    (max - mean). trigger iff recoverable_s > migrate_cost_s * safety. Consequence of the
    measured magnitudes: a 56s restart-based rebalance almost never pays for a single
    WRF case (recoverable < 56s); only a cheap runtime/in-place migration clears the gate.

    UPDATE 2026-09-05: that last sentence held at the 2 h scale it was measured at and does
    NOT hold at 12 h. Per-segment recoverable seconds over ten 12 h cases run 17.6-152.8s,
    and 6 of 10 clear 56s in every segment (marine_cao 128-153s, i.e. 2.3-2.7x the restart
    cost). Same gate, same threshold, opposite answer -- which is exactly why it is written
    in absolute seconds rather than a spread ratio.

    UPDATE 2026-09-05 (again): SWITCH_COST_S was re-measured and is 11.6s, not 56s -- read
    10.69s, write 0.90s, karina_cat4 at 4 nodes / 448 ranks with io_form_restart=14. The old
    figure predates the ADIOS2 I/O work. At 11.6s EVERY segment measured clears the gate (the
    minimum recoverable is 17.6s), so the switch decision is no longer marginal: it is
    unconditionally yes. A cheap in-place migration mechanism is therefore NOT needed to make
    rebalancing pay -- the restart path already does.
    """
    xs = [x for x in per_rank_compute_s if x is not None]
    if len(xs) < 2:
        return {"trigger": False, "reason": "insufficient per-rank data (<2 ranks)"}
    n = len(xs)
    mean = sum(xs) / n
    mx = max(xs)
    recoverable_s = mx - mean
    spread_ratio = (mx / mean) if mean else 1.0
    threshold_s = migrate_cost_s * safety
    trigger = recoverable_s > threshold_s
    return {
        "trigger": bool(trigger),
        "recoverable_s": round(recoverable_s, 3),
        "spread_ratio": round(spread_ratio, 4),
        "migrate_cost_s": migrate_cost_s,
        "threshold_s": round(threshold_s, 3),
        "n_ranks": n,
        "reason": ("recoverable %.1fs > cost %.1fs -> rebalance" % (recoverable_s, threshold_s))
                  if trigger else
                  ("recoverable %.1fs <= cost %.1fs -> hold (imbalance below correction cost)"
                   % (recoverable_s, threshold_s)),
    }


def imbalance_gate_predicted(pred_per_rank_s, migrate_cost_s, hit_rate, safety=1.0):
    """Same gate, but on PREDICTED per-rank cost, so a decision can be made BEFORE the
    segment runs instead of after measuring it.

    A prediction cannot be fed in as if it were a measurement. The rebalancer acts on the
    ranks the model NAMES as hot, so what is actually recoverable is not (max - mean) but
    that quantity times the share of named ranks that are genuinely hot -- the hot-spot hit
    rate, which is measured per PBL scheme:

        bl=1 YSU  ~68%      bl=2 MYJ  ~63%      bl=5 MYNN ~76% (with its own head)

    Discounting by hit_rate makes the gate conservative in exactly the right way: a model
    that cannot find the hot ranks cannot justify paying to move them. Pass the hit rate
    measured for the scheme in play, not a global average -- they differ by 13 points.
    """
    if not 0.0 < hit_rate <= 1.0:
        return {"trigger": False, "reason": "hit_rate must be in (0,1]; got %r" % (hit_rate,)}
    g = imbalance_gate(pred_per_rank_s, migrate_cost_s, safety)
    if "recoverable_s" not in g:
        return g
    eff = g["recoverable_s"] * hit_rate
    trigger = eff > g["threshold_s"]
    g.update({
        "trigger": bool(trigger),
        "predicted": True,
        "hit_rate": hit_rate,
        "effective_recoverable_s": round(eff, 3),
        "reason": ("predicted %.1fs x hit %.0f%% = %.1fs > cost %.1fs -> rebalance"
                   % (g["recoverable_s"], 100 * hit_rate, eff, g["threshold_s"])) if trigger else
                  ("predicted %.1fs x hit %.0f%% = %.1fs <= cost %.1fs -> hold"
                   % (g["recoverable_s"], 100 * hit_rate, eff, g["threshold_s"])),
    })
    return g


def recommend_initial(domain, nodes, user_max_tasks_per_node=None, mem_max_tasks_per_node=None):
    """Zero-measurement best STATIC config for a fresh (unseen) case at a FIXED node count.

    This is what WTune outputs for a random large case up front -- no cluster sweep. All
    from calibrated case-general invariants: decomposition via patch aspect ~9 (8x56-class),
    numtiles via tile ~251, ADIOS2 io_form=14, and the partition that admits this node count.
    Physics eligibility (hydrostatic/radt) is a separate vet, not encoded here.
    """
    nx, ny = domain["nx"], domain["ny"]
    tpn = effective_tasks_per_node(user_max_tasks_per_node, mem_max_tasks_per_node)
    ranks = nodes * tpn
    px, py = choose_decomp(ranks, nx, ny)
    return {
        "nodes": nodes,
        "ntasks_per_node": tpn,
        "ranks": ranks,
        "nproc_x": px,
        "nproc_y": py,
        "numtiles": choose_numtiles(px, py, nx, ny),
        "partition": partition_for_nodes(nodes),
        "io_form_history": 14,
        "io_form_restart": 14,
    }


def _hold(reason):
    return {"action": "hold", "reason": reason}


def _build_switch(new_nodes, tpn, nx, ny, s_now, cur_nodes, reason,
                  steps_remaining, memory_or_forced=False, node_hours=False, risk=None):
    """Assemble a switch decision, applying the wall-benefit guard for scale-up.

    The wall-benefit guard is for scale-UP (does it save enough wall to beat the switch
    cost). memory/forced and node-hours (scale-down) switches are justified by their own
    criterion at the call site and bypass it.
    """
    ranks = new_nodes * tpn
    px, py = choose_decomp(ranks, nx, ny)
    numtiles = choose_numtiles(px, py, nx, ny)
    proj_cur = s_now * steps_remaining
    if new_nodes in EFFICIENCY and cur_nodes in EFFICIENCY:
        proj_new = project_s_per_step(s_now, cur_nodes, new_nodes) * steps_remaining
    else:
        proj_new = proj_cur
    benefit_s = proj_cur - proj_new  # wall saved by switching (>0 if faster)

    partition = partition_for_nodes(new_nodes)
    decision = {
        "next": {
            "nodes": new_nodes,
            "ntasks_per_node": tpn,
            "ranks": ranks,
            "nproc_x": px,
            "nproc_y": py,
            "numtiles": numtiles,
            "partition": partition,
        },
        "reason": reason,
        "switch_cost_s": SWITCH_COST_S,
        "projected_wall_saved_s": round(benefit_s, 1),
    }
    if risk:
        decision["risk"] = risk

    # memory/forced/node-hours/tasks-only changes bypass the wall-benefit guard; each is
    # justified by its own criterion at the call site.
    if memory_or_forced or node_hours or new_nodes == cur_nodes:
        decision["action"] = "switch"
        decision["guard"] = ("bypassed (node-hours optimization, fits budget)" if node_hours
                             else "bypassed (memory/policy mandated)")
        return decision

    if benefit_s > SWITCH_COST_S or risk:
        decision["action"] = "switch"
        decision["guard"] = "passed (benefit %.1fs > cost %.1fs)" % (benefit_s, SWITCH_COST_S)
        return decision

    # Guard fails: switching costs more wall than it saves -> hold.
    return _hold("would switch to %d nodes but projected wall saved %.1fs <= switch cost "
                 "%.1fs; holding" % (new_nodes, benefit_s, SWITCH_COST_S))


def decide(state):
    """Decide the next-segment config. See module docstring for the contract.

    state keys: domain{nx,ny}, nodes, tasks_per_node, s_per_step, steps_remaining,
    wall_budget_remaining_s, available_nodes[], mem_max_tasks_per_node,
    cost_regime{shifted,stable}, user_policy{allow_switch,max_tasks_per_node,
    forced_nodes,optimize_node_hours}.
    """
    policy = state.get("user_policy", {})
    if not policy.get("allow_switch", True):
        return _hold("switching disabled by user policy")

    nx = state["domain"]["nx"]
    ny = state["domain"]["ny"]
    nodes = state["nodes"]
    tpn_cur = state["tasks_per_node"]
    s_now = state["s_per_step"]
    steps_remaining = state["steps_remaining"]
    budget = state["wall_budget_remaining_s"]
    # cost_regime may be given explicitly, or computed from a per-segment s/step history.
    # An absent OR empty cost_regime falls back to the history-based detector.
    regime = state.get("cost_regime")
    if not regime and state.get("segment_history"):
        regime = assess_cost_regime(state["segment_history"],
                                    state.get("baseline_s_per_step"),
                                    state.get("regime_threshold", 0.10),
                                    state.get("regime_k", 2))
    regime = regime or {}
    tpn = effective_tasks_per_node(policy.get("max_tasks_per_node"),
                                   state.get("mem_max_tasks_per_node"))
    # A hard node-count cap (e.g. a rule of "at most 4 nodes") removes the scale-up lever.
    max_nodes = policy.get("max_nodes")
    available = sorted(n for n in state.get("available_nodes", [nodes])
                       if n in EFFICIENCY and (max_nodes is None or n <= max_nodes))

    # 1. User-forced node count wins outright.
    forced = policy.get("forced_nodes")
    if forced:
        if forced == nodes and tpn == tpn_cur:
            return _hold("at forced config (%d nodes, %d tasks/node)" % (nodes, tpn))
        return _build_switch(forced, tpn, nx, ny, s_now, nodes,
                             "forced by user policy", steps_remaining, memory_or_forced=True)

    proj_now = s_now * steps_remaining
    slack_now = budget - proj_now

    # 2. Behind deadline -> scale up (only if the slowdown is stable, not a transient spike).
    if slack_now < 0:
        if regime.get("shifted") and not regime.get("stable", False):
            return _hold("behind pace but the cost spike is transient (stable=False); "
                         "holding to avoid churning on a wiggle")
        for n in available:
            if n <= nodes:
                continue
            proj = project_s_per_step(s_now, nodes, n) * steps_remaining
            if proj <= budget:
                return _build_switch(n, tpn, nx, ny, s_now, nodes,
                                     "behind deadline; smallest node count that fits budget",
                                     steps_remaining)
        biggest = available[-1] if available else nodes
        if biggest > nodes:
            return _build_switch(biggest, tpn, nx, ny, s_now, nodes,
                                 "behind deadline; max available nodes (budget still at risk)",
                                 steps_remaining, risk="budget_may_be_missed")
        return _hold("behind deadline but no more nodes available; cannot switch")

    # 3. tasks/node must change (memory or policy) even at the same node count.
    if tpn != tpn_cur:
        return _build_switch(nodes, tpn, nx, ny, s_now, nodes,
                             "apply tasks-per-node cap (memory/policy)",
                             steps_remaining, memory_or_forced=True)

    # 4. Way ahead of deadline -> optionally scale down to save node-hours (opt-in).
    if policy.get("optimize_node_hours") and slack_now > proj_now:
        for n in reversed(available):
            if n >= nodes:
                continue
            proj = project_s_per_step(s_now, nodes, n) * steps_remaining
            if proj + SWITCH_COST_S <= budget:  # slower but still fits, incl. switch cost
                return _build_switch(n, tpn, nx, ny, s_now, nodes,
                                     "ahead of deadline; scale down to save node-hours",
                                     steps_remaining, node_hours=True)

    return _hold("on pace; no switch (slack %.0fs)" % slack_now)


# ------------------------------- CLI -------------------------------

def cmd_plan(args):
    raw = sys.stdin.read() if args.state in ("-", None) else open(args.state).read()
    decision = decide(json.loads(raw))
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def parse_rsl_step_seconds(path):
    """Extract per-timestep solver seconds from an rsl.out.* file.

    Lines look like: 'Timing for main: time 2019-11-26_23:12:00 on domain 1: 0.842 elapsed seconds'.
    The median over all main steps is robust to the few (radiation-cadence) outliers.
    """
    secs = []
    with open(path) as stream:
        for line in stream:
            if "Timing for main" in line and "elapsed seconds" in line:
                try:
                    tail = line.rsplit(":", 1)[-1]          # ' 0.842 elapsed seconds'
                    secs.append(float(tail.strip().split()[0]))
                except (ValueError, IndexError):
                    pass
    return secs


def read_results_io_last(path):
    """Return the last data row of a results-io.csv as a dict (or {})."""
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def advance(segment_dir, context):
    """Decide the NEXT segment's config from a COMPLETED segment directory + context.

    Reads the measured solver s/step (rsl.out.0000 median) and nodes/ranks
    (results-io.csv), appends the measurement to the s/step history, then runs decide().
    context supplies domain, wall_budget_remaining_s, steps_remaining, available_nodes,
    user_policy, baseline_s_per_step, and (optional) prior segment_history / regime knobs.
    """
    secs = parse_rsl_step_seconds(os.path.join(segment_dir, "rsl.out.0000"))
    s_now = _median(secs)
    rio = read_results_io_last(os.path.join(segment_dir, "results-io.csv"))
    nodes = int(rio.get("nodes") or context["nodes"])
    ranks = int(rio.get("ranks") or (nodes * HW_TASKS_PER_NODE))
    tpn = ranks // nodes if nodes else HW_TASKS_PER_NODE

    history = list(context.get("segment_history", []))
    if s_now is not None:
        history.append(s_now)

    state = dict(context)
    state.update({"s_per_step": s_now, "nodes": nodes,
                  "tasks_per_node": tpn, "segment_history": history})
    decision = decide(state)
    decision["measured"] = {"s_per_step_median": s_now, "steps_parsed": len(secs),
                            "nodes": nodes, "ranks": ranks, "tasks_per_node": tpn}
    return decision


def cmd_advance(args):
    if args.context in ("-", None):
        context = json.loads(sys.stdin.read())
    else:
        context = json.loads(open(args.context).read())
    print(json.dumps(advance(args.segment_dir, context), indent=2, sort_keys=True))
    return 0


def cmd_recommend(args):
    raw = sys.stdin.read() if args.state in ("-", None) else open(args.state).read()
    s = json.loads(raw)
    r = recommend_initial(s["domain"], s["nodes"],
                          s.get("user_policy", {}).get("max_tasks_per_node"),
                          s.get("mem_max_tasks_per_node"))
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


def cmd_regime(args):
    raw = sys.stdin.read() if args.state in ("-", None) else open(args.state).read()
    s = json.loads(raw)
    r = assess_cost_regime(s["history"], s.get("baseline"),
                           s.get("threshold", 0.10), s.get("k", 2))
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


def cmd_imbalance(args):
    raw = sys.stdin.read() if args.state in ("-", None) else open(args.state).read()
    s = json.loads(raw)
    r = imbalance_gate(s["per_rank_compute_s"], s.get("migrate_cost_s", SWITCH_COST_S),
                       s.get("safety", 1.0))
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


def cmd_selftest(args):
    dom = {"nx": 1501, "ny": 1201}
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # analytic config invariants
    px, py = choose_decomp(448, 1501, 1201)
    check("decomp 448 -> 8x56 (aspect optimum)", (px, py) == (8, 56), "%dx%d" % (px, py))
    check("numtiles 8x56 -> 16 (tile~251)", choose_numtiles(8, 56, 1501, 1201) == 16,
          str(choose_numtiles(8, 56, 1501, 1201)))
    px2, py2 = choose_decomp(896, 1501, 1201)
    check("decomp 896 wide-flat (px<py)", px2 < py2, "%dx%d" % (px2, py2))

    base = {
        "domain": dom, "nodes": 4, "tasks_per_node": 112, "s_per_step": 0.874,
        "steps_remaining": 10000, "available_nodes": [4, 8, 16, 32],
        "cost_regime": {}, "user_policy": {},
    }

    # on pace -> hold
    d = decide(dict(base, wall_budget_remaining_s=0.874 * 10000 + 5000))
    check("on pace -> hold", d["action"] == "hold", d.get("reason", ""))

    # behind, stable -> scale up to smallest fitting node count
    d = decide(dict(base, wall_budget_remaining_s=4800,
                    cost_regime={"shifted": True, "stable": True}))
    check("behind+stable -> switch up", d["action"] == "switch", json.dumps(d.get("reason", "")))
    check("behind -> chose >4 nodes", d.get("action") == "switch" and d["next"]["nodes"] > 4,
          json.dumps(d.get("next", {})))

    # behind but transient spike -> hold (do not churn)
    d = decide(dict(base, wall_budget_remaining_s=4800,
                    cost_regime={"shifted": True, "stable": False}))
    check("behind+transient -> hold", d["action"] == "hold", d.get("reason", ""))

    # behind but even max nodes only marginal, guard: still switch under risk to best-effort
    d = decide(dict(base, wall_budget_remaining_s=1.0, steps_remaining=100000,
                    cost_regime={"shifted": True, "stable": True}))
    check("behind, no node fits -> best-effort switch w/ risk",
          d["action"] == "switch" and d.get("risk") == "budget_may_be_missed",
          json.dumps(d.get("reason", "")))

    # memory pressure caps tasks/node -> switch applies cap at same node count
    d = decide(dict(base, wall_budget_remaining_s=0.874 * 10000 + 5000,
                    mem_max_tasks_per_node=56))
    check("memory cap 56 -> tasks/node=56", d["action"] == "switch" and d["next"]["ntasks_per_node"] == 56,
          json.dumps(d.get("next", {})))
    check("memory cap -> ranks=4*56=224", d.get("action") == "switch" and d["next"]["ranks"] == 224,
          json.dumps(d.get("next", {})))

    # user cap below hardware respected
    d = decide(dict(base, wall_budget_remaining_s=0.874 * 10000 + 5000,
                    user_policy={"max_tasks_per_node": 64}))
    check("user cap 64 -> tasks/node=64", d["action"] == "switch" and d["next"]["ntasks_per_node"] == 64,
          json.dumps(d.get("next", {})))

    # allow_switch=False -> always hold even when behind
    d = decide(dict(base, wall_budget_remaining_s=1.0,
                    cost_regime={"shifted": True, "stable": True},
                    user_policy={"allow_switch": False}))
    check("policy allow_switch=False -> hold", d["action"] == "hold", d.get("reason", ""))

    # forced_nodes overrides
    d = decide(dict(base, wall_budget_remaining_s=0.874 * 10000 + 5000,
                    user_policy={"forced_nodes": 8}))
    check("forced_nodes=8 -> switch to 8", d["action"] == "switch" and d["next"]["nodes"] == 8,
          json.dumps(d.get("next", {})))

    # switch-cost guard: tiny remaining work, scaling up saves < 56s -> hold
    d = decide(dict(base, s_per_step=0.874, steps_remaining=40, wall_budget_remaining_s=10.0,
                    cost_regime={"shifted": True, "stable": True}))
    check("guard blocks marginal switch (< 56s benefit)", d["action"] == "hold",
          d.get("reason", ""))

    # ahead + optimize_node_hours -> scale down
    d = decide(dict(base, s_per_step=0.487, nodes=16, tasks_per_node=112,
                    steps_remaining=2000, wall_budget_remaining_s=100000,
                    user_policy={"optimize_node_hours": True}))
    check("ahead + optimize -> scale down", d["action"] == "switch" and d["next"]["nodes"] < 16,
          json.dumps(d.get("next", {})))

    # --- cost-regime detector ---
    r = assess_cost_regime([0.9, 1.4, 1.5], baseline=0.9, threshold=0.10, k=2)
    check("regime: sustained slowdown -> stable+shifted", r["shifted"] and r["stable"], json.dumps(r))
    r = assess_cost_regime([0.9, 0.9, 1.5], baseline=0.9, threshold=0.10, k=2)
    check("regime: one-segment spike -> shifted, NOT stable", r["shifted"] and not r["stable"], json.dumps(r))
    r = assess_cost_regime([0.9, 0.91, 0.9], baseline=0.9, threshold=0.10, k=2)
    check("regime: on baseline -> neither", (not r["shifted"]) and (not r["stable"]), json.dumps(r))

    # --- decide computes regime from segment_history (no explicit cost_regime) ---
    d = decide(dict(base, s_per_step=1.5, wall_budget_remaining_s=4800,
                    baseline_s_per_step=0.874, segment_history=[0.874, 1.4, 1.5]))
    check("history-driven: behind+stable -> switch up", d["action"] == "switch",
          json.dumps(d.get("reason", "")))
    d = decide(dict(base, s_per_step=1.5, wall_budget_remaining_s=4800,
                    baseline_s_per_step=0.874, segment_history=[0.874, 0.874, 1.5]))
    check("history-driven: behind+transient -> hold", d["action"] == "hold",
          json.dumps(d.get("reason", "")))

    # --- rsl parser + median + advance() end-to-end on a synthetic segment dir ---
    check("median odd", _median([3.0, 1.0, 2.0]) == 2.0)
    check("median even", _median([1.0, 2.0, 3.0, 4.0]) == 2.5)
    import tempfile
    seg = tempfile.mkdtemp(prefix="wtune-adv-")
    with open(os.path.join(seg, "rsl.out.0000"), "w") as fh:
        fh.write("Timing for main: time A on domain   1:    1.50 elapsed seconds\n")
        fh.write("some other line\n")
        fh.write("Timing for main: time B on domain   1:    1.40 elapsed seconds\n")
        fh.write("Timing for main: time C on domain   1:    1.60 elapsed seconds\n")
        fh.write("d01 ... wrf: SUCCESS COMPLETE WRF\n")
    with open(os.path.join(seg, "results-io.csv"), "w") as fh:
        fh.write("logical_run_id,segment_id,run,run_mode,enable_io_optimization,burst_buffer,"
                 "nodes,ranks,steps,writes,write_median_s,out_bytes,out_human,sanity\n")
        fh.write("r,0,seg,restart,true,none,4,448,60,1,0.35,100,100,ok\n")
    secs = parse_rsl_step_seconds(os.path.join(seg, "rsl.out.0000"))
    check("rsl parser finds 3 steps", len(secs) == 3, str(secs))
    check("rsl median = 1.50", _median(secs) == 1.50, str(_median(secs)))
    adv = advance(seg, {"domain": dom, "steps_remaining": 6000, "wall_budget_remaining_s": 4800,
                        "available_nodes": [4, 8, 16, 32], "baseline_s_per_step": 0.874,
                        "segment_history": [0.874, 1.45], "user_policy": {}})
    check("advance: reads nodes=4 ranks=448 from csv",
          adv["measured"]["nodes"] == 4 and adv["measured"]["ranks"] == 448, json.dumps(adv.get("measured", {})))
    check("advance: sustained slowdown behind budget -> switch up",
          adv["action"] == "switch" and adv["next"]["nodes"] > 4, json.dumps(adv.get("reason", "")))

    # --- partition mapping (f1 QOS CpuPerJob tiers) ---
    check("partition 1 node -> ct112", partition_for_nodes(1) == "ct112")
    check("partition 4 nodes -> ct448", partition_for_nodes(4) == "ct448")
    check("partition 8 nodes -> ct1k (NOT ct448: QOSMaxCpuPerJobLimit)", partition_for_nodes(8) == "ct1k")
    check("partition 20 nodes -> ct2k", partition_for_nodes(20) == "ct2k")
    check("partition >80 nodes -> None", partition_for_nodes(81) is None)
    # budget 5000: 4N (proj 8740) behind, 8N (proj ~4886) fits -> chooses 8N -> ct1k
    d = decide(dict(base, wall_budget_remaining_s=5000, cost_regime={"shifted": True, "stable": True}))
    check("switch to 8N emits partition ct1k", d["action"] == "switch" and d["next"].get("partition") == "ct1k",
          json.dumps(d.get("next", {})))

    # --- fixed-node regime: max_nodes cap removes scale-up; recommend_initial ---
    rec = recommend_initial(dom, 4)
    check("recommend 4N -> 8x56", (rec["nproc_x"], rec["nproc_y"]) == (8, 56), "%dx%d" % (rec["nproc_x"], rec["nproc_y"]))
    check("recommend 4N -> numtiles 16 (invariant, not shipped 8)", rec["numtiles"] == 16, str(rec["numtiles"]))
    check("recommend 4N -> ct448, ranks 448, io_form 14",
          rec["partition"] == "ct448" and rec["ranks"] == 448 and rec["io_form_restart"] == 14, json.dumps(rec))
    d = decide(dict(base, wall_budget_remaining_s=1000, cost_regime={"shifted": True, "stable": True},
                    user_policy={"max_nodes": 4}))
    check("max_nodes=4 + behind -> cannot scale, hold",
          d["action"] == "hold" and "no more nodes" in d.get("reason", ""), d.get("reason", ""))

    # --- imbalance gate (absolute recoverable seconds, not ratio) ---
    small = [12.4] * 35 + [13.3]                 # 12km/36-rank: recoverable ~0.9s
    big = [130.0] * 446 + [140.0, 145.0]         # 3km/448-rank/2h: recoverable ~15s
    check("imbalance gate: 12km recoverable ~0.9s -> no rebalance (any cost)",
          imbalance_gate(small, SWITCH_COST_S)["trigger"] is False and
          imbalance_gate(small, 3.0)["trigger"] is False,
          json.dumps(imbalance_gate(small, 3.0)))
    check("imbalance gate: big case vs 56s restart -> no (recoverable < restart cost)",
          imbalance_gate(big, SWITCH_COST_S)["trigger"] is False,
          json.dumps(imbalance_gate(big, SWITCH_COST_S)))
    check("imbalance gate: big case vs cheap 3s runtime migration -> rebalance",
          imbalance_gate(big, 3.0)["trigger"] is True,
          json.dumps(imbalance_gate(big, 3.0)))
    check("imbalance gate: <2 ranks -> no trigger",
          imbalance_gate([], SWITCH_COST_S)["trigger"] is False)

    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print("%-52s %s%s" % (name, "PASS" if passed else "FAIL",
                              ("  [%s]" % detail) if (detail and not passed) else ""))
    print("\n%d/%d passed" % (sum(1 for c in checks if c[1]), len(checks)))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="WTune segment transition controller")
    sub = p.add_subparsers(dest="command")
    q = sub.add_parser("plan", help="read a segment state JSON (file or '-'), emit a decision")
    q.add_argument("--state", default="-")
    q.set_defaults(func=cmd_plan)
    q = sub.add_parser("recommend", help="zero-measurement best static config at a fixed node count: {domain,nodes,...}")
    q.add_argument("--state", default="-")
    q.set_defaults(func=cmd_recommend)
    q = sub.add_parser("advance", help="decide next-segment config from a completed segment dir + context JSON")
    q.add_argument("--segment-dir", required=True)
    q.add_argument("--context", default="-")
    q.set_defaults(func=cmd_advance)
    q = sub.add_parser("regime", help="classify cost regime from {history[],baseline,threshold?,k?}")
    q.add_argument("--state", default="-")
    q.set_defaults(func=cmd_regime)
    q = sub.add_parser("imbalance", help="per-rank imbalance gate: {per_rank_compute_s[],migrate_cost_s?,safety?}")
    q.add_argument("--state", default="-")
    q.set_defaults(func=cmd_imbalance)
    q = sub.add_parser("selftest", help="run built-in decision tests")
    q.set_defaults(func=cmd_selftest)
    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
