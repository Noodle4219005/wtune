"""Effort levels -O0..-O3 and the pre-run estimate of their tuning (RUNNING) time.

A level sets only the layout-search effort and whether physics advice runs; the prediction
points and the error band come from the case type (wtune.cost_drivers.plan_points_for_case).
"""
from __future__ import annotations

import math
from typing import Any

import wtune.machine
import wtune.nodes
import wtune.proxy as _proxy

LEVELS = {
    0: dict(name="O0", method="no jobs: cold-start predict + proxy layout"),
    1: dict(name="O1", method="joint (N, layout) proxy search, one leader per N re-timed (without "
                              "the proxy: one t0 wave over N in {1,2,4,8,max healthy})"),
    2: dict(name="O2", method="joint (N, layout) proxy search over every legal N, no real N-node "
                              "probe (the real node sweep when the proxy cannot run)"),
    3: dict(name="O3", method="full layout search: every legal N (coarse-to-fine), "
                              "cores/node, ranks x omp, nx x ny, numtiles at the finalists + "
                              "confirm reps"),
}
DEFAULT_LEVEL = 2

# effort-model guesses, replaced by measurements once they exist
EST_REBUILD_S = wtune.machine.MEASURED["rebuild_s"]    # built-in; s per ERA5 point rebuild
EST_REFINE_ROUNDS = 2                # coarse-to-fine refinement rounds around the best N
EST_ADAPTIVE_ROUNDS = 3              # round0 -> adaptive proposer rounds (O3 only)
EST_ROUND0_CONFIGS = 6               # decomposition candidates round0 samples at the winner N
EST_SPINUP_STEPS = 8                 # spin-up steps of a point job
EST_PROXY_RUN_START_S = 30.0         # per proxy run before its first step (crop cached): ~30 s on
                                     # Big Chill 2024873 (Grace) and ~26 s on karina band 2025378 (x86)
EST_PHYSICS_RUNS = 5                 # baseline + 3 candidates + ANCHOREND, per point job
EST_PHYSICS_POINTS = 2               # point jobs of the physics phase


def layout_estimates(*, s1: float, steps_probe: int, startup_s: float, max_nodes: int,
                     n_full: int = 0, finalists: int = 2, proxy: bool = True,
                     proxy_node_cap: int = 0) -> dict[int, float]:
    """{level: RUNNING s of the layout search} from the 1-node s/step `s1` (ideal scaling s1/N).
    O1: the {1,2,4,8,max} wave of 2-rep jobs; O2: + refinement, a cores/node wave and round0 at
    the winner -- with `proxy`, O1 and O2 are the joint proxy search alone (proxy_joint_s, O1 with
    one leader per N); O3: the full space (`n_full` 1-rep runs) at `finalists` N + confirm (no
    proxy)."""
    S = float(startup_s)
    job = lambda n, reps=2: reps * (S + steps_probe * s1 / n)            # noqa: E731
    seeds = sorted({n for n in (1, 2, 4, 8, max_nodes) if 1 <= n <= max_nodes})
    t1 = sum(max(job(n) for n in wave) for wave in wtune.nodes.sweep_waves(seeds, max_nodes))
    nstar = max(1, min(4, max_nodes))
    wave_n = job(nstar)
    slots = max(1, max_nodes // nstar)
    t_round0 = math.ceil(EST_ROUND0_CONFIGS / slots) * (S + steps_probe * s1 / nstar)
    before = t1 + EST_REFINE_ROUNDS * wave_n + wave_n
    if proxy:
        t1 = proxy_joint_s(S, steps_probe, s1, max_nodes, leaders=1, node_cap=proxy_node_cap)
    t2 = (proxy_joint_s(S, steps_probe, s1, max_nodes, node_cap=proxy_node_cap) if proxy
          else before + t_round0)
    full = finalists * math.ceil(max(n_full, EST_ROUND0_CONFIGS) / float(slots)) * (
        S + steps_probe * s1 / nstar)
    t3 = before + full + wave_n
    return {0: 0.0, 1: t1, 2: t2, 3: t3}


def proxy_round0_s(startup_s: float, steps: int, s_n: float, n: int, max_nodes: int,
                   time_points: int = 0) -> float:
    """RUNNING s of proxy_stage.select at N=n: the one-node pool (PROXY_POOL_FACTOR x the sample, one
    crop position, one job per layout, up to PROXY_JOB_NODES_FACTOR x max_nodes jobs at once), the
    band round of PROXY_BAND_KEEP leaders (PROXY_POSITIONS positions + `time_points` runs each)
    when n > PROXY_BAND_NODES, then PROXY_REAL_KEEP real probes in max_nodes // n slots. A proxy run
    of min(steps, PROXY_MAX_STEPS) steps gives each rank the target's patch, so its s/step is taken
    as the N-node `s_n` and it starts in EST_PROXY_RUN_START_S. The real probes run `steps` steps
    and start in `startup_s`."""
    from . import proxy_stage as P
    cap = P.PROXY_JOB_NODES_FACTOR * max_nodes
    run = EST_PROXY_RUN_START_S + min(steps, _proxy.PROXY_MAX_STEPS) * s_n
    pool = min(P.PROXY_POOL_MAX, P.PROXY_POOL_FACTOR * EST_ROUND0_CONFIGS)
    t = math.ceil(pool / max(1, min(cap, pool))) * run
    if n > P.PROXY_BAND_NODES and max_nodes >= P.PROXY_BAND_NODES:
        lead = P.PROXY_BAND_KEEP
        t += math.ceil(lead / max(1, min(cap // P.PROXY_BAND_NODES, lead))) * (_proxy.PROXY_POSITIONS + time_points) * run
    return t + math.ceil(P.PROXY_REAL_KEEP / max(1, max_nodes // n)) * (startup_s + steps * s_n)


def proxy_joint_s(startup_s: float, steps: int, s1: float, max_nodes: int, time_points: int = 0,
                  leaders: int | None = None, node_cap: int = 0) -> float:
    """RUNNING s of the O2 joint proxy search (node_search._run_node_search_proxy), legal N taken as
    1..max_nodes: PROXY_JOINT_PER_N one-node layouts per N at one crop position, one job per
    layout, up to `node_cap` (PROXY_JOB_NODES_FACTOR x max_nodes) nodes at once (s/step the mean of s1/N); the
    `leaders` (PROXY_JOINT_LEADERS) leaders of each N re-timed (PROXY_POSITIONS positions + `time_points` runs
    each) as bands when N > PROXY_BAND_NODES, as the whole domain on N nodes when 1 < N <=
    PROXY_BAND_NODES (one position); every proxy run min(steps, PROXY_MAX_STEPS) steps. No real
    N-node probe follows."""
    from . import proxy_stage as P
    leaders = leaders or P.PROXY_JOINT_LEADERS
    cap, ps, npos = node_cap or P.PROXY_JOB_NODES_FACTOR * max_nodes, min(steps, _proxy.PROXY_MAX_STEPS), _proxy.PROXY_POSITIONS
    ns = range(1, max_nodes + 1)
    lays = max_nodes * P.PROXY_JOINT_PER_N
    t = math.ceil(lays / max(1, min(cap, lays))) * (EST_PROXY_RUN_START_S + ps * s1 * sum(1.0 / n for n in ns) / max_nodes)
    big = [n for n in ns if n > P.PROXY_BAND_NODES] if min(max_nodes, cap) >= P.PROXY_BAND_NODES else []
    if big:
        lead = leaders * len(big)
        t += math.ceil(lead / max(1, min(cap // P.PROXY_BAND_NODES, lead))) * (npos + time_points) * (
            EST_PROXY_RUN_START_S + ps * s1 * sum(1.0 / n for n in big) / len(big))
    for n in range(2, min(P.PROXY_BAND_NODES, max_nodes, cap) + 1):     # whole-domain leaders on n nodes
        lead = leaders
        t += math.ceil(lead / max(1, min(cap // n, lead))) * (1 + time_points) * (EST_PROXY_RUN_START_S + ps * s1 / n)
    return t


def points_estimate(*, s1: float, startup_s: float, max_nodes: int, n_points: int, layouts: int,
                    layout_s: float, physics: bool, nstar: int | None = None,
                    window_steps: int = 32, restart_read_s: float = 30.0) -> float:
    """RUNNING s of the prediction part: the rebuild time that sticks out past the layout search,
    one point job (spin-up + `layouts` + ANCHOREND runs) per point in ceil(points x N / max_nodes)
    waves, and (physics) two point jobs of EST_PHYSICS_RUNS runs."""
    if n_points <= 0:
        return 0.0
    S = float(startup_s)
    nstar = nstar or max(1, min(4, max_nodes))
    spin = S + EST_SPINUP_STEPS * s1 / nstar
    per_run = S + restart_read_s + window_steps * s1 / nstar

    def point_jobs(points, runs):
        waves = math.ceil(max(1, points) * nstar / float(max_nodes))
        return waves * (spin + runs * per_run)

    exposed = max(0.0, wtune.machine.measured("rebuild_s") - layout_s)
    total = exposed + point_jobs(n_points, layouts + 1)
    if physics:
        total += point_jobs(EST_PHYSICS_POINTS, EST_PHYSICS_RUNS)
    return total


def level_estimates(*, s1: float, steps_probe: int, startup_s: float, max_nodes: int,
                    n_points: int, n_full: int = 0, proxy: bool = True, proxy_node_cap: int = 0,
                    **_ignored: Any) -> dict[int, float]:
    """{level: total RUNNING s} = layout_estimates + points_estimate of the winner layout at the
    case type's n_points."""
    lay = layout_estimates(s1=s1, steps_probe=steps_probe, startup_s=startup_s,
                           max_nodes=max_nodes, n_full=n_full, proxy=proxy,
                           proxy_node_cap=proxy_node_cap)
    out = {0: 0.0}
    for lv in (1, 2, 3):
        out[lv] = lay[lv] + points_estimate(
            s1=s1, startup_s=startup_s, max_nodes=max_nodes, n_points=n_points,
            layouts=1, layout_s=lay[lv], physics=False)
    return out


def highest_level_within(estimates: dict, real_run_s: float, requested: int) -> int:
    """The highest level <= `requested` whose estimate fits the real run; never below O1."""
    for lv in range(min(3, requested), 0, -1):
        if estimates[lv] <= real_run_s:
            return lv
    return 1


def format_level_estimates(estimates: dict, chosen: int) -> str:
    return ("  ".join("O%d ≈ %.0f min%s" % (lv, max(estimates[lv], 0) / 60.0,
                                           " <-" if lv == chosen else "")
                       for lv in (1, 2, 3)) + "  (running time; queue wait extra)")
