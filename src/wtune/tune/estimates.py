"""Effort levels -O0..-O3 and the pre-run estimate of their tuning (RUNNING) time.

A level sets only the layout-search effort and whether physics advice runs; the prediction
points and the error band come from the case type (wtune.cost_drivers.plan_points_for_case).
"""
from __future__ import annotations

import math
from typing import Any

import wtune.nodes

LEVELS = {
    0: dict(name="O0", method="no jobs: cold-start predict + proxy layout"),
    1: dict(name="O1", method="one t0 wave over N in {1,2,4,8,max healthy}"),
    2: dict(name="O2", method="O1 + refinement, cores/node, sampled round0 at the winner"),
    3: dict(name="O3", method="DEFAULT, full layout search: every legal N (coarse-to-fine), "
                              "cores/node, ranks x omp, nx x ny, numtiles at the finalists + "
                              "confirm reps"),
}
DEFAULT_LEVEL = 2

# effort-model guesses, replaced by measurements once they exist
EST_REBUILD_S = 180.0                # s per ERA5 point rebuild: 1:21-3:00 measured (sc26p1)
EST_REFINE_ROUNDS = 2                # coarse-to-fine refinement rounds around the best N
EST_ADAPTIVE_ROUNDS = 3              # round0 -> adaptive proposer rounds (O3 only)
EST_ROUND0_CONFIGS = 6               # decomposition candidates measured at the winner N
EST_SPINUP_STEPS = 8                 # spin-up steps of a point job
EST_PHYSICS_RUNS = 5                 # baseline + 3 candidates + ANCHOREND, per point job
EST_PHYSICS_POINTS = 2               # point jobs of the physics phase


def layout_estimates(*, s1: float, steps_probe: int, startup_s: float, max_nodes: int,
                     n_full: int = 0, finalists: int = 2) -> dict[int, float]:
    """{level: RUNNING s of the layout search} from the 1-node s/step `s1` (ideal scaling s1/N).
    O1: the {1,2,4,8,max} wave of 2-rep jobs; O2: + refinement, a cores/node wave and a sampled
    round0 at the winner; O3: the full space (`n_full` 1-rep runs) at `finalists` N + confirm."""
    S = float(startup_s)
    job = lambda n, reps=2: reps * (S + steps_probe * s1 / n)            # noqa: E731
    seeds = sorted({n for n in (1, 2, 4, 8, max_nodes) if 1 <= n <= max_nodes})
    t1 = sum(max(job(n) for n in wave) for wave in wtune.nodes.sweep_waves(seeds, max_nodes))
    nstar = max(1, min(4, max_nodes))
    wave_n = job(nstar)
    slots = max(1, max_nodes // nstar)
    t_round0 = math.ceil(EST_ROUND0_CONFIGS / slots) * (S + steps_probe * s1 / nstar)
    t2 = t1 + EST_REFINE_ROUNDS * wave_n + wave_n + t_round0
    full = finalists * math.ceil(max(n_full, EST_ROUND0_CONFIGS) / float(slots)) * (
        S + steps_probe * s1 / nstar)
    t3 = t2 - t_round0 + full + wave_n
    return {0: 0.0, 1: t1, 2: t2, 3: t3}


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

    exposed = max(0.0, EST_REBUILD_S - layout_s)
    total = exposed + point_jobs(n_points, layouts + 1)
    if physics:
        total += point_jobs(EST_PHYSICS_POINTS, EST_PHYSICS_RUNS)
    return total


def level_estimates(*, s1: float, steps_probe: int, startup_s: float, max_nodes: int,
                    n_points: int, n_full: int = 0, **_ignored: Any) -> dict[int, float]:
    """{level: total RUNNING s} = layout_estimates + points_estimate of the winner layout at the
    case type's n_points."""
    lay = layout_estimates(s1=s1, steps_probe=steps_probe, startup_s=startup_s,
                           max_nodes=max_nodes, n_full=n_full)
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
