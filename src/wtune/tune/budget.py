"""Run-time estimates: t0 job sizing, the point-job cost model (decision 0055) and the point-job
shape it sizes (ANCHOR/ANCHOREND wrapping, ranking context, sbatch exports)."""
from __future__ import annotations

import math
import os
import statistics

from ._deps import sweep, wtune_predict
from .common import ANCHOR_END_ID, _parse_slurm_elapsed, compute_slots
from .context import DEFAULT_STARTUP_S, Ctx

RANKING_R_RAD = 3.93                   # radiation-step cost / normal step: calibration's day level
                                       # (r_rad_default_day) -- P* is the most daylit point
RESTART_READ_S_BY_IO = {14: 30.0, 2: 120.0}    # s/run restart read on top of startup_s: ADIOS2 BP
                                       # vs serial netCDF (conus25 11.5 GB, job 1231: 120 s)


def estimate_run_time(n_or_pairs: int | list, reps: int, steps: int, s_est: float | None = None,
                      startup_s: float = DEFAULT_STARTUP_S) -> dict:
    """One job of `reps` replicate(s): `n_or_pairs` is a config count sized at `s_est`, or
    [(config, s_per_step)] sized per config. total_run_s is the RAW expected time, the only
    number budget decisions may use; job_hours/time_limit/timeout_s are scc_sweep's padded
    sbatch request (x1.5 + 10 min, capped) -- never a budget input (2026-09-23 conus25)."""
    if isinstance(n_or_pairs, int):
        per_run_s_list = [startup_s + steps * s_est] * n_or_pairs
    else:
        per_run_s_list = [startup_s + steps * s for _cfg, s in n_or_pairs]
    total_run_s = reps * sum(per_run_s_list)
    job_hours = total_run_s / 3600.0
    padded_hours = job_hours * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN / 60.0
    capped_hours = min(padded_hours, sweep.TIME_CAP_HOURS)
    max_per_run_s = max(per_run_s_list) if per_run_s_list else 0.0
    timeout_s = max(sweep.TIMEOUT_MIN_S, int(math.ceil(sweep.TIMEOUT_MULTIPLIER * max_per_run_s)))
    return dict(total_run_s=total_run_s, job_hours=capped_hours, uncapped_hours=padded_hours,
                time_limit=sweep.campaign_experiment._format_hms(capped_hours * 60.0),
                timeout_s=timeout_s)


def measure_startup_s(job_configs: dict, states: dict, wave_results: dict,
                      steps: int) -> tuple[float | None, int]:
    """(startup_s, n_runs): per-run overhead = (Slurm elapsed - steps x measured s/step) over
    the COMPLETED REPS=1 jobs of a wave ({job: [config]}); (None, 0) when none is usable."""
    total_elapsed_s, total_compute_s, n_runs = 0.0, 0.0, 0
    for job_id, configs in job_configs.items():
        state, elapsed_str = states.get(job_id, ("UNKNOWN", ""))
        if state != "COMPLETED":
            continue
        elapsed_s = _parse_slurm_elapsed(elapsed_str)
        if elapsed_s is None:
            continue
        compute_s, counted = 0.0, 0
        for c in configs:
            r = wave_results.get(c["id"])
            if r is None:
                continue
            compute_s += steps * r["median"]
            counted += 1
        if counted == 0:
            continue
        total_elapsed_s += elapsed_s
        total_compute_s += compute_s
        n_runs += counted
    if n_runs == 0:
        return None, 0
    return max(0.0, (total_elapsed_s - total_compute_s) / n_runs), n_runs


def update_s_est(all_results: dict, fallback: float) -> float:
    """Median of this run's measured medians, else `fallback`: later stages must size jobs off
    what this run measured, not the stale pre-round0 estimate."""
    medians = [r["median"] for r in all_results.values() if r.get("median")]
    if not medians:
        return fallback
    return statistics.median(medians)


def _fallback_s_per_step(nl: dict, ranks: int, omp: int) -> float:
    """s/step of the uncalibrated cell-step model (calibration fallback_s_per_cell_step_ns)."""
    max_dom = int(wtune_predict.g(nl, "domains", "max_dom", 1))
    cal = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    return (cal["fallback_s_per_cell_step_ns"] * 1e-9
            * wtune_predict.nested_equivalent_cells(nl, max_dom) / max(int(ranks) * int(omp), 1))


def rebuild_segment_timeout_min(case_dir: str, spinup_min: float, ranks: int, omp: int = 1,
                                safety: float = 3.0, floor_min: int = 30) -> int:
    """wrf.exe watchdog (min) of one rebuild segment: the cell-step model x `safety`, never below
    `floor_min`; 90 when the namelist cannot be read. It only guards hangs, so it errs long
    (coldstart's 15 min killed every maria1km segment, 2026-09-27)."""
    try:
        nl = wtune_predict.parse_namelist(os.path.join(case_dir, "namelist.input"))
        _n, _run_s, dt, _note = wtune_predict.get_d01_steps(nl)
        stepra = wtune_predict.steps_per_radiation_call(
            float(wtune_predict.g(nl, "physics", "radt", None)), dt)
        max_dom = int(wtune_predict.g(nl, "domains", "max_dom", 1))
        cal = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
        s_step = (cal["fallback_s_per_cell_step_ns"] * 1e-9
                  * wtune_predict.nested_equivalent_cells(nl, max_dom) / max(ranks * omp, 1))
        steps = (spinup_min * 60.0) / dt + max(30, 2 * stepra)
        est_s = steps * s_step * 1.5 + 120.0          # x1.5 radiation steps, + 120 s init
        return max(floor_min, int(math.ceil(safety * est_s / 60.0)))
    except Exception:
        return 90


def rad_factor(steps: float, stepra: int | None, r_rad: float = RANKING_R_RAD) -> float:
    """Mean cost of `steps` consecutive steps over a normal step: ceil(steps/stepra) radiation
    steps at r_rad each; 1.0 without a stepra."""
    if not stepra or steps <= 0:
        return 1.0
    return 1.0 + math.ceil(float(steps) / stepra) * (r_rad - 1.0) / float(steps)


def ranking_cost_model(startup_s: float, restart_read_s: float, steps: int, spin_steps: float,
                       stepra: int | None, s_step: float) -> dict:
    """The running-time model of a point job, returned with its inputs:
    per_run_s = startup + restart read + steps x s_step x rad_factor(steps);
    spin_s = startup + spin_steps x s_step x rad_factor(spin_steps)."""
    f_run, f_spin = rad_factor(steps, stepra), rad_factor(spin_steps, stepra)
    return dict(per_run_s=startup_s + restart_read_s + steps * s_step * f_run,
                spin_s=startup_s + spin_steps * s_step * f_spin, s_step=s_step, steps=steps,
                spin_steps=spin_steps, stepra=stepra, rad_factor_run=f_run,
                rad_factor_spin=f_spin, r_rad=RANKING_R_RAD, startup_s=startup_s,
                restart_read_s=restart_read_s)


def _restart_read_s(ctx: Ctx) -> float:
    return RESTART_READ_S_BY_IO.get(int(ctx.restart_io_form), 120.0)


def stage2_concurrency(ctx: Ctx) -> int:
    """Point jobs assumed to run at once: ctx.stage1_max_running, at least 1, at most
    floor(max_nodes/nodes)."""
    cap = compute_slots(ctx.max_nodes, ctx.nodes)
    return max(1, min(cap, int(ctx.stage1_max_running or 0) or 1))


def anchor_end_of(anchor: dict) -> dict:
    """ANCHOREND: the ANCHOR re-measured as the last run of a ranking job (drift probe)."""
    return dict(anchor, id=ANCHOR_END_ID, config_source="anchor_end")


def wrap_with_anchors(configs: list[dict], anchor: dict | None) -> list[dict]:
    """[ANCHOR] + configs (any ANCHOR/ANCHOREND removed, order kept) + [ANCHOREND]; idempotent.
    RANKING_START=spinup spins up on the ANCHOR row and REPS=1 keeps file order."""
    if anchor is None:
        raise ValueError("wrap_with_anchors: no ANCHOR config -- RANKING_START=spinup needs one")
    rest = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    return [dict(anchor, id="ANCHOR", config_source="anchor")] + rest + [anchor_end_of(anchor)]


def ranking_context(case_id: str, cases_json: str, spinup_min: float, steps: int, anchor: dict,
                    restart_io_form: int = 14, spinup_timeout_s: int | None = None,
                    strict_restart: int = 1) -> dict:
    """The ranking context measure_candidates_at_points submits with."""
    return dict(case_id=case_id, cases_json=cases_json, ranking_start="spinup",
                spinup_minutes=int(spinup_min), strict_restart=int(strict_restart),
                spinup_timeout_s=spinup_timeout_s, anchor=anchor, steps=int(steps),
                restart_io_form=int(restart_io_form))


def ranking_export(ranking: dict, time_est: dict) -> list[tuple]:
    """[(KEY, VALUE), ...] for scc_sweep.build_sbatch_command's extra_export."""
    spin_to = ranking.get("spinup_timeout_s") or time_est.get("spinup_timeout_s") \
        or time_est["timeout_s"]
    return [("CASES_JSON", ranking["cases_json"]), ("RANKING_START", ranking["ranking_start"]),
            ("SPINUP_MINUTES", ranking["spinup_minutes"]),
            ("RANKING_STRICT_RESTART", ranking["strict_restart"]),
            ("SPINUP_TIMEOUT_S", int(spin_to)),
            ("RANKING_RESTART_IO_FORM", ranking["restart_io_form"])]


def _job_time_est(ctx: Ctx, n_or_pairs: int | list, reps: int, s_est: float | None) -> dict:
    """estimate_run_time for ctx."""
    if isinstance(n_or_pairs, int):
        return estimate_run_time(n_or_pairs, reps, ctx.steps, s_est=s_est,
                                 startup_s=ctx.startup_s)
    return estimate_run_time(n_or_pairs, reps, ctx.steps, startup_s=ctx.startup_s)
