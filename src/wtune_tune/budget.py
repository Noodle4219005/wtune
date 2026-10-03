"""Run-time estimates and budget checks: t0 job sizing, the predicted-mode ranking-job cost model
and stage reserves (decision 0055), and the ranking-job shape they size (ANCHOR/ANCHOREND
wrapping, ranking context, sbatch exports)."""
from __future__ import annotations

import math
import os
import statistics
from typing import Callable

from ._deps import sweep, wtune_predict
from .common import ANCHOR_END_ID, _fmt_hms, _parse_slurm_elapsed, compute_slots
from .context import DEFAULT_STARTUP_S, Ctx

TOP_K_STAGE2 = 3                       # stage-2 shortlist size (plus ANCHOR/ANCHOREND)
RANKING_R_RAD = 3.93                   # radiation-step cost / normal step: calibration's day level
                                       # (r_rad_default_day) -- P* is the most daylit point
RESTART_READ_S_BY_IO = {14: 30.0, 2: 120.0}    # s/run restart read on top of startup_s: ADIOS2 BP
                                       # vs serial netCDF (conus25 11.5 GB, job 1231: 120 s); zeroed
                                       # once a ranking job's per-run overhead is measured


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
    """The running-time model of every predicted-mode estimate, returned with its inputs:
    per_run_s = startup + restart read + steps x s_step x rad_factor(steps);
    spin_s = startup + spin_steps x s_step x rad_factor(spin_steps)."""
    f_run, f_spin = rad_factor(steps, stepra), rad_factor(spin_steps, stepra)
    return dict(per_run_s=startup_s + restart_read_s + steps * s_step * f_run,
                spin_s=startup_s + spin_steps * s_step * f_spin, s_step=s_step, steps=steps,
                spin_steps=spin_steps, stepra=stepra, rad_factor_run=f_run,
                rad_factor_spin=f_spin, r_rad=RANKING_R_RAD, startup_s=startup_s,
                restart_read_s=restart_read_s)


def describe_cost_model(m: dict) -> str:
    return ("per-run %.0fs = startup %.0fs + restart read %.0fs + %d steps x %.3fs x rad factor "
            "%.3f (r_rad %.2f every %s steps); spin-up %.0fs (%d steps, rad factor %.3f)"
            % (m["per_run_s"], m["startup_s"], m["restart_read_s"], m["steps"], m["s_step"],
               m["rad_factor_run"], m["r_rad"], m["stepra"] or "-", m["spin_s"], m["spin_steps"],
               m["rad_factor_spin"]))


def _restart_read_s(ctx: Ctx) -> float:
    return ctx.restart_read_s if ctx.restart_read_s is not None else \
        RESTART_READ_S_BY_IO.get(int(ctx.restart_io_form), 120.0)


def estimate_stage2_s(n_points: int, n_configs: int, steps: int, spinup_min: float, dt: float,
                      s_step: float, slots: int, startup_s: float, restart_read_s: float = 30.0,
                      stepra: int | None = None) -> float:
    """Expected RUNNING time of stage 2: one point job (spin-up + n_configs runs) per point,
    `slots` at a time, so ceil(n_points/slots) waves."""
    m = ranking_cost_model(startup_s, restart_read_s, steps, spinup_min * 60.0 / dt, stepra,
                           s_step)
    waves = int(math.ceil(float(n_points) / max(1, int(slots))))
    return waves * (m["spin_s"] + n_configs * m["per_run_s"])


def _reserve_s_step(ctx: Ctx) -> float | None:
    """s/step of the stage-2 reserve: median of stage 1's per-job anchor normalizers, else the
    prior."""
    vals = [j["s_anchor_job"] for j in ctx.stage1_jobs.values() if j.get("s_anchor_job")]
    return statistics.median(vals) if vals else ctx.s_prior


def stage2_concurrency(ctx: Ctx) -> int:
    """Stage-2 point jobs assumed to run at once: the most stage-1 jobs OBSERVED running
    together, at least 1, at most floor(max_nodes/nodes) (conus25-pred3: assuming the cap
    overran stage 2 on a busy cluster)."""
    cap = compute_slots(ctx.max_nodes, ctx.nodes)
    return max(1, min(cap, int(ctx.stage1_max_running or 0) or 1))


def _stage2_estimate(ctx: Ctx, n_configs: int, s_step: float) -> float:
    r = ctx.reserve
    return estimate_stage2_s(r["n_points"], n_configs, r["steps"], r["spinup_min"], r["dt"],
                             s_step, stage2_concurrency(ctx), ctx.startup_s, _restart_read_s(ctx),
                             stepra=r.get("stepra"))


def _reserve_k(ctx: Ctx, s_step: float) -> int:
    """Candidates the stage-2 reserve holds room for: TOP_K_STAGE2 once a stage-1 job reported;
    before that the largest k that still leaves one minimal stage-1 job (else the full reserve
    blocks stage 1 at concurrency 1; fit_stage2 trims stage 2 later)."""
    if ctx.stage1_jobs:
        return TOP_K_STAGE2
    # one minimal stage-1 job: pstar = spin-up + 3 ranking runs; t0 = 3 plain t0-STEPS runs
    min_job = _job_time_est(ctx, 3, 1, s_step)["total_run_s"]
    for k in range(TOP_K_STAGE2, 0, -1):
        if ctx.elapsed() + min_job + ctx.stage1_extra_reserve_s \
                + _stage2_estimate(ctx, k + 2, s_step) <= ctx.budget_s:
            return k
    return 1


def stage2_reserve_s(ctx: Ctx) -> float:
    """Running time stage 1 leaves for stage 2: _reserve_k candidates + ANCHOR/ANCHOREND at every
    point (0.0 outside predicted-mode stage 1)."""
    r = ctx.reserve
    if not _stage1_active(ctx) or not r:
        return 0.0
    s_step = _reserve_s_step(ctx)
    if not s_step:
        return 0.0
    return _stage2_estimate(ctx, _reserve_k(ctx, s_step) + 2, s_step)


def stage1_reserve_s(ctx: Ctx) -> float:
    """What stage-1 decisions hold back from --minutes: the stage-2 reserve (+ extra) when
    reserve_in_stage1 (pstar); 0.0 in stage1=t0, where stage 2 is a separate announced cost."""
    if not _stage1_active(ctx) or not ctx.reserve_in_stage1:
        return 0.0
    return stage2_reserve_s(ctx) + ctx.stage1_extra_reserve_s


def _stage1_active(ctx: Ctx) -> bool:
    """True during predicted-mode stage 1, in either stage1 mode."""
    return bool(ctx.ranking or ctx.stage1)


def _stage1_anchor(ctx: Ctx) -> dict | None:
    if ctx.ranking:
        return ctx.ranking["anchor"]
    return (ctx.stage1 or {}).get("anchor")


def describe_stage2_reserve(ctx: Ctx) -> str:
    """One log line: how the stage-2 reserve was sized (to judge it against reality)."""
    r = ctx.reserve
    s_step = _reserve_s_step(ctx)
    if not _stage1_active(ctx) or not r or not s_step:
        return "stage-2 reserve: none"
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), r["steps"],
                           r["spinup_min"] * 60.0 / r["dt"], r.get("stepra"), s_step)
    conc = stage2_concurrency(ctx)
    waves = int(math.ceil(float(r["n_points"]) / conc))
    return ("stage-2 %s %s = %d wave(s) [%d point job(s), %d concurrent: observed max %d "
            "stage-1 job(s) running at once, cap floor(max_nodes/nodes)=%d] x (spin-up + %d "
            "runs); %s"
            % ("reserve" if ctx.reserve_in_stage1 else
               "estimate (NOT held back from --minutes, stage1=t0)",
               _fmt_hms(stage2_reserve_s(ctx)), waves, r["n_points"], conc,
               ctx.stage1_max_running, compute_slots(ctx.max_nodes, ctx.nodes),
               _reserve_k(ctx, s_step) + 2, describe_cost_model(m)))


def _stage1_may_start(ctx: Ctx, job_run_s: float) -> tuple[bool, str | None]:
    """(ok, reason): a stage-1 job may start while elapsed + its run time + the stage-2 reserve
    fits the budget; always ok outside predicted mode."""
    if not _stage1_active(ctx):
        return True, None
    reserve = stage1_reserve_s(ctx)
    if ctx.elapsed() + job_run_s + reserve > ctx.budget_s:
        return False, ("elapsed %s + this job %s + stage-2 reserve %s > budget %s"
                       % (_fmt_hms(ctx.elapsed()), _fmt_hms(job_run_s), _fmt_hms(reserve),
                          _fmt_hms(ctx.budget_s)))
    return True, None


def _stage1_cutoff(ctx: Ctx) -> Callable[[], bool] | None:
    """pending_cutoff of stage-1 waits: true once elapsed + the stage-2 reserve reaches the
    budget; None outside predicted mode."""
    if not _stage1_active(ctx):
        return None
    return lambda: ctx.elapsed() + stage1_reserve_s(ctx) >= ctx.budget_s


def _ledger_case(ctx: Ctx) -> str:
    """Case id of the ledgers / RUNDIRs: the rebuilt point's in pstar, else ctx.case."""
    return ctx.ranking["case_id"] if ctx.ranking else ctx.case


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


def _ranking_wrap(ctx: Ctx, configs: list[dict]) -> list[dict]:
    """wrap_with_anchors in predicted-mode stage 1, else `configs` itself (same list)."""
    anchor = _stage1_anchor(ctx) if _stage1_active(ctx) else None
    return wrap_with_anchors(configs, anchor) if anchor else configs


def ranking_context(case_id: str, cases_json: str, spinup_min: float, steps: int, anchor: dict,
                    restart_io_form: int = 14, spinup_timeout_s: int | None = None,
                    strict_restart: int = 1) -> dict:
    """The ranking context submit_configs_job / measure_candidates_at_points submit with."""
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


def _padded_limit(total_run_s: float) -> tuple[float, str]:
    """(job_hours, time_limit): estimate_run_time's padding (x1.5 + 10 min, capped)."""
    padded = (total_run_s / 3600.0) * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN / 60.0
    capped = min(padded, sweep.TIME_CAP_HOURS)
    return capped, sweep.campaign_experiment._format_hms(capped * 60.0)


def _job_time_est(ctx: Ctx, n_or_pairs: int | list, reps: int, s_est: float | None,
                  ranking: dict | None = None) -> dict:
    """estimate_run_time for ctx; with a ranking context the runs follow ranking_cost_model and
    the shared ANCHOR spin-up is added (raw total and padded --time, own SPINUP_TIMEOUT_S)."""
    ranking = ctx.ranking if ranking is None else ranking
    if isinstance(n_or_pairs, int):
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                s_est=s_est, startup_s=ctx.startup_s)
    else:
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                startup_s=ctx.startup_s)
    if not ranking:
        return est
    dt = float(ranking.get("dt") or 0) or None
    spin_steps = (ranking["spinup_minutes"] * 60.0 / dt) if dt else 0.0
    steps = ranking["steps"]
    pairs = ([(None, s_est)] * n_or_pairs if isinstance(n_or_pairs, int)
             else [(c, s) for c, s in n_or_pairs])
    runs = [ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                               ranking.get("stepra"), s)["per_run_s"] for _c, s in pairs]
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                           ranking.get("stepra"), s_est if s_est else (pairs[0][1] if pairs else 0))
    total = reps * sum(runs) + m["spin_s"]
    est = dict(est, total_run_s=total, cost_model=m,
               timeout_s=max(sweep.TIMEOUT_MIN_S,
                             int(math.ceil(sweep.TIMEOUT_MULTIPLIER * max(runs or [0])))),
               spinup_timeout_s=max(sweep.TIMEOUT_MIN_S,
                                    int(math.ceil(sweep.TIMEOUT_MULTIPLIER * m["spin_s"]))))
    est["job_hours"], est["time_limit"] = _padded_limit(est["total_run_s"])
    return est
