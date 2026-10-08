"""Rebuilt prediction points (decision 0055): plan them, rebuild each from ERA5 once, measure
candidates from each point's shared restart in one job per point, gate and rank the results
by predicted whole-run time, and pick which points to measure."""
from __future__ import annotations

import concurrent.futures
import copy
import datetime
import json
import math
import os
import re
import threading
from typing import Any

from ._deps import coldstart, sweep, wtune_predict, wtune_probe_plan
from .budget import (_fallback_s_per_step, _restart_read_s, ranking_context, ranking_cost_model,
                     ranking_export, stage2_concurrency, wrap_with_anchors)
from .common import ANCHOR_END_ID, _cluster_rank, _fmt_hms, _log_key
from .context import Ctx
from .jobs import evaluate_job_gates
from . import budget as _budget
from . import jobs as _jobs

PREDICTED_FORCING = "era5_arco"        # wtune_rebuild_predict.ERA5_PIN (1 h cadence, with cloud)
PREDICTED_SPINUP_MIN = 20              # min of ERA5 spin-up (rebuild phase 3, 2026-09-27)
POINT_MODES = ("default", "no_end", "fused")
_BUDGET_LEFT = object()                # fit_stage2 default: what is left of --minutes


def ranking_job_timeouts(ctx: Ctx, configs: list[dict], steps: int, spinup_min: float, dt: float,
                         s_by_cid: dict | None = None, nl: dict | None = None,
                         safety: float = 3.0, restart_read_s: float | None = None,
                         stepra: int | None = None) -> tuple[int, int, str, float]:
    """(TIMEOUT_S, SPINUP_TIMEOUT_S, --time, expected s) of one point job from ranking_cost_model,
    s/step per config = its measured median, else the cell-step model, else the conservative
    fallback. The slowest run sets TIMEOUT_S (one per job) at `safety` x expected."""
    s_by_cid = s_by_cid or {}

    def s_of(c):
        if s_by_cid.get(c["id"]):
            return s_by_cid[c["id"]]
        if s_by_cid.get("ANCHOR") and c.get("config_source") in ("anchor", "anchor_end"):
            return s_by_cid["ANCHOR"]
        try:
            return _fallback_s_per_step(nl, c["ranks"], c.get("omp", 1))
        except Exception:
            return sweep.CONSERVATIVE_S_PER_STEP

    rr = _restart_read_s(ctx) if restart_read_s is None else restart_read_s
    spin_steps = spinup_min * 60.0 / dt
    per_run = [ranking_cost_model(ctx.startup_s, rr, steps, spin_steps, stepra,
                                  s_of(c))["per_run_s"] for c in configs]
    cand_timeout = max(sweep.TIMEOUT_MIN_S, int(math.ceil(safety * max(per_run))))
    anchor = next((c for c in configs if c.get("id") == "ANCHOR"), configs[0])
    spin = ranking_cost_model(ctx.startup_s, rr, steps, spin_steps, stepra,
                              s_of(anchor))["spin_s"]
    spin_timeout = max(sweep.TIMEOUT_MIN_S, int(math.ceil(safety * spin)))
    expected = spin + sum(per_run)
    limit_s = max(expected * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN * 60.0,
                  spin_timeout + sum(per_run) + sweep.TIME_PAD_MIN * 60.0)
    limit_h = min(limit_s / 3600.0, sweep.TIME_CAP_HOURS)
    return (cand_timeout, spin_timeout, sweep.campaign_experiment._format_hms(limit_h * 60.0),
            expected)


def choose_primary_point(points: list, sample_points: Any) -> tuple[int, Any, float]:
    """P* (decision 0055): (index, point, sunlit fraction) of the most sunlit planned point
    (ties: the earliest) -- the daylit radiation + convective load discriminates decompositions
    most; without sample points every fraction is 0 and P* is the earliest."""
    if not points:
        raise ValueError("choose_primary_point: no planned points")
    best = None
    for i, p in enumerate(points):
        frac = (wtune_predict.sunlit_fraction(sample_points, p.at) if sample_points else None) or 0.0
        key = (-frac, p.at, i)
        if best is None or key < best[0]:
            best = (key, i, p, frac)
    return best[1], best[2], best[3]


def rebuilt_case_id(case: str, index: int) -> str:
    """Underscore-free case id of a rebuilt point (RUN_ID splits the case at the first '_')."""
    return re.sub(r"[^A-Za-z0-9]", "", case) + "p%d" % index


def rebuilt_steps(stepra: int) -> int:
    """Candidate STEPS at a rebuilt point: max(30, 2*stepra) + 2 (two radiation steps, the
    dropped first step, one spare)."""
    return max(30, 2 * int(stepra)) + 2


def rebuilt_measure_steps(steps: int, dt: float) -> int:
    """coldstart measure_steps covering the candidates' run: strict_campaign runs ceil(STEPS*dt/60)
    whole minutes, so the window needs that many minutes of steps after the spin-up."""
    run_min = int(math.ceil(steps * dt / 60.0))
    return int(math.ceil(run_min * 60.0 / dt))


def write_rebuilt_case_entry(case: str, case_entry: dict, index: int, point: Any, plan: dict,
                             plan_dir: str, spinup_min: float, stepra: int, anchor: dict,
                             steps: int | None = None) -> tuple[str, dict]:
    """(case_id, entry): a strict_campaign case entry for one COMPLETED real_only plan. Writes
    <plan_dir>/namelist.template.input (coldstart's window renderer, run_seconds=0) and links the
    case's input_files into win_0/real; writes only under plan_dir. RuntimeError when real.exe
    output is missing."""
    real_dir = os.path.join(plan_dir, "win_0", "real")
    need = ["namelist.input", "wrfbdy_d01"] + ["wrfinput_d%02d" % d for d in
                                                range(1, int(plan.get("max_dom", 1)) + 1)]
    missing = [f for f in need if not os.path.exists(os.path.join(real_dir, f))]
    if missing:
        raise RuntimeError("rebuilt point %d (%s): real.exe output missing in %s: %s"
                           % (index, point.at.isoformat(), real_dir, ", ".join(missing)))
    with open(os.path.join(real_dir, "namelist.input")) as f:
        real_text = f.read()
    T = datetime.datetime.strptime(plan["windows"][0]["t"], "%Y-%m-%d_%H:%M:%S")
    dt = float(plan["dt_s"])
    win_steps = int(plan["measure_steps"]) + int(round(float(plan["spinup_min"]) * 60.0 / dt))
    txt = coldstart.build_wrf_window_namelist_text(real_text, T, win_steps, dt, anchor["nx"],
                                                   anchor["ny"], anchor.get("numtiles", 1))
    txt = coldstart.set_list(txt, "run_seconds", 0, 1)
    template = os.path.join(plan_dir, "namelist.template.input")
    with open(template, "w") as f:
        f.write(txt)
    orig_prep = sweep._expand(case_entry.get("prep_dir") or "")
    for name in case_entry.get("input_files") or []:
        dst = os.path.join(real_dir, name)
        if not os.path.lexists(dst) and orig_prep:
            os.symlink(os.path.join(orig_prep, name), dst)
    case_id = rebuilt_case_id(case, index)
    entry = copy.deepcopy(case_entry)
    # bp_dir="": real.exe still writes netCDF (read here and by the state tools); probe staging
    # converts this prep_dir once into its BP cache (configs.py stage-bp-inputs)
    entry.update(prep_dir=os.path.abspath(real_dir), bp_dir="", namelist_template=template,
                 steps=int(steps if steps is not None else rebuilt_steps(stepra)), enabled=True,
                 rebuilt_from=case, rebuilt_point_at=point.at.isoformat(),
                 rebuilt_start=T.strftime("%Y-%m-%d_%H:%M:%S"), rebuilt_spinup_min=spinup_min)
    return case_id, entry


def prepare_rebuilt_points(ctx: Ctx, points: list, anchor: dict, case_dir: str,
                           namelist_path: str, case_entry: dict, out_dir: str, spinup_min: float,
                           stepra: int, dt: float, forcing: str = PREDICTED_FORCING,
                           grib_dir: str | None = None,
                           date_shift_to: Any = None) -> list[dict]:
    """submit_rebuilt_points + finish_rebuilt_points (stage1=pstar needs the points first)."""
    return finish_rebuilt_points(ctx, submit_rebuilt_points(
        ctx, points, anchor, case_dir, namelist_path, case_entry, out_dir, spinup_min, stepra,
        dt, forcing=forcing, grib_dir=grib_dir, date_shift_to=date_shift_to))


def submit_rebuilt_points(ctx: Ctx, points: list, anchor: dict, case_dir: str,
                          namelist_path: str, case_entry: dict, out_dir: str, spinup_min: float,
                          stepra: int, dt: float, forcing: str = PREDICTED_FORCING,
                          grib_dir: str | None = None, date_shift_to: Any = None) -> dict:
    """Plan, fetch, stage and SUBMIT one ERA5 real_only rebuild job per point, in parallel
    (T = M - spinup_min); returns the handle finish_rebuilt_points waits on. Each job needs one
    node (WPS + real.exe do not depend on the decomposition; conus25-pred1 2026-09-27: 4-node
    prep jobs queued one at a time). Refuses an out_dir inside case_dir."""
    case_dir = os.path.abspath(sweep._expand(case_dir))
    out_dir = os.path.abspath(out_dir)
    if os.path.commonpath([out_dir, case_dir]) == case_dir:
        raise RuntimeError("prepare_rebuilt_points: out_dir %s is inside the case dir %s"
                           % (out_dir, case_dir))
    missing_env = [v for v in ("WPS_DIR", "WRF_TREE") if not os.environ.get(v)]
    if missing_env:
        raise RuntimeError("prepare_rebuilt_points: %s not set -- source the site env first"
                           % ", ".join(missing_env))
    steps = rebuilt_steps(stepra)
    measure_steps = rebuilt_measure_steps(steps, dt)
    one = sweep.build_anchor(ctx.case_json, ctx.cores_per_node, 1, ctx.hardware_id)
    shape = one or anchor                  # the anchor's shape when no 1-node layout is legal
    config = dict(ranks=shape["ranks"], omp=shape["omp"], nx=shape["nx"], ny=shape["ny"],
                  numtiles=shape.get("numtiles", 1), nodes=1 if one else ctx.nodes)
    seg_timeout_min = _budget.rebuild_segment_timeout_min(case_dir, spinup_min, config["ranks"],
                                                          config["omp"])
    campaign_dir = sweep.campaign_dir_for(ctx.sbatch_script)

    def _prepare_and_submit(i, p):
        pdir = os.path.join(out_dir, "pt_%d" % i)
        plan = wtune_probe_plan.plan_rebuild_segments(
            case_dir, [p], config, pdir, spinup_min=spinup_min, date_shift_to=date_shift_to,
            namelist=namelist_path, grib_dir=grib_dir, forcing=forcing,
            measure_steps=measure_steps, real_only=True)
        total_bytes, _forcing = coldstart.resolve_and_fetch_forcing(
            plan, pdir, os.path.join(pdir, "grib"), grib_dir=grib_dir)
        print("rebuilt point %d (%s): fetched %.1f MB forcing"
              % (i, p.at.isoformat(), total_bytes / 1e6), flush=True)
        sbatch_path = coldstart.stage_plan(pdir, site="scc", per_window_min=seg_timeout_min)
        cmd = ("sbatch --export=ALL,CAMPAIGN=%s %s" % (campaign_dir, sbatch_path)
               if campaign_dir else "sbatch %s" % sbatch_path)
        job_id = sweep.submit_job(cmd, sbatch_runner=ctx.sbatch_runner)
        with owned_lock:       # owned at once: another point failing must not orphan this job
            ctx.owned_jobs.append(job_id)
        print("rebuilt point %d: real_only job %s (%s)" % (i, job_id, sbatch_path), flush=True)
        return plan, pdir, job_id

    owned_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(points))) as pool:
        done = list(pool.map(lambda ip: _prepare_and_submit(*ip), enumerate(points)))
    job_ids = [d[2] for d in done]
    for j in job_ids:                      # all submitted: back to point order
        ctx.owned_jobs.remove(j)
    ctx.owned_jobs.extend(job_ids)
    return dict(points=points, done=done, job_ids=job_ids, anchor=anchor, case_entry=case_entry,
                out_dir=out_dir, spinup_min=spinup_min, stepra=stepra, dt=dt,
                steps=rebuilt_steps(stepra))


def finish_rebuilt_points(ctx: Ctx, handle: dict) -> list[dict]:
    """Wait for the rebuild jobs, write each point's case entry and ONE rebuilt_cases.json
    (CASES_JSON). Returns [{index, point, case_id, T, cases_json, prep_dir, template, steps,
    spinup_min, stepra, dt, plan_dir, job_id}]; raises when a job did not complete.
    handle["exempt_budget"] (stage1=t0: --minutes is stage 1 only) keeps the budget from
    cancelling the jobs."""
    points, done, job_ids = handle["points"], handle["done"], handle["job_ids"]
    anchor, case_entry, out_dir = handle["anchor"], handle["case_entry"], handle["out_dir"]
    spinup_min, stepra, dt, steps = (handle["spinup_min"], handle["stepra"], handle["dt"],
                                     handle["steps"])
    states = _jobs.wait_for_jobs(ctx, job_ids, "rebuild-prep",
                                 exempt_budget=bool(handle.get("exempt_budget")))
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("rebuilt-point real_only job(s) %s not COMPLETED -- stop and diagnose "
                           "read-only" % bad)
    cases_json = os.path.join(out_dir, "rebuilt_cases.json")
    entries, out = {}, []
    for i, (p, (plan, pdir, jid)) in enumerate(zip(points, done)):
        case_id, entry = write_rebuilt_case_entry(ctx.case, case_entry, i, p, plan, pdir,
                                                  spinup_min, stepra, anchor, steps=steps)
        entries[case_id] = entry
        out.append(dict(index=i, point=p, case_id=case_id,
                        T=datetime.datetime.strptime(entry["rebuilt_start"], "%Y-%m-%d_%H:%M:%S"),
                        cases_json=cases_json, prep_dir=entry["prep_dir"],
                        template=entry["namelist_template"], steps=steps, spinup_min=spinup_min,
                        stepra=stepra, dt=dt, plan_dir=pdir, job_id=jid))
    with open(cases_json, "w") as f:
        json.dump({"cases": entries}, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


def point_runs(wrapped: list[dict], mode: str) -> list[dict]:
    """The configs a point job runs: default = ANCHOR + candidates + ANCHOREND; no_end
    (ANCHOR_END=0) drops ANCHOREND; fused (POINT_FUSED=1) runs only the single candidate,
    spin-up and window in one wrf.exe (decisions 0055/0059)."""
    if mode == "fused":
        return [c for c in wrapped if c["id"] not in ("ANCHOR", ANCHOR_END_ID)]
    if mode == "no_end":
        return [c for c in wrapped if c["id"] != ANCHOR_END_ID]
    return list(wrapped)


def point_export(mode: str) -> list[tuple]:
    """Extra sbatch --export entries of a point-job mode (WINDOW_EXACT=1: window = STEPS*dt)."""
    if mode == "fused":
        return [("POINT_FUSED", 1), ("ANCHOR_END", 0), ("WINDOW_EXACT", 1)]
    if mode == "no_end":
        return [("ANCHOR_END", 0)]
    return []


def choose_point_mode(configs: list[dict], anchor: dict | None) -> str:
    """fused for one candidate without a window frame (winner-only), no_end for ANCHOR + one
    config, else default (drift gate on)."""
    cands = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    framed = any(c.get("window_frame") for c in cands) or bool((anchor or {}).get("window_frame"))
    if len(cands) == 1 and not framed:
        return "fused"
    if len(cands) + 1 <= 2:
        return "no_end"
    return "default"


def _measure_run(ctx: Ctx, rp: dict, cid: str, job_id: str, dt: float,
                 first_global_step: float | None = None) -> tuple[Any, float]:
    """(PointMeasurement, first global step) of one point-job run from its RUNDIR's steps.txt;
    `.at` = the measured window's midpoint on the case timeline. Raises without steps.txt."""
    rundir = os.path.join(ctx.ledger_dir, "%s_%s_r1_%s" % (rp["case_id"], cid, job_id))
    steps_path = os.path.join(rundir, "steps.txt")
    if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
        raise RuntimeError("%s missing/empty" % steps_path)
    # a fused row keeps only the post-spin-up steps: its first step is known, the rsl log says 1
    g0 = (first_global_step if first_global_step is not None else
          wtune_probe_plan.first_global_step_from_rsl(os.path.join(rundir, "rsl.error.0000"),
                                                      rp["T"], dt))
    at = rp["point"].at + datetime.timedelta(seconds=rp["steps"] * dt / 2.0)
    pm = wtune_probe_plan.measurement_from_steps_file(
        steps_path, rp["stepra"], at, rp["point"].reason, "rebuild", first_global_step=g0)
    return pm, g0


def measure_candidates_at_points(ctx: Ctx, configs: list[dict], rebuilt_points: list[dict],
                                 anchor: dict, s_by_cid: dict | None = None,
                                 restart_io_form: int | None = None, name_prefix: str = "stage2",
                                 mode: str | None = None) -> tuple[dict, list, list]:
    """One point job per rebuilt point (ANCHOR + configs + ANCHOREND from the point's shared
    restart), all submitted before waiting; exempt from the running budget once started. Every
    run must pass the job gates and resume at spin-up + 1, else RuntimeError (nothing dropped).
    Returns ({cid: [PointMeasurement] in point order}, gate records, job ids)."""
    restart_io_form = restart_io_form or ctx.restart_io_form
    mode = mode or "default"
    if mode not in POINT_MODES:
        raise ValueError("unknown point-job mode %r" % (mode,))
    wrapped = wrap_with_anchors(configs, anchor)
    runs = point_runs(wrapped, mode)              # what the runner actually executes
    # rebuilt prep_dirs are new: convert once before the point jobs (scc jobs 2884/2885)
    _jobs.ensure_bp_inputs(ctx, [(rp["case_id"], rp["cases_json"]) for rp in rebuilt_points],
                           os.path.join(ctx.out_dir, name_prefix, "bp_convert"))
    jobs = _submit_point_jobs(ctx, wrapped, runs, rebuilt_points, anchor, s_by_cid,
                              restart_io_form, name_prefix, mode)
    job_ids = [j for _rp, j in jobs]
    states = _jobs.wait_for_jobs(ctx, job_ids, name_prefix, exempt_budget=True)
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("%s job(s) %s not COMPLETED -- stop and diagnose read-only"
                           % (name_prefix, bad))
    meas_by_cid, gates = _gate_point_jobs(ctx, jobs, runs, name_prefix, mode)
    return meas_by_cid, gates, job_ids


def _submit_point_jobs(ctx: Ctx, wrapped: list[dict], runs: list[dict], rebuilt_points: list[dict],
                       anchor: dict, s_by_cid: dict | None, restart_io_form: int,
                       name_prefix: str, mode: str) -> list[tuple[dict, str]]:
    """Submit one point job per rebuilt point (timeouts from ranking_job_timeouts). Returns
    [(rebuilt point, job id)]."""
    nl = None
    try:
        nl = wtune_predict.parse_namelist(ctx.namelist_path)
    except Exception:
        pass
    jobs = []
    for rp in rebuilt_points:
        dt = float(rp["dt"])
        cand_to, spin_to, time_limit, expected = ranking_job_timeouts(
            ctx, runs, rp["steps"], rp["spinup_min"], dt, s_by_cid=s_by_cid, nl=nl,
            stepra=rp.get("stepra"))
        job_out = os.path.join(ctx.out_dir, name_prefix, "pt_%d" % rp["index"])
        os.makedirs(job_out, exist_ok=True)
        configs_path = os.path.join(job_out, "configs.json")
        # the ANCHOR placeholder names the spin-up layout; ANCHOREND is dropped with ANCHOR_END=0
        sweep.write_configs_json(configs_path, [c for c in wrapped if mode == "default"
                                                or c["id"] != ANCHOR_END_ID])
        ranking = ranking_context(rp["case_id"], rp["cases_json"], rp["spinup_min"], rp["steps"],
                                  anchor, restart_io_form=restart_io_form,
                                  spinup_timeout_s=spin_to)
        fresh = _jobs.fresh_input_export(ctx, rp["case_id"], job_out, len(wrapped) + 2,
                                         rp["cases_json"])
        cmd = sweep.build_sbatch_command(
            rp["case_id"], ctx.nodes, ctx.cores_per_node, time_limit, cand_to, 1, rp["steps"],
            os.path.abspath(configs_path), job_out, env_source=ctx.env_source,
            sbatch_script=ctx.sbatch_script, campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
            extra_export=ranking_export(ranking, {"timeout_s": cand_to}) + point_export(mode)
            + fresh)
        jid = sweep.submit_job(cmd, sbatch_runner=ctx.sbatch_runner)
        ctx.owned_jobs.append(jid)
        print("%s point %d (%s, case %s): job %s, %d runs, TIMEOUT_S=%d SPINUP_TIMEOUT_S=%d "
              "--time=%s (expected %s)" % (name_prefix, rp["index"], rp["point"].at.isoformat(),
                                           rp["case_id"], jid, len(runs), cand_to, spin_to,
                                           time_limit, _fmt_hms(expected)) + (
            " [%s]" % mode if mode != "default" else ""))
        jobs.append((rp, jid))
    return jobs


def _gate_point_jobs(ctx: Ctx, jobs: list[tuple[dict, str]], runs: list[dict], name_prefix: str,
                     mode: str) -> tuple[dict, list]:
    """Measure every run of the finished point jobs and gate each job (records go to
    ctx.ranking_gates too). Returns ({cid: [PointMeasurement]}, gates); RuntimeError naming
    every failed job."""
    meas_by_cid: dict = {}
    gates, errors = [], []
    for rp, jid in jobs:
        dt = float(rp["dt"])
        rows = ctx.ledger_reader(rp["case_id"], [jid])
        s_by, extra = {}, []
        # resumed at T + spinup: first global step = spinup*60/dt + 1 exactly (predicted_spinup_min)
        expect_g0 = rp["spinup_min"] * 60.0 / dt + 1.0
        for c in runs:
            try:
                pm, g0 = _measure_run(ctx, rp, c["id"], jid, dt,
                                      first_global_step=expect_g0 if mode == "fused" else None)
            except Exception as e:
                extra.append("%s: no measurement (%s: %s)" % (c["id"], type(e).__name__, e))
                continue
            if g0 != expect_g0:
                extra.append("%s: first global step %d, expected %g -- run did not resume at "
                             "the shared restart" % (c["id"], g0, expect_g0))
            s_by[c["id"]] = pm.s_normal
            if c["id"] != ANCHOR_END_ID:
                meas_by_cid.setdefault(c["id"], []).append(pm)
        # drift gate only when ANCHOREND ran; a fused row's restart_used=1 proves no restart read
        rec = evaluate_job_gates(jid, rows, s_by, [c["id"] for c in runs],
                                 check_drift=(mode == "default"))
        rec["point_mode"] = mode
        rec["failures"].extend(extra)
        rec.update(ok=not rec["failures"], stage=name_prefix, point_index=rp["index"],
                   point_at=rp["point"].at.isoformat(), case_id=rp["case_id"])
        gates.append(rec)
        ctx.ranking_gates.append(rec)
        if not rec["ok"]:
            errors.append("job %s (point %d %s): %s" % (jid, rp["index"],
                                                        rp["point"].at.isoformat(),
                                                        "; ".join(rec["failures"])))
    if errors:
        raise RuntimeError("%s validity gates FAILED -- no ranking is reported from these "
                           "measurements:\n  " % name_prefix + "\n  ".join(errors))
    return meas_by_cid, gates


def rank_by_predicted_total(meas_by_cid: dict, run_start: Any, run_end: Any, dt: float,
                            stepra: int, calibration: dict, sample_points: Any, case_dir: str,
                            nl: Any, ranks_by_cid: dict, nodes: int) -> list[dict]:
    """[{cid, pred_total_s, pred_low_s, pred_high_s, result}] from wtune_probe_plan.combine() per
    cid (never ANCHOREND), NOISE-clustered ascending (ANCHOR wins an in-noise tie)."""
    entries = []
    for cid, pms in meas_by_cid.items():
        if cid == ANCHOR_END_ID or not pms:
            continue
        res = wtune_probe_plan.combine(
            sorted(pms, key=lambda m: m.at), run_start, run_end, dt, stepra,
            calibration=calibration, sample_points=sample_points, case_dir=case_dir, nl=nl,
            ranks=ranks_by_cid.get(cid), nodes=nodes)
        entries.append(dict(cid=cid, pred_total_s=res["total_s"], pred_low_s=res["range_low_s"],
                            pred_high_s=res["range_high_s"], result=res))
    order_index = {cid: i for i, cid in enumerate(sorted(e["cid"] for e in entries))}
    return _cluster_rank(entries, lambda e: _log_key(e["pred_total_s"]), "ANCHOR", order_index)


def _predicted_winner_text(entry: dict) -> str:
    """The full-run prediction lines of a rank_by_predicted_total entry."""
    res = entry["result"]
    lines = ["=== full-run prediction (rank_by=predicted, %s, %d rebuilt point(s)) ==="
             % (entry["cid"], len(res["points"])),
             "compute_total: %s   init: %.1fs   TOTAL: %s (range %s - %s, +-%.1f%% drift)"
             % (_fmt_hms(res["compute_total_s"]), res["init_s"], _fmt_hms(res["total_s"]),
                _fmt_hms(res["range_low_s"]), _fmt_hms(res["range_high_s"]), res["drift_pct"]),
             "r_rad: night=%.3f day=%.3f source=%s" % (res["r_night"], res["r_day"],
                                                        res["r_source"])]
    if res.get("init_note"):
        lines.append("NOTE: " + res["init_note"])
    return "\n".join(lines)


def predicted_spinup_min(dt: float, floor_min: int = PREDICTED_SPINUP_MIN,
                         max_min: int = 60) -> int:
    """Smallest whole minute >= floor_min that is a multiple of dt, so the shared restart falls
    on a time step (dt 36 -> 21; dt 15/10/25 -> 20). SystemExit when none up to max_min."""
    for m in range(int(floor_min), int(max_min) + 1):
        if abs((m * 60.0) / float(dt) - round((m * 60.0) / float(dt))) < 1e-9:
            return m
    raise SystemExit("--rank-by predicted: no whole-minute spin-up in [%d, %d] min is a multiple "
                     "of time_step=%gs -- the shared restart would not fall on a time step"
                     % (floor_min, max_min, dt))


def plan_predicted_points(ctx: Ctx, spinup_min: int = PREDICTED_SPINUP_MIN) -> tuple:
    """wtune_rebuild_predict.plan with the ERA5 pin, `spinup_min` and probe.max_points on the
    case's own namelist: (nl, run_start, run_end, dt, stepra, points, notes)."""
    import wtune_rebuild_predict
    return wtune_rebuild_predict.plan(
        sweep._expand(ctx.case_json["prep_dir"]), PREDICTED_FORCING, spinup_min,
        ctx.probe_cfg.get("max_points"), namelist_path=ctx.namelist_path)


def fit_stage2(ctx: Ctx, top: list[dict], anchor: dict, rebuilt: list[dict], s_by_cid: dict,
               nl: Any = None, left_s: Any = _BUDGET_LEFT,
               mode: str | None = None) -> tuple[list[dict], float | None, str | None]:
    """(top_fit, expected running s, skip_reason): drop the lowest-ranked candidates until
    ceil(points/stage2_concurrency) waves of the slowest point job fit `left_s` (default: what
    is left of --minutes; None: no cap). Skips with a reason when not even the best candidate
    (or ANCHOR alone) fits."""
    waves = int(math.ceil(float(len(rebuilt)) / stage2_concurrency(ctx)))
    left = ctx.budget_s - ctx.elapsed() if left_s is _BUDGET_LEFT else left_s
    est = None
    for k in range(len(top), min(1, len(top)) - 1, -1):
        wrapped = point_runs(wrap_with_anchors(top[:k], anchor), mode or "default")
        per_job = max(ranking_job_timeouts(ctx, wrapped, rp["steps"], rp["spinup_min"],
                                           float(rp["dt"]), s_by_cid=s_by_cid, nl=nl,
                                           stepra=rp.get("stepra"))[3]
                      for rp in rebuilt)
        est = waves * per_job
        if left is None or est <= left:
            if k < len(top):
                print("stage 2: dropped %s to fit the budget (expected %s, left %s)"
                      % (", ".join(c["id"] for c in top[k:]), _fmt_hms(est), _fmt_hms(left)))
            return top[:k], est, None
    return [], est, ("even %s needs an expected %s of running time; only %s of the budget is "
                     "left (elapsed %s / budget %s)"
                     % ("the best candidate" if top else "ANCHOR alone", _fmt_hms(est or 0),
                        _fmt_hms(max(0.0, left)), _fmt_hms(ctx.elapsed()),
                        _fmt_hms(ctx.budget_s if left_s is _BUDGET_LEFT else left)))


def pick_precision_points(wrapped: list[dict], k: int, sample_points: Any = None) -> list[dict]:
    """k prediction points: 1 -> the costliest alone (t0 is free phase-1 data), 2 -> t0 + the
    costliest, 3+ -> those plus the next costliest."""
    if k <= 0:
        return []
    if k == 1 and len(wrapped) > 1:
        return [wrapped[_costliest_point_index(wrapped, sample_points)]]
    return pick_points(wrapped, k, sample_points)


def _costliest_point_index(rebuilt: list[dict], sample_points: Any) -> int:
    """Index of the point expected to cost most per step (the most sunlit, choose_primary_point);
    the last point when that fails."""
    try:
        return choose_primary_point([rp["point"] for rp in rebuilt], sample_points)[0]
    except Exception:                                       # noqa: BLE001
        return len(rebuilt) - 1


def pick_points(rebuilt: list[dict], k: int, sample_points: Any = None,
                s_normal: list | None = None) -> list[dict]:
    """k points in rebuilt order: always the earliest (t0) plus the costliest -- by measured
    s_normal (parallel to `rebuilt`) when given, else by the sunlit-fraction proxy."""
    if len(rebuilt) <= k:
        return list(rebuilt)
    idx = list(range(len(rebuilt)))
    earliest = min(idx, key=lambda i: rebuilt[i]["point"].at)
    if s_normal:
        rest = sorted((i for i in idx if i != earliest),
                      key=lambda i: -(s_normal[i] if i < len(s_normal) else 0.0))
    else:
        c = _costliest_point_index(rebuilt, sample_points)
        rest = [c] if c != earliest else []
        rest += [i for i in reversed(idx) if i != earliest and i not in rest]
    return [rebuilt[i] for i in sorted([earliest] + rest[:k - 1])]
