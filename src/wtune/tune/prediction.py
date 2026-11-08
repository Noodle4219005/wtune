"""Whole-run wall-time prediction: per-candidate predictions for the ranking, the winner's
full-run prediction (single probe, or combine() over measured points), the C3 point
measurements, and the pipeline's prediction points, error bands and run-length I/O."""
from __future__ import annotations

import concurrent.futures
import threading
import csv
import dataclasses
import datetime
import json
import math
import os
from typing import Any

import wtune.machine
from ._deps import coldstart, sweep, cost_drivers, io_model, predict, probe_plan
from .common import UNSET, Record, _fmt_hms, _fmt_sbatch_time, is_set
from .context import Ctx
from .points import pick_precision_points
from . import budget as _budget
from . import jobs as _jobs

_PROBE_LEDGER_FIELDS = ["case_base", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles",
                        "s_per_step_median"]
COLD_START_BAND = wtune.machine.MEASURED["cold_start_band"]   # built-in; +-25%: `wtune predict`
                                     # held-out median 5.8%, padded for
                                     # weather it cannot see
FALLBACK_BANDS = (                   # (rebuilt points, method, band) when wtune.cost_drivers is
    (0, "t0 anchor (phase-1 data)", 0.15),     # not importable: t0-only underestimates ~9.6%
    (1, "t0 + costliest point", 0.10),         # (conus25); missing the costliest point -13.6%
    (3, "3+ points", 0.07),                    # (elevconv); 3+ points via combine() <= ~7%
)
TIME_STEP_S = 15 * 60.0              # s: --time suggestions are rounded UP to this granularity


def _run_window(nl: dict) -> tuple:
    """(run_start, run_end, dt, stepra) of a parsed namelist."""
    _steps_n, run_s, dt, _note = predict.get_d01_steps(nl)
    run_start = predict.get_time_control_datetime(nl, "start")
    run_end = run_start + datetime.timedelta(seconds=run_s)
    radt = predict.g(nl, "physics", "radt", None)
    stepra = predict.steps_per_radiation_call(float(radt), dt)
    return run_start, run_end, dt, stepra


def _drift_calibration(ctx: Ctx) -> dict:
    """The default calibration, with --config prediction.drift_band_pct applied."""
    calibration = predict.load_calibration(predict.calibration_path())
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration = dict(calibration, drift_band_pct=ctx.prediction_cfg["drift_band_pct"])
    return calibration


def write_probe_ledger(ctx: Ctx, all_results: dict, confirm_results: dict | None) -> str | None:
    """Write this run's measurements (confirm wins) as <out>/predict_probe.csv, one row per
    replicate in the columns wtune.predict.resolve_probe reads; configs missing a decomposition
    field are skipped. Returns the path, or None when nothing was measured."""
    merged = dict(all_results)
    merged.update(confirm_results or {})
    entries = [r for r in merged.values() if r.get("config") and r.get("s_values")]
    if not entries:
        return None
    os.makedirs(ctx.out_dir, exist_ok=True)
    path = os.path.join(ctx.out_dir, "predict_probe.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROBE_LEDGER_FIELDS)
        w.writeheader()
        for r in entries:
            cfg = r["config"]
            if any(k not in cfg for k in ("ranks", "omp", "nx", "ny", "numtiles")):
                continue
            for s in r["s_values"]:
                w.writerow(dict(case_base=ctx.case, n_ranks=cfg["ranks"], omp_threads=cfg["omp"],
                                 nproc_x=cfg["nx"], nproc_y=cfg["ny"], numtiles=cfg["numtiles"],
                                 s_per_step_median=s))
    return path


def _rsl_paths_for_entry(ctx: Ctx, entry: dict | None) -> list[str]:
    """The entry's kept rsl.error.0000 logs: RUNDIR = ledger_dir/<run_id> (strict_campaign keeps
    rsl.error.0000 in every MODE); [] without run_ids or logs."""
    if not entry:
        return []
    return [p for rid in (entry.get("run_ids") or [])
            for p in [os.path.join(ctx.ledger_dir, rid, "rsl.error.0000")]
            if os.path.exists(p)]


def predict_candidates(ctx: Ctx, nodes: int, results: dict) -> dict:
    """{cid: results[cid] + pred_total_s, pred_low_s, pred_high_s, pred_source, pred_error}: one
    wtune.predict.build_prediction (compute + init, include_io=False by the 2026-09-24 decision)
    per candidate, sharing one probe CSV that matches each config's own rows. A failure sets
    pred_total_s None and pred_error instead of raising."""
    out = {}
    if predict is None:
        for cid, entry in results.items():
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="wtune.predict not importable")
        return out

    probe_path = write_probe_ledger(ctx, results, None)
    calibration_path = predict.calibration_path()
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration_path, _note = _calibration_with_drift_override(
            ctx, calibration_path, ctx.prediction_cfg["drift_band_pct"])

    for cid, entry in results.items():
        cfg = entry.get("config")
        if cfg is None:
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="no config")
            continue
        rsl_paths = _rsl_paths_for_entry(ctx, entry)
        try:
            pred = predict.build_prediction(
                namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
                config=dict(ranks=cfg["ranks"], omp=cfg["omp"], nx=cfg["nx"], ny=cfg["ny"],
                            numtiles=cfg["numtiles"]),
                ledger_paths=[probe_path] if probe_path else None,
                calibration_path=calibration_path, io_form_override=None,
                io_target=ctx.prediction_cfg.get("io_target") or "nfs", case_name=ctx.case,
                rsl_paths=rsl_paths or None, include_io=False)
        except Exception as e:
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="%s: %s" % (type(e).__name__, e))
            continue
        out[cid] = dict(entry, pred_total_s=pred["total_s"], pred_low_s=pred["range_low_s"],
                         pred_high_s=pred["range_high_s"],
                         pred_source=pred["compute"]["r_rad_source"], pred_error=None)
    return out


def _calibration_with_drift_override(ctx: Ctx, calibration_path: str,
                                     drift_pct: float) -> tuple[str, str]:
    """(calibration path, note): build_prediction reads drift_band_pct only from its calibration
    file, so an override is a copy of that file in out_dir with the field replaced."""
    try:
        with open(calibration_path) as f:
            calib = json.load(f)
    except (OSError, ValueError) as e:
        return calibration_path, ("NOTE: prediction.drift_band_pct=%s requested but the default "
                                   "calibration file %s could not be read (%s); using it unmodified"
                                   % (drift_pct, calibration_path, e))
    calib["drift_band_pct"] = drift_pct
    os.makedirs(ctx.out_dir, exist_ok=True)
    out_path = os.path.join(ctx.out_dir, "calibration_override.json")
    with open(out_path, "w") as f:
        json.dump(calib, f)
    return out_path, ("NOTE: prediction.drift_band_pct=%s applied via calibration override %s"
                       % (drift_pct, out_path))


def _cost_series_for_ctx(ctx: Ctx, run_start: Any, run_end: Any, case_dir: str,
                         grib_dir: str | None = None) -> Any:
    """wtune.cost_drivers.CostSeries when probe.points_by == "cost-indicators", else None (also
    on any failure: the points are then placed by time). Never raises."""
    if (ctx.probe_cfg or {}).get("points_by") != "cost-indicators":
        return None
    try:
        import wtune.cost_drivers
        return cost_drivers.cost_series_for_case(
            case_dir, run_start, run_end, ctx.probe_cfg.get("indicator_sources"), grib_dir,
            namelist_path=ctx.namelist_path)
    except Exception:
        return None


def _probe_plan_preview(ctx: Ctx, nodes: int, best_config: dict) -> tuple | None:
    """(points, notes, run_start, run_end, dt, stepra) wtune.probe_plan.plan_points would probe
    for the winner -- an informational preview; None on any failure."""
    if predict is None or probe_plan is None:
        return None
    try:
        nl = predict.parse_namelist(ctx.namelist_path)
        steps_n, run_s, dt, _note = predict.get_d01_steps(nl)
        run_start = predict.get_time_control_datetime(nl, "start")
        if run_start is None:
            return None
        run_end = run_start + datetime.timedelta(seconds=run_s)
        radt = predict.g(nl, "physics", "radt", None)
        if radt is None:
            return None
        stepra = predict.steps_per_radiation_call(float(radt), dt)
        spinup_s = float(ctx.probe_cfg.get("spinup_min") or 0) * 60.0
        points, notes = probe_plan.plan_points(
            nl, run_start, run_end, dt, stepra, spinup_s, ctx.probe_cfg,
            case_dir=ctx.case_json.get("prep_dir"),
            indicators=_cost_series_for_ctx(ctx, run_start, run_end, ctx.case_json.get("prep_dir")))
        return points, notes, run_start, run_end, dt, stepra
    except Exception:
        return None


REBUILD_WPS_PER_GRIB = 4.7          # haitang2017: wps/ (ungrib + met_em) 2.4 GB per 514 MB of GRIB
REBUILD_WINDOW_BYTES = 0.25e9       # win_*/ (real.exe output, WRF run dir): 158 MB measured


def _check_rebuild_space(out_dir: str, grib_bytes: list) -> None:
    """Refuse to submit when the points' WPS and WRF files will not fit where they are written
    (GPFS quota or free space, wtune.run.space): past a soft quota ungrib dies on "Disk quota
    exceeded" and the point is silently lost (haitang2017 tune, 2026-10-27, 3 of 5 points)."""
    from wtune.run import space
    need = sum(REBUILD_WPS_PER_GRIB * b + REBUILD_WINDOW_BYTES for b in grib_bytes)
    where = os.path.abspath(out_dir)
    while not os.path.isdir(where) and os.path.dirname(where) != where:
        where = os.path.dirname(where)
    free, src = space.writable_bytes(where)
    if need > free:
        raise RuntimeError("rebuilt points need ~%.1f GB under %s but only %.1f GB is writable "
                           "(%s) -- free space or move --out" % (need / 1e9, out_dir, free / 1e9,
                                                                  src))


def _clean_rebuild_points(plans: list) -> None:
    """Drop each scored point's grib/ and wps/ (3 GB per point on haitang2017; reproducible),
    keeping plan.json, the logs and win_*/ (rsl files, the I/O probe). WTUNE_KEEP_REBUILD=1 keeps
    everything."""
    import shutil
    if os.environ.get("WTUNE_KEEP_REBUILD") == "1":
        return
    for plan in plans:
        pdir = plan.get("_plan_dir")
        for sub_dir in ("grib", "wps"):
            if pdir:
                shutil.rmtree(os.path.join(pdir, sub_dir), ignore_errors=True)


def rebuild_wave(nodes: int, max_nodes: int | None = None) -> int:
    """How many rebuild points run at once: as many `nodes`-node jobs as fit in max_nodes
    (default the machine's), at least one."""
    cap = max_nodes or wtune.machine.max_nodes()
    return max(1, int(cap) // max(1, int(nodes)))


def run_rebuild_points(ctx: Ctx, points: list, best_config: dict, nodes: int, case_dir: str,
                       out_dir: str, run_start: Any, spinup_min: float = 60,
                       grib_dir: str | None = None, date_shift_to: Any = None,
                       forcing: str | None = None,
                       measure_steps: int | None = None) -> tuple | list:
    """Measure the rebuild points: one coldstart plan dir + job per point, fetched/staged/
    submitted concurrently (decision 0054), scored by score_windows_from_rsl. `spinup_min` must
    equal plan_points'; each `.at` is the measured window's midpoint, not its start T. Returns
    ([PointMeasurement], "job,job"); RuntimeError on any infrastructure failure."""
    if not points:
        return []
    missing_env = [v for v in ("WPS_DIR", "WRF_TREE") if not os.environ.get(v)]
    if missing_env:          # job 1236: staged without these, the windows died in 3 s
        raise RuntimeError("run_rebuild_points: %s not set -- source the site env (%s) before "
                           "running the driver"
                           % (", ".join(missing_env), wtune.machine.env_source()))
    config = dict(ranks=best_config["ranks"], omp=best_config["omp"], nx=best_config["nx"],
                  ny=best_config["ny"], numtiles=best_config.get("numtiles", 1), nodes=nodes)
    # one plan dir per point: windows sharing $PLAN/wps would overwrite each other's met_em files
    seg_timeout_min = _budget.rebuild_segment_timeout_min(case_dir, spinup_min, config["ranks"],
                                                          config.get("omp", 1))

    owned_lock = threading.Lock()

    def _prepare(i, p):
        pdir = os.path.join(out_dir, "pt_%d" % i)
        plan = probe_plan.plan_rebuild_segments(
            case_dir, [p], config, pdir, spinup_min=spinup_min, date_shift_to=date_shift_to,
            grib_dir=grib_dir, forcing=forcing, measure_steps=measure_steps)
        # the point's own PLAN/grib: a shared cache left GRIB links dangling (zhengzhou 2026-09-27)
        total_bytes, _forcing = coldstart.resolve_and_fetch_forcing(
            plan, pdir, os.path.join(pdir, "grib"), grib_dir=grib_dir)
        print("C3 rebuild point %d (%s): fetched %.1f MB forcing"
              % (i, p.at.isoformat(), total_bytes / 1e6), flush=True)
        sbatch_path = coldstart.stage_plan(pdir, per_window_min=seg_timeout_min)
        plan["_plan_dir"] = pdir
        return plan, sbatch_path, total_bytes

    # ERA5 fetches are network-bound (~3.5 min per point): fetch and stage concurrently; the
    # submissions go in waves that keep the points' nodes within the machine's max_nodes (a
    # point i waits, afterany, on point i - wave), so 5 points x 3 nodes on 8 nodes run 2 at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(points))) as pool:
        done = list(pool.map(lambda ip: _prepare(*ip), enumerate(points)))
    plans = [d[0] for d in done]
    _check_rebuild_space(out_dir, [d[2] for d in done])
    wave = rebuild_wave(nodes, getattr(ctx, "max_nodes", None))
    job_ids = []
    for i, (_plan, sbatch_path, _b) in enumerate(done):
        dep = (" --dependency=afterany:%s" % job_ids[i - wave]) if i >= wave else ""
        _jobs.stop_before_submit()
        job_id = sweep.submit_job("sbatch%s %s" % (dep, sbatch_path),
                                  sbatch_runner=ctx.sbatch_runner)
        with owned_lock:       # owned at once: another point failing must not orphan this job
            _jobs.record_job(ctx, job_id)
        job_ids.append(job_id)
        print("C3 rebuild job submitted: %s (%s)%s" % (job_id, sbatch_path, dep), flush=True)
    states = _jobs.wait_for_jobs(ctx, job_ids, "c3-rebuild")
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("C3 rebuild job(s) %s not COMPLETED -- infrastructure failure, stop "
                           "and diagnose read-only" % bad)
    sample_points, _geom_src = predict.read_wrfinput_sample_points(case_dir)
    calibration = predict.load_calibration(predict.calibration_path())
    measurements, failed = [], []
    for i, plan in enumerate(plans):
        try:
            windows_out, _pooled, anchors = coldstart.score_windows_from_rsl(
                plan, sample_points, calibration)
        except Exception as e:                              # noqa: BLE001 -- one point, not all
            failed.append("point %d (%s): %s: %s" % (i, points[i].at, type(e).__name__, e))
            continue
        for w, (anchor_s, _s_k) in zip(windows_out, anchors):
            at = run_start + datetime.timedelta(seconds=anchor_s)
            measurements.append(probe_plan.PointMeasurement(
                at=at, reason=w["reason"], source="rebuild", s_normal=w["s_k"],
                n_rad=w["n_rad"], r=w["r_local"]))
    _clean_rebuild_points(plans)
    if failed:
        print("WARNING: %d of %d rebuilt point(s) gave no measurement -- predicting from the "
              "rest: %s" % (len(failed), len(plans), "; ".join(failed)), flush=True)
    if failed and not measurements:
        raise RuntimeError("no rebuilt point gave a measurement: %s" % "; ".join(failed))
    return measurements, ",".join(job_ids)


def build_full_run_prediction(ctx: Ctx, best_config: dict | None, nodes: int, all_results: dict,
                              confirm_results: dict | None,
                              probe_measurements: list | None = None) -> str:
    """The winner's full-run prediction text: combine() over `probe_measurements` when given,
    else wtune.predict.build_prediction fed this run's probe ledger (+ notes). Without
    wtune.predict or a winner: the `wtune/cli.py predict` command to run instead. Never raises."""
    if predict is None or best_config is None:
        return ("full-run prediction: run `wtune predict %s --nodes %d --config "
                "'%s'`" % (ctx.case_json["prep_dir"], nodes,
                           json.dumps(dict(ranks=best_config["ranks"], omp=best_config["omp"],
                                            nx=best_config["nx"], ny=best_config["ny"],
                                            numtiles=best_config["numtiles"]))
                           if best_config else "..."))
    if probe_measurements and probe_plan is not None:
        try:
            return _multipoint_prediction_text(ctx, best_config, nodes, probe_measurements)
        except Exception as e:
            return ("full-run prediction (multi-point) FAILED (%s: %s); falling back would "
                     "need a re-run -- run `wtune predict ...` manually"
                     % (type(e).__name__, e))
    try:
        return _single_probe_prediction_text(ctx, best_config, nodes, all_results,
                                             confirm_results)
    except Exception as e:
        return ("full-run prediction FAILED (%s: %s); run `wtune predict ...` manually"
                 % (type(e).__name__, e))


def _multipoint_prediction_text(ctx: Ctx, best_config: dict, nodes: int,
                                probe_measurements: list) -> str:
    """combine() of the winner's measured points, as report lines."""
    nl = predict.parse_namelist(ctx.namelist_path)
    run_start, run_end, dt, stepra = _run_window(nl)
    calibration = _drift_calibration(ctx)
    sample_points, _src = predict.read_wrfinput_sample_points(
        ctx.case_json["prep_dir"])
    result = probe_plan.combine(
        probe_measurements, run_start, run_end, dt, stepra, calibration=calibration,
        sample_points=sample_points, case_dir=ctx.case_json["prep_dir"], nl=nl,
        ranks=best_config.get("ranks"), nodes=nodes,
        cost_series=_cost_series_for_ctx(ctx, run_start, run_end,
                                         ctx.case_json["prep_dir"]))
    lines = ["=== full-run prediction (MULTI-POINT, wtune.probe_plan.combine, "
             "%d measured point(s)) ===" % len(probe_measurements),
             "compute_total: %s   init: %.1fs   TOTAL: %s (range %s - %s, "
             "+-%.1f%% drift)"
             % (_fmt_hms(result["compute_total_s"]), result["init_s"],
                _fmt_hms(result["total_s"]), _fmt_hms(result["range_low_s"]),
                _fmt_hms(result["range_high_s"]), result["drift_pct"]),
             "r_rad: night=%.3f day=%.3f source=%s"
             % (result["r_night"], result["r_day"], result["r_source"])]
    if result.get("cost_indicators") and "error" not in result["cost_indicators"]:
        ci = result["cost_indicators"]
        lines.append("cost-indicator totals (compute+init): interpolated %s, stratified "
                     "%s, regression %s (R2 %s), spread %s%%; sources %s"
                     % (_fmt_hms(result["total_s"]),
                        _fmt_hms(result["stratified_total_s"])
                        if result.get("stratified_total_s") else "-",
                        _fmt_hms(result["regression_total_s"])
                        if result.get("regression_total_s") else "-",
                        ("%.3f" % result["regression_r2"])
                        if result.get("regression_r2") is not None else "-",
                        ("%.1f" % ci["spread_pct"]) if ci.get("spread_pct") is not None
                        else "-", ci.get("sources")))
    for p in result["points"]:
        lines.append("  point %s reason=%s source=%s s_normal=%.4f n_rad=%d r=%s"
                      % (p["at"].isoformat(), p["reason"], p["source"], p["s_normal"],
                         p["n_rad"], ("%.3f" % p["r"]) if p["r"] is not None else "-"))
    if result.get("init_note"):
        lines.append("NOTE: " + result["init_note"])
    return "\n".join(lines)


def _single_probe_prediction_text(ctx: Ctx, best_config: dict, nodes: int, all_results: dict,
                                  confirm_results: dict | None) -> str:
    """wtune.predict.build_prediction of the winner with this run's probe ledger and rsl logs
    (ledger_paths=None used to force no_probe_fallback, +-50%), rendered, plus the drift,
    measure_r_rad and multi-point preview notes."""
    probe_path = write_probe_ledger(ctx, all_results, confirm_results)
    calibration_path = predict.calibration_path()
    drift_note = None
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration_path, drift_note = _calibration_with_drift_override(
            ctx, calibration_path, ctx.prediction_cfg["drift_band_pct"])
    entry = (confirm_results or {}).get(best_config.get("id")) or \
        (all_results or {}).get(best_config.get("id"))
    rsl_paths = _rsl_paths_for_entry(ctx, entry)
    pred = predict.build_prediction(
        namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
        config=dict(ranks=best_config["ranks"], omp=best_config["omp"], nx=best_config["nx"],
                    ny=best_config["ny"], numtiles=best_config["numtiles"]),
        ledger_paths=[probe_path] if probe_path else None,
        calibration_path=calibration_path, io_form_override=None,
        io_target=ctx.prediction_cfg.get("io_target") or "nfs", case_name=ctx.case,
        rsl_paths=rsl_paths or None, include_io=False)
    text = predict.render_text(pred)
    if entry and entry.get("fidelity"):
        text += ("\nNOTE: the s/step above is the joint proxy's (%s), not a real N-node run; on f1 "
                 "x86 band scores read -1 to +6%% vs real (conus25, isc-synth, Big Chill)"
                 % entry["fidelity"])
    if drift_note:
        text += "\n" + drift_note
    if ctx.prediction_cfg.get("measure_r_rad"):
        # say whether r_rad came from this run's own probe logs (wtune.config schema honesty rule)
        r_rad_source = pred["compute"]["r_rad_source"]
        if r_rad_source in ("fit_day_night", "ratio_from_night", "ratio_from_day"):
            text += ("\nNOTE: prediction.measure_r_rad requested -- derived from this run's "
                      "own probe rsl.error.0000 logs (source=%s)." % r_rad_source)
        else:
            text += ("\nNOTE: prediction.measure_r_rad requested (derive the radiation-step "
                      "cost multiplier r_rad from this run's own probe measurements) -- no "
                      "usable rsl.error.0000 probe log for this candidate's own runs "
                      "(source=%s); deriving r_rad from measurements is NOT YET IMPLEMENTED "
                      "without one -- used the fixed calibration default instead."
                      % r_rad_source)
    preview = _probe_plan_preview(ctx, nodes, best_config)
    if preview:
        preview_points, preview_notes, _rs, _re, _dt, _stepra = preview
        text += ("\nNOTE: multi-point probe planning (wtune.probe_plan.plan_points) would "
                 "place %d point(s) for this winner (%s) -- none measured for this run, so "
                 "the single t0-probe prediction above was used; see this task's report for "
                 "the Layer A/B wiring status."
                 % (len(preview_points),
                    ", ".join("%s@%s" % (p.reason, p.at.strftime("%m-%d_%H:%M"))
                              for p in preview_points)))
        for n in preview_notes:
            text += "\nNOTE: " + n
    return text


def _prediction_context(ctx: Ctx) -> tuple:
    """(nl, run_start, run_end, dt, stepra, calibration, sample_points, case_dir) for
    combine()-based prediction."""
    nl = predict.parse_namelist(ctx.namelist_path)
    run_start, run_end, dt, stepra = _run_window(nl)
    calibration = _drift_calibration(ctx)
    case_dir = sweep._expand(ctx.case_json["prep_dir"])
    sample_points, _src = predict.read_wrfinput_sample_points(case_dir)
    return nl, run_start, run_end, dt, stepra, calibration, sample_points, case_dir


@dataclasses.dataclass
class WholeRunPrediction(Record):
    """The pipeline's whole-run prediction (tune_report.json pipeline.prediction). Every field
    is optional: UNSET fields are left out of to_dict(), so each path keeps its own key set."""
    layout_id: Any = UNSET
    nodes: Any = UNSET
    cores_per_node: Any = UNSET
    config: Any = UNSET                    # {ranks, omp, nx, ny, numtiles[, dist]}
    predicted_total_s: Any = UNSET         # s: compute + init + run-length I/O
    predicted_total_hms: Any = UNSET
    predicted_range_s: Any = UNSET         # [low, high] s
    time_suggestion: Any = UNSET           # sbatch --time HH:MM:00
    points_used: Any = UNSET               # rebuilt points behind the prediction
    degraded: Any = UNSET                  # phase 2 ran a smaller plan than the full one
    band: Any = UNSET                      # relative error band (0.10 = +-10%)
    error_min: Any = UNSET                 # band x total, minutes
    source: Any = UNSET
    candidates: Any = UNSET                # [{nodes, cores, pid, predicted_total_s, config}]
    compute_s: Any = UNSET
    init_s: Any = UNSET
    io: Any = UNSET                        # wtune.io_model.run_length_io terms, or None
    init_source: Any = UNSET
    io_s: Any = UNSET
    breakdown: Any = UNSET                 # 'compute X + history Y + ... = TOTAL' line
    segment_note: Any = UNSET

    def add_run_length_io(self, io: dict | None, compute_s: float, init_s: float, band: float,
                          init_source: str | None = None) -> WholeRunPrediction:
        """Total = compute_s + init_s + run-length I/O; the range and every candidate shift by the
        same amount (I/O is layout-independent: the order is kept), time_suggestion / error_min
        are recomputed at `band`, and the breakdown line and segment note are recorded."""
        io_s = io["total_s"] if io else 0.0
        T = compute_s + init_s + io_s
        shift = T - ((self.predicted_total_s if is_set(self.predicted_total_s) else None) or 0.0)
        self.compute_s, self.init_s, self.io = compute_s, init_s, io
        self.init_source, self.io_s = init_source, io_s
        self.predicted_total_s, self.predicted_total_hms = T, _fmt_hms(T)
        if is_set(self.predicted_range_s) and self.predicted_range_s:
            self.predicted_range_s = [x + shift for x in self.predicted_range_s]
        for c in (self.candidates if is_set(self.candidates) else None) or []:
            if c.get("predicted_total_s") is not None:
                c["predicted_total_s"] += shift
        self.time_suggestion = _fmt_sbatch_time(time_suggestion_s(T, band))
        self.error_min = band * T / 60.0
        if io:
            self.breakdown = io_model.breakdown_line(compute_s, io, init_s)
            self.segment_note = io_model.segment_note(init_s)
        else:
            self.breakdown = ("compute %.0fs + init %.0fs = %.0fs (run-length I/O NOT included)"
                              % (compute_s, init_s, T))
            self.segment_note = None
        return self


def add_run_length_io(pred: dict | WholeRunPrediction, io: dict | None, compute_s: float,
                      init_s: float, band: float,
                      init_source: str | None = None) -> dict | WholeRunPrediction:
    """WholeRunPrediction.add_run_length_io, also for a plain prediction dict (updated in place;
    keys that are not prediction fields are left alone)."""
    if not isinstance(pred, dict):
        return pred.add_run_length_io(io, compute_s, init_s, band, init_source)
    names = {f.name for f in dataclasses.fields(WholeRunPrediction)}
    p = WholeRunPrediction(**{k: v for k, v in pred.items() if k in names})
    pred.update(p.add_run_length_io(io, compute_s, init_s, band, init_source).to_dict())
    return pred


def _band_for_points(points: int, measured: bool) -> tuple[str, float]:
    if points <= 0:
        return (("t0 anchor (phase-1 data)", 0.15) if measured
                else ("cold-start predict", wtune.machine.measured("cold_start_band")))
    return (FALLBACK_BANDS[1][1], FALLBACK_BANDS[1][2]) if points < 3 else (
        FALLBACK_BANDS[2][1], FALLBACK_BANDS[2][2])


def _fallback_point_plan(T_s: float, precision_s: float, measured: bool,
                         n_available: int) -> tuple[int, str, float]:
    """(points, method, band): the fewest points whose fallback band x T meets `precision_s`."""
    options = [(0,) + _band_for_points(0, measured), (1,) + _band_for_points(1, measured),
               (3,) + _band_for_points(3, measured)]
    for pts, method, band in options:
        if pts <= n_available and band * T_s <= precision_s:
            return pts, method, band
    pts = n_available
    method, band = _band_for_points(pts, measured)
    return pts, method, band


def _case_types_and_track(ctx: Ctx) -> tuple[list, Any]:
    """(case type labels, TC best track or None): wtune.advise.build_case ->
    wtune.cost_drivers.case_types_for (decision-0057 classifier)."""
    import wtune.advise
    case = wtune.advise.build_case(ctx.case)
    types = list(cost_drivers.case_types_for(case))
    track = wtune.advise.best_track(case) if "tc" in types else None
    return types, track


def _nearest_wrapped(wrapped: list[dict], times: list) -> list[dict]:
    """The planned point nearest to each of `times`, each used once, in index order."""
    out, free = [], list(wrapped)
    for t in times:
        if not free:
            break
        best = min(free, key=lambda w: abs((w["point"].at - t).total_seconds()))
        free.remove(best)
        out.append(best)
    return sorted(out, key=lambda w: w["index"])


PRECISION_FLOOR_MIN = 2.0     # a relative precision never asks for less than this many minutes


def precision_minutes(precision: Any, T_est_s: float) -> float:
    """Minutes the whole-run prediction must hit: `precision` is minutes (a number) or "N%" of
    the run's estimated wall time T_est_s, at least PRECISION_FLOOR_MIN. A fixed +-15 min let a
    30 min run (haitang2017, 2026-10-27) through with no prediction point: off by 38 %."""
    text = str(precision).strip()
    if text.endswith("%"):
        return max(PRECISION_FLOOR_MIN, float(text[:-1]) / 100.0 * float(T_est_s or 0.0) / 60.0)
    return float(text)


def precision_label(precision: Any) -> str:
    """'+-10%' or '+-15 min' for messages."""
    text = str(precision).strip()
    return "±%s" % text if text.endswith("%") else "±%g min" % float(text)


def plan_prediction_points(ctx: Ctx, wrapped: list[dict], precision_min: Any, T_est_s: float,
                           measured: bool, sample_points: Any = None, run_start: Any = None,
                           run_end: Any = None, window_s: float = 0.0) -> dict:
    """Prediction points + whole-run band for `precision_min`: wtune.cost_drivers.
    plan_points_for_case (case types, cost indicators, TC track) with each needs_rebuild time
    mapped onto the nearest planned point; on failure (printed) the fallback band table.
    Returns {selected, band, method, source, ...}. precision_min: minutes or "N%" of T_est_s."""
    precision_min = precision_minutes(precision_min, T_est_s)
    api = cost_drivers
    if api is not None and getattr(api, "plan_points_for_case", None) and run_start is not None:
        try:
            types, track = _case_types_and_track(ctx)
            indicators = _cost_series_for_ctx(ctx, run_start, run_end,
                                              sweep._expand(ctx.case_json["prep_dir"]))
            res = api.plan_points_for_case(
                types, run_start, run_end, precision_min, T_est_s, indicators=indicators,
                best_track=track, window_s=window_s)
            sel = _nearest_wrapped(wrapped, res.get("needs_rebuild") or [])
            why = (res.get("rationale") or [""])
            return dict(selected=sel, band=float(res["band_pct"]) / 100.0,
                        method="%s: %s" % ("/".join(res.get("types") or types), why[0]),
                        source="wtune.cost_drivers", types=list(res.get("types") or types),
                        meets_precision=res.get("meets_precision"),
                        confidence=res.get("confidence"), rationale=list(why),
                        points=[(t.isoformat(), r) for t, r in res.get("points", [])])
        except Exception as e:                              # noqa: BLE001
            print("pipeline: wtune.cost_drivers.plan_points_for_case FAILED (%s: %s) -- using the "
                  "fallback band table instead" % (type(e).__name__, e))
    pts, method, band = _fallback_point_plan(T_est_s or 0.0, precision_min * 60.0, measured,
                                             len(wrapped))
    return dict(selected=pick_precision_points(wrapped, pts, sample_points), band=band,
                method=method, source="fallback band table")


def time_suggestion_s(T_s: float, band: float, step_s: float = TIME_STEP_S) -> float:
    """sbatch --time for the real run: T rounded UP to `step_s` plus band x T, at least 1.2 x T."""
    return max(math.ceil(T_s / step_s) * step_s + band * T_s, 1.2 * T_s)


def pipeline_run_length_io(namelist_path: str | None, io: str | None = None) -> dict | None:
    """wtune.io_model.run_length_io of the case namelist (history, boundary, auxinput4,
    restart I/O and iofields re-parsing of one unsegmented run); `io` adios2 (default) |
    netcdf; None, with the reason printed, when unavailable."""
    if io_model is None or predict is None or not namelist_path:
        return None
    try:
        nl = predict.parse_namelist(namelist_path)
        cal = predict.load_calibration(predict.calibration_path())
        return io_model.run_length_io(
            nl, io=io or io_model.DEFAULT_IO_FORMAT,
            iofields_calls=io_model.case_iofields_calls(
                nl, os.path.dirname(os.path.abspath(namelist_path))),
            call_s=io_model.iofields_call_s(cal))
    except Exception as e:                                  # noqa: BLE001
        print("pipeline: run-length I/O model unavailable (%s: %s) -- whole-run prediction is "
              "compute + init only" % (type(e).__name__, e))
        return None


def prediction_init(ctx: Ctx, fallback_s: float, fallback_source: str) -> tuple[float, str]:
    """(init_s, source): round0's startup measured on this allocation when there is one (scc:
    predict_init charged 125.6 s vs ~50 s measured on climate-ea50), else the fallback."""
    if str(getattr(ctx, "startup_source", "") or "").startswith("measured"):
        n = getattr(ctx, "startup_runs", None)
        return float(ctx.startup_s), ("measured round0 startup (%d runs)" % n if n
                                      else "measured round0 startup (%s)" % ctx.startup_source)
    return float(fallback_s), fallback_source


def cold_start_prediction(ctx: Ctx, nodes: int, config: dict | None,
                          total_steps: int) -> tuple[float, float, str]:
    """(whole-run s, s/step at `nodes`, source): wtune.predict's no-probe prediction, else the
    ledger estimate (10 s/step fallback). Crude by design: the O0 band is +-25%."""
    if predict is not None and config:
        try:
            pred = predict.build_prediction(
                namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
                config=dict(ranks=config["ranks"], omp=config["omp"], nx=config["nx"],
                            ny=config["ny"], numtiles=config["numtiles"]),
                ledger_paths=None, calibration_path=predict.calibration_path(),
                io_form_override=None, io_target=ctx.prediction_cfg.get("io_target") or "nfs",
                case_name=ctx.case, rsl_paths=None, include_io=False)
            T = float(pred["total_s"])
            comp = float(pred["compute"]["compute_total_s"])   # T's init is predict's, not startup_s
            return T, max(1e-6, comp / max(1, total_steps)), "wtune.predict cold start"
        except Exception:                                   # noqa: BLE001
            pass
    s_est, src, _k = ctx.estimate_s_per_step(nodes)
    return total_steps * s_est + ctx.startup_s, s_est, "ledger estimate (%s)" % src
