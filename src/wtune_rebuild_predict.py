"""Whole-run wall-time prediction from rebuilt segments (decision 0054's final architecture).

Every sampled point is rebuilt from analysis/reanalysis forcing (WPS + real.exe), spun up and
measured on the TARGET hardware config, all points as parallel jobs; the measurements are then
combined into one prediction (wtune_probe_plan.combine). Validated on scc (2026-09-27):
conus25 C3 -0.98 %, zhengzhou -5.78 % / -5.67 % (phase 7), ERA5 spin-up 20 min (phase 3).

    wtune predict-rebuild CASE_DIR --nodes 4 --nx 16 --ny 18 [--numtiles 1] [--omp 1]
        [--forcing era5|auto] [--spinup-min 20] [--max-points N] [--measure-steps N|short] [--no-io-probe] [--rain-snow]
        [--no-io-probe] [--out DIR] [--dry-run]

CASE_DIR needs namelist.input and wrfinput_d01 (the geometry and, for --forcing era5, the ERA5
crop box come from it); nothing is written inside it. Must run with the site env loaded (WPS_DIR, WRF_TREE). The prediction is
written to OUT/pred_rebuild.json before anything else is compared with it.

I/O is predicted separately from compute+init and MEASURED (--io-probe, default on): the last
point's segment also writes one history frame per domain (+ one restart when the case writes
restarts) at its end, with the case's own output settings; each timed write is multiplied by the
number of frames the case's own run writes. The recommended Slurm --time is range_high +
2 x I/O + a margin of 10% of range_high clamped to 2-5 min, rounded up to 1 min below 2 h and to
5 min above (history writes are 21-48 % of wall on scc and the same volume varies up to 2.4x
there, hence the factor 2).
"""
import argparse
import copy
import dataclasses
import datetime as dt
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "tools"), os.path.join(_HERE, "predictor"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ERA5_PIN = "era5_arco"
IO_FACTOR = 2.0          # recommended --time covers 2x the measured I/O (scc NFS: 2.4x spread)
# Margin and rounding scale with the run (T1 on scc, 2026-09-27: a fixed 5 min margin + 5 min
# rounding gave dorian2019's 1718 s run a 45:00 --time, 1.57x; maria1km 1.29x, zhengzhou 1.09x).
TIME_MARGIN_FRAC = 0.10  # of range_high
TIME_MARGIN_MIN_S = 120.0
TIME_MARGIN_MAX_S = 300.0
TIME_ROUND_S = 300       # runs of 2 h or more
TIME_ROUND_SHORT_S = 60  # below 2 h
TIME_ROUND_SHORT_BELOW_S = 7200.0


def _measure_steps(arg, stepra):
    """None -> coldstart's default window; 'short' -> max(30, stepra+10) (at least one radiation
    step plus margin; the 30-step floor keeps >= ~20 normal samples when stepra is small -- lafe,
    stepra 6: stepra+10 left 13); else an int >= stepra+1. Most of the saving comes from the
    shorter spin-up, not from these steps."""
    if arg is None:
        return None
    n = max(30, stepra + 10) if arg == "short" else int(arg)
    if n < stepra + 1:
        raise SystemExit("--measure-steps %d < stepra+1 (%d)" % (n, stepra + 1))
    return n


def plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",), namelist_path=None,
         points_by=None, indicators=None):
    """(nl, run_start, run_end, dt_s, stepra, points, notes) for CASE_DIR. `namelist_path`
    (default CASE_DIR/namelist.input) lets a caller whose case uses a different namelist
    template (cases.json namelist_template, e.g. lafe's d01-only one) plan on that instead.
    points_by/indicators: decision 0060, see wtune_probe_plan.plan_points (None = time-based)."""
    import wtune_config as wcfg
    import wtune_predict as wp
    import wtune_probe_plan as pp
    nl = wp.parse_namelist(namelist_path or os.path.join(case_dir, "namelist.input"))
    _n, run_s, dt_s, _note = wp.get_d01_steps(nl)
    run_start = wp.get_time_control_datetime(nl, "start")
    run_end = run_start + dt.timedelta(seconds=run_s)
    stepra = wp.steps_per_radiation_call(float(wp.g(nl, "physics", "radt", None)), dt_s)
    cfg = copy.deepcopy(wcfg.DEFAULTS["probe"])
    cfg["sources"] = list(sources)
    cfg["spinup_min"] = spinup_min
    if max_points:
        cfg["max_points"] = max_points
    if points_by:
        cfg["points_by"] = points_by
    interval_h = 1 if forcing == ERA5_PIN else None
    points, notes = pp.plan_points(nl, run_start, run_end, dt_s, stepra, spinup_min * 60.0, cfg,
                                   case_dir=case_dir, forcing_interval_h=interval_h,
                                   indicators=indicators)
    return nl, run_start, run_end, dt_s, stepra, points, notes


def _with_io_probe(p):
    if dataclasses.is_dataclass(p):
        return dataclasses.replace(p, io_probe=True)
    q = copy.copy(p)
    q.io_probe = True
    return q


def user_io_frames(nl, run_s):
    """Frames the USER's run writes (WRF semantics, coldstart_sample.wrf_interval_minutes):
    {'history': [n per domain], 'restart': n, 'notes': [...]}. History per domain: 0 when
    io_form_history is 0 or the interval is 0/absent, else floor((run - begin)/interval) + 1
    (0 when begin > run); a resumed run (restart=.true.) skips the t=0 frame unless
    write_hist_at_0h_rst. Restart: floor(run/restart_interval) when restart_interval > 0."""
    import coldstart_sample as cs
    import wtune_predict as wp
    run_min = run_s / 60.0
    io_form = wp.g(nl, "time_control", "io_form_history", 2)
    resumed = bool(wp.g(nl, "time_control", "restart", False))
    skip_t0 = resumed and not wp.g(nl, "time_control", "write_hist_at_0h_rst", False)
    notes = []
    history = []
    for d in range(wp.get_max_dom(nl)):
        iv = cs.wrf_interval_minutes(nl, "history_interval", d)
        begin = cs.wrf_interval_minutes(nl, "history_begin", d)
        if not io_form or iv <= 0 or begin > run_min:
            history.append(0)
            continue
        n = int(math.floor((run_min - begin) / iv + 1e-9)) + 1
        if begin == 0 and skip_t0:
            n -= 1
        history.append(n)
    ri = cs.wrf_interval_minutes(nl, "restart_interval")
    restart = int(math.floor(run_min / ri + 1e-9)) if ri > 0 else 0
    if resumed and restart and not wp.g(nl, "time_control", "override_restart_timers", False):
        notes.append("resumed run without override_restart_timers: WRF keeps the restart file's "
                     "restart interval, so the restart count may differ from the namelist's")
    return {"history": history, "restart": restart, "notes": notes}


def read_probe_writes(pdir):
    """(probe block, parse_rsl_write_seconds result or None, probe frame sim time) for the probe
    segment's plan dir; (None, None, None) when that plan carries no io_probe."""
    import coldstart_sample as cs
    with open(os.path.join(pdir, "plan.json")) as f:
        plan = json.load(f)
    probe = plan.get("io_probe")
    if not probe:
        return None, None, None
    k = probe["window"]
    win_steps = plan["measure_steps"] + int(round(plan["spinup_min"] * 60.0 / plan["dt_s"]))
    at = (dt.datetime.strptime(plan["windows"][k]["t"], cs._WRF_TIME_FMT)
          + dt.timedelta(seconds=int(round(win_steps * plan["dt_s"]))))
    rsl = os.path.join(pdir, "win_%d" % k, "wrf", "rsl.error.0000")
    if not os.path.exists(rsl):
        return probe, None, at
    return probe, cs.parse_rsl_write_seconds(rsl, probe.get("history_prefix", "wrfout_d")), at


def io_estimate(frames, writes, probe_at):
    """io_s = sum_d frames_d x median(history write s of d) + restart frames x sum_d median(restart
    write s of d). 'ok' is False (io_s None) when a stream the user's run writes has no measured
    write line."""
    missing = []
    io_s = 0.0
    history = []
    writes = writes or {"history": {}, "restart": {}}
    for d, n in enumerate(frames["history"], 1):
        samples = writes["history"].get(d, [])
        e = {"domain": d, "frames": n, "samples_s": samples}
        if n and not samples:
            missing.append("history d%02d" % d)
        elif n:
            e["median_s"] = statistics.median(samples)
            e["seconds"] = n * e["median_s"]
            io_s += e["seconds"]
        history.append(e)
    restart = {"frames": frames["restart"],
               "samples_s": {"d%02d" % d: v for d, v in sorted(writes["restart"].items())}}
    if frames["restart"] and not writes["restart"]:
        missing.append("restart")
    elif frames["restart"]:
        restart["per_write_s"] = sum(statistics.median(v) for v in writes["restart"].values())
        restart["seconds"] = frames["restart"] * restart["per_write_s"]
        io_s += restart["seconds"]
    ok = not missing
    return {"ok": ok, "io_s": io_s if ok else None, "missing": missing, "history": history,
            "restart": restart, "notes": list(frames.get("notes", [])),
            "source": "measured (1 frame at %s)" % (probe_at.strftime("%Y-%m-%d_%H:%M:%S")
                                                     if probe_at else "?")}


def recommend_time(range_high_s, io_s):
    """range_high + IO_FACTOR x io_s + margin, margin = TIME_MARGIN_FRAC x range_high clamped to
    [TIME_MARGIN_MIN_S, TIME_MARGIN_MAX_S], rounded UP to 1 min below 2 h and to 5 min above;
    io_s None (no measured I/O) -> range_high + margin, flagged as not covering writes."""
    margin = min(max(TIME_MARGIN_FRAC * range_high_s, TIME_MARGIN_MIN_S), TIME_MARGIN_MAX_S)
    raw = range_high_s + (IO_FACTOR * io_s if io_s is not None else 0.0) + margin
    round_s = TIME_ROUND_SHORT_S if raw < TIME_ROUND_SHORT_BELOW_S else TIME_ROUND_S
    time_s = int(math.ceil(round(raw, 6) / round_s)) * round_s
    return {"time_s": time_s, "time": "%02d:%02d:%02d" % (time_s // 3600, time_s % 3600 // 60,
                                                             time_s % 60),
            "range_high_s": range_high_s, "io_s": io_s, "io_factor": IO_FACTOR,
            "margin_s": margin, "round_s": round_s, "raw_s": raw,
            "covers_io": io_s is not None}


def _ctx(case_dir, nodes, out_dir, max_wait_hours=24.0):
    import scc_sweep as sweep
    import wtune_tune
    clock = wtune_tune.RealClock()
    ctx = wtune_tune.Ctx(
        case=os.path.basename(os.path.abspath(case_dir)), case_json={"prep_dir": case_dir},
        namelist_path=os.path.join(case_dir, "namelist.input"), nodes=nodes, cores_per_node=72,
        hardware_id=sweep.DEFAULT_HARDWARE_ID, steps=50, seed=1, kappa=0.5, budget_s=4 * 3600.0,
        max_nodes=64, out_dir=out_dir, ledger_dir=out_dir, env_source="$HOME/wtune-scc/env.sh",
        sbatch_script="", sacct_cmd="sacct", poll_interval=20.0,
        sbatch_runner=sweep.default_sbatch_runner, sacct_runner=sweep.default_sacct_runner,
        scancel_runner=wtune_tune.default_scancel_runner, ledger_reader=None, proposer=None,
        clock=clock)
    ctx.start_time = clock.now()
    # The 4 h budget counts RUNNING time only; queueing is charged to a separate wall cap
    # (elevconv, 2026-09-27: three 8-node segments queued 4 h behind another user's jobs and the
    # wall-clock budget cancelled them before they ever started).
    ctx.budget_mode = "running"
    ctx.max_wall_s = float(max_wait_hours) * 3600.0
    return ctx


def _json_default(x):
    return x.isoformat() if isinstance(x, (dt.datetime, dt.date)) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wtune predict-rebuild", description=__doc__.split("\n\n")[0])
    ap.add_argument("case_dir")
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--numtiles", type=int, default=1)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--ranks", type=int, default=None, help="default nx*ny")
    ap.add_argument("--forcing", choices=("era5", "auto"), default="era5",
                    help="era5: ARCO-ERA5 with cloud water/ice, 1 h (default); auto: GFS/RDA/... chain")
    ap.add_argument("--spinup-min", type=int, default=None,
                    help="default 10 with era5 (short window, 2026-09-28), 60 with auto")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--points-by", choices=("time", "cost-indicators"), default="time",
                    help="decision 0060: place points by sun/time (default) or by a weather cost "
                         "proxy from --indicator-sources; also reports stratified/regression totals")
    ap.add_argument("--indicator-sources", default=None,
                    help="comma list, ordered fallback (default case_forcing,era5,gfs,stage4,mrms)")
    ap.add_argument("--measure-steps", default=None,
                    help="steps measured after spin-up: an integer >= stepra+1, 'short' for "
                         "max(30, stepra+10) (default with era5), or 'long' for max(30, 2*stepra) "
                         "(default with auto)")
    ap.add_argument("--rain-snow", action="store_true",
                    help="era5 only: add rain/snow from the 137-level store (optional; ~+1 pp on "
                         "zhengzhou, within rerun noise)")
    ap.add_argument("--io-probe", dest="io_probe", action="store_true", default=True,
                    help="time one history frame (+ one restart if the case writes restarts) at "
                         "the end of the last point's segment (default on)")
    ap.add_argument("--no-io-probe", dest="io_probe", action="store_false")
    ap.add_argument("--out", default=None,
                    help="default ./wtune-rebuild-<case>-<utc>, or <project>/predict/<id>/ when "
                         "--project (or $WTUNE_PROJECT / a project.yaml above cwd) resolves "
                         "and --out is not given (decision 0058; never inside CASE_DIR)")
    ap.add_argument("--project", default=None, help="project dir (decision 0058) -- see --out")
    ap.add_argument("--max-wait-hours", type=float, default=24.0,
                    help="wall-clock cap including queueing (default 24); the 4 h budget counts "
                         "only time with a segment running")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    a = ap.parse_args(argv)

    case_dir = os.path.abspath(a.case_dir)
    import wtune_extensions
    refusal = wtune_extensions.refusal_predict_rebuild(case_dir)
    if refusal:
        print("error: %s" % refusal, file=sys.stderr)
        return 2
    forcing = ERA5_PIN if a.forcing == "era5" else "auto"
    # ERA5 default = the short window (spin-up 10 min, max(30, stepra+10) steps): same-day paired
    # checks moved the prediction -1.11% (dorian2019) and +0.38% (lafe) vs 20 min + 2*stepra, and
    # cut zhengzhou's driver wall 805 -> 627 s. GFS (auto) keeps 60 min + the long window.
    spinup = a.spinup_min if a.spinup_min is not None else (10 if forcing == ERA5_PIN else 60)
    ranks = a.ranks or a.nx * a.ny
    t_start = time.time()
    cost_series = None
    if a.points_by == "cost-indicators":
        import wtune_cost_drivers
        import wtune_predict as wp
        _nl0 = wp.parse_namelist(os.path.join(case_dir, "namelist.input"))
        _s0 = wp.get_time_control_datetime(_nl0, "start")
        cost_series = wtune_cost_drivers.cost_series_for_case(
            case_dir, _s0, _s0 + dt.timedelta(seconds=wp.get_d01_steps(_nl0)[1]),
            [x for x in (a.indicator_sources or "").split(",") if x] or None)
    plan_kw = dict(points_by=a.points_by, indicators=cost_series) \
        if a.points_by != "time" else {}
    nl, run_start, run_end, dt_s, stepra, points, notes = plan(
        case_dir, forcing, spinup, a.max_points, **plan_kw)
    measure_arg = a.measure_steps if a.measure_steps is not None else (
        "short" if forcing == ERA5_PIN else None)
    measure_steps = _measure_steps(None if measure_arg == "long" else measure_arg, stepra)
    probe_idx = len(points) - 1 if a.io_probe and points else None
    if probe_idx is not None:
        points[probe_idx] = _with_io_probe(points[probe_idx])
    print("run %s -> %s, dt %.0f s, stepra %d, forcing %s, spin-up %d min, measure %s steps"
          % (run_start, run_end, dt_s, stepra, forcing, spinup,
             measure_steps if measure_steps is not None else "max(30,2*stepra)"))
    for i, p in enumerate(points):
        print("  point %s  %-10s %s%s" % (p.at.strftime("%Y-%m-%d_%H:%M"), p.reason, p.source,
                                          "  + I/O probe" if i == probe_idx else ""))
    for n in notes:
        print("  note: %s" % n)
    print("hardware: %d nodes, %d ranks, %dx%d, numtiles %d, omp %d -> %d parallel jobs x %d nodes"
          % (a.nodes, ranks, a.nx, a.ny, a.numtiles, a.omp, len(points), a.nodes))
    if a.dry_run:
        return 0

    missing = [v for v in ("WPS_DIR", "WRF_TREE") if not os.environ.get(v)]
    if missing:
        print("error: %s not set -- load the site env first" % ", ".join(missing), file=sys.stderr)
        return 2
    if a.rain_snow:
        os.environ["WTUNE_ERA5_RAIN_SNOW"] = "1"
    import wtune_predict as wp
    import wtune_probe_plan as pp
    import wtune_tune
    if a.out:
        out = os.path.abspath(a.out)
    else:
        import wtune_project
        _project_dir = wtune_project.find_project(explicit=getattr(a, "project", None))
        out = os.path.abspath(
            wtune_project.default_out_dir(_project_dir, "predict") if _project_dir
            else "wtune-rebuild-%s-%s" % (os.path.basename(case_dir),
                                          dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")))
    if os.path.commonpath([out, case_dir]) == case_dir:
        print("error: --out must be outside CASE_DIR", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    best = dict(id="target", ranks=ranks, omp=a.omp, nx=a.nx, ny=a.ny, numtiles=a.numtiles,
                dist="block", blocktime=0)
    ctx = _ctx(case_dir, a.nodes, out, max_wait_hours=a.max_wait_hours)
    meas, job_ids = wtune_tune.run_rebuild_points(
        ctx, points, best, a.nodes, case_dir, os.path.join(out, "rebuild"), run_start,
        spinup_min=spinup, forcing=None if forcing == "auto" else forcing,
        measure_steps=measure_steps)
    cal = wp.load_calibration(str(wp.DEFAULT_CALIBRATION))
    sample_points, _g = wp.read_wrfinput_sample_points(case_dir)
    result = pp.combine(meas, run_start, run_end, dt_s, stepra, calibration=cal,
                        sample_points=sample_points, case_dir=case_dir, nl=nl, ranks=ranks,
                        nodes=a.nodes, cost_series=cost_series)
    io = None
    if probe_idx is not None:
        pdir = os.path.join(out, "rebuild", "pt_%d" % probe_idx)
        probe, writes, probe_at = read_probe_writes(pdir)
        io = io_estimate(user_io_frames(nl, (run_end - run_start).total_seconds()), writes,
                         probe_at)
        io.update(point=probe_idx, plan_dir=pdir, probe=probe)
    rec = recommend_time(result["range_high_s"], io["io_s"] if io else None)
    record = {"case_dir": case_dir, "hardware": best, "nodes": a.nodes, "forcing": forcing,
              "spinup_min": spinup, "measure_steps": measure_steps, "rain_snow": a.rain_snow, "job_ids": job_ids, "notes": notes,
              "points": [dict(at=p.at.isoformat(), reason=p.reason, source=p.source,
                              io_probe=bool(getattr(p, "io_probe", False))) for p in points],
              "result": result, "io": io, "recommendation": rec,
              "prediction_wall_s": round(time.time() - t_start, 1),
              "saved_utc": dt.datetime.utcnow().isoformat()}
    path = os.path.join(out, "pred_rebuild.json")
    with open(path, "w") as f:
        json.dump(record, f, default=_json_default, indent=1)
    h = lambda s: "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)  # noqa: E731
    print("predicted compute+init: %s (range %s .. %s; I/O separate)"
          % (h(result["total_s"]), h(result["range_low_s"]), h(result["range_high_s"])))
    if result.get("stratified_total_s") is not None or result.get("regression_total_s") is not None:
        ci = result.get("cost_indicators") or {}
        print("cost-indicator totals (compute+init): interpolated %s, stratified %s, regression %s "
              "(R2 %s), spread %s%%; sources %s"
              % (h(result["total_s"]),
                 h(result["stratified_total_s"]) if result.get("stratified_total_s") else "-",
                 h(result["regression_total_s"]) if result.get("regression_total_s") else "-",
                 ("%.3f" % result["regression_r2"]) if result.get("regression_r2") is not None else "-",
                 ("%.1f" % ci["spread_pct"]) if ci.get("spread_pct") is not None else "-",
                 ci.get("sources")))
    if io is None:
        print("I/O: not measured (--no-io-probe)")
    elif io["ok"]:
        parts = ["d%02d %d x %.1f s" % (e["domain"], e["frames"], e["median_s"])
                 for e in io["history"] if e.get("frames")]
        if io["restart"].get("frames"):
            parts.append("restart %d x %.1f s" % (io["restart"]["frames"],
                                                  io["restart"]["per_write_s"]))
        print("I/O estimate: %s (%s; %s)" % (h(io["io_s"]), ", ".join(parts) or "no output",
                                              io["source"]))
    else:
        print("I/O probe failed: no write line for %s in %s" % (", ".join(io["missing"]),
                                                                io["plan_dir"]))
    for n in (io or {}).get("notes", []) + ((io or {}).get("probe") or {}).get("notes", []):
        print("  note: %s" % n)
    for w in ((io or {}).get("probe") or {}).get("warnings", []):
        print("  WARNING: %s" % w)
    print("recommended --time: %s" % rec["time"])
    if not rec["covers_io"]:
        print("WARNING: recommended --time = compute+init range high + margin only; history/"
              "restart writes are NOT covered")
    print("prediction took %s; saved %s" % (h(record["prediction_wall_s"]), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
