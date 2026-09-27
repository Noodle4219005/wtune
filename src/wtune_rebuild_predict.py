"""Whole-run wall-time prediction from rebuilt segments (decision 0054's final architecture).

Every sampled point is rebuilt from analysis/reanalysis forcing (WPS + real.exe), spun up and
measured on the TARGET hardware config, all points as parallel jobs; the measurements are then
combined into one prediction (wtune_probe_plan.combine). Validated on scc (2026-09-27):
conus25 C3 -0.98 %, zhengzhou -5.78 % / -5.67 % (phase 7), ERA5 spin-up 20 min (phase 3).

    wtune predict-rebuild CASE_DIR --nodes 4 --nx 16 --ny 18 [--numtiles 1] [--omp 1]
        [--forcing era5|auto] [--spinup-min 20] [--max-points 5] [--rain-snow]
        [--out DIR] [--dry-run]

CASE_DIR needs namelist.input and wrfinput_d01 (the geometry and, for --forcing era5, the ERA5
crop box come from it); nothing is written inside it. Must run with the site env loaded (WPS_DIR, WRF_TREE). The prediction is
written to OUT/pred_rebuild.json before anything else is compared with it.
"""
import argparse
import copy
import datetime as dt
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "tools"), os.path.join(_HERE, "predictor"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ERA5_PIN = "era5_arco"


def plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",)):
    """(nl, run_start, run_end, dt_s, stepra, points, notes) for CASE_DIR."""
    import wtune_config as wcfg
    import wtune_predict as wp
    import wtune_probe_plan as pp
    nl = wp.parse_namelist(os.path.join(case_dir, "namelist.input"))
    _n, run_s, dt_s, _note = wp.get_d01_steps(nl)
    run_start = wp.get_time_control_datetime(nl, "start")
    run_end = run_start + dt.timedelta(seconds=run_s)
    stepra = wp.steps_per_radiation_call(float(wp.g(nl, "physics", "radt", None)), dt_s)
    cfg = copy.deepcopy(wcfg.DEFAULTS["probe"])
    cfg["sources"] = list(sources)
    cfg["spinup_min"] = spinup_min
    if max_points:
        cfg["max_points"] = max_points
    interval_h = 1 if forcing == ERA5_PIN else None
    points, notes = pp.plan_points(nl, run_start, run_end, dt_s, stepra, spinup_min * 60.0, cfg,
                                   case_dir=case_dir, forcing_interval_h=interval_h)
    return nl, run_start, run_end, dt_s, stepra, points, notes


def _ctx(case_dir, nodes, out_dir):
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
                    help="default 20 with era5 (phase 3), 60 with auto")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--rain-snow", action="store_true",
                    help="era5 only: add rain/snow from the 137-level store (optional; ~+1 pp on "
                         "zhengzhou, within rerun noise)")
    ap.add_argument("--out", default=None,
                    help="default ./wtune-rebuild-<case>-<utc> (never inside CASE_DIR: the user's case "
                         "directory is not modified)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    a = ap.parse_args(argv)

    case_dir = os.path.abspath(a.case_dir)
    forcing = ERA5_PIN if a.forcing == "era5" else "auto"
    spinup = a.spinup_min if a.spinup_min is not None else (20 if forcing == ERA5_PIN else 60)
    ranks = a.ranks or a.nx * a.ny
    t_start = time.time()
    nl, run_start, run_end, dt_s, stepra, points, notes = plan(case_dir, forcing, spinup,
                                                               a.max_points)
    print("run %s -> %s, dt %.0f s, stepra %d, forcing %s, spin-up %d min"
          % (run_start, run_end, dt_s, stepra, forcing, spinup))
    for p in points:
        print("  point %s  %-10s %s" % (p.at.strftime("%Y-%m-%d_%H:%M"), p.reason, p.source))
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
    out = os.path.abspath(a.out or "wtune-rebuild-%s-%s" % (
        os.path.basename(case_dir), dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")))
    if os.path.commonpath([out, case_dir]) == case_dir:
        print("error: --out must be outside CASE_DIR", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    best = dict(id="target", ranks=ranks, omp=a.omp, nx=a.nx, ny=a.ny, numtiles=a.numtiles,
                dist="block", blocktime=0)
    ctx = _ctx(case_dir, a.nodes, out)
    meas, job_ids = wtune_tune.run_rebuild_points(
        ctx, points, best, a.nodes, case_dir, os.path.join(out, "rebuild"), run_start,
        spinup_min=spinup, forcing=None if forcing == "auto" else forcing)
    cal = wp.load_calibration(str(wp.DEFAULT_CALIBRATION))
    sample_points, _g = wp.read_wrfinput_sample_points(case_dir)
    result = pp.combine(meas, run_start, run_end, dt_s, stepra, calibration=cal,
                        sample_points=sample_points, case_dir=case_dir, nl=nl, ranks=ranks,
                        nodes=a.nodes)
    record = {"case_dir": case_dir, "hardware": best, "nodes": a.nodes, "forcing": forcing,
              "spinup_min": spinup, "rain_snow": a.rain_snow, "job_ids": job_ids, "notes": notes,
              "points": [dict(at=p.at.isoformat(), reason=p.reason, source=p.source) for p in points],
              "result": result, "prediction_wall_s": round(time.time() - t_start, 1),
              "saved_utc": dt.datetime.utcnow().isoformat()}
    path = os.path.join(out, "pred_rebuild.json")
    with open(path, "w") as f:
        json.dump(record, f, default=_json_default, indent=1)
    h = lambda s: "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)  # noqa: E731
    print("predicted compute+init: %s (range %s .. %s; I/O not included)"
          % (h(result["total_s"]), h(result["range_low_s"]), h(result["range_high_s"])))
    print("prediction took %s; saved %s" % (h(record["prediction_wall_s"]), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
