"""Offline rescore of a predict-rebuild OUT dir whose segments COMPLETED but whose driver died
before combine (dorian_short, netCDF thread race). Same steps as run_rebuild_points' tail and
wtune_rebuild_predict.main's combine; reads only plan.json + rsl. Usage: CASE OUT NODES RANKS"""
import datetime, glob, json, os, sys
sys.path[:0] = [os.path.expanduser("~/wtune-src/src"), os.path.expanduser("~/wtune-src/src/tools")]
import coldstart_sample as coldstart
import wtune_predict as wp
import wtune_probe_plan as pp
import wtune_rebuild_predict as wrp

case, out, nodes, ranks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
plans = []
for pdir in sorted(glob.glob(os.path.join(out, "rebuild", "pt_*"))):
    plan = json.load(open(os.path.join(pdir, "plan.json")))
    plan["_plan_dir"] = pdir
    plans.append(plan)
p0 = plans[0]
nl, run_start, run_end, dt_s, stepra, _pts, _notes = wrp.plan(case, "era5_arco" if "arco" in json.dumps(p0.get("forcing", "arco")) else "auto",
                                                         int(p0["spinup_min"]), len(plans))
sample_points, _g = wp.read_wrfinput_sample_points(case)
cal = wp.load_calibration(str(wp.DEFAULT_CALIBRATION))
meas = []
for plan in plans:
    wo, _pooled, anchors = coldstart.score_windows_from_rsl(plan, sample_points, cal)
    for w, (a_s, _s) in zip(wo, anchors):
        meas.append(pp.PointMeasurement(at=run_start + datetime.timedelta(seconds=a_s), reason=w["reason"],
                                        source="rebuild", s_normal=w["s_k"], n_rad=w["n_rad"], r=w["r_local"]))
        print("point", w["t"], "s_k %.6f n_normal %d n_rad %d r %s" % (w["s_k"], w["n_normal"], w["n_rad"], w["r_local"]))
res = pp.combine(meas, run_start, run_end, dt_s, stepra, calibration=cal, sample_points=sample_points,
                 case_dir=case, nl=nl, ranks=ranks, nodes=nodes)
print({k: res[k] for k in ("compute_total_s", "init_s", "total_s", "range_low_s", "range_high_s", "r_source")})
json.dump({"rescored_offline": True, "result": res}, open(os.path.join(out, "pred_rebuild_rescored.json"), "w"),
          default=str, indent=1)
