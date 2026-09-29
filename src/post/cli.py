"""post/cli.py -- `wtune post <sub>` subcommands (decision 0056): inventory, extract, plot,
sounding, tc, compare, report, recipe, custom, fetch (+ fetch-era5). Only stdlib (argparse, json) at import time; every
subcommand's own module does its own heavy (netCDF4/wrf-python/matplotlib/metpy/cartopy)
imports lazily, so `wtune post --help` and `wtune --help` never need them installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_vars(text):
    return [v.strip() for v in text.split(",") if v.strip()]


def _parse_levels(text):
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_point(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--points expects LAT,LON[,NAME]: %r" % text)
    lat, lon = float(parts[0]), float(parts[1])
    name = parts[2] if len(parts) == 3 else "%.3f,%.3f" % (lat, lon)
    return (lat, lon, name)


def _parse_box(text):
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box expects lat0,lat1,lon0,lon1: %r" % text)
    return tuple(parts)


def cmd_inventory(args):
    import post.inventory as inventory
    inv = inventory.inventory(args.run_dir)
    if args.json:
        print(json.dumps(inv, indent=2, default=str))
    else:
        print(inventory.render_summary(inv))
    return 0


def cmd_extract(args):
    import post.extract as extract
    points = [_parse_point(p) for p in args.points] if args.points else None
    box = _parse_box(args.box) if args.box else None
    result = extract.extract(args.run_dir, _parse_vars(args.vars), args.out,
                             domain=args.domain, times=_parse_vars(args.times) if args.times else None,
                             levels_hpa=_parse_levels(args.levels) if args.levels else None,
                             points=points, box=box, workers=args.workers)
    print("wrote %s (%s, %d frame(s))" % (result["out"], result["kind"], result["n_frames"]))
    return 0


def _project_context(args):
    """--project DIR / $WTUNE_PROJECT / walk-up discovery (decision 0058 item 2). Returns
    (project_dir_or_None, project_doc_or_None). Never raises for "no project found" -- only
    an explicit --project/$WTUNE_PROJECT that does not resolve is an error
    (wtune_project.find_project's own contract, surfaced by the caller)."""
    import wtune_project
    project_dir = wtune_project.find_project(explicit=getattr(args, "project", None))
    if not project_dir:
        return None, None
    return project_dir, wtune_project.load_project(project_dir)


def cmd_plot(args):
    import post.plotting as plotting
    project_dir, _doc = _project_context(args)
    run_dir = args.run_dir
    out = args.out
    if not out:
        if not project_dir:
            raise SystemExit("--out required (or --project DIR for a default post/<id>/)")
        import wtune_project
        out = wtune_project.default_out_dir(project_dir, "post", tag="plot")
    written = plotting.plot_field(run_dir, args.field, out, domain=args.domain,
                                  times=_parse_vars(args.times) if args.times else None,
                                  level_hpa=args.level, overlay_wind=args.overlay == "wind",
                                  basemap=args.basemap, fmt=args.format)
    print("wrote %d figure(s) to %s" % (len(written), out))
    if project_dir:
        import wtune_project
        wtune_project.manifest_append(project_dir, ["wtune", "post", "plot", run_dir],
                                      out_dir=out, exit_code=0, key_paths=[run_dir])
    return 0


def cmd_sounding(args):
    import post.sounding as sounding
    lat, lon = (float(v) for v in args.point.split(","))
    result = sounding.make_sounding(args.run_dir, lat, lon, args.time, args.out,
                                    domain=args.domain)
    print("wrote %s  CAPE=%.0f J/kg  CIN=%.0f J/kg"
         % (result["out"], result["cape_j_kg"], result["cin_j_kg"]))
    return 0


def cmd_tc(args):
    import post.tc as tc
    project_dir, project_doc = _project_context(args)

    run_dirs = list(args.run_dirs)
    if not run_dirs:
        if not project_dir:
            raise SystemExit("RUN_DIRS required (or --project DIR to auto-discover runs/*/)")
        import wtune_project
        run_dirs = [p for _id, p in wtune_project.discover_runs(project_dir)]
        if not run_dirs:
            raise SystemExit("--project %s has no runs/*/ or case/ with wrfout frames" % project_dir)

    labels = _parse_vars(args.labels) if args.labels else \
        [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("--labels must have one entry per RUN_DIR (%d given, %d run dirs)"
                         % (len(labels), len(run_dirs)))

    hurdat2, storm = args.hurdat2, args.storm
    if project_dir and not hurdat2 and not storm:
        import wtune_project
        storm, hurdat2 = wtune_project.project_storm(project_doc)
    if not (hurdat2 and storm):
        raise SystemExit("--hurdat2/--storm required (or set them in project.yaml's storm:)")

    out = args.out
    if not out:
        if not project_dir:
            raise SystemExit("--out required (or --project DIR for a default post/<id>/)")
        import wtune_project
        out = wtune_project.default_out_dir(project_dir, "post", tag="tc")

    summary = tc.run_tc_report(run_dirs, labels, hurdat2, storm, out,
                               ebtrk_path=args.ebtrk, bdeck_path=args.bdeck,
                               smooth_km=args.track_smooth_km)
    print("wrote %d a-deck(s), %d figure(s) to %s" % (len(summary["adecks"]),
                                                       len(summary["figures"]), out))
    for w in summary["warnings"]:
        print("warning: %s" % w)
    for n in summary.get("notes", []):
        print("warning: tracking grid: %s" % n)
    for label, st in summary.get("track_steps", {}).items():
        print(tc.format_track_steps(label, st))
    if project_dir:
        import wtune_project
        wtune_project.manifest_append(project_dir, ["wtune", "post", "tc"] + run_dirs,
                                      out_dir=out, exit_code=0, key_paths=run_dirs + [hurdat2])
    return 0


def cmd_compare(args):
    import post.compare as compare
    variables = _parse_vars(args.vars) if args.vars else None
    result = compare.compare_runs(args.run_a, args.run_b, variables=variables, domain=args.domain)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(compare.render_summary(result))
    return 0 if result["identical"] or not result["results"] else 1


def cmd_report(args):
    import post.report as report
    out_path = report.build_report(args.out_dir, title=args.title)
    print("wrote %s" % out_path)
    return 0


def cmd_recipe(args):
    import post.recipe as recipe
    project_dir, project_doc = _project_context(args)

    run_dirs = list(args.run_dirs)
    if not run_dirs:
        if not project_dir:
            raise SystemExit("RUN_DIRS required (or --project DIR to auto-discover runs/*/)")
        import wtune_project
        run_dirs = [p for _id, p in wtune_project.discover_runs(project_dir)]
        if not run_dirs:
            raise SystemExit("--project %s has no runs/*/ or case/ with wrfout frames" % project_dir)

    labels = _parse_vars(args.labels) if args.labels else \
        ([os.path.basename(os.path.normpath(d)) for d in run_dirs] if project_dir else
         [("run%d" % i) for i in range(len(run_dirs))])
    if len(labels) != len(run_dirs):
        raise SystemExit("--labels must have one entry per RUN_DIR (%d given, %d run dirs)"
                         % (len(labels), len(run_dirs)))

    out = args.out
    if not out:
        if not project_dir:
            raise SystemExit("--out required (or --project DIR for a default post/<id>/)")
        import wtune_project
        out = wtune_project.default_out_dir(project_dir, "post", tag="recipe")

    hurdat2, storm = args.hurdat2, args.storm
    if project_dir and not hurdat2 and not storm:
        import wtune_project
        storm, hurdat2 = wtune_project.project_storm(project_doc)

    cross_sections = None
    xsec_top_km = xsec_halfwidth_km = None
    if project_dir:
        import wtune_project
        cross_sections = dict(
            terrain=wtune_project.project_cross_section(project_doc, "terrain"),
            moisture=wtune_project.project_cross_section(project_doc, "moisture"))
        xsec_cfg = wtune_project.project_xsec_config(project_doc)
        xsec_top_km, xsec_halfwidth_km = xsec_cfg.get("top_km"), xsec_cfg.get("halfwidth_km")

    if project_dir:
        from post import fetch
        line = fetch.offline_summary(project_dir, run_dirs[0])
        if line:
            print("note: %s" % line)

    types_override = None if args.types in (None, "auto") else args.types
    result = recipe.run_recipe(
        run_dirs, out, types_override=types_override, ensemble=args.ensemble,
        spinup_hours=args.spinup_hours, hurdat2_path=hurdat2, storm_id=storm,
        bdeck_path=args.bdeck, ebtrk_path=args.ebtrk, labels=labels,
        paraview=not args.no_paraview, decomposition_only=args.repro_pairs, title=args.title,
        every=args.every, times=_parse_vars(args.times) if args.times else None,
        cross_sections=cross_sections, animate=args.animate,
        xsec_top_km=xsec_top_km, xsec_halfwidth_km=xsec_halfwidth_km, project_dir=project_dir,
        spinup_days=args.climate_spinup_days, track_smooth_km=args.track_smooth_km)
    n_fig = sum(1 for f in result["figures"] if not f["missing"])
    n_missing = sum(1 for f in result["figures"] if f["missing"])
    print("detected types: %s" % ", ".join(result["case_info"]["types"]))
    print("wrote %d figure(s), %d missing-observation entr(y/ies), report: %s"
         % (n_fig, n_missing, result["report_path"]))
    for n in result.get("notes", []):
        print("%s: %s: %s" % (n["level"], n["source"], n["text"]))
    if project_dir:
        import wtune_project
        wtune_project.manifest_append(project_dir, ["wtune", "post", "recipe"] + run_dirs,
                                      out_dir=out, exit_code=0, key_paths=run_dirs)
    return 0


def cmd_custom(args):
    import post.custom as custom
    import wtune_project
    project_dir = wtune_project.find_project(explicit=args.project)
    if not project_dir:
        raise SystemExit("wtune post custom needs a project (--project DIR or $WTUNE_PROJECT "
                         "or a project.yaml above the current directory)")
    out = args.out or wtune_project.default_out_dir(project_dir, "post", tag="custom")
    run_dirs = wtune_project.discover_runs(project_dir)
    result = custom.run_custom(project_dir, out, run_dirs)
    n_ok = sum(1 for s in result["scripts"] if s["ok"])
    n_err = sum(1 for s in result["scripts"] if not s["ok"])
    print("ran %d script(s) (%d ok, %d error(s)), report: %s"
         % (len(result["scripts"]), n_ok, n_err, result["report_path"]))
    for s in result["scripts"]:
        if not s["ok"]:
            print("error in %s: %s" % (s["script"], s["error"].splitlines()[-1]))
    wtune_project.manifest_append(project_dir, ["wtune", "post", "custom"], out_dir=out,
                                  exit_code=0)
    return 0


def cmd_fetch(args):
    from post import fetch
    if getattr(args, "list", False):
        print(fetch.list_text())
        return 0
    name = getattr(args, "fetch_name", None)
    if not name:
        raise SystemExit("wtune post fetch <name> --project P [--run-dir R] ... (or --list); "
                         "available: %s" % ", ".join(sorted(fetch.discover()[0])))
    paths = fetch.run_module(name, args)
    print("%s: %d file(s)%s" % (name, len(paths), "" if paths else " -- nothing fetched (reason above)"))
    return 0


def _add_fetch_module_parser(subparsers, name, mod, **kw):
    sp = subparsers.add_parser(name, help=mod.HELP, description=mod.HELP, **kw)
    sp.add_argument("--project", required=True, help="project dir; data goes to its %s"
                    % ("input/era5/" if name == "era5" else "input/obs/%s/" % name))
    sp.add_argument("--run-dir", default=None, dest="run_dir", help="run whose wrfout gives the "
                    "time window and region")
    mod.add_arguments(sp)
    sp.set_defaults(fetch_name=name)
    return sp


def build_parser():
    parser = argparse.ArgumentParser(prog="wtune.py post")
    sub = parser.add_subparsers(dest="post_command")

    p_inv = sub.add_parser("inventory", help="domains/times/grid/variables/missing-frame summary")
    p_inv.add_argument("run_dir")
    p_inv.add_argument("--json", action="store_true")

    p_ex = sub.add_parser("extract", help="wrf-python diagnostics -> points CSV or box NetCDF")
    p_ex.add_argument("run_dir")
    p_ex.add_argument("--vars", required=True)
    p_ex.add_argument("--domain", type=int, default=1)
    p_ex.add_argument("--times", default=None, help="comma-separated WRF time strings")
    p_ex.add_argument("--levels", default=None, help="comma-separated hPa, for z/tc/ua/va/wspd")
    p_ex.add_argument("--points", action="append", default=None, metavar="LAT,LON[,NAME]")
    p_ex.add_argument("--box", default=None, metavar="LAT0,LAT1,LON0,LON1")
    p_ex.add_argument("--out", required=True)
    p_ex.add_argument("--workers", type=int, default=1)

    p_pl = sub.add_parser("plot", help="filled-contour maps on lat/lon, one file per time")
    p_pl.add_argument("run_dir")
    p_pl.add_argument("--field", required=True)
    p_pl.add_argument("--domain", type=int, default=1)
    p_pl.add_argument("--times", default=None)
    p_pl.add_argument("--level", type=float, default=None, metavar="HPA")
    p_pl.add_argument("--overlay", choices=["wind"], default=None)
    p_pl.add_argument("--basemap", choices=["landmask", "cartopy"], default="cartopy")
    p_pl.add_argument("--format", choices=["png", "svg"], default="png")
    p_pl.add_argument("--out", default=None, help="output directory (default: a project's "
                      "post/<id>/ -- decision 0058)")
    p_pl.add_argument("--project", default=None, help="project dir (decision 0058); default: "
                      "$WTUNE_PROJECT or a project.yaml above the current directory")

    p_sd = sub.add_parser("sounding", help="MetPy Skew-T at the nearest grid column")
    p_sd.add_argument("run_dir")
    p_sd.add_argument("--point", required=True, metavar="LAT,LON")
    p_sd.add_argument("--time", required=True, metavar="WRF_TIME")
    p_sd.add_argument("--domain", type=int, default=1)
    p_sd.add_argument("--out", required=True, help="output PNG path")

    p_tc = sub.add_parser("tc", help="TC track (TempestExtremes) + ATCF a-deck vs. b-deck "
                          "(MET tc_pairs/tc_stat/tc_rmw)")
    p_tc.add_argument("run_dirs", nargs="*", help="default: a project's runs/*/ or case/")
    p_tc.add_argument("--hurdat2", default=None, help="HURDAT2 file (first-guess centre + "
                      "b-deck fallback if --bdeck is not given and the NHC download fails); "
                      "default: a project's storm.hurdat2")
    p_tc.add_argument("--storm", default=None, help="e.g. AL122005 (basin+cyclone+year); "
                      "default: a project's storm.id")
    p_tc.add_argument("--bdeck", default=None, help="local ATCF b-deck file (default: "
                      "download from the NHC archive, else fall back to HURDAT2)")
    p_tc.add_argument("--ebtrk", default=None, help="Extended Best Track file (RMW only, "
                      "recorded in obs.json)")
    p_tc.add_argument("--track-smooth-km", type=float, default=30.0, dest="track_smooth_km",
                    metavar="KM", help="TC track position is searched on SLP smoothed by CDO "
                    "(cdo smooth, this radius, the same on every domain; intensity is read from "
                    "the unsmoothed fields). Default 30; 0 = raw SLP minimum (hops between "
                    "mesovortices in a large eye)")
    p_tc.add_argument("--labels", default=None, help="comma-separated, one per RUN_DIR "
                      "(also the ATCF a-deck TECH id, truncated to 4 chars)")
    p_tc.add_argument("--out", default=None, help="output directory (default: a project's "
                      "post/<id>/ -- decision 0058)")
    p_tc.add_argument("--project", default=None, help="project dir (decision 0058); default: "
                      "$WTUNE_PROJECT or a project.yaml above the current directory")

    p_cmp = sub.add_parser("compare", help="verify a STRICT change did not move the answer")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--vars", default=None)
    p_cmp.add_argument("--domain", type=int, default=None)
    p_cmp.add_argument("--json", action="store_true")

    p_rep = sub.add_parser("report", help="collect PNGs/CSVs in a dir into one self-contained HTML")
    p_rep.add_argument("out_dir")
    p_rep.add_argument("--title", default=None)

    p_rec = sub.add_parser("recipe", help="case-type-driven standard figure set + HTML report "
                           "(decision 0057), plus optional --every/--times time series, change "
                           "maps and animations (decision 0058)")
    p_rec.add_argument("run_dirs", nargs="*", help="default: a project's runs/*/ or case/")
    p_rec.add_argument("--out", default=None, help="output directory (default: a project's "
                       "post/<id>/ -- decision 0058)")
    p_rec.add_argument("--project", default=None, help="project dir (decision 0058); default: "
                       "$WTUNE_PROJECT or a project.yaml above the current directory")
    p_rec.add_argument("--types", default="auto", help="'auto' (default) or a comma-separated "
                       "list, e.g. 'tc,convection' (common is always included)")
    p_rec.add_argument("--ensemble", action="store_true", help="treat RUN_DIRs as ensemble "
                       "members (enables the ensemble figure set)")
    p_rec.add_argument("--spinup-hours", type=float, default=6.0, dest="spinup_hours")
    p_rec.add_argument("--hurdat2", default=None, help="HURDAT2 file, for tc detection + track "
                       "comparison; default: a project's storm.hurdat2")
    p_rec.add_argument("--storm", default=None, help="e.g. AL122005 (required with --hurdat2); "
                       "default: a project's storm.id")
    p_rec.add_argument("--bdeck", default=None, help="local ATCF b-deck (overrides HURDAT2 for "
                       "the tc comparison; also usable alone for tc detection)")
    p_rec.add_argument("--ebtrk", default=None, help="Extended Best Track file (RMW only)")
    p_rec.add_argument("--labels", default=None, help="comma-separated, one per RUN_DIR")
    p_rec.add_argument("--repro-pairs", action="store_true", dest="repro_pairs",
                       help="extra RUN_DIRs differ only in decomposition -- draw the "
                       "reproducibility-noise envelope (common figure set)")
    p_rec.add_argument("--no-paraview", action="store_true", dest="no_paraview",
                       help="skip the ParaView 3-D cloud rendering (tc/convection) even if "
                       "pvbatch is available")
    p_rec.add_argument("--every", default=None, metavar="6h|1d", help="one full figure set "
                       "per valid time at this interval, in post/<id>/timeseries/ (decision "
                       "0058 P3); mutually usable with --times")
    p_rec.add_argument("--times", default=None, help="comma-separated WRF times or "
                       "YYYYmmddHH, snapped to the nearest available frame -- an explicit "
                       "alternative to --every")
    p_rec.add_argument("--animate", action="store_true", help="also build an MP4 per "
                       "time-series field (needs 2+ selected times; requires ffmpeg)")
    p_rec.add_argument("--track-smooth-km", type=float, default=30.0, dest="track_smooth_km",
                    metavar="KM", help="TC track position is searched on SLP smoothed by CDO "
                    "(cdo smooth, this radius, the same on every domain; intensity is read from "
                    "the unsmoothed fields). Default 30; 0 = raw SLP minimum (hops between "
                    "mesovortices in a large eye)")
    p_rec.add_argument("--climate-spinup-days", type=float, default=None,
                       dest="climate_spinup_days", metavar="N", help="days of spin-up the climate "
                       "recipe excludes (default: that recipe's own, 30)")
    p_rec.add_argument("--title", default=None)

    p_cus = sub.add_parser("custom", help="run every analysis/*.py script in a project "
                           "(decision 0058 section 2), collect their figures/tables into one "
                           "HTML report")
    p_cus.add_argument("--project", default=None, help="project dir; default: $WTUNE_PROJECT "
                       "or a project.yaml above the current directory")
    p_cus.add_argument("--out", default=None, help="output directory (default: a project's "
                       "post/<id>/)")

    from post import fetch
    mods, _errors = fetch.discover()
    p_fet = sub.add_parser("fetch", help="download observation/reanalysis data into a project's "
                           "input/ (login node: compute nodes have no internet)")
    p_fet.add_argument("--list", action="store_true", help="list fetch modules: NAME, HELP, "
                       "coverage_time")
    fsub = p_fet.add_subparsers(dest="fetch_name")
    for name in sorted(mods):
        try:
            _add_fetch_module_parser(fsub, name, mods[name])
        except argparse.ArgumentError as e:     # a module's own option clashes with --project/...
            print("post fetch: module %s disabled: %s" % (name, e), file=sys.stderr)
    if "era5" in mods:
        p_e5 = sub.add_parser("fetch-era5", help="alias of `fetch era5`")
        p_e5.add_argument("--project", required=True)
        p_e5.add_argument("--run-dir", default=None, dest="run_dir")
        mods["era5"].add_arguments(p_e5)
        p_e5.set_defaults(fetch_name="era5")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = dict(inventory=cmd_inventory, extract=cmd_extract, plot=cmd_plot,
                    sounding=cmd_sounding, tc=cmd_tc, compare=cmd_compare, report=cmd_report,
                    recipe=cmd_recipe, custom=cmd_custom, fetch=cmd_fetch)
    dispatch["fetch-era5"] = cmd_fetch
    fn = dispatch.get(args.post_command)
    if fn is None:
        parser.print_help()
        return 1
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
