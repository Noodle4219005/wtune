"""post/cli.py -- `wtune post <sub>` subcommands (decision 0056). Only stdlib at import time: each
subcommand's module imports its heavy deps (netCDF4/wrf-python/matplotlib/metpy/cartopy) lazily,
so `wtune post --help` and `wtune --help` never need them installed."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_vars(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def _parse_levels(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_point(text: str) -> tuple[float, float, str]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--points expects LAT,LON[,NAME]: %r" % text)
    lat, lon = float(parts[0]), float(parts[1])
    name = parts[2] if len(parts) == 3 else "%.3f,%.3f" % (lat, lon)
    return (lat, lon, name)


def _parse_box(text: str) -> tuple[float, ...]:
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box expects lat0,lat1,lon0,lon1: %r" % text)
    return tuple(parts)


def cmd_inventory(args: argparse.Namespace) -> int:
    import post.inventory as inventory
    inv = inventory.inventory(args.run_dir)
    if args.json:
        print(json.dumps(inv, indent=2, default=str))
    else:
        print(inventory.render_summary(inv))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    import post.extract as extract
    points = [_parse_point(p) for p in args.points] if args.points else None
    box = _parse_box(args.box) if args.box else None
    result = extract.extract(args.run_dir, _parse_vars(args.vars), args.out,
                             domain=args.domain, times=_parse_vars(args.times) if args.times else None,
                             levels_hpa=_parse_levels(args.levels) if args.levels else None,
                             points=points, box=box, workers=args.workers)
    print("wrote %s (%s, %d frame(s))" % (result["out"], result["kind"], result["n_frames"]))
    return 0


def _project_context(args: argparse.Namespace) -> tuple:
    """--project DIR / $WTUNE_PROJECT / walk-up discovery (decision 0058 item 2). Returns
    (project_dir_or_None, project_doc_or_None). Never raises for "no project found" -- only
    an explicit --project/$WTUNE_PROJECT that does not resolve is an error
    (wtune_project.find_project's own contract, surfaced by the caller)."""
    import wtune_project
    project_dir = wtune_project.find_project(explicit=getattr(args, "project", None))
    if not project_dir:
        return None, None
    return project_dir, wtune_project.load_project(project_dir)


def _resolve_out(args: argparse.Namespace, project_dir: str | None, tag: str) -> str:
    """--out, else the project's default post/<id>/ directory (decision 0058)."""
    out = args.out
    if out:
        return out
    if not project_dir:
        raise SystemExit("--out required (or --project DIR for a default post/<id>/)")
    import wtune_project
    return wtune_project.default_out_dir(project_dir, "post", tag=tag)


def _resolve_run_dirs(args: argparse.Namespace, project_dir: str | None) -> list[str]:
    """RUN_DIRS, else the project's runs/*/ (or case/) that hold wrfout frames."""
    run_dirs = list(args.run_dirs)
    if run_dirs:
        return run_dirs
    if not project_dir:
        raise SystemExit("RUN_DIRS required (or --project DIR to auto-discover runs/*/)")
    import wtune_project
    run_dirs = [p for _id, p in wtune_project.discover_runs(project_dir)]
    if not run_dirs:
        raise SystemExit("--project %s has no runs/*/ or case/ with wrfout frames" % project_dir)
    return run_dirs


def _check_labels(args: argparse.Namespace, run_dirs: list[str], default: list[str]) -> list[str]:
    labels = _parse_vars(args.labels) if args.labels else default
    if len(labels) != len(run_dirs):
        raise SystemExit("--labels must have one entry per RUN_DIR (%d given, %d run dirs)"
                         % (len(labels), len(run_dirs)))
    return labels


def _basenames(run_dirs: list[str]) -> list[str]:
    return [os.path.basename(os.path.normpath(d)) for d in run_dirs]


def _record_manifest(project_dir: str | None, argv: list[str], out: str, key_paths: list[str],
                     exit_code: int = 0) -> None:
    if project_dir:
        import wtune_project
        wtune_project.manifest_append(project_dir, argv, out_dir=out, exit_code=exit_code,
                                      key_paths=key_paths)


def cmd_plot(args: argparse.Namespace) -> int:
    import post.plotting as plotting
    project_dir, _doc = _project_context(args)
    run_dir = args.run_dir
    out = _resolve_out(args, project_dir, "plot")
    written = plotting.plot_field(run_dir, args.field, out, domain=args.domain,
                                  times=_parse_vars(args.times) if args.times else None,
                                  level_hpa=args.level, overlay_wind=args.overlay == "wind",
                                  basemap=args.basemap, fmt=args.format)
    print("wrote %d figure(s) to %s" % (len(written), out))
    _record_manifest(project_dir, ["wtune", "post", "plot", run_dir], out, [run_dir])
    return 0


def cmd_sounding(args: argparse.Namespace) -> int:
    import post.sounding as sounding
    lat, lon = (float(v) for v in args.point.split(","))
    result = sounding.make_sounding(args.run_dir, lat, lon, args.time, args.out,
                                    domain=args.domain)
    print("wrote %s  CAPE=%.0f J/kg  CIN=%.0f J/kg"
         % (result["out"], result["cape_j_kg"], result["cin_j_kg"]))
    return 0


def _print_tc_summary(tc, summary: dict, out: str) -> None:
    print("wrote %d a-deck(s), %d figure(s) to %s" % (len(summary["adecks"]),
                                                       len(summary["figures"]), out))
    for w in summary["warnings"]:
        print("warning: %s" % w)
    for n in summary.get("notes", []):
        print("warning: tracking grid: %s" % n)
    for label, st in summary.get("track_steps", {}).items():
        print(tc.format_track_steps(label, st))


def cmd_tc(args: argparse.Namespace) -> int:
    import post.tc as tc
    project_dir, project_doc = _project_context(args)
    run_dirs = _resolve_run_dirs(args, project_dir)
    labels = _check_labels(args, run_dirs, _basenames(run_dirs))

    hurdat2, storm = args.hurdat2, args.storm
    if project_dir and not hurdat2 and not storm:
        import wtune_project
        storm, hurdat2 = wtune_project.project_storm(project_doc)
    if not (hurdat2 and storm):
        raise SystemExit("--hurdat2/--storm required (or set them in project.yaml's storm:)")

    out = _resolve_out(args, project_dir, "tc")
    summary = tc.run_tc_report(run_dirs, labels, hurdat2, storm, out,
                               ebtrk_path=args.ebtrk, bdeck_path=args.bdeck,
                               smooth_km=args.track_smooth_km)
    _print_tc_summary(tc, summary, out)
    _record_manifest(project_dir, ["wtune", "post", "tc"] + run_dirs, out, run_dirs + [hurdat2])
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import post.compare as compare
    variables = _parse_vars(args.vars) if args.vars else None
    try:
        result = compare.compare_runs(args.run_a, args.run_b, variables=variables,
                                      domain=args.domain)
    except compare.CompareError as e:       # fail closed: an uncomparable pair is not identical
        print("error: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(compare.render_summary(result))
    if result["n_frames_compared"] == 0:
        return 2
    return 0 if result["identical"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    import post.report as report
    out_path = report.build_report(args.out_dir, title=args.title)
    print("wrote %s" % out_path)
    return 0


def _recipe_cross_sections(project_doc: dict) -> tuple[dict, float | None, float | None]:
    """(cross_sections, xsec_top_km, xsec_halfwidth_km) from the project doc."""
    import wtune_project
    cross_sections = dict(
        terrain=wtune_project.project_cross_section(project_doc, "terrain"),
        moisture=wtune_project.project_cross_section(project_doc, "moisture"))
    xsec_cfg = wtune_project.project_xsec_config(project_doc)
    return cross_sections, xsec_cfg.get("top_km"), xsec_cfg.get("halfwidth_km")


def _print_recipe_result(result: dict) -> None:
    n_fig = sum(1 for f in result["figures"] if not f["missing"])
    n_missing = sum(1 for f in result["figures"] if f["missing"])
    print("detected types: %s" % ", ".join(result["case_info"]["types"]))
    print("wrote %d figure(s), %d missing-observation entr(y/ies), report: %s"
         % (n_fig, n_missing, result["report_path"]))
    for n in result.get("notes", []):
        print("%s: %s: %s" % (n["level"], n["source"], n["text"]))


def cmd_recipe(args: argparse.Namespace) -> int:
    import post.recipe as recipe
    project_dir, project_doc = _project_context(args)
    run_dirs = _resolve_run_dirs(args, project_dir)
    labels = _check_labels(args, run_dirs, _basenames(run_dirs) if project_dir else
                           [("run%d" % i) for i in range(len(run_dirs))])
    out = _resolve_out(args, project_dir, "recipe")
    if project_dir:
        import wtune_project
        defaults = wtune_project.project_post_defaults(project_doc)
        used = []
        if args.types is None and defaults["types"]:
            args.types = defaults["types"]
            used.append("--types %s" % args.types)
        if args.every is None and args.times is None and defaults["every"]:
            args.every = defaults["every"]
            used.append("--every %s" % args.every)
        if used:
            print("project defaults: %s" % " ".join(used))

    hurdat2, storm = args.hurdat2, args.storm
    if project_dir and not hurdat2 and not storm:
        import wtune_project
        storm, hurdat2 = wtune_project.project_storm(project_doc)

    cross_sections = xsec_top_km = xsec_halfwidth_km = None
    if project_dir:
        cross_sections, xsec_top_km, xsec_halfwidth_km = _recipe_cross_sections(project_doc)
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
    _print_recipe_result(result)
    argv = getattr(args, "argv", ["recipe"] + run_dirs)
    _record_manifest(project_dir, ["wtune", "post"] + argv, out, run_dirs)
    return 0


def _cf3d_command(args: argparse.Namespace, out: str) -> str:
    """The resolved command line, for the NetCDF `history` attribute (sys.argv differs between
    `wtune post ...` and a direct post/cli.py main(argv) call)."""
    cmd = ["wtune", "post", "cf3d", args.run_dir]
    for flag in ("domain", "every", "times"):
        if getattr(args, flag):
            cmd += ["--%s" % flag, str(getattr(args, flag))]
    return " ".join(cmd + ["--out", out])


def _domains_and_out(args: argparse.Namespace, tag: str) -> tuple:
    """(project_dir, out_dir, domains, bdy_width) for `post cf3d` / `post vapor-bundle`: --domain, else the
    recipe's own rule (d01 + the finest nest, post/recipes/cf3d.cf_domains); without a namelist
    next to the wrfout, d01 and WRF's default boundary width."""
    from post import caseinfo
    from post.recipes import cf3d
    project_dir, _doc = _project_context(args)
    out = _resolve_out(args, project_dir, tag)
    try:
        bdy_width = caseinfo.spec_bdy_width(args.run_dir)
        domains = [args.domain] if args.domain else \
            cf3d.cf_domains(dict(domains=caseinfo.domain_physics(args.run_dir)[1]))
    except caseinfo.CaseInfoError:
        bdy_width, domains = caseinfo.DEFAULT_SPEC_BDY_WIDTH, [args.domain or 1]
    return project_dir, out, domains, bdy_width


def cmd_cf3d(args: argparse.Namespace) -> int:
    from post import cf_pressure
    project_dir, out, domains, bdy_width = _domains_and_out(args, "cf3d")
    results = cf_pressure.export(args.run_dir, out, domains, bdy_width, every=args.every,
                                 times=_parse_vars(args.times) if args.times else None,
                                 command=_cf3d_command(args, out))
    n_err = 0
    for r in results:
        if r.get("error"):
            n_err += 1
            print("error: d%02d %s: %s" % (r["domain"], r["valid_time"], r["error"]))
        else:
            print("wrote %s (%d levels, %dx%d lat-lon, %.1f MB)" % (
                r["path"], len(r["levels"]), r["nlat"], r["nlon"], r["size_bytes"] / 2.0 ** 20))
    _record_manifest(project_dir, ["wtune", "post", "cf3d", args.run_dir], out, [args.run_dir],
                     exit_code=1 if n_err else 0)
    return 1 if n_err or not results else 0


def cmd_vapor_bundle(args: argparse.Namespace) -> int:
    from post import vapor_bundle
    project_dir, out, domains, _bdy = _domains_and_out(args, "vapor")
    results = vapor_bundle.export(args.run_dir, out, domains, every=args.every,
                                  times=_parse_vars(args.times) if args.times else None,
                                  allow_copy=args.allow_copy)
    n_err = 0
    for r in results:
        if r.get("error"):
            n_err += 1
            print("error: d%02d: %s" % (r["domain"], r["error"]))
        else:
            print("bundle %s: %d wrfout file(s), %.1f MB, %s" % (
                r["dir"], r["n_files"], r["total_bytes"] / 2.0 ** 20, r["link_note"]))
            print("  copy with: rsync -av <host>:%s/ ./  (see %s)"
                  % (os.path.abspath(r["dir"]), os.path.basename(r["readme"])))
    _record_manifest(project_dir, ["wtune", "post", "vapor-bundle", args.run_dir], out,
                     [args.run_dir], exit_code=1 if n_err else 0)
    return 1 if n_err or not results else 0


def cmd_custom(args: argparse.Namespace) -> int:
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


def cmd_fetch(args: argparse.Namespace) -> int:
    from post import fetch
    if getattr(args, "list", False):
        print(fetch.list_text())
        return 0
    name = getattr(args, "fetch_name", None)
    if not name:
        raise SystemExit("wtune post fetch <name> --project P [--run-dir R] ... (or --list); "
                         "available: %s" % ", ".join(sorted(fetch.discover()[0])))
    from post import sources
    try:
        paths = fetch.run_module(name, args)
    except sources.FetchIncomplete as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    print("%s: %d file(s)%s" % (name, len(paths), "" if paths else " -- nothing fetched (reason above)"))
    return 0


def _add_fetch_module_parser(subparsers, name, mod, **kw):
    sp = subparsers.add_parser(name, help=mod.HELP, description=mod.HELP, **kw)
    sp.add_argument("--project", required=True, help="project dir; data goes to its %s"
                    % ("input/era5/" if name == "era5" else "input/obs/%s/" % name))
    sp.add_argument("--run-dir", default=None, dest="run_dir", help="run whose wrfout gives the "
                    "time window and region")
    mod.add_arguments(sp)
    _add_nc_cache_arg(sp)
    sp.set_defaults(fetch_name=name)
    return sp


def _add_nc_cache_arg(p):
    p.add_argument("--nc-cache", default=None, dest="nc_cache", metavar="DIR",
                   help="where ADIOS2 BP history (io_form_history=14) is converted to netCDF "
                   "once and reused (default: $WTUNE_NC_CACHE, else <project or output "
                   "dir>/_nc_cache, else <run dir>/_nc_cache)")


_DIR_OUT_COMMANDS = ("plot", "tc", "recipe", "cf3d", "vapor-bundle", "custom")
_FILE_OUT_COMMANDS = ("extract", "sounding")


def _nc_cache_base(args: argparse.Namespace) -> str | None:
    """Default base of the BP -> netCDF cache (<base>/_nc_cache): the project dir, else the
    command's output dir (a file output's directory); None leaves <run dir>/_nc_cache."""
    try:
        project_dir, _doc = _project_context(args)
    except Exception:                                       # noqa: BLE001 -- the command's own
        project_dir = None                                  # project lookup reports it
    if project_dir:
        return project_dir
    out = getattr(args, "out", None)
    if not out:
        return None
    if args.post_command in _DIR_OUT_COMMANDS:
        return out
    if args.post_command in _FILE_OUT_COMMANDS:
        return os.path.dirname(os.path.abspath(out))
    return None


def _arg(*flags: str, **kw) -> tuple[tuple[str, ...], dict]:
    """One add_argument call as data: (flags, kwargs)."""
    return flags, kw


_PROJECT = _arg("--project", default=None, help="project dir (decision 0058); default: "
                "$WTUNE_PROJECT or a project.yaml above the current directory")
_OUT_DIR = _arg("--out", default=None, help="output directory (default: a project's "
                "post/<id>/ -- decision 0058)")
_TRACK_SMOOTH = _arg("--track-smooth-km", type=float, default=30.0, dest="track_smooth_km",
                     metavar="KM", help="TC track position is searched on SLP smoothed by CDO "
                     "(cdo smooth, this radius, the same on every domain; intensity is read from "
                     "the unsmoothed fields). Default 30; 0 = raw SLP minimum (hops between "
                     "mesovortices in a large eye)")
_CF_DOMAIN = _arg("--domain", type=int, default=None, help="default: the finest domain, "
                  "plus d01 when there is a nest (same as `post recipe`)")
_CF_TIMES = _arg("--times", default=None, help="comma-separated WRF times or YYYYmmddHH, "
                 "snapped to the nearest available frame")

# subcommand -> (add_parser kwargs, [add_argument specs in --help order])
_COMMANDS = [
    ("inventory", dict(help="domains/times/grid/variables/missing-frame summary"), [
        _arg("run_dir"), _arg("--json", action="store_true")]),
    ("extract", dict(help="wrf-python diagnostics -> points CSV or box NetCDF"), [
        _arg("run_dir"), _arg("--vars", required=True),
        _arg("--domain", type=int, default=1),
        _arg("--times", default=None, help="comma-separated WRF time strings"),
        _arg("--levels", default=None, help="comma-separated hPa, for z/tc/ua/va/wspd"),
        _arg("--points", action="append", default=None, metavar="LAT,LON[,NAME]"),
        _arg("--box", default=None, metavar="LAT0,LAT1,LON0,LON1"),
        _arg("--out", required=True), _arg("--workers", type=int, default=1)]),
    ("plot", dict(help="filled-contour maps on lat/lon, one file per time"), [
        _arg("run_dir"), _arg("--field", required=True), _arg("--domain", type=int, default=1),
        _arg("--times", default=None),
        _arg("--level", type=float, default=None, metavar="HPA"),
        _arg("--overlay", choices=["wind"], default=None),
        _arg("--basemap", choices=["landmask", "cartopy"], default="cartopy"),
        _arg("--format", choices=["png", "svg"], default="png"), _OUT_DIR, _PROJECT]),
    ("sounding", dict(help="MetPy Skew-T at the nearest grid column"), [
        _arg("run_dir"), _arg("--point", required=True, metavar="LAT,LON"),
        _arg("--time", required=True, metavar="WRF_TIME"),
        _arg("--domain", type=int, default=1),
        _arg("--out", required=True, help="output PNG path")]),
    ("tc", dict(help="TC track (TempestExtremes) + ATCF a-deck vs. b-deck "
                "(MET tc_pairs/tc_stat/tc_rmw)"), [
        _arg("run_dirs", nargs="*", help="default: a project's runs/*/ or case/"),
        _arg("--hurdat2", default=None, help="HURDAT2 file (first-guess centre + "
             "b-deck fallback if --bdeck is not given and the NHC download fails); "
             "default: a project's storm.hurdat2"),
        _arg("--storm", default=None, help="e.g. AL122005 (basin+cyclone+year); "
             "default: a project's storm.id"),
        _arg("--bdeck", default=None, help="local ATCF b-deck file (default: "
             "download from the NHC archive, else fall back to HURDAT2)"),
        _arg("--ebtrk", default=None, help="Extended Best Track file (RMW only, "
             "recorded in obs.json)"),
        _TRACK_SMOOTH,
        _arg("--labels", default=None, help="comma-separated, one per RUN_DIR "
             "(also the ATCF a-deck TECH id, truncated to 4 chars)"),
        _OUT_DIR, _PROJECT]),
    ("compare", dict(help="verify a STRICT change did not move the answer"), [
        _arg("run_a"), _arg("run_b"), _arg("--vars", default=None),
        _arg("--domain", type=int, default=None), _arg("--json", action="store_true")]),
    ("report", dict(help="collect PNGs/CSVs in a dir into one self-contained HTML"), [
        _arg("out_dir"), _arg("--title", default=None)]),
    ("recipe", dict(help="case-type-driven standard figure set + HTML report "
                    "(decision 0057), plus optional --every/--times time series, change "
                    "maps and animations (decision 0058)"), [
        _arg("run_dirs", nargs="*", help="default: a project's runs/*/ or case/"),
        _OUT_DIR, _PROJECT,
        _arg("--types", default=None, help="'auto' or a comma-separated list, e.g. "
             "'tc,convection' (common is always included); default: project.yaml "
             "post.types, else auto"),
        _arg("--ensemble", action="store_true", help="treat RUN_DIRs as ensemble "
             "members (enables the ensemble figure set)"),
        _arg("--spinup-hours", type=float, default=6.0, dest="spinup_hours"),
        _arg("--hurdat2", default=None, help="HURDAT2 file, for tc detection + track "
             "comparison; default: a project's storm.hurdat2"),
        _arg("--storm", default=None, help="e.g. AL122005 (required with --hurdat2); "
             "default: a project's storm.id"),
        _arg("--bdeck", default=None, help="local ATCF b-deck (overrides HURDAT2 for "
             "the tc comparison; also usable alone for tc detection)"),
        _arg("--ebtrk", default=None, help="Extended Best Track file (RMW only)"),
        _arg("--labels", default=None, help="comma-separated, one per RUN_DIR"),
        _arg("--repro-pairs", action="store_true", dest="repro_pairs",
             help="extra RUN_DIRs differ only in decomposition -- draw the "
             "reproducibility-noise envelope (common figure set)"),
        _arg("--no-paraview", action="store_true", dest="no_paraview",
             help="skip the ParaView 3-D cloud rendering (tc/convection) even if "
             "pvbatch is available"),
        _arg("--every", default=None, metavar="6h|1d", help="one full figure set "
             "per valid time at this interval, in post/<id>/timeseries/ (decision "
             "0058 P3); mutually usable with --times"),
        _arg("--times", default=None, help="comma-separated WRF times or "
             "YYYYmmddHH, snapped to the nearest available frame -- an explicit "
             "alternative to --every"),
        _arg("--animate", action="store_true", help="also build an MP4 per "
             "time-series field (needs 2+ selected times; requires ffmpeg)"),
        _TRACK_SMOOTH,
        _arg("--climate-spinup-days", type=float, default=None,
             dest="climate_spinup_days", metavar="N", help="days of spin-up the climate "
             "recipe excludes (default: that recipe's own, 30)"),
        _arg("--title", default=None)]),
    ("cf3d", dict(help="CF-1.8 NetCDF on pressure levels and a regular "
                  "lon-lat grid (Met.3D/ParaView), wrf_d0N_pl_YYYYMMDD_HHMM.nc under "
                  "OUT/cf3d/d0N/"), [
        _arg("run_dir"), _CF_DOMAIN,
        _arg("--every", default=None, metavar="6h|1d", help="one file per valid time "
             "at this interval (default: the last frame only)"),
        _CF_TIMES, _OUT_DIR, _PROJECT]),
    ("vapor-bundle", dict(help="ready-to-copy NCAR VAPOR bundle (wrfout "
                          "frames hard-linked, vapor_render.py, README.md, bundle.json) under "
                          "OUT/vapor/d0N/ -- VAPOR itself runs on an x86_64/macOS workstation"), [
        _arg("run_dir"), _CF_DOMAIN,
        _arg("--every", default=None, metavar="6h|1d", help="frames at this interval "
             "(default: the last frame only)"),
        _CF_TIMES,
        _arg("--allow-copy", action="store_true", dest="allow_copy",
             help="copy the frames when they cannot be hard-linked (OUT on another "
             "filesystem); without it that domain fails with the size it would copy"),
        _OUT_DIR, _PROJECT]),
    ("custom", dict(help="run every analysis/*.py script in a project "
                    "(decision 0058 section 2), collect their figures/tables into one "
                    "HTML report"), [
        _arg("--project", default=None, help="project dir; default: $WTUNE_PROJECT "
             "or a project.yaml above the current directory"),
        _arg("--out", default=None, help="output directory (default: a project's "
             "post/<id>/)")]),
]
_WITH_NC_CACHE = ("inventory", "extract", "plot", "sounding", "tc", "compare", "recipe", "cf3d",
                  "vapor-bundle", "custom")


def _add_fetch_parsers(sub) -> dict:
    """`fetch` (one subparser per fetch module) and the `fetch-era5` alias."""
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
        _add_nc_cache_arg(p_e5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wtune.py post", epilog="--srun (any subcommand): run it as one exclusive "
        "compute-node job step (srun -N1 --exclusive -t %d), excluding head and the project's "
        "exclude_nodes; --srun-minutes N changes that time limit" % POST_JOB_MINUTES)
    sub = parser.add_subparsers(dest="post_command")
    parsers = {}
    for name, kw, specs in _COMMANDS:
        parsers[name] = sub.add_parser(name, **kw)
        for flags, akw in specs:
            parsers[name].add_argument(*flags, **akw)
    _add_fetch_parsers(sub)
    for name in _WITH_NC_CACHE:
        _add_nc_cache_arg(parsers[name])
    return parser


POST_JOB_MINUTES = 120


def srun_command(post_argv: list[str], minutes: int = POST_JOB_MINUTES) -> list[str]:
    """`wtune post <post_argv>` as one exclusive compute-node job step, avoiding the same nodes
    as every other wtune job (head, $WTUNE_SBATCH_EXCLUDE / project exclude_nodes)."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(src, "tools"))
    import scc_sweep
    wtune_bin = os.path.join(os.path.dirname(src), "bin", "wtune")
    return (["srun", "--exclude=" + ",".join(scc_sweep.excluded_nodes()), "-N1", "--exclusive",
             "-t", str(minutes), wtune_bin, "post"] + list(post_argv))


def _pop_srun_minutes(argv: list[str]) -> tuple[list[str], int]:
    """(argv without --srun-minutes N / --srun-minutes=N, minutes)."""
    out, minutes, i = [], POST_JOB_MINUTES, 0
    while i < len(argv):
        a = argv[i]
        if a == "--srun-minutes" or a.startswith("--srun-minutes="):
            value = a.partition("=")[2] if "=" in a else "".join(argv[i + 1:i + 2])
            i += 1 if "=" in a else 2
            if not value.isdigit() or int(value) < 1:
                raise SystemExit("--srun-minutes expects a positive integer: %r" % value)
            minutes = int(value)
            continue
        out.append(a)
        i += 1
    return out, minutes


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    argv, minutes = _pop_srun_minutes(argv)
    if "--srun" in argv:
        cmd = srun_command([a for a in argv if a != "--srun"], minutes)
        print("post job: %s" % " ".join(cmd), flush=True)
        os.execvp(cmd[0], cmd)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = argv
    dispatch = dict(inventory=cmd_inventory, extract=cmd_extract, plot=cmd_plot,
                    sounding=cmd_sounding, tc=cmd_tc, compare=cmd_compare, report=cmd_report,
                    recipe=cmd_recipe, custom=cmd_custom, fetch=cmd_fetch, cf3d=cmd_cf3d)
    dispatch["fetch-era5"] = cmd_fetch
    dispatch["vapor-bundle"] = cmd_vapor_bundle
    fn = dispatch.get(args.post_command)
    if fn is None:
        parser.print_help()
        return 1
    from post import extract
    try:
        with extract.nc_cache(directory=getattr(args, "nc_cache", None),
                              base=_nc_cache_base(args)):
            return fn(args)
    except extract.BPConversionError as e:  # ADIOS2 BP history, converter unavailable
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
