#!/usr/bin/env python3
"""analysis/cli.py -- `wtune/cli.py analyze` subcommands: tc-track, diag.

    wtune/cli.py analyze tc-track --wrfout 'glob' --hurdat2 FILE --storm AL122005 --out DIR
        [--first-guess LAT,LON] [--vars slp,wspd10,pw,td2]
        [--mintime-h H] [--maxgap-h H] [--range-deg D] [--ocean-only-naive]

        Writes DIR/fields.nc (analysis.diagnostics), DIR/track.json (full result incl.
        TempestExtremes provenance), DIR/track.csv (chosen track), DIR/compare.csv
        (tracker vs. naive vs. best track per matched time), prints a short table.

    wtune/cli.py analyze diag --wrfout 'glob' --vars slp,pw,wspd10 --out FILE.nc

        Writes wrf-python derived diagnostic fields only (analysis.diagnostics), no
        tracking. Useful to inspect the fields TempestExtremes will see.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wtune.analysis import diagnostics as diag_mod        # analysis/diagnostics.py
from wtune.analysis import tc_track                       # analysis/tc_track.py


def _parse_vars(text):
    return [v.strip() for v in text.split(",") if v.strip()]


def _parse_latlon(text):
    lat_s, lon_s = text.split(",")
    return float(lat_s), float(lon_s)


def _serialize_track(track):
    return [{**pt, "time": pt["time"].strftime(tc_track.WRF_TIME_FMT)} for pt in track]


def cmd_diag(args):
    variables = _parse_vars(args.vars) if args.vars else list(diag_mod.DEFAULT_VARIABLES)
    meta = diag_mod.wrfout_to_tc_fields(args.wrfout, args.out, variables=variables)
    print(f"wrote {meta['out_nc']}  ({meta['n_times']} frames, grid={meta['grid_type']}, "
         f"map_proj={meta['map_proj_name']})")
    if not meta["separable"]:
        print(f"  non-separable grid: max_lat_dev={meta['max_lat_dev_deg']:.4g} deg, "
             f"max_lon_dev={meta['max_lon_dev_deg']:.4g} deg -- wrote flattened ncol layout")
    return 0


def cmd_tc_track(args):
    os.makedirs(args.out, exist_ok=True)
    variables = _parse_vars(args.vars) if args.vars else list(diag_mod.DEFAULT_VARIABLES)
    if "slp" not in variables:
        variables = ["slp"] + variables
    if "wspd10" not in variables:
        variables = variables + ["wspd10"]

    fields_nc = os.path.join(args.out, "fields.nc")
    field_meta = diag_mod.wrfout_to_tc_fields(args.wrfout, fields_nc, variables=variables)

    params = tc_track.TCTrackParams(
        mintime_hours=args.mintime_h, maxgap_hours=args.maxgap_h,
        stitch_range_deg=args.range_deg)
    tracks, provenance = tc_track.detect_and_stitch(fields_nc, args.out, params=params)
    if not tracks:
        print("no tracks found (DetectNodes/StitchNodes produced none passing the "
             "closed-contour/mintime criteria)", file=sys.stderr)
        return 1

    best = tc_track.parse_hurdat2(args.hurdat2, args.storm)

    netcdf4 = __import__("netCDF4")
    with netcdf4.Dataset(fields_nc) as ds:
        t0_hours = float(ds.variables["time"][0])
    from datetime import timedelta
    t0 = datetime(1970, 1, 1) + timedelta(hours=t0_hours)

    if args.first_guess:
        guess_lat, guess_lon = _parse_latlon(args.first_guess)
    else:
        nearest = min(best, key=lambda p: abs((p["time"] - t0).total_seconds()))
        guess_lat, guess_lon = nearest["lat"], nearest["lon"]

    chosen = tc_track.select_track(tracks, guess_lat, guess_lon, t0)
    if chosen is None:
        print("select_track picked nothing (empty tracks list)", file=sys.stderr)
        return 1

    comparison = tc_track.compare_best_track(
        chosen, best, wrfout_paths=args.wrfout, ocean_only=not args.no_ocean_only_naive)

    track_json = {
        "field_meta": field_meta,
        "provenance": provenance,
        "storm_id": args.storm.upper(),
        "hurdat2_file": os.path.abspath(args.hurdat2),
        "first_guess": {"lat": guess_lat, "lon": guess_lon, "t0": t0.strftime(tc_track.WRF_TIME_FMT)},
        "n_tracks_found": len(tracks),
        "chosen_track": _serialize_track(chosen),
        "comparison": comparison,
    }
    with open(os.path.join(args.out, "track.json"), "w") as fh:
        json.dump(track_json, fh, indent=2)

    with open(os.path.join(args.out, "track.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "lat", "lon", "pmin_hpa", "vmax_ms"])
        for pt in chosen:
            w.writerow([pt["time"].strftime(tc_track.WRF_TIME_FMT), pt["lat"], pt["lon"],
                       pt["pmin_hpa"], pt["vmax_ms"]])

    compare_rows = comparison["rows"]
    with open(os.path.join(args.out, "compare.csv"), "w", newline="") as fh:
        if compare_rows:
            w = csv.DictWriter(fh, fieldnames=list(compare_rows[0].keys()))
            w.writeheader()
            w.writerows(compare_rows)

    print(f"{len(tracks)} track(s) found; chosen track has {len(chosen)} points "
         f"(first-guess {guess_lat:.2f},{guess_lon:.2f} @ {t0.strftime(tc_track.WRF_TIME_FMT)})")
    print(f"{'lead_h':>7} {'trk_err_km':>11} {'naive_err_km':>13} "
         f"{'trk_pmin_err':>13} {'naive_pmin_err':>15} {'trk_vmax_err':>13} {'naive_vmax_err':>15}")
    for r in compare_rows:
        def _fmt(v):
            return f"{v:.2f}" if v is not None else "n/a"
        print(f"{r['lead_h']:7.1f} {_fmt(r['tracker_track_error_km']):>11} "
             f"{_fmt(r['naive_track_error_km']):>13} {_fmt(r['tracker_pmin_error_hpa']):>13} "
             f"{_fmt(r['naive_pmin_error_hpa']):>15} {_fmt(r['tracker_vmax_error_ms']):>13} "
             f"{_fmt(r['naive_vmax_error_ms']):>15}")
    print(f"tracker mean track error: {comparison['tracker']['track_error_km']['mean']}")
    if comparison["naive"]:
        print(f"naive   mean track error: {comparison['naive']['track_error_km']['mean']}")
    print(comparison["vmax_caveat"])
    print(f"wrote {os.path.join(args.out, 'track.json')}, track.csv, compare.csv")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="wtune.py analyze")
    sub = parser.add_subparsers(dest="analyze_command")

    p_diag = sub.add_parser("diag", help="wrf-python derived diagnostic fields, no tracking")
    p_diag.add_argument("--wrfout", required=True, action="append",
                        help="wrfout glob/path (repeatable)")
    p_diag.add_argument("--vars", default=None,
                        help="comma-separated: slp,wspd10,u10,v10,pw,td2 (default: all)")
    p_diag.add_argument("--out", required=True, help="output NetCDF path")

    p_tc = sub.add_parser("tc-track", help="TempestExtremes DetectNodes+StitchNodes TC track "
                          "vs. HURDAT2 best track, side by side with the naive method")
    p_tc.add_argument("--wrfout", required=True, action="append",
                      help="wrfout glob/path (repeatable)")
    p_tc.add_argument("--hurdat2", required=True, help="HURDAT2 best-track file")
    p_tc.add_argument("--storm", required=True, help="HURDAT2 storm id, e.g. AL122005")
    p_tc.add_argument("--out", required=True, help="output directory")
    p_tc.add_argument("--first-guess", default=None, metavar="LAT,LON",
                      help="first-guess centre (default: HURDAT2 position nearest t0)")
    p_tc.add_argument("--vars", default=None,
                      help="extra diagnostic fields beyond slp,wspd10 (comma-separated)")
    p_tc.add_argument("--mintime-h", type=float, default=24.0,
                      help="StitchNodes --mintime, hours (default 24)")
    p_tc.add_argument("--maxgap-h", type=float, default=24.0,
                      help="StitchNodes --maxgap, hours (default 24)")
    p_tc.add_argument("--range-deg", type=float, default=8.0,
                      help="StitchNodes --range, degrees (default 8.0)")
    p_tc.add_argument("--no-ocean-only-naive", action="store_true",
                      help="do not restrict the naive comparison's storm_centres() search "
                           "to ocean points")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.analyze_command == "diag":
        return cmd_diag(args)
    if args.analyze_command == "tc-track":
        return cmd_tc_track(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
