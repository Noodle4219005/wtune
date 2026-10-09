"""scenario.cli -- `wtune scenario ...` subcommands: list, generate, inspect. Wired into
src/wtune_cli.py via a single 'scenario' subparser that forwards its remainder args to main()
(same REMAINDER-forwarding pattern wtune_cli.py already uses for 'metdata'/'analyze')."""

from __future__ import annotations

import argparse
import datetime
import inspect
import sys
from pathlib import Path

from . import wps_intermediate as wi
from .base import ScenarioError, ScenarioRequest, get_generator, list_generators, valid_times
from .fill import compute_missing
from .generators.analog import select_analog_date


def _parse_bbox(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ScenarioError("--bbox needs 4 comma-separated values: lat_min,lat_max,lon_min,lon_max")
    return tuple(parts)


def _parse_iso(s: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        raise ScenarioError("bad date %r: use ISO form, e.g. 2019-01-28T00:00" % s)


def _interval_h_from_case(case_dir):
    """(interval_h, source): hours between forcing times, and which file it came from. Tries the
    case's own namelist.wps &share.interval_seconds first -- SAME field and parse path as
    tools/metdata.py's resolve_wps_times() (parse_namelist() + first_value()), so a case's forcing
    cadence means one thing across both tools. Falls back to namelist.input's own
    &time_control.interval_seconds (the field real.exe/wrf.exe themselves enforce for boundary
    reads) when namelist.wps is absent -- most user cases ship only namelist.input, not a
    namelist.wps. 21600s (6h, WPS's own conventional default) when neither file exists under
    case_dir or has a usable interval_seconds."""
    import wtune_cli as wt

    wps_path = Path(case_dir) / "namelist.wps"
    if wps_path.exists():
        try:
            nlw = wt.parse_namelist(str(wps_path))
            share = nlw.get("share", {})
            if "interval_seconds" in share:
                interval_s = int(float(wt.first_value(share["interval_seconds"])))
                return interval_s / 3600.0, "%s (&share.interval_seconds)" % wps_path
        except Exception:
            pass

    input_path = Path(case_dir) / "namelist.input"
    if input_path.exists():
        try:
            nlw = wt.parse_namelist(str(input_path))
            time_control = nlw.get("time_control", {})
            if "interval_seconds" in time_control:
                interval_s = int(float(wt.first_value(time_control["interval_seconds"])))
                return interval_s / 3600.0, "%s (&time_control.interval_seconds)" % input_path
        except Exception:
            pass

    return 21600 / 3600.0, "default (no namelist.wps/namelist.input interval_seconds under %s)" % case_dir


def _resolve_interval_h(args, caps):
    """(interval_h, note_or_None, coarsened_dict_or_None). Precedence: explicit --interval-h wins
    outright; else --case-dir's own namelist.wps interval; else the ScenarioRequest default (6h).
    If the CHOSEN generator can't produce forcing that dense (Caps.min_interval_h coarser than
    what was resolved), coarsen to min_interval_h rather than interpolating in time -- the
    generator's own forecast cadence is a hard floor, not something this layer fabricates data
    between."""
    if args.interval_h is not None:
        requested_h = float(args.interval_h)
        source = "--interval-h"
    elif args.case_dir is not None:
        requested_h, source = _interval_h_from_case(args.case_dir)
    else:
        requested_h = 6.0
        source = "default (no --interval-h, no --case-dir)"

    used_h = requested_h
    coarsened = None
    if caps.min_interval_h > requested_h:
        used_h = float(caps.min_interval_h)
        coarsened = {"requested_h": requested_h, "used_h": used_h}
        note = ("scenario generate: interval %gh from %s; %r's min_interval_h=%gh is coarser "
                 "-- coarsening to %gh, NOT interpolating in time"
                 % (requested_h, source, args.generator, used_h, used_h))
    else:
        note = "scenario generate: interval %gh from %s" % (requested_h, source)
    return used_h, note, coarsened


def cmd_list(args) -> int:
    rows = list_generators()
    widths = {"name": 12, "license": 62, "noncommercial": 15, "needs_gpu": 11, "implemented": 11}
    header = ("name".ljust(widths["name"]) + "license".ljust(widths["license"])
              + "noncommercial".ljust(widths["noncommercial"]) + "needs_gpu".ljust(widths["needs_gpu"])
              + "implemented")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(str(row["name"]).ljust(widths["name"])
              + str(row["license"]).ljust(widths["license"])
              + str(row["noncommercial"]).ljust(widths["noncommercial"])
              + str(row["needs_gpu"]).ljust(widths["needs_gpu"])
              + str(row["implemented"]))
    return 0


def cmd_generate(args) -> int:
    bbox = _parse_bbox(args.bbox)
    start = _parse_iso(args.start)
    end = _parse_iso(args.end)
    analog_date = _parse_iso(args.analog_date) if args.analog_date else None

    gen = get_generator(args.generator, allow_noncommercial=args.allow_noncommercial)
    caps = gen.capabilities()
    interval_h, note, coarsened = _resolve_interval_h(args, caps)
    if note:
        print(note)

    req = ScenarioRequest(bbox=bbox, start=start, end=end, interval_h=interval_h,
                            out_dir=args.out, case_dir=args.case_dir, analog_date=analog_date)

    if args.dry_run:
        analog_date_chosen = req.analog_date or select_analog_date(req.start)
        missing, fill_rules = compute_missing(caps, generator_name=args.generator)
        times = valid_times(req)
        print("generator: %s (license: %s)" % (args.generator, caps.license))
        print("analog date: %s" % analog_date_chosen.isoformat())
        print("interval_h: %g (interval_seconds=%d)" % (interval_h, round(interval_h * 3600)))
        print("files expected: %d (%s)" % (len(times), ", ".join(t.strftime("%Y-%m-%d_%H")
                                                                    for t in times)))
        if missing:
            print("fields missing + fill sources:")
            for f in missing:
                print("  %s -> %s" % (f, fill_rules[f]))
        else:
            print("fields missing: none")
        print("license: %s" % caps.license)
        return 0

    # Wire in the analog generator as the soil/SST/snow/seaice/LANDSEA fill source for any
    # generator whose generate() accepts one (currently: aifs) -- without this, `wtune scenario
    # generate --generator aifs` silently produces files missing every field aifs.py's own
    # netCDF-to-WPS conversion never supplies, and metgrid/real.exe fail or spin those fields up
    # from nothing instead of using real analog forcing.
    gen_kwargs = {}
    if "analog_generator" in inspect.signature(gen.generate).parameters and args.generator != "analog":
        gen_kwargs["analog_generator"] = get_generator(
            "analog", allow_noncommercial=args.allow_noncommercial)
    forcing = gen.generate(req, **gen_kwargs)
    # interval_seconds is what a downstream namelist-rendering stage must set on ITS OWN COPY of
    # namelist.wps/namelist.input (&share/&time_control interval_seconds) to match the files this
    # call actually wrote -- never the formal case's own namelist. interval_coarsened is present
    # only when the generator's own min_interval_h forced a coarser cadence than requested.
    forcing.provenance["interval_seconds"] = round(interval_h * 3600)
    if coarsened:
        forcing.provenance["interval_coarsened"] = coarsened
    forcing.to_json(args.out / "forcing.json")
    print("wrote %d file(s) to %s (manifest: %s)"
          % (len(forcing.files), args.out, args.out / "forcing.json"))
    return 0


def cmd_inspect(args) -> int:
    records = wi.read_fields(args.file)
    print("%-19s %-9s %12s %5s %5s %14s %14s" % ("HDATE", "FIELD", "XLVL", "NX", "NY", "MIN", "MAX"))
    for rec in records:
        print("%-19s %-9s %12.1f %5d %5d %14.6g %14.6g"
              % (rec.hdate, rec.field, rec.xlvl, rec.nx, rec.ny,
                 float(rec.slab.min()), float(rec.slab.max())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wtune scenario")
    sub = parser.add_subparsers(dest="scenario_command", required=True)

    sub.add_parser("list", help="list every registered scenario generator")

    p_gen = sub.add_parser("generate", help="generate WPS forcing for a fictional date")
    p_gen.add_argument("--generator", required=True)
    p_gen.add_argument("--start", required=True, help="ISO-8601 fictional start time")
    p_gen.add_argument("--end", required=True, help="ISO-8601 fictional end time")
    p_gen.add_argument("--bbox", required=True, help="lat_min,lat_max,lon_min,lon_max")
    p_gen.add_argument("--out", required=True, type=Path)
    p_gen.add_argument("--allow-noncommercial", action="store_true", dest="allow_noncommercial")
    p_gen.add_argument("--case-dir", default=None, dest="case_dir", type=Path)
    p_gen.add_argument("--interval-h", default=None, dest="interval_h", type=float,
                        help="forcing cadence in hours; default: derived from "
                             "--case-dir/namelist.wps's interval_seconds when --case-dir is "
                             "given, else 6h. Coarsened up to the generator's own "
                             "Caps.min_interval_h when that's larger (never interpolated).")
    p_gen.add_argument("--analog-date", default=None, dest="analog_date")
    p_gen.add_argument("--dry-run", action="store_true", dest="dry_run")

    p_ins = sub.add_parser("inspect", help="print per-record header info for one intermediate file")
    p_ins.add_argument("file")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.scenario_command == "list":
            return cmd_list(args)
        if args.scenario_command == "generate":
            return cmd_generate(args)
        if args.scenario_command == "inspect":
            return cmd_inspect(args)
    except ScenarioError as exc:
        print("scenario: %s" % exc, file=sys.stderr)
        return 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
