"""scenario.cli -- `wtune scenario ...` subcommands: list, generate, inspect. Wired into
src/wtune_cli.py via a single 'scenario' subparser that forwards its remainder args to main()
(same REMAINDER-forwarding pattern wtune_cli.py already uses for 'metdata'/'analyze')."""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from . import wps_intermediate as wi
from .base import ScenarioError, ScenarioRequest, get_generator, list_generators, valid_times
from .config import parse_scenario
from .fill import compute_missing
from .generators.analog import select_analog_date


def _parse_bbox(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ScenarioError("--bbox needs 4 comma-separated values: lat_min,lat_max,lon_min,lon_max")
    return tuple(parts)


def _parse_iso(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s)


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

    req = ScenarioRequest(bbox=bbox, start=start, end=end, out_dir=args.out,
                            case_dir=args.case_dir, analog_date=analog_date)

    if args.dry_run:
        analog_date_chosen = req.analog_date or select_analog_date(req.start)
        gen = get_generator(args.generator, allow_noncommercial=args.allow_noncommercial)
        caps = gen.capabilities()
        missing, fill_rules = compute_missing(caps)
        times = valid_times(req)
        print("generator: %s (license: %s)" % (args.generator, caps.license))
        print("analog date: %s" % analog_date_chosen.isoformat())
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

    gen = get_generator(args.generator, allow_noncommercial=args.allow_noncommercial)
    forcing = gen.generate(req)
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
