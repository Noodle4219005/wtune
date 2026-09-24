#!/usr/bin/env python3
"""Calibration gate for result-affecting config-layer knobs (radt, non_hydrostatic, cudt...).

Bit-identity (diffwrf) cannot judge a knob that legitimately changes the answer. This
compares a candidate wrfout against a reference field by field (max|d|, RMS of the
difference, RMS relative to the field) and passes when every field is within a tolerance
that was CALIBRATED from a pair the user already accepts (`calibrate` mode prints the
per-field table and a suggested --tol-rel = max rel_rms x safety).

One variable is read at a time (wrfout files are ~800 MB); last time record by default.
"""

import argparse
import json
import math
import sys

import netCDF4
import numpy as np


def _names(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    return {item.strip() for item in value if item.strip()}


def _selected_shape(var, time_index):
    if var.dimensions and var.dimensions[0] == "Time":
        size = var.shape[0]
        index = time_index if time_index >= 0 else size + time_index
        if index < 0 or index >= size:
            raise IndexError(
                f"time index {time_index} is out of range for {size} records"
            )
        return var.shape[1:]
    return var.shape


def _read_value(var, time_index):
    if var.dimensions and var.dimensions[0] == "Time":
        size = var.shape[0]
        index = time_index if time_index >= 0 else size + time_index
        if index < 0 or index >= size:
            raise IndexError(
                f"time index {time_index} is out of range for {size} records"
            )
        key = (index,) + (slice(None),) * (var.ndim - 1)
        value = var[key]
    else:
        value = var[:]
    return np.asarray(value, dtype=np.float64)


def _structural_result(name, n, reason):
    return {
        "name": name,
        "n": int(n),
        "n_diff": 0,
        "max_abs": math.inf,
        "rms_diff": math.inf,
        "rms_ref": math.inf,
        "rel_rms": math.inf,
        "status": "fail",
        "reason": reason,
    }


def _metrics(name, ref, cand):
    n = int(ref.size)
    different = ref != cand
    n_diff = int(np.count_nonzero(different))
    nonfinite = bool(np.any(~np.isfinite(ref)) or np.any(~np.isfinite(cand)))
    if nonfinite:
        rms_ref = math.inf if np.any(~np.isfinite(ref)) else float(
            np.sqrt(np.mean(np.square(ref)))
        )
        return {
            "name": name,
            "n": n,
            "n_diff": n_diff,
            "max_abs": math.inf,
            "rms_diff": math.inf,
            "rms_ref": rms_ref,
            "rel_rms": math.inf,
            "status": "fail",
            "reason": "nonfinite",
        }

    if n:
        difference = cand - ref
        max_abs = float(np.max(np.abs(difference)))
        rms_diff = float(np.sqrt(np.mean(np.square(difference))))
        rms_ref = float(np.sqrt(np.mean(np.square(ref))))
    else:
        max_abs = 0.0
        rms_diff = 0.0
        rms_ref = 0.0
    if rms_ref > 0.0:
        rel_rms = rms_diff / rms_ref
    else:
        rel_rms = 0.0 if rms_diff == 0.0 else math.inf
    return {
        "name": name,
        "n": n,
        "n_diff": n_diff,
        "max_abs": max_abs,
        "rms_diff": rms_diff,
        "rms_ref": rms_ref,
        "rel_rms": rel_rms,
        "status": "pass",
        "reason": "",
    }


def compare(ref_path, cand_path, fields=None, skip=None, time_index=-1):
    selected = _names(fields)
    skipped = _names(skip) or set()
    results = []
    with netCDF4.Dataset(ref_path) as ref_file, netCDF4.Dataset(cand_path) as cand_file:
        ref_file.set_auto_mask(False)
        cand_file.set_auto_mask(False)
        for name, ref_var in ref_file.variables.items():
            if selected is not None and name not in selected:
                continue
            if name in skipped or ref_var.ndim == 0:
                continue
            if not np.issubdtype(ref_var.dtype, np.number):
                continue

            ref_shape = _selected_shape(ref_var, time_index)
            n = int(np.prod(ref_shape, dtype=np.int64))
            cand_var = cand_file.variables.get(name)
            if cand_var is None:
                results.append(_structural_result(name, n, "missing"))
                continue
            if cand_var.ndim == 0 or not np.issubdtype(cand_var.dtype, np.number):
                results.append(_structural_result(name, n, "missing"))
                continue
            cand_shape = _selected_shape(cand_var, time_index)
            if ref_shape != cand_shape:
                results.append(_structural_result(name, n, "shape"))
                continue

            ref_value = _read_value(ref_var, time_index)
            cand_value = _read_value(cand_var, time_index)
            results.append(_metrics(name, ref_value, cand_value))
    return results


def _format_float(value):
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _print_table(results, top):
    ordered = sorted(results, key=lambda item: item["rel_rms"], reverse=True)
    shown = ordered if top == 0 else ordered[:top]
    print("field n n_diff max_abs rms_diff rms_ref rel_rms status reason")
    for item in shown:
        print(
            f"{item['name']} {item['n']} {item['n_diff']} "
            f"{_format_float(item['max_abs'])} "
            f"{_format_float(item['rms_diff'])} "
            f"{_format_float(item['rms_ref'])} "
            f"{_format_float(item['rel_rms'])} "
            f"{item['status']} {item['reason']}"
        )


def _check_results(results, tol_rel, tol_abs):
    failed = []
    for item in results:
        if item["reason"] in {"missing", "shape", "nonfinite"}:
            item["status"] = "fail"
        elif item["rel_rms"] <= tol_rel or item["max_abs"] <= tol_abs:
            item["status"] = "pass"
            item["reason"] = ""
        else:
            item["status"] = "fail"
            item["reason"] = "tolerance"
        if item["status"] == "fail":
            failed.append(item["name"])
    return failed


def _json_value(value):
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "inf"
        if math.isnan(float(value)):
            return "nan"
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def _payload(mode, args, results, failed, tol_rel, tol_abs, verdict):
    fields = []
    for item in results:
        fields.append({key: _json_value(value) for key, value in item.items()})
    return {
        "mode": mode,
        "ref": args.ref,
        "cand": args.cand,
        "time_index": args.time_index,
        "tol_rel": _json_value(tol_rel),
        "tol_abs": _json_value(tol_abs),
        "fields": fields,
        "failed": failed,
        "verdict": verdict,
    }


def _run_check(args):
    results = compare(
        args.ref,
        args.cand,
        fields=args.fields,
        skip=args.skip_fields,
        time_index=args.time_index,
    )
    failed = _check_results(results, args.tol_rel, args.tol_abs)
    _print_table(results, args.top)
    if failed:
        print(
            f"GATE FAIL fields={len(results)} failed={len(failed)} "
            f"tol_rel={args.tol_rel} tol_abs={args.tol_abs} "
            f"failed_fields={','.join(failed)}"
        )
        verdict = "FAIL"
        exit_code = 1
    else:
        print(
            f"GATE PASS fields={len(results)} failed=0 "
            f"tol_rel={args.tol_rel} tol_abs={args.tol_abs}"
        )
        verdict = "PASS"
        exit_code = 0
    if args.json:
        _write_json(
            args.json,
            _payload(
                "check",
                args,
                results,
                failed,
                args.tol_rel,
                args.tol_abs,
                verdict,
            ),
        )
    return exit_code


def _run_calibrate(args):
    results = compare(
        args.ref,
        args.cand,
        fields=args.fields,
        skip=args.skip_fields,
        time_index=args.time_index,
    )
    max_rel = max((item["rel_rms"] for item in results), default=0.0)
    suggested = max_rel * args.safety
    _print_table(results, args.top)
    print(f"suggested --tol-rel {_format_float(suggested)}")
    print("tolerance-setting fields")
    for item in sorted(results, key=lambda row: row["rel_rms"], reverse=True)[:5]:
        print(f"{item['name']} {_format_float(item['rel_rms'])}")
    if args.json:
        _write_json(
            args.json,
            _payload(
                "calibrate",
                args,
                results,
                [],
                suggested,
                0.0,
                "PASS",
            ),
        )
    return 0


def _parser():
    parser = argparse.ArgumentParser(description="Calibrate and check WRF config-layer output")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--ref", required=True)
    check.add_argument("--cand", required=True)
    check.add_argument("--tol-rel", type=float, default=1e-6)
    check.add_argument("--tol-abs", type=float, default=0.0)
    check.add_argument("--fields")
    check.add_argument("--skip-fields")
    check.add_argument("--time-index", type=int, default=-1)
    check.add_argument("--json")
    check.add_argument("--top", type=int, default=20)
    check.set_defaults(handler=_run_check)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--ref", required=True)
    calibrate.add_argument("--cand", required=True)
    calibrate.add_argument("--fields")
    calibrate.add_argument("--skip-fields")
    calibrate.add_argument("--time-index", type=int, default=-1)
    calibrate.add_argument("--safety", type=float, default=3.0)
    calibrate.add_argument("--json")
    calibrate.add_argument("--top", type=int, default=20)
    calibrate.set_defaults(handler=_run_calibrate)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.mode == "check" and (args.tol_rel < 0.0 or args.tol_abs < 0.0):
        parser.error("tolerances must be non-negative")
    if args.mode == "calibrate" and args.safety < 0.0:
        parser.error("--safety must be non-negative")
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError, IndexError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
