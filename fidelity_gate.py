#!/usr/bin/env python3
"""Fidelity gate for PHYSICS-GRAY WTune recommendations (decision 0050).

PHYSICS-GRAY knobs (mp_physics, bl_pbl_physics, radt, time_step, ...) legitimately
change the computed solution, so bit-identity (`diffwrf`) and the roundoff-tolerance
gate (`wtune_config_gate.py`) cannot judge them. This implements the "chaos-floor"
methodology from [[wrf-within-tolerance-speed-knobs-conus25]]: a candidate is judged
against how far a same-physics, different-decomposition TWIN run drifts from the
reference by pure round-off/decomposition divergence. That divergence is the FLOOR a
candidate cannot be distinguished from if it is truly within tolerance.

  reference   the accepted/baseline run
  twin        same physics as reference, different MPI ranks x OpenMP threads
              (STRICT decomposition change only) -> measures the chaos floor
  candidate   the PHYSICS-GRAY run under test

Each role may be one or more wrfout-style NetCDF files (e.g. one per output time);
records are matched across roles by the `Times` variable, not by file order. Only
valid times present in ALL THREE roles are compared.

Labels (PROVISIONAL thresholds; override with --thresholds, always echoed into the
JSON output so a label is reproducible from the file alone):
  INVALID       no common valid times, a required core field missing, or NaN/Inf
  WITHIN-FLOOR  every key field's floor_ratio <= floor_ratio_max at the last
                common time (floor_ratio = rms_diff(ref,cand) / rms_diff(ref,twin))
  SMALL-DRIFT   not within floor, but at the last common time EVERY key field passes
                nd = rms_diff(ref,cand)/std(ref) <= nd_max OR floor_ratio <=
                floor_ratio_max (a field individually indistinguishable from the
                twin's own chaos floor passes even if its nd alone is high -- a sparse
                hydrometeor field can have a tiny std(ref), so a decomposition-scale
                absolute difference normalizes to a large nd; measured: karina
                nd(QSNOW)=0.117 with floor_ratio=1, conus12 nd(QICE)=0.032), AND
                |bias(T2)| <= bias_t2_max, AND |bias(SWDOWN)| <= bias_swdown_max (if
                SWDOWN present), AND |precip accumulation rel diff| <= precip_rel_max
                for every present accumulated-precip field
  LARGE-DRIFT   otherwise

This tool reports numbers and a provisional label. It does not decide acceptability;
passing WITHIN-FLOOR or SMALL-DRIFT here is not an official WRF ruling (report as
PHYSICS-GRAY / UNCONFIRMED per wtune_compliance.py).

usage:
  python3 fidelity_gate.py --reference REF.nc [REF2.nc ...] \\
      --twin TWIN.nc [...] --candidate CAND.nc [...] \\
      --family {microphysics,cumulus,radiation,pbl_surface,dynamics,all} \\
      [--bdy-width 5] [--thresholds '{"nd_max": 0.05}' | --thresholds path.json] \\
      --out-prefix PREFIX
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import netCDF4
import numpy as np

CORE_FIELDS = ["T2", "Q2", "U10", "V10", "PSFC", "T", "U", "V", "W", "QVAPOR"]

FAMILY_FIELDS = {
    "microphysics": [
        "RAINNC", "SNOWNC", "GRAUPELNC", "QCLOUD", "QRAIN", "QICE", "QSNOW",
        "QGRAUP", "REFL_10CM",
    ],
    "cumulus": ["RAINC"],
    "radiation": ["SWDOWN", "GLW", "OLR", "TSK"],
    "pbl_surface": ["PBLH", "HFX", "LH", "UST", "TSK"],
    "dynamics": [],  # core-only family; W (already in core) is the field of interest
}
FAMILIES = ("microphysics", "cumulus", "radiation", "pbl_surface", "dynamics", "all")

PRECIP_FIELDS = {"RAINNC", "RAINC", "SNOWNC", "GRAUPELNC"}

HORIZ_DIM_PREFIXES = ("south_north", "west_east")

DEFAULT_THRESHOLDS = {
    "floor_ratio_max": 2.0,
    "nd_max": 0.10,
    "bias_t2_max": 0.1,
    "bias_swdown_max": 5.0,
    "precip_rel_max": 0.10,
    "tiny": 1e-9,
}

LABEL_INVALID = "INVALID"
LABEL_WITHIN_FLOOR = "WITHIN-FLOOR"
LABEL_SMALL_DRIFT = "SMALL-DRIFT"
LABEL_LARGE_DRIFT = "LARGE-DRIFT"


def family_field_list(family):
    if family == "all":
        seen, out = set(), []
        for names in FAMILY_FIELDS.values():
            for n in names:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        return out
    if family not in FAMILY_FIELDS:
        raise ValueError("unknown family %r (choose from %s)" % (family, ", ".join(FAMILIES)))
    return list(FAMILY_FIELDS[family])


def requested_fields(family):
    seen, out = set(), []
    for n in CORE_FIELDS + family_field_list(family):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def field_group(name):
    """Human-readable group tag(s) for a field name, for the CSV output."""
    groups = []
    if name in CORE_FIELDS:
        groups.append("core")
    for fam, names in FAMILY_FIELDS.items():
        if name in names:
            groups.append(fam)
    return ",".join(groups) if groups else "?"


def rms(diff):
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diff))))


def trim_boundary(data, dims, bdy_width):
    """Drop `bdy_width` cells from each edge of every horizontal dim (south_north*,
    west_east*, staggered or not). Leaves a dim untouched if it is too small to trim."""
    if bdy_width <= 0:
        return data
    slicer = []
    for axis, dname in enumerate(dims):
        if dname.startswith(HORIZ_DIM_PREFIXES):
            size = data.shape[axis]
            lo, hi = bdy_width, size - bdy_width
            slicer.append(slice(lo, hi) if hi > lo else slice(None))
        else:
            slicer.append(slice(None))
    return data[tuple(slicer)]


class Role:
    """One gate role (reference/twin/candidate), backed by one or more NetCDF files,
    indexed by decoded `Times` string so records are matched by content, not order."""

    def __init__(self, name, paths):
        self.name = name
        self.paths = [str(p) for p in paths]
        self._datasets = {}
        self.time_map = {}  # time_str -> (path, time_index_in_file)
        for p in self.paths:
            ds = self._open(p)
            if "Times" not in ds.variables:
                continue
            for i, t in enumerate(netCDF4.chartostring(ds.variables["Times"][:])):
                t = str(t)
                self.time_map.setdefault(t, (p, i))

    def _open(self, path):
        ds = self._datasets.get(path)
        if ds is None:
            ds = netCDF4.Dataset(path)
            ds.set_auto_mask(False)
            self._datasets[path] = ds
        return ds

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets = {}

    def times(self):
        return set(self.time_map)

    def has_field(self, name, time_str):
        entry = self.time_map.get(time_str)
        if entry is None:
            return False
        path, _ = entry
        var = self._open(path).variables.get(name)
        return var is not None and var.ndim > 0 and np.issubdtype(var.dtype, np.number)

    def read(self, name, time_str):
        path, idx = self.time_map[time_str]
        var = self._open(path).variables[name]
        dims = var.dimensions
        if dims and dims[0] == "Time":
            data = var[idx]
            dims = dims[1:]
        else:
            data = var[:]
        return np.asarray(data, dtype=np.float64), dims


def field_metrics(name, ref_role, twin_role, cand_role, time_str, bdy_width, tiny):
    result = {"name": name, "time": time_str, "group": field_group(name)}
    if not ref_role.has_field(name, time_str):
        result["status"] = "missing"
        result["missing_in"] = ["reference"]
        return result
    missing_in = [
        role.name for role in (twin_role, cand_role) if not role.has_field(name, time_str)
    ]
    if missing_in:
        result["status"] = "missing"
        result["missing_in"] = missing_in
        return result

    ref_data, ref_dims = ref_role.read(name, time_str)
    twin_data, twin_dims = twin_role.read(name, time_str)
    cand_data, cand_dims = cand_role.read(name, time_str)
    ref_t = trim_boundary(ref_data, ref_dims, bdy_width)
    twin_t = trim_boundary(twin_data, twin_dims, bdy_width)
    cand_t = trim_boundary(cand_data, cand_dims, bdy_width)

    if not (ref_t.shape == twin_t.shape == cand_t.shape):
        result["status"] = "shape_mismatch"
        result["shapes"] = {
            "reference": list(ref_t.shape), "twin": list(twin_t.shape), "candidate": list(cand_t.shape),
        }
        return result

    if not (np.all(np.isfinite(ref_t)) and np.all(np.isfinite(twin_t)) and np.all(np.isfinite(cand_t))):
        result["status"] = "nonfinite"
        return result

    rms_diff = rms(cand_t - ref_t)
    floor = rms(twin_t - ref_t)
    floor_zero = floor == 0.0
    floor_ratio = rms_diff / max(floor, tiny)
    std_ref = float(np.std(ref_t))
    nd = rms_diff / max(std_ref, tiny)
    bias = float(np.mean(cand_t - ref_t))

    result.update(
        status="ok",
        rms_diff=rms_diff,
        floor=floor,
        floor_zero=floor_zero,
        floor_ratio=floor_ratio,
        std_ref=std_ref,
        nd=nd,
        bias=bias,
    )
    if name in PRECIP_FIELDS:
        mean_ref = float(np.mean(ref_t))
        mean_cand = float(np.mean(cand_t))
        result["precip_rel_diff"] = (mean_cand - mean_ref) / max(abs(mean_ref), tiny)
    return result


def _fmt(name, value):
    return "%s=%.6g" % (name, value)


def decide_label(rows_at_last_time, thresholds):
    """rows_at_last_time: dict field_name -> metrics dict (status=='ok' filtered by caller
    for the 'key fields' used in the decision; missing/non-ok fields are reported but not
    fatal here -- INVALID for a missing CORE field is decided by the caller before this)."""
    if not rows_at_last_time:
        return LABEL_INVALID, "no key fields available for comparison"

    worst_floor = max(rows_at_last_time.items(), key=lambda kv: kv[1]["floor_ratio"])
    within_floor = worst_floor[1]["floor_ratio"] <= thresholds["floor_ratio_max"]
    if within_floor:
        return LABEL_WITHIN_FLOOR, "max floor_ratio %s <= %.6g" % (
            _fmt(worst_floor[0], worst_floor[1]["floor_ratio"]), thresholds["floor_ratio_max"],
        )

    reasons = []
    ok = True

    # Per-field drift test (2026-09-15 fix): a field passes if nd <= nd_max OR its OWN
    # floor_ratio <= floor_ratio_max -- i.e. it is individually indistinguishable from the
    # twin's chaos floor even though its nd alone is high. The OLD rule (worst nd over ALL
    # key fields <= nd_max) labelled that as drift: a sparse hydrometeor field can have a
    # tiny std(ref), so a decomposition-scale absolute difference (the same size as the
    # twin's own floor) normalizes to a large nd purely from the small denominator.
    # Measured on real output: karina nd(QSNOW)=0.117 (> nd_max 0.10) with floor_ratio=1;
    # conus12 nd(QICE)=0.032 after 30 simulated minutes.
    failing = [
        (name, m) for name, m in rows_at_last_time.items()
        if m["nd"] > thresholds["nd_max"] and m["floor_ratio"] > thresholds["floor_ratio_max"]
    ]
    if failing:
        ok = False
        worst_name, worst_m = max(failing, key=lambda kv: kv[1]["nd"])
        reasons.append(
            "nd(%s)=%.6g > %.6g and floor_ratio(%s)=%.6g > %.6g" % (
                worst_name, worst_m["nd"], thresholds["nd_max"],
                worst_name, worst_m["floor_ratio"], thresholds["floor_ratio_max"],
            )
        )

    if "T2" in rows_at_last_time:
        b = abs(rows_at_last_time["T2"]["bias"])
        if b > thresholds["bias_t2_max"]:
            ok = False
            reasons.append("|bias(T2)|=%.6g > %.6g" % (b, thresholds["bias_t2_max"]))

    if "SWDOWN" in rows_at_last_time:
        b = abs(rows_at_last_time["SWDOWN"]["bias"])
        if b > thresholds["bias_swdown_max"]:
            ok = False
            reasons.append("|bias(SWDOWN)|=%.6g > %.6g" % (b, thresholds["bias_swdown_max"]))

    for field in sorted(PRECIP_FIELDS):
        m = rows_at_last_time.get(field)
        if m is None or "precip_rel_diff" not in m:
            continue
        r = abs(m["precip_rel_diff"])
        if r > thresholds["precip_rel_max"]:
            ok = False
            reasons.append("|precip_rel_diff(%s)|=%.6g > %.6g" % (field, r, thresholds["precip_rel_max"]))

    if ok:
        worst_nd = max(rows_at_last_time.items(), key=lambda kv: kv[1]["nd"])
        return LABEL_SMALL_DRIFT, "not within floor (%s); all SMALL-DRIFT checks passed, max nd %s" % (
            _fmt(worst_floor[0], worst_floor[1]["floor_ratio"]), _fmt(worst_nd[0], worst_nd[1]["nd"]),
        )
    return LABEL_LARGE_DRIFT, "; ".join(reasons)


def run_gate(reference, twin, candidate, family, bdy_width, thresholds, out_prefix):
    thr = dict(DEFAULT_THRESHOLDS)
    thr.update(thresholds or {})

    ref_role = Role("reference", reference)
    twin_role = Role("twin", twin)
    cand_role = Role("candidate", candidate)
    try:
        common_times = sorted(ref_role.times() & twin_role.times() & cand_role.times())
        fields = requested_fields(family)

        payload = {
            "label": None,
            "deciding": None,
            "family": family,
            "bdy_width": bdy_width,
            "thresholds": thr,
            "files": {"reference": ref_role.paths, "twin": twin_role.paths, "candidate": cand_role.paths},
            "fields_requested": fields,
            "common_times": common_times,
        }

        if not common_times:
            payload["label"] = LABEL_INVALID
            payload["deciding"] = "no common valid times across reference/twin/candidate"
            _write_outputs(out_prefix, payload, [])
            return payload

        last_time = common_times[-1]
        payload["last_time"] = last_time

        all_rows = []
        for t in common_times:
            for f in fields:
                all_rows.append(field_metrics(f, ref_role, twin_role, cand_role, t, bdy_width, thr["tiny"]))

        nonfinite_rows = [r for r in all_rows if r["status"] == "nonfinite"]
        if nonfinite_rows:
            bad = nonfinite_rows[0]
            payload["label"] = LABEL_INVALID
            payload["deciding"] = "NaN/Inf in field %s at time %s" % (bad["name"], bad["time"])
            _write_outputs(out_prefix, payload, all_rows)
            return payload

        last_time_rows = {r["name"]: r for r in all_rows if r["time"] == last_time}
        missing_summary = {
            r["name"]: r["missing_in"] for r in all_rows if r["status"] == "missing"
        }
        payload["fields_missing"] = missing_summary

        bad_core = [
            name for name in CORE_FIELDS
            if last_time_rows.get(name, {}).get("status") != "ok"
        ]
        if bad_core:
            name = bad_core[0]
            status = last_time_rows.get(name, {}).get("status", "missing")
            payload["label"] = LABEL_INVALID
            payload["deciding"] = "required core field %s is %s at last common time %s" % (
                name, status, last_time,
            )
            _write_outputs(out_prefix, payload, all_rows)
            return payload

        key_fields = {
            name: m for name, m in last_time_rows.items() if m["status"] == "ok"
        }
        label, deciding = decide_label(key_fields, thr)
        payload["label"] = label
        payload["deciding"] = deciding
        payload["per_field_summary_at_last_time"] = {
            name: _json_safe(m) for name, m in last_time_rows.items()
        }
        _write_outputs(out_prefix, payload, all_rows)
        return payload
    finally:
        ref_role.close()
        twin_role.close()
        cand_role.close()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


CSV_COLUMNS = [
    "time", "field", "group", "status", "missing_in", "shapes",
    "rms_diff", "floor", "floor_zero", "floor_ratio", "std_ref", "nd", "bias", "precip_rel_diff",
]


def _write_outputs(out_prefix, payload, all_rows):
    json_path = Path(str(out_prefix) + ".json")
    csv_path = Path(str(out_prefix) + ".csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
        fh.write("\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for row in all_rows:
            writer.writerow([
                row.get("time", ""),
                row.get("name", ""),
                row.get("group", ""),
                row.get("status", ""),
                "|".join(row.get("missing_in", [])) if row.get("missing_in") else "",
                json.dumps(row["shapes"]) if "shapes" in row else "",
                _csv_num(row.get("rms_diff")),
                _csv_num(row.get("floor")),
                row.get("floor_zero", ""),
                _csv_num(row.get("floor_ratio")),
                _csv_num(row.get("std_ref")),
                _csv_num(row.get("nd")),
                _csv_num(row.get("bias")),
                _csv_num(row.get("precip_rel_diff")),
            ])


def _csv_num(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _parse_thresholds(raw):
    if raw is None:
        return {}
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    with open(raw, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parser():
    p = argparse.ArgumentParser(
        description="Fidelity gate for PHYSICS-GRAY WTune recommendations (chaos-floor method)",
    )
    p.add_argument("--reference", nargs="+", required=True, help="reference wrfout NetCDF file(s)")
    p.add_argument("--twin", nargs="+", required=True,
                   help="same-physics different-decomposition twin wrfout file(s) (measures the chaos floor)")
    p.add_argument("--candidate", nargs="+", required=True, help="candidate wrfout NetCDF file(s)")
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--bdy-width", type=int, default=5,
                   help="cells excluded from each horizontal edge (lateral boundary relaxation zone)")
    p.add_argument("--thresholds", help="JSON string or path to a JSON file overriding DEFAULT_THRESHOLDS")
    p.add_argument("--out-prefix", required=True, help="writes PREFIX.json and PREFIX.csv")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.bdy_width < 0:
        _parser().error("--bdy-width must be non-negative")
    try:
        thresholds = _parse_thresholds(args.thresholds)
    except (OSError, json.JSONDecodeError) as error:
        print("ERROR: --thresholds: %s" % error, file=sys.stderr)
        return 2
    try:
        payload = run_gate(
            args.reference, args.twin, args.candidate, args.family,
            args.bdy_width, thresholds, args.out_prefix,
        )
    except (OSError, RuntimeError, ValueError, IndexError, KeyError, TypeError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 2
    print("FIDELITY-GATE %s deciding=%r out=%s.json,%s.csv" % (
        payload["label"], payload["deciding"], args.out_prefix, args.out_prefix,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
