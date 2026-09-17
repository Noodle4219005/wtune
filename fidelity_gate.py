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
records are matched across roles by the `Times` variable, not by file order or an exact
string match -- roles are matched within --time-tol-s seconds of each other (default 180),
since a candidate that changes time_step (e.g. PHYSICS-GRAY template P03) drifts its
history-write times away from the reference's exact seconds. See match_times().

Labels (PROVISIONAL thresholds; override with --thresholds, always echoed into the
JSON output so a label is reproducible from the file alone):
  INVALID       no time-matched triple at all ("no common valid times..."), no
                time-matched triple strictly after the reference's first (initial) valid
                time ("no matched valid time after initialization" -- prevents a
                time_step-changing candidate whose later outputs all drift outside
                tolerance from being silently compared only at t=0, which reads as "no
                change"), a required core field missing, or NaN/Inf
  WITHIN-FLOOR  every key field is within its own chaos floor at the last matched
                post-init time: floor_ratio <= floor_ratio_max (floor_ratio =
                rms_diff(ref,cand) / rms_diff(ref,twin)), OR -- when the twin is
                bit-identical to the reference for that field (floor == 0, "floor_zero")
                -- rms_diff(ref,cand) == 0 too (an undefined/infinite ratio is never
                treated as passing)
  SMALL-DRIFT   not within floor, but at the last matched post-init time EVERY key field
                passes nd = rms_diff(ref,cand)/std(ref) <= nd_max OR is within its own
                chaos floor as above (a field individually indistinguishable from the
                twin's own chaos floor passes even if its nd alone is high -- a sparse
                hydrometeor field can have a tiny std(ref), so a decomposition-scale
                absolute difference normalizes to a large nd; measured: karina
                nd(QSNOW)=0.117 with floor_ratio=1, conus12 nd(QICE)=0.032). nd itself is
                left undefined ("undefined_low_variance") when std(ref) is below the
                field's min_std (see min_std_threshold()) -- a reference field with ~zero
                spatial variance (e.g. SWDOWN at night) would otherwise normalize a small
                absolute diff into a huge nd (measured: maria1km nd(SWDOWN)=1.4e6); such
                a field is then judged only by floor_ratio (when the floor is non-zero)
                and the checks below. All of the above AND |bias(T2)| <= bias_t2_max, AND
                |bias(SWDOWN)| <= bias_swdown_max (if SWDOWN present), AND |precip
                accumulation rel diff| <= precip_rel_max for every present
                accumulated-precip field
  LARGE-DRIFT   otherwise

This tool reports numbers and a provisional label. It does not decide acceptability;
passing WITHIN-FLOOR or SMALL-DRIFT here is not an official WRF ruling (report as
PHYSICS-GRAY / UNCONFIRMED per wtune_compliance.py).

usage:
  python3 fidelity_gate.py --reference REF.nc [REF2.nc ...] \\
      --twin TWIN.nc [...] --candidate CAND.nc [...] \\
      --family {microphysics,cumulus,radiation,pbl_surface,dynamics,diagnostics,nesting,all} \\
      [--bdy-width 5] [--time-tol-s 180] \\
      [--thresholds '{"nd_max": 0.05}' | --thresholds path.json] \\
      --out-prefix PREFIX
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
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
    # Output-only diagnostics (nwp_diagnostics/hailcast_opt/do_radar_ref, template D04): they are
    # never fed back into the prognostic equations, so turning them off must leave the CORE fields
    # untouched -- that is exactly the claim to test, and it is a core-only comparison. Added
    # 2026-09-17 after jobs 1081237/1081238 ran clean and then lost their verdict here:
    # "--family: invalid choice: 'diagnostics'".
    "diagnostics": [],
    # Nesting changes (feedback=0, template D06) alter how the nest writes back onto its parent;
    # the visible effect is on the parent's own core state. Same missing-choice failure
    # (zhengzhou job 1080703 reported got_fidelity=2 of 3).
    "nesting": [],
}
FAMILIES = ("microphysics", "cumulus", "radiation", "pbl_surface", "dynamics", "diagnostics",
            "nesting", "all")

PRECIP_FIELDS = {"RAINNC", "RAINC", "SNOWNC", "GRAUPELNC"}

# Field groups with a physically sensible minimum reference spatial std, below which a
# relative metric (nd) is not meaningful (2026-09-15 fix, see min_std_threshold()).
RADIATION_FLUX_FIELDS = {"SWDOWN", "GLW", "OLR"}
HYDROMETEOR_FIELDS = {"QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"}

HORIZ_DIM_PREFIXES = ("south_north", "west_east")

WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
EPOCH = datetime(1970, 1, 1)

DEFAULT_THRESHOLDS = {
    "floor_ratio_max": 2.0,
    "nd_max": 0.10,
    "bias_t2_max": 0.1,
    "bias_swdown_max": 5.0,
    "precip_rel_max": 0.10,
    "tiny": 1e-9,
    # Matching-tolerance for comparing roles whose output times do not coincide exactly
    # (e.g. a candidate time_step change shifts history-write times away from the
    # reference's). See --time-tol-s / match_times().
    "time_tol_s": 180.0,
    # Per-field-group minimum reference spatial std (std_ref) below which nd is left
    # undefined ("undefined_low_variance") rather than computed -- see min_std_threshold().
    "min_std_by_group": {
        "radiation_flux": 1.0,      # W/m^2
        "precip_accum": 0.01,       # mm (accumulated)
        "hydrometeor": 1e-6,        # kg/kg
    },
    "min_std_relative": 1e-6,       # fallback for ungrouped fields: relative to |mean_ref|
    "min_std_absolute_floor": 1e-6, # ...with this absolute floor
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


def field_variance_group(name):
    """Which min-std group (see DEFAULT_THRESHOLDS['min_std_by_group']) a field falls in,
    or None for the generic relative-to-mean fallback."""
    if name in RADIATION_FLUX_FIELDS:
        return "radiation_flux"
    if name in PRECIP_FIELDS:
        return "precip_accum"
    if name in HYDROMETEOR_FIELDS:
        return "hydrometeor"
    return None


def min_std_threshold(name, mean_ref, thresholds):
    """Minimum reference spatial std below which nd(name) is left undefined -- see
    field_metrics(). Per-group defaults (radiation fluxes, precip accumulations,
    hydrometeor mixing ratios) come from thresholds['min_std_by_group']; every other
    field falls back to a relative fraction of |mean_ref| with an absolute floor, so a
    field that is legitimately near-zero everywhere (e.g. SWDOWN at night) does not get a
    meaningless nd from dividing by an almost-zero std."""
    group = field_variance_group(name)
    by_group = thresholds.get("min_std_by_group", {})
    if group is not None and group in by_group:
        return by_group[group]
    return max(thresholds["min_std_relative"] * abs(mean_ref), thresholds["min_std_absolute_floor"])


def _parse_wrf_time(t):
    return datetime.strptime(t, WRF_TIME_FORMAT)


def _epoch_seconds(t):
    return (_parse_wrf_time(t) - EPOCH).total_seconds()


def match_times(ref_role, twin_role, cand_role, tol_s):
    """Match each reference valid time to the nearest twin and candidate valid time within
    tol_s seconds (2026-09-15 fix). Exact-string time matching silently degenerated to
    "compare only t=0" whenever a candidate changed time_step: a different dt shifts
    history-write times away from the reference's, so no time after the first coincided
    exactly (measured: karina/isc21 P03, a 1.2x time_step multiplier, matched only at
    t=0 and were labelled WITHIN-FLOOR with nd=0/floor_ratio=0 -- a false "no change").

    Returns (matches, first_ref_time) where matches is a list of dicts, one per reference
    valid time that has a within-tolerance twin AND candidate match, ordered the same as
    sorted(ref_role.times()):
      {"reference": t_ref, "twin": t_twin, "candidate": t_cand,
       "twin_offset_s": |t_twin - t_ref|, "candidate_offset_s": |t_cand - t_ref|}
    A reference time with no such twin+candidate pair within tolerance is simply absent
    from `matches` (it does not make the whole comparison INVALID by itself -- only "no
    matches at all" or "no matches after the first/init time" does, in run_gate()).
    """
    ref_times = sorted(ref_role.times())
    if not ref_times:
        return [], None
    first_ref_time = ref_times[0]

    def _epochs(role):
        out = []
        for t in role.times():
            try:
                out.append((t, _epoch_seconds(t)))
            except ValueError:
                continue  # not a parseable WRF Times string; cannot be time-matched
        return out

    twin_epochs = _epochs(twin_role)
    cand_epochs = _epochs(cand_role)

    def _nearest(target_epoch, epochs):
        best_t, best_diff = None, None
        for t, e in epochs:
            diff = abs(e - target_epoch)
            if diff <= tol_s and (best_diff is None or diff < best_diff):
                best_t, best_diff = t, diff
        return best_t, best_diff

    matches = []
    for t_ref in ref_times:
        try:
            e_ref = _epoch_seconds(t_ref)
        except ValueError:
            continue
        t_twin, d_twin = _nearest(e_ref, twin_epochs)
        t_cand, d_cand = _nearest(e_ref, cand_epochs)
        if t_twin is not None and t_cand is not None:
            matches.append({
                "reference": t_ref, "twin": t_twin, "candidate": t_cand,
                "twin_offset_s": d_twin, "candidate_offset_s": d_cand,
            })
    return matches, first_ref_time


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


def field_metrics(name, ref_role, twin_role, cand_role, time_match, bdy_width, thresholds):
    """time_match: one entry from match_times() -- {"reference", "twin", "candidate",
    "twin_offset_s", "candidate_offset_s"} -- since roles are no longer guaranteed to share
    an identical time string (see match_times())."""
    t_ref = time_match["reference"]
    t_twin = time_match["twin"]
    t_cand = time_match["candidate"]
    tiny = thresholds["tiny"]
    result = {
        "name": name, "time": t_ref, "group": field_group(name),
        "twin_offset_s": time_match.get("twin_offset_s"),
        "candidate_offset_s": time_match.get("candidate_offset_s"),
    }
    if not ref_role.has_field(name, t_ref):
        result["status"] = "missing"
        result["missing_in"] = ["reference"]
        return result
    missing_in = []
    if not twin_role.has_field(name, t_twin):
        missing_in.append("twin")
    if not cand_role.has_field(name, t_cand):
        missing_in.append("candidate")
    if missing_in:
        result["status"] = "missing"
        result["missing_in"] = missing_in
        return result

    ref_data, ref_dims = ref_role.read(name, t_ref)
    twin_data, twin_dims = twin_role.read(name, t_twin)
    cand_data, cand_dims = cand_role.read(name, t_cand)
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
    # floor_zero (2026-09-15 fix): the twin is bit-identical to the reference for this
    # field -- floor_ratio = rms_diff / floor would otherwise divide by ~0 (the old
    # tiny-fallback) and print an astronomically large, uninformative number (measured:
    # maria1km/lafe twins bit-identical to reference -> floor_ratio 1e9-1e10 for every
    # field). floor_ratio is left undefined (null) instead; the decision falls back to
    # nd/bias/precip, and WITHIN-FLOOR for this field requires rms_diff == 0 exactly.
    floor_zero = floor == 0.0
    floor_ratio = None if floor_zero else (rms_diff / floor)

    mean_ref = float(np.mean(ref_t))
    std_ref = float(np.std(ref_t))
    # undefined_low_variance (2026-09-15 fix): nd = rms_diff / std_ref explodes when the
    # reference field has ~zero spatial variance at this time (a genuinely small absolute
    # difference normalizes to a huge relative one). Measured: maria1km SWDOWN at night
    # nd=1.4e6; karina QGRAUP nd=464; conus25 SNOWNC nd=1959. Below the field's min_std,
    # nd is left undefined and the field is judged only by floor_ratio (when the floor is
    # non-zero) and the bias/precip criteria.
    min_std = min_std_threshold(name, mean_ref, thresholds)
    low_variance = std_ref < min_std
    nd = None if low_variance else (rms_diff / max(std_ref, tiny))
    bias = float(np.mean(cand_t - ref_t))

    result.update(
        status="ok",
        rms_diff=rms_diff,
        floor=floor,
        floor_zero=floor_zero,
        floor_ratio=floor_ratio,
        std_ref=std_ref,
        min_std=min_std,
        low_variance=low_variance,
        nd=nd,
        nd_status="undefined_low_variance" if low_variance else "ok",
        bias=bias,
    )
    if name in PRECIP_FIELDS:
        mean_cand = float(np.mean(cand_t))
        result["precip_rel_diff"] = (mean_cand - mean_ref) / max(abs(mean_ref), tiny)
    return result


def _fmt(name, value):
    return "%s=%.6g" % (name, value)


def _floor_ok(m, thresholds):
    """Whether a field is individually within the chaos floor. floor_zero fields (twin
    bit-identical to reference) have no floor_ratio to compare against thresholds; the
    only floor-consistent outcome for them is the candidate ALSO being bit-identical."""
    if m.get("floor_zero"):
        return m["rms_diff"] == 0.0
    return m["floor_ratio"] <= thresholds["floor_ratio_max"]


def _floor_sort_key(m):
    """Order fields by how far they are from being within-floor, for picking the 'worst'
    field to report. A failing floor_zero field (nonzero diff against an exact-zero floor)
    ranks worse than any finite floor_ratio; a passing one ranks as the best."""
    if m.get("floor_zero"):
        return -1.0 if m["rms_diff"] == 0.0 else float("inf")
    return m["floor_ratio"]


def _field_drift_ok(m, thresholds):
    """Per-field pass test for the SMALL-DRIFT/LARGE-DRIFT decision (WITHIN-FLOOR is
    decided separately, and more strictly, by _floor_ok). A field passes if:
      - nd is defined and nd <= nd_max ("small enough drift relative to the field's own
        spatial variance"), OR
      - the floor is non-zero and floor_ratio <= floor_ratio_max ("indistinguishable from
        the twin's own chaos floor"), OR
      - the floor is zero (floor_zero) and rms_diff == 0 (bit-identical to reference), OR
      - nd is undefined (std_ref below the field's min_std -- "undefined_low_variance")
        AND the floor is also zero: neither ratio is meaningful for such a field (its
        reference values are ~0 everywhere AND its twin never diverges from the reference
        at all), so a nonzero-but-below-min_std candidate diff is left entirely to the
        bias/precip checks rather than being forced to LARGE-DRIFT by this per-field rule
        (2026-09-15 fix).
    """
    nd = m.get("nd")
    if nd is not None and nd <= thresholds["nd_max"]:
        return True
    if m.get("floor_zero"):
        if m["rms_diff"] == 0.0:
            return True
        return nd is None
    return m["floor_ratio"] <= thresholds["floor_ratio_max"]


def decide_label(rows_at_last_time, thresholds):
    """rows_at_last_time: dict field_name -> metrics dict (status=='ok' filtered by caller
    for the 'key fields' used in the decision; missing/non-ok fields are reported but not
    fatal here -- INVALID for a missing CORE field is decided by the caller before this)."""
    if not rows_at_last_time:
        return LABEL_INVALID, "no key fields available for comparison"

    all_within = all(_floor_ok(m, thresholds) for m in rows_at_last_time.values())
    worst_name, worst_m = max(rows_at_last_time.items(), key=lambda kv: _floor_sort_key(kv[1]))

    if all_within:
        if worst_m.get("floor_zero"):
            deciding = "all key fields within floor (worst %s floor_zero, rms_diff=%.6g)" % (
                worst_name, worst_m["rms_diff"],
            )
        else:
            deciding = "max floor_ratio %s <= %.6g" % (
                _fmt(worst_name, worst_m["floor_ratio"]), thresholds["floor_ratio_max"],
            )
        return LABEL_WITHIN_FLOOR, deciding

    reasons = []
    ok = True

    # Per-field drift test (2026-09-15 fix): see _field_drift_ok. The OLD rule (worst nd
    # over ALL key fields <= nd_max) labelled a field as drift whenever its nd alone was
    # high: a sparse hydrometeor field can have a tiny std(ref), so a decomposition-scale
    # absolute difference (the same size as the twin's own floor) normalizes to a large nd
    # purely from the small denominator. Measured on real output: karina nd(QSNOW)=0.117
    # (> nd_max 0.10) with floor_ratio=1; conus12 nd(QICE)=0.032 after 30 simulated
    # minutes. An undefined nd (undefined_low_variance -- reference std below the field's
    # min_std) never fails a field by itself: it is judged only by floor_ratio when the
    # floor is non-zero, or left entirely to bias/precip when the floor is also zero.
    failing = [(name, m) for name, m in rows_at_last_time.items() if not _field_drift_ok(m, thresholds)]
    if failing:
        ok = False
        worst_fail_name, worst_fail_m = max(
            failing, key=lambda kv: kv[1]["nd"] if kv[1].get("nd") is not None else 0.0,
        )
        if worst_fail_m.get("nd") is None:
            # _field_drift_ok only lets a low-variance field into `failing` when its floor
            # is non-zero (a floor_zero low-variance field always passes, deferred to
            # bias/precip), so floor_ratio is always defined here.
            reasons.append(
                "%s: nd undefined_low_variance (std_ref=%.6g < min_std=%.6g) and floor_ratio=%.6g > %.6g" % (
                    worst_fail_name, worst_fail_m["std_ref"], worst_fail_m["min_std"],
                    worst_fail_m["floor_ratio"], thresholds["floor_ratio_max"],
                )
            )
        else:
            floor_txt = "floor_zero" if worst_fail_m.get("floor_zero") else "%.6g" % worst_fail_m["floor_ratio"]
            reasons.append(
                "nd(%s)=%.6g > %.6g and floor_ratio(%s)=%s > %.6g" % (
                    worst_fail_name, worst_fail_m["nd"], thresholds["nd_max"],
                    worst_fail_name, floor_txt, thresholds["floor_ratio_max"],
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
        nd_candidates = [(n, m) for n, m in rows_at_last_time.items() if m.get("nd") is not None]
        if nd_candidates:
            worst_nd_name, worst_nd_m = max(nd_candidates, key=lambda kv: kv[1]["nd"])
            nd_txt = _fmt(worst_nd_name, worst_nd_m["nd"])
        else:
            nd_txt = "n/a (all key fields undefined_low_variance)"
        worst_txt = (
            "%s floor_zero rms_diff=%.6g" % (worst_name, worst_m["rms_diff"])
            if worst_m.get("floor_zero") else _fmt(worst_name, worst_m["floor_ratio"])
        )
        return LABEL_SMALL_DRIFT, "not within floor (%s); all SMALL-DRIFT checks passed; max nd %s" % (
            worst_txt, nd_txt,
        )
    return LABEL_LARGE_DRIFT, "; ".join(reasons)


def run_gate(reference, twin, candidate, family, bdy_width, thresholds, out_prefix, time_tol_s=None):
    thr = dict(DEFAULT_THRESHOLDS)
    thr.update(thresholds or {})
    if time_tol_s is not None:
        thr["time_tol_s"] = time_tol_s

    ref_role = Role("reference", reference)
    twin_role = Role("twin", twin)
    cand_role = Role("candidate", candidate)
    try:
        # 2026-09-15 fix: match times within thr["time_tol_s"] instead of requiring an
        # exact string match -- see match_times() for why exact matching silently
        # degenerated to "compare only t=0" for any candidate that changes time_step.
        matches, first_ref_time = match_times(ref_role, twin_role, cand_role, thr["time_tol_s"])
        fields = requested_fields(family)

        payload = {
            "label": None,
            "deciding": None,
            "family": family,
            "bdy_width": bdy_width,
            "thresholds": thr,
            "time_tol_s": thr["time_tol_s"],
            "files": {"reference": ref_role.paths, "twin": twin_role.paths, "candidate": cand_role.paths},
            "fields_requested": fields,
            "common_times": [m["reference"] for m in matches],
            "time_matches": matches,
        }

        if not matches:
            payload["label"] = LABEL_INVALID
            payload["deciding"] = (
                "no common valid times across reference/twin/candidate (time_tol_s=%.6g)" % thr["time_tol_s"]
            )
            _write_outputs(out_prefix, payload, [])
            return payload

        # A candidate that only coincides with the reference at its very first valid time
        # (e.g. a time_step change whose later history-write times drift away, so nothing
        # after t=0 matches within tolerance) must not be silently compared at t=0 only --
        # that reads as "no change" (nd=0, floor_ratio=0) when nothing was actually
        # compared post-initialization. Require at least one matched time strictly after
        # the reference's first valid time.
        post_init_matches = [m for m in matches if m["reference"] != first_ref_time]
        if not post_init_matches:
            payload["label"] = LABEL_INVALID
            payload["deciding"] = "no matched valid time after initialization"
            _write_outputs(out_prefix, payload, [])
            return payload

        last_match = post_init_matches[-1]  # matches is ordered by ascending reference time
        last_time = last_match["reference"]
        payload["last_time"] = last_time
        payload["last_time_match"] = last_match

        all_rows = []
        for m in matches:
            for f in fields:
                all_rows.append(field_metrics(f, ref_role, twin_role, cand_role, m, bdy_width, thr))

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
    "twin_offset_s", "candidate_offset_s",
    "rms_diff", "floor", "floor_zero", "floor_ratio",
    "std_ref", "min_std", "low_variance", "nd", "nd_status", "bias", "precip_rel_diff",
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
                _csv_num(row.get("twin_offset_s")),
                _csv_num(row.get("candidate_offset_s")),
                _csv_num(row.get("rms_diff")),
                _csv_num(row.get("floor")),
                row.get("floor_zero", ""),
                _csv_num(row.get("floor_ratio")),
                _csv_num(row.get("std_ref")),
                _csv_num(row.get("min_std")),
                row.get("low_variance", ""),
                _csv_num(row.get("nd")),
                row.get("nd_status", ""),
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
    p.add_argument("--time-tol-s", type=float, default=None,
                   help="seconds of slack when matching valid times across roles "
                        "(default 180; a candidate that changes time_step drifts its "
                        "history-write times away from the reference's exact seconds)")
    p.add_argument("--out-prefix", required=True, help="writes PREFIX.json and PREFIX.csv")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.bdy_width < 0:
        _parser().error("--bdy-width must be non-negative")
    if args.time_tol_s is not None and args.time_tol_s < 0:
        _parser().error("--time-tol-s must be non-negative")
    try:
        thresholds = _parse_thresholds(args.thresholds)
    except (OSError, json.JSONDecodeError) as error:
        print("ERROR: --thresholds: %s" % error, file=sys.stderr)
        return 2
    try:
        payload = run_gate(
            args.reference, args.twin, args.candidate, args.family,
            args.bdy_width, thresholds, args.out_prefix, time_tol_s=args.time_tol_s,
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
