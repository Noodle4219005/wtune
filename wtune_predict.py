#!/usr/bin/env python3
"""wtune_predict.py -- offline wall-time predictor for a WRF run of ANY length,
WITHOUT running WRF.

Standalone module (no import of wtune.py or any of its physics/knobs/surrogate
dependencies), so it can be imported and unit-tested cheaply on its own. wtune.py
hooks it in lazily from its `predict` subcommand (see the `predict` parser in
wtune.py's main()).

MODEL (every number below carries its source; see machine/calibration/scc.json
for the calibration constants themselves)
------------------------------------------------------------------------------
steps
    d01 steps = run length (run_days/hours/minutes/seconds from &time_control)
    / d01 time_step (+ time_step_fract_num/den). Nested domains are NOT counted
    separately: WRF prints nested-domain timings on their own, but the d01 step
    wall-clock already includes every nested substep inside it (ASSUMPTION,
    given; see maria1km's 432 d01 steps containing 2160 d02 substeps at
    parent_time_step_ratio=5).

compute (normal, non-radiation, non-I/O step cost)
    1. exact probe match (same case_base + same ranks/omp/nproc_x/nproc_y/
       numtiles) in a --probe ledger CSV -> 'measured'.
    2. else nearest measured config of the SAME case at a DIFFERENT rank
       count, scaled by ideal strong scaling (s_per_step *= ranks_old/ranks_new)
       -> 'extrapolated'.
    3. else calibration.fallback_s_per_cell_step_ns (documented UNCALIBRATED
       GUESS from the one conus25 reference run) -> 'no_probe_fallback'.

radiation
    radiation steps = floor(run_minutes / d01 radt). Each one costs r_rad times
    a normal step (r_rad default 5.0, source: atlas CONUS 2.5km measured range
    4.63-5.66); modeled as an ADD-ON: extra_s = rad_steps * (r_rad-1) * s_normal,
    since s_normal (probe median) already represents a normal/non-radiation step.

I/O
    history: frames = floor(run_minutes / history_interval) + 1 (t0 frame),
    per domain with io_form_history != 0. bytes/frame = e_we*e_sn*e_vert *
    calibration.derived.bytes_per_cell_level. seconds = bytes /
    calibration effective_rate[io_form][io_target].
    auxhist<N>: same shape, keyed off io_form_auxhistN + auxhistN_interval[_s|_m].
    When iofields_filename is set the stream cannot be sized exactly -> uses
    calibration.fallback_auxhist_fraction of a full same-domain frame, flagged
    'size unknown'.
    restart: floor(run_minutes / restart_interval) writes (io_form_restart,
    default = io_form_history), sized as calibration.restart.
    size_multiplier_vs_history_frame times a same-domain history frame -- an
    UNCALIBRATED GUESS, see scc.json. Never modeled for the exact conus25
    calibration fixture (see scc.json conus25_calibration_run.writing_total_note)
    to avoid double-counting against that run's own empirical total.

init
    read_s = (on-disk wrfinput_d*/wrfbdy_d* bytes in case_dir, else an ESTIMATE
    = bytes_per_cell_level * d01 cells) / calibration.derived.read_rate_bytes_per_s,
    plus calibration.fixed_startup_s (default 0.0, not independently measured).

drift
    +-calibration.drift_band_pct on the whole total, labelled NOT VALIDATED.

range
    combines (root-sum-square, in absolute seconds) probe noise on the compute
    total, the drift band on the grand total, and the I/O calibration
    uncertainty on the I/O total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = HERE / "machine" / "calibration" / "scc.json"

CORES_PER_NODE = 112  # scc site convention (see predictor/ledger.py CORES_PER_NODE)
SEGMENT_HOURS = 6.0   # long-run compute display chunking
LONG_RUN_HOURS = 12.0  # threshold above which the compute row is segmented

PROBE_NOISE_MEASURED_PCT = 3.0
PROBE_NOISE_EXTRAPOLATED_PCT = 15.0
PROBE_NOISE_FALLBACK_PCT = 50.0


class PredictError(Exception):
    """A user-facing modeling/input error (bad namelist, no timestep, etc.)."""


# --------------------------------------------------------------------------
# Namelist parsing (self-contained; list/per-domain aware)
# --------------------------------------------------------------------------

def _parse_scalar(tok):
    tok = tok.strip()
    if not tok:
        return None
    low = tok.lower()
    if low in (".true.", ".t.", "true"):
        return True
    if low in (".false.", ".f.", "false"):
        return False
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
        return tok[1:-1]
    m = re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?", tok)
    if m:
        s = tok.replace("d", "e").replace("D", "E")
        if "." in s or "e" in s.lower():
            try:
                return float(s)
            except ValueError:
                return tok
        return int(s)
    return tok


def parse_namelist(path):
    """Minimal Fortran namelist reader, list/per-domain aware.

    Returns {group: {key: [scalar, ...]}}, group/key lowercased. A line with
    no '=' that isn't a group header or '/' is treated as a continuation of
    the previous key's value list (WRF namelists sometimes wrap long
    per-domain lists across lines).
    """
    text = Path(path).read_text()
    text = re.sub(r"!.*$", "", text, flags=re.M)

    groups = {}
    group = None
    last_key = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r"&([A-Za-z_]\w*)", line)
        if m:
            group = m.group(1).lower()
            groups.setdefault(group, {})
            last_key = None
            continue

        if line.startswith("/"):
            group = None
            last_key = None
            continue

        if group is None:
            continue

        m = re.match(r"([A-Za-z_]\w*)\s*=\s*(.*)$", line)
        if m:
            key = m.group(1).lower()
            rest = m.group(2)
            last_key = key
            groups[group].setdefault(key, [])
        elif last_key is not None:
            key = last_key
            rest = line
        else:
            continue

        rest = rest.strip().rstrip(",").strip()
        if not rest:
            continue
        for tok in rest.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            groups[group][key].append(_parse_scalar(tok))

    return groups


def g(nl, group, key, default=None):
    """Namelist-global scalar (first value only)."""
    vals = nl.get(group, {}).get(key)
    if not vals:
        return default
    return vals[0]


def g_dom(nl, group, key, dom_index, default=None):
    """Per-domain value with WRF's broadcast-last-value convention: if fewer
    values are given than max_dom, every remaining domain reuses the last one.
    """
    vals = nl.get(group, {}).get(key)
    if not vals:
        return default
    if dom_index < len(vals):
        return vals[dom_index]
    return vals[-1]


def get_max_dom(nl):
    return int(g(nl, "domains", "max_dom", 1) or 1)


def get_domain_geom(nl, dom_index):
    e_we = g_dom(nl, "domains", "e_we", dom_index)
    e_sn = g_dom(nl, "domains", "e_sn", dom_index)
    e_vert = g_dom(nl, "domains", "e_vert", dom_index)
    if e_we is None or e_sn is None or e_vert is None:
        raise PredictError(
            "domain %d: missing e_we/e_sn/e_vert in &domains" % (dom_index + 1))
    return float(e_we), float(e_sn), float(e_vert)


def cells_levels(e_we, e_sn, e_vert):
    return e_we * e_sn * e_vert


def get_run_seconds(nl):
    days = g(nl, "time_control", "run_days", 0) or 0
    hours = g(nl, "time_control", "run_hours", 0) or 0
    minutes = g(nl, "time_control", "run_minutes", 0) or 0
    seconds = g(nl, "time_control", "run_seconds", 0) or 0
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise PredictError(
            "run length is zero: no run_days/run_hours/run_minutes/run_seconds "
            "found in &time_control")
    return float(total)


def get_timestep_seconds(nl):
    ts = g(nl, "domains", "time_step", None)
    if ts is None:
        raise PredictError("time_step not found in &domains")
    num = g(nl, "domains", "time_step_fract_num", 0) or 0
    den = g(nl, "domains", "time_step_fract_den", 1) or 1
    frac = (num / den) if den else 0.0
    dt = float(ts) + frac
    if dt <= 0:
        raise PredictError("time_step must be > 0 (got %r)" % (dt,))
    return dt


def get_d01_steps(nl):
    run_s = get_run_seconds(nl)
    dt = get_timestep_seconds(nl)
    exact = run_s / dt
    steps = int(round(exact))
    note = None
    if abs(exact - steps) > 1e-6:
        note = ("run length (%.3fs) is not an exact multiple of time_step "
                 "(%.3fs); rounded %.4f steps to %d" % (run_s, dt, exact, steps))
    return steps, run_s, dt, note


def parent_time_step_ratio(nl, dom_index, default=1):
    v = g_dom(nl, "domains", "parent_time_step_ratio", dom_index, default)
    return int(v) if v is not None else default


def nested_equivalent_cells(nl, max_dom):
    """Sum of each domain's cells*levels weighted by its cumulative substep
    count relative to d01 (d1 weight 1; d2 weight = its parent_time_step_ratio;
    d3 weight = d2's ratio * d3's ratio; ...). Used only by the no-probe
    fallback compute path, to avoid ignoring nested-domain cost entirely when
    there is no measured s_per_step at all to fall back on.
    """
    total = 0.0
    cum = 1
    for d in range(max_dom):
        e_we, e_sn, e_vert = get_domain_geom(nl, d)
        if d > 0:
            cum *= parent_time_step_ratio(nl, d, default=1)
        total += cells_levels(e_we, e_sn, e_vert) * cum
    return total


def get_history_interval_minutes(nl, dom_index):
    for key, scale in (("history_interval_s", 1.0 / 60.0),
                       ("history_interval_m", 1.0),
                       ("history_interval", 1.0),
                       ("history_interval_h", 60.0)):
        vals = nl.get("time_control", {}).get(key)
        if vals:
            v = g_dom(nl, "time_control", key, dom_index)
            if v is not None:
                return float(v) * scale, key
    return None, None


def get_aux_interval_minutes(nl, n, dom_index):
    for suffix, scale in (("_s", 1.0 / 60.0), ("_m", 1.0), ("", 1.0)):
        key = "auxhist%d_interval%s" % (n, suffix)
        vals = nl.get("time_control", {}).get(key)
        if vals:
            v = g_dom(nl, "time_control", key, dom_index)
            if v is not None:
                return float(v) * scale, key
    return None, None


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def load_calibration(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise PredictError("cannot read calibration file %s: %s" % (path, exc))


def get_effective_rate(calibration, io_form, io_target):
    table = calibration.get("derived", {}).get("io_form_rate_table", {})
    entry = table.get(str(int(io_form)))
    if entry is None:
        return {"rate_bytes_per_s": None, "uncertainty_pct": None,
                "source": "no calibration entry for io_form=%s" % io_form,
                "unmeasured": True}
    target_entry = entry.get(io_target)
    if target_entry is None:
        return {"rate_bytes_per_s": None, "uncertainty_pct": None,
                "source": "no calibration entry for io_target=%s" % io_target,
                "unmeasured": True}
    rate = target_entry.get("rate_bytes_per_s")
    return {"rate_bytes_per_s": rate,
            "uncertainty_pct": target_entry.get("uncertainty_pct"),
            "source": target_entry.get("source", ""),
            "unmeasured": rate is None}


# --------------------------------------------------------------------------
# Probe ledger
# --------------------------------------------------------------------------

def load_ledger_rows(paths):
    rows = []
    for path in paths or []:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    return rows


def _row_int(row, key, default=None):
    v = row.get(key)
    if v in (None, ""):
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _row_float(row, key, default=None):
    v = row.get(key)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def resolve_probe(ledger_rows, case_name, config):
    """config: dict with ranks, omp, nx (nproc_x), ny (nproc_y), numtiles.

    Returns None (no probe data for this case at all) or a dict:
      {s_normal, source in {'measured','extrapolated'}, detail...}
    """
    ranks, omp, nx, ny, numtiles = (
        config["ranks"], config["omp"], config["nx"], config["ny"], config["numtiles"])

    same_case = [r for r in ledger_rows
                 if r.get("case_base") == case_name and _row_float(r, "s_per_step_median") is not None]
    if not same_case:
        return None

    exact = [r for r in same_case
             if _row_int(r, "n_ranks") == ranks
             and _row_int(r, "omp_threads") == omp
             and _row_int(r, "nproc_x") == nx
             and _row_int(r, "nproc_y") == ny
             and _row_int(r, "numtiles") == numtiles]
    if exact:
        meds = sorted(_row_float(r, "s_per_step_median") for r in exact)
        n = len(meds)
        s_normal = meds[n // 2] if n % 2 else (meds[n // 2 - 1] + meds[n // 2]) / 2.0
        spread_pct = None
        if n >= 2:
            spread_pct = 100.0 * (max(meds) - min(meds)) / s_normal if s_normal else None
        return {"s_normal": s_normal, "source": "measured", "n_reps": n,
                "spread_pct": spread_pct,
                "config": {"n_ranks": ranks, "omp_threads": omp,
                           "nproc_x": nx, "nproc_y": ny, "numtiles": numtiles}}

    best = min(same_case, key=lambda r: abs((_row_int(r, "n_ranks") or 0) - ranks))
    from_ranks = _row_int(best, "n_ranks")
    from_s = _row_float(best, "s_per_step_median")
    if not from_ranks or from_s is None:
        return None
    s_normal = from_s * (from_ranks / float(ranks))
    return {"s_normal": s_normal, "source": "extrapolated",
            "from_ranks": from_ranks, "to_ranks": ranks,
            "from_config": {"n_ranks": from_ranks,
                             "omp_threads": _row_int(best, "omp_threads"),
                             "nproc_x": _row_int(best, "nproc_x"),
                             "nproc_y": _row_int(best, "nproc_y"),
                             "numtiles": _row_int(best, "numtiles")}}


# --------------------------------------------------------------------------
# Compute model
# --------------------------------------------------------------------------

def predict_compute(nl, max_dom, steps, run_min, ranks, omp, probe_result, calibration):
    radt = g_dom(nl, "physics", "radt", 0, default=None)
    r_rad = calibration.get("radiation", {}).get("r_rad_default", 5.0)

    if probe_result is not None:
        s_normal = probe_result["s_normal"]
        source = probe_result["source"]
        if source == "measured":
            noise_pct = (probe_result.get("spread_pct")
                         if probe_result.get("spread_pct") is not None
                         else PROBE_NOISE_MEASURED_PCT)
        else:
            noise_pct = PROBE_NOISE_EXTRAPOLATED_PCT
        detail = dict(probe_result)
    else:
        cl_eq = nested_equivalent_cells(nl, max_dom)
        ns_per_cell_core = calibration.get("fallback_s_per_cell_step_ns")
        if ns_per_cell_core is None:
            raise PredictError("no probe data and no calibration.fallback_s_per_cell_step_ns")
        cores = max(ranks * omp, 1)
        s_normal = ns_per_cell_core * 1e-9 * cl_eq / cores
        source = "no_probe_fallback"
        noise_pct = PROBE_NOISE_FALLBACK_PCT
        detail = {"source": source, "nested_equivalent_cells": cl_eq, "cores": cores,
                  "fallback_s_per_cell_step_ns": ns_per_cell_core}

    rad_steps = 0
    rad_extra_s = 0.0
    if radt:
        rad_steps = int(math.floor(run_min / radt))
        rad_extra_s = rad_steps * (r_rad - 1.0) * s_normal

    compute_total_s = steps * s_normal + rad_extra_s
    noise_abs_s = compute_total_s * (noise_pct / 100.0)

    return {
        "s_normal": s_normal, "source": source, "detail": detail,
        "steps": steps, "radt_min": radt, "r_rad": r_rad,
        "rad_steps": rad_steps, "rad_extra_s": rad_extra_s,
        "compute_total_s": compute_total_s,
        "noise_pct": noise_pct, "noise_abs_s": noise_abs_s,
    }


def segment_compute(compute, run_hours, dt):
    """Split the compute total into SEGMENT_HOURS chunks for display when the
    run is long. Same s_per_step in every segment -- drift is NOT VALIDATED.
    """
    if run_hours <= LONG_RUN_HOURS:
        return None
    steps_per_segment = int(round(SEGMENT_HOURS * 3600.0 / dt))
    total_steps = compute["steps"]
    segments = []
    remaining = total_steps
    start_h = 0.0
    while remaining > 0:
        n = min(steps_per_segment, remaining)
        seg_hours = n * dt / 3600.0
        seg_compute_s = n * compute["s_normal"]
        # radiation extra distributed proportionally to this segment's share of steps
        share = n / float(total_steps) if total_steps else 0.0
        seg_rad_s = compute["rad_extra_s"] * share
        segments.append({
            "sim_time": "%.1f-%.1fh" % (start_h, start_h + seg_hours),
            "steps": n, "compute_s": seg_compute_s + seg_rad_s,
        })
        remaining -= n
        start_h += seg_hours
    return segments


# --------------------------------------------------------------------------
# I/O model
# --------------------------------------------------------------------------

def predict_io(nl, max_dom, run_min, calibration, io_form_override, io_target):
    bpcl = calibration.get("derived", {}).get("bytes_per_cell_level")
    if bpcl is None:
        raise PredictError("calibration.derived.bytes_per_cell_level missing")

    io_form_history = io_form_override if io_form_override is not None else \
        g(nl, "time_control", "io_form_history", 2)

    streams = []
    total_s = 0.0
    total_uncertainty_sq = 0.0

    if io_form_history:
        for d in range(max_dom):
            interval_min, key = get_history_interval_minutes(nl, d)
            if interval_min is None or interval_min <= 0:
                streams.append({"domain": d + 1, "stream": "history",
                                 "note": "no history_interval found for this domain; skipped"})
                continue
            e_we, e_sn, e_vert = get_domain_geom(nl, d)
            n_frames = int(math.floor(run_min / interval_min)) + 1
            bytes_frame = bpcl * cells_levels(e_we, e_sn, e_vert)
            rate = get_effective_rate(calibration, io_form_history, io_target)
            if rate["unmeasured"]:
                streams.append({"domain": d + 1, "stream": "history", "n_frames": n_frames,
                                 "bytes_per_frame": bytes_frame,
                                 "note": "UNMEASURED rate for io_form=%s io_target=%s (%s); seconds not estimated"
                                         % (io_form_history, io_target, rate["source"])})
                continue
            seconds = n_frames * bytes_frame / rate["rate_bytes_per_s"]
            unc = seconds * (rate["uncertainty_pct"] / 100.0) if rate["uncertainty_pct"] else 0.0
            streams.append({"domain": d + 1, "stream": "history", "interval_min": interval_min,
                             "interval_key": key, "n_frames": n_frames,
                             "bytes_per_frame": bytes_frame, "seconds": seconds,
                             "rate_source": rate["source"]})
            total_s += seconds
            total_uncertainty_sq += unc * unc

    for n in range(1, 25):
        form_key = "io_form_auxhist%d" % n
        form_val = g(nl, "time_control", form_key, 0)
        if not form_val:
            continue
        interval_min, ikey = get_aux_interval_minutes(nl, n, 0)
        if interval_min is None or interval_min <= 0:
            streams.append({"stream": "auxhist%d" % n,
                             "note": "io_form_auxhist%d set but no auxhist%d_interval[_s|_m] found; skipped" % (n, n)})
            continue
        e_we, e_sn, e_vert = get_domain_geom(nl, 0)
        full_bytes = bpcl * cells_levels(e_we, e_sn, e_vert)
        iofields = nl.get("time_control", {}).get("iofields_filename")
        size_unknown = bool(iofields)
        fraction = calibration.get("fallback_auxhist_fraction", 1.0) if size_unknown else 1.0
        bytes_frame = full_bytes * fraction
        n_frames = int(math.floor(run_min / interval_min)) + 1
        rate = get_effective_rate(calibration, form_val, io_target)
        entry = {"stream": "auxhist%d" % n, "interval_min": interval_min,
                 "interval_key": ikey, "n_frames": n_frames,
                 "bytes_per_frame": bytes_frame, "size_unknown": size_unknown,
                 "fraction_used": fraction}
        if rate["unmeasured"]:
            entry["note"] = ("UNMEASURED rate for io_form=%s io_target=%s (%s); seconds not estimated"
                              % (form_val, io_target, rate["source"]))
            streams.append(entry)
            continue
        seconds = n_frames * bytes_frame / rate["rate_bytes_per_s"]
        unc = seconds * (rate["uncertainty_pct"] / 100.0) if rate["uncertainty_pct"] else 0.0
        entry["seconds"] = seconds
        entry["rate_source"] = rate["source"]
        streams.append(entry)
        total_s += seconds
        total_uncertainty_sq += unc * unc

    restart_interval = g(nl, "time_control", "restart_interval", None)
    if restart_interval and restart_interval > 0:
        n_restart = int(math.floor(run_min / restart_interval))
        if n_restart > 0:
            io_form_restart = g(nl, "time_control", "io_form_restart", io_form_history)
            mult = calibration.get("restart", {}).get("size_multiplier_vs_history_frame", 1.0)
            restart_bytes = 0.0
            for d in range(max_dom):
                e_we, e_sn, e_vert = get_domain_geom(nl, d)
                restart_bytes += bpcl * cells_levels(e_we, e_sn, e_vert) * mult
            rate = get_effective_rate(calibration, io_form_restart, io_target)
            entry = {"stream": "restart", "n_writes": n_restart, "bytes_per_write": restart_bytes,
                     "size_multiplier_vs_history_frame": mult,
                     "note": "UNCALIBRATED GUESS restart size (see calibration.restart)"}
            if rate["unmeasured"]:
                entry["note"] += "; UNMEASURED rate for io_form=%s io_target=%s" % (io_form_restart, io_target)
            else:
                seconds = n_restart * restart_bytes / rate["rate_bytes_per_s"]
                unc = seconds * (rate["uncertainty_pct"] / 100.0) if rate["uncertainty_pct"] else 0.0
                entry["seconds"] = seconds
                total_s += seconds
                total_uncertainty_sq += unc * unc
            streams.append(entry)

    return {"streams": streams, "total_s": total_s,
            "noise_abs_s": math.sqrt(total_uncertainty_sq)}


# --------------------------------------------------------------------------
# Init (read) model
# --------------------------------------------------------------------------

def predict_init(case_dir, nl, calibration):
    read_rate = calibration.get("derived", {}).get("read_rate_bytes_per_s")
    if not read_rate:
        raise PredictError("calibration.derived.read_rate_bytes_per_s missing")

    case_path = Path(case_dir) if case_dir else None
    real_files = []
    if case_path and case_path.is_dir():
        real_files = sorted(case_path.glob("wrfinput_d*")) + sorted(case_path.glob("wrfbdy_d*"))

    if real_files:
        total_bytes = sum(p.stat().st_size for p in real_files)
        source = "on-disk: " + ",".join(p.name for p in real_files)
    else:
        bpcl = calibration.get("derived", {}).get("bytes_per_cell_level")
        e_we, e_sn, e_vert = get_domain_geom(nl, 0)
        total_bytes = bpcl * cells_levels(e_we, e_sn, e_vert)
        source = ("ESTIMATED (no wrfinput_d*/wrfbdy_d* found in case_dir): assumed "
                   "wrfinput bytes ~= one d01 history frame's bytes, same assumption "
                   "used to derive read_rate from the conus25 calibration case")

    read_s = total_bytes / read_rate
    fixed_startup_s = calibration.get("fixed_startup_s", 0.0)
    return {"bytes": total_bytes, "source": source, "read_s": read_s,
            "fixed_startup_s": fixed_startup_s, "total_s": read_s + fixed_startup_s}


# --------------------------------------------------------------------------
# Parameter table / launch line
# --------------------------------------------------------------------------

def build_param_table(config, nodes, io_form, io_target):
    ranks, omp, nx, ny, numtiles = (
        config["ranks"], config["omp"], config["nx"], config["ny"], config["numtiles"])
    tpn = ranks // nodes if nodes else None
    warn = None
    if nodes and ranks % nodes != 0:
        warn = "ranks (%d) not evenly divisible by nodes (%d); TPN rounded down to %d" % (
            ranks, nodes, tpn)

    launch = ("mpirun -np %d --map-by ppr:%d:node:PE=%d --bind-to core"
              % (ranks, tpn if tpn else 0, omp))
    env = {"OMP_NUM_THREADS": str(omp), "OMP_STACKSIZE": "1G",
           "OMP_PLACES": "cores", "OMP_PROC_BIND": "close", "ulimit -s": "unlimited"}
    namelist_lines = ["nproc_x = %d," % nx, "nproc_y = %d," % ny, "numtiles = %d," % numtiles]

    return {"nproc_x": nx, "nproc_y": ny, "numtiles": numtiles,
            "ranks": ranks, "omp": omp, "nodes": nodes, "ranks_per_node": tpn,
            "dist": "cyclic", "io_form": io_form, "io_target": io_target,
            "launch_line": launch, "env": env, "namelist_lines": namelist_lines,
            "warning": warn}


# --------------------------------------------------------------------------
# Top-level assembly
# --------------------------------------------------------------------------

def resolve_config(args):
    if args.config:
        try:
            raw = json.loads(args.config)
        except json.JSONDecodeError:
            raw = json.loads(Path(args.config).read_text())
        ranks = int(raw.get("ranks") or (raw.get("nx", 1) * raw.get("ny", 1)))
        omp = int(raw.get("omp", 1))
        nx = int(raw["nx"] if "nx" in raw else raw["nproc_x"])
        ny = int(raw["ny"] if "ny" in raw else raw["nproc_y"])
        numtiles = int(raw.get("numtiles", 1))
    else:
        if args.nx is None or args.ny is None:
            raise PredictError("must give --config JSON, or --nx and --ny "
                                "(with optional --ranks/--omp/--numtiles)")
        nx, ny = int(args.nx), int(args.ny)
        omp = int(args.omp) if args.omp is not None else 1
        numtiles = int(args.numtiles) if args.numtiles is not None else 1
        ranks = int(args.ranks) if args.ranks is not None else nx * ny
    return {"ranks": ranks, "omp": omp, "nx": nx, "ny": ny, "numtiles": numtiles}


def build_prediction(namelist_path, case_dir, nodes, config, ledger_paths,
                      calibration_path, io_form_override, io_target, case_name=None):
    nl = parse_namelist(namelist_path)
    max_dom = get_max_dom(nl)
    steps, run_s, dt, steps_note = get_d01_steps(nl)
    run_min = run_s / 60.0
    run_hours = run_s / 3600.0

    calibration = load_calibration(calibration_path)

    ledger_rows = load_ledger_rows(ledger_paths)
    name = case_name or (Path(case_dir).name if case_dir else "unknown")
    probe_result = resolve_probe(ledger_rows, name, config)

    compute = predict_compute(nl, max_dom, steps, run_min, config["ranks"], config["omp"],
                              probe_result, calibration)
    segments = segment_compute(compute, run_hours, dt)

    io_target = io_target or "nfs"
    io_form_history_effective = io_form_override if io_form_override is not None else \
        g(nl, "time_control", "io_form_history", 2)
    io = predict_io(nl, max_dom, run_min, calibration, io_form_override, io_target)

    init = predict_init(case_dir, nl, calibration)

    drift_pct = calibration.get("drift_band_pct", 5.0)
    total_s = compute["compute_total_s"] + io["total_s"] + init["total_s"]
    drift_abs_s = total_s * (drift_pct / 100.0)

    range_abs_s = math.sqrt(compute["noise_abs_s"] ** 2 + io["noise_abs_s"] ** 2 + drift_abs_s ** 2)
    low, high = max(0.0, total_s - range_abs_s), total_s + range_abs_s

    params = build_param_table(config, nodes, io_form_history_effective, io_target)

    return {
        "case_name": name, "namelist": str(namelist_path),
        "max_dom": max_dom, "steps": steps, "run_s": run_s, "run_min": run_min,
        "run_hours": run_hours, "dt_s": dt, "steps_note": steps_note,
        "compute": compute, "segments": segments, "io": io, "init": init,
        "total_s": total_s, "drift_pct": drift_pct, "drift_abs_s": drift_abs_s,
        "range_low_s": low, "range_high_s": high,
        "param_table": params, "nodes": nodes,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt_hms(seconds):
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return "%dh%02dm%05.2fs" % (h, m, s)


def render_text(pred):
    out = []
    out.append("=== wtune predict: %s ===" % pred["case_name"])
    out.append("namelist: %s | max_dom=%d | d01 dt=%.3fs | run=%.1f min (%.2f h) | d01 steps=%d"
                % (pred["namelist"], pred["max_dom"], pred["dt_s"], pred["run_min"],
                   pred["run_hours"], pred["steps"]))
    if pred["steps_note"]:
        out.append("  NOTE: " + pred["steps_note"])
    out.append("")

    out.append("--- per-stage predicted wall time ---")
    c = pred["compute"]
    out.append("compute (normal steps, source=%s): s/step=%.4f  steps=%d  -> %s"
                % (c["source"], c["s_normal"], c["steps"], _fmt_hms(c["steps"] * c["s_normal"])))
    if c["radt_min"]:
        out.append("  + radiation surcharge: radt=%.1fmin  rad_steps=%d  r_rad=%.2f  -> %s"
                    % (c["radt_min"], c["rad_steps"], c["r_rad"], _fmt_hms(c["rad_extra_s"])))
    out.append("  compute total: %s  (+-%.1f%% = +-%s, source=%s)"
                % (_fmt_hms(c["compute_total_s"]), c["noise_pct"], _fmt_hms(c["noise_abs_s"]), c["source"]))
    if pred["segments"]:
        out.append("  segmented (>%.0fh simulated, %.0fh chunks, NOT VALIDATED drift):"
                    % (LONG_RUN_HOURS, SEGMENT_HOURS))
        for seg in pred["segments"]:
            out.append("    %-12s steps=%-6d %s" % (seg["sim_time"], seg["steps"], _fmt_hms(seg["compute_s"])))

    io = pred["io"]
    out.append("I/O: total %s" % _fmt_hms(io["total_s"]))
    for st in io["streams"]:
        if "seconds" in st:
            label = st.get("stream", "?")
            dom = (" d%02d" % st["domain"]) if "domain" in st else ""
            extra = " [size unknown, fraction=%.2f]" % st["fraction_used"] if st.get("size_unknown") else ""
            out.append("  %s%s: frames=%d bytes/frame=%.3e -> %s%s"
                        % (label, dom, st.get("n_frames", st.get("n_writes", 0)),
                           st.get("bytes_per_frame", st.get("bytes_per_write", 0.0)),
                           _fmt_hms(st["seconds"]), extra))
        elif "note" in st:
            out.append("  %s: %s" % (st.get("stream", "?"), st["note"]))

    init = pred["init"]
    out.append("init (read): %s  [%s]" % (_fmt_hms(init["total_s"]), init["source"]))

    out.append("")
    out.append("TOTAL predicted wall: %s  (range %s .. %s, includes +-%.0f%% NOT VALIDATED drift band)"
                % (_fmt_hms(pred["total_s"]), _fmt_hms(pred["range_low_s"]),
                   _fmt_hms(pred["range_high_s"]), pred["drift_pct"]))

    out.append("")
    out.append("--- recommended parameters ---")
    p = pred["param_table"]
    out.append("nproc_x=%d nproc_y=%d numtiles=%d  ranks=%d x omp=%d  nodes=%d  ranks/node=%s  dist=%s"
                % (p["nproc_x"], p["nproc_y"], p["numtiles"], p["ranks"], p["omp"],
                   p["nodes"] or 0, p["ranks_per_node"], p["dist"]))
    out.append("io_form=%s  io_target=%s" % (p["io_form"], p["io_target"]))
    if p["warning"]:
        out.append("  WARNING: " + p["warning"])
    out.append("launch: %s" % p["launch_line"])
    out.append("env: " + " ".join("%s=%s" % kv for kv in p["env"].items()))
    out.append("namelist:")
    for line in p["namelist_lines"]:
        out.append("  " + line)
    return "\n".join(out)


def to_json(pred):
    return json.dumps(pred, indent=2, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_subparser(sub):
    p = sub.add_parser("predict", help="predict per-stage + total wall time for a WRF run, without running it")
    p.add_argument("case_dir", help="directory with namelist.input (+ optional wrfinput_d0N/wrfbdy_d0N)")
    p.add_argument("--namelist", default=None, help="explicit namelist.input path (default: <case_dir>/namelist.input)")
    p.add_argument("--nodes", type=int, required=True, help="number of nodes")
    p.add_argument("--config", default=None, help="JSON blob or path: {ranks,omp,nx,ny,numtiles}")
    p.add_argument("--ranks", type=int, default=None, help="total MPI ranks (default nx*ny)")
    p.add_argument("--omp", type=int, default=None, help="OMP threads per rank (default 1)")
    p.add_argument("--nx", type=int, default=None, help="nproc_x")
    p.add_argument("--ny", type=int, default=None, help="nproc_y")
    p.add_argument("--numtiles", type=int, default=None, help="numtiles (default 1)")
    p.add_argument("--probe", action="append", default=None,
                   help="strict_campaign ledger CSV (repeatable)")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                   help="calibration JSON (default machine/calibration/scc.json)")
    p.add_argument("--io-form", type=int, default=None, choices=[2, 11, 14],
                   help="override io_form_history (default: read from namelist)")
    p.add_argument("--io-target", choices=["nfs", "local"], default="nfs",
                   help="write target (default nfs; local is UNMEASURED)")
    p.add_argument("--json", default=None, help="also write the full prediction as JSON to this path")
    return p


def predict_cmd(args):
    namelist_path = args.namelist or str(Path(args.case_dir) / "namelist.input")
    if not Path(namelist_path).exists():
        print("ERROR: namelist not found: %s" % namelist_path, file=sys.stderr)
        return 1
    try:
        config = resolve_config(args)
        pred = build_prediction(
            namelist_path=namelist_path, case_dir=args.case_dir, nodes=args.nodes,
            config=config, ledger_paths=args.probe, calibration_path=args.calibration,
            io_form_override=args.io_form, io_target=args.io_target)
    except PredictError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    print(render_text(pred))
    if args.json:
        Path(args.json).write_text(to_json(pred) + "\n")
        print("\nwritten: %s" % args.json)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wtune_predict.py")
    sub = parser.add_subparsers(dest="command")
    add_subparser(sub)
    args = parser.parse_args(argv)
    if args.command == "predict":
        return predict_cmd(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
