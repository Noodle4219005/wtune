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
    3. else, when --probe-rsl gives one or more rsl.error.0000 probe logs and
       the ledger produced no probe_result at all (no --probe row matched this
       case), s_normal is the median SCHEDULE-identified (not cost-threshold)
       non-radiation d01 step time from those logs (see measure_rad_from_rsl
       and the 'radiation' section below) -> 'probe_rsl'.
    4. else calibration.fallback_s_per_cell_step_ns (documented UNCALIBRATED
       GUESS from the one conus25 reference run) -> 'no_probe_fallback'.

radiation (DAY/NIGHT model, 2026-09-23 5-run validation)
    Radiation steps are identified by WRF's own CALL SCHEDULE, not by cost:
    stepra = steps_per_radiation_call(radt, dt) = NINT(radt*60/dt) (Fortran
    NINT; ported from tools/state_timeline.py, see that function's docstring
    for why this is a port and not an import), and step i (1-indexed) is a
    radiation step iff i==1 or i%stepra==0 (classify_steps, the DOCUMENTED
    schedule -- used for the run being PREDICTED, which has no real timing
    data to detect a phase from) or the data-detected phase variant
    (classify_steps_detected, used against a --probe-rsl log's OWN step
    numbering, since real rsl line numbering does not always land on phase 0
    -- see detect_radiation_phase). This REPLACES the previous "step time >
    1.8x the median" cost heuristic, which silently misclassified severe2023's
    night radiation steps (measured cost only ~1.05x a normal step, well under
    1.8x) as normal steps.

    Radiation cost is NOT a single multiplier: RRTMG shortwave only runs for
    sunlit columns, so the per-call cost is a two-level day/night switch
    measured 2026-09-23 across 5 full runs: matthew 1.92-1.97 (night) vs
    3.42-3.64 (day), sc26p2 2.47-2.52 vs 4.57-4.70, sc26p2narr 2.45-2.48 vs
    4.61-4.71 (day/night ratio 1.85-1.90x, mean 1.87). Model:
        r(t) = r_night + (r_day - r_night) * sunlit_fraction(t)
    sunlit_fraction(t): fraction of a 25-point (5x5) lat/lon sample lattice
    over the domain (bilinearly filled from d01's 4 corners, centre sample
    overridden by the real projected CEN_LAT/CEN_LON -- see
    sample_lattice_5x5) with cos(solar zenith) > 0 at simulated time t (see
    solar_cos_zenith: declination + hour-angle formula, equation of time and
    refraction OMITTED -- see that function's own docstring). Domain geometry
    is read from case_dir/wrfinput_d01's global attributes via `ncdump -h`
    TEXT output (read_wrfinput_sample_points) -- no netCDF4 dependency, per
    this module's own standalone rule; when wrfinput_d01/ncdump aren't
    available the day/night model degrades to a single uniform level (see
    below), not a hard error.

    r_night/r_day are obtained, in order:
      1. --probe-rsl log(s) with >= 2 schedule-identified radiation steps AND
         a resolvable simulated time + domain geometry for each: least-
         squares fit of (radiation step seconds / s_normal) against that
         step's own sunlit_fraction(t) (fit_day_night_r_rad) when the
         observed sunlit fractions span >= 0.5 (both regimes seen) ->
         'fit_day_night'.
      2. same data but ALL observed radiation steps fall in one regime (a
         probe's t0 fixes its time of day, so a single probe window usually
         sees only one -- e.g. matthew's t0 is 00Z/night): the observed
         level's mean is r_night or r_day directly, and the OTHER level is
         derived via calibration.radiation.r_rad_day_night_ratio (1.87,
         MEASURED from the 3 cases above) -> 'ratio_from_night' /
         'ratio_from_day'.
      3. no usable radiation-step/sunlit-fraction data at all (no
         --probe-rsl, or geometry/sim-time unavailable): r_night = r_day =
         calibration.radiation.r_rad_default (5.0, UNCALIBRATED default) ->
         'default_unmeasured'.
    Predicted radiation extra: for each of the PREDICTED run's own schedule-
    identified radiation steps k at simulated time t_k (from &time_control's
    start_date + (k-1)*dt, when start_date is present), extra_s = sum_k
    (r(t_k) - 1) * s_normal. When start_date or domain geometry is
    unavailable, r(t_k) falls back to the uniform mean (r_night+r_day)/2 for
    every radiation step (documented "day/night phase unknown for this run"
    fallback), rather than refusing to predict at all.

I/O
    history: frames = floor(run_minutes / history_interval) + 1 (t0 frame),
    per domain with io_form_history != 0. bytes/frame = e_we*e_sn*e_vert *
    calibration.derived.bytes_per_cell_level. seconds = bytes /
    calibration effective_rate[io_form][io_target].
    auxhist<N>: per domain, keyed off io_form_auxhistN + auxhistN_interval[_s|_m],
    honoring auxhistN_begin/end[_s|_m|_h] as the write window (default: the
    whole run) -- frames = floor((end-begin)/interval) + 1. Sizing, no
    calibration constant involved: when iofields_filename is set, reads that
    per-domain field-list file (case_dir, or --iofields-dir) and sizes the
    frame as sum over its '+'-listed fields of (points x 4 bytes), points =
    e_we*e_sn*e_vert for a 3-D field or e_we*e_sn for 2-D (U/V get +1 point in
    their staggered horizontal dimension), classified via a small built-in
    name table (unknown names default to 3-D and are flagged as guessed). If
    the file can't be found/read/matched for that stream number, the stream
    is flagged 'size unknown' (no seconds estimated) rather than guessed at.
    With no iofields_filename at all, the stream is assumed to carry a full
    same-domain frame (bytes_per_cell_level model, same as history).
    restart: floor(run_minutes / restart_interval) writes, for EVERY active
    domain, each write costing calibration.restart.restart_to_history_time_ratio
    (MEASURED, job 427: 431.2s restart write / 100.7s mean history-frame write
    = 4.28x) times that domain's own history-frame write time (io_form_restart,
    default = io_form_history). Modeled as a normal, separate line item even
    for the conus25 calibration fixture itself, since history and restart
    write rates are calibrated independently (see scc.json conus25_calibration_run).

init
    read_s = (on-disk wrfinput_d*/wrfbdy_d* bytes in case_dir, else an ESTIMATE
    = bytes_per_cell_level * d01 cells) / calibration.derived.read_rate_bytes_per_s,
    plus calibration.fixed_startup_s (default 0.0, not independently measured;
    dead/legacy -- no calibration file sets it) plus calibration.startup_s
    (optional, a single measured seconds constant for MPI launch + wrfinput
    read + first step + finalise; a 2026-09-23 scc validation found wall -
    sum(per-step times) was 17-22s on every 1-node run, i.e. materially
    nonzero). When startup_s is null/absent, the text output prints
    "startup: UNCALIBRATED" rather than silently treating the gap as 0.

drift
    +-calibration.drift_band_pct: on the whole total when no I/O rate range is
    calibrated (see 'range' below), or scoped to the compute total ONLY when
    calibration.history_write_rate_MBps_low/_high are set, in which case I/O's
    own uncertainty is expressed as the fast/slow bracket instead. Labelled
    NOT VALIDATED either way.

range
    Without calibration.history_write_rate_MBps_low/_high: combines
    (root-sum-square, in absolute seconds) probe noise on the compute total,
    the drift band on the grand total, and the I/O calibration uncertainty on
    the I/O total.
    With calibration.history_write_rate_MBps_low/_high set: the printed range
    is instead [compute_total - compute_drift .. compute_total + compute_drift]
    + [I/O total computed at the HIGH (fast) rate .. I/O total computed at the
    LOW (slow) rate] + init_total, i.e. an explicit measured I/O bracket
    (NFS write time was observed to vary 2.4x for identical writes under
    concurrent load on 2026-09-23) plus a compute-only drift band, reported
    separately (compute.noise_abs_s is still computed and reported but not
    folded into this bracket) so the printed range's two sources -- I/O rate
    variance vs compute drift -- stay distinguishable rather than blended by
    root-sum-square.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import wtune_runlength as runlen  # repo root: resolve_run_seconds, the single-sourced run_*
                                   # keys vs. start/end date-span resolver -- dependency-free
                                   # (stdlib only), so importing it does not pull in wtune.py or
                                   # any of its physics/knobs/surrogate dependencies (this
                                   # module's own "standalone" property, see module docstring)

HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = HERE / "machine" / "calibration" / "scc.json"

CORES_PER_NODE = 112  # scc site convention (see predictor/ledger.py CORES_PER_NODE)
SEGMENT_HOURS = 6.0   # long-run compute display chunking
LONG_RUN_HOURS = 12.0  # threshold above which the compute row is segmented

PROBE_NOISE_MEASURED_PCT = 3.0
PROBE_NOISE_EXTRAPOLATED_PCT = 15.0
PROBE_NOISE_FALLBACK_PCT = 50.0
# A probe_rsl s_normal is a single run's own step-timing sample (not
# cross-validated against a separate ledger measurement the way a --probe
# 'measured' row often is), so it gets a slightly wider band than
# PROBE_NOISE_MEASURED_PCT -- an explicit, disclosed judgment call, not a
# measured uncertainty.
PROBE_NOISE_RSL_PCT = 5.0

# fit_day_night_r_rad: a --probe-rsl's observed sunlit fractions must span at
# least this much (max - min) to trust a least-squares day/night fit; below
# it, the probe effectively saw one regime (see module docstring's
# 'radiation' section, level 2: derive the other level via the calibrated
# day/night ratio instead of extrapolating a near-degenerate regression).
RADIATION_FIT_MIN_SUNLIT_SPREAD = 0.5

DEFAULT_R_RAD_DAY_NIGHT_RATIO = 1.87  # see module docstring; MEASURED mean of
                                       # matthew/sc26p2/sc26p2narr (1.85, 1.87, 1.90)


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


def _mask_quotes(line):
    """Replace the CONTENTS of single/double-quoted spans with '\\0' (same
    length, same positions), so a later scan for identifier '=' boundaries
    never mistakes a quoted string's contents for a key or matches an '='
    that happens to appear inside one. Quote characters themselves are kept
    so string-valued tokens still parse correctly downstream.
    """
    out = []
    in_quote = None
    for ch in line:
        if in_quote:
            out.append(ch if ch == in_quote else "\0")
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ("'", '"'):
                in_quote = ch
            out.append(ch)
    return "".join(out)


_ASSIGNMENT_RE = re.compile(r"[A-Za-z_]\w*\s*=(?!=)")


def _split_assignments(line):
    """Split a namelist line into [(key, raw_value_text), ...], handling any
    number of 'key = value[, value...]' assignments on one line (real WRF
    namelists commonly pack run_days/run_hours/run_minutes/run_seconds, or
    time_step/time_step_fract_num/time_step_fract_den, onto a single line).
    Returns [] if the line has no top-level 'identifier=' at all (a pure
    continuation line).
    """
    masked = _mask_quotes(line)
    matches = list(_ASSIGNMENT_RE.finditer(masked))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        key = m.group(0).split("=", 1)[0].strip()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        out.append((key, line[m.end():value_end]))
    return out


def _append_values(bucket, raw_value_text):
    rest = raw_value_text.strip().rstrip(",").strip()
    if not rest:
        return
    for tok in rest.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        bucket.append(_parse_scalar(tok))


def parse_namelist(path):
    """Minimal Fortran namelist reader, list/per-domain aware.

    Returns {group: {key: [scalar, ...]}}, group/key lowercased. A line may
    carry any number of 'key = value[, value...]' assignments (values may
    themselves be comma lists); the next key starts at the next identifier
    followed by '=', outside quotes. A line with no top-level 'key=' at all
    that isn't a group header or '/' is treated as a continuation of the
    previous key's value list (WRF namelists sometimes wrap long per-domain
    lists across lines).
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

        assignments = _split_assignments(line)
        if assignments:
            for key, raw_value in assignments:
                key = key.lower()
                last_key = key
                groups[group].setdefault(key, [])
                _append_values(groups[group][key], raw_value)
        elif last_key is not None:
            _append_values(groups[group][last_key], line)
        # else: stray line with no assignment and no prior key -- ignore.

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


def get_time_control_datetime(nl, prefix):
    """d01 <prefix>_year/month/day/hour/minute/second (first value only, same
    broadcast-last-value indifference as g() -- every active domain of a
    run-length-defining start/end pair is aligned to the same wall time) -> datetime, or None
    when year/month/day are not all present (minute/second default to 0; many real templates,
    e.g. lafe's, omit start_second/end_second when it is 0)."""
    y = g(nl, "time_control", prefix + "_year", None)
    mo = g(nl, "time_control", prefix + "_month", None)
    d = g(nl, "time_control", prefix + "_day", None)
    if y is None or mo is None or d is None:
        return None
    h = g(nl, "time_control", prefix + "_hour", 0) or 0
    mi = g(nl, "time_control", prefix + "_minute", 0) or 0
    s = g(nl, "time_control", prefix + "_second", 0) or 0
    # hour/minute/second may overflow their clock range: WRF accepts e.g. end_minute = 120
    # (idealized em_quarter_ss) because run_* decides the length -- normalise via a timedelta.
    return (datetime.datetime(int(y), int(mo), int(d))
            + datetime.timedelta(hours=int(h), minutes=int(mi), seconds=int(s)))


def get_run_seconds(nl):
    """days*86400 + hours*3600 + minutes*60 + seconds from &time_control's run_days/run_hours/
    run_minutes/run_seconds when that sum is > 0; otherwise the first domain's start_date ->
    end_date span (WRF's own fallback once every run_* key is 0/absent -- see
    wtune_runlength.py's module docstring for the lafe/isc21 evidence). Delegates to
    wtune_runlength.resolve_run_seconds (single-sourced with wtune.py:extract_case and
    tools/temporal_validation.py:read_run_length_hours) and translates its ValueError into this
    module's own PredictError so callers' existing except PredictError contract is unchanged."""
    days = g(nl, "time_control", "run_days", 0) or 0
    hours = g(nl, "time_control", "run_hours", 0) or 0
    minutes = g(nl, "time_control", "run_minutes", 0) or 0
    seconds = g(nl, "time_control", "run_seconds", 0) or 0
    start_dt = get_time_control_datetime(nl, "start")
    end_dt = get_time_control_datetime(nl, "end")
    try:
        return runlen.resolve_run_seconds(days, hours, minutes, seconds, start_dt, end_dt)
    except ValueError:
        raise PredictError(
            "run length is zero: no run_days/run_hours/run_minutes/run_seconds found in "
            "&time_control, and start_date/end_date do not establish a positive span either")


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


def get_aux_window_minutes(nl, n, dom_index, run_minutes):
    """auxhistN_begin[_s|_m|_h] / auxhistN_end[_s|_m|_h]: the sub-window of the
    run this stream actually writes in (WRF default: the whole run, i.e.
    begin=0, end=run length). Returns (begin_min, end_min), end clipped to
    run_minutes.
    """
    begin_min = 0.0
    end_min = run_minutes
    for prefix, default in (("begin", 0.0), ("end", run_minutes)):
        for suffix, scale in (("_s", 1.0 / 60.0), ("_m", 1.0), ("_h", 60.0), ("", 1.0)):
            key = "auxhist%d_%s%s" % (n, prefix, suffix)
            vals = nl.get("time_control", {}).get(key)
            if vals:
                v = g_dom(nl, "time_control", key, dom_index)
                if v is not None:
                    if prefix == "begin":
                        begin_min = float(v) * scale
                    else:
                        end_min = float(v) * scale
                    break
    return begin_min, min(end_min, run_minutes)


# --------------------------------------------------------------------------
# iofields_filename field-subset sizing (first principles: sum of listed
# fields' point counts x 4 bytes/point; no calibration constant involved)
# --------------------------------------------------------------------------

# Not exhaustive -- common WRF output fields only. Anything not listed here
# defaults to 3-D (the caller records which names were defaulted/guessed).
FIELD_3D_NAMES = {
    "U", "V", "W", "T", "PH", "PHB", "P", "PB", "QVAPOR", "QCLOUD", "QRAIN",
    "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "REFL_10CM", "THETA",
    "TKE", "CLDFRA", "ALT", "AL", "GEOPT", "P_HYD", "DNW", "RTHRATEN",
}
FIELD_2D_NAMES = {
    "T2", "Q2", "U10", "V10", "PSFC", "RAINNC", "RAINC", "SWDOWN", "OLR",
    "PBLH", "HFX", "LH", "TSK", "XLAT", "XLONG", "LU_INDEX", "HGT",
    "LANDMASK", "SST", "SNOW", "SNOWH", "SEAICE", "GLW", "ALBEDO", "EMISS",
    "UST", "ZNT", "QFX", "SR", "MOL", "GSW", "SNOWC", "COSZEN", "SNOWNC",
    "GRAUPELNC", "RAINSH", "TH2", "MU", "MUB",
}
# Horizontally staggered fields: +1 grid point in the given namelist
# dimension relative to an unstaggered (mass) field of the same e_we/e_sn.
FIELD_STAGGER = {"U": "x", "V": "y"}


def field_points(name, e_we, e_sn, e_vert):
    """Returns (points, is_guessed). points = e_we*e_sn*e_vert for a 3-D
    field or e_we*e_sn for 2-D, per the given sizing rule; U/V get +1 in
    their staggered horizontal dimension. Unknown names default to 3-D.
    """
    name_u = name.strip().upper()
    we, sn = e_we, e_sn
    stag = FIELD_STAGGER.get(name_u)
    if stag == "x":
        we = e_we + 1
    elif stag == "y":
        sn = e_sn + 1
    if name_u in FIELD_2D_NAMES:
        return we * sn, False
    if name_u in FIELD_3D_NAMES:
        return we * sn * e_vert, False
    return we * sn * e_vert, True  # unknown -> default to 3-D, flagged


def parse_iofields_file(path):
    """Parse a WRF iofields_filename file. Returns {stream_num: set(names)}.

    WRF convention, one entry per (usually one) line: '<+|-> : <h|i> :
    <comma/space-separated stream numbers> : <comma-separated field names>'.
    '+' adds the named fields to the given stream(s) (this is how an
    auxhist stream, which otherwise defaults to empty, gets populated);
    '-' removes fields (used to trim the primary history stream, which
    otherwise defaults to everything). A line that doesn't match the
    '<+|-> : ...' pattern is treated as a continuation of the previous
    entry's field list (long lists sometimes wrap). '#'/'!' lines are
    comments.
    """
    streams = {}
    pending_nums = None
    pending_add = True
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        m = re.match(r"^([+-])\s*:\s*[A-Za-z]\s*:\s*([\d,\s]+)\s*:\s*(.*)$", line)
        if m:
            sign, nums, names_part = m.groups()
            pending_add = (sign == "+")
            pending_nums = [int(x) for x in re.split(r"[,\s]+", nums.strip()) if x]
        elif pending_nums is not None:
            names_part = line
        else:
            continue
        names_part = names_part.strip().rstrip(",").strip()
        if not names_part:
            continue
        names = [nm.strip().upper() for nm in names_part.split(",") if nm.strip()]
        for sn in pending_nums:
            bucket = streams.setdefault(sn, set())
            if pending_add:
                bucket.update(names)
            else:
                bucket.difference_update(names)
    return streams


def iofields_frame_bytes(path, stream_num, e_we, e_sn, e_vert):
    """Returns {'bytes': float, 'names': [...], 'guessed': [...]} sized from
    an iofields_filename file's entries for this stream number, or None if
    the file has no '+' entries naming this stream (nothing to size from).
    Raises OSError/IOError if the file cannot be read (caller decides how to
    report that).
    """
    streams = parse_iofields_file(path)
    names = sorted(streams.get(stream_num, set()))
    if not names:
        return None
    total = 0.0
    guessed = []
    for nm in names:
        pts, is_guess = field_points(nm, e_we, e_sn, e_vert)
        total += pts * 4.0
        if is_guess:
            guessed.append(nm)
    return {"bytes": total, "names": names, "guessed": guessed}


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
# WRF radiation call schedule (ported from tools/state_timeline.py -- NOT
# imported: that module pulls in campaign/configs.py, predictor/decomp_space.py,
# tools/scc_sweep.py and tools/temporal_validation.py via sys.path
# manipulation, none of which this module may depend on, per its own
# "standalone" rule -- see module docstring's opening paragraph)
# --------------------------------------------------------------------------

def steps_per_radiation_call(radt_minutes, dt_seconds):
    """STEPRA = NINT(radt_minutes*60/dt_seconds), WRF's own radiation-call
    cadence (phys/module_radiation_driver.F via
    share/module_first_rk_step_part1.F). Same function/semantics as
    tools/state_timeline.py's steps_per_radiation_call. Fortran NINT rounds
    halves away from zero; Python's round() rounds halves to even and gets
    conus12 wrong (radt=15min, dt=72s -> 12.5; round()=12, floor(x+0.5)=13,
    the correct cadence -- campaign/cases.json's sc26p1 note: "radt=9 at
    dt=10 => STEPRA=54" is floor(9*60/10+0.5)==54). Floored at 1 (a radt so
    small relative to dt it would fire every step).
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if radt_minutes <= 0:
        raise ValueError("radt_minutes must be > 0")
    return max(1, int(math.floor(radt_minutes * 60.0 / dt_seconds + 0.5)))


def detect_radiation_phase(step_wall_times, stepra):
    """Offset p in [0, stepra) such that step i (1-indexed, i>1) is a
    radiation step when i % stepra == p, chosen as the phase whose steps
    have the highest MEDIAN wall time (median ignores rare I/O-write
    spikes) -- ported from tools/state_timeline.py. The documented
    mod(itimestep,STEPRA)==0 rule does not always match a real rsl log's
    own line numbering (state_timeline's matthew citation: dt 150s, radt
    15min, STEPRA 6, spikes land on phase 1, not phase 0). Comparing every
    candidate phase's median against every OTHER phase's median is also
    what makes severe2023's ~1.05x night-radiation bump detectable at all:
    it is a RELATIVE comparison across schedule phases, not a fixed
    absolute cost threshold (the bug this whole redesign fixes -- see
    module docstring). Falls back to phase 0 when there are fewer than
    2*stepra steps or the phases are statistically indistinguishable.
    """
    n = len(step_wall_times)
    if stepra <= 1 or n < 2 * stepra:
        return 0
    best, best_med = 0, None
    for p in range(stepra):
        vals = [w for i, w in enumerate(step_wall_times, start=1) if i > 1 and i % stepra == p]
        if not vals:
            continue
        med = statistics.median(vals)
        if best_med is None or med > best_med:
            best, best_med = p, med
    return best


def classify_steps_detected(step_wall_times, stepra):
    """[is_radiation, ...] (1-indexed, same length as step_wall_times) using
    the DATA-DETECTED phase (detect_radiation_phase) -- for a real
    --probe-rsl log. Ported from tools/state_timeline.py."""
    p = detect_radiation_phase(step_wall_times, stepra)
    return [(i == 1) or (stepra > 0 and i % stepra == p) for i in range(1, len(step_wall_times) + 1)]


def classify_steps(n_steps, stepra):
    """[is_radiation, ...] for 1-indexed itimestep in 1..n_steps, using WRF's
    DOCUMENTED schedule (itimestep==1 or itimestep%stepra==0) rather than a
    data-detected phase -- used for the run being PREDICTED, which has no
    real rsl timing to detect a phase from. Ported from
    tools/state_timeline.py."""
    if n_steps <= 0:
        return []
    return [(i == 1) or (i % stepra == 0) for i in range(1, n_steps + 1)]


# --------------------------------------------------------------------------
# Solar position / domain geometry (day/night radiation sunlit fraction)
# --------------------------------------------------------------------------

def solar_cos_zenith(lat_deg, lon_deg, dt_utc):
    """cos(solar zenith angle) at (lat_deg, lon_deg), dt_utc (a naive UTC
    datetime -- WRF simulated times are UTC). Declination + hour-angle
    formula:
      declination (Cooper 1969 approximation):
        dec_deg = 23.45 * sin(360/365 * (284 + day_of_year))
      hour angle (solar time approximated as UTC + lon_deg/15h, i.e. every
      15 degrees of longitude = 1 hour of solar time):
        H_deg = 15 * (solar_hour - 12)
      cos(zenith) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(H)
    EQUATION OF TIME (a seasonal +-16-minute correction to solar time) and
    atmospheric refraction are OMITTED -- explicitly out of scope per this
    feature's spec. > 0 means sunlit (sun above the local horizon, ignoring
    refraction's small horizon-dip correction).
    """
    day_of_year = dt_utc.timetuple().tm_yday
    dec = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (284 + day_of_year))))
    lat = math.radians(lat_deg)
    solar_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + lon_deg / 15.0
    hour_angle = math.radians(15.0 * (solar_hour - 12.0))
    return math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)


def _bilinear_latlon(sw, nw, ne, se, u, v):
    """Bilinear interpolation of 4 corner (lat, lon) pairs; u in [0,1] west
    (0) -> east (1), v in [0,1] south (0) -> north (1)."""
    bot_lat = sw[0] * (1 - u) + se[0] * u
    bot_lon = sw[1] * (1 - u) + se[1] * u
    top_lat = nw[0] * (1 - u) + ne[0] * u
    top_lon = nw[1] * (1 - u) + ne[1] * u
    return bot_lat * (1 - v) + top_lat * v, bot_lon * (1 - v) + top_lon * v


def sample_lattice_5x5(corner_sw, corner_nw, corner_ne, corner_se, center):
    """25 (lat, lon) sample points on a 5x5 grid, bilinearly interpolated
    from the domain's 4 corners (_bilinear_latlon), EXCEPT the exact centre
    sample (u=v=0.5), which is replaced by the wrfinput-reported
    CEN_LAT/CEN_LON -- the real map-projected centre, not necessarily the
    corners' arithmetic mean under WRF's Lambert/Mercator/polar-stereographic
    projections."""
    pts = []
    lattice = (0.0, 0.25, 0.5, 0.75, 1.0)
    for v in lattice:
        for u in lattice:
            if u == 0.5 and v == 0.5:
                pts.append(center)
            else:
                pts.append(_bilinear_latlon(corner_sw, corner_nw, corner_ne, corner_se, u, v))
    return pts


def sunlit_fraction(sample_points, dt_utc):
    """Fraction of sample_points ([(lat, lon), ...]) with cos(solar zenith)
    > 0 at dt_utc (solar_cos_zenith). None if sample_points is empty."""
    if not sample_points:
        return None
    lit = sum(1 for lat, lon in sample_points if solar_cos_zenith(lat, lon, dt_utc) > 0)
    return lit / float(len(sample_points))


def _parse_ncdump_number(tok):
    tok = tok.strip()
    tok = re.sub(r"[fFdD]$", "", tok)  # NetCDF float/double literal suffix (ncdump -h convention)
    return float(tok)


def _parse_ncdump_scalar(text, name):
    m = re.search(r":%s\s*=\s*([^;]+);" % re.escape(name), text)
    if not m:
        return None
    try:
        return _parse_ncdump_number(m.group(1))
    except ValueError:
        return None


def _parse_ncdump_array(text, name):
    m = re.search(r":%s\s*=\s*([^;]+);" % re.escape(name), text)
    if not m:
        return None
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(_parse_ncdump_number(tok))
        except ValueError:
            return None
    return out


def parse_wrfinput_corners(ncdump_text):
    """Parse CEN_LAT/CEN_LON/CORNER_LATS/CORNER_LONS global attributes out
    of `ncdump -h` TEXT output for a wrfinput_d01 (or wrfout) file -- no
    netCDF4 dependency, per this module's standalone rule. WRF's
    corner_lats/corner_lons global attributes hold 16 values: 4 corners
    (SW, NW, NE, SE) x 4 stagger variants (mass, u, v, unstaggered); the
    mass-point (unstaggered T-grid) corner for corner index c is array
    index c*4 (WRF Registry / share/module_bc.F corner ordering) --
    ASSUMED, not independently verified against a real wrfinput file in
    this repo (no netCDF tooling on this dev machine -- see
    read_wrfinput_sample_points). Falls back to treating a 4-element array
    as the four corners directly (SW, NW, NE, SE in file order) when the
    array isn't the expected 16-element form.
    Returns None when CEN_LAT/CEN_LON are missing (nothing usable at all);
    else {"cen": (lat, lon), "corners": (sw, nw, ne, se) or None (when
    CORNER_LATS/CORNER_LONS are absent, malformed, or an unrecognized
    length -- caller degrades to a centre-only sample)}.
    """
    cen_lat = _parse_ncdump_scalar(ncdump_text, "CEN_LAT")
    cen_lon = _parse_ncdump_scalar(ncdump_text, "CEN_LON")
    if cen_lat is None or cen_lon is None:
        return None
    lats = _parse_ncdump_array(ncdump_text, "CORNER_LATS")
    lons = _parse_ncdump_array(ncdump_text, "CORNER_LONS")
    corners = None
    if lats and lons and len(lats) == len(lons):
        if len(lats) >= 16:
            idx = (0, 4, 8, 12)
        elif len(lats) == 4:
            idx = (0, 1, 2, 3)
        else:
            idx = None
        if idx:
            corners = tuple((lats[i], lons[i]) for i in idx)
    return {"cen": (cen_lat, cen_lon), "corners": corners}


def domain_sample_points_from_ncdump(ncdump_text):
    """sample_lattice_5x5 built from parse_wrfinput_corners(ncdump_text), or
    a single-point [center] sample when corners aren't available (degrades
    the day/night model to a coarser but still non-trivial estimate), or
    None when even CEN_LAT/CEN_LON are missing."""
    parsed = parse_wrfinput_corners(ncdump_text)
    if parsed is None:
        return None
    if parsed["corners"] is None:
        return [parsed["cen"]]
    sw, nw, ne, se = parsed["corners"]
    return sample_lattice_5x5(sw, nw, ne, se, parsed["cen"])


# --------------------------------------------------------------------------
# Real XLAT/XLONG grid sampling (2026-09-23 scc finding: a real real.exe
# wrfinput_d01 has NO CORNER_LATS/CORNER_LONS global attributes at all --
# only CEN_LAT/CEN_LON -- so parse_wrfinput_corners/domain_sample_points_
# from_ncdump above NEVER fire on a real wrfinput and read_wrfinput_
# sample_points ALWAYS fell back to sample_points=None, i.e. the day/night
# model silently used the uniform (r_night+r_day)/2 level on every real scc
# run. Fixed by reading the actual XLAT/XLONG mass-point arrays -- exact
# grid values, no bilinear guess needed -- via netCDF4 when available, else
# `ncdump -v XLAT,XLONG` array text; CORNER_LATS/CORNER_LONS remains a
# fallback of last resort for a met_em/wrfout file that DOES carry them.
# --------------------------------------------------------------------------

def _lattice_indices(n, count=5):
    """count evenly spaced integer indices in [0, n-1], including BOTH
    edges (index 0 and n-1) whenever count >= 2. Used to pick a 5x5 lattice
    directly out of a real XLAT/XLONG grid (no interpolation needed -- the
    grid already has a real value at every index)."""
    if n <= 1:
        return [0] * count
    return [round(t * (n - 1) / (count - 1)) for t in range(count)]


def sample_lattice_5x5_from_grid(lat2d, lon2d):
    """25 (lat, lon) sample points read DIRECTLY from a real 2-D XLAT/XLONG
    grid (lat2d/lon2d, indexed [south_north][west_east], row 0 = the
    southernmost row -- WRF's south_north index increases northward), at 5
    evenly spaced south_north indices x 5 evenly spaced west_east indices
    (_lattice_indices), both including their domain edges. No bilinear
    interpolation, unlike sample_lattice_5x5 -- every sampled point is a
    real grid value, since we have the whole array."""
    n_rows = len(lat2d)
    n_cols = len(lat2d[0]) if n_rows else 0
    j_idx = _lattice_indices(n_rows)
    i_idx = _lattice_indices(n_cols)
    return [(lat2d[j][i], lon2d[j][i]) for j in j_idx for i in i_idx]


def _sample_points_via_netcdf4(path):
    """netCDF4 is an OPTIONAL import, done here (not at module level) so
    this module stays importable/standalone without it (see module
    docstring) -- most wtune_predict.py callers, including scc itself as of
    2026-09-23, don't have netCDF4 installed, so this path routinely
    returns None and the ncdump-array path below does the real work.
    Reads XLAT[0]/XLONG[0] (dropping the length-1 Time axis) and takes a
    5x5 lattice directly (sample_lattice_5x5_from_grid). Returns None on
    ANY failure (no netCDF4, file won't open, vars missing) -- this is a
    best-effort fast path, not a required one.
    """
    try:
        import netCDF4
    except ImportError:
        return None
    try:
        ds = netCDF4.Dataset(str(path))
    except Exception:
        return None
    try:
        if "XLAT" not in ds.variables or "XLONG" not in ds.variables:
            return None
        lat = ds.variables["XLAT"][0]
        lon = ds.variables["XLONG"][0]
        lat2d = [[float(lat[j][i]) for i in range(lat.shape[1])] for j in range(lat.shape[0])]
        lon2d = [[float(lon[j][i]) for i in range(lon.shape[1])] for j in range(lon.shape[0])]
        return sample_lattice_5x5_from_grid(lat2d, lon2d)
    except Exception:
        return None
    finally:
        ds.close()


def parse_ncdump_dims(ncdump_text):
    """Parse the 'dimensions:' block of `ncdump` TEXT output into
    {name: size} (int). Only scans between the 'dimensions:' and
    'variables:' section headers, so a same-named global attribute or data
    value elsewhere in the file can't be mistaken for a dimension."""
    m = re.search(r"dimensions:\s*\n(.*?)\nvariables:", ncdump_text, re.S)
    if not m:
        return {}
    dims = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*(\w+)\s*=\s*(\d+)\s*;", line)
        if mm:
            dims[mm.group(1)] = int(mm.group(2))
    return dims


def parse_ncdump_2d_var(ncdump_text, var_name, n_rows, n_cols):
    """Parse a flattened (Time, south_north, west_east) variable's data
    block (Time size 1 assumed, per this module's wrfinput_d01 use) out of
    `ncdump -v <var_name>[,...]` TEXT output into a 2-D list [row][col].
    ncdump prints a variable's values flattened in row-major (C) order,
    outermost dimension first -- for (Time=1, south_north, west_east) that
    is exactly south_north-major, west_east-minor, i.e. row 0 is the
    southernmost south_north index. \\b anchors the variable name so e.g.
    'XLAT' doesn't match inside 'XLAT_U'. Returns None if the variable's
    data block isn't found, a value fails to parse, or the value count
    isn't exactly n_rows*n_cols.
    """
    m = re.search(r"\b%s\s*=\s*([^;]+);" % re.escape(var_name), ncdump_text)
    if not m:
        return None
    toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
    try:
        vals = [_parse_ncdump_number(t) for t in toks]
    except ValueError:
        return None
    if len(vals) != n_rows * n_cols:
        return None
    return [vals[r * n_cols:(r + 1) * n_cols] for r in range(n_rows)]


def _find_ncdump_binary():
    """Ordered candidates: $NETCDF/bin/ncdump, then a bare 'ncdump' resolved
    via PATH. 2026-09-23 finding: on scc, `ncdump` is NOT on PATH even after
    sourcing the site's own env.sh, but env.sh exports NETCDF
    (/nfs/software/netcdf/<ver>) and $NETCDF/bin/ncdump exists there --
    without this, every ncdump-based path here silently no-ops on scc."""
    netcdf_root = os.environ.get("NETCDF")
    if netcdf_root:
        candidate = Path(netcdf_root) / "bin" / "ncdump"
        if candidate.exists():
            return str(candidate)
    return "ncdump"


def _run_ncdump(args, timeout=60):
    """subprocess.run(_find_ncdump_binary(), *args), returning stdout text
    or None on any failure (binary missing, non-zero exit, timeout)."""
    try:
        result = subprocess.run([_find_ncdump_binary()] + list(args),
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _sample_points_via_ncdump_arrays(path):
    """`ncdump -v XLAT,XLONG <path>` -> parse_ncdump_dims +
    parse_ncdump_2d_var -> sample_lattice_5x5_from_grid. Returns None on any
    failure (ncdump unavailable/fails, dims or either variable unparseable)."""
    text = _run_ncdump(["-v", "XLAT,XLONG", str(path)])
    if text is None:
        return None
    dims = parse_ncdump_dims(text)
    n_rows = dims.get("south_north")
    n_cols = dims.get("west_east")
    if not n_rows or not n_cols:
        return None
    lat2d = parse_ncdump_2d_var(text, "XLAT", n_rows, n_cols)
    lon2d = parse_ncdump_2d_var(text, "XLONG", n_rows, n_cols)
    if lat2d is None or lon2d is None:
        return None
    return sample_lattice_5x5_from_grid(lat2d, lon2d)


def read_wrfinput_sample_points(case_dir):
    """Locates case_dir/wrfinput_d01 and returns (sample_points, source),
    source in {'netcdf4', 'ncdump_arrays', 'corner_attrs', 'none'}, trying
    in order:
      1. netCDF4 (optional import), reading XLAT[0]/XLONG[0] directly
         (_sample_points_via_netcdf4) -> 'netcdf4'.
      2. `ncdump -v XLAT,XLONG` array text parsing
         (_sample_points_via_ncdump_arrays) -> 'ncdump_arrays'. This is the
         path that actually fires on scc (netCDF4 isn't installed there
         either, as of 2026-09-23).
      3. `ncdump -h` CORNER_LATS/CORNER_LONS global attributes
         (domain_sample_points_from_ncdump) -> 'corner_attrs'. Fallback of
         LAST resort: a real real.exe wrfinput_d01, checked 2026-09-23 on
         scc (conus12), has NO CORNER_LATS/CORNER_LONS at all (only
         CEN_LAT/CEN_LON), so this path never fires for a real wrfinput --
         it exists for met_em/wrfout files, which may carry these attrs.
      4. (None, 'none') when nothing above works -- case_dir/wrfinput_d01
         is missing, or every method above failed. Callers MUST surface
         this visibly (a WARNING that sunlit weighting is disabled), not
         degrade silently -- see render_text/predict_compute.
    """
    if not case_dir:
        return None, "none"
    path = Path(case_dir) / "wrfinput_d01"
    if not path.exists():
        return None, "none"

    points = _sample_points_via_netcdf4(path)
    if points:
        return points, "netcdf4"

    points = _sample_points_via_ncdump_arrays(path)
    if points:
        return points, "ncdump_arrays"

    text = _run_ncdump(["-h", str(path)], timeout=30)
    if text is not None:
        points = domain_sample_points_from_ncdump(text)
        if points:
            return points, "corner_attrs"

    return None, "none"


# --------------------------------------------------------------------------
# RSL probe (s_normal / r_night / r_day measured from a run's own d01 step
# timings)
# --------------------------------------------------------------------------

_RSL_TIMING_RE = re.compile(
    r"Timing for main: time (\S+) on domain\s+(\d+):\s+([\d.]+)\s+elapsed seconds")

_WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"  # WRF's own Times-variable timestamp format


def parse_rsl_d01_step_seconds(path):
    """Parse a WRF rsl.error.0000-style log's 'Timing for main: time <t> on
    domain   1:   <sec> elapsed seconds' lines for domain 1 (d01) only,
    returning [(sim_datetime_or_None, elapsed_seconds), ...] in file (i.e.
    step) order. Nested-domain lines (domain 2, 3, ...) are ignored -- same
    d01-step-cost-already-includes-nested-substeps convention as the rest of
    this module (see get_d01_steps/nested_equivalent_cells). sim_datetime is
    None when <t> isn't WRF's usual '%Y-%m-%d_%H:%M:%S' Times format (the
    step's elapsed_seconds is still returned; only the day/night model needs
    the timestamp).
    """
    text = Path(path).read_text()
    out = []
    for m in _RSL_TIMING_RE.finditer(text):
        if int(m.group(2)) != 1:
            continue
        try:
            sim_dt = datetime.datetime.strptime(m.group(1), _WRF_TIME_FMT)
        except ValueError:
            sim_dt = None
        out.append((sim_dt, float(m.group(3))))
    return out


def fit_day_night_r_rad(radiation_samples, day_night_ratio):
    """radiation_samples: [(sunlit_fraction, ratio), ...], ratio = one
    radiation step's own elapsed seconds / s_normal. Simple linear
    least-squares fit ratio = a + b*sunlit_fraction (a=r_night,
    b=r_day-r_night) when the observed sunlit fractions span >=
    RADIATION_FIT_MIN_SUNLIT_SPREAD (both regimes seen) -> 'fit_day_night'.
    Otherwise the single observed level's mean is taken directly and the
    OTHER level is derived via day_night_ratio (day = night*ratio) ->
    'ratio_from_night' / 'ratio_from_day' (the source name records which
    level was actually observed).
    Returns (r_night, r_day, source), or (None, None, None) for an empty
    radiation_samples.
    """
    if not radiation_samples:
        return None, None, None
    xs = [s for s, _ in radiation_samples]
    ys = [r for _, r in radiation_samples]
    spread = max(xs) - min(xs)
    if len(radiation_samples) >= 2 and spread >= RADIATION_FIT_MIN_SUNLIT_SPREAD:
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var = sum((x - mean_x) ** 2 for x in xs)
        if var > 0:
            b = cov / var
            a = mean_y - b * mean_x
            return a, a + b, "fit_day_night"
    # Single regime: the observed mean was taken at mean sunlit fraction mean_x, not at 0 or 1.
    # Invert the model r(f) = r_night * (1 + (k - 1) * f), k = day_night_ratio, at f = mean_x.
    # Treating the observation as a pure level double-counted the sun for large domains: conus12's
    # 12Z probe (east of the domain already in daylight) gave r_day 6.8 against ~3.9 measured.
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    r_night = mean_y / (1.0 + (day_night_ratio - 1.0) * mean_x)
    source = "ratio_from_day" if mean_x >= 0.5 else "ratio_from_night"
    return r_night, r_night * day_night_ratio, source


def measure_rad_from_rsl(rsl_paths, radt_minutes, dt_seconds, sample_points,
                         default_r_rad, day_night_ratio):
    """Measure s_normal and the day/night radiation model (r_night, r_day)
    from one or more rsl.error.0000 probe logs, using the SCHEDULE (not
    cost) to identify radiation steps -- see classify_steps_detected /
    steps_per_radiation_call above and the module docstring's 'radiation'
    section. This is the fix for the bug the previous ("step time > 1.8x
    the pooled median") heuristic had: severe2023's NIGHT radiation steps
    cost only ~1.05x a normal step (well under 1.8x), so that heuristic
    silently filed them as normal steps and a night-time probe fell all the
    way back to r_rad=5 -- the schedule now identifies them correctly
    regardless of their absolute cost.

    Each file's own step 1 is dropped from both categories (startup/
    first-step outlier, not representative of steady-state cost -- WRF does
    call radiation at itimestep 1, but that step's wall time is dominated by
    MPI/exchange setup, not physics cost, so it isn't a trustworthy
    radiation-cost sample either).

    sample_points (sample_lattice_5x5 / read_wrfinput_sample_points) and
    each step's own parsed simulated time are both required to place a
    radiation step's cost against its sunlit_fraction; a step with no
    parseable simulated time, or no sample_points at all, is excluded from
    the day/night fit (its wall time still counts toward s_normal if it's a
    normal/non-radiation step).

    Returns a dict: s_normal, r_night, r_day, source (in {'fit_day_night',
    'ratio_from_night', 'ratio_from_day', 'default_unmeasured'}),
    mean_sunlit_fraction (over the samples actually used in the fit, or
    None), n_steps_pooled, n_radiation_steps (the count usable for the
    day/night fit, not just schedule-identified).
    """
    stepra = steps_per_radiation_call(radt_minutes, dt_seconds)
    pooled_normal = []
    per_file = []
    n_steps_pooled = 0
    for path in rsl_paths or []:
        file_steps = parse_rsl_d01_step_seconds(path)
        if len(file_steps) < 2:
            continue
        wall = [sec for _, sec in file_steps]
        flags = classify_steps_detected(wall, stepra)
        per_file.append((file_steps, flags))
        n_steps_pooled += len(file_steps) - 1  # step 1 dropped
        for i in range(1, len(file_steps)):  # skip index 0 == step 1
            if not flags[i]:
                pooled_normal.append(file_steps[i][1])

    if not pooled_normal:
        return {"s_normal": None, "r_night": default_r_rad, "r_day": default_r_rad,
                "source": "default_unmeasured", "mean_sunlit_fraction": None,
                "n_steps_pooled": n_steps_pooled, "n_radiation_steps": 0}

    s_normal = statistics.median(pooled_normal)

    radiation_samples = []
    for file_steps, flags in per_file:
        for i in range(1, len(file_steps)):
            if not flags[i]:
                continue
            sim_dt, sec = file_steps[i]
            if sim_dt is None or not sample_points:
                continue
            frac = sunlit_fraction(sample_points, sim_dt)
            if frac is None:
                continue
            radiation_samples.append((frac, sec / s_normal))

    if len(radiation_samples) < 2:
        return {"s_normal": s_normal, "r_night": default_r_rad, "r_day": default_r_rad,
                "source": "default_unmeasured", "mean_sunlit_fraction": None,
                "n_steps_pooled": n_steps_pooled, "n_radiation_steps": len(radiation_samples)}

    r_night, r_day, source = fit_day_night_r_rad(radiation_samples, day_night_ratio)
    mean_frac = sum(f for f, _ in radiation_samples) / len(radiation_samples)
    return {"s_normal": s_normal, "r_night": r_night, "r_day": r_day, "source": source,
            "mean_sunlit_fraction": mean_frac,
            "n_steps_pooled": n_steps_pooled, "n_radiation_steps": len(radiation_samples)}


# --------------------------------------------------------------------------
# Compute model
# --------------------------------------------------------------------------

def predict_radiation_extra(steps, radt, dt_seconds, s_normal, r_night, r_day,
                            run_start_dt, sample_points, default_r_rad):
    """Identify the PREDICTED run's own radiation steps by SCHEDULE
    (classify_steps -- the DOCUMENTED phase, since a not-yet-run prediction
    has no real rsl data to detect a phase from) and price each one
    individually against its own simulated time's sunlit fraction:
        r(t_k) = r_night + (r_day - r_night) * sunlit_fraction(t_k)
        rad_extra_s = sum_k (r(t_k) - 1) * s_normal
    when both run_start_dt (from &time_control's start_date) and
    sample_points (domain geometry) are available. Otherwise every
    radiation step uses the single uniform level (r_night+r_day)/2 instead
    (falling back further to default_r_rad if r_night/r_day are themselves
    None) -- "day/night phase unknown for this run", not a refusal to
    predict.
    Returns (rad_steps, rad_extra_s, run_mean_sunlit_fraction,
    sampling_advice): sampling_advice is {night_sim_time,
    night_sunlit_fraction, day_sim_time, day_sunlit_fraction} (the run's own
    lowest/highest-sunlit-fraction radiation steps -- the recommended probe
    windows) when the per-step model ran, else None.
    """
    if not radt or not dt_seconds:
        return 0, 0.0, None, None
    stepra = steps_per_radiation_call(radt, dt_seconds)
    flags = classify_steps(steps, stepra)
    rad_step_indices = [i for i, is_rad in enumerate(flags, start=1) if is_rad]
    rad_steps = len(rad_step_indices)
    if not rad_steps:
        return 0, 0.0, None, None

    if run_start_dt is not None and sample_points:
        rad_extra_s = 0.0
        fracs = []
        driest_night = None   # (frac, sim_dt) with the lowest sunlit fraction seen
        brightest_day = None  # (frac, sim_dt) with the highest sunlit fraction seen
        for i in rad_step_indices:
            t_k = run_start_dt + datetime.timedelta(seconds=(i - 1) * dt_seconds)
            frac = sunlit_fraction(sample_points, t_k)
            fracs.append(frac)
            r_k = r_night + (r_day - r_night) * frac
            rad_extra_s += (r_k - 1.0) * s_normal
            if driest_night is None or frac < driest_night[0]:
                driest_night = (frac, t_k)
            if brightest_day is None or frac > brightest_day[0]:
                brightest_day = (frac, t_k)
        run_mean_sunlit_fraction = sum(fracs) / len(fracs)
        sampling_advice = {
            "night_sim_time": driest_night[1], "night_sunlit_fraction": driest_night[0],
            "day_sim_time": brightest_day[1], "day_sunlit_fraction": brightest_day[0],
        }
        return rad_steps, rad_extra_s, run_mean_sunlit_fraction, sampling_advice

    r_uniform = (r_night + r_day) / 2.0 if (r_night is not None and r_day is not None) else default_r_rad
    return rad_steps, rad_steps * (r_uniform - 1.0) * s_normal, None, None


def predict_compute(nl, max_dom, steps, run_min, ranks, omp, probe_result, calibration,
                    rsl_paths=None, dt_seconds=None, sample_points=None, run_start_dt=None,
                    geometry_source="none"):
    radt = g_dom(nl, "physics", "radt", 0, default=None)
    default_r_rad = calibration.get("radiation", {}).get("r_rad_default", 5.0)
    day_night_ratio = calibration.get("radiation", {}).get(
        "r_rad_day_night_ratio", DEFAULT_R_RAD_DAY_NIGHT_RATIO)

    rad_meas = None
    if rsl_paths and radt and dt_seconds:
        rad_meas = measure_rad_from_rsl(rsl_paths, radt, dt_seconds, sample_points,
                                        default_r_rad, day_night_ratio)
    elif rsl_paths and dt_seconds:
        # radt = 0 (no radiation scheme, e.g. idealized em_les / em_heldsuarez): every step after
        # the first is a normal step. Without this branch the probe was ignored and the
        # uncalibrated no-probe constant was used instead (em_les: 0.284 s/step vs 0.033 probed).
        pooled = []
        for path in rsl_paths:
            pooled += [sec for _, sec in parse_rsl_d01_step_seconds(path)[1:]]
        if pooled:
            rad_meas = {"s_normal": statistics.median(pooled), "r_night": default_r_rad,
                        "r_day": default_r_rad, "source": "no_radiation",
                        "mean_sunlit_fraction": None, "n_steps_pooled": len(pooled),
                        "n_radiation_steps": 0}

    if rad_meas is not None:
        r_night, r_day, r_rad_source = rad_meas["r_night"], rad_meas["r_day"], rad_meas["source"]
        probe_mean_sunlit_fraction = rad_meas["mean_sunlit_fraction"]
    else:
        r_night = r_day = default_r_rad
        r_rad_source = "default_unmeasured"
        probe_mean_sunlit_fraction = None

    # A probe shorter than one radiation interval sees no radiation step (karina: every 60 steps,
    # sc26p1: every 54, probes are 30). Prefer measured typical day/night levels over the flat
    # r_rad_default of 5, which alone produced severe2023's +25% (2026-09-23 validation).
    cal_rad = calibration.get("radiation", {})
    if r_rad_source == "default_unmeasured" and cal_rad.get("r_rad_default_night") and cal_rad.get("r_rad_default_day"):
        r_night, r_day = float(cal_rad["r_rad_default_night"]), float(cal_rad["r_rad_default_day"])
        r_rad_source = "default_calibrated"

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
    elif rad_meas is not None and rad_meas["s_normal"] is not None:
        s_normal = rad_meas["s_normal"]
        source = "probe_rsl"
        noise_pct = PROBE_NOISE_RSL_PCT
        detail = {"source": source, "rsl_paths": list(rsl_paths),
                  "n_steps_pooled": rad_meas["n_steps_pooled"],
                  "n_radiation_steps": rad_meas["n_radiation_steps"]}
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

    rad_steps, rad_extra_s, run_mean_sunlit_fraction, sampling_advice = predict_radiation_extra(
        steps, radt, dt_seconds, s_normal, r_night, r_day, run_start_dt, sample_points, default_r_rad)

    compute_total_s = steps * s_normal + rad_extra_s
    noise_abs_s = compute_total_s * (noise_pct / 100.0)

    return {
        "s_normal": s_normal, "source": source, "detail": detail,
        "steps": steps, "radt_min": radt,
        "r_night": r_night, "r_day": r_day, "r_rad_source": r_rad_source,
        "probe_mean_sunlit_fraction": probe_mean_sunlit_fraction,
        "run_mean_sunlit_fraction": run_mean_sunlit_fraction,
        "sampling_advice": sampling_advice, "geometry_source": geometry_source,
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


def _rate_override_calibration(calibration, io_form, io_target, rate_bytes_per_s):
    """Return a deep-copied calibration dict with ONLY
    derived.io_form_rate_table[io_form][io_target].rate_bytes_per_s replaced
    by rate_bytes_per_s. Every other stream's rate (a different io_form --
    e.g. an auxhist stream on a different io_form, or the restart write when
    io_form_restart != io_form_history -- or a different io_target) is left
    exactly as calibrated, since only the (io_form, io_target) pair actually
    named by the override should move. Used to price the SAME set of writes
    at a measured-low/measured-high NFS write rate instead of the point
    estimate, for the --io-range calibration bracket (see build_prediction).
    """
    patched = copy.deepcopy(calibration)
    table = patched.setdefault("derived", {}).setdefault("io_form_rate_table", {})
    entry = table.setdefault(str(int(io_form)), {})
    target_entry = entry.setdefault(io_target, {})
    target_entry["rate_bytes_per_s"] = rate_bytes_per_s
    return patched


# --------------------------------------------------------------------------
# I/O model
# --------------------------------------------------------------------------

def predict_io(nl, max_dom, run_min, calibration, io_form_override, io_target,
               case_dir=None, iofields_dir=None):
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
        rate = get_effective_rate(calibration, form_val, io_target)
        iofields_vals = nl.get("time_control", {}).get("iofields_filename")

        for d in range(max_dom):
            interval_min, ikey = get_aux_interval_minutes(nl, n, d)
            if interval_min is None or interval_min <= 0:
                streams.append({"domain": d + 1, "stream": "auxhist%d" % n,
                                 "note": "io_form_auxhist%d set but no auxhist%d_interval[_s|_m] found; skipped"
                                         % (n, n)})
                continue
            begin_min, end_min = get_aux_window_minutes(nl, n, d, run_min)
            window = end_min - begin_min
            if window < 0:
                streams.append({"domain": d + 1, "stream": "auxhist%d" % n,
                                 "note": "auxhist%d_end before auxhist%d_begin; skipped" % (n, n)})
                continue
            n_frames = int(math.floor(window / interval_min)) + 1
            e_we, e_sn, e_vert = get_domain_geom(nl, d)

            entry = {"domain": d + 1, "stream": "auxhist%d" % n, "interval_min": interval_min,
                     "interval_key": ikey, "n_frames": n_frames,
                     "begin_min": begin_min, "end_min": end_min}

            # Sizing, in priority order: (1) first-principles from the per-domain
            # iofields_filename file, when it can be found and read and has '+'
            # entries for this stream number; (2) a clear 'size unknown' note
            # (no seconds estimated) when iofields_filename is referenced but
            # the file can't be resolved/read/matched; (3) a full same-domain
            # frame (bytes_per_cell_level model) when no iofields_filename is
            # given at all -- this stream is presumed to carry everything.
            sized = None
            size_note = None
            if iofields_vals:
                fname = g_dom(nl, "time_control", "iofields_filename", d)
                search_dir = iofields_dir or case_dir
                if not fname:
                    size_note = "iofields_filename referenced but no filename resolved for domain %d; size unknown" % (d + 1)
                elif not search_dir:
                    size_note = ("iofields_filename=%s referenced but no case_dir/--iofields-dir given; size unknown"
                                  % fname)
                else:
                    fpath = Path(search_dir) / fname
                    if not fpath.exists():
                        size_note = ("iofields_filename=%s referenced but not found at %s; size unknown "
                                      "(pass --iofields-dir to point at the real files)" % (fname, fpath))
                    else:
                        try:
                            sized = iofields_frame_bytes(str(fpath), n, e_we, e_sn, e_vert)
                        except (OSError, IOError) as exc:
                            size_note = "iofields file %s unreadable: %s; size unknown" % (fpath, exc)
                        if sized is None and size_note is None:
                            size_note = ("iofields file %s has no '+' entries for stream %d; size unknown"
                                          % (fpath, n))

            if sized is not None:
                bytes_frame = sized["bytes"]
                entry["bytes_per_frame"] = bytes_frame
                entry["fields"] = sized["names"]
                if sized["guessed"]:
                    entry["fields_guessed_3d"] = sized["guessed"]
                entry["sizing"] = "first-principles: sum(listed fields x points x 4 bytes), see iofields file"
            elif size_note is not None:
                entry["note"] = size_note
                streams.append(entry)
                continue
            else:
                bytes_frame = bpcl * cells_levels(e_we, e_sn, e_vert)
                entry["bytes_per_frame"] = bytes_frame
                entry["sizing"] = "full frame (no iofields_filename given for this stream)"

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
            ratio = calibration.get("restart", {}).get("restart_to_history_time_ratio", 1.0)
            # restart write seconds for a domain = ratio * (that domain's history-frame
            # bytes / the history write rate) -- MEASURED ratio (see calibration.restart),
            # not a byte-size guess. Folding the ratio into an equivalent "bytes" term
            # keeps the same bytes/rate formula as every other stream.
            restart_bytes = 0.0
            for d in range(max_dom):
                e_we, e_sn, e_vert = get_domain_geom(nl, d)
                restart_bytes += bpcl * cells_levels(e_we, e_sn, e_vert) * ratio
            rate = get_effective_rate(calibration, io_form_restart, io_target)
            entry = {"stream": "restart", "n_writes": n_restart, "bytes_per_write": restart_bytes,
                     "restart_to_history_time_ratio": ratio,
                     "note": "MEASURED restart_to_history_time_ratio (see calibration.restart)"}
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
    startup_s = calibration.get("startup_s")
    startup_s_source = calibration.get("startup_s_source")
    startup_calibrated = startup_s is not None
    startup_term = float(startup_s) if startup_calibrated else 0.0
    return {"bytes": total_bytes, "source": source, "read_s": read_s,
            "fixed_startup_s": fixed_startup_s,
            "startup_s": startup_s, "startup_s_source": startup_s_source,
            "startup_calibrated": startup_calibrated,
            "total_s": read_s + fixed_startup_s + startup_term}


# --------------------------------------------------------------------------
# Parameter table / launch line
# --------------------------------------------------------------------------

def build_param_table(config, nodes, io_form, io_target):
    # ranks/omp/nx/ny are load-bearing INSIDE the launch line itself (-np R
    # ... PE=OMP); a half-"UNKNOWN" mpirun command is worse than no command,
    # so these are REQUIRED -- refuse loudly rather than guess or print junk.
    missing = [k for k in ("ranks", "omp", "nx", "ny") if config.get(k) is None]
    if missing:
        raise PredictError(
            "build_param_table: config is missing required field(s) %s (got %r); "
            "the launch line cannot be built without them" % (missing, config))
    ranks, omp, nx, ny = config["ranks"], config["omp"], config["nx"], config["ny"]

    # numtiles/dist/io_form are NOT load-bearing for the mpirun skeleton itself
    # (dist only ever adds/omits one optional trailing flag; numtiles/io_form
    # are reported separately, not embedded in launch_line's positional
    # values), so an absent value is marked UNKNOWN rather than silently
    # defaulted to a plausible-looking but possibly WRONG value -- see the
    # 2026-09-23 scc blind-test bug: this used to default dist to "cyclic"
    # and print a --rank-by node line for a config whose real (measured)
    # dist was "block", i.e. a launch line for a DIFFERENT run than the one
    # actually being recommended.
    dist = config.get("dist")
    numtiles = config.get("numtiles")
    io_form_display = io_form if io_form is not None else "UNKNOWN"

    tpn = ranks // nodes if nodes else None
    warnings = []
    if nodes and ranks % nodes != 0:
        warnings.append("ranks (%d) not evenly divisible by nodes (%d); TPN rounded down to %d" % (
            ranks, nodes, tpn))

    # Matches campaign/sites/scc.sh site_launch() exactly: --rank-by node reproduces
    # WRF's cyclic across-node rank order for dist=cyclic; block is OMPI5's own
    # default rank-by, so the block case adds no extra flag. dist=UNKNOWN adds
    # neither -- the line is INCOMPLETE, not equivalent to block, hence the note.
    launch = ("mpirun -np %d --map-by ppr:%d:node:PE=%d --bind-to core"
              % (ranks, tpn if tpn else 0, omp))
    if dist == "cyclic":
        launch += " --rank-by node"
    elif dist is None:
        warnings.append("dist not provided; launch line incomplete (cannot tell whether "
                         "--rank-by node is needed -- see campaign/sites/scc.sh site_launch())")

    env = {"OMP_NUM_THREADS": str(omp), "OMP_STACKSIZE": "1G",
           "OMP_PLACES": "cores", "OMP_PROC_BIND": "close", "ulimit -s": "unlimited"}
    numtiles_display = numtiles if numtiles is not None else "UNKNOWN"
    if numtiles is None:
        warnings.append("numtiles not provided; namelist numtiles line is UNKNOWN, not a guessed default")
    namelist_lines = ["nproc_x = %d," % nx, "nproc_y = %d," % ny, "numtiles = %s," % numtiles_display]

    return {"nproc_x": nx, "nproc_y": ny, "numtiles": numtiles_display,
            "ranks": ranks, "omp": omp, "nodes": nodes, "ranks_per_node": tpn,
            "dist": dist if dist is not None else "UNKNOWN",
            "io_form": io_form_display, "io_target": io_target,
            "launch_line": launch, "env": env, "namelist_lines": namelist_lines,
            "warning": "; ".join(warnings) if warnings else None}


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
        # dist is NOT defaulted (see build_param_table): a config dict or CLI
        # invocation that doesn't say block/cyclic must produce dist=UNKNOWN
        # downstream, never a silently-assumed "cyclic".
        dist = raw.get("dist", getattr(args, "dist", None))
    else:
        if args.nx is None or args.ny is None:
            raise PredictError("must give --config JSON, or --nx and --ny "
                                "(with optional --ranks/--omp/--numtiles/--dist)")
        nx, ny = int(args.nx), int(args.ny)
        omp = int(args.omp) if args.omp is not None else 1
        numtiles = int(args.numtiles) if args.numtiles is not None else 1
        ranks = int(args.ranks) if args.ranks is not None else nx * ny
        dist = getattr(args, "dist", None)
    return {"ranks": ranks, "omp": omp, "nx": nx, "ny": ny, "numtiles": numtiles, "dist": dist}


def build_prediction(namelist_path, case_dir, nodes, config, ledger_paths,
                      calibration_path, io_form_override, io_target, case_name=None,
                      iofields_dir=None, rsl_paths=None):
    nl = parse_namelist(namelist_path)
    max_dom = get_max_dom(nl)
    steps, run_s, dt, steps_note = get_d01_steps(nl)
    run_min = run_s / 60.0
    run_hours = run_s / 3600.0

    calibration = load_calibration(calibration_path)

    ledger_rows = load_ledger_rows(ledger_paths)
    # case_name lets the caller give the case's LEDGER identity explicitly
    # (case_base in the --probe CSV) when it differs from case_dir's own
    # basename -- e.g. a case whose prep directory is named after its data
    # source (doi4-lafe2017) rather than its short campaign name (lafe).
    # Falling back silently to the directory basename here is exactly the
    # bug this parameter exists to let the caller avoid: with no override,
    # resolve_probe's case_base match fails, probe_result is None end to
    # end, and the model drops to no_probe_fallback despite the ledger
    # holding an exact, measured row for this case/config.
    name = case_name or (Path(case_dir).name if case_dir else "unknown")
    probe_result = resolve_probe(ledger_rows, name, config)

    # Domain geometry (for the day/night sunlit-fraction radiation model) and
    # the run's own absolute start time (to place each PREDICTED radiation
    # step at a real simulated time) -- both optional; predict_compute
    # degrades gracefully (uniform r_night/r_day level, no sampling advice)
    # when either is unavailable (see that function's docstring and
    # read_wrfinput_sample_points' geometry_source values).
    sample_points, geometry_source = read_wrfinput_sample_points(case_dir)
    run_start_dt = get_time_control_datetime(nl, "start")

    compute = predict_compute(nl, max_dom, steps, run_min, config["ranks"], config["omp"],
                              probe_result, calibration, rsl_paths=rsl_paths, dt_seconds=dt,
                              sample_points=sample_points, run_start_dt=run_start_dt,
                              geometry_source=geometry_source)
    segments = segment_compute(compute, run_hours, dt)

    io_target = io_target or "nfs"
    io_form_history_effective = io_form_override if io_form_override is not None else \
        g(nl, "time_control", "io_form_history", 2)
    io = predict_io(nl, max_dom, run_min, calibration, io_form_override, io_target,
                    case_dir=case_dir, iofields_dir=iofields_dir)

    init = predict_init(case_dir, nl, calibration)

    drift_pct = calibration.get("drift_band_pct", 5.0)
    total_s = compute["compute_total_s"] + io["total_s"] + init["total_s"]

    # I/O rate range (calibration.history_write_rate_MBps_low/_high): when
    # calibrated, replace the generic drift-on-everything range with an
    # explicit [fast I/O .. slow I/O] bracket for the I/O component, and
    # scope the drift band to compute only (see module docstring's 'range').
    io_low_mbps = calibration.get("history_write_rate_MBps_low")
    io_high_mbps = calibration.get("history_write_rate_MBps_high")
    io_range = None
    if io_low_mbps is not None and io_high_mbps is not None:
        cal_slow = _rate_override_calibration(
            calibration, io_form_history_effective, io_target, io_low_mbps * 1e6)
        cal_fast = _rate_override_calibration(
            calibration, io_form_history_effective, io_target, io_high_mbps * 1e6)
        io_slow = predict_io(nl, max_dom, run_min, cal_slow, io_form_override, io_target,
                             case_dir=case_dir, iofields_dir=iofields_dir)
        io_fast = predict_io(nl, max_dom, run_min, cal_fast, io_form_override, io_target,
                             case_dir=case_dir, iofields_dir=iofields_dir)
        io_range = {"low_mbps": io_low_mbps, "high_mbps": io_high_mbps,
                    "fast_total_s": io_fast["total_s"], "slow_total_s": io_slow["total_s"]}

    if io_range is not None:
        compute_drift_abs_s = compute["compute_total_s"] * (drift_pct / 100.0)
        drift_abs_s = compute_drift_abs_s
        low = (max(0.0, compute["compute_total_s"] - compute_drift_abs_s)
               + io_range["fast_total_s"] + init["total_s"])
        high = (compute["compute_total_s"] + compute_drift_abs_s
                + io_range["slow_total_s"] + init["total_s"])
        range_source = ("I/O: measured %.1f-%.1f MB/s write-rate bracket; "
                        "compute: +-%.0f%% drift band (NOT VALIDATED)"
                        % (io_low_mbps, io_high_mbps, drift_pct))
    else:
        drift_abs_s = total_s * (drift_pct / 100.0)
        range_abs_s = math.sqrt(compute["noise_abs_s"] ** 2 + io["noise_abs_s"] ** 2 + drift_abs_s ** 2)
        low, high = max(0.0, total_s - range_abs_s), total_s + range_abs_s
        range_source = None

    params = build_param_table(config, nodes, io_form_history_effective, io_target)

    return {
        "case_name": name, "namelist": str(namelist_path),
        "max_dom": max_dom, "steps": steps, "run_s": run_s, "run_min": run_min,
        "run_hours": run_hours, "dt_s": dt, "steps_note": steps_note,
        "compute": compute, "segments": segments, "io": io, "io_range": io_range, "init": init,
        "total_s": total_s, "drift_pct": drift_pct, "drift_abs_s": drift_abs_s,
        "range_low_s": low, "range_high_s": high, "range_source": range_source,
        "param_table": params, "nodes": nodes,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt_hms(seconds):
    # Round to display precision FIRST, so e.g. 419.9996s doesn't decompose
    # into "6m59.9996s" and then print-round up to the impossible "6m60.00s".
    seconds = round(max(0.0, seconds), 2)
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
        out.append("  + radiation surcharge: radt=%.1fmin  rad_steps=%d  r_night=%.2f  r_day=%.2f "
                    "(source=%s)  -> %s"
                    % (c["radt_min"], c["rad_steps"], c["r_night"], c["r_day"], c["r_rad_source"],
                       _fmt_hms(c["rad_extra_s"])))
        out.append("    domain geometry source: %s" % c["geometry_source"])
        if c["geometry_source"] == "none":
            out.append("    WARNING: no domain geometry found for wrfinput_d01 (tried netCDF4, "
                        "ncdump -v XLAT,XLONG, and ncdump -h CORNER_LATS/CORNER_LONS) -- "
                        "sunlit-fraction weighting is DISABLED; every radiation step uses the "
                        "uniform (r_night+r_day)/2 level instead of its own simulated time's "
                        "sunlit fraction")
        if c["run_mean_sunlit_fraction"] is not None:
            out.append("    predicted run mean sunlit fraction (over its own radiation steps): %.2f"
                        % c["run_mean_sunlit_fraction"])
        if c["probe_mean_sunlit_fraction"] is not None:
            out.append("    probe mean sunlit fraction (radiation steps used in the r_night/r_day fit): %.2f"
                        % c["probe_mean_sunlit_fraction"])
        if c["sampling_advice"]:
            sa = c["sampling_advice"]
            out.append("    recommended probe windows -- night: sim time ~%s (sunlit=%.2f)  "
                        "day: sim time ~%s (sunlit=%.2f)"
                        % (sa["night_sim_time"], sa["night_sunlit_fraction"],
                           sa["day_sim_time"], sa["day_sunlit_fraction"]))
    out.append("  compute total: %s  (+-%.1f%% = +-%s, source=%s)"
                % (_fmt_hms(c["compute_total_s"]), c["noise_pct"], _fmt_hms(c["noise_abs_s"]), c["source"]))
    if pred["segments"]:
        out.append("  segmented (>%.0fh simulated, %.0fh chunks, NOT VALIDATED drift):"
                    % (LONG_RUN_HOURS, SEGMENT_HOURS))
        for seg in pred["segments"]:
            out.append("    %-12s steps=%-6d %s" % (seg["sim_time"], seg["steps"], _fmt_hms(seg["compute_s"])))

    io = pred["io"]
    out.append("I/O: total %s" % _fmt_hms(io["total_s"]))
    if pred.get("io_range"):
        ior = pred["io_range"]
        out.append("  I/O rate range: fast(%.1f MB/s)=%s .. slow(%.1f MB/s)=%s"
                    % (ior["high_mbps"], _fmt_hms(ior["fast_total_s"]),
                       ior["low_mbps"], _fmt_hms(ior["slow_total_s"])))
    for st in io["streams"]:
        if "seconds" in st:
            label = st.get("stream", "?")
            dom = (" d%02d" % st["domain"]) if "domain" in st else ""
            extra = ""
            if st.get("fields"):
                extra = " [fields: %s]" % ",".join(st["fields"])
                if st.get("fields_guessed_3d"):
                    extra += " (guessed 3-D: %s)" % ",".join(st["fields_guessed_3d"])
            out.append("  %s%s: frames=%d bytes/frame=%.3e -> %s%s"
                        % (label, dom, st.get("n_frames", st.get("n_writes", 0)),
                           st.get("bytes_per_frame", st.get("bytes_per_write", 0.0)),
                           _fmt_hms(st["seconds"]), extra))
        elif "note" in st:
            out.append("  %s: %s" % (st.get("stream", "?"), st["note"]))

    init = pred["init"]
    if init["startup_calibrated"]:
        out.append("init (read+startup): %s  [%s]  startup=%s [%s]"
                    % (_fmt_hms(init["total_s"]), init["source"],
                       _fmt_hms(init["startup_s"]), init["startup_s_source"] or "calibrated"))
    else:
        out.append("init (read): %s  [%s]  startup: UNCALIBRATED"
                    % (_fmt_hms(init["total_s"]), init["source"]))

    out.append("")
    if pred.get("range_source"):
        out.append("TOTAL predicted wall: %s  (range %s .. %s -- %s)"
                    % (_fmt_hms(pred["total_s"]), _fmt_hms(pred["range_low_s"]),
                       _fmt_hms(pred["range_high_s"]), pred["range_source"]))
    else:
        out.append("TOTAL predicted wall: %s  (range %s .. %s, includes +-%.0f%% NOT VALIDATED drift band)"
                    % (_fmt_hms(pred["total_s"]), _fmt_hms(pred["range_low_s"]),
                       _fmt_hms(pred["range_high_s"]), pred["drift_pct"]))

    out.append("")
    out.append("--- recommended parameters ---")
    p = pred["param_table"]
    out.append("nproc_x=%d nproc_y=%d numtiles=%s  ranks=%d x omp=%d  nodes=%d  ranks/node=%s  dist=%s"
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
    p.add_argument("--dist", choices=["block", "cyclic"], default=None,
                   help="srun/mpirun rank distribution (block or cyclic; NOT defaulted -- if "
                        "omitted and --config has no 'dist', the launch line prints dist=UNKNOWN "
                        "and omits --rank-by rather than guessing)")
    p.add_argument("--probe", action="append", default=None,
                   help="strict_campaign ledger CSV (repeatable)")
    p.add_argument("--probe-rsl", action="append", default=None,
                   help="path to a probe run's rsl.error.0000 (repeatable); measures r_rad "
                        "(and, when --probe gives no matching ledger row, s_normal) from the "
                        "probe's own d01 step timings instead of the hard-coded r_rad default "
                        "(see measure_rad_from_rsl -- ASSUMES the probe window has no mid-run "
                        "history/restart writes)")
    p.add_argument("--case-name", default=None,
                   help="the case's ledger identity (case_base in --probe's CSV), when it "
                        "differs from case_dir's own basename (default: case_dir's basename) "
                        "-- e.g. a case whose prep dir is /nfs/data/wrf/doi4-lafe2017 but whose "
                        "ledger case_base is 'lafe'; without this, --probe rows for that case "
                        "silently fail to match and prediction falls back to no_probe_fallback")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                   help="calibration JSON (default machine/calibration/scc.json)")
    p.add_argument("--io-form", type=int, default=None, choices=[2, 11, 14],
                   help="override io_form_history (default: read from namelist)")
    p.add_argument("--io-target", choices=["nfs", "local"], default="nfs",
                   help="write target (default nfs; local is UNMEASURED)")
    p.add_argument("--iofields-dir", default=None,
                   help="directory holding the real iofields_filename field-list file(s) "
                        "(default: case_dir); used to size field-subset aux streams exactly")
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
            io_form_override=args.io_form, io_target=args.io_target,
            case_name=args.case_name, iofields_dir=args.iofields_dir,
            rsl_paths=args.probe_rsl)
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
