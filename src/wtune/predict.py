#!/usr/bin/env python3
"""wtune/predict.py -- offline wall-time predictor for a WRF run of any length, without running WRF.

Standalone (stdlib only, never imports wtune/cli.py); `wtune predict` loads it lazily. Model below,
constants in machine/calibration/scc.json:

steps      d01 steps = run length / d01 time_step (+ fract). Nested substeps are inside the d01
           step's wall time (maria1km: 432 d01 steps hold 2160 d02 substeps).
compute    s_normal, first available: exact --probe ledger match ('measured'); same case at
           another rank count, ideal strong scaling ('extrapolated'); median schedule-identified
           non-radiation step of --probe-rsl logs ('probe_rsl'); calibration
           fallback_s_per_cell_step_ns ('no_probe_fallback', uncalibrated).
radiation  Steps by WRF's call schedule, not by cost: stepra = NINT(radt*60/dt), step i is
           radiation iff i == 1 or i % stepra == 0 (data-detected phase for a real rsl log).
           r(t) = r_night + (r_day - r_night) * sunlit_fraction(t), sunlit over a 5x5 domain
           lattice (solar_cos_zenith). r_night/r_day: least-squares fit when the probe's sunlit
           spread >= 0.5 ('fit_day_night'); one regime + day/night ratio 1.87
           ('ratio_from_night'/'_day'); else r_rad_default 5.0 ('default_unmeasured').
           extra = sum_k (r(t_k) - 1) * s_normal over the predicted radiation steps; uniform
           (r_night + r_day)/2 without start time or geometry. Validation 2026-09-23 (5 runs):
           night 1.92-2.52, day 3.42-4.71.
I/O        history: floor(run_min / history_interval) + 1 frames per domain with io_form != 0,
           cells x bytes_per_cell_level / rate[io_form][io_target]. auxhistN over begin..end,
           sized from iofields_filename '+' fields (4 bytes/point), 'size unknown' when that
           file cannot be matched, a full frame without one. restart: floor(run_min /
           restart_interval) writes per domain, each restart_to_history_time_ratio (4.28, job
           427) x that domain's history-frame time.
init       (wrfinput/wrfbdy bytes, else an estimate) / read rate + fixed_startup_s + startup_s
           (measured launch..finalise; printed UNCALIBRATED when absent).
drift      +-drift_band_pct (NOT VALIDATED) on the total, or on compute only when the
           history_write_rate_MBps_low/_high I/O bracket is calibrated.
range      root-sum-square of probe noise, drift and I/O uncertainty; with the I/O bracket:
           compute +- drift, I/O at the fast..slow rate, + init (NFS writes varied 2.4x)."""

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
from typing import Any

import wtune.machine
from wtune import runlength as runlen  # repo root: resolve_run_seconds, the single-sourced run_*
                                   # keys vs. start/end span resolver (stdlib only)

HERE = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION = HERE.parent / "machine" / "calibration" / "scc.json"  # HERE is src/; machine/ stays at repo root


def calibration_path() -> str:
    """This machine's calibration file (wtune.machine), else the built-in scc one."""
    return wtune.machine.calibration() or str(DEFAULT_CALIBRATION)


def cores_per_node(calibration=None):
    """Cores per node of this machine: its machine setting when one is set, else the
    calibration's, else the built-in default. (f1-arm, 144 cores, has no calibration of its own
    and was sized by scc.json's 72.)"""
    if os.environ.get(wtune.machine.ENV["cores_per_node"], "").strip():
        return wtune.machine.cores_per_node()
    v = (calibration or {}).get("cores_per_node")
    return int(v) if v else wtune.machine.cores_per_node()
SEGMENT_HOURS = 6.0   # long-run compute display chunking
LONG_RUN_HOURS = 12.0  # threshold above which the compute row is segmented

PROBE_NOISE_MEASURED_PCT = wtune.machine.MEASURED["probe_noise_measured_pct"]   # built-in
PROBE_NOISE_EXTRAPOLATED_PCT = wtune.machine.MEASURED["probe_noise_extrapolated_pct"]
PROBE_NOISE_FALLBACK_PCT = wtune.machine.MEASURED["probe_noise_fallback_pct"]
# A probe_rsl s_normal is one run's own sample, so its band is a little wider than a
# measured row's (a judgment call, not a measured uncertainty).
PROBE_NOISE_RSL_PCT = wtune.machine.MEASURED["probe_noise_rsl_pct"]

# Minimum sunlit-fraction spread (max - min) for a day/night least-squares fit; below it the
# probe saw one regime and the other level comes from the calibrated day/night ratio.
RADIATION_FIT_MIN_SUNLIT_SPREAD = 0.5  # sunlit fraction, max - min

DEFAULT_R_RAD_DAY_NIGHT_RATIO = wtune.machine.MEASURED["r_rad_day_night_ratio"]   # built-in;
                                     # see module docstring; MEASURED mean of
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
    if low in (".true.", ".t.", "true", "t"):
        return True
    if low in (".false.", ".f.", "false", "f"):
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


# 'key =' or the indexed form 'key(2) =' (kept as its own key "key(2)", never merged into the
# previous key's list).
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_]\w*\s*(?:\(\s*\d+\s*\)\s*)?=(?!=)")


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
        key = re.sub(r"\s+", "", m.group(0).split("=", 1)[0])
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        out.append((key, line[m.end():value_end]))
    return out


def _is_plain_scalar(tok):
    """True for a number/logical or a fully quoted string (what 'n*value' may repeat)."""
    tok = tok.strip()
    quoted = len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"')
    return quoted or not isinstance(_parse_scalar(tok), str)


def _append_values(bucket, raw_value_text):
    rest = raw_value_text.strip().rstrip(",").strip()
    if not rest:
        return
    for tok in rest.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        # Fortran repeat count 'n*value' (e_we = 2*100); a stray '*' in a string is left alone.
        m = re.fullmatch(r"(\d+)\*(.+)", tok)
        if m and _is_plain_scalar(m.group(2)):
            bucket.extend([_parse_scalar(m.group(2))] * int(m.group(1)))
            continue
        bucket.append(_parse_scalar(tok))


def parse_namelist(path: str | Path) -> dict[str, dict[str, list]]:
    """Minimal Fortran namelist reader -> {group: {key: [scalar, ...]}}, names lowercased.
    Several 'key = v[, v...]' assignments may share a line; a line without a top-level 'key=' that
    is not a group header or '/' continues the previous key's list (wrapped per-domain lists)."""
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
            line = line[m.end():].strip()  # assignments may follow the header on its line
            if not line:
                continue

        if line.startswith("/"):
            group = None
            last_key = None
            continue

        if group is None:
            continue

        # an unquoted trailing '/' ends the group; it is not part of the last value
        ends_group = _mask_quotes(line).endswith("/")
        if ends_group:
            line = line[:-1].rstrip()

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
        if ends_group:
            group = None
            last_key = None

    return groups


def g(nl: dict, group: str, key: str, default: Any = None) -> Any:
    """Namelist-global scalar (first value only)."""
    vals = nl.get(group, {}).get(key)
    if not vals:
        return default
    return vals[0]


def g_dom(nl: dict, group: str, key: str, dom_index: int, default: Any = None) -> Any:
    """Per-domain value with WRF's broadcast-last-value convention: if fewer
    values are given than max_dom, every remaining domain reuses the last one.
    """
    vals = nl.get(group, {}).get(key)
    if not vals:
        return default
    if dom_index < len(vals):
        return vals[dom_index]
    return vals[-1]


def get_max_dom(nl: dict) -> int:
    return int(g(nl, "domains", "max_dom", 1) or 1)


def get_domain_geom(nl: dict, dom_index: int) -> tuple[float, float, float]:
    e_we = g_dom(nl, "domains", "e_we", dom_index)
    e_sn = g_dom(nl, "domains", "e_sn", dom_index)
    e_vert = g_dom(nl, "domains", "e_vert", dom_index)
    if e_we is None or e_sn is None or e_vert is None:
        raise PredictError(
            "domain %d: missing e_we/e_sn/e_vert in &domains" % (dom_index + 1))
    return float(e_we), float(e_sn), float(e_vert)


def cells_levels(e_we: float, e_sn: float, e_vert: float) -> float:
    return e_we * e_sn * e_vert


def get_time_control_datetime(nl: dict, prefix: str) -> datetime.datetime | None:
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


def get_run_seconds(nl: dict) -> float:
    """Run length (s): run_days/hours/minutes/seconds when their sum > 0, else the d01
    start -> end span (WRF's own fallback). Delegates to wtune.runlength.resolve_run_seconds;
    raises PredictError instead of its ValueError."""
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


def get_timestep_seconds(nl: dict) -> float:
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


def get_d01_steps(nl: dict) -> tuple[int, float, float, str | None]:
    run_s = get_run_seconds(nl)
    dt = get_timestep_seconds(nl)
    exact = run_s / dt
    steps = int(round(exact))
    note = None
    if abs(exact - steps) > 1e-6:
        note = ("run length (%.3fs) is not an exact multiple of time_step "
                 "(%.3fs); rounded %.4f steps to %d" % (run_s, dt, exact, steps))
    return steps, run_s, dt, note


def adaptive_step_bounds(nl: dict, run_s: float) -> tuple[int, int, float, float] | None:
    """use_adaptive_time_step: WRF picks d01's dt each step within about [min_time_step,
    max_time_step] (dyn_em/start_em.F: -1 is NINT(3 or 8 * min(dx,dy)/1000) s; *_den divides;
    step_to_output_time may cut below the minimum), so the step count is only bounded ->
    (fewest steps, most steps, min dt, max dt); None without adaptive dt."""
    if not bool(g(nl, "domains", "use_adaptive_time_step", False)):
        return None
    dx = float(g(nl, "domains", "dx", 0) or 0)
    d_km = min(dx, float(g(nl, "domains", "dy", dx) or dx)) / 1000.0

    def bound(key, k):
        v = float(g_dom(nl, "domains", key, 0, -1) or -1)
        if v <= 0:
            v = float(math.floor(k * d_km + 0.5))            # Fortran NINT, not half-to-even
        den = float(g_dom(nl, "domains", key + "_den", 0, 0) or 0)   # adapt_timestep_em.F
        return v / den if den > 0 else v
    lo_dt, hi_dt = bound("min_time_step", 3), bound("max_time_step", 8)
    if lo_dt <= 0 or hi_dt < lo_dt:
        return None
    return math.ceil(run_s / hi_dt), math.ceil(run_s / lo_dt), lo_dt, hi_dt


def parent_time_step_ratio(nl: dict, dom_index: int, default: int = 1) -> int:
    v = g_dom(nl, "domains", "parent_time_step_ratio", dom_index, default)
    return int(v) if v is not None else default


def nested_equivalent_cells(nl: dict, max_dom: int) -> float:
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


def get_history_interval_minutes(nl: dict, dom_index: int) -> tuple[float | None, str | None]:
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


def get_aux_interval_minutes(nl: dict, n: int, dom_index: int) -> tuple[float | None, str | None]:
    for suffix, scale in (("_s", 1.0 / 60.0), ("_m", 1.0), ("", 1.0)):
        key = "auxhist%d_interval%s" % (n, suffix)
        vals = nl.get("time_control", {}).get(key)
        if vals:
            v = g_dom(nl, "time_control", key, dom_index)
            if v is not None:
                return float(v) * scale, key
    return None, None


def get_aux_window_minutes(nl: dict, n: int, dom_index: int, run_minutes: float
                           ) -> tuple[float, float]:
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


def field_points(name: str, e_we: int, e_sn: int, e_vert: int) -> tuple[int, bool]:
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


def parse_iofields_file(path: str | Path) -> dict[int, set[str]]:
    """Parse a WRF iofields_filename file -> {stream_num: set(names)}. Entries are
    '<+|-> : <h|i> : <streams> : <fields>' ('+' adds fields to a stream, '-' removes them); a
    non-matching line continues the previous entry's field list; '#'/'!' lines are comments."""
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


def iofields_frame_bytes(path: str | Path, stream_num: int, e_we: int, e_sn: int, e_vert: int
                         ) -> dict | None:
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

def load_calibration(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise PredictError("cannot read calibration file %s: %s" % (path, exc))


def get_effective_rate(calibration: dict, io_form: int, io_target: str) -> dict:
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

def load_ledger_rows(paths: list[str]) -> list[dict]:
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


def resolve_probe(ledger_rows: list[dict], case_name: str, config: dict) -> dict | None:
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

    # nearest row and scaling by CORES (ranks x omp): 36 ranks x omp 2 is the same 72 cores as
    # 72 ranks x omp 1, not half of them. A row without omp_threads counts as omp 1.
    def _row_cores(r):
        return (_row_int(r, "n_ranks") or 0) * (_row_int(r, "omp_threads") or 1)

    cores = ranks * (omp or 1)
    best = min(same_case, key=lambda r: abs(_row_cores(r) - cores))
    from_ranks = _row_int(best, "n_ranks")
    from_s = _row_float(best, "s_per_step_median")
    if not from_ranks or from_s is None:
        return None
    s_normal = from_s * (_row_cores(best) / float(cores))
    return {"s_normal": s_normal, "source": "extrapolated",
            "from_ranks": from_ranks, "to_ranks": ranks,
            "from_config": {"n_ranks": from_ranks,
                             "omp_threads": _row_int(best, "omp_threads"),
                             "nproc_x": _row_int(best, "nproc_x"),
                             "nproc_y": _row_int(best, "nproc_y"),
                             "numtiles": _row_int(best, "numtiles")}}


# --------------------------------------------------------------------------
# WRF radiation call schedule (ported, not imported, from tools/state_timeline.py, whose
# imports this standalone module may not pull in)
# --------------------------------------------------------------------------

def steps_per_radiation_call(radt_minutes: float, dt_seconds: float) -> int:
    """STEPRA = NINT(radt_minutes*60/dt_seconds), WRF's radiation-call cadence (Fortran NINT:
    halves away from zero; Python round() gets conus12's 12.5 wrong). Floored at 1. Same as
    tools/state_timeline.steps_per_radiation_call."""
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if radt_minutes <= 0:
        raise ValueError("radt_minutes must be > 0")
    return max(1, int(math.floor(radt_minutes * 60.0 / dt_seconds + 0.5)))


def detect_radiation_phase(step_wall_times: list[float], stepra: int) -> int:
    """Phase p in [0, stepra): step i > 1 is radiation when i % stepra == p, the phase with
    the highest MEDIAN wall time (robust to write spikes). The documented phase 0 does not always
    match a real log (matthew: phase 1); a relative comparison also catches severe2023's ~1.05x
    night radiation. Phase 0 with fewer than 2*stepra steps or indistinguishable phases."""
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


def classify_steps_detected(step_wall_times: list[float], stepra: int) -> list[bool]:
    """[is_radiation, ...] (1-indexed, same length as step_wall_times) using
    the DATA-DETECTED phase (detect_radiation_phase) -- for a real
    --probe-rsl log. Ported from tools/state_timeline.py."""
    p = detect_radiation_phase(step_wall_times, stepra)
    return [(i == 1) or (stepra > 0 and i % stepra == p) for i in range(1, len(step_wall_times) + 1)]


def classify_steps(n_steps: int, stepra: int) -> list[bool]:
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

def solar_cos_zenith(lat_deg: float, lon_deg: float, dt_utc: datetime.datetime) -> float:
    """cos(solar zenith) at (lat, lon) and naive-UTC dt: Cooper declination 23.45 *
    sin(360/365 * (284 + doy)), hour angle 15 * (UTC + lon/15 - 12). Equation of time and
    refraction omitted. > 0 means sunlit."""
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


def sample_lattice_5x5(corner_sw: tuple, corner_nw: tuple, corner_ne: tuple, corner_se: tuple,
                       center: tuple) -> list[tuple[float, float]]:
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


def sunlit_fraction(sample_points: list | None, dt_utc: datetime.datetime) -> float | None:
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


def parse_wrfinput_corners(ncdump_text: str) -> dict | None:
    """CEN_LAT/CEN_LON/CORNER_LATS/CORNER_LONS from `ncdump -h` text -> None without the
    centre, else {'cen': (lat, lon), 'corners': (sw, nw, ne, se) or None}. 16-value corner arrays:
    mass-point corner c at index c*4 (assumed from WRF's ordering, not verified on a file); a
    4-value array is taken as the corners directly."""
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


def domain_sample_points_from_ncdump(ncdump_text: str) -> list[tuple[float, float]] | None:
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
# Real XLAT/XLONG grid sampling: a real real.exe wrfinput_d01 has no CORNER_LATS/LONS (scc,
# 2026-09-23), so corner sampling silently disabled the day/night model; corners stay a last
# resort for met_em/wrfout files.
# --------------------------------------------------------------------------

def _lattice_indices(n, count=5):
    """count evenly spaced integer indices in [0, n-1], including BOTH
    edges (index 0 and n-1) whenever count >= 2. Used to pick a 5x5 lattice
    directly out of a real XLAT/XLONG grid (no interpolation needed -- the
    grid already has a real value at every index)."""
    if n <= 1:
        return [0] * count
    return [round(t * (n - 1) / (count - 1)) for t in range(count)]


def sample_lattice_5x5_from_grid(lat2d: Any, lon2d: Any) -> list[tuple[float, float]]:
    """25 (lat, lon) points read from real 2-D XLAT/XLONG ([south_north][west_east], row 0
    southernmost) at 5 x 5 evenly spaced indices including the edges -- real grid values, no
    interpolation (unlike sample_lattice_5x5)."""
    n_rows = len(lat2d)
    n_cols = len(lat2d[0]) if n_rows else 0
    j_idx = _lattice_indices(n_rows)
    i_idx = _lattice_indices(n_cols)
    return [(lat2d[j][i], lon2d[j][i]) for j in j_idx for i in i_idx]


def _sample_points_via_netcdf4(path):
    """5x5 lattice from XLAT[0]/XLONG[0] via netCDF4, imported lazily (the module stays
    standalone; scc lacks netCDF4, so the ncdump path usually does the work). None on any failure."""
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
        import numpy as np
        lat2d = np.ma.filled(np.ma.asarray(lat, dtype=float), np.nan).tolist()
        lon2d = np.ma.filled(np.ma.asarray(lon, dtype=float), np.nan).tolist()
        return sample_lattice_5x5_from_grid(lat2d, lon2d)
    except Exception:
        return None
    finally:
        ds.close()


def parse_ncdump_dims(ncdump_text: str) -> dict[str, int]:
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


def parse_ncdump_2d_var(ncdump_text: str, var_name: str, n_rows: int, n_cols: int
                        ) -> list[list[float]] | None:
    """A (Time=1, south_north, west_east) variable from `ncdump -v` text -> 2-D list
    [row][col], row 0 southernmost (ncdump prints row-major). None when the block is missing, a
    value does not parse, or the count is not n_rows*n_cols."""
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


def read_wrfinput_sample_points(case_dir: str | Path) -> tuple[list | None, str]:
    """(sample_points, source) for case_dir/wrfinput_d01, trying netCDF4 XLAT/XLONG
    ('netcdf4'), `ncdump -v XLAT,XLONG` ('ncdump_arrays', what fires on scc), `ncdump -h` corner
    attrs ('corner_attrs'; real real.exe wrfinput has none, met_em/wrfout may), else (None,
    'none'). Callers must warn visibly that sunlit weighting is disabled on 'none'."""
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


def parse_rsl_d01_step_seconds(path: str | Path) -> list[tuple[datetime.datetime | None, float]]:
    """d01 'Timing for main' lines of an rsl log -> [(sim_datetime or None, seconds)] in step
    order; nest lines ignored (the d01 step includes them). sim_datetime None for an unparseable
    time stamp."""
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


def fit_day_night_r_rad(radiation_samples: list[tuple[float, float]], day_night_ratio: float
                        ) -> tuple[float | None, float | None, str | None]:
    """radiation_samples [(sunlit_fraction, step s / s_normal)] -> (r_night, r_day, source).
    Least squares ratio = a + b*f when the sunlit spread >= RADIATION_FIT_MIN_SUNLIT_SPREAD
    ('fit_day_night'); else the one observed level, the other via day_night_ratio
    ('ratio_from_night'/'ratio_from_day'). (None, None, None) without samples."""
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
    # Single regime: invert r(f) = r_night * (1 + (k - 1) * f) at the observed mean f, not at 0/1
    # (as a pure level, conus12's 12Z probe gave r_day 6.8 vs ~3.9 measured).
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    r_night = mean_y / (1.0 + (day_night_ratio - 1.0) * mean_x)
    source = "ratio_from_day" if mean_x >= 0.5 else "ratio_from_night"
    return r_night, r_night * day_night_ratio, source


def measure_rad_from_rsl(rsl_paths: list[str], radt_minutes: float, dt_seconds: float,
                         sample_points: list | None, default_r_rad: float,
                         day_night_ratio: float) -> dict:
    """s_normal and the day/night radiation model from rsl probe logs, radiation steps found
    by SCHEDULE (a cost threshold missed severe2023's ~1.05x night steps). Each log's step 1 is
    dropped (setup dominated). Steps without a sim time or without sample_points are left out of
    the day/night fit only. -> {s_normal, r_night, r_day, source, mean_sunlit_fraction,
    n_steps_pooled, n_radiation_steps}."""
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

def predict_radiation_extra(steps: int, radt: float, dt_seconds: float, s_normal: float,
                            r_night: float | None, r_day: float | None,
                            run_start_dt: datetime.datetime | None, sample_points: list | None,
                            default_r_rad: float) -> tuple[int, float, float | None, dict | None]:
    """Price the predicted run's schedule radiation steps: r(t_k) = r_night + (r_day - r_night)
    * sunlit(t_k), extra = sum (r - 1) * s_normal; uniform (r_night + r_day)/2 (or default_r_rad)
    without run_start_dt or sample_points. -> (rad_steps, rad_extra_s, run_mean_sunlit_fraction,
    sampling_advice {night/day sim time and sunlit fraction} or None)."""
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


def domain_work_shares(nl: dict, max_dom: int) -> list[tuple[float, int]]:
    """(share of one d01 step's work, substeps per d01 step) per domain, work = cells x levels x
    substeps as in nested_equivalent_cells."""
    cum, rows = 1, []
    for d in range(max_dom):
        if d > 0:
            cum *= parent_time_step_ratio(nl, d, default=1)
        rows.append((cells_levels(*get_domain_geom(nl, d)) * cum, cum))
    total = sum(w for w, _ in rows) or 1.0
    return [(w / total, n) for w, n in rows]


def predict_radiation_extra_nested(nl: dict, max_dom: int, steps: int, dt_seconds: float,
                                   s_normal: float, r_night: float | None, r_day: float | None,
                                   run_start_dt, sample_points, default_r_rad: float):
    """predict_radiation_extra per domain, for an r that is one domain's radiation-step / normal-
    step cost: each domain's radiation calls (its own radt and substep dt) are priced at its share
    of the d01 step's work. Pricing every d01 radiation step at the whole nested step instead put
    Sundowner (3 domains, radt 1 min < dt 75 s) at 1h45m for a 36:43 run. One domain: unchanged.
    -> (d01 radiation steps, extra s, d01 run mean sunlit fraction, d01 sampling advice)."""
    out = None
    extra = 0.0
    for d, (share, sub) in enumerate(domain_work_shares(nl, max_dom)):
        radt_d = g_dom(nl, "physics", "radt", d, default=None)
        n, x, frac, advice = predict_radiation_extra(
            steps * sub, radt_d, dt_seconds / sub, s_normal * share / sub, r_night, r_day,
            run_start_dt, sample_points, default_r_rad)
        extra += x
        if d == 0:
            out = (n, frac, advice)
    return out[0], extra, out[1], out[2]


def scheme_default_r_rad(nl: dict, cal_rad: dict) -> tuple[float, float, str] | None:
    """Calibrated (r_night, r_day, source) for d01's ra_lw/ra_sw pair from
    calibration.radiation.r_rad_scheme_defaults, else None (the RRTMG-measured defaults apply)."""
    pair = "%s/%s" % (g_dom(nl, "physics", "ra_lw_physics", 0, default="?"),
                      g_dom(nl, "physics", "ra_sw_physics", 0, default="?"))
    row = (cal_rad.get("r_rad_scheme_defaults") or {}).get(pair)
    if not row:
        return None
    return float(row["night"]), float(row["day"]), "default_scheme_%s" % pair.replace("/", "_")


def predict_compute(nl: dict, max_dom: int, steps: int, run_min: float, ranks: int, omp: int,
                    probe_result: dict | None, calibration: dict, rsl_paths: list[str] | None = None,
                    dt_seconds: float | None = None, sample_points: list | None = None,
                    run_start_dt: datetime.datetime | None = None,
                    geometry_source: str = "none") -> dict:
    radt = g_dom(nl, "physics", "radt", 0, default=None)
    default_r_rad = calibration.get("radiation", {}).get(
        "r_rad_default", wtune.machine.measured("r_rad_default"))
    day_night_ratio = calibration.get("radiation", {}).get(
        "r_rad_day_night_ratio", wtune.machine.measured("r_rad_day_night_ratio"))

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
    if r_rad_source == "default_unmeasured" and scheme_default_r_rad(nl, cal_rad):
        r_night, r_day, r_rad_source = scheme_default_r_rad(nl, cal_rad)
    elif r_rad_source == "default_unmeasured" and cal_rad.get("r_rad_default_night") and cal_rad.get("r_rad_default_day"):
        r_night, r_day = float(cal_rad["r_rad_default_night"]), float(cal_rad["r_rad_default_day"])
        r_rad_source = "default_calibrated"

    if probe_result is not None:
        s_normal = probe_result["s_normal"]
        source = probe_result["source"]
        if source == "measured":
            noise_pct = (probe_result.get("spread_pct")
                         if probe_result.get("spread_pct") is not None
                         else wtune.machine.measured("probe_noise_measured_pct"))
        else:
            noise_pct = wtune.machine.measured("probe_noise_extrapolated_pct")
        detail = dict(probe_result)
    elif rad_meas is not None and rad_meas["s_normal"] is not None:
        s_normal = rad_meas["s_normal"]
        source = "probe_rsl"
        noise_pct = wtune.machine.measured("probe_noise_rsl_pct")
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
        noise_pct = wtune.machine.measured("probe_noise_fallback_pct")
        detail = {"source": source, "nested_equivalent_cells": cl_eq, "cores": cores,
                  "fallback_s_per_cell_step_ns": ns_per_cell_core}

    if r_rad_source.startswith("default") and max_dom > 1 and dt_seconds:
        # a default r is one domain's ratio; a d01-rsl-measured one already holds the nests
        rad_steps, rad_extra_s, run_mean_sunlit_fraction, sampling_advice = \
            predict_radiation_extra_nested(nl, max_dom, steps, dt_seconds, s_normal, r_night,
                                           r_day, run_start_dt, sample_points, default_r_rad)
    else:
        rad_steps, rad_extra_s, run_mean_sunlit_fraction, sampling_advice = predict_radiation_extra(
            steps, radt, dt_seconds, s_normal, r_night, r_day, run_start_dt, sample_points,
            default_r_rad)

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


def segment_compute(compute: dict, run_hours: float, dt: float) -> list[dict] | None:
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
    """Deep copy of `calibration` with only rate_table[io_form][io_target] replaced, for pricing
    the same writes at the measured low/high NFS rate (--io-range); other streams keep theirs."""
    patched = copy.deepcopy(calibration)
    table = patched.setdefault("derived", {}).setdefault("io_form_rate_table", {})
    entry = table.setdefault(str(int(io_form)), {})
    target_entry = entry.setdefault(io_target, {})
    target_entry["rate_bytes_per_s"] = rate_bytes_per_s
    return patched


# --------------------------------------------------------------------------
# I/O model
# --------------------------------------------------------------------------

def predict_io(nl: dict, max_dom: int, run_min: float, calibration: dict,
               io_form_override: int | None, io_target: str, case_dir: str | None = None,
               iofields_dir: str | None = None) -> dict:
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

            # Sizing: the iofields file's '+' fields for this stream; 'size unknown' when the
            # referenced file cannot be read/matched; a full frame without iofields_filename.
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
            # restart s = measured ratio x history-frame bytes / history rate, folded into
            # "bytes" so the stream uses the same bytes/rate formula
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

def _path_bytes(p: Path) -> int:
    """Bytes of a file, or of everything under an ADIOS2 `.bp` directory."""
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def predict_init(case_dir: str | None, nl: dict, calibration: dict, ranks: int | None = None,
                 nodes: int | None = None) -> dict:
    read_rate = calibration.get("derived", {}).get("read_rate_bytes_per_s")
    if not read_rate:
        raise PredictError("calibration.derived.read_rate_bytes_per_s missing")

    case_path = Path(case_dir) if case_dir else None
    real_files = []
    if case_path and case_path.is_dir():
        # a dangling symlink (a moved or cleaned .bp) is not a file we can size: skip it
        real_files = [p for p in sorted(case_path.glob("wrfinput_d*")) + sorted(case_path.glob("wrfbdy_d*"))
                      if p.exists()]

    if real_files:
        total_bytes = sum(_path_bytes(p) for p in real_files)
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
    # the cap grows with node count (2026-09-24, sc26p2 4 nodes/192 ranks: wall - sum(main)
    # ~32 s vs the 1-node 19.4 s cap, the out-of-sample miss); per-node slope comes from
    # wtune/cli.py calibrate; absent -> 0, i.e. the old single-node cap
    per_node = calibration.get("startup_s_per_node")
    if startup_calibrated and nodes and per_node and per_node > 0:
        startup_term += per_node * max(0, nodes - 1)
    # rank-dependent startup (2026-09-24): wall - sum(d01 'Timing for main') is 1.9-2.8 s at 1
    # rank, 3.5 s at 6, 6.5 s at 12 (idealized truths), ~17-22 s at 72 (the startup_s fit), so
    # min(startup_s, base + per_rank * ranks); startup_s stays the cap for large runs
    per_rank = calibration.get("startup_s_per_rank")
    # a negative slope would make min() pick an unbounded negative startup (a contaminated
    # calibration produced total_s = -376.6 s for conus12); only a non-negative fit is used
    if startup_calibrated and ranks and per_rank is not None and per_rank >= 0:
        startup_term = min(startup_term, calibration.get("startup_s_base", 0.0) + per_rank * ranks)
    return {"bytes": total_bytes, "source": source, "read_s": read_s,
            "fixed_startup_s": fixed_startup_s,
            "startup_s": startup_s, "startup_s_source": startup_s_source,
            "startup_term_s": startup_term,
            "startup_calibrated": startup_calibrated,
            "total_s": read_s + fixed_startup_s + startup_term}


# --------------------------------------------------------------------------
# Parameter table / launch line
# --------------------------------------------------------------------------

def build_param_table(config: dict, nodes: int, io_form: int, io_target: str,
                      cores_per_node: int | None = None) -> dict:
    # ranks/omp/nx/ny are load-bearing INSIDE the launch line itself (-np R
    # ... PE=OMP); a half-"UNKNOWN" mpirun command is worse than no command,
    # so these are REQUIRED -- refuse loudly rather than guess or print junk.
    missing = [k for k in ("ranks", "omp", "nx", "ny") if config.get(k) is None]
    if missing:
        raise PredictError(
            "build_param_table: config is missing required field(s) %s (got %r); "
            "the launch line cannot be built without them" % (missing, config))
    ranks, omp, nx, ny = config["ranks"], config["omp"], config["nx"], config["ny"]

    # An absent numtiles/dist/io_form is UNKNOWN, never a plausible default (a defaulted
    # "cyclic" printed a launch line for a different run, scc blind test 2026-09-23).
    dist = config.get("dist")
    numtiles = config.get("numtiles")
    io_form_display = io_form if io_form is not None else "UNKNOWN"

    tpn = ranks // nodes if nodes else None
    warnings = []
    if nodes and ranks % nodes != 0:
        warnings.append("ranks (%d) not evenly divisible by nodes (%d); TPN rounded down to %d" % (
            ranks, nodes, tpn))

    if tpn and cores_per_node and tpn * omp > cores_per_node:     # only when the machine is known
        warnings.append("%d ranks/node x omp %d = %d cores exceeds %d cores/node"
                        % (tpn, omp, tpn * omp, cores_per_node))

    # The launch line this machine's site file builds (campaign/sites/<site>.sh site_launch).
    # dist=UNKNOWN is shown as block, which may differ from what the run uses, hence the note.
    launch = (wtune.machine.site_launch_line(ranks, tpn or 0, omp, dist or "block", nodes or 1)
              or "site_launch %d %d %d %s (campaign/sites/%s.sh)"
              % (ranks, tpn or 0, omp, dist or "block", wtune.machine.site()))
    if dist is None:
        warnings.append("dist not provided; launch line shown for block (a cyclic layout "
                        "launches differently -- campaign/sites/%s.sh site_launch)"
                        % wtune.machine.site())

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

def resolve_config(args: argparse.Namespace) -> dict:
    if args.config:
        try:
            raw = json.loads(args.config)
        except json.JSONDecodeError:
            raw = json.loads(Path(args.config).read_text())
        omp = int(raw.get("omp", 1))
        nx = int(raw["nx"] if "nx" in raw else raw["nproc_x"])
        ny = int(raw["ny"] if "ny" in raw else raw["nproc_y"])
        ranks = int(raw.get("ranks") or nx * ny)
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


def build_prediction(namelist_path: str, case_dir: str | None, nodes: int, config: dict,
                     ledger_paths: list[str], calibration_path: str | Path,
                     io_form_override: int | None, io_target: str, case_name: str | None = None,
                     iofields_dir: str | None = None, rsl_paths: list[str] | None = None,
                     include_io: bool = False) -> dict:
    nl = parse_namelist(namelist_path)
    max_dom = get_max_dom(nl)
    steps, run_s, dt, steps_note = get_d01_steps(nl)
    run_min = run_s / 60.0
    run_hours = run_s / 3600.0

    calibration = load_calibration(calibration_path)

    ledger_rows = load_ledger_rows(ledger_paths)
    # case_name: the ledger case_base when it differs from the directory name
    # (doi4-lafe2017 vs lafe); else the probe match fails and no_probe_fallback is used.
    name = case_name or (Path(case_dir).name if case_dir else "unknown")
    probe_result = resolve_probe(ledger_rows, name, config)

    # Domain geometry and start time place the predicted radiation steps; without either
    # predict_compute uses the uniform level.
    sample_points, geometry_source = read_wrfinput_sample_points(case_dir)
    run_start_dt = get_time_control_datetime(nl, "start")

    compute = predict_compute(nl, max_dom, steps, run_min, config["ranks"], config["omp"],
                              probe_result, calibration, rsl_paths=rsl_paths, dt_seconds=dt,
                              sample_points=sample_points, run_start_dt=run_start_dt,
                              geometry_source=geometry_source)
    # WRF extensions (chem/hydro/moving nest/WRFDA): explicit cost factors with measured/placeholder
    # status and widened uncertainty (wtune/extensions.py); no-op for plain ARW.
    import wtune.extensions
    compute = wtune.extensions.apply_cost_factors(compute, nl, dt, run_dir=case_dir,
                                                  calibration=calibration)
    segments = segment_compute(compute, run_hours, dt)

    io_target = io_target or "nfs"
    io_form_history_effective = io_form_override if io_form_override is not None else \
        g(nl, "time_control", "io_form_history", 2)
    io = predict_io(nl, max_dom, run_min, calibration, io_form_override, io_target,
                    case_dir=case_dir, iofields_dir=iofields_dir)

    init = predict_init(case_dir, nl, calibration, ranks=config.get("ranks") if config else None,
                        nodes=nodes)

    drift_pct = calibration.get("drift_band_pct", 5.0)
    # History I/O is estimated and reported but, by default, NOT added to total_s or the range
    # (user decision 2026-09-24): the write rate was off in both directions on the SC26 atlas
    # (marshall -42%, maria +51%, isc21 +156%). include_io=True restores the old sum.
    io_w = 1.0 if include_io else 0.0
    total_s = compute["compute_total_s"] + io_w * io["total_s"] + init["total_s"]

    # With a calibrated I/O rate bracket: I/O at fast..slow rate, drift on compute only.
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
        ext_unc = compute.get("extension_uncertainty_pct") or 0.0
        if ext_unc:  # widen the compute band by the extension factors' uncertainty (quadrature)
            compute_drift_abs_s = math.hypot(compute_drift_abs_s,
                                             compute["compute_total_s"] * ext_unc / 100.0)
        drift_abs_s = compute_drift_abs_s
        # the s/step source noise (+-50% no-probe fallback, +-15% extrapolated, ...) belongs in
        # the compute band too; noise_abs_s already carries ext_unc, so combine it with the
        # bare drift (quadrature), as the no-bracket branch below does.
        compute_drift_abs_s = math.hypot(compute["compute_total_s"] * (drift_pct / 100.0),
                                         compute["noise_abs_s"])
        low = (max(0.0, compute["compute_total_s"] - compute_drift_abs_s)
               + io_w * io_range["fast_total_s"] + init["total_s"])
        high = (compute["compute_total_s"] + compute_drift_abs_s
                + io_w * io_range["slow_total_s"] + init["total_s"])
        range_source = (("I/O: measured %.1f-%.1f MB/s write-rate bracket; "
                         % (io_low_mbps, io_high_mbps) if io_w else
                         "I/O excluded (--include-io adds it); ")
                        + "compute: +-%.0f%% drift band (NOT VALIDATED) with +-%.0f%% s/step "
                          "source noise" % (drift_pct, compute["noise_pct"]))
        if ext_unc:
            range_source += "; widened by WRF-extension factor uncertainty +-%.0f%%" % ext_unc
    else:
        drift_abs_s = total_s * (drift_pct / 100.0)
        range_abs_s = math.sqrt(compute["noise_abs_s"] ** 2 + (io_w * io["noise_abs_s"]) ** 2
                                + drift_abs_s ** 2)
        low, high = max(0.0, total_s - range_abs_s), total_s + range_abs_s
        range_source = None

    bounds = adaptive_step_bounds(nl, run_s)
    if bounds:
        lo_n, hi_n, lo_dt, hi_dt = bounds
        rest = total_s - compute["compute_total_s"]       # init (+ I/O): not per step
        low = min(low, max(0.0, low - rest) * lo_n / steps + rest)
        high = max(high, (high - rest) * hi_n / steps + rest)
        steps_note = "; ".join(x for x in (steps_note, (
            "use_adaptive_time_step: WRF picks d01's dt between about %g and %g s, so about "
            "%d-%d steps; the total counts %d steps of time_step, the range covers both"
            % (lo_dt, hi_dt, lo_n, hi_n, steps))) if x)
        range_source = "; ".join(x for x in (range_source, "widened to the adaptive-dt "
                                             "step bounds") if x)

    params = build_param_table(config, nodes, io_form_history_effective, io_target,
                               cores_per_node(calibration))

    return {
        "case_name": name, "namelist": str(namelist_path),
        "max_dom": max_dom, "steps": steps, "run_s": run_s, "run_min": run_min,
        "run_hours": run_hours, "dt_s": dt, "steps_note": steps_note,
        "compute": compute, "segments": segments, "io": io, "io_range": io_range, "init": init,
        "io_included": include_io,
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


def render_text(pred: dict) -> str:
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
    import wtune.extensions
    out.extend(wtune.extensions.render_factor_lines(c))
    if pred["segments"]:
        out.append("  segmented (>%.0fh simulated, %.0fh chunks, NOT VALIDATED drift):"
                    % (LONG_RUN_HOURS, SEGMENT_HOURS))
        for seg in pred["segments"]:
            out.append("    %-12s steps=%-6d %s" % (seg["sim_time"], seg["steps"], _fmt_hms(seg["compute_s"])))

    io = pred["io"]
    out.append("I/O: total %s%s" % (_fmt_hms(io["total_s"]),
                                    "" if pred.get("io_included", True)
                                    else "  (estimate only, NOT in TOTAL; --include-io adds it)"))
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


def to_json(pred: dict) -> str:
    return json.dumps(pred, indent=2, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_subparser(sub: Any) -> argparse.ArgumentParser:
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
    p.add_argument("--calibration", default=calibration_path(),
                   help="calibration JSON (default: the machine's, built-in "
                        "machine/calibration/scc.json)")
    p.add_argument("--io-form", type=int, default=None, choices=[2, 11, 14],
                   help="override io_form_history (default: read from namelist)")
    p.add_argument("--io-target", choices=["nfs", "local"], default="nfs",
                   help="write target (default nfs; local is UNMEASURED)")
    p.add_argument("--include-io", action="store_true",
                   help="add the history I/O estimate to TOTAL and its range (off by default: "
                        "the write rate is not predictable enough, see build_prediction)")
    p.add_argument("--iofields-dir", default=None,
                   help="directory holding the real iofields_filename field-list file(s) "
                        "(default: case_dir); used to size field-subset aux streams exactly")
    p.add_argument("--json", default=None, help="also write the full prediction as JSON to this path")
    return p


def predict_cmd(args: argparse.Namespace) -> int:
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
            rsl_paths=args.probe_rsl, include_io=args.include_io)
    except PredictError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    print(render_text(pred))
    if args.json:
        Path(args.json).write_text(to_json(pred) + "\n")
        print("\nwritten: %s" % args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wtune predict")
    sub = parser.add_subparsers(dest="command")
    add_subparser(sub)
    args = parser.parse_args(argv)
    if args.command == "predict":
        return predict_cmd(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
