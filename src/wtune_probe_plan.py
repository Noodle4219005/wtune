#!/usr/bin/env python3
"""wtune_probe_plan.py -- multi-point probe planning + whole-run combine (2026-09-26 redesign).

A fixed 20-step t0 probe missed every radiation step and the spin-up cost ramp (conus25: -9.6 %).
plan_points() places probe points on the run's own timeline (coldstart_sample's window policy
and window sizing); combine() turns measured (time, s_normal, n_rad, r) points into a whole-run
compute prediction with coldstart_sample's own arithmetic. Point sources, in probe.sources
order: trajectory (the anchor's restart chain), restart (an existing wrfrst), rebuild (alias
coldstart: WPS/real.exe at that time from real forcing, spun up; a fictional date gets an
analog). An unservable point is dropped and reported. Pure: no job submission, no network."""
from __future__ import annotations

import dataclasses
import datetime
import math
import os
import re
import statistics
from typing import Any, Callable

import coldstart_sample as coldstart   # tools/coldstart_sample.py -- READ-ONLY dependency: this
                                        # module calls its functions, never edits them.
import wtune_predict as predict        # wtune_predict.py -- same read-only relationship.


class ProbePlanError(Exception):
    """A config-shaped error: an unparsable/out-of-range explicit probe.points entry, or any
    other input plan_points()/combine() cannot make sense of. Distinct from a per-point SOURCE
    drop (which is not an error -- see plan_points' `notes` return value)."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ProbePoint:
    at: datetime.datetime
    reason: str
    source: str            # "trajectory" | "restart" | "rebuild", or None if unresolved
    steps: int
    # requested_at: the targeted time P; `at` is what is measured (M = T + spinup for rebuild,
    # else equal)
    requested_at: "datetime.datetime | None" = None
    # io_probe: this point's rebuild segment also times one history frame (+ one restart when the
    # case writes restarts) at its end -- plan_rebuild_segments -> coldstart build_plan(io_probe=).
    io_probe: bool = False


@dataclasses.dataclass
class PointMeasurement:
    """One MEASURED probe point, as fed into combine()."""
    at: datetime.datetime
    reason: str
    source: str
    s_normal: float                 # median non-radiation-step wall time at this point
    n_rad: int = 0                  # count of radiation steps observed at this point
    r: "float | None" = None        # mean measured (radiation step wall time / s_normal) here,
                                     # or None if n_rad == 0


# ---------------------------------------------------------------------------
# step-count sizing for probe.steps='auto' and the per-point window: >= 1 radiation step
# ---------------------------------------------------------------------------
def steps_for_stepra(stepra: int) -> int:
    """max(30, stepra + 10) -- a probe window this long always contains >= 1 radiation step
    (WRF calls radiation at itimestep 1 and every stepra-th step after), with 10 steps of slack
    so the window isn't cut exactly at the boundary."""
    return max(30, int(stepra) + 10)


def resolve_ranking_steps(radt_minutes: float, dt_seconds: float) -> int:
    """steps_for_stepra(), computing stepra from radt/dt itself (wtune_predict.
    steps_per_radiation_call) -- the convenience form callers use when they don't already have
    stepra (e.g. wtune_tune.py resolving probe.steps='auto')."""
    stepra = predict.steps_per_radiation_call(radt_minutes, dt_seconds)
    return steps_for_stepra(stepra)


# ---------------------------------------------------------------------------
# source names / aliases
# ---------------------------------------------------------------------------
PROBE_SOURCES = ("trajectory", "restart", "rebuild")
_SOURCE_ALIASES = {"coldstart": "rebuild"}   # pre-2026-09-26 name, still accepted


def canonical_source(name: str) -> str:
    """Normalizes a source name -- 'coldstart' (legacy) -> 'rebuild'; anything else passes
    through unchanged (validation of whether it's actually a KNOWN name is the caller's job --
    see wtune_config.py, which validates against PROBE_SOURCES | _SOURCE_ALIASES)."""
    return _SOURCE_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# explicit point parsing ("+90m" / "+2h" / "70%" / "YYYY-MM-DD_HH:MM:SS", optionally wrapped in
# {"at": ..., "reason": ..., "source": ...})
# ---------------------------------------------------------------------------
_OFFSET_RE = re.compile(r"^\+(\d+(?:\.\d+)?)(m|h)$")
_PCT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
_ISO_FMT = "%Y-%m-%d_%H:%M:%S"


def parse_explicit_point(spec: Any, run_start: datetime.datetime, run_end: datetime.datetime
                         ) -> tuple[datetime.datetime, str, str | None]:
    """spec: a string ("+90m"/"+2h"/"70%"/"YYYY-MM-DD_HH:MM:SS") or a mapping {"at": <string>,
    "reason": <str, optional>, "source": <str, optional>}. Returns (datetime, reason_or_None,
    source_or_None). Raises ProbePlanError on a spec this function cannot parse -- the caller
    (plan_points) adds the point's index to the message."""
    reason = None
    source = None
    if isinstance(spec, dict):
        reason = spec.get("reason")
        source = spec.get("source")
        if source is not None:
            source = canonical_source(source)
        spec = spec.get("at")
        if spec is None:
            raise ProbePlanError("mapping form requires 'at', got no 'at' key")
    if not isinstance(spec, str):
        raise ProbePlanError("%r is not a string or {at, reason, source} mapping" % (spec,))

    m = _OFFSET_RE.match(spec)
    if m:
        value, unit = m.groups()
        seconds = float(value) * (60.0 if unit == "m" else 3600.0)
        return run_start + datetime.timedelta(seconds=seconds), reason, source

    m = _PCT_RE.match(spec)
    if m:
        frac = float(m.group(1)) / 100.0
        total_s = (run_end - run_start).total_seconds()
        return run_start + datetime.timedelta(seconds=frac * total_s), reason, source

    if _ISO_RE.match(spec):
        return datetime.datetime.strptime(spec, _ISO_FMT), reason, source

    raise ProbePlanError(
        "%r is not a recognized probe point form ('+<N>m', '+<N>h', '<N>%%', or "
        "'YYYY-MM-DD_HH:MM:SS')" % (spec,))


def _merge_close_points(points, min_gap_s):
    """points: sorted [(dt, reason, source), ...]. Consecutive points closer than min_gap_s to
    the last KEPT point are merged into it (its reason gets the merged-in one appended with
    '+', its source is kept as-is -- an explicit source on the dropped duplicate is not silently
    discarded: it is folded into the reason string so it stays visible)."""
    if not points:
        return []
    out = [list(points[0])]
    for at, reason, source in points[1:]:
        prev_at, prev_reason, prev_source = out[-1]
        if (at - prev_at).total_seconds() < min_gap_s:
            tag = reason + (("@" + source) if source else "")
            if tag not in prev_reason.split("+"):
                out[-1][1] = prev_reason + "+" + tag
            continue
        out.append([at, reason, source])
    return [tuple(p) for p in out]


# ---------------------------------------------------------------------------
# per-point source resolution
# ---------------------------------------------------------------------------
def _restart_file_glob(at):
    """WRF's own wrfrst_d01_<time> naming (any domain -- d01 is enough to prove the point's
    restart set exists; every domain's own wrfrst_dNN_<time> is written at the same time)."""
    return "wrfrst_d01_%04d-%02d-%02d_%02d:%02d:%02d" % (
        at.year, at.month, at.day, at.hour, at.minute, at.second)


def resolve_rebuild_point(p_at: datetime.datetime, run_start: datetime.datetime,
                          run_end: datetime.datetime, spinup_s: float,
                          rebuild_measure_window_s: float, forcing_interval_h: float
                          ) -> tuple[datetime.datetime | None, datetime.datetime]:
    """(M, T) for a rebuild point targeting P=p_at: T = the latest forcing-grid time (epoch
    multiples of forcing_interval_h, floored) with T + spinup_s <= P, M = T + spinup_s -- never P
    snapped to the grid. rebuild_measure_window_s must be the window the rebuild job really
    measures (compute_window_seconds), or build_plan rejects the point later. (None, T) when M's
    window overruns run_end. Never raises."""
    interval_s = forcing_interval_h * 3600.0
    limit = p_at - datetime.timedelta(seconds=spinup_s)
    epoch = datetime.datetime(1970, 1, 1)
    limit_s = (limit - epoch).total_seconds()
    t_s = math.floor(limit_s / interval_s) * interval_s
    T = epoch + datetime.timedelta(seconds=t_s)
    M = T + datetime.timedelta(seconds=spinup_s)
    if M + datetime.timedelta(seconds=rebuild_measure_window_s) > run_end:
        return None, T
    return M, T


def resolve_point_source(at: datetime.datetime, run_start: datetime.datetime,
                         run_end: datetime.datetime, sources_pref: list[str], spinup_s: float = 0.0,
                         measure_window_s: float = 0.0,
                         rebuild_measure_window_s: float | None = None, case_dir: str | None = None,
                         trajectory_horizon_s: float | None = None,
                         is_real_calendar: bool | None = None,
                         forcing_interval_h: float | None = None) -> tuple:
    """First source in sources_pref that can serve a point at `at` -> (source, effective_at,
    requested_at); (None, None, at) when none can. trajectory: within trajectory_horizon_s of
    run_start; restart: an existing case_dir/wrfrst_d01_<at>; rebuild: a real calendar date, known
    forcing_interval_h and a usable M from resolve_rebuild_point (effective_at = M)."""
    real = is_real_calendar if is_real_calendar is not None else (at.year >= 1900)
    rebuild_window = rebuild_measure_window_s if rebuild_measure_window_s is not None \
        else measure_window_s
    for src in sources_pref:
        src = canonical_source(src)
        if src == "trajectory":
            if trajectory_horizon_s is not None and (at - run_start).total_seconds() <= trajectory_horizon_s:
                return src, at, at
        elif src == "restart":
            if case_dir and os.path.exists(os.path.join(case_dir, _restart_file_glob(at))):
                return src, at, at
        elif src == "rebuild":
            if real and forcing_interval_h is not None:
                M, _T = resolve_rebuild_point(at, run_start, run_end, spinup_s, rebuild_window,
                                              forcing_interval_h)
                if M is not None:
                    return src, M, at
    return None, None, at


def resolve_rebuild_target_date(case_dir: str, at: datetime.datetime, namelist: str | None = None,
                                forcing_probe_fn: Callable | None = None,
                                analog_fn: Callable | None = None, grib_dir: str | None = None
                                ) -> tuple[datetime.datetime, bool, str | None]:
    """(effective_date, substituted, note): when no forcing source covers `at` (a fictional date),
    an analog real date from case_rebuild.cmd_analog (or analog_fn) with substituted=True and a note
    the report must show. Analog failures are caught and noted, never raised."""
    try:
        chain = coldstart.resolve_forcing_chain_for(at, grib_dir=grib_dir)
    except Exception as e:
        chain = []
        chain_note = "forcing-chain lookup failed (%s: %s)" % (type(e).__name__, e)
    else:
        chain_note = None
    if chain:
        return at, False, None

    if analog_fn is None:
        try:
            import case_rebuild  # tools/case_rebuild.py -- lazy import: heavy, network-using
            analog_fn = case_rebuild.cmd_analog
        except ImportError as e:
            return at, True, ("rebuild target %s has no forcing coverage (%s) and "
                               "case_rebuild.py is not importable (%s) -- no analog available; "
                               "this point cannot be honestly rebuilt" %
                               (at.strftime(_ISO_FMT), chain_note or "no source covers it", e))
    try:
        analog_dir = os.path.join(case_dir, ".wtune_probe_plan_analog") if case_dir else None
        result = analog_fn(case_dir, analog_dir, namelist=namelist) if case_dir else None
        chosen = (result or {}).get("chosen") if isinstance(result, dict) else None
        if not chosen or not chosen.get("date"):
            raise ValueError("cmd_analog returned no usable 'chosen.date'")
        analog_dt = datetime.datetime.strptime(chosen["date"], _ISO_FMT)
    except Exception as e:
        return at, True, ("rebuild target %s has no forcing coverage (%s) and the analog lookup "
                           "failed (%s: %s) -- no analog available; this point cannot be "
                           "honestly rebuilt" % (at.strftime(_ISO_FMT), chain_note or
                                                  "no source covers it", type(e).__name__, e))
    return analog_dt, True, ("rebuild target %s has no real forcing coverage -- substituted "
                              "analog date %s (case_rebuild.cmd_analog); the candidate is "
                              "measured against the ANALOG date's weather, not %s's"
                              % (at.strftime(_ISO_FMT), analog_dt.strftime(_ISO_FMT),
                                 at.strftime(_ISO_FMT)))


# ---------------------------------------------------------------------------
# plan_points
# ---------------------------------------------------------------------------
def _merge_resolved_points(points, min_gap_s):
    """Merge sorted [effective_at, reason, source, requested_at, priority_idx] lists in place:
    a point within min_gap_s of the last kept one folds into it (reason tagged with its source and
    requested time); priority_idx becomes the min of the two for budget trimming."""
    if not points:
        return []
    out = [list(points[0])]
    for at, reason, source, req_at, idx in points[1:]:
        prev = out[-1]
        if (at - prev[0]).total_seconds() < min_gap_s:
            tag = reason + "@" + source
            if req_at != at:
                tag += "(P=%s)" % req_at.isoformat()
            if tag not in prev[1].split("+"):
                prev[1] = prev[1] + "+" + tag
            prev[4] = min(prev[4], idx)
            continue
        out.append([at, reason, source, req_at, idx])
    return out


def plan_points(case_nl: dict | None, run_start: datetime.datetime, run_end: datetime.datetime,
                dt: float, stepra: int, spinup_s: float, cfg: dict, sample_points: list | None = None,
                precip_lookup: dict | None = None, s_est: float | None = None,
                predicted_run_s: float | None = None, case_dir: str | None = None,
                forcing_interval_h: float | None = None, grib_dir: str | None = None,
                points_by: str | None = None, indicators: Any = None
                ) -> tuple[list[ProbePoint], list[str]]:
    """Place probe points on [run_start, run_end] -> ([ProbePoint], notes). Notes cover short-run
    fallback, budget capping and dropped points; ProbePlanError only for a malformed explicit
    probe.points entry. Order: resolve the rebuild cadence, generate candidates (select_windows or
    explicit), resolve each source (rebuild -> M), merge collapsed points, trim to budget_pct by
    selection priority. points_by "cost-indicators" (decision 0060) places auto points by the
    caller-built cost series; unusable indicators fall back to time placement with a note."""
    notes = []
    steps_cfg = cfg.get("steps", "auto")
    steps = steps_for_stepra(stepra) if steps_cfg in (None, "auto") else int(steps_cfg)
    measure_window_s = steps * dt
    # the window a rebuild job really measures (compute_window_seconds), not steps_for_stepra
    rebuild_measure_window_s = max(30, 2 * stepra) * dt
    duration_s = (run_end - run_start).total_seconds()
    sources_pref = [canonical_source(s) for s in (cfg.get("sources") or list(PROBE_SOURCES))]
    max_points = cfg.get("max_points") or 5
    budget_pct = cfg.get("budget_pct")

    if forcing_interval_h is None and "rebuild" in sources_pref and case_dir:
        try:
            chain = coldstart.resolve_forcing_chain_for(run_start, grib_dir=grib_dir)
        except Exception as e:
            chain = []
            notes.append("probe: rebuild source requested but forcing-chain lookup failed for "
                         "%s (%s: %s)" % (run_start.isoformat(), type(e).__name__, e))
        else:
            if chain:
                forcing_interval_h = chain[0].forcing_interval_h
            else:
                notes.append("probe: rebuild source requested but no forcing source covers %s"
                             % run_start.isoformat())

    def _resolve(at, reason, forced_source):
        pref = [forced_source] if forced_source else sources_pref
        src, eff_at, req_at = resolve_point_source(
            at, run_start, run_end, pref, spinup_s=spinup_s, measure_window_s=measure_window_s,
            rebuild_measure_window_s=rebuild_measure_window_s, case_dir=case_dir,
            trajectory_horizon_s=trajectory_horizon_s, forcing_interval_h=forcing_interval_h)
        return src, eff_at, req_at, pref

    def _fallback_point(reason):
        src, eff_at, req_at, _pref = _resolve(run_start, reason, None)
        return [ProbePoint(at=eff_at or run_start, reason=reason, source=src, steps=steps,
                           requested_at=req_at)]

    trajectory_horizon_s = spinup_s + max_points * measure_window_s

    if duration_s < spinup_s + measure_window_s:
        notes.append(
            "probe: run too short for spin-up+measure window (%.0fs available < %.0fs "
            "needed: spinup=%.0fs + measure_window=%.0fs) -- falling back to a single t0 point"
            % (duration_s, spinup_s + measure_window_s, spinup_s, measure_window_s))
        return _fallback_point("t0_fallback"), notes

    explicit = cfg.get("points")
    if explicit and explicit != "auto":
        parsed = []
        for i, spec in enumerate(explicit):
            try:
                at, reason, source = parse_explicit_point(spec, run_start, run_end)
            except ProbePlanError as e:
                raise ProbePlanError("probe.points[%d]: %s" % (i, e))
            reason = reason or "explicit"
            if at < run_start or at + datetime.timedelta(seconds=measure_window_s) > run_end:
                raise ProbePlanError(
                    "probe.points[%d]: %s is outside [run_start, run_end - measure_window] "
                    "(run_start=%s, run_end=%s, measure_window=%.0fs)"
                    % (i, at.isoformat(), run_start.isoformat(), run_end.isoformat(),
                       measure_window_s))
            parsed.append((at, reason, source))
        parsed.sort(key=lambda p: p[0])
        merged = _merge_close_points(parsed, measure_window_s)
        raw_points = merged
    else:
        avail_s = duration_s - spinup_s - measure_window_s
        n_grid = min(200, max(20, int(avail_s / measure_window_s) + 1)) if avail_s > 0 else 1
        interval_s = (avail_s / (n_grid - 1)) if n_grid > 1 else max(measure_window_s, 1.0)
        t0 = run_start + datetime.timedelta(seconds=spinup_s)
        candidates = coldstart.candidate_times(t0, run_end, interval_s, measure_window_s)
        if not candidates:
            notes.append("probe: no auto-placement candidates fit [t0+spinup, run_end - "
                         "measure_window] -- falling back to a single t0 point")
            return _fallback_point("t0_fallback"), notes
        sel_kw = {}
        if (points_by or cfg.get("points_by") or "time") == "cost-indicators":
            if indicators is not None and indicators.usable:
                sel_kw["cost_lookup"] = indicators.cost_lookup(candidates, measure_window_s)
                notes.append("probe: points_by=cost-indicators, c(t) from %s"
                             % (sorted(set(indicators.used.values())),))
            else:
                why = (indicators.why_unusable() if indicators is not None
                       else "no indicators supplied")
                notes.append("probe: WARNING points_by=cost-indicators unavailable (%s) -- using "
                             "the time-based selection" % why)
            if precip_lookup is None and indicators is not None:
                precip_lookup = indicators.precip_lookup(candidates, measure_window_s) or None
        picks = coldstart.select_windows(candidates, 0.0, measure_window_s, max_points,
                                         sample_points=sample_points,
                                         precip_lookup=precip_lookup, **sel_kw)
        raw_points = [(at, reason, None) for at, reason in picks]

    resolved = []
    for idx, (at, reason, source) in enumerate(raw_points):
        eff_source, eff_at, req_at, pref = _resolve(at, reason, source)
        if eff_source is None:
            notes.append("probe: dropped point %s (reason=%s) -- no source in %s could serve it"
                         % (at.isoformat(), reason, pref))
            continue
        resolved.append([eff_at, reason, eff_source, req_at, idx])

    if not resolved:
        notes.append("probe: every placed point was dropped -- falling back to a single t0 point")
        return _fallback_point("t0_fallback"), notes

    resolved.sort(key=lambda p: p[0])
    merged = _merge_resolved_points(resolved, measure_window_s)

    if s_est and predicted_run_s and budget_pct and len(merged) > 1:
        cost_per_point = (spinup_s / dt + steps) * s_est
        budget_s = (budget_pct / 100.0) * predicted_run_s
        n_afford = max(1, int(budget_s / cost_per_point)) if cost_per_point > 0 else len(merged)
        if n_afford < len(merged):
            notes.append("probe: budget_pct=%s capped points to %d (of %d placed)"
                         % (budget_pct, n_afford, len(merged)))
            by_priority = sorted(merged, key=lambda p: p[4])[:n_afford]
            merged = sorted(by_priority, key=lambda p: p[0])

    points = [ProbePoint(at=eff_at, reason=reason, source=src, steps=steps, requested_at=req_at)
              for eff_at, reason, src, req_at, _idx in merged]
    return points, notes


# ---------------------------------------------------------------------------
# ranking-layer namelist text (restart-start probe candidate)
# ---------------------------------------------------------------------------
_HISTORY_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(history_interval\w*)\s*=", re.I | re.M)


def build_restart_start_namelist_text(namelist_text: str, restart_time: datetime.datetime,
                                      steps: int, dt_seconds: float, nproc_x: int | None = None,
                                      nproc_y: int | None = None, numtiles: int | None = None) -> str:
    """A ranking-probe namelist started from the restart at restart_time: every domain's start_* =
    restart_time, run_seconds = steps*dt, restart=.true., restart_interval and history_interval*
    (if present) forced huge. Mirror of coldstart_sample.build_wrf_window_namelist_text; the
    campaign's own copy of the rule is configs.edit_namelist(restart_from=...)."""
    text0 = namelist_text
    ndom = int(coldstart.get_first(text0, "max_dom", "1"))
    for k, v in (("year", restart_time.year), ("month", "%02d" % restart_time.month),
                 ("day", "%02d" % restart_time.day), ("hour", "%02d" % restart_time.hour),
                 ("minute", "%02d" % restart_time.minute), ("second", "%02d" % restart_time.second)):
        if coldstart.count_key(text0, "start_" + k) or k in ("minute", "second"):
            text0 = coldstart.set_list(text0, "start_" + k, v, ndom)
    run_seconds = int(round(steps * dt_seconds))
    for k, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0), ("run_seconds", run_seconds)):
        text0 = coldstart.set_list(text0, k, v, 1)
    if coldstart.count_key(text0, "restart"):
        text0 = coldstart.set_list(text0, "restart", ".true.", 1)
    if coldstart.count_key(text0, "restart_interval"):
        text0 = coldstart.set_list(text0, "restart_interval", 999999, 1)
    history_keys = sorted(set(m.group(1).lower() for m in _HISTORY_KEY_RE.finditer(text0)))
    for hk in history_keys:
        text0 = coldstart.set_list(text0, hk, 999999, ndom)
    for key, value in (("nproc_x", nproc_x), ("nproc_y", nproc_y), ("numtiles", numtiles)):
        if value is not None and coldstart.count_key(text0, key):
            text0 = coldstart.set_list(text0, key, value, 1)
    return text0


def build_spinup_writer_namelist_text(namelist_text: str, spinup_minutes: float,
                                      restart_io_form: int | None = None) -> str:
    """The anchor's ONE spin-up run: starts at the case's own t0 (restart untouched -- stays
    whatever the template already has, normally '.false.'), writes exactly one restart at
    t0+spinup_minutes (restart_interval = spinup_minutes; io_form_restart set to
    restart_io_form when given -- netCDF's 2 is decomposition-independent, unlike ADIOS2's 14),
    history off. Run length = spinup_minutes (run_days/hours/seconds=0, run_minutes=
    spinup_minutes)."""
    text0 = namelist_text
    ndom = int(coldstart.get_first(text0, "max_dom", "1"))
    for k, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", int(spinup_minutes)),
                 ("run_seconds", 0)):
        text0 = coldstart.set_list(text0, k, v, 1)
    if coldstart.count_key(text0, "restart_interval"):
        text0 = coldstart.set_list(text0, "restart_interval", int(spinup_minutes), 1)
    if restart_io_form is not None and coldstart.count_key(text0, "io_form_restart"):
        text0 = coldstart.set_list(text0, "io_form_restart", restart_io_form, 1)
    history_keys = sorted(set(m.group(1).lower() for m in _HISTORY_KEY_RE.finditer(text0)))
    for hk in history_keys:
        text0 = coldstart.set_list(text0, hk, 999999, ndom)
    return text0


# ---------------------------------------------------------------------------
# trajectory/restart measurement extraction (steps.txt -> PointMeasurement)
# ---------------------------------------------------------------------------
def first_global_step_from_rsl(rsl_path: str, run_start: datetime.datetime, dt: float) -> int:
    """Global d01 step number of the FIRST 'Timing for main' line of a (restart-started) run:
    the printed time is the END of that step, so g = (time - run_start) / dt. Needed because a
    window that does not start on the radiation cycle (e.g. a restart at 20:02 with radt=10 min)
    has its radiation steps at global g % stepra == 1, not at local i % stepra == 1 (C3 job 1231:
    the 20:02 window's real radiation step was local #33; local #41 read r=0.995)."""
    with open(rsl_path) as f:
        for line in f:
            if "Timing for main" in line:
                t = datetime.datetime.strptime(line.split("time", 1)[1].split()[0],
                                               "%Y-%m-%d_%H:%M:%S")
                return int(round((t - run_start).total_seconds() / dt))
    raise ProbePlanError("first_global_step_from_rsl: no 'Timing for main' line in %s" % rsl_path)


def measurement_from_steps_file(path: str, stepra: int, at: datetime.datetime, reason: str,
                                source: str, first_global_step: int | None = None
                                ) -> PointMeasurement:
    """PointMeasurement from a strict_campaign steps.txt (d01 step seconds, step order). Radiation
    steps are local i with i % stepra == 1 (schedule phase 1, verified on conus25/G0014 truth --
    not wtune_predict's phase 0; the phase detector needs 2*stepra steps); step 1 is dropped;
    medians. first_global_step (first_global_step_from_rsl) switches to global step numbers."""
    with open(path) as f:
        wall = [float(x) for x in f if x.strip()]
    g0 = 1 if first_global_step is None else int(first_global_step)
    normal, rad = [], []
    for i, w in enumerate(wall, start=1):
        if i == 1:
            continue
        (rad if (g0 + i - 1) % stepra == 1 else normal).append(w)
    if not normal:
        raise ProbePlanError("measurement_from_steps_file: %s has no usable normal steps" % path)
    s_normal = statistics.median(normal)
    r = (statistics.median(rad) / s_normal) if rad else None
    return PointMeasurement(at=at, reason=reason, source=source, s_normal=s_normal,
                            n_rad=len(rad), r=r)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------
def _resolve_r_rad_from_points(pooled_radiation_samples, calibration):
    """(r_night, r_day, source): >= 2 radiation samples -> wtune_predict.fit_day_night_r_rad;
    one -> that ratio flat (reproduces the constant-ratio model); none -> the calibration
    default."""
    calibration = calibration or {}
    cal_rad = calibration.get("radiation", {})
    default_r_rad = cal_rad.get("r_rad_default", 5.0)
    if len(pooled_radiation_samples) >= 2:
        day_night_ratio = cal_rad.get("r_rad_day_night_ratio", predict.DEFAULT_R_RAD_DAY_NIGHT_RATIO)
        r_night, r_day, source = predict.fit_day_night_r_rad(pooled_radiation_samples, day_night_ratio)
        if r_night is not None:
            return r_night, r_day, source
    if pooled_radiation_samples:
        r = pooled_radiation_samples[0][1] if len(pooled_radiation_samples) == 1 else \
            statistics.mean(r for _, r in pooled_radiation_samples)
        source = "single_point_flat" if len(pooled_radiation_samples) == 1 else "pooled_flat"
        return r, r, source
    if cal_rad.get("r_rad_default_night") and cal_rad.get("r_rad_default_day"):
        return float(cal_rad["r_rad_default_night"]), float(cal_rad["r_rad_default_day"]), \
            "default_calibrated"
    return default_r_rad, default_r_rad, "default_unmeasured"


def combine(points_measurements: list[PointMeasurement], run_start: datetime.datetime,
            run_end: datetime.datetime, dt: float, stepra: int, calibration: dict | None = None,
            sample_points: list | None = None, case_dir: str | None = None, nl: dict | None = None,
            ranks: int | None = None, nodes: int | None = None, cost_series: Any = None) -> dict:
    """Whole-run prediction from >= 1 PointMeasurement -> {compute_total_s, init_s, init_note,
    total_s, range_low_s, range_high_s, drift_pct, r_night, r_day, r_source, steps_n, points}.
    Arithmetic of coldstart_sample.predicted_compute_total/resolve_r_rad and
    wtune_predict.predict_init; range = +-drift_band_pct of total_s (no I/O term). cost_series
    (decision 0060) adds stratified/regression totals; total_s stays interpolated."""
    if not points_measurements:
        raise ProbePlanError("combine: no point measurements given")
    calibration = calibration or {}
    steps_n = int(round((run_end - run_start).total_seconds() / dt))

    anchors = [((pm.at - run_start).total_seconds(), pm.s_normal) for pm in points_measurements]

    pooled_radiation_samples = []
    for pm in points_measurements:
        if pm.n_rad and pm.r is not None:
            frac = predict.sunlit_fraction(sample_points, pm.at) if sample_points else 0.5
            pooled_radiation_samples.append((frac if frac is not None else 0.5, pm.r))

    r_night, r_day, r_source = _resolve_r_rad_from_points(pooled_radiation_samples, calibration)

    compute_total_s, _s_values = coldstart.predicted_compute_total(
        steps_n, dt, stepra, run_start, anchors, sample_points, r_night, r_day)

    init_s = 0.0
    init_note = None
    if nl is not None:
        try:
            init = predict.predict_init(case_dir, nl, calibration, ranks=ranks, nodes=nodes)
            init_s = init["total_s"]
        except Exception as e:
            init_note = "init prediction unavailable: %s: %s" % (type(e).__name__, e)

    total_s = compute_total_s + init_s
    drift_pct = calibration.get("drift_band_pct", 5.0)
    drift_abs_s = total_s * (drift_pct / 100.0)
    low, high = max(0.0, total_s - drift_abs_s), total_s + drift_abs_s

    table = [dict(at=pm.at, reason=pm.reason, source=pm.source, s_normal=pm.s_normal,
                  n_rad=pm.n_rad, r=pm.r) for pm in points_measurements]

    result = dict(compute_total_s=compute_total_s, init_s=init_s, init_note=init_note,
                  total_s=total_s, range_low_s=low, range_high_s=high, drift_pct=drift_pct,
                  r_night=r_night, r_day=r_day, r_source=r_source, steps_n=steps_n, points=table)
    if cost_series is not None and cost_series.usable:
        import wtune_cost_drivers
        wtune_cost_drivers.attach_weighted_totals(result, cost_series, points_measurements,
                                                  run_start, run_end, dt, stepra, sample_points)
    return result


# ---------------------------------------------------------------------------
# rebuild segments -- per-point real-data reconstruction through coldstart_sample's
# build_plan(times=[...]) pin
# ---------------------------------------------------------------------------
def plan_rebuild_segments(case_dir: str, points: list[ProbePoint], config: dict, out_dir: str,
                          spinup_min: float = 60, date_shift_to: datetime.datetime | None = None,
                          namelist: str | None = None, grib_dir: str | None = None,
                          forcing: str | None = None, now: datetime.datetime | None = None,
                          measure_steps: int | None = None, real_only: bool = False,
                          io_probe: bool = False) -> dict:
    """Plan (build_plan(times=...), no fetch/stage/run) every rebuild point of one prediction for
    one config as ONE job (decision 0054). Each point's grid start T = M - spinup_min*60 is
    recovered algebraically (spinup_min must match plan_points'), not re-aligned (that would
    round up). date_shift_to: the analog's weather on the stated timeline. Records
    rebuild_requested_at/rebuild_snapped_to. ProbePlanError when no forcing covers the target date.
    real_only/io_probe are forwarded to build_plan."""
    if not points:
        raise ProbePlanError("plan_rebuild_segments: no points given")
    spinup_s = spinup_min * 60.0
    target = date_shift_to or points[0].at
    try:
        chain = coldstart.resolve_forcing_chain_for(target, grib_dir=grib_dir, forcing_pin=forcing,
                                                     now=now)
    except Exception as e:
        raise ProbePlanError("plan_rebuild_segments: forcing-chain lookup for %s failed (%s: %s)"
                              % (target.isoformat(), type(e).__name__, e))
    if not chain:
        raise ProbePlanError("plan_rebuild_segments: no forcing source covers %s"
                              % target.isoformat())
    starts = [p.at - datetime.timedelta(seconds=spinup_s) for p in points]
    if date_shift_to:
        shift_s = (date_shift_to - points[0].at).total_seconds()
        pinned = [t + datetime.timedelta(seconds=shift_s) for t in starts]
    else:
        pinned = starts
    plan = coldstart.build_plan(
        case_dir, out_dir, k=len(points), nodes=config.get("nodes", 1), ranks=config["ranks"],
        omp=config.get("omp", 1), nx=config.get("nx"), ny=config.get("ny"),
        numtiles=config.get("numtiles", 1), measure_steps=measure_steps,
        spinup_min=spinup_min, namelist=namelist, grib_dir=grib_dir, forcing=forcing, now=now,
        date_shift_to=(date_shift_to.strftime(_ISO_FMT) if date_shift_to else None),
        times=pinned, real_only=real_only,
        io_probe=bool(io_probe or any(getattr(p, "io_probe", False) for p in points)))
    plan["rebuild_requested_at"] = [(p.requested_at or p.at).isoformat() for p in points]
    plan["rebuild_snapped_to"] = [p.at.isoformat() for p in points]   # M, the measured point
    plan["rebuild_starts_at"] = [t.isoformat() for t in pinned]        # T, the forcing-grid start
    return plan
