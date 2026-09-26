#!/usr/bin/env python3
"""wtune_probe_plan.py -- multi-point probe planning + whole-run combine (2026-09-26 redesign).

Motivation (see this task's report for the full evidence): the tuner used to probe every
candidate for a fixed 20 steps from t0 and predict the whole run from that ONE point. On a real
conus25 run the confirmed winner's t0-only probe missed every radiation step entirely (radt=10min,
dt=15s -> a radiation step fires every 40 steps; 20 steps never reaches one) AND missed the
cold-start spin-up cost ramp (no clouds/precip at t0 -> microphysics is artificially cheap for
~1h), so the predictor fell back to default radiation ratios and under-predicted the run by 9.6%.
This module fixes BOTH failure modes with two independent pieces:

  plan_points()  -- WHERE to place one or more probe measurement points along a case's own
                    timeline (the winner's full-run PREDICTION; see wtune_tune.build_full_run_
                    prediction). Placement borrows tools/coldstart_sample.py's own window-
                    selection policy (t0/night/day/late/precip_peak/farthest-fill,
                    select_windows) and its own radiation-window sizing rule (compute_window_
                    seconds), applied to the run's OWN timeline instead of a fixed grid.
  combine()      -- turns measured (time, s_normal, n_rad, r) points into a whole-run compute
                    prediction: piecewise-linear interpolation of s_normal between points
                    (constant extrapolation outside the measured span) plus a radiation-step
                    correction using a ratio r resolved from the points' OWN pooled radiation
                    samples (day/night split when >=2 points give both regimes) -- both pieces of
                    arithmetic are tools/coldstart_sample.py's own predicted_compute_total/
                    resolve_r_rad/interpolate_step_costs, not reimplemented here.

Point SOURCES (probe.sources, preference order -- see wtune_config.py's schema):
  trajectory -- the point is within reach of the RANKING layer's own anchor spin-up restart
                chain (see build_restart_start_namelist_text / campaign/configs.py's
                edit_namelist restart_from/write_restart_interval params, 2026-09-26): extending
                that chain one more measure-window is cheap, so points inside
                trajectory_horizon_s prefer it.
  restart     -- a user-supplied wrfrst file for exactly this time already exists under the
                case's directory (WRF's own wrfrst_d01_<time> naming).
  rebuild     -- (alias: "coldstart", accepted for backward compatibility with pre-2026-09-26
                configs) re-derive the point from real historical forcing: fetch -> WPS/real.exe
                regenerates wrfinput/wrfbdy AT that exact time -> the cold-start spin-up is run
                and thrown away -> the candidate is measured on the freshly rebuilt segment.
                This is tools/coldstart_sample.py's plan/fetch/stage/run.sbatch machinery,
                elevated here to a first-class, multi-point-addressable operation via
                plan_rebuild_segments() below (build_plan's own `times=` pin, added
                independently of this feature) plus run_rebuild_points() in wtune_tune.py, which
                fetches/stages/submits/scores it -- not a fallback, the DEFAULT choice once a
                point is outside the trajectory horizon, for any case
                whose date resolves to real, fetchable forcing. Only usable for a case whose
                start date is a real calendar date; resolve_rebuild_target_date() below detects a
                FICTIONAL case date (no forcing source covers it) and substitutes an analog real
                date via tools/case_rebuild.py's cmd_analog, disclosing the substitution -- never
                silently measuring the wrong date's weather.
If no source in probe.sources can serve a point, the point is DROPPED and the drop is reported
(plan_points' `notes` return value) -- never silent.

This module never submits a job or touches the network; every function here is a pure
computation over already-known inputs (namelist text/dict, datetimes, already-measured numbers)
or a read of files already on disk (case_dir globbing for an existing wrfrst). Actual measurement
execution (running WPS/real.exe/wrf.exe) is unchanged and lives where it always has: campaign/
strict_campaign.sbatch (RANKING layer) and tools/coldstart_sample.py's stage_plan/render_run_sbatch
(rebuild layer) -- see this task's report for the exact wiring status of each.
"""
from __future__ import annotations

import dataclasses
import datetime
import math
import os
import re
import statistics

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
    # requested_at (2026-09-26, cadence-aware rebuild placement): the ORIGINALLY targeted time
    # (P) that resolution actually served at `at` (M). Equal to `at` for every source except
    # 'rebuild', where a rebuild segment starts at a forcing-grid time T and is only measured
    # after its own spin-up (M = T + spinup_s) -- see resolve_rebuild_point -- so the measured
    # point can differ from what was originally targeted; both are always reported.
    requested_at: "datetime.datetime | None" = None


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
# step-count sizing (shared by the RANKING layer's probe.steps='auto' and this module's own
# per-point measurement window -- SAME formula, deliberately: both need >=1 radiation step
# inside a short probe)
# ---------------------------------------------------------------------------
def steps_for_stepra(stepra):
    """max(30, stepra + 10) -- a probe window this long always contains >= 1 radiation step
    (WRF calls radiation at itimestep 1 and every stepra-th step after), with 10 steps of slack
    so the window isn't cut exactly at the boundary."""
    return max(30, int(stepra) + 10)


def resolve_ranking_steps(radt_minutes, dt_seconds):
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


def canonical_source(name):
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


def parse_explicit_point(spec, run_start, run_end):
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


def resolve_rebuild_point(p_at, run_start, run_end, spinup_s, rebuild_measure_window_s,
                           forcing_interval_h):
    """(M, T) for a rebuild candidate targeting P=p_at (2026-09-26 cadence-aware placement rule):
    a rebuild segment starts AT a forcing-grid time T and is only usable after its own spin-up --
    the point actually MEASURED is M = T + spinup_s, never P snapped onto the grid. T is the
    LATEST forcing-grid time (multiples of forcing_interval_h hours since the UNIX epoch -- the
    same grid tools/coldstart_sample.align_to_interval uses, but FLOORED here instead of
    ceiling'd: 'P between grid times picks the earlier T') with T + spinup_s <= P.

    `rebuild_measure_window_s` MUST be the window length the rebuild job will ACTUALLY measure
    with (tools/coldstart_sample.compute_window_seconds's own max(30, 2*stepra)*dt sizing, NOT
    plan_points' own steps_for_stepra(stepra) window) -- using the wrong, smaller window here
    would pass planning-time validation and then fail for real inside build_plan's own 'time +
    window does not fit in the run' check (this task's report: exactly the bug this rule fixes).

    Returns (None, T) -- dropped -- when M + rebuild_measure_window_s > run_end (M's own
    measurement would overrun the run). T is always returned, even when M is None, so the
    caller/report can show what was tried. Never raises."""
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


def resolve_point_source(at, run_start, run_end, sources_pref, spinup_s=0.0,
                          measure_window_s=0.0, rebuild_measure_window_s=None, case_dir=None,
                          trajectory_horizon_s=None, is_real_calendar=None,
                          forcing_interval_h=None):
    """First entry of `sources_pref` (already alias-normalized) that can serve a probe point
    requested at `at` (=P). Returns (source, effective_at, requested_at): effective_at == at for
    'trajectory'/'restart' (those measure exactly the requested time); for 'rebuild',
    effective_at is M from resolve_rebuild_point (may differ from `at` -- see that function's
    docstring) and requested_at is always the original `at`. Returns (None, None, at) when
    nothing in sources_pref can serve it (caller drops the point).

      'trajectory' -- available when trajectory_horizon_s is given and
                      (at - run_start) <= trajectory_horizon_s (the anchor's own spin-up restart
                      chain, extended one more measure-window at a time, still reaches this
                      point within the probe budget -- see plan_points).
      'restart'    -- available when case_dir is given and a wrfrst_d01_<at> file already
                      exists there (a user-supplied restart, not one this tuner wrote).
      'rebuild'    -- available when the point's calendar date is real (is_real_calendar, or a
                      >= 1900 heuristic when the caller does not resolve it more precisely --
                      see resolve_rebuild_target_date for the real check, which also handles a
                      FICTIONAL case date via an analog substitution before this function is
                      ever asked to place the point) AND forcing_interval_h is known AND
                      resolve_rebuild_point finds a usable M for `at` (never overruns run_end).
    """
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


def resolve_rebuild_target_date(case_dir, at, namelist=None, forcing_probe_fn=None,
                                 analog_fn=None, grib_dir=None):
    """(effective_date, substituted, note). Checks whether ANY forcing source covers `at`
    (tools/coldstart_sample.resolve_forcing_chain_for) -- an empty chain means the case's own
    date is FICTIONAL (an idealized case, or a real-looking date with no matching forcing
    archive), and `at` cannot honestly be rebuilt. In that case this calls
    tools/case_rebuild.cmd_analog (best-effort: any failure -- no wrfinput, no network, missing
    deps -- is caught, never raised, and reported as 'no analog available') to pick a real
    analog date instead, and returns substituted=True with a note identifying the swap -- the
    report MUST surface this note; a 'rebuild' point must never silently measure the wrong
    date's weather. `analog_fn` overrides the analog lookup (offline tests inject a fake here,
    never touching the network); default is case_rebuild.cmd_analog."""
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
    """points: sorted [[effective_at, reason, source, requested_at, priority_idx], ...] (lists,
    mutated in place -- same convention as _merge_close_points). Consecutive points within
    min_gap_s of the last KEPT point's effective_at are merged into it: reason gains the
    merged-in one (tagged with its own source, and its own requested time when that differs from
    its effective time -- e.g. a rebuild point whose M coincides with an existing trajectory
    point's own time); priority_idx becomes the MIN of the two, so a lower-priority auto pick
    that happens to collapse onto an earlier, higher-priority one is never treated as lower-
    priority than it now effectively is for budget trimming."""
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


def plan_points(case_nl, run_start, run_end, dt, stepra, spinup_s, cfg,
                 sample_points=None, precip_lookup=None, s_est=None,
                 predicted_run_s=None, case_dir=None, forcing_interval_h=None,
                 grib_dir=None):
    """Places one or more probe measurement points on [run_start, run_end]. Returns
    (points: [ProbePoint, ...], notes: [str, ...]) -- notes covers short-run fallback, budget
    capping, and any point dropped for lack of a usable source; never raises for THOSE
    conditions. Raises ProbePlanError only for a malformed/out-of-range EXPLICIT probe.points
    entry (a config error, not a placement judgement call).

    cfg: the resolved probe.* dict (steps, points, max_points, budget_pct, sources, ...; see
    wtune_config.py's schema). case_nl is unused by this function directly today (kept in the
    signature for callers/future placement rules that need namelist facts beyond
    dt/stepra/run_start/run_end -- e.g. per-domain geometry); accepted and ignored otherwise.

    Pipeline (2026-09-26, cadence-aware rebuild placement -- a bug fix to auto placement, not a
    post-hoc change: resolve the forcing chain/cadence for rebuild candidates FIRST, then place,
    then trim by budget, so the returned plan is always executable):
      1. resolve forcing_interval_h (if 'rebuild' is a possible source and not given explicitly --
         tests inject it directly; a real caller lets this resolve from case_dir/grib_dir).
      2. generate raw candidates (auto via tools/coldstart_sample.select_windows, or parsed from
         an explicit probe.points list) -- UNCHANGED, source-agnostic.
      3. resolve EACH candidate's source; for 'rebuild' this ALSO resolves its actual measured
         time M (resolve_rebuild_point) -- may differ from the candidate's own requested time P.
      4. merge points whose EFFECTIVE (post-resolution) times collapse within one measure window
         of each other -- this can merge a rebuild point's M onto an unrelated trajectory/restart
         point's own time, which could only be detected after step 3.
      5. trim to budget_pct (by ORIGINAL selection priority, not by time -- the merge in step 4
         tracks each merged point's best/lowest priority index for exactly this).
    The trajectory/restart sources are unaffected by any of this (their effective time always
    equals what was requested)."""
    notes = []
    steps_cfg = cfg.get("steps", "auto")
    steps = steps_for_stepra(stepra) if steps_cfg in (None, "auto") else int(steps_cfg)
    measure_window_s = steps * dt
    # coldstart_sample's OWN window sizing (compute_window_seconds: max(30, 2*stepra) steps) --
    # NOT plan_points' own steps_for_stepra(stepra) -- is what a rebuild job will ACTUALLY
    # measure with (tools/coldstart_sample.build_plan/render_run_sbatch), so the drop-if-overrun
    # check in resolve_rebuild_point must use THIS window, or planning can approve a point that
    # then fails for real inside build_plan's own bounds check (this task's report).
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
        picks = coldstart.select_windows(candidates, 0.0, measure_window_s, max_points,
                                         sample_points=sample_points,
                                         precip_lookup=precip_lookup)
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


def build_restart_start_namelist_text(namelist_text, restart_time, steps, dt_seconds,
                                       nproc_x=None, nproc_y=None, numtiles=None):
    """The tuner's own PREDICTION-time model of a RANKING-probe candidate namelist that starts
    from an existing WRF restart at `restart_time` instead of the case's own t0 -- the mirror
    image of tools/coldstart_sample.build_wrf_window_namelist_text (which sets restart=.false.
    and re-derives the window from forcing instead). Reuses that module's own set_list/count_key
    primitives -- no new namelist text-editing rule is introduced here (the campaign's OWN copy
    of this rule, used by the real submission pipeline, is campaign/configs.py's
    edit_namelist(restart_from=..., write_restart_interval=..., restart_io_form=...), added
    alongside this module 2026-09-26 -- see that function's docstring; the two are independent
    implementations of the same rule, each tested in its own file).

    Sets, for EVERY active domain: start_* = restart_time, run length = steps*dt_seconds
    (run_days/run_hours/run_minutes=0, run_seconds=steps*dt), restart=.true., restart_interval
    (if present) forced huge (this short probe run must never itself write a restart), and every
    present history_interval* key forced huge (probe measures compute only)."""
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


def build_spinup_writer_namelist_text(namelist_text, spinup_minutes, restart_io_form=None):
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
def first_global_step_from_rsl(rsl_path, run_start, dt):
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


def measurement_from_steps_file(path, stepra, at, reason, source, first_global_step=None):
    """Reads a strict_campaign.sbatch-produced steps.txt (one d01 'Timing for main' wall-clock
    seconds per line, in step order -- the SAME file configs.py step-stats/the RESULT line's
    med=/mean= are computed from) and returns a PointMeasurement.

    Does NOT use wtune_predict.classify_steps_detected here: that function's phase detector
    needs n >= 2*stepra samples to trust its own result (see its docstring) and otherwise falls
    back to an assumed schedule phase 0 -- a RANKING-probe window (probe.steps='auto' ==
    max(30, stepra+10), typically far short of 2*stepra) never has enough data for that detector
    to be reliable. This task's diagnosis (2026-09-26, conus25/G0014 C2/C3 validation) instead
    directly VERIFIED the schedule empirically against both a 52-step restart-started probe and
    the full 1440-step truth run's own rsl: radiation fires at LOCAL integration step i=1 and
    every i with (i mod stepra) == 1 thereafter (i=41,81,... for stepra=40) -- schedule PHASE 1,
    not phase 0 (wtune_predict.classify_steps's own documented i==1-or-i%stepra==0 formula, i.e.
    phase 0, does NOT match this WRF build's actual behaviour). Step 1 (i==1) is dropped from
    BOTH buckets entirely -- startup/restart-read overhead dominates it regardless of whether the
    schedule also computes radiation there -- matching wtune_predict.measure_rad_from_rsl's own
    convention. s_normal/r use the MEDIAN (robust to an occasional history-write step landing on
    the same index), matching step_stats()'s and this task's diagnosis's own methodology.

    first_global_step: when given (see first_global_step_from_rsl), radiation steps are located
    by GLOBAL step number (g % stepra == 1) instead of the local index, which is only correct
    for windows that start on the radiation cycle (t0, or a restart at a multiple of radt)."""
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
    """(r_night, r_day, source). >=2 points: delegate to wtune_predict.fit_day_night_r_rad
    (real day/night least-squares fit when the pooled samples' sunlit fractions actually span
    both regimes; a single-regime projection via calibration's day_night_ratio otherwise) --
    this is 'radiation ratio pooled from points'. Exactly 1 point: use ITS OWN measured ratio
    flat (r_night == r_day == that ratio) -- deliberately NOT fit_day_night_r_rad's single-
    sample day/night projection, so a single-point combine() reproduces today's constant-ratio
    model exactly (see this module's test suite). 0 points (no radiation step observed at any
    measured point): the calibration file's default, same fallback resolve_r_rad already uses."""
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


def combine(points_measurements, run_start, run_end, dt, stepra, calibration=None,
            sample_points=None, case_dir=None, nl=None, ranks=None, nodes=None):
    """points_measurements: [PointMeasurement, ...] (>=1). Returns a dict:
      compute_total_s, init_s, init_note, total_s, range_low_s, range_high_s, drift_pct,
      r_night, r_day, r_source, steps_n, points (per-point table: at/reason/source/s_normal/
      n_rad/r).

    compute_total_s/r_* reuse tools/coldstart_sample.predicted_compute_total/resolve_r_rad's own
    arithmetic verbatim (via _resolve_r_rad_from_points, which only changes the <2-sample
    fallback -- see its docstring); init_s reuses wtune_predict.predict_init. The +-drift-band
    range reproduces wtune_predict.build_prediction's own (compute-only) drift band --
    calibration.drift_band_pct percent of total_s, both directions -- but not its separate I/O
    noise term (this module never estimates I/O)."""
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

    return dict(compute_total_s=compute_total_s, init_s=init_s, init_note=init_note,
                total_s=total_s, range_low_s=low, range_high_s=high, drift_pct=drift_pct,
                r_night=r_night, r_day=r_day, r_source=r_source, steps_n=steps_n, points=table)


# ---------------------------------------------------------------------------
# rebuild segments -- first-class per-point real-data reconstruction (2026-09-26 coordinator
# direction: "rebuild" is a first-class point source, not a fallback). Rather than adding a new
# entry point to tools/coldstart_sample.py, this composes its ALREADY-PUBLIC, already-tested
# `build_plan(..., times=[...])` pin (added independently of this feature -- see build_plan's own
# T1 comment) -- `times` already lets a caller replace the K-auto-window selection with exactly
# the pinned time(s) it wants, which is precisely "one specific time, one specific config".
# ---------------------------------------------------------------------------
def plan_rebuild_segments(case_dir, points, config, out_dir, spinup_min=60, date_shift_to=None,
                           namelist=None, grib_dir=None, forcing=None, now=None):
    """Plans (writes plan.json/namelist.wps/... under out_dir, exactly as build_plan always has
    -- does not fetch/stage/run) ALL rebuild-sourced segments of ONE prediction, for ONE
    decomposition `config` ({"ranks", "omp", "nx", "ny", "nodes"} -- nx/ny/nodes/omp optional,
    same defaults as build_plan), in a SINGLE build_plan(times=[...]) call -- decision 0054 "one
    allocation where feasible": every rebuild point of one prediction shares ONE plan/fetch/stage/
    run.sbatch, so it runs as one job (build_plan's own T1 `times=` pin already replaces its
    K-auto-window selection with exactly the pinned times given, in file order -- render_run_
    sbatch then runs them as sequential windows inside that one allocation).

    `points`: [ProbePoint, ...], all source='rebuild', as produced by plan_points -- each `.at`
    is ALREADY the resolved measured point M = T + spinup_min*60 (resolve_rebuild_point's own
    floor-to-grid rule; `.requested_at` is P, what was originally targeted). `spinup_min` here
    MUST equal the spinup_s (in minutes) plan_points was called with -- it is used to recover
    each point's own forcing-grid start T = M - spinup_min*60s, which is what is actually pinned
    via build_plan's `times=` (a rebuild segment starts AT T and is only usable after ITS OWN
    spin-up; the measured point is T+spinup_min, never M or P re-snapped onto the grid -- see
    resolve_rebuild_point's docstring). T is NOT re-derived via tools/coldstart_sample.
    align_to_interval here (that rounds UP/ceiling and would silently move the point plan_points
    already correctly floor-resolved -- this task's report: exactly the bug this fixes); it is
    recovered algebraically from the already-resolved M instead, and build_plan's own internal
    grid-alignment check (which raises if a pinned time is not exactly on the grid) is the proof
    that stays correct. `date_shift_to` (a datetime): use this ANALOG date's weather for EVERY
    point instead of each point's own calendar date (see resolve_rebuild_target_date) --
    forwarded to build_plan's own date_shift_to, which keeps the PREDICTED run's own timeline
    (run_start/run_end) on the case's stated dates while the fetched/measured weather comes from
    the analog. The returned plan dict records both P and M, in point order, under
    'rebuild_requested_at'/'rebuild_snapped_to'. Raises ProbePlanError when no forcing source
    covers the (shifted) target date at all. Actual fetch/stage/run of the segments this plans
    still goes through coldstart_sample's own resolve_and_fetch_forcing/stage_plan/
    render_run_sbatch, unchanged."""
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
        numtiles=config.get("numtiles", 1),
        spinup_min=spinup_min, namelist=namelist, grib_dir=grib_dir, forcing=forcing, now=now,
        date_shift_to=(date_shift_to.strftime(_ISO_FMT) if date_shift_to else None),
        times=pinned)
    plan["rebuild_requested_at"] = [(p.requested_at or p.at).isoformat() for p in points]
    plan["rebuild_snapped_to"] = [p.at.isoformat() for p in points]   # M, the measured point
    plan["rebuild_starts_at"] = [t.isoformat() for t in pinned]        # T, the forcing-grid start
    return plan
