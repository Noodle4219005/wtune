#!/usr/bin/env python3
"""wtune_watch.py -- read-only monitor for a running (or finished) WRF run directory.

    wtune watch RUNDIR [--interval SEC=60] [--once] [--json] [--replay-fraction F[,F...]]
        [--init-limit SEC=600] [--hang-factor K=10] [--hang-min SEC=300]

Reads ONLY RUNDIR/namelist.input and RUNDIR/rsl.error.0000 + rsl.out.0000 (reopened and
re-stat'ed on every poll: on NFS a long-lived handle can serve stale content). Never writes,
never signals the job.

Progress
    d01 'Timing for main: time <T> on domain   1: <s> elapsed seconds' lines; T is the END of
    that step. d01 elapsed INCLUDES the nested-domain integration and history/restart writes of
    that d01 step (zhengzhou, 3 domains: sum d01 = 17789.5 s vs job wall 17827.6 s; summing all
    domains double counts to 47794 s), so wall elapsed so far ~= init + sum(d01 elapsed).
    rsl.error.0000 and rsl.out.0000 can each be truncated on high-print cases: both are read and
    the one with more d01 Timing lines wins (tie -> rsl.error.0000).

Remaining time
    remaining d01 steps x rate, rate = MEAN d01 elapsed over the most recent
    W = max(2*stepra, 50, 25 % of completed) steps, W rounded up to a whole number of periods of
    every periodic cost (d01 radiation stepra, every domain's radt and history interval, all
    expressed in d01 steps; their LCM P) when P fits in the completed steps, else to a whole
    number of stepra. A mean over whole periods carries the radiation and write spikes at their
    true duty cycle; a median would drop them.

Warnings (code, text); exit 3 on FAILED, 2 on HANG/HANG_INIT/NEST_STOPPED/NEST_END_EARLY, else 0
    HANG          live: newest mtime of the chosen rsl file older than
                  max(hang_min, hang_factor x median d01 step of the rate window).
    HANG_INIT     no d01 Timing line yet and the rsl file older than init_limit (birth time when
                  the platform reports it, else ctime). The known silent hang (elevconv, scc:
                  missing iofields file) ends with 'Problem opening storm_diags_d01.txt' after
                  'med_initialdata_input: calling input_input' and never prints a Timing line;
                  a 'Problem opening' tail line is quoted as the hint.
    NEST_END_EARLY  from namelist.input alone: an active nest whose own end_* datetime is before
                  d01's run end (run_* wins for d01 only; yagi truth 1038 edited only run_*, so
                  d02/d03 stopped while d01 kept advancing and the job looked healthy).
    NEST_STOPPED  a nest's last sim time is behind d01's by > 2 parent steps while its own
                  namelist end is still ahead of it.
    FAILED        'FATAL CALLED' or 'forrtl' in the tail of either 0000 file, or a NaN Timing.
    CFL           informational: 'cfl' lines in the chosen file grew since the previous poll
                  (--once / first poll: any present).
    DONE is a status ('SUCCESS COMPLETE WRF'), not a warning.

Replay (--replay-fraction F, implies --once; hang/live checks off)
    Pretend the run is at fraction F of its observed d01 Timing lines: only d01 lines up to that
    point and nest lines up to the same sim time are used; reports the estimate, the actual
    remaining = sum of the later d01 elapsed, and error % = (est - actual) / actual where est is
    rate x the number of later d01 lines (equal to the remaining-time estimate on a complete log;
    on a partial log it compares only the span the log covers, flagged actual_partial).
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import wtune_predict as wp  # noqa: E402  (namelist helpers, stepra, run length)

RSL_FILES = ("rsl.error.0000", "rsl.out.0000")  # tie -> the first
TAIL_LINES = 200
WRF_FMT = "%Y-%m-%d_%H:%M:%S"
# Elapsed token is \S+ (not [\d.]+ as in wtune_predict._RSL_TIMING_RE) so NaN lines are seen.
_TIMING_RE = re.compile(
    r"Timing for main: time (\S+) on domain\s+(\d+):\s+(\S+)\s+elapsed seconds")
BLOCKING = ("HANG", "HANG_INIT", "NEST_STOPPED", "NEST_END_EARLY")


def _fmt_t(t):
    return t.strftime(WRF_FMT) if t is not None else None


def _hms(s):
    if s is None:
        return "?"
    s = int(round(s))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def parse_timing(text):
    """All 'Timing for main' lines -> [(domain, sim_datetime_or_None, elapsed), ...] in file order.
    elapsed is float('nan') for a NaN token; lines whose token is neither a number nor NaN
    (Fortran '*****' overflow) are dropped."""
    out = []
    for m in _TIMING_RE.finditer(text):
        try:
            sec = float(m.group(3))
        except ValueError:
            continue
        try:
            t = dt.datetime.strptime(m.group(1), WRF_FMT)
        except ValueError:
            t = None
        out.append((int(m.group(2)), t, sec))
    return out


def read_rsl(rundir):
    """Read both rank-0 rsl files fresh. Returns (chosen, all) where each entry is
    {name, path, text, stat, timing, n_d01}; chosen is None when neither exists. Chosen = more d01
    Timing lines, tie -> rsl.error.0000 (either one can be truncated on high-print cases)."""
    found = []
    for name in RSL_FILES:
        path = os.path.join(rundir, name)
        try:
            st = os.stat(path)
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        timing = parse_timing(text)
        found.append({"name": name, "path": path, "text": text, "stat": st, "timing": timing,
                      "n_d01": sum(1 for d, _, _ in timing if d == 1)})
    chosen = None
    for r in found:
        if chosen is None or r["n_d01"] > chosen["n_d01"]:
            chosen = r
    return chosen, found


def _dom_datetime(nl, prefix, idx):
    """Domain idx's (0-based) <prefix>_year..second datetime, WRF broadcast-last-value, or None."""
    y = wp.g_dom(nl, "time_control", prefix + "_year", idx)
    mo = wp.g_dom(nl, "time_control", prefix + "_month", idx)
    d = wp.g_dom(nl, "time_control", prefix + "_day", idx)
    if y is None or mo is None or d is None:
        return None
    h = wp.g_dom(nl, "time_control", prefix + "_hour", idx, 0) or 0
    mi = wp.g_dom(nl, "time_control", prefix + "_minute", idx, 0) or 0
    s = wp.g_dom(nl, "time_control", prefix + "_second", idx, 0) or 0
    return dt.datetime(int(y), int(mo), int(d)) + dt.timedelta(hours=int(h), minutes=int(mi),
                                                               seconds=int(s))


def _int_period(x):
    """x (d01 steps) -> int period when x is a positive integer (to 1e-6), else None."""
    if x is None or x < 1:
        return None
    r = int(round(x))
    return r if abs(x - r) < 1e-6 else None


def run_info(nl):
    """Run geometry from a parsed namelist: d01 start/end (run_* wins, wtune_runlength rule), dt,
    total d01 steps, stepra, the periodic-cost periods in d01 steps, and per active domain
    {id, parent, dt, parent_dt, start, end} where a nest's end is its OWN end_* datetime (None when
    absent) -- WRF applies run_* to d01 only."""
    steps, run_s, dt_s, _ = wp.get_d01_steps(nl)
    start = wp.get_time_control_datetime(nl, "start")
    end = start + dt.timedelta(seconds=run_s) if start is not None else None
    periods = []
    stepra = None
    radt = wp.g_dom(nl, "physics", "radt", 0)
    if radt:
        stepra = wp.steps_per_radiation_call(float(radt), dt_s)
        periods.append(stepra)
    doms = []
    for i in range(wp.get_max_dom(nl)):
        if i == 0:
            ddt, parent, pdt = dt_s, None, None
            dstart, dend = start, end
        else:
            parent = int(wp.g_dom(nl, "domains", "parent_id", i, 1) or 1)
            pdt = doms[parent - 1]["dt"] if 0 < parent <= len(doms) else dt_s
            ddt = pdt / wp.parent_time_step_ratio(nl, i, default=1)
            dstart = _dom_datetime(nl, "start", i) or start
            dend = _dom_datetime(nl, "end", i)
            r = wp.g_dom(nl, "physics", "radt", i)
            if r:
                p = _int_period(float(r) * 60.0 / dt_s)
                if p:
                    periods.append(p)
        hist_min, _ = wp.get_history_interval_minutes(nl, i)
        if hist_min and hist_min > 0:
            p = _int_period(hist_min * 60.0 / dt_s)
            if p:
                periods.append(p)
        doms.append({"id": i + 1, "parent": parent, "dt": ddt, "parent_dt": pdt,
                     "start": dstart, "end": dend})
    return {"start": start, "end": end, "dt": dt_s, "run_s": run_s, "total_steps": steps,
            "stepra": stepra, "periods": sorted(set(periods)), "domains": doms}


def rate_window(elapsed, stepra, periods):
    """(rate, W, median) over the most recent W of the completed d01 step times `elapsed`.
    W = max(2*stepra, 50, 25 % of completed), rounded up to a multiple of P = LCM(periods) when
    that fits in the completed steps, else to a multiple of stepra that fits; capped at completed.
    rate is the MEAN (radiation/write spikes kept at their duty cycle), median feeds HANG."""
    n = len(elapsed)
    if n == 0:
        return None, 0, None
    w = max(2 * (stepra or 0), 50, int(math.ceil(0.25 * n)))
    lcm = 1
    for p in periods:
        lcm = lcm * p // math.gcd(lcm, p)
    for unit in (lcm, stepra or 1):
        if unit and unit <= n:
            w = min(int(math.ceil(w / unit)) * unit, (n // unit) * unit)
            break
    w = max(1, min(w, n))
    win = elapsed[-w:]
    return sum(win) / w, w, statistics.median(win)


def _tail(text, n=TAIL_LINES):
    return text.splitlines()[-n:]


def _file_age_origin(st):
    """First-creation time when the platform reports it (st_birthtime), else ctime."""
    return getattr(st, "st_birthtime", None) or st.st_ctime


def snapshot(rundir, now=None, replay_fraction=None, hang_factor=10.0, hang_min=300.0,
             init_limit=600.0, prev_cfl=None):
    """One read-only observation of RUNDIR -> JSON-serializable dict. Pure function of the files'
    contents/stat and `now` (epoch seconds; default time.time()). replay_fraction switches to
    replay (no live checks). prev_cfl: the previous poll's cfl count (None -> warn on any)."""
    now = time.time() if now is None else now
    replay = replay_fraction is not None
    warnings = []

    def warn(code, text):
        warnings.append({"code": code, "text": text})

    snap = {"rundir": os.path.abspath(rundir), "now": now, "mode": "replay" if replay else "live",
            "status": None, "warnings": warnings}

    info = None
    nml = os.path.join(rundir, "namelist.input")
    try:
        info = run_info(wp.parse_namelist(nml))
    except (OSError, wp.PredictError, ValueError) as e:
        warn("NO_NAMELIST", "namelist.input unusable: %s" % e)
    if info:
        snap.update(run_start=_fmt_t(info["start"]), run_end=_fmt_t(info["end"]), dt=info["dt"],
                    stepra=info["stepra"], total_steps=info["total_steps"],
                    periods=info["periods"])
        for d in info["domains"][1:]:
            if d["end"] is not None and info["end"] is not None and d["end"] < info["end"]:
                warn("NEST_END_EARLY",
                     "d%02d end_* %s is before d01 run end %s (run_* applies to d01 only; "
                     "set the nest end_* too)" % (d["id"], _fmt_t(d["end"]), _fmt_t(info["end"])))

    chosen, found = read_rsl(rundir)
    if chosen is None:
        snap["status"] = "no_rsl"
        return snap
    snap["rsl_file"] = chosen["name"]
    snap["rsl_d01_lines"] = {r["name"]: r["n_d01"] for r in found}
    timing = chosen["timing"]
    d01 = [(t, s) for d, t, s in timing if d == 1]
    replay_cut = None
    if replay:
        k = min(len(d01), max(1, int(round(replay_fraction * len(d01))))) if d01 else 0
        rest = d01[k:]
        d01 = d01[:k]
        replay_cut = d01[-1][0] if d01 else None
        snap["replay_fraction"] = replay_fraction
        snap["actual_remaining_s"] = sum(s for _, s in rest if not math.isnan(s))
        snap["actual_partial"] = bool(
            info and info["end"] and (d01 + rest) and (d01 + rest)[-1][0] is not None
            and (d01 + rest)[-1][0] < info["end"])

    tails = {r["name"]: _tail(r["text"]) for r in found}
    tail_all = [ln for r in found for ln in tails[r["name"]]]
    success = any("SUCCESS COMPLETE WRF" in ln for ln in tail_all)
    fatal = [ln for ln in tail_all if "FATAL CALLED" in ln or "forrtl" in ln]
    nan = any(math.isnan(s) for _, s in d01)
    problem = [ln.strip() for ln in tails[chosen["name"]] if "Problem opening" in ln]

    if not replay:
        if fatal or nan:
            warn("FAILED", fatal[-1].strip() if fatal else "NaN in a d01 Timing line")
        snap["status"] = ("failed" if fatal or nan else "done" if success
                          else "running" if d01 else "init")
    else:
        snap["status"] = "replay"

    # HANG_INIT
    if not d01 and not (not replay and (success or fatal)):
        hint = (" hint: %r (missing file, e.g. an iofields_filename target)" % problem[-1]
                if problem else "")
        if replay:
            if not success:
                warn("HANG_INIT", "no d01 Timing line in the log" + hint)
        else:
            age = now - _file_age_origin(chosen["stat"])
            if age > init_limit:
                warn("HANG_INIT", "no d01 Timing line after %s (limit %s)%s"
                     % (_hms(age), _hms(init_limit), hint))

    elapsed = [s for _, s in d01 if not math.isnan(s)]
    rate, w, med = rate_window(elapsed, info["stepra"] if info else None,
                               info["periods"] if info else [])
    last_t = d01[-1][0] if d01 else None
    completed = len(d01)
    if info and last_t is not None:
        completed = int(round((last_t - info["start"]).total_seconds() / info["dt"]))
    snap.update(n_d01_steps=len(d01), completed_steps=completed, d01_last_time=_fmt_t(last_t),
                sum_d01_elapsed_s=sum(elapsed), rate_s_per_step=rate, rate_window=w,
                median_step_s=med)
    remaining = None
    if info and rate is not None:
        rem_steps = max(0, info["total_steps"] - completed)
        remaining = rem_steps * rate
        snap["fraction"] = min(1.0, completed / info["total_steps"]) if info["total_steps"] else None
        snap["remaining_steps"] = rem_steps
    if snap["status"] == "done":
        remaining = 0.0
        snap["fraction"] = 1.0
    snap["remaining_s"] = remaining
    if not replay and remaining is not None:
        snap["eta"] = dt.datetime.fromtimestamp(now + remaining).strftime("%Y-%m-%d %H:%M:%S")
    if replay:
        # Compare over the steps the log actually covers (a partial log's actual is not the rest
        # of the run): estimate for those rest steps = n_rest x rate.
        act = snap["actual_remaining_s"]
        est = len(rest) * rate if rate is not None else None
        snap["est_for_actual_s"] = est
        snap["error_pct"] = 100.0 * (est - act) / act if est is not None and act > 0 else None

    # HANG (live, progressing)
    if not replay and snap["status"] == "running" and med is not None:
        limit = max(hang_min, hang_factor * med)
        age = now - chosen["stat"].st_mtime
        snap["log_age_s"] = age
        if age > limit:
            warn("HANG", "no rsl output for %s (limit %s = max(%gs, %g x median step %.2fs)); "
                 "last d01 time %s" % (_hms(age), _hms(limit), hang_min, hang_factor, med,
                                       _fmt_t(last_t)))

    # Per-domain progress and NEST_STOPPED
    doms = []
    if info:
        for d in info["domains"]:
            ts = [t for dd, t, _ in timing if dd == d["id"] and t is not None
                  and (replay_cut is None or t <= replay_cut)]
            if replay and replay_cut is None:
                ts = []
            dlast = max(ts) if ts else None
            doms.append({"id": d["id"], "last_time": _fmt_t(dlast), "n_lines": len(ts),
                         "start": _fmt_t(d["start"]), "end": _fmt_t(d["end"] or info["end"])})
            if d["id"] == 1 or last_t is None or snap["status"] == "failed":
                continue
            ref = dlast
            if ref is None:
                if d["start"] is None or last_t <= d["start"]:
                    continue
                ref = d["start"]
            dend = d["end"] or info["end"]
            if ((last_t - ref).total_seconds() > 2 * d["parent_dt"]
                    and (dend is None or dend > ref)):
                warn("NEST_STOPPED", "d%02d last time %s is behind d01 %s while its end is %s"
                     % (d["id"], _fmt_t(ref), _fmt_t(last_t), _fmt_t(dend)))
    snap["domains"] = doms

    # CFL (informational; live only -- the whole file would leak post-cut lines into a replay)
    if replay:
        return snap
    cfl = sum(1 for ln in chosen["text"].splitlines() if "cfl" in ln.lower())
    snap["cfl_count"] = cfl
    if cfl > (0 if prev_cfl is None else prev_cfl):
        warn("CFL", "%d cfl line(s) in %s%s" % (
            cfl, chosen["name"], "" if prev_cfl is None else " (+%d since last poll)"
            % (cfl - prev_cfl)))
    return snap


def exit_code(snap):
    codes = {w["code"] for w in snap["warnings"]}
    if "FAILED" in codes:
        return 3
    if codes & set(BLOCKING):
        return 2
    return 0


def format_line(snap):
    """One compact text line per snapshot."""
    frac = snap.get("fraction")
    parts = [snap["status"].upper()]
    if snap["mode"] == "replay":
        parts = ["replay F=%g" % snap["replay_fraction"]]
    parts.append("%5.1f%%" % (100 * frac) if frac is not None else "  ?%")
    if snap.get("d01_last_time"):
        parts.append("sim %s (%s/%s)" % (snap["d01_last_time"], snap.get("completed_steps"),
                                          snap.get("total_steps", "?")))
    if snap.get("rate_s_per_step") is not None:
        parts.append("elapsed %s rate %.3fs/step[w=%d]" % (
            _hms(snap["sum_d01_elapsed_s"]), snap["rate_s_per_step"], snap["rate_window"]))
    if snap.get("remaining_s") is not None:
        parts.append("remaining %s" % _hms(snap["remaining_s"]))
    if snap.get("eta"):
        parts.append("ETA %s" % snap["eta"])
    if snap["mode"] == "replay":
        if snap.get("actual_partial"):
            parts.append("est-to-log-end %s" % _hms(snap.get("est_for_actual_s")))
        parts.append("actual %s%s" % (_hms(snap["actual_remaining_s"]),
                                      " (partial log)" if snap.get("actual_partial") else ""))
        if snap.get("error_pct") is not None:
            parts.append("err %+.2f%%" % snap["error_pct"])
    line = " ".join(parts)
    if snap["warnings"]:
        line += " | " + "; ".join("%s: %s" % (w["code"], w["text"]) for w in snap["warnings"])
    return line


def _emit(snap, as_json):
    print(json.dumps(snap, sort_keys=True) if as_json else format_line(snap), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wtune watch", description=__doc__.split("\n\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--interval", type=float, default=60.0, help="poll period, s (default 60)")
    ap.add_argument("--once", action="store_true", help="one snapshot, then exit")
    ap.add_argument("--json", action="store_true", help="one JSON object per snapshot")
    ap.add_argument("--replay-fraction", default=None,
                    help="F or F1,F2,... in (0,1]: replay a finished log at that fraction of its "
                         "d01 steps and compare the estimate with the actual remaining (implies --once)")
    ap.add_argument("--init-limit", type=float, default=600.0,
                    help="HANG_INIT after this many s with no d01 Timing line (default 600)")
    ap.add_argument("--hang-factor", type=float, default=10.0)
    ap.add_argument("--hang-min", type=float, default=300.0)
    a = ap.parse_args(argv)
    kw = dict(hang_factor=a.hang_factor, hang_min=a.hang_min, init_limit=a.init_limit)

    if a.replay_fraction is not None:
        try:
            fracs = [float(x) for x in a.replay_fraction.split(",") if x.strip()]
        except ValueError:
            ap.error("--replay-fraction: expected F or a comma list of floats")
        if not fracs or any(not 0 < f <= 1 for f in fracs):
            ap.error("--replay-fraction values must be in (0, 1]")
        rc = 0
        for f in fracs:
            snap = snapshot(a.rundir, replay_fraction=f, **kw)
            _emit(snap, a.json)
            rc = max(rc, exit_code(snap))
        return rc

    prev_cfl, snap = None, None
    try:
        while True:
            snap = snapshot(a.rundir, prev_cfl=prev_cfl, **kw)
            prev_cfl = snap.get("cfl_count", prev_cfl)
            _emit(snap, a.json)
            if a.once or snap["status"] in ("done", "failed"):
                return exit_code(snap)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return exit_code(snap) if snap else 0


if __name__ == "__main__":
    sys.exit(main())
