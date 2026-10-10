#!/usr/bin/env python3
"""wtune_watch.py -- read-only monitor for a running (or finished) WRF run directory.

    wtune watch RUNDIR [--interval SEC=60] [--once] [--json] [--replay-fraction F[,F...]]
        [--init-limit SEC=600] [--hang-factor K=10] [--hang-min SEC=300] [--prediction PRED_JSON]

Reads only namelist.input, rsl.error/out.0000 (reopened every poll: NFS handles go stale), the
wrfout_d01_* names/mtimes and a run root's run_report.json; never writes or signals the job.
Exit 3 on FAILED, 2 on HANG/HANG_INIT/NEST_STOPPED/NEST_END_EARLY, else 0. Estimator, warnings
and replay accuracy: README.md `wtune watch`."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Callable

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_HERE, "tools"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wtune import predict as wp  # noqa: E402  (namelist helpers, stepra, run length)

RSL_FILES = ("rsl.error.0000", "rsl.out.0000")  # tie -> the first
TAIL_LINES = 200   # lines of each rsl file scanned for FATAL / SUCCESS
WRF_FMT = "%Y-%m-%d_%H:%M:%S"
# Elapsed token is \S+ (not [\d.]+ as in wtune_predict._RSL_TIMING_RE) so NaN lines are seen.
# d01 elapsed includes nest integration and that step's writes: wall ~= init + sum(d01 elapsed)
# (zhengzhou, 3 domains: 17789.5 s vs job wall 17827.6 s).
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


def parse_timing(text: str) -> list[tuple[int, dt.datetime | None, float]]:
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


def parse_d01_steps(text: str) -> tuple[list[dict], list[tuple[str, int, float]]]:
    """d01 steps with the writes each one absorbed: ([{t, sec, writes: [(name, dom, s), ...]}],
    trailing_writes). A 'Timing for Writing' line (tools/coldstart_sample._RSL_WRITE_RE) is
    attributed to the NEXT d01 'Timing for main' line (nest Timing lines in between allowed) --
    rsl_write_absorbing_d01_times' rule; writes after the last d01 line (end-of-run frame) are
    trailing. Same d01 line filter as parse_timing, so the lists align."""
    import coldstart_sample as cs
    steps, pending = [], []
    for line in text.splitlines():
        m = cs._RSL_WRITE_RE.search(line)
        if m:
            pending.append((m.group(1).strip(), int(m.group(2)), float(m.group(3))))
            continue
        m = _TIMING_RE.search(line)
        if not m or int(m.group(2)) != 1:
            continue
        try:
            sec = float(m.group(3))
        except ValueError:
            continue
        try:
            t = dt.datetime.strptime(m.group(1), WRF_FMT)
        except ValueError:
            t = None
        steps.append({"t": t, "sec": sec, "writes": pending})
        pending = []
    return steps, pending


def read_rsl(rundir: str) -> tuple[dict | None, list[dict]]:
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


def run_info(nl: dict) -> dict:
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
        doms.append({"id": i + 1, "parent": parent, "dt": ddt, "parent_dt": pdt,
                     "start": dstart, "end": dend})
    info = {"start": start, "end": end, "dt": dt_s, "run_s": run_s, "total_steps": steps,
            "stepra": stepra, "periods": sorted(set(periods)), "domains": doms}
    info["streams"] = write_streams(nl, info)
    return info


def _frame_minutes(begin, interval, end):
    """begin, begin+interval, ... <= end (minutes)."""
    out, j = [], 0
    while interval > 0 and begin + j * interval <= end + 1e-6:
        out.append(begin + j * interval)
        j += 1
    return out


def write_streams(nl: dict, info: dict) -> list[dict]:
    """Every write stream the namelist schedules: [{stream, dom, prefix, frames}], frames in
    sim minutes from the run start. history/auxhistN per domain over begin..end (a resumed run skips
    t=0 unless write_hist_at_0h_rst), restart at j x restart_interval. prefix matches an observed
    'Timing for Writing <name>' line ('restart' is WRF's literal name)."""
    import coldstart_sample as cs
    run_min = info["run_s"] / 60.0
    resumed = bool(wp.g(nl, "time_control", "restart", False))
    skip_t0 = resumed and not wp.g(nl, "time_control", "write_hist_at_0h_rst", False)
    io_hist = wp.g(nl, "time_control", "io_form_history", 2)
    hist_prefix = cs.history_file_prefix(nl)
    aux = [n for n in range(1, 25) if wp.g(nl, "time_control", "io_form_auxhist%d" % n, 0)]
    ri = cs.wrf_interval_minutes(nl, "restart_interval")
    out = []
    for i, d in enumerate(info["domains"]):
        off = ((d["start"] - info["start"]).total_seconds() / 60.0
               if d["start"] is not None and info["start"] is not None else 0.0)
        dend = run_min
        if d["end"] is not None and info["start"] is not None:
            dend = min(run_min, (d["end"] - info["start"]).total_seconds() / 60.0)
        iv = cs.wrf_interval_minutes(nl, "history_interval", i)
        if io_hist and iv > 0:
            hend = cs.wrf_interval_minutes(nl, "history_end", i)
            fr = _frame_minutes(off + cs.wrf_interval_minutes(nl, "history_begin", i), iv,
                                min(dend, off + hend) if hend > 0 else dend)
            if skip_t0:
                fr = [f for f in fr if f > 1e-6]
            out.append({"stream": "history", "dom": d["id"], "prefix": hist_prefix, "frames": fr})
        if ri > 0:
            out.append({"stream": "restart", "dom": d["id"], "prefix": "restart",
                        "frames": [f for f in _frame_minutes(ri, ri, dend) if f >= off]})
        for n in aux:
            iv = cs.wrf_interval_minutes(nl, "auxhist%d_interval" % n, i)
            if iv <= 0:
                continue
            aend = cs.wrf_interval_minutes(nl, "auxhist%d_end" % n, i)
            name = (wp.g(nl, "time_control", "auxhist%d_outname" % n, None)
                    or "auxhist%d_d<domain>_<date>" % n)
            out.append({"stream": "auxhist%d" % n, "dom": d["id"],
                        "prefix": os.path.basename(str(name).strip()).split("<")[0],
                        "frames": _frame_minutes(
                            off + cs.wrf_interval_minutes(nl, "auxhist%d_begin" % n, i), iv,
                            min(dend, off + aend) if aend > 0 else dend)})
    return out


def _stream_of(name, dom, streams):
    """The stream key (stream, dom) an observed write belongs to, or ('other', dom)."""
    base = os.path.basename(name)
    for st in streams:
        if st["dom"] != dom:
            continue
        if st["stream"] == "restart":
            if name == "restart":
                return st["stream"], dom
        elif st["prefix"] and base.startswith(st["prefix"]):
            return st["stream"], dom
    return "other", dom


def remaining_writes(info: dict, observed: dict, lo_min: float, hi_min: float | None = None,
                     prediction: dict | None = None) -> tuple[float, list[dict], list[str]]:
    """Seconds of the writes of frames at sim minute T with lo_min <= T (< hi_min when given, else
    to the run end), per stream = median observed seconds per frame x frames. observed:
    {(stream, dom): [s, ...]}. Falls back to prediction['io'] (history median_s per domain,
    restart per_write_s on d01) for a stream with nothing observed, else 0 + note.
    Returns (total_s, [per-stream rows], notes)."""
    io = (prediction or {}).get("io") or {}
    pred_hist = {e.get("domain"): e.get("median_s") for e in io.get("history") or []}
    pred_rst = (io.get("restart") or {}).get("per_write_s")
    total, rows, notes = 0.0, [], []
    for st in info["streams"]:
        n = sum(1 for f in st["frames"] if f >= lo_min - 1e-6
                and (hi_min is None or f < hi_min - 1e-6))
        samples = observed.get((st["stream"], st["dom"])) or []
        per, src = None, "observed"
        if samples:
            per = statistics.median(samples)
        elif st["stream"] == "history" and pred_hist.get(st["dom"]) is not None:
            per, src = float(pred_hist[st["dom"]]), "prediction"
        elif st["stream"] == "restart" and st["dom"] == 1 and pred_rst is not None:
            per, src = float(pred_rst), "prediction"  # per_write_s already sums the domains
        elif st["stream"] == "restart" and st["dom"] > 1 and pred_rst is not None:
            per, src = 0.0, "prediction (in d01's per_write_s)"
        else:
            per, src = 0.0, "none"
            if n:
                notes.append("%s d%02d: no write observed yet%s, its %d remaining frame(s) "
                             "counted as 0 s" % (st["stream"], st["dom"],
                                                 "" if prediction is None else
                                                 " and none in the prediction's io block", n))
        rows.append({"stream": st["stream"], "dom": st["dom"], "per_frame_s": per,
                     "source": src, "n_observed": len(samples), "remaining_frames": n,
                     "seconds": per * n})
        total += per * n
    return total, rows, notes


# Mean over whole radiation periods, no outlier removal: dropping write-absorbing steps lost
# the writes (marshall -53.5 % at 25 % replay, 9f7544e).
LEAD_FACTOR = 2.0     # leading steps costlier than this x the median are init, not compute
LEAD_MAX_FRAC = 0.25  # ... but at most this fraction of the logged steps


def drop_lead_in(series: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], int]:
    """series: [(step_index, sec), ...] in log order. Drops the log's first step ALWAYS (it
    carries the first-call setup of every scheme, not the steady compute), then every following
    leading step costlier than LEAD_FACTOR x the median of the rest, at most LEAD_MAX_FRAC of the
    steps. On scc truth this drops only the first step on every case, isc21 included: its slow
    start is a multi-step cold-start regime, not an init spike, and is deliberately not chased
    with heuristics. Returns (kept_series, n_dropped)."""
    if len(series) < 2:
        return list(series), 0
    med = statistics.median(sec for _, sec in series[1:])
    cap = max(1, int(len(series) * LEAD_MAX_FRAC))
    n = 1
    while n < cap and series[n][1] > LEAD_FACTOR * med:
        n += 1
    return list(series[n:]), n


def _as_series(elapsed):
    return [e if isinstance(e, tuple) else (k + 1, e) for k, e in enumerate(elapsed)]


def rate_window(elapsed: list, stepra: int | None, periods: list[int]
                ) -> tuple[float | None, int, float | None]:
    """Rate over the most recent W of the (lead-in-dropped, write-free) completed d01 compute
    parts. elapsed: [sec, ...] or [(step_index, sec), ...]. Returns (rate, W, median).
    W = max(2*stepra, 50, 25 % of completed), rounded up to a multiple of P = LCM(periods)
    (radiation periods) when that fits in the completed steps, else to a multiple of stepra that
    fits; capped at completed. rate is the MEAN (radiation spikes kept at their duty cycle; no
    outlier removal), median feeds HANG."""
    series = _as_series(elapsed)
    n = len(series)
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
    win = [sec for _, sec in series[-w:]]
    return sum(win) / w, w, statistics.median(win)


# --------------------------------------------------------------------------
# --prediction: wtune predict-rebuild's OUT/pred_rebuild.json as the cost-curve shape
# --------------------------------------------------------------------------

def load_prediction(path: str, rundir: str | None = None) -> dict:
    """pred_rebuild.json -> {anchors_at: [(datetime, s_normal)], r_night, r_day, steps_n, total_s,
    compute_total_s, init_s, sample_points, sample_source, path}. sample_points come from
    RUNDIR/wrfinput_d01 when present (read-only; wtune_predict.read_wrfinput_sample_points), for
    the same day/night radiation weighting combine() used; None -> the (r_night+r_day)/2 mean,
    exactly as predicted_compute_total does without them."""
    with open(path) as f:
        rec = json.load(f)
    res = rec.get("result", rec)
    anchors = []
    for p in res.get("points") or []:
        at = p.get("at")
        if at is None or p.get("s_normal") is None:
            continue
        at = at if isinstance(at, dt.datetime) else dt.datetime.fromisoformat(str(at))
        anchors.append((at, float(p["s_normal"])))
    if not anchors:
        raise ValueError("%s: result.points carries no (at, s_normal) anchor" % path)
    sp, src = (None, "none")
    if rundir:
        try:
            sp, src = wp.read_wrfinput_sample_points(rundir)
        except Exception as e:  # noqa: BLE001 -- sunlit weighting is optional here
            sp, src = None, "error: %s" % e
    return {"anchors_at": anchors, "r_night": res.get("r_night"), "r_day": res.get("r_day"),
            "steps_n": res.get("steps_n"), "total_s": res.get("total_s"),
            "compute_total_s": res.get("compute_total_s"), "init_s": res.get("init_s"),
            "io": rec.get("io"), "sample_points": sp, "sample_source": src,
            "path": os.path.abspath(path), "_cum": {}}


def pred_cum(pred: dict, info: dict, m: int) -> float:
    """Predicted compute seconds of d01 steps 1..m -- tools/coldstart_sample.
    predicted_compute_total over the first m steps (the exact arithmetic wtune_probe_plan.combine
    uses: piecewise-linear s_normal anchors + WRF-schedule radiation steps x r(t); both are
    prefix-stable in m, so cum(b) - cum(a) is the prediction for steps a+1..b). Memoised per m."""
    m = int(m)
    if m <= 0:
        return 0.0
    if m not in pred["_cum"]:
        import coldstart_sample as coldstart
        r_night = pred["r_night"] if pred["r_night"] is not None else 1.0
        r_day = pred["r_day"] if pred["r_day"] is not None else r_night
        anchors = [((at - info["start"]).total_seconds(), s) for at, s in pred["anchors_at"]]
        pred["_cum"][m] = coldstart.predicted_compute_total(
            m, info["dt"], info["stepra"] or 10 ** 9, info["start"], anchors,
            pred["sample_points"], r_night, r_day)[0]
    return pred["_cum"][m]


def pred_scale(pred: dict, info: dict, kept: list[tuple[int, float]]) -> float | None:
    """k = observed / predicted compute over the steps already run: `kept` is the lead-in-dropped
    [(step_index, compute seconds), ...] (writes already subtracted -- the prediction is
    compute-only). None when unusable."""
    if not kept:
        return None
    a, b = kept[0][0], kept[-1][0]
    den = pred_cum(pred, info, b) - pred_cum(pred, info, a - 1)
    return sum(sec for _, sec in kept) / den if den > 0 else None


def _tail(text, n=TAIL_LINES):
    return text.splitlines()[-n:]


def _file_age_origin(st):
    """First-creation time when the platform reports it (st_birthtime), else ctime."""
    return getattr(st, "st_birthtime", None) or st.st_ctime


_WRFOUT_D01_RE = re.compile(r"^wrfout_d01_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)")


def _is_empty_dir(path: str) -> bool:
    """A pre-created output dir WRF has not written yet (`wtune run`)."""
    try:
        return os.path.isdir(path) and not os.listdir(path)
    except OSError:
        return False


def latest_wrfout(rundir: str) -> tuple[dt.datetime | None, float | None]:
    """(latest wrfout_d01_<valid time> datetime, newest wrfout_d01 mtime) in rundir, (None, None)
    without one. On NFS the compute node's rsl.error.0000 can reach the login node minutes late
    (client write-back) while new wrfout files show up promptly: a frame's valid time is a floor
    on d01's sim time."""
    t_max = m_max = None
    try:
        names = os.listdir(rundir)
    except OSError:
        return None, None
    for name in names:
        m = _WRFOUT_D01_RE.match(name)
        if not m or _is_empty_dir(os.path.join(rundir, name)):
            continue
        try:
            t = dt.datetime.strptime(m.group(1), WRF_FMT)
            mt = os.stat(os.path.join(rundir, name)).st_mtime
        except (ValueError, OSError):
            continue
        t_max = t if t_max is None or t > t_max else t_max
        m_max = mt if m_max is None or mt > m_max else m_max
    return t_max, m_max


def _load_run_info(rundir: str, snap: dict, warn: Callable[[str, str], None]) -> dict | None:
    """run_info of RUNDIR/namelist.input into snap (NO_NAMELIST when unusable) + NEST_END_EARLY."""
    info = None
    try:
        info = run_info(wp.parse_namelist(os.path.join(rundir, "namelist.input")))
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
    return info


def _cut_for_replay(snap: dict, steps: list, trailing: list, info: dict | None,
                    fraction: float) -> tuple[list, list, dt.datetime | None]:
    """Keep the first `fraction` of the d01 steps -> (steps, rest_steps, replay_cut); records
    the actual remaining seconds (+ end-of-run writes on a complete log) in snap."""
    k = min(len(steps), max(1, int(round(fraction * len(steps))))) if steps else 0
    rest_steps = steps[k:]
    steps = steps[:k]
    replay_cut = steps[-1]["t"] if steps else None
    all_last = (steps + rest_steps)[-1]["t"] if steps + rest_steps else None
    snap["replay_fraction"] = fraction
    snap["actual_partial"] = bool(info and info["end"] and all_last is not None
                                  and all_last < info["end"])
    snap["actual_remaining_s"] = (
        sum(st["sec"] for st in rest_steps if not math.isnan(st["sec"]))
        + (0.0 if snap["actual_partial"] else sum(w for _, _, w in trailing)))
    return steps, rest_steps, replay_cut


def _set_status(snap: dict, found: list, chosen: dict, d01: list, lagging: bool, replay: bool,
                warn: Callable[[str, str], None]) -> tuple[bool, list[str], list[str]]:
    """status from the 0000 file tails (FAILED warning) -> (success, fatal_lines, problem_lines)."""
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
                          else "running" if d01 or lagging else "init")
    else:
        snap["status"] = "replay"
    return success, fatal, problem


def _check_hang_init(chosen: dict, problem: list[str], success: bool, replay: bool, now: float,
                     init_limit: float, warn: Callable[[str, str], None]) -> None:
    """HANG_INIT: no d01 Timing line yet (callers skip it once a wrfout frame or a final status
    shows the model is past init)."""
    hint = (" hint: %r (missing file, e.g. an iofields_filename target)" % problem[-1]
            if problem else "")
    if replay:
        if not success:
            warn("HANG_INIT", "no d01 Timing line in the log" + hint)
        return
    age = now - _file_age_origin(chosen["stat"])
    if age > init_limit:
        warn("HANG_INIT", "no d01 Timing line after %s (limit %s)%s"
             % (_hms(age), _hms(init_limit), hint))


def _step_index(info: dict | None, j: int, t: dt.datetime | None) -> int:
    """d01 step number of the step ending at t (j + 1 without a usable namelist or time)."""
    return (int(round((t - info["start"]).total_seconds() / info["dt"]))
            if info and t is not None else j + 1)


def _compute_series(info: dict | None, steps: list) -> list[tuple[int, float]]:
    """[(step_index, step seconds - its absorbed writes)], NaN/non-positive parts left out."""
    out = []
    for j, st in enumerate(steps):
        c = st["sec"] - sum(w for _, _, w in st["writes"])
        if not math.isnan(c) and c > 0:
            out.append((_step_index(info, j, st["t"]), c))
    return out


def _observed_writes(info: dict | None, steps: list, trailing: list,
                     replay: bool) -> dict[tuple, list[float]]:
    """{(stream, dom): [write s, ...]} from the absorbed writes (+ trailing ones when live)."""
    observed: dict[tuple, list[float]] = {}
    writes = [w for st in steps for w in st["writes"]] + ([] if replay else list(trailing))
    for name, dom, w in writes:
        key = _stream_of(name, dom, info["streams"]) if info else ("other", dom)
        observed.setdefault(key, []).append(w)
    return observed


def _estimate_remaining(snap: dict, info: dict | None, rate: float | None, completed: int,
                        lagging: bool, last_t: dt.datetime | None, kept: list, observed: dict,
                        prediction: dict | None, warn: Callable[[str, str], None]) -> tuple:
    """Remaining compute (rate-only and prediction-scaled) + writes -> (c_rate, c_pred, k, rem_w);
    fraction/remaining_steps/writes rows go into snap."""
    c_rate = c_pred = k = rem_w = None
    if info and rate is not None:
        rem_steps = max(0, info["total_steps"] - completed)
        c_rate = rem_steps * rate
        snap["fraction"] = min(1.0, completed / info["total_steps"]) if info["total_steps"] else None
        snap["remaining_steps"] = rem_steps
        if prediction is not None:
            if prediction.get("steps_n") not in (None, info["total_steps"]):
                warn("PRED_MISMATCH", "prediction steps_n %s != namelist d01 steps %d (the "
                     "namelist's run is used)" % (prediction["steps_n"], info["total_steps"]))
            k = pred_scale(prediction, info, kept)
            if k is not None:
                c_pred = k * (pred_cum(prediction, info, info["total_steps"])
                              - pred_cum(prediction, info, completed))
        rem_w, rows, notes = remaining_writes(info, observed, completed * info["dt"] / 60.0,
                                              prediction=prediction)
        snap.update(writes=rows, write_notes=notes)
    elif info and lagging and last_t is not None and info["total_steps"]:  # floor, no rsl rate
        snap["fraction"] = min(1.0, completed / info["total_steps"])
        snap["remaining_steps"] = max(0, info["total_steps"] - completed)
    return c_rate, c_pred, k, rem_w


def _replay_errors(snap: dict, info: dict | None, rate: float | None, steps: list,
                   rest_steps: list, completed: int, observed: dict, prediction: dict | None,
                   k: float | None) -> None:
    """Replay estimate vs the actual remaining over exactly what the log covers: compute of the
    later logged d01 steps and the frames they absorb (+ end-of-run frames on a complete log)."""
    act = snap["actual_remaining_s"]
    est_rate = est_pred = None
    if info and rate is not None and rest_steps:
        last_i = _step_index(info, len(steps) + len(rest_steps) - 1, rest_steps[-1]["t"])
        est_w, _, _ = remaining_writes(
            info, observed, completed * info["dt"] / 60.0,
            None if not snap["actual_partial"] else last_i * info["dt"] / 60.0,
            prediction=prediction)
        est_rate = len(rest_steps) * rate + est_w
        if k is not None:
            est_pred = k * (pred_cum(prediction, info, last_i)
                            - pred_cum(prediction, info, completed)) + est_w

    def err(e):
        return 100.0 * (e - act) / act if e is not None and act > 0 else None
    est = est_pred if est_pred is not None else est_rate
    snap.update(est_for_actual_rate_s=est_rate, est_for_actual_pred_s=est_pred,
                est_for_actual_s=est, replay_error_pct=err(est),
                replay_error_pct_rate=err(est_rate), replay_error_pct_pred=err(est_pred))
    snap["error_pct"] = snap["replay_error_pct"]  # pre-e5aebb5-follow-up name, kept as alias


def _check_hang(snap: dict, chosen: dict, wrf_mtime: float | None, med: float | None,
                last_t: dt.datetime | None, now: float, hang_factor: float, hang_min: float,
                init_limit: float, warn: Callable[[str, str], None]) -> None:
    """HANG: no rsl or wrfout_d01 output for max(hang_min, hang_factor x median step), or
    max(hang_min, init_limit) with frames but no Timing line yet (the rsl can lag on NFS)."""
    limit = (max(hang_min, hang_factor * med) if med is not None
             else max(hang_min, init_limit))
    age = now - max(chosen["stat"].st_mtime, wrf_mtime or 0.0)
    snap["log_age_s"] = age
    if age > limit:
        warn("HANG", "no rsl/wrfout output for %s (limit %s = %s); last d01 time %s"
             % (_hms(age), _hms(limit),
                "max(%gs, %g x median step %.2fs)" % (hang_min, hang_factor, med)
                if med is not None else "max(%gs, init limit %gs)" % (hang_min, init_limit),
                _fmt_t(last_t)))


def _domain_progress(snap: dict, info: dict, timing: list, replay: bool,
                     replay_cut: dt.datetime | None, rsl_last_t: dt.datetime | None,
                     warn: Callable[[str, str], None]) -> list[dict]:
    """Per-domain last time; NEST_STOPPED when a nest trails d01 by > 2 parent steps while its
    own end is still ahead (rsl vs rsl, never the wrfout floor)."""
    doms = []
    for d in info["domains"]:
        ts = [t for dd, t, _ in timing if dd == d["id"] and t is not None
              and (replay_cut is None or t <= replay_cut)]
        if replay and replay_cut is None:
            ts = []
        dlast = max(ts) if ts else None
        doms.append({"id": d["id"], "last_time": _fmt_t(dlast), "n_lines": len(ts),
                     "start": _fmt_t(d["start"]), "end": _fmt_t(d["end"] or info["end"])})
        if d["id"] == 1 or rsl_last_t is None or snap["status"] == "failed":
            continue
        ref = dlast
        if ref is None:
            if d["start"] is None or rsl_last_t <= d["start"]:
                continue
            ref = d["start"]
        dend = d["end"] or info["end"]
        if (rsl_last_t - ref).total_seconds() > 2 * d["parent_dt"] and (dend is None or dend > ref):
            warn("NEST_STOPPED", "d%02d last time %s is behind d01 %s while its end is %s"
                 % (d["id"], _fmt_t(ref), _fmt_t(rsl_last_t), _fmt_t(dend)))
    return doms


def _check_cfl(snap: dict, chosen: dict, prev_cfl: int | None,
               warn: Callable[[str, str], None]) -> None:
    """CFL (informational): cfl lines in the chosen file grew since the previous poll."""
    cfl = sum(1 for ln in chosen["text"].splitlines() if "cfl" in ln.lower())
    snap["cfl_count"] = cfl
    if cfl > (0 if prev_cfl is None else prev_cfl):
        warn("CFL", "%d cfl line(s) in %s%s" % (
            cfl, chosen["name"], "" if prev_cfl is None else " (+%d since last poll)"
            % (cfl - prev_cfl)))


def snapshot(rundir: str, now: float | None = None, replay_fraction: float | None = None,
             hang_factor: float = 10.0, hang_min: float = 300.0, init_limit: float = 600.0,
             prev_cfl: int | None = None, prediction: dict | None = None) -> dict:
    """One read-only observation of RUNDIR -> JSON-serializable dict; pure in the files and `now`
    (epoch s, default time.time()). replay_fraction: replay mode, no live checks. prev_cfl: the
    previous poll's cfl count (None -> warn on any). prediction: load_prediction()'s dict, the
    primary estimate when usable."""
    now = time.time() if now is None else now
    replay = replay_fraction is not None
    warnings: list[dict] = []

    def warn(code, text):
        warnings.append({"code": code, "text": text})

    snap = {"rundir": os.path.abspath(rundir), "now": now, "mode": "replay" if replay else "live",
            "status": None, "warnings": warnings}
    info = _load_run_info(rundir, snap, warn)

    chosen, found = read_rsl(rundir)
    if chosen is None:
        snap["status"] = "no_rsl"
        return snap
    snap["rsl_file"] = chosen["name"]
    snap["rsl_d01_lines"] = {r["name"]: r["n_d01"] for r in found}
    timing = chosen["timing"]
    steps, trailing = parse_d01_steps(chosen["text"])
    rest_steps, replay_cut = [], None
    if replay:
        steps, rest_steps, replay_cut = _cut_for_replay(snap, steps, trailing, info,
                                                        replay_fraction)
    d01 = [(st["t"], st["sec"]) for st in steps]
    rsl_last_t = d01[-1][0] if d01 else None
    # wrfout floor (live only): a d01 frame later than the rsl's last d01 line means the rsl lags
    wrf_t, wrf_mtime = (None, None) if replay else latest_wrfout(rundir)
    floor = rsl_last_t or (info["start"] if info else None)   # the t=0 frame is no progress
    lagging = wrf_t is not None and (floor is None or wrf_t > floor)

    success, fatal, problem = _set_status(snap, found, chosen, d01, lagging, replay, warn)
    if not d01 and not lagging and not (not replay and (success or fatal)):
        _check_hang_init(chosen, problem, success, replay, now, init_limit, warn)

    elapsed = [st["sec"] for st in steps if not math.isnan(st["sec"])]
    observed = _observed_writes(info, steps, trailing, replay)
    kept, n_lead = drop_lead_in(_compute_series(info, steps))
    rate, w, med = rate_window(kept, info["stepra"] if info else None,
                               info["periods"] if info else [])
    last_t = rsl_last_t
    if lagging and snap["status"] == "running":
        last_t = wrf_t
        snap.update(wrfout_last_time=_fmt_t(wrf_t), rsl_d01_last_time=_fmt_t(rsl_last_t))
        warn("RSL_LAGGING", "rsl behind wrfout (latest wrfout %s); rate/ETA from rsl may be stale"
             % _fmt_t(wrf_t))
    completed = len(d01)
    if info and last_t is not None:
        completed = int(round((last_t - info["start"]).total_seconds() / info["dt"]))
    snap.update(n_d01_steps=len(d01), completed_steps=completed, d01_last_time=_fmt_t(last_t),
                sum_d01_elapsed_s=sum(elapsed),
                writes_so_far_s=sum(sum(v) for v in observed.values()),
                n_write_absorbing_steps=sum(1 for st in steps if st["writes"]),
                rate_s_per_step=rate, rate_window=w, median_step_s=med, rate_lead_dropped=n_lead)
    if "other" in {key[0] for key in observed}:
        snap["unclassified_writes_s"] = sum(sum(v) for key, v in observed.items()
                                            if key[0] == "other")
    c_rate, c_pred, k, rem_w = _estimate_remaining(snap, info, rate, completed, lagging, last_t,
                                                   kept, observed, prediction, warn)
    snap.update(remaining_compute_rate_s=c_rate, remaining_compute_pred_s=c_pred,
                remaining_writes_s=rem_w, pred_scale_k=k,
                remaining_rate_s=None if c_rate is None else c_rate + rem_w,
                remaining_pred_s=None if c_pred is None else c_pred + rem_w,
                estimate="prediction" if c_pred is not None else "rate")
    remaining = snap["remaining_pred_s"] if c_pred is not None else snap["remaining_rate_s"]
    snap["remaining_compute_s"] = c_pred if c_pred is not None else c_rate
    if snap["status"] == "done":
        remaining = 0.0
        snap["fraction"] = 1.0
    snap["remaining_s"] = remaining
    if not replay and remaining is not None:
        snap["eta"] = dt.datetime.fromtimestamp(now + remaining).strftime("%Y-%m-%d %H:%M:%S")
    if replay:
        _replay_errors(snap, info, rate, steps, rest_steps, completed, observed, prediction, k)
    if not replay and snap["status"] == "running" and (med is not None or lagging):
        _check_hang(snap, chosen, wrf_mtime, med, last_t, now, hang_factor, hang_min, init_limit,
                    warn)
    snap["domains"] = (_domain_progress(snap, info, timing, replay, replay_cut, rsl_last_t, warn)
                       if info else [])
    if not replay:   # live only: the whole file would leak post-cut lines into a replay
        _check_cfl(snap, chosen, prev_cfl, warn)
    return snap


# --------------------------------------------------------------------------
# `wtune run --out` root: OUT/run_report.json + OUT/seg_<k>/ (wtune_run.run_controller)
# --------------------------------------------------------------------------
_SEG_DIR_RE = re.compile(r"^seg_(\d+)(_rescue)?(?:_r(\d+))?$")


def _parse_wrf_t(s):
    try:
        return dt.datetime.strptime(s, WRF_FMT) if s else None
    except (TypeError, ValueError):
        return None


def follow_run_root(root: str) -> dict | None:
    """None for a run dir (namelist.input present) or a dir without seg_<k>/; else the segment to
    watch: {dir, segment: k, n_segments: K or None, run_start, run_end, run_status} -- the
    highest-numbered seg_<k> (its latest attempt: seg_<k>_rescue from the CFL rescue,
    seg_<k>[_rescue]_r<n> from an auto-resume), K and the whole
    window (first segment start .. last segment end) from run_report.json when it lists the
    segments. Re-read on every poll (the controller rewrites the report after each attempt)."""
    if os.path.exists(os.path.join(root, "namelist.input")):
        return None
    try:
        names = os.listdir(root)
    except OSError:
        return None
    segs = sorted((int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0), n) for n, m in
                  ((n, _SEG_DIR_RE.match(n)) for n in names)
                  if m and os.path.isdir(os.path.join(root, n)))
    if not segs:
        return None
    k, _, _, name = segs[-1]
    try:
        with open(os.path.join(root, "run_report.json")) as f:
            rep = json.load(f)
    except (OSError, ValueError):
        rep = None
    rep = rep if isinstance(rep, dict) else {}
    rsegs = rep.get("segments") if isinstance(rep.get("segments"), list) else []
    return {"dir": os.path.join(root, name), "segment": k, "n_segments": len(rsegs) or None,
            "run_start": _parse_wrf_t(rsegs[0].get("start")) if rsegs else None,
            "run_end": _parse_wrf_t(rsegs[-1].get("end")) if rsegs else None,
            "run_status": rep.get("status")}


def annotate_run_root(snap: dict, root: str, fol: dict) -> dict:
    """Turn a segment snapshot into a whole-run one: fraction = sim time since the first
    segment's start / whole window (run_report.json; else the segment's own namelist window);
    the segment's own fraction moves to segment_fraction. Remaining/ETA stay per segment."""
    start = fol["run_start"] or _parse_wrf_t(snap.get("run_start"))
    end = fol["run_end"] or _parse_wrf_t(snap.get("run_end"))
    sim = _parse_wrf_t(snap.get("d01_last_time"))
    if snap["status"] == "done":
        sim = _parse_wrf_t(snap.get("run_end")) or sim
    if sim is None:                      # init / no rsl: the segment's start is reached
        sim = _parse_wrf_t(snap.get("run_start"))
    frac = None
    if start is not None and end is not None and sim is not None and end > start:
        frac = min(1.0, max(0.0, (sim - start).total_seconds() / (end - start).total_seconds()))
    snap.update(run_root=os.path.abspath(root), segment=fol["segment"],
                n_segments=fol["n_segments"], run_status=fol["run_status"],
                whole_run_start=_fmt_t(start), whole_run_end=_fmt_t(end),
                segment_fraction=snap.get("fraction"), fraction=frac)
    if fol["run_status"] == "failed" and snap["status"] != "failed":
        snap["status"] = "failed"        # the controller stopped the run: not RUNNING, rc 3
        snap["warnings"].append({"code": "FAILED",
                                 "text": "run_report.json: the run failed (see its error)"})
    return snap


def root_finished(snap: dict, fol: dict) -> bool:
    """Stop polling a run root: a failed segment, the controller's report final, or the last
    segment done (with no report to say how many there are: any segment done)."""
    if snap["status"] == "failed" or fol["run_status"] in ("completed", "failed"):
        return True
    if snap["status"] != "done":
        return False
    return fol["n_segments"] is None or fol["segment"] >= fol["n_segments"] - 1


def exit_code(snap: dict) -> int:
    codes = {w["code"] for w in snap["warnings"]}
    if "FAILED" in codes:
        return 3
    if codes & set(BLOCKING):
        return 2
    return 0


def format_line(snap: dict) -> str:
    """One compact text line per snapshot."""
    frac = snap.get("fraction")
    parts = [snap["status"].upper()]
    if snap["mode"] == "replay":
        parts = ["replay F=%g" % snap["replay_fraction"]]
    if "segment" in snap:
        parts.insert(0, "seg %d/%s" % (snap["segment"], snap["n_segments"] or "?"))
    parts.append("%5.1f%%" % (100 * frac) if frac is not None else "  ?%")
    if snap.get("d01_last_time"):
        parts.append("sim %s (%s/%s)" % (snap["d01_last_time"], snap.get("completed_steps"),
                                          snap.get("total_steps", "?")))
    if snap.get("rate_s_per_step") is not None:
        parts.append("elapsed %s rate %.3fs/step[w=%d]" % (
            _hms(snap["sum_d01_elapsed_s"]), snap["rate_s_per_step"], snap["rate_window"]))
    if snap.get("remaining_s") is not None:
        parts.append("remaining %s" % _hms(snap["remaining_s"]))
        if snap["status"] != "done" and snap.get("remaining_compute_s") is not None:
            parts.append("(compute %s%s + writes %s%s)" % (
                _hms(snap["remaining_compute_s"]),
                " pred x%.3f" % snap["pred_scale_k"] if snap["estimate"] == "prediction" else "",
                _hms(snap["remaining_writes_s"]),
                "; rate-only %s" % _hms(snap["remaining_rate_s"])
                if snap["estimate"] == "prediction" else ""))
    if snap.get("eta"):
        parts.append("ETA %s" % snap["eta"])
    if snap["mode"] == "replay":
        if snap.get("actual_partial"):
            parts.append("est-to-log-end %s" % _hms(snap.get("est_for_actual_s")))
        parts.append("actual %s%s" % (_hms(snap["actual_remaining_s"]),
                                      " (partial log)" if snap.get("actual_partial") else ""))
        if snap.get("replay_error_pct") is not None:
            parts.append("err %+.2f%%" % snap["replay_error_pct"])
        if snap.get("replay_error_pct_pred") is not None and snap.get("replay_error_pct_rate") is not None:
            parts.append("(rate-only %+.2f%%)" % snap["replay_error_pct_rate"])
    line = " ".join(parts)
    if snap["warnings"]:
        line += " | " + "; ".join("%s: %s" % (w["code"], w["text"]) for w in snap["warnings"])
    return line


def _emit(snap, as_json):
    print(json.dumps(snap, sort_keys=True) if as_json else format_line(snap), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune watch", description=__doc__.split("\n\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--interval", type=float, default=60.0, help="poll period, s (default 60)")
    ap.add_argument("--once", action="store_true", help="one snapshot, then exit")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per snapshot (a JSON list for a --replay-fraction list)")
    ap.add_argument("--replay-fraction", default=None,
                    help="F or F1,F2,... in (0,1]: replay a finished log at that fraction of its "
                         "d01 steps and compare the estimate with the actual remaining (implies --once)")
    ap.add_argument("--init-limit", type=float, default=600.0,
                    help="HANG_INIT after this many s with no d01 Timing line (default 600)")
    ap.add_argument("--hang-factor", type=float, default=10.0)
    ap.add_argument("--hang-min", type=float, default=300.0)
    ap.add_argument("--prediction", default=None, metavar="PRED_JSON",
                    help="wtune predict-rebuild's OUT/pred_rebuild.json: remaining = its compute "
                         "curve over the unrun steps x (observed / predicted so far); primary ETA")
    a = ap.parse_args(argv)
    kw = dict(hang_factor=a.hang_factor, hang_min=a.hang_min, init_limit=a.init_limit)
    if a.prediction:
        try:
            fol = follow_run_root(a.rundir)
            kw["prediction"] = load_prediction(a.prediction, fol["dir"] if fol else a.rundir)
        except (OSError, ValueError) as e:
            ap.error("--prediction: %s" % e)

    if a.replay_fraction is not None:
        try:
            fracs = [float(x) for x in a.replay_fraction.split(",") if x.strip()]
        except ValueError:
            ap.error("--replay-fraction: expected F or a comma list of floats")
        if not fracs or any(not 0 < f <= 1 for f in fracs):
            ap.error("--replay-fraction values must be in (0, 1]")
        snaps = [snapshot(a.rundir, replay_fraction=f, **kw) for f in fracs]
        if a.json and len(snaps) > 1:
            print(json.dumps(snaps, sort_keys=True), flush=True)  # one JSON list for a comma list
        else:
            for snap in snaps:
                _emit(snap, a.json)
        return max(exit_code(snap) for snap in snaps)

    prev_cfl, prev_dir, snap = None, None, None
    try:
        while True:
            fol = follow_run_root(a.rundir)      # a `wtune run --out` root: its newest seg_<k>
            target = fol["dir"] if fol else a.rundir
            if target != prev_dir:
                prev_cfl, prev_dir = None, target
            snap = snapshot(target, prev_cfl=prev_cfl, **kw)
            prev_cfl = snap.get("cfl_count", prev_cfl)
            if fol:
                annotate_run_root(snap, a.rundir, fol)
            _emit(snap, a.json)
            if a.once or (root_finished(snap, fol) if fol
                          else snap["status"] in ("done", "failed")):
                return exit_code(snap)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return exit_code(snap) if snap else 0


if __name__ == "__main__":
    sys.exit(main())
