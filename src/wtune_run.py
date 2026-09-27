#!/usr/bin/env python3
"""wtune_run.py -- segmented WRF executor with restart hand-offs (`wtune run`).

    wtune run CASE_DIR --nodes N --nx X --ny Y [--numtiles T] [--omp O]
        [--wall-limit HH:MM:SS | --segment-minutes M] [--stop-at YYYY-MM-DD_HH:MM:SS]
        [--prediction pred_rebuild.json] [--tune-report tune_report.json]
        [--restart-io-form 14|2] [--cfl-rescue] --out OUT [--dry-run]

CASE_DIR is never modified and --out may not be inside it. The run [start, end] is the case's own
(run_* wins for d01, wtune_predict.get_d01_steps), cut at --stop-at when that is earlier (it must
fall on a whole d01 time step). The run is split into segments; segment k runs in OUT/seg_<k> as
its own Slurm job and hands off to k+1 through the restart files it writes at its end.

Segment planning (plan_run; boundaries are d01 step indices)
    ONE restart quantum L for the whole run (segments never set override_restart_timers, so the
    restart file carries every alarm's phase AND segment 0's restart interval to later segments --
    scc R1: with the override, d01's 60-min history alarm re-anchored at a 00:30 boundary and the
    01:00 frame was never written). L is whole minutes on the d01 step grid (quantum = lcm(dt,
    60 s); a nest step divides its parent's, so every domain's step divides it); every segment is
    one L block [jL, (j+1)L] from the run start, the last possibly shorter, and restart_interval =
    L in every segment (a segment ending on the grid, the last included, writes a restart).
    --segment-minutes M   L = M.
    --wall-limit W        L = the largest quantum multiple for which EVERY L-grid segment's
                          predicted wall <= 0.8 W (choose_restart_quantum), where predicted wall =
                          compute over the interval + init + restart read (k > 0) + restart write
                          (end on the grid) + history frames written inside the interval.
    neither               one segment, no restart (restart_interval = RESTART_OFF_MIN).
    Compute over [a, b] = C(b) - C(a), C a prefix sum of tools/coldstart_sample.
    predicted_compute_total's per-step terms (the same arithmetic wtune_probe_plan.combine uses
    for the whole run; C(N) is asserted equal to its total). Its
    inputs come from --prediction's pred_rebuild.json (result.points at/s_normal, result.r_night/
    r_day, result.init_s, io.restart.per_write_s, io.history[].median_s) or from each
    --tune-report ranking row (points at/s_normal/r/n_rad; r_night/r_day pooled exactly like
    combine via wtune_probe_plan._resolve_r_rad_from_points). With neither: the uncalibrated
    cell-step model (wtune_tune._fallback_s_per_step) x FALLBACK_RADIATION_FACTOR. Without a
    measured `io` block, restart and history I/O scale from measured scc anchors by grid points x
    levels (fallback_overheads; conus25 restart per io_form, dorian2019 history). After each
    segment its observed overhead (job elapsed - sum d01) -- and, without a prediction, its mean
    d01 step -- replace the predictions in later segments' --time; never the L grid.

Segment namelist (render_segment_namelist; tools/coldstart_sample set_list & co. do the editing)
    start_* / end_* of EVERY active domain = the segment's start / end (yagi truth 1038: a nest
    whose end_* was not updated stopped early); run_days/hours/minutes = 0, run_seconds = segment
    length; nproc_x/nproc_y/numtiles of the segment's config; restart_interval = L minutes in every
    segment (restart_interval_d/_h/_s zeroed, _m set equal when present); io_form_restart per
    --restart-io-form; k > 0: restart=.true.; override_restart_timers is never set (a case value
    .true. is set .false.: the tuner's restart-start probes want the override, a chain must not
    have it). write_hist_at_0h_rst is never set (its default .false. avoids a duplicate history
    frame at the boundary); every other
    history setting is untouched. adios2_numaggregators, when the case sets it, follows the
    segment's rank count (configs._apply_adios2: it must equal the rank count).

Segment run dir (setup_segment_dir): symlinks to every regular file of CASE_DIR except namelist.*,
rsl.*, wrfout*, wrfrst*, wrf.out (plus ADIOS2 *.bp input directories), WRF_TREE/run/* not already
present, the previous segment's wrfrst_d<NN>_<start> for k > 0, a fresh namelist.input, and for
io_form 14 an adios2.xml selecting the BP4 engine for every restart file the segment reads or
writes (campaign/strict_campaign.sbatch write_adios2_restart_xml, same content; a case adios2.xml
is copied with these entries appended instead of being linked). History output stays in each
OUT/seg_<k>.

Segment job (render_segment_sbatch): exclusive nodes, --exclude=head[,WTUNE_SBATCH_EXCLUDE],
--time = 1.5 x predicted wall + 10 min rounded up to 5 min, capped at --wall-limit; the shared
signal-trap + node_hygiene reap+census preamble (coldstart_sample.render_signal_trap_and_hygiene_
preamble, dispatched by campaign/sites/scc.sh site_hygiene); OMP env as strict_campaign.sbatch
configure_omp_env does on scc; mpirun ppr/PE binding as sites/scc.sh site_launch; last line
'SEGMENT rc=<rc> wall=<s> ok=<SUCCESS COMPLETE WRF count>'.

Controller (run_controller): submit, wait (sacct), inspect with wtune_watch.snapshot. A segment is
done when watch says 'done' and (not last) every domain's wrfrst at its end exists. Otherwise, with
--cfl-rescue and a CFL line or NaN in the rsl, the SAME segment is re-run once per run from the
same restart with time_step = the largest integer in [0.5 dt, 0.75 dt] dividing the segment length
and L
(and the d01 history interval and radt when possible); the run is then PHYSICS-GRAY and later segments go
back to the case's time_step. Anything else stops with an error -- never a silent retry.
OUT/run_report.json is rewritten after every segment attempt.
"""
import argparse
import datetime as dt
import difflib
import fractions
import glob
import json
import math
import os
import re
import shlex
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (os.path.join(_HERE, "tools"), os.path.join(_HERE, "predictor"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coldstart_sample as coldstart   # noqa: E402  tools/: namelist editing, predicted compute,
                                        # the shared sbatch signal-trap/hygiene preamble
import scc_sweep as sweep              # noqa: E402  tools/: sacct_states/submit_job/exclude_arg
import wtune_predict as wp             # noqa: E402  namelist parsing, calibration, sample points
import wtune_probe_plan as pp          # noqa: E402  _resolve_r_rad_from_points (combine's r rule)
import wtune_watch                     # noqa: E402  snapshot(rundir): done/failed/CFL/NaN

try:
    import wtune_tune                  # _fallback_s_per_step, _parse_slurm_elapsed
except ImportError:                    # pragma: no cover
    wtune_tune = None

WRF_FMT = "%Y-%m-%d_%H:%M:%S"
CORES_PER_NODE = 72                    # scc GH200 (campaign/sites/scc.sh)
WALL_FRACTION = 0.8                    # --wall-limit: predicted segment wall <= 0.8 L
TIME_FACTOR = 1.5                      # --time = ceil5min(1.5 x predicted wall + 10 min)
TIME_PAD_S = 600.0
TIME_ROUND_S = 300
FALLBACK_INIT_S = 120.0                # init when unpredicted (wtune_tune rebuild_segment_timeout_min)
FALLBACK_RADIATION_FACTOR = 1.5        # cell-step model x this (same padding as wtune_tune)
FALLBACK_RESTART_IO_S = 300.0          # one restart write/read, scc marshall measured ~300 s
FALLBACK_HISTORY_FRAME_S = 30.0        # per domain frame when no I/O probe measured it
RESTART_OFF_MIN = 1000000              # last segment: no restart write (configs._apply_adios2 value)
CFL_RESCUE_FACTOR = 0.75
CFL_RESCUE_MIN_FACTOR = 0.5            # never search below half the case's dt
UNKNOWN_POLLS_MAX = 10                 # sacct not listing a fresh job id yet: poll this many times
_CASE_LINK_SKIP = ("namelist.", "rsl.", "wrfout", "wrfrst")
_WRF_RUN_SKIP = ("namelist.input", "namelist.output")
_WRF_RUN_SKIP_PREFIX = ("wrfout", "wrfinput", "wrfbdy", "wrfrst", "rsl.")
_TIME_PARTS = ("year", "month", "day", "hour", "minute", "second")


class RunError(Exception):
    pass


def _fmt(t):
    return t.strftime(WRF_FMT)


def _hms(s):
    s = int(round(s))
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def parse_hms(text):
    """'HH:MM:SS' (hours may exceed 24) or 'D-HH:MM:SS' -> seconds."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) != 3:
        raise ValueError("expected HH:MM:SS, got %r" % text)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _frac(x):
    return fractions.Fraction(x).limit_denominator(10000)


def _frac_lcm(a, b):
    a, b = _frac(a), _frac(b)
    den = a.denominator * b.denominator // math.gcd(a.denominator, b.denominator)
    na, nb = int(a * den), int(b * den)
    return fractions.Fraction(na * nb // math.gcd(na, nb), den)


# ---------------------------------------------------------------------------
# case geometry
# ---------------------------------------------------------------------------
def case_geometry(nl):
    """Run geometry from a parsed namelist: {start, end, dt, steps, stepra, max_dom, radt_s,
    quantum (d01 steps per whole-minute boundary), hist (per domain {iv_s, begin_s} or None),
    points (per domain e_we x e_sn x e_vert, for the I/O fallbacks)}."""
    steps, run_s, dt_s, note = wp.get_d01_steps(nl)
    if note:
        raise RunError("case: %s" % note)
    start = wp.get_time_control_datetime(nl, "start")
    if start is None:
        raise RunError("case namelist has no start_year/month/day")
    max_dom = wp.get_max_dom(nl)
    radt = wp.g_dom(nl, "physics", "radt", 0)
    stepra = wp.steps_per_radiation_call(float(radt), dt_s) if radt else None
    quantum = _frac_lcm(dt_s, 60) / _frac(dt_s)
    if quantum.denominator != 1:
        raise RunError("cannot place whole-minute boundaries on d01 steps of %gs" % dt_s)
    io_hist = wp.g(nl, "time_control", "io_form_history", 2)
    hist = []
    for d in range(max_dom):
        iv = coldstart.wrf_interval_minutes(nl, "history_interval", d) * 60.0
        begin = coldstart.wrf_interval_minutes(nl, "history_begin", d) * 60.0
        hist.append({"iv_s": iv, "begin_s": begin} if io_hist and iv > 0 else None)
    points = []
    for d in range(max_dom):
        e_we, e_sn, e_vert = wp.get_domain_geom(nl, d)
        points.append(e_we * e_sn * e_vert)
    return {"start": start, "end": start + dt.timedelta(seconds=run_s), "dt": dt_s,
            "steps": steps, "stepra": stepra, "max_dom": max_dom,
            "radt_s": float(radt) * 60.0 if radt else None, "quantum": int(quantum),
            "hist": hist, "points": points}


def domain_start(nl, d):
    """Domain d's (0-based) own start_* datetime (broadcast-last-value), None when absent."""
    vals = [wp.g_dom(nl, "time_control", "start_" + p, d, 0 if i > 2 else None)
            for i, p in enumerate(_TIME_PARTS)]
    if any(v is None for v in vals[:3]):
        return None
    return dt.datetime(int(vals[0]), int(vals[1]), int(vals[2])) + dt.timedelta(
        hours=int(vals[3] or 0), minutes=int(vals[4] or 0), seconds=int(vals[5] or 0))


def step_time(geo, n):
    return geo["start"] + dt.timedelta(seconds=float(_frac(geo["dt"]) * n))


def resolve_end_step(geo, stop_at=None):
    """d01 step index of the run end: the case's own end, or --stop-at when earlier. stop_at must
    fall on a whole d01 step after the run start."""
    if stop_at is None:
        return geo["steps"]
    t = dt.datetime.strptime(stop_at, WRF_FMT) if isinstance(stop_at, str) else stop_at
    off_ms = int(round((t - geo["start"]).total_seconds() * 1000))
    n = fractions.Fraction(off_ms, 1000) / _frac(geo["dt"])
    if n.denominator != 1:
        raise RunError("--stop-at %s is not on a whole time step (dt=%gs from %s)"
                       % (_fmt(t), geo["dt"], _fmt(geo["start"])))
    n = int(n)
    if n <= 0:
        raise RunError("--stop-at %s is not after the run start %s" % (_fmt(t), _fmt(geo["start"])))
    return min(n, geo["steps"])


# ---------------------------------------------------------------------------
# segment planning (pure)
# ---------------------------------------------------------------------------
# WRF restart semantics this plan relies on (scc R1, 2026-09-27, dorian2019 30+30 min chain):
# with override_restart_timers=.true. a resumed run RE-ANCHORS every alarm at its own start
# (d01 history 60 min resumed at 00:30 -> next frame 01:30; the 01:00 frame was lost), so segments
# never set it. With the default .false. the restart file carries every alarm's phase (history,
# auxhist, restart) across the boundary -- and also the restart interval written by segment 0,
# which then governs every later segment. Hence ONE restart quantum L for the whole run:
# restart_interval = L in every segment, every boundary a multiple of L from the run start.
def plan_segments_fixed(n_end, steps_per_segment, n0=0):
    """[(a, b), ...] d01 step intervals of `steps_per_segment` from n0 to n_end, last shorter."""
    if steps_per_segment <= 0:
        raise RunError("segment length must be > 0 steps")
    segs, a = [], n0
    while a < n_end:
        b = min(a + steps_per_segment, n_end)
        segs.append((a, b))
        a = b
    return segs


def segment_steps_for_minutes(geo, minutes):
    """d01 steps in `minutes`; RunError unless that is a whole number of d01 steps."""
    n = _frac(minutes * 60) / _frac(geo["dt"])
    if n.denominator != 1 or n <= 0:
        raise RunError("--segment-minutes %s is not a whole number of %gs time steps"
                       % (minutes, geo["dt"]))
    return int(n)


def choose_restart_quantum(n_end, quantum, wall_fn, budget_s):
    """--wall-limit: the largest L (d01 steps, a multiple of `quantum`, i.e. whole minutes on the
    step grid) such that EVERY segment of the L grid [jL, min((j+1)L, n_end)] has
    wall_fn(a, b, L) <= budget_s -- computed once for the whole run (the most expensive segment
    decides). None when the whole run fits in one segment; RunError when not even one quantum
    fits. Exhaustive over m (sum over m of n_end/(m q) segments ~ N log N wall evaluations), so no
    monotonicity assumption is needed."""
    if wall_fn(0, n_end, None) <= budget_s:
        return None
    for m in range((n_end - 1) // quantum, 0, -1):
        L = m * quantum
        if all(wall_fn(a, b, L) <= budget_s for a, b in plan_segments_fixed(n_end, L)):
            return L
    raise RunError("even %d-step segments exceed the %.0f s budget (%.0f s predicted for the "
                   "first)" % (quantum, budget_s, wall_fn(0, min(quantum, n_end), quantum)))


# ---------------------------------------------------------------------------
# wall model
# ---------------------------------------------------------------------------
# Fallback I/O anchors, used only when the prediction has no measured `io` block (scc R2: the
# constant 300 s restart / 30 s frame fallbacks predicted 578/548 s for segments observed at
# 198/168 s). Both scale linearly with grid points x levels summed over the domains written.
#  - restart: conus25 (1 domain 1500x1200x50 = 90e6 points, netCDF restart 11.5 GB) on scc wrote
#    in 324 s / read in 120 s with io_form 2; ADIOS2 BP4 (io_form 14) wrote 1.05 s / read 2.3 s.
#    Cross-check: dorian2019 netCDF restart at 6 h (f1 atlas profile, 26.7e6 points) wrote in
#    75.5 + 34.3 = 109.8 s vs 96 s from this scaling.
#  - history: dorian2019 scc truth (job 1019, 6 h; d01 664x700x40 every 60 min = 7 frames, d02
#    451x451x40 every 15 min = 25 frames) summed 300.5 s of history writes (rank 0 'Timing for
#    Writing') -> 300.5 / (7 x 18.592e6 + 25 x 8.136e6) = 9.0e-7 s per point per frame. The f1
#    atlas dorian netCDF frames agree (d01 ~15.7 s = 8.4e-7, d02 ~7.4 s = 9.1e-7). One netCDF
#    anchor is used for every io_form_history (conservative for ADIOS2 history).
CONUS25_RESTART_POINTS = 1500 * 1200 * 50
RESTART_IO_ANCHOR_S = {2: (324.0, 120.0), 14: (1.05, 2.3)}   # io_form: (write, read) at conus25
DORIAN_HISTORY_WRITE_S = 300.5
DORIAN_HISTORY_POINT_FRAMES = 7 * 664 * 700 * 40 + 25 * 451 * 451 * 40
HISTORY_S_PER_POINT_FRAME = DORIAN_HISTORY_WRITE_S / DORIAN_HISTORY_POINT_FRAMES


def fallback_restart_io_s(geo, io_form):
    """(write_s, read_s) of one restart (all domains) scaled from the conus25 anchor."""
    w, r = RESTART_IO_ANCHOR_S[int(io_form)]
    f = sum(geo["points"]) / float(CONUS25_RESTART_POINTS)
    return w * f, r * f


def fallback_history_frame_s(geo):
    """Per-domain seconds for one history frame, scaled from the dorian2019 anchor."""
    return [p * HISTORY_S_PER_POINT_FRAME for p in geo["points"]]


def fallback_overheads(geo, io_form_restart):
    w, r = fallback_restart_io_s(geo, io_form_restart)
    return overheads(init_s=FALLBACK_INIT_S, restart_read_s=r, restart_write_s=w,
                     history_frame_s=fallback_history_frame_s(geo),
                     source="fallback (conus25/dorian2019 I/O anchors)")


def compute_fn_from_points(geo, anchors_at, r_night, r_day, sample_points):
    """compute(a, b) seconds over d01 steps a+1..b = C(b) - C(a), C a prefix-sum array over the
    whole case run. Per-step terms: s(t_i) from coldstart_sample.predicted_compute_total's own
    s_values plus its radiation term (r(t_i)-1)*s(t_i) on predict.classify_steps' steps -- the
    same loop, restated only to keep the per-step values; C(N) is asserted equal to
    predicted_compute_total's total, so any drift in coldstart fails loudly here."""
    if not anchors_at:
        raise RunError("prediction has no measured points")
    start, dt_s, n = geo["start"], geo["dt"], geo["steps"]
    stepra = geo["stepra"] or 1
    anchors = [((at - start).total_seconds(), float(s)) for at, s in anchors_at]
    total, s_values = coldstart.predicted_compute_total(n, dt_s, stepra, start, anchors,
                                                        sample_points, r_night, r_day)
    flags = wp.classify_steps(n, stepra)
    cum = [0.0]
    for i in range(n):
        x = s_values[i]
        if flags[i]:
            t_i = start + dt.timedelta(seconds=(i + 1) * dt_s)
            frac = wp.sunlit_fraction(sample_points, t_i) if sample_points else None
            r_t = (r_night + (r_day - r_night) * frac) if frac is not None \
                else (r_night + r_day) / 2.0
            x += (r_t - 1.0) * s_values[i]
        cum.append(cum[-1] + x)
    if abs(cum[-1] - total) > 1e-6 * max(1.0, abs(total)):
        raise RunError("internal: prefix compute %.6f != predicted_compute_total %.6f"
                       % (cum[-1], total))
    return lambda a, b: cum[b] - cum[a]


def constant_compute_fn(s_per_step):
    return lambda a, b: (b - a) * s_per_step


def history_frames(geo, d, a, b):
    """History frames domain d writes in a segment over steps (a, b] (+ the t=0 frame for the
    first segment; a resumed segment does not write its start frame: write_hist_at_0h_rst is
    left at its .false. default, and the restart file carries the alarm phase)."""
    h = geo["hist"][d] if d < len(geo["hist"]) else None
    if not h:
        return 0
    dt_s = geo["dt"]

    def upto(x):              # frames with begin + j*iv <= x
        return 0 if x < h["begin_s"] else int(math.floor((x - h["begin_s"]) / h["iv_s"] + 1e-9)) + 1
    n = upto(b * dt_s) - upto(a * dt_s)
    if a == 0 and h["begin_s"] == 0:
        n += 1
    return n


def overheads(init_s=FALLBACK_INIT_S, restart_read_s=FALLBACK_RESTART_IO_S,
              restart_write_s=FALLBACK_RESTART_IO_S, history_frame_s=None, source="fallback"):
    return {"init_s": init_s, "restart_read_s": restart_read_s,
            "restart_write_s": restart_write_s, "history_frame_s": history_frame_s,
            "source": source}


def writes_restart(b, L):
    """Segment ending at step b writes a restart iff b is on the L grid (restart alarm)."""
    return bool(L) and b > 0 and b % L == 0


def segment_wall(geo, compute, over, a, b, L):
    """Predicted wall of one segment (a, b]: compute + init + restart read (a > 0) + restart
    write (b on the L grid) + history frames written inside it."""
    w = compute(a, b) + over["init_s"]
    if a > 0:
        w += over["restart_read_s"]
    if writes_restart(b, L):
        w += over["restart_write_s"]
    return w + history_wall(geo, over, a, b)


def history_wall(geo, over, a, b):
    """Seconds of history writes in segment (a, b]: frames x per-domain frame seconds."""
    per = over.get("history_frame_s")
    w = 0.0
    for d in range(geo["max_dom"]):
        fs = FALLBACK_HISTORY_FRAME_S if per is None else (per[d] if d < len(per) else per[-1])
        w += history_frames(geo, d, a, b) * (fs or 0.0)
    return w


def time_limit_s(pred_wall_s, wall_limit_s=None):
    raw = TIME_FACTOR * pred_wall_s + TIME_PAD_S
    t = int(math.ceil(round(raw, 6) / TIME_ROUND_S)) * TIME_ROUND_S
    return min(t, wall_limit_s) if wall_limit_s else t


def fallback_s_per_step(nl, ranks, omp):
    """Conservative constant s/step: wtune_tune._fallback_s_per_step (cell-step model) x
    FALLBACK_RADIATION_FACTOR; scc_sweep.CONSERVATIVE_S_PER_STEP when that model fails."""
    try:
        return wtune_tune._fallback_s_per_step(nl, ranks, omp) * FALLBACK_RADIATION_FACTOR
    except Exception:
        return sweep.CONSERVATIVE_S_PER_STEP


def _parse_at(s):
    return dt.datetime.fromisoformat(s) if "T" in s or " " in s else dt.datetime.strptime(s, WRF_FMT)


def model_from_prediction(pred, geo, sample_points, fallback):
    """(compute_fn, overheads) from a pred_rebuild.json record; its measured `io` block wins,
    `fallback` (fallback_overheads) fills whatever it does not measure."""
    res = pred["result"]
    pts = [(_parse_at(p["at"]), p["s_normal"]) for p in res["points"]]
    comp = compute_fn_from_points(geo, pts, res["r_night"], res["r_day"], sample_points)
    io = pred.get("io") or {}
    rst = (io.get("restart") or {}).get("per_write_s")
    hist = list(fallback["history_frame_s"])
    for d, e in enumerate(io.get("history") or []):
        if e.get("median_s") is not None and d < len(hist):
            hist[d] = e["median_s"]
    measured = bool(rst) or bool(io.get("history"))
    return comp, overheads(
        init_s=res.get("init_s") or FALLBACK_INIT_S,
        restart_read_s=rst if rst else fallback["restart_read_s"],
        restart_write_s=rst if rst else fallback["restart_write_s"], history_frame_s=hist,
        source="prediction" + (" (measured io)" if measured else " + " + fallback["source"]))


def compute_fn_from_tune_row(row, geo, sample_points, calibration):
    """compute_fn for one predicted-mode tune_report ranking row (points at/s_normal/r/n_rad),
    r_night/r_day pooled exactly as wtune_probe_plan.combine pools them; None without points."""
    pts = row.get("points") or []
    if not pts:
        return None
    pooled = []
    for p in pts:
        if p.get("n_rad") and p.get("r") is not None:
            frac = (wp.sunlit_fraction(sample_points, _parse_at(p["at"]))
                    if sample_points else 0.5)
            pooled.append((frac if frac is not None else 0.5, p["r"]))
    r_night, r_day, _src = pp._resolve_r_rad_from_points(pooled, calibration)
    return compute_fn_from_points(geo, [(_parse_at(p["at"]), p["s_normal"]) for p in pts],
                                  r_night, r_day, sample_points)


def tune_candidates(report, geo, sample_points, calibration, nodes):
    """[{config, compute}] from a predicted-mode tune_report.json's ranking rows."""
    if report.get("nodes") is not None and int(report["nodes"]) != int(nodes):
        raise RunError("--tune-report was measured at %s node(s), --nodes is %d"
                       % (report["nodes"], nodes))
    out = []
    for row in report.get("ranking") or []:
        comp = compute_fn_from_tune_row(row, geo, sample_points, calibration)
        if comp is None:
            continue
        cfg = make_config(row.get("nproc_x"), row.get("nproc_y"), row.get("numtiles") or 1,
                          row.get("omp") or 1, nodes, cid=row.get("id"),
                          dist=row.get("dist") or "block")
        out.append({"config": cfg, "compute": comp})
    if not out:
        raise RunError("--tune-report has no ranking row with measured points")
    return out


def make_config(nx, ny, numtiles, omp, nodes, cid="cli", dist="block"):
    nx, ny, numtiles, omp = int(nx), int(ny), int(numtiles), int(omp)
    ranks = nx * ny
    if ranks * omp > nodes * CORES_PER_NODE:
        raise RunError("config %s: %d ranks x %d threads exceeds %d node(s) x %d cores"
                       % (cid, ranks, omp, nodes, CORES_PER_NODE))
    return {"id": cid, "nx": nx, "ny": ny, "ranks": ranks, "numtiles": numtiles, "omp": omp,
            "nodes": nodes, "dist": dist}


def assign_segments(geo, bounds, candidates, over, L, wall_limit_s=None):
    """Segment dicts for step intervals `bounds`: the candidate with the lowest predicted compute
    over each interval (ties -> first listed), its predicted wall and --time."""
    segs = []
    for i, (a, b) in enumerate(bounds):
        best = min(candidates, key=lambda c: c["compute"](a, b))
        wall = segment_wall(geo, best["compute"], over, a, b, L)
        segs.append({"index": i, "start_step": a, "end_step": b,
                     "start": _fmt(step_time(geo, a)), "end": _fmt(step_time(geo, b)),
                     "config": dict(best["config"]), "writes_restart": writes_restart(b, L),
                     "predicted_compute_s": best["compute"](a, b), "predicted_wall_s": wall,
                     "time_limit": _hms(time_limit_s(wall, wall_limit_s))})
    return segs


def plan_run(geo, n_end, candidates, over, segment_minutes=None, wall_limit_s=None):
    """(L, segments): L = the restart quantum in d01 steps (None: one segment, no restart), each
    segment one L block [jL, (j+1)L], the last possibly shorter (see module docstring)."""
    if segment_minutes:
        L = segment_steps_for_minutes(geo, segment_minutes)
        if (L * geo["dt"]) % 60:
            raise RunError("--segment-minutes must give whole minutes")
        if L >= n_end:
            L = None
    elif wall_limit_s:
        def wall_fn(a, b, L):
            return min(segment_wall(geo, c["compute"], over, a, b, L) for c in candidates)
        L = choose_restart_quantum(n_end, geo["quantum"], wall_fn, WALL_FRACTION * wall_limit_s)
    else:
        L = None
    bounds = plan_segments_fixed(n_end, L) if L else [(0, n_end)]
    return L, assign_segments(geo, bounds, candidates, over, L, wall_limit_s)


def restart_interval_minutes(geo, L):
    """restart_interval (minutes) for quantum L; RESTART_OFF_MIN for a single segment."""
    if not L:
        return RESTART_OFF_MIN
    m = _frac(geo["dt"]) * L / 60
    if m.denominator != 1:
        raise RunError("restart quantum %d steps is not whole minutes" % L)
    return int(m)


# ---------------------------------------------------------------------------
# segment namelist (pure)
# ---------------------------------------------------------------------------
def _time_values(t):
    return (t.year, "%02d" % t.month, "%02d" % t.day, "%02d" % t.hour, "%02d" % t.minute,
            "%02d" % t.second)


def _set_one(txt, key, value, group="time_control"):
    """One-value key: campaign/configs._set_scalar in place when the key is assigned exactly once
    (keeps the line layout -- coldstart set_list joins a comma-less 'key = v' line with the next
    line), else coldstart set_list (absent -> inserted into &group; packed/odd lines)."""
    raw = len(re.findall(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), txt))
    if raw == 1 and coldstart.count_key(txt, key) == 1:
        return sweep.campaign_configs._set_scalar(txt, key, value)
    return coldstart.set_list(txt, key, value, 1, new_key_group=group)


def render_segment_namelist(text, seg_start, seg_end, k, config, restart_interval_min,
                            restart_io_form=14, time_step=None):
    """The case's namelist.input text -> segment k's (see module docstring). seg_start/seg_end
    are datetimes; time_step (int s) overrides the d01 time step (CFL rescue)."""
    cs = coldstart
    ndom = int(cs.get_first(text, "max_dom", "1"))
    length_s = (seg_end - seg_start).total_seconds()
    if length_s <= 0 or length_s != int(length_s):
        raise RunError("segment length %r s is not a positive whole number of seconds" % length_s)
    length_s = int(length_s)
    txt = text
    for prefix, t in (("start_", seg_start), ("end_", seg_end)):
        for part, v in zip(_TIME_PARTS, _time_values(t)):
            txt = cs.set_list(txt, prefix + part, v, ndom)
    for key, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0),
                   ("run_seconds", length_s)):
        txt = _set_one(txt, key, v)
    for key, v in (("nproc_x", config["nx"]), ("nproc_y", config["ny"]),
                   ("numtiles", config["numtiles"])):
        txt = _set_one(txt, key, v, "domains")
    rst_min = int(restart_interval_min)
    txt = _set_one(txt, "restart_interval", rst_min)
    for suffix, v in (("_d", 0), ("_h", 0), ("_s", 0), ("_m", rst_min)):
        if cs.count_key(text, "restart_interval" + suffix):
            txt = _set_one(txt, "restart_interval" + suffix, v)
    txt = _set_one(txt, "io_form_restart", int(restart_io_form))
    if k > 0:
        txt = _set_one(txt, "restart", ".true.")
    elif cs.count_key(text, "restart"):
        txt = _set_one(txt, "restart", ".false.")
    if cs.count_key(text, "override_restart_timers"):       # never .true.: see planning notes
        txt = _set_one(txt, "override_restart_timers", ".false.")
    if cs.count_key(text, "adios2_numaggregators"):
        txt = _set_one(txt, "adios2_numaggregators", config["ranks"], "namelist_adios2")
    required = (["%s_%s" % (p, q) for p in ("start", "end") for q in _TIME_PARTS]
                + ["run_days", "run_hours", "run_minutes", "run_seconds", "nproc_x", "nproc_y",
                   "numtiles", "restart_interval", "io_form_restart"]
                + (["restart"] if k > 0 else []))
    if time_step is not None:
        txt = _set_one(txt, "time_step", int(time_step), "domains")
        if cs.count_key(text, "time_step_fract_num"):
            txt = _set_one(txt, "time_step_fract_num", 0, "domains")
        required.append("time_step")
    cs.assert_no_key_drift(text, txt, "segment %d namelist" % k, required)
    return txt


def namelist_diff_summary(old, new):
    """Changed lines only ('- old' / '+ new')."""
    return [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))]


# ---------------------------------------------------------------------------
# segment run directory
# ---------------------------------------------------------------------------
def render_adios2_restart_xml(restart_times, max_dom, base=None):
    """BP4 engine per restart file name (strict_campaign.sbatch write_adios2_restart_xml). `base`
    (the case's own adios2.xml text, if any) keeps its entries: ours go before </adios-config>."""
    ios = ""
    for t in restart_times:
        for d in range(1, max_dom + 1):
            ios += '  <io name="wrfrst_d%02d_%s">\n    <engine type="BP4"/>\n  </io>\n' % (d, t)
    if base and "</adios-config>" in base:
        return base.replace("</adios-config>", ios + "</adios-config>", 1)
    return '<?xml version="1.0"?>\n<adios-config>\n' + ios + "</adios-config>\n"


def restart_files(rundir, t, max_dom):
    """{domain: [paths]} of wrfrst_d<NN>_<t>* in rundir."""
    return {d: sorted(glob.glob(os.path.join(rundir, "wrfrst_d%02d_%s*" % (d, glob.escape(t)))))
            for d in range(1, max_dom + 1)}


def _case_link_ok(name, path, skip_adios2_xml):
    if name == "wrf.out" or name.startswith(_CASE_LINK_SKIP):
        return False
    if skip_adios2_xml and name == "adios2.xml":
        return False
    if os.path.isfile(path):
        return True
    return os.path.isdir(path) and name.endswith(".bp")   # ADIOS2 wrfinput/wrfbdy inputs


def setup_segment_dir(case_dir, seg_dir, namelist_text, restart_links=(), adios2_xml=None,
                      wrf_tree=None):
    """Create seg_dir (must not exist) and populate it; writes only inside seg_dir."""
    case_dir = os.path.abspath(case_dir)
    seg_dir = os.path.abspath(seg_dir)
    if os.path.commonpath([seg_dir, case_dir]) == case_dir:
        raise RunError("segment dir %s is inside CASE_DIR" % seg_dir)
    os.makedirs(seg_dir)
    for name in sorted(os.listdir(case_dir)):
        src = os.path.join(case_dir, name)
        if _case_link_ok(name, src, adios2_xml is not None):
            os.symlink(src, os.path.join(seg_dir, name))
    for src in restart_links:
        os.symlink(os.path.abspath(src), os.path.join(seg_dir, os.path.basename(src)))
    run_dir = os.path.join(wrf_tree, "run") if wrf_tree else None
    if run_dir and os.path.isdir(run_dir):
        for name in sorted(os.listdir(run_dir)):
            if name in _WRF_RUN_SKIP or name.startswith(_WRF_RUN_SKIP_PREFIX):
                continue
            dst = os.path.join(seg_dir, name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(run_dir, name), dst)
    nml = os.path.join(seg_dir, "namelist.input")
    if os.path.lexists(nml):
        os.remove(nml)            # never write through a link
    with open(nml, "w") as f:
        f.write(namelist_text)
    if adios2_xml is not None:
        with open(os.path.join(seg_dir, "adios2.xml"), "w") as f:
            f.write(adios2_xml)


# ---------------------------------------------------------------------------
# segment sbatch
# ---------------------------------------------------------------------------
def render_segment_sbatch(seg_dir, k, config, time_limit, job_name,
                          env_source="$HOME/wtune-scc/env.sh", campaign_dir=_CAMPAIGN_DIR):
    nodes, ranks, omp = config["nodes"], config["ranks"], config["omp"]
    ppn = -(-ranks // nodes)
    seg_dir = os.path.abspath(seg_dir)
    lines = ["#!/usr/bin/env bash",
             "#SBATCH --job-name=%s" % job_name,
             "#SBATCH -p compute -N %d --exclusive --ntasks-per-node=%d --time=%s %s"
             % (nodes, CORES_PER_NODE, time_limit, sweep.exclude_arg()),
             "#SBATCH --signal=B:USR1@90",
             "#SBATCH --output=%s/segment-%%j.out" % seg_dir,
             "set -uo pipefail",
             "SEG=%s" % shlex.quote(seg_dir),
             "SITE=scc",
             "CAMPAIGN=${CAMPAIGN:-%s}" % shlex.quote(os.path.abspath(campaign_dir)),
             "source \"$CAMPAIGN/sites/$SITE.sh\"   # site_hygiene / W",
             "source \"%s\"" % env_source,
             "W=${WRF_TREE:-$W}",
             "ulimit -s unlimited"]
    lines += coldstart.render_signal_trap_and_hygiene_preamble(
        nodes, extra_lines_after_alloc_nodes=["STATUS=\"$SEG/segment_status.txt\"",
                                              ": > \"$STATUS\""],
        record_label="SEGMENT")
    lines += ["cd \"$SEG\"",
              "for f in \"$W\"/run/*; do",
              "  b=$(basename \"$f\")",
              "  case \"$b\" in namelist.input|namelist.output|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;",
              "    *) [ -e \"$b\" ] || ln -sf \"$f\" \"$b\" ;; esac",
              "done",
              "export OMP_NUM_THREADS=%d" % omp,
              "unset KMP_BLOCKTIME"]
    lines.append("export OMP_PLACES=cores OMP_PROC_BIND=close" if omp > 1
                 else "unset OMP_PLACES OMP_PROC_BIND")
    launch = "mpirun -np %d --map-by ppr:%d:node:PE=%d --bind-to core" % (ranks, ppn, omp)
    if config.get("dist") == "cyclic":
        launch += " --rank-by node"
    lines += ["CUR_WINDOW=%d" % k,
              "T0=$(date +%s)",
              "%s ./wrf.exe &" % launch,
              "WRF_PID=$!",
              "wait \"$WRF_PID\"; rc=$?",
              "WRF_PID=\"\"; CUR_WINDOW=\"\"",
              "wall=$(( $(date +%s) - T0 ))",
              "ok=$(grep -c 'SUCCESS COMPLETE WRF' rsl.error.0000 2>/dev/null); ok=${ok:-0}",
              "echo \"SEGMENT rc=$rc wall=$wall ok=$ok\" | tee -a \"$STATUS\""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------
def rescue_time_step(dt_s, seg_len_s, multiples=()):
    """Largest integer step in [ceil(0.5 dt), floor(0.75 dt)] dividing seg_len_s and every
    nonzero entry of `multiples` (history interval, radt in s); retried with seg_len_s alone
    (a tiny step that divides everything is not a rescue); None if none."""
    top = int(math.floor(CFL_RESCUE_FACTOR * dt_s))
    bottom = max(1, int(math.ceil(CFL_RESCUE_MIN_FACTOR * dt_s)))
    for extra in (tuple(int(m) for m in multiples if m and float(m) == int(m)), ()):
        for c in range(top, bottom - 1, -1):
            if seg_len_s % c == 0 and all(m % c == 0 for m in extra):
                return c
    return None


def rsl_has_cfl(rundir):
    """True when any rsl.error.* / rsl.out.* line mentions cfl (every rank, not just 0000)."""
    for path in glob.glob(os.path.join(rundir, "rsl.*")):
        try:
            with open(path, "r", errors="replace") as f:
                for ln in f:
                    if "cfl" in ln.lower():
                        return True
        except OSError:
            continue
    return False


def wait_for_job(job_id, sacct_runner, sacct_cmd="sacct", poll_s=60.0, sleep=time.sleep,
                 log=print):
    """Poll sacct until job_id leaves PENDING/RUNNING -> (state, elapsed_s or None)."""
    unknown = 0
    last = None
    while True:
        state, elapsed = sweep.sacct_states([job_id], sacct_runner=sacct_runner,
                                            sacct_cmd=sacct_cmd).get(str(job_id), ("UNKNOWN", ""))
        if state != last:
            log("  job %s: %s" % (job_id, state))
            last = state
        if state == "UNKNOWN":
            unknown += 1
            if unknown > UNKNOWN_POLLS_MAX:
                return state, None
        elif state not in sweep.IN_FLIGHT_STATES:
            return state, (wtune_tune._parse_slurm_elapsed(elapsed) if wtune_tune else None)
        sleep(poll_s)


def _write_report(path, report):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1, default=str)
    os.replace(tmp, path)


def run_controller(ctx):
    """Execute ctx['segments'] in order (see module docstring). Returns 0 on success, 1 on a
    stop. ctx keys: case_dir, out, text, geo, n_end, segments, restart_min (restart_interval of
    every segment), over, restart_io_form, cfl_rescue, replan (bool: no prediction), wall_limit_s,
    wrf_tree, env_source, campaign_dir, sbatch_runner, sacct_runner, sacct_cmd, snapshot_fn,
    poll_s, sleep, log, report."""
    geo, log, report = ctx["geo"], ctx["log"], ctx["report"]
    report_path = os.path.join(ctx["out"], "run_report.json")
    segs = ctx["segments"]
    report["segments"] = segs
    rescue_used = False
    prev_dir = None
    k = 0
    while k < len(segs):
        seg = segs[k]
        is_last = k == len(segs) - 1
        seg.setdefault("attempts", [])
        time_step = None
        attempt = 0
        while True:
            name = "seg_%d" % k + ("_rescue" if attempt else "")
            seg_dir = os.path.join(ctx["out"], name)
            t_a = dt.datetime.strptime(seg["start"], WRF_FMT)
            t_b = dt.datetime.strptime(seg["end"], WRF_FMT)
            txt = render_segment_namelist(ctx["text"], t_a, t_b, k, seg["config"],
                                          ctx["restart_min"], ctx["restart_io_form"],
                                          time_step=time_step)
            links = []
            if k > 0:
                found = restart_files(prev_dir, seg["start"], geo["max_dom"])
                missing = [d for d, p in found.items() if not p]
                if missing:
                    raise RunError("segment %d: no wrfrst for d%s at %s in %s"
                                   % (k, ",".join("%02d" % d for d in missing), seg["start"],
                                      prev_dir))
                links = [p for d in sorted(found) for p in found[d]]
            xml = None
            if int(ctx["restart_io_form"]) == 14:
                times = (([seg["start"]] if k > 0 else [])
                         + ([seg["end"]] if seg["writes_restart"] else []))
                xml = render_adios2_restart_xml(times, geo["max_dom"], ctx.get("case_adios2_xml"))
            setup_segment_dir(ctx["case_dir"], seg_dir, txt, links, xml, ctx.get("wrf_tree"))
            script = render_segment_sbatch(
                seg_dir, k, seg["config"], seg["time_limit"],
                "wtrun-%s-s%d" % (os.path.basename(ctx["case_dir"]), k),
                env_source=ctx["env_source"], campaign_dir=ctx["campaign_dir"])
            spath = os.path.join(seg_dir, "segment.sbatch")
            with open(spath, "w") as f:
                f.write(script)
            job_id = sweep.submit_job("sbatch %s" % shlex.quote(spath),
                                      sbatch_runner=ctx["sbatch_runner"])
            log("segment %d/%d %s -> %s  %s  --time=%s  job %s%s"
                % (k, len(segs) - 1, seg["start"], seg["end"], seg["config"]["id"],
                   seg["time_limit"], job_id, "  (CFL rescue dt=%ds)" % time_step
                   if time_step else ""))
            rec = {"dir": seg_dir, "job_id": job_id, "time_step": time_step,
                   "submitted_utc": dt.datetime.utcnow().isoformat()}
            seg["attempts"].append(rec)
            _write_report(report_path, report)
            state, elapsed = wait_for_job(job_id, ctx["sacct_runner"], ctx["sacct_cmd"],
                                          ctx["poll_s"], ctx["sleep"], log)
            snap = ctx["snapshot_fn"](seg_dir)
            codes = [w["code"] for w in snap.get("warnings", [])]
            nan = any(w["code"] == "FAILED" and "NaN" in w["text"]
                      for w in snap.get("warnings", []))
            rst_ok = is_last or all(restart_files(seg_dir, seg["end"], geo["max_dom"]).values())
            ok = snap.get("status") == "done" and rst_ok
            rec.update(state=state, elapsed_s=elapsed, watch_status=snap.get("status"),
                       warnings=codes, sum_d01_elapsed_s=snap.get("sum_d01_elapsed_s"),
                       n_d01_steps=snap.get("n_d01_steps"), restart_ok=rst_ok, ok=ok)
            if ok:
                seg["observed_wall_s"] = elapsed
                seg["run_dir"] = seg_dir
                break
            cfl = "CFL" in codes or nan or rsl_has_cfl(seg_dir)
            if ctx["cfl_rescue"] and cfl and not rescue_used:
                seg_len = int(round((t_b - t_a).total_seconds()))
                if ctx["restart_min"] != RESTART_OFF_MIN:      # keep the restart alarm whole too
                    seg_len = math.gcd(seg_len, ctx["restart_min"] * 60)
                hist = geo["hist"][0]["iv_s"] if geo["hist"][0] else None
                new_dt = rescue_time_step(geo["dt"], seg_len, (hist, geo["radt_s"]))
                if new_dt is None:
                    raise RunError("segment %d failed with CFL/NaN and no integer time step in "
                                   "[%.2f, %.2f] s divides its %d s length" % (
                                       k, CFL_RESCUE_MIN_FACTOR * geo["dt"],
                                       CFL_RESCUE_FACTOR * geo["dt"], seg_len))
                rescue_used = True
                time_step = new_dt
                attempt += 1
                report["physics_gray"] = True
                report["physics_gray_reason"] = (
                    "segment %d re-run with time_step %d s (case %g s) after CFL/NaN"
                    % (k, new_dt, geo["dt"]))
                report["decisions"].append({"segment": k, "decision": "cfl_rescue",
                                            "time_step": new_dt})
                log("PHYSICS-GRAY: " + report["physics_gray_reason"])
                continue
            reason = ("watch status %s, sacct %s, restart %s, warnings %s%s"
                      % (snap.get("status"), state, "ok" if rst_ok else "MISSING",
                         ",".join(codes) or "none", ", CFL/NaN seen" if cfl else ""))
            if cfl and rescue_used:
                reason += " (CFL rescue already used once in this run)"
            raise RunError("segment %d (%s) failed: %s" % (k, seg_dir, reason))
        prev_dir = seg["run_dir"]
        report["decisions"].append({"segment": k, "decision": "done" if is_last else "next"})
        if not is_last:
            _update_from_observed(ctx, k, snap, elapsed)
        _write_report(report_path, report)
        k += 1
    return 0


def _update_from_observed(ctx, k, snap, elapsed):
    """Segment k's observed overhead (job elapsed - sum d01: init, restart read/write, end-of-run
    writes) replaces the predicted overheads in every later segment's predicted wall and --time;
    a segment-0 observation gets the restart read added (segment 0 reads none). Without a
    prediction, the observed mean d01 step (history writes included) also replaces the compute
    model. The L grid (segment boundaries) never changes."""
    geo, over = ctx["geo"], ctx["over"]
    sum_d01, n = snap.get("sum_d01_elapsed_s"), snap.get("n_d01_steps")
    if elapsed is None or not n or sum_d01 is None:
        return
    ovh = max(0.0, elapsed - sum_d01)
    read = over["restart_read_s"] if k == 0 else 0.0
    s_step = sum_d01 / n if ctx["replan"] else None
    for seg in ctx["segments"][k + 1:]:
        a, b = seg["start_step"], seg["end_step"]
        if s_step is not None:
            seg["predicted_compute_s"] = s_step * (b - a)
            wall = seg["predicted_compute_s"] + ovh + read
        else:
            wall = seg["predicted_compute_s"] + ovh + read + history_wall(geo, over, a, b)
        seg["predicted_wall_s"] = wall
        seg["time_limit"] = _hms(time_limit_s(wall, ctx["wall_limit_s"]))
        seg["wall_source"] = "observed overhead of segment %d" % k
    ctx["report"]["decisions"].append({
        "segment": k, "decision": "observed_overhead", "overhead_s": ovh,
        "observed_s_per_step": sum_d01 / n, "compute_replaced": s_step is not None})
    ctx["log"]("segment %d observed: overhead %.0f s, %.3f s/step -> later --time updated"
               % (k, ovh, sum_d01 / n))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(prog="wtune run", description=__doc__.split("\n\n")[0])
    ap.add_argument("case_dir")
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--numtiles", type=int, default=1)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--wall-limit", default=None, help="HH:MM:SS per segment job")
    ap.add_argument("--segment-minutes", type=int, default=None)
    ap.add_argument("--stop-at", default=None, help="YYYY-MM-DD_HH:MM:SS (on a whole time step)")
    ap.add_argument("--prediction", default=None, help="pred_rebuild.json (wtune predict-rebuild)")
    ap.add_argument("--tune-report", default=None,
                    help="predicted-mode tune_report.json: per segment the lowest-predicted "
                         "candidate is run")
    ap.add_argument("--restart-io-form", type=int, choices=(14, 2), default=14)
    ap.add_argument("--cfl-rescue", action="store_true",
                    help="re-run a CFL/NaN-failed segment once with 0.75 dt (marks PHYSICS-GRAY)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wrf-tree", default=None, help="default $WRF_TREE (run/ files are linked "
                                                     "by the job too)")
    ap.add_argument("--env-source", default="$HOME/wtune-scc/env.sh")
    ap.add_argument("--campaign-dir", default=_CAMPAIGN_DIR)
    ap.add_argument("--sacct-cmd", default="sacct")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    return ap


def main(argv=None, sbatch_runner=None, sacct_runner=None, snapshot_fn=None, sleep=None,
         log=print):
    a = build_arg_parser().parse_args(argv)
    case_dir = os.path.abspath(a.case_dir)
    out = os.path.abspath(a.out)
    try:
        if os.path.commonpath([out, case_dir]) == case_dir:
            raise RunError("--out must be outside CASE_DIR (the case is never modified)")
        nml = os.path.join(case_dir, "namelist.input")
        with open(nml) as f:
            text = f.read()
        nl = wp.parse_namelist(nml)
        if wp.g(nl, "time_control", "restart", False) is True:
            raise RunError("the case itself is a restart run (restart=.true.): not supported")
        for key in ("rst_outname", "rst_inname"):
            if coldstart.count_key(text, key):
                raise RunError("%s is set: segments assume the default wrfrst_d<domain>_<date> "
                               "names" % key)
        geo = case_geometry(nl)
        for d in range(1, geo["max_dom"]):
            ds = domain_start(nl, d)
            if ds is not None and ds != geo["start"]:
                raise RunError("d%02d starts at %s, not with d01 (%s): staggered nest starts are "
                               "not supported" % (d + 1, _fmt(ds), _fmt(geo["start"])))
        n_end = resolve_end_step(geo, a.stop_at)
        wall_limit_s = parse_hms(a.wall_limit) if a.wall_limit else None
        config = make_config(a.nx, a.ny, a.numtiles, a.omp, a.nodes)

        sample_points = None
        if a.prediction or a.tune_report:
            sample_points, _src = wp.read_wrfinput_sample_points(case_dir)
        over = fallback_overheads(geo, a.restart_io_form)
        candidates, notes = [], []
        if a.prediction:
            with open(a.prediction) as f:
                pred = json.load(f)
            comp, over = model_from_prediction(pred, geo, sample_points, over)
            hw = pred.get("hardware") or {}
            if (hw.get("nx"), hw.get("ny"), hw.get("omp", 1)) != (a.nx, a.ny, a.omp) \
                    or (pred.get("nodes") not in (None, a.nodes)):
                notes.append("prediction was made for %s nodes %sx%s omp %s, not this config"
                             % (pred.get("nodes"), hw.get("nx"), hw.get("ny"), hw.get("omp")))
            candidates.append({"config": config, "compute": comp})
        if a.tune_report:
            with open(a.tune_report) as f:
                rep = json.load(f)
            cal = wp.load_calibration(str(wp.DEFAULT_CALIBRATION))
            candidates += tune_candidates(rep, geo, sample_points, cal, a.nodes)
        replan = not candidates
        if replan:
            s = fallback_s_per_step(nl, config["ranks"], config["omp"])
            candidates = [{"config": config, "compute": constant_compute_fn(s)}]
            notes.append("no prediction: %.3f s/step fallback (cell-step model x %.1f); later "
                         "segments' --time from each finished segment's observed rate"
                         % (s, FALLBACK_RADIATION_FACTOR))
        L, segs = plan_run(geo, n_end, candidates, over, a.segment_minutes, wall_limit_s)
        restart_min = restart_interval_minutes(geo, L)
        if coldstart.get_first(text, "override_restart_timers", ".false.").lower().startswith(".t"):
            notes.append("case sets override_restart_timers=.true.: set .false. in every segment "
                         "(it would re-anchor the history alarms at each segment start)")
        if segs and geo["hist"][0] is None:
            notes.append("d01 writes no history")
        if coldstart.get_first(text, "write_hist_at_0h_rst", ".false.").lower().startswith(".t"):
            notes.append("case sets write_hist_at_0h_rst=.true.: every resumed segment repeats "
                         "its start history frame")
    except (RunError, OSError, ValueError, wp.PredictError, coldstart.ColdstartError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2

    log("run %s -> %s (%d d01 steps of %gs)%s; %d segment(s), restart_interval %s, overheads: %s"
        % (_fmt(geo["start"]), _fmt(step_time(geo, n_end)), n_end, geo["dt"],
           " stop-at" if n_end < geo["steps"] else "", len(segs),
           "%d min (every segment)" % restart_min if L else "off (one segment)", over["source"]))
    for s in segs:
        c = s["config"]
        log("  seg %d  %s -> %s  %-8s %dx%d t%d omp%d  predicted %s  --time=%s"
            % (s["index"], s["start"], s["end"], c["id"], c["nx"], c["ny"], c["numtiles"],
               c["omp"], _hms(s["predicted_wall_s"]), s["time_limit"]))
    for n in notes:
        log("  note: %s" % n)
    if a.dry_run:
        for s in segs[:2]:
            t_a = dt.datetime.strptime(s["start"], WRF_FMT)
            t_b = dt.datetime.strptime(s["end"], WRF_FMT)
            new = render_segment_namelist(text, t_a, t_b, s["index"], s["config"], restart_min,
                                          a.restart_io_form)
            log("namelist diff, segment %d:" % s["index"])
            for ln in namelist_diff_summary(text, new):
                log("  " + ln)
        return 0

    os.makedirs(out, exist_ok=True)
    report = {"case_dir": case_dir, "out": out, "run_start": _fmt(geo["start"]),
              "run_end": _fmt(step_time(geo, n_end)), "stop_at": a.stop_at, "dt": geo["dt"],
              "restart_io_form": a.restart_io_form, "cfl_rescue": a.cfl_rescue,
              "prediction": a.prediction, "tune_report": a.tune_report, "overheads": over,
              "restart_quantum_steps": L, "restart_interval_min": restart_min,
              "notes": notes, "physics_gray": False, "decisions": [], "status": "running",
              "started_utc": dt.datetime.utcnow().isoformat()}
    ctx = dict(case_dir=case_dir, out=out, text=text, geo=geo, n_end=n_end, segments=segs,
               restart_min=restart_min, over=over, restart_io_form=a.restart_io_form,
               cfl_rescue=a.cfl_rescue, replan=replan, wall_limit_s=wall_limit_s, wrf_tree=a.wrf_tree or os.environ.get("WRF_TREE"),
               env_source=a.env_source, campaign_dir=a.campaign_dir,
               sbatch_runner=sbatch_runner or sweep.default_sbatch_runner,
               sacct_runner=sacct_runner or sweep.default_sacct_runner, sacct_cmd=a.sacct_cmd,
               snapshot_fn=snapshot_fn or wtune_watch.snapshot, poll_s=a.poll_interval,
               sleep=sleep or time.sleep, log=log, report=report)
    case_xml = os.path.join(case_dir, "adios2.xml")
    if os.path.isfile(case_xml):
        with open(case_xml) as f:
            ctx["case_adios2_xml"] = f.read()
    path = os.path.join(out, "run_report.json")
    try:
        rc = run_controller(ctx)
        report["status"] = "completed"
    except (RunError, RuntimeError, OSError, coldstart.ColdstartError) as e:
        report["status"] = "failed"
        report["error"] = str(e)
        _write_report(path, report)
        print("error: %s" % e, file=sys.stderr)
        return 1
    report["finished_utc"] = dt.datetime.utcnow().isoformat()
    _write_report(path, report)
    log("done: %d segment(s)%s; report %s" % (len(ctx["segments"]),
                                              " -- PHYSICS-GRAY" if report["physics_gray"] else "",
                                              path))
    return rc


if __name__ == "__main__":
    sys.exit(main())
