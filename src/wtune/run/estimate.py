"""Wall model of a segment: compute over a step interval (from measured points, or a constant
rate) plus init, restart read/write and history frames (Overheads); --time from that wall.
Without a measured `io` block the I/O terms scale from wtune.io_model's scc anchors."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Callable, Optional

import wtune.machine
from ._deps import coldstart, iom, pp, sweep, wp, tune
from .common import WRF_FMT, Record, RunError, _frac
from .geometry import Geometry

TIME_FACTOR = 1.5                      # --time = ceil5min(1.5 x predicted wall + 10 min)
TIME_PAD_S = 600.0
TIME_ROUND_S = 300
# built-in values (wtune.machine.MEASURED); the functions below read the machine's
FALLBACK_INIT_S = wtune.machine.MEASURED["init_s"]
FALLBACK_RADIATION_FACTOR = wtune.machine.MEASURED["radiation_factor"]
FALLBACK_RESTART_IO_S = wtune.machine.MEASURED["restart_io_s"]
FALLBACK_HISTORY_FRAME_S = wtune.machine.MEASURED["history_frame_s"]
# Fallback I/O anchors (wtune.io_model owns the numbers), linear in grid points x levels; the
# constant fallbacks above predicted 578 s for a segment observed at 198 s (scc R2).
#  restart: conus25 (90e6 points) on scc, io_form 2 wrote 324 s / read 120 s, BP4 1.05 / 2.3 s
#  history: dorian2019 scc job 1019, 300.5 s over 7 d01 + 25 d02 frames = 9.0e-7 s/point/frame
CONUS25_RESTART_POINTS = iom.CONUS25_RESTART_POINTS
RESTART_IO_ANCHOR_S = iom.RESTART_IO_ANCHOR_S
DORIAN_HISTORY_WRITE_S = iom.DORIAN_HISTORY_WRITE_S
DORIAN_HISTORY_POINT_FRAMES = iom.DORIAN_HISTORY_POINT_FRAMES
HISTORY_S_PER_POINT_FRAME = iom.HISTORY_S_PER_POINT_FRAME

ComputeFn = Callable[[int, int], float]    # seconds of compute over d01 steps a+1..b


@dataclasses.dataclass(frozen=True, eq=False)
class Overheads(Record):
    """Non-compute seconds of a segment, and where the numbers come from."""
    init_s: float
    restart_read_s: float
    restart_write_s: float
    history_frame_s: Optional[list]    # s per frame, per domain; None: FALLBACK_HISTORY_FRAME_S
    source: str


def overheads(init_s: Optional[float] = None, restart_read_s: Optional[float] = None,
              restart_write_s: Optional[float] = None,
              history_frame_s: Optional[list] = None, source: str = "fallback") -> Overheads:
    m = wtune.machine.measured
    init_s = m("init_s") if init_s is None else init_s
    restart_read_s = m("restart_io_s") if restart_read_s is None else restart_read_s
    restart_write_s = m("restart_io_s") if restart_write_s is None else restart_write_s
    return Overheads(init_s=init_s, restart_read_s=restart_read_s,
                     restart_write_s=restart_write_s, history_frame_s=history_frame_s,
                     source=source)


def fallback_restart_io_s(geo: Geometry, io_form: int) -> tuple[float, float]:
    """(write_s, read_s) of one restart (all domains) scaled from the conus25 anchor."""
    geo = Geometry.coerce(geo)
    w, r = iom.restart_anchor_s(io_form)
    f = sum(geo.points) / float(CONUS25_RESTART_POINTS)
    return w * f, r * f


def fallback_history_frame_s(geo: Geometry, io: str = "adios2") -> list:
    """Per-domain seconds for one history frame of the run's --io format (adios2 | netcdf)."""
    return iom.history_frame_s(Geometry.coerce(geo).points, iom.normalize_io_format(io))


def fallback_overheads(geo: Geometry, io_form_restart: int, io: str = "adios2") -> Overheads:
    w, r = fallback_restart_io_s(geo, io_form_restart)
    return overheads(restart_read_s=r, restart_write_s=w,
                     history_frame_s=fallback_history_frame_s(geo, io),
                     source="fallback (conus25/dorian2019 I/O anchors)")


def compute_fn_from_points(geo: Geometry, anchors_at: list, r_night: float, r_day: float,
                           sample_points: Optional[list]) -> ComputeFn:
    """compute(a, b) = C(b) - C(a), C a prefix sum over the whole case run of
    coldstart.predicted_compute_total's per-step terms (s plus its radiation term); C(N) is
    asserted equal to that function's total, so drift in coldstart fails loudly here."""
    if not anchors_at:
        raise RunError("prediction has no measured points")
    start, dt_s, n = geo.start, geo.dt, geo.steps
    stepra = geo.stepra or 1
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


def constant_compute_fn(s_per_step: float) -> ComputeFn:
    return lambda a, b: (b - a) * s_per_step


def history_frames(geo: Geometry, d: int, a: int, b: int) -> int:
    """History frames domain d writes over steps (a, b], plus the t=0 frame for a == 0. A
    resumed segment writes no start frame (write_hist_at_0h_rst stays .false.)."""
    h = geo.hist[d] if d < len(geo.hist) else None
    if not h:
        return 0
    dt_s = geo.dt

    def upto(x):              # frames with begin + j*iv <= x
        return 0 if x < h.begin_s else int(math.floor((x - h.begin_s) / h.iv_s + 1e-9)) + 1
    n = upto(b * dt_s) - upto(a * dt_s)
    if a == 0 and h.begin_s == 0:
        n += 1
    return n


def writes_restart(b: int, L: Optional[int]) -> bool:
    """Segment ending at step b writes a restart iff b is on the L grid (restart alarm)."""
    return bool(L) and b > 0 and b % L == 0


def restart_write_steps(geo: Geometry, a: int, b: int, cadence_s: Optional[float]) -> list[int]:
    """d01 steps in (a, b] at which a restart alarm of `cadence_s` (phase: the run start) rings;
    [] without a cadence. WRF rings an alarm at the first step at or past its time, so a
    cadence off the step grid rounds up to the next step."""
    if not cadence_s:
        return []
    dt_f, c = _frac(geo.dt), _frac(cadence_s)
    first = int(math.floor(_frac(a) * dt_f / c)) + 1
    out: list[int] = []
    j = first
    while True:
        n = int(math.ceil(j * c / dt_f))
        if n > b:
            return out
        if n > a:
            out.append(n)
        j += 1


def segment_wall(geo: Geometry, compute: ComputeFn, over: Overheads, a: int, b: int,
                 L: Optional[int], cadence_s: Optional[float] = None) -> float:
    """Predicted wall of segment (a, b]: compute + init + restart read (a > 0) + restart writes
    (b on the L grid; with cadence_s every alarm inside (a, b] instead) + history frames."""
    w = compute(a, b) + over.init_s
    if a > 0:
        w += over.restart_read_s
    if cadence_s:
        w += over.restart_write_s * len(restart_write_steps(geo, a, b, cadence_s))
    elif writes_restart(b, L):
        w += over.restart_write_s
    return w + history_wall(geo, over, a, b)


def history_wall(geo: Geometry, over: Overheads, a: int, b: int) -> float:
    """Seconds of history writes in segment (a, b]: frames x per-domain frame seconds."""
    per = over.history_frame_s
    w = 0.0
    for d in range(geo.max_dom):
        fs = (wtune.machine.measured("history_frame_s") if per is None
              else (per[d] if d < len(per) else per[-1]))
        w += history_frames(geo, d, a, b) * (fs or 0.0)
    return w


def time_limit_s(pred_wall_s: float, wall_limit_s: Optional[float] = None) -> float:
    raw = TIME_FACTOR * pred_wall_s + TIME_PAD_S
    t = int(math.ceil(round(raw, 6) / TIME_ROUND_S)) * TIME_ROUND_S
    return min(t, wall_limit_s) if wall_limit_s else t


def fallback_s_per_step(nl: dict, ranks: int, omp: int) -> float:
    """Conservative constant s/step: wtune.tune's cell-step model x FALLBACK_RADIATION_FACTOR;
    scc_sweep.CONSERVATIVE_S_PER_STEP when that model fails."""
    try:
        return (tune._fallback_s_per_step(nl, ranks, omp)
                * wtune.machine.measured("radiation_factor"))
    except Exception:
        return wtune.machine.measured("s_per_step")


def _parse_at(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s) if "T" in s or " " in s else dt.datetime.strptime(s, WRF_FMT)


def model_from_prediction(pred: dict, geo: Geometry, sample_points: Optional[list],
                          fallback: Overheads) -> tuple[ComputeFn, Overheads]:
    """(compute_fn, overheads) from a pred_rebuild.json record; its measured `io` block wins,
    `fallback` (fallback_overheads) fills whatever it does not measure."""
    res = pred["result"]
    pts = [(_parse_at(p["at"]), p["s_normal"]) for p in res["points"]]
    comp = compute_fn_from_points(geo, pts, res["r_night"], res["r_day"], sample_points)
    io = pred.get("io") or {}
    rst = (io.get("restart") or {}).get("per_write_s")
    hist = list(fallback.history_frame_s)
    for d, e in enumerate(io.get("history") or []):
        if e.get("median_s") is not None and d < len(hist):
            hist[d] = e["median_s"]
    measured = bool(rst) or bool(io.get("history"))
    return comp, overheads(
        init_s=res.get("init_s") or wtune.machine.measured("init_s"),
        restart_read_s=rst if rst else fallback.restart_read_s,
        restart_write_s=rst if rst else fallback.restart_write_s, history_frame_s=hist,
        source="prediction" + (" (measured io)" if measured else " + " + fallback.source))


def compute_fn_from_tune_row(row: dict, geo: Geometry, sample_points: Optional[list],
                             calibration: dict) -> Optional[ComputeFn]:
    """compute_fn of one predicted-mode tune_report ranking row (points at/s_normal/r/n_rad),
    r_night/r_day pooled exactly as wtune.probe_plan.combine pools them; None without points."""
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
