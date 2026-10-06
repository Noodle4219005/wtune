"""Case geometry: the d01 time axis (start, step, run length) and per-domain sizes, read from a
parsed namelist, and the conversions between times and d01 step indices."""
from __future__ import annotations

import dataclasses
import datetime as dt
import fractions
from typing import Optional, Union

from ._deps import coldstart, wp
from .common import WRF_FMT, Record, RunError, _TIME_PARTS, _fmt, _frac, _frac_lcm


@dataclasses.dataclass(frozen=True, eq=False)
class HistoryStream(Record):
    """One domain's history cadence."""
    iv_s: float                        # s between frames
    begin_s: float                     # s from the run start to the first frame


@dataclasses.dataclass(frozen=True, eq=False)
class Geometry(Record):
    """Run geometry of a case."""
    start: dt.datetime
    end: dt.datetime
    dt: float                          # s, d01 time step
    steps: int                         # d01 steps of the case's own run
    stepra: Optional[int]              # d01 steps per radiation call
    max_dom: int
    radt_s: Optional[float]            # s, d01 radiation interval
    quantum: int                       # d01 steps per whole-minute boundary
    hist: list                         # per domain: HistoryStream, or None (no history)
    points: list                       # per domain: e_we x e_sn x e_vert (I/O fallbacks)


def _history_streams(nl: dict, max_dom: int) -> list:
    io_hist = wp.g(nl, "time_control", "io_form_history", 2)
    hist = []
    for d in range(max_dom):
        iv = coldstart.wrf_interval_minutes(nl, "history_interval", d) * 60.0
        begin = coldstart.wrf_interval_minutes(nl, "history_begin", d) * 60.0
        hist.append(HistoryStream(iv_s=iv, begin_s=begin) if io_hist and iv > 0 else None)
    return hist


def case_geometry(nl: dict) -> Geometry:
    """Run geometry from a parsed namelist; RunError when the run cannot be laid on d01 steps."""
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
    hist = _history_streams(nl, max_dom)
    points = []
    for d in range(max_dom):
        e_we, e_sn, e_vert = wp.get_domain_geom(nl, d)
        points.append(e_we * e_sn * e_vert)
    return Geometry(start=start, end=start + dt.timedelta(seconds=run_s), dt=dt_s, steps=steps,
                    stepra=stepra, max_dom=max_dom,
                    radt_s=float(radt) * 60.0 if radt else None, quantum=int(quantum),
                    hist=hist, points=points)


def domain_start(nl: dict, d: int) -> Optional[dt.datetime]:
    """Domain d's (0-based) own start_* datetime (broadcast-last-value), None when absent."""
    vals = [wp.g_dom(nl, "time_control", "start_" + p, d, 0 if i > 2 else None)
            for i, p in enumerate(_TIME_PARTS)]
    if any(v is None for v in vals[:3]):
        return None
    return dt.datetime(int(vals[0]), int(vals[1]), int(vals[2])) + dt.timedelta(
        hours=int(vals[3] or 0), minutes=int(vals[4] or 0), seconds=int(vals[5] or 0))


def step_time(geo: Geometry, n: int) -> dt.datetime:
    return geo.start + dt.timedelta(seconds=float(_frac(geo.dt) * n))


def _step_fraction(geo: Geometry, t: dt.datetime) -> fractions.Fraction:
    """Datetime t as a (possibly fractional) number of d01 steps from the run start."""
    off_ms = int(round((t - geo.start).total_seconds() * 1000))
    return fractions.Fraction(off_ms, 1000) / _frac(geo.dt)


def step_of(geo: Geometry, t: str) -> Optional[int]:
    """d01 step index of WRF_FMT time t; None when it is not on a whole step."""
    n = _step_fraction(geo, dt.datetime.strptime(t, WRF_FMT))
    return int(n) if n.denominator == 1 else None


def resolve_end_step(geo: Geometry, stop_at: Union[str, dt.datetime, None] = None) -> int:
    """d01 step index of the run end: the case's own end, or --stop-at when earlier. stop_at must
    fall on a whole d01 step after the run start."""
    if stop_at is None:
        return geo.steps
    t = dt.datetime.strptime(stop_at, WRF_FMT) if isinstance(stop_at, str) else stop_at
    n = _step_fraction(geo, t)
    if n.denominator != 1:
        raise RunError("--stop-at %s is not on a whole time step (dt=%gs from %s)"
                       % (_fmt(t), geo.dt, _fmt(geo.start)))
    n = int(n)
    if n <= 0:
        raise RunError("--stop-at %s is not after the run start %s" % (_fmt(t), _fmt(geo.start)))
    return min(n, geo.steps)
