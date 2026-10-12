"""Segment planning (pure): ONE restart quantum L for the whole run, every segment one L block
from the run start (the last possibly shorter), cut on input record times when inputs are
per-record ADIOS2, plus the restart cadence every segment namelist gets.
Segments never set override_restart_timers: a resumed run would re-anchor every alarm at its
own start and lose history frames (scc R1, 2026-09-27, dorian2019 30+30 min chain)."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional, Union

from ._deps import coldstart, wp
from .common import (UNSET, Record, RunError, _PER_RECORD_KIND_RE, _fmt, _frac, _frac_lcm, _hms)
from .estimate import Overheads, restart_write_steps, segment_wall, time_limit_s, writes_restart
from .geometry import Geometry, step_time
from .layout import Candidate, Config

WALL_FRACTION = 0.8                    # --wall-limit: predicted segment wall <= 0.8 L
Checkpoint = Union[None, str, int]     # no extra restarts, "auto", or an explicit cadence
RESTART_OFF_MIN = 1000000              # no restart write (configs._apply_adios2 value)


@dataclasses.dataclass(eq=False)
class Segment(Record):
    """One segment of the plan and, once it runs, what its jobs left (run_report.json)."""
    index: int
    start_step: int
    end_step: int
    start: str                         # WRF_FMT
    end: str
    config: Config
    writes_restart: bool
    predicted_compute_s: float
    predicted_wall_s: float
    time_limit: str                    # sbatch --time
    wall_source: Any = UNSET           # set when an observed overhead replaced the prediction
    attempts: Any = UNSET              # [AttemptRecord], one per submitted job
    observed_wall_s: Any = UNSET
    run_dir: Any = UNSET               # the attempt dir the next segment restarts from


@dataclasses.dataclass(frozen=True, eq=False)
class RestartSetting(Record):
    """What every segment namelist does with restart_interval."""
    set_min: Optional[int]             # minutes to write; None: the case's own keys stay
    cadence_s: Optional[float]         # restart alarm period; None: no restart write
    label: str                         # plan-line text


@dataclasses.dataclass(frozen=True, eq=False)
class RecordAlignment(Record):
    """Segment-cut unit of per-record ADIOS2 inputs."""
    steps: int                         # d01 steps
    seconds: float
    label: str


def plan_segments_fixed(n_end: int, steps_per_segment: int, n0: int = 0) -> list[tuple[int, int]]:
    """[(a, b), ...] d01 step intervals of `steps_per_segment` from n0 to n_end, last shorter."""
    if steps_per_segment <= 0:
        raise RunError("segment length must be > 0 steps")
    segs, a = [], n0
    while a < n_end:
        b = min(a + steps_per_segment, n_end)
        segs.append((a, b))
        a = b
    return segs


def segment_steps_for_minutes(geo: Geometry, minutes: int) -> int:
    """d01 steps in `minutes`; RunError unless that is a whole number of d01 steps."""
    n = _frac(minutes * 60) / _frac(geo.dt)
    if n.denominator != 1 or n <= 0:
        raise RunError("--segment-minutes %s is not a whole number of %gs time steps"
                       % (minutes, geo.dt))
    return int(n)


def choose_restart_quantum(n_end: int, quantum: int, wall_fn: Callable,
                           budget_s: float) -> Optional[int]:
    """--wall-limit: the largest L (a multiple of `quantum` d01 steps) such that EVERY segment
    of the L grid has wall_fn(a, b, L) <= budget_s. None when the whole run fits in one
    segment; RunError when not even one quantum fits. Exhaustive: no monotonicity assumed."""
    if wall_fn(0, n_end, None) <= budget_s:
        return None
    for m in range((n_end - 1) // quantum, 0, -1):
        L = m * quantum
        if all(wall_fn(a, b, L) <= budget_s for a, b in plan_segments_fixed(n_end, L)):
            return L
    raise RunError("even %d-step segments exceed the %.0f s budget (%.0f s predicted for the "
                   "first)" % (quantum, budget_s, wall_fn(0, min(quantum, n_end), quantum)))


def assign_segments(geo: Geometry, bounds: list, candidates: list, over: Overheads,
                    L: Optional[int], wall_limit_s: Optional[float] = None,
                    cadence_s: Optional[float] = None) -> list[Segment]:
    """Segments for step intervals `bounds`: the candidate with the lowest predicted compute
    over each interval (ties -> first listed), its predicted wall and --time."""
    candidates = [Candidate.coerce(c) for c in candidates]
    segs = []
    for i, (a, b) in enumerate(bounds):
        best = min(candidates, key=lambda c: c.compute(a, b))
        wall = segment_wall(geo, best.compute, over, a, b, L, cadence_s)
        segs.append(Segment(index=i, start_step=a, end_step=b, start=_fmt(step_time(geo, a)),
                            end=_fmt(step_time(geo, b)), config=best.config,
                            writes_restart=writes_restart(b, L),
                            predicted_compute_s=best.compute(a, b), predicted_wall_s=wall,
                            time_limit=_hms(time_limit_s(wall, wall_limit_s))))
    return segs


# ---------------------------------------------------------------------------
# input record alignment (per-record ADIOS2 inputs)
# ---------------------------------------------------------------------------
def _record_intervals(nl: dict, kinds: dict) -> list[tuple[str, float]]:
    """[(label, seconds)] of every per-record input kind's record interval."""
    parts = []
    if "wrfbdy" in kinds:
        iv = float(wp.g(nl, "time_control", "interval_seconds", 0) or 0)
        if iv <= 0:
            raise RunError("per-record ADIOS2 wrfbdy needs the case's interval_seconds "
                           "(&time_control) to place segment cuts on boundary records")
        parts.append(("interval_seconds %s" % _hms(iv), iv))
    for kind, label, iv_of in (
            ("wrflowinp", "auxinput4_interval",
             lambda d: coldstart.wrf_interval_minutes(nl, "auxinput4_interval", d - 1) * 60.0),
            ("wrffdda", "gfdda_interval_m",
             lambda d: float(wp.g_dom(nl, "fdda", "gfdda_interval_m", d - 1, 0) or 0) * 60.0)):
        ivs = sorted(set(iv_of(d) for d in kinds.get(kind, ())))
        ivs = [x for x in ivs if x > 0]       # a zero/absent aux interval adds no constraint
        if ivs:
            lcm = ivs[0]
            for x in ivs[1:]:
                lcm = float(_frac_lcm(lcm, x))
            parts.append(("%s %s" % (label, "/".join(_hms(x) for x in ivs)), lcm))
    return parts


def input_record_alignment(nl: dict, geo: Geometry,
                           per_record_names: list) -> Optional[RecordAlignment]:
    """Segment-cut unit for per-record ADIOS2 inputs: WRF opens <name>_<current time>.bp at a
    segment start, so every cut must be a record time of each input (wrfbdy interval_seconds,
    wrflowinp auxinput4_interval, wrffdda gfdda_interval_m, from the run start). The unit is
    their lcm with the whole-minute step quantum; None without per-record inputs."""
    kinds: dict[str, list[int]] = {}
    for n in per_record_names or ():
        m = _PER_RECORD_KIND_RE.match(n)
        if m:
            kinds.setdefault(m.group(1), []).append(int(m.group(2)))
    parts = _record_intervals(nl, kinds)
    if not parts:
        return None
    unit = _frac_lcm(geo.dt, 60)
    for _label, s in parts:
        unit = _frac_lcm(unit, s)
    steps = unit / _frac(geo.dt)
    if steps.denominator != 1:          # unit is a multiple of dt by construction
        raise RunError("internal: input record unit %s s is not whole d01 steps" % unit)
    return RecordAlignment(steps=int(steps), seconds=float(unit),
                           label=" + ".join(lab for lab, _s in parts))


def missing_record_files(links: dict, per_record_names: list,
                         segments: list[Segment]) -> list[tuple[int, str]]:
    """[(segment index, record file)] a segment start lacks among the run-dir links, for every
    per-record input whose record files are known (dry run: one still to be converted is not
    checked). WRF opens <name>_<segment start>.bp first, so each must exist."""
    known = [n for n in per_record_names or ()
             if any(l.startswith(n + "_") for l in links or ())]
    out = []
    for s in segments:
        for n in known:
            rec = "%s_%s.bp" % (n, s.start)
            if rec not in links:
                out.append((s.index, rec))
    return out


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _quantum_for_minutes(geo: Geometry, n_end: int, segment_minutes: int,
                         align: Optional[RecordAlignment],
                         notes: Optional[list]) -> Optional[int]:
    """--segment-minutes -> L, rounded DOWN to the input record grid (a note), never up."""
    L = segment_steps_for_minutes(geo, segment_minutes)
    if (L * geo.dt) % 60:
        raise RunError("--segment-minutes must give whole minutes")
    if L >= n_end:
        return None
    if not (align and L % align.steps):
        return L
    cut = L // align.steps * align.steps
    if cut == 0:
        raise RunError(
            "--segment-minutes %d is shorter than one input record interval: per-record "
            "ADIOS2 inputs (%s) exist only every %s from the run start, and WRF opens the "
            "file of a resumed segment's start time -- use --segment-minutes %d (or a "
            "multiple), or --io netcdf" % (segment_minutes, align.label, _hms(align.seconds),
                                           int(align.seconds // 60)))
    if notes is not None:
        notes.append("--segment-minutes %d rounded down to %d: segment cuts must fall on "
                     "input record times (%s, every %s from the run start)"
                     % (segment_minutes, int(float(_frac(geo.dt) * cut) // 60),
                        align.label, _hms(align.seconds)))
    return cut


def _quantum_for_wall_limit(n_end: int, unit: int, wall_fn: Callable, wall_limit_s: float,
                            align: Optional[RecordAlignment],
                            checkpoint: Checkpoint) -> Optional[int]:
    """--wall-limit -> L, searched over multiples of the cut unit only."""
    if checkpoint not in (None, "auto"):      # an explicit cadence must divide every L
        unit = unit * checkpoint // math.gcd(unit, checkpoint)
    try:
        return choose_restart_quantum(n_end, unit, wall_fn, WALL_FRACTION * wall_limit_s)
    except RunError as e:
        if not align:
            raise
        raise RunError("%s; segment cuts must fall on input record times (%s, every %s): "
                       "raise --wall-limit, or use --io netcdf" % (e, align.label,
                                                                  _hms(align.seconds)))


def plan_run(geo: Geometry, n_end: int, candidates: list, over: Overheads,
             segment_minutes: Optional[int] = None, wall_limit_s: Optional[float] = None,
             case_restart_s: Optional[float] = None, align: Optional[RecordAlignment] = None,
             notes: Optional[list] = None,
             checkpoint: Checkpoint = None) -> tuple[Optional[int], list[Segment]]:
    """(L, segments): L the restart quantum in d01 steps (None: one segment). case_restart_s
    and checkpoint are restart_setting's (their writes are in the walls); with `align` L is a
    multiple of align.steps (rounding goes into `notes`)."""
    candidates = [Candidate.coerce(c) for c in candidates]
    align = RecordAlignment.coerce(align) if align else None
    unit = align.steps if align else geo.quantum

    def rst(L):
        return restart_setting(geo, L, n_end, case_restart_s, checkpoint, align)

    def wall_fn(a, b, L):
        cad = rst(L).cadence_s
        return min(segment_wall(geo, c.compute, over, a, b, L, cad) for c in candidates)
    if segment_minutes:
        L = _quantum_for_minutes(geo, n_end, segment_minutes, align, notes)
    elif wall_limit_s:
        L = _quantum_for_wall_limit(n_end, unit, wall_fn, wall_limit_s, align, checkpoint)
    else:
        L = None
    bounds = plan_segments_fixed(n_end, L) if L else [(0, n_end)]
    return L, assign_segments(geo, bounds, candidates, over, L, wall_limit_s, rst(L).cadence_s)


# ---------------------------------------------------------------------------
# restart cadence and checkpoints
# ---------------------------------------------------------------------------
def case_restart_interval_s(nl: dict) -> float:
    """The case's own restart interval in seconds (WRF's restart_interval[_d|_h|_m|_s] sum);
    0 when the case sets none."""
    return float(coldstart.wrf_interval_minutes(nl, "restart_interval")) * 60.0


def checkpoint_request(value: str, resume: int) -> Checkpoint:
    """--checkpoint-minutes -> None (no extra restarts), "auto" or whole minutes. "auto" asks
    for checkpoints only when --resume could use them."""
    if value == "auto":
        return "auto" if resume > 0 else None
    try:
        minutes = int(value)
    except ValueError:
        minutes = -1
    if minutes < 0:
        raise RunError("--checkpoint-minutes takes 'auto' or whole minutes >= 0, not %r" % value)
    return minutes or None


def checkpoint_steps_for_minutes(geo: Geometry, minutes: int,
                                 align: Optional[RecordAlignment] = None) -> int:
    """d01 steps of an explicit --checkpoint-minutes; RunError unless it is a multiple of the
    step every restart must sit on (input record times, else whole minutes on the step grid)."""
    align = RecordAlignment.coerce(align) if align else None
    unit = align.steps if align else geo.quantum
    n = _frac(minutes * 60) / _frac(geo.dt)
    if n.denominator != 1 or n % unit:
        raise RunError("--checkpoint-minutes %d is not a multiple of the %d min %s"
                       % (minutes, float(_frac(geo.dt) * unit) // 60,
                          "input record step (%s)" % align.label if align
                          else "step grid"))
    return int(n)


def auto_checkpoint_steps(n_end: int, L: Optional[int], unit: int) -> Optional[int]:
    """Checkpoint cadence in d01 steps: the smallest multiple of `unit` >= run length / 8 that
    falls inside the run -- with segments, inside each one and dividing L. None: none fits."""
    m = max(1, -(-n_end // (8 * unit))) * unit
    while m < (L or n_end):
        if not L or L % m == 0:
            return m
        m += unit
    return None


def _checkpoint_setting(geo: Geometry, L: Optional[int], n_end: int, checkpoint: Checkpoint,
                        align: Optional[RecordAlignment]) -> Optional[RestartSetting]:
    """restart_setting's result for a checkpoint cadence ("auto" or explicit d01 steps), or
    None when no auto checkpoint fits."""
    explicit = checkpoint != "auto"
    m = checkpoint if explicit else auto_checkpoint_steps(
        n_end, L, align.steps if align else geo.quantum)
    if m is None:
        return None
    if L and L % m:
        raise RunError("--checkpoint-minutes %d does not divide the %s segment length: every "
                       "segment must end on a restart" % (restart_interval_minutes(geo, m),
                                                          _hms(float(_frac(geo.dt) * L))))
    minutes = restart_interval_minutes(geo, m)
    source = "--checkpoint-minutes" if explicit else "auto, for --resume"
    return RestartSetting(set_min=minutes, cadence_s=minutes * 60.0,
                          label="checkpoints every %d min (%s)" % (minutes, source))


def restart_setting(geo: Geometry, L: Optional[int], n_end: int,
                    case_restart_s: Optional[float] = None, checkpoint: Checkpoint = None,
                    align: Optional[RecordAlignment] = None) -> RestartSetting:
    """The restart cadence of every segment namelist. The case's own interval is kept when it
    writes inside the run (one segment) or divides L (climate-ea50 lost its checkpoints);
    checkpoint (None, "auto", explicit d01 steps) adds restarts for --resume when the case
    puts none inside a segment; else L, or off for one segment."""
    align = RecordAlignment.coerce(align) if align else None
    run_s = float(_frac(geo.dt) * n_end)
    case_s = float(case_restart_s or 0)
    L_s = _frac(geo.dt) * L if L else None
    keep = None
    if not L and 0 < case_s < run_s:
        keep = "every %s (case's restart_interval kept; one segment)" % _hms(case_s)
    elif L and case_s > 0 and L_s % _frac(case_s) == 0:
        keep = ("every %s (case's restart_interval kept; it divides the %s segment quantum)"
                % (_hms(case_s), _hms(float(L_s))))
    inside = keep and (not L or case_s < L_s)       # the case restarts inside every segment
    if checkpoint and (checkpoint != "auto" or not inside):
        ckpt = _checkpoint_setting(geo, L, n_end, checkpoint, align)
        if ckpt:
            return ckpt
    if keep:
        return RestartSetting(set_min=None, cadence_s=case_s, label=keep)
    if not L:
        return RestartSetting(
            set_min=RESTART_OFF_MIN, cadence_s=None, label="off (one segment%s)" % (
                "; case restart_interval %s >= run length" % _hms(case_s) if case_s else ""))
    m = restart_interval_minutes(geo, L)
    return RestartSetting(
        set_min=m, cadence_s=float(L_s), label="%d min (every segment%s)" % (
            m, "; case restart_interval %s does not divide it" % _hms(case_s) if case_s else ""))


def restart_times(geo: Geometry, a: int, b: int, cadence_s: Optional[float]) -> list[str]:
    """WRF_FMT times of the restart writes in (a, b] (restart_write_steps)."""
    return [_fmt(step_time(geo, n)) for n in restart_write_steps(geo, a, b, cadence_s)]


def restart_interval_minutes(geo: Geometry, L: Optional[int]) -> int:
    """restart_interval (minutes) for quantum L; RESTART_OFF_MIN for a single segment."""
    if not L:
        return RESTART_OFF_MIN
    m = _frac(geo.dt) * L / 60
    if m.denominator != 1:
        raise RunError("restart quantum %d steps is not whole minutes" % L)
    return int(m)
