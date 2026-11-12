"""Typed state of one --pipeline full run (decision 0059): the report section it builds
(PipelineSummary and its records, serialised by to_dict()) and the run state its phase
helpers share (PipelineRun)."""
from __future__ import annotations

import dataclasses
from typing import Any

from .common import UNSET, Record
from .context import Ctx
from .jobs import job_timing

NOT_RUN_BUDGET = "not run (budget)"


@dataclasses.dataclass(frozen=True)
class ClockMark:
    """The three clocks at one instant; the budget runs on `run`."""
    run: float        # s our jobs were RUNNING (the budget clock)
    wall: float       # s wall time
    queue: float      # s accumulated queue-only wait


def _mark(ctx: Ctx) -> ClockMark:
    return ClockMark(run=ctx.elapsed(), wall=ctx.wall_elapsed(), queue=ctx.queue_wait_s)


@dataclasses.dataclass
class PhaseRecord(Record):
    """One phase of pipeline.phases. start/end/limit are on the RUNNING clock; wall - running -
    queue wait is controller time (submission, reports, predictions, gate). `extra` keys follow
    `status` in the JSON; the job timing fields are added by _attach_phase_timing."""
    name: str
    start_s: float
    end_s: float
    running_s: float
    wall_s: float
    queue_wait_s: float
    controller_s: float
    limit_s: float
    status: str
    extra: dict = dataclasses.field(default_factory=dict)
    jobs: Any = UNSET
    queue_s_sum: Any = UNSET           # s, summed over concurrent jobs (can exceed wall_s)
    queue_s_max: Any = UNSET           # s, the longest single job's queue wait
    job_run_s_sum: Any = UNSET
    job_run_s_max: Any = UNSET

    def to_dict(self) -> dict:
        d = {k: v for k, v in Record.to_dict(self).items() if k != "extra"}
        timing = {k: d.pop(k) for k in ("jobs", "queue_s_sum", "queue_s_max", "job_run_s_sum",
                                        "job_run_s_max") if k in d}
        d.update(self.extra)
        d.update(timing)
        return d


def _phase_record(name: str, m0: ClockMark, m1: ClockMark, limit_s: float, status: str,
                  **extra: Any) -> PhaseRecord:
    running = max(0.0, m1.run - m0.run)
    wall = max(0.0, m1.wall - m0.wall)
    queue = max(0.0, m1.queue - m0.queue)
    return PhaseRecord(name=name, start_s=m0.run, end_s=m1.run, running_s=running, wall_s=wall,
                       queue_wait_s=queue, controller_s=max(0.0, wall - running - queue),
                       limit_s=limit_s, status=status, extra=extra)


def _attach_phase_timing(ctx: Ctx, phases: list[PhaseRecord], job_slices: dict) -> None:
    """Per phase, sacct queue / run time of the jobs it SUBMITTED (the rebuild jobs belong to
    phase 0 although they run beside phase 1); None where sacct reports nothing."""
    all_ids = [j for ids in job_slices.values() for j in ids]
    timing = job_timing(ctx, all_ids)
    for ph in phases:
        ids = job_slices.get(ph.name, [])
        q = [timing[j]["queue_s"] for j in ids if timing.get(j, {}).get("queue_s") is not None]
        r = [timing[j]["run_s"] for j in ids if timing.get(j, {}).get("run_s") is not None]
        ph.jobs = list(ids)
        ph.queue_s_sum, ph.queue_s_max = (sum(q), max(q)) if q else (None, None)
        ph.job_run_s_sum, ph.job_run_s_max = (sum(r), max(r)) if r else (None, None)


@dataclasses.dataclass
class StepDown(Record):
    """One automatic level step-down (pre-run, or after a measurement wave)."""
    when: str
    frm: Any = UNSET
    to: int = 0
    T_real_s: float | None = None      # s, the real run the level estimates were held against
    message: Any = UNSET


@dataclasses.dataclass
class LevelInfo(Record):
    """pipeline.level: requested and effective -O level and the per-level estimates."""
    requested: int
    effective: int
    estimate_s: dict                   # {"O1": s, ...} expected RUNNING time per level
    estimate_info: Any = UNSET
    T_cold_s: Any = UNSET
    stepped_down: list = dataclasses.field(default_factory=list)
    actual_running_s: Any = UNSET


@dataclasses.dataclass
class RebuildInfo(Record):
    """Phase 0's point plan and rebuild jobs (pipeline.rebuild, and the plan phase's extra keys),
    filled step by step so a failure keeps what was known."""
    points: Any = UNSET                # planned point times (isoformat)
    spinup_min: Any = UNSET
    notes: Any = UNSET
    prediction_points: Any = UNSET     # points this case needs at the requested precision
    points_source: Any = UNSET
    rebuild_jobs: Any = UNSET
    status: Any = UNSET


@dataclasses.dataclass
class PredictionPlanRecord(Record):
    """pipeline.prediction_plan: the points re-planned after phase 1."""
    points: list
    band: float
    method: str
    source: str
    T_est_s: float | None
    precision_min: Any
    measured: bool
    rebuild_cancelled: Any = UNSET


@dataclasses.dataclass
class Phase2Plan(Record):
    """pipeline.predict_plan: the degradation ladder tried and the plan chosen."""
    left_s: float
    tried: list
    startup_s: float
    chosen: str | None
    degraded: bool


@dataclasses.dataclass
class ShortRun(Record):
    message: str
    real_run_s: float


@dataclasses.dataclass
class PredictionBand(Record):
    method: str
    band: float
    error_min: float | None


@dataclasses.dataclass
class PipelineSummary(Record):
    """report["pipeline"]: budget, level, phases, prediction, physics advice and totals of one
    pipeline (or -O0) run. UNSET fields are absent from the JSON."""
    total_steps: Any = UNSET
    B_s: Any = UNSET                   # s hard cap of RUNNING time, None without --minutes
    budget_source: Any = UNSET
    T_updated_s: Any = UNSET           # s whole run implied by the best sweep s/step (info)
    phases: list = dataclasses.field(default_factory=list)
    rebuild: Any = UNSET
    extension: Any = UNSET
    max_queue_s: Any = UNSET
    queue_cap_hit: Any = UNSET
    shares: Any = UNSET                # {phase: share of B}
    level: Any = UNSET
    n_points_est: Any = UNSET
    n_full_est: Any = UNSET
    T_cold_s: Any = UNSET
    T_cold_compute_s: Any = UNSET
    short_run: Any = UNSET
    T_updated_compute_s: Any = UNSET
    prediction_plan: Any = UNSET
    predict_plan: Any = UNSET
    prediction: Any = UNSET
    physics_weights: Any = UNSET
    physics_points: Any = UNSET
    physics_fidelity_gate: Any = UNSET
    physics_baseline: Any = UNSET
    physics_advice: Any = UNSET
    frames_kept: Any = UNSET
    frames_deleted: Any = UNSET
    time_suggestion: Any = UNSET
    prediction_band: Any = UNSET
    total_running_s: Any = UNSET
    total_wall_s: Any = UNSET
    jobs_busy_s: Any = UNSET           # sacct: seconds during which any job of the tune ran
    total_queue_wait_s: Any = UNSET
    total_controller_s: Any = UNSET
    predicted_real_s: Any = UNSET
    predicted_real_source: Any = UNSET
    hard_deadline_s: Any = UNSET
    over_deadline_s: Any = UNSET


@dataclasses.dataclass
class PointPreview:
    """The prediction points planned from the namelist alone (no job); ok False with `error`
    when planning failed."""
    ok: bool
    spinup_min: Any = None
    dt: Any = None
    stepra: Any = None
    points: list = dataclasses.field(default_factory=list)
    pkw: dict = dataclasses.field(default_factory=dict)     # run_start / run_end / window_s
    notes: list = dataclasses.field(default_factory=list)
    wrapped: list = dataclasses.field(default_factory=list)  # [{index, point}]
    error: str | None = None


@dataclasses.dataclass
class ColdEstimate:
    """The pre-run cold-start estimate: per-level tuning time and the point plan behind it."""
    est: dict | None = None
    T_cold: float | None = None
    est_info: dict = dataclasses.field(default_factory=dict)
    pplan0: dict | None = None
    n_full: int = 0


@dataclasses.dataclass
class RebuildSubmission:
    """What phase 0 submitted the rebuild jobs from (a later re-plan needs it)."""
    wrapped: list
    anchor: dict | None
    case_dir: str
    spinup_min: int
    stepra: int
    dt: float
    sample_points: Any


@dataclasses.dataclass
class MeasuredLayout:
    """One layout measured at the rebuilt points in phase 2, with its predicted whole-run time;
    phase 3 falls back to phase 1's best layout with only layout / nodes / cores set."""
    layout: dict
    nodes: int
    cores: int
    cid: str | None = None
    pred_total_s: float | None = None
    pred_low_s: float | None = None
    pred_high_s: float | None = None
    result: dict | None = None
    meas: list | None = None
    jobs: list | None = None
    points: list | None = None         # rebuilt point indices it was measured at


@dataclasses.dataclass
class PipelineRun:
    """Mutable state of one run_pipeline_full call, shared by its phase helpers. `plan` is the
    {"level"} dict run_node_search reads live (a step-down takes effect there); B is the hard
    cap in s of RUNNING time (inf without --minutes), cum the cumulative phase shares."""
    ctx: Ctx
    t_start: float
    level_req: int
    plan: dict
    physics: bool
    extension: bool
    guard: bool
    total_steps: int | None
    B: float
    cum: dict
    pipe: PipelineSummary
    slices: dict = dataclasses.field(default_factory=dict)     # phase -> job ids it submitted
    _last: int = dataclasses.field(init=False, default=0)
    pre: PointPreview | None = None
    pplan0: dict | None = None
    prep_handle: dict | None = None
    sub: RebuildSubmission | None = None
    rebuilt: list | None = None
    rebuild_pred: dict | None = None   # phase 2 rebuild-winner result (combine() + its jobs)
    winner: MeasuredLayout | None = None
    ranked_all: list = dataclasses.field(default_factory=list)  # [MeasuredLayout] by pred time
    frame_runs: list = dataclasses.field(default_factory=list)  # [(rp, cid, job)] with frames
    nl: Any = None                     # prediction context (pc_ok once it was derived)
    run_start: Any = None
    run_end: Any = None
    dt: Any = None
    stepra: Any = None
    calibration: Any = None
    sample_points: Any = None
    case_dir: Any = None
    pc_ok: bool = False
    advice: Any = None

    def __post_init__(self) -> None:
        self._last = len(self.ctx.owned_jobs)

    def limit(self, name: str) -> float:
        """Cumulative running-time limit of phase `name` (inf without a cap)."""
        return self.t_start + self.cum[name] * self.B

    def cap_hit(self) -> bool:
        """True once the --max-queue-hours cap stopped the run."""
        return bool(self.ctx.shared_queue["cap_hit"])

    def cut(self, name: str) -> None:
        """Close phase `name`'s job slice: the jobs owned since the previous cut."""
        self.slices[name] = list(self.ctx.owned_jobs[self._last:])
        self._last = len(self.ctx.owned_jobs)
