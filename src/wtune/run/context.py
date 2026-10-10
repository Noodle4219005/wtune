"""The records `wtune run` passes between its stages: the CLI options, the plan made before
the first job, the injected runners (Slurm, run-dir inspector, clock, log) and the
controller's context."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from .estimate import Overheads
from .geometry import Geometry
from .inputs import InputPlan
from .plan import RestartSetting, Segment


@dataclasses.dataclass(frozen=True)
class RunOptions:
    """The parsed `wtune run` command line (build_arg_parser's destinations)."""
    case_dir: str
    nodes: Optional[int]
    nx: Optional[int]
    ny: Optional[int]
    numtiles: Optional[int]
    omp: Optional[int]
    wall_limit: Optional[str]          # HH:MM:SS per segment job
    segment_minutes: Optional[int]
    stop_at: Optional[str]
    prediction: Optional[str]          # pred_rebuild.json path
    tune_report: Optional[str]         # tune_report.json path
    restart_io_form: int
    io: str                            # adios2 | netcdf
    cfl_rescue: bool
    resume: int                        # auto-resumes allowed per run
    checkpoint_minutes: str            # "auto" or whole minutes
    sidecar: bool
    wrfda_forecast_only: bool
    out: Optional[str]
    project: Optional[str]
    dry_run: bool
    wrf_tree: Optional[str]
    env_source: str
    campaign_dir: str
    sacct_cmd: str
    poll_interval: float               # s between sacct polls


@dataclasses.dataclass
class RunPlan:
    """What is decided before the first job: the case, the wall model, the inputs, the
    segments and their restart cadence, and the notes printed with the plan."""
    case_dir: str
    out: str
    text: str                          # the case's namelist.input
    geo: Geometry
    n_end: int                         # d01 step the run ends at (--stop-at)
    wall_limit_s: Optional[float]
    over: Overheads
    replan: bool                       # no prediction: later --time from observed steps
    inputs: InputPlan
    L: Optional[int]                   # restart quantum, d01 steps
    segments: list[Segment]
    restart: RestartSetting
    notes: list[str]
    nc_root: Optional[str] = None      # --sidecar: the `wtune post` netCDF cache


@dataclasses.dataclass(frozen=True)
class Runners:
    """The outside world, injected so tests replace it."""
    sbatch_runner: Callable
    sacct_runner: Callable
    snapshot_fn: Callable              # wtune_watch.snapshot(rundir)
    sleep: Callable
    log: Callable


@dataclasses.dataclass
class RunState:
    """Budgets that are per run, not per segment."""
    rescue_used: bool = False
    resumes: int = 0


@dataclasses.dataclass
class RunContext:
    """What the controller runs with: options, plan, runners and the report it keeps."""
    opts: RunOptions
    plan: RunPlan
    runners: Runners
    report: dict                       # run_report.json, rewritten after every attempt
    wrf_tree: Optional[str] = None
    bp_copier: Optional[Callable] = None       # fresh per-record input copy
    case_adios2_xml: Optional[str] = None      # the case's own adios2.xml text
    sidecar: Any = None                        # wtune_sidecar.Sidecar (--sidecar)
    state: RunState = dataclasses.field(default_factory=RunState)

    @property
    def geo(self) -> Geometry:
        return self.plan.geo

    @property
    def segments(self) -> list[Segment]:
        return self.plan.segments

    @property
    def log(self) -> Callable:
        return self.runners.log
