"""The run context (Ctx) every stage reads and updates, its injectable runners, and case
resolution (case json, namelist, profile, step counts)."""
from __future__ import annotations

import copy
import csv
import dataclasses
import os
import subprocess
import time
from typing import Any, Callable

from ._deps import (campaign_configs, sweep, wtune_case, wtune_config, wtune_predict,
                    wtune_probe_plan)

DEFAULT_MAX_NODES = 13                 # scc compute pool: work[1-13], head excluded (scc_sweep.py)
DEFAULT_STEPS = 20                     # probe steps per run (t0 search)
DEFAULT_SEED = 1
DEFAULT_KAPPA = 0.5                    # light exploration (spec: "kappa small, e.g. 0.5")
DEFAULT_HARDWARE_ID = sweep.DEFAULT_HARDWARE_ID
CONFIRM_REPS = 2                       # replicates per config in the confirm job
DEFAULT_POLL_INTERVAL_S = 20.0         # s between sacct polls
DEFAULT_STARTUP_S = sweep.STARTUP_S    # s/run startup, scc_sweep's documented constant, until
                                       # round0 measures this case's own (measure_startup_s)


class QueueCapExceeded(RuntimeError):
    """--max-queue-hours reached: in-flight jobs were cancelled and the run stops."""


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def default_scancel_runner(job_ids: list) -> str:
    if not job_ids:
        return ""
    out = subprocess.run(["scancel"] + [str(j) for j in job_ids],
                          capture_output=True, text=True, check=False)
    return out.stdout


def default_ledger_reader(ledger_dir: str, case: str, job_ids: list) -> list[dict]:
    """Every row of ledger_<case>_<job>.csv for these job ids (strict_campaign.sbatch naming)."""
    rows: list[dict] = []
    for jid in job_ids:
        path = os.path.join(ledger_dir, "ledger_%s_%s.csv" % (case, jid))
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


@dataclasses.dataclass
class Ctx:
    """Everything a tuning stage needs: the case, the allocation, the budget clock, the injected
    runners (sbatch / sacct / scancel / ledger reader / proposer / clock) and the run's job log.
    Stages copy it with dataclasses.replace() for a finalist or a phase."""
    case: str
    case_json: dict
    namelist_path: str
    nodes: int
    cores_per_node: int
    hardware_id: str
    steps: int
    seed: int
    kappa: float
    budget_s: float
    max_nodes: int
    out_dir: str
    ledger_dir: str
    env_source: str
    sbatch_script: str
    sacct_cmd: str
    poll_interval: float
    sbatch_runner: Callable[[str], str]
    sacct_runner: Callable[[list], str]
    scancel_runner: Callable[[list], Any]
    ledger_reader: Callable[[str, list], list]
    proposer: Any
    clock: Any
    start_time: float = 0.0
    owned_jobs: list = dataclasses.field(default_factory=list)
    cancelled_jobs: list = dataclasses.field(default_factory=list)
    # --config (wtune/config.py); the defaults are a no-op filter / the pre---config behaviour
    search_cfg: dict = dataclasses.field(
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["search"]))
    exclude_rules: list = dataclasses.field(default_factory=list)
    is_fire: bool = False
    chem_e_sn: Any = None              # e_sn list of an active WRF-Chem case (j-patch guard)
    omp_cap: Any = None                # 1 for WRF-Chem/Hydro/WRFDA (dmpar-only builds)
    confirm_reps: int = CONFIRM_REPS
    prediction_cfg: dict = dataclasses.field(
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["prediction"]))
    probe_cfg: dict = dataclasses.field(          # probe.* (points, spinup_min, ...) for the winner
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["probe"]))
    ranking_start_note: str | None = None
    startup_s: float = DEFAULT_STARTUP_S          # s/run every run-time estimate sizes off
    startup_source: str = "documented constant (scc_sweep.STARTUP_S)"
    startup_runs: int | None = None               # completed round0 runs behind a measurement
    restart_io_form: int = 14                     # io_form of the point jobs' shared restart
    ranking_gates: list = dataclasses.field(default_factory=list)   # evaluate_job_gates records
    budget_mode: str = "wall"                     # "running": queue-only waits are not charged
    queue_only_s: float = 0.0
    max_wall_s: float | None = None               # hard wall-clock cap (running budget only)
    stage1_max_running: int = 0                   # point jobs assumed to run at once
    background_jobs: list = dataclasses.field(default_factory=list)  # RUNNING for the clock only
    anchor_override: dict | None = None           # node search: the sweep's hybrid ANCHOR at nodes
    core_budget: int | None = None                # node search: cores/node the layouts may use
    full_space: bool = False                      # enumerate every legal layout (O3)
    layout_notes: dict | None = None              # shared across ctx copies: untried / enumerated
    node_health: dict | None = None               # node_preflight result
    # --pipeline full: one {"s", "cap_hit", "run_s"} accumulator shared by every ctx copy
    shared_queue: dict | None = None
    max_queue_s: float | None = None              # raise QueueCapExceeded at this queue-only wait

    @property
    def queue_wait_s(self) -> float:
        return self.shared_queue["s"] if self.shared_queue is not None else self.queue_only_s

    def add_queue_wait(self, seconds: float) -> None:
        if self.shared_queue is not None:
            self.shared_queue["s"] += seconds
        else:
            self.queue_only_s += seconds

    def add_running(self, seconds: float) -> None:
        """--pipeline full: charge `seconds` during which at least one of our jobs was RUNNING."""
        if self.shared_queue is not None and "run_s" in self.shared_queue:
            self.shared_queue["run_s"] += seconds

    def wall_elapsed(self) -> float:
        return self.clock.now() - self.start_time

    def elapsed(self) -> float:
        """The budget clock: wall time, or (budget_mode "running") time our jobs ran."""
        if self.budget_mode == "running":
            if self.shared_queue is not None and "run_s" in self.shared_queue:
                return self.shared_queue["run_s"]     # pipeline: controller work excluded too
            return self.clock.now() - self.start_time - self.queue_wait_s
        return self.clock.now() - self.start_time


def resolve_case(case_id: str, cores_per_node: int) -> tuple[dict, str, dict]:
    """(case json, namelist path, wtune_case profile) of a registered case."""
    case_json = sweep.resolve_case_json(case_id)
    prep_dir = sweep._expand(case_json["prep_dir"])
    namelist_path = sweep._expand(campaign_configs._case_namelist_path(case_json))
    _, wrfinput_path = wtune_case.find_case_inputs(prep_dir)
    profile = wtune_case.build_profile(namelist_path, wrfinput_path, cores_per_node)
    return case_json, namelist_path, profile


def resolve_auto_probe_steps(namelist_path: str) -> tuple[int, str]:
    """probe.steps='auto': (steps, note) with steps = max(30, stepra+10), so every probe holds a
    radiation step (wtune_probe_plan.steps_for_stepra). Never raises: DEFAULT_STEPS plus a note
    when the namelist or the prediction modules are unavailable."""
    if wtune_predict is None or wtune_probe_plan is None:
        return DEFAULT_STEPS, ("probe.steps='auto' requested but wtune_predict/wtune_probe_plan "
                                "is not importable -- using DEFAULT_STEPS=%d instead" % DEFAULT_STEPS)
    try:
        nl = wtune_predict.parse_namelist(namelist_path)
        dt = wtune_predict.get_timestep_seconds(nl)
        radt = wtune_predict.g(nl, "physics", "radt", None)
        if radt is None:
            raise ValueError("no radt in &physics")
        stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
        steps = wtune_probe_plan.steps_for_stepra(stepra)
        return steps, ("probe.steps='auto' resolved to %d steps (dt=%.3fs, radt=%smin, stepra=%d)"
                        % (steps, dt, radt, stepra))
    except Exception as e:
        return DEFAULT_STEPS, ("probe.steps='auto' requested but could not be resolved from %s "
                                "(%s: %s) -- using DEFAULT_STEPS=%d instead"
                                % (namelist_path, type(e).__name__, e, DEFAULT_STEPS))


def _total_steps(namelist_path: str) -> int | None:
    """d01 steps of the whole run (run length / time_step), or None when unavailable."""
    if wtune_predict is None:
        return None
    try:
        steps_n, _run_s, _dt, _note = wtune_predict.get_d01_steps(
            wtune_predict.parse_namelist(namelist_path))
        return int(steps_n) if steps_n and steps_n > 0 else None
    except Exception:
        return None


def cores_used(ctx: Ctx) -> int:
    """Cores per node the layouts may use: the finalist's core_budget, else all of them."""
    return ctx.core_budget or ctx.cores_per_node


def _namelist_dt(ctx: Ctx) -> float:
    """d01 time step (s) of the case's namelist."""
    return wtune_predict.get_d01_steps(wtune_predict.parse_namelist(ctx.namelist_path))[2]
