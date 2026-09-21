"""Common interface every profiler backend implements. Policy and state-model code depend
only on this module, never on polling.py or privileged.py directly, so swapping the
backend behind capability.select_backend() never touches downstream code.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Dict, List, Optional


class ThreadRunState(enum.Enum):
    """The raw /proc 'state' field, translated. Distinct from state_model.ThreadState:
    this is the kernel's instantaneous report, not our own accumulated inference."""
    RUNNING_OR_RUNNABLE = "R"
    SLEEPING = "S"
    DISK_WAIT = "D"
    ZOMBIE = "Z"
    STOPPED = "T"
    OTHER = "?"


@dataclasses.dataclass
class ThreadSample:
    tid: int
    timestamp_ns: int
    run_state: ThreadRunState
    on_cpu_ns_delta: int   # utime+stime delta since the previous sample, in ns; 0 on first sample
    wall_ns_delta: int     # wall-clock time since the previous sample, in ns; 0 on first sample
    last_cpu: int          # /proc stat 'processor' field: which CPU it last ran on
    migrated: bool         # last_cpu changed since the previous sample
    hw_counters: Optional[Dict[str, int]] = None  # cycles/instructions/cache-misses if sampled this cycle


class Profiler:
    """Abstract backend. attach() is called once per discovered/refreshed TID set;
    sample() once per polling cycle by the engine."""

    def attach(self, tids: List[int]) -> None:
        raise NotImplementedError

    def sample(self) -> Dict[int, ThreadSample]:
        raise NotImplementedError

    def detach(self) -> None:
        """Release any resources (subprocess handles, fds). Must be idempotent."""
        raise NotImplementedError
