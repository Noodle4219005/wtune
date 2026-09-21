"""Out-of-process discovery of WRF MPI rank PIDs and OpenMP thread TIDs.

Never touches WRF's address space (no ptrace, no /proc/<pid>/mem) -- only reads /proc
metadata the kernel already exposes for same-uid processes.
"""
from __future__ import annotations

import dataclasses
import os
import socket
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

WRF_EXE_NAME = "wrf.exe"

# Same priority order wtimer/wtlewi.c:89-90 already reads in-process; kept identical so a
# future cross-check against that mechanism has nothing to reconcile.
_RANK_ENV_VARS = ("PMI_RANK", "SLURM_PROCID")


@dataclasses.dataclass
class RankInfo:
    rank: int
    pid: int
    tids: List[int]
    node: str
    orig_affinity: Dict[int, FrozenSet[int]]  # tid -> original sched_getaffinity() mask


class DiscoveryError(RuntimeError):
    """Raised when a WRF process was found but its rank cannot be established -- this is a
    hard failure, never silently skipped, because a caller that treated it as "no rank
    here" would silently run with partial coverage."""


def _read_environ(pid: int) -> Dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    env: Dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if b"=" not in entry:
            continue
        k, _, v = entry.partition(b"=")
        env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def _rank_from_environ(env: Dict[str, str]) -> Optional[int]:
    for var in _RANK_ENV_VARS:
        if var in env:
            try:
                return int(env[var])
            except ValueError:
                continue
    return None


def _cmdline_matches(pid: int, exe_name: str) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    if not raw:
        return False
    argv0 = raw.split(b"\0", 1)[0]
    return Path(argv0.decode(errors="replace")).name == exe_name


def _list_tids(pid: int) -> List[int]:
    try:
        return sorted(int(p.name) for p in Path(f"/proc/{pid}/task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return []


def _affinity_snapshot(tids: List[int]) -> Dict[int, FrozenSet[int]]:
    snap: Dict[int, FrozenSet[int]] = {}
    for tid in tids:
        try:
            snap[tid] = frozenset(os.sched_getaffinity(tid))
        except (ProcessLookupError, PermissionError):
            continue
    return snap


def discover_ranks(exe_name: str = WRF_EXE_NAME, uid: Optional[int] = None) -> Dict[int, RankInfo]:
    """Scan /proc for WRF rank processes owned by `uid` (default: our own uid).

    Returns {rank: RankInfo}. A rank whose PID cannot be correlated to a rank number
    (no PMI_RANK/SLURM_PROCID in its environment) is SKIPPED, not guessed. A rank whose
    environ is unreadable despite a uid match raises DiscoveryError immediately -- that
    means the environ-based approach this was designed around does not work on this host
    (see the rtctl plan's Open Items: this must be re-verified on an actual f1 compute
    node, not assumed from the login-node check), and callers must not proceed as if
    discovery silently degrades to zero coverage.
    """
    if uid is None:
        uid = os.getuid()
    node = socket.gethostname()
    ranks: Dict[int, RankInfo] = {}

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if entry.stat().st_uid != uid:
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue  # process exited mid-scan; not a real failure
        if not _cmdline_matches(pid, exe_name):
            continue
        try:
            env = _read_environ(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited mid-scan
        except PermissionError as exc:
            raise DiscoveryError(
                f"/proc/{pid}/environ not readable even though pid is owned by uid {uid} -- "
                "the environ-based rank correlation this design assumes does not work on "
                "this host. Implement the /proc/<pid>/cgroup fallback (not yet written) "
                "before relying on discover_ranks() here."
            ) from exc
        rank = _rank_from_environ(env)
        if rank is None:
            continue
        tids = _list_tids(pid)
        ranks[rank] = RankInfo(
            rank=rank, pid=pid, tids=tids, node=node,
            orig_affinity=_affinity_snapshot(tids),
        )
    return ranks


def refresh_tids(info: RankInfo) -> RankInfo:
    """Re-enumerate TIDs for a rank whose OpenMP threads may have grown since discovery.
    Newly-seen TIDs get their affinity recorded as a rollback baseline; existing TIDs and
    their recorded baselines are left untouched."""
    tids = _list_tids(info.pid)
    orig_affinity = dict(info.orig_affinity)
    for tid in tids:
        if tid not in orig_affinity:
            orig_affinity.update(_affinity_snapshot([tid]))
    return dataclasses.replace(info, tids=tids, orig_affinity=orig_affinity)
