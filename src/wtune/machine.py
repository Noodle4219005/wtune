"""wtune/machine.py -- the machine's shape and scheduler names.

The built-in values are scc's. A project's project.yaml `machine:` block replaces them
(wtune.project.apply_project_env exports it), and so do the WTUNE_* variables when set by hand.
Read at call time, because the project is applied after import."""
from __future__ import annotations

import os

DEFAULTS = {"cores_per_node": 72,       # hardware cores of one compute node (scc GH200)
            "max_nodes": 13,            # compute nodes one job may use (scc work[1-13])
            "partition": "compute",     # Slurm partition
            "login_nodes": ("head",),   # nodes no job is sent to
            "account": ""}              # Slurm account, for sites that bill by project
ENV = {"cores_per_node": "WTUNE_CORES_PER_NODE", "max_nodes": "WTUNE_MAX_NODES",
       "partition": "WTUNE_PARTITION", "login_nodes": "WTUNE_LOGIN_NODES",
       "account": "WTUNE_ACCOUNT"}


class MachineError(ValueError):
    pass


def _positive_int(key: str) -> int:
    raw = os.environ.get(ENV[key], "").strip()
    if not raw:
        return DEFAULTS[key]
    if not raw.isdigit() or int(raw) < 1:
        raise MachineError("%s=%r: must be a positive integer" % (ENV[key], raw))
    return int(raw)


def cores_per_node() -> int:
    return _positive_int("cores_per_node")


def max_nodes() -> int:
    return _positive_int("max_nodes")


def partition() -> str:
    return os.environ.get(ENV["partition"], "").strip() or DEFAULTS["partition"]


def account_arg() -> str:
    """'--account=<name>', or '' when none is set."""
    name = os.environ.get(ENV["account"], "").strip()
    return "--account=" + name if name else ""


def login_nodes() -> list[str]:
    """Set but empty means the machine has none."""
    raw = os.environ.get(ENV["login_nodes"])
    if raw is None:
        return list(DEFAULTS["login_nodes"])
    return [n.strip() for n in raw.split(",") if n.strip()]
