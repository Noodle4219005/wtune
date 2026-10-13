"""wtune/machine.py -- what is specific to one machine: its shape, scheduler names and paths.

The built-in values are scc's. They are replaced, in this order, by the user's own file
(user_file(), one per home directory, so one per machine), by a project's project.yaml
`machine:` block, and by the variables in ENV when set by hand. The dispatcher exports the two
files into the environment before a command runs; the getters read it at call time."""
from __future__ import annotations

import os

DEFAULTS = {"cores_per_node": 72,       # hardware cores of one compute node (scc GH200)
            "max_nodes": 13,            # compute nodes one job may use (scc work[1-13])
            "partition": "compute",     # Slurm partition
            "login_nodes": ("head",),   # nodes no job is sent to
            "account": "",              # Slurm account, for sites that bill by project
            "hardware_id": "scc-gh200",  # name of the node type in the measurement ledgers
            "numa_per_node": 1,         # NUMA domains with cores in one node
            "calibration": "",          # calibration JSON of wtune.predict; "" = scc's
            "site": "scc",              # campaign/sites/<site>.sh and cases.<site>.json
            "env_source": "$HOME/wtune-scc/env.sh",   # sourced by every job
            "wrf_tree": "$HOME/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF",
            "geog": "/nfs/data/wrf/WPS_GEOG/"}        # WPS static data
ENV = {"cores_per_node": "WTUNE_CORES_PER_NODE", "max_nodes": "WTUNE_MAX_NODES",
       "partition": "WTUNE_PARTITION", "login_nodes": "WTUNE_LOGIN_NODES",
       "account": "WTUNE_ACCOUNT", "hardware_id": "WTUNE_HARDWARE_ID",
       "numa_per_node": "WTUNE_NUMA_PER_NODE", "calibration": "WTUNE_CALIBRATION",
       "site": "WTUNE_SITE", "env_source": "WTUNE_ENV_SOURCE", "wrf_tree": "WRF_TREE",
       "geog": "WPS_GEOG"}


INT_KEYS = ("cores_per_node", "max_nodes", "numa_per_node")
PATH_KEYS = ("wrf_tree", "geog")        # read by python: ~ and $VARS are expanded on export


class MachineError(ValueError):
    pass


def check(block, where="machine") -> dict:
    """`block` (project.yaml's machine block, or the user file) as {key: value}, only the keys
    it sets. MachineError on an unknown key or a bad value."""
    block = {} if block is None else block
    if not isinstance(block, dict):
        raise MachineError("%s: must be a mapping" % where)
    out = {}
    for key, value in block.items():
        at = "%s.%s" % (where, key)
        if key not in DEFAULTS:
            raise MachineError("%s: unknown key (known: %s)" % (at, ", ".join(DEFAULTS)))
        if value is None:
            continue
        if key in INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MachineError("%s: %r must be a positive integer" % (at, value))
        elif key == "login_nodes":
            names = value.split(",") if isinstance(value, str) else value
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise MachineError("%s: %r must be a list of node names" % (at, value))
            value = [n.strip() for n in names if n.strip()]
        elif key == "calibration":
            value = os.path.expanduser(value) if isinstance(value, str) else value
            if not isinstance(value, str) or not os.path.isabs(value) or not os.path.isfile(value):
                raise MachineError("%s: %r must be the absolute path of an existing file"
                                   % (at, value))
        else:
            if not isinstance(value, str) or (key != "account" and not value.strip()):
                raise MachineError("%s: %r must be a name" % (at, value))
            value = value.strip()
        out[key] = value
    return out


def export(block: dict) -> None:
    """Put checked settings into the environment; a variable that is already set wins."""
    for key, value in block.items():
        name = ENV[key]
        if os.environ.get(name) or (key == "login_nodes" and name in os.environ):
            continue
        if key in PATH_KEYS:
            value = os.path.expandvars(os.path.expanduser(value))
        os.environ[name] = ",".join(value) if isinstance(value, list) else str(value)


def user_file() -> str:
    """The user's own settings: $WTUNE_MACHINE_FILE, else ~/.config/wtune/machine.yaml."""
    return os.environ.get("WTUNE_MACHINE_FILE") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "wtune",
        "machine.yaml")


def load_user() -> dict:
    """The checked settings of user_file(); {} when there is no such file."""
    path = user_file()
    if not os.path.isfile(path):
        return {}
    import yaml
    try:
        with open(path) as fh:
            block = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise MachineError("%s: %s" % (path, e)) from e
    return check(block, path)


def save_user(changes: dict) -> dict:
    """Merge `changes` into user_file(), creating it; returns the settings it now holds."""
    import yaml
    path = user_file()
    block = check(dict(load_user(), **changes), path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        yaml.safe_dump(block, fh, sort_keys=False, allow_unicode=True)
    os.replace(path + ".tmp", path)
    return block


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


def numa_per_node() -> int:
    return _positive_int("numa_per_node")


def _name(key: str) -> str:
    return os.environ.get(ENV[key], "").strip() or DEFAULTS[key]


def partition() -> str:
    return _name("partition")


def hardware_id() -> str:
    return _name("hardware_id")


def site() -> str:
    return _name("site")


def env_source() -> str:
    return _name("env_source")


def wrf_tree() -> str:
    return _name("wrf_tree")


def geog() -> str:
    return _name("geog")


def calibration() -> str:
    """Path of this machine's calibration file, or '' for the built-in one."""
    return _name("calibration")


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
