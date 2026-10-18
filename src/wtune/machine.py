"""wtune/machine.py -- what is specific to one machine: its shape, scheduler names and paths.

The built-in values are scc's. They are replaced, in this order, by the user's own file
(user_file(), one per home directory, so one per machine), by a project's project.yaml
`machine:` block, and by the variables in ENV when set by hand. The dispatcher exports the two
files into the environment before a command runs; the getters read it at call time."""
from __future__ import annotations

import json
import os

DEFAULTS = {"cores_per_node": 72,       # hardware cores of one compute node (scc GH200)
            "max_nodes": 13,            # compute nodes one job may use (scc work[1-13])
            "partition": "compute",     # Slurm partition, or a list tried in order
            "login_nodes": ("head",),   # nodes no job is sent to
            "account": "",              # Slurm account, for sites that bill by project
            "hardware_id": "scc-gh200",  # name of the node type in the measurement ledgers
            "numa_per_node": 1,         # NUMA domains with cores in one node
            "calibration": "",          # calibration JSON of wtune.predict; "" = scc's
            "site": "scc",              # campaign/sites/<site>.sh and cases.<site>.json
            "env_source": "$HOME/wtune-scc/env.sh",   # sourced by every job
            "wrf_tree": "$HOME/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF",
            "geog": "/nfs/data/wrf/WPS_GEOG/",        # WPS static data
            "max_hours": 0,             # longest job the partition accepts; 0 = no cap
            "max_omp": 0,               # threads per rank WRF runs with; 1 = no OpenMP, 0 = no cap
            "node_preflight": True,     # one srun per node before a tune (wtune.tune.preflight)
            "node_memory_gib": 0,       # memory of one compute node; 0 = unknown, no memory check
            "copy_probe_inputs": True,  # each probe job reads its own copy of the BP inputs
            "measured": {}}             # replacements for MEASURED below
ENV = {"cores_per_node": "WTUNE_CORES_PER_NODE", "max_nodes": "WTUNE_MAX_NODES",
       "partition": "WTUNE_PARTITION", "login_nodes": "WTUNE_LOGIN_NODES",
       "account": "WTUNE_ACCOUNT", "hardware_id": "WTUNE_HARDWARE_ID",
       "numa_per_node": "WTUNE_NUMA_PER_NODE", "calibration": "WTUNE_CALIBRATION",
       "site": "WTUNE_SITE", "env_source": "WTUNE_ENV_SOURCE", "wrf_tree": "WRF_TREE",
       "geog": "WPS_GEOG", "max_hours": "WTUNE_MAX_HOURS", "max_omp": "WTUNE_MAX_OMP",
       "node_preflight": "WTUNE_NODE_PREFLIGHT", "node_memory_gib": "WTUNE_NODE_MEMORY_GIB",
       "copy_probe_inputs": "WTUNE_COPY_PROBE_INPUTS",
       "measured": "WTUNE_MEASURED"}

# Numbers measured on scc that estimates fall back on when nothing was measured for the
# case at hand. Another machine replaces the ones it knows in its `measured:` map.
MEASURED = {
    "noise_frac": 0.03,                 # run-to-run repeatability: ties and early stop
    "anchor_drift_max": 0.01,           # drift allowed between the two ANCHOR runs of a job
    "startup_s": 120.0,                 # one run's startup, until round 0 measures it
    "s_per_step": 10.0,                 # s/step when no ledger row and no cell model applies
    "init_s": 120.0,                    # WRF init of a segment that has no prediction
    "restart_io_s": 300.0,              # one restart write or read (marshall)
    "history_frame_s": 30.0,            # one history frame of one domain
    "radiation_factor": 1.5,            # padding on the cell-step model for radiation steps
    "io_factor": 2.0,                   # --time covers this x the measured I/O (NFS: 2.4x spread)
    "rebuild_s": 180.0,                 # one ERA5 point rebuild (sc26p1: 1:21-3:00)
    "r_rad_default": 5.0,               # radiation step / normal step, no calibration
    "r_rad_day_night_ratio": 1.87,
    "ranking_r_rad": 3.93,              # the calibration's day level
    "probe_noise_measured_pct": 3.0,    # % of compute, by where s/step came from
    "probe_noise_extrapolated_pct": 15.0,
    "probe_noise_rsl_pct": 5.0,
    "probe_noise_fallback_pct": 50.0,
    "cold_start_band": 0.25,            # +-25% on a prediction with no rebuilt point
    "vec_kv": 50.0,                     # vectorisation gain of wide patches (wtune.cli)
    # I/O anchors, each at the grid wtune.io_model names next to it
    "restart_write_s_netcdf": 324.0, "restart_read_s_netcdf": 120.0,       # conus25
    "restart_write_s_adios2": 1.05, "restart_read_s_adios2": 2.3,
    "history_write_s_netcdf": 300.5,    # dorian2019, all frames of the run
    "history_write_s_adios2": 0.042,    # SC26-P2, one frame
    "boundary_s_netcdf": 0.31, "boundary_s_adios2": 0.0216,                # climate-ea50
    "aux4_s_netcdf": 260.0 / 1584, "aux4_s_adios2": 0.033,
    "iofields_call_ms": 0.9,
    "probe_restart_read_s_adios2": 30.0, "probe_restart_read_s_netcdf": 120.0,
    # memory per node = bytes per grid cell x cells + GiB per rank x ranks (isc21, f1-arm)
    "mem_bytes_per_cell": 1600.0, "mem_gib_per_rank": 0.94,
}


INT_KEYS = ("cores_per_node", "max_nodes", "numa_per_node")
_NUMBER_UNITS = {"max_hours": ("hours", "no cap"), "node_memory_gib": ("GiB", "unknown")}
_measured_cache = ("", {})
PATH_KEYS = ("wrf_tree", "geog", "env_source")   # ~ and $VARS are expanded on export (a job
                                                  # sources env_source from the variable)


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
        elif key in ("max_hours", "node_memory_gib"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise MachineError("%s: %r must be a number of %s, 0 for %s"
                                   % ((at, value) + _NUMBER_UNITS[key]))
        elif key in ("node_preflight", "copy_probe_inputs"):
            if not isinstance(value, bool):
                raise MachineError("%s: %r must be true or false" % (at, value))
        elif key == "max_omp":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MachineError("%s: %r must be a number of threads, 0 for no cap"
                                   % (at, value))
        elif key == "measured":
            if not isinstance(value, dict):
                raise MachineError("%s: must be a mapping of name: number" % at)
            for name, v in value.items():
                if name not in MEASURED:
                    raise MachineError("%s.%s: unknown name (known: %s)"
                                       % (at, name, ", ".join(MEASURED)))
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                    raise MachineError("%s.%s: %r must be a number >= 0" % (at, name, v))
            value = {name: float(v) for name, v in value.items()}
        elif key == "login_nodes":
            names = value.split(",") if isinstance(value, str) else value
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise MachineError("%s: %r must be a list of node names" % (at, value))
            value = [n.strip() for n in names if n.strip()]
        elif key == "partition" and not isinstance(value, str):
            if (not isinstance(value, list) or not value
                    or not all(isinstance(n, str) and n.strip() for n in value)):
                raise MachineError("%s: %r must be a name or a list of names" % (at, value))
            value = [n.strip() for n in value]
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
        if key == "measured":
            _export_measured(value)
            continue
        if os.environ.get(name) or (key == "login_nodes" and name in os.environ):
            continue
        if key in PATH_KEYS:
            value = os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, bool):
            value = "1" if value else "0"
        os.environ[name] = ",".join(value) if isinstance(value, list) else str(value)


def _export_measured(value: dict) -> None:
    """Name by name: what the environment already holds wins."""
    merged = dict(value, **json.loads(os.environ.get(ENV["measured"]) or "{}"))
    if merged:
        os.environ[ENV["measured"]] = json.dumps(merged, sort_keys=True)


def merged(old: dict, changes: dict) -> dict:
    """`changes` over `old`; the measured map is merged name by name."""
    out = dict(old or {}, **changes)
    if "measured" in changes and (old or {}).get("measured"):
        out["measured"] = dict(old["measured"], **changes["measured"])
    return out


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
    block = check(merged(load_user(), changes), path)
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


def partitions() -> list[str]:
    """The partitions a job may go to, in the order they are tried (wtune.tools.scc_sweep
    .submit_job moves on when sbatch refuses one)."""
    return [p.strip() for p in _name("partition").split(",") if p.strip()]


def partition() -> str:
    """The first of partitions(): the one of srun steps and of written sbatch headers."""
    return partitions()[0]


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


def _non_negative(key: str) -> float:
    raw = os.environ.get(ENV[key], "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        value = -1.0
    if value < 0:
        raise MachineError("%s=%r: must be a number of %s" % (ENV[key], raw, _NUMBER_UNITS[key][0]))
    return value


def max_hours() -> float:
    """Longest job the partition accepts, in hours; 0 when no cap is set."""
    return _non_negative("max_hours")


def node_memory_gib() -> float:
    """Memory of one compute node in GiB; 0 when unknown (no memory check before probes)."""
    return _non_negative("node_memory_gib")


def node_preflight() -> bool:
    """False where a probe job on every node is not wanted: a cluster that limits how many
    jobs a user may have submitted rejects most of them, and a rejected probe reads as a bad
    node."""
    raw = os.environ.get(ENV["node_preflight"], "").strip().lower()
    if raw in ("", "1", "true", "on"):
        return True
    if raw in ("0", "false", "off"):
        return False
    raise MachineError("%s=%r: must be 1 or 0" % (ENV["node_preflight"], raw))


def copy_probe_inputs() -> bool:
    """False where the probes may read the shared BP input cache: a copy per job avoids the scc
    NFS open stall, but on a filesystem with a quota a wave of copies can fill it."""
    raw = os.environ.get(ENV["copy_probe_inputs"], "").strip().lower()
    if raw in ("", "1", "true", "on"):
        return True
    if raw in ("0", "false", "off"):
        return False
    raise MachineError("%s=%r: must be 1 or 0" % (ENV["copy_probe_inputs"], raw))


def max_omp() -> int:
    """Most OpenMP threads per rank this machine's WRF runs with (1: built without OpenMP);
    0 when no cap is set."""
    raw = os.environ.get(ENV["max_omp"], "").strip()
    try:
        value = int(raw) if raw else 0
    except ValueError:
        value = -1
    if value < 0:
        raise MachineError("%s=%r: must be a number of threads" % (ENV["max_omp"], raw))
    return value


def omp_cap(case_cap=None):
    """The lower of a case's own thread cap (None: none) and max_omp(); None for no cap."""
    caps = [c for c in (case_cap, max_omp()) if c]
    return min(caps) if caps else None


def cap_seconds(seconds: float) -> float:
    """`seconds`, not above max_hours() when one is set."""
    cap = max_hours() * 3600.0
    return min(seconds, cap) if cap else seconds


def measured(name: str) -> float:
    """MEASURED[name], or this machine's own value for it."""
    global _measured_cache
    raw = os.environ.get(ENV["measured"], "")
    if raw != _measured_cache[0]:
        try:
            _measured_cache = (raw, check({"measured": json.loads(raw or "{}")},
                                          ENV["measured"])["measured"])
        except ValueError as e:
            raise MachineError("%s: %s" % (ENV["measured"], e)) from e
    return _measured_cache[1].get(name, MEASURED[name])


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
