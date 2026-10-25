"""Node health pre-flight (all modes): sinfo states plus one srun probe per node, so nodes that
lost /nfs/home while sinfo still said idle (scc 2026-10-01: work2/work10, jobs died in 0-1 s)
are excluded from every sbatch of this process."""
from __future__ import annotations

import concurrent.futures
import getpass
import os
import shlex
import subprocess
import time
from typing import Any, Callable

import wtune.machine
from ._deps import sweep

PREFLIGHT_TIMEOUT_S = 45.0             # s per srun probe
IMMEDIATE_S = 30                       # srun --immediate: give up an allocation not granted by then
# srun lines of an allocation the scheduler refused or did not grant in time: a statement about
# the queue (a reservation, a per-user submit limit), not about the node
SCHEDULER_REFUSAL = ("Unable to allocate resources", "Immediate execution impossible",
                     "QOSMax", "AssocMax", "violates accounting/QOS policy", "Job submit/allocate failed")
SINFO_TIMEOUT_S = 30                   # s for the sinfo call
BUSY_STATES = ("alloc", "mix")         # sinfo %T of a node running someone's job
HEALTH_BAD_STATES = ("down", "drain", "fail", "maint", "unk", "no_respond", "inval", "powered")

Runner = Callable[[list, float], tuple]


def default_preflight_runner(argv: list, timeout_s: float) -> tuple[int, str]:
    """(returncode, stdout+stderr); raises subprocess.TimeoutExpired / OSError."""
    out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return out.returncode, ((out.stdout or "") + (out.stderr or "")).strip()


def _probe_script(case_dir: str, src_dir: str, wrf_exe: str) -> str:
    q = shlex.quote
    return ('[ -d "$HOME" ] || { echo "no $HOME"; exit 3; }; '
            '[ -r %s ] || { echo "case dir unreadable: "%s; exit 3; }; '
            '[ -r %s ] || { echo "wtune-src unreadable: "%s; exit 3; }; '
            '[ -x %s ] || { echo "wrf.exe not executable: "%s; exit 3; }; echo ok'
            % (q(case_dir), q(case_dir), q(src_dir), q(src_dir), q(wrf_exe), q(wrf_exe)))


def _bad_state(state: str) -> bool:
    return any(b in state.lower() for b in HEALTH_BAD_STATES) or state.endswith("*")


def _sinfo_states(text: str) -> dict[str, str]:
    """{node: state} of `sinfo -h -N -o "%n %T"` output, head excluded; a node listed in several
    partitions keeps its first bad state."""
    states: dict[str, str] = {}
    login = wtune.machine.login_nodes()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in login:
            prev = states.get(parts[0])
            if prev is None or (_bad_state(parts[1]) and not _bad_state(prev)):
                states[parts[0]] = parts[1]
    return states


def _probe_node(runner: Runner, n: str, state: str, script: str,
                timeout_s: float) -> tuple[str, dict]:
    """(node, {state, ok, probed, reason}) of one srun probe. Kept, unprobed: a node the
    scheduler did not allocate within IMMEDIATE_S or refused to (reserved, busy, a per-user
    limit -- scc 2026-10-13 excluded such nodes as unhealthy), and a timeout on a node other
    jobs are using. A probe that timed out has its job cancelled by name: killing srun left
    it PENDING (scc job 4921)."""
    login = wtune.machine.login_nodes()
    name = "wtune-preflight-%d-%s" % (os.getpid(), n)
    argv = (["srun", "-p", wtune.machine.partition(), "-w", n, "-N1", "-n1", "-t", "1",
             "--immediate=%d" % IMMEDIATE_S, "--job-name=" + name]
            + [a for a in ("--exclude=" + ",".join(login) if login else "",
                           wtune.machine.account_arg()) if a] + ["bash", "-c", script])
    try:
        rc, out = runner(argv, timeout_s)
    except subprocess.TimeoutExpired:
        try:
            runner(["scancel", "--user=" + getpass.getuser(), "--name=" + name], SINFO_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            pass
        if state.lower().startswith(BUSY_STATES):
            return n, dict(state=state, ok=True, probed=False,
                           reason="in use by another job: not probed (timed out waiting)")
        return n, dict(state=state, ok=False, probed=True,
                       reason="probe timed out after %gs" % timeout_s)
    except (OSError, subprocess.SubprocessError) as e:
        return n, dict(state=state, ok=False, probed=True, reason="probe error: %s" % e)
    if rc == 0:
        return n, dict(state=state, ok=True, probed=True, reason="")
    refusal = [ln for ln in (out or "").splitlines() if any(r in ln for r in SCHEDULER_REFUSAL)
               and "Invalid" not in ln]     # Invalid account/partition/qos: a configuration error
    if refusal:
        return n, dict(state=state, ok=True, probed=False,
                       reason="not allocated, not probed: %s" % refusal[-1].strip())
    return n, dict(state=state, ok=False, probed=True,
                   reason="probe failed rc=%d: %s" % (rc, (out or "").splitlines()[-1]
                                                       if out else "no output"))


def node_preflight(runner: Runner, *, case_dir: str, src_dir: str, wrf_exe: str, requested: int,
                   timeout_s: float = PREFLIGHT_TIMEOUT_S, when: str = "start") -> dict:
    """The node_health dict: sinfo states (head and bad states unusable), then one parallel srun
    probe per other node ($HOME, case dir, wtune-src, wrf.exe). Failures are published in
    WTUNE_PREFLIGHT_EXCLUDE for this process's sbatch commands; max_nodes_effective =
    min(requested, healthy). Skipped, and says so, when sinfo is unavailable."""
    t0 = time.monotonic()
    health: dict[str, Any] = dict(when=when, skipped=None, nodes={}, failed=[], user_exclude=[], exclusions=[],
                  healthy=None, max_nodes_requested=int(requested),
                  max_nodes_effective=int(requested), probe_s=0.0)
    user = [n.strip() for n in os.environ.get("WTUNE_SBATCH_EXCLUDE", "").split(",") if n.strip()]
    try:
        rc, text = runner(["sinfo", "-h", "-N", "-p", wtune.machine.partition(), "-o", "%n %T"],
                          SINFO_TIMEOUT_S)
        if rc != 0:
            raise OSError("sinfo rc=%d: %s" % (rc, text))
    except (OSError, subprocess.SubprocessError) as e:
        health["skipped"] = "sinfo unavailable (%s: %s)" % (type(e).__name__, e)
        health["exclusions"] = sweep.excluded_nodes()
        return health
    states = _sinfo_states(text)
    to_probe = []
    for n, st in sorted(states.items()):
        if n in user:
            health["nodes"][n] = dict(state=st, ok=False, probed=False,
                                      reason="excluded by WTUNE_SBATCH_EXCLUDE")
        elif _bad_state(st):
            health["nodes"][n] = dict(state=st, ok=False, probed=False,
                                      reason="sinfo state %s" % st)
        else:
            to_probe.append(n)
    script = _probe_script(case_dir, src_dir, wrf_exe)
    if to_probe:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_probe)) as pool:
            for n, res in pool.map(lambda n: _probe_node(runner, n, states[n], script, timeout_s),
                                   to_probe):
                health["nodes"][n] = res
    health["failed"] = sorted(n for n, r in health["nodes"].items() if not r["ok"])
    healthy = sorted(n for n, r in health["nodes"].items() if r["ok"])
    health.update(user_exclude=user, healthy=len(healthy),
                  healthy_nodes=healthy)
    os.environ[sweep.PREFLIGHT_EXCLUDE_ENV] = ",".join(health["failed"])
    health["exclusions"] = sweep.excluded_nodes()
    if healthy:
        health["max_nodes_effective"] = min(int(requested), len(healthy))
    else:
        health["warning"] = "no healthy node found; max_nodes left unchanged"
    health["probe_s"] = time.monotonic() - t0
    print("node pre-flight (%s): %d healthy of %d node(s); excluded %s; max_nodes %d -> %d (%.1fs, "
          "controller time)" % (when, len(healthy), len(states), ",".join(health["failed"]) or "-",
                                 int(requested), health["max_nodes_effective"],
                                 health["probe_s"]))
    return health


def render_node_health_text(h: dict) -> str:
    """The node-health section of the report (latest check plus every re-check)."""
    lines = ["", "--- node health pre-flight (latest check: %s) ---" % h.get("when", "start")]
    if h.get("skipped"):
        lines.append("skipped: %s" % h["skipped"])
        return "\n".join(lines)
    for n, r in sorted(h["nodes"].items()):
        lines.append("%-8s %-12s %-5s %s" % (n, r["state"], "ok" if r["ok"] else "FAIL",
                                              r["reason"]))
    lines.append("exclusions: %s; healthy %d; max_nodes %d -> %d; probe %.1fs (controller time, "
                 "not budget)" % (",".join(h["exclusions"]), h["healthy"], h["max_nodes_requested"],
                                  h["max_nodes_effective"], h["probe_s"]))
    for c in h.get("checks") or []:
        lines.append("  check %-14s healthy %s, failed %s, max_nodes -> %s, probe %.1fs"
                     % (c.get("when"), c.get("healthy"), ",".join(c.get("failed") or []) or "-",
                        c.get("max_nodes_effective"), c.get("probe_s") or 0.0))
    return "\n".join(lines)
