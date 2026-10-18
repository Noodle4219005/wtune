"""`wtune stop`: stop a running `wtune tune` the way Ctrl-C does -- it cancels its own jobs and
writes tune_report.json from what was measured -- or, when that process is gone (killed, or quit
with a second Ctrl-C), cancel the jobs it left in Slurm. Finds tunes by tune/<id>/tune.state.json
(pid and job ids), which `wtune tune` keeps while it runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal

STATE = "tune.state.json"


def _is_tune(pid: int) -> bool:
    """pid is a live `wtune tune` (its argv), not a reused pid."""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            argv = fh.read().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(b"wtune") for a in argv) and b"tune" in argv


def stop(tune_dirs: list, dry_run: bool = False, sacct_runner=None, scancel_runner=None,
         kill=os.kill) -> list:
    """One line per tune dir with a state file: what was (or, with dry_run, would be) done."""
    from wtune.tools import scc_sweep as sweep
    from wtune.tune.context import default_scancel_runner
    lines = []
    for d in tune_dirs:
        path = os.path.join(d, STATE)
        with open(path) as fh:
            state = json.load(fh)
        pid, jobs = state["pid"], [str(j) for j in state.get("jobs", [])]
        if _is_tune(pid):
            if not dry_run:
                kill(pid, signal.SIGTERM)
            lines.append("%s: %s tune pid %d to stop; it cancels its jobs and writes "
                         "tune_report.json within one poll (~20 s)"
                         % (d, "would ask" if dry_run else "asked", pid))
            continue
        states = sweep.sacct_states(jobs, **({"sacct_runner": sacct_runner} if sacct_runner
                                             else {}))
        left = [j for j in jobs if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        if not dry_run:
            if left:
                (scancel_runner or default_scancel_runner)(left)
            os.remove(path)
        lines.append("%s: tune pid %d is gone; %s" % (
            d, pid, ("%s job(s) %s" % ("would cancel" if dry_run else "cancelled", ",".join(left))
                     if left else "no job of it left in Slurm")))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune stop", description=__doc__.split("\n\n")[0])
    ap.add_argument("tune_dirs", nargs="*", help="default: every tune/<id>/ of the project with a "
                    "%s" % STATE)
    ap.add_argument("--project", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what would be done")
    args = ap.parse_args(argv)
    dirs = args.tune_dirs
    if not dirs:
        import wtune.project
        project_dir = wtune.project.find_project(explicit=args.project)
        if not project_dir:
            raise SystemExit("wtune stop: pass a tune/<id>/ directory or run it inside a project")
        dirs = sorted(os.path.dirname(p) for p in
                      glob.glob(os.path.join(project_dir, "tune", "*", STATE)))
        if not dirs:
            print("wtune stop: no running or interrupted tune in %s/tune/" % project_dir)
            return 0
    missing = [d for d in dirs if not os.path.isfile(os.path.join(d, STATE))]
    if missing:
        raise SystemExit("wtune stop: no %s in %s" % (STATE, ", ".join(missing)))
    for line in stop(dirs, dry_run=args.dry_run):
        print(line)
    return 0
