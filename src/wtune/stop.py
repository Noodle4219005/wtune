"""`wtune stop`: stop a running `wtune tune` or `wtune run` the way Ctrl-C does -- it cancels its
own jobs and writes its report (a tune from what it measured, a run with status stopped) -- or,
when that process is gone (killed, or quit with a second Ctrl-C), cancel the jobs it left in
Slurm. Finds them by tune/<id>/tune.state.json and runs/<id>/run.state.json (pid and job ids),
which they keep while they run."""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal

STATES = ("tune/*/tune.state.json", "runs/*/run.state.json")
REPORT = dict(tune="tune_report.json", run="run_report.json")


def _is_wtune(pid: int, kind: str) -> bool:
    """pid is a live `wtune <kind>` (its argv), not a reused pid."""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            argv = fh.read().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(b"wtune") for a in argv) and kind.encode() in argv


def stop(state_files: list, dry_run: bool = False, sacct_runner=None, scancel_runner=None,
         kill=os.kill) -> list:
    """One line per state file: what was (or, with dry_run, would be) done."""
    from wtune.tools import scc_sweep as sweep
    from wtune.tune.context import default_scancel_runner
    lines = []
    for path in state_files:
        with open(path) as fh:
            state = json.load(fh)
        d, kind = os.path.dirname(path), state.get("kind", "tune")
        pid, jobs = state["pid"], [str(j) for j in state.get("jobs", [])]
        if _is_wtune(pid, kind):
            if not dry_run:
                kill(pid, signal.SIGTERM)
            lines.append("%s: %s %s pid %d to stop; it cancels its jobs and writes %s at its "
                         "next poll" % (d, "would ask" if dry_run else "asked", kind, pid,
                                        REPORT.get(kind, "its report")))
            continue
        states = sweep.sacct_states(jobs, **({"sacct_runner": sacct_runner} if sacct_runner
                                             else {}))
        left = [j for j in jobs if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        if not dry_run:
            if left:
                (scancel_runner or default_scancel_runner)(left)
            os.remove(path)
        lines.append("%s: %s pid %d is gone; %s" % (
            d, kind, pid, ("%s job(s) %s" % ("would cancel" if dry_run else "cancelled", ",".join(left))
                     if left else "no job of it left in Slurm")))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune stop", description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*", help="tune/<id>/ or runs/<id>/ directories; default: every "
                    "one of the project with a state file")
    ap.add_argument("--project", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what would be done")
    args = ap.parse_args(argv)
    if args.dirs:
        files = [p for d in args.dirs for p in glob.glob(os.path.join(d, "*.state.json"))]
        missing = [d for d in args.dirs if not glob.glob(os.path.join(d, "*.state.json"))]
        if missing:
            raise SystemExit("wtune stop: no tune.state.json or run.state.json in %s"
                             % ", ".join(missing))
    else:
        import wtune.project
        project_dir = wtune.project.find_project(explicit=args.project)
        if not project_dir:
            raise SystemExit("wtune stop: pass a tune/<id>/ or runs/<id>/ directory or run it "
                             "inside a project")
        files = sorted(p for pat in STATES for p in glob.glob(os.path.join(project_dir, pat)))
        if not files:
            print("wtune stop: no running or interrupted tune or run in %s" % project_dir)
            return 0
    for line in stop(files, dry_run=args.dry_run):
        print(line)
    return 0
