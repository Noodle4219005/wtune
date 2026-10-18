"""Ctrl-C and `wtune stop` for the commands that wait on Slurm jobs (tune, run). The first SIGINT or
SIGTERM sets STOP["reason"]; the command sees it at its next poll, cancels its own jobs and writes
its report. A second signal quits at once. <out>/<kind>.state.json (pid, job ids) lets `wtune stop`
find the process or, after it died, its jobs; a normal end removes it."""

from __future__ import annotations

import contextlib
import json
import os
import signal

STOP: dict = {"reason": None}          # set by the signal handler, read at every poll


def write_state(path: str, kind: str, jobs: list) -> None:
    """<kind>.state.json = {kind, pid, jobs}, replaced atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(kind=kind, pid=os.getpid(), jobs=list(jobs)), fh)
    os.replace(tmp, path)


@contextlib.contextmanager
def stop_on_signal(state_file: str, kind: str, poll_s: float):
    """Inside: SIGINT/SIGTERM ask `wtune <kind>` to stop at its next poll; the state file exists.
    A normal exit removes the state file; an exception (a second Ctrl-C) keeps it for
    `wtune stop`."""
    def handler(signum, _frame):
        if STOP["reason"]:
            raise KeyboardInterrupt
        STOP["reason"] = "stopped by %s" % ("Ctrl-C" if signum == signal.SIGINT else "wtune stop")
        print("\nwtune %s: %s -- at the next poll (every %.0f s) cancelling its jobs and writing "
              "its report; Ctrl-C again quits at once (its jobs then keep running: `wtune stop` "
              "cancels them)" % (kind, STOP["reason"], poll_s), flush=True)

    STOP["reason"] = None
    write_state(state_file, kind, [])
    try:
        old = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    except ValueError:                 # not the main thread (embedded use): no handler
        old = {}
    try:
        yield
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    os.remove(state_file)
