#!/usr/bin/env python3
"""Persistently monitor one exact Slurm job with one-shot ``at`` callbacks.

This records scheduler and native-log evidence.  It never submits, retries,
cancels, or declares scientific verification.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import stat
import subprocess
import sys


STATE_NAME = "monitor-state.json"
LOCK_NAME = "monitor.lock"
CALLBACK_LOG = "monitor-callback.log"
TAIL_BYTES = 65536
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT",
    "TIMEOUT",
}
FAILURE = TERMINAL - {"COMPLETED"}


class MonitorError(Exception):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(value):
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value):
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise MonitorError("deadline must use YYYY-MM-DDTHH:MM:SSZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def checked_text(label, value):
    if not value or len(value) > 4096 or "\x00" in value:
        raise MonitorError("%s must be nonempty and at most 4096 characters" % label)
    return value


def checked_dir(value):
    path = Path(value)
    if not path.is_absolute():
        raise MonitorError("artifact directory must be absolute")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MonitorError("artifact directory is unavailable: %s" % exc)
    if not path.is_dir():
        raise MonitorError("artifact path is not a directory")
    return path


def checked_file(label, value, executable=False):
    path = Path(value)
    if not path.is_absolute():
        raise MonitorError("%s path must be absolute" % label)
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MonitorError("%s is unavailable: %s" % (label, exc))
    if not path.is_file() or (executable and not os.access(str(path), os.X_OK)):
        raise MonitorError("%s must be %s" %
                           (label, "an executable file" if executable else "a regular file"))
    return path


def checked_log_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise MonitorError("native log path must be absolute")
    try:
        resolved = path.resolve(strict=False)
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MonitorError("native log parent is unavailable: %s" % exc)
    if resolved != path or parent != path.parent or not parent.is_dir():
        raise MonitorError("native log path must be canonical with no symlink components")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise MonitorError("native log path is unavailable: %s" % exc)
    if not stat.S_ISREG(mode):
        raise MonitorError("native log must be a regular file when it exists")
    return path


class LockedState:
    def __init__(self, root):
        self.root = root
        self.stream = None

    def __enter__(self):
        self.stream = open(str(self.root / LOCK_NAME), "a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise MonitorError("another monitor callback holds the artifact lock")
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()

    def read(self, required=True):
        path = self.root / STATE_NAME
        if not path.exists():
            if required:
                raise MonitorError("monitor state does not exist")
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise MonitorError("monitor state is unreadable: %s" % exc)
        if not isinstance(value, dict) or value.get("version") != 1:
            raise MonitorError("monitor state schema is invalid")
        return value

    def write(self, state):
        path = self.root / STATE_NAME
        temporary = self.root / (STATE_NAME + ".new.%d.%s" %
                                 (os.getpid(), secrets.token_hex(4)))
        data = json.dumps(state, indent=2, sort_keys=True) + "\n"
        fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(path))
            directory = os.open(str(self.root), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()


def run_exact(argv):
    try:
        return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MonitorError("command failed to execute: %s: %s" % (argv[0], exc))


def callback_command(root, nonce):
    argv = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
            "callback", "--artifact-dir", str(root), "--nonce", nonce]
    command = " ".join(shlex.quote(part) for part in argv)
    return ("exec %s >> %s 2>&1\n" %
            (command, shlex.quote(str(root / CALLBACK_LOG))))


def schedule_at(state, root, when, locked):
    nonce = secrets.token_hex(16)
    state["status"] = "scheduling"
    state["scheduled_nonce"] = nonce
    state["next_utc"] = iso(when)
    state["native_at_id"] = None
    locked.write(state)
    timestamp = when.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M.%S")
    env = os.environ.copy()
    env["TZ"] = "UTC"
    try:
        result = subprocess.run([state["tools"]["at"], "-t", timestamp],
            input=callback_command(root, nonce).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MonitorError("at failed to execute: %s" % exc)
    output = (result.stdout + b"\n" + result.stderr).decode("utf-8", "replace")
    match = re.search(r"\bjob\s+([1-9][0-9]*)\b", output)
    if result.returncode or not match:
        raise MonitorError("at scheduling failed (exit %d): %s" %
                           (result.returncode, output.strip()[-1000:]))
    state["native_at_id"] = match.group(1)
    state["status"] = "scheduled"
    return nonce


def write_bytes(path, data):
    temporary = path.with_name(path.name + ".new.%d" % os.getpid())
    with open(str(temporary), "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def native_log_tail(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        raise MonitorError("native log is unreadable: %s" % exc)
    try:
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode):
            raise MonitorError("native log ceased to be a regular file")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            stream.seek(max(0, details.st_size - TAIL_BYTES))
            return details.st_size, stream.read(TAIL_BYTES)
    finally:
        if fd >= 0:
            os.close(fd)


def exact_squeue_state(job_id, raw):
    rows = [line for line in raw.decode("utf-8", "replace").splitlines() if line]
    matches = [line for line in rows if line.split("|", 1)[0] == job_id]
    if len(matches) > 1:
        raise MonitorError("squeue returned duplicate exact-job rows")
    if not matches:
        return None
    fields = matches[0].split("|")
    if len(fields) < 2 or not fields[1].strip():
        raise MonitorError("squeue returned a malformed exact-job row")
    return fields[1].strip().upper()


def exact_sacct_state(job_id, raw):
    rows = [line for line in raw.decode("utf-8", "replace").splitlines() if line]
    matches = [line for line in rows if line.split("|", 1)[0] == job_id]
    if len(matches) > 1:
        raise MonitorError("sacct returned duplicate exact-job rows")
    if not matches:
        return None
    fields = matches[0].split("|")
    if len(fields) < 2 or not fields[1].strip():
        raise MonitorError("sacct returned a malformed exact-job row")
    return fields[1].strip().upper().split()[0].rstrip("+")


def collect(state, root, now):
    job = state["job_id"]
    sequence = state["observation_count"] + 1
    prefix = root / ("observation-%04d" % sequence)
    squeue_argv = [state["tools"]["squeue"], "-h", "-j", job, "-o",
                   "%i|%T|%M|%V|%S|%R"]
    sacct_argv = [state["tools"]["sacct"], "-n", "-X", "-j", job,
                  "--format=JobIDRaw,State,ExitCode,Elapsed,NNodes,NCPUS",
                  "--parsable2"]
    squeue = run_exact(squeue_argv)
    sacct = run_exact(sacct_argv)
    write_bytes(prefix.with_name(prefix.name + "-squeue.txt"), squeue.stdout + squeue.stderr)
    write_bytes(prefix.with_name(prefix.name + "-sacct.txt"), sacct.stdout + sacct.stderr)
    size, tail = native_log_tail(state["native_log"])
    if tail is None:
        tail_name = prefix.name + "-native-log-missing.txt"
        write_bytes(prefix.with_name(tail_name),
                    ("missing: %s\n" % state["native_log"]).encode("utf-8"))
    else:
        tail_name = prefix.name + "-native-tail.bin"
        write_bytes(prefix.with_name(tail_name), tail)
    if squeue.returncode or sacct.returncode:
        raise MonitorError("exact-job query failed: squeue=%d sacct=%d" %
                           (squeue.returncode, sacct.returncode))
    queue_state = exact_squeue_state(job, squeue.stdout)
    account_state = exact_sacct_state(job, sacct.stdout)
    observed = queue_state or account_state or "UNKNOWN"
    state["observation_count"] = sequence
    state["last_native_observation"] = {
        "utc": iso(now), "slurm_state": observed,
        "squeue_file": prefix.name + "-squeue.txt",
        "sacct_file": prefix.name + "-sacct.txt",
        "native_log_tail_file": tail_name,
        "native_log_exists": tail is not None, "native_log_size": size,
    }
    gate = state["startup_gate"]
    if observed == "RUNNING" and tail is not None:
        if gate["running_since_utc"] is None:
            gate["running_since_utc"] = iso(now)
            gate["running_observations"] = 1
        else:
            gate["running_observations"] += 1
        elapsed = max(0, int((now - parse_utc(gate["running_since_utc"])).total_seconds()))
        gate["elapsed_seconds"] = elapsed
        if gate["running_observations"] >= 4 and elapsed >= 240:
            gate["complete"] = True
    elif not gate["complete"]:
        gate["running_since_utc"] = None
        gate["running_observations"] = 0
        gate["elapsed_seconds"] = 0
    return observed


def blocked(state, reason, next_check):
    state["status"] = "blocked"
    state["blocked_reason"] = reason
    state["next_proposed_check"] = next_check
    state["scheduled_nonce"] = None
    state["native_at_id"] = None
    state["next_utc"] = None


def start(args):
    if not re.fullmatch(r"[1-9][0-9]*", args.job_id):
        raise MonitorError("job id must be a positive decimal Slurm job id")
    root = checked_dir(args.artifact_dir)
    native_log = checked_log_path(args.native_log)
    tools = {name: str(checked_file(name, getattr(args, name), executable=True))
             for name in ("at", "squeue", "sacct")}
    deadline = parse_utc(args.deadline)
    now = utc_now()
    if deadline <= now:
        raise MonitorError("deadline must be in the future")
    with LockedState(root) as locked:
        existing = locked.read(required=False)
        if existing:
            print("monitor already exists; no duplicate callback scheduled")
            return 0
        state = {
            "version": 1, "status": "new", "job_id": args.job_id,
            "operation": checked_text("operation", args.operation),
            "artifact_dir": str(root), "native_log": str(native_log),
            "pass_criterion": checked_text("pass criterion", args.pass_criterion),
            "dependent_action": checked_text("dependent action", args.dependent_action),
            "cadence": {
                "startup_seconds": 60,
                "sparse_seconds": 300,
                "sparse_reason": checked_text("time-critical reason",
                                               args.time_critical_reason),
            },
            "deadline_utc": iso(deadline), "created_utc": iso(now), "tools": tools,
            "observation_count": 0, "last_native_observation": None,
            "startup_gate": {"required_running_observations": 4,
                             "required_elapsed_seconds": 240,
                             "running_observations": 0,
                             "running_since_utc": None,
                             "elapsed_seconds": 0, "complete": False},
            "native_at_id": None, "next_utc": None, "scheduled_nonce": None,
            "next_proposed_check": "Collect exact job status and native log tail.",
        }
        try:
            first_check = min(now + dt.timedelta(seconds=60), deadline)
            schedule_at(state, root, first_check, locked)
        except MonitorError as exc:
            blocked(state, str(exc), "Check the recorded at error and at service availability; do not submit or retry the native job.")
            locked.write(state)
            raise
        locked.write(state)
    print("scheduled monitor at-id=%s next=%s" %
          (state["native_at_id"], state["next_utc"]))
    return 0


def callback(args):
    root = checked_dir(args.artifact_dir)
    if not re.fullmatch(r"[0-9a-f]{32}", args.nonce):
        raise MonitorError("callback nonce is invalid")
    with LockedState(root) as locked:
        state = locked.read()
        if not secrets.compare_digest(str(args.nonce), str(state.get("scheduled_nonce"))):
            print("stale callback ignored")
            return 0
        if state.get("status") != "scheduled":
            print("inactive callback ignored")
            return 0
        state["status"] = "observing"
        state["scheduled_nonce"] = None
        state["native_at_id"] = None
        state["next_utc"] = None
        locked.write(state)
        now = utc_now()
        try:
            observed = collect(state, root, now)
            deadline = parse_utc(state["deadline_utc"])
            if observed in TERMINAL:
                state["startup_gate"]["incomplete_at_terminal"] = not state["startup_gate"]["complete"]
                if observed in FAILURE:
                    blocked(state, "Slurm terminal failure observed: %s" % observed,
                            "Inspect the recorded exact sacct row and native log tail; diagnose only, with no automatic retry.")
                else:
                    state["status"] = "terminal_observed"
                    state["next_proposed_check"] = (
                        "Verify the native artifact against this pass criterion: " +
                        state["pass_criterion"] + ". Only then decide whether to run: " +
                        state["dependent_action"])
            elif now >= deadline:
                blocked(state, "monitoring deadline reached; the native job was not cancelled",
                        "Request a new monitoring decision and query only job %s before any dependent action." % state["job_id"])
            else:
                delay = (state["cadence"]["sparse_seconds"]
                         if state["startup_gate"]["complete"]
                         else state["cadence"]["startup_seconds"])
                try:
                    next_check = min(now + dt.timedelta(seconds=delay), deadline)
                    schedule_at(state, root, next_check, locked)
                    state["next_proposed_check"] = "Collect exact job status and native log tail."
                except MonitorError as exc:
                    blocked(state, str(exc),
                            "Check the recorded at error and at service availability; do not submit, cancel, or retry the native job.")
            locked.write(state)
        except (MonitorError, OSError) as exc:
            blocked(state, str(exc),
                    "Inspect this callback's exact-job command artifacts and native log; do not retry or cancel automatically.")
            locked.write(state)
            raise
    print("monitor status=%s next=%s" % (state["status"], state["next_utc"]))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command")
    begin = commands.add_parser("start")
    begin.add_argument("--job-id", required=True)
    begin.add_argument("--operation", required=True)
    begin.add_argument("--artifact-dir", required=True)
    begin.add_argument("--native-log", required=True)
    begin.add_argument("--pass-criterion", required=True)
    begin.add_argument("--dependent-action", required=True)
    begin.add_argument("--time-critical-reason", required=True,
                       help="why the requested five-minute sparse cadence is necessary")
    begin.add_argument("--deadline", required=True)
    begin.add_argument("--at", required=True)
    begin.add_argument("--squeue", required=True)
    begin.add_argument("--sacct", required=True)
    follow = commands.add_parser("callback")
    follow.add_argument("--artifact-dir", required=True)
    follow.add_argument("--nonce", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command is None:
        parser().error("a command is required")
    try:
        return start(args) if args.command == "start" else callback(args)
    except MonitorError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
