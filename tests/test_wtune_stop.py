"""wtune stop: a live tune is asked to stop (SIGTERM, as Ctrl-C); after a dead one only its own
jobs still in Slurm are cancelled."""
import json
import signal
import subprocess
import sys

import wtune.stop


def _state(d, pid, jobs):
    d.mkdir()
    (d / "tune.state.json").write_text(json.dumps(dict(pid=pid, jobs=jobs)))
    return str(d)


def test_a_live_tune_gets_sigterm_and_a_dead_ones_leftover_jobs_are_cancelled(tmp_path):
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "x/wtune", "tune"])
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    live_dir = _state(tmp_path / "t1", live.pid, ["5"])
    dead_dir = _state(tmp_path / "t2", gone.pid, ["11", "12"])
    sacct = lambda cmd: "11|RUNNING|00:00:10\n12|COMPLETED|00:01:00"
    cancelled = []

    lines = wtune.stop.stop([dead_dir], dry_run=True, sacct_runner=sacct,
                            scancel_runner=cancelled.append)
    assert "would cancel job(s) 11" in lines[0] and not cancelled
    assert (tmp_path / "t2" / "tune.state.json").exists()

    lines = wtune.stop.stop([live_dir, dead_dir], sacct_runner=sacct,
                            scancel_runner=cancelled.append)
    assert live.wait(timeout=10) == -signal.SIGTERM and "asked tune pid" in lines[0]
    assert cancelled == [["11"]] and "cancelled job(s) 11" in lines[1]
    assert not (tmp_path / "t2" / "tune.state.json").exists()
    assert (tmp_path / "t1" / "tune.state.json").exists()     # the live tune removes its own


def test_first_ctrl_c_asks_to_stop_and_the_second_quits_keeping_the_state(tmp_path, monkeypatch):
    import os
    import types

    import pytest
    from wtune.tune import cli, context
    monkeypatch.setitem(context.STOP, "reason", None)
    ctx = types.SimpleNamespace(poll_interval=20.0, state_file=str(tmp_path / "tune.state.json"))
    with cli._stop_on_signal(ctx):                          # a normal end removes the state
        assert json.loads((tmp_path / "tune.state.json").read_text())["pid"] == os.getpid()
    assert not (tmp_path / "tune.state.json").exists()
    with pytest.raises(KeyboardInterrupt):
        with cli._stop_on_signal(ctx):
            os.kill(os.getpid(), signal.SIGINT)
            assert context.STOP["reason"] == "stopped by Ctrl-C"
            os.kill(os.getpid(), signal.SIGINT)
    assert (tmp_path / "tune.state.json").exists()          # for `wtune stop` to clean up
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_ctrl_c_in_submit_lets_the_step_stop_itself_and_a_second_one_kills_it(monkeypatch):
    import pytest
    import wtune.submit
    calls = []

    class FakeStep:
        def __init__(self, interrupts):
            self.interrupts = interrupts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def wait(self):
            calls.append("wait")
            if self.interrupts:
                self.interrupts -= 1
                raise KeyboardInterrupt
            return 0

        def kill(self):
            calls.append("kill")

    for interrupts, want in ((1, ["wait", "wait"]), (2, ["wait", "wait", "kill"])):
        calls.clear()
        monkeypatch.setattr(wtune.submit.subprocess, "Popen",
                            lambda cmd, cwd, _n=interrupts: FakeStep(_n))
        with pytest.raises(KeyboardInterrupt):
            wtune.submit._call(["wtune", "tune"], ".")
        assert calls == want            # one Ctrl-C: the step finishes its own stop, unkilled
