"""post/recipe.py runs the independent figure sets in parallel processes, in task order."""
import os
import time

import wtune.post.recipe as recipe


def _set(name, delay):
    time.sleep(delay)
    return [recipe.fig(name, "%s.png" % name, "run0", extra={"pid": os.getpid()})]


def _dies(name):
    os._exit(3)


def test_sets_run_in_separate_processes_and_keep_task_order(monkeypatch):
    monkeypatch.setenv("WTUNE_POST_WORKERS", "3")
    tasks = [(n, _set, (n, d), {}) for n, d in (("common", 0.3), ("data", 0.0), ("tc", 0.1))]
    out = recipe._in_parallel(tasks)
    assert [e["id"] for e in out] == ["common", "data", "tc"]
    assert len({e["pid"] for e in out}) > 1
    assert os.getpid() not in {e["pid"] for e in out}


def test_one_worker_is_the_serial_path(monkeypatch):
    monkeypatch.setenv("WTUNE_POST_WORKERS", "1")
    out = recipe._in_parallel([("common", _set, ("common", 0.0), {})])
    assert out[0]["pid"] == os.getpid()


def test_a_dead_worker_becomes_missing_entries(monkeypatch):
    monkeypatch.setenv("WTUNE_POST_WORKERS", "2")
    out = recipe._in_parallel([("tc", _dies, ("tc",), {}), ("pbl", _set, ("pbl", 0.5), {})])
    assert {e["id"] for e in out} <= {"tc_figures", "pbl_figures", "pbl"}
    assert any(e["id"] == "tc_figures" and "worker process died" in str(e) for e in out)
