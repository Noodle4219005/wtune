import datetime as dt
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "monitor_native_job.py"
ARTIFACTS = ROOT.parent / "artifacts/20260913-node-local-physics/monitor-native-job"


AT_MOCK = r'''#!/usr/bin/env python3
import os, pathlib, sys
root = pathlib.Path(os.environ["MOCK_ROOT"])
counter = root / "at-count"
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
(root / ("scheduled-%d.txt" % count)).write_text(sys.stdin.read())
(root / ("at-argv-%d.txt" % count)).write_text("\n".join(sys.argv[1:]))
if (root / "fail-at").exists():
    print("mock at unavailable", file=sys.stderr)
    sys.exit(7)
print("job %d at mock" % (9000 + count), file=sys.stderr)
'''

QUERY_MOCK = r'''#!/usr/bin/env python3
import os, pathlib, sys
root = pathlib.Path(os.environ["MOCK_ROOT"])
name = pathlib.Path(sys.argv[0]).name
with (root / "query-argv.txt").open("a") as stream:
    stream.write(name + "\t" + "\t".join(sys.argv[1:]) + "\n")
source = root / (name + ".out")
if source.exists():
    sys.stdout.write(source.read_text())
if (root / ("fail-" + name)).exists():
    print("mock query failure", file=sys.stderr)
    sys.exit(8)
'''


class NativeJobMonitorTest(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        base = pathlib.Path(tempfile.mkdtemp(prefix="case-", dir=str(ARTIFACTS)))
        self.work = base / "job artifacts 'quoted'"
        self.work.mkdir()
        self.bin = self.work / "bin"
        self.bin.mkdir()
        self.at = self.make_executable("at-mock", AT_MOCK)
        self.squeue = self.make_executable("squeue-mock", QUERY_MOCK)
        self.sacct = self.make_executable("sacct-mock", QUERY_MOCK)
        self.log = self.work / "native log.bin"
        self.log.write_bytes(bytes(range(251)) * 300)
        self.env = dict(os.environ, MOCK_ROOT=str(self.work))

    def make_executable(self, name, body):
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)
        return path.resolve()

    @property
    def state_path(self):
        return self.work / "monitor-state.json"

    def state(self):
        return json.loads(self.state_path.read_text())

    def replace_state(self, state):
        replacement = self.work / "fixture-state.new"
        replacement.write_text(json.dumps(state, indent=2) + "\n")
        replacement.replace(self.state_path)

    def command(self, *args, expected=0):
        result = subprocess.run(
            [sys.executable, str(TOOL)] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, env=self.env)
        self.assertEqual(result.returncode, expected,
                         result.stdout.decode() + result.stderr.decode())
        return result

    def start(self, job_id="123", expected=0):
        deadline = (dt.datetime.now(dt.timezone.utc) +
                    dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.command(
            "start", "--job-id", job_id,
            "--operation", "one exact native WRF validation run",
            "--artifact-dir", str(self.work.resolve()),
            "--native-log", str(self.log.resolve()),
            "--pass-criterion", "native exit zero and bitwise report passes",
            "--dependent-action", "review evidence, then decide whether to compare performance",
            "--time-critical-reason", "the short probe gates same-session artifact review",
            "--deadline", deadline,
            "--at", str(self.at), "--squeue", str(self.squeue),
            "--sacct", str(self.sacct), expected=expected)

    def callback(self, nonce=None, expected=0):
        if nonce is None:
            nonce = self.state()["scheduled_nonce"]
        return self.command("callback", "--artifact-dir", str(self.work.resolve()),
                            "--nonce", nonce, expected=expected)

    def set_scheduler(self, queue="", account=""):
        (self.work / "squeue-mock.out").write_text(queue)
        (self.work / "sacct-mock.out").write_text(account)

    def test_startup_cadence_exact_queries_and_tail(self):
        self.start()
        state = self.state()
        state["created_utc"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.replace_state(state)
        self.set_scheduler("123|PENDING|00:20|now|later|Resources\n", "")
        self.callback()
        state = self.state()
        self.assertEqual(state["startup_gate"]["running_observations"], 0)
        self.assertEqual(state["startup_gate"]["elapsed_seconds"], 0)
        self.assertFalse(state["startup_gate"]["complete"])

        self.set_scheduler("123|RUNNING|00:01|now|now|node\n", "")
        self.callback()
        state = self.state()
        self.assertEqual(state["startup_gate"]["running_observations"], 1)
        self.set_scheduler("123|PENDING|00:21|now|later|Requeued\n", "")
        self.callback()
        state = self.state()
        self.assertEqual(state["startup_gate"]["running_observations"], 0)
        self.assertIsNone(state["startup_gate"]["running_since_utc"])
        self.set_scheduler("123|RUNNING|00:01|now|now|node\n", "")
        self.callback()
        state = self.state()
        state["startup_gate"]["running_since_utc"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=180)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.replace_state(state)
        for number in range(2, 5):
            self.callback()
            state = self.state()
            self.assertEqual(state["startup_gate"]["running_observations"], number)
            observed = dt.datetime.strptime(
                state["last_native_observation"]["utc"], "%Y-%m-%dT%H:%M:%SZ")
            following = dt.datetime.strptime(state["next_utc"], "%Y-%m-%dT%H:%M:%SZ")
            self.assertEqual(int((following - observed).total_seconds()), 60)
        self.assertFalse(state["startup_gate"]["complete"])
        self.assertLess(state["startup_gate"]["elapsed_seconds"], 240)
        state["startup_gate"]["running_since_utc"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=240)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.replace_state(state)
        self.callback()
        state = self.state()
        self.assertTrue(state["startup_gate"]["complete"])
        observed = dt.datetime.strptime(
            state["last_native_observation"]["utc"], "%Y-%m-%dT%H:%M:%SZ")
        following = dt.datetime.strptime(state["next_utc"], "%Y-%m-%dT%H:%M:%SZ")
        self.assertEqual(int((following - observed).total_seconds()), 300)
        self.assertEqual((self.work / "at-count").read_text(), "9")
        queries = (self.work / "query-argv.txt").read_text()
        self.assertEqual(queries.count("squeue-mock\t-h\t-j\t123\t-o"), 8)
        self.assertEqual(queries.count("sacct-mock\t-n\t-X\t-j\t123\t--format="), 8)
        self.assertNotIn("\t-u\t", queries)
        tail = self.work / state["last_native_observation"]["native_log_tail_file"]
        self.assertEqual(tail.read_bytes(), self.log.read_bytes()[-65536:])
        scheduled = (self.work / "scheduled-1.txt").read_text()
        self.assertIn("'\"'\"'quoted'\"'\"'", scheduled)
        self.assertNotIn("one exact native WRF", scheduled)

    def test_missing_native_log_does_not_start_gate_or_create_log(self):
        self.log.unlink()
        self.start()
        self.set_scheduler("123|PENDING|00:01|now|later|Resources\n", "")
        self.callback()
        state = self.state()
        self.assertFalse(state["last_native_observation"]["native_log_exists"])
        self.assertEqual(state["startup_gate"]["running_observations"], 0)
        self.assertFalse(self.log.exists())
        missing = self.work / state["last_native_observation"]["native_log_tail_file"]
        self.assertIn(b"missing:", missing.read_bytes())

        self.set_scheduler("123|RUNNING|00:01|now|now|node\n", "")
        self.callback()
        state = self.state()
        self.assertEqual(state["startup_gate"]["running_observations"], 0)
        self.assertFalse(self.log.exists())

        self.log.write_bytes(b"real native output\n")
        self.callback()
        state = self.state()
        self.assertTrue(state["last_native_observation"]["native_log_exists"])
        self.assertEqual(state["startup_gate"]["running_observations"], 1)
        tail = self.work / state["last_native_observation"]["native_log_tail_file"]
        self.assertEqual(tail.read_bytes(), b"real native output\n")

    def test_lock_existing_state_and_stale_nonce_prevent_duplicates(self):
        self.start()
        first_nonce = self.state()["scheduled_nonce"]
        lock = (self.work / "monitor.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self.start(expected=2)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        self.start()
        self.assertEqual((self.work / "at-count").read_text(), "1")

        self.set_scheduler("123|PENDING|00:00|now|later|Resources\n", "")
        self.callback()
        self.assertEqual((self.work / "at-count").read_text(), "2")
        stale = self.callback(first_nonce)
        self.assertIn(b"stale callback ignored", stale.stdout)
        self.assertEqual((self.work / "at-count").read_text(), "2")

    def test_early_terminal_records_incomplete_gate_without_dependent_action(self):
        self.start()
        self.set_scheduler("", "123|COMPLETED|0:0|00:03|1|112\n")
        self.callback()
        state = self.state()
        self.assertEqual(state["status"], "terminal_observed")
        self.assertTrue(state["startup_gate"]["incomplete_at_terminal"])
        self.assertIsNone(state["next_utc"])
        self.assertIsNone(state["native_at_id"])
        self.assertEqual((self.work / "at-count").read_text(), "1")
        self.assertIn(state["dependent_action"], state["next_proposed_check"])
        self.assertIn("Verify", state["next_proposed_check"])

    def test_failed_at_is_blocked_with_next_check(self):
        (self.work / "fail-at").write_text("fail\n")
        self.start(expected=2)
        state = self.state()
        self.assertEqual(state["status"], "blocked")
        self.assertIn("at scheduling failed", state["blocked_reason"])
        self.assertIn("at service", state["next_proposed_check"])
        self.assertIsNone(state["native_at_id"])
        self.assertIsNone(state["next_utc"])

    def test_slurm_failure_is_observed_and_blocked_without_retry(self):
        self.start()
        self.set_scheduler("", "123|FAILED|1:0|00:02|1|112\n")
        self.callback()
        state = self.state()
        self.assertEqual(state["status"], "blocked")
        self.assertIn("terminal failure observed: FAILED", state["blocked_reason"])
        self.assertTrue(state["startup_gate"]["incomplete_at_terminal"])
        self.assertEqual((self.work / "at-count").read_text(), "1")
        self.assertIn("no automatic retry", state["next_proposed_check"])

    def test_deadline_stops_monitoring_without_cancelling_job(self):
        self.start()
        state = self.state()
        state["deadline_utc"] = "2000-01-01T00:00:00Z"
        self.replace_state(state)
        self.set_scheduler("123|RUNNING|00:05|now|now|node\n", "")
        self.callback()
        state = self.state()
        self.assertEqual(state["status"], "blocked")
        self.assertIn("not cancelled", state["blocked_reason"])
        self.assertEqual(state["observation_count"], 1)
        self.assertEqual((self.work / "at-count").read_text(), "1")
        queries = (self.work / "query-argv.txt").read_text().lower()
        self.assertNotIn("cancel", queries)
        self.assertNotIn("submit", queries)

    def test_invalid_job_id_never_schedules(self):
        result = self.start(job_id="123; squeue -u somebody", expected=2)
        self.assertIn(b"positive decimal", result.stderr)
        self.assertFalse((self.work / "at-count").exists())
        self.assertFalse(self.state_path.exists())


if __name__ == "__main__":
    unittest.main()
