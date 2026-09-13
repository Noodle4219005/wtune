import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class RemoteCTest(unittest.TestCase):
    def test_local_process_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "wt_remote_test"
            sources = [
                "wt_wire.c", "wt_stream.c", "wt_ledger.c", "wt_ledger_validate.c",
                "wt_policy.c", "wt_remote.c", "tests/test_wt_remote.c",
            ]
            subprocess.run(
                shlex.split(os.environ.get("CC", "gcc")) + [
                    "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I", str(ROOT / "plov"),
                ] + [str(ROOT / "plov" / source) for source in sources] + [
                    "-lm", "-o", str(executable),
                ],
                check=True, cwd=ROOT,
            )
            subprocess.run([str(executable)], check=True, cwd=ROOT, timeout=60)


if __name__ == "__main__":
    unittest.main()
