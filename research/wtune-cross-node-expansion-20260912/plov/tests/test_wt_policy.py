import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class PolicyCTest(unittest.TestCase):
    def test_portable_policy_core(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "wt_policy_test"
            subprocess.run(
                shlex.split(os.environ.get("CC", "gcc")) + [
                    "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I", str(ROOT / "plov"),
                    str(ROOT / "plov" / "wt_policy.c"),
                    str(pathlib.Path(__file__).with_name("test_wt_policy.c")),
                    "-lm", "-o", str(executable),
                ],
                check=True,
                cwd=ROOT,
            )
            subprocess.run([str(executable)], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
