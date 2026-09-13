import os
import pathlib
import shlex
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT.parent / "artifacts/20260913-node-local-physics/mynn"


class MynnNodeTest(unittest.TestCase):
    def test_node(self):
        ART.mkdir(parents=True, exist_ok=True)
        for sanitized in (False, True):
            tag = "node-sanitized" if sanitized else "node-strict"
            exe = ART / tag
            cmd = shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g",
            ]
            if sanitized:
                cmd += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                        "-fno-omit-frame-pointer"]
            cmd += ["-I", str(ROOT), str(ROOT / "wt_mynn_layout.c"),
                    str(ROOT / "wt_mynn_batch.c"), str(ROOT / "wt_mynn_node.c"),
                    str(ROOT / "wt_node_local.c"), str(ROOT / "tests/test_mynn_node.c"),
                    "-pthread", "-o", str(exe)]
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                       UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
            with (ART / (tag + ".txt")).open("w") as log:
                log.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=60, env=env)
                result = subprocess.run([str(exe)], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=60, env=env)
                log.write(result.stdout.decode())
                self.assertEqual(result.returncode, 0, result.stdout.decode())
                self.assertIn(b"PASS", result.stdout)
                self.assertNotIn(b"runtime error:", result.stdout)


if __name__ == "__main__":
    unittest.main()
