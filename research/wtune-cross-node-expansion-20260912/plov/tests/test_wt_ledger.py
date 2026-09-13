import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LedgerCTest(unittest.TestCase):
    def test_ledger_contract(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "test_wt_ledger"
            command = (
                shlex.split(os.environ.get("CC", "cc"))
                + shlex.split(os.environ.get("CFLAGS", ""))
                + [
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-DWT_LEDGER_TESTING",
                    "-I",
                    str(ROOT),
                    str(ROOT / "wt_wire.c"),
                    str(ROOT / "wt_ledger.c"),
                    str(ROOT / "wt_ledger_validate.c"),
                    str(ROOT / "tests" / "test_wt_ledger.c"),
                    "-o",
                    str(exe),
                ]
            )
            subprocess.run(command, cwd=ROOT, check=True, timeout=60)
            subprocess.run([str(exe)], cwd=ROOT, check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
