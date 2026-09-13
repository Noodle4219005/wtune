import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("WTUNE_SFCLAY_SOURCE", "/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/phys/module_sf_sfclay.F"))
SOURCE_SHA256 = "dfd7aae5ef1f0bd979246480af5276a1ee956bc216ec8dc6e82eb963ded761dc"
GNU_SYMBOL = "__module_sf_sfclay_MOD_sfclay"


class SfclayFortranTests(unittest.TestCase):
    def test_frozen_sfclay_matches_transactional_wrap(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        fc = shlex.split(os.environ["FC"])
        cc = shlex.split(os.environ["CC"])
        with tempfile.TemporaryDirectory(prefix="wt-sfclay-fortran-") as tmp:
            tmp = Path(tmp)
            bridge = tmp / "wt_sfclay_bridge.c"
            made = subprocess.run([
                sys.executable, "plov/gen_sfclay_bridge.py",
                "--source", str(SOURCE), "--schema", "plov/sfclay_schema.json",
                "--symbol", GNU_SYMBOL, "--output", str(bridge),
            ], cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(made.returncode, 0, made.stdout + made.stderr)
            self.assertIn("void __wrap_" + GNU_SYMBOL, bridge.read_text())
            fobj, cobj, bobj, default_bobj, coreobj, exe = (tmp / "sfclay.o", tmp / "fixture.o",
                tmp / "bridge.o", tmp / "bridge-default.o", tmp / "core.o", tmp / "fixture")
            compiled = subprocess.run(fc + ["-cpp", "-DEM_CORE=1", "-ffree-form", "-fcheck=all", "-fsanitize=address,undefined",
                "-J", str(tmp), "-c", str(SOURCE), "-o", str(fobj)],
                cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            default = subprocess.run(cc + ["-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                "-Iplov", "-c", str(bridge), "-o", str(default_bobj)], cwd=ROOT,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
            symbols = subprocess.run(["nm", "-g", str(default_bobj)], text=True, capture_output=True, timeout=60)
            self.assertEqual(symbols.returncode, 0, symbols.stdout + symbols.stderr)
            for helper in ("wt_sfclay_bridge_real", "wt_sfclay_bridge_call", "wt_sfclay_bridge_wrapped_calls"):
                self.assertNotIn(helper, symbols.stdout)
            for source, output, extra in (("plov/tests/sfclay_fortran_fixture.c", cobj, []),
                                          (bridge, bobj, ["-DWT_SFCLAY_BRIDGE_TEST_HELPERS"]),
                                          ("plov/wt_sfclay.c", coreobj, [])):
                built = subprocess.run(cc + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-Iplov"] + extra + ["-c", str(source), "-o", str(output)],
                    cwd=ROOT, text=True, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            linked = subprocess.run(fc + ["-fsanitize=address,undefined", str(fobj), str(cobj), str(bobj), str(coreobj),
                "-Wl,--wrap=" + GNU_SYMBOL, "-o", str(exe)], cwd=ROOT,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            ran = subprocess.run([str(exe)], cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("SFCLAY Fortran transactional wrap checks passed", ran.stdout)


if __name__ == "__main__":
    unittest.main()
