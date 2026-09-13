"""Frozen GNU SFCLAY through the production live bridge; mock grid metadata."""
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
GNU_INIT = "__module_sf_sfclay_MOD_sfclayinit"

class SfclayLiveFortranTests(unittest.TestCase):
    def test_frozen_kernel_through_production_live_dispatch(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        fc, cc = shlex.split(os.environ["FC"]), shlex.split(os.environ["CC"])
        def run(cmd):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 0, " ".join(map(str, cmd)) + "\n" + r.stdout + r.stderr)
            return r.stdout
        with tempfile.TemporaryDirectory(prefix="wt-sfclay-live-fortran-") as tmp:
            tmp = Path(tmp)
            bridge = tmp / "bridge.c"
            run([sys.executable, "plov/gen_sfclay_bridge.py", "--source", str(SOURCE),
                 "--schema", "plov/sfclay_schema.json", "--symbol", GNU_SYMBOL, "--output", str(bridge)])
            fobj = tmp / "sfclay.o"
            run(fc + ["-cpp", "-DEM_CORE=1", "-ffree-form", "-fcheck=all", "-g",
                      "-fsanitize=address,undefined", "-pthread", "-J", str(tmp), "-c", str(SOURCE), "-o", str(fobj)])
            sources = [("plov/tests/sfclay_live_fortran_fixture.c", []),
                       (str(bridge), ["-DWT_SFCLAY_BRIDGE_TEST_HELPERS"])]
            for module in ("wt_sfclay_live", "wt_sfclay_init_lease", "wt_sfclay_capacity", "wt_sfclay", "wt_live_spans"):
                sources.append(("plov/" + module + ".c", []))
            sources.append(("plov/wt_sfclay_dispatch.c", ["-DWT_SFCLAY_INIT_WRAP=__wrap_" + GNU_INIT,
                                                        "-DWT_SFCLAY_INIT_REAL=__real_" + GNU_INIT]))
            default = tmp / "dispatch-default.o"
            run(cc + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-Iplov",
                      "-c", "plov/wt_sfclay_dispatch.c", "-o", str(default)])
            symbols = run(["nm", "-g", str(default)])
            self.assertIn(" T __wrap_module_sf_sfclay_mp_sfclayinit_", symbols)
            self.assertIn(" U __real_module_sf_sfclay_mp_sfclayinit_", symbols)
            self.assertIn(" T wt_sfclay_bridge_dispatch", symbols)
            objects = [str(fobj)]
            for index, (source, extra) in enumerate(sources):
                output = tmp / (str(index) + ".o")
                run(cc + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-pthread",
                          "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Iplov"] + extra +
                         ["-c", source, "-o", str(output)])
                objects.append(str(output))
            exe = tmp / "fixture"
            run(fc + ["-fsanitize=address,undefined", "-pthread"] + objects +
                ["-Wl,--wrap=" + GNU_SYMBOL, "-Wl,--wrap=" + GNU_INIT,
                 "-Wl,--wrap=wt_sfclay_init_dispatch", "-o", str(exe)])
            output = run([str(exe)])
            self.assertIn("73 cases, 146 stages, 1 initializer, bitwise backing arrays PASS", output)

if __name__ == "__main__":
    unittest.main()
