"""Real frozen MYNN init + two continuation calls through a forked GNU helper."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from plov.tests.test_mynn_bridge_fortran import SOURCES, SCHEMA, GNU_SYMBOL, fixture_source

ROOT = Path(__file__).resolve().parents[2]
PLOV = ROOT / "plov"
ART = ROOT / "artifacts/20260913-node-local-physics/mynn/gnu-node"
BIN = Path("/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/.pixi/envs/default/bin")


class MynnNodeFortranTests(unittest.TestCase):
    def test_real_init_then_two_continuations_on_forked_helper(self):
        ART.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=ART))
        src, mod, obj = (work / part for part in ("src", "mod", "obj"))
        for directory in (src, mod, obj):
            directory.mkdir()
        env = dict(os.environ, TMPDIR=str(work), TMP=str(work), TEMP=str(work),
                   ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:disable_coredump=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        fc, cc = str(BIN / "x86_64-conda-linux-gnu-gfortran"), str(BIN / "x86_64-conda-linux-gnu-cc")
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        manifest = {"scope": "GNU fork after real initialization; no Intel/WRF attestation",
                    "work": str(work), "source_sha256": {}, "compilers": {"fc": fc, "cc": cc}}
        for name in ("test_mynn_node_fortran.py", "test_mynn_node_fortran.c"):
            original = PLOV / "tests" / name
            shutil.copyfile(original, src / name)
            manifest.setdefault("test_source_sha256", {})[name] = hashlib.sha256(original.read_bytes()).hexdigest()
        with (work / "commands.txt").open("w") as log:
            def run(command, timeout=180):
                log.write(shlex.join(command) + "\n"); log.flush()
                result = subprocess.run(command, cwd=work, env=env, text=True,
                                        capture_output=True, timeout=timeout)
                log.write(result.stdout + result.stderr + "RETURN %d\n" % result.returncode); log.flush()
                self.assertEqual(result.returncode, 0, str(work) + "\n" + result.stdout + result.stderr)
                return result

            san = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
            fflags = ["-cpp", "-ffree-form", "-O1", "-g", "-fcheck=all", *san,
                      "-J", str(mod), "-I", str(mod)]
            objects = []
            for original, expected in SOURCES:
                copied = src / original.name
                shutil.copyfile(original, copied)
                self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), expected)
                manifest["source_sha256"][original.name] = expected
                output = obj / (original.stem + ".o")
                run([fc, *fflags, "-c", str(copied), "-o", str(output)])
                objects.append(str(output))

            # Fingerprint actual compiler-materialized common PARAMETER bytes.
            # SAVE at common:23 covers only PARAMETERs. mym_initialize's five
            # implicit SAVE literals at core:1163-1164 are never assigned; the
            # sole explicit SAVE icall:5574 is unreferenced and mix_chem=false.
            common = (src / "module_bl_mynnedmf_common.f90").read_text()
            names = []
            for line in common.splitlines():
                if "parameter::" in line.lower().replace(" ", ""):
                    names.extend(re.findall(r"\b(\w+)\s*=", line.split("::", 1)[1]))
            self.assertGreater(len(names), 60)
            parameter = src / "parameter-bits.f90"
            parameter.write_text("subroutine mynn_parameter_bits(bits) bind(C)\n"
                                 "use iso_c_binding, only: c_int32_t\n"
                                 "use module_bl_mynnedmf_common\nimplicit none\n"
                                 "integer(c_int32_t), intent(out) :: bits(%d)\n" % len(names) +
                                 "\n".join("bits(%d)=transfer(%s,bits(1))" % (i+1, n) for i, n in enumerate(names)) +
                                 "\nend subroutine\n")
            parameter_obj = obj / "parameter-bits.o"
            run([fc, *fflags, "-c", str(parameter), "-o", str(parameter_obj)])
            objects.append(str(parameter_obj))
            manifest["state_identity"] = {
                "common_parameter_names": names,
                "method": "FNV64 of actual compiled PARAMETER bits after initflag1; child inherits same initialized process by fork",
                "readonly_save": "core:1163-1164 pmz=1 phh=1 flt=0 fltv=0 flq=0, no writes in mym_initialize",
                "unused_save": "core:5574 icall declaration is its sole occurrence; driver:210 mix_chem PARAMETER false",
                "scientific_arrays": "All132 argument backing state compared externally after both continuations"}
            manifest["nm_defined"] = run(["nm", "-S", "--defined-only", *objects]).stdout

            caller = fixture_source()
            call = re.search(r"  call mynnedmf_driver\(.*?\)\n", caller, re.S).group()
            self.assertEqual(caller.count(call), 2)
            caller = caller.replace(call + call, call + "  initflag = 0\n" + call + call)
            caller = caller.replace("  rublten_before = rublten\n", "")
            caller = caller.replace(call + "  initflag = 0\n", call + "  rublten_before = rublten\n  initflag = 0\n", 1)
            caller_path = src / "caller.f90"; caller_path.write_text(caller)
            caller_obj = obj / "caller.o"
            run([fc, *fflags, "-c", str(caller_path), "-o", str(caller_obj)])
            objects.append(str(caller_obj))
            direct = work / "direct"
            run([fc, *san, *objects, "-pthread", "-o", str(direct)])

            profile = hashlib.sha256(json.dumps(manifest["source_sha256"], sort_keys=True).encode() +
                                     (PLOV / "mynn_schema.json").read_bytes()).hexdigest()
            (work / "mynn-node-profile.h").write_text(
                '#define MYNN_PROFILE UINT64_C(0x%s)\n#define MYNN_PARAMETER_COUNT %d\n' % (profile[:16], len(names)) +
                '#define MYNN_ARTIFACT_DIRECTORY "%s"\n' % work)
            bridge = src / "bridge.c"
            run([sys.executable, str(PLOV / "gen_mynn_bridge.py"), "--source", str(src / "module_bl_mynnedmf_driver.f90"),
                 "--schema", str(PLOV / "mynn_schema.json"), "--symbol", GNU_SYMBOL, "--output", str(bridge)])
            node_objects = []
            for source in [bridge, src / "test_mynn_node_fortran.c", *[PLOV / (name + ".c") for name in
                           ("wt_mynn_layout", "wt_mynn_batch", "wt_mynn_node", "wt_node_local")]]:
                output = obj / (source.stem + ".o")
                run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g", *san,
                     "-I", str(PLOV), "-I", str(work), "-pthread", "-c", str(source), "-o", str(output)])
                node_objects.append(str(output))
            wrapped = work / "node"
            run([fc, *san, *objects, *node_objects, "-pthread", "-Wl,--wrap=" + GNU_SYMBOL, "-o", str(wrapped)])
            direct_state, node_state = work / "direct.bin", work / "node.bin"
            run([str(direct), str(direct_state)], 30)
            node_run = run([str(wrapped), str(node_state)], 30)
            self.assertEqual(direct_state.read_bytes(), node_state.read_bytes(), str(work))
            self.assertIn("MYNN_GNU_NODE calls=3 continuation=2 helper=4 full132=1", node_run.stderr)
            self.assertEqual(node_run.stderr.count("MYNN_GNU_HELPER"), 2)
            manifest.update(profile_sha256=profile, bytes=direct_state.stat().st_size,
                            state_sha256=hashlib.sha256(direct_state.read_bytes()).hexdigest(),
                            native_stderr=node_run.stderr, status="PASS")
            (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (ART / "latest.json").write_text(json.dumps({"work": str(work), "status": "PASS"}) + "\n")
            print("PASS MYNN true GNU node differential: " + str(work))


if __name__ == "__main__":
    unittest.main()
