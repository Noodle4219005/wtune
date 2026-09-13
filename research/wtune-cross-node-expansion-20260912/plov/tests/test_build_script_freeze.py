"""Run the real build script with synthetic inputs and recording tool mocks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "plov/build.sbatch"


class BuildScriptFreezeTests(unittest.TestCase):
    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_fixture(self, mode):
        # Prevent running the historical script, whose object path was hardcoded.
        script = SCRIPT.read_text()
        self.assertIn("WTIMER_OBJ_DIR", script)
        self.assertIn("prepare_native_link.py", script)
        with tempfile.TemporaryDirectory(prefix="wt-build-freeze-") as tmp:
            base = Path(tmp)
            wrf, src, obj, binary_dir, bin_dir = [base / name for name in
                                                 ("frozen", "source", "objects", "output", "tools")]
            for directory in (wrf / "main", src, binary_dir, bin_dir):
                directory.mkdir(parents=True)
            (wrf / "configure.wrf").write_text("OMP = -qopenmp\nLDFLAGS_LOCAL = -O3\n")
            for name in ("wrf.exe", "wrf.o", "module_wrf_top.o", "libwrflib.a"):
                (wrf / "main" / name).write_text("frozen fixture " + name)
            (wrf / "compile.log").write_text(
                "time mpiifx -o wrf.exe -qopenmp -O3 wrf.o module_wrf_top.o libwrflib.a -lm\n")
            for name in ("wtimer.S", "wtimer.c", "wtmpi.c", "wtlewi.c", "wt_call.S",
                         "wt_micro.c", "wt_rad.c", "wt_alloc.c", "fixture.h"):
                (src / name).write_text("source fixture\n")
            (src / "prepare_native_link.py").write_bytes((ROOT / "plov/prepare_native_link.py").read_bytes())
            record = base / "calls.jsonl"
            driver = "#!" + sys.executable + "\n" + r'''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["MOCK_RECORD"], "a") as f:
    f.write(json.dumps({"name": name, "argv": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
if name in ("icx", "mpiicx", "mpiifx"):
    if name == "mpiifx" and os.environ.get("MOCK_LINK_FAIL"):
        sys.exit(17)
    output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    output.write_text("mock object or executable\n")
elif name == "nm":
    prefix, count = ("MPI_", 7) if "wtmpi_c.o" in sys.argv[-1] else ("__wrap_", 6)
    for index in range(count):
        print("00000000 T " + prefix + str(index))
elif name == "objdump":
    print("call <__wrap_for_allocate_handle>\n" * 501)
elif name in ("make", "cp", "mv"):
    sys.exit(91)
'''
            for name in ("icx", "mpiicx", "mpiifx", "nm", "objdump", "make", "cp", "mv"):
                path = bin_dir / name
                path.write_text(driver)
                path.chmod(0o700)
            gate = src / "gate0.sh"
            gate.write_text("#!" + sys.executable + "\nimport json,os,sys\n"
                            "with open(os.environ['MOCK_RECORD'],'a') as f:\n"
                            " f.write(json.dumps({'name':'gate0','argv':sys.argv[1:]})+'\\n')\n")
            gate.chmod(0o700)
            output = binary_dir / "wrf_wtimer.exe"
            if mode == "existing":
                output.write_text("existing output")
            elif mode == "frozen":
                output = wrf / "main/new.exe"
            elif mode == "symlink":
                alias = base / "alias"
                alias.symlink_to(wrf / "main", target_is_directory=True)
                output = alias / "new.exe"
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in wrf.rglob("*") if p.is_file()}
            env = {"PATH": str(bin_dir) + ":/usr/bin:/bin", "TREE": str(wrf),
                   "SRC": str(src), "WTIMER_OBJ_DIR": str(obj),
                   "WTIMER_OUTPUT": str(output), "WTIMER_PYTHON": sys.executable,
                   "MOCK_RECORD": str(record), "SLURM_JOB_ID": "synthetic-no-job"}
            if mode == "link_fail":
                env["MOCK_LINK_FAIL"] = "1"
            result = subprocess.run(["bash", str(SCRIPT)], cwd=base, env=env,
                                    capture_output=True, text=True, timeout=60)
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in wrf.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
            self.assertFalse(any(call["name"] in ("make", "cp", "mv") for call in calls))
            if mode == "success":
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(output.is_file())
                self.assertTrue(Path(str(output) + ".build").is_file())
                manifest = json.loads(Path(str(output) + ".link.json").read_text())
                link_calls = [call for call in calls if call["name"] == "mpiifx"]
                self.assertEqual(len(link_calls), 1)
                self.assertEqual(link_calls[0]["argv"], manifest["argv"][1:])
                self.assertEqual(link_calls[0]["cwd"], str(binary_dir))
                self.assertEqual(manifest["argv"][-1], "-lrt")
                self.assertEqual(len(manifest["wrapper_objects"]), 8)
                self.assertEqual(len(manifest["wrap_symbols"]), 7)
                self.assertEqual([call["argv"] for call in calls if call["name"] == "gate0"], [[str(output)]])
                compile_calls = [call for call in calls if call["name"] in ("icx", "mpiicx")]
                self.assertEqual(len(compile_calls), 8)
                self.assertTrue(all(Path(call["argv"][-1]).is_relative_to(obj) for call in compile_calls))
                expected = [
                    ("icx", ["-O2", "-c", str(src / "wtimer.S"), "-o", str(obj / "wtimer_asm.o")]),
                    ("icx", ["-O2", "-qopenmp", "-c", str(src / "wtimer.c"), "-o", str(obj / "wtimer_c.o")]),
                    ("mpiicx", ["-O2", "-c", str(src / "wtmpi.c"), "-o", str(obj / "wtmpi_c.o")]),
                    ("icx", ["-O2", "-qopenmp", "-c", str(src / "wtlewi.c"), "-o", str(obj / "wtlewi_c.o")]),
                    ("icx", ["-O2", "-c", str(src / "wt_call.S"), "-o", str(obj / "wt_call.o")]),
                    ("icx", ["-O2", "-I" + str(src), "-c", str(src / "wt_micro.c"), "-o", str(obj / "wt_micro.o")]),
                    ("icx", ["-O2", "-I" + str(src), "-c", str(src / "wt_rad.c"), "-o", str(obj / "wt_rad.o")]),
                    ("icx", ["-O2", "-c", str(src / "wt_alloc.c"), "-o", str(obj / "wt_alloc.o")])]
                self.assertEqual([(call["name"], call["argv"]) for call in compile_calls], expected)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(call["name"] == "gate0" for call in calls))
                if mode == "existing":
                    self.assertEqual(output.read_text(), "existing output")
                else:
                    self.assertFalse(output.exists())
                if mode != "link_fail":
                    self.assertEqual(calls, [])

    def test_separate_output_link_and_original_frozen_bytes(self):
        self.run_fixture("success")

    def test_refuse_existing_frozen_or_symlink_output_before_compile(self):
        for mode in ("existing", "frozen", "symlink"):
            with self.subTest(mode=mode):
                self.run_fixture(mode)

    def test_failed_link_never_touches_frozen_executable_or_runs_gate(self):
        self.run_fixture("link_fail")


if __name__ == "__main__":
    unittest.main()
