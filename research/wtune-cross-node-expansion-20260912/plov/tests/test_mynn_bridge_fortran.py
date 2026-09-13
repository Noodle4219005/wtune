"""GNU-only real-MYNN driver differential through the generated wrap bridge."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLOV = ROOT / "plov"
ARTIFACTS = ROOT / "artifacts/20260913-node-local-physics/mynn"
WRF = Path("/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF")
SCHEMA = json.loads((PLOV / "mynn_schema.json").read_text())
SOURCE_SHA = SCHEMA["source_sha256"]
SOURCES = (
    (WRF / "phys/ccpp_kind_types.f90", SOURCE_SHA["phys/ccpp_kind_types.f90"]),
    (WRF / "share/module_model_constants.f90", "c387a51c2992da393551ac63291c79afbf6b78efbca2045a18cf9c94381f9954"),
    (WRF / "phys/module_bl_mynnedmf_common.f90", "960a7370de0f33a85bc948eaf13e2a9811f6553450ac111e8f6839ea8f132cd8"),
    (WRF / "phys/module_bl_mynnedmf.f90", "a93d0c0e453738c16751525aec04403ec129b735b577e5f399b1edfd2129937b"),
    (WRF / "phys/module_bl_mynnedmf_driver.f90", SOURCE_SHA["phys/module_bl_mynnedmf_driver.f90"]),
)
GNU_SYMBOL = "__module_bl_mynnedmf_driver_MOD_mynnedmf_driver"
NARG = 132
GNU_ASAN_LIB = Path("/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/lib/gcc/x86_64-conda-linux-gnu/16.2.0")
GEOMETRY = {"ids", "ide", "jds", "jde", "kds", "kde", "ims", "ime", "jms", "jme",
            "kms", "kme", "its", "ite", "jts", "jte", "kts", "kte"}


def compiler(name, fallback):
    return shlex.split(os.environ.get(name, fallback))


def declaration(argument):
    base = {"real": "real(kind_phys)", "integer": "integer", "logical": "logical"}[argument["type"]]
    if argument["name"] in GEOMETRY:
        return "integer, parameter :: %s = %s" % (argument["name"], initial_value(argument))
    if argument["rank"] == 0:
        return "%s :: %s" % (base, argument["name"])
    dimensions = ",".join(argument["shape"])
    return "%s :: %s(%s)" % (base, argument["name"], dimensions)


def initial_value(argument):
    name = argument["name"]
    if argument["type"] == "logical":
        return ".false."
    if argument["type"] == "integer":
        values = {"ids": 1, "ide": 4, "jds": 1, "jde": 4, "kds": 1, "kde": 33,
                  "ims": 1, "ime": 4, "jms": 1, "jme": 4, "kms": 1, "kme": 33,
                  "its": 2, "ite": 3, "jts": 2, "jte": 3, "kts": 1, "kte": 32,
                  "initflag": 1, "tke_budget": 0, "bl_mynn_edmf": 0,
                  "bl_mynn_edmf_dd": 0, "bl_mynn_output": 0, "spp_pbl": 0}
        return str(values.get(name, 0))
    values = {"delt": "30.0_kind_phys", "dxc": "1000.0_kind_phys",
              "bl_mynn_closure": "2.5_kind_phys"}
    return values.get(name, "0.0_kind_phys")


def array_value(argument):
    name = argument["name"]
    values = {"xland": "1.0_kind_phys", "ps": "100000.0_kind_phys",
              "ts": "290.0_kind_phys", "qsfc": "0.010_kind_phys",
              "ust": "0.30_kind_phys", "ch": "0.010_kind_phys",
              "hfx": "30.0_kind_phys", "qfx": "0.0_kind_phys",
              "wspd": "5.0_kind_phys", "znt": "0.10_kind_phys",
              "dz": "100.0_kind_phys", "u": "5.0_kind_phys", "v": "1.0_kind_phys",
              "th": "300.0_kind_phys", "t3d": "290.0_kind_phys",
              "p": "90000.0_kind_phys", "exner": "0.97_kind_phys",
              "rho": "1.0_kind_phys", "qv": "0.010_kind_phys",
              "qke": "0.10_kind_phys", "pblh": "100.0_kind_phys",
              "kpbl": "1"}
    return values.get(name, "0.0_kind_phys")


def fixture_source():
    args = SCHEMA["arguments"]
    declarations = "\n  ".join(declaration(arg) for arg in args)
    scalars = "\n  ".join("%s = %s" % (arg["name"], initial_value(arg))
                           for arg in args if arg["rank"] == 0 and arg["name"] not in GEOMETRY)
    arrays = "\n  ".join("%s = %s" % (arg["name"], array_value(arg))
                          for arg in args if arg["rank"] != 0)
    call = ", &\n       ".join("%s=%s" % (arg["name"], arg["name"]) for arg in args)
    writes = ", &\n       ".join(arg["name"] for arg in args)
    finite = "\n  ".join("if (.not. all(ieee_is_finite(%s))) error stop \"nonfinite %s\"" %
                         (arg["name"], arg["name"]) for arg in args if arg["type"] == "real" and arg["rank"] != 0)
    return """program mynn_bridge_driver
  use, intrinsic :: ieee_arithmetic
  use ccpp_kind_types, only: kind_phys
  use module_bl_mynnedmf_driver, only: mynnedmf_driver
  implicit none
  integer :: unit, status
  character(len=512) :: output
  real(kind_phys) :: rublten_before(1:4,1:33,1:4)
  %s
  call get_command_argument(1, output, status=status)
  if (status /= 0 .or. len_trim(output) == 0) error stop "need output path"
  %s
  %s
  rublten_before = rublten
  call mynnedmf_driver(%s)
  call mynnedmf_driver(%s)
  if (all(rublten(2:3,1:32,2:3) == rublten_before(2:3,1:32,2:3))) error stop "MYNN made no tendency change"
  if (any(rublten(1,:,:) /= rublten_before(1,:,:))) error stop "tile i-low sentinel changed"
  if (any(rublten(4,:,:) /= rublten_before(4,:,:))) error stop "tile i-high sentinel changed"
  if (any(rublten(:,:,1) /= rublten_before(:,:,1))) error stop "tile j-low sentinel changed"
  if (any(rublten(:,:,4) /= rublten_before(:,:,4))) error stop "tile j-high sentinel changed"
  %s
  open(newunit=unit, file=trim(output), status="replace", access="stream", form="unformatted", action="write")
  write(unit) %s
  close(unit)
end program mynn_bridge_driver
""" % (declarations, scalars, arrays, call, call, finite, writes)


class MynnBridgeFortranTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="fortran-diff-", dir=ARTIFACTS)
        cls.work = Path(cls.tmp.name)
        cls.src, cls.mod, cls.obj = cls.work / "src", cls.work / "mod", cls.work / "obj"
        for directory in (cls.src, cls.mod, cls.obj): directory.mkdir()
        cls.env = os.environ.copy()
        for variable in ("TMPDIR", "TMP", "TEMP"): cls.env[variable] = str(ARTIFACTS)
        cls.env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        cls.env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        cls.fc, cls.cc = compiler("FC", "gfortran"), compiler("WTUNE_GNU_CC", "cc")
        cls.copy_and_build_closure()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def invoke(cls, command):
        result = subprocess.run(command, cwd=cls.work, env=cls.env, text=True,
                                capture_output=True, timeout=180)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    @classmethod
    def copy_and_build_closure(cls):
        for original, expected in SOURCES:
            copied = cls.src / original.name
            shutil.copyfile(original, copied)
            if hashlib.sha256(copied.read_bytes()).hexdigest() != expected:
                raise AssertionError("frozen source SHA mismatch: " + original.name)
        flags = ["-cpp", "-ffree-form", "-O1", "-g", "-fcheck=all", "-fsanitize=address,undefined",
                 "-fno-sanitize-recover=undefined", "-J", str(cls.mod), "-I", str(cls.mod), "-c"]
        for original, _ in SOURCES:
            cls.invoke(cls.fc + flags + [str(cls.src / original.name), "-o", str(cls.obj / (original.stem + ".o"))])

    def test_real_driver_direct_and_wrapped_two_call_state_match(self):
        caller = self.work / "mynn_bridge_driver.f90"
        caller.write_text(fixture_source())
        caller_obj = self.obj / "caller.o"
        self.invoke(self.fc + ["-cpp", "-ffree-form", "-O1", "-g", "-fcheck=all", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=undefined", "-J", str(self.mod), "-I", str(self.mod),
                            "-c", str(caller), "-o", str(caller_obj)])
        direct = self.work / "direct"
        objects = [str(path) for path in sorted(self.obj.glob("*.o"))]
        sanitizer_link = ["-L", str(GNU_ASAN_LIB), "-Wl,-rpath," + str(GNU_ASAN_LIB.parents[2]),
                          "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
        self.invoke(self.fc + [*sanitizer_link, *objects, "-o", str(direct)])
        bridge = self.work / "wt_mynn_bridge.c"
        self.invoke([sys.executable, str(PLOV / "gen_mynn_bridge.py"), "--source", str(WRF / "phys/module_bl_mynnedmf_driver.f90"),
                  "--schema", str(PLOV / "mynn_schema.json"), "--symbol", GNU_SYMBOL, "--output", str(bridge)])
        dispatcher = self.work / "dispatcher.c"
        dispatcher.write_text("#include <assert.h>\n#include <stddef.h>\n#include <stdio.h>\nstatic unsigned calls;\n"
                              "void wt_mynn_bridge_dispatch(void **a, void (*f)(void **), size_t n) { assert(a&&f&&n==132); ++calls; f(a); }\n"
                              "__attribute__((destructor)) static void report(void) { fprintf(stderr, \"MYNN_GNU_DISPATCH=%u\\n\", calls); }\n")
        bridge_obj, dispatch_obj = self.obj / "bridge.o", self.obj / "dispatcher.o"
        for source, output in ((bridge, bridge_obj), (dispatcher, dispatch_obj)):
            self.invoke(self.cc + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                                "-fno-sanitize-recover=undefined", "-c", str(source), "-o", str(output)])
        wrapped = self.work / "wrapped"
        self.invoke(self.fc + [*sanitizer_link, *objects,
                            str(bridge_obj), str(dispatch_obj), "-Wl,--wrap=" + GNU_SYMBOL, "-o", str(wrapped)])
        direct_state, wrapped_state = self.work / "direct.bin", self.work / "wrapped.bin"
        direct_run = self.invoke([str(direct), str(direct_state)])
        wrapped_run = self.invoke([str(wrapped), str(wrapped_state)])
        self.assertEqual(direct_state.read_bytes(), wrapped_state.read_bytes())
        self.assertIn("MYNN_GNU_DISPATCH=2", wrapped_run.stderr)
        self.assertNotIn("runtime error:", direct_run.stderr + wrapped_run.stderr)
        self.assertNotIn("AddressSanitizer", direct_run.stderr + wrapped_run.stderr)
        evidence = ARTIFACTS / "gnu-fortran-differential"
        evidence.mkdir(exist_ok=True)
        shutil.copyfile(direct_state, evidence / "direct-two-call-state.bin")
        shutil.copyfile(wrapped_state, evidence / "wrapped-two-call-state.bin")
        (evidence / "manifest.json").write_text(json.dumps({
            "scope": "GNU-only source-closure MYNN driver differential; not Intel/WRF attestation",
            "source_sha256": {str(path.relative_to(WRF)): digest for path, digest in SOURCES},
            "driver_symbol": GNU_SYMBOL, "arguments": NARG, "serial_calls": 2,
            "direct_bytes": direct_state.stat().st_size, "wrapped_bytes": wrapped_state.stat().st_size,
            "wrapped_dispatch_stderr": wrapped_run.stderr,
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
