"""Local GNU-wrap checks for the generated MYNN reference bridge.

The fixture is deliberately C-only.  It proves pointer transport through the
external 132-reference boundary; it cannot attest the frozen Intel/WRF ABI or
the numerical MYNN kernel.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLOV = ROOT / "plov"
TOOL = PLOV / "gen_mynn_bridge.py"
SOURCE = Path("/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/phys/module_bl_mynnedmf_driver.f90")
SCHEMA = PLOV / "mynn_schema.json"
SYMBOL = "module_bl_mynnedmf_driver_mp_mynnedmf_driver_"
NARG = 132
ARTIFACTS = ROOT / "artifacts/20260913-node-local-physics/mynn"


def formals():
    return ", ".join("void *a%d" % index for index in range(NARG))


def names(prefix="a"):
    return ", ".join("%s%d" % (prefix, index) for index in range(NARG))


def slots(prefix):
    return ", ".join("%s[%d]" % (prefix, index) for index in range(NARG))


class MynnBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="bridge-", dir=ARTIFACTS)
        cls.build = Path(cls.tmp.name)
        cls.env = os.environ.copy()
        for variable in ("TMPDIR", "TMP", "TEMP"):
            cls.env[variable] = str(ARTIFACTS)
        cls.env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        cls.env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_generator(self, source=SOURCE, schema=SCHEMA, output=None, extra=()):
        output = output or (self.build / "wt_mynn_bridge.c")
        return subprocess.run(
            [sys.executable, str(TOOL), "--source", str(source), "--schema", str(schema),
             "--output", str(output), *extra], cwd=ROOT, env=self.env,
            text=True, capture_output=True, timeout=60)

    def test_locked_generator_produces_exact_reference_bridge(self):
        output = self.build / "generated.c"
        result = self.run_generator(output=output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        generated = output.read_text()
        self.assertEqual(generated, (PLOV / "wt_mynn_bridge.c").read_text())
        self.assertIn("132 arguments, 47 optional", generated)
        self.assertIn("void *args[WT_MYNN_BRIDGE_NARG]", generated)
        self.assertIn("extern void wt_mynn_bridge_dispatch", generated)
        self.assertNotIn("static void wt_mynn_bridge_dispatch", generated)
        self.assertNotIn("static void *wt_argv", generated)
        self.assertNotIn("__asm", generated)
        self.assertNotIn("register", generated)

    def rejected(self, source=SOURCE, schema=SCHEMA, extra=()):
        output = self.build / ("rejected-%d.c" % len(list(self.build.glob("rejected-*.c"))))
        output.write_text("preserved\n")
        result = self.run_generator(source=source, schema=schema, output=output, extra=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "preserved\n")
        return result

    def test_stale_hash_same_length_reorder_unsupported_abi_and_cli_are_refused(self):
        drifted = self.build / "driver-drift.f90"
        drifted.write_bytes(SOURCE.read_bytes() + b"! source drift\n")
        self.assertIn("source SHA-256", self.rejected(source=drifted).stderr)

        reordered = self.build / "schema-reordered.json"
        data = json.loads(SCHEMA.read_text())
        data["arguments"][0], data["arguments"][1] = data["arguments"][1], data["arguments"][0]
        data["arguments"][0]["index"], data["arguments"][1]["index"] = 0, 1
        reordered.write_text(json.dumps(data))
        self.assertIn("schema SHA-256", self.rejected(schema=reordered).stderr)

        unsupported = self.build / "driver-pointer.f90"
        unsupported.write_text(SOURCE.read_text().replace(
            "INTEGER, intent(in) :: initflag,                      &",
            "INTEGER, pointer, intent(in) :: initflag,             &", 1))
        self.assertIn("unsupported MYNN ABI declaration", self.rejected(source=unsupported).stderr)
        self.assertIn("unrecognized arguments", self.rejected(extra=("--not-an-option",)).stderr)

    def test_linker_wrap_forwards_all_pointers_null_optionals_nested_and_threads(self):
        schema = json.loads(SCHEMA.read_text())
        optional = {argument["index"] for argument in schema["arguments"] if argument["optional"]}
        self.assertEqual(len(optional), 47)
        bridge = self.build / "bridge.c"
        result = self.run_generator(output=bridge)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (self.build / "fixture.h").write_text("void %s(%s);\n" % (SYMBOL, formals()))
        (self.build / "real.c").write_text(self.real_fixture())
        (self.build / "nested.c").write_text(self.nested_fixture(optional))
        (self.build / "dispatcher.c").write_text(self.dispatcher_fixture())
        (self.build / "caller.c").write_text(self.caller_fixture(optional))
        compiler = shlex.split(os.environ.get("CC", "cc"))
        executable, link_map = self.build / "bridge", self.build / "bridge.map"
        command = compiler + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
            "-Wpedantic", "-pthread", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
            "-fno-omit-frame-pointer", "real.c", "nested.c", "dispatcher.c", "caller.c", str(bridge), "-Wl,--wrap=" + SYMBOL,
            "-Wl,-Map," + str(link_map), "-o", str(executable)]
        built = subprocess.run(command, cwd=self.build, env=self.env, text=True,
                               capture_output=True, timeout=60)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        self.assertIn("__wrap_" + SYMBOL, link_map.read_text())
        ran = subprocess.run([str(executable)], cwd=self.build, env=self.env, text=True,
                             capture_output=True, timeout=60)
        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
        self.assertEqual(ran.stdout, "MYNN bridge wrap checks PASS\n")
        self.assertNotIn("runtime error:", ran.stderr)
        self.assertNotIn("AddressSanitizer", ran.stderr)

    @staticmethod
    def call_arguments(array):
        return slots(array)

    def test_sanitizer_configuration_rejects_an_actual_ub_witness(self):
        witness, executable = self.build / "ub-witness.c", self.build / "ub-witness"
        witness.write_text("#include <limits.h>\nint main(void) { volatile int n=INT_MAX; return n+1; }\n")
        compiler = shlex.split(os.environ.get("CC", "cc"))
        built = subprocess.run(compiler + ["-std=c11", "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined", str(witness), "-o", str(executable)],
            cwd=self.build, env=self.env, text=True, capture_output=True, timeout=60)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        ran = subprocess.run([str(executable)], cwd=self.build, env=self.env, text=True,
                             capture_output=True, timeout=60)
        self.assertNotEqual(ran.returncode, 0)
        self.assertIn("runtime error:", ran.stderr)

    def real_fixture(self):
        return """#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>
#include "fixture.h"
void run_nested(void **);
static _Thread_local void *expected[%d];
static _Thread_local int nested;
static _Atomic unsigned calls;
void set_expected(void **table) { for (size_t i=0; i<%d; ++i) expected[i]=table[i]; }
void %s(%s) {
  void *seen[%d] = {%s};
  for (size_t i=0; i<%d; ++i) assert(seen[i] == expected[i]);
  atomic_fetch_add_explicit(&calls, 1u, memory_order_relaxed);
  if (seen[0] && *(int *)seen[0] == 100 && !nested) {
    nested=1; run_nested(expected); nested=0;
    for (size_t i=0; i<%d; ++i) assert(seen[i] == expected[i]);
  }
}
unsigned observed_calls(void) { return atomic_load_explicit(&calls, memory_order_relaxed); }
""" % (NARG, NARG, SYMBOL, formals(), NARG, names(), NARG, NARG)

    def nested_fixture(self, optional):
        return """#include <stddef.h>
#include "fixture.h"
void set_expected(void **);
void run_nested(void **outer) {
  int values[%d]; void *table[%d]; void *saved[%d];
  for (size_t i=0; i<%d; ++i) { values[i]=1000+(int)i; table[i]=&values[i]; saved[i]=outer[i]; }
%s
  set_expected(table); %s(%s); set_expected(saved);
}
""" % (NARG, NARG, NARG, NARG,
         "".join("  if ((%d %% 2) == 0) table[%d]=NULL;\n" % (index, index)
                 for index in sorted(optional)),
         SYMBOL, self.call_arguments("table"))

    def dispatcher_fixture(self):
        return """#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>
static _Atomic unsigned calls;
void wt_mynn_bridge_dispatch(void **argv, void (*invoke)(void **), size_t nargs) {
  assert(argv != NULL); assert(invoke != NULL); assert(nargs == %d);
  atomic_fetch_add_explicit(&calls, 1u, memory_order_relaxed); invoke(argv);
}
unsigned dispatched_calls(void) { return atomic_load_explicit(&calls, memory_order_relaxed); }
""" % NARG

    def caller_fixture(self, optional):
        return """#define _XOPEN_SOURCE 700
#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include "fixture.h"
void set_expected(void **); unsigned observed_calls(void);
unsigned dispatched_calls(void);
struct worker { int seed; unsigned rounds; pthread_barrier_t *barrier; };
static void invoke(int seed, int mode) {
  int values[%d]; void *table[%d];
  for (size_t i=0; i<%d; ++i) { values[i]=seed+(int)i; table[i]=&values[i]; }
%s
  set_expected(table); %s(%s);
}
static void *thread_main(void *opaque) {
  struct worker *worker=opaque; pthread_barrier_wait(worker->barrier);
  for (unsigned i=0; i<worker->rounds; ++i) invoke(worker->seed+(int)(1000*i), (int)(i%%3));
  return NULL;
}
int main(void) {
  pthread_t one,two; pthread_barrier_t barrier; struct worker a={10000,32,&barrier}, b={20000,32,&barrier};
  invoke(100,1); invoke(200,0); invoke(300,2); assert(pthread_barrier_init(&barrier,NULL,2)==0);
  assert(pthread_create(&one,NULL,thread_main,&a)==0); assert(pthread_create(&two,NULL,thread_main,&b)==0);
  assert(pthread_join(one,NULL)==0); assert(pthread_join(two,NULL)==0); assert(pthread_barrier_destroy(&barrier)==0);
  assert(observed_calls()==68u); assert(dispatched_calls()==68u); puts("MYNN bridge wrap checks PASS"); return 0;
}
""" % (NARG, NARG, NARG,
         "".join("  if (mode == 1 || (mode == 2 && (%d %% 2) == 0)) table[%d]=NULL;\n" %
                 (index, index) for index in sorted(optional)),
         SYMBOL, self.call_arguments("table"))


if __name__ == "__main__":
    unittest.main()
