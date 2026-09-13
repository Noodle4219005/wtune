"""Exercise the x86-64 outer-driver trampoline with actual stack arguments."""
import os
import platform
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''
#include <assert.h>
#include <stdint.h>
static int step, values[10];
void wt_pre(int idx) { assert(idx == 5 && step == 0); step = 1; }
void wt_record(int idx, uint64_t cycles) {
  (void)cycles; assert(idx == 5 && step == 3); step = 4;
}
void wt_cumulus_grid_begin(void *grid) {
  assert(grid == &values[0] && step == 1); step = 2;
}
void wt_cumulus_grid_end(void) { assert(step == 4); step = 5; }
void __real_module_cumulus_driver_mp_cumulus_driver_(
    void *a,void *b,void *c,void *d,void *e,void *f,
    void *g,void *h,void *i,void *j) {
  void *args[] = {a,b,c,d,e,f,g,h,i,j};
#ifdef HOOK_ON
  assert(step == 2);
#else
  assert(step == 1);
#endif
  for (int k=0;k<10;++k) { assert(args[k] == &values[k]); values[k] += k+1; }
  step = 3;
}
void __real_module_radiation_driver_mp_pre_radiation_driver_(void) {}
void __real_module_pbl_driver_mp_pbl_driver_(void) {}
void __real_module_surface_driver_mp_surface_driver_(void) {}
void wt_micro_call(void) {}
void wt_rad_call(void) {}
extern void __wrap_module_cumulus_driver_mp_cumulus_driver_(
  void*,void*,void*,void*,void*,void*,void*,void*,void*,void*);
int main(void) {
  for (int call=0;call<2;++call) {
    step=0;
    __wrap_module_cumulus_driver_mp_cumulus_driver_(
      &values[0],&values[1],&values[2],&values[3],&values[4],
      &values[5],&values[6],&values[7],&values[8],&values[9]);
#ifdef HOOK_ON
    assert(step == 5);
#else
    assert(step == 4);
#endif
    for (int k=0;k<10;++k) assert(values[k] == (call+1)*(k+1));
  }
  return 0;
}
'''


@unittest.skipUnless(platform.machine() in ("x86_64", "AMD64"),
                     "existing Intel x86-64 trampoline ABI")
class CumulusGridHookTests(unittest.TestCase):
    def check_hook(self, enabled):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mock.c"
            source.write_text(MOCK)
            exe = Path(tmp) / "hook"
            command = shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
            if enabled:
                command += ["-DWT_CUMULUS_LIVE_SPANS", "-DHOOK_ON"]
            command += [str(ROOT / "wtimer.S"), str(source), "-o", str(exe)]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            subprocess.run([str(exe)], check=True, capture_output=True, timeout=60)

    def test_enabled_preserves_arguments_and_scope(self):
        self.check_hook(True)

    def test_disabled_preserves_existing_path(self):
        self.check_hook(False)
