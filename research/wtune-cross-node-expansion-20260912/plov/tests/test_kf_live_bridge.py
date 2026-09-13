"""Link the actual outer trampoline, generated CPS bridge and live transaction."""
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CPS = "module_cu_kfeta_mp_kf_eta_cps_"
OUTER = "module_cumulus_driver_mp_cumulus_driver_"
FORMALS = ",".join(f"void *a{i}" for i in range(79))
ACTUALS = ",".join(f"expected[{i}]" for i in range(79))

MAIN = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "wt_kf_live.h"
#include "wt_cumulus.h"
extern void module_cumulus_driver_mp_cumulus_driver_(void*);
void *expected[WT_KF_NARG]; int real_calls, captures, marker;
static _Alignas(4) unsigned char storage[WT_KF_NARG][128];
static int32_t scalar[19];
static int omp_no(void) { return 0; }
static int capture(void *grid, wt_live_spans *spans, size_t limit, int *tiles) {
  assert(grid == &marker && limit == 128); ++captures; *tiles=1;
  for (unsigned i=0;i<WT_KF_NARG;++i) {
    if (wt_kf_shape[i] == WT_KF_SCALAR || i == WT_KF_PRATEC) continue;
    int type = i == WT_KF_CU_ACT_FLAG ? WT_SPAN_LOGICAL : WT_SPAN_REAL;
    int rank = wt_kf_shape[i] == WT_KF_3D ? 3 : 2;
    assert(wt_live_spans_add(spans,expected[i],128,type,4,rank) == WT_SPAN_OK);
  }
  return 1;
}
int main(void) {
  struct wt_kf_live_config config = {0};
  struct wt_kf_live_counters counters;
  for (unsigned i=0;i<WT_KF_NARG;++i) {
    memset(storage[i],(int)i+1,sizeof storage[i]); expected[i]=storage[i];
  }
  for (unsigned i=0;i<19;++i) { scalar[i]=1; expected[i]=&scalar[i]; }
  scalar[WT_KF_KME]=scalar[WT_KF_KDE]=2;
  int32_t zero=0;
  memcpy(expected[WT_KF_ADAPT_STEP_FLAG],&zero,4);
  memcpy(expected[WT_KF_KF_EDRATES],&zero,4);
  config.enabled=1; config.source_abi_verified=EXPECT_STAGE;
  config.max_spans=128; config.max_nodes=128; config.byte_budget=100000;
  config.capture=capture; config.omp_query=omp_no;
  assert(wt_kf_live_configure(&config));
  for (int call=0;call<2;++call) {
    module_cumulus_driver_mp_cumulus_driver_(&marker);
    assert(real_calls == call+1);
    assert(storage[WT_KF_RAINCV][0] == WT_KF_RAINCV+2+call);
  }
  wt_kf_live_get_counters(&counters);
  assert(counters.requested == 2);
  assert(counters.staged == (EXPECT_STAGE ? 2 : 0));
  assert(counters.fallback == (EXPECT_STAGE ? 0 : 2));
  assert(captures == (EXPECT_STAGE ? 2 : 0));
  return 0;
}
'''


@unittest.skipUnless(platform.machine() in ("x86_64", "AMD64"),
                     "existing Intel x86-64 trampoline ABI")
class KFLiveBridgeTests(unittest.TestCase):
    def check_chain(self, enabled):
        with tempfile.TemporaryDirectory(prefix="wt-kf-live-bridge-") as directory:
            tmp = Path(directory)
            (tmp / "main.c").write_text(MAIN)
            (tmp / "outer.c").write_text(f'''
#include <assert.h>
#include <stdint.h>
extern void *expected[79]; extern int marker;
void {CPS}({FORMALS});
void {OUTER}(void *grid) {{ assert(grid == &marker); {CPS}({ACTUALS}); }}
void wt_pre(int idx) {{ assert(idx == 5); }}
void wt_record(int idx,uint64_t cycles) {{ assert(idx == 5); (void)cycles; }}
void wt_micro_call(void) {{}}
void wt_rad_call(void) {{}}
void __real_module_radiation_driver_mp_pre_radiation_driver_(void) {{}}
void __real_module_pbl_driver_mp_pbl_driver_(void) {{}}
void __real_module_surface_driver_mp_surface_driver_(void) {{}}
''')
            args = ",".join(f"a{i}" for i in range(79))
            (tmp / "real.c").write_text(f'''
#include <assert.h>
#include <string.h>
#include "wt_cumulus.h"
extern void *expected[79]; extern int real_calls;
void {CPS}({FORMALS}) {{
  void *actual[79]={{{args}}}; ++real_calls;
  for (unsigned i=0;i<79;++i) assert(!memcmp(actual[i],expected[i],4));
  assert((actual[WT_KF_RAINCV] != expected[WT_KF_RAINCV]) == EXPECT_STAGE);
  ++((unsigned char*)actual[WT_KF_RAINCV])[0];
}}
''')
            compiler = shlex.split(os.environ.get("CC", "cc"))
            flags = shlex.split(os.environ.get("CFLAGS", "")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT), f"-DEXPECT_STAGE={int(enabled)}"]
            subprocess.run(compiler + flags + [
                "-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch", "-c",
                str(ROOT / "wt_kf_bridge.c"), "-o", "bridge.o"],
                cwd=tmp, check=True, capture_output=True, timeout=60)
            modules = ["wt_live_spans.c", "wt_kf_capacity.c", "wt_kf_dispatch.c",
                       "wt_cumulus.c", "wt_kf_live.c", "wtimer.S"]
            subprocess.run(compiler + flags + ["-DWT_CUMULUS_LIVE_SPANS"] +
                           [str(ROOT / name) for name in modules] +
                           ["main.c", "outer.c", "real.c", "bridge.o",
                            f"-Wl,--wrap={CPS}", f"-Wl,--wrap={OUTER}",
                            "-o", "fixture"],
                           cwd=tmp, check=True, capture_output=True, timeout=60)
            subprocess.run([str(tmp / "fixture")], cwd=tmp, check=True,
                           capture_output=True, timeout=60)

    def test_two_outer_calls_stage_and_commit(self):
        self.check_chain(True)

    def test_unattested_source_forwards_once(self):
        self.check_chain(False)
