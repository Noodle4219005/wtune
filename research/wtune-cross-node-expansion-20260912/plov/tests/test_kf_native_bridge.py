"""End-to-end test of the generated 79-pointer KF bridge and transaction adapter."""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYMBOL = "module_cu_kfeta_mp_kf_eta_cps_"
N = 79


def c_formals():
    return ", ".join(f"void *a{i}" for i in range(N))


def c_args(prefix="a"):
    return ", ".join(f"{prefix}{i}" for i in range(N))


class KfNativeBridgeTests(unittest.TestCase):
    def build_and_run(self, direct):
        bridge = ROOT / "plov" / "wt_kf_bridge.c"
        self.assertTrue(bridge.exists(), "root-generated bridge is required")
        with tempfile.TemporaryDirectory(prefix="wt-kf-native-") as tmp:
            tmp = Path(tmp)
            real = f'''#include <assert.h>
#include <stdint.h>
#include "wt_kf_dispatch.h"
extern void *expected[{N}]; extern int real_calls;
void {SYMBOL}({c_formals()}) {{
  void *p[{N}]={{{c_args()}}}; ++real_calls;
  for(int i=0;i<{N};++i) {{
    unsigned char want = i < 19 ? ((i==WT_KF_KDE || i==WT_KF_KME) ? 2 : 1) : (unsigned char)(i+1);
    if (i==WT_KF_ADAPT_STEP_FLAG) want=0;
    assert(p[i] != 0 && ((unsigned char *)p[i])[0] == want);
  }}
  assert(p[WT_KF_RAINCV] {'==' if direct else '!='} expected[WT_KF_RAINCV]);
  ((unsigned char *)p[WT_KF_RAINCV])[0]++;
}}
'''
            caller = f'''#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "wt_kf_dispatch.h"
void {SYMBOL}({c_formals()});
void *expected[{N}]; int real_calls;
static unsigned char storage[{N}][128]; static int32_t scalar[19];
static int cap(void *x,unsigned i,void *p,size_t *n) {{ (void)x;(void)p; *n=i<19?4:128; return 1; }}
static int omp_no(void *x) {{ (void)x; return 0; }}
int wt_kf_omp_in_parallel(void) {{ return 0; }}
int main(void) {{ void *a[{N}]; struct wt_kf_dispatch_context c={{0}};
  (void)c; (void)cap; (void)omp_no;
  for(int i=0;i<{N};++i) {{ memset(storage[i],i+1,128); a[i]=storage[i]; expected[i]=a[i]; }}
  for(int i=0;i<19;++i) {{ scalar[i]=1; a[i]=&scalar[i]; expected[i]=a[i]; }}
  scalar[WT_KF_KDE]=scalar[WT_KF_KME]=2; scalar[WT_KF_TRIGGER]=1;
  {{ int32_t zero=0; memcpy(a[WT_KF_ADAPT_STEP_FLAG],&zero,sizeof(zero)); }}
  {'c.enabled=c.abi_verified=1; c.transaction.enabled=1; c.transaction.num_tiles=1; c.transaction.byte_budget=100000; c.capacity=cap; c.omp_query=omp_no; assert(wt_kf_bridge_configure(&c)); assert(!wt_kf_bridge_configure(&c));' if not direct else ''}
  {SYMBOL}({', '.join('a[%d]' % i for i in range(N))});
  assert(real_calls==1); assert(storage[WT_KF_RAINCV][0]==(unsigned char)(WT_KF_RAINCV+2)); return 0;
}}
'''
            (tmp / "real.c").write_text(real)
            (tmp / "caller.c").write_text(caller)
            compiler = shlex.split(os.environ.get("CC", "cc"))
            flags = shlex.split(os.environ.get("CFLAGS", ""))
            command = compiler + ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "plov")] + flags + [
                str(ROOT / "plov" / "wt_cumulus.c"), str(ROOT / "plov" / "wt_kf_dispatch.c"),
                str(bridge), "caller.c", "real.c", f"-Wl,--wrap={SYMBOL}", "-o", "bridge-test"]
            built = subprocess.run(command, cwd=tmp, text=True, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(tmp / "bridge-test")], text=True, capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_configured_bridge_stages_and_commits_mock_cps(self):
        self.build_and_run(direct=False)

    def test_unconfigured_bridge_forwards_original_pointers(self):
        self.build_and_run(direct=True)


if __name__ == "__main__":
    unittest.main()
