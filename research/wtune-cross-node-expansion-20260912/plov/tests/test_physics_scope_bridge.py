"""Verify the shared first-RK wrapper composes SFCLAY and MYNN scopes."""

import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/20260913-node-local-physics/sfclay/scope-compose"
DEFAULT_CC = Path("/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/"
                  ".pixi/envs/default/bin/x86_64-conda-linux-gnu-cc")


def fixture_source(expected):
    formals=", ".join("void *a%d" % i for i in range(84))
    args=", ".join("a%d" % i for i in range(84))
    expected_values=", ".join(str(x) for x in expected)
    return r'''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK %%d: %%s\n",__LINE__,#x); exit(2); } } while(0)
static int event[8],n; static void *want[84];
uint64_t wt_sfclay_first_rk_begin(void *g,void *c) { CHECK(g==want[0] && c==want[1]); event[n++]=1; return 11; }
void wt_sfclay_first_rk_end(uint64_t t) { CHECK(t==11); event[n++]=5; }
uint64_t wt_mynn_first_rk_begin(void *g,void *c) { CHECK(g==want[0] && c==want[1]); event[n++]=2; return 22; }
void wt_mynn_first_rk_end(uint64_t t) { CHECK(t==22); event[n++]=4; }
void module_first_rk_step_part1_mp_first_rk_step_part1_(%s) {
  void *got[84]={%s}; event[n++]=3;
  for(unsigned i=0;i<84;++i) CHECK(got[i]==want[i]);
}
extern void wt_sfclay_scope_test_call(void **);
int main(void) {
  int storage[84]; static const int expected[]={%s};
  for(unsigned i=0;i<84;++i) want[i]=&storage[i];
  want[80]=want[81]=want[82]=want[83]=0;
  wt_sfclay_scope_test_call(want);
  CHECK(n==sizeof(expected)/sizeof(expected[0]));
  for(unsigned i=0;i<(unsigned)n;++i) CHECK(event[i]==expected[i]);
  puts("shared physics first-RK scope forwarding passed"); return 0;
}
''' % (formals,args,expected_values)


class PhysicsScopeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACT.mkdir(parents=True,exist_ok=True)
        include=ARTIFACT/"include"; include.mkdir(exist_ok=True)
        (include/"wt_mynn_node_live.h").write_text(
            "#ifndef WT_MYNN_NODE_LIVE_H\n#define WT_MYNN_NODE_LIVE_H\n"
            "#include <stdint.h>\nuint64_t wt_mynn_first_rk_begin(void *,void *);\n"
            "void wt_mynn_first_rk_end(uint64_t);\n#endif\n")
        cls.cc=shlex.split(os.environ.get("CC","")) or [str(DEFAULT_CC)]
        cls.environment=os.environ.copy()
        cls.environment.update(ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1",
                               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
                               TMPDIR=str(ARTIFACT),TMP=str(ARTIFACT),TEMP=str(ARTIFACT))

    def build_run(self,name,macros,expected):
        source=ARTIFACT/(name+".c"); executable=ARTIFACT/name
        source.write_text(fixture_source(expected))
        command=[*self.cc,"-std=c11","-O1","-g","-Wall","-Wextra","-Werror",
                 "-pedantic","-fsanitize=address,undefined","-fno-sanitize-recover=all",
                 "-fno-omit-frame-pointer","-DWT_SFCLAY_SCOPE_TEST_HELPERS",
                 "-I"+str(ARTIFACT/"include"),"-Iplov",*macros,
                 "plov/wt_sfclay_scope_bridge.c",str(source),
                 "-Wl,--wrap=module_first_rk_step_part1_mp_first_rk_step_part1_",
                 "-o",str(executable)]
        compile_result=subprocess.run(command,cwd=ROOT,env=self.environment,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
        self.assertEqual(compile_result.returncode,0,
                         compile_result.stdout+compile_result.stderr)
        result=subprocess.run([str(executable)],cwd=ROOT,env=self.environment,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
        (ARTIFACT/(name+".txt")).write_text(
            " ".join(shlex.quote(x) for x in command)+"\n"+
            compile_result.stdout+compile_result.stderr+result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("scope forwarding passed",result.stdout)

    def test_sf_historical_mynn_only_and_composed_order(self):
        cases=(("sf-default",[],[1,3,5]),
               ("mynn-only",["-DWT_MYNN_NODE_LOCAL"],[2,3,4]),
               ("sf-mynn",["-DWT_MYNN_NODE_LOCAL","-DWT_SFCLAY_NODE_LOCAL"],
                [1,2,3,4,5]))
        for name,macros,expected in cases:
            with self.subTest(name=name): self.build_run(name,macros,expected)


if __name__=="__main__":
    unittest.main()
