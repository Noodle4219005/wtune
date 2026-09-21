"""Full MYNN production plumbing with real frozen GNU arithmetic and fork/OMP2.

Reuses the separately accepted GNU dependency closure without rebuilding WRF.
Every run copies and hashes its input sources/objects into a unique retained dir.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from plov.tests.test_mynn_bridge_fortran import SOURCES, fixture_source, GNU_SYMBOL

ROOT=Path(__file__).resolve().parents[2]
PLOV=ROOT/"plov"
ART=ROOT/"artifacts/20260913-node-local-physics/mynn/production-live"
BIN=Path("/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/.pixi/envs/default/bin")

class MynnNodeLiveTests(unittest.TestCase):
    def test_first_rk_to_real_gnu_node_with_omp2(self):
        ART.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix="run-",dir=ART))
        src,mod,obj=(work/n for n in ("src","mod","obj"))
        for p in (src,mod,obj):p.mkdir()
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        env=dict(os.environ,TMPDIR=str(work),TMP=str(work),TEMP=str(work),
                 ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:disable_coredump=1",
                 UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        fc=str(BIN/"x86_64-conda-linux-gnu-gfortran");cc=str(BIN/"x86_64-conda-linux-gnu-cc")
        accepted=ROOT/"artifacts/20260913-node-local-physics/mynn/gnu-node/latest.json"
        cache=Path(json.loads(accepted.read_text())["work"])
        self.assertEqual(json.loads((cache/"manifest.json").read_text())["status"],"PASS")
        manifest={"scope":"Real GNU arithmetic + production bootstrap/firstRK84/bridge132/fork/OMP2; native grid/cfg are fixture adapters",
                  "accepted_closure":str(cache),"inputs":{},"work":str(work)}
        def copy(original,target):
            shutil.copyfile(original,target)
            manifest["inputs"][str(original)]=hashlib.sha256(target.read_bytes()).hexdigest()
        with (work/"commands.txt").open("w") as log:
            def run(command,expected=0,timeout=60,extra=None):
                log.write(shlex.join(command)+"\n");log.flush()
                e=env.copy();e.update(extra or {})
                p=subprocess.run(command,cwd=work,env=e,text=True,capture_output=True,timeout=timeout)
                log.write(p.stdout+p.stderr+"RETURN %d\n"%p.returncode);log.flush()
                self.assertEqual(p.returncode,expected,str(work)+"\n"+p.stdout+p.stderr)
                return p
            objects=[]
            for original,sha in SOURCES:
                cached=cache/"src"/original.name
                self.assertEqual(hashlib.sha256(cached.read_bytes()).hexdigest(),sha)
                copy(cached,src/original.name)
                output=obj/(original.stem+".o");copy(cache/"obj"/output.name,output);objects.append(str(output))
            for original in (cache/"mod").glob("*.mod"):copy(original,mod/original.name)
            for name in ("wt_mynn_static_state.f90","tests/test_mynn_node_live.c"):
                copy(PLOV/name,src/Path(name).name)
            san=["-fsanitize=address,undefined","-fno-sanitize-recover=all","-fno-omit-frame-pointer"]
            fflags=["-cpp","-ffree-form","-O1","-g","-fcheck=all",*san,"-J",str(mod),"-I",str(mod)]
            static=obj/"wt_mynn_static_state.o"
            run([fc,*fflags,"-c",str(src/"wt_mynn_static_state.f90"),"-o",str(static)])
            objects.append(str(static))
            caller=fixture_source()
            caller=caller.replace("rublten_before(1:4,1:33,1:4)","rublten_before(1:4,1:33,1:6)")
            caller=caller.replace("integer, parameter :: jme = 4","integer, parameter :: jme = 6")
            caller=caller.replace("integer, parameter :: jde = 4","integer, parameter :: jde = 6")
            caller=caller.replace("integer, parameter :: jte = 3","integer, parameter :: jte = 5")
            caller=caller.replace("rublten(:,:,4)","rublten(:,:,6)").replace("rublten_before(:,:,4)","rublten_before(:,:,6)")
            call=re.search(r"  call mynnedmf_driver\(.*?\)\n",caller,re.S).group()
            self.assertEqual(caller.count(call),2);caller=caller.replace(call+call,call)
            (src/"caller.f90").write_text(caller)
            caller_obj=obj/"caller.o"
            run([fc,*fflags,"-c",str(src/"caller.f90"),"-o",str(caller_obj)])
            objects.append(str(caller_obj))
            bridge=src/"bridge.c"
            run([sys.executable,str(PLOV/"gen_mynn_bridge.py"),"--source",str(src/"module_bl_mynnedmf_driver.f90"),
                 "--schema",str(PLOV/"mynn_schema.json"),"--symbol",GNU_SYMBOL,"--output",str(bridge)])
            formals=",".join("void *a%d"%i for i in range(84))
            actual=",".join("a%d"%i for i in range(84))
            abi=("#include <stddef.h>\nextern void mynn_live_first_real(void **);\n"
                 "void module_first_rk_step_part1_mp_first_rk_step_part1_("+formals+") { void *a[]={"+actual+"}; mynn_live_first_real(a); }\n"
                 "extern void "+GNU_SYMBOL+"("+",".join(["void *"]*132)+");\n"
                 "void mynn_live_driver_call(void **a) { "+GNU_SYMBOL+"("+",".join("a[%d]"%i for i in range(132))+"); }\n")
            (src/"abi.c").write_text(abi)
            (work/"mynn-live-abi.h").write_text('void mynn_live_driver_call(void **);\n#define MYNN_LIVE_ARTIFACT_DIRECTORY "%s"\n'%work)
            units=["wt_live_spans","wt_mynn_layout","wt_mynn_batch","wt_mynn_node","wt_node_local","wt_kf_node_shm",
                   "wt_mynn_native","wt_mynn_node_live","wt_mynn_bootstrap","wt_sfclay_scope_bridge"]
            copied=set()
            def copy_unit(name):
                if name in copied:return
                copied.add(name);copy(PLOV/name,src/name)
                for dependency in re.findall(r'#include "([^"/]+)"',(PLOV/name).read_text()):
                    if (PLOV/dependency).is_file():copy_unit(dependency)
            for unit in units:copy_unit(unit+".c")
            for tag in ("strict","sanitized","unattested"):
                flags=["-std=c11","-Wall","-Wextra","-Werror","-O1","-g","-fopenmp","-pthread",
                       "-DWT_MYNN_NODE_LOCAL","-DWT_SFCLAY_SCOPE_TEST_HELPERS","-I",str(PLOV),"-I",str(work),
                       '-DWT_KF_NODE_SHM_DIRECTORY="%s"'%work,'-DWT_MYNN_BUILD_PROFILE_ID="GNU-production-fixture-only"']
                if tag!="strict":flags+=san
                if tag=="unattested":flags+=["-DMYNN_LIVE_UNATTESTED"]
                else:flags += ["-DWT_MYNN_BUILD_ABI_VERIFIED=1","-DWT_MYNN_BUILD_CALLER_VERIFIED=1","-DWT_MYNN_BUILD_STATIC_STATE_VERIFIED=1"]
                extra_objects=[]
                for source in [bridge,src/"abi.c",src/"test_mynn_node_live.c",*[src/(u+".c") for u in units]]:
                    output=obj/(source.stem+"-"+tag+".o")
                    run([cc,*flags,"-c",str(source),"-o",str(output)])
                    extra_objects.append(str(output))
                exe=work/tag
                run([fc,*san,"-fopenmp","-pthread",*objects,*extra_objects,"-Wl,--wrap="+GNU_SYMBOL,
                     "-Wl,--wrap=module_first_rk_step_part1_mp_first_rk_step_part1_","-Wl,--wrap=wt_mynn_bridge_dispatch","-o",str(exe)])
                for case in ([3] if tag=="unattested" else [0,1,2,3,4,5,6,7,8]):
                    outcome=run([str(exe),str(work/(tag+"-"+str(case)+".bin"))],
                                -signal.SIGABRT if case in (4,5,7) else 0,30,{"MYNN_LIVE_CASE":str(case)})
                    if case not in (4,5,7):self.assertIn("PASS MYNN production",outcome.stderr)
                    elif case==5:
                        entries=[line for line in outcome.stderr.splitlines() if line.startswith("MYNN_NATIVE")]
                        self.assertGreaterEqual(len(entries),2)
                        self.assertEqual(len(entries),len(set(entries)),"post-native original tile replay")
            manifest["status"]="PASS"
            (work/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            (ART/"latest.json").write_text(json.dumps({"status":"PASS","work":str(work)})+"\n")
            print("MYNN production live PASS "+str(work))

if __name__=="__main__":unittest.main()
