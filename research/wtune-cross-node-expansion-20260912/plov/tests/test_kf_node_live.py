import os
import pathlib
import shlex
import signal
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "artifacts/20260913-node-local-physics/kf/live"

class KFNodeLiveTest(unittest.TestCase):
    def test_live_production_chain(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        header=[]
        real=[]
        for name,n in (("INIT",35),("CPS",79)):
            header += ["#define KF_%s_FORMALS " % name + ",".join("void *a%d" % i for i in range(n)),
                       "#define KF_%s_ARGV {" % name + ",".join("a%d" % i for i in range(n)) + "}",
                       "#define KF_%s_FROM_ARRAY(a) " % name + ",".join("a[%d]" % i for i in range(n))]
            symbol = "module_cu_kfeta_mp_kf_eta_" + ("init_" if name == "INIT" else "cps_")
            formals = ",".join("void *a%d" % i for i in range(n))
            actual = ",".join("a%d" % i for i in range(n))
            fixture = "kf_fixture_" + name.lower() + "_real"
            header.append("extern void " + symbol + "(" + formals + ");")
            real.append("extern void " + fixture + "(" + formals + ");\nvoid " + symbol + "(" + formals + ") { " + fixture + "(" + actual + "); }")
        (ARTIFACTS / "real-abi.c").write_text("\n".join(real)+"\n")
        (ARTIFACTS / "kf-node-live-abi.h").write_text("\n".join(header)+"\n")
        units = ["wt_live_spans", "wt_kf_capacity", "wt_kf_dispatch", "wt_cumulus", "wt_kf_live", "wt_kf_native", "wt_kf_bootstrap", "wt_kf_batch", "wt_kf_batch_codec", "wt_kf_node", "wt_node_local", "wt_kf_node_live", "wt_kf_node_shm", "wt_kf_init_bridge"]
        for sanitized in (False,True):
            tag = "sanitized" if sanitized else "strict"
            exe = ARTIFACTS / ("live-" + tag)
            compiler = shlex.split(os.environ.get("CC", "cc"))
            flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g", "-fopenmp", "-pthread", "-DWT_KF_NODE_LOCAL", "-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1", "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1", "-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1", '-DWT_KF_BUILD_PROFILE_ID="test-fixture-only"', '-DWT_KF_NODE_SHM_DIRECTORY="%s"' % ARTIFACTS, "-I", str(ROOT), "-I", str(ARTIFACTS)]
            if sanitized:
                flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
            bridge_obj=ARTIFACTS / ("bridge-"+tag+".o")
            bridge = compiler+flags+["-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch", "-c", str(ROOT/"wt_kf_bridge.c"), "-o", str(bridge_obj)]
            command = compiler+flags+[str(ROOT/(unit+".c")) for unit in units]+[str(ROOT/"tests/test_kf_node_live.c"), str(ARTIFACTS/"real-abi.c"), str(bridge_obj), "-Wl,--wrap=module_cu_kfeta_mp_kf_eta_cps_", "-Wl,--wrap=module_cu_kfeta_mp_kf_eta_init_", "-Wl,--wrap=wt_nl_try_execute", "-Wl,--wrap=wt_nl_owner_execute", "-o", str(exe)]
            env=dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:disable_coredump=1", UBSAN_OPTIONS="halt_on_error=1")
            with (ARTIFACTS/(tag+".txt")).open("w") as log:
                for cmd in (bridge,command):
                    log.write(shlex.join(cmd)+"\n"); log.flush()
                    built=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,env=env)
                    log.write(built.stdout.decode()); log.flush()
                    self.assertEqual(built.returncode,0,built.stdout.decode())
                for mode in ("init", "mapper", "normal", "init-mismatch", "epoch-mismatch", "uninitialized", "init-guard", "reinit", "deadline", "postnative-guard", "postnative-readonly", "bounded-row", "slot-too-small", "claim-rollback", "owner-only", "shadow", "bad-shadow", "bad-owner-only"):
                    log.write("RUN "+mode+"\n"); log.flush()
                    result=subprocess.run([str(exe),mode],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30,env=env)
                    log.write(result.stdout.decode()+"RETURN %d\n" % result.returncode); log.flush()
                    if mode in ("init-guard", "reinit", "deadline", "postnative-guard", "postnative-readonly"):
                        self.assertEqual(result.returncode,-signal.SIGABRT,result.stdout.decode())
                        entries=[line for line in result.stdout.decode().splitlines() if line.startswith("KF_NATIVE_ENTER")]
                        if mode.startswith("postnative"):
                            self.assertGreaterEqual(len(entries),2)
                            self.assertEqual(len(entries),len(set(entries)),"native head replay")
                    else:
                        self.assertEqual(result.returncode,0,result.stdout.decode())
                        self.assertIn(b"PASS",result.stdout)

    def test_frozen_gnu_live_chain(self):
        import ast
        import hashlib
        import shutil
        directory=ARTIFACTS / "gnu"
        directory.mkdir(parents=True,exist_ok=True)
        frozen=pathlib.Path("/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/phys/module_cu_kfeta.f90")
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),"9533f8600ad060a2e8aa491ed9b7fece633c3d1e2f445e001208cdceed6f1c05")
        shutil.copyfile(frozen,directory/frozen.name)
        (directory/"module_wrf_error.f90").write_text("""module module_wrf_error
contains
subroutine wrf_message(message)
character(*),intent(in)::message
write(*,*) trim(message)
end subroutine
subroutine wrf_error_fatal3(file,line,message)
character(*),intent(in)::file,message
integer,intent(in)::line
write(*,*) trim(file),line,trim(message)
error stop 1
end subroutine
end module
""")
        tree=ast.parse((ROOT/"tests/test_kf_batch.py").read_text())
        fixtures=[node.value.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="fixture" for t in node.targets) and isinstance(node.value,ast.Constant)]
        self.assertEqual(len(fixtures),1)
        sounding=fixtures[0].split("static void real_scalar",1)[1].split("int main(void)",1)[0]
        sounding="static void real_scalar"+sounding
        header=["#define KFB_FIXTURE_ONLY", '#include "test_kf_batch.c"', "#include <math.h>"]
        for name,n in (("init",35),("cps",79)):
            header.append("extern void __module_cu_kfeta_MOD_kf_eta_"+name+"("+",".join(["void *"]*n)+");")
        compute="void wt_kf_fixture_gnu_compute(void **a) { __module_cu_kfeta_MOD_kf_eta_cps("+",".join("a[%d]" % i for i in range(79))+"); }"
        initargs=["f.args[WT_KF_%s]" % name for name in ("RTHCUTEN","RQVCUTEN","RQCCUTEN","RQRCUTEN","RQICUTEN","RQSCUTEN","NCA","W0AVG")]+["&one","&one"]+["f.args[WT_KF_%s]" % name for name in ("SVP1","SVP2","SVP3","SVPT0")]+["&one","&zero","&one"]+["&f.scalar[%d]" % i for i in range(18)]
        init="void wt_kf_fixture_gnu_init(void) { struct fixture f; int one=1,zero=0; sounding(&f,1); __module_cu_kfeta_MOD_kf_eta_init("+",".join(initargs)+"); }"
        export="void wt_kf_fixture_gnu_sounding(void *f,int active) { sounding(f,active); (void)cps; (void)oracle_copy; (void)unchanged; (void)checks; }"
        (directory/"native-adapter.c").write_text("\n".join(header)+"\n"+sounding+compute+init+export)
        abi=[];real=[]
        for name,n in (("INIT",35),("CPS",79)):
            formals=",".join("void *a%d" % i for i in range(n));actual=",".join("a%d" % i for i in range(n))
            symbol="module_cu_kfeta_mp_kf_eta_"+name.lower()+"_";fixture="kf_fixture_"+name.lower()+"_real"
            abi += ["#define KF_%s_FORMALS " % name+formals,"#define KF_%s_ARGV {" % name+actual+"}","#define KF_%s_FROM_ARRAY(a) " % name+",".join("a[%d]" % i for i in range(n)),"extern void "+symbol+"("+formals+");"]
            real.append("extern void "+fixture+"("+formals+"); void "+symbol+"("+formals+") { "+fixture+"("+actual+"); }")
        (directory/"kf-node-live-abi.h").write_text("\n".join(abi)+"\n")
        (directory/"real-abi.c").write_text("\n".join(real)+"\n")
        cc=shlex.split(os.environ.get("CC","cc"))
        flags=["-std=c11","-Wall","-Wextra","-Werror","-O1","-g","-gdwarf-4","-gz=none","-fopenmp","-pthread","-DSTORAGE=4080","-DWT_KF_NODE_GNU_CPS","-DWT_KF_NODE_LOCAL","-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1","-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1","-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1",'-DWT_KF_BUILD_PROFILE_ID="gnu-fixture-frozen-cps"','-DWT_KF_NODE_SHM_DIRECTORY="%s"' % directory,"-I",str(ROOT),"-I",str(ROOT/"tests"),"-I",str(directory)]
        flags += ["-Dmodule_cu_kfeta_mp_%s_=__module_cu_kfeta_MOD_%s" % (name,name) for name in ("ttab","qstab","the0k","alu","rdpr","rdthk","plutop")]
        units=["wt_live_spans","wt_kf_capacity","wt_kf_dispatch","wt_cumulus","wt_kf_live","wt_kf_native","wt_kf_bootstrap","wt_kf_batch","wt_kf_batch_codec","wt_kf_node","wt_node_local","wt_kf_node_live","wt_kf_node_shm","wt_kf_init_bridge"]
        commands=[
            ["/usr/bin/gfortran","-O1","-g","-fcheck=all","-fbacktrace","-c","module_wrf_error.f90","module_cu_kfeta.f90"],
            cc+flags+["-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch","-c",str(ROOT/"wt_kf_bridge.c"),"-o","bridge.o"],
            cc+flags+["-c"]+[str(ROOT/(unit+".c")) for unit in units]+[str(ROOT/"tests/test_kf_node_live.c"),"real-abi.c","native-adapter.c"],
            ["/usr/bin/gfortran","-fopenmp","-pthread"]+[unit+".o" for unit in units]+["test_kf_node_live.o","real-abi.o","native-adapter.o","module_wrf_error.o","module_cu_kfeta.o","bridge.o","-Wl,--wrap=module_cu_kfeta_mp_kf_eta_cps_","-Wl,--wrap=module_cu_kfeta_mp_kf_eta_init_","-Wl,--wrap=wt_nl_try_execute","-Wl,--wrap=wt_nl_owner_execute","-lm","-o","gnu-live"],
            [str(directory/"gnu-live"),"normal"],
        ]
        with (directory/"commands-results.txt").open("w") as log:
            for command in commands:
                log.write(shlex.join(command)+"\n");log.flush()
                env=dict(os.environ,WT_KF_NODE_G0_TRACE="1")
                result=subprocess.run(command,cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,env=env)
                log.write(result.stdout.decode());log.flush()
                self.assertEqual(result.returncode,0,result.stdout.decode())
            self.assertIn(b"owner_remote_committed=",result.stdout)
            self.assertIn(b"PASS KF production",result.stdout)
        (directory/"sha256.txt").write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n" for path in sorted(directory.iterdir()) if path.suffix in (".c",".f90",".o")))
