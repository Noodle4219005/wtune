import os
import pathlib
import shlex
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "artifacts/20260913-node-local-physics/kf"

class KFBatchTest(unittest.TestCase):
    def test_batch(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for sanitized in (False, True):
            tag = "batch-sanitized" if sanitized else "batch-strict"
            exe = ARTIFACTS / tag
            cmd = shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g"]
            if sanitized:
                cmd += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
            cmd += ["-I", str(ROOT), str(ROOT / "wt_kf_batch.c"), str(ROOT / "wt_kf_batch_codec.c"), str(ROOT / "tests/test_kf_batch.c"), "-o", str(exe)]
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
            with (ARTIFACTS / (tag + ".txt")).open("w") as log:
                log.write(shlex.join(cmd) + "\n"); log.flush()
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60, env=env)
                result = subprocess.run([str(exe)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, env=env)
                log.write(result.stdout.decode()); log.flush()
                self.assertEqual(result.returncode, 0, result.stdout.decode())
                self.assertIn(b"PASS", result.stdout)

    def test_frozen_gnu_cps(self):
        import hashlib
        import shutil
        closure = ARTIFACTS / "gnu-closure"
        closure.mkdir(parents=True, exist_ok=True)
        source = pathlib.Path("/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/phys/module_cu_kfeta.f90")
        if not source.is_file() or not pathlib.Path("/usr/bin/gfortran").is_file():
            self.skipTest("frozen CPS source or system GNU Fortran unavailable")
        expected = "9533f8600ad060a2e8aa491ed9b7fece633c3d1e2f445e001208cdceed6f1c05"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
        shutil.copyfile(source, closure / source.name)
        (closure / "module_wrf_error.f90").write_text("""module module_wrf_error
use iso_c_binding
integer(c_int), bind(C,name="kf_diag_messages") :: messages=0
integer(c_int), bind(C,name="kf_diag_fatals") :: fatals=0
contains
subroutine wrf_message(message)
character(*), intent(in) :: message
messages=messages+1
write(*,*) trim(message)
end subroutine
subroutine wrf_error_fatal3(file,line,message)
character(*), intent(in) :: file,message
integer, intent(in) :: line
fatals=fatals+1
write(*,*) trim(file),line,trim(message)
error stop 1
end subroutine
end module
""")
        callback = "extern void __module_cu_kfeta_MOD_kf_eta_cps(" + ",".join(["void *"] * 79) + ");\n"
        callback += "static int native(void **a,void *ctx) { (void)ctx; __module_cu_kfeta_MOD_kf_eta_cps(" + ",".join("a[%d]" % i for i in range(79)) + "); return 0; }\n"
        fixture = r'''#define _POSIX_C_SOURCE 200809L
#define KFB_FIXTURE_ONLY
#define STORAGE (6*5*34*4)
#include "test_kf_batch.c"
#include "wt_kf_node.h"
#include <math.h>
#include <sys/wait.h>
#include <unistd.h>
extern void __module_cu_kfeta_MOD_kf_lutab(float *,float *,float *,float *);
extern int kf_diag_messages,kf_diag_fatals;
CALLBACK
static void real_scalar(struct fixture *f,unsigned a,float x) { memcpy(&f->scalar[a],&x,4); }
static void sounding(struct fixture *f,int active)
{
    unsigned a; int j,k,i;
    setup(f); f->scalar[WT_KF_KME]=33; f->scalar[WT_KF_KDE]=33; f->scalar[WT_KF_KTE]=32;
    f->scalar[WT_KF_STEPCU]=1; f->scalar[WT_KF_KTAU]=1;
    f->scalar[WT_KF_KF_EDRATES]=active;
    for(a=WT_KF_F_QV;a<=WT_KF_F_QS;++a) f->scalar[a]=1;
    if(active) for(a=WT_KF_UDR_KF;a<=WT_KF_TIMEC_KF;++a) { f->args[a]=f->arrays[a]; f->cap[a]=STORAGE; }
    real_scalar(f,WT_KF_DT,30); real_scalar(f,WT_KF_DX,12000); real_scalar(f,WT_KF_CUDT,0.5f);
    real_scalar(f,WT_KF_XLV0,3.15e6f); real_scalar(f,WT_KF_XLV1,2369);
    real_scalar(f,WT_KF_XLS0,2.905e6f); real_scalar(f,WT_KF_XLS1,259);
    real_scalar(f,WT_KF_CP,1004); real_scalar(f,WT_KF_R,287); real_scalar(f,WT_KF_G,9.81f);
    real_scalar(f,WT_KF_EP1,0.608f); real_scalar(f,WT_KF_EP2,0.622f);
    real_scalar(f,WT_KF_SVP1,0.6112f); real_scalar(f,WT_KF_SVP2,17.67f);
    real_scalar(f,WT_KF_SVP3,29.65f); real_scalar(f,WT_KF_SVPT0,273.15f);
    for(j=0;j<5;++j) for(k=0;k<34;++k) for(i=0;i<6;++i) {
        size_t q=(size_t)i+6u*((size_t)k+34u*(size_t)j);
        float t=300-3.0f*k,p=100000*expf(-k*500.0f/8000),pi=powf(p/100000,0.2854f);
        f->arrays[WT_KF_T][q]=t; f->arrays[WT_KF_PCPS][q]=p; f->arrays[WT_KF_PI][q]=pi;
        f->arrays[WT_KF_TH][q]=t/pi; f->arrays[WT_KF_RHO][q]=p/(287*t);
        f->arrays[WT_KF_QV][q]=0.008f*expf(-k/7.0f);
        f->arrays[WT_KF_U][q]=5; f->arrays[WT_KF_V][q]=2;
        f->arrays[WT_KF_W][q]=0.2f; f->arrays[WT_KF_W0AVG][q]=0.1f;
        f->arrays[WT_KF_DZ8W][q]=500;
    }
    for(i=0;i<30;++i) f->arrays[WT_KF_NCA][i]=-100;
}
int main(void)
{
    int active; float svp1=.6112f,svp2=17.67f,svp3=29.65f,svpt0=273.15f;
    (void)cps;
    __module_cu_kfeta_MOD_kf_lutab(&svp1,&svp2,&svp3,&svpt0);
    for(active=0;active<2;++active) {
        struct fixture f,expected,before;
        struct wt_nl_service service; struct wt_nl_config cfg={2,2,65536};
        struct wt_kf_node_call call={0}; struct wt_nl_ticket ticket; enum wt_kf_status bs;
        struct wt_nl_signature sig={WT_KF_NODE_PHASE,19,WT_KF_BATCH_SCHEMA,4,7,1,23};
        struct wt_kf_node_context owner={&service,{0},0,0,BUDGET,native,NULL,0};
        struct wt_nl_counters counters;
        char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/native-service-XXXXXX";
        int fd=mkstemp(path),status; pid_t pid; size_t mapbytes;
        CHECK(fd>=0); CHECK(wt_nl_init_fd(fd,&cfg,&service)==WT_NL_OK); mapbytes=service.mapping_bytes;
        sounding(&f,active); oracle_copy(&expected,&f); before=f;
        CHECK(native(expected.args,NULL)==0);
        CHECK(!unchanged(&before,&expected));
        owner.signature=sig;
        CHECK(wt_kf_node_publish(&owner,f.args,f.cap,1,&call,&bs)==WT_NL_OK);
        pid=fork(); CHECK(pid>=0);
        if(pid==0) {
            struct wt_nl_service attached; struct wt_kf_node_context helper=owner;
            wt_nl_detach(&service);
            if(wt_nl_attach_fd(fd,mapbytes,&attached)!=WT_NL_OK) _exit(21);
            helper.service=&attached; helper.rank=1;
            if(wt_kf_node_progress(&helper,(uint64_t)getpid(),&ticket)!=WT_NL_OK) _exit(22);
            if(kf_diag_fatals) _exit(23);
            wt_nl_detach(&attached); close(fd); _exit(0);
        }
        CHECK(waitpid(pid,&status,0)==pid); CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
        CHECK(unchanged(&f,&before)); CHECK(wt_kf_node_collect(&owner,&call)==WT_NL_OK);
        CHECK(unchanged(&f,&expected)); CHECK(wt_kf_node_finish(&owner,&call)==WT_NL_OK);
        CHECK(wt_nl_get_counters(&service,&counters)==WT_NL_OK);
        CHECK(counters.helper_executed==1 && counters.committed==1 && counters.failed==0);
        wt_kf_node_destroy(&call); wt_nl_detach(&service); close(fd);
    }
    CHECK(kf_diag_fatals==0);
    printf("PASS frozen GNU CPS full-state bitwise, EDRATES 0/1, other-process helper, %d checks, diagnostics %d/%d\n",checks,kf_diag_messages,kf_diag_fatals);
    return 0;
}
'''
        (closure / "native_fixture.c").write_text(fixture.replace("CALLBACK", callback))
        commands = [
            ["/usr/bin/gfortran", "--version"],
            ["/usr/bin/gfortran", "-O1", "-g", "-fcheck=all", "-fbacktrace", "-J", str(closure), "-I", str(closure), "-c", "module_wrf_error.f90", "module_cu_kfeta.f90"],
            shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g", "-gdwarf-4", "-gz=none", "-I", str(ROOT), "-I", str(ROOT / "tests"), "-c", str(ROOT / "wt_kf_batch.c"), str(ROOT / "wt_kf_batch_codec.c"), str(ROOT / "wt_kf_node.c"), str(ROOT / "wt_node_local.c"), "native_fixture.c"],
            ["/usr/bin/gfortran", "module_wrf_error.o", "module_cu_kfeta.o", "wt_kf_batch.o", "wt_kf_batch_codec.o", "wt_kf_node.o", "wt_node_local.o", "native_fixture.o", "-pthread", "-lm", "-o", "native-fixture"],
            [str(closure / "native-fixture")],
        ]
        with (closure / "commands-results.txt").open("w") as log:
            for cmd in commands:
                log.write(shlex.join(cmd) + "\n"); log.flush()
                result = subprocess.run(cmd, cwd=closure, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                log.write(result.stdout.decode()); log.flush()
                self.assertEqual(result.returncode, 0, result.stdout.decode())
            self.assertIn(b"PASS frozen GNU CPS", result.stdout)
        (closure / "sha256.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in sorted(closure.iterdir()) if p.suffix in (".c", ".f90", ".o")))
