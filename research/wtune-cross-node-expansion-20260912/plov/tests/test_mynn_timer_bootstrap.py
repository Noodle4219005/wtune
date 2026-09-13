"""Actual timer constructor/report with the MYNN bootstrap boundary mocked."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from plov.tests.test_kf_timer_bootstrap import MOCK as KF_MOCK

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/"artifacts/20260913-node-local-physics/mynn/production-live"
CC="/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-cc"
MOCK=KF_MOCK.replace("WT_CUMULUS_LIVE_SPANS","WT_MYNN_NODE_LOCAL").replace("wt_kf_bootstrap","wt_mynn_bootstrap")

class MynnTimerBootstrapTests(unittest.TestCase):
    def test_constructor_macro_off_and_requested_stage_state(self):
        ART.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix="timer-",dir=ART))
        mock=work/"mock.c";mock.write_text(MOCK)
        with (work/"commands.txt").open("w") as log:
            for enabled in (False,True):
                exe=work/("on" if enabled else "off")
                cmd=[CC,"-std=c11","-Wall","-Wextra","-Werror","-O1","-I",str(ROOT/"plov")]
                if enabled:cmd.append("-DWT_MYNN_NODE_LOCAL")
                cmd += [str(ROOT/"plov/wtimer.c"),str(mock),"-o",str(exe)]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                log.write(" ".join(cmd)+"\n"+p.stdout+p.stderr);log.flush()
                self.assertEqual(p.returncode,0,p.stderr)
                for requested in (0,1):
                    for state,label in ((-1,"unchecked"),(0,"refused/off"),(1,"enabled")):
                        directory=Path(tempfile.mkdtemp(prefix="report-",dir=work))
                        env={k:os.environ[k] for k in ("PATH","LD_LIBRARY_PATH") if k in os.environ}
                        env.update(WTIMER_DIR=str(directory),WTIMER_STAGE_PBL=str(requested),WTIMER_STAGE_GRID="4",MOCK_BOOTSTRAP_STATE=str(state))
                        p=subprocess.run([str(exe)],cwd=directory,env=env,capture_output=True,text=True,timeout=10)
                        log.write(p.stdout+p.stderr);log.flush();self.assertEqual(p.returncode,0,p.stderr)
                        report=(directory/"wtimer.unknown.csv").read_text()
                        effective=next(x for x in report.splitlines() if x.startswith("# stage_effective "))
                        fields=dict(x.split("=",1) for x in effective.split()[2:])
                        self.assertEqual(fields["WTIMER_STAGE_PBL"],("disabled" if not requested else label) if enabled else ("refused" if requested else "disabled"))
                        if enabled:self.assertIn("constructor calls=1 requested=%d grid=4"%requested,p.stdout)
                        else:self.assertIn("no bootstrap symbols linked",p.stdout)

    def test_native_fortran_scope_adapter(self):
        ART.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix="scope-",dir=ART))
        fc=str(Path(CC).with_name("x86_64-conda-linux-gnu-gfortran"))
        source=work/"fixture.f90"
        source.write_text('''module module_domain_type
use iso_c_binding
type, bind(C) :: domain
integer(c_int) :: id,itimestep
end type
end module
module module_configure
use iso_c_binding
type, bind(C) :: grid_config_rec_type
integer(c_int) :: bl_pbl_physics
end type
end module
module module_state_description
integer,parameter :: MYNNPBLSCHEME=5
end module
''')
        harness=work/"main.c"
        harness.write_text('''#include <assert.h>
extern void wt_mynn_native_scope_(void*,void*,int*,int*,int*);
int main(void) { int g[2]={4,27},cfg=5,id=0,e=0,s=2;
wt_mynn_native_scope_(g,&cfg,&id,&e,&s);
#ifdef WIDE
assert(s==2 && id==0 && e==0);
#else
assert(s==0 && id==4 && e==27);cfg=1;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
cfg=5;g[1]=0;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
g[0]=-1;g[1]=27;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
#endif
return 0; }
''')
        with (work/"commands.txt").open("w") as log:
            for wide in (False,True):
                tag="wide" if wide else "normal"
                flags=["-fdefault-real-8"] if wide else []
                for src,out in ((source,work/"fixture.o"),(ROOT/"plov/wt_mynn_native_scope.f90",work/"scope.o")):
                    cmd=[fc,*flags,"-J",str(work),"-I",str(work),"-c",str(src),"-o",str(out)]
                    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);log.write(" ".join(cmd)+"\n"+p.stderr);self.assertEqual(p.returncode,0,p.stderr)
                cmd=[CC,"-std=c11","-Wall","-Wextra","-Werror",*(["-DWIDE"] if wide else []),"-c",str(harness),"-o",str(work/"main.o")]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);self.assertEqual(p.returncode,0,p.stderr)
                cmd=[fc,str(work/"main.o"),str(work/"fixture.o"),str(work/"scope.o"),"-o",str(work/tag)]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);self.assertEqual(p.returncode,0,p.stderr)
                p=subprocess.run([str(work/tag)],capture_output=True,text=True,timeout=10);log.write(p.stdout+p.stderr+"RETURN %d\n"%p.returncode);self.assertEqual(p.returncode,0,p.stderr)

if __name__=="__main__":unittest.main()
