"""Exercise the timer constructor/report at the SFCLAY bootstrap boundary."""

import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/20260913-node-local-physics/sfclay/timer"

MOCK = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int wt_micro_stage_state = -1, wt_rad_stage_state = -1;
uint64_t wt_micro_calls, wt_micro_staged, wt_micro_refused;
uint64_t wt_micro_share_out, wt_micro_share_in;
uint64_t wt_micro_share_wait_ticks, wt_micro_share_work_ticks;
uint64_t wt_micro_rows_out, wt_micro_chunks_in, wt_micro_sig_mismatch;
uint64_t wt_rad_calls, wt_rad_steps, wt_rad_staged, wt_rad_refused;
uint64_t wt_rad_share_out, wt_rad_share_in, wt_rad_share_wait_us;
uint64_t wt_rad_sig_mismatch;
uint64_t wt_lewi_lend, wt_lewi_acq, wt_lewi_acq_cores, wt_lewi_miss;
uint64_t wt_lewi_repin, wt_lewi_stolen, wt_lewi_short;
uint64_t wt_lewi_probe_n, wt_lewi_probe_sum, wt_lewi_probe_hist[8];
void wt_lewi_init(void) {}
void wt_lewi_borrow(void) {}
void wt_lewi_return(void) {}
void wt_micro_share_finalize(void) {}
void wt_lewi_trace_dump(const char *dir, const char *rank)
{ (void)dir; (void)rank; }

#ifdef WT_SFCLAY_NODE_LOCAL
#include "wt_sfclay_bootstrap.h"
static int start_calls, seen_requested=-99, seen_grid=-99;
int wt_sfclay_bootstrap_start(int requested,int grid_selector)
{
    ++start_calls; seen_requested=requested; seen_grid=grid_selector;
    return requested;
}
int wt_sfclay_bootstrap_state(void)
{
    const char *value=getenv("MOCK_SF_STATE"); return value ? atoi(value) : -1;
}
void wt_sfclay_bootstrap_dump(FILE *stream)
{
    fprintf(stream,"# mock_sfclay requested=%d grid=%d state=%d calls=%d\n",
            seen_requested,seen_grid,wt_sfclay_bootstrap_state(),start_calls);
}
#endif

int main(void)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    printf("constructor calls=%d requested=%d grid=%d\n",
           start_calls,seen_requested,seen_grid);
#else
    puts("guard off: no SFCLAY bootstrap symbols linked");
#endif
    return 0;
}
"""


class SfclayTimerBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        mock = ARTIFACT / "mock.c"
        mock.write_text(MOCK)
        cls.executables = {}
        cc = shlex.split(os.environ.get("CC", "cc"))
        for enabled in (False, True):
            executable = ARTIFACT / ("guard-on" if enabled else "guard-off")
            command = [*cc, *shlex.split(os.environ.get("CFLAGS", "")),
                       "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Iplov"]
            if enabled:
                command += ["-DWT_SFCLAY_NODE_LOCAL"]
            command += ["plov/wtimer.c", str(mock), "-o", str(executable)]
            result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True,
                                    timeout=60)
            (ARTIFACT / f"compile-{'on' if enabled else 'off'}.txt").write_text(
                " ".join(shlex.quote(x) for x in command)+"\n"+
                result.stdout+result.stderr)
            if result.returncode:
                raise AssertionError(result.stdout+result.stderr)
            cls.executables[enabled]=executable

    def run_timer(self,enabled,requested,state=-1,grid="0"):
        run_dir=ARTIFACT / f"run-{int(enabled)}-{requested}-{state}-{grid}"
        run_dir.mkdir(exist_ok=True)
        env={key:os.environ[key] for key in ("PATH","LD_LIBRARY_PATH")
             if key in os.environ}
        env.update(WTIMER_DIR=str(run_dir),WTIMER_STAGE_SURFACE=str(requested),
                   MOCK_SF_STATE=str(state))
        if grid is not None:
            env["WTIMER_STAGE_GRID"]=grid
        result=subprocess.run([str(self.executables[enabled])],cwd=run_dir,
                              env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,universal_newlines=True,
                              timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=(run_dir/"wtimer.unknown.csv").read_text()
        effective=next(line for line in report.splitlines()
                       if line.startswith("# stage_effective "))
        fields=dict(field.split("=",1) for field in effective.split()[2:])
        return result.stdout,report,fields["WTIMER_STAGE_SURFACE"]

    def test_macro_off_preserves_refusal(self):
        for requested in (0,1):
            stdout,report,effective=self.run_timer(False,requested)
            self.assertIn("no SFCLAY bootstrap symbols linked",stdout)
            self.assertEqual(effective,"refused" if requested else "disabled")
            self.assertEqual("STAGE_SURFACE requested but not implemented" in report,
                             bool(requested))

    def test_macro_on_forwards_request_grid_and_state(self):
        for requested in (0,1):
            for state,label in ((-1,"unchecked"),(0,"refused/off"),(1,"enabled")):
                for grid in ("0","2"):
                    with self.subTest(requested=requested,state=state,grid=grid):
                        stdout,report,effective=self.run_timer(True,requested,state,grid)
                        self.assertEqual(stdout.strip(),
                            f"constructor calls=1 requested={requested} grid={grid}")
                        self.assertEqual(effective,label if requested else "disabled")
                        self.assertIn(f"# mock_sfclay requested={requested} grid={grid} "
                                      f"state={state} calls=1\n",report)

    def test_grid_default_and_malformed_fallback(self):
        for grid in (None,"","bad","-1","2junk"):
            with self.subTest(grid=grid):
                stdout,report,_=self.run_timer(True,1,grid=grid)
                self.assertEqual(stdout.strip(),
                                 "constructor calls=1 requested=1 grid=1")
                self.assertEqual("malformed, treated as unset" in report,
                                 grid in ("bad","-1","2junk"))


if __name__ == "__main__":
    unittest.main()
