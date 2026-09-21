
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
