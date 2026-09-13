/* wtimer -- collection side.  See wtimer.S for the interception mechanism.
 *
 * Fills the blind spot the BENCH timers cannot reach: dyn_em/module_first_rk_step_part1.F:3-4
 * defines BENCH_START(A) and BENCH_END(A) to nothing, unconditionally, so radiation, PBL,
 * surface and cumulus carry no timer at all in any -DBENCH build.  That is 45.4% of solve.
 *
 * Slot 0 is microphysics, which -DBENCH *does* time.  It exists purely as a self-check:
 * if wtimer's micro column does not reproduce micro_driver_tim, nothing else here is
 * trustworthy either.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NSYM     6
#define MAXCALL  8192

static const char *SYM[NSYM] = {
    "micro", "radiation", "pre_radiation", "pbl", "surface", "cumulus"
};

static uint64_t  samp[NSYM][MAXCALL];
static uint64_t  wsnap[MAXCALL];   /* cumulative MPI-wait cycles, sampled once per step   */
static uint64_t  csnap[MAXCALL];   /* cumulative MPI-wait call count, same instant        */
static uint64_t  wait_cycles, wait_calls;
static uint64_t  kind_n[7], kind_c[7];
static const char *KIND[7] = {"Wait","Waitall","Waitany","Barrier","Allreduce","Bcast","Sendrecv"};
void wt_add_kind(int k, uint64_t cycles) { if (k>=0 && k<7) { kind_n[k]++; kind_c[k]+=cycles; } }
static unsigned  ncall[NSYM];
static uint64_t  overflow[NSYM];
static uint64_t  tsc0;
static int       nthr_seen;        /* omp_get_max_threads() sampled at first driver call */
static int       ncpu_owned;       /* affinity mask size sampled in the CONSTRUCTOR       */
static int       nproc_omp;        /* omp_get_num_procs() at first driver call            */
static int       micro_thr;        /* threads to use INSIDE microphysics; 0 = leave alone */
static int       base_thr;         /* what to restore afterwards                          */
static int       my_cpu0 = -1;     /* first core this rank owns, -1 = untouched            */
static int       my_ncpu = 0;      /* how many, exclusive to this rank                     */
static int       base_cpu0 = -1, base_ncpu = 0;   /* the even 2-core share                 */
static int       dyn_mode;         /* swap masks around micro instead of pinning once      */
static int       ph_base, ph_micro;               /* phase-wise plan: team sizes           */
static int       ph_core[64];                     /* thread i -> this core, pinned once    */
static int       stage_micro = 1, stage_rad;
static int       stage_cumulus, stage_pbl, stage_surface;
static int       stage_grid = 1, stage_grid_malformed;
static char      stage_grid_text[128] = "unset";

static int wt_stage_flag(const char *name, int default_value)
{
    const char *e = getenv(name);
    if (!e || !*e) return default_value;
    if (e[0] == '1' && !e[1]) return 1;
    if (e[0] == '0' && !e[1]) return 0;
    return default_value;
}

/* A missing grid selector means grid 1.  Keep this parser strict: atoi() would
 * accept a valid prefix and silently turn malformed values into a different
 * grid. */
static int wt_stage_grid_parse(const char *text, int *malformed)
{
    char *end;
    long value;

    *malformed = 0;
    if (!text || !*text) return 1;
    if (text[0] < '0' || text[0] > '9') {
        *malformed = 1;
        return 1;
    }
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0 || value > INT_MAX) {
        *malformed = 1;
        return 1;
    }
    return (int)value;
}

int wt_stage_grid_value(void) { return stage_grid; }

int wt_stage_micro_enabled(void) { return stage_micro; }
int wt_stage_rad_enabled(void) { return stage_rad; }

extern int wt_micro_stage_state, wt_rad_stage_state;

/* Give this rank cores [c0, c0+n) and n threads, one thread per core.
 *
 * The team must be re-pinned, not just the master. sched_setaffinity(0,...) moves only the
 * CALLING thread, and OpenMP workers inherit their mask when the pool is first created -- so
 * widening the master's mask before microphysics left the workers on the original two cores
 * and four threads ran on two cores. That read as micro max 18.87 s against a 10.34 s
 * baseline: the cost of one thread, which is exactly what it was. Opening a parallel region
 * here makes every thread pin itself, and any thread the runtime has to create inherits the
 * master's already-widened mask. */
static void set_cpus(int c0, int n)
{
    cpu_set_t d; CPU_ZERO(&d);
    for (int c = c0; c < c0 + n; c++) CPU_SET(c, &d);
    if (sched_setaffinity(0, sizeof d, &d) != 0) return;
#ifdef _OPENMP
    omp_set_num_threads(n);
    if (n > 1) {
        #pragma omp parallel num_threads(n)
        {
            cpu_set_t m; CPU_ZERO(&m);
            CPU_SET(c0 + (omp_get_thread_num() % n), &m);
            sched_setaffinity(0, sizeof m, &m);
        }
    }
#endif
}
static struct timespec wall0;
static int       armed;

static uint64_t rdtsc_now(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* Called from every intercepted MPI call in wtmpi.c. Kept to two adds: WRF issues tens of
 * thousands of these per step. */
void wt_add_wait(uint64_t cycles, int calls)
{
    wait_cycles += cycles;
    wait_calls  += (uint64_t)calls;
}

/* Called from the assembly PROLOGUE, before __real_ runs.
 *
 * This is the actuator. microphysics_driver opens its own !$OMP PARALLEL DO
 * (phys/module_microphysics_driver.F:900) and uses whatever team size is current, so setting
 * the ICV here changes only that phase and nothing else in the timestep. The call site is
 * outside every OpenMP region (dyn_em/solve_em.F:3881), so this runs single-threaded and
 * needs no synchronisation. */
void wt_lewi_init(void);
void wt_lewi_borrow(void);
void wt_lewi_return(void);
extern uint64_t wt_lewi_lend, wt_lewi_acq, wt_lewi_acq_cores, wt_lewi_miss, wt_lewi_repin,
                wt_lewi_stolen, wt_lewi_short;
void wt_lewi_trace_dump(const char *dir, const char *rank);
extern uint64_t wt_micro_calls, wt_micro_staged, wt_micro_refused;
extern uint64_t wt_micro_share_out, wt_micro_share_in;
extern uint64_t wt_micro_share_wait_ticks, wt_micro_share_work_ticks;
extern uint64_t wt_micro_rows_out, wt_micro_chunks_in;
extern uint64_t wt_micro_sig_mismatch;
extern void wt_micro_share_finalize(void);
extern uint64_t wt_rad_calls, wt_rad_steps, wt_rad_staged, wt_rad_refused;
extern uint64_t wt_rad_share_out, wt_rad_share_in, wt_rad_share_wait_us;
extern uint64_t wt_rad_sig_mismatch;
extern uint64_t wt_lewi_probe_n, wt_lewi_probe_sum, wt_lewi_probe_hist[8];

void wt_pre(int idx)
{
    if (idx != 0) return;
    if (getenv("WTIMER_LEWI")) { wt_lewi_borrow(); return; }
#ifdef _OPENMP
    if (ph_micro > 0) { omp_set_num_threads(ph_micro); return; }   /* two ICV writes, no syscall */
#endif
    if (dyn_mode && my_ncpu > 0) set_cpus(my_cpu0, my_ncpu);
#ifdef _OPENMP
    else if (micro_thr > 0) omp_set_num_threads(micro_thr);
#endif
}

/* Called from the assembly epilogue.  Must stay cheap: it runs on every driver call. */
void wt_record(int idx, uint64_t cycles)
{
    if (idx < 0 || idx >= NSYM) return;
    /* Sampled here rather than at init: OMP_NUM_THREADS can be honoured or silently
     * overridden by the launcher's pinning (I_MPI_PIN_DOMAIN=core once put two threads on
     * one core and cost 3.6x while every other gate passed clean). What matters is the value
     * in force when the driver actually runs, and how many CPUs this rank really owns. */
    if (!nthr_seen) {
#ifdef _OPENMP
        nthr_seen = omp_get_max_threads();
        nproc_omp = omp_get_num_procs();
#else
        nthr_seen = nproc_omp = -1;
#endif
    }
#ifdef _OPENMP
    if (idx == 0 && ph_micro > 0) omp_set_num_threads(ph_base);
    else
#endif
    if (idx == 0 && dyn_mode && base_ncpu > 0) set_cpus(base_cpu0, base_ncpu);
#ifdef _OPENMP
    else if (idx == 0 && micro_thr > 0) omp_set_num_threads(base_thr);
#endif
    /* microphysics_driver is called exactly once per timestep, so its call index doubles as
     * the step number and gives the wait accumulator a step boundary for free. */
    if (idx == 0 && getenv("WTIMER_LEWI")) wt_lewi_return();
    if (idx == 0 && ncall[0] < MAXCALL) { wsnap[ncall[0]] = wait_cycles; csnap[ncall[0]] = wait_calls; }
    if (ncall[idx] < MAXCALL) samp[idx][ncall[idx]++] = cycles;
    else                      overflow[idx]++;
}

static const char *rank_str(void);

static const char *rank_str(void)
{
    const char *e;
    if ((e = getenv("PMI_RANK")))              return e;   /* Intel MPI / Hydra */
    if ((e = getenv("SLURM_PROCID")))          return e;   /* srun              */
    if ((e = getenv("OMPI_COMM_WORLD_RANK")))  return e;
    return "unknown";
}

/* TSC is calibrated against CLOCK_MONOTONIC over the whole run rather than over a short
 * sleep at startup: the interval is minutes, so the ratio is far more stable, and it costs
 * two clock reads instead of a calibration loop. */
static void wt_dump(void)
{
    struct timespec w1;
    uint64_t tsc1 = rdtsc_now();
    clock_gettime(CLOCK_MONOTONIC, &w1);

    double wall = (w1.tv_sec - wall0.tv_sec) + 1e-9 * (w1.tv_nsec - wall0.tv_nsec);
    double spc  = (tsc1 > tsc0 && wall > 0) ? wall / (double)(tsc1 - tsc0) : 0.0;

    const char *dir = getenv("WTIMER_DIR");
    char path[4096];
    snprintf(path, sizeof path, "%s/wtimer.%s.csv", dir ? dir : ".", rank_str());
    FILE *f = fopen(path, "w");
    if (!f) { wt_micro_share_finalize(); return; }

    unsigned n = 0;
    for (int s = 0; s < NSYM; s++) if (ncall[s] > n) n = ncall[s];

    fprintf(f, "# wtimer: per-call seconds of WRF physics drivers, via -Wl,--wrap (no source change)\n");
    fprintf(f, "# rank=%s tsc_hz=%.6g wall_s=%.6f\n", rank_str(), spc > 0 ? 1.0 / spc : 0.0, wall);
    if (stage_grid_malformed)
        fprintf(f, "# stage_grid=%s (malformed, treated as unset)\n", stage_grid_text);
    else
        fprintf(f, "# stage_grid=%s\n", stage_grid_text);
    fprintf(f, "# stage_flags WTIMER_STAGE_MICRO=%d WTIMER_STAGE_RAD=%d WTIMER_STAGE_CUMULUS=%d"
               " WTIMER_STAGE_PBL=%d WTIMER_STAGE_SURFACE=%d\n",
            stage_micro, stage_rad, stage_cumulus, stage_pbl, stage_surface);
    fprintf(f, "# stage_effective WTIMER_STAGE_MICRO=%s WTIMER_STAGE_RAD=%s"
               " WTIMER_STAGE_CUMULUS=%s WTIMER_STAGE_PBL=%s WTIMER_STAGE_SURFACE=%s\n",
            stage_micro ? (wt_micro_stage_state > 0 ? "enabled" :
                           wt_micro_stage_state == 0 ? "refused/off" : "unchecked") : "disabled",
            stage_rad ? (wt_rad_stage_state > 0 ? "enabled" :
                         wt_rad_stage_state == 0 ? "refused/off" : "unchecked") : "disabled",
            stage_cumulus ? "refused" : "disabled",
            stage_pbl ? "refused" : "disabled",
            stage_surface ? "refused" : "disabled");
    if (stage_cumulus) fputs("# STAGE_CUMULUS requested but not implemented (refused)\n", f);
    if (stage_pbl) fputs("# STAGE_PBL requested but not implemented (refused)\n", f);
    if (stage_surface) fputs("# STAGE_SURFACE requested but not implemented (refused)\n", f);
    fprintf(f, "# micro_thr=%d base_thr=%d base_cpu=%d+%d micro_cpu=%d+%d mode=%s\n",
            micro_thr, base_thr, base_cpu0, base_ncpu, my_cpu0, my_ncpu, dyn_mode ? "dynamic" : "static");
    fprintf(f, "# alloc_ncpu=%d ph_base=%d ph_micro=%d\n",
            my_ncpu > 0 ? my_ncpu : ncpu_owned, ph_base, ph_micro);
    fprintf(f, "# omp_max_threads=%d omp_num_procs=%d cpus_at_init=%d omp_num_threads_env=%s\n",
            nthr_seen, nproc_omp, ncpu_owned,
            getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "unset");
    fprintf(f, "# ncall:");
    for (int s = 0; s < NSYM; s++) fprintf(f, " %s=%u", SYM[s], ncall[s]);
    fprintf(f, "\n# overflow:");
    for (int s = 0; s < NSYM; s++) fprintf(f, " %s=%llu", SYM[s], (unsigned long long)overflow[s]);
    fprintf(f, "\n# micro_calls=%llu micro_staged=%llu micro_refused=%llu"
               " micro_share_out=%llu micro_share_in=%llu micro_share_wait_ms=%.6f micro_share_work_ms=%.6f"
               " micro_rows_out=%llu micro_chunks_in=%llu micro_sig_mismatch=%llu\n",
            (unsigned long long)wt_micro_calls, (unsigned long long)wt_micro_staged,
            (unsigned long long)wt_micro_refused, (unsigned long long)wt_micro_share_out,
            (unsigned long long)wt_micro_share_in,
            1000.0 * spc * (double)wt_micro_share_wait_ticks,
            1000.0 * spc * (double)wt_micro_share_work_ticks,
            (unsigned long long)wt_micro_rows_out, (unsigned long long)wt_micro_chunks_in,
            (unsigned long long)wt_micro_sig_mismatch);
    fprintf(f, "# rad_calls=%llu rad_steps=%llu rad_staged=%llu rad_refused=%llu"
            " rad_share_out=%llu rad_share_in=%llu rad_wait_us=%llu rad_sig_mismatch=%llu\n",
            (unsigned long long)wt_rad_calls, (unsigned long long)wt_rad_steps,
            (unsigned long long)wt_rad_staged, (unsigned long long)wt_rad_refused,
            (unsigned long long)wt_rad_share_out, (unsigned long long)wt_rad_share_in,
            (unsigned long long)wt_rad_share_wait_us,
            (unsigned long long)wt_rad_sig_mismatch);
    fprintf(f, "# probe_n=%llu probe_sum=%llu probe_hist=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
            (unsigned long long)wt_lewi_probe_n, (unsigned long long)wt_lewi_probe_sum,
            (unsigned long long)wt_lewi_probe_hist[0],(unsigned long long)wt_lewi_probe_hist[1],
            (unsigned long long)wt_lewi_probe_hist[2],(unsigned long long)wt_lewi_probe_hist[3],
            (unsigned long long)wt_lewi_probe_hist[4],(unsigned long long)wt_lewi_probe_hist[5],
            (unsigned long long)wt_lewi_probe_hist[6],(unsigned long long)wt_lewi_probe_hist[7]);
    fprintf(f, "# lewi_lend=%llu lewi_acq=%llu lewi_acq_cores=%llu lewi_miss=%llu lewi_repin=%llu"
               " lewi_stolen=%llu lewi_short=%llu\n",
            (unsigned long long)wt_lewi_lend, (unsigned long long)wt_lewi_acq,
            (unsigned long long)wt_lewi_acq_cores, (unsigned long long)wt_lewi_miss,
            (unsigned long long)wt_lewi_repin, (unsigned long long)wt_lewi_stolen,
            (unsigned long long)wt_lewi_short);
    wt_lewi_trace_dump(dir, rank_str());
    fprintf(f, "# mpi_wait_total_s=%.6f mpi_wait_calls=%llu\n",
            spc * (double)wait_cycles, (unsigned long long)wait_calls);
    fprintf(f, "# by_kind:");
    for (int k = 0; k < 7; k++)
        fprintf(f, " %s=%llu/%.4fs", KIND[k], (unsigned long long)kind_n[k], spc*(double)kind_c[k]);
    fputc('\n', f);
    fprintf(f, "call");
    for (int s = 0; s < NSYM; s++) fprintf(f, ",%s", SYM[s]);
    fputs(",mpi_wait,mpi_calls\n", f);

    for (unsigned i = 0; i < n; i++) {
        fprintf(f, "%u", i);
        for (int s = 0; s < NSYM; s++) {
            if (i < ncall[s]) fprintf(f, ",%.9f", spc * (double)samp[s][i]);
            else              fputs(",", f);
        }
        /* per-step wait = difference of consecutive snapshots */
        if (i < ncall[0] && i > 0)
            fprintf(f, ",%.9f,%llu", spc * (double)(wsnap[i] - wsnap[i-1]),
                    (unsigned long long)(csnap[i] - csnap[i-1]));
        else fputs(",,", f);
        fputc('\n', f);
    }
    fclose(f);
    wt_micro_share_finalize();
}

__attribute__((constructor))
static void wt_init(void)
{
    if (armed) return;
    armed = 1;
    stage_micro = wt_stage_flag("WTIMER_STAGE_MICRO", 1);
    stage_rad = wt_stage_flag("WTIMER_STAGE_RAD", 0);
    stage_cumulus = wt_stage_flag("WTIMER_STAGE_CUMULUS", 0);
    stage_pbl = wt_stage_flag("WTIMER_STAGE_PBL", 0);
    stage_surface = wt_stage_flag("WTIMER_STAGE_SURFACE", 0);
    {
        const char *grid = getenv("WTIMER_STAGE_GRID");
        stage_grid = wt_stage_grid_parse(grid, &stage_grid_malformed);
        if (grid && *grid)
            snprintf(stage_grid_text, sizeof stage_grid_text, "%s", grid);
        else
            snprintf(stage_grid_text, sizeof stage_grid_text, "unset");
    }
    clock_gettime(CLOCK_MONOTONIC, &wall0);
    tsc0 = rdtsc_now();
    /* WTIMER_CPU: four integers per line -- "base_first base_n micro_first micro_n".
     *
     * Two allocations per rank, both EXCLUSIVE inside the rank's 14-core NUMA domain. The
     * first implementation gave all 7 ranks of a domain the same 14-core mask and let the OS
     * arbitrate; with thread counts left uniform that alone cost +10.8% (arm B, job 1068554),
     * and with them varied it cost +64% while making the micro imbalance WORSE (2.41x ->
     * 3.55x), because an extra thread bought contention rather than capacity.
     *
     * base  = the even split, 2 cores each, what an unmodified run gets.
     * micro = the cost-weighted split, sized to this rank's microphysics work.
     *
     * WTIMER_MODE=static applies the micro split for the whole run; =dynamic keeps base and
     * swaps to micro only around microphysics_driver, so the phases that are already balanced
     * keep their even share. A setaffinity pair costs a couple of microseconds against a
     * ~33 ms phase.
     *
     * Set here, in a constructor, before main and before any OpenMP thread exists, so the
     * team inherits the mask. Launch with --cpu-bind=none so this narrows, never widens.
     * OMP_PROC_BIND must be false: with close/spread the runtime can fix its place list at
     * library load, and constructor order against libiomp5 is not guaranteed. */
    /* WTIMER_PH: "n_base n_micro core0 core1 ..." per rank.
     *
     * Pin the whole pool once, then change only the team size per phase. Non-micro runs at
     * n_base=2 on the rank's own cores; micro runs at n_micro, the extra threads sitting on
     * cores whose owners drop to one thread for exactly that phase. Nothing is oversubscribed
     * at any instant and no affinity call happens after startup.
     *
     * KMP_BLOCKTIME must be 0: with Intel's 200 ms default a parked pool thread spins, and a
     * lender's second thread would keep burning the very core the borrower is meant to use. */
    const char *php = getenv("WTIMER_PH");
    if (php) {
        FILE *g = fopen(php, "r");
        if (g) {
            long want = atol(rank_str()); long i2 = 0; char line[1024];
            while (fgets(line, sizeof line, g)) {
                if (i2++ != want) continue;
                char *t = strtok(line, " \t\n"); if (!t) break;
                ph_base = atoi(t); t = strtok(NULL, " \t\n"); if (!t) break;
                ph_micro = atoi(t);
                for (int i = 0; i < ph_micro && i < 64; i++) {
                    t = strtok(NULL, " \t\n"); if (!t) { ph_micro = i; break; }
                    ph_core[i] = atoi(t);
                }
                break;
            }
            fclose(g);
        }
#ifdef _OPENMP
        if (ph_micro > 0) {
            cpu_set_t u; CPU_ZERO(&u);
            for (int i = 0; i < ph_micro; i++) CPU_SET(ph_core[i], &u);
            sched_setaffinity(0, sizeof u, &u);       /* so new pool threads inherit the union */
            omp_set_num_threads(ph_micro);
            #pragma omp parallel num_threads(ph_micro)
            {
                int t = omp_get_thread_num();
                cpu_set_t m; CPU_ZERO(&m); CPU_SET(ph_core[t], &m);
                sched_setaffinity(0, sizeof m, &m);
            }
            omp_set_num_threads(ph_base);
            my_cpu0 = ph_core[0]; my_ncpu = ph_micro;
        }
#endif
    }
    const char *dp = getenv("WTIMER_CPU");
    const char *md = getenv("WTIMER_MODE");
    dyn_mode = (md && md[0] == 'd');
    if (dp) {
        FILE *g = fopen(dp, "r");
        if (g) {
            long want = atol(rank_str()); long i2 = 0; int a, b, c, d;
            while (fscanf(g, "%d %d %d %d", &a, &b, &c, &d) == 4)
                if (i2++ == want) { base_cpu0 = a; base_ncpu = b; my_cpu0 = c; my_ncpu = d; break; }
            fclose(g);
        }
        if (my_cpu0 >= 0 && my_ncpu > 0) {
            if (dyn_mode) set_cpus(base_cpu0, base_ncpu);
            else          set_cpus(my_cpu0, my_ncpu);
        }
    }
    /* WTIMER_MAP: one integer per line, line r = threads rank r should use inside micro.
     * Absent or unreadable -> micro_thr stays 0 and the wrapper is a pure timer, so a missing
     * map degrades to the verified baseline rather than to something silently different. */
#ifdef _OPENMP
    base_thr = omp_get_max_threads();     /* sampled AFTER the affinity decision above */
#endif
    const char *mp = getenv("WTIMER_MAP");
    if (mp) {
        FILE *g = fopen(mp, "r");
        if (g) {
            long want = atol(rank_str()); long i = 0; int v;
            while (fscanf(g, "%d", &v) == 1) { if (i++ == want) { micro_thr = v; break; } }
            fclose(g);
        }
    }
    cpu_set_t m;
    ncpu_owned = (sched_getaffinity(0, sizeof m, &m) == 0) ? CPU_COUNT(&m) : -1;
    wt_lewi_init();
    atexit(wt_dump);
}
