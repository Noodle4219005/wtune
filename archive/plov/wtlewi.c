/* wtlewi -- node-local core lending, driven by actual MPI wait.
 *
 * The two halves have never been in place at once. Earlier DLB/LeWI work reported 0 acquires
 * and was read as "there is no idle time"; the real reason is that those runs were pure MPI,
 * so a borrower could not use an extra core even when one was free. Meanwhile
 * I_MPI_WAIT_MODE=1 releases the core during a wait and buys +0.10% -- released to nobody.
 * Measured pool: 39.3% of a run is MPI wait (job 1068657).
 *
 * SCOPE. Lending never crosses a 14-core NUMA domain (crossing costs 1.9x), and a domain lives
 * inside one node, so the whole protocol is node-local shared memory. No MPI, no global
 * agreement, no data movement -- which is what keeps it clear of the 101 GB/s wall that killed
 * column migration.
 *
 * WHY THREADS ARE PINNED ONCE. Re-pinning the team on every call was measured at +29%: opening
 * a parallel region per call churns the OpenMP thread pool. So each rank pins a FIXED set of up
 * to 8 threads at startup -- its own 2 cores plus one core from each of its 6 domain-mates --
 * and borrowing only changes the TEAM SIZE. Thread k>=2 always sits on the same lent core, so
 * "borrow" is: check that core is free, claim it, raise the team size.
 *
 * WHAT IS LENT. Rank i owns cores 2i and 2i+1 of its domain. During an MPI wait its master
 * thread spins on 2i but the OpenMP workers are parked (KMP_BLOCKTIME=0 puts them to sleep),
 * so exactly one core -- 2i+1 -- is genuinely free. Seven ranks, so at most 6 cores can be
 * borrowed by any one rank, capping a team at 8.
 *
 * RACE. `lent` is owned by the lender, `taken` by the borrower. A borrower may still be using a
 * core when its owner returns from the wait and reclaims; that is a transient overlap of a few
 * microseconds, not a correctness problem, and it is counted (wt_lewi_stolen) rather than
 * prevented, because preventing it needs a handshake on the lender's critical path.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NDOM 8
#define MAXT 8

/* One cache line per domain. Packing all eight domains' words together would false-share the
 * line across 56 ranks of a node, and this is touched twice per MPI wait -- 63,170 times per
 * rank per run. */
struct dom { uint64_t lent; uint64_t taken; char pad[48]; };
/* node_lent: one bit per rank-slot on the node (56 of 64 used), maintained by every lender
 * regardless of domain. Borrowing across a NUMA domain costs 2.93x so it is never done, but
 * COUNTING what would have been available answers the question the 5% hit rate raises: is the
 * supply missing because my six domain-mates are in the same phase as me, or because every
 * rank on the node is? Pure observation, its own cache line. */
struct nodew { uint64_t lent; uint64_t taken; char pad[48]; };
struct pool { struct dom d[NDOM]; struct nodew n; };

static struct pool *pool;
static int   my_dom = -1;
static int   own[2], exp_core[MAXT], n_exp;
static int   base_team = 2, cur_extra;
/* cur_slot[i]: which expansion entry thread 2+i is serving this call; cur_pin[i]: the core
 * that thread is pinned to right now (left where it was between calls -- repinning is the
 * thing that cost +29% when done unconditionally). */
static int   cur_slot[MAXT], cur_pin[MAXT];
uint64_t     wt_lewi_repin;
static uint64_t exp_bit[MAXT];
static uint64_t my_lend_bit, my_node_bit;
static int      my_slot;
uint64_t wt_lewi_probe_n, wt_lewi_probe_sum, wt_lewi_probe_hist[8];
/* counters -- the G0 gate. A lending mechanism that never fires reports a clean zero, which is
 * indistinguishable from "lending does not help". */
uint64_t wt_lewi_lend, wt_lewi_acq, wt_lewi_acq_cores, wt_lewi_miss;

#define NWIDX 4096
static uint64_t pred_cyc[NWIDX];        /* EMA of cycles for wait index i within a step */
static int      widx, lending_now;
static uint64_t lend_t0, minwait_cyc;
static double   tsc_ghz = 2.0;
uint64_t wt_lewi_stolen, wt_lewi_short;
/* trace: (step, idx, us, lent) for one rank, dumped by wt_lewi_trace_dump */
static int   trace_on, trace_step; static uint32_t *trace_buf; static size_t trace_n, trace_cap;

static inline uint64_t tsc(void) { uint32_t lo, hi; __asm__ __volatile__("rdtsc":"=a"(lo),"=d"(hi)); return ((uint64_t)hi<<32)|lo; }

static const char *rank_env(void)
{
    const char *e;
    if ((e = getenv("PMI_RANK")))     return e;
    if ((e = getenv("SLURM_PROCID"))) return e;
    return "0";
}

void wt_lewi_init(void)
{
    const char *f = getenv("WTIMER_LEWI");
    if (!f) return;
    FILE *g = fopen(f, "r");
    if (!g) return;
    long want = atol(rank_env()), i = 0;
    char line[512];
    while (fgets(line, sizeof line, g)) {
        if (i++ != want) continue;
        char *t = strtok(line, " \t\n");
        my_dom = atoi(t);
        own[0] = atoi(strtok(NULL, " \t\n"));
        own[1] = atoi(strtok(NULL, " \t\n"));
        while ((t = strtok(NULL, " \t\n")) && n_exp < MAXT - 2) exp_core[n_exp++] = atoi(t);
        break;
    }
    fclose(g);
    if (my_dom < 0) return;

    char nm[128], host[64] = "h";
    gethostname(host, sizeof host - 1);
    snprintf(nm, sizeof nm, "/wtlewi.%s.%s", getenv("SLURM_JOB_ID") ? getenv("SLURM_JOB_ID") : "0", host);
    int fd = shm_open(nm, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { my_dom = -1; return; }
    if (ftruncate(fd, sizeof(struct pool)) != 0) { close(fd); my_dom = -1; return; }
    pool = mmap(NULL, sizeof(struct pool), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (pool == MAP_FAILED) { pool = NULL; my_dom = -1; return; }

    /* NODE-WIDE POOL. Bits are indexed by rank SLOT (0..55), not by core within a domain:
     * each rank lends exactly one core -- its second -- so slot and lent core are 1:1, and a
     * borrower can then reach any slot on the node rather than only the six in its own domain.
     *
     * That reach is the whole point. Domain-only lending acquired on 5.2% of microphysics
     * calls while 76.6% of those moments had an idle rank SOMEWHERE on the node (job 1069286),
     * and a borrowed REMOTE core turns out to be worth 99-100% of a local one for microphysics
     * (job 1069299) -- the recorded 2.93x was eight threads spread one per domain, not one or
     * two borrowed. Other phases do pay ~16% for remote placement, which is why the borrowed
     * thread is only ever in the team for the duration of microphysics. */
    my_lend_bit = 1ULL << (own[1] - my_dom * 14);
    { const char *r = getenv("SLURM_PROCID"); if (!r) r = getenv("PMI_RANK");
      my_slot = r ? (atoi(r) % 56) : 0; }
    my_node_bit = 1ULL << my_slot;
    for (int k = 0; k < n_exp; k++) {
        int c = exp_core[k];                       /* node-local core id of the lent core   */
        int slot = (c / 14) * 7 + ((c % 14) / 2);  /* which rank slot owns it               */
        exp_bit[k] = 1ULL << slot;
    }

#ifdef _OPENMP
    /* Pin the whole pool once: thread 0,1 on the rank's own cores, thread 2+k on lent core k. */
    cpu_set_t u; CPU_ZERO(&u);
    CPU_SET(own[0], &u); CPU_SET(own[1], &u);
    for (int k = 0; k < n_exp; k++) CPU_SET(exp_core[k], &u);
    sched_setaffinity(0, sizeof u, &u);
    int nt = 2 + n_exp;
    omp_set_num_threads(nt);
    #pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        int c = (t < 2) ? own[t] : exp_core[t - 2];
        cpu_set_t m; CPU_ZERO(&m); CPU_SET(c, &m);
        sched_setaffinity(0, sizeof m, &m);
    }
    omp_set_num_threads(base_team);
#endif
    for (int k = 0; k < n_exp; k++) cur_pin[k] = exp_core[k];

    if (getenv("WTIMER_TSC_GHZ")) tsc_ghz = atof(getenv("WTIMER_TSC_GHZ"));
    if (getenv("WTIMER_LEWI_MINWAIT_US"))
        minwait_cyc = (uint64_t)(atof(getenv("WTIMER_LEWI_MINWAIT_US")) * tsc_ghz * 1000.0);
    { const char *t = getenv("WTIMER_LEWI_TRACE"), *d = getenv("WTIMER_DEBUG_RANK");
      if (t && atoi(t) && (!d || atoi(d) == atoi(rank_env()))) {
          trace_cap = 4u * 400u * 2048u;               /* 400 steps x 2048 waits */
          trace_buf = malloc(trace_cap * sizeof *trace_buf);
          trace_on  = trace_buf != NULL;
      } }
}

/* Lender: called around every intercepted MPI call.
 *
 * GRANULARITY MISMATCH. A borrow lasts one microphysics call (~23 ms); the mean MPI wait is
 * ~0.35 ms. So a core lent "for the duration of this wait" is in practice lent for 60x longer,
 * and the lender's own thread 1 shares it for the rest of the borrow -- which is where job
 * 1069556 B1's +13.5% micro median came from (24% hit rate, yet slower). The fix is to lend only
 * waits that will be LONG. WRF's per-step wait sequence is periodic (same ~225 MPI_Wait sites
 * in the same order every step), so the duration of wait #i last step predicts wait #i this
 * step; lend only when that prediction exceeds WTIMER_LEWI_MINWAIT_US (default 0 = old
 * behaviour, so the counters can be compared). */
void wt_lewi_lend_begin(void)
{
    if (!pool) return;
    lend_t0 = tsc();
    lending_now = (widx < NWIDX) && (pred_cyc[widx] >= minwait_cyc);
    if (lending_now) {
        __atomic_fetch_or(&pool->n.lent, my_node_bit, __ATOMIC_RELAXED);
        wt_lewi_lend++;
    } else wt_lewi_short++;
}
void wt_lewi_lend_end(void)
{
    if (!pool) return;
    uint64_t dt = tsc() - lend_t0;
    if (lending_now) {
        __atomic_fetch_and(&pool->n.lent, ~my_node_bit, __ATOMIC_RELAXED);
        /* my core is still held by a borrower while I go back to computing */
        if (__atomic_load_n(&pool->n.taken, __ATOMIC_RELAXED) & my_node_bit) wt_lewi_stolen++;
    }
    if (widx < NWIDX) pred_cyc[widx] = (pred_cyc[widx] + dt) / 2;   /* EMA, alpha 0.5 */
    if (trace_on && trace_n + 4 <= trace_cap) {
        trace_buf[trace_n++] = (uint32_t)trace_step; trace_buf[trace_n++] = (uint32_t)widx;
        trace_buf[trace_n++] = (uint32_t)(dt / (tsc_ghz * 1000.0)); trace_buf[trace_n++] = (uint32_t)lending_now;
    }
    widx++;
}
void wt_lewi_trace_dump(const char *dir, const char *rank)
{
    if (!trace_on || !trace_buf) return;
    char p[4096]; snprintf(p, sizeof p, "%s/wait.%s.csv", dir ? dir : ".", rank);
    FILE *f = fopen(p, "w"); if (!f) return;
    fprintf(f, "step,idx,us,lent\n");
    for (size_t i = 0; i + 3 < trace_n; i += 4)
        fprintf(f, "%u,%u,%u,%u\n", trace_buf[i], trace_buf[i+1], trace_buf[i+2], trace_buf[i+3]);
    fclose(f);
}

/* Borrower: called around microphysics. Claims whatever of its expansion cores are free right
 * now -- ANY subset, not a prefix -- and raises the team size by that many.
 *
 * The prefix rule ("thread 2 sits on exp[0], so if exp[0] is busy nothing can be borrowed")
 * was the real 7% ceiling: job 1069329 rank 0 saw >=7 idle ranks at 87% of its microphysics
 * entries and acquired on 1.3% of them. The price of dropping it is that a borrowed thread may
 * have to move to a different lent core, done by that thread itself inside a tiny parallel
 * region, and only when its current pin is not the core it just claimed. */
void wt_lewi_borrow(void)
{
    if (!pool) return;
    cur_extra = 0;
    widx = 0; trace_step++;                       /* microphysics entry is the step marker */
    /* how many of the node's other 55 ranks are inside an MPI wait at this instant */
    uint64_t nl = __atomic_load_n(&pool->n.lent, __ATOMIC_RELAXED) & ~my_node_bit;
    int navail = __builtin_popcountll(nl & 0x00FFFFFFFFFFFFFFULL);
    wt_lewi_probe_n++; wt_lewi_probe_sum += (uint64_t)navail;
    wt_lewi_probe_hist[navail < 7 ? navail : 7]++;
    uint64_t lent = nl;                               /* node-wide, not just my domain */
    int need_repin = 0;
    for (int k = 0; k < n_exp; k++) {
        if (!(lent & exp_bit[k])) continue;
        uint64_t old = __atomic_fetch_or(&pool->n.taken, exp_bit[k], __ATOMIC_RELAXED);
        if (old & exp_bit[k]) continue;               /* someone else already has it */
        cur_slot[cur_extra] = k;
        if (cur_pin[cur_extra] != exp_core[k]) need_repin = 1;
        cur_extra++;
    }
    if (cur_extra) { wt_lewi_acq++; wt_lewi_acq_cores += (uint64_t)cur_extra; }
    else            wt_lewi_miss++;
#ifdef _OPENMP
    if (cur_extra) {
        int nt = base_team + cur_extra;
        omp_set_num_threads(nt);
        if (need_repin) {
            wt_lewi_repin++;
            #pragma omp parallel num_threads(nt)
            {
                int t = omp_get_thread_num();
                if (t >= base_team) {
                    int i = t - base_team, c = exp_core[cur_slot[i]];
                    if (cur_pin[i] != c) {
                        cpu_set_t m; CPU_ZERO(&m); CPU_SET(c, &m);
                        sched_setaffinity(0, sizeof m, &m);
                        cur_pin[i] = c;
                    }
                }
            }
        }
    }
#endif
}
void wt_lewi_return(void)
{
    if (!pool || !cur_extra) return;
    uint64_t m = 0;
    for (int i = 0; i < cur_extra; i++) m |= exp_bit[cur_slot[i]];
    __atomic_fetch_and(&pool->n.taken, ~m, __ATOMIC_RELAXED);
    cur_extra = 0;
#ifdef _OPENMP
    omp_set_num_threads(base_team);
#endif
}
