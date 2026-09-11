/* The microphysics call path: collect, optionally retarget, marshal, invoke.
 *
 * Stage 0 (this file, staging off) is pure pass-through THROUGH the marshaller. It exists as a
 * gate in its own right: if rebuilding a 402-argument frame is not bit-identical to the
 * original call, nothing built on top of it can be trusted, and the failure is loud.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include "wt_argmap.h"

void  wt_call_n(void *fn, void **argv, int n);
void  wt_pre(int idx);
void  wt_record(int idx, uint64_t cycles);
extern void *wt_argv[];
/* --wrap resolves a reference to __real_X back to X, so this names the original driver. */
extern void __real_module_microphysics_driver_mp_microphysics_driver_(void);

uint64_t wt_micro_calls, wt_micro_staged;

static inline uint64_t tsc(void)
{
    uint32_t lo, hi; __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* ---- dense-rectangle staging -------------------------------------------------------------
 *
 * Fortran stores A(ims:ime, kms:kme, jms:jme) column-major, so
 *     off3(i,k,j) = (i-ims) + NI*((k-kms) + NK*(j-jms))
 * The staging buffer is S(1:N, kms:kme, 1:1): off = (n-1) + N*(k-kms).
 *
 * WHY A WHOLE ROW BAND. Selecting scattered columns would need an index map on every array;
 * taking the last R rows of the patch keeps the copy contiguous in i and lets the un-staged
 * part stay a legal tile rectangle, which is all the remaining tile list has to express.
 *
 * WHY ids/ide/jds/jde ARE WIDENED. The driver narrows its own its/ite with
 * its = max(ips, ids+spec_zone), ite = min(ipe, ide-1): a staged rectangle that looks like it
 * touches the domain edge would get boundary treatment the original columns did not. Pushing
 * the domain bounds far outside the rectangle makes every staged column interior.
 */
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#define IARG(k) (*(int *)wt_argv[k])

static float *stage_buf;
static size_t stage_cap;

/* A slot is published before its owner starts pass 1 and is not reused until the owner has
 * committed it. That invariant keeps the packed source and the result stable while a neighbour
 * computes the task, without a node-wide barrier. */
#define WT_SHARE_EMPTY       0
#define WT_SHARE_PUBLISHED   1
#define WT_SHARE_CLAIMED_SELF 2
#define WT_SHARE_CLAIMED_OTHER 3
#define WT_SHARE_DONE         4
typedef struct __attribute__((aligned(64))) {
    uint64_t work_ticks;
    int state, ncol, nk, n3, n2, owner, step;
    int reserved[7];
} wt_share_hdr;
typedef struct {
    uint64_t t_full_ticks, step;
} wt_share_stats;
typedef char wt_share_hdr_size_check[(sizeof(wt_share_hdr) == 64) ? 1 : -1];

static int share_init_state = -1;
static int share_fd = -1, share_me, share_nlocal;
static int share_k = 1;
static void *share_map;
static size_t share_bytes, share_slot_bytes, share_cap;
static wt_share_stats *share_stats;
static char share_path[4096];
static uint64_t share_step;
static uint64_t adapt_cal_tsc;
static struct timespec adapt_cal_wall;
static double adapt_tsc_hz;
static int adapt_R_prev = -1;                       /* integrator state: rows published last call */
static int adapt_stole_prev;                        /* chunks stolen last call: a stealer is not overloaded */
static double adapt_gain = 0.5;

static int wt_stage_share_enabled(void)
{
    const char *e = getenv("WTIMER_STAGE_SHARE");
    return e && atoi(e) == 1;
}

static int wt_stage_adapt_enabled(void)
{
    const char *e = getenv("WTIMER_STAGE_ADAPT");
    return e && atoi(e) == 1;
}

static int wt_share_first_int(const char *s)
{
    char *end;
    long v;
    if (!s || !*s) return 0;
    v = strtol(s, &end, 10);
    return end != s && v > 0 && v <= INT_MAX ? (int)v : 0;
}

static int wt_share_local_id(const char *s)
{
    char *end;
    long v;
    if (!s || !*s) return -1;
    v = strtol(s, &end, 10);
    return end != s && v >= 0 && v <= INT_MAX ? (int)v : -1;
}

static size_t wt_share_cap_from_env(void)
{
    const char *e = getenv("WTIMER_STAGE_SLOT_MB");
    unsigned long long mb = 4;
    unsigned long long v;
    char *end;
    size_t mib = 1024u * 1024u;

    if (e && *e) {
        v = strtoull(e, &end, 10);
        if (end != e) mb = v;
    }
    if (mb > SIZE_MAX / mib) return 0;
    return (size_t)mb * mib;
}

static int wt_share_chunks_from_env(void)
{
    const char *e = getenv("WTIMER_STAGE_CHUNKS");
    return e && *e ? wt_share_first_int(e) : 4;
}

static int wt_share_init(size_t need, int first, int adaptive)
{
    const char *e, *jid;
    int nlocal, me;
    size_t nslots;

    if (share_init_state >= 0) return share_init_state;
    share_init_state = 0;
    share_k = adaptive ? wt_share_chunks_from_env() : 1;
    if (share_k <= 0) return 0;
    e = getenv("SLURM_LOCALID");
    me = wt_share_local_id(e);
    /* step-accurate counts first: the batch env's SLURM_NTASKS_PER_NODE describes the
     * allocation (112), not this srun step (56) */
    nlocal = wt_share_first_int(getenv("SLURM_STEP_TASKS_PER_NODE"));
    if (!nlocal) nlocal = wt_share_first_int(getenv("SLURM_TASKS_PER_NODE"));
    if (!nlocal) nlocal = wt_share_first_int(getenv("SLURM_NTASKS_PER_NODE"));
    if (me < 0 || !nlocal || me >= nlocal) return 0;

    share_cap = wt_share_cap_from_env();
    if (share_cap > SIZE_MAX - sizeof(wt_share_hdr)) return 0;
    share_slot_bytes = sizeof(wt_share_hdr) + share_cap;
    if ((size_t)nlocal > SIZE_MAX / (size_t)share_k) return 0;
    nslots = (size_t)nlocal * share_k;
    if (nslots > SIZE_MAX / share_slot_bytes) return 0;
    share_bytes = nslots * share_slot_bytes;
    if (adaptive) {
        if ((size_t)nlocal > (SIZE_MAX - share_bytes) / sizeof(wt_share_stats)) return 0;
        share_bytes += (size_t)nlocal * sizeof(wt_share_stats);
    }
    if (share_bytes > (size_t)LLONG_MAX) return 0;

    jid = getenv("SLURM_JOB_ID");
    if (!jid || !*jid) {
        snprintf(share_path, sizeof share_path, "/dev/shm/wtstage_%ld", (long)getppid());
    } else {
        snprintf(share_path, sizeof share_path, "/dev/shm/wtstage_%s", jid);
    }
    share_fd = open(share_path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (share_fd < 0) { share_fd = -1; return 0; }
    if (ftruncate(share_fd, (off_t)share_bytes) != 0) {
        close(share_fd); share_fd = -1; return 0;
    }
    share_map = mmap(NULL, share_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, share_fd, 0);
    if (share_map == MAP_FAILED) {
        share_map = NULL; close(share_fd); share_fd = -1; return 0;
    }
    share_me = me; share_nlocal = nlocal;
    share_stats = adaptive ? (wt_share_stats *)((char *)share_map + nslots * share_slot_bytes) : NULL;
    /* Only the owner resets its slots: other ranks may already have published work. */
    for (int c = 0; c < share_k; c++) {
        wt_share_hdr *h = (wt_share_hdr *)((char *)share_map +
                                           ((size_t)me * share_k + c) * share_slot_bytes);
        __atomic_store_n(&h->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
        __atomic_store_n(&h->work_ticks, 0, __ATOMIC_RELAXED);
    }
    if (share_stats) {
        __atomic_store_n(&share_stats[me].t_full_ticks, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&share_stats[me].step, 0, __ATOMIC_RELEASE);
    }
    share_init_state = 1;
    if (first)
        fprintf(stderr, "[wtimer] share: me=%d nlocal=%d slot=%zu bytes cap=%zu need=%zu file=%s\n",
                share_me, share_nlocal, share_slot_bytes, share_cap, need, share_path);
    return 1;
}

void wt_micro_share_finalize(void)
{
    if (share_init_state != 1) return;
    if (share_map) { munmap(share_map, share_bytes); share_map = NULL; }
    if (share_fd >= 0) { close(share_fd); share_fd = -1; }
    if (share_me == 0) unlink(share_path);
    share_stats = NULL;
    share_init_state = 0;
}

static inline wt_share_hdr *wt_share_slot(int n)
{
    return (wt_share_hdr *)((char *)share_map + (size_t)n * share_slot_bytes);
}

static void wt_adapt_calibrate(void)
{
    struct timespec now;
    uint64_t t = tsc();
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (adapt_cal_tsc) {
        double sec = (now.tv_sec - adapt_cal_wall.tv_sec) +
                     1e-9 * (now.tv_nsec - adapt_cal_wall.tv_nsec);
        if (sec > 0 && t > adapt_cal_tsc) adapt_tsc_hz = (double)(t - adapt_cal_tsc) / sec;
    } else {
        adapt_cal_tsc = t;
        adapt_cal_wall = now;
    }
}

static double wt_adapt_ms(uint64_t ticks)
{
    return adapt_tsc_hz > 0 ? 1000.0 * (double)ticks / adapt_tsc_hz : 0.0;
}

static inline void wt_share_mark(int first, const char *mark)
{
    if (first) { fprintf(stderr, "[wtimer] mark %s\n", mark); fflush(stderr); }
}

static inline void wt_share_pause(unsigned n)
{
    if ((n & 63u) == 63u) sched_yield();
    else __asm__ __volatile__("pause");
}

static void wt_share_pack(float *buf, int ims, int jms, int kms, int NI, int NK,
                          int i0, int i1, int j0, int j1, int NCOL,
                          int n3, const int *w3)
{
    float *p = buf;
    int NIC = i1 - i0 + 1;
    for (int a = 0; a < n3; a++) {
        float *src = (float *)wt_argv[w3[a]];
        for (int k = 0; k < NK; k++)
            for (int j = j0; j <= j1; j++)
                memcpy(p + (size_t)NCOL*k + (size_t)NIC*(j-j0),
                       src + (size_t)(i0-ims) + (size_t)NI*((size_t)k + (size_t)NK*(j-jms)),
                       (size_t)NIC * sizeof(float));
        p += (size_t)NCOL * NK;
    }
    for (int a = 0; a < WT_W2D; a++) {
        float *src = (float *)wt_argv[wt_w2d[a]];
        for (int j = j0; j <= j1; j++)
            memcpy(p + (size_t)NIC*(j-j0),
                   src + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms),
                   (size_t)NIC * sizeof(float));
        p += NCOL;
    }
}

static void wt_share_commit(float *buf, void **save, int ims, int jms, int NI, int NK,
                            int i0, int i1, int j0, int j1, int NCOL,
                            int n3, const int *w3)
{
    float *p = buf;
    int NIC = i1 - i0 + 1;
    for (int a = 0; a < n3; a++) {
        float *dst = (float *)save[w3[a]];
        for (int k = 0; k < NK; k++)
            for (int j = j0; j <= j1; j++)
                memcpy(dst + (size_t)(i0-ims) + (size_t)NI*((size_t)k + (size_t)NK*(j-jms)),
                       p + (size_t)NCOL*k + (size_t)NIC*(j-j0),
                       (size_t)NIC * sizeof(float));
        p += (size_t)NCOL * NK;
    }
    for (int a = 0; a < WT_W2D; a++) {
        float *dst = (float *)save[wt_w2d[a]];
        for (int j = j0; j <= j1; j++)
            memcpy(dst + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms),
                   p + (size_t)NIC*(j-j0), (size_t)NIC * sizeof(float));
        p += NCOL;
    }
}

static void wt_share_pass1(void **save, int jsplit, int first)
{
    static int i0[64], i1[64], j0[64], j1[64], nt1;
    int nt = IARG(WT_A_NUM_TILES); if (nt > 64) nt = 64;
    const int *oi0 = wt_argv[WT_A_I_START], *oi1 = wt_argv[WT_A_I_END];
    const int *oj0 = wt_argv[WT_A_J_START], *oj1 = wt_argv[WT_A_J_END];
    nt1 = 0;
    for (int t = 0; t < nt; t++) {
        int je = oj1[t] < jsplit - 1 ? oj1[t] : jsplit - 1;
        if (je < oj0[t]) continue;
        i0[nt1] = oi0[t]; i1[nt1] = oi1[t]; j0[nt1] = oj0[t]; j1[nt1] = je; nt1++;
    }
    wt_argv[WT_A_I_START] = i0; wt_argv[WT_A_I_END] = i1;
    wt_argv[WT_A_J_START] = j0; wt_argv[WT_A_J_END] = j1;
    wt_argv[WT_A_NUM_TILES] = &nt1;
    wt_share_mark(first, "pass1-call");
    if (nt1 > 0)
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
    memcpy(wt_argv, save, sizeof(void *) * WT_NARG);
    wt_share_mark(first, "pass1-done");
}

static void wt_share_pass2(float *buf, int ncol, int NK, int n3, const int *w3,
                           void **save, int nt, int first)
{
    static int s_ims, s_ime, s_jms, s_jme, s_ips, s_ipe, s_jps, s_jpe;
    static int s_ids, s_ide, s_jds, s_jde;
    static int i0[64], i1[64], j0[64], j1[64], nt2;
    float *p = buf;
    s_ims = s_ips = s_jms = s_jps = 1; s_ime = s_ipe = ncol; s_jme = s_jpe = 1;
    s_ids = -1000; s_ide = ncol + 1000; s_jds = -1000; s_jde = 1 + 1000;
    for (int a = 0; a < n3; a++) { wt_argv[w3[a]] = p; p += (size_t)ncol * NK; }
    for (int a = 0; a < WT_W2D; a++) { wt_argv[wt_w2d[a]] = p; p += ncol; }
    wt_argv[WT_A_IMS]=&s_ims; wt_argv[WT_A_IME]=&s_ime;
    wt_argv[WT_A_JMS]=&s_jms; wt_argv[WT_A_JME]=&s_jme;
    wt_argv[WT_A_IPS]=&s_ips; wt_argv[WT_A_IPE]=&s_ipe;
    wt_argv[WT_A_JPS]=&s_jps; wt_argv[WT_A_JPE]=&s_jpe;
    wt_argv[WT_A_IDS]=&s_ids; wt_argv[WT_A_IDE]=&s_ide;
    wt_argv[WT_A_JDS]=&s_jds; wt_argv[WT_A_JDE]=&s_jde;
    nt2 = nt < ncol ? nt : ncol; if (nt2 < 1) nt2 = 1;
    for (int t = 0; t < nt2; t++) {
        i0[t] = 1 + (int)((long)ncol * t / nt2); i1[t] = (int)((long)ncol * (t+1) / nt2);
        j0[t] = 1; j1[t] = 1;
    }
    wt_argv[WT_A_I_START]=i0; wt_argv[WT_A_I_END]=i1;
    wt_argv[WT_A_J_START]=j0; wt_argv[WT_A_J_END]=j1;
    wt_argv[WT_A_NUM_TILES]=&nt2;
    wt_share_mark(first, "pass2-call");
    wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
    wt_share_mark(first, "pass2-done");
}

uint64_t wt_micro_share_out, wt_micro_share_in;
uint64_t wt_micro_share_wait_ticks, wt_micro_share_work_ticks;
uint64_t wt_micro_rows_out, wt_micro_chunks_in;

static int wt_share_steal(int NK, int n3, const int *w3, int nt, int first)
{
    int stolen = 0;
    if (share_nlocal > 1) for (;;) {
        int claimed = 0;
        for (int off = 1; off < share_nlocal; off++) {
            int owner = (share_me + off) % share_nlocal;
            wt_share_hdr *h = wt_share_slot(owner);
            int expected = WT_SHARE_PUBLISHED;
            if (__atomic_load_n(&h->state, __ATOMIC_ACQUIRE) != WT_SHARE_PUBLISHED) continue;
            if (!__atomic_compare_exchange_n(&h->state, &expected, WT_SHARE_CLAIMED_OTHER, 0,
                                             __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
            if (h->n3 != n3 || h->nk != NK || h->n2 != WT_W2D || h->ncol <= 0 ||
                (size_t)h->ncol > share_cap / (((size_t)n3 * NK + WT_W2D) * sizeof(float))) {
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            void *save[WT_NARG];
            memcpy(save, wt_argv, sizeof save);
            uint64_t t0 = tsc();
            wt_micro_share_in++;
            wt_share_pass2((float *)((char *)h + sizeof *h), h->ncol, NK, n3, w3, save, nt, first);
            memcpy(wt_argv, save, sizeof save);
            wt_micro_share_work_ticks += tsc() - t0;
            __atomic_store_n(&h->state, WT_SHARE_DONE, __ATOMIC_RELEASE);
            claimed = 1; stolen++;
        }
        if (!claimed) break;
    }
    if (first) {
        char mark[64]; snprintf(mark, sizeof mark, "steal %d", stolen); wt_share_mark(first, mark);
    }
    return stolen;
}

static void wt_adapt_read_stats(uint64_t previous, uint64_t *t_me, uint64_t *t_mean, int *nvalid)
{
    uint64_t sum = 0;
    *t_me = *t_mean = 0;
    *nvalid = 0;
    if (!share_stats || !previous) return;
    for (int r = 0; r < share_nlocal; r++) {
        wt_share_stats *s = &share_stats[r];
        uint64_t s1 = __atomic_load_n(&s->step, __ATOMIC_ACQUIRE);
        if (s1 != previous) continue;
        uint64_t full = __atomic_load_n(&s->t_full_ticks, __ATOMIC_ACQUIRE);
        uint64_t s2 = __atomic_load_n(&s->step, __ATOMIC_ACQUIRE);
        if (s1 != s2 || full == 0) continue;
        if (UINT64_MAX - sum < full) continue;
        sum += full;
        (*nvalid)++;
        if (r == share_me) *t_me = full;
    }
    if (*nvalid) *t_mean = sum / (uint64_t)*nvalid;
}

static int wt_adapt_need_fits(int NIC, int rows, int NK, int n3, size_t cap)
{
    size_t fields = (size_t)n3 * (size_t)NK + WT_W2D;
    if (NIC <= 0 || rows <= 0 || fields > SIZE_MAX / sizeof(float)) return 0;
    if ((size_t)NIC > SIZE_MAX / (size_t)rows) return 0;
    size_t cols = (size_t)NIC * (size_t)rows;
    if (cols > SIZE_MAX / fields) return 0;
    size_t elems = cols * fields;
    return elems <= cap / sizeof(float);
}

static int wt_adapt_steal(int NK, int n3, const int *w3, int nt, int first,
                          uint64_t call_start, uint64_t t_mean)
{
    size_t total = (size_t)share_nlocal * (size_t)share_k;
    size_t start = (size_t)((share_me + 1) % share_nlocal) * (size_t)share_k;
    int stolen = 0;
    if (!total) return 0;
    for (;;) {
        int claimed = 0;
        for (size_t off = 0; off < total; off++) {
            if (t_mean && tsc() - call_start >= t_mean) goto done;
            int slotno = (int)((start + off) % total);
            int owner = slotno / share_k;
            if (owner == share_me) continue;
            wt_share_hdr *h = wt_share_slot(slotno);
            int expected = WT_SHARE_PUBLISHED;
            if (__atomic_load_n(&h->state, __ATOMIC_ACQUIRE) != WT_SHARE_PUBLISHED) continue;
            if (!__atomic_compare_exchange_n(&h->state, &expected, WT_SHARE_CLAIMED_OTHER, 0,
                                             __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
            if (h->n3 != n3 || h->nk != NK || h->n2 != WT_W2D || h->ncol <= 0 ||
                !wt_adapt_need_fits(1, h->ncol, NK, n3, share_cap)) {
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            void *save[WT_NARG];
            memcpy(save, wt_argv, sizeof save);
            uint64_t work0 = tsc();
            wt_micro_share_in++;
            wt_micro_chunks_in++;
            wt_share_pass2((float *)((char *)h + sizeof *h), h->ncol, NK, n3, w3, save, nt, first);
            memcpy(wt_argv, save, sizeof save);
            uint64_t work = tsc() - work0;
            __atomic_store_n(&h->work_ticks, work, __ATOMIC_RELEASE);
            wt_micro_share_work_ticks += work;
            __atomic_store_n(&h->state, WT_SHARE_DONE, __ATOMIC_RELEASE);
            claimed = 1; stolen++;
        }
        if (!claimed) break;
    }
done:
    if (first) {
        char mark[64]; snprintf(mark, sizeof mark, "steal %d", stolen); wt_share_mark(first, mark);
    }
    return stolen;
}

static int wt_micro_adaptive_call(int CH, int ims, int jms, int kms, int NI, int NK,
                                  int ips, int ipe, int jps, int jpe,
                                  int ids, int ide, int jds, int jde, int sz,
                                  int n3, const int *w3, int first, int debug_rank)
{
    int i0 = ips > ids + sz ? ips : ids + sz;
    int i1 = ipe < ide - 1 - sz ? ipe : ide - 1 - sz;
    int jp0 = jps > jds + sz ? jps : jds + sz;
    int jp1 = jpe < jde - 1 - sz ? jpe : jde - 1 - sz;
    int NIC = i1 - i0 + 1;
    int NJp = jp1 >= jp0 ? jp1 - jp0 + 1 : 0;
    int K = wt_share_chunks_from_env();
    size_t cap = wt_share_cap_from_env();
    size_t fields = (size_t)n3 * (size_t)NK + WT_W2D;
    size_t max_need = 0;
    int max_rows = CH < NJp ? CH : NJp;
    uint64_t call_start = tsc();
    uint64_t step = wt_micro_calls + 1;             /* wt_micro_calls++ happens after the call; 0 is the invalid sentinel */
    uint64_t previous = step ? step - 1 : 0, t_me, t_mean;
    int nvalid, R, nchunks = 0;
    int nt = IARG(WT_A_NUM_TILES); if (nt > 64) nt = 64;

    if (debug_rank) wt_adapt_calibrate();
    { const char *ge = getenv("WTIMER_STAGE_GAIN"); if (ge && atof(ge) > 0) adapt_gain = atof(ge); }
    if (K <= 0 || !wt_adapt_need_fits(NIC, max_rows, NK, n3, cap)) return 0;
    max_need = (size_t)NIC * (size_t)max_rows * fields * sizeof(float);
    if (!wt_share_init(max_need, first, 1)) return 0;
    wt_adapt_read_stats(previous, &t_me, &t_mean, &nvalid);

    /* Feedback on the MEASURED wrapped-call time (pack + passes + waits + steals), not on an
     * open-loop estimate: job 1070120 showed a rank giving 34% of its rows dropped only
     * 45.2 -> 37.3 ms (predicted 29.9) because the tail rows are colder than its average and
     * the clipped tile list is uneven across threads. Integrate until t_call meets the node mean. */
    if (adapt_R_prev < 0) adapt_R_prev = CH;
    if (nvalid * 2 < share_nlocal || !t_me) {
        R = adapt_R_prev;
    } else {
        long long maxr = (long long)CH * K;
        const char *pe = getenv("WTIMER_STAGE_MAX_PCT");   /* cap on published rows, % of NJp */
        int pct = pe ? atoi(pe) : 50;
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        int limit = NJp * pct / 100;
        if (maxr < limit) limit = (int)maxr;
        double raw = adapt_gain * (double)NJp * ((double)t_me - (double)t_mean) / (double)t_me;
        if (raw > 0 && adapt_stole_prev > 0) raw = 0;   /* my time is high only because I stole: do not publish more */
        double next = (double)adapt_R_prev + (raw >= 0 ? raw + 0.5 : raw - 0.5);
        if (next <= 0) R = 0;
        else if (next >= (double)limit) R = limit;
        else R = (int)next;
    }
    adapt_R_prev = R;
    if (R > NJp) R = NJp;
    long long max_chunks = (long long)CH * K;
    if ((long long)R > max_chunks) R = max_chunks > INT_MAX ? INT_MAX : (int)max_chunks;
    if (R > 0) nchunks = (int)(((long long)R + CH - 1) / CH);
    if (debug_rank && (step <= 3 || step % 10 == 0))
        fprintf(stderr, "[wtimer] adapt step=%d t_me=%.3fms t_mean=%.3fms nvalid=%d R=%d chunks=%d\n",
                (int)step, wt_adapt_ms(t_me), wt_adapt_ms(t_mean), nvalid, R, nchunks);

    uint64_t t_full = 0;
    if (R == 0) {
        uint64_t t0 = tsc();
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        t_full = tsc() - t0;
    } else {
        void *save[WT_NARG];
        int jsplit = jp1 - R + 1;
        uint64_t own_work = 0;
        memcpy(save, wt_argv, sizeof save);
        for (int c = 0; c < nchunks; c++) {
            int cj0 = jsplit + c * CH;
            int cj1 = cj0 + CH - 1;
            if (cj1 > jp1) cj1 = jp1;
            int rows = cj1 - cj0 + 1;
            int ncol = NIC * rows;
            wt_share_hdr *h = wt_share_slot(share_me * share_k + c);
            wt_share_pack((float *)((char *)h + sizeof *h), ims, jms, kms, NI, NK,
                          i0, i1, cj0, cj1, ncol, n3, w3);
            h->ncol = ncol; h->nk = NK; h->n3 = n3; h->n2 = WT_W2D;
            h->owner = share_me; h->step = (int)step;
            __atomic_store_n(&h->work_ticks, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
        }
        wt_micro_rows_out += (uint64_t)R;
        uint64_t pass0 = tsc();
        wt_share_pass1(save, jsplit, first);
        t_full = tsc() - pass0;
        for (int c = 0; c < nchunks; c++) {
            int cj0 = jsplit + c * CH;
            int cj1 = cj0 + CH - 1;
            if (cj1 > jp1) cj1 = jp1;
            int rows = cj1 - cj0 + 1;
            int ncol = NIC * rows;
            wt_share_hdr *h = wt_share_slot(share_me * share_k + c);
            int expected = WT_SHARE_PUBLISHED;
            int claimed_self = __atomic_compare_exchange_n(&h->state, &expected, WT_SHARE_CLAIMED_SELF, 0,
                                                           __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
            uint64_t wait0 = 0;
            if (!claimed_self) {
                unsigned spin = 0;
                wait0 = tsc();
                for (;;) {
                    int state = __atomic_load_n(&h->state, __ATOMIC_ACQUIRE);
                    if (state == WT_SHARE_DONE) break;
                    if (state == WT_SHARE_PUBLISHED) {
                        expected = WT_SHARE_PUBLISHED;
                        if (__atomic_compare_exchange_n(&h->state, &expected, WT_SHARE_CLAIMED_SELF, 0,
                                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                            claimed_self = 1;
                            break;
                        }
                    }
                    wt_share_pause(spin++);
                }
            }
            if (claimed_self) {
                uint64_t work0 = tsc();
                wt_share_pass2((float *)((char *)h + sizeof *h), ncol, NK, n3, w3, save, nt, first);
                uint64_t work = tsc() - work0;
                __atomic_store_n(&h->work_ticks, work, __ATOMIC_RELEASE);
                own_work += work;
                wt_share_commit((float *)((char *)h + sizeof *h), save, ims, jms, NI, NK,
                                i0, i1, cj0, cj1, ncol, n3, w3);
                memcpy(wt_argv, save, sizeof save);
                __atomic_store_n(&h->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            } else {
                wt_micro_share_out++;
                wt_micro_share_wait_ticks += tsc() - wait0;
                uint64_t work = __atomic_load_n(&h->work_ticks, __ATOMIC_ACQUIRE);
                own_work += work;
                wt_share_commit((float *)((char *)h + sizeof *h), save, ims, jms, NI, NK,
                                i0, i1, cj0, cj1, ncol, n3, w3);
                memcpy(wt_argv, save, sizeof save);
                __atomic_store_n(&h->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            }
        }
        t_full += own_work;
        wt_micro_staged++;
    }
    adapt_stole_prev = wt_adapt_steal(NK, n3, w3, nt, first, call_start, nvalid * 2 >= share_nlocal ? t_mean : 0);
    (void)t_full;
    t_full = tsc() - call_start;                    /* published sample = whole wrapped call */
    if (!t_full) t_full = 1;
    /* Invalidate only now, not at entry: ranks enter micro within ms of each other in random
     * order, and an early invalidation would leave the last rank in with no valid neighbours.
     * Readers reject a previous step number paired with a half-written t_full_ticks. */
    __atomic_store_n(&share_stats[share_me].step, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&share_stats[share_me].t_full_ticks, t_full, __ATOMIC_RELAXED);
    __atomic_store_n(&share_stats[share_me].step, step, __ATOMIC_RELEASE);
    return 1;
}

static int wt_micro_shared_call(int jsplit, int ims, int jms, int kms, int NI, int NK,
                                int i0, int i1, int j0, int j1, int NCOL,
                                int n3, const int *w3, size_t need, int first)
{
    void *save[WT_NARG];
    int nt;
    int publish;
    wt_share_hdr *own;
    float *buf;

    if (!wt_share_init(need, first, 0)) return 0;
    publish = need <= share_cap;
    if (!publish) {
        if (need > stage_cap) { free(stage_buf); stage_buf = malloc(need); stage_cap = need; }
        if (!stage_buf) return 0;
    }
    memcpy(save, wt_argv, sizeof save);
    nt = IARG(WT_A_NUM_TILES); if (nt > 64) nt = 64;
    own = wt_share_slot(share_me * share_k);
    buf = publish ? (float *)((char *)own + sizeof *own) : stage_buf;
    if (publish) {
        wt_share_pack(buf, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL, n3, w3);
        own->ncol = NCOL; own->nk = NK; own->n3 = n3; own->n2 = WT_W2D;
        own->owner = share_me; own->step = (int)++share_step;
        wt_share_mark(first, "publish");
        __atomic_store_n(&own->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
    }
    wt_share_pass1(save, jsplit, first);
    if (!publish) {
        wt_share_pack(buf, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL, n3, w3);
        wt_share_mark(first, "pack-done");
        wt_share_pass2(buf, NCOL, NK, n3, w3, save, nt, first);
        wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL, n3, w3);
        memcpy(wt_argv, save, sizeof save);
    } else {
        int expected = WT_SHARE_PUBLISHED;
        int claimed_self = __atomic_compare_exchange_n(&own->state, &expected, WT_SHARE_CLAIMED_SELF, 0,
                                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        uint64_t wait0 = 0;
        if (!claimed_self) {
            unsigned spin = 0;
            wait0 = tsc();
            wt_share_mark(first, "own-wait");
            for (;;) {
                int state = __atomic_load_n(&own->state, __ATOMIC_ACQUIRE);
                if (state == WT_SHARE_DONE) break;
                /* A mismatched stealer puts the slot back so its owner can retry. */
                if (state == WT_SHARE_PUBLISHED) {
                    expected = WT_SHARE_PUBLISHED;
                    if (__atomic_compare_exchange_n(&own->state, &expected, WT_SHARE_CLAIMED_SELF, 0,
                                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                        claimed_self = 1;
                        break;
                    }
                }
                wt_share_pause(spin++);
            }
        }
        if (claimed_self) {
            wt_share_mark(first, "own-claim");
            wt_share_pass2(buf, NCOL, NK, n3, w3, save, nt, first);
            wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL, n3, w3);
            memcpy(wt_argv, save, sizeof save);
            __atomic_store_n(&own->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            wt_share_mark(first, "own-done");
        } else {
            wt_micro_share_out++;
            wt_micro_share_wait_ticks += tsc() - wait0;
            wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL, n3, w3);
            memcpy(wt_argv, save, sizeof save);
            __atomic_store_n(&own->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            wt_share_mark(first, "own-done");
        }
    }
    /* w3/n3 came from this rank's flags above; a slot contributes packed bytes, never its owner's
     * pointer list or scalar state. */
    wt_share_steal(NK, n3, w3, nt, first);
    wt_micro_staged++;
    return 1;
}

static int stage_rows(void)   /* how many of the patch's j-rows to stage; 0 = off */
{
    const char *e = getenv("WTIMER_STAGE_ROWS");
    return e ? atoi(e) : 0;
}

uint64_t wt_micro_refused;      /* staging requested but the self-check said no: ran unstaged */

/* Self-check of the argument map at the positions that matter for staging. The old map was
 * derived from the .F source and was right for every scalar and bound (all < 112) while being
 * wrong for every array pointer past 111 (16 #if-excluded dummies), so bounds alone prove
 * nothing. What is checked: Thompson is the scheme, its species flags are set, the six moist
 * species are distinct addresses of one 4D array (spacing a multiple of the 3D block), and
 * the two scalar species differ. Runs once; failure disables staging for the run. */
static int stage_ok = -1;
static int wt_stage_selfcheck(int NI, int NK, int NJ, int verbose)
{
    static const int fl[] = {WT_A_F_QV, WT_A_F_QC, WT_A_F_QR, WT_A_F_QI, WT_A_F_QS, WT_A_F_QG,
                             WT_A_F_QNI, WT_A_F_QNR};
    int ok = 1, mp = IARG(WT_A_MP_PHYSICS);
    if (mp != 8) ok = 0;                                    /* whitelist is Thompson's */
    for (unsigned k = 0; k < sizeof fl / sizeof *fl; k++)
        if (!wt_argv[fl[k]] || IARG(fl[k]) == 0) ok = 0;
    char *q[6]; size_t blk = (size_t)NI * NK * NJ * sizeof(float);
    for (int k = 0; k < 6; k++) q[k] = (char *)wt_argv[wt_w3d[5 + k]];   /* qv qc qr qi qs qg */
    for (int k = 1; k < 6; k++) {
        if (!q[k] || q[k] == q[0]) ok = 0;
        else if ((size_t)(q[k] > q[0] ? q[k] - q[0] : q[0] - q[k]) % blk) ok = 0;
    }
    if (!wt_argv[wt_w3d[11]] || !wt_argv[wt_w3d[12]] || wt_argv[wt_w3d[11]] == wt_argv[wt_w3d[12]]) ok = 0;
    if (verbose) {
        fprintf(stderr, "[wtimer] selfcheck: mp_physics=%d dt=%.3f dx=%.1f f_qv..f_qg,f_qni,f_qnr=", mp,
                *(float *)wt_argv[WT_A_DT], *(float *)wt_argv[WT_A_DX]);
        for (unsigned k = 0; k < sizeof fl / sizeof *fl; k++) fprintf(stderr, "%d,", wt_argv[fl[k]] ? IARG(fl[k]) : -99);
        fprintf(stderr, " moist spacing/blk=");
        for (int k = 1; k < 6; k++) fprintf(stderr, "%ld,", q[k] && q[0] ? (long)((q[k] - q[0]) / (long)blk) : -99L);
        fprintf(stderr, " do_radar_ref=%d has_reqc/i/s=%d/%d/%d specified=%d spec_zone=%d => %s\n",
                wt_argv[WT_A_DO_RADAR_REF] ? IARG(WT_A_DO_RADAR_REF) : -99,
                wt_argv[WT_A_HAS_REQC] ? IARG(WT_A_HAS_REQC) : -99, wt_argv[WT_A_HAS_REQI] ? IARG(WT_A_HAS_REQI) : -99,
                wt_argv[WT_A_HAS_REQS] ? IARG(WT_A_HAS_REQS) : -99, IARG(WT_A_SPECIFIED), IARG(WT_A_SPEC_ZONE),
                ok ? "STAGING ON" : "STAGING REFUSED");
        fflush(stderr);
    }
    return ok;
}

static void wt_micro_staged_call(int rows)
{
    int ims=IARG(WT_A_IMS), ime=IARG(WT_A_IME);
    int jms=IARG(WT_A_JMS), jme=IARG(WT_A_JME);
    int kms=IARG(WT_A_KMS), kme=IARG(WT_A_KME);
    int ips=IARG(WT_A_IPS), ipe=IARG(WT_A_IPE);
    int jps=IARG(WT_A_JPS), jpe=IARG(WT_A_JPE);
    int ids=IARG(WT_A_IDS), ide=IARG(WT_A_IDE);
    int jds=IARG(WT_A_JDS), jde=IARG(WT_A_JDE);
    int NI = ime-ims+1, NK = kme-kms+1, NJ = jme-jms+1;
    const char *rk = getenv("SLURM_PROCID"); if (!rk) rk = getenv("PMI_RANK");
    const char *want = getenv("WTIMER_DEBUG_RANK");
    int me = rk && atoi(rk) == (want ? atoi(want) : 0);
    int dbg = getenv("WTIMER_STAGE_DEBUG") != NULL;
    static int dbg_once; int first = dbg && !dbg_once && me; dbg_once = 1;
#define MARK(x) do{ if (first) { fprintf(stderr,"[wtimer] mark %s\n",x); fflush(stderr);} }while(0)

    if (stage_ok < 0) stage_ok = wt_stage_selfcheck(NI, NK, NJ, me);
    /* per-call 3D list: the whitelist plus the optional diagnostics Thompson fills only when
     * their flag is on (otherwise they are 1-element dummies and must not be read) */
    int w3[WT_W3D + 4], n3 = 0;
    for (int a = 0; a < WT_W3D; a++) w3[n3++] = wt_w3d[a];
    if (wt_argv[WT_A_DO_RADAR_REF] && IARG(WT_A_DO_RADAR_REF)) w3[n3++] = WT_A_REFL_10CM;
    if (wt_argv[WT_A_HAS_REQC] && IARG(WT_A_HAS_REQC)) w3[n3++] = WT_A_RE_CLOUD;
    if (wt_argv[WT_A_HAS_REQI] && IARG(WT_A_HAS_REQI)) w3[n3++] = WT_A_RE_ICE;
    if (wt_argv[WT_A_HAS_REQS] && IARG(WT_A_HAS_REQS)) w3[n3++] = WT_A_RE_SNOW;
    int refuse = !stage_ok;
    for (int a = 0; a < n3 && !refuse; a++) {
        if (!wt_argv[w3[a]]) refuse = 1;
        for (int b = 0; b < a; b++) if (wt_argv[w3[b]] == wt_argv[w3[a]]) refuse = 1;
    }
    for (int a = 0; a < WT_W2D && !refuse; a++) if (!wt_argv[wt_w2d[a]]) refuse = 1;
    /* the driver only touches columns inside the specified zone: stage exactly those */
    int sz = IARG(WT_A_SPECIFIED) ? IARG(WT_A_SPEC_ZONE) : 0;
    if (wt_argv[WT_A_CHANNEL_SWITCH] && IARG(WT_A_CHANNEL_SWITCH)) sz = 0;
    int jsplit = jpe - rows + 1;                    /* staged rows are jsplit..jpe */
    int i0 = ips > ids+sz ? ips : ids+sz, i1 = ipe < ide-1-sz ? ipe : ide-1-sz;
    int j0 = jsplit > jds+sz ? jsplit : jds+sz, j1 = jpe < jde-1-sz ? jpe : jde-1-sz;
    int NCOL = (i1 >= i0 && j1 >= j0) ? (i1-i0+1) * (j1-j0+1) : 0;
    if (first) {
        fprintf(stderr, "[wtimer] stage geometry: ims..ime=%d..%d jms..jme=%d..%d kms..kme=%d..%d"
                " ips..ipe=%d..%d jps..jpe=%d..%d sz=%d staged i=%d..%d j=%d..%d NCOL=%d n3=%d tiles=%d refuse=%d\n",
                ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, sz, i0, i1, j0, j1, NCOL, n3, IARG(WT_A_NUM_TILES), refuse);
        for (int a = 0; a < n3; a++) fprintf(stderr, "[wtimer] 3d %s=%p\n", wt_argname[w3[a]], wt_argv[w3[a]]);
        for (int a = 0; a < WT_W2D; a++) fprintf(stderr, "[wtimer] 2d %s=%p\n", wt_argname[wt_w2d[a]], wt_argv[wt_w2d[a]]);
        fflush(stderr);
    }
    if (refuse || NCOL == 0) {
        wt_micro_refused++;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        return;
    }
    size_t need = (size_t)NCOL * ((size_t)n3 * NK + WT_W2D) * sizeof(float);
    if (wt_stage_share_enabled()) {
        if (wt_stage_adapt_enabled() &&
            wt_micro_adaptive_call(rows, ims, jms, kms, NI, NK, ips, ipe, jps, jpe,
                                   ids, ide, jds, jde, sz, n3, w3, first, dbg && me))
            return;
        if (wt_micro_shared_call(jsplit, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL,
                                 n3, w3, need, first))
            return;
    }
    if (need > stage_cap) { free(stage_buf); stage_buf = malloc(need); stage_cap = need; }
    if (!stage_buf) {                                    /* fail closed: run it unstaged */
        wt_micro_refused++;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        return;
    }

    /* ---- pass 1: everything except the staged rows, through the original arrays ---- */
    void *save[WT_NARG];
    memcpy(save, wt_argv, sizeof save);
    /* the driver's OpenMP parallelism is across tiles, so both passes must hand it several:
     * pass 1 keeps the original tile list with j_end clipped (empty tiles dropped), pass 2
     * splits the staged rectangle along i into the same number of tiles */
#define WT_MAXT 64
    static int t1_i0[WT_MAXT], t1_i1[WT_MAXT], t1_j0[WT_MAXT], t1_j1[WT_MAXT], nt1;
    static int t2_i0[WT_MAXT], t2_i1[WT_MAXT], t2_j0[WT_MAXT], t2_j1[WT_MAXT], nt2;
    int nt = IARG(WT_A_NUM_TILES); if (nt > WT_MAXT) nt = WT_MAXT;
    const int *o_i0 = wt_argv[WT_A_I_START], *o_i1 = wt_argv[WT_A_I_END];
    const int *o_j0 = wt_argv[WT_A_J_START], *o_j1 = wt_argv[WT_A_J_END];
    nt1 = 0;
    for (int t = 0; t < nt; t++) {
        int je = o_j1[t] < jsplit - 1 ? o_j1[t] : jsplit - 1;
        if (je < o_j0[t]) continue;
        t1_i0[nt1] = o_i0[t]; t1_i1[nt1] = o_i1[t]; t1_j0[nt1] = o_j0[t]; t1_j1[nt1] = je; nt1++;
    }
    wt_argv[WT_A_I_START] = t1_i0; wt_argv[WT_A_I_END] = t1_i1;
    wt_argv[WT_A_J_START] = t1_j0; wt_argv[WT_A_J_END] = t1_j1;
    wt_argv[WT_A_NUM_TILES] = &nt1;
    MARK("pass1-call");
    if (nt1 > 0)
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
    memcpy(wt_argv, save, sizeof save);
    MARK("pass1-done");

    /* ---- pass 2: the staged columns, in a dense rectangle S(1:NCOL, kms:kme, 1:1) ---- */
    float *p = stage_buf;
    float *base3[WT_W3D + 4], *base2[WT_W2D];
    int NIC = i1-i0+1;
    for (int a = 0; a < n3; a++) {
        float *src = (float *)wt_argv[w3[a]];
        base3[a] = p;
        for (int k = 0; k < NK; k++)
            for (int j = j0; j <= j1; j++)
                memcpy(p + (size_t)NCOL*k + (size_t)NIC*(j-j0),
                       src + (size_t)(i0-ims) + (size_t)NI*((size_t)k + (size_t)NK*(j-jms)), (size_t)NIC * sizeof(float));
        p += (size_t)NCOL * NK;
    }
    for (int a = 0; a < WT_W2D; a++) {
        float *src = (float *)wt_argv[wt_w2d[a]];
        base2[a] = p;
        for (int j = j0; j <= j1; j++)
            memcpy(p + (size_t)NIC*(j-j0), src + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms), (size_t)NIC * sizeof(float));
        p += NCOL;
    }
    MARK("pack-done");
    static int s_ims, s_ime, s_jms, s_jme, s_ips, s_ipe, s_jps, s_jpe;
    static int s_ids, s_ide, s_jds, s_jde;
    s_ims = s_ips = s_jms = s_jps = 1; s_ime = s_ipe = NCOL; s_jme = s_jpe = 1;
    s_ids = -1000; s_ide = NCOL + 1000; s_jds = -1000; s_jde = 1 + 1000;
    for (int a = 0; a < n3; a++) wt_argv[w3[a]] = base3[a];
    for (int a = 0; a < WT_W2D; a++) wt_argv[wt_w2d[a]] = base2[a];
    wt_argv[WT_A_IMS]=&s_ims; wt_argv[WT_A_IME]=&s_ime;
    wt_argv[WT_A_JMS]=&s_jms; wt_argv[WT_A_JME]=&s_jme;
    wt_argv[WT_A_IPS]=&s_ips; wt_argv[WT_A_IPE]=&s_ipe;
    wt_argv[WT_A_JPS]=&s_jps; wt_argv[WT_A_JPE]=&s_jpe;
    wt_argv[WT_A_IDS]=&s_ids; wt_argv[WT_A_IDE]=&s_ide;
    wt_argv[WT_A_JDS]=&s_jds; wt_argv[WT_A_JDE]=&s_jde;
    nt2 = nt < NCOL ? nt : NCOL; if (nt2 < 1) nt2 = 1;
    for (int t = 0; t < nt2; t++) {
        t2_i0[t] = 1 + (int)((long)NCOL * t / nt2); t2_i1[t] = (int)((long)NCOL * (t+1) / nt2);
        t2_j0[t] = 1; t2_j1[t] = 1;
    }
    wt_argv[WT_A_I_START]=t2_i0; wt_argv[WT_A_I_END]=t2_i1;
    wt_argv[WT_A_J_START]=t2_j0; wt_argv[WT_A_J_END]=t2_j1;
    wt_argv[WT_A_NUM_TILES]=&nt2;
    MARK("pass2-call");
    wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
    MARK("pass2-done");

    /* ---- commit ---- */
    for (int a = 0; a < n3; a++) {
        float *dst = (float *)save[w3[a]];
        for (int k = 0; k < NK; k++)
            for (int j = j0; j <= j1; j++)
                memcpy(dst + (size_t)(i0-ims) + (size_t)NI*((size_t)k + (size_t)NK*(j-jms)),
                       base3[a] + (size_t)NCOL*k + (size_t)NIC*(j-j0), (size_t)NIC * sizeof(float));
    }
    for (int a = 0; a < WT_W2D; a++) {
        float *dst = (float *)save[wt_w2d[a]];
        for (int j = j0; j <= j1; j++)
            memcpy(dst + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms), base2[a] + (size_t)NIC*(j-j0), (size_t)NIC * sizeof(float));
    }
    MARK("commit-done");
    memcpy(wt_argv, save, sizeof save);
    wt_micro_staged++;
}

void wt_micro_call(void)
{
    uint64_t t0 = tsc();
    wt_pre(0);
    wt_micro_calls++;
    int rows = stage_rows();
    if (rows > 0 && IARG(WT_A_JPE) - IARG(WT_A_JPS) + 1 > rows)
        wt_micro_staged_call(rows);
    else
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                  wt_argv, WT_NARG);
    wt_record(0, tsc() - t0);
}
