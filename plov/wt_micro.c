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
extern int wt_stage_micro_enabled(void);
extern int wt_stage_grid_value(void);

uint64_t wt_micro_calls, wt_micro_staged;
uint64_t wt_micro_sig_mismatch;
int wt_micro_stage_state = -1; /* -1: not checked, 0: refused/off, 1: eligible */

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
#include "wt_plan.h"
#define IARG(k) (*(int *)wt_argv[k])

extern size_t wt_alloc_size(const void *p);

struct wt_micro_field {
    int arg;
    int gate;
};

struct wt_micro_layout {
    int mp;
    const struct wt_micro_field *w3;
    int n3;
    const struct wt_micro_field *w2;
    int n2;
    const int *moist;
    int scalar_a, scalar_b;
    const int *required;
    int nrequired;
};

#define WT_GATE_ALWAYS    -1
#define WT_GATE_WIF_BC    -2
#define WT_GATE_FIRE      -3
#define WT_GATE_FIRE_BC   -4

#define WT_FIELD(a)       { (a), WT_GATE_ALWAYS }
#define WT_FIELD_IF(a, g) { (a), (g) }

static const int wt_mp6_moist[] = {
    WT_A_QV_CURR, WT_A_QC_CURR, WT_A_QR_CURR,
    WT_A_QI_CURR, WT_A_QS_CURR, WT_A_QG_CURR
};
static const int wt_mp8_moist[] = {
    WT_A_QV_CURR, WT_A_QC_CURR, WT_A_QR_CURR,
    WT_A_QI_CURR, WT_A_QS_CURR, WT_A_QG_CURR
};
static const int wt_mp28_moist[] = {
    WT_A_QV_CURR, WT_A_QC_CURR, WT_A_QR_CURR,
    WT_A_QI_CURR, WT_A_QS_CURR, WT_A_QG_CURR
};

static const int wt_mp6_required[] = {
    WT_A_F_QV, WT_A_F_QC, WT_A_F_QR, WT_A_F_QI, WT_A_F_QS, WT_A_F_QG
};
static const int wt_mp8_required[] = {
    WT_A_F_QV, WT_A_F_QC, WT_A_F_QR, WT_A_F_QI, WT_A_F_QS, WT_A_F_QG,
    WT_A_F_QNI, WT_A_F_QNR
};
static const int wt_mp28_required[] = {
    WT_A_F_QV, WT_A_F_QC, WT_A_F_QR, WT_A_F_QI, WT_A_F_QS, WT_A_F_QG,
    WT_A_F_QNI, WT_A_F_QNR, WT_A_F_QNC, WT_A_F_QNWFA, WT_A_F_QNIFA
};

/* Keep the mp=8 entries in the exact historical order.  Fields whose Registry
 * package can be absent are table entries with a runtime gate, not ad hoc
 * branches in the packing code. */
static const struct wt_micro_field wt_mp8_w3[] = {
    WT_FIELD(WT_A_TH), WT_FIELD(WT_A_PI_PHY), WT_FIELD(WT_A_P),
    WT_FIELD(WT_A_W), WT_FIELD(WT_A_DZ8W),
    WT_FIELD(WT_A_QV_CURR), WT_FIELD(WT_A_QC_CURR), WT_FIELD(WT_A_QR_CURR),
    WT_FIELD(WT_A_QI_CURR), WT_FIELD(WT_A_QS_CURR), WT_FIELD(WT_A_QG_CURR),
    WT_FIELD(WT_A_QNI_CURR), WT_FIELD(WT_A_QNR_CURR),
    WT_FIELD_IF(WT_A_REFL_10CM, WT_A_DO_RADAR_REF),
    WT_FIELD_IF(WT_A_RE_CLOUD, WT_A_HAS_REQC),
    WT_FIELD_IF(WT_A_RE_ICE, WT_A_HAS_REQI),
    WT_FIELD_IF(WT_A_RE_SNOW, WT_A_HAS_REQS)
};
static const struct wt_micro_field wt_mp8_w2[] = {
    WT_FIELD(WT_A_RAINNC), WT_FIELD(WT_A_RAINNCV), WT_FIELD(WT_A_SNOWNC),
    WT_FIELD(WT_A_SNOWNCV), WT_FIELD(WT_A_GRAUPELNC), WT_FIELD(WT_A_GRAUPELNCV),
    WT_FIELD(WT_A_SR)
};

static const struct wt_micro_field wt_mp6_w3[] = {
    WT_FIELD(WT_A_TH), WT_FIELD(WT_A_RHO), WT_FIELD(WT_A_PI_PHY),
    WT_FIELD(WT_A_P), WT_FIELD(WT_A_W), WT_FIELD(WT_A_DZ8W),
    WT_FIELD(WT_A_QV_CURR), WT_FIELD(WT_A_QC_CURR), WT_FIELD(WT_A_QR_CURR),
    WT_FIELD(WT_A_QI_CURR), WT_FIELD(WT_A_QS_CURR), WT_FIELD(WT_A_QG_CURR),
    WT_FIELD_IF(WT_A_REFL_10CM, WT_A_DO_RADAR_REF),
    WT_FIELD_IF(WT_A_RE_CLOUD, WT_A_HAS_REQC),
    WT_FIELD_IF(WT_A_RE_ICE, WT_A_HAS_REQI),
    WT_FIELD_IF(WT_A_RE_SNOW, WT_A_HAS_REQS)
};
static const struct wt_micro_field wt_mp6_w2[] = {
    WT_FIELD(WT_A_RAINNC), WT_FIELD(WT_A_RAINNCV), WT_FIELD(WT_A_SNOWNC),
    WT_FIELD(WT_A_SNOWNCV), WT_FIELD(WT_A_GRAUPELNC), WT_FIELD(WT_A_GRAUPELNCV),
    WT_FIELD(WT_A_SR)
};

static const struct wt_micro_field wt_mp28_w3[] = {
    WT_FIELD(WT_A_TH), WT_FIELD(WT_A_PI_PHY), WT_FIELD(WT_A_P),
    WT_FIELD(WT_A_W), WT_FIELD(WT_A_DZ8W),
    WT_FIELD(WT_A_QV_CURR), WT_FIELD(WT_A_QC_CURR), WT_FIELD(WT_A_QR_CURR),
    WT_FIELD(WT_A_QI_CURR), WT_FIELD(WT_A_QS_CURR), WT_FIELD(WT_A_QG_CURR),
    WT_FIELD(WT_A_QNI_CURR), WT_FIELD(WT_A_QNR_CURR),
    WT_FIELD_IF(WT_A_QNC_CURR, WT_A_F_QNC),
    WT_FIELD_IF(WT_A_QNWFA_CURR, WT_A_F_QNWFA),
    WT_FIELD_IF(WT_A_QNIFA_CURR, WT_A_F_QNIFA),
    WT_FIELD_IF(WT_A_QNBCA_CURR, WT_GATE_WIF_BC),
    WT_FIELD_IF(WT_A_Z_AT_MASS, WT_GATE_FIRE),
    WT_FIELD_IF(WT_A_REFL_10CM, WT_A_DO_RADAR_REF),
    WT_FIELD_IF(WT_A_RE_CLOUD, WT_A_HAS_REQC),
    WT_FIELD_IF(WT_A_RE_ICE, WT_A_HAS_REQI),
    WT_FIELD_IF(WT_A_RE_SNOW, WT_A_HAS_REQS)
};
static const struct wt_micro_field wt_mp28_w2[] = {
    WT_FIELD(WT_A_RAINNC), WT_FIELD(WT_A_RAINNCV), WT_FIELD(WT_A_SNOWNC),
    WT_FIELD(WT_A_SNOWNCV), WT_FIELD(WT_A_GRAUPELNC), WT_FIELD(WT_A_GRAUPELNCV),
    WT_FIELD(WT_A_SR),
    WT_FIELD_IF(WT_A_QNWFA2D, WT_A_F_QNWFA),
    WT_FIELD_IF(WT_A_QNIFA2D, WT_A_F_QNIFA),
    WT_FIELD_IF(WT_A_QNBCA2D, WT_GATE_WIF_BC),
    WT_FIELD_IF(WT_A_PBLH, WT_GATE_FIRE),
    WT_FIELD_IF(WT_A_QNOCBB2D, WT_GATE_FIRE),
    WT_FIELD_IF(WT_A_QNBCBB2D, WT_GATE_FIRE_BC)
};

static const struct wt_micro_layout wt_micro_layouts[] = {
    { 8, wt_mp8_w3, (int)(sizeof wt_mp8_w3 / sizeof *wt_mp8_w3),
         wt_mp8_w2, (int)(sizeof wt_mp8_w2 / sizeof *wt_mp8_w2),
         wt_mp8_moist, WT_A_QNI_CURR, WT_A_QNR_CURR,
         wt_mp8_required, (int)(sizeof wt_mp8_required / sizeof *wt_mp8_required) },
    { 6, wt_mp6_w3, (int)(sizeof wt_mp6_w3 / sizeof *wt_mp6_w3),
         wt_mp6_w2, (int)(sizeof wt_mp6_w2 / sizeof *wt_mp6_w2),
         wt_mp6_moist, -1, -1,
         wt_mp6_required, (int)(sizeof wt_mp6_required / sizeof *wt_mp6_required) },
    { 28, wt_mp28_w3, (int)(sizeof wt_mp28_w3 / sizeof *wt_mp28_w3),
         wt_mp28_w2, (int)(sizeof wt_mp28_w2 / sizeof *wt_mp28_w2),
         wt_mp28_moist, WT_A_QNI_CURR, WT_A_QNR_CURR,
         wt_mp28_required, (int)(sizeof wt_mp28_required / sizeof *wt_mp28_required) }
};

static const struct wt_micro_layout *stage_layout;
static int stage_w3[32], stage_w2[16], stage_n3, stage_n2;
static int stage_alloc_ok = 1;
static int stage_wif_bc, stage_fire, stage_fire_bc;

static size_t wt_micro_full_2d(int NI, int NJ)
{
    size_t n;
    if (NI <= 0 || NJ <= 0 || (size_t)NI > SIZE_MAX / (size_t)NJ) return 0;
    n = (size_t)NI * (size_t)NJ;
    return n <= SIZE_MAX / sizeof(float) ? n * sizeof(float) : 0;
}

static size_t wt_micro_full_3d(int NI, int NK, int NJ)
{
    size_t n = wt_micro_full_2d(NI, NJ);
    if (!n || NK <= 0 || n / sizeof(float) > SIZE_MAX / (size_t)NK) return 0;
    n = (n / sizeof(float)) * (size_t)NK;
    return n <= SIZE_MAX / sizeof(float) ? n * sizeof(float) : 0;
}

/* WIF and fire state fields are allocated either at full patch size or as a
 * one-element Registry dummy.  Treat any other or unknown size as unsafe. */
static int wt_micro_alloc_state(int arg, size_t full, int *active)
{
    size_t got;
    if (!wt_argv[arg] || full <= sizeof(float)) return 0;
    got = wt_alloc_size(wt_argv[arg]);
    if (got == full) { *active = 1; return 1; }
    if (got == sizeof(float)) { *active = 0; return 1; }
    return 0;
}

/* Not-a-dummy check for a staged field.  Species (qv..qg, qni, qnr, ...) are slices of a
 * 4-D moist/scalar allocation, so their base pointer is either unknown to wt_alloc (0) or
 * the whole 4-D block; an exact match against the 3-D patch size cannot be required.  What
 * can be refused is an exact 4-byte Registry dummy or a recorded block smaller than one
 * patch (that is the size-1-dummy trap, see wt_micro_alloc_state). */
static int wt_micro_alloc_not_dummy(int arg, size_t full)
{
    size_t got;
    if (!wt_argv[arg] || full == 0) return 0;
    got = wt_alloc_size(wt_argv[arg]);
    if (got == sizeof(float)) return 0;
    if (got && got < full) return 0;
    return 1;
}

static int wt_micro_field_active(const struct wt_micro_field *f)
{
    if (f->gate == WT_GATE_WIF_BC) return stage_wif_bc;
    if (f->gate == WT_GATE_FIRE) return stage_fire;
    if (f->gate == WT_GATE_FIRE_BC) return stage_fire_bc;
    return f->gate < 0 || (wt_argv[f->gate] && IARG(f->gate));
}

static const struct wt_micro_layout *wt_micro_find_layout(int mp)
{
    for (unsigned i = 0; i < sizeof wt_micro_layouts / sizeof *wt_micro_layouts; i++)
        if (wt_micro_layouts[i].mp == mp) return &wt_micro_layouts[i];
    return NULL;
}

static void wt_micro_select_fields(int NI, int NK, int NJ)
{
    size_t full2 = 0, full3 = 0;
    stage_n3 = stage_n2 = 0;
    stage_alloc_ok = 1;
    stage_wif_bc = stage_fire = stage_fire_bc = 0;
    if (!stage_layout) return;
    if (stage_layout->mp == 28) {
        full2 = wt_micro_full_2d(NI, NJ);
        full3 = wt_micro_full_3d(NI, NK, NJ);
        stage_alloc_ok = full2 && full3 &&
            wt_micro_alloc_state(WT_A_QNBCA2D, full2, &stage_wif_bc) &&
            wt_micro_alloc_state(WT_A_QNOCBB2D, full2, &stage_fire) &&
            wt_micro_alloc_state(WT_A_QNBCBB2D, full2, &stage_fire_bc);
        if (stage_fire &&
            (!wt_argv[WT_A_Z_AT_MASS] || wt_alloc_size(wt_argv[WT_A_Z_AT_MASS]) != full3 ||
             !wt_argv[WT_A_PBLH] || wt_alloc_size(wt_argv[WT_A_PBLH]) != full2))
            stage_alloc_ok = 0;
    }
    for (int i = 0; i < stage_layout->n3; i++)
        if (wt_micro_field_active(&stage_layout->w3[i])) stage_w3[stage_n3++] = stage_layout->w3[i].arg;
    for (int i = 0; i < stage_layout->n2; i++)
        if (wt_micro_field_active(&stage_layout->w2[i])) stage_w2[stage_n2++] = stage_layout->w2[i].arg;
    if (stage_layout->mp == 28 && stage_alloc_ok) {
        for (int i = 0; i < stage_n3; i++)
            if (!wt_micro_alloc_not_dummy(stage_w3[i], full3)) stage_alloc_ok = 0;
        for (int i = 0; i < stage_n2; i++)
            if (!wt_micro_alloc_not_dummy(stage_w2[i], full2)) stage_alloc_ok = 0;
    }
}

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
#define WT_MICRO_SHARE_LAYOUT_VERSION 1u
typedef struct __attribute__((aligned(64))) {
    uint64_t work_ticks;
    int state, ncol, nk, n3, n2, owner, step;
    uint32_t layout_version;
    int mp, ni, rows, grid;
    int reserved[2];
} wt_share_hdr;
_Static_assert(sizeof(wt_share_hdr) == 64, "slot header must stay one cache line: payload offset is sizeof *h");
typedef struct {
    uint64_t t_full_ticks, step;
    uint64_t pred_ticks;   /* EWMA(0.7) of this rank's own micro work: pass 1 + its chunks, wherever computed */
    int64_t nj;            /* stageable j-rows of this rank's patch */
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
/* Predictive PLOV (WTIMER_STAGE_PREDICT): 0 reactive adaptive (production), 1 predicted R,
 * 2 predicted R + node-local owner->donor assignment taken after the donor's own work,
 * 3 as 2 but the donor takes its assigned chunks at micro entry, before its own work. */
static double pred_own_ewma;
static int predict_mode = -1, trace_on = -1;
static FILE *trace_fp;
static int wt_predict_mode(void)
{
    if (predict_mode < 0) {
        const char *e = getenv("WTIMER_STAGE_PREDICT");
        predict_mode = e ? atoi(e) : 0;
        if (predict_mode < 0 || predict_mode > 3) predict_mode = 0;
    }
    return predict_mode;
}
static int wt_trace_on(void)
{
    if (trace_on < 0) { const char *e = getenv("WTIMER_TRACE"); trace_on = e && e[0] == '1' && !e[1]; }
    return trace_on;
}

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

static void wt_share_set_signature(wt_share_hdr *h, int mp, int NI, int NK,
                                   int n3, int n2, int rows, int grid)
{
    h->layout_version = WT_MICRO_SHARE_LAYOUT_VERSION;
    h->mp = mp;
    h->ni = NI;
    h->grid = grid;
    h->nk = NK;
    h->n3 = n3;
    h->n2 = n2;
    h->rows = rows;
}

static int wt_share_signature_ok(const wt_share_hdr *h, int mp, int NI, int NK,
                                 int n3, int n2, int grid, int min_rows, int max_rows)
{
    /* ni and rows are recorded for diagnosis only: the payload is a self-contained list of
     * ncol columns, so a neighbour with a different i-extent, or an edge rank whose
     * spec-zone clip left it fewer rows, is still a valid donor (it was before the header
     * carried a signature, and refusing it would silently cut the steal rate). */
    (void)NI; (void)min_rows; (void)max_rows;
    return h->layout_version == WT_MICRO_SHARE_LAYOUT_VERSION &&
           h->mp == mp && h->grid == grid && h->nk == NK &&
           h->n3 == n3 && h->n2 == n2 && h->rows > 0;
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
        __atomic_store_n(&share_stats[me].pred_ticks, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&share_stats[me].nj, 0, __ATOMIC_RELAXED);
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
                          int n3, const int *w3, int n2, const int *w2)
{
    float *p = buf;
    (void)kms;
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
    for (int a = 0; a < n2; a++) {
        float *src = (float *)wt_argv[w2[a]];
        for (int j = j0; j <= j1; j++)
            memcpy(p + (size_t)NIC*(j-j0),
                   src + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms),
                   (size_t)NIC * sizeof(float));
        p += NCOL;
    }
}

static void wt_share_commit(float *buf, void **save, int ims, int jms, int NI, int NK,
                            int i0, int i1, int j0, int j1, int NCOL,
                            int n3, const int *w3, int n2, const int *w2)
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
    for (int a = 0; a < n2; a++) {
        float *dst = (float *)save[w2[a]];
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
                           int n2, const int *w2,
                           void **save, int nt, int first)
{
    static int s_ims, s_ime, s_jms, s_jme, s_ips, s_ipe, s_jps, s_jpe;
    static int s_ids, s_ide, s_jds, s_jde;
    static int i0[64], i1[64], j0[64], j1[64], nt2;
    float *p = buf;
    (void)save;
    s_ims = s_ips = s_jms = s_jps = 1; s_ime = s_ipe = ncol; s_jme = s_jpe = 1;
    s_ids = -1000; s_ide = ncol + 1000; s_jds = -1000; s_jde = 1 + 1000;
    for (int a = 0; a < n3; a++) { wt_argv[w3[a]] = p; p += (size_t)ncol * NK; }
    for (int a = 0; a < n2; a++) { wt_argv[w2[a]] = p; p += ncol; }
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

static int wt_share_steal(int mp, int NI, int NK, const int *w3, int n3,
                          const int *w2, int n2, int grid, int rows, int nt, int first)
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
            if (!wt_share_signature_ok(h, mp, NI, NK, n3, n2, grid, rows, rows)) {
                wt_micro_sig_mismatch++;
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            if (h->ncol <= 0 ||
                (size_t)h->ncol > share_cap / (((size_t)n3 * NK + n2) * sizeof(float))) {
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            void *save[WT_NARG];
            memcpy(save, wt_argv, sizeof save);
            uint64_t t0 = tsc();
            wt_micro_share_in++;
            wt_share_pass2((float *)((char *)h + sizeof *h), h->ncol, NK, n3, w3,
                           n2, w2, save, nt, first);
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

static int wt_adapt_need_fits(int NIC, int rows, int NK, int n3, int n2, size_t cap)
{
    size_t fields = (size_t)n3 * (size_t)NK + (size_t)n2;
    if (NIC <= 0 || rows <= 0 || fields > SIZE_MAX / sizeof(float)) return 0;
    if ((size_t)NIC > SIZE_MAX / (size_t)rows) return 0;
    size_t cols = (size_t)NIC * (size_t)rows;
    if (cols > SIZE_MAX / fields) return 0;
    size_t elems = cols * fields;
    return elems <= cap / sizeof(float);
}

static int wt_adapt_steal(int mp, int NI, int NK, const int *w3, int n3,
                          const int *w2, int n2, int grid, int max_rows, int nt, int first,
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
            if (!wt_share_signature_ok(h, mp, NI, NK, n3, n2, grid, 1, max_rows)) {
                wt_micro_sig_mismatch++;
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            if (h->ncol <= 0 ||
                !wt_adapt_need_fits(1, h->ncol, NK, n3, n2, share_cap)) {
                __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                continue;
            }
            void *save[WT_NARG];
            memcpy(save, wt_argv, sizeof save);
            uint64_t work0 = tsc();
            wt_micro_share_in++;
            wt_micro_chunks_in++;
            wt_share_pass2((float *)((char *)h + sizeof *h), h->ncol, NK, n3, w3,
                           n2, w2, save, nt, first);
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

/* Take chunk c of local owner o if it is PUBLISHED now; returns its work ticks, 0 if not taken.
 * Same gates, save/restore and state transitions as wt_adapt_steal, for one addressed slot. */
static uint64_t wt_take_chunk(int o, int c, int mp, int NI, int NK, const int *w3, int n3,
                              const int *w2, int n2, int grid, int max_rows, int nt, int first)
{
    if (o < 0 || o >= share_nlocal || o == share_me || c < 0 || c >= share_k) return 0;
    wt_share_hdr *h = wt_share_slot(o * share_k + c);
    int expected = WT_SHARE_PUBLISHED;
    if (__atomic_load_n(&h->state, __ATOMIC_ACQUIRE) != WT_SHARE_PUBLISHED) return 0;
    if (!__atomic_compare_exchange_n(&h->state, &expected, WT_SHARE_CLAIMED_OTHER, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    if (!wt_share_signature_ok(h, mp, NI, NK, n3, n2, grid, 1, max_rows)) {
        wt_micro_sig_mismatch++;
        __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
        return 0;
    }
    if (h->ncol <= 0 || !wt_adapt_need_fits(1, h->ncol, NK, n3, n2, share_cap)) {
        __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
        return 0;
    }
    void *save[WT_NARG];
    memcpy(save, wt_argv, sizeof save);
    uint64_t work0 = tsc();
    wt_micro_share_in++;
    wt_micro_chunks_in++;
    wt_share_pass2((float *)((char *)h + sizeof *h), h->ncol, NK, n3, w3, n2, w2, save, nt, first);
    memcpy(wt_argv, save, sizeof save);
    uint64_t work = tsc() - work0;
    if (!work) work = 1;
    __atomic_store_n(&h->work_ticks, work, __ATOMIC_RELEASE);
    wt_micro_share_work_ticks += work;
    __atomic_store_n(&h->state, WT_SHARE_DONE, __ATOMIC_RELEASE);
    return work;
}

static int wt_micro_adaptive_call(int CH, int mp, int grid, int ims, int jms, int kms, int NI, int NK,
                                  int ips, int ipe, int jps, int jpe,
                                  int ids, int ide, int jds, int jde, int sz,
                                  int n3, const int *w3, int n2, const int *w2,
                                  int first, int debug_rank)
{
    int64_t i0x = (int64_t)ips > (int64_t)ids + sz ? ips : (int64_t)ids + sz;
    int64_t i1x = (int64_t)ipe < (int64_t)ide - 1 - sz ? ipe : (int64_t)ide - 1 - sz;
    int64_t jp0x = (int64_t)jps > (int64_t)jds + sz ? jps : (int64_t)jds + sz;
    int64_t jp1x = (int64_t)jpe < (int64_t)jde - 1 - sz ? jpe : (int64_t)jde - 1 - sz;
    int i0, i1, jp0, jp1, NIC, NJp;
    int K = wt_share_chunks_from_env();
    size_t cap = wt_share_cap_from_env();
    size_t fields = (size_t)n3 * (size_t)NK + (size_t)n2;
    size_t max_need = 0;
    int max_rows;
    uint64_t call_start = tsc();
    uint64_t step = wt_micro_calls + 1;             /* wt_micro_calls++ happens after the call; 0 is the invalid sentinel */
    uint64_t previous = step ? step - 1 : 0, t_me, t_mean;
    int nvalid, R, nchunks = 0;
    int nt = IARG(WT_A_NUM_TILES); if (nt > 64) nt = 64;

    if (i0x < INT_MIN || i0x > INT_MAX || i1x < INT_MIN || i1x > INT_MAX ||
        jp0x < INT_MIN || jp0x > INT_MAX || jp1x < INT_MIN || jp1x > INT_MAX)
        return 0;
    i0 = (int)i0x; i1 = (int)i1x; jp0 = (int)jp0x; jp1 = (int)jp1x;
    NIC = i1 - i0 + 1;
    NJp = jp1 >= jp0 ? jp1 - jp0 + 1 : 0;
    max_rows = CH < NJp ? CH : NJp;

    if (debug_rank) wt_adapt_calibrate();
    { const char *ge = getenv("WTIMER_STAGE_GAIN"); if (ge && atof(ge) > 0) adapt_gain = atof(ge); }
    if (K <= 0 || !wt_adapt_need_fits(NIC, max_rows, NK, n3, n2, cap)) return 0;
    max_need = (size_t)NIC * (size_t)max_rows * fields * sizeof(float);
    if (!wt_share_init(max_need, first, 1)) return 0;
    wt_adapt_read_stats(previous, &t_me, &t_mean, &nvalid);
    int mode = wt_predict_mode(), tron = wt_trace_on();
    if ((mode || tron) && !debug_rank) wt_adapt_calibrate();
    double hz = adapt_tsc_hz > 0 ? adapt_tsc_hz : 2.0e9;
    double pred_prev = pred_own_ewma;
    static wt_plan plan;
    int plan_ok = 0;
    plan.nslot = 0;
    if (mode) {
        double predv[WT_PLAN_MAXR];
        int njv[WT_PLAN_MAXR];
        int n = share_nlocal < WT_PLAN_MAXR ? share_nlocal : WT_PLAN_MAXR;
        for (int r = 0; r < n; r++) {
            wt_share_stats *st = &share_stats[r];
            uint64_t s1 = __atomic_load_n(&st->step, __ATOMIC_ACQUIRE);
            uint64_t pv = __atomic_load_n(&st->pred_ticks, __ATOMIC_ACQUIRE);
            int64_t nv = __atomic_load_n(&st->nj, __ATOMIC_ACQUIRE);
            uint64_t s2 = __atomic_load_n(&st->step, __ATOMIC_ACQUIRE);
            int ok = previous && s1 == previous && s1 == s2 && pv > 0 && nv > 0 && nv < INT_MAX;
            predv[r] = ok ? (double)pv : 0.0;
            njv[r] = ok ? (int)nv : 0;
        }
        const char *pe = getenv("WTIMER_STAGE_MAX_PCT");
        int ppct = pe ? atoi(pe) : 50;
        if (ppct < 0) ppct = 0;
        if (ppct > 100) ppct = 100;
        const char *oe = getenv("WTIMER_PREDICT_OVH_MS");
        double ovh_ms = oe ? atof(oe) : 1.0;
        wt_plan_build(&plan, n, predv, njv, CH, K, ppct / 100.0, ovh_ms * hz / 1000.0);
        plan_ok = plan.valid * 2 >= share_nlocal && share_me < plan.n;
    }

    /* Feedback on the MEASURED wrapped-call time (pack + passes + waits + steals), not on an
     * open-loop estimate: job 1070120 showed a rank giving 34% of its rows dropped only
     * 45.2 -> 37.3 ms (predicted 29.9) because the tail rows are colder than its average and
     * the clipped tile list is uneven across threads. Integrate until t_call meets the node mean. */
    if (adapt_R_prev < 0) adapt_R_prev = CH;
    if (mode && plan_ok) {
        R = plan.chunks_out[share_me] * CH;         /* frozen per-rank EWMA predictor + node-local plan */
    } else if (nvalid * 2 < share_nlocal || !t_me) {
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

    int des_n = 0, des_owner[WT_PLAN_MAXS], des_chunk[WT_PLAN_MAXS], des_done[WT_PLAN_MAXS];
    double assigned_out_ms = 0, assigned_in_ms = 0;
    char role = 'N';
    if (mode && plan_ok) {
        assigned_out_ms = plan.assigned_out_ms[share_me] * 1e3 / hz;
        assigned_in_ms = plan.assigned_in_ms[share_me] * 1e3 / hz;
        role = plan.chunks_out[share_me] ? 'O' : plan.chunks_in[share_me] ? 'D' : 'N';
    }
    if (mode >= 2 && plan_ok) {
        int off[WT_PLAN_MAXR];
        memset(off, 0, sizeof off);
        for (int k = 0; k < plan.nslot; k++) {
            for (int c = 0; c < plan.slot[k].chunks; c++) {
                int idx = off[plan.slot[k].owner]++;
                if (plan.slot[k].donor == share_me && des_n < WT_PLAN_MAXS && idx < share_k) {
                    des_owner[des_n] = plan.slot[k].owner;
                    des_chunk[des_n] = idx;
                    des_done[des_n] = 0;
                    des_n++;
                }
            }
        }
    }
    uint64_t out0 = wt_micro_share_out, in0 = wt_micro_share_in;
    uint64_t waitk0 = wt_micro_share_wait_ticks, workk0 = wt_micro_share_work_ticks;
    uint64_t idle_ticks = 0, pack_ticks = 0, commit_ticks = 0;
    if (mode == 3 && des_n) {
        const char *we = getenv("WTIMER_PREDICT_WAIT_US");
        double wait_us = we ? atof(we) : 2000.0;
        uint64_t deadline = call_start + (uint64_t)(wait_us * 1e-6 * hz);
        int left = des_n;
        unsigned spin = 0;
        while (left) {
            int progress = 0;
            for (int i = 0; i < des_n; i++) {
                if (des_done[i]) continue;
                if (wt_take_chunk(des_owner[i], des_chunk[i], mp, NI, NK, w3, n3, w2, n2, grid, CH, nt, first)) {
                    des_done[i] = 1; left--; progress = 1;
                }
            }
            if (!left || progress) continue;
            if (tsc() >= deadline) break;
            uint64_t p0 = tsc();
            wt_share_pause(spin++);
            idle_ticks += tsc() - p0;
        }
    }

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
        uint64_t pack0 = tsc();
        for (int c = 0; c < nchunks; c++) {
            int cj0 = jsplit + c * CH;
            int cj1 = cj0 + CH - 1;
            if (cj1 > jp1) cj1 = jp1;
            int rows = cj1 - cj0 + 1;
            int ncol = NIC * rows;
            wt_share_hdr *h = wt_share_slot(share_me * share_k + c);
            wt_share_pack((float *)((char *)h + sizeof *h), ims, jms, kms, NI, NK,
                          i0, i1, cj0, cj1, ncol, n3, w3, n2, w2);
            wt_share_set_signature(h, mp, NI, NK, n3, n2, rows, grid);
            h->ncol = ncol;
            h->owner = share_me; h->step = (int)step;
            __atomic_store_n(&h->work_ticks, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
        }
        pack_ticks = tsc() - pack0;
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
                wt_share_pass2((float *)((char *)h + sizeof *h), ncol, NK, n3, w3,
                               n2, w2, save, nt, first);
                uint64_t work = tsc() - work0;
                __atomic_store_n(&h->work_ticks, work, __ATOMIC_RELEASE);
                own_work += work;
                uint64_t cm0 = tsc();
                wt_share_commit((float *)((char *)h + sizeof *h), save, ims, jms, NI, NK,
                                i0, i1, cj0, cj1, ncol, n3, w3, n2, w2);
                commit_ticks += tsc() - cm0;
                memcpy(wt_argv, save, sizeof save);
                __atomic_store_n(&h->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            } else {
                wt_micro_share_out++;
                wt_micro_share_wait_ticks += tsc() - wait0;
                uint64_t work = __atomic_load_n(&h->work_ticks, __ATOMIC_ACQUIRE);
                own_work += work;
                uint64_t cm0 = tsc();
                wt_share_commit((float *)((char *)h + sizeof *h), save, ims, jms, NI, NK,
                                i0, i1, cj0, cj1, ncol, n3, w3, n2, w2);
                commit_ticks += tsc() - cm0;
                memcpy(wt_argv, save, sizeof save);
                __atomic_store_n(&h->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            }
        }
        t_full += own_work;
        wt_micro_staged++;
    }
    uint64_t own_ticks = t_full, own_done = tsc();
    if (mode >= 2 && plan_ok) {
        uint64_t sp0 = tsc(), wsum = 0;
        int got = 0;
        for (int i = 0; i < des_n; i++) {
            if (des_done[i]) continue;
            uint64_t w = wt_take_chunk(des_owner[i], des_chunk[i], mp, NI, NK, w3, n3, w2, n2, grid, CH, nt, first);
            if (w) { des_done[i] = 1; wsum += w; got++; }
        }
        uint64_t spent = tsc() - sp0;
        idle_ticks += spent > wsum ? spent - wsum : 0;
        adapt_stole_prev = got;
    } else {
        uint64_t sp0 = tsc(), wk0 = wt_micro_share_work_ticks;
        adapt_stole_prev = wt_adapt_steal(mp, NI, NK, w3, n3, w2, n2, grid, CH, nt, first, call_start,
                                           nvalid * 2 >= share_nlocal ? t_mean : 0);
        uint64_t spent = tsc() - sp0, wk = wt_micro_share_work_ticks - wk0;
        idle_ticks += spent > wk ? spent - wk : 0;
    }
    (void)t_full;
    t_full = tsc() - call_start;                    /* published sample = whole wrapped call */
    if (!t_full) t_full = 1;
    /* Invalidate only now, not at entry: ranks enter micro within ms of each other in random
     * order, and an early invalidation would leave the last rank in with no valid neighbours.
     * Readers reject a previous step number paired with a half-written t_full_ticks. */
    if (own_ticks)
        pred_own_ewma = pred_own_ewma > 0 ? 0.7 * (double)own_ticks + 0.3 * pred_own_ewma : (double)own_ticks;
    __atomic_store_n(&share_stats[share_me].step, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&share_stats[share_me].t_full_ticks, t_full, __ATOMIC_RELAXED);
    __atomic_store_n(&share_stats[share_me].pred_ticks, (uint64_t)pred_own_ewma, __ATOMIC_RELAXED);
    __atomic_store_n(&share_stats[share_me].nj, (int64_t)NJp, __ATOMIC_RELAXED);
    __atomic_store_n(&share_stats[share_me].step, step, __ATOMIC_RELEASE);
    if (tron) {
        if (!trace_fp) {
            const char *dir = getenv("WTIMER_DIR"), *rk = getenv("SLURM_PROCID");
            char path[4096];
            if (!rk) rk = getenv("PMI_RANK");
            snprintf(path, sizeof path, "%s/plov.%s.csv", dir ? dir : ".", rk ? rk : "x");
            trace_fp = fopen(path, "w");
            if (trace_fp)
                fprintf(trace_fp, "step,mode,role,entry_tsc,own_done_tsc,exit_tsc,pred_ms,rows_pub,chunks_pub,chunks_out,"
                                  "chunks_in,work_in_ms,assigned_out_ms,assigned_in_ms,wait_ms,idle_ms,pack_ms,commit_ms\n");
        }
        if (trace_fp) {
            char rl = mode ? role : (R > 0 ? 'O' : (wt_micro_share_in > in0 ? 'D' : 'N'));
            fprintf(trace_fp, "%llu,%d,%c,%llu,%llu,%llu,%.4f,%d,%d,%llu,%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    (unsigned long long)step, mode, rl, (unsigned long long)call_start, (unsigned long long)own_done,
                    (unsigned long long)tsc(), pred_prev * 1e3 / hz, R, nchunks,
                    (unsigned long long)(wt_micro_share_out - out0), (unsigned long long)(wt_micro_share_in - in0),
                    (double)(wt_micro_share_work_ticks - workk0) * 1e3 / hz, assigned_out_ms, assigned_in_ms,
                    (double)(wt_micro_share_wait_ticks - waitk0) * 1e3 / hz, (double)idle_ticks * 1e3 / hz,
                    (double)pack_ticks * 1e3 / hz, (double)commit_ticks * 1e3 / hz);
            fflush(trace_fp);
        }
    }
    return 1;
}

static int wt_micro_shared_call(int jsplit, int ims, int jms, int kms, int NI, int NK,
                                int i0, int i1, int j0, int j1, int NCOL,
                                int n3, const int *w3, int n2, const int *w2,
                                size_t need, int mp, int grid, int first)
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
        wt_share_pack(buf, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL,
                      n3, w3, n2, w2);
        wt_share_set_signature(own, mp, NI, NK, n3, n2, j1 - j0 + 1, grid);
        own->ncol = NCOL;
        own->owner = share_me; own->step = (int)++share_step;
        wt_share_mark(first, "publish");
        __atomic_store_n(&own->state, WT_SHARE_PUBLISHED, __ATOMIC_RELEASE);
    }
    wt_share_pass1(save, jsplit, first);
    if (!publish) {
        wt_share_pack(buf, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL,
                      n3, w3, n2, w2);
        wt_share_mark(first, "pack-done");
        wt_share_pass2(buf, NCOL, NK, n3, w3, n2, w2, save, nt, first);
        wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL,
                        n3, w3, n2, w2);
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
            wt_share_pass2(buf, NCOL, NK, n3, w3, n2, w2, save, nt, first);
            wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL,
                            n3, w3, n2, w2);
            memcpy(wt_argv, save, sizeof save);
            __atomic_store_n(&own->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            wt_share_mark(first, "own-done");
        } else {
            wt_micro_share_out++;
            wt_micro_share_wait_ticks += tsc() - wait0;
            wt_share_commit(buf, save, ims, jms, NI, NK, i0, i1, j0, j1, NCOL,
                            n3, w3, n2, w2);
            memcpy(wt_argv, save, sizeof save);
            __atomic_store_n(&own->state, WT_SHARE_EMPTY, __ATOMIC_RELEASE);
            wt_share_mark(first, "own-done");
        }
    }
    /* w3/n3 came from this rank's flags above; a slot contributes packed bytes, never its owner's
     * pointer list or scalar state. */
    wt_share_steal(mp, NI, NK, w3, n3, w2, n2, grid, j1 - j0 + 1, nt, first);
    wt_micro_staged++;
    return 1;
}

static int stage_rows(void)   /* how many of the patch's j-rows to stage; 0 = off */
{
    const char *e = getenv("WTIMER_STAGE_ROWS");
    return e ? atoi(e) : 0;
}

static int wt_stage_grid_selected(void)
{
    int grid = wt_stage_grid_value();
    return grid == 0 || (wt_argv[WT_A_ID] && IARG(WT_A_ID) == grid);
}

uint64_t wt_micro_refused;      /* staging requested but the self-check said no: ran unstaged */

static int wt_micro_extent(int lo, int hi, int *extent)
{
    int64_t n = (int64_t)hi - (int64_t)lo + 1;
    if (n <= 0 || n > INT_MAX) return 0;
    *extent = (int)n;
    return 1;
}

/* Validate the original frame before choosing pass 1/pass 2.  All later
 * indexing assumes these extents are representable positive int values. */
static int wt_micro_geometry_ok(int rows, int *jrange)
{
    const int scalar[] = {
        WT_A_IMS, WT_A_IME, WT_A_JMS, WT_A_JME, WT_A_KMS, WT_A_KME,
        WT_A_IPS, WT_A_IPE, WT_A_JPS, WT_A_JPE, WT_A_KPS, WT_A_KPE,
        WT_A_IDS, WT_A_IDE, WT_A_JDS, WT_A_JDE, WT_A_NUM_TILES,
        WT_A_MP_PHYSICS, WT_A_SPECIFIED, WT_A_SPEC_ZONE, WT_A_CHANNEL_SWITCH
    };
    int ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe;
    int ids, ide, jds, jde, nt, ti, tj, tk, ni, nj, nk, sz;
    const int *is, *ie, *js, *je;

    if (rows <= 0 || !jrange) return 0;
    for (unsigned a = 0; a < sizeof scalar / sizeof *scalar; a++)
        if (!wt_argv[scalar[a]]) return 0;
    if (!wt_argv[WT_A_I_START] || !wt_argv[WT_A_I_END] ||
        !wt_argv[WT_A_J_START] || !wt_argv[WT_A_J_END]) return 0;

    ims = IARG(WT_A_IMS); ime = IARG(WT_A_IME);
    jms = IARG(WT_A_JMS); jme = IARG(WT_A_JME);
    kms = IARG(WT_A_KMS); kme = IARG(WT_A_KME);
    ips = IARG(WT_A_IPS); ipe = IARG(WT_A_IPE);
    jps = IARG(WT_A_JPS); jpe = IARG(WT_A_JPE);
    kps = IARG(WT_A_KPS); kpe = IARG(WT_A_KPE);
    ids = IARG(WT_A_IDS); ide = IARG(WT_A_IDE);
    jds = IARG(WT_A_JDS); jde = IARG(WT_A_JDE);
    nt = IARG(WT_A_NUM_TILES);
    if (!wt_micro_extent(ims, ime, &ni) || !wt_micro_extent(jms, jme, &nj) ||
        !wt_micro_extent(kms, kme, &nk) ||
        !wt_micro_extent(ips, ipe, &ti) || !wt_micro_extent(jps, jpe, &tj) ||
        !wt_micro_extent(kps, kpe, &tk) || nt <= 0 || nt > 64)
        return 0;
    /* no sign test on the bounds themselves: WRF's halo puts ims/jms at -4 on edge ranks */
    if (ni < ti || nj < tj || nk < tk ||
        ips < ims || ipe > ime || jps < jms || jpe > jme ||
        kps < kms || kpe > kme || rows > tj)
        return 0;
    sz = IARG(WT_A_SPECIFIED) ? IARG(WT_A_SPEC_ZONE) : 0;
    if (IARG(WT_A_CHANNEL_SWITCH)) sz = 0;
    if (sz < 0) return 0;

    is = (const int *)wt_argv[WT_A_I_START]; ie = (const int *)wt_argv[WT_A_I_END];
    js = (const int *)wt_argv[WT_A_J_START]; je = (const int *)wt_argv[WT_A_J_END];
    for (int t = 0; t < nt; t++)
        if (is[t] > ie[t] || js[t] > je[t] || is[t] < ips || ie[t] > ipe ||
            js[t] < jps || je[t] > jpe)
            return 0;
    (void)ids; (void)ide; (void)jds; (void)jde; (void)sz;
    *jrange = tj;
    return 1;
}

/* Self-check of the argument map at the positions that matter for staging.  The selected
 * descriptor owns both the package-specific whitelist and the flags that make optional
 * fields live.  The mp=8 descriptor deliberately retains the historical order and checks. */
static int stage_ok = -1;
static int wt_stage_selfcheck(int NI, int NK, int NJ, int verbose)
{
    int ok = 1, mp = wt_argv[WT_A_MP_PHYSICS] ? IARG(WT_A_MP_PHYSICS) : 0;
    size_t full3 = wt_micro_full_3d(NI, NK, NJ);
    stage_layout = wt_micro_find_layout(mp);
    if (!stage_layout) ok = 0;
    if (stage_layout) {
        for (int k = 0; k < stage_layout->nrequired; k++) {
            int f = stage_layout->required[k];
            if (!wt_argv[f] || IARG(f) == 0) ok = 0;
        }
    }
    const char *q[6];
    for (int k = 0; k < 6; k++)
        q[k] = stage_layout && wt_argv[stage_layout->moist[k]]
             ? (const char *)wt_argv[stage_layout->moist[k]] : NULL;
    for (int k = 1; k < 6; k++) {
        if (!q[k] || q[k] == q[0]) ok = 0;
    }
    if (!full3) ok = 0;
    for (int k = 0; k < 6; k++) {
        size_t got = wt_alloc_size(q[k]);
        if (got && got < full3) ok = 0;
    }
    if (stage_layout && stage_layout->scalar_a >= 0 &&
        (!wt_argv[stage_layout->scalar_a] || !wt_argv[stage_layout->scalar_b] ||
         wt_argv[stage_layout->scalar_a] == wt_argv[stage_layout->scalar_b])) ok = 0;
    if (verbose) {
        fprintf(stderr, "[wtimer] selfcheck: mp_physics=%d dt=%.3f dx=%.1f flags=", mp,
                wt_argv[WT_A_DT] ? *(float *)wt_argv[WT_A_DT] : 0.0,
                wt_argv[WT_A_DX] ? *(float *)wt_argv[WT_A_DX] : 0.0);
        if (stage_layout) for (int k = 0; k < stage_layout->nrequired; k++) {
            int f = stage_layout->required[k];
            fprintf(stderr, "%d,", wt_argv[f] ? IARG(f) : -99);
        }
        fprintf(stderr, " moist_alloc=");
        for (int k = 0; k < 6; k++) fprintf(stderr, "%zu,", wt_alloc_size(q[k]));
        fprintf(stderr, " do_radar_ref=%d has_reqc/i/s=%d/%d/%d specified=%d spec_zone=%d => %s\n",
                wt_argv[WT_A_DO_RADAR_REF] ? IARG(WT_A_DO_RADAR_REF) : -99,
                wt_argv[WT_A_HAS_REQC] ? IARG(WT_A_HAS_REQC) : -99, wt_argv[WT_A_HAS_REQI] ? IARG(WT_A_HAS_REQI) : -99,
                wt_argv[WT_A_HAS_REQS] ? IARG(WT_A_HAS_REQS) : -99,
                wt_argv[WT_A_SPECIFIED] ? IARG(WT_A_SPECIFIED) : -99,
                wt_argv[WT_A_SPEC_ZONE] ? IARG(WT_A_SPEC_ZONE) : -99,
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

    int mp = IARG(WT_A_MP_PHYSICS);
    int grid = wt_argv[WT_A_ID] ? IARG(WT_A_ID) : 0;
    static int stage_checked_mp = -1, stage_checked_grid = -1;
    if (stage_checked_mp != mp || stage_checked_grid != grid) {   /* once per (mp, grid) */
        stage_ok = wt_stage_selfcheck(NI, NK, NJ, me);
        stage_checked_mp = mp; stage_checked_grid = grid;
    }
    wt_micro_select_fields(NI, NK, NJ);
    int *w3 = stage_w3, *w2 = stage_w2, n3 = stage_n3, n2 = stage_n2;
    int refuse = !stage_ok || !stage_alloc_ok;
    for (int a = 0; a < n3 && !refuse; a++) {
        if (!wt_argv[w3[a]]) refuse = 1;
        for (int b = 0; b < a; b++) if (wt_argv[w3[b]] == wt_argv[w3[a]]) refuse = 1;
    }
    for (int a = 0; a < n2 && !refuse; a++) if (!wt_argv[w2[a]]) refuse = 1;
    /* the driver only touches columns inside the specified zone: stage exactly those */
    int sz = IARG(WT_A_SPECIFIED) ? IARG(WT_A_SPEC_ZONE) : 0;
    if (wt_argv[WT_A_CHANNEL_SWITCH] && IARG(WT_A_CHANNEL_SWITCH)) sz = 0;
    int64_t jsplitx = (int64_t)jpe - rows + 1;       /* staged rows are jsplit..jpe */
    int64_t i0x = (int64_t)ips > (int64_t)ids + sz ? ips : (int64_t)ids + sz;
    int64_t i1x = (int64_t)ipe < (int64_t)ide - 1 - sz ? ipe : (int64_t)ide - 1 - sz;
    int64_t j0x = jsplitx > (int64_t)jds + sz ? jsplitx : (int64_t)jds + sz;
    int64_t j1x = (int64_t)jpe < (int64_t)jde - 1 - sz ? jpe : (int64_t)jde - 1 - sz;
    int jsplit = 0, i0 = 0, i1 = 0, j0 = 0, j1 = 0;
    if (jsplitx < INT_MIN || jsplitx > INT_MAX || i0x < INT_MIN || i0x > INT_MAX ||
        i1x < INT_MIN || i1x > INT_MAX || j0x < INT_MIN || j0x > INT_MAX ||
        j1x < INT_MIN || j1x > INT_MAX)
        refuse = 1;
    else {
        jsplit = (int)jsplitx; i0 = (int)i0x; i1 = (int)i1x;
        j0 = (int)j0x; j1 = (int)j1x;
    }
    int64_t ncol64 = (!refuse && i1 >= i0 && j1 >= j0) ?
                     (int64_t)(i1 - i0 + 1) * (int64_t)(j1 - j0 + 1) : 0;
    int NCOL = ncol64 > 0 && ncol64 <= INT_MAX ? (int)ncol64 : 0;
    if (first) {
        fprintf(stderr, "[wtimer] stage geometry: ims..ime=%d..%d jms..jme=%d..%d kms..kme=%d..%d"
                " ips..ipe=%d..%d jps..jpe=%d..%d sz=%d staged i=%d..%d j=%d..%d NCOL=%d n3=%d tiles=%d refuse=%d\n",
                ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, sz, i0, i1, j0, j1, NCOL, n3, IARG(WT_A_NUM_TILES), refuse);
        for (int a = 0; a < n3; a++) fprintf(stderr, "[wtimer] 3d %s=%p\n", wt_argname[w3[a]], wt_argv[w3[a]]);
        for (int a = 0; a < n2; a++) fprintf(stderr, "[wtimer] 2d %s=%p\n", wt_argname[w2[a]], wt_argv[w2[a]]);
        fflush(stderr);
    }
    wt_micro_stage_state = refuse || NCOL == 0 ? 0 : 1;
    if (refuse || NCOL == 0) {
        wt_micro_refused++;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        return;
    }
    size_t fields = (size_t)n3 * (size_t)NK + (size_t)n2;
    size_t need = 0;
    if (!fields || fields > SIZE_MAX / sizeof(float) || (size_t)NCOL > SIZE_MAX / fields ||
        (size_t)NCOL * fields > SIZE_MAX / sizeof(float)) {
        wt_micro_stage_state = 0;
        wt_micro_refused++;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        return;
    }
    need = (size_t)NCOL * fields * sizeof(float);
    if (wt_stage_share_enabled()) {
        if (wt_stage_adapt_enabled() &&
            wt_micro_adaptive_call(rows, mp, grid, ims, jms, kms, NI, NK, ips, ipe, jps, jpe,
                                   ids, ide, jds, jde, sz, n3, w3, n2, w2,
                                   first, dbg && me))
            return;
        if (wt_micro_shared_call(jsplit, ims, jms, kms, NI, NK, i0, i1, j0, j1, NCOL,
                                 n3, w3, n2, w2, need, mp, grid, first))
            return;
    }
    if (need > stage_cap) { free(stage_buf); stage_buf = malloc(need); stage_cap = need; }
    if (!stage_buf) {                                    /* fail closed: run it unstaged */
        wt_micro_stage_state = 0;
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
    float *base3[32], *base2[16];
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
    for (int a = 0; a < n2; a++) {
        float *src = (float *)wt_argv[w2[a]];
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
    for (int a = 0; a < n2; a++) wt_argv[w2[a]] = base2[a];
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
    for (int a = 0; a < n2; a++) {
        float *dst = (float *)save[w2[a]];
        for (int j = j0; j <= j1; j++)
            memcpy(dst + (size_t)(i0-ims) + (size_t)NI*(size_t)(j-jms), base2[a] + (size_t)NIC*(j-j0), (size_t)NIC * sizeof(float));
    }
    MARK("commit-done");
    memcpy(wt_argv, save, sizeof save);
    wt_micro_staged++;
}

/* Tile probe (research measurement, off unless WTIMER_TILE_PROBE=1): call the real driver once per
 * original tile and time each call, so this rank's per-tile cost map is observable. Every tile's
 * columns are still computed exactly once per step; only the OpenMP grouping changes (one tile per
 * call), the same tile-list rewrite that pass 1 already uses bit-identically. Staging is bypassed. */
static int tile_probe = -1;
static FILE *tile_fp;

static int wt_tile_probe_enabled(void)
{
    if (tile_probe < 0) {
        const char *e = getenv("WTIMER_TILE_PROBE");
        tile_probe = e && e[0] == '1' && e[1] == 0;
    }
    return tile_probe;
}

static void wt_micro_tile_probe(void)
{
    static int i0[1], i1[1], j0[1], j1[1], one = 1;
    int nt = IARG(WT_A_NUM_TILES); if (nt > 64) nt = 64;
    const int *oi0 = wt_argv[WT_A_I_START], *oi1 = wt_argv[WT_A_I_END];
    const int *oj0 = wt_argv[WT_A_J_START], *oj1 = wt_argv[WT_A_J_END];
    void *save[WT_NARG];
    memcpy(save, wt_argv, sizeof save);
    if (!tile_fp) {
        const char *dir = getenv("WTIMER_DIR"), *r = getenv("SLURM_PROCID");
        char path[4096];
        if (!r) r = getenv("PMI_RANK");
        snprintf(path, sizeof path, "%s/tiles.%s.csv", dir ? dir : ".", r ? r : "x");
        tile_fp = fopen(path, "w");
        if (tile_fp) fprintf(tile_fp, "step,tile,its,ite,jts,jte,ticks\n");
    }
    for (int t = 0; t < nt; t++) {
        i0[0] = oi0[t]; i1[0] = oi1[t]; j0[0] = oj0[t]; j1[0] = oj1[t];
        wt_argv[WT_A_I_START] = i0; wt_argv[WT_A_I_END] = i1;
        wt_argv[WT_A_J_START] = j0; wt_argv[WT_A_J_END] = j1;
        wt_argv[WT_A_NUM_TILES] = &one;
        uint64_t a = tsc();
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_, wt_argv, WT_NARG);
        uint64_t b = tsc();
        memcpy(wt_argv, save, sizeof save);
        if (tile_fp)
            fprintf(tile_fp, "%llu,%d,%d,%d,%d,%d,%llu\n", (unsigned long long)wt_micro_calls, t,
                    oi0[t], oi1[t], oj0[t], oj1[t], (unsigned long long)(b - a));
    }
    if (tile_fp) fflush(tile_fp);
}

void wt_micro_call(void)
{
    uint64_t t0 = tsc();
    wt_pre(0);
    wt_micro_calls++;
    if (wt_tile_probe_enabled()) {
        wt_micro_stage_state = 0;
        wt_micro_tile_probe();
    } else if (!wt_stage_micro_enabled()) {
        wt_micro_stage_state = 0;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                  wt_argv, WT_NARG);
    } else if (!wt_stage_grid_selected()) {
        wt_micro_stage_state = 0;
        wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                  wt_argv, WT_NARG);
    } else {
        int rows = stage_rows();
        int jrange;
        if (rows <= 0) {
            wt_micro_stage_state = 0;
            wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                      wt_argv, WT_NARG);
        } else if (!wt_micro_geometry_ok(rows, &jrange)) {
            wt_micro_stage_state = 0;
            wt_micro_refused++;
            wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                      wt_argv, WT_NARG);
        } else if (jrange > rows) {
            wt_micro_staged_call(rows);
        } else {
            wt_micro_stage_state = 0;
            wt_call_n((void *)__real_module_microphysics_driver_mp_microphysics_driver_,
                      wt_argv, WT_NARG);
        }
    }
    wt_record(0, tsc() - t0);
}
