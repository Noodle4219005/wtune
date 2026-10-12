/* R1 radiation staging: one rank, one dense tail rectangle, no sharing yet.
 *
 * The driver is entered through the collector in wtimer.S because its 422
 * by-reference arguments live partly in the caller's stack frame.  The only
 * arrays retargeted here are column-local fields whose declarations and
 * active RRTMG path have been audited against module_radiation_driver.f90.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "wt_argmap_rad.h"

void wt_call_n(void *fn, void **argv, int n);
void wt_pre(int idx);
void wt_record(int idx, uint64_t cycles);
extern void *wt_argv[];
extern void __real_module_radiation_driver_mp_radiation_driver_(void);

uint64_t wt_rad_calls, wt_rad_steps, wt_rad_staged, wt_rad_refused;
uint64_t wt_rad_share_out, wt_rad_share_in, wt_rad_share_wait_us;

static inline uint64_t rad_tsc(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#define RARG(k) (*(int *)wt_argv[k])
#define RREAL(k) (*(float *)wt_argv[k])

/* These are radiation-driver scalars present in the generated arg list but
 * intentionally not in the strict scalar subset emitted by gen_argmap.py. */
#define RAD_A_ICLOUD_CU       261
#define RAD_A_CALC_CLEAN      262
#define RAD_A_SF_SURFACE      329
#define RAD_A_AERCU_OPT       372
#define RAD_A_DIFFUSE_FRAC    311

static int rad_iarg(int k)
{
    return wt_argv[k] ? *(int *)wt_argv[k] : 0;
}

static float rad_farg(int k)
{
    return wt_argv[k] ? *(float *)wt_argv[k] : 0.0f;
}

static int rad_flag(int k)
{
    return wt_argv[k] && *(int *)wt_argv[k] != 0;
}

static int rad_rank(void)
{
    const char *s = getenv("SLURM_PROCID");
    if (!s) s = getenv("PMI_RANK");
    return s ? atoi(s) : -1;
}

static int rad_debug_rank(void)
{
    const char *s = getenv("WTIMER_DEBUG_RANK");
    return s && atoi(s) == rad_rank();
}

static void rad_real_call(void)
{
    wt_call_n((void *)__real_module_radiation_driver_mp_radiation_driver_,
              wt_argv, WTR_NARG);
}

/* These groups are the fields that the RRTMG path actually traverses.  A
 * present optional field is included; an absent optional remains NULL and is
 * never dereferenced by a pack or by the staged driver call. */
static const int rad_base3[] = {
    WTR_A_DZ8W, WTR_A_O3RAD, WTR_A_P8W, WTR_A_P, WTR_A_PI, WTR_A_RHO,
    WTR_A_RTHRATEN, WTR_A_RTHRATENLW, WTR_A_RTHRATENSW,
    WTR_A_RTHRATENLWC, WTR_A_RTHRATENSWC, WTR_A_T8W, WTR_A_T,
    WTR_A_CLDFRA, WTR_A_RE_CLOUD, WTR_A_RE_ICE, WTR_A_RE_SNOW,
    /* f_ice_phy/f_rain_phy are NOT here: Registry packages fer_mp_hires
     * (mp=5) and etampnew (mp=95) only (in_use_for_config_f.inc:7-36), so
     * under Thompson they are size-1 dummies; cal_cldfra1 reads them only
     * when F_QC.and..not.F_QI.and.F_QS (.f90:3545) and RRTMG never for mp=8. */
};

static const int rad_base2[] = {
    WTR_A_ALBEDO, WTR_A_EMISS, WTR_A_GLW, WTR_A_GSW, WTR_A_RLWTOA,
    WTR_A_SNOW, WTR_A_TSK, WTR_A_XICE, WTR_A_XLAND, WTR_A_XLAT,
    WTR_A_XLONG, WTR_A_COSZEN, WTR_A_HRANG, WTR_A_SWDOWN,
    WTR_A_SWDOWNC, WTR_A_SWDDIR, WTR_A_SWDDNI, WTR_A_SWDDIF,
    WTR_A_SWDDIRC, WTR_A_SWDDNIC,
    WTR_A_LWUPT, WTR_A_LWUPTC, WTR_A_LWUPTCLN, WTR_A_LWDNT,
    WTR_A_LWDNTC, WTR_A_LWDNTCLN, WTR_A_LWUPB, WTR_A_LWUPBC,
    WTR_A_LWUPBCLN, WTR_A_LWDNB, WTR_A_LWDNBC, WTR_A_LWDNBCLN,
    WTR_A_SWUPT, WTR_A_SWUPTC, WTR_A_SWUPTCLN, WTR_A_SWDNT,
    WTR_A_SWDNTC, WTR_A_SWDNTCLN, WTR_A_SWUPB, WTR_A_SWUPBC,
    WTR_A_SWUPBCLN, WTR_A_SWDNB, WTR_A_SWDNBC, WTR_A_SWDNBCLN,
    WTR_A_LWCF, WTR_A_SWCF, WTR_A_OLR,
    WTR_A_ALSWVISDIR, WTR_A_ALSWVISDIF, WTR_A_ALSWNIRDIR,
    WTR_A_ALSWNIRDIF, WTR_A_SWVISDIR, WTR_A_SWVISDIF,
    WTR_A_SWNIRDIR, WTR_A_SWNIRDIF, WTR_A_CLDT,
    WTR_A_ACSWUPT, WTR_A_ACSWUPTC, WTR_A_ACSWDNT, WTR_A_ACSWDNTC,
    WTR_A_ACSWUPB, WTR_A_ACSWUPBC, WTR_A_ACSWDNB, WTR_A_ACSWDNBC,
    WTR_A_ACLWUPT, WTR_A_ACLWUPTC, WTR_A_ACLWDNT, WTR_A_ACLWDNTC,
    WTR_A_ACLWUPB, WTR_A_ACLWUPBC, WTR_A_ACLWDNB, WTR_A_ACLWDNBC,
};

static const int rad_s3x[] = {
    WTR_A_SWUPFLX, WTR_A_SWUPFLXC, WTR_A_SWDNFLX, WTR_A_SWDNFLXC,
    WTR_A_LWUPFLX, WTR_A_LWUPFLXC, WTR_A_LWDNFLX, WTR_A_LWDNFLXC,
};

struct rad_layout {
    int w3[WTR_W3D];
    int n3;
    int w2[WTR_W2D + 16];
    int n2;
    int s3x;
};

struct rad_probe_info {
    struct rad_layout layout;
    int levsiz, n_ozmixm;
    size_t cal3, cal2, cal1;
};

static struct rad_probe_info rad_probe;

static void rad_add(int *v, int *n, int cap, int k)
{
    int i;
    for (i = 0; i < *n; i++) if (v[i] == k) return;
    if (*n < cap) v[(*n)++] = k;
}

static int rad_s3x_state(void)
{
    int i, n = 0;
    for (i = 0; i < (int)(sizeof rad_s3x / sizeof *rad_s3x); i++)
        if (wt_argv[rad_s3x[i]]) n++;
    if (n == 0) return 0; /* the current Intel call site omits all eight */
    if (n != (int)(sizeof rad_s3x / sizeof *rad_s3x)) return -1;
    for (i = 1; i < (int)(sizeof rad_s3x / sizeof *rad_s3x); i++)
        if (wt_argv[rad_s3x[i]] == wt_argv[rad_s3x[0]]) return -1;
    return 1;
}

static int rad_is_cln(int k)
{
    return k == WTR_A_LWUPTCLN || k == WTR_A_LWDNTCLN ||
           k == WTR_A_LWUPBCLN || k == WTR_A_LWDNBCLN ||
           k == WTR_A_SWUPTCLN || k == WTR_A_SWDNTCLN ||
           k == WTR_A_SWUPBCLN || k == WTR_A_SWDNBCLN;
}

static void rad_build_layout(struct rad_layout *l)
{
    int i;
    memset(l, 0, sizeof *l);
    for (i = 0; i < (int)(sizeof rad_base3 / sizeof *rad_base3); i++)
        rad_add(l->w3, &l->n3, WTR_W3D, rad_base3[i]);

    /* Registry package membership, not PRESENT(), decides allocation.  The
     * moisture flags select only the active thompson slabs in this case. */
    if (rad_flag(WTR_A_F_QV)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QV);
    if (rad_flag(WTR_A_F_QC)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QC);
    if (rad_flag(WTR_A_F_QR)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QR);
    if (rad_flag(WTR_A_F_QI)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QI);
    if (rad_flag(WTR_A_F_QS)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QS);
    if (rad_flag(WTR_A_F_QG)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QG);
    if (rad_flag(WTR_A_F_QH)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QH);
    if (rad_flag(WTR_A_F_QI2)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QI2);
    if (rad_flag(WTR_A_F_QI3)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QI3);
    if (rad_flag(WTR_A_F_QNDROP)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QNDROP);
    if (rad_flag(WTR_A_F_QNIFA)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QNIFA);
    if (rad_flag(WTR_A_F_QNWFA)) rad_add(l->w3, &l->n3, WTR_W3D, WTR_A_QNWFA);

    /* cldfra_bl/qc_bl/qi_bl, CAMMGMP, BMJ/KF, CAM radius and Deng fields
     * are package-inactive here; their ABI dummies are non-NULL size-one
     * allocations.  The self-check rejects a changed runtime gate. */
    for (i = 0; i < (int)(sizeof rad_base2 / sizeof *rad_base2); i++) {
        int k = rad_base2[i];
        /* Exact membership, never an index range: the argmap interleaves
         * the real fluxes with their *CLN twins (LWUPT 233, LWUPTC 234,
         * LWUPTCLN 235, LWDNT 236, ...).  A range 235..244 silently dropped
         * LWDNT/LWDNTC/LWUPB/LWUPBC/LWDNB/LWDNBC (+6 SW), which pass2 then
         * handed to the driver unstaged -> G0 job 1070403 failed bit-identity
         * with RMS(R) = RMS(A)*sqrt(1-rows/NJ) on exactly those fields. */
        if (rad_is_cln(k) && rad_iarg(RAD_A_CALC_CLEAN) == 0) continue;
        if (k >= WTR_A_ALSWVISDIR && k <= WTR_A_ALSWNIRDIF &&
            rad_iarg(RAD_A_SF_SURFACE) != 8) continue;
        rad_add(l->w2, &l->n2, WTR_W2D + 16, k);
    }
    /* This state field is updated after the RRTMG call whenever present. */
    rad_add(l->w2, &l->n2, WTR_W2D + 16, RAD_A_DIFFUSE_FRAC);
    l->s3x = rad_s3x_state() == 1;
}

static int rad_ptr_distinct(const struct rad_layout *l)
{
    int i, j;
    for (i = 0; i < l->n3; i++) {
        if (!wt_argv[l->w3[i]]) return 0;
        for (j = 0; j < i; j++)
            if (wt_argv[l->w3[i]] == wt_argv[l->w3[j]]) return 0;
    }
    return 1;
}

static int rad_moist_spacing(int NI, int NK, int NJ)
{
    const int q[] = {WTR_A_QV, WTR_A_QC, WTR_A_QR, WTR_A_QI,
                     WTR_A_QS, WTR_A_QG};
    size_t block = (size_t)NI * (size_t)NK * (size_t)NJ * sizeof(float);
    int i;
    if (!wt_argv[q[0]] || !wt_argv[q[1]]) return 0;
    if ((char *)wt_argv[q[1]] - (char *)wt_argv[q[0]] != (ptrdiff_t)block)
        return 0;
    for (i = 2; i < (int)(sizeof q / sizeof *q); i++) {
        if (!rad_flag((int[]){WTR_A_F_QR, WTR_A_F_QI, WTR_A_F_QS, WTR_A_F_QG}[i-2]))
            continue;
        if (!wt_argv[q[i]] ||
            (char *)wt_argv[q[i]] - (char *)wt_argv[q[0]] != (ptrdiff_t)i * (ptrdiff_t)block)
            return 0;
    }
    return 1;
}

static int rad_is_4d(int k)
{
    return k == WTR_A_QV || k == WTR_A_QC || k == WTR_A_QR ||
           k == WTR_A_QI || k == WTR_A_QI2 || k == WTR_A_QI3 ||
           k == WTR_A_QS || k == WTR_A_QG || k == WTR_A_QH ||
           k == WTR_A_QNDROP || k == WTR_A_QNIFA || k == WTR_A_QNWFA;
}

/* Byte size recorded by the for_allocate_handle wrapper (wt_alloc.c) for the
 * ALLOCATE whose data starts at p; 0 = not a recorded ALLOCATE base. */
extern size_t wt_alloc_size(const void *p);
extern unsigned long wt_alloc_calls, wt_alloc_dropped;

/* At this call site (module_first_rk_step_part1.f90:267) DZ8W=dz8w, PI=pi_phy
 * and T8W=t8w are Registry i1 scratch: automatic arrays of solve_em
 * (inc/i1_decl.inc), stack storage since configure.wrf has no -heap-arrays.
 * They are full-size by construction and are not ALLOCATE bases, so the size
 * table cannot answer for them. Everything else staged is grid% state
 * (ALLOCATE via for_allocate_handle) or a moist/scalar slab (rad_is_4d, skipped). */
static int rad_is_stack_scratch(int k)
{
    return k == WTR_A_DZ8W || k == WTR_A_PI || k == WTR_A_T8W;
}

static int rad_usable_at_least(int k, size_t need, char *reason, size_t nreason)
{
    size_t usable;
    if (!wt_argv[k]) {
        snprintf(reason, nreason, "staged array %s is NULL", wtr_argname[k]);
        return 0;
    }
    if (rad_is_stack_scratch(k)) return 1;
    usable = wt_alloc_size(wt_argv[k]);
    if (usable == 0) {
        snprintf(reason, nreason, "no ALLOCATE record for %s", wtr_argname[k]);
        return 0;
    }
    if (usable < need) {
        snprintf(reason, nreason, "ALLOCATE of %s is %zu B, staging needs %zu B (size-1 dummy?)",
                 wtr_argname[k], usable, need);
        return 0;
    }
    return 1;
}

static int rad_usable_calibration(int k, size_t need, size_t *value,
                                  char *reason, size_t nreason)
{
    if (!wt_argv[k]) {
        *value = 0;
        snprintf(reason, nreason, "usable-size probe unreliable");
        return 0;
    }
    *value = wt_alloc_size(wt_argv[k]);
    if (*value != need) {   /* allocs_*.f90 allocate exactly nba = product * 4 bytes */
        snprintf(reason, nreason, "ALLOCATE table calibration failed on %s: %zu B recorded, %zu B expected",
                 wtr_argname[k], *value, need);
        return 0;
    }
    return 1;
}

static int rad_selfcheck(char *reason, size_t nreason)
{
    struct rad_layout l;
    int NI, NK, NJ, nt, i, j;
    size_t req3, req2;
    const int geom[] = {WTR_A_IDS, WTR_A_IDE, WTR_A_JDS, WTR_A_JDE,
                        WTR_A_KDS, WTR_A_KDE, WTR_A_IMS, WTR_A_IME,
                        WTR_A_JMS, WTR_A_JME, WTR_A_KMS, WTR_A_KME,
                        WTR_A_I_START, WTR_A_I_END, WTR_A_J_START,
                        WTR_A_J_END, WTR_A_KTS, WTR_A_KTE, WTR_A_NUM_TILES};
    const int *is, *ie, *js, *je;
    int s3;

    memset(&rad_probe, 0, sizeof rad_probe);
    rad_probe.levsiz = rad_iarg(WTR_A_LEVSIZ);
    rad_probe.n_ozmixm = rad_iarg(WTR_A_N_OZMIXM);
    rad_build_layout(&l);
    rad_probe.layout = l;

    /* Layout census: every base2 entry not excluded by a runtime gate must be
     * staged (+1 for diffuse_frac).  49 instead of 61 is how job 1070403's
     * range bug would have been caught before the driver ran. */
    {
        int want = (int)(sizeof rad_base2 / sizeof *rad_base2) + 1
                 - (rad_iarg(RAD_A_CALC_CLEAN) == 0 ? 8 : 0)
                 - (rad_iarg(RAD_A_SF_SURFACE) != 8 ? 4 : 0);
        if (l.n2 != want) {
            snprintf(reason, nreason, "2D layout census %d != %d expected", l.n2, want);
            return 0;
        }
    }

    if (rad_iarg(WTR_A_LW_PHYSICS) != 4 || rad_iarg(WTR_A_SW_PHYSICS) != 4) {
        snprintf(reason, nreason, "lw/sw physics is not RRTMG (4/4)"); return 0;
    }
    if (rad_iarg(WTR_A_ICLOUD) != 1) {
        snprintf(reason, nreason, "icloud is not 1"); return 0;
    }
    if (rad_iarg(WTR_A_O3INPUT) != 2) {
        snprintf(reason, nreason, "o3input is not 2"); return 0;
    }
    if (rad_iarg(WTR_A_AER_OPT) != 0) {
        snprintf(reason, nreason, "aer_opt is not 0 (aerosol arrays are not staged)"); return 0;
    }
    if (rad_iarg(WTR_A_MP_PHYSICS) != 8 || rad_iarg(WTR_A_CU_PHYSICS) != 0) {
        snprintf(reason, nreason, "this whitelist requires mp_physics=8 and cu_physics=0"); return 0;
    }
    if (rad_iarg(RAD_A_ICLOUD_CU) != 0 || rad_iarg(RAD_A_AERCU_OPT) != 0) {
        snprintf(reason, nreason, "icloud_cu/aercu_opt path needs unstaged CU fields"); return 0;
    }
    if (rad_iarg(WTR_A_ICLOUD_BL) != 0) {
        snprintf(reason, nreason, "icloud_bl needs MYNN package fields"); return 0;
    }
    if (rad_flag(WTR_A_IS_CAMMGMP_USED)) {
        snprintf(reason, nreason, "CAMMGMP package fields are inactive"); return 0;
    }
    if (rad_iarg(WTR_A_SHCU_PHYSICS) == 5) {
        snprintf(reason, nreason, "Deng shallow-cloud fields are inactive"); return 0;
    }
    if (rad_flag(WTR_A_F_QH) || rad_flag(WTR_A_F_QI2) ||
        rad_flag(WTR_A_F_QI3) || rad_flag(WTR_A_F_QNDROP) ||
        rad_flag(WTR_A_F_QNIFA) || rad_flag(WTR_A_F_QNWFA)) {
        snprintf(reason, nreason, "unsupported inactive moisture package flag"); return 0;
    }
    if (rad_iarg(WTR_A_SWINT_OPT) != 0) {
        snprintf(reason, nreason, "swint_opt needs an unstaged diagnostic path"); return 0;
    }
    if (rad_iarg(WTR_A_SOLAR_OPT) != 0) {
        snprintf(reason, nreason, "solar_opt needs unstaged moisture diagnostics"); return 0;
    }
    if (rad_iarg(WTR_A_SLOPE_RAD) != 0 || rad_iarg(WTR_A_TOPO_SHADING) != 0) {
        snprintf(reason, nreason, "slope_rad/topo_shading is enabled"); return 0;
    }
    if (rad_flag(WTR_A_ADAPT_STEP_FLAG)) {
        snprintf(reason, nreason, "adaptive radiation timing is enabled"); return 0;
    }
    if (wt_argv[WTR_A_SW_ECLIPSE] && rad_iarg(WTR_A_SW_ECLIPSE) != 0) {
        snprintf(reason, nreason, "solar eclipse path is enabled"); return 0;
    }
    if (rad_iarg(WTR_A_LEVSIZ) <= 0 || rad_iarg(WTR_A_N_OZMIXM) <= 0 ||
        !wt_argv[WTR_A_OZMIXM]) {
        snprintf(reason, nreason, "o3input=2 has no valid ozmixm/levsiz/n_ozmixm"); return 0;
    }
    s3 = rad_s3x_state();
    if (s3 < 0) {
        snprintf(reason, nreason, "S3X optionals are partial/aliased"); return 0;
    }
    rad_probe.layout.s3x = s3 == 1;

    NI = rad_iarg(WTR_A_IME) - rad_iarg(WTR_A_IMS) + 1;
    NK = rad_iarg(WTR_A_KME) - rad_iarg(WTR_A_KMS) + 1;
    NJ = rad_iarg(WTR_A_JME) - rad_iarg(WTR_A_JMS) + 1;
    nt = rad_iarg(WTR_A_NUM_TILES);
    if (NI <= 0 || NK <= 0 || NJ <= 0 || nt <= 0 || nt > 4096) {
        snprintf(reason, nreason, "invalid geometry or tile count"); return 0;
    }
    if (rad_iarg(WTR_A_KME) < rad_iarg(WTR_A_KTE) + 1) {
        snprintf(reason, nreason, "kme < kte+1"); return 0;
    }
    for (i = 0; i < (int)(sizeof geom / sizeof *geom); i++) {
        if (!wt_argv[geom[i]]) {
            snprintf(reason, nreason, "geometry argument %d is NULL", geom[i]); return 0;
        }
    }
    is = (const int *)wt_argv[WTR_A_I_START]; ie = (const int *)wt_argv[WTR_A_I_END];
    js = (const int *)wt_argv[WTR_A_J_START]; je = (const int *)wt_argv[WTR_A_J_END];
    if (!is || !ie || !js || !je) {
        snprintf(reason, nreason, "tile arrays are absent"); return 0;
    }
    for (i = 0; i < nt; i++) {
        if (is[i] > ie[i] || js[i] > je[i]) {
            snprintf(reason, nreason, "invalid tile bounds"); return 0;
        }
    }

    for (i = 0; i < (int)(sizeof rad_base3 / sizeof *rad_base3); i++) {
        if (!wt_argv[rad_base3[i]]) {
            snprintf(reason, nreason, "mandatory 3-D argument %d is NULL", rad_base3[i]); return 0;
        }
    }
    if ((size_t)NI > SIZE_MAX / (size_t)NK ||
        (size_t)NI * (size_t)NK > SIZE_MAX / (size_t)NJ ||
        (size_t)NI > SIZE_MAX / (size_t)NJ) {
        snprintf(reason, nreason, "geometry size overflows usable-size probe"); return 0;
    }
    if (!rad_moist_spacing(NI, NK, NJ)) {
        snprintf(reason, nreason, "moisture species are not adjacent 3-D slabs"); return 0;
    }
    /* All fields selected above are package-active or unconditionally full
     * state.  Check their pointers explicitly; PRESENT/non-NULL is not an
     * allocation proof for fields in inactive Registry packages. */
    for (i = 0; i < l.n3; i++) if (!wt_argv[l.w3[i]]) {
        snprintf(reason, nreason, "active 3-D argument %d is NULL", l.w3[i]); return 0;
    }
    if (!rad_ptr_distinct(&l)) {
        snprintf(reason, nreason, "whitelisted 3-D arguments are aliased"); return 0;
    }
    for (i = 0; i < l.n2; i++) if (!wt_argv[l.w2[i]]) {
        snprintf(reason, nreason, "active 2-D argument %d is NULL", l.w2[i]); return 0;
    }
    if (s3 == 1) {
        for (i = 0; i < 8; i++) for (j = 0; j < i; j++)
            if (wt_argv[rad_s3x[i]] == wt_argv[rad_s3x[j]]) {
                snprintf(reason, nreason, "S3X arguments are aliased"); return 0;
            }
    }

    if ((size_t)NI * (size_t)NK * (size_t)NJ > SIZE_MAX / sizeof(float) ||
        (size_t)NI * (size_t)NJ > SIZE_MAX / sizeof(float)) {
        snprintf(reason, nreason, "geometry size overflows usable-size probe"); return 0;
    }
    req3 = (size_t)NI * (size_t)NK * (size_t)NJ * sizeof(float);
    req2 = (size_t)NI * (size_t)NJ * sizeof(float);
    if (!rad_usable_calibration(WTR_A_T, req3, &rad_probe.cal3, reason, nreason) ||
        !rad_usable_calibration(WTR_A_XLAT, req2, &rad_probe.cal2, reason, nreason))
        return 0;
    if (!wt_argv[WTR_A_LRADIUS]) {
        snprintf(reason, nreason, "usable-size probe unreliable");
        return 0;
    }
    rad_probe.cal1 = wt_alloc_size(wt_argv[WTR_A_LRADIUS]);   /* cammgmp only: must be the 4-byte dummy */
    if (rad_probe.cal1 != 4) {
        snprintf(reason, nreason, "ALLOCATE table negative calibration failed on LRADIUS: %zu B recorded, 4 expected",
                 rad_probe.cal1);
        return 0;
    }
    if (wt_alloc_dropped) {
        snprintf(reason, nreason, "ALLOCATE table overflowed (%lu dropped)", wt_alloc_dropped);
        return 0;
    }
    for (i = 0; i < l.n3; i++) {
        if (rad_is_4d(l.w3[i])) continue;
        if (!rad_usable_at_least(l.w3[i], req3, reason, nreason)) return 0;
    }
    for (i = 0; i < l.n2; i++)
        if (!rad_usable_at_least(l.w2[i], req2, reason, nreason)) return 0;
    if (s3 == 1) {
        size_t reqs3x;
        if (NK > INT_MAX - 2 || (size_t)NI > SIZE_MAX / (size_t)(NK + 2) ||
            (size_t)NI * (size_t)(NK + 2) > SIZE_MAX / (size_t)NJ) {
            snprintf(reason, nreason, "S3X geometry size overflows usable-size probe"); return 0;
        }
        reqs3x = (size_t)NI * (size_t)(NK + 2) * (size_t)NJ * sizeof(float);
        for (i = 0; i < 8; i++)
            if (!rad_usable_at_least(rad_s3x[i], reqs3x, reason, nreason)) return 0;
    }
    {
        size_t reqs4;
        int levsiz = rad_probe.levsiz, n_ozmixm = rad_probe.n_ozmixm;
        if ((size_t)NI > SIZE_MAX / (size_t)levsiz ||
            (size_t)NI * (size_t)levsiz > SIZE_MAX / (size_t)NJ ||
            (size_t)NI * (size_t)levsiz * (size_t)NJ > SIZE_MAX / (size_t)n_ozmixm ||
            (size_t)NI * (size_t)levsiz * (size_t)NJ * (size_t)n_ozmixm >
                SIZE_MAX / sizeof(float)) {
            snprintf(reason, nreason, "ozmixm size overflows usable-size probe"); return 0;
        }
        reqs4 = (size_t)NI * (size_t)levsiz * (size_t)NJ *
                (size_t)n_ozmixm * sizeof(float);
        if (!rad_usable_at_least(WTR_A_OZMIXM, reqs4, reason, nreason)) return 0;
    }
    snprintf(reason, nreason, "ok");
    return 1;
}

static int rad_due(void)
{
    int it = rad_iarg(WTR_A_ITIMESTEP);
    int stepra = rad_iarg(WTR_A_STEPRA);
    int offset = rad_iarg(WTR_A_RA_CALL_OFFSET);
    int adaptive = rad_flag(WTR_A_ADAPT_STEP_FLAG);

    if (it == 1 || rad_farg(WTR_A_RADT) == 0.0f || stepra == 1) return 1;
    if (!adaptive && stepra > 0 && it % stepra == 1 + offset) return 1;
    if (adaptive && wt_argv[WTR_A_CURR_SECS] && wt_argv[WTR_A_RADTACTTIME] &&
        rad_farg(WTR_A_CURR_SECS) >= rad_farg(WTR_A_RADTACTTIME)) return 1;
    return 0;
}

static void rad_print_selfcheck(int rank, int ok, const char *reason)
{
    int i;
    const struct rad_layout *l = &rad_probe.layout;
    fprintf(stderr, "[wtimer] rad selfcheck rank=%d NARG=%d n3=%d n2=%d s3x=%d "
            "levsiz=%d n_ozmixm=%d cal3=%zu cal2=%zu cal1=%zu allocs=%lu w3=",
            rank, WTR_NARG, l->n3, l->n2, l->s3x, rad_probe.levsiz,
            rad_probe.n_ozmixm, rad_probe.cal3, rad_probe.cal2, rad_probe.cal1,
            wt_alloc_calls);
    for (i = 0; i < l->n3; i++) fprintf(stderr, "%s%d", i ? "," : "", l->w3[i]);
    fputs(" w2=", stderr);
    for (i = 0; i < l->n2; i++) fprintf(stderr, "%s%d", i ? "," : "", l->w2[i]);
    fprintf(stderr, " s4_idx=%d s3x_idx=", WTR_A_OZMIXM);
    if (l->s3x) for (i = 0; i < 8; i++) fprintf(stderr, "%s%d", i ? "," : "", rad_s3x[i]);
    else fputs("-", stderr);
    if (ok) fputs(" STAGING ON\n", stderr);
    else fprintf(stderr, " STAGING OFF: %s\n", reason);
}

static int rad_patch_bounds(int *i0, int *i1, int *j0, int *j1)
{
    const int *is = (const int *)wt_argv[WTR_A_I_START];
    const int *ie = (const int *)wt_argv[WTR_A_I_END];
    const int *js = (const int *)wt_argv[WTR_A_J_START];
    const int *je = (const int *)wt_argv[WTR_A_J_END];
    int n = rad_iarg(WTR_A_NUM_TILES), t;
    if (!is || !ie || !js || !je || n <= 0) return 0;
    *i0 = is[0]; *i1 = ie[0]; *j0 = js[0]; *j1 = je[0];
    for (t = 1; t < n; t++) {
        if (is[t] < *i0) *i0 = is[t];
        if (ie[t] > *i1) *i1 = ie[t];
        if (js[t] < *j0) *j0 = js[t];
        if (je[t] > *j1) *j1 = je[t];
    }
    return *i0 <= *i1 && *j0 <= *j1;
}


/* ---- Pass-2 geometry (G0 #4) ---------------------------------------------
 * Job 1070413 refuted the dense-rectangle pass 2 (NIC*rows columns x 1 row cut
 * into nt i-tiles): COSZEN, a pure per-point function of (xlat, xlon, time),
 * differed at 220 points in the last bit.  A per-point result that depends on
 * where the point sits inside the i loop means the driver's unrolled /
 * vectorised loop bodies and their remainder tails round differently
 * (calc_coszen: 4 sincosf vs 33 cosf / 16 sinf call sites; -fp-model precise
 * still lets each code version contract FMAs its own way).  Pass 2 therefore
 * keeps the ORIGINAL i geometry: ims:ime full width, the original its:ite of
 * each tile, i-splits only at its+m*isplit, and every buffered field's staged
 * block at the same address mod 64 as the original block so alignment peel
 * loops take the same trips.
 *
 * Variants (env; RN/RS need no buffer and only isolate the tile change):
 *   WTIMER_RAD_BUF    0: pass 2 runs on the original arrays        (RN/RS)
 *                     1: staged rows copied to a private buffer     (default)
 *   WTIMER_RAD_ISPLIT -1: original tiles clipped to the staged rows (RN)
 *                      0: one tile per staged row, no i split
 *                      m: one tile per staged row per its+m*k chunk (default 32)
 *   WTIMER_RAD_FULLJ  1: buffer keeps the full jms:jme shape so the driver's
 *                        automatic arrays get the same shape too   (RF)
 *                     0: buffer j extent = staged rows only         (RH, default)
 */
struct rad_geom {
    int ims, ime, jms, jme, kms, kme;   /* original patch memory bounds */
    int ids, ide, jds, jde;             /* original computational bounds */
    int NI, NJ, NK, levsiz, n_ozmixm;
    int jsplit, j1, rows;               /* staged rows jsplit..j1 */
    int bjms, bjme, BNJ;                /* buffer j shape */
};

struct rad_fld { int k, lev, nslab; };  /* arg index, floats per i-row, slabs */

static int rad_fields(const struct rad_layout *l, const struct rad_geom *g,
                      struct rad_fld *f)
{
    int a, n = 0;
    for (a = 0; a < l->n3; a++) { f[n].k = l->w3[a]; f[n].lev = g->NK; f[n].nslab = 1; n++; }
    for (a = 0; a < l->n2; a++) { f[n].k = l->w2[a]; f[n].lev = 1; f[n].nslab = 1; n++; }
    if (l->s3x) for (a = 0; a < 8; a++) {
        f[n].k = rad_s3x[a]; f[n].lev = g->NK + 2; f[n].nslab = 1; n++;
    }
    f[n].k = WTR_A_OZMIXM; f[n].lev = g->levsiz; f[n].nslab = g->n_ozmixm; n++;
    return n;
}

/* Lay the fields out in buf (base==NULL: only count, worst-case padding).
 * Each field owns nslab x BNJ rows of NI*lev floats, padded so that its
 * staged block (row jsplit) has the same address mod 64 as the original.
 * With FULLJ the slab stride equals the original one, so every ozmixm slab
 * matches; with NJ'=rows only slab 0 does. */
static size_t rad_layout_buf(const struct rad_fld *f, int nf, const struct rad_geom *g,
                             void **orig, const float *base, size_t *off)
{
    size_t pos = 0;
    int a;
    for (a = 0; a < nf; a++) {
        size_t rowl = (size_t)g->NI * (size_t)f[a].lev;
        if (base) {
            uintptr_t want = ((uintptr_t)orig[f[a].k] +
                              rowl * (size_t)(g->jsplit - g->jms) * 4) % 64;
            uintptr_t have = ((uintptr_t)(base + pos) +
                              rowl * (size_t)(g->jsplit - g->bjms) * 4) % 64;
            pos += (size_t)((want + 64 - have) % 64) / 4;
        } else {
            pos += 16;
        }
        off[a] = pos;
        pos += rowl * (size_t)g->BNJ * (size_t)f[a].nslab;
    }
    return pos;
}

/* The staged block (ims:ime, 1:lev, jsplit:j1) of every slab is contiguous, so
 * pack and commit are one memcpy per slab.  ozmixm is INTENT(IN): packed,
 * never committed. */
static void rad_copy(const struct rad_fld *f, int nf, const struct rad_geom *g,
                     void **orig, float *base, const size_t *off, int to_buf)
{
    int a, m;
    for (a = 0; a < nf; a++) {
        size_t rowl = (size_t)g->NI * (size_t)f[a].lev;
        size_t n = rowl * (size_t)g->rows * sizeof(float);
        if (!to_buf && f[a].k == WTR_A_OZMIXM) continue;
        for (m = 0; m < f[a].nslab; m++) {
            float *b = base + off[a] + rowl * (size_t)g->BNJ * (size_t)m
                     + rowl * (size_t)(g->jsplit - g->bjms);
            float *o = (float *)orig[f[a].k] + rowl * (size_t)g->NJ * (size_t)m
                     + rowl * (size_t)(g->jsplit - g->jms);
            if (to_buf) memcpy(b, o, n); else memcpy(o, b, n);
        }
    }
}

static int rad_env_int(const char *name, int dflt)
{
    const char *s = getenv(name);
    return s && *s ? atoi(s) : dflt;
}


static int rad_pass1(void **save, int jsplit, int nt)
{
    int *is = malloc((size_t)nt * sizeof *is), *ie = malloc((size_t)nt * sizeof *ie);
    int *js = malloc((size_t)nt * sizeof *js), *je = malloc((size_t)nt * sizeof *je);
    const int *ois = (const int *)save[WTR_A_I_START];
    const int *oie = (const int *)save[WTR_A_I_END];
    const int *ojs = (const int *)save[WTR_A_J_START];
    const int *oje = (const int *)save[WTR_A_J_END];
    int n = 0, t;
    /* All four allocations happen before retargeting or the real call, so a
     * malloc failure cannot leave a partially executed radiation driver. */
    if (!is || !ie || !js || !je) goto out;
    for (t = 0; t < nt; t++) {
        int y1 = oje[t] < jsplit - 1 ? oje[t] : jsplit - 1;
        if (y1 < ojs[t]) continue;
        is[n] = ois[t]; ie[n] = oie[t]; js[n] = ojs[t]; je[n] = y1; n++;
    }
    wt_argv[WTR_A_I_START] = is; wt_argv[WTR_A_I_END] = ie;
    wt_argv[WTR_A_J_START] = js; wt_argv[WTR_A_J_END] = je;
    wt_argv[WTR_A_NUM_TILES] = &n;
    if (n > 0) rad_real_call();
    {
        int ok = is && ie && js && je;
        memcpy(wt_argv, save, sizeof(void *) * WTR_NARG);
        free(is); free(ie); free(js); free(je);
        return ok;
    }
out:
    memcpy(wt_argv, save, sizeof(void *) * WTR_NARG);
    free(is); free(ie); free(js); free(je);
    return 0;
}

/* Pass-2 tiles: the original tiles restricted to the staged rows.  isplit<0
 * keeps their j extents (clipped); otherwise one tile per staged row, and
 * isplit>0 additionally chunks its:ite at its+m*isplit so every point keeps
 * its slot in the driver's unrolled i loops and the remainder stays at the
 * true ite.  Returns the tile count, 0 on allocation failure. */
static int rad_pass2_tiles(void **save, const struct rad_geom *g, int nt, int isplit,
                           int **is, int **ie, int **js, int **je)
{
    const int *ois = (const int *)save[WTR_A_I_START];
    const int *oie = (const int *)save[WTR_A_I_END];
    const int *ojs = (const int *)save[WTR_A_J_START];
    const int *oje = (const int *)save[WTR_A_J_END];
    int t, pass, n = 0;
    *is = *ie = *js = *je = NULL;
    for (pass = 0; pass < 2; pass++) {
        if (pass) {
            *is = malloc((size_t)n * sizeof **is); *ie = malloc((size_t)n * sizeof **ie);
            *js = malloc((size_t)n * sizeof **js); *je = malloc((size_t)n * sizeof **je);
            if (!*is || !*ie || !*js || !*je) return 0;
            n = 0;
        }
        for (t = 0; t < nt; t++) {
            int ys = ojs[t] > g->jsplit ? ojs[t] : g->jsplit;
            int ye = oje[t] < g->j1 ? oje[t] : g->j1;
            int y, y0, y1;
            if (ys > ye) continue;
            for (y0 = ys; y0 <= ye; y0 = y1 + 1) {
                y1 = isplit < 0 ? ye : y0;
                for (y = ois[t]; y <= oie[t]; ) {
                    int x1 = isplit > 0 && y + isplit - 1 < oie[t] ? y + isplit - 1 : oie[t];
                    if (pass) { (*is)[n] = y; (*ie)[n] = x1; (*js)[n] = y0; (*je)[n] = y1; }
                    n++;
                    y = x1 + 1;
                }
            }
        }
    }
    return n;
}

/* Tiles were built (and allocated) before pass 1, so nothing can fail here
 * between the two driver calls. */
static void rad_pass2(void **save, const struct rad_fld *f, int nf,
                      const struct rad_geom *g, float *buf, const size_t *off,
                      int n, int *is, int *ie, int *js, int *je)
{
    int s_jms = g->bjms, s_jme = g->bjme, a;
    int s_ims = g->ims, s_ime = g->ime, s_kms = g->kms, s_kme = g->kme;
    int s_ids = g->ids, s_ide = g->ide, s_jds = g->jds, s_jde = g->jde;
    if (buf) {
        for (a = 0; a < nf; a++) wt_argv[f[a].k] = buf + off[a];
        /* Every bound comes from g, i.e. from the OWNER of the rows: a thief's
         * own ims:ime is a different column under cyclic placement, and the tile
         * table holds the owner's absolute i indices (1070712: every thief
         * SIGSEGVed on its first stolen row). For the owner g is its own geometry,
         * so only the j memory extent actually changes (shrinks, or not with FULLJ). */
        wt_argv[WTR_A_IMS] = &s_ims; wt_argv[WTR_A_IME] = &s_ime;
        wt_argv[WTR_A_JMS] = &s_jms; wt_argv[WTR_A_JME] = &s_jme;
        wt_argv[WTR_A_KMS] = &s_kms; wt_argv[WTR_A_KME] = &s_kme;
        wt_argv[WTR_A_IDS] = &s_ids; wt_argv[WTR_A_IDE] = &s_ide;
        wt_argv[WTR_A_JDS] = &s_jds; wt_argv[WTR_A_JDE] = &s_jde;
    }
    wt_argv[WTR_A_I_START] = is; wt_argv[WTR_A_I_END] = ie;
    wt_argv[WTR_A_J_START] = js; wt_argv[WTR_A_J_END] = je;
    wt_argv[WTR_A_NUM_TILES] = &n;
    rad_real_call();
    memcpy(wt_argv, save, sizeof(void *) * WTR_NARG);
}

/* R2 uses one complete RH buffer per rank, with one claim mark per staged row.
 * The field order and offsets are the owner's rad_layout_buf result; a thief
 * never repacks or reconstructs them.  The large fixed header keeps every
 * header/data boundary 64-byte aligned in /dev/shm. */
#define RAD_SHARE_EMPTY        0u
#define RAD_SHARE_PUBLISHED    1u
#define RAD_SHARE_CLAIMED_SELF 2u
#define RAD_SHARE_CLAIMED_OTHER 3u
#define RAD_SHARE_DONE         4u
#define RAD_SHARE_MAX_CHUNKS   64
#define RAD_SHARE_MAX_FIELDS   (WTR_NARG + 16)
#define RAD_SHARE_MAX_TILES    8192

struct rad_share_tile { int is, ie, js, je; };

struct __attribute__((aligned(64))) rad_share_hdr {
    uint32_t state[RAD_SHARE_MAX_CHUNKS];
    int runner[RAD_SHARE_MAX_CHUNKS];
    uint64_t work_ticks[RAD_SHARE_MAX_CHUNKS];
    uint64_t layout_hash, call_id;
    int owner, nchunks, nf, ntiles;
    struct rad_geom g;
    struct rad_fld fields[RAD_SHARE_MAX_FIELDS];
    size_t off[RAD_SHARE_MAX_FIELDS];
    struct rad_share_tile tiles[RAD_SHARE_MAX_TILES];
};
typedef char rad_share_header_alignment[(sizeof(struct rad_share_hdr) % 64 == 0) ? 1 : -1];

static int rad_share_init_state = -1;
static int rad_share_fd = -1, rad_share_me, rad_share_nlocal;
static size_t rad_share_slot_bytes, rad_share_cap, rad_share_bytes;
static void *rad_share_map;
static char rad_share_path[4096];
static uint64_t rad_share_call_id;
static int rad_share_timeout_printed;

static int rad_share_first_int(const char *s)
{
    char *end;
    long v;
    if (!s || !*s) return 0;
    v = strtol(s, &end, 10);
    return end != s && v > 0 && v <= INT_MAX ? (int)v : 0;
}

static int rad_share_local_id(const char *s)
{
    char *end;
    long v;
    if (!s || !*s) return -1;
    v = strtol(s, &end, 10);
    return end != s && v >= 0 && v <= INT_MAX ? (int)v : -1;
}

static size_t rad_share_slot_from_env(void)
{
    const char *s = getenv("WTIMER_RAD_SLOT_BYTES");
    unsigned long long v = 4u * 1024u * 1024u;
    char *end;
    if (s && *s) {
        unsigned long long x = strtoull(s, &end, 10);
        if (end != s) v = x;
    }
    return v <= SIZE_MAX ? (size_t)v : 0;
}

static inline struct rad_share_hdr *rad_share_slot(int owner)
{
    return (struct rad_share_hdr *)((char *)rad_share_map +
                                    (size_t)owner * rad_share_slot_bytes);
}

static inline float *rad_share_data(struct rad_share_hdr *h)
{
    return (float *)((char *)h + sizeof *h);
}

static inline void rad_share_pause(unsigned n)
{
    if ((n & 63u) == 63u) sched_yield();
    else __asm__ __volatile__("pause");
}

static uint64_t rad_share_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void rad_share_finalize(void)
{
    if (rad_share_init_state != 1) return;
    if (rad_share_map) { munmap(rad_share_map, rad_share_bytes); rad_share_map = NULL; }
    if (rad_share_fd >= 0) { close(rad_share_fd); rad_share_fd = -1; }
    if (rad_share_me == 0) unlink(rad_share_path);
    rad_share_init_state = 0;
}

static int rad_share_init(size_t need, int first)
{
    const char *jid;
    int me, nlocal;
    size_t raw;

    if (rad_share_init_state >= 0) return rad_share_init_state;
    rad_share_init_state = 0;
    me = rad_share_local_id(getenv("SLURM_LOCALID"));
    nlocal = rad_share_first_int(getenv("SLURM_STEP_TASKS_PER_NODE"));
    if (!nlocal) nlocal = rad_share_first_int(getenv("SLURM_TASKS_PER_NODE"));
    if (!nlocal) nlocal = rad_share_first_int(getenv("SLURM_NTASKS_PER_NODE"));
    raw = rad_share_slot_from_env();
    if (me < 0 || !nlocal || me >= nlocal || raw < sizeof(struct rad_share_hdr) ||
        raw % 64 != 0 || (size_t)nlocal > SIZE_MAX / raw) return 0;

    jid = getenv("SLURM_JOB_ID");
    if (jid && *jid) snprintf(rad_share_path, sizeof rad_share_path,
                              "/dev/shm/wtrad_%s", jid);
    else snprintf(rad_share_path, sizeof rad_share_path, "/dev/shm/wtrad_%ld", (long)getppid());
    rad_share_fd = open(rad_share_path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (rad_share_fd < 0) { rad_share_fd = -1; return 0; }
    rad_share_slot_bytes = raw;
    rad_share_cap = raw - sizeof(struct rad_share_hdr);
    rad_share_bytes = (size_t)nlocal * raw;
    if (rad_share_bytes > (size_t)LLONG_MAX || ftruncate(rad_share_fd, (off_t)rad_share_bytes) != 0) {
        close(rad_share_fd); rad_share_fd = -1; return 0;
    }
    rad_share_map = mmap(NULL, rad_share_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, rad_share_fd, 0);
    if (rad_share_map == MAP_FAILED) {
        rad_share_map = NULL; close(rad_share_fd); rad_share_fd = -1; return 0;
    }
    rad_share_me = me; rad_share_nlocal = nlocal;
    memset(rad_share_slot(me), 0, sizeof(struct rad_share_hdr));
    for (int y = 0; y < RAD_SHARE_MAX_CHUNKS; y++)
        __atomic_store_n(&rad_share_slot(me)->state[y], RAD_SHARE_EMPTY, __ATOMIC_RELEASE);
    if (atexit(rad_share_finalize) != 0) {
        rad_share_finalize();
        return 0;
    }
    rad_share_init_state = 1;
    if (first)
        fprintf(stderr, "[wtimer] rad share: me=%d nlocal=%d slot=%zu header=%zu cap=%zu need=%zu file=%s\n",
                rad_share_me, rad_share_nlocal, rad_share_slot_bytes,
                sizeof(struct rad_share_hdr), rad_share_cap, need, rad_share_path);
    return 1;
}

/* Shape-only projection of rad_geom for cross-rank compatibility. ims/ime/jms/jme,
 * jsplit/j1 and bjms/bjme are absolute patch bounds and differ on every rank (1070559:
 * share=0 on all 224 ranks because the whole struct was hashed and memcmp'd). The thief
 * runs pass2 with the owner's absolute geometry from the header, so only the extents
 * that size the thief's own non-staged arguments have to agree. */
struct rad_shape { int NI, NJ, NK, kms, kme, ids, ide, jds, jde, levsiz, n_ozmixm, rows, BNJ; };

static struct rad_shape rad_shape_of(const struct rad_geom *g)
{
    struct rad_shape s;
    memset(&s, 0, sizeof s);
    s.NI = g->NI; s.NJ = g->NJ; s.NK = g->NK; s.kms = g->kms; s.kme = g->kme;
    s.ids = g->ids; s.ide = g->ide; s.jds = g->jds; s.jde = g->jde;
    s.levsiz = g->levsiz; s.n_ozmixm = g->n_ozmixm; s.rows = g->rows; s.BNJ = g->BNJ;
    return s;
}

static uint64_t rad_share_hash(const struct rad_fld *f, int nf, const struct rad_geom *g)
{
    uint64_t h = UINT64_C(1469598103934665603);
    struct rad_shape sh = rad_shape_of(g);
    const int *gp = (const int *)&sh;
    int i;
    for (i = 0; i < (int)(sizeof sh / sizeof(int)); i++) {
        h ^= (uint32_t)gp[i]; h *= UINT64_C(1099511628211);
    }
    for (i = 0; i < nf; i++) {
        h ^= (uint32_t)f[i].k; h *= UINT64_C(1099511628211);
        h ^= (uint32_t)f[i].lev; h *= UINT64_C(1099511628211);
        h ^= (uint32_t)f[i].nslab; h *= UINT64_C(1099511628211);
    }
    return h;
}

static int rad_share_payload_ok(const struct rad_share_hdr *h, size_t cap)
{
    size_t end, rowl;
    int a;
    if (h->nf <= 0 || h->nf > RAD_SHARE_MAX_FIELDS || h->ntiles <= 0 ||
        h->ntiles > RAD_SHARE_MAX_TILES || h->nchunks <= 0 ||
        h->nchunks > RAD_SHARE_MAX_CHUNKS) return 0;
    for (a = 0; a < h->nf; a++) {
        if (h->fields[a].lev <= 0 || h->fields[a].nslab <= 0 ||
            h->g.NI <= 0 || h->g.BNJ <= 0) return 0;
        rowl = (size_t)h->g.NI * (size_t)h->fields[a].lev;
        if ((size_t)h->g.BNJ > SIZE_MAX / (size_t)h->fields[a].nslab ||
            rowl > SIZE_MAX / ((size_t)h->g.BNJ * (size_t)h->fields[a].nslab)) return 0;
        end = h->off[a] + rowl * (size_t)h->g.BNJ * (size_t)h->fields[a].nslab;
        if (end < h->off[a] || end > cap / sizeof(float)) return 0;
    }
    for (a = 0; a < h->ntiles; a++) {
        const struct rad_share_tile *t = &h->tiles[a];
        if (t->is < h->g.ims || t->ie > h->g.ime || t->is > t->ie ||
            t->js < h->g.jsplit || t->je > h->g.j1 || t->js != t->je)
            return 0;
    }
    return 1;
}

static int rad_share_compatible(const struct rad_share_hdr *h,
                                const struct rad_fld *f, int nf,
                                const struct rad_geom *g)
{
    int a;
    struct rad_shape mine = rad_shape_of(g), theirs = rad_shape_of(&h->g);
    if (h->layout_hash != rad_share_hash(f, nf, g) || h->nf != nf ||
        memcmp(&theirs, &mine, sizeof mine) != 0) return 0;
    for (a = 0; a < nf; a++)
        if (memcmp(&h->fields[a], &f[a], sizeof f[a]) != 0) return 0;
    return rad_share_payload_ok(h, rad_share_cap);
}

static int rad_share_filter_tiles(const int *is0, const int *ie0,
                                  const int *js0, const int *je0, int n0, int row,
                                  int *is, int *ie, int *js, int *je)
{
    int i, n = 0;
    for (i = 0; i < n0; i++) {
        if (js0[i] != row || je0[i] != row) continue;
        is[n] = is0[i]; ie[n] = ie0[i]; js[n] = row; je[n] = row; n++;
    }
    return n;
}

static int rad_share_filter_header(const struct rad_share_hdr *h, int row,
                                   int *is, int *ie, int *js, int *je)
{
    int i, n = 0;
    for (i = 0; i < h->ntiles; i++) {
        if (h->tiles[i].js != row || h->tiles[i].je != row) continue;
        is[n] = h->tiles[i].is; ie[n] = h->tiles[i].ie;
        js[n] = row; je[n] = row; n++;
    }
    return n;
}

/* Allocate the pass-1 tile vectors before publication.  After publication
 * there is no allocation failure path that could leave a partially executed
 * driver or an orphaned published slot. */
static int rad_share_pass1_prepare(void **save, int jsplit, int nt,
                                   int **is, int **ie, int **js, int **je)
{
    const int *ois = (const int *)save[WTR_A_I_START];
    const int *oie = (const int *)save[WTR_A_I_END];
    const int *ojs = (const int *)save[WTR_A_J_START];
    const int *oje = (const int *)save[WTR_A_J_END];
    int n = 0, t;
    *is = *ie = *js = *je = NULL;
    *is = malloc((size_t)nt * sizeof **is); *ie = malloc((size_t)nt * sizeof **ie);
    *js = malloc((size_t)nt * sizeof **js); *je = malloc((size_t)nt * sizeof **je);
    if (!*is || !*ie || !*js || !*je) return 0;
    for (t = 0; t < nt; t++) {
        int y1 = oje[t] < jsplit - 1 ? oje[t] : jsplit - 1;
        if (y1 < ojs[t]) continue;
        (*is)[n] = ois[t]; (*ie)[n] = oie[t]; (*js)[n] = ojs[t]; (*je)[n] = y1; n++;
    }
    return n;
}

static void rad_share_pass1_run(void **save, int n, int **is, int **ie, int **js, int **je)
{
    wt_argv[WTR_A_I_START] = *is; wt_argv[WTR_A_I_END] = *ie;
    wt_argv[WTR_A_J_START] = *js; wt_argv[WTR_A_J_END] = *je;
    wt_argv[WTR_A_NUM_TILES] = &n;
    if (n > 0) rad_real_call();
    memcpy(wt_argv, save, sizeof(void *) * WTR_NARG);
}

static void rad_share_free_tiles(int **is, int **ie, int **js, int **je)
{
    free(*is); free(*ie); free(*js); free(*je);
    *is = *ie = *js = *je = NULL;
}

static int rad_share_timeout_ms(void)
{
    int ms = rad_env_int("WTIMER_STAGE_SHARE_TIMEOUT_MS", 60000);
    return ms < 0 ? 0 : ms;
}

static void rad_share_steal(const struct rad_fld *ownf, int nf,
                            const struct rad_geom *owng)
{
    int *is = malloc((size_t)RAD_SHARE_MAX_TILES * sizeof *is);
    int *ie = malloc((size_t)RAD_SHARE_MAX_TILES * sizeof *ie);
    int *js = malloc((size_t)RAD_SHARE_MAX_TILES * sizeof *js);
    int *je = malloc((size_t)RAD_SHARE_MAX_TILES * sizeof *je);
    int stolen = 0;
    if (!is || !ie || !js || !je) goto out;
    for (;;) {
        int claimed = 0, off;
        for (off = 1; off < rad_share_nlocal; off++) {
            int owner = (rad_share_me + off) % rad_share_nlocal;
            struct rad_share_hdr *h = rad_share_slot(owner);
            int y;
            /* The owner self-schedules its rows from y=0 upward (G0 1070522:
             * with rows=4 and a monolithic pass1 nothing was ever stolen);
             * thieves take from the tail so the two fronts meet in the middle. */
            for (y = RAD_SHARE_MAX_CHUNKS - 1; y >= 0; y--) {
                uint32_t expected = RAD_SHARE_PUBLISHED;
                if (__atomic_load_n(&h->state[y], __ATOMIC_ACQUIRE) != RAD_SHARE_PUBLISHED)
                    continue;
                if (!__atomic_compare_exchange_n(&h->state[y], &expected,
                                                 RAD_SHARE_CLAIMED_OTHER, 0,
                                                 __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    continue;
                if (h->nchunks <= y || h->nchunks <= 0 ||
                    h->nchunks > RAD_SHARE_MAX_CHUNKS) {
                    __atomic_store_n(&h->state[y], RAD_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                    continue;
                }
                if (!rad_share_compatible(h, ownf, nf, owng) ||
                    rad_share_filter_header(h, h->g.jsplit + y,
                                            is, ie, js, je) <= 0) {
                    __atomic_store_n(&h->state[y], RAD_SHARE_PUBLISHED, __ATOMIC_RELEASE);
                    continue;
                }
                {
                    void *save[WTR_NARG];
                    uint64_t w0 = rad_tsc();
                    int n = rad_share_filter_header(h, h->g.jsplit + y,
                                                    is, ie, js, je);
                    memcpy(save, wt_argv, sizeof save);
                    wt_rad_share_in++;
                    rad_pass2(save, h->fields, h->nf, &h->g, rad_share_data(h), h->off,
                              n, is, ie, js, je);
                    h->runner[y] = rad_share_me;
                    h->work_ticks[y] = rad_tsc() - w0;
                    __atomic_store_n(&h->state[y], RAD_SHARE_DONE, __ATOMIC_RELEASE);
                    claimed = 1; stolen++;
                }
            }
        }
        if (!claimed) break;
    }
out:
    (void)stolen;
    free(is); free(ie); free(js); free(je);
}

static int rad_share_stage(int rows)
{
    struct rad_layout l;
    struct rad_geom g;
    struct rad_fld f[RAD_SHARE_MAX_FIELDS];
    size_t off[RAD_SHARE_MAX_FIELDS], worst, used;
    void *save[WTR_NARG];
    int i0, i1, j0, j1, nt, nf, n2, y, isplit;
    int *is = NULL, *ie = NULL, *js = NULL, *je = NULL;
    int *cis = NULL, *cie = NULL, *cjs = NULL, *cje = NULL;
    int *p1is = NULL, *p1ie = NULL, *p1js = NULL, *p1je = NULL, p1n;
    struct rad_share_hdr *h;
    float *buf;
    uint64_t wait0;
    int any_other = 0, timed_out = 0;

    if (rad_env_int("WTIMER_RAD_BUF", 1) != 1 || rad_env_int("WTIMER_RAD_FULLJ", 0) != 0)
        return 0; /* the RH slot is deliberately not a different G0 variant */
    isplit = rad_env_int("WTIMER_RAD_ISPLIT", 32);
    if (isplit < 0) return 0; /* no one-row claim groups in clipped multi-row tiles */
    g.ims = rad_iarg(WTR_A_IMS); g.ime = rad_iarg(WTR_A_IME);
    g.jms = rad_iarg(WTR_A_JMS); g.jme = rad_iarg(WTR_A_JME);
    g.kms = rad_iarg(WTR_A_KMS); g.kme = rad_iarg(WTR_A_KME);
    g.ids = rad_iarg(WTR_A_IDS); g.ide = rad_iarg(WTR_A_IDE);
    g.jds = rad_iarg(WTR_A_JDS); g.jde = rad_iarg(WTR_A_JDE);
    g.levsiz = rad_iarg(WTR_A_LEVSIZ); g.n_ozmixm = rad_iarg(WTR_A_N_OZMIXM);
    if (!rad_patch_bounds(&i0, &i1, &j0, &j1) || rows <= 0 || rows >= j1-j0+1 ||
        rows > RAD_SHARE_MAX_CHUNKS) return 0;
    (void)i0; (void)i1;
    g.jsplit = j1 - rows + 1; g.j1 = j1; g.rows = rows;
    nt = rad_iarg(WTR_A_NUM_TILES);
    g.NI = g.ime-g.ims+1; g.NJ = g.jme-g.jms+1; g.NK = g.kme-g.kms+1;
    if (g.NI <= 0 || g.NJ <= 0 || g.NK <= 0 || nt <= 0 || g.jsplit < g.jms || j1 > g.jme)
        return 0;
    g.bjms = g.jsplit; g.bjme = j1; g.BNJ = rows;
    rad_build_layout(&l);
    nf = rad_fields(&l, &g, f);
    if (nf <= 0 || nf > RAD_SHARE_MAX_FIELDS) return 0;
    memcpy(save, wt_argv, sizeof save);
    n2 = rad_pass2_tiles(save, &g, nt, isplit, &is, &ie, &js, &je);
    if (n2 <= 0 || n2 > RAD_SHARE_MAX_TILES) goto refuse;
    cis = malloc((size_t)n2 * sizeof *cis); cie = malloc((size_t)n2 * sizeof *cie);
    cjs = malloc((size_t)n2 * sizeof *cjs); cje = malloc((size_t)n2 * sizeof *cje);
    if (!cis || !cie || !cjs || !cje) goto refuse;
    for (y = 0; y < rows; y++) {
        int n = rad_share_filter_tiles(is, ie, js, je, n2, g.jsplit + y,
                                       cis, cie, cjs, cje);
        if (n <= 0) goto refuse;
    }
    p1n = rad_share_pass1_prepare(save, g.jsplit, nt, &p1is, &p1ie, &p1js, &p1je);
    if (p1n < 0 || !p1is || !p1ie || !p1js || !p1je) goto refuse;
    worst = rad_layout_buf(f, nf, &g, save, NULL, off);
    if (worst > SIZE_MAX / sizeof(float) || !rad_share_init(worst * sizeof(float),
                                                              getenv("WTIMER_STAGE_DEBUG") && rad_debug_rank()))
        goto refuse;
    if (worst > rad_share_cap / sizeof(float)) goto refuse;
    h = rad_share_slot(rad_share_me);
    buf = rad_share_data(h);
    used = rad_layout_buf(f, nf, &g, save, buf, off);
    if (used > rad_share_cap / sizeof(float)) goto refuse;
    rad_copy(f, nf, &g, save, buf, off, 1);
    memset(h, 0, sizeof *h);
    h->owner = rad_share_me; h->nchunks = rows; h->nf = nf; h->ntiles = n2;
    h->g = g; h->layout_hash = rad_share_hash(f, nf, &g); h->call_id = ++rad_share_call_id;
    for (y = 0; y < nf; y++) { h->fields[y] = f[y]; h->off[y] = off[y]; }
    for (y = 0; y < n2; y++) {
        h->tiles[y].is = is[y]; h->tiles[y].ie = ie[y];
        h->tiles[y].js = js[y]; h->tiles[y].je = je[y];
    }
    for (y = 0; y < rows; y++) {
        h->runner[y] = -1;
        __atomic_store_n(&h->state[y], RAD_SHARE_PUBLISHED, __ATOMIC_RELEASE);
    }
    rad_share_pass1_run(save, p1n, &p1is, &p1ie, &p1js, &p1je);
    rad_share_free_tiles(&p1is, &p1ie, &p1js, &p1je);
    for (y = 0; y < rows; y++) {
        uint32_t expected = RAD_SHARE_PUBLISHED;
        if (__atomic_compare_exchange_n(&h->state[y], &expected, RAD_SHARE_CLAIMED_SELF, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            int n = rad_share_filter_tiles(is, ie, js, je, n2, g.jsplit + y,
                                           cis, cie, cjs, cje);
            uint64_t w0 = rad_tsc();
            h->runner[y] = rad_share_me;
            rad_pass2(save, f, nf, &g, buf, off, n, cis, cie, cjs, cje);
            h->work_ticks[y] = rad_tsc() - w0;
            __atomic_store_n(&h->state[y], RAD_SHARE_DONE, __ATOMIC_RELEASE);
        } else any_other = 1;
    }
    wait0 = rad_share_now_us();
    for (y = 0; y < rows; y++) {
        unsigned spin = 0;
        for (;;) {
            uint32_t state = __atomic_load_n(&h->state[y], __ATOMIC_ACQUIRE);
            if (state == RAD_SHARE_DONE) break;
            if (!timed_out && rad_share_timeout_ms() > 0 &&
                rad_share_now_us() - wait0 >= (uint64_t)rad_share_timeout_ms() * 1000u) {
                if (!rad_share_timeout_printed) {
                    fprintf(stderr, "[wtimer] rad share: owner wait exceeded %d ms; waiting for DONE\n",
                            rad_share_timeout_ms());
                    rad_share_timeout_printed = 1;
                }
                wt_rad_refused++; timed_out = 1;
            }
            rad_share_pause(spin++);
        }
        if (h->runner[y] != rad_share_me) {
            wt_rad_share_out++;
            any_other = 1;
        }
    }
    if (any_other) wt_rad_share_wait_us += rad_share_now_us() - wait0;
    rad_copy(f, nf, &g, save, buf, off, 0);
    memcpy(wt_argv, save, sizeof save);
    for (y = 0; y < rows; y++) __atomic_store_n(&h->state[y], RAD_SHARE_EMPTY, __ATOMIC_RELEASE);
    rad_share_free_tiles(&is, &ie, &js, &je);
    rad_share_free_tiles(&cis, &cie, &cjs, &cje);
    rad_share_steal(f, nf, &g);
    return 1;
refuse:
    rad_share_free_tiles(&p1is, &p1ie, &p1js, &p1je);
    rad_share_free_tiles(&is, &ie, &js, &je);
    rad_share_free_tiles(&cis, &cie, &cjs, &cje);
    memcpy(wt_argv, save, sizeof save);
    return 0;
}

static int rad_stage(int rows)
{
    struct rad_layout l;
    struct rad_geom g;
    struct rad_fld f[WTR_NARG + 9];
    size_t off[WTR_NARG + 9];
    void *save[WTR_NARG];
    int i0, i1, j0, j1, nt, nf, n2;
    int *is, *ie, *js, *je;
    int use_buf = rad_env_int("WTIMER_RAD_BUF", 1);
    int isplit = rad_env_int("WTIMER_RAD_ISPLIT", 32);
    int fullj = rad_env_int("WTIMER_RAD_FULLJ", 0);
    size_t elems;
    float *buf = NULL;

    g.ims = rad_iarg(WTR_A_IMS); g.ime = rad_iarg(WTR_A_IME);
    g.jms = rad_iarg(WTR_A_JMS); g.jme = rad_iarg(WTR_A_JME);
    g.kms = rad_iarg(WTR_A_KMS); g.kme = rad_iarg(WTR_A_KME);
    g.ids = rad_iarg(WTR_A_IDS); g.ide = rad_iarg(WTR_A_IDE);
    g.jds = rad_iarg(WTR_A_JDS); g.jde = rad_iarg(WTR_A_JDE);
    g.levsiz = rad_iarg(WTR_A_LEVSIZ); g.n_ozmixm = rad_iarg(WTR_A_N_OZMIXM);
    if (!rad_patch_bounds(&i0, &i1, &j0, &j1)) return 0;
    if (rows <= 0 || rows >= j1-j0+1) return 0;
    g.jsplit = j1 - rows + 1; g.j1 = j1; g.rows = rows;
    nt = rad_iarg(WTR_A_NUM_TILES);
    g.NI = g.ime-g.ims+1; g.NJ = g.jme-g.jms+1; g.NK = g.kme-g.kms+1;
    if (g.NI <= 0 || g.NJ <= 0 || g.NK <= 0 || nt <= 0) return 0;
    if (g.jsplit < g.jms || j1 > g.jme) return 0;
    if (fullj) { g.bjms = g.jms; g.bjme = g.jme; }
    else       { g.bjms = g.jsplit; g.bjme = j1; }
    g.BNJ = g.bjme - g.bjms + 1;
    rad_build_layout(&l);
    nf = rad_fields(&l, &g, f);
    memcpy(save, wt_argv, sizeof save);
    /* Everything that can fail happens before the first driver call. */
    n2 = rad_pass2_tiles(save, &g, nt, isplit, &is, &ie, &js, &je);
    if (n2 <= 0) goto refuse;
    if (use_buf) {
        elems = rad_layout_buf(f, nf, &g, save, NULL, off);
        if (elems > SIZE_MAX / sizeof(float)) goto refuse;
        if (posix_memalign((void **)&buf, 64, elems * sizeof(float)) != 0) { buf = NULL; goto refuse; }
        rad_layout_buf(f, nf, &g, save, buf, off);
        rad_copy(f, nf, &g, save, buf, off, 1);
    }
    if (wt_rad_staged == 0 && getenv("WTIMER_STAGE_DEBUG") && rad_debug_rank())
        fprintf(stderr, "[wtimer] rad stage geometry: buf=%d isplit=%d fullj=%d rows=%d jsplit=%d "
                "NI=%d NJ=%d NK=%d BNJ=%d fields=%d elems=%zu tiles=%d (orig %d) first=[%d-%d]x[%d-%d] last=[%d-%d]x[%d-%d]\n",
                use_buf, isplit, fullj, rows, g.jsplit, g.NI, g.NJ, g.NK, g.BNJ, nf,
                use_buf ? elems : (size_t)0, n2, nt, is[0], ie[0], js[0], je[0],
                is[n2-1], ie[n2-1], js[n2-1], je[n2-1]);
    if (!rad_pass1(save, g.jsplit, nt)) goto refuse;
    rad_pass2(save, f, nf, &g, buf, off, n2, is, ie, js, je);
    if (use_buf) rad_copy(f, nf, &g, save, buf, off, 0);
    memcpy(wt_argv, save, sizeof save);
    free(buf); free(is); free(ie); free(js); free(je);
    return 1;
refuse:
    memcpy(wt_argv, save, sizeof save);
    free(buf); free(is); free(ie); free(js); free(je);
    return 0;
}

void wt_rad_call(void)
{
    uint64_t t0 = rad_tsc();
    int rows, rank;
    wt_pre(1);
    wt_rad_calls++;
    rows = 0;
    {
        const char *s = getenv("WTIMER_RAD_ROWS");
        if (s) rows = atoi(s);
    }
    if (rows <= 0) {
        rad_real_call();
    } else {
        static int checked = 0, ok = 0;
        static char reason[160];
        int due = rad_due();
        rank = rad_rank();
        if (!checked) {
            ok = rad_selfcheck(reason, sizeof reason);
            checked = 1;
            if (getenv("WTIMER_STAGE_DEBUG") && rad_debug_rank()) {
                rad_print_selfcheck(rank, ok, reason);
            }
        }
        if (!due) {
            rad_real_call();
        } else {
            wt_rad_steps++;
            if (!ok) {
                wt_rad_refused++;
                rad_real_call();
            } else if (rad_env_int("WTIMER_RAD_SHARE", 0) == 1) {
                if (rad_share_stage(rows)) {
                    wt_rad_staged++;
                } else {
                    wt_rad_refused++;
                    rad_real_call();
                }
            } else if (!rad_stage(rows)) {
                wt_rad_refused++;
                rad_real_call();
            } else {
                wt_rad_staged++;
            }
        }
    }
    wt_record(1, rad_tsc() - t0);
}
