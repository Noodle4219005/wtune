#include "wt_rk_layout.h"
#include <stdint.h>
#include <string.h>

#define META(n, s, t, h, w) [WT_RK_##n] = {s, t, h, w},
const struct wt_rk_arg_metadata wt_rk_args[WT_RK_NARG] = { WT_RK_SCHEMA_ARGS(META) };
#undef META

static const char *const canonical[WT_RK_NARG] = {
    "scs","sce","config_flags","tenddec","rk_step","dt_step","ru","rv","ww","wwE","wwI","u_old","v_old","mut","mub","mu_old","c1h","c2h","c1f","c2f","alt","scalar_old","scalar","scalar_tends","advect_tend","h_tendency","z_tendency","RQVFTEN","base","moist_step","fnm","fnp","msfux","msfuy","msfvx","msfvx_inv","msfvy","msftx","msfty","rdx","rdy","rdn","rdnw","khdif","kvdif","xkmhd","diff_6th_opt","diff_6th_factor","adv_opt","phb","ph","mix2_off","mix6_off","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"
};

static int schema_ok(void) {
    unsigned i;
    for (i = 0; i < WT_RK_NARG; ++i)
        if (!wt_rk_args[i].name || strcmp(wt_rk_args[i].name, canonical[i])) return 0;
    return wt_rk_args[WT_RK_CONFIG_FLAGS].type == WT_RK_OPAQUE &&
           wt_rk_args[WT_RK_CONFIG_FLAGS].shape == WT_RK_SHAPE_SCALAR;
}
static int multiply(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b; return 1;
}
static int extent(int64_t low, int64_t high, size_t *out) {
    uint64_t width;
    if (low > high) return 0;
    width = (uint64_t)(high - low) + 1u;
    if (width > SIZE_MAX) return 0;
    *out = (size_t)width; return 1;
}
static int32_t integer(const struct wt_rk_layout *p, unsigned i) {
    int32_t value; memcpy(&value, &p->scalar_bits[i], sizeof value); return value;
}
static int boolean(int value) { return value == 0 || value == 1; }

static int query_valid(const struct wt_rk_native_result *q) {
    return q && q->status == WT_RK_NATIVE_OK && q->rk_ord > 0 &&
           boolean(q->zadvect_implicit) && q->h_sca_adv_order > 0 && q->v_sca_adv_order > 0 &&
           boolean(q->specified) && boolean(q->nested) && boolean(q->periodic_x) &&
           boolean(q->polar) && boolean(q->open_xs) && boolean(q->open_xe) &&
           boolean(q->open_ys) && boolean(q->open_ye);
}
int wt_rk_selector_equal(const struct wt_rk_native_result *a, const struct wt_rk_native_result *b) {
    return a && b && a->status == b->status && a->rk_ord == b->rk_ord &&
           a->zadvect_implicit == b->zadvect_implicit && a->h_sca_adv_order == b->h_sca_adv_order &&
           a->v_sca_adv_order == b->v_sca_adv_order && a->cu_physics == b->cu_physics &&
           a->diff_opt == b->diff_opt && a->bl_pbl_physics == b->bl_pbl_physics &&
           a->time_step == b->time_step && a->specified == b->specified && a->nested == b->nested &&
           a->periodic_x == b->periodic_x && a->polar == b->polar && a->open_xs == b->open_xs &&
           a->open_xe == b->open_xe && a->open_ys == b->open_ys && a->open_ye == b->open_ye;
}

static enum wt_rk_layout_status dimensions(const struct wt_rk_layout *p, size_t *vector, size_t *plane, size_t *volume) {
    int64_t ids=integer(p,WT_RK_IDS), ide=integer(p,WT_RK_IDE), jds=integer(p,WT_RK_JDS), jde=integer(p,WT_RK_JDE);
    int64_t kds=integer(p,WT_RK_KDS), kde=integer(p,WT_RK_KDE), ims=integer(p,WT_RK_IMS), ime=integer(p,WT_RK_IME);
    int64_t jms=integer(p,WT_RK_JMS), jme=integer(p,WT_RK_JME), kms=integer(p,WT_RK_KMS), kme=integer(p,WT_RK_KME);
    int64_t its=integer(p,WT_RK_ITS), ite=integer(p,WT_RK_ITE), jts=integer(p,WT_RK_JTS), jte=integer(p,WT_RK_JTE);
    int64_t kts=integer(p,WT_RK_KTS), kte=integer(p,WT_RK_KTE), ktf; size_t ni,nj,nk;
    if (ids>=ide || jds>=jde || kds>=kde || ims>ime || jms>jme || kms>kme || its>ite || jts>jte || kts>kte ||
        its<ids+1 || ite>ide-2 || jts<jds+1 || jte>jde-2 || ims>its-1 || ite+1>ime ||
        jms>jts-1 || jte+1>jme || kms>kts) return WT_RK_LAYOUT_BOUNDS;
    ktf = kte < kde-1 ? kte : kde-1;
    if (kts+1>ktf || ktf+1>kte || kte>kme) return WT_RK_LAYOUT_BOUNDS;
    if (!extent(ims,ime,&ni) || !extent(jms,jme,&nj) || !extent(kms,kme,&nk) ||
        !multiply(nk,4,vector) || !multiply(ni,nj,plane) || !multiply(*plane,4,plane) ||
        !multiply(*plane,nk,volume)) return WT_RK_LAYOUT_OVERFLOW;
    return WT_RK_LAYOUT_OK;
}
static int config_valid(const struct wt_rk_layout_config *c) {
    return c && boolean(c->enabled) && boolean(c->source_abi_verified) && boolean(c->caller_closure_verified) &&
           boolean(c->config_read_lease) && boolean(c->actual_omp_in_parallel) && boolean(c->exclusive_owner) &&
           boolean(c->allocation_descriptor_present) && (c->logical_true_bits==1 || c->logical_true_bits==UINT32_MAX);
}
static enum wt_rk_layout_status bytes_for_shape(unsigned shape, size_t vector, size_t plane, size_t volume, size_t *out) {
    switch (shape) {
    case WT_RK_SHAPE_SCALAR: *out=4; return WT_RK_LAYOUT_OK;
    case WT_RK_SHAPE_VECTOR: *out=vector; return WT_RK_LAYOUT_OK;
    case WT_RK_SHAPE_PLANE: *out=plane; return WT_RK_LAYOUT_OK;
    case WT_RK_SHAPE_VOLUME: case WT_RK_SHAPE_SPECIES_VOLUME: *out=volume; return WT_RK_LAYOUT_OK;
    default: return WT_RK_LAYOUT_CONFIG;
    }
}

enum wt_rk_layout_status wt_rk_layout_build(void *const argv[WT_RK_NARG], const size_t capacities[WT_RK_NARG], const struct wt_rk_layout_config *c, struct wt_rk_layout *out) {
    struct wt_rk_layout plan={0};
    struct wt_rk_native_result before, after;
    size_t vector,plane,volume; enum wt_rk_layout_status status; unsigned i;
    if (!out) return WT_RK_LAYOUT_ARGUMENT;
    memset(out,0,sizeof *out);
    if (!config_valid(c)) return WT_RK_LAYOUT_CONFIG;
    if (!c->enabled) return WT_RK_LAYOUT_DISABLED;
    if (!c->source_abi_verified || !c->caller_closure_verified) return WT_RK_LAYOUT_ABI;
    if (!c->config_read_lease || c->actual_omp_in_parallel || !c->exclusive_owner) return WT_RK_LAYOUT_CONFIG;
    if (!c->byte_budget && c->allocation_descriptor_present) return WT_RK_LAYOUT_BUDGET;
    if (!argv || !capacities || !c->query_before || !c->query_after || !schema_ok()) return WT_RK_LAYOUT_ARGUMENT;
    if (!c->config_identity || argv[WT_RK_CONFIG_FLAGS]!=c->config_identity || capacities[WT_RK_CONFIG_FLAGS]) return WT_RK_LAYOUT_ARGUMENT;
    for (i=0;i<WT_RK_NARG;++i) {
        uintptr_t address;
        if (i==WT_RK_CONFIG_FLAGS) { plan.present[i]=1; continue; }
        address=(uintptr_t)argv[i];
        if (!argv[i]) return WT_RK_LAYOUT_ARGUMENT;
        if (capacities[i]<4 || capacities[i]>UINTPTR_MAX-address) return WT_RK_LAYOUT_CAPACITY;
        if (address%4) return WT_RK_LAYOUT_ARGUMENT;
        plan.present[i]=1;
    }
    for (i=0;i<WT_RK_NARG;++i) if (wt_rk_args[i].shape==WT_RK_SHAPE_SCALAR && i!=WT_RK_CONFIG_FLAGS) {
        memcpy(&plan.scalar_bits[i],argv[i],4);
        if (wt_rk_args[i].type==WT_RK_LOGICAL && plan.scalar_bits[i] && plan.scalar_bits[i]!=c->logical_true_bits) return WT_RK_LAYOUT_LOGICAL;
    }
    if (c->query_before == c->query_after) return WT_RK_LAYOUT_QUERY;
    before = *c->query_before;
    after = *c->query_after;
    if (!query_valid(&before) || !query_valid(&after) || !wt_rk_selector_equal(&before,&after)) return WT_RK_LAYOUT_QUERY;
    plan.selectors=before;
    if (integer(&plan,WT_RK_SCS)!=integer(&plan,WT_RK_SCE) || plan.scalar_bits[WT_RK_MOIST_STEP] ||
        integer(&plan,WT_RK_RK_STEP)<2 || integer(&plan,WT_RK_RK_STEP)>=plan.selectors.rk_ord ||
        plan.selectors.h_sca_adv_order!=2 || plan.selectors.v_sca_adv_order!=2) return WT_RK_LAYOUT_BRANCH;
    status=dimensions(&plan,&vector,&plane,&volume);
    if (status!=WT_RK_LAYOUT_OK) return status;
    for (i=0;i<WT_RK_NARG;++i) {
        size_t bytes;
        if (i==WT_RK_CONFIG_FLAGS) continue;
        status=bytes_for_shape(wt_rk_args[i].shape,vector,plane,volume,&bytes);
        if (status!=WT_RK_LAYOUT_OK) return status;
        if (bytes>capacities[i] || bytes>UINTPTR_MAX-(uintptr_t)argv[i]) return WT_RK_LAYOUT_CAPACITY;
        if (plan.total_bytes>SIZE_MAX-bytes) return WT_RK_LAYOUT_OVERFLOW;
        plan.required_bytes[i]=bytes; plan.total_bytes+=bytes; plan.writable[i]=wt_rk_args[i].writable;
    }
    if (c->byte_budget && plan.total_bytes>c->byte_budget) return WT_RK_LAYOUT_BUDGET;
    plan.ready=1; *out=plan; return WT_RK_LAYOUT_OK;
}
