#include "wt_mynn_layout.h"
#include <limits.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8 && sizeof(uint32_t) == 4 && sizeof(int32_t) == 4,
               "MYNN layout needs exact four-byte integers");

static int multiply(size_t a, size_t b, size_t *value)
{
    if (b && a > SIZE_MAX / b) return 0;
    *value = a * b;
    return 1;
}

static int extent(int32_t low, int32_t high, size_t *value)
{
    uint64_t n;
    if (high < low) return 0;
    n = (uint64_t)((int64_t)high - (int64_t)low) + 1;
    if (n > SIZE_MAX) return 0;
    *value = (size_t)n;
    return 1;
}

static int32_t integer(const struct wt_mynn_layout *plan, unsigned index)
{
    int32_t value;
    memcpy(&value, &plan->scalar_bits[index], 4);
    return value;
}

static int condition(unsigned code, unsigned present, const struct wt_mynn_branches *b)
{
    switch (code) {
    case WT_MYNN_COND_NEVER: return 0;
    case WT_MYNN_COND_ALWAYS: return 1;
    case WT_MYNN_COND_PRESENT: return present != 0;
    case WT_MYNN_COND_QC: return b->qc;
    case WT_MYNN_COND_QI: return b->qi;
    case WT_MYNN_COND_QS: return b->qs;
    case WT_MYNN_COND_QNC: return b->qnc;
    case WT_MYNN_COND_QNI: return b->qni;
    case WT_MYNN_COND_QNWFA: return b->qnwfa;
    case WT_MYNN_COND_QNIFA: return b->qnifa;
    case WT_MYNN_COND_QNBCA: return b->qnbca;
    case WT_MYNN_COND_PLUME: return b->bl_mynn_edmf > 0;
    case WT_MYNN_COND_OUTPUT: return b->bl_mynn_output > 0;
    case WT_MYNN_COND_SPP: return b->spp_pbl > 0;
    case WT_MYNN_COND_BUDGET: return b->tke_budget == 1;
    default: return -1;
    }
}

static enum wt_mynn_layout_status shape_bytes(const struct wt_mynn_layout *p,
                                              size_t *plane, size_t *volume)
{
    int32_t ims = integer(p, WT_MYNN_IMS), ime = integer(p, WT_MYNN_IME);
    int32_t jms = integer(p, WT_MYNN_JMS), jme = integer(p, WT_MYNN_JME);
    int32_t kms = integer(p, WT_MYNN_KMS), kme = integer(p, WT_MYNN_KME);
    int32_t its = integer(p, WT_MYNN_ITS), ite = integer(p, WT_MYNN_ITE);
    int32_t jts = integer(p, WT_MYNN_JTS), jte = integer(p, WT_MYNN_JTE);
    int32_t kts = integer(p, WT_MYNN_KTS), kte = integer(p, WT_MYNN_KTE);
    size_t ni, nj, nk;
    /* Driver 339-353/585-700 indexes complete tile, not its unused itf/jtf.
     * Packed locals use kts:kte (190ff), while pre/post_run 819-840/900-929
     * have explicit 1:kte and parameter kts=1. Preserve that actual indexing.
     * Require nonempty domain bounds; this also makes ide-1/jde-1 safe (266-267).
     * This does not validate numerical/physical constraints inside the MYNN core.
     */
    if (integer(p, WT_MYNN_IDS) >= integer(p, WT_MYNN_IDE) ||
        integer(p, WT_MYNN_JDS) >= integer(p, WT_MYNN_JDE) ||
        integer(p, WT_MYNN_KDS) >= integer(p, WT_MYNN_KDE) ||
        ims > its || its > ite || ite > ime || jms > jts || jts > jte || jte > jme ||
        kts != 1 || kte < 1 || kms > 1 || kte > kme)
        return WT_MYNN_LAYOUT_BOUNDS;
    if (!extent(ims, ime, &ni) || !extent(jms, jme, &nj) || !extent(kms, kme, &nk) ||
        !multiply(ni, nj, plane) || !multiply(*plane, 4, plane) ||
        !multiply(*plane, nk, volume)) return WT_MYNN_LAYOUT_OVERFLOW;
    return WT_MYNN_LAYOUT_OK;
}

enum wt_mynn_layout_status wt_mynn_layout_build(
    void *const argv[WT_MYNN_NARG], const size_t capacities[WT_MYNN_NARG],
    const struct wt_mynn_layout_config *config, struct wt_mynn_layout *out)
{
    struct wt_mynn_layout plan = {0};
    struct wt_mynn_branches *b = &plan.branches;
    enum wt_mynn_layout_status status;
    size_t plane, volume;
    unsigned i;
    if (!out) return WT_MYNN_LAYOUT_ARGUMENT;
    memset(out, 0, sizeof *out);
    if (!config) return WT_MYNN_LAYOUT_CONFIG;
    if ((config->enabled != 0 && config->enabled != 1) ||
        (config->abi_verified != 0 && config->abi_verified != 1)) return WT_MYNN_LAYOUT_CONFIG;
    if (!config->enabled) return WT_MYNN_LAYOUT_DISABLED;
    if (!config->abi_verified) return WT_MYNN_LAYOUT_ABI;
    if (config->logical_true_bits != 1 && config->logical_true_bits != UINT32_MAX)
        return WT_MYNN_LAYOUT_CONFIG;
    if (!argv || !capacities) return WT_MYNN_LAYOUT_ARGUMENT;

    /* Complete interval preflight precedes ALL owner reads, even scalar reads. */
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        uintptr_t address = (uintptr_t)argv[i];
        if (!argv[i]) {
            if (!wt_mynn_args[i].optional || capacities[i]) return WT_MYNN_LAYOUT_ARGUMENT;
            continue;
        }
        if (capacities[i] < 4 || capacities[i] > UINTPTR_MAX - address)
            return WT_MYNN_LAYOUT_CAPACITY;
        if (address % 4) return WT_MYNN_LAYOUT_ARGUMENT;
        plan.present[i] = 1;
    }
    for (i = 0; i < WT_MYNN_NARG; ++i) if (!wt_mynn_args[i].rank) {
        memcpy(&plan.scalar_bits[i], argv[i], 4);
        if (wt_mynn_args[i].type == WT_MYNN_LOGICAL && plan.scalar_bits[i] != 0 &&
            plan.scalar_bits[i] != config->logical_true_bits) return WT_MYNN_LAYOUT_LOGICAL;
    }
    b->initflag = integer(&plan, WT_MYNN_INITFLAG);
    b->tke_budget = integer(&plan, WT_MYNN_TKE_BUDGET);
    b->bl_mynn_edmf = integer(&plan, WT_MYNN_BL_MYNN_EDMF);
    b->bl_mynn_output = integer(&plan, WT_MYNN_BL_MYNN_OUTPUT);
    b->spp_pbl = integer(&plan, WT_MYNN_SPP_PBL);
    b->restart = plan.scalar_bits[WT_MYNN_RESTART] != 0;
    b->cycling = plan.scalar_bits[WT_MYNN_CYCLING] != 0;
    b->tkeadvect = plan.scalar_bits[WT_MYNN_BL_MYNN_TKEADVECT] != 0;
    b->qc = plan.scalar_bits[WT_MYNN_FLAG_QC] != 0;
    b->qi = plan.scalar_bits[WT_MYNN_FLAG_QI] != 0;
    b->qs = plan.scalar_bits[WT_MYNN_FLAG_QS] != 0;
    b->qnc = plan.scalar_bits[WT_MYNN_FLAG_QNC] != 0;
    b->qni = plan.scalar_bits[WT_MYNN_FLAG_QNI] != 0;
    b->qnwfa = plan.scalar_bits[WT_MYNN_FLAG_QNWFA] != 0;
    b->qnifa = plan.scalar_bits[WT_MYNN_FLAG_QNIFA] != 0;
    b->qnbca = plan.scalar_bits[WT_MYNN_FLAG_QNBCA] != 0;
    if (b->tke_budget != 0) return WT_MYNN_LAYOUT_BRANCH;
    status = shape_bytes(&plan, &plane, &volume);
    if (status != WT_MYNN_LAYOUT_OK) return status;

    for (i = 0; i < WT_MYNN_NARG; ++i) {
        const struct wt_mynn_arg_metadata *arg = &wt_mynn_args[i];
        int active = condition(arg->access, plan.present[i], b);
        int write = condition(arg->write, plan.present[i], b);
        size_t bytes = 4;
        if (active < 0 || write < 0 || (!arg->rank && write)) return WT_MYNN_LAYOUT_CONFIG;
        if ((active || write) && !plan.present[i]) return WT_MYNN_LAYOUT_ARGUMENT;
        if (!plan.present[i]) continue;
        if (active || write) {
            if (arg->rank == 2) bytes = plane;
            else if (arg->rank == 3) bytes = volume;
            else if (arg->rank != 0) return WT_MYNN_LAYOUT_CONFIG;
        }
        if (bytes > capacities[i] || bytes > UINTPTR_MAX - (uintptr_t)argv[i])
            return WT_MYNN_LAYOUT_CAPACITY;
        if (bytes > SIZE_MAX - plan.total_bytes) return WT_MYNN_LAYOUT_OVERFLOW;
        plan.total_bytes += bytes;
        plan.required_bytes[i] = bytes;
        plan.active[i] = (unsigned char)active;
        plan.writable[i] = (unsigned char)write;
    }
    if (config->byte_budget && plan.total_bytes > config->byte_budget) return WT_MYNN_LAYOUT_BUDGET;
    plan.ready = 1;
    *out = plan;
    return WT_MYNN_LAYOUT_OK;
}
