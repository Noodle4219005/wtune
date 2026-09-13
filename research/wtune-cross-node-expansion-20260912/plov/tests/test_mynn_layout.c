#include "wt_mynn_layout.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t backing[WT_MYNN_NARG][64];
static void *args[WT_MYNN_NARG];
static size_t caps[WT_MYNN_NARG];
static struct wt_mynn_layout_config config;
static struct wt_mynn_layout plan;

static void set_i(unsigned i, int32_t value) { memcpy(backing[i], &value, 4); }
static void absent(unsigned i) { args[i] = NULL; caps[i] = 0; }
static void present(unsigned i) { args[i] = backing[i]; caps[i] = 256; }
static void setup(void)
{
    unsigned i;
    memset(backing, 0, sizeof backing);
    config = (struct wt_mynn_layout_config){1, 1, 1, 0};
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        args[i] = backing[i];
        caps[i] = wt_mynn_args[i].rank ? 256 : 4;
        if (wt_mynn_args[i].optional) absent(i);
    }
    set_i(WT_MYNN_IDS, 0); set_i(WT_MYNN_IDE, 2);
    set_i(WT_MYNN_JDS, -1); set_i(WT_MYNN_JDE, 1);
    set_i(WT_MYNN_KDS, 1); set_i(WT_MYNN_KDE, 4);
    set_i(WT_MYNN_IMS, -1); set_i(WT_MYNN_IME, 2);
    set_i(WT_MYNN_JMS, -2); set_i(WT_MYNN_JME, 1);
    set_i(WT_MYNN_KMS, 0); set_i(WT_MYNN_KME, 3);
    set_i(WT_MYNN_ITS, 0); set_i(WT_MYNN_ITE, 1);
    set_i(WT_MYNN_JTS, -1); set_i(WT_MYNN_JTE, 0);
    set_i(WT_MYNN_KTS, 1); set_i(WT_MYNN_KTE, 3);
    present(WT_MYNN_QV); present(WT_MYNN_QC_BL);
    present(WT_MYNN_QI_BL); present(WT_MYNN_CLDFRA_BL);
}

static enum wt_mynn_layout_status build(void)
{
    return wt_mynn_layout_build(args, caps, &config, &plan);
}

static void refuse(enum wt_mynn_layout_status status)
{
    struct wt_mynn_layout zero = {0};
    memset(&plan, 0xa5, sizeof plan);
    assert(build() == status);
    assert(memcmp(&plan, &zero, sizeof plan) == 0);
}

static void basic(void)
{
    unsigned i;
    size_t sum = 0;
    setup(); assert(build() == WT_MYNN_LAYOUT_OK); assert(plan.ready);
    assert(plan.required_bytes[WT_MYNN_U] == 256);
    assert(plan.required_bytes[WT_MYNN_PBLH] == 64);
    assert(plan.required_bytes[WT_MYNN_IDS] == 4);
    assert(plan.active[WT_MYNN_QC_BL] && plan.writable[WT_MYNN_QC_BL]);
    assert(plan.active[WT_MYNN_QV] && !plan.writable[WT_MYNN_QV]);
    assert(!plan.present[WT_MYNN_DQKE] && !plan.required_bytes[WT_MYNN_DQKE]);
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        sum += plan.required_bytes[i];
        if (!wt_mynn_args[i].rank) assert(!plan.writable[i]);
    }
    assert(sum == plan.total_bytes);
    set_i(WT_MYNN_IDS, 777);
    assert(plan.scalar_bits[WT_MYNN_IDS] == 0);
    setup(); config.byte_budget = sum; assert(build() == WT_MYNN_LAYOUT_OK);
    config.byte_budget = sum - 1; refuse(WT_MYNN_LAYOUT_BUDGET);
    setup(); backing[WT_MYNN_DELT][0] = UINT32_C(0x7fc01234);
    assert(build() == WT_MYNN_LAYOUT_OK);
    assert(plan.scalar_bits[WT_MYNN_DELT] == UINT32_C(0x7fc01234));
}

static void early_guards(void)
{
    struct wt_mynn_layout zero = {0};
    volatile uintptr_t inaccessible = 1;
    setup(); config.enabled = 0;
    assert(wt_mynn_layout_build((void *const *)inaccessible, (size_t *)inaccessible, &config, &plan) == WT_MYNN_LAYOUT_DISABLED);
    config.enabled = 1; config.abi_verified = 0;
    assert(wt_mynn_layout_build((void *const *)inaccessible, (size_t *)inaccessible, &config, &plan) == WT_MYNN_LAYOUT_ABI);
    config.abi_verified = 2; refuse(WT_MYNN_LAYOUT_CONFIG);
    config.abi_verified = 1; config.enabled = -1; refuse(WT_MYNN_LAYOUT_CONFIG);
    setup(); config.logical_true_bits = 2; refuse(WT_MYNN_LAYOUT_CONFIG);
    setup(); args[WT_MYNN_IDS] = (void *)1; caps[WT_MYNN_IDS] = 0;
    refuse(WT_MYNN_LAYOUT_CAPACITY);
    /* An invalid late array must be refused before a preceding scalar is read. */
    setup(); args[WT_MYNN_IDS] = (void *)4;
    args[WT_MYNN_PATTERN_SPP_PBL] = (void *)1; caps[WT_MYNN_PATTERN_SPP_PBL] = 0;
    refuse(WT_MYNN_LAYOUT_CAPACITY);
    setup(); args[WT_MYNN_IDS] = (void *)1; refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); args[WT_MYNN_U] = (void *)(UINTPTR_MAX - 3); caps[WT_MYNN_U] = 256;
    refuse(WT_MYNN_LAYOUT_CAPACITY);
    setup(); absent(WT_MYNN_IDS); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); caps[WT_MYNN_DQKE] = 4; refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); assert(wt_mynn_layout_build(NULL, caps, &config, &plan) == WT_MYNN_LAYOUT_ARGUMENT);
    assert(memcmp(&plan, &zero, sizeof plan) == 0);
    assert(wt_mynn_layout_build(args, caps, &config, NULL) == WT_MYNN_LAYOUT_ARGUMENT);
}

static void species(void)
{
    const unsigned flags[] = {WT_MYNN_FLAG_QC, WT_MYNN_FLAG_QI, WT_MYNN_FLAG_QS,
        WT_MYNN_FLAG_QNC, WT_MYNN_FLAG_QNI, WT_MYNN_FLAG_QNWFA, WT_MYNN_FLAG_QNIFA, WT_MYNN_FLAG_QNBCA};
    const unsigned fields[] = {WT_MYNN_QC, WT_MYNN_QI, WT_MYNN_QS, WT_MYNN_QNC,
        WT_MYNN_QNI, WT_MYNN_QNWFA, WT_MYNN_QNIFA, WT_MYNN_QNBCA};
    unsigned i, repr;
    for (repr = 0; repr < 2; ++repr) for (i = 0; i < 8; ++i) {
        setup(); config.logical_true_bits = repr ? UINT32_MAX : 1;
        present(fields[i]); caps[fields[i]] = 4;
        assert(build() == WT_MYNN_LAYOUT_OK);
        assert(plan.present[fields[i]] && !plan.active[fields[i]]);
        assert(plan.required_bytes[fields[i]] == 4 && !plan.writable[fields[i]]);
        backing[flags[i]][0] = config.logical_true_bits;
        refuse(WT_MYNN_LAYOUT_CAPACITY);
        caps[fields[i]] = 256; assert(build() == WT_MYNN_LAYOUT_OK);
        assert(plan.active[fields[i]] && !plan.writable[fields[i]]);
        absent(fields[i]); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    }
    setup(); absent(WT_MYNN_QV); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    for (i = 0; i < WT_MYNN_NARG; ++i) if (wt_mynn_args[i].type == WT_MYNN_LOGICAL) {
        unsigned j;
        setup(); backing[i][0] = 2; refuse(WT_MYNN_LAYOUT_LOGICAL);
        backing[i][0] = UINT32_MAX; refuse(WT_MYNN_LAYOUT_LOGICAL);
        for (j = 0; j < 8; ++j) present(fields[j]);
        config.logical_true_bits = UINT32_MAX; assert(build() == WT_MYNN_LAYOUT_OK);
        assert(plan.scalar_bits[i] == UINT32_MAX);
    }
}

static void conditional_arrays(void)
{
    unsigned i;
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        unsigned condition = wt_mynn_args[i].access;
        if (condition == WT_MYNN_COND_PRESENT) {
            setup(); present(i); caps[i] = 4; refuse(WT_MYNN_LAYOUT_CAPACITY);
            caps[i] = 256; assert(build() == WT_MYNN_LAYOUT_OK);
            assert(plan.active[i] && plan.writable[i]);
        }
        if (condition == WT_MYNN_COND_PLUME || condition == WT_MYNN_COND_OUTPUT || condition == WT_MYNN_COND_SPP) {
            unsigned option = condition == WT_MYNN_COND_PLUME ? WT_MYNN_BL_MYNN_EDMF :
                condition == WT_MYNN_COND_OUTPUT ? WT_MYNN_BL_MYNN_OUTPUT : WT_MYNN_SPP_PBL;
            unsigned j;
            setup(); present(i); caps[i] = 4;
            assert(build() == WT_MYNN_LAYOUT_OK && plan.required_bytes[i] == 4 && !plan.writable[i]);
            set_i(option, -3);
            assert(build() == WT_MYNN_LAYOUT_OK && !plan.active[i]);
            set_i(option, 1);
            /* Supply other arrays in the same active group to isolate i. */
            for (j = 0; j < WT_MYNN_NARG; ++j) if (wt_mynn_args[j].access == condition && j != i) present(j);
            refuse(WT_MYNN_LAYOUT_CAPACITY);
            present(i); assert(build() == WT_MYNN_LAYOUT_OK);
            assert(plan.active[i]);
            assert(plan.writable[i] == (condition != WT_MYNN_COND_SPP));
            absent(i); refuse(WT_MYNN_LAYOUT_ARGUMENT);
        }
        if (condition == WT_MYNN_COND_BUDGET) {
            setup(); present(i); caps[i] = 4;
            assert(build() == WT_MYNN_LAYOUT_OK && !plan.active[i] && !plan.writable[i]);
            set_i(WT_MYNN_TKE_BUDGET, 1); refuse(WT_MYNN_LAYOUT_BRANCH);
        }
    }
    setup(); set_i(WT_MYNN_INITFLAG, 1);
    absent(WT_MYNN_QC_BL); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); absent(WT_MYNN_QI_BL); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); absent(WT_MYNN_CLDFRA_BL); refuse(WT_MYNN_LAYOUT_ARGUMENT);
    setup(); set_i(WT_MYNN_TKE_BUDGET, -1); refuse(WT_MYNN_LAYOUT_BRANCH);
    setup(); set_i(WT_MYNN_TKE_BUDGET, 2); refuse(WT_MYNN_LAYOUT_BRANCH);
    setup();
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        if (wt_mynn_args[i].optional) present(i);
        if (wt_mynn_args[i].type == WT_MYNN_LOGICAL) backing[i][0] = 1;
    }
    set_i(WT_MYNN_BL_MYNN_EDMF, 1); set_i(WT_MYNN_BL_MYNN_OUTPUT, 1); set_i(WT_MYNN_SPP_PBL, 1);
    assert(build() == WT_MYNN_LAYOUT_OK);
    assert(plan.branches.qc && plan.branches.qnbca && plan.branches.restart);
    assert(plan.branches.bl_mynn_edmf == 1 && plan.branches.bl_mynn_output == 1);
    assert(plan.present[WT_MYNN_DQKE] && !plan.active[WT_MYNN_DQKE]);
}

static void bounds(void)
{
    setup(); set_i(WT_MYNN_KTS, 0); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_KTE, 0); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_KMS, 2); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_ITS, -2); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_JTE, 2); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_ITE, -1); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_KTE, 4); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); caps[WT_MYNN_U] = 255; refuse(WT_MYNN_LAYOUT_CAPACITY);
    setup(); caps[WT_MYNN_PBLH] = 63; refuse(WT_MYNN_LAYOUT_CAPACITY);
    setup(); set_i(WT_MYNN_IMS, INT32_MIN); set_i(WT_MYNN_IME, INT32_MAX);
    set_i(WT_MYNN_JMS, INT32_MIN); set_i(WT_MYNN_JME, INT32_MAX);
    refuse(WT_MYNN_LAYOUT_OVERFLOW);
    setup(); set_i(WT_MYNN_IDE, INT32_MIN); refuse(WT_MYNN_LAYOUT_BOUNDS);
    setup(); set_i(WT_MYNN_KTE, 1); assert(build() == WT_MYNN_LAYOUT_OK);
    if (SIZE_MAX > UINT32_MAX) {
        unsigned i;
        /* Deliberately synthetic interval metadata exercises arithmetic only;
         * no array data are dereferenced by layout, no native call is made. */
        setup();
        set_i(WT_MYNN_IMS, 0); set_i(WT_MYNN_IME, 1073741823);
        set_i(WT_MYNN_JMS, 0); set_i(WT_MYNN_JME, 268435455);
        set_i(WT_MYNN_JTS, 0); set_i(WT_MYNN_JTE, 0);
        set_i(WT_MYNN_KMS, 1); set_i(WT_MYNN_KME, 1); set_i(WT_MYNN_KTE, 1);
        for (i = 0; i < WT_MYNN_NARG; ++i)
            if (args[i] && wt_mynn_args[i].rank) caps[i] = (size_t)UINT64_C(1152921504606846976);
        refuse(WT_MYNN_LAYOUT_OVERFLOW);
    }
}

int main(void)
{
    basic(); early_guards(); species(); conditional_arrays(); bounds();
    puts("MYNN layout checks PASS");
    return 0;
}
