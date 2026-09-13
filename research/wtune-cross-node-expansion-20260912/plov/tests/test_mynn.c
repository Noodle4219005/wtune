#include "wt_mynn.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t);
void __real_free(void *);
void *__real_memcpy(void *, const void *, size_t);
static size_t malloc_calls, free_calls, last_size;
static void *last_allocation;
static int fail_allocation;
static uint32_t *change_at_allocation;
void *__wrap_malloc(size_t n)
{
    ++malloc_calls; last_size = n;
    if (fail_allocation) { fail_allocation = 0; return NULL; }
    if (change_at_allocation) { *change_at_allocation = 1; change_at_allocation = NULL; }
    last_allocation = __real_malloc(n);
    return last_allocation;
}
void __wrap_free(void *p)
{
    if (p) { assert(p == last_allocation); ++free_calls; }
    __real_free(p);
}

static uint32_t data[WT_MYNN_NARG][96];
static void *args[WT_MYNN_NARG];
static size_t capacities[WT_MYNN_NARG];
static struct wt_mynn_layout_config config;
static wt_mynn_tx *transaction;
static unsigned calls;
static int mode;
static unsigned char *retained;
static unsigned char *retained_readonly;
static size_t offset;
static int count_writes;
static size_t owner_writes, alias_owner_writes;
#define BUDGET ((size_t)1048576)

void *__wrap_memcpy(void *to, const void *from, size_t bytes)
{
    uintptr_t p = (uintptr_t)to;
    if (count_writes && p >= (uintptr_t)data && p < (uintptr_t)data + sizeof data) {
        ++owner_writes;
        if (to == args[WT_MYNN_RUBLTEN]) ++alias_owner_writes;
    }
    return __real_memcpy(to, from, bytes);
}

static enum wt_mynn_status counted_commit(void)
{
    enum wt_mynn_status result;
    owner_writes = alias_owner_writes = 0; count_writes = 1;
    result = wt_mynn_commit(transaction);
    count_writes = 0;
    return result;
}

static void put(unsigned i, int32_t value) { memcpy(data[i], &value, 4); }
static void setup(void)
{
    unsigned i, j;
    memset(data, 0, sizeof data);
    for (i = 0; i < WT_MYNN_NARG; ++i) {
        args[i] = data[i]; capacities[i] = wt_mynn_args[i].rank ? sizeof data[i] : 4;
        if (wt_mynn_args[i].rank) for (j = 0; j < 96; ++j) data[i][j] = i * 1000 + j;
        if (wt_mynn_args[i].optional) { args[i] = NULL; capacities[i] = 0; }
    }
    config = (struct wt_mynn_layout_config){1, 1, 1, 0};
    put(WT_MYNN_IDS, 0); put(WT_MYNN_IDE, 2); put(WT_MYNN_JDS, -1); put(WT_MYNN_JDE, 1);
    put(WT_MYNN_KDS, 1); put(WT_MYNN_KDE, 4);
    put(WT_MYNN_IMS, -1); put(WT_MYNN_IME, 2); put(WT_MYNN_JMS, -2); put(WT_MYNN_JME, 1);
    put(WT_MYNN_KMS, 0); put(WT_MYNN_KME, 3);
    put(WT_MYNN_ITS, 0); put(WT_MYNN_ITE, 1); put(WT_MYNN_JTS, -1); put(WT_MYNN_JTE, 0);
    put(WT_MYNN_KTS, 1); put(WT_MYNN_KTE, 3);
    args[WT_MYNN_QV] = data[WT_MYNN_QV]; capacities[WT_MYNN_QV] = sizeof data[0];
    args[WT_MYNN_QC_BL] = data[WT_MYNN_QC_BL]; capacities[WT_MYNN_QC_BL] = sizeof data[0];
    args[WT_MYNN_QI_BL] = data[WT_MYNN_QI_BL]; capacities[WT_MYNN_QI_BL] = sizeof data[0];
    args[WT_MYNN_CLDFRA_BL] = data[WT_MYNN_CLDFRA_BL]; capacities[WT_MYNN_CLDFRA_BL] = sizeof data[0];
    transaction = NULL; calls = 0; mode = 0; retained = NULL; offset = 0;
}

static enum wt_mynn_status prepare(size_t budget)
{ return wt_mynn_prepare(args, capacities, &config, budget, &transaction); }

static void callback(void **v)
{
    uint32_t *ru = v[WT_MYNN_RUBLTEN];
    ++calls;
    assert(v != args && v[WT_MYNN_IDS] != args[WT_MYNN_IDS]);
    assert(*(int32_t *)v[WT_MYNN_KTS] == 1);
    assert(wt_mynn_get_state(transaction) == WT_MYNN_ENTERED);
    retained = (unsigned char *)ru;
    retained_readonly = v[WT_MYNN_U];
    if (mode == 1) {
        assert((unsigned char *)v[WT_MYNN_RVBLTEN] == (unsigned char *)ru + offset);
        ru[offset / 4] = 778;
        assert(*(uint32_t *)v[WT_MYNN_RVBLTEN] == 778);
    }
    if (mode == 2) {
        assert((unsigned char *)v[WT_MYNN_V] == (unsigned char *)v[WT_MYNN_U] + offset);
    }
    ru[21] = 777; /* One tile element; other full-view bytes must be preserved. */
    if (mode == 3) ((uint32_t *)v[WT_MYNN_U])[0] ^= 1;
    if (mode == 4) ((unsigned char *)ru)[-1] ^= 1;
    if (mode == 5) ((unsigned char *)ru)[256] ^= 1;
    if (mode == 6) v[WT_MYNN_RUBLTEN] = (void *)1;
    if (mode == 7) data[WT_MYNN_U][0] ^= 1;
    if (mode == 8) assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE);
    if (mode == 9) assert(wt_mynn_commit(transaction) == WT_MYNN_STATE);
    if (mode == 10) assert(wt_mynn_destroy(transaction) == WT_MYNN_STATE);
    if (mode == 11) ((uint32_t *)v[WT_MYNN_DQKE])[0] ^= 1;
    if (mode == 12) ((uint32_t *)v[WT_MYNN_IDS])[0] ^= 1;
    if (mode == 13) v[WT_MYNN_DQKE] = ru; /* Rebinding an absent argument also fails. */
    if (mode == 14) {
        wt_mynn_tx *before_handle = transaction;
        size_t before = malloc_calls;
        volatile uintptr_t bad = 1;
        assert(wt_mynn_prepare((void *const *)bad, (size_t *)bad, &config, BUDGET, &transaction) == WT_MYNN_STATE);
        assert(transaction == before_handle && malloc_calls == before);
        assert(wt_mynn_get_state(transaction) == WT_MYNN_FAILED);
    }
}

static void preparation_refusals(void)
{
    size_t before, exact;
    volatile uintptr_t bad = 1;
    setup(); before = malloc_calls;
    config.abi_verified = 0; transaction = (wt_mynn_tx *)1;
    assert(wt_mynn_prepare((void *const *)bad, (size_t *)bad, &config, BUDGET, &transaction) == WT_MYNN_LAYOUT);
    assert(!transaction && malloc_calls == before);
    setup(); args[WT_MYNN_IDS] = (void *)1; capacities[WT_MYNN_IDS] = 0;
    assert(prepare(BUDGET) == WT_MYNN_LAYOUT && !transaction && malloc_calls == before);
    if (SIZE_MAX > UINT32_MAX) {
        unsigned i, n = 1;
        size_t huge = (size_t)UINT64_C(288230376151711744);
        /* Synthetic non-overlapping intervals: force two-copy budget overflow
         * before allocation/copy; array backing is intentionally never read. */
        setup(); put(WT_MYNN_IMS, 0); put(WT_MYNN_IME, 268435455);
        put(WT_MYNN_JMS, 0); put(WT_MYNN_JME, 268435455);
        put(WT_MYNN_JTS, 0); put(WT_MYNN_JTE, 0);
        put(WT_MYNN_KMS, 1); put(WT_MYNN_KME, 1); put(WT_MYNN_KTE, 1);
        for (i = 0; i < WT_MYNN_NARG; ++i) if (args[i] && wt_mynn_args[i].rank) {
            args[i] = (void *)(uintptr_t)(huge * n++); capacities[i] = huge;
        }
        assert(prepare(SIZE_MAX) == WT_MYNN_OVERFLOW && !transaction && malloc_calls == before);
    }
    setup(); put(WT_MYNN_KTS, 0);
    assert(prepare(BUDGET) == WT_MYNN_LAYOUT && !transaction && malloc_calls == before);
    setup(); assert(prepare(0) == WT_MYNN_ARGUMENT && !transaction);
    assert(prepare(1) == WT_MYNN_BUDGET && !transaction && malloc_calls == before);
    {
        struct wt_mynn_layout layout;
        assert(wt_mynn_layout_build(args, capacities, &config, &layout) == WT_MYNN_LAYOUT_OK);
        assert(prepare(layout.total_bytes) == WT_MYNN_BUDGET && !transaction);
    }
    assert(prepare(BUDGET) == WT_MYNN_OK); exact = last_size;
    assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    before = malloc_calls; transaction = NULL;
    assert(prepare(exact - 1) == WT_MYNN_BUDGET && !transaction && malloc_calls == before);
    assert(prepare(exact) == WT_MYNN_OK); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); before = malloc_calls; fail_allocation = 1;
    assert(prepare(BUDGET) == WT_MYNN_ALLOCATION && !transaction && malloc_calls == before + 1);
    setup(); change_at_allocation = data[WT_MYNN_TKE_BUDGET];
    assert(prepare(BUDGET) == WT_MYNN_STALE && !transaction);
    setup(); args[WT_MYNN_RVBLTEN] = args[WT_MYNN_U]; before = malloc_calls;
    assert(prepare(BUDGET) == WT_MYNN_ALIAS && !transaction && malloc_calls == before);
    setup(); args[WT_MYNN_DQKE] = args[WT_MYNN_RUBLTEN]; capacities[WT_MYNN_DQKE] = 4;
    assert(prepare(BUDGET) == WT_MYNN_ALIAS && !transaction && malloc_calls == before);
    setup(); args[WT_MYNN_U] = (void *)(UINTPTR_MAX - 3); capacities[WT_MYNN_U] = 256;
    assert(prepare(BUDGET) == WT_MYNN_LAYOUT && !transaction && malloc_calls == before);
}

static void successful_aliases(void)
{
    uint32_t original[96];
    size_t separate_size, same_size;
    unsigned i;
    setup(); assert(prepare(BUDGET) == WT_MYNN_OK); separate_size = last_size;
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK && calls == 1);
    assert(data[WT_MYNN_RUBLTEN][21] != 777);
    assert(counted_commit() == WT_MYNN_OK && alias_owner_writes == 1);
    assert(wt_mynn_get_state(transaction) == WT_MYNN_COMMITTED);
    assert(wt_mynn_commit(transaction) == WT_MYNN_STATE);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE && calls == 1);
    assert(wt_mynn_get_state(transaction) == WT_MYNN_COMMITTED);
    assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); args[WT_MYNN_RVBLTEN] = args[WT_MYNN_RUBLTEN]; mode = 1;
    assert(prepare(BUDGET) == WT_MYNN_OK); same_size = last_size;
    assert(same_size < separate_size);
    memcpy(original, data[WT_MYNN_RUBLTEN], sizeof original);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK && calls == 1);
    assert(counted_commit() == WT_MYNN_OK && alias_owner_writes == 1);
    for (i = 0; i < 96; ++i)
        assert(data[WT_MYNN_RUBLTEN][i] == (i == 0 ? 778 : i == 21 ? 777 : original[i]));
    assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); args[WT_MYNN_RVBLTEN] = &data[WT_MYNN_RUBLTEN][16];
    capacities[WT_MYNN_RVBLTEN] = 320; mode = 1; offset = 64;
    assert(prepare(BUDGET) == WT_MYNN_OK); assert(last_size > same_size && last_size < separate_size);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK);
    assert(counted_commit() == WT_MYNN_OK && alias_owner_writes == 1 && data[WT_MYNN_RUBLTEN][16] == 778);
    assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); args[WT_MYNN_V] = &data[WT_MYNN_U][16]; capacities[WT_MYNN_V] = 320; mode = 2; offset = 64;
    assert(prepare(BUDGET) == WT_MYNN_OK);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK);
    assert(wt_mynn_commit(transaction) == WT_MYNN_OK); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    /* Adjacent 4-byte read-only and writable plane views must NOT merge. */
    setup(); args[WT_MYNN_IDS] = data[WT_MYNN_PBLH]; put(WT_MYNN_PBLH, 0);
    args[WT_MYNN_PBLH] = &data[WT_MYNN_PBLH][1]; capacities[WT_MYNN_PBLH] = 380;
    assert(prepare(BUDGET) == WT_MYNN_OK); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
}

static void state_and_corruption(void)
{
    int m;
    const enum wt_mynn_status statuses[] = {WT_MYNN_READONLY, WT_MYNN_GUARD, WT_MYNN_GUARD,
        WT_MYNN_REBIND, WT_MYNN_STALE, WT_MYNN_STATE, WT_MYNN_STATE, WT_MYNN_STATE,
        WT_MYNN_READONLY, WT_MYNN_READONLY, WT_MYNN_REBIND, WT_MYNN_STATE};
    for (m = 3; m <= 14; ++m) {
        uint32_t original;
        size_t freed;
        setup(); mode = m;
        if (m == 11) { args[WT_MYNN_DQKE] = data[WT_MYNN_DQKE]; capacities[WT_MYNN_DQKE] = 4; }
        original = data[WT_MYNN_RUBLTEN][21];
        assert(prepare(BUDGET) == WT_MYNN_OK); freed = free_calls;
        assert(wt_mynn_execute_once(transaction, callback) == statuses[m - 3]);
        assert(calls == 1 && data[WT_MYNN_RUBLTEN][21] == original && free_calls == freed);
        assert(wt_mynn_get_state(transaction) == WT_MYNN_FAILED);
        assert(wt_mynn_commit(transaction) == WT_MYNN_STATE);
        assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE && calls == 1);
        assert(wt_mynn_destroy(transaction) == WT_MYNN_OK && free_calls == freed + 1);
    }
    setup(); assert(prepare(BUDGET) == WT_MYNN_OK);
    assert(wt_mynn_commit(transaction) == WT_MYNN_STATE && calls == 0);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE && calls == 0);
    assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); assert(prepare(BUDGET) == WT_MYNN_OK);
    assert(wt_mynn_execute_once(transaction, NULL) == WT_MYNN_ARGUMENT);
    assert(wt_mynn_get_state(transaction) == WT_MYNN_FAILED); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    setup(); assert(prepare(BUDGET) == WT_MYNN_OK);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE && calls == 1);
    assert(wt_mynn_commit(transaction) == WT_MYNN_STATE); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
}

static void stale_and_atomic_commit(void)
{
    unsigned variant;
    for (variant = 0; variant < 3; ++variant) {
        setup(); assert(prepare(BUDGET) == WT_MYNN_OK);
        if (variant == 0) data[WT_MYNN_U][0] ^= 1;
        if (variant == 1) args[WT_MYNN_U] = data[WT_MYNN_V];
        if (variant == 2) --capacities[WT_MYNN_U];
        assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STALE && calls == 0);
        assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    }
    for (variant = 0; variant < 4; ++variant) {
        uint32_t original;
        setup(); original = data[WT_MYNN_RUBLTEN][21]; assert(prepare(BUDGET) == WT_MYNN_OK);
        assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK && calls == 1);
        if (variant == 0) data[WT_MYNN_BL_MYNN_ESS][0] ^= 1;
        if (variant == 1) retained[-1] ^= 1;
        if (variant == 2) retained[256] ^= 1;
        if (variant == 3) retained_readonly[0] ^= 1;
        assert(counted_commit() == (variant == 0 ? WT_MYNN_STALE : variant == 3 ? WT_MYNN_READONLY : WT_MYNN_GUARD));
        assert(owner_writes == 0);
        assert(data[WT_MYNN_RUBLTEN][21] == original);
        assert(wt_mynn_get_state(transaction) == WT_MYNN_FAILED);
        assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    }
    setup(); args[WT_MYNN_QC] = data[WT_MYNN_QC]; capacities[WT_MYNN_QC] = 4;
    assert(prepare(BUDGET) == WT_MYNN_OK); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    put(WT_MYNN_FLAG_QC, 1); assert(prepare(BUDGET) == WT_MYNN_LAYOUT && !transaction);
}

static void *foreign_thread(void *unused)
{
    (void)unused;
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_STATE);
    assert(wt_mynn_commit(transaction) == WT_MYNN_STATE);
    assert(wt_mynn_destroy(transaction) == WT_MYNN_STATE);
    return NULL;
}

int main(void)
{
    pthread_t thread;
    unsigned i;
    preparation_refusals(); successful_aliases(); state_and_corruption(); stale_and_atomic_commit();
    setup(); assert(prepare(BUDGET) == WT_MYNN_OK);
    for (i = 0; i < 4; ++i) {
        assert(pthread_create(&thread, NULL, foreign_thread, NULL) == 0);
        assert(pthread_join(thread, NULL) == 0);
    }
    assert(wt_mynn_get_state(transaction) == WT_MYNN_PREPARED && calls == 0);
    assert(wt_mynn_execute_once(transaction, callback) == WT_MYNN_OK);
    assert(wt_mynn_commit(transaction) == WT_MYNN_OK); assert(wt_mynn_destroy(transaction) == WT_MYNN_OK);
    puts("MYNN transaction checks PASS");
    return 0;
}
