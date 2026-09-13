#include "wt_mynn.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define GUARD_BYTES ((size_t)32)
#define LEFT_GUARD 0xa7
#define RIGHT_GUARD 0x5d

struct region {
    uintptr_t begin, end;
    size_t bytes;
    unsigned char *clone, *snapshot;
    unsigned writable;
};

struct wt_mynn_tx {
    uint64_t owner_nonce;                 /* Immutable; check before mutable state. */
    enum wt_mynn_state state;
    unsigned in_callback;
    size_t count;
    struct wt_mynn_layout layout;
    void *const *owner_table;
    const size_t *capacity_table;
    void *owners[WT_MYNN_NARG], *staged[WT_MYNN_NARG];
    size_t capacities[WT_MYNN_NARG];
    struct region regions[WT_MYNN_NARG];
};

static _Atomic uint64_t next_nonce = 1;
static _Thread_local uint64_t thread_nonce;
static _Thread_local wt_mynn_tx *entered_transaction;

static uint64_t current_nonce(void)
{
    uint64_t next;
    if (thread_nonce) return thread_nonce;
    next = atomic_load_explicit(&next_nonce, memory_order_relaxed);
    for (;;) {
        if (next == UINT64_MAX) return 0;  /* Permanent saturation, never wrap/reuse. */
        if (atomic_compare_exchange_weak_explicit(&next_nonce, &next, next + 1,
                memory_order_relaxed, memory_order_relaxed)) {
            thread_nonce = next;
            return next;
        }
    }
}

static int owned(const wt_mynn_tx *tx)
{
    uint64_t nonce = current_nonce();
    return nonce && tx->owner_nonce == nonce;
}

static enum wt_mynn_status poison(wt_mynn_tx *tx, enum wt_mynn_status status)
{
    if (tx->state != WT_MYNN_COMMITTED) tx->state = WT_MYNN_FAILED;
    return status;
}

static int reentry(void)
{
    if (!entered_transaction) return 0;
    poison(entered_transaction, WT_MYNN_STATE);
    return 1;
}

static int add_size(size_t *total, size_t value)
{
    if (value > SIZE_MAX - *total) return 0;
    *total += value;
    return 1;
}

/* No heap allocation or owner data copy until all regions and the budget pass. */
static enum wt_mynn_status regions_for(void *const *argv, const struct wt_mynn_layout *layout,
    struct region *regions, size_t *count, size_t *total)
{
    size_t n = 0, merged = 0, i, j;
    *total = sizeof(wt_mynn_tx);
    for (i = 0; i < WT_MYNN_NARG; ++i) if (layout->present[i]) {
        struct region r = {0};
        r.begin = (uintptr_t)argv[i];
        if (layout->required_bytes[i] > UINTPTR_MAX - r.begin) return WT_MYNN_OVERFLOW;
        r.end = r.begin + layout->required_bytes[i];
        r.writable = layout->writable[i];
        j = n;
        while (j && regions[j - 1].begin > r.begin) {
            regions[j] = regions[j - 1]; --j;
        }
        regions[j] = r; ++n;
    }
    for (i = 0; i < n; ++i) {
        struct region r = regions[i];
        if (merged && r.begin < regions[merged - 1].end) {
            struct region *previous = &regions[merged - 1];
            if (previous->writable != r.writable) return WT_MYNN_ALIAS;
            if (r.end > previous->end) previous->end = r.end;
        } else regions[merged++] = r;
    }
    for (i = 0; i < merged; ++i) {
        uintptr_t length = regions[i].end - regions[i].begin;
        if (length > SIZE_MAX) return WT_MYNN_OVERFLOW;
        regions[i].bytes = (size_t)length;
        if (!add_size(total, regions[i].bytes) || !add_size(total, regions[i].bytes) ||
            !add_size(total, GUARD_BYTES * 2)) return WT_MYNN_OVERFLOW;
    }
    *count = merged;
    return WT_MYNN_OK;
}

static enum wt_mynn_status validate(const wt_mynn_tx *tx, int pristine)
{
    size_t i, j;
    for (i = 0; i < WT_MYNN_NARG; ++i)
        if (tx->owner_table[i] != tx->owners[i] || tx->capacity_table[i] != tx->capacities[i])
            return WT_MYNN_STALE;
    for (i = 0; i < tx->count; ++i) {
        const struct region *r = &tx->regions[i];
        for (j = 0; j < GUARD_BYTES; ++j)
            if ((r->clone - GUARD_BYTES)[j] != LEFT_GUARD || r->clone[r->bytes + j] != RIGHT_GUARD)
                return WT_MYNN_GUARD;
        if ((pristine || !r->writable) && memcmp(r->clone, r->snapshot, r->bytes))
            return WT_MYNN_READONLY;
        if (memcmp((const void *)r->begin, r->snapshot, r->bytes)) return WT_MYNN_STALE;
    }
    return WT_MYNN_OK;
}

enum wt_mynn_status wt_mynn_prepare(void *const argv[WT_MYNN_NARG],
    const size_t capacities[WT_MYNN_NARG], const struct wt_mynn_layout_config *config,
    size_t total_budget, wt_mynn_tx **out)
{
    struct wt_mynn_layout layout;
    struct region regions[WT_MYNN_NARG];
    void *owners[WT_MYNN_NARG];
    size_t saved_capacities[WT_MYNN_NARG];
    size_t count, total, i, j;
    uint64_t nonce;
    wt_mynn_tx *tx;
    unsigned char *cursor;
    enum wt_mynn_status status;
    if (reentry()) return WT_MYNN_STATE;
    if (out) *out = NULL;
    if (!out) return WT_MYNN_ARGUMENT;
    if (!total_budget) return WT_MYNN_ARGUMENT;
    if (wt_mynn_layout_build(argv, capacities, config, &layout) != WT_MYNN_LAYOUT_OK)
        return WT_MYNN_LAYOUT;
    memcpy(owners, argv, sizeof owners);
    memcpy(saved_capacities, capacities, sizeof saved_capacities);
    status = regions_for(owners, &layout, regions, &count, &total);
    if (status != WT_MYNN_OK) return status;
    if (total > total_budget) return WT_MYNN_BUDGET;
    nonce = current_nonce();
    if (!nonce) return WT_MYNN_STATE;
    tx = malloc(total);
    if (!tx) return WT_MYNN_ALLOCATION;
    memset(tx, 0, sizeof *tx);
    tx->owner_nonce = nonce;
    tx->state = WT_MYNN_PREPARED;
    tx->count = count;
    tx->layout = layout;
    tx->owner_table = argv;
    tx->capacity_table = capacities;
    memcpy(tx->owners, owners, sizeof owners);
    memcpy(tx->capacities, saved_capacities, sizeof saved_capacities);
    memcpy(tx->regions, regions, count * sizeof *regions);
    cursor = (unsigned char *)(tx + 1);
    for (i = 0; i < count; ++i) {
        struct region *r = &tx->regions[i];
        memset(cursor, LEFT_GUARD, GUARD_BYTES); cursor += GUARD_BYTES;
        r->clone = cursor; cursor += r->bytes;
        memset(cursor, RIGHT_GUARD, GUARD_BYTES); cursor += GUARD_BYTES;
        r->snapshot = cursor; cursor += r->bytes;
        memcpy(r->snapshot, (const void *)r->begin, r->bytes);
        memcpy(r->clone, r->snapshot, r->bytes);
    }
    for (i = 0; i < WT_MYNN_NARG; ++i) if (layout.present[i]) {
        uintptr_t address = (uintptr_t)owners[i];
        for (j = 0; j < count; ++j) {
            const struct region *r = &tx->regions[j];
            if (address >= r->begin && address < r->end) {
                tx->staged[i] = r->clone + (size_t)(address - r->begin);
                break;
            }
        }
        if (!tx->staged[i] || (!wt_mynn_args[i].rank &&
            memcmp(tx->staged[i], &layout.scalar_bits[i], 4))) {
            free(tx);
            return WT_MYNN_STALE;
        }
    }
    status = validate(tx, 1);
    if (status != WT_MYNN_OK) { free(tx); return status; }
    *out = tx;
    return WT_MYNN_OK;
}

enum wt_mynn_status wt_mynn_execute_once(wt_mynn_tx *tx, void (*real)(void **))
{
    void *temporary[WT_MYNN_NARG];
    enum wt_mynn_status status;
    size_t i;
    if (reentry()) return WT_MYNN_STATE;
    if (!tx) return WT_MYNN_ARGUMENT;
    if (!owned(tx)) return WT_MYNN_STATE;
    if (tx->state != WT_MYNN_PREPARED || tx->in_callback) return poison(tx, WT_MYNN_STATE);
    if (!real) return poison(tx, WT_MYNN_ARGUMENT);
    status = validate(tx, 1);
    if (status != WT_MYNN_OK) return poison(tx, status);
    memcpy(temporary, tx->staged, sizeof temporary);
    tx->state = WT_MYNN_ENTERED;
    tx->in_callback = 1;
    entered_transaction = tx;
    real(temporary);
    entered_transaction = NULL;
    tx->in_callback = 0;
    if (tx->state != WT_MYNN_ENTERED) return poison(tx, WT_MYNN_STATE);
    for (i = 0; i < WT_MYNN_NARG; ++i)
        if (temporary[i] != tx->staged[i]) return poison(tx, WT_MYNN_REBIND);
    status = validate(tx, 0);
    if (status != WT_MYNN_OK) return poison(tx, status);
    tx->state = WT_MYNN_EXECUTED;
    return WT_MYNN_OK;
}

enum wt_mynn_status wt_mynn_commit(wt_mynn_tx *tx)
{
    enum wt_mynn_status status;
    size_t i;
    if (reentry()) return WT_MYNN_STATE;
    if (!tx) return WT_MYNN_ARGUMENT;
    if (!owned(tx)) return WT_MYNN_STATE;
    if (tx->state != WT_MYNN_EXECUTED || tx->in_callback) return poison(tx, WT_MYNN_STATE);
    /* No owner is written unless EVERY region and original argument table passes. */
    status = validate(tx, 0);
    if (status != WT_MYNN_OK) return poison(tx, status);
    for (i = 0; i < tx->count; ++i) {
        const struct region *r = &tx->regions[i];
        if (r->writable) memcpy((void *)r->begin, r->clone, r->bytes);
    }
    tx->state = WT_MYNN_COMMITTED;
    return WT_MYNN_OK;
}

enum wt_mynn_status wt_mynn_destroy(wt_mynn_tx *tx)
{
    if (reentry()) return WT_MYNN_STATE;
    if (!tx) return WT_MYNN_ARGUMENT;
    if (!owned(tx)) return WT_MYNN_STATE;
    if (tx->in_callback) return poison(tx, WT_MYNN_STATE);
    /* tx is the private allocation base, never a callback-visible argv pointer. */
    free(tx);
    return WT_MYNN_OK;
}

enum wt_mynn_state wt_mynn_get_state(const wt_mynn_tx *tx)
{
    if (!tx || !owned(tx)) return WT_MYNN_FAILED;
    return tx->state;
}
