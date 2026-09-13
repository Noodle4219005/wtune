#include "wt_cumulus.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GUARD_BYTES 16u

enum transaction_state { PREPARED, EXECUTING, EXECUTED, COMMITTED, FAILED };

struct wt_kf_txn {
    void *owner[WT_KF_NARG];
    unsigned char *base[WT_KF_NARG]; /* Allocation provenance stays private. */
    unsigned char *snapshot[WT_KF_NARG];
    size_t bytes[WT_KF_NARG];
    int32_t kf_edrates;
    enum transaction_state state;
    int in_callback;
};

_Static_assert(sizeof(int32_t) == 4, "requires 4-byte INTEGER/LOGICAL ABI");
_Static_assert(sizeof(float) == 4, "requires 4-byte REAL ABI");

static int checked_add(size_t a, size_t b, size_t *result)
{
    if (b > SIZE_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int checked_mul(size_t a, size_t b, size_t *result)
{
    if (a && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

static int interval_end(const void *pointer, size_t bytes, uintptr_t *end)
{
    uintptr_t start = (uintptr_t)pointer;
    if (bytes > UINTPTR_MAX - start) return 0;
    *end = start + bytes;
    return 1;
}

static int extent(int32_t lower, int32_t upper, size_t *size)
{
    int64_t value = (int64_t)upper - lower + 1;
    if (value <= 0 || (uint64_t)value > SIZE_MAX) return 0;
    *size = (size_t)value;
    return 1;
}

/* Call only after four-byte capacity and interval validation. */
static int32_t integer_arg(void *const arguments[], unsigned index)
{
    int32_t value;
    memcpy(&value, arguments[index], sizeof(value));
    return value;
}

static enum wt_kf_status geometry(void *const arguments[], int32_t kf_edrates,
                                  size_t bytes[])
{
    int32_t bounds[18];
    size_t ni, nj, nk, plane, volume;
    unsigned i;

    for (i = 0; i < 18; ++i) bounds[i] = integer_arg(arguments, i);
    if (bounds[12] < bounds[6] || bounds[13] > bounds[7] ||
        bounds[12] > bounds[13] || bounds[14] < bounds[8] ||
        bounds[15] > bounds[9] || bounds[14] > bounds[15] ||
        bounds[16] < bounds[10] || bounds[17] > bounds[11] ||
        bounds[16] > bounds[17]) return WT_KF_BAD_ARGUMENT;

    /* The native leaf reads P0(1); W0AVG also reads W(:,KTE+1,:). */
    if (bounds[16] != 1 || (int64_t)bounds[17] + 1 > bounds[11])
        return WT_KF_BAD_ARGUMENT;
    if (!extent(bounds[6], bounds[7], &ni) ||
        !extent(bounds[8], bounds[9], &nj) ||
        !extent(bounds[10], bounds[11], &nk) ||
        !checked_mul(ni, nj, &plane) ||
        !checked_mul(plane, nk, &volume) ||
        !checked_mul(plane, 4, &plane) ||
        !checked_mul(volume, 4, &volume)) return WT_KF_BAD_ARGUMENT;

    for (i = 0; i < WT_KF_NARG; ++i) {
        bytes[i] = wt_kf_shape[i] == WT_KF_3D ? volume :
                   wt_kf_shape[i] == WT_KF_2D ? plane : 4;
    }
    /* Audited KF_eta_CPS source touches these only under KF_EDRATES==1. */
    if (kf_edrates != 1) {
        bytes[WT_KF_UDR_KF] = bytes[WT_KF_DDR_KF] = 4;
        bytes[WT_KF_UER_KF] = bytes[WT_KF_DER_KF] = 4;
        bytes[WT_KF_TIMEC_KF] = 4;
    }
    return WT_KF_OK;
}

static enum wt_kf_status validate(void *const arguments[], const size_t capacities[],
                                 const struct wt_kf_config *config, size_t bytes[],
                                 int32_t *kf_edrates)
{
    uintptr_t ends[WT_KF_NARG];
    size_t total = sizeof(struct wt_kf_txn), clone_size, field_size;
    unsigned i, j;
    enum wt_kf_status status;

    if (!arguments || !capacities || !config || !config->enabled)
        return WT_KF_DISABLED;
    if (config->num_tiles != 1 || config->in_parallel) return WT_KF_BAD_ARGUMENT;

    /* No scalar dereference precedes this pass, including argument zero.
     * Actual allocation validity remains the caller's explicit contract. */
    for (i = 0; i < WT_KF_NARG; ++i) {
        if (!arguments[i]) {
            if (i == WT_KF_SHALL) continue;
            return i < 19 ? WT_KF_CAPACITY : WT_KF_BAD_ARGUMENT;
        }
        if (capacities[i] < 4) return WT_KF_CAPACITY;
        if (!interval_end(arguments[i], 4, &ends[i])) return WT_KF_BAD_ARGUMENT;
    }
    if (integer_arg(arguments, WT_KF_TRIGGER) != 1 ||
        integer_arg(arguments, WT_KF_ADAPT_STEP_FLAG) != 0) return WT_KF_BAD_ARGUMENT;
    *kf_edrates = integer_arg(arguments, WT_KF_KF_EDRATES);
    status = geometry(arguments, *kf_edrates, bytes);
    if (status != WT_KF_OK) return status;

    for (i = 0; i < WT_KF_NARG; ++i) {
        if (!arguments[i]) continue;
        if (capacities[i] < bytes[i]) return WT_KF_CAPACITY;
        if (!interval_end(arguments[i], bytes[i], &ends[i])) return WT_KF_BAD_ARGUMENT;
        if (!checked_add(bytes[i], 2 * GUARD_BYTES, &clone_size) ||
            !checked_add(clone_size, bytes[i], &field_size) ||
            !checked_add(total, field_size, &total)) return WT_KF_BUDGET;
        for (j = 0; j < i; ++j) {
            if (arguments[j] && (uintptr_t)arguments[i] < ends[j] &&
                (uintptr_t)arguments[j] < ends[i]) return WT_KF_OVERLAP;
        }
    }
    return total > config->byte_budget ? WT_KF_BUDGET : WT_KF_OK;
}

static unsigned char *cloned_arg(const struct wt_kf_txn *transaction, unsigned i)
{
    return transaction->base[i] ? transaction->base[i] + GUARD_BYTES : NULL;
}

static int inactive_arg(const struct wt_kf_txn *transaction, unsigned i)
{
    return transaction->kf_edrates != 1 &&
           (i == WT_KF_UDR_KF || i == WT_KF_DDR_KF ||
            i == WT_KF_UER_KF || i == WT_KF_DER_KF ||
            i == WT_KF_TIMEC_KF);
}

static enum wt_kf_status fail(struct wt_kf_txn *transaction, enum wt_kf_status status)
{
    transaction->state = FAILED;
    return status;
}

enum wt_kf_status wt_kf_prepare(void *arguments[WT_KF_NARG],
                              const size_t capacities[WT_KF_NARG],
                              const struct wt_kf_config *config, struct wt_kf_txn **out)
{
    struct wt_kf_txn *transaction;
    size_t bytes[WT_KF_NARG], allocation_size;
    enum wt_kf_status status;
    int32_t kf_edrates;
    unsigned i;

    if (!out) return WT_KF_BAD_ARGUMENT;
    *out = NULL;
    status = validate(arguments, capacities, config, bytes, &kf_edrates);
    if (status != WT_KF_OK) return status;
    transaction = calloc(1, sizeof(*transaction));
    if (!transaction) return WT_KF_NOMEM;
    transaction->state = PREPARED;
    transaction->kf_edrates = kf_edrates;
    for (i = 0; i < WT_KF_NARG; ++i) {
        unsigned char *clone;
        if (!arguments[i]) continue;
        if (!checked_add(bytes[i], 2 * GUARD_BYTES, &allocation_size)) {
            wt_kf_destroy(transaction);
            return WT_KF_BUDGET;
        }
        transaction->base[i] = malloc(allocation_size);
        transaction->snapshot[i] = malloc(bytes[i]);
        if (!transaction->base[i] || !transaction->snapshot[i]) {
            wt_kf_destroy(transaction);
            return WT_KF_NOMEM;
        }
        transaction->owner[i] = arguments[i];
        transaction->bytes[i] = bytes[i];
        clone = cloned_arg(transaction, i);
        memset(transaction->base[i], 0xa5, GUARD_BYTES);
        memset(clone + bytes[i], 0x5a, GUARD_BYTES);
        if (i == WT_KF_KF_EDRATES) {
            memcpy(clone, &transaction->kf_edrates, bytes[i]);
            memcpy(transaction->snapshot[i], &transaction->kf_edrates, bytes[i]);
        } else {
            memcpy(clone, arguments[i], bytes[i]);
            memcpy(transaction->snapshot[i], arguments[i], bytes[i]);
        }
    }
    *out = transaction;
    return WT_KF_OK;
}

enum wt_kf_status wt_kf_execute(struct wt_kf_txn *transaction,
                              wt_kf_callback callback, void *context)
{
    void *arguments[WT_KF_NARG];
    int result;
    unsigned i;

    if (!transaction) return WT_KF_BAD_STATE;
    if (transaction->in_callback) return fail(transaction, WT_KF_BAD_STATE);
    if (transaction->state != PREPARED) return WT_KF_BAD_STATE;
    if (!callback) return fail(transaction, WT_KF_BAD_STATE);
    for (i = 0; i < WT_KF_NARG; ++i) arguments[i] = cloned_arg(transaction, i);
    transaction->state = EXECUTING;
    transaction->in_callback = 1;
    result = callback(arguments, context);
    transaction->in_callback = 0;
    if (result != 0 || transaction->state != EXECUTING)
        return fail(transaction, WT_KF_BAD_STATE);
    for (i = 0; i < WT_KF_NARG; ++i) {
        if (arguments[i] != cloned_arg(transaction, i))
            return fail(transaction, WT_KF_BAD_STATE);
    }
    transaction->state = EXECUTED;
    return WT_KF_OK;
}

enum wt_kf_status wt_kf_commit(struct wt_kf_txn *transaction)
{
    unsigned i, k;

    if (!transaction) return WT_KF_BAD_STATE;
    if (transaction->in_callback) return fail(transaction, WT_KF_BAD_STATE);
    if (transaction->state != EXECUTED) {
        if (transaction->state == PREPARED) transaction->state = FAILED;
        return WT_KF_BAD_STATE;
    }
    /* All validation precedes all owner writes. This requires exclusive owner
     * access for the entire transaction; it is not a concurrent commit API. */
    for (i = 0; i < WT_KF_NARG; ++i) {
        unsigned char *clone;
        if (!transaction->owner[i]) continue;
        clone = cloned_arg(transaction, i);
        for (k = 0; k < GUARD_BYTES; ++k) {
            if (transaction->base[i][k] != 0xa5 ||
                clone[transaction->bytes[i] + k] != 0x5a)
                return fail(transaction, WT_KF_GUARD);
        }
        if (memcmp(transaction->owner[i], transaction->snapshot[i], transaction->bytes[i]) ||
            ((wt_kf_intent[i] == WT_KF_IN || inactive_arg(transaction, i)) &&
             memcmp(clone, transaction->snapshot[i], transaction->bytes[i])))
            return fail(transaction, WT_KF_STALE);
    }
    for (i = 0; i < WT_KF_NARG; ++i) {
        if (transaction->owner[i] && wt_kf_intent[i] != WT_KF_IN &&
            !inactive_arg(transaction, i))
            memcpy(transaction->owner[i], cloned_arg(transaction, i), transaction->bytes[i]);
    }
    transaction->state = COMMITTED;
    return WT_KF_OK;
}

void wt_kf_destroy(struct wt_kf_txn *transaction)
{
    unsigned i;
    if (!transaction) return;
    /* A reentrant destroy cannot release storage still used by execute. */
    if (transaction->in_callback) {
        transaction->state = FAILED;
        return;
    }
    for (i = 0; i < WT_KF_NARG; ++i) {
        free(transaction->base[i]);
        free(transaction->snapshot[i]);
    }
    free(transaction);
}
