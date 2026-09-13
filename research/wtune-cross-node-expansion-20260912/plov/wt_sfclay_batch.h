#ifndef WT_SFCLAY_BATCH_H
#define WT_SFCLAY_BATCH_H

#include "wt_sfclay.h"
#include <stdint.h>

/* One original SFCLAY J row with the original full-I bounds and alignment.
 * The owner freezes this row's active I cells and all scalars through apply.
 * Disjoint J rows may be owned concurrently. Optional controls 81..83 may be
 * absent; EM_CORE arrays 76..80 are required. */
#define WT_SFCLAY_BATCH_SCHEMA UINT64_C(0x5346434c41590101)

struct wt_sfclay_batch;
/* Callback returns zero after successful native execution. Any nonzero return
 * is post-entry fatal; pre-entry rejection occurs only inside batch_execute. */
typedef int (*wt_sfclay_batch_callback)(void *args[WT_SFCLAY_NARG],
                                        void *context);

enum wt_sfclay_batch_status {
    WT_SFCLAY_BATCH_OK = 0,
    WT_SFCLAY_BATCH_BAD_ARGUMENT,
    WT_SFCLAY_BATCH_BOUNDS,
    WT_SFCLAY_BATCH_CAPACITY,
    WT_SFCLAY_BATCH_BUDGET,
    WT_SFCLAY_BATCH_OVERFLOW,
    WT_SFCLAY_BATCH_OVERLAP,
    WT_SFCLAY_BATCH_NOMEM,
    WT_SFCLAY_BATCH_STATE,
    WT_SFCLAY_BATCH_STALE,
    WT_SFCLAY_BATCH_GUARD
};

enum wt_sfclay_batch_status wt_sfclay_batch_prepare(
    void *args[WT_SFCLAY_NARG], const size_t capacity[WT_SFCLAY_NARG],
    size_t byte_budget, struct wt_sfclay_batch **out);
const void *wt_sfclay_batch_request(const struct wt_sfclay_batch *,
                                    size_t *bytes);
/* invoked becomes one immediately before callback entry. Any error after that
 * is fatal for this invocation and must not be replayed. */
enum wt_sfclay_batch_status wt_sfclay_batch_execute(
    const void *request, size_t bytes, void *result, size_t result_capacity,
    size_t byte_budget, wt_sfclay_batch_callback callback, void *context,
    int *invoked);
/* validate performs every stale/guard/length check without owner writes.
 * apply is owner-thread-only and writes only the active I cells of this row. */
enum wt_sfclay_batch_status wt_sfclay_batch_validate(
    struct wt_sfclay_batch *, const void *result, size_t bytes);
enum wt_sfclay_batch_status wt_sfclay_batch_apply(struct wt_sfclay_batch *);
enum wt_sfclay_batch_status wt_sfclay_batch_commit(
    struct wt_sfclay_batch *, const void *result, size_t bytes);
void wt_sfclay_batch_destroy(struct wt_sfclay_batch *);

#endif
