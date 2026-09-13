#ifndef WT_SFCLAY_H
#define WT_SFCLAY_H

#include <stddef.h>
#include "wt_sfclay_schema.h"

enum wt_sfclay_status {
    WT_SFCLAY_OK = 0, WT_SFCLAY_BAD_ARGUMENT, WT_SFCLAY_BOUNDS,
    WT_SFCLAY_CAPACITY, WT_SFCLAY_BUDGET, WT_SFCLAY_OVERFLOW,
    WT_SFCLAY_OVERLAP, WT_SFCLAY_STATE, WT_SFCLAY_STALE
};

struct wt_sfclay_tx;
typedef struct wt_sfclay_tx wt_sfclay_tx;

/* The caller supplies the frozen 84-address ABI, validated live capacities,
 * verified ABI, and exclusive owner access.  Prepare makes no owner writes.
 * Invoke the native routine exactly once on cloned_argv, then call commit once.
 * Any post-invocation failure returns an error without fallback or owner writes. */
int wt_sfclay_prepare(void *const argv[WT_SFCLAY_NARG],
                      const size_t capacity[WT_SFCLAY_NARG],
                      size_t byte_budget, wt_sfclay_tx **out);
void **wt_sfclay_cloned_argv(wt_sfclay_tx *tx);
int wt_sfclay_commit(wt_sfclay_tx *tx);
void wt_sfclay_destroy(wt_sfclay_tx *tx);

#endif
