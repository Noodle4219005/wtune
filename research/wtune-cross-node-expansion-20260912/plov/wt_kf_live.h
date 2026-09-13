#ifndef WT_KF_LIVE_H
#define WT_KF_LIVE_H
#include <stddef.h>
#include "wt_live_spans.h"

enum wt_kf_live_reason {
    WT_KF_LIVE_DISABLED, WT_KF_LIVE_SOURCE_ABI, WT_KF_LIVE_NO_SCOPE,
    WT_KF_LIVE_OMP, WT_KF_LIVE_CAPACITY, WT_KF_LIVE_TILES,
    WT_KF_LIVE_ARGUMENT_ABI, WT_KF_LIVE_PREPARE, WT_KF_LIVE_REASON_COUNT
};
struct wt_kf_live_config {
    int enabled, source_abi_verified;
    size_t max_spans, max_nodes, byte_budget;
    /* Return 1 only after the complete bounded live list was captured. Set
     * num_tiles from the current grid. Do not begin/finish/end the registry.
     * The core has no native Fortran ABI dependency; the adapter owns that. */
    int (*capture)(void *grid, wt_live_spans *, size_t max_nodes, int *num_tiles);
    /* Actual runtime query: only exactly zero authorizes staging. */
    int (*omp_query)(void);
    /* Independently attested caller objects, ABI, and all CPS relocations.
     * Default zero; neither an environment flag nor a CPS-only source hash
     * establishes this prerequisite for automatic-actual shape capacities. */
    int caller_chain_verified;
};
struct wt_kf_live_counters {
    unsigned long requested, staged, fallback;
    unsigned long reason[WT_KF_LIVE_REASON_COUNT];
};
/* Configure once while quiescent, before any outer invocation. The copied
 * configuration is immutable. Registry lifetime is the process lifetime.
 * Outer begin/end must be serialized and nonrecursive; all dispatch calls
 * must finish before end (after the native outer OpenMP barrier). No freeing,
 * reallocating, or rebinding captured storage during that scope. Violations
 * detected during an invocation abort; no transaction is replayed.
 * Read-only registry publication is process-shared, not thread-local.
 * Counter snapshots are individually atomic, not a coherent concurrent sum. */
int wt_kf_live_configure(const struct wt_kf_live_config *);
void wt_kf_live_get_counters(struct wt_kf_live_counters *);
void wt_cumulus_grid_begin(void *grid);
void wt_cumulus_grid_end(void);
/* Redirect only the generated bridge translation unit to this entry point. */
void wt_kf_live_dispatch(void **argv, void (*invoke)(void **), size_t nargs);
#endif
