#ifndef WT_SFCLAY_LIVE_H
#define WT_SFCLAY_LIVE_H

#include <stddef.h>
#include <stdint.h>
#include "wt_live_spans.h"
#include "wt_sfclay_native_query.h"

enum wt_sfclay_live_reason {
  WT_SF_LIVE_DISABLED, WT_SF_LIVE_SOURCE_ABI, WT_SF_LIVE_NO_SCOPE,
  WT_SF_LIVE_OMP, WT_SF_LIVE_NATIVE_QUERY, WT_SF_LIVE_TILES,
  WT_SF_LIVE_ARGUMENT_ABI, WT_SF_LIVE_CAPACITY, WT_SF_LIVE_PREPARE,
  WT_SF_LIVE_INIT, WT_SF_LIVE_REASON_COUNT
};
struct wt_sfclay_live_config {
  int enabled, source_abi_verified, caller_chain_verified;
  int initializer_relocation_verified;
  size_t max_spans, max_nodes, byte_budget;
  int (*capture)(void *, wt_live_spans *, size_t, int *);
  int (*omp_query)(void);
  int (*native_query)(void *, void *, struct wt_sfclay_native_result *);
};
struct wt_sfclay_live_counters {
  unsigned long requested, staged, fallback, fatal;
  unsigned long reason[WT_SF_LIVE_REASON_COUNT];
};
/* One immutable installation at process quiescence; the calling pthread becomes
 * the only staging owner. All proof flags are strictly boolean. Initialize may
 * precede configure: its state is owned by the independent writer/readers lease.
 * Compile AND link both live.c and init_lease.c with -pthread.
 *
 * Begin/end tokens pair even for disabled or direct-surface scopes. Cancellation
 * stays disabled until every scope/dispatch/lease on this thread has unwound.
 * Signals, longjmp and external asynchronous cancellation across these scopes
 * are unsupported; no native timeout, worker-pool or remote-execution guarantee.
 */
int wt_sfclay_live_configure(const struct wt_sfclay_live_config *);
uint64_t wt_sfclay_first_rk_begin(void *grid, void *cfg);
void wt_sfclay_first_rk_end(uint64_t token);
uint64_t wt_sfclay_surface_begin(void);
void wt_sfclay_surface_end(uint64_t token);
void wt_sfclay_live_dispatch(void **argv, void (*real)(void **), size_t nargs);
void wt_sfclay_init_dispatch(int *allowed_to_read, void (*real)(int *));
void wt_sfclay_live_get_counters(struct wt_sfclay_live_counters *);
#endif
