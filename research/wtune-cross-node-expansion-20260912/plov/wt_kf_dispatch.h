#ifndef WT_KF_DISPATCH_H
#define WT_KF_DISPATCH_H
#include <stddef.h>
#include <stdatomic.h>
#include "wt_cumulus.h"

enum wt_kf_dispatch_reason { WT_KF_DISPATCH_DISABLED, WT_KF_DISPATCH_ABI,
  WT_KF_DISPATCH_OMP, WT_KF_DISPATCH_CAPACITY, WT_KF_DISPATCH_PREPARE,
  WT_KF_DISPATCH_REENTRY, WT_KF_DISPATCH_FATAL, WT_KF_DISPATCH_REASON_COUNT };
struct wt_kf_dispatch_counters { _Atomic unsigned long requested, staged, fatal;
  _Atomic unsigned long fallback[WT_KF_DISPATCH_REASON_COUNT]; };
typedef int (*wt_kf_capacity_provider)(void *opaque, unsigned index, void *arg, size_t *bytes);
typedef int (*wt_kf_omp_query)(void *opaque);
typedef void (*wt_kf_fatal_handler)(void *opaque, enum wt_kf_dispatch_reason reason);
typedef void (*wt_kf_native_invoke)(void **argv);
struct wt_kf_dispatch_context {
  int enabled, abi_verified; _Atomic int busy;
  struct wt_kf_config transaction;
  void *opaque;
  wt_kf_capacity_provider capacity;
  wt_kf_omp_query omp_query;
  wt_kf_fatal_handler fatal;
  struct wt_kf_dispatch_counters counters;
};
/* argv and all owner allocations remain live and exclusively owned until return.
 * Configuration and callback-owned metadata must remain immutable during calls;
 * capacity/OMP callbacks can run before the staging busy guard, so the caller
 * must serialize them or make their read-only state safe for concurrent queries.
 * The busy guard detects overlapping staging; it does not lock owner memory.
 * Callbacks must not retain argv or reenter this dispatcher. A native invocation
 * followed by failed validation is fatal, never a second direct invocation. */
void wt_kf_dispatch(struct wt_kf_dispatch_context *, void **argv, size_t nargs,
                    wt_kf_native_invoke native_invoke);
/* Defaults disabled. Configuration is accepted at most once, while quiescent;
 * pointed-to opaque state must outlive every subsequent bridge invocation. */
int wt_kf_bridge_configure(const struct wt_kf_dispatch_context *);
void wt_kf_bridge_dispatch(void **argv, void (*invoke)(void **), size_t nargs);
/* Default real-runtime query; applications provide this hook when ctx omits omp_query. */
int wt_kf_omp_in_parallel(void);
#endif
