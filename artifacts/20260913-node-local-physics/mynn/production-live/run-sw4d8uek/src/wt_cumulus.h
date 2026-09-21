#ifndef WT_CUMULUS_H
#define WT_CUMULUS_H
#include <stddef.h>
#include "wt_kf_cps_schema.h"

/* Local payload prototype only: it is not connected to WRF or redistribution.
 * Capacities and configuration assertions are a caller trust boundary.  It
 * assumes 4-byte int32/float/logical ABI values.  `in_parallel` is only a
 * caller assertion: callers must exclusively own all arguments; a future WRF
 * wrapper must check actual omp_in_parallel.  No concurrent atomicity exists. */
enum wt_kf_status { WT_KF_OK=0, WT_KF_DISABLED, WT_KF_BAD_ARGUMENT,
  WT_KF_CAPACITY, WT_KF_OVERLAP, WT_KF_BUDGET, WT_KF_NOMEM, WT_KF_BAD_STATE,
  WT_KF_STALE, WT_KF_GUARD };
/* byte_budget covers requested heap storage (metadata, snapshots, clones and
 * guards), excluding allocator overhead and stack. Zero initialization disables
 * staging. The caller keeps exclusive access to every owner buffer until destroy.
 * When snapshotted KF_EDRATES is not one, UDR_KF, DDR_KF, UER_KF, DER_KF and
 * TIMEC_KF each require only independent four-byte backing. They are guarded
 * immutable placeholders and are never copied back to their owners. */
struct wt_kf_config { int enabled, num_tiles, in_parallel; size_t byte_budget; };
struct wt_kf_txn;
/* Return zero only after the intended operation completes. The callback must
 * neither rebind argument pointers nor retain them beyond the transaction.
 * Native CPS invocation and initialization remain the adapter's responsibility. */
typedef int (*wt_kf_callback)(void **cloned_args, void *context);
enum wt_kf_status wt_kf_prepare(void *argv[WT_KF_NARG], const size_t capacity_bytes[WT_KF_NARG],
  const struct wt_kf_config *config, struct wt_kf_txn **out);
enum wt_kf_status wt_kf_execute(struct wt_kf_txn *, wt_kf_callback, void *);
enum wt_kf_status wt_kf_commit(struct wt_kf_txn *);
void wt_kf_destroy(struct wt_kf_txn *);
/* Callback/commit validation failure is terminal. Destroy remains required.
 * A destroy attempted inside the callback marks failure and defers freeing;
 * the owning caller must destroy again after execute returns. */
#endif
