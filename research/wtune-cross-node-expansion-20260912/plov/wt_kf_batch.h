#ifndef WT_KF_BATCH_H
#define WT_KF_BATCH_H
#include "wt_cumulus.h"
#include <stdint.h>
/* One original CPS tile, unchanged full-i loop geometry. Caller guarantees live
 * capacities, frozen read inputs, and exclusive ownership of this tile's output
 * cells until commit/destroy. Distinct threads may own disjoint rectangles of
 * the same arrays. Module LUT equality is a node-adapter admission obligation.
 * Only trigger=1, fixed DT, KTS=1; SHALL may be absent, other optionals required.
 * No payload contains a process-private pointer. */
#define WT_KF_BATCH_SCHEMA UINT64_C(0x4b46435053373902)
struct wt_kf_batch;
enum wt_kf_status wt_kf_batch_prepare(void *args[WT_KF_NARG],
    const size_t capacity[WT_KF_NARG], size_t byte_budget,
    struct wt_kf_batch **out);
const void *wt_kf_batch_request(const struct wt_kf_batch *, size_t *bytes);
/* Independent helper budget includes all reconstructed arrays and guards.
 * invoked is set immediately before callback entry, including failure paths.
 * Failed invocation must never be replayed. Request/result buffers disjoint. */
enum wt_kf_status wt_kf_batch_execute(const void *request, size_t bytes,
    void *result, size_t result_capacity, size_t byte_budget,
    wt_kf_callback callback, void *context, int *invoked);
/* validate performs every check without owner writes. apply is permitted once,
 * immediately following successful validate under the caller's ownership.
 * prepare/request/validate/apply are bound to the same actual thread; input and
 * owner metadata must remain frozen across validate/apply. */
enum wt_kf_status wt_kf_batch_validate(struct wt_kf_batch *,
    const void *result, size_t bytes);
enum wt_kf_status wt_kf_batch_apply(struct wt_kf_batch *);
enum wt_kf_status wt_kf_batch_commit(struct wt_kf_batch *,
    const void *result, size_t bytes);
void wt_kf_batch_destroy(struct wt_kf_batch *);
#endif
