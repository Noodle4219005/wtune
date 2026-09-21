#ifndef WT_KF_NODE_H
#define WT_KF_NODE_H
#include "wt_kf_batch.h"
#include "wt_node_local.h"
#define WT_KF_NODE_PHASE UINT64_C(0x4b46)
/* profile certifies the frozen CPS object/ABI/caller chain. init_generation must
 * identify verified identical initialized KF LUT bytes on every admitted rank,
 * not merely count independent local initialization calls. Both are nonzero.
 * signature branch=1 means trigger1 fixedDT. Context is process/thread local. */
struct wt_kf_node_context {
    struct wt_nl_service *service;
    struct wt_nl_signature signature;
    uint32_t rank, thread;
    size_t byte_budget;
    wt_kf_callback native;
    void *native_context;
    /* Owner validation only. Set after the service reports a different-rank
     * executor; it runs a private reference from the immutable request. */
    int shadow_remote;
};
/* Zero initialize before first publish. An active call cannot be republished. */
struct wt_kf_node_call {
    struct wt_kf_batch *batch;
    struct wt_nl_ticket ticket;
    struct wt_nl_descriptor descriptor;
    int32_t geometry[18];
    uint32_t owner_rank, owner_thread;
    int published, committed;
    size_t shadow_bytes;
    int shadow_checked, shadow_failed;
};
enum wt_nl_status wt_kf_node_publish(struct wt_kf_node_context *,
    void *args[WT_KF_NARG], const size_t capacity[WT_KF_NARG],
    uint64_t call_sequence, struct wt_kf_node_call *, enum wt_kf_status *batch_status);
/* One bounded scan: executes only another local rank's compatible CPS request. */
enum wt_nl_status wt_kf_node_progress(struct wt_kf_node_context *,
    uint64_t executor_tag, struct wt_nl_ticket *executed);
/* Explicit fallback, never called implicitly by publish/collect/progress. */
enum wt_nl_status wt_kf_node_owner_execute(struct wt_kf_node_context *,
    struct wt_kf_node_call *, uint64_t executor_tag);
enum wt_nl_status wt_kf_node_collect(struct wt_kf_node_context *, struct wt_kf_node_call *);
enum wt_nl_status wt_kf_node_finish(struct wt_kf_node_context *, struct wt_kf_node_call *);
/* Destroy is local memory cleanup only. A published uncommitted slot must be
 * poisoned explicitly through service before cleanup; never replay that call. */
void wt_kf_node_destroy(struct wt_kf_node_call *);
#endif
