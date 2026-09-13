#ifndef WT_MYNN_NODE_H
#define WT_MYNN_NODE_H

#include "wt_mynn_batch.h"
#include "wt_node_local.h"

#define WT_MYNN_NODE_PHASE UINT64_C(0x4d594e4e)
/* signature binds the independently reviewed endpoint/profile, geometry epoch,
 * admitted branch and identical initialized read-only MYNN state.  profile and
 * init_generation identify a reviewed closure fingerprint and the matching
 * initialized static-state generation; neither is a locally invented counter
 * or inferred from source hashes. branch is WT_MYNN_BATCH_SCHEMA. */
struct wt_mynn_node_context {
    struct wt_nl_service *service;
    struct wt_nl_signature signature;
    uint32_t rank, thread;
    size_t byte_budget;
    wt_mynn_batch_callback native;
    void *native_context;
};
struct wt_mynn_node_call {
    struct wt_mynn_batch *batch;
    struct wt_nl_ticket ticket;
    struct wt_nl_descriptor descriptor;
    int32_t geometry[18];
    uint32_t owner_rank, owner_thread;
    int published, committed;
    /* Set only after the owner observes DONE from a different rank. */
    int shadow_remote, shadow_checked, shadow_failed;
    size_t shadow_bytes;
};

enum wt_nl_status wt_mynn_node_publish(struct wt_mynn_node_context *,
    void *argv[WT_MYNN_NARG], const size_t capacities[WT_MYNN_NARG],
    const struct wt_mynn_layout_config *, uint64_t call_sequence,
    struct wt_mynn_node_call *, enum wt_mynn_status *batch_status);
enum wt_nl_status wt_mynn_node_progress(struct wt_mynn_node_context *,
    uint64_t executor_tag, struct wt_nl_ticket *executed);
enum wt_nl_status wt_mynn_node_owner_execute(struct wt_mynn_node_context *,
    struct wt_mynn_node_call *, uint64_t executor_tag);
enum wt_nl_status wt_mynn_node_collect(struct wt_mynn_node_context *,
    struct wt_mynn_node_call *);
enum wt_nl_status wt_mynn_node_finish(struct wt_mynn_node_context *,
    struct wt_mynn_node_call *);
void wt_mynn_node_destroy(struct wt_mynn_node_call *);

#endif
