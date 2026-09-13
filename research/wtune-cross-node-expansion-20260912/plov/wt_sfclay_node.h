#ifndef WT_SFCLAY_NODE_H
#define WT_SFCLAY_NODE_H

#include "wt_node_local.h"
#include "wt_sfclay_batch.h"

#define WT_SFCLAY_NODE_PHASE UINT64_C(0x5346434c4159)
/* Frozen module_sf_sfclay is SFCLAYSCHEME=91. Scheme 1 is the distinct
 * module_sf_sfclayrev ABI; scheme 5 is MYNNSFCSCHEME. */
#define WT_SFCLAY_NODE_BRANCH UINT64_C(91)

/* profile certifies the frozen SFCLAY object/ABI/caller chain.
 * init_generation must be the fingerprint of the initialized PSIMTB/PSIHTB
 * bytes, not a per-process initialization counter. */
struct wt_sfclay_node_context {
    struct wt_nl_service *service;
    struct wt_nl_signature signature;
    uint32_t rank, thread;
    size_t byte_budget;
    wt_sfclay_batch_callback native;
    void *native_context;
    /* Owner validation only. Set after the service reports a different-rank
     * executor; it runs a private reference from the immutable request. */
    int shadow_remote;
};

struct wt_sfclay_node_call {
    struct wt_sfclay_batch *batch;
    struct wt_nl_ticket ticket;
    struct wt_nl_descriptor descriptor;
    int32_t geometry[18];
    uint32_t owner_rank, owner_thread;
    const void *bound_thread;
    int published, committed;
    size_t shadow_bytes;
    int shadow_checked, shadow_failed;
};

/* Zero initialize call before its first publish. A live call cannot be
 * republished, and all owner operations use the publishing actual thread. */
enum wt_nl_status wt_sfclay_node_publish(
    struct wt_sfclay_node_context *, void *args[WT_SFCLAY_NARG],
    const size_t capacity[WT_SFCLAY_NARG], uint64_t call_sequence,
    struct wt_sfclay_node_call *, enum wt_sfclay_batch_status *);
/* One bounded scan. The generic service excludes same-rank slots. */
enum wt_nl_status wt_sfclay_node_progress(struct wt_sfclay_node_context *,
    uint64_t executor_tag, struct wt_nl_ticket *executed);
enum wt_nl_status wt_sfclay_node_owner_execute(
    struct wt_sfclay_node_context *, struct wt_sfclay_node_call *,
    uint64_t executor_tag);
enum wt_nl_status wt_sfclay_node_collect(struct wt_sfclay_node_context *,
                                         struct wt_sfclay_node_call *);
enum wt_nl_status wt_sfclay_node_finish(struct wt_sfclay_node_context *,
                                        struct wt_sfclay_node_call *);
/* Local cleanup only, and a no-op from a thread other than the publisher.
 * Poison a published, uncommitted service slot first. */
void wt_sfclay_node_destroy(struct wt_sfclay_node_call *);

#endif
