#ifndef WT_MYNN_NODE_LIVE_H
#define WT_MYNN_NODE_LIVE_H
#include "wt_mynn_node.h"
#include "wt_live_spans.h"
#include <stdio.h>

struct wt_mynn_node_live_config {
    int enabled, abi_verified, caller_verified, static_state_verified;
    int concurrency_verified, grid_selector, owner_only, shadow_verify_remote;
    uint32_t logical_true_bits, threads, rank;
    uint64_t profile, deadline_ns;
    size_t byte_budget, max_spans, max_nodes, slot_bytes, shared_budget;
    struct wt_nl_service *service;
    int (*capture)(void *, wt_live_spans *, size_t, int *);
    int (*scope)(void *, void *, uint64_t *, uint64_t *);
    uint64_t (*fingerprint)(void);
};
enum wt_mynn_live_refusal {
    WT_MLN_CONFIG, WT_MLN_ABI, WT_MLN_STATIC, WT_MLN_SCOPE, WT_MLN_MAPPER,
    WT_MLN_OMP, WT_MLN_CAPACITY, WT_MLN_GEOMETRY, WT_MLN_INIT,
    WT_MLN_SINGLE_ROW, WT_MLN_ALIAS, WT_MLN_BUDGET, WT_MLN_SLOT,
    WT_MLN_PREPARE, WT_MLN_REASON_COUNT
};
struct wt_mynn_node_live_counters {
    uint64_t requested, donated, committed, self_executed, helper_executed;
    uint64_t remote_committed, fallback, fatal, pack_ns, native_ns, progress_ns;
    uint64_t shadow_checked, shadow_failed, shadow_bytes;
    uint64_t refusal[WT_MLN_REASON_COUNT];
};
/* Configure only during serialized startup; no native proof is inferred. */
int wt_mynn_node_live_configure(const struct wt_mynn_node_live_config *);
/* Shared first-RK bridge owns serialized begin/end around all joined OMP work.
 * token is a local lifetime guard, never the scientific epoch/signature. */
uint64_t wt_mynn_first_rk_begin(void *grid, void *cfg);
void wt_mynn_first_rk_end(uint64_t token);
void wt_mynn_bridge_dispatch(void **argv, void (*invoke)(void **), size_t nargs);
void wt_mynn_node_live_get_counters(struct wt_mynn_node_live_counters *);
void wt_mynn_node_live_dump(FILE *);
void wt_mynn_node_live_close(void);
#endif
