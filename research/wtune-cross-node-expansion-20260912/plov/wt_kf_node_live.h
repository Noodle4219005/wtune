#ifndef WT_KF_NODE_LIVE_H
#define WT_KF_NODE_LIVE_H
#include "wt_kf_live.h"
#include "wt_kf_node.h"
#include <stdint.h>
#include <stdio.h>
/* Configure once at startup. A supplied service is fixture/application-owned;
 * NULL selects the strict Slurm shm mapper. Proof/profile are build evidence. */
struct wt_kf_node_live_options {
    struct wt_nl_service *service;
    uint32_t rank, threads;
    uint64_t profile, deadline_ns;
    int concurrency_verified, shadow_verify_remote, owner_only;
    size_t slot_bytes, shared_budget;
    int (*grid_identity)(void *grid);
};
enum wt_kf_node_refusal {
    WT_KFN_CONFIG, WT_KFN_ABI, WT_KFN_INIT, WT_KFN_SCOPE, WT_KFN_MAPPER,
    WT_KFN_OMP, WT_KFN_CAPACITY, WT_KFN_GEOMETRY, WT_KFN_SINGLE_ROW,
    WT_KFN_ALIAS, WT_KFN_BUDGET, WT_KFN_SLOT, WT_KFN_PREPARE, WT_KFN_REASON_COUNT
};
struct wt_kf_node_live_counters {
    uint64_t requested, donated, committed, self_executed, helper_executed;
    uint64_t fallback, fatal, pack_ns, native_ns, progress_ns;
    uint64_t owner_remote_committed, refusal[WT_KFN_REASON_COUNT];
    uint64_t shadow_checked, shadow_failed, shadow_bytes;
};
int wt_kf_node_live_configure(const struct wt_kf_live_config *,
    const struct wt_kf_node_live_options *);
int wt_kf_node_live_from_env(const struct wt_kf_live_config *, int (*grid_identity)(void *));
int wt_kf_node_live_active(void);
void wt_kf_node_live_begin(void *grid);
void wt_kf_node_live_end(void);
void wt_kf_node_live_dispatch(void **, void (*invoke)(void **), size_t nargs);
void wt_kf_node_live_get_counters(struct wt_kf_node_live_counters *);
void wt_kf_node_live_dump(FILE *);
void wt_kf_node_live_close(void);
/* Initialization bridge lease. A writer encountering any live reader is fatal;
 * no native init may race fallback, direct head, helper, or commit. Reader hash
 * zero means initialization/table attestation is unknown and cannot stage. */
uint64_t wt_kf_init_reader_enter(void);
void wt_kf_init_reader_leave(void);
uint64_t wt_kf_init_fingerprint(void);
#endif
