#ifndef WT_SFCLAY_NODE_LIVE_H
#define WT_SFCLAY_NODE_LIVE_H

#include "wt_sfclay_live.h"
#include "wt_sfclay_node.h"
#include <stdint.h>
#include <stdio.h>

/* Configure once at serialized startup. A supplied service is owned by the
 * fixture/application; NULL selects the shared phase-aware node mapper.
 * WT_SFCLAY_BUILD_CONCURRENT_VERIFIED must be a frozen-build proof for OMP2,
 * and wt_sfclay_node_epoch.f90 must use that build's domain ABI. */
struct wt_sfclay_node_live_options {
    struct wt_nl_service *service;
    uint32_t rank, threads;
    uint64_t profile, deadline_ns;
    int concurrency_verified, shadow_verify_remote, owner_only;
    size_t slot_bytes, shared_budget;
};

enum wt_sfclay_node_refusal {
    WT_SFN_CONFIG, WT_SFN_ABI, WT_SFN_INIT, WT_SFN_SCOPE, WT_SFN_MAPPER,
    WT_SFN_OMP, WT_SFN_CAPACITY, WT_SFN_GEOMETRY, WT_SFN_SINGLE_ROW,
    WT_SFN_ALIAS, WT_SFN_BUDGET, WT_SFN_SLOT, WT_SFN_PREPARE,
    WT_SFN_NATIVE_QUERY, WT_SFN_REASON_COUNT
};

struct wt_sfclay_node_live_counters {
    uint64_t requested, donated, committed, self_executed, helper_executed;
    uint64_t fallback, fatal, pack_ns, native_ns, progress_ns;
    uint64_t owner_remote_committed, refusal[WT_SFN_REASON_COUNT];
    uint64_t shadow_checked, shadow_failed, shadow_bytes;
};

int wt_sfclay_node_live_configure(const struct wt_sfclay_live_config *,
    const struct wt_sfclay_node_live_options *);
int wt_sfclay_node_live_from_env(const struct wt_sfclay_live_config *);
int wt_sfclay_node_live_active(void);
uint64_t wt_sfclay_node_live_first_rk_begin(void *grid, void *cfg);
void wt_sfclay_node_live_first_rk_end(uint64_t token);
uint64_t wt_sfclay_node_live_surface_begin(void);
void wt_sfclay_node_live_surface_end(uint64_t token);
void wt_sfclay_node_live_dispatch(void **argv, void (*invoke)(void **),
                                  size_t nargs);
void wt_sfclay_node_live_get_counters(struct wt_sfclay_node_live_counters *);
void wt_sfclay_node_live_dump(FILE *);
void wt_sfclay_node_live_close(void);

/* Exact hash of PSIMTB/PSIHTB after the real wrapped initializer. Zero means
 * no admitted initializer has completed. The caller must hold a reader lease
 * across every fallback/native/helper/commit use of the tables. */
uint64_t wt_sfclay_init_fingerprint(void);

#endif
