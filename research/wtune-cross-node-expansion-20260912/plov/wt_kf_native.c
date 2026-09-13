#include "wt_kf_native.h"
#include "wt_kf_live.h"
#ifdef WT_KF_NODE_LOCAL
#include "wt_kf_node_live.h"
#include <stdlib.h>
#include <string.h>
#endif
#include <omp.h>
#include <stdatomic.h>

#ifndef WT_GRID_CAPTURE_F77
#define WT_GRID_CAPTURE_F77 wt_grid_capture_
#endif
#ifndef WT_GRID_ID_F77
#define WT_GRID_ID_F77 wt_grid_identity_
#endif
static atomic_int native_configured; /* 0 standalone, 1 configuring, 2 immutable */
static int selected_grid;
/* Plain Fortran external, NOT BIND(C): grid is the native derived-type
 * argument; the C_PTR value and all scalar arguments are passed by reference.
 * This declaration is valid only with the explicitly matched build ABI. */
extern void WT_GRID_CAPTURE_F77(void *grid, void **context,
                               const size_t *max_nodes, int *num_tiles, int *status);
extern void WT_GRID_ID_F77(void *grid, int *id, int *status);

int wt_kf_capture_grid(void *grid, wt_live_spans *registry, size_t max_nodes,
                       int *num_tiles)
{
    void *context = registry;
    int status = WT_SPAN_INVALID, tiles = 0;
    if (num_tiles) *num_tiles = 0;
    if (!grid || !registry || !num_tiles) {
        wt_live_spans_fail(registry);
        return 0;
    }
    int state = atomic_load_explicit(&native_configured, memory_order_acquire);
    if (state == 1) {
        wt_live_spans_fail(registry);
        return 0;
    }
    if (state == 2 && selected_grid > 0) {
        int id = 0;
        WT_GRID_ID_F77(grid, &id, &status);
        if (status != WT_SPAN_OK || id != selected_grid) {
            wt_live_spans_fail(registry);
            return 0;
        }
        status = WT_SPAN_INVALID;
    }
    WT_GRID_CAPTURE_F77(grid, &context, &max_nodes, &tiles, &status);
    if (status != WT_SPAN_OK || tiles <= 0) {
        wt_live_spans_fail(registry);
        return 0;
    }
    *num_tiles = tiles;
    return 1;
}

#ifdef WT_KF_NODE_LOCAL
static int grid_identity(void *grid)
{
    int id=0,status=WT_SPAN_INVALID;
    WT_GRID_ID_F77(grid,&id,&status);
    return status==WT_SPAN_OK && id>0 ? id : 0;
}
#endif

int wt_kf_native_configure(int enabled, int source_abi_verified,
                           int caller_chain_verified, size_t max_spans,
                           size_t max_nodes, size_t byte_budget)
{
    return wt_kf_native_configure_for_grid(enabled, source_abi_verified,
        caller_chain_verified, max_spans, max_nodes, byte_budget, 0);
}

int wt_kf_native_configure_for_grid(int enabled, int source_abi_verified,
                                    int caller_chain_verified, size_t max_spans,
                                    size_t max_nodes, size_t byte_budget,
                                    int grid_selector)
{
    int expected = 0;
    if (grid_selector < 0 || !atomic_compare_exchange_strong(&native_configured, &expected, 1))
        return 0;
    struct wt_kf_live_config config = {
        .enabled = enabled,
        .source_abi_verified = source_abi_verified,
        .max_spans = max_spans,
        .max_nodes = max_nodes,
        .byte_budget = byte_budget,
        .capture = wt_kf_capture_grid,
        .omp_query = omp_in_parallel,
        .caller_chain_verified = caller_chain_verified
    };
    int success;
#ifdef WT_KF_NODE_LOCAL
    const char *node_mode=getenv("WT_KF_NODE_G0");
    if (enabled && node_mode && !strcmp(node_mode,"1"))
        success=wt_kf_node_live_from_env(&config,grid_identity);
    else if (enabled && node_mode && strcmp(node_mode,"0")) success=0;
    else
#endif
        success=wt_kf_live_configure(&config);
    if (!success) {
        atomic_store_explicit(&native_configured, 0, memory_order_release);
        return 0;
    }
    selected_grid = grid_selector;
    atomic_store_explicit(&native_configured, 2, memory_order_release);
    return 1;
}
