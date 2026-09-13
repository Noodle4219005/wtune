#include "wt_sfclay_native.h"
#include "wt_sfclay_live.h"
#include "wt_sfclay_native_query.h"
#ifdef WT_SFCLAY_NODE_LOCAL
#include "wt_sfclay_node_live.h"
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

/* This declaration is trusted only under the frozen-build proof documented in
 * wt_sfclay_native.h.  The C_PTR value and scalar arguments are references. */
extern void WT_GRID_CAPTURE_F77(void *grid, void **context,
    const size_t *max_nodes, int *num_tiles, int *status);
extern void WT_GRID_ID_F77(void *grid, int *grid_id, int *status);

static atomic_int native_state;
static int selected_grid;

static int boolean(int value)
{
    return value == 0 || value == 1;
}

int wt_sfclay_native_state(void)
{
    return atomic_load_explicit(&native_state, memory_order_acquire);
}

int wt_sfclay_native_capture(void *grid, wt_live_spans *registry,
                             size_t max_nodes, int *num_tiles)
{
    void *context = registry;
    int status = WT_SPAN_INVALID;
    int grid_id = 0;
    int tiles = 0;

    if (num_tiles) *num_tiles = 0;
    if (!grid || !registry || !num_tiles || !max_nodes ||
        wt_sfclay_native_state() != WT_SFCLAY_NATIVE_IMMUTABLE) {
        wt_live_spans_fail(registry);
        return 0;
    }

    WT_GRID_ID_F77(grid, &grid_id, &status);
    if (status != WT_SPAN_OK || grid_id <= 0 ||
        (selected_grid > 0 && grid_id != selected_grid)) {
        wt_live_spans_fail(registry);
        return 0;
    }

    status = WT_SPAN_INVALID;
    WT_GRID_CAPTURE_F77(grid, &context, &max_nodes, &tiles, &status);
    if (status != WT_SPAN_OK || tiles <= 0) {
        wt_live_spans_fail(registry);
        return 0;
    }
    *num_tiles = tiles;
    return 1;
}

int wt_sfclay_native_configure(int enabled, int source_abi_verified,
    int caller_chain_verified, int initializer_relocation_verified,
    size_t max_spans, size_t max_nodes, size_t byte_budget,
    int grid_selector)
{
    int expected = WT_SFCLAY_NATIVE_UNCONFIGURED;
    struct wt_sfclay_live_config config;

    if (!boolean(enabled) || !boolean(source_abi_verified) ||
        !boolean(caller_chain_verified) ||
        !boolean(initializer_relocation_verified) || enabled != 1 ||
        source_abi_verified != 1 || caller_chain_verified != 1 ||
        initializer_relocation_verified != 1 || grid_selector < 0 ||
        !max_spans || !max_nodes || !byte_budget)
        return 0;

    if (!atomic_compare_exchange_strong_explicit(
            &native_state, &expected, WT_SFCLAY_NATIVE_INSTALLING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;

    selected_grid = grid_selector;
    config = (struct wt_sfclay_live_config){
        .enabled = 1,
        .source_abi_verified = source_abi_verified,
        .caller_chain_verified = caller_chain_verified,
        .initializer_relocation_verified = initializer_relocation_verified,
        .max_spans = max_spans,
        .max_nodes = max_nodes,
        .byte_budget = byte_budget,
        .capture = wt_sfclay_native_capture,
        .omp_query = omp_in_parallel,
        .native_query = wt_sfclay_native_query,
    };
#ifdef WT_SFCLAY_NODE_LOCAL
    {
        const char *node_g0=getenv("WT_SFCLAY_NODE_G0");
        int installed;
        if (!node_g0 || !strcmp(node_g0,"0"))
            installed=wt_sfclay_live_configure(&config);
        else if (!strcmp(node_g0,"1"))
            installed=wt_sfclay_node_live_from_env(&config);
        else
            installed=0;
        if (!installed) {
#else
    if (!wt_sfclay_live_configure(&config)) {
#endif
        selected_grid = 0;
        atomic_store_explicit(&native_state,
                              WT_SFCLAY_NATIVE_UNCONFIGURED,
                              memory_order_release);
        return 0;
    }
#ifdef WT_SFCLAY_NODE_LOCAL
    }
#endif
    atomic_store_explicit(&native_state, WT_SFCLAY_NATIVE_IMMUTABLE,
                          memory_order_release);
    return 1;
}
