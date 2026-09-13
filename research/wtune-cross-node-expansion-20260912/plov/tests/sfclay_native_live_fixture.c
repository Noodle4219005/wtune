#include "wt_sfclay_live.h"
#include "wt_sfclay_native.h"
#include "wt_sfclay_native_query.h"
#include "wt_sfclay_schema.h"

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK %d: %s\n", __LINE__, #expr); \
        exit(80); \
    } \
} while (0)

struct grid_fixture {
    int id;
    int tiles;
    void *args[WT_SFCLAY_NARG];
    size_t bytes[WT_SFCLAY_NARG];
};

static struct grid_fixture grid;
static const char *mode;
static int identity_calls, capture_calls, query_calls, native_calls, init_calls;
static int init_logical = 37;

void wt_grid_identity_(void *opaque, int *grid_id, int *status);
void wt_grid_capture_(void *opaque, void **context, const size_t *max_nodes,
                      int *num_tiles, int *status);
void wt_sfclay_native_query_(void *opaque, void *config, int *grid_id,
    int *actual_scheme, int *actual_fractional, int *expected_scheme,
    int *status);

void wt_grid_identity_(void *opaque, int *grid_id, int *status)
{
    struct grid_fixture *current = opaque;
    ++identity_calls;
    *grid_id = current->id;
    *status = WT_SPAN_OK;
}

void wt_grid_capture_(void *opaque, void **context, const size_t *max_nodes,
                      int *num_tiles, int *status)
{
    struct grid_fixture *current = opaque;
    ++capture_calls;
    CHECK(current == &grid && context && *context && *max_nodes == 33);
    for (unsigned i = 0; i < WT_SFCLAY_NARG; ++i) {
        int rank;
        if (wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_SCALAR) continue;
        rank = wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_3D ? 3 : 2;
        if (wt_live_spans_add(*context, current->args[i], current->bytes[i],
                              WT_SPAN_REAL, sizeof(float), rank) != WT_SPAN_OK) {
            *status = WT_SPAN_INVALID;
            *num_tiles = 0;
            return;
        }
    }
    *num_tiles = current->tiles;
    *status = WT_SPAN_OK;
}

void wt_sfclay_native_query_(void *opaque, void *config, int *grid_id,
    int *actual_scheme, int *actual_fractional, int *expected_scheme,
    int *status)
{
    struct grid_fixture *current = opaque;
    ++query_calls;
    CHECK(current == &grid && config == (void *)(uintptr_t)0x1234U);
    *grid_id = current->id;
    *actual_scheme = 7;
    *actual_fractional = 0;
    *expected_scheme = 7;
    *status = current->id > 0 ? WT_SFCLAY_NATIVE_OK :
                               WT_SFCLAY_NATIVE_GRID_ID;
}

static void initialize(int *logical)
{
    ++init_calls;
    CHECK(logical == &init_logical && *logical == 37);
}

static void native(void **args)
{
    ++native_calls;
    if (strcmp(mode, "selected-staged")) {
        CHECK(args == (void **)1);
        return;
    }
    CHECK(args && args != grid.args);
    for (unsigned i = 0; i < WT_SFCLAY_NARG; ++i) {
        CHECK(args[i] && args[i] != grid.args[i]);
        if (wt_sfclay_schema_access[i] == WT_SFCLAY_WRITABLE) {
            CHECK(grid.bytes[i] >= 2U * sizeof(float));
            ((float *)args[i])[1] = (float)(1000U + i);
        }
    }
}

static void make_arguments(void)
{
    for (unsigned i = 0; i < WT_SFCLAY_NARG; ++i) {
        grid.bytes[i] = wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_3D ?
                        48U : wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_2D ?
                        24U : 4U;
        grid.args[i] = calloc(1, grid.bytes[i]);
        CHECK(grid.args[i]);
        for (size_t j = 0; j < grid.bytes[i] / sizeof(float); ++j)
            ((float *)grid.args[i])[j] = (float)(i + 1U);
    }
    for (unsigned i = 58; i < 76; ++i)
        *(int *)grid.args[i] = i % 2U == 0U ? 1 :
                              (i == 63U || i == 69U || i == 75U ? 2 : 3);
    *(int *)grid.args[61] = 2;
    *(int *)grid.args[67] = 2;
    *(int *)grid.args[70] = 2;
    *(int *)grid.args[71] = 2;
    *(int *)grid.args[72] = 1;
    *(int *)grid.args[73] = 1;
    grid.tiles = 1;
}

static void free_arguments(void)
{
    for (unsigned i = 0; i < WT_SFCLAY_NARG; ++i) free(grid.args[i]);
}

static void configured_scope(void **args, int in_parallel)
{
    uint64_t outer = wt_sfclay_first_rk_begin(
        &grid, (void *)(uintptr_t)0x1234U);
    uint64_t surface = wt_sfclay_surface_begin();
    if (in_parallel) {
#pragma omp parallel num_threads(2)
        {
#pragma omp masked
            wt_sfclay_live_dispatch(args, native, WT_SFCLAY_NARG);
        }
    } else {
        wt_sfclay_live_dispatch(args, native, WT_SFCLAY_NARG);
    }
    wt_sfclay_surface_end(surface);
    wt_sfclay_first_rk_end(outer);
}

static void check_counters(unsigned long requested, unsigned long staged,
                           unsigned long fallback)
{
    struct wt_sfclay_live_counters counters;
    wt_sfclay_live_get_counters(&counters);
    CHECK(counters.requested == requested);
    CHECK(counters.staged == staged);
    CHECK(counters.fallback == fallback && counters.fatal == 0UL);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    mode = argv[1];
    make_arguments();
    if (!strcmp(mode, "proof0")) {
        CHECK(!wt_sfclay_native_configure(1, 0, 0, 0, 128, 33,
                                           1U << 20, 0));
        wt_sfclay_live_dispatch((void **)1, native, WT_SFCLAY_NARG);
        CHECK(identity_calls == 0 && capture_calls == 0 && query_calls == 0);
        check_counters(1, 0, 1);
    } else if (!strcmp(mode, "omp")) {
        grid.id = 2;
        wt_sfclay_init_dispatch(&init_logical, initialize);
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 128, 33,
                                          1U << 20, 0));
        configured_scope((void **)1, 1);
        CHECK(init_calls == 1 && identity_calls == 1 && capture_calls == 1);
        CHECK(query_calls == 1 && native_calls == 1);
        check_counters(1, 0, 1);
    } else if (!strcmp(mode, "selected-miss")) {
        grid.id = 1;
        wt_sfclay_init_dispatch(&init_logical, initialize);
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 128, 33,
                                          1U << 20, 2));
        configured_scope((void **)1, 0);
        CHECK(identity_calls == 1 && capture_calls == 0 && query_calls == 1);
        CHECK(native_calls == 1);
        check_counters(1, 0, 1);
    } else if (!strcmp(mode, "selected-staged")) {
        grid.id = 2;
        wt_sfclay_init_dispatch(&init_logical, initialize);
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 128, 33,
                                          1U << 20, 2));
        configured_scope(grid.args, 0);
        CHECK(init_calls == 1 && identity_calls == 1 && capture_calls == 1);
        CHECK(query_calls == 3 && native_calls == 1);
        check_counters(1, 1, 0);
        for (unsigned i = 0; i < WT_SFCLAY_NARG; ++i) {
            float expected = (float)(i + 1U);
            if (wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR)
                CHECK(((float *)grid.args[i])[0] == expected);
            if (wt_sfclay_schema_access[i] == WT_SFCLAY_WRITABLE)
                CHECK(((float *)grid.args[i])[1] == (float)(1000U + i));
            else if (wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR)
                CHECK(((float *)grid.args[i])[1] == expected);
        }
    } else {
        CHECK(0);
    }
    free_arguments();
    printf("PASS %s\n", mode);
    return 0;
}
