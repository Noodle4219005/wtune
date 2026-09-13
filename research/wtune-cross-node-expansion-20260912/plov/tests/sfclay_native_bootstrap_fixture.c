#include "wt_sfclay_native.h"
#include "wt_sfclay_live.h"
#include "wt_sfclay_native_query.h"
#include "wt_kf_live.h"
#include "wt_kf_native.h"

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

static struct wt_sfclay_live_config sf_config;
static struct wt_kf_live_config kf_config;
static int sf_config_calls, kf_config_calls, reject_sf;
static int identity_calls, capture_calls;
static int identity_override = -1;

void wt_grid_identity_(void *grid, int *grid_id, int *status);
void wt_grid_capture_(void *grid, void **context, const size_t *max_nodes,
                      int *num_tiles, int *status);

void wt_grid_identity_(void *grid, int *grid_id, int *status)
{
    ++identity_calls;
    *grid_id = identity_override >= 0 ? identity_override : (int)(intptr_t)grid;
    *status = WT_SPAN_OK;
}

void wt_grid_capture_(void *grid, void **context, const size_t *max_nodes,
                      int *num_tiles, int *status)
{
    (void)grid;
    CHECK(context && *context && max_nodes && *max_nodes == 19);
    ++capture_calls;
    *num_tiles = 1;
    *status = WT_SPAN_OK;
}

int wt_sfclay_native_query(void *grid, void *config,
                           struct wt_sfclay_native_result *result)
{
    (void)grid;
    (void)config;
    (void)result;
    return 0;
}

int wt_sfclay_live_configure(const struct wt_sfclay_live_config *config)
{
    ++sf_config_calls;
    if (reject_sf) return 0;
    CHECK(config);
    sf_config = *config;
    return 1;
}

int wt_kf_live_configure(const struct wt_kf_live_config *config)
{
    ++kf_config_calls;
    CHECK(config);
    kf_config = *config;
    return 1;
}

static void begin_capture(wt_live_spans *spans)
{
    CHECK(wt_live_spans_begin(spans) != 0);
}

static void exact_config(void)
{
    CHECK(sf_config_calls == 1);
    CHECK(sf_config.enabled == 1 && sf_config.source_abi_verified == 1);
    CHECK(sf_config.caller_chain_verified == 1);
    CHECK(sf_config.initializer_relocation_verified == 1);
    CHECK(sf_config.max_spans == 17 && sf_config.max_nodes == 19);
    CHECK(sf_config.byte_budget == 1048579);
    CHECK(sf_config.capture == wt_sfclay_native_capture);
    CHECK(sf_config.omp_query == omp_in_parallel);
    CHECK(sf_config.native_query == wt_sfclay_native_query);
}

static void capture_result(wt_live_spans *spans, intptr_t grid, int expected)
{
    int tiles = -1;
    begin_capture(spans);
    CHECK(wt_sfclay_native_capture((void *)grid, spans, 19, &tiles) == expected);
    CHECK(tiles == expected);
}

int main(int argc, char **argv)
{
    wt_live_spans *spans = wt_live_spans_create(4);
    CHECK(argc == 2 && spans);
    if (!strcmp(argv[1], "proof-gates")) {
        int tiles = 9;
        begin_capture(spans);
        CHECK(!wt_sfclay_native_capture((void *)2, spans, 19, &tiles));
        CHECK(tiles == 0 && identity_calls == 0 && capture_calls == 0);
        CHECK(!wt_sfclay_native_configure(0, 1, 1, 1, 17, 19, 1048579, 2));
        CHECK(!wt_sfclay_native_configure(1, 0, 1, 1, 17, 19, 1048579, 2));
        CHECK(!wt_sfclay_native_configure(1, 1, 0, 1, 17, 19, 1048579, 2));
        CHECK(!wt_sfclay_native_configure(1, 1, 1, 0, 17, 19, 1048579, 2));
        CHECK(!wt_sfclay_native_configure(1, 1, 1, 1, 0, 19, 1048579, 2));
        CHECK(!wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, -1));
        CHECK(sf_config_calls == 0 && wt_sfclay_native_state() == 0);
    } else if (!strcmp(argv[1], "rollback")) {
        reject_sf = 1;
        CHECK(!wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 2));
        CHECK(wt_sfclay_native_state() == WT_SFCLAY_NATIVE_UNCONFIGURED);
        reject_sf = 0;
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 2));
        CHECK(sf_config_calls == 2);
    } else if (!strcmp(argv[1], "immutable")) {
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 2));
        exact_config();
        CHECK(!wt_sfclay_native_configure(1, 1, 1, 1, 99, 99, 99, 3));
        CHECK(sf_config_calls == 1);
        capture_result(spans, 2, 1);
        capture_result(spans, 3, 0);
    } else if (!strcmp(argv[1], "selector-all")) {
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 0));
        identity_override = 0;
        capture_result(spans, 1, 0);
        identity_override = -1;
        capture_result(spans, 4, 1);
        CHECK(identity_calls == 2 && capture_calls == 1);
    } else if (!strcmp(argv[1], "omp-callback")) {
        int inside = 0;
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 0));
        exact_config();
#pragma omp parallel num_threads(2) reduction(+:inside)
        { inside += sf_config.omp_query() != 0; }
        CHECK(inside == 2);
    } else if (!strcmp(argv[1], "kf-independent")) {
        CHECK(wt_kf_native_configure_for_grid(1, 1, 1, 17, 19, 1048579, 7));
        CHECK(wt_sfclay_native_configure(1, 1, 1, 1, 17, 19, 1048579, 2));
        CHECK(kf_config_calls == 1 && sf_config_calls == 1);
        capture_result(spans, 2, 1);
        { int tiles = 0; begin_capture(spans);
          CHECK(kf_config.capture((void *)7, spans, 19, &tiles) == 1);
          CHECK(tiles == 1); }
        capture_result(spans, 7, 0);
    } else {
        CHECK(0);
    }
    wt_live_spans_destroy(spans);
    printf("PASS %s\n", argv[1]);
    return 0;
}
