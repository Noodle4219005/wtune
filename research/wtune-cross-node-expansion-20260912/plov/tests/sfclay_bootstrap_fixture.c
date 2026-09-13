#define _POSIX_C_SOURCE 200809L
#include "wt_sfclay_bootstrap.h"
#include "wt_sfclay_live.h"
#include "wt_sfclay_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK %d: %s\n", __LINE__, #expr); \
        exit(80); \
    } \
} while (0)

static int native_calls;
static int native_result = 1;
static int got_enabled, got_source, got_caller, got_initializer, got_selector;
static size_t got_spans, got_nodes, got_budget;
static struct wt_sfclay_live_counters live_counters;

int wt_sfclay_native_configure(int enabled, int source_abi_verified,
    int caller_chain_verified, int initializer_relocation_verified,
    size_t max_spans, size_t max_nodes, size_t byte_budget, int grid_selector)
{
    ++native_calls;
    got_enabled = enabled;
    got_source = source_abi_verified;
    got_caller = caller_chain_verified;
    got_initializer = initializer_relocation_verified;
    got_spans = max_spans;
    got_nodes = max_nodes;
    got_budget = byte_budget;
    got_selector = grid_selector;
    return native_result;
}

void wt_sfclay_live_get_counters(struct wt_sfclay_live_counters *out)
{
    if (out) *out = live_counters;
}

static struct wt_sfclay_bootstrap_snapshot snapshot(void)
{
    struct wt_sfclay_bootstrap_snapshot result;
    CHECK(wt_sfclay_bootstrap_get_state(&result) == 1);
    return result;
}

static void check_refusal(enum wt_sfclay_bootstrap_reason expected)
{
    struct wt_sfclay_bootstrap_snapshot state;
    CHECK(wt_sfclay_bootstrap_start(7, 2) == 0);
    state = snapshot();
    CHECK(state.started == 1 && state.requested == 7 && state.selector == 2);
    CHECK(state.configured == 0 && state.reason == expected);
    CHECK(state.startup_state == 0 && native_calls == 0);
}

int main(int argc, char **argv)
{
    struct wt_sfclay_bootstrap_snapshot state;
    CHECK(argc == 2);
    if (!strcmp(argv[1], "before-start")) {
        state = snapshot();
        CHECK(state.started == 0 && state.reason == WT_SFCLAY_BOOT_NOT_STARTED);
        CHECK(wt_sfclay_bootstrap_state() == 0 && native_calls == 0);
    } else if (!strcmp(argv[1], "disabled")) {
        CHECK(wt_sfclay_bootstrap_start(0, 2) == 0);
        state = snapshot();
        CHECK(state.requested == 0 && state.reason == WT_SFCLAY_BOOT_DISABLED);
        CHECK(state.startup_state == 0 && native_calls == 0);
    } else if (!strcmp(argv[1], "negative-request")) {
        CHECK(wt_sfclay_bootstrap_start(-9, 2) == 0);
        state = snapshot();
        CHECK(state.requested == -9 && state.reason == WT_SFCLAY_BOOT_DISABLED);
        CHECK(native_calls == 0);
    } else if (!strcmp(argv[1], "invalid-selector")) {
        CHECK(wt_sfclay_bootstrap_start(1, -1) == 0);
        state = snapshot();
        CHECK(state.reason == WT_SFCLAY_BOOT_INVALID_SELECTOR);
        CHECK(state.selector == -1 && native_calls == 0);
    } else if (!strcmp(argv[1], "expect-source-proof")) {
        check_refusal(WT_SFCLAY_BOOT_SOURCE_ABI);
    } else if (!strcmp(argv[1], "expect-caller-proof")) {
        check_refusal(WT_SFCLAY_BOOT_CALLER_CHAIN);
    } else if (!strcmp(argv[1], "expect-initializer-proof")) {
        check_refusal(WT_SFCLAY_BOOT_INITIALIZER_RELOCATION);
    } else if (!strcmp(argv[1], "invalid-budget")) {
        CHECK(wt_sfclay_bootstrap_start(1, 0) == 0);
        state = snapshot();
        CHECK(state.reason == WT_SFCLAY_BOOT_INVALID_BUDGET);
        CHECK(native_calls == 0);
    } else if (!strcmp(argv[1], "valid-budget")) {
        CHECK(wt_sfclay_bootstrap_start(1, 3) == 1);
        CHECK(native_calls == 1 && got_enabled == 1 && got_source == 1);
        CHECK(got_caller == 1 && got_initializer == 1 && got_selector == 3);
        CHECK(got_spans == 17 && got_nodes == 19 && got_budget == 1048579);
    } else if (!strcmp(argv[1], "duplicate")) {
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 1);
        CHECK(setenv("WT_SFCLAY_MAX_SPANS", "99", 1) == 0);
        CHECK(wt_sfclay_bootstrap_start(9, 7) == 1);
        state = snapshot();
        CHECK(native_calls == 1 && state.requested == 1 && state.selector == 2);
        CHECK(state.configured == 1 && state.reason == WT_SFCLAY_BOOT_CONFIGURED);
    } else if (!strcmp(argv[1], "native-rejected")) {
        native_result = 0;
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 0);
        native_result = 1;
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 0);
        state = snapshot();
        CHECK(native_calls == 1 && state.reason == WT_SFCLAY_BOOT_NATIVE_REJECTED);
    } else if (!strcmp(argv[1], "zero-requests")) {
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 1);
        CHECK(wt_sfclay_bootstrap_state() == -1);
        state = snapshot();
        CHECK(state.max_spans == 4096 && state.max_nodes == 4096);
        CHECK(state.byte_budget == (size_t)1073741824);
    } else if (!strcmp(argv[1], "post-fallback")) {
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 1);
        live_counters.requested = 1;
        live_counters.fallback = 1;
        CHECK(wt_sfclay_bootstrap_state() == -1);
    } else if (!strcmp(argv[1], "actual-staged")) {
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 1);
        live_counters.requested = 3;
        live_counters.staged = 1;
        live_counters.fallback = 2;
        CHECK(wt_sfclay_bootstrap_state() == 1);
        state = snapshot();
        CHECK(state.startup_state == 1 && state.counters.staged == 1UL);
    } else if (!strcmp(argv[1], "dump")) {
        CHECK(wt_sfclay_bootstrap_start(7, 2) == 1);
        live_counters.requested = 8;
        live_counters.staged = 3;
        live_counters.fallback = 5;
        live_counters.fatal = 11;
        for (int i = 0; i < WT_SF_LIVE_REASON_COUNT; ++i)
            live_counters.reason[i] = (unsigned long)(i + 1);
        wt_sfclay_bootstrap_dump(NULL);
        wt_sfclay_bootstrap_dump(stdout);
    } else if (!strcmp(argv[1], "dump-invalid-profile")) {
        CHECK(wt_sfclay_bootstrap_start(1, 2) == 0);
        state = snapshot();
        CHECK(state.reason == WT_SFCLAY_BOOT_INVALID_PROFILE);
        wt_sfclay_bootstrap_dump(stdout);
    } else {
        CHECK(0);
    }
    return 0;
}
