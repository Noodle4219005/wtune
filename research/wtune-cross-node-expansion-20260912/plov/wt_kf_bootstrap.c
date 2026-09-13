#include "wt_kf_bootstrap.h"
#include "wt_kf_live.h"
#include "wt_kf_native.h"
#ifdef WT_KF_NODE_LOCAL
#include "wt_kf_node_live.h"
#endif

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef WT_KF_BUILD_SOURCE_ABI_VERIFIED
#define WT_KF_BUILD_SOURCE_ABI_VERIFIED 0
#endif
#ifndef WT_KF_BUILD_CALLER_CHAIN_VERIFIED
#define WT_KF_BUILD_CALLER_CHAIN_VERIFIED 0
#endif
#ifndef WT_KF_BUILD_PROFILE_ID
#define WT_KF_BUILD_PROFILE_ID "unattested"
#endif

#if (WT_KF_BUILD_SOURCE_ABI_VERIFIED != 0) && \
    (WT_KF_BUILD_SOURCE_ABI_VERIFIED != 1)
#error "WT_KF_BUILD_SOURCE_ABI_VERIFIED must be 0 or 1"
#endif
#if (WT_KF_BUILD_CALLER_CHAIN_VERIFIED != 0) && \
    (WT_KF_BUILD_CALLER_CHAIN_VERIFIED != 1)
#error "WT_KF_BUILD_CALLER_CHAIN_VERIFIED must be 0 or 1"
#endif
#if WT_KF_BUILD_CALLER_CHAIN_VERIFIED && !WT_KF_BUILD_SOURCE_ABI_VERIFIED
#error "caller-chain proof requires source-ABI proof"
#endif

enum { DEFAULT_MAX_SPANS = 4096, DEFAULT_MAX_NODES = 8192 };
#define DEFAULT_BYTE_BUDGET ((size_t)256 * 1024 * 1024)

struct bootstrap_config {
    int started, configured, requested, selector;
    size_t spans, nodes, budget;
    const char *refusal;
};
static struct bootstrap_config state = {0, 0, 0, 0, 0, 0, 0, "unchecked"};

static int decimal_size(const char *name, size_t fallback, size_t *out)
{
    const char *text = getenv(name);
    size_t value = 0;
    if (!text) { *out = fallback; return 1; }
    if (!*text) return 0;
    for (; *text; ++text) {
        unsigned digit;
        if (*text < '0' || *text > '9') return 0;
        digit = (unsigned)(*text - '0');
        if (value > (SIZE_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    if (!value) return 0;
    *out = value;
    return 1;
}

int wt_kf_bootstrap_start(int requested, int grid_selector)
{
    int enabled = requested > 0;
    if (state.started) return state.configured;
    state.started = 1;
    state.requested = enabled;
    state.selector = grid_selector;
    state.spans = DEFAULT_MAX_SPANS;
    state.nodes = DEFAULT_MAX_NODES;
    state.budget = DEFAULT_BYTE_BUDGET;

    if (!enabled) {
        state.refusal = "disabled-by-request";
        state.configured = wt_kf_native_configure_for_grid(0,
            WT_KF_BUILD_SOURCE_ABI_VERIFIED,
            WT_KF_BUILD_CALLER_CHAIN_VERIFIED, state.spans, state.nodes,
            state.budget, grid_selector) == 1;
        if (!state.configured) state.refusal = "native-configure-rejected";
        return state.configured;
    }
    if (!WT_KF_BUILD_SOURCE_ABI_VERIFIED) {
        state.refusal = "source-abi-unattested";
        return 0;
    }
    if (!decimal_size("WT_KF_MAX_SPANS", DEFAULT_MAX_SPANS, &state.spans) ||
        !decimal_size("WT_KF_MAX_NODES", DEFAULT_MAX_NODES, &state.nodes) ||
        !decimal_size("WT_KF_BYTE_BUDGET", DEFAULT_BYTE_BUDGET, &state.budget)) {
        state.refusal = "invalid-budget";
        return 0;
    }
    state.configured = wt_kf_native_configure_for_grid(1,
        WT_KF_BUILD_SOURCE_ABI_VERIFIED, WT_KF_BUILD_CALLER_CHAIN_VERIFIED,
        state.spans, state.nodes, state.budget, grid_selector) == 1;
    state.refusal = state.configured ? "configured" : "native-configure-rejected";
    return state.configured;
}

int wt_kf_bootstrap_state(void)
{
    struct wt_kf_live_counters counters = {0};
    if (!state.started) return -1;
    if (!state.configured || !state.requested) return 0;
    wt_kf_live_get_counters(&counters);
    if (counters.staged) return 1;
    return counters.requested ? 0 : -1;
}

void wt_kf_bootstrap_dump(FILE *stream)
{
    static const char *const labels[WT_KF_LIVE_REASON_COUNT] = {
        "disabled", "source_abi", "no_scope", "omp", "capacity", "tiles",
        "argument_abi", "prepare"
    };
    struct wt_kf_live_counters counters = {0};
    if (!stream) return;
    wt_kf_live_get_counters(&counters);
    fprintf(stream,
        "# wt_kf_bootstrap profile=%s source_abi_verified=%d caller_chain_verified=%d "
        "requested=%d grid_selector=%d max_spans=%zu max_nodes=%zu byte_budget=%zu "
        "configured=%d refusal=%s dispatch_requested=%lu staged=%lu fallback=%lu\n",
        WT_KF_BUILD_PROFILE_ID, WT_KF_BUILD_SOURCE_ABI_VERIFIED,
        WT_KF_BUILD_CALLER_CHAIN_VERIFIED, state.requested, state.selector,
        state.spans, state.nodes, state.budget, state.configured, state.refusal,
        counters.requested, counters.staged, counters.fallback);
    for (unsigned i = 0; i < WT_KF_LIVE_REASON_COUNT; ++i)
        fprintf(stream, "# wt_kf_bootstrap reason.%s=%lu\n", labels[i], counters.reason[i]);
#ifdef WT_KF_NODE_LOCAL
    if (wt_kf_node_live_active()) wt_kf_node_live_dump(stream);
#endif
}
