#include "wt_sfclay_bootstrap.h"
#include "wt_sfclay_native.h"

#include <stdint.h>
#include <stdlib.h>

#ifndef WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED
#define WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED 0
#endif
#ifndef WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED
#define WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED 0
#endif
#ifndef WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED
#define WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED 0
#endif
#ifndef WT_SFCLAY_BUILD_PROFILE_ID
#define WT_SFCLAY_BUILD_PROFILE_ID "unattested"
#endif

#if (WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED != 0) && \
    (WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED != 1)
#error "WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED must be 0 or 1"
#endif
#if (WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED != 0) && \
    (WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED != 1)
#error "WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED must be 0 or 1"
#endif
#if (WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED != 0) && \
    (WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED != 1)
#error "WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED must be 0 or 1"
#endif
#if !WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED && \
    (WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED || \
     WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED)
#error "caller/initializer proofs require the source ABI proof"
#endif

enum {
    DEFAULT_MAX_SPANS = 4096,
    DEFAULT_MAX_NODES = 4096
};
#define DEFAULT_BYTE_BUDGET ((size_t)UINT64_C(1073741824))

struct boot_state {
    int started;
    int requested;
    int selector;
    int configured;
    size_t max_spans;
    size_t max_nodes;
    size_t byte_budget;
    enum wt_sfclay_bootstrap_reason reason;
};

static struct boot_state boot = {
    .reason = WT_SFCLAY_BOOT_NOT_STARTED,
};

static int positive_decimal(const char *name, size_t fallback, size_t *out)
{
    const char *text = getenv(name);
    size_t value = 0;
    if (!out) return 0;
    if (!text) {
        *out = fallback;
        return 1;
    }
    if (!*text) return 0;
    for (; *text; ++text) {
        unsigned digit;
        if (*text < '0' || *text > '9') return 0;
        digit = (unsigned)(*text - '0');
        if (value > (SIZE_MAX - digit) / 10U) return 0;
        value = value * 10U + digit;
    }
    if (!value) return 0;
    *out = value;
    return 1;
}

static int profile_valid(void)
{
    const unsigned char *text =
        (const unsigned char *)WT_SFCLAY_BUILD_PROFILE_ID;
    if (!*text) return 0;
    for (; *text; ++text)
        if (*text < 0x20U || *text == 0x7fU) return 0;
    return 1;
}

const char *wt_sfclay_bootstrap_reason_name(
    enum wt_sfclay_bootstrap_reason reason)
{
    static const char *const names[] = {
        "not-started", "disabled-by-request", "invalid-selector",
        "invalid-budget", "invalid-profile", "source-abi-unverified",
        "caller-chain-unverified", "initializer-relocation-unverified",
        "native-configure-rejected", "configured"
    };
    if (reason < WT_SFCLAY_BOOT_NOT_STARTED ||
        reason > WT_SFCLAY_BOOT_CONFIGURED)
        return "invalid-reason";
    return names[(unsigned)reason];
}

int wt_sfclay_bootstrap_start(int requested, int grid_selector)
{
    if (boot.started) return boot.configured;
    boot.started = 1;
    boot.requested = requested;
    boot.selector = grid_selector;
    boot.max_spans = DEFAULT_MAX_SPANS;
    boot.max_nodes = DEFAULT_MAX_NODES;
    boot.byte_budget = DEFAULT_BYTE_BUDGET;

    if (requested <= 0) {
        boot.reason = WT_SFCLAY_BOOT_DISABLED;
        return 0;
    }
    if (grid_selector < 0) {
        boot.reason = WT_SFCLAY_BOOT_INVALID_SELECTOR;
        return 0;
    }
    if (!positive_decimal("WT_SFCLAY_MAX_SPANS", boot.max_spans,
                          &boot.max_spans) ||
        !positive_decimal("WT_SFCLAY_MAX_NODES", boot.max_nodes,
                          &boot.max_nodes) ||
        !positive_decimal("WT_SFCLAY_BYTE_BUDGET", boot.byte_budget,
                          &boot.byte_budget)) {
        boot.reason = WT_SFCLAY_BOOT_INVALID_BUDGET;
        return 0;
    }
    if (!profile_valid()) {
        boot.reason = WT_SFCLAY_BOOT_INVALID_PROFILE;
        return 0;
    }
    if (!WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED) {
        boot.reason = WT_SFCLAY_BOOT_SOURCE_ABI;
        return 0;
    }
    if (!WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED) {
        boot.reason = WT_SFCLAY_BOOT_CALLER_CHAIN;
        return 0;
    }
    if (!WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED) {
        boot.reason = WT_SFCLAY_BOOT_INITIALIZER_RELOCATION;
        return 0;
    }

    boot.configured = wt_sfclay_native_configure(
        1, WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED,
        WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED,
        WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED,
        boot.max_spans, boot.max_nodes, boot.byte_budget, grid_selector);
    boot.reason = boot.configured ? WT_SFCLAY_BOOT_CONFIGURED :
                                   WT_SFCLAY_BOOT_NATIVE_REJECTED;
    return boot.configured;
}

int wt_sfclay_bootstrap_state(void)
{
    struct wt_sfclay_live_counters counters = {0};
    if (!boot.configured) return 0;
    wt_sfclay_live_get_counters(&counters);
    return counters.staged > 0UL ? 1 : -1;
}

int wt_sfclay_bootstrap_get_state(
    struct wt_sfclay_bootstrap_snapshot *out)
{
    if (!out) return 0;
    *out = (struct wt_sfclay_bootstrap_snapshot){
        .started = boot.started,
        .requested = boot.requested,
        .selector = boot.selector,
        .configured = boot.configured,
        .startup_state = wt_sfclay_bootstrap_state(),
        .source_abi_verified = WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED,
        .caller_chain_verified = WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED,
        .initializer_relocation_verified =
            WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED,
        .max_spans = boot.max_spans,
        .max_nodes = boot.max_nodes,
        .byte_budget = boot.byte_budget,
        .reason = boot.reason,
    };
    wt_sfclay_live_get_counters(&out->counters);
    return 1;
}

static void csv_text(FILE *stream, const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;
    fputc('"', stream);
    for (; *cursor; ++cursor) {
        if (*cursor == '"') {
            fputc('"', stream);
            fputc('"', stream);
        } else if (*cursor < 0x20U || *cursor == 0x7fU) {
            fputc('?', stream);
        } else {
            fputc((int)*cursor, stream);
        }
    }
    fputc('"', stream);
}

static void csv_key(FILE *stream, const char *key)
{
    fputs("# sfclay_bootstrap,", stream);
    fputs(key, stream);
    fputc(',', stream);
}

void wt_sfclay_bootstrap_dump(FILE *stream)
{
    _Static_assert(WT_SF_LIVE_REASON_COUNT == 10,
                   "update bootstrap CSV reason names");
    static const char *const live_reasons[WT_SF_LIVE_REASON_COUNT] = {
        "disabled", "source-abi", "no-scope", "omp", "native-query",
        "tiles", "argument-abi", "capacity", "prepare", "init"
    };
    struct wt_sfclay_bootstrap_snapshot state;
    if (!stream) return;
    if (!wt_sfclay_bootstrap_get_state(&state)) return;

    csv_key(stream, "profile");
    csv_text(stream, WT_SFCLAY_BUILD_PROFILE_ID);
    fputc('\n', stream);
#define DUMP_INT(key, value) do { \
    csv_key(stream, key); fprintf(stream, "%d\n", value); \
} while (0)
#define DUMP_SIZE(key, value) do { \
    csv_key(stream, key); fprintf(stream, "%zu\n", value); \
} while (0)
#define DUMP_ULONG(key, value) do { \
    csv_key(stream, key); fprintf(stream, "%lu\n", value); \
} while (0)
    DUMP_INT("started", state.started);
    DUMP_INT("requested", state.requested);
    DUMP_INT("selector", state.selector);
    DUMP_INT("configured", state.configured);
    DUMP_INT("startup_state", state.startup_state);
    DUMP_INT("source_abi_verified", state.source_abi_verified);
    DUMP_INT("caller_chain_verified", state.caller_chain_verified);
    DUMP_INT("initializer_relocation_verified",
             state.initializer_relocation_verified);
    DUMP_SIZE("max_spans", state.max_spans);
    DUMP_SIZE("max_nodes", state.max_nodes);
    DUMP_SIZE("byte_budget", state.byte_budget);
    csv_key(stream, "reason");
    csv_text(stream, wt_sfclay_bootstrap_reason_name(state.reason));
    fputc('\n', stream);
    DUMP_ULONG("requested_calls", state.counters.requested);
    DUMP_ULONG("staged", state.counters.staged);
    DUMP_ULONG("fallback", state.counters.fallback);
    DUMP_ULONG("fatal", state.counters.fatal);
    for (unsigned i = 0; i < WT_SF_LIVE_REASON_COUNT; ++i) {
        char key[64];
        int written = snprintf(key, sizeof key, "reason.%s", live_reasons[i]);
        if (written > 0 && (size_t)written < sizeof key)
            DUMP_ULONG(key, state.counters.reason[i]);
    }
#undef DUMP_INT
#undef DUMP_SIZE
#undef DUMP_ULONG
}
