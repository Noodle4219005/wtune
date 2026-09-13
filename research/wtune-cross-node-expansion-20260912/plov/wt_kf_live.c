#include "wt_kf_live.h"
#ifdef WT_KF_NODE_LOCAL
#include "wt_kf_node_live.h"
#endif
#include "wt_kf_capacity.h"
#include "wt_kf_dispatch.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

static struct wt_kf_live_config config;
static atomic_int configured; /* 0 unconfigured, 1 installing, 2 immutable */
static wt_live_spans *registry;
static _Atomic uint64_t published_generation;
static atomic_int outer_active, scope_reason = WT_KF_LIVE_NO_SCOPE;
static atomic_uint readers;
static atomic_int staging_busy;
static int scope_tiles;
static _Thread_local int dispatch_depth, control_depth;
struct invocation {
    void (*invoke)(void **);
    uint64_t generation;
};
static _Thread_local struct invocation *active_invocation;
static _Atomic unsigned long requested, staged, fallback;
static _Atomic unsigned long reasons[WT_KF_LIVE_REASON_COUNT];

int wt_kf_live_configure(const struct wt_kf_live_config *input)
{
    int expected = 0;
    if (!input || dispatch_depth || control_depth || atomic_load(&outer_active) ||
        !atomic_compare_exchange_strong(&configured, &expected, 1)) return 0;
    if (input->enabled) {
        registry = wt_live_spans_create(input->max_spans);
        if (!registry) { atomic_store(&configured, 0); return 0; }
    }
    config = *input;
    atomic_store_explicit(&configured, 2, memory_order_release);
    return 1;
}

void wt_kf_live_get_counters(struct wt_kf_live_counters *out)
{
#ifdef WT_KF_NODE_LOCAL
    if (out && wt_kf_node_live_active()) {
        struct wt_kf_node_live_counters c;
        wt_kf_node_live_get_counters(&c);
        *out=(struct wt_kf_live_counters){0};
        out->requested=(unsigned long)c.requested; out->staged=(unsigned long)c.committed;
        out->fallback=(unsigned long)c.fallback;
        return;
    }
#endif
    if (!out) return;
    out->requested = atomic_load(&requested);
    out->staged = atomic_load(&staged);
    out->fallback = atomic_load(&fallback);
    for (unsigned i = 0; i < WT_KF_LIVE_REASON_COUNT; ++i)
        out->reason[i] = atomic_load(&reasons[i]);
}

void wt_cumulus_grid_begin(void *grid)
{
#ifdef WT_KF_NODE_LOCAL
    if (wt_kf_node_live_active()) { wt_kf_node_live_begin(grid); return; }
#endif
    uint64_t generation;
    int tiles = 0;
    if (dispatch_depth || control_depth || atomic_exchange(&outer_active, 1)) abort();
    if (atomic_load(&readers)) abort();
    ++control_depth;
    atomic_store_explicit(&published_generation, 0, memory_order_release);
    atomic_store(&scope_reason, WT_KF_LIVE_NO_SCOPE);
    if (atomic_load_explicit(&configured, memory_order_acquire) != 2 || !config.enabled) {
        atomic_store(&scope_reason, WT_KF_LIVE_DISABLED);
        goto done;
    }
    if (!config.source_abi_verified) {
        atomic_store(&scope_reason, WT_KF_LIVE_SOURCE_ABI);
        goto done;
    }
    if (!config.omp_query || config.omp_query() != 0) {
        atomic_store(&scope_reason, WT_KF_LIVE_OMP);
        goto done;
    }
    if (!grid || !config.capture) {
        atomic_store(&scope_reason, WT_KF_LIVE_CAPACITY);
        goto done;
    }
    generation = wt_live_spans_begin(registry);
    if (!generation || config.capture(grid, registry, config.max_nodes, &tiles) != 1 ||
        wt_live_spans_finish(registry) != WT_SPAN_OK) {
        wt_live_spans_end(registry);
        atomic_store(&scope_reason, WT_KF_LIVE_CAPACITY);
        goto done;
    }
    if (tiles != 1) {
        wt_live_spans_end(registry);
        atomic_store(&scope_reason, WT_KF_LIVE_TILES);
        goto done;
    }
    scope_tiles = tiles;
    atomic_store_explicit(&published_generation, generation, memory_order_release);
done:
    --control_depth;
}

void wt_cumulus_grid_end(void)
{
#ifdef WT_KF_NODE_LOCAL
    if (wt_kf_node_live_active()) { wt_kf_node_live_end(); return; }
#endif
    if (dispatch_depth || control_depth) abort();
    atomic_store_explicit(&published_generation, 0, memory_order_release);
    if (atomic_load(&readers)) abort();
    if (registry) wt_live_spans_end(registry);
    atomic_store(&scope_reason, WT_KF_LIVE_NO_SCOPE);
    atomic_store(&outer_active, 0);
}

static void direct(enum wt_kf_live_reason reason, void **argv, void (*invoke)(void **))
{
    ++fallback;
    ++reasons[reason];
    invoke(argv);
}

static int query(void *opaque)
{
    (void)opaque;
    return config.omp_query ? config.omp_query() : -1;
}

static void invoke_current(void **argv)
{
    struct invocation *call = active_invocation;
    if (!call) abort();
    call->invoke(argv);
    /* This callback returns to wt_kf_execute before commit. Invalidating the
     * live scope during native execution must never permit an owner commit. */
    if (atomic_load_explicit(&published_generation, memory_order_acquire) != call->generation ||
        !wt_live_spans_is_current(registry, call->generation)) abort();
}

static enum wt_kf_live_reason map_reason(unsigned reason)
{
    switch (reason) {
    case WT_KF_DISPATCH_DISABLED: return WT_KF_LIVE_DISABLED;
    case WT_KF_DISPATCH_ABI: return WT_KF_LIVE_ARGUMENT_ABI;
    case WT_KF_DISPATCH_OMP: return WT_KF_LIVE_OMP;
    case WT_KF_DISPATCH_CAPACITY: return WT_KF_LIVE_CAPACITY;
    default: return WT_KF_LIVE_PREPARE;
    }
}

void wt_kf_live_dispatch(void **argv, void (*invoke)(void **), size_t nargs)
{
#ifdef WT_KF_NODE_LOCAL
    if (wt_kf_node_live_active()) { wt_kf_node_live_dispatch(argv,invoke,nargs); return; }
#endif
    struct wt_kf_capacity provider;
    struct wt_kf_dispatch_context context = {0};
    struct invocation call = {invoke, 0};
    uint64_t generation;
    if (!argv || !invoke || dispatch_depth || control_depth) abort();
    ++dispatch_depth;
    ++requested;
    if (atomic_load_explicit(&configured, memory_order_acquire) != 2 || !config.enabled) {
        direct(WT_KF_LIVE_DISABLED, argv, invoke);
        goto done;
    }
    if (!config.source_abi_verified) {
        direct(WT_KF_LIVE_SOURCE_ABI, argv, invoke);
        goto done;
    }
    if (nargs != WT_KF_NARG) {
        direct(WT_KF_LIVE_ARGUMENT_ABI, argv, invoke);
        goto done;
    }
    /* The actual runtime probe precedes reading even the argv vector, shape
     * scalars, or provider metadata. Missing/negative probes refuse staging. */
    if (!config.omp_query || config.omp_query() != 0) {
        direct(WT_KF_LIVE_OMP, argv, invoke);
        goto done;
    }
    atomic_fetch_add(&readers, 1);
    generation = atomic_load_explicit(&published_generation, memory_order_acquire);
    if (!generation) {
        enum wt_kf_live_reason reason = atomic_load(&scope_reason);
        direct(reason, argv, invoke);
        goto release;
    }
    /* Acquire before provider_init reads owner scalars; keep until the entire
     * transaction commits or takes its pre-invocation fallback. Another call
     * may already have executed, so a competing staging attempt is fatal. */
    if (atomic_exchange(&staging_busy, 1)) abort();
    if (!wt_kf_capacity_init_attested(&provider, argv, registry, generation,
            config.source_abi_verified, config.caller_chain_verified)) {
        direct(WT_KF_LIVE_CAPACITY, argv, invoke);
        goto unlock;
    }
    context.enabled = context.abi_verified = 1;
    context.transaction = (struct wt_kf_config){1, scope_tiles, 0, config.byte_budget};
    context.opaque = &provider;
    context.capacity = wt_kf_capacity_lookup;
    context.omp_query = query;
    call.generation = generation;
    active_invocation = &call;
    wt_kf_dispatch(&context, argv, nargs, invoke_current);
    active_invocation = NULL;
    atomic_fetch_add(&staged, atomic_load(&context.counters.staged));
    for (unsigned i = 0; i < WT_KF_DISPATCH_REASON_COUNT; ++i) {
        unsigned long n = atomic_load(&context.counters.fallback[i]);
        atomic_fetch_add(&fallback, n);
        atomic_fetch_add(&reasons[map_reason(i)], n);
    }
    wt_kf_capacity_reset(&provider);
unlock:
    atomic_store(&staging_busy, 0);
release:
    atomic_fetch_sub(&readers, 1);
done:
    --dispatch_depth;
}
