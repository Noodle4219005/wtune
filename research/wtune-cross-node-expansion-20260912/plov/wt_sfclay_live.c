#include "wt_sfclay_live_internal.h"
#include "wt_sfclay_capacity.h"
#include "wt_sfclay.h"
#ifdef WT_SFCLAY_NODE_LOCAL
#include "wt_sfclay_node_live.h"
#endif
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static struct wt_sfclay_live_config config;
static pthread_t owner;
static uint64_t owner_nonce;
static wt_live_spans *spans;
static atomic_int configured;
static atomic_flag scope_gate = ATOMIC_FLAG_INIT;
static atomic_flag stage_gate = ATOMIC_FLAG_INIT;
static _Atomic uint64_t tokens;
static _Atomic unsigned long requested, staged, fallback, fatal;
static _Atomic unsigned long reasons[WT_SF_LIVE_REASON_COUNT];

_Noreturn void wt_sf_die(void)
{
    atomic_fetch_add(&fatal, 1);
    abort();
}

static uint64_t next_token(void)
{
    uint64_t previous = atomic_load(&tokens);
    do {
        if (previous == UINT64_MAX) wt_sf_die();
    } while (!atomic_compare_exchange_weak(&tokens, &previous, previous + 1));
    return previous + 1;
}

static int boolean(int value) { return value == 0 || value == 1; }

int wt_sfclay_live_configure(const struct wt_sfclay_live_config *input)
{
    int saved, success = 0;
    if (wt_sf_tls.control_depth || wt_sf_tls.dispatch_depth ||
        wt_sf_tls.init_depth) wt_sf_die();
    if (!input || !boolean(input->enabled) ||
        !boolean(input->source_abi_verified) ||
        !boolean(input->caller_chain_verified) ||
        !boolean(input->initializer_relocation_verified)) return 0;
    wt_sf_pcheck(pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &saved));
    wt_sf_pcheck(pthread_mutex_lock(&wt_sf_lifecycle_mutex));
    if (wt_sf_active || atomic_load(&configured)) goto done;
    if (input->enabled) {
        spans = wt_live_spans_create(input->max_spans);
        if (!spans) goto done;
    }
    config = *input;
    owner = pthread_self();
    owner_nonce = next_token();
    wt_sf_tls.owner_nonce = owner_nonce;
    /* Initialization belongs to the independent lease and is never reset. */
    atomic_store(&configured, 1);
    success = 1;
done:
    wt_sf_pcheck(pthread_mutex_unlock(&wt_sf_lifecycle_mutex));
    wt_sf_pcheck(pthread_setcancelstate(saved, NULL));
    return success;
}

void wt_sfclay_live_get_counters(struct wt_sfclay_live_counters *out)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (out && wt_sfclay_node_live_active()) {
        struct wt_sfclay_node_live_counters node;
        wt_sfclay_node_live_get_counters(&node);
        memset(out,0,sizeof(*out));
        out->requested=(unsigned long)node.requested;
        out->staged=(unsigned long)node.committed;
        out->fallback=(unsigned long)node.fallback;
        out->fatal=(unsigned long)node.fatal;
        return;
    }
#endif
    if (!out) return;
    out->requested = atomic_load(&requested);
    out->staged = atomic_load(&staged);
    out->fallback = atomic_load(&fallback);
    out->fatal = atomic_load(&fatal);
    for (unsigned i = 0; i < WT_SF_LIVE_REASON_COUNT; ++i)
        out->reason[i] = atomic_load(&reasons[i]);
}

/* TLS nonce is checked first: after the configuring thread exits, its saved
 * pthread_t may be expired or reused and must not reach pthread_equal. */
static int is_owner(void)
{
    return wt_sf_tls.owner_nonce && wt_sf_tls.owner_nonce == owner_nonce &&
           pthread_equal(owner, pthread_self());
}

/* This gate never calls callbacks and never inspects grid, config or argv. */
static enum wt_sfclay_live_reason base_reason(void)
{
    if (!atomic_load(&configured) || !config.enabled) return WT_SF_LIVE_DISABLED;
    if (!config.source_abi_verified || !config.caller_chain_verified ||
        !config.initializer_relocation_verified) return WT_SF_LIVE_SOURCE_ABI;
    if (!is_owner()) return WT_SF_LIVE_NO_SCOPE;
    if (!config.omp_query) return WT_SF_LIVE_OMP;
    if (!config.native_query) return WT_SF_LIVE_NATIVE_QUERY;
    if (!config.capture) return WT_SF_LIVE_CAPACITY;
    return WT_SF_LIVE_REASON_COUNT;
}

static int omp_serial(void)
{
    int value;
    if (wt_sf_tls.control_depth) wt_sf_die();
    wt_sf_tls.control_depth = 1;
    value = config.omp_query();
    wt_sf_tls.control_depth = 0;
    return value == 0;
}

static int query(struct wt_sfclay_native_result *result)
{
    int success;
    if (wt_sf_tls.control_depth) wt_sf_die();
    memset(result, 0, sizeof *result);
    wt_sf_tls.control_depth = 1;
    success = config.native_query(wt_sf_tls.grid, wt_sf_tls.config, result);
    wt_sf_tls.control_depth = 0;
    return success == 1 && result->status == WT_SFCLAY_NATIVE_OK &&
           result->grid_id > 0 && result->actual_scheme == result->expected_scheme &&
           result->actual_fractional == 0;
}

static int same_query(const struct wt_sfclay_native_result *a,
                      const struct wt_sfclay_native_result *b)
{
    return a->grid_id == b->grid_id && a->actual_scheme == b->actual_scheme &&
           a->actual_fractional == b->actual_fractional &&
           a->expected_scheme == b->expected_scheme && a->status == b->status;
}

static int current_scope(void)
{
    return wt_sf_tls.outer && wt_sf_tls.surface && wt_sf_tls.outer_ok &&
           wt_sf_tls.surface_ok && is_owner() &&
           wt_live_spans_is_current(spans, wt_sf_tls.generation);
}

uint64_t wt_sfclay_first_rk_begin(void *grid, void *cfg)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (wt_sfclay_node_live_active())
        return wt_sfclay_node_live_first_rk_begin(grid,cfg);
#endif
    int captured, tiles = 0;
    if (wt_sf_tls.outer || wt_sf_tls.surface || wt_sf_tls.dispatch_depth ||
        wt_sf_tls.control_depth || wt_sf_tls.init_depth) wt_sf_die();
    wt_sf_activity_enter();
    wt_sf_tls.outer = next_token();
    wt_sf_tls.reason = base_reason();
    if (wt_sf_tls.reason != WT_SF_LIVE_REASON_COUNT) return wt_sf_tls.outer;
    wt_sf_tls.reason = WT_SF_LIVE_OMP;
    if (!omp_serial()) return wt_sf_tls.outer;
    wt_sf_tls.reason = WT_SF_LIVE_NATIVE_QUERY;
    if (!grid || !cfg) return wt_sf_tls.outer;
    wt_sf_tls.grid = grid;
    wt_sf_tls.config = cfg;
    if (atomic_flag_test_and_set(&scope_gate)) wt_sf_die();
    if (!query(&wt_sf_tls.result)) goto fail_gate;
    wt_sf_tls.reason = WT_SF_LIVE_CAPACITY;
    wt_sf_tls.generation = wt_live_spans_begin(spans);
    if (!wt_sf_tls.generation) goto fail_spans;
    wt_sf_tls.control_depth = 1;
    captured = config.capture(grid, spans, config.max_nodes, &tiles);
    wt_sf_tls.control_depth = 0;
    if (captured != 1 || wt_live_spans_finish(spans) != WT_SPAN_OK) goto fail_spans;
    wt_sf_tls.reason = WT_SF_LIVE_TILES;
    if (tiles != 1) goto fail_spans;
    wt_sf_tls.outer_ok = 1;
    wt_sf_tls.reason = WT_SF_LIVE_REASON_COUNT;
    return wt_sf_tls.outer;
fail_spans:
    wt_live_spans_end(spans);
    wt_sf_tls.generation = 0;
fail_gate:
    atomic_flag_clear(&scope_gate);
    return wt_sf_tls.outer;
}

void wt_sfclay_first_rk_end(uint64_t token)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (wt_sfclay_node_live_active()) {
        wt_sfclay_node_live_first_rk_end(token); return;
    }
#endif
    if (!token || token != wt_sf_tls.outer || wt_sf_tls.surface ||
        wt_sf_tls.dispatch_depth || wt_sf_tls.control_depth ||
        wt_sf_tls.init_depth) wt_sf_die();
    if (wt_sf_tls.outer_ok) {
        wt_sf_tls.outer_ok = 0;
        wt_sf_tls.generation = 0;
        wt_live_spans_end(spans);
        atomic_flag_clear(&scope_gate);
    }
    wt_sf_tls.outer = 0;
    wt_sf_tls.grid = wt_sf_tls.config = NULL;
    memset(&wt_sf_tls.result, 0, sizeof wt_sf_tls.result);
    wt_sf_tls.reason = WT_SF_LIVE_NO_SCOPE;
    wt_sf_activity_leave();
}

uint64_t wt_sfclay_surface_begin(void)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (wt_sfclay_node_live_active())
        return wt_sfclay_node_live_surface_begin();
#endif
    if (wt_sf_tls.surface || wt_sf_tls.dispatch_depth ||
        wt_sf_tls.control_depth || wt_sf_tls.init_depth) wt_sf_die();
    wt_sf_activity_enter();
    wt_sf_tls.surface = next_token();
    wt_sf_tls.surface_ok = wt_sf_tls.outer_ok && wt_sf_tls.outer &&
        is_owner() &&
        wt_live_spans_is_current(spans, wt_sf_tls.generation);
    return wt_sf_tls.surface;
}

void wt_sfclay_surface_end(uint64_t token)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (wt_sfclay_node_live_active()) {
        wt_sfclay_node_live_surface_end(token); return;
    }
#endif
    if (!token || token != wt_sf_tls.surface || wt_sf_tls.dispatch_depth ||
        wt_sf_tls.control_depth || wt_sf_tls.init_depth) wt_sf_die();
    wt_sf_tls.surface = 0;
    wt_sf_tls.surface_ok = 0;
    wt_sf_activity_leave();
}

static void direct(enum wt_sfclay_live_reason reason, void **args,
                   void (*real)(void **))
{
    atomic_fetch_add(&fallback, 1);
    atomic_fetch_add(&reasons[reason], 1);
    real(args);
}

void wt_sfclay_live_dispatch(void **args, void (*real)(void **), size_t nargs)
{
#ifdef WT_SFCLAY_NODE_LOCAL
    if (wt_sfclay_node_live_active()) {
        wt_sfclay_node_live_dispatch(args,real,nargs); return;
    }
#endif
    struct wt_sfclay_native_result result;
    struct wt_sfclay_capacity capacity;
    wt_sfclay_tx *tx = NULL;
    uint64_t init_epoch, outer, surface, epoch;
    enum wt_sfclay_live_reason reason;
    int stage_locked = 0, initialized;
    if (!args || !real || nargs != WT_SFCLAY_NARG || wt_sf_tls.dispatch_depth ||
        wt_sf_tls.control_depth || wt_sf_tls.init_depth || wt_sf_tls.reader_depth)
        wt_sf_die();
    wt_sf_activity_enter();
    wt_sf_tls.dispatch_depth = 1;
    atomic_fetch_add(&requested, 1);
    reason = base_reason();
    if (reason != WT_SF_LIVE_REASON_COUNT) goto fallback_lease;
    if (!current_scope()) {
        reason = wt_sf_tls.outer && wt_sf_tls.reason != WT_SF_LIVE_REASON_COUNT ?
                 wt_sf_tls.reason : WT_SF_LIVE_NO_SCOPE;
        goto fallback_lease;
    }
    reason = WT_SF_LIVE_OMP;
    if (!omp_serial()) goto fallback_lease;
    reason = WT_SF_LIVE_NATIVE_QUERY;
    if (!query(&result) || !same_query(&result, &wt_sf_tls.result)) goto fallback_lease;
    /* Lock order: scope -> staging -> reader -> provider/transaction. */
    if (atomic_flag_test_and_set(&stage_gate)) wt_sf_die();
    stage_locked = 1;
    initialized = wt_sf_reader_get(&init_epoch);
    reason = WT_SF_LIVE_INIT;
    if (!initialized) goto fallback_held;
    reason = WT_SF_LIVE_NO_SCOPE;
    if (!current_scope()) goto fallback_held;
    outer = wt_sf_tls.outer;
    surface = wt_sf_tls.surface;
    epoch = wt_sf_tls.generation;
    reason = WT_SF_LIVE_CAPACITY;
    if (!wt_sfclay_capacity_init_attested(&capacity, args, spans, epoch,
            config.source_abi_verified,
            config.caller_chain_verified && current_scope())) goto fallback_held;
    reason = WT_SF_LIVE_PREPARE;
    if (wt_sfclay_prepare(args, capacity.bytes, config.byte_budget, &tx) != WT_SFCLAY_OK)
        goto fallback_held;

    real(wt_sfclay_cloned_argv(tx));
    /* Native has started: every subsequent failure is fatal, with no replay. */
    if (!current_scope() || outer != wt_sf_tls.outer || surface != wt_sf_tls.surface ||
        epoch != wt_sf_tls.generation || !wt_sf_reader_current(init_epoch) ||
        !query(&result) || !same_query(&result, &wt_sf_tls.result) ||
        !current_scope() || outer != wt_sf_tls.outer || surface != wt_sf_tls.surface ||
        epoch != wt_sf_tls.generation || !wt_sf_reader_current(init_epoch) ||
        wt_sfclay_commit(tx) != WT_SFCLAY_OK) wt_sf_die();
    wt_sfclay_destroy(tx);
    wt_sfclay_capacity_reset(&capacity);
    atomic_fetch_add(&staged, 1);
    goto release;

fallback_lease:
    (void)wt_sf_reader_get(&init_epoch);
fallback_held:
    direct(reason, args, real);
release:
    wt_sf_reader_put();
    if (stage_locked) atomic_flag_clear(&stage_gate);
    wt_sf_tls.dispatch_depth = 0;
    wt_sf_activity_leave();
}
