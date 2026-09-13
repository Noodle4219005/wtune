#include "wt_kf_dispatch.h"
#include <stdlib.h>
#include <stdatomic.h>
#if defined(__GNUC__)
extern int wt_kf_omp_in_parallel(void) __attribute__((weak));
#endif

static void terminal(struct wt_kf_dispatch_context *ctx, enum wt_kf_dispatch_reason reason)
{
    ++ctx->counters.fatal;
    if (ctx->fatal) ctx->fatal(ctx->opaque, reason);
    abort();
}
static void direct(struct wt_kf_dispatch_context *ctx, enum wt_kf_dispatch_reason reason,
                   void **argv, wt_kf_native_invoke invoke)
{
    ++ctx->counters.fallback[reason];
    invoke(argv);
}
struct invoke_context { wt_kf_native_invoke invoke; };
static _Thread_local int callback_depth;
static int invoke_clone(void **argv, void *opaque)
{
    ++callback_depth;
    ((struct invoke_context *)opaque)->invoke(argv);
    --callback_depth;
    return 0;
}
static struct wt_kf_dispatch_context bridge_context;
static atomic_int bridge_state; /* 0 default, 1 configuring, 2 immutable configured */
int wt_kf_bridge_configure(const struct wt_kf_dispatch_context *context)
{
    int expected = 0;
    if (!context || !atomic_compare_exchange_strong(&bridge_state, &expected, 1)) return 0;
    bridge_context = *context;
    atomic_store(&bridge_context.busy, 0);
    atomic_store_explicit(&bridge_state, 2, memory_order_release);
    return 1;
}
void wt_kf_bridge_dispatch(void **argv, void (*invoke)(void **), size_t nargs)
{
    if (atomic_load_explicit(&bridge_state, memory_order_acquire) == 2)
        wt_kf_dispatch(&bridge_context, argv, nargs, invoke);
    else { struct wt_kf_dispatch_context disabled = {0}; wt_kf_dispatch(&disabled, argv, nargs, invoke); }
}
void wt_kf_dispatch(struct wt_kf_dispatch_context *ctx, void **argv, size_t nargs,
                    wt_kf_native_invoke invoke)
{
    size_t capacity[WT_KF_NARG] = {0};
    struct wt_kf_txn *transaction = NULL;
    enum wt_kf_status status;
    struct invoke_context callback = { invoke };
    unsigned i;
    int in_parallel = 1; /* A missing runtime probe never authorizes staging. */
    if (!ctx || !invoke || !argv) abort();
    ++ctx->counters.requested;
    if (callback_depth) terminal(ctx, WT_KF_DISPATCH_REENTRY);
    if (!ctx->enabled) { direct(ctx, WT_KF_DISPATCH_DISABLED, argv, invoke); return; }
    if (!ctx->abi_verified || nargs != WT_KF_NARG) { direct(ctx, WT_KF_DISPATCH_ABI, argv, invoke); return; }
    if (ctx->omp_query) { ++callback_depth; in_parallel = ctx->omp_query(ctx->opaque); --callback_depth; }
#if defined(__GNUC__)
    else if (wt_kf_omp_in_parallel) { ++callback_depth; in_parallel = wt_kf_omp_in_parallel(); --callback_depth; }
#else
    else in_parallel = 1; /* No portable default probe: do not stage. */
#endif
    if (in_parallel) { direct(ctx, WT_KF_DISPATCH_OMP, argv, invoke); return; }
    if (!ctx->capacity) { direct(ctx, WT_KF_DISPATCH_CAPACITY, argv, invoke); return; }
    for (i = 0; i < WT_KF_NARG; ++i)
        { int known; ++callback_depth; known = ctx->capacity(ctx->opaque, i, argv[i], &capacity[i]); --callback_depth;
          if (!known) {
            direct(ctx, WT_KF_DISPATCH_CAPACITY, argv, invoke); return;
          }
        }
    if (atomic_exchange(&ctx->busy, 1)) terminal(ctx, WT_KF_DISPATCH_REENTRY);
    status = wt_kf_prepare(argv, capacity, &ctx->transaction, &transaction);
    if (status != WT_KF_OK) { atomic_store(&ctx->busy, 0); direct(ctx, WT_KF_DISPATCH_PREPARE, argv, invoke); return; }
    status = wt_kf_execute(transaction, invoke_clone, &callback);
    if (status == WT_KF_OK) status = wt_kf_commit(transaction);
    wt_kf_destroy(transaction);
    atomic_store(&ctx->busy, 0);
    if (status != WT_KF_OK) terminal(ctx, WT_KF_DISPATCH_FATAL);
    ++ctx->counters.staged;
}
