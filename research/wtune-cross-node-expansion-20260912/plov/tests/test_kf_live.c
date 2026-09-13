#define _GNU_SOURCE
#include "wt_kf_live.h"
#include "wt_kf_cps_schema.h"
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static void *args[WT_KF_NARG];
static int32_t scalar[WT_KF_NARG];
static float backing[WT_KF_NARG][32];
static int omp_active, capture_mode, actual_tiles = 1, captures;
static _Atomic int calls;
static int changed_owner, invoke_end;
static float *shared_raincv;
static wt_live_spans *captured_spans;
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t entered = PTHREAD_COND_INITIALIZER;
static int native_entered;

static void initialize(void)
{
    memset(backing, 0, sizeof backing);
    memset(scalar, 0, sizeof scalar);
    for (unsigned i = 0; i < WT_KF_NARG; ++i) {
        args[i] = backing[i] + 4; /* exercise live interior pointers */
        if (wt_kf_shape[i] == WT_KF_SCALAR) { args[i] = &scalar[i]; scalar[i] = 1; }
    }
    scalar[WT_KF_KDE] = scalar[WT_KF_KME] = 2;
    scalar[WT_KF_TRIGGER] = 1;
    scalar[WT_KF_ADAPT_STEP_FLAG] = 0;
    scalar[WT_KF_KF_EDRATES] = 1;
    args[WT_KF_RAINCV] = shared_raincv + 4;
    captures = calls = omp_active = capture_mode = changed_owner = invoke_end = 0;
    actual_tiles = 1;
}
static int probe(void) { return omp_active; }
static int capture(void *grid, wt_live_spans *spans, size_t limit, int *tiles)
{
    assert(grid == backing && limit == 200);
    ++captures;
    captured_spans = spans;
    *tiles = actual_tiles;
    for (unsigned i = 0; i < WT_KF_NARG; ++i) {
        if (wt_kf_shape[i] == WT_KF_SCALAR) continue;
        if (capture_mode == 2 && i == WT_KF_U) continue;
        if (capture_mode == 3 && (i == WT_KF_TH || i == WT_KF_PI ||
            i == WT_KF_DZ8W || i == WT_KF_CU_ACT_FLAG)) continue;
        int type = i == WT_KF_CU_ACT_FLAG ? WT_SPAN_LOGICAL : WT_SPAN_REAL;
        assert(wt_live_spans_add(spans, (float *)args[i] - 4, sizeof backing[i], type, 4, 1) == WT_SPAN_OK);
        if (capture_mode == 1) return 0; /* partial capture must never publish */
    }
    return 1;
}
static struct wt_kf_live_config configuration(void)
{
    return (struct wt_kf_live_config){.enabled = 1, .source_abi_verified = 1,
        .max_spans = 200, .max_nodes = 200, .byte_budget = 100000,
        .capture = capture, .omp_query = probe, .caller_chain_verified = 0};
}
static void native(void **a)
{
    ++calls;
    ((float *)a[WT_KF_RAINCV])[0] += 1.0f;
    if (changed_owner) ((float *)args[WT_KF_RAINCV])[0] += 5.0f;
    if (invoke_end == 1) wt_cumulus_grid_end();
    if (invoke_end == 2) wt_live_spans_end(captured_spans);
}
static void no_read(void **a) { (void)a; ++calls; }
static struct wt_kf_live_counters counts(void)
{
    struct wt_kf_live_counters c;
    wt_kf_live_get_counters(&c);
    return c;
}
static void scenario(int mode)
{
    struct wt_kf_live_config config = configuration();
    initialize();
    if (mode == 0) {
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        assert(calls == 1 && counts().reason[WT_KF_LIVE_DISABLED] == 1);
        return;
    }
    if (mode == 1) config.source_abi_verified = 0;
    if (mode == 2) config.omp_query = NULL;
    if (mode == 3) config.capture = NULL;
    if (mode == 4) actual_tiles = 2;
    if (mode == 5) capture_mode = 1;
    if (mode == 6) capture_mode = 2;
    if (mode == 7) config.byte_budget = 1;
    if (mode == 20 || mode == 21) capture_mode = 3;
    if (mode == 21) config.caller_chain_verified = 1;
    assert(wt_kf_live_configure(&config));
    assert(!wt_kf_live_configure(&config));
    if (mode == 8) {
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        assert(calls == 1 && counts().reason[WT_KF_LIVE_NO_SCOPE] == 1);
        return;
    }
    wt_cumulus_grid_begin(backing);
    if (mode == 9 || mode == 10) {
        omp_active = mode == 9 ? 1 : -1;
        wt_kf_live_dispatch((void **)1, no_read, WT_KF_NARG);
        assert(calls == 1 && counts().reason[WT_KF_LIVE_OMP] == 1);
    } else if (mode == 11) {
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        if (counts().staged != 2) {
            struct wt_kf_live_counters c = counts();
            fprintf(stderr, "staging calls %d staged %lu fallback %lu prepare %lu capacity %lu\n",
                    calls, c.staged, c.fallback, c.reason[WT_KF_LIVE_PREPARE], c.reason[WT_KF_LIVE_CAPACITY]);
        }
        assert(calls == 2 && counts().staged == 2 && counts().fallback == 0);
        assert(((float *)args[WT_KF_RAINCV])[0] == 2.0f);
    } else if (mode == 12) {
        wt_cumulus_grid_end();
        wt_kf_live_dispatch((void **)1, no_read, WT_KF_NARG);
        assert(calls == 1 && counts().reason[WT_KF_LIVE_NO_SCOPE] == 1);
        return;
    } else if (mode == 13) {
        wt_kf_live_dispatch(args, native, WT_KF_NARG - 1);
        assert(calls == 1 && counts().reason[WT_KF_LIVE_ARGUMENT_ABI] == 1);
    } else if (mode == 14) {
        changed_owner = 1;
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        _exit(99);
    } else if (mode == 15) {
        invoke_end = 1;
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        _exit(99);
    } else if (mode == 16) {
        wt_cumulus_grid_begin(backing);
        _exit(99);
    } else if (mode == 19) {
        invoke_end = 2;
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        _exit(99);
    } else if (mode == 21) {
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        assert(calls == 1 && counts().staged == 1 && counts().fallback == 0);
        assert(shared_raincv[4] == 1.0f);
    } else {
        wt_kf_live_dispatch(args, native, WT_KF_NARG);
        enum wt_kf_live_reason reason = mode == 1 ? WT_KF_LIVE_SOURCE_ABI :
            mode == 2 ? WT_KF_LIVE_OMP : mode == 4 ? WT_KF_LIVE_TILES :
            mode == 7 ? WT_KF_LIVE_PREPARE : WT_KF_LIVE_CAPACITY;
        if (counts().reason[reason] != 1) {
            struct wt_kf_live_counters c = counts();
            fprintf(stderr, "mode %d requested %lu staged %lu fallback %lu reasons:",
                    mode, c.requested, c.staged, c.fallback);
            for (unsigned i = 0; i < WT_KF_LIVE_REASON_COUNT; ++i) fprintf(stderr, " %lu", c.reason[i]);
            fputc('\n', stderr);
        }
        assert(calls == 1 && counts().fallback == 1 && counts().reason[reason] == 1);
        assert(counts().staged == 0);
        assert(((float *)args[WT_KF_RAINCV])[0] == 1.0f);
    }
    wt_cumulus_grid_end();
}
static void *thread_dispatch(void *unused)
{
    (void)unused;
    wt_kf_live_dispatch((void **)1, no_read, WT_KF_NARG);
    return NULL;
}
static void threaded_refusal(void)
{
    pthread_t a, b;
    struct wt_kf_live_config config = configuration();
    initialize(); assert(wt_kf_live_configure(&config));
    wt_cumulus_grid_begin(backing);
    omp_active = 1;
    assert(!pthread_create(&a, NULL, thread_dispatch, NULL));
    assert(!pthread_create(&b, NULL, thread_dispatch, NULL));
    assert(!pthread_join(a, NULL) && !pthread_join(b, NULL));
    assert(calls == 2 && counts().requested == 2 && counts().reason[WT_KF_LIVE_OMP] == 2);
    wt_cumulus_grid_end();
}
static void blocked_native(void **a)
{
    (void)a;
    assert(!pthread_mutex_lock(&gate));
    native_entered = 1;
    assert(!pthread_cond_broadcast(&entered));
    for (;;) assert(!pthread_cond_wait(&entered, &gate));
}
static void *staging_thread(void *unused)
{
    (void)unused;
    wt_kf_live_dispatch(args, blocked_native, WT_KF_NARG);
    return NULL;
}
static void concurrent_staging(void)
{
    pthread_t first;
    struct wt_kf_live_config config = configuration();
    initialize(); assert(wt_kf_live_configure(&config));
    wt_cumulus_grid_begin(backing);
    assert(!pthread_create(&first, NULL, staging_thread, NULL));
    assert(!pthread_mutex_lock(&gate));
    while (!native_entered) assert(!pthread_cond_wait(&entered, &gate));
    assert(!pthread_mutex_unlock(&gate));
    wt_kf_live_dispatch(args, no_read, WT_KF_NARG);
    _exit(99);
}
int main(void)
{
    struct rlimit limit = {0, 0};
    assert(!setrlimit(RLIMIT_CORE, &limit));
    shared_raincv = mmap(NULL, 128, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    assert(shared_raincv != MAP_FAILED);
    for (int mode = 0; mode <= 21; ++mode) {
        memset(shared_raincv, 0, 128);
        pid_t child = fork();
        int status;
        assert(child >= 0);
        if (!child) {
            if (mode == 17) threaded_refusal();
            else if (mode == 18) concurrent_staging();
            else scenario(mode);
            _exit(0);
        }
        assert(waitpid(child, &status, 0) == child);
        if ((mode >= 14 && mode <= 16) || (mode >= 18 && mode <= 19)) {
            assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
            if (mode == 15 || mode == 19) assert(shared_raincv[4] == 0.0f);
        }
        else { if (!WIFEXITED(status) || WEXITSTATUS(status)) fprintf(stderr,"failed mode %d\n",mode);
               assert(WIFEXITED(status) && WEXITSTATUS(status) == 0); }
    }
    assert(!munmap(shared_raincv, 128));
    puts("KF live scope checks passed");
    return 0;
}
