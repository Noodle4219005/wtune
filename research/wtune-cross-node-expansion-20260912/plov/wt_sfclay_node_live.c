#define _POSIX_C_SOURCE 200809L
#include "wt_sfclay_node_live.h"
#include "wt_sfclay_live_internal.h"
#include "wt_sfclay_capacity.h"
#include "wt_kf_node_shm.h"
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifndef WT_SFCLAY_BUILD_PROFILE_ID
#define WT_SFCLAY_BUILD_PROFILE_ID "unattested"
#endif
#ifndef WT_SFCLAY_BUILD_CONCURRENT_VERIFIED
#define WT_SFCLAY_BUILD_CONCURRENT_VERIFIED 0
#endif
#ifndef WT_SFCLAY_GRID_EPOCH_F77
#define WT_SFCLAY_GRID_EPOCH_F77 wt_sfclay_grid_epoch_
#endif
extern void WT_SFCLAY_GRID_EPOCH_F77(void *grid,int *epoch,int *status);
struct scope_tls {
    uint64_t outer, surface, init_generation;
    int outer_lease;
    unsigned dispatch_depth;
};
static struct wt_sfclay_live_config config;
static struct wt_sfclay_node_live_options options;
static struct wt_kf_node_shm shm;
static wt_live_spans *registry;
static pthread_t control_owner;
static char control_marker;
static _Thread_local const void *control_identity;
static _Thread_local struct scope_tls scope_tls;
static pthread_mutex_t native_mutex=PTHREAD_MUTEX_INITIALIZER;
static atomic_int configured,outer_active,surface_active;
static atomic_uint dispatch_readers;
static _Atomic uint64_t tokens,scope_generation,scope_grid,scope_epoch;
static _Atomic uint64_t call_sequence,last_epoch;
static _Atomic uint64_t requested,donated,committed,self_executed,helper_executed;
static _Atomic uint64_t fallback,fatal,pack_ns,native_ns,progress_ns;
static _Atomic uint64_t owner_remote_committed,refusals[WT_SFN_REASON_COUNT];
static _Atomic uint64_t shadow_checked,shadow_failed,shadow_bytes;
static const char *setup_refusal="not-selected";
static int trace_enabled;
static atomic_uint trace_lines;
static uint64_t clock_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) abort();
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static _Noreturn void die_at(const char *function,int line)
{
    atomic_fetch_add_explicit(&fatal,1,memory_order_relaxed);
    if (trace_enabled)
        fprintf(stderr,"# wt_sfclay_node_g0 fatal function=%s line=%d\n",
                function,line);
    abort();
}
#define die() die_at(__func__,__LINE__)
static uint64_t next_token(void)
{
    uint64_t old=atomic_load_explicit(&tokens,memory_order_relaxed);
    do {
        if (old==UINT64_MAX) die();
    } while (!atomic_compare_exchange_weak_explicit(&tokens,&old,old+1,
                memory_order_relaxed,memory_order_relaxed));
    return old+1;
}
static uint64_t next_call(void)
{
    uint64_t old=atomic_load_explicit(&call_sequence,memory_order_relaxed);
    do {
        if (old==UINT64_MAX) die();
    } while (!atomic_compare_exchange_weak_explicit(&call_sequence,&old,old+1,
                memory_order_relaxed,memory_order_relaxed));
    return old+1;
}
static int is_control_owner(void)
{
    return control_identity==&control_marker &&
           pthread_equal(control_owner,pthread_self());
}
static void trace(const char *event,const struct wt_sfclay_node_context *c,
                  const struct wt_nl_ticket *ticket,uint32_t executor_rank)
{
    if (!trace_enabled || atomic_fetch_add(&trace_lines,1)>=64) return;
    fprintf(stderr,"# wt_sfclay_node_g0 event=%s pid=%lu local_rank=%u thread=%u owner_rank=%u owner_thread=%u executor_rank=%u work_id=%llu phase=%llu grid=%llu epoch=%llu profile=%llu init=%llu\n",
        event,(unsigned long)getpid(),c->rank,c->thread,ticket->owner_rank,
        ticket->owner_thread,executor_rank,(unsigned long long)ticket->work_id,
        (unsigned long long)c->signature.phase,
        (unsigned long long)c->signature.grid,
        (unsigned long long)c->signature.epoch,
        (unsigned long long)c->signature.profile,
        (unsigned long long)c->signature.init_generation);
}
static void real_call(void **argv,void (*invoke)(void **))
{
    uint64_t start;
    if (!options.concurrency_verified && pthread_mutex_lock(&native_mutex)) die();
    start=clock_ns();
    invoke(argv);
    atomic_fetch_add(&native_ns,clock_ns()-start);
    if (!options.concurrency_verified && pthread_mutex_unlock(&native_mutex)) die();
}
int wt_sfclay_node_live_active(void)
{
    return atomic_load_explicit(&configured,memory_order_acquire)==2;
}
int wt_sfclay_node_live_configure(const struct wt_sfclay_live_config *c,
    const struct wt_sfclay_node_live_options *o)
{
    int expected=0;
    if (!c || !o || !atomic_compare_exchange_strong(&configured,&expected,1))
        return 0;
    config=*c; options=*o; setup_refusal="invalid-configuration";
    trace_enabled=getenv("WT_SFCLAY_NODE_G0_TRACE") &&
        !strcmp(getenv("WT_SFCLAY_NODE_G0_TRACE"),"1");
    if (config.enabled!=1 || config.source_abi_verified!=1 ||
        config.caller_chain_verified!=1 ||
        config.initializer_relocation_verified!=1 || !config.capture ||
        !config.omp_query || !config.native_query || !config.max_spans ||
        !config.max_nodes || !config.byte_budget || !options.profile ||
        !options.threads || options.threads>64 || !options.deadline_ns ||
        !options.slot_bytes || !options.shared_budget ||
        (options.concurrency_verified!=0 && options.concurrency_verified!=1) ||
        (options.shadow_verify_remote!=0 && options.shadow_verify_remote!=1) ||
        (options.owner_only!=0 && options.owner_only!=1) ||
        (options.shadow_verify_remote && options.owner_only))
        goto refused;
    registry=wt_live_spans_create(config.max_spans);
    if (!registry) { setup_refusal="registry-allocation"; goto refused; }
    if (!options.service) {
        if (!wt_kf_node_shm_configure_phase(&shm,WT_SFCLAY_NODE_PHASE,
                options.threads,options.slot_bytes,options.shared_budget)) {
            setup_refusal="slurm-identity-or-shared-budget"; goto refused;
        }
        options.rank=shm.rank;
        if (wt_kf_node_shm_attach(&shm)<0) {
            setup_refusal="mapper-create-or-attach"; goto refused;
        }
        if (shm.registered) options.service=&shm.service;
        if (atexit(wt_sfclay_node_live_close)) goto refused;
    }
    control_owner=pthread_self(); control_identity=&control_marker;
    setup_refusal="configured";
    atomic_store_explicit(&configured,2,memory_order_release);
    return 1;
refused:
    atomic_store_explicit(&configured,3,memory_order_release);
    return 0;
}
static int env_size(const char *name,size_t fallback,size_t *out)
{
    const char *s=getenv(name); size_t n=0;
    if (!s) { *out=fallback; return 1; }
    if (!*s) return 0;
    for (;*s;++s) {
        unsigned digit=(unsigned)(*s-'0');
        if (digit>9 || n>(SIZE_MAX-digit)/10) return 0;
        n=n*10+digit;
    }
    if (!n) return 0;
    *out=n; return 1;
}
int wt_sfclay_node_live_from_env(const struct wt_sfclay_live_config *c)
{
    struct wt_sfclay_node_live_options o={0};
    const unsigned char *s=(const unsigned char *)WT_SFCLAY_BUILD_PROFILE_ID;
    const char *shadow=getenv("WT_SFCLAY_NODE_G0_SHADOW");
    const char *owner_only=getenv("WT_SFCLAY_NODE_G0_OWNER_ONLY");
    size_t deadline;
    o.profile=UINT64_C(14695981039346656037);
    for (;*s;++s) { o.profile^=*s; o.profile*=UINT64_C(1099511628211); }
    if (!strcmp(WT_SFCLAY_BUILD_PROFILE_ID,"unattested") || !o.profile) return 0;
    if (!env_size("WT_SFCLAY_NODE_SLOT_BYTES",1024u*1024u,&o.slot_bytes) ||
        !env_size("WT_SFCLAY_NODE_SHARED_BUDGET",256u*1024u*1024u,
                  &o.shared_budget) ||
        !env_size("WT_SFCLAY_NODE_DEADLINE_MS",30000,&deadline) ||
        deadline>600000) return 0;
    o.deadline_ns=(uint64_t)deadline*UINT64_C(1000000);
    o.threads=(uint32_t)omp_get_max_threads();
    o.concurrency_verified=WT_SFCLAY_BUILD_CONCURRENT_VERIFIED;
    if (shadow && strcmp(shadow,"0") && strcmp(shadow,"1")) return 0;
    if (owner_only && strcmp(owner_only,"0") && strcmp(owner_only,"1")) return 0;
    o.shadow_verify_remote=shadow && !strcmp(shadow,"1");
    o.owner_only=owner_only && !strcmp(owner_only,"1");
    return wt_sfclay_node_live_configure(c,&o);
}
uint64_t wt_sfclay_node_live_first_rk_begin(void *grid,void *cfg)
{
    struct wt_sfclay_native_result result={0};
    uint64_t token,g=0,init=0;
    int captured=0,tiles=0,actual_epoch=0,epoch_status=1;
    if (!is_control_owner() || scope_tls.outer || scope_tls.surface ||
        scope_tls.dispatch_depth || omp_in_parallel() ||
        atomic_exchange(&outer_active,1) || atomic_load(&dispatch_readers)) die();
    wt_sf_activity_enter();
    scope_tls.outer=token=next_token();
    scope_tls.outer_lease=1;
    (void)wt_sf_reader_get(&scope_tls.init_generation);
    atomic_store_explicit(&scope_generation,0,memory_order_release);
    if (!options.service && wt_kf_node_shm_attach(&shm)==1)
        options.service=&shm.service;
    init=wt_sfclay_init_fingerprint();
    if (!grid || !cfg || !options.service || !init || config.omp_query())
        return token;
    wt_sf_tls.control_depth=1;
    captured=config.native_query(grid,cfg,&result);
    wt_sf_tls.control_depth=0;
    if (captured!=1 || result.status!=WT_SFCLAY_NATIVE_OK ||
        result.grid_id<=0 || result.actual_scheme!=result.expected_scheme ||
        result.actual_scheme!=(int)WT_SFCLAY_NODE_BRANCH ||
        result.actual_fractional!=0) return token;
    WT_SFCLAY_GRID_EPOCH_F77(grid,&actual_epoch,&epoch_status);
    if (epoch_status || actual_epoch<=0) return token;
    g=wt_live_spans_begin(registry);
    if (!g) return token;
    wt_sf_tls.control_depth=1;
    captured=config.capture(grid,registry,config.max_nodes,&tiles);
    wt_sf_tls.control_depth=0;
    if (captured!=1 || tiles<=0 ||
        wt_live_spans_finish(registry)!=WT_SPAN_OK) {
        wt_live_spans_end(registry); return token;
    }
    atomic_store(&scope_grid,(uint64_t)result.grid_id);
    atomic_store(&scope_epoch,(uint64_t)(uint32_t)actual_epoch);
    atomic_store_explicit(&scope_generation,g,memory_order_release);
    return token;
}
void wt_sfclay_node_live_first_rk_end(uint64_t token)
{
    uint64_t g;
    if (!is_control_owner() || !token || token!=scope_tls.outer ||
        scope_tls.surface || scope_tls.dispatch_depth || omp_in_parallel()) die();
    atomic_store_explicit(&scope_generation,0,memory_order_release);
    if (atomic_load(&dispatch_readers)) die();
    g=scope_tls.outer;
    (void)g;
    if (registry) wt_live_spans_end(registry);
    atomic_store(&surface_active,0);
    atomic_store(&outer_active,0);
    scope_tls.outer=0;
    if (!scope_tls.outer_lease) die();
    scope_tls.outer_lease=0;
    wt_sf_reader_put();
    scope_tls.init_generation=0;
    wt_sf_activity_leave();
}
uint64_t wt_sfclay_node_live_surface_begin(void)
{
    uint64_t token;
    if (!is_control_owner() || !scope_tls.outer || scope_tls.surface ||
        scope_tls.dispatch_depth || omp_in_parallel() ||
        atomic_exchange(&surface_active,1)) die();
    wt_sf_activity_enter();
    token=next_token(); scope_tls.surface=token;
    return token;
}
void wt_sfclay_node_live_surface_end(uint64_t token)
{
    if (!is_control_owner() || !token || token!=scope_tls.surface ||
        scope_tls.dispatch_depth || omp_in_parallel() ||
        atomic_load(&dispatch_readers)) die();
    scope_tls.surface=0;
    atomic_store(&surface_active,0);
    wt_sf_activity_leave();
}
struct native_context {
    void (*invoke)(void **);
    uint64_t generation,init,lease_generation;
};
static int native(void **argv,void *opaque)
{
    struct native_context *n=opaque;
    if (atomic_load_explicit(&scope_generation,memory_order_acquire)!=n->generation ||
        !wt_live_spans_is_current(registry,n->generation) ||
        wt_sfclay_init_fingerprint()!=n->init ||
        !wt_sf_reader_current(n->lease_generation)) die();
    real_call(argv,n->invoke);
    if (atomic_load_explicit(&scope_generation,memory_order_acquire)!=n->generation ||
        !wt_live_spans_is_current(registry,n->generation) ||
        wt_sfclay_init_fingerprint()!=n->init ||
        !wt_sf_reader_current(n->lease_generation)) die();
    return 0;
}
static void help(struct wt_sfclay_node_context *c,uint64_t deadline)
{
    struct wt_nl_ticket ticket;
    uint64_t begin=clock_ns();
    if (options.owner_only) return;
    for (unsigned i=0;i<options.threads*2;++i) {
        enum wt_nl_status status;
        if (clock_ns()>=deadline) break;
        status=wt_sfclay_node_progress(c,((uint64_t)c->rank<<32)|c->thread,
                                       &ticket);
        if (status==WT_NL_IDLE) break;
        if (status!=WT_NL_OK) die();
        atomic_fetch_add(&helper_executed,1);
        trace("helper",c,&ticket,c->rank);
    }
    atomic_fetch_add(&progress_ns,clock_ns()-begin);
}
static enum wt_nl_status collect_ready(struct wt_sfclay_node_context *c,
    struct wt_sfclay_node_call *call,uint32_t *executor_rank)
{
    uint64_t tag;
    enum wt_nl_status status=wt_nl_ticket_executor(c->service,c->rank,c->thread,
                                                    &call->ticket,&tag);
    if (status!=WT_NL_OK) return status;
    *executor_rank=(uint32_t)(tag>>32);
    c->shadow_remote=options.shadow_verify_remote && *executor_rank!=c->rank;
    status=wt_sfclay_node_collect(c,call);
    if (call->shadow_checked) {
        atomic_fetch_add(&shadow_checked,1);
        atomic_fetch_add(&shadow_bytes,call->shadow_bytes);
        if (call->shadow_failed) atomic_fetch_add(&shadow_failed,1);
    }
    return status;
}
void wt_sfclay_node_live_dispatch(void **argv,void (*invoke)(void **),
                                  size_t nargs)
{
    struct wt_sfclay_capacity provider;
    struct wt_sfclay_node_call call={0};
    struct native_context n={.invoke=invoke};
    struct wt_sfclay_node_context c={0};
    size_t capacity[WT_SFCLAY_NARG];
    void *tail[WT_SFCLAY_NARG],*head[WT_SFCLAY_NARG];
    int32_t jts,jte,tail_jts,head_jte;
    enum wt_sfclay_batch_status batch_status;
    enum wt_nl_status status;
    enum wt_sfclay_node_refusal why=WT_SFN_CONFIG;
    uint64_t start,deadline,sequence;
    int self_ran=0,reader_acquired=0;
    uint32_t executor_rank=UINT32_MAX;
    int thread=omp_get_thread_num(),team=omp_get_num_threads();
    if (!argv || !invoke || scope_tls.dispatch_depth) die();
    ++scope_tls.dispatch_depth; wt_sf_activity_enter();
    atomic_fetch_add(&requested,1); atomic_fetch_add(&dispatch_readers,1);
    if (wt_sf_tls.reader_depth) {
        /* The OpenMP master retains the first-RK control thread's lease. */
        if (!is_control_owner() || !scope_tls.outer_lease) die();
        n.lease_generation=scope_tls.init_generation;
    } else {
        (void)wt_sf_reader_get(&n.lease_generation);
        reader_acquired=1;
    }
    n.init=wt_sfclay_init_fingerprint();
    n.generation=atomic_load_explicit(&scope_generation,memory_order_acquire);
    if (atomic_load_explicit(&configured,memory_order_acquire)!=2) goto direct;
    if (nargs!=WT_SFCLAY_NARG) { why=WT_SFN_ABI; goto direct; }
    if (!n.init) { why=WT_SFN_INIT; goto direct; }
    if (!n.generation || !atomic_load(&outer_active) ||
        !atomic_load(&surface_active)) { why=WT_SFN_SCOPE; goto direct; }
    if (!options.service) { why=WT_SFN_MAPPER; goto direct; }
    if (thread<0 || team<=0 || (unsigned)thread>=options.threads ||
        (unsigned)team>options.threads || omp_get_active_level()>1) {
        why=WT_SFN_OMP; goto direct;
    }
    why=WT_SFN_CAPACITY;
    if (!wt_sfclay_capacity_init_attested(&provider,argv,registry,n.generation,
            config.source_abi_verified,config.caller_chain_verified)) goto direct;
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a) {
        capacity[a]=0;
        if (argv[a] &&
            !wt_sfclay_capacity_lookup(&provider,a,argv[a],&capacity[a])) {
            wt_sfclay_capacity_reset(&provider); goto direct;
        }
    }
    wt_sfclay_capacity_reset(&provider);
    memcpy(&jts,argv[72],4); memcpy(&jte,argv[73],4);
    if (jts>jte) { why=WT_SFN_GEOMETRY; goto direct; }
    c.service=options.service; c.rank=options.rank; c.thread=(uint32_t)thread;
    c.byte_budget=config.byte_budget; c.native=native; c.native_context=&n;
    c.signature=(struct wt_nl_signature){WT_SFCLAY_NODE_PHASE,options.profile,
        WT_SFCLAY_BATCH_SCHEMA,atomic_load(&scope_grid),atomic_load(&scope_epoch),
        WT_SFCLAY_NODE_BRANCH,n.init};
    atomic_store(&last_epoch,c.signature.epoch);
    start=clock_ns(); deadline=start+options.deadline_ns;
    if (deadline<start) die();
    if (jts==jte) {
        atomic_fetch_add(&fallback,1);
        atomic_fetch_add(&refusals[WT_SFN_SINGLE_ROW],1);
        native(argv,&n); help(&c,deadline); goto done;
    }
    tail_jts=jte; head_jte=jte-1;
    memcpy(tail,argv,sizeof(tail)); memcpy(head,argv,sizeof(head));
    tail[72]=&tail_jts; head[73]=&head_jte;
    capacity[72]=capacity[73]=4;
    sequence=next_call();
    status=wt_sfclay_node_publish(&c,tail,capacity,sequence,&call,&batch_status);
    atomic_fetch_add(&pack_ns,clock_ns()-start);
    if (status!=WT_NL_OK) {
        why=batch_status==WT_SFCLAY_BATCH_OVERLAP ? WT_SFN_ALIAS :
            batch_status==WT_SFCLAY_BATCH_BUDGET ? WT_SFN_BUDGET :
            batch_status==WT_SFCLAY_BATCH_CAPACITY ? WT_SFN_CAPACITY :
            batch_status==WT_SFCLAY_BATCH_BOUNDS ? WT_SFN_GEOMETRY :
            status==WT_NL_OVERFLOW ? WT_SFN_SLOT : WT_SFN_PREPARE;
        goto direct;
    }
    atomic_fetch_add(&donated,1);
    native(head,&n);
    help(&c,deadline);
    status=collect_ready(&c,&call,&executor_rank);
    while (status==WT_NL_PENDING) {
        if (clock_ns()>=deadline) {
            (void)wt_nl_owner_poison(c.service,c.rank,c.thread,&call.ticket);
            die();
        }
        status=wt_sfclay_node_owner_execute(&c,&call,
                    ((uint64_t)c.rank<<32)|c.thread);
        if (status==WT_NL_OK) {
            self_ran=1; atomic_fetch_add(&self_executed,1);
        } else if (status!=WT_NL_PENDING && status!=WT_NL_BUSY) die();
        help(&c,deadline);
        status=collect_ready(&c,&call,&executor_rank);
        if (status==WT_NL_PENDING) sched_yield();
    }
    if (status!=WT_NL_OK) die();
    {
        uint64_t tag;
        if (wt_nl_ticket_executor(c.service,c.rank,c.thread,&call.ticket,&tag)
                !=WT_NL_OK) die();
        if (executor_rank!=(uint32_t)(tag>>32)) die();
        if ((self_ran && executor_rank!=c.rank) ||
            (!self_ran && executor_rank==c.rank)) die();
        atomic_fetch_add(&committed,1);
        if (executor_rank!=c.rank) atomic_fetch_add(&owner_remote_committed,1);
        trace("commit",&c,&call.ticket,executor_rank);
    }
    if (wt_sfclay_node_finish(&c,&call)!=WT_NL_OK) die();
    wt_sfclay_node_destroy(&call);
    goto done;
direct:
    atomic_fetch_add(&fallback,1); atomic_fetch_add(&refusals[why],1);
    real_call(argv,invoke);
done:
    if (reader_acquired) wt_sf_reader_put();
    atomic_fetch_sub(&dispatch_readers,1); wt_sf_activity_leave();
    --scope_tls.dispatch_depth;
}
void wt_sfclay_node_live_get_counters(struct wt_sfclay_node_live_counters *out)
{
    if (!out) return;
#define COPY(name) out->name=atomic_load(&name)
    COPY(requested); COPY(donated); COPY(committed); COPY(self_executed);
    COPY(helper_executed); COPY(fallback); COPY(fatal); COPY(pack_ns);
    COPY(native_ns); COPY(progress_ns); COPY(owner_remote_committed);
    COPY(shadow_checked); COPY(shadow_failed); COPY(shadow_bytes);
    for (unsigned i=0;i<WT_SFN_REASON_COUNT;++i)
        out->refusal[i]=atomic_load(&refusals[i]);
#undef COPY
}
void wt_sfclay_node_live_dump(FILE *stream)
{
    struct wt_sfclay_node_live_counters c;
    static const char *const reason[WT_SFN_REASON_COUNT]={"config","abi","init",
        "scope","mapper","omp","capacity","geometry","single_row","alias",
        "budget","slot","prepare","native_query"};
    if (!stream) return;
    wt_sfclay_node_live_get_counters(&c);
    fprintf(stream,"# wt_sfclay_node_identity pid=%lu local_rank=%u phase=%llu grid=%llu epoch=%llu configuration=%s owner_remote_committed=%llu\n",
        (unsigned long)getpid(),options.rank,(unsigned long long)WT_SFCLAY_NODE_PHASE,
        (unsigned long long)atomic_load(&scope_grid),
        (unsigned long long)atomic_load(&last_epoch),setup_refusal,
        (unsigned long long)c.owner_remote_committed);
    fprintf(stream,"# wt_sfclay_node mode=%s adaptive=pending serialized_native=%d shadow_remote=%d scheme=91 fractional_seaice=0 profile=%llu init=%llu requested=%llu donated=%llu self=%llu helper=%llu committed=%llu fallback=%llu fatal=%llu pack_ns=%llu native_ns=%llu progress_ns=%llu shadow_checked=%llu shadow_failed=%llu shadow_bytes=%llu\n",
        options.owner_only ? "forced-g0-owner-only" : "forced-g0",
        !options.concurrency_verified,options.shadow_verify_remote,
        (unsigned long long)options.profile,
        (unsigned long long)wt_sfclay_init_fingerprint(),
        (unsigned long long)c.requested,(unsigned long long)c.donated,
        (unsigned long long)c.self_executed,(unsigned long long)c.helper_executed,
        (unsigned long long)c.committed,(unsigned long long)c.fallback,
        (unsigned long long)c.fatal,(unsigned long long)c.pack_ns,
        (unsigned long long)c.native_ns,(unsigned long long)c.progress_ns,
        (unsigned long long)c.shadow_checked,
        (unsigned long long)c.shadow_failed,
        (unsigned long long)c.shadow_bytes);
    for (unsigned i=0;i<WT_SFN_REASON_COUNT;++i)
        fprintf(stream,"# wt_sfclay_node refusal.%s=%llu\n",reason[i],
                (unsigned long long)c.refusal[i]);
}
void wt_sfclay_node_live_close(void)
{
    if (atomic_load(&outer_active) || atomic_load(&surface_active) ||
        atomic_load(&dispatch_readers)) die();
    if (shm.registered) wt_kf_node_shm_close(&shm);
}
