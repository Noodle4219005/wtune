#define _POSIX_C_SOURCE 200809L
#include "wt_kf_node_live.h"
#include "wt_kf_node_shm.h"
#include "wt_kf_capacity.h"
#include "wt_kf_batch_internal.h"
#include <omp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>

#ifndef WT_KF_BUILD_PROFILE_ID
#define WT_KF_BUILD_PROFILE_ID "unattested"
#endif
#ifndef WT_KF_BUILD_CONCURRENT_CPS_VERIFIED
#define WT_KF_BUILD_CONCURRENT_CPS_VERIFIED 0
#endif
static struct wt_kf_live_config config;
static struct wt_kf_node_live_options options;
static struct wt_kf_node_shm shm;
static wt_live_spans *registry;
static atomic_int configured,outer_active;
static atomic_uint readers;
static _Atomic uint64_t generation,scope_grid;
static _Thread_local unsigned depth;
static pthread_mutex_t native_mutex=PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t requested,donated,committed,self_executed,helper_executed;
static _Atomic uint64_t fallback,fatal,pack_ns,native_ns,progress_ns,owner_remote_committed;
static _Atomic uint64_t refusals[WT_KFN_REASON_COUNT],last_epoch;
static const char *setup_refusal="not-selected";
static int trace_enabled;
static atomic_uint trace_lines;
static void trace(const char *event,const struct wt_kf_node_context *c,
                  const struct wt_nl_ticket *ticket,uint32_t executor_rank)
{
    if (!trace_enabled || atomic_fetch_add(&trace_lines,1)>=64) return;
    fprintf(stderr,"# wt_kf_node_g0 event=%s pid=%lu local_rank=%u thread=%u owner_rank=%u owner_thread=%u executor_rank=%u work_id=%llu grid=%llu ktau=%llu profile=%llu init=%llu\n",
        event,(unsigned long)getpid(),c->rank,c->thread,ticket->owner_rank,ticket->owner_thread,executor_rank,
        (unsigned long long)ticket->work_id,(unsigned long long)c->signature.grid,
        (unsigned long long)c->signature.epoch,(unsigned long long)c->signature.profile,
        (unsigned long long)c->signature.init_generation);
}
static uint64_t clock_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) abort();
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static void die(void)
{ atomic_fetch_add(&fatal,1); abort(); }
static void real_call(void **argv,void (*invoke)(void **))
{
    uint64_t start;
    /* The pinned Intel CPS object eliminates lastdt. Concurrency still needs
     * the complete native callee/ABI proof; absent that build evidence, every
     * native entry (including fallback) serializes. No peer wait holds this lock. */
    if (!options.concurrency_verified && pthread_mutex_lock(&native_mutex)) die();
    start=clock_ns(); invoke(argv);
    atomic_fetch_add(&native_ns,clock_ns()-start);
    if (!options.concurrency_verified && pthread_mutex_unlock(&native_mutex)) die();
}
int wt_kf_node_live_active(void)
{ return atomic_load_explicit(&configured,memory_order_acquire)!=0; }
int wt_kf_node_live_configure(const struct wt_kf_live_config *c,
                            const struct wt_kf_node_live_options *o)
{
    int expected=0;
    if (!c || !o || !atomic_compare_exchange_strong(&configured,&expected,1)) return 0;
    config=*c; options=*o; setup_refusal="invalid-configuration";
    trace_enabled=getenv("WT_KF_NODE_G0_TRACE") && !strcmp(getenv("WT_KF_NODE_G0_TRACE"),"1");
    if (!config.enabled || !config.source_abi_verified || !config.caller_chain_verified ||
        !config.capture || !options.grid_identity || !options.profile ||
        !options.threads || options.threads>64 || !options.deadline_ns ||
        !options.slot_bytes || !options.shared_budget) goto refused;
    registry=wt_live_spans_create(config.max_spans);
    if (!registry) { setup_refusal="registry-allocation"; goto refused; }
    if (!options.service) {
        if (!wt_kf_node_shm_configure(&shm,options.threads,options.slot_bytes,options.shared_budget)) { setup_refusal="slurm-identity-or-shared-budget"; goto refused; }
        options.rank=shm.rank;
        if (wt_kf_node_shm_attach(&shm)<0) { setup_refusal="mapper-create-or-attach"; goto refused; }
        if (shm.registered) options.service=&shm.service;
        if (atexit(wt_kf_node_live_close)) goto refused;
    }
    setup_refusal="configured";
    atomic_store_explicit(&configured,2,memory_order_release); return 1;
 refused:
    atomic_store_explicit(&configured,3,memory_order_release); return 0;
}
static int env_size(const char *name,size_t default_value,size_t *out)
{
    const char *s=getenv(name); size_t n=0;
    if (!s) { *out=default_value; return 1; }
    if (!*s) return 0;
    for (;*s;++s) {
        unsigned digit=(unsigned)(*s-'0');
        if (digit>9 || n>(SIZE_MAX-digit)/10) return 0;
        n=n*10+digit;
    }
    if (!n) return 0;
    *out=n; return 1;
}
int wt_kf_node_live_from_env(const struct wt_kf_live_config *c,int (*identity)(void *))
{
    struct wt_kf_node_live_options o={0};
    const unsigned char *s=(const unsigned char *)WT_KF_BUILD_PROFILE_ID;
    size_t deadline;
    o.profile=UINT64_C(14695981039346656037);
    for (;*s;++s) { o.profile^=*s; o.profile*=UINT64_C(1099511628211); }
    if (!strcmp(WT_KF_BUILD_PROFILE_ID,"unattested") || !o.profile) return 0;
    if (!env_size("WT_KF_NODE_SLOT_BYTES",1024u*1024u,&o.slot_bytes) ||
        !env_size("WT_KF_NODE_SHARED_BUDGET",256u*1024u*1024u,&o.shared_budget) ||
        !env_size("WT_KF_NODE_DEADLINE_MS",30000,&deadline) || deadline>600000) return 0;
    o.deadline_ns=(uint64_t)deadline*1000000;
    o.threads=(uint32_t)omp_get_max_threads();
    o.grid_identity=identity;
    o.concurrency_verified=WT_KF_BUILD_CONCURRENT_CPS_VERIFIED;
    return wt_kf_node_live_configure(c,&o);
}
void wt_kf_node_live_begin(void *grid)
{
    uint64_t g; int tiles=0,id;
    if (depth || omp_in_parallel() || atomic_exchange(&outer_active,1) || atomic_load(&readers)) die();
    (void)wt_kf_init_reader_enter();
    atomic_store_explicit(&generation,0,memory_order_release);
    if (atomic_load_explicit(&configured,memory_order_acquire)!=2) return;
    if (!options.service && wt_kf_node_shm_attach(&shm)==1) options.service=&shm.service;
    if (!grid || !options.service || !wt_kf_init_fingerprint()) return;
    id=options.grid_identity(grid);
    if (id<=0) return;
    g=wt_live_spans_begin(registry);
    if (!g || config.capture(grid,registry,config.max_nodes,&tiles)!=1 || tiles<=0 ||
        wt_live_spans_finish(registry)!=WT_SPAN_OK) { wt_live_spans_end(registry); return; }
    atomic_store(&scope_grid,(uint64_t)id);
    atomic_store_explicit(&generation,g,memory_order_release);
}
void wt_kf_node_live_end(void)
{
    if (depth || omp_in_parallel() || !atomic_load(&outer_active)) die();
    atomic_store_explicit(&generation,0,memory_order_release);
    if (atomic_load(&readers)) die();
    if (registry) wt_live_spans_end(registry);
    atomic_store(&outer_active,0); wt_kf_init_reader_leave();
}
struct native_context { void (*invoke)(void **); uint64_t generation,init; };
static int native(void **argv,void *opaque)
{
    struct native_context *n=opaque;
    if (atomic_load_explicit(&generation,memory_order_acquire)!=n->generation ||
        !wt_live_spans_is_current(registry,n->generation) || wt_kf_init_fingerprint()!=n->init) die();
    real_call(argv,n->invoke);
    if (atomic_load_explicit(&generation,memory_order_acquire)!=n->generation ||
        !wt_live_spans_is_current(registry,n->generation) || wt_kf_init_fingerprint()!=n->init) die();
    return 0;
}
static void help(struct wt_kf_node_context *c,uint64_t deadline)
{
    struct wt_nl_ticket ticket;
    uint64_t begin=clock_ns();
    /* Finite scan budget, no drain-until-empty starvation. */
    for (unsigned i=0;i<options.threads*2;++i) {
        enum wt_nl_status s;
        if (clock_ns()>=deadline) break;
        s=wt_kf_node_progress(c,((uint64_t)c->rank<<32)|c->thread,&ticket);
        if (s==WT_NL_IDLE) break;
        if (s!=WT_NL_OK) die();
        atomic_fetch_add(&helper_executed,1);
        trace("helper",c,&ticket,c->rank);
    }
    atomic_fetch_add(&progress_ns,clock_ns()-begin);
}
static int payload_fits(void **argv)
{
    struct kfb_header h={0}; struct kfb_layout l;
    size_t used;
    h.schema=WT_KF_BATCH_SCHEMA; h.state=1;
    for (unsigned a=0;a<WT_KF_NARG;++a) if (argv[a]) {
        h.present[a]=1; h.alignment[a]=(unsigned char)((uintptr_t)argv[a]&63);
        if (wt_kf_shape[a]==WT_KF_SCALAR) memcpy(&h.scalar[a],argv[a],4);
    }
    /* wt_node_local version1 parts_for: two32-byte guards + geometry +
     * request + result. This reads no array bytes and allocates nothing. */
    return kfb_layout(&h,&l)==WT_KF_OK && kfb_mul(l.bytes,2,&used) &&
        kfb_add(used,64+18*sizeof(int32_t),&used) && used<=options.slot_bytes;
}
void wt_kf_node_live_dispatch(void **argv,void (*invoke)(void **),size_t nargs)
{
    struct wt_kf_capacity provider;
    struct wt_kf_node_call call={0};
    struct native_context n={invoke,0,0};
    struct wt_kf_node_context c={0};
    size_t capacity[WT_KF_NARG];
    void *tail[WT_KF_NARG],*head[WT_KF_NARG];
    int32_t jts,jte,tail_start,head_end,ktau;
    enum wt_kf_status bs;
    enum wt_nl_status status;
    enum wt_kf_node_refusal why=WT_KFN_CONFIG;
    int self_ran=0;
    uint64_t start,deadline;
    int thread=omp_get_thread_num(),team=omp_get_num_threads();
    if (!argv || !invoke || depth) die();
    ++depth; atomic_fetch_add(&requested,1); atomic_fetch_add(&readers,1);
    n.init=wt_kf_init_reader_enter();
    n.generation=atomic_load_explicit(&generation,memory_order_acquire);
    if (atomic_load_explicit(&configured,memory_order_acquire)!=2) goto direct;
    if (nargs!=WT_KF_NARG) { why=WT_KFN_ABI; goto direct; }
    if (!n.init) { why=WT_KFN_INIT; goto direct; }
    if (!options.service) { why=WT_KFN_MAPPER; goto direct; }
    if (!n.generation) { why=WT_KFN_SCOPE; goto direct; }
    if (thread<0 || team<=0 || (unsigned)thread>=options.threads ||
        (unsigned)team>options.threads || omp_get_active_level()>1) { why=WT_KFN_OMP; goto direct; }
    why=WT_KFN_CAPACITY;
    if (!wt_kf_capacity_init_attested(&provider,argv,registry,n.generation,
            config.source_abi_verified,config.caller_chain_verified)) goto direct;
    for (unsigned a=0;a<WT_KF_NARG;++a) {
        capacity[a]=0;
        if (argv[a] && !wt_kf_capacity_lookup(&provider,a,argv[a],&capacity[a])) {
            wt_kf_capacity_reset(&provider); goto direct;
        }
    }
    wt_kf_capacity_reset(&provider);
    memcpy(&jts,argv[WT_KF_JTS],4); memcpy(&jte,argv[WT_KF_JTE],4);
    memcpy(&ktau,argv[WT_KF_KTAU],4);
    if (ktau<0 || jts>jte) { why=WT_KFN_GEOMETRY; goto direct; }
    atomic_store(&last_epoch,(uint64_t)ktau);
    c.service=options.service; c.rank=options.rank; c.thread=(uint32_t)thread;
    c.byte_budget=config.byte_budget; c.native=native; c.native_context=&n;
    c.signature=(struct wt_nl_signature){WT_KF_NODE_PHASE,options.profile,WT_KF_BATCH_SCHEMA,
        atomic_load(&scope_grid),(uint64_t)ktau,1,n.init};
    start=clock_ns(); deadline=start+options.deadline_ns;
    if (deadline<start) die();
    if (jts==jte) { /* A one-row tile supplies useful work, then helps peers. */
        atomic_fetch_add(&fallback,1); atomic_fetch_add(&refusals[WT_KFN_SINGLE_ROW],1);
        native(argv,&n); help(&c,deadline); goto done;
    }
    tail_start=(int32_t)((int64_t)jts+((int64_t)jte-jts+1)/2);
    head_end=tail_start-1;
    memcpy(tail,argv,sizeof(tail)); memcpy(head,argv,sizeof(head));
    tail[WT_KF_JTS]=&tail_start; head[WT_KF_JTE]=&head_end;
    capacity[WT_KF_JTS]=4;
    if (!payload_fits(tail)) {
        tail_start=jte; head_end=jte-1;
        if (!payload_fits(tail)) { why=WT_KFN_SLOT; goto direct; }
    }
    status=wt_kf_node_publish(&c,tail,capacity,(uint64_t)(uint32_t)ktau,&call,&bs);
    if (status==WT_NL_REJECTED && bs==WT_KF_BUDGET && tail_start<jte) {
        /* Exactly one smaller pre-native attempt. A publish rejection has no
         * native side effect and leaves no published ticket or owner writes. */
        tail_start=jte; head_end=jte-1;
        status=wt_kf_node_publish(&c,tail,capacity,(uint64_t)(uint32_t)ktau,&call,&bs);
    }
    atomic_fetch_add(&pack_ns,clock_ns()-start);
    if (status!=WT_NL_OK) {
        why=bs==WT_KF_OVERLAP ? WT_KFN_ALIAS : bs==WT_KF_BUDGET ? WT_KFN_BUDGET :
            bs==WT_KF_CAPACITY ? WT_KFN_CAPACITY : bs==WT_KF_BAD_ARGUMENT ? WT_KFN_GEOMETRY :
            bs==WT_KF_OK ? WT_KFN_SLOT : WT_KFN_PREPARE;
        goto direct;
    }
    atomic_fetch_add(&donated,1);
    /* Real head work creates the donation opportunity. Nothing after this
     * native entry may replay the unsplit original tile. */
    native(head,&n);
    help(&c,deadline);
    status=wt_kf_node_collect(&c,&call);
    if (status==WT_NL_PENDING) {
        if (clock_ns()>=deadline) {
            (void)wt_nl_owner_poison(c.service,c.rank,c.thread,&call.ticket); die();
        }
        status=wt_kf_node_owner_execute(&c,&call,((uint64_t)c.rank<<32)|c.thread);
        if (status==WT_NL_OK) { self_ran=1; atomic_fetch_add(&self_executed,1); }
        else if (status!=WT_NL_PENDING && status!=WT_NL_BUSY) die();
        status=wt_kf_node_collect(&c,&call);
    }
    while (status==WT_NL_PENDING) {
        if (clock_ns()>=deadline) {
            (void)wt_nl_owner_poison(c.service,c.rank,c.thread,&call.ticket); die();
        }
        help(&c,deadline);
        status=wt_kf_node_collect(&c,&call);
        if (status==WT_NL_PENDING) sched_yield();
    }
    if (status!=WT_NL_OK) die();
    {
        uint64_t executor_tag;
        uint32_t executor_rank;
        if (wt_nl_ticket_executor(c.service,c.rank,c.thread,&call.ticket,&executor_tag)!=WT_NL_OK) die();
        executor_rank=(uint32_t)(executor_tag>>32);
        if ((self_ran && executor_rank!=c.rank) || (!self_ran && executor_rank==c.rank)) die();
        atomic_fetch_add(&committed,1);
        if (executor_rank!=c.rank) atomic_fetch_add(&owner_remote_committed,1);
        trace("commit",&c,&call.ticket,executor_rank);
    }
    if (wt_kf_node_finish(&c,&call)!=WT_NL_OK) die();
    wt_kf_node_destroy(&call);
    goto done;
 direct:
    atomic_fetch_add(&fallback,1); atomic_fetch_add(&refusals[why],1); real_call(argv,invoke);
 done:
    wt_kf_init_reader_leave(); atomic_fetch_sub(&readers,1); --depth;
}
void wt_kf_node_live_get_counters(struct wt_kf_node_live_counters *out)
{
    if (!out) return;
#define COPY(name) out->name=atomic_load(&name)
    COPY(requested); COPY(donated); COPY(committed); COPY(self_executed); COPY(helper_executed);
    COPY(fallback); COPY(fatal); COPY(pack_ns); COPY(native_ns); COPY(progress_ns); COPY(owner_remote_committed);
    for (unsigned i=0;i<WT_KFN_REASON_COUNT;++i) out->refusal[i]=atomic_load(&refusals[i]);
#undef COPY
}
void wt_kf_node_live_dump(FILE *stream)
{
    struct wt_kf_node_live_counters c;
    static const char *const reasons[WT_KFN_REASON_COUNT]={"config","abi","init","scope","mapper","omp","capacity","geometry","single_row","alias","budget","slot","prepare"};
    if (!stream) return;
    wt_kf_node_live_get_counters(&c);
    fprintf(stream,"# wt_kf_node_identity pid=%lu local_rank=%u grid=%llu last_ktau=%llu configuration=%s owner_remote_committed=%llu\n",
        (unsigned long)getpid(),options.rank,(unsigned long long)atomic_load(&scope_grid),
        (unsigned long long)atomic_load(&last_epoch),setup_refusal,(unsigned long long)c.owner_remote_committed);
    fprintf(stream,"# wt_kf_node mode=forced-g0 adaptive=pending serialized_native=%d profile=%llu init=%llu requested=%llu donated=%llu self=%llu helper=%llu committed=%llu fallback=%llu fatal=%llu pack_ns=%llu native_ns=%llu progress_ns=%llu\n",
        !options.concurrency_verified,(unsigned long long)options.profile,
        (unsigned long long)wt_kf_init_fingerprint(),(unsigned long long)c.requested,
        (unsigned long long)c.donated,(unsigned long long)c.self_executed,(unsigned long long)c.helper_executed,
        (unsigned long long)c.committed,(unsigned long long)c.fallback,(unsigned long long)c.fatal,
        (unsigned long long)c.pack_ns,(unsigned long long)c.native_ns,(unsigned long long)c.progress_ns);
    for (unsigned i=0;i<WT_KFN_REASON_COUNT;++i)
        fprintf(stream,"# wt_kf_node refusal.%s=%llu\n",reasons[i],(unsigned long long)c.refusal[i]);
}
void wt_kf_node_live_close(void)
{
    if (atomic_load(&outer_active) || atomic_load(&readers)) die();
    if (shm.registered) wt_kf_node_shm_close(&shm);
}
