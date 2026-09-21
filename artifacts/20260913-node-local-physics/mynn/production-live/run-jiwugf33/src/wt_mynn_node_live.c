#define _POSIX_C_SOURCE 200809L
#include "wt_mynn_node_live.h"
#include "wt_mynn_native.h"
#include "wt_kf_node_shm.h"
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static struct wt_mynn_node_live_config cfg;
static struct wt_kf_node_shm mapper;
static wt_live_spans *spans;
static atomic_int configured,outer_active,init_mode;
static atomic_uint readers,trace_lines;
static _Atomic uint64_t generation,grid_id,epoch_id,static_id;
static _Thread_local unsigned depth;
static pthread_mutex_t native_mutex=PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t requested,donated,committed,self_executed,helper_executed;
static _Atomic uint64_t remote_committed,fallback,fatal,pack_ns,native_ns,progress_ns;
static _Atomic uint64_t shadow_checked,shadow_failed,shadow_bytes;
static _Atomic uint64_t refusals[WT_MLN_REASON_COUNT];
static int tracing;
static uint64_t now(void)
{
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) abort();
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static void die(void)
{ atomic_fetch_add(&fatal,1); abort(); }
static void real_call(void **a,void (*invoke)(void **))
{
    uint64_t begin;
    if(!cfg.concurrency_verified && pthread_mutex_lock(&native_mutex)) die();
    begin=now(); invoke(a); atomic_fetch_add(&native_ns,now()-begin);
    if(!cfg.concurrency_verified && pthread_mutex_unlock(&native_mutex)) die();
}
static void trace(const char *event,const struct wt_mynn_node_context *c,
                  const struct wt_nl_ticket *ticket,uint32_t executor)
{
    if(!tracing || atomic_fetch_add(&trace_lines,1)>=64) return;
    fprintf(stderr,"# wt_mynn_node_g0 event=%s pid=%lu rank=%u thread=%u owner=%u owner_thread=%u executor=%u work=%llu grid=%llu epoch=%llu init=%llu\n",
        event,(unsigned long)getpid(),c->rank,c->thread,ticket->owner_rank,ticket->owner_thread,executor,
        (unsigned long long)ticket->work_id,(unsigned long long)c->signature.grid,
        (unsigned long long)c->signature.epoch,(unsigned long long)c->signature.init_generation);
}
int wt_mynn_node_live_configure(const struct wt_mynn_node_live_config *c)
{
    int expected=0;
    if(!c || !atomic_compare_exchange_strong(&configured,&expected,1)) return 0;
    cfg=*c;
    if(cfg.enabled!=1 || cfg.abi_verified!=1 || cfg.caller_verified!=1 || cfg.static_state_verified!=1 ||
       (cfg.concurrency_verified!=0 && cfg.concurrency_verified!=1) ||
       (cfg.owner_only!=0 && cfg.owner_only!=1) ||
       (cfg.shadow_verify_remote!=0 && cfg.shadow_verify_remote!=1) ||
       (cfg.owner_only && cfg.shadow_verify_remote) ||
       !cfg.profile || !cfg.capture || !cfg.scope || !cfg.fingerprint ||
       !cfg.byte_budget || !cfg.max_spans || !cfg.max_nodes || cfg.grid_selector<0 ||
       !cfg.threads || cfg.threads>64 || !cfg.deadline_ns ||
       !cfg.slot_bytes || !cfg.shared_budget ||
       (cfg.logical_true_bits!=1 && cfg.logical_true_bits!=UINT32_MAX)) return 0;
    spans=wt_live_spans_create(cfg.max_spans);
    if(!spans) return 0;
    if(!cfg.service) {
        if(!wt_kf_node_shm_configure_phase(&mapper,WT_MYNN_NODE_PHASE,cfg.threads,
                                          cfg.slot_bytes,cfg.shared_budget)) goto refused;
        cfg.rank=mapper.rank;
        if(wt_kf_node_shm_attach(&mapper)<0) goto refused;
        if(mapper.registered) cfg.service=&mapper.service;
    }
    tracing=getenv("WT_MYNN_NODE_G0_TRACE") && !strcmp(getenv("WT_MYNN_NODE_G0_TRACE"),"1");
    if(atexit(wt_mynn_node_live_close)) goto refused;
    atomic_store_explicit(&configured,2,memory_order_release); return 1;
 refused:
    wt_mynn_node_live_close(); return 0;
}
uint64_t wt_mynn_first_rk_begin(void *grid,void *native_cfg)
{
    uint64_t id,epoch,token,fingerprint;
    int tiles=0;
    if(atomic_load_explicit(&configured,memory_order_acquire)!=2) return 0;
    if(depth || omp_in_parallel() || atomic_exchange(&outer_active,1) || atomic_load(&readers)) die();
    atomic_store(&generation,0); atomic_store(&init_mode,0);
    if(!cfg.service && wt_kf_node_shm_attach(&mapper)==1) cfg.service=&mapper.service;
    if(!cfg.service || !grid || !native_cfg || !cfg.scope(grid,native_cfg,&id,&epoch) ||
       !id || !epoch || (cfg.grid_selector && id!=(uint64_t)cfg.grid_selector)) goto refused;
    fingerprint=cfg.fingerprint();
    if(!fingerprint) goto refused;
    token=wt_live_spans_begin(spans);
    if(!token || !cfg.capture(grid,spans,cfg.max_nodes,&tiles) || tiles<=0 ||
       wt_live_spans_finish(spans)!=WT_SPAN_OK) goto refused;
    atomic_store(&grid_id,id); atomic_store(&epoch_id,epoch); atomic_store(&static_id,fingerprint);
    atomic_store_explicit(&generation,token,memory_order_release);
    return token;
 refused:
    wt_live_spans_end(spans); atomic_store(&outer_active,0); return 0;
}
void wt_mynn_first_rk_end(uint64_t token)
{
    if(!token) return;
    if(depth || omp_in_parallel() || !atomic_load(&outer_active) || atomic_load(&readers) ||
       token!=atomic_load(&generation) || cfg.fingerprint()!=atomic_load(&static_id)) die();
    atomic_store_explicit(&generation,0,memory_order_release);
    wt_live_spans_end(spans); atomic_store(&outer_active,0);
}
struct native_context { void (*invoke)(void **); uint64_t token,init; };
static int native(void *opaque,void *a[WT_MYNN_NARG])
{
    struct native_context *c=opaque;
    if(c->token!=atomic_load(&generation) || !wt_live_spans_is_current(spans,c->token) ||
       c->init!=cfg.fingerprint()) die();
    real_call(a,c->invoke);
    if(c->token!=atomic_load(&generation) || c->init!=cfg.fingerprint()) die();
    return 0;
}
static void help(struct wt_mynn_node_context *c,uint64_t deadline)
{
    uint64_t begin=now(); struct wt_nl_ticket ticket;
    if(cfg.owner_only) return;
    for(unsigned n=0;n<cfg.threads*2 && now()<deadline;++n) {
        enum wt_nl_status s=wt_mynn_node_progress(c,((uint64_t)c->rank<<32)|c->thread,&ticket);
        if(s==WT_NL_IDLE) break;
        if(s!=WT_NL_OK) die();
        atomic_fetch_add(&helper_executed,1); trace("helper",c,&ticket,c->rank);
    }
    atomic_fetch_add(&progress_ns,now()-begin);
}
static enum wt_nl_status collect_ready(struct wt_mynn_node_context *c,
    struct wt_mynn_node_call *call,uint32_t *executor_rank)
{
    uint64_t tag;
    enum wt_nl_status status=wt_nl_ticket_executor(c->service,c->rank,c->thread,
                                                    &call->ticket,&tag);
    if(status!=WT_NL_OK) return status;
    *executor_rank=(uint32_t)(tag>>32);
    call->shadow_remote=cfg.shadow_verify_remote && *executor_rank!=c->rank;
    status=wt_mynn_node_collect(c,call);
    if(call->shadow_checked) {
        atomic_fetch_add(&shadow_checked,1);
        atomic_fetch_add(&shadow_bytes,call->shadow_bytes);
        if(call->shadow_failed) atomic_fetch_add(&shadow_failed,1);
    }
    return status;
}
void wt_mynn_bridge_dispatch(void **argv,void (*invoke)(void **),size_t nargs)
{
    struct native_context n={invoke,0,0};
    struct wt_mynn_node_context c={0};
    struct wt_mynn_node_call call={0};
    struct wt_mynn_layout_config layout={1,1,0,0};
    struct wt_mynn_layout view;
    alignas(64) unsigned char scalar[WT_MYNN_NARG][128];
    void *tail[WT_MYNN_NARG],*head[WT_MYNN_NARG];
    size_t cap[WT_MYNN_NARG];
    enum wt_mynn_status bs=WT_MYNN_ARGUMENT;
    enum wt_nl_status s;
    enum wt_mynn_live_refusal why=WT_MLN_CONFIG;
    int32_t jts,jte,head_end,init;
    uint64_t begin,deadline,tag;
    uint32_t executor_rank=UINT32_MAX;
    int thread=omp_get_thread_num(),team=omp_get_num_threads(),self=0,expected;
    if(!argv || !invoke || depth) die();
    ++depth; atomic_fetch_add(&readers,1); atomic_fetch_add(&requested,1);
    if(atomic_load_explicit(&configured,memory_order_acquire)!=2) goto direct;
    if(nargs!=WT_MYNN_NARG) { why=WT_MLN_ABI; goto direct; }
    n.token=atomic_load_explicit(&generation,memory_order_acquire);
    if(!cfg.service) { why=WT_MLN_MAPPER; goto direct; }
    if(!n.token) { why=WT_MLN_SCOPE; goto direct; }
    n.init=atomic_load(&static_id);
    if(!n.init || n.init!=cfg.fingerprint()) { why=WT_MLN_STATIC; goto direct; }
    if(thread<0 || team<1 || (unsigned)thread>=cfg.threads || (unsigned)team>cfg.threads || omp_get_active_level()>1) {
        why=WT_MLN_OMP; goto direct;
    }
    if(!wt_mynn_native_capacities(argv,spans,n.token,&cfg,cap)) { why=WT_MLN_CAPACITY; goto direct; }
    memcpy(&init,argv[WT_MYNN_INITFLAG],4);
    expected=0;
    if(!atomic_compare_exchange_strong(&init_mode,&expected,init==0 ? 2 : 1) && expected!=(init==0 ? 2 : 1)) die();
    /* Scientific initialization writes owner argument arrays, not mutable LUTs.
     * All same-scope init calls run original native under the native entry lease;
     * mixed init/continuation calls in one grid epoch are refused fatally. */
    if(init!=0) { why=WT_MLN_INIT; goto direct; }
    layout.logical_true_bits=cfg.logical_true_bits; layout.byte_budget=cfg.byte_budget;
    if(wt_mynn_layout_build(argv,cap,&layout,&view)!=WT_MYNN_LAYOUT_OK) { why=WT_MLN_GEOMETRY; goto direct; }
    memcpy(&jts,argv[WT_MYNN_JTS],4); memcpy(&jte,argv[WT_MYNN_JTE],4);
    c.service=cfg.service; c.rank=cfg.rank; c.thread=(uint32_t)thread;
    c.byte_budget=cfg.byte_budget; c.native=native; c.native_context=&n;
    c.signature=(struct wt_nl_signature){WT_MYNN_NODE_PHASE,cfg.profile,WT_MYNN_BATCH_SCHEMA,
        atomic_load(&grid_id),atomic_load(&epoch_id),WT_MYNN_BATCH_SCHEMA,n.init};
    begin=now(); deadline=begin+cfg.deadline_ns;
    if(deadline<begin) die();
    if(jts==jte) {
        atomic_fetch_add(&fallback,1); atomic_fetch_add(&refusals[WT_MLN_SINGLE_ROW],1);
        native(&n,argv); help(&c,deadline); goto done;
    }
    memcpy(tail,argv,sizeof(tail)); memcpy(head,argv,sizeof(head));
    /* Every scalar is INTENT(IN) in the frozen132 schema. Separate immutable
     * copies admit compiler-pooled scalar constants; native_capacities already
     * rejected scalar/array overlaps. Preserve each original64-byte alignment. */
    for(unsigned a=0;a<WT_MYNN_NARG;++a) if(argv[a] && !wt_mynn_args[a].rank) {
        tail[a]=scalar[a]+((uintptr_t)argv[a]&63); memcpy(tail[a],argv[a],4);
    }
    memcpy(tail[WT_MYNN_JTS],&jte,4); head_end=jte-1; head[WT_MYNN_JTE]=&head_end;
    s=wt_mynn_node_publish(&c,tail,cap,&layout,c.signature.epoch,&call,&bs);
    atomic_fetch_add(&pack_ns,now()-begin);
    if(s!=WT_NL_OK) {
        why=bs==WT_MYNN_ALIAS ? WT_MLN_ALIAS : bs==WT_MYNN_BUDGET ? WT_MLN_BUDGET :
            bs==WT_MYNN_OK ? WT_MLN_SLOT : WT_MLN_PREPARE;
        goto direct;
    }
    atomic_fetch_add(&donated,1);
    /* One full-I tail row is a bounded donation. Useful original head work
     * creates the helper opportunity without a delay or native replay. */
    native(&n,head); help(&c,deadline);
    s=collect_ready(&c,&call,&executor_rank);
    while(s==WT_NL_PENDING) {
        if(now()>=deadline) { (void)wt_nl_owner_poison(c.service,c.rank,c.thread,&call.ticket); die(); }
        s=wt_mynn_node_owner_execute(&c,&call,((uint64_t)c.rank<<32)|c.thread);
        if(s==WT_NL_OK) { self=1; atomic_fetch_add(&self_executed,1); }
        else if(s!=WT_NL_PENDING && s!=WT_NL_BUSY) die();
        help(&c,deadline); s=collect_ready(&c,&call,&executor_rank);
        if(s==WT_NL_PENDING) sched_yield();
    }
    if(s!=WT_NL_OK || wt_nl_ticket_executor(c.service,c.rank,c.thread,&call.ticket,&tag)!=WT_NL_OK ||
       executor_rank!=(uint32_t)(tag>>32)) die();
    if((self && (tag>>32)!=c.rank) || (!self && (tag>>32)==c.rank)) die();
    atomic_fetch_add(&committed,1);
    if((tag>>32)!=c.rank) atomic_fetch_add(&remote_committed,1);
    trace("commit",&c,&call.ticket,(uint32_t)(tag>>32));
    if(wt_mynn_node_finish(&c,&call)!=WT_NL_OK) die();
    wt_mynn_node_destroy(&call); goto done;
 direct:
    atomic_fetch_add(&fallback,1); atomic_fetch_add(&refusals[why],1);
    real_call(argv,invoke);
 done:
    atomic_fetch_sub(&readers,1); --depth;
}
void wt_mynn_node_live_get_counters(struct wt_mynn_node_live_counters *out)
{
    if(!out) return;
#define COPY(x) out->x=atomic_load(&x)
    COPY(requested); COPY(donated); COPY(committed); COPY(self_executed); COPY(helper_executed);
    COPY(remote_committed); COPY(fallback); COPY(fatal); COPY(pack_ns); COPY(native_ns); COPY(progress_ns);
    COPY(shadow_checked); COPY(shadow_failed); COPY(shadow_bytes);
#undef COPY
    for(unsigned i=0;i<WT_MLN_REASON_COUNT;++i) out->refusal[i]=atomic_load(&refusals[i]);
}
void wt_mynn_node_live_dump(FILE *f)
{
    static const char *names[]={"config","abi","static","scope","mapper","omp","capacity","geometry",
        "init","single_row","alias","budget","slot","prepare"};
    struct wt_mynn_node_live_counters c;
    if(!f) return;
    wt_mynn_node_live_get_counters(&c);
    fprintf(f,"# wt_mynn_node mode=%s adaptive=pending shadow_remote=%d pid=%lu rank=%u grid=%llu epoch=%llu init=%llu serialized=%d requested=%llu donated=%llu committed=%llu self=%llu helper=%llu remote=%llu fallback=%llu fatal=%llu pack_ns=%llu native_ns=%llu progress_ns=%llu shadow_checked=%llu shadow_failed=%llu shadow_bytes=%llu\n",
        cfg.owner_only ? "forced-g0-owner-only" : "forced-g0",cfg.shadow_verify_remote,
        (unsigned long)getpid(),cfg.rank,(unsigned long long)atomic_load(&grid_id),(unsigned long long)atomic_load(&epoch_id),
        (unsigned long long)atomic_load(&static_id),!cfg.concurrency_verified,(unsigned long long)c.requested,
        (unsigned long long)c.donated,(unsigned long long)c.committed,(unsigned long long)c.self_executed,
        (unsigned long long)c.helper_executed,(unsigned long long)c.remote_committed,(unsigned long long)c.fallback,
        (unsigned long long)c.fatal,(unsigned long long)c.pack_ns,(unsigned long long)c.native_ns,(unsigned long long)c.progress_ns,
        (unsigned long long)c.shadow_checked,(unsigned long long)c.shadow_failed,(unsigned long long)c.shadow_bytes);
    for(unsigned i=0;i<WT_MLN_REASON_COUNT;++i) fprintf(f,"# wt_mynn_node refusal.%s=%llu\n",names[i],(unsigned long long)c.refusal[i]);
}
void wt_mynn_node_live_close(void)
{
    if(atomic_load(&outer_active) || atomic_load(&readers)) die();
    if(spans) { wt_live_spans_destroy(spans); spans=NULL; }
    wt_kf_node_shm_close(&mapper); cfg.service=NULL;
}
