#include "wt_mynn_bootstrap.h"
#include "wt_mynn_native.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#ifndef WT_MYNN_BUILD_ABI_VERIFIED
#define WT_MYNN_BUILD_ABI_VERIFIED 0
#endif
#ifndef WT_MYNN_BUILD_CALLER_VERIFIED
#define WT_MYNN_BUILD_CALLER_VERIFIED 0
#endif
#ifndef WT_MYNN_BUILD_STATIC_STATE_VERIFIED
#define WT_MYNN_BUILD_STATIC_STATE_VERIFIED 0
#endif
#ifndef WT_MYNN_BUILD_CONCURRENT_VERIFIED
#define WT_MYNN_BUILD_CONCURRENT_VERIFIED 0
#endif
#ifndef WT_MYNN_BUILD_LOGICAL_TRUE_BITS
#define WT_MYNN_BUILD_LOGICAL_TRUE_BITS 1
#endif
#ifndef WT_MYNN_BUILD_PROFILE_ID
#define WT_MYNN_BUILD_PROFILE_ID "unattested"
#endif
static int started,requested_state,configured;
static int number(const char *name,size_t def,size_t *out)
{
    const char *p=getenv(name); size_t value=0;
    if(!p) { *out=def; return 1; }
    if(!*p) return 0;
    for(;*p;++p) { unsigned d=(unsigned)(*p-'0'); if(d>9 || value>(SIZE_MAX-d)/10) return 0; value=value*10+d; }
    if(!value) return 0;
    *out=value; return 1;
}
int wt_mynn_bootstrap_start(int requested,int selector)
{
    struct wt_mynn_node_live_config c={0}; size_t deadline;
    const char *g0=getenv("WT_MYNN_NODE_G0");
    const unsigned char *p=(const unsigned char *)WT_MYNN_BUILD_PROFILE_ID;
    if(started) return configured;
    started=1; requested_state=requested;
    if(!requested || selector<0 || !g0 || strcmp(g0,"1")) return 0;
    c.enabled=1; c.abi_verified=WT_MYNN_BUILD_ABI_VERIFIED;
    {
        const char *owner=getenv("WT_MYNN_NODE_G0_OWNER_ONLY");
        const char *shadow=getenv("WT_MYNN_NODE_G0_SHADOW");
        if(owner && strcmp(owner,"0") && strcmp(owner,"1")) return 0;
        if(shadow && strcmp(shadow,"0") && strcmp(shadow,"1")) return 0;
        c.owner_only=owner && !strcmp(owner,"1");
        c.shadow_verify_remote=shadow && !strcmp(shadow,"1");
        if(c.owner_only && c.shadow_verify_remote) return 0;
    }
    c.caller_verified=WT_MYNN_BUILD_CALLER_VERIFIED;
    c.static_state_verified=WT_MYNN_BUILD_STATIC_STATE_VERIFIED;
    c.concurrency_verified=WT_MYNN_BUILD_CONCURRENT_VERIFIED;
    c.logical_true_bits=WT_MYNN_BUILD_LOGICAL_TRUE_BITS;
    c.grid_selector=selector; c.threads=(uint32_t)omp_get_max_threads();
    c.profile=UINT64_C(14695981039346656037);
    for(;*p;++p) { c.profile^=*p; c.profile*=UINT64_C(1099511628211); }
    if(!strcmp(WT_MYNN_BUILD_PROFILE_ID,"unattested")) c.profile=0;
    if(!number("WT_MYNN_NODE_SLOT_BYTES",1024*1024,&c.slot_bytes) ||
       !number("WT_MYNN_NODE_SHARED_BUDGET",256u*1024*1024,&c.shared_budget) ||
       !number("WT_MYNN_NODE_BYTE_BUDGET",256u*1024*1024,&c.byte_budget) ||
       !number("WT_MYNN_NODE_DEADLINE_MS",30000,&deadline) || deadline>600000) return 0;
    c.deadline_ns=(uint64_t)deadline*1000000;
    c.max_spans=4096; c.max_nodes=8192;
    configured=wt_mynn_native_configure(&c);
    return configured;
}
int wt_mynn_bootstrap_state(void)
{
    struct wt_mynn_node_live_counters c;
    if(!started) return -1;
    if(!configured || !requested_state) return 0;
    wt_mynn_node_live_get_counters(&c);
    return c.committed ? 1 : c.requested ? 0 : -1;
}
void wt_mynn_bootstrap_dump(FILE *f)
{
    if(!f) return;
    fprintf(f,"# wt_mynn_bootstrap requested=%d configured=%d profile=%s abi=%d caller=%d static=%d\n",
        requested_state,configured,WT_MYNN_BUILD_PROFILE_ID,WT_MYNN_BUILD_ABI_VERIFIED,
        WT_MYNN_BUILD_CALLER_VERIFIED,WT_MYNN_BUILD_STATIC_STATE_VERIFIED);
    wt_mynn_node_live_dump(f);
}
