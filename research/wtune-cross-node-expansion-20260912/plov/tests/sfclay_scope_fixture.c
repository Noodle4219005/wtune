#include "wt_sfclay_scope.h"
#include "wt_sfclay_live.h"
#include "wt_sfclay_schema.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK %d: %s\n",__LINE__,#x); exit(80); } } while (0)
extern void wt_sfclay_scope_test_call(void **);
extern void wt_sfclay_bridge_call(void **);
extern void module_sf_sfclay_mp_sfclayinit_(int *);
static const char *mode;
static void *rk_args[84], *sf_args[84];
static uint64_t rk_values[84];
static size_t bytes[84];
static unsigned rk_calls, sf_calls, captures, queries;
static int expect_staged;

static int is(const char *s) { return !strcmp(mode,s); }
static void cancel_disabled(void) {
    int old;
    CHECK(!pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,&old));
    CHECK(old==PTHREAD_CANCEL_DISABLE);
}
static int capture(void *grid, wt_live_spans *spans, size_t max_nodes, int *tiles) {
    CHECK(grid==rk_args[0] && max_nodes==128); ++captures; *tiles=1;
    cancel_disabled();
    for(int i=0;i<84;++i) {
        if(wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR) continue;
        int rank=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?3:2;
        if(wt_live_spans_add(spans,sf_args[i],bytes[i],WT_SPAN_REAL,4,rank)!=WT_SPAN_OK) return 0;
    }
    return 1;
}
static int query(void *grid, void *cfg, struct wt_sfclay_native_result *result) {
    CHECK(grid==rk_args[0] && cfg==rk_args[1]); ++queries;
    cancel_disabled();
    *result=(struct wt_sfclay_native_result){1,7,0,7,WT_SFCLAY_NATIVE_OK};
    return 1;
}
static int serial(void) { cancel_disabled(); return 0; }

void fixture_sf_native(void **actual) {
    ++sf_calls; cancel_disabled();
    /* The generated callee has its own argv array, so compare referents. */
    for(int i=0;i<84;++i) CHECK((actual[i]!=sf_args[i])==expect_staged);
    ((float *)actual[12])[1]=42.f;
}
void fixture_rk_native(void **actual) {
    ++rk_calls; cancel_disabled();
    for(int i=0;i<84;++i) {
        CHECK(actual[i]==rk_args[i]);
        if(actual[i]) CHECK(*(uint64_t *)actual[i]==UINT64_C(0xabc00000)+(unsigned)i);
    }
    CHECK(captures==1 && queries==1);
    if(is("nested-rk")) wt_sfclay_scope_test_call(rk_args);
    wt_sfclay_surface_hook_begin();
    if(is("nested-hook")) wt_sfclay_surface_hook_begin();
    expect_staged=1;
    wt_sfclay_bridge_call(sf_args);
    wt_sfclay_surface_hook_end();
    if(actual[83]) *(uint64_t *)actual[83]=UINT64_C(0xbeef);
}
static void fixture(void) {
    for(int i=0;i<84;++i) {
        rk_values[i]=UINT64_C(0xabc00000)+(unsigned)i; rk_args[i]=&rk_values[i];
        bytes[i]=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?48:
                 wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_2D?24:4;
        sf_args[i]=calloc(1,bytes[i]); CHECK(sf_args[i]);
        for(size_t j=0;j<bytes[i]/4;++j) ((float *)sf_args[i])[j]=1.f;
    }
    for(int i=58;i<76;++i) *(int *)sf_args[i]=i%2==0?1:2;
    *(int *)sf_args[65]=3; *(int *)sf_args[71]=3;
    if(is("optional-null")) for(int i=80;i<84;++i) rk_args[i]=NULL;
}
static void direct_surface(void) {
    expect_staged=0;
    wt_sfclay_surface_hook_begin();
    wt_sfclay_bridge_call(sf_args);
    wt_sfclay_surface_hook_end();
}
int main(int argc, char **argv) {
    CHECK(argc==2); mode=argv[1]; fixture();
    int allowed=1;
    module_sf_sfclay_mp_sfclayinit_(&allowed);
    struct wt_sfclay_live_config config={1,1,1,1,128,128,1u<<20,capture,serial,query};
    CHECK(wt_sfclay_live_configure(&config));
    if(is("unmatched-hook")) wt_sfclay_surface_hook_end();
    if(is("direct")) direct_surface();
    else {
        volatile uint64_t canary[4]={11,22,33,44};
        wt_sfclay_scope_test_call(rk_args);
        CHECK(canary[0]==11 && canary[1]==22 && canary[2]==33 && canary[3]==44);
        CHECK(rk_calls==1 && sf_calls==1 && captures==1 && queries==3);
        CHECK(((float *)sf_args[12])[1]==42.f);
        for(int i=0;i<84;++i)
            CHECK(rk_values[i]==((i==83 && rk_args[i])?UINT64_C(0xbeef):UINT64_C(0xabc00000)+(unsigned)i));
        /* A disabled direct surface after return proves the outer scope cleared. */
        direct_surface();
    }
    struct wt_sfclay_live_counters counters;
    wt_sfclay_live_get_counters(&counters);
    CHECK(counters.staged==(unsigned)!is("direct") && counters.fallback==1);
    CHECK(sf_calls==(is("direct")?1u:2u));
    if(is("direct")) CHECK(captures==0 && queries==0 && rk_calls==0);
    int old;
    CHECK(!pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,&old)); CHECK(old==PTHREAD_CANCEL_ENABLE);
    CHECK(!pthread_setcancelstate(old,NULL));
    for(int i=0;i<84;++i) free(sf_args[i]);
    puts("84-reference first-RK forwarding and paired surface scopes PASS");
    return 0;
}
