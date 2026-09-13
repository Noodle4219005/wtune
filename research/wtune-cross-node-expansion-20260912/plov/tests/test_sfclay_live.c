#define _POSIX_C_SOURCE 200809L
#include "wt_sfclay_live.h"
#include "wt_sfclay_schema.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK %d: %s\n",__LINE__,#x); exit(80); } } while (0)
#define PC(x) CHECK((x)==0)
static void *args[84];
static size_t bytes[84];
static const char *mode;
static int calls, init_calls, captures, queries, omps, tiles=1, omp_value;
static int fractional, scheme=7, query_status, query_fail, skip_index=-1, span_type=WT_SPAN_REAL;
static int expect_staged, logical_value=37;
static uint64_t outer, surface;
static wt_live_spans *captured_spans;
static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv=PTHREAD_COND_INITIALIZER;
static int reader_entered, writer_waited, reader_done;
static _Thread_local int writer_probe, later_reader_probe;
static int later_waited, release_first, initializer_entered, release_initializer;
static int initializer_finished, later_calls;
static unsigned char configured_owner_bytes[sizeof(pthread_t)];
int __real_pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
int __wrap_pthread_cond_wait(pthread_cond_t *c,pthread_mutex_t *m) {
    if((writer_probe || later_reader_probe) && c!=&cv) {
        PC(pthread_mutex_lock(&mu));
        if(writer_probe) writer_waited=1;
        if(later_reader_probe) later_waited=1;
        PC(pthread_cond_broadcast(&cv)); PC(pthread_mutex_unlock(&mu));
    }
    return __real_pthread_cond_wait(c,m);
}
static int is(const char *s) { return strcmp(mode,s)==0; }
static void check_cancel(void) {
    int old; PC(pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,&old)); CHECK(old==PTHREAD_CANCEL_DISABLE);
}
static void native(void **a);
static void init(int *p) {
    check_cancel(); CHECK(p==&logical_value && *p==37); ++init_calls;
    if(is("recursive-init")) wt_sfclay_init_dispatch(p,init);
}
static int oq(void) {
    ++omps; check_cancel();
    if(is("omp-reentry")) wt_sfclay_surface_begin();
    return omp_value;
}
static int nq(void *g,void *c,struct wt_sfclay_native_result *r) {
    CHECK(g==(void *)17 && c==(void *)23); ++queries; check_cancel();
    if(is("query-reentry")) wt_sfclay_live_dispatch((void **)1,native,84);
    if(is("query-postspan") && queries==3) wt_live_spans_end(captured_spans);
    *r=(struct wt_sfclay_native_result){1,scheme,fractional,scheme + is("scheme-mismatch"),query_status}; return !query_fail;
}
static int capture(void *g,wt_live_spans *s,size_t max_nodes,int *t) {
    CHECK(g==(void *)17 && max_nodes==128); ++captures; check_cancel();
    if(is("capture-reentry")) wt_sfclay_first_rk_begin(g,(void *)23);
    captured_spans=s; *t=tiles;
    for(int i=0;i<84;++i) {
        if(i==skip_index || wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR) continue;
        int rank=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?3:2;
        if(wt_live_spans_add(s,args[i],bytes[i],span_type,4,rank)!=WT_SPAN_OK) return 0;
    }
    return !is("capture-fail");
}
static struct wt_sfclay_live_config config(void) {
    return (struct wt_sfclay_live_config){1,1,1,1,128,128,1u<<20,capture,oq,nq};
}
static void fixture(void) {
    for(int i=0;i<84;++i) {
        bytes[i]=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?48:
            wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_2D?24:4;
        args[i]=calloc(1,bytes[i]); CHECK(args[i]);
        for(size_t j=0;j<bytes[i]/4;++j) ((float *)args[i])[j]=(float)(i+1);
    }
    for(int i=58;i<76;++i) *(int *)args[i]=i%2==0?1:(i==63||i==69||i==75?2:3);
    *(int *)args[61]=*(int *)args[67]=*(int *)args[73]=2;
    *(int *)args[70]=*(int *)args[71]=2;
    *(int *)args[72]=*(int *)args[73]=1;
}
static void native(void **a) {
    ++calls; CHECK(write(1,"native\n",7)==7); check_cancel();
    if(is("opaque")||is("unconfigured")||is("missing-omp")||is("missing-query")||is("missing-capture")||is("chain-proof")||is("init-proof")) {
        CHECK(a==(void **)1); return;
    }
    CHECK((a!=args)==expect_staged);
    if(is("reader-init")) wt_sfclay_init_dispatch(&logical_value,init);
    if(is("dispatch-reentry")) wt_sfclay_live_dispatch(a,native,84);
    if(is("token-in-native")) wt_sfclay_surface_end(surface);
    if(is("guard")) ((unsigned char *)a[12])[bytes[12]]=0;
    if(is("readonly")) ((float *)a[0])[0]=99;
    if(is("owner")) ((float *)args[12])[0]=99;
    if(is("postquery")) scheme=8;
    if(is("postquery-fail")) query_fail=1;
    if(is("postspan")) wt_live_spans_end(captured_spans);
    for(int i=0;i<84;++i) if(wt_sfclay_schema_access[i]==WT_SFCLAY_WRITABLE)
        ((float *)a[i])[1]=(float)(1000+i);
}
static void scope_call(void **a) {
    outer=wt_sfclay_first_rk_begin((void *)17,(void *)23); surface=wt_sfclay_surface_begin();
    wt_sfclay_live_dispatch(a,native,84);
    wt_sfclay_surface_end(surface); wt_sfclay_first_rk_end(outer);
}
static void *other_owner(void *unused) { (void)unused; scope_call(args); return NULL; }
static void held_native(void **a) {
    CHECK(a==(void **)1); ++calls; check_cancel();
    PC(pthread_mutex_lock(&mu)); reader_entered=1; PC(pthread_cond_broadcast(&cv));
    while(!writer_waited) PC(pthread_cond_wait(&cv,&mu));
    CHECK(init_calls==0); reader_done=1; PC(pthread_mutex_unlock(&mu));
}
static void *reader(void *unused) { (void)unused; wt_sfclay_live_dispatch((void **)1,held_native,84); return NULL; }
static void held_init(int *p) {
    PC(pthread_mutex_lock(&mu)); CHECK(reader_done); PC(pthread_mutex_unlock(&mu)); init(p);
}
static void *writer(void *unused) { (void)unused; writer_probe=1; wt_sfclay_init_dispatch(&logical_value,held_init); return NULL; }
static void lease_test(int configured) {
    pthread_t r,w;
    if(configured) { struct wt_sfclay_live_config c=config(); c.source_abi_verified=0; CHECK(wt_sfclay_live_configure(&c)); }
    PC(pthread_create(&r,NULL,reader,NULL));
    PC(pthread_mutex_lock(&mu)); while(!reader_entered) PC(pthread_cond_wait(&cv,&mu)); PC(pthread_mutex_unlock(&mu));
    PC(pthread_create(&w,NULL,writer,NULL)); PC(pthread_join(r,NULL)); PC(pthread_join(w,NULL));
    CHECK(writer_waited && calls==1 && init_calls==1);
}
/* Keep the first reader until both the queued writer and the later reader
 * have reached native condition waits. All fixture state uses this mutex. */
static void first_reader_native(void **a) {
    CHECK(a==(void **)1); check_cancel();
    PC(pthread_mutex_lock(&mu)); ++calls; reader_entered=1;
    PC(pthread_cond_broadcast(&cv));
    while(!release_first) PC(pthread_cond_wait(&cv,&mu));
    CHECK(!initializer_entered && later_calls==0);
    PC(pthread_mutex_unlock(&mu));
}
static void *first_reader(void *unused) {
    (void)unused; wt_sfclay_live_dispatch((void **)1,first_reader_native,84); return NULL;
}
static void paused_initializer(int *p) {
    check_cancel(); CHECK(p==&logical_value && *p==37);
    PC(pthread_mutex_lock(&mu)); initializer_entered=1;
    PC(pthread_cond_broadcast(&cv));
    while(!release_initializer) PC(pthread_cond_wait(&cv,&mu));
    CHECK(later_calls==0); ++init_calls; initializer_finished=1;
    PC(pthread_mutex_unlock(&mu));
}
static void *paused_writer(void *unused) {
    (void)unused; writer_probe=1;
    wt_sfclay_init_dispatch(&logical_value,paused_initializer); return NULL;
}
static void later_native(void **a) {
    CHECK(a==(void **)1); check_cancel();
    PC(pthread_mutex_lock(&mu));
    CHECK(initializer_finished && init_calls==1);
    ++calls; ++later_calls;
    PC(pthread_mutex_unlock(&mu));
}
static void *later_reader(void *unused) {
    (void)unused; later_reader_probe=1;
    wt_sfclay_live_dispatch((void **)1,later_native,84); return NULL;
}
static void wait_flag(int *flag) {
    PC(pthread_mutex_lock(&mu));
    while(!*flag) PC(pthread_cond_wait(&cv,&mu));
    PC(pthread_mutex_unlock(&mu));
}
static void exclusion_test(void) {
    pthread_t first, w, later;
    int preference=is("writer-preference");
    if(preference || is("active-writer-disabled")) {
        struct wt_sfclay_live_config c=config();
        c.source_abi_verified=0;
        if(!preference) c.enabled=0;
        CHECK(wt_sfclay_live_configure(&c));
    }
    if(preference) {
        PC(pthread_create(&first,NULL,first_reader,NULL)); wait_flag(&reader_entered);
    }
    PC(pthread_create(&w,NULL,paused_writer,NULL));
    wait_flag(preference ? &writer_waited : &initializer_entered);
    PC(pthread_create(&later,NULL,later_reader,NULL)); wait_flag(&later_waited);
    if(preference) {
        PC(pthread_mutex_lock(&mu)); release_first=1;
        PC(pthread_cond_broadcast(&cv)); PC(pthread_mutex_unlock(&mu));
        wait_flag(&initializer_entered);
    }
    PC(pthread_mutex_lock(&mu)); CHECK(later_calls==0);
    release_initializer=1; PC(pthread_cond_broadcast(&cv)); PC(pthread_mutex_unlock(&mu));
    if(preference) PC(pthread_join(first,NULL));
    PC(pthread_join(w,NULL)); PC(pthread_join(later,NULL));
    CHECK(later_waited && later_calls==1 && init_calls==1 && calls==1+preference);
}
static void *short_lived_owner(void *unused) {
    (void)unused;
    struct wt_sfclay_live_config c=config(); pthread_t self=pthread_self();
    memcpy(configured_owner_bytes,&self,sizeof self);
    CHECK(wt_sfclay_live_configure(&c)); return NULL;
}
static void *replacement_owner(void *unused) {
    (void)unused; pthread_t self=pthread_self();
    /* Compare saved object bytes only: pthread_equal on an expired ID is invalid.
     * The sequential glibc stack-cache reuse is a precondition checked here. */
    CHECK(!memcmp(configured_owner_bytes,&self,sizeof self));
    scope_call(args);
    CHECK(queries==0 && captures==0 && omps==0 && calls==1);
    struct wt_sfclay_live_counters count;
    wt_sfclay_live_get_counters(&count);
    CHECK(count.staged==0 && count.fallback==1); return NULL;
}
static void reused_owner_test(void) {
    pthread_t t;
    wt_sfclay_init_dispatch(&logical_value,init);
    PC(pthread_create(&t,NULL,short_lived_owner,NULL)); PC(pthread_join(t,NULL));
    PC(pthread_create(&t,NULL,replacement_owner,NULL)); PC(pthread_join(t,NULL));
}
int main(int argc,char **argv) {
    CHECK(argc==2); mode=argv[1]; fixture();
    struct wt_sfclay_live_config c=config(); struct wt_sfclay_live_counters count;
    if(is("reused-owner")) { reused_owner_test(); goto done; }
    if(is("writer-preference")||is("active-writer-unconfigured")||is("active-writer-disabled")) { exclusion_test(); goto done; }
    if(is("lease")||is("lease-proof0")) { lease_test(is("lease-proof0")); goto done; }
    if(is("zero-outer")) { wt_sfclay_first_rk_end(0); return 0; }
    if(is("zero-surface")) { wt_sfclay_surface_end(0); return 0; }
    if(is("wrong-nargs")) { wt_sfclay_live_dispatch((void **)1,native,83); return 0; }
    if(is("recursive-init")) { wt_sfclay_init_dispatch(&logical_value,init); return 0; }
    if(is("invalid-flag")) { c.source_abi_verified=2; CHECK(!wt_sfclay_live_configure(&c)); goto done; }
    if(is("configure-active")) { outer=wt_sfclay_first_rk_begin(NULL,NULL); CHECK(!wt_sfclay_live_configure(&c)); wt_sfclay_first_rk_end(outer); CHECK(wt_sfclay_live_configure(&c)); goto done; }
    if(is("unconfigured")) { wt_sfclay_live_dispatch((void **)1,native,84); CHECK(calls==1); goto done; }
    if(!is("noinit")) wt_sfclay_init_dispatch(&logical_value,init);
    if(is("opaque")) c.source_abi_verified=0;
    if(is("chain-proof")) c.caller_chain_verified=0;
    if(is("init-proof")) c.initializer_relocation_verified=0;
    if(is("missing-omp")) c.omp_query=NULL;
    if(is("missing-query")) c.native_query=NULL;
    if(is("missing-capture")) c.capture=NULL;
    if(is("budget")) c.byte_budget=1;
    if(is("capacity")) skip_index=2;
    if(is("caller-owned")) skip_index=0;
    if(is("wrong-type")) span_type=WT_SPAN_INTEGER;
    if(is("tiles")) tiles=2;
    if(is("omp")) omp_value=2;
    if(is("negative-omp")) omp_value=-1;
    if(is("fractional")) fractional=1;
    if(is("query-status")) query_status=WT_SFCLAY_NATIVE_SCHEME;
    if(is("query-fail")) query_fail=1;
    CHECK(wt_sfclay_live_configure(&c)); CHECK(!wt_sfclay_live_configure(&c));
    expect_staged=is("staged")||is("caller-owned")||is("guard")||is("readonly")||is("owner")||is("postquery")||is("postquery-fail")||is("postspan")||is("reader-init")||is("dispatch-reentry")||is("token-in-native")||is("query-postspan");
    if(is("nonowner")) { pthread_t t; PC(pthread_create(&t,NULL,other_owner,NULL)); PC(pthread_join(t,NULL)); CHECK(queries==0 && captures==0 && omps==0); }
    else if(is("direct-surface")) { surface=wt_sfclay_surface_begin(); wt_sfclay_live_dispatch(args,native,84); wt_sfclay_surface_end(surface); }
    else if(is("opaque")||is("chain-proof")||is("init-proof")||is("missing-omp")||is("missing-query")||is("missing-capture")) {
        scope_call((void **)1); CHECK(queries==0 && captures==0 && omps==0);
    } else {
        outer=wt_sfclay_first_rk_begin((void *)17,(void *)23);
        if(is("nested-outer")) wt_sfclay_first_rk_begin((void *)17,(void *)23);
        surface=wt_sfclay_surface_begin();
        if(is("nested-surface")) wt_sfclay_surface_begin();
        if(is("bad-token")) wt_sfclay_surface_end(surface+1);
        if(is("prequery")) scheme=8;
        if(is("stale-span")) wt_live_spans_end(captured_spans);
        wt_sfclay_live_dispatch(args,native,84);
        if(is("fractional")) wt_sfclay_live_dispatch(args,native,84);
        wt_sfclay_surface_end(surface); wt_sfclay_first_rk_end(outer);
    }
    wt_sfclay_live_get_counters(&count); CHECK(calls==(is("fractional")?2:1));
    CHECK(count.requested==(unsigned long)calls && count.staged==(unsigned long)expect_staged);
    CHECK(count.fallback==(unsigned long)(calls-expect_staged));
    if(is("scheme-mismatch")) CHECK(queries==1 && captures==0);
    if(expect_staged) {
        CHECK(queries==3 && captures==1);
        for(int i=0;i<84;++i) if(wt_sfclay_schema_access[i]==WT_SFCLAY_WRITABLE) {
            CHECK(((float *)args[i])[0]==(float)(i+1)); CHECK(((float *)args[i])[1]==(float)(1000+i)); CHECK(((float *)args[i])[2]==(float)(i+1));
        }
    }
done:
    { int old; PC(pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,&old)); CHECK(old==PTHREAD_CANCEL_ENABLE); PC(pthread_setcancelstate(old,NULL)); }
    for(int i=0;i<84;++i) free(args[i]);
    puts("PASS"); return 0;
}
