#define _POSIX_C_SOURCE 200809L
#include "wt_mynn_node_live.h"
#include "wt_mynn_bootstrap.h"
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "mynn-live-abi.h"
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"MYNN LIVE FAIL %d: %s\n",__LINE__,#x); abort(); } } while(0)
struct grid { int id,epoch; void **args; };
static struct grid grid;
static int initialized,rank_id,mode,scope_init;
static void (*native_driver)(void **);
static _Atomic unsigned *coord;
static size_t capacity(unsigned a)
{ return !wt_mynn_args[a].rank ? 4 : wt_mynn_args[a].rank==2 ? 4*6*4 : 4*33*6*4; }
extern void __real_wt_mynn_bridge_dispatch(void **,void (*)(void **),size_t);
extern void wt_sfclay_scope_test_call(void **);
void wt_mynn_native_scope_(void *opaque,void *cfg,int *id,int *epoch,int *status)
{
    struct grid *g=opaque; (void)cfg;
    *id=g->id; *epoch=g->epoch; *status=mode==2 ? WT_SPAN_INVALID : WT_SPAN_OK;
}
void wt_grid_capture_(void *opaque,void **registry,const size_t *max_nodes,int *tiles,int *status)
{
    struct grid *g=opaque; (void)max_nodes; *tiles=2; *status=WT_SPAN_OK;
    for(unsigned a=0;a<WT_MYNN_NARG;++a) if(wt_mynn_args[a].rank && g->args[a]) {
        if(wt_live_spans_add(*registry,g->args[a],capacity(a),a==WT_MYNN_KPBL ? WT_SPAN_INTEGER : WT_SPAN_REAL,
                            4,wt_mynn_args[a].rank)!=WT_SPAN_OK) *status=WT_SPAN_INVALID;
    }
}
static void observed_native(void **a)
{
    int j=*(int32_t *)a[WT_MYNN_JTS];
    size_t off=1+4*(33*(size_t)(j-1));
    int input_rank=((float *)a[WT_MYNN_U])[off]>5.5f;
    struct timespec delay={0,1000000};
    if(mode>=4) fprintf(stderr,"MYNN_NATIVE pid=%lu epoch=%d j=%d\n",(unsigned long)getpid(),grid.epoch,j);
    if(mode==0 && rank_id==0 && input_rank==0 && !(j&1)) {
        atomic_store(&coord[0],1);
        for(unsigned n=0;!atomic_load(&coord[1]);++n) { CHECK(n<3000); nanosleep(&delay,NULL); }
    }
    if(mode==4) { delay.tv_nsec=10000000; nanosleep(&delay,NULL); }
    native_driver(a);
    if(mode==5 && (j&1)) ((float *)a[WT_MYNN_RHO])[off]+=1;
    if(rank_id==1 && input_rank==0) atomic_fetch_add(&coord[1],1);
}
void mynn_live_first_real(void **args)
{
    struct grid *g=args[0];
    CHECK(g==&grid && args[1]==&grid);
#pragma omp parallel num_threads(2)
    {
        void *a[WT_MYNN_NARG]; int32_t jts=2+2*omp_get_thread_num(),jte=jts+1;
        int32_t init=scope_init || (mode==7 && omp_get_thread_num()==1);
        CHECK(omp_get_num_threads()==2);
        memcpy(a,g->args,sizeof(a)); a[WT_MYNN_JTS]=&jts; a[WT_MYNN_JTE]=&jte; a[WT_MYNN_INITFLAG]=&init;
        mynn_live_driver_call(a);
    }
}
static void environment(unsigned job,int rank)
{
    char text[32]; rank_id=rank;
    snprintf(text,sizeof(text),"%u",job); CHECK(!setenv("SLURM_JOB_ID",text,1));
    CHECK(!setenv("SLURM_STEP_ID","7",1)); CHECK(!setenv("SLURM_LOCALID",rank ? "1" : "0",1));
    CHECK(!setenv("SLURM_STEP_TASKS_PER_NODE","2",1));
    CHECK(!setenv("WT_MYNN_NODE_G0","1",1));
    CHECK(!setenv("WT_MYNN_NODE_G0_OWNER_ONLY",mode==8 ? "bad" : mode==3 ? "1" : "0",1));
    CHECK(!setenv("WT_MYNN_NODE_DEADLINE_MS",mode==4 ? "1" : "5000",1));
    CHECK(!setenv("WT_MYNN_NODE_G0_TRACE","1",1));
    omp_set_dynamic(0); omp_set_num_threads(2);
}
static void rank_run(void **args)
{
    void *oracle[WT_MYNN_NARG],*real_args[WT_MYNN_NARG],*scope[84];
    int32_t init=0;
    struct wt_mynn_node_live_counters counts;
    grid=(struct grid){4,1,args};
    for(size_t i=0;i<capacity(WT_MYNN_U)/4;++i) ((float *)args[WT_MYNN_U])[i]=(float)(5+rank_id);
    for(unsigned a=0;a<WT_MYNN_NARG;++a) {
        oracle[a]=malloc(capacity(a)); CHECK(oracle[a]); memcpy(oracle[a],args[a],capacity(a));
    }
    memcpy(real_args,oracle,sizeof(real_args)); real_args[WT_MYNN_INITFLAG]=&init;
    for(unsigned a=0;a<84;++a) scope[a]=&grid;
    if(mode==6) {
        init=1; native_driver(real_args); init=0;
        scope_init=1; wt_sfclay_scope_test_call(scope); scope_init=0;
        for(unsigned a=0;a<WT_MYNN_NARG;++a) CHECK(!memcmp(args[a],oracle[a],capacity(a)));
    }
    for(unsigned e=1;e<=2;++e) {
        native_driver(real_args);
        grid.epoch=(int)e+(mode==1 && rank_id==1 ? 20 : mode==6 ? 1 : 0);
        for(unsigned a=0;a<84;++a) scope[a]=&grid;
        wt_sfclay_scope_test_call(scope);
        for(unsigned a=0;a<WT_MYNN_NARG;++a) {
            if(memcmp(args[a],oracle[a],capacity(a))) fprintf(stderr,"mismatch arg=%u epoch=%u\n",a,e);
            CHECK(!memcmp(args[a],oracle[a],capacity(a)));
        }
    }
    wt_mynn_node_live_get_counters(&counts); wt_mynn_bootstrap_dump(stderr);
    CHECK(counts.requested==(mode==6 ? 7u : 5u) && counts.fatal==0);
#ifdef MYNN_LIVE_UNATTESTED
    CHECK(counts.donated==0 && counts.fallback==5);
#else
    if(mode==2 || mode==8) CHECK(counts.donated==0 && counts.fallback==5 && counts.refusal[mode==2 ? WT_MLN_SCOPE : WT_MLN_CONFIG]>=4);
    else {
        CHECK(counts.donated==4 && counts.committed==4);
        CHECK(counts.committed==counts.self_executed+counts.remote_committed);
        if(mode==0 && rank_id==0) CHECK(counts.remote_committed>0);
        if(mode==0 && rank_id==1) CHECK(counts.helper_executed>0);
        if(mode==1) CHECK(counts.helper_executed==0);
        if(mode==3) CHECK(counts.self_executed==4 && counts.helper_executed==0 && counts.remote_committed==0);
        if(mode==6) CHECK(counts.refusal[WT_MLN_INIT]==2);
    }
#endif
    CHECK(wt_mynn_bootstrap_state()==(counts.committed ? 1 : 0));
    for(unsigned a=0;a<WT_MYNN_NARG;++a) free(oracle[a]);
    wt_mynn_node_live_close();
}
void __wrap_wt_mynn_bridge_dispatch(void **args,void (*invoke)(void **),size_t nargs)
{
    unsigned job=(unsigned)getpid(); pid_t child; int status,fd,start[2]; char byte;
    char path[]=MYNN_LIVE_ARTIFACT_DIRECTORY "/coord-XXXXXX";
    if(initialized) { __real_wt_mynn_bridge_dispatch(args,observed_native,nargs); return; }
    alarm(20); initialized=1; native_driver=invoke;
    CHECK(nargs==WT_MYNN_NARG && *(int32_t *)args[WT_MYNN_INITFLAG]==1);
    __real_wt_mynn_bridge_dispatch(args,invoke,nargs); /* Real scientific init. */
    mode=getenv("MYNN_LIVE_CASE") ? atoi(getenv("MYNN_LIVE_CASE")) : 0;
    fd=mkstemp(path); CHECK(fd>=0); CHECK(!unlink(path)); CHECK(!ftruncate(fd,4096));
    coord=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); CHECK(coord!=MAP_FAILED);
    atomic_init(&coord[0],0); atomic_init(&coord[1],0);
    if(mode==4 || mode==5 || mode==7) { environment(job,0); CHECK(wt_mynn_bootstrap_start(1,4)); rank_run(args); abort(); }
    CHECK(!pipe(start)); child=fork(); CHECK(child>=0);
    if(!child) {
        alarm(20); close(start[1]); CHECK(read(start[0],&byte,1)==1); environment(job,1);
#ifdef MYNN_LIVE_UNATTESTED
        CHECK(!wt_mynn_bootstrap_start(1,4));
#else
        CHECK(wt_mynn_bootstrap_start(1,4)==(mode!=8));
#endif
        if(mode==0) {
            struct timespec delay={0,1000000};
            for(unsigned n=0;!atomic_load(&coord[0]);++n) { CHECK(n<3000); nanosleep(&delay,NULL); }
        }
        rank_run(args); close(fd); _exit(0);
    }
    close(start[0]); environment(job,0);
#ifdef MYNN_LIVE_UNATTESTED
    CHECK(!wt_mynn_bootstrap_start(1,4));
#else
    CHECK(wt_mynn_bootstrap_start(1,4)==(mode!=8));
#endif
    CHECK(write(start[1],"s",1)==1); close(start[1]); rank_run(args);
    CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    CHECK(!munmap(coord,4096)); close(fd); alarm(0);
    fprintf(stderr,"PASS MYNN production bootstrap/firstRK84/bridge132/fork/OMP2 realGNU full132 both epochs\n");
}
