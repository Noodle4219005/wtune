#define _POSIX_C_SOURCE 200809L
#define KFB_FIXTURE_ONLY
#include "test_kf_batch.c"
#include "wt_kf_node.h"
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

static struct wt_nl_signature signature(void)
{
    struct wt_nl_signature s={WT_KF_NODE_PHASE,19,WT_KF_BATCH_SCHEMA,4,7,1,23};
    return s;
}
static struct wt_kf_node_context context(struct wt_nl_service *service,
                                        unsigned rank,unsigned thread,struct compute *c)
{
    struct wt_kf_node_context out={service,{0},rank,thread,BUDGET,cps,c,0};
    out.signature=signature(); return out;
}
static int make_service(struct wt_nl_service *s)
{
    char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/service-XXXXXX";
    struct wt_nl_config cfg={2,2,32768};
    int fd=mkstemp(path);
    CHECK(fd>=0); CHECK(wt_nl_init_fd(fd,&cfg,s)==WT_NL_OK);
    return fd;
}
struct thread_job {
    struct wt_kf_node_context context;
    struct wt_kf_node_call call;
    struct fixture *fixture;
    pthread_barrier_t *barrier;
    int error;
};
static void *owner_thread(void *arg)
{
    struct thread_job *job=arg;
    enum wt_kf_status bs;
    /* Do not use CHECK here: its global diagnostic count is not synchronized. */
    if(wt_kf_node_publish(&job->context,job->fixture->args,job->fixture->cap,
                         100+job->context.thread,&job->call,&bs)!=WT_NL_OK) job->error=1;
    (void)pthread_barrier_wait(job->barrier);
    (void)pthread_barrier_wait(job->barrier);
    if(!job->error && wt_kf_node_collect(&job->context,&job->call)!=WT_NL_OK) job->error=2;
    if(!job->error && wt_kf_node_finish(&job->context,&job->call)!=WT_NL_OK) job->error=3;
    return NULL;
}
static void forked_disjoint(void)
{
    struct wt_nl_service service;
    struct fixture owner,second,expected,expected_second;
    struct compute c[2]={{0},{0}},direct={0};
    struct thread_job jobs[2]; pthread_t threads[2]; pthread_barrier_t barrier;
    struct wt_nl_counters counters;
    struct wt_nl_ticket ticket;
    int fd=make_service(&service),pipes[2],status;
    pid_t child; unsigned a;
    setup(&owner); setup(&second);
    second.scalar[WT_KF_JTS]=2; second.scalar[WT_KF_JTE]=3;
    second.scalar[WT_KF_ITS]=0; /* Legitimately nonuniform original tile width. */
    for(a=0;a<WT_KF_NARG;++a) if(wt_kf_shape[a]!=WT_KF_SCALAR) second.args[a]=owner.args[a];
    oracle_copy(&expected,&owner); oracle_copy(&expected_second,&second);
    for(a=0;a<WT_KF_NARG;++a) if(wt_kf_shape[a]!=WT_KF_SCALAR) expected_second.args[a]=expected.args[a];
    CHECK(cps(expected.args,&direct)==0); CHECK(cps(expected_second.args,&direct)==0);
    CHECK(pipe(pipes)==0);
    child=fork(); CHECK(child>=0);
    if(child==0) {
        struct compute helper={0}; struct wt_kf_node_context hc;
        struct wt_nl_service attached; char signal; size_t mapping_bytes=service.mapping_bytes;
        close(pipes[1]);
        wt_nl_detach(&service);
        if(wt_nl_attach_fd(fd,mapping_bytes,&attached)!=WT_NL_OK) _exit(11);
        hc=context(&attached,1,0,&helper);
        if(read(pipes[0],&signal,1)!=1) _exit(12);
        if(wt_kf_node_progress(&hc,(uint64_t)getpid(),&ticket)!=WT_NL_OK) _exit(13);
        if(wt_kf_node_progress(&hc,(uint64_t)getpid(),&ticket)!=WT_NL_OK) _exit(14);
        if(helper.calls!=2) _exit(15);
        if(wt_kf_node_progress(&hc,(uint64_t)getpid(),&ticket)!=WT_NL_IDLE) _exit(16);
        wt_nl_detach(&attached); close(fd); close(pipes[0]); _exit(0);
    }
    close(pipes[0]); CHECK(pthread_barrier_init(&barrier,NULL,3)==0);
    memset(jobs,0,sizeof(jobs));
    for(a=0;a<2;++a) {
        jobs[a].context=context(&service,0,a,&c[a]);
        jobs[a].fixture=a ? &second : &owner; jobs[a].barrier=&barrier;
        CHECK(pthread_create(&threads[a],NULL,owner_thread,&jobs[a])==0);
    }
    (void)pthread_barrier_wait(&barrier);
    CHECK(!jobs[0].error && !jobs[1].error);
    CHECK(wt_kf_node_progress(&jobs[0].context,1,&ticket)==WT_NL_IDLE);
    CHECK(write(pipes[1],"x",1)==1); close(pipes[1]);
    CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
    CHECK(c[0].calls==0 && c[1].calls==0);
    (void)pthread_barrier_wait(&barrier);
    for(a=0;a<2;++a) { CHECK(pthread_join(threads[a],NULL)==0); CHECK(jobs[a].error==0); wt_kf_node_destroy(&jobs[a].call); }
    CHECK(unchanged(&owner,&expected));
    CHECK(wt_nl_get_counters(&service,&counters)==WT_NL_OK);
    CHECK(counters.staged==2 && counters.helper_executed==2 && counters.self_executed==0 && counters.committed==2 && counters.failed==0);
    CHECK(pthread_barrier_destroy(&barrier)==0); wt_nl_detach(&service); close(fd);
}
static void failure_and_fallback(void)
{
    int mode;
    for(mode=0;mode<3;++mode) {
        struct wt_nl_service s; int fd=make_service(&s);
        struct fixture f,before; struct compute c={0};
        struct wt_kf_node_context owner=context(&s,0,0,&c),helper=context(&s,1,0,&c);
        struct wt_kf_node_call call={0}; struct wt_nl_ticket ticket; enum wt_kf_status bs;
        struct wt_nl_counters counts;
        setup(&f); before=f;
        CHECK(wt_kf_node_publish(&owner,f.args,f.cap,1,&call,&bs)==WT_NL_OK);
        CHECK(wt_kf_node_publish(&owner,f.args,f.cap,2,&call,&bs)==WT_NL_BUSY);
        CHECK(wt_kf_node_collect(&owner,&call)==WT_NL_PENDING);
        ++helper.signature.init_generation;
        CHECK(wt_kf_node_progress(&helper,99,&ticket)==WT_NL_IDLE);
        --helper.signature.init_generation;
        if(mode==0) {
            CHECK(wt_kf_node_owner_execute(&owner,&call,98)==WT_NL_OK);
            CHECK(c.calls==1);
            CHECK(wt_kf_node_collect(&helper,&call)==WT_NL_OWNER);
            CHECK(wt_kf_node_collect(&owner,&call)==WT_NL_OK);
            CHECK(wt_kf_node_collect(&owner,&call)!=WT_NL_OK);
            CHECK(wt_kf_node_finish(&owner,&call)==WT_NL_OK);
            CHECK(wt_nl_get_counters(&s,&counts)==WT_NL_OK);
            CHECK(counts.self_executed==1 && counts.committed==1);
        } else {
            if(mode==1) helper.byte_budget=1; else c.mode=1;
            CHECK(wt_kf_node_progress(&helper,99,&ticket)!=WT_NL_OK);
            CHECK(c.calls==(mode==2));
            CHECK(wt_kf_node_collect(&owner,&call)==(mode==1 ? WT_NL_REJECTED : WT_NL_FAILED));
            CHECK(wt_kf_node_owner_execute(&owner,&call,98)!=WT_NL_OK);
            CHECK(wt_kf_node_finish(&owner,&call)!=WT_NL_OK);
            CHECK(unchanged(&f,&before));
        }
        wt_kf_node_destroy(&call); wt_nl_detach(&s); close(fd);
    }
}
struct foreign_validation { struct wt_kf_batch *batch; void *result; size_t bytes; enum wt_kf_status validated,applied; };
static void *foreign_thread(void *opaque)
{
    struct foreign_validation *v=opaque;
    v->validated=wt_kf_batch_validate(v->batch,v->result,v->bytes);
    v->applied=wt_kf_batch_apply(v->batch);
    return NULL;
}
static void actual_thread_identity(void)
{
    struct fixture f; struct compute c={0}; struct foreign_validation v={0};
    const void *q; int invoked; pthread_t thread;
    setup(&f); CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&v.batch)==WT_KF_OK);
    q=wt_kf_batch_request(v.batch,&v.bytes); v.result=malloc(v.bytes); CHECK(v.result);
    CHECK(wt_kf_batch_execute(q,v.bytes,v.result,v.bytes,BUDGET,cps,&c,&invoked)==WT_KF_OK);
    CHECK(pthread_create(&thread,NULL,foreign_thread,&v)==0); CHECK(pthread_join(thread,NULL)==0);
    CHECK(v.validated==WT_KF_BAD_STATE && v.applied==WT_KF_BAD_STATE);
    CHECK(wt_kf_batch_commit(v.batch,v.result,v.bytes)==WT_KF_OK);
    free(v.result); wt_kf_batch_destroy(v.batch);
}
int main(void)
{
    forked_disjoint(); failure_and_fallback(); actual_thread_identity();
    printf("PASS KF node %d checks, forked helper and 2 owner threads\n",checks);
    return 0;
}
