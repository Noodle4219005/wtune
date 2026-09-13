#define _POSIX_C_SOURCE 200809L
/* Reuse the exact 84-argument fixture and row-result checker. */
#define main sfclay_batch_fixture_main
#include "test_sfclay_batch.c"
#undef main

#include "wt_sfclay_node.h"

#include <fcntl.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct helper {
    struct wt_sfclay_node_context context;
    struct callback_context callback;
    enum wt_nl_status status;
    struct wt_nl_ticket ticket;
};

struct child_report {
    int ok, idle, calls, winner;
    uint32_t marker;
};

static struct wt_nl_signature signature(uint64_t epoch, uint64_t init)
{
    struct wt_nl_signature s={WT_SFCLAY_NODE_PHASE,UINT64_C(0xfeed1234),
        WT_SFCLAY_BATCH_SCHEMA,UINT64_C(0x99),epoch,WT_SFCLAY_NODE_BRANCH,init};
    return s;
}

static void *progress_thread(void *opaque)
{
    struct helper *h=opaque;
    h->status=wt_sfclay_node_progress(&h->context,
        ((uint64_t)h->context.rank<<32)|h->context.thread,&h->ticket);
    return NULL;
}

static void two_helpers(struct wt_nl_service *service,
                        struct wt_nl_signature sig,
                        uint32_t rank, struct helper h[2])
{
    pthread_t thread[2]; unsigned i;
    memset(h,0,2*sizeof(*h));
    for (i=0;i<2;++i) {
        h[i].context.service=service; h[i].context.signature=sig;
        h[i].context.rank=rank; h[i].context.thread=i;
        h[i].context.byte_budget=1u<<20; h[i].context.native=compute;
        h[i].context.native_context=&h[i].callback;
        h[i].callback.marker=(uint32_t)getpid();
        CHECK(pthread_create(&thread[i],NULL,progress_thread,&h[i])==0);
    }
    for (i=0;i<2;++i) CHECK(pthread_join(thread[i],NULL)==0);
}

struct wrong_owner {
    struct wt_sfclay_node_context context;
    struct wt_sfclay_node_call *call;
    enum wt_nl_status status;
};

static void *wrong_collect(void *opaque)
{
    struct wrong_owner *w=opaque;
    w->status=wt_sfclay_node_collect(&w->context,w->call);
    wt_sfclay_node_destroy(w->call); /* Must be a wrong-thread no-op. */
    return NULL;
}

static void make_service(const char *directory, size_t wire_bytes, int *fd,
                         size_t *mapping_bytes, struct wt_nl_service *service)
{
    struct wt_nl_config config;
    char path[1024];
    int n=snprintf(path,sizeof(path),"%s/sfclay-node.shm",directory);
    CHECK(n>0 && (size_t)n<sizeof(path));
    *fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0600); CHECK(*fd>=0);
    config.max_ranks=2; config.max_threads=2;
    config.slot_payload_bytes=2*wire_bytes+18*sizeof(int32_t)+128;
    CHECK(wt_nl_required_bytes(&config,mapping_bytes)==WT_NL_OK);
    CHECK(ftruncate(*fd,(off_t)*mapping_bytes)==0);
    CHECK(wt_nl_init_fd(*fd,&config,service)==WT_NL_OK);
}

static size_t request_bytes(struct fixture *f)
{
    struct wt_sfclay_batch *batch=NULL; const void *request; size_t bytes;
    CHECK(wt_sfclay_batch_prepare(f->arg,f->cap,1u<<20,&batch)==
          WT_SFCLAY_BATCH_OK);
    request=wt_sfclay_batch_request(batch,&bytes); CHECK(request);
    wt_sfclay_batch_destroy(batch); return bytes;
}

static void test_helper_commit(const char *directory)
{
    struct fixture f;
    struct wt_nl_service service,attached;
    struct wt_sfclay_node_context owner,bad;
    struct wt_sfclay_node_call call={0};
    struct helper same[2],helpers[2];
    struct wrong_owner wrong;
    struct child_report report={0};
    enum wt_sfclay_batch_status batch_status;
    enum wt_nl_status status;
    struct wt_nl_ticket ignored;
    uint64_t executor=0;
    int fd,pipefd[2],child_status;
    size_t mapping_bytes;
    pid_t child;
    unsigned i;
    init_fixture(&f,3,1);
    make_service(directory,request_bytes(&f),&fd,&mapping_bytes,&service);
    memset(&owner,0,sizeof(owner)); owner.service=&service;
    owner.signature=signature(7,UINT64_C(0x7073696d7462));
    owner.rank=0; owner.thread=0; owner.byte_budget=1u<<20;
    owner.native=compute;
    CHECK(wt_sfclay_node_publish(&owner,f.arg,f.cap,55,&call,&batch_status)==
          WT_NL_OK && batch_status==WT_SFCLAY_BATCH_OK);
    /* Same-rank threads and incompatible signatures cannot claim. */
    two_helpers(&service,owner.signature,0,same);
    CHECK(same[0].status==WT_NL_IDLE && same[1].status==WT_NL_IDLE);
    bad=owner; bad.rank=1; bad.signature.epoch++;
    CHECK(wt_sfclay_node_progress(&bad,1,&ignored)==WT_NL_IDLE);
    bad.signature=owner.signature; bad.signature.init_generation++;
    CHECK(wt_sfclay_node_progress(&bad,1,&ignored)==WT_NL_IDLE);
    bad.signature=owner.signature; bad.signature.phase++;
    CHECK(wt_sfclay_node_progress(&bad,1,&ignored)==WT_NL_INVALID);
    wrong.context=owner; wrong.call=&call; wrong.status=WT_NL_OK;
    { pthread_t t; CHECK(pthread_create(&t,NULL,wrong_collect,&wrong)==0);
      CHECK(pthread_join(t,NULL)==0); }
    CHECK(wrong.status==WT_NL_OWNER && call.batch && call.published);
    CHECK(pipe(pipefd)==0); child=fork(); CHECK(child>=0);
    if (!child) {
        close(pipefd[0]); wt_nl_detach(&service);
        CHECK(wt_nl_attach_fd(fd,mapping_bytes,&attached)==WT_NL_OK);
        two_helpers(&attached,owner.signature,1,helpers);
        for (i=0;i<2;++i) {
            report.ok+=helpers[i].status==WT_NL_OK;
            report.idle+=helpers[i].status==WT_NL_IDLE;
            report.calls+=helpers[i].callback.calls;
            if (helpers[i].status==WT_NL_OK) report.winner=(int)i;
        }
        report.marker=(uint32_t)getpid();
        CHECK(write(pipefd[1],&report,sizeof(report))==(ssize_t)sizeof(report));
        wt_nl_detach(&attached); close(pipefd[1]); close(fd); _exit(0);
    }
    close(pipefd[1]);
    CHECK(read(pipefd[0],&report,sizeof(report))==(ssize_t)sizeof(report));
    CHECK(waitpid(child,&child_status,0)==child && WIFEXITED(child_status) &&
          WEXITSTATUS(child_status)==0);
    CHECK(report.ok==1 && report.idle==1 && report.calls==1);
    CHECK(!memcmp(f.arg[12],f.initial[12],f.cap[12]));
    CHECK(wt_nl_ticket_executor(&service,0,0,&call.ticket,&executor)==WT_NL_OK);
    CHECK(executor==((UINT64_C(1)<<32)|(uint64_t)report.winner));
    status=wt_sfclay_node_collect(&owner,&call); CHECK(status==WT_NL_OK);
    CHECK(call.committed); check_committed(&f,3,report.marker);
    CHECK(wt_sfclay_node_finish(&owner,&call)==WT_NL_OK);
    wt_sfclay_node_destroy(&call);
    close(pipefd[0]); wt_nl_detach(&service); close(fd); destroy_fixture(&f);
}

static void test_owner_and_failure(const char *directory)
{
    struct fixture f; struct wt_nl_service service;
    struct wt_sfclay_node_context owner; struct wt_sfclay_node_call call={0};
    struct callback_context callback={UINT32_C(0xaabbccdd),0,0};
    enum wt_sfclay_batch_status batch_status; uint64_t executor;
    int fd; size_t mapping_bytes;
    init_fixture(&f,4,0); make_service(directory,request_bytes(&f),&fd,
                                      &mapping_bytes,&service);
    memset(&owner,0,sizeof(owner)); owner.service=&service;
    owner.signature=signature(8,UINT64_C(0x7073696d7462));
    owner.rank=0; owner.thread=0; owner.byte_budget=1u<<20;
    owner.native=compute; owner.native_context=&callback;
    CHECK(wt_sfclay_node_publish(&owner,f.arg,f.cap,56,&call,&batch_status)==WT_NL_OK);
    CHECK(wt_sfclay_node_owner_execute(&owner,&call,UINT64_C(0))==WT_NL_OK);
    CHECK(wt_nl_ticket_executor(&service,0,0,&call.ticket,&executor)==WT_NL_OK &&
          executor==0);
    CHECK(wt_sfclay_node_collect(&owner,&call)==WT_NL_OK);
    check_committed(&f,4,callback.marker);
    CHECK(wt_sfclay_node_finish(&owner,&call)==WT_NL_OK);
    wt_sfclay_node_destroy(&call); destroy_fixture(&f);

    init_fixture(&f,3,1); memset(&call,0,sizeof(call)); callback.mode=3;
    callback.calls=0;
    CHECK(wt_sfclay_node_publish(&owner,f.arg,f.cap,57,&call,&batch_status)==WT_NL_OK);
    CHECK(wt_sfclay_node_owner_execute(&owner,&call,1)==WT_NL_FAILED);
    CHECK(wt_sfclay_node_owner_execute(&owner,&call,2)==WT_NL_FAILED);
    CHECK(callback.calls==1 && wt_sfclay_node_collect(&owner,&call)==WT_NL_FAILED);
    wt_sfclay_node_destroy(&call); destroy_fixture(&f);
    wt_nl_detach(&service); close(fd);
}

int main(int argc, char **argv)
{
    CHECK(argc==2); test_helper_commit(argv[1]); test_owner_and_failure(argv[1]);
    puts("SFCLAY node-local tests passed"); return 0;
}
