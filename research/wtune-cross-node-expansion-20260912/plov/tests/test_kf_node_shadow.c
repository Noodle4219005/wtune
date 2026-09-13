#define _POSIX_C_SOURCE 200809L
#define KFB_FIXTURE_ONLY
#include "test_kf_batch.c"
#include "wt_kf_node.h"

#include <unistd.h>

static struct wt_nl_signature signature(void)
{
    return (struct wt_nl_signature){WT_KF_NODE_PHASE,29,WT_KF_BATCH_SCHEMA,
                                    4,12,1,37};
}

static struct wt_kf_node_context node_context(struct wt_nl_service *service,
    uint32_t rank,wt_kf_callback native,void *native_context,int shadow)
{
    struct wt_kf_node_context c={0};
    c.service=service; c.signature=signature(); c.rank=rank; c.thread=0;
    c.byte_budget=BUDGET; c.native=native; c.native_context=native_context;
    c.shadow_remote=shadow;
    return c;
}

struct wrong_context { struct compute compute; };

static int wrong_numeric(void **args,void *opaque)
{
    struct wrong_context *w=opaque;
    int32_t b[18]; size_t ni,nk,offset;
    if (cps(args,&w->compute)) return 1;
    for (unsigned i=0;i<18;++i) memcpy(&b[i],args[i],4);
    ni=(size_t)(b[7]-b[6]+1); nk=(size_t)(b[11]-b[10]+1);
    offset=(size_t)(b[12]-b[6])+ni*((size_t)(b[16]-b[10])+
                                    nk*(size_t)(b[14]-b[8]));
    /* A valid active output cell: guards/read-only checks still pass. */
    ((float *)args[WT_KF_RTHCUTEN])[offset]+=0.25f;
    return 0;
}

static int make_service(struct wt_nl_service *service)
{
    char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/shadow/service-XXXXXX";
    struct wt_nl_config config={2,1,32768};
    int fd=mkstemp(path);
    CHECK(fd>=0); CHECK(unlink(path)==0);
    CHECK(wt_nl_init_fd(fd,&config,service)==WT_NL_OK);
    return fd;
}

static void one_case(int shadow,int wrong)
{
    struct wt_nl_service service;
    struct fixture owner,before,expected;
    struct compute reference={0},helper_compute={0},direct={0};
    struct wrong_context wrong_state={{0}};
    struct wt_kf_node_context owner_context,helper_context;
    struct wt_kf_node_call call={0};
    struct wt_nl_ticket ticket;
    enum wt_kf_status batch_status;
    enum wt_nl_status status;
    int fd=make_service(&service);

    setup(&owner); before=owner; oracle_copy(&expected,&owner);
    CHECK(cps(expected.args,&direct)==0);
    owner_context=node_context(&service,0,cps,&reference,shadow);
    helper_context=node_context(&service,1,
        wrong ? wrong_numeric : cps,
        wrong ? (void *)&wrong_state : (void *)&helper_compute,0);
    CHECK(wt_kf_node_publish(&owner_context,owner.args,owner.cap,1,&call,
                             &batch_status)==WT_NL_OK);
    CHECK(wt_kf_node_progress(&helper_context,UINT64_C(1)<<32,&ticket)==WT_NL_OK);
    status=wt_kf_node_collect(&owner_context,&call);
    if (shadow && wrong) {
        CHECK(status==WT_NL_FAILED);
        CHECK(call.shadow_checked && call.shadow_failed && call.shadow_bytes>0);
        CHECK(reference.calls==1 && wrong_state.compute.calls==1);
        CHECK(unchanged(&owner,&before));
        CHECK(wt_kf_node_owner_execute(&owner_context,&call,0)==WT_NL_FAILED);
        CHECK(wt_kf_node_finish(&owner_context,&call)!=WT_NL_OK);
    } else {
        CHECK(status==WT_NL_OK && call.committed);
        CHECK(call.shadow_checked==shadow && !call.shadow_failed);
        CHECK(reference.calls==shadow);
        CHECK(wt_kf_node_finish(&owner_context,&call)==WT_NL_OK);
        CHECK(unchanged(&owner,&expected));
    }
    wt_kf_node_destroy(&call); wt_nl_detach(&service); close(fd);
}

int main(void)
{
    one_case(0,0); /* Default: one remote native execution, no reference. */
    one_case(1,0); /* Remote helper and private reference are bitwise equal. */
    one_case(1,1); /* Valid wire with wrong numerics cannot reach owner writes. */
    printf("PASS KF node shadow %d checks\n",checks);
    return 0;
}
