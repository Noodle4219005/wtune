#define _POSIX_C_SOURCE 200809L
#define main sfclay_batch_fixture_main
#include "test_sfclay_batch.c"
#undef main
#include "wt_sfclay_node.h"

#include <unistd.h>

static struct wt_nl_signature shadow_signature(void)
{
    return (struct wt_nl_signature){WT_SFCLAY_NODE_PHASE,29,
        WT_SFCLAY_BATCH_SCHEMA,4,12,WT_SFCLAY_NODE_BRANCH,37};
}

static struct wt_sfclay_node_context node_context(
    struct wt_nl_service *service,uint32_t rank,
    wt_sfclay_batch_callback native,void *native_context,int shadow)
{
    struct wt_sfclay_node_context c={0};
    c.service=service; c.signature=shadow_signature();
    c.rank=rank; c.thread=0; c.byte_budget=1u<<20;
    c.native=native; c.native_context=native_context; c.shadow_remote=shadow;
    return c;
}

struct wrong_context { struct callback_context callback; };

static int wrong_numeric(void **args,void *opaque)
{
    struct wrong_context *wrong=opaque;
    uint32_t value;
    unsigned char *cell;
    if (compute(args,&wrong->callback)) return 1;
    /* Argument 12 is writable. Change one active cell without touching guards,
     * request metadata, read-only storage, or the encoded layout. */
    cell=(unsigned char *)args[12]+active_offset(args,12);
    memcpy(&value,cell,4); value^=UINT32_C(1); memcpy(cell,&value,4);
    return 0;
}

static size_t wire_bytes(struct fixture *fixture)
{
    struct wt_sfclay_batch *batch=NULL;
    const void *request;
    size_t bytes=0;
    CHECK(wt_sfclay_batch_prepare(fixture->arg,fixture->cap,1u<<20,&batch)==
          WT_SFCLAY_BATCH_OK);
    request=wt_sfclay_batch_request(batch,&bytes); CHECK(request && bytes);
    wt_sfclay_batch_destroy(batch); return bytes;
}

static int make_service(struct wt_nl_service *service,size_t wire)
{
    char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/sfclay/shadow/service-XXXXXX";
    struct wt_nl_config config={2,1,2*wire+18*sizeof(int32_t)+128};
    int fd=mkstemp(path);
    CHECK(fd>=0); CHECK(unlink(path)==0);
    CHECK(wt_nl_init_fd(fd,&config,service)==WT_NL_OK);
    return fd;
}

static int unchanged(const struct fixture *fixture)
{
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a)
        if (fixture->arg[a] &&
            memcmp(fixture->arg[a],fixture->initial[a],fixture->cap[a]))
            return 0;
    return 1;
}

static void one_case(int shadow,int wrong_result)
{
    const uint32_t marker=UINT32_C(0x1234abcd);
    struct wt_nl_service service;
    struct fixture owner;
    struct callback_context reference={marker,0,0},helper={marker,0,0};
    struct wrong_context wrong={{marker,0,0}};
    struct wt_sfclay_node_context owner_context,helper_context;
    struct wt_sfclay_node_call call={0};
    struct wt_nl_ticket ticket;
    enum wt_sfclay_batch_status batch_status;
    enum wt_nl_status status;
    int fd;

    init_fixture(&owner,3,1); fd=make_service(&service,wire_bytes(&owner));
    owner_context=node_context(&service,0,compute,&reference,shadow);
    helper_context=node_context(&service,1,
        wrong_result ? wrong_numeric : compute,
        wrong_result ? (void *)&wrong : (void *)&helper,0);
    CHECK(wt_sfclay_node_publish(&owner_context,owner.arg,owner.cap,1,&call,
                                 &batch_status)==WT_NL_OK);
    CHECK(wt_sfclay_node_progress(&helper_context,UINT64_C(1)<<32,&ticket)==
          WT_NL_OK);
    status=wt_sfclay_node_collect(&owner_context,&call);
    if (shadow && wrong_result) {
        CHECK(status==WT_NL_FAILED);
        CHECK(call.shadow_checked && call.shadow_failed && call.shadow_bytes>0);
        CHECK(reference.calls==1 && wrong.callback.calls==1);
        CHECK(unchanged(&owner));
        CHECK(wt_sfclay_node_owner_execute(&owner_context,&call,0)==WT_NL_FAILED);
        CHECK(wt_sfclay_node_finish(&owner_context,&call)!=WT_NL_OK);
    } else {
        CHECK(status==WT_NL_OK && call.committed);
        CHECK(call.shadow_checked==shadow && !call.shadow_failed);
        CHECK(reference.calls==shadow);
        CHECK(wt_sfclay_node_finish(&owner_context,&call)==WT_NL_OK);
        check_committed(&owner,3,marker);
    }
    wt_sfclay_node_destroy(&call); wt_nl_detach(&service); close(fd);
    destroy_fixture(&owner);
}

int main(void)
{
    one_case(0,0);
    one_case(1,0);
    one_case(1,1);
    puts("PASS SFCLAY node shadow");
    return 0;
}
