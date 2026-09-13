#define _POSIX_C_SOURCE 200809L
#define main mynn_node_fixture_main
#include "test_mynn_node.c"
#undef main

struct wrong_context { struct callback callback; };

static int wrong_numeric(void *opaque,void *argv[WT_MYNN_NARG])
{
    struct wrong_context *wrong=opaque;
    unsigned char *cell;
    if(native(&wrong->callback,argv)) return 1;
    /* PBLH is an always-writable active 2-D cell. The wire remains structurally
     * valid, so only the private-reference bit comparison can reject this. */
    cell=(unsigned char *)argv[WT_MYNN_PBLH]+active_offset(argv,WT_MYNN_PBLH);
    *cell^=1;
    return 0;
}

static int shadow_service(struct wt_nl_service *service)
{
    char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/mynn/shadow/service-XXXXXX";
    struct wt_nl_config config={2,1,65536};
    int fd=mkstemp(path);
    CHECK(fd>=0); CHECK(!unlink(path));
    CHECK(wt_nl_init_fd(fd,&config,service)==WT_NL_OK);
    return fd;
}

static struct wt_mynn_node_context shadow_context(struct wt_nl_service *service,
    unsigned rank,wt_mynn_batch_callback callback,void *opaque)
{
    struct wt_mynn_node_context c={0};
    c.service=service; c.signature=signature(); c.rank=rank; c.thread=0;
    c.byte_budget=BUDGET; c.native=callback; c.native_context=opaque;
    return c;
}

static void one_case(int shadow,int wrong_result)
{
    struct wt_nl_service service;
    struct fixture owner;
    struct callback reference={0x5a,0,0},helper={0x5a,0,0};
    struct wrong_context wrong={{0x5a,0,0}};
    struct wt_mynn_node_context oc,hc;
    struct wt_mynn_node_call call={0};
    struct wt_nl_ticket ticket;
    enum wt_mynn_status batch_status;
    enum wt_nl_status status;
    uint64_t executor=0;
    int fd=shadow_service(&service);

    setup(&owner,1);
    oc=shadow_context(&service,0,native,&reference);
    hc=shadow_context(&service,1,wrong_result ? wrong_numeric : native,
                      wrong_result ? (void *)&wrong : (void *)&helper);
    CHECK(wt_mynn_node_publish(&oc,owner.argv,owner.cap,&layout_config,1,&call,
                               &batch_status)==WT_NL_OK);
    CHECK(wt_mynn_node_progress(&hc,UINT64_C(1)<<32,&ticket)==WT_NL_OK);
    CHECK(wt_nl_ticket_executor(&service,0,0,&call.ticket,&executor)==WT_NL_OK);
    CHECK((uint32_t)(executor>>32)==1);
    call.shadow_remote=shadow;
    status=wt_mynn_node_collect(&oc,&call);
    if(shadow && wrong_result) {
        CHECK(status==WT_NL_FAILED);
        CHECK(call.shadow_checked && call.shadow_failed && call.shadow_bytes>0);
        CHECK(reference.calls==1 && wrong.callback.calls==1);
        unchanged_except_rows(&owner,0,0);
        CHECK(wt_mynn_node_owner_execute(&oc,&call,0)==WT_NL_FAILED);
        CHECK(wt_mynn_node_finish(&oc,&call)!=WT_NL_OK);
    } else {
        CHECK(status==WT_NL_OK && call.committed);
        CHECK(call.shadow_checked==shadow && !call.shadow_failed);
        CHECK(reference.calls==shadow);
        CHECK(wt_mynn_node_finish(&oc,&call)==WT_NL_OK);
        unchanged_except_rows(&owner,1,0x5a);
    }
    wt_mynn_node_destroy(&call); wt_nl_detach(&service); close(fd);
    destroy(&owner);
}

int main(void)
{
    one_case(0,0);
    one_case(1,0);
    one_case(1,1);
    puts("PASS MYNN node shadow");
    return 0;
}
