#define _POSIX_C_SOURCE 200809L
#include "wt_mynn_node.h"
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"MYNN node failure %d: %s\n",__LINE__,#x); abort(); } } while (0)
#define BUDGET ((size_t)64*1024*1024)
#include "mynn-node-profile.h"
extern void mynn_parameter_bits(int32_t *);
static unsigned calls,continuations,remote_batches;
static uint64_t initialized;
static uint64_t fingerprint(void)
{
    int32_t bits[MYNN_PARAMETER_COUNT];
    const unsigned char *p=(const unsigned char *)bits;
    uint64_t h=UINT64_C(14695981039346656037);
    mynn_parameter_bits(bits);
    for(size_t i=0;i<sizeof(bits);++i) { h^=p[i]; h*=UINT64_C(1099511628211); }
    return h;
}
struct native_context { void (*invoke)(void **); unsigned count; pid_t pid; };
static int native(void *opaque,void *argv[WT_MYNN_NARG])
{
    struct native_context *c=opaque;
    CHECK(getpid()==c->pid && fingerprint()==initialized);
    ++c->count; c->invoke(argv);
    CHECK(fingerprint()==initialized);
    return 0;
}
void wt_mynn_bridge_dispatch(void **argv,void (*invoke)(void **),size_t nargs)
{
    struct wt_nl_service service;
    struct wt_nl_config service_config={2,2,1024*1024};
    struct wt_mynn_layout_config layout={1,1,1,BUDGET};
    struct wt_mynn_node_context owner={0};
    struct wt_mynn_node_call batch[2]={{0}};
    struct native_context native_owner={invoke,0,0};
    struct wt_nl_counters counters;
    alignas(64) unsigned char scalar[2][WT_MYNN_NARG][128];
    void *row[2][WT_MYNN_NARG];
    size_t capacity[WT_MYNN_NARG];
    int fd,status; pid_t child;
    char path[]=MYNN_ARTIFACT_DIRECTORY "/service-XXXXXX";
    CHECK(argv && invoke && nargs==WT_MYNN_NARG);
    alarm(20);
    ++calls;
    if(*(int32_t *)argv[WT_MYNN_INITFLAG]==1) {
        CHECK(calls==1); invoke(argv); initialized=fingerprint(); CHECK(initialized);
        alarm(0); return;
    }
    CHECK(calls>=2 && calls<=3 && initialized==fingerprint());
    CHECK(*(int32_t *)argv[WT_MYNN_INITFLAG]==0);
    CHECK(*(int32_t *)argv[WT_MYNN_JTS]==2 && *(int32_t *)argv[WT_MYNN_JTE]==3);
    CHECK(*(int32_t *)argv[WT_MYNN_IMS]==1 && *(int32_t *)argv[WT_MYNN_IME]==4);
    CHECK(*(int32_t *)argv[WT_MYNN_JMS]==1 && *(int32_t *)argv[WT_MYNN_JME]==4);
    CHECK(*(int32_t *)argv[WT_MYNN_KMS]==1 && *(int32_t *)argv[WT_MYNN_KME]==33);
    ++continuations;
    fd=mkstemp(path); CHECK(fd>=0); CHECK(unlink(path)==0);
    CHECK(wt_nl_init_fd(fd,&service_config,&service)==WT_NL_OK);
    native_owner.pid=getpid();
    owner.service=&service; owner.rank=0; owner.byte_budget=BUDGET;
    owner.native=native; owner.native_context=&native_owner;
    owner.signature=(struct wt_nl_signature){WT_MYNN_NODE_PHASE,MYNN_PROFILE,
        WT_MYNN_BATCH_SCHEMA,4,calls,WT_MYNN_BATCH_SCHEMA,initialized};
    for(unsigned a=0;a<WT_MYNN_NARG;++a)
        capacity[a]=!wt_mynn_args[a].rank ? 4 : wt_mynn_args[a].rank==2 ? 4*4*4 : 4*33*4*4;
    for(unsigned j=0;j<2;++j) {
        enum wt_mynn_status bs;
        memcpy(row[j],argv,sizeof(row[j]));
        /* The typed GNU caller may pool read-only scalar constants. Preserve
         * each scalar's bits/alignment in separate storage before admission. */
        for(unsigned a=0;a<WT_MYNN_NARG;++a) if(!wt_mynn_args[a].rank) {
            row[j][a]=scalar[j][a]+((uintptr_t)argv[a]&63);
            memcpy(row[j][a],argv[a],4);
        }
        *(int32_t *)row[j][WT_MYNN_JTS]=*(int32_t *)row[j][WT_MYNN_JTE]=(int32_t)j+2;
        owner.thread=j;
        status=wt_mynn_node_publish(&owner,row[j],capacity,&layout,calls*2+j,&batch[j],&bs);
        if(status!=WT_NL_OK) fprintf(stderr,"publish row=%u service=%d batch=%d\n",j,status,bs);
        CHECK(status==WT_NL_OK);
    }
    child=fork(); CHECK(child>=0);
    if(!child) {
        alarm(20);
        size_t bytes=service.mapping_bytes;
        struct wt_nl_service attached;
        struct wt_nl_ticket ticket;
        struct native_context helper_native={invoke,0,getpid()};
        wt_nl_detach(&service);
        CHECK(wt_nl_attach_fd(fd,bytes,&attached)==WT_NL_OK);
        owner.service=&attached; owner.rank=1; owner.thread=0; owner.native_context=&helper_native;
        CHECK(fingerprint()==initialized);
        CHECK(wt_mynn_node_progress(&owner,UINT64_C(1)<<32,&ticket)==WT_NL_OK);
        CHECK(wt_mynn_node_progress(&owner,UINT64_C(1)<<32,&ticket)==WT_NL_OK);
        CHECK(helper_native.count==2);
        fprintf(stderr,"MYNN_GNU_HELPER pid=%lu owner_pid=%lu epoch=%u rows=2 init=%llu\n",
            (unsigned long)getpid(),(unsigned long)getppid(),calls,(unsigned long long)initialized);
        wt_nl_detach(&attached); close(fd); _exit(0);
    }
    CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
    for(unsigned j=0;j<2;++j) {
        uint64_t tag=0;
        owner.thread=j;
        CHECK(wt_mynn_node_collect(&owner,&batch[j])==WT_NL_OK);
        CHECK(wt_nl_ticket_executor(&service,0,j,&batch[j].ticket,&tag)==WT_NL_OK);
        CHECK((tag>>32)==1);
        CHECK(wt_mynn_node_finish(&owner,&batch[j])==WT_NL_OK);
        wt_mynn_node_destroy(&batch[j]);
    }
    CHECK(wt_nl_get_counters(&service,&counters)==WT_NL_OK);
    CHECK(counters.helper_executed==2 && counters.committed==2 && counters.self_executed==0);
    CHECK(native_owner.count==0 && fingerprint()==initialized);
    remote_batches+=(unsigned)counters.helper_executed;
    wt_nl_detach(&service); close(fd); alarm(0);
}
__attribute__((destructor)) static void report(void)
{
    CHECK(calls==3 && continuations==2 && remote_batches==4);
    fprintf(stderr,"MYNN_GNU_NODE calls=%u continuation=%u helper=%u full132=1\n",
        calls,continuations,remote_batches);
}
