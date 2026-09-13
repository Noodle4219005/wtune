#define _POSIX_C_SOURCE 200809L
#define main sfclay_original_fixture_main
#include "sfclay_fortran_fixture.c"
#undef main

#include "wt_sfclay_node.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

extern float __module_sf_sfclay_MOD_psimtb[1001];
extern float __module_sf_sfclay_MOD_psihtb[1001];

static uint64_t hash_tables(void)
{
    const unsigned char *p=(const unsigned char *)__module_sf_sfclay_MOD_psimtb;
    const unsigned char *q=(const unsigned char *)__module_sf_sfclay_MOD_psihtb;
    uint64_t h=UINT64_C(1469598103934665603); size_t i;
    for (i=0;i<1001*sizeof(float);++i) { h^=p[i]; h*=UINT64_C(1099511628211); }
    for (i=0;i<1001*sizeof(float);++i) { h^=q[i]; h*=UINT64_C(1099511628211); }
    return h ? h : 1;
}

static int native_real(void *args[WT_SFCLAY_NARG], void *context)
{ (void)context; wt_sfclay_bridge_real(args); return 0; }

static struct wt_nl_signature native_signature(uint64_t table_hash)
{
    struct wt_nl_signature s={WT_SFCLAY_NODE_PHASE,UINT64_C(0xdfd7aae5ef1f0bd9),
        WT_SFCLAY_BATCH_SCHEMA,UINT64_C(0x01020304),UINT64_C(1),
        WT_SFCLAY_NODE_BRANCH,table_hash};
    return s;
}

static void compare_full(const struct context *direct,
                         const struct context *stolen)
{
    int i;
    for (i=0;i<WT_SFCLAY_NARG;++i) if (direct->arg[i])
        check(!memcmp(direct->arg[i],stolen->arg[i],direct->cap[i]),
              "native helper/direct full backing mismatch\n");
}

static void run_native_cases(struct wt_nl_service *service, int command_fd,
                             int result_fd, uint64_t table_hash)
{
    int stable,water,isftc,iz0,scm,sequence=0;
    for (stable=0;stable<2;++stable) for (water=0;water<2;++water)
      for (isftc=0;isftc<3;++isftc) for (iz0=0;iz0<3;++iz0)
       for (scm=0;scm<2;++scm) {
        struct context direct,stolen;
        struct wt_sfclay_node_context owner={0};
        struct wt_sfclay_node_call call={0};
        enum wt_sfclay_batch_status batch_status;
        enum wt_nl_status helper_status;
        char command='g';
        init_context(&direct,stable,water,isftc,iz0,scm,1);
        init_context(&stolen,stable,water,isftc,iz0,scm,1);
        tile(&direct,0,3); tile(&stolen,0,3);
        wt_sfclay_bridge_real(direct.arg);
        owner.service=service; owner.signature=native_signature(table_hash);
        owner.rank=0; owner.thread=0; owner.byte_budget=1u<<20;
        owner.native=native_real;
        check(wt_sfclay_node_publish(&owner,stolen.arg,stolen.cap,
              (uint64_t)++sequence,&call,&batch_status)==WT_NL_OK,
              "native node publish\n");
        check(write(command_fd,&command,1)==1,"native helper command\n");
        check(read(result_fd,&helper_status,sizeof(helper_status))==
              (ssize_t)sizeof(helper_status),"native helper result\n");
        check(helper_status==WT_NL_OK,"native helper execute\n");
        check(wt_sfclay_node_collect(&owner,&call)==WT_NL_OK,
              "native owner collect\n");
        check(wt_sfclay_node_finish(&owner,&call)==WT_NL_OK,
              "native owner finish\n");
        wt_sfclay_node_destroy(&call); compare_full(&direct,&stolen);
        destroy_context(&direct); destroy_context(&stolen);
      }
    {
        struct context direct,stolen;
        struct wt_sfclay_node_context owner={0};
        struct wt_sfclay_node_call call={0};
        enum wt_sfclay_batch_status batch_status;
        enum wt_nl_status helper_status;
        char command='g';
        init_context(&direct,0,0,0,0,0,0);
        init_context(&stolen,0,0,0,0,0,0);
        tile(&direct,0,3); tile(&stolen,0,3);
        wt_sfclay_bridge_real(direct.arg);
        owner.service=service; owner.signature=native_signature(table_hash);
        owner.rank=0; owner.thread=0; owner.byte_budget=1u<<20;
        owner.native=native_real;
        check(wt_sfclay_node_publish(&owner,stolen.arg,stolen.cap,
              (uint64_t)++sequence,&call,&batch_status)==WT_NL_OK,
              "optional-control node publish\n");
        check(write(command_fd,&command,1)==1,"optional helper command\n");
        check(read(result_fd,&helper_status,sizeof(helper_status))==
              (ssize_t)sizeof(helper_status),"optional helper result\n");
        check(helper_status==WT_NL_OK,"optional helper execute\n");
        check(wt_sfclay_node_collect(&owner,&call)==WT_NL_OK,
              "optional owner collect\n");
        check(wt_sfclay_node_finish(&owner,&call)==WT_NL_OK,
              "optional owner finish\n");
        wt_sfclay_node_destroy(&call); compare_full(&direct,&stolen);
        destroy_context(&direct); destroy_context(&stolen);
    }
}

int main(int argc, char **argv)
{
    struct context sizing;
    struct wt_sfclay_batch *batch=NULL;
    struct wt_nl_config config={2,1,0};
    struct wt_nl_service service,attached;
    const void *request; size_t wire,mapping; uint64_t table_hash;
    int true_value=1,fd,command[2],result[2],status;
    char path[1024],byte; pid_t child;
    check(argc==2,"artifact directory\n");
    __module_sf_sfclay_MOD_sfclayinit(&true_value); table_hash=hash_tables();
    init_context(&sizing,0,0,0,0,0,1); tile(&sizing,0,3);
    check(wt_sfclay_batch_prepare(sizing.arg,sizing.cap,1u<<20,&batch)==
          WT_SFCLAY_BATCH_OK,"native sizing prepare\n");
    request=wt_sfclay_batch_request(batch,&wire); check(request!=NULL,"native sizing request\n");
    wt_sfclay_batch_destroy(batch); destroy_context(&sizing);
    config.slot_payload_bytes=2*wire+18*sizeof(int32_t)+128;
    check(wt_nl_required_bytes(&config,&mapping)==WT_NL_OK,"native mapping size\n");
    check(snprintf(path,sizeof(path),"%s/sfclay-native.shm",argv[1])>0,"native path\n");
    fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0600); check(fd>=0,"native open\n");
    check(ftruncate(fd,(off_t)mapping)==0,"native truncate\n");
    check(wt_nl_init_fd(fd,&config,&service)==WT_NL_OK,"native init fd\n");
    check(pipe(command)==0 && pipe(result)==0,"native pipes\n");
    child=fork(); check(child>=0,"native fork\n");
    if (!child) {
        struct wt_sfclay_node_context helper={0};
        struct wt_nl_ticket executed;
        enum wt_nl_status helper_status;
        close(command[1]); close(result[0]); wt_nl_detach(&service);
        check(wt_nl_attach_fd(fd,mapping,&attached)==WT_NL_OK,"native attach\n");
        helper.service=&attached; helper.signature=native_signature(table_hash);
        helper.rank=1; helper.thread=0; helper.byte_budget=1u<<20;
        helper.native=native_real;
        while (read(command[0],&byte,1)==1) {
            helper_status=wt_sfclay_node_progress(&helper,UINT64_C(1)<<32,&executed);
            check(write(result[1],&helper_status,sizeof(helper_status))==
                  (ssize_t)sizeof(helper_status),"native status write\n");
        }
        wt_nl_detach(&attached); close(fd); _exit(0);
    }
    close(command[0]); close(result[1]);
    run_native_cases(&service,command[1],result[0],table_hash);
    close(command[1]); close(result[0]);
    check(waitpid(child,&status,0)==child && WIFEXITED(status) &&
          WEXITSTATUS(status)==0,"native child exit\n");
    wt_nl_detach(&service); close(fd);
    puts("SFCLAY frozen GNU helper differential passed 73 cases"); return 0;
}
