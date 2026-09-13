#define _POSIX_C_SOURCE 200809L
#define KFB_FIXTURE_ONLY
#ifndef STORAGE
#define STORAGE 1080
#endif
#include "test_kf_batch.c"
#include "kf-node-live-abi.h"
#include "wt_kf_node_live.h"
#include "wt_kf_node_shm.h"
#include "wt_kf_bootstrap.h"
#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifdef WT_KF_NODE_GNU_CPS
#define KF_TABLE extern unsigned char
extern void wt_kf_fixture_gnu_init(void);
extern void wt_kf_fixture_gnu_compute(void **);
extern void wt_kf_fixture_gnu_sounding(void *,int);
#else
#define KF_TABLE unsigned char
#endif
KF_TABLE module_cu_kfeta_mp_ttab_[220000],module_cu_kfeta_mp_qstab_[220000];
KF_TABLE module_cu_kfeta_mp_the0k_[880],module_cu_kfeta_mp_alu_[800];
KF_TABLE module_cu_kfeta_mp_rdpr_[4],module_cu_kfeta_mp_rdthk_[4],module_cu_kfeta_mp_plutop_[4];
static void *init_expected[35];
static int init_storage[35],init_seed=17,init_calls,rank_id,scenario;
static _Atomic unsigned native_calls,native_rows[9];
static _Atomic unsigned *peer_seen;
struct grid { struct fixture tiles[2]; int id; };
/* Deterministic reproduction of a legal helper reservation rollback. The real
 * service owns its CLAIMED_HELPER state while compatibility is paused. */
extern enum wt_nl_status __real_wt_nl_try_execute(struct wt_nl_service *,uint32_t,uint32_t,
    const struct wt_nl_signature *,wt_nl_geometry_compatible_fn,wt_nl_execute_fn,void *,uint64_t,struct wt_nl_ticket *);
extern enum wt_nl_status __real_wt_nl_owner_execute(struct wt_nl_service *,uint32_t,uint32_t,
    const struct wt_nl_ticket *,wt_nl_execute_fn,void *,uint64_t);
struct paused_compatibility {
    wt_nl_geometry_compatible_fn compatible; wt_nl_execute_fn execute; void *context;
};
static int pause_reject(void *opaque,const struct wt_nl_descriptor *d,const void *g,size_t n)
{
    struct timespec pause={0,1000000}; unsigned attempts=0;
    (void)opaque; (void)d; (void)g; (void)n;
    if(atomic_load(&peer_seen[3])) return 0;
    atomic_store(&peer_seen[1],1);
    while(!atomic_load(&peer_seen[2])) { if(++attempts>2000) abort(); nanosleep(&pause,NULL); }
    atomic_store(&peer_seen[3],1); /* Retire after returning reservation. */
    fprintf(stderr,"ROLLBACK helper reserved, owner observed pending, helper retired\n");
    return 0;
}
static int paused_execute(void *opaque,const struct wt_nl_descriptor *d,
    const void *g,size_t gn,const void *q,size_t qn,void *r,size_t rn,size_t *written)
{
    struct paused_compatibility *p=opaque;
    return p->execute(p->context,d,g,gn,q,qn,r,rn,written);
}
enum wt_nl_status __wrap_wt_nl_try_execute(struct wt_nl_service *s,uint32_t rank,uint32_t thread,
    const struct wt_nl_signature *signature,wt_nl_geometry_compatible_fn compatible,
    wt_nl_execute_fn execute,void *context,uint64_t tag,struct wt_nl_ticket *ticket)
{
    struct paused_compatibility p={compatible,execute,context};
    if(scenario!=10 || rank!=1) return __real_wt_nl_try_execute(s,rank,thread,signature,compatible,execute,context,tag,ticket);
    if(atomic_load(&peer_seen[3])) return WT_NL_IDLE;
    return __real_wt_nl_try_execute(s,rank,thread,signature,pause_reject,paused_execute,&p,tag,ticket);
}
enum wt_nl_status __wrap_wt_nl_owner_execute(struct wt_nl_service *s,uint32_t rank,uint32_t thread,
    const struct wt_nl_ticket *ticket,wt_nl_execute_fn execute,void *context,uint64_t tag)
{
    enum wt_nl_status result=__real_wt_nl_owner_execute(s,rank,thread,ticket,execute,context,tag);
    if(scenario==10 && rank==0 && result==WT_NL_PENDING) atomic_store(&peer_seen[2],1);
    return result;
}


void kf_fixture_init_real(KF_INIT_FORMALS)
{
    void *actual[35]=KF_INIT_ARGV;
    for(unsigned a=0;a<35;++a) if(actual[a]!=init_expected[a]) abort();
    ++init_calls;
#ifdef WT_KF_NODE_GNU_CPS
    wt_kf_fixture_gnu_init();
#else
    memset(module_cu_kfeta_mp_ttab_,init_seed,sizeof(module_cu_kfeta_mp_ttab_));
    memset(module_cu_kfeta_mp_qstab_,init_seed+1,sizeof(module_cu_kfeta_mp_qstab_));
    memset(module_cu_kfeta_mp_the0k_,init_seed+2,sizeof(module_cu_kfeta_mp_the0k_));
    memset(module_cu_kfeta_mp_alu_,init_seed+3,sizeof(module_cu_kfeta_mp_alu_));
    memset(module_cu_kfeta_mp_rdpr_,init_seed+4,4);
    memset(module_cu_kfeta_mp_rdthk_,init_seed+5,4);
    memset(module_cu_kfeta_mp_plutop_,init_seed+6,4);
#endif
}
static void initialize_native(void)
{
    for(unsigned a=0;a<35;++a) init_expected[a]=&init_storage[a];
    module_cu_kfeta_mp_kf_eta_init_(KF_INIT_FROM_ARRAY(init_expected));
}
void kf_fixture_cps_real(KF_CPS_FORMALS)
{
    void *actual[WT_KF_NARG]=KF_CPS_ARGV;
    struct compute compute={0};
    int j=*(int32_t *)actual[WT_KF_JTS];
    size_t nk=(size_t)(*(int32_t *)actual[WT_KF_KME]-*(int32_t *)actual[WT_KF_KMS]+1);
    size_t offset=1+6u*(1+nk*(size_t)(j+1));
#ifdef WT_KF_NODE_GNU_CPS
    int input_rank=((float *)actual[WT_KF_U])[offset]<5.5f ? 0 : 1;
    (void)compute;
#else
    int input_rank=((float *)actual[WT_KF_RHO])[offset]<150 ? 0 : 1;
#endif
    struct timespec delay={0,1000000};
    atomic_fetch_add(&native_calls,1);
    if(scenario==10 && input_rank==rank_id) atomic_fetch_add(&native_rows[j],1);
    if (scenario>=4 && scenario<=7) { fprintf(stderr,"KF_NATIVE_ENTER j=%d\n",j); fflush(stderr); }
    if(scenario==4) { initialize_native(); abort(); }
    if(scenario==5) { delay.tv_nsec=10000000; nanosleep(&delay,NULL); }
    /* A deterministic slow owner head gives the other actual process time to
     * execute useful donated work; the production scheduler has no such wait. */
    if((scenario==0 || scenario==10 || scenario==12) && rank_id==0 &&
       input_rank==0 && !(j&1)) {
        unsigned tries=0;
        atomic_store(&peer_seen[4],1);
        while(!atomic_load(&peer_seen[scenario==10 ? 1 : 0])) {
            if(++tries>3000) abort();
            nanosleep(&delay,NULL);
        }
    }
#ifdef WT_KF_NODE_GNU_CPS
    wt_kf_fixture_gnu_compute(actual);
#else
    if(cps(actual,&compute)) abort();
#endif
    if(scenario==6 && (j&1)) ((unsigned char *)actual[WT_KF_W])[-1]=0;
    if(scenario==7 && (j&1)) ((float *)actual[WT_KF_RHO])[offset]+=1;
    if(rank_id==1 && input_rank==0) atomic_fetch_add(peer_seen,1);
}
void wt_grid_identity_(void *opaque,int *id,int *status)
{ struct grid *g=opaque; *id=g->id; *status=WT_SPAN_OK; }
void wt_grid_capture_(void *opaque,void **context,const size_t *max_nodes,int *tiles,int *status)
{
    struct grid *g=opaque;
    (void)max_nodes; *status=WT_SPAN_OK; *tiles=2;
    for(unsigned a=0;a<WT_KF_NARG;++a) if(wt_kf_shape[a]!=WT_KF_SCALAR && g->tiles[0].args[a]) {
        int type=a==WT_KF_CU_ACT_FLAG ? WT_SPAN_LOGICAL : WT_SPAN_REAL;
        int rank=wt_kf_shape[a]==WT_KF_3D ? 3 : 2;
        if(wt_live_spans_add(*context,g->tiles[0].args[a],g->tiles[0].cap[a],type,4,rank)!=WT_SPAN_OK) *status=WT_SPAN_INVALID;
    }
}
static void setup_grid(struct grid *g,int rank)
{
#ifdef WT_KF_NODE_GNU_CPS
    wt_kf_fixture_gnu_sounding(&g->tiles[0],1); wt_kf_fixture_gnu_sounding(&g->tiles[1],1);
#else
    setup(&g->tiles[0]); setup(&g->tiles[1]);
#endif
    g->id=4;
    g->tiles[0].scalar[WT_KF_KTAU]=g->tiles[1].scalar[WT_KF_KTAU]=12;
    g->tiles[1].scalar[WT_KF_JTS]=2; g->tiles[1].scalar[WT_KF_JTE]=3;
    if(scenario==8) {
        g->tiles[0].scalar[WT_KF_JME]=g->tiles[1].scalar[WT_KF_JME]=7;
        g->tiles[0].scalar[WT_KF_JTE]=3;
        g->tiles[1].scalar[WT_KF_JTS]=4; g->tiles[1].scalar[WT_KF_JTE]=7;
    }
    for(unsigned a=0;a<WT_KF_NARG;++a) if(wt_kf_shape[a]!=WT_KF_SCALAR)
        g->tiles[1].args[a]=g->tiles[0].args[a];
    for(unsigned i=0;i<STORAGE/4;++i) {
#ifdef WT_KF_NODE_GNU_CPS
        g->tiles[0].arrays[WT_KF_U][i]=(float)(5+rank);
#else
        g->tiles[0].arrays[WT_KF_RHO][i]=(float)(100+rank*100);
#endif
    }
}
static void env_rank(int rank,unsigned job)
{
    char text[32];
    snprintf(text,sizeof(text),"%u",job); CHECK(setenv("SLURM_JOB_ID",text,1)==0);
    CHECK(setenv("SLURM_STEP_ID","7",1)==0);
    CHECK(setenv("SLURM_LOCALID",rank ? "1" : "0",1)==0);
    CHECK(setenv("SLURM_STEP_TASKS_PER_NODE","2",1)==0);
    CHECK(setenv("WT_KF_NODE_G0","1",1)==0);
    CHECK(setenv("WT_KF_NODE_G0_SHADOW",scenario==12 ? "1" : "0",1)==0);
    CHECK(setenv("WT_KF_NODE_G0_OWNER_ONLY",scenario==11 ? "1" : "0",1)==0);
    CHECK(setenv("WT_KF_NODE_SLOT_BYTES",scenario==8 ? "4500" : scenario==9 ? "2000" : "32768",1)==0);
    CHECK(setenv("WT_KF_NODE_SHARED_BUDGET","1048576",1)==0);
    CHECK(setenv("WT_KF_NODE_DEADLINE_MS",scenario==5 ? "1" : scenario==10 ? "500" : "5000",1)==0);
    omp_set_dynamic(0); omp_set_num_threads(2);
}
static void run_rank(int rank,unsigned job)
{
    struct grid g; struct fixture oracle,second;
    struct compute direct={0}; struct wt_kf_node_live_counters counters;
    rank_id=rank; env_rank(rank,job); setup_grid(&g,rank);
    if(scenario==1 && rank==1) init_seed=66;
    if(scenario!=2) initialize_native();
    if(scenario==3 && rank==1) g.tiles[0].scalar[WT_KF_KTAU]=g.tiles[1].scalar[WT_KF_KTAU]=13;
    oracle_copy(&oracle,&g.tiles[0]); oracle_copy(&second,&g.tiles[1]);
    for(unsigned a=0;a<WT_KF_NARG;++a) if(wt_kf_shape[a]!=WT_KF_SCALAR) second.args[a]=oracle.args[a];
#ifdef WT_KF_NODE_GNU_CPS
    (void)direct;
    wt_kf_fixture_gnu_compute(oracle.args); wt_kf_fixture_gnu_compute(second.args);
#else
    CHECK(cps(oracle.args,&direct)==0); CHECK(cps(second.args,&direct)==0);
#endif
    CHECK(wt_kf_bootstrap_start(1,4)==1);
    wt_cumulus_grid_begin(&g);
#pragma omp parallel num_threads(2)
    {
        int thread=omp_get_thread_num();
        void **args=g.tiles[thread].args;
        module_cu_kfeta_mp_kf_eta_cps_(KF_CPS_FROM_ARRAY(args));
    }
    wt_cumulus_grid_end();
    CHECK(unchanged(&g.tiles[0],&oracle));
    wt_kf_node_live_get_counters(&counters);
    wt_kf_bootstrap_dump(stdout); fflush(stdout);
    CHECK(counters.requested==2 && counters.fatal==0);
    if(scenario==2 || scenario==9) {
        CHECK(counters.donated==0 && counters.fallback==2 && atomic_load(&native_calls)==2);
        CHECK(counters.refusal[scenario==2 ? WT_KFN_INIT : WT_KFN_SLOT]==2);
    }
    else CHECK(counters.donated==2 && counters.committed==2);
    if(scenario==0 && rank==1) CHECK(counters.helper_executed>0);
    if(scenario==0 && rank==0) CHECK(counters.owner_remote_committed>0);
    CHECK(counters.committed==counters.owner_remote_committed+counters.self_executed);
    if(scenario==10 && rank==0) {
        CHECK(atomic_load(&peer_seen[2]) && atomic_load(&peer_seen[3]));
        CHECK(counters.self_executed==2 && counters.owner_remote_committed==0);
        for(unsigned j=0;j<4;++j) CHECK(atomic_load(&native_rows[j])==1);
    }
    if(scenario==1 || scenario==3) CHECK(counters.helper_executed==0);
    if(scenario==11) {
        CHECK(counters.helper_executed==0 && counters.owner_remote_committed==0);
        CHECK(counters.self_executed==2 && counters.committed==2 &&
              counters.fallback==0);
    }
    if(scenario==12) {
        CHECK(counters.shadow_checked==counters.owner_remote_committed);
        CHECK(counters.shadow_failed==0);
        CHECK((counters.shadow_checked==0)==(counters.shadow_bytes==0));
        if(rank==0) CHECK(counters.shadow_checked>0);
    }
    wt_kf_node_live_close();
}
static void mapper_cases(unsigned job)
{
    struct wt_kf_node_shm owner,peer,duplicate;
    char private_path[240]; int fd;
    env_rank(1,job);
    CHECK(wt_kf_node_shm_configure(&peer,2,32768,1048576)==1);
    snprintf(private_path,sizeof(private_path),"%s.creating-paused",peer.path);
    fd=open(private_path,O_CREAT|O_EXCL|O_WRONLY,0600); CHECK(fd>=0);
    CHECK(wt_kf_node_shm_attach(&peer)==0); /* Creator private inode is hidden. */
    close(fd);
    env_rank(0,job); CHECK(wt_kf_node_shm_configure(&owner,2,32768,1048576)==1);
    CHECK(wt_kf_node_shm_attach(&owner)==1);
    CHECK(wt_kf_node_shm_attach(&peer)==1);
    env_rank(1,job); CHECK(wt_kf_node_shm_configure(&duplicate,2,32768,1048576)==1);
    CHECK(wt_kf_node_shm_attach(&duplicate)==-1); /* Duplicate rank excluded. */
    wt_kf_node_shm_close(&peer); wt_kf_node_shm_close(&owner);
    CHECK(unlink(private_path)==0);
    CHECK(setenv("SLURM_STEP_TASKS_PER_NODE","2,3",1)==0);
    CHECK(wt_kf_node_shm_configure(&peer,2,32768,1048576)==0);
    env_rank(0,job+1); CHECK(wt_kf_node_shm_configure(&owner,2,32768,1048576)==1);
    fd=open(owner.path,O_CREAT|O_EXCL|O_WRONLY,0600); CHECK(fd>=0); close(fd);
    CHECK(wt_kf_node_shm_attach(&owner)==-1);
    CHECK(unlink(owner.path)==0);
    {
        struct wt_kf_node_shm other,other_peer,wrong;
        env_rank(0,job+2);
        CHECK(wt_kf_node_shm_configure(&owner,2,32768,1048576)==1);
        CHECK(wt_kf_node_shm_configure_phase(&other,UINT64_C(0x5346),2,32768,1048576)==1);
        CHECK(strcmp(owner.path,other.path));
        CHECK(wt_kf_node_shm_attach(&owner)==1 && wt_kf_node_shm_attach(&other)==1);
        env_rank(1,job+2);
        CHECK(wt_kf_node_shm_configure_phase(&wrong,UINT64_C(0x5346),2,32768,1048576)==1);
        /* Even a deliberately redirected filename cannot bypass phase metadata. */
        memcpy(wrong.path,owner.path,sizeof(wrong.path));
        CHECK(wt_kf_node_shm_attach(&wrong)==-1);
        CHECK(wt_kf_node_shm_configure(&peer,2,32768,1048576)==1);
        CHECK(wt_kf_node_shm_configure_phase(&other_peer,UINT64_C(0x5346),2,32768,1048576)==1);
        CHECK(wt_kf_node_shm_attach(&peer)==1 && wt_kf_node_shm_attach(&other_peer)==1);
        wt_kf_node_shm_close(&peer); wt_kf_node_shm_close(&owner);
        wt_kf_node_shm_close(&other_peer); wt_kf_node_shm_close(&other);
        CHECK(wt_kf_node_shm_configure_phase(&wrong,0,2,32768,1048576)==0);
    }
}
int main(int argc,char **argv)
{
    unsigned job=(unsigned)getpid();
    (void)cps;
    if(argc!=2) return 2;
    if(!strcmp(argv[1],"init")) {
        uint64_t first;
        initialize_native(); CHECK(init_calls==1); first=wt_kf_init_fingerprint(); CHECK(first!=0);
#pragma omp parallel num_threads(8)
        { module_cu_kfeta_mp_kf_eta_init_(KF_INIT_FROM_ARRAY(init_expected)); }
        CHECK(init_calls==9 && wt_kf_init_fingerprint()==first);
        init_seed=99; initialize_native(); CHECK(init_calls==10 && wt_kf_init_fingerprint()!=first);
        printf("PASS KF init35 exact forwarding and LUT fingerprint\n"); return 0;
    }
    if(!strcmp(argv[1],"init-guard")) {
        initialize_native(); (void)wt_kf_init_reader_enter(); initialize_native(); return 9;
    }
    if(!strcmp(argv[1],"mapper")) { mapper_cases(job); printf("PASS KF atomic mapper readiness/stale/duplicate/count\n"); return 0; }
    scenario=!strcmp(argv[1],"normal") ? 0 : !strcmp(argv[1],"init-mismatch") ? 1 :
        !strcmp(argv[1],"uninitialized") ? 2 : !strcmp(argv[1],"epoch-mismatch") ? 3 :
        !strcmp(argv[1],"reinit") ? 4 : !strcmp(argv[1],"deadline") ? 5 :
        !strcmp(argv[1],"postnative-guard") ? 6 : !strcmp(argv[1],"postnative-readonly") ? 7 :
        !strcmp(argv[1],"bounded-row") ? 8 : !strcmp(argv[1],"slot-too-small") ? 9 :
        !strcmp(argv[1],"claim-rollback") ? 10 :
        !strcmp(argv[1],"owner-only") ? 11 : 12;
    if(!strcmp(argv[1],"bad-shadow") || !strcmp(argv[1],"bad-owner-only")) {
        scenario=0; env_rank(0,job); initialize_native();
        CHECK(setenv(!strcmp(argv[1],"bad-shadow") ? "WT_KF_NODE_G0_SHADOW" :
                     "WT_KF_NODE_G0_OWNER_ONLY","yes",1)==0);
        CHECK(wt_kf_bootstrap_start(1,4)==0);
        puts("PASS KF node G0 malformed mode rejected"); return 0;
    }
    if(scenario>=4 && scenario<=7) { run_rank(0,job); return 10; }
    {
        char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/coord-XXXXXX";
        int fd=mkstemp(path),start[2],ready[2],status; pid_t child; char byte;
        CHECK(fd>=0); CHECK(ftruncate(fd,4096)==0);
        peer_seen=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); CHECK(peer_seen!=MAP_FAILED);
        for(unsigned i=0;i<5;++i) atomic_init(&peer_seen[i],0);
        CHECK(pipe(start)==0 && pipe(ready)==0);
        child=fork(); CHECK(child>=0);
        if(child==0) {
            close(start[1]); close(ready[0]);
            if(read(start[0],&byte,1)!=1) _exit(21);
            /* Rank0 has configured the production mapper before this signal. */
            env_rank(1,job);
            if(write(ready[1],"r",1)!=1) _exit(22);
            if(scenario==0 || scenario==10 || scenario==12) {
                struct timespec delay={0,1000000}; unsigned tries=0;
                /* Test rendezvous: helper cannot leave its phase before the
                 * owner has published the head's associated tail. */
                while(!atomic_load(&peer_seen[4])) {
                    if(++tries>3000) _exit(23);
                    nanosleep(&delay,NULL);
                }
            }
            run_rank(1,job); _exit(0);
        }
        close(start[0]); close(ready[1]);
        /* Configure only after fork, so both actual processes execute the
         * same production bootstrap independently with different Slurm IDs. */
        env_rank(0,job); CHECK(wt_kf_bootstrap_start(1,4)==1);
        CHECK(write(start[1],"s",1)==1); CHECK(read(ready[0],&byte,1)==1);
        run_rank(0,job);
        CHECK(waitpid(child,&status,0)==child); CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
        if(scenario==0) CHECK(atomic_load(peer_seen)>0);
        munmap(peer_seen,4096); close(fd); close(start[1]); close(ready[0]);
    }
    printf("PASS KF production bootstrap/outer/bridge/node with actual OMP2, %d checks\n",checks);
    return 0;
}
