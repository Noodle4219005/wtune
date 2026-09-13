#define _POSIX_C_SOURCE 200809L
#include "wt_sfclay_bootstrap.h"
#include "wt_sfclay_live.h"
#include "wt_sfclay_node_live.h"
#include "wt_sfclay_scope.h"
#include "wt_sfclay_schema.h"
#include "wt_live_spans.h"

#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK %s:%d: %s\n",\
    __FILE__,__LINE__,#x); exit(82); } } while (0)

enum { IMS=-1, IME=2, JMS=3, JME=6, KMS=1, KME=2, ITS=0, ITE=1 };

struct call {
    void *arg[WT_SFCLAY_NARG];
    size_t cap[WT_SFCLAY_NARG];
    uint32_t scalar[WT_SFCLAY_NARG];
};
struct grid { struct call tile[2]; int id,rank,itimestep; };
struct cfg { int scheme,fractional; };

float test_psimtb[1001],test_psihtb[1001];
static struct grid *active_grid;
static struct cfg *active_cfg;
static _Atomic unsigned *peer_seen;
static int process_rank,scenario;

extern void module_sf_sfclay_mp_sfclayinit_(int *);
extern void wt_sfclay_bridge_call(void **);
extern void wt_sfclay_scope_test_call(void **);

static size_t width(void) { return (size_t)(IME-IMS+1); }
static size_t rows(void) { return (size_t)(JME-JMS+1); }
static size_t levels(void) { return (size_t)(KME-KMS+1); }

static size_t field_bytes(unsigned a)
{
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D)
        return width()*rows()*levels()*4;
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_2D)
        return width()*rows()*4;
    return 4;
}

static void scalar_i(struct call *c,unsigned a,int32_t value)
{ memcpy(&c->scalar[a],&value,4); }

static void call_init(struct call *c,int jts,int jte,float marker)
{
    int32_t bound[18]={-1,2,3,6,1,2,IMS,IME,JMS,JME,KMS,KME,
                       ITS,ITE,jts,jte,2,2};
    memset(c,0,sizeof(*c));
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a) {
        c->cap[a]=field_bytes(a);
        if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR) {
            c->arg[a]=&c->scalar[a];
        } else {
            c->arg[a]=malloc(c->cap[a]); CHECK(c->arg[a]);
            memset(c->arg[a],(int)(0x20+a),c->cap[a]);
        }
    }
    for (unsigned a=6;a<=10;++a) {
        float value=a==6 ? marker : (float)a*.125f;
        memcpy(&c->scalar[a],&value,4);
    }
    scalar_i(c,45,1);
    for (unsigned a=47;a<=57;++a) {
        float value=(float)a*.01f; memcpy(&c->scalar[a],&value,4);
    }
    for (unsigned a=58;a<=75;++a) scalar_i(c,a,bound[a-58]);
    for (unsigned a=81;a<84;++a) scalar_i(c,a,(int32_t)(a&1));
    /* Real call sites may reuse read-only constant scalar actuals. */
    c->arg[48]=c->arg[47];
}

static void grid_init(struct grid *g,int rank)
{
    call_init(&g->tile[0],3,4,rank ? 20.0f : 10.0f);
    call_init(&g->tile[1],5,6,rank ? 20.0f : 10.0f);
    g->id=4; g->rank=rank; g->itimestep=scenario==2 && rank ? 13 : 12;
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a)
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR) {
            free(g->tile[1].arg[a]);
            g->tile[1].arg[a]=g->tile[0].arg[a];
        }
}

static void grid_destroy(struct grid *g)
{
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a)
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR)
            free(g->tile[0].arg[a]);
}

static void grid_copy(struct grid *to,const struct grid *from)
{
    grid_init(to,from->rank); to->id=from->id; to->itimestep=from->itimestep;
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a) {
        if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR) {
            memcpy(&to->tile[0].scalar[a],&from->tile[0].scalar[a],4);
            memcpy(&to->tile[1].scalar[a],&from->tile[1].scalar[a],4);
        } else {
            memcpy(to->tile[0].arg[a],from->tile[0].arg[a],from->tile[0].cap[a]);
        }
    }
}

static size_t active_offset(unsigned a,int j)
{
    size_t row=(size_t)(j-JMS),i=(size_t)(ITS-IMS);
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D)
        return ((row*levels()+(size_t)(1-KMS))*width()+i)*4;
    return (row*width()+i)*4;
}

static void compute(void **argv,int wait_for_peer)
{
    int32_t jts,jte; float marker;
    memcpy(&jts,argv[72],4); memcpy(&jte,argv[73],4);
    memcpy(&marker,argv[6],4);
    if ((scenario==0 || scenario==6) && wait_for_peer && process_rank==0 &&
        marker==10.0f &&
        (jts&1)) {
        for (unsigned spins=0;!atomic_load(peer_seen);++spins) {
            if (spins>2000000) { fputs("WAIT TIMEOUT\n",stderr); abort(); }
            sched_yield();
        }
    }
    for (int j=jts;j<=jte;++j)
        for (unsigned a=0;a<WT_SFCLAY_NARG;++a)
            if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR &&
                wt_sfclay_schema_access[a]==WT_SFCLAY_WRITABLE) {
                unsigned char *p=(unsigned char *)argv[a]+active_offset(a,j);
                for (unsigned i=0;i<(unsigned)(ITE-ITS+1);++i) {
                    uint32_t value=(uint32_t)(marker*1000.0f)^((uint32_t)a<<16)^
                        ((uint32_t)j<<8)^i;
                    memcpy(p+i*4,&value,4);
                }
            }
    if (process_rank==1 && marker==10.0f) atomic_store(peer_seen,1);
    if (scenario==4 && !(jts&1)) ((unsigned char *)argv[12])[-1]=0;
    if (scenario==3 && process_rank==0 && (jts&1)) {
        int allowed=1; module_sf_sfclay_mp_sfclayinit_(&allowed);
    }
}

/* The generated bridge passes 84 address arguments. A variadic fixture keeps
 * this test source compact while consuming the same pointer-only C ABI. */
void module_sf_sfclay_mp_sfclay_(void *a0,...)
{
    void *argv[WT_SFCLAY_NARG]; va_list ap;
    argv[0]=a0; va_start(ap,a0);
    for (unsigned a=1;a<WT_SFCLAY_NARG;++a) argv[a]=va_arg(ap,void *);
    va_end(ap); compute(argv,1);
}

void fixture_sfclay_init_real(int *allowed)
{
    CHECK(allowed && *allowed);
    for (unsigned i=0;i<1001;++i) {
        test_psimtb[i]=(float)i*.001f+(scenario==1 && process_rank ? 1.0f : 0.0f);
        test_psihtb[i]=(float)i*.002f;
    }
}

void wt_grid_identity_(void *opaque,int *id,int *status)
{
    struct grid *g=opaque; *id=g->id; *status=WT_SPAN_OK;
}

void wt_sfclay_grid_epoch_(void *opaque,int *epoch,int *status)
{
    struct grid *g=opaque; *epoch=g->itimestep;
    *status=g->itimestep>0 ? 0 : 1;
}

void wt_grid_capture_(void *opaque,void **context,const size_t *max_nodes,
                      int *tiles,int *status)
{
    struct grid *g=opaque;
    (void)max_nodes; *tiles=2; *status=WT_SPAN_OK;
    for (unsigned a=0;a<WT_SFCLAY_NARG;++a)
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR &&
            wt_live_spans_add(*context,g->tile[0].arg[a],g->tile[0].cap[a],
                WT_SPAN_REAL,4,wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D?3:2)
                    !=WT_SPAN_OK) *status=WT_SPAN_INVALID;
}

void wt_sfclay_native_query_(void *grid,void *cfg,int *grid_id,
    int *actual_scheme,int *actual_fractional,int *expected_scheme,int *status)
{
    struct grid *g=grid; struct cfg *c=cfg;
    *grid_id=g->id; *actual_scheme=c->scheme;
    *actual_fractional=c->fractional; *expected_scheme=91;
    *status=c->scheme==91 && !c->fractional ? WT_SFCLAY_NATIVE_OK :
                                             WT_SFCLAY_NATIVE_SCHEME;
}

static void direct_grid(struct grid *g)
{
    int saved=process_rank; process_rank=-1;
    compute(g->tile[0].arg,0); compute(g->tile[1].arg,0);
    process_rank=saved;
}

static void compare_grid(const struct grid *a,const struct grid *b)
{
    for (unsigned x=0;x<WT_SFCLAY_NARG;++x)
        if (wt_sfclay_schema_shape[x]!=WT_SFCLAY_SHAPE_SCALAR)
            CHECK(!memcmp(a->tile[0].arg[x],b->tile[0].arg[x],a->tile[0].cap[x]));
}

/* Called by the generated first-RK wrapper after it has opened the scope. */
void module_first_rk_step_part1_mp_first_rk_step_part1_(void *grid,...)
{
    CHECK(grid==active_grid && active_cfg);
    wt_sfclay_surface_hook_begin();
#pragma omp parallel num_threads(2)
    {
        unsigned thread=(unsigned)omp_get_thread_num();
        wt_sfclay_bridge_call(active_grid->tile[thread].arg);
    }
    wt_sfclay_surface_hook_end();
}

static void env_rank(int rank,unsigned job)
{
    char text[32];
    snprintf(text,sizeof(text),"%u",job); CHECK(!setenv("SLURM_JOB_ID",text,1));
    CHECK(!setenv("SLURM_STEP_ID","17",1));
    CHECK(!setenv("SLURM_LOCALID",rank ? "1" : "0",1));
    CHECK(!setenv("SLURM_STEP_TASKS_PER_NODE","2",1));
    CHECK(!setenv("WT_SFCLAY_NODE_G0","1",1));
    CHECK(!setenv("WT_SFCLAY_NODE_SLOT_BYTES","65536",1));
    CHECK(!setenv("WT_SFCLAY_NODE_SHARED_BUDGET","1048576",1));
    CHECK(!setenv("WT_SFCLAY_NODE_DEADLINE_MS","5000",1));
    CHECK(!unsetenv("WT_SFCLAY_NODE_G0_SHADOW"));
    CHECK(!unsetenv("WT_SFCLAY_NODE_G0_OWNER_ONLY"));
    if (scenario==6) CHECK(!setenv("WT_SFCLAY_NODE_G0_SHADOW","1",1));
    if (scenario==7) CHECK(!setenv("WT_SFCLAY_NODE_G0_OWNER_ONLY","1",1));
    omp_set_dynamic(0); omp_set_num_threads(2);
}

static void run_rank(int rank,unsigned job)
{
    struct grid g,oracle; struct cfg cfg={91,0};
    struct wt_sfclay_node_live_counters count;
    void *scope_args[84]={0}; int allowed=1;
    process_rank=rank; env_rank(rank,job); grid_init(&g,rank); grid_copy(&oracle,&g);
    active_grid=&g; active_cfg=&cfg; scope_args[0]=&g; scope_args[1]=&cfg;
    if (scenario==5) compute(oracle.tile[0].arg,0);
    else direct_grid(&oracle);
    module_sf_sfclay_mp_sfclayinit_(&allowed);
    CHECK(wt_sfclay_bootstrap_start(1,0)==1);
    if (!scenario && rank==1) {
        /* Compatibility uses actual grid%itimestep, not skewed local tokens. */
        uint64_t skew=wt_sfclay_first_rk_begin(NULL,NULL);
        wt_sfclay_first_rk_end(skew);
    }
    if (scenario==5) {
        wt_sfclay_bridge_call(g.tile[0].arg);
    } else {
        wt_sfclay_scope_test_call(scope_args);
    }
    if (scenario<3 || scenario==5 || scenario==6 || scenario==7)
        compare_grid(&g,&oracle);
    wt_sfclay_node_live_get_counters(&count);
    wt_sfclay_node_live_dump(stdout); fflush(stdout);
    CHECK(count.requested==2 || scenario==5);
    if (scenario==5) CHECK(count.donated==0 && count.fallback==1 &&
                           count.refusal[WT_SFN_SCOPE]==1);
    else if (scenario<3 || scenario==6 || scenario==7) {
        CHECK(count.donated==2 && count.committed==2 && count.fatal==0);
        CHECK(count.committed==count.self_executed+count.owner_remote_committed);
        if (scenario==1 || scenario==2) CHECK(count.helper_executed==0);
    }
    if (scenario==6) {
        CHECK(count.shadow_checked==count.owner_remote_committed);
        CHECK(count.shadow_failed==0);
        CHECK((count.shadow_checked==0)==(count.shadow_bytes==0));
        if (!rank) CHECK(count.shadow_checked>0);
    }
    if (scenario==7)
        CHECK(count.helper_executed==0 && count.owner_remote_committed==0 &&
              count.self_executed==2 && count.committed==2 && !count.fallback);
    wt_sfclay_node_live_close(); grid_destroy(&oracle); grid_destroy(&g);
}

int main(int argc,char **argv)
{
    char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/sfclay/coord-XXXXXX";
    int fd,start[2],ready[2],status; char byte; pid_t child; unsigned job=(unsigned)getpid();
    CHECK(argc==2);
    scenario=!strcmp(argv[1],"normal") ? 0 : !strcmp(argv[1],"init-mismatch") ? 1 :
        !strcmp(argv[1],"epoch-mismatch") ? 2 : !strcmp(argv[1],"reinit") ? 3 :
        !strcmp(argv[1],"postnative-guard") ? 4 :
        !strcmp(argv[1],"outside-scope") ? 5 :
        !strcmp(argv[1],"shadow") ? 6 : 7;
    if (!strcmp(argv[1],"bad-shadow") ||
        !strcmp(argv[1],"bad-owner-only") ||
        !strcmp(argv[1],"bad-combined")) {
        scenario=0; env_rank(0,job);
        if (!strcmp(argv[1],"bad-combined")) {
            CHECK(!setenv("WT_SFCLAY_NODE_G0_SHADOW","1",1));
            CHECK(!setenv("WT_SFCLAY_NODE_G0_OWNER_ONLY","1",1));
        } else CHECK(!setenv(!strcmp(argv[1],"bad-shadow") ?
            "WT_SFCLAY_NODE_G0_SHADOW" : "WT_SFCLAY_NODE_G0_OWNER_ONLY",
            "yes",1));
        CHECK(wt_sfclay_bootstrap_start(1,0)==0);
        puts("PASS SFCLAY node G0 malformed mode rejected"); return 0;
    }
    fd=mkstemp(path); CHECK(fd>=0 && !ftruncate(fd,4096));
    peer_seen=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    CHECK(peer_seen!=MAP_FAILED); atomic_init(peer_seen,0);
    if (scenario==3 || scenario==4) {
        env_rank(0,job); run_rank(0,job); return 9;
    }
    CHECK(!pipe(start) && !pipe(ready)); child=fork(); CHECK(child>=0);
    if (!child) {
        close(start[1]); close(ready[0]); CHECK(read(start[0],&byte,1)==1);
        env_rank(1,job); CHECK(write(ready[1],"r",1)==1);
        run_rank(1,job); _exit(0);
    }
    close(start[0]); close(ready[1]);
    env_rank(0,job);
    {
        struct grid setup; struct cfg cfg={91,0}; int allowed=1;
        process_rank=0; grid_init(&setup,0); active_grid=&setup; active_cfg=&cfg;
        module_sf_sfclay_mp_sfclayinit_(&allowed);
        CHECK(wt_sfclay_bootstrap_start(1,0)==1); grid_destroy(&setup);
    }
    CHECK(write(start[1],"s",1)==1 && read(ready[0],&byte,1)==1);
    run_rank(0,job);
    CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) &&
          WEXITSTATUS(status)==0);
    if (!scenario) CHECK(atomic_load(peer_seen)>0);
    munmap(peer_seen,4096); close(fd); unlink(path);
    puts("SFCLAY production node-live fork/OMP2 chain passed");
    return 0;
}
