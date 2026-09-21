#define _POSIX_C_SOURCE 200809L
#define KFB_FIXTURE_ONLY
#define STORAGE (6*5*34*4)
#include "test_kf_batch.c"
#include "wt_kf_node.h"
#include <math.h>
#include <sys/wait.h>
#include <unistd.h>
extern void __module_cu_kfeta_MOD_kf_lutab(float *,float *,float *,float *);
extern int kf_diag_messages,kf_diag_fatals;
extern void __module_cu_kfeta_MOD_kf_eta_cps(void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *);
static int native(void **a,void *ctx) { (void)ctx; __module_cu_kfeta_MOD_kf_eta_cps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19],a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[27],a[28],a[29],a[30],a[31],a[32],a[33],a[34],a[35],a[36],a[37],a[38],a[39],a[40],a[41],a[42],a[43],a[44],a[45],a[46],a[47],a[48],a[49],a[50],a[51],a[52],a[53],a[54],a[55],a[56],a[57],a[58],a[59],a[60],a[61],a[62],a[63],a[64],a[65],a[66],a[67],a[68],a[69],a[70],a[71],a[72],a[73],a[74],a[75],a[76],a[77],a[78]); return 0; }

static void real_scalar(struct fixture *f,unsigned a,float x) { memcpy(&f->scalar[a],&x,4); }
static void sounding(struct fixture *f,int active)
{
    unsigned a; int j,k,i;
    setup(f); f->scalar[WT_KF_KME]=33; f->scalar[WT_KF_KDE]=33; f->scalar[WT_KF_KTE]=32;
    f->scalar[WT_KF_STEPCU]=1; f->scalar[WT_KF_KTAU]=1;
    f->scalar[WT_KF_KF_EDRATES]=active;
    for(a=WT_KF_F_QV;a<=WT_KF_F_QS;++a) f->scalar[a]=1;
    if(active) for(a=WT_KF_UDR_KF;a<=WT_KF_TIMEC_KF;++a) { f->args[a]=f->arrays[a]; f->cap[a]=STORAGE; }
    real_scalar(f,WT_KF_DT,30); real_scalar(f,WT_KF_DX,12000); real_scalar(f,WT_KF_CUDT,0.5f);
    real_scalar(f,WT_KF_XLV0,3.15e6f); real_scalar(f,WT_KF_XLV1,2369);
    real_scalar(f,WT_KF_XLS0,2.905e6f); real_scalar(f,WT_KF_XLS1,259);
    real_scalar(f,WT_KF_CP,1004); real_scalar(f,WT_KF_R,287); real_scalar(f,WT_KF_G,9.81f);
    real_scalar(f,WT_KF_EP1,0.608f); real_scalar(f,WT_KF_EP2,0.622f);
    real_scalar(f,WT_KF_SVP1,0.6112f); real_scalar(f,WT_KF_SVP2,17.67f);
    real_scalar(f,WT_KF_SVP3,29.65f); real_scalar(f,WT_KF_SVPT0,273.15f);
    for(j=0;j<5;++j) for(k=0;k<34;++k) for(i=0;i<6;++i) {
        size_t q=(size_t)i+6u*((size_t)k+34u*(size_t)j);
        float t=300-3.0f*k,p=100000*expf(-k*500.0f/8000),pi=powf(p/100000,0.2854f);
        f->arrays[WT_KF_T][q]=t; f->arrays[WT_KF_PCPS][q]=p; f->arrays[WT_KF_PI][q]=pi;
        f->arrays[WT_KF_TH][q]=t/pi; f->arrays[WT_KF_RHO][q]=p/(287*t);
        f->arrays[WT_KF_QV][q]=0.008f*expf(-k/7.0f);
        f->arrays[WT_KF_U][q]=5; f->arrays[WT_KF_V][q]=2;
        f->arrays[WT_KF_W][q]=0.2f; f->arrays[WT_KF_W0AVG][q]=0.1f;
        f->arrays[WT_KF_DZ8W][q]=500;
    }
    for(i=0;i<30;++i) f->arrays[WT_KF_NCA][i]=-100;
}
int main(void)
{
    int active; float svp1=.6112f,svp2=17.67f,svp3=29.65f,svpt0=273.15f;
    (void)cps;
    __module_cu_kfeta_MOD_kf_lutab(&svp1,&svp2,&svp3,&svpt0);
    for(active=0;active<2;++active) {
        struct fixture f,expected,before;
        struct wt_nl_service service; struct wt_nl_config cfg={2,2,65536};
        struct wt_kf_node_call call={0}; struct wt_nl_ticket ticket; enum wt_kf_status bs;
        struct wt_nl_signature sig={WT_KF_NODE_PHASE,19,WT_KF_BATCH_SCHEMA,4,7,1,23};
        struct wt_kf_node_context owner={&service,{0},0,0,BUDGET,native,NULL};
        struct wt_nl_counters counters;
        char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/native-service-XXXXXX";
        int fd=mkstemp(path),status; pid_t pid; size_t mapbytes;
        CHECK(fd>=0); CHECK(wt_nl_init_fd(fd,&cfg,&service)==WT_NL_OK); mapbytes=service.mapping_bytes;
        sounding(&f,active); oracle_copy(&expected,&f); before=f;
        CHECK(native(expected.args,NULL)==0);
        CHECK(!unchanged(&before,&expected));
        owner.signature=sig;
        CHECK(wt_kf_node_publish(&owner,f.args,f.cap,1,&call,&bs)==WT_NL_OK);
        pid=fork(); CHECK(pid>=0);
        if(pid==0) {
            struct wt_nl_service attached; struct wt_kf_node_context helper=owner;
            wt_nl_detach(&service);
            if(wt_nl_attach_fd(fd,mapbytes,&attached)!=WT_NL_OK) _exit(21);
            helper.service=&attached; helper.rank=1;
            if(wt_kf_node_progress(&helper,(uint64_t)getpid(),&ticket)!=WT_NL_OK) _exit(22);
            if(kf_diag_fatals) _exit(23);
            wt_nl_detach(&attached); close(fd); _exit(0);
        }
        CHECK(waitpid(pid,&status,0)==pid); CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
        CHECK(unchanged(&f,&before)); CHECK(wt_kf_node_collect(&owner,&call)==WT_NL_OK);
        CHECK(unchanged(&f,&expected)); CHECK(wt_kf_node_finish(&owner,&call)==WT_NL_OK);
        CHECK(wt_nl_get_counters(&service,&counters)==WT_NL_OK);
        CHECK(counters.helper_executed==1 && counters.committed==1 && counters.failed==0);
        wt_kf_node_destroy(&call); wt_nl_detach(&service); close(fd);
    }
    CHECK(kf_diag_fatals==0);
    printf("PASS frozen GNU CPS full-state bitwise, EDRATES 0/1, other-process helper, %d checks, diagnostics %d/%d\n",checks,kf_diag_messages,kf_diag_fatals);
    return 0;
}
