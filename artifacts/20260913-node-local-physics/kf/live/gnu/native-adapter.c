#define KFB_FIXTURE_ONLY
#include "test_kf_batch.c"
#include <math.h>
extern void __module_cu_kfeta_MOD_kf_eta_init(void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *);
extern void __module_cu_kfeta_MOD_kf_eta_cps(void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *,void *);
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
void wt_kf_fixture_gnu_compute(void **a) { __module_cu_kfeta_MOD_kf_eta_cps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19],a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[27],a[28],a[29],a[30],a[31],a[32],a[33],a[34],a[35],a[36],a[37],a[38],a[39],a[40],a[41],a[42],a[43],a[44],a[45],a[46],a[47],a[48],a[49],a[50],a[51],a[52],a[53],a[54],a[55],a[56],a[57],a[58],a[59],a[60],a[61],a[62],a[63],a[64],a[65],a[66],a[67],a[68],a[69],a[70],a[71],a[72],a[73],a[74],a[75],a[76],a[77],a[78]); }void wt_kf_fixture_gnu_init(void) { struct fixture f; int one=1,zero=0; sounding(&f,1); __module_cu_kfeta_MOD_kf_eta_init(f.args[WT_KF_RTHCUTEN],f.args[WT_KF_RQVCUTEN],f.args[WT_KF_RQCCUTEN],f.args[WT_KF_RQRCUTEN],f.args[WT_KF_RQICUTEN],f.args[WT_KF_RQSCUTEN],f.args[WT_KF_NCA],f.args[WT_KF_W0AVG],&one,&one,f.args[WT_KF_SVP1],f.args[WT_KF_SVP2],f.args[WT_KF_SVP3],f.args[WT_KF_SVPT0],&one,&zero,&one,&f.scalar[0],&f.scalar[1],&f.scalar[2],&f.scalar[3],&f.scalar[4],&f.scalar[5],&f.scalar[6],&f.scalar[7],&f.scalar[8],&f.scalar[9],&f.scalar[10],&f.scalar[11],&f.scalar[12],&f.scalar[13],&f.scalar[14],&f.scalar[15],&f.scalar[16],&f.scalar[17]); }void wt_kf_fixture_gnu_sounding(void *f,int active) { sounding(f,active); (void)cps; (void)oracle_copy; (void)unchanged; (void)checks; }