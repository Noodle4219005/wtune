#include "wt_kf_batch.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUDGET (16u*1024u*1024u)
#ifndef STORAGE
#define STORAGE 600
#endif
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1); } ++checks; } while(0)
static int checks;
struct fixture {
    void *args[WT_KF_NARG];
    size_t cap[WT_KF_NARG];
    int32_t scalar[WT_KF_NARG];
    uint32_t dummy[5];
    float arrays[WT_KF_NARG][STORAGE/4];
};
struct compute { int calls,mode; unsigned char alignment[WT_KF_NARG]; int check_alignment; };
static void setup(struct fixture *f)
{
    unsigned a,k;
    memset(f,0,sizeof(*f));
    for(a=0;a<WT_KF_NARG;++a) {
        f->args[a]=wt_kf_shape[a]==WT_KF_SCALAR ? (void *)&f->scalar[a] : (void *)f->arrays[a];
        f->cap[a]=wt_kf_shape[a]==WT_KF_SCALAR ? 4 : STORAGE;
        for(k=0;k<STORAGE/4;++k) f->arrays[a][k]=(float)(a*2+k+1)/8;
    }
    f->scalar[WT_KF_IDS]=-1; f->scalar[WT_KF_IDE]=3;
    f->scalar[WT_KF_JDS]=0; f->scalar[WT_KF_JDE]=4;
    f->scalar[WT_KF_KDS]=1; f->scalar[WT_KF_KDE]=4;
    f->scalar[WT_KF_IMS]=-2; f->scalar[WT_KF_IME]=3;
    f->scalar[WT_KF_JMS]=-1; f->scalar[WT_KF_JME]=3;
    f->scalar[WT_KF_KMS]=0; f->scalar[WT_KF_KME]=4;
    f->scalar[WT_KF_ITS]=-1; f->scalar[WT_KF_ITE]=2;
    f->scalar[WT_KF_JTS]=0; f->scalar[WT_KF_JTE]=1;
    f->scalar[WT_KF_KTS]=1; f->scalar[WT_KF_KTE]=3;
    f->scalar[WT_KF_TRIGGER]=1;
    for(a=0;a<5;++a) {
        f->dummy[a]=0x12340+a;
        f->args[WT_KF_UDR_KF+a]=&f->dummy[a]; f->cap[WT_KF_UDR_KF+a]=4;
    }
}
static void oracle_copy(struct fixture *to,const struct fixture *from)
{
    setup(to); memcpy(to->scalar,from->scalar,sizeof(to->scalar));
    memcpy(to->dummy,from->dummy,sizeof(to->dummy));
    memcpy(to->arrays,from->arrays,sizeof(to->arrays));
    if(!from->args[WT_KF_SHALL]) to->args[WT_KF_SHALL]=NULL;
    if(from->scalar[WT_KF_KF_EDRATES]==1) {
        unsigned a; for(a=WT_KF_UDR_KF;a<=WT_KF_TIMEC_KF;++a) {
            to->args[a]=to->arrays[a]; to->cap[a]=STORAGE;
        }
    }
}
static int unchanged(const struct fixture *a,const struct fixture *b)
{
    return !memcmp(a->scalar,b->scalar,sizeof(a->scalar)) &&
           !memcmp(a->dummy,b->dummy,sizeof(a->dummy)) &&
           !memcmp(a->arrays,b->arrays,sizeof(a->arrays));
}
/* CPS-shaped callback: uses actual Fortran XZY offsets, the original full-i
 * vector loop, and W(KTE+1). This is transport evidence, not native KF evidence. */
static int cps(void **args,void *context)
{
    struct compute *c=context;
    int32_t b[18];
    int i,j,k; unsigned a;
    size_t ni,nk;
    ++c->calls;
    for(a=0;a<18;++a) memcpy(&b[a],args[a],4);
    ni=(size_t)(b[7]-b[6]+1); nk=(size_t)(b[11]-b[10]+1);
    if(c->check_alignment) for(a=0;a<WT_KF_NARG;++a)
        if(args[a] && ((uintptr_t)args[a]&63)!=c->alignment[a]) return 10;
    if(c->mode==1) return 7;
    if(c->mode==2) { args[WT_KF_RHO]=NULL; return 0; }
    if(c->mode==3) { ((unsigned char *)args[WT_KF_W])[-1]=0; return 0; }
    if(c->mode==4) { ((float *)args[WT_KF_RHO])[0]=999; return 0; }
    if(c->mode==5) { *(float *)args[WT_KF_TH]=999; return 0; }
    if(c->mode==6) { *(float *)args[WT_KF_RQVFTEN]=999; return 0; }
    if(c->mode==7) { *(int32_t *)args[WT_KF_F_QV]=99; return 0; }
    if(c->mode==8) { *(uint32_t *)args[WT_KF_UDR_KF]=99; return 0; }
    for(a=0;a<WT_KF_NARG;++a) {
        int inactive=*(int32_t *)args[WT_KF_KF_EDRATES]!=1 && a>=WT_KF_UDR_KF && a<=WT_KF_TIMEC_KF;
        if(!args[a] || wt_kf_shape[a]==WT_KF_SCALAR || wt_kf_intent[a]==WT_KF_IN || inactive || a==WT_KF_RQVFTEN) continue;
        for(j=b[14];j<=b[15];++j) for(k=b[16];k<=(wt_kf_shape[a]==WT_KF_3D ? b[17] : b[16]);++k)
            for(i=b[12];i<=b[13];++i) {
                size_t off=(size_t)(i-b[6])+ni*(size_t)(j-b[8]);
                if(wt_kf_shape[a]==WT_KF_3D) off=(size_t)(i-b[6])+ni*((size_t)(k-b[10])+nk*(size_t)(j-b[8]));
                if(a==WT_KF_W0AVG)
                    ((float *)args[a])[off]=(((float *)args[a])[off]+((float *)args[WT_KF_W])[off]+((float *)args[WT_KF_W])[off+ni])*0.5f;
                else ((float *)args[a])[off]+=(float)(a+1);
            }
    }
    return 0;
}
#ifndef KFB_FIXTURE_ONLY
static void equivalence(int active,int absent)
{
    struct fixture f,expected,before;
    struct wt_kf_batch *b=NULL;
    struct compute c={0},direct={0};
    const void *request; void *result; size_t bytes; int invoked; unsigned a;
    setup(&f);
    if(active) { f.scalar[WT_KF_KF_EDRATES]=1; for(a=WT_KF_UDR_KF;a<=WT_KF_TIMEC_KF;++a) { f.args[a]=f.arrays[a]; f.cap[a]=STORAGE; } }
    if(absent) f.args[WT_KF_SHALL]=NULL;
    oracle_copy(&expected,&f); before=f;
    CHECK(cps(expected.args,&direct)==0);
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OK);
    request=wt_kf_batch_request(b,&bytes); CHECK(request && bytes<sizeof(f.arrays));
    result=malloc(bytes); CHECK(result);
    c.check_alignment=1; for(a=0;a<WT_KF_NARG;++a) c.alignment[a]=(unsigned char)((uintptr_t)f.args[a]&63);
    CHECK(wt_kf_batch_execute(request,bytes,result,bytes,BUDGET,cps,&c,&invoked)==WT_KF_OK);
    CHECK(invoked==1 && c.calls==1 && unchanged(&f,&before));
    CHECK(wt_kf_batch_commit(b,result,bytes)==WT_KF_OK);
    CHECK(unchanged(&f,&expected));
    CHECK(wt_kf_batch_commit(b,result,bytes)==WT_KF_BAD_STATE);
    free(result); wt_kf_batch_destroy(b);
}
static void failures(void)
{
    struct fixture f,before;
    struct wt_kf_batch *b=NULL;
    size_t bytes; const void *q; void *r; int invoked,mode;
    for(mode=1;mode<=8;++mode) {
        struct compute c={0}; c.mode=mode; setup(&f); before=f;
        CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OK);
        q=wt_kf_batch_request(b,&bytes); r=malloc(bytes); CHECK(r);
        CHECK(wt_kf_batch_execute(q,bytes,r,bytes,BUDGET,cps,&c,&invoked)!=WT_KF_OK);
        CHECK(invoked==1 && c.calls==1 && unchanged(&f,&before));
        CHECK(wt_kf_batch_commit(b,r,0)!=WT_KF_OK);
        CHECK(wt_kf_batch_commit(b,r,bytes)==WT_KF_BAD_STATE);
        free(r); wt_kf_batch_destroy(b);
    }
    setup(&f); f.cap[WT_KF_W]=4;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_CAPACITY && !b);
    setup(&f); f.args[WT_KF_U]=f.args[WT_KF_V];
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OVERLAP);
    setup(&f); f.args[WT_KF_F_QS]=NULL;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); CHECK(wt_kf_batch_prepare(f.args,f.cap,1,&b)==WT_KF_BUDGET);
    setup(&f); f.scalar[WT_KF_TRIGGER]=2;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); f.scalar[WT_KF_ADAPT_STEP_FLAG]=1;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); f.scalar[WT_KF_KTS]=0;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); f.scalar[WT_KF_KME]=3;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); f.scalar[WT_KF_IMS]=INT32_MIN; f.scalar[WT_KF_IME]=INT32_MAX;
    f.scalar[WT_KF_JMS]=INT32_MIN; f.scalar[WT_KF_JME]=INT32_MAX;
    CHECK(wt_kf_batch_prepare(f.args,f.cap,SIZE_MAX,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); f.args[WT_KF_U]=(void *)(UINTPTR_MAX-3);
    CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_BAD_ARGUMENT);
    setup(&f); CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OK);
    q=wt_kf_batch_request(b,&bytes); r=malloc(bytes); CHECK(r);
    { struct compute c={0};
      CHECK(wt_kf_batch_execute(q,bytes,r,bytes,1,cps,&c,&invoked)==WT_KF_BUDGET);
      CHECK(invoked==0 && c.calls==0);
      CHECK(wt_kf_batch_execute(q,bytes,r,bytes-1,BUDGET,cps,&c,&invoked)==WT_KF_CAPACITY);
      CHECK(wt_kf_batch_execute(q,bytes,(void *)q,bytes,BUDGET,cps,&c,&invoked)==WT_KF_OVERLAP);
      CHECK(wt_kf_batch_execute(q,bytes,r,bytes,BUDGET,cps,&c,&invoked)==WT_KF_OK);
    }
    f.scalar[WT_KF_KF_EDRATES]=1;
    before=f;
    CHECK(wt_kf_batch_commit(b,r,bytes)==WT_KF_STALE);
    CHECK(unchanged(&f,&before));
    f.scalar[WT_KF_KF_EDRATES]=0;
    CHECK(wt_kf_batch_commit(b,r,bytes)==WT_KF_BAD_STATE);
    free(r); wt_kf_batch_destroy(b);
}
static void unused_and_stale_closure(void)
{
    struct fixture f;
    struct wt_kf_batch *b;
    struct compute c={0};
    const void *q; void *r; size_t bytes; int invoked;
    setup(&f); CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OK);
    q=wt_kf_batch_request(b,&bytes); r=malloc(bytes); CHECK(r);
    CHECK(wt_kf_batch_execute(q,bytes,r,bytes,BUDGET,cps,&c,&invoked)==WT_KF_OK);
    /* Unused TH/RQVFTEN owner backing may change independently: no read or
     * write closure includes it, and commit must preserve the new bytes. */
    f.arrays[WT_KF_TH][0]=987; f.arrays[WT_KF_RQVFTEN][0]=654;
    CHECK(wt_kf_batch_commit(b,r,bytes)==WT_KF_OK);
    CHECK(f.arrays[WT_KF_TH][0]==987 && f.arrays[WT_KF_RQVFTEN][0]==654);
    free(r); wt_kf_batch_destroy(b);
    setup(&f); CHECK(wt_kf_batch_prepare(f.args,f.cap,BUDGET,&b)==WT_KF_OK);
    q=wt_kf_batch_request(b,&bytes); r=malloc(bytes); CHECK(r);
    CHECK(wt_kf_batch_execute(q,bytes,r,bytes,BUDGET,cps,&c,&invoked)==WT_KF_OK);
    /* First tile column at KTE+1: proves the extra W read is snapshotted. */
    f.arrays[WT_KF_W][1+6*(4+5)] += 1;
    CHECK(wt_kf_batch_commit(b,r,bytes)==WT_KF_STALE);
    free(r); wt_kf_batch_destroy(b);
}
int main(void)
{
    equivalence(0,0); equivalence(0,1); equivalence(1,0); failures(); unused_and_stale_closure();
    printf("PASS KF batch %d checks\n",checks); return 0;
}
#endif
