#define _POSIX_C_SOURCE 200809L
#include "wt_sfclay_batch.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"check failed %s:%d: %s\n",\
    __FILE__,__LINE__,#x); exit(2); } } while (0)

enum { IMS=-1, IME=2, JMS=3, JME=4, KMS=1, KME=2, ITS=0, ITE=1 };

struct fixture {
    void *arg[WT_SFCLAY_NARG];
    size_t cap[WT_SFCLAY_NARG];
    unsigned char *initial[WT_SFCLAY_NARG];
};

struct callback_context { uint32_t marker; int mode, calls; };

static size_t ni(void) { return (size_t)(IME-IMS+1); }
static size_t nj(void) { return (size_t)(JME-JMS+1); }
static size_t nk(void) { return (size_t)(KME-KMS+1); }
static size_t active(void) { return (size_t)(ITE-ITS+1); }

static size_t bytes_for(unsigned a)
{
    return wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D ? ni()*nj()*nk()*4 :
        wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_2D ? ni()*nj()*4 : 4;
}

static void put_i(struct fixture *f, unsigned a, int32_t value)
{ memcpy(f->arg[a],&value,4); }

static void init_fixture(struct fixture *f, int row, int controls)
{
    unsigned a;
    int32_t bounds[18]={-1,2,3,4,1,2,IMS,IME,JMS,JME,KMS,KME,
                         ITS,ITE,row,row,2,2};
    memset(f,0,sizeof(*f));
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        if (!controls && a>=81) continue;
        f->cap[a]=bytes_for(a);
        f->arg[a]=malloc(f->cap[a]); f->initial[a]=malloc(f->cap[a]);
        CHECK(f->arg[a] && f->initial[a]);
        memset(f->arg[a],(int)(0x20+a),f->cap[a]);
    }
    /* Correct common scalar types are irrelevant to transport except bounds;
     * provide finite float bit patterns and valid integer controls. */
    for (a=6;a<=10;++a) { float x=(float)(a+1); memcpy(f->arg[a],&x,4); }
    put_i(f,45,1);
    for (a=47;a<=57;++a) { float x=(float)a*.01f; memcpy(f->arg[a],&x,4); }
    for (a=58;a<=75;++a) put_i(f,a,bounds[a-58]);
    for (a=81;a<84;++a) if (f->arg[a]) put_i(f,a,(int32_t)(a&1));
    for (a=0;a<WT_SFCLAY_NARG;++a) if (f->arg[a])
        memcpy(f->initial[a],f->arg[a],f->cap[a]);
}

static void destroy_fixture(struct fixture *f)
{
    unsigned a;
    for (a=0;a<WT_SFCLAY_NARG;++a) { free(f->arg[a]); free(f->initial[a]); }
}

static size_t active_offset(void **argv, unsigned a)
{
    int32_t ims,ime,jms,kms,its,jts;
    size_t width,levels,j,i,k=0;
    memcpy(&ims,argv[64],4); memcpy(&ime,argv[65],4);
    memcpy(&jms,argv[66],4); memcpy(&kms,argv[68],4);
    memcpy(&its,argv[70],4); memcpy(&jts,argv[72],4);
    width=(size_t)((int64_t)ime-ims+1);
    j=(size_t)((int64_t)jts-jms); i=(size_t)((int64_t)its-ims);
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D) {
        int32_t kme; memcpy(&kme,argv[69],4);
        levels=(size_t)((int64_t)kme-kms+1); k=(size_t)(1-kms);
        return ((j*levels+k)*width+i)*4;
    }
    return (j*width+i)*4;
}

static size_t full_bytes(void **argv, unsigned a)
{
    int32_t ims,ime,jms,jme,kms,kme;
    size_t n;
    memcpy(&ims,argv[64],4); memcpy(&ime,argv[65],4);
    memcpy(&jms,argv[66],4); memcpy(&jme,argv[67],4);
    memcpy(&kms,argv[68],4); memcpy(&kme,argv[69],4);
    n=(size_t)((int64_t)ime-ims+1)*(size_t)((int64_t)jme-jms+1)*4;
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D)
        n*= (size_t)((int64_t)kme-kms+1);
    return n;
}

static int compute(void *argv[WT_SFCLAY_NARG], void *opaque)
{
    struct callback_context *c=opaque;
    unsigned a,i;
    ++c->calls;
    if (c->mode==3) return 1;
    if (c->mode==1) ((unsigned char *)argv[0])[active_offset(argv,0)]^=1;
    if (c->mode==2) {
        unsigned a0=12;
        ((unsigned char *)argv[a0])[full_bytes(argv,a0)]=1;
        return 0;
    }
    for (a=0;a<WT_SFCLAY_NARG;++a)
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR &&
            wt_sfclay_schema_access[a]==WT_SFCLAY_WRITABLE) {
            unsigned char *p=(unsigned char *)argv[a]+active_offset(argv,a);
            for (i=0;i<active();++i) {
                uint32_t value=c->marker ^ ((uint32_t)a<<16) ^ i;
                memcpy(p+i*4,&value,4);
            }
        }
    return 0;
}

static void check_committed(const struct fixture *f, int row, uint32_t marker)
{
    unsigned a,j;
    size_t row0=(size_t)(row-JMS)*ni();
    for (a=0;a<WT_SFCLAY_NARG;++a) if (f->arg[a] &&
        wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR) {
        size_t plane=wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D ? nk()*ni() : ni();
        size_t k=wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D ? (size_t)(1-KMS)*ni() : 0;
        for (j=0;j<f->cap[a]/4;++j) {
            int touched=j>=row0*(plane/ni())+k+(size_t)(ITS-IMS) &&
                j<row0*(plane/ni())+k+(size_t)(ITS-IMS)+active();
            if (touched && wt_sfclay_schema_access[a]==WT_SFCLAY_WRITABLE) {
                uint32_t got,want=marker^((uint32_t)a<<16)^
                    (uint32_t)(j-(row0*(plane/ni())+k+(size_t)(ITS-IMS)));
                memcpy(&got,(unsigned char *)f->arg[a]+j*4,4); CHECK(got==want);
            } else CHECK(!memcmp((unsigned char *)f->arg[a]+j*4,
                                  f->initial[a]+j*4,4));
        }
    }
}

static void execute_batch(struct wt_sfclay_batch *b,
                          struct callback_context *callback,
                          unsigned char **result, size_t *bytes)
{
    const void *request=wt_sfclay_batch_request(b,bytes);
    int invoked=-1;
    CHECK(request); *result=malloc(*bytes); CHECK(*result);
    CHECK(wt_sfclay_batch_execute(request,*bytes,*result,*bytes,1u<<20,
          compute,callback,&invoked)==WT_SFCLAY_BATCH_OK);
    CHECK(invoked==1 && callback->calls==1);
}

struct execute_job {
    const void *request; size_t bytes;
    unsigned char *result;
    struct callback_context *callback;
    enum wt_sfclay_batch_status status;
    int invoked;
};

static void *execute_thread(void *opaque)
{
    struct execute_job *j=opaque;
    j->status=wt_sfclay_batch_execute(j->request,j->bytes,j->result,j->bytes,
        1u<<20,compute,j->callback,&j->invoked);
    return NULL;
}

static void test_one_row(void)
{
    struct fixture f; struct wt_sfclay_batch *b=NULL;
    struct callback_context callback={0x12345678u,0,0};
    unsigned char *result; size_t bytes;
    init_fixture(&f,3,0);
    CHECK(wt_sfclay_batch_prepare(f.arg,f.cap,1u<<20,&b)==WT_SFCLAY_BATCH_OK);
    execute_batch(b,&callback,&result,&bytes);
    CHECK(!memcmp(f.arg[12],f.initial[12],f.cap[12]));
    CHECK(wt_sfclay_batch_commit(b,result,bytes)==WT_SFCLAY_BATCH_OK);
    check_committed(&f,3,callback.marker);
    CHECK(wt_sfclay_batch_apply(b)==WT_SFCLAY_BATCH_STATE);
    free(result); wt_sfclay_batch_destroy(b); destroy_fixture(&f);
}

static void test_disjoint_rows(void)
{
    struct fixture f,g; struct wt_sfclay_batch *a=NULL,*b=NULL;
    struct callback_context ca={0x11112222u,0,0},cb={0x33334444u,0,0};
    struct execute_job jobs[2]; pthread_t threads[2];
    unsigned char *ra,*rb; size_t na,nb;
    unsigned x;
    init_fixture(&f,3,1); init_fixture(&g,4,1);
    /* Real concurrent calls have private scalar actuals but share field arrays. */
    for (x=0;x<WT_SFCLAY_NARG;++x)
        if (wt_sfclay_schema_shape[x]!=WT_SFCLAY_SHAPE_SCALAR) {
            free(g.arg[x]); free(g.initial[x]);
            g.arg[x]=f.arg[x]; g.initial[x]=NULL; g.cap[x]=f.cap[x];
        }
    CHECK(wt_sfclay_batch_prepare(f.arg,f.cap,1u<<20,&a)==WT_SFCLAY_BATCH_OK);
    CHECK(wt_sfclay_batch_prepare(g.arg,g.cap,1u<<20,&b)==WT_SFCLAY_BATCH_OK);
    jobs[0].request=wt_sfclay_batch_request(a,&na); jobs[0].bytes=na;
    jobs[1].request=wt_sfclay_batch_request(b,&nb); jobs[1].bytes=nb;
    ra=malloc(na); rb=malloc(nb); CHECK(ra && rb);
    jobs[0].result=ra; jobs[0].callback=&ca; jobs[0].invoked=0;
    jobs[1].result=rb; jobs[1].callback=&cb; jobs[1].invoked=0;
    CHECK(pthread_create(&threads[0],NULL,execute_thread,&jobs[0])==0);
    CHECK(pthread_create(&threads[1],NULL,execute_thread,&jobs[1])==0);
    CHECK(pthread_join(threads[0],NULL)==0 && pthread_join(threads[1],NULL)==0);
    CHECK(jobs[0].status==WT_SFCLAY_BATCH_OK && jobs[0].invoked==1 && ca.calls==1);
    CHECK(jobs[1].status==WT_SFCLAY_BATCH_OK && jobs[1].invoked==1 && cb.calls==1);
    CHECK(wt_sfclay_batch_commit(a,ra,na)==WT_SFCLAY_BATCH_OK);
    check_committed(&f,3,ca.marker);
    { for (x=0;x<WT_SFCLAY_NARG;++x) if (f.arg[x])
        memcpy(f.initial[x],f.arg[x],f.cap[x]); }
    /* Applying row 3 must not make row 4's snapshot stale. */
    CHECK(wt_sfclay_batch_commit(b,rb,nb)==WT_SFCLAY_BATCH_OK);
    check_committed(&f,4,cb.marker);
    free(ra); free(rb); wt_sfclay_batch_destroy(a); wt_sfclay_batch_destroy(b);
    for (x=0;x<WT_SFCLAY_NARG;++x)
        if (wt_sfclay_schema_shape[x]!=WT_SFCLAY_SHAPE_SCALAR) g.arg[x]=NULL;
    destroy_fixture(&g);
    destroy_fixture(&f);
}

static void test_failures(void)
{
    struct fixture f; struct wt_sfclay_batch *b=NULL;
    struct callback_context c={7,0,0}; const void *request;
    unsigned char *result; size_t bytes; int invoked;
    init_fixture(&f,3,1);
    put_i(&f,73,4);
    CHECK(wt_sfclay_batch_prepare(f.arg,f.cap,1u<<20,&b)==WT_SFCLAY_BATCH_BOUNDS);
    put_i(&f,73,3);
    CHECK(wt_sfclay_batch_prepare(f.arg,f.cap,1u<<20,&b)==WT_SFCLAY_BATCH_OK);
    request=wt_sfclay_batch_request(b,&bytes); result=malloc(bytes); CHECK(result);
    invoked=-1;
    CHECK(wt_sfclay_batch_execute(request,bytes,result,bytes-1,1u<<20,
          compute,&c,&invoked)==WT_SFCLAY_BATCH_CAPACITY && invoked==0);
    c.mode=2;
    CHECK(wt_sfclay_batch_execute(request,bytes,result,bytes,1u<<20,
          compute,&c,&invoked)==WT_SFCLAY_BATCH_GUARD && invoked==1);
    c.mode=1;
    CHECK(wt_sfclay_batch_execute(request,bytes,result,bytes,1u<<20,
          compute,&c,&invoked)==WT_SFCLAY_BATCH_STALE && invoked==1);
    wt_sfclay_batch_destroy(b); free(result); destroy_fixture(&f);

    init_fixture(&f,3,1); b=NULL; c.mode=0; c.calls=0;
    CHECK(wt_sfclay_batch_prepare(f.arg,f.cap,1u<<20,&b)==WT_SFCLAY_BATCH_OK);
    execute_batch(b,&c,&result,&bytes);
    ((unsigned char *)f.arg[0])[(size_t)(ITS-IMS)*4]^=1;
    CHECK(wt_sfclay_batch_validate(b,result,bytes)==WT_SFCLAY_BATCH_STALE);
    CHECK(!memcmp(f.arg[12],f.initial[12],f.cap[12]));
    free(result); wt_sfclay_batch_destroy(b); destroy_fixture(&f);
}

int main(void)
{
    test_one_row(); test_disjoint_rows(); test_failures();
    puts("SFCLAY row batch tests passed"); return 0;
}
