#include "wt_kf_batch_internal.h"
#include <stdlib.h>
#include <string.h>

int kfb_add(size_t a, size_t b, size_t *r)
{ if (b > SIZE_MAX-a) return 0; *r=a+b; return 1; }
int kfb_mul(size_t a, size_t b, size_t *r)
{ if (a && b > SIZE_MAX/a) return 0; *r=a*b; return 1; }
int kfb_interval(const void *p, size_t n, uintptr_t *end)
{ if (n > UINTPTR_MAX-(uintptr_t)p) return 0; *end=(uintptr_t)p+n; return 1; }
static int extent(int32_t lo, int32_t hi, size_t *n)
{
    int64_t x=(int64_t)hi-lo+1;
    if (x<=0 || (uint64_t)x>SIZE_MAX) return 0;
    *n=(size_t)x; return 1;
}

enum wt_kf_status kfb_layout(const struct kfb_header *h, struct kfb_layout *l)
{
    const int32_t *b=h->scalar;
    size_t ni,nj,nk,ti,tj,tk,plane,volume,total=sizeof(*h),heap=0;
    unsigned a;
    if (h->schema!=WT_KF_BATCH_SCHEMA || h->reserved ||
        (h->state!=1 && h->state!=2)) return WT_KF_BAD_ARGUMENT;
    if (b[WT_KF_TRIGGER]!=1 || b[WT_KF_ADAPT_STEP_FLAG]!=0 ||
        b[WT_KF_KTS]!=1 || b[WT_KF_ITS]<b[WT_KF_IMS] ||
        b[WT_KF_ITE]>b[WT_KF_IME] || b[WT_KF_JTS]<b[WT_KF_JMS] ||
        b[WT_KF_JTE]>b[WT_KF_JME] || b[WT_KF_KTS]<b[WT_KF_KMS] ||
        (int64_t)b[WT_KF_KTE]+1>b[WT_KF_KME]) return WT_KF_BAD_ARGUMENT;
    if (!extent(b[6],b[7],&ni) || !extent(b[8],b[9],&nj) ||
        !extent(b[10],b[11],&nk) || !extent(b[12],b[13],&ti) ||
        !extent(b[14],b[15],&tj) || !extent(b[16],b[17],&tk) ||
        !kfb_mul(ni,nk,&plane) || !kfb_mul(plane,nj,&volume) ||
        !kfb_mul(volume,4,&volume) || !kfb_mul(plane,4,&plane))
        return WT_KF_BAD_ARGUMENT;
    memset(l,0,sizeof(*l));
    for (a=0;a<WT_KF_NARG;++a) {
        struct kfb_field *f=&l->field[a];
        size_t full2,sz;
        int inactive=b[WT_KF_KF_EDRATES]!=1 && a>=WT_KF_UDR_KF && a<=WT_KF_TIMEC_KF;
        if (h->present[a]>1 || (!h->present[a] && a!=WT_KF_SHALL) ||
            h->alignment[a]>63 || (h->alignment[a]&3)) return WT_KF_BAD_ARGUMENT;
        if (!h->present[a]) {
            if (h->alignment[a] || b[a]) return WT_KF_BAD_ARGUMENT;
            continue;
        }
        if (wt_kf_shape[a]!=WT_KF_SCALAR && b[a]) return WT_KF_BAD_ARGUMENT;
        if (!kfb_mul(ni,nj,&full2) || !kfb_mul(full2,4,&full2)) return WT_KF_BAD_ARGUMENT;
        f->full=wt_kf_shape[a]==WT_KF_3D ? volume : wt_kf_shape[a]==WT_KF_2D ? full2 : 4;
        f->writable=wt_kf_shape[a]!=WT_KF_SCALAR && wt_kf_intent[a]!=WT_KF_IN;
        if (inactive) { f->full=4; f->writable=0; }
        f->offset=total;
        if (wt_kf_shape[a]==WT_KF_SCALAR) {
            f->packed=0; /* Scalars are carried once, in the immutable header. */
        } else if (inactive) {
            f->row=f->packed=4; f->rows=1;
        } else if (a==WT_KF_TH || a==WT_KF_RQVFTEN) {
            /* CPS TH is never referenced; RQVFTEN only trigger2 :280-282,369.
             * Full backing remains available, but neither owner reads nor writes.
             * Source SHA256 is pinned by the generated CPS79 schema. */
            f->writable=0;
        } else {
            size_t levels=wt_kf_shape[a]==WT_KF_3D ? tk+(a==WT_KF_W) : 1;
            f->row=ti*4; f->rows=levels;
            f->stride=ni*4;
            f->plane=wt_kf_shape[a]==WT_KF_3D ? plane : ni*4;
            f->first=(size_t)((int64_t)b[14]-b[8])*f->plane+
                (size_t)((int64_t)b[12]-b[6])*4;
            if (wt_kf_shape[a]==WT_KF_3D)
                f->first+=(size_t)((int64_t)b[16]-b[10])*f->stride;
            if (!kfb_mul(f->row,levels,&sz) || !kfb_mul(sz,tj,&f->packed)) return WT_KF_BAD_ARGUMENT;
        }
        if (!kfb_add(total,f->packed,&total) ||
            !kfb_add(f->full,3*KFB_GUARD,&sz) || !kfb_add(heap,sz,&heap)) return WT_KF_BUDGET;
    }
    l->bytes=total; l->heap=heap;
    return WT_KF_OK;
}

/* All offsets and lengths originate in the checked layout, never wire offsets. */
int kfb_transfer(const struct kfb_header *h, unsigned a,
                 const struct kfb_field *f, void *full, void *packed, int direction)
{
    size_t j,k,p=0,rows=f->rows;
    size_t nj=(size_t)((int64_t)h->scalar[WT_KF_JTE]-h->scalar[WT_KF_JTS]+1);
    unsigned char *x=full,*y=packed;
    (void)a;
    if (!f->packed) return 1;
    if (f->full==4) nj=1;
    for (j=0;j<nj;++j) for (k=0;k<rows;++k) {
        size_t off=f->first+j*f->plane+k*f->stride;
        if (direction==0) memcpy(y+p,x+off,f->row);
        else if (direction==1) memcpy(x+off,y+p,f->row);
        else if (memcmp(x+off,y+p,f->row)) return 0;
        p+=f->row;
    }
    return 1;
}

int kfb_writes_only(const struct kfb_header *h, unsigned a,
                    const struct kfb_field *f, const unsigned char *full,
                    const unsigned char *before)
{
    size_t j,k,pos=0,p=0;
    size_t nj=(size_t)((int64_t)h->scalar[WT_KF_JTE]-h->scalar[WT_KF_JTS]+1);
    if (wt_kf_shape[a]==WT_KF_SCALAR)
        return !memcmp(full,&h->scalar[a],4);
    if (!f->packed) {
        for (pos=0;pos<f->full;++pos) if (full[pos]!=KFB_FILL) return 0;
        return 1;
    }
    if (f->full==4) nj=1;
    for (j=0;j<nj;++j) for (k=0;k<f->rows;++k) {
        size_t off=f->first+j*f->plane+k*f->stride;
        while (pos<off) if (full[pos++]!=KFB_FILL) return 0;
        if (!f->writable && memcmp(full+off,before+p,f->row)) return 0;
        pos=off+f->row; p+=f->row;
    }
    while (pos<f->full) if (full[pos++]!=KFB_FILL) return 0;
    return 1;
}

enum wt_kf_status wt_kf_batch_execute(const void *request, size_t bytes,
    void *result, size_t result_capacity, size_t budget,
    wt_kf_callback callback, void *context, int *invoked)
{
    struct kfb_header h;
    struct kfb_layout l;
    unsigned char *base[WT_KF_NARG]={0};
    void *argv[WT_KF_NARG]={0},*original[WT_KF_NARG]={0};
    const unsigned char *wire=request;
    uintptr_t re,qe;
    size_t total;
    unsigned a,k;
    enum wt_kf_status status;
    if (!invoked) return WT_KF_BAD_ARGUMENT;
    *invoked=0;
    if (!request || !result || !callback || bytes<sizeof(h) ||
        !kfb_interval(request,bytes,&qe) || !kfb_interval(result,result_capacity,&re)) return WT_KF_BAD_ARGUMENT;
    if ((uintptr_t)request<re && (uintptr_t)result<qe) return WT_KF_OVERLAP;
    memcpy(&h,request,sizeof(h));
    status=kfb_layout(&h,&l);
    if (status!=WT_KF_OK) return status;
    if (h.state!=1 || h.bytes!=bytes || l.bytes!=bytes) return WT_KF_BAD_ARGUMENT;
    if (result_capacity<bytes) return WT_KF_CAPACITY;
    if (!kfb_add(l.heap,sizeof(l)+sizeof(h)+sizeof(base)+sizeof(argv)+sizeof(original),&total) || total>budget) return WT_KF_BUDGET;
    for (a=0;a<WT_KF_NARG;++a) {
        size_t n=l.field[a].full;
        uintptr_t aligned;
        if (!h.present[a]) continue;
        base[a]=malloc(n+3*KFB_GUARD);
        if (!base[a]) { status=WT_KF_NOMEM; goto done; }
        aligned=((uintptr_t)base[a]+16+63)&~(uintptr_t)63;
        /* At most 63 leading bytes +64 guard+63 alignment residue. */
        argv[a]=(void *)(aligned+h.alignment[a]);
        /* Allocate enough tail despite original modulo64 alignment. */
        if ((uintptr_t)argv[a]+n+16>(uintptr_t)base[a]+n+3*KFB_GUARD) {
            status=WT_KF_BAD_ARGUMENT; goto done;
        }
        original[a]=argv[a];
        memset((unsigned char *)argv[a]-16,0x5a,16);
        memset(argv[a],KFB_FILL,n);
        memset((unsigned char *)argv[a]+n,0xa5,16);
        if (wt_kf_shape[a]==WT_KF_SCALAR) memcpy(argv[a],&h.scalar[a],4);
        else kfb_transfer(&h,a,&l.field[a],argv[a],(void *)(wire+l.field[a].offset),1);
    }
    *invoked=1;
    if (callback(argv,context)) { status=WT_KF_BAD_STATE; goto done; }
    for (a=0;a<WT_KF_NARG;++a) {
        if (argv[a]!=original[a]) { status=WT_KF_BAD_STATE; goto done; }
        if (!argv[a]) continue;
        for (k=0;k<16;++k) {
            if (((unsigned char *)original[a])[(ptrdiff_t)k-16]!=0x5a ||
                ((unsigned char *)original[a])[l.field[a].full+k]!=0xa5) {
                status=WT_KF_GUARD; goto done;
            }
        }
        if (!kfb_writes_only(&h,a,&l.field[a],original[a],wire+l.field[a].offset)) {
            status=WT_KF_STALE; goto done;
        }
    }
    memcpy(result,request,bytes);
    h.state=2; memcpy(result,&h,sizeof(h));
    for (a=0;a<WT_KF_NARG;++a) if (l.field[a].writable)
        kfb_transfer(&h,a,&l.field[a],original[a],(unsigned char *)result+l.field[a].offset,0);
    status=WT_KF_OK;
 done:
    for (a=0;a<WT_KF_NARG;++a) free(base[a]);
    return status;
}
