#include "wt_sfclay_batch_internal.h"

#include <string.h>

_Static_assert(sizeof(float) == 4, "SFCLAY requires four-byte REAL");
_Static_assert(sizeof(int32_t) == 4, "SFCLAY requires four-byte INTEGER");

int sfb_add(size_t a, size_t b, size_t *r)
{ if (b > SIZE_MAX-a) return 0; *r=a+b; return 1; }

int sfb_mul(size_t a, size_t b, size_t *r)
{ if (a && b > SIZE_MAX/a) return 0; *r=a*b; return 1; }

int sfb_interval(const void *p, size_t n, uintptr_t *end)
{ if (!p || n > UINTPTR_MAX-(uintptr_t)p) return 0; *end=(uintptr_t)p+n; return 1; }

static int extent(int32_t lo, int32_t hi, size_t *n)
{
    int64_t x=(int64_t)hi-lo+1;
    if (x<=0 || (uint64_t)x>SIZE_MAX) return 0;
    *n=(size_t)x; return 1;
}

enum wt_sfclay_batch_status sfb_layout(const struct sfb_header *h,
                                       struct sfb_layout *l)
{
    const int32_t *b=h->scalar;
    size_t ni,nj,nk,n2,n3,row,total=sizeof(*h),heap=0;
    unsigned a;
    if (h->schema!=WT_SFCLAY_BATCH_SCHEMA || h->reserved ||
        (h->state!=1 && h->state!=2)) return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    if (!extent(b[SFB_IMS],b[SFB_IME],&ni) ||
        !extent(b[SFB_JMS],b[SFB_JME],&nj) ||
        !extent(b[SFB_KMS],b[SFB_KME],&nk) ||
        b[SFB_ITS]<b[SFB_IMS] || b[SFB_ITE]>b[SFB_IME] ||
        b[SFB_ITS]>b[SFB_ITE] || b[SFB_JTS]!=b[SFB_JTE] ||
        b[SFB_JTS]<b[SFB_JMS] || b[SFB_JTS]>b[SFB_JME] ||
        b[SFB_KTS]>b[SFB_KTE] || b[SFB_KTS]<b[SFB_KMS] ||
        b[SFB_KTE]>b[SFB_KME] || b[SFB_KMS]>1 || b[SFB_KME]<1)
        return WT_SFCLAY_BATCH_BOUNDS;
    if (!extent(b[SFB_ITS],b[SFB_ITE],&row) || !sfb_mul(ni,nj,&n2) ||
        !sfb_mul(n2,4,&n2) || !sfb_mul(ni,nk,&n3) ||
        !sfb_mul(n3,nj,&n3) || !sfb_mul(n3,4,&n3) ||
        !sfb_mul(row,4,&row)) return WT_SFCLAY_BATCH_OVERFLOW;
    memset(l,0,sizeof(*l));
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        struct sfb_field *f=&l->field[a];
        size_t allocation;
        int optional_control=a>=81;
        if (h->present[a]>1 || h->alignment[a]>63 ||
            (h->alignment[a]&3) || (!h->present[a] && !optional_control))
            return WT_SFCLAY_BATCH_BAD_ARGUMENT;
        if (!h->present[a]) {
            if (h->alignment[a] || h->scalar[a])
                return WT_SFCLAY_BATCH_BAD_ARGUMENT;
            continue;
        }
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR && h->scalar[a])
            return WT_SFCLAY_BATCH_BAD_ARGUMENT;
        f->full=wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D ? n3 :
            wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_2D ? n2 : 4;
        f->writable=wt_sfclay_schema_access[a]==WT_SFCLAY_WRITABLE;
        f->offset=total;
        if (wt_sfclay_schema_shape[a]!=WT_SFCLAY_SHAPE_SCALAR) {
            size_t j=(size_t)((int64_t)b[SFB_JTS]-b[SFB_JMS]);
            size_t i=(size_t)((int64_t)b[SFB_ITS]-b[SFB_IMS]);
            f->packed=row;
            if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_3D) {
                size_t k=(size_t)((int64_t)1-b[SFB_KMS]);
                if (!sfb_mul(j,nk,&f->first) ||
                    !sfb_add(f->first,k,&f->first) ||
                    !sfb_mul(f->first,ni,&f->first) ||
                    !sfb_add(f->first,i,&f->first) ||
                    !sfb_mul(f->first,4,&f->first))
                    return WT_SFCLAY_BATCH_OVERFLOW;
            } else {
                if (!sfb_mul(j,ni,&f->first) ||
                    !sfb_add(f->first,i,&f->first) ||
                    !sfb_mul(f->first,4,&f->first))
                    return WT_SFCLAY_BATCH_OVERFLOW;
            }
        }
        if (!sfb_add(total,f->packed,&total) ||
            !sfb_add(f->full,3*SFB_GUARD,&allocation) ||
            !sfb_add(heap,allocation,&heap))
            return WT_SFCLAY_BATCH_OVERFLOW;
    }
    l->bytes=total; l->heap=heap; return WT_SFCLAY_BATCH_OK;
}

int sfb_transfer(const struct sfb_field *f, void *full, void *packed,
                 int direction)
{
    unsigned char *x=full,*y=packed;
    if (!f->packed) return 1;
    if (direction==0) memcpy(y,x+f->first,f->packed);
    else if (direction==1) memcpy(x+f->first,y,f->packed);
    else if (memcmp(x+f->first,y,f->packed)) return 0;
    return 1;
}

int sfb_writes_only(const struct sfb_header *h, unsigned a,
                    const struct sfb_field *f, const unsigned char *full,
                    const unsigned char *before)
{
    size_t p;
    if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR)
        return !memcmp(full,&h->scalar[a],4);
    for (p=0;p<f->first;++p) if (full[p]!=SFB_FILL) return 0;
    if (!f->writable && memcmp(full+f->first,before,f->packed)) return 0;
    for (p=f->first+f->packed;p<f->full;++p)
        if (full[p]!=SFB_FILL) return 0;
    return 1;
}
