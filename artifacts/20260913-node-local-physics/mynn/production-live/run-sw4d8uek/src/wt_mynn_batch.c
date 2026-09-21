#include "wt_mynn_batch.h"

#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

#define MYB_REQUEST 1u
#define MYB_RESULT 2u

struct myb_header {
    uint64_t schema;
    uint32_t state, logical_true_bits;
    size_t bytes;
    int32_t j;
    unsigned char present[WT_MYNN_NARG], active[WT_MYNN_NARG], writable[WT_MYNN_NARG];
    unsigned char alignment[WT_MYNN_NARG];
    int32_t scalar[WT_MYNN_NARG];
    size_t offset[WT_MYNN_NARG], packed[WT_MYNN_NARG], full[WT_MYNN_NARG];
};
struct wt_mynn_batch {
    const void *owner_thread;
    struct myb_header header;
    struct wt_mynn_layout layout;
    void *owner[WT_MYNN_NARG];
    size_t capacity[WT_MYNN_NARG];
    unsigned char *request, *validated;
    unsigned state;
};
struct scalar_slot { alignas(64) int32_t value; };
static _Thread_local unsigned char owner_thread_token;

enum { MYB_PREPARED, MYB_VALIDATED, MYB_COMMITTED, MYB_FAILED };

static enum wt_mynn_status fail(struct wt_mynn_batch *b, enum wt_mynn_status s)
{ b->state=MYB_FAILED; return s; }
static int add(size_t a, size_t b, size_t *out)
{ if (b>SIZE_MAX-a) return 0; *out=a+b; return 1; }
static int mul(size_t a, size_t b, size_t *out)
{ if (a && b>SIZE_MAX/a) return 0; *out=a*b; return 1; }
static int extent(int32_t low, int32_t high, size_t *out)
{ int64_t n=(int64_t)high-(int64_t)low+1; if(n<=0 || (uint64_t)n>SIZE_MAX) return 0; *out=(size_t)n; return 1; }
static int interval(const void *p, size_t n, uintptr_t *end)
{ uintptr_t a=(uintptr_t)p; if (!p || n>UINTPTR_MAX-a) return 0; *end=a+n; return 1; }
static int scalar(const struct myb_header *h, unsigned n)
{ return h->scalar[n]; }
static int condition(unsigned code, unsigned present, const struct myb_header *h)
{
    switch(code) {
    case WT_MYNN_COND_NEVER: return 0; case WT_MYNN_COND_ALWAYS: return 1;
    case WT_MYNN_COND_PRESENT: return present; case WT_MYNN_COND_QC: return scalar(h,WT_MYNN_FLAG_QC)!=0;
    case WT_MYNN_COND_QI: return scalar(h,WT_MYNN_FLAG_QI)!=0; case WT_MYNN_COND_QS: return scalar(h,WT_MYNN_FLAG_QS)!=0;
    case WT_MYNN_COND_QNC: return scalar(h,WT_MYNN_FLAG_QNC)!=0; case WT_MYNN_COND_QNI: return scalar(h,WT_MYNN_FLAG_QNI)!=0;
    case WT_MYNN_COND_QNWFA: return scalar(h,WT_MYNN_FLAG_QNWFA)!=0; case WT_MYNN_COND_QNIFA: return scalar(h,WT_MYNN_FLAG_QNIFA)!=0;
    case WT_MYNN_COND_QNBCA: return scalar(h,WT_MYNN_FLAG_QNBCA)!=0; case WT_MYNN_COND_PLUME: return scalar(h,WT_MYNN_BL_MYNN_EDMF)>0;
    case WT_MYNN_COND_OUTPUT: return scalar(h,WT_MYNN_BL_MYNN_OUTPUT)>0; case WT_MYNN_COND_SPP: return scalar(h,WT_MYNN_SPP_PBL)>0;
    case WT_MYNN_COND_BUDGET: return scalar(h,WT_MYNN_TKE_BUDGET)==1;
    default: return -1;
    }
}

static int row_shape(const struct myb_header *h, unsigned n, size_t *bytes, size_t *at)
{
    size_t ni,nk,active,row,plane;
    int32_t j=scalar(h,WT_MYNN_JTS);
    if (!extent(scalar(h,WT_MYNN_IMS),scalar(h,WT_MYNN_IME),&ni) ||
        !extent(scalar(h,WT_MYNN_KMS),scalar(h,WT_MYNN_KME),&nk) ||
        !extent(scalar(h,WT_MYNN_ITS),scalar(h,WT_MYNN_ITE),&active) ||
        j!=scalar(h,WT_MYNN_JTE) || j<scalar(h,WT_MYNN_JMS) || j>scalar(h,WT_MYNN_JME) ||
        !mul(ni,4,&row) || !mul((size_t)((int64_t)j-scalar(h,WT_MYNN_JMS)),row,&plane) ||
        !mul(active,4,bytes)) return 0;
    if (wt_mynn_args[n].rank==2) { *at=plane; return 1; }
    if (wt_mynn_args[n].rank==3 && mul(*bytes,nk,bytes) && mul(plane,nk,at)) return 1;
    return 0;
}
/* Direction 0 packs owner to wire; 1 unpacks wire to owner; 2 compares. */
static int transfer(const struct myb_header *h, unsigned n, void *owner, void *wire, int direction)
{
    size_t bytes,at,ni,nk,active,segment,k;
    unsigned char *p,*q=wire;
    if (!h->packed[n]) return 1;
    if (!wt_mynn_args[n].rank) {
        if (direction==0) memcpy(wire,owner,4);
        else if (direction==1) memcpy(owner,wire,4);
        else return !memcmp(owner,wire,4);
        return 1;
    }
    if (h->packed[n]==4 && !h->active[n] && !h->writable[n]) {
        if (direction==0) memcpy(wire,owner,4);
        else if (direction==2) return !memcmp(owner,wire,4);
        return direction!=1; /* inactive optional is never an output target */
    }
    if (!row_shape(h,n,&bytes,&at) || bytes!=h->packed[n] ||
        !extent(scalar(h,WT_MYNN_IMS),scalar(h,WT_MYNN_IME),&ni) ||
        !extent(scalar(h,WT_MYNN_ITS),scalar(h,WT_MYNN_ITE),&active) ||
        !mul(active,4,&segment) || !extent(scalar(h,WT_MYNN_KMS),scalar(h,WT_MYNN_KME),&nk)) return 0;
    p=(unsigned char *)owner+at+(size_t)((int64_t)scalar(h,WT_MYNN_ITS)-scalar(h,WT_MYNN_IMS))*4;
    if (wt_mynn_args[n].rank==2) nk=1;
    for(k=0;k<nk;++k) {
        if (direction==0) memcpy(q,p,segment);
        else if (direction==1) memcpy(p,q,segment);
        else if (memcmp(p,q,segment)) return 0;
        q+=segment; p+=ni*4;
    }
    return 1;
}
static int header_ok(const struct myb_header *h, size_t bytes)
{
    struct myb_header expected;
    size_t cursor=sizeof *h,ni,nj,nk;
    unsigned n;
    if (!h || h->schema!=WT_MYNN_BATCH_SCHEMA || h->bytes!=bytes || bytes<sizeof *h ||
        (h->logical_true_bits!=1 && h->logical_true_bits!=UINT32_MAX) ||
        scalar(h,WT_MYNN_IDS)>=scalar(h,WT_MYNN_IDE) || scalar(h,WT_MYNN_JDS)>=scalar(h,WT_MYNN_JDE) ||
        scalar(h,WT_MYNN_KDS)>=scalar(h,WT_MYNN_KDE) || !extent(scalar(h,WT_MYNN_IMS),scalar(h,WT_MYNN_IME),&ni) ||
        !extent(scalar(h,WT_MYNN_JMS),scalar(h,WT_MYNN_JME),&nj) || !extent(scalar(h,WT_MYNN_KMS),scalar(h,WT_MYNN_KME),&nk) ||
        scalar(h,WT_MYNN_ITS)<scalar(h,WT_MYNN_IMS) || scalar(h,WT_MYNN_ITE)>scalar(h,WT_MYNN_IME) ||
        scalar(h,WT_MYNN_JTS)<scalar(h,WT_MYNN_JMS) || scalar(h,WT_MYNN_JTE)>scalar(h,WT_MYNN_JME)) return 0;
    expected=*h;
    expected.j=scalar(h,WT_MYNN_JTS);
    for(n=0;n<WT_MYNN_NARG;++n) {
        const struct wt_mynn_arg_metadata *a=&wt_mynn_args[n]; int active,write; size_t full=4,row,at;
        if(h->present[n]>1 || (!h->present[n] && (h->alignment[n] || h->scalar[n])) ||
           (h->present[n] && (h->alignment[n]>63 || (h->alignment[n]&3))) ||
           (a->type==WT_MYNN_LOGICAL && h->present[n] && !a->rank &&
           h->scalar[n] && (uint32_t)h->scalar[n]!=h->logical_true_bits)) return 0;
        if(!h->present[n] && !a->optional) return 0;
        active=condition(a->access,h->present[n],h); write=condition(a->write,h->present[n],h);
        if(active<0||write<0||((active||write)&&!h->present[n])) return 0;
        if(a->rank==2 && (!mul(ni,nj,&full)||!mul(full,4,&full))) return 0;
        if(a->rank==3 && (!mul(ni,nk,&full)||!mul(full,nj,&full)||!mul(full,4,&full))) return 0;
        expected.active[n]=(unsigned char)active; expected.writable[n]=(unsigned char)write;
        expected.full[n]=h->present[n] ? ((!a->rank||active||write)?full:4) : 0; expected.offset[n]=0; expected.packed[n]=0;
        if(h->present[n]) { expected.offset[n]=cursor; expected.packed[n]=!a->rank||(!active&&!write)?4:0;
            if(!expected.packed[n] && !row_shape(h,n,&row,&at)) return 0;
            if(!expected.packed[n]) expected.packed[n]=row;
            if(!add(cursor,expected.packed[n],&cursor)) return 0;
        }
    }
    expected.bytes=cursor;
    return cursor==bytes && !memcmp(&expected,h,sizeof expected);
}
static int continuation_geometry(const struct myb_header *h)
{
    return scalar(h,WT_MYNN_TKE_BUDGET)==0 && scalar(h,WT_MYNN_SPP_PBL)==0 &&
        scalar(h,WT_MYNN_INITFLAG)==0 && scalar(h,WT_MYNN_KTS)==1 &&
        scalar(h,WT_MYNN_KMS)==1 && (int64_t)scalar(h,WT_MYNN_KME)-(int64_t)scalar(h,WT_MYNN_KTE)==1 &&
        scalar(h,WT_MYNN_JTS)==scalar(h,WT_MYNN_JTE);
}

/* No owner content read happens before wt_mynn_layout_build preflights all 132
 * intervals.  Full formal intervals are then checked for aliases even though
 * only a row is transported. */
enum wt_mynn_status wt_mynn_batch_prepare(void *argv[WT_MYNN_NARG],
    const size_t capacities[WT_MYNN_NARG], const struct wt_mynn_layout_config *cfg,
    size_t budget, struct wt_mynn_batch **out)
{
    struct wt_mynn_layout layout;
    struct myb_header h;
    struct wt_mynn_batch *b;
    uintptr_t ends[WT_MYNN_NARG];
    size_t total,twice,cursor;
    unsigned n,m;
    if (!out) return WT_MYNN_ARGUMENT;
    *out=NULL;
    if (!argv || !capacities || !budget ||
        wt_mynn_layout_build(argv,capacities,cfg,&layout)!=WT_MYNN_LAYOUT_OK) return WT_MYNN_LAYOUT;
    memset(&h,0,sizeof h);
    h.schema=WT_MYNN_BATCH_SCHEMA; h.state=MYB_REQUEST; h.logical_true_bits=cfg->logical_true_bits;
    for (n=0;n<WT_MYNN_NARG;++n) if (layout.present[n]) {
        h.present[n]=1; h.active[n]=layout.active[n]; h.writable[n]=layout.writable[n];
        h.alignment[n]=(unsigned char)((uintptr_t)argv[n]&63); h.full[n]=layout.required_bytes[n];
        if (!wt_mynn_args[n].rank) memcpy(&h.scalar[n],argv[n],4);
        if (capacities[n]<layout.required_bytes[n] || !interval(argv[n],layout.required_bytes[n],&ends[n])) return WT_MYNN_LAYOUT;
        for (m=0;m<n;++m) if (layout.present[m] && (uintptr_t)argv[n]<ends[m] && (uintptr_t)argv[m]<ends[n]) return WT_MYNN_ALIAS;
    }
    h.j=h.scalar[WT_MYNN_JTS];
    if (!continuation_geometry(&h)) return WT_MYNN_LAYOUT;
    cursor=sizeof h;
    for (n=0;n<WT_MYNN_NARG;++n) if (h.present[n]) {
        size_t row,at;
        h.offset[n]=cursor;
        if (!wt_mynn_args[n].rank || (!h.active[n] && !h.writable[n])) h.packed[n]=4;
        else if (!row_shape(&h,n,&row,&at)) return WT_MYNN_LAYOUT;
        else h.packed[n]=row;
        if (!add(cursor,h.packed[n],&cursor)) return WT_MYNN_OVERFLOW;
    }
    h.bytes=cursor;
    if (!mul(cursor,2,&twice) || !add(sizeof *b,twice,&total) || total>budget) return WT_MYNN_BUDGET;
    b=calloc(1,sizeof *b);
    if (!b) return WT_MYNN_ALLOCATION;
    b->request=malloc(cursor); b->validated=malloc(cursor);
    if (!b->request || !b->validated) { wt_mynn_batch_destroy(b); return WT_MYNN_ALLOCATION; }
    b->header=h; b->layout=layout; memcpy(b->owner,argv,sizeof b->owner); memcpy(b->capacity,capacities,sizeof b->capacity);
    memcpy(b->request,&h,sizeof h);
    for (n=0;n<WT_MYNN_NARG;++n) if (h.present[n] && !transfer(&h,n,argv[n],b->request+h.offset[n],0)) { wt_mynn_batch_destroy(b); return WT_MYNN_LAYOUT; }
    b->owner_thread=&owner_thread_token; b->state=MYB_PREPARED; *out=b;
    return WT_MYNN_OK;
}
const void *wt_mynn_batch_request(const struct wt_mynn_batch *b, size_t *bytes)
{
    if (!b || !bytes || b->owner_thread!=&owner_thread_token || b->state!=MYB_PREPARED) return NULL;
    *bytes=b->header.bytes; return b->request;
}
int wt_mynn_batch_geometry(const struct wt_mynn_batch *b, int32_t out[18])
{
    if (!b || !out || b->owner_thread!=&owner_thread_token || b->state!=MYB_PREPARED) return 0;
    memcpy(out,b->header.scalar,sizeof(int32_t)*18); return 1;
}

static int helper_total(const struct myb_header *h, size_t *total)
{
    size_t sum=0,n;
    for (n=0;n<WT_MYNN_NARG;++n) if (h->present[n] && wt_mynn_args[n].rank) {
        size_t allocation=h->active[n] || h->writable[n] ? h->full[n] : 4;
        if (!add(allocation,127,&allocation) || !add(sum,allocation,&sum)) return 0;
    }
    *total=sum; return 1;
}
static unsigned char *aligned_view(unsigned char *allocation, unsigned char alignment)
{
    uintptr_t p=(uintptr_t)(allocation+32), base=p&~(uintptr_t)63;
    p=base+alignment; if (p<(uintptr_t)(allocation+32)) p+=64;
    return (unsigned char *)p;
}
static int outside_ok(const struct myb_header *h, unsigned n, const unsigned char *view,
                      size_t full, const unsigned char *row, int writable)
{
    size_t bytes,at,ni,nk,active,segment,i,cursor=0;
    if (!row_shape(h,n,&bytes,&at) || !extent(scalar(h,WT_MYNN_IMS),scalar(h,WT_MYNN_IME),&ni) ||
        !extent(scalar(h,WT_MYNN_ITS),scalar(h,WT_MYNN_ITE),&active) || !mul(active,4,&segment) ||
        !extent(scalar(h,WT_MYNN_KMS),scalar(h,WT_MYNN_KME),&nk)) return 0;
    if (wt_mynn_args[n].rank==2) nk=1;
    for(i=0;i<nk;++i) {
        size_t base=at+i*ni*4+(size_t)((int64_t)scalar(h,WT_MYNN_ITS)-scalar(h,WT_MYNN_IMS))*4, j;
        if(base<cursor || base>full || segment>full-base) return 0;
        for(j=cursor;j<base;++j) if(view[j]!=0xcd) return 0;
        cursor=base+segment;
    }
    for(i=cursor;i<full;++i) if(view[i]!=0xcd) return 0;
    return writable || transfer(h,n,(void *)view,(void *)row,2);
}
enum wt_mynn_status wt_mynn_batch_execute(const void *request, size_t bytes,
    void *result, size_t result_capacity, size_t budget, wt_mynn_batch_callback callback,
    void *context, int *invoked)
{
    struct myb_header header;
    const struct myb_header *h=&header;
    struct scalar_slot private_scalar[WT_MYNN_NARG];
    unsigned char *allocation[WT_MYNN_NARG]={0}, *view[WT_MYNN_NARG]={0};
    void *argv[WT_MYNN_NARG]={0}, *saved_argv[WT_MYNN_NARG];
    size_t heap,n;
    enum wt_mynn_status status=WT_MYNN_OK;
    if (invoked) *invoked=0;
    if (!callback || !request || !result || bytes<sizeof header || bytes>result_capacity ||
        ((uintptr_t)request<(uintptr_t)result+bytes && (uintptr_t)result<(uintptr_t)request+bytes)) return WT_MYNN_ARGUMENT;
    memcpy(&header,request,sizeof header);
    if (!header_ok(h,bytes) ||
        h->state!=MYB_REQUEST || !continuation_geometry(h) || !helper_total(h,&heap) ||
        heap>budget || bytes>budget-heap) return WT_MYNN_ARGUMENT;
    memcpy(result,request,bytes);
    for(n=0;n<WT_MYNN_NARG;++n) private_scalar[n].value=h->scalar[n];
    for (n=0;n<WT_MYNN_NARG;++n) if (h->present[n]) {
        if (!wt_mynn_args[n].rank) { argv[n]=&private_scalar[n].value; continue; }
        {
            size_t full=h->active[n] || h->writable[n] ? h->full[n] : 4, at=0, row=h->packed[n];
            allocation[n]=malloc(full+127);
            if (!allocation[n]) { status=WT_MYNN_ALLOCATION; goto done; }
            view[n]=aligned_view(allocation[n],h->alignment[n]);
            memset(view[n]-32,0xa6,32); memset(view[n],0xcd,full); memset(view[n]+full,0x6a,32);
            if (full!=4 && !row_shape(h,n,&row,&at)) { status=WT_MYNN_LAYOUT; goto done; }
            if (full==4) memcpy(view[n],(unsigned char *)request+h->offset[n],4);
            else if (!transfer(h,n,view[n],(unsigned char *)request+h->offset[n],1)) { status=WT_MYNN_LAYOUT; goto done; }
            argv[n]=view[n];
        }
    }
    memcpy(saved_argv,argv,sizeof argv);
    if (invoked) *invoked=1;
    if (callback(context,argv)) { status=WT_MYNN_STATE; goto done; }
    for (n=0;n<WT_MYNN_NARG;++n) if (argv[n]!=saved_argv[n] || (!wt_mynn_args[n].rank && h->present[n] &&
        memcmp(&private_scalar[n].value,&h->scalar[n],4))) { status=WT_MYNN_READONLY; goto done; }
    for (n=0;n<WT_MYNN_NARG;++n) if (view[n]) {
        size_t full=h->active[n] || h->writable[n] ? h->full[n] : 4, at=0, row=h->packed[n], i;
        if (full!=4 && !row_shape(h,n,&row,&at)) { status=WT_MYNN_LAYOUT; goto done; }
        for (i=0;i<32;++i) if (view[n][-32+(ptrdiff_t)i]!=0xa6 || view[n][full+i]!=0x6a) { status=WT_MYNN_GUARD; goto done; }
        if (full==4) {
            if (memcmp(view[n],(const unsigned char *)request+h->offset[n],4)) { status=WT_MYNN_READONLY; goto done; }
            memcpy((unsigned char *)result+h->offset[n],view[n],4);
        } else {
            if (!outside_ok(h,n,view[n],full,(const unsigned char *)request+h->offset[n],h->writable[n])) { status=WT_MYNN_READONLY; goto done; }
            if (!transfer(h,n,view[n],(unsigned char *)result+h->offset[n],0)) { status=WT_MYNN_LAYOUT; goto done; }
        }
    }
    header.state=MYB_RESULT;
    memcpy(result,&header,sizeof header);
done:
    for (n=0;n<WT_MYNN_NARG;++n) free(allocation[n]);
    return status;
}

enum wt_mynn_status wt_mynn_batch_validate(struct wt_mynn_batch *b, const void *result, size_t bytes)
{
    struct myb_header h;
    unsigned n;
    if (!b || b->owner_thread!=&owner_thread_token || b->state!=MYB_PREPARED) return WT_MYNN_STATE;
    if (!result || bytes!=b->header.bytes) return fail(b,WT_MYNN_ARGUMENT);
    memcpy(&h,result,sizeof h);
    if (!header_ok(&h,bytes) || h.state!=MYB_RESULT) return fail(b,WT_MYNN_STATE);
    h.state=MYB_REQUEST;
    if (memcmp(&h,&b->header,sizeof h)) return fail(b,WT_MYNN_STALE);
    for (n=0;n<WT_MYNN_NARG;++n) if (b->header.present[n]) {
        const unsigned char *old=b->request+b->header.offset[n];
        const unsigned char *newbytes=(const unsigned char *)result+b->header.offset[n];
        if (!wt_mynn_args[n].rank && memcmp(b->owner[n],&b->header.scalar[n],4)) return fail(b,WT_MYNN_STALE);
        if (wt_mynn_args[n].rank && !transfer(&b->header,n,b->owner[n],(void *)old,2)) return fail(b,WT_MYNN_STALE);
        if (!b->header.writable[n] && memcmp(old,newbytes,b->header.packed[n])) return fail(b,WT_MYNN_READONLY);
    }
    memcpy(b->validated,result,bytes); b->state=MYB_VALIDATED; return WT_MYNN_OK;
}
enum wt_mynn_status wt_mynn_batch_apply(struct wt_mynn_batch *b)
{
    unsigned n;
    if (!b || b->owner_thread!=&owner_thread_token || b->state!=MYB_VALIDATED) return WT_MYNN_STATE;
    for (n=0;n<WT_MYNN_NARG;++n) if (b->header.writable[n] &&
        !transfer(&b->header,n,b->owner[n],b->validated+b->header.offset[n],1)) return fail(b,WT_MYNN_LAYOUT);
    b->state=MYB_COMMITTED; return WT_MYNN_OK;
}
enum wt_mynn_status wt_mynn_batch_commit(struct wt_mynn_batch *b, const void *result, size_t bytes)
{ enum wt_mynn_status s=wt_mynn_batch_validate(b,result,bytes); return s==WT_MYNN_OK ? wt_mynn_batch_apply(b) : s; }
void wt_mynn_batch_destroy(struct wt_mynn_batch *b)
{ if (b) { free(b->request); free(b->validated); free(b); } }
