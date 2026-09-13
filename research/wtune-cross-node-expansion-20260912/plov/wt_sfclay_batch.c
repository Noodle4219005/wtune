#include "wt_sfclay_batch_internal.h"

#include <stdlib.h>
#include <string.h>

static _Thread_local unsigned char sfb_owner_thread_token;

enum sfb_state { SFB_PREPARED, SFB_VALIDATED, SFB_COMMITTED, SFB_FAILED };

struct wt_sfclay_batch {
    const void *owner_thread;
    struct sfb_header header;
    struct sfb_layout layout;
    void *owner[WT_SFCLAY_NARG];
    unsigned char *request, *validated;
    enum sfb_state state;
};

static enum wt_sfclay_batch_status fail(struct wt_sfclay_batch *b,
                                        enum wt_sfclay_batch_status s)
{ b->state=SFB_FAILED; return s; }

enum wt_sfclay_batch_status wt_sfclay_batch_prepare(
    void *args[WT_SFCLAY_NARG], const size_t capacity[WT_SFCLAY_NARG],
    size_t budget, struct wt_sfclay_batch **out)
{
    struct sfb_header h;
    struct sfb_layout l;
    struct wt_sfclay_batch *b;
    uintptr_t ends[WT_SFCLAY_NARG]={0};
    size_t total,wire;
    unsigned a,j;
    enum wt_sfclay_batch_status status;
    if (!out) return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    *out=NULL;
    if (!args || !capacity) return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    memset(&h,0,sizeof(h)); h.schema=WT_SFCLAY_BATCH_SCHEMA; h.state=1;
    /* Validate every four-byte cell before reading any scalar. */
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        if (!args[a]) {
            if (a<81) return WT_SFCLAY_BATCH_BAD_ARGUMENT;
            continue;
        }
        if (capacity[a]<4) return WT_SFCLAY_BATCH_CAPACITY;
        if (((uintptr_t)args[a]&3) || !sfb_interval(args[a],4,&ends[a]))
            return WT_SFCLAY_BATCH_BAD_ARGUMENT;
        h.present[a]=1;
        h.alignment[a]=(unsigned char)((uintptr_t)args[a]&63);
    }
    for (a=0;a<WT_SFCLAY_NARG;++a)
        if (h.present[a] && wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR)
            memcpy(&h.scalar[a],args[a],4);
    status=sfb_layout(&h,&l);
    if (status!=WT_SFCLAY_BATCH_OK) return status;
    for (a=0;a<WT_SFCLAY_NARG;++a) if (args[a]) {
        if (capacity[a]<l.field[a].full) return WT_SFCLAY_BATCH_CAPACITY;
        if (!sfb_interval(args[a],l.field[a].full,&ends[a]))
            return WT_SFCLAY_BATCH_OVERFLOW;
        for (j=0;j<a;++j) if (args[j] &&
            (wt_sfclay_schema_access[a]==WT_SFCLAY_WRITABLE ||
             wt_sfclay_schema_access[j]==WT_SFCLAY_WRITABLE) &&
            (uintptr_t)args[a]<ends[j] && (uintptr_t)args[j]<ends[a])
            return WT_SFCLAY_BATCH_OVERLAP;
    }
    if (!sfb_mul(l.bytes,2,&wire) || !sfb_add(sizeof(*b),wire,&total) ||
        !sfb_add(total,l.heap,&total)) return WT_SFCLAY_BATCH_OVERFLOW;
    if (total>budget) return WT_SFCLAY_BATCH_BUDGET;
    b=calloc(1,sizeof(*b));
    if (!b) return WT_SFCLAY_BATCH_NOMEM;
    b->request=malloc(l.bytes); b->validated=malloc(l.bytes);
    if (!b->request || !b->validated) {
        wt_sfclay_batch_destroy(b); return WT_SFCLAY_BATCH_NOMEM;
    }
    h.bytes=l.bytes; b->header=h; b->layout=l;
    memcpy(b->request,&h,sizeof(h));
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        b->owner[a]=args[a];
        if (args[a]) sfb_transfer(&l.field[a],args[a],
                                  b->request+l.field[a].offset,0);
    }
    b->owner_thread=&sfb_owner_thread_token;
    b->state=SFB_PREPARED; *out=b; return WT_SFCLAY_BATCH_OK;
}

const void *wt_sfclay_batch_request(const struct wt_sfclay_batch *b,
                                    size_t *bytes)
{
    if (!b || b->owner_thread!=&sfb_owner_thread_token ||
        b->state!=SFB_PREPARED || !bytes) return NULL;
    *bytes=b->layout.bytes; return b->request;
}

enum wt_sfclay_batch_status wt_sfclay_batch_validate(
    struct wt_sfclay_batch *b, const void *result, size_t bytes)
{
    struct sfb_header h;
    unsigned a;
    if (!b || b->owner_thread!=&sfb_owner_thread_token ||
        b->state!=SFB_PREPARED) return WT_SFCLAY_BATCH_STATE;
    if (!result || bytes!=b->layout.bytes)
        return fail(b,WT_SFCLAY_BATCH_BAD_ARGUMENT);
    memcpy(&h,result,sizeof(h));
    if (h.state!=2) return fail(b,WT_SFCLAY_BATCH_STATE);
    h.state=1;
    if (memcmp(&h,&b->header,sizeof(h)))
        return fail(b,WT_SFCLAY_BATCH_STALE);
    for (a=0;a<WT_SFCLAY_NARG;++a) if (b->owner[a]) {
        const struct sfb_field *f=&b->layout.field[a];
        const unsigned char *old=b->request+f->offset;
        const unsigned char *newbytes=(const unsigned char *)result+f->offset;
        if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR) {
            if (memcmp(b->owner[a],&b->header.scalar[a],4))
                return fail(b,WT_SFCLAY_BATCH_STALE);
        } else if (!sfb_transfer(f,b->owner[a],(void *)old,2)) {
            return fail(b,WT_SFCLAY_BATCH_STALE);
        }
        if (!f->writable && f->packed && memcmp(old,newbytes,f->packed))
            return fail(b,WT_SFCLAY_BATCH_STALE);
    }
    memcpy(b->validated,result,bytes);
    b->state=SFB_VALIDATED; return WT_SFCLAY_BATCH_OK;
}

enum wt_sfclay_batch_status wt_sfclay_batch_apply(struct wt_sfclay_batch *b)
{
    unsigned a;
    if (!b || b->owner_thread!=&sfb_owner_thread_token ||
        b->state!=SFB_VALIDATED) return WT_SFCLAY_BATCH_STATE;
    for (a=0;a<WT_SFCLAY_NARG;++a) if (b->layout.field[a].writable)
        sfb_transfer(&b->layout.field[a],b->owner[a],
                     b->validated+b->layout.field[a].offset,1);
    b->state=SFB_COMMITTED; return WT_SFCLAY_BATCH_OK;
}

enum wt_sfclay_batch_status wt_sfclay_batch_commit(
    struct wt_sfclay_batch *b, const void *result, size_t bytes)
{
    enum wt_sfclay_batch_status s=wt_sfclay_batch_validate(b,result,bytes);
    return s==WT_SFCLAY_BATCH_OK ? wt_sfclay_batch_apply(b) : s;
}

void wt_sfclay_batch_destroy(struct wt_sfclay_batch *b)
{
    if (b) { free(b->request); free(b->validated); free(b); }
}

enum wt_sfclay_batch_status wt_sfclay_batch_execute(
    const void *request, size_t bytes, void *result, size_t result_capacity,
    size_t budget, wt_sfclay_batch_callback callback, void *context,
    int *invoked)
{
    struct sfb_header h;
    struct sfb_layout l;
    unsigned char *base[WT_SFCLAY_NARG]={0};
    void *argv[WT_SFCLAY_NARG]={0},*original[WT_SFCLAY_NARG]={0};
    const unsigned char *wire=request;
    uintptr_t request_end,result_end;
    size_t total;
    unsigned a,k;
    enum wt_sfclay_batch_status status;
    if (!invoked) return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    *invoked=0;
    if (!request || !result || !callback || bytes<sizeof(h) ||
        !sfb_interval(request,bytes,&request_end) ||
        !sfb_interval(result,result_capacity,&result_end))
        return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    if ((uintptr_t)request<result_end && (uintptr_t)result<request_end)
        return WT_SFCLAY_BATCH_OVERLAP;
    memcpy(&h,request,sizeof(h));
    status=sfb_layout(&h,&l);
    if (status!=WT_SFCLAY_BATCH_OK) return status;
    if (h.state!=1 || h.bytes!=bytes || l.bytes!=bytes)
        return WT_SFCLAY_BATCH_BAD_ARGUMENT;
    if (result_capacity<bytes) return WT_SFCLAY_BATCH_CAPACITY;
    if (!sfb_add(l.heap,sizeof(l)+sizeof(h)+sizeof(base)+sizeof(argv)+
                 sizeof(original),&total)) return WT_SFCLAY_BATCH_OVERFLOW;
    if (total>budget) return WT_SFCLAY_BATCH_BUDGET;
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        size_t n=l.field[a].full;
        uintptr_t aligned;
        if (!h.present[a]) continue;
        base[a]=malloc(n+3*SFB_GUARD);
        if (!base[a]) { status=WT_SFCLAY_BATCH_NOMEM; goto done; }
        aligned=((uintptr_t)base[a]+SFB_EDGE_GUARD+63)&~(uintptr_t)63;
        argv[a]=(void *)(aligned+h.alignment[a]); original[a]=argv[a];
        if ((uintptr_t)argv[a]<((uintptr_t)base[a]+SFB_EDGE_GUARD) ||
            (uintptr_t)argv[a]+n+SFB_EDGE_GUARD>
                (uintptr_t)base[a]+n+3*SFB_GUARD) {
            status=WT_SFCLAY_BATCH_BAD_ARGUMENT; goto done;
        }
        memset((unsigned char *)argv[a]-SFB_EDGE_GUARD,0x5a,SFB_EDGE_GUARD);
        memset(argv[a],SFB_FILL,n);
        memset((unsigned char *)argv[a]+n,0xa5,SFB_EDGE_GUARD);
        if (wt_sfclay_schema_shape[a]==WT_SFCLAY_SHAPE_SCALAR)
            memcpy(argv[a],&h.scalar[a],4);
        else sfb_transfer(&l.field[a],argv[a],
                          (void *)(wire+l.field[a].offset),1);
    }
    *invoked=1;
    if (callback(argv,context)) { status=WT_SFCLAY_BATCH_STATE; goto done; }
    for (a=0;a<WT_SFCLAY_NARG;++a) {
        if (argv[a]!=original[a]) { status=WT_SFCLAY_BATCH_STATE; goto done; }
        if (!argv[a]) continue;
        for (k=0;k<SFB_EDGE_GUARD;++k) {
            if (((unsigned char *)original[a])[(ptrdiff_t)k-SFB_EDGE_GUARD]!=0x5a ||
                ((unsigned char *)original[a])[l.field[a].full+k]!=0xa5) {
                status=WT_SFCLAY_BATCH_GUARD; goto done;
            }
        }
        if (!sfb_writes_only(&h,a,&l.field[a],original[a],
                             wire+l.field[a].offset)) {
            status=WT_SFCLAY_BATCH_STALE; goto done;
        }
    }
    memcpy(result,request,bytes); h.state=2; memcpy(result,&h,sizeof(h));
    for (a=0;a<WT_SFCLAY_NARG;++a) if (l.field[a].writable)
        sfb_transfer(&l.field[a],original[a],
                     (unsigned char *)result+l.field[a].offset,0);
    status=WT_SFCLAY_BATCH_OK;
done:
    for (a=0;a<WT_SFCLAY_NARG;++a) free(base[a]);
    return status;
}
