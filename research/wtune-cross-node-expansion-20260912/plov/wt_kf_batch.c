#include "wt_kf_batch_internal.h"
#include <stdlib.h>
#include <string.h>

static _Thread_local unsigned char owner_thread_token;

enum kfb_state { KFB_PREPARED, KFB_VALIDATED, KFB_COMMITTED, KFB_FAILED };
struct wt_kf_batch {
    const void *owner_thread;
    struct kfb_header header;
    struct kfb_layout layout;
    void *owner[WT_KF_NARG];
    unsigned char *request, *validated;
    enum kfb_state state;
};
static enum wt_kf_status fail(struct wt_kf_batch *b, enum wt_kf_status s)
{ b->state=KFB_FAILED; return s; }

enum wt_kf_status wt_kf_batch_prepare(void *args[WT_KF_NARG],
    const size_t capacity[WT_KF_NARG], size_t budget, struct wt_kf_batch **out)
{
    struct kfb_header h;
    struct kfb_layout l;
    struct wt_kf_batch *b;
    uintptr_t ends[WT_KF_NARG];
    enum wt_kf_status s;
    size_t total,wiretotal;
    unsigned a,j;
    if (!out) return WT_KF_BAD_ARGUMENT;
    *out=NULL;
    if (!args || !capacity) return WT_KF_BAD_ARGUMENT;
    memset(&h,0,sizeof(h));
    h.schema=WT_KF_BATCH_SCHEMA; h.state=1;
    /* Complete minimum capacity validation before dereferencing any scalar. */
    for (a=0;a<WT_KF_NARG;++a) {
        if (!args[a]) {
            if (a!=WT_KF_SHALL) return WT_KF_BAD_ARGUMENT;
            continue;
        }
        if (capacity[a]<4) return WT_KF_CAPACITY;
        if (((uintptr_t)args[a]&3) || !kfb_interval(args[a],4,&ends[a])) return WT_KF_BAD_ARGUMENT;
        h.present[a]=1; h.alignment[a]=(unsigned char)((uintptr_t)args[a]&63);
    }
    for (a=0;a<WT_KF_NARG;++a)
        if (h.present[a] && wt_kf_shape[a]==WT_KF_SCALAR) memcpy(&h.scalar[a],args[a],4);
    s=kfb_layout(&h,&l);
    if (s!=WT_KF_OK) return s;
    /* Full formal capacities and aliases remain checked even for unused arrays.
     * They are a live-storage assertion, never inferred from dummy shape. */
    for (a=0;a<WT_KF_NARG;++a) if (args[a]) {
        if (capacity[a]<l.field[a].full) return WT_KF_CAPACITY;
        if (!kfb_interval(args[a],l.field[a].full,&ends[a])) return WT_KF_BAD_ARGUMENT;
        for (j=0;j<a;++j) if (args[j] &&
            (uintptr_t)args[a]<ends[j] && (uintptr_t)args[j]<ends[a]) return WT_KF_OVERLAP;
    }
    if (!kfb_mul(l.bytes,2,&wiretotal) || !kfb_add(sizeof(*b),wiretotal,&total) ||
        !kfb_add(total,l.heap,&total) || total>budget) return WT_KF_BUDGET;
    b=calloc(1,sizeof(*b));
    if (!b) return WT_KF_NOMEM;
    b->request=malloc(l.bytes); b->validated=malloc(l.bytes);
    if (!b->request || !b->validated) { wt_kf_batch_destroy(b); return WT_KF_NOMEM; }
    h.bytes=l.bytes;
    b->header=h; b->layout=l;
    memcpy(b->request,&h,sizeof(h));
    for (a=0;a<WT_KF_NARG;++a) {
        b->owner[a]=args[a];
        if (args[a]) kfb_transfer(&h,a,&l.field[a],args[a],b->request+l.field[a].offset,0);
    }
    b->owner_thread=&owner_thread_token;
    b->state=KFB_PREPARED; *out=b;
    return WT_KF_OK;
}

const void *wt_kf_batch_request(const struct wt_kf_batch *b, size_t *bytes)
{
    if (!b || b->owner_thread!=&owner_thread_token || b->state!=KFB_PREPARED || !bytes) return NULL;
    *bytes=b->layout.bytes; return b->request;
}

enum wt_kf_status wt_kf_batch_validate(struct wt_kf_batch *b,
    const void *result, size_t bytes)
{
    struct kfb_header h;
    unsigned a;
    if (!b || b->owner_thread!=&owner_thread_token || b->state!=KFB_PREPARED) return WT_KF_BAD_STATE;
    if (!result || bytes!=b->layout.bytes) return fail(b,WT_KF_BAD_ARGUMENT);
    memcpy(&h,result,sizeof(h));
    if (h.state!=2) return fail(b,WT_KF_BAD_STATE);
    h.state=1;
    if (memcmp(&h,&b->header,sizeof(h))) return fail(b,WT_KF_STALE);
    /* Snapshot just the permitted owner bytes, avoiding other tile writes.
     * Scalar flags are immutable, including NO_INTENT logical optionals. */
    for (a=0;a<WT_KF_NARG;++a) if (b->owner[a]) {
        struct kfb_field *f=&b->layout.field[a];
        const unsigned char *old=b->request+f->offset;
        const unsigned char *newbytes=(const unsigned char *)result+f->offset;
        if (wt_kf_shape[a]==WT_KF_SCALAR) {
            if (memcmp(b->owner[a],&b->header.scalar[a],4)) return fail(b,WT_KF_STALE);
        } else if (!kfb_transfer(&h,a,f,b->owner[a],(void *)old,2)) return fail(b,WT_KF_STALE);
        if (!f->writable && f->packed && memcmp(old,newbytes,f->packed)) return fail(b,WT_KF_STALE);
    }
    /* Private validated copy makes apply independent of mutable shared bytes. */
    memcpy(b->validated,result,bytes); b->state=KFB_VALIDATED;
    return WT_KF_OK;
}

enum wt_kf_status wt_kf_batch_apply(struct wt_kf_batch *b)
{
    unsigned a;
    if (!b || b->owner_thread!=&owner_thread_token || b->state!=KFB_VALIDATED) return WT_KF_BAD_STATE;
    for (a=0;a<WT_KF_NARG;++a) if (b->layout.field[a].writable)
        kfb_transfer(&b->header,a,&b->layout.field[a],b->owner[a],
                     b->validated+b->layout.field[a].offset,1);
    b->state=KFB_COMMITTED; return WT_KF_OK;
}

enum wt_kf_status wt_kf_batch_commit(struct wt_kf_batch *b,
    const void *result, size_t bytes)
{
    enum wt_kf_status s=wt_kf_batch_validate(b,result,bytes);
    return s==WT_KF_OK ? wt_kf_batch_apply(b) : s;
}

void wt_kf_batch_destroy(struct wt_kf_batch *b)
{ if (b) { free(b->request); free(b->validated); free(b); } }
