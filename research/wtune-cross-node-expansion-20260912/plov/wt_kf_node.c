#include "wt_kf_node.h"
#include "wt_kf_batch_internal.h"
#include <stdlib.h>
#include <string.h>

static int valid_context(const struct wt_kf_node_context *c)
{
    return c && c->service && c->native && c->byte_budget &&
        c->signature.phase==WT_KF_NODE_PHASE && c->signature.schema==WT_KF_BATCH_SCHEMA &&
        c->signature.branch==1 && c->signature.profile && c->signature.init_generation;
}
static int same_signature(const struct wt_nl_signature *a, const struct wt_nl_signature *b)
{
    return a->phase==b->phase && a->profile==b->profile && a->schema==b->schema &&
        a->grid==b->grid && a->epoch==b->epoch && a->branch==b->branch &&
        a->init_generation==b->init_generation;
}
static int compatible(void *context, const struct wt_nl_descriptor *d,
                      const void *geometry, size_t bytes)
{
    struct wt_kf_node_context *c=context;
    return valid_context(c) && same_signature(&c->signature,&d->signature) &&
        geometry && bytes==18*sizeof(int32_t) && !d->reserved &&
        d->request_bytes==d->result_bytes;
}
static int execute(void *context, const struct wt_nl_descriptor *d,
    const void *geometry, size_t geometry_bytes, const void *request, size_t bytes,
    void *result, size_t capacity, size_t *written)
{
    struct wt_kf_node_context *c=context;
    struct kfb_header h;
    int invoked=0;
    enum wt_kf_status s;
    *written=0;
    if (!compatible(context,d,geometry,geometry_bytes) || bytes<sizeof(h) ||
        bytes!=d->request_bytes) return WT_NL_EXECUTE_REJECTED;
    memcpy(&h,request,sizeof(h));
    if (memcmp(h.scalar,geometry,geometry_bytes)) return WT_NL_EXECUTE_REJECTED;
    s=wt_kf_batch_execute(request,bytes,result,capacity,c->byte_budget,c->native,c->native_context,&invoked);
    if (s!=WT_KF_OK) return invoked ? WT_NL_EXECUTE_FATAL : WT_NL_EXECUTE_REJECTED;
    *written=bytes; return WT_NL_EXECUTE_OK;
}

enum wt_nl_status wt_kf_node_publish(struct wt_kf_node_context *c,
    void *args[WT_KF_NARG], const size_t capacity[WT_KF_NARG], uint64_t sequence,
    struct wt_kf_node_call *call, enum wt_kf_status *batch_status)
{
    enum wt_nl_status s;
    const void *request;
    size_t bytes;
    struct kfb_header h;
    if (!call || !batch_status || !valid_context(c)) return WT_NL_INVALID;
    if (call->batch || call->published) return WT_NL_BUSY;
    memset(call,0,sizeof(*call));
    *batch_status=wt_kf_batch_prepare(args,capacity,c->byte_budget,&call->batch);
    if (*batch_status!=WT_KF_OK) return WT_NL_REJECTED;
    request=wt_kf_batch_request(call->batch,&bytes);
    memcpy(&h,request,sizeof(h));
    memcpy(call->geometry,h.scalar,sizeof(call->geometry));
    call->descriptor.signature=c->signature;
    call->descriptor.call_sequence=sequence;
    call->descriptor.geometry_bytes=sizeof(call->geometry);
    call->descriptor.request_bytes=call->descriptor.result_bytes=bytes;
    call->owner_rank=c->rank; call->owner_thread=c->thread;
    s=wt_nl_publish(c->service,c->rank,c->thread,&call->descriptor,call->geometry,request,&call->ticket);
    if (s!=WT_NL_OK) { wt_kf_node_destroy(call); return s; }
    call->published=1; return WT_NL_OK;
}

enum wt_nl_status wt_kf_node_progress(struct wt_kf_node_context *c,
    uint64_t tag, struct wt_nl_ticket *executed)
{
    if (!valid_context(c)) return WT_NL_INVALID;
    return wt_nl_try_execute(c->service,c->rank,c->thread,&c->signature,compatible,execute,c,tag,executed);
}

static int owns(const struct wt_kf_node_context *c, const struct wt_kf_node_call *call)
{
    return valid_context(c) && call && call->published && call->batch &&
        c->rank==call->owner_rank && c->thread==call->owner_thread &&
        same_signature(&c->signature,&call->descriptor.signature);
}
enum wt_nl_status wt_kf_node_owner_execute(struct wt_kf_node_context *c,
    struct wt_kf_node_call *call, uint64_t tag)
{
    if (!owns(c,call)) return WT_NL_OWNER;
    return wt_nl_owner_execute(c->service,c->rank,c->thread,&call->ticket,execute,c,tag);
}
struct validate_context {
    struct wt_kf_node_context *node;
    struct wt_kf_node_call *call;
};
static int validate(void *context, const struct wt_nl_descriptor *d,
    const void *geometry, size_t geometry_bytes, const void *request, size_t bytes,
    const void *result, size_t result_bytes)
{
    struct validate_context *v=context;
    struct wt_kf_node_call *call=v->call;
    size_t expected_bytes;
    const void *expected=wt_kf_batch_request(call->batch,&expected_bytes);
    if (!expected || !same_signature(&d->signature,&call->descriptor.signature) ||
        d->call_sequence!=call->descriptor.call_sequence ||
        geometry_bytes!=sizeof(call->geometry) ||
        memcmp(geometry,call->geometry,sizeof(call->geometry)) ||
        bytes!=expected_bytes || memcmp(request,expected,bytes)) return 0;
    if (v->node->shadow_remote) {
        void *reference=malloc(bytes);
        int invoked=0;
        enum wt_kf_status status;
        call->shadow_checked=1; call->shadow_bytes=bytes;
        if (!reference) { call->shadow_failed=1; return 0; }
        status=wt_kf_batch_execute(request,bytes,reference,bytes,
            v->node->byte_budget,v->node->native,v->node->native_context,&invoked);
        if (status!=WT_KF_OK || !invoked || result_bytes!=bytes ||
            memcmp(reference,result,bytes)) call->shadow_failed=1;
        free(reference);
        if (call->shadow_failed) return 0;
    }
    return wt_kf_batch_validate(call->batch,result,result_bytes)==WT_KF_OK;
}
static void commit(void *context, const struct wt_nl_descriptor *d,
    const void *geometry, size_t geometry_bytes, const void *request, size_t bytes,
    const void *result, size_t result_bytes)
{
    struct validate_context *v=context;
    struct wt_kf_node_call *call=v->call;
    (void)d; (void)geometry; (void)geometry_bytes; (void)request; (void)bytes;
    (void)result; (void)result_bytes;
    /* validate froze the result privately; service invokes this exactly once. */
    call->committed=wt_kf_batch_apply(call->batch)==WT_KF_OK;
}
enum wt_nl_status wt_kf_node_collect(struct wt_kf_node_context *c, struct wt_kf_node_call *call)
{
    struct validate_context context={c,call};
    if (!owns(c,call)) return WT_NL_OWNER;
    return wt_nl_owner_collect(c->service,c->rank,c->thread,&call->ticket,
                               validate,commit,&context);
}
enum wt_nl_status wt_kf_node_finish(struct wt_kf_node_context *c, struct wt_kf_node_call *call)
{
    enum wt_nl_status s;
    if (!owns(c,call) || !call->committed) return WT_NL_OWNER;
    s=wt_nl_finish(c->service,c->rank,c->thread,&call->ticket);
    if (s==WT_NL_OK) call->published=0;
    return s;
}
void wt_kf_node_destroy(struct wt_kf_node_call *call)
{
    if (!call) return;
    wt_kf_batch_destroy(call->batch); memset(call,0,sizeof(*call));
}
