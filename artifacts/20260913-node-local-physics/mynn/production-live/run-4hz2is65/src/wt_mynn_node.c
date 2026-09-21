#include "wt_mynn_node.h"
#include <string.h>

static int same_signature(const struct wt_nl_signature *a, const struct wt_nl_signature *b)
{
    return a && b && a->phase==b->phase && a->profile==b->profile &&
        a->schema==b->schema && a->grid==b->grid && a->epoch==b->epoch &&
        a->branch==b->branch && a->init_generation==b->init_generation;
}
/* profile/init_generation are reviewed closure-state identifiers supplied by
 * integration; this adapter cannot manufacture their attestation. */
static int valid_context(const struct wt_mynn_node_context *c)
{
    return c && c->service && c->native && c->byte_budget &&
        c->signature.phase==WT_MYNN_NODE_PHASE && c->signature.schema==WT_MYNN_BATCH_SCHEMA &&
        c->signature.branch==WT_MYNN_BATCH_SCHEMA && c->signature.profile &&
        c->signature.init_generation;
}
static int compatible(void *context, const struct wt_nl_descriptor *d,
                      const void *geometry, size_t geometry_bytes)
{
    struct wt_mynn_node_context *c=context;
    return valid_context(c) && d && !d->reserved && geometry &&
        geometry_bytes==18*sizeof(int32_t) && d->request_bytes==d->result_bytes &&
        same_signature(&c->signature,&d->signature);
}
static int execute(void *context, const struct wt_nl_descriptor *d,
    const void *geometry, size_t geometry_bytes, const void *request, size_t bytes,
    void *result, size_t capacity, size_t *written)
{
    struct wt_mynn_node_context *c=context;
    int invoked=0;
    enum wt_mynn_status status;
    (void)geometry;
    *written=0;
    if (!compatible(c,d,geometry,geometry_bytes) || bytes!=d->request_bytes) return WT_NL_EXECUTE_REJECTED;
    status=wt_mynn_batch_execute(request,bytes,result,capacity,c->byte_budget,c->native,c->native_context,&invoked);
    if (status!=WT_MYNN_OK) return invoked ? WT_NL_EXECUTE_FATAL : WT_NL_EXECUTE_REJECTED;
    *written=bytes; return WT_NL_EXECUTE_OK;
}
enum wt_nl_status wt_mynn_node_publish(struct wt_mynn_node_context *c,
    void *argv[WT_MYNN_NARG], const size_t capacities[WT_MYNN_NARG],
    const struct wt_mynn_layout_config *layout_config, uint64_t sequence,
    struct wt_mynn_node_call *call, enum wt_mynn_status *batch_status)
{
    const void *request;
    size_t bytes;
    enum wt_nl_status status;
    if (!call || !batch_status || !valid_context(c) || call->batch || call->published) return WT_NL_INVALID;
    memset(call,0,sizeof *call);
    *batch_status=wt_mynn_batch_prepare(argv,capacities,layout_config,c->byte_budget,&call->batch);
    if (*batch_status!=WT_MYNN_OK) return WT_NL_REJECTED;
    request=wt_mynn_batch_request(call->batch,&bytes);
    if (!request || !wt_mynn_batch_geometry(call->batch,call->geometry)) { wt_mynn_node_destroy(call); return WT_NL_REJECTED; }
    call->descriptor.signature=c->signature; call->descriptor.call_sequence=sequence;
    call->descriptor.geometry_bytes=sizeof call->geometry;
    call->descriptor.request_bytes=call->descriptor.result_bytes=bytes;
    call->owner_rank=c->rank; call->owner_thread=c->thread;
    status=wt_nl_publish(c->service,c->rank,c->thread,&call->descriptor,call->geometry,request,&call->ticket);
    if (status!=WT_NL_OK) wt_mynn_node_destroy(call); else call->published=1;
    return status;
}
enum wt_nl_status wt_mynn_node_progress(struct wt_mynn_node_context *c,
    uint64_t tag, struct wt_nl_ticket *executed)
{
    if (!valid_context(c)) return WT_NL_INVALID;
    return wt_nl_try_execute(c->service,c->rank,c->thread,&c->signature,compatible,execute,c,tag,executed);
}
static int owns(const struct wt_mynn_node_context *c, const struct wt_mynn_node_call *call)
{
    return valid_context(c) && call && call->published && call->batch &&
        c->rank==call->owner_rank && c->thread==call->owner_thread &&
        same_signature(&c->signature,&call->descriptor.signature);
}
enum wt_nl_status wt_mynn_node_owner_execute(struct wt_mynn_node_context *c,
    struct wt_mynn_node_call *call, uint64_t tag)
{
    if (!owns(c,call)) return WT_NL_OWNER;
    return wt_nl_owner_execute(c->service,c->rank,c->thread,&call->ticket,execute,c,tag);
}
static int validate(void *context, const struct wt_nl_descriptor *d,
    const void *geometry, size_t geometry_bytes, const void *request, size_t bytes,
    const void *result, size_t result_bytes)
{
    struct wt_mynn_node_call *call=context;
    size_t expected_bytes;
    const void *expected=wt_mynn_batch_request(call->batch,&expected_bytes);
    return expected && same_signature(&d->signature,&call->descriptor.signature) &&
        d->call_sequence==call->descriptor.call_sequence && geometry_bytes==sizeof call->geometry &&
        !memcmp(geometry,call->geometry,sizeof call->geometry) && bytes==expected_bytes &&
        !memcmp(request,expected,bytes) && wt_mynn_batch_validate(call->batch,result,result_bytes)==WT_MYNN_OK;
}
static void commit(void *context, const struct wt_nl_descriptor *d, const void *geometry,
    size_t geometry_bytes, const void *request, size_t bytes, const void *result, size_t result_bytes)
{
    struct wt_mynn_node_call *call=context;
    (void)d; (void)geometry; (void)geometry_bytes; (void)request; (void)bytes; (void)result; (void)result_bytes;
    call->committed=wt_mynn_batch_apply(call->batch)==WT_MYNN_OK;
}
enum wt_nl_status wt_mynn_node_collect(struct wt_mynn_node_context *c, struct wt_mynn_node_call *call)
{
    if (!owns(c,call)) return WT_NL_OWNER;
    return wt_nl_owner_collect(c->service,c->rank,c->thread,&call->ticket,validate,commit,call);
}
enum wt_nl_status wt_mynn_node_finish(struct wt_mynn_node_context *c, struct wt_mynn_node_call *call)
{
    enum wt_nl_status status;
    if (!owns(c,call) || !call->committed) return WT_NL_OWNER;
    status=wt_nl_finish(c->service,c->rank,c->thread,&call->ticket);
    if (status==WT_NL_OK) call->published=0;
    return status;
}
void wt_mynn_node_destroy(struct wt_mynn_node_call *call)
{
    if (!call) return;
    wt_mynn_batch_destroy(call->batch); memset(call,0,sizeof *call);
}
