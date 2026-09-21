#ifndef WT_MYNN_BATCH_H
#define WT_MYNN_BATCH_H

#include "wt_mynn.h"
#include <stdint.h>

/* A wire transaction is one complete original I row at one J index, with all
 * KMS:KME levels for 3-D actuals.  Execute retains all original formal scalar
 * bounds and makes full-formal private backings, placing only that row at its
 * original stride offset.  This is a
 * bounded transport adapter, not evidence that a production MYNN endpoint is
 * eligible for node-local execution. */
#define WT_MYNN_BATCH_SCHEMA UINT64_C(0x4d594e4e524f5731)

struct wt_mynn_batch;
typedef int (*wt_mynn_batch_callback)(void *context, void *argv[WT_MYNN_NARG]);

/* Requires layout ABI verification, KTS=1, TKE_BUDGET=0, SPP_PBL=0, and a
 * continuation call (INITFLAG=0).  Present inactive arrays carry one 4-byte
 * dummy.  Any overlap among the full formal owner views is rejected. */
enum wt_mynn_status wt_mynn_batch_prepare(void *argv[WT_MYNN_NARG],
    const size_t capacities[WT_MYNN_NARG],
    const struct wt_mynn_layout_config *layout_config, size_t byte_budget,
    struct wt_mynn_batch **out);
const void *wt_mynn_batch_request(const struct wt_mynn_batch *, size_t *bytes);
/* Copies the original IDS:KTE scalar geometry used to allocate the private
 * helper backings.  It is available only while the batch remains prepared. */
int wt_mynn_batch_geometry(const struct wt_mynn_batch *, int32_t out[18]);

/* Calls callback once after reconstructing a private one-J external-driver
 * argument vector.  A callback rejection is reported before result admission;
 * a callback that returns after mutation is never replayed by this API. */
enum wt_mynn_status wt_mynn_batch_execute(const void *request, size_t bytes,
    void *result, size_t result_capacity, size_t byte_budget,
    wt_mynn_batch_callback callback, void *context, int *invoked);
enum wt_mynn_status wt_mynn_batch_validate(struct wt_mynn_batch *,
    const void *result, size_t bytes);
enum wt_mynn_status wt_mynn_batch_apply(struct wt_mynn_batch *);
enum wt_mynn_status wt_mynn_batch_commit(struct wt_mynn_batch *,
    const void *result, size_t bytes);
void wt_mynn_batch_destroy(struct wt_mynn_batch *);

#endif
