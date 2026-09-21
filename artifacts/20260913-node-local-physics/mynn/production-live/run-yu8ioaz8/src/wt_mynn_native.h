#ifndef WT_MYNN_NATIVE_H
#define WT_MYNN_NATIVE_H
#include "wt_mynn_node_live.h"
int wt_mynn_native_configure(struct wt_mynn_node_live_config *);
/* All scalar types are source-proven four-byte inputs. Active arrays require
 * registered live spans, except the three proven first-RK scratch arguments. */
int wt_mynn_native_capacities(void *const argv[WT_MYNN_NARG],
    const wt_live_spans *, uint64_t generation, const struct wt_mynn_node_live_config *,
    size_t capacities[WT_MYNN_NARG]);
#endif
