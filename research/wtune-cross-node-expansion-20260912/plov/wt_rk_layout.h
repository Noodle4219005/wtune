#ifndef WT_RK_LAYOUT_H
#define WT_RK_LAYOUT_H
#include <stddef.h>
#include <stdint.h>
#include "wt_rk_schema.h"
#include "wt_rk_native_query.h"
enum wt_rk_layout_status { WT_RK_LAYOUT_OK=0, WT_RK_LAYOUT_DISABLED, WT_RK_LAYOUT_ABI,
 WT_RK_LAYOUT_CONFIG, WT_RK_LAYOUT_ARGUMENT, WT_RK_LAYOUT_CAPACITY, WT_RK_LAYOUT_LOGICAL,
 WT_RK_LAYOUT_BOUNDS, WT_RK_LAYOUT_BRANCH, WT_RK_LAYOUT_OVERFLOW, WT_RK_LAYOUT_BUDGET, WT_RK_LAYOUT_QUERY };
struct wt_rk_layout_config {
 int enabled, source_abi_verified, caller_closure_verified, config_read_lease;
 int actual_omp_in_parallel, exclusive_owner, allocation_descriptor_present;
 uint32_t logical_true_bits; size_t byte_budget; const void *config_identity;
 /* Two distinct, stable observations under the caller's config read lease. */
 const struct wt_rk_native_result *query_before, *query_after;
};
struct wt_rk_layout { size_t required_bytes[WT_RK_NARG], total_bytes; uint32_t scalar_bits[WT_RK_NARG];
 unsigned char present[WT_RK_NARG], writable[WT_RK_NARG], ready; struct wt_rk_native_result selectors; };
/* out is a valid writable layout object and must not overlap config, argv,
 * capacities, either query snapshot, or any supplied intrinsic owner. */
enum wt_rk_layout_status wt_rk_layout_build(void *const argv[WT_RK_NARG], const size_t capacities[WT_RK_NARG],
 const struct wt_rk_layout_config *config, struct wt_rk_layout *out);
int wt_rk_selector_equal(const struct wt_rk_native_result *, const struct wt_rk_native_result *);
#endif
