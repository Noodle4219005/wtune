#ifndef WT_MYNN_LAYOUT_H
#define WT_MYNN_LAYOUT_H

#include <stddef.h>
#include <stdint.h>
#include "wt_mynn_schema.h"

enum wt_mynn_layout_status {
    WT_MYNN_LAYOUT_OK = 0, WT_MYNN_LAYOUT_DISABLED, WT_MYNN_LAYOUT_ABI,
    WT_MYNN_LAYOUT_CONFIG, WT_MYNN_LAYOUT_ARGUMENT, WT_MYNN_LAYOUT_CAPACITY,
    WT_MYNN_LAYOUT_LOGICAL, WT_MYNN_LAYOUT_BOUNDS, WT_MYNN_LAYOUT_BRANCH,
    WT_MYNN_LAYOUT_OVERFLOW, WT_MYNN_LAYOUT_BUDGET
};

struct wt_mynn_layout_config {
    int enabled, abi_verified;             /* exact 0/1; no inference from source */
    uint32_t logical_true_bits;            /* attested 1 or UINT32_MAX; false is 0 */
    size_t byte_budget;                    /* 0: no limit on checked layout sum */
};

struct wt_mynn_branches {
    int32_t initflag, tke_budget, bl_mynn_edmf, bl_mynn_output, spp_pbl;
    unsigned char restart, cycling, tkeadvect;
    unsigned char qc, qi, qs, qnc, qni, qnwfa, qnifa, qnbca;
};

struct wt_mynn_layout {
    size_t required_bytes[WT_MYNN_NARG], total_bytes;
    unsigned char present[WT_MYNN_NARG], active[WT_MYNN_NARG], writable[WT_MYNN_NARG];
    uint32_t scalar_bits[WT_MYNN_NARG];     /* exact bits for every scalar; arrays 0 */
    struct wt_mynn_branches branches;
    int ready;
};

/* Allocation-free descriptor only; this is NOT transaction/native admission.
 * Caller supplies stable 132-entry arrays and truthful live byte capacities.
 * No allocation, callback, native invocation, owner write, or alias resolution.
 * Caller must exclude concurrent owner/argv/capacity/config changes throughout.
 * Independent ABI proof covers 32-bit REAL/INTEGER/LOGICAL, references, optional
 * representation and the selected compiled endpoint; source hash alone is not it.
 * All present actuals need aligned, nonwrapping >=4-byte intervals before ANY
 * scalar is read. C cannot verify that a numerically valid interval is allocated.
 * Active arrays require full formal extents; inactive present optionals retain a
 * 4-byte guard view and are not writable. Absent optional is NULL/capacity 0.
 * The guard bytes are a future clone requirement; layout never dereferences arrays.
 * total_bytes is a checked, non-deduplicated sum of formal views. Future transaction
 * budget must separately include metadata, snapshots, guards, clones and aliases.
 * On every failure *out is all-zero (when non-NULL); only success sets ready=1.
 * out must not overlap config, argv, capacities or any supplied owner interval.
 */
enum wt_mynn_layout_status wt_mynn_layout_build(
    void *const argv[WT_MYNN_NARG], const size_t capacities[WT_MYNN_NARG],
    const struct wt_mynn_layout_config *config, struct wt_mynn_layout *out);

#endif
