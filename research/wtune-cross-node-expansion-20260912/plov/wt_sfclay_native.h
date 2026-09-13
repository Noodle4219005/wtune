#ifndef WT_SFCLAY_NATIVE_H
#define WT_SFCLAY_NATIVE_H

#include <stddef.h>
#include "wt_live_spans.h"

enum wt_sfclay_native_state {
    WT_SFCLAY_NATIVE_UNCONFIGURED = 0,
    WT_SFCLAY_NATIVE_INSTALLING = 1,
    WT_SFCLAY_NATIVE_IMMUTABLE = 2
};

/* The grid helpers are ordinary Fortran externals, not BIND(C).  The source,
 * caller-chain, initializer-relocation, compiler/module, and default-kind ABI
 * proofs must all belong to the same frozen build.  A successful configure is
 * immutable.  A rejected live installation rolls back to UNCONFIGURED.
 *
 * capture is safe to expose as a callback: before immutable publication it
 * invalidates the supplied span collection without invoking either Fortran
 * helper.  Once published it requires a positive native grid identity;
 * selector 0 admits every positive identity and selector >0 admits only that
 * identity.  Negative selectors are invalid configuration.
 */
int wt_sfclay_native_capture(void *grid, wt_live_spans *spans,
                             size_t max_nodes, int *num_tiles);
int wt_sfclay_native_configure(int enabled, int source_abi_verified,
    int caller_chain_verified, int initializer_relocation_verified,
    size_t max_spans, size_t max_nodes, size_t byte_budget,
    int grid_selector);
int wt_sfclay_native_state(void);

#endif
