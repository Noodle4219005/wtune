#ifndef WT_KF_NATIVE_H
#define WT_KF_NATIVE_H
#include <stddef.h>
#include "wt_live_spans.h"

/* Native adapter trust boundary: compile Fortran against the frozen WRF .mod
 * files with the matching compiler and kind options; link the same OpenMP
 * runtime used by WRF. Unit tests with a fake module do not verify that ABI.
 * Override WT_GRID_CAPTURE_F77 when that compiler mangles the ordinary
 * Fortran external differently (the default symbol is wt_grid_capture_).
 * Capture returns 1 only for a complete list and positive actual num_tiles.
 * Begin/finish/end of the registry remain the live runtime's responsibility. */
int wt_kf_capture_grid(void *grid, wt_live_spans *registry, size_t max_nodes,
                       int *num_tiles);
/* No constructor or environment activation. Both attestation flags are
 * explicit build-evidence assertions; caller_chain_verified requires the
 * complete caller objects/ABI/CPS relocations, not a CPS-only source hash.
 * The live runtime copies this configuration once while quiescent. */
int wt_kf_native_configure(int enabled, int source_abi_verified,
                           int caller_chain_verified, size_t max_spans,
                           size_t max_nodes, size_t byte_budget);
/* selector 0 captures all grids; positive selectors require an equal positive
 * native grid%id. Negative selectors and duplicate configurations are rejected
 * without changing the active filter. WT_GRID_ID_F77 overrides the ordinary
 * identity helper symbol (default wt_grid_identity_). */
int wt_kf_native_configure_for_grid(int enabled, int source_abi_verified,
                                    int caller_chain_verified, size_t max_spans,
                                    size_t max_nodes, size_t byte_budget,
                                    int grid_selector);
#endif
