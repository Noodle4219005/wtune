#ifndef WT_SFCLAY_NATIVE_QUERY_H
#define WT_SFCLAY_NATIVE_QUERY_H

enum wt_sfclay_native_status {
    WT_SFCLAY_NATIVE_OK = 0,
    WT_SFCLAY_NATIVE_NULL = 1,
    WT_SFCLAY_NATIVE_KIND = 2,
    WT_SFCLAY_NATIVE_GRID_ID = 3,
    WT_SFCLAY_NATIVE_CONFIG_MISMATCH = 4,
    WT_SFCLAY_NATIVE_SCHEME = 5,
    WT_SFCLAY_NATIVE_FRACTIONAL = 6
};

struct wt_sfclay_native_result {
    int grid_id;
    int actual_scheme;
    int actual_fractional;
    int expected_scheme;
    int status;
};

/* Returns 1 only for matching live config/getters, compiled SFCLAYSCHEME and
 * fractional_seaice=0. Caller owns live native grid/config pointers for this
 * synchronous query; no C layout view, caching, environment or activation.
 * Output fields are initialized on all calls with a non-null result pointer.
 *
 * The ordinary Fortran helper has two native derived-type references and five
 * integer(c_int) output references. Link only a shim compiled with the matching
 * WRF .mod/compiler/default-kind ABI. Unit fixtures do not establish that ABI.
 * WT_SFCLAY_NATIVE_QUERY_F77 can override its default trailing-underscore symbol.
 */
int wt_sfclay_native_query(void *grid, void *cfg,
                          struct wt_sfclay_native_result *result);

#endif
