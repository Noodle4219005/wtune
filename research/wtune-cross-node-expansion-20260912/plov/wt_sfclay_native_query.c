#include "wt_sfclay_native_query.h"
#include <limits.h>

_Static_assert(sizeof(int) * CHAR_BIT == 32, "native query requires 32-bit C int");

#ifndef WT_SFCLAY_NATIVE_QUERY_F77
#define WT_SFCLAY_NATIVE_QUERY_F77 wt_sfclay_native_query_
#endif

extern void WT_SFCLAY_NATIVE_QUERY_F77(void *grid, void *cfg, int *grid_id,
                                     int *actual_scheme, int *actual_fractional,
                                     int *expected_scheme, int *status);

int wt_sfclay_native_query(void *grid, void *cfg,
                          struct wt_sfclay_native_result *result)
{
    if (!result) return 0;
    *result = (struct wt_sfclay_native_result){0, 0, 0, 0, WT_SFCLAY_NATIVE_NULL};
    if (!grid || !cfg) return 0;
    WT_SFCLAY_NATIVE_QUERY_F77(grid, cfg, &result->grid_id,
                              &result->actual_scheme, &result->actual_fractional,
                              &result->expected_scheme, &result->status);
    return result->status == WT_SFCLAY_NATIVE_OK;
}
