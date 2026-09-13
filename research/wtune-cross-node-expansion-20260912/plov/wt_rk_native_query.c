#include "wt_rk_native_query.h"

#include <limits.h>

_Static_assert(sizeof(int) * CHAR_BIT == 32, "native query requires 32-bit C int");

#ifndef WT_RK_NATIVE_QUERY_F77
#define WT_RK_NATIVE_QUERY_F77 wt_rk_native_query_
#endif

enum { WT_RK_NATIVE_VALUE_COUNT = 17 };

extern void WT_RK_NATIVE_QUERY_F77(void *opaque_cfg, int *rk_step,
                                   int values[WT_RK_NATIVE_VALUE_COUNT]);

int wt_rk_native_query(void *opaque_cfg, int rk_step,
                       struct wt_rk_native_result *result)
{
    int values[WT_RK_NATIVE_VALUE_COUNT] = {0};

    if (!result) return 0;
    *result = (struct wt_rk_native_result){
        .status = WT_RK_NATIVE_NULL,
        .rk_ord = 0,
        .zadvect_implicit = 0,
        .h_sca_adv_order = 0,
        .v_sca_adv_order = 0,
        .cu_physics = 0,
        .diff_opt = 0,
        .bl_pbl_physics = 0,
        .time_step = 0,
        .specified = 0,
        .nested = 0,
        .periodic_x = 0,
        .polar = 0,
        .open_xs = 0,
        .open_xe = 0,
        .open_ys = 0,
        .open_ye = 0,
    };
    if (!opaque_cfg) return 0;

    WT_RK_NATIVE_QUERY_F77(opaque_cfg, &rk_step, values);
    result->status = values[0];
    result->rk_ord = values[1];
    result->zadvect_implicit = values[2];
    result->h_sca_adv_order = values[3];
    result->v_sca_adv_order = values[4];
    result->cu_physics = values[5];
    result->diff_opt = values[6];
    result->bl_pbl_physics = values[7];
    result->time_step = values[8];
    result->specified = values[9];
    result->nested = values[10];
    result->periodic_x = values[11];
    result->polar = values[12];
    result->open_xs = values[13];
    result->open_xe = values[14];
    result->open_ys = values[15];
    result->open_ye = values[16];
    return result->status == WT_RK_NATIVE_OK;
}
