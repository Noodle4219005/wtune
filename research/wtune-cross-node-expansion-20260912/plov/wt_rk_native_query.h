#ifndef WT_RK_NATIVE_QUERY_H
#define WT_RK_NATIVE_QUERY_H

enum wt_rk_native_status {
    WT_RK_NATIVE_OK = 0,
    WT_RK_NATIVE_NULL = 1,
    WT_RK_NATIVE_KIND = 2,
    WT_RK_NATIVE_STEP = 3
};

struct wt_rk_native_result {
    int status;
    int rk_ord;
    int zadvect_implicit;
    int h_sca_adv_order;
    int v_sca_adv_order;
    int cu_physics;
    int diff_opt;
    int bl_pbl_physics;
    int time_step;
    int specified;
    int nested;
    int periodic_x;
    int polar;
    int open_xs;
    int open_xe;
    int open_ys;
    int open_ye;
};

/* Read a native grid_config_rec_type through a matching ordinary-Fortran
 * helper.  C treats opaque_cfg solely as an address: it never copies, sizes,
 * decodes, or otherwise assumes the layout of the derived type.  The caller
 * owns the synchronous native configuration lifetime.  This is selector
 * observation only; it does not stage data, change activation, or do halos.
 *
 * The helper uses native default INTEGER/LOGICAL/REAL before exporting c_int
 * scalar values.  Link it only with matching WRF module/compiler/default-kind
 * ABI.  WT_RK_NATIVE_QUERY_F77 can override the trailing-underscore symbol.
 */
int wt_rk_native_query(void *opaque_cfg, int rk_step,
                       struct wt_rk_native_result *result);

#endif
