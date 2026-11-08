"""Test for post/ivt.py's self-written IVT integral (decision 0057 update T3: "IVT 用合成的
等值場測試，數值等於解析解"). A constant q/u/v profile over [p0, p1] has a closed-form IVT:

    ivt_u = q * u * (p1 - p0) / g      ivt_v = q * v * (p1 - p0) / g

and numpy.trapz is EXACT (not merely close) for a piecewise-linear integrand, which a constant
function trivially is -- so ivt_components()/ivt_field() must reproduce the closed form to
floating-point precision, at any number of sample levels and regardless of level spacing.
"""
import numpy as np

from wtune.post.ivt import ivt_components, ivt_field


def test_ivt_field_matches_scalar_ivt_components_per_pixel():
    levels = (1000.0, 850.0, 700.0, 500.0)
    ny, nx = 3, 2
    q_levels = {lv: np.full((ny, nx), 0.01 * (lv / 1000.0)) for lv in levels}
    u_levels = {lv: np.full((ny, nx), 5.0) for lv in levels}
    v_levels = {lv: np.full((ny, nx), -2.0) for lv in levels}

    ivt_u_field, ivt_v_field, ivt_mag_field = ivt_field(q_levels, u_levels, v_levels, levels)

    q_prof = [q_levels[lv][0, 0] for lv in levels]
    u_prof = [u_levels[lv][0, 0] for lv in levels]
    v_prof = [v_levels[lv][0, 0] for lv in levels]
    p_pa = [lv * 100.0 for lv in levels]
    expected_u, expected_v, expected_mag = ivt_components(q_prof, u_prof, v_prof, p_pa)

    assert ivt_u_field.shape == (ny, nx)
    assert np.allclose(ivt_u_field, expected_u)
    assert np.allclose(ivt_v_field, expected_v)
    assert np.allclose(ivt_mag_field, expected_mag)


# -- below-ground (NaN) levels: 2026-09-29 lead review -- moisture_ivt_d02.png on the katrina
# P7 project came out blank over the storm itself, because 1000 hPa sits below the local
# surface pressure right at a low's own centre. wrf.interplevel() must be asked for
# missing=np.nan there (post/recipes/moisture.py); this module's integral must then treat
# that NaN as "integrate only the above-ground part of the column", not propagate NaN (or,
# before the fix, wrf-python's own huge finite sentinel) through the whole result.
# ------------------------------------------------------------------------------------------
