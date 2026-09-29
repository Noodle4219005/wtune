"""Test for post/ivt.py's self-written IVT integral (decision 0057 update T3: "IVT 用合成的
等值場測試，數值等於解析解"). A constant q/u/v profile over [p0, p1] has a closed-form IVT:

    ivt_u = q * u * (p1 - p0) / g      ivt_v = q * v * (p1 - p0) / g

and numpy.trapz is EXACT (not merely close) for a piecewise-linear integrand, which a constant
function trivially is -- so ivt_components()/ivt_field() must reproduce the closed form to
floating-point precision, at any number of sample levels and regardless of level spacing.
"""
import numpy as np

from post.ivt import G, ivt_components, ivt_field


def test_ivt_components_matches_analytic_solution_for_constant_profile():
    q, u, v = 0.01, 10.0, 5.0          # kg/kg, m/s, m/s
    p0, p1 = 700e2, 1000e2             # Pa (700 hPa .. 1000 hPa)
    n_levels = 7
    pressure_pa = np.linspace(p0, p1, n_levels)

    ivt_u, ivt_v, ivt_mag = ivt_components([q] * n_levels, [u] * n_levels, [v] * n_levels,
                                           pressure_pa)

    expected_u = q * u * (p1 - p0) / G
    expected_v = q * v * (p1 - p0) / G
    assert np.isclose(ivt_u, expected_u, rtol=1e-12)
    assert np.isclose(ivt_v, expected_v, rtol=1e-12)
    assert np.isclose(ivt_mag, (expected_u ** 2 + expected_v ** 2) ** 0.5, rtol=1e-12)


def test_ivt_components_is_independent_of_level_order():
    q, u, v = 0.005, -3.0, 8.0
    p0, p1 = 300e2, 1000e2
    pressure_pa = np.linspace(p0, p1, 5)
    forward = ivt_components([q] * 5, [u] * 5, [v] * 5, pressure_pa)
    reversed_ = ivt_components([q] * 5, [u] * 5, [v] * 5, pressure_pa[::-1])
    assert np.allclose(forward, reversed_)


def test_ivt_components_uneven_spacing_still_exact_for_constant_profile():
    q, u, v = 0.02, 4.0, -2.0
    pressure_pa = np.array([1000e2, 950e2, 900e2, 700e2, 300e2])  # uneven, descending
    ivt_u, ivt_v, _ = ivt_components([q] * 5, [u] * 5, [v] * 5, pressure_pa)
    p0, p1 = pressure_pa.min(), pressure_pa.max()
    assert np.isclose(ivt_u, q * u * (p1 - p0) / G, rtol=1e-12)
    assert np.isclose(ivt_v, q * v * (p1 - p0) / G, rtol=1e-12)


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

def test_ivt_components_skips_below_ground_nan_levels():
    """1000 hPa and 925 hPa are "underground" (NaN) here; 850..300 hPa are valid. The result
    must equal the analytic solution over the above-ground range [300, 850] hPa only, not NaN
    and not the full [300, 1000] hPa range."""
    q, u, v = 0.015, 8.0, -4.0
    levels_hpa = [1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0]
    pressure_pa = np.array([lv * 100.0 for lv in levels_hpa])
    q_prof = [np.nan, np.nan, q, q, q, q, q]
    u_prof = [np.nan, np.nan, u, u, u, u, u]
    v_prof = [np.nan, np.nan, v, v, v, v, v]

    ivt_u, ivt_v, ivt_mag = ivt_components(q_prof, u_prof, v_prof, pressure_pa)

    p_top, p_lowest_valid = 300e2, 850e2
    expected_u = q * u * (p_lowest_valid - p_top) / G
    expected_v = q * v * (p_lowest_valid - p_top) / G
    assert not np.isnan(ivt_u) and not np.isnan(ivt_v) and not np.isnan(ivt_mag)
    assert np.isclose(ivt_u, expected_u, rtol=1e-12)
    assert np.isclose(ivt_v, expected_v, rtol=1e-12)
    assert np.isclose(ivt_mag, (expected_u ** 2 + expected_v ** 2) ** 0.5, rtol=1e-12)


def test_ivt_components_returns_nan_when_entire_profile_is_invalid():
    """A column with no valid segment at all (e.g. terrain above every requested level) must
    report NaN, not a false zero that would look like "no moisture transport" on a map."""
    n = 4
    nans = [float("nan")] * n
    pressure_pa = np.linspace(300e2, 1000e2, n)
    ivt_u, ivt_v, ivt_mag = ivt_components(nans, nans, nans, pressure_pa)
    assert np.isnan(ivt_u) and np.isnan(ivt_v) and np.isnan(ivt_mag)


def test_ivt_field_storm_centre_pixel_still_covered_by_below_ground_level():
    """Reproduces the actual bug against a gridded field: one pixel (the "storm centre") has
    its bottom (1000 hPa) level NaN while its neighbours are fully valid. The storm-centre
    pixel must come out finite and equal to the analytic solution over its own shorter valid
    range -- the IVT field must cover the storm, not show a hole there."""
    levels = (1000.0, 850.0, 700.0, 500.0)
    q, u, v = 0.012, 6.0, 3.0
    ny, nx = 2, 2
    q_levels, u_levels, v_levels = {}, {}, {}
    for lv in levels:
        q_levels[lv] = np.full((ny, nx), q)
        u_levels[lv] = np.full((ny, nx), u)
        v_levels[lv] = np.full((ny, nx), v)
    q_levels[1000.0][0, 0] = np.nan   # storm centre: 1000 hPa is locally below ground
    u_levels[1000.0][0, 0] = np.nan
    v_levels[1000.0][0, 0] = np.nan

    ivt_u, ivt_v, ivt_mag = ivt_field(q_levels, u_levels, v_levels, levels)

    assert not np.isnan(ivt_u[0, 0]) and not np.isnan(ivt_mag[0, 0])
    expected_u_storm = q * u * (850e2 - 500e2) / G
    expected_u_clear = q * u * (1000e2 - 500e2) / G
    assert np.isclose(ivt_u[0, 0], expected_u_storm, rtol=1e-12)
    assert np.isclose(ivt_u[1, 1], expected_u_clear, rtol=1e-12)
    assert ivt_u[0, 0] != ivt_u[1, 1]   # the storm pixel's column is genuinely shorter
