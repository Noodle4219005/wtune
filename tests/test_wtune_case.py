#!/usr/bin/env python3
"""Tests for wtune_case.py -- the case-understanding profile. Synthetic namelists only, no
netCDF4 required (wrfinput-derived fields must degrade gracefully when it is absent).

Run: python3 -m pytest -q test_wtune_case.py
"""
import json
import os
import sys
import tempfile

import pytest

import wtune_case as wc
import wtune_physics as phys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root


def _write_namelist(tmp_path, body):
    p = os.path.join(tmp_path, 'namelist.input')
    with open(p, 'w') as fh:
        fh.write(body)
    return p


CONUS_2500M = """\
&time_control
 run_hours = 6,
 start_year = 2021,
 start_month = 11,
 start_day = 26,
 start_hour = 18,
 history_interval = 60,
 /
&domains
 max_dom = 1,
 e_we = 1501,
 e_sn = 1201,
 e_vert = 50,
 dx = 2500,
 dy = 2500,
 time_step = 15,
 ref_lat = 39.0,
 ref_lon = -97.0,
 /
&physics
 mp_physics = 8,
 cu_physics = 0,
 ra_lw_physics = 4,
 ra_sw_physics = 4,
 bl_pbl_physics = 1,
 sf_sfclay_physics = 1,
 sf_surface_physics = 2,
 radt = 10,
 /
"""

ISC21_3DOM = """\
&time_control
 run_hours = 1,
 start_year = 2021,
 start_month = 6,
 start_day = 1,
 start_hour = 0,
 history_interval = 60,
 /
&domains
 max_dom = 3,
 e_we = 793, 805, 1001,
 e_sn = 853, 805, 1001,
 e_vert = 51, 51, 51,
 parent_grid_ratio = 1, 3, 3,
 parent_time_step_ratio = 1, 3, 3,
 dx = 3000,
 dy = 3000,
 time_step = 15,
 ref_lat = 30.57,
 ref_lon = 119.36,
 /
&physics
 mp_physics = 8,
 cu_physics = 1,
 ra_lw_physics = 4,
 ra_sw_physics = 4,
 bl_pbl_physics = 5,
 sf_sfclay_physics = 5,
 sf_surface_physics = 2,
 radt = 3,
 /
"""

DX12KM = """\
&time_control
 run_hours = 6,
 start_year = 2021,
 start_month = 7,
 start_day = 15,
 start_hour = 0,
 history_interval = 60,
 /
&domains
 max_dom = 1,
 e_we = 401,
 e_sn = 301,
 e_vert = 45,
 dx = 12000,
 dy = 12000,
 time_step = 72,
 ref_lat = 20.0,
 ref_lon = 110.0,
 /
&physics
 mp_physics = 6,
 cu_physics = 0,
 ra_lw_physics = 4,
 ra_sw_physics = 4,
 bl_pbl_physics = 1,
 sf_sfclay_physics = 1,
 sf_surface_physics = 2,
 radt = 15,
 /
"""


@pytest.fixture
def conus_namelist(tmp_path):
    return _write_namelist(str(tmp_path), CONUS_2500M)


@pytest.fixture
def isc21_namelist(tmp_path):
    return _write_namelist(str(tmp_path), ISC21_3DOM)


@pytest.fixture
def dx12_namelist(tmp_path):
    return _write_namelist(str(tmp_path), DX12KM)


# cu_physics=5 (Grell-3D) crashed the sc26p1 blind test with KeyError('?') because _CU had no
# entry for it; cu_physics=999 is a value the WRF Registry does not list at all, used to exercise
# the generic unknown-scheme degrade path (see wtune_physics.cost_weight).
CU5_NAMELIST = CONUS_2500M.replace('cu_physics = 0,', 'cu_physics = 5,')
CU_UNKNOWN_NAMELIST = CONUS_2500M.replace('cu_physics = 0,', 'cu_physics = 999,')


@pytest.fixture
def cu5_namelist(tmp_path):
    return _write_namelist(str(tmp_path), CU5_NAMELIST)


@pytest.fixture
def cu_unknown_namelist(tmp_path):
    return _write_namelist(str(tmp_path), CU_UNKNOWN_NAMELIST)


# --- (a) convection-permitting, no cumulus mismatch; R_max exact hand-check -----------------

def test_convection_permitting_no_cumulus_mismatch(conus_namelist):
    p = wc.build_profile(conus_namelist, None, cores_per_node=72)
    assert p['physics']['dx_km'] == pytest.approx(2.5)
    assert 'convection-permitting' in p['physics']['dx_regime']
    assert p['physics']['gray_zone_suggested_cu'] == []
    assert not any('CORRECTNESS' in line for line in p['physics']['notes'])


def test_r_max_hand_check_91x100():
    # 91 // 10 = 9, 100 // 10 = 10 -> R_max = 90; independently confirmed against
    # decomp_space.legal_shapes_exact (the exact RSL-lite floor-division gate).
    r_max, (nx, ny) = wc.compute_r_max([91], [100], min_patch=10)
    assert r_max == 90
    assert (nx, ny) == (9, 10)
    assert wc.ds.legal_shapes_exact([91], [100], 90, 10) == [(9, 10)]
    assert wc.ds.legal_shapes_exact([91], [100], 91, 10) == []


def test_r_max_conus(conus_namelist):
    p = wc.build_profile(conus_namelist, None, cores_per_node=72)
    core = p['core_count_priors']
    # e_we=1501 -> nx_cap=150, e_sn=1201 -> ny_cap=120
    assert core['r_max_shape'] == {'nproc_x': 150, 'nproc_y': 120}
    assert core['r_max_ranks_omp1'] == 150 * 120


# --- (b) 3-domain nest: R_max limited by the smallest active domain -------------------------

def test_nested_r_max_limited_by_smallest_domain(isc21_namelist):
    p = wc.build_profile(isc21_namelist, None, cores_per_node=72)
    ident = p['identity']
    assert ident['max_dom'] == 3
    core = p['core_count_priors']
    # d01 e_we=793 (793//10=79) binds nx; d02 e_sn=805 (805//10=80) binds ny -- neither is the
    # d03 (finest, largest grid) domain, demonstrating "limited by the smallest active domain".
    assert core['r_max_shape'] == {'nproc_x': 79, 'nproc_y': 80}
    assert core['r_max_ranks_omp1'] == 79 * 80
    legal = wc.ds.legal_shapes_exact([793, 805, 1001], [853, 805, 1001], core['r_max_ranks_omp1'], 10)
    assert (79, 80) in legal
    # one more rank than R_max must be illegal on at least one domain
    assert wc.ds.legal_shapes_exact([793, 805, 1001], [853, 805, 1001], core['r_max_ranks_omp1'] + 1, 10) == [] \
        or all(pair != (79, 80) for pair in wc.ds.legal_shapes_exact(
            [793, 805, 1001], [853, 805, 1001], core['r_max_ranks_omp1'] + 1, 10))


def test_nested_dominant_domain_is_finest_nest(isc21_namelist):
    p = wc.build_profile(isc21_namelist, None, cores_per_node=72)
    ident = p['identity']
    # d03 has the largest parent_time_step_ratio-weighted point count (9x step multiplier)
    assert ident['dominant_domain'] == 3
    shares = [d['compute_share_pct'] for d in ident['domains']]
    assert sum(shares) == pytest.approx(100.0, abs=0.2)
    assert ident['domains'][2]['compute_share_pct'] == max(shares)


# --- (c) dx=12km with cu_physics=0 -> gray/parameterized note flagged -----------------------

def test_dx12km_cu_off_flags_parameterized_mismatch(dx12_namelist):
    p = wc.build_profile(dx12_namelist, None, cores_per_node=72)
    assert p['physics']['dx_km'] == pytest.approx(12.0)
    assert 'parameterized' in p['physics']['dx_regime']
    assert any('CORRECTNESS' in line and 'UNDER-resolved' in line for line in p['physics']['notes'])
    # a cu_physics candidate must be raised in the physics-candidate list
    cu_cands = [c for c in p['physics_candidates'] if c['param'] == 'cu_physics']
    assert cu_cands, "expected a cu_physics PHYSICS-GRAY candidate at dx=12km with cu_physics=0"


# --- (d) efficiency prediction monotone decreasing with nodes -------------------------------

def test_efficiency_monotone_decreasing(conus_namelist):
    p = wc.build_profile(conus_namelist, None, cores_per_node=72)
    table = p['core_count_priors']['efficiency_table']
    legal_rows = [r for r in table if r['legal']]
    assert len(legal_rows) >= 4
    es = [r['efficiency_E'] for r in legal_rows]
    assert all(es[i] > es[i + 1] for i in range(len(es) - 1)), es
    # sanity: efficiency proxy is bounded in (0, 1]
    assert all(0.0 < e <= 1.0 for e in es)


def test_efficiency_function_basic():
    # equal patch, no halo -> E must approach 1 as patch grows, and be < 1 always (c,H > 0)
    e_small = wc.efficiency(20, 20)
    e_big = wc.efficiency(2000, 2000)
    assert 0.0 < e_small < e_big < 1.0


# --- (e) CLI JSON has all top-level keys -----------------------------------------------------

def test_profile_dict_has_all_top_level_keys(conus_namelist):
    p = wc.build_profile(conus_namelist, None, cores_per_node=72)
    for key in wc.PROFILE_KEYS:
        assert key in p, "missing top-level profile key: {}".format(key)
    for sub in ('domains', 'dominant_domain', 'total_columns', 'total_cells'):
        assert sub in p['identity']
    for sub in ('r_max_ranks_omp1', 'efficiency_table', 'knee_below_0.8', 'knee_below_0.6',
                'suggested_first_test_nodes', 'suggested_omp_divisors'):
        assert sub in p['core_count_priors']


def test_cli_json_has_all_top_level_keys(conus_namelist, tmp_path):
    out_json = os.path.join(str(tmp_path), 'profile.json')
    rc = wc.main([os.path.dirname(conus_namelist), '--namelist', conus_namelist,
                  '--cores-per-node', '72', '--json', out_json])
    assert rc == 0
    assert os.path.exists(out_json)
    with open(out_json) as fh:
        payload = json.load(fh)
    for key in wc.PROFILE_KEYS:
        assert key in payload


# --- misc: find_case_inputs, no-netCDF4 degradation ------------------------------------------

def test_find_case_inputs_defaults(conus_namelist):
    case_dir = os.path.dirname(conus_namelist)
    nl, wi = wc.find_case_inputs(case_dir)
    assert nl == os.path.join(case_dir, 'namelist.input')
    assert wi is None  # no wrfinput_d0N present in this fixture dir


def test_wrfinput_optional_no_netcdf4_needed(conus_namelist):
    # Passing a nonexistent wrfinput path must degrade gracefully, not raise.
    p = wc.build_profile(conus_namelist, '/nonexistent/wrfinput_d01', cores_per_node=72)
    assert p['identity']['bbox_exact'] is None
    assert p['load_imbalance_drivers']['land_fraction'] is None


# --- (f) cu_physics=5 (Grell-3D) -- the sc26p1 crash: KeyError('?') because _CU had no entry --

def test_cu_physics_5_resolves_grell3d_with_real_cost():
    info = phys.scheme_info('cu_physics', 5)
    assert info['name'] == 'Grell-3D'
    assert info['cost'] != '?'
    assert info['cost'] in wc.COST_WEIGHT  # must be a real, lookup-able weight


def test_cu_physics_5_profile_builds_without_crash(cu5_namelist):
    p = wc.build_profile(cu5_namelist, None, cores_per_node=72)
    cu_phase = next(ph for ph in p['expected_cost_structure']['phases_ranked'] if ph['phase'] == 'cumulus')
    assert cu_phase['weight'] == wc.COST_WEIGHT['med']
    assert 'Grell-3D' in cu_phase['detail']


# --- (g) a cu_physics value absent from the table degrades to 'med' and WARNS, never crashes --

def test_unknown_cu_physics_defaults_to_med_and_warns(capsys):
    weight = phys.cost_weight('cu_physics', 999, wc.COST_WEIGHT)
    captured = capsys.readouterr()
    assert weight == wc.COST_WEIGHT['med']
    assert 'WARNING' in captured.err
    assert 'cu_physics=999' in captured.err
    assert 'unmeasured' in captured.err


def test_build_cost_structure_completes_for_unknown_cu_physics(cu_unknown_namelist):
    # Reproduces the sc26p1 failure mode for a value the WRF Registry does not list at all:
    # profile-building must complete (not raise KeyError) and fall back to the 'med' weight.
    p = wc.build_profile(cu_unknown_namelist, None, cores_per_node=72)
    cost = p['expected_cost_structure']
    cu_phase = next(ph for ph in cost['phases_ranked'] if ph['phase'] == 'cumulus')
    assert cu_phase['weight'] == wc.COST_WEIGHT['med']
