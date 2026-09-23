#!/usr/bin/env python3
"""Offline tests for tools/metdata.py.

No real network access: HTTP-touching functions (_http_head/_http_get_range_status/
_http_get_text/_http_download) are monkeypatched directly (unittest.mock.patch.object on the
already-imported metdata module -- this works even though `requests` itself is not installed in
this test environment, since we never import `requests`, only replace metdata's own thin
wrappers). Covers: date-rule source selection, URL construction (nomads/aws/rda/local), .idx
parsing + byte-range math, check_url against fake HTTP responses, fetch_message/fetch_idx_message
against a fake .idx + fake byte-range download, bbox crop, the APCP f000->f003 trap, and the
WPS/state field-spec tables. GRIB decoding (open_grib_field) is not exercised here -- see
tools/test_grib_state.py's own skip-if-missing pattern.

Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q tools/test_metdata.py
"""
import datetime
import inspect
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metdata as md   # noqa: E402


# ---------------------------------------------------------------------------
# Source selection (date rules, injectable `now`)
# ---------------------------------------------------------------------------

class TestSelectSource(unittest.TestCase):
    NOW = datetime.datetime(2026, 9, 22, 12, 0, 0)

    def test_recent_uses_nomads(self):
        dt = self.NOW - datetime.timedelta(days=2)
        src, reason = md.select_source(dt, now=self.NOW)
        self.assertEqual(src, 'nomads')
        self.assertIn('NOMADS', reason)

    def test_just_outside_nomads_window_uses_aws(self):
        dt = self.NOW - datetime.timedelta(days=15)
        src, reason = md.select_source(dt, now=self.NOW)
        self.assertEqual(src, 'aws')

    def test_2022_uses_aws(self):
        dt = datetime.datetime(2022, 6, 1)
        src, reason = md.select_source(dt, now=self.NOW)
        self.assertEqual(src, 'aws')

    def test_2019_uses_rda(self):
        dt = datetime.datetime(2019, 1, 30)
        src, reason = md.select_source(dt, now=self.NOW)
        self.assertEqual(src, 'rda')

    def test_before_rda_cutover_no_grib_dir_returns_none_with_alternatives(self):
        dt = datetime.datetime(2010, 1, 1)
        src, reason = md.select_source(dt, now=self.NOW)
        self.assertIsNone(src)
        self.assertIn('ds083.2', reason)
        self.assertIn('ds633.0', reason)
        self.assertIn('ds608.0', reason)

    def test_before_rda_cutover_with_grib_dir_uses_local(self):
        dt = datetime.datetime(2010, 1, 1)
        src, reason = md.select_source(dt, now=self.NOW, grib_dir='/nfs/data/wrf/sc26-p1/grib')
        self.assertEqual(src, 'local')
        self.assertIn('/nfs/data/wrf/sc26-p1/grib', reason)

    def test_explicit_override_wins_over_everything(self):
        dt = self.NOW - datetime.timedelta(days=1)   # would otherwise be nomads
        src, reason = md.select_source(dt, now=self.NOW, override='rda')
        self.assertEqual(src, 'rda')
        self.assertIn('override', reason)

    def test_auto_is_not_treated_as_override(self):
        dt = datetime.datetime(2019, 1, 30)
        src, _reason = md.select_source(dt, now=self.NOW, override='auto')
        self.assertEqual(src, 'rda')

    def test_recent_grib_dir_does_not_force_local(self):
        # --grib-dir given but the date is recent -- explicit --source local required to force
        # local; select_source alone must not silently prefer local over nomads/aws/rda here.
        dt = self.NOW - datetime.timedelta(days=2)
        src, _reason = md.select_source(dt, now=self.NOW, grib_dir='/some/dir')
        self.assertEqual(src, 'nomads')


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

class TestUrlConstruction(unittest.TestCase):
    def test_aws_grib_url(self):
        dt = datetime.datetime(2021, 6, 1, 12)
        url = md.aws_grib_url(dt, 6)
        self.assertEqual(url, 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20210601/12/atmos/'
                               'gfs.t12z.pgrb2.0p25.f006')

    def test_aws_idx_url(self):
        dt = datetime.datetime(2021, 6, 1, 12)
        self.assertEqual(md.aws_idx_url(dt, 6), md.aws_grib_url(dt, 6) + '.idx')

    def test_nomads_grib_url_mirrors_aws_layout(self):
        dt = datetime.datetime(2026, 9, 20, 0)
        url = md.nomads_grib_url(dt, 3)
        self.assertEqual(url, 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
                               'gfs.20260920/00/atmos/gfs.t00z.pgrb2.0p25.f003')

    def test_rda_fileserver_url_default_host(self):
        dt = datetime.datetime(2019, 1, 30, 0)
        url = md.rda_fileserver_url(dt, 3)
        self.assertEqual(url, 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/'
                               '2019/20190130/gfs.0p25.2019013000.f003.grib2')

    def test_rda_fileserver_url_custom_host(self):
        dt = datetime.datetime(2019, 1, 30, 0)
        url = md.rda_fileserver_url(dt, 3, host='https://thredds.gdex.ucar.edu/thredds')
        self.assertTrue(url.startswith('https://thredds.gdex.ucar.edu/thredds/fileServer/'))

    def test_rda_catalog_url(self):
        dt = datetime.datetime(2019, 1, 30, 0)
        url = md.rda_catalog_url(dt)
        self.assertEqual(url, 'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d084001/'
                               '2019/20190130/catalog.html')

    def test_nam_local_path(self):
        dt = datetime.datetime(2019, 1, 30, 0)
        path = md.nam_local_path('/nfs/data/wrf/sc26-p1/grib', dt, 0)
        self.assertEqual(path, '/nfs/data/wrf/sc26-p1/grib/20190130.nam.t00z.awphys00.tm00.grib2')

    def test_narr_local_path(self):
        dt = datetime.datetime(2019, 1, 30, 6)
        path = md.narr_local_path('/nfs/data/wrf/sc26-p2/grib-narr', dt)
        self.assertEqual(path, '/nfs/data/wrf/sc26-p2/grib-narr/merged_AWIP32.2019013006.RS.grib2')

    def test_url_for_local_prefers_existing_candidate(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dt = datetime.datetime(2019, 1, 30, 0)
            narr_path = md.narr_local_path(d, dt)
            open(narr_path, 'w').close()
            got = md.url_for('local', dt, 0, grib_dir=d)
            self.assertEqual(got, narr_path)

    def test_gfs_cycle_for_exact_and_offset(self):
        cycle, fhour = md.gfs_cycle_for(datetime.datetime(2021, 6, 1, 15))
        self.assertEqual(cycle, datetime.datetime(2021, 6, 1, 12))
        self.assertEqual(fhour, 3)


# ---------------------------------------------------------------------------
# .idx parsing + byte-range selection
# ---------------------------------------------------------------------------

SAMPLE_IDX = """\
1:0:d=2021060112:PRMSL:mean sea level:anl:
2:1044169:d=2021060112:APCP:surface:0-3 hour acc fcst:
3:1602384:d=2021060112:TCDC:entire atmosphere:0-3 hour ave fcst:
4:1789233:d=2021060112:HPBL:surface:anl:
5:1900001:d=2021060112:UGRD:10 m above ground:anl:
6:2001111:d=2021060112:VGRD:10 m above ground:anl:
7:2050000:d=2021060112:SHTFL:surface:0-3 hour ave fcst:
"""


class TestIdxParsing(unittest.TestCase):
    def test_parse_and_find(self):
        entries = md.parse_idx(SAMPLE_IDX)
        self.assertEqual(len(entries), 7)
        e = md.find_message(entries, 'apcp')
        self.assertEqual(e['num'], 2)
        self.assertEqual(e['level'], 'surface')

    def test_byte_range_middle_and_last(self):
        entries = md.parse_idx(SAMPLE_IDX)
        e_mid = md.find_message(entries, 'TCDC')
        start, end = md.byte_range(entries, e_mid)
        self.assertEqual((start, end), (1602384, 1789233 - 1))
        e_last = md.find_message(entries, 'SHTFL')
        start2, end2 = md.byte_range(entries, e_last)
        self.assertEqual(start2, 2050000)
        self.assertIsNone(end2)


# ---------------------------------------------------------------------------
# check_url against fake HTTP
# ---------------------------------------------------------------------------

class TestCheckUrl(unittest.TestCase):
    def test_head_200_means_exists(self):
        with mock.patch.object(md, '_http_head', return_value=(200, {})):
            r = md.check_url('https://example.com/file.grib2')
        self.assertTrue(r['exists'])
        self.assertEqual(r['status'], 200)

    def test_head_404_means_missing(self):
        with mock.patch.object(md, '_http_head', return_value=(404, {})):
            r = md.check_url('https://example.com/missing.grib2')
        self.assertFalse(r['exists'])
        self.assertEqual(r['status'], 404)

    def test_head_transport_error_means_missing(self):
        def raiser(*a, **k):
            raise OSError('connection refused')
        with mock.patch.object(md, '_http_head', side_effect=raiser):
            r = md.check_url('https://example.com/x.grib2')
        self.assertFalse(r['exists'])
        self.assertIn('connection refused', r['detail'])

    def test_head_405_falls_back_to_ranged_get(self):
        with mock.patch.object(md, '_http_head', return_value=(405, {})), \
             mock.patch.object(md, '_http_get_range_status', return_value=(206, {})):
            r = md.check_url('https://example.com/x.grib2')
        self.assertTrue(r['exists'])
        self.assertEqual(r['status'], 206)

    def test_local_path_exists(self):
        r = md.check_url(__file__)
        self.assertTrue(r['exists'])
        self.assertEqual(r['status'], 'local')

    def test_local_path_missing(self):
        r = md.check_url('/definitely/not/a/real/path/xyz.grib2')
        self.assertFalse(r['exists'])


# ---------------------------------------------------------------------------
# fetch_idx_message / fetch_message against fake HTTP + a tmp cache dir
# ---------------------------------------------------------------------------

class TestFetchIdxMessage(unittest.TestCase):
    def test_fetch_idx_message_downloads_only_the_byte_range(self):
        import tempfile
        calls = {'idx': 0, 'download': []}

        def fake_get_text(url, timeout=60):
            calls['idx'] += 1
            return SAMPLE_IDX

        def fake_download(url, dest_path, range_header=None, timeout=120):
            calls['download'].append((url, range_header))
            with open(dest_path, 'wb') as fh:
                fh.write(b'FAKEGRIB')
            return dest_path

        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', side_effect=fake_get_text), \
                 mock.patch.object(md, '_http_download', side_effect=fake_download):
                dt = datetime.datetime(2021, 6, 1, 12)
                path = md.fetch_idx_message('aws', dt, 3, 'TCDC', 'entire atmosphere', cache_dir)
                self.assertTrue(os.path.exists(path))
                with open(path, 'rb') as fh:
                    self.assertEqual(fh.read(), b'FAKEGRIB')
                self.assertEqual(calls['idx'], 1)
                self.assertEqual(len(calls['download']), 1)
                _url, rng = calls['download'][0]
                self.assertEqual(rng, 'bytes=1602384-1789232')

                # second fetch of the SAME message must not re-download or re-fetch the .idx
                path2 = md.fetch_idx_message('aws', dt, 3, 'TCDC', 'entire atmosphere', cache_dir)
                self.assertEqual(path2, path)
                self.assertEqual(calls['idx'], 1)
                self.assertEqual(len(calls['download']), 1)

    def test_fetch_idx_message_missing_var_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', return_value=SAMPLE_IDX):
                dt = datetime.datetime(2021, 6, 1, 12)
                with self.assertRaises(RuntimeError):
                    md.fetch_idx_message('aws', dt, 3, 'CAPE', None, cache_dir)

    def test_fetch_message_dispatches_by_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, 'fetch_idx_message', return_value='/cache/msg.grib2') as m:
                dt = datetime.datetime(2021, 6, 1, 12)
                path = md.fetch_message('tcdc', dt, 3, 'aws', cache_dir)
                self.assertEqual(path, '/cache/msg.grib2')
                m.assert_called_once()

    def test_fetch_message_local_requires_grib_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache_dir:
            dt = datetime.datetime(2019, 1, 30, 0)
            with self.assertRaises(RuntimeError):
                md.fetch_message('tcdc', dt, 0, 'local', cache_dir, grib_dir=None)

    def test_fetch_local_finds_nam_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as grib_dir:
            dt = datetime.datetime(2019, 1, 30, 0)
            nam_path = md.nam_local_path(grib_dir, dt, 0)
            open(nam_path, 'w').close()
            got = md.fetch_local(grib_dir, dt, 0)
            self.assertEqual(got, nam_path)

    def test_fetch_local_raises_when_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as grib_dir:
            dt = datetime.datetime(2019, 1, 30, 0)
            with self.assertRaises(RuntimeError):
                md.fetch_local(grib_dir, dt, 0)


# ---------------------------------------------------------------------------
# bbox crop (pure numpy)
# ---------------------------------------------------------------------------

class TestCropBbox(unittest.TestCase):
    def test_crop_pm180_grid(self):
        lats = np.arange(30.0, 50.0, 1.0)
        lons = np.arange(-110.0, -80.0, 1.0)
        values = np.arange(lats.size * lons.size).reshape(lats.size, lons.size)
        clats, clons, cval = md.crop_bbox(lats, lons, values, 35.0, 40.0, -100.0, -95.0, margin_deg=1.0)
        self.assertTrue(clats.min() >= 34.0 and clats.max() <= 41.0)
        self.assertTrue(clons.min() >= -101.0 and clons.max() <= -94.0)
        self.assertEqual(cval.shape, (clats.size, clons.size))

    def test_crop_empty_raises(self):
        lats = np.arange(30.0, 40.0, 1.0)
        lons = np.arange(-110.0, -100.0, 1.0)
        values = np.zeros((lats.size, lons.size))
        with self.assertRaises(ValueError):
            md.crop_bbox(lats, lons, values, 60.0, 65.0, -100.0, -95.0, margin_deg=0.0)


# ---------------------------------------------------------------------------
# APCP f000 trap
# ---------------------------------------------------------------------------

class TestResolveApcpRequest(unittest.TestCase):
    def test_f000_substitutes_f003_with_warning(self):
        cycle = datetime.datetime(2021, 6, 1, 12)
        eff_cycle, eff_fhour, window, warn = md.resolve_apcp_request(cycle, 0)
        self.assertEqual(eff_cycle, cycle)
        self.assertEqual(eff_fhour, 3)
        self.assertEqual(window, 3.0)
        self.assertIsNotNone(warn)
        self.assertIn('f000', warn)

    def test_nonzero_multiple_of_3_no_warning(self):
        cycle = datetime.datetime(2021, 6, 1, 12)
        eff_cycle, eff_fhour, window, warn = md.resolve_apcp_request(cycle, 6)
        self.assertEqual(eff_fhour, 6)
        self.assertEqual(window, 3.0)
        self.assertIsNone(warn)


class TestResolveFieldRequest(unittest.TestCase):
    # Confirmed live (2026-09-21 real AWS .idx): APCP/ULWRF/SHTFL are absent from f000, TCDC/
    # HPBL/UGRD/VGRD are present at f000 with fcst='anl'. See FLUX_FIELDS_NO_F000.
    def test_flux_fields_substitute_f003_at_f000(self):
        cycle = datetime.datetime(2021, 6, 1, 12)
        for key in ('apcp', 'ulwrf', 'shtfl'):
            eff_cycle, eff_fhour, warn = md.resolve_field_request(key, cycle, 0)
            self.assertEqual(eff_cycle, cycle)
            self.assertEqual(eff_fhour, 3)
            self.assertIsNotNone(warn)
            self.assertIn(key.upper(), warn)

    def test_instantaneous_fields_unchanged_at_f000(self):
        cycle = datetime.datetime(2021, 6, 1, 12)
        for key in ('tcdc', 'hpbl', 'u10', 'v10'):
            eff_cycle, eff_fhour, warn = md.resolve_field_request(key, cycle, 0)
            self.assertEqual(eff_cycle, cycle)
            self.assertEqual(eff_fhour, 0)
            self.assertIsNone(warn)

    def test_any_field_unchanged_at_nonzero_fhour(self):
        cycle = datetime.datetime(2021, 6, 1, 12)
        for key in ('apcp', 'ulwrf', 'shtfl', 'tcdc', 'hpbl', 'u10', 'v10'):
            eff_cycle, eff_fhour, warn = md.resolve_field_request(key, cycle, 6)
            self.assertEqual(eff_fhour, 6)
            self.assertIsNone(warn)


# ---------------------------------------------------------------------------
# Field spec tables
# ---------------------------------------------------------------------------

class TestFieldSpecs(unittest.TestCase):
    def test_state_field_specs_cover_grib_state_needs(self):
        for key in ('apcp', 'tcdc', 'ulwrf', 'hpbl', 'u10', 'v10', 'shtfl'):
            self.assertIn(key, md.STATE_FIELD_SPECS)
            spec = md.STATE_FIELD_SPECS[key]
            self.assertIn('var', spec)
            self.assertIn('level', spec)
            self.assertIn('cfgrib_short', spec)

    def test_state_field_specs_cover_analog_needs(self):
        # R6-C (case_rebuild.py analog): HGT 500mb, TMP 850mb, UGRD/VGRD 850mb, PWAT, TMP 2m.
        expect = {
            'hgt500': ('HGT', '500 mb'), 'tmp850': ('TMP', '850 mb'),
            'ugrd850': ('UGRD', '850 mb'), 'vgrd850': ('VGRD', '850 mb'),
            'tmp2m': ('TMP', '2 m above ground'),
        }
        for key, (var, level) in expect.items():
            self.assertIn(key, md.STATE_FIELD_SPECS)
            self.assertEqual(md.STATE_FIELD_SPECS[key]['var'], var)
            self.assertEqual(md.STATE_FIELD_SPECS[key]['level'], level)
        self.assertEqual(md.STATE_FIELD_SPECS['pwat']['var'], 'PWAT')

    def test_wps_field_specs_nonempty_and_cover_plevels(self):
        self.assertGreater(len(md.WPS_FIELD_SPECS), 100)
        plevel_keys = [k for k in md.WPS_FIELD_SPECS if k.startswith('TMP_') and k.endswith('mb')]
        self.assertEqual(len(plevel_keys), len(md.GFS_PRESSURE_LEVELS_HPA))

    def test_wps_field_specs_cover_surface_set(self):
        surf_vars = {spec['var'] for spec in md.WPS_FIELD_SPECS.values() if 'mb' not in spec['level']}
        self.assertIn('PRMSL', surf_vars)
        self.assertIn('SOILW', surf_vars)

    def test_wps_field_specs_soil_entries_are_tsoil_m_not_tmp_cm(self):
        # 2026-09-24 fix: the old entries were ('TMP', '0-10 cm below ground')-style and matched
        # NOTHING in a real GFS file (severe2023: NUM_METGRID_SOIL_LEVELS ended up 0).
        soil_specs = [spec for spec in md.WPS_FIELD_SPECS.values() if 'below ground' in spec['level']]
        self.assertEqual(len(soil_specs), 8)
        for spec in soil_specs:
            self.assertIn(spec['var'], ('TSOIL', 'SOILW'))
            self.assertTrue(spec['level'].endswith(' m below ground'), spec['level'])
            self.assertNotIn('cm', spec['level'])
        self.assertEqual({spec['level'] for spec in soil_specs},
                         {'0-0.1 m below ground', '0.1-0.4 m below ground',
                          '0.4-1 m below ground', '1-2 m below ground'})


# ---------------------------------------------------------------------------
# select_wps_messages / validate_wps_selection (2026-09-24) -- idx-driven WPS message selection,
# replacing the hardcoded-list lookup for the aws/nomads per-message fetch path. Fixture built
# from the EXACT strings verified live (2023-03-31 00Z f000): 41 pressure levels (8 of them
# below 1 mb, which must be excluded), the surface set, TSOIL/SOILW soil layers in meters, plus
# a couple of irrelevant messages (SOILL, TCDC) that must NOT be selected.
# ---------------------------------------------------------------------------

_VERIFIED_PLEVELS_MB = (0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 3, 5, 7, 10, 15, 20,
                        30, 40, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
                        650, 700, 750, 800, 850, 900, 925, 950, 975, 1000)


def _fmt_mb(x):
    return ('%g' % x)


def _build_severe2023_idx_text():
    lines = []
    n = 1
    offset = 0
    def add(var, level, fcst='anl'):
        nonlocal n, offset
        lines.append('%d:%d:d=2023033100:%s:%s:%s:' % (n, offset, var, level, fcst))
        n += 1
        offset += 1000

    for var in ('HGT', 'TMP', 'RH', 'UGRD', 'VGRD'):
        for p in _VERIFIED_PLEVELS_MB:
            add(var, '%s mb' % _fmt_mb(p))
    for var, level in (('PRMSL', 'mean sea level'), ('PRES', 'surface'), ('HGT', 'surface'),
                       ('TMP', 'surface'), ('LAND', 'surface'), ('ICEC', 'surface'),
                       ('WEASD', 'surface'), ('SNOD', 'surface'),
                       ('TMP', '2 m above ground'), ('RH', '2 m above ground'),
                       ('UGRD', '10 m above ground'), ('VGRD', '10 m above ground')):
        add(var, level)
    for var in ('TSOIL', 'SOILW'):
        for layer in ('0-0.1 m below ground', '0.1-0.4 m below ground',
                     '0.4-1 m below ground', '1-2 m below ground'):
            add(var, layer)
    # irrelevant messages that must NOT be selected
    add('SOILL', '0-0.1 m below ground')
    add('TCDC', 'entire atmosphere')
    return '\n'.join(lines)


class TestSelectWpsMessages(unittest.TestCase):
    def setUp(self):
        self.entries = md.parse_idx(_build_severe2023_idx_text())
        self.selected = md.select_wps_messages(self.entries)

    def test_selected_count_matches_33_levels_x_5_vars_plus_surface_plus_soil(self):
        n_plevel = sum(1 for e in self.selected if e['var'].upper() in
                      ('HGT', 'TMP', 'RH', 'UGRD', 'VGRD') and e['level'].endswith('mb'))
        self.assertEqual(n_plevel, 33 * 5)
        self.assertEqual(len(self.selected), 33 * 5 + 12 + 8)

    def test_all_8_soil_messages_selected(self):
        soil = [(e['var'].upper(), e['level'].strip()) for e in self.selected
               if e['var'].upper() in ('TSOIL', 'SOILW')]
        self.assertEqual(len(soil), 8)
        for var in ('TSOIL', 'SOILW'):
            for layer in ('0-0.1 m below ground', '0.1-0.4 m below ground',
                         '0.4-1 m below ground', '1-2 m below ground'):
                self.assertIn((var, layer), soil)

    def test_sub_1mb_levels_excluded(self):
        selected_pairs = {(e['var'].upper(), e['level'].strip()) for e in self.selected}
        for p in (0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7):
            self.assertNotIn(('TMP', '%s mb' % _fmt_mb(p)), selected_pairs)

    def test_soill_and_tcdc_excluded(self):
        selected_vars = {e['var'].upper() for e in self.selected}
        self.assertNotIn('SOILL', selected_vars)
        self.assertNotIn('TCDC', selected_vars)


class TestValidateWpsSelection(unittest.TestCase):
    def test_complete_selection_passes(self):
        entries = md.parse_idx(_build_severe2023_idx_text())
        selected = md.select_wps_messages(entries)
        self.assertTrue(md.validate_wps_selection(selected))

    def test_missing_one_soil_layer_fails_and_lists_it(self):
        text = _build_severe2023_idx_text()
        # drop the TSOIL:1-2 m below ground line
        text_missing = '\n'.join(l for l in text.splitlines()
                                 if 'TSOIL:1-2 m below ground' not in l)
        entries = md.parse_idx(text_missing)
        selected = md.select_wps_messages(entries)
        with self.assertRaises(RuntimeError) as ctx:
            md.validate_wps_selection(selected)
        self.assertIn('TSOIL', str(ctx.exception))
        self.assertIn('1-2 m below ground', str(ctx.exception))

    def test_too_few_pressure_levels_fails(self):
        lines = []
        n = 1
        offset = 0
        for p in (1000, 900, 800):   # only 3 levels, well under the 20 minimum
            lines.append('%d:%d:d=2023033100:TMP:%s mb:anl:' % (n, offset, p))
            n += 1
            offset += 1000
        for var, level in (('LAND', 'surface'), ('TMP', 'surface'), ('PRES', 'surface')):
            lines.append('%d:%d:d=2023033100:%s:%s:anl:' % (n, offset, var, level))
            n += 1
            offset += 1000
        for var in ('TSOIL', 'SOILW'):
            for layer in ('0-0.1 m below ground', '0.1-0.4 m below ground',
                         '0.4-1 m below ground', '1-2 m below ground'):
                lines.append('%d:%d:d=2023033100:%s:%s:anl:' % (n, offset, var, layer))
                n += 1
                offset += 1000
        entries = md.parse_idx('\n'.join(lines))
        selected = md.select_wps_messages(entries)
        with self.assertRaises(RuntimeError) as ctx:
            md.validate_wps_selection(selected)
        self.assertIn('TMP', str(ctx.exception))
        self.assertIn('pressure levels', str(ctx.exception))


class TestFetchWpsMessagesIdxDriven(unittest.TestCase):
    def test_downloads_only_the_selected_messages_via_byte_range(self):
        import tempfile
        idx_text = _build_severe2023_idx_text()
        downloads = []

        def fake_download(url, dest_path, range_header=None, timeout=120):
            downloads.append(range_header)
            with open(dest_path, 'wb') as fh:
                fh.write(b'FAKEGRIB')
            return dest_path

        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', return_value=idx_text), \
                 mock.patch.object(md, '_http_download', side_effect=fake_download):
                dt = datetime.datetime(2023, 3, 31, 0)
                out = md.fetch_wps_messages_idx_driven(dt, 0, 'aws', cache_dir)
        self.assertEqual(len(out), 33 * 5 + 12 + 8)
        self.assertEqual(len(downloads), 33 * 5 + 12 + 8)   # one ranged GET per selected message
        selected_vars = {e['var'].upper() for e, _path in out}
        self.assertNotIn('SOILL', selected_vars)
        self.assertNotIn('TCDC', selected_vars)

    def test_incomplete_idx_raises_before_downloading(self):
        import tempfile
        text_missing = '\n'.join(l for l in _build_severe2023_idx_text().splitlines()
                                 if 'TSOIL:1-2 m below ground' not in l)
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', return_value=text_missing), \
                 mock.patch.object(md, '_http_download') as dl:
                dt = datetime.datetime(2023, 3, 31, 0)
                with self.assertRaises(RuntimeError) as ctx:
                    md.fetch_wps_messages_idx_driven(dt, 0, 'aws', cache_dir)
                dl.assert_not_called()
        self.assertIn('TSOIL', str(ctx.exception))


class TestDoFetchWpsIdxDriven(unittest.TestCase):
    def _wps_entry(self, source='aws'):
        return {
            'valid_time': datetime.datetime(2023, 3, 31, 0, 0, 0),
            'cycle': datetime.datetime(2023, 3, 31, 0, 0, 0), 'fhour': 0,
            'source': source, 'reason': 'test',
            'messages': [{'field': 'HGT_1000mb', 'var': 'HGT', 'level': '1000 mb', 'fhour': 0,
                         'url': 'https://example/x.grib2'}],
        }

    def test_wps_aws_routes_through_idx_driven_selection(self):
        import tempfile
        idx_text = _build_severe2023_idx_text()
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', return_value=idx_text), \
                 mock.patch.object(md, '_http_download',
                                   side_effect=lambda url, dest, **kw: open(dest, 'wb').write(b'x')):
                results = md.do_fetch([self._wps_entry()], cache_dir, False, None, None)
        ok = [r for r in results if r['status'] == 'ok']
        self.assertEqual(len(ok), 33 * 5 + 12 + 8)

    def test_wps_aws_incomplete_selection_reports_fail_not_ok(self):
        # This is the exact 2026-09-24 regression: the OLD static-spec path printed OK for
        # every message even when the real file couldn't support ungrib's soil-level needs.
        import tempfile
        text_missing = '\n'.join(l for l in _build_severe2023_idx_text().splitlines()
                                 if 'TSOIL:1-2 m below ground' not in l)
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, '_http_get_text', return_value=text_missing), \
                 mock.patch.object(md, '_http_download') as dl:
                results = md.do_fetch([self._wps_entry()], cache_dir, False, None, None)
            dl.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'error')
        self.assertIn('TSOIL', results[0]['detail'])

    def test_rda_source_unaffected_stays_on_whole_file_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(md, 'download_full', return_value='/cache/rda.grib2') as dl:
                results = md.do_fetch([self._wps_entry(source='rda')], cache_dir, False, None, None)
        self.assertEqual(results[0]['status'], 'ok')
        self.assertEqual(results[0]['path'], '/cache/rda.grib2')
        dl.assert_called_once()


# ---------------------------------------------------------------------------
# build_plan / find (offline: no network, uses a synthetic namelist)
# ---------------------------------------------------------------------------

class TestBuildPlan(unittest.TestCase):
    def _write_case(self, tmp_path, extra_domains=''):
        nl_path = os.path.join(tmp_path, 'namelist.input')
        with open(nl_path, 'w') as fh:
            fh.write("""\
&time_control
 run_hours = 6,
 start_year = 2019,
 start_month = 1,
 start_day = 30,
 start_hour = 0,
 /
&domains
 max_dom = 1,
 e_we = 21,
 e_sn = 21,
 e_vert = 50,
 dx = 3000.0,
 dy = 3000.0,
 ref_lat = 39.0,
 ref_lon = -97.0,
 %s
 /
&physics
 /
""" % extra_domains)
        return tmp_path

    def test_build_plan_state_purpose_uses_rda_for_2019_case(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            plan, warnings = md.build_plan(d, every=3.0,
                                            now=datetime.datetime(2026, 9, 22))
            self.assertGreaterEqual(len(plan), 3)   # 0,3,6h
            self.assertEqual(plan[0]['source'], 'rda')
            self.assertEqual(len(plan[0]['messages']), len(md.STATE_FIELD_SPECS))
            self.assertIn(md.W_SOUTH_NORTH, warnings)

    def test_build_plan_apcp_message_has_f003_substitution_at_t0(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            plan, _w = md.build_plan(d, every=3.0, now=datetime.datetime(2026, 9, 22))
            apcp_msg = next(m for m in plan[0]['messages'] if m['field'] == 'apcp')
            self.assertEqual(apcp_msg['fhour'], 3)
            self.assertIn('warning', apcp_msg)

    def test_build_plan_ulwrf_and_shtfl_also_substitute_f003_at_t0(self):
        # Confirmed live (2026-09-21 real AWS .idx): ULWRF/SHTFL are absent from f000 exactly
        # like APCP -- build_plan must apply the same substitution to all three, not just apcp.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            plan, _w = md.build_plan(d, every=3.0, now=datetime.datetime(2026, 9, 22))
            for key in ('ulwrf', 'shtfl'):
                msg = next(m for m in plan[0]['messages'] if m['field'] == key)
                self.assertEqual(msg['fhour'], 3)
                self.assertIn('warning', msg)
            for key in ('tcdc', 'hpbl', 'u10', 'v10'):
                msg = next(m for m in plan[0]['messages'] if m['field'] == key)
                self.assertEqual(msg['fhour'], 0)
                self.assertNotIn('warning', msg)

    def test_build_plan_wps_purpose_uses_default_6h_without_namelist_wps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            plan, warnings = md.build_plan(d, purpose='wps', now=datetime.datetime(2026, 9, 22))
            self.assertEqual(len(plan), 2)   # 0h, 6h over a 6h run at 6h interval
            self.assertEqual(len(plan[0]['messages']), len(md.WPS_FIELD_SPECS))
            self.assertIn(md.W_WPS_GFSDIR, warnings)

    def test_build_plan_wps_purpose_reads_namelist_wps_interval(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            with open(os.path.join(d, 'namelist.wps'), 'w') as fh:
                fh.write("&share\n interval_seconds = 10800,\n /\n")
            plan, _w = md.build_plan(d, purpose='wps', now=datetime.datetime(2026, 9, 22))
            self.assertEqual(len(plan), 3)   # 0,3,6h at 3h interval over a 6h run

    def test_build_plan_wps_purpose_lafe_shape_five_times(self):
        """Regression for the wtune.py:extract_case bug where run_total_hours came out 0 for
        the lafe shape (every run_* key is 0; length is expressed only via start_date
        (2017-08-23 06:00) -> end_date (2017-08-24 06:00) = 24h) -- 'metdata find --purpose wps'
        used to emit ONE GRIB time (0h only) instead of the FIVE (0,6,12,18,24h) a 24h case at
        the default 6h WPS interval actually needs; WPS would get an initial condition and no
        lateral boundary data. Fixture transcribed verbatim from the real cluster namelist
        (leading-zero months, per-domain comma lists, irregular spacing preserved -- see
        tools/test_temporal_validation.py's NML_LAFE_SHAPE for the same shape). No namelist.wps
        is provided, so resolve_wps_times falls back to its own 6h default interval_seconds,
        which happens to equal this template's own &time_control interval_seconds=21600."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            nl_path = os.path.join(d, 'namelist.input')
            with open(nl_path, 'w') as fh:
                fh.write("""\
 &time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2017, 2017, 2017,
 start_month                         = 08,   08,   08,
 start_day                           = 23,   23,   23,
 start_hour                          = 06,   06,   06,
 start_minute                        = 00,   00,   00,
 start_second                        = 00,   00,   00,
 end_year                            = 2017, 2017, 2017,
 end_month                           = 08,   08,   08,
 end_day                             = 24,   24,   24,
 end_hour                            = 06,   06,   06,
 end_minute                          = 00,   00,   00,
 end_second                          = 00,   00,   00,
 interval_seconds                    = 21600,
 /
 &domains
 max_dom                             = 1,
 e_we                                = 1000,
 e_sn                                = 1000,
 e_vert                              = 100,
 dx                                  = 2500,
 dy                                  = 2500,
 ref_lat                             = 36.21,
 ref_lon                             = -97.48,
 /
 &physics
 /
""")
            plan, _w = md.build_plan(d, purpose='wps', now=datetime.datetime(2026, 9, 22))
            self.assertEqual(len(plan), 5)   # 0,6,12,18,24h at the default 6h interval over 24h
            self.assertEqual(plan[0]['valid_time'], datetime.datetime(2017, 8, 23, 6))
            self.assertEqual(plan[-1]['valid_time'], datetime.datetime(2017, 8, 24, 6))

    def test_build_plan_too_old_returns_none_source_and_no_messages(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            nl_path = os.path.join(d, 'namelist.input')
            with open(nl_path, 'w') as fh:
                fh.write("""\
&time_control
 run_hours = 0,
 start_year = 2005,
 start_month = 1,
 start_day = 1,
 start_hour = 0,
 /
&domains
 max_dom = 1,
 e_we = 21,
 e_sn = 21,
 e_vert = 50,
 dx = 3000.0,
 dy = 3000.0,
 ref_lat = 39.0,
 ref_lon = -97.0,
 /
&physics
 /
""")
            plan, _w = md.build_plan(d, every=3.0, now=datetime.datetime(2026, 9, 22))
            self.assertIsNone(plan[0]['source'])
            self.assertEqual(plan[0]['messages'], [])

    def test_print_find_smoke(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_case(d)
            plan, warnings = md.build_plan(d, every=3.0, now=datetime.datetime(2026, 9, 22))
            md.print_find(plan, warnings)   # must not raise


# ---------------------------------------------------------------------------
# Forcing source fallback chain (2026-09-24) -- resolve_forcing_chain date bands, verified URL
# patterns, FNL grib1/grib2 cutover, CFSR tar spans. All pure/offline (date rules + string
# formatting only -- no network).
# ---------------------------------------------------------------------------

class TestForcingChainDateBands(unittest.TestCase):
    NOW = datetime.datetime(2026, 9, 24, 12, 0, 0)

    def _datasets(self, dt):
        return [s.dataset for s in md.resolve_forcing_chain(dt, now=self.NOW)]

    def test_2023_03_31_aws_then_d084001_then_fnl(self):
        # R4 (2026-09-24): FNL (d083002) is a legitimate last-resort fallback here too -- its
        # THREDDS catalog runs to the present, not just back to 1999-07-30.
        self.assertEqual(self._datasets(datetime.datetime(2023, 3, 31)),
                         ['gfs_aws', 'd084001', 'd083002'])

    def test_2019_01_28_d084001_then_fnl(self):
        self.assertEqual(self._datasets(datetime.datetime(2019, 1, 28)),
                         ['d084001', 'd083002'])

    def test_2010_01_01_ncei_g4_then_fnl(self):
        self.assertEqual(self._datasets(datetime.datetime(2010, 1, 1)),
                         ['ncei_g4', 'd083002'])

    def test_2005_01_01_ncei_g3_then_fnl(self):
        self.assertEqual(self._datasets(datetime.datetime(2005, 1, 1)),
                         ['ncei_g3', 'd083002'])

    def test_2001_06_01_fnl_only(self):
        self.assertEqual(self._datasets(datetime.datetime(2001, 6, 1)), ['d083002'])

    def test_1990_01_03_cfsr_reanalysis_fallback(self):
        chain = md.resolve_forcing_chain(datetime.datetime(1990, 1, 3), now=self.NOW)
        self.assertEqual([s.dataset for s in chain], ['d093000'])
        self.assertEqual(chain[0].source_kind, 'reanalysis')

    def test_1975_01_01_empty_no_grib_dir(self):
        self.assertEqual(md.resolve_forcing_chain(datetime.datetime(1975, 1, 1), now=self.NOW), [])

    def test_1975_01_01_local_fallback_with_grib_dir(self):
        chain = md.resolve_forcing_chain(datetime.datetime(1975, 1, 1), now=self.NOW,
                                         grib_dir='/some/grib/dir')
        self.assertEqual([s.dataset for s in chain], ['local'])


# ---------------------------------------------------------------------------
# T13 (2026-09-24, V5 sc26p2): ncei_g3/g4 + FNL use Vtable.GFS_PRMSL (PRMSL, not MSLET) --
# real.exe FATAL'd 'ERROR in psfc' on a 2005 ncei_g3 GRIB1 file whose only sea-level-pressure
# message was PRMSL, which stock Vtable.GFS's active MSLET-only mapping never picks up.
# ---------------------------------------------------------------------------

class TestSourceVtableMapping(unittest.TestCase):
    def test_ncei_and_fnl_use_prmsl_vtable(self):
        for src in (md.SRC_NCEI_G3, md.SRC_NCEI_G4, md.SRC_FNL):
            self.assertEqual(src.vtable, 'Vtable.GFS_PRMSL', src.dataset)

    def test_current_era_gfs_and_cfsr_keep_their_own_vtable(self):
        self.assertEqual(md.SRC_NOMADS_WPS.vtable, 'Vtable.GFS')
        self.assertEqual(md.SRC_AWS_WPS.vtable, 'Vtable.GFS')
        self.assertEqual(md.SRC_RDA_D084001.vtable, 'Vtable.GFS')
        self.assertEqual(md.SRC_CFSR.vtable, 'Vtable.CFSR')

    def test_to_dict_carries_the_vtable_choice(self):
        # forcing.json's 'chosen' record is exactly ForcingSource.to_dict() -- confirms T13's
        # vtable choice reaches the record with no further plumbing needed.
        self.assertEqual(md.SRC_NCEI_G3.to_dict()['vtable'], 'Vtable.GFS_PRMSL')
        self.assertEqual(md.SRC_RDA_D084001.to_dict()['vtable'], 'Vtable.GFS')


class TestForcingUrlPatterns(unittest.TestCase):
    def test_ncei_g4_url_cycle_and_fhour(self):
        self.assertEqual(
            md.ncei_g4_url(datetime.datetime(2010, 1, 1, 0)),
            'https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files-old/'
            '201001/20100101/gfsanl_4_20100101_0000_000.grb2')
        self.assertEqual(
            md.ncei_g4_url(datetime.datetime(2010, 1, 1, 3)),
            'https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files-old/'
            '201001/20100101/gfsanl_4_20100101_0000_003.grb2')

    def test_ncei_g3_url_grb_extension(self):
        self.assertEqual(
            md.ncei_g3_url(datetime.datetime(2005, 1, 1, 0), 'grb'),
            'https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g3-anl-files-old/'
            '200501/20050101/gfsanl_3_20050101_0000_000.grb')

    def test_ncei_g3_wps_entry_offers_both_extensions_early_grb_first(self):
        fname, urls = md.ncei_g3_wps_entry(datetime.datetime(2005, 1, 1))[0]
        self.assertEqual(urls[0].rsplit('.', 1)[1], 'grb')
        self.assertEqual(urls[1].rsplit('.', 1)[1], 'grb2')

    def test_ncei_g3_wps_entry_offers_both_extensions_late_grb2_first(self):
        fname, urls = md.ncei_g3_wps_entry(datetime.datetime(2010, 1, 1))[0]
        self.assertEqual(urls[0].rsplit('.', 1)[1], 'grb2')
        self.assertEqual(urls[1].rsplit('.', 1)[1], 'grb')

    def test_fnl_url_grib2(self):
        self.assertEqual(
            md.fnl_url(datetime.datetime(2010, 1, 1, 0), 'grib2'),
            'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d083002/grib2/'
            '2010/2010.01/fnl_20100101_00_00.grib2')

    def test_fnl_url_grib1(self):
        self.assertEqual(
            md.fnl_url(datetime.datetime(2005, 1, 1, 0), 'grib1'),
            'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d083002/grib1/'
            '2005/2005.01/fnl_20050101_00_00.grib1')

    def test_fnl_grib1_vs_grib2_switch_at_2007_12_06(self):
        before = md.fnl_wps_entry(datetime.datetime(2007, 12, 5, 0))[0][1]
        on = md.fnl_wps_entry(datetime.datetime(2007, 12, 6, 0))[0][1]
        after = md.fnl_wps_entry(datetime.datetime(2007, 12, 7, 0))[0][1]
        self.assertTrue(before[0].endswith('.grib1'))
        self.assertTrue(on[0].endswith('.grib2'))
        self.assertTrue(after[0].endswith('.grib2'))

    def test_cfsr_tar_urls(self):
        pgb, flx = md.cfsr_tar_urls(datetime.datetime(1990, 1, 3))
        self.assertEqual(
            pgb, 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d093000/1990/'
                 'pgbh06.gdas.19900101-19900105.tar')
        self.assertEqual(
            flx, 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d093000/1990/'
                 'flxf06.gdas.19900101-19900105.tar')


class TestCfsrTarSpan(unittest.TestCase):
    def test_1990_01_03(self):
        self.assertEqual(md.cfsr_tar_span(datetime.date(1990, 1, 3)),
                         (datetime.date(1990, 1, 1), datetime.date(1990, 1, 5)))

    def test_1990_01_28(self):
        self.assertEqual(md.cfsr_tar_span(datetime.date(1990, 1, 28)),
                         (datetime.date(1990, 1, 26), datetime.date(1990, 1, 31)))

    def test_1992_02_27_leap_year(self):
        self.assertEqual(md.cfsr_tar_span(datetime.date(1992, 2, 27)),
                         (datetime.date(1992, 2, 26), datetime.date(1992, 2, 29)))


class TestProbeForcingEntries(unittest.TestCase):
    def test_all_found(self):
        with mock.patch.object(md, '_http_head', return_value=(200, {})):
            ok, results = md.probe_forcing_entries([('a.grib2', ['https://x/a.grib2'])])
        self.assertTrue(ok)
        self.assertEqual(results[0]['chosen_url'], 'https://x/a.grib2')

    def test_missing_one_fails(self):
        def fake_head(url, timeout=30):
            return (200, {}) if 'ok' in url else (404, {})
        with mock.patch.object(md, '_http_head', side_effect=fake_head):
            ok, results = md.probe_forcing_entries([('a', ['https://x/ok.grib2']),
                                                     ('b', ['https://x/missing.grib2'])])
        self.assertFalse(ok)
        self.assertEqual(results[1]['chosen_url'], None)
        self.assertEqual(results[0]['chosen_url'], 'https://x/ok.grib2')

    def test_alternates_first_missing_second_found(self):
        def fake_head(url, timeout=30):
            return (200, {}) if url.endswith('.grib2') else (404, {})
        with mock.patch.object(md, '_http_head', side_effect=fake_head):
            ok, results = md.probe_forcing_entries(
                [('a', ['https://x/a.grb', 'https://x/a.grib2'])])
        self.assertTrue(ok)
        self.assertEqual(results[0]['chosen_url'], 'https://x/a.grib2')


class TestProbeUrlWithRetry(unittest.TestCase):
    def test_503_twice_then_200_exists_after_retry(self):
        calls = {'n': 0}
        sleeps = []

        def fake_head(url, timeout=30):
            calls['n'] += 1
            if calls['n'] <= 2:
                return (503, {})
            return (200, {})

        with mock.patch.object(md, '_http_head', side_effect=fake_head):
            result, attempts = md.probe_url_with_retry('https://x/a.grib2',
                                                        sleep_fn=sleeps.append)
        self.assertTrue(result['exists'])
        self.assertEqual(calls['n'], 3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([a['status'] for a in attempts], [503, 503, 200])
        # backoff schedule used: 2s then 5s (only 2 sleeps needed before the 3rd, successful try)
        self.assertEqual(sleeps, [2, 5])

    def test_404_is_missing_without_retry(self):
        calls = {'n': 0}
        sleeps = []

        def fake_head(url, timeout=30):
            calls['n'] += 1
            return (404, {})

        with mock.patch.object(md, '_http_head', side_effect=fake_head):
            result, attempts = md.probe_url_with_retry('https://x/missing.grib2',
                                                        sleep_fn=sleeps.append)
        self.assertFalse(result['exists'])
        self.assertEqual(calls['n'], 1)          # no retry at all
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['status'], 404)
        self.assertEqual(sleeps, [])

    def test_exhausts_all_retries_on_persistent_5xx(self):
        sleeps = []
        with mock.patch.object(md, '_http_head', return_value=(500, {})):
            result, attempts = md.probe_url_with_retry('https://x/a.grib2', sleep_fn=sleeps.append)
        self.assertFalse(result['exists'])
        self.assertEqual(len(attempts), 4)       # 1 initial + 3 retries
        self.assertEqual(sleeps, [2, 5, 15])

    def test_connection_error_status_none_is_transient(self):
        def raiser(url, timeout=30):
            raise OSError("connection reset")
        sleeps = []
        with mock.patch.object(md, '_http_head', side_effect=raiser):
            result, attempts = md.probe_url_with_retry('https://x/a.grib2', sleep_fn=sleeps.append)
        self.assertFalse(result['exists'])
        self.assertEqual(len(attempts), 4)
        self.assertTrue(all(a['status'] is None for a in attempts))
        self.assertEqual(sleeps, [2, 5, 15])


class TestProbeForcingEntriesRetryIntegration(unittest.TestCase):
    def test_transient_failure_retried_then_found(self):
        calls = {'n': 0}

        def fake_head(url, timeout=30):
            calls['n'] += 1
            return (503, {}) if calls['n'] == 1 else (200, {})

        with mock.patch.object(md, '_http_head', side_effect=fake_head):
            ok, results = md.probe_forcing_entries([('a.grib2', ['https://x/a.grib2'])],
                                                    sleep_fn=lambda s: None)
        self.assertTrue(ok)
        self.assertEqual(results[0]['chosen_url'], 'https://x/a.grib2')
        self.assertEqual(len(results[0]['checked'][0]['attempts']), 2)


class TestProbeNoThreadPool(unittest.TestCase):
    def test_metdata_module_never_imports_a_thread_pool(self):
        # T4c (2026-09-24): concurrent probing against the same RDA THREDDS upstream is exactly
        # what caused the transient-failure-treated-as-missing bug T4a retries for -- probing
        # must stay sequential. metdata.py has no other use for a thread pool, so the strongest
        # honest check is a static one: this module must never import concurrent.futures /
        # ThreadPoolExecutor / threading at all (a regression here would silently reintroduce
        # concurrent probing without touching probe_forcing_entries's own source).
        src = inspect.getsource(md)
        self.assertNotIn('ThreadPoolExecutor', src)
        self.assertNotIn('concurrent.futures', src)
        self.assertNotIn('import threading', src)


if __name__ == '__main__':
    unittest.main()


class TestNceiCadenceIsSixHours(unittest.TestCase):
    def test_ncei_sources_are_6h_f000_only(self):
        # gfsanl _003 files hold ~8 accumulation messages, no 3-D fields (scc 2026-09-24).
        self.assertEqual(md.SRC_NCEI_G4.forcing_interval_h, 6)
        self.assertEqual(md.SRC_NCEI_G3.forcing_interval_h, 6)
