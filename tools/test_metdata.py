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

    def test_wps_field_specs_nonempty_and_cover_plevels(self):
        self.assertGreater(len(md.WPS_FIELD_SPECS), 100)
        plevel_keys = [k for k in md.WPS_FIELD_SPECS if k.startswith('TMP_') and k.endswith('mb')]
        self.assertEqual(len(plevel_keys), len(md.GFS_PRESSURE_LEVELS_HPA))

    def test_wps_field_specs_cover_surface_set(self):
        surf_vars = {spec['var'] for spec in md.WPS_FIELD_SPECS.values() if 'mb' not in spec['level']}
        self.assertIn('PRMSL', surf_vars)
        self.assertIn('SOILW', surf_vars)


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


if __name__ == '__main__':
    unittest.main()
