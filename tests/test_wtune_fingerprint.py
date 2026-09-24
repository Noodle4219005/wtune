"""Tests for the autotuner step-2 phase fingerprinter (fingerprint/).

stdlib only, unittest (pytest-compatible). Python 3.6.
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

FINGERPRINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fingerprint')
sys.path.insert(0, FINGERPRINT_DIR)

import phase_classify  # noqa: E402
import fingerprint as fp  # noqa: E402

FINGERPRINT_PY = os.path.join(FINGERPRINT_DIR, 'fingerprint.py')


# --------------------------------------------------------------------------
# phase_classify.classify
# --------------------------------------------------------------------------

class ClassifyDsoRuleTests(unittest.TestCase):
    def test_mpi_dso(self):
        self.assertEqual(phase_classify.classify('MPI_Waitall', '/usr/lib64/libmpi.so.12'), 'MPI')

    def test_inherit_dso_substrings(self):
        for dso in ('/opt/intel/lib/libiomp5.so', '/lib/libgomp.so.1',
                    '/opt/intel/lib/libimf.so', '/opt/intel/lib/libsvml.so',
                    '/opt/intel/lib/libirc.so', '/opt/intel/lib/libintlc.so.5'):
            self.assertEqual(phase_classify.classify('foo', dso), 'INHERIT', dso)

    def test_inherit_libm_prefix(self):
        self.assertEqual(phase_classify.classify('foo', '/lib64/libm.so.6'), 'INHERIT')
        self.assertEqual(phase_classify.classify('foo', '/lib64/libm-2.28.so'), 'INHERIT')

    def test_io_dso_substrings(self):
        for dso in ('/opt/netcdf/lib/libnetcdf.so.19', '/lib/libhdf5.so.103',
                    '/lib/libpnetcdf.so', '/lib/libadios2_core.so', '/lib/libz.so.1',
                    '/lib/libsz.so.2', '/lib/libblosc.so', '/lib/libzstd.so.1',
                    '/lib/libbz2.so.1'):
            self.assertEqual(phase_classify.classify('foo', dso), 'IO', dso)

    def test_non_wrf_dso_defaults_to_inherit(self):
        self.assertEqual(phase_classify.classify('foo', '/lib64/libc.so.6'), 'INHERIT')
        self.assertEqual(phase_classify.classify('foo', '/lib64/ld-linux-x86-64.so.2'), 'INHERIT')
        self.assertEqual(phase_classify.classify('[unknown]', '[unknown]'), 'INHERIT')
        self.assertEqual(phase_classify.classify('foo', '[vdso]'), 'INHERIT')


class ClassifyWrfSymRuleTests(unittest.TestCase):
    def test_micro(self):
        self.assertEqual(phase_classify.classify('module_mp_thompson', '/x/wrf.exe'), 'MICRO')
        self.assertEqual(phase_classify.classify('microphysics_driver_run', '/x/wrf.exe'), 'MICRO')

    def test_radiation(self):
        self.assertEqual(phase_classify.classify('module_ra_rrtmg', '/x/wrf.exe'), 'RADIATION')
        self.assertEqual(phase_classify.classify('radiation_driver', '/x/wrf.exe'), 'RADIATION')
        self.assertEqual(phase_classify.classify('pre_radiation', '/x/wrf.exe'), 'RADIATION')

    def test_pbl(self):
        self.assertEqual(phase_classify.classify('module_bl_ysu', '/x/wrf.exe'), 'PBL')
        self.assertEqual(phase_classify.classify('pbl_driver', '/x/wrf.exe'), 'PBL')

    def test_surface(self):
        self.assertEqual(phase_classify.classify('module_sf_sfclay', '/x/wrf.exe'), 'SURFACE')
        self.assertEqual(phase_classify.classify('surface_driver', '/x/wrf.exe'), 'SURFACE')

    def test_cumulus(self):
        self.assertEqual(phase_classify.classify('module_cu_kf', '/x/wrf.exe'), 'CUMULUS')
        self.assertEqual(phase_classify.classify('cumulus_driver', '/x/wrf.exe'), 'CUMULUS')

    def test_io_inside_wrf(self):
        for sym in ('ext_ncd_open', 'wrf_put_dom_ti', 'wrf_get_dom_ti', 'med_hist_out',
                    'module_io_int', 'collect_on_comm', 'output_wrf', 'ext_adios2_open'):
            self.assertEqual(phase_classify.classify(sym, '/x/wrf.exe'), 'IO', sym)

    def test_unknown_and_unmatched_inside_wrf_is_dynamics_other(self):
        self.assertEqual(phase_classify.classify('[unknown]', '/x/wrf.exe'), 'DYNAMICS_OTHER')
        self.assertEqual(phase_classify.classify('solve_em', '/x/wrf.exe'), 'DYNAMICS_OTHER')

    def test_wtimer_wrapper_symbols_inherit_only_on_wtimer_binary(self):
        self.assertEqual(phase_classify.classify('__wrap_MPI_Waitall', '/x/wrf_wtimer.exe'),
                          'INHERIT')
        self.assertEqual(phase_classify.classify('wt_start', '/x/wrf_wtimer.exe'), 'INHERIT')
        # same prefix, but on plain wrf.exe it is not a recognized wrapper -> DYNAMICS_OTHER
        self.assertEqual(phase_classify.classify('__wrap_MPI_Waitall', '/x/wrf.exe'),
                          'DYNAMICS_OTHER')

    def test_phase_rules_still_apply_on_wtimer_binary(self):
        self.assertEqual(phase_classify.classify('module_mp_thompson', '/x/wrf_wtimer.exe'),
                          'MICRO')


# --------------------------------------------------------------------------
# INHERIT resolution + time-weighted attribution
# --------------------------------------------------------------------------

class AttributeStreamTests(unittest.TestCase):
    def test_inherit_after_micro_counts_as_micro(self):
        samples = [
            (0.0, 'module_mp_thompson', '/x/wrf.exe'),
            (1.0, 'somefunc', '/opt/intel/lib/libimf.so'),
        ]
        out = fp.attribute_stream(samples)
        self.assertEqual(out[0][1], 'MICRO')
        self.assertEqual(out[1][1], 'MICRO')

    def test_inherit_at_stream_start_is_dynamics_other(self):
        samples = [(0.0, 'somefunc', '/opt/intel/lib/libimf.so')]
        out = fp.attribute_stream(samples)
        self.assertEqual(out[0][1], 'DYNAMICS_OTHER')

    def test_dt_cap_applies_and_last_sample_gets_median_dt(self):
        samples = [
            (0.0, 'module_mp_thompson', '/x/wrf.exe'),
            (5.0, 'module_mp_thompson', '/x/wrf.exe'),
        ]
        out = fp.attribute_stream(samples, dt_cap=1.0)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][2], 1.0)  # capped from 5.0
        self.assertAlmostEqual(out[1][2], 1.0)  # last sample: median([1.0]) == 1.0

    def test_unsorted_input_is_sorted_by_time(self):
        samples = [
            (2.0, 'module_ra_rrtmg', '/x/wrf.exe'),
            (0.0, 'module_mp_thompson', '/x/wrf.exe'),
        ]
        out = fp.attribute_stream(samples)
        self.assertEqual([t for t, _p, _d in out], [0.0, 2.0])
        self.assertEqual(out[0][1], 'MICRO')
        self.assertEqual(out[1][1], 'RADIATION')


class SegmentMicroCallsTests(unittest.TestCase):
    def test_micro_micro_mpi_micro_gives_two_calls(self):
        attributed = [
            (0, 'MICRO', 1.0),
            (1, 'MICRO', 2.0),
            (2, 'MPI', 3.0),
            (3, 'MICRO', 4.0),
        ]
        calls = fp.segment_micro_calls(attributed)
        self.assertEqual(calls, [3.0, 4.0])

    def test_no_micro_gives_no_calls(self):
        attributed = [(0, 'MPI', 1.0), (1, 'DYNAMICS_OTHER', 1.0)]
        self.assertEqual(fp.segment_micro_calls(attributed), [])


class BuildRankDataMasterWorkerTests(unittest.TestCase):
    def test_master_tid_equals_pid_worker_does_not(self):
        samples_by_stream = {
            ('h1', 10, 10): [  # master: tid == pid
                (0.0, 'module_mp_thompson', '/x/wrf.exe'),
                (1.0, 'module_mp_thompson', '/x/wrf.exe'),
                (2.0, 'module_mp_thompson', '/x/wrf.exe'),
            ],
            ('h1', 10, 11): [  # worker thread
                (0.0, 'module_mp_thompson', '/x/wrf.exe'),
                (0.5, 'module_mp_thompson', '/x/wrf.exe'),
            ],
        }
        pid_info = {('h1', 10): {'rank': 0, 'threads': 2, 'home': ''}}
        ranks = fp.build_rank_data(samples_by_stream, pid_info)
        self.assertIn(0, ranks)
        rd = ranks[0]
        self.assertEqual(rd['master_micro_samples'], 3)
        self.assertEqual(rd['worker_micro_samples'], 2)
        self.assertGreater(rd['total'], 0.0)

    def test_samples_for_unknown_pid_are_dropped(self):
        samples_by_stream = {('h1', 999, 999): [(0.0, 'module_mp_thompson', '/x/wrf.exe')]}
        pid_info = {('h1', 10): {'rank': 0, 'threads': 1, 'home': ''}}
        ranks = fp.build_rank_data(samples_by_stream, pid_info)
        self.assertEqual(ranks, {})


# --------------------------------------------------------------------------
# perf-script line parser tolerance
# --------------------------------------------------------------------------

class LoadSamplesParserTests(unittest.TestCase):
    def _write_sampler_dir(self, tmp, lines):
        with open(os.path.join(tmp, 'meta.h1.txt'), 'w') as fh:
            fh.write('PID host=h1 pid=300 rank=0 threads=1 home=0\n')
        with gzip.open(os.path.join(tmp, 'perf.h1.txt.gz'), 'wt') as fh:
            fh.write('\n'.join(lines) + '\n')

    def test_tolerant_regex_and_unparsed_count(self):
        lines = [
            '   300/300 [  1] 10.000000:  400000 module_mp_thompson (/x/wrf.exe)',
            '   300/300 [  1] 10.010000:  400010 module_mp_thompson+0x1a (/x/wrf.exe)',
            '   300/300 [  1] 10.020000:  400020 [unknown] (/x/wrf.exe)',
            '     300/300   [   2]   10.030000:    400030    module_mp_thompson   (/x/wrf.exe)   ',
            'this is not a valid perf script line at all',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_sampler_dir(tmp, lines)
            pid_info, _hw, _hosts = fp.load_meta(tmp)
            samples, n_parsed, n_unparsed, n_ignored, n_files = fp.load_samples(tmp, pid_info)
            self.assertEqual(n_parsed, 4)
            self.assertEqual(n_unparsed, 1)
            self.assertEqual(n_ignored, 0)
            self.assertEqual(n_files, 1)
            self.assertEqual(len(samples[('h1', 300, 300)]), 4)

    def test_lines_without_cpu_column(self):
        # sampler v2 asks perf script for pid,tid,time,ip,sym,dso: there is no "[cpu]" column
        # (job 1074372: requesting cpu without `perf record --sample-cpu` made perf print nothing)
        lines = [
            '   300/300 10.000000:  400000 module_mp_thompson (/x/wrf.exe)',
            '   300/301 10.010000:  400010 module_mp_thompson+0x1a (/x/wrf.exe)',
            '     300/300    10.020000:    400020    [unknown]   (/x/wrf.exe)   ',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_sampler_dir(tmp, lines)
            pid_info, _hw, _hosts = fp.load_meta(tmp)
            samples, n_parsed, n_unparsed, n_ignored, n_files = fp.load_samples(tmp, pid_info)
            self.assertEqual(n_parsed, 3)
            self.assertEqual(n_unparsed, 0)
            self.assertEqual(len(samples[('h1', 300, 300)]), 2)
            self.assertEqual(len(samples[('h1', 300, 301)]), 1)

    def test_unparsed_over_one_percent_fails_cli(self):
        good = '   300/300 [  1] 10.000000:  400000 module_mp_thompson (/x/wrf.exe)'
        garbage = ['not a perf line %d' % i for i in range(20)]
        with tempfile.TemporaryDirectory() as sampler_dir, \
                tempfile.TemporaryDirectory() as run_dir:
            self._write_sampler_dir(sampler_dir, [good] + garbage)
            with open(os.path.join(run_dir, 'rsl.out.0000'), 'w') as fh:
                fh.write('')
            with open(os.path.join(run_dir, 'namelist.input'), 'w') as fh:
                fh.write('&domains\n nproc_x=1, nproc_y=1, numtiles=1,\n/\n')
            out_prefix = os.path.join(run_dir, 'out')
            rc = fp.main(['--sampler-dir', sampler_dir, '--run-dir', run_dir,
                          '--out', out_prefix])
            self.assertEqual(rc, 1)


# --------------------------------------------------------------------------
# diagnosis thresholds
# --------------------------------------------------------------------------

class DiagnosisThresholdTests(unittest.TestCase):
    def test_micro_halo_and_spatial_labels_fire(self):
        phase_share = {'MICRO': 0.10, 'RADIATION': 0.05, 'MPI': 0.30, 'IO': 0.05}
        rank_imbalance = {'max_mean': 2.0}
        thread_imbalance = {'tiles_divisible': True, 'busy_ratio_median': 1.05,
                             'numtiles': 12, 'omp_threads': 2}
        phase_predictability = {'half_window_spearman': 0.85}
        labels = fp.compute_diagnosis(phase_share, rank_imbalance, thread_imbalance,
                                       phase_predictability)
        names = set(d['label'] for d in labels)
        self.assertEqual(names, {'MICRO_HEAVY', 'HALO_DOMINATED', 'SPATIALLY_IMBALANCED'})

    def test_omp_tile_and_predictable_labels_fire_independently(self):
        phase_share = {'MICRO': 0.01, 'RADIATION': 0.01, 'MPI': 0.01, 'IO': 0.01}
        rank_imbalance = {'max_mean': 1.0}  # below spatial threshold
        thread_imbalance = {'tiles_divisible': False, 'busy_ratio_median': 1.05,
                             'numtiles': 13, 'omp_threads': 2}
        phase_predictability = {'half_window_spearman': 0.95}
        labels = fp.compute_diagnosis(phase_share, rank_imbalance, thread_imbalance,
                                       phase_predictability)
        names = set(d['label'] for d in labels)
        self.assertEqual(names, {'OMP_TILE_IMBALANCED', 'PREDICTABLE_PHASE_COST'})

    def test_nothing_fires_below_all_thresholds(self):
        phase_share = {'MICRO': 0.01, 'RADIATION': 0.01, 'MPI': 0.01, 'IO': 0.01}
        rank_imbalance = {'max_mean': 1.0}
        thread_imbalance = {'tiles_divisible': True, 'busy_ratio_median': 1.0,
                             'numtiles': 12, 'omp_threads': 2}
        phase_predictability = {'half_window_spearman': 0.5}
        labels = fp.compute_diagnosis(phase_share, rank_imbalance, thread_imbalance,
                                       phase_predictability)
        self.assertEqual(labels, [])


# --------------------------------------------------------------------------
# full synthetic pipeline: validation math + CLI end-to-end
# --------------------------------------------------------------------------

# per-rank perf sample plan: (phase, count) tuples: `count` samples of that
# phase, each 0.01s after the previous one; deliberately built so that, with
# uniform 0.01s gaps, the phase totals come out to EXACT round numbers that
# match the synthetic wtimer ground truth below.
_TICK = 0.01
_RANK_PLAN = [
    ('MICRO', 'module_mp_thompson', '/x/wrf.exe', 16),
    ('RADIATION', 'module_ra_rrtmg', '/x/wrf.exe', 6),
    ('PBL', 'module_bl_ysu', '/x/wrf.exe', 3),
    ('SURFACE', 'module_sf_sfclay', '/x/wrf.exe', 3),
    ('MPI', 'MPI_Waitall', '/opt/lib/libmpi.so.12', 15),
    ('DYNAMICS_OTHER', 'solve_em', '/x/wrf.exe', 16),
]
# one extra trailing DYNAMICS_OTHER sample: as the last sample of the whole
# stream it gets dt = median(all other dts) = _TICK (all gaps are uniform),
# bringing DYNAMICS_OTHER's total from 16 to 17 ticks (0.17s).


def _build_perf_lines_for_pid(pid):
    lines = []
    t = 0.0
    for _phase, sym, dso, count in _RANK_PLAN:
        for _ in range(count):
            lines.append('   %d/%d [  0] %.6f:  400000 %s (%s)' % (pid, pid, t, sym, dso))
            t += _TICK
    # trailing sentinel sample: same DYNAMICS_OTHER symbol, one more tick later
    lines.append('   %d/%d [  0] %.6f:  400000 solve_em (/x/wrf.exe)' % (pid, pid, t))
    return lines


def _write_synthetic_case(sampler_dir, run_dir, wtimer_dir, wtimer_micro_scale=1.0):
    # meta: 2 ranks, single (master) thread each, window steps 2..5
    with open(os.path.join(sampler_dir, 'meta.h1.txt'), 'w') as fh:
        fh.write('PID host=h1 pid=100 rank=0 threads=1 home=0\n')
        fh.write('PID host=h1 pid=200 rank=1 threads=1 home=1\n')
        fh.write('START host=h1 step=2 epoch=1000.0\n')
        fh.write('STOP host=h1 step=5 epoch=1010.0\n')

    lines = _build_perf_lines_for_pid(100) + _build_perf_lines_for_pid(200)
    with gzip.open(os.path.join(sampler_dir, 'perf.h1.txt.gz'), 'wt') as fh:
        fh.write('\n'.join(lines) + '\n')

    with open(os.path.join(run_dir, 'namelist.input'), 'w') as fh:
        fh.write('&domains\n nproc_x=2, nproc_y=1, numtiles=2,\n/\n')

    rsl_elapsed = [0.10, 0.10, 0.20, 0.20, 0.20, 0.10]  # window is indices 2,3,4 -> sum 0.60
    with open(os.path.join(run_dir, 'rsl.out.0000'), 'w') as fh:
        for i, e in enumerate(rsl_elapsed):
            fh.write('Timing for main: time 2026-09-14_00:00:%02d on domain   1:    %.5f elapsed seconds\n'
                      % (i, e))

    micro_rows = [0.05, 0.06, 0.05]
    for rank in (0, 1):
        with open(os.path.join(wtimer_dir, 'wtimer.%d.csv' % rank), 'w') as fh:
            fh.write('# tsc_hz=2000000000 wall_s=0.6\n')
            for i, m in enumerate(micro_rows):
                call_index = 2 + i
                micro = m * wtimer_micro_scale
                fh.write('%d,%.6f,0.02,0.0,0.01,0.01,0.0,0.05,5\n' % (call_index, micro))
            fh.write('# end\n')


class ValidationMathTests(unittest.TestCase):
    def test_matching_shares_pass(self):
        with tempfile.TemporaryDirectory() as sampler_dir, \
                tempfile.TemporaryDirectory() as run_dir, \
                tempfile.TemporaryDirectory() as wtimer_dir:
            _write_synthetic_case(sampler_dir, run_dir, wtimer_dir, wtimer_micro_scale=1.0)

            pid_info, host_window, _hosts = fp.load_meta(sampler_dir)
            samples_by_stream, n_parsed, n_unparsed, _ignored, _files = \
                fp.load_samples(sampler_dir, pid_info)
            self.assertEqual(n_unparsed, 0)
            self.assertGreater(n_parsed, 0)

            ranks = fp.build_rank_data(samples_by_stream, pid_info)
            phase_share = fp.pooled_phase_share(ranks)

            # designed exactly: fp shares should equal the wtimer-derived shares
            self.assertAlmostEqual(phase_share['MICRO'], 0.16 / 0.60, places=6)
            self.assertAlmostEqual(phase_share['MPI'], 0.15 / 0.60, places=6)

            validation = fp.run_validation(wtimer_dir, run_dir, host_window, ranks, phase_share)
            self.assertIsNotNone(validation)
            self.assertTrue(validation['pass'])
            for key in ('micro', 'radiation', 'dynamics_other', 'mpi'):
                self.assertLessEqual(validation['diff_pp'][key], fp.VALIDATION_PASS_PP)

    def test_perturbed_shares_fail(self):
        with tempfile.TemporaryDirectory() as sampler_dir, \
                tempfile.TemporaryDirectory() as run_dir, \
                tempfile.TemporaryDirectory() as wtimer_dir:
            # wtimer micro tripled: ground truth diverges sharply from the
            # (unchanged) perf-derived fingerprint.
            _write_synthetic_case(sampler_dir, run_dir, wtimer_dir, wtimer_micro_scale=3.0)

            pid_info, host_window, _hosts = fp.load_meta(sampler_dir)
            samples_by_stream, _p, _u, _i, _f = fp.load_samples(sampler_dir, pid_info)
            ranks = fp.build_rank_data(samples_by_stream, pid_info)
            phase_share = fp.pooled_phase_share(ranks)

            validation = fp.run_validation(wtimer_dir, run_dir, host_window, ranks, phase_share)
            self.assertIsNotNone(validation)
            self.assertFalse(validation['pass'])
            self.assertGreater(validation['diff_pp']['micro'], fp.VALIDATION_PASS_PP)


class CliEndToEndTests(unittest.TestCase):
    def test_cli_runs_and_writes_full_json(self):
        with tempfile.TemporaryDirectory() as sampler_dir, \
                tempfile.TemporaryDirectory() as run_dir, \
                tempfile.TemporaryDirectory() as wtimer_dir:
            _write_synthetic_case(sampler_dir, run_dir, wtimer_dir, wtimer_micro_scale=1.0)
            out_prefix = os.path.join(run_dir, 'fingerprint')

            proc = subprocess.Popen(
                [sys.executable, FINGERPRINT_PY,
                 '--sampler-dir', sampler_dir, '--run-dir', run_dir,
                 '--omp-threads', '1', '--wtimer-dir', wtimer_dir, '--out', out_prefix],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, _ = proc.communicate(timeout=60)
            self.assertEqual(proc.returncode, 0, stdout.decode('utf-8', 'replace'))
            self.assertIn(b'FINGERPRINT', stdout)
            self.assertIn(b'VALIDATION', stdout)

            with open(out_prefix + '.json') as fh:
                doc = json.load(fh)
            for key in ('meta', 'phase_share', 'per_rank', 'rank_imbalance',
                        'thread_imbalance', 'mpi_floor', 'phase_predictability',
                        'spatial', 'io_share', 'cpu_vs_memory', 'diagnosis',
                        'thresholds', 'validation'):
                self.assertIn(key, doc)
            self.assertTrue(os.path.exists(out_prefix + '_per_rank.csv'))
            self.assertTrue(os.path.exists(out_prefix + '_micro_calls.csv'))


# --------------------------------------------------------------------------
# preflight mode
# --------------------------------------------------------------------------

class PreflightModeTests(unittest.TestCase):
    def test_preflight_parses_old_format_and_filters_ref_cycles(self):
        lines = [
            '   500   1.000000:       1000 cycles:u:      400000 module_mp_thompson (/x/wrf.exe)',
            '   500   1.010000:       1000 cycles:u:      400010 module_ra_rrtmg (/x/wrf.exe)',
            '   500   1.020000:       1000 ref-cycles:u:  400020 module_mp_thompson (/x/wrf.exe)',
            'totally not a perf script line',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'old.txt')
            with open(path, 'w') as fh:
                fh.write('\n'.join(lines) + '\n')
            samples, parsed, total = fp.parse_preflight_stream(path)
            self.assertEqual(total, 4)
            self.assertEqual(parsed, 3)
            # the ref-cycles line matched the line format but is filtered from samples
            self.assertEqual(len(samples), 2)
            syms = sorted(s[1] for s in samples)
            self.assertEqual(syms, ['module_mp_thompson', 'module_ra_rrtmg'])

    def test_run_preflight_cli_mode_returns_zero(self):
        lines = ['   500   1.000000:       1000 cycles:u:      400000 module_mp_thompson (/x/wrf.exe)']
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'old.txt')
            with open(path, 'w') as fh:
                fh.write('\n'.join(lines) + '\n')
            rc = fp.main(['--preflight-perf-script', path])
            self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
