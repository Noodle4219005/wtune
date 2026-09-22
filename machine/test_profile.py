"""machine/test_profile.py -- pytest suite for the machine profile tooling.

Builds and runs every machine/bench/* source with tiny sizes using
whatever toolchain is actually on the local machine, and validates each
bench's JSON line -- and a full merged profile -- against the schema
documented in machine/profile.py. A missing compiler/launcher is reported
via pytest.skip for exactly the benches that need it (never silently
dropped): the skip reason names the missing tool.

Run: python3 -m pytest -q machine/test_profile.py
"""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE / 'bench'


def _load_profile_module():
    spec = importlib.util.spec_from_file_location('wtune_machine_profile_test', HERE / 'profile.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load_profile_module()

CC = shutil.which('gcc') or shutil.which('cc')
FC = shutil.which('gfortran')
MPICC = shutil.which('mpicc')
MPIRUN = shutil.which('mpirun')

HAVE_C = CC is not None
HAVE_FORTRAN = FC is not None
HAVE_MPI = MPICC is not None and MPIRUN is not None
IS_OPENMPI = False
if MPIRUN:
    try:
        v = subprocess.run([MPIRUN, '--version'], capture_output=True, text=True, timeout=10)
        IS_OPENMPI = 'Open MPI' in (v.stdout + v.stderr)
    except Exception:
        IS_OPENMPI = False


@pytest.fixture(scope='session')
def build_dir():
    with tempfile.TemporaryDirectory(prefix='machine_bench_build_') as d:
        yield Path(d)


def _compile(build_dir, name, compiler, extra_flags, src_ext='.c', libs=('-lm',)):
    src = BENCH_DIR / (name + src_ext)
    out = build_dir / name
    cmd = [compiler] + list(extra_flags) + ['-O3', '-Wall']
    if src_ext == '.f90':
        cmd += ['-J', str(build_dir)]  # gfortran drops .mod in the cwd otherwise
    cmd += [str(src), '-o', str(out)] + list(libs)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, 'compile {} failed:\ncmd={}\n{}'.format(name, cmd, r.stderr)
    return out


def _run_json(binary, args):
    r = subprocess.run([str(binary)] + list(args), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, 'run {} failed (rc={}):\nstdout={}\nstderr={}'.format(
        binary, r.returncode, r.stdout, r.stderr)
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 1, '{} printed {} lines, expected exactly 1 JSON line:\n{}'.format(
        binary, len(lines), r.stdout)
    return json.loads(lines[0])


def _validate(name, obj):
    errors = []
    mp._validate_bench_obj(name, obj, errors, 'test:' + name)
    assert not errors, 'schema errors for {}: {}'.format(name, errors)


# --- per-bench build+run+schema tests ---------------------------------------

@pytest.mark.skipif(not HAVE_C, reason='no C compiler (gcc/cc) found')
def test_freq(build_dir):
    binf = _compile(build_dir, 'freq', CC, ['-fopenmp'])
    obj = _run_json(binf, ['--seconds', '0.1'])
    _validate('freq', obj)
    assert obj['ghz_1core'] > 0
    assert obj['ghz_1core'] < 20  # sanity ceiling; catches the strength-reduction
                                  # regression this bench hit during development
                                  # (a naive "x = x + 1" chain got optimized into a
                                  # single "x += n" and reported tens of thousands
                                  # of "GHz" -- see freq.c's add_chain_step comment)
    assert len(obj['per_core_ghz_allcores']) == obj['nthreads']


@pytest.mark.skipif(not HAVE_C, reason='no C compiler (gcc/cc) found')
def test_stream(build_dir):
    binf = _compile(build_dir, 'stream', CC, ['-fopenmp'])
    obj = _run_json(binf, ['--llc-bytes', '1048576', '--iters', '3'])
    _validate('stream', obj)
    assert obj['array_elements'] >= 1000000
    assert len(obj['results']) >= 1
    for r in obj['results']:
        assert r['gbps'] > 0


@pytest.mark.skipif(not HAVE_C, reason='no C compiler (gcc/cc) found')
def test_omp(build_dir):
    binf = _compile(build_dir, 'omp', CC, ['-fopenmp'])
    obj = _run_json(binf, ['--seconds', '0.05', '--threads-list', '1,2'])
    _validate('omp', obj)
    assert len(obj['forkjoin']) == 2
    assert len(obj['barrier']) == 2
    for entry in obj['forkjoin'] + obj['barrier']:
        assert entry['reps'] > 0  # catches the fixed shared-vs-private "elapsed"
                                  # deadlock bug (non-master threads spun forever)


@pytest.mark.skipif(not HAVE_C, reason='no C compiler (gcc/cc) found')
def test_io(build_dir, tmp_path):
    binf = _compile(build_dir, 'io', CC, [], libs=())
    obj = _run_json(binf, ['--dir', str(tmp_path), '--size-mb', '4',
                           '--writers', '2', '--block-kb', '256'])
    _validate('io', obj)
    assert obj['single_writer']['gbps'] > 0
    assert obj['concurrent_writers']['ok'] == 2
    assert obj['concurrent_writers']['aggregate_gbps'] > 0
    assert list(tmp_path.iterdir()) == []  # no leftover temp files


@pytest.mark.skipif(not HAVE_FORTRAN, reason='no gfortran found')
def test_kernels(build_dir):
    binf = _compile(build_dir, 'kernels', FC, ['-fopenmp'], src_ext='.f90', libs=())
    obj = _run_json(binf, ['--nx', '24', '--ny', '24', '--nz', '12',
                           '--ncalls', '2', '--threads-full', '2'])
    _validate('kernels', obj)
    for k in ('advection', 'microphysics', 'tridiag'):
        assert obj[k]['ns_per_point_single'] > 0
        assert obj[k]['ns_per_point_full'] > 0


def _mpirun_np(n, extra_args):
    cmd = [MPIRUN]
    if IS_OPENMPI:
        cmd.append('--oversubscribe')
    cmd += ['-np', str(n)] + list(extra_args)
    return cmd


@pytest.mark.skipif(not HAVE_MPI, reason='no mpicc/mpirun found')
def test_halo(build_dir):
    binf = _compile(build_dir, 'halo', MPICC, [])
    cmd = _mpirun_np(4, [str(binf), '--tag', 'test', '--halo', '5', '--evert', '8',
                         '--nfields', '2', '--reps', '3', '--patches', '8,16',
                         '--pingpong-max-bytes', '256'])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, 'halo run failed:\ncmd={}\n{}'.format(cmd, r.stderr)
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    _validate('halo', obj)
    assert obj['nranks'] == 4
    assert len(obj['exchange']) == 2
    assert obj['pingpong'] is not None and len(obj['pingpong']) >= 1


# --- profile.py merge/validate/summary, no toolchain required --------------

def _synthetic_bench_objs():
    freq = {"bench": "freq", "ncores_allowed": 4, "nthreads": 4,
            "duration_s_requested": 0.1, "duration_s_1core": 0.1,
            "cycles_per_op_assumed": 1, "ghz_1core": 2.0, "ghz_allcores_mean": 1.8,
            "ghz_allcores_min": 1.5, "ghz_allcores_max": 2.0,
            "per_core_ghz_allcores": [1.5, 1.8, 2.0, 1.9], "scaling_cur_freq_mhz": None}
    stream = {"bench": "stream", "array_elements": 1000000, "array_bytes_total": 24000000,
              "llc_bytes_assumed": 1048576, "iters": 3,
              "results": [{"threads": 1, "gbps": 5.0, "time_s": 0.01}]}
    kernels = {"bench": "kernels", "nx": 10, "ny": 10, "nz": 10, "ncalls": 1, "threads_full": 2,
               "advection": {"ns_per_point_single": 5.0, "ns_per_point_full": 3.0},
               "microphysics": {"ns_per_point_single": 50.0, "ns_per_point_full": 30.0},
               "tridiag": {"ns_per_point_single": 20.0, "ns_per_point_full": 10.0}}
    halo_intra = {"bench": "halo", "tag": "intra", "nranks": 4, "px": 2, "py": 2,
                  "halo_width": 5, "e_vert": 33, "nfields": 4,
                  "exchange": [{"patch": 20, "msg_bytes_ew": 1, "msg_bytes_ns": 1,
                                "us_per_exchange": 10.0, "gbps": 1.0}],
                  "pingpong": [{"bytes": 8, "us": 1.0, "gbps": 0.001}]}
    omp = {"bench": "omp", "ncores": 4,
           "forkjoin": [{"threads": 1, "us_per_region": 1.0, "reps": 100}],
           "barrier": [{"threads": 1, "us_per_barrier": 0.5, "reps": 100}]}
    io = {"bench": "io", "dir": "/tmp", "size_mb": 10, "block_kb": 1024,
          "single_writer": {"bytes": 1000, "seconds": 0.1, "gbps": 0.01},
          "concurrent_writers": {"n": 2, "ok": 2, "bytes_each": 1000, "wall_seconds": 0.1,
                                  "slowest_writer_seconds": 0.1, "aggregate_gbps": 0.02}}
    return {'freq': freq, 'stream': stream, 'kernels': kernels,
            'halo_intra': halo_intra, 'omp': omp, 'io': io}


def test_merge_and_summary_synthetic(tmp_path):
    objs = _synthetic_bench_objs()
    files = []
    for name, obj in objs.items():
        p = tmp_path / (name + '.json')
        p.write_text(json.dumps(obj))
        files.append(str(p))
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({"uname": "x", "page_size_bytes": 4096, "node_count": 1}))

    doc = mp.merge(files, str(meta_path), 'testhost', '2026-01-01')
    errors = mp.validate(doc)
    assert errors == [], errors
    assert doc['benches']['halo'][0]['tag'] == 'intra'
    assert doc['missing_benches'] == []

    lines = mp.summarize(doc)
    text = '\n'.join(lines)
    for word in ('freq', 'stream', 'kernels', 'halo', 'omp', 'io'):
        assert word in text


def test_merge_two_halo_placements(tmp_path):
    objs = _synthetic_bench_objs()
    halo_inter = dict(objs['halo_intra'])
    halo_inter['tag'] = 'inter'
    files = []
    for name, obj in list(objs.items()) + [('halo_inter', halo_inter)]:
        p = tmp_path / (name + '.json')
        p.write_text(json.dumps(obj))
        files.append(str(p))
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({"uname": "x", "page_size_bytes": 4096, "node_count": 2}))
    doc = mp.merge(files, str(meta_path), 'h', '2026-01-01')
    assert mp.validate(doc) == []
    tags = sorted(h['tag'] for h in doc['benches']['halo'])
    assert tags == ['inter', 'intra']


def test_validate_catches_missing_field():
    doc = {"schema_version": 1, "host": "h", "date": "2026-01-01",
           "metadata": {"uname": "x", "page_size_bytes": 4096, "node_count": 1},
           "benches": {"freq": {"bench": "freq"}},
           "missing_benches": []}
    errors = mp.validate(doc)
    assert errors


def test_validate_catches_bench_key_mismatch():
    doc = {"schema_version": 1, "host": "h", "date": "2026-01-01",
           "metadata": {"uname": "x", "page_size_bytes": 4096, "node_count": 1},
           "benches": {"freq": dict(_synthetic_bench_objs()['stream'])},
           "missing_benches": []}
    errors = mp.validate(doc)
    assert any('"bench" field' in e for e in errors)


def test_missing_bench_reported(tmp_path):
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({"uname": "x", "page_size_bytes": 4096, "node_count": 1}))
    doc = mp.merge([], str(meta_path), 'h', '2026-01-01')
    assert set(doc['missing_benches']) == set(mp.KNOWN_BENCHES)
    errors = mp.validate(doc)
    # an empty "benches" dict IS schema-valid (validate checks shape of
    # whatever is present); absence is tracked via missing_benches instead.
    assert errors == []


# --- end-to-end driver, only when the full toolchain is present ------------

@pytest.mark.skipif(not (HAVE_C and HAVE_FORTRAN and HAVE_MPI),
                     reason='need gcc + gfortran + mpicc + mpirun for the end-to-end driver test')
def test_run_machine_profile_end_to_end(tmp_path):
    script = HERE / 'run_machine_profile.sh'
    env = dict(os.environ)
    env.update({
        'MACHINE_PROFILE_OUT_DIR': str(tmp_path),
        'MACHINE_PROFILE_TMP': str(tmp_path / 'tmp'),
        'MP_FREQ_SECONDS': '0.05', 'MP_OMP_SECONDS': '0.05',
        'MP_STREAM_ITERS': '2', 'MP_STREAM_LLC_BYTES': '1048576',
        'MP_KERNELS_NX': '16', 'MP_KERNELS_NY': '16', 'MP_KERNELS_NZ': '8', 'MP_KERNELS_NCALLS': '1',
        'MP_IO_SIZE_MB': '4', 'MP_IO_WRITERS': '2',
        'MP_HALO_PATCHES': '8', 'MP_HALO_REPS': '2', 'MP_HALO_PINGPONG_MAX_BYTES': '64',
        'MP_HALO_INTRA_NP': '4',
    })
    r = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, 'driver failed:\nstdout={}\nstderr={}'.format(r.stdout, r.stderr)
    outputs = list(tmp_path.glob('machine_profile_*.json'))
    assert len(outputs) == 1, outputs
    doc = json.loads(outputs[0].read_text())
    errors = mp.validate(doc)
    assert errors == [], errors
    for name in mp.KNOWN_BENCHES:
        assert name in doc['benches'], '{} missing from end-to-end profile: {}'.format(
            name, doc['missing_benches'])
    # single-node local run: only the "intra" halo placement should exist
    assert len(doc['benches']['halo']) == 1
    assert doc['benches']['halo'][0]['tag'] == 'intra'
