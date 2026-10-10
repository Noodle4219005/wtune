#!/usr/bin/env python3
"""machine/profile.py -- merge, validate, and summarize a MACHINE PROFILE.

A machine profile is what `machine/run_machine_profile.sh` produces on an
allocated node: metadata about the machine plus one JSON object per bench
in `machine/bench/` (freq, stream, kernels, halo, omp, io). Each bench
binary emits exactly one JSON line to stdout; the driver script collects
those lines and calls this module's `merge` CLI action to assemble them,
with metadata, into a single `machine_profile_<host>_<date>.json`.

This module has two jobs:
  1. `merge`    -- read raw per-bench JSON files + a metadata JSON file,
                   validate them against the schema below, and write the
                   combined document.
  2. `summary`  -- read an already-merged document and print a short
                   human-readable summary (this is what
                   `wtune/cli.py machine-profile <json>` calls).

MERGED SCHEMA (schema_version 1)
---------------------------------
{
  "schema_version": 1,
  "host": str,
  "date": "YYYY-MM-DD",
  "metadata": {
    "uname": str,
    "lscpu": str,              # raw `lscpu` text, or "" if unavailable
    "cc_version": str,
    "fc_version": str,
    "mpi_version": str,
    "page_size_bytes": int,
    "node_count": int,
    "ncores_per_node": int | null
  },
  "benches": {
    # present only for benches that ran and produced valid JSON; a bench
    # that failed to build/run is instead listed in "missing_benches".
    "freq":    {see machine/bench/freq.c header for fields},
    "stream":  {see machine/bench/stream.c header for fields},
    "kernels": {see machine/bench/kernels.f90 header for fields},
    "halo":    [ {..., "tag": "intra"}, {..., "tag": "inter"} ],
      # halo is special: the driver runs it twice (intra/inter placement)
      # so "halo" maps to a LIST of 1 or 2 halo JSON objects, not one.
    "omp":     {see machine/bench/omp.c header for fields},
    "io":      {see machine/bench/io.c header for fields}
  },
  "missing_benches": [str, ...]   # bench names that could not be run
}

Each per-bench object always carries its own "bench" field naming itself
(e.g. {"bench": "freq", ...}); `validate()` cross-checks that field against
the key it is filed under.

Only the Python standard library is used, per the project's
dependency-free constraint for `machine/`.
"""

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Bench name -> required top-level keys in that bench's own JSON object
# (beyond "bench" itself, which is checked separately). Kept intentionally
# shallow: this validates SHAPE (did the bench emit the fields callers
# depend on), not every numeric invariant.
REQUIRED_FIELDS = {
    'freq': ['ncores_allowed', 'nthreads', 'ghz_1core', 'ghz_allcores_mean',
              'ghz_allcores_min', 'ghz_allcores_max', 'per_core_ghz_allcores'],
    'stream': ['array_elements', 'array_bytes_total', 'llc_bytes_assumed', 'results'],
    'kernels': ['nx', 'ny', 'nz', 'ncalls', 'threads_full',
                'advection', 'microphysics', 'tridiag'],
    'halo': ['tag', 'nranks', 'px', 'py', 'halo_width', 'e_vert', 'nfields',
             'exchange', 'pingpong'],
    'omp': ['ncores', 'forkjoin', 'barrier'],
    'io': ['dir', 'size_mb', 'single_writer', 'concurrent_writers'],
}

KNOWN_BENCHES = tuple(REQUIRED_FIELDS.keys())


class ProfileError(Exception):
    pass


def _validate_bench_obj(name, obj, errors, where):
    if not isinstance(obj, dict):
        errors.append('{}: not a JSON object'.format(where))
        return
    if obj.get('bench') != name:
        errors.append('{}: "bench" field is {!r}, expected {!r}'.format(
            where, obj.get('bench'), name))
    for field in REQUIRED_FIELDS[name]:
        if field not in obj:
            errors.append('{}: missing required field {!r}'.format(where, field))


def validate(doc):
    """Validate a merged profile document. Returns a list of error strings
    (empty list == valid)."""
    errors = []
    if doc.get('schema_version') != SCHEMA_VERSION:
        errors.append('schema_version is {!r}, expected {}'.format(
            doc.get('schema_version'), SCHEMA_VERSION))
    for key in ('host', 'date'):
        if not doc.get(key):
            errors.append('missing or empty top-level field {!r}'.format(key))
    meta = doc.get('metadata')
    if not isinstance(meta, dict):
        errors.append('"metadata" missing or not an object')
    else:
        for key in ('uname', 'page_size_bytes', 'node_count'):
            if key not in meta:
                errors.append('metadata: missing field {!r}'.format(key))

    benches = doc.get('benches')
    if not isinstance(benches, dict):
        errors.append('"benches" missing or not an object')
        benches = {}
    for name, obj in benches.items():
        if name not in KNOWN_BENCHES:
            errors.append('benches: unknown bench key {!r}'.format(name))
            continue
        if name == 'halo':
            if not isinstance(obj, list) or not obj:
                errors.append('benches.halo: expected a non-empty list of halo objects')
            else:
                for i, item in enumerate(obj):
                    _validate_bench_obj('halo', item, errors, 'benches.halo[{}]'.format(i))
        else:
            _validate_bench_obj(name, obj, errors, 'benches.{}'.format(name))

    missing = doc.get('missing_benches', [])
    if not isinstance(missing, list):
        errors.append('"missing_benches" must be a list')

    return errors


def merge(bench_files, metadata_file, host, date):
    """Build a merged profile document from raw per-bench JSON files.

    bench_files: list of paths to single-JSON-object files, each one
        line of output captured from a bench binary (halo may appear
        twice, once per placement tag -- both are folded into the
        "halo" list, keyed by the bench's own "bench" field, not by
        filename).
    metadata_file: path to a JSON file with the "metadata" fields
        (uname/lscpu/compiler+MPI versions/page size/node count).
    """
    benches = {}
    missing = []
    for path in bench_files:
        p = Path(path)
        if not p.exists():
            missing.append(p.stem)
            continue
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            missing.append('{} ({})'.format(p.name, exc))
            continue
        name = obj.get('bench')
        if name not in KNOWN_BENCHES:
            missing.append('{} (unrecognized bench {!r})'.format(p.name, name))
            continue
        if name == 'halo':
            benches.setdefault('halo', []).append(obj)
        else:
            benches[name] = obj

    for name in KNOWN_BENCHES:
        if name not in benches:
            missing.append(name)

    metadata = {}
    if metadata_file:
        mp = Path(metadata_file)
        if mp.exists():
            try:
                metadata = json.loads(mp.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise ProfileError('bad metadata file {}: {}'.format(metadata_file, exc))

    doc = {
        'schema_version': SCHEMA_VERSION,
        'host': host,
        'date': date,
        'metadata': metadata,
        'benches': benches,
        'missing_benches': missing,
    }
    return doc


def _fmt(x, digits=2):
    try:
        return '{:.{d}f}'.format(float(x), d=digits)
    except (TypeError, ValueError):
        return str(x)


def summarize(doc):
    """Return a list of human-readable summary lines for a merged profile."""
    lines = []
    host = doc.get('host', '?')
    date = doc.get('date', '?')
    lines.append('=== machine profile: {} ({}) ==='.format(host, date))

    meta = doc.get('metadata') or {}
    nnodes = meta.get('node_count')
    ncores = meta.get('ncores_per_node')
    if nnodes is not None or ncores is not None:
        lines.append('nodes={} cores_per_node={}'.format(nnodes, ncores))

    benches = doc.get('benches') or {}

    f = benches.get('freq')
    if f:
        lines.append('freq: 1-core {} GHz, all-core mean {} GHz (min {} / max {}) over {} threads'.format(
            _fmt(f.get('ghz_1core')), _fmt(f.get('ghz_allcores_mean')),
            _fmt(f.get('ghz_allcores_min')), _fmt(f.get('ghz_allcores_max')),
            f.get('nthreads')))

    s = benches.get('stream')
    if s:
        results = s.get('results') or []
        if results:
            best = max(results, key=lambda r: r.get('gbps', 0))
            lines.append('stream: triad peak {} GB/s @ {} threads (array {} MB, LLC assumed {} MB)'.format(
                _fmt(best.get('gbps')), best.get('threads'),
                _fmt(s.get('array_bytes_total', 0) / 1e6),
                _fmt(s.get('llc_bytes_assumed', 0) / 1e6)))

    k = benches.get('kernels')
    if k:
        adv, mp, tri = k.get('advection', {}), k.get('microphysics', {}), k.get('tridiag', {})
        lines.append('kernels ({}x{}x{}, {} threads full): '
                     'advection {}/{} ns/pt (1c/Nc), '
                     'microphysics {}/{} ns/pt, tridiag {}/{} ns/pt'.format(
            k.get('nx'), k.get('ny'), k.get('nz'), k.get('threads_full'),
            _fmt(adv.get('ns_per_point_single')), _fmt(adv.get('ns_per_point_full')),
            _fmt(mp.get('ns_per_point_single')), _fmt(mp.get('ns_per_point_full')),
            _fmt(tri.get('ns_per_point_single')), _fmt(tri.get('ns_per_point_full'))))

    halo_list = benches.get('halo') or []
    for h in halo_list:
        exch = h.get('exchange') or []
        pp = h.get('pingpong') or []
        exch_desc = ', '.join('patch={} {}us/{}GB/s'.format(
            e.get('patch'), _fmt(e.get('us_per_exchange')), _fmt(e.get('gbps'))) for e in exch)
        pp_desc = ''
        if pp:
            biggest = pp[-1]
            pp_desc = ' pingpong: {}us@8B .. {}GB/s@{}B'.format(
                _fmt(pp[0].get('us')), _fmt(biggest.get('gbps')), biggest.get('bytes'))
        lines.append('halo [{}] ({}x{} ranks): {}{}'.format(
            h.get('tag'), h.get('px'), h.get('py'), exch_desc, pp_desc))

    o = benches.get('omp')
    if o:
        fj = o.get('forkjoin') or []
        br = o.get('barrier') or []
        if fj:
            lines.append('omp: fork/join {}us @1 thread .. {}us @{} threads'.format(
                _fmt(fj[0].get('us_per_region')), _fmt(fj[-1].get('us_per_region')), fj[-1].get('threads')))
        if br:
            lines.append('omp: barrier {}us @1 thread .. {}us @{} threads'.format(
                _fmt(br[0].get('us_per_barrier')), _fmt(br[-1].get('us_per_barrier')), br[-1].get('threads')))

    io = benches.get('io')
    if io:
        sw = io.get('single_writer', {})
        cw = io.get('concurrent_writers', {})
        lines.append('io: single-writer {} GB/s; {} concurrent writers aggregate {} GB/s'.format(
            _fmt(sw.get('gbps')), cw.get('n'), _fmt(cw.get('aggregate_gbps'))))

    missing = doc.get('missing_benches') or []
    if missing:
        lines.append('missing/failed benches: {}'.format(', '.join(str(m) for m in missing)))

    errors = validate(doc)
    if errors:
        lines.append('SCHEMA VALIDATION ERRORS:')
        for e in errors:
            lines.append('  ' + e)

    return lines


def _cmd_merge(args):
    doc = merge(args.bench_json, args.metadata, args.host, args.date)
    errors = validate(doc)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(doc, indent=2))
    print('wrote {}'.format(out_path))
    if errors:
        print('WARNING: {} schema issue(s):'.format(len(errors)), file=sys.stderr)
        for e in errors:
            print('  ' + e, file=sys.stderr)
        return 1
    return 0


def _cmd_summary(args):
    doc = json.loads(Path(args.profile_json).read_text())
    for line in summarize(doc):
        print(line)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='machine/profile.py')
    sub = parser.add_subparsers(dest='action')

    p_merge = sub.add_parser('merge', help='merge raw per-bench JSON + metadata into one profile')
    p_merge.add_argument('bench_json', nargs='+', help='raw per-bench JSON files (one JSON object each)')
    p_merge.add_argument('--metadata', default=None, help='metadata JSON file')
    p_merge.add_argument('--host', required=True)
    p_merge.add_argument('--date', required=True)
    p_merge.add_argument('--out', required=True, help='output merged profile JSON path')

    p_sum = sub.add_parser('summary', help='print a human summary of a merged profile JSON')
    p_sum.add_argument('profile_json')

    args = parser.parse_args(argv)
    if args.action == 'merge':
        return _cmd_merge(args)
    elif args.action == 'summary':
        return _cmd_summary(args)
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
