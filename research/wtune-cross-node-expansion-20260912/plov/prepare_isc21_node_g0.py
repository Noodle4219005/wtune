#!/usr/bin/env python3
"""Plan exact ISC21 recovery; material recovery requires --extract and a pinned plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

ARCHIVE = Path('/home/u5727520/wrf-bench/atlas/isc21/WRF_ISC21_3_DOMAIN.tar')
TARGET = Path('/home/u5727520/wrf-stack/wtune-kf-g0-20260913-r1')
PREFIX = 'WRF_ISC21_3_DOMAIN/'
MEMBERS = {'wrfbdy_d01': 964952328, 'wrfinput_d01': 1087552017,
           'namelist.input': 4179, 'qr_acr_qsV2.dat': 43764288,
           'freezeH2O.dat': 254944848, 'qr_acr_qgV2.dat': 89959776}
TABLES = ('qr_acr_qsV2.dat', 'freezeH2O.dat', 'qr_acr_qgV2.dat')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(st):
    return {k: getattr(st, 'st_' + k) for k in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def inventory(stream, expected):
    found = {}
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode='r:') as archive:
        for member in archive:
            require(not member.name.startswith('/') and '..' not in PurePosixPath(member.name).parts,
                    'unsafe archive member name')
            if member.name not in {PREFIX + name for name in expected}:
                continue
            name = member.name[len(PREFIX):]
            require(name not in found, 'duplicate archive member: ' + name)
            require(member.isreg() and not member.issparse(), 'nonregular/sparse member: ' + name)
            require(member.size == expected[name], 'unexpected member size: ' + name)
            found[name] = {'bytes': member.size, 'offset': member.offset_data}
    require(set(found) == set(expected), 'required archive members missing')
    return found


def read_member(stream, entry):
    stream.seek(entry['offset'])
    value = stream.read(entry['bytes'])
    require(len(value) == entry['bytes'], 'truncated archive member')
    return value


def namelist_g0(source):
    """Keep native namelist groups; archived text outside groups was never read by WRF."""
    groups = {}
    # Horizontal whitespace only: consuming newlines after an empty group name
    # skips its closing slash and merges the following group. Digits are valid
    # in names (the original archive contains an empty &grib2 group).
    for match in re.finditer(r'^[ \t]*&([a-zA-Z_]\w*)\b(.*?)^[ \t]*/[ \t]*(?:![^\n]*)?$',
                             source, re.M | re.S):
        name, body = match.group(1).lower(), match.group(2)
        require(name not in groups, 'duplicate namelist group')
        require(re.search(r'^[ \t]*&', body, re.M) is None, 'unclosed/nested namelist group')
        fields = {}
        matches = list(re.finditer(r'\b([a-zA-Z_]\w*)\s*=', body))
        for index, item in enumerate(matches):
            key = item.group(1).lower()
            require(key not in fields, 'duplicate namelist field: ' + key)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            value = re.sub(r'!.*', '', body[item.end():end]).strip().rstrip(',').strip()
            fields[key] = value
        groups[name] = fields
    require({'time_control', 'domains', 'physics'} <= set(groups), 'missing namelist groups')
    for group, key, expected in [('domains', 'time_step', 15), ('domains', 'e_we', 793),
                                 ('domains', 'e_sn', 853), ('domains', 'e_vert', 51),
                                 ('physics', 'cu_physics', 1), ('physics', 'bl_pbl_physics', 5),
                                 ('physics', 'sf_sfclay_physics', 5)]:
        require(int(groups[group][key].split(',')[0]) == expected, 'physical baseline mismatch: ' + key)
    time = groups['time_control']
    for key in list(time):
        if key.startswith('auxhist24_') or key in ('history_interval_s', 'iofields_filename'):
            del time[key]
    time.update(run_days='0', run_hours='0', run_minutes='5', run_seconds='0',
                start_year='2016', start_month='8', start_day='29', start_hour='9',
                start_minute='0', start_second='0', end_year='2016', end_month='8',
                end_day='29', end_hour='9', end_minute='5', end_second='0',
                history_interval='5', frames_per_outfile='1', io_form_input='2',
                io_form_boundary='2', io_form_history='2', io_form_auxhist24='0',
                restart='.false.', write_input='.false.', input_inname='"wrfinput_d<domain>"',
                bdy_inname='"wrfbdy_d<domain>"')
    groups['domains'].update(max_dom='1', nproc_x='7', nproc_y='8', numtiles='12',
                             use_adaptive_time_step='.false.')
    return ''.join('&' + group + '\n' + ''.join('  ' + k + ' = ' + v + ',\n'
                   for k, v in fields.items()) + '/\n\n' for group, fields in groups.items())


def make_plan(archive, output, expected=None):
    expected = MEMBERS if expected is None else expected
    require(not output.exists() and not output.is_symlink(), 'input output already exists')
    with archive.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'archive must be regular')
        members = inventory(stream, expected)
        require(members['namelist.input']['bytes'] <= 65536, 'namelist exceeds bound')
        raw = read_member(stream, members['namelist.input'])
        require(len(raw) <= 65536, 'namelist exceeds bound')
        edited = namelist_g0(raw.decode('ascii'))
        require(identity(before) == identity(os.fstat(stream.fileno())), 'archive changed during plan')
    return {'artifact_type': 'wtune_isc21_g0_recovery_plan', 'version': 1,
            'archive': str(archive.resolve()), 'archive_identity': identity(before),
            'output': str(output.absolute()), 'prefix': PREFIX, 'members': members,
            'total_extract_bytes': sum(expected.values()),
            'extract_archive_hash_read_bytes': before.st_size,
            'source_namelist_sha256': hashlib.sha256(raw).hexdigest(),
            'namelist': edited, 'namelist_sha256': hashlib.sha256(edited.encode()).hexdigest(),
            'status': 'plan-only', 'steps': 20, 'dt': 15,
            'note': 'No payload recovery performed; extraction hashes full archive and six outputs.'}


def extract(plan, expected=None):
    expected = MEMBERS if expected is None else expected
    require(plan['artifact_type'] == 'wtune_isc21_g0_recovery_plan' and plan['version'] == 1,
            'unknown plan schema')
    archive, output = Path(plan['archive']), Path(plan['output'])
    current = make_plan(archive, output, expected)
    require(current == plan, 'plan or source changed; no extraction')
    # Exclusive directory creation is the irreversible boundary. Failures are preserved, never retried.
    output.mkdir(mode=0o700)
    result = {'artifact_type': 'wtune_isc21_g0_recovery', 'version': 1, 'status': 'started',
              'plan': plan, 'members': {}}
    try:
        with archive.open('rb') as source:
            require(identity(os.fstat(source.fileno())) == plan['archive_identity'], 'archive replaced')
            for name, entry in plan['members'].items():
                source.seek(entry['offset'])
                digest, remaining = hashlib.sha256(), entry['bytes']
                with (output / name).open('xb') as dest:
                    while remaining:
                        block = source.read(min(8 * 1024 * 1024, remaining))
                        require(bool(block), 'truncated extraction: ' + name)
                        dest.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    dest.flush()
                    os.fsync(dest.fileno())
                (output / name).chmod(0o444)
                result['members'][name] = {'bytes': entry['bytes'], 'sha256': digest.hexdigest()}
            source.seek(0)
            digest = hashlib.sha256()
            remaining = plan['archive_identity']['size']
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                require(bool(block), 'truncated archive hash')
                digest.update(block)
                remaining -= len(block)
            result['archive_sha256'] = digest.hexdigest()
            require(identity(os.fstat(source.fileno())) == plan['archive_identity'], 'archive changed')
            require(identity(archive.stat()) == plan['archive_identity'], 'archive pathname replaced')
        require(result['members']['namelist.input']['sha256'] == plan['source_namelist_sha256'],
                'namelist changed')
        with (output / 'namelist.g0').open('x') as stream:
            stream.write(plan['namelist'])
        (output / 'namelist.g0').chmod(0o444)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        write_json(output / 'recovery.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=TARGET / 'inputs')
    parser.add_argument('--plan', type=Path, help='new small plan JSON; otherwise stdout')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--expected-plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.extract:
            require(args.expected_plan is not None and args.plan is None, '--extract requires --expected-plan')
            plan = json.loads(args.expected_plan.read_text())
            require(Path(plan['archive']) == args.archive.resolve() and Path(plan['output']) == args.output.absolute(),
                    'explicit paths do not match plan')
            extract(plan)
        else:
            require(args.expected_plan is None, '--expected-plan only with --extract')
            plan = make_plan(args.archive, args.output)
            if args.plan:
                args.plan.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.plan, plan)
            else:
                print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print('REFUSED: ' + str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
