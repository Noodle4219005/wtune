"""Fail-closed native KF G0 provenance, launch environment and evidence contracts."""
import hashlib
import json
import os
from pathlib import Path
import re
from plov.prepare_isc21_node_g0 import MEMBERS, require, sha256

RANKS = 56
PREFIXES = ('WTIMER_', 'WT_KF_', 'WT_MYNN_', 'WT_SFCLAY_', 'OMP_', 'KMP_', 'I_MPI_')
ATTESTATION = {'source_abi_verified', 'caller_chain_verified', 'baseline_closure_verified',
               'compiler_module_abi_verified', 'library_closure_verified'}
SUPPORT = ('GENPARM.TBL', 'LANDUSE.TBL', 'SOILPARM.TBL', 'VEGPARM.TBL',
           'RRTMG_LW_DATA', 'RRTMG_SW_DATA', 'ozone.formatted', 'ozone_lat.formatted',
           'ozone_plev.formatted', 'aerosol.formatted', 'aerosol_lat.formatted',
           'aerosol_lon.formatted', 'aerosol_plev.formatted', 'CAMtr_volume_mixing_ratio')


def file_record(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'not a regular file: ' + str(path))
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed during hash')
    return {'path': str(path), 'size': before.st_size, 'sha256': digest}


def check_record(record):
    require(file_record(record['path']) == record, 'file identity mismatch: ' + str(record.get('path')))


def candidate_identity(candidate, expected_hash, manifest_path, terminal_path):
    require(re.fullmatch('[0-9a-f]{64}', expected_hash) is not None, 'explicit candidate SHA256 required')
    manifest_record, terminal_record = file_record(manifest_path), file_record(terminal_path)
    manifest = json.loads(Path(manifest_path).read_text())
    terminal = json.loads(Path(terminal_path).read_text())
    require(manifest.get('artifact_type') == 'wtune_kf_build_recipe' and manifest.get('version') == 3,
            'unrecognized build recipe')
    require(manifest.get('executed') is False, 'immutable build recipe expected')
    require(set(manifest.get('attestation', {})) == ATTESTATION and
            all(value is True for value in manifest['attestation'].values()), 'build unattested')
    node = manifest.get('node_local_profile', {})
    expected_node = {'source_abi_verified': True, 'caller_chain_verified': True,
                     'init_verified': True, 'concurrency_verified': False}
    require(node.get('enabled') is True and node.get('name') == 'reviewed-g0-v1' and
            node.get('proof_activation_implemented') is True and
            set(node.get('attestation', {})) == set(expected_node) and
            all(node['attestation'][key] is value for key, value in expected_node.items()),
            'reviewed G0 activation missing')
    profile = node.get('profile_id', '')
    require(re.fullmatch(r'reviewed-g0-v1:[0-9a-f]{64}', profile) is not None, 'bad reviewed profile')
    require(re.fullmatch(r'[0-9a-f]{64}', manifest.get('plan_sha256', '')) is not None, 'bad build plan identity')
    require(sha256(node['review_record']['path']) == node['review_record']['sha256'], 'review changed')
    require(terminal.get('status') == 'complete' and terminal.get('link_executed') is True,
            'build/link not terminal complete')
    require(terminal.get('manifest_sha256') == manifest_record['sha256'] and
            terminal.get('plan_sha256') == manifest.get('plan_sha256'), 'terminal recipe mismatch')
    linked = terminal['linked_candidate']
    require(linked['path'] == str(Path(candidate).resolve()) and linked['sha256'] == expected_hash,
            'explicit candidate does not match terminal link')
    for key in ('linked_candidate', 'link_map', 'native_identity_record'):
        check_record(terminal[key])
    native = json.loads(Path(terminal['native_identity_record']['path']).read_text())
    require(native.get('artifact_type') == 'wtune_kf_linked_candidate_identity' and native.get('version') == 1,
            'unknown native identity')
    require(native.get('profile_id') == profile and native.get('candidate') == linked and
            native.get('link_map') == terminal['link_map'] and
            native.get('build_manifest_sha256') == manifest_record['sha256'] and
            native.get('plan_sha256') == manifest['plan_sha256'], 'native identity cross-link mismatch')
    # No compile-time record claims actual native ABI or numerical acceptance.
    return {'candidate': linked, 'manifest': manifest_record, 'terminal': terminal_record,
            'native_identity': terminal['native_identity_record'], 'profile_id': profile}


def inputs_identity(path):
    path = Path(path).resolve(strict=True)
    recovery = json.loads((path / 'recovery.json').read_text())
    require(recovery.get('artifact_type') == 'wtune_isc21_g0_recovery' and
            recovery.get('status') == 'complete', 'input recovery incomplete')
    require(set(recovery['members']) == set(MEMBERS), 'recovery members differ')
    records = {}
    for name, size in MEMBERS.items():
        entry, record = recovery['members'][name], file_record(path / name)
        require(not (path / name).is_symlink() and entry['bytes'] == size and record['size'] == size and
                record['sha256'] == entry['sha256'], 'input identity mismatch: ' + name)
        require((path / name).stat().st_mode & 0o222 == 0, 'recovered inputs must be read-only')
        records[name] = record
    records['namelist.g0'] = file_record(path / 'namelist.g0')
    require(records['namelist.g0']['sha256'] == recovery['plan']['namelist_sha256'], 'G0 namelist changed')
    records['recovery.json'] = file_record(path / 'recovery.json')
    return records


def arm_environment(arm, directory, inherited=None):
    require(arm in ('A', 'B', 'C'), 'unknown experiment arm')
    env = {key: value for key, value in (os.environ if inherited is None else inherited).items()
           if not key.startswith(PREFIXES)}
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(directory),
               OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE', OMP_STACKSIZE='512M',
               OMP_PROC_BIND='close', OMP_PLACES='cores', KMP_BLOCKTIME='0',
               I_MPI_PMI_LIBRARY='/usr/lib64/libpmi2.so', I_MPI_PIN='1', I_MPI_PIN_DOMAIN='omp',
               WTIMER_DIR=str(directory), WTIMER_STAGE_GRID='1', WTIMER_STAGE_MICRO='0',
               WTIMER_STAGE_RAD='0', WTIMER_STAGE_PBL='0', WTIMER_STAGE_SURFACE='0',
               WTIMER_STAGE_CUMULUS='0' if arm == 'A' else '1', WT_KF_NODE_G0='1',
               WT_KF_NODE_G0_OWNER_ONLY='1' if arm == 'B' else '0',
               WT_KF_NODE_G0_SHADOW='1' if arm == 'C' else '0', WT_KF_NODE_G0_TRACE='1',
               WT_KF_NODE_SLOT_BYTES=str(4 * 1024 ** 2), WT_KF_NODE_SHARED_BUDGET=str(1024 ** 3),
               WT_KF_BYTE_BUDGET=str(256 * 1024 ** 2), WT_KF_NODE_DEADLINE_MS='30000')
    return env


def launch_argv(python, runner, candidate):
    return ['timeout', '--signal=TERM', '--kill-after=20s', '420s', 'srun', '--mpi=pmi2',
            '--nodes=1', '--ntasks=56', '--ntasks-per-node=56', '--cpus-per-task=2',
            '--distribution=block:block', '--cpu-bind=cores', '--hint=nomultithread',
            str(python), str(runner), 'rank-exec', '--candidate', str(candidate)]


def fields(line):
    return dict(re.findall(r'([\w.]+)=([^\s]+)', line))


def one_line(text, prefix):
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, 'missing/duplicate diagnostic: ' + prefix)
    return fields(lines[0])


def numeric(record, key):
    require(key in record and re.fullmatch(r'[0-9]+', record[key]) is not None, 'missing/noninteger ' + key)
    return int(record[key])


def profile_number(profile):
    value = 14695981039346656037
    for byte in profile.encode():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def validate_arm(directory, arm, profile, ranks=RANKS):
    directory = Path(directory)
    error = (directory / 'rsl.error.0000').read_text(errors='replace')
    output = (directory / 'rsl.out.0000').read_text(errors='replace')
    require(error.count('SUCCESS COMPLETE WRF') == 1, 'missing/duplicate WRF success')
    require(len(re.findall(r'^Timing for main:', output, re.M)) == 20, 'expected exactly twenty WRF steps')
    for path in [directory / 'wrf.out', *directory.glob('rsl.*')]:
        with path.open(errors='replace') as stream:
            for line in stream:
                require(not re.search(r'(?i)segmentation|forrtl:|backtrace|SIGABRT|SIGSEGV|FATAL CALLED|ERROR ABORT|AddressSanitizer|runtime error:|CFL.*exceeded', line),
                        'fatal/log guard: ' + str(path))
    launches = sorted(directory.glob('launch.*.json'))
    csvs = sorted(directory.glob('wtimer.*.csv'))
    require(len(launches) == ranks and len(csvs) == ranks, 'rank evidence count mismatch')
    launch = {item['world_rank']: item for item in (json.loads(p.read_text()) for p in launches)}
    require(set(launch) == set(range(ranks)), 'world-rank identities incomplete')
    require({item['local_rank'] for item in launch.values()} == set(range(ranks)), 'local-rank identities incomplete')
    require(len({item['hostname'] for item in launch.values()}) == 1 and
            len({(item['job'], item['step']) for item in launch.values()}) == 1, 'not one native node/step')
    sums = {key: 0 for key in ('requested', 'donated', 'self', 'helper', 'committed', 'fallback', 'fatal',
                              'owner_remote_committed', 'shadow_checked', 'shadow_failed', 'shadow_bytes')}
    evidence, inits, timer_ranks = [], set(), set()
    for path in csvs:
        text = path.read_text()
        rank = numeric(one_line(text, '# rank='), 'rank')
        require(rank in launch and rank not in timer_ranks, 'unknown/duplicate timer rank')
        timer_ranks.add(rank)
        omp = one_line(text, '# omp_max_threads=')
        require(numeric(omp, 'omp_max_threads') == 2 and omp['omp_num_threads_env'] == '2', 'constructor not OMP2')
        flags = one_line(text, '# stage_flags ')
        require(flags == {key: arm_environment(arm, directory)[key] for key in flags} and
                set(flags) == {'WTIMER_STAGE_' + phase for phase in ('MICRO', 'RAD', 'CUMULUS', 'PBL', 'SURFACE')},
                'staging controls differ')
        bootstrap = one_line(text, '# wt_kf_bootstrap ')
        require(bootstrap['profile'] == profile, 'wrong compiled native profile')
        if arm == 'A':
            require(numeric(bootstrap, 'requested') == numeric(bootstrap, 'staged') == 0,
                    'baseline requested staging')
            # With stage_cumulus=0 the native bootstrap never configures the node backend.
            # Its absent dump is expected; requiring one would reject the real baseline.
            if not any(line.startswith('# wt_kf_node mode=') for line in text.splitlines()):
                evidence.append({'rank': rank, 'bootstrap': bootstrap, 'node': None,
                                 'identity': None, 'refusals': {}})
                continue
        node, ident = one_line(text, '# wt_kf_node mode='), one_line(text, '# wt_kf_node_identity ')
        require(numeric(ident, 'pid') == launch[rank]['pid'], 'timer PID not launched native process')
        for key in sums:
            sums[key] += numeric(ident if key == 'owner_remote_committed' else node, key)
        refusals = {}
        for line in text.splitlines():
            if line.startswith('# wt_kf_node refusal.'):
                for key, value in fields(line).items():
                    require(value.isdecimal() and key not in refusals, 'invalid refusal diagnostics')
                    refusals[key] = int(value)
        require(len(refusals) == 13, 'incomplete refusal reason census')
        require(sum(refusals.values()) == numeric(node, 'fallback'), 'unexplained fallback count')
        if arm != 'A':
            require(numeric(bootstrap, 'configured') == 1 and numeric(node, 'requested') > 0,
                    'unconfigured or unentered native backend')
            require(numeric(ident, 'local_rank') == launch[rank]['local_rank'] and
                    numeric(ident, 'grid') == 1 and numeric(ident, 'last_ktau') > 0,
                    'native grid/epoch/rank identity mismatch')
            require(numeric(node, 'profile') == profile_number(profile) and numeric(node, 'init') > 0,
                    'native scientific identity missing')
            inits.add(node['init'])
            require(node['mode'] == ('forced-g0-owner-only' if arm == 'B' else 'forced-g0') and
                    numeric(node, 'shadow_remote') == (arm == 'C') and numeric(node, 'serialized_native') == 1,
                    'wrong G0/concurrency mode')
            require(numeric(node, 'donated') == numeric(node, 'committed') and
                    numeric(node, 'committed') == numeric(node, 'self') + numeric(ident, 'owner_remote_committed'),
                    'owner exact-once accounting mismatch')
            require(numeric(node, 'requested') == numeric(node, 'donated') + numeric(node, 'fallback'),
                    'unaccounted native dispatch')
        evidence.append({'rank': rank, 'bootstrap': bootstrap, 'node': node, 'identity': ident, 'refusals': refusals})
    require(sums['fatal'] == 0 and sums['shadow_failed'] == 0, 'native failure counter')
    if arm == 'A':
        require(sums['donated'] == sums['helper'] == sums['committed'] == 0, 'baseline staged work')
    else:
        require(len(inits) == 1 and sums['committed'] > 0, 'no commit or mismatched initialization')
        if arm == 'B':
            require(sums['self'] > 0 and sums['helper'] == sums['owner_remote_committed'] == sums['shadow_checked'] == 0,
                    'owner-only control performed helper work')
        else:
            require(sums['owner_remote_committed'] > 0 and
                    sums['owner_remote_committed'] == sums['helper'] == sums['shadow_checked'],
                    'zero steal or remote/helper/shadow accounting mismatch')
    output_paths = list(directory.glob('wrfout*'))
    require(all(path.is_file() and not path.is_symlink() and re.fullmatch(
        r'wrfout_d01_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', path.name) for path in output_paths),
        'unexpected or nonregular WRF output')
    outputs = sorted(path.name for path in output_paths)
    require(len(outputs) >= 2 and 'wrfout_d01_2016-08-29_09:00:00' in outputs and
            'wrfout_d01_2016-08-29_09:05:00' in outputs, 'initial/final WRF outputs missing')
    return {'arm': arm, 'sums': sums, 'ranks': evidence, 'launches': launch, 'outputs': outputs}
