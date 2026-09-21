#!/usr/bin/env python3
"""Read-only recovery/NetCDF readability checks; not scientific correctness validation."""
import argparse
import datetime
import json
from pathlib import Path
import sys

REPO = Path('/home/u5727520/wrf-stack/wtune')
sys.path.insert(0, str(REPO))
from plov.kf_node_g0_contract import file_record
from plov.prepare_isc21_node_g0 import MEMBERS, require, write_json
from tools.wrf_netcdf_bitwise import NetCDF, primitive_type, product


def inspect_netcdf(api, path, errors):
    result = {'path': str(path), 'variables': [], 'times': []}
    ncid = None
    try:
        ncid = api.open(path)
        ndims, nvars, natts, unlimited = api.inq(ncid)
        dims = api.dimensions(ncid, ndims)
        result.update(dimensions=[{'name': name.decode(), 'length': length, 'unlimited': unlim}
                                  for name, length, unlim in dims],
                      variable_count=nvars, global_attribute_count=natts,
                      first_unlimited_dimension_id=unlimited)
        known = {name.decode(): length for name, length, _ in dims}
        for name, expected in {'west_east': 792, 'south_north': 852, 'bottom_top': 50}.items():
            if known.get(name) != expected:
                errors.append('%s: dimension %s expected %d, found %s' % (path.name, name, expected, known.get(name)))
        if known.get('Time', 0) <= 0:
            errors.append(path.name + ': Time dimension is absent or empty')
        require(api.child_group_count(ncid) == 0, 'unexpected NetCDF child groups')
        for varid in range(nvars):
            item = {'id': varid}
            result['variables'].append(item)
            try:
                name, xtype, dimids, nattrs = api.variable(ncid, varid)
                label = name.decode('ascii')
                shape = tuple(dims[index][1] for index in dimids)
                typename, width = primitive_type(xtype)
                item.update(name=label, type=typename, shape=shape, attributes=nattrs)
                require(all(length > 0 for length in shape), 'empty variable')
                for position, start in [('first', (0,) * len(shape)),
                                        ('last', tuple(length - 1 for length in shape))]:
                    try:
                        item[position + '_hex'] = api.data(ncid, varid, start, (1,) * len(shape), width).hex()
                    except Exception as error:
                        errors.append('%s/%s/%s: %s' % (path.name, label, position, error))
                if label == 'Times':
                    require(xtype == 2 and len(shape) == 2 and dims[dimids[0]][0] == b'Time', 'unexpected Times schema')
                    require(product(shape) <= 1024 * 1024, 'Times exceeds 1 MiB bound')
                    raw = api.data(ncid, varid, (0, 0), shape, product(shape))
                    result['times'] = [raw[index:index + shape[1]].decode('ascii').rstrip('\0 ')
                                       for index in range(0, len(raw), shape[1])]
            except Exception as error:
                errors.append('%s/variable-%d: %s' % (path.name, varid, error))
        dates = [datetime.datetime.strptime(value, '%Y-%m-%d_%H:%M:%S') for value in result['times']]
        begin = datetime.datetime(2016, 8, 29, 9)
        end = begin + datetime.timedelta(minutes=5)
        require(dates and all(a < b for a, b in zip(dates, dates[1:])), 'Times missing or not strictly increasing')
        if path.name == 'wrfinput_d01':
            require(begin in dates, 'wrfinput Times lacks 2016-08-29_09:00:00')
        else:
            require(dates[0] <= begin and dates[-1] >= end, 'wrfbdy Times does not cover five-minute run')
    except Exception as error:
        errors.append(path.name + ': ' + str(error))
    finally:
        if ncid is not None:
            try:
                api.close(ncid)
            except Exception as error:
                errors.append(path.name + '/close: ' + str(error))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    draft = Path('/home/u5727520/wrf-stack/wtune-kf-g0-20260913-r1')
    parser.add_argument('--inputs', type=Path, default=draft / 'inputs')
    parser.add_argument('--plan', type=Path, default=draft / 'prepare-plan-v2.json')
    parser.add_argument('--library', default='/home/u5727520/wrf-stack/intel/lib/libnetcdf.so')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.report.exists(), 'report already exists')
    report = {'artifact_type': 'wtune_isc21_recovered_input_readability', 'version': 1,
              'status': 'failed', 'scope': 'Recovery identity, metadata, Times and endpoint reads only; no scientific correctness claim.',
              'geometry_basis': 'e_we=793/e_sn=853/e_vert=51 imply mass dimensions 792/852/50; other dimensions preserved.',
              'members': {}, 'netcdf': {}, 'errors': []}
    errors = report['errors']
    try:
        plan = json.loads(args.plan.read_text())
        recovery = json.loads((args.inputs / 'recovery.json').read_text())
        require(recovery.get('status') == 'complete' and recovery['plan'] == plan, 'incomplete recovery or wrong v2 plan')
        require(set(recovery['members']) == set(MEMBERS), 'recovery member inventory differs')
        report.update(plan=file_record(args.plan), recovery=file_record(args.inputs / 'recovery.json'))
        for name, expected in MEMBERS.items():
            try:
                record = file_record(args.inputs / name)
                report['members'][name] = record
                require(record['size'] == expected and recovery['members'][name] ==
                        {'bytes': expected, 'sha256': record['sha256']}, 'size/hash mismatch')
            except Exception as error:
                errors.append(name + ': ' + str(error))
        report['namelist_g0'] = file_record(args.inputs / 'namelist.g0')
        require(report['namelist_g0']['sha256'] == plan['namelist_sha256'] and
                (args.inputs / 'namelist.g0').read_text() == plan['namelist'], 'v2 namelist mismatch')
        api = NetCDF(args.library)
        report['library'] = dict(file_record(api.path), requested_path=args.library, version=api.version)
        report['netcdf'] = {name: inspect_netcdf(api, args.inputs / name, errors)
                            for name in ('wrfinput_d01', 'wrfbdy_d01')}
        report['status'] = 'verified-readable' if not errors else 'failed'
    except Exception as error:
        errors.append(str(error))
    write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'errors': errors, 'report': str(args.report)}))
    return 0 if report['status'] == 'verified-readable' else 1


if __name__ == '__main__':
    raise SystemExit(main())
