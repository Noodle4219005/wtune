#!/usr/bin/env python3
"""Generate a KF bridge only for the source and ordering audited by its core."""
import argparse
import json
from pathlib import Path

from gen_phase_schema import SchemaError, extract, write_atomic
from gen_ref_bridge import generate


def generate_checked(source, schema_path):
    with open(schema_path, encoding='utf-8') as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict) or schema.get('narg') != 79:
        raise SchemaError('KF transaction core requires 79 arguments')
    manifest = extract(source, 'KF_eta_CPS')
    if len(manifest['arguments']) != 79:
        raise SchemaError('KF source must also declare 79 arguments')
    if schema.get('source_sha256') != manifest['source']['sha256']:
        raise SchemaError('KF source hash differs from the audited schema')
    if schema.get('ordered_args') != [arg['name'] for arg in manifest['arguments']]:
        raise SchemaError('KF argument order differs from the audited schema')
    return generate(source, 'KF_eta_CPS', 'module_cu_kfeta_mp_kf_eta_cps_',
                    'wt_kf_bridge_dispatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--schema', default=str(Path(__file__).with_name('kf_cps_schema.json')))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = generate_checked(args.source, args.schema)
        write_atomic(args.output, result)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
