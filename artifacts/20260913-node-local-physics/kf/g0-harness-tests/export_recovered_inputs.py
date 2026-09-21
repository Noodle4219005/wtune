#!/usr/bin/env python3
"""Export six already-verified recovery records; never read scientific payloads."""
import csv
import hashlib
import io
import json
from pathlib import Path

root = Path('/home/u5727520/wrf-stack/wtune-kf-g0-20260913-r1')
recovery_path, report_path = root / 'inputs/recovery.json', root / 'input-readability.json'
recovery, report = json.loads(recovery_path.read_text()), json.loads(report_path.read_text())
assert recovery['status'] == 'complete' and report['status'] == 'verified-readable' and not report['errors']
assert len(recovery['members']) == 6 and set(recovery['members']) == set(report['members'])
buffer = io.StringIO(newline='')
writer = csv.writer(buffer, lineterminator='\n')
writer.writerow(['member', 'path', 'bytes', 'sha256', 'archive_path', 'archive_sha256', 'recovery_path'])
for name, value in sorted(recovery['members'].items()):
    actual = report['members'][name]
    assert (actual['size'], actual['sha256']) == (value['bytes'], value['sha256'])
    assert actual['path'] == str(root / 'inputs' / name)
    writer.writerow([name, actual['path'], value['bytes'], value['sha256'],
                     recovery['plan']['archive'], recovery['archive_sha256'], str(recovery_path)])
data = buffer.getvalue().encode('utf-8')
outputs = [Path('/home/u5727520/my-llm-wiki/data/wtune/2026-09-13-isc21-g0-recovered-inputs.csv'),
           Path('/home/u5727520/wrf-stack/wtune/results/derived-data/2026-09-13-isc21-g0-recovered-inputs.csv')]
assert all(not path.exists() for path in outputs), 'refuse existing output'
for path in outputs:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
assert outputs[0].read_bytes() == outputs[1].read_bytes() == data
assert len(list(csv.DictReader(io.StringIO(data.decode())))) == 6
print(json.dumps({'rows': 6, 'csv_sha256': hashlib.sha256(data).hexdigest(),
    'outputs': list(map(str, outputs)), 'recovery_path': str(recovery_path), 'readability_path': str(report_path),
    'netcdf': {name: {'dimensions': info['dimensions'], 'variable_count': info['variable_count'], 'times': info['times']}
               for name, info in report['netcdf'].items()}}, indent=2))
