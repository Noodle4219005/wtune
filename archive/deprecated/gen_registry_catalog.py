#!/usr/bin/env python3
"""Generate the COMPLETE WRF namelist parameter catalog from the Registry.

The Registry is the authoritative, version-matched source of every namelist
variable (README.namelist / the docx is only a documented subset). This parses
all Registry files' `rconfig ... namelist,<group> ...` lines into a JSON catalog
covering the FORECAST-RELEVANT groups (a normal wrf.exe run) -- WRFDA (wrfvar*),
chem, and fire are excluded as separate subsystems.

Usage: python3 gen_registry_catalog.py <WRF_Registry_dir> <out.json>
"""
import json
import re
import sys
from pathlib import Path

# groups a normal ARW forecast (wrf.exe / real.exe) actually reads
FORECAST_GROUPS = {
    'time_control', 'domains', 'physics', 'dynamics', 'bdy_control',
    'noah_mp', 'fdda', 'stoch', 'scm', 'dfi_control', 'diags', 'afwa',
    'namelist_quilt', 'grib2', 'tc', 'ideal', 'physics_suite',
}

# rconfig  <type>  <name>  namelist,<group>  <how_set>  <default>  [rest...]
RCONFIG = re.compile(
    r'^\s*rconfig\s+(\w+)\s+(\w+)\s+namelist,(\w+)\s+(\S+)\s+(\S+)(.*)$'
)
QUOTED = re.compile(r'"([^"]*)"')


def is_io_stream(name):
    """True for the per-stream I/O plumbing that inflates time_control."""
    n = name.lower()
    keys = ('auxinput', 'auxhist', 'inname', 'outname', 'frames_per',
            'iofields', '_outname', '_inname')
    if any(k in n for k in keys):
        return True
    if n.startswith('io_form_'):
        return True
    return False


def parse_registry(reg_dir):
    reg = Path(reg_dir)
    # Registry.EM_COMMON is the canonical ARW forecast registry -> read it (and
    # Registry.EM) FIRST so its defaults win the first-seen dedup.
    def prio(p):
        order = {'Registry.EM_COMMON': 0, 'Registry.EM': 1, 'Registry': 2}
        return (order.get(p.name, 3), p.name)
    files = sorted((p for p in reg.iterdir()
                    if p.is_file() and not p.name.endswith('.var')), key=prio)
    seen = {}   # name -> entry (first wins; canonical files read first)
    for f in files:
        try:
            text = f.read_text(errors='replace')
        except Exception:
            continue
        for line in text.splitlines():
            m = RCONFIG.match(line)
            if not m:
                continue
            typ, name, group, how_set, default, rest = m.groups()
            if group not in FORECAST_GROUPS:
                continue
            if name in seen:
                continue
            descs = QUOTED.findall(rest or '')
            # the longer quoted string is usually the human description
            meaning = ''
            for d in descs:
                if len(d) > len(meaning):
                    meaning = d
            seen[name] = {
                'name': name,
                'group': group,
                'type': typ,
                'per_domain': (how_set.lower() == 'max_domains'),
                'default': default.strip('"'),
                'meaning_en': meaning,
                'io_stream': is_io_stream(name),
                'registry_file': f.name,
            }
    return seen


def main():
    if len(sys.argv) != 3:
        print("usage: gen_registry_catalog.py <Registry_dir> <out.json>",
              file=sys.stderr)
        sys.exit(2)
    entries = parse_registry(sys.argv[1])
    groups = {}
    for e in entries.values():
        groups.setdefault(e['group'], []).append(e)
    for g in groups:
        groups[g].sort(key=lambda x: x['name'])

    # counts (real time_control excludes I/O-stream plumbing)
    counts = {}
    for g, lst in groups.items():
        real = [e for e in lst if not e['io_stream']]
        counts[g] = {'total': len(lst), 'real': len(real)}

    catalog = {
        'meta': {
            'source': 'WRF Registry (rconfig namelist entries)',
            'scope': 'forecast-relevant groups only (WRFDA/chem/fire excluded)',
            'registry_dir': sys.argv[1],
        },
        'counts': counts,
        'total_params': sum(len(v) for v in groups.values()),
        'groups': groups,
    }
    Path(sys.argv[2]).write_text(json.dumps(catalog, indent=1))
    print("wrote {} params across {} groups to {}".format(
        catalog['total_params'], len(groups), sys.argv[2]))
    for g in sorted(counts, key=lambda k: -counts[k]['total']):
        print("  {:16s} total={:4d}  real={:4d}".format(
            g, counts[g]['total'], counts[g]['real']))


if __name__ == '__main__':
    main()
