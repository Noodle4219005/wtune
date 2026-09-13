#!/usr/bin/env python3
"""Generate the checked-in C schema.  Standard library only."""
from __future__ import print_function
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--check'); p.add_argument('--output', default=os.path.join(ROOT,'wt_kf_cps_schema.h')); ns=p.parse_args()
    with open(os.path.join(ROOT,'kf_cps_schema.json')) as f: d=json.load(f)
    names=d['ordered_args']; abi=d['abi']
    if len(names)!=d['narg'] or len(set(names))!=len(names): raise SystemExit('schema names/count invalid')
    groups={}
    for trait, vals in abi.items():
        for name in vals:
            if name not in names or name in groups: raise SystemExit('bad/duplicate trait: '+name)
            groups[name]=trait
    if set(groups)!=set(names): raise SystemExit('schema does not cover every name exactly once')
    out=['/* generated; source sha256: %s */'%d['source_sha256'], '#ifndef WT_KF_CPS_SCHEMA_H','#define WT_KF_CPS_SCHEMA_H','#define WT_KF_NARG 79', 'enum wt_kf_arg {']
    out += ['  WT_KF_%s = %d,'%(n.upper(),i) for i,n in enumerate(names)]
    out += ['};','enum wt_kf_shape { WT_KF_SCALAR, WT_KF_2D, WT_KF_3D };','enum wt_kf_intent { WT_KF_IN, WT_KF_OUT, WT_KF_INOUT, WT_KF_NO_INTENT };','static const unsigned char wt_kf_shape[WT_KF_NARG] = {']
    for n in names:
        g=groups[n]; out.append('  %s,' % ('WT_KF_3D' if '_3d_' in g else 'WT_KF_2D' if '_2d_' in g else 'WT_KF_SCALAR'))
    out += ['};','static const unsigned char wt_kf_intent[WT_KF_NARG] = {']
    for n in names:
        g=groups[n]; out.append('  %s,' % ('WT_KF_NO_INTENT' if 'no_intent' in g else 'WT_KF_OUT' if g.endswith('_out') else 'WT_KF_INOUT' if 'inout' in g else 'WT_KF_IN'))
    out += ['};','static const unsigned char wt_kf_optional[WT_KF_NARG] = {']
    for n in names: out.append('  %d,' % int('optional' in groups[n]))
    out += ['};','#endif']
    text='\n'.join(out)+'\n'
    target=ns.check or ns.output
    if ns.check:
        try:
            with open(target) as f: actual=f.read()
        except IOError: return 1
        return 0 if actual==text else 1
    with open(target,'w') as f: f.write(text)
    return 0
if __name__=='__main__': sys.exit(main())
