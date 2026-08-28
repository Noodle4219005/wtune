"""wtune_rankfile.py -- turn an Oracle cost map into an applicable rank->core placement
for a ZERO-SOURCE over-decomposition load-balance experiment.

Mechanism (no WRF source touched):
  * namelist nproc_x*nproc_y = ranks*factor  (over-decompose -- namelist only)
  * launch with MPI over-subscription (ranks > cores)
  * bind ranks to cores per the map below (rankfile / affinity -- launcher only)

The map comes from the Oracle per-cell cost -> per-fine-rank load -> locality-preserving
balanced cut (wtune_placement.balance_assignment_local): heavy regions get FEWER ranks per
core, light regions MORE, each core owning a CONTIGUOUS (snake-order) local block so the
extra halo is intra-core / intra-node (cheap).

Usage:
  python3 wtune_rankfile.py <snapshot> <factor> <ncore> <doy> <utc0> <run_hours> [base_px base_py]
Outputs a CSV (fine_rank,core) + a summary (predicted imbalance, ranks/core, factor).
"""
import sys
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle as oc
import wtune_placement as pl


def generate(snapshot, factor, ncore, doy, utc0, run_hours, base_px=8, base_py=56, stride=4, out_csv=None):
    # over-decompose: keep the good base aspect, multiply px by `factor` (finer E-W where the
    # terminator/storm gradients live); py fixed.
    fpx, fpy = base_px * factor, base_py
    feats = oc.extract_features(snapshot, stride=stride)
    fields = oc.phase_cost_fields(feats, doy, utc0, run_hours=run_hours)
    cell = np.zeros_like(fields['micro'])
    for name, f in fields.items():
        cell += oc.PHASE_WEIGHT.get(name, 0.0) * (f / f.mean())
    loads = oc.to_ranks(cell, fpx, fpy).ravel()          # fine-rank loads, WRF order iy*px+ix
    nranks = len(loads)
    r2c, core_loads = pl.balance_assignment_local(loads, ncore, fpx, fpy)
    counts = pl.counts_per_core(r2c, ncore)
    # default contiguous block map, for the imbalance-recovered estimate
    per = int(np.ceil(nranks / ncore)); blk = np.zeros(ncore)
    for i, l in enumerate(loads):
        blk[min(i // per, ncore - 1)] += l
    info = {
        'snapshot': snapshot, 'factor': factor, 'nranks': nranks, 'ncore': ncore,
        'fine_px': fpx, 'fine_py': fpy,
        'ranks_per_core_min': int(min(counts)), 'ranks_per_core_max': int(max(counts)),
        'imbalance_default_block_pct': round(100 * (blk.max() / blk.mean() - 1), 2),
        'imbalance_locality_balanced_pct': round(100 * pl.imbalance(core_loads), 2),
    }
    if out_csv:
        with open(out_csv, 'w') as f:
            f.write('fine_rank,core\n')
            for r, c in enumerate(r2c):
                f.write('{},{}\n'.format(r, c))
        info['rankmap_csv'] = out_csv
    return info


if __name__ == '__main__':
    a = sys.argv
    snap = a[1]; factor = int(a[2]); ncore = int(a[3])
    doy = int(a[4]); utc0 = float(a[5]); rh = float(a[6])
    bpx = int(a[7]) if len(a) > 7 else 8; bpy = int(a[8]) if len(a) > 8 else 56
    out = a[9] if len(a) > 9 else '/home/u5727520/.claude/jobs/489fe64b/tmp/maria/rankmap_f{}.csv'.format(factor)
    info = generate(snap, factor, ncore, doy, utc0, rh, bpx, bpy, out_csv=out)
    for k, v in info.items():
        print('{:32s} {}'.format(k, v))
