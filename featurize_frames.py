#!/usr/bin/env python3
"""Reduce each wrfout frame to a 448 x n_feature rank matrix so the raw frame can be deleted.

A 901^2 x 45-level frame is ~1.1 GB (2.4 GB at 100 levels); the features it contributes are
~40 KB. Keeping 10 cases x 7 frames would need ~110 GB -- and copying them to wrf-results
would double that. That is what killed the first training attempt: the batch hit the GPFS
home1 quota and netCDF silently truncated every output into a header-only but structurally
valid file (see knowledge/hpc/2026-09-03-gpfs-quota-truncates-netcdf-silently.md).

Segmented on purpose: over a 12 h run the solar terminator sweeps 180 degrees, so one
whole-run feature vector is the mean of two opposite hot-spot patterns. Each frame is
featurised against its OWN utc hour and the segment length it represents, which is what
pairs it with the 5-minute label buckets from blockbar_bucket.

usage: featurize_frames.py <case_dir> <out_dir> [seg_hours] [--keep]
writes <out_dir>/features_<YYYY-MM-DD_HH>.csv   (448 rows + header)
deletes each wrfout frame once its CSV is on disk, unless --keep is given.
"""
import sys, os, glob, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wrf_frame
import wtune_oracle as oc

# Decomposition is a PARAMETER, not a constant: the geometry features (geo_i/geo_j/geo_nnb)
# carry roughly half the per-rank signal, and whether that prior survives a different
# nproc_x x nproc_y is unverified. Testing it requires reading a 14x64 run with the same code.
PX = int(os.environ.get('WTUNE_PX', 8))
PY = int(os.environ.get('WTUNE_PY', 56))
STRIDE = 2
RANK_KEYS = ('radiation', 'micro', 'surf')
# Turbulence / boundary-layer drivers, read straight from the frame.
# Motivation is a measured one, not a guess: with weather held EXACTLY fixed (same met_em,
# only the scheme changed), bl_pbl_physics=5 (MYNN) costs 14-26 points of hot-spot hit rate
# versus YSU/MYJ. MYNN is a TKE-PREDICTING scheme -- its work scales with how many levels sit
# inside the boundary layer and how hard the surface is forcing it -- and not one of the 11
# weather features describes any of that.
PBL_2D = ('PBLH', 'HFX', 'LH', 'UST')
# MYNN's own prognostic turbulence, present only when bl_pbl_physics=5. The surface proxies
# above did NOT close the MYNN hot-spot gap (18.6pp -> 21.3pp), and a repeat run showed the
# pattern is 93.3% reproducible -- so the information exists and these 2-D proxies simply are
# not it. MYNN iterates over the levels that are actively mixing, so the cost-shaped quantity
# is a LEVEL COUNT, not a surface flux.  QKE = 2*TKE, EL_PBL = mixing length.
QKE_MIN = 0.1                                   # m2/s2; below this a level is quiescent
FEAT_KEYS = ('TCWV', 'LWP', 'RWP', 'IWP', 'SWP', 'GWP', 'nspecies', 'mixed_phase')
STAMP = re.compile(r'wrfout_d01_(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2})')


def parse_stamp(path):
    m = STAMP.search(os.path.basename(path))
    if not m:
        raise ValueError("unparsable frame name: %s" % path)
    y, mo, d, h, mi = map(int, m.groups())
    doy = datetime.date(y, mo, d).timetuple().tm_yday
    # minutes belong in the tag: sub-hourly history would otherwise silently overwrite
    # (the first test on 15-min squall_line frames collapsed 9 frames into 3 CSVs)
    return doy, h + mi / 60.0, "%04d-%02d-%02d_%02d%02d" % (y, mo, d, h, mi)


def frame_matrix(wrfout, seg_hours):
    doy, utc0, _ = parse_stamp(wrfout)
    feats = oc.extract_features(wrfout, stride=STRIDE)
    fields = oc.phase_cost_fields(feats, doy, utc0, run_hours=seg_hours)
    cols = {}
    for k in RANK_KEYS:
        cols[k] = oc.to_ranks(fields[k], PX, PY).ravel()
    for k in FEAT_KEYS:
        v = feats.get(k)
        if v is not None:
            cols[k] = oc.to_ranks(np.asarray(v, float), PX, PY).ravel()
    with wrf_frame.open_frame(wrfout) as ds:
        # Contiguous read, stride in NumPy -- the rule wtune_oracle._read_strided_4d already
        # documents and that these additions violated: a multidimensional strided hyperslab
        # makes the NetCDF layer re-read/decompress far more than the slice it returns.
        # Ignoring it made featurisation ~10x slower than the solve it follows.
        for k in PBL_2D:
            if k in ds.variables:
                v = np.asarray(ds.variables[k][0, :, :], float)[::STRIDE, ::STRIDE]
                cols[k] = oc.to_ranks(v, PX, PY).ravel()
        if 'W' in ds.variables:                    # column max |w|: convective vigour
            # read a few WHOLE levels rather than a strided 4-D slice: netCDF4 serves a
            # contiguous 2-D slab fast, but a strided read across the level axis is ~6x slower
            # and dominated the whole featurisation step.
            nz = ds.variables['W'].shape[1]
            w = None
            for k in np.unique(np.linspace(2, max(2, nz - 2), 6).astype(int)):
                lev = np.abs(np.asarray(ds.variables['W'][0, int(k), :, :], float))[::STRIDE, ::STRIDE]
                w = lev if w is None else np.maximum(w, lev)
            cols['WMAX'] = oc.to_ranks(w, PX, PY).ravel()
        if 'QKE' in ds.variables:
            q = np.asarray(ds.variables['QKE'][0], float)[:, ::STRIDE, ::STRIDE]
            cols['QKEMAX'] = oc.to_ranks(q.max(axis=0), PX, PY).ravel()
            cols['QKEINT'] = oc.to_ranks(q.sum(axis=0), PX, PY).ravel()
            # the loop count MYNN actually pays for
            cols['QKELEV'] = oc.to_ranks((q > QKE_MIN).sum(axis=0).astype(float), PX, PY).ravel()
        if 'EL_PBL' in ds.variables:
            e = np.asarray(ds.variables['EL_PBL'][0], float)[:, ::STRIDE, ::STRIDE]
            cols['ELMAX'] = oc.to_ranks(e.max(axis=0), PX, PY).ravel()
        if 'PBLH' in cols and 'PH' in ds.variables and 'PHB' in ds.variables:
            # PBL depth as a fraction of the model column ~ the share of levels a TKE scheme
            # iterates over. PH alone is the PERTURBATION geopotential and can be negative --
            # using it gave PBLFRAC up to 1.375, i.e. a boundary layer taller than the model.
            top = (np.asarray(ds.variables['PH'][0, -1, :, :], float)[::STRIDE, ::STRIDE]
                   + np.asarray(ds.variables['PHB'][0, -1, :, :], float)[::STRIDE, ::STRIDE]) / 9.81
            cols['PBLFRAC'] = cols['PBLH'] / np.maximum(oc.to_ranks(top, PX, PY).ravel(), 1.0)
    names = list(cols)
    return names, np.column_stack([cols[n] for n in names])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    keep = '--keep' in sys.argv[1:]
    case_dir, out_dir = args[0], args[1]
    seg_hours = float(args[2]) if len(args) > 2 else 2.0
    os.makedirs(out_dir, exist_ok=True)
    # ADIOS2 writes each frame as a DIRECTORY (.bp), netCDF as a file -- accept both
    frames = sorted(glob.glob(os.path.join(case_dir, 'wrfout_d01_*')))
    if not frames:
        print("FATAL: no wrfout frames in %s" % case_dir); return 2
    ok = 0
    for f in frames:
        try:
            names, X = frame_matrix(f, seg_hours)
        except Exception as e:                      # a truncated frame must not kill the rest
            print("SKIP %s: %s" % (os.path.basename(f), e)); continue
        _, _, tag = parse_stamp(f)
        dest = os.path.join(out_dir, 'features_%s.csv' % tag)
        tmp = dest + '.tmp'
        np.savetxt(tmp, X, delimiter=',', header=','.join(names), comments='', fmt='%.6g')
        os.replace(tmp, dest)                       # atomic: never leave a half-written CSV
        ok += 1
        print("%s -> %s  X%s" % (os.path.basename(f), os.path.basename(dest), X.shape))
        if not keep:                                # only after the CSV is durably in place
            import shutil
            shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
    print("featurised %d/%d frames%s" % (ok, len(frames), "" if keep else ", raw frames deleted"))
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
