"""Make a same-date perturbed copy of a WRF history frame for a diffwrf positive control.
T[0,0,0,0] += 1.0 (gross) and QVAPOR[0,0,0,0] -> nextafter (1 ulp, tests bitwise detection)."""
import shutil, sys
import numpy as np
from netCDF4 import Dataset
src, dst = sys.argv[1:3]
shutil.copyfile(src, dst)
with Dataset(dst, "r+") as nc:
    t = nc.variables["T"]; v = t[0, 0, 0, 0]; t[0, 0, 0, 0] = v + 1.0
    q = nc.variables["QVAPOR"]; w = q[0, 0, 0, 0]
    q[0, 0, 0, 0] = np.nextafter(np.float32(w), np.float32(np.inf))
    print("T %r -> %r ; QVAPOR %r -> %r" % (v, t[0, 0, 0, 0], w, q[0, 0, 0, 0]))
