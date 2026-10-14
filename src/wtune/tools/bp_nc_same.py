"""bp_nc_same.py BP NC -- exit 0 when every variable of a WRF ADIOS2 history BP is in the netCDF
file with the same numbers (NaN where NaN); prints the first difference and exits 1 otherwise.
Runs under a python that has adios2 and netCDF4 (wtune.bp._run_adios2)."""
import sys

import adios2
import numpy as np
from netCDF4 import Dataset


def same(bp: str, nc: str) -> str:
    """'' when the files hold the same variables and numbers, else the first difference."""
    with adios2.FileReader(bp) as r, Dataset(nc) as d:
        d.set_auto_mask(False)
        steps = r.num_steps()
        for name, info in sorted(r.available_variables().items()):
            if name not in d.variables:
                return "%s: not in the netCDF file" % name
            want = np.asarray(d.variables[name][:])
            if want.dtype.kind in "SUO" or info.get("Type") == "string":
                continue                    # Times: the file name carries the time as well
            got = np.asarray(r.read(name, step_selection=[0, steps]) if steps > 1
                             else r.read(name))
            if got.size != want.size:
                return "%s: %d values in BP, %d in netCDF" % (name, got.size, want.size)
            a, b = got.ravel().astype(np.float64), want.ravel().astype(np.float64)
            if not np.array_equal(a, b, equal_nan=True):
                return "%s: values differ (max |diff| %g)" % (name, float(np.nanmax(np.abs(a - b))))
    return ""


if __name__ == "__main__":
    why = same(sys.argv[1], sys.argv[2])
    if why:
        print(why)
    sys.exit(1 if why else 0)
