#!/usr/bin/env python3
"""Open a WRF frame as netCDF4 or ADIOS2 BP behind one interface.

Project policy (2026-09-04, user): every WRF experiment writes ADIOS2 -- `io_form_*=14`,
which the Registry binds to `io_adios2` and this WRF build already links (`-lwrfio_adios2`).
The analysis side has to read both, because the older training frames are netCDF and the new
ones are BP.

What this is NOT for: speed. Featurisation was ~30 min per case and is now ~25 s, and every
second of that came from replacing multidimensional strided hyperslabs with a contiguous read
plus a NumPy stride -- the trap `wtune_oracle._read_strided_4d` already documents. The format
was never the bottleneck. ADIOS2 is adopted here as the project's I/O standard on the WRF
WRITE path (measured -30% on history writes, which are 12.6% of loop wall), not as a fix for
a reader that was slow for its own reasons.

The adapter exposes the small slice of the netCDF4 surface the oracle uses:
`ds.variables` (a mapping), `ds.variables[name][...]`, `.shape`, and the context manager.
ADIOS2 has no lazy hyperslab of its own here, so a variable is read whole once and cached --
which is exactly the access pattern that turned out to be fastest for netCDF too.
"""
import os

__all__ = ['open_frame', 'is_bp']


def is_bp(path):
    """WRF's ADIOS2 engine writes a .bp DIRECTORY, not a file."""
    return os.path.isdir(path) or path.endswith('.bp')


class _Var:
    __slots__ = ('_a',)

    def __init__(self, arr):
        self._a = arr

    def __getitem__(self, key):
        return self._a[key]

    def __array__(self, dtype=None):
        return self._a if dtype is None else self._a.astype(dtype)

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype


class _BPVars:
    """Mapping that materialises a variable on first access and caches it."""

    def __init__(self, reader):
        self._r = reader
        self._names = set(reader.available_variables().keys())
        self._cache = {}

    def __contains__(self, k):
        return k in self._names

    def __iter__(self):
        return iter(self._names)

    def keys(self):
        return self._names

    def __getitem__(self, k):
        if k not in self._cache:
            if k not in self._names:
                raise KeyError(k)
            import numpy as np
            a = np.asarray(self._r.read(k))
            # ADIOS2 returns WRF fields WITHOUT the leading Time axis that netCDF carries:
            # PBLH is (900,900) here but (1,900,900) there, T is (44,900,900) vs
            # (1,44,900,900). Every reader in this codebase indexes [0, ...], so restore the
            # axis and both backends behave identically. Times comes back as a 0-d string.
            a = a.reshape(1) if a.ndim == 0 else a[np.newaxis, ...]
            self._cache[k] = _Var(a)
        return self._cache[k]

    def get(self, k, default=None):
        return self[k] if k in self._names else default


class BPDataset:
    def __init__(self, path):
        import adios2
        self._r = adios2.FileReader(path)
        self.variables = _BPVars(self._r)

    def close(self):
        try:
            self._r.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def open_frame(path):
    """-> a dataset exposing `.variables[name][...]`, for either backend."""
    if is_bp(path):
        return BPDataset(path)
    import netCDF4 as nc
    return nc.Dataset(path)
