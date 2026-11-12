"""post/condensate.py: every hydrometeor counts, dry-mass water paths, cloud top, OLR Tb."""
import numpy as np
import pytest

from wtune.post import condensate as cd


class _Var:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def __getitem__(self, k):
        return self.a[k]


class _DS:
    def __init__(self, **kw):
        self.variables = {k: _Var(v) for k, v in kw.items()}


def _ds(nz=4, ny=2, nx=3, **q):
    dnw = np.full((1, nz), -1.0 / nz)
    mu = np.full((1, ny, nx), 1000.0)           # perturbation
    mub = np.full((1, ny, nx), 9000.0)          # base: total 10000 Pa of dry air
    ph = np.zeros((1, nz + 1, ny, nx))
    phb = (np.arange(nz + 1) * 1000.0 * cd.G)[None, :, None, None] * np.ones((1, 1, ny, nx))
    base = dict(DNW=dnw, MU=mu, MUB=mub, PH=ph, PHB=phb)
    for k, v in q.items():
        base[k] = np.broadcast_to(np.asarray(v, dtype=float), (1, nz, ny, nx)).copy()
    return _DS(**base)


def test_total_condensate_without_hydrometeors_raises():
    with pytest.raises(KeyError):
        cd.total_condensate(_ds())


def test_water_paths_use_dry_layer_mass():
    # 10000 Pa of dry air -> 10000/9.81 kg m-2; q = 1e-4 kg/kg everywhere -> 1e-4 * 1019.4 kg m-2
    ds = _ds(QCLOUD=1e-4, QICE=2e-4, QSNOW=1e-4, QGRAUP=1e-4)
    wp = cd.water_paths(ds)
    col = 10000.0 / cd.G
    assert wp["LWP"] == pytest.approx(np.full((2, 3), 1e-4 * col * 1000.0))
    assert wp["IWP"] == pytest.approx(np.full((2, 3), 2e-4 * col * 1000.0))
    assert wp["SWP"] == pytest.approx(np.full((2, 3), 2e-4 * col * 1000.0))
    assert "RWP" not in wp
    assert wp["TWP"] == pytest.approx(np.full((2, 3), 5e-4 * col * 1000.0))
