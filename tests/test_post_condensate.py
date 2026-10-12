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


def test_total_condensate_sums_every_species_present_and_clips_negatives():
    ds = _ds(QCLOUD=1e-4, QSNOW=3e-4, QGRAUP=-1e-6)
    assert cd.species_present(ds) == ["QCLOUD", "QSNOW", "QGRAUP"]
    assert np.allclose(cd.total_condensate(ds), 4e-4)


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


def test_hybrid_coordinate_c1h_c2h_used_when_present():
    ds = _ds(QCLOUD=1e-4)
    ds.variables["C1H"] = _Var(np.full((1, 4), 0.5))
    ds.variables["C2H"] = _Var(np.full((1, 4), 2000.0))
    dm = cd.layer_dry_mass(ds)
    assert dm[0, 0, 0] == pytest.approx((0.5 * 10000.0 + 2000.0) * 0.25 / cd.G)


def test_cloud_top_height_highest_level_over_threshold():
    q = np.zeros((1, 4, 2, 3))
    q[0, :3, 0, 0] = 1e-4          # cloudy through level 2 -> top at level 2 mid-height 2500 m
    q[0, 1, 1, 2] = 1e-6           # below threshold
    ds = _ds(QCLOUD=q)
    top = cd.cloud_top_height(ds)
    assert top[0, 0] == pytest.approx(2500.0)
    assert np.isnan(top[1, 2])


def test_olr_brightness_temperature():
    ds = _DS(OLR=np.full((1, 2, 2), cd.SIGMA_SB * 250.0 ** 4))
    assert np.allclose(cd.olr_brightness_temperature(ds), 250.0)
