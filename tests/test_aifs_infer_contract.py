"""Contract tests for scenario.generators.aifs_infer -- no GPU, no network, no torch model
load. Runs against synthetic data to lock down the parts of the module that a downstream
consumer (scenario.generators.aifs) relies on: the output netCDF's dims/coords/vars, and the
bbox/longitude-convention handling. Skips entirely if torch or xarray are unavailable (this
repo's default env is CPU-only; the aifs.* stack lives in envs/ai/pixi.toml).
"""

import datetime as dt
import os
import sys

import pytest

pytest.importorskip("xarray")
pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wtune.scenario.generators import aifs_infer as m  # noqa: E402


def test_parse_bbox_normalizes_order():
    assert m.parse_bbox(None) is None
    assert m.parse_bbox("55,20,-60,-130") == (20.0, 55.0, -130.0, -60.0)
    assert m.parse_bbox("20,55,-130,-60") == (20.0, 55.0, -130.0, -60.0)
    with pytest.raises(ValueError):
        m.parse_bbox("1,2,3")


def test_shift_lon_to_180_is_monotonic_and_reindexes_data():
    import numpy as np

    lon = np.arange(0, 360, 90.0)  # [0, 90, 180, 270]
    arr = np.array([[10.0, 20.0, 30.0, 40.0]])  # (1, 4), one row
    shifted, lon_shift = m._shift_lon_to_180(arr, lon)
    assert list(lon_shift) == [-180.0, -90.0, 0.0, 90.0]
    assert list(shifted[0]) == [30.0, 40.0, 10.0, 20.0]
    assert (lon_shift[1:] > lon_shift[:-1]).all()


def _synthetic_era5(n_lat=5, n_lon=8):
    import numpy as np
    import xarray as xr

    lat = np.linspace(40.0, 30.0, n_lat)  # descending, like ERA5
    lon = np.arange(0, 360, 360.0 / n_lon)  # 0..360 convention, like ARCO-ERA5
    levels = m.LEVELS_HPA

    data_vars = {}
    rng = np.random.default_rng(0)
    for long in set(m.SFC_PROGNOSTIC.values()) | set(m.CONST_FORCING.values()):
        data_vars[long] = (("lag", "lat", "lon"), rng.standard_normal((2, n_lat, n_lon)).astype("float32"))
    for long in set(m.PL_PROGNOSTIC.values()):
        data_vars[long] = (
            ("lag", "level", "lat", "lon"),
            rng.standard_normal((2, len(levels), n_lat, n_lon)).astype("float32"),
        )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lag": [0, 1], "level": levels, "lat": lat, "lon": lon},
    )
    return ds


def _synthetic_state(n_points=40):
    import numpy as np

    rng = np.random.default_rng(1)
    fields = {}
    for v in m.OUT_SFC_DYNAMIC:
        fields[v] = rng.standard_normal(n_points).astype("float32")
    for v in m.OUT_PL_VARS + ["w"]:
        for lev in m.LEVELS_HPA:
            fields[f"{v}_{lev}"] = rng.standard_normal(n_points).astype("float32")
    return {"fields": fields}


def test_assemble_dataset_matches_output_contract(monkeypatch):
    import numpy as np

    era5_ds = _synthetic_era5()
    states = [_synthetic_state() for _ in range(2)]  # pretend lead_h=12 (2 steps)

    # Stub the N320<->regular regrid matrix: identity-like reshape so we don't need the real
    # earthkit-regrid matrices (network + large sparse arrays) just to test dataset assembly.
    n_lat, n_lon = era5_ds.sizes["lat"], era5_ds.sizes["lon"]

    class _IdentityMatrix:
        def __matmul__(self, flat):
            # flat: (n_points_in, K) with n_points_in == len(state field) == 40 == n_lat*n_lon/...
            # just tile/broadcast to the right output size deterministically.
            k = flat.shape[1]
            out = np.zeros((n_lat * n_lon, k), dtype=np.float32)
            n = min(flat.shape[0], out.shape[0])
            out[:n] = flat[:n]
            return out

    monkeypatch.setattr(m, "_regrid_matrix", lambda direction: (_IdentityMatrix(), [n_lat, n_lon]))

    init_time = dt.datetime(2024, 9, 26, 0)
    ds = m.assemble_dataset(era5_ds, states, init_time, bbox=None)

    assert set(ds.dims) >= {"time", "level", "lat", "lon"}
    assert ds.sizes["time"] == 1 + len(states)
    assert list(ds["time"].values)[0] == np.datetime64(init_time)
    assert set(ds["level"].values) == set(m.LEVELS_HPA)

    for v in m.OUT_PL_VARS:
        assert ds[v].dims == ("time", "level", "lat", "lon")
    for v in m.OUT_SFC_DYNAMIC + m.OUT_SFC_STATIC:
        assert ds[v].dims == ("time", "lat", "lon")

    # 'w' is a required AIFS input but must never appear in the output.
    assert "w" not in ds.data_vars

    # static fields are replicated identically across every output time.
    for v in m.OUT_SFC_STATIC:
        for t in range(ds.sizes["time"] - 1):
            assert np.array_equal(ds[v].values[t], ds[v].values[t + 1])


def test_assemble_dataset_bbox_crops_with_no_margin(monkeypatch):
    import numpy as np

    era5_ds = _synthetic_era5(n_lat=10, n_lon=16)
    n_lat, n_lon = era5_ds.sizes["lat"], era5_ds.sizes["lon"]

    class _IdentityMatrix:
        def __matmul__(self, flat):
            k = flat.shape[1]
            out = np.zeros((n_lat * n_lon, k), dtype=np.float32)
            n = min(flat.shape[0], out.shape[0])
            out[:n] = flat[:n]
            return out

    monkeypatch.setattr(m, "_regrid_matrix", lambda direction: (_IdentityMatrix(), [n_lat, n_lon]))

    init_time = dt.datetime(2024, 9, 26, 0)
    ds_full = m.assemble_dataset(era5_ds, [], init_time, bbox=None)
    bbox = (32.0, 38.0, -50.0, -10.0)
    ds_cropped = m.assemble_dataset(era5_ds, [], init_time, bbox=bbox)

    assert ds_cropped.sizes["lat"] <= ds_full.sizes["lat"]
    assert ds_cropped.sizes["lon"] <= ds_full.sizes["lon"]
    assert bool(((ds_cropped["lat"].values >= bbox[0]) & (ds_cropped["lat"].values <= bbox[1])).all())
    assert bool(((ds_cropped["lon"].values >= bbox[2]) & (ds_cropped["lon"].values <= bbox[3])).all())
