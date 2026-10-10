"""ECMWF AIFS-single-1.0 inference generator.

CLI:
    python -m scenario.generators.aifs_infer \
        --init YYYY-MM-DDTHH --lead-h N --out FILE.nc \
        [--cache DIR] [--bbox lat0,lat1,lon0,lon1] [--device cuda|cpu]

Pipeline
--------
1. Fetch ERA5 initial conditions for t-6h and t0 from the public, key-free Google
   ARCO-ERA5 zarr store (gs://gcp-public-data-arco-era5), on its native 0.25 deg
   regular lat/lon grid.
2. Regrid those fields to AIFS's native N320 reduced-Gaussian octahedral grid
   (via earthkit-regrid, the same interpolation ECMWF's own AIFS notebook uses).
3. Run ECMWF's `ecmwf/aifs-single-1.0` checkpoint (anemoi-inference `SimpleRunner`)
   autoregressively out to `--lead-h` hours, 6h per step.
4. Regrid each forecast step back to the original 0.25 deg regular grid, crop to
   `--bbox` if given, and write a netCDF file.

Output netCDF contract (consumed by another module -- keep exact)
-------------------------------------------------------------------
Dims: (time, level, lat, lon).
Coords:
    time  -- CF datetime, every 6h from --init to --init+--lead-h inclusive
             (index 0 is the ERA5 analysis at --init itself, not a model output).
    level -- hPa, AIFS's 13 native levels: 50,100,150,200,250,300,400,500,600,
             700,850,925,1000 (a superset of the "at least" list in the contract).
    lat, lon -- degrees; lon in -180..180 convention.
Pressure-level vars (dims time,level,lat,lon): t [K], u, v [m s-1], q [kg kg-1],
    z [m2 s-2]. ('w' is a required AIFS *input* but is intentionally not written,
    per contract.)
Single-level vars (dims time,lat,lon): 2t, 10u, 10v, msl, sp, skt, lsm, z_sfc
    [m2 s-2]. lsm and z_sfc are ERA5-static (constant AIFS forcings) and are
    replicated across every output time rather than re-fetched or forecast.
Global attrs: model, model_version, weights_sha256, license, ic_source,
    init_time, created_utc, elapsed_inference_s.

No conversions are applied to any variable: ARCO-ERA5 and AIFS both already use
K, m/s, kg/kg, Pa and m2 s-2 (geopotential, not geopotential height) natively.
Inputs are deterministic: ERA5 analysis fields at fixed timestamps, no sampling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time

LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# AIFS short name -> ARCO-ERA5 long name, single-level prognostic fields.
SFC_PROGNOSTIC = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2d": "2m_dewpoint_temperature",
    "2t": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "tcw": "total_column_water",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
}

# AIFS short name -> ARCO-ERA5 long name, required constant forcings (time-invariant).
CONST_FORCING = {
    "lsm": "land_sea_mask",
    "z": "geopotential_at_surface",
    "slor": "slope_of_sub_gridscale_orography",
    "sdor": "standard_deviation_of_orography",
}

# AIFS variable prefix -> ARCO-ERA5 long name, pressure-level prognostic fields
# (each expands to "{prefix}_{level}" for level in LEVELS_HPA).
PL_PROGNOSTIC = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "z": "geopotential",
}

ARCO_ERA5_URL = "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

OUT_PL_VARS = ["t", "u", "v", "q", "z"]
OUT_SFC_DYNAMIC = ["2t", "10u", "10v", "msl", "sp", "skt"]
OUT_SFC_STATIC = ["lsm", "z_sfc"]

HF_REPO = "ecmwf/aifs-single-1.0"
MODEL_LICENSE = "CC-BY-4.0"
MODEL_NAME = "AIFS-single"
MODEL_VERSION = "1.0"


def log(msg: str) -> None:
    print(f"[aifs_infer] {msg}", file=sys.stderr, flush=True)


def parse_bbox(s: str | None):
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be lat0,lat1,lon0,lon1")
    lat0, lat1, lon0, lon1 = parts
    return (min(lat0, lat1), max(lat0, lat1), min(lon0, lon1), max(lon0, lon1))


def fetch_era5_slice(init_time: dt.datetime, cache_dir: str):
    """Return an xarray.Dataset with dims (lag, level, lat, lon) / (lag, lat, lon),
    lag=0 is init_time-6h, lag=1 is init_time, on the native 0.25 deg regular grid,
    containing exactly the fields AIFS needs. Cached to netCDF under
    <cache_dir>/era5/<init_iso>.nc.
    """
    import numpy as np
    import xarray as xr

    cache_path = os.path.join(cache_dir, "era5", f"{init_time:%Y%m%dT%H}.nc")
    if os.path.exists(cache_path):
        log(f"ERA5 cache hit: {cache_path}")
        return xr.load_dataset(cache_path)

    import gcsfs

    log(f"fetching ERA5 from {ARCO_ERA5_URL} (anonymous GCS)")
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(ARCO_ERA5_URL)
    ds = xr.open_zarr(mapper, consolidated=True, chunks=None)

    times64 = [np.datetime64(init_time - dt.timedelta(hours=6)), np.datetime64(init_time)]
    long_names = sorted(set(SFC_PROGNOSTIC.values()) | set(CONST_FORCING.values()) | set(PL_PROGNOSTIC.values()))

    sub = ds[long_names].sel(time=times64, level=LEVELS_HPA)
    sub = sub.load()
    sub = sub.rename({"latitude": "lat", "longitude": "lon", "time": "lag"})
    sub = sub.assign_coords(lag=[0, 1])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp{os.getpid()}"
    sub.to_netcdf(tmp)
    os.replace(tmp, cache_path)
    return sub


def _regrid_matrix(direction: str):
    from earthkit.regrid.db import find

    if direction == "to_n320":
        return find({"grid": (0.25, 0.25)}, {"grid": "N320"}, "linear")
    return find({"grid": "N320"}, {"grid": (0.25, 0.25)}, "linear")


def _regrid_batch(matrix, out_shape, batch):
    """batch: (K, *in_shape) -> (K, *out_shape), via one sparse matmul for all K fields."""
    import numpy as np

    k = batch.shape[0]
    flat = np.asarray(batch, dtype=np.float32).reshape(k, -1)
    out = np.asarray(matrix @ flat.T, dtype=np.float32).T
    return out.reshape((k,) + tuple(out_shape))


def build_input_fields(era5_ds) -> dict:
    """Regrid the loaded ERA5 slice to AIFS's N320 grid and assemble the `fields`
    dict expected by anemoi-inference's SimpleRunner: name -> ndarray (2, 542080).
    """
    matrix, out_shape = _regrid_matrix("to_n320")

    names, arrays = [], []
    for short, long in SFC_PROGNOSTIC.items():
        names.append(short)
        arrays.append(era5_ds[long].transpose("lag", "lat", "lon").values)
    for short, long in CONST_FORCING.items():
        names.append(short)
        arrays.append(era5_ds[long].transpose("lag", "lat", "lon").values)
    for prefix, long in PL_PROGNOSTIC.items():
        da = era5_ds[long]
        for lev in LEVELS_HPA:
            names.append(f"{prefix}_{lev}")
            arrays.append(da.sel(level=lev).transpose("lag", "lat", "lon").values)

    import numpy as np

    stacked = np.stack(arrays, axis=0)  # (n_fields, 2, lat, lon)
    n_fields = stacked.shape[0]
    flat_batch = stacked.reshape(n_fields * 2, *stacked.shape[2:])
    regridded = _regrid_batch(matrix, out_shape, flat_batch)  # (n_fields*2, 542080)
    regridded = regridded.reshape(n_fields, 2, -1)

    return {name: regridded[i] for i, name in enumerate(names)}


def load_runner(cache_dir: str, device: str, checkpoint_repo: str = HF_REPO):
    """Construct the SimpleRunner and force the weights onto `device` now (this is
    the "model load" phase); returns (runner, model_load_seconds).
    """
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
    from anemoi.inference.runners.simple import SimpleRunner

    t0 = time.time()
    runner = SimpleRunner({"huggingface": checkpoint_repo}, device=device)
    _ = runner.model  # cached_property: torch.load(...).to(device) happens here
    model_load_s = time.time() - t0
    return runner, model_load_s


def run_forecast(runner, fields: dict, init_time: dt.datetime, lead_h: int):
    """Run the autoregressive rollout. Returns (forecast_states, inference_seconds,
    peak_gpu_bytes). forecast_states is the list of per-timestep State dicts yielded
    by SimpleRunner.run (t0+6h .. t0+lead_h), NOT including the t0 analysis itself.
    """
    import inspect

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    input_state = dict(date=init_time, fields=fields)
    # anemoi-inference has used both `input_state=` (<=0.4.x, singular) and `input_states=`
    # (>=0.7ish, plural, multi-dataset) for this kwarg; pick whichever the installed
    # SimpleRunner.run actually has rather than pinning to one API generation.
    run_params = inspect.signature(runner.run).parameters
    kwarg = "input_states" if "input_states" in run_params else "input_state"
    t0 = time.time()
    states = list(runner.run(**{kwarg: input_state}, lead_time=lead_h))
    inference_s = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return states, inference_s, peak_mem


def _shift_lon_to_180(arr, lon):
    """arr: (..., lon) on the 0..359.75 axis -> (..., lon) on -180..179.75, monotonic."""
    import numpy as np

    n = lon.shape[0]
    half = n // 2
    lon_shift = np.concatenate([lon[half:] - 360.0, lon[:half]])
    arr_shift = np.concatenate([arr[..., half:], arr[..., :half]], axis=-1)
    return arr_shift, lon_shift


def assemble_dataset(era5_ds, forecast_states: list, init_time: dt.datetime, bbox):
    """Build the contract-shaped xarray.Dataset from the ERA5 analysis (time=0) and
    the model forecast states (time=6h..lead_h), regridded back to the regular grid.
    """
    import numpy as np
    import xarray as xr

    lat = era5_ds["lat"].values
    lon0_360 = era5_ds["lon"].values
    n_times = 1 + len(forecast_states)
    times = [np.datetime64(init_time) + np.timedelta64(6 * i, "h") for i in range(n_times)]

    matrix, out_shape = _regrid_matrix("to_regular")

    pl = {v: np.empty((n_times, len(LEVELS_HPA), len(lat), len(lon0_360)), dtype=np.float32) for v in OUT_PL_VARS}
    sfc = {v: np.empty((n_times, len(lat), len(lon0_360)), dtype=np.float32) for v in OUT_SFC_DYNAMIC}

    # time index 0: ERA5 analysis at init_time (lag=1), already on the regular grid.
    for v in OUT_PL_VARS:
        long = PL_PROGNOSTIC[v]
        for li, lev in enumerate(LEVELS_HPA):
            pl[v][0, li] = era5_ds[long].sel(level=lev, lag=1).transpose("lat", "lon").values
    for v in OUT_SFC_DYNAMIC:
        long = SFC_PROGNOSTIC[v]
        sfc[v][0] = era5_ds[long].sel(lag=1).transpose("lat", "lon").values

    # time index 1..: model forecast states, regridded N320 -> regular grid, batched.
    if forecast_states:
        names = []
        for v in OUT_PL_VARS:
            for lev in LEVELS_HPA:
                names.append(("pl", v, lev))
        for v in OUT_SFC_DYNAMIC:
            names.append(("sfc", v, None))

        for ti, state in enumerate(forecast_states, start=1):
            f = state["fields"]
            batch = np.stack(
                [
                    (f[f"{v}_{lev}"] if kind == "pl" else f[v])
                    for kind, v, lev in names
                ],
                axis=0,
            )
            regridded = _regrid_batch(matrix, out_shape, batch)
            for (kind, v, lev), field in zip(names, regridded):
                if kind == "pl":
                    li = LEVELS_HPA.index(lev)
                    pl[v][ti, li] = field
                else:
                    sfc[v][ti] = field

    # static fields, replicated across all output times.
    lsm_reg = era5_ds[CONST_FORCING["lsm"]].sel(lag=1).transpose("lat", "lon").values
    zsfc_reg = era5_ds[CONST_FORCING["z"]].sel(lag=1).transpose("lat", "lon").values
    static = {
        "lsm": np.broadcast_to(lsm_reg, (n_times,) + lsm_reg.shape).astype(np.float32),
        "z_sfc": np.broadcast_to(zsfc_reg, (n_times,) + zsfc_reg.shape).astype(np.float32),
    }

    # lon convention: shift 0..359.75 -> -180..179.75.
    all_arrays = {}
    lon_shift = None
    for v, arr in {**pl, **sfc, **static}.items():
        shifted, lon_shift = _shift_lon_to_180(arr, lon0_360)
        all_arrays[v] = shifted

    lat_arr, lon_arr = lat, lon_shift
    if bbox is not None:
        lat0, lat1, lon0, lon1 = bbox
        lat_mask = (lat_arr >= lat0) & (lat_arr <= lat1)
        lon_mask = (lon_arr >= lon0) & (lon_arr <= lon1)
        lat_arr = lat_arr[lat_mask]
        lon_arr = lon_arr[lon_mask]
        for v in list(all_arrays):
            arr = all_arrays[v]
            arr = arr[..., lat_mask, :]
            arr = arr[..., :, lon_mask]
            all_arrays[v] = arr

    data_vars = {}
    for v in OUT_PL_VARS:
        data_vars[v] = (("time", "level", "lat", "lon"), all_arrays[v])
    for v in OUT_SFC_DYNAMIC + OUT_SFC_STATIC:
        data_vars[v] = (("time", "lat", "lon"), all_arrays[v])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "level": np.array(LEVELS_HPA, dtype=np.int32),
            "lat": lat_arr,
            "lon": lon_arr,
        },
    )
    ds["level"].attrs["units"] = "hPa"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["t"].attrs["units"] = "K"
    ds["u"].attrs["units"] = "m s-1"
    ds["v"].attrs["units"] = "m s-1"
    ds["q"].attrs["units"] = "kg kg-1"
    ds["z"].attrs["units"] = "m2 s-2"
    ds["z_sfc"].attrs["units"] = "m2 s-2"
    ds["msl"].attrs["units"] = "Pa"
    ds["sp"].attrs["units"] = "Pa"
    ds["2t"].attrs["units"] = "K"
    ds["skt"].attrs["units"] = "K"
    ds["10u"].attrs["units"] = "m s-1"
    ds["10v"].attrs["units"] = "m s-1"
    ds["lsm"].attrs["units"] = "1"
    return ds


def sanity_check(ds) -> dict:
    """Cheap, non-fatal range/NaN checks over the written domain. Returns a dict of
    stats (also logged) -- does not raise, callers decide what to do with violations.
    """
    import numpy as np

    stats = {}
    t = ds["t"].values
    msl = ds["msl"].values
    z500 = ds["z"].sel(level=500).values / 9.80665
    stats["nan_count"] = int(np.isnan(t).sum() + np.isnan(msl).sum() + np.isnan(z500).sum())
    stats["t_min"], stats["t_max"] = float(np.nanmin(t)), float(np.nanmax(t))
    stats["msl_min"], stats["msl_max"] = float(np.nanmin(msl)), float(np.nanmax(msl))
    stats["z500_gpm_min"], stats["z500_gpm_max"] = float(np.nanmin(z500)), float(np.nanmax(z500))
    stats["t_range_ok"] = 180.0 <= stats["t_min"] and stats["t_max"] <= 330.0
    stats["msl_range_ok"] = 95000.0 <= stats["msl_min"] and stats["msl_max"] <= 105000.0
    stats["z500_range_ok"] = 5000.0 <= stats["z500_gpm_min"] and stats["z500_gpm_max"] <= 6000.0
    log(f"sanity: {json.dumps(stats)}")
    return stats


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True, help="YYYY-MM-DDTHH, the t0 analysis time")
    p.add_argument("--lead-h", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=os.environ.get("AIFS_CACHE", os.path.expanduser("~/wtune-runs/aifs/cache")))
    p.add_argument("--bbox", default=None, help="lat0,lat1,lon0,lon1")
    p.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = p.parse_args(argv)

    timings = {}
    t0 = time.time()
    import torch  # noqa: F401  (forces the heavy import set to be timed together)
    import xarray  # noqa: F401
    import anemoi.inference  # noqa: F401

    timings["import_s"] = time.time() - t0

    init_time = dt.datetime.strptime(args.init, "%Y-%m-%dT%H")
    bbox = parse_bbox(args.bbox)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} init={init_time.isoformat()} lead_h={args.lead_h} bbox={bbox}")

    t0 = time.time()
    era5_ds = fetch_era5_slice(init_time, args.cache)
    input_fields = build_input_fields(era5_ds)
    timings["fetch_s"] = time.time() - t0

    runner, model_load_s = load_runner(args.cache, device)
    timings["model_load_s"] = model_load_s

    forecast_states, inference_s, peak_mem = run_forecast(runner, input_fields, init_time, args.lead_h)
    timings["inference_s"] = inference_s

    t0 = time.time()
    ds = assemble_dataset(era5_ds, forecast_states, init_time, bbox)
    weights_path = runner.checkpoint.path
    weights_sha256 = sha256_file(weights_path)
    ds.attrs.update(
        {
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "weights_sha256": weights_sha256,
            "license": MODEL_LICENSE,
            "ic_source": f"ERA5 (ARCO, {ARCO_ERA5_URL})",
            "init_time": init_time.isoformat(),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_inference_s": f"{inference_s:.3f}",
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ds.to_netcdf(args.out)
    timings["write_s"] = time.time() - t0

    stats = sanity_check(ds)

    result = {
        "timings_s": timings,
        "gpu_peak_mem_bytes": int(peak_mem),
        "weights_sha256": weights_sha256,
        "weights_path": weights_path,
        "sanity": stats,
        "out": os.path.abspath(args.out),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
