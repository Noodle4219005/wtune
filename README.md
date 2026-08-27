# wtune — a physics-aware WRF configuration autotuner

`wtune` predicts a good WRF run configuration **from the namelist and initial
condition, before running the full model** — and, where the analytic model can
only narrow, measures the shortlist empirically. It targets the WRF load
imbalance (the day/night solar terminator + convective storms) and the ordinary
cache/vectorization/NUMA cost of a decomposition, and it is **case-general**: the
hotspot set is derived from the case's active physics, not hardcoded.

## What it does

- **Decomposition cost model** — a cache · vectorization · halo · NUMA score over
  the valid `nproc_x × nproc_y` shapes. On CONUS 2.5 km it re-derives the measured
  winner (`8×56`) without running the sweep (`wtune.py selftest`).
- **Physics hotspot model** — per-rank load = a cost-share-weighted sum over the
  *active* phases (microphysics→moisture, radiation→solar terminator, cumulus,
  GWD→terrain, …). A different testcase yields a different hotspot map.
  - *radiation*: analytic, time-integrated solar geometry (`wtune_solar.py`).
  - *convection*: a fast global SpeedyWeather.jl surrogate (`wtune_surrogate.py`
    + `speedy_hotspot.jl`) — the moving storm hotspot the terminator can't give.
- **Knob advisor** — 20 input-derived rules over a Registry-generated catalog of
  ~2062 forecast parameters (`num_soil_layers`↔LSM, `radt`≈dx, PBL↔sfclay pairing,
  `cu_physics` vs dx, …), flagging mismatches (`wtune_knobs.py`, `wtune_catalog_registry.json`).
- **Hotspot-driven placement** — over-decompose into equal fixed patches and assign
  them to cores by predicted load (node-aware LPT), so no rank data ever moves
  (`wtune_placement.py`). Applied at launch via an `LD_PRELOAD` affinity shim.
- **Dynamic re-pin** — because the optimal placement inverts as the terminator
  sweeps (84–99% churn per window), a time-phased schedule re-pins per window
  (`shims/blockbar7.c`, zero data movement — just `sched_setaffinity`).
- **Empirical probe (Stage C)** — short WRF runs + ABBA to pick the winner on
  measured wall/step, not a proxy (`wtune_probe.py`).

## CLI

```
python3 wtune.py plan    --namelist namelist.input [--ranks N] [--wrfinput wrfinput_d01]
python3 wtune.py physics --namelist namelist.input [--menu] [--wrfinput ...] [--probe-json speedy_probe.json]
python3 wtune.py knobs   --namelist namelist.input [--wrfinput ...] [--all]
python3 wtune.py hotspot --namelist namelist.input --wrfinput ... [--run-surrogate | --surrogate-json J]
python3 wtune.py catalog [--group physics] [--grep mp_]
python3 wtune.py audit   --namelist namelist.input   # Registry-driven scan classifying ALL ~2062 catalog params (any case)
python3 wtune.py screen  --namelist namelist.input [--probe-json probe.json]  # 3-stage funnel: all params -> SpeedyWeather vet -> effective few
python3 wtune.py selftest

# physics-context self-validation via a fast SpeedyWeather forecast:
julia --project=jlenv speedy_probe.jl --hours 6 --lat0 22 --lat1 51 --lon0 -118 --lon1 -67 --out probe.json
python3 wind_probe.py --wrfinput wrfinput_d01 --dx 2500      # exact t=0 steering wind (needs netCDF4)
```

Lat/lon/date are read from a `wrfinput`/`wrfout` header via `ncdump -h` (cheap,
header-only) when the namelist lacks them.

## Modules

| file | role |
|---|---|
| `wtune.py` | CLI + decomposition cost model + namelist/wrfinput parsing |
| `wtune_physics.py` | physics-suite resolver + scheme knowledge base + numerics advice |
| `wtune_knobs.py` | 20 input-derived knob rules + mismatch flags |
| `wtune_solar.py` | analytic time-integrated terminator load map |
| `wtune_surrogate.py` + `speedy_hotspot.jl` | SpeedyWeather.jl convective hotspot |
| `wtune_placement.py` | hotspot-driven rank→core placement + dynamic schedule |
| `wtune_probe.py` | Stage C empirical probe harness (submit/parse/ABBA) |
| `gen_registry_catalog.py` | regenerate `wtune_catalog_registry.json` from a WRF Registry |
| `shims/blockbar6.c` | static placement (`LD_PRELOAD`, pin + release-on-wait) |
| `shims/blockbar7.c` | dynamic per-window re-pin |

## Requirements

- Python 3.6+ (stdlib only).
- `ncdump` (NetCDF tools) to read lat/lon from a `wrfinput`.
- For the convective surrogate: Julia + `SpeedyWeather.jl`
  (`julia --project=jlenv -e 'using Pkg; Pkg.instantiate()'`).
- To apply placement: Intel MPI (`mpiicx`) to build the shims; a WRF `dmpar` build.

## Status

Research tool, developed against CONUS 2.5 km on the f1/NCHC cluster. The
decomposition, physics, knob, and hotspot layers are validated; the
over-subscription placement experiment is ongoing (static round: +5.5% vs
baseline; the vehicle cost sits below the prize, so the residual imbalance is the
limiter — dynamic + inter-node rebalance is the open experiment).
