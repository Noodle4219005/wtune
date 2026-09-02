# wtune — a physics-aware WRF configuration advisor, cost predictor, and profiler

`wtune` answers three questions about a WRF run **without modifying WRF**:

1. **What configuration should I use?** — derive a good decomposition / knob set from the
   namelist and initial condition, before running the full model.
2. **How long will it take, and which ranks will be hot?** — predict total runtime and
   per-rank cost from the atmospheric state alone.
3. **Where is the time actually going?** — measure per-rank compute/wait at scale and
   decompose the load imbalance into what is recoverable and what is not.

It targets WRF's load imbalance (the day/night solar terminator + convective storms) and the
ordinary cache/vectorization/NUMA cost of a decomposition. It is **case-general**: the hotspot
set is derived from the case's active physics, not hardcoded.

---

## Hard constraint: WRF source is never modified

**Every** wtune capability operates through namelist, runtime/env, launcher, `LD_PRELOAD`
shims, and post-hoc analysis. No `.F`/`.c` file of WRF is ever edited. This is a project rule,
not a preference, and it defines the design:

- Profiling uses PMPI `LD_PRELOAD` shims, never source instrumentation.
- Load balancing is done by **rank→core placement** (no cell ownership ever moves), never by
  changing WRF's decomposition or halo code.
- Optimizations that would need source changes are **measured and priced, then parked** —
  see *Measured-closed* below, so they are not re-proposed.

---

## Architecture

```
                 namelist.input + wrfinput/wrfout
                              │
      ┌───────────────────────┼───────────────────────┐
      ▼                       ▼                       ▼
 ADVISORY layer          PREDICTION layer        MEASUREMENT layer
 (what config?)          (how long? who's hot?)  (where did time go?)
 wtune.py                wtune_oracle*.py        blockbar_*.c (PMPI shims)
 wtune_physics.py        wtune_runtime_predict   wtune_phase_analysis.py
 wtune_knobs.py          wtune_crosscase_*.py    wtune_replay_sim.py
 wtune_solar/surrogate            │                      │
      └──────────► wtune_placement.py ◄──────────────────┘
                   (hotspot → rank/core placement, zero data movement)
                              │
                   g0a/wtune-transition-controller.py
                   (segment transitions, node scaling, imbalance gate)
```

---

## 1. Advisory layer — what configuration to use

- **Decomposition cost model** — a cache · vectorization · halo · NUMA score over the valid
  `nproc_x × nproc_y` shapes. On CONUS 2.5 km it re-derives the measured winner (`8×56`)
  without running the sweep.
- **Physics hotspot model** — per-rank load = cost-share-weighted sum over the *active* phases
  (microphysics→moisture, radiation→solar terminator, cumulus, GWD→terrain…).
  - *radiation*: analytic time-integrated solar geometry (`wtune_solar.py`).
  - *convection*: a fast SpeedyWeather.jl surrogate (`wtune_surrogate.py` + `speedy_hotspot.jl`).
- **Knob advisor** — 20 input-derived rules over a Registry-generated catalog of ~2062
  forecast parameters (`num_soil_layers`↔LSM, `radt`≈dx, PBL↔sfclay pairing, `cu_physics` vs dx…).
- **Placement** — over-decompose into equal fixed patches, assign to cores by predicted load
  (node-aware LPT). **No rank data ever moves**; applied at launch via an `LD_PRELOAD` shim.
  Because the optimum inverts as the terminator sweeps (84–99% churn per window),
  `shims/blockbar7.c` re-pins per window with `sched_setaffinity` only.

```bash
python3 wtune.py plan    --namelist namelist.input [--ranks N] [--wrfinput wrfinput_d01]
python3 wtune.py physics --namelist namelist.input [--menu] [--probe-json speedy_probe.json]
python3 wtune.py knobs   --namelist namelist.input [--all]
python3 wtune.py hotspot --namelist namelist.input --wrfinput ... [--run-surrogate]
python3 wtune.py audit   --namelist namelist.input     # classify all ~2062 catalog params
python3 wtune.py screen  --namelist namelist.input     # 3-stage funnel to the effective few
python3 wtune.py catalog [--group physics] [--grep mp_]
python3 wtune.py selftest
```

Lat/lon/date are read from a `wrfinput`/`wrfout` header via `ncdump -h` (header-only) when the
namelist lacks them.

## 2. Prediction layer — runtime and per-rank cost

Features come from a `wrfout` snapshot, aggregated per rank patch (11 features: `radiation,
micro, surf, TCWV, LWP, RWP, IWP, SWP, GWP, nspecies, mixed_phase`).

| tool | predicts | validation |
|---|---|---|
| `wtune_runtime_predict.py` | **total runtime**, case level | leave-one-out **MAE 0.69 s (~3.5%)**, 3× better than a mean baseline; best single feature = radiation load |
| `wtune_crosscase_blk.py` | **per-rank cost**, 3 km/448-rank corpus | within-case Spearman **+0.44…+0.85 (all positive)**; cross-regime LOCO **+0.50**; pooled 5-fold +0.64 |
| `wtune_crosscase_train.py` | per-rank cost, 12 km/36-rank corpus | LOCO only **+0.14** — see *scale* below |
| `wtune_oracle_train.py` | per-rank cost, single CONUS run | within-run OOF Spearman 0.884 |

```bash
PY=../wtune-pixi-env-20260830/.pixi/envs/default/bin/python3
$PY wtune_runtime_predict.py      # case-level runtime LOO
$PY wtune_crosscase_blk.py        # per-rank within-case + LOCO over the big-case corpus
```

**Per-rank prediction is a scale problem, not a model problem.** At 12 km/36 rank the per-rank
compute is ~12 s with ~1.2 s spread — the same order as profiling noise, so nothing is learnable
(LOCO +0.14, several cases negative). At 3 km/448 rank/2 h the compute is ~150 s with 6.8–17.1 s
spread, and the identical model reaches within-case ρ +0.44…+0.85. Ask whether the signal is
above the noise floor *before* asking whether the model is good enough.

**best-param factors out of the weather**: config = f(grid, nodes); the regime only sets the
absolute runtime level. So the advisory layer needs no regime features.

## 3. Measurement layer — where the time goes

PMPI `LD_PRELOAD` shims (in `../lend-microbench/`), because **APS and `I_MPI_STATS` both hang at
448 ranks** (25-min timeouts, zero integration steps) while these do not.

| shim | measures | output |
|---|---|---|
| `blockbar_prof.c` | per-rank MPI wait fraction | `grank lrank wait_frac`; label = `1 − wait_frac` |
| `blockbar_phase.c` | per-rank compute/wait per MPI segment, barrier-synced timestamps | `rank,seg,comp_s,wait_s,kind,tend_s` |

```bash
mpicc -O2 -fPIC -shared -o libblockbar_prof.so blockbar_prof.c
mpirun -np 448 env LD_PRELOAD=./libblockbar_prof.so BLOCKBAR_PROF_OUT=load.txt ./wrf.exe
$PY wtune_phase_analysis.py phase_seg.csv     # imbalance decomposition
$PY wtune_replay_sim.py                       # reprice the settlement tax from preserved traces
```

### Three profiling traps (all hit, all real)

1. **Launch with `mpirun`, never `srun`.** On f1, `srun -n N ./wrf.exe` does not bootstrap
   Intel MPI's PMI: every rank becomes a `comm_size=1` singleton and wrf.exe dies in
   `module_dm.F` with a `nest_pes` FATAL. Diagnostic: `ls rsl.out.* | wc -l` must equal the
   rank count.
2. **MPI-call index cannot be aligned across ranks.** A halo is `Irecv/Isend/Wait × N` where
   `N` = neighbour count (interior 4, edge 3, corner 2), so calls-per-step differ per rank and
   the index drifts (measured cross-rank series correlation only +0.22). Indexing phases by
   call number produces garbage. Also, Fortran collectives (`mpi_allreduce_`) route through
   internal `PMPI_` and **cannot be intercepted** — of 1.79 M recorded segments, every one was
   `MPI_Wait`.
3. **Bound the window by time, not by call count.** A call-count cap makes ranks with more
   neighbours cover less wall-clock, producing impossible results (`max_r C > wall`, C+W spread
   112%). With barrier-synced timestamps and a common time window the spread drops to 3.8%.

## 4. Transition controller (`../g0a/wtune-transition-controller.py`)

Segment-to-segment decisions for long runs: zero-measurement initial config, node-scaling with
a projection model, cost-regime detection, and the imbalance gate. `selftest` covers 40 cases.

```bash
python3 ../g0a/wtune-transition-controller.py recommend  --state -   # {domain,nodes}
python3 ../g0a/wtune-transition-controller.py plan       --state -
python3 ../g0a/wtune-transition-controller.py imbalance  --state -   # per-rank rebalance gate
python3 ../g0a/wtune-transition-controller.py selftest
```

**The imbalance gate is on absolute recoverable seconds, not the ratio.** Measured recoverable
wall `(max − mean)` per-rank compute is **6.8–17.1 s**, while an ADIOS2-restart rebalance costs
**~56 s** — so a restart-based rebalance never pays for a single case; only a cheap in-place
migration clears the gate. A ratio threshold misfires across scales (1.05× is worthless on 12 s
and worth having on 150 s).

---

## Measured results

| item | result | conditions |
|---|---|---|
| `8×56` decomposition | zero-code shape win | 448 rank / 4 node |
| `numtiles = patch/251` | **−1.72%** s/step | same-node A/B, n=1/arm |
| 4N→8N node scaling | −44.6% solver, projection error 1% | validates the controller model; **blocked by the 4-node rule** |
| total-runtime prediction | **MAE 0.69 s (~3.5%)** | case-level LOO, single feature |
| per-rank prediction | within-case ρ +0.44…+0.85, LOCO +0.50 | 12-case 3 km corpus |
| load imbalance (3 km/448) | pooling 11.5% / residual 11.8% / total 23.4% | time-aligned dense measurement |

## Measured-closed — do not re-propose

Each was enumerated and falsified by measurement or by reading WRF's source; they are closed
under the no-source-modification rule, not merely untried.

| direction | why it is closed |
|---|---|
| breaking the halo barrier externally | structurally impossible: 5 mechanisms enumerated, each with a precise blocker — own future work needs the very halo being waited on; work-stealing needs source; async progress only hides transfer, and transfer is free (send-wait 5.5 µs vs recv-wait 3597 µs); speculation needs source and yields wrong values; core-lending fails because static-OMP cannot add threads mid-region (DLB Acquire = 0) |
| the 29% settlement tax | its only levers require editing RSL_LITE/solve_em ⇒ forbidden. Priced anyway (pooling 11.5% is genuinely poolable — the busiest rank itself waits 11.3%, so the straggler rotates) and parked as a blueprint |
| per-rank rebalancing via restart | recoverable 6.8–17.1 s < 56 s restart cost |
| frequency / power shifting | cluster pinned at 3.0 GHz; 112→28 cores gave +0.2% |
| rank placement for imbalance | ownership invariance |
| AMPI / over-decomposition | multi-week port, stopped at Phase 2 |
| reduced precision | WRF is already single precision |
| more nodes | 4-node cap is a rule |

## Requirements

- Python 3.6+ (stdlib only) for the advisory layer; the prediction layer needs numpy + netCDF4
  (use the pixi env at `../wtune-pixi-env-20260830/`).
- `ncdump` to read lat/lon from a `wrfinput` when the namelist lacks them.
- Convective surrogate: Julia + `SpeedyWeather.jl`
  (`julia --project=jlenv -e 'using Pkg; Pkg.instantiate()'`).
- Shims and WRF runs: Intel MPI (`mpiicx`/`mpicc`) and a WRF `dmpar` build.

## Modules

| file | role |
|---|---|
| `wtune.py` | CLI + decomposition cost model + namelist/wrfinput parsing |
| `wtune_physics.py` | physics-suite resolver + scheme knowledge base + numerics advice |
| `wtune_knobs.py` | 20 input-derived knob rules + mismatch flags |
| `wtune_solar.py` | analytic time-integrated terminator load map |
| `wtune_surrogate.py` + `speedy_hotspot.jl` | SpeedyWeather.jl convective hotspot |
| `wtune_placement.py` | hotspot-driven rank→core placement + dynamic schedule |
| `wtune_probe.py` | empirical probe harness (submit/parse/ABBA) |
| `wtune_oracle.py` / `wtune_oracle_train.py` | per-rank feature extraction + GBRT/ridge cost model |
| `wtune_runtime_predict.py` | case-level total-runtime prediction (LOO) |
| `wtune_crosscase_blk.py` / `wtune_crosscase_train.py` | cross-regime per-rank generalization (LOCO) |
| `wtune_phase_analysis.py` | dense per-rank imbalance decomposition (pooling vs residual) |
| `wtune_replay_sim.py` | replay/reprice the settlement tax from preserved traces |
| `gen_registry_catalog.py` | regenerate `wtune_catalog_registry.json` from a WRF Registry |
| `shims/blockbar6.c` / `blockbar7.c` | static placement / dynamic per-window re-pin |
| `../lend-microbench/blockbar_prof.c` / `blockbar_phase.c` | per-rank wait / per-segment profilers |

## Status

Research tool, developed against CONUS 2.5 km and a 12-case 3 km multi-regime corpus on
f1/NCHC. The advisory, prediction, and measurement layers are validated. Under the
no-source-modification rule the remaining wall-time optimization space is essentially closed
(see *Measured-closed*), so the live value is **prediction and scheduling**: knowing a case's
runtime to ~3.5% and its hot ranks before it runs.
