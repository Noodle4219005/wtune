# wtune — profile-guided, compliance-aware WRF performance and physics configuration autotuner (decision 0050)

WTune = profile-guided, compliance-aware WRF performance and physics configuration autotuner
(decision 0050). The full autotuner pipeline — profile → diagnosis → ceiling → candidates →
experiments → correctness gate → compliance classification → selection — is being built in
steps. Landed so far: **step 1** (the compliance model: `wtune_compliance.py`, `wtune.py
compliance`, `--compliance` on `knobs`/`physics`/`screen`), **step 2** (the phase
fingerprinter), **steps 4-5** (the configuration predictor, the STRICT and PHYSICS-GRAY
campaigns, the fidelity gate) and **research step D1** (the decomposition space and its
active search). What is not yet built is the automatic end-to-end loop: the pieces below are
run individually, and the selection at the end is always the user's.

`wtune` does four things to a WRF run, **none of which modifies a WRF source file**:

1. **Research oracle (RESEARCH)** — `plov/`: link-time `--wrap` interposition of the
   microphysics and radiation drivers, row staging into `/dev/shm`, node-local work stealing
   with a feedback controller. **-13.0% s/step over 400 steps, bit-identical** (job 1070876).
   This is an upper bound on execution-flow interposition, not a STRICT-tuning result — see
   Compliance model below.
2. **Pick the launch configuration** — `wtune.py plan/physics/knobs/screen`: decomposition
   shape, tiles, physics-consistent knobs, `srun --distribution=cyclic`, `KMP_BLOCKTIME=0`.
   Recommendations carry a STRICT or PHYSICS-GRAY compliance label; filter with `--compliance`.
3. **Gate result-affecting knobs** — `wtune_config_gate.py`: `radt`, `non_hydrostatic`,
   `cudt`, … legitimately change the answer, so they are compared field-by-field against a
   reference `wrfout` with a tolerance *calibrated* from a known-acceptable pair. This is the
   fidelity gate that PHYSICS-GRAY knobs must pass; bit-identity (`diffwrf`) is the gate for
   everything else.
4. **Predict runtime** — `wtune_runtime_predict.py` / `wtune_cost_model.py`: total runtime to
   ~5% (nested LOO MAE 1.07 s) from a `wrfout` snapshot; cost(case, config) as a recomputable
   lookup plus a GBRT residual.

Everything that was measured and found null lives in `deprecated/` (kept runnable, not
extended) and every one-off study in `research/`. See the layout at the end.

---

## Compliance model (decision 0050)

Invariant, unchanged from before: **no WRF `.F`/`.c`/`.h` is ever edited.**

Every knob/lever WTune touches is classified into exactly one of three classes
(`wtune_compliance.py`); **the three classes are never ranked together**:

- **STRICT** — WRF source unchanged, normal build, simulation semantics unchanged
  (decomposition, tiling, placement, binding, OpenMP/MPI runtime, compiler flags, I/O layout).
- **PHYSICS-GRAY** — WRF source unchanged, official namelist/physics only, but the
  numerical/physical solution may change (scheme swaps, `radt`, `cudt`, hydrostatic, ...).
  Recommend and test behind a fidelity gate; **passing a WTune fidelity gate is NOT an
  official ruling** — PHYSICS-GRAY results are reported as **UNCONFIRMED**.
- **RESEARCH** — changes the execution flow of the binary/application (PLOV, `--wrap`,
  `LD_PRELOAD`, owner/helper work migration). Upper bound / oracle only. `-Wl,--wrap`,
  `LD_PRELOAD`, and PMPI shims are **RESEARCH — not allowed in STRICT tuning.**

Every catalog entry (`wtune_compliance.CATALOG`) carries: `name`, `kind`, `classification`,
`source_change`, `numerics_change`, `execution_interposition`, `requires_correctness_gate`,
`applicable_phases`, `hardware_requirements`, `status`, `evidence`. `numerics_change` maps to
the correctness gate that class must clear:

| `numerics_change` | meaning | `requires_correctness_gate` |
|---|---|---|
| `none` | output expected bit-identical to the reference | `diffwrf` |
| `roundoff` | same algorithm, floating-point roundoff may differ | `roundoff_gate` (tight field tolerance) |
| `solution` | the computed solution legitimately changes | `fidelity_gate` |

`plov/gate0.sh` proves from `objdump` that the driver call sites go through the `__wrap_*`
trampolines; `diffwrf` against an unmodified run proves the output is unchanged — together
these are the RESEARCH-path proofs for PLOV.

```bash
python3 wtune.py compliance list --compliance strict
python3 wtune.py knobs --namelist namelist.input --compliance strict
```

---

## 1. PLOV — the live rebalancer (`plov/`)

The load imbalance in WRF is in the physics (convective microphysics, the solar terminator in
radiation), and it moves every step. Nothing that acts at restart boundaries or through rank
placement can follow it (`deprecated/README.md`). PLOV acts inside the step:

- the `--wrap` trampoline captures the driver's 402/422 arguments, re-enters the real driver
  with the owner's `JME` shrunk by R rows (pass 1), then runs the R staged rows from a
  `/dev/shm` slot (pass 2) — either itself or an idle rank on the same node that claimed
  them (CAS PUBLISHED->CLAIMED, single writer, so the result is bit-identical);
- R adapts per rank from measured driver time, `R += 0.5*NJ*(t_me - t_mean)/t_me`, chunked
  in 6, capped at 75% of the patch ("adapt75"); no prediction is needed;
- `srun --distribution=cyclic` spreads the hot region over the four nodes first (-2.2% on
  its own), because stealing is node-local.

| arm (f1, 4 nodes x 56 x 2 threads, 1500x1500, 400 steps) | s/step | delta | bit-identical |
|---|---:|---:|---|
| A: block placement, unmodified | 0.3349 | — | reference |
| **Ec: cyclic + micro rows=4 adapt75** | **0.2913** | **-13.0%** | yes |
| ER: Ec + radiation rows=4 + row stealing | 0.2925 | rad steps -0.15..-0.23 s each (6/6 negative, sum -1.06 s); -0.6% net | yes |

Plus, independent of PLOV: `KMP_BLOCKTIME=0` on 224x2 is **-2.8%** (ABAB), and the hybrid
224x2 with the right `numtiles` beats 448x1 by 5.7%.

Per-step budget after PLOV (0.292 s): dynamics ~143 ms, MPI floor 37 ms (13 halo closures x
2.8 ms, source-locked), radiation ~46 ms averaged, microphysics ~34 ms (at its arithmetic
ceiling), residual imbalance ~18 ms (between nodes; cross-node stealing is bounded at
<=1.3%), PBL+surface ~11 ms. Whether ER (radiation) becomes production is still an open
decision; it is measured and safe.

Build, run, verify: `plov/README.md`; the exact environment: `plov/production.env`.

## 2. Advisory layer — `wtune.py`

```bash
python3 wtune.py plan    --namelist namelist.input [--ranks N] [--wrfinput wrfinput_d01]
python3 wtune.py physics --namelist namelist.input [--menu] [--probe-json speedy_probe.json]
python3 wtune.py knobs   --namelist namelist.input [--all]
python3 wtune.py screen  --namelist namelist.input     # 3-stage funnel to the effective few
python3 wtune.py audit   --namelist namelist.input     # classify all ~2062 catalog params
python3 wtune.py catalog [--group physics] [--grep mp_]
python3 wtune.py hotspot --namelist namelist.input --wrfinput ... [--run-surrogate]
python3 wtune.py feedback ...                          # PLOV controller replay / decisions CSV
python3 wtune.py selftest
```

- **Decomposition** — cache · vectorization · halo · NUMA score over the legal
  `nproc_x x nproc_y` shapes; re-derives the measured CONUS 2.5 km winner (`8x56`) without
  running the sweep (`wtune_decomp.py`, `wtune_shape.py` for the patch geometry).
- **Physics suite + knobs** — 20 input-derived rules over a Registry-generated catalog of
  ~2062 parameters (`num_soil_layers`<->LSM, `radt`≈dx, PBL<->sfclay pairing, `cu_physics` vs
  dx …). Knobs are labelled by what they can change: `code_enforced` (WRF overrides them),
  performance-only (`numtiles`, decomposition, I/O), and **RESULTS-affecting numerics**
  (`radt`, `non_hydrostatic`, `cudt`, …) which `wtune.py` prints under "gated tuning only".
- **Hotspot preview** — analytic solar terminator (`wtune_solar.py`) plus the SpeedyWeather
  convective surrogate (`wtune_surrogate.py` + `speedy_hotspot.jl`; always run the probe).
  Used to *explain* a case; PLOV does not consume it.
- **Transition controller** — `wtune_transition_controller.py` (copied from `../g0a/`):
  zero-measurement initial config and node-scaling projection (4N->8N measured -44.6%,
  projection error 1%; blocked by the 4-node rule). Its restart-boundary imbalance gate is
  deprecated; the 2/40 selftest failures are both in that branch.

Lat/lon/date come from a `wrfinput`/`wrfout` header via `ncdump -h` when the namelist lacks
them.

## 3. Calibration gate for result-affecting knobs — `wtune_config_gate.py`

Config-layer changes such as `radt` 10->15 or `non_hydrostatic` are worth up to -15% on a
15 km case but change the fields, so bit-identity cannot judge them. The gate compares a
candidate `wrfout` with a reference field by field (`max|d|`, RMS of the difference, RMS
relative to the field) and passes when every field is within a tolerance that was
**calibrated from a pair the user already accepts** (two runs of the accepted config, or
reference vs an accepted change):

```bash
PY="pixi run --manifest-path ../wtune-pixi-env-20260830/pixi.toml python"   # numpy + netCDF4
$PY wtune_config_gate.py calibrate --ref accepted_A/wrfout_d01_... --cand accepted_B/wrfout_d01_...
#   -> prints the per-field table and a suggested --tol-rel (max rel_rms x safety 3)
$PY wtune_config_gate.py check --ref ref/wrfout_d01_... --cand cand/wrfout_d01_... --tol-rel <that> --json gate.json
#   -> GATE PASS / GATE FAIL <fields>, exit 0/1
```

One variable is read at a time (files are ~800 MB), the last time record by default. Run it
on a compute node like everything else. A bit-identical pair reports every field at 0 and
passes at any tolerance.

## 4. Prediction layer — runtime and cost

Features come from a `wrfout` snapshot aggregated per rank patch (`wtune_oracle.py`;
`featurize_frames.py` reduces a 1-2 GB frame to 448 x 17 and deletes it; `wtune_segment_train.py`
aligns segmented features to segmented labels).

| tool | predicts | validation |
|---|---|---|
| `wtune_runtime_predict.py` | **total runtime**, case level | nested LOO **MAE 1.07 s (~5.2%)**, 1.9x better than the mean; best single feature = radiation load |
| `wtune_runtime12.py` | same on the 12 km corpus | — |
| `wtune_cost_model.py` | cost(case, config) = f(mp) + g(bl) + B(weather) | separable; the per-PBL head is **REFUTED on a diverse corpus (2026-09-05)** and stays opt-in via `WTUNE_PER_BL_HEAD` |

Per-rank prediction is a scale problem, not a model problem: at 12 km/36 ranks the per-rank
spread is at the profiling noise floor; at 3 km/448 ranks it is learnable (within-case
Spearman +0.44..+0.85). But no live actuator consumes "which rank is hot" — PLOV measures
it — so the per-rank hotspot model itself is in `deprecated/` (decision 0045).

## 5. Measurement — where the time goes

- **Per-rank driver timers** come free with PLOV: `wtimer.<rank>.csv` under `WTIMER_DIR`
  (per-call microphysics/radiation/PBL/surface/cumulus time, staged/refused/shared counters,
  wait time). A run whose counters read `staged 0` did nothing, however fast it looks.
- **PMPI shims** (RESEARCH; `profilers/blockbar_prof.c`, `blockbar_phase.c`, `LD_PRELOAD`):
  per-rank wait fraction and per-segment compute/wait, because APS and `I_MPI_STATS` both
  hang at 448 ranks. `wtune_probe.py` is the ABBA submit/parse harness.

Three profiling traps, all hit: launch with `srun --mpi=pmi2` (plain `srun` leaves Intel
MPI as `comm_size=1` singletons; diagnostic `ls rsl.out.* | wc -l`); never align phases by
MPI-call index across ranks (neighbour count differs per rank; Fortran collectives cannot be
intercepted); bound windows by wall time, not call count.

### Phase fingerprinter (autotuner step 2, validated; for calibration / second-pass profile runs only)

`fingerprint/` observes an unmodified `wrf.exe` from outside: `sampler_node.sh`
attaches `perf record` (user-mode cycles sampling) to its pids, and
`fingerprint.py` time-weights the resulting `perf script` samples into per-phase
shares (MICRO/RADIATION/PBL/SURFACE/CUMULUS/DYNAMICS_OTHER/MPI/IO), a
`fingerprint.json`, and diagnosis labels. No `--wrap`, no `LD_PRELOAD`.

    python3 fingerprint/fingerprint.py --sampler-dir OUTDIR --run-dir RUNDIR \
      [--omp-threads N] [--wtimer-dir DIR] --out OUTPREFIX

`--wtimer-dir` compares pooled shares against in-process `wtimer.<rank>.csv`
timers (ground truth only). Validation job 1074484 (karina 224x2, 4 nodes, 100 Hz,
`POLL_S=5`): out-of-sample share error micro 0.71 / radiation 0.30 / dynamics 0.96 /
MPI 1.87 pp, diffwrf identical; overhead 1.40% (gate was <=1%). 2026-09-14 user ruling:
overhead accepted because the fingerprinter is used only for second-pass profile runs,
never on production timing runs.

### Configuration predictor and STRICT data campaign

`campaign/strict_campaign.sbatch` measures the 16 STRICT templates of `campaign/configs.py`
per case (all-ADIOS2 I/O, palindromic replicates) into `ledger_<case>_<job>.csv`;
`predictor/run_benchmark.sbatch` builds the ledger (`predictor/ledger.py`) and runs the
grouped-CV benchmark (`predictor/benchmark.py`). First cf448 campaign: jobs 1075293-1075304;
benchmark 1075654 -- leave-one-case-out median regret catboost_pairwise 0.4% vs naive (always
C01) 15.2% over 6 cases, but conus12 and leave-one-hardware-out do not generalise. The
predictor outputs a ranked shortlist; the final configuration is the user's decision.

`predictor/recommend.py` is the user-facing shortlist command (STRICT and PHYSICS-GRAY); see
`predictor/README.md` for flags, evidence warnings, and status rules. It has two objective
modes, `--objective perf` (default) and `--objective accuracy` (`OBJECTIVES` at
`predictor/recommend.py:1502`): perf-first ranks on cost subject to a fidelity constraint,
acc-first ranks on error and reports what the accuracy costs. Both emit a Pareto front over
(cost, error) rather than a single answer.

**The one invariant to preserve when touching physics templates**: the repo has *two* template
engines and they must agree. The dict engine
(`predictor/physics.py:physics_candidates_for_case`) reads the `physics` dict from
`campaign/cases.json` and never opens a namelist; the text engine
(`campaign/configs.py:resolve_physics_config`) edits the real `namelist.input` that WRF runs.
`predictor/tests/test_physics.py:596` asserts they agree on applicability and on every changed
key's target value across six real cases and every template. The known asymmetry is documented
there at `:640-643` — the dict engine cannot know the native value of a namelist-only knob
(`swint_opt`, `aer_opt`, `use_aero_icbc`), so the *from* value is compared only for
`physics.CHANGE_FIELDS`.

### I/O: a guarantee, not a tuning axis

All WRF I/O goes through ADIOS2 (`io_form 14`). The codec is not searched: `campaign/configs.py`
writes `adios2_compression_enable=.true.` and `adios2_blosc_compressor="zstd"` into every run's
`&namelist_adios2`, because zstd wins on both axes measured (write 0.468 s vs lz4 0.496 s vs
1.014 s uncompressed; 166.7 MB vs lz4 181.1 MB vs 595.9 MB for the same frame).
`adios2_numaggregators` **must** equal the MPI rank count — values at or below the node count are
silently ignored.

`tools/bp_compression_check.py` asserts that compression actually applied, from the ratio of
logical to stored bytes, because this ADIOS2 build carries no codec marker in BP metadata at all.
It is needed: in this WRF build only the *first* ADIOS2 write stream of a run is compressed
(`external/io_adios2/wrf_io.F90:1283-1285` assigns the codec name only on the branch that defines
the operator), so `--history-mode interval` silently writes frames 2+ uncompressed. That is a WRF
source bug, not a namelist one, and it is not patched here.

### PHYSICS-GRAY data campaign and fidelity gate (decision 0051)

- `campaign/physics_templates.json`: 19 fixed templates in three families, plus an `excluded`
  list with reasons, derived from a WRF 4.8 source compatibility survey. Each template carries a
  `direction` field (`cost` or `accuracy`) that ties it to the objective modes below.
  - **P00-P06 — physics** (`direction: cost`). P00 reference; P01 `swint_opt`; P02 `radt`×2+`swint_opt`;
    P03 `time_step`×1.2; P04 WSM6; P05 PBL+sfclay swap; P06 RRTM/Dudhia.
  - **D01-D07 — `&dynamics` cost levers** (`direction: cost`): advection options, `rk_ord`=2,
    output diagnostics off, `bldt`, one-way nesting, `time_step_sound`.
  - **A01-A05 — accuracy mirrors** (`direction: accuracy`): the same knobs pushed the other way, so
    `recommend.py --objective accuracy` has candidates to rank.
- `strict_campaign.sbatch MODE=physics` (env `PCONFIGS`, `DECOMP`, `TWIN_DECOMP`, `SIM_MINUTES`, `HIST_MINUTES`, `CONFIG_FILE`) runs three kinds of run:
  - reference P00 at DECOMP;
  - a same-physics twin at TWIN_DECOMP;
  - each candidate at DECOMP.
  `CONFIG_FILE` is the only way to run a case that has no template of its own — it supplies the
  decomposition pair directly (`campaign/configs.py decomp-pair`); see
  `campaign/strict_campaign.sbatch:38-46`.
- Replicate 1 writes interval history (fidelity), replicate 2 is timing only. Every candidate is scored by `fidelity_gate.py`.
- `fidelity_gate.py --reference --twin --candidate --family --out-prefix` computes, per field:
  - the chaos-floor ratio RMS(ref,cand)/RMS(ref,twin)
  - the normalized distance
  - bias and precipitation accumulation difference

  It gives a provisional, overridable label WITHIN-FLOOR / SMALL-DRIFT / LARGE-DRIFT / INVALID. The label describes model self-drift, not observed skill; acceptability is the user's call.

  **Read `floor_zero=True` as "the floor was not measured", not "the candidate is fine."** Only
  `nproc_x` changes WRF's arithmetic (the default tiling splits in J alone), so a twin that keeps
  the reference's `nproc_x` has a chaos floor of exactly zero; WITHIN-FLOOR then degenerates into
  "bit-identical to the reference" and every perturbation reads as LARGE-DRIFT.

### Decomposition space and active search (research step D1)

`predictor/decomp_space.py` enumerates the full legal stock-WRF decomposition and tiling space per case with WRF's exact patch and tile rules (`enumerate`), draws a stratified ground-truth sample that `strict_campaign.sbatch CONFIG_FILE=...` can measure (`sample`), and emits a cold-start design of two shapes per thread count — most balanced and widest-in-x — for a case with no measurements at all (`seed`).

`predictor/decomp_search.py` is the D3 active search: a bootstrap CatBoost ensemble with LCB acquisition proposes each round's batch, stopping when the remaining predicted gain falls inside the timing noise floor. `campaign/decomp_search_campaign.sbatch` runs seed → rounds → confirm → rank in one allocation; `campaign/phase_ranking.py` ranks per window from `rsl.error.0000` and settles the verdict with a co-winner rule inside the noise band, and `campaign/decomp_summary.py` groups results by the decomposition WRF actually used.

### Judging accuracy against observations, not against another model run

`predictor/verify_obs.py` is the only path in the repo that scores a configuration against something outside WRF. Two modes: `station` compares nearest-gridpoint T2 against an hourly observation CSV (RMSE, bias); `track` locates the storm centre from SLP and compares it with an HURDAT2 best track (great-circle km, `pmin` hPa, `vmax` m/s). `case` runs whichever the case declares. Everything else in this repo — including the fidelity gate — measures model self-drift, so this is what makes `--objective accuracy` mean anything.

`predictor/stability.py` replicates WRF's acoustic sub-step formula (`dyn_em/solve_em.F:451-464`) and reports the acoustic CFL margin before a run, which is what makes the D07 `time_step_sound` template safe to propose per case. Its documented limit: acoustic CFL does **not** explain the P03 `time_step`×1.2 crashes.

## Measured results

| item | result | conditions |
|---|---|---|
| **PLOV Ec** (cyclic + micro staging + adaptive stealing) [RESEARCH] | **-13.0% s/step, bit-identical** | 400 steps, 4 nodes, job 1070876 |
| PLOV ER (+ radiation staging) [RESEARCH] | -0.15..-0.23 s per rad step, -0.6% net | same job; adoption pending |
| `srun --distribution=cyclic` | -2.2% | 4 nodes, no staging |
| `KMP_BLOCKTIME=0` | -2.8% | 224x2 ABAB |
| hybrid 224x2 + correct `numtiles` vs 448x1 | -5.7% (-6.4% with blocktime) | same node set |
| `numtiles = patch/251` | -1.72% | same-node A/B |
| `8x56` decomposition | shape win, zero code | 448 ranks / 4 nodes |
| 4N->8N node scaling | -44.6%, projection error 1% | blocked by the 4-node rule |
| total-runtime prediction | MAE 1.07 s (~5.2%) | case-level nested LOO |
| config layer (`radt`, `non_hydrostatic`, `cudt`) [PHYSICS-GRAY] | up to -15% | 15 km case; result-affecting, behind the calibration gate; UNCONFIRMED until an official ruling |

## Measured-closed — do not re-propose

| direction | why it is closed |
|---|---|
| restart-boundary rebalancing / migration | restart ~56 s > 7-17 s recoverable per window; PLOV does it in place |
| rank placement (LPT, x-pair, per-window re-pin) | equalises node micro to 1%, s/step unchanged |
| DLB/LeWI core lending | nothing beyond `KMP_BLOCKTIME=0`; static OpenMP cannot add threads mid-region |
| per-rank hotspot prediction as an actuator input | no consumer; PLOV measures |
| rows > 4 in PLOV, cross-node stealing | each extra driver entry costs 13-16 ms; cross-node rad bounded at 1.3% |
| breaking the halo barrier externally | 13 closures x 2.8 ms = 37 ms/step floor, every lever needs RSL_LITE/solve_em source |
| the 29% settlement tax | levers are in solve_em; priced and parked |
| frequency / power shifting | cluster pinned at 3.0 GHz; 112->28 cores gave +0.2% |
| AMPI / over-decomposition | multi-week port, stopped at Phase 2 |
| reduced precision | WRF is already single precision |
| more nodes | 4-node cap is a rule |

## Requirements

- Python 3.6+ (stdlib only) for `wtune.py` and the transition controller; the prediction
  layer and the config gate need numpy + netCDF4 — use the pixi env at
  `../wtune-pixi-env-20260830/` (`pixi run python`, never the env's `bin/python` directly:
  HPC scripts put the Intel stack first on `LD_LIBRARY_PATH`).
- `ncdump` for headers; Julia + SpeedyWeather.jl for the convective probe
  (`julia --project=jlenv -e 'using Pkg; Pkg.instantiate()'`).
- PLOV: Intel compilers + Intel MPI and a WRF `dm+sm` build (`-qopenmp` live) with
  `libwrflib.a`; see `plov/build.sbatch`.

## Layout

| path | status | contents |
|---|---|---|
| `plov/` | **live, production** | wrap trampolines, stagers, build/gate/harness scripts, `production.env` |
| `wtune.py`, `wtune_physics.py`, `wtune_knobs.py`, `wtune_solar.py`, `wtune_surrogate.py`, `speedy_probe.jl`, `speedy_hotspot.jl`, `wind_probe.py`, `wtune_decomp.py`, `wtune_shape.py`, `gen_registry_catalog.py`, `wtune_catalog_registry.json` | live | advisory layer |
| `wtune_config_gate.py` (+ test) | live | calibration gate for result-affecting knobs |
| `wtune_transition_controller.py` | live (gate branch deprecated) | zero-measurement config + node-scaling projection |
| `wtune_feedback.py` (+ test), `wtune_probe.py` (+ test) | live | PLOV controller replay; ABBA probe harness |
| `wtune_oracle.py`, `wtune_oracle_train.py`, `wtune_segment_train.py`, `wrf_frame.py`, `featurize_frames.py`, `wtune_runtime_predict.py`, `wtune_runtime12.py`, `wtune_cost_model.py` | live | prediction layer |
| `campaign/` | **live, production** | the measurement side: `strict_campaign.sbatch`, `configs.py`, `cases.json`, `physics_templates.json`, the decomposition-search campaign and its rankers |
| `predictor/` | **live, production** | the decision side: ledger, benchmark, `recommend.py`, decomposition space and search, `stability.py`, `verify_obs.py` (`predictor/README.md`) |
| `fidelity_gate.py` (+ test) | live | chaos-floor scoring of a candidate against a reference and a twin |
| `fingerprint/` | live | out-of-process phase fingerprinter (autotuner step 2) |
| `tools/` | live | `bp_compression_check.py`, `run_tests.sbatch`, the ADIOS2→netCDF4 converter |
| `experiments/` | live | the harnesses that produced the conclusions, vendored with sha256 provenance (`experiments/README.md`) |
| `results/derived-data/` | data | canonical processed CSV/JSON behind every reported number |
| `profilers/` | live | PMPI `LD_PRELOAD` profilers (RESEARCH) |
| `research/` | kept | one-off studies behind wiki notes (`research/README.md`) |
| `deprecated/` | closed | migration/restart, placement, hotspot model and their tests (`deprecated/README.md`) |

Tests: `sbatch tools/run_tests.sbatch` runs the whole suite on a compute node and is the only
command that covers all of it — **758 tests across two pixi environments**, because
`predictor/tests` needs `../wtune-predictor-env` while everything else uses
`../wtune-pixi-env-20260830`; running `predictor/` in the wrong one turns 18 skips into 19
lightgbm failures that look like a regression. The pieces, if you need one of them directly:
`pytest test_wtune_*.py test_fidelity_gate.py` at the root, `pytest campaign tools experiments`,
`pytest predictor` (predictor env), `pytest deprecated/` (archived), `python3 wtune.py selftest`
(also runs `wtune_compliance.validate()`), `python3 wtune_transition_controller.py selftest`
(38/40; both failures are in the deprecated imbalance-gate branch).

## Status

Developed against CONUS 2.5 km, a 12-case 3 km multi-regime corpus, and a 1500x1500 4-node
production case on f1/NCHC. The wall-time work is now in PLOV (-13%, bit-identical, no
source change); the remaining per-step budget is dynamics + a source-locked MPI floor, so
further gains come from the config layer behind the calibration gate, not from rebalancing.
