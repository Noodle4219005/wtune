# wtune — profile-guided, compliance-aware WRF performance and physics configuration autotuner (decision 0050)

WTune = profile-guided, compliance-aware WRF performance and physics configuration autotuner
(decision 0050). The full autotuner pipeline — profile → diagnosis → ceiling → candidates →
experiments → correctness gate → compliance classification → selection — is being built in
steps. Landed so far: **step 1** (the compliance model: `wtune_compliance.py`, `wtune.py
compliance`, `--compliance` on `knobs`/`physics`/`screen`), **step 2** (the phase
fingerprinter), **steps 4-5** (the configuration predictor, the STRICT and PHYSICS-GRAY
campaigns, the fidelity gate) and **research step D1** (the decomposition space and its
active search). What is not yet built is the automatic end-to-end loop: the pieces below are
run individually, and the selection at the end is always the user's. The one exception is
opt-in: `wtune run --tune-report` (section 6) applies a predicted-mode tune report's
per-segment choice, and only when the user passes that report.

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

### Cold-start sampling (`tools/coldstart_sample.py`) — unknown-case prediction

For a case with only a `namelist.input` (+ a projection source) and NO wrfinput/wrfbdy/restart
files, `wtune_predict.py`'s probe-based model has nothing to measure from. `coldstart_sample.py`
closes that gap: it picks K "characteristic point" sampling times inside the run, fetches only
the GFS GRIB times those windows need, stages ONE sbatch running geogrid -> ungrib -> metgrid ->
real.exe (one per window) -> wrf.exe (one short cold-start window per sampling time, all at the
configuration being predicted), then scores the windows and prints a whole-run wall-time
prediction. Each window is an independent cold start from GFS state at time `T_k` (not a
restart — an unknown case has none). Measured on the cluster (conus12, 3 times x paired with
true restarts): cold-start cost equals restart cost within 0.35% once the first ~1 simulated
hour is discarded, so every window budgets a spin-up discard before it starts measuring.

    python3 src/tools/coldstart_sample.py inspect --case-dir D [--namelist-wps P | --geo-from-wrfinput P] [--grib-dir G]
    python3 src/tools/coldstart_sample.py plan --case-dir D --out PLAN_DIR --k 4 --nodes N --ranks R --omp T --nx X --ny Y
    python3 src/tools/coldstart_sample.py fetch --plan PLAN_DIR         # login node; needs network
    python3 src/tools/coldstart_sample.py stage --plan PLAN_DIR --site scc
    sbatch PLAN_DIR/run.sbatch
    python3 src/tools/coldstart_sample.py score --plan PLAN_DIR --json PLAN_DIR/score.json

`inspect` names every blocker explicitly (no projection source; a run time older than the GFS
0.25deg archive's 2015-01-15 start with no `--grib-dir`; `ifire>0` — WRF-SFIRE unsupported; a
nest whose start time differs from d01's; `sf_lake_physics>0` with no `lake_depth` dataset under
geog; `sf_urban_physics` 2/3 — BEP/BEM need urban parameter data geog doesn't carry; `chem_opt>0`
— WRF-Chem emissions; `obs_nudge_opt>0` — observation files) and, separately, what is
completable but "filled differently" than the case's own namelist asked for (hydrometeors/number
concentrations start at zero — always, from GFS via Vtable.GFS — which is *why* cold-start
windows discard a spin-up hour; `mp_physics=28` with no aerosol climatology wired in; a
`num_land_cat` mismatch against whichever landuse dataset the geog directory actually has;
`sf_surface_physics=3` with `num_soil_layers!=4`; `grid_fdda>0` nudging files only covering the
sampled windows). The geog capability check reads the real `geog_data_path` directory listing
(`--geog-path`, else `$WPS_GEOG`/`/nfs/data/wrf/WPS_GEOG/`) rather than a hardcoded dataset list,
so it degrades to "skipped" (never a false blocker) on a site whose mirror can't be listed, and
still works on a site whose geog mirror differs from scc's. Projection (namelist.input carries
none) comes from `--namelist-wps`/`D/namelist.wps` first, else `--geo-from-wrfinput`/
`D/wrfinput_d01`'s global attributes — generalised from the proven-on-cluster, conus12-hardcoded
`.route1-ref/make_namelist_wps.py` recipe, dropping its nested-case MOAD_CEN_LAT refusal (d01's
own MOAD_CEN_LAT always equals CEN_LAT, so that check never distinguished a real nest; nest
placement comes from namelist.input's own per-domain lists instead). Namelist edits reuse the
same packed-line-safe `set_list`/`count_key` approach `.route1-ref/gate2_stage_window.py` proved
on the cluster (key-count invariants; a `history_interval*` key that doesn't already exist is
never added; `fix-real-namelist`'s `num_land_cat` overwrite from met_em is required by real.exe
and stays, but now prints old->new instead of changing it silently). `ungrib`/`metgrid` run once
PER WINDOW (`fix-wps-namelist` narrows namelist.wps `&share` start/end to that window's own
`[t, t_end]` first — running them once against the whole run's span makes metgrid walk every
interval and die on the first GRIB time `fetch` never downloaded), and a window that fails at
any step is recorded `ok=0` in `windows_status.txt` and does not abort the remaining windows;
`score` skips `ok=0` windows and errors only if none are usable. `run.sbatch` sources
`campaign/sites/scc.sh` for `site_env`/`site_launch`/`site_hygiene` and
`campaign/node_hygiene.sh` for orphan reap/dirty-node census rather than reimplementing either —
`--exclude=head`, `--signal=B:USR1@90` (USR1/TERM/INT kill the in-flight run, reap, record
`INTERRUPTED`, and exit 3), no `LD_PRELOAD`, no wrapper around `wrf.exe`. `tools/metdata.py` and
`tools/grib_state.py` (both need numpy, which scc's compute-node `python3` doesn't have) are
imported lazily and only by `inspect`/`plan --state`/`fetch` — every compute-node subcommand
(`fix-wps-namelist`, `fix-real-namelist`, `fix-wrf-namelist`, `stage`, `score`) never touches
them. `score` classifies radiation steps from each window's own data
(`classify_steps_detected`), fits a day/night radiation ratio (`fit_day_night_r_rad`), and
linearly interpolates each window's measured step cost across the full run between window
measured-part midpoints; I/O and init come from `wtune_predict.build_prediction` against
`PLAN_DIR/win_0/real` (the generated wrfinput), never the original case_dir, which in an unknown
case has none and in a held-out validation case would be leakage.

Tests: `tools/test_coldstart_sample.py` (65 cases, offline/no network — namelist.wps field
derivation including per-window time narrowing, nested-domain list copying and sbatch linking,
inspect blockers and capability-gap warnings driven by a real tmp geog listing, window selection
on a synthetic 72h run, the `dt=72/radt=15 -> stepra=13` window-size arithmetic, wrf-window
namelist key-count invariants, score's spin-up exclusion + exact trapezoid interpolation +
`ok=0` window skipping + case_dir non-leakage, the USR1-trap/TOOL-path/GRIB-fallback shape of
the generated sbatch text, and that every compute-node code path stays importable with numpy
blocked).

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

## 6. Whole-run prediction, monitoring and segmented execution (0.2.0)

Four subcommands of `bin/wtune` (`wtune.py <cmd>`), plus preflight rules. None edits a WRF
source file or injects anything; none writes inside `CASE_DIR`. Every sbatch they generate
carries `--exclude=head[,$WTUNE_SBATCH_EXCLUDE]`. All evidence below is scc (72 ranks/node),
2026-09-27; errors are against a truth run of the same case and configuration.

### `wtune predict-rebuild` — whole-run wall time from rebuilt segments (decision 0054)

```bash
wtune predict-rebuild CASE_DIR --nodes N --nx X --ny Y [--numtiles T] [--omp O]
    [--forcing era5|auto] [--spinup-min 20] [--max-points K] [--measure-steps N|short]
    [--rain-snow] [--no-io-probe] [--out DIR] [--dry-run]
```

Places up to K points on the run's timeline (default 5; on the forcing's hourly grid,
`T + spin-up <= P`), rebuilds each from ERA5 with cloud water/ice (ARCO-ERA5 written straight to
WPS intermediate, no ungrib; `--forcing auto` = GFS chain with a 60-min spin-up), runs WPS +
real.exe + a 20-min spin-up + `max(30, 2*stepra)` measured d01 steps **on the target
decomposition**, one Slurm job per point, all submitted at once, then combines the points into
compute+init (`wtune_probe_plan.combine`: per-point normal-step cost, day/night radiation ratio).
`OUT/pred_rebuild.json` is written before anything is compared with it. Needs the site env
(`WPS_DIR`, `WRF_TREE`) and `CASE_DIR/wrfinput_d01` (geometry and ERA5 crop box).

**I/O is measured, not modelled** (`--io-probe`, default on): the last point's segment also
writes one history frame per domain (+ one restart if the case writes restarts) with the case's
own `io_form`/`iofields`; I/O = Σ frames × timed write per domain. The printed **recommended
`--time`** = range_high + 2 × I/O + margin (10% of range_high, clamped to 2–5 min), rounded up to
1 min below 2 h, else 5 min. Factor 2: history writes are 21–48% of wall on scc and the same
volume varies up to 2.4× there.

| case | run, config | predicted compute+init | truth | error | same-day drift-corrected | driver wall |
|---|---|---:|---:|---:|---:|---|
| conus25 | 6 h, 4 nodes 12×24 nt 8 | 1800.9 s | 1783.4 s | +0.98% | not measured | 19:40 (jobs 1309–1313; 51% of the 38:40 run — just run it) |
| maria1km | 3 h, 4 nodes 16×18 | 4268.3 s | 4270 s (job 977) | −0.04% | −2.19% (job 1321) | jobs 1325–1327 in parallel |
| zhengzhou | 30 h, 3 domains, 4 nodes 16×18 | 15970.2 s | 16949.8 s (job 1020) | −5.78% (−5.67% rerun) | ≈ −2% (job 1320) | 2244 s (mostly queue); 804.7 s = 4.5% of wall with 3 points |
| lafe | 24 h, 1000×1000, 8 nodes 24×24 | 13637.8 s | 13616.1 s (job 975) | +0.16% | +0.41% (job 1346) | 32:41 (jobs 1343–1345, serialised on 12 free nodes) |

- The same run on scc varies ±2–4% between days, in both directions (zhengzhou 3.4–4.1% faster,
  maria1km 2.2% slower). Validate a prediction against a same-day control on the original input.
- One point at t0 misses the diurnal radiation: lafe −10.5%, conus25 −9.6%.
- `--spinup-min 10 --measure-steps short` (`max(30, stepra+10)` steps, radiation flagged by
  known phase `n % stepra == 1`) is opt-in. It was picked in-sample on zhengzhou (627 s = 3.5% of
  wall); out-of-sample it moves the prediction −1.11% (dorian2019, one clean pair) and +0.38%
  (lafe, driver 829 → 687 s) against the default window.

`--time` check T1 (truth wall ≤ recommended ≤ 1.5 × wall):

| case | measured I/O | truth Σ writes | I/O error | recommended `--time` | ÷ truth wall |
|---|---:|---:|---:|---:|---:|
| dorian2019 (6 h) | 299.1 s | 300.5 s | −0.5% | 38:00 | 1.33 (1718.0 s) |
| maria1km | 389.2 s | 391 s | −0.5% | 1:37:00 | 1.25 (4661.5 s) |
| zhengzhou | 905.2 s | 877.8 s | +3.1% | 5:25:00 | 1.09 (17827.6 s) |

**Not yet validated:** the first rule (fixed 5-min margin, 5-min rounding) failed T1 on dorian
(45:00 = 1.57×); the scaled margin was chosen after seeing that, so the passing column is
in-sample. zhengzhou's compute range_high (17387.4 s) is *below* its real wall — only the 2× I/O
covers it, so do not lower that factor. The out-of-sample check on elevconv2022 (15 h, 8 nodes,
writes 21% of wall: `--time`, I/O probe within 15%, watch with prediction) is registered and
**PENDING**. Typhoon vortex bogussing (`tools/tc_bogus.py`) failed its criteria and is opt-in.

### `wtune watch` — read-only run monitor

```bash
wtune watch RUNDIR [--interval 60] [--once] [--json] [--prediction pred_rebuild.json]
    [--init-limit 600] [--hang-factor 10] [--hang-min 300] [--replay-fraction F[,F...]]
```

Reads only `namelist.input` and `rsl.error.0000`/`rsl.out.0000`; never writes, never signals.
Reports progress, remaining **compute and writes separately** (plus their sum), and warnings;
exit 3 on FAILED, 2 on HANG / HANG_INIT / NEST_STOPPED / NEST_END_EARLY, else 0 (CFL is
informational).

- A d01 step that absorbed `Timing for Writing` lines has those seconds subtracted; compute rate
  = mean compute over a window aligned to the radiation/history LCM period. Remaining writes =
  per-stream median seconds per frame × frames still due per the namelist (history, restart,
  auxhist, per domain); a stream not written yet uses `pred_rebuild.json`'s measured io block.
- `--prediction`: remaining compute = k × predicted compute over the remaining steps, k =
  observed ÷ predicted so far — the prediction supplies the trend, the run its level.
- HANG_INIT caught the known silent hang (elevconv, missing iofields file); NEST_END_EARLY caught
  yagi truth 1038 (nest `end_*` not updated, d01 kept advancing).

Replay on scc truth logs at 25% progress, error of estimated vs actual remaining (wtune 3ee3808):

| case | rate only | with `--prediction` |
|---|---:|---:|
| zhengzhou / dorian2019 / lafe | −0.7% / −1.1% / −3.3% | −1.8% / −1.2% / +0.6% |
| maria1km / conus25-g0014 | −2.9% / −3.8% | −1.7% / −6.4% |
| marshall / isc21 (30 min) / c34 (truncated log) | −6.2% / +0.4% / −0.7% | — |
| elevconv (576 ranks) | **−15.6%** | — (PENDING) |

This is the third attempt at a pre-registered ≤10% criterion (first: 3 of 9 failed; second
dropped write-absorbing steps as outliers, marshall −53.5%). The with-prediction column is
in-sample: those predictions were made against the same truths. Rate-only cannot see a rising
per-step cost (elevconv, convection developing) — use `--prediction` for long runs. Zero false
positives across healthy runs was part of the criterion and has not been reported separately.
Limits: rank 0's rsl only, so an error printed on another rank is not seen; files are reopened
and re-stat'ed every poll because a long-lived NFS handle can serve stale content, and HANG is
only as fresh as the login node's view of the rsl mtime.

### `wtune run` — segmented execution with restart hand-offs

```bash
wtune run CASE_DIR --nodes N --nx X --ny Y [--numtiles T] [--omp O]
    [--wall-limit HH:MM:SS | --segment-minutes M] [--stop-at YYYY-MM-DD_HH:MM:SS]
    [--prediction pred_rebuild.json] [--tune-report tune_report.json]
    [--restart-io-form 14|2] [--cfl-rescue] --out OUT [--dry-run]
```

A login-node controller: segment k runs in `OUT/seg_k` as its own Slurm job (inputs symlinked;
`--out` may not be inside `CASE_DIR`), is inspected with `watch`, and hands off to k+1 through
its restart files. `OUT/run_report.json` is rewritten after every attempt; any failure other
than the rescue case below stops with an error, never a silent retry.

- **One restart quantum L for the whole run.** Every segment is an L block from the run start
  (the last may be shorter) and `restart_interval = L` in every segment. `--segment-minutes M`
  sets L = M; `--wall-limit W` picks the largest L for which every segment's predicted wall
  (compute + init + restart read/write + history frames inside it) ≤ 0.8 W; neither = one
  segment. Segment `--time` = 1.5 × predicted wall + 10 min (5-min steps), capped at W; after
  each segment its observed overhead replaces the prediction in later `--time`s, never the grid.
- **`override_restart_timers` is never set here**, so WRF restores every alarm's phase from the
  restart file. With it (first R1 attempt), d01's 60-min history alarm re-anchored at the 00:30
  boundary and the 01:00 frame was never written. The tuner's candidates need the opposite (next
  section): they must *not* inherit the spin-up's restart interval.
- Every active domain's `start_*`/`end_*` is set per segment (yagi 1038: a nest whose `end_*` was
  left alone stopped early). Restarts default to ADIOS2 BP4 via `adios2.xml` (io_form 14).
- `--tune-report`: each segment takes the candidate a predicted-mode tune report predicts
  cheapest over that segment's interval (default: one config throughout). A switch that changes
  `nproc_x` changes roundoff, so such a run is not bit-identical to a single-config run.
  Per-segment switching has **not** been run on hardware.
- `--cfl-rescue` (off by default, **PHYSICS-GRAY**): a segment that failed with a CFL line or NaN
  is rerun once per run from the same restart with `time_step` = the largest integer in
  [0.5, 0.75] × dt dividing the segment length and L; later segments return to the case's dt.
  Unit-tested decision logic only — **not validated on hardware**.

Evidence (dorian2019 first hour, 4 nodes 16×18): **R1** 30 + 30 min chain (jobs 1405 + 1410) vs
one-shot (job 1392): 01:00 `wrfout` for d01 and d02, **202 variables each, bit-identical**, no
NaN. **R3** `--dry-run` prints the plan and submits nothing. **R2** (each segment's wall inside
its predicted range): first attempt failed — predicted 478/578/548 s vs measured 326/198/168 s,
because the prediction had no io block and restart/history fell back to constants; fallbacks now
scale from measured scc anchors (conus25 restart, dorian2019 history) by grid points × levels.
The rerun with an io-block prediction is **PENDING**.

### `wtune tune --rank-by predicted` — rank by predicted whole-run time (decision 0055, opt-in)

```bash
wtune tune CASE --site scc --minutes 60 --nodes 4 --max-nodes 12 --rank-by predicted \
    [--max-points 3] [--max-wall-minutes M] [--restart-io-form 14|2] --out DIR
```

The default (`t0`) is unchanged. In predicted mode:

1. Plan points as `predict-rebuild` does (ERA5; spin-up = smallest whole minute ≥ 20 on the
   time-step grid, dt 36 s → 21 min) and rebuild every point once (real_only, 1 node each, in
   parallel — real.exe output does not depend on the candidates' decomposition).
2. Stage 1 at P* (the point with the largest sunlit fraction): each Slurm job spins up once and
   every candidate in it starts from that shared restart; ANCHOR runs first and ANCHOREND last in
   every job, and each run is normalised by its own job's anchors. Jobs with foreign load or a
   run that did not start from the restart are excluded.
3. Stage 2: top 3 + ANCHOR/ANCHOREND at every point in parallel, one `combine()` per candidate;
   the smallest predicted compute+init wins. Gates raise on: restart not used, |ANCHOREND ÷
   ANCHOR − 1| > 1%, foreign load, wrong first global step.

Candidate runs set `override_restart_timers = .true.`: gate G0 found that with
`restart_interval = 1000000` WRF still inherited the spin-up's 20-min restart interval and wrote
a restart inside every measured window (netCDF 299 s on a radiation step, r ≈ 5 → 121; ADIOS2
0.6 s). `run_seconds` is zeroed next to `run_minutes`, because WRF adds all four `run_*` keys.

Budget: `--minutes` counts **running** time only (queue time is reported separately; hard wall
cap `--max-wall-minutes`, default 3 × budget). Before anything is submitted a feasibility check
computes the least running budget (rebuild prep + one stage-1 job of ANCHOR + 1 candidate +
ANCHOREND + stage 2 for that one candidate) and refuses with that number. A stage-1 job starts
only if elapsed + its own estimate + the stage-2 reserve fits; stage-1 jobs are trimmed to fit
(floor ANCHOR + 1 + ANCHOREND); pending stage-1 jobs at the cutoff are cancelled and reported as
not measured; stage 2 drops its lowest candidates until it fits, or is skipped with the reason.
All estimates use one cost model: per run = startup + restart read + steps × s/step ×
(1 + ⌈steps/stepra⌉ (r_rad − 1)/steps), r_rad 3.93 (conus25, 82 steps: 1.107; the earlier blanket
×1.5 overstated it ~1.36×), with per-run overhead re-measured after the first wave.
`--max-points` defaults to 3 in this mode.

| gate / attempt | result |
|---|---|
| **G0** restart hand-off across rank counts (jobs 1371 io 14, 1372 io 2; conus25, 4 nodes, 82 steps after 20 min) | **pass.** ADIOS2 restart read at 288/144/72 ranks; candidate ÷ ANCHOR differs between io forms by ≤ 0.38% (gate 2%); in-allocation drift −0.18% / −0.11%; first global step 81 in all 10 runs. Refutes the code comment that io_form 14 restarts are decomposition-dependent. |
| e2e 1 (conus25-pred1) | **failed on budget**: queue time consumed the 60 min (6 free nodes, five 4-node rebuild jobs ran one at a time, ~22 min); stage-2 jobs 1393–1397 cancelled while PENDING |
| e2e 2 (conus25-pred2) | **failed**: rebuild took 7 min, then every stage-1 slot was skipped (7:04 + 22:26–26:11 per job + 37:39 reserve > 60 min) — nothing measured |
| e2e 3 (conus25-pred3: shrink-to-fit, feasibility check, one cost model) | **PENDING.** G1 (mechanism), G2 (winner's prediction ≤ 5% vs same-day run), G3 (winner ≤ 1.01 × the t0 winner G0014), G4 (≤ 60 min) not yet judged |

Node hygiene (every command in this section): `WTUNE_SBATCH_EXCLUDE=work5[,...]` adds nodes to
`--exclude=head`.
On 2026-09-27 scc work5 carried a root VS Code server outside Slurm (91% of one core) plus an
`rm -rf`, which slowed a 576-rank `wrf.exe` from 1.25 to 6.3 s/step; `--exclusive` does not stop
that. `campaign/node_hygiene.sh`'s census now prints `FOREIGN_LOAD` for any process not owned by
the job's user with lifetime %CPU > 20 and elapsed ≥ 30 s (kernel threads ignored).

### `wtune check` — physics tables and wrfinput sanity

- **C3 physics tables follow WRF 4.8's own `OPEN` statements**, not `README.physics_files`,
  which over-lists; each rule cites its source line. E.g. Ferrier (5/95) opens only
  `ETAMPNEW_DATA.expanded_rain`; `CCN_ACTIVATE.BIN` only for aerosol-aware Thompson (28/38); P3
  table 1 matched by prefix, table 2 only for nCat > 1 (52); full SBM (32) but not fast SBM (30,
  a different file set); ozone on d01 for RRTMG with `o3input=2` and on any CAM domain;
  `aerosol*.formatted` only for `aer_opt=1` on d01; `URBPARM_LCZ.TBL` *replaces* `URBPARM.TBL`
  when `use_wudapt_lcz=1`; GFDL (co2tf-dependent), Goddard, eclipse, slab/RUC/CLM/PX/SSiB,
  `HLC.TBL` for chem.
- **C10 wrfinput sanity uses land/water masks.** TSLB on land only: exactly 0 is real.exe's
  "undefined" fill → WARN; non-zero outside [200, 350] K → FATAL above 0.1% of the domain's land
  TSLB values, WARN below. SST on water only (exact 0 skipped: inland lakes); SMOIS < 0 → WARN,
  > 1 → FATAL; TSK everywhere; any NaN/Inf in T/U/V/QVAPOR/TSLB/SMOIS/PSFC → FATAL. The 0.1% line:
  dorian2019 d02, 20/(14806×4) = 0.034%, ran 1 h with no NaN (job 1392) → WARN; tw3km-2031 case_I,
  13186/(37588×4) = 8.8%, went NaN by hour 1 while printing SUCCESS → FATAL. On scc: 11 healthy
  truth dirs zero FATAL, case_I FATAL, case_A clean. The first C10 was false-FATAL on zhengzhou
  and dorian2019, both of which ran.
- **No "small patch hangs" rule.** The hypothesis (sc26p2 17×13 hangs at init with 10-cell
  patches) is refuted: 102 successful ledger runs had a minimum patch ≤ 10 cells (matthew 9×10
  and severe2023 10×10 at 9 cells, 97 and 109 runs; sc26p2 2×12 at 10 cells, 30 runs). Only
  sc26p2 17×13 hangs (7 of 7). The guard is per-candidate timeouts plus `watch` HANG_INIT.

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
| `wtune predict-rebuild` compute+init | +0.98% / −0.04% / −5.78% / +0.16% | conus25 / maria1km / zhengzhou / lafe on scc; ≈ ±2% after same-day drift correction |
| measured I/O probe | −0.5% / −0.5% / +3.1% of truth Σ writes | dorian2019 / maria1km / zhengzhou |
| ADIOS2 BP4 restart vs netCDF | write 1.05 s vs 323.5 s, read 2.3-2.5 s vs 120 s, bit-identical 60 min later; readable across 288/144/72 ranks | conus25, jobs 1277/1288/1289, 1371/1372 |
| `wtune run` 30 + 30 min restart chain vs one-shot | 202 + 202 variables bit-identical | dorian2019 d01+d02, jobs 1405+1410 vs 1392 |

## Measured-closed — do not re-propose

| direction | why it is closed |
|---|---|
| restart-boundary rebalancing / migration | restart ~56 s > 7-17 s recoverable per window; PLOV does it in place. That restart cost is netCDF: an ADIOS2 BP4 restart is 1.05 s write / 2.3-2.5 s read (conus25, jobs 1277/1288/1289), so `wtune run --tune-report` switches config at restart boundaries — whether a switch recovers more than it costs is not measured |
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
  (`julia --project=src/jlenv -e 'using Pkg; Pkg.instantiate()'`).
- PLOV: Intel compilers + Intel MPI and a WRF `dm+sm` build (`-qopenmp` live) with
  `libwrflib.a`; see `plov/build.sbatch`.

## Layout

The repo root is user-facing only; everything else lives under one of these directories
(2026-09-24 `src/` layout refactor — pure move + import/path fix, no behaviour change).

| path | status | contents |
|---|---|---|
| `wtune.py` | live | thin entry point: puts `src/` on `sys.path` and calls `wtune_cli.main()` |
| `bin/wtune` | live | runs `wtune.py` under the `envs/wtune` pixi env |
| `src/wtune_cli.py` | live | the ~1800-line CLI body (formerly `wtune.py` itself) |
| `src/wtune_physics.py`, `wtune_knobs.py`, `wtune_solar.py`, `wtune_surrogate.py`, `speedy_probe.jl`, `speedy_hotspot.jl`, `jlenv/`, `wind_probe.py`, `wtune_decomp.py`, `wtune_shape.py`, `gen_registry_catalog.py`, `wtune_catalog_registry.json` | live | advisory layer |
| `src/wtune_config_gate.py` (+ test) | live | calibration gate for result-affecting knobs |
| `src/wtune_transition_controller.py` | live (gate branch deprecated) | zero-measurement config + node-scaling projection |
| `src/wtune_feedback.py` (+ test), `wtune_probe.py` (+ test) | live | PLOV controller replay; ABBA probe harness |
| `src/wtune_rebuild_predict.py`, `wtune_watch.py`, `wtune_run.py`, `wtune_tune.py`, `wtune_preflight.py`, `tools/coldstart_sample.py`, `tools/era5_intermediate.py`, `tools/rescore_rebuild.py`, `tools/tc_bogus.py` | live | `predict-rebuild`, `watch`, `run`, `tune`, `check` and their helpers (section 6) |
| `src/wtune_oracle.py`, `wtune_oracle_train.py`, `wtune_segment_train.py`, `wrf_frame.py`, `featurize_frames.py`, `wtune_runtime_predict.py`, `wtune_runtime12.py`, `wtune_cost_model.py` | live | prediction layer |
| `campaign/` | **live, production** (unmoved) | the measurement side: `strict_campaign.sbatch`, `configs.py`, `cases.json`, `physics_templates.json`, the decomposition-search campaign and its rankers |
| `machine/` | **live, production** (unmoved) | hardware calibration data and probes (`machine/calibration/*.json`, `machine/bench/`) |
| `src/predictor/` | **live, production** | the decision side: ledger, benchmark, `recommend.py`, decomposition space and search, `stability.py`, `verify_obs.py` (`src/predictor/README.md`) |
| `src/fidelity_gate.py` (+ test) | live | chaos-floor scoring of a candidate against a reference and a twin |
| `src/fingerprint/` | live | out-of-process phase fingerprinter (autotuner step 2) -- moved into `src/` late in the refactor after `tests/test_wtune_fingerprint.py` proved it is imported, not dead |
| `src/tools/` | live | `bp_compression_check.py`, `run_tests.sbatch`, the ADIOS2→netCDF4 converter, `predict_validate/` |
| `src/analysis/` | live (not yet wired/tested) | wrf-python + TempestExtremes wrappers |
| `tests/` | live | every test (`tests/`, `tests/tools/`, `tests/predictor/` incl. fixtures) + `tests/conftest.py`, which puts `src/`, `src/predictor`, `src/tools` on `sys.path` |
| `results/derived-data/` | data (unmoved) | canonical processed CSV/JSON behind every reported number |
| `examples/`, `docs/` | unmoved | example configs; docs |
| `archive/` | **not imported by any live code or test** — see below | `plov/`, `research/`, `experiments/`, `profilers/`, `deprecated/` |

`archive/` holds code confirmed (by grep for real `import`/`sys.path`/`subprocess`
references, not just comment mentions) to have zero coupling to this repo's Python import
graph: `archive/plov/` (wrap trampolines, stagers, build/gate/harness scripts —
**its own `README.md` calls it "live, production"; it is a standalone C/asm rebalancer
built and deployed independently of the wtune Python tooling, not something this Python
refactor's "imported by live code or a test" test can certify as dead in a deployment
sense**), `archive/experiments/` (harnesses behind reported numbers, vendored with sha256
provenance — same caveat, see `archive/experiments/README.md`), `archive/research/`
(one-off studies behind wiki notes), `archive/profilers/` (PMPI `LD_PRELOAD` profilers),
`archive/deprecated/` (migration/restart, placement, hotspot model and their tests).
`fingerprint/` looked like a candidate too but is not archived (see above).

Running the suite: `python3 -m pytest -q tests/` (pixi env at `envs/wtune/`, or
`uv run --with pytest --with numpy --with netCDF4 --with scikit-learn --with catboost
python -m pytest -q tests/`) — **1657 passed / 14 skipped / 3 known pre-existing failures** (2026-09-27, 0.2.0)
(`tests/predictor/test_decomp_search.py::test_real_lafe_round2_incumbent_is_now_on_the_campaign_reference_scale`,
`tests/predictor/test_physics.py::test_template_engine_consistency_across_real_case_namelists`,
`tests/predictor/test_physics.py::test_physics_candidates_from_namelist_matches_text_engine_for_karina`).
`src/tools/run_tests.sbatch` runs the whole suite on a compute node the same way it always
did (`predictor/tests` needs the predictor pixi env; running it in the other one turns 18
skips into 19 lightgbm failures that look like a regression). The pieces, if you need one
directly: `pytest tests/test_wtune_*.py tests/test_fidelity_gate.py`, `pytest campaign
tests/tools archive/experiments`, `pytest tests/predictor` (predictor env), `pytest
archive/deprecated/` (archived), `python3 wtune.py selftest` (also runs
`wtune_compliance.validate()`), `python3 src/wtune_transition_controller.py selftest`
(38/40; both failures are in the deprecated imbalance-gate branch).

## Status

Developed against CONUS 2.5 km, a 12-case 3 km multi-regime corpus, and a 1500x1500 4-node
production case on f1/NCHC. The wall-time work is now in PLOV (-13%, bit-identical, no
source change); the remaining per-step budget is dynamics + a source-locked MPI floor, so
further gains come from the config layer behind the calibration gate, not from rebalancing.
