# WTune configuration predictor -- phase A

Goal: on a barely-run new WRF case, quickly exclude bad configurations and recommend top
candidates with high confidence. Metric: `regret = runtime(predicted)/runtime(best) - 1`
(target usually 1-2%), not absolute s/step.

## Levels
- **Level 0** (implemented): `case_features` (grid/physics/cadence/hardware) + `config_features`
  (parallel geometry/runtime knobs). See `features.py`.
- **Level 1** (implemented): Level 0 + `profile_features` (per-rank micro/radiation/dynamics/mpi/
  io share, imbalance). `validate.profile_ablation()` reports "no profile data" when the ledger
  has none (true for every current source -- see Honesty below).
- **Level 2** (NOT implemented): a Gaussian-process model over the full feature space.

## Targets and metrics (`targets.py`, `validate.py`)
`y = log(s_per_step_median / s_baseline)`; baseline = most frequent config within the decision
group (`features.decision_group_key`), ties broken by the outcome-independent
`targets._reference_order` (plain-MPI default first, then lexical -- C01 for the campaign), never
by measured speed (that made the naive baseline the oracle, 2026-09-14); replicates averaged
in log space. `NOISE = log(1.005)` (measured 0.4-1.6% run-to-run floor) marks near-ties as
unreliable rather than real orderings.

Per held-out case (>=2 distinct configs): top-1/top-3 hit rate, `regret`, pairwise accuracy
(non-tie pairs), regression false-negative rate (`regress_1pct = y > log(1.01)`). Aggregates
report `n_eval_cases`, flag `UNDERPOWERED` below 5. STRICT/PHYSICS-GRAY reported separately;
**RESEARCH rows (PLOV, wtimer-wrapped) are never evaluated**, only kept in the ledger, flagged.

## Validation schemes (`validate.py`)
Never random splits: leave-one-run-out (weak/optimistic baseline), leave-one-case-out (the
leakage-safe default), leave-one-family-out (documented heuristic regime grouping, see `ledger.py`
`_FAMILY_MAP`), leave-one-hardware-out (two pools now: `f1-spr-icpnq` = ct448 SNC 8x14 and
`f1-spr-icpnp` = cf448 2 NUMA; benchmark 1075654 shows pairwise accuracy 0.37-0.54, i.e. do not
recommend across pools).

Uncertainty: a grouped CV+/jackknife+ conformal interval on `y` (documented simplification: one
model per left-out CASE, not per point -- see `validate.conformal_intervals`), with empirical
out-of-fold coverage at 90%/95%, plus a CatBoost `Quantile(0.05/0.95)` variant when available. Per
recommendation: predicted gain `= 1 - exp(y_pred)`, label **SAFE** when even the pessimistic
(upper-bound) end of the 95% interval still beats baseline, else **EXPERIMENTAL**.

## Compliance (`ledger.py`, via the repo root's `wtune_compliance.py`)
A row is STRICT if every config knob differing from its case's baseline is STRICT under
`wtune_compliance.classify()`, PHYSICS-GRAY if any differing knob is a physics scheme, always
RESEARCH for PLOV/wtimer-wrapped/`--wrap` binaries regardless of knobs. The fidelity-risk model
for PHYSICS-GRAY is separate from the performance model and returns **UNCONFIRMED** for every
candidate when no `fidelity_label` exists (true today: no adapted source carries one).

## Commands
```
python3 predictor/ledger.py build --wiki-data /home/u5727520/my-llm-wiki/data --out predictor/ledger/runs.csv
python3 predictor/ledger.py validate --ledger predictor/ledger/runs.csv
python3 predictor/benchmark.py --ledger predictor/ledger/runs.csv --out <dir>
pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml python -m pytest predictor/tests -v
```
(all run through the pixi env on a compute node in practice; `sbatch predictor/run_benchmark.sbatch`
builds the ledger with the cf448 campaign ledgers 1075293-1075304 via `CAMPAIGN_GLOB`, drops the
removed training cases `EXCLUDED_CASE_BASES`, and skips campaign correctness-mode rows)

## Shortlist command (`recommend.py`, archived)
Moved 2026-10-10 to `archive/unused-2026-10/src/predictor/recommend.py` (no `wtune` command reached
it, and its verify plans print `MODE=physics` commands `campaign/strict_campaign.sbatch` no longer
takes). The same happened to the `experiment.py` plan/record/reconcile engine
(`experiment_legacy.py` there); `experiment.py` keeps only `_generated_baseline` and `_format_hms`.
What follows describes the archived command.

The tuner only narrows the search: every output ends with the verify plan and a statement that the
user makes the final choice. Nothing is applied automatically.

- **STRICT:** `recommend.py --ledger runs.csv --namelist N --case-name C --hardware-id H [--nodes 4] [--max-dom M] [--holdout-case C] [--top-k 5] [--json OUT]`
  - Ranks the legal campaign templates for the grid with `catboost_pairwise`.
  - Adds a `catboost_point` magnitude (% vs C01) with a conformal interval and a SAFE/EXPERIMENTAL label.
  - Evidence warnings: noise floor, same-hardware training counts, nearest cases with hardware, out-of-range grid, IN-SAMPLE.
  - Verify plan: measure top-3 plus the reference in one allocation, palindromic.
  - Out-of-sample check (`--holdout-case`, jobs 1076102/1076103): lafe #1 measured 3/16 (+2.2% regret); conus12 #1 measured 6/16 (+16.4%).
- **PHYSICS-GRAY:** `recommend.py --compliance PHYSICS-GRAY ... [--decomp Cxx] [--physics-ledger P ...] [--allowed-labels WITHIN-FLOOR,SMALL-DRIFT] (--probe-screen SCREEN.txt | --run-probe --latlon LAT,LON)`
  - Candidates come from `campaign/physics_templates.json` (P01–P06) applied to the user's namelist through `campaign/configs.apply_physics_template`, the single template engine.
  - New cases get per-case pairing from `physics.generic_extra_edits`; a test keeps it consistent with the JSON for the six campaign cases.
  - The SpeedyWeather probe always runs (or is read from a saved `wtune.py screen` output); it is marked invalid below dx 1 km.
  - Status RECOMMENDABLE needs all of: predicted fidelity label allowed, probe not REJECT, same-hardware physics data, a performance prediction. Everything else is UNCONFIRMED with reasons. Until a physics campaign exists, every candidate is UNCONFIRMED.
  - Fidelity labels mean model self-drift against a same-physics twin, not observed forecast skill.

## PHYSICS-GRAY model (`physics.py`)
- **Grouping:** physics groups `(case_base, hardware_id, n_nodes, total_cores, decomp)`, with P00 at DECOMP as the baseline.
- **Targets:** performance `y = log(s_candidate/s_P00)`; fidelity `log10(nd_max)` and the ordinal label `WITHIN-FLOOR < SMALL-DRIFT < LARGE-DRIFT` from `../fidelity_gate.py`.
- **Evaluation:** leave-one-case-out and leave-one-template-out; regret restricted to allowed labels; dangerous-miss rate (predicted allowed, measured LARGE-DRIFT); probe-verdict confusion.
- **Benchmark:** `sbatch --export=ALL,PHYSICS_LEDGERS="..." predictor/run_physics_benchmark.sbatch` (refuses to run without physics campaign ledgers).

## Decomposition space (`decomp_space.py`, research step D1)
- **Enumeration:** the full legal stock-WRF space per case — ranks×threads, every `nproc_x×nproc_y`, tiling (`numtiles`, `numtiles_x/y`, `tile_sz`, 1D/2D), distribution, `KMP_BLOCKTIME`.
- **Exact WRF rules:**
  - patch legality `e_we/nproc >= 10` with integer division (`share/module_check_a_mundo.F:3014-3029`)
  - RSL-lite remainder at both ends (`external/RSL_LITE/task_for_point.c`)
  - recursive tile split (`frame/module_tiles.F`, `frame/module_machine.F`)
- **Space size (cf448, 448 cores):** 2,576–4,072 candidates per case.
- **Sampling and running:** `decomp_space.py sample CASE --n N --seed S --out F.json` gives a stratified ground-truth sample; `CONFIG_FILE=F.json` makes `campaign/strict_campaign.sbatch` measure it.

## Status 2026-09-14: first STRICT campaign landed
cf448 campaign (6 cases x 16 templates x 2 reps) -> benchmark 1075654, leave-one-case-out over the
6 campaign groups: catboost_pairwise median regret 0.4% (naive always-C01 15.2%), catboost_point
1.0%, ridge 5.2%; conus12 (smallest grid) ~= naive for every model; conformal intervals
over-cover (0.996 at 90%/95%). Exploratory, n=6 -- the section below describes the pre-campaign
corpus and is kept for history.

## Honesty (pre-campaign corpus): expect UNDERPOWERED for STRICT config ranking
Almost every row uses **448 ranks, 8x56, numtiles=8, 901x901** (or the CONUS 1501x1201 grid's own
single shipped/rec pair). Only **two** cases have real STRICT sweeps: **CONUS** (decomposition
sweep + numtiles 8-vs-16) and **karina** (224x2 elastic-core/numtiles-12-13-14 arms, ~30 rows).
Every other case_base has 1-2 STRICT rows -- enough for the cost baseline, not for cross-case
config generalisation; expect `leave-one-case-out`/STRICT to be `UNDERPOWERED` until more sweeps
land. PHYSICS-GRAY (the `mp`x`bl` scheme grid) has more cases but no fidelity labels, so its
recommendations stay UNCONFIRMED regardless of measured regret.
