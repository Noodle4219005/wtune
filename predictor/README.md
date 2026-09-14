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
`y = log(s_per_step_median / s_baseline)`; baseline = most frequent config within
`(case_base, hardware_id, compliance_class)`, ties broken by smallest s/step; replicates averaged
in log space. `NOISE = log(1.005)` (measured 0.4-1.6% run-to-run floor) marks near-ties as
unreliable rather than real orderings.

Per held-out case (>=2 distinct configs): top-1/top-3 hit rate, `regret`, pairwise accuracy
(non-tie pairs), regression false-negative rate (`regress_1pct = y > log(1.01)`). Aggregates
report `n_eval_cases`, flag `UNDERPOWERED` below 5. STRICT/PHYSICS-GRAY reported separately;
**RESEARCH rows (PLOV, wtimer-wrapped) are never evaluated**, only kept in the ledger, flagged.

## Validation schemes (`validate.py`)
Never random splits: leave-one-run-out (weak/optimistic baseline), leave-one-case-out (the
leakage-safe default), leave-one-family-out (documented heuristic regime grouping, see `ledger.py`
`_FAMILY_MAP`), leave-one-hardware-out (raises `NotEnoughGroups`: one machine so far, `f1-spr-icpnq`).

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
(all three run through the pixi env in practice; see `run_benchmark.sbatch`, written, not submitted)

## Honesty: expect UNDERPOWERED for STRICT config ranking
Almost every row uses **448 ranks, 8x56, numtiles=8, 901x901** (or the CONUS 1501x1201 grid's own
single shipped/rec pair). Only **two** cases have real STRICT sweeps: **CONUS** (decomposition
sweep + numtiles 8-vs-16) and **karina** (224x2 elastic-core/numtiles-12-13-14 arms, ~30 rows).
Every other case_base has 1-2 STRICT rows -- enough for the cost baseline, not for cross-case
config generalisation; expect `leave-one-case-out`/STRICT to be `UNDERPOWERED` until more sweeps
land. PHYSICS-GRAY (the `mp`x`bl` scheme grid) has more cases but no fidelity labels, so its
recommendations stay UNCONFIRMED regardless of measured regret.
