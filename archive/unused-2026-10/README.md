# unused-2026-10 — code no current command reaches

Moved on 2026-10-09 (`git mv`, history kept). Paths below this directory are the original
ones (`src/...`, `tests/...`), so anything here can be moved back unchanged.

How the list was made: a full `wtune submit -O2` on SC26-P1 (project init, doctor, check,
advise, calc, tune, run, observation fetches, post, watch) was run with every Python function
call recorded; none of these files was entered, and no file outside this list imports them
or names them as a script.

| what | files |
|---|---|
| temporal-validation research harness | `src/tools/state_timeline.py`, `temporal_validation.py`, `temporal_run.sbatch` |
| per-rank fingerprint sampling | `src/fingerprint/` |
| earlier runtime/cost models | `src/wtune_transition_controller.py`, `wtune_segment_train.py`, `wtune_cost_model.py`, `wtune_runtime12.py`, `wtune_runtime_predict.py` |
| prediction-validation harness | `src/tools/predict_validate/`, `rescore_rebuild.py`, `run_tests.sbatch` |
| offline physics-model evaluation | `src/predictor/physics_holdout_eval.py`, `stability.py`, `run_benchmark.sbatch`, `run_physics_benchmark.sbatch` |
| typhoon vortex bogussing (failed its phase-5 criteria) | `src/tools/tc_bogus.py` |
| small tools | `src/wtune_probe.py`, `featurize_frames.py`, `wind_probe.py`, `src/tools/bp_compression_check.py` |

Their tests are under `tests/` here and are not part of the main suite.
Known defects found by the 2026-10-09 review in these files were not fixed.
