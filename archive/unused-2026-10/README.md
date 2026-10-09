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

## Retired commands (2026-10-10)

Nine early in-process `wtune` subcommands the current workflow no longer uses were removed from
`src/wtune_cli.py`: `plan`, `hotspot`, `oracle-map`, `restart-window`, `feedback`, `knobs`,
`catalog`, `audit`, `profile`. A tenth, `screen`, was meant to go with them but is still wired:
`src/predictor/recommend.py` (`run_probe_screen`) runs `wtune.py screen --json` as a subprocess.

| what | files |
|---|---|
| their parsers, handlers and command-only helpers, verbatim as at 18a9b9f (one file, not wired to anything, not importable as is) | `src/wtune_cli_legacy.py` |
| PLOV feedback controller (only `feedback` imported it) | `src/wtune_feedback.py`, `tests/test_wtune_feedback.py` |
| the two tests of `wtune profile` / `wtune knobs`, cut from `tests/test_wtune_case.py` and `tests/test_wtune_compliance.py` | `tests/test_retired_cli_commands.py` |

Left in place because live code still reaches them: `src/wtune_surrogate.py` (`wtune physics`
and `screen`), `src/wtune_oracle.py` and `src/wrf_frame.py` (`wtune_oracle_train.py` ->
`tools/state_correction_model.py`), `src/wtune_knobs.py` (no importer left;
`tests/test_wtune_compliance.py` reads its source), `src/wtune_case.py` (tune, calc, advise).
