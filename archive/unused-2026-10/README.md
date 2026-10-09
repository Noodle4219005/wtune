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

## Retired `wtune tune --rank-by predicted` (2026-10-10)

The older predicted-ranking mode (stage 1 search, stage 2 at every rebuilt point) and its flags
`--rank-by`, `--stage1`, `--stage2-minutes`, `--max-wall-minutes` were removed from `wtune tune`.
`-O0..-O3`, `--nodes N|auto|search` and `predict-rebuild` are unchanged: the pipeline goldens
(`tests/fixtures/golden/wtune_tune/pipeline_*`, `pkg_pipeline_*`, `fixed_nodes`, `node_search`,
`level0`) pass byte-identical.

| what | files |
|---|---|
| the mode itself (`run_predicted`, its report) | `src/wtune_tune/predicted.py` |
| functions cut from `cli.py`, `budget.py`, `jobs.py`, `rounds.py`, `prediction.py`, verbatim as at 1fa9fe8 (one file, not importable as is); includes the dead `run_c3_prediction`, `run_trajectory_points`, `build_ranking_sbatch_command`, `_match_points`, `select_confirm_configs` | `src/wtune_tune/predicted_hooks_legacy.py` |
| their tests, cut from `tests/test_wtune_rank_predicted.py`, `test_wtune_tune.py`, `test_wtune_ranking_submit.py`, `test_wtune_tune_fresh_inputs.py` and the two `test_refactor_golden_wtune_tune*.py` (one file, not runnable as is) | `tests/test_retired_predicted_mode.py` |
| golden snapshots of the mode and of `run_c3_prediction` | `tests/fixtures/golden/wtune_tune/predicted_*`, `pkg_run_c3_prediction.txt` |

The branches removed from inside functions that stayed (`submit_configs_job`, `_job_time_est`,
`_submit_round0_wave`, `_round0_size`, `run_round0`, `_plan_round_job`, `_wait_running_budget`, the
`Ctx` fields) are in 1fa9fe8's history only.

Left in place: the point-job code `-O1..-O3` use (`points.py`, `ranking_cost_model`,
`ranking_context`, `ranking_export`, `wrap_with_anchors`, `stage2_concurrency`, `fit_stage2`,
`evaluate_job_gates`, the running budget); `run_rebuild_points` (`predict-rebuild`); the
`probe.rank_by` / `stage1` / `stage2_minutes` keys of `src/wtune_config.py`, still accepted so
existing YAML files load (`rank_by: predicted` is now refused by `wtune tune`).
