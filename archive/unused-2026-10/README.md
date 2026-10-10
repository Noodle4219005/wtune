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
(`recommend.py` was archived later the same day; see the last section.)

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

## Retired probe-script modes (2026-10-10)

`campaign/strict_campaign.sbatch` went from 1797 to 1213 lines: `MODE=physics`, `MODE=correctness`,
the fixed C01-C16 run set (no `CONFIG_FILE`) and the `CONFIGS=` subset filter were deleted. The
script now takes `MODE=timing` with a required `CONFIG_FILE`, which is all `wtune tune` submits.

| what | files |
|---|---|
| the script as at 30572ce, byte-identical (runs as before from this path if `CAMPAIGN` is set) | `campaign/strict_campaign_full.sbatch` |
| `configs.py` subcommands only the physics block called (`decomp-pair`, `physics-order`, `physics-describe`, `fidelity-tokens`), verbatim, not importable as is | `campaign/configs_cli_legacy.py` |
| their tests and the script-text tests of the physics block, cut from `campaign/test_configs.py` (not runnable as is) | `campaign/test_configs_legacy.py` |

Their entries were dropped from `tests/fixtures/golden/configs/{cli,api,surface}.json`; the only
changed values there are the three usage texts.

Left in place: the helpers those subcommands called (`resolve_decomp_pair`, `physics_ledger_columns`,
`probe_verdicts_for_case`, `format_physics_edits`, `describe_field`, `fidelity_notes_tokens`; still
exported by `configs.py` and tested), `edit-namelist --physics-template` and `--history-mode interval`.

## Retired shortlist command and experiment engine (2026-10-10)

`src/predictor/recommend.py` and most of `src/predictor/experiment.py` printed `MODE=physics` and
`MODE=timing,CONFIGS=...` sbatch command lines that the trimmed probe script refuses. No `wtune`
command, tuner module, campaign script or shell script imported or ran either one (grep for
imports, `recommend.py` / `experiment.py` as a script name, `importlib` / `pkgutil` /
`__import__`); only `src/tools/scc_sweep.py` uses `experiment._generated_baseline` and
`experiment._format_hms`.

| what | files |
|---|---|
| the shortlist command, whole file | `src/predictor/recommend.py` |
| its tests and golden snapshots | `tests/predictor/test_recommend.py`, `tests/test_refactor_golden_recommend.py`, `tests/fixtures/golden/recommend/` |
| four `recommend` tests cut from `tests/predictor/test_verify_obs.py` (not runnable as is) | `tests/predictor/test_verify_obs_recommend_legacy.py` |
| `experiment.py` minus the five names below, verbatim as at 35c1c10 (plan, record, screen, reconcile, summarize, `_sbatch_command`; not importable as is) | `src/predictor/experiment_legacy.py` |
| its tests (not runnable as is) | `tests/predictor/test_experiment_legacy.py` |

Left in place: `src/predictor/experiment.py` with `_format_hms`, `_import_decomp_space`,
`_max_legal_ranks`, `_generated_baseline`, `GENERATED_BASELINE_RULE` (1550 -> 115 lines) and their
three tests; every other `src/predictor/` module (`physics.py` via `wtune_physics_advice`,
`verify_obs.py` via `analysis/tc_track.py`, `validate.py` via `benchmark.py`, the rest via
`decomp_search.py`). `wtune screen` has no caller left except its own CLI entry and tests; it was
not removed.

## Unreachable files and functions (2026-10-10, second pass)

Checked by grep over `src campaign bin tests docs README.md`: no import, attribute access or
script invocation outside tests reaches any of these. Cut at 33a1ef7.

| what | files |
|---|---|
| per-rank cost oracle and its frame reader, whole files; `ridge_fit` / `ridge_pred`, the only names still used, were copied into `src/tools/state_correction_model.py` | `src/wtune_oracle.py`, `src/wtune_oracle_train.py`, `src/wrf_frame.py` |
| knob advisor, whole file (no importer), and the catalog parity checks that read its source, cut from `tests/test_wtune_compliance.py` (not runnable as is) | `src/wtune_knobs.py`, `tests/test_wtune_knobs_parity_legacy.py` |
| `scc_sweep.py` plan/submit/status command line and manifest, verbatim, not importable as is; its tests, cut from `tests/tools/test_scc_sweep.py` | `src/tools/scc_sweep_cli_legacy.py`, `tests/tools/test_scc_sweep_cli_legacy.py` |
| SpeedyWeather hotspot runner and sampler cut from `wtune_surrogate.py` (`run_speedy`, `sample_domain`, `convective_summary`, helpers, `DEFAULT_*`) | `src/wtune_surrogate_legacy.py` |
| shape advisor cut from `wtune_shape.py` (`tiling`, `factor_pairs`, `evaluate`, `rank`, `rank_nested`, `even_grids`, `validate`, `main`) | `src/wtune_shape_legacy.py` |
| `grib_state.py` `trajectory` command, CSV writer, patch-imbalance helpers, `cloud_frac_*`, `precip_rate_from_apcp_diff`; their tests, cut from `tests/tools/test_grib_state.py` and `tests/test_review_fixes_core.py` | `src/tools/grib_state_legacy.py`, `tests/tools/test_grib_state_legacy.py` |

Left in place: the `scc_sweep.py` library `wtune tune` / `wtune run` use, with the CLI-only
constants (`DEFAULT_K`, `DEFAULT_ROOT`, `MANIFEST_FIELDS`, ...); `wtune_surrogate.py`
`load_probe`, `analyze_probe`, `probe_steering_wind`, `speedy_vet` (`wtune physics`, `screen`);
`wtune_shape.py` `patch_dims` and the measured constants (`wtune_case.py`); `grib_state.py`
`target_grid`, `regrid_nearest_regular`, `precip_frac`, `precip_rate_from_apcp_window`
(`coldstart_sample.py`, `case_rebuild.py`). Not moved although nothing calls them now:
`grib_state.py` `precip_mean`, `cos_zenith_grid`, `sunlit_frac`, `pblh_*`, `wind10_max`, `hfx_mean`,
`load_case`, `resolve_times` (still tested), and `src/speedy_hotspot.jl` (only `run_speedy` ran it).
