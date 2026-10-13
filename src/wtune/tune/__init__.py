"""wtune tune: the budgeted on-cluster layout tuner for a WRF case on scc (decisions 0055, 0059).

Layers, bottom up: _deps, common, context -> budget, jobs -> candidates, points, prediction ->
rounds, report, node_search -> pipeline (+ physics_phase) -> cli. Every name of the
former single module is re-exported here; names tests monkeypatch are called through their
defining module (`_jobs.wait_for_jobs`), so patch them there (wtune.tune.jobs.wait_for_jobs)."""
from ._deps import (  # noqa: F401
    _HERE, sweep, wc, pred_targets, predict, probe_plan, io_model, coldstart, physics_advice,
    cost_drivers, campaign_configs, decomp_space,
)
from .common import (  # noqa: F401
    ANCHOR_END_ID, is_set, if_set, Record, _fmt_hms, _fmt_sbatch_time, parse_run_id,
    compute_slots, would_exceed_budget, chunk_round_robin, _parse_slurm_elapsed,
)
from .context import (  # noqa: F401
    DEFAULT_STEPS, DEFAULT_SEED, DEFAULT_KAPPA,
    CONFIRM_REPS, DEFAULT_POLL_INTERVAL_S, DEFAULT_STARTUP_S, QueueCapExceeded, RealClock,
    default_scancel_runner, default_ledger_reader, Ctx, resolve_case, resolve_auto_probe_steps,
    _total_steps, cores_used, _namelist_dt,
)
from .budget import (  # noqa: F401
    RANKING_R_RAD, RESTART_READ_S_BY_IO, estimate_run_time, measure_startup_s,
    update_s_est, _fallback_s_per_step, rebuild_segment_timeout_min, rad_factor,
    ranking_cost_model, stage2_concurrency, anchor_end_of, wrap_with_anchors, ranking_context,
    ranking_export,
)
from .jobs import (  # noqa: F401
    PROGRESS_BAR_WIDTH, ANCHOR_DRIFT_MAX, MAXRSS_NOTE, print_progress, cancel_jobs,
    wait_for_jobs, submit_configs_job, evaluate_job_gates, job_timing, _maxrss_mb_by_job,
)
from .candidates import (  # noqa: F401
    FULL_SPACE_CONFIGS_PER_JOB, _apply_search_filter, sample_more_diverse_configs,
    aggregate_by_cid, rank_predicted_entries, enumerate_full_space,
)
from .estimates import (  # noqa: F401
    LEVELS, DEFAULT_LEVEL, EST_REBUILD_S, EST_REFINE_ROUNDS, EST_ADAPTIVE_ROUNDS,
    EST_ROUND0_CONFIGS, EST_SPINUP_STEPS, EST_PHYSICS_RUNS, EST_PHYSICS_POINTS,
    layout_estimates, points_estimate, level_estimates, highest_level_within,
    format_level_estimates,
)
from .points import (  # noqa: F401
    PREDICTED_FORCING, PREDICTED_SPINUP_MIN, POINT_MODES, ranking_job_timeouts,
    choose_primary_point, rebuilt_case_id, rebuilt_steps, rebuilt_measure_steps,
    write_rebuilt_case_entry, prepare_rebuilt_points, submit_rebuilt_points,
    finish_rebuilt_points, point_runs, point_export, choose_point_mode, _measure_run,
    measure_candidates_at_points, rank_by_predicted_total, predicted_spinup_min,
    plan_predicted_points, fit_stage2, pick_precision_points, pick_points,
)
from .prediction import (  # noqa: F401
    COLD_START_BAND, FALLBACK_BANDS, TIME_STEP_S, write_probe_ledger, predict_candidates,
    _cost_series_for_ctx, run_rebuild_points, build_full_run_prediction, _prediction_context,
    WholeRunPrediction, add_run_length_io, _fallback_point_plan, _case_types_and_track,
    plan_prediction_points, time_suggestion_s, pipeline_run_length_io, prediction_init,
    cold_start_prediction,
)
from .rounds import (  # noqa: F401
    ROUND0_BUDGET_FRACTION, ADAPTIVE_STOP_FRACTION, ADAPTIVE_MAX_ROUNDS,
    KAPPA_ESCALATION_FACTOR, KAPPA_MAX_ESCALATIONS, Wave, select_confirm_entries,
    _submit_round0_wave, run_round0, _default_proposer, run_adaptive_rounds, run_confirm,
    run_tuning_at,
)
from .preflight import (  # noqa: F401
    PREFLIGHT_TIMEOUT_S, SINFO_TIMEOUT_S, HEALTH_BAD_STATES, DEFAULT_WRF_TREE, Runner,
    default_preflight_runner, node_preflight, render_node_health_text,
)
from .report import (  # noqa: F401
    LAYOUT_ROWS_SHOWN, local_param_table, _assert_report_is_strict_only, build_report,
    render_report_text, render_node_selection_text, render_pipeline_text, _attach_advice,
    final_winner, write_report_json,
)
from .node_search import (  # noqa: F401
    NODE_SWEEP_REPS, DEFAULT_SWEEP_BUDGET_FRAC, KNEE_MARGINAL_SPEEDUP, FinalistOutcome,
    NodeSelection, choose_nodes_min, choose_nodes_cap, choose_nodes_knee, NODES_POLICIES,
    choose_nodes, run_node_sweep, run_core_sweep, node_choices, run_node_search,
    build_layout_table,
)
from .pipeline_state import (  # noqa: F401
    NOT_RUN_BUDGET, ClockMark, PhaseRecord, StepDown, LevelInfo, RebuildInfo,
    PredictionPlanRecord, Phase2Plan, ShortRun, PredictionBand, PipelineSummary, PointPreview,
    ColdEstimate, RebuildSubmission, MeasuredLayout, PipelineRun,
)
from .physics_phase import (  # noqa: F401
    PIPELINE_PHYSICS_MAX_CANDIDATES, PIPELINE_PHYSICS_POINTS, PHYSICS_RUNNER_MARKER,
    WINDOW_FRAME_LEVEL, MIN_WINDOW_MIN, physics_runner_supported, phase2_writes_frames,
    build_physics_window_namelists,
)
from .pipeline import (  # noqa: F401
    PIPELINE_FRACS, PIPELINE_MAX_QUEUE_HOURS, PIPELINE_DEFAULT_MINUTES, PRECISION_DEFAULT_MIN,
    pipeline_layouts, phase2_plans, run_pipeline_full, run_level0,
)
from .cli import (  # noqa: F401
    run, build_arg_parser, main,
)
