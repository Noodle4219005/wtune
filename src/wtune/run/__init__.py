"""wtune run: segmented WRF executor with restart hand-offs. CASE_DIR's run is cut into
segments; segment k runs in OUT/seg_<k> as its own Slurm job and hands off to k+1 through the
restart files it writes at its end.
Layers, bottom up: _deps, common -> geometry -> estimate -> layout -> plan -> namelist,
staging, sbatch, inputs -> context -> report -> attempts -> controller -> cli. Every name of
the former single module is re-exported here."""
from ._deps import (  # noqa: F401
    _HERE, _REPO_ROOT, _CAMPAIGN_DIR, coldstart, sweep, iom, wp, pp, resume, tune,
)
from .common import (  # noqa: F401
    WRF_FMT, ADIOS2_IO_FORM, _TIME_PARTS, _PER_RECORD_KIND_RE, RunError, UNSET,
    Record, to_plain, _fmt, _hms, parse_hms, _frac, _frac_lcm,
)
from .geometry import (  # noqa: F401
    HistoryStream, Geometry, case_geometry, domain_start, step_time, _step_fraction, step_of,
    resolve_end_step,
)
from .estimate import (  # noqa: F401
    TIME_FACTOR, TIME_PAD_S, TIME_ROUND_S, FALLBACK_INIT_S, FALLBACK_RADIATION_FACTOR,
    FALLBACK_RESTART_IO_S, FALLBACK_HISTORY_FRAME_S, CONUS25_RESTART_POINTS,
    RESTART_IO_ANCHOR_S, DORIAN_HISTORY_WRITE_S, DORIAN_HISTORY_POINT_FRAMES,
    HISTORY_S_PER_POINT_FRAME, Overheads, overheads, fallback_restart_io_s,
    fallback_history_frame_s, fallback_overheads, compute_fn_from_points, constant_compute_fn,
    history_frames, writes_restart, restart_write_steps, segment_wall, history_wall,
    time_limit_s, fallback_s_per_step, _parse_at, model_from_prediction,
    compute_fn_from_tune_row,
)
from .layout import (  # noqa: F401
    Config, Candidate, TuneLayout, Layout, make_config, tune_candidates, tune_report_layout,
    resolve_layout, rank_mapping_notes,
)
from .plan import (  # noqa: F401
    WALL_FRACTION, RESTART_OFF_MIN, Segment, RestartSetting, RecordAlignment,
    plan_segments_fixed, segment_steps_for_minutes, choose_restart_quantum, assign_segments,
    input_record_alignment, missing_record_files, plan_run, case_restart_interval_s,
    checkpoint_request, checkpoint_steps_for_minutes, auto_checkpoint_steps,
    _checkpoint_setting, restart_setting, restart_times, restart_interval_minutes,
)
from .namelist import (  # noqa: F401
    _time_values, _set_one, render_segment_namelist, namelist_diff_summary,
)
from .staging import (  # noqa: F401
    _CASE_LINK_SKIP, _WRF_RUN_SKIP, _WRF_RUN_SKIP_PREFIX, _XML_IO_RE, FRESH_BP_SUBDIR,
    merge_adios2_xml, render_adios2_restart_xml, restart_files, _stamps, output_dir_names,
    precreate_output_dirs, remove_empty_output_dirs, _case_link_ok, _missing_namelist_files,
    fresh_bp_links, setup_segment_dir,
)
from .sbatch import render_segment_sbatch  # noqa: F401
from .inputs import (  # noqa: F401
    InputPlan, find_project_dir, bp_cache_dir, adios2_streams_or_raise, _per_record_of,
    io_plan_line, plan_inputs, convert_inputs, _logging_copier,
)
from .context import RunOptions, RunPlan, Runners, RunState, RunContext  # noqa: F401
from .report import _write_report, _save_report, new_report  # noqa: F401
from .attempts import (  # noqa: F401
    UNKNOWN_POLLS_MAX, Attempt, AttemptRecord, Outcome, attempt_dir_name, _attempt_times,
    _attempt_namelist, _restart_links, _attempt_xml, _stage_attempt, _submit_attempt,
    wait_for_job, _run_attempt,
)
from .controller import (  # noqa: F401
    CFL_RESCUE_FACTOR, CFL_RESCUE_MIN_FACTOR, rescue_time_step, rsl_has_cfl, resume_time_limit,
    _resume_attempt, _rescue_dt, _cfl_rescue_attempt, _cfl_advice, _next_attempt,
    run_controller, _update_from_observed, _make_sidecar,
)
from .cli import (  # noqa: F401
    DESCRIPTION, build_arg_parser, parse_options, build_plan, print_plan, print_dry_run,
    execute, main,
)
