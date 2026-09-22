#!/usr/bin/env python3
"""pytest suite for tools/state_timeline.py. Pure-function feature math, the multi-domain
radiation-aware rsl parser, namelist/iofields editing, and patch mapping cross-checked against
predictor/decomp_space.py -- all exercised over plain python lists/dicts, no netCDF4, no numpy, no
cluster access (extract()'s netCDF path is deliberately NOT covered here -- see this module's own
docstring on lazy netCDF4 import). Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q tools/test_state_timeline.py tools/test_temporal_validation.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_timeline as st  # noqa: E402
import decomp_space          # noqa: E402

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45, dt=20, prep_dir="/nonexistent", bp_dir="",
                     input_files=[], case_family="synthetic", steps=20, enabled=True)

NML_1DOM = """&time_control
 run_days = 0,
 run_hours = 12,
 run_minutes = 0,
 restart = .false.,
 restart_interval = 5000,
 io_form_history = 2
 io_form_restart = 2
 io_form_input = 2
 io_form_boundary = 2
/
&domains
 max_dom = 1,
 e_we = 901,
 e_sn = 901,
/
&physics
 mp_physics = 8, ra_lw_physics = 4, ra_sw_physics = 4, radt = 9,
/
&namelist_quilt
 nio_tasks_per_group = 4,
 nio_groups = 1,
/
"""

NML_2DOM = NML_1DOM.replace("max_dom = 1,", "max_dom = 2,").replace(
    "radt = 9,", "radt = 9, 3,").replace(
    "/\n&namelist_quilt", " parent_time_step_ratio = 1, 3,\n/\n&namelist_quilt")


def _rsl_line(step_second, domain, elapsed):
    return ("Timing for main: time 2018-08-25_00:%02d:00 on domain %4d:   %8.5f elapsed seconds"
            % (step_second, domain, elapsed))


# ---------------------------------------------------------------------------
# 1. rsl parsing, multi-domain
# ---------------------------------------------------------------------------
def test_parse_step_series_multi_domain_keeps_each_domains_own_series():
    text = "\n".join([
        _rsl_line(0, 1, 2.0), _rsl_line(0, 2, 0.4),
        _rsl_line(1, 1, 1.8), _rsl_line(1, 2, 0.5),
        _rsl_line(2, 1, 2.2),
    ])
    out = st.parse_step_series(text, max_dom=2)
    assert out[1] == [2.0, 1.8, 2.2]
    assert out[2] == [0.4, 0.5]


def test_parse_step_series_ignores_domain_beyond_max_dom():
    text = "\n".join([_rsl_line(0, 1, 2.0), _rsl_line(0, 3, 9.9)])
    out = st.parse_step_series(text, max_dom=1)
    assert out == {1: [2.0]}


def test_parse_step_series_empty_text():
    assert st.parse_step_series("", max_dom=2) == {1: [], 2: []}


# ---------------------------------------------------------------------------
# 2. radiation-step separation (STEPRA)
# ---------------------------------------------------------------------------
def test_steps_per_radiation_call_matches_sc26p1_worked_example():
    # campaign/cases.json sc26p1 note: "radt=9 at dt=10 => STEPRA=54"
    assert st.steps_per_radiation_call(radt_minutes=9, dt_seconds=10) == 54


def test_steps_per_radiation_call_rounds_and_floors_at_one():
    assert st.steps_per_radiation_call(radt_minutes=1, dt_seconds=100) == 1
    assert st.steps_per_radiation_call(radt_minutes=15, dt_seconds=20) == round(15 * 60 / 20)


def test_classify_steps_step_one_and_every_stepra_is_radiation():
    flags = st.classify_steps(n_steps=10, stepra=3)
    # itimestep 1..10, radiation at 1,3,6,9
    assert flags == [True, False, True, False, False, True, False, False, True, False]


def test_classify_steps_empty():
    assert st.classify_steps(0, 3) == []


def test_step_target_features_splits_normal_and_radiation_per_bucket():
    # dt=60s, stepra=3 (radt=3min) -> radiation at itimestep 1,3,6,9,... ; interval=1 minute (60s)
    # steps 1..6, each bucket holds exactly one step (60s per bucket == dt)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = st.step_target_features(values, dt_seconds=60, radt_minutes=3, interval_minutes=1)
    assert out[0] == dict(s_step_normal_median=None, s_step_rad_median=1.0, n_steps=1)  # step 1: rad
    assert out[1] == dict(s_step_normal_median=2.0, s_step_rad_median=None, n_steps=1)   # step 2
    assert out[2] == dict(s_step_normal_median=None, s_step_rad_median=3.0, n_steps=1)   # step 3: rad (3%3==0)


def test_step_target_features_bucketing_groups_multiple_steps_per_bucket():
    # dt=30s, interval=1min(60s) -> 2 steps per bucket. stepra huge -> nothing is radiation except step 1.
    values = [1.0, 2.0, 3.0, 4.0]
    out = st.step_target_features(values, dt_seconds=30, radt_minutes=1000, interval_minutes=1)
    assert out[0]["n_steps"] == 2
    assert out[0]["s_step_rad_median"] == 1.0     # step 1 only
    assert out[0]["s_step_normal_median"] == 2.0  # step 2 only
    assert out[1]["n_steps"] == 2
    assert out[1]["s_step_rad_median"] is None
    assert out[1]["s_step_normal_median"] == 3.5  # median(3.0, 4.0)


def test_step_target_features_empty_series():
    assert st.step_target_features([], dt_seconds=20, radt_minutes=9, interval_minutes=30) == {}


# ---------------------------------------------------------------------------
# 3. namelist array readers (radt, parent_time_step_ratio, dt-by-domain)
# ---------------------------------------------------------------------------
def test_read_namelist_int_list_single_value_pads_to_max_dom():
    assert st.read_namelist_int_list(NML_1DOM, "radt", max_dom=1) == [9]


def test_read_namelist_int_list_multi_value():
    assert st.read_namelist_int_list(NML_2DOM, "radt", max_dom=2) == [9, 3]


def test_read_namelist_int_list_missing_key_with_default():
    assert st.read_namelist_int_list(NML_1DOM, "parent_time_step_ratio", max_dom=1, default=1) == [1]


def test_read_namelist_int_list_missing_key_no_default_raises():
    try:
        st.read_namelist_int_list(NML_1DOM, "no_such_key", max_dom=1, default=None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_read_namelist_int_list_fewer_values_than_max_dom_repeats_last():
    text = "&physics\n radt = 9,\n/\n"
    assert st.read_namelist_int_list(text, "radt", max_dom=3) == [9, 9, 9]


def test_domain_dt_seconds_nested_ratio():
    # d01 dt=20s, d02 parent_time_step_ratio=4 -> d02 dt=5s
    out = st.domain_dt_seconds(20, [1, 4])
    assert out == [20.0, 5.0]


def test_domain_dt_seconds_single_domain():
    assert st.domain_dt_seconds(30, [1]) == [30.0]


# ---------------------------------------------------------------------------
# 4. auxiliary-stream detection / K-vs-dt validation / iofields content
# ---------------------------------------------------------------------------
def test_pick_free_aux_stream_defaults_to_23_when_unused():
    assert st.pick_free_aux_stream(NML_1DOM) == 23


def test_pick_free_aux_stream_falls_back_when_23_used():
    text = NML_1DOM.replace("&domains", "&time_control\n auxhist23_interval = 10,\n/\n&domains")
    assert st.pick_free_aux_stream(text) == 1


def test_pick_free_aux_stream_skips_every_used_stream():
    used = "\n".join(" auxhist%d_interval = 5," % n for n in range(1, 24))
    text = NML_1DOM.replace("&domains", "&time_control\n%s\n/\n&domains" % used)
    assert st.pick_free_aux_stream(text) == 24


def test_pick_free_aux_stream_raises_when_all_used():
    used = "\n".join(" io_form_auxhist%d = 2," % n for n in range(1, 25))
    text = NML_1DOM.replace("&domains", "&time_control\n%s\n/\n&domains" % used)
    try:
        st.pick_free_aux_stream(text)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_used_aux_streams_detects_interval_and_ioform():
    text = "auxhist7_interval = 30,\nio_form_auxhist12 = 2,\n! auxhist9_interval = 1, (comment)\n"
    assert st.used_aux_streams(text) == {7, 12}


def test_validate_aux_interval_minutes_passes_at_exactly_20x():
    st.validate_aux_interval_minutes(interval_minutes=(20 * 10) / 60.0, dt_seconds=10)  # exactly 20x


def test_validate_aux_interval_minutes_rejects_below_20x():
    try:
        st.validate_aux_interval_minutes(interval_minutes=1, dt_seconds=10)  # 60s < 20*10=200s
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_iofields_content_format():
    out = st.build_iofields_content(["RAINNC", "RAINC", "SWDOWN"], stream=23)
    assert out == "+:h:23:RAINNC,RAINC,SWDOWN\n"


# ---------------------------------------------------------------------------
# 5. Registry field validation
# ---------------------------------------------------------------------------
REGISTRY_TEXT = """
# comment line, not a state entry
state    real   RAINNC   ij    misc1   1  -  irh0230u  "RAINNC" "ACCUMULATED PRECIP" "mm"
state    real   RAINC    ij    misc1   1  -  irh0230u  "RAINC" "ACCUMULATED CU PRECIP" "mm"
state    real   SWDOWN   ij    misc1   1  -  irh0230u  "SWDOWN" "DOWNWARD SW" "W m-2"
rconfig  integer  radt   namelist,physics  max_domains  10  irh  "radt" ""
"""


def test_filter_fields_by_registry_keeps_present_drops_absent():
    kept, dropped = st.filter_fields_by_registry(["RAINNC", "COSZEN", "SWDOWN"], REGISTRY_TEXT)
    assert kept == ["RAINNC", "SWDOWN"]
    assert dropped == ["COSZEN"]


def test_filter_fields_by_registry_does_not_match_substring():
    # RAINNC must not match a hypothetical 'RAINNCX' state line partially
    text = "state real RAINNCX ij misc1 1 - x \"RAINNCX\" \"\" \"\"\n"
    kept, dropped = st.filter_fields_by_registry(["RAINNC"], text)
    assert kept == []
    assert dropped == ["RAINNC"]


def test_read_registry_dir_missing_returns_none(tmp_path):
    assert st.read_registry_dir(str(tmp_path / "no-such-wrf-tree")) is None


def test_read_registry_dir_concatenates_fragments(tmp_path):
    reg = tmp_path / "Registry"
    reg.mkdir()
    (reg / "Registry.EM_COMMON").write_text("state real RAINNC ij misc1 1 - x \"RAINNC\" \"\" \"\"\n")
    (reg / "Registry.EM_OTHER").write_text("state real SWDOWN ij misc1 1 - x \"SWDOWN\" \"\" \"\"\n")
    text = st.read_registry_dir(str(tmp_path))
    assert "RAINNC" in text and "SWDOWN" in text


# ---------------------------------------------------------------------------
# 6. edit_namelist_timeline
# ---------------------------------------------------------------------------
def test_edit_namelist_timeline_single_domain():
    out = st.edit_namelist_timeline(
        NML_1DOM, nproc_x=9, nproc_y=8, numtiles=1, ranks=72, max_dom=1, run_minutes=720,
        restart_every_minutes=180, aux_stream=23, aux_interval_minutes=30,
        iofields_filename="iofields_timeline.txt", frames_per_auxhist=500)
    assert "restart = .false." in out
    assert "auxhist23_interval = 30," in out
    assert "io_form_auxhist23=2," in out
    assert "frames_per_auxhist23 = 500," in out
    assert 'iofields_filename = "iofields_timeline.txt",' in out
    assert "&namelist_adios2" not in out   # edit_namelist_long's adios2=False carried through


def test_edit_namelist_timeline_multi_domain_zeroes_nest_aux_interval():
    out = st.edit_namelist_timeline(
        NML_2DOM, nproc_x=8, nproc_y=9, numtiles=1, ranks=72, max_dom=2, run_minutes=720,
        restart_every_minutes=180, aux_stream=5, aux_interval_minutes=30,
        iofields_filename="iof.txt", frames_per_auxhist=100)
    assert "auxhist5_interval = 30, 0," in out
    assert "frames_per_auxhist5 = 100, 100," in out
    assert 'iofields_filename = "iof.txt", "iof.txt",' in out


# ---------------------------------------------------------------------------
# 7. MPI patch mapping vs predictor/decomp_space.py ground truth
# ---------------------------------------------------------------------------
def test_patch_bounds_widths_sum_to_extent():
    bounds = st.patch_bounds(extent=97, nproc=7)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 97
    assert sum(hi - lo for lo, hi in bounds) == 97
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]   # contiguous, no gaps/overlaps


def test_patch_bounds_min_max_matches_decomp_space_rsl_patch_extents():
    for extent, nproc in [(901, 56), (901, 4), (150, 11), (33, 5)]:
        bounds = st.patch_bounds(extent, nproc)
        widths = [hi - lo for lo, hi in bounds]
        expected_min, expected_max = decomp_space.rsl_patch_extents(extent, nproc)
        assert min(widths) == expected_min
        assert max(widths) == expected_max


def test_patch_of_index_matches_patch_bounds():
    bounds = st.patch_bounds(extent=20, nproc=3)
    for idx in range(20):
        r = st.patch_of_index(bounds, idx)
        lo, hi = bounds[r]
        assert lo <= idx < hi


def test_patch_of_index_out_of_range_raises():
    bounds = st.patch_bounds(extent=10, nproc=2)
    try:
        st.patch_of_index(bounds, 10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_patch_aggregate_on_small_grid():
    # 4x4 grid, 2x2 ranks -> each patch is 2x2 cells. Value = column index (0..3), same every row.
    grid = [[0.0, 1.0, 2.0, 3.0] for _ in range(4)]
    agg = st.patch_aggregate(grid, we=4, sn=4, nproc_x=2, nproc_y=2)
    assert agg[(0, 0)] == 0.5   # mean(0,1)
    assert agg[(1, 0)] == 2.5   # mean(2,3)
    assert agg[(0, 1)] == 0.5
    assert agg[(1, 1)] == 2.5


def test_max_mean_over_patches():
    mx, mean = st.max_mean_over_patches({(0, 0): 1.0, (1, 0): 3.0})
    assert mx == 3.0
    assert mean == 2.0


def test_max_mean_over_patches_empty():
    assert st.max_mean_over_patches({}) == (None, None)


def test_imbalance_ratio():
    assert st.imbalance_ratio(4.0, 2.0) == 2.0
    assert st.imbalance_ratio(None, 2.0) is None
    assert st.imbalance_ratio(4.0, 0) is None


# ---------------------------------------------------------------------------
# 8. state feature math (numpy-free)
# ---------------------------------------------------------------------------
def test_precip_rate_mm_per_h():
    assert st.precip_rate_mm_per_h(0.0, 0.0, 2.0, 1.0, interval_hours=0.5) == 6.0  # 3mm/0.5h


def test_precip_rate_mm_per_h_clips_negative_to_zero():
    assert st.precip_rate_mm_per_h(5.0, 0.0, 4.0, 0.0, interval_hours=1.0) == 0.0


def test_precip_frac_and_mean():
    frac, mean = st.precip_frac_and_mean([0.0, 2.0, 5.0, 0.5])
    assert frac == 0.5     # 2 of 4 columns > 1.0 mm/h
    assert mean == 1.875


def test_precip_frac_and_mean_empty():
    assert st.precip_frac_and_mean([]) == (None, None)


def test_precip_indicator():
    assert st.precip_indicator([0.0, 2.0, 1.0, 1.5]) == [0.0, 1.0, 0.0, 1.0]


def test_cloud_frac():
    assert st.cloud_frac([100.0, 250.0, 199.9, 300.0]) == 0.5


def test_cloud_frac_empty():
    assert st.cloud_frac([]) is None


def test_sunlit_frac_swdown_only():
    assert st.sunlit_frac([0.0, 5.0, 0.5, 100.0]) == 0.5


def test_sunlit_frac_with_coszen_fallback():
    swdown = [0.0, 0.0]
    coszen = [0.1, -0.1]
    assert st.sunlit_frac(swdown, coszen) == 0.5   # first column sunlit via coszen>0


def test_sunlit_frac_empty():
    assert st.sunlit_frac([]) is None


def test_sunlit_indicator_matches_sunlit_frac_rule():
    swdown = [0.0, 5.0]
    ind = st.sunlit_indicator(swdown)
    assert ind == [0.0, 1.0]


def test_pblh_stats():
    mean, p90 = st.pblh_stats([100.0, 200.0, 300.0, 400.0, 500.0])
    assert mean == 300.0
    assert p90 == 500.0   # ceil(0.9*5)-1 = 4 -> last (largest) element


def test_pblh_stats_empty():
    assert st.pblh_stats([]) == (None, None)


def test_wind10_max():
    assert st.wind10_max([3.0, 0.0], [4.0, 1.0]) == 5.0


def test_wind10_max_empty():
    assert st.wind10_max([], []) is None


def test_mean_or_none():
    assert st.mean_or_none([1.0, 2.0, 3.0]) == 2.0
    assert st.mean_or_none([]) is None


# ---------------------------------------------------------------------------
# 9. row assembly (build_state_row / merge_interval_rows) -- pure integration, no I/O
# ---------------------------------------------------------------------------
def test_build_state_row_without_decomposition_skips_imbalance():
    row = st.build_state_row(
        precip_rates=[0.0, 2.0], olr=[150.0, 250.0], swdown=[0.0, 10.0], coszen=None,
        pblh=[100.0, 200.0], u10=[1.0], v10=[1.0], hfx=[5.0, 15.0],
        we=2, sn=1, nproc_x=None, nproc_y=None)
    assert row["precip_frac"] == 0.5
    assert row["precip_frac_patch_max"] is None
    assert row["sunlit_frac_patch_mean"] is None


def test_build_state_row_with_decomposition_computes_imbalance():
    # 4x2 grid (we=4, sn=2), 2x1 patches: left half all dry, right half all wet.
    precip_rates = [0.0, 0.0, 5.0, 5.0,   0.0, 0.0, 5.0, 5.0]
    row = st.build_state_row(
        precip_rates=precip_rates, olr=[], swdown=[], coszen=None,
        pblh=[], u10=[], v10=[], hfx=[],
        we=4, sn=2, nproc_x=2, nproc_y=1)
    assert row["precip_frac_patch_mean"] == 0.5   # (0 + 1) / 2
    assert row["precip_frac_patch_max"] == 1.0
    assert row["precip_frac_imbalance"] == 2.0


def test_merge_interval_rows_missing_state_or_target_is_none_not_dropped():
    state_by_bucket = {0: {c: 1.0 for c in st.STATE_COLUMNS}}
    target_by_domain_bucket = {1: {0: dict(s_step_normal_median=0.5, s_step_rad_median=None, n_steps=10),
                                    1: dict(s_step_normal_median=0.6, s_step_rad_median=None, n_steps=10)}}
    rows = st.merge_interval_rows(state_by_bucket, target_by_domain_bucket, max_dom=1,
                                   interval_minutes=30)
    assert [r["interval_index"] for r in rows] == [0, 1]
    assert rows[0]["t_sim_hours"] == 0.0
    assert rows[1]["t_sim_hours"] == 0.5
    assert rows[1]["precip_frac"] is None          # no state for bucket 1
    assert rows[0]["s_step_normal_median_d01"] == 0.5


def test_timeline_csv_fieldnames_includes_per_domain_targets():
    fields = st.timeline_csv_fieldnames(max_dom=2)
    assert "s_step_normal_median_d01" in fields
    assert "s_step_rad_median_d02" in fields
    assert "n_steps_d02" in fields


def test_write_timeline_csv_round_trip(tmp_path):
    rows = st.merge_interval_rows(
        {0: {c: 1.0 for c in st.STATE_COLUMNS}}, {1: {0: dict(
            s_step_normal_median=0.5, s_step_rad_median=1.5, n_steps=30)}}, max_dom=1,
        interval_minutes=30)
    path = str(tmp_path / "case__cfg__timeline.csv")
    st.write_timeline_csv(path, rows, max_dom=1)
    with open(path) as f:
        import csv as csv_mod
        out_rows = list(csv_mod.DictReader(f))
    assert out_rows[0]["s_step_normal_median_d01"] == "0.5"
    assert out_rows[0]["precip_frac"] == "1.0"


def test_write_timeline_header_json_round_trip(tmp_path):
    path = str(tmp_path / "hdr.json")
    st.write_timeline_header_json(path, dict(case="synth", dt=20))
    with open(path) as f:
        assert json.load(f) == dict(case="synth", dt=20)


# ---------------------------------------------------------------------------
# 10. CLI plumbing (aux-stream pick / iofields check+write / namelist-timeline edit)
# ---------------------------------------------------------------------------
def test_cmd_pick_aux_stream(tmp_path, capsys):
    p = tmp_path / "namelist.input"
    p.write_text(NML_1DOM)
    args = argparse.Namespace(namelist_file=str(p), preferred=23)
    st.cmd_pick_aux_stream(args)
    assert capsys.readouterr().out.strip() == "23"


def test_cmd_check_iofields_no_registry_passes_through(tmp_path, capsys):
    args = argparse.Namespace(wrf_tree=str(tmp_path / "nope"), fields="RAINNC,COSZEN")
    st.cmd_check_iofields(args)
    assert capsys.readouterr().out.strip() == "RAINNC,COSZEN"


def test_cmd_check_iofields_drops_missing(tmp_path, capsys):
    reg = tmp_path / "Registry"
    reg.mkdir()
    (reg / "Registry.EM_COMMON").write_text(REGISTRY_TEXT)
    args = argparse.Namespace(wrf_tree=str(tmp_path), fields="RAINNC,COSZEN,SWDOWN")
    st.cmd_check_iofields(args)
    out = capsys.readouterr()
    assert out.out.strip() == "RAINNC,SWDOWN"
    assert "DROPPED COSZEN" in out.err


def test_cmd_write_iofields(tmp_path):
    outfile = tmp_path / "iof.txt"
    args = argparse.Namespace(outfile=str(outfile), stream=23, fields="RAINNC,SWDOWN")
    st.cmd_write_iofields(args)
    assert outfile.read_text() == "+:h:23:RAINNC,SWDOWN\n"


def test_cmd_edit_namelist_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(st.tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    infile = tmp_path / "in.input"
    outfile = tmp_path / "out.input"
    infile.write_text(NML_1DOM)
    args = argparse.Namespace(
        infile=str(infile), outfile=str(outfile), case="synth", hours=12, every=3,
        nproc_x=9, nproc_y=8, numtiles=1, ranks=72, max_dom=1, aux_stream=23, aux_interval=30,
        frames_per_auxhist=500, iofields_filename="iof.txt", dt=None)
    st.cmd_edit_namelist_timeline(args)
    out = outfile.read_text()
    assert "auxhist23_interval = 30," in out
    assert "run_minutes = 720" in out


def test_cmd_edit_namelist_timeline_raises_interval_below_20x_dt_to_the_floor(tmp_path, monkeypatch):
    # Behaviour changed after scc job 464: the CLI raises a too-short interval to the 20xdt floor
    # instead of aborting (the pure validate_aux_interval_minutes still raises, tested above).
    monkeypatch.setattr(st.tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    infile = tmp_path / "in.input"
    infile.write_text(NML_1DOM)
    outfile = tmp_path / "out.input"
    args = argparse.Namespace(
        infile=str(infile), outfile=str(outfile), case="synth", hours=1, every=1,
        nproc_x=1, nproc_y=1, numtiles=1, ranks=1, max_dom=1, aux_stream=23, aux_interval=1,
        frames_per_auxhist=10, iofields_filename="iof.txt", dt=None)
    # MATTHEW_LIKE dt=20s -> floor is ceil(20*20/60) = 7 min
    st.cmd_edit_namelist_timeline(args)
    assert args.aux_interval == 7
    assert "auxhist23_interval = 7," in outfile.read_text()


def test_cmd_default_fields(capsys):
    st.cmd_default_fields(argparse.Namespace())
    assert capsys.readouterr().out.strip() == ",".join(st.DEFAULT_TIMELINE_FIELDS)


# ---------------------------------------------------------------------------
# 11. plan-training CLI (ledger-driven estimate, "unknown" fallback)
# ---------------------------------------------------------------------------
def test_cmd_plan_training_unknown_when_no_ledger(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(st.tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    monkeypatch.setattr(st.tv, "auto_select_cases", lambda min_hours=0: ["synth"])
    args = argparse.Namespace(cases=["synth"], hours=6, nodes=1, k=2, hardware_id="scc-gh200",
                               ledger_dir=str(tmp_path / "no-ledger-dir"))
    st.cmd_plan_training(args)
    out = capsys.readouterr().out
    assert "s/step unknown" in out
    assert "submit-timeline --case synth" in out
