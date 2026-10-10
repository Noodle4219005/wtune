"""coldstart_sample I/O probe: window-namelist edits, write-line parsing, scoring exclusion of the
step that absorbed a write. Offline, synthetic namelists and rsl files."""
import datetime
import json
import os

from wtune.tools import coldstart_sample as cs
from wtune import predict as predict

FMT = cs._WRF_TIME_FMT

CASE_NL = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 24,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2019, 2019,
 start_month                         = 11, 11,
 start_day                           = 26, 26,
 start_hour                          = 00, 00,
 end_year                            = 2019, 2019,
 end_month                           = 11, 11,
 end_day                             = 27, 27,
 end_hour                            = 00, 00,
 interval_seconds                    = 10800
 history_interval_m                  = 60, 0,
 history_begin_h                     = 1, 1,
 frames_per_outfile                  = 1, 1,
 restart                             = .false.,
 restart_interval                    = 720,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 iofields_filename                   = 'io_d01.txt', 'io_d02.txt',
 /

 &domains
 time_step                           = 60,
 max_dom                             = 2,
 e_we                                = 100, 100,
 e_sn                                = 100, 100,
 e_vert                              = 40, 40,
 /

 &physics
 radt                                = 15, 15,
 /
"""

T = datetime.datetime(2019, 11, 26, 3)


def _nl(tmp_path, text=CASE_NL):
    p = tmp_path / "namelist.input"
    p.write_text(text)
    return predict.parse_namelist(str(p))


def _window(probe, text=CASE_NL, win_steps=40, dt_s=60.0):
    return cs.build_wrf_window_namelist_text(text, T, win_steps, dt_s, 4, 4, 1, io_probe=probe)


def _vals(txt, key):
    import re
    m = re.search(r"(?<![A-Za-z0-9_])%s\s*=\s*([^\n]*)" % key, txt, re.I)
    return [v.strip() for v in m.group(1).split(",") if v.strip()] if m else None


def test_default_off_window_namelist_is_byte_identical():
    base = cs.build_wrf_window_namelist_text(CASE_NL, T, 40, 60.0, 4, 4, 1)
    assert _window(None) == base
    assert _vals(base, "history_interval_m") == ["0", "0"]
    assert _vals(base, "restart_interval") == ["999999"]


def test_plan_io_probe_reads_case_output_settings(tmp_path):
    (tmp_path / "io_d01.txt").write_text("-:h:0:RAINC\n")
    probe = cs.plan_io_probe(_nl(tmp_path), str(tmp_path), 86400)
    assert probe["history"] == [True, False]          # d02 history_interval_m = 0 -> off
    assert probe["restart"] is True                   # 720 min <= 24 h
    assert probe["history_prefix"] == "wrfout_d"
    assert probe["iofields"][0]["path"] == str(tmp_path / "io_d01.txt")
    assert probe["iofields"][1] is None               # missing -> not used, warned
    assert len(probe["warnings"]) == 1 and "io_d02.txt" in probe["warnings"][0]
    assert probe["notes"] == []


def test_restart_probe_only_when_the_case_writes_restarts(tmp_path):
    assert cs.plan_io_probe(_nl(tmp_path, CASE_NL.replace("= 720,", "= 2000,")), str(tmp_path),
                            86400)["restart"] is False
    assert cs.plan_io_probe(_nl(tmp_path, CASE_NL.replace("= 720,", "= 0,")), str(tmp_path),
                            86400)["restart"] is False


def test_auxhist_is_noted_not_probed(tmp_path):
    text = CASE_NL.replace(" io_form_restart", " io_form_auxhist2 = 2,\n io_form_restart")
    assert cs.plan_io_probe(_nl(tmp_path, text), str(tmp_path), 86400)["notes"] == [
        "auxhist not probed (io_form_auxhist2 set)"]


def test_probe_window_writes_one_end_frame_per_history_domain():
    probe = {"history": [True, False], "restart": True,
             "iofields": [{"name": "io_d01.txt", "path": "/c/io_d01.txt"}, None]}
    txt = _window(probe)                                  # W = 40 x 60 s = 40 min
    assert _vals(txt, "history_interval") == ["40", "0"]
    assert _vals(txt, "history_begin") == ["40", "0"]     # begin > 0: no t=0 frame
    assert _vals(txt, "history_interval_m") == ["0", "0"]  # WRF sums _m (else the plain key)
    assert _vals(txt, "history_begin_h") == ["0", "0"]     # ...and begin_h
    assert _vals(txt, "restart_interval") == ["40"]
    assert _vals(txt, "iofields_filename") == ["'io_d01.txt'", "'NONE_SPECIFIED'"]
    for k in ("history_interval", "history_begin", "restart_interval", "iofields_filename"):
        assert cs.count_key(txt, k) == 1


def test_probe_window_uses_seconds_keys_when_window_is_not_whole_minutes():
    txt = _window({"history": [True, True], "restart": False, "iofields": None}, win_steps=41,
                  dt_s=45.0)                              # 1845 s
    assert _vals(txt, "history_interval_s") == ["1845", "1845"]
    assert _vals(txt, "history_begin_s") == ["1845", "1845"]
    assert _vals(txt, "history_begin_h") == ["0", "0"]
    assert _vals(txt, "restart_interval") == ["999999"]   # restart not probed: stays off
    assert _vals(txt, "iofields_filename") == ["'io_d01.txt'", "'io_d02.txt'"]  # untouched


def test_fix_wrf_namelist_applies_probe_only_to_the_probe_window(tmp_path):
    plan = {"windows": [{"t": T.strftime(FMT)}, {"t": T.strftime(FMT)}], "measure_steps": 20,
            "spinup_min": 20, "dt_s": 60.0,
            "io_probe": {"window": 1, "history": [True, False], "restart": False, "iofields": None}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out = {}
    for k in (0, 1):
        os.makedirs(tmp_path / ("win_%d" % k) / "real")
        os.makedirs(tmp_path / ("win_%d" % k) / "wrf")
        (tmp_path / ("win_%d" % k) / "real" / "namelist.input").write_text(CASE_NL)
        cs.cmd_fix_wrf_namelist(str(tmp_path), k, 4, 4, 1)
        out[k] = (tmp_path / ("win_%d" % k) / "wrf" / "namelist.input").read_text()
    assert out[0] == cs.build_wrf_window_namelist_text(CASE_NL, T, 40, 60.0, 4, 4, 1)
    assert _vals(out[1], "history_begin") == ["40", "0"]


def _write(line_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(line_list) + "\n")


def _main(t, dom, sec):
    return "Timing for main: time %s on domain   %d:   %.5f elapsed seconds" % (
        t.strftime(FMT), dom, sec)


def test_parse_write_lines_history_per_domain_and_restart(tmp_path):
    p = str(tmp_path / "rsl.error.0000")
    _write([_main(T, 1, 1.0),
            "Timing for Writing wrfout_d01_2019-11-26_18:00:00 for domain        1:   13.90594 elapsed seconds",
            "Timing for Writing wrfout_d02_2019-11-26_18:00:00 for domain        2:    4.50000 elapsed seconds",
            "Timing for Writing auxhist2_d01_2019-11-26_18:00:00 for domain        1:    2.00000 elapsed seconds",
            "Timing for Writing restart for domain        1:  298.77524 elapsed seconds",
            "Timing for Writing restart for domain        2:   50.00000 elapsed seconds"], p)
    w = cs.parse_rsl_write_seconds(p, "wrfout_d")
    assert w["history"] == {1: [13.90594], 2: [4.5]}
    assert w["restart"] == {1: [298.77524], 2: [50.0]}
    assert w["other"] == [("auxhist2_d01_2019-11-26_18:00:00", 1, 2.0)]


def test_score_excludes_the_step_that_absorbed_a_write(tmp_path):
    dt_s, spin = 60.0, 600.0
    lines = [_main(T + datetime.timedelta(seconds=dt_s * i), 1, 50.0) for i in range(1, 10)]
    t_abs = T + datetime.timedelta(seconds=spin + dt_s * 6)
    for i in range(1, 21):
        t = T + datetime.timedelta(seconds=spin + dt_s * i)
        if t == t_abs:
            lines.append("Timing for Writing wrfout_d02_x for domain        2:   30.00000 elapsed seconds")
            lines.append(_main(t, 2, 0.5))                 # a nest line between: still d01's step
        lines.append(_main(t, 1, 31.0 if t == t_abs else 1.0 + 0.001 * i))
    _write(lines, str(tmp_path / "win_0" / "wrf" / "rsl.error.0000"))
    assert cs.rsl_write_absorbing_d01_times(str(tmp_path / "win_0" / "wrf" / "rsl.error.0000")) \
        == {t_abs.strftime(FMT)}
    plan = {"dt_s": dt_s, "radt_min": 100000.0,
            "stepra": predict.steps_per_radiation_call(100000.0, dt_s),
            "spinup_min": spin / 60.0, "W_seconds": spin + 20 * dt_s,
            "run_start": T.strftime(FMT), "windows": [{"t": T.strftime(FMT), "reason": "t0"}],
            "_plan_dir": str(tmp_path)}
    out, _pooled, _anchors = cs.score_windows_from_rsl(plan, None, {})
    # 20 kept steps, 1 radiation-flagged (i==1), 1 absorbed a write -> 18 normal samples
    assert out[0]["n_normal"] == 18
    assert out[0]["s_k"] < 1.1


def test_stage_plan_links_iofields_and_extends_timeout(tmp_path, monkeypatch):
    monkeypatch.delenv("WPS_DIR", raising=False)
    monkeypatch.delenv("WRF_TREE", raising=False)
    seen = {}
    monkeypatch.setattr(cs, "render_run_sbatch",
                        lambda plan, site, per_window_forcing, per_window_estimate_min:
                        seen.setdefault("min", per_window_estimate_min) and "#!/bin/bash\n")
    monkeypatch.setattr(cs, "_per_window_forcing_from_forcing_json", lambda d, p: None)
    src = tmp_path / "case_io.txt"
    src.write_text("+:h:0:RAINC\n")
    pdir = tmp_path / "plan"
    pdir.mkdir()
    plan = {"windows": [{"t": T.strftime(FMT)}],
            "io_probe": {"window": 0, "iofields": [{"name": "case_io.txt", "path": str(src)}]}}
    (pdir / "plan.json").write_text(json.dumps(plan))
    cs.stage_plan(str(pdir), per_window_min=30)
    assert os.path.realpath(pdir / "win_0" / "wrf" / "case_io.txt") == str(src)
    assert seen["min"] == 30 + cs.IO_PROBE_TIMEOUT_EXTRA_MIN



_ELEV_LIKE = """&time_control
 run_hours = 15,
 start_year = 2022, 2022,
 start_month = 05, 05,
 start_day = 12, 12,
 start_hour = 00, 00,
 end_year = 2022, 2022,
 end_month = 05, 05,
 end_day = 12, 12,
 end_hour = 15, 15,
 history_interval = 60, 15,
 restart = .false.,
 restart_interval = 1440,
 iofields_filename = 'storm_diags_d01.txt', 'storm_diags_d01.txt',
/
 iofields_filename = 'stray.txt',
&domains
 time_step = 15,
 max_dom = 2,
 e_we = 400, 300,
 e_sn = 400, 300,
/
&physics
 nwp_diagnostics = 1,
/
"""


def test_out_of_group_assignment_is_commented_not_counted():
    from wtune.tools import coldstart_sample as cs
    t = cs.comment_out_of_group_assignments(_ELEV_LIKE)
    assert cs.count_key(t, "iofields_filename") == 1
    assert "! (outside any namelist group; ignored by Fortran)  iofields_filename = 'stray.txt'" in t
    assert cs.count_key(t, "nwp_diagnostics") == 1          # inside &physics: untouched


def test_nwp_diagnostics_window_keeps_history_interval_nonzero_and_no_frame():
    import datetime
    from wtune.tools import coldstart_sample as cs
    txt = cs.build_wrf_window_namelist_text(_ELEV_LIKE, datetime.datetime(2022, 5, 12, 14), 100,
                                            15.0, 16, 18, 1)
    assert "history_interval = 999999, 999999" in txt.replace("  ", " ") or \
        cs.get_first(txt, "history_interval") == "999999"
    assert cs.get_first(txt, "history_begin") == "999999"
    plain = _ELEV_LIKE.replace(" nwp_diagnostics = 1,", " nwp_diagnostics = 0,")
    txt0 = cs.build_wrf_window_namelist_text(plain, datetime.datetime(2022, 5, 12, 14), 100,
                                             15.0, 16, 18, 1)
    assert cs.get_first(txt0, "history_interval") == "0" and cs.count_key(txt0, "history_begin") == 0


def test_run_sbatch_stops_wrf_after_fatal():
    import subprocess, tempfile, os
    from wtune.tools import coldstart_sample as cs
    plan = {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}], "nodes": 1,
            "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}}
    text = cs.render_run_sbatch(plan, site="scc")
    assert "grep -lqs 'FATAL CALLED' rsl.error.*" in text and "FATAL_WATCH_PID" in text
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(text)
    try:
        assert subprocess.run(["bash", "-n", f.name]).returncode == 0
    finally:
        os.unlink(f.name)


def test_every_window_links_the_case_iofields_file(tmp_path):
    from wtune.tools import coldstart_sample as cs
    case = tmp_path / "case"; case.mkdir()
    (case / "namelist.input").write_text(_ELEV_LIKE)
    (case / "storm_diags_d01.txt").write_text("")
    plan = {"case_dir": str(case), "namelist_input": str(case / "namelist.input"),
            "windows": [{"t": "a"}, {"t": "b"}]}
    out = cs.case_iofields_links(plan, str(tmp_path / "plan"))
    assert sorted(k for k, _ in out) == [0, 1]
    for k in (0, 1):
        assert (tmp_path / "plan" / ("win_%d" % k) / "wrf" / "storm_diags_d01.txt").is_symlink()


def test_run_sbatch_stops_a_silent_init_hang():
    from wtune.tools import coldstart_sample as cs
    plan = {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}], "nodes": 1,
            "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}}
    text = cs.render_run_sbatch(plan, site="scc")
    assert "grep -qs 'Problem opening' rsl.error.0000" in text and "po -ge 120" in text


def test_nwp_diagnostics_probe_window_uses_plain_minutes():
    import datetime
    from wtune.tools import coldstart_sample as cs
    probe = {"window": 0, "history": [True, True], "restart": False, "iofields": None}
    # 1200 s window (whole minutes): plain keys, both = 20 min, nothing huge left
    txt = cs.build_wrf_window_namelist_text(_ELEV_LIKE, datetime.datetime(2022, 5, 12, 14), 80,
                                            15.0, 16, 18, 1, io_probe=probe)
    assert cs.get_first(txt, "history_interval") == "20" and cs.get_first(txt, "history_begin") == "20"
    assert "999999" not in txt.split("restart_interval")[0].split("history")[-1] or \
        cs.get_first(txt, "history_interval") != "999999"
    # a 27.5-min window is refused rather than silently writing nothing
    import pytest
    with pytest.raises(cs.ColdstartError):
        cs.build_wrf_window_namelist_text(_ELEV_LIKE, datetime.datetime(2022, 5, 12, 14), 110,
                                          15.0, 16, 18, 1, io_probe=probe)


def test_probe_with_seconds_zeroes_plain_keys():
    import datetime
    from wtune.tools import coldstart_sample as cs
    plain = _ELEV_LIKE.replace(" nwp_diagnostics = 1,", " nwp_diagnostics = 0,")
    probe = {"window": 0, "history": [True, True], "restart": False, "iofields": None}
    txt = cs.build_wrf_window_namelist_text(plain, datetime.datetime(2022, 5, 12, 14), 110,
                                            15.0, 16, 18, 1, io_probe=probe)
    assert cs.get_first(txt, "history_interval_s") == "1650"
    assert cs.get_first(txt, "history_interval") == "0"
