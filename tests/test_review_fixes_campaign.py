"""Regression tests for the 8bdcd3c source-review fixes in campaign/strict_campaign.sbatch,
campaign/nc2bp.sbatch, campaign/node_hygiene.sh, tools/coldstart_sample.py, tools/case_rebuild.py.
Offline: bash functions are cut out of the scripts and run with fakes; nothing is submitted."""
import datetime
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from wtune.tools import case_rebuild as cr  # noqa: E402
from wtune.tools import coldstart_sample as cs  # noqa: E402
import test_coldstart_sample as tcs  # noqa: E402

SBATCH = os.path.join(REPO, "campaign", "strict_campaign.sbatch")
NC2BP = os.path.join(REPO, "campaign", "nc2bp.sbatch")
HYGIENE = os.path.join(REPO, "campaign", "node_hygiene.sh")


def _text(path=SBATCH):
    with open(path) as fh:
        return fh.read()


def _bash_fn(name, text=None):
    """One top-level `name () { ... }` / `name() { ... }` function, verbatim."""
    m = re.search(r"^(%s ?\(\) \{.*?\n\}\n)" % re.escape(name), text or _text(), re.S | re.M)
    assert m, name
    return m.group(1)


def _bash(script, **kw):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------
# strict_campaign.sbatch
# ---------------------------------------------------------------------------
def test_bash_syntax():
    for f in (SBATCH, NC2BP, HYGIENE):
        subprocess.run(["bash", "-n", f], check=True)


def test_spinup_edit_gets_the_candidate_namelist_args():
    text = _text()
    spin = text[text.index("--write-restart-interval") - 500:text.index("--write-restart-interval")]
    spin = spin[spin.rindex("edit-namelist"):]
    assert '"${A_SET_ARGS[@]}" "${A_DELETE_ARGS[@]}" "${A_NEST_ARGS[@]}" "${A_TILE_ARGS[@]}"' in spin
    # built before the spin-up loop, from the same sources the candidate loop reads
    setup = text[text.index("A_NTASKS_PER_NODE="):text.index('for M in "${TARGET_MIN_ARR[@]}"; do')]
    assert 'for kv in ${NAMELIST_OVERRIDES:-}; do A_SET_ARGS+=(--set "$kv"); done' in setup
    assert 'for k in ${NAMELIST_DELETE_KEYS:-}; do A_DELETE_ARGS+=(--delete-key "$k"); done' in setup
    assert '--tile-strategy "${CFG_TSTRAT[ANCHOR]:-0}"' in setup
    # the rest of the spin-up edit is unchanged
    assert '--run-minutes "$M" --max-dom "$MAX_DOM" --history-mode off --ranks "$A_RANKS"' in spin


def test_restart_used_only_after_a_successful_restart_edit():
    text = _text()
    strict = text[text.index("RESTART_SKIP_REASON=\"spinup_failed\""):]
    block = strict[strict.index("--restart-from"):strict.index("link_namelist_files")]
    ok_branch, fail_branch = block.split("      else\n", 1)
    assert 'RESTART_EDIT_RC=$?' in ok_branch
    assert 'if [ "$RESTART_EDIT_RC" = "0" ]; then' in ok_branch
    assert "RESTART_USED=1" in ok_branch and "RESTART_USED=1" not in fail_branch
    assert "mv -f ./namelist.input.restart_orig ./namelist.input" in fail_branch
    assert 'RESTART_SKIP_REASON="restart_edit_failed"' in fail_branch
    # the restart-only adios2.xml is written only for a run that really restarts
    assert "write_adios2_restart_xml . " in ok_branch
    # RANKING_STRICT_RESTART=1 then writes a SKIPPED row instead of launching
    assert 'write_restart_skip_row "${CASE}_${cid}_r${rep}_${J}" "$cid" "$rep" "$RESTART_SKIP_REASON"' \
        in block


def test_strict_loop_validates_the_restart_start_time():
    text = _text()
    strict = text[text.index('for cid in "${ORDER[@]}"; do'):]
    assert strict.count("RESTART_CHECK_FAILED") == 1
    check = strict[strict.index('if [ "$RESTART_USED" = "1" ] && [ "$FUSED_ON" != "1" ]'):]
    check = check[:check.index("# step_stats()")]
    assert "RESTART_USED=0" in check
    # behaviour: a first step at/before RESTART_TIME is a t0 start; a later one is a restart
    script = ('cid=G1 FUSED_ON=0 RESTART_TIME=2019-11-26_12:20:00\n'
              'for t in 2019-11-26_12:00:18 2019-11-26_12:20:18; do\n'
              '  RESTART_USED=1\n'
              '  echo "Timing for main: time $t on domain   1:    1.5 elapsed seconds" > rsl.out.0000\n'
              + check + '  echo "$t $RESTART_USED"\ndone\n')
    out = _bash("cd $(mktemp -d)\n" + script)
    assert out.stdout.split("\n")[:2] == ["2019-11-26_12:00:18 0", "2019-11-26_12:20:18 1"], out
    assert "RESTART_CHECK_FAILED cid=G1" in out.stderr


@pytest.mark.parametrize("job_wide,per_cfg,want", [
    ("0", "0", "0"), ("0", "", "0"), ("1", "0", "1"), ("2", "0", "1"),
    ("0", "1", "1"), ("0", "2", "1")])
def test_window_frame_any_accepts_level_2(job_wide, per_cfg, want):
    text = _text()
    block = text[text.index("WINDOW_FRAME_ANY=0\n"):text.index('if [ "$MODE" = "timing" ] && [ "$WINDOW_FRAME_ANY"')]
    script = ("declare -A CFG_WINDOW_FRAME=([G1]=%s)\nWINDOW_FRAME=%s\n" % (per_cfg or '""', job_wide)
              + block + 'echo "$WINDOW_FRAME_ANY"\n')
    assert _bash(script).stdout.strip() == want


def test_adios2_restart_xml_does_not_write_through_a_symlink(tmp_path):
    shared = tmp_path / "shared.xml"
    shared.write_text("ORIGINAL\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "adios2.xml").symlink_to(shared)
    script = ("RANKING_RESTART_IO_FORM=14\n" + _bash_fn("write_adios2_restart_xml")
              + 'write_adios2_restart_xml "%s" 2019-11-26_12:20:00 2\n' % run)
    assert _bash(script).returncode == 0
    assert shared.read_text() == "ORIGINAL\n"
    assert not (run / "adios2.xml").is_symlink()
    assert 'wrfrst_d02_2019-11-26_12:20:00' in (run / "adios2.xml").read_text()


# ---------------------------------------------------------------------------
# node hygiene: fail-closed
# ---------------------------------------------------------------------------
def test_node_hygiene_prints_one_ok_line_per_invocation():
    for mode in ("census", "reap+census"):
        out = subprocess.run(["bash", HYGIENE, mode], capture_output=True, text=True, check=True,
                             env=dict(os.environ, USER="wtune-no-such-user")).stdout
        ok = [l for l in out.splitlines() if l.startswith("HYGIENE_OK ")]
        assert len(ok) == 1 and ok[0].endswith(" mode=%s" % mode), out
        assert out.splitlines()[-1] == ok[0]
    bad = subprocess.run(["bash", HYGIENE, "bogus"], capture_output=True, text=True)
    assert bad.returncode == 1 and "HYGIENE_OK" not in bad.stdout


def _run_hygiene(fake_body, mode="reap+census", alloc=2, site_cmd='(bash -c "$FAKE")'):
    script = ('ALLOC_NODES=%d\nNODE_DIRTY_NOTE="node_dirty=none"\nFOREIGN_LOAD=0\n'
              'FOREIGN_LOAD_NOTE="foreign_load=0"\nFAKE=%s\n'
              'site_hygiene () { SITE_HYGIENE_CMD=%s; }\n' % (alloc, cs._sh_quote(fake_body), site_cmd)
              + _bash_fn("run_node_hygiene")
              + 'run_node_hygiene "%s" 20 >/dev/null 2>&1\necho "$NODE_DIRTY_NOTE|$REAP_N"\n' % mode)
    return _bash(script).stdout.strip()


def test_run_node_hygiene_clean_only_when_every_node_reported():
    ok2 = 'echo "HYGIENE_OK host=a mode=reap+census"; echo "HYGIENE_OK host=b mode=reap+census"'
    assert _run_hygiene(ok2) == "node_dirty=none|0"
    # dispatch failed / never ran / one node missing / wrong mode -> unknown, never "none"
    assert _run_hygiene("exit 1") == "node_dirty=unknown|0"
    assert _run_hygiene(ok2 + "; exit 3") == "node_dirty=unknown|0"
    assert _run_hygiene('echo "HYGIENE_OK host=a mode=reap+census"') == "node_dirty=unknown|0"
    assert _run_hygiene(ok2.replace("reap+census", "reap")) == "node_dirty=unknown|0"


def test_run_node_hygiene_keeps_a_real_dirty_finding_and_reap_counts():
    body = ('echo "REAPED_ORPHANS host=a n=3 pids=1,2,3,"; echo "NODE_DIRTY host=a cpu=64 busy=99";'
            ' echo "HYGIENE_OK host=a mode=reap+census"')
    assert _run_hygiene(body) == "node_dirty=a:cpu64:99|3"       # incomplete, but not downgraded
    # a reap-only call says nothing about dirtiness: the sticky note is not touched
    assert _run_hygiene("exit 1", mode="reap") == "node_dirty=none|0"


def test_run_node_hygiene_noop_site_stays_none():
    # f1: site_hygiene is `true` (srun tracks its own steps) -- nothing to count, notes unchanged
    assert _run_hygiene("", site_cmd="(true)") == "node_dirty=none|0"


def test_nc2bp_unknown_case_does_not_end_the_job():
    text = _text(NC2BP)
    assert "exit 1' % sys.argv[1]" not in text
    assert "return 1' % sys.argv[1]" in text
    # an eval'ed `return 1` leaves only the function; the loop goes on and the summary prints
    script = ('convert_case() { eval "$(echo \'echo "unknown case $1" >&2; return 1\')"; echo converted; }\n'
              'fail=0\nfor c in a b; do convert_case "$c" || fail=1; done\necho "=== done fail=$fail"\n')
    out = _bash(script)
    assert out.stdout == "=== done fail=1\n" and out.stderr.count("unknown case") == 2


# ---------------------------------------------------------------------------
# coldstart_sample / case_rebuild
# ---------------------------------------------------------------------------
def test_rebuild_sbatch_honours_exclude_env(monkeypatch):
    from wtune.tools import scc_sweep
    plan = {"nodes": 2, "config": {"ranks": 100, "omp": 2, "nx": 10, "ny": 10, "numtiles": 1},
            "max_dom": 1, "verify_steps": 0, "span_start": "2020-01-01_00:00:00",
            "span_end": "2020-01-01_06:00:00", "interval_seconds": 10800}
    chosen = {"prefix": "GFS", "vtable": "Vtable.GFS", "forcing_interval_h": 3}
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    monkeypatch.delenv(scc_sweep.PREFLIGHT_EXCLUDE_ENV, raising=False)
    assert "#SBATCH --exclude=head\n" in cr.render_rebuild_sbatch(dict(plan), chosen)
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work5")
    monkeypatch.setenv(scc_sweep.PREFLIGHT_EXCLUDE_ENV, "work3,work5")
    assert "#SBATCH --exclude=head,work5,work3\n" in cr.render_rebuild_sbatch(dict(plan), chosen)
    assert cs.sbatch_excluded_nodes() == ["head", "work5", "work3"]


_NL = ("&time_control\n iofields_filename = '/abs/io.txt', '/abs/io.txt',\n"
       " history_interval = 60, 60,\n/\n&domains\n max_dom = 2,\n/\n")


def test_set_list_replaces_quoted_values_containing_slash():
    out = cs.set_list_values(_NL, "iofields_filename", ["'/new/a.txt'", "'/new/b.txt'"])
    assert out == _NL.replace("'/abs/io.txt', '/abs/io.txt',", "'/new/a.txt', '/new/b.txt',")
    assert cs.count_key(out, "iofields_filename") == 1
    # idempotent on its own output, and the group terminator is still the only bare '/'
    assert cs.set_list_values(out, "iofields_filename", ["'/new/a.txt'", "'/new/b.txt'"]) == out
    assert [l for l in out.splitlines() if l.strip() == "/"] == ["/", "/"]


def test_set_list_quoted_slash_without_trailing_comma_and_packed_line():
    nl = "&time_control\n iofields_filename = '/abs/io.txt' history_interval = 60,\n/\n"
    out = cs.set_list_values(nl, "iofields_filename", ["'x'"])
    assert " iofields_filename = 'x'," in out and "/abs" not in out
    assert "history_interval = 60," in out
    # unquoted lists and the '/' terminator on the value line behave as before
    nl2 = "&domains\n e_we = 100, 200, /\n"
    assert cs.set_list_values(nl2, "e_we", [1, 2]) == "&domains\n e_we = 1, 2, /\n"


def _plan_two(d):
    T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
    T1 = datetime.datetime(2020, 1, 1, 6, 0, 0)
    for k, T in enumerate((T0, T1)):
        wrf_dir = os.path.join(d, "win_%d" % k, "wrf")
        os.makedirs(wrf_dir)
        with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
            f.write("\n".join(tcs._rsl_line(T + datetime.timedelta(seconds=72 * i), float(k + 1))
                              for i in range(1, 35)) + "\n")
    return {"dt_s": 72.0, "radt_min": 100000.0,
            "stepra": cs.predict.steps_per_radiation_call(100000.0, 72.0),
            "spinup_min": 1.0, "W_seconds": 60.0 + 30 * 72.0,
            "run_start": T0.strftime(cs._WRF_TIME_FMT),
            "windows": [{"t": T0.strftime(cs._WRF_TIME_FMT), "reason": "t0"},
                        {"t": T1.strftime(cs._WRF_TIME_FMT), "reason": "late"}],
            "_plan_dir": d}


def test_score_window_without_status_line_is_not_ok(tmp_path):
    d = str(tmp_path)
    plan = _plan_two(d)
    # the job died in window 1: no status line and no rsl for it; window 0 is still usable
    os.remove(os.path.join(d, "win_1", "wrf", "rsl.error.0000"))
    with open(os.path.join(d, "windows_status.txt"), "w") as f:
        f.write("WINDOW 0 rc=0 ok=1\n")
    out, _pooled, _anchors = cs.score_windows_from_rsl(plan, None, {})
    assert [w["t"] for w in out] == [plan["windows"][0]["t"]]


def test_score_unlisted_window_is_skipped_even_with_a_partial_rsl(tmp_path):
    d = str(tmp_path)
    plan = _plan_two(d)
    with open(os.path.join(d, "windows_status.txt"), "w") as f:
        f.write("WINDOW 0 rc=0 ok=1\n")
    out, _pooled, _anchors = cs.score_windows_from_rsl(plan, None, {})
    assert len(out) == 1 and out[0]["s_k"] == 1.0


def test_score_without_status_file_keeps_all_and_skips_a_missing_rsl(tmp_path):
    d = str(tmp_path)
    plan = _plan_two(d)
    assert len(cs.score_windows_from_rsl(plan, None, {})[0]) == 2     # unchanged: no status file
    os.remove(os.path.join(d, "win_1", "wrf", "rsl.error.0000"))
    assert len(cs.score_windows_from_rsl(plan, None, {})[0]) == 1
    os.remove(os.path.join(d, "win_0", "wrf", "rsl.error.0000"))
    with pytest.raises(cs.ColdstartError, match="no usable windows"):
        cs.score_windows_from_rsl(plan, None, {})


def _local_chain(*_a, **_k):
    return cs._metdata().resolve_forcing_chain(datetime.datetime(1970, 1, 1), grib_dir="/grib",
                                               now=datetime.datetime(2026, 1, 1))


def test_local_grib_dir_forcing_is_refused_with_a_clear_error(monkeypatch):
    chain = _local_chain()
    assert [s.dataset for s in chain] == ["local"] and chain[0].forcing_interval_h is None
    with pytest.raises(cs.ColdstartError, match="--grib-dir.*no known cadence"):
        cs._forcing_grid(datetime.datetime(1970, 1, 1), 10800, "/grib", None,
                         datetime.datetime(2026, 1, 1))


@pytest.mark.parametrize("sfx", ["", "_d", "_h", "_m", "_s"])
def test_fill_run_defaults_respects_interval_variants(sfx):
    nl = ("&time_control\n history_interval%s = 3,\n restart_interval%s = 6,\n/\n"
          "&domains\n max_dom = 1,\n/\n" % (sfx, sfx))
    ed = cr._NamelistEditor(nl, {}, 1)
    cr._fill_run_defaults(ed, {"dx_km": 12.0}, 6 * 3600)
    assert cs.count_key(ed.txt, "history_interval") == (1 if sfx == "" else 0)
    assert cs.count_key(ed.txt, "restart_interval") == (1 if sfx == "" else 0)
    assert not [c for c in ed.changes if c["key"] in ("history_interval", "restart_interval")]


def test_fill_run_defaults_still_fills_when_no_variant_is_present():
    ed = cr._NamelistEditor("&time_control\n/\n&domains\n max_dom = 1,\n/\n", {}, 1)
    cr._fill_run_defaults(ed, {"dx_km": 12.0}, 6 * 3600)
    assert cs.count_key(ed.txt, "history_interval") == 1
    assert cs.get_first(ed.txt, "restart_interval").strip() == "360"


def test_score_when_every_step_calls_radiation(tmp_path):
    # Sundowner (radt 1 min, dt 75 s): stepra 1, every step was flagged a radiation step and both
    # rebuilt points died on "no normal (non-radiation) steps after spin-up"
    d = str(tmp_path)
    plan = _plan_two(d)
    plan.update(radt_min=1.0, stepra=cs.predict.steps_per_radiation_call(1.0, 72.0))
    assert plan["stepra"] == 1
    out, pooled, _anchors = cs.score_windows_from_rsl(plan, None, {})
    assert [(w["s_k"], w["n_rad"], w["r_local"]) for w in out] == [(1.0, 0, None), (2.0, 0, None)] and not pooled
