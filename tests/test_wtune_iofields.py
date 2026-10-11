#!/usr/bin/env python3
"""wtune.iofields (merge the lines WRF re-parses at every I/O call) and where it is applied:
`wtune run`'s segment dirs, the probes' link-namelist-files staging, `wtune check` C11."""
import os

import pytest

from wtune import iofields as iof
from wtune import predict as wp
from wtune import preflight as pf
from wtune import run as wr
from test_wtune_run import (NAMELIST_3DOM, Fakes, _make_case, _run, _tree_state,
                            fake_bp)  # noqa: F401  (autouse fixture)


def climate_iofields(lines=12, fields=167, prefix="-:h:0:"):
    """`fields` names over `lines` lines of one prefix (climate-ea50's file: 12 x 167)."""
    names = ["F%03d" % i for i in range(fields)]
    per = -(-fields // lines)
    return "".join(prefix + ",".join(names[i:i + per]) + "\n" for i in range(0, fields, per))


# ---------------------------------------------------------------------------
# parse / count / merge
# ---------------------------------------------------------------------------
def test_parse_counts_and_calls():
    text = "# kept fields\n-:h:0:U,V\n\n+:h:7:T2, Q2\nnot an entry\n"
    got = iof.parse(text)
    assert [(e.op, e.kind, e.stream, e.fields) for e in got] == [
        (None, None, None, ()), ("-", "h", "0", ("U", "V")), (None, None, None, ()),
        ("+", "h", "7", ("T2", " Q2")), (None, None, None, ())]
    assert [e.raw for e in got] == text.splitlines()
    assert iof.counts(text) == (2, 4) and iof.parse_calls(text) == 5 * 2 + 4
    assert iof.parse_calls(climate_iofields()) == 227 and iof.parse_calls("") == 0


def test_merge_adjacent_same_prefix_keeps_order_and_spelling():
    text = "-:h:0:U,V\n-:h:0:w, T\n-:H:0:QVAPOR\n+:h:7:T2\n+:h:7:Q2\n-:h:0:P\n"
    assert iof.merged(text) == "-:h:0:U,V,w, T,QVAPOR\n+:h:7:T2,Q2\n-:h:0:P\n"   # P not adjacent
    one = iof.merged(climate_iofields())
    assert iof.counts(one) == (1, 167) and iof.parse_calls(one) == 172
    assert one == "-:h:0:" + ",".join("F%03d" % i for i in range(167)) + "\n"


def test_other_lines_stay_and_break_adjacency():
    text = "-:h:0:U\n# note\n-:h:0:V\n\n-:h:0:W\n-:h:0:T"          # no final newline either
    assert iof.merged(text) == "-:h:0:U\n# note\n-:h:0:V\n\n-:h:0:W,T"
    odd = "-:h:0:U,,V\n-:h:0:W\n"                # WRF ends the first list at the empty name
    assert iof.merged(odd) is odd


def test_unchanged_when_nothing_merges():
    for text in ("", "-:h:0:U,V\n", "-:h:0:U\n+:h:0:V\n-:i:0:W\n-:h:1:T\n", "# only\n"):
        assert iof.merged(text) is text


def test_lines_split_only_at_the_limits(monkeypatch):
    big = climate_iofields(lines=40, fields=2000)
    out = iof.merged(big)
    assert [len(e.fields) for e in iof.parse(out)] == [900, 900, 200]
    assert [f for e in iof.parse(out) for f in e.fields] == ["F%03d" % i for i in range(2000)]
    monkeypatch.setattr(iof, "MAX_LINE_CHARS", 30)
    out = iof.merged("-:h:0:AAAA,BBBB\n-:h:0:CCCC,DDDD\n-:h:0:EEEE,FFFF\n-:h:0:GGGG\n")
    assert out == "-:h:0:AAAA,BBBB,CCCC,DDDD\n-:h:0:EEEE,FFFF,GGGG\n"
    assert all(len(ln) < 30 for ln in out.splitlines())
    monkeypatch.setattr(iof, "MAX_LINE_CHARS", 12)           # no gain: the lines stay as written
    two = "-:h:0:AAAA\n-:h:0:BBBB\n"
    assert iof.merged(two) is two
    assert iof.MAX_LINE_FIELDS < 999


def test_merge_in_dir_replaces_the_link_not_its_target(tmp_path):
    case, run = tmp_path / "case", tmp_path / "run"
    case.mkdir()
    run.mkdir()
    (case / "io.txt").write_text(climate_iofields())
    (case / "one.txt").write_text("-:h:0:U\n")
    for name in ("io.txt", "one.txt"):
        os.symlink(str(case / name), str(run / name))
    done = iof.merge_in_dir(str(run), ["io.txt", "one.txt", "absent.txt", "../case/io.txt",
                                       str(case / "io.txt")])
    assert done == [("io.txt", 12, 1)]
    assert not (run / "io.txt").is_symlink()
    assert iof.counts((run / "io.txt").read_text()) == (1, 167)
    assert (run / "one.txt").is_symlink()                    # nothing to merge: still the link
    assert (case / "io.txt").read_text() == climate_iofields()
    assert iof.merge_log_lines(done) == [
        "  iofields: io.txt 12 -> 1 line(s) (same fields; WRF parses it at every I/O call)"]


# ---------------------------------------------------------------------------
# wtune run / probe staging
# ---------------------------------------------------------------------------
IOFIELDS_KEY = ' iofields_filename = "iofields_d01.txt", "iofields_d01.txt", "iofields_d01.txt",\n'
NAMELIST_IOFIELDS = NAMELIST_3DOM.replace(" io_form_history ", IOFIELDS_KEY + " io_form_history ")
LOG = "  iofields: iofields_d01.txt 12 -> 1 line(s) (same fields; WRF parses it at every I/O call)"


def _case_with_iofields(tmp_path, text=None):
    case = _make_case(tmp_path, NAMELIST_IOFIELDS)
    (case / "iofields_d01.txt").write_text(climate_iofields() if text is None else text)
    return case


@pytest.mark.parametrize("extra", [[], ["--io", "netcdf"]])
def test_run_stages_a_merged_copy_and_never_touches_the_case(tmp_path, extra):
    case = _case_with_iofields(tmp_path)
    before = _tree_state(case)
    rc, out, rep, lines = _run(tmp_path, Fakes(), extra)       # two segments
    assert rc == 0 and lines.count(LOG) == 1                   # logged once, staged in both
    for seg in ("seg_0", "seg_1"):
        staged = out / seg / "iofields_d01.txt"
        assert staged.is_file() and not staged.is_symlink()
        assert staged.read_text() == iof.merged(climate_iofields())
    assert _tree_state(case) == before
    assert (case / "iofields_d01.txt").read_text() == climate_iofields()


def test_run_leaves_a_one_line_file_linked_and_does_nothing_without_iofields(tmp_path):
    case = _case_with_iofields(tmp_path, "-:h:0:U,V\n")
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0 and not any(ln.startswith("  iofields:") for ln in lines)
    assert os.readlink(out / "seg_0" / "iofields_d01.txt") == str(case / "iofields_d01.txt")
    other = tmp_path / "plain"
    other.mkdir()
    rc, out, rep, lines = _run(other, Fakes())                 # the case names no iofields file
    assert rc == 0 and not any(ln.startswith("  iofields:") for ln in lines)   # tmp paths may say iofields


def test_probe_staging_merges_too(tmp_path, capsys):
    case = _case_with_iofields(tmp_path)
    run = tmp_path / "probe"
    run.mkdir()
    cc = wr.sweep.campaign_configs
    assert cc._cmd_link_namelist_files([str(case / "namelist.input"), str(run), str(case)]) == 0
    assert LOG in capsys.readouterr().out.splitlines()
    assert not (run / "iofields_d01.txt").is_symlink()
    assert iof.counts((run / "iofields_d01.txt").read_text()) == (1, 167)
    assert (case / "iofields_d01.txt").read_text() == climate_iofields()


# ---------------------------------------------------------------------------
# wtune check C11
# ---------------------------------------------------------------------------
CLIMATE = """ &time_control
 run_days = %d, run_hours = 0, run_minutes = 0, run_seconds = 0,
 start_year = 2022, start_month = 12, start_day = 01, start_hour = 00,
 interval_seconds = 21600
 history_interval = 360,
 frames_per_outfile = 4,
 restart_interval = 43200,
 iofields_filename = "iofields_d01.txt",
 io_form_history = 2
 auxinput4_interval = 360,
 io_form_auxinput4 = 2
 /
 &domains
 time_step = 240, max_dom = 1, e_we = 151, e_sn = 111, e_vert = 45,
 /
 &physics
 radt = 32, sst_update = 1,
 /
"""


def _c11(tmp_path, days=396, iofields=climate_iofields(), compute_s=None):
    (tmp_path / "namelist.input").write_text(CLIMATE % days)
    if iofields is not None:
        (tmp_path / "iofields_d01.txt").write_text(iofields)
    findings = []
    pf.check_c11_iofields_parse(wp.parse_namelist(str(tmp_path / "namelist.input")),
                                str(tmp_path), findings, compute_s=compute_s)
    return findings


def test_check_warns_with_the_cost_before_and_after_merging(tmp_path):
    (f,) = _c11(tmp_path)
    assert (f.severity, f.id) == (pf.WARN, "C11")
    # 6734 parses x 227 (172 merged) calls x 0.9 ms
    assert "about 1376 s of this run as written, 1042 s after `wtune run` merges" in f.message
    assert "iofields_d01.txt 12 line(s) / 167 field(s)" in f.message
    assert "6734 parses, 227 get_fieldstr calls each" in f.message
    assert "`+:h:<n>:FIELD,FIELD,...`" in f.fix
    assert "(5 + kept) calls per parse instead of 227" in f.fix


def test_check_is_silent_below_the_threshold(tmp_path):
    assert _c11(tmp_path, days=10) == []                       # 171 parses: 35 s < 60 s
    assert _c11(tmp_path, compute_s=100000.0) == []            # 1376 s < 2% of the compute time
    assert len(_c11(tmp_path, compute_s=2649.0)) == 1
    assert _c11(tmp_path, iofields="") == []
    os.remove(tmp_path / "iofields_d01.txt")
    assert _c11(tmp_path, iofields=None) == []                 # C2 reports the missing file


def test_run_checks_includes_c11(tmp_path):
    (tmp_path / "namelist.input").write_text(CLIMATE % 396)
    (tmp_path / "iofields_d01.txt").write_text(climate_iofields())
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    ids = [f.id for f in pf.run_checks(nl, tmp_path, tmp_path / "namelist.input")
           if f.severity == pf.WARN]
    assert "C11" in ids
