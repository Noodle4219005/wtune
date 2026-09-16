import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import phase_ranking as pr  # noqa: E402

LINE = " Timing for main: time 2019-11-26_12:%02d:00 on domain   %d:    %.5f elapsed seconds\n"


def _run_dir(tmp_path, name, times, domain=1, other_domain_times=()):
    d = tmp_path / name
    d.mkdir()
    lines = [LINE % (i % 60, domain, t) for i, t in enumerate(times)]
    lines += [LINE % (i % 60, domain + 1, t) for i, t in enumerate(other_domain_times)]
    (d / "rsl.error.0000").write_text("".join(lines))
    return str(d)


def test_per_step_times_reads_only_the_requested_domain(tmp_path):
    d = _run_dir(tmp_path, "c_A_r1_1", [1.0, 2.0, 3.0], other_domain_times=[9.0, 9.0])
    assert pr.per_step_times(d, domain=1) == [1.0, 2.0, 3.0]
    assert pr.per_step_times(d, domain=2) == [9.0, 9.0]


def test_windows_drop_the_first_step(tmp_path):
    assert pr.windows([9.0, 1.0, 2.0, 3.0, 4.0], 2, skip_first=1) == [[1.0, 2.0], [3.0, 4.0]]


def test_stable_ranking_is_reported_as_stable(tmp_path, capsys):
    a = _run_dir(tmp_path, "c_A_r1_1", [9.0] + [1.0] * 6)
    b = _run_dir(tmp_path, "c_B_r1_1", [9.0] + [2.0] * 6)
    assert pr.main(["--run-dir", a, b, "--windows", "3"]) == 0
    out = capsys.readouterr().out
    assert "RANKING STABLE: True" in out
    assert "window 1 ranking: A < B" in out and "window 3 ranking: A < B" in out


def test_ranking_flip_between_windows_is_reported(tmp_path, capsys):
    # A is fastest early and slowest late; B the other way round
    a = _run_dir(tmp_path, "c_A_r1_1", [9.0, 1.0, 1.0, 5.0, 5.0])
    b = _run_dir(tmp_path, "c_B_r1_1", [9.0, 2.0, 2.0, 3.0, 3.0])
    assert pr.main(["--run-dir", a, b, "--windows", "2"]) == 0
    out = capsys.readouterr().out
    assert "RANKING STABLE: False" in out
    assert "best per window: A -> B" in out


def test_missing_rsl_is_skipped_not_fatal(tmp_path, capsys):
    good = _run_dir(tmp_path, "c_A_r1_1", [9.0] + [1.0] * 6)
    empty = tmp_path / "c_B_r1_1"
    empty.mkdir()
    assert pr.main(["--run-dir", good, str(empty), "--windows", "2"]) == 0
    assert "B" not in capsys.readouterr().out.split("window 1")[0].split("\n")[1]
