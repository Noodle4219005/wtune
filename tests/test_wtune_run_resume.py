#!/usr/bin/env python3
"""`wtune run --resume`: failure classification, valid restarts, the resumed attempt, and how
post/watch list its directory. Fakes only: no job, no adios2."""
import datetime as dt
import os


import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import predict as wp
from wtune import resume as wres
import wtune.watch
from test_wtune_run import (REAL_BP, Fakes, fake_bp)  # noqa: F401  (autouse fixture)

FMT = "%Y-%m-%d_%H:%M:%S"
T0 = dt.datetime(2020, 6, 1, 0)
MAIN = "Timing for main: time %s on domain   1:    1.00000 elapsed seconds"
RST = "Timing for Writing restart for domain %8d:    2.00000 elapsed seconds"


def _at(hours):
    return T0 + dt.timedelta(hours=hours)


def _rsl(start, stop, rst_times, max_dom, step_s=60):
    """rsl of a job that ran d01 steps from start to stop and finished a restart at rst_times."""
    lines, t = [], start
    while t < stop:
        t += dt.timedelta(seconds=step_s)
        lines.append(MAIN % t.strftime(FMT))
        if t in rst_times:
            lines += [RST % d for d in range(1, max_dom + 1)]
    return "\n".join(lines) + "\n"


class Dying(Fakes):
    """An outcome ("died", STATE, HOURS) is a job that wrote the restarts its namelist asks for
    (restart_interval from the run start), stopped HOURS after the run start without SUCCESS,
    and that sacct reports as STATE."""

    def __init__(self, outcomes=None, rsl_extra=None):
        super().__init__(outcomes)
        self.states, self.steps, self.rsl_extra = {}, {}, rsl_extra or {}

    def sbatch(self, cmd):
        if not (self.outcomes and isinstance(self.outcomes[0], tuple)):
            return super().sbatch(cmd)
        _died, state, hours = self.outcomes.pop(0)
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        nl = wp.parse_namelist(os.path.join(seg, "namelist.input"))
        info = wtune.watch.run_info(nl)
        every = dt.timedelta(minutes=nl["time_control"]["restart_interval"][0])
        rst = [T0 + j * every for j in range(1, 100)
               if info["start"] < T0 + j * every <= _at(hours)]
        with open(os.path.join(seg, "rsl.error.0000"), "w") as f:
            f.write(_rsl(info["start"], _at(hours), rst, len(info["domains"])))
        for name, text in self.rsl_extra.items():
            with open(os.path.join(seg, name), "w") as f:
                f.write(text)
        for t in rst:
            for d in range(1, len(info["domains"]) + 1):
                p = os.path.join(seg, "wrfrst_d%02d_%s" % (d, t.strftime(FMT)))
                os.makedirs(p, exist_ok=True)
                open(os.path.join(p, "md.idx"), "w").close()
        self.submitted.append((seg, "died"))
        self.snap_by_dir[seg] = "died"
        self.steps[seg] = int((_at(hours) - info["start"]).total_seconds() // 60)
        self.states[str(1000 + len(self.submitted))] = state
        return "Submitted batch job %d\n" % (1000 + len(self.submitted))

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        return "%s|%s|00:10:00\n" % (jid, self.states.get(jid, "COMPLETED"))

    def snapshot(self, rundir):
        if self.snap_by_dir[rundir] != "died":
            return super().snapshot(rundir)
        n = self.steps[rundir]
        return {"status": "running", "warnings": [], "sum_d01_elapsed_s": 2.0 * n,
                "n_d01_steps": n}


def _real_bp(path):
    os.makedirs(path)
    for name in ("md.idx", "data.0"):
        with open(os.path.join(path, name), "w") as f:
            f.write("x")


def test_valid_restart_needs_complete_dirs_and_the_rsl_line(tmp_path):
    seg = tmp_path / "seg_0"
    seg.mkdir()
    name = lambda d, h: str(seg / ("wrfrst_d%02d_%s" % (d, _at(h).strftime(FMT))))  # noqa: E731
    for d in (1, 2):
        _real_bp(name(d, 2))                             # complete, write finished
        _real_bp(name(d, 8))                             # complete, but no rsl line
    _real_bp(name(1, 4))
    os.makedirs(name(2, 4))                              # d02 empty (pre-created only)
    _real_bp(name(1, 6))
    os.makedirs(name(2, 6))
    open(os.path.join(name(2, 6), "data.0"), "w").close()   # d02 partial: no md.idx
    (seg / "rsl.error.0000").write_text(_rsl(T0, _at(7), [_at(2), _at(4), _at(6)], 2))
    valid = lambda **kw: wres.valid_restart_times(str(seg), 2, REAL_BP.is_bp, **kw)  # noqa: E731
    assert valid() == ["2020-06-01_02:00:00"]
    assert valid(skip={"2020-06-01_02:00:00"}) == []
    assert wres.restart_sets_on_disk(str(seg), 2, REAL_BP.is_bp) == [
        "2020-06-01_02:00:00", "2020-06-01_08:00:00"]
    assert wres.last_d01_time(str(seg)) == _at(7)


def test_resume_point_rules():
    assert wres.resume_point([120, 240, 360, 480], 0, 720, 360, 540) == (360, None)
    assert wres.resume_point([120, 240, 360, 480], 0, 720, 1, 540) == (480, None)
    assert wres.resume_point([120, 240], 0, 720, 1, 240) == (120, None)    # run not past 240
    assert wres.resume_point([120, 240], 240, 720, 1, 700)[0] is None      # none after the start
    assert wres.resume_point([None, 720], 0, 720, 1, 720)[0] is None       # off grid / the end
    assert wres.resume_point([], 0, 720, 1, 700)[1].startswith("no valid restart")


# ---------------------------------------------------------------------------
# post / watch see the attempts as one run
# ---------------------------------------------------------------------------
def _frames(d, hours):
    d.mkdir(parents=True, exist_ok=True)
    for h in hours:
        (d / ("wrfout_d01_%s" % _at(h).strftime(FMT))).write_text("x")
