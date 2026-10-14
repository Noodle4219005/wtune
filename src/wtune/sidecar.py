#!/usr/bin/env python3
"""wtune/sidecar.py -- `wtune run --sidecar`: work done next to a running segment job.

Every controller poll records the newly valid restarts (wtune.resume) in run_report.json and
hands finished history frames to ONE srun in the job's allocation, on the cores WRF leaves idle,
which converts them BP -> netCDF into the `wtune post` cache (`wtune/sidecar.py convert`);
leftovers are converted after the run on a node of their own. The head process never converts.
With `--compress-history` each converted frame is checked against its BP (tools/bp_nc_same.py)
and then takes the BP directory's place: zlib netCDF that post reads directly, ~4x smaller."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Optional

if not __package__:                    # run by path: src/wtune itself is not an import root
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wtune.post import extract        # noqa: E402  the cache layout is post's, never restated here
from wtune import resume as resume   # noqa: E402
import wtune.machine                # noqa: E402

MAX_BATCH = 64                  # frames per in-job srun
_SAME_PATH = os.path.join(_HERE, "wtune", "tools", "bp_nc_same.py")
FINAL_TIME = "01:00:00"         # --time of the after-run conversion


def post_cache_root(run_dir: str, project_dir: Optional[str] = None) -> str:
    """The netCDF cache root `wtune post` uses for run_dir: $WTUNE_NC_CACHE, else
    <project>/_nc_cache, else <run_dir>/_nc_cache."""
    with extract.nc_cache(base=project_dir):
        return extract.nc_cache_root(run_dir)


def idle_cores(config: Mapping, cores_per_node: int) -> list[int]:
    """Core ids of a node the launch leaves free: mpirun packs ranks x omp from core 0."""
    used = -(-config["ranks"] // config["nodes"]) * config["omp"]
    return list(range(used, cores_per_node))


def cpu_mask(cores: Iterable[int]) -> str:
    return hex(sum(1 << c for c in cores))


def bound_within(mask: str) -> bool:
    """True when this process may run only on cores of hex `mask` (the srun binding took)."""
    return all(int(mask, 16) >> c & 1 for c in os.sched_getaffinity(0))


def bind_to(mask: str) -> bool:
    """Restrict this process to its allowed cores inside hex `mask`; False when there are none.
    scc has no Slurm TaskPlugin, so srun --cpu-bind is a no-op there."""
    wanted = {c for c in os.sched_getaffinity(0) if int(mask, 16) >> c & 1}
    try:
        if wanted:
            os.sched_setaffinity(0, wanted)
    except OSError:
        return False
    return bool(wanted)


def finished_frames(seg_dir: str, job_ended: bool) -> list[str]:
    """BP history dirs of seg_dir that WRF has closed: every written frame a later frame of
    its domain follows -- all of them once the job ended."""
    by_dom = {}
    for dom, _t, path, is_bp in extract._dir_entries(seg_dir):
        if is_bp:
            by_dom.setdefault(dom, []).append(path)
    return sorted(p for paths in by_dom.values() for p in (paths if job_ended else paths[:-1]))


def in_job_srun(job_id: str, cores: list, cores_per_node: int, nodes: int) -> list[str]:
    """srun prefix: one task on the LAST node of job_id's allocation (rank 0 is on the first),
    bound to `cores`."""
    # -c is the whole node so the mask lies inside the step's cpuset, whichever cores Slurm picks
    return ["srun", "--jobid=%s" % job_id, "--overlap", "-N1", "-n1", "--relative=%d" % (nodes - 1),
            "-c", str(cores_per_node), "--cpu-bind=verbose,mask_cpu:%s" % cpu_mask(cores)]


def own_srun(cores_per_node: int, exclude_arg: str) -> list[str]:
    """srun prefix: one whole node of its own (the run's allocation is gone)."""
    return (["srun", "-p", wtune.machine.partition(), "-N1", "-n1", "-c", str(cores_per_node),
             "--time=%s" % FINAL_TIME]
            + [a for a in (exclude_arg, wtune.machine.account_arg()) if a])


def convert_argv(run_dir: str, cache_root: str, frames: Iterable[str],
                 mask: Optional[str] = None, replace: bool = False) -> list[str]:
    """The conversion command: `frames` (none: every frame of run_dir); with `mask` it refuses
    to run unless bound inside it; `replace`: the checked netCDF takes the BP's place."""
    return ([sys.executable, os.path.abspath(__file__), "convert"]
            + (["--cpu-mask", mask] if mask else []) + (["--replace"] if replace else [])
            + ["--run-dir", run_dir, "--nc-cache", cache_root] + list(frames))


def popen_runner(log_path: str) -> Callable[[list], Any]:
    """runner(argv) -> a started subprocess.Popen, its output appended to log_path."""
    def run(argv):
        with open(log_path, "a") as f:
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f,
                                    stderr=subprocess.STDOUT)
    return run


def _never_fatal(method):
    """A sidecar error is logged once and switches the sidecar off; the run goes on."""
    def wrapped(self, *args):
        if self.off:
            return
        try:
            method(self, *args)
        except Exception as e:      # noqa: BLE001
            self.off = True
            self.log("  sidecar: switched off (%s: %s)" % (type(e).__name__, e))
    return wrapped


class Sidecar:
    """One run's sidecar state. runner(argv) starts a process and returns a handle with
    poll()/wait(); at most one is alive at a time."""

    def __init__(self, run_dir: str, cache_root: str, max_dom: int,
                 is_bp: Callable[[str], bool], report: dict, save: Callable[[], None],
                 log: Callable[[str], None], cores_per_node: int, exclude_arg: str,
                 runner: Callable[[list], Any], replace: bool = False) -> None:
        self.run_dir, self.cache_root = run_dir, cache_root
        self.max_dom, self.is_bp = max_dom, is_bp
        self.save, self.log, self.runner, self.off = save, log, runner, False
        self.cores_per_node, self.exclude_arg = cores_per_node, exclude_arg
        self.replace = replace
        self.rec = report.setdefault("sidecar", {"cache": cache_root, "valid_restarts": [],
                                                 "last_valid_restart": None, "converted": 0,
                                                 "not_converted": []})
        self.proc, self.batch = None, []      # the conversion srun in flight and its frames
        self.in_job = True                    # off once an in-job batch converts nothing
        self.sent = set()                     # frames already handed to an in-job srun
        self.backlog = []                     # finished frames of attempts whose job is gone

    def watcher(self, seg_dir: str, job_id: str, config: Mapping) -> Callable[[str], None]:
        """The controller's on_poll(state) for one attempt's job."""
        return lambda state: self.poll(seg_dir, job_id, config, state)

    @_never_fatal
    def poll(self, seg_dir: str, job_id: str, config: Mapping, state: str) -> None:
        self.record_restarts(seg_dir)
        self._reap()
        if state == "RUNNING" and self.proc is None and self.in_job:
            self._convert_in_job(seg_dir, job_id, config)

    def record_restarts(self, seg_dir: str) -> None:
        """Add seg_dir's newly valid restart times to the report."""
        known = set(r["time"] for r in self.rec["valid_restarts"] if r["dir"] == seg_dir)
        new = resume.valid_restart_times(seg_dir, self.max_dom, self.is_bp, skip=known)
        for t in new:
            self.rec["valid_restarts"].append({"dir": seg_dir, "time": t})
            self.log("  sidecar: restart %s valid" % t)
        if new:
            self.rec["last_valid_restart"] = self.rec["valid_restarts"][-1]
            self.save()

    @_never_fatal
    def job_ended(self, seg_dir: str) -> None:
        """The attempt's job is gone: last restart check; its unconverted frames wait for the
        next job or finish()."""
        self.record_restarts(seg_dir)
        self._reap(wait=True)       # an in-job srun dies with the allocation
        self.backlog += [f for f in finished_frames(seg_dir, True)
                         if f not in self.sent and not self._converted(f)]

    @_never_fatal
    def finish(self) -> None:
        """After the last segment: convert every frame still missing from the cache."""
        self._reap(wait=True)
        left = self._missing()
        if left:
            self.log("  sidecar: converting %d frame(s) after the run" % len(left))
            self.runner(own_srun(self.cores_per_node, self.exclude_arg)
                        + convert_argv(self.run_dir, self.cache_root, [],
                                       replace=self.replace)).wait()
            self.rec["not_converted"] = self._missing()
            self.rec["converted"] += len(left) - len(self.rec["not_converted"])
        self._drop_partial_conversions()
        self.log("  sidecar: %d frame(s) in %s%s" % (
            self.rec["converted"], self.cache_root,
            ", %d NOT converted" % len(self.rec["not_converted"])
            if self.rec["not_converted"] else ""))
        self.save()

    def _drop_partial_conversions(self):
        """Temporary files of a conversion srun killed with its job (nothing converts now)."""
        for d, _sub, files in os.walk(self.cache_root):
            for f in files:
                if ".wtune-tmp-" in f:
                    os.remove(os.path.join(d, f))

    def _missing(self):
        return [p for _d, _t, p, is_bp in extract._raw_entries(self.run_dir)
                if is_bp and not self._converted(p)]

    def _converted(self, bp):
        """The cache manifest post.extract writes last exists for BP dir bp, or bp has already
        become its netCDF file (--compress-history)."""
        if os.path.isfile(bp):
            return True
        with extract.nc_cache(directory=self.cache_root):
            return os.path.isfile(extract._sidecar(extract.cached_nc_path(self.run_dir, bp)))

    def _reap(self, wait=False):
        """Book the conversion srun once it has exited (frames it missed stay for finish())."""
        if self.proc is None or (not wait and self.proc.poll() is None):
            return
        self.proc.wait()
        n = sum(1 for f in self.batch if self._converted(f))
        if n == 0 and not wait:     # not the job ending under it: binding or converter trouble
            self.in_job = False
            self.log("  sidecar: in-job batch converted nothing (sidecar.log); the rest waits "
                     "for the end of the run")
        self.rec["converted"] += n
        self.proc, self.batch = None, []
        self.save()

    def _convert_in_job(self, seg_dir, job_id, config):
        cores, nice = idle_cores(config, self.cores_per_node), []
        if not cores and self.replace:  # a full node: share its last core at the lowest priority
            cores, nice = [self.cores_per_node - 1], ["nice", "-n", "19"]
        if not cores:               # WRF uses every core: finish() converts after the run
            return
        ready = [f for f in self.backlog + finished_frames(seg_dir, False)
                 if f not in self.sent][:MAX_BATCH]
        if not ready:
            return
        self.sent.update(ready)
        self.log("  sidecar: converting %d frame(s) in job %s on %d idle core(s)"
                 % (len(ready), job_id, len(cores)))
        self.batch = ready
        self.proc = self.runner(
            in_job_srun(job_id, cores, self.cores_per_node, config["nodes"]) + nice
            + convert_argv(self.run_dir, self.cache_root, ready, cpu_mask(cores), self.replace))


def replace_with_nc(bp: str, nc: str) -> Optional[str]:
    """BP history dir bp becomes its netCDF copy nc, under bp's name, once tools/bp_nc_same.py
    finds the same numbers in both -> None, else why bp was kept."""
    import wtune.bp
    try:
        wtune.bp._run_adios2([_SAME_PATH, bp, nc], "check %s against %s" % (nc, bp))
    except Exception as e:      # noqa: BLE001 -- the BP stays, the reason is reported
        return "kept as BP: %s" % e
    old = bp + ".bp-replaced"   # not a history name: post never reads it
    os.rename(bp, old)
    shutil.move(nc, bp)
    shutil.rmtree(old)
    if os.path.lexists(extract._sidecar(nc)):
        os.remove(extract._sidecar(nc))
    return None


def convert(run_dir: str, cache_root: str, frames: list, replace: bool = False) -> int:
    """Convert `frames` (none: every BP history frame of run_dir) into the post cache -> rc;
    `replace`: each checked netCDF then takes its BP's place."""
    with extract.nc_cache(directory=cache_root):
        todo = list(frames) or [p for _d, _t, p, is_bp in extract._raw_entries(run_dir) if is_bp]
        failed = extract.convert_frames(run_dir, todo)
        if replace:
            for bp in todo:
                if bp not in failed:
                    why = replace_with_nc(bp, extract.cached_nc_path(run_dir, bp))
                    if why:
                        failed[bp] = why
    for bp, err in sorted(failed.items()):
        print("sidecar: %s not converted: %s" % (bp, err), file=sys.stderr)
    return 1 if failed else 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune/sidecar.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("convert")
    cv.add_argument("--run-dir", required=True)
    cv.add_argument("--nc-cache", required=True)
    cv.add_argument("--cpu-mask", default=None)
    cv.add_argument("--replace", action="store_true")
    cv.add_argument("frames", nargs="*")
    a = ap.parse_args(argv)
    if a.cpu_mask and not bound_within(a.cpu_mask) and not bind_to(a.cpu_mask):
        print("sidecar: not bound inside the idle cores %s (running on %s): nothing converted"
              % (a.cpu_mask, sorted(os.sched_getaffinity(0))), file=sys.stderr)
        return 3
    return convert(a.run_dir, a.nc_cache, a.frames, a.replace)


if __name__ == "__main__":
    sys.exit(main())
