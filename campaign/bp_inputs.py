#!/usr/bin/env python3
"""A case's WRF inputs as ADIOS2 BP: plan which files exist as BP or need converting, convert
them into a cache (src/wtune/bp.py), link them, and make fresh per-run copies.

netCDF is no longer used for WRF I/O (2026-10-03): every input stream is read as BP.
Stdlib only, Python 3.6.
"""
import glob
import hashlib
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from case_tables import Case, _case_namelist_path
from namelist_streams import _PER_RECORD_NAME_RE, Stream, adios2_input_streams
from src_bridge import _import_wtune_bp

Plan = Dict[str, Any]
CaseLookup = Callable[[str], Case]      # case id -> case; KeyError for an unknown id
Log = Callable[[str], Any]

FRESH_COPY_PREFIX = "copy_"            # <WTUNE_BP_FRESH_DIR>/copy_<i>/<name>/: run i's inputs


def bp_cache_dir(case_id: str, src_dir: str, root: Optional[str] = None) -> str:
    """${WTUNE_BP_CACHE:-$HOME/wtune-runs/bp-cache}/<case_id>/<sha1(realpath(src_dir))[:12]>.

    Keyed by the source dir too: two prep dirs of one case id never share a cache.
    """
    root = root or os.environ.get("WTUNE_BP_CACHE") or os.path.join(
        os.path.expanduser("~"), "wtune-runs", "bp-cache")
    key = hashlib.sha1(os.path.realpath(src_dir).encode()).hexdigest()[:12]
    return os.path.join(root, case_id, key)


def _src_path(src_dir: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(src_dir, name)


def _concrete_names(src_dir: str, name: str) -> List[str]:
    """`name` itself, or -- for a '<date>' pattern -- the files of src_dir that match it."""
    if "<" not in name:
        return [name]
    pat = _src_path(src_dir, re.sub(r"<[^>]*>", "*", name))
    hits = sorted(set(p[:-3] if p.endswith(".bp") else p
                      for p in glob.glob(pat) + glob.glob(pat + ".bp")))
    return [os.path.relpath(p, src_dir) if not os.path.isabs(name) else p for p in hits]


class _InputPlan(object):
    """resolve_bp_inputs' result under construction, plus what is still to convert."""

    def __init__(self, streams: List[Stream], src_dir: str, cache_dir: str, bp: Any) -> None:
        self.src_dir = src_dir
        self.cache_dir = cache_dir
        self.bp = bp                    # src/wtune/bp.py; imported only when a step needs it
        self.todo = {}                  # type: Dict[str, List[Tuple[str, str]]]  # dir -> names
        self.res = {"links": {}, "convert": [], "existing": [], "missing": [], "ready": [],
                    "needs_convert": [], "per_record": {}, "single_multirecord": [],
                    "streams": streams}     # type: Plan

    def wtune_bp(self) -> Any:
        self.bp = self.bp or _import_wtune_bp()
        return self.bp

    def _existing_bp(self, path: str, base: str) -> Optional[str]:
        """The BP directory already there for `path` (<path>.bp, or <path> itself), if usable.

        A single multi-record BP of a per-record name is not, when its netCDF is next to it:
        that netCDF is converted per record instead.
        """
        existing = (path + ".bp" if os.path.isdir(path + ".bp")
                    else path if os.path.isdir(path) else None)
        if existing is not None and _PER_RECORD_NAME_RE.match(base) and os.path.isfile(path):
            if not self.wtune_bp().is_per_record_dir(existing):
                return None
        return existing

    def _use_existing(self, target: str, base: str) -> None:
        res = self.res
        target = os.path.abspath(target)
        res["existing"].append(target)
        if _PER_RECORD_NAME_RE.match(base):
            bp = self.wtune_bp()
            if bp.is_per_record_dir(target):
                res["per_record"][base] = target
                res["links"].update(bp.bp_input_links(base, target))
                return
            res["single_multirecord"].append(target)
        res["links"][base + ".bp"] = target

    def _plan_convert(self, path: str, base: str, stream: str) -> None:
        res = self.res
        self.todo.setdefault(os.path.dirname(os.path.abspath(path)), []).append(
            (base, base + ".bp"))
        res["convert"].append(os.path.abspath(path))
        if stream not in res["needs_convert"]:
            res["needs_convert"].append(stream)
        if _PER_RECORD_NAME_RE.match(base):
            res["per_record"][base] = os.path.abspath(os.path.join(self.cache_dir, base))

    def _add_file(self, stream: str, name: str) -> bool:
        """Plan one concrete file -> True when it exists as BP or as netCDF."""
        path = _src_path(self.src_dir, name)
        base = os.path.basename(name)
        existing = self._existing_bp(path, base)
        if existing is not None:
            self._use_existing(existing, base)
        elif os.path.isfile(path):
            self._plan_convert(path, base, stream)
        else:
            return False
        return True

    def add_stream(self, st: Stream) -> bool:
        """Plan every file of one stream -> True when all its required names resolved."""
        ok = True
        for name in st["names"]:
            found = [self._add_file(st["stream"], c)
                     for c in _concrete_names(self.src_dir, name)]
            if not any(found) and name in st["required"]:
                self.res["missing"].append((st["stream"], name))
                ok = False
        return ok

    def convert(self, log: Optional[Log]) -> None:
        """Convert what was planned (cached by wtune_bp's manifest) and link the results."""
        bp = self.wtune_bp()
        bp.require_adios2_python()
        for d in sorted(self.todo):
            names = [b for b, _ in self.todo[d]]
            if log:
                hit = getattr(bp, "cached_inputs", lambda *a: None)(d, self.cache_dir, names)
                verb = "using cached" if hit else "converting"
                log("ADIOS2 inputs: %s %s from %s in %s (BP4)"
                    % (verb, " ".join(names), d, self.cache_dir))
            got = bp.convert_inputs(d, self.cache_dir, names=names)   # engine: BP4 (the default)
            for base, link in self.todo[d]:
                self._link_converted(os.path.join(d, base), base, link, (got or {}).get(base))

    def _link_converted(self, src: str, base: str, link: str, target: Optional[str]) -> None:
        if not target or not os.path.isdir(target):
            raise RuntimeError("convert_inputs gave no BP directory for %s (got %r)"
                               % (src, target))
        target = os.path.abspath(target)
        if self.bp.is_per_record_dir(target):
            self.res["per_record"][base] = target
            self.res["links"].update(self.bp.bp_input_links(base, target))
        else:
            self.res["per_record"].pop(base, None)
            self.res["links"][link] = target


def resolve_bp_inputs(text: str, src_dir: str, cache_dir: str, convert: bool = True,
                      bp: Any = None, log: Optional[Log] = None) -> Plan:
    """Every input-stream file of `text` as a BP directory: <src>/<name>.bp, else <src>/<name>
    when it is a directory, else the netCDF converted into cache_dir/<name> (wtune_bp: cached,
    lock-serialized; a per-record name becomes a directory of one-step BPs). convert=False
    plans only. Returns {"links": {link: target}, "convert", "existing", "missing": [(stream,
    name)], "ready", "needs_convert", "per_record": {name: dir}, "single_multirecord", "streams"}.
    """
    streams = adios2_input_streams(text)
    plan = _InputPlan(streams, src_dir, cache_dir, bp)
    status = {}
    for st in streams:
        status[st["stream"]] = plan.add_stream(st)
    plan.res["ready"] = [st["stream"] for st in streams if status[st["stream"]]]
    if convert and plan.todo:
        plan.convert(log)
    return plan.res


# --------------------------------------------------------------------------- a campaign case
def _case_bp_plan(case_id: str, lookup: CaseLookup, bp: Any = None
                  ) -> Tuple[Plan, str, List[str]]:
    """(plan, cache dir, stale names) of a campaign case's inputs, read-only: stale = netCDF
    names whose cached conversion is missing or out of date. KeyError: no such case."""
    c = lookup(case_id)
    src = os.path.expanduser(os.path.expandvars(c.get("bp_dir") or c["prep_dir"]))
    cache = src if c.get("bp_dir") else bp_cache_dir(case_id, src)
    with open(os.path.expanduser(os.path.expandvars(_case_namelist_path(c)))) as f:
        res = resolve_bp_inputs(f.read(), src, cache, convert=False, bp=bp)
    bp = bp or _import_wtune_bp()
    by_dir = {}
    for p in res["convert"]:
        by_dir.setdefault(os.path.dirname(p), []).append(os.path.basename(p))
    stale = [n for d, names in sorted(by_dir.items())
             if bp.cached_inputs(d, cache, names) is None for n in names]
    return res, cache, stale


def bp_cache_status(case_id: str, lookup: CaseLookup, bp: Any = None) -> Tuple[str, str]:
    """("current" | "stale" | "none", detail) of a case's BP input cache; "none" = nothing a
    probe converts (legacy netCDF, unknown case, unreadable namelist)."""
    if os.environ.get("WTUNE_INPUT_FORMAT") == "netcdf":
        return "none", "WTUNE_INPUT_FORMAT=netcdf"
    try:
        _res, cache, stale = _case_bp_plan(case_id, lookup, bp)
    except (KeyError, OSError) as e:
        return "none", "%s: %s" % (type(e).__name__, e)
    if stale:
        return "stale", "%s: %s" % (cache, " ".join(stale))
    return "current", cache


def per_record_sources(case_id: str, lookup: CaseLookup, bp: Any = None
                       ) -> Tuple[Dict[str, str], Optional[str]]:
    """({name: per-record dir} a probe of the case reads, None) or ({}, reason). Read-only:
    a netCDF input must already be current in the BP cache."""
    try:
        res, cache, stale = _case_bp_plan(case_id, lookup, bp)
    except KeyError:
        return {}, "unknown case %s" % case_id
    stale = [n for n in stale if n in res["per_record"]]
    if stale:
        return {}, "BP cache %s not current for %s (run campaign/nc2bp.sbatch first)" % (
            cache, " ".join(stale))
    return dict(res["per_record"]), None


def fresh_probe_inputs(case_id: str, dest_dir: str, copies: Any, lookup: CaseLookup,
                       bp: Any = None) -> Optional[str]:
    """dest_dir holding `copies` fresh per-record input copies (copy_<i>/<name>/), or None
    with a WARNING on stderr.

    Every wrf.exe of a probe job must open its own inodes (NFS open stall, wiki 2026-10-03
    adios2 gate). Without reflink one copy only: full copies cost too much.
    """
    bp = bp or _import_wtune_bp()
    sources, why = per_record_sources(case_id, lookup, bp=bp)
    if not sources:
        if why:
            sys.stderr.write("WARNING: no fresh BP input copy for %s: %s\n" % (case_id, why))
        return None
    for i in range(max(1, int(copies))):
        stats = {}
        bp.fresh_input_copy(sources, os.path.join(dest_dir, "%s%d" % (FRESH_COPY_PREFIX, i)),
                            stats=stats)
        if not stats.get("reflink"):
            break
    return dest_dir


# --------------------------------------------------------------------------- linking a plan
def _link_idempotent(dst: str, target: str) -> None:
    """dst -> target symlink; an identical link is kept, a stale link replaced, a real file or
    dir refused (never deleted). dst == target (a cache entry itself) is a no-op."""
    if os.path.abspath(dst) == os.path.abspath(target):
        return
    if os.path.islink(dst):
        if os.readlink(dst) == target:
            return
        os.remove(dst)
    elif os.path.lexists(dst):
        raise RuntimeError("%s exists and is not a link -- refusing to replace it" % dst)
    os.symlink(target, dst)


def link_plan_into_cache(res: Plan, cache: str, log: Log) -> None:
    """Make cache/<name>.bp (single BPs) and cache/<name> (per-record containers) point at
    every input of the plan, so the sbatch finds them all under $BP_DIR."""
    containers = set(res["per_record"].values())
    for link, target in sorted(res["links"].items()):
        if os.path.dirname(target) in containers:
            continue                       # a record file: reached via its container below
        _link_idempotent(os.path.join(cache, link), target)
        log("BP input %s -> %s" % (link, target))
    for name, target in sorted(res["per_record"].items()):
        _link_idempotent(os.path.join(cache, name), target)     # no-op when converted here
        log("BP input %s -> %s (per record, %d files)"
            % (name, target, sum(1 for t in res["links"].values()
                                 if os.path.dirname(t) == target)))
