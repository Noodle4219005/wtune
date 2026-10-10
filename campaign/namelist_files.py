#!/usr/bin/env python3
"""Stage the files a namelist names into a run dir (symlinks), and merge its iofields files.

WRF 4.8 hangs at input_input when an iofields file is missing (climate-ea50, scc probe jobs
2541-2545), so every file the NAMELIST references is staged, not only the case's input list.
Stdlib only, Python 3.6.
"""
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from namelist_streams import namelist_file_refs
from src_bridge import add_src_to_path

StageResult = Dict[str, List[tuple]]


def _search_dirs(search_dirs: Sequence[Optional[str]]) -> List[str]:
    """The existing directories among search_dirs, absolute, first occurrence only."""
    dirs = []
    for s in search_dirs:
        if s and os.path.isdir(s) and os.path.abspath(s) not in dirs:
            dirs.append(os.path.abspath(s))
    return dirs


def _usable(path: str, name: str) -> bool:
    """A name without .bp is a netCDF/text file: a directory of that name is a BP input or a
    per-record container (BP cache) and must not be staged in its place."""
    return name.endswith(".bp") or not os.path.isdir(path)


def _find_sources(dirs: Sequence[str], rel: str, name: str) -> Tuple[Optional[str], List[str]]:
    """(dir, sorted matches of `rel` in it) for the first of `dirs` holding any; else (None, [])."""
    for d in dirs:
        srcs = sorted(p for p in glob.glob(os.path.join(d, rel)) if _usable(p, name))
        if srcs:
            return d, srcs
    return None, []


def _link_sources(key: str, src_dir: str, srcs: Sequence[str], run_dir: str) -> List[tuple]:
    """Symlink srcs into run_dir (same relative path); an existing entry is never replaced."""
    linked = []
    for src in srcs:
        rel = os.path.relpath(src, src_dir)
        dst = os.path.join(run_dir, rel)
        if os.path.lexists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
        linked.append((key, rel, src))
    return linked


def stage_namelist_files(text: str, run_dir: str,
                         search_dirs: Sequence[Optional[str]]) -> StageResult:
    """Symlink every relative file namelist_file_refs(text) names that run_dir lacks, from the
    first of search_dirs holding it. A '..' or absolute name is only checked, never staged.

    Returns {"linked": [(key, name, src)], "present": [(key, name)], "missing": [(key, name)]};
    missing = required and found nowhere, left absent for the caller to report.
    """
    out = {"linked": [], "present": [], "missing": []}
    dirs = _search_dirs(search_dirs)
    for key, name, required in namelist_file_refs(text):
        rel = os.path.normpath(re.sub(r"<[^>]*>", "*", name))
        outside = os.path.isabs(rel) or rel.split(os.sep)[0] == ".."
        if glob.glob(rel if os.path.isabs(rel) else os.path.join(run_dir, rel)):
            out["present"].append((key, name))
            continue
        src_dir, srcs = (None, []) if outside else _find_sources(dirs, rel, name)
        if srcs:
            out["linked"].extend(_link_sources(key, src_dir, srcs, run_dir))
        elif required:
            out["missing"].append((key, name))
    return out


def merge_iofields(text: str, run_dir: str) -> List[str]:
    """Merge the lines of run_dir's iofields files (same fields, fewer lines for WRF to
    re-parse at every I/O call: src/wtune/iofields.py) -> log lines. Only run_dir is written."""
    add_src_to_path()
    from wtune import iofields as wtune_iofields
    names = [n for key, n, _req in namelist_file_refs(text) if key == "iofields_filename"]
    return wtune_iofields.merge_log_lines(wtune_iofields.merge_in_dir(run_dir, names))


def format_namelist_stage(result: StageResult, search_dirs: Sequence[Optional[str]]) -> List[str]:
    """Log lines for stage_namelist_files' result (missing ones as WARNING lines)."""
    lines = ["namelist file: linked %s (%s) from %s" % (n, k, s) for k, n, s in result["linked"]]
    where = " / ".join(d for d in search_dirs if d) or "(no search dir)"
    for k, n in result["missing"]:
        lines.append("WARNING: the namelist's %s names %s, which exists neither in the run dir "
                     "nor in %s -- left absent%s"
                     % (k, n, where, " (WRF 4.8 hangs at input_input when the iofields file is "
                                     "missing, even with ignore_iofields_warning=.true.)"
                        if k == "iofields_filename" else ""))
    return lines
