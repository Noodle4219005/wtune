"""A segment's run directory: links to the case's files and WRF_TREE/run, a fresh namelist,
an adios2.xml selecting BP4 for its restart files, fresh per-record input copies and
pre-created ADIOS2 output directories. Writes only inside the segment dir."""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
from typing import Callable, Iterable, Optional

from ._deps import coldstart, sweep, wp
from .common import WRF_FMT, RunError

_CASE_LINK_SKIP = ("namelist.", "rsl.", "wrfout", "wrfrst")
_WRF_RUN_SKIP = ("namelist.input", "namelist.output")
_WRF_RUN_SKIP_PREFIX = ("wrfout", "wrfinput", "wrfbdy", "wrfrst", "rsl.")
_XML_IO_RE = re.compile(r'(?s)[ \t]*<io\s+name="([^"]*)".*?</io>[ \t]*\n?')
FRESH_BP_SUBDIR = "_bp_inputs"         # per-attempt copy of the per-record inputs


# ---------------------------------------------------------------------------
# adios2.xml (pure)
# ---------------------------------------------------------------------------
def merge_adios2_xml(base: Optional[str], extra: str) -> str:
    """`base` (the case's adios2.xml, or None) plus every <io> block of `extra` whose name
    `base` does not define -- ADIOS2 refuses a config declaring one io name twice."""
    if not base or "</adios-config>" not in base:
        return extra
    if not extra:
        return base
    have = set(m.group(1) for m in _XML_IO_RE.finditer(base))
    add = "".join(m.group(0) if m.group(0).endswith("\n") else m.group(0) + "\n"
                  for m in _XML_IO_RE.finditer(extra) if m.group(1) not in have)
    return base.replace("</adios-config>", add + "</adios-config>", 1)


def render_adios2_restart_xml(restart_times: list, max_dom: int,
                              base: Optional[str] = None) -> str:
    """BP4 engine per restart file name (strict_campaign.sbatch write_adios2_restart_xml).
    `base` keeps its entries: ours go before </adios-config>, skipping a name it defines."""
    have = set(m.group(1) for m in _XML_IO_RE.finditer(base or ""))
    ios = ""
    for t in restart_times:
        for d in range(1, max_dom + 1):
            if "wrfrst_d%02d_%s" % (d, t) not in have:
                ios += ('  <io name="wrfrst_d%02d_%s">\n    <engine type="BP4"/>\n  </io>\n'
                        % (d, t))
    if base and "</adios-config>" in base:
        return base.replace("</adios-config>", ios + "</adios-config>", 1)
    return '<?xml version="1.0"?>\n<adios-config>\n' + ios + "</adios-config>\n"


# ---------------------------------------------------------------------------
# restart and output names
# ---------------------------------------------------------------------------
def restart_files(rundir: str, t: str, max_dom: int) -> dict:
    """{domain: [paths]} of wrfrst_d<NN>_<t>* in rundir."""
    return {d: sorted(glob.glob(os.path.join(rundir, "wrfrst_d%02d_%s*" % (d, glob.escape(t)))))
            for d in range(1, max_dom + 1)}


def _stamps(t_a: dt.datetime, t_b: dt.datetime, step_s: float, include_start: bool) -> list[str]:
    """WRF_FMT times from t_a (inclusive or not) to t_b inclusive, every step_s seconds."""
    out, t = [], t_a if include_start else t_a + dt.timedelta(seconds=step_s)
    while step_s > 0 and t <= t_b:
        out.append(t.strftime(WRF_FMT))
        t += dt.timedelta(seconds=step_s)
    return out


def output_dir_names(nl: dict, t_a: dt.datetime, t_b: dt.datetime, max_dom: int) -> list[str]:
    """Names of the ADIOS2 history/restart outputs a segment [t_a, t_b] writes (parsed namelist).
    A custom history_outname gives no history names; a stream whose io_form is not 14 none (a
    netCDF writer cannot create a file over a pre-created directory); other streams are never
    listed."""
    tc = lambda key, d=0, default=None: wp.g_dom(nl, "time_control", key, d, default)  # noqa: E731
    bp = lambda key: int(wp.g(nl, "time_control", key, 14) or 0) == 14  # noqa: E731
    names = []
    for d in range(max_dom if bp("io_form_history") else 0):
        iv = round(coldstart.wrf_interval_minutes(nl, "history_interval", d) * 60)
        if iv > 0 and not wp.g(nl, "time_control", "history_outname"):
            per_file = max(1, int(tc("frames_per_outfile", d, 1) or 1))
            names += ["wrfout_d%02d_%s" % (d + 1, t)
                      for t in _stamps(t_a, t_b, iv, True)[::per_file]]
    rst = round(coldstart.wrf_interval_minutes(nl, "restart_interval") * 60)
    for t in _stamps(t_a, t_b, rst if bp("io_form_restart") else 0, False):
        names += ["wrfrst_d%02d_%s" % (d, t) for d in range(1, max_dom + 1)]
    return names


def precreate_output_dirs(seg_dir: str, names: list) -> int:
    """Empty output directories in seg_dir, so rank 0 sends no NFS CREATE during the run."""
    for n in names:
        os.makedirs(os.path.join(seg_dir, n), exist_ok=True)
    return len(names)


def remove_empty_output_dirs(seg_dir: str, names: list) -> None:
    """Drop pre-created directories the job left empty (os.rmdir refuses a non-empty one)."""
    for n in names:
        try:
            os.rmdir(os.path.join(seg_dir, n))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# the directory
# ---------------------------------------------------------------------------
def _in_wrf_tree(path: str) -> bool:
    """True when `path`, links resolved, lies inside a WRF source tree."""
    d = os.path.dirname(os.path.realpath(path))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "main")) and os.path.isdir(os.path.join(d, "Registry")):
            return True
        d = os.path.dirname(d)
    return False


def _case_link_ok(name: str, path: str, skip_adios2_xml: bool,
                  wrf_run_names: Iterable[str] = ()) -> bool:
    """A case that was once a run directory still holds the wrf.exe and the table links of
    the WRF it ran with. The run uses WRF_TREE's: never an executable of the case, and not a
    link into some WRF tree for a file WRF_TREE/run has. A table the case ships as its own
    file stays the case's."""
    if name == "wrf.out" or name.startswith(_CASE_LINK_SKIP) or name.endswith(".exe"):
        return False
    if name in wrf_run_names and os.path.islink(path) and _in_wrf_tree(path):
        return False
    if skip_adios2_xml and name == "adios2.xml":
        return False
    if os.path.isfile(path):
        return True
    return os.path.isdir(path) and name.endswith(".bp")   # ADIOS2 wrfinput/wrfbdy inputs


def _missing_namelist_files(text: str, case_dir: str) -> list[tuple[str, str]]:
    """[(key, name)] files the namelist requires (configs.namelist_file_refs) that neither
    CASE_DIR (relative name) nor the filesystem (absolute name) has."""
    out = []
    for key, name, required in sweep.campaign_configs.namelist_file_refs(text):
        pat = re.sub(r"<[^>]*>", "*", name)
        if required and not glob.glob(pat if os.path.isabs(pat) else os.path.join(case_dir, pat)):
            out.append((key, name))
    return out


def fresh_bp_links(bp_links: dict, sources: dict, copied: dict) -> dict:
    """bp_links with every record link into a source container (sources {name: dir}) redirected
    to the same file in its fresh copy (copied {name: dir})."""
    by_src = {os.path.abspath(d): copied[n] for n, d in sources.items() if n in copied}
    out = {}
    for name, target in bp_links.items():
        new = by_src.get(os.path.dirname(os.path.abspath(target)))
        out[name] = os.path.join(new, os.path.basename(target)) if new else target
    return out


def _link_case_files(case_dir: str, seg_dir: str, restart_links: Iterable[str],
                     skip_adios2_xml: bool, wrf_tree: Optional[str] = None) -> None:
    run_dir = os.path.join(wrf_tree, "run") if wrf_tree else None
    wrf_run_names = set(os.listdir(run_dir)) if run_dir and os.path.isdir(run_dir) else set()
    for name in sorted(os.listdir(case_dir)):
        src = os.path.join(case_dir, name)
        if _case_link_ok(name, src, skip_adios2_xml, wrf_run_names):
            os.symlink(src, os.path.join(seg_dir, name))
    for src in restart_links:
        os.symlink(os.path.abspath(src), os.path.join(seg_dir, os.path.basename(src)))


def _link_wrf_run_files(wrf_tree: Optional[str], seg_dir: str) -> None:
    """WRF_TREE/run/* not already present (tables, wrf.exe)."""
    run_dir = os.path.join(wrf_tree, "run") if wrf_tree else None
    if not (run_dir and os.path.isdir(run_dir)):
        return
    for name in sorted(os.listdir(run_dir)):
        if name in _WRF_RUN_SKIP or name.startswith(_WRF_RUN_SKIP_PREFIX):
            continue
        dst = os.path.join(seg_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(run_dir, name), dst)


def _write_regular_files(seg_dir: str, namelist_text: str, adios2_xml: Optional[str]) -> None:
    nml = os.path.join(seg_dir, "namelist.input")
    if os.path.lexists(nml):
        os.remove(nml)            # never write through a link
    with open(nml, "w") as f:
        f.write(namelist_text)
    if adios2_xml is not None:
        with open(os.path.join(seg_dir, "adios2.xml"), "w") as f:
            f.write(adios2_xml)


def setup_segment_dir(case_dir: str, seg_dir: str, namelist_text: str,
                      restart_links: Iterable[str] = (),
                      adios2_xml: Optional[str] = None, wrf_tree: Optional[str] = None,
                      bp_links: Optional[dict] = None, fresh_sources: Optional[dict] = None,
                      copier: Optional[Callable] = None) -> dict:
    """Create seg_dir (must not exist) and populate it. bp_links {name: BP} win over same-named
    case links; fresh_sources {name: per-record dir} are copied by `copier` into
    seg_dir/_bp_inputs and linked there (NFS open stall, wiki 2026-10-03 adios2 gate).
    Returns configs.stage_namelist_files' result."""
    case_dir = os.path.abspath(case_dir)
    seg_dir = os.path.abspath(seg_dir)
    if os.path.commonpath([seg_dir, case_dir]) == case_dir:
        raise RunError("segment dir %s is inside CASE_DIR" % seg_dir)
    os.makedirs(seg_dir)
    _link_case_files(case_dir, seg_dir, restart_links, adios2_xml is not None, wrf_tree)
    _link_wrf_run_files(wrf_tree, seg_dir)
    _write_regular_files(seg_dir, namelist_text, adios2_xml)
    if fresh_sources:
        copied = copier(fresh_sources, os.path.join(seg_dir, FRESH_BP_SUBDIR))
        bp_links = fresh_bp_links(bp_links or {}, fresh_sources, copied)
    for name, target in sorted((bp_links or {}).items()):
        dst = os.path.join(seg_dir, name)
        if os.path.islink(dst):
            os.remove(dst)        # a case *.bp link of the same name: the resolved one wins
        os.symlink(target, dst)
    # files the namelist names (iofields_filename, wrflowinp, fdda, ...), also in subdirectories
    return sweep.campaign_configs.stage_namelist_files(namelist_text, seg_dir, [case_dir])
