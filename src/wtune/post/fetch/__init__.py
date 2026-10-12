"""post/fetch/ -- observation/reanalysis fetch modules behind `wtune post fetch <name>`.

Every module `post/fetch/<name>.py` is discovered with pkgutil and must expose

    NAME        "<name>" (== the file name)
    HELP        one line
    PROVENANCE  {dataset, provider, url, license, citation, coverage_time, coverage_space,
                 resolution, access}   (all strings)
    add_arguments(parser)     the module's own argparse options
    run(args) -> list[str]    paths written (existing, sha-matching files are skipped and still
                              returned); [] with a printed reason when the data does not exist
                              for the requested time/region

`args` carries the common --project P (required) and --run-dir R (optional, time window and bbox
come from its wrfout, see post/sources.py resolve_window). Output goes under
<project>/input/obs/<NAME>/ (era5: <project>/input/era5/) and the module writes its own
provenance.json (post/sources.py write_provenance).

Optional hook: `present(project, run_dir) -> bool` tells the offline summary that the dataset is
already local. Without it a dataset counts as local when its provenance.json lists a file that
exists on disk.

Compute nodes on scc have no internet: everything here runs on the login node, and
offline_summary() is what `wtune post recipe` prints on a node without it.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import socket

REQUIRED_PROVENANCE = ("dataset", "provider", "url", "license", "citation", "coverage_time",
                       "coverage_space", "resolution", "access")

_cache = None


def discover(refresh=False):
    """({name: module}, {name: error text}) for every post/fetch/*.py. A module that fails to
    import or breaks the contract is reported in the second dict, never raised: one broken
    module must not take `wtune post` down."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    mods, errors = {}, {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        name = info.name
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module("%s.%s" % (__name__, name))
        except Exception as e:  # noqa: BLE001 -- see docstring
            errors[name] = "import failed: %s: %s" % (type(e).__name__, e)
            continue
        problems = check_module(mod)
        if problems:
            errors[name] = "contract: " + "; ".join(problems)
            continue
        mods[mod.NAME] = mod
    _cache = (mods, errors)
    return _cache


def check_module(mod):
    problems = []
    for attr in ("NAME", "HELP", "PROVENANCE", "add_arguments", "run"):
        if not hasattr(mod, attr):
            problems.append("missing %s" % attr)
    prov = getattr(mod, "PROVENANCE", None)
    if isinstance(prov, dict):
        absent = [k for k in REQUIRED_PROVENANCE if not isinstance(prov.get(k), str)]
        if absent:
            problems.append("PROVENANCE lacks string field(s) %s" % ", ".join(absent))
    elif prov is not None:
        problems.append("PROVENANCE is not a dict")
    return problems


def list_text():
    """`wtune post fetch --list`: NAME, HELP and PROVENANCE.coverage_time per module."""
    mods, errors = discover()
    lines = []
    for name in sorted(mods):
        m = mods[name]
        lines.append("%-10s %s\n%-10s coverage_time: %s" % (name, m.HELP, "", m.PROVENANCE["coverage_time"]))
    for name, err in sorted(errors.items()):
        lines.append("%-10s (unavailable) %s" % (name, err))
    return "\n".join(lines)


def run_module(name, args):
    mods, errors = discover()
    if name not in mods:
        raise SystemExit("unknown fetch module %r (%s)" % (
            name, errors.get(name) or "available: " + ", ".join(sorted(mods))))
    return list(mods[name].run(args) or [])


# --------------------------------------------------------------------------------------
# "what is missing locally, and how to get it" (printed by `wtune post recipe` offline)
# --------------------------------------------------------------------------------------

def has_internet(timeout=2.0):
    """Cheap reachability probe (a TCP connect to a public IP, so a node with no DNS or no route
    answers in <= timeout s instead of stalling in gcsfs/urllib retries). An https proxy in the
    environment counts as internet."""
    if any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
        return True
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=timeout):
            return True
    except OSError:
        return False


def is_present(mod, project_dir, run_dir=None):
    hook = getattr(mod, "present", None)
    if hook is not None:
        try:
            return bool(hook(project_dir, run_dir))
        except Exception:  # noqa: BLE001
            return False
    from wtune.post import sources
    prov = sources.read_provenance(project_dir, mod.NAME)
    base = sources.dataset_dir(project_dir, mod.NAME)
    return bool(prov and any(os.path.exists(os.path.join(base, rel)) for rel in prov.get("files", {})))


def fetch_command(name, project_dir, run_dir=None):
    cmd = "wtune post fetch %s --project %s" % (name, project_dir)
    if run_dir:
        cmd += " --run-dir %s" % run_dir
    return cmd


def missing_locally(project_dir, run_dir=None):
    mods, _ = discover()
    return [n for n in sorted(mods) if not is_present(mods[n], project_dir, run_dir)]


def offline_summary(project_dir, run_dir=None, online=None):
    """One line naming every fetchable dataset that is not in <project>, with the login-node
    command that fetches it -- or None when the node has internet (nothing to say: the recipe
    can fetch/stream itself) or nothing is missing."""
    if not project_dir:
        return None
    if online is None:
        online = has_internet()
    if online:
        return None
    names = missing_locally(project_dir, run_dir)
    if not names:
        return None
    return ("no internet on this node; not fetched locally yet: %s -- run on the login node: %s"
            % (", ".join(names), "; ".join("`%s`" % fetch_command(n, project_dir, run_dir)
                                            for n in names)))
