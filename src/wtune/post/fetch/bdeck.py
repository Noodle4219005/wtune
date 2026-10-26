"""post/fetch/bdeck.py -- the NHC ATCF b-deck of the project's storm (project.yaml storm.id,
an ATCF id or a HURDAT2 name with storm.hurdat2), for `wtune post tc` and the tc recipe.

Downloads need a login node; a post job on a compute node reads only this file (or --bdeck),
else it falls back to HURDAT2. File: <project>/input/obs/bdeck/b<basin><nn><year>.dat.
"""
from __future__ import annotations

import os

from wtune.post import atcf, sources

NAME = "bdeck"
HELP = "NHC ATCF best-track b-deck of the project's storm (project.yaml storm:)"
PROVENANCE = dict(sources.REGISTRY["bdeck"])


def add_arguments(parser):
    sources.add_window_arguments(parser)     # wtune submit passes them; a b-deck is the storm's
    parser.add_argument("--storm", default=None,
                        help="ATCF id (AL142016) or HURDAT2 name; default project.yaml storm.id")


def _storm(project, storm=None):
    from wtune.post import tc
    if storm and tc.ATCF_ID_RE.match(storm.strip()):
        return storm.strip().upper()
    import wtune.project
    sid, hurdat2 = wtune.project.project_storm(wtune.project.load_project(project))
    return tc.resolve_storm(hurdat2, storm or sid) if (storm or sid) else None


def local_path(project, storm_id):
    """The fetched b-deck of `storm_id` (an ATCF id) in the project, or None."""
    from wtune.post import tc
    basin, cy, year = tc.storm_id_parts(storm_id)
    p = os.path.join(sources.dataset_dir(project, NAME), "b%s%s%d.dat" % (basin.lower(), cy, year))
    return p if os.path.isfile(p) else None


def present(project, run_dir=None):
    """Nothing to fetch without a storm; else whether its b-deck is here."""
    from wtune.post import tc
    try:
        sid = _storm(project)
    except OSError:                      # no project.yaml: not a project, nothing to fetch
        return True
    except tc.TCError:                   # a storm that does not resolve is not here either
        return False
    return not sid or local_path(project, sid) is not None


def run(args):
    from wtune.post import tc
    try:
        sid = _storm(args.project, getattr(args, "storm", None))
    except (OSError, tc.TCError) as e:
        print("bdeck: %s" % e)
        return []
    if not sid:
        print("bdeck: no storm (--storm, or project.yaml storm.id)")
        return []
    basin, cy, year = tc.storm_id_parts(sid)
    name = "b%s%s%d.dat" % (basin.lower(), cy, year)
    dest = os.path.join(sources.dataset_dir(args.project, NAME), name)
    url = atcf.NHC_BDECK_URL.format(year=year, basin=basin.lower(), cy=cy)
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    if os.path.isfile(dest) and known.get(name) == sources.sha256_file(dest):
        return [dest]
    try:
        atcf.fetch_bdeck(year, basin, cy, dest)
    except atcf.ATCFError as e:
        print("bdeck: %s" % e)
        return []
    sources.write_provenance(args.project, NAME, PROVENANCE, [url],
                             {name: sources.sha256_file(dest)})
    return [dest]
