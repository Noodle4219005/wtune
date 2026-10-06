"""The MPI/OpenMP layout a segment runs with (Config), where it comes from (CLI flags over a
tune report's winner) and the per-segment candidates a tune report's ranking offers."""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from .common import CORES_PER_NODE, Record, RunError
from .estimate import ComputeFn, compute_fn_from_tune_row
from .geometry import Geometry


@dataclasses.dataclass(frozen=True, eq=False)
class Config(Record):
    """One layout: ranks = nx x ny on `nodes` nodes, omp threads per rank."""
    id: str
    nx: int
    ny: int
    ranks: int
    numtiles: int
    omp: int
    nodes: Optional[int] = None
    dist: str = "block"                # block | cyclic (mpirun --rank-by node)

    @classmethod
    def coerce(cls, value: Any) -> "Config":
        """A dict may omit id/nodes/dist (wtune_tune.physics_phase passes the bare layout)."""
        return value if isinstance(value, cls) else cls(**{"id": "cli", **value})


@dataclasses.dataclass(frozen=True, eq=False)
class Candidate(Record):
    """A layout and its predicted compute over any step interval."""
    config: Config
    compute: ComputeFn

    @classmethod
    def coerce(cls, value: Any) -> "Candidate":
        if isinstance(value, cls):
            return value
        return cls(config=Config.coerce(value["config"]), compute=value["compute"])


@dataclasses.dataclass(frozen=True, eq=False)
class TuneLayout(Record):
    """The measured winner layout of a tune_report.json."""
    nodes: int
    nx: int
    ny: int
    numtiles: int
    omp: int
    cores_per_node: Optional[int]
    median: Optional[float]            # s/step
    cid: Optional[str]


@dataclasses.dataclass(frozen=True, eq=False)
class Layout(Record):
    """The layout a run uses: CLI flags over the tune report's winner."""
    nodes: int
    nx: int
    ny: int
    numtiles: int
    omp: int


def make_config(nx: int, ny: int, numtiles: int, omp: int, nodes: int, cid: str = "cli",
                dist: str = "block") -> Config:
    nx, ny, numtiles, omp = int(nx), int(ny), int(numtiles), int(omp)
    ranks = nx * ny
    if ranks * omp > nodes * CORES_PER_NODE:
        raise RunError("config %s: %d ranks x %d threads exceeds %d node(s) x %d cores"
                       % (cid, ranks, omp, nodes, CORES_PER_NODE))
    return Config(id=cid, nx=nx, ny=ny, ranks=ranks, numtiles=numtiles, omp=omp, nodes=nodes,
                  dist=dist)


def tune_candidates(report: dict, geo: Geometry, sample_points: Optional[list],
                    calibration: dict, nodes: int) -> list[Candidate]:
    """One Candidate per ranking row of a predicted-mode tune_report.json with measured points."""
    if report.get("nodes") is not None and int(report["nodes"]) != int(nodes):
        raise RunError("--tune-report was measured at %s node(s), --nodes is %d"
                       % (report["nodes"], nodes))
    out = []
    for row in report.get("ranking") or []:
        comp = compute_fn_from_tune_row(row, geo, sample_points, calibration)
        if comp is None:
            continue
        cfg = make_config(row.get("nproc_x"), row.get("nproc_y"), row.get("numtiles") or 1,
                          row.get("omp") or 1, nodes, cid=row.get("id"),
                          dist=row.get("dist") or "block")
        out.append(Candidate(config=cfg, compute=comp))
    if not out:
        raise RunError("--tune-report has no ranking row with measured points and no winner "
                       "with a measured median s/step")
    return out


def tune_report_layout(report: dict) -> Optional[TuneLayout]:
    """The winner of a tune_report.json: top-level 'winner', else node_selection.winner, else
    the fastest node_selection.layouts row at the chosen node count (.final, else .chosen).
    None when the report names none."""
    ns = report.get("node_selection") or {}
    w = next((x for x in (report.get("winner"), ns.get("winner")) if isinstance(x, dict)), None)
    if w is None:
        n = ns.get("final", ns.get("chosen"))
        rows = [r for r in ns.get("layouts") or []
                if n is not None and r.get("nodes") == n and r.get("median") is not None]
        w = min(rows, key=lambda r: r["median"]) if rows else None
    if w is None or any(w.get(k) is None for k in ("nodes", "nx", "ny")):
        return None
    return TuneLayout(nodes=int(w["nodes"]), nx=int(w["nx"]), ny=int(w["ny"]),
                      numtiles=int(w.get("numtiles") or 1), omp=int(w.get("omp") or 1),
                      cores_per_node=w.get("cores_per_node"),
                      median=float(w["median"]) if w.get("median") is not None else None,
                      cid=w.get("cid"))


def resolve_layout(a: Any, layout: Optional[TuneLayout]) -> tuple[Layout, list[str]]:
    """(the run's layout, the flags that override the report) from the CLI options `a` over
    the tune report's winner; RunError when neither gives nodes/nx/ny."""
    out, cli = {}, []
    for k, default in (("nodes", None), ("nx", None), ("ny", None), ("numtiles", 1), ("omp", 1)):
        v = getattr(a, k)
        if v is not None:
            out[k] = v
            if layout is not None and v != getattr(layout, k):
                cli.append("--" + k)
        elif layout is not None:
            out[k] = getattr(layout, k)
        else:
            out[k] = default
    missing = ["--" + k for k in ("nodes", "nx", "ny") if out[k] is None]
    if missing:
        raise RunError("%s required: give them, or a --tune-report whose winner layout names them"
                       % ", ".join(missing))
    return Layout(**out), cli


def rank_mapping_notes(config: Config, cores_per_node: Optional[int] = None) -> list[str]:
    """Warnings about the launch's per-node placement (mpirun ppr:<ceil(ranks/nodes)>:node):
    uneven ranks/node, more cores than a node has, a placement other than the tune report's."""
    nodes, ranks, omp = config.nodes, config.ranks, config.omp
    ppn = -(-ranks // nodes)
    notes = []
    if ranks % nodes:
        notes.append("WARNING: %d ranks do not divide over %d nodes: ppr %d/node places %d "
                     "(wtune tune measured ranks//nodes = %d/node)"
                     % (ranks, nodes, ppn, ppn * nodes, ranks // nodes))
    if ppn * omp > CORES_PER_NODE:
        notes.append("WARNING: %d ranks/node x %d threads = %d cores > %d per node"
                     % (ppn, omp, ppn * omp, CORES_PER_NODE))
    if cores_per_node is not None and ppn * omp != int(cores_per_node):
        notes.append("WARNING: %d ranks/node x %d threads = %d cores/node, the tune report "
                     "measured %s" % (ppn, omp, ppn * omp, cores_per_node))
    return notes
