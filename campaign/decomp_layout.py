#!/usr/bin/env python3
"""MPI decomposition of a case: legal shapes, the STRICT template table, run order, and the
MODE=physics reference/twin pair. Pure: the scorer and the tables are passed in.
Stdlib only, Python 3.6.
"""
import math
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from case_tables import Case, _active_domain_extents, load_generated_configs

Config = Dict[str, Any]
Shape = Tuple[int, int]
Extent = Union[int, Sequence[int]]
Skipped = List[Tuple[str, str]]
Resolved = Tuple[List[Config], Skipped, Dict[str, List[str]]]
DecompLoader = Callable[[], Optional[ModuleType]]
Decomps = Tuple[Optional[List[str]], Optional[Dict[str, Config]], Optional[str]]
DecompPair = Tuple[Optional[Config], Optional[Config], Optional[str]]

MIN_PATCH = 10      # WRF's smallest patch, share/module_check_a_mundo.F:3014-3029

# ranks*omp == 448 (f1: 4 nodes x 112) on every row; blocktime is the KMP_BLOCKTIME to export,
# 'default' = do not export it.
TEMPLATES = [
    dict(id="C01", ranks=448, omp=1, shape="best", numtiles=1, dist="cyclic", blocktime="0"),
    dict(id="C02", ranks=448, omp=1, shape="best", numtiles=8, dist="cyclic", blocktime="0"),
    dict(id="C03", ranks=448, omp=1, shape="alt1", numtiles=8, dist="cyclic", blocktime="0"),
    dict(id="C04", ranks=448, omp=1, shape="best", numtiles=8, dist="block", blocktime="0"),
    dict(id="C05", ranks=224, omp=2, shape="best", numtiles=13, dist="cyclic", blocktime="0"),
    dict(id="C06", ranks=224, omp=2, shape="best", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C07", ranks=224, omp=2, shape="best", numtiles=14, dist="cyclic", blocktime="0"),
    dict(id="C08", ranks=224, omp=2, shape="best", numtiles=2, dist="cyclic", blocktime="0"),
    dict(id="C09", ranks=224, omp=2, shape="alt1", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C10", ranks=224, omp=2, shape="alt2", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C11", ranks=224, omp=2, shape="best", numtiles=12, dist="block", blocktime="0"),
    dict(id="C12", ranks=224, omp=2, shape="best", numtiles=12, dist="cyclic", blocktime="default"),
    dict(id="C13", ranks=112, omp=4, shape="best", numtiles=16, dist="cyclic", blocktime="0"),
    dict(id="C14", ranks=112, omp=4, shape="best", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C15", ranks=112, omp=4, shape="alt1", numtiles=16, dist="cyclic", blocktime="0"),
    dict(id="C16", ranks=112, omp=4, shape="best", numtiles=13, dist="cyclic", blocktime="0"),
]

_SHAPE_INDEX = {"best": 0, "alt1": 1, "alt2": 2}

_GENERATED_TILE_FIELDS = ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")

# What makes two decompositions different RUNS (patch_x/y derive from nx/ny; ids are labels).
DECOMP_IDENTITY_FIELDS = ("ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime",
                          "numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")


# --------------------------------------------------------------------------- legal shapes
def factor_pairs(n: int) -> List[Shape]:
    return [(i, n // i) for i in range(1, n + 1) if n % i == 0]


def patch_dims(extent: int, nproc: int) -> int:
    """ceil(extent / nproc): the reported patch size (wtune_decomp reports the floor)."""
    q, rem = divmod(extent, nproc)
    return q + 1 if rem else q


def wrf_patch_legal(extent: int, nproc: int, min_patch: int = MIN_PATCH) -> bool:
    """WRF's startup gate: extent // nproc >= min_patch (module_check_a_mundo.F:3014-3029)."""
    return extent // nproc >= min_patch


def legal_shapes(e_we: Extent, e_sn: Extent, ranks: int,
                 min_patch: int = MIN_PATCH) -> List[Tuple[int, int, int, int]]:
    """(nx, ny, patch_x, patch_y) with nx*ny == ranks that WRF accepts on EVERY domain.

    e_we / e_sn: one extent or a per-domain list; the patch sizes reported are d01's.
    """
    we_list = list(e_we) if isinstance(e_we, (list, tuple)) else [e_we]
    sn_list = list(e_sn) if isinstance(e_sn, (list, tuple)) else [e_sn]
    out = []
    for nx, ny in factor_pairs(ranks):
        px = patch_dims(we_list[0], nx)
        py = patch_dims(sn_list[0], ny)
        if all(wrf_patch_legal(we_d, nx, min_patch) and wrf_patch_legal(sn_d, ny, min_patch)
               for we_d, sn_d in zip(we_list, sn_list)):
            out.append((nx, ny, px, py))
    return out


def _fallback_score(e_we: int, e_sn: int, nx: int, ny: int) -> float:
    """|log(patch aspect)| + 0.1 when nx > ny; lower is better."""
    wx = e_we / float(nx)
    wy = e_sn / float(ny)
    return abs(math.log(wx / wy)) + (0.1 if nx > ny else 0.0)


def _scorer_shapes(decomp: ModuleType, e_we: int, e_sn: int, e_vert: int, ranks: int, omp: int,
                   nodes: int) -> List[Shape]:
    """wtune_decomp's distinct (nx, ny) for this ranks x omp group, best score first."""
    cfgs = decomp.enumerate_configs(e_we, e_sn, e_vert, ranks * omp, nodes)
    group = [c for c in cfgs if c["nranks"] == ranks and c["threads"] == omp]
    group.sort(key=lambda c: c["score"])
    seen = []
    for c in group:
        pair = (c["nx"], c["ny"])
        if pair not in seen:
            seen.append(pair)
    return seen


def ranked_shapes(e_we: int, e_sn: int, e_vert: int, ranks: int, omp: int, nodes: int = 4,
                  load_decomp: Optional[DecompLoader] = None) -> Tuple[List[Shape], str]:
    """(legal (nx, ny) pairs best first, 'wtune_decomp' | 'fallback').

    load_decomp() gives src/wtune/decomp.py or None (it needs Python >= 3.7; f1's is 3.6).
    """
    decomp = load_decomp() if load_decomp is not None else None
    if decomp is not None:
        try:
            seen = _scorer_shapes(decomp, e_we, e_sn, e_vert, ranks, omp, nodes)
            if seen:
                return seen, "wtune_decomp"
        except Exception as e:
            sys.stderr.write(
                "NOTE: wtune_decomp.enumerate_configs failed (%s: %s); using fallback\n"
                % (type(e).__name__, e)
            )
    shapes = legal_shapes(e_we, e_sn, ranks)
    shapes.sort(key=lambda s: _fallback_score(e_we, e_sn, s[0], s[1]))
    return [(nx, ny) for nx, ny, px, py in shapes], "fallback"


# --------------------------------------------------------------------------- STRICT templates
def resolve_config(case: Case, template: Config, total_cores: int,
                   load_decomp: Optional[DecompLoader] = None
                   ) -> Tuple[Optional[Config], Optional[str]]:
    """One template on one case -> (config, None) or (None, why it has no legal shape).

    Shapes are ranked on d01, then filtered to those legal on every active domain. A case's
    max_omp caps threads per rank (WRF-Fire crashes with omp > 1: c34, marshall).
    """
    ranks, omp = template["ranks"], template["omp"]
    if ranks * omp != total_cores:
        return None, "ranks(%d)*omp(%d) != %d" % (ranks, omp, total_cores)
    max_omp = case.get("max_omp")
    if max_omp is not None and omp > max_omp:
        return None, "omp(%d) exceeds case max_omp=%d" % (omp, max_omp)
    shapes, method = ranked_shapes(case["e_we"], case["e_sn"], case["e_vert"], ranks, omp,
                                   load_decomp=load_decomp)
    we_list, sn_list = _active_domain_extents(case)
    if len(we_list) > 1:
        legal_pairs = set((nx, ny) for nx, ny, px, py in legal_shapes(we_list, sn_list, ranks))
        shapes = [s for s in shapes if s in legal_pairs]
    idx = _SHAPE_INDEX[template["shape"]]
    if idx >= len(shapes):
        return None, (
            "no legal '%s' shape (index %d) at ranks=%d: only %d legal shape(s) "
            "found via %s (max_dom=%d)" % (template["shape"], idx, ranks, len(shapes),
                                            method, case.get("max_dom", 1))
        )
    nx, ny = shapes[idx]
    px = patch_dims(case["e_we"], nx)
    py = patch_dims(case["e_sn"], ny)
    return dict(
        id=template["id"], ranks=ranks, omp=omp, nx=nx, ny=ny,
        numtiles=template["numtiles"], dist=template["dist"], blocktime=template["blocktime"],
        patch_x=px, patch_y=py, shape=template["shape"], shape_method=method,
    ), None


def dedupe(resolved: List[Config]) -> Tuple[List[Config], Dict[str, List[str]]]:
    """Drop configs that are the same run -> (kept, {kept id: [dropped ids]})."""
    seen = {}
    kept = []
    aliases = {}
    for cfg in resolved:
        key = (cfg["ranks"], cfg["omp"], cfg["nx"], cfg["ny"], cfg["numtiles"],
               cfg["dist"], cfg["blocktime"])
        if key in seen:
            aliases.setdefault(seen[key], []).append(cfg["id"])
        else:
            seen[key] = cfg["id"]
            kept.append(cfg)
    return kept, aliases


def resolve_case(case: Case, templates: Sequence[Config], total_cores: int,
                 load_decomp: Optional[DecompLoader] = None) -> Resolved:
    """Every template on one case -> (kept, [(template id, why skipped)], aliases)."""
    resolved = []
    skipped = []
    for t in templates:
        cfg, err = resolve_config(case, t, total_cores, load_decomp)
        if err:
            skipped.append((t["id"], err))
        else:
            resolved.append(cfg)
    kept, aliases = dedupe(resolved)
    return kept, skipped, aliases


def build_order(config_ids: Sequence[str], reps: Any) -> List[str]:
    """Run order: forward, reversed, forward, ... so drift is split evenly between configs."""
    reps = int(reps)
    if reps <= 0:
        return []
    forward = list(config_ids)
    backward = list(reversed(config_ids))
    order = []
    for i in range(reps):
        order.extend(forward if i % 2 == 0 else backward)
    return order


# --------------------------------------------------------------------------- generated sets
def window_frame_level(value: Any) -> int:
    """0 off, 1 end frame only, 2 mid+end frames. Accepts bools, 0/1/2, 'end', 'mid_end'."""
    if value in (None, False, 0, "0", "", "off", "false", "False"):
        return 0
    if value in (2, "2", "mid_end"):
        return 2
    return 1


def config_extras_rows(configs_list: Sequence[Config], case_id: str) -> List[Tuple[Any, str, str]]:
    """(config_id, physics_namelist, window_frame level) per CONFIG_FILE config (decision 0059).

    physics_namelists[case_id] replaces the case's namelist template for that run ('' = none);
    window_frame is the level of window_frame_level (the fidelity gate needs mid + end frames).
    """
    rows = []
    for cfg in configs_list:
        nls = cfg.get("physics_namelists") or {}
        rows.append((cfg.get("id"), str(nls.get(case_id) or ""),
                     str(window_frame_level(cfg.get("window_frame")))))
    return rows


# --------------------------------------------------------------------------- reference / twin
def _decomp_identity(cfg: Config) -> Tuple[str, ...]:
    out = []
    for k in DECOMP_IDENTITY_FIELDS:
        v = cfg.get(k)
        out.append("0" if v is None or v == "" else str(v))
    return tuple(out)


def describe_decomp(cfg: Config) -> str:
    """One-line human form of a decomposition, for error messages and the job log."""
    return ("ranks=%s omp=%s nproc_x=%s nproc_y=%s numtiles=%s dist=%s blocktime=%s"
            % (cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime")))


def _generated_decomps(config_file: str) -> Decomps:
    """(ids in file order, {id: config}, None) of a generated set, or (None, None, reason)."""
    ids, by_id = [], {}
    for c in load_generated_configs(config_file) or ():
        if c.get("id") is None:
            continue
        cid = str(c["id"])
        if cid not in by_id:
            ids.append(cid)
        by_id[cid] = c
    if not ids:
        return None, None, ("CONFIG_FILE %s carries no configs with an 'id' field "
                            "(expected predictor/decomp_space.py sample|seed output: a bare "
                            "JSON list or {\"configs\": [...]})" % (config_file,))
    return ids, by_id, None


def available_decomps(case_id: str, cases: Dict[str, Case],
                      resolve_strict: Callable[[Case], Resolved],
                      config_file: Optional[str] = None) -> Decomps:
    """(ordered ids, {id: config}, None) or (None, None, reason).

    config_file: that generated set, in file order. Otherwise the case's kept STRICT configs
    (resolve_strict(case)), i.e. the ids `configs.py CASE_ID` prints.
    """
    case = cases.get(case_id)
    if case is None:
        return None, None, "unknown case %r" % (case_id,)
    if not case.get("enabled", True):
        return None, None, ("case %s is disabled: %s"
                            % (case_id, case.get("reason", "(no reason recorded)")))
    if config_file:
        return _generated_decomps(config_file)
    kept, skipped, _aliases = resolve_strict(case)
    if not kept:            # so every template was skipped: len(skipped) is the table's size
        first = skipped[0][1] if skipped else "no reason recorded"
        return None, None, (
            "case %s resolves ZERO of the %d STRICT templates C01-C16 (first reason: %s), so no "
            "template id can name a decomposition for it. Generate a decomposition set instead -- "
            "predictor/decomp_space.py sample %s --n N --seed S --total-cores C --nodes N "
            "--out FILE.json -- and run with CONFIG_FILE=FILE.json plus DECOMP/TWIN_DECOMP ids "
            "taken from that file" % (case_id, len(skipped), first, case_id))
    return [c["id"] for c in kept], dict((c["id"], c) for c in kept), None


def _pick_decomp(role: str, cid: Any, ids: List[str], by_id: Dict[str, Config], case_id: str,
                 source: str) -> Tuple[Optional[Config], Optional[str]]:
    """(config named by DECOMP / TWIN_DECOMP, None) or (None, why the id is unusable)."""
    cid = "" if cid is None else str(cid)
    if not cid:
        return None, "%s is empty: name one of %s" % (role, ", ".join(ids))
    if "_" in cid or cid.split() != [cid]:
        return None, (
            "%s=%r contains an underscore or whitespace. A config id is embedded verbatim in "
            "the ledger run_id (case_cid_rN_job, parsed by predictor/decomp_search.py "
            "parse_run_id) and in the space-separated key=value notes field, so neither "
            "character can appear in it" % (role, cid))
    cfg = by_id.get(cid)
    if cfg is None:
        return None, (
            "%s=%r is not a resolved config for case %s from %s -- available ids: %s"
            % (role, cid, case_id, source, ", ".join(ids)))
    return cfg, None


def resolve_decomp_pair(case_id: str, decomp_id: Any, twin_id: Any, cases: Dict[str, Case],
                        resolve_strict: Callable[[Case], Resolved],
                        config_file: Optional[str] = None) -> DecompPair:
    """MODE=physics's (reference config, twin config, None) or (None, None, reason).

    The twin is the same physics on a DIFFERENT decomposition: the reference-vs-twin spread is
    the chaos floor of fidelity_gate.py, so an identical pair is refused.
    """
    ids, by_id, reason = available_decomps(case_id, cases, resolve_strict, config_file)
    if reason:
        return None, None, reason
    source = ("CONFIG_FILE=%s" % config_file) if config_file else "the STRICT templates C01-C16"
    picked = []
    for role, cid in (("DECOMP", decomp_id), ("TWIN_DECOMP", twin_id)):
        cfg, reason = _pick_decomp(role, cid, ids, by_id, case_id, source)
        if reason:
            return None, None, reason
        picked.append(cfg)
    ref, twin = picked
    if _decomp_identity(ref) == _decomp_identity(twin):
        return None, None, (
            "TWIN_DECOMP=%s is the SAME decomposition as DECOMP=%s (%s). The twin is not a "
            "replicate: it is the same physics on a DIFFERENT decomposition, and the "
            "reference-vs-twin spread is the chaos floor fidelity_gate.py measures every "
            "candidate against. An identical twin reports a floor of ~0 and would relabel every "
            "real candidate LARGE-DRIFT, so pick a twin differing in at least one of: %s"
            % (twin_id, decomp_id, describe_decomp(ref), ", ".join(DECOMP_IDENTITY_FIELDS)))
    return ref, twin, None


# --------------------------------------------------------------------------- point jobs
def resolve_point_plan(order: Sequence[str], point_fused: bool = False,
                       anchor_end: Optional[bool] = None,
                       window_frame: bool = False) -> Dict[str, Any]:
    """A prediction-point job's run list from its ORDER (ANCHOR first, ANCHOREND last).

    Returns dict(order, fused, cid, anchor_end, note). anchor_end None = on, except off when
    fused. Fused only when exactly ONE run remains besides the anchors and no window frame is
    wanted; it drops ANCHOR too and refuses anchor_end (there is no restart to bracket).
    """
    order = list(order)
    note = ""
    cands = [c for c in order if c not in ("ANCHOR", "ANCHOREND")]
    if not cands and order:
        cands = [c for c in order if c == "ANCHOR"]
    fused = bool(point_fused) and len(cands) == 1 and not window_frame
    if point_fused and not fused:
        note = "POINT_FUSED ignored: %s" % (
            "window_frame requested" if window_frame and len(cands) == 1
            else "%d runs after dropping anchors (need exactly 1)" % len(cands))
    if fused:
        if anchor_end is True:
            note = "ANCHOR_END=1 ignored in fused mode (no restart to bracket)"
        return dict(order=cands, fused=True, cid=cands[0], anchor_end=False, note=note)
    if anchor_end is None:
        anchor_end = True
    if not anchor_end:
        order = [c for c in order if c != "ANCHOREND"]
    return dict(order=order, fused=False, cid="", anchor_end=bool(anchor_end), note=note)
