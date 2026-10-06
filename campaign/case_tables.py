#!/usr/bin/env python3
"""The campaign's case table: cases.json (+ a site overlay), generated config sets, and the
per-case fields other modules read. Stdlib only, Python 3.6.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

Case = Dict[str, Any]

# Deterministic order for --all (not the JSON's own order).
CASE_ORDER = [
    "karina", "conus25", "conus12", "isc21", "maria1km", "lafe",
    "sc26p1", "sc26p2", "sc26p2narr",
    "severe2023", "matthew", "c34", "zhengzhou", "elevconv", "marshall", "bigchill",
]

# (e_we, e_sn, e_vert) of d01 -> (s per step, init + I/O s). GUESSES from the task spec, except
# isc21 / maria1km: one smoke run each (jobs 1074515-1074517), all-steps mean, nest I/O off.
COST_GUESS = {
    (901, 901, 45): (0.30, 70),    # karina, edouard_landfall, lowell_cat4
    (901, 901, 80): (0.50, 110),   # deep_conv_ot, arctic_sbl
    (1501, 1201, 50): (0.60, 120),  # conus25
    (425, 300, 50): (0.08, 30),    # conus12
    (793, 853, 51): (15.0, 230),   # isc21 (measured)
    (900, 900, 50): (5.8, 250),    # maria1km (measured)
    (1000, 1000, 100): (4.0, 150),  # lafe (d01+d02, 500 m nest at ratio 5)
}


def load_cases(path: str, site: Optional[str], overlay_dir: str) -> Dict[str, Case]:
    """cases.json's cases, with <overlay_dir>/cases.<site>.json merged over them key by key.

    Site "f1" (or none) never overlays. A str override may use $VAR / ~ (a site's $HOME is not
    known when the overlay is written).
    """
    with open(path) as f:
        cases = json.load(f)["cases"]
    if not site or site == "f1":
        return cases
    overlay_path = os.path.join(overlay_dir, "cases.%s.json" % site)
    if not os.path.exists(overlay_path):
        return cases
    with open(overlay_path) as f:
        overlay = json.load(f)["cases"]
    merged = dict(cases)
    for cid, over in overlay.items():
        base = dict(merged.get(cid, {}))
        for k, v in over.items():
            if isinstance(v, str):
                v = os.path.expanduser(os.path.expandvars(v))
            base[k] = v
        merged[cid] = base
    return merged


def load_generated_configs(path: str) -> Any:
    """Config dicts of a predictor/decomp_space.py file: a bare list or {"configs": [...]}."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("configs", data)
    return data


def _active_domain_extents(case: Case) -> Tuple[List[int], List[int]]:
    """(e_we list, e_sn list) of the case's active domains (d01's scalars when not nested)."""
    max_dom = case.get("max_dom", 1)
    we_list = case.get("e_we_list", [case["e_we"]])[:max_dom]
    sn_list = case.get("e_sn_list", [case["e_sn"]])[:max_dom]
    return we_list, sn_list


def case_frozen_keys(case: Optional[Case]) -> List[str]:
    """Namelist keys the case's TASK fixes ('frozen_keys'), lowercased; editing one breaks the
    rules of the competition (SC26-P1: time_step, debug_level)."""
    return [str(k).strip().lower() for k in (case or {}).get("frozen_keys") or []]


def _case_effective_max_dom(case: Case) -> int:
    """max_dom as edit_namelist sees it: namelist_overrides.max_dom wins over the case's own."""
    overrides = case.get("namelist_overrides") or {}
    if "max_dom" in overrides:
        try:
            return int(overrides["max_dom"])
        except (TypeError, ValueError):
            pass
    return case.get("max_dom", 1)


def _case_namelist_path(case: Case) -> str:
    return case.get("namelist_template") or os.path.join(case["prep_dir"], "namelist.input")
