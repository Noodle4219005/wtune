"""`wtune post brief POST_DIR` (decision 0078, stage 2): post/<id>/brief.json, what an agent
reviewing a recipe needs to know before it looks at a figure -- the case, the runs, which
observations passed their checks, what was drawn and what is missing (with each figure's
question from the catalog), and the case's key numbers computed here, so the agent quotes them
instead of reading them off a colour bar.

Written by WTune only; the agent's own outputs are review.json, analysis/*.py and
assessment.md, which `wtune post check` verifies (post/check.py).
"""
from __future__ import annotations

import json
import os

SCHEMA = "wtune-post-brief/1"
BRIEF_FILE = "brief.json"
HIGH_TERRAIN_M = 500.0      # SLP over higher ground is a reduction: no minimum taken there

REVIEW_WHAT = ("every drawn figure: verdict ok|fix|drop, its issues (clarity, colour scale, "
               "units, extent, valid time, where a point/line was sampled) and, for a fix, the "
               "action: project_set (a `wtune project set` drawing flag), analysis (the "
               "analysis/*.py that redraws it) or wtune_defect (a WTune bug report)")
PREVIEW_STEPS = [
    dict(step="review", output="review.json", what=REVIEW_WHAT + "; check the domains against "
         "the case description (where the target area and the storm are)"),
    dict(step="plan", output="review.json `plan`",
         what="the figures to draw after the run: figure, why (the question), how (fields, "
              "times, area, colour scale, observations), and the action that makes it -- "
              "project_set for post.focus/cross-sections, analysis for the case's own figures"),
    dict(step="apply", output="`wtune project set ...` for every project_set action now"),
    dict(step="check", output="`wtune post check post/preview` exits 0"),
]
AGENT_STEPS = [
    dict(step="review", output="review.json", what=REVIEW_WHAT),
    dict(step="analysis", output="<project>/analysis/*.py, run by `wtune post custom`",
         what="the case's own figures: observation and model side by side on one scale, "
              "box-mean time series, threshold scores, Hovmoller, track vs best track "
              "(ctx.kit: post/analysis_kit.py)"),
    dict(step="assessment", output="assessment.md",
         what="sections: setup, large scale, target variables (where, when, how much), "
              "scores, mechanism, conclusion and limits; every paragraph cites a figure or CSV "
              "as a markdown link; decimals next to a CSV link must be in that CSV"),
    dict(step="check", output="`wtune post check POST_DIR` exits 0"),
]


def _observations(project_dir, run_dir):
    from wtune.post import agent_obs
    out = dict(agent=[], fetched=[], not_fetched=[])
    for name, d, prov in agent_obs.datasets(project_dir):
        errors, _warnings = agent_obs.check(d)
        out["agent"].append(dict(name=name, kind=prov.get("kind"), dataset=prov.get("dataset"),
                                 path=d, ok=not errors, first_error=errors[0] if errors else None))
    try:
        from wtune.post import fetch
        mods, _ = fetch.discover()
        for name in sorted(mods):
            key = "fetched" if fetch.is_present(mods[name], project_dir, run_dir) else "not_fetched"
            out[key].append(name)
    except Exception as e:                                  # noqa: BLE001
        out["fetch_error"] = "%s: %s" % (type(e).__name__, e)
    return out


def _slp_min(run_dir, domain, spinup_hours):
    """dict(hpa, time, lat, lon) of the lowest sea-level pressure after the spin-up over cells
    below HIGH_TERRAIN_M (and 2 cells off higher ground) inside the boundary zone; None."""
    import netCDF4
    import numpy as np
    from scipy.ndimage import binary_dilation
    from wtune.post import caseinfo
    from wtune.post.focus import _hours
    from wtune.post.extract import list_frame_refs
    wrf_mod = __import__("wrf")
    frames = list_frame_refs(run_dir, domain=domain)
    if not frames:
        return None
    after = [f for f in frames if _hours(frames[0][0], f[0]) >= spinup_hours] or frames
    try:
        bw = caseinfo.spec_bdy_width(run_dir)
    except (caseinfo.CaseInfoError, OSError):
        bw = caseinfo.DEFAULT_SPEC_BDY_WIDTH
    best = None
    for t, path, ti in after:
        with netCDF4.Dataset(path) as ds:
            slp = np.asarray(wrf_mod.getvar(ds, "slp", timeidx=ti, units="hPa"), dtype="f8")
            hgt = np.asarray(ds.variables["HGT"][ti], dtype="f8")
            ok = ~binary_dilation(hgt > HIGH_TERRAIN_M, iterations=2)
            if bw:
                ok[:bw] = ok[-bw:] = False
                ok[:, :bw] = ok[:, -bw:] = False
            if not ok.any():
                continue
            j, i = np.unravel_index(np.nanargmin(np.where(ok, slp, np.inf)), slp.shape)
            if best is None or slp[j, i] < best["hpa"]:
                best = dict(hpa=round(float(slp[j, i]), 1), time=t,
                            lat=round(float(ds.variables["XLAT"][ti][j, i]), 3),
                            lon=round(float(ds.variables["XLONG"][ti][j, i]), 3))
    return best


def _event(project_dir):
    path = os.path.join(project_dir, "event.yaml")
    if not os.path.isfile(path):
        return None
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    val = lambda x: x.get("value") if isinstance(x, dict) else x   # noqa: E731
    return dict(path=path, event=val(doc.get("event")), region=val(doc.get("region")),
                types=[val(t) for t in doc.get("types") or []],
                verify=[val(v) for v in doc.get("verify") or []])


def build_brief(post_dir, project_dir=None, project_doc=None, spinup_hours=6.0):
    """The brief.json document of a `wtune post recipe` output directory."""
    from wtune.post.focus import rain_focus
    from wtune.post.recipe import INDEX_FILE, load_catalog
    import wtune.project
    post_dir = os.path.abspath(post_dir)
    index_path = os.path.join(post_dir, INDEX_FILE)
    if not os.path.isfile(index_path):
        raise FileNotFoundError("%s: no %s -- run `wtune post recipe` (WTune >= this version) "
                                "into it first" % (post_dir, INDEX_FILE))
    with open(index_path) as fh:
        index = json.load(fh)
    kind = index.get("kind", "recipe")
    catalog = load_catalog()
    expected = []
    exp_path = os.path.join(post_dir, "expected_figures.csv")
    if os.path.isfile(exp_path):
        import csv
        with open(exp_path, newline="") as fh:
            expected = list(csv.DictReader(fh))
    case = index.get("case_info") or {}
    runs = index.get("runs") or []
    run_dir = runs[0]["path"] if runs else None
    domains = [d["domain"] for d in case.get("domains") or []] or [1]
    finest = max(domains)
    focus_box = (wtune.project.project_post_defaults(project_doc)["focus"]
                 if project_doc else None)

    drawn, missing = [], []
    for f in index.get("figures") or []:
        if f.get("missing"):
            missing.append(dict(id=f.get("id"), dataset=f.get("dataset"), reason=f.get("reason")))
        elif f.get("path"):
            q = ((catalog.get(f.get("id")) or {}).get("en") or {}).get("question")
            drawn.append(dict(id=f.get("id"), path=f["path"], domain=f.get("domain"),
                              valid_time=f.get("valid_time"), question=q))
    key = dict(domain=finest)
    if run_dir:
        try:
            focus = rain_focus(run_dir, finest, spinup_hours, box=focus_box)
            key["rain_focus"] = focus
        except Exception as e:                              # noqa: BLE001
            key["rain_focus_error"] = "%s: %s" % (type(e).__name__, e)
        try:
            key["slp_min"] = _slp_min(run_dir, finest, spinup_hours)
        except Exception as e:                              # noqa: BLE001
            key["slp_min_error"] = "%s: %s" % (type(e).__name__, e)
    storm = None
    if project_doc:
        storm_id, hurdat2 = wtune.project.project_storm(project_doc)
        if storm_id:
            storm = dict(id=storm_id, hurdat2=hurdat2)
    return dict(
        schema=SCHEMA, kind=kind, post_dir=post_dir, project=project_dir,
        case=dict(types=case.get("types"), evidence=case.get("evidence"),
                  run_length_h=case.get("run_length_h"), domains=case.get("domains"),
                  spinup_hours=spinup_hours, focus_box=focus_box),
        runs=runs, storm=storm,
        event=_event(project_dir) if project_dir else None,
        observations=_observations(project_dir, run_dir) if project_dir else None,
        key_numbers=key,
        figures=dict(drawn=drawn, missing=missing, after_the_run=expected),
        notes=[n.get("text") for n in index.get("notes") or []],
        agent_steps=PREVIEW_STEPS if kind == "preview" else AGENT_STEPS)


def write_brief(post_dir, **kw) -> str:
    doc = build_brief(post_dir, **kw)
    path = os.path.join(os.path.abspath(post_dir), BRIEF_FILE)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
    return path
