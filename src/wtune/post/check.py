"""`wtune post check POST_DIR` (decision 0078, stage 5): the program-checked half of an agent's
post-processing. The agent writes post/<id>/review.json (its verdict on every recipe figure)
and post/<id>/assessment.md (the layered judgement); this checks that

- review.json is wtune-post-review/1, every reviewed figure exists, a non-ok verdict says why,
  a `fix` names its action (ACTIONS: a `wtune project set` drawing option, an analysis/*.py
  script that redraws it, or a WTune defect report), and (warning) no drawn figure was left
  unreviewed;
- for `wtune post preview` (figures.json kind "preview"): review.json also carries the
  figure plan for after the run (`plan`: what to draw, how, from which script), and there is
  no assessment yet;
- assessment.md has the six sections, every paragraph cites at least one existing figure or
  CSV as a markdown link, and every decimal number in a paragraph that cites a CSV appears in
  one of those CSVs at the precision written (so numbers come from WTune or a script's output,
  never from reading a colour bar).

Exit 1 on any error; warnings are printed but pass.
"""
from __future__ import annotations

import csv
import json
import os
import re

REVIEW_FILE = "review.json"
REVIEW_SCHEMA = "wtune-post-review/1"
ASSESSMENT_FILE = "assessment.md"
VERDICTS = ("ok", "fix", "drop")
ACTIONS = ("project_set", "analysis", "wtune_defect")
# `wtune project set` flags a figure fix may use (drawing options, never machine or run ones)
DRAWING_FLAGS = ("--post-focus", "--cross-section", "--post-types", "--post-every")
# section -> heading words that satisfy it (English or Traditional Chinese)
SECTIONS = {
    "setup": ("setup", "configuration", "設定"),
    "large scale": ("large scale", "large-scale", "synoptic", "大尺度", "綜觀"),
    "target variables": ("target", "目標"),
    "scores": ("score", "verification", "分數", "校驗"),
    "mechanism": ("mechanism", "機制"),
    "conclusion and limits": ("conclusion", "limit", "結論", "限制"),
}
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")


def _resolve(target, post_dir, project_dir):
    if os.path.isabs(target):
        return target if os.path.exists(target) else None
    for base in (post_dir, project_dir):
        if base and os.path.exists(os.path.join(base, target)):
            return os.path.join(base, target)
    return None


def _check_action(where, action, project_dir, preview, errors, warnings):
    kind = (action or {}).get("kind")
    if kind not in ACTIONS:
        errors.append("%s: action.kind must be one of %s" % (where, "|".join(ACTIONS)))
    elif kind == "project_set":
        args = action.get("args") or []
        flags = [a.split("=")[0] for a in args if a.startswith("--")]
        bad = [f for f in flags if f not in DRAWING_FLAGS]
        if not flags or bad:
            errors.append("%s: project_set args must use %s (got %s)"
                          % (where, ", ".join(DRAWING_FLAGS), args))
        else:
            import wtune.project
            try:
                ns = wtune.project.build_parser().parse_args(["set"] + list(args))
                wtune.project.apply_drawing_options({}, ns.post_focus, ns.cross_section)
            except (SystemExit, wtune.project.ProjectError) as e:
                errors.append("%s: project_set args %s rejected: %s" % (where, args, e))
    elif kind == "analysis":
        script = action.get("script") or ""
        found = project_dir and os.path.isfile(os.path.join(project_dir, script))
        if not script.startswith("analysis/") or not script.endswith(".py"):
            errors.append("%s: analysis action needs script analysis/<name>.py" % where)
        elif not found:
            (warnings if preview else errors).append(
                "%s: %s does not exist yet in the project" % (where, script))
    elif not (action.get("text") or "").strip():
        errors.append("%s: wtune_defect needs text" % where)


def check_review(post_dir, errors, warnings, project_dir=None, preview=False):
    path = os.path.join(post_dir, REVIEW_FILE)
    if not os.path.isfile(path):
        errors.append("%s: missing" % REVIEW_FILE)
        return
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except ValueError as e:
        errors.append("%s: not JSON: %s" % (REVIEW_FILE, e))
        return
    if doc.get("schema") != REVIEW_SCHEMA:
        errors.append("%s: schema must be %r" % (REVIEW_FILE, REVIEW_SCHEMA))
    reviewed = set()
    for k, e in enumerate(doc.get("figures") or []):
        where = "%s figures[%d]" % (REVIEW_FILE, k)
        p = e.get("path")
        if not p or not os.path.isfile(os.path.join(post_dir, p)):
            errors.append("%s: figure %r does not exist under %s" % (where, p, post_dir))
        reviewed.add(os.path.normpath(p or ""))
        if e.get("verdict") not in VERDICTS:
            errors.append("%s: verdict must be one of %s" % (where, "|".join(VERDICTS)))
        elif e.get("verdict") != "ok" and not e.get("issues"):
            errors.append("%s: verdict %s without issues" % (where, e["verdict"]))
        if e.get("verdict") == "fix" and not e.get("action"):
            errors.append("%s: verdict fix without an action" % where)
        if e.get("action"):
            _check_action(where, e["action"], project_dir, preview, errors, warnings)
    if preview:
        plan = doc.get("plan") or []
        if not plan:
            errors.append("%s: a preview review needs `plan`, the figures to draw after the "
                          "run" % REVIEW_FILE)
        for k, p in enumerate(plan):
            where = "%s plan[%d]" % (REVIEW_FILE, k)
            if not (p.get("figure") and p.get("how")):
                errors.append("%s: needs figure and how" % where)
            if p.get("action"):
                _check_action(where, p["action"], project_dir, True, errors, warnings)
    index = os.path.join(post_dir, "figures.json")
    if os.path.isfile(index):
        with open(index) as fh:
            drawn = [f["path"] for f in json.load(fh).get("figures") or []
                     if not f.get("missing") and f.get("path")
                     and f["path"].lower().endswith((".png", ".svg", ".jpg"))]
        left = [p for p in drawn if os.path.normpath(p) not in reviewed]
        if left:
            warnings.append("%s: %d drawn figure(s) not reviewed, e.g. %s"
                            % (REVIEW_FILE, len(left), left[0]))


def _paragraphs(text):
    """[(first line number, text)] of the non-heading blocks between blank lines."""
    out, block, start = [], [], 0
    for n, line in enumerate(text.splitlines() + [""], 1):
        if line.strip() and not line.lstrip().startswith("#"):
            if not block:
                start = n
            block.append(line)
        elif block:
            out.append((start, "\n".join(block)))
            block = []
    return out


def _csv_numbers(path):
    vals = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            for cell in row:
                try:
                    vals.append(float(cell))
                except ValueError:
                    pass
    return vals


def _in_csv(token, numbers):
    decimals = len(token.split(".")[1])
    v = float(token)
    return any(round(x, decimals) == round(v, decimals) for x in numbers)


def check_assessment(post_dir, project_dir, errors, warnings):
    path = os.path.join(post_dir, ASSESSMENT_FILE)
    if not os.path.isfile(path):
        errors.append("%s: missing" % ASSESSMENT_FILE)
        return
    with open(path) as fh:
        text = fh.read()
    headings = [ln.lstrip("#").strip().lower() for ln in text.splitlines()
                if ln.startswith("#")]
    for name, words in SECTIONS.items():
        if not any(w in h for h in headings for w in words):
            errors.append("%s: no section for %s (a heading with one of: %s)"
                          % (ASSESSMENT_FILE, name, ", ".join(words)))
    for line, para in _paragraphs(text):
        targets = _LINK.findall(para)
        if not targets:
            errors.append("%s:%d: paragraph cites no figure or CSV" % (ASSESSMENT_FILE, line))
            continue
        resolved = []
        for t in targets:
            r = _resolve(t, post_dir, project_dir)
            if r is None:
                errors.append("%s:%d: %s does not exist" % (ASSESSMENT_FILE, line, t))
            else:
                resolved.append(r)
        csvs = [r for r in resolved if r.lower().endswith(".csv")]
        if not csvs:
            continue
        numbers = [x for c in csvs for x in _csv_numbers(c)]
        for token in _DECIMAL.findall(_LINK.sub(" ", para)):
            if not _in_csv(token, numbers):
                errors.append("%s:%d: %s is not in %s" % (
                    ASSESSMENT_FILE, line, token,
                    ", ".join(os.path.basename(c) for c in csvs)))


def index_kind(post_dir):
    """figures.json's kind ("recipe" or "preview"), None without one."""
    path = os.path.join(post_dir, "figures.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh).get("kind", "recipe")


def check(post_dir, project_dir=None) -> tuple[list, list]:
    """(errors, warnings) of an agent's post/<id>/ (or post/preview/) outputs."""
    post_dir = os.path.abspath(post_dir)
    errors, warnings = [], []
    preview = index_kind(post_dir) == "preview"
    check_review(post_dir, errors, warnings, project_dir=project_dir, preview=preview)
    if not preview:
        check_assessment(post_dir, project_dir, errors, warnings)
    return errors, warnings
