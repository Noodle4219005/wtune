"""post/custom.py -- `wtune post custom` (decision 0058 section 2): runs every analysis/*.py
script under a project's analysis/ directory, each handed a post/custom_api.Context, and folds
every figure/table it registers into one HTML report (reusing post/report.py's image/CSV
rendering helpers -- these outputs have no case-type/catalog caption, since they are the
user's own analysis, not one of the standard recipes).

A script's top-level `run(ctx)` is the required entry point (decision 0058: "腳本拿到一個簡單的
介面"); scripts are imported via importlib (not exec), so one script can `import` another
analysis/*.py helper through the normal package rules. A script that raises, or has no
run(ctx), gets one error entry in the report instead of aborting the whole `post custom` run.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
import traceback

from wtune.post import custom_api


def _load_script(path):
    name = "wtune_custom_%s" % os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_custom(project_dir, out_dir, run_dirs):
    """run_dirs: [(run_id, path), ...] (wtune_project.discover_runs's shape). Returns
    {"out_dir", "scripts": [{"script","ok","error","outputs","logs"}, ...], "report_path"}."""
    os.makedirs(out_dir, exist_ok=True)
    analysis_dir = os.path.join(project_dir, "analysis")
    scripts = sorted(glob.glob(os.path.join(analysis_dir, "*.py")))
    results = []
    for script_path in scripts:
        name = os.path.basename(script_path)
        script_out = os.path.join(out_dir, os.path.splitext(name)[0])
        os.makedirs(script_out, exist_ok=True)
        ctx = custom_api.Context(project_dir, name, script_out, run_dirs)
        entry = dict(script=name, ok=False, error=None, outputs=[], logs=[])
        try:
            module = _load_script(script_path)
            if not hasattr(module, "run"):
                raise AttributeError("%s has no top-level run(ctx) function" % name)
            module.run(ctx)
            entry["ok"] = True
        except Exception:
            entry["error"] = traceback.format_exc()
        entry["outputs"] = ctx.outputs
        entry["logs"] = ctx.logs
        results.append(entry)

    report_path = _build_custom_report(out_dir, results)
    return dict(out_dir=out_dir, scripts=results, report_path=report_path)


def _build_custom_report(out_dir, results):
    import html as html_mod
    from wtune.post.report import IMAGE_EXTS, _embed_image, _render_csv_table

    parts = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
             "<title>wtune post custom</title>",
             "<style>body{font-family:sans-serif;margin:2em;max-width:1000px}"
             "img{max-width:100%;border:1px solid #ccc;margin:0.5em 0}"
             ".error{background:#fff3f3;border:1px solid #e0a0a0;padding:0.75em;"
             "white-space:pre-wrap;font-size:0.85em}</style></head><body>"
             "<h1>wtune post custom</h1>"]
    if not results:
        parts.append("<p><em>no scripts found in analysis/</em></p>")
    for entry in results:
        parts.append("<h2>%s</h2>" % html_mod.escape(entry["script"]))
        for log in entry["logs"]:
            parts.append("<p><em>%s</em></p>" % html_mod.escape(log))
        if not entry["ok"]:
            parts.append("<div class='error'>%s</div>" % html_mod.escape(entry["error"]))
            continue
        if not entry["outputs"]:
            parts.append("<p><em>(no figures/tables registered)</em></p>")
        for kind, path, _script in entry["outputs"]:
            parts.append("<h3>%s</h3>" % html_mod.escape(os.path.basename(path)))
            if kind == "figure" or os.path.splitext(path)[1].lower() in IMAGE_EXTS:
                parts.append("<img src='%s' alt='%s'>" % (_embed_image(path),
                                                           html_mod.escape(os.path.basename(path))))
            else:
                parts.append(_render_csv_table(path))
    parts.append("</body></html>")
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    return out_path
