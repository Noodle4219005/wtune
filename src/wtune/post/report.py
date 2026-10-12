"""post/report.py -- `wtune post report OUT_DIR [--title]`: collect the PNG/SVG figures
and CSV tables OUT_DIR's other `wtune post` subcommands wrote into one self-contained HTML
report (images base64-embedded, no external resources -- opens offline, from a mail
attachment, anywhere). Stdlib only (base64, csv, glob, html).

build_recipe_report() (decision 0057) is the richer sibling `wtune post recipe` uses: it groups
post/recipe.py's figure records by detected case type, embeds each figure's zh/en caption from
post/recipes/catalog.json, stamps run/domain/valid time/spin-up, and lists every missing-
observation entry explicitly instead of just scanning a directory for images.
"""
from __future__ import annotations

import base64
import csv
import glob
import html
import os

IMAGE_EXTS = (".png", ".svg", ".jpg", ".jpeg")
_MIME = {".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _embed_image(path):
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "application/octet-stream")
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime, data)


def _render_csv_table(path, max_rows=200):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "<p><em>(empty)</em></p>"
    truncated = len(rows) - 1 > max_rows
    body_rows = rows[1:1 + max_rows]
    out = ["<table border='1' cellspacing='0' cellpadding='4'>", "<tr>"]
    out += ["<th>%s</th>" % html.escape(c) for c in rows[0]]
    out.append("</tr>")
    for row in body_rows:
        out.append("<tr>" + "".join("<td>%s</td>" % html.escape(c) for c in row) + "</tr>")
    out.append("</table>")
    if truncated:
        out.append("<p><em>... %d more row(s) not shown</em></p>" % (len(rows) - 1 - max_rows))
    return "\n".join(out)


def build_report(out_dir, title=None):
    """Writes out_dir/report.html (embedding every image/csv found directly in out_dir,
    sorted by name) and returns its path."""
    title = title or ("wtune post report: %s" % os.path.basename(os.path.normpath(out_dir)))

    images = sorted(p for p in glob.glob(os.path.join(out_dir, "*"))
                    if os.path.splitext(p)[1].lower() in IMAGE_EXTS)
    csvs = sorted(glob.glob(os.path.join(out_dir, "*.csv")))

    parts = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
             "<title>%s</title>" % html.escape(title),
             "<style>body{font-family:sans-serif;margin:2em} "
             "img{max-width:100%;border:1px solid #ccc;margin-bottom:1em} "
             "table{border-collapse:collapse;margin-bottom:2em;font-size:0.9em} "
             "th,td{border:1px solid #ccc;padding:4px 8px}</style>",
             "</head><body>", "<h1>%s</h1>" % html.escape(title)]

    if images:
        parts.append("<h2>Figures</h2>")
        for path in images:
            parts.append("<h3>%s</h3>" % html.escape(os.path.basename(path)))
            parts.append("<img src='%s' alt='%s'>" % (_embed_image(path),
                                                       html.escape(os.path.basename(path))))
    if csvs:
        parts.append("<h2>Tables</h2>")
        for path in csvs:
            parts.append("<h3>%s</h3>" % html.escape(os.path.basename(path)))
            parts.append(_render_csv_table(path))
    if not images and not csvs:
        parts.append("<p><em>no PNG/SVG figures or CSV tables found in %s</em></p>"
                     % html.escape(out_dir))

    parts.append("</body></html>")
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    return out_path


# --------------------------------------------------------------------------------------
# `wtune post recipe` report (decision 0057): case-type header + evidence, per-type figure
# groups with zh/en captions, explicit missing-observation entries.
# --------------------------------------------------------------------------------------

_TYPE_ID_PREFIX = dict(common="common_", tc="tc_", convection="conv_", precip="precip_",
                       synoptic="syn_", terrain="terrain_", moisture="moisture_", pbl="pbl_",
                       chem="chem_", climate="climate_", ensemble="ens_")


def _figures_for_type(figures, type_name):
    """The type's figures by id prefix, plus post.recipe's whole-set missing entry
    ("<type>_figures", e.g. convection_figures, which does not start with conv_)."""
    prefix = _TYPE_ID_PREFIX.get(type_name, type_name + "_")
    return [f for f in figures
            if f["id"].startswith(prefix) or f["id"] == type_name + "_figures"]


def _in_time_series(f):
    fid = f.get("id", "")
    return fid == "timeseries" or fid.startswith(("timeseries_", "chg_", "anim_"))


def _caption_block(catalog, fig_id):
    entry = catalog.get(fig_id)
    if not entry:
        return "<p><em>(no caption catalog entry for %s)</em></p>" % html.escape(fig_id)
    parts = []
    for lang in ("zh", "en"):
        c = entry.get(lang, {})
        parts.append(
            "<p class='caption-%s'><strong>%s</strong> %s<br>"
            "<strong>%s</strong> %s<br><strong>%s</strong> %s</p>" % (
                lang,
                "問題:" if lang == "zh" else "Question:", html.escape(c.get("question", "")),
                "判讀:" if lang == "zh" else "How to read:", html.escape(c.get("how_to_read", "")),
                "常見誤讀:" if lang == "zh" else "Common misreadings:",
                html.escape(c.get("common_misreadings", ""))))
    return "\n".join(parts)


def _source_lines(f):
    """HTML for a figure's data sources (`obs_sources` from post/sources.py source_tags) and the
    MET tool/window/mask/note that produced its numbers."""
    parts = []
    for src in f.get("obs_sources") or []:
        parts.append("<p class='stamp'>資料來源 / data: <strong>%s</strong> -- %s; <a href='%s'>%s</a>; "
                     "license: %s; fetched: %s<br>citation: %s</p>" % (
                         html.escape(src.get("dataset", "")), html.escape(src.get("provider", "")),
                         html.escape(_first_url(src.get("url", ""))), html.escape(src.get("url", "")),
                         html.escape(src.get("license", "")), html.escape(src.get("fetched_utc", "")),
                         html.escape(src.get("citation", ""))))
    extra = [("tool", "MET tool"), ("window", "window"), ("mask", "mask")]
    bits = ["%s: %s" % (label, f[k]) for k, label in extra if f.get(k)]
    if bits:
        parts.append("<p class='stamp'>%s</p>" % html.escape("; ".join(bits)))
    if f.get("note"):
        parts.append("<p class='note'>%s</p>" % html.escape(str(f["note"])))
    return "\n".join(parts)


def _first_url(text):
    import re
    m = re.search(r"https?://[^\s;,)]+", text or "")
    return m.group(0) if m else "#"


def _notes_box(notes):
    """Warning box for summary["notes"] = [{"source", "level", "text"}]. Notes are caveats about
    how to read the report, not missing observations."""
    if not notes:
        return ""
    items = []
    for n in notes:
        items.append("<li><strong>[%s]</strong> %s <span class='stamp'>(%s)</span></li>" % (
            html.escape(str(n.get("level", "note"))), html.escape(str(n.get("text", ""))),
            html.escape(str(n.get("source", "")))))
    return "<div class='warnbox'><strong>Notes</strong><ul>%s</ul></div>" % "".join(items)


def _sources_section(figures, out_dir):
    """"資料來源與授權": every observation/reanalysis dataset the report's figures used, with
    link, license, citation and (when fetched by wtune) the UTC fetch time."""
    from wtune.post import sources
    names = sources.used_sources(figures)
    if not names:
        return ""
    try:
        from wtune.project import find_project
        project_dir = find_project(out_dir)
    except Exception:
        project_dir = None
    rows = ["<h2>資料來源與授權 (Data sources and licenses)</h2>",
            "<table><tr><th>dataset</th><th>provider</th><th>link</th><th>license</th>"
            "<th>citation</th><th>fetched (UTC)</th></tr>"]
    for r in sources.dataset_records(project_dir, names):
        rows.append("<tr><td>%s</td><td>%s</td><td><a href='%s'>%s</a></td><td>%s</td><td>%s</td>"
                    "<td>%s</td></tr>" % tuple(html.escape(x) if i != 2 else x for i, x in enumerate((
                        r["dataset"], r["provider"], html.escape(_first_url(r["url"])), r["url"],
                        r["license"], r["citation"], r["fetched_utc"]))))
    rows.append("</table>")
    return "\n".join(rows)


def _human_size(n):
    if n is None:
        return "size unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024.0


def _stamp(f, spinup_hours):
    bits = []
    if f.get("run"):
        bits.append("run=%s" % f["run"])
    if f.get("domain"):
        bits.append("domain=%s" % f["domain"])
    if f.get("valid_time"):
        bits.append("valid=%s" % f["valid_time"])
    bits.append("spin-up excluded: %g h" % spinup_hours)
    return "; ".join(bits)


def _build_time_series_section(out_dir, figures, catalog, spinup_hours):
    """`wtune post recipe --every/--times` section (decision 0058 P3): groups
    post/timeseries.py's and post/changemap.py's figure records by forecast tag
    (fNNN_<validZ>Z), with prev/next navigation between times, plus an Animations
    sub-section (post/anim.py's MP4/GIF records, referenced by relative path next to
    report.html -- never base64-embedded, decision 0058 update: "報告裡可以直接播放")."""
    anim_figs = [f for f in figures if f.get("id", "").startswith("anim_")]
    ts_and_chg = [f for f in figures
                  if _in_time_series(f) and not f["id"].startswith("anim_")]
    if not ts_and_chg and not anim_figs:
        return ""

    groups = {}
    for f in ts_and_chg:
        tag = f.get("forecast_tag") or f.get("valid_time") or "unknown"
        groups.setdefault(tag, []).append(f)
    tags = sorted(groups)

    parts = ["<h2>Time series</h2>"]
    if tags:
        parts.append("<p class='stamp'>navigate: " +
                     " | ".join("<a href='#ts-%s'>%s</a>" % (html.escape(t), html.escape(t))
                               for t in tags) + "</p>")
    for idx, tag in enumerate(tags):
        parts.append("<h3 id='ts-%s'>%s</h3>" % (html.escape(tag), html.escape(tag)))
        nav = []
        if idx > 0:
            nav.append("<a href='#ts-%s'>&laquo; prev</a>" % html.escape(tags[idx - 1]))
        if idx < len(tags) - 1:
            nav.append("<a href='#ts-%s'>next &raquo;</a>" % html.escape(tags[idx + 1]))
        if nav:
            parts.append("<p class='stamp'>" + " | ".join(nav) + "</p>")
        for f in groups[tag]:
            if f.get("missing"):
                parts.append(
                    "<div class='missing'><strong>missing: %s</strong> (dataset: %s)<br>"
                    "reason: %s</div>" % (html.escape(f["id"]), html.escape(f.get("dataset", "")),
                                          html.escape(f.get("reason", ""))))
                continue
            parts.append("<h4>%s</h4>" % html.escape(f["id"]))
            parts.append("<img src='%s' alt='%s'>" % (_embed_image(f["path"]),
                                                       html.escape(f["id"])))
            parts.append("<p class='stamp'>%s</p>" % html.escape(_stamp(f, spinup_hours)))
            parts.append(_caption_block(catalog, f["id"]))

    if anim_figs:
        parts.append("<h3>Animations</h3>")
        for f in anim_figs:
            if f.get("missing"):
                parts.append("<div class='missing'><strong>missing: %s</strong><br>"
                             "reason: %s</div>" % (html.escape(f["id"]), html.escape(f.get("reason", ""))))
                continue
            rel = os.path.relpath(f["path"], out_dir)
            parts.append("<h4>%s</h4>" % html.escape(f["id"]))
            ext = os.path.splitext(f["path"])[1].lower()
            if ext == ".gif":
                parts.append("<img src='%s' alt='%s'>" % (html.escape(rel), html.escape(f["id"])))
            else:
                parts.append("<video controls style='max-width:100%%' src='%s'></video>"
                             % html.escape(rel))
    return "\n".join(parts)


def build_recipe_report(out_dir, case_info, figures, run_dirs, labels, spinup_hours, catalog,
                        title=None, notes=None, advice_html=None):
    """Writes out_dir/report.html for `wtune post recipe`. figures: post/recipe.py's fig()/
    missing() records. Returns the report path."""
    title = title or ("wtune post recipe: %s" % ", ".join(labels))
    parts = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
             "<title>%s</title>" % html.escape(title),
             "<style>body{font-family:sans-serif;margin:2em;max-width:1100px}"
             "img{max-width:100%;border:1px solid #ccc;margin:0.5em 0}"
             "table{border-collapse:collapse;margin-bottom:1em;font-size:0.9em}"
             "th,td{border:1px solid #ccc;padding:4px 8px}"
             ".missing{background:#fff3f3;border:1px solid #e0a0a0;padding:0.75em;margin:0.5em 0}"
             ".stamp{color:#555;font-size:0.85em}"
             ".caption-en{color:#333}.caption-zh{color:#333}"
             ".warnbox{background:#fff8e1;border:1px solid #e0b040;padding:0.75em;margin:0.75em 0}"
             ".note{color:#7a4b00;font-size:0.9em}"
             "h2{border-bottom:2px solid #333;padding-bottom:0.2em}</style>",
             "</head><body>", "<h1>%s</h1>" % html.escape(title)]
    parts.append(_notes_box(notes if notes is not None else case_info.get("notes")))
    if advice_html:      # decision 0061: verification-triggered, advice only (wtune.advise.render_html)
        parts.append(advice_html)

    parts.append("<h2>Run(s)</h2><ul>")
    for run_dir, label in zip(run_dirs, labels):
        parts.append("<li>%s: %s</li>" % (html.escape(label), html.escape(os.path.abspath(run_dir))))
    parts.append("</ul>")

    parts.append("<h2>Detected case types</h2>")
    parts.append("<p><strong>%s</strong></p>" % html.escape(", ".join(case_info["types"])))
    parts.append("<table><tr><th>type</th><th>detected</th><th>evidence</th></tr>")
    for t in sorted(case_info["evidence"]):
        detected = t in case_info["types"]
        parts.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            html.escape(t), "yes" if detected else "no", html.escape(case_info["evidence"][t])))
    parts.append("</table>")
    parts.append("<p class='stamp'>spin-up excluded: %g h; run length: %.1f h; d01 extent: "
                 "%.0f x %.0f km</p>" % (spinup_hours, case_info["run_length_h"],
                                        case_info["d01_extent_km"][0], case_info["d01_extent_km"][1]))

    data_ids_shown = set()
    shown = set()       # id() of every entry some section rendered

    def section(t, type_figs):
        if not type_figs:
            return
        shown.update(id(f) for f in type_figs)
        parts.append("<h2>%s</h2>" % html.escape(t))
        for f in type_figs:
            if f.get("missing"):
                parts.append(
                    "<div class='missing'><strong>missing: %s</strong> (dataset: %s)<br>"
                    "reason: %s</div>" % (html.escape(f["id"]), html.escape(f.get("dataset", "")),
                                          html.escape(f.get("reason", ""))))
                continue
            if f.get("data_file"):
                # a data product (e.g. post/recipes/cf3d.py's NetCDF), one per domain x time:
                # one heading + caption per id, then a compact line per file
                if f["id"] not in data_ids_shown:
                    data_ids_shown.add(f["id"])
                    parts.append("<h3>%s</h3>" % html.escape(f["id"]))
                    parts.append(_caption_block(catalog, f["id"]))
                rel = os.path.relpath(f["path"], out_dir)
                links = "".join(" | <a href='%s'>%s</a>" % (
                    html.escape(os.path.relpath(p, out_dir)), html.escape(os.path.basename(p)))
                    for p in f.get("links") or [])
                parts.append("<p class='datafile'>data file: <a href='%s'>%s</a> (%s)%s<br>"
                             "<span class='stamp'>%s</span></p>" % (
                                 html.escape(rel), html.escape(rel),
                                 _human_size(f.get("size_bytes")), links,
                                 html.escape(_stamp(f, spinup_hours))))
                parts.append(_source_lines(f))
                continue
            parts.append("<h3>%s</h3>" % html.escape(f["id"]))
            ext = os.path.splitext(f["path"])[1].lower()
            if ext in IMAGE_EXTS:
                parts.append("<img src='%s' alt='%s'>" % (_embed_image(f["path"]),
                                                           html.escape(f["id"])))
            else:
                parts.append("<p><em>%s (not an image -- open at %s)</em></p>"
                             % (html.escape(os.path.basename(f["path"])), html.escape(f["path"])))
            parts.append("<p class='stamp'>%s</p>" % html.escape(_stamp(f, spinup_hours)))
            parts.append(_source_lines(f))
            parts.append(_caption_block(catalog, f["id"]))

    for t in case_info["types"]:
        section(t, _figures_for_type(figures, t))
    parts.append(_build_time_series_section(out_dir, figures, catalog, spinup_hours))
    # catch-all: an entry no section claimed is listed, never silently dropped
    section("other", [f for f in figures if id(f) not in shown and not _in_time_series(f)])
    parts.append(_sources_section(figures, out_dir))

    parts.append("</body></html>")
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    return out_path
