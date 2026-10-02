"""post/recipes/paraview3d.py -- wrapper that finds pvbatch (WTUNE_PARAVIEW_BIN, default
envs/paraview/.pixi/envs/default/bin, per bin/install-postproc.sh) and runs it against
post/paraview/render_volume.py: a VOLUME rendering of total condensate (every hydrometeor,
post/condensate.py) over terrain, for the tc and convection figure sets (decision 0057 item 3).
post/xwrf_convert.build_volume() grids one wrfout frame onto a regular km grid (.vti) first.

2026-10-02: replaces the QCLOUD+QICE >= 0.1 g/kg isosurface (post/paraview/render_cloud.py,
kept on disk but no longer called): that field was 10% of the condensate mass in SC26-P2 and the
whole-domain, x30, far-away view showed the hurricane as one small opaque tube with no eye,
eyewall or rainbands. render_cloud() keeps its signature and now delegates to
render_condensate_volume() with the whole-domain view.

If pvbatch is not found, figures() returns a clear missing-observation-style record instead of
failing the whole report -- same contract as a missing MET tool in post/tc.py's run_tc_report().
"""
from __future__ import annotations

import json
import os
import shutil

from post import tools

_PV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paraview")
_SCRIPT = os.path.join(_PV_DIR, "render_volume.py")


class ParaViewError(RuntimeError):
    pass


def find_pvbatch():
    env_bin = os.environ.get("WTUNE_PARAVIEW_BIN")
    bin_dirs = [env_bin] if env_bin else []
    bin_dirs.append(os.path.join(tools.REPO_ROOT, "envs", "paraview", ".pixi", "envs", "default", "bin"))
    path, searched = tools.locate("pvbatch", bin_dirs + [None, tools.postproc_view_bin()])
    if path:
        return path
    raise tools.ToolNotFoundError(
        "pvbatch not found (searched: %s): install ParaView (bin/install-postproc.sh) and export "
        "WTUNE_PARAVIEW_BIN=<repo>/envs/paraview/.pixi/envs/default/bin, or put pvbatch on PATH."
        % ", ".join(searched))


VERTICAL_EXAGGERATION = 30.0   # legacy isosurface value; the volume picks its own (default_vexag)
ISOVALUE = 1e-4                # legacy, ignored by render_cloud() since 2026-10-02

# storm-centred view: from the south-south-west, 25 deg up; whole domain: from the south
STORM_BEARING_DEG = 202.5
STORM_ELEVATION_DEG = 25.0
STORM_FOCAL_Z_KM = 5.0
STORM_ZOOM = 1.45             # scc 2026-10-02: 1.25 left margins, 1.7 clips the box
DOMAIN_BEARING_DEG = 180.0
DOMAIN_ELEVATION_DEG = 35.0
DOMAIN_FOCAL_Z_KM = 3.0
DOMAIN_ZOOM = 1.15


def build_title(run_label, domain, time_str, wrfout_path=None):
    """First title line: run, domain, init/valid/lead (post.conventions.path_time_label when the
    wrfout path is known, else the valid time alone). render_condensate_volume() appends the
    quantity / vertical-exaggeration line."""
    from post import conventions
    when = (conventions.path_time_label(wrfout_path, time_str) if wrfout_path
            else conventions.time_label(None, time_str))
    return "%s  d%02d  %s" % (run_label, domain, when)


_WRF_TIME_RE = r"\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}(?::\d{2})?"


def _frame_valid_time(wrfout_path, timeidx=0):
    """The frame's own valid time (Times record `timeidx`), or None."""
    try:
        import netCDF4
        with netCDF4.Dataset(wrfout_path) as ds:
            return b"".join(ds.variables["Times"][timeidx].tolist()).decode().strip() or None
    except Exception:   # noqa: BLE001
        return None


def title_with_time_label(title, wrfout_path, timeidx=0):
    """Make the caller's first title line carry the 2-D figures' 'init ... valid ... (Fhhh)'
    label (post.conventions.path_time_label): a raw WRF time in it ('2019-01-31_00:00:00') is
    replaced; a title without any time gets the frame's label appended; a title that already
    says 'valid' is left alone."""
    import re
    from post import conventions
    if "valid" in title:
        return title
    m = re.search(_WRF_TIME_RE, title)
    if m:
        return title[:m.start()] + conventions.path_time_label(wrfout_path, m.group()) \
            + title[m.end():]
    valid = _frame_valid_time(wrfout_path, timeidx)
    if not valid:
        return title
    label = conventions.path_time_label(wrfout_path, valid)
    return (title.rstrip() + "  " + label) if title.strip() else label


def volume_title_line(vexag, ztop_km=None):
    top = ", volume top %g km" % ztop_km if ztop_km else ""
    return ("Total condensate (all hydrometeors) volume rendering, opacity from 0.01 g/kg%s, "
            "vertical exaggeration \u00d7%g" % (top, vexag))


def opacity_unit_km(dx_eff_km):
    """ScalarOpacityUnitDistance: dx_eff / 4, floor 0.5 km. scc sweep 2026-10-02 (SC26-P2,
    dx 12 km): 3 km shows the cloud shield and spiral anvil with an opaque core; the earlier
    width/60 rule (24 km) left the far half nearly invisible."""
    return max(0.5, dx_eff_km / 4.0)


def ruler_placement(info, bearing_deg):
    """(x, y, lx, ly, tick_km) for the height ruler: at the box corner furthest to the
    screen-left of a camera at bearing_deg (ties -> the far one), ticks pointing screen-left,
    2.5% of the box width long."""
    import math
    x0, x1, y0, y1 = info["extent_km"]
    b = math.radians(bearing_deg)
    fx, fy = -math.sin(b), -math.cos(b)          # camera -> centre, horizontal
    lx, ly = -fy, fx                             # 90 deg counter-clockwise: screen-left
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    corner = max(((x, y) for x in (x0, x1) for y in (y0, y1)),
                 key=lambda c: (round((c[0] - cx) * lx + (c[1] - cy) * ly, 6),
                                (c[0] - cx) * fx + (c[1] - cy) * fy))
    return (corner[0], corner[1], round(lx, 12), round(ly, 12),
            0.025 * info["width_km"])


def view_params(info, storm, cutaway=False):
    """Camera and opacity settings for the render script, from build_volume()'s info."""
    vexag = info["vexag"]
    dx_eff = info["dx_km"] * info["stride"]
    p = dict(bearing=STORM_BEARING_DEG if storm else DOMAIN_BEARING_DEG,
             elevation=STORM_ELEVATION_DEG if storm else DOMAIN_ELEVATION_DEG,
             focal_z=min(STORM_FOCAL_Z_KM if storm else DOMAIN_FOCAL_Z_KM,
                         0.4 * info.get("ztop_km", 18.0)) * vexag,
             zoom=STORM_ZOOM if storm else DOMAIN_ZOOM,
             opacity_unit=opacity_unit_km(dx_eff))
    if cutaway:
        p["vt_max"] = info.get("vt_max") or 60.0
    return p


def dx_note(info):
    dx = info["dx_km"]
    note = "dx = %g km" % round(dx, 2)
    if info["stride"] > 1:
        note += " (volume every %d points, %g km)" % (info["stride"], round(dx * info["stride"], 2))
    return note


def legend_note(cutaway):
    n = ("grey: total condensate, transparent below 0.01 g/kg, blue-grey >= 1 g/kg;  "
         "base: terrain, blue = sea, black = coastline and lake shores")
    if cutaway:
        n += ";  near half cut away"
    return n


def build_command(pvbatch, info, out_png, title, params, cutaway, save_state=None):
    cx, cy = info["center_km"]
    cmd = [pvbatch, "--force-offscreen-rendering", _SCRIPT, info["vti"], info["terrain"], out_png,
           "--vexag", "%g" % info["vexag"], "--center", "%.3f" % cx, "%.3f" % cy,
           "--bearing", "%g" % params["bearing"], "--elevation", "%g" % params["elevation"],
           "--focal-z", "%.3f" % params["focal_z"], "--zoom", "%g" % params["zoom"],
           "--opacity-unit", "%.3f" % params["opacity_unit"],
           "--ztop", "%g" % info["ztop_km"],
           "--ruler=" + ",".join("%.6f" % v for v in ruler_placement(info, params["bearing"])),
           "--title", title, "--note", legend_note(cutaway), "--dx-note", dx_note(info)]
    if cutaway:
        cmd += ["--cutaway", "--vt-max", "%g" % params["vt_max"]]
    if info.get("coast"):
        cmd += ["--coast", info["coast"]]
    if save_state:
        cmd += ["--save-state", save_state]
    return cmd


def _bdy_width(wrfout_path):
    from post import caseinfo
    try:
        return caseinfo.spec_bdy_width(os.path.dirname(os.path.abspath(wrfout_path)))
    except Exception:   # noqa: BLE001 -- no namelist next to the frame: WRF's default
        return caseinfo.DEFAULT_SPEC_BDY_WIDTH


def render_condensate_volume(wrfout_path, out_png, work_dir, *, center_latlon=None, box_km=None,
                             cutaway=False, title="", vexag=None, save_state=None, timeidx=0):
    """Volume rendering of total condensate (all hydrometeors) from one wrfout frame -> out_png.
    timeidx: the valid time's index inside a multi-frame wrfout (post.extract.frame_ref).

    center_latlon=(lat, lon): storm-centred box (box_km, default 600 km) seen from the SSW at
    25 deg; otherwise the whole domain interior (spec_bdy_width zone excluded) seen from the
    south. cutaway (needs center_latlon): the camera-side half is removed and the cloudy part
    of the cut plane is coloured by tangential wind. vexag=None picks
    xwrf_convert.default_vexag (rendered column ~1/5 of the width, cap 30; a storm box at least
    box/75, 600 km -> 8); the volume top adapts to the cloud tops (4-18 km). title: the
    caller's first line (build_title); a raw WRF time in it becomes the init/valid/lead label
    (title_with_time_label) and the quantity / exaggeration line is appended.

    Raises tools.ToolNotFoundError when pvbatch is missing (ImportError when netCDF4 is),
    ParaViewError when pvbatch produced no PNG, ValueError when the centre is outside the
    domain. Runs pvbatch with check=False and judges success by whether out_png exists, not by
    the exit code: on scc (offscreen) pvbatch writes a correct PNG and then segfaults during its
    own teardown (exit 139 AFTER SaveScreenshot) -- that must not discard a real render."""
    pvbatch = find_pvbatch()
    from post import xwrf_convert
    if cutaway and center_latlon is None:
        raise ValueError("cutaway needs center_latlon")
    os.makedirs(work_dir, exist_ok=True)
    # no ':' in the converted file names: ParaView's file-series readers trip on them
    stem = os.path.splitext(os.path.basename(wrfout_path))[0].replace(":", "")
    if timeidx:
        stem += "_t%d" % timeidx           # several valid times of one multi-frame file
    prefix = os.path.join(work_dir, stem)
    storm = center_latlon is not None
    info = xwrf_convert.build_volume(
        wrfout_path, prefix, center_latlon=center_latlon, box_km=box_km, cutaway=cutaway,
        bearing_deg=STORM_BEARING_DEG if storm else DOMAIN_BEARING_DEG, vexag=vexag,
        bdy_width=_bdy_width(wrfout_path), timeidx=timeidx)
    params = view_params(info, storm, cutaway)
    title = title_with_time_label(title, wrfout_path, timeidx)
    full_title = (((title + "\n") if title else "")
                  + volume_title_line(info["vexag"], info.get("ztop_km")))
    with open(prefix + "_volume.json", "w") as f:
        json.dump(dict(info, view=params, center_latlon=center_latlon, box_km=box_km,
                       cutaway=cutaway, title=full_title), f, indent=1)
    if os.path.exists(out_png):
        os.remove(out_png)   # a stale PNG must not pass the exists() check below
    cmd = build_command(pvbatch, info, out_png, full_title, params, cutaway, save_state)
    rc, out, err = tools.run(cmd, check=False)
    if not os.path.exists(out_png):
        raise ParaViewError("pvbatch (exit %d) did not produce %s:\n%s" % (
            rc, out_png, (err or out).strip()[-2000:]))
    return out_png


def render_cloud(wrfout_path, out_png, work_dir, isovalue=ISOVALUE, title="", vexag=None,
                 save_state=None, timeidx=0):
    """Whole-domain 3-D cloud figure. Since 2026-10-02 this is render_condensate_volume() with
    the whole-domain view: total condensate volume, not a QCLOUD+QICE isosurface. isovalue is
    accepted for old callers and ignored; vexag=None picks the volume's own default."""
    return render_condensate_volume(wrfout_path, out_png, work_dir, title=title, vexag=vexag,
                                    save_state=save_state, timeidx=timeidx)


_README = """\
WTune 3-D 雲場 ParaView 場景（%(catalog_id)s）

開啟方式（ParaView >= 5.11）：
  File -> Load State -> 選 %(catalog_id)s.pvsm
  -> 在 "Load State Options" 選 "Search files under specified directory"，指向本資料夾。
  （.pvsm 內的資料路徑是產生當下的絕對路徑；資料夾搬移或複製後必須用此選項重新對應。）

檔案：
%(files)s
體積渲染（volume rendering）：總凝結物（所有 hydrometeor 混合比之和）log10(g/kg)，
0.01 g/kg 以下透明；垂直放大倍率：x%(vexag)s（體積、地形與海岸線 z 軸皆乘此倍率，僅為視覺用）
"""


def write_readme(pv_dir, catalog_id):
    names = sorted(os.listdir(pv_dir))
    desc = {".pvsm": "ParaView 狀態檔（場景、相機、著色）"}
    vexag = "?"
    lines = []
    for n in names:
        if n == "README.txt":
            continue
        if n.endswith("_condensate.vti"):
            d = "總凝結物 3-D 規則網格（VTK ImageData，km；log10_q、q_gkg、切面用 vt）"
        elif n.endswith("_terrain.vtk"):
            d = "地形曲面（legacy VTK structured grid）"
        elif n.endswith("_coast.vtk"):
            d = "海岸線與湖岸線折線（legacy VTK，Natural Earth 50m coastline + lakes）"
        elif n.endswith("_volume.json"):
            d = "網格與相機參數（中心、範圍、dx、垂直放大倍率）"
            try:
                with open(os.path.join(pv_dir, n)) as f:
                    vexag = "%g" % json.load(f)["vexag"]
            except Exception:   # noqa: BLE001
                pass
        else:
            d = desc.get(os.path.splitext(n)[1], "")
        lines.append("  %s  %s" % (n, d))
    with open(os.path.join(pv_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(_README % dict(catalog_id=catalog_id, files="\n".join(lines) + "\n",
                               vexag=vexag))


def figures(catalog_id, wrfout_path, out_dir, work_dir, run_label, domain, time_str, timeidx=0):
    """work_dir is kept for call-site compatibility but unused: the converted files are written
    straight into <out_dir>/<catalog_id>_paraview/ (no copy), next to the .pvsm and README.txt,
    and the .pvsm is also written to <out_dir>/<catalog_id>.pvsm."""
    from post.recipe import fig, missing
    out_png = os.path.join(out_dir, "%s.png" % catalog_id)
    pv_dir = os.path.join(out_dir, "%s_paraview" % catalog_id)
    pvsm = os.path.join(pv_dir, "%s.pvsm" % catalog_id)
    try:
        render_cloud(wrfout_path, out_png, pv_dir,
                     title=build_title(run_label, domain, time_str, wrfout_path),
                     save_state=pvsm, timeidx=timeidx)
        if os.path.exists(pvsm):
            shutil.copy(pvsm, os.path.join(out_dir, "%s.pvsm" % catalog_id))
        else:
            print("WARNING: no ParaView state saved for %s" % catalog_id)
        write_readme(pv_dir, catalog_id)
        return [fig(catalog_id, out_png, run_label, domain="d%02d" % domain, valid_time=time_str)]
    except (tools.ToolNotFoundError, ImportError) as e:
        return [missing(catalog_id, "ParaView pvbatch", str(e))]
    except (ParaViewError, tools.ToolError, KeyError, ValueError) as e:
        # a pvbatch that runs and fails (or a frame without hydrometeors) is a missing figure
        # with its reason, not a failed report
        return [missing(catalog_id, "ParaView pvbatch", str(e).splitlines()[0])]
