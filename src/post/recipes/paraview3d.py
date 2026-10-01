"""post/recipes/paraview3d.py -- wrapper that finds pvbatch (WTUNE_PARAVIEW_BIN, default
envs/paraview/.pixi/envs/default/bin, per bin/install-postproc.sh) and runs it against
post/paraview/render_cloud.py (decision 0057 item 3: a cloud isosurface over terrain, for the
tc and convection figure sets). post/xwrf_convert.py builds the two small CF NetCDF inputs
render_cloud.py needs from one wrfout frame. If pvbatch is not found, this returns a clear
missing-observation-style record instead of failing the whole report -- same contract as a
missing MET tool in post/tc.py's run_tc_report().
"""
from __future__ import annotations

import os
import shutil

from post import tools

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paraview", "render_cloud.py")


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


VERTICAL_EXAGGERATION = 30.0
ISOVALUE = 1e-4   # kg/kg, QCLOUD+QICE = 0.1 g/kg


def build_title(run_label, domain, time_str, isovalue=ISOVALUE, vexag=VERTICAL_EXAGGERATION):
    """Two-line figure title: run, domain, valid time / isosurface quantity, threshold, and the
    vertical exaggeration."""
    return ("%s  d%02d  valid %s\nCloud isosurface QCLOUD+QICE \u2265 %g g/kg  "
            "(vertical exaggeration \u00d7%g)" % (run_label, domain, time_str,
                                                  isovalue * 1000.0, vexag))


def render_cloud(wrfout_path, out_png, work_dir, isovalue=ISOVALUE, title="",
                 vexag=VERTICAL_EXAGGERATION, save_state=None):
    """Converts wrfout_path (one frame) via post/xwrf_convert.py, then runs pvbatch against
    render_cloud.py. Returns out_png. Raises tools.ToolNotFoundError if pvbatch/xwrf/wrf-python
    is unavailable, ParaViewError if pvbatch ran but truly produced no output.

    Runs pvbatch with check=False and judges success by whether out_png exists, not by the
    subprocess's exit code: on scc (headless, offscreen/software GL) pvbatch reliably writes a
    correct PNG via SaveScreenshot and then segfaults during its OWN process teardown/OpenGL
    context cleanup (observed exit code 139, SIGSEGV, AFTER out_png was already written and
    verified complete) -- treating that as a hard failure discarded a real, correct render.
    """
    pvbatch = find_pvbatch()
    from post.xwrf_convert import convert_frame
    os.makedirs(work_dir, exist_ok=True)
    # no ':' in the converted file names: ParaView's file-series reader fails on
    # wrfout_d01_2005-08-29_00:00:00_cloud.nc and reads the same file renamed (scc, work1)
    stem = os.path.splitext(os.path.basename(wrfout_path))[0].replace(":", "")
    prefix = os.path.join(work_dir, stem)
    cloud_nc, terrain_nc, coast_vtk = convert_frame(wrfout_path, prefix)
    if os.path.exists(out_png):
        os.remove(out_png)   # a stale PNG must not pass the exists() check below
    cmd = [pvbatch, "--force-offscreen-rendering", _SCRIPT, cloud_nc, terrain_nc, out_png,
           "--isovalue", str(isovalue), "--vexag", str(vexag), "--title", title]
    if coast_vtk:
        cmd += ["--coast", coast_vtk]
    if save_state:
        cmd += ["--save-state", save_state]
    rc, out, err = tools.run(cmd, check=False)
    if not os.path.exists(out_png):
        raise ParaViewError("pvbatch (exit %d) did not produce %s:\n%s" % (
            rc, out_png, (err or out).strip()[-2000:]))
    return out_png


_README = """\
WTune 3-D 雲場 ParaView 場景（%(catalog_id)s）

開啟方式（ParaView >= 5.11）：
  File -> Load State -> 選 %(catalog_id)s.pvsm
  -> 在 "Load State Options" 選 "Search files under specified directory"，指向本資料夾。
  （.pvsm 內的資料路徑是產生當下的絕對路徑；資料夾搬移或複製後必須用此選項重新對應。）

檔案：
%(files)s
垂直放大倍率：x%(vexag)g（地形與雲場 z 軸皆乘此倍率，僅為視覺用）
等值面：QCLOUD+QICE = %(iso)g kg/kg（%(iso_g)g g/kg）
"""


def write_readme(pv_dir, catalog_id, isovalue=ISOVALUE, vexag=VERTICAL_EXAGGERATION):
    names = sorted(os.listdir(pv_dir))
    desc = {".pvsm": "ParaView 狀態檔（場景、相機、著色）", ".vtk": "海岸線折線（legacy VTK）"}
    lines = []
    for n in names:
        if n == "README.txt":
            continue
        ext = os.path.splitext(n)[1]
        if n.endswith("_cloud.nc"):
            d = "雲場 QCLOUD+QICE（CF NetCDF，等值面來源）"
        elif n.endswith("_terrain.nc"):
            d = "地形高度（CF NetCDF）"
        else:
            d = desc.get(ext, "")
        lines.append("  %s  %s" % (n, d))
    with open(os.path.join(pv_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(_README % dict(catalog_id=catalog_id, files="\n".join(lines) + "\n",
                               vexag=vexag, iso=isovalue, iso_g=isovalue * 1000.0))


def figures(catalog_id, wrfout_path, out_dir, work_dir, run_label, domain, time_str):
    """work_dir is kept for call-site compatibility but unused: the converted files are written
    straight into <out_dir>/<catalog_id>_paraview/ (no copy), next to the .pvsm and README.txt,
    and the .pvsm is also written to <out_dir>/<catalog_id>.pvsm."""
    from post.recipe import fig, missing
    out_png = os.path.join(out_dir, "%s.png" % catalog_id)
    pv_dir = os.path.join(out_dir, "%s_paraview" % catalog_id)
    pvsm = os.path.join(pv_dir, "%s.pvsm" % catalog_id)
    try:
        render_cloud(wrfout_path, out_png, pv_dir,
                     title=build_title(run_label, domain, time_str), save_state=pvsm)
        if os.path.exists(pvsm):
            shutil.copy(pvsm, os.path.join(out_dir, "%s.pvsm" % catalog_id))
        else:
            print("WARNING: no ParaView state saved for %s" % catalog_id)
        write_readme(pv_dir, catalog_id)
        return [fig(catalog_id, out_png, run_label, domain="d%02d" % domain, valid_time=time_str)]
    except (tools.ToolNotFoundError, ImportError) as e:
        return [missing(catalog_id, "ParaView pvbatch", str(e))]
    except (ParaViewError, tools.ToolError) as e:
        # a pvbatch that runs and fails is a missing figure with its reason, not a failed report
        return [missing(catalog_id, "ParaView pvbatch", str(e).splitlines()[0])]
