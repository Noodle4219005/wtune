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
    candidates = []
    if env_bin:
        candidates.append(os.path.join(env_bin, "pvbatch"))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    candidates.append(os.path.join(repo_root, "envs", "paraview", ".pixi", "envs", "default",
                                   "bin", "pvbatch"))
    for c in candidates:
        if os.path.exists(c):
            return c
    path = shutil.which("pvbatch")
    if path:
        return path
    raise tools.ToolNotFoundError(
        "pvbatch not found: install ParaView (bin/install-postproc.sh) and export "
        "WTUNE_PARAVIEW_BIN=<repo>/envs/paraview/.pixi/envs/default/bin, or put pvbatch on PATH.")


def render_cloud(wrfout_path, out_png, work_dir, isovalue=1e-5):
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
    cloud_nc, terrain_nc = convert_frame(wrfout_path, prefix)
    rc, out, err = tools.run([pvbatch, "--force-offscreen-rendering", _SCRIPT, cloud_nc,
                              terrain_nc, out_png, str(isovalue)], check=False)
    if not os.path.exists(out_png):
        raise ParaViewError("pvbatch (exit %d) did not produce %s:\n%s" % (
            rc, out_png, (err or out).strip()[-2000:]))
    return out_png


def figures(catalog_id, wrfout_path, out_dir, work_dir, run_label, domain, time_str):
    from post.recipe import fig, missing
    out_png = os.path.join(out_dir, "%s.png" % catalog_id)
    try:
        render_cloud(wrfout_path, out_png, work_dir)
        return [fig(catalog_id, out_png, run_label, domain="d%02d" % domain, valid_time=time_str)]
    except (tools.ToolNotFoundError, ImportError) as e:
        return [missing(catalog_id, "ParaView pvbatch", str(e))]
    except (ParaViewError, tools.ToolError) as e:
        # a pvbatch that runs and fails is a missing figure with its reason, not a failed report
        return [missing(catalog_id, "ParaView pvbatch", str(e).splitlines()[0])]
