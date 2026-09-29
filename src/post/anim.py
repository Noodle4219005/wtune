"""post/anim.py -- ffmpeg-based animation builder (decision 0058 section 2 update,
2026-09-29): turns an ordered PNG time series (post/timeseries.py's own per-time figure
files) into an MP4 (always) and, optionally, a GIF, using ffmpeg -- animation is not
something wrf-python/CDO/NCO/MetPy/MET/ParaView provide, so this wraps the one established
tool that does it (added to envs/wtune/pixi.toml, conda-forge).
"""
from __future__ import annotations

import os
import tempfile

from post import tools


def build_animation(png_paths, out_path, fps=2, gif=False):
    """png_paths: ordered list of PNG files (already time-sorted by the caller -- this
    function does not sort). Writes out_path (.mp4). When gif=True also writes a .gif next
    to it. Uses ffmpeg's concat demuxer via a temp file list, so frames need not share one
    naming pattern or directory (post/timeseries.py's per-time subdirectories don't)."""
    if not png_paths:
        raise ValueError("build_animation: no frames given")
    ffmpeg = tools.find_on_path("ffmpeg", install_hint="add 'ffmpeg' to "
                                "envs/wtune/pixi.toml (conda-forge) and reinstall the env.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fd, listfile = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as fh:
            for p in png_paths:
                fh.write("file '%s'\nduration %g\n" % (os.path.abspath(p), 1.0 / fps))
            # ffmpeg's concat demuxer drops the final `duration` unless the last file is
            # repeated once more with no following duration line.
            fh.write("file '%s'\n" % os.path.abspath(png_paths[-1]))

        tools.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                  "-vf", "fps=%g,format=yuv420p" % fps, out_path])
        outputs = [out_path]
        if gif:
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            tools.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                      "-vf", "fps=%g" % fps, gif_path])
            outputs.append(gif_path)
        return outputs
    finally:
        os.remove(listfile)
