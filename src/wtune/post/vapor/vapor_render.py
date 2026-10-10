"""vapor_render.py -- WTune VAPOR bundle renderer (copied into every <out>/vapor/d0N/ bundle by
post/vapor_bundle.py). Runs on the user's own x86_64 Linux / macOS workstation inside a conda
env with NCAR's official VAPOR (`conda create -n vapor -c conda-forge -c ncar-vapor vapor`);
there is no linux-aarch64 VAPOR build, so it never runs on scc itself.

    python vapor_render.py                  # PNG per time step + a .vs3 session in ./vapor_png
    python vapor_render.py --list           # variable names and time-step count only
    python vapor_render.py --session-only   # .vs3 for the GUI (File > Open Session), no render

Scene: terrain (ImageRenderer with HGT as its height variable, VAPOR's built-in map image) plus
one VolumeIsoRenderer isosurface on QCLOUD and one on QICE (skipped when the run has no QICE).

Two classes of VAPOR Python call are used, and the difference matters because this script was
written without access to a VAPOR install:
  documented (Session/OpenDataset/GetDataVarNames/GetNumTimeSteps/NewRenderer/SetVariableName/
    SetIsoValues/SetHeightVariableName/SetResolution/GetCamera/ViewAll/Render/Save) -- called
    directly; if one of these fails, the script stops with VAPOR's own error.
  unverified (Session.SetTimestep, the oblique-camera Get/SetPosition/SetDirection/SetUp, the
    isosurface colour via GetPrimaryTransferFunction) -- each goes through _try(), which prints
    a note and continues, and the final summary lists every one that failed. If SetTimestep is
    ignored or fails, every PNG may show the SAME time; the script cannot detect that.
Pure stdlib + vapor, deliberately: the bundle is copied to machines with nothing but VAPOR.
"""
import argparse
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ISO_CLOUD = 1e-4      # kg/kg = 0.1 g/kg, the same threshold as WTune's ParaView cloud render
ISO_ICE = 1e-5        # ice mixing ratios are typically an order of magnitude below cloud water
CLOUD_RGB = (1.0, 1.0, 1.0)
ICE_RGB = (0.6, 0.8, 1.0)


class _Log:
    """What happened, for the final summary: rendered PNGs, skipped scene parts, and the
    unverified calls that failed."""

    def __init__(self):
        self.rendered, self.skipped, self.failed = [], [], []

    def summary(self):
        lines = ["", "== vapor_render.py summary ==",
                 "rendered: %d PNG(s)%s" % (len(self.rendered),
                                            (" -> %s" % os.path.dirname(self.rendered[0]))
                                            if self.rendered else "")]
        lines += ["skipped: %s" % s for s in self.skipped]
        lines += ["unverified call failed (continued without it): %s" % f for f in self.failed]
        if not self.skipped and not self.failed:
            lines.append("nothing skipped, every unverified call succeeded")
        return "\n".join(lines)


def _try(log, what, fn):
    """Runs an UNVERIFIED VAPOR call; on any exception prints a note, records it and returns
    None instead of stopping the whole render."""
    try:
        return fn()
    except Exception as e:                      # noqa: BLE001 -- any VAPOR-side failure
        msg = "%s: %s: %s" % (what, type(e).__name__, e)
        print("note: unverified VAPOR call failed, continuing: %s" % msg)
        log.failed.append(msg)
        return None


def _oblique(cam):
    """Tilts ViewAll's top-down view to ~45 deg from the south (unverified Camera API): keeps
    the target and the distance, moves the eye south and down."""
    pos = [float(v) for v in cam.GetPosition()]
    tgt = [float(v) for v in cam.GetTarget()]
    dist = math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, tgt)))
    s = math.sqrt(0.5)
    eye = [tgt[0], tgt[1] - dist * s, tgt[2] + dist * s]
    cam.SetPosition(eye)
    cam.SetDirection([(t - e) / dist for t, e in zip(tgt, eye)])
    cam.SetUp([0.0, s, s])


def _iso(log, data, renderer, names, var, value, rgb):
    if var not in names:
        log.skipped.append("%s isosurface (no %s in these files)" % (var, var))
        return None
    iso = data.NewRenderer(renderer.VolumeIsoRenderer)
    iso.SetVariableName(var)
    iso.SetIsoValues([value])
    _try(log, "%s colour (GetPrimaryTransferFunction)" % var,
         lambda: iso.GetPrimaryTransferFunction().SetColorRGBList([list(rgb), list(rgb)]))
    return iso


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", default=os.path.join(HERE, "wrfout_*"),
                    help="glob of the WRF files, one time per file (default: this bundle's)")
    ap.add_argument("--out", default="vapor_png", help="output directory (default ./vapor_png)")
    ap.add_argument("--iso-cloud", type=float, default=ISO_CLOUD, dest="iso_cloud",
                    help="QCLOUD isovalue, kg/kg (default %g)" % ISO_CLOUD)
    ap.add_argument("--iso-ice", type=float, default=ISO_ICE, dest="iso_ice",
                    help="QICE isovalue, kg/kg (default %g)" % ISO_ICE)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--list", action="store_true", help="print variable names and the time-"
                    "step count, then stop")
    ap.add_argument("--session-only", action="store_true", dest="session_only",
                    help="save the .vs3 session without rendering (open it in the GUI)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(glob.glob(args.files))
    if not files:
        print("error: no files match %s" % args.files)
        return 2
    try:
        from vapor import dataset, renderer, session
    except ImportError as e:
        print("error: cannot import vapor (%s) -- run inside the VAPOR conda env: "
              "conda activate vapor" % e)
        return 3

    log = _Log()
    ses = session.Session()
    data = ses.OpenDataset(dataset.WRF, files)
    names = list(data.GetDataVarNames())
    probe = "QCLOUD" if "QCLOUD" in names else (names[0] if names else "HGT")
    n_steps = int(data.GetNumTimeSteps(probe))
    print("%d file(s), %d time step(s) (of %s)" % (len(files), n_steps, probe))
    if args.list:
        print("variables: %s" % ", ".join(sorted(names)))
        return 0

    ses.SetResolution(args.width, args.height)
    if "HGT" in names:
        terrain = data.NewRenderer(renderer.ImageRenderer)
        terrain.SetHeightVariableName("HGT")
    else:
        log.skipped.append("terrain (no HGT in these files)")
    _iso(log, data, renderer, names, "QCLOUD", args.iso_cloud, CLOUD_RGB)
    _iso(log, data, renderer, names, "QICE", args.iso_ice, ICE_RGB)

    cam = ses.GetCamera()
    cam.ViewAll()
    _try(log, "oblique camera (GetPosition/GetTarget/SetPosition/SetDirection/SetUp)",
         lambda: _oblique(cam))

    os.makedirs(args.out, exist_ok=True)
    vs3 = os.path.join(args.out, "wtune_session.vs3")
    if args.session_only:
        ses.Save(vs3)
        print("saved session %s (no render: --session-only)" % vs3)
        print(log.summary())
        return 0

    # one PNG per time step, named after its file when there is exactly one time per file
    stems = ([os.path.basename(f) for f in files] if len(files) == n_steps
             else ["step_%03d" % i for i in range(n_steps)])
    for i in range(n_steps):
        if n_steps > 1:
            _try(log, "SetTimestep(%d)" % i, lambda i=i: ses.SetTimestep(i))
        png = os.path.join(args.out, "%s.png" % stems[i])
        ses.Render(png)
        log.rendered.append(png)
        print("rendered %s" % png)
    ses.Save(vs3)
    print("saved session %s" % vs3)
    print(log.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
