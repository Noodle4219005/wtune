"""post/paraview/render_volume.py -- pvbatch script: total-condensate VOLUME rendering of one
wrfout frame (2026-10-02; replaces the QCLOUD+QICE isosurface of render_cloud.py, which drew
10% of the condensate mass as one opaque tube).

Inputs are written by post/xwrf_convert.build_volume():
  VOLUME.vti   ImageData, km (z already times the vertical exaggeration); point data log10_q
               (log10 g/kg clipped to [-2, 1]; -2 = transparent, also used for the cutaway's
               removed half), q_gkg, and vt (tangential wind, m s-1) for a cutaway
  TERRAIN.vtk  legacy structured grid at z = HGT * vexag, TERRAIN_H for land/sea colouring
  COAST.vtk    optional Natural Earth coastline polylines (km)

Run via (post/recipes/paraview3d.render_condensate_volume builds this command line):
  pvbatch --force-offscreen-rendering render_volume.py VOLUME.vti TERRAIN.vtk OUT.png
    --vexag N --center XC YC --bearing DEG --elevation DEG --focal-z KM --zoom F
    --opacity-unit KM --ztop KM [--ruler X Y LX LY TICK] [--cutaway --vt-max V] [--coast COAST.vtk] [--title T] [--note T]
    [--dx-note T] [--save-state PATH]

pvbatch on scc may segfault during teardown AFTER SaveScreenshot wrote a correct PNG; the
wrapper judges success by the PNG, not the exit code (see post/recipes/paraview3d.py).
"""
import argparse
import math
import sys

# log10(q g/kg) -> colour: light grey cloud, blue-grey precipitation cores (not a rainbow;
# the white background needs the low end slightly darker than white to stay visible)
VOLUME_RGB = [-2.0, 0.93, 0.93, 0.93,
              -1.0, 0.84, 0.85, 0.87,
              0.0, 0.66, 0.70, 0.76,
              1.0, 0.43, 0.50, 0.62]
# (value, opacity, midpoint, sharpness): 0 below 0.01 g/kg, ~0.7 at 1 g/kg, saturating above
VOLUME_OPACITY = [-2.0, 0.0, 0.5, 0.0,
                  -1.5, 0.06, 0.5, 0.0,
                  -1.0, 0.22, 0.5, 0.0,
                  0.0, 0.70, 0.5, 0.0,
                  1.0, 0.80, 0.5, 0.0]
# matplotlib 'inferno' sampled at 0, .15, ... 1 -- the fallback when ParaView's own "Inferno"
# preset is not found (ParaView 6.1 renamed its presets: "Viridis (matplotlib)" silently failed
# there and left the default rainbow-like map)
VT_PRESET = "Inferno"
INFERNO_RGB = [(0.0, 0.001, 0.000, 0.014), (0.15, 0.170, 0.042, 0.341),
               (0.30, 0.416, 0.090, 0.433), (0.45, 0.658, 0.179, 0.373),
               (0.60, 0.865, 0.317, 0.226), (0.75, 0.978, 0.558, 0.035),
               (0.90, 0.964, 0.844, 0.273), (1.0, 0.988, 0.998, 0.645)]
RULER_STEP_KM = 2.0
TERRAIN_RGB = [-500.0, 0.66, 0.80, 0.92, -1.0, 0.66, 0.80, 0.92,
               0.0, 0.70, 0.84, 0.60, 300.0, 0.86, 0.82, 0.55,
               1000.0, 0.66, 0.50, 0.34, 2500.0, 0.95, 0.95, 0.95]


def camera_offset(bearing_deg, elevation_deg):
    """Unit vector from the focal point to the camera (x east, y north, z up) for a camera at
    compass bearing bearing_deg from the target, elevation_deg above the horizon."""
    b = math.radians(bearing_deg)
    e = math.radians(elevation_deg)
    return (math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e))


def inferno_points(vmax, vmin=0.0):
    """Flat RGBPoints list of INFERNO_RGB spread over [vmin, vmax]."""
    out = []
    for t, r, g, b in INFERNO_RGB:
        out += [vmin + t * (vmax - vmin), r, g, b]
    return out


def ruler_geometry(x, y, lx, ly, ztop_km, vexag, tick_km, step_km=RULER_STEP_KM):
    """Vertical height ruler at (x, y) (rendered km), ticks every step_km of TRUE height
    pointing along the horizontal unit vector (lx, ly). Returns (segments, labels): segments
    [((x0, y0, z0), (x1, y1, z1))], labels [((x, y, z), text)] in rendered coordinates
    (z = true km * vexag)."""
    segs = [((x, y, 0.0), (x, y, ztop_km * vexag))]
    labels = []
    n = int(ztop_km // step_km + 1e-9)
    for k in range(n + 1):
        h = k * step_km
        z = h * vexag
        segs.append(((x, y, z), (x + lx * tick_km, y + ly * tick_km, z)))
        labels.append(((x + 1.5 * lx * tick_km, y + 1.5 * ly * tick_km, z),
                       "%g km" % h if k == n else "%g" % h))
    return segs, labels


def _set(obj, name, value):
    """Set a property that not every ParaView version has; never fatal."""
    try:
        setattr(obj, name, value)
    except Exception as e:   # noqa: BLE001
        print("render_volume: cannot set %s: %s" % (name, e), file=sys.stderr)


def _text(view, text, location, size, justification="Left"):
    from paraview.simple import Show, Text
    t = Text(Text=text)
    d = Show(t, view)
    d.Color = [0.0, 0.0, 0.0]
    d.FontSize = size
    d.WindowLocation = location
    _set(d, "Justification", justification)
    return d


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("volume_vti")
    ap.add_argument("terrain_vtk")
    ap.add_argument("out_png")
    ap.add_argument("--coast", default=None)
    ap.add_argument("--vexag", type=float, required=True)
    ap.add_argument("--center", type=float, nargs=2, required=True, metavar=("XC", "YC"))
    ap.add_argument("--bearing", type=float, default=180.0)
    ap.add_argument("--elevation", type=float, default=30.0)
    ap.add_argument("--focal-z", type=float, default=0.0, help="focal height, rendered km")
    ap.add_argument("--zoom", type=float, default=1.2)
    ap.add_argument("--opacity-unit", type=float, default=10.0)
    ap.add_argument("--cutaway", action="store_true")
    ap.add_argument("--vt-max", type=float, default=60.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--dx-note", default="")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 1100])
    ap.add_argument("--ztop", type=float, default=18.0, help="true km")
    # one comma-separated value: five separate floats break argparse when one prints as
    # "-1.2e-17" (cos/sin of the bearing), which its negative-number test does not recognise
    ap.add_argument("--ruler", type=lambda v: [float(x) for x in v.split(",")], default=None,
                    metavar="X,Y,LX,LY,TICK",
                    help="height ruler at X Y, ticks along (LX, LY), TICK km long")
    ap.add_argument("--save-state", default=None, metavar="PATH")
    return ap.parse_args(argv[1:])


def main(argv):
    a = parse_args(argv)
    from paraview.simple import (XMLImageDataReader, LegacyVTKReader, Slice, Clip, Show, Hide,
                                 Line, Text, ColorBy, GetActiveViewOrCreate, GetColorTransferFunction,
                                 GetOpacityTransferFunction, GetScalarBar, SaveScreenshot)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = list(a.size)
    view.Background = [1, 1, 1]
    _set(view, "UseColorPaletteForBackground", 0)   # ParaView >= 5.10: else Background ignored
    view.OrientationAxesVisibility = 0

    # terrain base: land height green-brown-white, flat blue sea (same table as render_cloud)
    terrain = LegacyVTKReader(FileNames=[a.terrain_vtk])
    td = Show(terrain, view)
    td.SetRepresentationType("Surface")
    ColorBy(td, ("POINTS", "TERRAIN_H"))
    t_lut = GetColorTransferFunction("TERRAIN_H")
    t_lut.RGBPoints = TERRAIN_RGB
    t_lut.ColorSpace = "RGB"
    t_lut.AutomaticRescaleRangeMode = "Never"
    td.SetScalarBarVisibility(view, False)
    td.Specular = 0.0
    td.Ambient = 0.6
    td.Diffuse = 0.4

    if a.coast:
        coast = LegacyVTKReader(FileNames=[a.coast])
        cd = Show(coast, view)
        cd.SetRepresentationType("Surface")
        cd.AmbientColor = [0.0, 0.0, 0.0]
        cd.DiffuseColor = [0.0, 0.0, 0.0]
        cd.LineWidth = 1.5

    vol = XMLImageDataReader(FileName=[a.volume_vti])
    vol.UpdatePipeline()
    vd = Show(vol, view)
    vd.SetRepresentationType("Volume")
    ColorBy(vd, ("POINTS", "log10_q"))
    lut = GetColorTransferFunction("log10_q")
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RGBPoints = VOLUME_RGB
    lut.ColorSpace = "RGB"
    pwf = GetOpacityTransferFunction("log10_q")
    _set(pwf, "AutomaticRescaleRangeMode", "Never")
    pwf.Points = VOLUME_OPACITY
    vd.ScalarOpacityFunction = pwf
    vd.ScalarOpacityUnitDistance = a.opacity_unit
    _set(vd, "Shade", 1)
    vd.SetScalarBarVisibility(view, False)

    if a.cutaway:
        # the camera-side half of log10_q is already transparent (xwrf_convert.build_volume);
        # this shows the cut face itself: the cloudy part (q >= 0.01 g/kg) of the vertical
        # plane through the centre, coloured by tangential wind
        b = math.radians(a.bearing)
        normal = [math.sin(b), math.cos(b), 0.0]
        sl = Slice(Input=vol)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [a.center[0], a.center[1], 0.0]
        sl.SliceType.Normal = normal
        cut = Clip(Input=sl)
        cut.ClipType = "Scalar"
        cut.Scalars = ["POINTS", "q_gkg"]
        cut.Value = 0.01
        cut.Invert = 0          # keep q >= 0.01 g/kg
        cd = Show(cut, view)
        cd.SetRepresentationType("Surface")
        ColorBy(cd, ("POINTS", "vt"))
        vlut = GetColorTransferFunction("vt")
        vlut.AutomaticRescaleRangeMode = "Never"
        try:
            ok = vlut.ApplyPreset(VT_PRESET, True)
        except Exception as e:   # noqa: BLE001
            ok = False
            print("render_volume: ApplyPreset(%r) raised %s" % (VT_PRESET, e), file=sys.stderr)
        if ok is False:
            print("WARNING: render_volume: ParaView preset %r not found; using built-in inferno "
                  "RGBPoints" % VT_PRESET, file=sys.stderr)
            vlut.RGBPoints = inferno_points(a.vt_max)
            vlut.ColorSpace = "RGB"
        vlut.RescaleTransferFunction(0.0, a.vt_max)
        cd.SetScalarBarVisibility(view, True)
        bar = GetScalarBar(vlut, view)
        bar.Title = "Tangential wind on the cut face (m s-1)"
        bar.ComponentTitle = ""
        bar.TitleColor = [0, 0, 0]
        bar.LabelColor = [0, 0, 0]
        _set(bar, "RangeLabelFormat", "%-#6.0f")
        _set(bar, "LabelFormat", "%-#6.0f")
        _set(bar, "AddRangeLabels", 1)
        bar.WindowLocation = "Any Location"
        bar.Position = [0.90, 0.25]
        bar.ScalarBarLength = 0.4
        bar.TitleFontSize = 20
        bar.LabelFontSize = 18

    if a.ruler:
        x, y, lx, ly, tick = a.ruler
        segs, labels = ruler_geometry(x, y, lx, ly, a.ztop, a.vexag, tick)
        for p0, p1 in segs:
            ld = Show(Line(Point1=list(p0), Point2=list(p1)), view)
            ld.ColorArrayName = ["POINTS", ""]   # solid colour; ColorBy(rep, None) raises in 6.1
            ld.AmbientColor = [0.0, 0.0, 0.0]
            ld.DiffuseColor = [0.0, 0.0, 0.0]
            ld.LineWidth = 2.0
        top = labels[-1][0]
        labels.append(((top[0], top[1], top[2] + 0.06 * a.ztop * a.vexag + tick),
                       "height (km, true; vertical exaggeration \u00d7%g)" % a.vexag))
        for k, (pos, txt) in enumerate(labels):
            title_label = k == len(labels) - 1
            t = Text(Text=txt)
            d = Show(t, view)
            d.Color = [0.0, 0.0, 0.0]
            d.FontSize = 16
            try:
                # screen-facing text anchored at a 3-D point (offscreen-safe, no widget)
                d.TextPropMode = "Billboard 3D Text"
                d.BillboardPosition = list(pos)
            except Exception as e:   # noqa: BLE001 -- never a stray 2-D label
                print("render_volume: no billboard text (%s); ruler unlabelled" % e,
                      file=sys.stderr)
                Hide(t, view)
                continue
            _set(d, "Justification", "Left" if title_label else "Right")
            _set(d, "VerticalJustification", "Bottom" if title_label else "Center")

    if a.title:
        _text(view, a.title, "Upper Center", 24, "Center")
    if a.dx_note:
        _text(view, a.dx_note, "Lower Left Corner", 18)
    if a.note:
        _text(view, a.note, "Lower Right Corner", 16, "Right")

    # camera: look at the centre from bearing/elevation; ResetCamera along that direction
    # fits the visible bounds, then the focal point is moved onto the centre at that distance
    view.CameraParallelProjection = 0
    off = camera_offset(a.bearing, a.elevation)
    cam = view.GetActiveCamera()
    fp = (a.center[0], a.center[1], a.focal_z)
    cam.SetFocalPoint(*fp)
    cam.SetPosition(fp[0] + off[0], fp[1] + off[1], fp[2] + off[2])
    cam.SetViewUp(0.0, 0.0, 1.0)
    view.ResetCamera(False)
    dist = cam.GetDistance()
    cam.SetFocalPoint(*fp)
    cam.SetPosition(fp[0] + off[0] * dist, fp[1] + off[1] * dist, fp[2] + off[2] * dist)
    cam.SetViewUp(0.0, 0.0, 1.0)
    cam.Zoom(a.zoom)

    if a.save_state:
        # absolute data paths; Load State -> "Search files under specified directory" repoints
        # them after a copy. A failure here must never fail the render.
        try:
            from paraview.simple import SaveState
            SaveState(a.save_state)
        except Exception as e:   # noqa: BLE001
            print("WARNING: SaveState(%s) failed: %s" % (a.save_state, e), file=sys.stderr)

    SaveScreenshot(a.out_png, view, ImageResolution=list(a.size),
                   FontScaling="Do not scale fonts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
