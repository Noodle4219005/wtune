"""post/paraview/render_cloud.py -- pvpython/pvbatch script for `wtune post recipe`'s tc/
convection 3-D cloud rendering (decision 0057 item 3). Reads the two NetCDF files
post/xwrf_convert.py builds from one wrfout frame (a cloud file: QCLOUD_QICE=QCLOUD+QICE
and Z_HEIGHT on the full 3-D grid; a terrain file: HGT on the 2-D grid), warps each onto its own
true physical height (WarpByScalar, ParaView's own point-set warp filter -- no filter does an
eta-level-index-to-height transform automatically) with a fixed vertical exaggeration (the
domain is O(1000 km) wide and O(20 km) tall, true-scale relief is visually flat from any oblique
angle -- standard practice in weather/ocean visualization), contours a cloud isosurface and a
terrain surface, and saves one offscreen PNG screenshot from an oblique angle.

Run via:
  pvbatch --force-offscreen-rendering render_cloud.py CLOUD.nc TERRAIN.nc OUT.png
    [--coast COAST.vtk] [--isovalue kg/kg] [--vexag N] [--title TEXT]
(post/recipes/paraview3d.py is the wrapper that locates pvbatch (WTUNE_PARAVIEW_BIN) and calls
this).

Fixed 2026-09-29 against a real pvbatch on scc (see the delegation report / lead's R6 review):
two bugs made the isosurface come out empty ("vtkPVContourFilter: Contour array is null",
"Could not determine array range"):

1. `SphericalCoordinates` on vtkNetCDFReader defaults to True and is NOT data-dependent --
   left at its default, it maps the x/y/z dimensions as if they were (longitude, latitude,
   radius), collapsing point coordinates into a near-unit-sphere range regardless of what the
   file actually contains. Every reader here now sets `SphericalCoordinates = 0` explicitly.
2. The original approach used a `Calculator` with `CoordinateResults=1` to replace the
   Z coordinate with Z_HEIGHT. `vtkArrayCalculator`'s coordinate-output mode requires
   polydata or an unstructured grid; vtkNetCDFReader's own output is a structured grid, so the
   calculator silently failed ("Coordinate output specified, but output is not polydata or
   unstructured grid"), leaving every point at a degenerate (+-1,+-1,+-1) placeholder and the
   Contour filter with nothing to contour. `WarpByScalar` (vtkWarpScalar) accepts a
   structured-grid vtkPointSet directly and ADDS scalar*Normal to the existing coordinates
   instead of replacing them outright; since the reader's own Z (WRF's eta level, 0..~1) is
   negligible next to Z_HEIGHT (0..~20 km), warping by Z_HEIGHT with Normal=(0,0,1) gives the
   physical height to within order-1 metres, which is what this script now does.

Also fixed: post/xwrf_convert.py no longer attaches XLAT/XLONG as coordinates on the output
files -- their presence (as a CF `coordinates` attribute) is what makes vtkNetCDFReader try
degree-valued geometry in the first place; the native x(x)/y(y) dimension coordinates (already
metres, on the same physical scale as Z_HEIGHT) are what this script now relies on.

2026-10-01 rework: Z-up camera 35 deg above the horizon looking from the SSE; terrain base
plane coloured by TERRAIN_H (land height, flat blue over sea) with the coastline polyline
(post/xwrf_convert.py, Natural Earth 50m) drawn over it; cloud surface coloured by Z_KM with a
0..16 km colour bar; explicit vertical exaggeration (also in the title); white background.
"""
import argparse
import math
import sys

VERTICAL_EXAGGERATION = 30.0
ISOVALUE = 1e-4          # kg/kg QCLOUD+QICE (0.1 g/kg)
CAMERA_BEARING_DEG = 157.5   # camera sits SSE of the storm, looking NNW
CAMERA_ELEVATION_DEG = 35.0


def camera_offset(bearing_deg=CAMERA_BEARING_DEG, elevation_deg=CAMERA_ELEVATION_DEG):
    """Unit vector from the focal point to the camera (x east, y north, z up) for a camera
    standing at compass bearing bearing_deg from the target, elevation_deg above the horizon."""
    b = math.radians(bearing_deg)
    e = math.radians(elevation_deg)
    return (math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e))


def _set(obj, name, value):
    """Set a property that not every ParaView version has; never fatal."""
    try:
        setattr(obj, name, value)
    except Exception as e:
        print("render_cloud: cannot set %s: %s" % (name, e), file=sys.stderr)


def _warp(reader, scalar, factor):
    from paraview.simple import WarpByScalar
    w = WarpByScalar(Input=reader)
    w.Scalars = ["POINTS", scalar]
    w.Normal = [0.0, 0.0, 1.0]
    w.ScaleFactor = factor
    w.UpdatePipeline()
    return w


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud_nc")
    ap.add_argument("terrain_nc")
    ap.add_argument("out_png")
    ap.add_argument("--coast", default=None)
    ap.add_argument("--isovalue", type=float, default=ISOVALUE)
    ap.add_argument("--vexag", type=float, default=VERTICAL_EXAGGERATION)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv[1:])
    vexag = a.vexag

    from paraview.simple import (NetCDFReader, LegacyVTKReader, Transform, Contour, Show, Text,
                                 ColorBy, GetActiveViewOrCreate, GetColorTransferFunction,
                                 GetScalarBar, SaveScreenshot, UpdatePipeline)

    cloud_reader = NetCDFReader(FileName=[a.cloud_nc])
    cloud_reader.SphericalCoordinates = 0
    cloud_reader.UpdatePipeline()
    cloud_warp = _warp(cloud_reader, "Z_HEIGHT", vexag)

    terrain_reader = NetCDFReader(FileName=[a.terrain_nc])
    terrain_reader.SphericalCoordinates = 0
    terrain_reader.UpdatePipeline()
    terrain_warp = _warp(terrain_reader, "HGT", vexag)

    view = GetActiveViewOrCreate("RenderView")
    view.Background = [1, 1, 1]
    try:
        view.UseColorPaletteForBackground = 0   # ParaView >= 5.10: else Background is ignored
    except Exception:
        pass
    view.OrientationAxesVisibility = 0

    # base plane: one Show() of the terrain surface only (land height; sea is a flat blue)
    terrain_disp = Show(terrain_warp, view)
    terrain_disp.SetRepresentationType("Surface")
    ColorBy(terrain_disp, ("POINTS", "TERRAIN_H"))
    t_lut = GetColorTransferFunction("TERRAIN_H")
    t_lut.RGBPoints = [-500.0, 0.66, 0.80, 0.92,   -1.0, 0.66, 0.80, 0.92,
                       0.0, 0.70, 0.84, 0.60,      300.0, 0.86, 0.82, 0.55,
                       1000.0, 0.66, 0.50, 0.34,   2500.0, 0.95, 0.95, 0.95]
    t_lut.ColorSpace = "RGB"
    t_lut.AutomaticRescaleRangeMode = "Never"
    terrain_disp.SetScalarBarVisibility(view, False)
    terrain_disp.Specular = 0.0
    terrain_disp.Ambient = 0.7
    terrain_disp.Diffuse = 0.3

    if a.coast:
        coast = LegacyVTKReader(FileNames=[a.coast])
        coast_t = Transform(Input=coast)
        coast_t.Transform.Scale = [1.0, 1.0, vexag]
        coast_disp = Show(coast_t, view)
        coast_disp.SetRepresentationType("Surface")
        coast_disp.AmbientColor = [0.0, 0.0, 0.0]
        coast_disp.DiffuseColor = [0.0, 0.0, 0.0]
        coast_disp.LineWidth = 1.5

    cloud = Contour(Input=cloud_warp)
    cloud.ContourBy = ["POINTS", "QCLOUD_QICE"]
    cloud.Isosurfaces = [a.isovalue]
    cloud.UpdatePipeline()
    cloud_disp = Show(cloud, view)
    ColorBy(cloud_disp, ("POINTS", "Z_KM"))
    c_lut = GetColorTransferFunction("Z_KM")
    c_lut.AutomaticRescaleRangeMode = "Never"
    c_lut.RescaleTransferFunction(0.0, 16.0)
    try:
        c_lut.ApplyPreset("Viridis (matplotlib)", True)
        c_lut.RescaleTransferFunction(0.0, 16.0)
    except Exception:
        pass
    cloud_disp.Opacity = 0.85
    cloud_disp.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(c_lut, view)
    bar.Title = "Cloud-surface height (km)"
    bar.ComponentTitle = ""
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    bar.UseCustomLabels = 1
    bar.CustomLabels = [0.0, 4.0, 8.0, 12.0, 16.0]
    _set(bar, "AddRangeLabels", 0)
    _set(bar, "DrawTickMarks", 1)
    _set(bar, "RangeLabelFormat", "%-#6.0f")
    _set(bar, "LabelFormat", "%-#6.0f")
    bar.WindowLocation = "Any Location"
    bar.Position = [0.90, 0.25]
    bar.ScalarBarLength = 0.4
    bar.TitleFontSize = 22
    bar.LabelFontSize = 20

    if a.title:
        title = Text(Text=a.title)
        td = Show(title, view)
        td.Color = [0.0, 0.0, 0.0]
        td.FontSize = 26
        td.WindowLocation = "Upper Center"
        td.Justification = "Center"
    note = Text(Text="Base: terrain height (land, green-brown-white; blue = sea); "
                     "black line: Natural Earth 50m coast")
    nd = Show(note, view)
    nd.Color = [0.0, 0.0, 0.0]
    nd.FontSize = 18
    nd.WindowLocation = "Lower Center"
    nd.Justification = "Center"

    # Z up, camera SSE of the storm, 35 deg above the horizon
    view.CameraParallelProjection = 0
    view.ResetCamera(False)
    cam = view.GetActiveCamera()      # the vtkCamera, so position/view-up are applied now
    fp = cam.GetFocalPoint()
    off = camera_offset()
    dist = 2.0e6
    cam.SetPosition(fp[0] + off[0] * dist, fp[1] + off[1] * dist, fp[2] + off[2] * dist)
    cam.SetViewUp(0.0, 0.0, 1.0)
    view.ResetCamera(False)   # refit along the same direction so the whole storm is in frame
    cam.Zoom(1.1)            # the fit is to the exaggerated bounding sphere; tighten it a little

    SaveScreenshot(a.out_png, view, ImageResolution=[1600, 1100],
                   FontScaling="Do not scale fonts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
