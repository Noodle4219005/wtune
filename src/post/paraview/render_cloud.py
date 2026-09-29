"""post/paraview/render_cloud.py -- pvpython/pvbatch script for `wtune post recipe`'s tc/
convection 3-D cloud rendering (decision 0057 item 3). Reads the two NetCDF files
post/xwrf_convert.py builds from one wrfout frame (a cloud file: CLOUD_TOTAL=QCLOUD+QICE+QSNOW
and Z_HEIGHT on the full 3-D grid; a terrain file: HGT on the 2-D grid), warps each onto its own
true physical height (WarpByScalar, ParaView's own point-set warp filter -- no filter does an
eta-level-index-to-height transform automatically) with a fixed vertical exaggeration (the
domain is O(1000 km) wide and O(20 km) tall, true-scale relief is visually flat from any oblique
angle -- standard practice in weather/ocean visualization), contours a cloud isosurface and a
terrain surface, and saves one offscreen PNG screenshot from an oblique angle.

Run via:
  pvbatch --force-offscreen-rendering render_cloud.py CLOUD.nc TERRAIN.nc OUT.png [isovalue]
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
"""
import sys

VERTICAL_EXAGGERATION = 15.0


def main(argv):
    if len(argv) < 4:
        print("usage: render_cloud.py CLOUD.nc TERRAIN.nc OUT.png [isovalue_kg_kg]",
             file=sys.stderr)
        return 2
    cloud_nc, terrain_nc, out_png = argv[1], argv[2], argv[3]
    isovalue = float(argv[4]) if len(argv) > 4 else 1e-5

    from paraview.simple import (NetCDFReader, WarpByScalar, Contour, Show, ColorBy,
                                 GetActiveViewOrCreate, ResetCamera, SaveScreenshot)

    cloud_reader = NetCDFReader(FileName=[cloud_nc])
    cloud_reader.SphericalCoordinates = 0
    cloud_reader.UpdatePipeline()
    cloud_warp = WarpByScalar(Input=cloud_reader)
    cloud_warp.Scalars = ["POINTS", "Z_HEIGHT"]
    cloud_warp.Normal = [0.0, 0.0, 1.0]
    cloud_warp.ScaleFactor = VERTICAL_EXAGGERATION
    cloud_warp.UpdatePipeline()

    terrain_reader = NetCDFReader(FileName=[terrain_nc])
    terrain_reader.SphericalCoordinates = 0
    terrain_reader.UpdatePipeline()
    terrain_warp = WarpByScalar(Input=terrain_reader)
    terrain_warp.Scalars = ["POINTS", "HGT"]
    terrain_warp.Normal = [0.0, 0.0, 1.0]
    terrain_warp.ScaleFactor = VERTICAL_EXAGGERATION
    terrain_warp.UpdatePipeline()

    view = GetActiveViewOrCreate("RenderView")
    view.Background = [1, 1, 1]

    terrain_disp = Show(terrain_warp, view)
    ColorBy(terrain_disp, ("POINTS", "HGT"))
    terrain_disp.SetRepresentationType("Surface")

    cloud = Contour(Input=cloud_warp)
    cloud.ContourBy = ["POINTS", "CLOUD_TOTAL"]
    cloud.Isosurfaces = [isovalue]
    cloud.UpdatePipeline()
    cloud_disp = Show(cloud, view)
    ColorBy(cloud_disp, ("POINTS", "Z_HEIGHT"))
    cloud_disp.SetScalarBarVisibility(view, True)
    cloud_disp.Opacity = 0.75

    # An oblique angle: ResetCamera() alone frames this near-flat (real aspect ~1000 km wide x
    # 20 km tall, even after exaggeration) object almost top-down, which hides the vertical
    # structure the exaggeration above is meant to reveal.
    ResetCamera(view)
    cam = view.GetActiveCamera()
    cam.Elevation(25)
    cam.Azimuth(35)
    ResetCamera(view)   # refit the view to the rotated camera so nothing is cut off

    SaveScreenshot(out_png, view, ImageResolution=[1200, 900])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
