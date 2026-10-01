"""Total-condensate volume rendering (post/xwrf_convert.build_volume + post/recipes/paraview3d.
render_condensate_volume + post/paraview/render_volume.py): grid construction, regridding,
log scaling, tangential wind, the .vti writer and the pvbatch command line. No ParaView needed;
pvbatch itself is exercised on scc."""
import importlib.util
import os
import py_compile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from post import tools, xwrf_convert as xc
from post.recipes import paraview3d

RENDER_VOLUME = os.path.join(os.path.dirname(paraview3d.__file__), "..", "paraview",
                             "render_volume.py")


def read_vti(path):
    """Independent reader for write_vti's format: (attrs of ImageData, {name: (nz,ny,nx)})."""
    raw = open(path, "rb").read()
    head, _, body = raw.partition(b'<AppendedData encoding="raw">')
    start = body.index(b"_") + 1
    root = ET.fromstring(head.decode() + "</VTKFile>")
    img = root.find("ImageData")
    ext = [int(v) for v in img.get("WholeExtent").split()]
    nx, ny, nz = ext[1] + 1, ext[3] + 1, ext[5] + 1
    arrays = {}
    for da in img.iter("DataArray"):
        off = start + int(da.get("offset"))
        n = int(np.frombuffer(body[off:off + 8], "<u8")[0])
        a = np.frombuffer(body[off + 8:off + 8 + n], "<f4")
        arrays[da.get("Name")] = a.reshape(nz, ny, nx)
    return img.attrib, arrays


# ------------------------------------------------------------------ pure grid helpers


def test_levels_0_to_18_km_every_250_m():
    lv = xc.volume_levels_km()
    assert len(lv) == 73 and lv[0] == 0.0 and lv[-1] == 18.0
    assert np.allclose(np.diff(lv), 0.25)


def test_crop_whole_domain_drops_boundary_zone():
    assert xc.crop_bounds(100, 120, 3.0, 3.0, 5) == (5, 95, 5, 115)


def test_crop_box_around_centre_and_clipped_to_interior():
    # 60 km box at 3 km -> +-10 cells around (50, 60)
    assert xc.crop_bounds(100, 120, 3.0, 3.0, 5, (50, 60), 60.0) == (40, 61, 50, 71)
    # centre near the south-west corner: the box is cut at the boundary zone, not the edge
    assert xc.crop_bounds(100, 120, 3.0, 3.0, 5, (7, 8), 60.0) == (5, 18, 5, 19)
    with pytest.raises(ValueError):
        xc.crop_bounds(8, 8, 3.0, 3.0, 5)


def test_nearest_index_and_stride():
    lat, lon = np.meshgrid(np.linspace(20, 30, 11), np.linspace(-90, -80, 21), indexing="ij")
    j, i, d = xc.nearest_index(lat, lon, 24.1, -86.9)
    assert (j, i) == (4, 6) and d < 20.0
    assert xc.horizontal_stride(300, 400) == 1 and xc.horizontal_stride(300, 401) == 2


def test_regrid_linear_in_height_zero_below_ground_and_above_top():
    ny, nx = 2, 3
    hgt = np.array([[0.0, 1000.0, 0.0], [0.0, 0.0, 0.0]])
    zk = np.array([500.0, 2000.0, 4000.0, 8000.0])          # mass levels above the terrain
    z = zk[:, None, None] + hgt[None]
    f = z.copy()                                             # field == its own height
    out = xc.regrid_to_heights(f, z, hgt, np.array([0.0, 0.25, 1.0, 3.0, 6.0, 9.0, 10.0]))
    sea = out[:, 0, 0]
    # 0 km (== HGT over sea, below the first level) -> lowest value; 0.25 km same
    assert sea[0] == pytest.approx(500.0) and sea[1] == pytest.approx(500.0)
    assert sea[2] == pytest.approx(1000.0) and sea[3] == pytest.approx(3000.0)
    assert sea[4] == pytest.approx(6000.0)
    assert sea[5] == 0.0 and sea[6] == 0.0                   # above the model top
    hill = out[:, 0, 1]                                      # HGT 1000 m, first level 1500 m
    assert hill[0] == 0.0 and hill[1] == 0.0                 # below ground
    assert hill[2] == pytest.approx(1500.0)                  # HGT <= z < first level
    assert hill[3] == pytest.approx(3000.0)
    assert out.dtype == np.float32 and out.shape == (7, ny, nx)


def test_log_condensate_clips_to_minus2_plus1():
    lq = xc.log_condensate(np.array([0.0, 1e-4, 0.01, 0.1, 1.0, 50.0]))
    assert np.allclose(lq, [-2, -2, -2, -1, 0, 1], atol=1e-6)


def test_tangential_wind_solid_body_vortex():
    x = np.arange(-50.0, 51.0, 10.0)
    y = np.arange(-30.0, 31.0, 10.0)
    X, Y = np.meshgrid(x, y)
    w = 0.5                                                  # m/s per km
    u, v = -w * Y, w * X                                     # counter-clockwise
    vt = xc.tangential_wind(u[None], v[None], x, y, 0.0, 0.0)[0]
    assert np.allclose(vt, w * np.hypot(X, Y), atol=1e-4)
    assert vt[3, 5] == 0.0                                   # centre point
    vt_sh = xc.tangential_wind(u, v, x, y, 0.0, 0.0, cyclonic_sign=-1.0)
    assert np.allclose(vt_sh, -vt, atol=1e-4)
    # radial outflow has no tangential component
    assert np.allclose(xc.tangential_wind(X, Y, x, y, 0.0, 0.0), 0.0, atol=1e-5)


def test_camera_side_mask_removes_half_facing_camera():
    x = np.arange(-2.0, 3.0)
    y = np.arange(-2.0, 3.0)
    m = xc.camera_side_mask(x, y, 0.0, 0.0, 180.0)           # camera due south
    assert m[0].all() and m[1].all() and not m[2:].any()
    m = xc.camera_side_mask(x, y, 0.0, 0.0, 202.5)           # SSW: the south-west half
    assert m[0, 0] and not m[4, 4] and not m[2, 2]
    # slab: also 1.5 cells behind the plane (the plane itself stays at y = 0)
    m = xc.camera_side_mask(x, y, 0.0, 0.0, 180.0, beyond_km=1.5)
    assert m[:4].all() and not m[4].any()


def test_default_vexag_column_one_fifth_of_width():
    assert xc.default_vexag(600.0, 18.0, box_km=600.0) == 8.0      # storm rule wins (6.7)
    assert xc.default_vexag(600.0, 4.0, box_km=600.0) == 30.0      # 600/20 = 30
    assert xc.default_vexag(3000.0, 18.0) == 30.0                  # 33 -> cap
    assert xc.default_vexag(300.0, 4.0) == 15.0                    # SC26-P1 lake effect
    assert xc.default_vexag(50.0, 18.0) == 1.0


def test_volume_top_adapts_to_cloud_tops():
    ct = np.full(1000, np.nan)
    ct[:900] = 2400.0                          # lake-effect deck, 2.4 km tops
    assert xc.volume_top_km(ct) == 4.0         # 3.4 -> 4 (also the floor)
    ct[:900] = 4100.0
    assert xc.volume_top_km(ct) == 6.0         # 5.1 -> 6
    ct[:] = 16500.0
    assert xc.volume_top_km(ct) == 18.0        # 17.5 -> 18 (cap)
    assert xc.volume_top_km(np.full(5, np.nan)) == 18.0
    assert np.allclose(xc.volume_levels_km(6.0)[-1], 6.0)


def test_write_vti_round_trip(tmp_path):
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    b = -a
    p = xc.write_vti(str(tmp_path / "v.vti"), (10.0, 20.0, 0.0), (3.0, 3.0, 2.0),
                     {"log10_q": a, "q_gkg": b})
    attrs, arr = read_vti(p)
    assert attrs["WholeExtent"] == "0 3 0 2 0 1"
    assert [float(s) for s in attrs["Spacing"].split()] == [3.0, 3.0, 2.0]
    assert [float(s) for s in attrs["Origin"].split()] == [10.0, 20.0, 0.0]
    assert np.array_equal(arr["log10_q"], a) and np.array_equal(arr["q_gkg"], b)
    # x varies fastest: element (z=0, y=0, x=1) is the second value written
    assert arr["log10_q"][0, 0, 1] == 1.0


# ------------------------------------------------------------------ synthetic wrfout


def _wrfout(path, nz=10, ny=40, nx=40, dx=3000.0, omega=1e-3):
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("bottom_top_stag", nz + 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("west_east_stag", nx + 1)
        ds.DX, ds.DY = dx, dx
        ds.SIMULATION_START_DATE = "2005-08-27_00:00:00"
        m3 = ("Time", "bottom_top", "south_north", "west_east")
        hgt = np.zeros((ny, nx))
        hgt[:, 30:] = 1000.0                                  # a plateau in the east
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        for name, val in (("HGT", hgt), ("LANDMASK", (hgt > 0).astype(float)),
                          ("XLAT", 20.0 + jj * 0.027), ("XLONG", -80.0 + ii * 0.0285)):
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = val
        zs = hgt[None] + 1500.0 * np.arange(nz + 1)[:, None, None]
        ds.createVariable("PHB", "f4", ("Time", "bottom_top_stag", "south_north",
                                        "west_east"))[0] = zs * 9.81
        ds.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north",
                                       "west_east"))[0] = 0.0
        qc = np.zeros((nz, ny, nx))
        qc[2:5] = 1e-3                                        # 1 g/kg at ~3.75-6.75 km AGL
        qs = np.zeros((nz, ny, nx))
        qs[6:8] = 2e-3
        ds.createVariable("QCLOUD", "f4", m3)[0] = qc
        ds.createVariable("QSNOW", "f4", m3)[0] = qs
        ds.createVariable("QICE", "f4", m3)[0] = 0.0
        # solid-body vortex about grid point (20, 20), grid-relative
        y_m = (np.arange(ny) - 20) * dx / 1000.0
        x_m = (np.arange(nx) - 20) * dx / 1000.0
        U = -omega * 1000.0 * np.broadcast_to(y_m[:, None], (ny, nx + 1))
        V = omega * 1000.0 * np.broadcast_to(x_m[None, :], (ny + 1, nx))
        ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north",
                                      "west_east_stag"))[0] = np.broadcast_to(U, (nz, ny, nx + 1))
        ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag",
                                      "west_east"))[0] = np.broadcast_to(V, (nz, ny + 1, nx))
    return path


@pytest.fixture
def no_projection(monkeypatch):
    # the synthetic file has no map-projection attributes: index * DX axes, no coastline
    monkeypatch.setattr(xc, "_projected_axes",
                        lambda p, ny, nx, dx, dy: (np.arange(nx) * dx, np.arange(ny) * dy, None))


def test_build_volume_storm_box_cutaway(tmp_path, no_projection):
    import netCDF4
    w = _wrfout(str(tmp_path / "wrfout_d01_2005-08-29_00:00:00"))
    with netCDF4.Dataset(w) as ds:
        lat, lon = float(ds["XLAT"][0, 20, 20]), float(ds["XLONG"][0, 20, 20])
    info = xc.build_volume(w, str(tmp_path / "s"), center_latlon=(lat, lon), box_km=60.0,
                           cutaway=True, bearing_deg=180.0, bdy_width=5)
    # cloud tops: snow to 11.25 km over sea, 12.25 km over the plateau -> top 14 km
    assert info["ztop_km"] == 14.0
    assert info["vexag"] == 2.0                    # 60/(5*14) -> 1; storm rule 60/75 -> 2
    assert info["center_km"] == (60.0, 60.0)
    assert info["extent_km"] == (30.0, 90.0, 30.0, 90.0)        # +-10 cells of 3 km
    assert info["shape"] == (57, 21, 21) and info["coast"] is None
    attrs, arr = read_vti(info["vti"])
    assert [float(s) for s in attrs["Spacing"].split()] == [3.0, 3.0, 0.5]
    assert set(arr) == {"log10_q", "q_gkg", "vt"}
    q, lq, vt = arr["q_gkg"], arr["log10_q"], arr["vt"]
    lv = xc.volume_levels_km(14.0)
    k5 = int(np.argmin(abs(lv - 5.0)))
    # total condensate, not QCLOUD+QICE: snow layer ~9.75-11.25 km is present at 2 g/kg
    k10 = int(np.argmin(abs(lv - 10.5)))
    assert q[k5, 15, 5] == pytest.approx(1.0, rel=1e-5)
    assert q[k10, 15, 5] == pytest.approx(2.0, rel=1e-5)
    # plateau (x >= 90 km is the last column at i=30): below its 1 km ground = 0
    assert q[0, :, -1].max() == 0.0
    # cutaway: camera due south -> rows south of the centre (y < 60 km) transparent, and the
    # 4.5 km (1.5 dx) slab behind the plane: y = 60, 63 too; the slice stays at y = 60
    assert (lq[:, :12, :] == -2.0).all()
    assert lq[k5, 12, 5] == pytest.approx(0.0, abs=1e-6)
    assert lq[k5, 15, 5] == pytest.approx(0.0, abs=1e-6)
    assert q[k5, 5, 5] == pytest.approx(1.0, rel=1e-5)        # q itself is not masked
    # solid-body vortex: vt = omega * r, r in metres
    r = np.hypot(15.0, 15.0) * 1000.0
    assert vt[k5, 15, 15] == pytest.approx(1e-3 * r, rel=1e-4)
    assert info["vt_max"] >= 40.0
    hdr = open(info["terrain"]).read(400)
    assert "DIMENSIONS 21 21 1" in hdr


def test_build_volume_whole_domain_interior(tmp_path, no_projection):
    w = _wrfout(str(tmp_path / "wrfout_d01_x"))
    info = xc.build_volume(w, str(tmp_path / "d"), bdy_width=5)
    assert info["shape"] == (57, 30, 30)
    assert info["extent_km"] == (15.0, 102.0, 15.0, 102.0)
    assert info["vexag"] == 1.0 and info["vt_max"] is None
    _, arr = read_vti(info["vti"])
    assert set(arr) == {"log10_q", "q_gkg"}


def test_build_volume_centre_outside_domain(tmp_path, no_projection):
    w = _wrfout(str(tmp_path / "wrfout_d01_x"))
    with pytest.raises(ValueError):
        xc.build_volume(w, str(tmp_path / "d"), center_latlon=(45.0, -40.0), box_km=60.0)


# ------------------------------------------------------------------ wrapper / pvbatch


def _fake_build(calls):
    def build(wrfout, prefix, **kw):
        calls.append(kw)
        for suf in ("_condensate.vti", "_terrain.vtk", "_coast.vtk"):
            open(prefix + suf, "w").write("x")
        storm = kw.get("center_latlon") is not None
        vexag = kw.get("vexag") or (8.0 if storm else 28.0)
        return dict(vti=prefix + "_condensate.vti", terrain=prefix + "_terrain.vtk",
                    coast=prefix + "_coast.vtk", center_km=(100.0, 200.0),
                    extent_km=(0, 600, 0, 600), width_km=600.0, dx_km=3.0, stride=1,
                    vexag=vexag, vt_max=70.0 if kw.get("cutaway") else None, ztop_km=18.0,
                    shape=(73, 201, 201))
    return build


def test_command_line_storm_cutaway(monkeypatch, tmp_path):
    calls, runs = [], []
    monkeypatch.setattr(paraview3d, "find_pvbatch", lambda: "/opt/pv/pvbatch")
    monkeypatch.setattr(xc, "build_volume", _fake_build(calls))

    def run(cmd, check=True):
        runs.append(cmd)
        open(cmd[5], "w").write("png")
        return 139, "", "Segmentation fault"         # teardown segfault after the PNG
    monkeypatch.setattr(tools, "run", run)
    out = str(tmp_path / "o.png")
    got = paraview3d.render_condensate_volume(
        "/r/wrfout_d01_2005-08-29_00:00:00", out, str(tmp_path / "w"),
        center_latlon=(25.0, -88.0), box_km=600.0, cutaway=True, title="katrina  d01  init x",
        save_state=str(tmp_path / "s.pvsm"))
    assert got == out
    cmd = runs[0]
    assert cmd[:3] == ["/opt/pv/pvbatch", "--force-offscreen-rendering",
                       os.path.abspath(RENDER_VOLUME)]
    assert cmd[3].endswith("wrfout_d01_2005-08-29_000000_condensate.vti")
    assert cmd[5] == out

    def arg(name, n=1):
        i = cmd.index(name)
        return cmd[i + 1:i + 1 + n] if n > 1 else cmd[i + 1]
    assert arg("--vexag") == "8"
    assert arg("--center", 2) == ["100.000", "200.000"]
    assert arg("--bearing") == "202.5" and arg("--elevation") == "25"
    assert float(arg("--focal-z")) == pytest.approx(5.0 * 8)
    assert float(arg("--opacity-unit")) == pytest.approx(0.75)
    assert arg("--ztop") == "18"
    rx, ry, lx, ly, tick = [float(v) for v in next(c for c in cmd if c.startswith("--ruler=")).split("=", 1)[1].split(",")]
    assert (rx, ry) == (0.0, 600.0) and tick == pytest.approx(15.0)   # NW corner
    assert lx < 0 and ly > 0
    assert "--cutaway" in cmd and arg("--vt-max") == "70"
    assert arg("--coast").endswith("_coast.vtk")
    assert arg("--save-state") == str(tmp_path / "s.pvsm")
    assert arg("--dx-note") == "dx = 3 km"
    title = arg("--title")
    assert title.splitlines()[0] == "katrina  d01  init x"
    assert ("Total condensate (all hydrometeors) volume rendering, opacity from 0.01 g/kg, "
            "volume top 18 km, vertical exaggeration ×8") == title.splitlines()[1]
    kw = calls[0]
    assert kw["center_latlon"] == (25.0, -88.0) and kw["cutaway"] is True
    assert kw["bearing_deg"] == 202.5 and kw["vexag"] is None
    assert os.path.exists(os.path.join(str(tmp_path / "w"),
                                       "wrfout_d01_2005-08-29_000000_volume.json"))


def test_whole_domain_view_and_missing_png(monkeypatch, tmp_path):
    calls, runs = [], []
    monkeypatch.setattr(paraview3d, "find_pvbatch", lambda: "pvbatch")
    monkeypatch.setattr(xc, "build_volume", _fake_build(calls))
    monkeypatch.setattr(tools, "run", lambda cmd, check=True: (runs.append(cmd), (1, "", "boom"))[1])
    out = tmp_path / "o.png"
    out.write_text("stale")                       # must be removed, not accepted
    with pytest.raises(paraview3d.ParaViewError, match="boom"):
        paraview3d.render_condensate_volume("/r/wrfout_d01_t", str(out), str(tmp_path / "w"))
    cmd = runs[0]
    assert cmd[cmd.index("--bearing") + 1] == "180"
    assert "--cutaway" not in cmd and "--vt-max" not in cmd
    assert calls[0]["center_latlon"] is None
    assert "×28" in cmd[cmd.index("--title") + 1]


def test_missing_pvbatch_is_tool_not_found(monkeypatch, tmp_path):
    monkeypatch.delenv("WTUNE_PARAVIEW_BIN", raising=False)
    monkeypatch.setattr(tools, "locate", lambda name, dirs: (None, ["nowhere"]))
    with pytest.raises(tools.ToolNotFoundError):
        paraview3d.render_condensate_volume("/r/wrfout_d01_t", str(tmp_path / "o.png"),
                                            str(tmp_path / "w"))


def test_cutaway_needs_centre(monkeypatch, tmp_path):
    monkeypatch.setattr(paraview3d, "find_pvbatch", lambda: "pvbatch")
    with pytest.raises(ValueError):
        paraview3d.render_condensate_volume("/r/w", str(tmp_path / "o.png"),
                                            str(tmp_path / "w"), cutaway=True)


def test_render_cloud_delegates_to_volume(monkeypatch):
    seen = {}

    def fake(wrfout, out_png, work_dir, **kw):
        seen.update(kw, args=(wrfout, out_png, work_dir))
        return out_png
    monkeypatch.setattr(paraview3d, "render_condensate_volume", fake)
    assert paraview3d.render_cloud("w", "o.png", "d", title="T", save_state="s") == "o.png"
    assert seen["args"] == ("w", "o.png", "d")
    assert seen["title"] == "T" and seen["save_state"] == "s" and seen["vexag"] is None
    assert "center_latlon" not in seen or seen["center_latlon"] is None


def test_view_params_defaults():
    info = dict(vexag=8.0, dx_km=3.0, stride=1, width_km=600.0, vt_max=None)
    p = paraview3d.view_params(info, storm=True, cutaway=True)
    assert p["bearing"] == 202.5 and p["elevation"] == 25.0
    assert p["opacity_unit"] == pytest.approx(0.75) and p["vt_max"] == 60.0
    assert p["zoom"] == 1.45 and p["focal_z"] == pytest.approx(40.0)
    info.update(width_km=3000.0, dx_km=12.0, stride=2, vexag=28.0, ztop_km=4.0)
    p = paraview3d.view_params(info, storm=False)
    assert p["bearing"] == 180.0 and p["opacity_unit"] == pytest.approx(6.0)
    assert p["elevation"] == 35.0 and p["focal_z"] == pytest.approx(1.6 * 28.0)
    assert paraview3d.opacity_unit_km(1.0) == 0.5                     # floor
    assert paraview3d.dx_note(info) == "dx = 12 km (volume every 2 points, 24 km)"


# ------------------------------------------------------------------ render script


def _render_volume_module():
    spec = importlib.util.spec_from_file_location("render_volume", RENDER_VOLUME)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_render_script_compiles_and_parses_wrapper_command(tmp_path):
    py_compile.compile(RENDER_VOLUME, cfile=str(tmp_path / "rv.pyc"), doraise=True)
    rv = _render_volume_module()            # paraview is imported only inside main()
    a = rv.parse_args(["x", "v.vti", "t.vtk", "o.png", "--vexag", "8", "--center", "-1", "2",
                       "--ruler=-300,250,-1.2e-17,0.38,15", "--ztop", "6",
                       "--bearing", "202.5", "--elevation", "25", "--cutaway", "--vt-max", "70",
                       "--title", "a\nb"])
    assert a.ruler == [-300.0, 250.0, -1.2e-17, 0.38, 15.0] and a.ztop == 6.0
    assert a.center == [-1.0, 2.0] and a.cutaway and a.vt_max == 70.0 and a.size == [1600, 1100]
    x, y, z = rv.camera_offset(202.5, 25.0)
    assert x < 0 and y < 0 and abs(x / y) == pytest.approx(np.tan(np.radians(22.5)))
    assert np.degrees(np.arcsin(z)) == pytest.approx(25.0)
    # opacity: 0 at 0.01 g/kg, 0.6-0.8 at 1 g/kg; colours not a rainbow (grey-blue only)
    op = dict(zip(rv.VOLUME_OPACITY[0::4], rv.VOLUME_OPACITY[1::4]))
    assert op[-2.0] == 0.0 and 0.6 <= op[0.0] <= 0.8
    rgb = np.array(rv.VOLUME_RGB).reshape(-1, 4)[:, 1:]
    assert (rgb[:, 2] >= rgb[:, 0]).all() and (np.ptp(rgb, axis=1) <= 0.2).all()


def test_inferno_fallback_points_and_preset_name():
    rv = _render_volume_module()
    assert rv.VT_PRESET == "Inferno"          # ParaView 6.1 name; "(matplotlib)" ones fail
    pts = np.array(rv.inferno_points(70.0)).reshape(-1, 4)
    assert pts[0, 0] == 0.0 and pts[-1, 0] == 70.0 and np.all(np.diff(pts[:, 0]) > 0)
    assert tuple(pts[0, 1:]) == (0.001, 0.0, 0.014) and pts[-1, 1] > 0.98   # black -> yellow
    src = open(RENDER_VOLUME).read()
    assert "ApplyPreset(VT_PRESET" in src and "if ok is False" in src


def test_ruler_ticks_every_2_km_true_height():
    rv = _render_volume_module()
    segs, labels = rv.ruler_geometry(10.0, 20.0, -1.0, 0.0, 7.0, 8.0, 5.0)
    assert segs[0] == ((10.0, 20.0, 0.0), (10.0, 20.0, 56.0))      # 7 km * 8
    assert [lb[1] for lb in labels] == ["0", "2", "4", "6 km"]
    assert [lb[0][2] for lb in labels] == [0.0, 16.0, 32.0, 48.0]
    assert segs[2] == ((10.0, 20.0, 16.0), (5.0, 20.0, 16.0))       # tick points along -x
    assert labels[1][0][:2] == (2.5, 20.0)


def test_ruler_placement_screen_left_corner():
    info = dict(extent_km=(0.0, 600.0, 0.0, 600.0), width_km=600.0)
    x, y, lx, ly, tick = paraview3d.ruler_placement(info, 180.0)   # camera south
    assert (x, y) == (0.0, 600.0) and (lx, ly) == (-1.0, 0.0)      # west side, far corner
    assert tick == pytest.approx(15.0)
    x, y, lx, ly, tick = paraview3d.ruler_placement(info, 202.5)
    assert (x, y) == (0.0, 600.0)


def test_shorelines_include_lakes_filtered_to_bbox(monkeypatch):
    from post import basemap
    lake = np.array([[-87.5, 42.0], [-86.5, 42.5], [-87.0, 44.0]])
    far = np.array([[10.0, 50.0], [11.0, 51.0]])
    coast = np.array([[-80.0, 25.0], [-79.0, 26.0]])
    files = {"coastline": "c.shp", "lakes": "l.shp"}
    segs = {"c.shp": [(-80.0, 25.0, -79.0, 26.0, coast)],
            "l.shp": [(-87.5, 42.0, -86.5, 44.0, lake), (10.0, 50.0, 11.0, 51.0, far)]}
    names = []
    monkeypatch.setattr(basemap, "shapefile_path",
                        lambda cat, name, res: (names.append((cat, name, res)), files.get(name))[1])
    monkeypatch.setattr(basemap, "_all_segments", lambda path: segs[path])
    got = xc._shoreline_lines((-90.0, -84.0, 40.0, 46.0))
    assert len(got) == 1 and np.array_equal(got[0], lake)
    assert names == [("physical", "coastline", "50m"), ("physical", "lakes", "50m")]
    assert len(xc._shoreline_lines((-90.0, -70.0, 20.0, 46.0))) == 2


def test_build_volume_writes_lake_shoreline(tmp_path, monkeypatch):
    import netCDF4
    w = _wrfout(str(tmp_path / "wrfout_d01_x"))
    with netCDF4.Dataset(w) as ds:
        lat, lon = np.asarray(ds["XLAT"][0]), np.asarray(ds["XLONG"][0])
    # a lake ring inside the grid; lon/lat -> km by inverting the synthetic grid
    ring = np.array([[lon[0, 25], lat[15, 0]], [lon[0, 30], lat[15, 0]],
                     [lon[0, 30], lat[25, 0]], [lon[0, 25], lat[15, 0]]])
    seen = []
    monkeypatch.setattr(xc, "_shoreline_lines", lambda bbox, **kw: (seen.append(bbox), [ring])[1])

    def to_km(lo, la):
        return ((np.asarray(lo) + 80.0) / 0.0285 * 3.0, (np.asarray(la) - 20.0) / 0.027 * 3.0)
    monkeypatch.setattr(xc, "_projected_axes",
                        lambda p, ny, nx, dx, dy: (np.arange(nx) * dx, np.arange(ny) * dy, to_km))
    info = xc.build_volume(w, str(tmp_path / "d"), bdy_width=5)
    lo0, lo1, la0, la1 = seen[0]
    assert lo0 < lon[0, 5] and lo1 > lon[0, 34] and la0 < lat[5, 0] and la1 > lat[34, 0]
    txt = open(info["coast"]).read().splitlines()
    assert "POINTS 4 float" in txt and "LINES 1 5" in txt
    x, y, z = (float(v) for v in txt[5].split())
    assert (x, y) == pytest.approx((75.0, 45.0), abs=0.1)
    # over sea (HGT 0) the shoreline is lifted 50 m: z = 0.05 km * vexag
    assert z == pytest.approx(0.05 * info["vexag"], abs=1e-3)


def test_sample_bilinear_and_drape_shoreline_on_terrain():
    x = np.array([0.0, 3.0, 6.0])
    y = np.array([10.0, 13.0])
    f = 100.0 * x[None, :] + 1000.0 * (y[:, None] - 10.0)            # linear: exact
    got = xc.sample_bilinear(x, y, f, np.array([1.5, 4.5, 99.0]), np.array([11.5, 10.0, 13.0]))
    assert np.allclose(got, [150.0 + 1500.0, 450.0, 600.0 + 3000.0])  # 99 -> clipped to edge
    hgt = np.array([[0.0, 176.0, 176.0], [-5.0, 176.0, 176.0]])         # sea | lake surface
    pl = xc.drape_polylines([np.array([[0.0, 10.0], [6.0, 13.0]])], x, y, hgt, 13.0)
    assert pl[0].shape == (2, 3)
    assert pl[0][0, 2] == pytest.approx(0.050 * 13.0)                  # sea: 50 m lift
    assert pl[0][1, 2] == pytest.approx(0.226 * 13.0)                  # Lake Michigan + 50 m


def test_write_vtk_polylines_per_vertex_z(tmp_path):
    p = tmp_path / "c.vtk"
    xc.write_vtk_polylines(str(p), [np.array([[1.0, 2.0, 2.938], [3.0, 4.0, 0.65]])])
    txt = p.read_text().splitlines()
    assert "1.000 2.000 2.938" in txt and "3.000 4.000 0.650" in txt


def _timed_wrfout(path, valid=b"2019-01-31_00:00:00"):
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.SIMULATION_START_DATE = "2019-01-28_00:00:00"
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.frombuffer(valid, "S1")
    return path


def test_title_time_label_like_2d_figures(tmp_path):
    w = _timed_wrfout(str(tmp_path / "wrfout_d02_2019-01-31_00:00:00"))
    lab = "init 2019-01-28 00Z  valid 2019-01-31 00Z (F072)"
    t = paraview3d.title_with_time_label("p1  d02  total condensate volume  2019-01-31_00:00:00", w)
    assert t == "p1  d02  total condensate volume  " + lab
    assert paraview3d.title_with_time_label("p1  d02  volume", w) == "p1  d02  volume  " + lab
    assert paraview3d.title_with_time_label("", w) == lab
    already = "p1  d02  valid 2019-01-31 00Z"
    assert paraview3d.title_with_time_label(already, w) == already
    # unreadable file: the raw time still becomes a 'valid ...' label, nothing crashes
    assert paraview3d.title_with_time_label("x 2019-01-31_00:00:00", "/nope") == \
        "x valid 2019-01-31 00Z"
    assert paraview3d.title_with_time_label("x", "/nope") == "x"
