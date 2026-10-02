"""post/recipes/tc_structure.py -- 2-D views of tropical-cyclone STRUCTURE (eye, eyewall,
rainbands, warm core, secondary circulation), run when the case type includes "tc". The track
figures of post/tc.py answer "where"; these answer "what does the storm look like".

Storm centre, one per frame: the chosen track post/tc.py wrote (tc/track_domains_<label>.csv,
TempestExtremes on CDO-smoothed SLP, finest domain holding the storm) when that file has the
frame's time; otherwise the minimum of SLP smoothed over 3x3 grid cells (scipy uniform_filter)
inside the domain interior (spec_bdy_width edge cells excluded). The centre is a grid cell
(+- dx/2). Domain: the track's domain at the last frame, else the finest domain whose smoothed-
SLP minimum lies at least 3 cells inside its interior, else d01.

Storm-relative wind: radial/tangential components come from wrf-python ua/va rotated about the
centre IN THE NATIVE PROJECTED GRID: x/y offsets are (i - ic) dx, (j - jc) dx and ua/va are
grid-relative (WRF's U/V are along the grid axes; uvmet would rotate them to earth axes), so
vr = (u x + v y) / r and vt = (-u y + v x) / r are exact on the projected plane (the map
factor, ~1 over a storm-sized area, is ignored). Height fields use wrf.interplevel (z MSL,
0-18 km every 0.5 km; height AGL for the 2 km reflectivity) and wrf.vertcross.

  tc_struct_reflectivity_2km     A1  dbz at 2 km AGL, storm-centred 600x600 km, NWS table,
                                     10 m wind 34/50/64 kt, RMW ring, centre
  tc_struct_ir_proxy             A2  OLR-derived Tb (post/recipes/cloud.plot_ir_proxy); not drawn
                                     when the cloud set already draws it for that domain
  tc_struct_rz_wind              A3a azimuthal-mean Vt + (Vr, w x10) vectors + RMW(z)
  tc_struct_rz_warmcore          A3b T' = <T>(r,z) - mean over the 500-1000 km annulus
                                     (largest full annulus when the domain is smaller) + theta
  tc_struct_rz_condensate        A3c total condensate + w + 0 degC
  tc_struct_xsec                 A4  W-E and S-N sections through the centre, +-300 km
  tc_struct_warmcore_hovmoller   A5  eye T'(z, t)
  tc_struct_condensate_3d        B5  ParaView volume, storm-centred cutaway (another module)
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime

from post import tools
from post.extract import list_frames
from post.recipes import cloud as cl

BOX_HALF_KM = 300.0                 # A1: 600 km x 600 km
R_MAX_KM = 300.0                    # A3 radius window
Z_TOP_KM = 18.0
DZ_KM = 0.5
XSEC_DZ_KM = 0.25
WIND_RADII_KT = (34, 50, 64)        # NHC wind-radii thresholds
ENV_ANNULUS_KM = (500.0, 1000.0)
HOVMOLLER_MAX_FRAMES = 60
W_VECTOR_SCALE = 10.0               # w drawn x10 in the (Vr, w) vectors
SLP_SMOOTH_CELLS = 3
CENTRE_MARGIN_CELLS = 3
WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"


# ---------------------------------------------------------------- numerics

def grid_offsets_km(shape, jc, ic, dx_km):
    """(X, Y): native-grid offsets (km) of every cell from the centre cell (jc, ic)."""
    import numpy as np
    ny, nx = shape
    x = (np.arange(nx) - ic) * dx_km
    y = (np.arange(ny) - jc) * dx_km
    return np.meshgrid(x, y)


def radial_tangential(u, v, X, Y):
    """(vr, vt) of grid-relative (u, v) about the origin of X/Y: vr outward-positive, vt
    counter-clockwise (cyclonic in the NH) positive. 0 at r = 0. u/v may have leading dims."""
    import numpy as np
    r = np.hypot(X, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(r > 0, X / r, 0.0)
        sy = np.where(r > 0, Y / r, 0.0)
    u = np.asarray(u, "f8")
    v = np.asarray(v, "f8")
    return u * cx + v * sy, -u * sy + v * cx


def radius_edges(dx_km, rmax_km=R_MAX_KM):
    """Bin edges 0, dx, 2dx, ... <= rmax (bins one grid length wide)."""
    import numpy as np
    n = max(1, int(math.floor(rmax_km / dx_km + 1e-9)))
    return np.arange(n + 1) * float(dx_km)


def azimuthal_mean(field, r, edges, dx_km=None, min_coverage=0.5):
    """NaN-aware mean of field (..., ny, nx) in radius bins `edges` -> (..., nbins). With dx_km,
    a bin whose valid cell count is below min_coverage x the full-ring count (pi (r2^2 - r1^2) /
    dx^2, checked for rings of 8+ expected cells) is NaN (ring cut by the domain edge). Bins with
    no valid cell are NaN."""
    import numpy as np
    f = np.asarray(field, "f8")
    lead = f.shape[:-2]
    f2 = f.reshape(-1, f.shape[-2] * f.shape[-1])
    nb = len(edges) - 1
    idx = np.digitize(np.asarray(r, "f8").ravel(), edges) - 1
    inb = (idx >= 0) & (idx < nb)
    out = np.full((f2.shape[0], nb), np.nan)
    expected = None
    if dx_km:
        expected = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / float(dx_km) ** 2
    for k in range(f2.shape[0]):
        row = f2[k]
        ok = inb & np.isfinite(row)
        s = np.bincount(idx[ok], weights=row[ok], minlength=nb)
        c = np.bincount(idx[ok], minlength=nb).astype("f8")
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, s / c, np.nan)
        if expected is not None:
            # the innermost rings hold a handful of cells whose count jitters around the
            # area estimate: only rings expected to hold 8+ cells are coverage-checked
            m = np.where((c >= min_coverage * expected) | (expected < 8.0), m, np.nan)
        out[k] = m
    return out.reshape(lead + (nb,)) if lead else out[0]


def full_annulus_radius_km(shape, jc, ic, dx_km, bdy_width=0):
    """Radius (km) of the largest circle about (jc, ic) that lies wholly in the interior."""
    ny, nx = shape
    bw = int(bdy_width)
    cells = min(ic - bw, nx - 1 - bw - ic, jc - bw, ny - 1 - bw - jc)
    return max(0.0, cells * float(dx_km))


def environment_annulus(r_full_km, want=ENV_ANNULUS_KM):
    """(r_in, r_out, reduced): the wanted 500-1000 km annulus when the domain holds it, else
    the largest full annulus available with the same inner/outer ratio (r_out = r_full)."""
    if r_full_km >= want[1]:
        return float(want[0]), float(want[1]), False
    if r_full_km <= 0:
        raise ValueError("storm centre on the domain boundary: no environment annulus")
    return r_full_km * want[0] / want[1], float(r_full_km), True


def annulus_mean(field_levels, r, annulus):
    """Mean of (nlev, ny, nx) over cells with r_in <= r <= r_out -> (nlev,)."""
    import numpy as np
    mask = (np.asarray(r) >= annulus[0]) & (np.asarray(r) <= annulus[1])
    f = np.asarray(field_levels, "f8")[:, mask]
    with np.errstate(invalid="ignore"):
        return np.nanmean(f, axis=1) if f.size else np.full(f.shape[0], np.nan)


def annulus_label(annulus):
    r_in, r_out, reduced = annulus
    if reduced:
        return ("environment = %.0f-%.0f km annulus (largest full annulus the domain holds; "
                "500-1000 km does not fit)" % (r_in, r_out))
    return "environment = %.0f-%.0f km annulus" % (r_in, r_out)


def rmw_profile(vt_rz, r_centres):
    """Radius of max azimuthal-mean Vt per level (NaN where the level is all NaN)."""
    import numpy as np
    a = np.asarray(vt_rz, "f8")
    if a.ndim == 1:
        a = a[None]
    out = np.full(a.shape[0], np.nan)
    for k in range(a.shape[0]):
        if np.isfinite(a[k]).any():
            out[k] = r_centres[int(np.nanargmax(a[k]))]
    return out


def centre_from_slp(slp, bdy_width=0, size=SLP_SMOOTH_CELLS):
    """(j, i) of the minimum of SLP smoothed over size x size cells, interior only."""
    import numpy as np
    from scipy.ndimage import uniform_filter
    a = np.asarray(slp, "f8")
    sm = uniform_filter(np.where(np.isfinite(a), a, np.nanmax(a)), size=size, mode="nearest")
    ny, nx = a.shape
    bw = int(bdy_width) if 2 * int(bdy_width) + 1 < min(ny, nx) else 0
    sub = sm[bw:ny - bw, bw:nx - bw]
    j, i = np.unravel_index(int(np.argmin(sub)), sub.shape)
    return int(j + bw), int(i + bw)


def read_track_csv(path):
    """{time_str: (lat, lon, domain)} from post/tc.py's track_domains_<label>.csv."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                dom = int(str(row.get("domain", "d01")).lstrip("d") or 1)
                out[row["time"]] = (float(row["lat"]), float(row["lon"]), dom)
            except (KeyError, ValueError):
                continue
    return out


def hovmoller_frames(all_times, every=None, times=None, cap=HOVMOLLER_MAX_FRAMES):
    """Times for the warm-core Hovmoller: the --every/--times cadence when given, else every
    frame when there are at most `cap`, else `cap` evenly spaced frames (first and last kept)."""
    import numpy as np
    if every or times:
        from post.timeseries import select_times
        return select_times(all_times, every=every, times=times)
    if len(all_times) <= cap:
        return list(all_times)
    idx = np.unique(np.round(np.linspace(0, len(all_times) - 1, cap)).astype(int))
    return [all_times[k] for k in idx]


def hovmoller_assemble(columns, nlev):
    """columns: [(time_str, profile (nlev,) or None)] -> (datetimes, (nlev, nt) array, NaN for
    a frame without a centre)."""
    import numpy as np
    times = [datetime.strptime(t, WRF_TIME_FMT) for t, _ in columns]
    arr = np.full((nlev, len(columns)), np.nan)
    for k, (_t, prof) in enumerate(columns):
        if prof is not None:
            arr[:, k] = np.asarray(prof, "f8")[:nlev]
    return times, arr


def eye_anomaly(t_levels, r, dx_km, annulus):
    """Eye T'(z): mean T over r <= dx (centre cell + 4 neighbours) minus the annulus mean."""
    import numpy as np
    eye = np.asarray(r) <= float(dx_km) + 1e-6
    f = np.asarray(t_levels, "f8")
    with np.errstate(invalid="ignore"):
        return np.nanmean(f[:, eye], axis=1) - annulus_mean(f, r, annulus)


def symmetric_levels(a, step=1.0, cap=20.0):
    import numpy as np
    m = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else step
    m = min(max(step, math.ceil(m / step) * step), cap)
    return np.arange(-m, m + step / 2.0, step)


# ---------------------------------------------------------------- centre / domain

def frame_centre(ds, time_str, track, bdy_width):
    """(j, i, source) of the storm centre in this frame."""
    import numpy as np
    wrf_mod = cl._wrf()
    ny, nx = np.shape(ds.variables["XLAT"][0])
    if time_str in track:
        lat, lon = track[time_str][:2]
        x, y = (int(v) for v in np.asarray(wrf_mod.ll_to_xy(ds, lat, lon, meta=False)).ravel()[:2])
        if 0 <= x < nx and 0 <= y < ny:
            return y, x, "minimum-SLP track"
    slp = np.asarray(wrf_mod.getvar(ds, "slp", timeidx=0, meta=False), "f8")
    j, i = centre_from_slp(slp, bdy_width)
    return j, i, "minimum SLP (smoothed)"


def structure_domain(run_dir, case_info, track, bdy_width):
    """The domain the structure figures are drawn on (see the module docstring)."""
    import netCDF4
    if track:
        last = track[max(track)]
        if list_frames(run_dir, domain=last[2]):
            return last[2]
    doms = sorted({int(d["domain"]) for d in case_info.get("domains", [])} | {1})
    for dom in reversed(doms):
        frames = list_frames(run_dir, domain=dom)
        if not frames:
            continue
        if dom == 1:
            return 1
        with netCDF4.Dataset(frames[-1][1]) as ds:
            j, i, _ = frame_centre(ds, frames[-1][0], {}, bdy_width)
            ny, nx = ds.variables["XLAT"][0].shape
        if full_annulus_radius_km((ny, nx), j, i, 1.0, bdy_width) >= CENTRE_MARGIN_CELLS:
            return dom
    return 1


# ---------------------------------------------------------------- A1

def reflectivity_2km_figure(path, time_str, centre, out_path, run_label, domain):
    import numpy as np
    import cartopy.crs as ccrs
    import netCDF4
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    from post.conventions import DBZ_LEVELS, KT_PER_MS, ds_time_label, reflectivity_cmap
    wrf_mod = cl._wrf()
    jc, ic, source = centre
    with netCDF4.Dataset(path) as ds:
        dx = cl.dx_km_of(ds)
        dbz = np.asarray(wrf_mod.getvar(ds, "dbz", timeidx=0, meta=False), "f8")
        hagl = np.asarray(wrf_mod.getvar(ds, "height_agl", timeidx=0, units="m", meta=False), "f8")
        dbz2 = cl.interp_heights(dbz, hagl, [2000.0])[0]
        u10 = np.asarray(ds.variables["U10"][0], "f8")
        v10 = np.asarray(ds.variables["V10"][0], "f8")
        lats = np.asarray(ds.variables["XLAT"][0], "f8")
        lons = np.asarray(ds.variables["XLONG"][0], "f8")
        crs = wrf_mod.get_cartopy(wrfin=ds)
        ny, nx = lats.shape
        X, Y = grid_offsets_km((ny, nx), jc, ic, dx)
        r = np.hypot(X, Y)
        _, vt10 = radial_tangential(u10, v10, X, Y)
        edges = radius_edges(dx)
        rc = 0.5 * (edges[1:] + edges[:-1])
        vt_r = azimuthal_mean(vt10, r, edges, dx_km=dx)
        rmw = float(rmw_profile(vt_r, rc)[0])
        spd_kt = np.hypot(u10, v10) * KT_PER_MS
        half = int(round(BOX_HALF_KM / dx))
        j0, j1 = max(0, jc - half), min(ny, jc + half + 1)
        i0, i1 = max(0, ic - half), min(nx, ic + half + 1)
        sl = (slice(j0, j1), slice(i0, i1))
        cm = reflectivity_cmap()
        norm = BoundaryNorm(DBZ_LEVELS, cm.N, extend="max")
        fig, (ax,) = cl.plan_figure(crs)
        pc = ccrs.PlateCarree()
        X_, Y_ = cl.native_xy(crs, lats[sl], lons[sl])
        field = np.where(dbz2[sl] < DBZ_LEVELS[0], np.nan, dbz2[sl])
        if np.isfinite(field).any():
            cf = ax.contourf(X_, Y_, field, levels=DBZ_LEVELS, cmap=cm, norm=norm,
                             extend="max", transform=crs)
        else:
            from matplotlib.cm import ScalarMappable
            cf = ScalarMappable(norm=norm, cmap=cm)
        fig.colorbar(cf, ax=ax, shrink=0.9, ticks=DBZ_LEVELS,
                     label="reflectivity at 2 km AGL (dBZ, NWS table)")
        handles = []
        box_kt = spd_kt[sl]
        top_kt = float(np.nanmax(box_kt)) if np.isfinite(box_kt).any() else -np.inf
        for kt, lw in zip(WIND_RADII_KT, (0.8, 1.3, 1.9)):
            if top_kt >= kt:            # legend lists only the wind-radius contours drawn
                ax.contour(X_, Y_, box_kt, levels=[kt], colors="k",
                           linewidths=lw, transform=crs)
                handles.append(Line2D([], [], color="k", lw=lw, label="10 m wind %d kt" % kt))
        no_kt = "" if top_kt >= WIND_RADII_KT[0] else "; no 10 m wind >= %d kt" % WIND_RADII_KT[0]
        if np.isfinite(rmw):
            th = np.linspace(0, 2 * np.pi, 121)
            rx = ic + rmw / dx * np.cos(th)
            ry = jc + rmw / dx * np.sin(th)
            rl = cl.line_values(lats, np.column_stack([rx, ry]))
            ro = cl.line_values(lons, np.column_stack([rx, ry]))
            ax.plot(ro, rl, "--", color="magenta", lw=1.4, transform=pc)
            handles.append(Line2D([], [], color="magenta", ls="--", lw=1.4,
                                  label="RMW %.0f km (10 m azimuthal mean)" % rmw))
        ax.plot([lons[jc, ic]], [lats[jc, ic]], marker="+", color="k", ms=14, mew=2.2,
                transform=pc, zorder=7)
        handles.append(Line2D([], [], color="k", marker="+", ls="none", ms=10, label="centre"))
        cl.finish_plan(ax, ds, lats[sl], lons[sl], crs=crs)
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=7,
                   frameon=False)
        note = cl.grid_limit_note(dx)
        ax.set_title(cl.wrap_title(
            "%s  d%02d  reflectivity at 2 km AGL, storm-centred %.0f x %.0f km (clipped to "
            "domain) + 10 m wind 34/50/64 kt + RMW; centre: %s%s%s\n%s"
            % (run_label, domain, 2 * BOX_HALF_KM, 2 * BOX_HALF_KM, source, no_kt,
               ("\n" + note) if note else "", ds_time_label(ds, time_str))), fontsize=9)
        fig.savefig(out_path)
        cl._plt().close(fig)
    return out_path, rmw


# ---------------------------------------------------------------- A3

def rz_analysis(path, centre, bdy_width):
    """Azimuthal means on (z, r) about the centre for the A3 figures."""
    import numpy as np
    import netCDF4
    from post import condensate
    wrf_mod = cl._wrf()
    jc, ic, source = centre
    levels = np.arange(0.0, Z_TOP_KM * 1000.0 + 1.0, DZ_KM * 1000.0)
    with netCDF4.Dataset(path) as ds:
        dx = cl.dx_km_of(ds)
        z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=0, units="m", meta=False), "f8")
        hgt = np.asarray(ds.variables["HGT"][0], "f8")

        def lev(name, **kw):
            return cl.interp_heights(np.asarray(wrf_mod.getvar(ds, name, timeidx=0, meta=False,
                                                               **kw), "f8"), z, levels, hgt)
        ua, va, wa = lev("ua"), lev("va"), lev("wa")
        tk, th = lev("tk"), lev("th")
        q = cl.interp_heights(condensate.total_condensate(ds, 0), z, levels, hgt)
    ny, nx = hgt.shape
    X, Y = grid_offsets_km((ny, nx), jc, ic, dx)
    r = np.hypot(X, Y)
    vr, vt = radial_tangential(ua, va, X, Y)
    edges = radius_edges(dx)
    rc = 0.5 * (edges[1:] + edges[:-1])
    am = lambda f: azimuthal_mean(f, r, edges, dx_km=dx)      # noqa: E731
    annulus = environment_annulus(full_annulus_radius_km((ny, nx), jc, ic, dx, bdy_width))
    t_env = annulus_mean(tk, r, annulus)
    vt_rz = am(vt)
    return dict(dx=dx, r=rc, z_km=levels / 1000.0, vt=vt_rz, vr=am(vr), w=am(wa),
                tprime=am(tk) - t_env[:, None], theta=am(th), tc=am(tk) - 273.15,
                q_gkg=am(q) * 1000.0, ter_km=am(hgt[None])[0] / 1000.0,
                rmw=rmw_profile(vt_rz, rc), annulus=annulus, source=source)


def _rz_axes():
    plt = cl._plt()
    fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    return plt, fig, ax


def _rz_finish(fig, ax, rz, title, out_path, handles=()):
    import numpy as np
    ter = np.nan_to_num(rz["ter_km"], nan=0.0)
    handles = list(handles)
    if np.nanmax(ter) > 0:
        ax.fill_between(rz["r"], 0, ter, facecolor="0.85", edgecolor="0.4", hatch="////",
                        zorder=6)
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="0.85", edgecolor="0.4", hatch="////",
                             label="below ground (azimuthal-mean terrain)"))
    ax.set_xlim(0, R_MAX_KM)
    ax.set_ylim(0, Z_TOP_KM)
    ax.set_xlabel("radius from centre (km)")
    ax.set_ylabel("height MSL (km)")
    cl.figure_legend(fig, handles, loc="outside lower center", ncol=max(min(len(handles), 4), 1),
                     fontsize=7, frameon=False)
    ax.set_title(cl.wrap_title(title), fontsize=9)
    fig.savefig(out_path)
    cl._plt().close(fig)
    return out_path


def _rz_head(run_label, domain, what, rz, tlabel):
    note = cl.grid_limit_note(rz["dx"])
    return ("%s  d%02d  %s\nazimuthal mean about the centre (%s); %g km radius bins"
            "%s\n%s" % (run_label, domain, what, rz["source"], rz["dx"],
                         ("; " + note) if note else "", tlabel))


def rz_wind_figure(rz, out_path, run_label, domain, tlabel):
    import numpy as np
    from matplotlib.lines import Line2D
    plt, fig, ax = _rz_axes()
    vt = rz["vt"]
    top = max(5.0, math.ceil(np.nanmax(vt) / 5.0) * 5.0) if np.isfinite(vt).any() else 5.0
    levels = np.arange(0.0, top + 2.5, 5.0)
    if np.isfinite(vt).any():
        cf = ax.contourf(rz["r"], rz["z_km"], vt, levels=levels, cmap="YlOrRd", extend="both")
        fig.colorbar(cf, ax=ax, label="tangential wind V$_t$ (m s$^{-1}$, 5 m s$^{-1}$ steps)")
    has_neg = bool(np.isfinite(vt).any() and np.nanmin(vt) <= -5.0)
    if has_neg:
        neg = np.arange(-5.0 * math.ceil(-np.nanmin(vt) / 5.0), -4.9, 5.0)
        ax.contour(rz["r"], rz["z_km"], vt, levels=neg, colors="k", linewidths=0.6,
                   linestyles="dashed")
    sr = max(1, len(rz["r"]) // 25)
    sz = 2
    q = ax.quiver(rz["r"][::sr], rz["z_km"][::sz], rz["vr"][::sz, ::sr],
                  W_VECTOR_SCALE * rz["w"][::sz, ::sr], angles="uv", scale_units="width",
                  scale=200.0, width=0.0022, color="0.15")
    ax.quiverkey(q, 0.95, 0.95, 10.0, "10 m s$^{-1}$ (V$_r$); w drawn x%g" % W_VECTOR_SCALE,
                 labelpos="W", coordinates="axes", fontproperties={"size": 7},
                 labelcolor="k", zorder=8)
    handles = []
    if np.isfinite(np.asarray(rz["rmw"], "f8")).any():
        ax.plot(rz["rmw"], rz["z_km"], "b-", lw=1.6)
        handles.append(Line2D([], [], color="b", lw=1.6, label="RMW(z)"))
    handles.append(Line2D([], [], color="0.15", marker=r"$\rightarrow$", ls="none",
                          label="(V$_r$, %g x w) vectors" % W_VECTOR_SCALE))
    if has_neg:
        handles.append(Line2D([], [], color="k", ls="--", lw=0.6,
                              label="V$_t$ < 0 (every 5 m s$^{-1}$)"))
    return _rz_finish(fig, ax, rz, _rz_head(run_label, domain, "tangential wind (fill), "
                                            "secondary circulation (V$_r$, w) vectors, RMW(z)",
                                            rz, tlabel), out_path, handles)


def rz_warmcore_figure(rz, out_path, run_label, domain, tlabel):
    import numpy as np
    from matplotlib.lines import Line2D
    plt, fig, ax = _rz_axes()
    tp = rz["tprime"]
    levels = symmetric_levels(tp)
    cf = ax.contourf(rz["r"], rz["z_km"], tp, levels=levels, cmap="RdBu_r", extend="both")
    fig.colorbar(cf, ax=ax, label="T' (K, 1 K steps)")
    th = rz["theta"]
    if np.isfinite(th).any():
        tl = np.arange(10.0 * math.floor(np.nanmin(th) / 10.0), np.nanmax(th) + 10.0, 10.0)
        cs = ax.contour(rz["r"], rz["z_km"], th, levels=tl, colors="k", linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
    handles = ([Line2D([], [], color="k", lw=0.6, label="theta (K, every 10 K)")]
               if np.isfinite(th).any() else [])
    return _rz_finish(fig, ax, rz, _rz_head(
        run_label, domain, "temperature anomaly T' = <T>(r,z) minus the %s at the same height"
        % annulus_label(rz["annulus"]), rz, tlabel), out_path, handles)


def rz_condensate_figure(rz, out_path, run_label, domain, tlabel):
    plt, fig, ax = _rz_axes()
    cf = cl.condensate_fill(ax, rz["r"], rz["z_km"], rz["q_gkg"])
    cl.condensate_colorbar(fig, cf, ax)
    handles = cl.draw_w_contours(ax, rz["r"], rz["z_km"], rz["w"])
    handles += cl.draw_isotherms(ax, rz["r"], rz["z_km"], rz["tc"], [(0.0, "solid", 1.8, "0 °C")])
    return _rz_finish(fig, ax, rz, _rz_head(
        run_label, domain, "total condensate (all species, log 0.01-1 g/kg) + w + 0 °C",
        rz, tlabel), out_path, handles)


# ---------------------------------------------------------------- A4

def xsec_figure(path, time_str, centre, out_path, run_label, domain):
    import numpy as np
    import netCDF4
    from matplotlib.colors import BoundaryNorm
    from post.conventions import DBZ_LEVELS, ds_time_label, reflectivity_cmap
    wrf_mod = cl._wrf()
    jc, ic, source = centre
    levels = np.arange(0.0, Z_TOP_KM * 1000.0 + 1.0, XSEC_DZ_KM * 1000.0)
    with netCDF4.Dataset(path) as ds:
        dx = cl.dx_km_of(ds)
        z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=0, units="m", meta=False), "f8")
        dbz = np.asarray(wrf_mod.getvar(ds, "dbz", timeidx=0, meta=False), "f8")
        wa = np.asarray(wrf_mod.getvar(ds, "wa", timeidx=0, meta=False), "f8")
        tc = np.asarray(wrf_mod.getvar(ds, "tc", timeidx=0, meta=False), "f8")
        hgt = np.asarray(ds.variables["HGT"][0], "f8")
        tlabel = ds_time_label(ds, time_str)
    ny, nx = hgt.shape
    cm = reflectivity_cmap()
    norm = BoundaryNorm(DBZ_LEVELS, cm.N, extend="max")
    plt = cl._plt()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.6), layout="constrained")
    cf = None
    w_all, t_all = [], []
    iso = [(0.0, "solid", 1.6, "0 °C"), (-20.0, "dashed", 1.0, "-20 °C")]
    for ax, (name, ux, uy, sign) in zip(axes, (("W-E (grid x)", 1.0, 0.0, "east"),
                                               ("S-N (grid y)", 0.0, 1.0, "north"))):
        start, end, _ = cl.line_endpoints(ic, jc, ux, uy, BOX_HALF_KM / dx, nx, ny)
        ds_, xy = cl.section(dbz, z, start, end, levels)
        ws, _ = cl.section(wa, z, start, end, levels)
        ts, _ = cl.section(tc, z, start, end, levels)
        x = cl.signed_distance_km(xy, ic, jc, ux, uy, dx)
        y = levels / 1000.0
        f = np.where(ds_ < DBZ_LEVELS[0], np.nan, ds_)
        if np.isfinite(f).any():
            cf = ax.contourf(x, y, f, levels=DBZ_LEVELS, cmap=cm, norm=norm, extend="max")
        cl.draw_w_contours(ax, x, y, ws)
        cl.draw_isotherms(ax, x, y, ts, iso)
        w_all.append(ws)
        t_all.append(ts)
        cl.draw_terrain(ax, x, cl.line_values(hgt, xy) / 1000.0)
        ax.set_xlim(-BOX_HALF_KM, BOX_HALF_KM)
        ax.set_ylim(0, Z_TOP_KM)
        ax.set_ylabel("height MSL (km)")
        ax.set_xlabel("distance from centre (km, + = %s)" % sign)
        ax.set_title(name, fontsize=8)
    if cf is None:
        from matplotlib.cm import ScalarMappable
        cf = ScalarMappable(norm=norm, cmap=cm)
    fig.colorbar(cf, ax=list(axes), ticks=DBZ_LEVELS, label="reflectivity (dBZ, NWS table)")
    # legend from BOTH panels, listing only the contour levels actually drawn
    handles = (cl.w_contour_handles(cl.flat_all(w_all))
               + cl.isotherm_handles(cl.flat_all(t_all), iso))
    cl.figure_legend(fig, handles, loc="outside lower center", ncol=max(len(handles), 1),
                     fontsize=7, frameon=False)
    note = cl.grid_limit_note(dx)
    fig.suptitle(cl.wrap_title(
        "%s  d%02d  sections through the storm centre (%s), +-%.0f km (clipped to domain): "
        "reflectivity + w + 0/-20 °C, terrain grey; lines follow the model grid axes%s\n%s"
        % (run_label, domain, source, BOX_HALF_KM, ("\n" + note) if note else "", tlabel), 120),
        fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- A5

def eye_profile(path, time_str, track, bdy_width, levels):
    """(T'(z) at the eye, annulus) for one frame, or (None, None) without a usable centre."""
    import numpy as np
    import netCDF4
    wrf_mod = cl._wrf()
    with netCDF4.Dataset(path) as ds:
        dx = cl.dx_km_of(ds)
        jc, ic, _src = frame_centre(ds, time_str, track, bdy_width)
        z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=0, units="m", meta=False), "f8")
        hgt = np.asarray(ds.variables["HGT"][0], "f8")
        tk = cl.interp_heights(np.asarray(wrf_mod.getvar(ds, "tk", timeidx=0, meta=False), "f8"),
                               z, levels, hgt)
    shape = hgt.shape
    try:
        annulus = environment_annulus(full_annulus_radius_km(shape, jc, ic, dx, bdy_width))
    except ValueError:
        return None, None
    X, Y = grid_offsets_km(shape, jc, ic, dx)
    return eye_anomaly(tk, np.hypot(X, Y), dx, annulus), annulus


def hovmoller_figure(run_dir, domain, frame_times, track, bdy_width, out_path, run_label):
    import numpy as np
    import matplotlib.dates as mdates
    from post.conventions import path_time_label
    from post.extract import frame_path
    levels = np.arange(0.0, Z_TOP_KM * 1000.0 + 1.0, DZ_KM * 1000.0)
    cols, reduced = [], []
    for t in frame_times:
        prof, ann = eye_profile(frame_path(run_dir, domain, t), t, track, bdy_width, levels)
        cols.append((t, prof))
        if ann is not None:
            reduced.append(ann)
    times, arr = hovmoller_assemble(cols, len(levels))
    plt = cl._plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.2), layout="constrained")
    cf = ax.contourf(times, levels / 1000.0, arr, levels=symmetric_levels(arr), cmap="RdBu_r",
                     extend="both")
    fig.colorbar(cf, ax=ax, label="eye T' (K, 1 K steps)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
    ax.set_xlabel("valid time (day/hour UTC)")
    ax.set_ylabel("height MSL (km)")
    ax.set_ylim(0, Z_TOP_KM)
    if reduced and any(a[2] for a in reduced):
        env = ("environment = largest full annulus per frame (%.0f-%.0f km at the last frame; "
               "500-1000 km does not fit)" % (reduced[-1][0], reduced[-1][1]))
    else:
        env = "environment = 500-1000 km annulus"
    ax.set_title(cl.wrap_title(
        "%s  d%02d  warm-core Hovmoller: eye T' = mean T within r <= dx of the centre minus the "
        "%s, per frame (%d frames)\n%s"
        % (run_label, domain, env, len(frame_times),
           path_time_label(frame_path(run_dir, domain, frame_times[-1]), frame_times[-1]))),
        fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- figure set

_ERRORS = (tools.ToolNotFoundError, ImportError, KeyError, ValueError, IndexError)


def _reason(e):
    return "%s: %s" % (type(e).__name__, e)


def figures(run_dir, run_label, out_dir, case_info, track_csv=None, every=None, times=None,
            paraview=True):
    """TC structure figure records (post.recipe fig()/missing()) for one run."""
    import netCDF4
    from post.conventions import path_time_label
    from post.recipe import fig, missing
    os.makedirs(out_dir, exist_ok=True)
    bdy_width = cl.bdy_width_for(run_dir)
    track = read_track_csv(track_csv)
    out = []
    try:
        domain = structure_domain(run_dir, case_info, track, bdy_width)
    except _ERRORS + (OSError,) as e:
        return [missing("tc_struct_reflectivity_2km", "wrfout / storm centre", _reason(e),
                        run=run_label)]
    dom = "d%02d" % domain
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return [missing("tc_struct_reflectivity_2km", "wrfout_%s_*" % dom, "no frames found",
                        run=run_label)]
    last_time, last_path = frames[-1]
    try:
        with netCDF4.Dataset(last_path) as ds:
            centre = frame_centre(ds, last_time, track, bdy_width)
            dx = cl.dx_km_of(ds)
            clat = float(ds.variables["XLAT"][0][centre[0], centre[1]])
            clon = float(ds.variables["XLONG"][0][centre[0], centre[1]])
    except _ERRORS + (OSError,) as e:
        return [missing("tc_struct_reflectivity_2km", "wrfout_%s SLP (storm centre)" % dom,
                        _reason(e), run=run_label)]
    note = cl.grid_limit_note(dx)
    extra = {"note": note} if note else None
    tlabel = path_time_label(last_path, last_time)

    p = os.path.join(out_dir, "tc_struct_reflectivity_2km_%s.png" % dom)
    try:
        reflectivity_2km_figure(last_path, last_time, centre, p, run_label, domain)
        out.append(fig("tc_struct_reflectivity_2km", p, run_label, domain=dom,
                       valid_time=last_time, extra=extra))
    except _ERRORS as e:
        out.append(missing("tc_struct_reflectivity_2km", "wrfout_%s dbz/U10/V10" % dom,
                           _reason(e), run=run_label))

    from post.recipes.cf3d import cf_domains
    if not ("cloud" in case_info.get("types", ()) and domain in cf_domains(case_info)):
        p = os.path.join(out_dir, "tc_struct_ir_proxy_%s.png" % dom)
        try:
            with netCDF4.Dataset(last_path) as ds:
                cl.plot_ir_proxy(ds, last_time, p, run_label, domain)
            out.append(fig("tc_struct_ir_proxy", p, run_label, domain=dom, valid_time=last_time,
                           extra={"note": cl.IR_TITLE}))
        except _ERRORS as e:
            out.append(missing("tc_struct_ir_proxy", "wrfout_%s OLR" % dom, _reason(e),
                               run=run_label))

    try:
        rz = rz_analysis(last_path, centre, bdy_width)
    except _ERRORS as e:
        rz = None
        for fid in ("tc_struct_rz_wind", "tc_struct_rz_warmcore", "tc_struct_rz_condensate"):
            out.append(missing(fid, "wrfout_%s z/ua/va/wa/tk/th + hydrometeors" % dom,
                               _reason(e), run=run_label))
    if rz is not None:
        for fid, fn in (("tc_struct_rz_wind", rz_wind_figure),
                        ("tc_struct_rz_warmcore", rz_warmcore_figure),
                        ("tc_struct_rz_condensate", rz_condensate_figure)):
            p = os.path.join(out_dir, "%s_%s.png" % (fid, dom))
            try:
                fn(rz, p, run_label, domain, tlabel)
                e_ = dict(extra or {})
                if fid == "tc_struct_rz_warmcore":
                    e_["note"] = "; ".join(x for x in (note, annulus_label(rz["annulus"])) if x)
                out.append(fig(fid, p, run_label, domain=dom, valid_time=last_time,
                               extra=e_ or None))
            except _ERRORS as e:
                out.append(missing(fid, "azimuthal-mean section", _reason(e), run=run_label))

    p = os.path.join(out_dir, "tc_struct_xsec_%s.png" % dom)
    try:
        xsec_figure(last_path, last_time, centre, p, run_label, domain)
        out.append(fig("tc_struct_xsec", p, run_label, domain=dom, valid_time=last_time))
    except _ERRORS as e:
        out.append(missing("tc_struct_xsec", "wrfout_%s dbz/wa/tc" % dom, _reason(e),
                           run=run_label))

    hov_times = hovmoller_frames([t for t, _ in frames], every=every, times=times)
    if len(hov_times) < 2:
        out.append(missing("tc_struct_warmcore_hovmoller", "wrfout_%s_*" % dom,
                           "a Hovmoller needs 2+ frames (%d selected)" % len(hov_times),
                           run=run_label))
    else:
        p = os.path.join(out_dir, "tc_struct_warmcore_hovmoller_%s.png" % dom)
        try:
            hovmoller_figure(run_dir, domain, hov_times, track, bdy_width, p, run_label)
            out.append(fig("tc_struct_warmcore_hovmoller", p, run_label, domain=dom,
                           valid_time=last_time))
        except _ERRORS as e:
            out.append(missing("tc_struct_warmcore_hovmoller", "wrfout_%s tk" % dom,
                               _reason(e), run=run_label))

    if paraview:
        p = os.path.join(out_dir, "tc_struct_condensate_3d_%s.png" % dom)
        try:
            written = cl.condensate_volume(
                last_path, p, os.path.join(out_dir, "_paraview_%s" % dom),
                "%s  %s  total condensate volume, storm-centred %.0f km cutaway  %s"
                % (run_label, dom, 2 * BOX_HALF_KM, last_time),
                center_latlon=(clat, clon), box_km=2 * BOX_HALF_KM, cutaway=True)
            out.append(fig("tc_struct_condensate_3d", written or p, run_label, domain=dom,
                           valid_time=last_time))
        except Exception as e:    # noqa: BLE001 -- pvbatch absent/failed, not implemented: a gap, never a failed report
            out.append(missing("tc_struct_condensate_3d",
                               "pvbatch / paraview3d.render_condensate_volume", _reason(e),
                               run=run_label))
    return out
