#!/usr/bin/env python3
"""wtune calc -- rules of thumb and unit conversions for namelist parameters (decision 0062).

    wtune calc <topic> [--args]            compute a recommended value, with formula and source
    wtune calc check <namelist.input>      apply every applicable rule to an existing namelist

Topics: dt cfl radt nest domain vert physics geog decomp io run units
        nudge scales les acoustic bdy movenest longrun spinup

Every number carries its rule from wtune_calc_rules.json: formula, `source_level`
(wrf-doc | wtune-measured | rule-of-thumb) and `source_ref`. Advice only, the namelist is never
modified (decision 0056). `check` reports ERROR only for hard=true wrf-doc rules that WRF itself
enforces with wrf_error_fatal; a deviation from a rule of thumb is DEVIATES, never ERROR.

The pure functions below return {"topic", "inputs", "items": [...]}; every item is
{rule_id, label, value, unit, note, formula, source_level, source_ref}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RULES_PATH = HERE / "wtune_calc_rules.json"

R_EARTH_KM = 6370.0                      # WRF earth radius (share/module_model_constants.F:43)
KM_PER_DEG = math.pi / 180.0 * R_EARTH_KM
OMEGA = 7.2921e-5                        # WRF EOMEG (share/module_model_constants.F:83)
G = 9.81
GAMMA_D = 9.8                            # K/km, dry adiabatic lapse rate
KT_MS = 1852.0 / 3600.0
ICAO_P0, ICAO_T11_P, ICAO_ZSLOPE = 1013.25, 226.32, 44330.77
ICAO_EXP, ICAO_ISO_H = 0.190263, 6341.62
TOPICS = ("dt", "cfl", "radt", "nest", "domain", "vert", "physics", "geog", "decomp", "io", "run",
          "units", "nudge", "scales", "les", "acoustic", "bdy", "movenest", "longrun", "spinup")
GEOG_STANDARD_ARCSEC = (("10m", 600.0), ("5m", 300.0), ("2m", 120.0), ("30s", 30.0))

# io defaults: variable counts of a real matthew wrfinput_d01 (91x100x45): 19 mass-level + 3
# staggered-level 3-D fields, 70 + 6 2-D fields; formula gives 38 MB vs 40.3 MB uncompressed.
IO_N3D, IO_N2D = 22, 76


# ------------------------------------------------------------------------------ rules / items

_RULES = None


def rules():
    global _RULES
    if _RULES is None:
        data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        _RULES = {r["id"]: r for r in data["rules"]}
    return _RULES


def rule(rid):
    return rules()[rid]


def _item(rid, label, value, unit="", note=""):
    r = rule(rid)
    return {"rule_id": rid, "label": label, "value": value, "unit": unit, "note": note,
            "formula": r["formula"], "source_level": r["source_level"], "source_ref": r["source_ref"]}


def _result(topic, inputs, items):
    return {"topic": topic, "inputs": inputs, "items": items}


# ------------------------------------------------------------------------------------- units

def deg_to_km(deg, lat=0.0, axis="lat"):
    km = deg * KM_PER_DEG
    return km * math.cos(math.radians(lat)) if axis == "lon" else km


def km_to_deg(km, lat=0.0, axis="lat"):
    per_deg = KM_PER_DEG * (math.cos(math.radians(lat)) if axis == "lon" else 1.0)
    return km / per_deg


def arcsec_to_m(arcsec, lat=0.0, axis="lat"):
    return deg_to_km(arcsec / 3600.0, lat, axis) * 1000.0


def m_to_arcsec(m, lat=0.0, axis="lat"):
    return km_to_deg(m / 1000.0, lat, axis) * 3600.0


def icao_height_m(p_hpa):
    """Geopotential height of pressure p (hPa) in the ICAO standard atmosphere."""
    if p_hpa >= ICAO_T11_P:
        return ICAO_ZSLOPE * (1.0 - (p_hpa / ICAO_P0) ** ICAO_EXP)
    return 11000.0 + ICAO_ISO_H * math.log(ICAO_T11_P / p_hpa)


def icao_pressure_hpa(h_m):
    if h_m <= 11000.0:
        return ICAO_P0 * (1.0 - h_m / ICAO_ZSLOPE) ** (1.0 / ICAO_EXP)
    return ICAO_T11_P * math.exp(-(h_m - 11000.0) / ICAO_ISO_H)


def kt_to_ms(kt):
    return kt * KT_MS


def ms_to_kt(ms):
    return ms / KT_MS


_TIME_S = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def convert_time(value, src, dst):
    return value * _TIME_S[src] / _TIME_S[dst]


UNIT_KINDS = ("deg-km", "km-deg", "arcsec-m", "m-arcsec", "hpa-m", "m-hpa", "kt-ms", "ms-kt",
              "s-min", "min-s", "h-min", "min-h", "d-h", "h-d", "s-h", "h-s")


def units(kind, value, lat=0.0, axis="lat"):
    if kind == "deg-km":
        out, unit, rid = deg_to_km(value, lat, axis), "km", "units.earth_radius"
    elif kind == "km-deg":
        out, unit, rid = km_to_deg(value, lat, axis), "deg", "units.earth_radius"
    elif kind == "arcsec-m":
        out, unit, rid = arcsec_to_m(value, lat, axis), "m", "units.earth_radius"
    elif kind == "m-arcsec":
        out, unit, rid = m_to_arcsec(value, lat, axis), "arcsec", "units.earth_radius"
    elif kind == "hpa-m":
        out, unit, rid = icao_height_m(value), "m", "units.icao"
    elif kind == "m-hpa":
        out, unit, rid = icao_pressure_hpa(value), "hPa", "units.icao"
    elif kind == "kt-ms":
        out, unit, rid = kt_to_ms(value), "m/s", "units.knot"
    elif kind == "ms-kt":
        out, unit, rid = ms_to_kt(value), "kt", "units.knot"
    elif kind in UNIT_KINDS:
        src, dst = kind.split("-")
        out, unit, rid = convert_time(value, src, dst), dst, "units.knot"
    else:
        raise ValueError("unknown units kind %r (choose from %s)" % (kind, ", ".join(UNIT_KINDS)))
    note = "lat=%g axis=%s" % (lat, axis) if kind in ("deg-km", "km-deg", "arcsec-m", "m-arcsec") else ""
    if kind in UNIT_KINDS[8:]:
        note = "exact (time unit definition)"
    return _result("units", {"kind": kind, "value": value, "lat": lat, "axis": axis},
                   [_item(rid, kind, out, unit, note)])


# ---------------------------------------------------------------------------------- dt / cfl

def dt(dx_km, terrain="typical", max_map_factor=1.0, parent_grid_ratio=None):
    eff = dx_km / max_map_factor
    typical = 6.0 * eff
    steep = [4.0 * eff, 5.0 * eff]
    items = [
        _item("dt.six_dx", "time_step recommended", typical if terrain == "typical" else steep,
              "s", "terrain=%s" % terrain),
        _item("dt.steep_terrain", "time_step steep terrain / high latitude", steep, "s"),
        _item("dt.reasonable_ratio", "d01 abort limit (default reasonable_time_step_ratio 6)",
              6.0 * dx_km, "s", "dt/dx_km > 6 stops WRF (map factor not included in the gate)"),
        _item("dt.map_factor", "effective dx", eff, "km", "dx / max_map_factor=%g" % max_map_factor),
        _item("dt.adaptive_defaults", "adaptive starting/max/min time step",
              [4.0 * dx_km, 8.0 * dx_km, 3.0 * dx_km], "s",
              "target_cfl 1.2, target_hcfl 0.84, max_step_increase_pct 5 (Registry); units of -1 = N*dx not stated in README"),
    ]
    if parent_grid_ratio:
        nest_dt = (typical if terrain == "typical" else steep[0]) / parent_grid_ratio
        items.append(_item("nest.time_step_ratio", "nest time_step for parent_grid_ratio=%d" % parent_grid_ratio,
                           nest_dt, "s", "parent_time_step_ratio = %d (same as grid ratio)" % parent_grid_ratio))
    return _result("dt", {"dx_km": dx_km, "terrain": terrain, "max_map_factor": max_map_factor,
                          "parent_grid_ratio": parent_grid_ratio}, items)


def cfl(u, dx_m, dt_s, w=None, dz_m=None):
    ch = abs(u) * dt_s / dx_m
    items = [
        _item("cfl.limit", "horizontal Courant number", ch, "",
              "<= target_hcfl 0.84: %s; < RK3 limit ~1.4: %s" % ("yes" if ch <= 0.84 else "NO", "yes" if ch < 1.4 else "NO")),
        _item("cfl.adaptive_targets", "max dt for target_hcfl 0.84", 0.84 * dx_m / abs(u) if u else None, "s"),
    ]
    if w is not None and dz_m:
        cv = abs(w) * dt_s / dz_m
        items.append(_item("cfl.adaptive_targets", "vertical Courant number", cv, "",
                           "<= target_cfl/w_crit_cfl 1.2: %s%s" % ("yes" if cv <= 1.2 else "NO",
                                                                  "; consider w_damping=1 / epssm" if cv > 1.2 else "")))
    return _result("cfl", {"u": u, "dx_m": dx_m, "dt_s": dt_s, "w": w, "dz_m": dz_m}, items)


def radt(dx_km, dt_s=None):
    items = [_item("radt.one_min_per_km", "radt recommended", dx_km, "min", "same value on all nests")]
    if dt_s:
        items.append(_item("radt.one_min_per_km", "radiation call every N time steps", dx_km * 60.0 / dt_s, "steps",
                           "arithmetic: radt*60/time_step"))
    return _result("radt", {"dx_km": dx_km, "dt_s": dt_s}, items)


# ------------------------------------------------------------------------------ nest / domain

def nest_dim_ok(e, ratio):
    return (int(e) - 1) % int(ratio) == 0


def nearest_nest_dim(e, ratio):
    n = max(1, round((int(e) - 1) / ratio))
    return int(n * ratio + 1)


def nest_footprint(e_we, e_sn, ratio, i_start, j_start):
    """(i0, i1, j0, j1) parent MASS cells covered by a nest."""
    return (i_start, i_start + (e_we - 1) // ratio - 1, j_start, j_start + (e_sn - 1) // ratio - 1)


def nest_margin(e_we, e_sn, ratio, i_start, j_start, parent_e_we, parent_e_sn):
    """Smallest distance (parent cells) from the nest footprint to the parent boundary; < 0 outside."""
    i0, i1, j0, j1 = nest_footprint(e_we, e_sn, ratio, i_start, j_start)
    return min(i0 - 1, (parent_e_we - 1) - i1, j0 - 1, (parent_e_sn - 1) - j1)


def nest(parent_grid_ratio, parent_cells_x=None, parent_cells_y=None, parent_dx_km=None,
         i_parent_start=None, j_parent_start=None, parent_e_we=None, parent_e_sn=None,
         e_we=None, e_sn=None, spec_bdy_width=5):
    ratio = int(parent_grid_ratio)
    items = [_item("nest.parent_ratio_odd", "parent_grid_ratio", ratio, "",
                   "odd: %s (3 or 5 usual)" % ("yes" if ratio % 2 == 1 else "NO, README requires odd for real-data")),
             _item("nest.time_step_ratio", "parent_time_step_ratio", ratio, "", "= parent_grid_ratio")]
    if parent_cells_x and parent_cells_y:
        we, sn = parent_cells_x * ratio + 1, parent_cells_y * ratio + 1
        items.append(_item("nest.dim_legal", "e_we / e_sn for %d x %d parent cells" % (parent_cells_x, parent_cells_y),
                           [we, sn], "", "n*ratio+1"))
        if parent_dx_km:
            items.append(_item("nest.dim_legal", "nest dx", parent_dx_km / ratio, "km"))
        e_we, e_sn = e_we or we, e_sn or sn
    if e_we and e_sn:
        ok = nest_dim_ok(e_we, ratio) and nest_dim_ok(e_sn, ratio)
        items.append(_item("nest.dim_legal", "e_we=%d, e_sn=%d legal" % (e_we, e_sn),
                           ok, "", "" if ok else "nearest legal: %d x %d" % (nearest_nest_dim(e_we, ratio), nearest_nest_dim(e_sn, ratio))))
        if None not in (i_parent_start, j_parent_start, parent_e_we, parent_e_sn):
            m = nest_margin(e_we, e_sn, ratio, i_parent_start, j_parent_start, parent_e_we, parent_e_sn)
            i0, i1, j0, j1 = nest_footprint(e_we, e_sn, ratio, i_parent_start, j_parent_start)
            items.append(_item("nest.inside_parent", "margin to parent boundary", m, "parent cells",
                               "footprint i %d..%d j %d..%d; need >= spec_bdy_width %d; ~10 cells is the comfortable convention (bdy.nest_distance)"
                               % (i0, i1, j0, j1, spec_bdy_width)))
    return _result("nest", {"parent_grid_ratio": ratio}, items)


def bdy(spec_bdy_width=5, parent_cells=None):
    items = [_item("bdy.spec_width", "spec_bdy_width / spec_zone / relax_zone",
                   [spec_bdy_width, 1, spec_bdy_width - 1], "", "WRF overwrites spec_zone=1, relax_zone=width-1 for specified d01"),
             _item("bdy.nest_distance", "nest distance from parent boundary", 10, "parent cells",
                   "or ~1/3 of the nest width%s" % ("" if not parent_cells else " = %.0f cells" % (parent_cells / 3.0)))]
    return _result("bdy", {"spec_bdy_width": spec_bdy_width, "parent_cells": parent_cells}, items)


def projection_for_lat(lat):
    a = abs(lat)
    return "mercator" if a <= 20 else ("lambert" if a <= 65 else "polar")


def domain(center_lat, center_lon, width_km, height_km, dx_km):
    we, sn = int(round(width_km / dx_km)) + 1, int(round(height_km / dx_km)) + 1
    proj = projection_for_lat(center_lat)
    half = km_to_deg(height_km, 0.0) / 2.0
    lo, hi = center_lat - half, center_lat + half
    t1, t2 = lo + (hi - lo) / 6.0, lo + 5.0 * (hi - lo) / 6.0
    items = [_item("domain.dims", "e_we, e_sn", [we, sn], "", "dx=%g km, extent %g x %g km" % (dx_km, width_km, height_km)),
             _item("domain.dims", "dx / dy", dx_km * 1000.0, "m"),
             _item("domain.dims", "ref_lat, ref_lon / stand_lon", [center_lat, center_lon], "deg"),
             _item("domain.projection", "map_proj", proj, "", "|lat| %.1f" % abs(center_lat))]
    if proj == "lambert":
        items.append(_item("domain.truelat", "truelat1, truelat2", [round(t1, 2), round(t2, 2)], "deg",
                           "1/6 and 5/6 of the %.1f..%.1f lat extent" % (lo, hi)))
    elif proj == "mercator":
        items.append(_item("domain.truelat", "truelat1", center_lat, "deg", "true latitude near the domain centre"))
    else:
        items.append(_item("domain.truelat", "truelat1", math.copysign(60.0, center_lat), "deg", "polar stereographic: 60 deg of the hemisphere"))
    return _result("domain", {"center_lat": center_lat, "center_lon": center_lon, "width_km": width_km,
                              "height_km": height_km, "dx_km": dx_km}, items)


# --------------------------------------------------------------------------------- vert/physics

def vert(e_vert, p_top_pa=5000.0, p_sfc_hpa=ICAO_P0, eta=None):
    ztop = icao_height_m(p_top_pa / 100.0) - icao_height_m(p_sfc_hpa)
    zdamp = min(5000.0, 0.25 * ztop)
    items = [_item("vert.ptop", "model top height (standard atmosphere)", ztop, "m", "p_top=%g Pa" % p_top_pa),
             _item("vert.ptop", "mean layer thickness", ztop / max(e_vert - 1, 1), "m", "even eta; real levels are stretched"),
             _item("vert.damp", "damp_opt / dampcoef / zdamp", [3, 0.2, zdamp], "-/s^-1/m",
                   "w-Rayleigh for real-data; zdamp default 5000 m, must stay below model top")]
    if eta:
        zs = []
        for e in eta:
            p = (p_top_pa + e * (p_sfc_hpa * 100.0 - p_top_pa)) / 100.0
            zs.append([e, round(p, 2), round(icao_height_m(p) - icao_height_m(p_sfc_hpa), 1)])
        items.append(_item("vert.ptop", "eta -> [eta, hPa, height m]", zs, "", "p = p_top + eta*(p_sfc - p_top)"))
    return _result("vert", {"e_vert": e_vert, "p_top_pa": p_top_pa}, items)


def dx_regime(dx_km):
    """(text, suggested cu_physics) -- same thresholds as wtune_case._dx_regime."""
    if dx_km <= 4.0:
        return "可解析對流（dx<=4 km）：cu_physics 應為 0", [0]
    if dx_km <= 10.0:
        return "灰色地帶 4-10 km：用尺度感知積雲（3 Grell-Freitas、11 MSKF）", [3, 11]
    return "dx>10 km：通常需要積雲方案（1 KF、6 Tiedtke）", [1, 6]


def physics(dx_km, cu_physics=None):
    text, rec = dx_regime(dx_km)
    note = text if cu_physics is None else "%s; current cu_physics=%s" % (text, cu_physics)
    return _result("physics", {"dx_km": dx_km, "cu_physics": cu_physics},
                   [_item("physics.cumulus", "cu_physics suggestion", rec, "", note)])


def geog(dx_km, res_arcsec=30.0, lat=0.0):
    dx_m = dx_km * 1000.0
    res_m = arcsec_to_m(res_arcsec, lat)
    fine_enough = res_m <= dx_m / 2.0
    pick = next((n for n, a in GEOG_STANDARD_ARCSEC if arcsec_to_m(a, lat) <= dx_m / 2.0), None)
    items = [_item("geog.res_vs_dx", "geog resolution %g arcsec" % res_arcsec, res_m, "m",
                   "dx %.0f m: %s" % (dx_m, "coarser than the grid" if res_m > dx_m else ("2x finer: yes" if fine_enough else "finer but < 2x"))),
             _item("geog.res_vs_dx", "30 arcsec in metres", arcsec_to_m(30.0, lat), "m"),
             _item("geog.res_vs_dx", "coarsest standard dataset >= 2x finer than the grid",
                   pick or "none (30s is the finest standard WPS set; use user data)", "", "standard WPS: 10m, 5m, 2m, 30s")]
    return _result("geog", {"dx_km": dx_km, "res_arcsec": res_arcsec, "lat": lat}, items)


# ------------------------------------------------------------------------------------ decomp

def _wc():
    import wtune.case
    return wtune.case


def decomp(we_list, sn_list, ranks=None, cores_per_node=None):
    wc = _wc()
    r_max, caps = wc.compute_r_max(we_list, sn_list)
    items = [_item("decomp.min_patch", "max MPI ranks (omp=1)", r_max, "ranks",
                   "nproc_x<=%d, nproc_y<=%d (min patch 10 on every domain)" % caps),
             _item("decomp.min_patch_measured", "measured example", "Front Range 42x42 + 52x31 -> 12 ranks (4x3)", "")]
    if ranks:
        legal = wc.ds.legal_shapes_exact(we_list, sn_list, ranks)
        best = min(legal, key=lambda s: abs(math.log((we_list[0] / s[0]) / (sn_list[0] / s[1])))) if legal else None
        square = wc.least_aspect_legal(we_list, sn_list, ranks)
        items.append(_item("decomp.min_patch", "legal nproc_x x nproc_y for %d ranks" % ranks, legal, ""))
        items.append(_item("decomp.min_patch", "closest to square patches on d01", list(best) if best else None, "",
                           "patch aspect follows the domain aspect"))
        items.append(_item("decomp.min_patch", "most square rank grid", list(square) if square else None, ""))
    if cores_per_node:
        combos = [[cores_per_node // o, o] for o in range(1, cores_per_node + 1) if cores_per_node % o == 0]
        items.append(_item("decomp.min_patch", "[ranks_per_node, omp_threads] filling %d cores" % cores_per_node,
                           combos, "", "numtiles is a request; see wtune tune for measured shapes"))
    return _result("decomp", {"e_we": list(we_list), "e_sn": list(sn_list), "ranks": ranks}, items)


# --------------------------------------------------------------------------------------- io

def io_frame_bytes(e_we, e_sn, e_vert, n3d=IO_N3D, n2d=IO_N2D):
    return 4.0 * (e_we * e_sn * (e_vert - 1) * n3d + e_we * e_sn * n2d)


def _calibrated_bytes_per_cell_level():
    try:
        import wtune.predict
        cal = wtune.predict.load_calibration(wtune.predict.DEFAULT_CALIBRATION)
        return float(cal["derived"]["bytes_per_cell_level"])
    except Exception:
        return None


def io(e_we, e_sn, e_vert, frames=1, n3d=None, n2d=None):
    n3d = IO_N3D if n3d is None else n3d
    n2d = IO_N2D if n2d is None else n2d
    fb = io_frame_bytes(e_we, e_sn, e_vert, n3d, n2d)
    items = [_item("io.frame_first_principles", "frame size (ESTIMATE)", fb / 1e6, "MB",
                   "n3d=%d n2d=%d%s" % (n3d, n2d, " (defaults from a matthew wrfinput, not a wrfout)" if n3d == IO_N3D and n2d == IO_N2D else "")),
             _item("io.frame_first_principles", "total for %d frame(s) (ESTIMATE)" % frames, fb * frames / 1e9, "GB")]
    bpcl = _calibrated_bytes_per_cell_level()
    if bpcl:
        cb = bpcl * e_we * e_sn * e_vert
        items.append(_item("io.frame_calibrated", "frame size (conus25 calibration)", cb / 1e6, "MB"))
        items.append(_item("io.frame_calibrated", "total for %d frame(s) (conus25 calibration)" % frames, cb * frames / 1e9, "GB"))
    return _result("io", {"e_we": e_we, "e_sn": e_sn, "e_vert": e_vert, "frames": frames}, items)


# --------------------------------------------------------------------------------------- run

def _wp():
    import wtune.predict
    return wtune.predict


def run(namelist, nodes=None, nx=None, ny=None, calibration=None):
    wp = _wp()
    nl = wp.parse_namelist(namelist)
    steps, run_s, dt_s, note = wp.get_d01_steps(nl)
    max_dom = wp.get_max_dom(nl)
    items = [_item("run.steps", "d01 steps", steps, "steps", "run %.2f h at dt %g s%s" % (run_s / 3600.0, dt_s, "; " + note if note else ""))]
    cum = 1
    for d in range(1, max_dom):
        cum *= wp.parent_time_step_ratio(nl, d, 1)
        items.append(_item("run.steps", "d%02d steps" % (d + 1), steps * cum, "steps", "cumulative parent_time_step_ratio %d" % cum))
    if nodes and nx and ny:
        pred = wp.build_prediction(namelist_path=namelist, case_dir=str(Path(namelist).parent), nodes=nodes,
                                   config={"ranks": nx * ny, "omp": 1, "nx": nx, "ny": ny, "numtiles": 1, "dist": None},
                                   ledger_paths=None, calibration_path=calibration or str(wp.DEFAULT_CALIBRATION),
                                   io_form_override=None, io_target="nfs")
        items.append(_item("run.predict", "predicted wall time (wtune_predict)", pred["total_s"], "s",
                           "%d nodes, %dx%d ranks; accuracy caveats live in wtune_predict" % (nodes, nx, ny)))
    return _result("run", {"namelist": str(namelist), "nodes": nodes, "nx": nx, "ny": ny}, items)


# ------------------------------------------------------------------------------ nudge/scales...

def nudge(e_we, e_sn, dx_m, wavelength_km=1000.0, g=3e-4):
    lx, ly = (e_we - 1) * dx_m / 1000.0, (e_sn - 1) * dx_m / 1000.0
    xw, yw = int(round(lx / wavelength_km)), int(round(ly / wavelength_km))
    tau_min = 1.0 / g / 60.0
    items = [
        _item("nudge.wavenumber", "xwavenum, ywavenum", [xw, yw], "", "L = %.0f x %.0f km, lambda = %g km (typical 1000-1500 km)" % (lx, ly, wavelength_km)),
        _item("nudge.wavenumber", "xwavenum range for lambda 1000-1500 km",
              [[int(round(lx / 1500.0)), int(round(lx / 1000.0))], [int(round(ly / 1500.0)), int(round(ly / 1000.0))]], ""),
        _item("nudge.coefficients", "guv = gt = gph", g, "s^-1", "tau = 1/g = %.1f min; README example 3e-4, Registry default 0 (set explicitly)" % tau_min),
        _item("nudge.moisture", "gq", 0.0, "s^-1", "recommendation: do not nudge moisture; README example is 1e-5, Registry default 0"),
        _item("nudge.pbl", "if_no_pbl_nudging_uv/t/ph", [1, 1, 1], "", "if_no_pbl_nudging_q = 1 as well if q were nudged"),
        _item("nudge.zfac", "alternative to PBL switch", "if_zfac_uv/t/ph = 1, k_zfac_* = model level, dk_zfac_* = ramp", ""),
        _item("nudge.outer_only", "grid_fdda", "grid_fdda = 2 on d01, 0 on nests", ""),
        _item("nudge.timing", "gfdda_interval_m / gfdda_end_h / fgdt", "= interval_seconds/60 / >= run hours / 0", ""),
        _item("nudge.no_search", "policy", "WTune never searches nudging strengths automatically: minimising error against the "
              "driving analysis turns the model into a copy of it", ""),
    ]
    return _result("nudge", {"e_we": e_we, "e_sn": e_sn, "dx_m": dx_m, "wavelength_km": wavelength_km, "g": g}, items)


def coriolis(lat):
    return 2.0 * OMEGA * math.sin(math.radians(lat))


def brunt_vaisala(T=288.0, lapse_rate_k_per_km=6.5):
    n2 = G / T * (GAMMA_D - lapse_rate_k_per_km) / 1000.0
    return math.sqrt(n2) if n2 > 0 else float("nan")


def froude(U, N, h):
    return U / (N * h)


def scales(lat=None, dx_km=None, N=None, H=None, U=None, h=None, T=288.0, lapse_rate=None, feature_km=None):
    items = []
    f = None
    if lat is not None:
        f = coriolis(lat)
        items.append(_item("scales.coriolis", "f", f, "s^-1", "lat %g deg" % lat))
    if lapse_rate is not None:
        N = brunt_vaisala(T, lapse_rate)
        items.append(_item("scales.brunt_vaisala", "N", N, "s^-1", "T=%g K, lapse rate %g K/km%s" % (T, lapse_rate, "" if N == N else " (unstable: N^2<0)")))
    lr = None
    if N and H and f:
        lr = N * H / abs(f) / 1000.0
        items.append(_item("scales.rossby", "L_R = N*H/f", lr, "km", "N=%.4g H=%g m" % (N, H)))
    if dx_km:
        items.append(_item("scales.effective_resolution", "effective resolution", 7.0 * dx_km, "km", "7*dx"))
    if U and N and h:
        fr = froude(U, N, h)
        items.append(_item("scales.froude", "Fr = U/(N*h)", fr, "", "blocked / flow around (Fr<1)" if fr < 1 else "flow over (Fr>=1)"))
    if feature_km or lr:
        cand = [10.0 * feature_km] if feature_km else []
        if lr:
            cand.append(lr)
        items.append(_item("scales.min_domain", "suggested minimum domain width", max(cand), "km",
                           "max(10*feature, L_R)"))
    return _result("scales", {"lat": lat, "dx_km": dx_km, "N": N, "H": H, "U": U, "h": h}, items)


def les_regime(dx_m):
    if dx_m <= 200.0:
        return "les", "LES：diff_opt=2、km_opt=2 或 3（或 5）、bl_pbl_physics=0"
    if dx_m < 1000.0:
        return "gray", "灰色地帶 200 m..1 km：PBL 方案與 3-D 混合同時部分作用；文件記載的尺度自適應選項是 km_opt=5（SMS-3DTKE，需 diff_opt=2、關 PBL）"
    return "pbl", "用 PBL 方案搭 km_opt=4、diff_opt=1（km_opt 2/3 '不建議用於 DX > 2 km'）"


def les(dx_m):
    kind, text = les_regime(dx_m)
    return _result("les", {"dx_m": dx_m}, [
        _item("les.regime", "regime", kind, "", text),
        _item("les.regime", "khdif (constant-diffusion option km_opt=1)", 0.1 * dx_m, "m^2/s", "README: typical 0.1*DX (m)")])


def sound_steps_auto(dt_s, dx_m):
    """WRF's automatic time_step_sound (dyn_em/solve_em.F:451-466; fixed time step, 300 m/s)."""
    return max(2 * (int(300.0 * dt_s / dx_m - 0.01) + 1), 4)


def acoustic(dt_s, dx_m, c_s=340.0, time_step_sound=0):
    n_auto = sound_steps_auto(dt_s, dx_m)
    n = time_step_sound or n_auto
    dts = dt_s / n
    cac = c_s * dts / dx_m
    return _result("acoustic", {"dt_s": dt_s, "dx_m": dx_m, "c_s": c_s, "time_step_sound": time_step_sound}, [
        _item("acoustic.sound_steps", "time_step_sound (auto)", n_auto, "", "WRF code assumes 300 m/s"),
        _item("acoustic.sound_steps", "acoustic time step dts", dts, "s", "n=%d%s" % (n, "" if time_step_sound else " (auto)")),
        _item("acoustic.sound_steps", "acoustic Courant number", cac, "", "c_s=%g m/s" % c_s),
        _item("acoustic.epssm", "epssm", 0.1, "", "default; 0.3-0.5 for steep terrain (rule-of-thumb)"),
        _item("acoustic.w_damping", "w_damping / w_crit_cfl", [1, 1.2], "", "turn on when vertical CFL is high; w_crit_cfl ~2.0 with IEVA")])


def movenest(dx_m, storm_speed=None, parent_dx_km=None, vortex_interval=15, max_vortex_speed=40, corral_dist=8):
    radius_m = 60.0 * vortex_interval * max_vortex_speed
    items = [_item("movenest.vortex", "search radius", radius_m / 1000.0, "km",
                   "%.1f nest cells = 60*%g min*%g m/s / dx" % (radius_m / dx_m, vortex_interval, max_vortex_speed)),
             _item("movenest.vortex", "defaults", [15, 40, 8], "min/m s^-1/parent cells", "vortex_interval, max_vortex_speed, corral_dist; track_level 50000 Pa")]
    if storm_speed:
        items.append(_item("movenest.vortex", "max_vortex_speed suggestion", [1.5 * storm_speed, 2.0 * storm_speed], "m/s",
                           "1.5-2 x storm speed (rule-of-thumb)"))
        if parent_dx_km:
            items.append(_item("movenest.vortex", "vortex_interval so the storm moves <= 1 parent cell",
                               parent_dx_km * 1000.0 / storm_speed / 60.0, "min", "rule-of-thumb"))
    return _result("movenest", {"dx_m": dx_m, "storm_speed": storm_speed}, items)


def longrun(run_days=None, interval_seconds=21600):
    items = [_item("longrun.sst", "sst_update / auxinput4_interval", [1, interval_seconds / 60.0], "-/min",
                   "also auxinput4_inname='wrflowinp_d<domain>', io_form_auxinput4=2, usemonalb=.true.; tmn_update=1 (lagday 150), sst_skin optional"),
             _item("longrun.threshold", "threshold", 5, "days", "sst_update=1 recommended beyond ~5 days (rule-of-thumb)")]
    if run_days is not None:
        items.append(_item("longrun.threshold", "this run", run_days, "days", "sst_update recommended" if run_days > 5 else "not needed"))
    return _result("longrun", {"run_days": run_days}, items)


def spinup(run_hours=None):
    items = [_item("spinup.mesoscale", "discard first", [6, 12], "h", "cold start from coarse analyses"),
             _item("spinup.cost_measured", "cost spin-up (not meteorological)", "ERA5 with cloud water: <=3% cost error by T+20..40 min (conus25)", "")]
    if run_hours is not None:
        items.append(_item("spinup.mesoscale", "usable part of this run", max(0.0, run_hours - 12.0), "h", "after discarding 12 h"))
    return _result("spinup", {"run_hours": run_hours}, items)


# ============================================================================== check (namelist)

def _fmt(rid, key, **ctx):
    try:
        return rule(rid)["message"][key].format(**ctx)
    except (KeyError, IndexError, ValueError):
        return rule(rid)["message"][key]


def _find(rid, domain, status, reason):
    r = rule(rid)
    return {"rule_id": rid, "domain": domain, "status": status, "reason": reason,
            "source_level": r["source_level"], "source_ref": r["source_ref"]}


def _num(x, default=None):
    return default if x is None else x


class _NL:
    """Namelist view with per-domain geometry/time-step/dx resolved the way WRF resolves them."""

    def __init__(self, path):
        wp = _wp()
        self.wp = wp
        self.nl = wp.parse_namelist(path)
        self.max_dom = wp.get_max_dom(self.nl)
        self.doms = []
        for i in range(self.max_dom):
            d = {"i": i, "name": "d%02d" % (i + 1)}
            for k in ("e_we", "e_sn", "e_vert"):
                v = self.dom("domains", k, i)
                d[k] = int(v) if v is not None else None
            d["ratio"] = int(self.dom("domains", "parent_grid_ratio", i, 1))
            d["ptsr"] = int(self.dom("domains", "parent_time_step_ratio", i, 1))
            d["parent"] = int(self.dom("domains", "parent_id", i, 0 if i == 0 else 1))
            d["i0"] = self.dom("domains", "i_parent_start", i)
            d["j0"] = self.dom("domains", "j_parent_start", i)
            self.doms.append(d)
        dxs = self.nl.get("domains", {}).get("dx", [])
        for d in self.doms:
            i = d["i"]
            if i < len(dxs):
                d["dx_m"] = float(dxs[i])
            elif dxs:
                d["dx_m"] = float(dxs[0]) / self._chain(d, "ratio")
            else:
                d["dx_m"] = None
        ts = self.nl.get("domains", {}).get("time_step", [None])[0]
        try:
            self.dt0 = wp.get_timestep_seconds(self.nl) if ts is not None else None
        except Exception:
            self.dt0 = None
        for d in self.doms:
            d["dt"] = self.dt0 / self._chain(d, "ptsr") if self.dt0 else None
        self.adaptive = bool(self.get("domains", "use_adaptive_time_step", False))
        try:
            self.run_s = wp.get_run_seconds(self.nl)
        except Exception:
            self.run_s = None

    def get(self, group, key, default=None):
        return self.wp.g(self.nl, group, key, default)

    def dom(self, group, key, i, default=None):
        return self.wp.g_dom(self.nl, group, key, i, default)

    def has(self, group, key):
        return bool(self.nl.get(group, {}).get(key))

    def _chain(self, d, key):
        f, cur, guard = d[key], d, 0
        while cur["parent"] > 1 and guard < 10:
            cur = self.doms[min(cur["parent"] - 1, len(self.doms) - 1)]
            f *= cur[key]
            guard += 1
        return f if d["i"] > 0 else 1


def _chk_dt(n):
    out = []
    for d in n.doms:
        if not (d["dt"] and d["dx_m"]):
            continue
        dxk = d["dx_m"] / 1000.0
        ratio = d["dt"] / dxk
        if n.adaptive:
            out.append(_find("dt.six_dx", d["name"], "OK", "use_adaptive_time_step=.true.：time_step 只是名義值，不檢查"))
            continue
        bad = ratio > 6.0 + 1e-9
        out.append(_find("dt.six_dx", d["name"], "DEVIATES" if bad else "OK",
                         _fmt("dt.six_dx", "deviates" if bad else "ok", ratio=ratio, rec=6.0 * dxk)))
        if d["i"] == 0:
            limit = float(n.get("domains", "reasonable_time_step_ratio", 6.0))
            bad = ratio > limit
            out.append(_find("dt.reasonable_ratio", d["name"], "ERROR" if bad else "OK",
                             _fmt("dt.reasonable_ratio", "deviates" if bad else "ok", ratio=ratio, limit=limit)))
        elif d["ptsr"] != d["ratio"]:
            out.append(_find("nest.time_step_ratio", d["name"], "DEVIATES",
                             _fmt("nest.time_step_ratio", "deviates", ptsr=d["ptsr"], ratio=d["ratio"], nest_dt=d["dt"])))
    return out


def _chk_radt(n):
    out, vals = [], []
    for d in n.doms:
        r = n.dom("physics", "radt", d["i"])
        if r is None or not d["dx_m"]:
            continue
        dxk = d["dx_m"] / 1000.0
        vals.append(float(r))
        if d["i"] > 0:       # README: nests reuse the d01 value, so the 1 min/km target applies to d01 only
            continue
        ok = 0.5 <= float(r) / dxk <= 1.5
        out.append(_find("radt.one_min_per_km", d["name"], "OK" if ok else "DEVIATES",
                         _fmt("radt.one_min_per_km", "ok" if ok else "deviates", radt=float(r), rec=dxk, dx_km=dxk)))
    if len(set(vals)) > 1:
        out.append(_find("radt.nest_same", "all", "DEVIATES", _fmt("radt.nest_same", "deviates", vals=vals)))
    elif len(vals) > 1:
        out.append(_find("radt.nest_same", "all", "OK", _fmt("radt.nest_same", "ok", vals=vals)))
    return out


def _chk_nest(n):
    out = []
    for d in n.doms:
        if d["e_we"] is None or d["e_sn"] is None:
            continue
        ok = nest_dim_ok(d["e_we"], d["ratio"]) and nest_dim_ok(d["e_sn"], d["ratio"])
        out.append(_find("nest.dim_legal", d["name"], "OK" if ok else "ERROR",
                         _fmt("nest.dim_legal", "ok" if ok else "deviates", e_we=d["e_we"], e_sn=d["e_sn"], ratio=d["ratio"],
                              ok_we=nearest_nest_dim(d["e_we"], d["ratio"]), ok_sn=nearest_nest_dim(d["e_sn"], d["ratio"]))))
        if d["i"] == 0 or d["ratio"] <= 1:
            continue
        odd = d["ratio"] % 2 == 1
        out.append(_find("nest.parent_ratio_odd", d["name"], "OK" if odd else "DEVIATES",
                         _fmt("nest.parent_ratio_odd", "ok" if odd else "deviates", ratio=d["ratio"])))
        par = n.doms[min(d["parent"] - 1, len(n.doms) - 1)]
        if None in (d["i0"], d["j0"], par["e_we"], par["e_sn"]):
            continue
        i0, i1, j0, j1 = nest_footprint(d["e_we"], d["e_sn"], d["ratio"], int(d["i0"]), int(d["j0"]))
        m = nest_margin(d["e_we"], d["e_sn"], d["ratio"], int(d["i0"]), int(d["j0"]), par["e_we"], par["e_sn"])
        width = int(n.get("bdy_control", "spec_bdy_width", 5))
        if m < width:
            why = "超出母網格（幾何錯誤，real/wrf 會失敗）" if m < 0 else "距母網格邊界不到 spec_bdy_width %d 格" % width
            out.append(_find("nest.inside_parent", d["name"], "DEVIATES",
                             _fmt("nest.inside_parent", "deviates", i0=i0, i1=i1, j0=j0, j1=j1, pwe=par["e_we"], psn=par["e_sn"], why=why)))
        else:
            reason = _fmt("nest.inside_parent", "ok", margin=m)
            if m < 10:
                reason += "（低於約 10 格的慣例餘裕，見 bdy.nest_distance）"
            out.append(_find("nest.inside_parent", d["name"], "OK", reason))
    return out


def _chk_bdy(n):
    if not n.has("bdy_control", "spec_bdy_width"):
        return []
    w = int(n.get("bdy_control", "spec_bdy_width"))
    sz = int(n.get("bdy_control", "spec_zone", 1))
    rz = int(n.get("bdy_control", "relax_zone", w - 1))
    ok = sz == 1 and rz == w - sz
    return [_find("bdy.spec_width", "d01", "OK" if ok else "DEVIATES",
                  _fmt("bdy.spec_width", "ok" if ok else "deviates", w=w, sz=sz, rz=rz, rz_eff=w - 1))]


def _chk_vert(n):
    dopt = int(n.get("dynamics", "damp_opt", 3))
    zdamp = float(n.dom("dynamics", "zdamp", 0, 5000.0))
    ptop = float(n.get("domains", "p_top_requested", 5000.0))
    ztop = icao_height_m(ptop / 100.0) - icao_height_m(ICAO_P0)
    if dopt != 3:
        why = "damp_opt = %d（README：real-data 用 3 w-Rayleigh，2 僅限理想化案例；Registry 預設 3）" % dopt
        status = "DEVIATES"
    elif zdamp >= ztop:
        why, status = "zdamp %g m >= 模式頂 %.0f m" % (zdamp, ztop), "DEVIATES"
    else:
        why, status = "", "OK"
    reason = why or _fmt("vert.damp", "ok", damp_opt=dopt, zdamp=zdamp, ztop=ztop)
    return [_find("vert.damp", "d01", status, reason)]


def _chk_physics(n):
    out = []
    for d in n.doms:
        cu = n.dom("physics", "cu_physics", d["i"])
        if cu is None or cu == -1 or not d["dx_m"]:
            if cu == -1:
                out.append(_find("physics.cumulus", d["name"], "OK", "cu_physics = -1（由 physics_suite 決定），此處不評估"))
            continue
        dxk = d["dx_m"] / 1000.0
        text, rec = dx_regime(dxk)
        bad = (dxk <= 4.0 and cu > 0) or (dxk > 10.0 and cu == 0)
        out.append(_find("physics.cumulus", d["name"], "DEVIATES" if bad else "OK",
                         _fmt("physics.cumulus", "deviates" if bad else "ok", dx_km=dxk, cu=cu, regime=text)))
    return out


def _chk_geog(n, geog_res_arcsec):
    out = []
    if not geog_res_arcsec:
        return out
    for d in n.doms:
        if not d["dx_m"]:
            continue
        res_m = arcsec_to_m(geog_res_arcsec)
        if res_m > d["dx_m"]:
            out.append(_find("geog.res_vs_dx", d["name"], "DEVIATES",
                             _fmt("geog.res_vs_dx", "deviates", res_arcsec=geog_res_arcsec, res_m=res_m, dx_m=d["dx_m"])))
        else:
            note = "2x finer" if res_m <= d["dx_m"] / 2 else "finer than the grid but not 2x"
            out.append(_find("geog.res_vs_dx", d["name"], "OK",
                             _fmt("geog.res_vs_dx", "ok", res_arcsec=geog_res_arcsec, res_m=res_m, dx_m=d["dx_m"], note=note)))
    return out


def _chk_decomp(n):
    out = []
    we = [d["e_we"] for d in n.doms if d["e_we"]]
    sn = [d["e_sn"] for d in n.doms if d["e_sn"]]
    if not we or len(we) != len(sn):
        return out
    wc = _wc()
    r_max, _caps = wc.compute_r_max(we, sn)
    nx, ny = int(n.get("domains", "nproc_x", -1)), int(n.get("domains", "nproc_y", -1))
    if nx > 0 and ny > 0:
        legal = (nx, ny) in wc.ds.legal_shapes_exact(we, sn, nx * ny)
        pw, ph = min(w // nx for w in we), min(s // ny for s in sn)
        out.append(_find("decomp.min_patch", "all", "OK" if legal else "ERROR",
                         _fmt("decomp.min_patch", "ok" if legal else "deviates", nx=nx, ny=ny, pw=pw, ph=ph, r_max=r_max)))
        if int(n.get("chem", "chem_opt", 0) or 0) > 0:
            bad = ph < 21
            out.append(_find("decomp.chem_j_patch", "all", "DEVIATES" if bad else "OK",
                             _fmt("decomp.chem_j_patch", "deviates" if bad else "ok", chem_opt=n.get("chem", "chem_opt"), pj=ph)))
    else:
        out.append(_find("decomp.min_patch", "all", "OK", "nproc_x/nproc_y 自動分解；R_max = %d ranks（所有網格最小 patch 10）" % r_max))
    return out


def _chk_nudge(n):
    out = []
    fd = [int(n.dom("fdda", "grid_fdda", d["i"], 0) or 0) for d in n.doms]
    if not any(fd):
        return out
    d0 = n.doms[0]
    lam_ref = 1000.0
    if fd[0] == 2 and d0["e_we"] and d0["dx_m"]:
        lx, ly = (d0["e_we"] - 1) * d0["dx_m"] / 1000.0, (d0["e_sn"] - 1) * d0["dx_m"] / 1000.0
        xw, yw = int(n.dom("fdda", "xwavenum", 0, 0) or 0), int(n.dom("fdda", "ywavenum", 0, 0) or 0)
        rxr = (int(round(lx / 1500.0)), int(round(lx / lam_ref)))
        ryr = (int(round(ly / 1500.0)), int(round(ly / lam_ref)))
        ok = rxr[0] <= xw <= rxr[1] and ryr[0] <= yw <= ryr[1]
        lam_x = lx / xw if xw else float("inf")
        lam_y = ly / yw if yw else float("inf")
        out.append(_find("nudge.wavenumber", "d01", "OK" if ok else "DEVIATES",
                         _fmt("nudge.wavenumber", "ok" if ok else "deviates", xw=xw, yw=yw, rx=rxr, ry=ryr, Lx=lx, Ly=ly, lam=lam_ref,
                              lam_eff_x=lam_x, lam_eff_y=lam_y)))
    if fd[0] > 0:
        keys = ("guv", "gt", "gph") if fd[0] == 2 else ("guv", "gt")
        vals = [float(n.dom("fdda", k, 0, 0.0) or 0.0) for k in keys]
        tau = 1.0 / vals[0] / 60.0 if vals[0] else float("inf")
        ok = all(abs(v - 3e-4) < 1e-12 for v in vals)
        why = "未設定時 Registry 預設為 0（等於沒有 nudging 效果）" if not any(vals) else ""
        out.append(_find("nudge.coefficients", "d01", "OK" if ok else "DEVIATES",
                         _fmt("nudge.coefficients", "ok" if ok else "deviates", vals=vals, tau=tau) + ("; " + why if why else "")))
        gq = float(n.dom("fdda", "gq", 0, 0.0) or 0.0)
        out.append(_find("nudge.moisture", "d01", "DEVIATES" if gq > 0 else "OK", _fmt("nudge.moisture", "deviates" if gq > 0 else "ok", gq=gq)))
        if fd[0] == 2:
            flags = {k: int(n.dom("fdda", "if_no_pbl_nudging_" + k, 0, 0) or 0) for k in ("uv", "t", "ph")}
            zf = any(int(n.dom("fdda", "if_zfac_" + k, 0, 0) or 0) for k in ("uv", "t", "ph"))
            ok = all(flags.values()) or zf
            out.append(_find("nudge.pbl", "d01", "OK" if ok else "DEVIATES",
                             _fmt("nudge.pbl", "ok" if ok else "deviates", flags=flags)))
        gi = n.dom("fdda", "gfdda_interval_m", 0)
        ge = n.dom("fdda", "gfdda_end_h", 0)
        isec = n.get("time_control", "interval_seconds")
        rh = n.run_s / 3600.0 if n.run_s else None
        why = []
        if gi is not None and isec and abs(float(gi) - float(isec) / 60.0) > 1e-9:
            why.append("gfdda_interval_m %g 分鐘 != interval_seconds/60 = %g" % (gi, float(isec) / 60.0))
        if ge is not None and rh and float(ge) < rh - 1e-9:
            why.append("gfdda_end_h %g h < 預報長度 %.2f h：nudging 提早結束" % (ge, rh))
        if gi is None:
            why.append("未設定 gfdda_interval_m")
        out.append(_find("nudge.timing", "d01", "DEVIATES" if why else "OK",
                         _fmt("nudge.timing", "deviates" if why else "ok", why="; ".join(why), gi=gi, ge=ge, rh=rh or 0.0)))
    nests = [d["name"] for d, f in zip(n.doms, fd) if d["i"] > 0 and f > 0]
    out.append(_find("nudge.outer_only", "all", "DEVIATES" if nests else "OK",
                     _fmt("nudge.outer_only", "deviates" if nests else "ok", doms=",".join(nests))))
    return out


def _chk_les(n):
    out = []
    for d in n.doms:
        if not d["dx_m"]:
            continue
        kind, text = les_regime(d["dx_m"])
        diff = int(n.dom("dynamics", "diff_opt", d["i"], -1))
        km = int(n.dom("dynamics", "km_opt", d["i"], -1))
        pbl = n.dom("physics", "bl_pbl_physics", d["i"])
        why = []
        if kind == "les":
            if diff != 2:
                why.append("diff_opt = %d（LES 要 2）" % diff)
            if km not in (2, 3, 5):
                why.append("km_opt = %d（LES 要 2/3/5）" % km)
            if pbl not in (None, 0):
                why.append("bl_pbl_physics = %s（LES 要 0）" % pbl)
        if why:
            out.append(_find("les.regime", d["name"], "DEVIATES", _fmt("les.regime", "deviates", dx_m=d["dx_m"], regime=text, why="; ".join(why))))
        elif kind == "les":
            out.append(_find("les.regime", d["name"], "OK", _fmt("les.regime", "ok", dx_m=d["dx_m"], regime=text)))
        elif kind == "gray" and pbl not in (None, 0):
            out.append(_find("les.regime", d["name"], "OK", _fmt("les.regime", "ok", dx_m=d["dx_m"], regime=text)))
        if d["dx_m"] > 2000.0 and km in (2, 3):
            out.append(_find("les.km_opt_dx", d["name"], "DEVIATES", _fmt("les.km_opt_dx", "deviates", km=km, dx_m=d["dx_m"])))
    return out


def _chk_acoustic(n):
    out = []
    for d in n.doms:
        if not (d["dt"] and d["dx_m"]):
            continue
        tss = int(n.dom("dynamics", "time_step_sound", d["i"], 0) or 0)
        auto = sound_steps_auto(d["dt"], d["dx_m"])
        cac_auto = 340.0 * (d["dt"] / auto) / d["dx_m"]
        if tss == 0:
            out.append(_find("acoustic.sound_steps", d["name"], "OK",
                             _fmt("acoustic.sound_steps", "ok", n=auto, dts=d["dt"] / auto, cac=cac_auto) + "（自動）"))
        else:
            cac = 340.0 * (d["dt"] / tss) / d["dx_m"]
            bad = tss < auto
            out.append(_find("acoustic.sound_steps", d["name"], "DEVIATES" if bad else "OK",
                             _fmt("acoustic.sound_steps", "deviates" if bad else "ok", n=tss, dts=d["dt"] / tss, cac=cac, cac_auto=cac_auto, n_auto=auto)))
    return out


def _chk_movenest(n):
    if not any(n.has("domains", k) for k in ("vortex_interval", "max_vortex_speed", "corral_dist")):
        return []
    out = []
    for d in n.doms[1:]:
        if not d["dx_m"]:
            continue
        vi = float(n.dom("domains", "vortex_interval", d["i"], 15))
        vs = float(n.dom("domains", "max_vortex_speed", d["i"], 40))
        out.append(_find("movenest.vortex", d["name"], "OK",
                         "渦旋搜尋半徑 %.1f km = %.1f 個巢狀格點（vortex_interval %g 分鐘、max_vortex_speed %g m/s、corral_dist %s 個母格點）"
                         % (60 * vi * vs / 1000.0, 60 * vi * vs / d["dx_m"], vi, vs, n.dom("domains", "corral_dist", d["i"], 8))))
    return out


def _chk_longrun(n):
    out = []
    if not n.run_s:
        return out
    days = n.run_s / 86400.0
    sst = int(n.get("physics", "sst_update", 0) or 0)
    if days > 5 and not sst:
        out.append(_find("longrun.threshold", "all", "DEVIATES",
                         "預報 %.1f 天 > ~5 天但 sst_update = 0：SST 固定在初始值（建議 sst_update=1 並設 auxinput4_interval）" % days))
    if sst:
        why = []
        a4 = n.dom("time_control", "auxinput4_interval", 0)
        isec = n.get("time_control", "interval_seconds")
        if a4 is None:
            why.append("sst_update=1 但未設 auxinput4_interval")
        elif isec and abs(float(a4) - float(isec) / 60.0) > 1e-9:
            why.append("auxinput4_interval %g 分鐘 != interval_seconds/60 = %g（README：一般與其一致）" % (a4, float(isec) / 60.0))
        if not n.get("physics", "usemonalb", False):
            why.append("usemonalb 不是 .true.（README 建議搭配 sst_update=1）")
        out.append(_find("longrun.sst", "all", "DEVIATES" if why else "OK", "; ".join(why) or "sst_update=1 且 auxinput4_interval 一致"))
    return out


def _chk_io_run(n):
    out = []
    for d in n.doms:
        if d["e_we"] and d["e_sn"] and d["e_vert"]:
            fb = io_frame_bytes(d["e_we"], d["e_sn"], d["e_vert"])
            out.append(_find("io.frame_first_principles", d["name"], "OK", "每個 history frame 約 %.0f MB（估計值，n3d=%d n2d=%d，下限）" % (fb / 1e6, IO_N3D, IO_N2D)))
    if n.dt0 and n.run_s:
        steps = int(round(n.run_s / n.dt0))
        out.append(_find("run.steps", "d01", "OK", _fmt("run.steps", "ok", steps=steps, run_h=n.run_s / 3600.0, dt=n.dt0)))
    return out


CHECKERS = (_chk_dt, _chk_radt, _chk_nest, _chk_bdy, _chk_vert, _chk_physics, _chk_decomp, _chk_nudge,
            _chk_les, _chk_acoustic, _chk_movenest, _chk_longrun, _chk_io_run)


def check(namelist_path, geog_res_arcsec=30.0):
    n = _NL(namelist_path)
    findings = []
    for fn in CHECKERS:
        findings.extend(fn(n))
    findings.extend(_chk_geog(n, geog_res_arcsec))
    counts = {s: sum(1 for f in findings if f["status"] == s) for s in ("OK", "DEVIATES", "ERROR")}
    return {"namelist": str(namelist_path), "geog_res_arcsec": geog_res_arcsec, "max_dom": n.max_dom,
            "findings": findings, "counts": counts}


# ================================================================================== rendering

_LEVEL_ZH = {"wrf-doc": "WRF 官方文件", "wtune-measured": "WTune 實測", "rule-of-thumb": "經驗值"}
_STATUS_ZH = {"OK": "符合", "DEVIATES": "偏離", "ERROR": "錯誤"}


def _fmt_value(v):
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def render_result(res):
    lines = ["wtune calc %s  輸入: %s" % (res["topic"], json.dumps(res["inputs"], ensure_ascii=False))]
    for it in res["items"]:
        lines.append("- %s: %s %s%s" % (it["label"], _fmt_value(it["value"]), it["unit"], "（%s）" % it["note"] if it["note"] else ""))
        lines.append("    公式: %s" % it["formula"])
        lines.append("    來源等級: %s（%s）  出處: %s" % (it["source_level"], _LEVEL_ZH[it["source_level"]], it["source_ref"]))
    return "\n".join(lines)


def render_check(rep):
    lines = ["wtune calc check %s  （地形/地理資料解析度假設 %s 角秒；僅提示，不修改 namelist）" % (rep["namelist"], rep["geog_res_arcsec"])]
    for f in rep["findings"]:
        lines.append("[%s %s] %s %s: %s" % (f["status"], _STATUS_ZH[f["status"]], f["domain"], f["rule_id"], f["reason"]))
        lines.append("    來源等級: %s（%s）  出處: %s" % (f["source_level"], _LEVEL_ZH[f["source_level"]], f["source_ref"]))
    c = rep["counts"]
    lines.append("合計: 符合 %d、偏離 %d、錯誤 %d（只有 WRF 程式碼會中止的 wrf-doc 硬性條件才算錯誤；經驗值偏離只是偏離）"
                 % (c["OK"], c["DEVIATES"], c["ERROR"]))
    return "\n".join(lines)


# ======================================================================================== CLI

def _build_parser():
    p = argparse.ArgumentParser(prog="wtune calc", description="rules of thumb and unit conversions (decision 0062); advice only")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    sub = p.add_subparsers(dest="topic")

    def add(name, help_, *args):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable JSON output")
        for a, kw in args:
            sp.add_argument(a, **kw)
        return sp

    f = lambda **kw: dict(type=float, **kw)
    i = lambda **kw: dict(type=int, **kw)
    add("dt", "time_step, adaptive-dt defaults, nest dt", ("--dx-km", f(required=True)),
        ("--terrain", dict(choices=["typical", "steep"], default="typical")), ("--max-map-factor", f(default=1.0)),
        ("--parent-grid-ratio", i(default=None)))
    add("cfl", "Courant numbers", ("--u", f(required=True)), ("--dx-m", f(required=True)), ("--dt", f(required=True)),
        ("--w", f(default=None)), ("--dz-m", f(default=None)))
    add("radt", "radiation call interval", ("--dx-km", f(required=True)), ("--dt", f(default=None)))
    add("nest", "nest ratio, dimensions, placement", ("--parent-grid-ratio", i(required=True)),
        ("--parent-cells-x", i(default=None)), ("--parent-cells-y", i(default=None)), ("--parent-dx-km", f(default=None)),
        ("--i-parent-start", i(default=None)), ("--j-parent-start", i(default=None)),
        ("--parent-e-we", i(default=None)), ("--parent-e-sn", i(default=None)),
        ("--e-we", i(default=None)), ("--e-sn", i(default=None)))
    add("domain", "e_we/e_sn, projection, truelat from centre and extent", ("--lat", f(required=True)), ("--lon", f(required=True)),
        ("--width-km", f(required=True)), ("--height-km", f(required=True)), ("--dx-km", f(required=True)))
    add("vert", "model top, layer thickness, damping, eta -> height", ("--e-vert", i(required=True)),
        ("--p-top-pa", f(default=5000.0)), ("--eta", dict(type=float, nargs="*", default=None)))
    add("physics", "cumulus recommendation from dx", ("--dx-km", f(required=True)), ("--cu-physics", i(default=None)))
    add("geog", "geog data resolution vs dx", ("--dx-km", f(required=True)), ("--res-arcsec", f(default=30.0)), ("--lat", f(default=0.0)))
    add("decomp", "max ranks / legal decompositions", ("--e-we", dict(type=int, nargs="+", required=True)),
        ("--e-sn", dict(type=int, nargs="+", required=True)), ("--ranks", i(default=None)), ("--cores-per-node", i(default=None)))
    add("io", "history/restart frame size", ("--e-we", i(required=True)), ("--e-sn", i(required=True)), ("--e-vert", i(required=True)),
        ("--frames", i(default=1)), ("--n3d", i(default=None)), ("--n2d", i(default=None)))
    add("run", "steps and (with --nodes/--nx/--ny) wtune_predict wall time", ("namelist", dict()),
        ("--nodes", i(default=None)), ("--nx", i(default=None)), ("--ny", i(default=None)))
    add("units", "unit conversions", ("kind", dict(choices=UNIT_KINDS)), ("value", dict(type=float)),
        ("--lat", f(default=0.0)), ("--axis", dict(choices=["lat", "lon"], default="lat")))
    add("nudge", "spectral/analysis nudging parameters", ("--e-we", i(required=True)), ("--e-sn", i(required=True)),
        ("--dx-m", f(required=True)), ("--wavelength-km", f(default=1000.0)), ("--g", f(default=3e-4)))
    add("scales", "Coriolis, Rossby radius, N, Froude, effective resolution", ("--lat", f(default=None)), ("--dx-km", f(default=None)),
        ("--N", f(default=None)), ("--H", f(default=None)), ("--U", f(default=None)), ("--h", f(default=None)),
        ("--T", f(default=288.0)), ("--lapse-rate", f(default=None)), ("--feature-km", f(default=None)))
    add("les", "LES / gray-zone settings from dx", ("--dx-m", f(required=True)))
    add("acoustic", "sound steps, acoustic Courant, epssm", ("--dt", f(required=True)), ("--dx-m", f(required=True)),
        ("--c-s", f(default=340.0)), ("--time-step-sound", i(default=0)))
    add("bdy", "boundary zone and nest distance", ("--spec-bdy-width", i(default=5)), ("--parent-cells", f(default=None)))
    add("movenest", "vortex-following moving nest parameters", ("--dx-m", f(required=True)), ("--storm-speed", f(default=None)),
        ("--parent-dx-km", f(default=None)), ("--vortex-interval", f(default=15)), ("--max-vortex-speed", f(default=40)))
    add("longrun", "SST update for long runs", ("--run-days", f(default=None)), ("--interval-seconds", f(default=21600)))
    add("spinup", "spin-up guidance", ("--run-hours", f(default=None)))
    add("check", "apply every applicable rule to a namelist.input", ("namelist", dict()),
        ("--geog-res-arcsec", f(default=30.0)))
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.topic:
        parser.print_help()
        return 0
    t = args.topic
    as_json = getattr(args, "json", False)
    try:
        if t == "check":
            rep = check(args.namelist, args.geog_res_arcsec)
            print(json.dumps(rep, indent=2, ensure_ascii=False) if as_json else render_check(rep))
            return 1 if rep["counts"]["ERROR"] else 0
        if t == "dt":
            res = dt(args.dx_km, args.terrain, args.max_map_factor, args.parent_grid_ratio)
        elif t == "cfl":
            res = cfl(args.u, args.dx_m, args.dt, args.w, args.dz_m)
        elif t == "radt":
            res = radt(args.dx_km, args.dt)
        elif t == "nest":
            res = nest(args.parent_grid_ratio, args.parent_cells_x, args.parent_cells_y, args.parent_dx_km,
                       args.i_parent_start, args.j_parent_start, args.parent_e_we, args.parent_e_sn, args.e_we, args.e_sn)
        elif t == "domain":
            res = domain(args.lat, args.lon, args.width_km, args.height_km, args.dx_km)
        elif t == "vert":
            res = vert(args.e_vert, args.p_top_pa, eta=args.eta)
        elif t == "physics":
            res = physics(args.dx_km, args.cu_physics)
        elif t == "geog":
            res = geog(args.dx_km, args.res_arcsec, args.lat)
        elif t == "decomp":
            res = decomp(args.e_we, args.e_sn, args.ranks, args.cores_per_node)
        elif t == "io":
            res = io(args.e_we, args.e_sn, args.e_vert, args.frames, args.n3d, args.n2d)
        elif t == "run":
            res = run(args.namelist, args.nodes, args.nx, args.ny)
        elif t == "units":
            res = units(args.kind, args.value, args.lat, args.axis)
        elif t == "nudge":
            res = nudge(args.e_we, args.e_sn, args.dx_m, args.wavelength_km, args.g)
        elif t == "scales":
            res = scales(args.lat, args.dx_km, args.N, args.H, args.U, args.h, args.T, args.lapse_rate, args.feature_km)
        elif t == "les":
            res = les(args.dx_m)
        elif t == "acoustic":
            res = acoustic(args.dt, args.dx_m, args.c_s, args.time_step_sound)
        elif t == "bdy":
            res = bdy(args.spec_bdy_width, args.parent_cells)
        elif t == "movenest":
            res = movenest(args.dx_m, args.storm_speed, args.parent_dx_km, args.vortex_interval, args.max_vortex_speed)
        elif t == "longrun":
            res = longrun(args.run_days, args.interval_seconds)
        else:
            res = spinup(args.run_hours)
    except Exception as exc:  # advisory tool: report, never traceback
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, ensure_ascii=False) if as_json else render_result(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
