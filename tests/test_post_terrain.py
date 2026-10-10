"""Tests for post/recipes/terrain.py.

_terrain_domain() (2026-09-29 third lead review): every terrain figure must target the
finest domain that ITSELF satisfies the terrain relief/peak rule, not blindly the run's
overall finest domain -- katrina m1's real case has d01 (the coarse parent, covering
Mexico's own relief) qualifying while d02 (a nest entirely over the Gulf of Mexico) does
not, and every terrain_*_d02.png was meaningless as a result (a stray coastal hill picked as
the "ridge crest", a flat silhouette, a cross-section line running outside d02 itself).

_find_valley_point()/_valley_time_height() (2026-09-29 fourth lead review): the previous
version picked the domain's global HGT minimum with no land/interior/relief check, landing
on a sea point in the lateral boundary zone (HGT=-1 m, flat theta profile). A genuine valley
must be land (LANDMASK==1), inside the domain interior, and a real local minimum with
sufficient surrounding relief; otherwise the figure is reported missing(), not drawn at a
meaningless point.

_fill_below_valid() (2026-09-29 sixth lead review, item 1): matplotlib's contourf drops any
grid quad with even one masked corner, so masking below-ground cells (the fifth review's own
fix) was itself the bug -- a masked cell in a taller neighbouring column wedged out valid,
above-ground cells in a shorter column right next to it. Fill every column's NaN levels below
its own lowest FINITE sample with that sample's value, all the way down to the array's own
bottom (index 0) -- never above the lowest valid sample, but no longer stopping at the local
terrain height. The terrain silhouette polygon (drawn at a higher zorder in
_cross_section_figure) is what hides the underground portion now, not a data mask.
"""
import os
import sys
import types

import numpy as np

from wtune.post.recipes import terrain


def _case_info(domains, terrain_domains):
    return dict(domains=[{"domain": d} for d in domains], terrain_domains=terrain_domains)


def test_terrain_domain_picks_finest_qualifying_domain_when_only_d01_qualifies():
    case_info = _case_info(domains=[1, 2], terrain_domains=[1])
    assert terrain._terrain_domain(case_info) == 1


def test_terrain_domain_picks_finest_qualifying_domain_when_both_qualify():
    case_info = _case_info(domains=[1, 2], terrain_domains=[1, 2])
    assert terrain._terrain_domain(case_info) == 2


def test_terrain_domain_falls_back_to_overall_finest_when_none_qualify():
    """--types terrain forced on a case where no domain actually meets the rule (there is no
    "correct" domain in that situation) -- keeps the pre-existing default rather than
    failing outright."""
    case_info = _case_info(domains=[1, 2, 3], terrain_domains=[])
    assert terrain._terrain_domain(case_info) == 3


def test_terrain_domain_handles_missing_key_gracefully():
    """case_info dicts built before this fix (or from a code path that never set
    terrain_domains) must not crash -- same fallback as the empty-list case."""
    case_info = dict(domains=[{"domain": 1}, {"domain": 2}])
    assert terrain._terrain_domain(case_info) == 2


# -- _find_valley_point() ---------------------------------------------------------------

_INTERIOR_BOUNDS = (5, 10, 5, 10)   # a 15x15 grid, bdy_width=5 on every edge


def _flat(value=500.0, shape=(15, 15)):
    return np.full(shape, value, dtype=float)


def test_find_valley_point_rejects_a_sea_point():
    """The only local-minimum-with-relief candidate is SEA (LANDMASK==0) -- must be
    rejected, not picked because it happens to be the deepest 'valley'."""
    hgt = _flat()
    landmask = np.ones((15, 15))
    hgt[7, 7] = 0.0
    landmask[7, 7] = 0.0
    assert terrain._find_valley_point(hgt, landmask, _INTERIOR_BOUNDS) is None


def test_find_valley_point_rejects_a_point_in_the_boundary_zone():
    """A genuine land valley, but at (1,1) -- outside the 5-cell interior -- must be
    rejected even though it would otherwise qualify."""
    hgt = _flat()
    landmask = np.ones((15, 15))
    hgt[1, 1] = 0.0
    assert terrain._find_valley_point(hgt, landmask, _INTERIOR_BOUNDS) is None


def test_find_valley_point_rejects_insufficient_surrounding_relief():
    """Only 50 m lower than its neighbourhood (< the 300 m ridge-relief rule) -- not a real
    valley, just locally low terrain."""
    hgt = _flat()
    landmask = np.ones((15, 15))
    hgt[7, 7] = 450.0
    assert terrain._find_valley_point(hgt, landmask, _INTERIOR_BOUNDS) is None


def test_find_valley_point_finds_a_genuine_interior_land_valley():
    hgt = _flat()
    landmask = np.ones((15, 15))
    hgt[7, 7] = 0.0   # land, interior, 500 m of surrounding relief
    assert terrain._find_valley_point(hgt, landmask, _INTERIOR_BOUNDS) == (7, 7)


def test_find_valley_point_returns_none_for_a_perfectly_flat_domain():
    hgt = _flat()
    landmask = np.ones((15, 15))
    assert terrain._find_valley_point(hgt, landmask, _INTERIOR_BOUNDS) is None


# -- _valley_time_height()'s missing() propagation ---------------------------------------

class _FakeDataset:
    def __init__(self, path):
        self.variables = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_valley_time_height_reports_missing_when_no_point_found(monkeypatch, tmp_path):
    """_find_valley_point() returning None must surface as (None, a "missing" reason), not a
    crash or a figure drawn at a meaningless point -- the actual selection logic is covered
    by the _find_valley_point tests above; this only checks the propagation contract."""
    monkeypatch.setattr(terrain, "list_frame_refs",
                        lambda run_dir, domain=1: [("t0", "/fake/t0", 0), ("t1", "/fake/t1", 0)])
    monkeypatch.setattr("netCDF4.Dataset", _FakeDataset)
    fake_wrf = types.SimpleNamespace(
        getvar=lambda ds, name, timeidx=0: np.zeros((15, 15)),
        latlon_coords=lambda var, as_np=True: (np.zeros((15, 15)), np.zeros((15, 15))))
    monkeypatch.setitem(sys.modules, "wrf", fake_wrf)
    monkeypatch.setattr(terrain, "_find_valley_point", lambda hgt, landmask, bounds: None)

    path, reason = terrain._valley_time_height("/fake/run", "run0", 1, str(tmp_path),
                                                bdy_width=5)
    assert path is None
    assert reason and "valley" in reason


def test_valley_time_height_reports_missing_with_fewer_than_two_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(terrain, "list_frame_refs", lambda run_dir, domain=1: [("t0", "/fake/t0", 0)])
    path, reason = terrain._valley_time_height("/fake/run", "run0", 1, str(tmp_path))
    assert path is None
    assert "fewer than 2 frames" in reason


# -- _fill_below_valid() -------------------------------------------------------------------

def test_fill_below_valid_extrapolates_all_the_way_to_the_bottom():
    """Sixth lead review: fill down to index 0 unconditionally (no terrain boundary any
    more) -- this is what lets the terrain silhouette polygon, not a data mask, be the only
    thing hiding the underground portion, since a data mask was itself dropping valid
    neighbouring-column quads in matplotlib's contourf."""
    arr = np.array([
        [np.nan, np.nan],   # index 0
        [np.nan, 5.0],      # index 1
        [10.0, 6.0],        # index 2 -- column 0's first valid sample
        [11.0, 7.0],
        [12.0, 8.0],
    ])
    out = terrain._fill_below_valid(arr)

    # column 0: every level below its first valid sample (index 2, value 10.0) is filled with
    # that value, all the way to index 0 -- nothing left NaN.
    assert list(out[:3, 0]) == [10.0, 10.0, 10.0]
    assert list(out[3:, 0]) == [11.0, 12.0]   # values above the first valid sample untouched

    # column 1: first valid sample already at index 1 (5.0) -> only index 0 is filled.
    assert list(out[:, 1]) == [5.0, 5.0, 6.0, 7.0, 8.0]


def test_fill_below_valid_never_fills_above_the_lowest_valid_sample():
    arr = np.array([[np.nan], [np.nan], [20.0], [21.0]])
    out = terrain._fill_below_valid(arr)
    assert list(out[:, 0]) == [20.0, 20.0, 20.0, 21.0]


def test_fill_below_valid_does_not_modify_the_input_array():
    arr = np.array([[np.nan], [5.0]])
    terrain._fill_below_valid(arr)
    assert np.isnan(arr[0, 0])   # the original array is untouched -- the function returns a copy


def test_fill_below_valid_leaves_an_all_nan_column_untouched():
    arr = np.full((3, 1), np.nan)
    out = terrain._fill_below_valid(arr)
    assert np.all(np.isnan(out))


# --- lee-side station diagnostic -------------------------------------------------------

def _stn(sid, lat, lon):
    return dict(id=sid, name=sid, icao="", lat=str(lat), lon=str(lon), elev_m="100")


def test_along_off_km_flow_toward_east():
    along, off = terrain.along_off_km((40.0, -110.0), 90.0, 40.0, -109.4)   # ~51 km east
    assert 50 < along < 52 and abs(off) < 0.5
    along, off = terrain.along_off_km((40.0, -110.0), 90.0, 40.0, -110.6)   # upstream
    assert along < -50


def test_lee_candidates_window_and_ordering():
    crest = (40.0, -110.0)
    st = [_stn("UP", 40.0, -110.5),          # upstream
          _stn("NEAR", 40.0, -109.8),        # ~17 km: too close
          _stn("LEE1", 40.0, -109.4),        # ~51 km
          _stn("LEE2", 40.0, -109.0),        # ~85 km
          _stn("OFF", 40.4, -109.4),         # 44 km across
          _stn("FAR", 40.0, -108.5)]         # ~128 km
    cands, reason = terrain.lee_station_candidates(st, crest, 90.0)
    assert [c["station"]["id"] for c in cands] == ["LEE1", "LEE2"] and reason is None


def test_lee_candidates_none_reason_is_explicit():
    cands, reason = terrain.lee_station_candidates([_stn("UP", 40.0, -110.5)], (40.0, -110.0), 90.0)
    assert cands == []
    assert "no fetched station within 25 km of the section line, 30-100 km downstream of the crest" in reason
    assert "UP" in reason and "not implemented" not in reason


def test_leeward_station_missing_reasons(tmp_path):
    assert "no wtune project" in terrain._leeward_station("r", "l", 2, str(tmp_path), None, 6, [])[1]
    proj = tmp_path / "p"
    (proj / "input" / "obs" / "stations").mkdir(parents=True)
    path, reason = terrain._leeward_station("r", "l", 2, str(tmp_path), str(proj), 6, [])
    assert path is None and "not fetched yet" in reason


def _write_isd(proj, sid, t0, n, temp=100, dew=0, wspd=50):
    import gzip
    usaf, wban = sid.split("-")
    d = proj / "input" / "obs" / "stations"
    d.mkdir(parents=True, exist_ok=True)
    from datetime import timedelta
    lines = []
    for k in range(n):
        t = t0 + timedelta(hours=k)
        lines.append("%d %02d %02d %02d %d %d -9999 180 %d -9999 -9999 -9999" %
                     (t.year, t.month, t.day, t.hour, temp, dew, wspd))
    with gzip.open(d / ("%s-%s-%d.gz" % (usaf, wban, t0.year)), "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def test_leeward_station_end_to_end_synthetic(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    proj = tmp_path / "p"
    (proj / "project.yaml").parent.mkdir(parents=True, exist_ok=True)
    stations = [_stn("111111-22222", 40.0, -109.4), _stn("333333-44444", 40.0, -109.0)]
    sd = proj / "input" / "obs" / "stations"
    sd.mkdir(parents=True)
    with open(sd / "stations.csv", "w") as fh:
        fh.write("id,name,icao,lat,lon,elev_m\n")
        for s in stations:
            fh.write("%s,%s,,%s,%s,100\n" % (s["id"], s["name"], s["lat"], s["lon"]))
    t0 = datetime(2019, 1, 28)
    _write_isd(proj, "333333-44444", t0, 24)            # only the second station has obs
    frames = [((t0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S"), "/f%d" % h) for h in range(24)]
    info = dict(peak_ll=(40.0, -110.0), bearing_deg=90.0, interior_bounds=(5, 95, 5, 95))

    def fake_model(sel, lat, lon, bounds):
        return dict(times=[datetime.strptime(t, "%Y-%m-%d_%H:%M:%S") for t, _ in sel],
                    t2_c=[5.0] * len(sel), rh2=[50.0] * len(sel), wspd10=[3.0] * len(sel),
                    hgt_m=120.0, i=10, j=10)
    monkeypatch.setattr(terrain, "_model_series_at", fake_model)
    path, extra = terrain._leeward_station("run", "lab", 2, str(tmp_path), str(proj), 6, frames,
                                           line_info=info)
    assert path and path.endswith("terrain_leeward_station_d02.png") and os.path.getsize(path) > 0
    assert extra["station"] == "333333-44444"          # first candidate skipped: no observations
    assert extra["n_obs"] == 18 and 80 < extra["along_km"] < 90


def test_leeward_station_skipped_when_outside_interior(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    proj = tmp_path / "p"
    sd = proj / "input" / "obs" / "stations"
    sd.mkdir(parents=True)
    (sd / "stations.csv").write_text("id,name,icao,lat,lon,elev_m\n111111-22222,A,,40.0,-109.4,100\n")
    t0 = datetime(2019, 1, 28)
    _write_isd(proj, "111111-22222", t0, 24)
    frames = [((t0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S"), "/f") for h in range(24)]
    info = dict(peak_ll=(40.0, -110.0), bearing_deg=90.0, interior_bounds=(5, 95, 5, 95))
    monkeypatch.setattr(terrain, "_model_series_at", lambda *a: None)
    path, reason = terrain._leeward_station("r", "l", 2, str(tmp_path), str(proj), 6, frames, line_info=info)
    assert path is None and "outside model interior" in reason


def test_terrain_figures_takes_project_dir_and_no_longer_hardcodes_missing():
    import inspect
    assert "project_dir" in inspect.signature(terrain.figures).parameters
    src = inspect.getsource(terrain.figures)
    assert "not implemented" not in src and "no station observations fetched this round" not in src
