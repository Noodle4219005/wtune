#!/usr/bin/env python3
"""pytest suite for predictor/decomp_space.py (WTune D1 decomposition-space enumeration).

Every geometry number asserted below is either hand-computed against the cited WRF source
(external/RSL_LITE/task_for_point.c for MPI patches, frame/module_machine.F / frame/
module_tiles.F for tile splitting) or derived from one of those hand-computed facts -- see each
test's docstring/comment for the arithmetic. Run only via the predictor pixi env (decomp_space.py
needs Python >= 3.7), never the login node.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtune.predictor import decomp_space as ds   # noqa: E402
from wtune.predictor import features as feat     # noqa: E402


def _group_sizes(npoints, nparts):
    """[size of group 0, size of group 1, ...] via ds._region_bounds -- 0 for an empty group."""
    sizes = []
    for p in range(nparts):
        s, e = ds._region_bounds(1, npoints, nparts, p)
        sizes.append(0 if s is None else e - s + 1)
    return sizes


# ---------------------------------------------------------------------------
# RSL-lite MPI patch extents: external/RSL_LITE/task_for_point.c
# ---------------------------------------------------------------------------
class TestRslPatchExtents:
    def test_901_over_8(self):
        # 901 = 8*112 + 5 -> min patch 112 (this IS the value WRF's own legality gate divides,
        # share/module_check_a_mundo.F:3017-3018's Fortran integer division), max 113.
        assert ds.rsl_patch_extents(901, 8) == (112, 113)

    def test_805_over_79_isc21_d02_extent(self):
        # 805 (isc21's e_we_list[1]/e_sn_list[1], campaign/cases.json) / 79 = 79*10 + 15
        assert ds.rsl_patch_extents(805, 79) == (10, 11)

    def test_exact_division_no_remainder(self):
        assert ds.rsl_patch_extents(90, 9) == (10, 10)

    def test_odd_remainder(self):
        assert ds.rsl_patch_extents(10, 3) == (3, 4)


class TestLegalShapesExact:
    def test_boundary_10_is_legal_9_is_illegal(self):
        """WRF's real startup gate is a strict '< 10' on Fortran INTEGER division
        (share/module_check_a_mundo.F:3017-3018) -- floor(extent/nproc) == 10 passes,
        == 9 fails. 90 // 9 == 10 exactly; 89 // 9 == 9."""
        assert (9, 9) in ds.legal_shapes_exact([90], [90], 81)
        assert (9, 9) not in ds.legal_shapes_exact([89], [89], 81)

    def test_illegal_on_smaller_nested_domain_even_when_legal_on_d01(self):
        # d01 100x100: floor(100/10) = 10, legal. d02 50x50 at the SAME (nx,ny): floor(50/10) = 5,
        # illegal -- the whole shape must be rejected, not scored on d01 alone.
        shapes = ds.legal_shapes_exact([100, 50], [100, 50], 100)
        assert (10, 10) not in shapes

    def test_legal_when_every_domain_passes(self):
        shapes = ds.legal_shapes_exact([100, 100], [100, 100], 100)
        assert (10, 10) in shapes

    def test_three_domain_boundary_isc21_shaped(self):
        """isc21-shaped 3-domain case (campaign/cases.json e_we_list=[793,805,1001]):
        floor(90/9)=10 legal on all three when every domain is exactly 90-wide; shrinking the
        THIRD domain alone to 89 makes floor(89/9)=9 < 10 and must reject the whole shape even
        though d01 and d02 both still pass."""
        we_all_90 = [90, 90, 90]
        assert (9, 9) in ds.legal_shapes_exact(we_all_90, we_all_90, 81)
        we_d03_shrunk = [90, 90, 89]
        assert (9, 9) not in ds.legal_shapes_exact(we_d03_shrunk, we_d03_shrunk, 81)

    def test_returns_only_factor_pairs_of_ranks(self):
        shapes = ds.legal_shapes_exact([1000], [1000], 12)
        for nx, ny in shapes:
            assert nx * ny == 12


# ---------------------------------------------------------------------------
# Tile splitting: frame/module_machine.F rlocproc/locproc/region_bounds -- a DIFFERENT remainder
# algorithm from RSL-lite's (see decomp_space.py's module docstring).
# ---------------------------------------------------------------------------
class TestRlocprocHandComputed:
    def test_ten_points_three_groups_remainder_goes_entirely_to_front(self):
        """Hand-traced from frame/module_machine.F:15-46: maxi=10, nproc=3 -> rem=1 (odd, <=2),
        so the width bump (`rem>0 .and. (mod(rem,2)==0 .or. rem>2 .or. p<=mid)`) applies only to
        queries with p<=mid=5. Tracing p=0..9 gives group0={0,1,2,3} (size4), group1={4,5,6}
        (size3), group2={7,8,9} (size3) -- the lone extra point goes ENTIRELY to the front group,
        unlike RSL-lite's MPI-patch rule which would split it front+back."""
        assert _group_sizes(10, 3) == [4, 3, 3]

    def test_eleven_points_three_groups_splits_both_ends(self):
        """maxi=11, nproc=3 -> rem=2 (even), so the width bump is UNCONDITIONAL (rem%2==0 is
        already true) -- both bl and br see width+1 for every query, so both the front AND back
        groups get the extra point: sizes (4, 3, 4)."""
        assert _group_sizes(11, 3) == [4, 3, 4]

    def test_even_division_no_remainder(self):
        assert _group_sizes(12, 3) == [4, 4, 4]

    def test_single_group_gets_everything(self):
        assert _group_sizes(7, 1) == [7]

    def test_sizes_always_sum_to_npoints(self):
        for npoints, nparts in ((17, 5), (23, 4), (100, 7), (8, 16)):
            assert sum(_group_sizes(npoints, nparts)) == npoints


class TestLeastAspect:
    def test_12_gives_3_by_4(self):
        # divisor pairs of 12 in ascending y: (1,12)(2,6)(3,4)(4,3)(6,2)(12,1); |y-x| strictly
        # decreases through (1,12)->11, (2,6)->4, (3,4)->1, then (4,3) ties at 1 (not <, so the
        # ascending-y loop keeps the FIRST minimum it found: (3,4)).
        assert ds.least_aspect(12) == (3, 4)

    def test_prime_gives_1_by_n(self):
        assert ds.least_aspect(13) == (1, 13)

    def test_perfect_square(self):
        assert ds.least_aspect(16) == (4, 4)


class TestTileExtents:
    def test_1d_column_split_matches_hand_computed_10_3(self):
        tiles = ds.tile_extents(patch_x=10, patch_y=1, numtiles_x=3, numtiles_y=1)
        assert tiles == [(1, 4, 1, 1), (5, 7, 1, 1), (8, 10, 1, 1)]

    def test_2d_even_split(self):
        tiles = ds.tile_extents(patch_x=6, patch_y=4, numtiles_x=2, numtiles_y=2)
        assert tiles == [(1, 3, 1, 2), (4, 6, 1, 2), (1, 3, 3, 4), (4, 6, 3, 4)]

    def test_more_groups_than_points_skips_empty_tiles(self):
        # patch_y=8 split into 16 y-groups: sum of realized sizes == 8, so AT MOST 8 of the 16
        # groups can be non-empty -- and since 8 points can't split more than 8 ways, exactly 8
        # realize (each getting 1 point). numtiles_x=1 means every realized tile spans the full
        # width (1..225).
        tiles = ds.tile_extents(patch_x=225, patch_y=8, numtiles_x=1, numtiles_y=16)
        assert len(tiles) == 8
        total_h = 0
        for i_s, i_e, j_s, j_e in tiles:
            assert (i_s, i_e) == (1, 225)
            total_h += j_e - j_s + 1
        assert total_h == 8


class TestResolveTileGeometry:
    def test_default_1d_y_no_switch_needed(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=8)
        assert r["num_tiles"] == 8
        assert (r["numtiles_x"], r["numtiles_y"]) == (1, 8)
        assert r["tile_strategy_used"] == ds.TILE_Y
        assert r["went_2d"] is False

    def test_auto_switch_to_2d_when_numtiles_exceeds_patch_height(self):
        # frame/module_tiles.F:333-348: numtiles(16) > patch_y(8)//MIN_TILE_SIZE(1) -> switch.
        # numtiles_y = 8 (one row per point), then numtiles_x grows until product >= 16 -> 2x8=16.
        r = ds.resolve_tile_geometry(patch_x=225, patch_y=8, numtiles=16)
        assert r["went_2d"] is True
        assert r["tile_strategy_used"] == ds.TILE_XY
        assert (r["numtiles_x"], r["numtiles_y"]) == (2, 8)
        assert r["num_tiles"] == 16

    def test_explicit_strategy_bypasses_auto_switch(self):
        # the auto-2D-switch only fires inside the TILE_NONE branch -- an explicit tile_strategy
        # skips it entirely, so 16 tiles get requested in 1D-Y over a height-8 patch (under-realized
        # at runtime, see TestTileExtents.test_more_groups_than_points_skips_empty_tiles).
        r = ds.resolve_tile_geometry(patch_x=225, patch_y=8, numtiles=16, tile_strategy=ds.TILE_Y)
        assert r["went_2d"] is False
        assert (r["numtiles_x"], r["numtiles_y"]) == (1, 16)
        assert r["num_tiles"] == 16

    def test_tile_sz_overrides_numtiles_entirely(self):
        # frame/module_tiles.F:353-359: 20//7=2 remainder 6 -> 3 tiles; 10//4=2 remainder 2 -> 3.
        r = ds.resolve_tile_geometry(patch_x=20, patch_y=10, numtiles=1, tile_sz_x=7, tile_sz_y=4)
        assert (r["numtiles_x"], r["numtiles_y"]) == (3, 3)
        assert r["num_tiles"] == 9

    def test_explicit_numtiles_x_and_y_both_given(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=1, numtiles_x=4, numtiles_y=2)
        assert (r["numtiles_x"], r["numtiles_y"]) == (4, 2)
        assert r["num_tiles"] == 8

    def test_explicit_numtiles_x_only_derives_y(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=8, numtiles_x=4)
        assert (r["numtiles_x"], r["numtiles_y"]) == (4, 2)

    def test_tile_x_strategy(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=6, tile_strategy=ds.TILE_X)
        assert (r["numtiles_x"], r["numtiles_y"]) == (6, 1)

    def test_xy_strategy_uses_least_aspect(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=12, tile_strategy=ds.TILE_XY)
        assert (r["numtiles_y"], r["numtiles_x"]) == ds.least_aspect(12, 1, 1) == (3, 4)


class TestDomainPatchGeometry:
    def test_single_domain_matches_rsl_patch_extents(self):
        doms = ds.domain_patch_geometry([901], [901], 8, 56)
        assert len(doms) == 1
        d = doms[0]
        assert (d["patch_x_min"], d["patch_x_max"]) == ds.rsl_patch_extents(901, 8)
        assert (d["patch_y_min"], d["patch_y_max"]) == ds.rsl_patch_extents(901, 56)
        assert d["halo_interior_ratio_min"] > 0

    def test_multi_domain_returns_one_dict_per_domain(self):
        doms = ds.domain_patch_geometry([793, 805, 1001], [853, 805, 1001], 8, 56)
        assert len(doms) == 3
        assert doms[1]["e_we"] == 805 and doms[1]["e_sn"] == 805


# ---------------------------------------------------------------------------
# Space enumeration -- hand-counted on a small synthetic grid
# ---------------------------------------------------------------------------
_SYNTH_TILE_OPTS = dict(numtiles_multiples=(1,), max_numtiles=4, include_2d=False,
                         tile_sz_variants=())


def _synth_space():
    case = dict(e_we=100, e_sn=100, e_vert=10)
    candidates, axis = ds.enumerate_space(case, total_cores=8, nodes=1, omp_options=[1, 2],
                                           tile_options=_SYNTH_TILE_OPTS)
    return candidates, axis


class TestEnumerateSpaceSynthetic:
    def test_hand_counted_small_grid(self):
        candidates, axis = _synth_space()
        # ranks=8 (omp=1): factor_pairs(8)=(1,8)(2,4)(4,2)(8,1); floor(100/n)>=10 for n in
        # {1,2,4,8} (100//8=12) -> ALL 4 legal.
        # ranks=4 (omp=2): factor_pairs(4)=(1,4)(2,2)(4,1); 100//4=25 -> ALL 3 legal.
        assert axis["shapes_count"] == {1: 4, 2: 3}
        # numtiles candidates = {1} u {omp*1} u (16 STRICT templates' numtiles values, capped at
        # max_numtiles=4) = {1} u {omp} u {1,2} -- {1,2} for both omp=1 and omp=2.
        assert axis["numtiles_options"][1] == [1, 2]
        assert axis["numtiles_options"][2] == [1, 2]
        assert axis["tile_shape_options"] == ["1d_y"]
        expected_total = (4 + 3) * 2 * 1 * 2 * 2   # shapes * numtiles * tile_shape * dist * blocktime
        assert expected_total == 56
        assert axis["total"] == len(candidates) == expected_total

    def test_no_legal_shapes_below_min_patch_gives_empty_space(self):
        tiny_case = dict(e_we=15, e_sn=15, e_vert=1)
        candidates, axis = ds.enumerate_space(tiny_case, total_cores=8, nodes=1, omp_options=[1],
                                               tile_options=_SYNTH_TILE_OPTS)
        assert axis["shapes_count"][1] == 0
        assert candidates == []

    def test_every_candidate_carries_the_required_fields(self):
        candidates, _axis = _synth_space()
        required = {"ranks", "omp", "nx", "ny", "numtiles", "tile_shape", "numtiles_x",
                    "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy", "dist", "blocktime"}
        assert required.issubset(candidates[0])


class TestCampaignTemplatePredicates:
    def test_karina_predicates_are_well_formed(self):
        case = ds.campaign_configs.CASES["karina"]
        preds = ds.campaign_template_predicates(case)
        assert 1 <= len(preds) <= 16
        for p in preds:
            assert set(p) == {"ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime"}


class TestSampleDesign:
    def test_deterministic_same_seed_same_result(self):
        space, _axis = _synth_space()
        chosen1, cov1 = ds.sample_design(space, 10, seed=42)
        chosen2, cov2 = ds.sample_design(space, 10, seed=42)
        assert [ds._candidate_key(c) for c in chosen1] == [ds._candidate_key(c) for c in chosen2]
        assert cov1 == cov2

    def test_different_seed_can_differ(self):
        space, _axis = _synth_space()
        chosen1, _c1 = ds.sample_design(space, 10, seed=1)
        chosen2, _c2 = ds.sample_design(space, 10, seed=999)
        keys1 = [ds._candidate_key(c) for c in chosen1]
        keys2 = [ds._candidate_key(c) for c in chosen2]
        assert keys1 != keys2 or len(space) < 10  # not a hard guarantee, but true for this space

    def test_respects_requested_size(self):
        space, _axis = _synth_space()
        chosen, cov = ds.sample_design(space, 10, seed=1, must_include=None)
        assert len(chosen) == 10
        assert cov["n_chosen"] == 10

    def test_must_include_always_present_even_if_alone_it_meets_the_budget(self):
        space, _axis = _synth_space()
        predicate = dict(omp=1, nx=1, ny=8, numtiles=1, dist="block", blocktime="0")
        assert any(ds._matches_predicate(c, predicate) for c in space)
        chosen, cov = ds.sample_design(space, n=1, seed=1, must_include=[predicate])
        assert any(ds._matches_predicate(c, predicate) for c in chosen)
        assert cov["must_include_matched"] == 1
        assert cov["must_include_unmatched"] == 0

    def test_must_include_unmatched_is_reported_not_silently_dropped(self):
        space, _axis = _synth_space()
        impossible = dict(omp=999)
        _chosen, cov = ds.sample_design(space, n=5, seed=1, must_include=[impossible])
        assert cov["must_include_unmatched"] == 1
        assert cov["must_include_matched"] == 0

    def test_stratified_coverage_hits_every_omp_level_when_n_allows(self):
        space, _axis = _synth_space()
        _chosen, cov = ds.sample_design(space, n=20, seed=7, must_include=None)
        assert cov["omp"]["levels_in_sample"] == cov["omp"]["levels_in_space"] == 2


class TestToCampaignConfigsRoundTrip:
    def test_ids_zero_padded_and_ordered(self):
        space, _axis = _synth_space()
        out = ds.to_campaign_configs(space[:2])
        assert [c["id"] for c in out] == ["G0001", "G0002"]

    def test_edit_namelist_round_trip_writes_exactly_one_of_each_tile_key(self):
        case = dict(e_we=100, e_sn=100, e_vert=10)
        topts = dict(numtiles_multiples=(2,), max_numtiles=64, include_2d=True, tile_sz_variants=())
        candidates, _axis = ds.enumerate_space(case, total_cores=8, nodes=1, omp_options=[2],
                                                tile_options=topts)
        twod = [c for c in candidates if c["tile_shape"] == "2d"]
        assert twod, "expected at least one 2D tile-shape candidate in this space"
        cfg = ds.to_campaign_configs(twod[:1])[0]
        assert cfg["numtiles_x"] >= 1 and cfg["numtiles_y"] >= 1

        text = "&domains\n time_step=15, e_we=100, e_sn=100, e_vert=10,\n/\n"
        out = ds.campaign_configs.edit_namelist(
            text, cfg["nx"], cfg["ny"], cfg["numtiles"], 5, adios2=False,
            numtiles_x=cfg["numtiles_x"], numtiles_y=cfg["numtiles_y"],
            tile_sz_x=cfg["tile_sz_x"], tile_sz_y=cfg["tile_sz_y"],
            tile_strategy=cfg["tile_strategy"])
        for key in ("numtiles_x=", "numtiles_y=", "tile_sz_x=", "tile_sz_y=", "tile_strategy="):
            assert out.count(key) == 1, key
        assert ("numtiles_x=%d" % cfg["numtiles_x"]) in out
        assert ("numtiles_y=%d" % cfg["numtiles_y"]) in out


# ---------------------------------------------------------------------------
# predictor/features.py additive tile-geometry features
# ---------------------------------------------------------------------------
class TestFeaturesAdditiveTileGeometry:
    BASE_ROW = dict(e_we=901.0, e_sn=901.0, e_vert=45.0, nproc_x=8, nproc_y=56, numtiles=8,
                     omp_threads=2, n_ranks=224, n_nodes=4, hardware_id="f1-spr-icpnp",
                     distribution="cyclic")

    def test_old_row_without_tile_keys_leaves_existing_fields_unchanged(self):
        cf = feat.config_features(self.BASE_ROW)
        # unchanged pre-existing fields (regression check for the additive-only requirement)
        assert cf["patch_x"] == 113
        assert cf["patch_y"] == 17
        assert cf["patch_cells"] == 113 * 17
        # f1-spr-icpnp: numa_per_node=2 (NOT the other hardware entry's 8) -> 224/4/2 = 28
        assert cf["ranks_per_numa"] == pytest.approx(28.0)

    def test_old_row_without_tile_keys_gets_none_for_fields_that_need_them(self):
        """tiles_x/y, cells_per_tile_*, tile_aspect genuinely need numtiles_x/numtiles_y --
        None for a row that lacks them. numa_aligned is deliberately NOT in this list: it only
        needs omp_threads/hardware_id, both pre-existing columns already used by
        ranks_per_numa/etc above, so it is populated for old rows too (see
        test_numa_aligned_does_not_need_new_tile_keys)."""
        cf = feat.config_features(self.BASE_ROW)
        for key in ("tiles_x", "tiles_y", "cells_per_tile_min", "cells_per_tile_max",
                    "tile_aspect"):
            assert cf[key] is None, key

    def test_numa_aligned_does_not_need_new_tile_keys(self):
        cf = feat.config_features(self.BASE_ROW)
        assert cf["numa_aligned"] == 1.0   # cores_per_numa=56, omp_threads=2, 56 % 2 == 0

    def test_row_with_tile_keys_computes_geometry(self):
        row = dict(self.BASE_ROW, numtiles_x=2, numtiles_y=4)
        cf = feat.config_features(row)
        assert cf["tiles_x"] == 2.0 and cf["tiles_y"] == 4.0
        # patch_x=113 (ceil(901/8)), patch_y=17 (ceil(901/56)); per-tile floor/ceil split:
        # 113//2=56, 17//4=4 -> min cells = 56*4*45
        assert cf["cells_per_tile_min"] == pytest.approx((113 // 2) * (17 // 4) * 45.0)
        assert cf["cells_per_tile_max"] == pytest.approx((113 // 2 + 1) * (17 // 4 + 1) * 45.0)
        assert cf["tile_aspect"] is not None
        assert cf["numa_aligned"] == 1.0   # cores_per_numa=56, omp_threads=2, 56%2==0

    def test_numa_misaligned_omp_is_flagged(self):
        feats = feat.tile_geometry_features(dict(hardware_id="f1-spr-icpnp", omp_threads=3))
        assert feats["numa_aligned"] == 0.0   # 56 % 3 != 0

    def test_new_fields_dropped_by_models_none_filter(self):
        """models.py's `_feature_dict` drops None-valued keys entirely (see models.py:79) --
        confirm a pre-existing row's tile-key-dependent new fields would vanish under that same
        filter, i.e. this addition changes nothing for any row that predates it. numa_aligned is
        excluded here since it is legitimately non-None for this row (see
        test_numa_aligned_does_not_need_new_tile_keys) -- it never needed a new key to begin
        with, so its presence is not a regression."""
        cf = feat.config_features(self.BASE_ROW)
        filtered = {k: v for k, v in cf.items() if v is not None}
        for key in ("tiles_x", "tiles_y", "cells_per_tile_min", "cells_per_tile_max",
                    "tile_aspect"):
            assert key not in filtered


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
class TestCLI:
    SYNTHETIC_CASE = dict(
        e_we=100, e_sn=100, e_vert=10, max_dom=1, steps=1, physics_sim_minutes=1,
        case_family="synthetic", prep_dir="/tmp", bp_dir="/tmp", restart=False, input_files=[],
        history_key="history_interval", dt=1, enabled=True,
        physics=dict(mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                     sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4, radt=1, cudt=0),
    )

    def test_summary_and_sample_cli(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setitem(ds.campaign_configs.CASES, "_ds_test_case", self.SYNTHETIC_CASE)

        rc = ds.main(["summary", "_ds_test_case", "--total-cores", "8", "--nodes", "1"])
        assert rc == 0
        assert "_ds_test_case" in capsys.readouterr().out

        out_file = str(tmp_path / "sample.json")
        rc = ds.main(["sample", "_ds_test_case", "--n", "5", "--seed", "1",
                      "--out", out_file, "--total-cores", "8", "--nodes", "1"])
        assert rc == 0
        capsys.readouterr()
        with open(out_file) as f:
            data = json.load(f)
        assert data["case_id"] == "_ds_test_case"
        assert len(data["configs"]) <= 5
        assert data["configs"][0]["id"].startswith("G")

    def test_unknown_case_returns_2(self, capsys):
        rc = ds.main(["summary", "no_such_case_at_all"])
        assert rc == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
