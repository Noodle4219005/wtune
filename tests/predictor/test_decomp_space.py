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


class TestTileExtents:


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


    def test_tile_sz_overrides_numtiles_entirely(self):
        # frame/module_tiles.F:353-359: 20//7=2 remainder 6 -> 3 tiles; 10//4=2 remainder 2 -> 3.
        r = ds.resolve_tile_geometry(patch_x=20, patch_y=10, numtiles=1, tile_sz_x=7, tile_sz_y=4)
        assert (r["numtiles_x"], r["numtiles_y"]) == (3, 3)
        assert r["num_tiles"] == 9


    def test_explicit_numtiles_x_only_derives_y(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=8, numtiles_x=4)
        assert (r["numtiles_x"], r["numtiles_y"]) == (4, 2)

    def test_tile_x_strategy(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=6, tile_strategy=ds.TILE_X)
        assert (r["numtiles_x"], r["numtiles_y"]) == (6, 1)

    def test_xy_strategy_uses_least_aspect(self):
        r = ds.resolve_tile_geometry(patch_x=100, patch_y=50, numtiles=12, tile_strategy=ds.TILE_XY)
        assert (r["numtiles_y"], r["numtiles_x"]) == ds.least_aspect(12, 1, 1) == (3, 4)


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


class TestCampaignTemplatePredicates:
    def test_karina_predicates_are_well_formed(self):
        case = ds.campaign_configs.CASES["karina"]
        preds = ds.campaign_template_predicates(case)
        assert 1 <= len(preds) <= 16
        for p in preds:
            assert set(p) == {"ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime"}


class TestSampleDesign:


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
