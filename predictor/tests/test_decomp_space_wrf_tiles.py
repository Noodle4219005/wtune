#!/usr/bin/env python3
"""predictor/tests/test_decomp_space_wrf_tiles.py -- cross-check decomp_space.py's tile geometry
against REAL WRF-printed rank-0 tile extents.

Fixture: predictor/tests/fixtures/tile_smoke_1078036_wrf_tiles.txt, copied verbatim from
/home/u5727520/wrf-bench/wscore/decomp/tile_smoke_1078036_wrf_tiles.txt (smoke job 1078036: karina
901x901x45, cf448, 7 generated configs from
/home/u5727520/wrf-bench/wscore/decomp/tile_smoke_karina.json -- G0001, G0002, G0019, G0033,
G0046, G0051, T0001). Each config's rank-0 process printed "WRF TILE n IS .. IE .. JS .. JE .."
and "WRF NUMBER OF TILES = n" lines; this is the ground truth this module must reproduce.

Bug found and fixed 2026-09-15 (WTune D3/D4 work): `enumerate_space`'s own patch_x/patch_y fields
are deliberately the SMALLEST-patch envelope bound across every rank (see that function's
docstring -- kept for 1D->2D-switch risk assessment), NOT rank 0's actual patch. Hand-checked
against this fixture: karina's 8x56 shape needs patch_y=17 for rank 0 (the LARGEST y-patch, not
the smallest 16) to reproduce G0002's 8-way tile split; its 4x56 shape needs patch_x=225 (the
SMALLEST x-patch, matching by coincidence) but patch_y=17 again (the largest). Neither a fixed
"always min" nor "always max" choice works -- which extreme rank 0 lands on is independent per
dimension (depends on that dimension's own RSL-lite front/back remainder split). Fixed by adding
`rsl_patch_width_for_rank` (an exact per-rank width function) and new `patch_x_rank0`/
`patch_y_rank0` fields (domain_patch_geometry) plus rank0-specific derived tile-geometry fields
(numtiles_x_rank0/went_2d_rank0/etc., enumerate_space) -- purely additive; the pre-existing
min/max-envelope fields and their semantics are unchanged.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decomp_space as ds   # noqa: E402

# karina: e_we=e_sn=901 (campaign/cases.json), the case this smoke job actually ran.
E_WE = E_SN = 901

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tile_smoke_1078036_wrf_tiles.txt")


def _load_fixture():
    """{config_id: {"namelist": {key: int}, "tiles": [(is,ie,js,je), ...], "n_tiles": int}}."""
    configs = {}
    with open(_FIXTURE) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if " namelist: " in line:
                cid, rest = line.split(" namelist: ", 1)
                kv = {}
                for tok in rest.split(","):
                    tok = tok.strip()
                    if not tok or "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    kv[k.strip()] = int(v.strip())
                configs[cid] = dict(namelist=kv, tiles=[], n_tiles=None)
            elif " rank0: WRF TILE " in line:
                cid, rest = line.split(" rank0: ", 1)
                # "WRF TILE   1 IS      1 IE    113 JS      1 JE     17"
                nums = [int(t) for t in re.findall(r"-?\d+", rest)]
                _tile_idx, is_, ie, js, je = nums
                configs[cid]["tiles"].append((is_, ie, js, je))
            elif " rank0: WRF NUMBER OF TILES" in line:
                cid, rest = line.split(" rank0: ", 1)
                configs[cid]["n_tiles"] = int(re.findall(r"-?\d+", rest)[0])
    return configs


class TestWrfGroundTruthTileGeometry:
    def test_fixture_has_all_seven_configs(self):
        configs = _load_fixture()
        assert set(configs) == {"G0001", "G0002", "G0019", "G0033", "G0046", "G0051", "T0001"}
        for cid, data in configs.items():
            assert data["n_tiles"] == len(data["tiles"]), cid

    def test_rank0_patch_dims_hand_checked(self):
        """Hand-computed in the investigation (see module docstring): karina's 8x56 shape rank 0
        gets the LARGEST y-patch (17, not the envelope min 16); its 4x56 shape rank 0 gets the
        SMALLEST x-patch (225) but still the LARGEST y-patch (17) -- confirming the two dimensions
        resolve independently and no fixed min/max choice stands in for rank 0 in general."""
        assert ds.rsl_patch_width_for_rank(E_WE, 8, 0) == 113
        assert ds.rsl_patch_width_for_rank(E_SN, 56, 0) == 17
        assert ds.rsl_patch_extents(E_SN, 56) == (16, 17)  # envelope min != rank0's 17
        assert ds.rsl_patch_width_for_rank(E_WE, 4, 0) == 225
        assert ds.rsl_patch_extents(E_WE, 4) == (225, 226)  # here rank0's 225 == envelope min
        assert ds.rsl_patch_width_for_rank(E_WE, 64, 0) == 15
        assert ds.rsl_patch_width_for_rank(E_SN, 1, 0) == 901
        assert ds.rsl_patch_width_for_rank(E_SN, 8, 0) == 113

    def test_all_seven_configs_reproduce_wrf_rank0_tiles(self):
        configs = _load_fixture()
        for cid, data in configs.items():
            nl = data["namelist"]
            patch_x = ds.rsl_patch_width_for_rank(E_WE, nl["nproc_x"], 0)
            patch_y = ds.rsl_patch_width_for_rank(E_SN, nl["nproc_y"], 0)
            resolved = ds.resolve_tile_geometry(
                patch_x, patch_y, numtiles=nl["numtiles"],
                numtiles_x=nl["numtiles_x"], numtiles_y=nl["numtiles_y"],
                tile_sz_x=nl["tile_sz_x"], tile_sz_y=nl["tile_sz_y"],
                tile_strategy=nl["tile_strategy"])
            extents = ds.tile_extents(patch_x, patch_y,
                                       resolved["numtiles_x"], resolved["numtiles_y"])
            assert len(extents) == data["n_tiles"], (
                "%s: WRF printed %d tiles, decomp_space computed %d (patch=%dx%d resolved=%s)"
                % (cid, data["n_tiles"], len(extents), patch_x, patch_y, resolved))
            assert extents == data["tiles"], (
                "%s: tile extents mismatch vs WRF rank0 printout (patch=%dx%d resolved=%s)\n"
                "  decomp_space: %s\n  WRF:          %s"
                % (cid, patch_x, patch_y, resolved, extents, data["tiles"]))

    def test_t0001_tile_sz_is_not_literal_fixed_width(self):
        """T0001 (tile_sz_x=28, tile_sz_y=4, rank0 patch 113x17) produces 25 tiles (5x5) with
        widths 22-23 and heights 3-4 -- i.e. tile_sz only sets the TILE COUNT via
        ceil(patch/tile_sz) (module_tiles.F), then the patch is split EVENLY into that many groups
        by the same recursive rlocproc/region_bounds algorithm as any other tiling, not literally
        chopped into fixed tile_sz-sized blocks (which would give widths of 28,28,28,28,1)."""
        configs = _load_fixture()
        t0001 = configs["T0001"]
        widths = sorted({ie - is_ + 1 for is_, ie, _js, _je in t0001["tiles"]})
        heights = sorted({je - js + 1 for _is, _ie, js, je in t0001["tiles"]})
        assert t0001["n_tiles"] == 25
        assert widths == [22, 23]
        assert heights == [3, 4]
        assert 28 not in widths and 4 in heights  # literal tile_sz_x=28 never realized as a width

    def test_enumerate_space_carries_rank0_fields_additively(self):
        """New rank0-specific fields are additive: every pre-existing candidate field (patch_x,
        patch_x_max, numtiles_x_used, went_2d, cells_per_tile_min/max, ...) is unchanged."""
        case = dict(e_we=E_WE, e_sn=E_SN, e_vert=45, max_dom=1)
        candidates, axis = ds.enumerate_space(
            case, total_cores=448, nodes=4, hardware_id="f1-spr-icpnp",
            omp_options=[1], tile_options=dict(numtiles_multiples=(1, 8), max_numtiles=16))
        assert candidates, "expected a non-empty candidate space"
        c = next(c for c in candidates if c["nx"] == 8 and c["ny"] == 56 and c["numtiles"] == 1)
        for key in ("patch_x", "patch_x_max", "numtiles_x_used", "went_2d",
                    "cells_per_tile_min", "cells_per_tile_max"):
            assert key in c
        assert c["patch_x_rank0"] == 113
        assert c["patch_y_rank0"] == 17
        assert c["realized_num_tiles_rank0"] == 1
