import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decomp_space as ds  # noqa: E402

CONUS12 = dict(e_we=425, e_sn=300, e_vert=50, max_dom=1)


def test_seed_design_one_balanced_shape_per_thread_count_on_one_node():
    space, axis = ds.enumerate_space(CONUS12, 112, 1, ds.DEFAULT_HARDWARE)
    seed = ds.seed_design(space)
    assert seed, "seed design must not be empty"
    assert {c["omp"] for c in seed} == {o for o in axis["omp_options"] if o <= 14}
    for c in seed:
        assert c["ranks"] * c["omp"] == 112
        assert c["tile_shape"] == "1d_y" and c["dist"] == "block" and str(c["blocktime"]) == "0"
        group = [s for s in space if s["omp"] == c["omp"] and s["patch_x"] and s["patch_y"]]
        best = min(max(s["patch_x"], s["patch_y"]) / min(s["patch_x"], s["patch_y"]) for s in group)
        assert max(c["patch_x"], c["patch_y"]) / min(c["patch_x"], c["patch_y"]) == best
    # at most two tile counts per thread count, no duplicates
    keys = [ds._candidate_key(c) for c in seed]
    assert len(keys) == len(set(keys)) <= 2 * len(axis["omp_options"])


def test_seed_design_is_deterministic():
    space, _ = ds.enumerate_space(CONUS12, 448, 4, ds.DEFAULT_HARDWARE)
    assert ds.seed_design(space) == ds.seed_design(list(reversed(space)))
