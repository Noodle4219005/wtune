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
    keys = [ds._candidate_key(c) for c in seed]
    assert len(keys) == len(set(keys))  # no duplicates
    # up to 2 shapes x 2 tile counts per thread count
    assert len(keys) <= 4 * len(axis["omp_options"])


def test_seed_design_covers_both_the_balanced_and_the_widest_x_shape():
    """zhengzhou 112x4: the min-aspect shape (14x8) measured 3.477 s/step while the widest-in-x
    shape (4x28) measured 2.780 -- the seed must not offer only the balanced one (jobs 1080108,
    1078389)."""
    zz = dict(e_we=601, e_sn=372, e_vert=60, max_dom=1)
    space, _ = ds.enumerate_space(zz, 448, 4, ds.DEFAULT_HARDWARE)
    seed = ds.seed_design(space)
    shapes_at_omp4 = {(c["nx"], c["ny"]) for c in seed if c["omp"] == 4}
    assert len(shapes_at_omp4) == 2
    nxs = sorted(nx for nx, _ny in shapes_at_omp4)
    assert nxs[0] < nxs[1]  # one narrow-in-nproc_x (= wide patch), one balanced


def test_seed_design_is_deterministic():
    space, _ = ds.enumerate_space(CONUS12, 448, 4, ds.DEFAULT_HARDWARE)
    assert ds.seed_design(space) == ds.seed_design(list(reversed(space)))
