import math


from wtune.tune import shortlist


def _cand(nx, ny, omp=1, numtiles=1, dist="block", tile_strategy=0, ntx=0, nty=0):
    dom = [dict(patch_x_max=math.ceil(561 / nx), patch_y_max=math.ceil(450 / ny)),
           dict(patch_x_max=math.ceil(671 / nx), patch_y_max=math.ceil(791 / ny))]
    return dict(nx=nx, ny=ny, omp=omp, numtiles=numtiles, dist=dist, tile_strategy=tile_strategy,
                numtiles_x=ntx, numtiles_y=nty, domains=dom)


def _model(coef):
    n = len(coef)
    return dict(scaler_mean=[0.0] * n, scaler_scale=[1.0] * n, coef=coef, intercept=0.0)


def test_rank_keeps_input_order_on_ties_and_empty_is_empty():
    a, b = _cand(12, 12), _cand(12, 12, dist="cyclic")
    assert shortlist.rank(_model([0] * 10), [a, b], 86) == [a, b]
    assert shortlist.rank(_model([0] * 10), [], 86) == []
