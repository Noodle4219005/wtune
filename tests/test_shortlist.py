import json
import math
import os

import pytest

from wtune.tune import shortlist


def _cand(nx, ny, omp=1, numtiles=1, dist="block", tile_strategy=0, ntx=0, nty=0):
    dom = [dict(patch_x_max=math.ceil(561 / nx), patch_y_max=math.ceil(450 / ny)),
           dict(patch_x_max=math.ceil(671 / nx), patch_y_max=math.ceil(791 / ny))]
    return dict(nx=nx, ny=ny, omp=omp, numtiles=numtiles, dist=dist, tile_strategy=tile_strategy,
                numtiles_x=ntx, numtiles_y=nty, domains=dom)


def _model(coef):
    n = len(coef)
    return dict(scaler_mean=[0.0] * n, scaler_scale=[1.0] * n, coef=coef, intercept=0.0)


def test_rank_orders_by_the_score_and_skips_layouts_the_model_was_not_fitted_on():
    # only the log_aspect feature (index 2) counts: the squarer patch (px/py nearest 1) -> a
    # negative coefficient on |aspect| is not linear, so use the sign: smaller log(px/py) first
    coef = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]
    wide, tall = _cand(8, 40), _cand(40, 8)
    hybrid, odd_tile = _cand(8, 18, omp=2), _cand(16, 20, tile_strategy=3)
    got = shortlist.rank(_model(coef), [wide, tall, hybrid, odd_tile], 86)
    assert got == [tall, wide]                    # hybrid and non-default tile are not ranked


def test_rank_keeps_input_order_on_ties_and_empty_is_empty():
    a, b = _cand(12, 12), _cand(12, 12, dist="cyclic")
    assert shortlist.rank(_model([0] * 10), [a, b], 86) == [a, b]
    assert shortlist.rank(_model([0] * 10), [], 86) == []


def test_load_needs_a_model_file_and_honours_the_switch(monkeypatch):
    shortlist._cache.clear()
    assert shortlist.load("no-such-machine") is None
    m = shortlist.load("f1-grace")
    assert m and len(m["coef"]) == len(m["scaler_mean"]) == 10
    monkeypatch.setenv("WTUNE_SHORTLIST", "0")
    assert shortlist.load("f1-grace") is None


def test_the_shipped_f1_grace_model_ranks_real_candidates():
    shortlist._cache.clear()
    m = shortlist.load("f1-grace")
    cands = [_cand(nx, ny) for nx, ny in ((12, 12), (6, 24), (24, 6), (16, 9), (9, 16))]
    got = shortlist.rank(m, cands, 86)
    assert len(got) == 5 and set(map(id, got)) == set(map(id, cands))
