"""Tests for what is left in predictor/experiment.py (the rest: archive/unused-2026-10)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "campaign"))

import experiment as ex          # noqa: E402
import configs as campaign_configs  # noqa: E402
import decomp_space              # noqa: E402


def test_time_limit_formatting():
    # job_minutes = 0.16*60 = 9.6; ceil(1.5*9.6)=ceil(14.4)=15; +10 = 25 -> 00:25:00
    assert ex._format_hms(1.5 * 9.6 + 10) == "00:25:00"
    # exact boundary: 125 minutes -> 02:05:00
    assert ex._format_hms(125.0) == "02:05:00"
    # fractional minutes always round UP (never under-request wall-clock)
    assert ex._format_hms(59.01) == "01:00:00"


def test_c34_grid_has_no_legal_shape_at_112_ranks_but_110_is_legal():
    # confirms the diagnosed root cause directly against the real decomp_space module (imported
    # for real at the top of this file, no fake).
    case_json = campaign_configs.CASES["c34"]
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    assert decomp_space.legal_shapes_exact(we_list, sn_list, 112) == []
    assert decomp_space.legal_shapes_exact(we_list, sn_list, 110) == [(10, 11), (11, 10)]


def test_max_legal_ranks_scans_down_to_110_for_c34_at_omp1():
    case_json = campaign_configs.CASES["c34"]
    ranks, shapes = ex._max_legal_ranks(case_json, 112)
    assert ranks == 110
    assert (10, 11) in shapes
