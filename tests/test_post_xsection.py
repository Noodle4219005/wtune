"""Tests for post/recipes/xsection.py's geodesic line helpers, especially
clip_halfwidth_to_domain() (2026-09-29 third/fourth lead review): a cross-section line built
by extending +/-halfwidth_km from a centre point must never draw a point outside the region
it is supposed to represent -- bbox_predicate() for a simple lat/lon rectangle, or (fourth
review) a grid-index predicate for the true (possibly non-rectangular) domain interior. Real
network-free geometry (cartopy.geodesic), no netCDF/wrf-python needed.
"""

from wtune.post.recipes.xsection import (bbox_predicate, clip_halfwidth_to_domain, line_unit_vector)


def test_clip_returns_full_halfwidth_when_it_already_fits():
    # a generous bbox around a centre point well away from any edge
    inside = bbox_predicate((-10.0, 10.0, -10.0, 10.0))
    clipped = clip_halfwidth_to_domain(0.0, 0.0, 90.0, 50.0, inside)
    assert clipped == 50.0


def test_clip_returns_zero_when_centre_itself_is_outside_bbox():
    inside = bbox_predicate((-1.0, 1.0, -1.0, 1.0))
    assert clip_halfwidth_to_domain(50.0, 50.0, 90.0, 150.0, inside) == 0.0


def test_line_unit_vector_points_east_for_a_due_east_line():
    di, dj = line_unit_vector((10.0, 0.0), (10.0, 1.0))
    assert di > 0.99 and abs(dj) < 0.05
