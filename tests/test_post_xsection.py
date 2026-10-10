"""Tests for post/recipes/xsection.py's geodesic line helpers, especially
clip_halfwidth_to_domain() (2026-09-29 third/fourth lead review): a cross-section line built
by extending +/-halfwidth_km from a centre point must never draw a point outside the region
it is supposed to represent -- bbox_predicate() for a simple lat/lon rectangle, or (fourth
review) a grid-index predicate for the true (possibly non-rectangular) domain interior. Real
network-free geometry (cartopy.geodesic), no netCDF/wrf-python needed.
"""
import cartopy.geodesic as cgeo

from wtune.post.recipes.xsection import (bbox_predicate, clip_halfwidth_to_domain, line_from_point_bearing, line_unit_vector)


def test_clip_returns_full_halfwidth_when_it_already_fits():
    # a generous bbox around a centre point well away from any edge
    inside = bbox_predicate((-10.0, 10.0, -10.0, 10.0))
    clipped = clip_halfwidth_to_domain(0.0, 0.0, 90.0, 50.0, inside)
    assert clipped == 50.0


def test_clip_shortens_a_line_near_the_domain_edge():
    """The centre point sits close to the domain's eastern edge; a 150 km half-width to the
    east would run outside the domain. The clipped line's endpoints must both stay inside,
    and the centre must remain the exact midpoint (a symmetric half-width by construction)."""
    # ~0.3 degrees longitude at the equator is ~33 km -- a domain edge close enough that a
    # 150 km half-width overshoots it well before the west side is any constraint.
    bbox = (-5.0, 5.0, -5.0, 0.3)
    inside = bbox_predicate(bbox)
    centre_lat, centre_lon = 0.0, 0.2
    requested_halfwidth_km = 150.0

    clipped = clip_halfwidth_to_domain(centre_lat, centre_lon, 90.0, requested_halfwidth_km, inside)
    assert 0.0 < clipped < requested_halfwidth_km

    start_ll, end_ll = line_from_point_bearing(centre_lat, centre_lon, 90.0, clipped)
    assert inside(*start_ll)
    assert inside(*end_ll)

    # one more increment outward must fail -- clipped is the largest value that still fits
    # (bisection tolerance: a small positive margin).
    start_over, end_over = line_from_point_bearing(centre_lat, centre_lon, 90.0, clipped + 1.0)
    assert not (inside(*start_over) and inside(*end_over))


def test_clip_keeps_centre_as_exact_midpoint():
    bbox = (-5.0, 5.0, -5.0, 0.3)
    inside = bbox_predicate(bbox)
    centre_lat, centre_lon = 0.0, 0.2
    clipped = clip_halfwidth_to_domain(centre_lat, centre_lon, 90.0, 150.0, inside)
    start_ll, end_ll = line_from_point_bearing(centre_lat, centre_lon, 90.0, clipped)

    geod = cgeo.Geodesic()
    dist_to_start = geod.inverse([[centre_lon, centre_lat]], [[start_ll[1], start_ll[0]]])[0, 0]
    dist_to_end = geod.inverse([[centre_lon, centre_lat]], [[end_ll[1], end_ll[0]]])[0, 0]
    assert abs(dist_to_start - dist_to_end) < 1.0   # metres -- symmetric to within rounding


def test_clip_returns_zero_when_centre_itself_is_outside_bbox():
    inside = bbox_predicate((-1.0, 1.0, -1.0, 1.0))
    assert clip_halfwidth_to_domain(50.0, 50.0, 90.0, 150.0, inside) == 0.0


def test_clip_with_a_non_rectangular_index_predicate():
    """The fourth lead review's actual point: a lat/lon bbox is not the same shape as a
    Lambert-projected grid's interior. Simulate a domain whose "interior" (in an abstract
    index space standing in for wrf.ll_to_xy) is a DIAMOND, not a rectangle centred on the
    origin -- a bbox-based predicate would accept points a diamond-shaped predicate rejects,
    so this checks clip_halfwidth_to_domain() genuinely uses whatever inside_fn says, not an
    implicit rectangle assumption."""
    def diamond_inside(lat, lon):
        # |lat| + |lon| <= 1 degree, in this synthetic index-stand-in space
        return abs(lat) + abs(lon) <= 1.0

    # due east from the origin: the diamond only extends 1 degree (~111 km) that direction,
    # far short of the rectangle-equivalent that a naive bbox=(-5,5,-5,5) would have allowed.
    clipped = clip_halfwidth_to_domain(0.0, 0.0, 90.0, 500.0, diamond_inside)
    start_ll, end_ll = line_from_point_bearing(0.0, 0.0, 90.0, clipped)
    assert diamond_inside(*start_ll) and diamond_inside(*end_ll)
    assert clipped < 200.0   # nowhere near the full 500 km requested


def test_line_unit_vector_points_east_for_a_due_east_line():
    di, dj = line_unit_vector((10.0, 0.0), (10.0, 1.0))
    assert di > 0.99 and abs(dj) < 0.05
