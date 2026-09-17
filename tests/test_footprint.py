"""Check the polygon footprint helpers against analytic shapes.

``footprint_metrics`` measures length, area and perimeter on ONE convention:
length is the convex-hull max caliper (Schild's "surface length" -- hull
calipers of the four 2017 drone clouds give 738/740/528/528 m against the
published 733/729/518/515 m), area is the raw polygon area, and perimeter is
measured after simplification at a fixed tolerance. The shape factors are
therefore defined against the *caliper* bounding rectangle, so a circle gives
pi/4 = 0.785 for both, not 1.0.

The key asymmetry under test: smoothing leaves area alone but shortens
perimeter. That is why the tolerance is part of the measurement rather than a
detail -- a raw segmented perimeter double-counts the roughness factor.

Run standalone (no pytest needed):  python tests/test_footprint.py
"""

import math

from iceberg_geometry import footprint_metrics, footprint_table, max_caliper
from iceberg_geometry.footprint import DEFAULT_SIMPLIFY_TOLERANCE

try:
    from shapely.geometry import Point, Polygon, box
except ImportError:  # shapely is not a hard dependency of this package
    Point = None

R = 300.0  # m, big enough that the 10 m default tolerance is not destructive


def test_circle_factors_are_pi_over_four():
    """For a disc, both shape factors are the circle-in-square ratio pi/4."""
    circle = Point(0, 0).buffer(R, quad_segs=256)
    m = footprint_metrics(circle, simplify_tolerance=0)
    assert abs(m["length"] - 2 * R) / (2 * R) < 1e-3, m["length"]
    assert abs(m["width"] - 2 * R) / (2 * R) < 1e-2, m["width"]
    for key in ("footprint_factor", "perimeter_factor"):
        assert abs(m[key] - math.pi / 4) < 0.01, f"{key} = {m[key]:.4f}"
    assert abs(m["isoperimetric_quotient"] - 1.0) < 0.01, m["isoperimetric_quotient"]


def test_max_caliper_of_rectangle_is_the_diagonal():
    """Surface length is the longest extent, which for a box is its diagonal."""
    rect = box(0, 0, 1.62 * R, R)
    expected = math.hypot(1.62 * R, R)
    assert abs(max_caliper(rect) - expected) < 1e-6
    assert abs(footprint_metrics(rect, simplify_tolerance=0)["width"] - R) < 1e-6


def test_simplify_shortens_perimeter_but_not_area():
    """The coastline asymmetry: a ragged outline loses perimeter, keeps area."""
    # Sawtooth ring: a circle with a fine radial wobble, like a segmented outline.
    n, wobble = 720, 4.0
    pts = [((R + (wobble if i % 2 else -wobble)) * math.cos(2 * math.pi * i / n),
            (R + (wobble if i % 2 else -wobble)) * math.sin(2 * math.pi * i / n))
           for i in range(n)]
    ragged = Polygon(pts)

    raw = footprint_metrics(ragged, simplify_tolerance=0)
    smooth = footprint_metrics(ragged, simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE)

    area_change = abs(smooth["area"] / raw["area"] - 1.0)
    perim_change = 1.0 - smooth["perimeter"] / raw["perimeter"]
    assert area_change < 0.02, f"area moved {area_change:.1%} under smoothing"
    assert perim_change > 0.10, f"perimeter only shortened {perim_change:.1%}"
    # the raw outline reports a rougher (lower) isoperimetric quotient
    assert smooth["isoperimetric_quotient"] > raw["isoperimetric_quotient"]
    assert raw["perimeter"] == raw["perimeter_raw"]


def test_footprint_table_is_column_oriented_and_skips_bad_geometries():
    good = Point(0, 0).buffer(R, quad_segs=64)
    table = footprint_table([good, good], simplify_tolerance=0)
    assert set(table) == set(footprint_metrics(good, simplify_tolerance=0))
    assert all(len(v) == 2 for v in table.values())

    empty = Polygon()
    try:
        footprint_table([good, empty], simplify_tolerance=0)
    except (ValueError, AttributeError):
        pass
    else:
        raise AssertionError("degenerate geometry should raise without skip_errors")
    skipped = footprint_table([good, empty], simplify_tolerance=0, skip_errors=True)
    assert skipped["area"][0] > 0 and math.isnan(skipped["area"][1])


def test_negative_tolerance_rejected():
    circle = Point(0, 0).buffer(R, quad_segs=64)
    try:
        footprint_metrics(circle, simplify_tolerance=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative simplify_tolerance should raise ValueError")


if __name__ == "__main__":
    if Point is None:
        print("SKIP  shapely not installed (optional for this helper)")
        raise SystemExit(0)
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
