"""Validate the total-thickness cap: an iceberg is never thicker than its glacier.

The empirical keel laws are unbounded power laws of length, so without a cap a
1500 m berg comes out 802 m thick with keel_method='schild' -- deeper than any
Greenland terminus. ``max_thickness=`` bounds TH = keel + freeB by shortening
the keel.

Run standalone (no pytest needed):  python tests/test_thickness_cap.py
Or, if pytest is installed:          pytest tests/test_thickness_cap.py
"""

import warnings

import numpy as np

from iceberg_geometry import Iceberg
import iceberg_geometry.constants as const

RHO = const.DENSITY_RATIO_ICE_TO_WATER
C, X = const.AREA_VOLUME_COEFFICIENT, const.AREA_VOLUME_EXPONENT
LW = const.DEFAULT_LENGTH_TO_WIDTH_RATIO

# Helheim-scale terminus thickness, the motivating case.
HELHEIM_THICKNESS = 650.0


def _proxy_area(length):
    """Waterline footprint the model assumes when `area` is not measured."""
    return const.FOOTPRINT_SHAPE_FACTOR * length * (length / LW)


def _closed_form_freeboard(area):
    """Freeboard under V = c*A^x. Independent of keel depth -- that is the point."""
    return (1.0 - RHO) * C * area ** (X - 1.0)


def _quiet(**kwargs):
    """Build a berg, ignoring the keel/saturation warnings these tests provoke."""
    length = kwargs.pop("length")
    dz = kwargs.pop("dz", 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Iceberg(length=length, dz=dz).init_iceberg_size(**kwargs)


def test_no_cap_leaves_geometry_untouched():
    """max_thickness=None must reproduce the uncapped model bit for bit."""
    for length, area in ((733, 2.20e5), (518, 1.67e5), (1500, None)):
        kwargs = dict(keel_method="schild", volume_law="sulak")
        if area is not None:
            kwargs["area"] = area
        base = _quiet(length=length, **kwargs)
        same = _quiet(length=length, max_thickness=None, **kwargs)
        for var in ("keel", "TH", "totalV", "freeB", "W"):
            assert float(base[var]) == float(same[var]), f"L={length}: {var} moved"
        assert float(base.keel_capped) == 0.0
        assert np.isnan(float(base.maxTH))


def test_slack_cap_does_not_bind():
    """A cap well above the berg's own thickness changes nothing and warns nothing."""
    base = _quiet(length=733, keel_method="schild", volume_law="sulak", area=2.20e5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slack = Iceberg(length=733, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", area=2.20e5, max_thickness=5000)
    assert not [w for w in caught if "keel capped" in str(w.message)]
    assert float(slack.keel) == float(base.keel)
    assert float(slack.keel_capped) == 0.0
    assert float(slack.maxTH) == 5000.0


def test_cap_lands_thickness_exactly_on_the_glacier():
    """Freeboard is keel-independent under Sulak, so TH hits max_thickness exactly."""
    for length in (1300, 1500, 2000):   # uncapped TH 697 / 802 / 1063 m
        ds = _quiet(length=length, keel_method="schild", volume_law="sulak",
                    max_thickness=HELHEIM_THICKNESS)
        assert float(ds.keel_capped) == 1.0, f"L={length}: cap should have bound"
        assert abs(float(ds.TH) - HELHEIM_THICKNESS) < 1e-6, \
            f"L={length}: TH is {float(ds.TH):.3f} m, want {HELHEIM_THICKNESS}"
        expected_keel = HELHEIM_THICKNESS - _closed_form_freeboard(_proxy_area(length))
        assert abs(float(ds.keel) - expected_keel) < 1e-6
        # the cap costs no volume here -- the taper just relaxes toward straight walls
        ratio = float(ds.totalV) / (C * _proxy_area(length) ** X)
        assert abs(ratio - 1.0) < 1e-9, f"L={length}: totalV is {ratio:.6f} x c*A^x"


def test_cap_is_volume_free_above_the_volume_law_mean_thickness():
    """Volume survives the cap iff max_thickness >= c*A^(x-1), the law's mean thickness.

    Below that threshold even straight walls cannot hold c*A^x at the capped
    keel, so the taper saturates at a=0 and volume falls short -- by design.
    """
    area = 2.20e5
    mean_thickness = C * area ** (X - 1.0)          # 271.8 m for Iceberg A
    assert 270 < mean_thickness < 273, f"calibration moved: {mean_thickness:.1f} m"

    for max_th, volume_holds in ((300.0, True), (272.0, True),
                                 (250.0, False), (150.0, False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = Iceberg(length=733, dz=5).init_iceberg_size(
                keel_method="schild", volume_law="sulak", area=area,
                max_thickness=max_th)
        ratio = float(ds.totalV) / (C * area ** X)
        saturated = any("taper solve saturated" in str(w.message) for w in caught)
        if volume_holds:
            assert abs(ratio - 1.0) < 1e-9, f"H={max_th}: totalV is {ratio:.6f} x c*A^x"
            assert not saturated, f"H={max_th}: spurious saturation warning"
        else:
            assert ratio < 0.99, f"H={max_th}: expected an undershoot, got {ratio:.4f}"
            assert saturated, f"H={max_th}: undershoot must warn"
        # a berg holding less volume floats shallower, so TH stays under the cap
        assert float(ds.TH) <= max_th + 1e-9


def test_cap_warns_and_reports_what_it_did():
    """The cap is loud and self-documenting: a warning plus flag, maxTH and attrs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = Iceberg(length=1500, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak",
            max_thickness=HELHEIM_THICKNESS)
    assert any("keel capped" in str(w.message) for w in caught), "capping must warn"
    assert float(ds.keel_capped) == 1.0
    assert float(ds.maxTH) == HELHEIM_THICKNESS
    assert "CAPPED" in ds.attrs["thickness_cap"]
    # new variables carry metadata like every other output variable
    for var in ("maxTH", "keel_capped"):
        assert ds[var].attrs.get("long_name"), f"{var} is missing VARIABLE_ATTRS"


def test_thickness_never_exceeds_the_cap():
    """The invariant, swept over sizes, keel laws, volume laws and stability paths."""
    checked = 0
    for keel_method in ("schild", "barker", "hotzel", "mean"):
        for length in range(100, 2001, 100):
            for max_th in (300.0, 400.0, HELHEIM_THICKNESS, 900.0):
                for volume_law in ("sulak", None):
                    ds = _quiet(length=length, keel_method=keel_method,
                                volume_law=volume_law, max_thickness=max_th)
                    assert ds is not None
                    assert float(ds.TH) <= max_th + 1e-6, (
                        f"{keel_method} L={length} H={max_th} "
                        f"volume_law={volume_law}: TH={float(ds.TH):.3f}")
                    checked += 1
    assert checked == 4 * 20 * 4 * 2


def test_prism_path_solves_the_cap_by_iteration():
    """volume_law=None has keel-dependent freeboard, so TH is solved numerically."""
    for length, max_th in ((733, 300.0), (1500, 400.0), (200, 100.0)):
        ds = _quiet(length=length, max_thickness=max_th)
        assert float(ds.keel_capped) == 1.0
        assert abs(float(ds.TH) - max_th) < 1e-3, \
            f"L={length}: TH={float(ds.TH):.4f}, want {max_th}"


def test_cap_holds_through_the_equal_stability_branch():
    """Widening for stability grows the footprint; the cap must be re-derived."""
    for length, max_th in ((160, 100.0), (200, 120.0), (300, 150.0)):
        ds = _quiet(length=length, keel_method="barker", volume_law="sulak",
                    max_thickness=max_th)
        assert float(ds.TH) <= max_th + 1e-6, \
            f"L={length}: TH={float(ds.TH):.3f} breached H={max_th}"
        assert float(ds.W) / float(ds.TH) >= const.STABILITY_THRESHOLD_WH - 1e-9


def test_cap_holds_with_an_observed_wall_slope():
    """wall_slope sets the surface geometry; the cap still bounds the thickness."""
    for length, max_th, slope in ((1500, HELHEIM_THICKNESS, 10), (733, 350.0, 10)):
        ds = _quiet(length=length, keel_method="schild", volume_law="sulak",
                    wall_slope=slope, max_thickness=max_th)
        assert abs(float(ds.TH) - max_th) < 1e-6
        ratio = float(ds.totalV) / (C * _proxy_area(length) ** X)
        assert abs(ratio - 1.0) < 1e-9, f"L={length}: totalV is {ratio:.6f} x c*A^x"


def test_impossible_caps_raise():
    """Nonsense thicknesses are caller errors, not silently clamped."""
    for bad in (0, -5):
        try:
            Iceberg(length=733, dz=5).init_iceberg_size(max_thickness=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"max_thickness={bad} should raise ValueError")

    # below the freeboard a 2.2e5 m^2 footprint floats at, no keel depth works
    try:
        Iceberg(length=733, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", area=2.20e5, max_thickness=10)
    except ValueError as exc:
        assert "freeboard" in str(exc)
    else:
        raise AssertionError("a cap below the freeboard should raise ValueError")


if __name__ == "__main__":
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
