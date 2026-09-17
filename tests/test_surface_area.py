"""Check the model's wetted (submerged) surface area against Schild et al. (2021).

Measured submerged surface area is taken from Schild et al. (2021), Table 1, as
``Meshed SA_total - SA_above`` -- available only for the two complete-
circumnavigation scans, Iceberg A survey 2 and Iceberg B survey 1:

    A2:  12.3e5 - 3.14e5 = 9.16e5 m^2
    B1:   8.52e5 - 1.91e5 = 6.61e5 m^2

The model's ``wettedA`` is a smooth stack-of-slabs area (lateral walls + basal)
multiplied by a ``roughness`` factor. The DEFAULT is smooth (1.0), which
underestimates the measured area -- roughness is opt-in, because the measured
enhancement is scale-dependent and known from only three bergs. Opting in with
``roughness_factor=SURFACE_ROUGHNESS_OBSERVED`` (1.18, the mean drone enhancement
at ~1 m scale over the three complete Schild surveys) lands within ~7% of the
measured submerged area for both bergs.

Run standalone (no pytest needed):  python tests/test_surface_area.py
"""

import warnings

from iceberg_geometry import Iceberg
from iceberg_geometry import constants as const

# name -> (surface_length_m, footprint_area_m2, measured_submerged_SA_m2)
MEASURED_SA = {
    "A_survey2": (729, 2.17e5, 9.16e5),
    "B_survey1": (518, 1.67e5, 6.61e5),
}


def _base(length):
    return Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law=None
    )


def _calibrated(length, area, roughness_factor=None):
    return Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak", area=area,
        roughness_factor=roughness_factor,
    )


def test_wettedA_and_roughness_present():
    """Datasets carry a positive wetted area and the roughness factor (default smooth)."""
    for name, (length, area, _) in MEASURED_SA.items():
        ds = _calibrated(length, area)
        assert "wettedA" in ds.variables, f"{name}: wettedA missing"
        assert "roughness" in ds.variables, f"{name}: roughness missing"
        assert float(ds.wettedA) > 0.0, f"{name}: wettedA not positive"
        assert abs(float(ds.roughness) - const.SURFACE_ROUGHNESS_FACTOR) < 1e-9, (
            f"{name}: default roughness != {const.SURFACE_ROUGHNESS_FACTOR}")
        assert ds.wettedA.attrs.get("units") == "m2", f"{name}: wettedA units wrong"


def test_basalA_present_and_within_smooth_total():
    """basalA is the keel bottom footprint: positive, roughness-free, and a
    fraction of the smooth total area (wettedA / roughness = lateral + basalA)."""
    for name, (length, area, _) in MEASURED_SA.items():
        ds = _calibrated(length, area)
        assert "basalA" in ds.variables, f"{name}: basalA missing"
        smooth_total = float(ds.wettedA) / float(ds.roughness)
        assert 0.0 < float(ds.basalA) < smooth_total, (
            f"{name}: basalA {float(ds.basalA):.2e} not within smooth total {smooth_total:.2e}"
        )


def test_footprint_factor_uses_measured_area():
    """footprint_factor = measured area/(L*W) when area is given, else the default."""
    length, area, _ = MEASURED_SA["A_survey2"]
    with_area = _calibrated(length, area)
    expected = area / (length * (length / 1.62))
    assert abs(float(with_area.footprint_factor) - expected) < 1e-9
    # without measured area -> default FOOTPRINT_SHAPE_FACTOR
    no_area = Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak")
    assert abs(float(no_area.footprint_factor)
               - const.FOOTPRINT_SHAPE_FACTOR) < 1e-9


def test_footprint_factor_kwarg_overrides_default():
    """An explicit footprint_factor replaces FOOTPRINT_SHAPE_FACTOR in the
    length-to-area proxy, and is reported back on the dataset."""
    length, _, _ = MEASURED_SA["A_survey2"]
    for ff in (0.58, 0.70):
        ds = Iceberg(length=length, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", footprint_factor=ff)
        assert abs(float(ds.footprint_factor) - ff) < 1e-9, f"ff={ff} not reported"
    # V = c*A^x with A proportional to the factor, so volume scales as ff^x
    lo = Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak", footprint_factor=0.58)
    hi = Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak", footprint_factor=0.70)
    expected_ratio = (0.70 / 0.58) ** const.AREA_VOLUME_EXPONENT
    actual_ratio = float(hi.totalV) / float(lo.totalV)
    assert abs(actual_ratio / expected_ratio - 1.0) < 0.02, (
        f"volume ratio {actual_ratio:.3f} != expected {expected_ratio:.3f}")


def test_footprint_factor_guards():
    """footprint_factor must be a fraction, and cannot be combined with area."""
    length, area, _ = MEASURED_SA["A_survey2"]

    def _call(**kw):
        return Iceberg(length=length, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", **kw)

    for bad in (0.0, -0.1, 1.5):
        try:
            _call(footprint_factor=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"footprint_factor={bad} should raise ValueError")
    try:
        _call(area=area, footprint_factor=0.6)
    except ValueError:
        pass
    else:
        raise AssertionError("area + footprint_factor together should raise ValueError")


def test_perimeter_scales_lateral_area_only():
    """A measured perimeter shortens the lateral walls without touching the
    area-driven quantities (volume, freeboard) or the basal footprint."""
    length, area, _ = MEASURED_SA["A_survey2"]
    base = _calibrated(length, area)
    width = length / 1.62
    factor = 0.829  # drone-measured hull perimeter / bounding-rectangle perimeter
    ds = Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak", area=area,
        perimeter=factor * 2.0 * (length + width))

    assert abs(float(base.perimeter_factor) - 1.0) < 1e-9, "default should be the rectangle"
    assert abs(float(ds.perimeter_factor) - factor) < 1e-9
    assert abs(float(ds.totalV) - float(base.totalV)) < 1.0, "volume must not change"
    assert abs(float(ds.freeB) - float(base.freeB)) < 1e-9, "freeboard must not change"
    assert abs(float(ds.basalA) - float(base.basalA)) < 1e-6, "basal area must not change"

    # only the lateral term is scaled, so wettedA falls by less than `factor`
    smooth_base = float(base.wettedA) / float(base.roughness)
    lateral = smooth_base - float(base.basalA)
    expected = (lateral * factor + float(base.basalA)) * float(base.roughness)
    assert abs(float(ds.wettedA) / expected - 1.0) < 1e-9, (
        f"wettedA {float(ds.wettedA):.4e} != expected {expected:.4e}")


def test_ragged_perimeter_warns():
    """A perimeter longer than its own bounding rectangle means an unsimplified
    outline whose raggedness roughness already covers -- warn, don't silently use it."""
    length, area, _ = MEASURED_SA["A_survey2"]
    rect = 2.0 * (length + length / 1.62)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = Iceberg(length=length, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", area=area, perimeter=1.2 * rect)
    assert any("bounding-rectangle" in str(w.message) for w in caught), (
        "expected a warning for perimeter_factor > 1")
    assert abs(float(ds.perimeter_factor) - 1.2) < 1e-9, "factor still applied"

    for bad in (0.0, -10.0):
        try:
            Iceberg(length=length, dz=5).init_iceberg_size(
                keel_method="schild", volume_law="sulak", area=area, perimeter=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"perimeter={bad} should raise ValueError")


def test_roughness_factor_scales_area_linearly():
    """wettedA scales linearly with the roughness factor; 1.0 gives the smooth area."""
    length, area, _ = MEASURED_SA["A_survey2"]
    smooth = _calibrated(length, area, roughness_factor=1.0)
    rough = _calibrated(length, area, roughness_factor=1.3)
    assert abs(float(smooth.roughness) - 1.0) < 1e-9
    assert abs(float(rough.wettedA) / float(smooth.wettedA) - 1.3) < 1e-9


def test_calibrated_surface_area_matches_measured():
    """Opting in to the measured roughness (1.18), calibrated wetted area is within
    15% of the measured submerged surface area (actual: A2 1.05x, B1 0.93x).

    Passed explicitly, not taken from the default: the default is smooth, so this
    is a statement about what the geometry CAN reproduce, not about sweep output.
    """
    for name, (length, area, sa_meas) in MEASURED_SA.items():
        ds = _calibrated(length, area,
                         roughness_factor=const.SURFACE_ROUGHNESS_OBSERVED)
        ratio = float(ds.wettedA) / sa_meas
        assert 0.85 < ratio < 1.15, f"{name}: wettedA/measured {ratio:.2f} outside 15%"


def test_smooth_area_underestimates_measured():
    """Without roughness (factor=1.0) the smooth calibrated area falls below
    measured -- reproducing the idealized-geometry underestimate Schild warn of."""
    for name, (length, area, sa_meas) in MEASURED_SA.items():
        ratio = float(_calibrated(length, area, roughness_factor=1.0).wettedA) / sa_meas
        assert ratio < 1.0, f"{name}: smooth wettedA/measured {ratio:.2f} should be < 1"


def test_calibration_reduces_surface_area():
    """The volume calibration shrinks the geometry, so wetted area drops vs the
    base prism (compared at the same roughness) -- the core volume-vs-area tension."""
    for name, (length, area, _) in MEASURED_SA.items():
        base = float(_base(length).wettedA)
        cal = float(_calibrated(length, area).wettedA)
        assert cal < base, f"{name}: calibrated wettedA {cal:.2e} !< base {base:.2e}"


if __name__ == "__main__":
    tests = [obj for nm, obj in sorted(globals().items())
             if nm.startswith("test_") and callable(obj)]
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
