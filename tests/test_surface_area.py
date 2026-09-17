"""Check the model's wetted (submerged) surface area against Schild et al. (2021).

Measured submerged surface area is taken from Schild et al. (2021), Table 1, as
``Meshed SA_total - SA_above`` -- available only for the two complete-
circumnavigation scans, Iceberg A survey 2 and Iceberg B survey 1:

    A2:  12.3e5 - 3.14e5 = 9.16e5 m^2
    B1:   8.52e5 - 1.91e5 = 6.61e5 m^2

The model's ``wettedA`` is a smooth stack-of-slabs area (lateral walls + basal)
multiplied by a ``roughness`` factor (default SURFACE_ROUGHNESS_FACTOR = 1.18,
the mean drone enhancement at ~1 m scale over the three complete Schild surveys).
With the calibrated geometry this lands within ~7% of the measured submerged area
for both bergs; the smooth area alone (roughness_factor=1.0) underestimates it.

Run standalone (no pytest needed):  python tests/test_surface_area.py
"""

from iceberg_geometry import Iceberg

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
    """Datasets carry a positive wetted area and the roughness factor (default 1.19)."""
    for name, (length, area, _) in MEASURED_SA.items():
        ds = _calibrated(length, area)
        assert "wettedA" in ds.variables, f"{name}: wettedA missing"
        assert "roughness" in ds.variables, f"{name}: roughness missing"
        assert float(ds.wettedA) > 0.0, f"{name}: wettedA not positive"
        assert abs(float(ds.roughness) - 1.18) < 1e-9, f"{name}: default roughness != 1.18"
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
    """footprint_factor = measured area/(L*W) when area is given, else 0.68."""
    length, area, _ = MEASURED_SA["A_survey2"]
    with_area = _calibrated(length, area)
    expected = area / (length * (length / 1.62))
    assert abs(float(with_area.footprint_factor) - expected) < 1e-9
    # without measured area -> default FOOTPRINT_SHAPE_FACTOR
    no_area = Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak")
    assert abs(float(no_area.footprint_factor) - 0.68) < 1e-9


def test_roughness_factor_scales_area_linearly():
    """wettedA scales linearly with the roughness factor; 1.0 gives the smooth area."""
    length, area, _ = MEASURED_SA["A_survey2"]
    smooth = _calibrated(length, area, roughness_factor=1.0)
    rough = _calibrated(length, area, roughness_factor=1.3)
    assert abs(float(smooth.roughness) - 1.0) < 1e-9
    assert abs(float(rough.wettedA) / float(smooth.wettedA) - 1.3) < 1e-9


def test_calibrated_surface_area_matches_measured():
    """With the default roughness factor, calibrated wetted area is within 15% of
    the measured submerged surface area (actual: A2 1.05x, B1 0.93x)."""
    for name, (length, area, sa_meas) in MEASURED_SA.items():
        ratio = float(_calibrated(length, area).wettedA) / sa_meas
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
