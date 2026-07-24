"""Check the model's wetted (submerged) surface area against Schild et al. (2021).

Measured submerged surface area is taken from Schild et al. (2021), Table 1, as
``Meshed SA_total - SA_above`` -- available only for the two complete-
circumnavigation scans, Iceberg A survey 2 and Iceberg B survey 1:

    A2:  12.3e5 - 3.14e5 = 9.16e5 m^2
    B1:   8.52e5 - 1.91e5 = 6.61e5 m^2

These tests document a deliberate tension: the volume calibration
(``volume_law='sulak'``) was tuned to reproduce the measured *volume*, but it
shrinks the geometry, which *reduces* the wetted surface area and drives it
below the measurement. Volume and surface area cannot both be matched by a
single smooth shrink (volume ~ size^3, area ~ size^2). See ``wettedA`` in
init_iceberg_size, which is itself a smooth lower bound (no roughness).

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


def _calibrated(length, area):
    return Iceberg(length=length, dz=5).init_iceberg_size(
        keel_method="schild", volume_law="sulak", area=area
    )


def test_wettedA_present_and_positive():
    """Every geometry dataset carries a positive wetted-area variable with metadata."""
    for name, (length, area, _sa) in MEASURED_SA.items():
        ds = _calibrated(length, area)
        assert "wettedA" in ds.variables, f"{name}: wettedA missing"
        assert float(ds.wettedA) > 0.0, f"{name}: wettedA not positive"
        assert ds.wettedA.attrs.get("units") == "m2", f"{name}: wettedA units wrong"


def test_calibration_reduces_surface_area():
    """The volume calibration shrinks the geometry, so wetted area drops vs the
    base prism -- the core volume-vs-area tension."""
    for name, (length, area, _sa) in MEASURED_SA.items():
        base = float(_base(length).wettedA)
        cal = float(_calibrated(length, area).wettedA)
        assert cal < base, f"{name}: calibrated wettedA {cal:.2e} !< base {base:.2e}"


def test_calibrated_surface_area_underestimates_measured():
    """Calibrated wetted area lands below the measured submerged surface area
    (~0.79-0.89x) -- reproducing the idealized-geometry underestimate Schild
    et al. warn about. Encoded as a documented band, not an accuracy target."""
    for name, (length, area, sa_meas) in MEASURED_SA.items():
        ratio = float(_calibrated(length, area).wettedA) / sa_meas
        assert 0.65 < ratio < 1.0, f"{name}: wettedA/measured {ratio:.2f} outside expected band"


def test_base_prism_surface_area_in_range():
    """The un-calibrated box is roughly the right *area* (0.8-1.4x) even though
    its volume is ~2x too big -- the flip side of the tension."""
    for name, (length, area, sa_meas) in MEASURED_SA.items():
        ratio = float(_base(length).wettedA) / sa_meas
        assert 0.8 < ratio < 1.4, f"{name}: base wettedA/measured {ratio:.2f} outside expected band"


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
