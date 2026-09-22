"""Validate the Sermilik keel/volume calibration against Schild et al. (2021).

Run standalone (no pytest needed):  python tests/test_volume_calibration.py
Or, if pytest is installed:          pytest tests/test_volume_calibration.py
"""

import warnings

from iceberg_geometry import Iceberg
import iceberg_geometry.constants as const

# name -> (surface_length_m, keel_depth_m, waterline_footprint_area_m2, V_total_m3)
# length / keel / V_total: Schild et al. 2021, Table 1.
# footprint area: convex-hull plan area of the drone cloud (Drone_Iceberg*.txt).
MEASURED = {
    "A_survey1": (733, 386, 2.20e5, 6.52e7),
    "A_survey2": (729, 375, 2.17e5, 6.34e7),
    "B_survey1": (518, 255, 1.67e5, 3.17e7),
    "B_survey2": (515, 251, 1.67e5, 3.10e7),
}


def _calibrated(length, area):
    """Model dataset with the full Sermilik calibration and a measured footprint."""
    berg = Iceberg(length=length, dz=5)
    return berg.init_iceberg_size(
        keel_method="schild", volume_law="sulak", area=area
    )


def test_schild_keel_matches_measured():
    """schild keel (K = L / 1.98) is within 8% of the measured keel, all surveys."""
    for name, (length, keel, _, _) in MEASURED.items():
        model_keel = float(Iceberg(length=length, dz=5).keeldepth("schild"))
        rel = abs(model_keel / keel - 1.0)
        assert rel < 0.08, f"{name}: keel {model_keel:.0f} m vs {keel} m ({rel:.1%})"


def test_iceberg_A_volume_within_15pct():
    """Deep tabular berg (A): calibrated volume matches measured within 15%."""
    for name in ("A_survey1", "A_survey2"):
        length, _, area, v_meas = MEASURED[name]
        v_model = float(_calibrated(length, area).totalV.values)
        rel = abs(v_model / v_meas - 1.0)
        assert rel < 0.15, f"{name}: V {v_model:.3e} vs {v_meas:.3e} ({rel:+.1%})"


def test_proxy_area_matches_A_without_measurement():
    """The footprint_factor*L*W fallback (no measured area) stays within 20% of
    measured V for A."""
    for name in ("A_survey1", "A_survey2"):
        length, _, _, v_meas = MEASURED[name]
        ds = Iceberg(length=length, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak"
        )
        rel = abs(float(ds.totalV.values) / v_meas - 1.0)
        assert rel < 0.20, f"{name}: proxy V {float(ds.totalV.values):.3e} vs {v_meas:.3e} ({rel:+.1%})"


def test_iceberg_B_is_known_high_outlier():
    """Small near-equant berg (B) sits off the population fit: the volume law
    over-predicts by ~30%. Encoded as a regression guard, not an accuracy goal."""
    for name in ("B_survey1", "B_survey2"):
        length, _, area, v_meas = MEASURED[name]
        ratio = float(_calibrated(length, area).totalV.values) / v_meas
        assert 1.15 < ratio < 1.45, f"{name}: V ratio {ratio:.2f} outside expected outlier band"


def test_observed_width_changes_shape_not_volume():
    """Passing an observed `width` sets waterline W but leaves the calibrated
    (area-driven) total volume unchanged."""
    berg = Iceberg(length=518, dz=5)
    default = berg.init_iceberg_size(keel_method="schild", volume_law="sulak", area=1.67e5)
    equant = berg.init_iceberg_size(keel_method="schild", volume_law="sulak", area=1.67e5, width=463)
    assert abs(float(equant.W) - 463.0) < 1e-6, f"W {float(equant.W):.1f} != 463"
    assert abs(float(default.W) - 518 / 1.62) < 1e-6, "default W should use the 1.62 ratio"
    rel = abs(float(equant.totalV) / float(default.totalV) - 1.0)
    assert rel < 1e-9, f"total volume changed with width by {rel:.2e}"


def test_totalV_equals_uwV_plus_sailV():
    """Sanity: total volume is the sum of underwater and sail volumes."""
    ds = _calibrated(733, 2.20e5)
    total = float(ds.totalV.values)
    parts = float(ds.uwV.sum().values) + float(ds.sailV.values)
    assert abs(total / parts - 1.0) < 1e-6, f"totalV {total:.3e} != uwV+sailV {parts:.3e}"


def test_measured_area_hits_sulak_volume_exactly():
    """With a measured `area`, totalV IS c*A^x -- not an approximation to it.

    The taper is solved on the discrete Z grid, so the calibrated volume must be
    exact and independent of dz. Solving the continuous closed form instead left a
    quadrature deficit (-4% at dz=20, -0.8% at dz=5) that silently biased every run.
    """
    c, x = const.AREA_VOLUME_COEFFICIENT, const.AREA_VOLUME_EXPONENT
    for name, (length, _, area, _) in MEASURED.items():
        for dz in (20, 10, 5, 2, 1):
            ds = Iceberg(length=length, dz=dz).init_iceberg_size(
                keel_method="schild", volume_law="sulak", area=area
            )
            ratio = float(ds.totalV.values) / (c * area ** x)
            assert abs(ratio - 1.0) < 1e-9, (
                f"{name} dz={dz}: totalV is {ratio:.6f} x c*A^x, not exact")


def test_unsaturated_bergs_are_not_flagged():
    """Bergs the taper can actually solve hit c*A^x exactly and warn about nothing.

    L=250/300 m used to overshoot the target by 0.7-1.7% while the continuous
    saturation test said they were fine.
    """
    c, x = const.AREA_VOLUME_COEFFICIENT, const.AREA_VOLUME_EXPONENT
    for length in (200, 250, 300, 400, 733):
        area = 0.65 * length * (length / const.DEFAULT_LENGTH_TO_WIDTH_RATIO)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = Iceberg(length=length, dz=5).init_iceberg_size(
                keel_method="schild", volume_law="sulak", area=area
            )
        saturated = [w for w in caught if "taper solve saturated" in str(w.message)]
        ratio = float(ds.totalV.values) / (c * area ** x)
        assert not saturated, f"L={length}: spurious saturation warning at ratio {ratio:.4f}"
        assert abs(ratio - 1.0) < 1e-9, f"L={length}: totalV is {ratio:.6f} x c*A^x"


def test_saturated_bergs_still_warn():
    """The end stops are real: a keel too shallow for c*A^x must still warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Iceberg(length=50, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak",
            area=0.65 * 50 * (50 / const.DEFAULT_LENGTH_TO_WIDTH_RATIO),
        )
    assert any("taper solve saturated" in str(w.message) for w in caught), \
        "L=50 m cannot reach c*A^x and must warn"


def test_measured_area_ignores_footprint_default():
    """A measured `area` fully overrides the point-cloud FOOTPRINT_SHAPE_FACTOR.

    Guards the strict-Sulak path: nothing derived from the Schild point clouds may
    move the volume once the caller supplies their own footprint area.
    """
    c, x = const.AREA_VOLUME_COEFFICIENT, const.AREA_VOLUME_EXPONENT
    area = 2.20e5
    v = float(_calibrated(733, area).totalV.values)
    assert abs(v / (c * area ** x) - 1.0) < 1e-9
    # area and footprint_factor are mutually exclusive, so they cannot be mixed
    try:
        Iceberg(length=733, dz=5).init_iceberg_size(
            keel_method="schild", volume_law="sulak", area=area, footprint_factor=0.65)
    except ValueError:
        pass
    else:
        raise AssertionError("area + footprint_factor together should raise ValueError")


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
