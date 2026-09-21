"""Evidence that the length-only keel and the roughness factor *generalize*.

The keel ratio (K = L / 1.98) and the roughness factor were both derived from
Schild et al. (2021) Icebergs A and B, then compared back to A and B -- a fit,
not a test of prediction. These checks capture the cross-berg evidence that they
actually transfer, so the finding lives in the repo rather than only in analysis
notes:

Run standalone (no pytest needed):  python tests/test_generalization.py
"""

import iceberg_geometry.constants as const

# Schild et al. 2021, Table 1: (surface_length_m, keel_depth_m)
KEEL = {
    "A_survey1": (733, 386),
    "A_survey2": (729, 375),
    "B_survey1": (518, 255),
    "B_survey2": (515, 251),
}

# Drone-cloud roughness (3-D DEM area / plan area) at ~1 m grid scale, this repo.
ROUGHNESS_1M = {
    "A_t1": 1.190,
    "B_t1": 1.171,
    "B_t2": 1.167,
}


def _ratio(surveys):
    """Mean L / keel over the given surveys."""
    return sum(L / k for L, k in (KEEL[s] for s in surveys)) / len(surveys)


def test_keel_ratio_cross_prediction_within_8pct():
    """Training the L:keel ratio on one berg predicts the other within 8%."""
    r_from_A = _ratio(["A_survey1", "A_survey2"])
    r_from_B = _ratio(["B_survey1", "B_survey2"])
    # A-trained ratio predicts B
    for s in ("B_survey1", "B_survey2"):
        L, keel = KEEL[s]
        err = abs((L / r_from_A) / keel - 1.0)
        assert err < 0.08, f"A-ratio predicts {s} to {err:.1%} (>8%)"
    # B-trained ratio predicts A
    for s in ("A_survey1", "A_survey2"):
        L, keel = KEEL[s]
        err = abs((L / r_from_B) / keel - 1.0)
        assert err < 0.08, f"B-ratio predicts {s} to {err:.1%} (>8%)"


def test_default_keel_ratio_is_survey_mean():
    """The shipped ratio (1.98) is the mean over all four surveys."""
    mean_ratio = _ratio(list(KEEL))
    assert abs(const.SCHILD_LENGTH_TO_KEEL_RATIO - mean_ratio) < 0.02, (
        f"constant {const.SCHILD_LENGTH_TO_KEEL_RATIO} vs survey mean {mean_ratio:.3f}"
    )


def test_roughness_consistent_across_bergs():
    """The measured 1 m-scale roughness is nearly berg-independent (<5% spread)."""
    vals = list(ROUGHNESS_1M.values())
    spread = max(vals) - min(vals)
    assert spread < 0.05, f"roughness spread {spread:.3f} across bergs is too large"


def test_observed_roughness_is_survey_mean():
    """The documented opt-in roughness (1.18) is the mean of the drone surveys."""
    mean_rough = sum(ROUGHNESS_1M.values()) / len(ROUGHNESS_1M)
    assert abs(const.SURFACE_ROUGHNESS_OBSERVED - mean_rough) < 0.01, (
        f"constant {const.SURFACE_ROUGHNESS_OBSERVED} vs survey mean {mean_rough:.3f}"
    )


def test_default_roughness_is_smooth():
    """The SHIPPED DEFAULT is smooth (1.0), not the measured 1.18.

    The enhancement is scale-dependent and measured on three bergs, so it is too
    berg-specific to apply to whole-dataframe sweeps; wettedA is therefore a
    documented lower bound unless the caller opts in with roughness_factor=.
    """
    assert const.SURFACE_ROUGHNESS_FACTOR == 1.0, (
        f"default roughness should be 1.0, got {const.SURFACE_ROUGHNESS_FACTOR}")


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
