"""
Physical and Model Constants for Iceberg Geometry Calculations


References
----------
- Moon, T., et al. (2018). Nature Geoscience, 11(1), 49-54.
- Barker, A., et al. (2004). ISOPE Conference Proceedings.
- Wagner, T.J.W., et al. (2014). Geophysical Research Letters, 41, 5522-5529.
- Dowdeswell, J.A., et al. (1992). Journal of Geophysical Research, 97, 3515-3528.
- Sulak, D.J., et al. (2017). Annals of Glaciology, 58(74), 89-98.
- Schild, K.M., et al. (2021). Geophysical Research Letters, doi:10.1029/2020GL089765.
"""

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

# Densities (kg/m³)
RHO_ICE = 917                 # Density of glacial ice
RHO_SEAWATER = 1024           # Typical seawater density

# Derived density ratio (fraction of iceberg submerged by buoyancy)
DENSITY_RATIO_ICE_TO_WATER = RHO_ICE / RHO_SEAWATER  # ~0.895 (89.5% submerged)

# ==============================================================================
# EMPIRICAL PARAMETERS (from literature)
# ==============================================================================

# Barker et al. (2004) - Keel depth models
BARKER_COEFFICIENT_A = 2.91   # Coefficient in Barker keel depth formula: K = a * L^b
BARKER_EXPONENT_B = 0.71      # Exponent in Barker formula

HOTZEL_COEFFICIENT_A = 3.78   # Coefficient in Hotzel keel depth formula
HOTZEL_EXPONENT_B = 0.63      # Exponent in Hotzel formula

CONSTANT_KEEL_RATIO = 0.7     # Simple proportional keel depth: K = 0.7 * L

BARKER_HOTZEL_THRESHOLD = 160  # Meters - switch from Barker to Hotzel at this length

# Schild et al. (2021) - Sermilik Fjord large-iceberg keel depth: K = L / ratio.
# Barker's sub-linear law is more appropriate for small bergs.
SCHILD_LENGTH_TO_KEEL_RATIO = 1.98   # L / keel depth (dimensionless)
SCHILD_MIN_LENGTH = 400              # m - below this, warn: outside calibration range

# Barker et al. (2004) - Sail area coefficients (Table 4)
SAIL_AREA_COEFFICIENT_A = 28.194    # Coefficient for sail area calculation
SAIL_AREA_COEFFICIENT_B = -1420.2   # Offset for sail area calculation
SAIL_AREA_LENGTH_THRESHOLD = 65     # Meters - use quadratic formula below this length
SAIL_AREA_QUADRATIC_COEFF = 0.077   # For L < 65m: SailArea = 0.077 * L²

# Barker et al. (2004) - Tabular iceberg
TABULAR_ICEBERG_COEFFICIENT = 0.1211  # For sail area of tabular bergs

# ==============================================================================
# GEOMETRY AND STABILITY
# ==============================================================================

# Wagner et al. (2017) - Stability criterion
STABILITY_THRESHOLD_WH = 0.92  # W/H ratio threshold for stability (W/H ≥ 0.92 is stable)
STABILITY_WIDTH_FACTOR = 0.7   # L/TH ratio for rolling check

# Dowdeswell et al. (1992) - Shape ratios
DEFAULT_LENGTH_TO_WIDTH_RATIO = 1.62  # Typical L:W ratio for Greenland icebergs

# Waterline-footprint-area to total-volume relation: V_total = c * A^x.
AREA_VOLUME_COEFFICIENT = 6.0   # c in V_total = c * A^x  (Sulak et al. 2017)
AREA_VOLUME_EXPONENT = 1.31     # x  (Schild et al. 2021; Sulak 1.30)

FOOTPRINT_SHAPE_OBSERVATIONS = {
    "Schild2021_IcebergA": 0.66,   # drone-measured plan area
    "Schild2021_IcebergB": 0.70,   # drone-measured plan area
    "Schild2024_SF0419": 0.58,     # inferred from mesh volume via V = c * A^x
}
# Accuracy note: the 3-berg mean (0.65) from Schild 2021, 2024
FOOTPRINT_SHAPE_FACTOR = round(
    sum(FOOTPRINT_SHAPE_OBSERVATIONS.values()) / len(FOOTPRINT_SHAPE_OBSERVATIONS), 2
)  # 0.65


# Drone-cloud roughness (3-D DEM area / plan area) at ~1 m grid scale, measured on
# the Schild et al. (2021) point clouds in this repo.
SURFACE_ROUGHNESS_OBSERVATIONS = {
    "Schild2021_IcebergA_t1": 1.190,
    "Schild2021_IcebergB_t1": 1.171,
    "Schild2021_IcebergB_t2": 1.167,
}
# The 3-survey mean (1.18). OPT-IN, not the default: it is a scale-dependent
# enhancement measured on three bergs, too berg-specific to justify applying to
# whole-dataframe sweeps. Pass roughness_factor=SURFACE_ROUGHNESS_OBSERVED when
# comparing against meshed surface areas.
SURFACE_ROUGHNESS_OBSERVED = round(
    sum(SURFACE_ROUGHNESS_OBSERVATIONS.values()) / len(SURFACE_ROUGHNESS_OBSERVATIONS), 2
)  # 1.18

# Default: smooth stack-of-slabs geometry, no roughness enhancement. This makes
# wettedA a documented LOWER BOUND rather than a tuned estimate.
SURFACE_ROUGHNESS_FACTOR = 1.0

# Model depth discretization
DEFAULT_LAYER_THICKNESS_DZ = 5     # meters - default vertical layer thickness
ALTERNATIVE_LAYER_THICKNESS = 10   # meters - alternative layer thickness
# MAX_ICEBERG_DEPTH = 600            # meters - maximum depth modeled (defines z-grid)

TABULAR_THRESHOLD_DEPTH = 200      # meters - keel depth above which assume tabular shape

# ==============================================================================
# NUMERICAL PARAMETERS
# ==============================================================================

# Layer depth adjustments for dz=5m case
DZ_5M_INTERPOLATION_LAYERS = 40    # Number of interpolated layers for dz=5m

# Rounding precision for keel depth
KEEL_DEPTH_ROUNDING_MULTIPLE = 10  # Round length to nearest 10m for keel calculation

# ==============================================================================
# CONFIGURATION FLAGS (Default values)
# ==============================================================================

# Default stability method
DEFAULT_STABILITY_METHOD = 'equal'  # 'equal' or 'keel'


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_constants():
    """
    Validate that constants are physically reasonable.

    Raises
    ------
    ValueError
        If any constant is outside physically reasonable bounds.
    """
    # Density checks
    assert 900 < RHO_ICE < 920, f"Ice density unreasonable: {RHO_ICE}"
    assert 1020 < RHO_SEAWATER < 1030, f"Seawater density unreasonable: {RHO_SEAWATER}"

    # Ratio checks
    assert 0.89 < DENSITY_RATIO_ICE_TO_WATER < 0.90, "Ice/water ratio should be ~0.895"

    # Stability check
    assert 0.9 < STABILITY_THRESHOLD_WH < 1.0, "Stability threshold should be ~0.92"

    # Footprint fill must be a fraction of the bounding rectangle
    assert 0 < FOOTPRINT_SHAPE_FACTOR <= 1, (
        f"Footprint shape factor should be in (0, 1]: {FOOTPRINT_SHAPE_FACTOR}")
    assert all(0 < v <= 1 for v in FOOTPRINT_SHAPE_OBSERVATIONS.values()), (
        "Each footprint observation should be in (0, 1]")

    # Roughness can only add area; the default stays smooth (opt in explicitly)
    assert SURFACE_ROUGHNESS_FACTOR == 1.0, (
        f"Default roughness should be smooth (1.0): {SURFACE_ROUGHNESS_FACTOR}")
    assert all(v >= 1 for v in SURFACE_ROUGHNESS_OBSERVATIONS.values()), (
        "Each roughness observation should be >= 1 (roughness only adds area)")

    print("✓ All constants validated successfully")


if __name__ == "__main__":
    """Run validation when module is executed directly."""
    print("Iceberg Model Constants")
    print("=" * 70)
    print(f"Ice density: {RHO_ICE} kg/m³")
    print(f"Seawater density: {RHO_SEAWATER} kg/m³")
    print(f"Fraction submerged: {DENSITY_RATIO_ICE_TO_WATER:.1%}")
    print(f"Stability threshold (W/H): {STABILITY_THRESHOLD_WH}")
    print(f"Default L:W ratio: {DEFAULT_LENGTH_TO_WIDTH_RATIO}")
    print()

    validate_constants()
