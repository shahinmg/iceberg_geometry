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
# The surface-length-to-keel-depth ratio is ~2:1 (Schild et al. 2021, GRL,
# doi:10.1029/2020GL089765, Results; reported as consistent with the range from
# earlier multibeam work, Barker et al. 1999). The value below is the mean L/keel
# over the four Schild surveys (Table 1: 1.90, 1.94, 2.03, 2.05). Calibrated on
# two deep-keeled Sermilik icebergs (L ~ 500-730 m) -- valid for LARGE icebergs
# only; Barker's sub-linear law is more appropriate for small bergs.
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
# Waterline footprint fills ~0.68 of its L x W bounding rectangle (mean over the
# two Schild drone footprints: 0.66, 0.70), converting model L x W to real area.
FOOTPRINT_SHAPE_FACTOR = 0.68

# Model depth discretization
DEFAULT_LAYER_THICKNESS_DZ = 5     # meters - default vertical layer thickness
ALTERNATIVE_LAYER_THICKNESS = 10   # meters - alternative layer thickness
MAX_ICEBERG_DEPTH = 600            # meters - maximum depth modeled (defines z-grid)

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
