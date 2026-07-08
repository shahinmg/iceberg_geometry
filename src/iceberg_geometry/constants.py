"""
Physical and Model Constants for Iceberg Simulations

This module contains all physical constants, empirical parameters, and 
configuration values used in the iceberg geometry and melt model.

Constants follow PEP 8 convention of SCREAMING_SNAKE_CASE to indicate
they should not be modified during runtime.

References
----------
- Moon, T., et al. (2018). Nature Geoscience, 11(1), 49-54.
- Barker, A., et al. (2004). ISOPE Conference Proceedings.
- Wagner, T.J.W., et al. (2014). Geophysical Research Letters, 41, 5522-5529.
- Dowdeswell, J.A., et al. (1992). Journal of Geophysical Research, 97, 3515-3528.
- Jenkins, A. (2011). Journal of Physical Oceanography, 41, 2279-2294.
- Bigg, G.R., et al. (1997). Cold Regions Science and Technology, 26, 113-135.
"""

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

# Densities (kg/m³)
RHO_ICE = 917                  # Density of glacial ice
RHO_SEAWATER = 1024           # Typical seawater density
RHO_FRESHWATER = 1000         # Pure water at 4°C
RHO_AIR = 1.225               # Air density at sea level, 15°C

# Derived density ratios
DENSITY_RATIO_ICE_TO_WATER = RHO_ICE / RHO_SEAWATER  # ~0.895 (89.5% submerged)
DENSITY_RATIO_ICE_TO_FRESHWATER = RHO_ICE / RHO_FRESHWATER  # ~0.917

# Thermal properties
LATENT_HEAT_FUSION = 3.35e5   # J/kg - latent heat of fusion (ice→water) - CORRECTED to match original
SPECIFIC_HEAT_ICE = 2009      # J/(kg·K) - specific heat capacity of ice
SPECIFIC_HEAT_WATER = 3974    # J/(kg·K) - specific heat capacity of water

# Temperature
CORE_ICE_TEMPERATURE = -4     # °C - typical iceberg core temperature
ICE_SURFACE_TEMPERATURE = -4  # °C - ice surface temperature for melt calculations
TEMPERATURE_DIFFERENCE_CORE_SURFACE = 15  # K - temp difference for Antarctica (may be less for Greenland)

# Optical properties
ICE_ALBEDO = 0.7              # Fraction of solar radiation reflected by ice (dimensionless, 0-1)
SOLAR_ABSORPTION_FRACTION = 1 - ICE_ALBEDO  # Fraction absorbed (0.3)

# Fluid properties (air at ~15°C)
AIR_KINEMATIC_VISCOSITY = 1.46e-5  # m²/s - kinematic viscosity of air
AIR_THERMAL_DIFFUSIVITY = 2.16e-5  # m²/s - thermal diffusivity of air  
AIR_THERMAL_CONDUCTIVITY = 0.0249  # W/(m·K) - thermal conductivity at 0°C

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
STABILITY_WIDTH_FACTOR = 0.7   # L/TH ratio for rolling check (used in time evolution)

# Dowdeswell et al. (1992) - Shape ratios
DEFAULT_LENGTH_TO_WIDTH_RATIO = 1.62  # Typical L:W ratio for Greenland icebergs

# Model depth discretization
DEFAULT_LAYER_THICKNESS_DZ = 5     # meters - default vertical layer thickness
ALTERNATIVE_LAYER_THICKNESS = 10   # meters - alternative layer thickness
MAX_ICEBERG_DEPTH = 600            # meters - maximum depth modeled (defines z-grid)

TABULAR_THRESHOLD_DEPTH = 200      # meters - keel depth above which assume tabular shape

# ==============================================================================
# MELT PARAMETERS
# ==============================================================================

# Jenkins (2011) / Holland & Jenkins (1999) - Transfer coefficients
HEAT_TRANSFER_COEFFICIENT_GT = 1.1e-3   # Heat transfer coefficient (can be scaled by factor)
SALT_TRANSFER_COEFFICIENT_GS = 3.1e-5   # Salt transfer coefficient (can be scaled by factor)

# Jackson et al. (2020) - Adjustment factor for transfer coefficients
DEFAULT_TRANSFER_COEFFICIENT_FACTOR = 1  # Multiplicative factor for GT and GS

# Freezing point calculation coefficients
FREEZING_POINT_SALINITY_COEFF = -5.73e-2    # °C/PSU - salinity contribution (a)
FREEZING_POINT_CONSTANT = 8.32e-2           # °C - constant term (b)
FREEZING_POINT_PRESSURE_COEFF = -7.61e-4    # °C/dbar - pressure contribution (c)

# Alternative freezing point formula (Bigg method)
BIGG_FP_COEFF_1 = -0.036        # First coefficient
BIGG_FP_COEFF_2 = -0.0499       # Linear salinity term
BIGG_FP_COEFF_3 = -0.0001128    # Quadratic salinity term
BIGG_FP_EXPONENT_COEFF = -0.19  # Exponential term coefficient

# Buoyant convection (Bigg/CIS method) - El-Tahan formula
BUOYANT_MELT_LINEAR_COEFF = 7.62e-3    # m/(day·°C) - linear coefficient
BUOYANT_MELT_QUADRATIC_COEFF_BIGG = 1.3e-3   # m/(day·°C²) - quadratic (Bigg)
BUOYANT_MELT_QUADRATIC_COEFF_CIS = 1.29e-3   # m/(day·°C²) - quadratic (CIS)

# Wave erosion (Silva et al. / Bigg formulation)
WAVE_HEIGHT_WIND_COEFF_1 = 1.5     # Coefficient: wave_height = 1.5 * sqrt(wind)
WAVE_HEIGHT_WIND_COEFF_2 = 0.1     # Coefficient: wave_height = ... + 0.1 * wind
WAVE_MELT_DIVISOR = 12             # Divisor in melt rate formula
WAVE_MELT_TEMP_OFFSET = 2          # Temperature offset (SST + 2)
WAVE_HEIGHT_ESTIMATE_COEFF = 0.010125  # wind² coefficient for wave height estimation
MAX_WAVE_PENETRATION_FACTOR = 5    # Wave height multiplier for penetration depth

# Forced air convection - Nusselt number calculation  
NUSSELT_COEFF = 0.058              # Coefficient in Nu = 0.058 * Re^0.8 / Pr^0.4
NUSSELT_REYNOLDS_EXPONENT = 0.8    # Reynolds number exponent
NUSSELT_PRANDTL_EXPONENT = 0.4     # Prandtl number exponent

# ==============================================================================
# TIME CONVERSIONS
# ==============================================================================

SECONDS_PER_DAY = 86400           # Standard conversion
DAYS_PER_SECOND = 1 / SECONDS_PER_DAY  # Inverse

# Default timestep
DEFAULT_TIMESTEP_SECONDS = 86400  # 1 day
DEFAULT_TIMESTEP_DAYS = 1         # 1 day

# ==============================================================================
# MELT MECHANISM MULTIPLIERS
# ==============================================================================

# Number of surfaces affected by each melt mechanism
WAVE_LENGTH_SURFACES = 1    # Wave affects 1 length face
WAVE_WIDTH_SURFACES = 1     # Wave affects 1 width face

FORCED_WATER_LENGTH_SURFACES = 2   # Forced water affects 2 length faces (both sides)
FORCED_WATER_WIDTH_SURFACES = 1    # Forced water affects 1 width face (lee side sheltered)
FORCED_WATER_BASE_SURFACES = 1     # Forced water affects base

FORCED_AIR_LENGTH_SURFACES = 2     # Forced air affects 2 length faces
FORCED_AIR_WIDTH_SURFACES = 1      # Forced air affects 1 width face (lee sheltered)
FORCED_AIR_TOP_SURFACE_FRACTION = 0.5  # Forced air affects half of top surface

BUOYANT_WATER_LENGTH_SURFACES = 2  # Buoyant convection affects both length sides
BUOYANT_WATER_WIDTH_SURFACES = 2   # Buoyant convection affects both width sides

SOLAR_TOP_SURFACE_FRACTION = 1.0   # Solar affects entire top surface

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

# Default melt mechanisms to include
DEFAULT_MELT_MECHANISMS = {
    'wave': True,      # Wave erosion
    'turbw': True,     # Forced convection in water
    'turba': True,     # Forced convection in air
    'freea': True,     # Free convection (solar) in air
    'freew': True,     # Free (buoyant) convection in water
}

# Default stability methods
DEFAULT_STABILITY_METHOD = 'equal'  # 'equal' or 'keel'

# Default ice temperature usage
DEFAULT_USE_CONSTANT_TF = False  # Whether to use constant thermal forcing
DEFAULT_CONSTANT_TF = None       # Value if using constant thermal forcing

# ==============================================================================
# HELPER DERIVED CONSTANTS
# ==============================================================================

# Prandtl number (for forced air convection)
PRANDTL_NUMBER = AIR_KINEMATIC_VISCOSITY / AIR_THERMAL_DIFFUSIVITY

# Commonly used products
RHO_ICE_TIMES_LATENT_HEAT = RHO_ICE * LATENT_HEAT_FUSION  # kg/m³ * J/kg = J/m³


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
    
    # Thermal checks
    assert LATENT_HEAT_FUSION > 3e5, "Latent heat seems too small"
    
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