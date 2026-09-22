"""
Iceberg Geometry and Stability Model

This module implements iceberg size initialization and geometry calculations based on
empirical relationships and stability criteria from the scientific literature.

Original Model:
    Moon, T., Sutherland, D.A., Carroll, D., Felikson, D., Kehrl, L., & Straneo, F. (2018).
    Subsurface iceberg melt key to Greenland fjord freshwater budget.
    Nature Geoscience, 11(1), 49-54. https://doi.org/10.1038/s41561-017-0018-z
    
    Original MATLAB implementation available in supplementary materials.
    
Key References:
    - Barker, A., Sayed, M., & Carrieres, T. (2004). Determination of iceberg draft, mass 
      and cross-sectional areas. Proceedings of the 14th International Offshore and Polar 
      Engineering Conference.
    - Wagner, T.J.W., Wadhams, P., Bates, R., et al. (2014). The 'footloose' mechanism: 
      iceberg decay from hydrostatic stresses. Geophysical Research Letters, 41, 5522-5529.
      https://doi.org/10.1002/2014GL060832
    - Dowdeswell, J.A., Whittington, R.J., & Hodgkins, R. (1992). The sizes, frequencies, 
      and freeboards of East Greenland icebergs. Journal of Geophysical Research, 97, 
      3515-3528.


"""

import warnings
import numpy as np
import numpy.matlib
import xarray as xr
from importlib.metadata import version, PackageNotFoundError

# Import constants module
try:
    from . import constants as const
except ImportError:
    # Fallback for when running as standalone script
    import constants as const




class Iceberg:
    """
    A class for modeling iceberg geometry and stability.
    
    Parameters
    ----------
    length : float, optional
        Length of the iceberg in meters. Must be positive. If None, must be
        set before calling geometry calculation methods. Default is None.
    dz : float, optional
        Vertical layer thickness for depth discretization in meters. 
        Common values are 5m or 10m. Must be positive. Default is 5.
    
    Attributes
    ----------
    length : float or None
        Iceberg length in meters
    dz : float
        Layer thickness in meters
    
    Examples
    --------
    >>> # Create an iceberg with specified length
    >>> iceberg = Iceberg(length=200, dz=5)
    >>> keel = iceberg.keeldepth(method='barker')
    >>> geometry = iceberg.init_iceberg_size()
    
    >>> # Create with default dz, set length later
    >>> iceberg = Iceberg(dz=10)
    >>> iceberg.length = 150
    >>> geometry = iceberg.init_iceberg_size()
    
    References
    ----------
    Barker, A., Sayed, M., & Carrieres, T. (2004). Determination of iceberg 
    draft, mass and cross-sectional areas.
    
    Wagner, T. J., et al. (2017). The "footloose" mechanism: Iceberg decay from
    hydrostatic stresses. Geophysical Research Letters.
    """
    
    # Physical constants (imported from constants module)
    RHO_ICE = const.RHO_ICE
    RHO_WATER = const.RHO_SEAWATER
    STABILITY_THRESHOLD = const.STABILITY_THRESHOLD_WH
    DEFAULT_LW_RATIO = const.DEFAULT_LENGTH_TO_WIDTH_RATIO

    VARIABLE_ATTRS = {
        # Coordinate
        "Z": {
            "long_name": "Depth below waterline",
            "units": "m",
            "positive": "down",
            "description": "Depth of the bottom of each vertical layer below the waterline.",
        },
        # Depth-resolved (per-layer) variables from barker_carea()
        "depth": {
            "long_name": "Layer depth below waterline",
            "units": "m",
            "description": "Depth coordinate value at each underwater layer.",
        },
        "cross_area": {
            "long_name": "Underwater cross-sectional area",
            "units": "m2",
            "description": "Vertical cross-section along the long axis: uwL * dz "
                           "(Barker et al. 2004 for keel <= 200 m; tabular below). "
                           "Bounding-box section -- no footprint fill fraction.",
        },
        "uwL": {
            "long_name": "Underwater length",
            "units": "m",
            "description": "Iceberg length at each depth layer.",
        },
        "uwW": {
            "long_name": "Underwater width",
            "units": "m",
            "description": "Iceberg width at each depth layer, from underwater length "
                           "divided by the length-to-width ratio.",
        },
        "uwV": {
            "long_name": "Underwater layer volume",
            "units": "m3",
            "description": "Volume of the iceberg contained in each depth layer. The "
                           "exact formula depends on volume_law -- see the "
                           "'computation' attribute on this variable.",
        },
        # Scalar summary variables added by init_iceberg_size()
        "totalV": {
            "long_name": "Total iceberg volume",
            "units": "m3",
            "description": "Total iceberg volume, underwater plus above-water (sail).",
        },
        "sailV": {
            "long_name": "Sail (above-water) volume",
            "units": "m3",
            "description": "Above-water (sail) volume of the iceberg.",
        },
        "wettedA": {
            "long_name": "Wetted (submerged) surface area",
            "units": "m2",
            "description": "Submerged surface area (lateral walls plus basal footprint) ",
        },
        "basalA": {
            "long_name": "Basal (keel bottom) footprint area",
            "units": "m2",
            "description": "Plan-view area of the flat bottom face at the keel (deepest "
                           "layer length x width)",
        },
        "footprint_factor": {
            "long_name": "Waterline footprint fill fraction",
            "units": "1",
            "description": "Real plan-view footprint area divided by the L*W bounding "
                           "rectangle.",
        },
        "perimeter_factor": { # might remove this
            "long_name": "Waterline perimeter shape factor",
            "units": "1",
            "description": "Real (rounded) outline perimeter divided by the bounding-"
                           "rectangle perimeter 2*(L+W), applied to every layer's "
                           "lateral wall area.",
        },
        "roughness": { # also might remove this
            "long_name": "Surface roughness enhancement factor",
            "units": "1",
            "description": "Multiplier applied to the smooth-geometry wetted area to "
                           "account for surface roughness (true rough area / smooth area).",
        },
        "W": {
            "long_name": "Waterline width",
            "units": "m",
            "description": "Iceberg width at the waterline.",
        },
        "freeB": {
            "long_name": "Freeboard height",
            "units": "m",
            "description": "Height of the iceberg above the waterline.",
        },
        "L": {
            "long_name": "Waterline length",
            "units": "m",
            "description": "Iceberg length at the waterline.",
        },
        "keel": {
            "long_name": "Keel depth",
            "units": "m",
            "description": "Depth of the deepest part of the iceberg below the waterline.",
        },
        "TH": {
            "long_name": "Total thickness",
            "units": "m",
            "description": "Total iceberg height, keel depth plus freeboard.",
        },
        "keeli": {
            "long_name": "Deepest layer index",
            "units": "1",
            "description": "Index of the deepest iceberg layer (keel depth / dz, rounded up).",
        },
        "dz": {
            "long_name": "Layer thickness",
            "units": "m",
            "description": "Vertical layer thickness used for depth discretization.",
        },
        "dzk": {
            "long_name": "Keel partial-layer thickness",
            "units": "m",
            "description": "Thickness of the partial layer at the keel.",
        },
    }

    def _wetted_surface_area(self, ice, dz, perimeter_factor=1.0): # might remove 
        """Submerged (wetted) surface area of the iceberg: lateral walls + basal.



        Parameters
        ----------
        ice : xarray.Dataset
            Geometry with per-layer ``uwL`` and ``uwW`` (m).
        dz : float
            Layer thickness (m).
        perimeter_factor : float, optional
            Scales every layer's bounding-rectangle perimeter to the real
            (rounded) outline perimeter. Default 1.0 keeps the rectangle, which
            OVERSTATES the lateral area -- the 2017 drone hulls measure
            perimeter / 2*(L+W) = 0.83-0.88. Set from a measured ``perimeter``.

        Returns
        -------
        (float, float)
            ``(lateral, basal)`` smooth surface areas in m^2: the side-wall area
            (sum of perimeter * layer thickness) and the keel bottom footprint.
        """
        uwL = np.asarray(ice['uwL'].values, dtype=float).ravel()
        uwW = np.asarray(ice['uwW'].values, dtype=float).ravel()
        mask = ~np.isnan(uwL) & ~np.isnan(uwW)
        if not np.any(mask):
            return 0.0, 0.0
        L = uwL[mask]
        W = uwW[mask]
        lateral = float(np.sum(2.0 * (L + W)) * dz) * perimeter_factor  # perimeter * thickness
        basal = float(L[-1] * W[-1])                 # bottom footprint of the keel layer
        return lateral, basal

    @staticmethod
    def _resolve_footprint_factor(footprint_factor, area=None): # might remove 
        """Validate and resolve the waterline footprint fill fraction.

        Returns FOOTPRINT_SHAPE_FACTOR when ``footprint_factor`` is None. Raises
        if it is set alongside ``area``, since both specify the same quantity
        (``area`` measures it; ``footprint_factor`` estimates it from L*W).
        """
        if footprint_factor is None:
            return const.FOOTPRINT_SHAPE_FACTOR
        if area is not None:
            raise ValueError(
                "footprint_factor and area both set the waterline footprint; pass "
                "only one (area is the direct measurement, so prefer it)")
        ff = float(footprint_factor)
        if not 0 < ff <= 1:
            raise ValueError(
                f"footprint_factor must be in (0, 1], got {footprint_factor}")
        return ff

    @staticmethod
    def _resolve_perimeter_factor(perimeter, length, width): # might remove 
        """Resolve the waterline perimeter shape factor from a measured perimeter.

        """
        if perimeter is None:
            return 1.0
        perimeter = float(perimeter)
        if perimeter <= 0:
            raise ValueError(f"perimeter must be positive, got {perimeter}")
        rect = 2.0 * (float(length) + float(width))
        factor = perimeter / rect
        if factor > 1.0:
            warnings.warn(
                f"measured perimeter {perimeter:.0f} m exceeds the bounding-rectangle "
                f"perimeter {rect:.0f} m (factor {factor:.2f} > 1), which is impossible "
                "for a smooth outline -- the polygon is probably unsimplified, and its "
                "digitization raggedness is already counted by roughness_factor. "
                "Measure the perimeter on a fixed smoothing convention (see "
                "iceberg_geometry.footprint_metrics).",
                stacklevel=3)
        return factor

    def _add_surface_area(self, ice, dz, roughness_factor, footprint_factor,
                          perimeter_factor=1.0):
        """Add wetted-surface-area variables to a geometry dataset.

        Stores ``wettedA`` (smooth lateral+basal area times the roughness factor),
        ``basalA`` (the smooth keel bottom bounding-box footprint, roughness NOT
        applied), ``footprint_factor`` (real-footprint / L*W fill fraction),
        ``perimeter_factor`` (real-outline / bounding-rectangle perimeter) and
        ``roughness`` (the factor used). The smooth total area is recoverable as
        ``wettedA / roughness`` = lateral + basalA; the rounded keel footprint is
        ``footprint_factor * basalA``.
        """
        lateral, basal = self._wetted_surface_area(
            ice, dz, perimeter_factor=perimeter_factor)
        ice['wettedA'] = xr.DataArray(data=(lateral + basal) * roughness_factor, name='wettedA')
        ice['basalA'] = xr.DataArray(data=basal, name='basalA')
        ice['footprint_factor'] = xr.DataArray(data=float(footprint_factor), name='footprint_factor')
        ice['perimeter_factor'] = xr.DataArray(data=float(perimeter_factor), name='perimeter_factor')
        ice['roughness'] = xr.DataArray(data=float(roughness_factor), name='roughness')
        return ice

    def _assign_variable_attrs(self, ds):
        """Attach descriptive attrs (long_name, units, description) to a dataset.
        """
        for name, attrs in self.VARIABLE_ATTRS.items():
            if name in ds.variables:
                ds[name].attrs.update(attrs)
        return ds

    def _assign_global_attrs(self, ds):
        """Attach dataset-level (global) metadata so a saved netCDF is self-documenting.

        """
        try:
            pkg_version = version("iceberg-geometry")
        except PackageNotFoundError:
            pkg_version = "unknown"

        ds.attrs.update({
            "title": "Iceberg geometry and stability model output",
            "summary": (
                "Underwater and above-water iceberg geometry derived from waterline "
                "length using empirical shape relationships (Barker et al. 2004) and a "
                "hydrostatic stability criterion (Wagner et al. 2014)."
            ),
            "model_version": pkg_version,
            "references": " | ".join(self.citation().values()),
        })
        return ds


    def __init__(self, length=None, dz=5):
        """
        Initialize an Iceberg instance.
        
        Parameters
        ----------
        length : float, optional
            Length of the iceberg in meters. Must be positive.
        dz : float, optional
            Layer thickness in meters. Must be positive. Default is 5.
            
        Raises
        ------
        ValueError
            If length is provided and is not positive.
            If dz is not positive.
        """
        # Validate dz
        if dz <= 0:
            raise ValueError(f"dz must be positive, got {dz}")
        
        # Validate length if provided
        if length is not None:
            if length <= 0:
                raise ValueError(f"length must be positive, got {length}")
            # Convert to float for consistency
            self.length = float(length)
        else:
            self.length = length
        
        # Convert dz to float for consistency
        self.dz = float(dz)
        
        # Initialize computed attributes as None (calculated by methods)
        self._keel_depth = None
        self._geometry = None
    
    def __repr__(self):
        """Return string representation of the Iceberg."""
        if self.length is not None:
            return f"Iceberg(length={self.length:.1f}m, dz={self.dz:.1f}m)"
        else:
            return f"Iceberg(length=None, dz={self.dz:.1f}m)"
    
    def __str__(self):
        """Return human-readable string description of the Iceberg."""
        if self.length is not None:
            if self.length > 0:
                keel_depth = self.keeldepth()
                # Convert to scalar if it's an array
                keel_value = float(keel_depth) if isinstance(keel_depth, np.ndarray) else keel_depth
                keel_str = f", estimated keel: ~{keel_value:.1f}m"
            else:
                keel_str = ""
            return f"Iceberg with length {self.length:.1f}m{keel_str}"
        else:
            return "Iceberg (length not set)"
    
    def _validate_length(self):
        """
        Validate that length is set and positive.
        
        Raises
        ------
        ValueError
            If length is None or not positive.
        """
        if self.length is None:
            raise ValueError("Iceberg length must be set before calculating geometry. "
                           "Set it with: iceberg.length = <value>")
        if self.length <= 0:
            raise ValueError(f"Iceberg length must be positive, got {self.length}")
    
    @staticmethod
    def citation():
        """
        Return citation information for this iceberg model.
        
        Returns
        -------
        dict
            Dictionary containing citation information for the model and key references.
        
        Examples
        --------
        >>> print(Iceberg.citation()['primary'])
        Moon, T., et al. (2018). Subsurface iceberg melt key to Greenland fjord...
        """
        return {
            'primary': (
                "Moon, T., Sutherland, D.A., Carroll, D., Felikson, D., Kehrl, L., & "
                "Straneo, F. (2018). Subsurface iceberg melt key to Greenland fjord "
                "freshwater budget. Nature Geoscience, 11(1), 49-54. "
                "https://doi.org/10.1038/s41561-017-0018-z"
            ),
            'keel_depth': (
                "Barker, A., Sayed, M., & Carrieres, T. (2004). Determination of iceberg "
                "draft, mass and cross-sectional areas. Proceedings of the 14th "
                "International Offshore and Polar Engineering Conference."
            ),
            'stability': (
                "Wagner, T.J.W., Wadhams, P., & Bates, R. (2014). The 'footloose' "
                "mechanism: iceberg decay from hydrostatic stresses. Geophysical Research "
                "Letters, 41, 5522-5529. https://doi.org/10.1002/2014GL060832"
            ),
            'geometry': (
                "Dowdeswell, J.A., Whittington, R.J., & Hodgkins, R. (1992). The sizes, "
                "frequencies, and freeboards of East Greenland icebergs. Journal of "
                "Geophysical Research, 97, 3515-3528."
            )
        }



    def keeldepth(self, method='barker'):
        """
        Calculate iceberg keel depth using empirical relationships.
        
        Computes the deepest underwater extent of the iceberg based on its length
        using various empirical models from the literature.
        
        Parameters
        ----------
        method : {'barker', 'hotzel', 'constant', 'schild', 'mean'}, optional
            Keel depth calculation method. Default is 'barker'.

            - 'barker' : Barker et al. 2004 model: K = 2.91 * L^0.71
            - 'hotzel' : Hotzel model: K = 3.78 * L^0.63
            - 'constant' : Simple proportional: K = 0.7 * L
            - 'schild' : Sermilik large-iceberg ratio K = L / 1.98 (~2:1),
              from Schild et al. 2021. Calibrated on two deep-keeled Sermilik
              icebergs (L ~ 500-730 m); warns below ~400 m (outside its range).
            - 'mean' : Average of multiple methods including hybrid approach
    
        Returns
        -------
        keel_depth : float
            Deepest part of the iceberg in meters below the waterline.
            Returns a scalar float for single iceberg.
        
        Notes
        -----
        The Barker method is recommended for typical icebergs based on empirical
        observations.
        
        Raises
        ------
        ValueError
            If iceberg length is not set or not positive.
        
        Examples
        --------
        >>> iceberg = Iceberg(length=200)
        >>> keel = iceberg.keeldepth(method='barker')
        >>> print(f"Keel depth: {keel:.1f} m")
        
        >>> # Compare methods
        >>> for method in ['barker', 'hotzel', 'constant']:
        ...     keel = iceberg.keeldepth(method=method)
        ...     print(f"{method}: {keel:.1f} m")
        """
        # Validate that length is set
        self._validate_length()
        
        # Store whether input was scalar
        input_is_scalar = np.isscalar(self.length)
        
        # Ensure self.length is treated as an array for consistent indexing
        L_val = np.atleast_1d(self.length)
        L_10 = np.round(L_val / const.KEEL_DEPTH_ROUNDING_MULTIPLE) * const.KEEL_DEPTH_ROUNDING_MULTIPLE
        
        barker_mask = L_10 <= const.BARKER_HOTZEL_THRESHOLD
        hotzel_mask = L_10 > const.BARKER_HOTZEL_THRESHOLD
        
        if method == 'barker':
            result = const.BARKER_COEFFICIENT_A * np.power(L_10, const.BARKER_EXPONENT_B)
        
        elif method == 'hotzel':
            result = const.HOTZEL_COEFFICIENT_A * np.power(L_10, const.HOTZEL_EXPONENT_B)
        
        elif method == 'constant':
            result = const.CONSTANT_KEEL_RATIO * L_10

        elif method == 'schild':
            # Sermilik large-iceberg calibration: keel = L / ~2 (Schild et al.
            # 2021 ~2:1 surface-length-to-keel-depth ratio). Valid for large
            # bergs only; warn below the calibration range.
            if np.any(L_10 < const.SCHILD_MIN_LENGTH):
                warnings.warn(
                    f"method='schild' is calibrated on large Sermilik icebergs "
                    f"(L ~ 500-730 m); length < {const.SCHILD_MIN_LENGTH} m is "
                    f"outside its range -- consider method='barker'.",
                    stacklevel=2)
            result = L_10 / const.SCHILD_LENGTH_TO_KEEL_RATIO

        elif method == 'mean':
            # Create a 2D array: rows = icebergs, columns = different methods
            keel_arr = np.zeros((len(L_10), 4))
            
            # Column 0: Hybrid (Barker if small, Hotzel if large)
            keel_arr[barker_mask, 0] = const.BARKER_COEFFICIENT_A * np.power(L_10[barker_mask], const.BARKER_EXPONENT_B)
            keel_arr[hotzel_mask, 0] = const.HOTZEL_COEFFICIENT_A * np.power(L_10[hotzel_mask], const.HOTZEL_EXPONENT_B)
            
            # Column 1-3: Pure methods
            keel_arr[:, 1] = const.BARKER_COEFFICIENT_A * np.power(L_10, const.BARKER_EXPONENT_B)
            keel_arr[:, 2] = const.HOTZEL_COEFFICIENT_A * np.power(L_10, const.HOTZEL_EXPONENT_B)
            keel_arr[:, 3] = const.CONSTANT_KEEL_RATIO * L_10
            
            result = np.mean(keel_arr, axis=1)

        else:
            raise ValueError(
                f"Unknown keel depth method {method!r}; expected one of "
                "'barker', 'hotzel', 'constant', 'schild', 'mean'.")

        # Return scalar if input was scalar
        if input_is_scalar and result.size == 1:
            return float(result.item())
        else:
            return result


    def barker_carea(self, keel_depth, dz, LWratio=1.62, tabular=200, method='barker',
                     volume_law=None, area=None, wall_slope=None, footprint_factor=None): # prop should remove footprint factor
        """
        Calculate underwater cross-sectional areas and iceberg geometry using Barker et al. 2004 model.
        
        Parameters
        ----------
        keel_depth : float or array-like
            Deepest part of the iceberg in meters. Can be a scalar or array.
        dz : float
            Layer thickness for depth discretization in meters (typically 5 or 10).
            Each layer represents a horizontal slice of the iceberg.
        LWratio : float, optional
            Length-to-width ratio for the iceberg. Default is 1.62:1 based on 
            Dowdeswell et al. observations. Used to calculate underwater widths
            from underwater lengths.
        tabular : float, optional
            Keel depth threshold in meters for switching between Barker model and
            tabular shape assumption. Default is 200m.
        method : str, optional
            Iceberg shape model to use. Default is 'barker'. Currently only 'barker'
            is implemented in this method.
        volume_law : {None, 'sulak'}, optional
            If 'sulak', taper the underwater cross-section so total volume follows
            the empirical waterline-area-to-volume relation V = c*A^x
            (Sulak et al. 2017 / Schild et al. 2021). Default None keeps the prism
            volume. Note this makes uwV = shape_factor * dz * uwL * uwW (the
            rounded-footprint shape factor is folded into uwV), so accurate
            underwater volume is nansum(uwV), not dz*uwL*uwW.
        area : float, optional
            Measured waterline footprint area A (m², plan-view, e.g. from a
            segmented iceberg polygon). Only used when volume_law='sulak'. 
        footprint_factor : float, optional
            Waterline footprint fill fraction (real plan area / L*W) used for the
            length-to-area proxy ``A = footprint_factor * L * W``. Must be in
            (0, 1]. Mutually exclusive with ``area``, which measures the same
            quantity directly. Default None uses FOOTPRINT_SHAPE_FACTOR (0.65,
            the mean of the three Sermilik observations spanning 0.58-0.70 -- a
            real per-berg spread, so this is the knob to vary in a sensitivity
            test when no measured footprint is available).

        Returns
        -------
        icebergs : xarray.Dataset
            Dataset containing iceberg geometry variables with depth coordinate Z:
            
            - depth : Depth coordinate values (m)
            - cross_area : Cross-sectional area at each depth layer (m²)
            - uwL : Underwater length at each depth layer (m)
            - uwW : Underwater width at each depth layer (m)
            - uwV : Underwater volume for each depth layer (m³)
        
        Notes
        -----
        The method applies different calculation approaches based on keel depth:
        
        **For keel_depth ≤ tabular (default 200m):**
        Uses empirical coefficients from Barker et al. 2004 (Table 5) to calculate
        cross-sectional areas as: CA = a * L + b, where a and b are depth-dependent
        coefficients, and L is the iceberg length.
        
        **For keel_depth > tabular:**
        Assumes a tabular (rectangular) shape with constant cross-section:
        CA = L * dz for each layer.
        
        The underwater width is derived from underwater length using the specified
        length-to-width ratio. Volume for each layer is calculated as:
        V = length * width * layer_thickness
        
        References
        ----------
        Barker, A., Sayed, M., & Carrieres, T. (2004). Determination of iceberg 
        draft, mass and cross-sectional areas. In Proceedings of the 14th 
        International Offshore and Polar Engineering Conference (pp. 1-8).
        
        Dowdeswell, J. A., et al. Size and shape characteristics of icebergs.
        
        Examples
        --------
        >>> iceberg = Iceberg(length=150, dz=5)
        >>> keel = iceberg.keeldepth(method='barker')
        >>> geometry = iceberg.barker_carea(keel, dz=5)
        >>> print(geometry.uwL)  # Underwater lengths at each depth
        >>> print(geometry.uwV.sum())  # Total underwater volume
        """
        
        keel_depth = np.array([keel_depth])
        L = np.array([self.length])
        
        # Ensure dz is a scalar for use in np.arange
        if isinstance(dz, np.ndarray):
            dz = float(dz.item())
        else:
            dz = float(dz)
        
        if keel_depth is None:
            keel_depth = self.keeldepth(L,'barker') # K = keeldepth(L,'mean');
            dz = 10
            LWratio = 1.62
        
        # table 5
        if dz == 10: # originally for dz=10 m layers
            a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
            
            
            
            a = np.array(a).reshape((len(a),1))
            
            b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
            b = -1 * (np.array(b).reshape((len(b),1)))
            
        elif dz == 5:
    
            a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
            a = np.array(a).reshape((len(a),1))
            
            b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
            b = -1 * (np.array(b).reshape((len(b),1)))
        
            # a_lin = a[9:,:]
            # b_lin = b[9:,:]
            # model = LinearRegression().fit(a_lin, b_lin)
            # r_sq = model.score(a_lin, b_lin)
            
            # mean = np.mean(np.diff(a_lin,axis=0))
            # a2 = np.arange(a_lin[-1][0],0,mean).reshape((-1,1))
            # b2 = model.predict(a2)
            
            # a_stack = np.vstack((a,a2[1:,:]))
            # b_stack = np.vstack((b,b2[1:,:]))
            
            
            aa = np.empty(a.T.shape)
            aa[0] = a[0]
            bb = np.empty(b.T.shape)
            bb[0] = b[0]
            
            for i in range(len(a)-1):
                aa[0,i+1] = np.nanmean(a[i:i+2,:])
                bb[0,i+1] = np.nanmean(b[i:i+2,:])
            
            # kz = keel_depth[0] # keel depth
            # kza = np.ceil(kz/dz) # layer index for keel depth
            # newa = np.empty((a.size*2,1)) #np.ceil(kz/dz) instead of 40?
            newa = np.empty((40,1)) #np.ceil(kz/dz) instead of 40?
            # if kza <= 40:    
            #     newa = np.empty((40,1)) #np.ceil(kz/dz) instead of 40?
            # elif kza > 40:
            #     newa = np.empty((int(kza),1)) #np.ceil(kz/dz) instead of 40?
    
            newa[:] = np.nan
            newb = newa.copy()
            
            newa[::2] = aa.T
            newa[1::2] = a
            
            newb[::2] = bb.T
            newb[1::2] = b
            
            a = newa/2
            b = newb/2
        
        a_s = const.SAIL_AREA_COEFFICIENT_A  # for sail area table 4 barker et al 2004
        b_s = const.SAIL_AREA_COEFFICIENT_B    
        
        
    
        
        
        
        # initialize arrays
        # icebergs.Z = dz:dz:500; icebergs.Z=icebergs.Z';
        # zlen = length(icebergs.Z);
        # temp = nan.*ones(zlen,length(L));  # 100 layers of 5-m each, so up to 500 m deep berg
        # temps = nan.*ones(1,length(L));  # sail area
        
        # Depth grid extends to at least the keel depth (rounded up to a whole dz
        # layer), with 600 m kept only as a floor so shallow bergs are unchanged.
        # Previously this was hard-capped at 600 m, which broke deeper icebergs.
        max_depth = max(600.0, np.ceil(np.max(keel_depth) / dz) * dz)
        z_coord_flat = np.arange(dz, max_depth + dz, dz) # deepest iceberg is defined here
        z_coord = z_coord_flat.reshape(len(z_coord_flat),1)
        depth_layers = xr.DataArray(data=z_coord, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="Z")
        zlen = len(depth_layers.Z)
        # temp = nan.*ones(zlen,length(L))
        # need to make L an array
        
        temp = np.nan * np.ones((zlen, len(L)))
        temps = np.nan * np.ones((1, len(L)))
        
        
        # K_l200 = keel_depth[keel_depth<200] # might cause an issue?
        K_ltab = np.where(keel_depth<=tabular)[0] # get indices of keel_depth < tabular
        # if(~isempty(ind))
        if K_ltab.size != 0: # check if empty
            for i in range(len(K_ltab)):

                kz = keel_depth[K_ltab[i]] # keel depth
                # dz_np = np.array([dz],dtype=np.float64)
                kza = np.ceil(kz/dz) # layer index for keel depth
                # kza = ceil(kz,dz) # layer index for keel depth
                
                # Convert kza to scalar if it's an array
                if isinstance(kza, np.ndarray):
                    kza = int(kza.item())
                else:
                    kza = int(kza)
                
                for nl in range(kza):
                    # Extract scalar values from 2D arrays a and b
                    temp[nl,K_ltab[i]] = a[nl, 0] * L[K_ltab[i]] + b[nl, 0]
                    
            temps[K_ltab] = a_s * L[K_ltab] + b_s
            
            if L < const.SAIL_AREA_LENGTH_THRESHOLD:
                temps[L < const.SAIL_AREA_LENGTH_THRESHOLD] = const.SAIL_AREA_QUADRATIC_COEFF * np.power(L[L < const.SAIL_AREA_LENGTH_THRESHOLD], 2)  # fix for L<65, barker 2004
        
        
        # then do icebergs D>200 for tabular
        K_gtab = np.where(keel_depth>tabular)[0]
        if K_gtab.size != 0:
            for i in range(len(K_gtab)):
                
                kz = keel_depth[i] # keel depth
                kza = np.ceil(kz/dz) # layer index for keel depth
                
                # Convert kza to scalar if it's an array
                if isinstance(kza, np.ndarray):
                    kza = int(kza.item())
                else:
                    kza = int(kza)
                
                for nl in range(kza):
                    # temp[nl,i] = a[nl] * L[K_g200[i]] + b[nl]
                    temp[nl,i] = L[K_gtab[i]] * dz
            
            temps[K_gtab] = const.TABULAR_ICEBERG_COEFFICIENT * L[K_gtab] * keel_depth[K_gtab]
            
        
        cross_area = xr.DataArray(data=temp, coords = {"Z":z_coord_flat}, dims=["Z","X"], name="cross_area")
        # icebergs.uwL = temp./dz; 
        length_layers = xr.DataArray(data=temp/dz, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwL")
        
        # now use L/W ratio of 1.62:1 (from Dowdeswell et al.) to get widths I wonder if I can just get widths from Sid's data??
        widths = length_layers.values / LWratio 
        width_layers = xr.DataArray(data = widths, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwW")
        
        dznew = dz * np.ones(length_layers.values.shape)
        
        vol = dznew * length_layers.values * width_layers.values
        volume = xr.DataArray(data=vol, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwV")
        
        # I am ASSUMING everything is the same size. NEED TO CHECK when I get things running
        icebergs = xr.Dataset(data_vars={'depth':depth_layers,
                                         'cross_area':cross_area,
                                         'uwL':length_layers,
                                         'uwW':width_layers,
                                         'uwV':volume},
                              coords = {'Z': z_coord_flat}
                              )

        # Optional volume calibration: taper the underwater cross-section so the
        # total volume matches the empirical waterline-footprint-area to volume
        # relation (Sulak et al. 2017 / Schild et al. 2021)
        ff_proxy = self._resolve_footprint_factor(footprint_factor, area)

        # How uwV was actually produced, recorded per-berg on the variable itself.
        # Stored under 'computation' rather than 'description' so the static
        # VARIABLE_ATTRS update in _assign_variable_attrs cannot clobber it.
        uwV_computation = (
            "dz * uwL * uwW -- uncalibrated rectangular prism, no volume_law "
            "applied. pass volume_law='sulak' "
            "to calibrate against V = c*A^x.")

        if volume_law == 'sulak':
            L_wl = float(np.asarray(L).ravel()[0])
            W_wl = L_wl / LWratio
            kd = float(np.asarray(keel_depth).ravel()[0])
            # Waterline footprint area A fed into Sulak's V = c*A^x.
            if area is not None:
                A_wl = float(area)
                if A_wl <= 0:
                    raise ValueError(f"area must be positive, got {area}")
            else:
                A_wl = ff_proxy * L_wl * W_wl
            # effective fill fraction of the L x W bounding box for this berg;
            # equals the resolved footprint_factor when area is not supplied.
            shape_factor = A_wl / (L_wl * W_wl)
            a_src = ('measured' if area is not None
                     else f'from length (fill {ff_proxy:.2f})')
            V_uw_target = (const.AREA_VOLUME_COEFFICIENT
                           * A_wl ** const.AREA_VOLUME_EXPONENT
                           * const.DENSITY_RATIO_ICE_TO_WATER)
            # rounded-footprint prism volume (no taper): the ceiling to taper from
            prism = shape_factor * float(np.nansum(icebergs['uwV'].values))
            if wall_slope is not None:
                # Surface geometry from the OBSERVED wall undercut angle (Schild et al.
                # 2024, 3-23 deg) 
                z = icebergs['Z'].values.astype(float)[:, None]
                recede = 2.0 * np.tan(np.radians(wall_slope)) * z   # both faces of each dim
                uwL0 = icebergs['uwL'].values
                uwW0 = icebergs['uwW'].values
                with np.errstate(invalid='ignore', divide='ignore'):
                    taperL = np.where(uwL0 > 0, np.clip((uwL0 - recede) / uwL0, 0.0, 1.0), 1.0)
                    taperW = np.where(uwW0 > 0, np.clip((uwW0 - recede) / uwW0, 0.0, 1.0), 1.0)
                icebergs['uwL'] = icebergs['uwL'] * taperL
                icebergs['uwW'] = icebergs['uwW'] * taperW
                icebergs['cross_area'] = icebergs['cross_area'] * taperL
                icebergs['uwV'] = icebergs['uwV'] * shape_factor * taperL * taperW
                # icebergs['uwV'] = icebergs['uwV'] * taperL * taperW

                tot = float(np.nansum(icebergs['uwV'].values))
                if tot > 0:
                    icebergs['uwV'] = icebergs['uwV'] * (V_uw_target / tot)
                uwV_computation = (
                    f"shape_factor * dz * uwL * uwW (shape_factor = "
                    f"{shape_factor:.4f}), rescaled so nansum(uwV) = c*A^x. The "
                    f"uwL/uwW taper is set by wall_slope, not by the volume, so uwV "
                    f"is NOT recoverable from dz*uwL*uwW.")
                icebergs.attrs['volume_law'] = (
                    f"V=c*A^x (c={const.AREA_VOLUME_COEFFICIENT}, "
                    f"x={const.AREA_VOLUME_EXPONENT}, Sulak 2017/Schild 2021); "
                    f"A={a_src} "
                    f"({A_wl:.0f} m^2); surface geometry from wall_slope={wall_slope} deg "
                    f"undercut (Schild 2024), uwV scaled to preserve calibrated volume")
            elif prism > 0:
                # Solve for the linear-taper param a (cross-section scales
                # 1 - a*z/keel, a in [0, 1]) so nansum(uwV) lands on c*A^x.
                g_req = V_uw_target / prism
                z = icebergs['Z'].values.astype(float)
                base = shape_factor * icebergs['uwV'].values

                def _tapered_volume(a_try):
                    t = np.clip(1.0 - a_try * z / kd, 1.0 - a_try, 1.0)[:, None]
                    return float(np.nansum(base * t ** 2))

                # monotone decreasing in a: v_lo (straight walls) >= v_hi (pyramid)
                v_lo, v_hi = _tapered_volume(0.0), _tapered_volume(1.0)
                if v_lo <= V_uw_target:
                    a = 0.0        # even straight walls cannot hold c*A^x
                elif v_hi >= V_uw_target:
                    a = 1.0        # even a keel pinched to a point overshoots
                else:
                    lo, hi = 0.0, 1.0
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if _tapered_volume(mid) > V_uw_target:
                            lo = mid
                        else:
                            hi = mid
                    a = 0.5 * (lo + hi)
                taper = np.clip(1.0 - a * z / kd, 1.0 - a, 1.0)[:, None]
                icebergs['uwL'] = icebergs['uwL'] * taper
                icebergs['uwW'] = icebergs['uwW'] * taper
                icebergs['cross_area'] = icebergs['cross_area'] * taper
                icebergs['uwV'] = (icebergs['uwV'] * shape_factor
                                   * taper ** 2)
                # A linear taper spans only [v_hi, v_lo] of the prism volume: a=0 is a
                # straight-sided box, a=1 a pyramid pinched to zero at the keel.
                achieved = float(np.nansum(icebergs['uwV'].values)) / V_uw_target
                undershoot = v_lo < V_uw_target
                saturated = undershoot or v_hi > V_uw_target
                if saturated:
                    end = ("a=0 (no taper, straight walls)" if undershoot
                           else "a=1 (pyramid, keel pinched to a point)")
                    hint = ("The keel is too shallow to hold c*A^x; small bergs "
                            "(L < ~150 m with keel_method='barker') hit this end "
                            "stop, and raising footprint_factor makes it worse "
                            "(target grows as ff^x, prism only as ff)."
                            if undershoot else
                            "The keel is too deep for c*A^x; near-equant bergs hit "
                            "this end stop.")
                    warnings.warn(
                        f"taper solve saturated at {end} for L={L_wl:.0f} m, "
                        f"keel={kd:.0f} m: V=c*A^x wants {g_req:.2f} x the rounded "
                        f"L x W prism volume, but a linear taper only spans "
                        f"[{v_hi / v_lo:.2f}, 1.00]. nansum(uwV) is {achieved:.2f} x "
                        f"the calibrated target. {hint}",
                        stacklevel=3)
                uwV_computation = (
                    f"shape_factor * dz * uwL * uwW (shape_factor = "
                    f"{shape_factor:.4f}), uwL/uwW already tapered. Recoverable as "
                    f"footprint_factor * dz * uwL * uwW.")
                icebergs.attrs['volume_law'] = (
                    f"V=c*A^x (c={const.AREA_VOLUME_COEFFICIENT}, "
                    f"x={const.AREA_VOLUME_EXPONENT}, Sulak 2017/Schild 2021); "
                    f"A={a_src} "
                    f"({A_wl:.0f} m^2); linear keel taper to {1.0 - a:.2f} of waterline"
                    + (f"; TAPER SATURATED, uwV is {achieved:.2f} x the c*A^x target"
                       if saturated else ""))
        elif volume_law is not None:
            raise ValueError(
                f"Unknown volume_law {volume_law!r}; expected 'sulak' or None.")

        icebergs = self._assign_variable_attrs(icebergs)
        icebergs['uwV'].attrs['computation'] = uwV_computation
        icebergs = self._assign_global_attrs(icebergs)
        return icebergs

    def init_iceberg_size(self, stability_method='equal', quiet=True,
                          keel_method='barker', volume_law=None, area=None,
                          width=None, roughness_factor=None, wall_slope=None,
                          footprint_factor=None, perimeter=None):
        """
        Initialize complete iceberg geometry and ensure hydrostatic stability.
        
        
        Parameters
        ----------
        stability_method : {'equal', 'keel'}, optional
            Method to use for stabilizing unstable icebergs. Default is 'equal'.
            
            - 'equal' : Adjusts the length-to-width ratio to make the iceberg wider,
              bringing W/H ratio to the stability threshold while maintaining keel depth.
            - 'keel' : Reduces the keel depth to decrease total height, bringing W/H 
              ratio to the stability threshold while maintaining L:W ratio.
              
        quiet : bool, optional
            If False, prints diagnostic messages when stability adjustments are made.
            Default is True (suppresses output).
        keel_method : str, optional
            Keel depth method passed to :meth:`keeldepth` (default 'barker').
            Use 'schild' for the Sermilik large-iceberg calibration (K = L/1.98).
        volume_law : {None, 'sulak'}, optional
            If 'sulak', V = 6.0*A^1.31 (Sulak et al. 2017 / Schild et al. 2021)
        area : float, optional
            Measured waterline footprint area A
        width : float, optional
            Observed waterline width (m). 
        roughness_factor : float, optional
            Multiplier applied to the smooth-geometry wetted surface area to
            account for surface roughness
            Must be positive.
        wall_slope : float, optional
            Observed side-wall undercut angle in degrees (Schild et al. 2024
            measured 3-23 deg). 
        footprint_factor : float, optional
            Waterline footprint fill fraction 
        perimeter : float, optional
            Measured waterline outline perimeter (m, e.g. a segmented polygon's
            ``geom.length``). Sets ``perimeter_factor = perimeter / (2*(L+W))``



        Returns
        -------
        ice : xarray.Dataset
            Complete iceberg geometry dataset containing all variables from barker_carea
            plus additional derived parameters:
            
            **From barker_carea:**
            - depth : Depth coordinate (m)
            - cross_area : Cross-sectional area by depth (m²)
            - uwL : Underwater length by depth (m)
            - uwW : Underwater width by depth (m)
            - uwV : Underwater volume by depth (m³)
            
            **Additional variables:**
            - totalV : Total iceberg volume, underwater + sail (m³)
            - sailV : Above-water (sail) volume (m³)
            - W : Waterline width (m)
            - freeB : Freeboard height above waterline (m)
            - L : Iceberg length (m)
            - keel : Keel depth below waterline (m)
            - TH : Total thickness/height (keel + freeboard) (m)
            - keeli : Index of deepest layer (dimensionless)
            - dz : Layer thickness used (m)
            - dzk : Partial layer thickness at keel (m)
        
        Raises
        ------
        Exception
            If stability_method='equal' is unable to achieve stability (W/H < 0.92)
            after adjusting the length-to-width ratio.
        
        References
        ----------
        Wagner, T. J., et al. (2017). The "footloose" mechanism: Iceberg decay from
        hydrostatic stresses. Geophysical Research Letters, 44(2), 2017GL072671.
        
        Barker, A., Sayed, M., & Carrieres, T. (2004). Determination of iceberg 
        draft, mass and cross-sectional areas.
        
        Examples
        --------
        >>> # Create a stable iceberg with default parameters
        >>> iceberg = Iceberg(length=200, dz=5)
        >>> ice_geometry = iceberg.init_iceberg_size(stability_method='equal')
        >>> print(f"Keel depth: {ice_geometry.keel.values:.1f} m")
        >>> print(f"Freeboard: {ice_geometry.freeB.values:.1f} m")
        >>> print(f"Stability ratio (W/H): {ice_geometry.W.values/ice_geometry.TH.values:.3f}")
        
        >>> # Check if stability adjustment was needed
        >>> iceberg = Iceberg(length=500, dz=10)
        >>> ice_geometry = iceberg.init_iceberg_size(stability_method='keel', quiet=False)
        """
        
        # Ensure self.dz is scalar
        dz_val = float(self.dz.item()) if isinstance(self.dz, np.ndarray) else float(self.dz)

        if area is not None:
            if volume_law != 'sulak':
                raise ValueError(
                    "area is only used with volume_law='sulak'; "
                    f"got volume_law={volume_law!r}")
            if area <= 0:
                raise ValueError(f"area must be positive, got {area}")

        if wall_slope is not None:
            if volume_law != 'sulak':
                raise ValueError(
                    "wall_slope requires volume_law='sulak' (it sets the surface "
                    f"geometry while preserving the Sulak volume); got volume_law={volume_law!r}")
            if wall_slope < 0:
                raise ValueError(f"wall_slope must be >= 0 degrees, got {wall_slope}")

        # Waterline width: use the observed width if given, otherwise the
        # Dowdeswell default L:W ratio. Width does not affect the calibrated
        # total volume (area-driven)
        if width is not None:
            if width <= 0:
                raise ValueError(f"width must be positive, got {width}")
            lw_ratio_input = self.length / float(width)
        else:
            lw_ratio_input = const.DEFAULT_LENGTH_TO_WIDTH_RATIO

        # Surface roughness enhancement applied to the wetted area (default is the
        # measured value; pass 1.0 for pure smooth geometry).
        if roughness_factor is None:
            rf = const.SURFACE_ROUGHNESS_FACTOR
        else:
            if roughness_factor <= 0:
                raise ValueError(f"roughness_factor must be positive, got {roughness_factor}")
            rf = float(roughness_factor)

        # Effective waterline footprint fill fraction (real footprint / L*W). Uses
        # the MEASURED `area` when supplied (area / (L*W)), otherwise the caller's
        # `footprint_factor` or the FOOTPRINT_SHAPE_FACTOR default.
        _W_wl = self.length / lw_ratio_input
        ff_proxy = self._resolve_footprint_factor(footprint_factor, area)
        if area is not None:
            ff_effective = float(area) / (self.length * _W_wl)
        else:
            ff_effective = ff_proxy

        # Waterline footprint area used to convert sail volume -> freeboard height.
        def _footprint_area(width):
            if volume_law == 'sulak':
                if area is not None:
                    return float(area)
                return ff_proxy * self.length * width
            return self.length * width

        keel_depth = self.keeldepth(method=keel_method)
        
        # now get underwater shape, based on Barker for K<200, tabular for K>200, and
        ice = self.barker_carea(keel_depth, dz_val, LWratio=lw_ratio_input, volume_law=volume_law, area=area, wall_slope=wall_slope, footprint_factor=footprint_factor) # this gives you uwL, uwW, uwV, uwM, and vector Z down to keel depth

        # from underwater volume, calculate above water volume
        density_ratio = const.DENSITY_RATIO_ICE_TO_WATER  # ratio of ice density to water density

        total_volume = (1/density_ratio) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
        sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune

        waterline_width = self.length / lw_ratio_input
        freeB = sail_volume / _footprint_area(waterline_width) # Freeboard height
        # length = L.copy()
        thickness = keel_depth + freeB # total thickness
        deepest_keel = np.ceil(keel_depth/dz_val) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
        # dz = dzS
        dzk = -1*((deepest_keel - 1) * dz_val - keel_depth) #
        
        # check if stable
        stable_check = waterline_width / thickness[0]
        
        if stable_check > self.STABILITY_THRESHOLD:
                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(self.length),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz_val, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
                
                pf = self._resolve_perimeter_factor(perimeter, self.length, waterline_width)
                ice = self._add_surface_area(ice, dz_val, rf, ff_effective, pf)
                ice = self._assign_variable_attrs(ice)
                return ice
        
        
        if stable_check < self.STABILITY_THRESHOLD:
            # Not sure when to use either? MATLAB code has if(0) and if(1) for 'keel' and 'equal'
            if stability_method == 'keel':
                # change keeldepth to be shallower
                # if quiet == False:
                #     print(f'Fixing keel depth for L = {L} m size class')
                    
                
                diff_thick_width = thickness - waterline_width # Get stable thickness
                keel_new = keel_depth - density_ratio * diff_thick_width # change by percent of difference
                
                ice = self.barker_carea(keel_new, dz_val, LWratio=lw_ratio_input, volume_law=volume_law, area=area, wall_slope=wall_slope, footprint_factor=footprint_factor)
                total_volume = (1/density_ratio) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
                sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
                waterline_width = self.length / lw_ratio_input
                freeB = sail_volume / _footprint_area(waterline_width) # Freeboard height
                # length = L.copy()
                thickness = keel_depth + freeB # total thickness
                deepest_keel = np.ceil(keel_depth/dz_val) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
                # dz = dzS
                dzk = -1*((deepest_keel - 1) * dz_val - keel_depth) #
                stability = waterline_width/thickness[0]

                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(self.length),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz_val, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')

                if stability < self.STABILITY_THRESHOLD:
                    raise Exception("Still unstable, check W/H ratios")

                pf = self._resolve_perimeter_factor(perimeter, self.length, waterline_width)
                ice = self._add_surface_area(ice, dz_val, rf, ff_effective, pf)
                ice = self._assign_variable_attrs(ice)
                return ice

            elif stability_method == 'equal':
                # change W to equal L, recalculate volumes
                if not quiet:
                    print(f'Fixing width to equal L, for L = {self.length} m size class')
                # use L:W ratio of to make stable, set so L:W makes EC=EC_thresh
                
                width_temporary = self.STABILITY_THRESHOLD * thickness[0]
                lw_ratio = np.floor((100*self.length)/width_temporary)/100 # round down to hundredth place
                
                ice = self.barker_carea(keel_depth, dz_val, LWratio=lw_ratio, volume_law=volume_law, area=area, wall_slope=wall_slope, footprint_factor=footprint_factor)
                
                total_volume = (1/density_ratio) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
                sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
                waterline_width = self.length / lw_ratio 
                freeB = sail_volume / _footprint_area(waterline_width) # Freeboard height
                # length = L.copy()
                thickness = keel_depth + freeB # total thickness
                deepest_keel = np.ceil(keel_depth/dz_val) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
                # dz = dzS
                dzk = -1*((deepest_keel - 1) * dz_val - keel_depth) #
    
                
                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(self.length),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz_val, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
                EC = ice.W/ice.TH
                
                if EC < self.STABILITY_THRESHOLD:
                    raise Exception("Still unstable, check W/H ratios")

                pf = self._resolve_perimeter_factor(perimeter, self.length, waterline_width)
                ice = self._add_surface_area(ice, dz_val, rf, ff_effective, pf)
                ice = self._assign_variable_attrs(ice)
                return ice

    def plot_iceberg_shape(self, ice=None, dimension='length', ax=None,
                           figsize=(6, 8)):
        """
        Plot a vertical cross-section of the iceberg.


        Parameters
        ----------
        ice : xarray.Dataset, optional
            A dataset produced by :meth:`init_iceberg_size`. If None (default),
            geometry is computed by calling ``self.init_iceberg_size()``.
        dimension : {'length', 'width'}, optional
            Which horizontal dimension to section through. 'length' uses the
            underwater length (``uwL``); 'width' uses the underwater width
            (``uwW``). Default is 'length'.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, a new figure and axes are created.
        figsize : tuple of float, optional
            Figure size in inches, used only when ``ax`` is None. Default (6, 8).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            The figure and axes containing the plot.

        Examples
        --------
        >>> berg = Iceberg(length=300, dz=5)
        >>> fig, ax = berg.plot_iceberg_shape()
        >>> fig.savefig('iceberg.png', dpi=150, bbox_inches='tight')
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if ice is None:
            ice = self.init_iceberg_size()

        if dimension == 'length':
            profile = np.asarray(ice.uwL.values, dtype=float).ravel()
            waterline = float(ice.L)
            dim_label = 'Length'
        elif dimension == 'width':
            profile = np.asarray(ice.uwW.values, dtype=float).ravel()
            waterline = float(ice.W)
            dim_label = 'Width'
        else:
            raise ValueError("dimension must be 'length' or 'width'")

        z = np.asarray(ice.Z.values, dtype=float)
        keel = float(ice.keel)
        freeB = float(ice.freeB)

        # keep only valid underwater layers, and anchor the profile at the
        # waterline (z=0) with the full waterline dimension
        valid = ~np.isnan(profile)
        zt = np.concatenate(([0.0], z[valid]))
        half = np.concatenate(([waterline / 2.0], profile[valid] / 2.0))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # muted palette: light sail ice, deeper-blue keel ice, faint water
        sail_color = '#dbeafe'
        keel_color = '#93c5e8'
        edge_color = '#1e3a5f'
        water_color = '#eef5fb'
        waterline_color = '#2563a6'

        xmax = np.nanmax(half) * 1.25

        # water region below the waterline
        ax.axhspan(-keel * 1.08, 0.0, color=water_color, zorder=0)
        # underwater keel silhouette (depth as negative height)
        ax.fill_betweenx(-zt, -half, half, facecolor=keel_color,
                         edgecolor=edge_color, linewidth=1.2, zorder=2)
        # above-water sail block
        ax.add_patch(Rectangle((-waterline / 2.0, 0.0), waterline, freeB,
                               facecolor=sail_color, edgecolor=edge_color,
                               linewidth=1.2, zorder=2))
        # waterline
        ax.axhline(0.0, color=waterline_color, linestyle='--', linewidth=1.0,
                   zorder=3)

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-keel * 1.08, max(freeB * 2.5, keel * 0.08))
        ax.set_xlabel(f'{dim_label} (m)')
        ax.set_ylabel('Height relative to waterline (m)')
        ax.set_title(f'Iceberg cross-section  (L = {float(ice.L):.0f} m, '
                     f'keel = {keel:.0f} m)')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis='y', color='0.9', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_aspect('equal', adjustable='box')

        return fig, ax

    def plot_iceberg_3d(self, ice=None, ax=None, figsize=(7, 8),
                        elev=18, azim=-60):
        """
        Render a 3-D solid model of the iceberg.

        Parameters
        ----------
        ice : xarray.Dataset, optional
            A dataset produced by :meth:`init_iceberg_size`. If None (default),
            geometry is computed by calling ``self.init_iceberg_size()``.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            A 3-D axes to draw into. If None, a new figure and 3-D axes are made.
        figsize : tuple of float, optional
            Figure size in inches, used only when ``ax`` is None. Default (7, 8).
        elev, azim : float, optional
            Initial elevation and azimuth viewing angles in degrees.

        Returns
        -------
        fig, ax : matplotlib Figure and 3-D Axes
            The figure and axes containing the plot.

        Examples
        --------
        >>> berg = Iceberg(length=300, dz=5)
        >>> fig, ax = berg.plot_iceberg_3d()
        >>> fig.savefig('iceberg_3d.png', dpi=150, bbox_inches='tight')
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ice is None:
            ice = self.init_iceberg_size()

        uwL = np.asarray(ice.uwL.values, dtype=float).ravel()
        uwW = np.asarray(ice.uwW.values, dtype=float).ravel()
        z = np.asarray(ice.Z.values, dtype=float)
        L = float(ice.L)
        W = float(ice.W)
        keel = float(ice.keel)
        freeB = float(ice.freeB)

        valid = ~np.isnan(uwL) & ~np.isnan(uwW)

        # cross-sections from sail top down to the keel: (half_x, half_y, height)
        sections = [
            (L / 2.0, W / 2.0, freeB),   # sail top
            (L / 2.0, W / 2.0, 0.0),     # waterline
        ]
        for hx, hy, zc in zip(uwL[valid] / 2.0, uwW[valid] / 2.0, -z[valid]):
            sections.append((hx, hy, zc))

        def corners(sec):
            hx, hy, zc = sec
            return [(hx, hy, zc), (hx, -hy, zc), (-hx, -hy, zc), (-hx, hy, zc)]

        sail_color = '#dbeafe'
        keel_color = '#93c5e8'
        edge_color = '#1e3a5f'

        faces, face_colors = [], []
        # top cap (sail top)
        faces.append(corners(sections[0]))
        face_colors.append(sail_color)
        # side walls: 4 quads between each pair of consecutive cross-sections
        for i in range(len(sections) - 1):
            c0, c1 = corners(sections[i]), corners(sections[i + 1])
            col = sail_color if i == 0 else keel_color
            for e in range(4):
                faces.append([c0[e], c0[(e + 1) % 4],
                              c1[(e + 1) % 4], c1[e]])
                face_colors.append(col)
        # bottom cap (keel)
        faces.append(corners(sections[-1]))
        face_colors.append(keel_color)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        poly = Poly3DCollection(faces, facecolors=face_colors,
                                edgecolors=edge_color, linewidths=0.25,
                                alpha=0.95)
        ax.add_collection3d(poly)

        hx_max = max(s[0] for s in sections)
        hy_max = max(s[1] for s in sections)
        ax.set_xlim(-hx_max, hx_max)
        ax.set_ylim(-hy_max, hy_max)
        ax.set_zlim(-keel * 1.02, freeB)
        ax.set_box_aspect((2 * hx_max, 2 * hy_max, keel + freeB))
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_zlabel('Height rel. to waterline (m)')
        ax.set_title(f'Iceberg 3-D  (L = {L:.0f} m, keel = {keel:.0f} m)')
        ax.view_init(elev=elev, azim=azim)

        return fig, ax