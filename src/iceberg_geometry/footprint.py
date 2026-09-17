"""Footprint metrics from segmented iceberg polygons.

Examples
--------
>>> import geopandas as gpd
>>> from iceberg_geometry import Iceberg, footprint_metrics
>>> bergs = gpd.read_file("2023-07-27_all_merged_dims.gpkg")
>>> m = footprint_metrics(bergs.geometry.iloc[0])
>>> ice = Iceberg(length=m["length"], dz=5).init_iceberg_size(
...     keel_method="schild", volume_law="sulak",
...     area=m["area"], perimeter=m["perimeter"])
"""

# might get rid of this entire file
import numpy as np

# Simplification tolerance (m) defining the smooth-outline convention for
# perimeter. 10 m sits in the same regime as the drone convex hulls the
# roughness factor was calibrated against (isoperimetric quotient ~0.83,
# perimeter / bounding-rectangle perimeter ~0.85), while leaving plan area
# unchanged to within 0.1% on the Helheim segmented polygons.
DEFAULT_SIMPLIFY_TOLERANCE = 10.0

__all__ = ["DEFAULT_SIMPLIFY_TOLERANCE", "footprint_metrics", "footprint_table"]


def _hull_coords(geom):
    """(N, 2) array of convex-hull vertices, for any polygon-like geometry."""
    hull = geom.convex_hull
    try:
        coords = np.asarray(hull.exterior.coords, dtype=float)[:, :2]
    except AttributeError as exc:  # degenerate hull: point or line
        raise ValueError(
            f"geometry has no polygonal convex hull (got {hull.geom_type}); "
            "it is probably degenerate or empty") from exc
    return coords


def max_caliper(geom):
    """Longest distance between any two convex-hull vertices (m).

    This is the "surface length" convention: the longest straight-line extent
    of the footprint, which is what the Barker/Schild keel relations expect.
    """
    coords = _hull_coords(geom)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return float(d.max())


def footprint_metrics(geom, simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
                      lw_ratio=None):
    """Measure length, area and perimeter of one footprint polygon.

    Parameters
    ----------
    geom : polygon-like
        Plan-view footprint in a projected (metre) CRS -- NOT lat/lon, or area
        and perimeter are meaningless.
    simplify_tolerance : float, optional
        Douglas-Peucker tolerance (m) applied before measuring perimeter, which
        fixes the smoothing convention (see module docstring). Default
        DEFAULT_SIMPLIFY_TOLERANCE (10 m). Pass 0 for the raw perimeter, which
        will double-count roughness.
    lw_ratio : float, optional
        Length-to-width ratio used only for the reported ``footprint_factor`` /
        ``perimeter_factor`` diagnostics. Default None uses the model's
        DEFAULT_LENGTH_TO_WIDTH_RATIO. Ignored if ``width`` is derivable.

    Returns
    -------
    dict
        ``length`` (max caliper, m), ``width`` (min-area-rectangle width, m),
        ``area`` (m^2), ``perimeter`` (m, simplified), ``perimeter_raw`` (m),
        ``footprint_factor`` and ``perimeter_factor`` (both relative to the
        length x width bounding rectangle), ``isoperimetric_quotient``
        (4*pi*A/P^2 on the simplified outline; 1.0 is a circle, 0.74 a 1.62
        rectangle -- much below ~0.7 means the outline is still ragged) and
        ``simplify_tolerance``.

    Notes
    -----
    ``length`` comes from the geometry itself rather than any stored column. In
    the Helheim gpkg the stored ``max_dim`` disagrees with the geometry for some
    rows (implied area fill ranges 0.038 to 1.07, and >1 is impossible for a
    footprint inside its own bounding rectangle), and ``max_dim`` also sets keel
    depth -- so deriving it here keeps length, area and perimeter consistent.
    """
    from . import constants as const

    if simplify_tolerance < 0:
        raise ValueError(
            f"simplify_tolerance must be >= 0, got {simplify_tolerance}")

    area = float(geom.area)
    perimeter_raw = float(geom.length)
    if area <= 0 or perimeter_raw <= 0:
        raise ValueError("geometry has non-positive area or perimeter")

    smoothed = geom if simplify_tolerance == 0 else geom.simplify(simplify_tolerance)
    perimeter = float(smoothed.length)
    if perimeter <= 0:  # over-simplified into degeneracy
        raise ValueError(
            f"simplify_tolerance={simplify_tolerance} collapsed the outline; "
            "use a smaller tolerance")

    length = max_caliper(geom)
    width = _min_area_rect_width(geom, length)
    if width is None:
        ratio = (const.DEFAULT_LENGTH_TO_WIDTH_RATIO if lw_ratio is None
                 else float(lw_ratio))
        width = length / ratio

    rect_area = length * width
    rect_perimeter = 2.0 * (length + width)
    return {
        "length": length,
        "width": width,
        "area": area,
        "perimeter": perimeter,
        "perimeter_raw": perimeter_raw,
        "footprint_factor": area / rect_area,
        "perimeter_factor": perimeter / rect_perimeter,
        "isoperimetric_quotient": 4.0 * np.pi * area / perimeter ** 2,
        "simplify_tolerance": float(simplify_tolerance),
    }


def _min_area_rect_width(geom, length):
    """Short side of the minimum-area bounding rectangle (m), or None.

    Rotating calipers over the convex hull. The model's L:W default (1.62) is
    wrong for real bergs -- the 2017 drone clouds measure 1.22 (Iceberg A) and
    1.41 (Iceberg B) -- so measuring the width is preferable to assuming it.
    """
    coords = _hull_coords(geom)
    if len(coords) < 3:
        return None
    best = None
    for i in range(len(coords)):
        edge = coords[(i + 1) % len(coords)] - coords[i]
        norm = np.hypot(*edge)
        if norm == 0:
            continue
        u = edge / norm
        v = np.array([-u[1], u[0]])
        pu = coords @ u
        pv = coords @ v
        side_u = pu.max() - pu.min()
        side_v = pv.max() - pv.min()
        area = side_u * side_v
        if best is None or area < best[0]:
            best = (area, min(side_u, side_v))
    if best is None:
        return None
    # Guard against a hull so thin the caliper length and rect disagree badly.
    return min(best[1], length)


def footprint_table(geoms, simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
                    skip_errors=False):
    """Measure many footprints; returns a dict of 1-D numpy arrays.

    Column-oriented so it drops straight into a DataFrame::

        import pandas as pd
        df = pd.DataFrame(footprint_table(bergs.geometry))

    Parameters
    ----------
    geoms : iterable of polygon-like
        E.g. a GeoDataFrame's ``.geometry``.
    simplify_tolerance : float, optional
        Passed to :func:`footprint_metrics`.
    skip_errors : bool, optional
        If True, degenerate geometries yield NaN rows instead of raising.
        Default False.

    Returns
    -------
    dict of str -> numpy.ndarray
        Same keys as :func:`footprint_metrics`, one entry per geometry.
    """
    rows = []
    keys = None
    for geom in geoms:
        try:
            row = footprint_metrics(geom, simplify_tolerance=simplify_tolerance)
        except (ValueError, AttributeError):
            if not skip_errors:
                raise
            row = None
        if row is not None and keys is None:
            keys = list(row)
        rows.append(row)
    if keys is None:
        raise ValueError("no measurable geometries")
    return {k: np.array([np.nan if r is None else r[k] for r in rows], dtype=float)
            for k in keys}
