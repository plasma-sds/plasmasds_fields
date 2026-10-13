"""
Density allocation on W7-X flux surfaces and conversion of the surface
points to scattered and regular 2-D density fields.
"""
import copy
import warnings
import numpy as np
from . import poincare as pc

def add_density(flt, density_function, keep_unusable=False):
    """
    Assign densities to the surfaces from a radial profile.

    Each point-type group of a surface gets the density of the profile at the
    group's radius, the R of its polynomial fit at Z = 0 (see
    :func:`w7x.poincare.label_surfaces`). Unlabelled surfaces are labelled
    first. Point groups without a usable density are then removed, so the
    result can be passed directly to :func:`extract_points`.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Either a list of :class:`~w7x.poincare.FluxSurface` instances or an
        object exposing ``poincare_res.surfs``.
    density_function : callable
        Returns the density [m^-3] at a radius R [m], e.g. a
        :class:`tools.interpolate.ProfileInterpolator1D` instance (pass the
        object itself) or a ``scipy.interpolate.interp1d`` interpolator. Note
        that ``ProfileInterpolator1D`` extrapolates linearly outside its
        position range, so surfaces outside the profile get extrapolated
        densities.
    keep_unusable : bool, default False
        If False, point groups whose density is not positive (zero, negative or
        NaN) are removed, and surfaces left without points are dropped. If
        True, all surfaces and points are returned, with only the densities
        assigned.

    Returns
    -------
    list of FluxSurface
        The surfaces with assigned densities. The densities are written onto
        the input surfaces; surfaces that lose points are copied before
        trimming, so the input list keeps all its points.

    Raises
    ------
    TypeError
        If ``flt`` is neither a list of FluxSurface instances nor an object
        exposing ``poincare_res.surfs``, or if an element is not a FluxSurface,
        or if ``density_function`` is not callable.

    Warns
    -----
    UserWarning
        If some surfaces are not labelled yet;
        :func:`w7x.poincare.label_surfaces` is then run with its default
        parameters.
    """
    
    pc._check_surfaces(flt)
    
    if not callable(density_function):
        raise TypeError(
            f"density_function must be callable, "
            f"got {type(density_function).__name__}.")
    
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs
    
    if any(s.errors == [None] for s in surfs):
        warnings.warn(
            "Some surfaces are not labelled; running label_surfaces() "
            "first. Call label_surfaces() yourself to control its "
            "parameters.")
        surfs = pc.label_surfaces(surfs)
    
    usable = list()
    
    for surf in surfs:
        for i in range(len(surf.errors)):
            if surf.errors[i] is None: continue    # no points of this type
            dens = density_function(surf.coeffs[i, -1])
            surf.update_density(float(dens), ind=i)
        
        if keep_unusable:
            usable.append(surf)
            continue
        
        # Keep only the point groups with a usable (positive) density
        mask = np.zeros(surf.n, dtype=bool)
        for i, dens in enumerate(surf.density):
            if dens > 0:
                mask = mask | (surf.points.point_type == i)
        
        if np.all(mask):
            usable.append(surf)
        elif np.any(mask):
            surf = copy.deepcopy(surf)
            surf.filter_points(mask)
            usable.append(surf)
                
    return usable

def filter_surfaces_by_density(flt, density_range=None):
    """
    Filter flux surfaces and their points by their assigned density.

    Keeps only the points whose point-type group has a density within
    ``density_range``; surfaces left without points are dropped. Points without
    a usable density are already removed by :func:`add_density`.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Surfaces with assigned densities (see :func:`add_density`).
    density_range : sequence of float, default None
        Two-element ``[min, max]`` interval (inclusive) on the density of each
        point-type group [m^-3]. If None, all surfaces are kept.

    Returns
    -------
    list of FluxSurface
        Copies of the surfaces that keep at least one point, containing only
        the points within ``density_range``. The input surfaces are not
        modified.

    Raises
    ------
    TypeError
        If ``flt`` is neither a list of FluxSurface instances nor an object
        exposing ``poincare_res.surfs``, or if an element is not a FluxSurface,
        or if ``density_range`` has non-numeric elements.
    ValueError
        If ``density_range`` is not a two-element sequence with start < end,
        if it contains negative values, or if no density has been assigned
        (see :func:`add_density`).
    """
    
    pc._check_surfaces(flt)
    pc._check_range("density_range", density_range, allow_negative=False)
    
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs
    
    if all(not any(s.density) for s in surfs):
        raise ValueError("No density assigned to the surfaces; run "
                         "add_density() first.")
    
    filtered_surfs = list()
    
    for surf in surfs:
        
        mask = np.zeros(surf.n, dtype=bool)
        
        for i, dens in enumerate(surf.density):
            
            # skip point types the surface has no points of
            if surf.errors[i] is None and surf.errors != [None]: continue
            
            if density_range is None or (dens >= density_range[0] and 
                                         dens <= density_range[1]):
                mask = mask | (surf.points.point_type == i)
        
        
        # Only include surface if it has at least one point
        # within the ranges
        if np.any(mask):
            # Filter the points
            surf = copy.deepcopy(surf)
            surf.filter_points(mask)
            filtered_surfs.append(surf)

    return filtered_surfs

def extract_points(surfaces):
    """
    Flatten the points of all surfaces and their properties into 1-D arrays.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Surfaces with assigned densities (see :func:`add_density`), as a
        list or an object exposing ``poincare_res.surfs``.

    Returns
    -------
    dict of numpy.ndarray
        Arrays of shape ``(N,)``, one entry per point of all surfaces:

        * ``"x1"``, ``"x2"``, ``"x3"``: point coordinates [m].
        * ``"r"``: cylindrical radius [m].
        * ``"z"``: vertical position [m].
        * ``"density"``: density of the point's point-type group [m^-3].
        * ``"point_type"``: point type (0 non-island, 1 island outer side, 2
          island inner side).
        * ``"surface_radius"``: radius of the point's point-type group, the R
          of its fit at Z = 0 [m].

    Raises
    ------
    TypeError
        If ``surfaces`` is neither a list of FluxSurface instances nor an
        object exposing ``poincare_res.surfs``, or if an element is not a
        FluxSurface.
    ValueError
        If no density has been assigned (see :func:`add_density`).
    """
    
    pc._check_surfaces(surfaces)
    surfs = (surfaces if isinstance(surfaces, list)
             else surfaces.poincare_res.surfs)
    
    data = {"x1": list(), "x2": list(), "x3": list(), 
            "r": list(), "z": list(), "density": list(), 
            "point_type": list(), "surface_radius": list()}
    
    if all(not any(s.density) for s in surfs):
        raise ValueError("No density assigned to the surfaces; run "
                         "add_density() first.")
    
    for surf in surfs:
        data["x1"] += surf.points.x1.tolist()
        data["x2"] += surf.points.x2.tolist()
        data["x3"] += surf.points.x3.tolist()
        data["r"] += surf.points.r.tolist()
        data["z"] += surf.points.z.tolist()
        density = np.array(surf.density)[surf.points.point_type]
        data["density"] += density.tolist()
        data["point_type"] += surf.points.point_type.tolist()
        
        data["surface_radius"] += surf.coeffs[:, -1][surf.points.point_type
                                                    ].tolist()
        
    for key in data.keys():
        data[key] = np.array(data[key])
        
    return data

def make_regular_density_field(r, z, density, dr=0.0005, dz=0.0005, 
                               bottom_value=1e16, 
                               r_limits=None, z_limits=None, 
                               method="linear", fill_with_nearest=True):
    """
    Interpolate scattered ``(r, z, density)`` data onto a regular grid.

    Builds R and Z axes spanning the data (or ``r_limits`` / ``z_limits``) with
    spacings ``dr`` and ``dz``, and evaluates the density on the grid with
    :func:`scipy.interpolate.griddata`. Grid points outside the convex hull of
    the data are optionally filled by nearest-neighbour interpolation. Finally
    the field is clamped from below by ``bottom_value``, which also replaces
    any remaining NaNs.

    Parameters
    ----------
    r, z, density : array_like, shape (N,)
        Scattered data: cylindrical radius [m], vertical position [m] and
        density [m^-3] of each point, e.g. from :func:`extract_points`.
    dr, dz : float, default 0.0005
        Grid spacing along R and Z [m].
    bottom_value : float, default 1e16
        Floor density [m^-3]. No grid value is below it, and NaNs are replaced
        with it.
    r_limits, z_limits : sequence of float, default None
        Two-element ``[min, max]`` grid bounds [m]. If None, the data extent is
        used for that axis.
    method : str, default "linear"
        Interpolation method passed to :func:`scipy.interpolate.griddata`
        (``"linear"``, ``"cubic"`` or ``"nearest"``).
    fill_with_nearest : bool, default True
        If True, grid points left NaN by the interpolation (typically outside
        the convex hull of the data) are filled by a nearest-neighbour pass
        before the floor is applied.

    Returns
    -------
    R_axis : numpy.ndarray, shape (NR,)
        R coordinates of the grid [m].
    Z_axis : numpy.ndarray, shape (NZ,)
        Z coordinates of the grid [m].
    density_grid : numpy.ndarray, shape (NZ, NR)
        Density on the grid [m^-3], with ``density_grid[i, j]`` at
        ``(R_axis[j], Z_axis[i])`` (the layout used by
        ``matplotlib.pyplot.contourf``).

    Raises
    ------
    TypeError
        If ``r`` or ``z`` is not a sequence of numbers, if ``dr``, ``dz`` or
        ``bottom_value`` is not numeric, or if a limit has non-numeric
        elements.
    ValueError
        If ``dr``, ``dz`` or ``bottom_value`` is negative, or if ``r_limits``
        or ``z_limits`` is not a two-element sequence with start < end.
    """
    
    pc._check_sequence("r", r)
    pc._check_sequence("z", z)
    pc._check_scalar("dr", dr, min_value=0)
    pc._check_scalar("dz", dz, min_value=0)
    pc._check_scalar("bottom_value", bottom_value, min_value=0)
    pc._check_range("r_limits", r_limits)
    pc._check_range("z_limits", z_limits)
    
    from scipy.interpolate import griddata

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    density = np.asarray(density, dtype=float)

    if r_limits is None:
        r_min, r_max = float(r.min()), float(r.max())
    else:
        r_min, r_max = float(r_limits[0]), float(r_limits[1])

    if z_limits is None:
        z_min, z_max = float(z.min()), float(z.max())
    else:
        z_min, z_max = float(z_limits[0]), float(z_limits[1])

    # Add half a step so that the upper bound is included when it lands
    # on a grid point.
    R_axis = np.arange(r_min, r_max + 0.5 * dr, dr)
    Z_axis = np.arange(z_min, z_max + 0.5 * dz, dz)
    R_grid, Z_grid = np.meshgrid(R_axis, Z_axis)

    points = np.column_stack([r, z])
    density_grid = griddata(points, density, (R_grid, Z_grid), method=method)

    if fill_with_nearest:
        nan_mask = np.isnan(density_grid)
        if np.any(nan_mask):
            nearest_values = griddata(
                points,
                density,
                (R_grid[nan_mask], Z_grid[nan_mask]),
                method="nearest",
            )
            density_grid[nan_mask] = nearest_values

    # Replace any remaining NaNs with the floor, then clamp from below.
    density_grid = np.where(np.isnan(density_grid), bottom_value, density_grid)
    density_grid = np.maximum(density_grid, bottom_value)

    return R_axis, Z_axis, density_grid
