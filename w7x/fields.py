import numpy as np


def add_density_by_type(flt, density_function):
    """
    Assign a density to each surface based on the R of the surface's
    points, obtained prior (for example by polinom fit)

    Parameters
    ----------
    flt : list or object
        Either a list of Surf objects or an object with ``flt.poincare_res.surfs``.
    density_function : callable
        A scipy 1D interpolator (e.g. ``scipy.interpolate.interp1d``) that
        returns the density at a given radial position ``R``. Typically this
        is the ``interpolator`` attribute of a
        ``tools.interpolate.ProfileInterpolator1D`` instance, i.e. pass
        ``profile.interpolator`` rather than the ``ProfileInterpolator1D``
        object itself.
        --> note that scipy 1D interpolator can only interpolate, and cannot
        extrapolate.

    Returns
    -------
    list
        The updated list of surfaces.
    """
    
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs
    
    for surf in surfs:
        for i in range(len(surf.errors)):
            mask = surf.points.point_type == i
            if np.any(mask):
                dens = density_function(surf.coeffs[i, 2])
                surf.update_density(dens, mask=mask)
                
    return surfs

def filter_surfaces_by_density(flt, density_range=None, include_zero=False):
    """
    Filter list of FluxSurface objects by their assigned density.

    By default, surfaces whose ``density`` is exactly ``0`` (e.g. surfaces
    that were never assigned a density by :func:`add_density_by_type`)
    are dropped. Optionally a ``[lower, upper]`` density range can be
    supplied to keep only points in surfaces whose density falls
    within those bounds.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances or an object
        exposing ``flt.poincare_res.surfs``.
    density_range : sequence of float, optional
        Two-element ``[lower, upper]`` interval on ``surf.points.density``.
        If ``None`` (default), no upper/lower bound is imposed beyond the
        ``include_zero`` rule.
    include_zero : bool, default False
        If ``False`` (default), points with ``density == 0`` are always
        dropped. If ``True``, zero-density surfaces are kept (subject to
        ``density_range`` if it is supplied).

    Returns
    -------
    list of FluxSurface
        The surfaces (unmodified, by reference) that pass the filter.
    """
    
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs

    filtered_surfs = []
    for surf in surfs:
        density_temp = surf.points.density
        
        if density_range is not None:
            density_mask = ((density_temp >= density_range[0])
                            & (density_temp <= density_range[1]))
        else:
            density_mask = np.ones(surf.n, dtype=bool)
            
        if include_zero: 
            not_zero_mask = density_temp != 0
        else:
            not_zero_mask = np.ones(surf.n, dtype=bool)
        
        # Keep only points within both ranges
        mask = density_mask & not_zero_mask
        
        # Only include surface if it has at least one point
        # within the ranges
        if np.any(mask):
            # Filter the points
            surf.filter_points(mask)
            filtered_surfs.append(surf)

    return filtered_surfs

def extract_points(surfaces):
    """
    Flatten all surfaces' points and their properties into arrays, stored
    in a dictionary.

    Iterates over every surface and every point on each surface.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances or an object
        exposing ``flt.poincare_res.surfs``.

    Returns
    -------
    data : dictionary with the following keys:
        "x1" : ndarray of shape (N,)
            x1 coordinates of all the points in the list of surfaces.
        "x2" : ndarray of shape (N,)
            x2 coordinates of all the points in the list of surfaces.
        "x3" : ndarray of shape (N,)
            x3 coordinates of all the points in the list of surfaces.
        "r" : ndarray of shape (N,)
            r radial position of all the points in the list of surfaces.
        "z" : ndarray of shape (N,)
            z vertical position of all the points in the list of surfaces.
        "density" : ndarray of shape (N,)
            density of all the points in the list of surfaces.
        "point_type" : ndarray of shape (N,)
            the type of all the points in the list of surfaces, related to
            island trajectories.
        "surface_radius" : ndarray of shape (N,)
            The radial position of the surface (or part of it).
    """
    
    data = {"x1": list(), "x2": list(), "x3": list(), 
            "r": list(), "z": list(), "density": list(), 
            "point_type": list(), "surface_radius": list()}
    for surf in surfaces:
        data["x1"] += surf.points.x1.tolist()
        data["x2"] += surf.points.x2.tolist()
        data["x3"] += surf.points.x3.tolist()
        data["r"] += surf.points.r.tolist()
        data["z"] += surf.points.z.tolist()
        data["density"] += surf.points.density.tolist()
        data["point_type"] += surf.points.point_type.tolist()
        data["surface_radius"] += surf.coeffs[:, 2][surf.points.point_type
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

    Builds 1-D ``R`` and ``Z`` coordinate axes spanning the data extent
    (or the user-supplied ``r_limits`` / ``z_limits``) with spacings
    ``dr`` and ``dz``, then uses :func:`scipy.interpolate.griddata` to
    evaluate the density on the resulting 2-D grid. Points outside the
    convex hull of the input scatter are optionally filled in with a
    nearest-neighbour interpolation. Finally, the whole field is
    clamped from below by ``bottom_value`` (so any remaining NaNs and
    any values smaller than the floor are replaced with
    ``bottom_value``).

    Parameters
    ----------
    r, z, density : array_like, shape (N,)
        Scattered input data: cylindrical R coordinates, vertical Z
        coordinates, and density values at each ``(r, z)`` point.
    dr, dz : float, default 0.0005
        Grid spacings along R and Z respectively.
    bottom_value : float, default 1e17
        Floor density. The output field is clamped so that no value is
        below ``bottom_value``; any NaNs are also replaced with this
        value.
    r_limits, z_limits : sequence of float, optional
        Two-element ``[min, max]`` bounds for the regular grid. If
        ``None`` (default), the data ``min``/``max`` is used for that
        axis.
    method : str, default "linear"
        Interpolation method passed to :func:`scipy.interpolate.griddata`
        for the primary pass (``"linear"``, ``"cubic"``, or
        ``"nearest"``).
    fill_with_nearest : bool, default True
        If ``True``, points returning NaN from the primary
        interpolation (typically outside the convex hull of the input
        scatter) are filled with a separate ``"nearest"`` pass before
        the bottom-value clamp is applied.

    Returns
    -------
    R_axis : ndarray of shape (nR,)
        1-D array of R grid coordinates.
    Z_axis : ndarray of shape (nZ,)
        1-D array of Z grid coordinates.
    density_grid : ndarray of shape (nZ, nR)
        2-D density field on ``np.meshgrid(R_axis, Z_axis)``, with
        ``density_grid[i, j]`` corresponding to ``(R_axis[j], Z_axis[i])``.
    """
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
