import numpy as np


def add_density_to_fields(flt, density_function, r_range=None, z_range=None, R_exception_range=(6.22, 6.23)):
    """
    Assign a density to each surface based on the average R of that surface's
    points inside the requested R-Z window, excluding points in an optional
    R-exception interval from the average.

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
    r_range : tuple or list, optional
        ``(R_min, R_max)`` range used to select candidate points.
    z_range : tuple or list, optional
        ``(Z_min, Z_max)`` range used to select candidate points.
    R_exception_range : tuple or list, optional
        ``(R_exc_min, R_exc_max)``. Points within this R interval are excluded
        from the averaging, but only after the surface points were already
        selected by ``r_range`` and ``z_range``.

    Returns
    -------
    list
        The updated list of surfaces.
    """
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs

    for surf in surfs:
        if surf.points.x1 is None or len(surf.points.x1) == 0:
            continue

        x1 = np.asarray(surf.points.x1)
        x2 = np.asarray(surf.points.x2)
        z = np.asarray(surf.points.x3)

        R = np.sqrt(x1**2 + x2**2)

        # Step 1: keep only points within the requested R and Z window
        if r_range is not None:
            r_mask = (R >= r_range[0]) & (R <= r_range[1])
        else:
            r_mask = np.ones_like(R, dtype=bool)

        if z_range is not None:
            z_mask = (z >= z_range[0]) & (z <= z_range[1])
        else:
            z_mask = np.ones_like(z, dtype=bool)

        selection_mask = r_mask & z_mask

        # No points from this surface in the requested R-Z domain
        if not np.any(selection_mask):
            continue

        R_selected = R[selection_mask]

        # Step 2: exclude only selected points inside the exception R interval
        if R_exception_range is not None:
            exception_mask = (
                (R_selected >= R_exception_range[0]) &
                (R_selected <= R_exception_range[1])
            )
            R_for_average = R_selected[~exception_mask]
        else:
            R_for_average = R_selected

        # If all selected points were excluded, skip this surface
        if R_for_average.size == 0:
            continue

        # Step 3: average R of this surface
        R_avg = np.mean(R_for_average)
        # Step 4: interpolate density and assign to surface
        surf.density = density_function(R_avg)

    return surfs


def filter_surfaces_by_density(flt, density_range=None, include_zero=False):
    """
    Filter flux surfaces by their assigned density.

    By default, surfaces whose ``density`` is exactly ``0`` (e.g. surfaces
    that were never assigned a density by :func:`add_density_to_fields`)
    are dropped. Optionally a ``[lower, upper]`` density range can be
    supplied to keep only surfaces whose density falls within those bounds.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances or an object
        exposing ``flt.poincare_res.surfs``.
    density_range : sequence of float, optional
        Two-element ``[lower, upper]`` interval on ``surf.density``.
        If ``None`` (default), no upper/lower bound is imposed beyond the
        ``include_zero`` rule.
    include_zero : bool, default False
        If ``False`` (default), surfaces with ``density == 0`` are always
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
        density = surf.density

        if not include_zero and density == 0:
            continue

        if density_range is not None:
            if density < density_range[0] or density > density_range[1]:
                continue

        filtered_surfs.append(surf)

    return filtered_surfs


def extract_density_and_points(flt, r_range=None, z_range=None):
    """
    Flatten all surfaces' points into ``R``, ``Z``, and ``density`` arrays.

    Iterates over every surface and every point on each surface, keeping
    only points whose cylindrical ``R = sqrt(x1**2 + x2**2)`` and
    ``Z = x3`` fall within the supplied ranges. For each retained point,
    the density value comes from its parent surface's ``density``
    attribute (so a surface's density is repeated once per retained
    point on that surface).

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances or an object
        exposing ``flt.poincare_res.surfs``.
    r_range : sequence of float, optional
        Two-element ``[R_min, R_max]`` interval. If ``None`` (default),
        no ``R`` constraint is applied.
    z_range : sequence of float, optional
        Two-element ``[Z_min, Z_max]`` interval. If ``None`` (default),
        no ``Z`` constraint is applied.

    Returns
    -------
    R : ndarray of shape (N,)
        Major radius of each retained point.
    Z : ndarray of shape (N,)
        Vertical coordinate of each retained point.
    density : ndarray of shape (N,)
        Parent-surface density associated with each retained point.
    """
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs

    R_chunks = []
    Z_chunks = []
    density_chunks = []

    for surf in surfs:
        if surf.points.x1 is None or len(surf.points.x1) == 0:
            continue

        x1 = np.asarray(surf.points.x1)
        x2 = np.asarray(surf.points.x2)
        z = np.asarray(surf.points.x3)

        R = np.sqrt(x1**2 + x2**2)

        if r_range is not None:
            r_mask = (R >= r_range[0]) & (R <= r_range[1])
        else:
            r_mask = np.ones_like(R, dtype=bool)

        if z_range is not None:
            z_mask = (z >= z_range[0]) & (z <= z_range[1])
        else:
            z_mask = np.ones_like(z, dtype=bool)

        mask = r_mask & z_mask
        n_kept = int(mask.sum())
        if n_kept == 0:
            continue

        R_chunks.append(R[mask])
        Z_chunks.append(z[mask])
        density_chunks.append(np.full(n_kept, surf.density, dtype=float))

    if not R_chunks:
        empty = np.empty(0)
        return empty, empty.copy(), empty.copy()

    return (
        np.concatenate(R_chunks),
        np.concatenate(Z_chunks),
        np.concatenate(density_chunks),
    )


def make_regular_density_field(r, z, density, dr=0.0005, dz=0.0005, bottom_value=1e17, 
r_limits=None, z_limits=None, method="linear", fill_with_nearest=True):
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

    # Add half a step so that the upper bound is included when it lands on a grid point.
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
