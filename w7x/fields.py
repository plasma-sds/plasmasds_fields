import numpy as np


def make_2D_density(flt, density_function, r_range=None, z_range=None, R_exception_range=(6.22, 6.23)):
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
