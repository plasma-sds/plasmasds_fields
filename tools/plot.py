import numpy as np


def contour_density(
    R,
    Z,
    density,
    r_range=None,
    z_range=None,
    levels=None,
    filled=True,
    log=False,
    cmap=None,
    ax=None,
    colorbar=True,
    cbar_label="density",
):
    """
    Contour-plot a density field on the (R, Z) plane.

    Accepts either a regular grid (as produced by
    :func:`w7x.fields.make_regular_density_field`) or scattered data
    (as produced by :func:`w7x.fields.extract_density_and_points`):

    * **Regular**: ``R`` and ``Z`` are 1-D coordinate axes of length
      ``nR`` and ``nZ`` respectively, ``density`` is 2-D with shape
      ``(nZ, nR)``. Plotted with :meth:`~matplotlib.axes.Axes.contourf`
      / :meth:`~matplotlib.axes.Axes.contour`.
    * **Scattered**: ``R``, ``Z``, ``density`` are 1-D arrays of equal
      length ``N``. Plotted with
      :meth:`~matplotlib.axes.Axes.tricontourf` /
      :meth:`~matplotlib.axes.Axes.tricontour`.

    Detection is automatic from ``density.ndim``.

    Parameters
    ----------
    R, Z, density : array_like
        Density field, in either the regular or scattered layout
        described above.
    r_range, z_range : sequence of float, optional
        Two-element ``[min, max]`` ranges on R and Z. If supplied, the
        field is restricted to this window before plotting (regular
        grids are sliced; scattered points are masked).
    levels : int or array_like, optional
        Number of contour levels, or explicit level values, forwarded
        to matplotlib. If ``log=True`` and ``levels`` is ``None``, a
        log-spaced default is generated from the positive density
        values.
    filled : bool, default True
        ``True`` for filled contours (``contourf`` / ``tricontourf``),
        ``False`` for line contours (``contour`` / ``tricontour``).
    log : bool, default False
        If ``True``, use a logarithmic colour normalization
        (:class:`matplotlib.colors.LogNorm`).
    cmap : str or Colormap, optional
        Colormap forwarded to matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are
        created and ``plt.show()`` is called before returning.
    colorbar : bool, default True
        If ``True``, attach a colorbar to ``ax``.
    cbar_label : str, optional
        Label for the colorbar. Set to ``None`` or ``""`` to omit.

    Returns
    -------
    matplotlib.contour.QuadContourSet or TriContourSet
        The contour set produced by matplotlib (useful for further
        customization such as adding a custom colorbar). Use
        ``cs.axes`` to retrieve the underlying axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    R = np.asarray(R)
    Z = np.asarray(Z)
    density = np.asarray(density)

    is_regular = density.ndim == 2

    if is_regular:
        if R.ndim != 1 or Z.ndim != 1 or density.shape != (Z.size, R.size):
            raise ValueError(
                "For a regular grid, R and Z must be 1-D and density must "
                f"have shape (Z.size, R.size); got R{R.shape}, Z{Z.shape}, "
                f"density{density.shape}."
            )
        if r_range is not None:
            r_mask = (R >= r_range[0]) & (R <= r_range[1])
            R = R[r_mask]
            density = density[:, r_mask]
        if z_range is not None:
            z_mask = (Z >= z_range[0]) & (Z <= z_range[1])
            Z = Z[z_mask]
            density = density[z_mask, :]
    else:
        if R.shape != Z.shape or R.shape != density.shape:
            raise ValueError(
                "For scattered data, R, Z and density must be 1-D arrays of "
                f"the same length; got R{R.shape}, Z{Z.shape}, "
                f"density{density.shape}."
            )
        mask = np.ones_like(R, dtype=bool)
        if r_range is not None:
            mask &= (R >= r_range[0]) & (R <= r_range[1])
        if z_range is not None:
            mask &= (Z >= z_range[0]) & (Z <= z_range[1])
        R, Z, density = R[mask], Z[mask], density[mask]

    if log and levels is None:
        positive = density[density > 0]
        if positive.size:
            vmin = float(positive.min())
            vmax = float(density.max())
            if vmax > vmin:
                levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)

    norm = LogNorm() if log else None
    contour_kwargs = {"levels": levels, "cmap": cmap, "norm": norm}
    contour_kwargs = {k: v for k, v in contour_kwargs.items() if v is not None}

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots()

    if is_regular:
        plotter = ax.contourf if filled else ax.contour
    else:
        plotter = ax.tricontourf if filled else ax.tricontour
    cs = plotter(R, Z, density, **contour_kwargs)

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal")

    if colorbar:
        cbar = plt.colorbar(cs, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    if created_fig:
        plt.show()

    return cs
