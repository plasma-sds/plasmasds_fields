import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def contour_field(R, Z, field, r_range=None, z_range=None, levels=None, filled=True,
log=False, cmap=None, ax=None, colorbar=True, cbar_label="Density [m^-3]",
equal_aspect=True, title="W7X 2D density plot", contour_lines=False, save_image=None):

    """
    Contour-plot a scalar field on the (R, Z) plane.

    Accepts either a regular grid (as produced by
    :func:`w7x.fields.make_regular_density_field`) or scattered data
    (as produced by :func:`w7x.fields.extract_points`):

    * **Regular**: ``R`` and ``Z`` are 1-D coordinate axes of length
      ``nR`` and ``nZ`` respectively, ``field`` is 2-D with shape
      ``(nZ, nR)``. Plotted with :meth:`~matplotlib.axes.Axes.contourf`
      / :meth:`~matplotlib.axes.Axes.contour`.
    * **Scattered**: ``R``, ``Z``, ``field`` are 1-D arrays of equal
      length ``N``. Plotted with
      :meth:`~matplotlib.axes.Axes.tricontourf` /
      :meth:`~matplotlib.axes.Axes.tricontour`.

    Detection is automatic from ``field.ndim``.

    Parameters
    ----------
    R, Z, field : array_like
        Scalar field, in either the regular or scattered layout
        described above.
    r_range, z_range : sequence of float, optional
        Two-element ``[min, max]`` ranges on R and Z. If supplied, the
        field is restricted to this window before plotting (regular
        grids are sliced; scattered points are masked).
    levels : int or array_like, optional
        Number of contour levels, or explicit level values, forwarded
        to matplotlib. If ``log=True`` and ``levels`` is ``None``, a
        log-spaced default is generated from the positive field
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
    equal_aspect : bool, default True
        If ``True``, force ``ax.set_aspect("equal")`` so R and Z share
        the same scale. Set to ``False`` to let matplotlib stretch the
        plot to fill the axes.
    title : str, optional
        Axes title, drawn in bold. Defaults to ``"W7X 2D field plot"``;
        set to ``None`` or ``""`` to omit.
    contour_lines : bool, default False
        If ``True``, overlay thin black contour lines on top of the
        plot at the same levels as the (filled) contours, marking the
        level boundaries explicitly.
    save_image : str or path-like, optional
        If given, the figure is saved to this path via
        :func:`matplotlib.figure.Figure.savefig` before any interactive
        display. The format is inferred from the extension; pass a
        ``.png`` path to save as PNG.

    Returns
    -------
    matplotlib.contour.QuadContourSet or TriContourSet
        The contour set produced by matplotlib (useful for further
        customization such as adding a custom colorbar). Use
        ``cs.axes`` to retrieve the underlying axes.
    """


    R = np.asarray(R)
    Z = np.asarray(Z)
    field = np.asarray(field)

    is_regular = field.ndim == 2

    if is_regular:
        if R.ndim != 1 or Z.ndim != 1 or field.shape != (Z.size, R.size):
            raise ValueError(
                "For a regular grid, R and Z must be 1-D and field must "
                f"have shape (Z.size, R.size); got R{R.shape}, Z{Z.shape}, "
                f"field{field.shape}."
            )
        if r_range is not None:
            r_mask = (R >= r_range[0]) & (R <= r_range[1])
            R = R[r_mask]
            field = field[:, r_mask]
        if z_range is not None:
            z_mask = (Z >= z_range[0]) & (Z <= z_range[1])
            Z = Z[z_mask]
            field = field[z_mask, :]
    else:
        if R.shape != Z.shape or R.shape != field.shape:
            raise ValueError(
                "For scattered data, R, Z and field must be 1-D arrays of "
                f"the same length; got R{R.shape}, Z{Z.shape}, "
                f"field{field.shape}."
            )
        mask = np.ones_like(R, dtype=bool)
        if r_range is not None:
            mask &= (R >= r_range[0]) & (R <= r_range[1])
        if z_range is not None:
            mask &= (Z >= z_range[0]) & (Z <= z_range[1])
        R, Z, field = R[mask], Z[mask], field[mask]

    if log and levels is None:
        positive = field[field > 0]
        if positive.size:
            vmin = float(positive.min())
            vmax = float(field.max())
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
    cs = plotter(R, Z, field, **contour_kwargs)

    if contour_lines:
        line_plotter = ax.contour if is_regular else ax.tricontour
        line_plotter(R, Z, field, levels=cs.levels, colors="black", linewidths=0.5)

    ax.set_xlabel("R [m]", fontweight="bold")
    ax.set_ylabel("Z [m]", fontweight="bold")
    if equal_aspect:
        ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontweight="bold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontweight("bold")

    if colorbar:
        cbar = plt.colorbar(cs, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, fontweight="bold")
        for tick_label in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        cbar.ax.xaxis.get_offset_text().set_fontweight("bold")
        cbar.ax.yaxis.get_offset_text().set_fontweight("bold")

    if save_image is not None:
        ax.figure.savefig(save_image, bbox_inches="tight")

    if created_fig:
        plt.show()

    return cs


def animate_field(R, Z, t, field, field_name, save_path,
                  r_range=None, z_range=None, t_range=None,
                  fps=10, cmap=None, log=False, levels=30,
                  cbar_label=None, axis_order=None,
                  equal_aspect=True, figsize=None, dpi=100):
    """
    Animate the time evolution of a 2-D ``(R, Z)`` field as a GIF.

    Draws one filled-contour frame per time step on the ``(R, Z)``
    plane and writes the result to ``save_path`` as an animated GIF
    via matplotlib's :class:`~matplotlib.animation.PillowWriter`.

    Accepts inputs in either layout commonly produced in this
    project, with auto-detection via the lengths of ``R``, ``Z`` and
    ``t`` against ``field.shape``:

    * **HESEL** (``models.hesel.HESEL.extract_field``): field shape
      ``(time, Z, R)``, axes ``hesel.R_axis``, ``hesel.Z_axis``,
      ``hesel.time_axis``.
    * **Motion** (``models.motion.add_filament_to_field`` /
      ``models.motion.import_density_field``): dict with ``"X"``,
      ``"Y"``, ``"t"`` axes and a ``"Z"`` field of shape
      ``(Nt, Nx, Ny)``; pass ``R=data["X"]``, ``Z=data["Y"]``,
      ``t=data["t"]``, ``field=data["Z"]``.

    When the three axis sizes are ambiguous (e.g. ``nR == nZ``),
    pass ``axis_order`` explicitly.

    Color levels and the normalization are computed *once* from the
    global min / max of the cropped field, so the color mapping is
    stable across frames (no flickering colorbar).

    Parameters
    ----------
    R, Z, t : array_like
        1-D coordinate axes for the radial, vertical, and temporal
        dimensions.
    field : array_like
        3-D scalar field whose axes are some permutation of
        ``(R, Z, t)``.
    field_name : str
        Name of the field (e.g. ``"density"`` or
        ``"electron_temperature"``). Used in the frame title and as
        the default colorbar label.
    save_path : str or path-like
        Output path for the animated GIF (e.g. ``"density.gif"``).
    r_range, z_range, t_range : sequence of float, optional
        Two-element ``[min, max]`` crops applied to R, Z and t
        before animating.
    fps : int, default 10
        Frames per second for the output GIF.
    cmap : str or Colormap, optional
        Colormap forwarded to matplotlib.
    log : bool, default False
        If ``True``, use a logarithmic color normalization
        (:class:`~matplotlib.colors.LogNorm`). Non-positive entries
        in each frame are clipped to the global positive minimum so
        ``LogNorm`` does not choke.
    levels : int or array_like, default 30
        Number of contour levels, or explicit level values. When an
        int is given the levels are spaced linearly (or
        logarithmically when ``log=True``) over the *global*
        ``vmin`` / ``vmax`` of the cropped field.
    cbar_label : str, optional
        Colorbar label. Defaults to ``field_name`` when ``None``.
    axis_order : sequence of {"R", "Z", "t"}, optional
        Explicit labelling of the 3 axes of ``field``, e.g.
        ``("t", "Z", "R")`` for HESEL or ``("t", "R", "Z")`` for
        motion. If ``None``, sizes are matched automatically.
    equal_aspect : bool, default True
        If ``True``, force equal R/Z visual scale on the axes.
    figsize : tuple of float, optional
        Forwarded to :func:`matplotlib.pyplot.subplots`.
    dpi : int, default 100
        Resolution of the saved GIF.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The constructed animation (already saved to ``save_path``).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib import cm
    from pathlib import Path

    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        save_path = save_path.with_suffix(".gif")

    R = np.asarray(R)
    Z = np.asarray(Z)
    t = np.asarray(t)
    field = np.asarray(field)

    if R.ndim != 1 or Z.ndim != 1 or t.ndim != 1:
        raise ValueError(
            f"R, Z, t must be 1-D; got shapes R{R.shape}, Z{Z.shape}, t{t.shape}."
        )
    if field.ndim != 3:
        raise ValueError(f"field must be 3-D; got shape {field.shape}.")

    if axis_order is None:
        sizes = {"R": R.size, "Z": Z.size, "t": t.size}
        order = []
        for ax_size in field.shape:
            matches = [n for n, s in sizes.items() if s == ax_size]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot auto-detect axis layout: field.shape={field.shape} "
                    f"is ambiguous against sizes R={R.size}, Z={Z.size}, "
                    f"t={t.size}. Pass `axis_order` explicitly, e.g. "
                    "axis_order=('t','Z','R')."
                )
            order.append(matches[0])
    else:
        order = list(axis_order)
        if sorted(order) != ["R", "Z", "t"]:
            raise ValueError(
                "axis_order must be a permutation of ('R','Z','t'); "
                f"got {order}."
            )
        expected = tuple({"R": R.size, "Z": Z.size, "t": t.size}[n] for n in order)
        if expected != field.shape:
            raise ValueError(
                f"axis_order={tuple(order)} implies field.shape={expected}, "
                f"but got {field.shape}."
            )

    perm = [order.index(axname) for axname in ("t", "Z", "R")]
    field = np.transpose(field, perm)

    if r_range is not None:
        rm = (R >= r_range[0]) & (R <= r_range[1])
        R, field = R[rm], field[:, :, rm]
    if z_range is not None:
        zm = (Z >= z_range[0]) & (Z <= z_range[1])
        Z, field = Z[zm], field[:, zm, :]
    if t_range is not None:
        tm = (t >= t_range[0]) & (t <= t_range[1])
        t, field = t[tm], field[tm, :, :]

    if R.size < 2 or Z.size < 2 or t.size < 1:
        raise ValueError(
            "Insufficient samples after cropping: "
            f"R={R.size}, Z={Z.size}, t={t.size}."
        )

    if log:
        positive = field[field > 0]
        if positive.size == 0:
            raise ValueError("log=True but field has no positive values.")
        vmin = float(positive.min())
        vmax = float(field.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)
        if isinstance(levels, (int, np.integer)):
            levels = np.logspace(np.log10(vmin), np.log10(vmax), int(levels))
    else:
        vmin = float(field.min())
        vmax = float(field.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        if isinstance(levels, (int, np.integer)):
            levels = np.linspace(vmin, vmax, int(levels))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label or field_name, fontweight="bold")
    for tick_label in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    cbar.ax.xaxis.get_offset_text().set_fontweight("bold")
    cbar.ax.yaxis.get_offset_text().set_fontweight("bold")

    def update(frame):
        ax.clear()
        frame_data = field[frame]
        if log:
            frame_data = np.where(frame_data > 0, frame_data, vmin)
        ax.contourf(R, Z, frame_data, levels=levels, cmap=cmap, norm=norm)
        ax.set_xlabel("R [m]", fontweight="bold")
        ax.set_ylabel("Z [m]", fontweight="bold")
        if equal_aspect:
            ax.set_aspect("equal")
        ax.set_title(
            f"{field_name} at t = {t[frame]:.3e} s", fontweight="bold"
        )
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.xaxis.get_offset_text().set_fontweight("bold")
        ax.yaxis.get_offset_text().set_fontweight("bold")

    anim = FuncAnimation(
        fig, update, frames=t.size, interval=1000.0 / fps, blit=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return anim
