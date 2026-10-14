"""
Plotting of 2-D and time-dependent (R, Z) fields: contour plots, slices and
GIF animations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


def contour_field(R, Z, field, r_range=None, z_range=None, levels=None, filled=True,
log=False, cmap=None, ax=None, colorbar=True, cbar_label="Density [m^-3]",
equal_aspect=True, title="W7X 2D density plot", contour_lines=False, save_image=None):

    """
    Contour-plot a scalar field on the (R, Z) plane.

    Accepts either a regular grid (as produced by
    :func:`w7x.fields.make_regular_density_field`) or scattered data (as
    produced by :func:`w7x.fields.extract_points`); the layout is detected from
    ``field.ndim``:

    * **Regular**: ``R`` and ``Z`` are 1-D axes of length NR and NZ, and
      ``field`` is 2-D with shape (NZ, NR) or (NR, NZ). The layout is detected
      by matching ``field.shape`` against the axis lengths and transposed if
      needed, so R is always on the x-axis; a square field is taken as (NZ,
      NR). Plotted with :meth:`~matplotlib.axes.Axes.contourf` /
      :meth:`~matplotlib.axes.Axes.contour`.
    * **Scattered**: ``R``, ``Z`` and ``field`` are 1-D arrays of equal length
      N. Plotted with :meth:`~matplotlib.axes.Axes.tricontourf` /
      :meth:`~matplotlib.axes.Axes.tricontour`.

    Parameters
    ----------
    R, Z, field : array_like
        Coordinates [m] and field values, in the regular or scattered layout
        described above.
    r_range, z_range : sequence of float, default None
        Two-element ``[min, max]`` window on R and Z [m]. The field is
        restricted to it before plotting (regular grids are sliced, scattered
        points are masked).
    levels : int or array_like, default None
        Number of contour levels or explicit level values, forwarded to
        matplotlib. If ``log`` is True and ``levels`` is None, 30 log-spaced
        levels are generated from the positive field values.
    filled : bool, default True
        If True, draw filled contours (``contourf`` / ``tricontourf``),
        otherwise line contours (``contour`` / ``tricontour``).
    log : bool, default False
        If True, use a logarithmic colour normalization
        (:class:`matplotlib.colors.LogNorm`).
    cmap : str or Colormap, default None
        Colormap forwarded to matplotlib.
    ax : matplotlib.axes.Axes, default None
        Axes to draw on. If None, a new figure is created and ``plt.show()`` is
        called before returning.
    colorbar : bool, default True
        If True, attach a colorbar to ``ax``.
    cbar_label : str, default "Density [m^-3]"
        Label of the colorbar. None or ``""`` omits it.
    equal_aspect : bool, default True
        If True, use the same scale for R and Z (``ax.set_aspect("equal")``).
    title : str, default "W7X 2D density plot"
        Axes title, drawn in bold. None or ``""`` omits it.
    contour_lines : bool, default False
        If True, overlay thin black contour lines at the same levels.
    save_image : str or path-like, default None
        If given, the figure is saved to this path before it is shown; the
        format follows the file extension.

    Returns
    -------
    matplotlib.contour.QuadContourSet or matplotlib.tri.TriContourSet
        The contour set produced by matplotlib; ``cs.axes`` gives the axes.

    Raises
    ------
    ValueError
        If, for a regular grid, ``R`` or ``Z`` is not 1-D or ``field`` does not
        match their lengths, or if, for scattered data, ``R``, ``Z`` and
        ``field`` do not have the same shape.
    """


    R = np.asarray(R)
    Z = np.asarray(Z)
    field = np.asarray(field)

    is_regular = field.ndim == 2

    if is_regular:
        if R.ndim != 1 or Z.ndim != 1:
            raise ValueError(
                "For a regular grid, R and Z must be 1-D; "
                f"got R{R.shape}, Z{Z.shape}."
            )
        if field.shape == (Z.size, R.size):
            pass
        elif field.shape == (R.size, Z.size) and R.size != Z.size:
            field = field.T
        else:
            raise ValueError(
                f"field shape {field.shape} does not match R{R.shape} / "
                f"Z{Z.shape}. Expected ({Z.size}, {R.size}) or "
                f"({R.size}, {Z.size})."
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

    label_fs = plt.rcParams["font.size"] + 2
    ax.set_xlabel("R [m]", fontweight="bold", fontsize=label_fs)
    ax.set_ylabel("Z [m]", fontweight="bold", fontsize=label_fs)
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


def slice_field(R, Z, t, field, field_name,
                  at_t=None, at_r=None, at_z=None,
                  r_range=None, z_range=None, t_range=None,
                  cbar_label="", levels=30, cmap=None, log=False,
                  axis_order=None, equal_aspect=None,
                  contour_lines=False, ax=None, save_image=None,
                  time_resolution='s', figsize=None, dpi=100,
                  plot_3d=False):
    """
    Plot a 2-D slice of a 3-D (R, Z, t) field.

    Exactly one of ``at_t``, ``at_r``, ``at_z`` must be given; it selects the
    axis to slice and the coordinate of the slice (the nearest sample is used).
    The 2-D slice is plotted on the remaining axes:

    * ``at_t``: the (R, Z) plane (snapshot).
    * ``at_r``: the (t, Z) plane (time evolution at fixed R).
    * ``at_z``: the (t, R) plane (time evolution at fixed Z).

    The axis layout of ``field`` is detected as in :func:`animate_field`, e.g.
    (t, Z, R) from :meth:`models.hesel.HESEL.extract_field` or (t, R, Z) from
    :mod:`models.motion`. Pass ``axis_order`` when the axis lengths are
    ambiguous.

    Parameters
    ----------
    R, Z, t : array_like
        1-D coordinate axes: R [m], Z [m], t [s].
    field : array_like
        3-D scalar field; some permutation of (R, Z, t).
    field_name : str
        Name of the field, shown in the title together with the slice
        coordinate.
    at_t, at_r, at_z : float, default None
        Coordinate of the slice; exactly one must be given. ``at_t`` is in the
        unit selected by ``time_resolution``, ``at_r`` and ``at_z`` are in [m].
    r_range, z_range, t_range : sequence of float, default None
        Two-element ``[min, max]`` crops applied before slicing. ``r_range``
        and ``z_range`` are in [m], ``t_range`` in the unit selected by
        ``time_resolution``.
    cbar_label : str, default ""
        Label of the colorbar (e.g. ``"n [m^-3]"`` or ``"T_e [eV]"``).
    levels : int or array_like, default 30
        Number of contour levels or explicit level values. An int gives levels
        spaced linearly (logarithmically if ``log`` is True) between the
        minimum and maximum of the slice.
    cmap : str or Colormap, default None
        Colormap forwarded to matplotlib.
    log : bool, default False
        If True, use a logarithmic colour normalization
        (:class:`~matplotlib.colors.LogNorm`).
    axis_order : sequence of {"R", "Z", "t"}, default None
        Labels of the three axes of ``field``. If None, they are detected from
        the axis lengths.
    equal_aspect : bool, default None
        If True, use the same scale on both axes. If None, True for slices at
        fixed t (R and Z share units) and False otherwise.
    contour_lines : bool, default False
        If True, overlay thin black contour lines at the same levels.
    ax : matplotlib.axes.Axes, default None
        Axes to draw on. If None, a new figure is created and
        :func:`matplotlib.pyplot.show` is called before returning.
    save_image : str or path-like, default None
        If given, the figure is saved to this path before it is shown.
    time_resolution : {"s", "ms", "us"}, default "s"
        Unit of the time coordinate in ``at_t``, ``t_range``, the time axis and
        the title.
    figsize : tuple of float, default None
        Forwarded to :func:`matplotlib.pyplot.subplots`.
    dpi : int, default 100
        Figure resolution.
    plot_3d : bool, default False
        If True, render the slice as a 3-D surface with the field values on the
        vertical axis instead of a filled contour. ``equal_aspect`` is ignored,
        and ``contour_lines`` overlays 3-D contour lines.

    Returns
    -------
    matplotlib.artist.Artist
        The plotted artist: a :class:`~matplotlib.contour.QuadContourSet`, or a
        :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` for
        ``plot_3d=True``.

    Raises
    ------
    ValueError
        If not exactly one of ``at_t``, ``at_r``, ``at_z`` is given, if
        ``time_resolution`` is invalid, if ``R``, ``Z``, ``t`` are not 1-D or
        ``field`` is not 3-D, if the axis layout is ambiguous or ``axis_order``
        is invalid, if ``log`` is True and the slice has no positive values, or
        if ``plot_3d`` is True and ``ax`` is not a 3-D axes.
    """

    selectors = [("t", at_t), ("R", at_r), ("Z", at_z)]
    active = [(n, v) for n, v in selectors if v is not None]
    if len(active) != 1:
        raise ValueError(
            "Exactly one of at_t, at_r, at_z must be provided; "
            f"got at_t={at_t!r}, at_r={at_r!r}, at_z={at_z!r}."
        )
    slice_axis_name, slice_value = active[0]

    time_scales = {"s": 1.0, "ms": 1e3, "us": 1e6}
    if time_resolution not in time_scales:
        raise ValueError(
            f"time_resolution must be one of {list(time_scales)}; "
            f"got {time_resolution!r}."
        )
    time_scale = time_scales[time_resolution]

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
        tm = (t >= t_range[0] / time_scale) & (t <= t_range[1] / time_scale)
        t, field = t[tm], field[tm, :, :]

    if slice_axis_name == "t":
        slice_value_s = slice_value / time_scale
        idx = int(np.argmin(np.abs(t - slice_value_s)))
        slice_2d = field[idx, :, :]
        x_axis, y_axis = R, Z
        xlabel, ylabel = "R [m]", "Z [m]"
        title = (
            f"{field_name} at t = "
            f"{t[idx] * time_scale:.2f} {time_resolution}"
        )
        if equal_aspect is None:
            equal_aspect = True
    elif slice_axis_name == "R":
        idx = int(np.argmin(np.abs(R - slice_value)))
        slice_2d = field[:, :, idx].T
        x_axis, y_axis = t * time_scale, Z
        xlabel, ylabel = f"Time [{time_resolution}]", "Z [m]"
        title = f"{field_name} at R = {R[idx]:.3f} m"
        if equal_aspect is None:
            equal_aspect = False
    else:
        idx = int(np.argmin(np.abs(Z - slice_value)))
        slice_2d = field[:, idx, :].T
        x_axis, y_axis = t * time_scale, R
        xlabel, ylabel = f"Time [{time_resolution}]", "R [m]"
        title = f"{field_name} at Z = {Z[idx]:.3f} m"
        if equal_aspect is None:
            equal_aspect = False

    if log:
        positive = slice_2d[slice_2d > 0]
        if positive.size == 0:
            raise ValueError("log=True but the slice has no positive values.")
        vmin = float(positive.min())
        vmax = float(slice_2d.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)
        if isinstance(levels, (int, np.integer)):
            levels = np.logspace(np.log10(vmin), np.log10(vmax), int(levels))
    else:
        vmin = float(slice_2d.min())
        vmax = float(slice_2d.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        if isinstance(levels, (int, np.integer)):
            levels = np.linspace(vmin, vmax, int(levels))

    created_fig = ax is None
    if plot_3d:
        if created_fig:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
            if not hasattr(ax, "plot_surface"):
                raise ValueError(
                    "plot_3d=True requires a 3-D axes; create with "
                    "fig.add_subplot(projection='3d') and pass it as ax."
                )

        X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis)
        cmap_used = cmap if cmap is not None else "viridis"
        artist = ax.plot_surface(
            X_mesh, Y_mesh, slice_2d,
            cmap=cmap_used, norm=norm,
            linewidth=0, antialiased=True,
            shade=False,
        )
        if contour_lines:
            ax.contour(X_mesh, Y_mesh, slice_2d, levels=levels,
                       colors="black", linewidths=0.5)

        cbar = fig.colorbar(artist, ax=ax, shrink=0.6, pad=0.1)

        ax.set_zlabel(cbar_label or field_name, fontweight="bold",
                      rotation=90,
                      fontsize=plt.rcParams["font.size"] + 2)
        for tick_label in ax.get_zticklabels():
            tick_label.set_fontweight("bold")
        ax.zaxis.get_offset_text().set_fontweight("bold")
    else:
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        artist = ax.contourf(x_axis, y_axis, slice_2d,
                             levels=levels, cmap=cmap, norm=norm)
        if contour_lines:
            ax.contour(x_axis, y_axis, slice_2d, levels=artist.levels,
                       colors="black", linewidths=0.5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(artist, cax=cax)
        if equal_aspect:
            ax.set_aspect("equal")

    if cbar_label:
        cbar.set_label(cbar_label, fontweight="bold")
    for tick_label in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    cbar.ax.xaxis.get_offset_text().set_fontweight("bold")
    cbar.ax.yaxis.get_offset_text().set_fontweight("bold")

    label_fs = plt.rcParams["font.size"] + 2
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=label_fs)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=label_fs)
    ax.set_title(title, fontweight="bold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontweight("bold")

    if slice_axis_name == "Z" and plot_3d:
        ax.invert_yaxis()

    if save_image is not None:
        fig.savefig(save_image, bbox_inches="tight")
    if created_fig:
        plt.show()

    return artist


def animate_field(R, Z, t, field, field_name, cbar_label, save_path,
                  r_range=None, z_range=None, t_range=None,
                  fps=10, cmap=None, log=False, levels=30,
                  axis_order=None,
                  equal_aspect=True, figsize=None, dpi=100,
                  time_resolution='s'):
    """
    Animate the time evolution of one to four (R, Z) fields as a GIF.

    Renders one filled-contour frame per time step for each field and writes
    the animation to ``save_path`` with
    :class:`~matplotlib.animation.PillowWriter`. The fields are shown side by
    side (1 field: 1 x 1, 2: 1 x 2, 3: 1 x 3, 4: 2 x 2), each with its own
    colour scale and colorbar; the time stamp is drawn once as the figure
    title. The colour scale of each field is computed once from its minimum and
    maximum over the cropped volume, so it stays fixed across frames.

    For a single field, pass ``field`` as a 3-D array and ``cbar_label`` as a
    string; for several fields, pass both as lists of matching length. All
    fields must have the same shape and share the axes ``R``, ``Z``, ``t``.

    The axis layout is detected by matching the lengths of ``R``, ``Z`` and
    ``t`` against the field shape, so both layouts used in this project work:

    * HESEL (:meth:`models.hesel.HESEL.extract_field`): (t, Z, R); pass
      ``hesel.R_axis``, ``hesel.Z_axis``, ``hesel.time_axis``.
    * Motion (:func:`models.motion.perturbation_gaussian`,
      :func:`models.motion.perturbation_sine`,
      :func:`models.motion.import_density_field`): (t, R, Z); pass
      ``data["R"]``, ``data["Z"]``, ``data["t"]``, ``data["field"]``.

    Pass ``axis_order`` when the axis lengths are ambiguous (e.g. NR == NZ).

    Parameters
    ----------
    R, Z, t : array_like
        1-D coordinate axes shared by all fields: R [m], Z [m], t [s].
    field : array_like or sequence of array_like
        A 3-D field, or a list of up to four 3-D fields of identical shape;
        each a permutation of (R, Z, t).
    field_name : str
        Title of the animation, drawn once above all subplots.
    cbar_label : str or sequence of str
        Colorbar label of each field (e.g. ``"n [m^-3]"``); one per field.
    save_path : str or path-like
        Output path of the GIF; ``.gif`` is appended if missing.
    r_range, z_range, t_range : sequence of float, default None
        Two-element ``[min, max]`` crops applied to all fields. ``r_range`` and
        ``z_range`` are in [m], ``t_range`` in the unit selected by
        ``time_resolution``.
    fps : int, default 10
        Frames per second of the GIF.
    cmap : str or Colormap, default None
        Colormap forwarded to matplotlib, shared by all subplots.
    log : bool, default False
        If True, use a logarithmic colour normalization
        (:class:`~matplotlib.colors.LogNorm`) for every field; non-positive
        values are clipped to the field's smallest positive value.
    levels : int or array_like, default 30
        Number of contour levels or explicit level values. An int gives levels
        spaced linearly (logarithmically if ``log`` is True) between each
        field's minimum and maximum.
    axis_order : sequence of {"R", "Z", "t"}, default None
        Labels of the three axes of the fields, e.g. ``("t", "Z", "R")`` for
        HESEL or ``("t", "R", "Z")`` for motion. If None, they are detected
        from the axis lengths.
    equal_aspect : bool, default True
        If True, use the same scale for R and Z on every subplot.
    figsize : tuple of float, default None
        Forwarded to :func:`matplotlib.pyplot.subplots`. If None, ``(6 * ncols,
        4.5 * nrows)`` for the subplot layout.
    dpi : int, default 100
        Resolution of the GIF.
    time_resolution : {"s", "ms", "us"}, default "s"
        Unit of ``t_range`` and of the time stamp in the title (two decimal
        places).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, already saved to ``save_path``.

    Raises
    ------
    ValueError
        If ``time_resolution`` is invalid, if there are not 1-4 fields or the
        number of ``cbar_label`` entries differs, if ``R``, ``Z``, ``t`` are
        not 1-D or a field is not 3-D or differs in shape, if the axis layout
        is ambiguous or ``axis_order`` is invalid, if fewer than two R or Z
        samples or no time step remain after cropping, or if ``log`` is True
        and a field has no positive values.
    """

    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        save_path = save_path.with_suffix(".gif")

    time_scales = {"s": 1.0, "ms": 1e3, "us": 1e6}
    if time_resolution not in time_scales:
        raise ValueError(
            f"time_resolution must be one of {list(time_scales)}; "
            f"got {time_resolution!r}."
        )
    time_scale = time_scales[time_resolution]

    def _as_list(x):
        """
        Wrap a single value in a list; return lists and tuples as lists.
        """

        return list(x) if isinstance(x, (list, tuple)) else [x]

    fields = [np.asarray(f) for f in _as_list(field)]
    cbar_labels = _as_list(cbar_label)

    n_fields = len(fields)
    if not (1 <= n_fields <= 4):
        raise ValueError(
            f"number of fields must be between 1 and 4; got {n_fields}."
        )
    if len(cbar_labels) != n_fields:
        raise ValueError(
            "field and cbar_label must have matching lengths; "
            f"got {n_fields} fields and {len(cbar_labels)} cbar_label(s)."
        )

    R = np.asarray(R)
    Z = np.asarray(Z)
    t = np.asarray(t)

    if R.ndim != 1 or Z.ndim != 1 or t.ndim != 1:
        raise ValueError(
            f"R, Z, t must be 1-D; got shapes R{R.shape}, Z{Z.shape}, t{t.shape}."
        )
    for i, f in enumerate(fields):
        if f.ndim != 3:
            raise ValueError(f"field[{i}] must be 3-D; got shape {f.shape}.")
        if f.shape != fields[0].shape:
            raise ValueError(
                "all fields must share the same shape; "
                f"field[0].shape={fields[0].shape} but "
                f"field[{i}].shape={f.shape}."
            )

    if axis_order is None:
        sizes = {"R": R.size, "Z": Z.size, "t": t.size}
        order = []
        for ax_size in fields[0].shape:
            matches = [n for n, s in sizes.items() if s == ax_size]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot auto-detect axis layout: field.shape={fields[0].shape} "
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
        if expected != fields[0].shape:
            raise ValueError(
                f"axis_order={tuple(order)} implies field.shape={expected}, "
                f"but got {fields[0].shape}."
            )

    perm = [order.index(axname) for axname in ("t", "Z", "R")]
    fields = [np.transpose(f, perm) for f in fields]

    if r_range is not None:
        rm = (R >= r_range[0]) & (R <= r_range[1])
        R = R[rm]
        fields = [f[:, :, rm] for f in fields]
    if z_range is not None:
        zm = (Z >= z_range[0]) & (Z <= z_range[1])
        Z = Z[zm]
        fields = [f[:, zm, :] for f in fields]
    if t_range is not None:
        tm = (t >= t_range[0] / time_scale) & (t <= t_range[1] / time_scale)
        t = t[tm]
        fields = [f[tm, :, :] for f in fields]

    if R.size < 2 or Z.size < 2 or t.size < 1:
        raise ValueError(
            "Insufficient samples after cropping: "
            f"R={R.size}, Z={Z.size}, t={t.size}."
        )

    norms = []
    levels_list = []
    for i, f in enumerate(fields):
        if log:
            positive = f[f > 0]
            if positive.size == 0:
                raise ValueError(
                    f"log=True but field[{i}] has no positive values."
                )
            vmin = float(positive.min())
            vmax = float(f.max())
            norm = LogNorm(vmin=vmin, vmax=vmax)
            if isinstance(levels, (int, np.integer)):
                lvls = np.logspace(np.log10(vmin), np.log10(vmax), int(levels))
            else:
                lvls = np.asarray(levels)
        else:
            vmin = float(f.min())
            vmax = float(f.max())
            norm = Normalize(vmin=vmin, vmax=vmax)
            if isinstance(levels, (int, np.integer)):
                lvls = np.linspace(vmin, vmax, int(levels))
            else:
                lvls = np.asarray(levels)
        norms.append(norm)
        levels_list.append(lvls)

    layout = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}
    nrows, ncols = layout[n_fields]
    if figsize is None:
        figsize = (6.0 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes_list = [axes] if n_fields == 1 else np.asarray(axes).flatten().tolist()

    for ax, label, norm in zip(axes_list, cbar_labels, norms):
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label, fontweight="bold")
        for tick_label in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        cbar.ax.xaxis.get_offset_text().set_fontweight("bold")
        cbar.ax.yaxis.get_offset_text().set_fontweight("bold")

    label_fs = plt.rcParams["font.size"] + 2
    for ax in axes_list:
        ax.set_xlabel("R [m]", fontweight="bold", fontsize=label_fs)
        ax.set_ylabel("Z [m]", fontweight="bold", fontsize=label_fs)
        ax.set_anchor("N")
    fig.suptitle(f"{field_name}\nt = ", fontweight="bold", y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        """
        Draw the contour plots of all fields for animation frame ``frame``.
        """

        for ax, f_data, norm, lvls in zip(
            axes_list, fields, norms, levels_list
        ):
            ax.clear()
            frame_data = f_data[frame]
            if log:
                frame_data = np.where(frame_data > 0, frame_data, norm.vmin)
            ax.contourf(R, Z, frame_data, levels=lvls, cmap=cmap, norm=norm)
            ax.set_xlabel("R [m]", fontweight="bold", fontsize=label_fs)
            ax.set_ylabel("Z [m]", fontweight="bold", fontsize=label_fs)
            if equal_aspect:
                ax.set_aspect("equal")
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                tick_label.set_fontweight("bold")
            ax.xaxis.get_offset_text().set_fontweight("bold")
            ax.yaxis.get_offset_text().set_fontweight("bold")
        fig.suptitle(
            f"{field_name}\nt = {t[frame] * time_scale:.2f} {time_resolution}",
            fontweight="bold",
            y=0.99,
        )

    anim = FuncAnimation(
        fig, update, frames=t.size, interval=1000.0 / fps, blit=False
    )

    last_pct = -1
    def _progress(current, total):
        """
        Print a text progress bar while the GIF is rendered.
        """

        nonlocal last_pct
        pct = int((current + 1) / total * 100)
        if pct != last_pct:
            last_pct = pct
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "." * (bar_len - filled)
            end_char = "\n" if pct == 100 else ""
            print(
                f"\rRendering: [{bar}] {pct:3d}% ({current + 1}/{total})",
                end=end_char, flush=True,
            )

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi, progress_callback=_progress)
    plt.close(fig)

    return anim
