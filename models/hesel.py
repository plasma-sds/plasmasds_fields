import h5py as h5
import numpy as np


class HESEL:
    """
    Lightweight data handler for HESEL output stored in large HDF5 files.

    On construction the radial, vertical and temporal coordinate axes
    are loaded from the ``data/xanimation`` group and the corresponding
    grid spacings (``dR``, ``dZ``, ``dt``) are derived. The underlying
    HDF5 file is kept open as ``self.file`` so that bulk datasets can be
    sliced lazily on demand without paying the cost of re-opening the
    file.

    The handler can be used as a context manager::

        with HESEL("run.h5") as hesel:
            ...

    or closed explicitly via :meth:`close`.

    Parameters
    ----------
    path : str or path-like
        Path to the HESEL HDF5 output file.

    Attributes
    ----------
    path : str
        Path that was opened.
    file : h5py.File or None
        Open HDF5 file handle (read-only). Set to ``None`` after
        :meth:`close`.
    R_axis, Z_axis, time_axis : numpy.ndarray
        1-D coordinate axes for the radial, vertical, and temporal
        dimensions, loaded from ``data/xanimation/xgrid[0, :]``,
        ``data/xanimation/ygrid[:, 0]`` and ``data/xanimation/time``
        respectively.
    dR, dZ, dt : float
        Mean spacing of the corresponding axis (equal to the uniform
        step size when the axis is regular).
    """

    def __init__(self, path):
        self.path = str(path)
        self.file = h5.File(self.path, "r")

        self.R_axis = self.file["data/xanimation/xgrid"][0, :]
        self.Z_axis = self.file["data/xanimation/ygrid"][:, 0]
        self.time_axis = self.file["data/xanimation/time"][:, 0]

        self.dR = self._axis_spacing(self.R_axis)
        self.dZ = self._axis_spacing(self.Z_axis)
        self.dt = self._axis_spacing(self.time_axis)

    @staticmethod
    def _axis_spacing(axis):
        """Step size of a uniformly spaced 1-D axis."""
        return float(axis[1] - axis[0])

    @staticmethod
    def _range_to_selector(axis, rng):
        """
        Translate a ``None`` / scalar / ``[min, max]`` range into an
        h5py-compatible selector for the given 1-D ``axis``.

        * ``None``  -> ``slice(None)`` (keep full axis).
        * scalar    -> integer index of the closest axis sample
          (collapses the axis when used in slicing).
        * 2 values  -> contiguous ``slice(i_lo, i_hi + 1)`` covering
          the samples nearest to ``min`` and ``max``.
        """
        if rng is None:
            return slice(None)

        arr = np.atleast_1d(rng)
        if arr.size == 1:
            return int(np.argmin(np.abs(axis - arr[0])))
        if arr.size == 2:
            lo, hi = float(arr[0]), float(arr[1])
            if lo > hi:
                lo, hi = hi, lo
            i_lo = int(np.argmin(np.abs(axis - lo)))
            i_hi = int(np.argmin(np.abs(axis - hi)))
            if i_lo > i_hi:
                i_lo, i_hi = i_hi, i_lo
            return slice(i_lo, i_hi + 1)

        raise ValueError(
            "Range must be None, a scalar, or a two-element [min, max]; "
            f"got an array of size {arr.size}."
        )

    def extract_field(self, field, r_range=None, z_range=None, t_range=None):
        """
        Extract a HESEL field from ``data/xanimation/<field>``.

        The underlying dataset is 3-D with shape ``(time, Z, R)``.
        Only the bytes covered by the requested ranges are read from
        disk via h5py's lazy slicing.

        Parameters
        ----------
        field : {"density", "electron_temperature", "ion_temperature"}
            Name of the dataset to read.
        r_range, z_range, t_range : None, scalar, or sequence of two floats, optional
            Selector for the R, Z and time axes respectively:

            * ``None`` (default): keep the full extent along that axis.
            * Scalar: keep only the sample nearest to that coordinate;
              the axis is dropped from the returned array.
            * Two-element ``[min, max]``: keep the contiguous block of
              samples whose coordinates lie nearest to the endpoints;
              the axis is retained.

        Returns
        -------
        numpy.ndarray
            The requested field section. Axes selected by a scalar
            range are collapsed; axes selected by ``None`` or a
            two-element range are retained. The dimension order
            follows the dataset, i.e. ``(time, Z, R)`` before
            collapse.
        """
        valid = ("density", "electron_temperature", "ion_temperature")
        if field not in valid:
            raise ValueError(
                f"field must be one of {valid}; got {field!r}."
            )

        dataset = self.file[f"data/xanimation/{field}"]

        t_sel = self._range_to_selector(self.time_axis, t_range)
        z_sel = self._range_to_selector(self.Z_axis, z_range)
        r_sel = self._range_to_selector(self.R_axis, r_range)

        return dataset[t_sel, z_sel, r_sel]

    def close(self):
        """Close the underlying HDF5 file. Safe to call multiple times."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        nR = getattr(self.R_axis, "size", 0)
        nZ = getattr(self.Z_axis, "size", 0)
        nT = getattr(self.time_axis, "size", 0)
        return f"HESEL(path={self.path!r}, R={nR}, Z={nZ}, time={nT})"


def expand_hesel(field, R_axis, Z_axis, n_z=1, n_sol=0, n_edge=0,
                 r_axis=-1, z_axis=-2):
    """
    Expand a HESEL field along R and/or Z in any combination.

    Three independent expansions are supported, each gated by its
    count and any combination of them may be requested in a single
    call:

    * **Edge** (inner R, ``n_edge > 0``): prepend ``n_edge`` new R
      samples below ``R_axis[0]``, filled with a constant ceiling
      value per outer-dimension frame. The ceiling is the Z-average
      of the innermost R column of ``field``.
    * **SOL** (outer R, ``n_sol > 0``): append ``n_sol`` new R
      samples above ``R_axis[-1]``, filled with a constant bottom
      value per frame. The bottom is the Z-average of the outermost
      R column of ``field``.
    * **Z periodic stacking** (``n_z > 1``): concatenate ``n_z``
      copies of the field along the Z axis, assuming periodic
      boundary conditions in Z (no duplicated boundary row).

    Both extended axes stay on the original uniform grids
    (``dR = R_axis[1] - R_axis[0]``, ``dZ = Z_axis[1] - Z_axis[0]``).
    The defaults ``n_z=1, n_sol=0, n_edge=0`` make the call a no-op
    that simply returns the inputs coerced to arrays.

    Parameters
    ----------
    field : array-like
        The field to expand. Typical layouts are ``(time, Z, R)`` as
        produced by :meth:`HESEL.extract_field`, or 2-D snapshots
        ``(Z, R)`` when the time axis has been collapsed.
    R_axis, Z_axis : array-like
        1-D, uniformly spaced R and Z coordinate axes matching the
        ``r_axis`` and ``z_axis`` dimensions of ``field``.
    n_z : int, default 1
        Number of periodic copies to stack along Z. ``1`` leaves Z
        unchanged.
    n_sol : int, default 0
        Number of SOL samples to append on the high-R side. ``0``
        leaves the high-R end unchanged.
    n_edge : int, default 0
        Number of edge samples to prepend on the low-R side. ``0``
        leaves the low-R end unchanged.
    r_axis : int, default -1
        Axis of ``field`` corresponding to R.
    z_axis : int, default -2
        Axis of ``field`` corresponding to Z.

    Returns
    -------
    field_expanded : numpy.ndarray
        The expanded field. Same dtype as ``field``, with the R axis
        grown by ``n_edge + n_sol`` and the Z axis multiplied by
        ``n_z``.
    R_expanded : numpy.ndarray
        The extended R axis (ascending), length
        ``R_axis.size + n_edge + n_sol``.
    Z_expanded : numpy.ndarray
        The extended Z axis, length ``n_z * Z_axis.size``.
    """
    for name, n in (("n_z", n_z), ("n_sol", n_sol), ("n_edge", n_edge)):
        if not isinstance(n, (int, np.integer)):
            raise ValueError(f"{name} must be an integer; got {n!r}.")
    if n_z < 1:
        raise ValueError(f"n_z must be >= 1; got {n_z!r}.")
    if n_sol < 0 or n_edge < 0:
        raise ValueError(
            f"n_sol and n_edge must be >= 0; got n_sol={n_sol}, n_edge={n_edge}."
        )

    field = np.asarray(field)
    R_axis = np.asarray(R_axis)
    Z_axis = np.asarray(Z_axis)

    def _radial_extend(field, r_index, count):
        """Build a Z-averaged constant extension of ``count`` R samples
        taken at ``r_index`` (an integer R index into ``field``)."""
        slc = [slice(None)] * field.ndim
        slc[r_axis] = slice(r_index, r_index + 1) if r_index >= 0 else slice(r_index, None)
        column = field[tuple(slc)]
        fill = column.mean(axis=z_axis, keepdims=True)
        ext_shape = list(field.shape)
        ext_shape[r_axis] = count
        return np.broadcast_to(fill, ext_shape)

    if n_edge > 0:
        edge_ext = _radial_extend(field, 0, n_edge)
        field = np.concatenate([edge_ext, field], axis=r_axis)

    if n_sol > 0:
        sol_ext = _radial_extend(field, -1, n_sol)
        field = np.concatenate([field, sol_ext], axis=r_axis)

    if n_z > 1:
        field = np.concatenate([field] * n_z, axis=z_axis)

    dR = R_axis[1] - R_axis[0]
    R_expanded = R_axis
    if n_edge > 0:
        R_expanded = np.concatenate(
            [R_axis[0] - dR * np.arange(n_edge, 0, -1), R_expanded]
        )
    if n_sol > 0:
        R_expanded = np.concatenate(
            [R_expanded, R_axis[-1] + dR * np.arange(1, n_sol + 1)]
        )

    if n_z > 1:
        L = Z_axis.size * (Z_axis[1] - Z_axis[0])
        Z_expanded = np.concatenate([Z_axis + i * L for i in range(n_z)])
    else:
        Z_expanded = Z_axis

    return field, R_expanded, Z_expanded
