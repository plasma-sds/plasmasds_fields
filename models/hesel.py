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
        self.time_axis = self.file["data/xanimation/time"][:]

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


def expand_hesel(field, Z_axis, n, axis=-2):
    """
    Vertically stack a HESEL field ``n`` times along the Z direction.

    Assumes the field is periodic along ``axis`` so that ``n`` copies
    can be concatenated end-to-end without overlap, producing ``n``
    adjacent periods of the underlying field. The matching Z axis is
    extended on the same uniform grid (period ``L = N * dZ``, where
    ``N = Z_axis.size`` and ``dZ = Z_axis[1] - Z_axis[0]``).

    Parameters
    ----------
    field : array-like
        The field to expand. Typical layouts are ``(time, Z, R)`` as
        produced by :meth:`HESEL.extract_field`, or 2-D snapshots
        ``(Z, R)`` when the time axis has been collapsed.
    Z_axis : array-like
        1-D, uniformly spaced Z coordinate axis matching the ``axis``
        dimension of ``field``.
    n : int
        Number of copies to stack. Must be a positive integer.
    axis : int, default -2
        Axis along which to stack. The default ``-2`` corresponds to
        the Z dimension for both ``(time, Z, R)`` and ``(Z, R)``
        arrays.

    Returns
    -------
    field_expanded : numpy.ndarray
        The tiled field. Same dtype as ``field`` and same shape except
        the length of ``axis`` is multiplied by ``n``.
    Z_expanded : numpy.ndarray
        The extended Z axis, of length ``n * Z_axis.size`` and the
        same spacing as ``Z_axis``.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}.")
    field = np.asarray(field)
    Z_axis = np.asarray(Z_axis)
    if n == 1:
        return field, Z_axis

    field_expanded = np.concatenate([field] * n, axis=axis)

    N = Z_axis.size
    L = N * (Z_axis[1] - Z_axis[0])
    Z_expanded = np.concatenate([Z_axis + i * L for i in range(n)])

    return field_expanded, Z_expanded
