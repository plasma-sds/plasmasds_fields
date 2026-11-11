"""
Reading and extending HESEL simulation output.
"""
import h5py as h5
import numpy as np


class HESEL:
    """
    Data handler for HESEL simulation output stored in large HDF5 files.

    On construction the R, Z and time axes are loaded from the
    ``data/xanimation`` group and their grid spacings are derived. The HDF5
    file stays open as ``self.file``, so field datasets can be read lazily,
    section by section, with :meth:`extract_field`.

    The handler can be used as a context manager::

        with HESEL("run.h5") as hesel:
            ...

    or closed explicitly with :meth:`close`.

    Parameters
    ----------
    path : str or path-like
        Path to the HESEL HDF5 output file.

    Attributes
    ----------
    path : str
        Path of the opened file.
    file : h5py.File or None
        Open HDF5 file (read-only); None after :meth:`close`.
    R_axis, Z_axis, time_axis : numpy.ndarray
        1-D R [m], Z [m] and time [s] axes, read from
        ``data/xanimation/xgrid[0, :]``, ``data/xanimation/ygrid[:, 0]`` and
        ``data/xanimation/time``.
    dR, dZ, dt : float
        Step between the first two samples of each axis (the grid spacing for
        uniform axes).
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
        """
        Return the step between the first two samples of a 1-D axis.
        """
        return float(axis[1] - axis[0])

    @staticmethod
    def _range_to_selector(axis, rng):
        """
        Translate a range into an h5py selector for the 1-D ``axis``.

        * None: ``slice(None)``, the full axis.
        * Scalar: index of the nearest sample (collapses the axis).
        * ``[min, max]``: ``slice(i_lo, i_hi + 1)`` covering the samples
          nearest to ``min`` and ``max``.
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
        Read a section of a HESEL field from ``data/xanimation/<field>``.

        The dataset is 3-D with shape (time, Z, R). Only the requested section
        is read from disk.

        Parameters
        ----------
        field : {"density", "electron_temperature", "ion_temperature"}
            Name of the dataset to read.
        r_range, z_range, t_range : float or sequence of float, default None
            Selection along R [m], Z [m] and time [s]:

            * None: keep the whole axis.
            * Scalar: keep only the nearest sample; the axis is dropped from
              the result.
            * Two-element ``[min, max]``: keep the contiguous block of samples
              between the samples nearest to the endpoints.

        Returns
        -------
        numpy.ndarray
            The selected section, with axes in the order (time, Z, R), minus
            the axes selected by a scalar.

        Raises
        ------
        ValueError
            If ``field`` is not a valid dataset name, or if a range has more
            than two elements.
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
        """
        Close the HDF5 file. Safe to call more than once.
        """
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
    Expand a HESEL field along R and/or Z.

    Three independent expansions can be combined in one call:

    * **Edge** (``n_edge > 0``): prepend ``n_edge`` R samples below
      ``R_axis[0]``, filled per frame with the Z-average of the innermost R
      column of ``field``.
    * **SOL** (``n_sol > 0``): append ``n_sol`` R samples above ``R_axis[-1]``,
      filled per frame with the Z-average of the outermost R column.
    * **Periodic Z stacking** (``n_z > 1``): concatenate ``n_z`` copies of the
      field along Z, assuming periodic boundary conditions in Z.

    The extended axes keep the original grid spacings. With the defaults
    ``n_z=1, n_sol=0, n_edge=0`` the inputs are returned unchanged (as arrays).

    Parameters
    ----------
    field : array_like
        Field to expand, e.g. (time, Z, R) from :meth:`HESEL.extract_field` or
        a 2-D (Z, R) snapshot.
    R_axis, Z_axis : array_like
        1-D, uniformly spaced R and Z axes [m] matching the ``r_axis`` and
        ``z_axis`` dimensions of ``field``.
    n_z : int, default 1
        Number of periodic copies along Z; 1 leaves Z unchanged.
    n_sol : int, default 0
        Number of SOL samples appended on the high-R side.
    n_edge : int, default 0
        Number of edge samples prepended on the low-R side.
    r_axis : int, default -1
        Axis of ``field`` corresponding to R.
    z_axis : int, default -2
        Axis of ``field`` corresponding to Z.

    Returns
    -------
    field_expanded : numpy.ndarray
        The expanded field, with the R axis longer by ``n_edge + n_sol`` and
        the Z axis ``n_z`` times longer.
    R_expanded : numpy.ndarray
        Extended R axis (ascending), length ``R_axis.size + n_edge + n_sol``
        [m].
    Z_expanded : numpy.ndarray
        Extended Z axis, length ``n_z * Z_axis.size`` [m].

    Raises
    ------
    ValueError
        If ``n_z``, ``n_sol`` or ``n_edge`` is not an integer, if ``n_z`` < 1,
        or if ``n_sol`` or ``n_edge`` is negative.
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
        """
        Return a constant R extension of ``count`` samples.

        It is filled with the Z-average of the R column at index ``r_index``.
        """
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
