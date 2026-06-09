import h5py as h5


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
