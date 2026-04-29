import numpy as np
import xml.etree.ElementTree as ET

class Points(object):
    """
    Container for a set of 3D points stored as three coordinate arrays.

    The coordinates ``x1``, ``x2`` and ``x3`` are stored as NumPy arrays of
    equal length, where the i-th point is given by
    ``(x1[i], x2[i], x3[i])``.
    """

    def __init__(self, x1, x2, x3):
        """
        Initialize a ``Points`` instance from three coordinate sequences.

        Parameters
        ----------
        x1, x2, x3 : array_like
            Sequences (or arrays) of equal length holding the first, second
            and third coordinate of each point. They are converted to
            ``numpy.ndarray`` on assignment.
        """
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)

    def __array__(self, dtype=None):
        """
        NumPy array protocol: return the points as a stacked array.

        Allows the object to be used directly with ``np.asarray`` /
        ``np.array``.

        Parameters
        ----------
        dtype : data-type, optional
            If given, the returned array is cast to this dtype.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(3, N)`` whose rows are ``x1``, ``x2`` and
            ``x3`` respectively.
        """
        arr = np.array([self.x1, self.x2, self.x3])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


class FluxSurface(object):
    """
    Representation of a single magnetic flux surface.

    A flux surface is described by a cloud of points (stored in
    :class:`Points`), the toroidal reference angle ``phi0`` at which the
    surface was sampled, and optional plasma density attributes used by
    downstream models.
    """

    def __init__(self, x1, x2, x3, phi0, density=0, assymetric_island_density=0):
        """
        Initialize a flux surface.

        Parameters
        ----------
        x1, x2, x3 : array_like
            Coordinate sequences describing the points on the surface.
        phi0 : float
            Reference toroidal angle (in radians) at which the surface was
            sampled.
        density : float, optional
            Bulk plasma density associated with the surface. Defaults to 0.
        assymetric_island_density : float, optional
            Additional density contribution from an asymmetric magnetic
            island, if any. Defaults to 0.
        """
        self.points = Points(x1, x2, x3)
        self.phi0 = phi0
        self.density = density
        self.assymetric_island_density = assymetric_island_density

    def update_density(self, value, assymetric_island_density=None):
        """
        Update the (bulk and optional island) density of the surface.

        Parameters
        ----------
        value : float
            New bulk density value.
        assymetric_island_density : float, optional
            If given, also overwrite the asymmetric island density. If
            ``None`` (default), the island density is left unchanged.
        """
        self.density = value
        if assymetric_island_density is not None:
            self.assymetric_island_density = assymetric_island_density

def load_w7x_flux_surfaces(filename):
    """
    Load W7-X flux surfaces from an XML file produced by the field-line
    tracer.

    The XML is expected to contain one entry per flux surface, each with a
    ``phi0`` element and a ``points`` block listing ``x1``, ``x2`` and
    ``x3`` coordinate values.

    Parameters
    ----------
    filename : str or path-like
        Path to the XML file to parse.

    Returns
    -------
    list of FluxSurface
        One :class:`FluxSurface` per surface found in the file, in the
        order they appear.
    """
    tree = ET.parse(filename)
    root = tree.getroot()  # {fltracer.gsoap.boz.hgw.ipp.mpg.de}Result {}
    surfaces = list()

    for surf in root:
        phi0 = None
        x1 = list()
        x2 = list()
        x3 = list()
        for points in surf:
            if 'points' in points.tag:
                for point in points:
                    if 'x1' in point.tag:
                        x1.append(float(point.text))
                    elif 'x2' in point.tag:
                        x2.append(float(point.text))
                    elif 'x3' in point.tag:
                        x3.append(float(point.text))
            elif phi0 is None and 'phi0' in points.tag:
                phi0 = float(points.text)
        surface = FluxSurface(x1, x2, x3, phi0)
        surfaces.append(surface)

    return surfaces