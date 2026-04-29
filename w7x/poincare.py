import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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

def box_plot_coordinates(r_min, z_min, r_max, z_max):
    """
    Build the polyline coordinates of an axis-aligned rectangle.

    The returned arrays describe the four corners of the rectangle in the
    (R, z) plane, closed back to the starting point so that
    ``ax.plot(r, z)`` draws a complete outline.

    Parameters
    ----------
    r_min, z_min : float
        Lower-left corner of the box.
    r_max, z_max : float
        Upper-right corner of the box.

    Returns
    -------
    r, z : numpy.ndarray
        Five-element arrays of corner coordinates, with the first point
        repeated at the end to close the rectangle.
    """
    r = np.array([r_min, r_max, r_max, r_min, r_min])
    z = np.array([z_min, z_min, z_max, z_max, z_min])
    return r, z

def plot_w7x_flux_surfaces(surfaces, magnetic_conf='', r_range=None, z_range=None, phi=np.nan, boxes=None, aspect=False, save_image=False):
    """
    Plot a Poincaré section of W7-X flux surfaces in the (R, z) plane.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances (as returned by
        :func:`load_w7x_flux_surfaces`) or a field-line-tracer result
        object exposing ``poincare_res.surfs``.
    magnetic_conf : str, optional
        Name of the applied magnetic configuration, used in the title.
    r_range, z_range : list of float, optional
        Two-element ``[min, max]`` lists used to set the R and z axis
        limits. If ``None`` (default), matplotlib chooses automatically.
    s_range : optional
        Reserved for future use.
    phi : float, optional
        Toroidal angle (in radians) at which the Poincaré section was
        generated. Used in the title.
    boxes : iterable of 4-tuples, optional
        Box coordinates ``(x0, y0, x1, y1)`` to be overlaid on the plot.
    aspect : bool, optional
        If ``True``, use an equal aspect ratio. Otherwise, use the
        matplotlib default.
    save_image : bool or str, optional
        If ``False`` (default), the figure is not saved to disk. Otherwise
        the value is interpreted as the destination filename and the
        figure is written as a PNG (a ``.png`` extension is appended if
        missing).
    """
    if not isinstance(surfaces, list):
        surfaces = surfaces.poincare_res.surfs

    colors = ["red", "orange", "yellow", "green", "blue", "purple", "indigo", "violet"]
    fig, ax = plt.subplots()

    num_surfaces = len(surfaces)

    for i, surface in enumerate(surfaces):
        if surface.points.x1 is not None and len(surface.points.x1) > 0:
            r = np.sqrt(np.array(surface.points.x1)**2 + np.array(surface.points.x2)**2)
            z = np.array(surface.points.x3)
            ax.scatter(r, z, color=colors[i % 8], s=0.2, label="surface {}".format(i))
        else:
            print("Surface {} contains no points!".format(i + 1))

    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")

    if isinstance(r_range, list):
        ax.set_xlim(r_range)

    if isinstance(z_range, list):
        ax.set_ylim(z_range)

    if num_surfaces < 30:
        ax.legend()  # too cluttered for many surfaces

    if aspect:
        ax.set_aspect('equal')
    ax.set_title("Config: {}, toroidal angle: {:3.2f} rad, {:3.2f} deg.".format(magnetic_conf, phi, phi/np.pi*180.))

    if boxes:
        for box in boxes:
            r, z = box_plot_coordinates(box[0], box[1], box[2], box[3])
            ax.plot(r, z)

    if save_image:
        save_filename = str(save_image)
        if not save_filename.lower().endswith('.png'):
            save_filename = save_filename + '.png'
        fig.savefig(save_filename, format='png', dpi=300, bbox_inches="tight")

    plt.show()

def filter_surfaces_by_range(flt, surf_range=None, r_range=None, z_range=None):
    """
    Filter flux surfaces and their points by surface index and (R, z) range.

    The input may be either a list of :class:`FluxSurface` instances (as
    returned by :func:`load_w7x_flux_surfaces`) or a field-line-tracer
    result object exposing ``poincare_res.surfs``. The function first
    selects a contiguous slice of surfaces (``surf_range``), then for each
    remaining surface keeps only the points whose cylindrical
    ``R = sqrt(x1**2 + x2**2)`` and ``z = x3`` fall within the supplied
    ranges. Surfaces left with no points are dropped.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Source surfaces to filter.
    surf_range : sequence of int, optional
        Two-element ``[start, end]`` slice (Python half-open semantics)
        applied to the list of surfaces before point filtering. Bounds
        are clamped to the valid range. If ``None`` (default), all
        surfaces are kept.
    r_range : sequence of float, optional
        Two-element ``[min, max]`` interval (in metres) on the major
        radius ``R``. If ``None`` (default), no constraint on ``R``.
    z_range : sequence of float, optional
        Two-element ``[min, max]`` interval (in metres) on the vertical
        coordinate ``z``. If ``None`` (default), no constraint on ``z``.

    Returns
    -------
    list of FluxSurface
        New :class:`FluxSurface` instances containing only the points
        that pass both the ``R`` and ``z`` filters. ``phi0``, ``density``
        and ``assymetric_island_density`` are copied from the originals.
        Surfaces that end up empty (or that started empty / had ``None``
        coordinates) are omitted from the result.
    """
    if not isinstance(flt, list):
        surfs = flt.poincare_res.surfs
    else:
        surfs = flt
    
    # Apply surface range filter first
    if surf_range is not None:
        num_surfs = len(surfs)
        start_idx = max(0, surf_range[0])
        end_idx = min(num_surfs, surf_range[1])
        surfs = surfs[start_idx:end_idx]
    
    filtered_surfs = []
    
    for surf in surfs:
        if type(surf.points.x1) != type(None) and len(surf.points.x1) > 0:
            # Calculate R and Z coordinates
            r_temp = np.sqrt(np.array(surf.points.x1)**2 + np.array(surf.points.x2)**2)
            z_temp = np.array(surf.points.x3)
            
            # Create masks for filtering
            if r_range is not None:
                r_mask = (r_temp >= r_range[0]) & (r_temp <= r_range[1])
            else:
                r_mask = np.ones(len(r_temp), dtype=bool)
            
            if z_range is not None:
                z_mask = (z_temp >= z_range[0]) & (z_temp <= z_range[1])
            else:
                z_mask = np.ones(len(z_temp), dtype=bool)
            
            # Keep only points within both ranges
            mask = r_mask & z_mask
            
            # Only include surface if it has at least one point within the ranges
            if np.any(mask):
                # Filter the points
                x1_filtered = np.array(surf.points.x1)[mask].tolist()
                x2_filtered = np.array(surf.points.x2)[mask].tolist()
                x3_filtered = np.array(surf.points.x3)[mask].tolist()
                
                # Create new filtered surface
                filtered_surf = FluxSurface(x1_filtered, x2_filtered, x3_filtered, surf.phi0, surf.density, surf.assymetric_island_density)
                filtered_surfs.append(filtered_surf)
    
    return filtered_surfs