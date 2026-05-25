import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class Points(object):
    """
    Container for a set of 3D points stored as three coordinate arrays.

    The coordinates ``x1``, ``x2`` and ``x3`` are stored as NumPy arrays of
    equal length, where the i-th point is given by
    ``(x1[i], x2[i], x3[i])``.
    Further arrays store the radial position 'r', the vertical position 'z',
    the density 'd' and the type of the point 'point_type'. 
    
    r = sqrt(x1^2 + x2^2)
    z = x3
    d = 0 (initial value)
    The type of the point can be three values:
        - self.point_type = 0 -> not island point
        - self.point_type = 1 -> island point at outer surface
        - self.point_type = 2 -> island point at inner surface
    """

    def __init__(self, x1, x2, x3, density=0, point_type=0):
        """
        Initialize a ``Points`` instance from three coordinate sequences.

        Parameters
        ----------
        x1, x2, x3: array_like
            Sequences (or arrays) of equal length holding the first, second
            and third coordinate of each point. They are converted to
            ``numpy.ndarray`` on assignment.
        """
        self.n = len(x1)
        self.x1 = np.array(x1)                      # first coordinate
        self.x2 = np.array(x2)                      # second coordinate
        self.x3 = np.array(x3)                      # third coordinate
        self.r = np.sqrt(self.x1**2 + self.x2**2)   # radial distance
        self.z = self.x3                            # vertical coord
        
        self.d = np.zeros(self.n)
        self.d[:] = density
        
        self.point_type = np.zeros(self.n, dtype=np.uint8)
        self.point_type[:] = point_type

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

    def __init__(self, x1, x2, x3, phi0, 
                 point_type=0, density=0, surf_type="default"):
        """
        Initialize a flux surface.

        Parameters
        ----------
        x1, x2, x3 : array_like
            Coordinate sequences describing the points on the surface.
        phi0 : float
            Reference toroidal angle (in radians) at which the surface was
            sampled.
        density : float or array_like, optional
            Bulk plasma density or density distribution associated 
            with the surface. Defaults to 0.
        point_type : float or array_like, optional
            The type of the points on the surface. 
            Can be three values:
                - self.point_type = 0 -> not island point
                - self.point_type = 1 -> island point at outer surface
                - self.point_type = 2 -> island point at inner surface
            Defaults to 0.
            
        Further attributes:
        N : integer
            number of points in the self.points arrays
        coeffs : list
            list of coefficient for each 'point_type' group
        errors : list
            list of coefficient for each 'point_type' group
        """
        self.points = Points(x1, x2, x3, 
                             density=density, point_type=point_type)
        self.phi0 = phi0
        self.N = np.shape(np.asarray(self.points))[1]
        self.coeffs = [None]
        self.errors = [None]

    def update_density(self, value, mask=None):
        """
        Update the density of the surface.

        Parameters
        ----------
        value : float or array
            New bulk density value or density distribution.
        mask : array, optional
            If given, overwrite only the density values where the mask has 
            'True' value.
        """
        if mask is None: self.points.d[:] = value
        else: self.points.d[mask] = value
        
    def update_point_type(self, value, mask=None):
        """
        Update the type of the surface or points.

        Parameters
        ----------
        value : float or array
            New point type value for all points or each points separately.
        mask : array, optional
            If given, overwrite only the type values where the mask has 
            'True' value.
        """
        if mask is None: self.points.point_type[:] = value
        else: self.points.point_type[mask] = value
        
    def filter_points(self, mask):
        """
        Filter the points based on mask array. 

        Parameters
        ----------
        mask : array
            Remove point elements where the mask has 'False' value.
        """
        for attr, value in vars(self.points).items():
            if isinstance(value, np.ndarray):
                setattr(self.points, attr, value[mask])
        self.N = sum(mask)

class Range(object):
    """ 
    Used to extract the range of different data types in a list of FluxSurface
    objects or FluxSurface.Points objects:
        x1, x2, x3, r, z, density
        
        Attributes can give back the individual boundaries, or the list object
        of the range.
    """
    
    def __init__(self, flt):
        """
        By creating the Range object, the attributes calculated automatically
        during the initialisation.

        Parameters
        ----------
        flt : list of FluxSurface objects
            Filtered list of FluxSurface objects.

        """
        
        self.x1Min = min([np.min(surf.points.x1) for surf in flt])
        self.x1Max = max([np.max(surf.points.x1) for surf in flt])
        self.x2Min = min([np.min(surf.points.x2) for surf in flt])
        self.x2Max = max([np.max(surf.points.x2) for surf in flt])
        self.x3Min = min([np.min(surf.points.x3) for surf in flt])
        self.x3Max = max([np.max(surf.points.x3) for surf in flt])
        self.rMin  = min([np.min(surf.points.r ) for surf in flt])
        self.rMax  = max([np.max(surf.points.r ) for surf in flt])
        self.dMin  = min([np.min(surf.points.d ) for surf in flt])
        self.dMax  = max([np.max(surf.points.d ) for surf in flt])
        self.zMin  = self.x3Min
        self.zMax  = self.x3Max
        
        self.x1 = [self.x1Min, self.x1Max]
        self.x2 = [self.x2Min, self.x2Max]
        self.x3 = [self.x3Min, self.x3Max]
        self.r  = [self.rMin , self.rMax ]
        self.d  = [self.dMin , self.dMax ]
        self.z  = [self.zMin , self.zMax ]

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

def plot_w7x_flux_surfaces(surfaces, magnetic_conf='', 
                           r_range=None, z_range=None, phi=np.nan, 
                           boxes=None, aspect=False, 
                           save_image=False, legend=False):
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

    colors = ["red", "orange", "yellow", "green", "blue",
              "purple", "indigo", "violet"]
    fig, ax = plt.subplots()

    num_surfaces = len(surfaces)

    for i, surface in enumerate(surfaces):
        if surface.points.x1 is not None and len(surface.points.x1) > 0:
            r = surface.points.r
            z = surface.points.z
            ax.scatter(r, z, color=colors[i % 8], s=0.2, 
                       label="surface {}".format(i))
        else:
            print("Surface {} contains no points!".format(i + 1))

    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")

    if isinstance(r_range, list):
        ax.set_xlim(r_range)

    if isinstance(z_range, list):
        ax.set_ylim(z_range)

    if legend:
        ax.legend()  

    if aspect:
        ax.set_aspect('equal')
    ax.set_title(("Config: {}, toroidal angle: {:3.2f} rad, {:3.2f} deg."
                  ).format(magnetic_conf, phi, phi/np.pi*180.))

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
        and ``point_type`` are copied from the originals.
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
            r_temp = surf.points.r
            z_temp = surf.points.z
            
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
            
            # Only include surface if it has at least one point
            # within the ranges
            if np.any(mask):
                # Filter the points
                surf.filter_points(mask)
                filtered_surfs.append(surf)
    
    return filtered_surfs

def filter_surfaces_by_polyfit(flt, limit_error = 0.015,
                               limit_number = 100, order = 2):
    """
    Filter flux surfaces and their points by polyfit. Used to differentiate
    island flux surfaces. The method includes a polinom fit (ideally second
    order), which error can indicate the flux surface relevance. The 0 order
    coefficient can be also used to estimate the radial position of the 
    flux surface.
    
    
    Based on the polinom fit each points in a Flux Surface object
    are sorted into group of 'point_type':
        - surf.points.point_type[i] = 0 -> not island point
        - surf.points.point_type[i] = 1 -> island point at outer surface
        - surf.points.point_type[i] = 2 -> island point at inner surface
    Each group has a polinom fit function. The coefficient arrays and errors 
    related to every 'point_type' groups are added to a list (surf.coeffs 
    and surf.errors). Therefore these lists contain three element, one for 
    each 'point_type' group. If there is no point with a certain point_type
    in the surf object and no polinom fit is possible, a 'None' element
    is added to the list.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Source surfaces to filter.
    limit_error : float, optional
        Limit defined by the error of the polynomfit. 
        The default is 0.015.
    limit_number : float, optional
        Limit defined by the number of points in the FluxSurface object.
        The default is 100.
    order : integer, optional
        order of the polynom fit function. The default is 2.

    Returns
    ------- , , 
    surfaces : list of FluxSurface
        Returns the list of flux surfaces, with updated point_type for each 
        'Points' object attribute.
        
        Two more attributes are added: 
            - surf.coeffs: list, contains coefficient of polinom fit for each 
            'point_type' group.
            - surf.errors: list, contains the error of polinom fit for each 
            'point_type' group.
    """
    
    Surfs = list()
    
    for i, surf in enumerate(flt):
        print(surf.N)
        r, z = surf.points.r, surf.points.z
        coeff = np.polyfit(z[:], r[:], order)
        error = np.sqrt(np.mean((r - np.polyval(coeff, z))**2))
        
        
        # Conditions to differentiate island and not-island surfaces:
        # Error is high | or | the number of points are large
        condition_1 = error > limit_error
        condition_2 = surf.N > limit_number
        if (condition_1 or condition_2):
            
            
            # island case: split surface to low and high field side
            p = np.polyval(coeff, z)    # center line by polyfit
            o = r-p > 0                 # outer side point mask
            i = np.invert(o)            # inner side point mask
            
            # Outer side
            surf.update_point_type(1, mask=o)
            coeff_o = np.polyfit(z[o], r[o], order)
            error_o = np.sqrt(np.mean((r - np.polyval(coeff_o, z))**2))
            
            # Inner side
            surf.update_point_type(2, mask=i)
            coeff_i = np.polyfit(z[i], r[i], order)
            error_i = np.sqrt(np.mean((r - np.polyval(coeff_i, z))**2))
            
            surf.coeffs = [np.zeros(order+1), coeff_o, coeff_i]
            surf.errors = [None, error_o, error_i]
            
            Surfs.append(surf)
            
        else:
            # not-island case
            # Leave all surfaces as they are.
            surf.coeffs = [coeff, np.zeros(order+1), np.zeros(order+1)]
            surf.errors = [error, None, None]
            Surfs.append(surf)
        
    return Surfs

def plot_w7x_regimes(surfaces, labels, magnetic_conf='',
                     r_range=None, z_range=None, phi=np.nan, 
                     boxes=None, aspect=False, save_image=False, legend=False):
    """
    Plot a Poincaré section of W7-X flux surfaces in the (R, z) plane.
    Highlight different regimes based on the types array

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances (as returned by
        :func:`load_w7x_flux_surfaces`) or a field-line-tracer result
        object exposing ``poincare_res.surfs``.
    labels : list of strings
        contains the expression for each type - used in legend
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

    colors = ["red", "orange", "yellow", "green", "blue",
              "purple", "indigo", "violet"]
    fig, ax = plt.subplots()


    for i, surface in enumerate(surfaces):
        if surface.points.x1 is not None and len(surface.points.x1) > 0:
            r = surface.points.r
            z = surface.points.z
            c = [colors[i] for i in surface.points.point_type]
            ax.scatter(r, z, color=c, s=0.2)
        else:
            print("Surface {} contains no points!".format(i + 1))
    
    for i in range(len(labels)): plt.scatter([0],[0], s=50, 
                                   c=colors[i], label=labels[i])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")

    if isinstance(r_range, list):
        ax.set_xlim(r_range)

    if isinstance(z_range, list):
        ax.set_ylim(z_range)

    if legend:
        ax.legend()  

    if aspect:
        ax.set_aspect('equal')
    ax.set_title(("Config: {}, toroidal angle: {:3.2f} rad, {:3.2f} deg."
                  ).format(magnetic_conf, phi, phi/np.pi*180.))

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