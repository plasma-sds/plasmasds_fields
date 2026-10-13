"""
W7-X Poincaré flux surfaces: loading, filtering, labelling and plotting of
field-line tracer results.
"""
import copy
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class Points(object):
    """
    Set of 3-D points stored as coordinate arrays.

    The i-th point is ``(x1[i], x2[i], x3[i])``. The cylindrical radius ``r =
    sqrt(x1**2 + x2**2)`` and the vertical position ``z = x3`` are derived on
    construction.

    Parameters
    ----------
    x1, x2, x3 : array_like
        First, second and third coordinate of each point [m], of equal length.
    point_type : int or array_like, default 0
        Type of each point: 0 non-island, 1 island outer side, 2 island inner
        side.

    Attributes
    ----------
    x1, x2, x3 : numpy.ndarray
        Point coordinates [m].
    r : numpy.ndarray
        Cylindrical radius of each point [m].
    z : numpy.ndarray
        Vertical position of each point [m].
    point_type : numpy.ndarray of uint8
        Type of each point (see ``point_type`` above).
    """

    def __init__(self, x1, x2, x3, point_type=0):
        n = len(x1)
        self.x1 = np.array(x1)                      # first coordinate
        self.x2 = np.array(x2)                      # second coordinate
        self.x3 = np.array(x3)                      # third coordinate
        self.r = np.sqrt(self.x1**2 + self.x2**2)   # radial distance
        self.z = self.x3                            # vertical coord
        
        self.point_type = np.zeros(n, dtype=np.uint8)
        self.point_type[:] = point_type

    def __array__(self, dtype=None):
        """
        Return the points as a stacked ``(3, N)`` array (NumPy array protocol).

        Allows the object to be used directly with ``np.asarray`` /
        ``np.array``.

        Parameters
        ----------
        dtype : data-type, default None
            If given, the returned array is cast to this dtype.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(3, N)`` whose rows are ``x1``, ``x2`` and ``x3``.
        """
        
        arr = np.array([self.x1, self.x2, self.x3])
        if dtype is not None:
            arr = arr.astype(dtype)
            
        return arr

class FluxSurface(object):
    """
    Single magnetic flux surface, sampled as points in a poloidal plane.

    Parameters
    ----------
    x1, x2, x3 : array_like
        Coordinates of the points on the surface [m].
    phi0 : float
        Toroidal angle at which the surface was sampled [rad].
    point_type : int or array_like, default 0
        Type of each point: 0 non-island, 1 island outer side, 2 island inner
        side.
    density : float, default 0
        Initial density of the surface [m^-3].
    surf_type : str, default "default"
        Currently not used.

    Attributes
    ----------
    points : Points
        The points of the surface.
    phi0 : float
        Toroidal angle of the poloidal plane [rad].
    n : int
        Number of points.
    coeffs : list or numpy.ndarray
        Polynomial fit coefficients of R(Z), one row per point type, set by
        :func:`label_surfaces` (``[None]`` before labelling).
    errors : list
        RMS fit error per point type [m], None for point types without points
        (``[None]`` before labelling).
    density : list of float
        Density per point type [m^-3], set by :func:`w7x.fields.add_density`.
    """

    def __init__(self, x1, x2, x3, phi0, 
                 point_type=0, density=0, surf_type="default"):
        self.points = Points(x1, x2, x3, 
                             point_type=point_type)
        self.phi0 = phi0
        self.n = np.shape(np.asarray(self.points))[1]
        self.coeffs = [None]
        self.errors = [None]
        self.density = [density]

    def update_density(self, value, ind=None):
        """
        Set the density of one point-type group or of all groups.

        Parameters
        ----------
        value : float
            New density [m^-3].
        ind : int, default None
            Index of the point-type group to update. If None, all groups are
            set to ``value``.
        """
        if ind != None: self.density[ind] = value
        else: self.density = [value for i in range(len(self.density))]
        
    def update_point_type(self, value, mask=None):
        """
        Set the point type of all points or of the masked points.

        Parameters
        ----------
        value : int or array_like
            New point type(s).
        mask : array_like of bool, default None
            If given, only the points where ``mask`` is True are updated.
        """
        
        if mask is None: self.points.point_type[:] = value
        else: self.points.point_type[mask] = value
        
    def filter_points(self, mask):
        """
        Keep only the points where ``mask`` is True (in place).

        Parameters
        ----------
        mask : array_like of bool
            Points to keep.
        """
        
        for attr, value in vars(self.points).items():
            if isinstance(value, np.ndarray):
                setattr(self.points, attr, value[mask])
        self.n = sum(mask)

class Range(object):
    """
    Extent of the coordinates and densities of a list of flux surfaces.

    Parameters
    ----------
    flt : list of FluxSurface
        Surfaces to evaluate.

    Attributes
    ----------
    x1Min, x1Max, x2Min, x2Max, x3Min, x3Max : float
        Extent of the point coordinates [m].
    rMin, rMax, zMin, zMax : float
        Extent of the cylindrical radius and the vertical position [m].
    densityMin, densityMax : float
        Extent of the densities of the point-type groups that contain points
        [m^-3].
    x1, x2, x3, r, z, density : list of float
        The same extents as ``[min, max]`` pairs.
    """
    
    def __init__(self, flt):
        self.x1Min = min([np.min(surf.points.x1) for surf in flt])
        self.x1Max = max([np.max(surf.points.x1) for surf in flt])
        self.x2Min = min([np.min(surf.points.x2) for surf in flt])
        self.x2Max = max([np.max(surf.points.x2) for surf in flt])
        self.x3Min = min([np.min(surf.points.x3) for surf in flt])
        self.x3Max = max([np.max(surf.points.x3) for surf in flt])
        self.rMin  = min([np.min(surf.points.r ) for surf in flt])
        self.rMax  = max([np.max(surf.points.r ) for surf in flt])
        dens = [d for surf in flt for d, e in zip(surf.density, surf.errors)
                if e is not None or surf.errors == [None]]
        self.densityMin  = min(dens)
        self.densityMax  = max(dens)
        self.zMin  = self.x3Min
        self.zMax  = self.x3Max
        
        self.x1 = [self.x1Min, self.x1Max]
        self.x2 = [self.x2Min, self.x2Max]
        self.x3 = [self.x3Min, self.x3Max]
        self.r  = [self.rMin , self.rMax ]
        self.density  = [self.densityMin , self.densityMax ]
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
    root = tree.getroot()  # field-line tracer <Result> element
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

def _check_surfaces(flt):
    """
    Raise TypeError unless ``flt`` holds a list of FluxSurface instances.
    """
    
    if isinstance(flt, list): surfs = flt
    elif hasattr(flt, "poincare_res") and hasattr(flt.poincare_res, "surfs"):
        surfs = flt.poincare_res.surfs
        if not isinstance(surfs, list):
            raise TypeError(
                "flt.poincare_res.surfs must be a list of FluxSurface," 
                f"instances, got {type(surfs).__name__}." )
    else:
        raise TypeError(
            "flt must be a list of FluxSurface instances or an object "
            f"exposing 'poincare_res.surfs'; got {type(flt).__name__}." )
    
    if not all(type(s).__name__ == "FluxSurface" for s in surfs):
        bad_idx = next(i for i, s in enumerate(surfs) 
                       if not type(s).__name__ == "FluxSurface")
        raise TypeError(
            f"All elements must be FluxSurface instances; "
            f"element {bad_idx} is {type(surfs[bad_idx]).__name__}." )

def _check_labelled(surfs):
    """
    Raise ValueError if any surface has not been labelled.
    """
    if any(s.errors == [None] for s in surfs):
        raise ValueError("Surfaces are not labelled; run "
                         "label_surfaces() first.")

def _check_scalar(name, value, integer=False, min_value=None, max_value=None,
                  inclusive_min=True, inclusive_max=True, domain=None):
    """
    Validate the type, allowed values and bounds of a scalar argument.

    None is accepted and skips all checks.
    """
    
    if value is None: return

    # --- type check ---
    if integer:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer, got {value!r}.")
    else:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{name} must be numeric, got {value!r}.")

    # --- domain check ---
    if domain is not None and value not in domain:
        raise ValueError(
            f"{name} must be one of {sorted(domain)!r}, got {value!r}.")
    
    # --- bounds check ---
    if min_value is not None:
        ok = value >= min_value if inclusive_min else value > min_value
        if not ok:
            op = ">=" if inclusive_min else ">"
            raise ValueError(
                f"{name} must be {op} {min_value}, got {value!r}.")

    if max_value is not None:
        ok = value <= max_value if inclusive_max else value < max_value
        if not ok:
            op = "<=" if inclusive_max else "<"
            raise ValueError(
                f"{name} must be {op} {max_value}, got {value!r}.")

def _check_range(name, value, integer=False, allow_negative=True):
    """
    Validate an optional two-element ``[start, end]`` range argument.
    """
    if value is None: return

    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(
            f"{name} must be a two-element sequence "
            f"[start, end], got {value!r}.")

    lo, hi = value

    if integer:
        if not all(isinstance(v, (int, np.integer)) 
                   and not isinstance(v, bool)
                   for v in (lo, hi)):
            raise TypeError(
                f"{name} elements must be integers, got {value!r}.")
    else:
        if not all(isinstance(v, (int, float, np.integer, np.floating)) 
                   and not isinstance(v, bool)
                   for v in (lo, hi)):
            raise TypeError(
                f"{name} elements must be numeric, got {value!r}.")

    if not allow_negative and (lo < 0 or hi < 0):
        raise ValueError(
            f"{name} elements must be >= 0, got {value!r}.")

    if lo >= hi:
        raise ValueError(
            f"{name} must satisfy start < end, got {value!r}." )

def _check_sequence(name, value, integer=False, min_value=None, max_value=None,
                    inclusive_min=True, inclusive_max=True, domain=None, 
                    min_length=None, max_length=None,
                    length=None, seq_types=(list, tuple, np.ndarray)):
    """
    Validate a sequence argument, checking every element against the
    same conditions as :func:`_check_scalar`.
    """
    
    if value is None: return

    if not isinstance(value, seq_types):
        raise TypeError(
            f"{name} must be a sequence ({[t.__name__ for t in seq_types]}), "
            f"got {type(value).__name__}." )

    n = len(value)
    if length is not None and n != length:
        raise ValueError(
            f"{name} must have exactly {length} elements, got {n}.")

    if min_length is not None and n < min_length: 
        raise ValueError(
            f"{name} must have at least {min_length} elements, got {n}.")

    if max_length is not None and n > max_length:
        raise ValueError(
            f"{name} must have at most {max_length} elements, got {n}.")

    for i, elem in enumerate(value):
        try:
            _check_scalar(
                f"{name}[{i}]", elem,
                integer=integer, min_value=min_value, max_value=max_value,
                inclusive_min=inclusive_min, inclusive_max=inclusive_max,
                domain=domain)
        except (TypeError, ValueError) as e:
            raise type(e)(f"Invalid element in {name}: {e}") from e

def box_plot_coordinates(r_min, z_min, r_max, z_max):
    """
    Build the polyline coordinates of an axis-aligned rectangle.

    The returned arrays describe the four corners of the rectangle in the (R,
    Z) plane, closed back to the starting point so that ``ax.plot(r, z)`` draws
    a complete outline.

    Parameters
    ----------
    r_min, z_min : float
        Lower-left corner of the box [m].
    r_max, z_max : float
        Upper-right corner of the box [m].

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
    Plot a Poincaré section of W7-X flux surfaces in the (R, Z) plane.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances (as returned by
        :func:`load_w7x_flux_surfaces`) or a field-line-tracer result object
        exposing ``poincare_res.surfs``.
    magnetic_conf : str, default ''
        Name of the magnetic configuration, used in the title.
    r_range, z_range : sequence of float, default None
        Two-element ``[min, max]`` axis limits [m]. If None, matplotlib chooses
        the limits.
    phi : float, default nan
        Toroidal angle of the Poincaré section [rad], used in the title.
    boxes : iterable of 4-tuples, default None
        Boxes ``(r_min, z_min, r_max, z_max)`` [m] to overlay on the plot.
    aspect : bool, default False
        If True, use an equal aspect ratio.
    save_image : bool or str, default False
        If False, the figure is not saved. Otherwise the value is the
        destination filename and the figure is written as a PNG (``.png`` is
        appended if missing).
    legend : bool, default False
        If True, show a legend with one entry per surface.

    Raises
    ------
    TypeError
        If ``surfaces`` is neither a list of FluxSurface instances nor an
        object exposing ``poincare_res.surfs``, or if an element is not a
        FluxSurface, or if a range has non-numeric elements.
    ValueError
        If ``r_range`` or ``z_range`` is not a two-element sequence with
        start < end.

    Warns
    -----
    UserWarning
        If a surface contains no points.
    """
    
    _check_surfaces(surfaces)
    _check_range("r_range", r_range)
    _check_range("z_range", z_range)
    
    if not isinstance(surfaces, list):
        surfaces = surfaces.poincare_res.surfs

    colors = ["red", "orange", "yellow", "green", "blue",
              "purple", "indigo", "violet"]
    fig, ax = plt.subplots()

    for i, surface in enumerate(surfaces):
        if surface.points.x1 is not None and len(surface.points.x1) > 0:
            r = surface.points.r
            z = surface.points.z
            ax.scatter(r, z, color=colors[i % 8], s=0.2, 
                       label="surface {}".format(i))
        else:
            warnings.warn(f"Surface {i} contains no points and is not "
                          "plotted.")

    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")

    if r_range is not None:
        ax.set_xlim(r_range)

    if z_range is not None:
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
    Filter flux surfaces and their points by surface index and (R, Z) range.

    The function first selects a contiguous slice of surfaces (``surf_range``),
    then keeps only the points of each remaining surface whose cylindrical
    radius ``R = sqrt(x1**2 + x2**2)`` and vertical position ``Z = x3`` lie
    within the ranges. Surfaces left without points are dropped.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Either a list of :class:`FluxSurface` instances (as returned by
        :func:`load_w7x_flux_surfaces`) or a field-line-tracer result object
        exposing ``poincare_res.surfs``.
    surf_range : sequence of int, default None
        Two-element ``[start, end]`` slice (Python half-open semantics) of the
        surface list, applied before the point filtering. Bounds are clipped to
        the valid range. If None, all surfaces are kept.
    r_range : sequence of float, default None
        Two-element ``[min, max]`` interval (inclusive) on R [m]. If None, no
        constraint on R.
    z_range : sequence of float, default None
        Two-element ``[min, max]`` interval (inclusive) on Z [m]. If None, no
        constraint on Z.

    Returns
    -------
    list of FluxSurface
        Copies of the surfaces that keep at least one point, containing only
        the points within the ranges. The input surfaces are not modified.

    Raises
    ------
    TypeError
        If ``flt`` is neither a list of FluxSurface instances nor an object
        exposing ``poincare_res.surfs``, or if an element is not a FluxSurface,
        or if a range has non-numeric (``surf_range``: non-integer) elements.
    ValueError
        If ``surf_range``, ``r_range`` or ``z_range`` is not a two-element
        sequence with start < end, or if ``surf_range`` contains negative
        indices.
    """
    
    # --- validate range arguments ---
    _check_surfaces(flt)
    _check_range("surf_range", surf_range, integer=True, allow_negative=False)
    _check_range("r_range", r_range)
    _check_range("z_range", z_range)
    
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
        if type(surf.points.x1) != type(None) and surf.n > 0:
            # Calculate R and Z coordinates
            r_temp = surf.points.r
            z_temp = surf.points.z
            
            # Create masks for filtering
            if r_range is not None:
                r_mask = (r_temp >= r_range[0]) & (r_temp <= r_range[1])
            else:
                r_mask = np.ones(surf.n, dtype=bool)
            
            if z_range is not None:
                z_mask = (z_temp >= z_range[0]) & (z_temp <= z_range[1])
            else:
                z_mask = np.ones(surf.n, dtype=bool)
            
            # Keep only points within both ranges
            mask = r_mask & z_mask
            
            # Only include surface if it has at least one point
            # within the ranges
            if np.any(mask):
                # Filter the points
                surf = copy.deepcopy(surf)
                surf.filter_points(mask)
                filtered_surfs.append(surf)
    
    return filtered_surfs

# Surface labelling is validated for points with R >= R_MIN_LABEL and
# |Z| <= Z_MAX_LABEL. Outside it the code runs, but the labelling and the
# subsequent density allocation may be off.
R_MIN_LABEL = 6.1   # [m]
Z_MAX_LABEL = 1.0   # [m]

def _check_label_window(surfs):
    """
    Warn if any surface point has R < R_MIN_LABEL or |Z| > Z_MAX_LABEL.
    """
    r = np.concatenate([s.points.r for s in surfs])
    z = np.concatenate([s.points.z for s in surfs])
    n_r = np.count_nonzero(r < R_MIN_LABEL)
    n_z = np.count_nonzero(np.abs(z) > Z_MAX_LABEL)
    if n_r or n_z:
        warnings.warn(
            f"{n_r} point(s) with R < {R_MIN_LABEL} m (min R = {r.min():.3f} m)"
            f" and {n_z} point(s) with |Z| > {Z_MAX_LABEL} m "
            f"(Z = [{z.min():.3f}, {z.max():.3f}] m). Labelling, density "
            "allocation and further processing may be inaccurate; select the "
            "surfaces with filter_surfaces_by_range() first.")

def _fit_surface(z, r, order):
    """
    Polynomial fit r(z), with the order reduced when there are too few
    points (1 point -> constant, 2 points -> line). The coefficients are
    padded with leading zeros to length order+1. Returns (coeff, error).
    """
    deg = min(order, len(z) - 1)
    coeff = np.polyfit(z, r, deg)
    error = np.sqrt(np.mean((r - np.polyval(coeff, z))**2))
    return np.pad(coeff, (order - deg, 0)), error

def label_surfaces(flt, limit_error = 0.015,
                   limit_number = 100, order = 2):
    """
    Label surface points as non-island or island (outer or inner side).

    Each surface is fitted with a polynomial R(Z). A surface counts as an
    island if the fit error exceeds ``limit_error`` or if it has more than
    ``limit_number`` points. Island points are split into an outer side (point
    type 1) and an inner side (point type 2), each with its own fit; points at
    the island tips are assigned to the side whose fit is closer. Points of
    non-island surfaces get point type 0.

    The fit results are stored per point type on each surface: ``coeffs`` (one
    coefficient row per type, highest power first, so the last coefficient is R
    at Z = 0) and ``errors`` (RMS fit error [m], None for point types without
    points). The densities are reset to ``[0, 0, 0]``. The surfaces are
    labelled in place.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Surfaces to label.
    limit_error : float, default 0.015
        Fit error above which a surface counts as an island [m].
    limit_number : int, default 100
        Number of points above which a surface counts as an island.
    order : int, default 2
        Order of the polynomial fit. It is reduced automatically for surfaces
        or sides with too few points.

    Returns
    -------
    list of FluxSurface
        The labelled surfaces (the same objects as the input).

    Raises
    ------
    TypeError
        If ``flt`` is neither a list of FluxSurface instances nor an object
        exposing ``poincare_res.surfs``, or if an element is not a FluxSurface,
        if ``limit_error`` is not numeric, or if ``limit_number`` or ``order``
        is not an integer.
    ValueError
        If ``limit_error``, ``limit_number`` or ``order`` is negative.

    Warns
    -----
    UserWarning
        If any point has R < ``R_MIN_LABEL`` (6.1 m) or |Z| > ``Z_MAX_LABEL``
        (1.0 m). The labelling still runs, but its accuracy and that of the
        density allocation is not guaranteed.
    """
    
    _check_surfaces(flt)
    _check_scalar("limit_error", limit_error, min_value=0)
    _check_scalar("limit_number", limit_number, integer=True, min_value=0)
    _check_scalar("order", order, integer=True, min_value=0)
    
    surfs = flt if isinstance(flt, list) else flt.poincare_res.surfs
    _check_label_window(surfs)
    surfaces = list()
    
    for i, surf in enumerate(surfs):
        r, z = surf.points.r, surf.points.z
        coeff, error = _fit_surface(z, r, order)
        
        
        # Conditions to differentiate island and not-island surfaces:
        # Error is high | or | the number of points are large
        condition_1 = error > limit_error
        condition_2 = surf.n > limit_number
        if (condition_1 or condition_2):
            
            # island case: split surface to inner and outer side
            
            p = np.polyval(coeff, z)    # centre line by polyfit
            o = r-p > 0                 # outer side point mask
            i = np.invert(o)            # inner side point mask

            # Reassign each point to the side whose fit it is closer to:
            # the single centre line misplaces points at the island tips,
            # where both sides bend towards each other.
            side = np.where(o, 1, 2)
            for _ in range(3):
                coeff_o, _ = _fit_surface(z[side == 1], r[side == 1], order)
                coeff_i, _ = _fit_surface(z[side == 2], r[side == 2], order)
                new = np.where(np.abs(r - np.polyval(coeff_o, z)) <
                               np.abs(r - np.polyval(coeff_i, z)), 1, 2)
                if np.array_equal(new, side) or new.min() == new.max():
                    break
                side = new
            o = side == 1
            i = side == 2

            # Outer side
            surf.update_point_type(1, mask=o)
            coeff_o, error_o = _fit_surface(z[o], r[o], order)
            
            # Inner side
            surf.update_point_type(2, mask=i)
            coeff_i, error_i = _fit_surface(z[i], r[i], order)
            
            surf.coeffs = np.asarray([np.zeros(order+1), coeff_o, coeff_i])
            surf.errors = [None, error_o, error_i]
            surf.density = [0, 0, 0]
            
            surfaces.append(surf)
            
        else:
            
            # not-island case
            # Leave all surfaces as they are.
            
            surf.coeffs = np.asarray(
                [coeff, np.zeros(order+1), np.zeros(order+1)])
            surf.errors = [error, None, None]
            surf.density = [0, 0, 0]
            
            surfaces.append(surf)
        
    return surfaces

def filter_surfaces_by_type(flt, point_types=None):
    """
    Filter flux surfaces and their points by point type.

    Surfaces left without points are dropped.

    Parameters
    ----------
    flt : list of FluxSurface or field-line-tracer result
        Labelled surfaces (see :func:`label_surfaces`).
    point_types : sequence of int, default None
        Point types to keep: 0 non-island, 1 island outer side, 2 island inner
        side. If None, all points are kept.

    Returns
    -------
    list of FluxSurface
        Copies of the surfaces that contain points of the requested types, with
        only those points. The input surfaces are not modified.

    Raises
    ------
    TypeError
        If ``flt`` is neither a list of FluxSurface instances nor an object
        exposing ``poincare_res.surfs``, or if an element is not a FluxSurface,
        or if ``point_types`` is not a sequence of numbers.
    ValueError
        If ``point_types`` contains values other than 0, 1 or 2, or if the
        surfaces have not been labelled with :func:`label_surfaces`.
    """
    
    _check_surfaces(flt)
    _check_sequence("point_types", point_types, domain=[0,1,2])
    
    if not isinstance(flt, list): 
        surfs = flt.poincare_res.surfs
    else:
        surfs = flt
    
    _check_labelled(surfs)
    
    filtered_surfs = []
    
    for surf in surfs:
        if type(surf.points.x1) != type(None) and surf.n > 0:
            if point_types is None: mask = np.ones(surf.n, dtype=bool)
            else: mask = np.isin(surf.points.point_type, point_types)
            
            # Only include surface if it has at least one point
            if np.any(mask):
                # Filter the points
                surf = copy.deepcopy(surf)
                surf.filter_points(mask)
                filtered_surfs.append(surf)
        
    return filtered_surfs

def filter_surfaces_by_radius(surfaces, R_range=None):
    """
    Filter flux surfaces and their points by the radius of each point type.

    The radius of a point-type group is the R of its fit at Z = 0 (see
    :func:`label_surfaces`). Surfaces left without points are dropped.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Labelled surfaces (see :func:`label_surfaces`).
    R_range : sequence of float, default None
        Two-element ``[min, max]`` interval (exclusive) on the radius of the
        point-type groups [m]. If None, no constraint on the radius.

    Returns
    -------
    list of FluxSurface
        Copies of the surfaces that keep at least one point-type group, with
        only the points of the groups within ``R_range``. The input surfaces
        are not modified.

    Raises
    ------
    TypeError
        If ``surfaces`` is neither a list of FluxSurface instances nor an
        object exposing ``poincare_res.surfs``, or if an element is not a
        FluxSurface, or if ``R_range`` has non-numeric elements.
    ValueError
        If ``R_range`` is not a two-element sequence with start < end, or if
        the surfaces have not been labelled with :func:`label_surfaces`.
    """
    
    _check_surfaces(surfaces)
    _check_range("R_range", R_range)
    
    surfs = (surfaces if isinstance(surfaces, list) 
             else surfaces.poincare_res.surfs)
    
    _check_labelled(surfs)
    
    filtered_surfs = list()
    
    for i, surf in enumerate(surfs):
        type_to_save = list()
        for i in range(len(surf.errors)):
            if surf.errors[i] != None:
                R_temp = surf.coeffs[i,-1]
                if R_range is None or (R_range[0] < R_temp < R_range[1]):
                    type_to_save.append(i)
        mask = np.isin(surf.points.point_type, type_to_save)
        if np.any(mask):
            surf = copy.deepcopy(surf)
            surf.filter_points(mask)
            filtered_surfs.append(surf)
            
    return filtered_surfs

def filter_surfaces_by_error(surfaces, error_range=None):
    """
    Filter flux surfaces and their points by the polynomial fit error.

    Surfaces left without points are dropped.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Labelled surfaces (see :func:`label_surfaces`).
    error_range : sequence of float, default None
        Two-element ``[min, max]`` interval (exclusive) on the fit error of
        each point-type group [m]. If None, no constraint on the error.

    Returns
    -------
    list of FluxSurface
        Copies of the surfaces that keep at least one point-type group, with
        only the points of the groups within ``error_range``. The input
        surfaces are not modified.

    Raises
    ------
    TypeError
        If ``surfaces`` is neither a list of FluxSurface instances nor an
        object exposing ``poincare_res.surfs``, or if an element is not a
        FluxSurface, or if ``error_range`` has non-numeric elements.
    ValueError
        If ``error_range`` is not a two-element sequence with start < end,
        or if the surfaces have not been labelled with
        :func:`label_surfaces`.
    """
    
    _check_surfaces(surfaces)
    _check_range("error_range", error_range)
    
    surfs = (surfaces if isinstance(surfaces, list) 
             else surfaces.poincare_res.surfs)
    
    _check_labelled(surfs)
    
    filtered_surfs = list()
    
    for i, surf in enumerate(surfs):
        type_to_save = list()
        for i in range(len(surf.errors)):
            if surf.errors[i] != None:
                error_temp = surf.errors[i]
                if (error_range is None or
                    error_range[0] < error_temp < error_range[1]):
                    type_to_save.append(i)
        mask = np.isin(surf.points.point_type, type_to_save)
        if np.any(mask):
            surf = copy.deepcopy(surf)
            surf.filter_points(mask)
            filtered_surfs.append(surf)
            
    return filtered_surfs

def plot_w7x_island_types(surfaces, labels, magnetic_conf='',
                          r_range=None, z_range=None, phi=np.nan, 
                          boxes=None, aspect=False, 
                          save_image=False, legend=False):
    """
    Plot a Poincaré section in the (R, Z) plane, coloured by point type.

    Parameters
    ----------
    surfaces : list of FluxSurface or field-line-tracer result
        Labelled surfaces (see :func:`label_surfaces`).
    labels : list of str
        Legend label of each point type, in point-type order (0 non-island, 1
        island outer side, 2 island inner side).
    magnetic_conf : str, default ''
        Name of the magnetic configuration, used in the title.
    r_range, z_range : sequence of float, default None
        Two-element ``[min, max]`` axis limits [m]. If None, matplotlib chooses
        the limits.
    phi : float, default nan
        Toroidal angle of the Poincaré section [rad], used in the title.
    boxes : iterable of 4-tuples, default None
        Boxes ``(r_min, z_min, r_max, z_max)`` [m] to overlay on the plot.
    aspect : bool, default False
        If True, use an equal aspect ratio.
    save_image : bool or str, default False
        If False, the figure is not saved. Otherwise the value is the
        destination filename and the figure is written as a PNG (``.png`` is
        appended if missing).
    legend : bool, default False
        If True, show a legend with ``labels``.

    Raises
    ------
    TypeError
        If ``surfaces`` is neither a list of FluxSurface instances nor an
        object exposing ``poincare_res.surfs``, or if an element is not a
        FluxSurface, or if a range has non-numeric elements.
    ValueError
        If ``r_range`` or ``z_range`` is not a two-element sequence with
        start < end, or if the surfaces have not been labelled with
        :func:`label_surfaces`.

    Warns
    -----
    UserWarning
        If a surface contains no points.
    """
    
    _check_surfaces(surfaces)
    _check_range("r_range", r_range)
    _check_range("z_range", z_range)
    
    if not isinstance(surfaces, list):
        surfaces = surfaces.poincare_res.surfs
    
    _check_labelled(surfaces)

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
            warnings.warn(f"Surface {i} contains no points and is not "
                          "plotted.")
    
    for i in range(len(labels)): 
        ax.scatter([], [], s=50, color=colors[i], label=labels[i])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")

    if r_range is not None:
        ax.set_xlim(r_range)

    if z_range is not None:
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