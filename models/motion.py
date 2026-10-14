"""
Time-dependent density perturbations on 2-D background fields: Gaussian
filaments and sine waves, with HDF5 input and output.
"""
import numpy as np
import h5py

def _field_rz(R, Z, field):
    """
    Return ``field`` in (NR, NZ) layout.

    Accepts (NR, NZ), and (NZ, NR) as produced by
    w7x.fields.make_regular_density_field or HESEL snapshots. A square field
    (NR == NZ) is taken as (NR, NZ).
    """
    field = np.asarray(field)
    if field.shape == (len(R), len(Z)):
        return field
    if field.shape == (len(Z), len(R)):
        return field.T
    raise ValueError(
        f"field shape {field.shape} does not match R ({len(R)}) and "
        f"Z ({len(Z)}); expected ({len(R)}, {len(Z)}) or "
        f"({len(Z)}, {len(R)}).")

def perturbation_gaussian(R, Z, field, 
                          dt, r_trajectory , z_trajectory , 
                          perturbation_fwhm, perturbation_ampl,
                          file=None, return_array=True):
    """
    Simulate travelling Gaussian filaments on a 2-D background density field.

    At each time step, every filament is added to the background field as a
    round 2-D Gaussian, evaluated within ±3 sigma of its centre. The result can
    be written to an HDF5 file and/or returned in memory.

    Parameters
    ----------
    R : numpy.ndarray, shape (NR,)
        R coordinates of the grid [m].
    Z : numpy.ndarray, shape (NZ,)
        Z coordinates of the grid [m].
    field : numpy.ndarray, shape (NR, NZ) or (NZ, NR)
        Background density field [m^-3]. The (NZ, NR) layout of
        :func:`w7x.fields.make_regular_density_field` is transposed
        automatically; a square field is taken as (NR, NZ).
    dt : float
        Time step [s].
    r_trajectory : list of array_like, length n_filaments
        R position of each filament at each time step [m]:
        ``r_trajectory[f_i][t_i]`` is the R of filament ``f_i`` at step
        ``t_i``. None or NaN values mean the filament is absent.
    z_trajectory : list of array_like, length n_filaments
        Z position of each filament at each time step [m], indexed like
        ``r_trajectory``. None or NaN values mean the filament is absent.

        Trajectories (and per-step ``perturbation_fwhm`` /
        ``perturbation_ampl`` lists) may have different lengths: the number of
        frames is set by the longest trajectory, and a filament is absent in
        the frames beyond the end of its own data.
    perturbation_fwhm : float or list of array_like, length n_filaments
        Full width at half maximum of the filaments [m]. A scalar applies to
        all filaments and time steps; a list is indexed like ``r_trajectory``.
        The Gaussian sigma is FWHM / (2 sqrt(2 ln 2)) ≈ FWHM / 2.3548.
    perturbation_ampl : float or list of array_like, length n_filaments
        Peak amplitude of the filaments [m^-3]. A scalar applies to all
        filaments and time steps; a list is indexed like ``r_trajectory``.
    file : str or path-like, default None
        If given, the result is written to an HDF5 file at this path, with
        datasets ``"R"``, ``"Z"``, ``"t"`` and one dataset per time step named
        ``"field_TTTTT"`` (zero-padded frame number). The file is closed even
        if an error occurs.
    return_array : bool, default True
        If True, the time-resolved field is kept in memory and returned. Set to
        False to save memory when only the file output is needed.

    Returns
    -------
    dict or None
        If ``return_array`` is True, a dictionary with:

        * ``"R"``: R coordinates, shape (NR,) [m].
        * ``"Z"``: Z coordinates, shape (NZ,) [m].
        * ``"t"``: time axis, shape (Nt,) [s].
        * ``"field"``: density field at each time step, shape (Nt, NR, NZ)
          [m^-3].

        None if ``return_array`` is False.

    Raises
    ------
    ValueError
        If ``field`` does not match the lengths of ``R`` and ``Z``, or if
        ``z_trajectory``, ``perturbation_fwhm`` or ``perturbation_ampl`` (when
        given per filament) has a different number of filaments than
        ``r_trajectory``.
    """
    field = _field_rz(R, Z, field)
    
    _scalar_types = (int, float, np.integer, np.floating)
    fwhm_is_scalar = isinstance(perturbation_fwhm, _scalar_types)
    ampl_is_scalar = isinstance(perturbation_ampl, _scalar_types)
    n_f = len(r_trajectory)
    
    # every per-filament input must describe the same filaments
    per_filament = [("z_trajectory", z_trajectory)]
    if not fwhm_is_scalar: per_filament.append(("perturbation_fwhm",
                                                perturbation_fwhm))
    if not ampl_is_scalar: per_filament.append(("perturbation_ampl",
                                                perturbation_ampl))
    for name, values in per_filament:
        if len(values) != n_f:
            raise ValueError(f"{name} has {len(values)} filaments, "
                             f"r_trajectory has {n_f}.")
    
    # the longest trajectory sets the number of frames
    n_t = max(len(traj) for traj in list(r_trajectory) + list(z_trajectory))
    t = np.arange(n_t) * dt
    
    if return_array:
        data = {"R": R, "Z": Z, "t": t,
                "field": np.zeros((len(t), len(R), len(Z)))}
    
    # 2-D Gaussian, evaluated within a ±3 sigma window around each filament
    def gaussian_2d(R, Z, r0=0, z0=0, sigma_r=1, sigma_z=1, A=1):
        """
        Return a 2-D Gaussian of amplitude ``A`` centred at (r0, z0).
        """

        return A * np.exp(-(((R - r0) ** 2) / (2 * sigma_r ** 2)
                            + ((Z - z0) ** 2) / (2 * sigma_z ** 2)))

    fwhm_to_sigma = 2 * np.sqrt(2 * np.log(2))     # FWHM = 2.3548 sigma
    
    def step(values, f_i, t_i):
        """Value of filament f_i at step t_i, None beyond its data."""
        return values[f_i][t_i] if t_i < len(values[f_i]) else None
    
    f = None
    try:
        if file is not None:
            # create the output file, fill it after
            f = h5py.File(file, "w")
            f.create_dataset("R", data = R, compression="gzip")
            f.create_dataset("Z", data = Z, compression="gzip")
            f.create_dataset("t", data = t, compression="gzip")
        
        # iterate through all the frames and the filaments
        for t_i in range(len(t)):
            den_act = field.copy()
            for f_i in range(n_f):
                posR = step(r_trajectory, f_i, t_i)
                posZ = step(z_trajectory, f_i, t_i)
                fwhm_val = (perturbation_fwhm if fwhm_is_scalar
                            else step(perturbation_fwhm, f_i, t_i))
                ampl = (perturbation_ampl if ampl_is_scalar
                        else step(perturbation_ampl, f_i, t_i))
                # missing data (None or NaN): no filament in this frame
                if any(v is None or np.isnan(v)
                       for v in (posR, posZ, fwhm_val, ampl)):
                    continue
                sigR = sigZ = fwhm_val / fwhm_to_sigma
                effR = effZ = 3 * sigR
                # grid indices within the ±3 sigma window
                G_ind_r = np.where((R >= posR - effR) &
                                   (R <= posR + effR))[0]
                G_ind_z = np.where((Z >= posZ - effZ) &
                                   (Z <= posZ + effZ))[0]
                # filament window entirely outside the grid: nothing to add
                if G_ind_r.size == 0 or G_ind_z.size == 0:
                    continue
                GR, GZ = np.meshgrid(R[G_ind_r], Z[G_ind_z], indexing="ij")
                G_field = gaussian_2d(GR, GZ, r0 = posR, z0 = posZ,
                                          sigma_r = sigR, sigma_z = sigZ,
                                          A = ampl)
                # adding the Gaussian influence to the original den. field
                den_act[G_ind_r[0] : G_ind_r[-1] +1,
                        G_ind_z[0] : G_ind_z[-1] +1] += G_field
        
            if return_array: data["field"][t_i, :, :] = den_act
            
            if f is not None:
                # Save arrays to an HDF5 file
                f.create_dataset("field_" + str(t_i).zfill(5), 
                                  data = den_act, compression="gzip")
    finally:
        if f is not None: f.close()
    
    if return_array: return data
    else: return None

def perturbation_sine(R, Z, field, dt, nt, r_trajectory, perturbation_speed,
                      perturbation_wavelength, perturbation_ampl,
                      file=None, return_array=True):
    """
    Simulate a travelling sine perturbation of a 2-D background density field.

    The perturbation is a sine wave around a vertical centre line,
    ``r_trajectory``, travelling along Z. At each time step and Z grid point,
    all densities between the centre line and the displaced wave are set to the
    background density at the centre line, i.e. the density profile is
    flattened over the displacement. The result can be written to an HDF5 file
    and/or returned in memory.

    Parameters
    ----------
    R : numpy.ndarray, shape (NR,)
        R coordinates of the grid [m].
    Z : numpy.ndarray, shape (NZ,)
        Z coordinates of the grid [m].
    field : numpy.ndarray, shape (NR, NZ) or (NZ, NR)
        Background density field [m^-3]. The (NZ, NR) layout of
        :func:`w7x.fields.make_regular_density_field` is transposed
        automatically; a square field is taken as (NR, NZ).
    dt : float
        Time step [s].
    nt : int
        Number of time steps (total time nt * dt).
    r_trajectory : float or numpy.ndarray, shape (NZ,)
        R position of the centre line of the wave at each Z grid point [m],
        usually a parabolic line along Z. A scalar gives a straight vertical
        centre line.
    perturbation_speed : float
        Propagation speed of the wave along Z [m/s]; positive values travel
        towards larger Z.
    perturbation_wavelength : float
        Wavelength of the wave along Z [m].
    perturbation_ampl : float
        Amplitude of the radial displacement of the wave [m].
    file : str or path-like, default None
        If given, the result is written to an HDF5 file at this path, with
        datasets ``"R"``, ``"Z"``, ``"t"`` and one dataset per time step named
        ``"field_TTTTT"`` (zero-padded frame number). The file is closed even
        if an error occurs.
    return_array : bool, default True
        If True, the time-resolved field is kept in memory and returned. Set to
        False to save memory when only the file output is needed.

    Returns
    -------
    dict or None
        If ``return_array`` is True, a dictionary with:

        * ``"R"``: R coordinates, shape (NR,) [m].
        * ``"Z"``: Z coordinates, shape (NZ,) [m].
        * ``"t"``: time axis, shape (Nt,) [s].
        * ``"field"``: density field at each time step, shape (Nt, NR, NZ)
          [m^-3].

        None if ``return_array`` is False.

    Raises
    ------
    ValueError
        If ``field`` does not match the lengths of ``R`` and ``Z``, or if
        ``r_trajectory`` is neither a scalar nor one value per Z grid point.
    """
    
    field = _field_rz(R, Z, field)
    
    # centre line: one R value per Z grid point; a scalar gives a straight line
    r_trajectory = np.asarray(r_trajectory, dtype=float)
    if r_trajectory.ndim == 0:
        r_trajectory = np.full(len(Z), float(r_trajectory))
    if r_trajectory.shape != (len(Z),):
        raise ValueError(
            f"r_trajectory must be a scalar or have one value per Z grid "
            f"point ({len(Z)}), got shape {r_trajectory.shape}.")
    
    t = np.arange(nt) * dt
    
    if return_array:
        data = {"R": R, "Z": Z, "t": t,
                "field": np.zeros((len(t), len(R), len(Z)))}
    
    f = None
    try:
        if file is not None:
            # create the output file, fill it after
            f = h5py.File(file, "w")
            f.create_dataset("R", data = R, compression="gzip")
            f.create_dataset("Z", data = Z, compression="gzip")
            f.create_dataset("t", data = t, compression="gzip")
        
        # iterate through all the frames and the filaments
        mask = np.zeros(len(R), dtype=bool)
        for t_i in range(len(t)):
            den_act = field.copy()
        
            for z_i in range(len(Z)):
                r0_center = r_trajectory[z_i]
            
                k = 2 * np.pi / perturbation_wavelength
                omega = k * perturbation_speed
                r0_act = (r0_center + perturbation_ampl
                          * np.cos(k * Z[z_i] - omega * t[t_i]))
            
                peak = np.interp(r0_center, R, den_act[:, z_i])
            
            
                if r0_act < r0_center: r_min, r_max = r0_act, r0_center
                else: r_min, r_max = r0_center, r0_act
                mask[:] = (R < r_max) & (R > r_min)
            
                den_act[mask, z_i] = peak
            
        
            if return_array: data["field"][t_i, :, :] = den_act
            
            if f is not None:
                # Save arrays to an HDF5 file
                f.create_dataset("field_" + str(t_i).zfill(5), 
                                  data = den_act, compression="gzip")
    finally:
        if f is not None: f.close()
    
    if return_array: return data
    else: return None

def import_density_field(file):
    """
    Read a density field simulation from an HDF5 file.

    Reads files written by :func:`perturbation_gaussian` or
    :func:`perturbation_sine`, with datasets ``"R"``, ``"Z"``, ``"t"`` and one
    2-D snapshot per time step named ``"field_TTTTT"`` (zero-padded frame
    number).

    Parameters
    ----------
    file : str or path-like
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary with:

        * ``"R"``: R coordinates, shape (NR,) [m].
        * ``"Z"``: Z coordinates, shape (NZ,) [m].
        * ``"t"``: time axis, shape (Nt,) [s].
        * ``"field"``: density field at each time step, shape (Nt, NR, NZ)
          [m^-3], in order of the frame number in the dataset names.

    Raises
    ------
    KeyError
        If the file has no ``"R"``, ``"Z"`` or ``"t"`` dataset.
    """

    with h5py.File(file, "r") as f:
        Z, R = np.array(f["Z"]), np.array(f["R"])
        # sort by frame number, not alphabetically (robust beyond 5 digits)
        keys = sorted((key for key in f.keys() if key.startswith("field_")),
                      key=lambda key: int(key[len("field_"):]))
        field = np.asarray([f[key] for key in keys])
        t = np.array(f["t"])
    return {"R": R, "Z": Z, "field": field, "t": t}
        