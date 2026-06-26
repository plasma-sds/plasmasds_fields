# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:28:31 2026

@author: akosk
"""
import numpy as np
import h5py

def perturbation_on_field(R, Z, field, 
                          dt, r_trajectory , z_trajectory , 
                          perturbation_fwhm, perturbation_ampl,
                          file=None, return_array=True):
    """
    Simulates the evolution of a 2D plasma density field over time by
    superimposing travelling filaments, each modelled as a 2D Gaussian,
    onto a background density profile. Results can be saved to an HDF5
    file and/or returned as an in-memory dictionary.

    Parameters
    ----------
    R : numpy.ndarray, shape (NR,)
        1D array of R-coordinates of the spatial grid [m].
    Z : numpy.ndarray, shape (NZ,)
        1D array of Z-coordinates of the spatial grid [m].
    field : numpy.ndarray, shape (NR, NZ)
        2D background density field [m^-3] onto which filament
        contributions are added at each time step.
    t : numpy.ndarray, shape (Nt,)
        1D array of time stamps [s] defining the simulation time axis.
    r_trajectory : list of array-like, length n_filaments
        X-position [m] of each filament at every time step.
        r_trajectory[i][t_i] gives the x-coordinate of filament i at
        time index t_i. None values are silently skipped.
    z_trajectory : list of array-like, length n_filaments
        Y-position [m] of each filament at every time step.
        z_trajectory[i][t_i] gives the y-coordinate of filament i at
        time index t_i. None values are silently skipped.
    perturbation_fwhm : float or list of array-like, length n_filaments
        Full Width at Half Maximum [m] of the filament Gaussian. Used
        to derive the sigma (sigma = FWHM / 2.535) and the ±3-sigma
        evaluation window (half-width = FWHM / 2.535 * 3).

        - If a scalar (``float`` / ``int``), the same FWHM is applied
          to every filament at every time step.
        - If a list of array-like, indexing follows ``r_trajectory``:
          ``perturbation_fwhm[f_i][t_i]`` is the FWHM of filament
          ``f_i`` at time ``t_i``.
    perturbation_ampl : float or list of array-like, length n_filaments
        Peak amplitude [m^-3] of the filament Gaussian.

        - If a scalar, the same amplitude is applied to every
          filament at every time step.
        - If a list of array-like, indexing follows ``r_trajectory``:
          ``perturbation_ampl[f_i][t_i]`` is the amplitude of
          filament ``f_i`` at time ``t_i``.
    file : str or path-like, optional
        If provided, results are written to an HDF5 file at this path.
        The file contains datasets 'R', 'Z', 't', and one dataset per
        time step named 'field_TTTTT' (zero-padded index).
        The default is None (no file output).
    return_array : bool, optional
        If True, the time-resolved density field is accumulated in
        memory and returned as a dictionary. Set to False to reduce
        memory usage when only file output is needed.
        The default is True.

    Returns
    -------
    data : dict, only returned when return_array=True
        Dictionary with the following keys:

        - ``'R'``  : numpy.ndarray, shape (Nx,)  — x-coordinate array.
        - ``'Z'``  : numpy.ndarray, shape (Ny,)  — y-coordinate array.
        - ``'t'``  : numpy.ndarray, shape (Nt,)  — time array.
        - ``'field'``  : numpy.ndarray, shape (Nt, Nx, Ny) — density field
          at each time step, background plus all filament contributions.

        Returns None if return_array=False.
    """
    t = np.arange(len(r_trajectory[0])) * dt
    
    if not isinstance(file, type(None)):
        # create the output file, fill it after
        f = h5py.File(file, "w")
        f.create_dataset("R", data = R, compression="gzip")
        f.create_dataset("Z", data = Z, compression="gzip")
        f.create_dataset("t", data = t, compression="gzip")
    
    if return_array:
        data = {"R": R, "Z": Z, "t": t,
                "field": np.zeros((len(t), len(R), len(Z)))}
    
    # create the Gaussian distribution for reduced area, 6 sigma range
    def gaussian_2d(R, Z, r0=0, z0=0, sigma_r=1, sigma_z=1, A=1):
        return A * np.exp(-(((R - r0) ** 2) / (2 * sigma_r ** 2)
                            + ((Z - z0) ** 2) / (2 * sigma_z ** 2)))

    _scalar_types = (int, float, np.integer, np.floating)
    fwhm_is_scalar = isinstance(perturbation_fwhm, _scalar_types)
    ampl_is_scalar = isinstance(perturbation_ampl, _scalar_types)
    n_f = len(r_trajectory)
    
    # iterate through all the frames and the filaments
    for t_i in range(len(t)):
        den_act = field.copy()
        for f_i in range(n_f):
            try: # if nonetype in datafield -> it will be passed
                posR, posZ = r_trajectory[f_i][t_i], z_trajectory[f_i][t_i]
                fwhm_val = (perturbation_fwhm if fwhm_is_scalar
                            else perturbation_fwhm[f_i][t_i])
                ampl = (perturbation_ampl if ampl_is_scalar
                        else perturbation_ampl[f_i][t_i])
                effR = effZ = fwhm_val / 2.535 * 3
                sigR = sigZ = fwhm_val / 2.535
                # create the distribution in the 6 sigma range
                G_ind_r = np.where((R >= posR - effR) &
                                   (R <= posR + effR))[0]
                G_ind_z = np.where((Z >= posZ - effZ) &
                                   (Z <= posZ + effZ))[0]
                GR, GZ = np.meshgrid(R[G_ind_r], Z[G_ind_z])
                G_field = gaussian_2d(GR, GZ, r0 = posR, z0 = posZ,
                                          sigma_r = sigR, sigma_z = sigZ,
                                          A = ampl)
                # adding the Gaussian influence to the original den. field
                den_act[G_ind_r[0] : G_ind_r[-1] +1,
                        G_ind_z[0] : G_ind_z[-1] +1] += G_field
                
            except: continue
        
        if return_array: data["field"][t_i, :, :] = den_act
            
        if not isinstance(file, type(None)):
            # Save arrays to an HDF5 file
            f.create_dataset("field_" + str(t_i).zfill(5), 
                              data = den_act, compression="gzip")
    
    if return_array: return data
    else: return None

def import_density_field(file):
    """
    Reads a density field simulation from an HDF5 file produced by 
    `perturbation_on_field()` and returns the spatial coordinates,
    time axis, and 3D density array as an in-memory dictionary.

    Parameters
    ----------
    file : str or path-like
        Path to the HDF5 file containing the simulation output. The file
        is expected to have the following datasets:
        
        - ``'R'``             : 1D R-coordinate array [m].
        - ``'Z'``             : 1D Z-coordinate array [m].
        - ``'t'``             : 1D time array [s].
        - ``'field_TTTTT'``: one 2D density snapshot per time step,
          with a zero-padded integer suffix (e.g. ``'field_00000'``).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'R'`` : numpy.ndarray, shape (Nx,)       — R-coordinate array [m].
        - ``'Z'`` : numpy.ndarray, shape (Ny,)       — Z-coordinate array [m].
        - ``'t'`` : numpy.ndarray, shape (Nt,)       — time array [s].
        - ``'field'`` : numpy.ndarray, shape (Nt, Nx, Ny) — 
          density field at each time step [m^-3], stacked in chronological 
          order of the ``'field_TTTTT'`` datasets.
    """

    with h5py.File(file, "r") as f:
        Z, R = np.array(f["Z"]), np.array(f["R"])
        field = np.transpose(np.asarray(
            [f[key] for key in f.keys() if "field_" in key]), [0,1,2])
        t = np.array(f["t"])
    return {"R": R, "Z": Z, "field": field, "t": t}
        