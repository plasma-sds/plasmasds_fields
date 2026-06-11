# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:28:31 2026

@author: akosk
"""
import numpy as np
import h5py

def add_filament_to_field(X, Y, Z, 
        t, x_path_fil, y_path_fil, FWHMs_fil, ampls_fil,
        file=None, return_array=True):
    """
    Simulates the evolution of a 2D plasma density field over time by
    superimposing travelling filaments, each modelled as a 2D Gaussian,
    onto a background density profile. Results can be saved to an HDF5
    file and/or returned as an in-memory dictionary.

    Parameters
    ----------
    X : numpy.ndarray, shape (Nx,)
        1D array of x-coordinates of the spatial grid [m].
    Y : numpy.ndarray, shape (Ny,)
        1D array of y-coordinates of the spatial grid [m].
    Z : numpy.ndarray, shape (Nx, Ny)
        2D background density field [m^-3] onto which filament
        contributions are added at each time step.
    t : numpy.ndarray, shape (Nt,)
        1D array of time stamps [s] defining the simulation time axis.
    x_path_fil : list of array-like, length n_filaments
        X-position [m] of each filament at every time step.
        x_path_fil[i][t_i] gives the x-coordinate of filament i at
        time index t_i. None values are silently skipped.
    y_path_fil : list of array-like, length n_filaments
        Y-position [m] of each filament at every time step.
        y_path_fil[i][t_i] gives the y-coordinate of filament i at
        time index t_i. None values are silently skipped.
    FWHMs_fil : list of array-like, length n_filaments
        Full Width at Half Maximum [m] of each filament at every time
        step. Used to derive the Gaussian sigma
        (sigma = FWHM / 2.535) and the ±3-sigma evaluation window
        (half-width = FWHM / 2.535 * 3).
    ampls_fil : list of array-like, length n_filaments
        Peak amplitude [m^-3] of each filament's Gaussian density
        perturbation at every time step.
    file : str or path-like, optional
        If provided, results are written to an HDF5 file at this path.
        The file contains datasets 'X', 'Y', 't', and one dataset per
        time step named 'den_field_TTTTT' (zero-padded index).
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

        - ``'X'``  : numpy.ndarray, shape (Nx,)  — x-coordinate array.
        - ``'Y'``  : numpy.ndarray, shape (Ny,)  — y-coordinate array.
        - ``'t'``  : numpy.ndarray, shape (Nt,)  — time array.
        - ``'Z'``  : numpy.ndarray, shape (Nt, Nx, Ny) — density field
          at each time step, background plus all filament contributions.

        Returns None if return_array=False.
    """
    
    if not isinstance(file, type(None)):
        # create the output file, fill it after
        f = h5py.File(file, "w")
        f.create_dataset("X", data = X, compression="gzip")
        f.create_dataset("Y", data = Y, compression="gzip")
        f.create_dataset("t", data = t, compression="gzip")
    
    if return_array:
        data = {"X": X, "Y": Y, "t": t,
                "Z": np.zeros((len(t), len(X), len(Y)))}
    
    # create the Gaussian distribution for reduced area, 6 sigma range
    def gaussian_2d(X, Y, x0=0, y0=0, sigma_x=1, sigma_y=1, A=1):
        return A * np.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2)
                            + ((Y - y0) ** 2) / (2 * sigma_y ** 2)))
    n_f = len(FWHMs_fil)
    
    # iterate through all the frames and the filaments
    for t_i in range(len(t)):
        den_act = Z.copy()
        for f_i in range(n_f):
            try: # if nonetype in datafield -> it will be passed
                posX, posY = x_path_fil[f_i][t_i], y_path_fil[f_i][t_i]
                effX = effY = FWHMs_fil[f_i][t_i]/ 2.535*3
                sigX = sigY = FWHMs_fil[f_i][t_i] / 2.535
                ampl = ampls_fil[f_i][t_i]
                # create the distribution in the 6 sigma range
                G_ind_x = np.where((X >= posX - effX) &
                                   (X <= posX + effX))[0]
                G_ind_y = np.where((Y >= posY - effY) &
                                   (Y <= posY + effY))[0]
                GY, GX = np.meshgrid(Y[G_ind_y], X[G_ind_x])
                G_den_field = gaussian_2d(GX, GY, x0 = posX, y0 = posY,
                                          sigma_x = sigX, sigma_y = sigY,
                                          A = ampl)
                # adding the Gaussian influence to the original den. field
                den_act[G_ind_x[0] : G_ind_x[-1] +1,
                        G_ind_y[0] : G_ind_y[-1] +1] += G_den_field
                
            except: continue
        
        if return_array: data["Z"][t_i, :, :] = den_act
            
        if not isinstance(file, type(None)):
            # Save arrays to an HDF5 file
            f.create_dataset("den_field_" + str(t_i).zfill(5), 
                              data = den_act, compression="gzip")
    
    if return_array: return data
    else: None

def import_density_field(file):
    """
    Reads a density field simulation from an HDF5 file produced by 
    `add_filament_to_field()` and returns the spatial coordinates,
    time axis, and 3D density array as an in-memory dictionary.

    Parameters
    ----------
    file : str or path-like
        Path to the HDF5 file containing the simulation output. The file
        is expected to have the following datasets:
        
        - ``'X'``             : 1D x-coordinate array [m].
        - ``'Y'``             : 1D y-coordinate array [m].
        - ``'t'``             : 1D time array [s].
        - ``'den_field_TTTTT'``: one 2D density snapshot per time step,
          with a zero-padded integer suffix (e.g. ``'den_field_00000'``).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'X'`` : numpy.ndarray, shape (Nx,)       — x-coordinate array [m].
        - ``'Y'`` : numpy.ndarray, shape (Ny,)       — y-coordinate array [m].
        - ``'t'`` : numpy.ndarray, shape (Nt,)       — time array [s].
        - ``'Z'`` : numpy.ndarray, shape (Nt, Nx, Ny)— density field at each
          time step [m^-3], stacked in chronological order of the
          ``'den_field_TTTTT'`` datasets.
    """

    with h5py.File(file, "r") as f:
        Y, X = np.array(f["Y"]), np.array(f["X"])
        Z = np.transpose(np.asarray(
            [f[key] for key in f.keys() if "den_field_" in key]), [0,1,2])
        t = np.array(f["t"])
    return {"X":X, "Y":Y, "Z":Z, "t":t}
        