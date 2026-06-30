import numpy as np


def create_2d_profile(R0=0, R1=0.2, Z0=0, Z1=0.3,
                      dR=0.0005, dZ=0.0005,
                      edge_value=2e19, SOL_value=0.2e19,
                      g=20, R_LCFS=0.1):
    """
    Creates a 2D plasma density profile using a hyperbolic tangent (tanh) 
    transition across the Last Closed Flux Surface (LCFS). The density is
    uniform in Z and follows a tanh ramp in R, transitioning from 
    edge_value to SOL_value.

    Parameters
    ----------
    R0 : float, optional
        Start point of the domain in the R-direction [m]. 
        The default is 0.
    Z0 : float, optional
        Start point of the domain in the Z-direction [m]. 
        The default is 0.
    R1 : float, optional
        End point of the domain in the R-direction [m]. 
        The default is 0.2.
    Z1 : float, optional
        End point of the domain in the Z-direction [m]. 
        The default is 0.3.
    dR : float, optional
        Spatial resolution (grid spacing) in the R-direction [m]. 
        The default is 0.0005.
    dZ : float, optional
        Spatial resolution (grid spacing) in the Z-direction [m]. 
        The default is 0.0005.
    edge_value : float, optional
        Maximum (core) plasma density [m^-3]. 
        The default is 2e19.
    SOL_value : float, optional
        Minimum (edge/SOL) plasma density [m^-3]. 
        The default is 0.2e19.
    g : float, optional
        Gradient steepness parameter controlling the sharpness of the tanh
        transition at the LCFS. Higher values produce a steeper density ramp.
        The default is 20.
    R_LCF : float, optional
        Position of the Last Closed Flux Surface in the R-direction [m],
        where the density transition is centred. 
        The default is 0.1.

    Returns
    -------
    R : numpy.ndarray, shape (NR,)
        1D array of R-coordinates, ranging from 0 to R with spacing dR [m].
    Z : numpy.ndarray, shape (NZ,)
        1D array of Z-coordinates, ranging from 0 to Z with spacing dZ [m].
    field : numpy.ndarray, shape (NR, NZ)
        2D array of plasma density values [m^-3]. The profile varies along R
        following:
            SOL_value + (edge_value - SOL_value) / 2 * 
            (1 + tanh(g * (R - R_LCFS)))
        and is uniform (tiled) along Z.
    """
    
    R = np.arange(R0, R1, dR)
    Z = np.arange(Z0, Z1, dZ)
    
    # create the background denisty field
    field = np.tile( SOL_value + (edge_value - SOL_value) / 2 
                 * ( 1 + np.tanh(g * (R - R_LCFS)) ),
                 (len(Z), 1) ).transpose()
    return R, Z, field


