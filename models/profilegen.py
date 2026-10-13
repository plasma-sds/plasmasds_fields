import numpy as np


def create_2d_profile(R0=0, R1=0.2, Z0=0, Z1=0.3,
                      dR=0.0005, dZ=0.0005,
                      edge_value=2e19, SOL_value=0.2e19,
                      g=20, R_LCFS=0.1):
    """
    Creates a 2D plasma density profile using a hyperbolic tangent (tanh)
    transition across the Last Closed Flux Surface (LCFS). The density is
    uniform in Z and follows a tanh ramp in R centred at ``R_LCFS``: for
    g > 0 it approaches ``SOL_value`` well below ``R_LCFS`` (small R) and
    ``edge_value`` well above it (large R):

        SOL_value + (edge_value - SOL_value) / 2 * (1 + tanh(g * (R - R_LCFS)))

    Parameters
    ----------
    R0 : float, optional
        Start of the domain in the R-direction [m]. The default is 0.
    R1 : float, optional
        End of the domain in the R-direction [m], not included (as in
        ``numpy.arange``). The default is 0.2.
    Z0 : float, optional
        Start of the domain in the Z-direction [m]. The default is 0.
    Z1 : float, optional
        End of the domain in the Z-direction [m], not included. The default
        is 0.3.
    dR : float, optional
        Grid spacing in the R-direction [m]. The default is 0.0005.
    dZ : float, optional
        Grid spacing in the Z-direction [m]. The default is 0.0005.
    edge_value : float, optional
        Plasma density on the large-R side of the transition [m^-3].
        The default is 2e19.
    SOL_value : float, optional
        Plasma density on the small-R side of the transition [m^-3].
        The default is 0.2e19.
    g : float, optional
        Gradient steepness of the tanh transition [1/m]. Higher values give
        a steeper density ramp. The default is 20.
    R_LCFS : float, optional
        Position of the Last Closed Flux Surface in R [m], where the
        transition is centred. The default is 0.1.

    Returns
    -------
    R : numpy.ndarray, shape (NR,)
        R-coordinates from ``R0`` up to (excluding) ``R1``, spacing ``dR`` [m].
    Z : numpy.ndarray, shape (NZ,)
        Z-coordinates from ``Z0`` up to (excluding) ``Z1``, spacing ``dZ`` [m].
    field : numpy.ndarray, shape (NR, NZ)
        2D plasma density [m^-3], following the tanh profile along R and
        uniform (tiled) along Z.
    """
    
    R = np.arange(R0, R1, dR)
    Z = np.arange(Z0, Z1, dZ)
    
    # create the background denisty field
    field = np.tile( SOL_value + (edge_value - SOL_value) / 2 
                 * ( 1 + np.tanh(g * (R - R_LCFS)) ),
                 (len(Z), 1) ).transpose()
    return R, Z, field


