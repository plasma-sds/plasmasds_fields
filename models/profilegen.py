"""
Generic 2-D background density profiles.
"""
import numpy as np


def create_2d_profile(R0=0, R1=0.2, Z0=0, Z1=0.3,
                      dR=0.0005, dZ=0.0005,
                      edge_value=2e19, SOL_value=0.2e19,
                      g=20, R_LCFS=0.1):
    """
    Create a 2-D density profile with a tanh transition across the LCFS.

    The density is uniform in Z and follows a tanh ramp in R, centred at the
    last closed flux surface ``R_LCFS``:

        SOL_value + (edge_value - SOL_value) / 2
            * (1 + tanh(g * (R - R_LCFS)))

    For g > 0 it approaches ``SOL_value`` well below ``R_LCFS`` (small R) and
    ``edge_value`` well above it (large R).

    Parameters
    ----------
    R0 : float, default 0
        Start of the domain in R [m].
    R1 : float, default 0.2
        End of the domain in R [m], not included (as in ``numpy.arange``).
    Z0 : float, default 0
        Start of the domain in Z [m].
    Z1 : float, default 0.3
        End of the domain in Z [m], not included.
    dR : float, default 0.0005
        Grid spacing in R [m].
    dZ : float, default 0.0005
        Grid spacing in Z [m].
    edge_value : float, default 2e19
        Density on the large-R side of the transition [m^-3].
    SOL_value : float, default 0.2e19
        Density on the small-R side of the transition [m^-3].
    g : float, default 20
        Steepness of the tanh transition [1/m]; higher values give a steeper
        ramp.
    R_LCFS : float, default 0.1
        Position of the last closed flux surface in R [m], where the transition
        is centred.

    Returns
    -------
    R : numpy.ndarray, shape (NR,)
        R coordinates from ``R0`` up to (excluding) ``R1``, spacing ``dR`` [m].
    Z : numpy.ndarray, shape (NZ,)
        Z coordinates from ``Z0`` up to (excluding) ``Z1``, spacing ``dZ`` [m].
    field : numpy.ndarray, shape (NR, NZ)
        Density [m^-3], following the tanh profile along R and uniform along Z.
    """
    
    R = np.arange(R0, R1, dR)
    Z = np.arange(Z0, Z1, dZ)
    
    # create the background density field
    field = np.tile( SOL_value + (edge_value - SOL_value) / 2 
                 * ( 1 + np.tanh(g * (R - R_LCFS)) ),
                 (len(Z), 1) ).transpose()
    return R, Z, field


