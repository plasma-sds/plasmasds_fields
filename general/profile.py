import numpy as np


def create_profile(x0=0, x1=0.2, y0=0, y1=0.3,
                   dx=0.0005, dy=0.0005,
                   den_max=2e19, den_min=0.2e19,
                   g=20, x_LCF=0.1):
    """
    Creates a 2D plasma density profile using a hyperbolic tangent (tanh) 
    transition across the Last Closed Flux surface (LCF). The density is
    uniform in Y and follows a tanh ramp in X, transitioning from 
    den_min to den_max.

    Parameters
    ----------
    x0 : float, optional
        Start point of the domain in the x-direction [m]. 
        The default is 0.
    x0 : float, optional
        Start point of the domain in the x-direction [m]. 
        The default is 0.
    x1 : float, optional
        End point of the domain in the x-direction [m]. 
        The default is 0.2.
    y1 : float, optional
        End point of the domain in the y-direction [m]. 
        The default is 0.3.
    dx : float, optional
        Spatial resolution (grid spacing) in the x-direction [m]. 
        The default is 0.0005.
    dy : float, optional
        Spatial resolution (grid spacing) in the y-direction [m]. 
        The default is 0.0005.
    den_max : float, optional
        Maximum (core) plasma density [m^-3]. 
        The default is 2e19.
    den_min : float, optional
        Minimum (edge/SOL) plasma density [m^-3]. 
        The default is 0.2e19.
    g : float, optional
        Gradient steepness parameter controlling the sharpness of the tanh
        transition at the LCF. Higher values produce a steeper density ramp.
        The default is 20.
    x_LCF : float, optional
        Position of the Last Closed Flux surface in the x-direction [m],
        where the density transition is centred. 
        The default is 0.1.

    Returns
    -------
    x : numpy.ndarray, shape (Nx,)
        1D array of x-coordinates, ranging from 0 to X with spacing dx [m].
    y : numpy.ndarray, shape (Ny,)
        1D array of y-coordinates, ranging from 0 to Y with spacing dy [m].
    density : numpy.ndarray, shape (Nx, Ny)
        2D array of plasma density values [m^-3]. The profile varies along x
        following:
            den_min + (den_max - den_min) / 2 * (1 + tanh(g * (x - x_LCF)))
        and is uniform (tiled) along y.
    """
    
    x = np.arange(x0, x1, dx)
    y = np.arange(y0, y1, dy)
    
    # create the background denisty field
    density = np.tile( den_min + (den_max - den_min) / 2 
                        * ( 1 + np.tanh(g * (x - x_LCF)) ),
                        (len(y), 1) ).transpose()
    return x, y, density


