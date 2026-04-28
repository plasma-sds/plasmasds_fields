import numpy as np

class Points(object):
    def __init__(self, x1, x2, x3):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)


class FluxSurface(object):
    def __init__(self, x1, x2, x3, phi0, density=0, density_inner_island=0):
        self.points = Points(x1, x2, x3)
        self.phi0 = phi0
        self.density = density
        self.density_inner_island = density_inner_island

    def update_density(self, value, density_inner_island=None):
        self.density = value
        if density_inner_island is not None:
            self.density_inner_island = density_inner_island