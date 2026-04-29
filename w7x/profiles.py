import numpy as np

class Points(object):
    def __init__(self, x1, x2, x3):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)


class FluxSurface(object):
    def __init__(self, x1, x2, x3, phi0, density=0, assymetric_island_density=0):
        self.points = Points(x1, x2, x3)
        self.phi0 = phi0
        self.density = density
        self.assymetric_island_density = assymetric_island_density

    def update_density(self, value, assymetric_island_density=None):
        self.density = value
        if assymetric_island_density is not None:
            self.assymetric_island_density = assymetric_island_density