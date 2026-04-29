import numpy as np
import xml.etree.ElementTree as ET

class Points(object):
    def __init__(self, x1, x2, x3):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)

    def __array__(self, dtype=None):
        arr = np.array([self.x1, self.x2, self.x3])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


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

def load_w7x_flux_surfaces(filename):
    """
    This function loads a W7-X flux surface stored in an xml file.
    """
    tree = ET.parse(filename)
    root = tree.getroot()  # {fltracer.gsoap.boz.hgw.ipp.mpg.de}Result {}
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