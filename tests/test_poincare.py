"""Unit tests for w7x.poincare."""
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from tests.helpers import get_test_data, TEST_POINCARE
from w7x.poincare import (Points, FluxSurface, Range, load_w7x_flux_surfaces,
                          filter_surfaces_by_range)


def make_surface(r, z, phi0=0.0, **kwargs):
    """FluxSurface in the plane x2 = 0, so that R = x1 and Z = x3."""
    r = np.asarray(r, dtype=float)
    return FluxSurface(r, np.zeros(r.size), z, phi0, **kwargs)


# ======================================================================
# Batch 1: data classes
# ======================================================================

class PointsTest(unittest.TestCase):

    def test_coordinates_stored_as_arrays(self):
        points = Points([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        for name in ("x1", "x2", "x3", "r", "z", "point_type"):
            with self.subTest(attribute=name):
                self.assertIsInstance(getattr(points, name), np.ndarray)
        np.testing.assert_array_equal(points.x1, [1.0, 2.0])
        np.testing.assert_array_equal(points.x2, [3.0, 4.0])
        np.testing.assert_array_equal(points.x3, [5.0, 6.0])

    def test_r_and_z_derived(self):
        points = Points([3.0, -6.0, 0.0], [4.0, 8.0, 2.0], [0.5, -0.5, 1.0])
        np.testing.assert_allclose(points.r, [5.0, 10.0, 2.0])
        np.testing.assert_array_equal(points.z, [0.5, -0.5, 1.0])

    def test_point_type(self):
        default = Points([1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3)
        self.assertEqual(default.point_type.dtype, np.uint8)
        np.testing.assert_array_equal(default.point_type, [0, 0, 0])
        scalar = Points([1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3, point_type=2)
        np.testing.assert_array_equal(scalar.point_type, [2, 2, 2])
        array = Points([1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3,
                       point_type=[0, 1, 2])
        np.testing.assert_array_equal(array.point_type, [0, 1, 2])

    def test_asarray(self):
        points = Points([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        stacked = np.asarray(points)
        self.assertEqual(stacked.shape, (3, 2))
        np.testing.assert_array_equal(stacked, [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(np.asarray(points, dtype=np.float32).dtype,
                         np.float32)


class FluxSurfaceTest(unittest.TestCase):

    def setUp(self):
        self.surf = make_surface([6.1, 6.2, 6.3, 6.4], [-0.1, 0.0, 0.1, 0.2],
                                 phi0=1.26, density=5e18)

    def test_initial_state(self):
        self.assertEqual(self.surf.n, 4)
        self.assertEqual(self.surf.phi0, 1.26)
        self.assertEqual(self.surf.coeffs, [None])
        self.assertEqual(self.surf.errors, [None])
        self.assertEqual(self.surf.density, [5e18])
        np.testing.assert_allclose(self.surf.points.r, [6.1, 6.2, 6.3, 6.4])

    def test_update_density(self):
        self.surf.density = [0, 0, 0]          # as after label_surfaces
        self.surf.update_density(2e19, ind=1)
        self.assertEqual(self.surf.density, [0, 2e19, 0])
        self.surf.update_density(3e19)
        self.assertEqual(self.surf.density, [3e19, 3e19, 3e19])

    def test_update_point_type(self):
        self.surf.update_point_type(1)
        np.testing.assert_array_equal(self.surf.points.point_type,
                                      [1, 1, 1, 1])
        self.surf.update_point_type(2, mask=self.surf.points.r > 6.25)
        np.testing.assert_array_equal(self.surf.points.point_type,
                                      [1, 1, 2, 2])

    def test_filter_points(self):
        self.surf.update_point_type([0, 1, 2, 1])
        self.surf.filter_points(np.array([True, False, True, False]))
        self.assertEqual(self.surf.n, 2)
        np.testing.assert_allclose(self.surf.points.x1, [6.1, 6.3])
        np.testing.assert_allclose(self.surf.points.x2, [0.0, 0.0])
        np.testing.assert_allclose(self.surf.points.x3, [-0.1, 0.1])
        np.testing.assert_allclose(self.surf.points.r, [6.1, 6.3])
        np.testing.assert_allclose(self.surf.points.z, [-0.1, 0.1])
        np.testing.assert_array_equal(self.surf.points.point_type, [0, 2])


class RangeTest(unittest.TestCase):

    def test_coordinate_extents(self):
        surfaces = [FluxSurface([1.0, 2.0], [0.5, -1.0], [0.3, -0.2], 0.0),
                    FluxSurface([-3.0, 0.0], [4.0, 1.5], [0.1, 0.4], 0.0)]
        extent = Range(surfaces)
        self.assertEqual(extent.x1, [-3.0, 2.0])
        self.assertEqual(extent.x2, [-1.0, 4.0])
        self.assertEqual(extent.x3, [-0.2, 0.4])
        self.assertEqual(extent.z, [-0.2, 0.4])
        np.testing.assert_allclose(extent.r, [np.hypot(1.0, 0.5),
                                              np.hypot(-3.0, 4.0)])
        self.assertEqual((extent.rMin, extent.rMax), tuple(extent.r))
        self.assertEqual((extent.zMin, extent.zMax), tuple(extent.z))

    def test_density_extent_ignores_empty_point_types(self):
        # labelled surfaces: point types without points keep density 0
        regular = make_surface([6.1, 6.2], [0.0, 0.1])
        regular.errors, regular.density = [1e-4, None, None], [3e19, 0, 0]
        island = make_surface([6.24, 6.23], [0.0, 0.1], point_type=[1, 2])
        island.errors, island.density = [None, 2e-4, 3e-4], [0, 5e17, 2e18]
        self.assertEqual(Range([regular, island]).density, [5e17, 3e19])

    def test_density_extent_unlabelled(self):
        surfaces = [make_surface([6.1, 6.2], [0.0, 0.1], density=4e18),
                    make_surface([6.2, 6.3], [0.0, 0.1], density=1e19)]
        self.assertEqual(Range(surfaces).density, [4e18, 1e19])


# ======================================================================
# Batch 2: loading and range filter
# ======================================================================

class LoadTest(unittest.TestCase):
    """load_w7x_flux_surfaces on the public test dataset, compared with the
    XML file read directly."""

    @classmethod
    def setUpClass(cls):
        path = get_test_data(TEST_POINCARE)
        cls.surfaces = load_w7x_flux_surfaces(path)
        cls.xml_surfaces = list(ET.parse(path).getroot())

    def xml_values(self, surf_element, name):
        """All values of one coordinate of a surface element, as floats."""
        return [float(e.text) for e in surf_element.iter()
                if e.tag.endswith("}" + name)]

    def test_one_surface_per_xml_entry(self):
        self.assertEqual(len(self.surfaces), len(self.xml_surfaces))
        for surf in self.surfaces:
            self.assertIsInstance(surf, FluxSurface)

    def test_phi0(self):
        for surf, element in zip(self.surfaces, self.xml_surfaces):
            self.assertEqual(surf.phi0, self.xml_values(element, "phi0")[0])

    def test_coordinates(self):
        for i in (0, len(self.surfaces) // 2, len(self.surfaces) - 1):
            surf, element = self.surfaces[i], self.xml_surfaces[i]
            with self.subTest(surface=i):
                for name in ("x1", "x2", "x3"):
                    np.testing.assert_array_equal(
                        getattr(surf.points, name),
                        self.xml_values(element, name))

    def test_point_counts(self):
        counts = [len(self.xml_values(e, "x1")) for e in self.xml_surfaces]
        self.assertEqual([s.n for s in self.surfaces], counts)


class FilterByRangeTest(unittest.TestCase):
    """Coordinates are multiples of 1/8, exact in floating point, so that
    points can lie exactly on a bound."""

    RADII = (6.0, 6.125, 6.25, 6.375)
    Z = (-0.25, -0.125, 0.0, 0.125, 0.25)

    def setUp(self):
        self.surfaces = [make_surface(np.full(len(self.Z), r), self.Z)
                         for r in self.RADII]

    def test_r_and_z_bounds_inclusive(self):
        kept = filter_surfaces_by_range(self.surfaces, r_range=[6.125, 6.25],
                                        z_range=[-0.125, 0.125])
        self.assertEqual([s.points.r[0] for s in kept], [6.125, 6.25])
        for surf in kept:
            np.testing.assert_array_equal(surf.points.z, [-0.125, 0.0, 0.125])

    def test_none_ranges_keep_all(self):
        kept = filter_surfaces_by_range(self.surfaces)
        self.assertEqual([s.n for s in kept], [5, 5, 5, 5])

    def test_surf_range(self):
        kept = filter_surfaces_by_range(self.surfaces, surf_range=[1, 3])
        self.assertEqual([s.points.r[0] for s in kept], [6.125, 6.25])
        clipped = filter_surfaces_by_range(self.surfaces, surf_range=[2, 10])
        self.assertEqual([s.points.r[0] for s in clipped], [6.25, 6.375])

    def test_returns_copies(self):
        kept = filter_surfaces_by_range(self.surfaces, z_range=[0.0, 0.25])
        self.assertEqual([s.n for s in self.surfaces], [5, 5, 5, 5])
        for new, old in zip(kept, self.surfaces):
            self.assertIsNot(new, old)

    def test_empty_surfaces_dropped(self):
        surfaces = self.surfaces + [make_surface([], [])]
        kept = filter_surfaces_by_range(surfaces, r_range=[6.2, 6.3])
        self.assertEqual([s.points.r[0] for s in kept], [6.25])
        self.assertEqual(filter_surfaces_by_range(surfaces,
                                                  r_range=[7.0, 8.0]), [])

    def test_accepts_poincare_res_object(self):
        flt = types.SimpleNamespace(
            poincare_res=types.SimpleNamespace(surfs=self.surfaces))
        kept = filter_surfaces_by_range(flt, r_range=[6.125, 6.25])
        self.assertEqual(len(kept), 2)

    def test_invalid_ranges(self):
        value_errors = {"reversed r_range": dict(r_range=[6.25, 6.125]),
                        "three-element z_range": dict(z_range=[0, 1, 2]),
                        "negative surf_range": dict(surf_range=[-1, 2])}
        type_errors = {"non-numeric r_range": dict(r_range=["a", "b"]),
                       "non-integer surf_range": dict(surf_range=[0.5, 2])}
        for errors, exception in ((value_errors, ValueError),
                                  (type_errors, TypeError)):
            for name, kwargs in errors.items():
                with self.subTest(case=name):
                    with self.assertRaises(exception):
                        filter_surfaces_by_range(self.surfaces, **kwargs)


if __name__ == "__main__":
    unittest.main()
