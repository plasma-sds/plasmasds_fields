"""Unit tests for w7x.poincare."""
import types
import unittest
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from tests.helpers import get_test_data, TEST_POINCARE
from w7x.poincare import (Points, FluxSurface, Range, load_w7x_flux_surfaces,
                          filter_surfaces_by_range, label_surfaces,
                          filter_surfaces_by_type, filter_surfaces_by_radius,
                          filter_surfaces_by_error)


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


# ======================================================================
# Batch 3: label_surfaces
# ======================================================================

ISLAND_CENTRE = 6.236          # R of the synthetic island centre [m]


def regular_surface(n=20):
    """Points on the parabola R = 6.2 - 0.5 Z**2, |Z| <= 0.2 m."""
    z = np.linspace(-0.2, 0.2, n)
    return make_surface(6.2 - 0.5 * z**2, z)


def island_surface(n=200, half_width=0.012, half_height=0.15):
    """Points on an ellipse around (ISLAND_CENTRE, 0)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return make_surface(ISLAND_CENTRE + half_width * np.cos(theta),
                        half_height * np.sin(theta))


def label_quietly(surfaces, **kwargs):
    """label_surfaces, returning the labelled surfaces and any warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labelled = label_surfaces(surfaces, **kwargs)
    return labelled, caught


def is_island(surf):
    return surf.errors[1] is not None


class LabelSurfacesTest(unittest.TestCase):

    def test_regular_surface(self):
        surf = label_surfaces([regular_surface()])[0]
        np.testing.assert_array_equal(surf.points.point_type, 0)
        self.assertLess(surf.errors[0], 1e-12)
        self.assertEqual(surf.errors[1:], [None, None])
        np.testing.assert_allclose(surf.coeffs[0], [-0.5, 0.0, 6.2],
                                   atol=1e-10)
        np.testing.assert_array_equal(surf.coeffs[1:], 0.0)
        self.assertEqual(surf.density, [0, 0, 0])

    def test_island_by_point_count(self):
        surf = label_surfaces([island_surface(n=200)])[0]
        point_type = surf.points.point_type
        self.assertTrue(is_island(surf))
        self.assertIsNone(surf.errors[0])
        self.assertEqual((np.count_nonzero(point_type == 1),
                          np.count_nonzero(point_type == 2)), (100, 100))
        self.assertGreater(surf.points.r[point_type == 1].mean(),
                           ISLAND_CENTRE)
        self.assertLess(surf.points.r[point_type == 2].mean(), ISLAND_CENTRE)
        self.assertAlmostEqual(surf.coeffs[1, -1], ISLAND_CENTRE + 0.012,
                               delta=1e-3)
        self.assertAlmostEqual(surf.coeffs[2, -1], ISLAND_CENTRE - 0.012,
                               delta=1e-3)

    def test_island_by_fit_error(self):
        # 40 points (< limit_number), whole-surface fit error about 0.021
        surf = label_surfaces([island_surface(n=40, half_width=0.03)])[0]
        self.assertTrue(is_island(surf))
        self.assertEqual(set(np.unique(surf.points.point_type)), {1, 2})

    def test_limits_change_classification(self):
        cases = {
            "many points, high limits": (island_surface(n=200),
                                         dict(limit_number=1000)),
            "large error, high limit": (island_surface(n=40, half_width=0.03),
                                        dict(limit_error=1.0)),
            "error below default limit": (island_surface(n=40,
                                                         half_width=0.02),
                                          dict()),
        }
        for name, (surf, kwargs) in cases.items():
            with self.subTest(case=name):
                labelled = label_surfaces([surf], **kwargs)[0]
                self.assertFalse(is_island(labelled))
                np.testing.assert_array_equal(labelled.points.point_type, 0)

    def test_side_errors_from_own_points(self):
        # regression #27: side errors were computed over all points
        surf = label_surfaces([island_surface(n=200)])[0]
        r, z, point_type = surf.points.r, surf.points.z, surf.points.point_type
        for side in (1, 2):
            with self.subTest(side=side):
                own = point_type == side
                rms = np.sqrt(np.mean(
                    (r[own] - np.polyval(surf.coeffs[side], z[own]))**2))
                self.assertAlmostEqual(surf.errors[side], rms)

    def test_few_points_reduced_order(self):
        # regression: a single point got a fitted radius of about 2.09 m
        (single, pair), caught = label_quietly(
            [make_surface([6.25], [0.3]),
             make_surface([6.20, 6.22], [-0.1, 0.1])])
        self.assertEqual(caught, [])
        np.testing.assert_allclose(single.coeffs[0], [0.0, 0.0, 6.25])
        self.assertAlmostEqual(single.errors[0], 0.0)
        np.testing.assert_allclose(pair.coeffs[0], [0.0, 0.1, 6.21])
        self.assertAlmostEqual(pair.errors[0], 0.0)

    def test_order_parameter(self):
        z = np.linspace(-0.2, 0.2, 30)
        cubic = [0.3, -0.5, 0.1, 6.2]                 # highest power first
        for order in (3, 4):
            with self.subTest(order=order):
                surf = label_surfaces([make_surface(np.polyval(cubic, z), z)],
                                      order=order)[0]
                self.assertEqual(surf.coeffs.shape, (3, order + 1))
                self.assertAlmostEqual(surf.coeffs[0, -1], 6.2, places=8)
                if order == 3:
                    np.testing.assert_allclose(surf.coeffs[0], cubic,
                                               atol=1e-8)

    def test_window_warning(self):
        inside = make_surface([6.1, 6.2, 6.3], [-1.0, 0.0, 1.0])
        _, caught = label_quietly([inside])
        self.assertEqual(caught, [])
        cases = {"R < 6.1": make_surface([6.0, 6.2], [0.0, 0.1]),
                 "|Z| > 1.0": make_surface([6.2, 6.2], [0.0, 1.2])}
        for text, surf in cases.items():
            with self.subTest(case=text):
                with self.assertWarns(UserWarning) as context:
                    label_surfaces([surf])
                self.assertIn(text, str(context.warning))

    def test_labels_in_place(self):
        surfaces = [regular_surface(), island_surface()]
        labelled = label_surfaces(surfaces)
        for new, old in zip(labelled, surfaces):
            self.assertIs(new, old)
        self.assertTrue(is_island(surfaces[1]))

    def test_empty_list(self):
        self.assertEqual(label_surfaces([]), [])

    def test_invalid_parameters(self):
        type_errors = {"limit_error as string": dict(limit_error="0.01"),
                       "limit_number as float": dict(limit_number=100.0),
                       "order as float": dict(order=2.5)}
        value_errors = {"negative limit_error": dict(limit_error=-0.1),
                        "negative limit_number": dict(limit_number=-1),
                        "negative order": dict(order=-1)}
        for errors, exception in ((type_errors, TypeError),
                                  (value_errors, ValueError)):
            for name, kwargs in errors.items():
                with self.subTest(case=name):
                    with self.assertRaises(exception):
                        label_surfaces([regular_surface()], **kwargs)
        with self.subTest(case="surfaces as string"):
            with self.assertRaises(TypeError):
                label_surfaces("surfaces")


# ======================================================================
# Batch 4: filters on labelled surfaces
# ======================================================================

def hand_labelled(r, z, point_type, radii, errors):
    """A surface labelled by hand: fit radius (R at Z = 0) and fit error
    for each point type, None for point types without points."""
    surf = make_surface(r, z, point_type=point_type)
    surf.coeffs = np.zeros((3, 3))
    surf.coeffs[:, -1] = radii
    surf.errors = list(errors)
    surf.density = [0, 0, 0]
    return surf


def hand_labelled_surfaces():
    """A regular surface (radius 6.125, error 0.002) and an island with an
    inner side (6.25, 0.001) and an outer side (6.375, 0.004); all values
    are exact in floating point."""
    regular = hand_labelled([6.125] * 4, [-0.125, 0.0, 0.125, 0.25],
                            [0, 0, 0, 0], [6.125, 0, 0], [0.002, None, None])
    island = hand_labelled([6.375, 6.375, 6.25, 6.25], [-0.125, 0.125] * 2,
                           [1, 1, 2, 2], [0, 6.375, 6.25],
                           [None, 0.004, 0.001])
    return [regular, island]


def summary(surfaces):
    """(point count, sorted point types) of each surface."""
    return [(s.n, sorted(set(s.points.point_type.tolist())))
            for s in surfaces]


class FilterByTypeTest(unittest.TestCase):

    def setUp(self):
        self.surfaces = label_surfaces([regular_surface(), island_surface()])

    def test_selects_point_types(self):
        self.assertEqual(
            summary(filter_surfaces_by_type(self.surfaces, [1, 2])),
            [(200, [1, 2])])
        self.assertEqual(summary(filter_surfaces_by_type(self.surfaces, [0])),
                         [(20, [0])])
        self.assertEqual(summary(filter_surfaces_by_type(self.surfaces, [1])),
                         [(100, [1])])

    def test_none_keeps_all(self):
        self.assertEqual(summary(filter_surfaces_by_type(self.surfaces)),
                         [(20, [0]), (200, [1, 2])])

    def test_drops_surfaces_without_requested_types(self):
        kept = filter_surfaces_by_type(self.surfaces, [2])
        self.assertEqual(len(kept), 1)
        self.assertTrue(is_island(kept[0]))


class FilterByRadiusTest(unittest.TestCase):

    def setUp(self):
        self.surfaces = hand_labelled_surfaces()

    def test_selects_groups_by_radius(self):
        self.assertEqual(
            summary(filter_surfaces_by_radius(self.surfaces, [6.2, 6.3])),
            [(2, [2])])
        self.assertEqual(
            summary(filter_surfaces_by_radius(self.surfaces, [6.0, 6.4])),
            [(4, [0]), (4, [1, 2])])

    def test_bounds_exclusive(self):
        # the inner side (6.25) lies exactly on the lower bound
        self.assertEqual(
            summary(filter_surfaces_by_radius(self.surfaces, [6.25, 6.4])),
            [(2, [1])])

    def test_none_keeps_all(self):
        self.assertEqual(summary(filter_surfaces_by_radius(self.surfaces)),
                         [(4, [0]), (4, [1, 2])])


class FilterByErrorTest(unittest.TestCase):

    def setUp(self):
        self.surfaces = hand_labelled_surfaces()

    def test_selects_groups_by_error(self):
        self.assertEqual(
            summary(filter_surfaces_by_error(self.surfaces, [0.0005, 0.003])),
            [(4, [0]), (2, [2])])

    def test_bounds_exclusive(self):
        # the inner side (0.001) lies exactly on the lower bound
        self.assertEqual(
            summary(filter_surfaces_by_error(self.surfaces, [0.001, 0.005])),
            [(4, [0]), (2, [1])])

    def test_none_keeps_all(self):
        self.assertEqual(summary(filter_surfaces_by_error(self.surfaces)),
                         [(4, [0]), (4, [1, 2])])


class LabelledFiltersTest(unittest.TestCase):
    """Behaviour shared by the three filters on labelled surfaces; each is
    called with arguments that remove some points."""

    FILTERS = {
        "type": (filter_surfaces_by_type, dict(point_types=[1])),
        "radius": (filter_surfaces_by_radius, dict(R_range=[6.2, 6.3])),
        "error": (filter_surfaces_by_error,
                  dict(error_range=[0.0005, 0.003])),
    }

    def test_returns_copies(self):
        for name, (function, kwargs) in self.FILTERS.items():
            with self.subTest(filter=name):
                surfaces = hand_labelled_surfaces()
                kept = function(surfaces, **kwargs)
                self.assertEqual([s.n for s in surfaces], [4, 4])
                for surf in kept:
                    self.assertNotIn(surf, surfaces)

    def test_unlabelled_surfaces(self):
        unlabelled = make_surface([6.2, 6.21], [0.0, 0.1])
        for name, (function, kwargs) in self.FILTERS.items():
            for case, surfaces in (
                    ("unlabelled", [unlabelled]),
                    ("partially labelled",
                     hand_labelled_surfaces() + [unlabelled])):
                with self.subTest(filter=name, case=case):
                    with self.assertRaisesRegex(ValueError, "not labelled"):
                        function(surfaces, **kwargs)

    def test_empty_list(self):
        for name, (function, kwargs) in self.FILTERS.items():
            with self.subTest(filter=name):
                self.assertEqual(function([], **kwargs), [])

    def test_accepts_poincare_res_object(self):
        for name, (function, kwargs) in self.FILTERS.items():
            with self.subTest(filter=name):
                flt = types.SimpleNamespace(poincare_res=types.SimpleNamespace(
                    surfs=hand_labelled_surfaces()))
                self.assertEqual(
                    summary(function(flt, **kwargs)),
                    summary(function(hand_labelled_surfaces(), **kwargs)))

    def test_invalid_arguments(self):
        cases = [
            ("point_types outside 0-2", filter_surfaces_by_type,
             dict(point_types=[3]), ValueError),
            ("point_types as strings", filter_surfaces_by_type,
             dict(point_types=["1"]), TypeError),
            ("reversed R_range", filter_surfaces_by_radius,
             dict(R_range=[6.3, 6.2]), ValueError),
            ("non-numeric R_range", filter_surfaces_by_radius,
             dict(R_range=["a", "b"]), TypeError),
            ("reversed error_range", filter_surfaces_by_error,
             dict(error_range=[0.003, 0.001]), ValueError),
            ("non-numeric error_range", filter_surfaces_by_error,
             dict(error_range=["a", "b"]), TypeError),
        ]
        for name, function, kwargs, exception in cases:
            with self.subTest(case=name):
                with self.assertRaises(exception):
                    function(hand_labelled_surfaces(), **kwargs)


if __name__ == "__main__":
    unittest.main()
