"""Unit tests for w7x.fields."""
import unittest

import numpy as np

from plasmasds_fields.w7x.fields import (add_density,
                                         filter_surfaces_by_density,
                                         extract_points,
                                         make_regular_density_field)
from plasmasds_fields.w7x.poincare import FluxSurface, label_surfaces


def make_surface(r, z, point_type=0):
    """FluxSurface in the plane x2 = 0, so that R = x1 and Z = x3."""
    r = np.asarray(r, dtype=float)
    return FluxSurface(r, np.zeros(r.size), z, 0.0, point_type=point_type)


def hand_labelled(r, z, point_type, radii, errors, density=(0, 0, 0)):
    """A surface labelled by hand: fit radius (R at Z = 0) and fit error per
    point type, None for point types without points."""
    surf = make_surface(r, z, point_type=point_type)
    surf.coeffs = np.zeros((3, 3))
    surf.coeffs[:, -1] = radii
    surf.errors = list(errors)
    surf.density = list(density)
    return surf


def hand_labelled_surfaces():
    """A regular surface (radius 6.125) and an island with an outer side
    (6.375, point type 1) and an inner side (6.25, point type 2)."""
    regular = hand_labelled([6.125] * 4, [-0.125, 0.0, 0.125, 0.25],
                            [0, 0, 0, 0], [6.125, 0, 0], [0.002, None, None])
    island = hand_labelled([6.375, 6.375, 6.25, 6.25], [-0.125, 0.125] * 2,
                           [1, 1, 2, 2], [0, 6.375, 6.25],
                           [None, 0.004, 0.001])
    return [regular, island]


def parabola_surface():
    """An unlabelled regular surface R = 6.2 - 0.5 Z**2, |Z| <= 0.2 m."""
    z = np.linspace(-0.2, 0.2, 20)
    return make_surface(6.2 - 0.5 * z**2, z)


def linear_density(r):
    """2e19 m^-3 at R = 6.10 m, falling by 8e19 per metre: 1.8e19 at the
    regular surface, 8e18 at the inner and -2e18 at the outer island side."""
    return 2e19 - 8e19 * (np.asarray(r, dtype=float) - 6.10)


def summary(surfaces):
    """(point count, sorted point types) of each surface."""
    return [(int(s.n), sorted(set(s.points.point_type.tolist())))
            for s in surfaces]


class AddDensityTest(unittest.TestCase):

    def setUp(self):
        self.surfaces = hand_labelled_surfaces()

    def test_density_from_group_radius(self):
        add_density(self.surfaces, linear_density)
        regular, island = self.surfaces
        self.assertAlmostEqual(regular.density[0] / 1.8e19, 1.0)
        self.assertAlmostEqual(island.density[1] / -2e18, 1.0)
        self.assertAlmostEqual(island.density[2] / 8e18, 1.0)

    def test_absent_point_types_stay_zero(self):
        add_density(self.surfaces, linear_density)
        regular, island = self.surfaces
        self.assertEqual(regular.density[1:], [0, 0])
        self.assertEqual(island.density[0], 0)

    def test_unusable_groups_removed(self):
        # the outer island side has a negative density
        kept = add_density(self.surfaces, linear_density)
        self.assertEqual(summary(kept), [(4, [0]), (2, [2])])
        # zero and NaN count as unusable too; a surface without a usable
        # group is dropped
        values = {6.125: 0.0, 6.375: 1e19, 6.25: np.nan}
        kept = add_density(hand_labelled_surfaces(),
                           lambda r: values[float(r)])
        self.assertEqual(summary(kept), [(2, [1])])

    def test_keep_unusable(self):
        kept = add_density(self.surfaces, linear_density, keep_unusable=True)
        self.assertEqual(summary(kept), [(4, [0]), (4, [1, 2])])
        self.assertLess(kept[1].density[1], 0)

    def test_input_keeps_points(self):
        kept = add_density(self.surfaces, linear_density)
        self.assertEqual([s.n for s in self.surfaces], [4, 4])
        # densities are written onto the input surfaces
        self.assertAlmostEqual(self.surfaces[1].density[2] / 8e18, 1.0)
        # an untrimmed surface is returned as it is, a trimmed one is a copy
        self.assertIs(kept[0], self.surfaces[0])
        self.assertIsNot(kept[1], self.surfaces[1])

    def test_unlabelled_input_labelled_with_warning(self):
        cases = {"unlabelled": [parabola_surface()],
                 "partially labelled": label_surfaces([parabola_surface()])
                 + [parabola_surface()]}
        for name, surfaces in cases.items():
            with self.subTest(case=name):
                with self.assertWarnsRegex(UserWarning, "not labelled"):
                    kept = add_density(surfaces, linear_density)
                for surf in kept:
                    self.assertNotEqual(surf.errors, [None])
                    self.assertAlmostEqual(surf.density[0] / 1.2e19, 1.0)

    def test_invalid_input(self):
        with self.assertRaisesRegex(TypeError, "density_function must be "
                                               "callable"):
            add_density(self.surfaces, 5)
        with self.assertRaises(TypeError):
            add_density("surfaces", linear_density)


class FilterByDensityTest(unittest.TestCase):

    def setUp(self):
        regular, island = hand_labelled_surfaces()
        regular.density = [1.8e19, 0, 0]
        island.density = [0, 5e17, 8e18]
        self.surfaces = [regular, island]

    def test_selects_groups_by_density(self):
        # the inner island side (8e18) lies exactly on the lower bound
        kept = filter_surfaces_by_density(self.surfaces, [8e18, 2e19])
        self.assertEqual(summary(kept), [(4, [0]), (2, [2])])

    def test_none_keeps_all(self):
        self.assertEqual(summary(filter_surfaces_by_density(self.surfaces)),
                         [(4, [0]), (4, [1, 2])])

    def test_ignores_empty_point_types(self):
        # the range contains the placeholder 0 of the point types without
        # points, but none of the real densities
        self.assertEqual(filter_surfaces_by_density(self.surfaces, [0, 1e17]),
                         [])

    def test_returns_copies(self):
        kept = filter_surfaces_by_density(self.surfaces, [8e18, 2e19])
        self.assertEqual([s.n for s in self.surfaces], [4, 4])
        for surf in kept:
            self.assertNotIn(surf, self.surfaces)

    def test_no_density_assigned(self):
        with self.assertRaisesRegex(ValueError, "No density assigned"):
            filter_surfaces_by_density(hand_labelled_surfaces(), [0, 1e19])

    def test_invalid_range(self):
        cases = [("reversed", [2e19, 8e18], ValueError),
                 ("negative", [-1e18, 1e19], ValueError),
                 ("non-numeric", ["a", "b"], TypeError)]
        for name, density_range, exception in cases:
            with self.subTest(case=name):
                with self.assertRaises(exception):
                    filter_surfaces_by_density(self.surfaces, density_range)


class ExtractPointsTest(unittest.TestCase):

    def setUp(self):
        regular, island = hand_labelled_surfaces()
        regular.density = [1.8e19, 0, 0]
        island.density = [0, 5e17, 8e18]
        self.surfaces = [regular, island]

    def test_keys_and_shapes(self):
        data = extract_points(self.surfaces)
        self.assertEqual(set(data), {"x1", "x2", "x3", "r", "z", "density",
                                     "point_type", "surface_radius"})
        for key, values in data.items():
            with self.subTest(key=key):
                self.assertIsInstance(values, np.ndarray)
                self.assertEqual(values.shape, (8,))

    def test_values_per_point(self):
        data = extract_points(self.surfaces)
        np.testing.assert_allclose(data["r"], [6.125] * 4 + [6.375] * 2
                                   + [6.25] * 2)
        np.testing.assert_allclose(data["x1"], data["r"])
        np.testing.assert_array_equal(data["x2"], 0.0)
        np.testing.assert_allclose(data["z"], [-0.125, 0.0, 0.125, 0.25]
                                   + [-0.125, 0.125] * 2)
        np.testing.assert_array_equal(data["x3"], data["z"])
        np.testing.assert_array_equal(data["point_type"],
                                      [0, 0, 0, 0, 1, 1, 2, 2])
        # density and surface radius of each point's own point-type group
        np.testing.assert_allclose(data["density"], [1.8e19] * 4
                                   + [5e17] * 2 + [8e18] * 2)
        np.testing.assert_allclose(data["surface_radius"], data["r"])

    def test_no_density_assigned(self):
        with self.assertRaisesRegex(ValueError, "No density assigned"):
            extract_points(hand_labelled_surfaces())


class RegularDensityFieldTest(unittest.TestCase):
    """Coordinates and spacings are multiples of 1/16, exact in floating
    point."""

    def setUp(self):
        # corners and centre of the square R 6.0-6.25 m, Z -0.25-0.25 m;
        # the density rises along R only
        self.r = np.array([6.0, 6.25, 6.0, 6.25, 6.125])
        self.z = np.array([-0.25, -0.25, 0.25, 0.25, 0.0])
        self.density = 1e19 * (1 + 4 * (self.r - 6.0))

    def test_grid_axes_and_layout(self):
        R, Z, grid = make_regular_density_field(self.r, self.z, self.density,
                                                dr=0.0625, dz=0.0625)
        np.testing.assert_array_equal(R, np.arange(5) * 0.0625 + 6.0)
        np.testing.assert_array_equal(Z, np.arange(9) * 0.0625 - 0.25)
        self.assertEqual(grid.shape, (Z.size, R.size))      # (NZ, NR)
        # grid[i, j] belongs to (R[j], Z[i]): constant along Z, rising in R
        np.testing.assert_allclose(grid, np.broadcast_to(grid[0], grid.shape))
        self.assertTrue(np.all(np.diff(grid[0]) > 0))

    def test_limits(self):
        for limits in ([6.0625, 6.1875], (6.0625, 6.1875),
                       np.array([6.0625, 6.1875])):
            with self.subTest(type=type(limits).__name__):
                R, Z, grid = make_regular_density_field(
                    self.r, self.z, self.density, dr=0.0625, dz=0.125,
                    r_limits=limits, z_limits=[-0.125, 0.125])
                np.testing.assert_array_equal(R, [6.0625, 6.125, 6.1875])
                np.testing.assert_array_equal(Z, [-0.125, 0.0, 0.125])
                self.assertEqual(grid.shape, (3, 3))

    def test_nearest_fill_and_floor(self):
        # data only in the lower-left triangle of the square, constant 5e18
        r, z = np.array([6.0, 6.25, 6.0]), np.array([-0.25, -0.25, 0.25])
        density = np.full(3, 5e18)
        kwargs = dict(dr=0.0625, dz=0.0625, r_limits=[6.0, 6.25],
                      z_limits=[-0.25, 0.25], bottom_value=1e16)
        _, _, filled = make_regular_density_field(r, z, density, **kwargs)
        _, _, floored = make_regular_density_field(r, z, density,
                                                   fill_with_nearest=False,
                                                   **kwargs)
        # corner (6.25, 0.25) lies outside the data, (6.0, -0.25) inside
        self.assertEqual(filled[-1, -1], 5e18)
        self.assertEqual(floored[-1, -1], 1e16)
        self.assertEqual(floored[0, 0], 5e18)
        # values below the floor are raised to it
        _, _, low = make_regular_density_field(r, z, np.full(3, 1e15),
                                               **kwargs)
        np.testing.assert_array_equal(low, 1e16)

    def test_invalid_input(self):
        cases = [("dr zero", dict(dr=0), ValueError),
                 ("dz negative", dict(dz=-0.1), ValueError),
                 ("bottom_value negative", dict(bottom_value=-1.0),
                  ValueError),
                 ("reversed r_limits", dict(r_limits=[6.25, 6.0]),
                  ValueError)]
        for name, kwargs, exception in cases:
            with self.subTest(case=name):
                with self.assertRaises(exception):
                    make_regular_density_field(self.r, self.z, self.density,
                                               **kwargs)
        with self.subTest(case="non-numeric r"):
            with self.assertRaises(TypeError):
                make_regular_density_field(["a", "b", "c"], self.z[:3],
                                           self.density[:3])


if __name__ == "__main__":
    unittest.main()
