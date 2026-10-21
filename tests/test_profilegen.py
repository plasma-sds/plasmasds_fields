"""Unit tests for models.profilegen."""
import unittest

import numpy as np

from models.profilegen import create_2d_profile


def tanh_profile(R, edge_value, SOL_value, g, R_LCFS):
    """The documented profile of create_2d_profile along R."""
    return SOL_value + (edge_value - SOL_value) / 2 * (
        1 + np.tanh(g * (R - R_LCFS)))


class CreateProfileAxesTest(unittest.TestCase):
    """R and Z run from R0/Z0 up to, but not including, R1/Z1."""

    def test_default_axes(self):
        R, Z, _ = create_2d_profile()
        np.testing.assert_allclose(R, np.arange(0, 0.2, 0.0005))
        np.testing.assert_allclose(Z, np.arange(0, 0.3, 0.0005))
        self.assertEqual((R.size, Z.size), (400, 600))

    def test_custom_axes(self):
        R, Z, _ = create_2d_profile(R0=6.1, R1=6.3, Z0=-0.2, Z1=0.2,
                                    dR=0.01, dZ=0.05)
        self.assertAlmostEqual(R[0], 6.1)
        self.assertAlmostEqual(Z[0], -0.2)
        np.testing.assert_allclose(np.diff(R), 0.01)
        np.testing.assert_allclose(np.diff(Z), 0.05)
        self.assertLess(R[-1], 6.3)
        self.assertLess(Z[-1], 0.2)

    def test_end_point_excluded(self):
        R, Z, _ = create_2d_profile(R0=0, R1=0.3, dR=0.1, Z0=0, Z1=0.2, dZ=0.1)
        np.testing.assert_allclose(R, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(Z, [0.0, 0.1])


class CreateProfileFieldTest(unittest.TestCase):

    def test_shape(self):
        R, Z, field = create_2d_profile()
        self.assertEqual(field.shape, (R.size, Z.size))

    def test_uniform_along_z(self):
        _, _, field = create_2d_profile()
        np.testing.assert_array_equal(field, np.repeat(field[:, :1],
                                                       field.shape[1], axis=1))

    def test_follows_tanh_profile(self):
        params = dict(edge_value=3e19, SOL_value=1e18, g=35, R_LCFS=0.08)
        R, _, field = create_2d_profile(**params)
        np.testing.assert_allclose(field[:, 0], tanh_profile(R, **params))

    def test_mean_value_at_lcfs(self):
        R, _, _ = create_2d_profile()
        R_LCFS = R[150]                      # put the LCFS on a grid point
        _, _, field = create_2d_profile(R_LCFS=R_LCFS, edge_value=2e19,
                                        SOL_value=0.2e19)
        self.assertAlmostEqual(field[150, 0] / 1.1e19, 1.0)

    def test_end_values_far_from_lcfs(self):
        # with g = 20 [1/m], 1 m from the LCFS the tanh is saturated
        _, _, field = create_2d_profile(R0=-1.0, R1=1.2, dR=0.01, g=20,
                                        R_LCFS=0.1, edge_value=2e19,
                                        SOL_value=0.2e19)
        self.assertAlmostEqual(field[0, 0] / 0.2e19, 1.0)
        self.assertAlmostEqual(field[-1, 0] / 2e19, 1.0)


class CreateProfileOrientationTest(unittest.TestCase):
    """SOL_value at small R and edge_value at large R for g > 0, reversed
    for g < 0."""

    def test_positive_g_rises_towards_large_R(self):
        _, _, field = create_2d_profile(g=20, edge_value=2e19,
                                        SOL_value=0.2e19)
        self.assertTrue(np.all(np.diff(field[:, 0]) > 0))
        self.assertLess(abs(field[0, 0] - 0.2e19), abs(field[0, 0] - 2e19))
        self.assertLess(abs(field[-1, 0] - 2e19), abs(field[-1, 0] - 0.2e19))

    def test_negative_g_falls_towards_large_R(self):
        _, _, field = create_2d_profile(g=-20, edge_value=2e19,
                                        SOL_value=0.2e19)
        self.assertTrue(np.all(np.diff(field[:, 0]) < 0))
        self.assertLess(abs(field[0, 0] - 2e19), abs(field[0, 0] - 0.2e19))
        self.assertLess(abs(field[-1, 0] - 0.2e19), abs(field[-1, 0] - 2e19))

    def test_larger_g_gives_steeper_transition(self):
        R, _, gentle = create_2d_profile(g=20)
        _, _, steep = create_2d_profile(g=60)
        self.assertGreater(np.max(np.gradient(steep[:, 0], R)),
                           np.max(np.gradient(gentle[:, 0], R)))


if __name__ == "__main__":
    unittest.main()
