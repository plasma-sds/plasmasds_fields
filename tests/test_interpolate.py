"""Unit tests for tools.interpolate."""
import unittest

import numpy as np

from tests.helpers import get_test_data, TEST_PROFILE
from plasmasds_fields.tools.interpolate import ProfileInterpolator1D

# unsorted, integer-valued profile; interpolated values are exact
POSITION = [6.30, 6.10, 6.20, 6.25]
PROFILE = [1, 4, 3, 2]


class ProfileInterpolatorTest(unittest.TestCase):

    def setUp(self):
        self.interp = ProfileInterpolator1D(POSITION, PROFILE)

    def test_construction(self):
        for name in ("position", "profile", "original_position",
                     "original_profile"):
            with self.subTest(attribute=name):
                self.assertEqual(getattr(self.interp, name).dtype, float)
        np.testing.assert_array_equal(self.interp.position, POSITION)
        np.testing.assert_array_equal(self.interp.profile, PROFILE)
        np.testing.assert_array_equal(self.interp.original_position, POSITION)
        np.testing.assert_array_equal(self.interp.original_profile, PROFILE)
        self.assertIsNot(self.interp.position, self.interp.original_position)

    def test_reproduces_datapoints(self):
        np.testing.assert_allclose(self.interp(POSITION), PROFILE)

    def test_linear_interpolation(self):
        self.assertAlmostEqual(float(self.interp(6.225)), 2.5)
        self.assertAlmostEqual(float(self.interp(6.15)), 3.5)

    def test_linear_extrapolation(self):
        # continues the segments 6.10-6.20 and 6.25-6.30
        self.assertAlmostEqual(float(self.interp(6.05)), 4.5)
        self.assertAlmostEqual(float(self.interp(6.35)), 0.0)

    def test_correct_by_position(self):
        self.interp.correct_datapoints(6.21, 5.0)     # nearest: 6.20
        np.testing.assert_allclose(self.interp.profile, [1, 4, 5, 2])
        self.assertAlmostEqual(float(self.interp(6.20)), 5.0)

    def test_correct_by_index(self):
        self.interp.correct_datapoints([0, 1], [7.0, 8.0], by_index=True)
        np.testing.assert_allclose(self.interp.profile, [7, 8, 3, 2])
        self.assertAlmostEqual(float(self.interp(6.30)), 7.0)

    def test_scalar_broadcast(self):
        self.interp.correct_datapoints([6.10, 6.30], 9.0)
        np.testing.assert_allclose(self.interp.profile, [9, 9, 3, 2])

    def test_corrections_cumulative(self):
        self.interp.correct_datapoints(1, 6.0, by_index=True)
        self.interp.correct_datapoints(2, 7.0, by_index=True)
        np.testing.assert_allclose(self.interp.profile, [1, 6, 7, 2])

    def test_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            self.interp.correct_datapoints([6.10, 6.20], [1.0, 2.0, 3.0])

    def test_integer_input_keeps_float_corrections(self):
        # regression: 2.7 was stored as 2 in an integer profile
        self.interp.correct_datapoints(6.25, 2.7)
        self.assertEqual(self.interp.profile[3], 2.7)

    def test_add_datapoints(self):
        self.interp.add_datapoints(6.40, 0.5)
        self.assertEqual(self.interp.position[-1], 6.40)
        self.assertEqual(self.interp.profile[-1], 0.5)
        self.assertAlmostEqual(float(self.interp(6.40)), 0.5)
        self.assertAlmostEqual(float(self.interp(6.35)), 0.75)
        self.interp.add_datapoints([6.50, 6.60], 0.0)
        np.testing.assert_array_equal(self.interp.position[-2:], [6.50, 6.60])
        np.testing.assert_array_equal(self.interp.profile[-2:], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            self.interp.add_datapoints([6.7, 6.8], [1.0, 2.0, 3.0])

    def test_indices_stable_after_add_and_reset(self):
        # regression: add_datapoints() and reset() used to re-sort the
        # arrays, so by_index corrections hit other datapoints
        self.interp.add_datapoints(6.0, 5.0)      # would sort to the front
        self.interp.correct_datapoints(0, 7.0, by_index=True)
        self.assertEqual(self.interp.position[0], 6.30)
        self.assertEqual(self.interp.profile[0], 7.0)
        self.interp.reset()
        self.interp.correct_datapoints(0, 8.0, by_index=True)
        self.assertEqual(self.interp.position[0], 6.30)
        self.assertEqual(self.interp.profile[0], 8.0)

    def test_reset(self):
        self.interp.correct_datapoints(1, 9.0, by_index=True)
        self.interp.add_datapoints(6.40, 0.5)
        self.interp.reset()
        np.testing.assert_array_equal(self.interp.position, POSITION)
        np.testing.assert_array_equal(self.interp.profile, PROFILE)
        np.testing.assert_allclose(self.interp(POSITION), PROFILE)
        self.assertAlmostEqual(float(self.interp(6.35)), 0.0)


class ProfileDatasetTest(unittest.TestCase):
    """ProfileInterpolator1D on the public test profile: row 0 holds the
    positions R [m] (decreasing), row 1 the density [m^-3]."""

    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(get_test_data(TEST_PROFILE))

    def test_reproduces_dataset(self):
        interp = ProfileInterpolator1D(self.data[0], self.data[1])
        np.testing.assert_array_equal(interp.position, self.data[0])
        np.testing.assert_allclose(interp(self.data[0]), self.data[1])


if __name__ == "__main__":
    unittest.main()
