"""Unit tests for the HESEL class in models.hesel."""
import os
import pathlib
import tempfile
import unittest

import numpy as np

from models.hesel import HESEL
from tests.helpers import (HESEL_R, HESEL_Z, HESEL_T, HESEL_FIELDS,
                           hesel_values, write_hesel_file)

DR = HESEL_R[1] - HESEL_R[0]
DZ = HESEL_Z[1] - HESEL_Z[0]
DT = HESEL_T[1] - HESEL_T[0]
DENSITY = hesel_values(HESEL_FIELDS["density"])      # (t, Z, R)


class HeselTestCase(unittest.TestCase):
    """Writes the synthetic HESEL file to a temporary directory and opens
    it as ``self.hesel``; both are cleaned up after each test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = write_hesel_file(os.path.join(tmp.name, "hesel.h5"))
        self.hesel = HESEL(self.path)
        self.addCleanup(self.hesel.close)        # runs before tmp.cleanup


class HeselAxesTest(HeselTestCase):

    def test_axes(self):
        np.testing.assert_allclose(self.hesel.R_axis, HESEL_R)
        np.testing.assert_allclose(self.hesel.Z_axis, HESEL_Z)
        np.testing.assert_allclose(self.hesel.time_axis, HESEL_T)

    def test_axes_are_one_dimensional(self):
        for axis in (self.hesel.R_axis, self.hesel.Z_axis,
                     self.hesel.time_axis):
            with self.subTest(size=axis.size):
                self.assertEqual(axis.ndim, 1)

    def test_spacings(self):
        self.assertAlmostEqual(self.hesel.dR, DR)
        self.assertAlmostEqual(self.hesel.dZ, DZ)
        self.assertAlmostEqual(self.hesel.dt, DT)

    def test_path_is_stored_as_string(self):
        with HESEL(pathlib.Path(self.path)) as hesel:
            self.assertEqual(hesel.path, str(self.path))


class HeselExtractFieldTest(HeselTestCase):

    def test_whole_fields(self):
        for name, base in HESEL_FIELDS.items():
            with self.subTest(field=name):
                field = self.hesel.extract_field(name)
                self.assertEqual(field.shape,
                                 (HESEL_T.size, HESEL_Z.size, HESEL_R.size))
                np.testing.assert_allclose(field, hesel_values(base))

    def test_scalar_selects_nearest_sample_and_drops_axis(self):
        r = HESEL_R[4] + 0.4 * DR                 # nearest: index 4
        z = HESEL_Z[6] - 0.3 * DZ                 # nearest: index 6
        t = HESEL_T[2] + 0.45 * DT                # nearest: index 2
        cases = {
            "r_range": (dict(r_range=r), DENSITY[:, :, 4]),
            "z_range": (dict(z_range=z), DENSITY[:, 6, :]),
            "t_range": (dict(t_range=t), DENSITY[2, :, :]),
        }
        for name, (kwargs, expected) in cases.items():
            with self.subTest(selector=name):
                field = self.hesel.extract_field("density", **kwargs)
                self.assertEqual(field.ndim, 2)
                np.testing.assert_allclose(field, expected)

    def test_scalar_rounds_to_the_closer_sample(self):
        below = self.hesel.extract_field("density",
                                         r_range=HESEL_R[4] + 0.4 * DR)
        above = self.hesel.extract_field("density",
                                         r_range=HESEL_R[4] + 0.6 * DR)
        np.testing.assert_allclose(below, DENSITY[:, :, 4])
        np.testing.assert_allclose(above, DENSITY[:, :, 5])

    def test_one_element_sequence_acts_as_scalar(self):
        field = self.hesel.extract_field("density", r_range=[HESEL_R[3]])
        np.testing.assert_allclose(field, DENSITY[:, :, 3])

    def test_range_keeps_block_between_nearest_samples(self):
        r_range = [HESEL_R[2] + 0.1 * DR, HESEL_R[7] - 0.1 * DR]
        field = self.hesel.extract_field("density", r_range=r_range)
        np.testing.assert_allclose(field, DENSITY[:, :, 2:8])

    def test_reversed_range_gives_same_block(self):
        forward = self.hesel.extract_field(
            "density", z_range=[HESEL_Z[1], HESEL_Z[5]])
        reverse = self.hesel.extract_field(
            "density", z_range=[HESEL_Z[5], HESEL_Z[1]])
        np.testing.assert_allclose(forward, DENSITY[:, 1:6, :])
        np.testing.assert_allclose(reverse, forward)

    def test_range_beyond_axis_keeps_whole_axis(self):
        field = self.hesel.extract_field("density", t_range=[-1.0, 1.0])
        np.testing.assert_allclose(field, DENSITY)

    def test_scalar_beyond_axis_selects_end_sample(self):
        field = self.hesel.extract_field("density", r_range=10.0)
        np.testing.assert_allclose(field, DENSITY[:, :, -1])

    def test_combined_selection_keeps_t_z_r_order(self):
        field = self.hesel.extract_field(
            "electron_temperature",
            t_range=[HESEL_T[1], HESEL_T[3]],
            z_range=HESEL_Z[4],
            r_range=[HESEL_R[0], HESEL_R[5]])
        expected = hesel_values(HESEL_FIELDS["electron_temperature"])
        np.testing.assert_allclose(field, expected[1:4, 4, 0:6])
        self.assertEqual(field.shape, (3, 6))

    def test_all_scalars_give_single_value(self):
        field = self.hesel.extract_field("ion_temperature",
                                         t_range=HESEL_T[4],
                                         z_range=HESEL_Z[0],
                                         r_range=HESEL_R[11])
        expected = hesel_values(HESEL_FIELDS["ion_temperature"])[4, 0, 11]
        self.assertEqual(np.ndim(field), 0)
        self.assertAlmostEqual(float(field) / expected, 1.0)

    def test_invalid_field_name(self):
        with self.assertRaisesRegex(ValueError, "field must be one of"):
            self.hesel.extract_field("potential")

    def test_range_with_more_than_two_elements(self):
        with self.assertRaisesRegex(ValueError, "two-element"):
            self.hesel.extract_field("density",
                                     r_range=[HESEL_R[0], HESEL_R[1],
                                              HESEL_R[2]])


class HeselFileHandlingTest(HeselTestCase):

    def test_context_manager_closes_file(self):
        with HESEL(self.path) as hesel:
            self.assertIsNotNone(hesel.file)
            self.assertTrue(hesel.file.id.valid)
            handle = hesel.file
        self.assertIsNone(hesel.file)
        self.assertFalse(handle.id.valid)

    def test_close_twice_is_safe(self):
        self.hesel.close()
        self.hesel.close()
        self.assertIsNone(self.hesel.file)

    def test_closed_file_can_be_replaced(self):
        # the file is released on close, so it can be written again
        self.hesel.close()
        write_hesel_file(self.path)
        with HESEL(self.path) as hesel:
            np.testing.assert_allclose(hesel.R_axis, HESEL_R)

    def test_repr_shows_path_and_axis_sizes(self):
        text = repr(self.hesel)
        self.assertIn(repr(str(self.path)), text)
        self.assertIn(f"R={HESEL_R.size}", text)
        self.assertIn(f"Z={HESEL_Z.size}", text)
        self.assertIn(f"time={HESEL_T.size}", text)


if __name__ == "__main__":
    unittest.main()
