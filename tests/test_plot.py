"""Unit tests for tools.plot."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")          # no display; set before pyplot is used
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.contour import QuadContourSet
from matplotlib.tri import TriContourSet
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from tools.plot import contour_field, slice_field, animate_field

# small grid with different axis lengths, so layouts can be told apart
R = 6.0 + 0.01 * np.arange(12)          # [m]
Z = -0.05 + 0.01 * np.arange(10)        # [m]
T = 1e-6 * np.arange(5)                 # [s]
RR, ZZ = np.meshgrid(R, Z)              # (NZ, NR)
FIELD_ZR = 1e19 * (1 + 10 * (RR - 6.0)) + 1e17 * ZZ   # rises along R

# (t, Z, R) field whose values encode their indices
TI, ZI, RI = np.meshgrid(np.arange(T.size), np.arange(Z.size),
                         np.arange(R.size), indexing="ij")
FIELD_TZR = 1000.0 * TI + 10.0 * ZI + RI


class PlotTestCase(unittest.TestCase):
    """Replaces plt.show with a no-op, so figures stay open for inspection,
    and closes all figures after each test."""

    def setUp(self):
        patcher = mock.patch("matplotlib.pyplot.show")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(plt.close, "all")


class ContourFieldTest(PlotTestCase):

    def test_regular_grid_both_layouts(self):
        cs = contour_field(R, Z, FIELD_ZR)
        cs_transposed = contour_field(R, Z, FIELD_ZR.T)
        self.assertIsInstance(cs, QuadContourSet)
        np.testing.assert_allclose(cs_transposed.levels, cs.levels)
        self.assertEqual((cs_transposed.zmin, cs_transposed.zmax),
                         (cs.zmin, cs.zmax))

    def test_scattered_data(self):
        rng = np.random.default_rng(0)
        r = rng.uniform(R[0], R[-1], 200)
        z = rng.uniform(Z[0], Z[-1], 200)
        cs = contour_field(r, z, 1e19 * (r - 5.9))
        self.assertIsInstance(cs, TriContourSet)

    def test_ranges_crop(self):
        cs = contour_field(R, Z, FIELD_ZR, r_range=[R[3], R[7]],
                           z_range=[Z[2], Z[6]])
        cropped = FIELD_ZR[2:7, 3:8]
        self.assertAlmostEqual(cs.zmin / cropped.min(), 1.0)
        self.assertAlmostEqual(cs.zmax / cropped.max(), 1.0)
        np.testing.assert_allclose(cs.axes.get_xlim(), [R[3], R[7]])

    def test_save_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "field.png")
            contour_field(R, Z, FIELD_ZR, save_image=path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)


class SliceFieldTest(PlotTestCase):

    def assertSlice(self, artist, expected, xlabel, ylabel, title):
        self.assertIsInstance(artist, QuadContourSet)
        self.assertEqual((artist.zmin, artist.zmax),
                         (expected.min(), expected.max()))
        self.assertEqual(artist.axes.get_xlabel(), xlabel)
        self.assertEqual(artist.axes.get_ylabel(), ylabel)
        self.assertIn(title, artist.axes.get_title())

    def test_slice_selectors(self):
        cases = {
            "at_t": (dict(at_t=T[2] + 0.3e-6), FIELD_TZR[2], "R [m]",
                     "Z [m]", "at t = 0.00 s"),
            "at_r": (dict(at_r=R[4] + 0.002), FIELD_TZR[:, :, 4],
                     "Time [s]", "Z [m]", "at R = 6.040 m"),
            "at_z": (dict(at_z=Z[3] - 0.002), FIELD_TZR[:, 3, :],
                     "Time [s]", "R [m]", "at Z = -0.020 m"),
        }
        for name, (kwargs, expected, xlabel, ylabel, title) in cases.items():
            with self.subTest(selector=name):
                artist = slice_field(R, Z, T, FIELD_TZR, "density", **kwargs)
                self.assertSlice(artist, expected, xlabel, ylabel, title)

    def test_layout_detection(self):
        hesel = slice_field(R, Z, T, FIELD_TZR, "density", at_t=T[3])
        motion_layout = np.transpose(FIELD_TZR, (0, 2, 1))     # (t, R, Z)
        for name, kwargs in (("detected", dict()),
                             ("explicit", dict(axis_order=("t", "R", "Z")))):
            with self.subTest(layout=name):
                motion = slice_field(R, Z, T, motion_layout, "density",
                                     at_t=T[3], **kwargs)
                np.testing.assert_allclose(motion.levels, hesel.levels)
                self.assertEqual((motion.zmin, motion.zmax),
                                 (hesel.zmin, hesel.zmax))

    def test_plot_3d(self):
        artist = slice_field(R, Z, T, FIELD_TZR, "density", at_t=T[1],
                             plot_3d=True)
        self.assertIsInstance(artist, Poly3DCollection)


class AnimateFieldTest(PlotTestCase):

    def animate(self, fields, labels, path):
        """animate_field with a small figure, its progress bar silenced."""
        with contextlib.redirect_stdout(io.StringIO()):
            animate_field(R, Z, T, fields, "density", labels, path, dpi=20)

    def test_writes_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "density")          # no extension
            self.animate(FIELD_TZR, "n [m^-3]", path)
            gif = path + ".gif"
            self.assertTrue(os.path.isfile(gif))
            with Image.open(gif) as image:
                self.assertEqual(image.n_frames, T.size)

    def test_multiple_fields(self):
        # default figure size (6 * ncols, 4.5 * nrows) inches at 20 dpi
        sizes = {1: (120, 90), 2: (240, 90), 3: (360, 90), 4: (240, 180)}
        with tempfile.TemporaryDirectory() as tmp:
            for n, size in sizes.items():
                with self.subTest(fields=n):
                    path = os.path.join(tmp, f"fields_{n}.gif")
                    self.animate([FIELD_TZR] * n, ["n [m^-3]"] * n, path)
                    with Image.open(path) as image:
                        self.assertEqual(image.size, size)
                        self.assertEqual(image.n_frames, T.size)


if __name__ == "__main__":
    unittest.main()
