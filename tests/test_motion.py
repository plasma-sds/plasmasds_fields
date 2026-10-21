"""Unit tests for models.motion."""
import os
import tempfile
import unittest

import h5py
import numpy as np

from models.motion import (perturbation_gaussian, perturbation_sine,
                           import_density_field)
from models.profilegen import create_2d_profile

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))       # FWHM = 2.3548 sigma

# grid for the Gaussian tests (dR != dZ on purpose) and empty background
R = np.arange(0, 0.2, 0.001)                     # 200 points [m]
Z = np.arange(0, 0.3, 0.002)                     # 150 points [m]
EMPTY = np.zeros((R.size, Z.size))


def gaussian_frame(r0, z0, fwhm, ampl, R=R, Z=Z):
    """Expected filament: a round Gaussian within ±3 sigma of its centre,
    zero outside."""
    sigma = fwhm / FWHM_TO_SIGMA
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    frame = ampl * np.exp(-((RR - r0)**2 + (ZZ - z0)**2) / (2 * sigma**2))
    window = (np.abs(RR - r0) <= 3 * sigma) & (np.abs(ZZ - z0) <= 3 * sigma)
    return np.where(window, frame, 0.0)


def half_max_width(x, y):
    """Full width at half maximum of a single peak, with linear
    interpolation of the two half-maximum crossings."""
    half = y.max() / 2
    above = np.where(y >= half)[0]
    lo, hi = above[0], above[-1]
    x_lo = np.interp(half, [y[lo - 1], y[lo]], [x[lo - 1], x[lo]])
    x_hi = np.interp(half, [y[hi + 1], y[hi]], [x[hi + 1], x[hi]])
    return x_hi - x_lo


def gaussian(r_traj, z_traj, fwhm=0.02, ampl=1.0, field=EMPTY, dt=1e-6,
             **kwargs):
    """perturbation_gaussian on the test grid, returning the frames."""
    return perturbation_gaussian(R, Z, field, dt, r_traj, z_traj, fwhm,
                                 ampl, **kwargs)["field"]


class TemporaryDirectoryTestCase(unittest.TestCase):
    """Provides ``self.tmp``, a temporary directory removed after the
    test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name


# ======================================================================
# perturbation_gaussian
# ======================================================================

class GaussianSingleFilamentTest(unittest.TestCase):

    def test_output_dictionary(self):
        data = perturbation_gaussian(R, Z, EMPTY, 2e-6, [[0.1, 0.1, 0.1]],
                                     [[0.15, 0.15, 0.15]], 0.02, 1.0)
        self.assertEqual(set(data), {"R", "Z", "t", "field"})
        np.testing.assert_allclose(data["R"], R)
        np.testing.assert_allclose(data["Z"], Z)
        np.testing.assert_allclose(data["t"], [0, 2e-6, 4e-6])
        self.assertEqual(data["field"].shape, (3, R.size, Z.size))

    def test_peak_position_and_amplitude(self):
        frame = gaussian([[R[80]]], [[Z[60]]], ampl=3e18)[0]
        self.assertEqual(np.unravel_index(frame.argmax(), frame.shape),
                         (80, 60))
        self.assertAlmostEqual(frame.max() / 3e18, 1.0)

    def test_matches_analytic_gaussian(self):
        r0, z0 = 0.0734, 0.1611                  # between grid points
        frame = gaussian([[r0]], [[z0]], fwhm=0.015, ampl=2.0)[0]
        np.testing.assert_allclose(frame, gaussian_frame(r0, z0, 0.015, 2.0),
                                   atol=1e-12)

    def test_measured_fwhm_equals_requested(self):
        # regression #32: sigma = FWHM / 2.3548, not FWHM / 2.535
        R_fine = np.arange(0, 0.2, 0.0002)
        Z_fine = np.arange(0, 0.2, 0.0002)
        for fwhm in (0.01, 0.02):
            with self.subTest(fwhm=fwhm):
                frame = perturbation_gaussian(
                    R_fine, Z_fine, np.zeros((R_fine.size, Z_fine.size)),
                    1e-6, [[0.1]], [[0.1]], fwhm, 1.0)["field"][0]
                ir, iz = np.unravel_index(frame.argmax(), frame.shape)
                self.assertAlmostEqual(half_max_width(R_fine, frame[:, iz]),
                                       fwhm, delta=0.002 * fwhm)
                self.assertAlmostEqual(half_max_width(Z_fine, frame[ir, :]),
                                       fwhm, delta=0.002 * fwhm)

    def test_present_at_every_position(self):
        # regression #32: the patch was built in (Z, R) order, so filaments
        # vanished whenever the window held different numbers of R and Z
        # samples (grid edges, dR != dZ)
        for r0 in np.linspace(R[0], R[-1], 41):
            for z0 in (Z[0], 0.15, Z[-1]):
                with self.subTest(r0=r0, z0=z0):
                    frame = gaussian([[r0]], [[z0]])[0]
                    np.testing.assert_allclose(
                        frame, gaussian_frame(r0, z0, 0.02, 1.0), atol=1e-12)

    def test_background_is_added_and_not_modified(self):
        _, _, background = create_2d_profile(R1=0.2, Z1=0.3, dR=0.001,
                                             dZ=0.002)
        original = background.copy()
        frame = gaussian([[0.1]], [[0.15]], ampl=1e18,
                         field=background)[0]
        np.testing.assert_allclose(frame - background,
                                   gaussian_frame(0.1, 0.15, 0.02, 1e18),
                                   atol=1e3)
        np.testing.assert_array_equal(background, original)


class GaussianSeveralFilamentsTest(unittest.TestCase):

    def test_filaments_add_up(self):
        both = gaussian([[0.05], [0.12]], [[0.1], [0.12]])[0]
        first = gaussian([[0.05]], [[0.1]])[0]
        second = gaussian([[0.12]], [[0.12]])[0]
        np.testing.assert_allclose(both, first + second)

    def test_per_step_fwhm_and_amplitude(self):
        frames = gaussian([[0.1, 0.1]], [[0.15, 0.15]],
                          fwhm=[[0.01, 0.03]], ampl=[[1.0, 4.0]])
        np.testing.assert_allclose(frames[0],
                                   gaussian_frame(0.1, 0.15, 0.01, 1.0),
                                   atol=1e-12)
        np.testing.assert_allclose(frames[1],
                                   gaussian_frame(0.1, 0.15, 0.03, 4.0),
                                   atol=1e-12)

    def test_missing_positions_skip_the_filament(self):
        cases = {"None": [[0.1, None, 0.1]], "NaN": [[0.1, np.nan, 0.1]]}
        for name, r_traj in cases.items():
            with self.subTest(missing=name):
                frames = gaussian(r_traj, [[0.15, 0.15, 0.15]])
                np.testing.assert_array_equal(frames[1], EMPTY)
                self.assertGreater(frames[0].max(), 0.5)
                self.assertGreater(frames[2].max(), 0.5)

    def test_missing_z_position_skips_the_filament(self):
        frames = gaussian([[0.1, 0.1]], [[0.15, np.nan]])
        np.testing.assert_array_equal(frames[1], EMPTY)

    def test_filament_off_the_grid_is_skipped(self):
        frames = gaussian([[0.1, 5.0]], [[0.15, 0.15]])
        np.testing.assert_array_equal(frames[1], EMPTY)

    def test_trajectories_of_different_lengths(self):
        # the longest trajectory sets the number of frames; a filament is
        # absent after its own data ends
        frames = gaussian([[0.05], [0.12, 0.13, 0.14]],
                          [[0.1], [0.2, 0.2, 0.2]])
        self.assertEqual(frames.shape[0], 3)
        np.testing.assert_allclose(frames[0],
                                   gaussian_frame(0.05, 0.1, 0.02, 1.0)
                                   + gaussian_frame(0.12, 0.2, 0.02, 1.0),
                                   atol=1e-12)
        np.testing.assert_allclose(frames[2],
                                   gaussian_frame(0.14, 0.2, 0.02, 1.0),
                                   atol=1e-12)

    def test_short_fwhm_list_ends_the_filament(self):
        frames = gaussian([[0.1, 0.1]], [[0.15, 0.15]], fwhm=[[0.02]])
        self.assertGreater(frames[0].max(), 0.5)
        np.testing.assert_array_equal(frames[1], EMPTY)

    def test_mismatched_number_of_filaments(self):
        cases = {
            "z_trajectory": dict(r_traj=[[0.1], [0.12]], z_traj=[[0.1]]),
            "perturbation_fwhm": dict(r_traj=[[0.1]], z_traj=[[0.1]],
                                      fwhm=[[0.02], [0.02]]),
            "perturbation_ampl": dict(r_traj=[[0.1]], z_traj=[[0.1]],
                                      ampl=[[1.0], [1.0]]),
        }
        for name, kwargs in cases.items():
            with self.subTest(argument=name):
                with self.assertRaisesRegex(ValueError, name):
                    gaussian(**kwargs)


# ======================================================================
# perturbation_sine
# ======================================================================

# coarse background, uniform in Z
R_S, Z_S, BACKGROUND = create_2d_profile(R1=0.2, Z1=0.3, dR=0.002, dZ=0.002)
CENTRE = 0.1013                                  # between grid points [m]
WAVELENGTH, SPEED, AMPLITUDE = 0.1, 2e3, 0.0117  # [m], [m/s], [m]


def sine(centre=CENTRE, dt=5e-6, nt=4, field=BACKGROUND, **kwargs):
    """perturbation_sine on the coarse grid, returning the frames."""
    return perturbation_sine(R_S, Z_S, field, dt, nt, centre, SPEED,
                             WAVELENGTH, AMPLITUDE, **kwargs)["field"]


class SineTest(unittest.TestCase):

    def test_flattening_between_centre_line_and_wave(self):
        dt, nt = 5e-6, 4
        frames = sine(dt=dt, nt=nt)
        k = 2 * np.pi / WAVELENGTH
        for t_i in range(nt):
            displaced = CENTRE + AMPLITUDE * np.cos(k * Z_S - k * SPEED
                                                    * t_i * dt)
            lo, hi = np.minimum(displaced, CENTRE), np.maximum(displaced,
                                                               CENTRE)
            band = (R_S[:, None] > lo) & (R_S[:, None] < hi)    # (R, Z)
            centre_density = np.array([np.interp(CENTRE, R_S,
                                                 BACKGROUND[:, j])
                                       for j in range(Z_S.size)])
            with self.subTest(frame=t_i):
                self.assertTrue(band.any())
                np.testing.assert_array_equal(frames[t_i][~band],
                                              BACKGROUND[~band])
                np.testing.assert_allclose(
                    frames[t_i][band],
                    np.broadcast_to(centre_density, band.shape)[band])

    def test_background_not_modified(self):
        original = BACKGROUND.copy()
        sine()
        np.testing.assert_array_equal(BACKGROUND, original)

    def test_scalar_centre_line_equals_constant_array(self):
        np.testing.assert_array_equal(
            sine(centre=CENTRE), sine(centre=np.full(Z_S.size, CENTRE)))

    def test_curved_centre_line(self):
        centre = 0.1 + 0.5 * (Z_S - 0.15)**2
        frames = sine(centre=centre, nt=1)
        changed = frames[0] != BACKGROUND
        # changes only near the curved centre line
        rows = np.where(changed.any(axis=1))[0]
        self.assertLessEqual(R_S[rows].max(), centre.max() + AMPLITUDE)
        self.assertGreaterEqual(R_S[rows].min(), centre.min() - AMPLITUDE)

    def test_wave_travels_towards_positive_z(self):
        # after one time step the pattern has moved by speed * dt along Z
        shift = 5                                # grid cells
        dt = shift * (Z_S[1] - Z_S[0]) / SPEED
        frames = sine(dt=dt, nt=2)
        np.testing.assert_array_equal(frames[1][:, shift:],
                                      frames[0][:, :-shift])

    def test_field_repeats_after_one_period(self):
        nt = 8
        dt = WAVELENGTH / SPEED / nt
        frames = sine(dt=dt, nt=nt + 1)
        np.testing.assert_array_equal(frames[nt], frames[0])

    def test_invalid_centre_line(self):
        cases = {"too short": np.full(Z_S.size - 1, CENTRE),
                 "too long": np.full(Z_S.size + 3, CENTRE),
                 "two-dimensional": np.full((2, Z_S.size), CENTRE)}
        for name, centre in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "r_trajectory"):
                    sine(centre=centre)


# ======================================================================
# field layout (regression #31)
# ======================================================================

class FieldLayoutTest(unittest.TestCase):
    """make_regular_density_field returns (NZ, NR); both generators must
    accept it as well as (NR, NZ)."""

    def test_gaussian_accepts_both_layouts(self):
        _, _, background = create_2d_profile(R1=0.2, Z1=0.3, dR=0.001,
                                             dZ=0.002)
        np.testing.assert_array_equal(
            gaussian([[0.1]], [[0.15]], field=background.T),
            gaussian([[0.1]], [[0.15]], field=background))

    def test_sine_accepts_both_layouts(self):
        np.testing.assert_array_equal(sine(field=BACKGROUND.T), sine())

    def test_mismatching_field_shape(self):
        with self.subTest(generator="gaussian"):
            with self.assertRaisesRegex(ValueError, "field shape"):
                gaussian([[0.1]], [[0.15]], field=EMPTY[:-1])
        with self.subTest(generator="sine"):
            with self.assertRaisesRegex(ValueError, "field shape"):
                sine(field=BACKGROUND[:, :-1])


# ======================================================================
# HDF5 output and import_density_field
# ======================================================================

class FileOutputTest(TemporaryDirectoryTestCase):

    def test_gaussian_round_trip(self):
        path = os.path.join(self.tmp, "gaussian.h5")
        data = perturbation_gaussian(R, Z, EMPTY, 1e-6, [[0.05, 0.1, 0.15]],
                                     [[0.1, 0.15, 0.2]], 0.02, 1.0,
                                     file=path)
        loaded = import_density_field(path)
        for key in ("R", "Z", "t", "field"):
            with self.subTest(key=key):
                np.testing.assert_array_equal(loaded[key], data[key])

    def test_sine_round_trip(self):
        path = os.path.join(self.tmp, "sine.h5")
        data = perturbation_sine(R_S, Z_S, BACKGROUND, 5e-6, 3, CENTRE,
                                 SPEED, WAVELENGTH, AMPLITUDE, file=path)
        loaded = import_density_field(path)
        for key in ("R", "Z", "t", "field"):
            with self.subTest(key=key):
                np.testing.assert_array_equal(loaded[key], data[key])

    def test_return_array_false_still_writes_file(self):
        cases = {
            "gaussian": lambda path: perturbation_gaussian(
                R, Z, EMPTY, 1e-6, [[0.1, 0.12]], [[0.15, 0.15]], 0.02, 1.0,
                file=path, return_array=False),
            "sine": lambda path: perturbation_sine(
                R_S, Z_S, BACKGROUND, 5e-6, 2, CENTRE, SPEED, WAVELENGTH,
                AMPLITUDE, file=path, return_array=False),
        }
        for name, run in cases.items():
            with self.subTest(generator=name):
                path = os.path.join(self.tmp, f"{name}.h5")
                self.assertIsNone(run(path))
                self.assertEqual(import_density_field(path)["field"].shape[0],
                                 2)

    def test_frames_read_in_numeric_order(self):
        # alphabetically 'field_100000' sorts before 'field_99999'
        path = os.path.join(self.tmp, "many_frames.h5")
        numbers = (99998, 99999, 100000, 100001)
        with h5py.File(path, "w") as f:
            f["R"], f["Z"], f["t"] = np.arange(2.0), np.arange(3.0), \
                np.arange(4.0)
            for n in numbers:
                f["field_" + str(n).zfill(5)] = np.full((2, 3), float(n))
        field = import_density_field(path)["field"]
        np.testing.assert_array_equal(field[:, 0, 0], numbers)

    def test_missing_axis_dataset(self):
        for missing in ("R", "Z", "t"):
            with self.subTest(missing=missing):
                path = os.path.join(self.tmp, f"no_{missing}.h5")
                with h5py.File(path, "w") as f:
                    for name in {"R", "Z", "t"} - {missing}:
                        f[name] = np.arange(3.0)
                    f["field_00000"] = np.zeros((3, 3))
                with self.assertRaises(KeyError):
                    import_density_field(path)


class FileClosingTest(TemporaryDirectoryTestCase):
    """Regression #35: the HDF5 file must be closed when a run fails."""

    def assertFileClosed(self, path):
        # HDF5 refuses to re-create a file that is still open
        with h5py.File(path, "w"):
            pass

    def test_file_closed_after_error_during_run(self):
        cases = {
            # a string in the per-step FWHM list passes the input checks
            # and fails inside the frame loop, with the file open
            "gaussian": lambda path: perturbation_gaussian(
                R, Z, EMPTY, 1e-6, [[0.1]], [[0.15]], [["0.02"]], 1.0,
                file=path),
            "sine": lambda path: perturbation_sine(
                R_S, Z_S, BACKGROUND, 5e-6, 2, CENTRE, SPEED, WAVELENGTH,
                "0.01", file=path),
        }
        for name, run in cases.items():
            with self.subTest(generator=name):
                path = os.path.join(self.tmp, f"{name}.h5")
                try:
                    run(path)
                except TypeError as error:
                    kept = error             # keep the traceback alive
                else:
                    self.fail("expected a TypeError during the run")
                self.assertTrue(os.path.exists(path))
                self.assertFileClosed(path)
                del kept

    def test_no_file_when_input_checks_fail(self):
        cases = {
            "gaussian": lambda path: perturbation_gaussian(
                R, Z, EMPTY, 1e-6, [[0.1], [0.12]], [[0.15]], 0.02, 1.0,
                file=path),
            "sine": lambda path: perturbation_sine(
                R_S, Z_S, BACKGROUND, 5e-6, 2, np.full(3, CENTRE), SPEED,
                WAVELENGTH, AMPLITUDE, file=path),
        }
        for name, run in cases.items():
            with self.subTest(generator=name):
                path = os.path.join(self.tmp, f"{name}.h5")
                with self.assertRaises(ValueError):
                    run(path)
                self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
