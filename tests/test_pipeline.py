"""
Regression test of the full density-map pipeline on the public test
datasets (see tests/helpers.py):

load_w7x_flux_surfaces -> filter_surfaces_by_range -> label_surfaces
-> ProfileInterpolator1D -> add_density -> extract_points
-> make_regular_density_field

The reference values were computed with the code at the time this test was
written; a change in any of them means the pipeline output has changed.
The tests are skipped if the datasets cannot be downloaded.
"""
import os
import tempfile
import unittest
import warnings

import numpy as np
from plasmasds_utility import DataClient, PlasmasdsUtilityError
from scipy.spatial import cKDTree

from tests.helpers import DATA, get_test_data, TEST_POINCARE, TEST_PROFILE
from tools.interpolate import ProfileInterpolator1D
from w7x.fields import add_density, extract_points, make_regular_density_field
from w7x.poincare import (load_w7x_flux_surfaces, filter_surfaces_by_range,
                          label_surfaces)

RTOL = 1e-6     # floating-point results may differ slightly across platforms


def run_pipeline(surfaces, profile, r_range, z_range):
    """Run the pipeline as in the W7-X ABES notebook; return its results."""
    flt = filter_surfaces_by_range(surfaces, r_range=r_range, z_range=z_range)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labelled = label_surfaces(flt)
    density_function = ProfileInterpolator1D(profile[0], profile[1])
    density_function.correct_datapoints(
        position_value=[0, 1, 2, 7, 9],
        profile_value=[1.5e15, 1.5e15, 1.5e15, 2.6e17, 8.4e17],
        by_index=True)
    with_density = add_density(labelled, density_function)
    points = extract_points(with_density)
    grid = make_regular_density_field(points["r"], points["z"],
                                      points["density"])
    return dict(labelled=labelled, label_warnings=caught,
                with_density=with_density, points=points, grid=grid)


DATASET_SIZES = {TEST_POINCARE: 10848116, TEST_PROFILE: 2853}   # bytes


class DownloadTest(unittest.TestCase):
    """The requested datasets are downloaded and exist locally."""

    def test_requested_files_exist(self):
        for key, size in DATASET_SIZES.items():
            with self.subTest(key=key):
                path = get_test_data(key)
                self.assertTrue(os.path.isfile(path))
                self.assertEqual(os.path.getsize(path), size)
                self.assertEqual(path, DATA.local_path(key, private=False))

    def test_download_into_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = DataClient("plasmasds_fields", working_dir=tmp)
            target = client.local_path(TEST_PROFILE, private=False)
            self.assertFalse(target.exists())
            try:
                path = client.get(TEST_PROFILE, private=False)
            except PlasmasdsUtilityError as error:
                self.skipTest(f"test dataset not available: {error}")
            self.assertEqual(path, target)
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, DATASET_SIZES[TEST_PROFILE])


class PipelineTestCase(unittest.TestCase):
    """Loads the datasets and runs the pipeline once per test class."""

    r_range = z_range = None

    @classmethod
    def setUpClass(cls):
        cls.surfaces = load_w7x_flux_surfaces(get_test_data(TEST_POINCARE))
        cls.profile = np.loadtxt(get_test_data(TEST_PROFILE))
        cls.result = run_pipeline(cls.surfaces, cls.profile, cls.r_range,
                                  cls.z_range)

    def assertRelClose(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=RTOL)


class DatasetTest(PipelineTestCase):
    """The test datasets themselves."""

    r_range, z_range = [6.1, 6.3], [-0.5, 0.5]

    def test_poincare_dataset(self):
        self.assertEqual(len(self.surfaces), 300)
        self.assertEqual(sum(s.n for s in self.surfaces), 103428)
        self.assertEqual({s.phi0 for s in self.surfaces}, {1.256637})

    def test_profile_dataset(self):
        self.assertEqual(self.profile.shape, (3, 38))
        self.assertRelClose(self.profile[0].min(), 6.08661147477652)
        self.assertRelClose(self.profile[0].max(), 6.305570062814554)

    def test_loaded_surfaces_not_modified_by_pipeline(self):
        # the filter returns copies; the loaded surfaces keep all points
        self.assertEqual(sum(s.n for s in self.surfaces), 103428)

    def test_labelling_outside_window_warns(self):
        surfaces = load_w7x_flux_surfaces(get_test_data(TEST_POINCARE))
        with self.assertWarnsRegex(UserWarning, "R < 6.1 m"):
            label_surfaces(surfaces)


class ValidatedWindowTest(PipelineTestCase):
    """The validated window: R 6.1-6.3 m, |Z| <= 0.5 m."""

    r_range, z_range = [6.1, 6.3], [-0.5, 0.5]

    def test_window_selection(self):
        labelled = self.result["labelled"]
        self.assertEqual(len(labelled), 107)
        self.assertEqual(sum(s.n for s in labelled), 10621)

    def test_labelling(self):
        labelled = self.result["labelled"]
        self.assertEqual(self.result["label_warnings"], [])
        self.assertEqual(sum(s.errors[1] is not None for s in labelled), 23)
        point_types = np.concatenate([s.points.point_type for s in labelled])
        self.assertEqual(np.count_nonzero(point_types == 1), 3413)
        self.assertEqual(np.count_nonzero(point_types == 2), 3666)

    def test_fit_errors(self):
        # regression #27: island side errors were computed over all points
        # of the surface instead of the points of that side
        labelled = self.result["labelled"]
        side = [e for s in labelled for e in s.errors[1:] if e is not None]
        regular = [s.errors[0] for s in labelled if s.errors[0] is not None]
        self.assertEqual((len(side), len(regular)), (46, 84))
        self.assertRelClose(max(side), 2.3854560050e-3)
        self.assertRelClose(max(regular), 9.8220322168e-3)

    def test_single_point_fragments_keep_their_radius(self):
        # regression: one-point surfaces got a fitted radius of about 2.09 m
        single = [s for s in self.result["labelled"] if s.n == 1]
        self.assertEqual(len(single), 2)
        for surf in single:
            self.assertAlmostEqual(surf.coeffs[0, -1], surf.points.r[0])

    def test_density_allocation(self):
        self.assertEqual(len(self.result["with_density"]), 107)
        points = self.result["points"]
        self.assertEqual(points["r"].size, 10621)
        self.assertRelClose(points["density"].min(), 1.5e15)
        self.assertRelClose(points["density"].max(), 4.1182202219e19)
        self.assertRelClose(points["density"].sum(), 1.5040379762e23)
        self.assertRelClose(points["surface_radius"].min(), 6.1003967350)
        self.assertRelClose(points["surface_radius"].max(), 6.3839045116)

    def test_regular_grid(self):
        R, Z, grid = self.result["grid"]
        self.assertEqual((R.size, Z.size, grid.shape),
                         (401, 1999, (1999, 401)))
        self.assertRelClose([R[0], R[-1]], [6.100002, 6.300002])
        self.assertRelClose(grid.min(), 1e16)
        self.assertRelClose(grid.max(), 4.1182202219e19)
        self.assertRelClose(grid.mean(), 1.3800924642e19)


class NotebookWindowTest(PipelineTestCase):
    """The window of the W7-X ABES notebook: R 6.10-6.28 m, |Z| <= 0.2 m."""

    r_range, z_range = [6.10, 6.28], [-0.2, 0.2]

    def test_window_and_labelling(self):
        labelled = self.result["labelled"]
        self.assertEqual(len(labelled), 99)
        self.assertEqual(sum(s.n for s in labelled), 6708)
        self.assertEqual(sum(s.errors[1] is not None for s in labelled), 23)

    def test_density_allocation(self):
        points = self.result["points"]
        self.assertEqual(points["r"].size, 6708)
        self.assertRelClose(points["density"].min(), 2.6547022344e16)
        self.assertRelClose(points["density"].max(), 4.1182202219e19)
        self.assertRelClose(points["density"].sum(), 9.3071841971e22)

    def test_regular_grid(self):
        R, Z, grid = self.result["grid"]
        self.assertEqual(grid.shape, (801, 360))
        self.assertRelClose(grid.min(), 2.6547022344e16)
        self.assertRelClose(grid.mean(), 2.1582928924e19)

    def test_island_tip_point_on_inner_side(self):
        # regression: this island tip point was put on the outer side and
        # got 2.6e17 instead of about 2.4e19, a dark spot in the density map
        points = self.result["points"]
        k = np.argmin((points["r"] - 6.1937)**2 + (points["z"] - 0.1855)**2)
        self.assertEqual(points["point_type"][k], 2)
        self.assertRelClose(points["density"][k], 2.362198e19)

    def test_no_isolated_density_outliers(self):
        # no point differs by more than 10x from its 8 nearest neighbours
        points = self.result["points"]
        xy = np.column_stack([points["r"], points["z"]])
        _, neighbours = cKDTree(xy).query(xy, k=9)
        local = np.median(points["density"][neighbours[:, 1:]], axis=1)
        ratio = points["density"] / local
        self.assertEqual(np.count_nonzero((ratio < 0.1) | (ratio > 10)), 0)


if __name__ == "__main__":
    unittest.main()
