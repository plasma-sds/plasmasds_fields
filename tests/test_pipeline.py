"""
Tests that the public test datasets (see tests/helpers.py) are downloaded
and exist locally. The tests are skipped if the datasets cannot be
downloaded.
"""
import os
import tempfile
import unittest

from plasmasds_utility import DataClient, PlasmasdsUtilityError

from tests.helpers import DATA, get_test_data, TEST_POINCARE, TEST_PROFILE

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


if __name__ == "__main__":
    unittest.main()
