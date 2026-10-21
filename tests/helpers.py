"""
Shared helpers for the unit tests: the test datasets on the plasma-sds data
server, and a small synthetic HESEL output file, so the tests run without
the real HESEL data.
"""
import unittest

import h5py
import numpy as np
from plasmasds_utility import DataClient, PlasmasdsUtilityError


# ---------------------------------------------------------------- test data
# Public test datasets of plasmasds_fields on the plasma-sds data server,
# downloaded with plasmasds_utility on first use and cached locally.

DATA = DataClient("plasmasds_fields")
TEST_POINCARE = "test_dataset/test_poincare.xml"   # W7-X field-line tracer
TEST_PROFILE = "test_dataset/test_profile.txt"     # density profile


def get_test_data(key):
    """
    Return the local path of a public test dataset, downloading it if needed.

    Skips the calling test (``unittest.SkipTest``) if the file cannot be
    fetched, for example without network access.
    """
    try:
        return DATA.get(key, private=False)
    except PlasmasdsUtilityError as error:
        raise unittest.SkipTest(f"test dataset {key} not available: {error}")


# ---------------------------------------------------------------- HESEL HDF5
# Field values encode their indices, so tests can check which samples were
# selected: value = base + 1e17 * t_index + 1e16 * z_index + 1e15 * r_index.

HESEL_R = np.arange(12) * 0.005             # [m]
HESEL_Z = np.arange(10) * 0.004 - 0.02      # [m]
HESEL_T = np.arange(5) * 1e-6               # [s]
HESEL_FIELDS = {"density": 1e19, "electron_temperature": 2e19,
                "ion_temperature": 3e19}


def hesel_values(base):
    """Expected (time, Z, R) values of a synthetic HESEL field."""
    t, z, r = np.meshgrid(np.arange(HESEL_T.size), np.arange(HESEL_Z.size),
                          np.arange(HESEL_R.size), indexing="ij")
    return base + 1e17 * t + 1e16 * z + 1e15 * r


def write_hesel_file(path):
    """Write a small HDF5 file with the HESEL ``data/xanimation`` layout."""
    with h5py.File(path, "w") as f:
        g = f.create_group("data/xanimation")
        g["xgrid"] = np.tile(HESEL_R, (HESEL_Z.size, 1))          # (Z, R)
        g["ygrid"] = np.tile(HESEL_Z[:, None], (1, HESEL_R.size))  # (Z, R)
        g["time"] = HESEL_T[:, None]                               # (t, 1)
        for name, base in HESEL_FIELDS.items():
            g[name] = hesel_values(base)                           # (t, Z, R)
    return path
