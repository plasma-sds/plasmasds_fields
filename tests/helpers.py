"""
Shared helpers for the unit tests: a small synthetic HESEL output file, so
the tests run without the real HESEL data.
"""
import h5py
import numpy as np


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
