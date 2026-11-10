# Plasma SDS - Fields
The plasmasds_fields package is envisioned as a support package that generates and renders plasma field and magnetic geometry inputs for various synthetic diagnostics featured in the Plasma Synthetic Diagnostics Suite. The package also handles complex output fields and synchronises them to the original input fields.

> **Status:** early development (alpha), version 0.1.0; the interface may still change between releases.
> It runs on Linux and Windows with Python 3.11 or newer.

## Installation
Once the first release is on PyPI:

```bash
pip install plasmasds_fields
```

Until then, install it from GitHub (this needs `git`):

```bash
pip install "plasmasds_fields @ git+https://github.com/plasma-sds/plasmasds_fields"
```

pip also installs the dependencies: numpy, scipy, matplotlib, h5py, Pillow and `plasmasds_utility`.

For development, clone the repository and install it in editable mode, so changes to the code take effect without reinstalling:

```bash
git clone https://github.com/plasma-sds/plasmasds_fields
cd plasmasds_fields
pip install -e .
```

## Package structure

| Module | Contents |
|---|---|
| `plasmasds_fields.w7x.poincare` | W7-X Poincaré flux surfaces: loading field-line tracer results, selecting surfaces, labelling island and non-island points, plotting |
| `plasmasds_fields.w7x.fields` | Density allocation on the flux surfaces, and conversion to scattered and regular 2-D density fields |
| `plasmasds_fields.tools.interpolate` | `ProfileInterpolator1D`: a 1-D profile interpolator with editable datapoints |
| `plasmasds_fields.tools.plot` | Contour plots, slices and GIF animations of 2-D and time-dependent (R, Z) fields |
| `plasmasds_fields.models.hesel` | Reading and extending HESEL simulation output |
| `plasmasds_fields.models.motion` | Time-dependent perturbations of 2-D fields (Gaussian filaments, sine waves), with HDF5 output |
| `plasmasds_fields.models.profilegen` | Generic 2-D background density profiles |

## Usage
A 2-D density map of the W7-X edge, from Poincaré flux surfaces and a radial density profile. The example uses the package's public example data, which `plasmasds_utility` downloads on first use and caches locally:

```python
import numpy as np
from plasmasds_utility import DataClient

from plasmasds_fields.w7x.poincare import (load_w7x_flux_surfaces,
                                           filter_surfaces_by_range,
                                           label_surfaces)
from plasmasds_fields.w7x.fields import (add_density, extract_points,
                                         make_regular_density_field)
from plasmasds_fields.tools.interpolate import ProfileInterpolator1D
from plasmasds_fields.tools.plot import contour_field

# public example data: W7-X Poincaré surfaces and a radial density profile
data = DataClient("plasmasds_fields")
surfaces = load_w7x_flux_surfaces(
    data.get("test_dataset/test_poincare.xml", private=False))
profile = np.loadtxt(data.get("test_dataset/test_profile.txt", private=False))

# select the edge region and label the island and non-island points
surfaces = filter_surfaces_by_range(surfaces, r_range=[6.1, 6.3],
                                    z_range=[-0.2, 0.2])
surfaces = label_surfaces(surfaces)

# assign each surface the density of the profile at its radius
density_function = ProfileInterpolator1D(profile[0], profile[1])
surfaces = add_density(surfaces, density_function)

# interpolate onto a regular (R, Z) grid and plot the 2-D density map
points = extract_points(surfaces)
R, Z, density = make_regular_density_field(points["r"], points["z"],
                                           points["density"])
contour_field(R, Z, density, levels=100)
```

## Development
Work happens on feature branches, which are merged into `development` through pull requests; `master` holds released code. Planned work is tracked as [GitHub issues](https://github.com/plasma-sds/plasmasds_fields/issues).

The tests use Python's built-in `unittest` framework and run against the installed package. Install it once in editable mode, so changes to the code take effect without reinstalling, then run the tests from the repository root:

```bash
pip install -e .
python -m unittest -v
```

Most tests use small synthetic inputs. A few use two public test datasets, which are downloaded from the plasma-sds data server with `plasmasds_utility` on first use and cached locally; without network access these tests are skipped. The same tests run on GitHub Actions for every push and pull request.
