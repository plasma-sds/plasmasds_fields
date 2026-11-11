# Changelog

All notable changes to plasmasds_fields are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the version numbers are explained in the README, under "Releases".

## Unreleased

## 0.1.0

First release as a package that can be installed with pip.

### Added
- W7-X Poincaré flux surfaces: loading field-line tracer results, selecting surfaces, labelling island and non-island points (`label_surfaces`), assigning densities from a radial profile (`add_density`) and converting them to scattered and regular 2-D density fields.
- 1-D profile interpolation with editable datapoints (`ProfileInterpolator1D`).
- Contour plots, slices and GIF animations of 2-D and time-dependent (R, Z) fields.
- Reading and extending HESEL simulation output, generic 2-D background profiles, and Gaussian-filament and sine-wave perturbations with HDF5 output.
- Unit tests, run by GitHub Actions on Linux and Windows with Python 3.11 to 3.13; some use public test datasets fetched with `plasmasds_utility`.

### Fixed
Fixes made during the development leading to this release:
- Island density allocation: side fit errors from the points of each side, island tip points assigned to the nearest side, and reduced-order fits for surfaces with too few points (#27).
- Input checks, warnings and error messages of the W7-X functions (#29).
- Gaussian filaments: the requested FWHM, and no filaments missing near grid edges or on grids with different R and Z spacing (#32).
- The perturbation generators accept the (NZ, NR) field layout of `make_regular_density_field` (#31).
- HDF5 output files are closed also when a run fails (#35).
