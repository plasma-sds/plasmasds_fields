"""
Plasma field and magnetic geometry inputs for the plasma-sds synthetic
diagnostics.

Subpackages:

* ``plasmasds_fields.w7x``: W7-X Poincaré flux surfaces and density
  allocation on them.
* ``plasmasds_fields.tools``: profile interpolation and plotting of fields.
* ``plasmasds_fields.models``: HESEL output, background profiles and
  time-dependent perturbations.

Import what you need from the subpackages, for example
``from plasmasds_fields.w7x.poincare import load_w7x_flux_surfaces``.
"""

__version__ = "0.1.0"
