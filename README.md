# Plasma SDS - Fields
The plasmasds_fields package is envisioned as a support package that generates and renders plasma field and magnetic geometry inputs for various synthetic diagnostics featured in the Plasma Synthetic Diagnostics Suite. The package also handles complex output fields and synchronises them to the original input fields.

## Running the tests
The tests use Python's built-in `unittest` framework. Run them from the repository root:

```bash
python -m unittest -v
```

The tests use small synthetic inputs and run without any measurement or simulation data. Tests on real data run only when the environment variables `PLASMASDS_POINCARE_XML`, `PLASMASDS_DENSITY_PROFILE` or `PLASMASDS_HESEL_FILE` point to the corresponding files; otherwise they are skipped. The same tests run on GitHub Actions for every push and pull request.
