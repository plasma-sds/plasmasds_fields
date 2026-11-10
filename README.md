# Plasma SDS - Fields
The plasmasds_fields package is envisioned as a support package that generates and renders plasma field and magnetic geometry inputs for various synthetic diagnostics featured in the Plasma Synthetic Diagnostics Suite. The package also handles complex output fields and synchronises them to the original input fields.

## Running the tests
The tests use Python's built-in `unittest` framework. The code lives in `src/plasmasds_fields/`; until the package can be installed with pip, put `src` on the import path and run the tests from the repository root:

```bash
PYTHONPATH=src python -m unittest -v
```

In PowerShell, set the path first with `$env:PYTHONPATH = "src"`, then run `python -m unittest -v`.

Most tests use small synthetic inputs. A few use two public test datasets, which are downloaded from the plasma-sds data server with `plasmasds_utility` on first use and cached locally; without network access these tests are skipped. The same tests run on GitHub Actions for every push and pull request.
