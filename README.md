# Plasma SDS - Fields
The plasmasds_fields package is envisioned as a support package that generates and renders plasma field and magnetic geometry inputs for various synthetic diagnostics featured in the Plasma Synthetic Diagnostics Suite. The package also handles complex output fields and synchronises them to the original input fields.

> **Status:** early development (alpha), version 0.1.0; the interface may still change between releases.
> It runs on Linux and Windows with Python 3.11 or newer.

## Running the tests
The tests use Python's built-in `unittest` framework and run against the installed package. Install it once in editable mode, so changes to the code take effect without reinstalling, then run the tests from the repository root:

```bash
pip install -e .
python -m unittest -v
```

Most tests use small synthetic inputs. A few use two public test datasets, which are downloaded from the plasma-sds data server with `plasmasds_utility` on first use and cached locally; without network access these tests are skipped. The same tests run on GitHub Actions for every push and pull request.
