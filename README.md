# Dicomaster — Secure DICOM anonymizer & batch processor

[![PyPI version](https://badge.fury.io/py/dicomaster.svg)](https://badge.fury.io/py/dicomaster) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://github.com/santopaul/dicomaster/actions/workflows/ci.yml/badge.svg)](https://github.com/santopaul/dicomaster/actions)

Dicomaster is a compact, production-minded CLI tool for extracting metadata from DICOM files, securely anonymizing PHI, and producing researcher- and clinician-friendly outputs (JSON, CSV, FHIR ImagingStudy, thumbnails, HTML reports and more).

This repository contains the CLI and library logic needed to batch-process DICOM datasets for ML/AI research or clinical data pipelines while keeping an auditable anonymization map.

## Highlights

- Secure pseudonymization (PBKDF2 when `cryptography` is available; HMAC fallback otherwise)
- Streaming aggregation for large datasets (low memory footprint)
- Threaded batch processing with configurable worker count
- Multiple output formats: `json`, `csv`, `html`, `image`, `thumbnail`, `report`, `fhir`, `agg-csv`, `agg-json`
- Optional extras for thumbnails, progress bars and faster aggregation (`Pillow`, `tqdm`, `pandas`)
- Auditable anonymization maps (JSON) and reproducible pseudonyms via salt

## Is this ready for GitHub release?

Short answer: Yes — core functionality, tests, and packaging are in place. I ran the test-suite and packaging build locally. Before a public PyPI release consider these small polish items:

- Finalize `pyproject.toml` metadata (long description, homepage/URLs, author contact)
- Decide on final module layout (top-level module vs package directory). Current packaging uses `dicomaster.py` as the shipped module.
- Add a few more integration tests or fixtures (optional) to improve coverage
- Optionally convert the repo to a package layout (`dicomaster/` package) for future extensibility

If you want, I can make all of the above and prepare a TestPyPI release workflow.

## Quick install

Recommended: create an environment and install editable with extras for full features.

Windows (PowerShell):

```powershell
python -m venv .dicomaster
\.\.dicomaster\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[full]
```

Or for minimal/core features:

```powershell
pip install -e .
```

## Quick examples

Single file — colorful STAT (default):

```powershell
dicomaster .\path\to\file.dcm
```

Single file metadata + thumbnail:

```powershell
dicomaster sample_data/mri_1.dcm -o json,thumbnail -v
```

Batch anonymize and stream combined CSV:

```powershell
dicomaster --batch C:\data\dicoms -o agg-csv --anonymize --anonymize-salt mysecret --threads 8
```

Generate FHIR ImagingStudy outputs:

```powershell
dicomaster /studies -o fhir --batch --remove-private-tags
```

Use `dicomaster --help` for the full set of options and flags.

## Changelog

See `CHANGELOG.md` for release highlights.

## Developer notes

- Tests: run `pytest -q`. The repo includes unit and small integration tests that create a minimal DICOM at runtime.
- Build: `python -m build` (I validated the sdist and wheel locally).
- Editable install with extras: `pip install -e .[full]` — installs `pydicom`, `Pillow`, `pandas`, `cryptography`, etc.

## Security & best practices

- Prefer supplying `--anonymize-salt` for reproducible pseudonyms across runs. If you omit it, the tool will generate salts and store them in the anonymize map.
- Install `cryptography` to use PBKDF2HMAC (stronger) instead of the HMAC fallback.
- Always validate your anonymized outputs and mapping before publishing datasets.

## Files to include when you push for release

- `dicomaster.py` (module)
<!-- Duplicate lower section removed for clarity -->

1. Fork and open a PR.
2. Add tests for new features and bug fixes.
3. Format code with `black` (project style). Run `python -m pytest` to confirm tests.

---

## License

MIT — see `LICENSE.txt`.

---

## Acknowledgements

Built on `pydicom`, inspired by `deid` and the DICOM community. Created by Santo Paul to make DICOM preprocessing safer and faster for research and clinical workflows.

---

If you'd like, I will (choose one):

- add basic pytest tests and a small CI workflow, or
- patch the thread-safety and metadata-key issues I found, or
- prepare a `pyproject.toml` + `setup.cfg` that declares `extras_require` and a `console_scripts` entry point.

Tell me which and I'll implement it next.
