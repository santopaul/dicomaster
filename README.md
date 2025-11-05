# Dicomaster — Secure DICOM anonymizer & batch processor

[![PyPI version](https://badge.fury.io/py/dicomaster.svg)](https://badge.fury.io/py/dicomaster) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://github.com/santopaul/dicom-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/santopaul/dicom-tool/actions)

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
.\.dicomaster\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[full]
```

Or for minimal/core features:

```powershell
pip install -e .
```

## Quick examples

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
- `pyproject.toml` (packaging metadata)
- `requirements.txt` (developer requirements)
- `README.md`, `CHANGELOG.md`, `RELEASE.md`, `LICENSE.txt`
- `tests/` and `.github/workflows/ci.yml`
- `sample_data/` (small demo files) — optional but useful

## Contribution

1. Fork and open a pull request
2. Add tests for new features and run `pytest`
3. Keep changes small and document them in `CHANGELOG.md`

## License

MIT — see `LICENSE.txt`.

---

If you want, I can now:

- update the GitHub Actions CI to install `.[full]` and run lint/tests, or
- convert the project into a proper package layout (`dicomaster/` package) and update imports/tests, or
- prepare a TestPyPI release (build artifacts + guidance for upload).

Tell me which and I'll do it next.
# dicomaster
# Secure DICOM Anonymizer & Batch Processor

[![PyPI version](https://badge.fury.io/py/dicomaster.svg)](https://badge.fury.io/py/dicomaster)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/santopaul/dicom-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/santopaul/dicom-tool/actions)
[![Stars](https://img.shields.io/github/stars/santopaul/dicom-tool?style=social)](https://github.com/santopaul/dicom-tool/stargazers)

Dicomaster is a Python CLI that takes messy, PHI-loaded .dcm files and turns them into research-ready, AI-ready, hospital-ready datasets.

Think of it as:
👉 pydicom, but with a turbocharged CLI
👉 deid, but with stronger anonymization (PBKDF2)
👉 dicomsort, but with actual aggregation + FHIR support

Batch-process 10k+ DICOMs, anonymize securely, stream metadata to CSV/JSON without blowing RAM, and export straight into Kaggle or EHR-ready formats.

Why I Built This

When I first jumped into Kaggle’s RSNA Intracranial Aneurysm Detection challenge, I realized the bottleneck wasn’t the model — it was the DICOMs. Hidden PHI, private tags, slow manual preprocessing, inconsistent schemas.

So I built Dicomaster to:

🛡️ Protect patients — secure pseudonymization with PBKDF2 (100k iterations) or HMAC fallback.

⚡ Speed up research — threaded batch mode + streaming agg (no OOM, even for 300GB).

🧑‍⚕️ Help clinicians — STAT reports, urgency flagging (e.g., “stroke” keywords), thumbnails for a quick glance.

🔗 Bridge worlds — export to FHIR ImagingStudy (HL7) or Kaggle-style CSV schema.

This is my shot at making DICOM data safe, fast, and useful for AI and hospitals.

Quick Start
Install
pip install dicomaster
# or full features
pip install dicomaster[full]

Examples

Single File Metadata + Thumbnail

dicommaster sample_data/mri_1.dcm -o json,thumbnail -v
---

## Why Dicomaster

When prepping DICOMs for ML or research you need speed, traceability, and patient safety. Dicomaster was born from those needs (RSNA/Kaggle experience). It helps:

- Protect patients with strong pseudonymization
- Speed up large-scale preprocessing with threaded, streaming aggregation
- Produce clinician-friendly reports and basic urgency flags
- Export to FHIR ImagingStudy for EHR integration

---

## Features (high level)

- Extraction: STAT (critical fields) + full technical metadata + private-tag listing
- Clinician tools: urgency flags (aneurysm/hemorrhage/stroke keywords), thumbnails and visual reports
- Anonymization: `--anonymize` (pseudonymize with PBKDF2 or remove), `--anonymize-map` for auditable mappings
- Batch: recursive scanning of `.dcm/.ima/.img`, CPU-adaptive threading, `--max-depth`
- Outputs: `json`, `csv`, `html`, `image`, `thumbnail`, `report` (metadata-as-image), `fhir`, `agg-csv`, `agg-json`
- Performance: streaming aggregation (no full in-memory accumulation), optional `pandas` for convenience
- Safety: `--dry-run`, `--no-overwrite`, private tag removal, PHI detection warnings
- UX: REPL mode, `--check-deps` to list optional features, quiet/verbose logging

---


## Quick install

Recommended: create a venv first.

```bash
python -m venv .dicomaster
.dicomaster\Scripts\Activate.ps1   # on Windows (PowerShell)
pip install -U pip
pip install -e .[full]   # installs package in editable mode with optional extras
```

If you only need core features:

```bash
pip install -e .
```

Note: Optional extras provide thumbnails (`Pillow`), PBKDF2 pseudonyms (`cryptography`), streaming convenience (`pandas`), and progress bars (`tqdm`). Use `dicommaster --check-deps` to see what's available.

---

## Quick examples

Single file: Metadata (JSON) + thumbnail

```bash
dicommaster sample_data/mri_1.dcm -o json,thumbnail -v
```

Batch anonymize and stream to combined CSV (recommended for research)

```bash
dicommaster --batch /data/tcia -o agg-csv --anonymize --anonymize-salt mysecret --threads 8
```

Generate FHIR ImagingStudy outputs

```bash
dicommaster /studies -o fhir --batch --remove-private-tags
```

Interactive REPL

```bash
# interactive: run the installed CLI or the script directly during development
dicommaster
# enter file paths at the prompt
```

Urgency detection (flags in report image)

```bash
dicommaster brain.dcm -o report --check-urgent
```

Use `dicommaster --help` for a complete list of options.

---

## Command reference (high level)

- `-b/--batch` : treat path as a directory and scan recursively
- `-o/--output` : one or more outputs (comma-separated or repeated), e.g. `-o json,thumbnail`
- `--output-dir` : destination directory for generated files
- `--anonymize` : enable anonymization
- `--anonymize-mode` : `pseudonymize` (default) or `remove`
- `--anonymize-map` : path to write JSON mapping (original -> pseudonym)
- `--anonymize-salt` : provide salt for reproducible pseudonyms (recommended)
- `--dry-run` : print intended actions without writing files
- `-q/--quiet` and `-v/--verbose` : control logging

See `--help` for the complete set of flags.

---

## Security notes & recommended practice

- Prefer specifying `--anonymize-salt` or `--anonymize-map` for reproducible pseudonyms across runs.
- If `cryptography` is installed, PBKDF2HMAC is used (recommended). If missing, the tool falls back to HMAC and logs a warning — install `cryptography` for production.
- Generated salts are written to the anonymize map file but are not printed verbatim to logs.
- Always verify output and mapping files before publishing datasets. Anonymization configuration, dataset pairing (images vs metadata), and private tag removal are your responsibility.

---

## Code review summary (quick)

I reviewed `dicom_tool.py`. High-level findings:

- Strengths: clear CLI, robust optional-dependency handling, streaming aggregation implementation, PBKDF2 fallback, many useful outputs (HTML, FHIR, images).
- Minor bugs/risks to fix before production release:
  - Some functions use inconsistent metadata keys (e.g., `urgent` vs `Urgent`, `phi_flags` vs `PHI_Flags`) which can hide data in report/image generation.
  - Shared `run_mapping` is mutated from worker threads without synchronization — consider a `threading.Lock` or thread-safe queue to avoid race conditions.
  - HTML reports reference thumbnail file name but don't embed or ensure relative path, which can make the HTML portable only when run from the output dir.
  - Many optional imports cause static analysis warnings; these are expected but mention clearly in README and use `--check-deps`.

Overall: the tool is close to release-ready with a few small fixes (key consistency, thread-safety, small CI/tests). I can open PRs to address them if you want.

---

## Recommended next steps before publishing

1. Add a minimal test suite (pytest) covering: single-file metadata, anonymization mapping output, stream aggregation of a few files.
2. Add packaging config (`pyproject.toml`) and ensure `extras_require` for `full` features.
3. Add GitHub Actions: lint (flake8/ruff), tests, and a build-and-publish workflow for PyPI.
4. Fix thread-safety around `run_mapping` updates and metadata key mismatches in report/image functions.
5. Add a short CHANGELOG and release notes.

If you want, I can implement items 1–3 in this repo.

---

## Contributing

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
