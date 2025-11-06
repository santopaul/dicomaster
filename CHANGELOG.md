# Changelog

## 0.9.3.post1 — 2025-11-05

Docs-only postrelease:
- Fix README image links to use raw.githubusercontent URLs so images render on PyPI.
- No code changes.

## 0.9.3 — 2025-11-05

Patch:
- Ensure colorful STAT headline renders for --detail and --full (not just --minimal/default).
- Keep minimal summary as default for single-file runs; --minimal remains a no-op alias.
- No breaking changes; recommended patch upgrade.

## 0.9.2 — 2025-11-05

Hotfix:
- Fix import-time NameError for `NumPyModule` in tests by defining runtime aliases for protocol types.
- Slightly broaden DICOM dataset typing hints to include `.get` and `file_meta`.

## 0.9.1 — 2025-11-05

Highlights:
- Default single-file run shows a colorful STAT summary (no need for `--minimal`).
- Polished banner/tagline output and PHI warning line.
- Added “Important DICOM Tags” and enriched patient/study fields.
- CI hardened (Ruff, mypy best-effort, pytest+coverage) and non-blocking Codecov.
- Release workflow: tag-based build + publish to PyPI and GitHub Releases.

Breaking changes: none.

Notes:
- Sample `mri_*.dcm` assets are placeholders; use real DICOMs for testing or create a minimal one with `pydicom`.
# Changelog

All notable changes to this project will be documented in this file.

## [0.9.0] - Unreleased
- Initial release candidate: thread-safety fixes, metadata key normalization
- Basic CI (lint + pytest) and packaging scaffold (pyproject.toml)
- Minimal integration tests and README polish

Notes:
- See `RELEASE.md` for steps to publish to TestPyPI/PyPI.
