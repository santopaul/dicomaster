# Contributing to Dicomaster

Thanks for your interest in improving Dicomaster! Contributions are welcome via pull requests.

## Quick start
- Fork and create a feature branch.
- Create a virtualenv and install dev deps:
  - `pip install -e .[full]` (or `pip install -e .[dev]`)
- Run checks locally:
  - `pytest -q`
  - `ruff check .`

## Coding guidelines
- Keep CLI UX simple; default single-file run shows the STAT summary.
- Use clear, color-safe output (fallbacks for missing deps are required).
- Prefer small, focused PRs with a clear description.

## Good first issues
- Look for issues labeled `good first issue` and `enhancement`.
- Examples: Docker publish workflow, CodeQL, Dependabot, preview artifacts.

## Commit message style
- Conventional style is appreciated: `fix: ...`, `feat: ...`, `chore: ...`.

## License
- By contributing, you agree your code will be released under the MIT license.