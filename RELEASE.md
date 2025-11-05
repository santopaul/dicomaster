# Release checklist

This file documents the minimal steps to create a release and publish to TestPyPI/PyPI.

1. Update `CHANGELOG.md` with changes for the new version.
2. Bump version in `pyproject.toml`.
3. Update `README.md` `version` badges if present.
4. Build distributions:

```powershell
python -m pip install -U build twine
python -m build
```

5. Validate distribution:

```powershell
python -m twine check dist/*
```

6. Upload to TestPyPI (dry run):

```powershell
python -m twine upload --repository testpypi dist/*
```

7. After testing, upload to PyPI (use your real credentials):

```powershell
python -m twine upload dist/*
```

8. Tag release and push tag:

```powershell
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

9. Create a GitHub release and attach built wheels if needed.

Keep sensitive keys out of CI; use GitHub Actions secrets for PyPI API tokens.
