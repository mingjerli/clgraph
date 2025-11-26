# Release Process

This document describes how to publish new releases of `clpipe` to PyPI.

## Overview

The release process is automated using GitHub Actions with PyPI Trusted Publishing. When you create a new GitHub release, the package is automatically built, tested, and published to PyPI.

## Prerequisites

### One-Time Setup: Configure PyPI Trusted Publishing

Before you can publish packages, you need to set up trusted publishing on PyPI:

1. **Create a PyPI account** (if you don't have one):
   - Go to https://pypi.org/account/register/
   - Verify your email address

2. **Add the project to PyPI** (first release only):
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in the details:
     - **PyPI Project Name**: `clpipe`
     - **Owner**: `clpipe`
     - **Repository name**: `clpipe`
     - **Workflow name**: `publish-to-pypi.yml`
     - **Environment name**: `pypi`
   - Click "Add"

3. **Configure GitHub Environment** (recommended for protection):
   - Go to your repository settings → Environments
   - Create an environment named `pypi`
   - Add protection rules (optional but recommended):
     - Required reviewers
     - Deployment branches: Only allow default branch

### For Testing: Configure TestPyPI (Optional)

To test the publishing process without affecting production PyPI:

1. Create an account at https://test.pypi.org/account/register/
2. Set up trusted publishing with these details:
   - **PyPI Project Name**: `clpipe`
   - **Owner**: `clpipe`
   - **Repository name**: `clpipe`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `testpypi`

## Release Checklist

Before creating a release, ensure:

- [ ] All tests pass on the main branch
- [ ] `CHANGELOG.md` is updated with new features, fixes, and breaking changes (if any)
- [ ] Version number in `pyproject.toml` is updated following [Semantic Versioning](https://semver.org/)
  - **MAJOR** version for incompatible API changes
  - **MINOR** version for backwards-compatible new features
  - **PATCH** version for backwards-compatible bug fixes
- [ ] Documentation is up to date
- [ ] All planned features/fixes for the release are merged

## Creating a Release

### 1. Update Version Number

Edit `pyproject.toml` and update the version:

```toml
[project]
name = "clpipe"
version = "0.2.0"  # Update this line
```

### 2. Update Changelog

Create or update `CHANGELOG.md` with the new version and changes:

```markdown
## [0.2.0] - 2024-01-15

### Added
- New feature X
- New feature Y

### Changed
- Improved Z

### Fixed
- Bug fix A
```

### 3. Commit and Push Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

### 4. Create a GitHub Release

#### Option A: Via GitHub Web Interface

1. Go to https://github.com/clpipe/clpipe/releases/new
2. Click "Choose a tag"
3. Type the version number with a `v` prefix (e.g., `v0.2.0`) and click "Create new tag on publish"
4. Set the release title to the version number (e.g., `v0.2.0`)
5. In the description, paste the relevant section from your CHANGELOG
6. Check "Set as the latest release"
7. Click "Publish release"

#### Option B: Via GitHub CLI

```bash
# Create and push a tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Create a release
gh release create v0.2.0 \
  --title "v0.2.0" \
  --notes-file CHANGELOG.md \
  --latest
```

### 5. Monitor the Release

1. Go to the Actions tab: https://github.com/clpipe/clpipe/actions
2. Watch the "Publish to PyPI" workflow
3. The workflow will:
   - Build the package
   - Run package verification
   - Publish to PyPI
   - Upload build artifacts to the GitHub release

### 6. Verify the Release

After the workflow completes:

1. Check PyPI: https://pypi.org/project/clpipe/
2. Verify the new version is listed
3. Test installation:
   ```bash
   pip install --upgrade clpipe
   python -c "import clpipe; print(clpipe.__version__)"
   ```

## Testing Releases (TestPyPI)

To test the release process without publishing to production PyPI:

### Via GitHub Actions UI

1. Go to Actions → "Publish to PyPI"
2. Click "Run workflow"
3. Check "Publish to Test PyPI instead of production PyPI"
4. Click "Run workflow"

### Verify TestPyPI Release

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ clpipe

# Verify
python -c "import clpipe; print(clpipe.__version__)"
```

## Version Numbering Strategy

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **0.x.x**: Initial development (current phase)
  - Breaking changes allowed in minor versions
  - Use for pre-1.0 releases

- **1.0.0**: First stable release
  - Indicates the API is stable
  - Commit to backwards compatibility

- **x.y.0**: Minor releases
  - New features
  - Backwards compatible changes
  - Deprecation notices

- **x.y.z**: Patch releases
  - Bug fixes only
  - No new features
  - Backwards compatible

## Hotfix Releases

For urgent bug fixes:

1. Create a hotfix branch from the tag:
   ```bash
   git checkout -b hotfix/v0.1.1 v0.1.0
   ```

2. Make the fix and commit:
   ```bash
   git commit -am "fix: critical bug in XYZ"
   ```

3. Update version and changelog:
   ```bash
   # Edit pyproject.toml and CHANGELOG.md
   git commit -am "chore: bump version to 0.1.1"
   ```

4. Merge to main and create release:
   ```bash
   git checkout main
   git merge hotfix/v0.1.1
   git push origin main
   git tag v0.1.1
   git push origin v0.1.1
   ```

5. Create the GitHub release as described above

## Troubleshooting

### Build Failures

If the build fails:

1. Check the GitHub Actions logs
2. Reproduce locally:
   ```bash
   pip install build
   python -m build
   python -m twine check dist/*
   ```
3. Fix any issues and create a new release

### Publishing Failures

If publishing to PyPI fails:

1. **"File already exists"**: Version already published, increment version
2. **"Invalid credentials"**: Check trusted publishing configuration
3. **"Permission denied"**: Ensure environment settings allow the workflow

### Package Not Found on PyPI

After publishing:

1. Wait a few minutes for PyPI's CDN to update
2. Check the Actions logs to confirm successful upload
3. Verify the package name is correct (case-sensitive)

## Rolling Back a Release

If you need to remove a faulty release:

1. **Delete the GitHub release**:
   - Go to Releases → Click the release → Delete

2. **Yank the PyPI release** (don't delete):
   ```bash
   # Install twine
   pip install twine

   # Yank the release (it will still be available but marked as yanked)
   twine yank clpipe==0.2.0 -r pypi
   ```

3. **Create a new patched release** with an incremented version

Note: You cannot delete or replace a version on PyPI once published. Always increment the version for fixes.

## Resources

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github)
