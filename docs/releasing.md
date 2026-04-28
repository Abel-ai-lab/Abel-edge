# Releasing

This repository publishes the `abel-edge` package to PyPI through GitHub
Actions.

The release trigger is a Git tag, not a branch merge. That keeps package
releases tied to one immutable version marker:

- the package version in `pyproject.toml`
- `abel_edge.__version__`
- the Git tag, such as `v0.8.0`
- the GitHub Release
- the published PyPI artifact

## Recommended Release Model

Use whichever integration branch fits the team workflow:

- merge directly to `main`
- or merge to a release branch first for final verification

In both cases, the actual PyPI publish step should happen only when you tag the
exact commit you want to release.

Do not publish on every merge to `main`. A branch tip moves; a release tag
should not.

## One-Time Setup

### 1. Configure PyPI Trusted Publisher

On PyPI, add a Trusted Publisher for this repository.

Recommended values:

- owner: `Abel-ai-causality`
- repository: `Abel-edge`
- workflow filename: `release.yml`
- environment name: `pypi`
- PyPI project name: `abel-edge`

If the project already exists on PyPI, add the publisher under the project
settings.

If the project does not exist yet, create a pending publisher first and let the
first successful publish create the project. A pending publisher does not
reserve the project name, so confirm `https://pypi.org/pypi/abel-edge/json`
still returns `Not Found` before tagging.

### 2. Create The GitHub Environment

In GitHub repository settings, create an environment named `pypi`.

The release workflow publishes from that environment. Add environment
protection rules when you want manual approval before the publish step runs.
Manual approval is recommended for the first public release.

### 3. Confirm Release Permissions

The workflow uses PyPI Trusted Publishing, so GitHub does not need a long-lived
PyPI API token.

Required workflow permissions:

- PyPI publish job: `id-token: write`
- GitHub Release job: `contents: write`

Required maintainer permissions:

- permission to push release tags
- permission to approve the `pypi` environment if approvals are enabled
- PyPI owner or maintainer access for the `abel-edge` project, or account access
  to create the pending publisher

### 4. Keep Package Metadata Current

Before any release, verify:

- `project.version` in `pyproject.toml`
- `abel_edge.__version__`
- project URLs
- license metadata
- `CHANGELOG.md`

The release workflow fails if the pushed tag does not match the package version
or if `abel_edge.__version__` does not match `pyproject.toml`.

## Release Checklist

Before tagging a release:

1. Merge the intended changes to the branch that represents the release candidate.
2. Update `pyproject.toml` and `abel_edge/__init__.py` to the new version.
3. Update `CHANGELOG.md` so versioned release notes exist outside `Unreleased`.
4. Run the test suite locally or confirm CI is green.
5. Run `python -m build --sdist --wheel`.
6. Run `python -m twine check dist/*`.
7. Install the built wheel in a clean virtual environment and run:
   - `python -c "import abel_edge; print(abel_edge.__version__)"`
   - `abel-edge version`
8. Make sure the commit you are about to tag is the exact commit you want on
   PyPI.

## Release Steps

If you release from `main`:

```bash
git checkout main
git pull
git tag v0.8.0
git push origin v0.8.0
```

After the tag is pushed, [`.github/workflows/release.yml`](../.github/workflows/release.yml)
will:

1. validate version metadata
2. build the sdist and wheel
3. run `twine check` on the built distributions
4. publish the distributions to PyPI through Trusted Publishing
5. create a GitHub Release for the same tag

## Manual Build Verification

Maintainers can run the release workflow manually from GitHub Actions with
`workflow_dispatch`.

That manual run is for build verification only:

- it still validates package metadata
- it still builds artifacts
- it does not publish to PyPI
- it does not create a GitHub Release unless the workflow is running from a
  `v*` tag

Use that path when you want to test workflow changes before pushing a release
tag.

## TestPyPI Rehearsal

For the first public release, maintainers may configure TestPyPI Trusted
Publishing and run a pre-release rehearsal. If that path is used, install from
TestPyPI in a clean environment and confirm both the import package and CLI
entry point work.

## Versioning Notes

This repository currently uses pre-1.0 package versions. Until the project
reaches 1.0, treat version bumps deliberately and document the reasoning in
`CHANGELOG.md`.

As a general rule:

- patch release: backwards-compatible fixes or small maintenance changes
- minor release: backwards-compatible feature additions or meaningful public
  surface expansion
- major release: breaking API or contract changes

The `0.8.0` release is a breaking rename release:

- distribution name: `abel-edge`
- import package: `abel_edge`
- CLI command: `abel-edge`
- runtime fact contract: `abel-edge.runtime-facts/v1`
- strategy handoff contract: `abel-edge.strategy-handoff/v1`

## Failure Modes

Common release failures:

- the Git tag does not match `pyproject.toml`
- `abel_edge.__version__` does not match `pyproject.toml`
- the `pypi` environment does not exist in GitHub
- the Trusted Publisher settings in PyPI do not match this repository or
  workflow filename
- package metadata is invalid and `twine check` fails
- the pending PyPI project name was claimed before first publish

When release automation fails, fix the underlying issue before publishing
again. Do not work around the failure by manually uploading a different artifact
built from a different commit.
