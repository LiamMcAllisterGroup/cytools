# Contributing to CYTools

Thanks for your interest in CYTools. This document describes how to set up a
development environment, run the tests, and report problems.

## Supported platforms

CYTools runs on **Linux** and on **Apple Silicon (M-series) macOS**. Intel-based
Macs are not supported. See [INSTALL.md](INSTALL.md) for the full installation
instructions and for the non-development installation options.

## Setting up a development environment

Development uses the conda environment defined in `environment-dev.yml`, which
installs the dependencies (including `normaliz`, which the pip install does not
provide) and performs an editable install of CYTools:

```bash
git clone https://github.com/LiamMcAllisterGroup/cytools.git
cd cytools
conda env create -f environment-dev.yml
conda activate cytools-dev
```

The optional GNN triangulation sampler lives in a separate `dualgnn` package,
because it pulls in PyTorch. To add it on top of the development environment:

```bash
conda env update -n cytools-dev -f environment-gnn.yml
```

## Running the tests

From the repository root, with the environment activated:

```bash
pytest tests/
```

Some tests are skipped when an optional dependency is missing:

- `tests/test_gnn_sampler.py` skips unless the `dualgnn` package is importable.
- `tests/test_cone.py::test_hibert_basis` skips unless the external `normaliz`
  executable is on `PATH`.

## Continuous integration

`.github/workflows/build-test.yml` runs `pytest tests` on every pull request and
on pushes to `main`. The matrix covers Linux (x86-64 and arm64) and macOS on
Apple Silicon, across the supported Python versions, with the environment built
from `environment-dev.yml` via micromamba. Please make sure the test suite passes
locally before opening a pull request.

The other workflows are `website.yml`, which regenerates the documentation site
from the source docstrings, and `deploy.yml`, which builds the sdist and wheel
and publishes them to PyPI when a GitHub release is published.

## Code style and documentation

There is no automated formatter or linter configured in this repository, so
please match the style of the surrounding code.

Public functions and methods are documented with docstrings in the CYTools
convention (`**Description:**`, `**Arguments:**`, `**Returns:**`, and an
`**Example:**` block). These docstrings are the single source for the
documentation on [cy.tools](https://cy.tools), so new public API should be
documented the same way.

Every source file carries the GPL header; please keep it when adding new files.

## Changes and releases

- Add a short entry to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md)
  for user-visible changes.
- The version number lives in `src/cytools/__init__.py` and is read from there by
  the build backend; releases are tagged `vX.Y.Z`.

## AI usage policy

AI-generated contributions are allowed. **You are responsible for every line you
submit, however it was produced.** A pull request is your work once you open it,
and "the model wrote it" is not an explanation a reviewer can act on.

If you use an AI assistant to produce code for a contribution, please:

1. **Disclose how it was used** in the pull request, at least briefly (e.g.
   which parts were AI-generated, and whether it was assisted editing or
   wholesale generation).
2. **Check for an existing pull request** covering the same change before
   opening yours. If one exists, comment there and work with its author instead
   of opening a duplicate. AI assistants are prone to rediscovering issues that
   are already being addressed.
3. **Review the result yourself before submitting.** Read the diff line by line
   and satisfy yourself that it is correct, not merely that the tests pass.
4. **Be able to explain any part of it** when a maintainer asks.
5. **Verify claims about correctness and performance.** This is a scientific
   package: numerical results end up in published work. Assertions that a change
   is faster or that output is unchanged should come with a measurement or a
   comparison, not a plausible-sounding summary.

AI-generated prose — issue reports, pull request descriptions, review comments —
is likewise allowed, but **must be clearly marked as such**. The convention in
this repository is a leading line:

```
:robot: _AI text below_ :robot:
```

followed by a blank line. The same responsibility applies: you are accountable
for the accuracy of text posted under your account, including any bug report or
benchmark it contains. Where an AI assistant has generated a whole file, note it
in the file itself — [CHANGELOG.md](CHANGELOG.md) carries such a disclaimer.

## Reporting issues

Please open an issue at
<https://github.com/LiamMcAllisterGroup/cytools/issues>. A useful report
includes:

- the CYTools version (`import cytools; print(cytools.version)`),
- your platform and Python version, and how CYTools was installed,
- a minimal, self-contained snippet that reproduces the problem (a `Polytope`
  given by explicit points is ideal), and
- the full traceback or the incorrect output, together with what you expected.

Questions, comments and suggestions can also be sent to
[support@cy.tools](mailto:support@cy.tools).

## License

CYTools is distributed under the terms of the
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt).
By contributing, you agree that your contributions are licensed under the same
terms.
