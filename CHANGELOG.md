# Changelog

All notable changes to CYTools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Release dates correspond to the `vX.Y.Z` tags in this repository.

> **Note:** this file was introduced after the `1.4.12` release. The entries
> below were reconstructed from the git history, so they summarize the commits
> between tags rather than a curated release note. Releases before `1.4.0` are
> not covered here; see `git log` and the
> [GitHub releases page](https://github.com/LiamMcAllisterGroup/cytools/releases).

## [Unreleased]

### Removed

- Dead, commented-out code in the F-theory subpackage (`FT_CY.py`,
  `Uplift_functions.py`) and in `vector_config/fan.py`.

### Changed

- Added a `.mailmap` so that author names are merged in the git log output
  (this does not change the recorded history).

## [1.4.12] - 2026-07-15

### Added

- F-theory subpackage `cytools.f_theory` (`FT_CY.py`, `Uplift_functions.py`)
  and a `demos/f_theory_uplifts.ipynb` notebook.
- HiGHS solver backend for cone operations, made the default; `highspy` pinned
  to match the HiGHS version bundled by OR-Tools.
- Parallel computation of 2-face neighbor triangulations, and a toggle for
  `only_regular` on 2-face flips.

### Changed

- Installation reworked to be Docker-free: updated Linux and macOS install
  scripts, new uninstall scripts, and the Windows installer removed.
- The 2-neighbor interface was cleaned up and `fine_neighbors_2d` made private,
  with callers routed through `neighbor_triangulations`.
- `osqp` pinned and the associated deprecation warnings fixed; minimum
  `regfans` version raised.
- Bare `except` clauses removed (they were also swallowing Ctrl-C).

### Fixed

- Hodge number bug.
- Mapping an FRT of a point configuration to a fan.
- Broken update checker.

## [1.4.11] - 2026-06-12

### Added

- Opt-in uniform Calabi-Yau sampler.
- Seed arguments for reproducible sampling.

### Changed

- NTFE code sped up by multiple orders of magnitude in some cases.
- Safer cache writing.

### Fixed

- Documentation-website generation bug.
- Clearer error message for a common constructor mistake.

## [1.4.10] - 2026-06-03

### Changed

- Faster kappa contractions, intersection numbers, and NTFE counting.
- Option to avoid copying kappa.
- `numba` import made explicit.

### Fixed

- Missing dependency.

## [1.4.9] - 2026-05-04

### Added

- Codimension-2 and higher face enumeration for pointed cones; the cone face
  lattice was split from the facet computation.

### Changed

- The external `normaliz` dependency for `hilbert_basis` was made explicit.
- The Hodge number methods of `Polytope` default to `lattice="N"`.

### Fixed

- `Polytope.is_reflexive()` for translated reflexive polytopes.
- Call-order dependence in `Fan.intersection_numbers()`.
- A `ppl` rounding issue.
- Handling of facets of 1D cones, and of 0D/-1D polytopes in `HPolytope`.

## [1.4.8] - 2026-04-12

### Added

- Preliminary polytope sampling, including handling of `samples` larger than
  the number of available polytopes.
- `latticepts` used as the backend for polytope and small-cone lattice point
  enumeration.

### Changed

- Adapted to the new `triangulumancer` API.
- Lazy construction of secondary cones.
- `max_deg` and `min_points` may be left as `None` in `compute_gv`.
- Return types cleaned up and completed across several modules.

### Fixed

- Fetching of 5d reflexive polytopes.
- Row de-duplication that still allowed some duplicates.
- A cache that was being ignored, plus caching of the hash.

## [1.4.7] - 2026-03-08

### Added

- Flag to set the GLSM/divisor basis in a deterministic way, also honored under
  dualizing and Minkowski sums.

### Changed

- Faster second Chern class computation and better tracking of the Gale basis.
- Assorted micro-optimizations; `defaultdict` avoided where possible.

## [1.4.6] - 2026-02-23

### Removed

- Docker installation and the corresponding Docker CI test.

### Changed

- More `Cone` arguments exposed.
- Higher verbosity available for intersection number computations.
- `pplpy` listed in the environment files as well.

## [1.4.5] - 2026-01-20

### Added

- Vector configuration classes (`cytools.vector_config`).
- `cytools.ntfe` and `cytools.vector_config` exposed at the package level.
- Stretching option in `find_lattice_points`.

### Changed

- Python 3.14 allowed.
- Triangulation constructors take keyword arguments only.
- Lattice points and degrees are returned sorted; default to non-verbose.
- `face_triangulations` moved into the `ntfe` subpackage.

### Fixed

- Broken cache files are now cleared automatically.
- Import bugfixes in the NTFE code.

## [1.4.4] - 2025-10-17

### Changed

- Dependency cleanup; `numpy` constrained to work better with `numba`.
- Heights are copied to/from Kähler parameters.
- Charges are always returned as at least a list.

## [1.4.3] - 2025-09-08

### Added

- `is_trilayer` check.

### Changed

- `fetch_polytopes` defaults to `as_list=True` and to the N-lattice.
- Dense representations default to unique rows.

### Fixed

- Bugfixes for simplicial cones, 1D cones, and 5d polytopes.

## [1.4.2] - 2025-08-13

### Added

- Methods to convert hyperplanes to dense and 64-bit representations.

### Changed

- Heights are reduced by their GCD only when nonzero.
- Extra plotting libraries installed by default.
- Dependencies: `sympy` removed, `mosek` added.

## [1.4.1] - 2025-08-06

### Added

- `is_degenerate` method.

### Changed

- 16-bit hyperplane matrices and corrected homogenization for star
  triangulations.
- `is_extremal` roughly 35% faster; cleaner `is_pointed` methods.
- Better verbosity for NTFE computations.

## [1.4.0] - 2025-07-31

### Added

- Conda environment files (`environment.yml`, `environment-dev.yml`) and CI
  testing on Linux arm64.
- Lineality space method for cones; integral nullspace helper in `utils`.
- Lower bound on `find_interior_point`.

### Changed

- Parallelism moved from `multiprocessing` to `joblib`.
- Fine triangulations produced with TOPCOM.
- `flint` used instead of `sympy` in parts of the code.
- Extremal ray computation sped up and simplified.

[Unreleased]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.12...HEAD
[1.4.12]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.11...v1.4.12
[1.4.11]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.10...v1.4.11
[1.4.10]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.9...v1.4.10
[1.4.9]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.8...v1.4.9
[1.4.8]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.7...v1.4.8
[1.4.7]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.6...v1.4.7
[1.4.6]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.5...v1.4.6
[1.4.5]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/LiamMcAllisterGroup/cytools/compare/v1.3.0...v1.4.0
