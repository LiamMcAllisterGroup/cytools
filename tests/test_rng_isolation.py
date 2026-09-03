"""
Regression tests: CYTools must never seed or advance NumPy's *global* random
state as a side effect of an ordinary call. See issue #90.

Each test seeds the global RNG, calls into CYTools, and then checks that the
next global draw is exactly what it would have been had CYTools not been
called at all.
"""

import numpy as np

from cytools import Polytope

SEED = 1234567


# random_flips needs a polytope whose triangulation actually has an
# acceptable neighbour; on a square or a simplex the walk has nowhere to go
# and raises "There was an error in the random walk."
FLIPPABLE = [
    [0, 0, 1, 0], [-2, -2, -1, -2], [0, 0, 1, 2], [-1, 0, 1, 0],
    [1, 2, -2, -1], [-1, 0, 0, -1], [0, 1, 0, 0], [1, 0, 0, 0],
]


def _expected_draw():
    """The draw a caller would get if CYTools never touched the global RNG."""
    np.random.seed(SEED)  # noqa: NPY002
    return np.random.random(8)  # noqa: NPY002


def test_glsm_charge_matrix_does_not_perturb_global_rng():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    expected = _expected_draw()

    np.random.seed(SEED)  # noqa: NPY002
    p.glsm_charge_matrix()
    p.glsm_charge_matrix(include_origin=False)
    p.clear_cache()
    p.glsm_charge_matrix(include_points_interior_to_facets=True)

    assert np.array_equal(np.random.random(8), expected)  # noqa: NPY002


def test_glsm_charge_matrix_is_reproducible():
    verts = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    first = np.array(Polytope(verts).glsm_charge_matrix())
    second = np.array(Polytope(verts).glsm_charge_matrix())
    assert np.array_equal(first, second)


def test_random_triangulations_fast_does_not_perturb_global_rng():
    p = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
    expected = _expected_draw()

    np.random.seed(SEED)  # noqa: NPY002
    p.random_triangulations_fast(
        N=2, as_list=True, progress_bar=False, seed=17, make_star=False
    )
    p.random_triangulations_fast(
        N=2, as_list=True, progress_bar=False, make_star=False
    )

    assert np.array_equal(np.random.random(8), expected)  # noqa: NPY002


def test_random_triangulations_fast_is_reproducible():
    p = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
    kwargs = dict(N=3, as_list=True, progress_bar=False, seed=17, make_star=False)
    first = [t.simplices().tolist() for t in p.random_triangulations_fast(**kwargs)]
    second = [t.simplices().tolist() for t in p.random_triangulations_fast(**kwargs)]
    assert first == second


def test_random_flips_does_not_perturb_global_rng():
    p = Polytope(FLIPPABLE)
    t = p.triangulate()
    expected = _expected_draw()

    np.random.seed(SEED)  # noqa: NPY002
    t.random_flips(2, seed=3)
    t.random_flips(2)

    assert np.array_equal(np.random.random(8), expected)  # noqa: NPY002


def test_random_flips_is_reproducible():
    p = Polytope(FLIPPABLE)
    t = p.triangulate()
    first = t.random_flips(3, seed=3).simplices().tolist()
    second = t.random_flips(3, seed=3).simplices().tolist()
    assert first == second


def test_ntfe_hypers_does_not_perturb_global_rng():
    # this polytope has 36 joint FRT choices, so N=5 exercises the
    # random-subsampling branch of ntfe_hypers
    p = Polytope(
        [
            [1, -1, -2, 1],
            [1, -1, 0, -1],
            [-1, 0, 1, 1],
            [-1, 2, 1, -1],
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, -1],
        ]
    )
    expected = _expected_draw()

    np.random.seed(SEED)  # noqa: NPY002
    p.ntfe_hypers(N=5, seed=5)

    assert np.array_equal(np.random.random(8), expected)  # noqa: NPY002


def test_no_legacy_global_rng_calls(monkeypatch):
    """The affected entry points must not reach into `np.random.*` at all."""

    def _forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"np.random.{name} was called on the global RNG")

        return _raise

    for name in ("seed", "shuffle", "choice", "normal", "uniform", "randint"):
        monkeypatch.setattr(np.random, name, _forbidden(name))

    p4 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    p4.glsm_charge_matrix()

    p3 = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
    p3.random_triangulations_fast(
        N=2, as_list=True, progress_bar=False, seed=17, make_star=False
    )

    t = Polytope(FLIPPABLE).triangulate()
    t.random_flips(2, seed=3)
