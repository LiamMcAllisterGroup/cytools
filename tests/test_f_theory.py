import numpy as np
import pytest

from cytools import Cone, Polytope
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan

from cytools.f_theory import FT_CY as FT
from cytools.f_theory import Uplift_functions as UF


# ---------------------------------------------------------------------------
# is_Gorenstein / is_reflexive_Gorenstein / Gorenstein_index
# ---------------------------------------------------------------------------


def test_is_gorenstein_positive_cases():
    for rays in (
        [[1, 0], [0, 1]],
        [[1, 0], [1, 3]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]],
    ):
        cone = Cone(rays)
        is_gor, n = UF.is_Gorenstein(cone)
        assert is_gor
        assert np.array_equal(
            cone.extremal_rays() @ n, np.ones(len(cone.extremal_rays()), dtype=int)
        )


def test_is_gorenstein_rejects_nonintegral_functional():
    # The extremal rays [[1,0,0],[0,1,0],[1,1,2]] admit no integral functional
    # taking the value one on all of them (n_1 = n_2 = 1 forces 2 n_3 = -1).
    # Before the divisibility check this returned (True, [1,1,0]), for which
    # M @ n = [1,1,2] != [1,1,1].
    cone = Cone([[1, 0, 0], [0, 1, 0], [1, 1, 2]])
    assert UF.is_Gorenstein(cone) == (False, None)


def test_is_gorenstein_rejects_rational_functional():
    # n = (1/3, 1/3) is the unique solution here, so it is not Gorenstein.
    cone = Cone([[1, 2], [2, 1]])
    assert UF.is_Gorenstein(cone) == (False, None)


def test_is_gorenstein_never_returns_wrong_functional():
    rng = np.random.default_rng(1234)
    for _ in range(60):
        rays = rng.integers(-3, 4, size=(3, 3))
        if np.linalg.matrix_rank(rays) < 3:
            continue
        try:
            cone = Cone(rays)
            extremal = cone.extremal_rays()
        except Exception:
            continue
        is_gor, n = UF.is_Gorenstein(cone)
        if is_gor:
            assert np.array_equal(
                extremal @ n, np.ones(len(extremal), dtype=int)
            ), f"bad Gorenstein functional for {extremal.tolist()}"
        else:
            assert n is None


def test_is_reflexive_gorenstein_and_index():
    # The cone over the unit square at height one is reflexive Gorenstein.
    cone = Cone([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])
    assert UF.is_reflexive_Gorenstein(cone)
    assert UF.Gorenstein_index(cone) == 1

    # The cone over a unit square translated off the origin has index 2.
    cone = Cone([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    assert UF.is_reflexive_Gorenstein(cone)
    assert UF.Gorenstein_index(cone) == 2

    not_gorenstein = Cone([[1, 0, 0], [0, 1, 0], [1, 1, 2]])
    assert not UF.is_reflexive_Gorenstein(not_gorenstein)
    with pytest.raises(ValueError):
        UF.Gorenstein_index(not_gorenstein)


# ---------------------------------------------------------------------------
# refine_fan / find_cone
# ---------------------------------------------------------------------------


def rank_deficient_fan():
    """A fan of dimension 2 living in a three-dimensional lattice."""
    vectors = np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 0], [1, 1, 0]])
    return Fan(vc=VectorConfiguration(vectors), cones=[(1, 2), (2, 3), (1, 3)])


def test_refine_fan_single_argument_on_rank_deficient_fan():
    fan = rank_deficient_fan()
    assert fan.dim < len(fan.vc.vectors()[0])

    # This used to raise "cannot access local variable 'all_vectors'".
    refined = UF.refine_fan(fan)

    cones = {tuple(sorted(c)) for c in refined.cones()}
    assert cones == {(1, 3), (2, 3), (1, 4), (2, 4)}


def test_refine_fan_full_dimensional():
    vectors = np.array([[1, 0], [0, 1], [-1, -1], [1, 1]])
    fan = Fan(vc=VectorConfiguration(vectors), cones=[(1, 2), (2, 3), (1, 3)])
    refined = UF.refine_fan(fan)
    cones = {tuple(sorted(c)) for c in refined.cones()}
    assert cones == {(1, 3), (2, 3), (1, 4), (2, 4)}


def test_find_cone_skips_non_square_cone_matrices():
    # A 3x2 matrix cannot be handed to np.linalg.solve; this used to raise
    # LinAlgError instead of moving on to the next cone.
    all_vectors = np.array([[1, 0, 0], [0, 1, 0]])
    assert UF.find_cone(np.array([1, 1, 0]), [(1, 2)], all_vectors) is None


def test_find_cone_finds_carrier_face():
    all_vectors = np.array([[1, 0], [0, 1], [-1, -1]])
    assert UF.find_cone(np.array([1, 1]), [(1, 2), (2, 3)], all_vectors) == frozenset(
        {1, 2}
    )


# ---------------------------------------------------------------------------
# is_Cartier / Cartier_index
# ---------------------------------------------------------------------------


def non_simplicial_rank_deficient_fan():
    """A single rank-deficient, non-simplicial cone inside a 3d lattice."""
    vectors = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    return Fan(vc=VectorConfiguration(vectors), cones=[(1, 2, 3)])


@pytest.mark.parametrize("weights", [[1, 2, 0], [2, 1, 0], [3, 1, 1], [1, 1, 3]])
def test_is_cartier_rejects_inconsistent_local_systems(weights):
    # numpy returns an empty residual array for these rank-deficient systems,
    # so the old `sum(residuals) < 1e-10` test certified them as Cartier even
    # though no linear functional reproduces the weights.
    fan = non_simplicial_rank_deficient_fan()
    weights = np.array(weights)
    assert UF.is_Cartier(fan, weights) == (False, None)
    assert UF.Cartier_index(fan, weights) is None

    is_cartier, data = UF.is_Cartier(fan, weights, return_Q_Cartier_data=True)
    assert not is_cartier
    assert data == [None]


def test_is_cartier_accepts_genuine_solutions():
    fan = non_simplicial_rank_deficient_fan()
    weights = np.array([1, 1, 2])
    is_cartier, data = UF.is_Cartier(fan, weights)
    assert is_cartier
    assert np.array_equal(fan.vectors((1, 2, 3)) @ data[0], -weights)
    assert UF.Cartier_index(fan, weights) == 1


def test_is_cartier_finds_integral_solution_for_underdetermined_cone():
    # The minimum-norm least-squares solution is not integral here, but an
    # integral solution does exist and must be found.
    vectors = np.array([[2, 1, 0], [1, 2, 0]])
    fan = Fan(vc=VectorConfiguration(vectors), cones=[(1, 2)])

    weights = np.array([3, 3])
    is_cartier, data = UF.is_Cartier(fan, weights)
    assert is_cartier
    assert np.array_equal(vectors @ data[0], -weights)

    # No integral solution: m = (-1/3, -1/3, *).
    assert UF.is_Cartier(fan, np.array([1, 1])) == (False, None)
    assert UF.Cartier_index(fan, np.array([1, 1])) == 3


def test_is_cartier_on_a_smooth_fan():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    fan = p.triangulate().fan()
    weights = np.ones(len(fan.vectors()), dtype=int)
    assert UF.is_Cartier(fan, weights)[0]
    assert UF.Cartier_index(fan, weights) == 1


# ---------------------------------------------------------------------------
# solve_over_integers
# ---------------------------------------------------------------------------


def test_solve_over_integers_detects_non_divisible_systems():
    # 2x = 1 has no integral solution.
    assert UF.solve_over_integers(np.array([[2]]), np.array([-1])) == (False, None)


def test_solve_over_integers_returns_a_solution():
    M = np.array([[2, 1], [1, 3]])
    b = np.array([-5, -5])
    has_sol, x = UF.solve_over_integers(M, b)
    assert has_sol
    assert np.array_equal(M @ x, -b)


# ---------------------------------------------------------------------------
# compute_partition
# ---------------------------------------------------------------------------


def test_compute_partition_supports_more_than_two_divisors():
    rays = np.array([[1, 0], [0, 1], [-1, -1]])
    divisors = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    # Used to raise a reshape ValueError because of a hardcoded 2x2 identity.
    exists, shifted = UF.compute_partition(divisors, rays)
    assert exists
    assert shifted.shape == (3, 3)
    assert np.array_equal(shifted.sum(axis=0), np.ones(3, dtype=int))


def test_compute_partition_two_divisors():
    rays = np.array([[1, 0], [0, 1], [-1, -1]])
    divisors = [np.array([1, 1, 0]), np.array([0, 0, 1])]
    exists, shifted = UF.compute_partition(divisors, rays)
    assert exists
    assert np.array_equal(shifted.sum(axis=0), np.ones(3, dtype=int))


# ---------------------------------------------------------------------------
# divisor_intersections column ordering
# ---------------------------------------------------------------------------


def test_divisor_intersections_column_order_is_deterministic():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    fan = p.triangulate().fan()
    intersection_dict = fan.intersection_numbers()
    divisors = [np.ones(len(fan.vectors()), dtype=int)]
    basis = list(UF.basis_H2_toric_fan(fan))

    from_set = UF.divisor_intersections(
        fan, intersection_dict, divisors, set(basis), as_LLL=False
    )
    from_reversed = UF.divisor_intersections(
        fan, intersection_dict, divisors, list(reversed(basis)), as_LLL=False
    )
    from_sorted = UF.divisor_intersections(
        fan, intersection_dict, divisors, sorted(basis), as_LLL=False
    )

    assert np.array_equal(from_set, from_sorted)
    assert np.array_equal(from_reversed, from_sorted)


# ---------------------------------------------------------------------------
# FT_CY: empty NHC data must stay integral / must not be None
# ---------------------------------------------------------------------------


def test_empty_nhc_labels_are_usable_as_an_index_array(monkeypatch):
    orientifold = FT.CY_orientifold.__new__(FT.CY_orientifold)
    orientifold._CY_orientifold__NHC_labels = None
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    orientifold.vectors_orbifold = lambda labels=None: (
        vectors if labels is None else vectors[np.asarray(labels) - 1]
    )
    orientifold.line_bundle = lambda: np.ones(3, dtype=int)
    monkeypatch.setattr(UF, "sections", lambda pts, weights: np.array([]))

    labels = orientifold.NHC(as_labels=True)
    assert isinstance(labels, np.ndarray)
    # A float64 empty array cannot be used as an index array.
    assert np.issubdtype(labels.dtype, np.integer)
    vectors[labels - 1]

    rays = orientifold.NHC()
    assert isinstance(rays, np.ndarray)
    assert np.issubdtype(rays.dtype, np.integer)


def test_nhc_singular_uplift_returns_an_array_when_there_are_no_nhcs():
    uplift = FT.F_Theory_Uplift.__new__(FT.F_Theory_Uplift)
    uplift.NHC = lambda as_labels=False: np.array([], dtype=int)

    result = uplift.NHC_singular_uplift()
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
