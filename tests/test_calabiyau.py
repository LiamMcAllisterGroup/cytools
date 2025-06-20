import numpy as np

from cytools import Polytope


def test_ambient_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.ambient_dimension() == 4


def test_ambient_variety():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    cy = v.get_cy()
    assert cy.ambient_variety() is v


def test_chi():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.chi() == -540


def test_compute_curve_volumes():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vols = cy.compute_curve_volumes(tip)
    assert np.isclose(vols, [1, 4, 1]).all()


def test_compute_cy_volume():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vol = cy.compute_cy_volume(tip)
    assert np.isclose(vol, 3.5)


def test_compute_divisor_volumes():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vols = cy.compute_divisor_volumes(tip)
    assert np.isclose(vols, [2.5, 24, 16, 2.5, 2.5, 0.5]).all()


def test_compute_inverse_kahler_metric():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    km_inv = cy.compute_inverse_kahler_metric(tip)
    assert np.isclose(km_inv, [[11, -9], [-9, 43]]).all()


def test_compute_kappa_matrix():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    km = cy.compute_kappa_matrix(tip)
    assert np.isclose(km, [[1, 1], [1, -3]]).all()


def test_compute_kappa_vector():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    kv = cy.compute_kappa_vector(tip)
    assert np.isclose(kv, [5, 1]).all()


def test_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.dimension() == 3


def test_intersection_numbers():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    intnum_nobasis = cy.intersection_numbers()
    assert len(intnum_nobasis) == 56
    intnum_basis = cy.intersection_numbers(in_basis=True)
    assert len(intnum_basis) == 3


def test_is_smooth():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.is_smooth()


def test_is_trivially_equivalent():
    p = Polytope(
        [
            [-1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [2, -1, 0, -1],
            [2, 0, -1, -1],
            [2, -1, -1, -1],
            [-1, 0, 0, 1],
            [-1, 1, 0, 1],
            [-1, 0, 1, 1],
        ]
    )
    triangs = p.all_triangulations(as_list=True)
    cy0 = triangs[0].get_cy()
    cy1 = triangs[1].get_cy()
    assert not cy0.is_trivially_equivalent(cy1)

    cys_not_triv_eq = {t.get_cy() for t in triangs}
    assert len(triangs) == 102
    assert len(cys_not_triv_eq) == 5


def test_polytope():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.polytope() is p


def test_prime_toric_divisors():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.prime_toric_divisors() == (1, 2, 3, 4, 5, 6)


def test_second_chern_class():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.second_chern_class().tolist() == [-612, 36, 306, 204, 36, 36, -6]


def test_toric_effective_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c = cy.toric_effective_cone()
    assert len(c.rays()) == 6


def test_toric_kahler_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c = cy.toric_kahler_cone()
    assert len(c.hyperplanes()) == 3


def test_toric_mori_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c1 = cy.toric_mori_cone()
    assert c1.ambient_dimension() == 7
    assert len(c1.rays()) == 3
    c2 = cy.toric_mori_cone(in_basis=True)
    assert c2.ambient_dimension() == 2
    assert len(c2.rays()) == 3


def test_triangulation():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.triangulation() is t
