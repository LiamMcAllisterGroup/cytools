import numpy as np

from cytools import Polytope


def test_canonical_divisor_is_smooth():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.canonical_divisor_is_smooth()


def test_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.dimension() == 4


def test_divisor_basis():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()

    curve_basis = v.curve_basis(as_matrix=True)
    divisor_basis = v.divisor_basis(as_matrix=True)

    assert (divisor_basis.dot(curve_basis.T) == np.eye(2, dtype=int)).all()


def test_effective_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert len(v.effective_cone().rays()) == 6


def test_fan_cones():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert len(v.fan_cones()) == 9
    assert len(v.fan_cones(d=2)) == 15


def test_intersection_numbers():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    intnum_nobasis = v.intersection_numbers()
    assert len(intnum_nobasis) == 121

    intnum_basis = v.intersection_numbers(in_basis=True)
    assert len(intnum_basis) == 3


def test_is_compact():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.is_compact()


def test_is_smooth():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t1 = p.triangulate()
    v1 = t1.get_toric_variety()
    assert not v1.is_smooth()

    t2 = p.triangulate(include_points_interior_to_facets=True)
    v2 = t2.get_toric_variety()
    assert v2.is_smooth()


def test_kahler_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert len(v.kahler_cone().hyperplanes()) == 3


def test_mori_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert len(v.kahler_cone().rays()) == 2


def test_polytope():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.polytope() is p


def test_prime_toric_divisors():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.prime_toric_divisors() == (1, 2, 3, 4, 5, 6)


def test_sr_ideal():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert len(v.sr_ideal()) == 2


def test_triangulation():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    assert v.triangulation() is t


def test_equality():
    p1 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t1 = p1.triangulate()
    v1 = t1.get_toric_variety()
    p2 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t2 = p2.triangulate()
    v2 = t2.get_toric_variety()
    assert v1 == v1
    assert v1 != v2
