import numpy as np

from cytools import Polytope

# To compute nef partitions
from cytools import config

config._exp_features_enabled = True


def test_all_triangulations():
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -2, -1, -1],
            [-2, -1, -1, -1],
        ]
    )

    triang_list = p.all_triangulations(as_list=True)
    assert len(triang_list) == 2

    triang_list = p.all_triangulations(
        only_regular=False, only_star=False, only_fine=False, as_list=True
    )
    assert len(triang_list) == 6


def test_ambient_dimension():
    p = Polytope([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, -1, -1, 0]])
    assert p.ambient_dimension() == 4

    p = Polytope([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert p.ambient_dimension() == 4


def test_automorphisms():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    autos = p.automorphisms()
    assert len(autos) == 6

    autos2 = p.automorphisms(square_to_one=True)
    assert len(autos2) == 4

    for a in autos2:
        assert a.dot(a).tolist() == np.eye(4, dtype=int).tolist()


def test_chi():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    assert p.chi(lattice="N") == -540
    assert p.chi(lattice="M") == 540


def test_dimension():
    p = Polytope([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, -1, -1, 0]])
    assert p.dimension() == 3


def test_dual_polytope():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    p_dual = p.dual_polytope()

    assert p_dual.dual_polytope() is p


def test_faces():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    faces2d = p.faces(2)
    allfaces = p.faces()

    assert len(allfaces) == 5
    assert len(faces2d) == 10
    assert faces2d[0] is allfaces[2][0]


def test_facets():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    assert len(p.facets()) == 5


def test_find_2d_reflexive_subpolytopes():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    assert len(p.find_2d_reflexive_subpolytopes()) == 1


def test_glsm_basis():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    glsm = p.glsm_charge_matrix()
    assert np.linalg.matrix_rank(glsm) == np.linalg.matrix_rank(glsm[:, p.glsm_basis()])


def test_glsm_charge_matrix():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    glsm = p.glsm_charge_matrix()
    points = p.points_not_interior_to_facets()
    assert not any(glsm.dot(points).flat)


def test_glsm_linear_relations():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    glsm = p.glsm_charge_matrix()
    glsm_linrel = p.glsm_linear_relations()
    assert not any(glsm_linrel.dot(glsm.T).flat)


def test_hpq():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    assert p.hpq(0, 0, lattice="N") == 1
    assert p.hpq(0, 1, lattice="N") == 0
    assert p.hpq(1, 1, lattice="N") == 2
    assert p.hpq(1, 2, lattice="N") == 272


def test_inequalities():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    computed_ineq = set(tuple(i) for i in p.inequalities())
    real_ineq = set(
        tuple(i)
        for i in [
            [4, -1, -1, -1, 1],
            [-1, 4, -1, -1, 1],
            [-1, -1, 4, -1, 1],
            [-1, -1, -1, 4, 1],
            [-1, -1, -1, -1, 1],
        ]
    )
    assert computed_ineq == real_ineq


def test_is_affinely_equivalent():
    p1 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    p2 = Polytope(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 2], [-1, -1, -1, 0]]
    )
    assert p1.is_affinely_equivalent(p2)


def test_is_favorable():
    p1 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    p2 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -3, -6]]
    )
    assert p1.is_favorable(lattice="N")
    assert not p2.is_favorable(lattice="N")


def test_is_linearly_equivalent():
    p1 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    p2 = Polytope(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [1, 1, 1, 1]]
    )
    assert p1.is_linearly_equivalent(p2)


def test_is_reflexive():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    assert p.is_reflexive()


def test_is_solid():
    p = Polytope([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, -1, -1, 0]])
    assert not p.is_solid()
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    assert p.is_solid()


def test_minkowski_sum():
    p1 = Polytope([[1, 0, 0], [0, 1, 0], [-1, -1, 0]])
    p2 = Polytope([[0, 0, 1], [0, 0, -1]])
    p3 = p1.minkowski_sum(p2)
    assert len(p3.vertices()) == 6


def test_nef_partitions():
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]
    )
    nef_part_2 = p.nef_partitions()
    assert len(nef_part_2) == 5
    assert all(len(part) == 2 for part in nef_part_2)
    nef_part_3 = p.nef_partitions(codim=3)
    assert len(nef_part_3) == 5
    assert all(len(part) == 3 for part in nef_part_3)


def test_normal_form():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    nf = p.normal_form().tolist()
    real_nf = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-9, -6, -1, -1]]
    assert nf == real_nf

    anf = p.normal_form(affine_transform=True).tolist()
    real_anf = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [12, 17, 17, 18],
        [0, 0, 0, 0],
    ]
    assert anf == real_anf


def test_points():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    pts = set(tuple(pt) for pt in p.points())
    real_pts = set(
        tuple(pt)
        for pt in [
            [0, 0, 0, 0],
            [-1, -1, -6, -9],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -2, -3],
            [0, 0, -1, -2],
            [0, 0, -1, -1],
            [0, 0, 0, -1],
        ]
    )
    assert pts == real_pts


def test_points_to_indices():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    pts = p.points().tolist()

    assert pts[p.points_to_indices([-1, -1, -6, -9])] == [-1, -1, -6, -9]

    pts_to_check = [[-1, -1, -6, -9], [0, 0, 0, 0], [0, 0, 1, 0]]
    indices = p.points_to_indices(pts_to_check)
    pts_from_indices = [pts[i] for i in indices]
    assert pts_from_indices == pts_to_check


def test_vertices():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    vert = set(tuple(pt) for pt in p.vertices())
    real_vert = set(
        tuple(pt)
        for pt in [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ]
    )
    assert vert == real_vert


def test_volume():
    p1 = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    p2 = Polytope(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    assert p1.volume() == 1
    assert p2.volume() == 6


def test_equality():
    p1 = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    p2 = Polytope(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    assert p1 == p1
    assert p1 != p2
