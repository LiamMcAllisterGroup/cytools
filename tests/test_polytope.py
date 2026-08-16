import pytest
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


def test_is_reflexive_allow_translations_for_translated_polygon():
    p = Polytope([[11, 10], [10, 11], [9, 9]])
    assert p.is_reflexive()
    assert not p.is_reflexive(allow_translations=False)


def test_is_reflexive_allow_translations_for_embedded_polygon():
    p = Polytope([[0, 0, 11, 10], [0, 0, 10, 11], [0, 0, 9, 9]])
    assert p.is_reflexive()
    assert not p.is_reflexive(allow_translations=False)


def test_is_reflexive_cache_keeps_translation_modes_separate():
    p = Polytope([[11, 10], [10, 11], [9, 9]])
    assert not p.is_reflexive(allow_translations=False)
    assert p.is_reflexive()
    assert not p.is_reflexive(allow_translations=False)


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


def test_faces_from_dual_uses_vertices():
    # regression: the top-dimensional face built from the dual's faces used
    # the raw `_labels_vertices` attribute, which is None until `vertices()`
    # is called, so the face ended up treating every lattice point as a vertex
    quintic = [
        [-1, -1, -1, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    # force the dual's faces to be computed first
    p = Polytope(quintic)
    p.dual().faces()
    from_dual = p.faces()[4][0].vertices()

    # ... and compare against computing them from scratch
    p2 = Polytope(quintic)
    from_scratch = p2.faces()[4][0].vertices()

    assert len(from_dual) == 5
    assert sorted(map(tuple, from_dual)) == sorted(map(tuple, from_scratch))
    assert (0, 0, 0, 0) not in [tuple(pt) for pt in from_dual]


def test_faces_ordering_is_deterministic():
    # regression: the dual-derived branch of `faces()` did not sort the faces
    # by label, so the ordering depended on whether the dual's faces had been
    # computed first
    quintic = [
        [-1, -1, -1, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    p = Polytope(quintic)
    p.dual().faces()
    from_dual = [[f.labels for f in dim_faces] for dim_faces in p.faces()]

    p2 = Polytope(quintic)
    from_scratch = [[f.labels for f in dim_faces] for dim_faces in p2.faces()]

    assert from_dual == from_scratch


def test_is_reflexive_without_translations_nonsolid():
    # regression: the codimension was computed as the number of *rows* of the
    # integral nullspace (= the ambient dimension) rather than the number of
    # null vectors, so this always returned False for non-solid polytopes
    assert Polytope([[-1, 0], [1, 0]]).is_reflexive(allow_translations=False)
    assert Polytope([[-1, 0, 0], [1, 0, 0]]).is_reflexive(allow_translations=False)
    assert Polytope(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
    ).is_reflexive(allow_translations=False)

    # a non-reflexive, non-solid polytope must still come out False
    assert not Polytope([[-2, 0], [1, 0]]).is_reflexive(allow_translations=False)

    # solid polytopes are unaffected
    assert Polytope([[-1, -1], [1, 0], [0, 1]]).is_reflexive(allow_translations=False)
    assert not Polytope([[-2, -2], [1, 0], [0, 1]]).is_reflexive(
        allow_translations=False
    )


def test_huang_taylor_sets_are_disjoint():
    # regression: the three Huang-Taylor sets were built as `[[]] * 3`, i.e.
    # three references to the *same* list, so every membership test was really
    # done against the union of all three
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -6, -9],
        ]
    )
    S_1, S_2, S_3 = p._huang_taylor_sets()

    # the three sets must be distinct objects
    assert S_1 is not S_2 and S_2 is not S_3 and S_1 is not S_3

    # ... and must be a genuine partition by max dot product with dual vertices
    dual_vert = p.dual().vertices()
    for i, S in enumerate([S_1, S_2, S_3]):
        for pt in S:
            assert max(np.dot(pt, v) for v in dual_vert) == i + 1

    assert set(S_1).isdisjoint(S_2)
    assert set(S_1).isdisjoint(S_3)
    assert set(S_2).isdisjoint(S_3)

    # for this polytope each set is non-empty and strictly smaller than the
    # union, so the aliased version really did behave differently
    assert len(S_1) and len(S_2) and len(S_3)
    assert len(S_1) < len(S_1) + len(S_2) + len(S_3)

    # the public method still works
    assert len(p.find_2d_reflexive_subpolytopes()) == 1


def test_all_triangulations_simplex():
    # regression: the simplex shortcut indexed the *list* returned by
    # points(as_indices=True) with a tuple, raising a TypeError
    p = Polytope([[0, 0], [1, 0], [0, 1]])

    raw = p.all_triangulations(raw_output=True, as_list=True)
    assert len(raw) == 1
    assert np.array(raw[0]).shape == (1, 3)
    assert sorted(np.array(raw[0])[0].tolist()) == [0, 1, 2]

    # the non-raw output keeps working too
    triangs = p.all_triangulations(as_list=True)
    assert len(triangs) == 1

    # and the generator form
    assert len(list(p.all_triangulations(raw_output=True))) == 1


def test_glsm_index_validation_off_by_one():
    # regression: the validation used `>` instead of `>=`, so an index equal to
    # the number of points slipped through and surfaced as a raw IndexError
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -6, -9],
        ]
    )
    n = p.points().shape[0]

    for fct in (p.glsm_charge_matrix, p.glsm_linear_relations, p.glsm_basis):
        with pytest.raises(ValueError, match="out of the allowed range"):
            fct(points=[0, n])
        with pytest.raises(ValueError, match="out of the allowed range"):
            fct(points=[0, -1])

    # the largest valid index must still be accepted
    p.glsm_charge_matrix(points=list(range(n)))


def test_volume_1d():
    # regression: the 1D branch took max/min over an (n,1)-shaped array, so it
    # returned a length-1 array instead of an int
    p = Polytope([[-2], [2]])
    assert p.volume() == 4
    assert isinstance(p.volume(), int)

    p2 = Polytope([[0], [3]])
    assert p2.volume() == 3
    assert isinstance(p2.volume(), int)

    # 0-dimensional polytopes are unaffected
    assert Polytope([[0]]).volume() == 0
