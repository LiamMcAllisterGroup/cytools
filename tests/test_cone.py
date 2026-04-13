import numpy as np
import pytest

from cytools import Cone


def _canonical_face_rays(face):
    return tuple(sorted(tuple(ray) for ray in face.extremal_rays().tolist()))


def test_ambient_dimension():
    c = Cone([[0, 1, 0], [1, 1, 0]])
    assert c.ambient_dimension() == 3


def test_dimension():
    c = Cone([[0, 1, 0], [1, 1, 0]])
    assert c.dimension() == 2


def test_dual_cone():
    c = Cone([[0, 1], [1, 1]])
    assert len(c.dual_cone().rays()) == 2


def test_extremal_rays():
    c = Cone([[0, 1], [1, 1], [1, 0]])
    assert len(c.extremal_rays()) == 2


def test_face_lattice_simplicial_4d():
    c = Cone(np.eye(4, dtype=int))

    all_faces = c.face_lattice()
    all_faces_with_self = c.face_lattice(include_self=True)

    assert [len(fs) for fs in all_faces] == [4, 6, 4, 1]
    assert [len(fs) for fs in all_faces_with_self] == [1, 4, 6, 4, 1]
    assert all_faces_with_self[0][0] is c
    assert c.face_lattice(0) == (c,)
    assert c.face_lattice(4)[0].dim() == 0
    assert all(f.dim() == 2 for f in c.face_lattice(2))
    assert isinstance(c.facets(), list)
    assert {_canonical_face_rays(f) for f in c.facets()} == {
        _canonical_face_rays(f) for f in c.face_lattice(1)
    }
    assert c.face_lattice(2)[0] is c.face_lattice(include_self=True)[2][0]


def test_face_lattice_nonsimplicial_3d():
    c = Cone([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])

    expected_facets = {
        ((-1, 0, 1), (0, -1, 1)),
        ((-1, 0, 1), (0, 1, 1)),
        ((0, -1, 1), (1, 0, 1)),
        ((0, 1, 1), (1, 0, 1)),
    }
    expected_rays = {
        ((-1, 0, 1),),
        ((0, -1, 1),),
        ((0, 1, 1),),
        ((1, 0, 1),),
    }

    assert len(c.face_lattice(1)) == 4
    assert len(c.face_lattice(2)) == 4
    assert {_canonical_face_rays(f) for f in c.face_lattice(1)} == expected_facets
    assert {_canonical_face_rays(f) for f in c.face_lattice(2)} == expected_rays


def test_face_lattice_non_solid_pointed():
    c = Cone([[1, 0, 0], [0, 1, 0]])

    assert c.is_pointed()
    assert not c.is_solid()
    assert len(c.face_lattice()) == 2
    assert len(c.face_lattice(1)) == 2
    assert {_canonical_face_rays(f) for f in c.face_lattice(1)} == {
        ((1, 0, 0),),
        ((0, 1, 0),),
    }
    assert isinstance(c.facets(), list)
    assert {_canonical_face_rays(f) for f in c.facets()} == {
        _canonical_face_rays(f) for f in c.face_lattice(1)
    }


def test_face_lattice_one_dimensional_cone():
    c = Cone([[1, 0]])

    assert c.face_lattice()[-1][0].dim() == 0
    assert c.face_lattice(include_self=True)[0] == (c,)
    assert c.face_lattice(1)[0].dim() == 0
    assert c.facets()[0].dim() == 0


def test_face_lattice_non_pointed_not_implemented():
    c = Cone([[1, 0], [0, 1], [-1, 0]])

    with pytest.raises(NotImplementedError):
        c.face_lattice()


def test_facets_non_pointed_still_supported():
    c = Cone([[1, 0], [0, 1], [-1, 0]])

    facets = c.facets()

    assert len(facets) == 1
    assert facets[0].dim() == 1
    assert facets[0].contains([1, 0])
    assert facets[0].contains([-1, 0])
    assert not facets[0].contains([0, 1])


def find_interior_point():
    c = Cone([[3, 2], [5, 3]])
    pt = c.find_interior_point()
    assert c.contains(pt)


def test_find_lattice_points():
    c = Cone([[3, 2], [5, 3]])
    pts = c.find_lattice_points(min_points=20)
    assert len(pts) >= 20


def test_hibert_basis():
    c = Cone([[1, 3], [2, 1]])
    hb = c.hilbert_basis()
    assert len(hb) == 4


def test_intersection():
    c1 = Cone([[1, 0], [1, 2]])
    c2 = Cone([[0, 1], [2, 1]])
    c3 = c1.intersection(c2)
    assert len(c3.rays()) == 2


def test_is_pointed():
    c1 = Cone([[1, 0], [0, 1]])
    c2 = Cone([[1, 0], [0, 1], [-1, 0]])
    assert c1.is_pointed()
    assert not c2.is_pointed()


def test_is_simplicial():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]])
    assert c1.is_simplicial()
    assert not c2.is_simplicial()


def test_is_smooth():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    assert c1.is_smooth()
    assert not c2.is_smooth()


def test_is_solid():
    c1 = Cone([[1, 0], [0, 1]])
    c2 = Cone([[1, 0, 0], [0, 1, 0]])
    assert c1.is_solid()
    assert not c2.is_solid()


def test_tip_of_stretched_cone():
    c = Cone([[3, 2], [5, 3]])
    tip = c.tip_of_stretched_cone(1).tolist()
    assert np.isclose(tip, [8.0, 5.0]).all()


def test_equality():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    assert c1 == c1
    assert c1 != c2
