import numpy as np

from cytools import Polytope


def test_ambient_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.ambient_dim() == 4


def test_automorphism_orbit():
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
    t = p.triangulate()
    orbit_all_autos = t.automorphism_orbit()
    assert len(orbit_all_autos) == 36

    orbit_all_autos_2faces = t.automorphism_orbit(on_faces_dim=2)
    assert len(orbit_all_autos_2faces) == 36

    orbit_sixth_auto = t.automorphism_orbit(automorphism=5)
    assert len(orbit_sixth_auto) == 3

    orbit_list_autos = t.automorphism_orbit(automorphism=[5, 6, 9])
    assert len(orbit_list_autos) == 12


def test_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.dimension() == 4


def test_gkz_phi():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    assert t.gkz_phi().tolist() == [18, 12, 9, 12, 12, 12, 15]


def test_heights():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()

    heights = t.heights()
    t2 = p.triangulate(heights=heights)
    assert t == t2


def test_is_equivalent():
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1, 1, 1, 0],
            [0, -1, -1, 0],
            [0, 0, 0, 1],
            [1, -2, 1, 1],
            [-2, 2, 1, -1],
            [1, 1, -1, -1],
        ]
    )
    triangs_gen = p.all_triangulations()
    t1 = next(triangs_gen)
    t2 = next(triangs_gen)
    assert not t1.is_equivalent(t2)
    assert t1.is_equivalent(t2, on_faces_dim=2)
    assert t1.is_equivalent(t2, on_faces_dim=2, use_automorphisms=False)


def test_is_fine():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.is_fine()


def test_is_regular():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.is_regular()

    t = p.triangulate(simplices=t.simplices())
    assert t.is_regular()


def test_is_star():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.is_star()


def test_is_valid():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert t.is_valid()

    t = p.triangulate(simplices=t.simplices())
    assert t.is_valid()


def test_neighbor_triangulations():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    triangs = t.neighbor_triangulations()
    assert len(triangs) == 2


def test_fine_neighbors_2d():
    p = Polytope([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    t = p.triangulate(include_points_interior_to_facets=True)
    triangs = t._fine_neighbors_2d()
    assert len(triangs) == 4
    # each neighbor differs by exactly one diagonal (two simplices swapped)
    base = {tuple(sorted(s)) for s in t.simplices().tolist()}
    for n in triangs:
        other = {tuple(sorted(s)) for s in n.simplices().tolist()}
        assert len(base - other) == 2 and len(other - base) == 2
    # agrees with the general (TOPCOM) path
    assert len(t.neighbor_triangulations(only_fine=True)) == 4


def test_two_neighbors():
    p = Polytope(
        [[0, 0, 1, 0], [-2, -2, -1, -2], [0, 0, 1, 2], [-1, 0, 1, 0],
         [1, 2, -2, -1], [-1, 0, 0, -1], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    t = p.triangulate()
    triangs = t.neighbor_triangulations(two_neighbors=True)
    assert len(triangs) == 2
    base = t.restrict()
    for n in triangs:
        assert n.is_star() and n.is_fine() and n.is_regular()
        # a 2-neighbor differs from t in exactly one 2-face restriction
        assert sum(a != b for a, b in zip(base, n.restrict())) == 1


def test_two_neighbors_skips_unextendable_flips():
    # An h11=16 polytope with a 2-face flip that cannot be extended to a full
    # triangulation, so two_neighbors returns fewer FRSTs than there are flips.
    p = Polytope(
        [
            [-2, 2, 1, 0], [0, 0, 1, 0], [4, -2, -1, -2], [-2, 0, -1, 2],
            [-2, 2, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [1, -1, 1, -2],
            [1, 0, 0, 0],
        ]
    )
    t = p.triangulate()
    total_flips = sum(
        len(ft._fine_neighbors_2d()) for ft in t.restrict(as_poly=True)
    )
    neighbors = t.neighbor_triangulations(two_neighbors=True)
    assert total_flips == 8
    assert len(neighbors) == 7  # one flip is not realizable and is skipped
    base = t.restrict()
    for n in neighbors:
        assert sum(a != b for a, b in zip(base, n.restrict())) == 1


def test_points():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    assert t.points().tolist() == p.points_not_interior_to_facets().tolist()


def test_points_to_indices():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    pts = t.points().tolist()

    assert pts[t.points_to_indices([-1, -1, -6, -9])] == [-1, -1, -6, -9]

    pts_to_check = [[-1, -1, -6, -9], [0, 0, 0, 0], [0, 0, 1, 0]]
    indices = t.points_to_indices(pts_to_check)
    pts_from_indices = [pts[i] for i in indices]
    assert pts_from_indices == pts_to_check


def test_secondary_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    sc = t.secondary_cone()
    assert len(sc.hyperplanes()) == 3


def test_simplices():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    assert len(t.simplices()) == 5
    assert len(t.simplices(on_faces_dim=2)) == 10


def test_sr_ideal():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    assert len(t.sr_ideal()) == 2


def test_equality():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    triangs = p.all_triangulations(only_fine=False, only_star=False, only_regular=False)
    t1 = next(triangs)
    t2 = next(triangs)
    assert t1 == t1
    assert t1 != t2
