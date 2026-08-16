
import numpy as np
import pytest

from cytools import Polytope
from cytools.triangulation import _normalize_heights


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


def test_automorphism_orbit_with_filtered_automorphism():
    # automorphisms that don't preserve the triangulated point configuration
    # are filtered out (replaced by None). Explicitly asking for one used to
    # raise TypeError: 'NoneType' object is not subscriptable
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate(points=[0, 1, 2, 3, 4], make_star=False)

    autos = p.automorphisms(as_dictionary=True)
    filtered = [
        i
        for i, a in enumerate(autos)
        if any(
            (p.labels[j] in t.labels) != (p.labels[k] in t.labels)
            for j, k in a.items()
        )
    ]
    assert filtered  # otherwise this test isn't exercising anything

    assert len(t.automorphism_orbit(automorphism=filtered[0])) == 1
    assert len(t.automorphism_orbit(automorphism=[filtered[0], 1])) == 1


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


def test_heights_are_signed():
    # heights used to be stored in an *unsigned* dtype, so height differences
    # (the natural operation, e.g. for secondary-cone tests) wrapped around
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate(backend="cgal")
    h = t.heights()

    assert not np.issubdtype(h.dtype, np.unsignedinteger)
    assert (h >= 0).all()

    hi, lo = int(np.argmax(h)), int(np.argmin(h))
    assert h[hi] > h[lo]
    assert h[lo] - h[hi] < 0
    assert h[lo] - h[hi] == -(h[hi] - h[lo])


def test_heights_round_trip():
    # the stored heights must always regenerate the very same triangulation
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    for backend in ("cgal", "qhull"):
        for make_star in (True, False):
            t = p.triangulate(backend=backend, make_star=make_star)
            h = t.heights()
            assert (h >= 0).all()
            t2 = p.triangulate(backend=backend, heights=h, make_star=make_star)
            assert t == t2


def test_heights_round_trip_after_make_star():
    # with make_star=True and check_heights=False, QHull used to keep the
    # heights of the *pre-star* triangulation, so heights() described a
    # different triangulation than simplices()
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    heights_in = [17, 12, 10, 5, 6, 0, 1, 0, 3, 16]
    t = p.triangulate(
        backend="qhull", heights=heights_in, make_star=True, check_heights=False
    )
    assert t.is_star()

    h = t.heights()
    t2 = p.triangulate(
        backend="qhull", heights=h, make_star=False, check_heights=False
    )
    assert t == t2


def test_normalize_heights_no_wraparound():
    # the storage dtype used to be picked from the *unrounded* max, so a max
    # height in (2**k - 0.5, 2**k) rounded up to 2**k and wrapped to 0
    h = _normalize_heights(np.array([0.0, 255.6]))
    assert h[1] > h[0]
    assert float(h[1]) == pytest.approx(255.6)

    # integral heights straddling the old dtype boundaries survive exactly
    for m in (127, 128, 255, 256, 32767, 32768, 65535, 65536):
        out = _normalize_heights(np.array([0, 1, m]))
        assert not np.issubdtype(out.dtype, np.unsignedinteger)
        assert out.tolist() == [0, 1, m]


def test_normalize_heights_is_lossless():
    # non-integral heights are kept at full precision rather than being
    # divided by a float "gcd" and rounded
    raw = np.array([1.5, 0.25, -3.125, 7.0])
    out = _normalize_heights(raw)
    assert np.allclose(out, raw - raw.min())

    # integral heights get the gcd divided out and the minimum shifted to 0
    assert _normalize_heights(np.array([6, 12, -6])).tolist() == [2, 3, 0]
    assert _normalize_heights(np.array([0, 0, 0])).tolist() == [0, 0, 0]


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


def test_random_flips_defaults():
    # only_fine/only_regular/only_star default to None, which is documented as
    # "match the current triangulation". They used to be treated as False, so
    # the default walk was completely unrestricted
    p = Polytope(
        [[0, 0, 1, 0], [-2, -2, -1, -2], [0, 0, 1, 2], [-1, 0, 1, 0],
         [1, 2, -2, -1], [-1, 0, 0, -1], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    t = p.triangulate()
    assert t.is_fine() and t.is_star() and t.is_regular()

    for seed in range(6):
        flipped = t.random_flips(5, seed=seed)
        assert flipped.is_fine()
        assert flipped.is_star()
        assert flipped.is_regular()

    # explicitly opting out still allows non-fine/non-star triangulations
    results = [
        t.random_flips(5, only_fine=False, only_star=False, only_regular=False,
                       seed=seed)
        for seed in range(6)
    ]
    assert not all(r.is_fine() and r.is_star() for r in results)


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


def test_simplices_cache_not_aliased():
    # simplices(split_by_face=True, as_np_array=False) used to hand out the
    # cached mutable sets themselves
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()

    first = t.simplices(on_faces_dim=2, split_by_face=True, as_np_array=False)
    sizes = [len(face) for face in first]
    first[0].clear()
    first.pop()

    second = t.simplices(on_faces_dim=2, split_by_face=True, as_np_array=False)
    assert [len(face) for face in second] == sizes


def test_empty_simplices_raises_value_error():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    with pytest.raises(ValueError):
        p.triangulate(simplices=[])


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
