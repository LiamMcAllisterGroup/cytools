import shutil
import threading
import time

import numpy as np
import pytest

from cytools import config, Cone
from cytools import cone as cone_module
from cytools.cone import ExtremalityTimeLimit, feasibility, is_extremal


def _lagging_is_extremal(R, i, extFlags=None, method="lp", tol=1e-4, time_limit=None):
    """`is_extremal`, but every third ray takes much longer to check."""
    if i % 3 == 0:
        time.sleep(0.05)
    return is_extremal(R, i, extFlags, method=method, tol=tol, time_limit=time_limit)


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


def test_find_lattice_points_honors_filter_function_in_fast_mode():
    # the fast_mode shortcut cannot apply a filter, so it must not be taken
    def filter_function(pt):
        return all(coord % 2 for coord in pt)

    c = Cone([[3, 2], [5, 3]])
    pts = c.find_lattice_points(min_points=20, filter_function=filter_function)

    assert len(pts) == 6
    assert all(all(coord % 2 for coord in pt) for pt in pts)


def test_find_lattice_points_honors_process_function_in_fast_mode():
    processed = []

    c = Cone([[3, 2], [5, 3]])
    out = c.find_lattice_points(min_points=20, process_function=processed.append)

    assert out is None
    assert len(processed) >= 20
    assert (5, 3) in processed


def test_find_lattice_points_honors_max_deg_with_min_points():
    c = Cone([[3, 2], [5, 3]])
    grading = c.find_grading_vector()

    pts = c.find_lattice_points(min_points=100, max_deg=3)

    assert len(pts) > 0
    assert all(pt @ grading <= 3 for pt in pts)


@pytest.mark.skipif(
    shutil.which("normaliz") is None,
    reason="requires the external normaliz executable",
)
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


def test_is_pointed_lp_backend():
    # the LP has one variable per ray, which need not match the ambient dim + 1
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
    c2 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]])

    assert c1.is_pointed(backend="lp")
    assert not c2.is_pointed(backend="lp")


def test_extremal_rays_does_not_alias_cache():
    c = Cone([[0, 1], [1, 1], [1, 0]])

    first = c.extremal_rays()
    expected = first.tolist()
    first[0, 0] = 999

    assert c.extremal_rays().tolist() == expected
    assert c.rays().tolist() != [[999, 1]]


def test_extremal_rays_single_ray_does_not_alias_rays():
    c = Cone(hyperplanes=[[1, 0], [-1, 0], [0, 1]])

    ext = c.extremal_rays()
    expected = ext.tolist()
    ext[0, 0] = 999

    assert c.extremal_rays().tolist() == expected
    assert c.rays().tolist() == expected


def _serial_extremal_rays(rays):
    """
    Straightforward, single-threaded reference implementation of the
    extremal-ray computation, checking one ray at a time in index order. It
    mirrors what `Cone.extremal_rays` does, minus the parallel scheduling.
    """
    flags = [True] * len(rays)
    for i in range(len(rays)):
        idx, extremalQ, err = is_extremal(rays, i, flags)
        assert err is None
        flags[idx] = extremalQ
    return rays[flags]


def _deduped_rays(cone):
    # `Cone.extremal_rays` deduplicates its rays this way, and the resulting
    # order is what fixes the order of the returned extremal rays
    return np.array(list({tuple(r) for r in cone.rays()}))


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_extremal_rays_match_serial_reference(n_threads):
    """The parallel scheduling must not affect which rays come back, nor
    their order, no matter how the work is distributed."""
    rng = np.random.default_rng(20240607)
    cones = [
        Cone([[0, 1], [1, 1], [1, 0]]),
        Cone(np.eye(4, dtype=int)),
        Cone([[1, 0], [2, 0], [0, 1], [1, 1], [3, 0]]),
        Cone(np.vstack([np.eye(8, dtype=int), rng.integers(0, 4, size=(60, 8))])),
    ]

    old_n_threads = config.n_threads
    try:
        for c in cones:
            expected = _serial_extremal_rays(_deduped_rays(c))
            config.n_threads = n_threads
            found = Cone(c.rays()).extremal_rays()
            assert np.array_equal(found, expected)
    finally:
        config.n_threads = old_n_threads


def test_extremal_rays_unaffected_by_uneven_check_times(monkeypatch):
    """The per-ray checks of a real Mori cone take wildly different amounts of
    time. Since the checks are streamed to the workers, the results then come
    back in an order that has nothing to do with the ray order, which must not
    change the answer."""
    rng = np.random.default_rng(451)
    rays = np.vstack([np.eye(6, dtype=int), rng.integers(0, 3, size=(24, 6))])
    c = Cone(rays)

    expected = _serial_extremal_rays(_deduped_rays(c))

    old_n_threads = config.n_threads
    try:
        config.n_threads = 4
        monkeypatch.setattr(cone_module, "is_extremal", _lagging_is_extremal)
        found = Cone(rays).extremal_rays()
    finally:
        config.n_threads = old_n_threads

    assert np.array_equal(found, expected)


def test_extremal_rays_time_limit_keeps_undecided_rays():
    """A check that runs out of time is undecided, so the ray is kept (the
    answer stays a generating set) and the user is warned about it."""
    c = Cone([[1, 0], [0, 1], [1, 1]])
    assert len(c.extremal_rays()) == 2

    with pytest.warns(UserWarning, match="time limit"):
        found = Cone([[1, 0], [0, 1], [1, 1]]).extremal_rays(time_limit=1e-9)

    # no LP could be decided, so every ray is conservatively kept
    assert sorted(map(tuple, found.tolist())) == [(0, 1), (1, 0), (1, 1)]


def test_is_extremal_time_limit_reports_dedicated_error():
    rays = np.array([[1, 0], [0, 1], [1, 1]])

    idx, extremalQ, err = is_extremal(rays, 2, [True] * 3, time_limit=1e-9)
    assert idx == 2
    assert extremalQ is None
    assert isinstance(err, ExtremalityTimeLimit)

    # a limit that is not restrictive must not change anything
    assert is_extremal(rays, 2, [True] * 3, time_limit=60) == (2, False, None)


def test_feasibility_cpsat_honors_lower_bound():
    hyperplanes = np.array([[1, 0], [0, 1]])

    cpsat = feasibility(
        hyperplanes=hyperplanes,
        c=1,
        ambient_dim=2,
        backend="cpsat",
        lower_bound=5,
    )
    highs = feasibility(
        hyperplanes=hyperplanes,
        c=1,
        ambient_dim=2,
        backend="highs",
        lower_bound=5,
    )

    assert cpsat is not None
    assert np.all(np.asarray(cpsat) >= 5)
    assert np.all(np.asarray(highs) >= 5 - 1e-6)


def test_feasibility_accepts_sparse_rows_on_every_backend():
    # rows given as {column: value} maps, as the "highs" branch advertises
    sparse_rows = ({0: 1}, {1: 1})

    for backend in ("highs", "glop", "scip", "cpsat"):
        solution = feasibility(
            hyperplanes=sparse_rows,
            c=1,
            ambient_dim=2,
            backend=backend,
        )
        assert solution is not None, backend
        assert np.all(np.asarray(solution) >= 1 - 1e-6), backend


def test_find_interior_point_cpsat_honors_lower_bound():
    c = Cone(hyperplanes=[[1, 0], [0, 1]])
    pt = c.find_interior_point(lower=5, backend="cpsat")

    assert pt is not None
    assert np.all(np.asarray(pt) >= 5)


def test_empty_hyperplanes_accepts_plain_list():
    c = Cone(hyperplanes=[], ambient_dim=3)

    assert c.ambient_dim() == 3
    assert c.rays().shape[1] == 3
    assert not c.is_pointed()


def test_integer_dtypes_are_accepted():
    expected = [[1, 0], [0, 1]]
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        c = Cone(np.array(expected, dtype=dtype))
        assert sorted(c.rays().tolist()) == sorted(expected)

    with pytest.raises(NotImplementedError):
        Cone(np.array([[1, 0], [0, 1]], dtype=complex))


def test_rays_of_empty_dual_have_correct_shape():
    c = Cone(hyperplanes=[[1, 0], [-1, 0], [0, 1], [0, -1]])
    rays = c.rays()

    assert rays.shape == (0, 2)
    assert c.dim() == 0
    # the shape must be usable in downstream matrix operations
    assert (c.hyperplanes() @ rays.T).shape == (4, 0)
    assert np.vstack([rays, [[1, 1]]]).shape == (1, 2)


def test_is_solid_caches_ray_based_answer():
    c = Cone([[1, 0], [0, 1]])

    assert c.is_solid()
    assert c._is_solid is True


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
