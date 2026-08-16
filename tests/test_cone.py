import shutil
import threading

import numpy as np
import pytest

from cytools import Cone
from cytools.cone import feasibility


def _canonical_face_rays(face):
    return tuple(sorted(tuple(ray) for ray in face.extremal_rays().tolist()))


def _run_with_timeout(fn, timeout):
    """
    Runs `fn` in a daemon thread and returns (result, exception). Fails the test
    if `fn` does not finish within `timeout` seconds, so that a regression that
    reintroduces an infinite loop cannot hang the test suite.
    """
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            box["exception"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"call did not terminate within {timeout} seconds")
    return box.get("result"), box.get("exception")


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


def test_find_lattice_points_terminates_when_not_enough_points():
    # the cone holds far fewer than 50 points with |coord|<=2, so this must
    # fail loudly rather than increasing the degree forever
    c = Cone([[1, 0], [0, 1]])
    result, exc = _run_with_timeout(
        lambda: c.find_lattice_points(min_points=50, max_coord=2, fast_mode=False),
        60,
    )

    assert result is None
    assert isinstance(exc, RuntimeError)
    assert "min_points" in str(exc)


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
