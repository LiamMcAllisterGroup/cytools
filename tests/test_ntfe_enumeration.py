from itertools import islice

import numpy as np
import pytest

from cytools import Polytope

# small (h11=8) polytope whose 2-faces have 36 joint FRT choices, of
# which exactly 30 glue into an NTFE -- so both the DFS and the
# check-every-combination path finish in seconds, and pruning happens
p_h11_8 = Polytope(
    [
        [1, -1, -2, 1],
        [1, -1, 0, -1],
        [-1, 0, 1, 1],
        [-1, 2, 1, -1],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, -1],
    ]
)

# known NTFE counts (computed once with the pre-existing
# check-every-combination path and pinned here): 30 of 36 combinations
# for p_h11_8, all 81 of 81 for the h11=10 polytope below (nothing
# to prune), and 1 of 1 for p11169 (every 2-face has a single FRT, so
# the DFS sees only empty inequality blocks)
COUNTS = [
    (p_h11_8, 30),
    (Polytope(
        [
            [-2, 0, 1, 2],
            [-2, 1, 2, 0],
            [1, 1, -1, -1],
            [4, -2, -2, -1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [3, -1, -1, -1],
            [-1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ), 81),
    (Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [-1, -1, -6, -9],
        ]
    ), 1),
]


@pytest.mark.parametrize("poly,n_ntfes", COUNTS)
def test_ntfe_counts(poly, n_ntfes):
    assert len(poly.ntfe_frts(heights_only=True)) == n_ntfes


def ntfe_key(t):
    # an NTFE is identified by the triangulation's 2-face restrictions
    # (the witness FRT may differ between methods: any interior point of
    # the expanded secondary cone is valid)
    return tuple(tuple(map(tuple, f)) for f in t.restrict())


def test_matches_full_product():
    # the DFS must return exactly the NTFEs found by checking every
    # combination (passing hypers explicitly takes the pre-existing
    # path: one cold feasibility check per combination in the product)
    dfs = p_h11_8.ntfe_frts()
    product = p_h11_8.ntfe_frts(hypers=p_h11_8.ntfe_hypers())
    assert len(dfs) == len(product)
    assert {ntfe_key(t) for t in dfs} == {ntfe_key(t) for t in product}


def test_heights_realize_fine_star_triangulations():
    # raw witnesses need not use the origin; star-ifying must give FRSTs
    for h in p_h11_8.ntfe_frts(heights_only=True):
        t = p_h11_8.triangulate(heights=h, make_star=True)
        assert t.is_fine() and t.is_star()


def test_distinct_ntfes():
    frts = p_h11_8.ntfe_frts()
    assert len({ntfe_key(t) for t in frts}) == len(frts)


def test_generator():
    gen = p_h11_8.ntfe_frts(heights_only=True, as_generator=True)
    heights = list(islice(gen, 3))
    assert len(heights) == 3
    assert all(isinstance(h, np.ndarray) for h in heights)


def test_make_star():
    frsts = p_h11_8.ntfe_frts(make_star=True)
    assert all(t.is_star() for t in frsts)


# face_triangulations attaches these to Polytope on import
def test_ntfe_import_attaches_polytope_methods():
    for name in (
        "face_triangs",
        "n_2face_triangs",
        "num_2face_triangs",
        "grow_ft",
        "grow_frt",
    ):
        assert hasattr(Polytope, name), f"cytools.ntfe did not attach Polytope.{name}"


# 2-face index 19 of p_h11_8 is a 5-point polygon with 3 FRTs; it is the
# smallest face here on which a mislabeled 2-face triangulation is
# observable, since labels (3,4,7,9,12) are nowhere near 0..4
_MULTI_FRT_FACE = 19


def _all_face_frts(face_poly):
    return face_poly.all_triangulations(
        only_fine=True,
        only_star=False,
        only_regular=True,
        include_points_interior_to_facets=True,
        as_list=True,
    )


def _simps_key(t):
    return frozenset(frozenset(s) for s in t.simplices().tolist())


def test_grow_frt_keeps_parent_polytope_labels():
    # regression: grow_ft/grow_frt used to re-embed a non-2D-ambient face as
    # Polytope(pts) without labels=, so the returned triangulations carried
    # fresh 0..n-1 labels of a re-sorted polytope. Those labels are used as
    # column indices into the ambient polytope's height space by
    # _2d_frt_cone_ineqs, silently corrupting the default grow2d NTFE path.
    face = p_h11_8.faces(2)[_MULTI_FRT_FACE]
    face_poly = face.as_poly()

    grown = face_poly.grow_frt(N=3, seed=1)
    if not isinstance(grown, set):
        grown = {grown}

    assert len(grown) == 3
    for t in grown:
        assert set(t.labels) == set(face.labels)

    # ...and the grown FRTs are exactly the enumerated ones
    assert {_simps_key(t) for t in grown} == {
        _simps_key(t) for t in _all_face_frts(face_poly)
    }


def test_grow_frt_triangulations_survive_the_ntfe_extension():
    # the point of the labels: the FRT built from a grown 2-face
    # triangulation must actually restrict back to that triangulation
    face = p_h11_8.faces(2)[_MULTI_FRT_FACE]
    grown = p_h11_8.faces(2)[_MULTI_FRT_FACE].as_poly().grow_frt(N=3, seed=1)
    if not isinstance(grown, set):
        grown = {grown}

    for t in grown:
        frst = p_h11_8.triangfaces_to_frst([t])
        assert frst is not None
        restricted = frozenset(frozenset(s) for s in frst.restrict(face))
        assert restricted == _simps_key(t)


def test_ntfe_frts_grow2d_matches_full_enumeration():
    # with N_face_triangs large enough, sampling every 2-face by growth must
    # reproduce the same set of NTFEs as exhaustive enumeration
    exact = {ntfe_key(t) for t in p_h11_8.ntfe_frts()}
    grown = p_h11_8.ntfe_frts(
        hypers=p_h11_8.ntfe_hypers(
            max_npts=0, N_face_triangs=10, triang_method="grow2d", seed=0
        )
    )
    assert {ntfe_key(t) for t in grown} == exact


def test_ntfe_frts_accepts_cones():
    # regression: the documented cones= argument fed Cone objects straight
    # into the LP helper, raising TypeError
    cones = p_h11_8.ntfe_cones()
    from_cones = p_h11_8.ntfe_frts(cones=cones)
    assert {ntfe_key(t) for t in from_cones} == {
        ntfe_key(t) for t in p_h11_8.ntfe_frts()
    }


def test_ntfe_frts_does_not_shuffle_callers_list():
    # regression: random.shuffle(data) reordered the caller's list in place
    hypers = p_h11_8.ntfe_hypers()
    before = list(hypers)
    p_h11_8.ntfe_frts(hypers=hypers, N=5, seed=0, heights_only=True)
    assert list(hypers) == before


def test_ntfe_cones_handles_empty_and_generator_hypers():
    # regression: hypers[0] was probed with no guard -> IndexError/TypeError
    assert p_h11_8.ntfe_cones(hypers=[]) == []

    hypers = p_h11_8.ntfe_hypers()
    assert len(p_h11_8.ntfe_cones(hypers=iter(hypers))) == len(hypers)
