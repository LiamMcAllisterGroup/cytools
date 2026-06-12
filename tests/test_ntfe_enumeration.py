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
