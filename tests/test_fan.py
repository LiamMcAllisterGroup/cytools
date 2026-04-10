import numpy as np

from cytools import Polytope


def fan_fixture():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    return p.triangulate().fan()


def test_intersection_numbers_call_order_digits():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(digits=0, symmetrize=False)
    after = fan_after.intersection_numbers(symmetrize=False)
    fresh = fan_fresh.intersection_numbers(symmetrize=False)

    assert after == fresh
    assert len(after) == 121
    assert np.isclose(after[(1, 1, 2, 6)], 0.5)
    assert np.isclose(after[(1, 1, 6, 6)], 1 / 6)
    assert np.isclose(after[(2, 2, 2, 2)], 121.5)


def test_intersection_numbers_call_order_eps():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(eps=0.6, digits=None, symmetrize=False)
    after = fan_after.intersection_numbers(symmetrize=False)
    fresh = fan_fresh.intersection_numbers(symmetrize=False)

    assert after == fresh
    assert len(after) == 121
    assert (1, 1, 2, 6) in after
    assert np.isclose(after[(1, 1, 3, 6)], 1 / 3)


def test_mori_rays_after_low_precision_intersection_numbers():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(digits=0, symmetrize=False)

    after = sorted(map(tuple, fan_after.mori_rays().tolist()))
    fresh = sorted(map(tuple, fan_fresh.mori_rays().tolist()))

    assert after == fresh
