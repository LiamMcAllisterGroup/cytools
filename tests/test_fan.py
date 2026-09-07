import numpy as np
import pytest

from cytools import Polytope
from cytools.vector_config import VectorConfiguration


def fan_fixture():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    return p.triangulate().fan()


def nonreflexive_fan_fixture():
    """A fan over a (dilated, hence non-reflexive) vector configuration."""
    fan = fan_fixture()
    vc = VectorConfiguration(2 * fan.vc.vectors(), labels=fan.vc.labels)
    return vc.subdivide(cells=fan.cones())


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


def test_secondary_cone_cache_keyed_on_via_circuits():
    # the two constructions can genuinely disagree (only via_circuits=False is
    # usable for regularity checks), so they must not share a cache slot
    fan = fan_fixture()

    circ = fan.secondary_cone(via_circuits=True, verbosity=-1)
    fold = fan.secondary_cone(via_circuits=False, verbosity=-1)

    assert circ is not fold
    assert circ.hyperplanes().shape != fold.hyperplanes().shape

    # each variant is still cached
    assert fan.secondary_cone(via_circuits=True, verbosity=-1) is circ
    assert fan.secondary_cone(via_circuits=False, verbosity=-1) is fold

    # and neither is polluted by the other having been asked for first
    fresh_fold = fan_fixture().secondary_cone(via_circuits=False, verbosity=-1)
    assert sorted(map(tuple, fold.hyperplanes().tolist())) == sorted(
        map(tuple, fresh_fold.hyperplanes().tolist())
    )


def test_flop_linear_does_not_shadow_kappa_method():
    fan = fan_fixture()

    heights = np.array(fan.heights())
    out = fan.flop_linear(
        direction=-heights + np.array([1, 1, 5, 5, 5, 5]),
        max_N_flips=3,
        verbosity=-1,
    )
    end_fan = out[2]

    # the hooks stash an ndarray; it must not shadow the `kappa` method
    assert isinstance(end_fan._kappa_np, np.ndarray)
    assert callable(end_fan.kappa)
    assert np.array_equal(
        end_fan.kappa(pushed_down=True, in_basis=True, as_np_array=True),
        end_fan.intersection_numbers(
            pushed_down=True, in_basis=True, as_np_array=True
        ),
    )


def test_restricted_simps_padded():
    fan = fan_fixture()

    unpadded = fan.restricted_simps(to_dim=1)
    padded = fan.restricted_simps(to_dim=1, padded=True)

    # restrictions to edges have 2 points, so padding is exercised
    assert any(len(simp) == 2 for face in unpadded for simp in face)
    assert all(len(simp) == 3 for face in padded for simp in face)
    for face_unpadded, face_padded in zip(unpadded, padded):
        for simp, simp_padded in zip(face_unpadded, face_padded):
            assert simp_padded == simp + [simp[-1]]

    # and it works for the face-index representation too
    padded_inds = fan.restricted_simps(to_dim=1, padded=True, as_face_inds=True)
    assert all(len(simp) == 3 for face in padded_inds for simp in face)


def test_divisor_basis_requires_reflexive():
    fan = nonreflexive_fan_fixture()
    vc = fan.vc

    assert vc.is_reflexive is False
    with pytest.raises(NotImplementedError, match="reflexive"):
        vc.divisor_basis
    with pytest.raises(NotImplementedError, match="reflexive"):
        vc.divisor_basis_inds

    # the Mori cone doesn't need a basis unless one is requested
    assert fan.mori_cone().rays().shape[1] == len(vc.labels) + 1
    with pytest.raises(NotImplementedError, match="reflexive"):
        fan.mori_cone(in_basis=True)


def test_intersection_numbers_purges_stale_cache_keys():
    fan = fan_fixture()

    # emulate a cache populated for a different label set
    fan._kappa = {(-1, -1, -1, -1): 12.0}
    fan._kappa_known_labels = {-1}
    fan._kappa_view_cache = {}
    fan._kappa_default = None

    kappa = fan.intersection_numbers(symmetrize=False)

    assert (-1, -1, -1, -1) not in kappa
    assert kappa == fan_fixture().intersection_numbers(symmetrize=False)


def test_gale_forwards_set_basis():
    vc = fan_fixture().vc

    assert vc.is_reflexive
    assert np.array_equal(vc.gale(), vc.gale(set_basis=True))

    # the kwarg is honored rather than silently dropped: without a divisor
    # basis, asking for one is an error
    nonreflexive = nonreflexive_fan_fixture().vc
    assert not nonreflexive.is_reflexive
    with pytest.raises(AssertionError):
        nonreflexive.gale(set_basis=True)


def quintic_dual_polytope():
    """The polar dual of the standard quintic polytope; has facet-interior pts."""
    return Polytope(
        [
            [-1, -1, -1, -1],
            [4, -1, -1, -1],
            [-1, 4, -1, -1],
            [-1, -1, 4, -1],
            [-1, -1, -1, 4],
        ]
    )


def facet_interior_labels(p):
    return sorted(set(p.labels) - set(p.labels_not_facet))


def test_fan_rejects_triangulation_omitting_origin():
    p = quintic_dual_polytope()
    triang = p.triangulate(points=p.labels_vertices, verbosity=0)

    # previously built a fan with no rays at all, which only surfaced later as
    # a vstack shape mismatch inside intersection_numbers
    with pytest.raises(ValueError, match="omits the origin"):
        triang.fan()


def test_fan_rejects_polytope_without_origin():
    p = Polytope([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 2]])
    assert p.label_origin is None

    with pytest.raises(ValueError, match="not a lattice point"):
        p.triangulate(verbosity=0).fan()


def test_fan_rejects_unused_origin():
    # a non-fine triangulation can keep the origin as a label while using it in
    # no simplex, which reaches an empty star without omitting the origin
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    heights = [0] * len(p.labels)
    heights[list(p.labels).index(p.label_origin)] = 50
    triang = p.triangulate(heights=heights, make_star=False, verbosity=0)

    assert not triang.is_fine()
    assert p.label_origin in triang.labels

    with pytest.raises(ValueError, match="no simplex"):
        triang.fan()


def test_fan_includes_facet_interior_points_when_used():
    p = quintic_dual_polytope()
    points = list(p.labels_not_facet) + facet_interior_labels(p)[:1]
    triang = p.triangulate(points=points, verbosity=0)

    # the vc must contain every point the triangulation uses; this previously
    # raised a bare KeyError from inside regfans
    fan = triang.fan()

    assert fan.vectors().shape == (len(points) - 1, 4)
    assert len(triang.vc().labels) == len(p.labels) - 1


def test_fan_rejects_vc_missing_used_points():
    p = quintic_dual_polytope()
    points = list(p.labels_not_facet) + facet_interior_labels(p)[:1]
    triang = p.triangulate(points=points, verbosity=0)

    # explicitly asking for the narrower vc used to yield a Fan whose cones
    # referenced labels it did not contain, deferring the KeyError to vectors()
    with pytest.raises(ValueError, match="absent from the vector configuration"):
        triang.fan(include_points_interior_to_facets=False)


def test_fan_from_origin_and_vertices():
    p = quintic_dual_polytope()
    points = (p.label_origin,) + p.labels_vertices
    fan = p.triangulate(points=points, verbosity=0).fan()

    assert fan.vectors().shape == (5, 4)
    assert len(fan.simplices()) == 5
    assert set(map(tuple, fan.vectors())) == set(map(tuple, p.vertices()))


def test_fan_canonical_point_sets_unchanged():
    p = quintic_dual_polytope()

    assert p.triangulate(verbosity=0).fan().vectors().shape == (105, 4)

    all_pts = p.triangulate(include_points_interior_to_facets=True, verbosity=0)
    assert all_pts.fan().vectors().shape == (125, 4)
