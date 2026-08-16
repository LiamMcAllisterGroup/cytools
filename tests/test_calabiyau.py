import contextlib

import numpy as np

import cytools.config
from cytools import Polytope


def test_ambient_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.ambient_dimension() == 4


def test_ambient_variety():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    v = t.get_toric_variety()
    cy = v.get_cy()
    assert cy.ambient_variety() is v


def test_chi():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.chi() == -540


def test_compute_curve_volumes():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vols = cy.compute_curve_volumes(tip)
    assert np.isclose(vols, [1, 4, 1]).all()


def test_compute_cy_volume():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vol = cy.compute_cy_volume(tip)
    assert np.isclose(vol, 3.5)


def test_compute_divisor_volumes():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    vols = cy.compute_divisor_volumes(tip)
    assert np.isclose(vols, [2.5, 24, 16, 2.5, 2.5, 0.5]).all()


def test_compute_inverse_kahler_metric():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    km_inv = cy.compute_inverse_kahler_metric(tip)
    assert np.isclose(km_inv, [[11, -9], [-9, 43]]).all()


def test_compute_kappa_matrix():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    km = cy.compute_kappa_matrix(tip)
    assert np.isclose(km, [[1, 1], [1, -3]]).all()


def test_compute_kappa_vector():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    kv = cy.compute_kappa_vector(tip)
    assert np.isclose(kv, [5, 1]).all()


def test_dimension():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.dimension() == 3


def test_intersection_numbers():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    intnum_nobasis = cy.intersection_numbers()
    assert len(intnum_nobasis) == 56
    intnum_basis = cy.intersection_numbers(in_basis=True)
    assert len(intnum_basis) == 3


def test_intersection_numbers_zero_as_anticanonical():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate(backend="qhull")
    cy = t.get_cy()

    canonical = cy.intersection_numbers()
    anticanonical = cy.intersection_numbers(zero_as_anticanonical=True)

    assert len(anticanonical) == len(canonical)

    for ii, val in canonical.items():
        zero_count = sum(jj == 0 for jj in ii)
        expected = val * (-1 if zero_count % 2 == 1 else 1)
        assert np.isclose(anticanonical[ii], expected)

    assert cy.intersection_numbers() == canonical


def test_intersection_numbers_zero_as_anticanonical_formats():
    # regression test: computing the intersection numbers with
    # zero_as_anticanonical=True used to alias (and then sign-flip in place) the
    # cached plain intersection numbers, corrupting every subsequent call
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()

    intnums = dict(cy.intersection_numbers())
    assert intnums[(0, 2, 3)] == -324
    assert intnums[(0, 0, 0)] == -1944

    intnums_anticanon = cy.intersection_numbers(zero_as_anticanonical=True)

    # the plain intersection numbers must be untouched
    assert dict(cy.intersection_numbers()) == intnums

    # ... and repeated calls (also in other formats) must not toggle them back
    cy.intersection_numbers(zero_as_anticanonical=True, format="coo")
    cy.intersection_numbers(zero_as_anticanonical=True, format="dense")
    assert dict(cy.intersection_numbers()) == intnums

    # the anticanonical numbers themselves flip the sign of the entries with an
    # odd number of zeros
    assert intnums_anticanon[(0, 2, 3)] == 324
    assert intnums_anticanon[(0, 0, 0)] == 1944
    assert intnums_anticanon[(0, 0, 2)] == intnums[(0, 0, 2)]
    assert intnums_anticanon[(1, 2, 3)] == intnums[(1, 2, 3)]
    assert dict(cy.intersection_numbers(zero_as_anticanonical=True)) == dict(
        intnums_anticanon
    )


def test_clear_cache_resets_fan():
    # regression test: clear_cache used to leave self._fan (and hence the
    # intersection numbers it caches) in place
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy._fan is None
    cy.compute_cy_volume([1] * cy.h11())
    assert cy._fan is not None
    cy.clear_cache(recursive=True)
    assert cy._fan is None
    assert cy._optimal_ambient_var is None


def test_cicy_hash():
    # regression test: _nef_part used to be stored as a list, which made CICYs
    # unhashable
    from cytools import config

    exp_features_enabled = config._exp_features_enabled
    config._exp_features_enabled = True
    try:
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
        cy = p.triangulate().get_cy(((1, 2, 3, 6), (4, 5, 7, 8)))
        assert isinstance(cy._nef_part, tuple)
        assert hash(cy) == hash(cy)
        assert len({cy, cy}) == 1
    finally:
        config._exp_features_enabled = exp_features_enabled


def test_is_smooth():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.is_smooth()


def test_is_trivially_equivalent():
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
    triangs = p.all_triangulations(as_list=True)
    cy0 = triangs[0].get_cy()
    cy1 = triangs[1].get_cy()
    assert not cy0.is_trivially_equivalent(cy1)

    cys_not_triv_eq = {t.get_cy() for t in triangs}
    assert len(triangs) == 102
    assert len(cys_not_triv_eq) == 5


def test_polytope():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.polytope() is p


def test_prime_toric_divisors():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.prime_toric_divisors() == (1, 2, 3, 4, 5, 6)


def test_second_chern_class():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.second_chern_class().tolist() == [-612, 36, 306, 204, 36, 36, -6]


def test_toric_effective_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c = cy.toric_effective_cone()
    assert len(c.rays()) == 6


def test_toric_kahler_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c = cy.toric_kahler_cone()
    assert len(c.hyperplanes()) == 3


def test_toric_mori_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    c1 = cy.toric_mori_cone()
    assert c1.ambient_dimension() == 7
    assert len(c1.rays()) == 3
    c2 = cy.toric_mori_cone(in_basis=True)
    assert c2.ambient_dimension() == 2
    assert len(c2.rays()) == 3


def test_triangulation():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()
    cy = t.get_cy()
    assert cy.triangulation() is t


def test_gv_invariants():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate().get_cy()

    gvs = cy.compute_gvs(min_points=100)
    assert gvs.size == 104

    gvs = cy.compute_gvs(max_deg=10)
    assert gvs.size == 65

    m_cap = cy.mori_cone_cap(in_basis=True)
    m_cap_pts = m_cap.find_lattice_points(min_points=100, fast_mode=False)
    gvs = cy.compute_gvs(m_cap_pts)
    assert gvs.size == len(m_cap_pts) - 1


def test_gv_invariants_without_cutoff():
    # regression test: querying a charge that was not computed used to raise a
    # TypeError when no degree cutoff was known (i.e. whenever the GVs were
    # computed via min_points/target_points/mcap_generators rather than max_deg)
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate().get_cy()

    gvs = cy.compute_gvs(min_points=20)
    assert gvs.cutoff is None
    # a computed charge still works ...
    assert gvs.gv([0, 1]) == 540
    # ... and an uncomputed one is simply unknown rather than a crash
    assert gvs.gv([10**6, 10**6]) is None

    # with a degree cutoff, uncomputed charges below the cutoff are still 0
    gvs = cy.compute_gvs(max_deg=10)
    assert gvs.cutoff == 10
    assert gvs.gv([10**6, 10**6]) is None


def _reference_kappa(cy):
    """
    Intersection numbers in the current basis, taken straight from
    `intersection_numbers` (the source of truth for the basis).
    """
    return np.asarray(cy.intersection_numbers(in_basis=True, format="dense"))


@contextlib.contextmanager
def _experimental_features():
    prev = cytools.config._exp_features_enabled
    cytools.config._exp_features_enabled = True
    try:
        yield
    finally:
        cytools.config._exp_features_enabled = prev


def test_volume_and_kahler_methods_honor_matrix_divisor_basis():
    # regression test: the volume/Kahler methods used to read the intersection
    # numbers off the fan of the defining triangulation, which only knows the
    # polytope's default (Gale) basis. A basis set with set_divisor_basis was
    # silently ignored, giving wrong volumes and metrics.
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate().get_cy()

    with _experimental_features():
        # a unimodular change of the default basis [5, 6]
        cy.set_divisor_basis([[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1]])

        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        kappa = _reference_kappa(cy)

        K = np.tensordot(kappa, tip, axes=([-1], [0]))
        assert np.isclose(cy.compute_cy_volume(tip), (K @ tip) @ tip / 6)
        assert np.isclose(cy.compute_kappa_matrix(tip), K).all()
        assert np.isclose(cy.compute_kappa_vector(tip), K @ tip).all()
        assert np.isclose(
            cy.compute_divisor_volumes(tip, in_basis=True), (K @ tip) / 2
        ).all()

        xvol = (K @ tip) @ tip / 6
        tau = (K @ tip) / 2
        kinv = 4 * (np.outer(tau, tau) - K * xvol)
        assert np.isclose(cy.compute_inverse_kahler_metric(tip), kinv).all()
        assert np.isclose(cy.compute_kahler_metric(tip), np.linalg.inv(kinv)).all()

        # the total volume is basis independent
        assert np.isclose(cy.compute_cy_volume(tip), 3.5)


def test_volume_methods_honor_index_divisor_basis():
    # same bug as above, but with a basis given as a vector of indices
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate().get_cy()
    cy.set_divisor_basis([1, 6])

    tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
    K = np.tensordot(_reference_kappa(cy), tip, axes=([-1], [0]))

    assert np.isclose(cy.compute_cy_volume(tip), (K @ tip) @ tip / 6)
    assert np.isclose(cy.compute_kappa_matrix(tip), K).all()


def test_compute_divisor_volumes_out_of_basis_honors_basis():
    # the conversion of the basis divisor volumes to the volumes of the prime
    # toric divisors used the polytope's GLSM charge matrix, which is only the
    # right transformation for the default basis
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy_default = p.triangulate().get_cy()
    tip_default = cy_default.toric_kahler_cone().tip_of_stretched_cone(1)
    expected = cy_default.compute_divisor_volumes(tip_default)
    assert np.isclose(expected, [2.5, 24, 16, 2.5, 2.5, 0.5]).all()

    p2 = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p2.triangulate().get_cy()
    with _experimental_features():
        cy.set_divisor_basis([[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1]])
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        # the tip is the same point of the Kahler cone, written in a new basis
        assert np.isclose(
            tip.dot(cy.divisor_basis(as_matrix=True, include_origin=False)),
            tip_default.dot(
                cy_default.divisor_basis(as_matrix=True, include_origin=False)
            ),
        ).all()
        # ... so the volumes of the prime toric divisors must be unchanged
        assert np.isclose(cy.compute_divisor_volumes(tip), expected).all()


def test_curve_basis_with_all_points_triangulation():
    # regression test: curve_basis picked its default basis using the
    # triangulation point set while divisor_basis (and glsm_charge_matrix) use
    # the origin plus the prime toric divisors. For a CY built from a
    # triangulation of all the points of the polytope these differ, and
    # curve_basis raised "Indices are not in appropriate range."
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate(points=p.labels).get_cy()
    assert (cy.curve_basis() == [5, 6]).all()
    assert (cy.curve_basis() == cy.divisor_basis()).all()
