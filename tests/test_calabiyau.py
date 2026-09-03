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


# Genus-zero GV invariants of the degree-18 hypersurface in P(1,1,1,6,9), an
# elliptic fibration over P^2, indexed by (d_B, d_F) with d_B the degree along
# the base and d_F the degree along the fiber.
#
# Values from Chiang, Klemm, Yau, Zaslow, "Local Mirror Symmetry: Calculations
# and Interpretations" (hep-th/9903053): Table 2, column A gives the d_F = 0, 1,
# 2 entries, and Table 1 (local invariants of K_{P^2}) gives the whole d_F = 0
# column, i.e. the local P^2 invariants 3, -6, 27, ..., -360012150. These also
# appear in Candelas, Font, Katz, Morrison, hep-th/9403187.
P11169_GVS = {
    # d_F = 0: the local P^2 sequence
    (1, 0): 3,
    (2, 0): -6,
    (3, 0): 27,
    (4, 0): -192,
    (5, 0): 1695,
    (6, 0): -17064,
    (7, 0): 188454,
    (8, 0): -2228160,
    (9, 0): 27748899,
    (10, 0): -360012150,
    # d_B = 0: pure fiber classes
    (0, 1): 540,
    (0, 2): 540,
    (0, 3): 540,
    (0, 4): 540,
    (0, 5): 540,
    (0, 6): 540,
    # d_F = 1
    (1, 1): -1080,
    (2, 1): 2700,
    (3, 1): -17280,
    (4, 1): 154440,
    # d_F = 2
    (1, 2): 143370,
    (2, 2): -574560,
    (3, 2): 5051970,
    (4, 2): -57879900,
}


def test_gv_invariants_p11169_literature_values():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    cy = p.triangulate().get_cy()
    assert (cy.h11(), cy.h21()) == (2, 272)

    gvs = cy.compute_gvs(max_deg=10)

    # identify the base and fiber curve classes without assuming a particular
    # divisor basis: the base class B is the unique class with GV 3 (the local
    # P^2 degree-1 invariant), and the fiber class F is the lowest-degree class
    # with GV 540
    base_candidates = [c for c, gv in gvs.dok.items() if gv == 3]
    fiber_candidates = [c for c, gv in gvs.dok.items() if gv == 540]
    assert len(base_candidates) == 1
    assert len(fiber_candidates) == 10
    base = np.array(base_candidates[0], dtype=int)
    fiber = np.array(
        min(fiber_candidates, key=lambda c: np.dot(c, gvs.grading_vec)),
        dtype=int,
    )

    for (d_b, d_f), expected in P11169_GVS.items():
        charge = d_b * base + d_f * fiber
        assert gvs.gv(charge) == expected, f"(d_B, d_F) = {(d_b, d_f)}"

    # every class with a nonzero invariant is an effective combination of the
    # base and fiber classes
    lattice = np.array([base, fiber]).T
    for charge in gvs.dok:
        degs = np.linalg.solve(lattice, np.array(charge, dtype=float))
        assert np.allclose(degs, np.rint(degs))
        assert (np.rint(degs) >= 0).all()


def test_gv_invariants_quintic_literature_values():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    cy = p.triangulate().get_cy()
    assert (cy.h11(), cy.h21()) == (1, 101)

    gvs = cy.compute_gvs(max_deg=6)

    # genus-zero GV (instanton) numbers of the quintic threefold, from
    # Candelas, de la Ossa, Green, Parkes, Nucl. Phys. B359 (1991) 21.  n_1 is
    # Schubert's 2875 lines, n_2 = 609250 is Katz's count of conics, and
    # n_3 = 317206375 is the Ellingsrud-Stromme count of twisted cubics.
    expected = [
        2875,
        609250,
        317206375,
        242467530000,
        229305888887625,
        248249742118022000,
    ]

    assert gvs.size == len(expected)
    for deg, val in enumerate(expected, start=1):
        assert gvs.gv([deg]) == val, f"degree {deg}"



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
        # a unimodular change of whatever the current default basis is, rather
        # than a hardcoded matrix: the default is not part of the contract and
        # has changed before (#96)
        M = np.array([[1, 1], [0, 1]])
        basis_default = cy_default.divisor_basis(as_matrix=True, include_origin=True)
        cy.set_divisor_basis(M @ basis_default)

        # the same point of the Kahler cone, rewritten in the new basis
        tip = tip_default @ np.linalg.inv(M)
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
    # the point is that this does not raise; which basis is picked by default
    # is not part of the contract (it changed in #96), so only pin that
    # curve_basis and divisor_basis agree on it
    assert (cy.curve_basis() == cy.divisor_basis()).all()
    assert len(cy.curve_basis()) == cy.h11()
