import numpy as np

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
