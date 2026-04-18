import numpy as np

from cytools import Polytope


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


def test_default_height_audit_is_eager():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate()

    audit = t.construction_audit()
    assert audit["default_triangulation"] is True
    assert audit["height_check_pending"] is False
    assert audit["height_check_deferred"] is False
    assert audit["height_check_ran"] is True
    assert audit["height_check_method"] == "legacy-secondary-cone"
    assert audit["height_check_s"] is not None
    assert audit["height_check_relations_checked"] > 0
    assert audit["height_check_min_margin"] is not None


def test_opt_in_height_audit_is_lazy():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate(defer_height_check=True)

    audit = t.construction_audit()
    assert audit["default_triangulation"] is True
    assert audit["height_check_pending"] is True
    assert audit["height_check_deferred"] is True
    assert audit["height_check_ran"] is False

    heights = t.heights()
    audit = t.construction_audit()
    assert len(heights) == len(t.labels)
    assert audit["height_check_pending"] is False
    assert audit["height_check_ran"] is True
    assert audit["height_check_method"] == "legacy-secondary-cone"
    assert audit["height_check_s"] is not None
    assert audit["height_check_relations_checked"] > 0
    assert audit["height_check_min_margin"] is not None


def test_opt_in_fast_height_check_is_recorded():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    t = p.triangulate(fast_height_check=True)

    audit = t.construction_audit()
    assert audit["height_check_fast_requested"] is True
    assert audit["height_check_method"] == "native-local"
    assert audit["height_check_ran"] is True


def test_user_provided_heights_are_checked_eagerly():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
    )
    heights = p.triangulate().heights()
    t = p.triangulate(heights=heights)

    audit = t.construction_audit()
    assert audit["default_triangulation"] is False
    assert audit["height_check_pending"] is False
    assert audit["height_check_deferred"] is False
    assert audit["height_check_ran"] is True
    assert audit["height_check_valid"] is True
    assert audit["height_check_method"] == "legacy-secondary-cone"


def test_native_height_check_matches_secondary_cone():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )

    legacy = p.triangulate(check_heights=False)
    legacy_heights = np.asarray(legacy._heights)
    legacy_hyps = np.asarray(legacy.secondary_cone(as_cone=False))
    legacy_valid = True
    if len(legacy_hyps):
        legacy_valid = not ((legacy_hyps @ legacy_heights) < 1e-6).any()

    fast = p.triangulate(check_heights=False, fast_height_check=True)
    assert fast.check_heights(verbosity=0) == legacy_valid
    assert fast.construction_audit()["height_check_method"] == "native-local"


def test_fast_secondary_cone_matches_legacy():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    legacy = p.triangulate(check_heights=False)
    fast = p.triangulate(check_heights=False, fast_secondary_cone=True)

    legacy_hyps = sorted(map(tuple, np.asarray(legacy.secondary_cone(as_cone=False))))
    fast_hyps = sorted(
        map(tuple, np.asarray(fast.secondary_cone(as_cone=False, fast_secondary_cone=True)))
    )
    assert legacy_hyps == fast_hyps


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
