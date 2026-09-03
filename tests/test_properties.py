"""
Property-style invariant tests.

These encode mathematical identities that must hold for *any* geometry, rather
than pinned outputs for one particular example. They are deliberately run on
small polytopes so that the suite stays fast.
"""

import warnings

import numpy as np
import pytest

import cytools
from cytools import Polytope

# a few small reflexive 4d polytopes, kept module-level so each is built once
QUINTIC = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, -1, -1, -1],
]

P11169 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, -1, -6, -9],
]

# h11=8 polytope; its mirror has h11=24
H11_8 = [
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


@pytest.fixture
def experimental_features():
    """
    Temporarily enable the experimental features, restoring the previous state
    afterwards so that the global flag does not leak into other tests.
    """
    prev = cytools.config._exp_features_enabled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytools.config.enable_experimental_features()
    yield
    cytools.config._exp_features_enabled = prev


# mirror symmetry
# ---------------
@pytest.mark.parametrize("verts", [QUINTIC, P11169, H11_8])
def test_mirror_swaps_hodge_numbers(experimental_features, verts):
    """
    For a reflexive polytope, the CY hypersurface in the dual polytope's toric
    variety is the mirror: h11 and h21 are exchanged. This is Batyrev's mirror
    construction, and it is a genuine property test -- the two Hodge numbers
    are computed by completely different combinatorial expressions.

    The experimental features are enabled because the mirror of the h11=8
    example is non-favorable.
    """
    p = Polytope(verts)
    assert p.is_reflexive()

    cy = p.triangulate().get_cy()
    cy_mirror = p.dual().triangulate().get_cy()

    assert cy.h11() == cy_mirror.h21()
    assert cy.h21() == cy_mirror.h11()
    # ... hence the Euler characteristics are opposite
    assert cy.chi() == -cy_mirror.chi()


# linear equivalence
# ------------------
@pytest.mark.parametrize("verts", [QUINTIC, P11169, H11_8])
def test_linear_equivalence_contracts_to_zero(verts):
    """
    For every $m$ in the $M$ lattice, $\\sum_i \\langle m, v_i \\rangle D_i$ is
    linearly equivalent to zero, where the $v_i$ are the rays of the fan. Hence
    contracting the triple intersection numbers against any such vector must
    give identically zero.
    """
    p = Polytope(verts)
    cy = p.triangulate().get_cy()

    divs = cy.prime_toric_divisors()
    pts = p.points(which=divs)
    kappa = cy.intersection_numbers(in_basis=False, format="dense")

    # there are h11 + dim independent divisor classes and dim relations
    assert len(divs) == cy.h11() + p.dim()

    for m in np.eye(p.dim(), dtype=int):
        vec = np.zeros(kappa.shape[0])
        vec[list(divs)] = pts @ m
        # the relation must be non-trivial, else the test is vacuous
        assert np.any(vec != 0)

        contracted = np.tensordot(vec, kappa, axes=([0], [0]))
        assert np.allclose(contracted, 0, atol=1e-6)


# float-vs-exact intersection numbers
# -----------------------------------
@pytest.mark.parametrize("verts", [P11169, H11_8])
def test_float_and_exact_intersection_numbers_agree(experimental_features, verts):
    """
    The default intersection-number backend solves the linear system in
    floating point and then rounds; `exact_arithmetic=True` solves it over the
    rationals. The two must agree.
    """
    # use independent objects so that neither result can come from the other's
    # cache
    cy_float = Polytope(verts).triangulate().get_cy()
    cy_exact = Polytope(verts).triangulate().get_cy()

    intnums_float = cy_float.intersection_numbers(in_basis=True)
    intnums_exact = cy_exact.intersection_numbers(in_basis=True, exact_arithmetic=True)

    assert set(intnums_float) == set(intnums_exact)
    assert len(intnums_float) > 0
    for key, val in intnums_float.items():
        assert intnums_exact[key] == val


def test_float_and_exact_intersection_numbers_agree_mirror(experimental_features):
    """
    Same check at moderate h11 (=24, the mirror of the h11=8 example), where
    the float-solve-then-round path is doing real work. Also compares the full
    (non-basis) tensor rather than just the basis one.
    """
    cy_float = Polytope(H11_8).dual().triangulate().get_cy()
    cy_exact = Polytope(H11_8).dual().triangulate().get_cy()
    assert cy_float.h11() == 24

    for in_basis in (True, False):
        intnums_float = cy_float.intersection_numbers(in_basis=in_basis)
        intnums_exact = cy_exact.intersection_numbers(
            in_basis=in_basis, exact_arithmetic=True
        )
        assert set(intnums_float) == set(intnums_exact)
        assert len(intnums_float) > 0
        for key, val in intnums_float.items():
            assert intnums_exact[key] == val
