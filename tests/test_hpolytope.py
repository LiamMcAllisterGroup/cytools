import pytest
import re


from cytools import Polytope
from cytools.h_polytope import HPolytope


def test_hpolytope_from_polytope():
    p = Polytope(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    assert HPolytope(p.inequalities()) == p

    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -2, -1, -1],
            [-2, -1, -1, -1],
        ]
    )
    assert HPolytope(p.inequalities()) == p


def test_polyhedron():
    ineqs = [
        [1, 0, 0],
        [-1, 0, 0],
    ]

    with pytest.raises(
        ValueError, match=re.escape("A generator, line(0, 1), was not a point...")
    ):
        HPolytope(ineqs)


def test_nonsolid_hpolytope():
    ineqs = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 1],
    ]

    p = Polytope(
        [
            [0, 0],
            [0, 1],
        ]
    )

    assert HPolytope(ineqs) == p


def test_nonlattice_hpolytope():
    ineqs = [
        [1, 0, 0],
        [-1, 0, 1.5],
        [0, 1, 0],
        [0, -1, 1.5],
    ]

    p = Polytope(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    assert HPolytope(ineqs) == p


def test_empty_hpolytope():
    ineqs = [
        [1, 0, -0.25],
        [-1, 0, 0.75],
        [0, 1, -0.25],
        [0, -1, 0.75],
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No lattice points in the Polytope! The real-valued vertices are [[0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]]..., defined from inequalities [[1.0, 0.0, -0.25], [-1.0, 0.0, 0.75], [0.0, 1.0, -0.25], [0.0, -1.0, 0.75]]..."
        ),
    ):
        HPolytope(ineqs)


def test_lattice_hpolytope_stays_integral():
    # regression: the Newton-polytope branch built the vertices with a float
    # division, so a polytope that is already lattice never took the integer
    # shortcut and fell back to a slow lattice-point enumeration
    p = Polytope(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -2, -1, -1],
            [-2, -1, -1, -1],
        ]
    )
    h = HPolytope(p.inequalities())

    assert h._real_vertices.dtype.kind in "iu"
    assert h == p

    # a genuinely rational one must still be handled as before
    h_rational = HPolytope([[1, 0, 0], [-1, 0, 1.5], [0, 1, 0], [0, -1, 1.5]])
    assert h_rational._real_vertices.dtype.kind == "f"
    assert h_rational == Polytope([[0, 0], [1, 0], [0, 1], [1, 1]])


def test_hpolytope_requires_ineqs():
    # regression: `ineqs=None` produced a 0-d array and a confusing IndexError
    with pytest.raises(ValueError, match="are required"):
        HPolytope()

    with pytest.raises(ValueError, match="2D matrix"):
        HPolytope([1, 0, 0])
