import pytest
import re

import numpy as np

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
    
    with pytest.raises(ValueError, match=re.escape("A generator, line(0, 1), was not a point...")):
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
    
    with pytest.raises(ValueError, match=re.escape("No lattice points in the Polytope! The real-valued vertices are [[0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]]..., defined from inequalities [[1.0, 0.0, -0.25], [-1.0, 0.0, 0.75], [0.0, 1.0, -0.25], [0.0, -1.0, 0.75]]...")):
        HPolytope(ineqs)