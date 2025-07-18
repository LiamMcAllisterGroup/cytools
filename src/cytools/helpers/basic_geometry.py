# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  This module contains various basic geometry helpers
# -----------------------------------------------------------------------------

# 3rd party imports
import numpy as np

# CYTools imports
from cytools import Polytope
from cytools.helpers import matrix

# typing
import math
from numpy.typing import ArrayLike


def get_bdry(self) -> set:
    """
    **Description:**
    Calculate the boundary edges.

    Currently calculated via stepping through a triangulation. There might be
    a better way to do this.

    Trivial to generalize to 3+ dimension (get the facets of each simplex,
    find those only contained in 1 simplex)

    **Arguments:**
    None

    **Returns:**
    The boundary edges. Specified as a set of frozensets, each of which
    contains the labels of the points defining a boundary edge.

    **Example:**
    ```python {3}
    from cytools import Polytope
    import lib.geom.elementary
    p = Polytope([[0,0],[3,1],[2,2]])
    p.get_bdry()
    # {frozenset({2, 4}), frozenset({1, 4}), frozenset({1, 3}), frozenset({2, 3})}
    ```
    """
    simps = self.triangulate().simplices()
    edges = matrix.flatten_top(
        [[(s[0], s[1]), (s[0], s[2]), (s[1], s[2])] for s in simps]
    )

    # organize the edges
    bdry = set()
    while len(edges):
        e = edges.pop()
        try:
            edges.pop(edges.index(e))
        except:
            bdry.add(frozenset(e))

    return bdry


Polytope.get_bdry = get_bdry


def ccw(A: list, B: list, C: list) -> bool:
    """
    **Description:**
    Check if the line AC is CCW from the line AB.

    (from stackoverflow 3838329)

    **Arguments:**
    - `A`: One point
    - `B`: Another point
    - `C`: The final point

    **Returns:**
    True iff AC is CCW from AB.

    **Example:**
    ```python {3}
    ccw([0,0],[1,1],[2,1])
    # False
    ccw([0,0],[2,1],[1,1])
    # True
    ```
    """
    return (B[0] - A[0]) * (C[1] - A[1]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: list, B: list, C: list, D: list) -> bool:
    """
    **Description:**
    Check if the line segments AB and CD intersect in the strict interior
    of both segments.

    N.B. Fails if AB is parallel to CD.

    **Arguments:**
    - `A`: One point
    - `B`: Another point
    - `C`: Another point
    - `D`: The final point

    **Returns:**
    True iff AB intersects CD. (Fails if AB is parallel to CD).

    **Example:**
    ```python {3}
    from lib.geom.elementary import intersect
    intersect([0,0],[0,1],[0,0],[1,0])
    # False
    intersect([0,0],[0,1],[0,0.5],[1,0.5])
    # False
    intersect([0,0],[0,1],[-1,0.5],[1,0.5])
    # True
    ```
    """
    if (
        (A[0] == C[0] and A[1] == C[1])
        or (A[0] == D[0] and A[1] == D[1])
        or (B[0] == C[0] and B[1] == C[1])
        or (B[0] == D[0] and B[1] == D[1])
    ):
        # intersect on end-point -> return False
        return False

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def is_primitive(pt: list) -> bool:
    """
    **Description:**
    Check if a (lattice) point is primitive. I.e.,
        1) check if the components are relatively coprime (GCD=1);
        equivalently,
        2) check if the strict interior of the line segment (0,0)->pt contains
        no lattice points.

    **Arguments:**
    - `pt`: The point.

    **Returns:**
    True iff the point has GCD=1

    **Example:**
    ```python {3}
    is_primitive([1,3])
    # True
    is_primitive([2,2])
    # False
    ```
    """
    return math.gcd(*pt) == 1


def triangle_area_2x(pts: "ArrayLike") -> float:
    """
    **Description:**
    Calculate **twice** the area of the triangle defined by the convex hull of
    the points.

    Uses the Shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula

    **Arguments:**
    - `pts`: The 3x2 array whose rows are points defining the triangle.

    **Returns:**
    Twice the area of conv(pts).

    **Example:**
    ```python {3}
    triangle_area_2x([[0,0],[0,1],[1,0]])
    # 1
    triangle_area_2x([[0,0],[0,1],[1.5,0]])
    # 1.5
    triangle_area_2x([[0,1],[1,1],[2,1]])
    # 0
    ```
    """
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]

    return abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))


def check_3consecutive_sites(pts: "ArrayLike") -> "list | None":
    """
    **Description:**
    Check if a 3xdim integral array, pts, consists of '3 consecutive sites'.

    Let pts = [a,b,c]. Then this is defined as (up to permutations of labels)
        0) all points a, b, and c being distinct,
        1) delta = b-a is primitive, and
        1) b + delta = c.
    This is equivalent to conv({a,b,c}) = 1D and one point (say, b) being the
    unique interior lattice point.

    If the above conditions hold, return a linear ordering of the points.
    If not, return None.

    **Arguments:**
    - `pts`: The 3xdim integral array whose rows are points.

    **Returns:**
    An ordering of the points if the conditions hold. Else, None.

    **Example:**
    ```python {3}
    check_3consecutive_sites([[1,0,0],[1,1,0],[1,2,0]])
    # [0,1,2]
    check_3consecutive_sites([[1,1,0],[1,2,0],[1,0,0]])
    # [2,0,1]
    check_3consecutive_sites([[1,0,0],[0,1,0],[1,1,0]])
    # None
    ```
    """
    pts_aff = pts[1:] - pts[0]
    primitiveQ = [is_primitive(pt) for pt in pts_aff]

    if primitiveQ[0]:
        # 0->1 is primitive
        if primitiveQ[1]:
            return [1, 0, 2]  # line is 1->0->2
        elif np.all(2 * pts_aff[0] == pts_aff[1]):
            return [0, 1, 2]  # line is 0->1->2
        else:
            return
    elif primitiveQ[1]:
        # 0->2 is primitive
        if np.all(2 * pts_aff[1] == pts_aff[0]):
            return [0, 2, 1]  # line is 0->2->1
        else:
            return
    else:
        return
