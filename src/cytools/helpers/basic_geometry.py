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

"""Basic 2D geometry helpers.

Importing this module also adds a `get_bdry` method to
[`Polytope`][cytools.polytope.Polytope].
"""

# 'standard' imports
from collections import Counter

# CYTools imports
from cytools import Polytope
from cytools.helpers import matrix

# typing
from numpy.typing import ArrayLike


def get_bdry(self) -> set:
    """Compute the boundary edges of a 2-dimensional polytope.

    The edges are found from the default triangulation of the polytope (see
    [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate]): an edge is on
    the boundary if and only if it belongs to exactly one triangle. Consequently,
    only points used by that triangulation appear; in particular, points interior to
    facets (i.e. to boundary edges) are excluded by default. This could be
    generalized to higher dimensions by using the facets of each simplex.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as
    `Polytope.get_bdry` when `cytools.helpers.basic_geometry` is imported.

    Returns:
        set[frozenset[int]]: The boundary edges, each given as a frozenset with the
            labels of its two endpoints.

    Examples:
        >>> from cytools import Polytope
        >>> import cytools.helpers.basic_geometry
        >>> p = Polytope([[0,0],[3,1],[2,2]])
        >>> sorted(sorted(int(i) for i in edge) for edge in p.get_bdry())
        [[1, 2], [1, 3], [2, 3]]
    """
    simps = self.triangulate().simplices()
    edges = matrix.flatten_top(
        [[(s[0], s[1]), (s[0], s[2]), (s[1], s[2])] for s in simps]
    )

    # organize the edges
    #
    # An edge is on the boundary iff it belongs to an odd number of simplices
    # (i.e. exactly one, for a triangulation). Counting is O(E); the previous
    # pop()/index() pairing was O(E^2).
    counts = Counter(edges)

    return {frozenset(e) for e, n in counts.items() if n % 2}


Polytope.get_bdry = get_bdry


def ccw(A: list, B: list, C: list) -> bool:
    """Check whether three points are in counterclockwise order.

    That is, whether the ray from $A$ to $C$ is strictly counterclockwise from the
    ray from $A$ to $B$ (see
    [this Stack Overflow question](https://stackoverflow.com/questions/3838329)).
    Collinear points give `False`.

    Args:
        A (list): The first point, in 2D.
        B (list): The second point, in 2D.
        C (list): The third point, in 2D.

    Returns:
        bool: Whether $A$, $B$, $C$ are in strictly counterclockwise order.

    Examples:
        >>> from cytools.helpers.basic_geometry import ccw
        >>> ccw([0,0],[1,1],[2,1])
        False
        >>> ccw([0,0],[2,1],[1,1])
        True
    """
    return (B[0] - A[0]) * (C[1] - A[1]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: list, B: list, C: list, D: list) -> bool:
    """Check whether two line segments in the plane cross each other.

    Segments sharing an endpoint are reported as not intersecting. Degenerate
    configurations, such as collinear segments or an endpoint of one segment lying
    on the other segment, are not handled reliably.

    Args:
        A (list): The first endpoint of the first segment.
        B (list): The second endpoint of the first segment.
        C (list): The first endpoint of the second segment.
        D (list): The second endpoint of the second segment.

    Returns:
        bool: Whether the segments $AB$ and $CD$ intersect.

    Examples:
        >>> from cytools.helpers.basic_geometry import intersect
        >>> intersect([0,0],[0,1],[0,0],[1,0])
        False
        >>> intersect([0,0],[0,1],[-1,0.5],[1,0.5])
        True
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


def triangle_area_2x(pts: "ArrayLike") -> float:
    r"""Compute twice the area of a triangle.

    The area is computed with the [shoelace
    formula](https://en.wikipedia.org/wiki/Shoelace_formula), so the result is an
    integer for lattice triangles.

    Args:
        pts (ArrayLike): A $3 \times 2$ array whose rows are the vertices of the
            triangle.

    Returns:
        float: Twice the area of the triangle. It has the same type as the
            coordinates (e.g. an `int` for integer coordinates).

    Examples:
        >>> from cytools.helpers.basic_geometry import triangle_area_2x
        >>> triangle_area_2x([[0,0],[0,1],[1,0]])
        1
        >>> triangle_area_2x([[0,0],[0,1],[1.5,0]])
        1.5
        >>> triangle_area_2x([[0,1],[1,1],[2,1]])
        0
    """
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]

    return abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))

