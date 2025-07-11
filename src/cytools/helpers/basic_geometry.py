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

# typing
import math
from numpy.typing import ArrayLike


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
