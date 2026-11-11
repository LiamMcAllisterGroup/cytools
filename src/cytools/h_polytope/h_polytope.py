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
# Description:  This module contains tools designed to perform H-polytope
#               computations.
# -----------------------------------------------------------------------------

"""Lattice polytopes defined by hyperplane inequalities (H-representations)."""

# 'standard' imports
import itertools

# 3rd party imports
import numpy as np
from cytools._ppl import ppl  # ppl, with its FPU rounding side effect undone

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# CYTools imports
from cytools import polytope
from cytools.utils import gcd_list


class HPolytope(polytope.Polytope):
    r"""A lattice polytope constructed from hyperplane inequalities.

    An `HPolytope` is specified by an H-representation, i.e. a list of linear
    inequalities. The polytope cut out by the inequalities need not be a lattice
    polytope (its vertices may be rational), so it is converted into one in one of
    two ways:

    1. By taking its integer hull, i.e. the convex hull of the lattice points it
       contains (the default).
    2. By rescaling it by the smallest positive factor that makes all of its
       vertices integral (`dilate=True`). The result is normally equivalent to the
       original polytope.

    Both notions agree with the original polytope if and only if it is already a
    lattice polytope. The resulting lattice polytope is then used to construct the
    parent [`Polytope`][cytools.polytope.Polytope] object, so all of its methods are
    available.

    If every constant term $c_{i,d}$ equals 1, the inequalities describe the polar
    dual of the convex hull of the normal vectors, and the vertices are computed from
    the facets of that convex hull. Otherwise, they are computed with PPL. If the
    inequalities have no solution, an empty (-1-dimensional) polytope is constructed.

    Args:
        ineqs (ArrayLike): The defining inequalities, as a matrix $c$ where each row
            $c_i$ encodes the inequality
            $c_{i,0}x_0 + \dots + c_{i,d-1}x_{d-1} + c_{i,d} \geq 0$, with $d$ the
            ambient dimension. They must define a bounded region.
        dilate (bool): Whether to rescale the polytope so that it becomes a lattice
            polytope, instead of taking its integer hull.
        backend (str | None): The backend used to construct the convex hull. The
            options are `"ppl"`, `"qhull"`, and `"palp"`. If not specified, PPL is
            used when all constant terms equal 1; otherwise the default of
            [`Polytope`][cytools.polytope.Polytope] is used (PPL for dimensions up to
            four and PALP otherwise).
        verbosity (int): The verbosity level.

    Raises:
        ValueError: If `ineqs` is not given or is not a 2D matrix, if the region they
            define is unbounded, or if taking the integer hull and the polytope
            contains no lattice points.

    Examples:
        We construct the triangle with vertices $(0,0)$, $(1/2,0)$ and $(0,3/2)$,
        which is not a lattice polytope. By default, its integer hull is computed,
        while with `dilate=True` it is rescaled into a lattice polytope.

        >>> from cytools import HPolytope
        >>> ineqs = [[1,0,0],[0,1,0],[-6,-2,3]]
        >>> HPolytope(ineqs).vertices()
        array([[0, 0],
               [0, 1]])
        >>> HPolytope(ineqs, dilate=True).vertices()
        array([[0, 0],
               [0, 3],
               [1, 0]])

        When all constant terms are 1, the polytope is the polar dual of the convex
        hull of the normal vectors.

        >>> p = HPolytope([[1,0,1],[0,1,1],[-1,-1,1]])
        >>> p.vertices()
        array([[-1, -1],
               [-1,  2],
               [ 2, -1]])
        >>> p.is_reflexive()
        True
    """

    def __init__(
        self,
        ineqs: "ArrayLike" = None,
        dilate: bool = False,
        backend: str = None,
        verbosity: int = 0,
    ) -> None:
        # save inputs
        if ineqs is None:
            raise ValueError("The defining inequalities, `ineqs`, are required.")
        ineqs = np.array(ineqs)
        if ineqs.ndim != 2:
            raise ValueError(
                "`ineqs` must be a 2D matrix of inequalities, but the input "
                f"has {ineqs.ndim} dimension(s)."
            )
        self._ineqs = ineqs.copy()

        # compute the vertices
        if np.all(np.abs(self._ineqs[:, -1] - 1) < 1e-4):
            # constant terms are all =1. Compute Newton polytope
            if backend is None:
                backend = "ppl"
            dual, dual_poly = polytope.poly_v_to_h(self._ineqs[:, :-1], backend=backend)

            # check for rays
            if 0 in dual[:, -1]:
                raise ValueError("Dual was a polyhedron, not a polytope.")

            # map the ineqs into points. Divide integrally when we can, so
            # that a genuinely lattice polytope keeps an integer dtype (a
            # plain `/` would always give float64, and would then force the
            # slow lattice-point path below)
            dual = np.asarray(dual)
            nums, dens = dual[:, :-1], dual[:, -1:]
            if (dual.dtype.kind in "iu") and np.all(nums % dens == 0):
                self._real_vertices = nums // dens
            else:
                self._real_vertices = nums / dens
        else:
            # more complicated hyperplanes were input. Use dedicated function
            # (could likely be mapped to poly_v_to_h. Worry about rational inputs)
            self._real_vertices, _ = poly_h_to_v(self._ineqs, verbosity=verbosity)

        # shortcut if -1D
        if len(self._real_vertices) == 0:
            super().__init__(points=[], backend=backend)
            return

        # convert this into a lattice polytope
        if self._real_vertices.dtype == int:
            if verbosity > 0:
                print("HPolytope was naturally a lattice polytope!")

            points = self._real_vertices
        else:
            if verbosity > 0:
                print("Converting rational polytope to lattice polytope...")

            if dilate:
                # dilate so that the vertices are all integral
                gcd = gcd_list(self._real_vertices.flatten())
                points = np.rint(self._real_vertices / gcd).astype(int)
            else:
                # get the contained lattice points
                points = lattice_points(self._real_vertices, self._ineqs)
                if len(points) == 0:
                    error_msg = (
                        "No lattice points in the Polytope! "
                        + f"The real-valued vertices are {self._real_vertices.tolist()}..., "
                        + f"defined from inequalities {self._ineqs.tolist()}..."
                    )
                    raise ValueError(error_msg)

        # run Polytope initializer
        super().__init__(points=points, backend=backend)


# utils
# -----
def poly_h_to_v(
    hypers: "ArrayLike", verbosity: int = 0
) -> ("ArrayLike", "ppl.C_Polyhedron"):
    r"""Compute the V-representation of a polytope from its H-representation.

    That is, map hyperplane inequalities to the vertices of the polytope they define.
    The computation is done with PPL. If the inequalities are not integral, each row
    is first divided by the GCD of its entries and rounded, so rational inputs are
    made integral without changing the polytope.

    Args:
        hypers (ArrayLike): The inequalities, as a matrix $c$ where each row $c_i$
            encodes the inequality
            $c_{i,0}x_0 + \dots + c_{i,d-1}x_{d-1} + c_{i,d} \geq 0$.
        verbosity (int): The verbosity level.

    Returns:
        tuple[np.ndarray, ppl.C_Polyhedron]: The vertices of the polytope, one per
            row, and the PPL polyhedron. The vertices have an integer dtype if they
            are all integral and a float dtype otherwise. If the inequalities have no
            solution, the array of vertices is empty.

    Raises:
        ValueError: If the inequalities define an unbounded polyhedron.

    Examples:
        >>> from cytools.h_polytope.h_polytope import poly_h_to_v
        >>> pts, poly = poly_h_to_v([[1,0,0],[0,1,0],[-6,-2,3]])
        >>> pts
        array([[0.5, 0. ],
               [0. , 1.5],
               [0. , 0. ]])
    """
    hypers = np.array(hypers)  # don't use .asarray so as to ensure we copy them

    # preliminary
    dim = len(hypers[0]) - 1

    # scale hyperplanes to be integral
    if hypers.dtype != int:
        if verbosity >= 1:
            print("poly_h_to_v: Converting inequalities to be integral...")

        # divide by GCD
        for i in range(len(hypers)):
            hypers[i, :] /= gcd_list(hypers[i, :])

        # round/cast to int
        hypers = np.rint(hypers).astype(int)

    # do the work
    cs = ppl.Constraint_System()
    vrs = np.array([ppl.Variable(i) for i in range(dim)])

    # insert points to generator system
    for linexp in hypers[:, :-1] @ vrs + hypers[:, -1]:
        cs.insert(linexp >= 0)
        # cs.insert(sum(c[i] * vrs[i] for i in range(dim)) + c[-1] >= 0)

    # find polytope, vertices
    # -----------------------
    # use ppl to find the points, poly
    poly = ppl.C_Polyhedron(cs)
    pts = []
    for pt in poly.minimized_generators():
        if not pt.is_point():
            raise ValueError(f"A generator, {pt}, was not a point...")

        div = int(pt.divisor())
        if div == 1:
            # handle this separately to maintain integer typing
            pts.append([int(coeff) for coeff in pt.coefficients()])
        else:
            pts.append([int(coeff) / div for coeff in pt.coefficients()])
    pts = np.array(pts)

    # return
    return pts, poly


def lattice_points(verts: "ArrayLike", ineqs: "ArrayLike") -> "ArrayLike":
    r"""Enumerate the lattice points of a polytope.

    The polytope is given by both its vertices and its inequalities, and need not be
    a lattice polytope. All lattice points in the bounding box of the vertices are
    tested against the inequalities, so this is only suitable for small polytopes.
    This is simpler than `saturating_lattice_pts` in the `polytope` module since it
    only enumerates the lattice points.

    Args:
        verts (ArrayLike): The vertices of the polytope, one per row. They may be
            rational.
        ineqs (np.ndarray): The inequalities, as a matrix $c$ where each row $c_i$
            encodes the inequality
            $c_{i,0}x_0 + \dots + c_{i,d-1}x_{d-1} + c_{i,d} \geq 0$.

    Returns:
        list[list[int]]: The lattice points of the polytope.

    Examples:
        >>> import numpy as np
        >>> from cytools.h_polytope.h_polytope import lattice_points
        >>> ineqs = np.array([[1,0,0],[0,1,0],[-6,-2,3]])
        >>> lattice_points([[0,0],[0.5,0],[0,1.5]], ineqs)
        [[0, 0], [0, 1]]
    """
    # output variable
    _lattice_pts = []

    # basic helper variables
    dim = len(verts[0])

    # find bounding box for the lattice points
    box_min = np.ceil(np.min(verts, axis=0)).astype(int)
    box_max = np.floor(np.max(verts, axis=0)).astype(int)

    # try all lattice points
    x = np.empty(dim, dtype=int)
    for dx in itertools.product(*list(map(range, box_max - box_min + 1))):
        x = box_min + dx  # the point to try
        if all(ineqs[:, :-1] @ x + ineqs[:, -1] >= 0):
            # it passes all inequality checks!
            _lattice_pts.append(x.tolist())

    return _lattice_pts
