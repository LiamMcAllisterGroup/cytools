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

# 'standard' imports
import itertools

# 3rd party imports
import numpy as np
import ppl

# CYTools imports
from cytools import polytope
from cytools.utils import gcd_list


class HPolytope(polytope.Polytope):
    """
    This class handles all computations relating to H-polytopes. These are not
    always lattice. There are two primary methods for making them lattice:
        1) study their integer hull. This is the convex hull of their
           contained lattice points.
        2) study their smallest dilation such that they become lattice.
           This is a normally equivalent polytope.
    These notions both agree with the original polytope iff it was lattice.

    ## Constructor

    ### `cytools.h_polytope.h_polytope.HPolytope`

    **Description:**
    Constructs a `HPolytope` object describing a lattice polytope. This is
    handled by the hidden [`__init__`](#__init__) function.

    **Arguments:**
    - `ineqs`: The defining hyperplane inequalities. Of the form c, a
        matrix, such that each row, c[i], corresponds to an inequality:
            c[i][0]*x_0 + ... + c[i][dim-1]*x_{dim-1} + c[i][dim] >= 0
        If the corresponding vertices are rational, then convex hull of the
        contained lattice points will be stored.
    - `dilate`: Whether to dilate the rational polytope into a normally
        equivalent integer polytope or not. If False, then just compute the
        integer hull (convex hull of contained lattice points) instead.
    - `backend`: A string that specifies the backend used to construct the
        convex hull. The available options are "ppl", "qhull", or "palp". When
        not specified, it uses PPL for dimensions up to four, and palp
        otherwise.
    """

    def __init__(
        self,
        ineqs: "ArrayLike" = None,
        dilate: bool = False,
        backend: str = None,
        verbosity: int = 0,
    ) -> None:
        """
        **Description:**
        Initializes a `HPolytope` object describing a lattice polytope.

        **Arguments:**
        - `ineqs`: The defining hyperplane inequalities. Of the form c, a
            matrix, such that each row, c[i], corresponds to an inequality:
                c[i][0]*x_0 + ... + c[0][dim-1]*x_{dim-1} + c[0][dim] >= 0
            If the corresponding vertices are rational, then convex hull of the
            contained lattice points will be stored.
        - `backend`: A string that specifies the backend used to construct the
            convex hull. The available options are "ppl", "qhull", or "palp".
            When not specified, it uses PPL for dimensions up to four, and palp
            otherwise.

        **Returns:**
        Nothing.
        """
        # save inputs
        ineqs = np.array(ineqs)
        self._ineqs = ineqs.copy()

        # compute the vertices
        if np.all(np.abs(self._ineqs[:,-1]-1)<1e-4):
            # constant terms are all =1. Compute Newton polytope
            if backend is None:
                backend = 'ppl'
            dual, dual_poly = polytope.poly_v_to_h(self._ineqs[:,:-1], backend=backend)

            # check for rays
            if 0 in dual[:,-1]:
                raise ValueError("Dual was a polyhedron, not a polytope.")

            # map the ineqs into points
            self._real_vertices = [row[:-1]/row[-1] for row in dual]
            self._real_vertices = np.array(self._real_vertices)
        else:
            # more complicated hyperplanes were input. Use dedicated function
            # (could likely be mapped to poly_v_to_h. Worry about rational inputs)
            self._real_vertices, _ = poly_h_to_v(self._ineqs, verbosity=verbosity)

        # raise error if ineqs are not feasible
        if len(self._real_vertices) == 0:
            raise ValueError("Inequalities are not feasible.")
        elif len(self._real_vertices) == 1:
            raise ValueError("CYTools doesn't support 0D polytopes.")

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
def poly_h_to_v(hypers: "ArrayLike", verbosity: int = 0) -> ("ArrayLike", None):
    """
    **Description:**
    Generate the V-representation of a polytope, given the H-representation.
    I.e., map hyperplanes inequalities to points/vertices.

    The inequalities, c, must organized as a matrix for which each row is an
    inequality of the form
        c[i,0] * x_0 + ... + c[i,d-1] * x_{d-1} + c[i,d] >= 0

    Only works with ppl backend, currently.

    **Arguments:**
    - `hypers`: The hyperplanes inequalities.
    - `verbosity`: The verbosity level.

    **Returns:**
    The associated points of the polytope and the formal convex hull.
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
    for linexp in hypers[:,:-1]@vrs + hypers[:,-1]:
        cs.insert(linexp >= 0)
        #cs.insert(sum(c[i] * vrs[i] for i in range(dim)) + c[-1] >= 0)

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
    """
    **Description:**
    Enumerate all lattice points in a polytope with given vertices and
    hyperplanes inequalities.

    Simpler than the `saturating_lattice_pts` variant in polytope.py because,
    here, we just enumerate the lattice points.

    Also ***non-lattice polytopes are allowed.***

    **Arguments:**
    - `verts`: The vertices of the polytope.
    - `ineqs`: The hyperplanes inequalities. Of the form
        [[c_00, ... , c0(n-1), c0n], ...] which signifies
        c_00 r[0] + ... + c_0(n-1) r[n-1] + c_0n >= 0

    **Returns:**
    The lattice points in the polytope.
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
