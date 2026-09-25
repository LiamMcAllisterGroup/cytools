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
# Description:  This module contains tools designed for Calabi-Yau hypersurface
#               computations.
# -----------------------------------------------------------------------------

r"""Helper functions for Calabi-Yau orientifolds and their F-theory uplifts.

This module collects the lower-level toric geometry routines used by
[`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold] and
[`F_Theory_Uplift`][cytools.f_theory.FT_CY.F_Theory_Uplift]: GLSM charges,
Newton polytopes and sections of toric divisors, Cartier and nef checks, normal
fans and their refinements, $\mathbb{Z}_2$ torus actions, Hodge numbers of
codimension-two complete intersections, and assorted array utilities.

Unless stated otherwise, cones are tuples of one-indexed ray labels, and a toric
divisor $D = \sum_\rho a_\rho D_\rho$ is given by its coefficient vector
$(a_\rho)$ in the basis of toric prime divisors, ordered like the rays.
"""

# 'standard' imports
from itertools import combinations, combinations_with_replacement, permutations, product

# 3rd party imports
import numpy as np
import math
from flint import fmpz_mat
from sympy import Matrix, ZZ, lcm, fraction
from sympy.matrices.normalforms import smith_normal_decomp
from scipy.optimize import nnls
from scipy.optimize import milp, LinearConstraint

# CYTools imports
from cytools import Polytope, h_polytope, Cone
from cytools.utils import integral_nullspace, lll_reduce
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration


def compute_partition(divisors, rays):
    r"""Shift two toric divisors so that they partition the anticanonical divisor.

    Look for characters $m_1, m_2 \in M$ such that the shifted divisors
    $D_i + \operatorname{div}(\chi^{m_i})$, with coefficients
    $a_{i,\rho} + \langle m_i, v_\rho \rangle$, are effective and sum to the
    anticanonical divisor $\sum_\rho D_\rho$. Since the shifted coefficients are
    non-negative integers that add up to one on every ray, the two divisors then
    partition the set of toric prime divisors. The characters are found by solving
    an integer linear program with `scipy.optimize.milp`.

    Args:
        divisors (list[np.ndarray]): The coefficient vectors of the two divisors.
            Exactly two divisors are currently supported.
        rays (np.ndarray): The ray generators $v_\rho$ of the toric fan, one per
            row.

    Returns:
        tuple[bool, np.ndarray | None]: Whether such a shift exists and, if it does,
            an integer array of shape `(2, len(rays))` whose rows are the shifted
            divisor coefficient vectors (otherwise `None`).

    Examples:
        We shift two divisors on the Hirzebruch surface $\mathbb{F}_1$ into a
        partition of the anticanonical divisor.

        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import compute_partition
        >>> rays = np.array([[1, 0], [0, 1], [-1, 1], [0, -1]])
        >>> compute_partition([np.array([2, 0, 0, 0]), np.array([-1, 1, 1, 1])], rays)
        (True, array([[1, 0, 1, 0],
               [0, 1, 0, 1]]))
    """

    linear1 = np.tensordot(np.identity(2, dtype=int), rays, axes=0)
    linear1 = linear1.transpose(0, 2, 1, 3).reshape(
        len(divisors) * rays.shape[0], len(divisors) * rays.shape[1]
    )
    linear2 = np.hstack([rays] * len(divisors))

    affine1 = np.concatenate(divisors)
    affine2 = sum(divisors) - 1

    lower1 = -affine1
    upper1 = np.full_like(affine1, np.inf, dtype=float)
    lower2 = -affine2
    upper2 = -affine2

    linear = np.vstack([linear1, linear2])
    lower = np.concatenate([lower1, lower2])
    upper = np.concatenate([upper1, upper2])

    c = np.zeros(linear.shape[1])
    # noqa below: built but not passed to milp() two lines down, so the solve
    # may be running as an LP where an ILP was intended. Raised with the
    # f_theory authors; kept until they decide.
    integrality = np.ones_like(c)  # noqa: F841

    constraints = LinearConstraint(linear, lower, upper)
    res = milp(
        c=np.zeros(linear.shape[1]),
        constraints=constraints,
        integrality=np.ones(linear.shape[1]),
        bounds=(-np.inf, np.inf),
    )

    if not res.success or res.x is None:
        return (False, None)

    sol = np.rint(
        res.x.reshape(len(divisors), len(rays[0])) @ (rays.T) + np.array(divisors)
    ).astype(int)
    return (True, sol)


def contains_row(arr: np.array, target: np.array):
    """Check whether a vector appears as a row of a 2D array.

    Args:
        arr (np.ndarray): The 2D array to search.
        target (np.ndarray): The 1D array to look for.

    Returns:
        bool: Whether `target` is equal to some row of `arr`.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import contains_row
        >>> contains_row(np.array([[1, 0], [0, 1]]), np.array([0, 1]))
        np.True_
    """
    return np.any(np.all(arr == target, axis=1))


def contains_rows(arr: np.array, targets: np.array):
    """Check whether every row of one 2D array appears as a row of another.

    Args:
        arr (np.ndarray): The 2D array to search.
        targets (np.ndarray): The 2D array whose rows to look for.

    Returns:
        bool: Whether every row of `targets` is equal to some row of `arr`.
    """
    return np.any((targets[:, None, :] == arr[None, :, :]).all(axis=2), axis=1).all()


def get_same_rows(A: np.array, B: np.array):
    """Return the rows of `A` that also appear as rows of `B`.

    The rows are returned in the order in which they appear in `A`. A row of `A` is
    repeated once for each row of `B` that it matches.

    Args:
        A (np.ndarray): The 2D array whose rows are returned.
        B (np.ndarray): The 2D array to compare against.

    Returns:
        np.ndarray: The rows of `A` that appear in `B`.
    """
    return A[np.where((A[:, None, :] == B[None, :, :]).all(axis=2))[0]]


def same_rows(A, B):
    """Check whether two 2D arrays have the same rows, ignoring order.

    Rows are compared with multiplicity, so each distinct row must appear the same
    number of times in both arrays.

    Args:
        A (np.ndarray): The first 2D array.
        B (np.ndarray): The second 2D array.

    Returns:
        bool: Whether `A` and `B` contain the same rows with the same
            multiplicities.
    """
    rowsA, countsA = np.unique(A, axis=0, return_counts=True)
    rowsB, countsB = np.unique(B, axis=0, return_counts=True)
    return np.array_equal(rowsA, rowsB) and np.array_equal(countsA, countsB)


def dual_face_Cayley_polytope(Cdvert: np.array, f):
    """Return the face of the dual Cayley polytope dual to a given face.

    The dual face is the convex hull of the vertices of the dual Cayley polytope
    that pair to zero with every vertex of `f`.

    Args:
        Cdvert (np.ndarray): The vertices of the dual Cayley polytope, one per row.
        f (PolytopeFace): A face of the Cayley polytope.

    Returns:
        Polytope: The dual face.
    """
    return Polytope(Cdvert[np.all(Cdvert @ f.vertices().T == 0, axis=1)])


def h11_2_part(Cay: Polytope, Cayd: Polytope, det=False):
    """Compute $h^{1,1}$ of a Calabi-Yau defined by a two-part nef partition.

    The Calabi-Yau is the complete intersection of two hypersurfaces defined by a
    two-part nef partition, which is encoded by the Cayley polytope `Cay` and its
    dual `Cayd`. The Hodge number is computed with a combinatorial formula that
    counts the lattice points of faces of the two Cayley polytopes and of their
    dilations by a factor of two. Exchanging the two polytopes computes
    $h^{1,1}$ of the mirror instead, which for Calabi-Yau fourfolds is $h^{3,1}$
    of the original Calabi-Yau.

    Args:
        Cay (Polytope): The Cayley polytope.
        Cayd (Polytope): The dual Cayley polytope.
        det (bool): Whether to print the running total after each group of
            contributions.

    Returns:
        int: The Hodge number $h^{1,1}$.
    """
    Cdvert = Cayd.vertices()
    n = Cay.dim() - 1
    h11_ret = len(Cayd.points()) - n - 2

    if det:
        print("Trivial term: ", h11_ret)

    for f in Cayd.faces(n):
        h11_ret = h11_ret - len(Polytope(2 * (f.vertices())).interior_points())

    if det:
        print("After 2*dual facets: ", h11_ret)

    for f in Cayd.faces(n - 1):
        h11_ret = h11_ret + len(f.interior_points())

    if det:
        print("After dual codim-2: ", h11_ret)

    for f in Cay.faces(1):
        k = len(f.interior_points())
        if k > 0:
            h11_ret = h11_ret + (
                k
                * (
                    len(
                        Polytope(
                            2 * (dual_face_Cayley_polytope(Cdvert, f).vertices())
                        ).interior_points()
                    )
                )
            )

    if det:
        print("After 1-face/2*codim-2 dual face: ", h11_ret)

    for f in Cay.faces(2):
        k = len(Polytope(2 * (f.vertices())).interior_points())
        if k > 0:
            h11_ret = h11_ret - (
                k * len(dual_face_Cayley_polytope(Cdvert, f).interior_points())
            )

    if det:
        print("After 2*(2-face)/codim-3 dual face: ", h11_ret)

    for f in Cay.faces(3):
        k = len(Polytope(2 * (f.vertices())).interior_points())
        if k > 0:
            h11_ret = h11_ret + (
                k * len(dual_face_Cayley_polytope(Cdvert, f).interior_points())
            )
        k = len(dual_face_Cayley_polytope(Cdvert, f).interior_points())
        if k > 0:
            for g in f.faces(2):
                h11_ret = h11_ret - (len(g.interior_points()) * k)

    return h11_ret


def h21_2_part(Cay: Polytope, Cayd: Polytope, det=False):
    """Compute $h^{2,1}$ of a fourfold defined by a two-part nef partition.

    The Calabi-Yau fourfold is the complete intersection of two hypersurfaces in a
    six-dimensional toric variety defined by a two-part nef partition, which is
    encoded by the Cayley polytope `Cay` and its dual `Cayd`. The Hodge number is
    computed with a combinatorial formula that counts the lattice points of faces
    of the two Cayley polytopes and of their dilations by a factor of two. The face
    dimensions that enter the formula are specific to this fourfold case.

    Args:
        Cay (Polytope): The Cayley polytope.
        Cayd (Polytope): The dual Cayley polytope.
        det (bool): Whether to print the running total after each group of
            contributions.

    Returns:
        int: The Hodge number $h^{2,1}$.
    """
    Cdvert = Cayd.vertices()
    h21_ret = 0
    for f in Cay.faces(2):
        h21_ret = h21_ret + len(
            Polytope(
                2 * (dual_face_Cayley_polytope(Cdvert, f).vertices())
            ).interior_points()
        ) * len(f.interior_points())
    if det:
        print(h21_ret)
    for f in Cay.faces(4):
        h21_ret = h21_ret + len(Polytope(2 * (f.vertices())).interior_points()) * len(
            dual_face_Cayley_polytope(Cdvert, f).interior_points()
        )
    if det:
        print(h21_ret)
    for f in Cay.faces(3):
        k = len(dual_face_Cayley_polytope(Cdvert, f).interior_points())
        if k > 0:
            for g in f.faces(2):
                h21_ret = h21_ret - len(g.interior_points()) * k
    if det:
        print(h21_ret)
    for f in Cay.faces(4):
        k = len(dual_face_Cayley_polytope(Cdvert, f).interior_points())
        if k > 0:
            for g in f.faces(3):
                h21_ret = h21_ret - len(g.interior_points()) * k
    return h21_ret


def get_indices(arr: np.array, targets: np.array):
    """Return the indices of the rows of `arr` that appear among `targets`.

    Args:
        arr (np.ndarray): The 2D array to search.
        targets (np.ndarray): A single target row (1D array) or several target
            rows (2D array).

    Returns:
        np.ndarray: The sorted indices of the rows of `arr` that are equal to some
            target row.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import get_indices
        >>> arr = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])
        >>> get_indices(arr, np.array([[0, 1], [1, 1]]))
        array([1, 2, 3])
    """
    return np.where(np.any((arr[:, None, :] == targets).all(axis=2), axis=1))[0]


def get_index(arr: np.array, target: np.array):
    """Return the indices of the rows of `arr` equal to `target`.

    Args:
        arr (np.ndarray): The 2D array to search.
        target (np.ndarray): The 1D target row.

    Returns:
        np.ndarray: The sorted indices of the rows of `arr` equal to `target`.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import get_index
        >>> get_index(np.array([[1, 0], [0, 1], [1, 1], [0, 1]]), np.array([0, 1]))
        array([1, 3])
    """
    return np.where(np.all(arr == target, axis=1))[0]


def glsm_from_points(pts):
    r"""Return a GLSM charge matrix for a point configuration.

    The rows of the returned matrix form a $\mathbb{Z}$-basis of the lattice of
    integer linear relations $\sum_i q_i v_i = 0$ among the points $v_i$, computed
    with the Smith normal form. Column $i$ contains the charges of point $i$.

    Args:
        pts (ArrayLike): The points $v_i$, one per row.

    Returns:
        np.ndarray: The GLSM charge matrix, of shape `(n - r, n)`, where `n` is the
            number of points and `r` the rank of the configuration.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import glsm_from_points
        >>> glsm_from_points([[1, 0], [0, 1], [-1, -1]])
        array([[1, 1, 1]])
    """
    a, s, t = smith_normal_decomp(Matrix(np.array(pts).T), domain=ZZ)
    aa = np.array(a, dtype=int)
    tt = np.array(t, dtype=int)
    rank_a = np.linalg.matrix_rank(aa)
    return tt.T[rank_a:]


def points_from_glsm(glsm):
    r"""Return a point configuration with the given GLSM charges.

    This inverts
    [`glsm_from_points`][cytools.f_theory.Uplift_functions.glsm_from_points] up to
    a change of lattice basis: the columns of the returned matrix form a
    $\mathbb{Z}$-basis of the integer kernel of `glsm`, so the rows are points
    whose integer linear relations are generated by the rows of `glsm`.

    Args:
        glsm (ArrayLike): The GLSM charge matrix, with one column per point.

    Returns:
        np.ndarray: The points, one per row, of shape `(n, n - r)`, where `n` is
            the number of columns of `glsm` and `r` its rank.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import points_from_glsm
        >>> points_from_glsm([[1, 1, 1]])
        array([[-1, -1],
               [ 1,  0],
               [ 0,  1]])
    """
    D, U, V = smith_normal_decomp(Matrix(np.array(glsm).T), domain=ZZ)
    DD = np.array(D, dtype=int)
    UU = np.array(U, dtype=int)
    rank_D = np.linalg.matrix_rank(DD)
    return (UU[rank_D:].astype(int)).T


def find_trilayer_vertex_polytope(p, as_index=False):
    """Return the distinguished vertex of a trilayer polytope.

    A polytope is trilayer (see
    [`Polytope.is_trilayer`][cytools.polytope.Polytope.is_trilayer]) if one of its
    vertices has GLSM charge equal to half of the anticanonical class of the vertex
    configuration. This function finds that vertex.

    Args:
        p (Polytope): A trilayer polytope.
        as_index (bool): Whether to return the index of the vertex in
            `p.vertices()` instead of its coordinates.

    Returns:
        np.ndarray | int: The distinguished vertex, or its index if `as_index` is
            true.

    Examples:
        >>> from cytools import Polytope
        >>> from cytools.f_theory.Uplift_functions import find_trilayer_vertex_polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,-4],[0,0,0,1]])
        >>> find_trilayer_vertex_polytope(p)
        array([0, 0, 0, 1])
    """
    glsm_vert = glsm_from_points(p.vertices())
    half_anticanon = np.sum(glsm_vert, axis=1) // 2
    index = get_indices(glsm_vert.T, np.array([half_anticanon]))[0]
    if as_index:
        return index
    else:
        return p.vertices()[index]


def find_trilayer_vertex_vertices(V, as_vertex_index=False):
    """Return the distinguished vertex of a trilayer polytope from its vertices.

    This is the same as `find_trilayer_vertex_polytope`, but takes the vertices of
    the polytope as input.

    Args:
        V (np.ndarray): The vertices of a trilayer polytope, one per row.
        as_vertex_index (bool): Whether to return the index of the vertex in `V`
            instead of its coordinates.

    Returns:
        np.ndarray | int: The distinguished vertex, or its index if
            `as_vertex_index` is true.
    """

    glsm_vert = glsm_from_points(V)
    half_anticanon = np.sum(glsm_vert, axis=1) // 2
    index = get_indices(glsm_vert.T, np.array([half_anticanon]))[0]
    if as_vertex_index:
        return index
    else:
        return V[index]


def trilayer_normal_form(p):
    r"""Bring a trilayer polytope into trilayer normal form.

    Apply a $GL(n, \mathbb{Z})$ transformation that maps the distinguished vertex
    (see `find_trilayer_vertex_polytope`) to $(-1, 0, \ldots, 0)$ and all other
    vertices to points with first coordinate $1$. The lattice points of the
    resulting polytope then lie in the three layers $x_0 \in \{-1, 0, 1\}$.

    Args:
        p (Polytope): A trilayer polytope.

    Returns:
        Polytope: The polytope in trilayer normal form.

    Examples:
        >>> from cytools import Polytope
        >>> from cytools.f_theory.Uplift_functions import trilayer_normal_form
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,-4],[0,0,0,1]])
        >>> trilayer_normal_form(p).vertices()
        array([[-1,  0,  0,  0],
               [ 1, -1, -1, -3],
               [ 1,  0,  0,  1],
               [ 1,  0,  1,  1],
               [ 1,  1,  0,  1]])
    """
    verts = p.vertices()
    index = find_trilayer_vertex_vertices(verts, as_vertex_index=True)
    verts[[0, index]] = verts[[index, 0]]
    aa, ss, tt = smith_normal_decomp(Matrix(verts), domain=ZZ)
    a = np.array(aa, dtype=int)
    s = np.array(ss, dtype=int)
    t = np.array(tt, dtype=int)
    b = np.ones(len(verts), dtype=int)
    b[0] = -1
    c = s @ b
    y = np.zeros(a.shape[1], dtype=int)
    for ii in range(len(y)):
        if a[ii][ii] != 0:
            y[ii] = c[ii] / a[ii][ii]
    r = t @ y
    aa2, ss2, tt2 = smith_normal_decomp(Matrix(r[:, None]), domain=ZZ)
    s2 = np.array(ss2, dtype=int)
    U0 = np.round(np.linalg.inv(s2).T).astype(int)
    M = U0 @ verts.T
    for i in range(1, p.ambient_dim()):
        U0[i] = U0[i] + M[i, 0] * U0[0]
    return Polytope((U0 @ verts.T).T)


def Newton_Polytope(pts, weights):
    r"""Return the Newton polytope of a toric divisor.

    For a divisor $D = \sum_\rho a_\rho D_\rho$, this is the polytope
    $P_D = \{m \in M_\mathbb{R} : \langle m, v_\rho \rangle \geq -a_\rho\}$, whose
    lattice points correspond to the monomial global sections of
    $\mathcal{O}(D)$. It is constructed as an
    [`HPolytope`][cytools.h_polytope.h_polytope.HPolytope], i.e. as the convex hull
    of the lattice points satisfying the inequalities, so it has no lattice points
    if $D$ is not effective.

    Args:
        pts (ArrayLike): The rays $v_\rho$ of the toric fan, one per row.
        weights (ArrayLike): The divisor coefficients $a_\rho$.

    Returns:
        HPolytope: The Newton polytope of the divisor.

    Examples:
        We construct the Newton polytope of the anticanonical divisor of
        $\mathbb{P}^2$.

        >>> from cytools.f_theory.Uplift_functions import Newton_Polytope
        >>> Newton_Polytope([[1, 0], [0, 1], [-1, -1]], [1, 1, 1]).vertices()
        array([[-1, -1],
               [-1,  2],
               [ 2, -1]])
    """
    return h_polytope.HPolytope(np.column_stack([pts, weights]).astype(int))


def row_difference(A: np.array, B: np.array):
    """Return the rows of `A` that do not appear as rows of `B`.

    The rows are returned in the order in which they appear in `A`, including
    repeated rows.

    Args:
        A (np.ndarray): The 2D array whose rows are returned.
        B (np.ndarray): The 2D array of rows to exclude.

    Returns:
        np.ndarray: The rows of `A` that are not rows of `B`.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import row_difference
        >>> row_difference(np.array([[1, 0], [0, 1], [1, 1]]), np.array([[0, 1]]))
        array([[1, 0],
               [1, 1]])
    """
    return A[~np.any((A[:, None, :] == B[None, :, :]).all(axis=2), axis=1)]


def points_not_interior_to_facets_and_codim2_faces(p: Polytope):
    """Return the lattice points not interior to faces of codimension one or two.

    Start from the points not interior to facets (see
    [`Polytope.pts_not_facets`][cytools.polytope.Polytope.pts_not_facets]) and
    remove the points interior to codimension-two faces, keeping the order of the
    remaining points.

    Args:
        p (Polytope): The polytope.

    Returns:
        np.ndarray: The selected lattice points, one per row.
    """
    pts = p.points_not_interior_to_facets()
    for f in p.faces(p.dim() - 2):
        if len(f.interior_points()) > 0:
            pts = np.delete(pts, get_indices(pts, f.interior_points()), axis=0)
    return pts


def get_lower_dimensional_cones(cones, d):
    """Return all faces with a given number of rays of a collection of cones.

    Every `d`-element subset of the rays of each cone is returned once. Since all
    subsets are taken, these are exactly the `d`-ray faces only if the cones are
    simplicial.

    Args:
        cones (Iterable[tuple[int, ...]]): The cones, as tuples of ray labels.
        d (int): The number of rays of the faces.

    Returns:
        list[tuple[int, ...]]: The distinct `d`-ray faces, in arbitrary order.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import get_lower_dimensional_cones
        >>> sorted(get_lower_dimensional_cones([(1, 2, 3), (2, 3, 4)], 2))
        [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    """
    return list({combo for row in cones for combo in combinations(row, d)})


def lattice_refinement(q, denominator=2):
    r"""Return the change of basis to a refined lattice.

    The refined lattice is $N' = \mathbb{Z}^n + \mathbb{Z}\, q/k$, where $k$ is
    `denominator`. The returned integer matrix $\Lambda$ maps a vector of the
    original lattice $\mathbb{Z}^n$ to its coordinates with respect to an
    LLL-reduced basis of $N'$. Its determinant is $\pm[N' : \mathbb{Z}^n]$.

    Args:
        q (ArrayLike): The integer vector defining the refinement.
        denominator (int): The denominator $k$ of the fractional vector $q/k$.

    Returns:
        np.ndarray: The integer matrix $\Lambda$.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import lattice_refinement
        >>> lattice_refinement([1, 1])
        array([[ 1,  1],
               [ 1, -1]])
    """

    lattice_basis = denominator * np.identity(len(q)).astype(int)
    lattice_basis = np.vstack([denominator * np.identity(len(q)).astype(int), [q]])
    A = fmpz_mat(lattice_basis.tolist())
    A_lll = A.lll()
    scaled_up_basis = np.array(A_lll.tolist()).astype(int)
    vanishing_pos = np.where(np.all(scaled_up_basis == 0, axis=1))[0]
    scaled_up_basis = np.delete(scaled_up_basis, vanishing_pos, axis=0)
    Lambda = np.linalg.inv(scaled_up_basis.T)

    return np.rint(Lambda * denominator).astype(int)


def toric_orbifold(pts_CY_ambient, q, denominator=2):
    r"""Return the rays of the quotient of a toric variety by a finite torus action.

    The quotient by the subgroup of the torus generated by $q/k$, where $k$ is
    `denominator`, has the same fan but with respect to the refined lattice
    $N' = \mathbb{Z}^n + \mathbb{Z}\, q/k$. The rays are expressed in a basis of
    $N'$ (see
    [`lattice_refinement`][cytools.f_theory.Uplift_functions.lattice_refinement])
    and then made primitive, so each original ray is $v_\rho = r_\rho u_\rho$ with
    $u_\rho$ primitive in $N'$.

    Args:
        pts_CY_ambient (np.ndarray): The rays $v_\rho$ of the toric fan of the
            Calabi-Yau ambient variety, one per row.
        q (ArrayLike): The integer vector defining the torus action.
        denominator (int): The denominator $k$ of the fractional vector $q/k$.

    Returns:
        tuple[np.ndarray, np.ndarray]: The primitive rays $u_\rho$ of the orbifold,
            one per row, and the rescalings $r_\rho$.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import toric_orbifold
        >>> toric_orbifold([[1, 0], [0, 1], [-1, -1]], [1, 1])
        (array([[ 1,  1],
               [ 1, -1],
               [-1,  0]]), array([1, 1, 2]))
    """

    Lambda = lattice_refinement(q, denominator)
    orbifold_points = pts_CY_ambient @ (Lambda.T)
    rescalings = np.array([math.gcd(*list(i)) for i in orbifold_points])
    orbifold_points = np.rint((orbifold_points.T / rescalings).T).astype(int)
    return (orbifold_points, rescalings)


def O3O7_line_bundle(pts_CY_ambient, q, rescalings):
    r"""Return the orientifold line bundle on the toric orbifold.

    Pick the first lattice point $m$ of the Newton polytope of the anticanonical
    divisor with $\langle m, q \rangle$ odd, i.e. a monomial of the Calabi-Yau
    equation that is odd under the $\mathbb{Z}_2$ action. Its exponents
    $\langle m, v_\rho \rangle + 1$ are divided by the ray rescalings of the
    orbifold (and rounded to the nearest integer) to obtain the divisor
    coefficients on the orbifold.

    Args:
        pts_CY_ambient (np.ndarray): The rays $v_\rho$ of the toric fan of the
            Calabi-Yau ambient variety, one per row.
        q (ArrayLike): The integer vector such that $q/2$ defines the
            $\mathbb{Z}_2$ action.
        rescalings (ArrayLike): The ray rescalings of the orbifold, as returned by
            [`toric_orbifold`][cytools.f_theory.Uplift_functions.toric_orbifold].

    Returns:
        np.ndarray | None: The divisor coefficients of the line bundle on the
            orbifold, or `None` if no monomial is odd under the action.
    """

    CY3_equation_newton_polytope = Newton_Polytope(
        pts_CY_ambient, [1] * len(pts_CY_ambient)
    )
    projected_in_monomial_indices = np.where(
        np.mod(CY3_equation_newton_polytope.points() @ q, 2) == 1
    )[0]
    if len(projected_in_monomial_indices) == 0:
        return None
    arbitrary_monomial_point = CY3_equation_newton_polytope.points()[
        projected_in_monomial_indices[0]
    ]
    line_bundle_weights_CYhypersurface = pts_CY_ambient @ arbitrary_monomial_point + 1
    line_bundle_weights_FtheoryBase = np.rint(
        line_bundle_weights_CYhypersurface / rescalings
    ).astype(int)

    return line_bundle_weights_FtheoryBase


def Z2_fixed_locus(vc_triangulation, q, cone_dimension=None, denominator=2):
    """Return the cones whose toric strata are fixed by a finite torus action.

    A cone is selected if the sum of its generators plus $q$ is divisible by
    `denominator` in every component. If `cone_dimension` is not specified, all
    proper faces of the maximal cones are considered (the maximal cones themselves
    are excluded). Faces are enumerated as subsets of the generators, so the fan is
    assumed to be simplicial.

    Args:
        vc_triangulation (Fan): The toric fan of the ambient variety.
        q (ArrayLike): The integer vector such that $q/k$ generates the action,
            where $k$ is `denominator`.
        cone_dimension (int | None): If specified, only cones with this number of
            rays are considered.
        denominator (int): The denominator $k$ of the fractional vector $q/k$.

    Returns:
        list[tuple[int, ...]]: The fixed cones, as tuples of ray labels.
    """
    if cone_dimension is None:
        all_cones = {
            j
            for c in vc_triangulation.cones()
            for i in range(1, len(c))
            for j in combinations(c, i)
        }
        all_cones = [c for c in all_cones]
    else:
        all_cones = list(
            get_lower_dimensional_cones(vc_triangulation.cones(), cone_dimension)
        )
    fixed_locus_cones = [
        all_cones[i]
        for i in np.where(
            [
                np.all(
                    np.mod(
                        sum(vc_triangulation.vectors()[np.array(c) - 1]) + q,
                        denominator,
                    )
                    == 0
                )
                for c in all_cones
            ]
        )[0]
    ]

    return fixed_locus_cones


def inequivalent_Z2_actions(lattice_symmetries):
    r"""Return the $\mathbb{Z}_2$ torus actions up to lattice symmetries.

    The $\mathbb{Z}_2$ subgroups of the torus are generated by $q/2$ with
    $q \in \{0, 1\}^n$ nonzero. Two such vectors are equivalent if $s q \equiv q'$
    modulo 2 for some symmetry $s$. One representative of each equivalence class
    is returned.

    Args:
        lattice_symmetries (list[np.ndarray]): The lattice symmetry group, as square
            integer matrices acting from the left. It should include the identity,
            e.g. as returned by
            [`Polytope.automorphisms`][cytools.polytope.Polytope.automorphisms] with
            `action="left"`.

    Returns:
        np.ndarray: The representatives $q$, as 0/1 vectors, one per row.

    Examples:
        With only the symmetry that exchanges the two coordinates, $(1, 0)$ and
        $(0, 1)$ are equivalent.

        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import inequivalent_Z2_actions
        >>> inequivalent_Z2_actions([np.eye(2, dtype=int), np.array([[0, 1], [1, 0]])])
        array([[0, 1],
               [1, 1]])
    """

    dim = lattice_symmetries[0].shape[0]
    t_possibilities = {
        t for t0 in combinations_with_replacement([0, 1], dim) for t in permutations(t0)
    }
    t_possibilities = [t for t in t_possibilities]
    t_possibilities = np.delete(
        t_possibilities, t_possibilities.index(tuple([0] * dim)), 0
    )

    inequivalent_t_possibilities = {
        frozenset([tuple(y) for y in x])
        for x in np.transpose(
            np.array([np.mod(s @ (t_possibilities.T), 2) for s in lattice_symmetries]),
            [2, 0, 1],
        )
    }
    inequivalent_t_possibilities = np.array(
        [[y for y in x][0] for x in inequivalent_t_possibilities]
    )

    return inequivalent_t_possibilities


def linebundle_weights_from_Newton_Polytope(vectors, Newton_polytope: Polytope):
    r"""Return the divisor coefficients determined by a Newton polytope.

    The coefficient of each ray is
    $a_\rho = \max_m (-\langle m, v_\rho \rangle)$, with $m$ ranging over the
    lattice points of the polytope. This is the smallest divisor whose Newton
    polytope contains the given one.

    Args:
        vectors (np.ndarray): The rays $v_\rho$ of the toric fan, one per row.
        Newton_polytope (Polytope): The Newton polytope.

    Returns:
        np.ndarray: The divisor coefficients $a_\rho$.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import (
        ...     Newton_Polytope,
        ...     linebundle_weights_from_Newton_Polytope,
        ... )
        >>> rays = np.array([[1, 0], [0, 1], [-1, -1]])
        >>> newton_polytope = Newton_Polytope(rays, [1, 1, 1])
        >>> linebundle_weights_from_Newton_Polytope(rays, newton_polytope)
        array([1, 1, 1])
    """
    return np.max(-(vectors @ Newton_polytope.points().T), axis=1)


def is_Gorenstein(cone):
    """Check whether a cone is Gorenstein.

    A cone is Gorenstein if there is an integral linear functional $n$ that takes
    the value $1$ on the primitive generators of all of its extremal rays. The
    functional is computed from the Smith normal form of the matrix of extremal
    rays.

    Args:
        cone (Cone): The cone to test.

    Returns:
        tuple[bool, np.ndarray | None]: Whether the cone is Gorenstein and, if it
            is, the functional $n$ (otherwise `None`).

    Warning:
        The candidate functional is currently not checked against all of the
        equations, so this function can return `True`, together with an invalid
        functional, for cones that are not Gorenstein (e.g. the cone generated by
        $(1, 0)$ and $(2, 3)$).
    """

    M = cone.extremal_rays()
    ones = np.ones(M.shape[0], dtype=int)
    S, U, V = smith_normal_decomp(Matrix(M, domain=ZZ))
    s = np.array(S, dtype=int)
    u = np.array(U, dtype=int)
    v = np.array(V, dtype=int)
    c = u @ ones
    y = np.zeros(s.shape[1], dtype=int)
    for ii in range(np.min(s.shape)):
        if s[ii][ii] != 0:
            y[ii] = c[ii] / s[ii][ii]
        else:
            return (False, None)
    n = v @ y
    return (True, n)


def is_reflexive_Gorenstein(cone):
    """Check whether a cone and its dual are both Gorenstein.

    See [`is_Gorenstein`][cytools.f_theory.Uplift_functions.is_Gorenstein].

    Args:
        cone (Cone): The cone to test.

    Returns:
        bool: Whether the cone is reflexive Gorenstein.
    """
    if is_Gorenstein(cone)[0]:
        dual_cone = cone.dual()
        if is_Gorenstein(dual_cone)[0]:
            return True
    return False


def Gorenstein_index(cone):
    r"""Return the index of a reflexive Gorenstein cone.

    This is the pairing $\langle m, n \rangle$ between the Gorenstein functional
    $n$ of the cone and the Gorenstein functional $m$ of its dual.

    Args:
        cone (Cone): A reflexive Gorenstein cone.

    Returns:
        int: The Gorenstein index.

    Raises:
        ValueError: If the cone is not reflexive Gorenstein.

    Examples:
        The cone over a reflexive polygon is reflexive Gorenstein of index 1.

        >>> from cytools import Cone
        >>> from cytools.f_theory.Uplift_functions import Gorenstein_index
        >>> Gorenstein_index(Cone([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]))
        np.int64(1)
    """
    if is_reflexive_Gorenstein(cone):
        dual_cone = cone.dual()
        return is_Gorenstein(cone)[1] @ is_Gorenstein(dual_cone)[1]
    raise ValueError("Cone is not reflexive Gorenstein")


def Cartier_index(toric_fan, weights):
    r"""Return the Cartier index of a toric Weil divisor.

    For each maximal cone $\sigma$, solve for the local data $m_\sigma$ with
    $\langle m_\sigma, v_\rho \rangle = -a_\rho$ for all rays $\rho \in \sigma$
    (by least squares, in floating point). If there is no solution for some cone,
    the divisor is not $\mathbb{Q}$-Cartier. Otherwise, the Cartier index is the
    smallest positive integer $k$ such that all $k\, m_\sigma$ are integral.

    Args:
        toric_fan (Fan): The toric fan.
        weights (ArrayLike): The divisor coefficients $a_\rho$.

    Returns:
        int | None: The Cartier index, or `None` if the divisor is not
            $\mathbb{Q}$-Cartier.

    Examples:
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import Cartier_index
        >>> fan = VectorConfiguration([[1, 0], [1, 2]]).triangulate()
        >>> Cartier_index(fan, [1, 0])
        np.int64(2)
    """

    weights = np.array(weights)
    index_data = []
    for c in toric_fan.cones():
        arr = toric_fan.vectors(c)
        cone_gen_weights = weights[np.array(c) - 1]
        least_sq = np.linalg.lstsq(arr, cone_gen_weights)
        y = -least_sq[0]
        res = sum(least_sq[1])
        if res > 1e-10:
            return None
        cone_index = 1
        while not np.all(np.isclose(cone_index * y, np.round(cone_index * y))):
            cone_index += 1
        index_data.append(cone_index)
    return np.lcm.reduce(np.array(index_data))


def is_Cartier(toric_fan, weights, return_Q_Cartier_data=False, decimals=10):
    r"""Check whether a toric Weil divisor is Cartier.

    For each maximal cone $\sigma$, solve for the local data $m_\sigma$ with
    $\langle m_\sigma, v_\rho \rangle = -a_\rho$ for all rays $\rho \in \sigma$
    (by least squares, in floating point). The divisor is Cartier if all of these
    exist and are integral.

    Args:
        toric_fan (Fan): The toric fan.
        weights (ArrayLike): The divisor coefficients $a_\rho$.
        return_Q_Cartier_data (bool): Whether to keep going and return the local
            data if the divisor is not Cartier.
        decimals (int): The number of decimals to which non-integral local data is
            rounded.

    Returns:
        tuple[bool, list | None]: Whether the divisor is Cartier, and a list with
            the local data $m_\sigma$ for each maximal cone, in the order of
            `toric_fan.cones()`. If the divisor is not Cartier and
            `return_Q_Cartier_data` is false, the list is `None`. Otherwise, each
            entry is an integer array if $m_\sigma$ is integral, a rounded float
            array if it is non-integral, and `None` if it does not exist.

    Examples:
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import is_Cartier
        >>> fan = VectorConfiguration([[1, 0], [1, 2]]).triangulate()
        >>> is_Cartier(fan, [1, 0])
        (False, None)
        >>> is_Cartier(fan, [1, 0], return_Q_Cartier_data=True)
        (False, [array([-1. ,  0.5])])
    """
    weights = np.array(weights)
    cartier_data = []
    is_cartier = True
    for c in toric_fan.cones():
        arr = toric_fan.vectors(c)
        cone_gen_weights = weights[np.array(c) - 1]
        least_sq = np.linalg.lstsq(arr, cone_gen_weights)
        y = -least_sq[0]
        res = sum(least_sq[1])
        if np.all(np.isclose(y, np.round(y))) and res < 1e-10:
            cartier_data.append(np.round(y).astype(int))
        else:
            is_cartier = False
            if return_Q_Cartier_data:
                if res < 1e-10:
                    cartier_data.append(np.round(y, decimals=decimals))
                else:
                    cartier_data.append(None)
            else:
                return (False, None)

    return (is_cartier, cartier_data)


def is_nef(toric_fan, weights):
    r"""Check whether a toric divisor is nef.

    The divisor is nef if its coefficient vector lies in the closure of the
    secondary cone of the fan, i.e. if it pairs non-negatively with all of the
    secondary cone hyperplanes.

    Args:
        toric_fan (Fan): The toric fan.
        weights (ArrayLike): The divisor coefficients.

    Returns:
        bool: Whether the divisor is nef.

    Examples:
        On the Hirzebruch surface $\mathbb{F}_1$, the fiber is nef while the
        exceptional curve is not.

        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import is_nef
        >>> fan = VectorConfiguration([[1, 0], [0, 1], [-1, 1], [0, -1]]).triangulate()
        >>> is_nef(fan, [1, 0, 0, 0]), is_nef(fan, [0, 1, 0, 0])
        (np.True_, np.False_)
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes()) @ weights >= 0)


def is_ample(toric_fan, weights):
    """Check whether a toric divisor is ample.

    The divisor is ample if its coefficient vector lies in the interior of the
    secondary cone of the fan, i.e. if it pairs positively with all of the
    secondary cone hyperplanes.

    Args:
        toric_fan (Fan): The toric fan.
        weights (ArrayLike): The divisor coefficients.

    Returns:
        bool: Whether the divisor is ample.
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes()) @ weights > 0)


def is_effective(points, weights):
    """Check whether a toric divisor has a nonzero global section.

    This is the case if its Newton polytope (see
    [`Newton_Polytope`][cytools.f_theory.Uplift_functions.Newton_Polytope])
    contains a lattice point, i.e. if the divisor is linearly equivalent to an
    effective torus-invariant divisor.

    Args:
        points (np.ndarray): The rays of the toric fan, one per row.
        weights (ArrayLike): The divisor coefficients.

    Returns:
        bool: Whether the divisor has a nonzero global section.
    """

    try:
        NP = Newton_Polytope(points, weights)
        if len(NP.points()) > 0:
            return True
        return False
    except ValueError:
        return False


def moving_cone(toric_variety):
    r"""Return the moving cone of a toric variety.

    The moving cone is the intersection, over all rays $\rho$, of the cones
    generated by the classes of the toric prime divisors other than $D_\rho$. The
    returned cone lives in the space $\mathbb{R}^n$ of divisor coefficients: it is
    the preimage of the moving cone under the map to the divisor class group, so
    its lineality space contains the principal divisors.

    Args:
        toric_variety (Fan): The toric fan. Only its rays are used.

    Returns:
        Cone: The moving cone, in the space of divisor coefficients.
    """

    rays = toric_variety.vectors()
    glsm = np.array(integral_nullspace(np.asarray(rays.T, dtype=int))).T
    h_planes = (
        np.array(
            [
                h
                for i in range(len(rays))
                for h in Cone(np.delete(glsm.T, i, 0)).hyperplanes()
            ]
        )
        @ glsm
    )
    mov = Cone(hyperplanes=h_planes)

    return mov


def generic_section_factorizes(points, linebundle_weights):
    r"""Check whether the generic section of a toric divisor factorizes.

    Compute the exponents of the toric coordinates $x_\rho$ in all monomial
    sections of the divisor, and return `True` if some coordinate has exponent zero
    in every section, or if the divisor has no sections at all. Otherwise, return
    `False`.

    Args:
        points (np.ndarray): The rays of the toric fan, one per row.
        linebundle_weights (np.ndarray): The divisor coefficients.

    Returns:
        bool: Whether the generic section factorizes, according to the criterion
            above.
    """

    try:
        NP = Newton_Polytope(points, linebundle_weights)
    except ValueError:
        return True
    return ~np.all(
        np.any(points @ NP.points().T + linebundle_weights[:, None] != 0, axis=1)
    )


def attempt_to_make_nef(toric_variety, line_bundle, epsilon=1e-5):
    r"""Attempt to find a triangulation on which a divisor is nef.

    The vector configuration of `toric_variety` is triangulated with heights
    $h = a/\epsilon + h_0 + \epsilon\,(1 + \sin(i + 1))$, where $a$ are the divisor
    coefficients, $h_0$ the heights of `toric_variety`, and $i$ the index of the
    vector. The divisor is nef on the regular subdivision induced by the heights
    $a$ and on its refinements, and for small enough $\epsilon$ the other two
    terms only refine this subdivision to a triangulation. The result is not
    checked.

    Args:
        toric_variety (Fan): A regular fan. Its heights and its vector
            configuration are used.
        line_bundle (ArrayLike): The divisor coefficients $a$, one per vector of
            the vector configuration.
        epsilon (float): The size $\epsilon$ of the perturbation.

    Returns:
        Fan: The triangulation of the vector configuration. Not all of the vectors
            are necessarily used.
    """

    line_bundle = np.array(line_bundle)
    hts0 = toric_variety.heights() + epsilon * np.array(
        [(np.sin(i + 1) + 1) for i in range(len(line_bundle))]
    )
    hts1 = line_bundle

    return toric_variety.vc.triangulate(heights=hts1 / epsilon + hts0)


def basis(points):
    """Return the indices of rows that form a basis of the row span.

    The rows are scanned in order, and each row that is linearly independent of the
    previously selected ones is added, until the rank of the matrix is reached.

    Args:
        points (np.ndarray): The 2D array whose rows are considered.

    Returns:
        list[int]: The indices of the selected rows.

    Raises:
        ValueError: If no basis could be found (e.g. if `points` has no rows).

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import basis
        >>> basis(np.array([[1, 0], [2, 0], [0, 1]]))
        [0, 2]
    """

    n = points.shape[0]
    d = np.linalg.matrix_rank(points)
    basis_indices = []
    for i in range(n):
        test_indices = basis_indices + [i]
        if np.linalg.matrix_rank(points[test_indices]) == len(test_indices):
            basis_indices.append(i)
        if len(basis_indices) == d:
            return basis_indices
    raise ValueError("No basis could be found")


def sums_to_anticanonical(pts, L1, L2):
    r"""Check whether the sum of two toric divisors is linearly equivalent to $-K$.

    Look for a character $m$ such that $L_1 + L_2 + \operatorname{div}(\chi^m)$ is
    the anticanonical divisor $\sum_\rho D_\rho$, i.e.
    $\langle m, v_\rho \rangle = 1 - L_{1,\rho} - L_{2,\rho}$ for all rays. The
    system is solved by least squares, and the rounded solution is then checked
    exactly.

    Args:
        pts (np.ndarray): The rays $v_\rho$ of the toric fan, one per row.
        L1 (ArrayLike): The coefficients of the first divisor.
        L2 (ArrayLike): The coefficients of the second divisor.

    Returns:
        tuple[bool, np.ndarray | None]: Whether the sum is linearly equivalent to
            the anticanonical divisor and, if it is, the character $m$ (otherwise
            `None`).

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import sums_to_anticanonical
        >>> rays = np.array([[1, 0], [0, 1], [-1, 1], [0, -1]])
        >>> sums_to_anticanonical(rays, [0, 1, 1, 0], [0, 0, 1, 1])
        (True, array([1, 0]))
    """
    pts_float = np.array(pts, dtype=float)
    b_float = (1 - np.array(L1) - np.array(L2)).astype(float)

    try:
        x_float, residuals, rank, s = np.linalg.lstsq(pts_float, b_float, rcond=None)

        x_int = np.round(x_float).astype(int)

        pts_exact = np.array(pts, dtype=object)
        b_exact = 1 - np.array(L1) - np.array(L2)

        if np.array_equal(pts_exact @ x_int, b_exact):
            return True, x_int

    except np.linalg.LinAlgError:
        pass

    return False, None


def is_partition(points, L1, L2):
    r"""Check whether two toric divisors can be shifted to a partition of $-K$.

    Look for characters $m_1, m_2$ such that the shifted divisors
    $L_i + \operatorname{div}(\chi^{m_i})$, with coefficients
    $L_{i,\rho} + \langle m_i, v_\rho \rangle$, have all coefficients in
    $\{0, 1\}$ and sum to the anticanonical divisor $\sum_\rho D_\rho$. First, a
    character $m$ with $L_1 + L_2 + \operatorname{div}(\chi^m) = -K$ is found with
    [`sums_to_anticanonical`][cytools.f_theory.Uplift_functions.sums_to_anticanonical].
    Then $m_2$ is searched for by prescribing each of the two allowed values of
    $\langle m_2, v_\rho \rangle$ on a basis of rays (see
    [`basis`][cytools.f_theory.Uplift_functions.basis]), and $m_1 = m - m_2$.

    Args:
        points (np.ndarray): The rays $v_\rho$ of the toric fan, one per row.
        L1 (np.ndarray): The coefficients of the first divisor.
        L2 (np.ndarray): The coefficients of the second divisor.

    Returns:
        tuple[bool, bool, np.ndarray, np.ndarray]: A tuple
            `(is_partition, sums_to_anticanonical, m1, m2)`. If the sum is not
            linearly equivalent to $-K$, `m1` and `m2` are zero. If it is but no
            partition is found, `m1` is zero and `m2` is $m$.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import is_partition
        >>> rays = np.array([[1, 0], [0, 1], [-1, 1], [0, -1]])
        >>> is_partition(rays, np.array([2, 0, 0, 0]), np.array([-1, 1, 1, 1]))
        (True, True, array([-1,  0]), array([1, 0]))
    """

    sta = sums_to_anticanonical(points, L1, L2)
    if not sta[0]:
        return (
            False,
            False,
            np.zeros(points.shape[1], dtype=int),
            np.zeros(points.shape[1], dtype=int),
        )

    basis_indices = basis(points)

    basis_vectors = points[basis_indices]
    basis_L2 = L2[basis_indices]
    possible_targets = [[-w, 1 - w] for w in basis_L2]
    if len(basis_indices) == points.shape[1]:
        for target_comb in product(*possible_targets):
            target_vec = np.array(target_comb)

            m2_float = np.linalg.solve(basis_vectors, target_vec)

            m2_int = np.round(m2_float).astype(int)
            if not np.allclose(m2_float, m2_int):
                continue

            dot_products = points @ m2_int

            valid_lower = dot_products == -L2
            valid_upper = dot_products == 1 - L2

            if np.all(valid_lower | valid_upper):
                return (True, True, sta[1] - m2_int, m2_int)
    else:
        basis_vecs_pseudo = basis_vectors @ basis_vectors.T
        for target_comb in product(*possible_targets):
            target_vec = np.array(target_comb)

            m2_float = basis_vectors.T @ np.linalg.solve(basis_vecs_pseudo, target_vec)

            m2_int = np.round(m2_float).astype(int)
            if not np.allclose(m2_float, m2_int):
                continue

            dot_products = points @ m2_int

            valid_lower = dot_products == -L2
            valid_upper = dot_products == 1 - L2

            if np.all(valid_lower | valid_upper):
                return (True, True, sta[1] - m2_int, m2_int)

    return (False, True, np.zeros(points.shape[1], dtype=int), sta[1])


def attempt_to_make_Cartier(tri, D):
    """Attempt to refine a fan so that a divisor becomes Cartier and nef.

    If the divisor is already Cartier, the inputs are returned unchanged.
    Otherwise, the facet normals of its Newton polytope that are not yet rays of
    the fan are added as new vectors, with coefficients given by the constant terms
    of the corresponding facet inequalities (so that the Newton polytope does not
    change). The enlarged vector configuration is triangulated, and the result is
    passed to
    [`attempt_to_make_nef`][cytools.f_theory.Uplift_functions.attempt_to_make_nef].
    The result is not checked.

    Args:
        tri (Fan): The toric fan.
        D (ArrayLike): The divisor coefficients.

    Returns:
        tuple[Fan, np.ndarray]: The new fan and the divisor coefficients on it, with
            the coefficients of the new vectors appended after the original ones.
    """
    if is_Cartier(tri, D)[0]:
        return (tri, D)
    else:
        NP = Newton_Polytope(tri.vectors(), D)
        inequalities = NP.inequalities()
        new_points = row_difference(inequalities[:, :-1], tri.vectors())
        indices_new_points = get_indices(inequalities[:, :-1], new_points)
        new_vectors = np.concatenate((tri.vectors(), new_points), axis=0)
        new_vc = VectorConfiguration(new_vectors)
        new_D = np.concatenate((D, inequalities[:, -1][indices_new_points]))
        tri_Cartier = attempt_to_make_nef(new_vc.triangulate(), new_D)
        return (tri_Cartier, new_D)


def BL(fan, lb):
    r"""Return the base locus of a toric divisor.

    This computes the sections of the divisor with
    [`sections`][cytools.f_theory.Uplift_functions.sections] and passes them to
    [`base_locus`][cytools.f_theory.Uplift_functions.base_locus], restricting to
    the cones of `fan`. The divisor must have at least one section.

    Args:
        fan (Fan): The toric fan.
        lb (np.ndarray): The divisor coefficients.

    Returns:
        list[tuple[int, ...]]: The minimal cones $\sigma$ of the fan such that all
            sections vanish on the orbit closure $V(\sigma)$. The base locus is the
            union of these orbit closures.

    Examples:
        On the Hirzebruch surface $\mathbb{F}_1$, the exceptional curve $D_2$ is
        its own base locus, while the fiber $D_1$ is base point free.

        >>> import numpy as np
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import BL
        >>> fan = VectorConfiguration([[1, 0], [0, 1], [-1, 1], [0, -1]]).triangulate()
        >>> BL(fan, np.array([0, 1, 0, 0])), BL(fan, np.array([1, 0, 0, 0]))
        ([(2,)], [])
    """
    return base_locus(sections(fan.vectors(), lb), cones=fan.cones())


def base_locus(sections, cones=None, dim=4):
    r"""Return the minimal sets of toric coordinates on which all sections vanish.

    All sections vanish on the locus $\{x_\rho = 0 : \rho \in S\}$ if every section
    contains some $x_\rho$ with $\rho \in S$ with a positive exponent. The minimal
    such sets $S$ are found by checking candidate sets of increasing size and
    skipping supersets of sets already found.

    Args:
        sections (np.ndarray): The exponent matrix of the sections, with one row
            per coordinate and one column per section, as returned by
            [`sections`][cytools.f_theory.Uplift_functions.sections].
        cones (list[tuple[int, ...]] | None): The maximal cones of a simplicial
            fan. If specified, only faces of these cones are considered as
            candidate sets. If not specified, all sets of at most `dim` coordinates
            are considered.
        dim (int): The maximum number of coordinates in a candidate set when
            `cones` is not specified.

    Returns:
        list[tuple[int, ...]]: The minimal sets, as tuples of one-indexed ray
            labels, in order of increasing size.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import base_locus
        >>> base_locus(np.array([[1, 1], [1, 0], [0, 1]]))
        [(1,), (2, 3)]
    """
    num_coords, num_sections = sections.shape
    B = sections > 0
    minimal_hitting_sets = []
    if cones is None:
        for codim in range(1, dim + 1):
            for combo in combinations(range(num_coords), codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[list(combo), :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)
        return [tuple(x + 1 for x in mhs) for mhs in minimal_hitting_sets]
    else:
        for codim in range(1, len(cones[0]) + 1):
            for combo in get_lower_dimensional_cones(cones, codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[np.array(combo) - 1, :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)
        return [mhs for mhs in minimal_hitting_sets]


def normal_fan(
    polytopes,
    inequalities=None,
    maximal_refinement=False,
    triangulate_refinement=False,
    return_unrefined_fan=False,
):
    r"""Construct the normal fan of a polytope or of a Minkowski sum of polytopes.

    The rays of the normal fan are the inward facet normals $u$ of the polytope
    $P$ (or of the Minkowski sum $P = P_0 + P_1 + \cdots$ if a list of polytopes
    is given), and there is one maximal cone for each vertex of $P$, generated by
    the normals of the facets containing it. The weight of a ray $u$ for a
    polytope $P_j$ is $w_j(u) = -\min_{m \in P_j} \langle m, u \rangle$, so that
    $P_j$ is the Newton polytope of the divisor $\sum_u w_j(u) D_u$ on the toric
    variety of the normal fan.

    If `maximal_refinement` is true, the normal fan of a list of two polytopes is
    also refined using `inequalities` $= (a_0, a_1, c)$. If some ray of the normal
    fan does not satisfy $0 \leq a_0 (1 - w_0(u)) - w_1(u) < c$, only `None`
    values are returned. Otherwise, the primitive vectors along all lattice points
    $u \neq 0$ of the support of the normal fan with
    $a_0 w_0(u) + a_1 w_1(u) \leq c$ are added as new rays.

    Args:
        polytopes (Polytope | list[Polytope]): A lattice polytope, or a list of
            lattice polytopes whose Minkowski sum is used.
        inequalities (ArrayLike | None): The coefficients $(a_0, a_1, c)$ that
            define the refinement. Required if `maximal_refinement` is true.
        maximal_refinement (bool): Whether to construct the refinement. This
            requires `polytopes` to be a list of two polytopes.
        triangulate_refinement (bool): Whether to return the refined fan, obtained
            by star subdividing a simplicial refinement of the normal fan at the
            new rays, instead of only its rays. Only used if `maximal_refinement`
            is true.
        return_unrefined_fan (bool): Whether to also return the normal fan. Only
            used if `maximal_refinement` is true.

    Returns:
        tuple: If `maximal_refinement` is false, a tuple `(fan, weights, cones)`
            with the normal fan, the ray weights, and its maximal cones as lists of
            one-indexed ray labels. The weights form a 1D array for a single
            polytope, and an array of shape `(num_rays, len(polytopes))` for a
            list. If `maximal_refinement` is true, a tuple `(rays, weights)` with
            the rays of the refinement (the rays of the normal fan first) and
            their weights. If `triangulate_refinement` is true, `rays` is replaced
            by the refined fan, and if `return_unrefined_fan` is true, the normal
            fan is appended to the tuple. If the check above fails, all entries are
            `None`.

    Raises:
        Exception: If `maximal_refinement` is true but `inequalities` is not
            specified.

    Examples:
        We construct the normal fan of the unit square, whose Newton polytope
        weights are those of the divisor $D_3 + D_4$.

        >>> from cytools import Polytope
        >>> from cytools.f_theory.Uplift_functions import normal_fan
        >>> fan, weights, cones = normal_fan(Polytope([[0, 0], [1, 0], [0, 1], [1, 1]]))
        >>> fan.vectors()
        array([[ 1,  0],
               [ 0,  1],
               [-1,  0],
               [ 0, -1]])
        >>> weights
        array([0, 0, 1, 1])
    """

    if isinstance(polytopes, list):
        msum_vertices = nested_sum([p.vertices() for p in polytopes])
        p = Polytope(np.unique(flatten(msum_vertices, len(polytopes) - 1), axis=0))
        vertex_split = np.array(
            [
                np.array(np.where(np.all(np.array(msum_vertices) - v == 0, axis=-1))).T[
                    0
                ]
                for v in p.vertices()
            ]
        )
    else:
        p = polytopes
        weights = p.inequalities().T[-1]

    hyperplane_saturations = [
        p.inequalities()[np.where(x == 0)[0]]
        for x in (
            np.vstack([p.vertices().T, [1] * len(p.vertices())]).T
            @ (p.inequalities().T)
        )
    ]
    normal_fan_edges = np.delete(p.inequalities().T, -1, 0).T
    normal_fan_vc = VectorConfiguration(normal_fan_edges)
    cones = [
        [
            int(np.where(np.all(normal_fan_edges - x == 0, axis=1))[0][0]) + 1
            for x in np.delete(s.T, -1, 0).T
        ]
        for s in hyperplane_saturations
    ]
    n_fan = Fan(vc=normal_fan_vc, cones=cones)

    if isinstance(polytopes, list):
        vertices_to_vertices_map = [
            vertex_split[np.where([i in c for c in cones])[0][0]]
            for i in range(1, len(normal_fan_edges) + 1)
        ]
        weights = np.array(
            [
                -np.array([polytopes[j].vertices()[x] for j, x in enumerate(pointers)])
                @ (normal_fan_edges[i])
                for i, pointers in enumerate(vertices_to_vertices_map)
            ]
        )

    if not maximal_refinement:
        return (normal_fan_vc.triangulate(cells=cones), weights, cones)

    if inequalities is None:
        raise Exception("Inequalities must be given to construct maximal refinement")

    inequalities = np.array(inequalities)

    if (
        np.max((1 - weights[:, 0]) * inequalities[0] - weights[:, 1])
        >= inequalities[-1]
    ):
        if return_unrefined_fan:
            return (None, None, None)
        else:
            return (None, None)
    if np.min((1 - weights[:, 0]) * inequalities[0] - weights[:, 1]) < 0:
        if return_unrefined_fan:
            return (None, None, None)
        else:
            return (None, None)

    maximal_blow_ups = [
        h_polytope.HPolytope(
            np.vstack(
                [
                    [
                        np.concatenate(
                            [
                                np.delete(inequalities, -1, 0)
                                @ np.array(
                                    [
                                        polytopes[j].vertices()[x]
                                        for j, x in enumerate(vertex_split[i])
                                    ]
                                ),
                                [inequalities[-1]],
                            ]
                        )
                    ],
                    np.vstack([(p.vertices() - m).T, [0] * len(p.vertices())]).T,
                ]
            )
        ).points()
        for i, m in enumerate(p.vertices())
    ]

    maximal_blow_ups = [
        np.unique(
            [
                np.rint(x / np.gcd.reduce(x)).astype(int)
                for x in np.delete(b, np.where(np.all(b == 0, axis=1))[0][0], 0)
            ],
            axis=0,
        )
        for b in maximal_blow_ups
    ]

    all_vectors = np.unique(np.array([y for x in maximal_blow_ups for y in x]), axis=0)
    all_weights = np.array(
        [
            -np.array(
                [
                    (
                        pol.vertices()[
                            vertex_split[
                                np.where(
                                    [
                                        np.any(np.all(y - x == 0, axis=-1))
                                        for y in maximal_blow_ups
                                    ]
                                )[0][0]
                            ][j]
                        ]
                    )
                    @ x
                    for x in all_vectors
                ]
            )
            for j, pol in enumerate(polytopes)
        ]
    )

    old_indices = np.where(
        [n_fan.vc.vectors_to_labels(v) is not None for v in all_vectors]
    )[0]
    blow_up_weights = np.delete(all_weights.T, old_indices, 0)
    blow_up_vectors = np.delete(all_vectors, old_indices, 0)

    all_vectors = np.vstack([n_fan.vectors(), blow_up_vectors])
    all_weights = np.vstack([weights, blow_up_weights])

    if not triangulate_refinement:
        if return_unrefined_fan:
            return (all_vectors, all_weights, n_fan)
        else:
            return (all_vectors, all_weights)

    if return_unrefined_fan:
        return (refine_fan(make_simplicial(n_fan), all_vectors), all_weights, n_fan)
    else:
        return (refine_fan(make_simplicial(n_fan), all_vectors), all_weights)


def nested_sum(lists, depth=0, acc=0):
    """Return all sums of one element from each list, as a nested list.

    The entry at position `[i_0][i_1]...` of the result is
    `acc + lists[depth][i_0] + lists[depth + 1][i_1] + ...`.

    Args:
        lists (list): The lists (or arrays) whose elements are summed.
        depth (int): The index of the first list to use. This is used for the
            recursion.
        acc (Any): The partial sum accumulated so far. This is used for the recursion.

    Returns:
        list: The sums, as a list nested `len(lists) - depth` levels deep. If
            `depth` equals `len(lists)`, `acc` itself is returned.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import nested_sum
        >>> nested_sum([[1, 2], [10, 20]])
        [[11, 21], [12, 22]]
    """

    if depth == len(lists):
        return acc
    return [nested_sum(lists, depth + 1, acc + x) for x in lists[depth]]


def flatten(lst, depth):
    """Flatten a nested list by a given number of levels.

    Only elements of type `list` are flattened; other elements, such as tuples or
    arrays, are kept as they are.

    Args:
        lst (list): The nested list.
        depth (int): The number of levels to flatten.

    Returns:
        list: The flattened list.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import flatten
        >>> flatten([[1, [2]], [3]], 1)
        [1, [2], 3]
    """

    if depth == 0:
        return lst
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(flatten(x, depth - 1))
        else:
            result.append(x)
    return result


def O7_cones(vc_orbifold, O7_labels, d):
    """Return the cones with a given number of rays whose rays are all O7-planes.

    Args:
        vc_orbifold (Fan): The toric fan of the orbifold. It is assumed to be
            simplicial.
        O7_labels (Iterable[int]): The labels of the rays whose divisors are
            O7-planes.
        d (int): The number of rays of the cones.

    Returns:
        list[tuple[int, ...]]: The `d`-ray cones of the fan whose rays are all in
            `O7_labels`, in arbitrary order.
    """
    d_cones = get_lower_dimensional_cones(vc_orbifold.cones(), d)
    relevant_d_cones = [t for t in d_cones if set(t).issubset(O7_labels)]
    return relevant_d_cones


def basis_H2_toric_fan(toric_fan):
    r"""Return ray labels whose divisor classes form a basis of the class group.

    Find the first smooth maximal cone $\sigma$ of the fan. Since its generators
    form a basis of the lattice, every divisor is linearly equivalent to a unique
    integral combination of the toric prime divisors $D_\rho$ with
    $\rho \notin \sigma$, so their classes form a $\mathbb{Z}$-basis of the divisor
    class group. The rays are assumed to be labeled $1, \ldots, n$.

    Args:
        toric_fan (Fan): The toric fan.

    Returns:
        np.ndarray: The labels of the rays not in $\sigma$, in increasing order.

    Raises:
        ValueError: If the fan has no smooth maximal cone.

    Examples:
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import basis_H2_toric_fan
        >>> fan = VectorConfiguration([[1, 0], [0, 1], [-1, 1], [0, -1]]).triangulate()
        >>> basis_H2_toric_fan(fan)
        array([3, 4])
    """
    for c in toric_fan.cones():
        if Cone(toric_fan.vectors(c)).is_smooth():
            mask = np.ones(len(toric_fan.vectors()), dtype=bool)
            mask[np.array(c) - 1] = False
            basis = np.arange(1, len(mask) + 1)[mask]
            return basis
    raise ValueError("No basis could be found")


def trilayer_5d_Ftheory_uplift(p, verbosity=1):
    r"""Construct the polytope of the F-theory uplift of a trilayer orientifold.

    The polytope is first brought to trilayer normal form (see
    [`trilayer_normal_form`][cytools.f_theory.Uplift_functions.trilayer_normal_form]).
    The lattice points of its top layer (first coordinate $1$), with the first
    coordinate dropped, are the rays of the toric base $B$. The exceptional
    divisors corresponding to the points of the middle layer (first coordinate
    $0$) that are not interior to facets are blown down.

    The rays of the base are then taken to be the primitive vectors along the
    lattice points $u$ of the normal fan of the Newton polytope of $-2K_B$ with
    weight $1 \leq w(u) \leq 2$ (see
    [`normal_fan`][cytools.f_theory.Uplift_functions.normal_fan]). The O7-plane
    divisors are the base divisors on which all sections of $-2K_B$ vanish, with
    some section vanishing to first order. The uplift polytope is the convex hull
    of the base rays lifted to $(u, 0, 1)$, the rays $v_x = (0, 3, 1)$,
    $v_y = (0, -2, -1)$, and $v_z = (0, 0, 1)$ of the $\mathbb{P}_{2,3,1}$ fiber
    (where $0$ denotes the origin of the base lattice),
    and the rays $\hat{u} + v_x + 2 v_y$ and $2 \hat{u} + 2 v_x + 3 v_y$ for each
    lifted O7-plane ray $\hat{u}$, which resolve the $D_4$ singularities over the
    O7-planes.

    Args:
        p (Polytope): A reflexive trilayer polytope.
        verbosity (int): The verbosity level. If positive, the number of blown-down
            exceptional divisors is printed.

    Returns:
        Polytope: The uplift polytope, of dimension `p.dim() + 1` (five for a
            four-dimensional `p`).

    Raises:
        Exception: If `p` is not trilayer.
    """

    if not p.is_trilayer():
        raise Exception("Polytope is not trilayer.")

    p = trilayer_normal_form(p)
    mid_layer_points = np.where(p.points().T[0] == 0)[0]
    mid_layer_points = np.array(
        [
            i
            for i in set(mid_layer_points).intersection(
                set(p.points_to_indices(p.points_not_interior_to_facets()))
            )
        ]
    )
    if len(mid_layer_points) > 1 and verbosity > 0:
        print(f"{len(mid_layer_points) - 1} exceptional divisors have been blown down.")

    p_dim = p.dim()

    p3 = Polytope(np.delete(p.points()[np.where(p.points().T[0] == 1)[0]].T, 0, 0).T)

    p2KB = Newton_Polytope(p3.points(), len(p3.points()) * [2])

    n_fan, wts, cns = normal_fan(p2KB)

    cone_hyperplanes = [Cone(n_fan.vectors(c)).dual().rays() for c in cns]
    blown_up = [
        h_polytope.HPolytope(
            np.vstack(
                [
                    np.vstack([h.T, [0] * len(h)]).T,
                    [
                        np.concatenate([p2KB.vertices()[i], [2]]),
                        np.concatenate([-p2KB.vertices()[i], [-1]]),
                    ],
                ]
            )
        ).points()
        for i, h in enumerate(cone_hyperplanes)
    ]
    blown_up = [
        [np.rint(x / np.gcd.reduce(x)).astype(int) for x in b] for b in blown_up
    ]
    blown_up = [np.unique(b, axis=0) for b in blown_up]
    all_vecs = np.unique([i for b in blown_up for i in b], axis=0)
    full_vc = VectorConfiguration(all_vecs)

    monomials = Newton_Polytope(
        full_vc.vectors(), [2] * len(full_vc.vectors())
    ).points() @ (full_vc.vectors().T) + np.array([2] * len(full_vc.vectors()))
    O7pos = np.where(np.array([min(m) for m in monomials.T]) == 1)[0]

    vx = np.concatenate([[0] * (p_dim - 1), [3, 1]])
    vy = np.concatenate([[0] * (p_dim - 1), [-2, -1]])
    vz = np.concatenate([[0] * (p_dim - 1), [0, 1]])
    uplift_vecs_singular = np.vstack(
        [
            np.vstack(
                [
                    full_vc.vectors().T,
                    [[0] * len(full_vc.vectors()), [1] * len(full_vc.vectors())],
                ]
            ).T,
            [vx, vy, vz],
        ]
    )

    D4res1 = uplift_vecs_singular[O7pos] + vx + 2 * vy
    D4res2 = 2 * uplift_vecs_singular[O7pos] + 2 * vx + 3 * vy
    uplift_vecs = np.vstack([uplift_vecs_singular, D4res1, D4res2])

    p5 = Polytope(uplift_vecs)

    return p5


def sections(points, weights):
    r"""Return the exponent vectors of the monomial sections of a toric divisor.

    Each lattice point $m$ of the Newton polytope (see
    [`Newton_Polytope`][cytools.f_theory.Uplift_functions.Newton_Polytope])
    corresponds to the section
    $\prod_\rho x_\rho^{\langle m, v_\rho \rangle + a_\rho}$.

    Args:
        points (np.ndarray): The rays $v_\rho$ of the toric fan, one per row.
        weights (np.ndarray): The divisor coefficients $a_\rho$.

    Returns:
        np.ndarray: The exponent matrix, with one row per ray and one column per
            section. If the divisor has no sections, an empty 1D array is
            returned.

    Examples:
        The divisor $D_1$ on $\mathbb{P}^2$ has the sections $x_3$, $x_2$, and
        $x_1$.

        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import sections
        >>> sections(np.array([[1, 0], [0, 1], [-1, -1]]), np.array([1, 0, 0]))
        array([[0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]])
    """
    NP = Newton_Polytope(points, weights)
    if len(NP.points()) == 0:
        return np.array([])
    else:
        return points @ NP.points().T + weights[:, None]


def solve_over_integers(M, b):
    r"""Find an integer solution of the linear system $M x = -b$.

    The system is solved with the Smith normal form of $M$. Note the sign: the
    system that is solved is $M x = -b$, not $M x = b$. If an inconsistent equation
    of the form $0 = c$ with $c \neq 0$ is encountered, `PROBLEM` is printed.

    Args:
        M (ArrayLike): The integer matrix $M$.
        b (ArrayLike): The integer vector $b$.

    Returns:
        tuple[bool, np.ndarray | None]: Whether an integer solution exists and, if
            it does, a solution $x$ (otherwise `None`).

    Examples:
        >>> from cytools.f_theory.Uplift_functions import solve_over_integers
        >>> solve_over_integers([[1, 0], [0, 2]], [3, 4])
        (True, array([-3, -2]))
        >>> solve_over_integers([[1, 0], [0, 2]], [3, 3])
        (False, None)
    """
    A, S, T = smith_normal_decomp(Matrix(M), domain=ZZ)
    a = np.array(A, dtype=int)
    s = np.array(S, dtype=int)
    t = np.array(T, dtype=int)
    c = -s @ b
    y = np.zeros(a.shape[1], dtype=int)
    for ii in range(np.min(a.shape)):
        if a[ii, ii] != 0:
            y[ii] = c[ii] / a[ii, ii]
        else:
            if c[ii] != 0:
                print("PROBLEM")
    if np.all(a @ y == c):
        return (True, t @ y)
    else:
        return (False, None)


def make_simplicial(fan):
    """Return a simplicial refinement of a fan.

    Each maximal cone with more rays than the dimension of the fan is replaced by
    the cones of a fine triangulation of its generators, and the simplicial cones
    are kept. The cones are triangulated independently of each other, so the
    result is only guaranteed to be a fan if non-simplicial cones do not share
    non-simplicial faces.

    Args:
        fan (Fan): The toric fan.

    Returns:
        Fan: The simplicial refinement, with the same vector configuration.
    """

    new_cones = set(fan.cones())
    dim = fan.dim
    for c in fan.cones():
        if len(c) > dim:
            new_cones.discard(c)
            carr = np.array(c)
            vc_tri = VectorConfiguration(fan.vectors(c)).triangulate(make_fine=True)
            for co in vc_tri.cones():
                new_cones.add(tuple(carr[np.array(co) - 1]))
    return Fan(vc=fan.vc, cones=new_cones)


def refine_fan(fan, blowups_or_all_vectors=None):
    r"""Insert new rays into a simplicial fan by star subdivisions.

    If `blowups_or_all_vectors` is specified, its rows that are not yet in the
    vector configuration of the fan are appended to it, keeping the cones of the
    fan. Then, each vector of the vector configuration that is not a ray of the fan
    is inserted: the smallest cone $\tau$ containing it is found, and every cone
    $\tau \cup c$, with $c$ in the link of $\tau$, is replaced by the cones
    $\tau' \cup c \cup \{\rho\}$, where $\rho$ is the new ray and $\tau'$
    ranges over the facets of $\tau$.

    Args:
        fan (Fan): The simplicial toric fan.
        blowups_or_all_vectors (np.ndarray | None): The new rays, one per row, or a
            full vector configuration containing both the old and the new rays. If
            not specified, the vectors of the vector configuration of `fan` that
            are not rays of the fan are inserted.

    Returns:
        Fan: The refined fan.

    Examples:
        We blow up the torus-fixed point of the first cone of $\mathbb{F}_1$.

        >>> import numpy as np
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import refine_fan
        >>> fan = VectorConfiguration([[1, 0], [0, 1], [-1, 1], [0, -1]]).triangulate()
        >>> refine_fan(fan, np.array([[1, 1]])).cones()
        ((1, 4), (1, 5), (2, 3), (2, 5), (3, 4))
    """

    if blowups_or_all_vectors is not None:
        blowups = row_difference(blowups_or_all_vectors, fan.vc.vectors())
        all_vectors = np.concatenate((fan.vc.vectors(), blowups), axis=0)
        vc_all = VectorConfiguration(all_vectors)
        new_fan = Fan(vc_all, cones=fan.cones())
    else:
        new_fan = fan
    vec_diff = row_difference(new_fan.vc.vectors(), new_fan.vectors())
    to_be_refined = set(new_fan.vc.vectors_to_labels(vec_diff))
    if fan.dim == len(new_fan.vc.vectors()[0]):
        for label in to_be_refined:
            all_cones = set(new_fan.cones())
            link_base = tuple(
                find_cone(new_fan.vc.vectors(label), all_cones, new_fan.vc.vectors())
            )
            link_base_len = len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan = Fan(vc=new_fan.vc, cones=all_cones)
        return new_fan
    else:
        for label in to_be_refined:
            all_cones = set(new_fan.cones())
            link_base = tuple(
                find_cone_general(new_fan.vc.vectors(label), all_cones, all_vectors)
            )
            link_base_len = len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan = Fan(vc=new_fan.vc, cones=all_cones)
        return new_fan


def find_cone_general(new_ray, current_cones, all_vectors):
    """Return the face of a cone that contains a given vector.

    For each cone in turn, solve for non-negative coefficients expressing
    `new_ray` in terms of the generators of the cone (with
    `scipy.optimize.nnls`). For the first cone where this succeeds, return the
    labels of the generators with positive coefficients. Unlike
    [`find_cone`][cytools.f_theory.Uplift_functions.find_cone], this also works for
    cones that are not simplicial or not full-dimensional, but for non-simplicial
    cones the returned labels need not generate the smallest face containing the
    vector.

    Args:
        new_ray (np.ndarray): The vector to locate.
        current_cones (Iterable[tuple[int, ...]]): The cones, as tuples of
            one-indexed labels.
        all_vectors (np.ndarray): The vectors, where the vector with label `i` is
            in row `i - 1`.

    Returns:
        frozenset[int] | None: The labels of the generators, or `None` if no cone
            contains the vector.
    """
    for cone in current_cones:
        cone_list = list(cone)
        numpy_indices = [idx - 1 for idx in cone_list]

        A = all_vectors[numpy_indices].T

        x, residual = nnls(A, new_ray)

        if residual < 1e-10:
            carrier_face = frozenset(
                cone_list[i] for i, coeff in enumerate(x) if coeff > 1e-10
            )
            return carrier_face

    return None


def array_to_latex(arr):
    r"""Convert a 2D array into a LaTeX `pmatrix`.

    Args:
        arr (np.ndarray): The 2D array.

    Returns:
        str: The LaTeX code of the matrix.

    Raises:
        ValueError: If `arr` has more than two dimensions.

    Examples:
        >>> import numpy as np
        >>> from cytools.f_theory.Uplift_functions import array_to_latex
        >>> print(array_to_latex(np.array([[1, 2], [3, 4]])))
        \begin{pmatrix}
          1 & 2 \\
          3 & 4 \\
        \end{pmatrix}
    """
    if len(arr.shape) > 2:
        raise ValueError("Only 2D matrices are supported.")

    lines = ["  " + " & ".join(map(str, row)) + " \\\\" for row in arr]
    return "\\begin{pmatrix}\n" + "\n".join(lines) + "\n\\end{pmatrix}"


def integral_gale_transform(points):
    r"""Return an integral Gale transform of a point configuration.

    The points are lifted by appending a coordinate equal to $1$, and a basis of
    the rational nullspace of the lifted configuration is computed with SymPy and
    scaled to be integral. The rows of the result form a basis of the affine
    dependencies $\sum_i c_i p_i = 0$, $\sum_i c_i = 0$ among the points $p_i$,
    and its columns are the Gale vectors of the points.

    Args:
        points (ArrayLike): The points, one per row, as an array of shape
            `(n, d)`.

    Returns:
        np.ndarray: The Gale transform, of shape `(n - d - 1, n)` if the points
            affinely span $\mathbb{R}^d$.

    Raises:
        ValueError: If there are not more than `d + 1` points.

    Warning:
        The nullspace is currently computed from a floating-point matrix, so its
        non-integral entries are truncated instead of being scaled to integers.
        This gives incorrect results whenever the nullspace basis found by SymPy
        has non-integral entries.

    Examples:
        >>> from cytools.f_theory.Uplift_functions import integral_gale_transform
        >>> integral_gale_transform([[0, 0], [1, 0], [0, 1], [1, 1]])
        array([[ 1, -1, -1,  1]])
    """
    points = np.array(points)
    n, d = points.shape

    if n <= d + 1:
        raise ValueError(
            f"Need strictly more points (n={n}) than dimensions + 1 (d+1={d + 1})."
        )

    lifted_points = np.hstack((points, np.ones((n, 1))))

    A = Matrix(lifted_points.T)

    null_basis_vectors = A.nullspace()

    if not null_basis_vectors:
        return np.array([])

    B = Matrix.hstack(*null_basis_vectors)

    for j in range(B.cols):
        LCM = lcm([fraction(B[i, j])[1] for i in range(B.rows)])
        B[:, j] = B[:, j] * LCM

    gale_points = np.array(B.T).astype(int)

    return gale_points


def find_cone(new_ray, current_cones, all_vectors, tol=1e-10):
    """Return the face of a simplicial cone that contains a given vector.

    For each cone in turn, the coefficients expressing `new_ray` in terms of the
    generators of the cone are computed. For the first cone where they are all
    non-negative (up to `tol`), return the labels of the generators with positive
    coefficients, which generate the smallest face of the cone containing the
    vector. The cones must be simplicial and full-dimensional.

    Args:
        new_ray (np.ndarray): The vector to locate.
        current_cones (Iterable[tuple[int, ...]]): The cones, as tuples of
            one-indexed labels.
        all_vectors (np.ndarray): The vectors, where the vector with label `i` is
            in row `i - 1`.
        tol (float): The numerical tolerance for the signs of the coefficients.

    Returns:
        frozenset[int] | None: The labels of the generators of the face, or `None`
            if no cone contains the vector.

    Examples:
        >>> import numpy as np
        >>> from cytools.vector_config import VectorConfiguration
        >>> from cytools.f_theory.Uplift_functions import find_cone
        >>> fan = VectorConfiguration([[1, 0], [0, 1], [-1, 1], [0, -1]]).triangulate()
        >>> find_cone(np.array([1, 1]), fan.cones(), fan.vectors())
        frozenset({1, 2})
    """
    for cone in current_cones:
        numpy_indices = np.array(cone) - 1

        A = all_vectors[numpy_indices].T

        coeffs = np.linalg.solve(A, new_ray)

        if np.all(coeffs >= -tol):
            return frozenset(idx for idx, coeff in zip(cone, coeffs) if coeff > tol)

    return None


def divisor_intersections(fan, intersection_dict, divisors, basis_set, as_LLL=True):
    r"""Compute the classes of the toric curves on a complete intersection.

    Let $X$ be the $d$-dimensional toric variety of `fan` and
    $Y = D_1 \cap \cdots \cap D_k$ the complete intersection of the given
    divisors. For each cone $s$ with $d - k - 1$ rays,
    $C_s = V(s) \cdot D_1 \cdots D_k$ is a curve on $Y$, and its class is
    recorded by the intersection numbers $D_i \cdot C_s$ with the basis divisors
    $i \in$ `basis_set` (e.g. obtained with
    [`basis_H2_toric_fan`][cytools.f_theory.Uplift_functions.basis_H2_toric_fan]).
    These are computed from `intersection_dict` by expanding each $D_a$ in terms
    of the toric prime divisors in the star of $s$. If some $D_a$ has no nonzero
    coefficient on these prime divisors, or none of them is a basis divisor, the
    row is left as zero.

    Args:
        fan (Fan): The toric fan.
        intersection_dict (dict[tuple[int, ...], float]): The intersection
            numbers of the toric variety, with sorted tuples of ray labels as
            keys, as returned by the
            [`intersection_numbers`][cytools.vector_config.fan.Fan.intersection_numbers]
            method of the fan.
        divisors (list[np.ndarray]): The coefficient vectors of the divisors $D_a$.
        basis_set (set[int]): The labels of the basis divisors.
        as_LLL (bool): Whether to return an LLL-reduced basis of the lattice
            generated by the curve classes instead of the classes themselves.

    Returns:
        np.ndarray: If `as_LLL` is false, the curve classes, with one row per cone
            $s$ (in arbitrary order) and one column per basis divisor (in the
            iteration order of `basis_set`). If `as_LLL` is true, the nonzero rows
            of the LLL reduction of this matrix.
    """

    codim_cicy = len(divisors)
    simplices = get_lower_dimensional_cones(fan.cones(), fan.dim - codim_cicy - 1)
    divisor_nonvanishing_sets = []
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div != 0)[0] + 1))

    curves_homology_in_basis = np.zeros((len(simplices), len(basis_set)), dtype=int)

    basis_idx_map = {b: idx for idx, b in enumerate(basis_set)}

    for s_idx, s in enumerate(simplices):
        star_s = fan.star(s)
        link_rays = {item for sub_tuple in star_s for item in sub_tuple}
        valid_intersections = []
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(link_rays))
        valid_i = basis_set.intersection(link_rays)

        if not (all(valid_intersections) and valid_i):
            continue
        for i in valid_i:
            i_idx = basis_idx_map.get(i)
            total_intersection = 0

            for ts in product(*valid_intersections):
                key = tuple(sorted(s + ts + (i,)))
                coefficient = np.prod(
                    [divisors[a][ray - 1] for a, ray in enumerate(ts)]
                )
                total_intersection += coefficient * intersection_dict.get(key, 0)

            curves_homology_in_basis[s_idx, i_idx] = total_intersection

    if as_LLL:
        reduced = np.array(
            lll_reduce(np.asarray(curves_homology_in_basis, dtype=int).T)
        ).T
        return reduced[np.any(reduced != 0, axis=1)]
    return curves_homology_in_basis
