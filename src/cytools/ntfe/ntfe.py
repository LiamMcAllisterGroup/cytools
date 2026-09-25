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
# Description:  Various functions relating calculation of CPL-inequalities,
#               generation of the secondary cone based off of 2-face
#               triangulation data, and generating FRSTs from said data.
# -----------------------------------------------------------------------------

"""Expanded secondary cones and NTFE triangulations.

This module implements the algorithm of
[arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for constructing
non-two-face-equivalent (NTFE) fine, regular (star) triangulations of a
polytope. It computes the CPL (convex piecewise-linear) inequalities that
height vectors must satisfy to induce given fine, regular triangulations of the
2-faces, intersects them into expanded secondary cones, and finds a height
vector in each such cone. See [`cytools.ntfe`][cytools.ntfe] for an overview.

Importing this module (which happens on `import cytools`) attaches most of
these functions to [`Polytope`][cytools.polytope.Polytope] as methods of the
same name, where `self` is the polytope. The exception is
[`cone_of_permissible_heights`][cytools.ntfe.ntfe.cone_of_permissible_heights],
which is a plain function.
"""

# 'standard' imports
import atexit
import collections
import itertools
import joblib
import math
import os
import random
import time

# 3rd party imports
import flint
import numba
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from tqdm import tqdm

# CYTools imports
from cytools.cone import Cone
from cytools.polytope import Polytope
from cytools.polytopeface import PolytopeFace
from cytools.triangulation import Triangulation
from cytools.helpers import matrix, misc
from cytools.utils import adjugate, integral_nullspace

# typing
from numpy.typing import ArrayLike
from typing import Generator, Union


# fast HiGHS feasibility helper for NTFE cones
# --------------------------------------------
# cap on batched integer products, safely below 2**63. Above it the batched
# circuit arithmetic could overflow, so those triangles solve pair by pair
_INT64_HEADROOM = 2**61


def _find_interior_point_highs(
    hyperplanes,
    ambient_dim: int,
    c: float = 1,
):
    r"""Find a point $x$ satisfying $H x \geq c$ using SciPy's HiGHS solver.

    For $c > 0$, such a point lies in the strict interior of the cone
    $H x \geq 0$. This is faster than the [`Cone`][cytools.cone.Cone] methods
    for this purpose.

    Args:
        hyperplanes (ArrayLike | sp.spmatrix | Cone): The defining hyperplanes
            $H$, one inward-facing normal per row. Dense arrays, SciPy sparse
            matrices, `CSR_stack` objects, and [`Cone`][cytools.cone.Cone]
            objects (whose hyperplanes are then used) are accepted.
        ambient_dim (int): The ambient dimension (the number of columns of
            $H$).
        c (float): The right-hand side of the inequalities, which sets how far
            from the walls of the cone the point must be.

    Returns:
        np.ndarray | None: A point satisfying the inequalities, or `None` if
            none was found (e.g. if the cone is not solid). If there are no
            hyperplanes, a vector of ones is returned.
    """
    # accept a formal Cone (e.g. from Polytope.ntfe_cones)
    if isinstance(hyperplanes, Cone):
        hyperplanes = hyperplanes.hyperplanes()

    # convert hyperplanes to a dense numpy array. The 2-face machinery hands
    # back bare CSR blocks, which have neither tolist nor __array__
    if sp.issparse(hyperplanes):
        H = hyperplanes.toarray().astype(np.float64)
    elif hasattr(hyperplanes, "tolist") and not isinstance(
            hyperplanes, (list, np.ndarray)):
        H = np.asarray(hyperplanes.tolist(), dtype=np.float64)
    else:
        H = np.asarray(hyperplanes, dtype=np.float64)

    # trivial cone (no constraints) -- interior is all of R^n
    if H.size == 0 or H.shape[0] == 0:
        return np.ones(ambient_dim)

    res = linprog(
        np.zeros(ambient_dim),
        A_ub=-H,
        b_ub=-c * np.ones(H.shape[0]),
        bounds=[(None, None)] * ambient_dim,
        method="highs",
    )
    if res.status == 0:
        return np.asarray(res.x, dtype=np.float64)
    return None


# incremental feasibility for NTFE enumeration
# --------------------------------------------
class _IncrementalLP:
    r"""Incremental LP for the feasibility of a stack of inequalities $H x \geq 1$.

    The inequalities are added and removed in blocks (one per depth of a
    depth-first search), and the HiGHS model is kept warm between solves.

    Args:
        npts (int): The number of variables, i.e. the dimension of the space of
            height vectors.

    Raises:
        ImportError: If the `highspy` package is not installed.
    """

    def __init__(self, npts: int):
        # lazy: keeps `import cytools` working without highspy installed
        try:
            import highspy
        except ImportError as e:
            raise ImportError(
                "NTFE enumeration needs the `highspy` LP solver: "
                "pip install highspy"
            ) from e
        self._highspy = highspy
        self.npts = npts
        self.h = highspy.Highs()
        self.h.silent()
        inf = highspy.kHighsInf
        self.h.addVars(npts, np.full(npts, -inf), np.full(npts, inf))
        self.depth_rows = []

    def push(self, rows: np.ndarray) -> bool:
        r"""Add a block of inequalities and check feasibility.

        If the enlarged system is infeasible, the block is removed again.

        Args:
            rows (np.ndarray): The inequalities to add, as a dense 2D float
                array with one row $r$ per inequality $r \cdot x \geq 1$. It
                may have no rows (e.g. for a 2-face with a single
                triangulation).

        Returns:
            bool: Whether the system with the new block is feasible.
        """
        n = len(rows)
        if n:
            ncols = rows.shape[1]
            starts = np.arange(n, dtype=np.int32) * ncols
            index = np.tile(np.arange(ncols, dtype=np.int32), n)
            self.h.addRows(n, np.ones(n),
                           np.full(n, self._highspy.kHighsInf),
                           n * ncols, starts, index, rows.ravel())
            self.h.run()
            ok = (self.h.getModelStatus()
                  == self._highspy.HighsModelStatus.kOptimal)
        else:  # e.g. an elementary 2-face: no inequalities
            ok = True
        self.depth_rows.append(n)
        if not ok:
            self.pop()
        return ok

    def pop(self):
        """Remove the most recently added block of inequalities (backtrack)."""
        n = self.depth_rows.pop()
        if n:
            total = self.h.getNumRow()
            self.h.deleteRows(
                n, np.arange(total - n, total, dtype=np.int32))

    def witness(self) -> np.ndarray:
        """Return the solution of the most recent feasible solve.

        Returns:
            np.ndarray: A point satisfying all current inequalities, or a
                vector of ones if there are none.
        """
        if self.h.getNumRow() == 0:  # unconstrained: any point works
            return np.ones(self.npts)
        return np.array(self.h.getSolution().col_value)


def _adjacency_order(poly):
    """Order the 2-faces so that ones sharing points are adjacent.

    The facets are visited in a greedy walk that moves to an unvisited facet
    sharing a 2-face with the current one whenever possible, and the 2-faces
    of each facet are listed in turn.

    Args:
        poly (Polytope): The polytope.

    Returns:
        list[int]: A permutation of the indices of `poly.faces(2)`.
    """
    face_pts = [set(int(v) for v in f.labels) for f in poly.faces(2)]
    facet_pts = [set(int(v) for v in f.labels) for f in poly.facets()]
    facet_faces = [[j for j, fp in enumerate(face_pts) if fp <= fl]
                   for fl in facet_pts]

    # greedy facet walk: hop to an unvisited facet sharing a 2-face with
    # the current one; fall back to any unvisited facet
    n_facets = len(facet_pts)
    visited = [False] * n_facets
    cur, facet_walk = 0, [0]
    visited[0] = True
    while len(facet_walk) < n_facets:
        shared = [i for i in range(n_facets) if not visited[i]
                  and set(facet_faces[i]) & set(facet_faces[cur])]
        cur = shared[0] if shared else visited.index(False)
        visited[cur] = True
        facet_walk.append(cur)

    order, seen = [], set()
    for i in facet_walk:
        for j in facet_faces[i]:
            if j not in seen:
                seen.add(j)
                order.append(j)
    assert len(order) == len(face_pts)
    return order


def _enumerate_ntfes_dfs(poly, face_ineqs, make_star, heights_only,
                         verbosity):
    """Generate all NTFEs by depth-first search over the 2-face FRT choices.

    The 2-faces are visited in the order given by `_adjacency_order`, and the
    inequalities of each choice are pushed onto an `_IncrementalLP`. Partial
    choices that are already infeasible are pruned.

    Args:
        poly (Polytope): The polytope.
        face_ineqs (list[list[sp.csr_matrix]]): For each 2-face (in the order
            of `poly.faces(2)`), the inequalities of each of its FRTs, as
            returned by [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs].
        make_star (bool): Whether to make the triangulations star.
        heights_only (bool): Whether to yield height vectors instead of
            triangulations.
        verbosity (int): The verbosity level.

    Yields:
        Triangulation | np.ndarray: For each NTFE, an FR(S)T of `poly`, or a
            height vector (indexed by point label) realizing it if
            `heights_only` is true.
    """
    # convert each FRT's sparse inequality block to the dense float
    # rows highspy takes, once up front (each is pushed many times)
    dense = [[np.asarray(t.toarray(), dtype=np.float64) for t in f]
             for f in face_ineqs]
    # 2-faces conflict only through shared points, so checking adjacent
    # ones early surfaces infeasibility at shallower depth
    dense = [dense[j] for j in _adjacency_order(poly)]
    lp = _IncrementalLP(len(poly.labels))
    n_faces = len(dense)
    n_out = 0
    t0 = time.perf_counter()

    # iterative DFS (explicit stack; recursion would hit Python's limit
    # on polytopes with many 2-faces)
    candidates = [list(range(len(dense[0])))[::-1]]
    while candidates:
        d = len(candidates) - 1
        if not candidates[d]:
            candidates.pop()
            if candidates:
                lp.pop()
            continue
        k = candidates[d].pop()
        if not lp.push(dense[d][k]):
            continue
        if d + 1 < n_faces:
            candidates.append(list(range(len(dense[d + 1])))[::-1])
            continue
        # leaf: a full feasible choice -- an NTFE
        h = lp.witness()
        lp.pop()
        n_out += 1
        if verbosity >= 1 and n_out % 500_000 == 0:
            dt = time.perf_counter() - t0
            print(f"  {n_out:,} NTFEs at {dt:.0f}s "
                  f"({n_out / dt:.0f}/s)", flush=True)
        if heights_only:
            yield h
        else:
            yield poly.triangulate(heights=h, make_star=make_star)


# (low-level) 2-face inequality functions
# ---------------------------------------
# prefix with '_' to indicate that these shouldn't directly be called by user

# a large slowdown in _2d_frt_cone_ineqs is calculating nullspaces...
# cache them here...

cache_path = os.path.join(misc.cache_dir, "twoface_ineqs.pkl.gz")

_ineq_cached = misc.load_zipped_pickle(cache_path)
if _ineq_cached is None:
    _ineq_cached = dict()


def _save_cache():
    """Save the cache of 2-face circuit inequalities to disk."""
    misc.save_zipped_pickle(_ineq_cached, cache_path)

atexit.register(_save_cache)


def _2d_frt_cone_ineq_rows(self, verbosity: int=0) -> list:
    r"""Compute the CPL inequalities of a 2-face triangulation.

    The inequalities cut out the secondary cone of the triangulation, lifted to
    the space of height vectors of the ambient polytope. This is analogous to
    `Triangulation.secondary_cone(on_faces_dim=2)`, except that point labels
    (rather than point indices) are used as column indices, and 2D speedups
    are used.

    The inequalities are obtained by looking at each pair of triangles that
    share an edge. For each pair, there are four relevant points
    $p_0, p_1, p_2, p_3$, ordered so that $p_2$ and $p_3$ define the shared
    edge. The normal of the associated inequality spans the (1-dimensional)
    nullspace of the homogenized matrix

    $$
    M = \begin{pmatrix}
    p_{0,x} & p_{1,x} & p_{2,x} & p_{3,x} \\
    p_{0,y} & p_{1,y} & p_{2,y} & p_{3,y} \\
    1 & 1 & 1 & 1
    \end{pmatrix},
    $$

    i.e. it is the circuit defined by these points, with its sign chosen so
    that the coefficients of the non-shared points $p_0, p_1$ are positive.
    Nullspaces are cached (and saved to disk at exit), keyed by the translated
    coordinates of the points.

    This function is attached to
    [`Triangulation`][cytools.triangulation.Triangulation] as a private method,
    where `self` is the triangulation of the 2-face. Its points must carry the
    labels of the ambient polytope.

    Args:
        verbosity (int): The verbosity level.

    Returns:
        list[dict[int, int]]: The inward-facing hyperplane normals, one per
            pair of adjacent triangles, each as a `{label: coefficient}`
            dictionary with zero coefficients omitted.
    """
    rows = []

    # relevant inputs
    simps = self.simplices()

    # for each point, find (the indices of) all simplices that include it
    pt_to_simp_inds = collections.defaultdict(list)
    for simp_ind, simp in enumerate(simps):
        for pt in simp:
            pt_to_simp_inds[pt].append(simp_ind)

    # for each pair of simplices, calculate the shared elements
    pair_to_shared = collections.defaultdict(set)
    for pt, simp_inds in pt_to_simp_inds.items():
        for pair in itertools.combinations(simp_inds, 2):
            pair_to_shared[pair].add(pt)

    # Find pairs of rows that share at least two common elements
    N_pairs = len(pair_to_shared)
    for i, (shared_simps, s) in enumerate(pair_to_shared.items()):
        if verbosity >= 1:
            print(f"Constructing inequalities associated to simplex pair {i+1}/{N_pairs}")

        # s are the shared points
        if len(s) <= 1:
            continue
        else:
            s = list(s)

        # get the simplices
        simp1 = simps[shared_simps[0]]
        simp2 = simps[shared_simps[1]]

        # calculate the not-shared points
        n_s = [x for x in list(simp1) + list(simp2) if (x not in s)]

        # find the dependency defining the circuit
        M = self.points(which=n_s + s, optimal=True).T
        M_tup = tuple(tuple(row[1:] - row[0]) for row in M)

        # Grab/calculate the nullspace
        ineq = _ineq_cached.get(M_tup, None)
        if ineq is None:
            # calculate the nullspace
            null = flint.fmpz_mat(M.tolist() + [[1, 1, 1, 1]]).nullspace()
            null = null[0].transpose().tolist()[0]

            # ensure the not-shared points have positive coordinates
            if null[0] < 0:
                ineq = [-int(x) for x in null]
            else:
                ineq = [int(x) for x in null]

            # cache this answer
            _ineq_cached[M_tup] = ineq

        # define the associated hyperplane normal
        rows.append({lab: c for lab, c in
                     zip((n_s[0], n_s[1], s[0], s[1]), ineq) if c})

    return rows


def _2d_frt_cone_ineqs(self, ambient_dim: int,
                       verbosity: int=0) -> "sp.csr_matrix":
    """Compute the CPL inequalities of a 2-face triangulation as a CSR matrix.

    See `_2d_frt_cone_ineq_rows` for the mathematics.

    This function is attached to
    [`Triangulation`][cytools.triangulation.Triangulation] as a private method,
    where `self` is the triangulation of the 2-face.

    Args:
        ambient_dim (int): The dimension of the space of height vectors (the
            number of points of the ambient polytope), i.e. the number of
            columns of the output.
        verbosity (int): The verbosity level.

    Returns:
        sp.csr_matrix: The inward-facing hyperplane normals, one per row, with
            columns indexed by point label.
    """
    return matrix.csr_dicts(_2d_frt_cone_ineq_rows(self, verbosity),
                            ambient_dim)


Triangulation._2d_frt_cone_ineqs = _2d_frt_cone_ineqs


def _2d_s_cone_ineqs(self,
    poly,
    ambient_dim: int,
    verbosity: int=0) -> "sp.csr_matrix":
    r"""Compute the inequalities enforcing starness on a 2-face triangulation.

    These are the CPL inequalities necessary to enforce that each 2-simplex
    of the 2-face triangulation is a face of a star 4-simplex (i.e. one
    containing the origin $o$) in the full triangulation. This is done by
    iterating over each 2-simplex $s$ and enforcing that each 4-simplex
    containing $s$ also contains the origin, which amounts to evaluating every
    relevant circuit containing $s \cup \{o\}$ and imposing the associated
    constraint on the heights.

    In more detail, $s \cup \{o\}$ is a 3-simplex. Pairs of 4-simplices
    containing it are of the form $s \cup \{o, i\}$ and $s \cup \{o, j\}$, with
    $i$ and $j$ boundary points (points interior to facets are skipped) of two
    different facets containing $s$. The set $s \cup \{o, i, j\}$ has six
    points but only spans a 4-dimensional space, so it defines a dependency
    $\lambda$. If both $s \cup \{o, i\}$ and $s \cup \{o, j\}$ appear in an
    FRST $T$, then the heights are constrained by requiring $h_i$ and $h_j$ to
    be large, i.e. $\lambda \cdot (h_s, h_o, h_i, h_j) \geq 0$. This can also be
    thought of as a constraint that $h_o$ is sufficiently low.

    The cone of interest is that of heights which respect the 2-face
    triangulations and define star triangulations. One can think of it as
    either

    1. allowing any flips that don't change the 2-face structure and don't make
       the triangulation non-star (dropping these hyperplanes), or
    2. disallowing any flips which change the 2-face structure or make the
       triangulation non-star (keeping these hyperplanes).

    Since the flip defined by $s \cup \{o, i, j\}$ makes the triangulation
    non-star, the constraint must be imposed if any FRST $T$ with the imposed
    2-face restrictions contains both $s \cup \{o, i\}$ and $s \cup \{o, j\}$.
    If the resulting cone (respecting the 2-face and star structure) is solid,
    then such heights exist: $h + \epsilon\lambda$ works for $h$ on the wall
    defined by $\lambda$ (i.e. $h \cdot \lambda = 0$) and sufficiently small
    $\epsilon$. If $h$ is on a wall of codimension 1, then $h + \epsilon\lambda$
    defines $T$ and hence the hyperplane $\lambda$ must be included. If $h$ is
    on a wall of higher codimension, then this circuit defines a flip to an
    irregular triangulation, but the constraint $h \cdot \lambda \geq 0$ does
    not cut the cone.

    This function is attached to
    [`Triangulation`][cytools.triangulation.Triangulation] as a private method,
    where `self` is the triangulation of the 2-face.

    Args:
        poly (Polytope): The ambient polytope.
        ambient_dim (int): The dimension of the space of height vectors (the
            number of points of the ambient polytope), i.e. the number of
            columns of the output.
        verbosity (int): The verbosity level.

    Returns:
        sp.csr_matrix: The inward-facing hyperplane normals enforcing
            starness, one per row, with columns indexed by point label.
    """
    blocks = []
    rows = []

    o = poly.label_origin

    # homogenized points, indexed by label
    pts = np.asarray(poly.points(), dtype=np.int64)
    pts_ext = np.zeros((max(poly.labels) + 1, pts.shape[1] + 1), dtype=np.int64)
    pts_ext[list(poly.labels)] = np.hstack(
        [pts, np.ones((len(pts), 1), dtype=np.int64)])
    pmax = int(np.abs(pts_ext).max())

    # find each facet containing each 2d simplex
    containing_facets = collections.defaultdict(list)
    for s in self.simplices(2):
        for f in poly.faces(3):
            if set(s).issubset(set(f.labels)):
                containing_facets[tuple(s)].append(f)

    simps = self.simplices(2)
    for i, s in enumerate(simps):
        if verbosity >= 1:
            print(f"Constructing inequalities associated to 2-simplex "
                  f"{i+1}/{len(simps)}")
        s = sorted(s.tolist())
        for f1, f2 in itertools.combinations(containing_facets[tuple(s)], 2):
            i1 = np.fromiter(sorted(set(f1.labels_bdry) - set(f2.labels_bdry)
                                    - set(s)), dtype=np.int64)
            i2 = np.fromiter(sorted(set(f2.labels_bdry) - set(f1.labels_bdry)
                                    - set(s)), dtype=np.int64)
            if not (len(i1) and len(i2)):
                continue

            # the points common to every circuit of this 2-simplex are fixed,
            # so their kernel and a pre-inverted subsystem are found once and
            # every apex pair below becomes a matrix multiply
            comm = s + [o]
            Q = pts_ext[comm]
            kernel = integral_nullspace(Q)
            if kernel.shape[1] != 1:
                continue
            u = kernel[:, 0]
            for cols in itertools.combinations(range(Q.shape[1]), Q.shape[0]):
                adj, det = adjugate(Q[:, cols])
                if det != 0:
                    break
            else:
                continue
            cols = np.array(cols)

            P1, P2 = pts_ext[i1], pts_ext[i2]
            a1, a2 = P1 @ u, P2 @ u
            if not ((a1 != 0).all() and (a2 != 0).all()):
                continue

            # fall back to one solve per pair where int64 could overflow
            amax = max(int(np.abs(a1).max()), int(np.abs(a2).max()))
            bound = max(amax * abs(det),
                        8 * amax * pmax * int(np.abs(adj).max()))
            if bound > _INT64_HEADROOM:
                for p1, p2 in itertools.product(i1.tolist(), i2.tolist()):
                    M = poly.points(which=[p1, p2] + comm, optimal=True).T
                    null = flint.fmpz_mat(
                        M.tolist() + [[1] * M.shape[1]]).nullspace()
                    if null[1] != 1:
                        continue
                    lam = [int(x) for x in null[0].transpose().tolist()[0]]
                    if lam[0] < 0:
                        lam = [-x for x in lam]
                    rows.append({lab: c for lab, c
                                 in zip([p1, p2] + comm, lam) if c})
                continue

            # primitive circuit coefficients, all apex pairs at once
            g = np.gcd.outer(np.abs(a1), np.abs(a2))
            A = a2[None, :] // g
            B = -a1[:, None] // g
            combo = (A[:, :, None] * P1[:, None, :]
                     + B[:, :, None] * P2[None, :, :])
            c = -(combo[:, :, cols] @ adj)
            V = np.concatenate([(A * det)[..., None], (B * det)[..., None], c],
                               axis=2).reshape(-1, 2 + len(comm))
            V //= np.gcd.reduce(np.abs(V), axis=1)[:, None]
            V *= np.sign(V[:, [0]])          # orient coeff(p1) > 0

            labs = np.empty(V.shape, dtype=np.int64)
            labs[:, 0] = np.repeat(i1, len(i2))
            labs[:, 1] = np.tile(i2, len(i1))
            labs[:, 2:] = comm
            blocks.append(matrix.csr_rows(labs, V, ambient_dim))

    blocks.append(matrix.csr_dicts(rows, ambient_dim))
    return matrix.csr_stack(blocks, ambient_dim)


Triangulation._2d_s_cone_ineqs = _2d_s_cone_ineqs


@numba.njit(cache=True)
def _2d_frt_subfan_search(xs, ys, grid, xmin, ymin, out):
    """Find the point tuples generating the CPL inequalities of a 2-face.

    This is the search half of `_2d_frt_subfan_ineq_rows`; see there for the
    mathematics. It is JIT-compiled with Numba because the case B loop is
    $O(N_{pts}^3)$ and, for the largest 2-faces in the Kreuzer-Skarke
    database, dominates the whole calculation.

    Args:
        xs (np.ndarray): The x coordinates (in optimal coordinates) of the
            points of the face.
        ys (np.ndarray): The y coordinates (in optimal coordinates) of the
            points of the face.
        grid (np.ndarray): A dense lookup table, where `grid[x-xmin, y-ymin]`
            is the index of the point $(x, y)$ in `xs`/`ys`, or -1 if it is not
            a point of the face.
        xmin (int): The x offset of `grid`.
        ymin (int): The y offset of `grid`.
        out (np.ndarray): The output buffer, of shape `(capacity, 4)`. Row `r`
            is filled with the indices `(i, j, k, m)` of the points involved:
            the vertices `i, j, k` (with `k = -1` in case A) and the interior
            point `m`.

    Returns:
        int: The number of tuples found. This is returned even if it exceeds
            the capacity of `out` (in which case only the leading rows were
            written), so that the caller can resize the buffer and try again.
    """
    N_pts = len(xs)
    capacity = len(out)
    count = 0

    # case A) three consecutive collinear points
    for i in range(N_pts):
        x0, y0 = xs[i], ys[i]
        for j in range(i + 1, N_pts):
            x2, y2 = xs[j], ys[j]

            if np.gcd(abs(x2 - x0), abs(y2 - y0)) != 2:
                continue  # not exactly one interior point on the segment

            if count < capacity:
                # the face is convex, so the midpoint is necessarily one of its
                # lattice points
                out[count, 0] = i
                out[count, 1] = j
                out[count, 2] = -1
                out[count, 3] = grid[(x0 + x2) // 2 - xmin, (y0 + y2) // 2 - ymin]
            count += 1

    # case B) simplex with a single interior point, no 'extra' boundary points
    for i in range(N_pts):
        x0, y0 = xs[i], ys[i]
        for j in range(i + 1, N_pts):
            x1, y1 = xs[j], ys[j]
            dx01, dy01 = x1 - x0, y1 - y0

            # every edge must be primitive, so a non-primitive (p0,p1) rules
            # out the whole inner loop
            if np.gcd(abs(dx01), abs(dy01)) != 1:
                continue

            for k in range(j + 1, N_pts):
                x2, y2 = xs[k], ys[k]

                # inlined 2x the triangle area
                if abs(dx01 * (y2 - y0) - dy01 * (x2 - x0)) != 3:
                    continue  # not the (N_bdry,N_int)=(3,1) case

                if (np.gcd(abs(x2 - x0), abs(y2 - y0)) != 1) or (
                    np.gcd(abs(x2 - x1), abs(y2 - y1)) != 1
                ):
                    continue  # bad case, (N_bdry,N_int)=(5,0)

                if count < capacity:
                    # good case, (N_bdry,N_int)=(3,1)
                    # centroid is interior lattice pt (math stackexchange 124553)
                    sum_x, sum_y = x0 + x1 + x2, y0 + y1 + y2
                    out[count, 0] = i
                    out[count, 1] = j
                    out[count, 2] = k
                    out[count, 3] = grid[sum_x // 3 - xmin, sum_y // 3 - ymin]
                count += 1

    return count


def _2d_frt_subfan_ineq_rows(self) -> list:
    r"""Compute the inequalities cutting out the heights giving FRTs of a 2-face.

    These define the support of the subfan of the secondary fan of the 2-face
    consisting of the secondary cones of its fine, regular triangulations
    (FRTs). The interior of this cone consists of the height vectors that lead
    to FRTs of the 2-face. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for a proof.

    Regularity is automatic (we are dealing with height vectors), so only
    fineness needs to be enforced. There are two ways in which fineness can be
    violated:

    - A) For three collinear points $p_0, p_1, p_2$, the middle point must lie
      below the line through the end points.
    - B) For a triangle $S$ with an interior point $p$, the point $p$ must lie
      below the plane defined by $S$.

    It suffices to consider

    - A) three *consecutive* collinear points, and
    - B) triangles with a *single* interior point and no boundary points other
      than their vertices.

    In practice, these are found by

    - A) iterating over all pairs of distinct points $(p_0, p_2)$ and keeping
      those whose difference has GCD 2. Such a segment contains exactly one
      further lattice point, its midpoint $p_1$, so we impose
      $2 h_{p_1} < h_{p_0} + h_{p_2}$.
    - B) iterating over all triples of distinct points $(p_0, p_1, p_2)$ with
      twice the area equal to 3. By Pick's theorem, such a triangle has either
      3 boundary points and 1 interior point, or 5 boundary points and no
      interior points. The latter case (which has a non-primitive edge) is
      skipped. In the former case, the interior point $p_3$ is the centroid,
      and we impose $3 h_{p_3} < h_{p_0} + h_{p_1} + h_{p_2}$.

    This function is attached to
    [`PolytopeFace`][cytools.polytopeface.PolytopeFace] as a private method,
    where `self` is the 2-face.

    Returns:
        list[dict[int, int]]: The inward-facing hyperplane normals, each as a
            `{label: coefficient}` dictionary.
    """
    pts = np.asarray(self.points(optimal=True), dtype=np.int64)
    N_pts = len(pts)
    if N_pts < 3:
        return []

    # the search kernel wants contiguous coordinates and a dense point lookup
    xs = np.ascontiguousarray(pts[:, 0])
    ys = np.ascontiguousarray(pts[:, 1])
    xmin, ymin = int(xs.min()), int(ys.min())
    grid = np.full(
        (int(xs.max()) - xmin + 1, int(ys.max()) - ymin + 1), -1, dtype=np.int64
    )
    grid[xs - xmin, ys - ymin] = np.arange(N_pts)

    # every face seen so far needs well under N_pts^2/2 rows, but grow-and-retry
    # rather than trust that
    capacity = max(64, N_pts**2 // 2)
    while True:
        out = np.empty((capacity, 4), dtype=np.int64)
        count = _2d_frt_subfan_search(xs, ys, grid, xmin, ymin, out)
        if count <= capacity:
            break
        capacity = count

    labels = self.labels
    rows = [
        (
            {labels[i]: 1, labels[j]: 1, labels[k]: 1, labels[m]: -3}
            if k >= 0
            else {labels[i]: 1, labels[j]: 1, labels[m]: -2}
        )
        for i, j, k, m in out[:count]
    ]

    return rows


PolytopeFace._2d_frt_subfan_ineq_rows = _2d_frt_subfan_ineq_rows


# generate secondary cone/fan
# ---------------------------
def cone_of_permissible_heights(
    triangs: [Triangulation],
    npts: int,
    poly: "Polytope" = None,
    require_star: bool = False,
    dense: bool = False,
    big_ints: bool = False,
    as_cone: bool = True,
    verbosity: int = 0,
) -> "sp.csr_matrix | Cone":
    """Return the cone of heights inducing given 2-face triangulations.

    Given triangulations of (some of) the 2-faces of a polytope, construct the
    cone whose strict interior consists of the height vectors whose induced
    regular subdivision restricts to the given triangulation on each of these
    2-faces. The height vectors have one entry per lattice point of the
    polytope, indexed by point label.

    This is akin to the expanded secondary cone (see
    [`ntfe_cones`][cytools.ntfe.ntfe.ntfe_cones]), except that some 2-faces
    may be left free, which is why the more generic name is used. Duplicate
    inequalities are removed.

    Args:
        triangs (list[Triangulation | None]): The triangulations of the
            2-faces to enforce, with points labelled as in the ambient
            polytope (e.g. as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs]).
            `None` entries are skipped, leaving the corresponding 2-face free.
        npts (int): The number of lattice points of the polytope. This is the
            ambient dimension of the cone.
        poly (Polytope | None): The ambient polytope. Only used, and then
            required, if `require_star` is true.
        require_star (bool): Whether to also include the inequalities enforcing
            starness of the resulting triangulation. This is usually not
            recommended, as triangulations can be made star simply by lowering
            the height of the origin. It is only recommended if the cone (or
            related ones, like the Kahler cone) is of independent interest.
        dense (bool): Whether to return the hyperplanes as a dense array
            instead of a sparse matrix. Only used if `as_cone` is false.
        big_ints (bool): Whether to convert dense hyperplanes to the default
            (64-bit) integer type instead of the 16-bit type used internally.
            Only used if `dense` is true and `as_cone` is false.
        as_cone (bool): Whether to return a [`Cone`][cytools.cone.Cone] object
            instead of its hyperplanes.
        verbosity (int): The verbosity level.

    Returns:
        Cone | sp.csr_matrix | np.ndarray: The cone, or its inward-facing
            hyperplane normals (one per row) as a sparse matrix or, if `dense`
            is true, as a dense array.

    Raises:
        ValueError: If `require_star` is true but `poly` is not specified.

    Examples:
        We construct the cone of heights that induce the first FRT of each
        2-face, and check that it is solid.

        >>> from cytools import Polytope
        >>> from cytools.ntfe import cone_of_permissible_heights
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> triangs = [ft[0] for ft in p.face_triangs()]
        >>> c = cone_of_permissible_heights(triangs, len(p.labels))
        >>> print(c)
        A rational polyhedral cone in RR^13 defined by 6 hyperplanes
        >>> c.is_solid()
        True
    """
    if require_star and (poly is None):
        raise ValueError("If `require_star=True`, then `poly` must be specified")

    blocks = []
    frt_rows = []

    # iterate over face triangulations
    for i,face_triang in enumerate(triangs):
        if verbosity >= 1:
            print(f"Studying 2-face {i}/{len(triangs)}...")
        # skip triangulation in case it is None
        if face_triang is None:
            continue

        # CPL inequalities associated with ith triangulation
        # (normally, this is the triangulation of the ith face, but it doesn't
        # need to be... you can decide to pass a subset of faces)
        if (verbosity >= 2) and require_star:
            print("The 2-face inequalities...")
        # accumulate rows across faces: one CSR build at the end costs far
        # less than one tiny block per face
        frt_rows.extend(_2d_frt_cone_ineq_rows(face_triang,
                                               verbosity=verbosity-1))
        if require_star:
            if (verbosity >= 2):
                print("The star inequalities...")
            blocks.append(_2d_s_cone_ineqs(face_triang, poly, npts,
                                           verbosity=verbosity-1))

    blocks.insert(0, matrix.csr_dicts(frt_rows, npts))

    # delete duplicate rows
    ineqs = matrix.csr_unique_rows(matrix.csr_stack(blocks, npts))

    # densify
    if dense or as_cone:
        ineqs = ineqs.toarray()
        if big_ints or as_cone:
            ineqs = ineqs.astype(int)

    # return
    if as_cone:
        return Cone(hyperplanes=ineqs, ambient_dim=npts,
                    parse_inputs=(len(ineqs)==0))
    else:
        return ineqs


def expanded_secondary_fan(
    self, dense: bool = False, big_ints: bool = False, as_cone: bool = True
) -> "sp.csr_matrix | Cone":
    """Return the support of the expanded secondary fan.

    This is the cone of all height vectors that induce fine, regular
    triangulations (FRTs) of every 2-face of the polytope (emphasis on fine).
    Equivalently, it is the union of all expanded secondary cones (see
    [`ntfe_cones`][cytools.ntfe.ntfe.ntfe_cones]). It is called the
    "expanded" (rather than "plain") secondary fan because only 2-face
    information is used. The height vectors have one entry per lattice point
    of the polytope, indexed by point label. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `expanded_secondary_fan`, where `self` is the polytope.

    Args:
        dense (bool): Whether to return the hyperplanes as a dense array
            instead of a sparse matrix. Only used if `as_cone` is false.
        big_ints (bool): Whether to convert dense hyperplanes to the default
            (64-bit) integer type instead of the 16-bit type used internally.
            Only used if `dense` is true and `as_cone` is false.
        as_cone (bool): Whether to return a [`Cone`][cytools.cone.Cone] object
            instead of its hyperplanes.

    Returns:
        Cone | sp.csr_matrix | np.ndarray: The support of the expanded
            secondary fan, or its inward-facing hyperplane normals (one per
            row) as a sparse matrix or, if `dense` is true, as a dense array.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> print(p.expanded_secondary_fan())
        A rational polyhedral cone in RR^13 defined by 7 hyperplanes
    """
    ambient_dim = len(self.labels)

    # iterate over face triangulations
    # one CSR build over every face, rather than a tiny block each
    rows = []
    for f in self.faces(2):
        rows.extend(_2d_frt_subfan_ineq_rows(f))
    ineqs = matrix.csr_dicts(rows, ambient_dim)

    if dense or as_cone:
        ineqs = ineqs.toarray()
        if big_ints or as_cone:
            ineqs = ineqs.astype(int)
    if as_cone:
        return Cone(hyperplanes=ineqs, ambient_dim=ambient_dim, parse_inputs=(len(ineqs)==0))
    else:
        return ineqs


Polytope.expanded_secondary_fan = expanded_secondary_fan


# extend face-triangulations to FR(S)T
# ------------------------------------
def triangfaces_to_frt(
    self,
    triangs: [Triangulation],
    make_star: bool = False,
    check_heights: bool = False,
    verbosity: int = 0,
) -> Triangulation:
    """Construct an FR(S)T that restricts to the given 2-face triangulations.

    A height vector in the strict interior of the cone given by
    [`cone_of_permissible_heights`][cytools.ntfe.ntfe.cone_of_permissible_heights]
    is found, and the polytope is triangulated with it (without points
    interior to facets). Some 2-faces can be left free by setting the
    corresponding entries of `triangs` to `None`, or by omitting them. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `triangfaces_to_frt`, where `self` is the polytope.

    Args:
        triangs (list[Triangulation | None]): The 2-face triangulations, with
            points labelled as in the polytope (e.g. as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs]).
            `None` entries leave the corresponding 2-face free.
        make_star (bool): Whether to make the triangulation star, turning the
            FRT into an FRST.
        check_heights (bool): Whether to check that the heights uniquely define
            the triangulation (see
            [`check_heights`][cytools.triangulation.Triangulation.check_heights]).
        verbosity (int): The verbosity level. Currently unused.

    Returns:
        Triangulation | None: An FR(S)T that restricts to the specified 2-face
            triangulations, or `None` if no such triangulation was found (i.e.
            if the cone of permissible heights is not solid).

    Examples:
        We construct an FRT that restricts to the first FRT of each 2-face.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> triangs = [ft[0] for ft in p.face_triangs()]
        >>> t = p.triangfaces_to_frt(triangs)
        >>> t.is_fine(), t.is_regular(), t.is_star()
        (True, True, False)
    """
    npts = len(self.labels)

    ineqs = cone_of_permissible_heights(
        triangs, poly=self, npts=npts, as_cone=False,
    )
    h = _find_interior_point_highs(ineqs, npts)

    if h is None:
        return None

    reduced_heights = np.delete(h, self.labels_facet)
    t = self.triangulate(
        heights=reduced_heights,
        include_points_interior_to_facets=False,
        make_star=make_star,
        check_heights=check_heights,
    )
    return t


Polytope.triangfaces_to_frt = triangfaces_to_frt


def triangfaces_to_frst(
    self,
    triangs: [Triangulation],
    check_heights: bool = False,
    verbosity: int = 0,
) -> Triangulation:
    """Construct an FRST that restricts to the given 2-face triangulations.

    This is [`triangfaces_to_frt`][cytools.ntfe.ntfe.triangfaces_to_frt] with
    `make_star=True`. Some 2-faces can be left free by setting the
    corresponding entries of `triangs` to `None`, or by omitting them. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `triangfaces_to_frst`, where `self` is the polytope.

    Args:
        triangs (list[Triangulation | None]): The 2-face triangulations, with
            points labelled as in the polytope (e.g. as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs]).
            `None` entries leave the corresponding 2-face free.
        check_heights (bool): Whether to check that the heights uniquely define
            the triangulation (see
            [`check_heights`][cytools.triangulation.Triangulation.check_heights]).
        verbosity (int): The verbosity level. Currently unused.

    Returns:
        Triangulation | None: An FRST that restricts to the specified 2-face
            triangulations, or `None` if no such triangulation was found.

    Examples:
        We construct an FRST that restricts to the first FRT of each 2-face.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> triangs = [ft[0] for ft in p.face_triangs()]
        >>> t = p.triangfaces_to_frst(triangs)
        >>> t.is_fine(), t.is_regular(), t.is_star()
        (True, True, True)
    """
    return self.triangfaces_to_frt(
        triangs=triangs,
        make_star=True,
        check_heights=check_heights,
        verbosity=verbosity,
    )


Polytope.triangfaces_to_frst = triangfaces_to_frst


# generate ALL 2-face inequivalent hyperplanes/cones/FRSTs
# --------------------------------------------------------
def triangface_ineqs(
    self,
    face_triangs: list = None,
    require_star: bool = False,
    max_npts: int = 17,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    return_triangs: bool = False,
    verbosity: int = 0,
) -> "[[sp.csr_matrix]]":
    """Compute the FRTs of the 2-faces and their CPL inequalities.

    For each FRT of each 2-face, this computes the inequalities cutting out
    its secondary cone, lifted to the space of height vectors of the polytope
    (one entry per lattice point, indexed by point label). These are the
    building blocks of the expanded secondary cones (see
    [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers]).

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `triangface_ineqs`, where `self` is the polytope.

    Args:
        face_triangs (list[list[Triangulation]] | None): The FRTs of each
            2-face, as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
            If not specified, they are computed.
        require_star (bool): Whether to also include the inequalities enforcing
            starness. This is not recommended unless such cones are of direct
            interest. If one only cares about NTFE FRSTs, it is more efficient
            to enforce starness when constructing the triangulation, by
            lowering the height of the origin (i.e. with `make_star=True`).
        max_npts (int): The maximum number of lattice points of a 2-face for
            which all FRTs are enumerated (if `face_triangs` is not
            specified). For 2-faces with more points, FRTs are only sampled.
        N_face_triangs (int): The maximum number of FRTs to sample for each
            2-face with more than `max_npts` points (if `face_triangs` is not
            specified).
        triang_method (str): The method used to sample FRTs of 2-faces with
            more than `max_npts` points (if `face_triangs` is not specified).
            The options are `"fast"`, `"fair"`, `"grow2d"`, and `"dualgnn"`;
            see [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
        return_triangs (bool): Whether to also return the 2-face
            triangulations.
        verbosity (int): The verbosity level.

    Returns:
        list | tuple: For each 2-face (in the order of `self.faces(2)`), a
            list with the inequalities of each of its FRTs, each as a sparse
            matrix with one inward-facing normal per row. If `return_triangs`
            is true, a tuple of this list and the 2-face triangulations is
            returned instead.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> ineqs = p.triangface_ineqs()
        >>> [len(f) for f in ineqs]  # number of FRTs of each 2-face
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1]
    """
    npts = len(self.labels)

    # find all 2-face triangulations
    if face_triangs is None:
        if verbosity > 1:
            print("Calculating the face triangulations...")
        face_triangs = self.face_triangs(
            dim=2,
            only_regular=True,
            max_npts=max_npts,
            N_face_triangs=N_face_triangs,
            triang_method=triang_method,
            verbosity=verbosity - 1,
        )

    # iterate over faces
    if verbosity > 1:
        print("Calculating the hyperplane inequalities...")
    ineqs = []
    iter_wrapper = (
        tqdm if verbosity >= 1 else lambda x: x
    )  # (for progress bars)
    for f_triangs in iter_wrapper(face_triangs):
        ineqs.append([])

        # iterate over triangulations of this face
        for f_triang in f_triangs:
            tmp_ineqs = _2d_frt_cone_ineqs(f_triang, npts)
            if require_star:
                tmp_ineqs.append(_2d_s_cone_ineqs(f_triang, self, npts))
            ineqs[-1].append(tmp_ineqs)

    if not return_triangs:
        return ineqs
    else:
        return ineqs, face_triangs


Polytope.triangface_ineqs = triangface_ineqs


def ntfe_hypers(
    self,
    require_star: bool = False,
    N: int = None,
    seed: int = None,
    face_ineqs: list = None,
    face_triangs: list = None,
    max_npts: int = 17,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    as_generator: bool = False,
    separate_boring: bool = True,
    verbosity: int = 0,
) -> Union[Generator["matrix.CSR_stack", None, None], list["matrix.CSR_stack"]]:
    """Return the hyperplanes of (some of) the expanded secondary cones.

    Each choice of one FRT per 2-face defines an expanded secondary cone: the
    intersection of the (lifted) secondary cones of the chosen 2-face FRTs,
    i.e. the set of height vectors inducing all of them. The choices whose
    cone is solid are exactly the NTFEs. This function returns the
    inequalities of these cones, either for all choices or for a random
    sample of `N` distinct choices. The cones are not checked for solidity.
    See [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `ntfe_hypers`, where `self` is the polytope.

    Args:
        require_star (bool): Whether to also include the inequalities enforcing
            starness. This is not recommended unless such cones are of direct
            interest. If one only cares about NTFE FRSTs, it is more efficient
            to enforce starness when constructing the triangulation, by
            lowering the height of the origin (i.e. with `make_star=True`).
        N (int | None): The number of distinct choices of 2-face FRTs (i.e.
            of candidate NTFEs) to sample uniformly at random. If not
            specified, or if it is at least the total number of choices, all
            choices are used.
        seed (int | None): The random seed used when sampling. If not
            specified, it is derived from the current time. NumPy's global
            random state is not affected.
        face_ineqs (list[list[sp.csr_matrix]] | None): The CPL inequalities of
            each FRT of each 2-face, as returned by
            [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs]. If not
            specified, they are computed.
        face_triangs (list[list[Triangulation]] | None): The FRTs of each
            2-face, as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
            Only used if `face_ineqs` is not specified. If not specified, they
            are computed.
        max_npts (int): The maximum number of lattice points of a 2-face for
            which all FRTs are enumerated (if `face_triangs` is not
            specified). For 2-faces with more points, FRTs are only sampled.
        N_face_triangs (int): The maximum number of FRTs to sample for each
            2-face with more than `max_npts` points (if `face_triangs` is not
            specified).
        triang_method (str): The method used to sample FRTs of 2-faces with
            more than `max_npts` points (if `face_triangs` is not specified).
            The options are `"fast"`, `"fair"`, `"grow2d"`, and `"dualgnn"`;
            see [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
        as_generator (bool): Whether to return a generator instead of a list.
            Use this if memory is a concern.
        separate_boring (bool): Whether to merge the inequalities of all
            2-faces with a single FRT into one block. This only changes the
            internal ordering of the 2-faces, and hence the order of the
            outputs and which cones are picked when sampling.
        verbosity (int): The verbosity level.

    Returns:
        list[CSR_stack] | Generator[CSR_stack, None, None]: The hyperplanes of
            each expanded secondary cone, as `CSR_stack` objects (from
            `cytools.helpers.matrix`) with one inward-facing normal per row
            and one column per lattice point of the polytope.

    Examples:
        We compute the hyperplanes of the expanded secondary cones of a
        polytope whose 2-faces have 36 combinations of FRTs.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> hypers = p.ntfe_hypers()
        >>> len(hypers)
        36
        >>> hypers[0]
        CSR_stack(shape=(11, 13))
    """
    # grab the cpl-cone inequalities
    if face_ineqs is None:
        if verbosity >= 1:
            print("Constructing hyperplanes for the 2-faces...")

        ineqs_array = self.triangface_ineqs(
            max_npts=max_npts,
            face_triangs=face_triangs,
            N_face_triangs=N_face_triangs,
            triang_method=triang_method,
            require_star=require_star,
            verbosity=verbosity - 1,
        )
    else:
        # copying face_ineqs... could be dangerous w.r.t. memory...
        ineqs_array = face_ineqs.copy()

    # separate/group the hyperplanes associated to 'boring' 2-faces
    # (i.e., the 2-faces which each only have 1 FRT)
    if separate_boring:
        ineqs_boring = []

        i = 0
        while i < len(ineqs_array):
            if len(ineqs_array[i]) == 1:
                ineqs_boring.append(ineqs_array.pop(i)[0])
            else:
                i += 1

        if len(ineqs_boring):
            # concatenation, not addition: LIL's __add__ stacked rows, which
            # is what this relied on
            ineqs_boring = matrix.csr_stack(ineqs_boring, ineqs_boring[0].shape[1])
            ineqs_array.append([ineqs_boring])

    # get number of triangulations per 2-face
    if verbosity >= 1:
        print("Calculating total number of ineqs...")
    choices_counts = list(map(len, ineqs_array))
    choices = list(map(range, choices_counts))

    # for each set of 2-face triangulations, group the inequalities
    #
    # the intersections/groups are specified by (an integer encoding of) a
    # list of indices such that the ith value indicates which
    # triangulation/inequalities to use for the ith 2-face
    #
    # this integer encoding is basically like binary,
    #   0 -> (0, 0, ..., 0)     i.e., use the '0th' FRT for each 2-face
    #   1 -> (0, 0, ..., 1)     i.e., use the '0th' FRT for all 2-faces, except
    #                                 the last 2-face which has >1 FRTs. Use
    #                                 the '1st' FRT for this last 2-face
    if verbosity >= 1:
        print("Intersecting face H-cones...", end=" ")
        print(f"(there are {math.prod(choices_counts)} total)")

    if (N is None) or (N >= math.prod(choices_counts)):
        if verbosity >= 1:
            print(
                f"Calculating all N={math.prod(choices_counts)} "
                "intersections..."
            )
        # due to the integer encoding that we use, we can specify our choices
        # simply by the numbers 0, 1, ..., math.prod(choices_counts)-1. Each
        # number corresponds to a choice
        chosen = range(math.prod(choices_counts))
    else:
        if verbosity >= 1:
            print(f"Sampling N={N} intersections...")

        # sample cones uniformly on chromosones
        chosen = set()

        # set the seed
        # (use a local generator so that we don't perturb NumPy's global RNG)
        if seed is None:
            seed = time.time_ns() % (2**32)
        rng = np.random.RandomState(seed)

        # choose the hypers
        while len(chosen) < N:
            choice = tuple(rng.choice(x) for x in choices)
            chosen.add(choice)

    # grab/return hyperplanes
    if as_generator:

        def gen():
            for choice in chosen:
                yield matrix.CSR_stack(ineqs_array, choice, choices_counts)

        return gen()

    else:
        if verbosity >= 1:
            hypers = [
                matrix.CSR_stack(ineqs_array, choice, choices_counts)
                for choice in tqdm(chosen)
            ]
        else:
            hypers = [
                matrix.CSR_stack(ineqs_array, choice, choices_counts)
                for choice in chosen
            ]

        return hypers


Polytope.ntfe_hypers = ntfe_hypers


def ntfe_cones(
    self,
    hypers: ["ArrayLike"] = None,
    require_star: bool = False,
    N: int = None,
    seed: int = None,
    face_ineqs: list = None,
    face_triangs: list = None,
    max_npts: int = 17,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    as_generator: bool = False,
    separate_boring: bool = True,
    verbosity=0,
) -> Union[Generator[Cone, None, None], list[Cone]]:
    """Return (some of) the expanded secondary cones of the polytope.

    Each choice of one FRT per 2-face defines an expanded secondary cone (see
    [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers]). The cones are not
    checked for solidity; the solid ones correspond to the NTFEs. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `ntfe_cones`, where `self` is the polytope.

    Args:
        hypers (list[ArrayLike] | None): The hyperplanes defining the cones,
            e.g. as returned by [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers].
            If not specified, they are computed with `ntfe_hypers`.
        require_star (bool): Whether to also include the inequalities enforcing
            starness. This is not recommended unless such cones are of direct
            interest. If one only cares about NTFE FRSTs, it is more efficient
            to enforce starness when constructing the triangulation, by
            lowering the height of the origin (i.e. with `make_star=True`).
        N (int | None): The number of cones to return. If `hypers` is not
            specified, this many distinct choices of 2-face FRTs are sampled
            by `ntfe_hypers`. Otherwise, `N` of the given hyperplanes are
            picked at random (this is ignored if `as_generator` is true). If
            not specified, all cones are returned.
        seed (int | None): The random seed used when sampling. If not
            specified, Python's default seeding (from system randomness or the
            current time) is used. Note that this reseeds Python's global
            `random` module.
        face_ineqs (list[list[sp.csr_matrix]] | None): The CPL inequalities of
            each FRT of each 2-face, as returned by
            [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs]. If not
            specified, they are computed.
        face_triangs (list[list[Triangulation]] | None): The FRTs of each
            2-face, as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
            Only used if `face_ineqs` is not specified. If not specified, they
            are computed.
        max_npts (int): The maximum number of lattice points of a 2-face for
            which all FRTs are enumerated (if `face_triangs` is not
            specified). For 2-faces with more points, FRTs are only sampled.
        N_face_triangs (int): The maximum number of FRTs to sample for each
            2-face with more than `max_npts` points (if `face_triangs` is not
            specified).
        triang_method (str): The method used to sample FRTs of 2-faces with
            more than `max_npts` points (if `face_triangs` is not specified).
            The options are `"fast"`, `"fair"`, `"grow2d"`, and `"dualgnn"`;
            see [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
        as_generator (bool): Whether to return a generator instead of a list.
            Use this if memory is a concern.
        separate_boring (bool): Currently not passed on to `ntfe_hypers`, so
            it has no effect.
        verbosity (int): The verbosity level.

    Returns:
        list[Cone] | Generator[Cone, None, None]: The expanded secondary cones.

    Examples:
        We compute the expanded secondary cones of a polytope and count how
        many of them are solid, i.e. how many NTFEs there are.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> cones = p.ntfe_cones()
        >>> len(cones)
        36
        >>> sum(c.is_solid() for c in cones)
        30
    """
    # random seed stuff
    random.seed(seed)
    seed1 = random.randint(0, 2**16 - 1)  # seed for self.ntfe_hypers
    seed2 = random.randint(0, 2**16 - 1)  # seed for subselecting hypers

    # input checking
    if hypers is None:
        if verbosity >= 1:
            print(
                "Computing hyperplane inequalities associated to 2face "
                "triangulations"
            )

        # generate the hyperplanes
        hypers = self.ntfe_hypers(
            require_star=require_star,
            N=N,
            max_npts=max_npts,
            face_ineqs=face_ineqs,
            face_triangs=face_triangs,
            N_face_triangs=N_face_triangs,
            seed=seed1,
            triang_method=triang_method,
            as_generator=as_generator,
            verbosity=verbosity - 1,
        )
        dim = len(self.labels)
    else:
        # a caller-supplied `hypers` may be a generator or empty, so probing
        # hypers[0] can raise. Materialize generators unless we're streaming.
        if not as_generator:
            try:
                len(hypers)
            except TypeError:
                hypers = list(hypers)

        # set dim
        dim = None
        try:
            first = hypers[0]
        except (TypeError, IndexError, KeyError):
            first = None

        if first is not None:
            if isinstance(first, matrix.CSR_stack):
                if not first.is_empty:
                    dim = len(first[0])
            elif len(first):
                dim = len(first[0])

        if dim is None:
            dim = len(self.labels)

    # if returning a generator, just do so here
    if as_generator:
        if N is not None:
            print(
                f"as_generator=True but N={N} (i.e., !=None)! "
                "ignoring the value of N, instead defaulting to N=None..."
            )

        def gen():
            for hyper in hypers:
                yield Cone(
                    hyperplanes=hyper, ambient_dim=dim, parse_inputs=(len(hyper)==0)
                )

        return gen()

    # not returning a generator...
    if (N is not None) and (N < len(hypers)):
        # randomly sample hypers
        hyper_inds = list(range(len(hypers)))

        # shuffle the indices and select the first N
        random.seed(seed2)
        random.shuffle(hyper_inds)
        hyper_inds = hyper_inds[:N]

        iterator = [hypers[i] for i in hyper_inds]
    else:
        # iterate over all hypers
        iterator = hypers

    # convert hyperplanes to cones
    if verbosity >= 1:
        print("Constructing the formal cones...")

    iter_wrapper = (
        tqdm if verbosity >= 1 else lambda x: x
    )  # (for progress bars)
    return [Cone(hyperplanes=hyper, ambient_dim=dim, parse_inputs=(len(hyper)==0))
            for hyper in iter_wrapper(iterator)]


Polytope.ntfe_cones = ntfe_cones


def ntfe_frts(
    self: "Polytope",
    cones: [Cone] = None,
    hypers: ["ArrayLike"] = None,
    make_star: bool = False,
    N: int = None,
    seed: int = None,
    face_ineqs: list = None,
    face_triangs: list = None,
    max_npts: int = 17,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    as_generator: bool = False,
    n_jobs: int = None,
    heights_only: bool = False,
    verbosity: int = 0,
):
    """Return (some of) the NTFE FR(S)Ts of the polytope.

    For each non-two-face-equivalent (NTFE) class, i.e. each solid expanded
    secondary cone, a height vector in its strict interior is found with an LP
    solver and used to construct a fine, regular (star) triangulation. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for details.

    If none of `N`, `cones`, and `hypers` is specified, all NTFEs are
    enumerated by a depth-first search over the choices of 2-face FRTs, which
    prunes partial choices that are already infeasible using an incremental
    LP. This is much faster than checking every combination of 2-face FRTs,
    with identical results. Otherwise, the given (or sampled) cones are
    checked one by one, and the non-solid ones are skipped, so fewer than `N`
    triangulations may be returned.

    If `heights_only` is true, no [`Triangulation`][cytools.triangulation.Triangulation]
    objects are constructed and the height vectors realizing each NTFE are
    returned instead. This is much faster when one only needs to count the
    NTFEs or post-process the heights.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `ntfe_frts`, where `self` is the polytope.

    Args:
        cones (list[Cone] | None): The expanded secondary cones to use, e.g.
            as returned by [`ntfe_cones`][cytools.ntfe.ntfe.ntfe_cones]. If
            neither `cones` nor `hypers` is specified, the cones are computed.
        hypers (list[ArrayLike] | None): The hyperplanes defining the expanded
            secondary cones to use, e.g. as returned by
            [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers]. Takes precedence
            over `cones`.
        make_star (bool): Whether to make the triangulations star, turning the
            FRTs into FRSTs.
        N (int | None): The number of expanded secondary cones (i.e. candidate
            NTFEs) to sample. If not specified, all of them are used.
        seed (int | None): The random seed used when sampling. If not
            specified, Python's default seeding (from system randomness or the
            current time) is used. Note that this reseeds Python's global
            `random` module. Unused when enumerating all NTFEs.
        face_ineqs (list[list[sp.csr_matrix]] | None): The CPL inequalities of
            each FRT of each 2-face, as returned by
            [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs]. If not
            specified, they are computed.
        face_triangs (list[list[Triangulation]] | None): The FRTs of each
            2-face, as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
            Only used if `face_ineqs` is not specified. If not specified, they
            are computed.
        max_npts (int): The maximum number of lattice points of a 2-face for
            which all FRTs are enumerated (if `face_triangs` is not
            specified). For 2-faces with more points, FRTs are only sampled.
        N_face_triangs (int): The maximum number of FRTs to sample for each
            2-face with more than `max_npts` points (if `face_triangs` is not
            specified).
        triang_method (str): The method used to sample FRTs of 2-faces with
            more than `max_npts` points (if `face_triangs` is not specified).
            The options are `"fast"`, `"fair"`, `"grow2d"`, and `"dualgnn"`;
            see [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
        as_generator (bool): Whether to return a generator instead of a list.
            Use this if memory is a concern.
        n_jobs (int | None): The number of parallel workers used to find a
            height vector in each cone. If not specified, 1 is used (i.e. the
            computation is sequential); -1 uses all cores. This is only worth
            raising when there are many cones, as each worker pays pickling
            overhead. Unused when enumerating all NTFEs or when `as_generator`
            is true.
        heights_only (bool): Whether to return the height vectors instead of
            the triangulations.
        verbosity (int): The verbosity level.

    Returns:
        list | Generator: The NTFE FR(S)Ts, as
            [`Triangulation`][cytools.triangulation.Triangulation] objects, or,
            if `heights_only` is true, the height vectors realizing them, as
            NumPy arrays with one entry per lattice point of the polytope.

    Examples:
        We find all NTFE FRTs of a polytope. Only 30 of the 36 combinations of
        2-face FRTs extend to a triangulation of the polytope.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> frts = p.ntfe_frts()
        >>> len(frts)
        30
        >>> frts[0].is_fine(), frts[0].is_regular(), frts[0].is_star()
        (True, True, False)
        >>> len(p.ntfe_frts(heights_only=True))
        30
    """
    # full enumeration -> DFS with prefix pruning
    if (N is None) and (cones is None) and (hypers is None):
        if face_ineqs is None:
            face_ineqs = self.triangface_ineqs(
                face_triangs=face_triangs,
                max_npts=max_npts,
                N_face_triangs=N_face_triangs,
                triang_method=triang_method,
                require_star=False,
                verbosity=verbosity - 1,
            )
        gen = _enumerate_ntfes_dfs(self, face_ineqs, make_star,
                                   heights_only, verbosity)
        return gen if as_generator else list(gen)

    # random seed stuff
    random.seed(seed)
    seed1 = random.randint(0, 2**16 - 1)  # seed for self.ntfe_hypers
    seed2 = random.randint(0, 2**16 - 1)  # seed for subselecting hypers

    # grab cones, if not provided
    if verbosity >= 1:
        print("Calculating expanded secondary cones...")

    if hypers is not None:
        data = hypers
    elif cones is not None:
        data = cones
    else:
        data = self.ntfe_hypers(
            require_star=False,
            N=N,
            max_npts=max_npts,
            seed=seed1,
            face_ineqs=face_ineqs,
            face_triangs=face_triangs,
            N_face_triangs=N_face_triangs,
            triang_method=triang_method,
            as_generator=as_generator,
            verbosity=verbosity - 1,
        )

    # randomly select N cones/hyperplanes
    # (might get fewer than N FRSTs, in case the cones aren't all solid)
    # (when data is a generator from ntfe_hypers, it already consists of N
    # randomly sampled cones, so no shuffling/slicing is needed)
    if (N is not None) and isinstance(data, list):
        random.seed(seed2)
        data = list(data)  # don't shuffle the caller's list in place!
        random.shuffle(data)
        data = data[:N]

    # if returning a generator, just do so here
    npts_amb = len(self.labels)
    if as_generator:
        def gen():
            for datum in data:
                h = _find_interior_point_highs(datum, npts_amb)
                if h is None:
                    continue
                if heights_only:
                    yield h
                    continue

                yield self.triangulate(heights=h, make_star=make_star)

        return gen()

    # for each expanded secondary cone, calculate the corresponding FRST
    time_per_cone = 0.1  # ~0.1s to try to find a point in each of these cones
    time_estimate = time_per_cone * len(data)
    if verbosity >= 1:
        print("Calculating the FRSTs (find 1x point in each cone)")
        print(f"(anticipated to take <~{time_estimate}s)")
    elif time_estimate > 180:
        print(f"Warning: there are {len(data)} cones. Finding a", end=" ")
        print(f"point in each is anticipated to take <~{time_estimate}s...")

    frsts = []

    def func(datum):
        h = _find_interior_point_highs(datum, npts_amb)
        if h is None:
            return None
        if heights_only:
            return h

        return self.triangulate(heights=h, make_star=make_star)

    # check the selected rays
    # (joblib.Parallel() with no n_jobs runs sequentially, so pass it through)
    results = joblib.Parallel(n_jobs=(1 if n_jobs is None else n_jobs))(
        joblib.delayed(func)(datum)
        for datum in data
    )

    for frst in results:
        if frst is not None:
            frsts.append(frst)

    return frsts


Polytope.ntfe_frts = ntfe_frts


def ntfe_frsts(
    self: "Polytope",
    cones: [Cone] = None,
    hypers: ["ArrayLike"] = None,
    N: int = None,
    seed: int = None,
    face_ineqs: list = None,
    face_triangs: list = None,
    max_npts: int = 17,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    as_generator: bool = False,
    n_jobs: int = None,
    heights_only: bool = False,
    verbosity: int = 0,
):
    """Return (some of) the NTFE FRSTs of the polytope.

    This is [`ntfe_frts`][cytools.ntfe.ntfe.ntfe_frts] with `make_star=True`;
    see there for details. See
    [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for the algorithm.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `ntfe_frsts`, where `self` is the polytope.

    Args:
        cones (list[Cone] | None): The expanded secondary cones to use, e.g.
            as returned by [`ntfe_cones`][cytools.ntfe.ntfe.ntfe_cones]. If
            neither `cones` nor `hypers` is specified, the cones are computed.
        hypers (list[ArrayLike] | None): The hyperplanes defining the expanded
            secondary cones to use, e.g. as returned by
            [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers]. Takes precedence
            over `cones`.
        N (int | None): The number of expanded secondary cones (i.e. candidate
            NTFEs) to sample. If not specified, all of them are used.
        seed (int | None): The random seed used when sampling. If not
            specified, Python's default seeding is used. Note that this
            reseeds Python's global `random` module.
        face_ineqs (list[list[sp.csr_matrix]] | None): The CPL inequalities of
            each FRT of each 2-face, as returned by
            [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs]. If not
            specified, they are computed.
        face_triangs (list[list[Triangulation]] | None): The FRTs of each
            2-face, as returned by
            [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
            Only used if `face_ineqs` is not specified. If not specified, they
            are computed.
        max_npts (int): The maximum number of lattice points of a 2-face for
            which all FRTs are enumerated (if `face_triangs` is not
            specified). For 2-faces with more points, FRTs are only sampled.
        N_face_triangs (int): The maximum number of FRTs to sample for each
            2-face with more than `max_npts` points (if `face_triangs` is not
            specified).
        triang_method (str): The method used to sample FRTs of 2-faces with
            more than `max_npts` points (if `face_triangs` is not specified).
            The options are `"fast"`, `"fair"`, `"grow2d"`, and `"dualgnn"`;
            see [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].
        as_generator (bool): Whether to return a generator instead of a list.
            Use this if memory is a concern.
        n_jobs (int | None): The number of parallel workers used to find a
            height vector in each cone. If not specified, 1 is used (i.e. the
            computation is sequential); -1 uses all cores.
        heights_only (bool): Whether to return the height vectors instead of
            the triangulations.
        verbosity (int): The verbosity level.

    Returns:
        list | Generator: The NTFE FRSTs, as
            [`Triangulation`][cytools.triangulation.Triangulation] objects, or,
            if `heights_only` is true, the height vectors realizing them.

    Examples:
        We find all NTFE FRSTs of a polytope.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> frsts = p.ntfe_frsts()
        >>> len(frsts)
        30
        >>> all(t.is_star() for t in frsts)
        True
    """
    return self.ntfe_frts(
        N=N,
        make_star=True,
        cones=cones,
        hypers=hypers,
        face_ineqs=face_ineqs,
        face_triangs=face_triangs,
        max_npts=max_npts,
        seed=seed,
        N_face_triangs=N_face_triangs,
        triang_method=triang_method,
        as_generator=as_generator,
        n_jobs=n_jobs,
        heights_only=heights_only,
        verbosity=verbosity,
    )


Polytope.ntfe_frsts = ntfe_frsts
