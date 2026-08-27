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
    """
    Find a point in the strict interior of the cone $H x \\geq c$ using
    SciPy's HiGHS solver. Faster than CYTools `Cone` methods for this case.

    **Arguments:**
    - `hyperplanes`: The defining hyperplanes $H$. A `Cone` is also accepted,
        in which case its hyperplanes are used.
    - `ambient_dim`: The ambient dimension (number of columns of $H$).
    - `c`: The 'stretching'. Default 1.
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
    """Warm LP for feasibility of a stack of inequalities H x >= 1."""

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
        """Add one face's rows; return feasibility (rolls back if not)."""
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
        """Remove the most recent level's rows (backtrack)."""
        n = self.depth_rows.pop()
        if n:
            total = self.h.getNumRow()
            self.h.deleteRows(
                n, np.arange(total - n, total, dtype=np.int32))

    def witness(self) -> np.ndarray:
        """The current solve's interior point."""
        if self.h.getNumRow() == 0:  # unconstrained: any point works
            return np.ones(self.npts)
        return np.array(self.h.getSolution().col_value)


def _adjacency_order(poly):
    """Order the 2-faces so ones sharing points are checked early."""
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
    """Generate all NTFEs by DFS over the per-2-face FRT choices."""
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
    misc.save_zipped_pickle(_ineq_cached, cache_path)

atexit.register(_save_cache)


def _2d_frt_cone_ineqs(self, ambient_dim: int,
                       verbosity: int=0) -> "sp.csr_matrix":
    """
    (Very analogous to Triangulation.secondary_cone(on_faces_dim=2)...
    main difference is that this treats point labels as column indices, while
    that method treats point indices as column indices. Also, this uses 2D
    speedups)

    **Description:**
    Compute the secondary cone for the 2-face triangulation.

    **Overview:**
    The hyperplane inequalities/normals are calculated by looking at each pair
    of simplices that share an edge. For each pair, there are thus 4 relevant
    points, p0, p1, p2, and p3. Order the points such that p0 and p1 define
    the shared edge.

    The associated inequalities/normals are then calculated as the (basis
    vectors of the) null-space of the matrix
        M = [[p0_x, p1_x, p2_x, p3_x],
             [p0_y, p1_y, p2_y, p3_y],
             [   1,    1,    1,    1]],
    This matrix is the homogenization of our points. This nullspace is 1D and
    corresponds to the normal defined by the circuit (one has to set the sign).

    **Arguments:**
    - `ambient_dim`: The ambient dimension of the secondary-cone space (i.e.,
        the number of points in the polytope).
    - `verbosity`: The verbosity level.

    **Returns:**
    Each row is an inwards-facing hyperplane normal. I.e., a CPL inequality
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

    return matrix.csr_dicts(rows, ambient_dim)


Triangulation._2d_frt_cone_ineqs = _2d_frt_cone_ineqs


def _2d_s_cone_ineqs(self,
    poly,
    ambient_dim: int,
    verbosity: int=0) -> "sp.csr_matrix":
    """
    **Description:**
    Compute the CPL-inequalities necessary to enforce that each simplex in each
    2-face is a face of a star simplex in the full triangulation.

    Operates by iterating over each 2-simplex s and enforcing that each
    4-simplex containing s also contains the origin o. This is done by
    evaluating every circuit containing s+[o] and enforcing the associated
    constraint on the heights.

    **Explanation:**
    Think s+[o] is a 3-simplex. Possible pair of 4-simplices containing this
    3-simplex can be made by s+[o,i] and s+[o,j] for i,j on the bdry of
    (since we skip pts interior to facets) 2x different facets containing s.

    The set s+[o,i,j] will have 6 elements but only be 4D, so it'll define a
    dependency. If both s+[o,i] and s+[o,j] appear in an FRST T, then the
    heights are constrained by enforcing that h_i and h_j are large:
        [h_s, h_o, h_i, h_j].lambda >= 0.
    This can also be thought of as a constraint that h_o is sufficiently low.

    The entire cone of interest is that of heights which respect the 2-face
    triangulations (and define star triangulations). This can be thought of as
        1) allow any flips that don't change 2-face structure and don't make
           the triangulation non-star... drop these hyperplanes OR
        2) disallow any flips which change 2-face structure or make the
           triangulation non-star... keep these hyperplanes.
    Since the flip defined by s+[o,i,j] makes the triangulation non-star, one
    must enforce it if any FRST has both simplices s+[o,i] and s+[o,j]. I.e.,
    if any FRST T exists with the imposed 2-face restrictions and with
    simplices s+[o,i] and s+[o,j].

    If the resultant cone (that respecting the 2-face+star structure) is solid,
    then such heights exist: h+eps*lambda works for
        -) h on the wall defined by lambda (i.e., h.lambda=0)
        -) eps sufficiently small. 
    If h is on a wall of codim-1, then h+eps*lambda will define T and hence the
    hyperplane lambda must be included. If h is on a wall of higher codim, then
    this circuit defines a flip to an irregular triangulation, but the
    constraint h.lambda>=0 does not cut the cone.

    **Arguments:**
    - `poly`: The ambient polytope.
    - `ambient_dim`: The ambient dimension of the secondary-cone space (i.e.,
        the number of points in the polytope).
    - `verbosity`: The verbosity level.

    **Returns:**
    Each row is an inwards-facing hyperplane normal enforcing starness.
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
    """
    **Description:**
    Find the point-tuples generating the CPL inequalities of a 2-face. This is
    the search half of `_2d_frt_subfan_ineqs`; see there for the mathematics.

    This is JIT-compiled because the case B) loop is O(N_pts^3) and, for the
    largest 2-faces in the KS database, dominates the whole calculation.

    **Arguments:**
    - `xs`, `ys`: The (optimal) coordinates of the points of the face.
    - `grid`: A dense lookup, `grid[x-xmin,y-ymin]` being the index of that
        point in `xs`/`ys` (or -1 if it isn't a point of the face).
    - `xmin`, `ymin`: The offsets of `grid`.
    - `out`: The output buffer. Row r is filled with (i,j,k,m), the indices of
        the points involved, `k` being -1 in case A).

    **Returns:**
    The number of tuples found. This is returned even if it exceeds the
    capacity of `out` (in which case only the leading rows were written), so
    that the caller can resize and try again.
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


def _2d_frt_subfan_ineqs(self, ambient_dim: int) -> "sp.csr_matrix":
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855 for proof

    Compute the (support of the) secondary subfan of FRTs for a 2-face.

    This is a cone whose interior gives the height-vectors which would lead to
    fine, regular triangulations (or, subdivisions).

    **Overview:**
    Regularity is baked-in to this method (we're talking about height-vectors)
    so all that we need to worry about is fine-ness. There are two cases where
    fine-ness can be violated:
        A) for three collinear points, p0, p1, and p2, the interior point must
        be below the line defined by the end points.
        B) for a simplex (S) with an interior point (p), the interior point, p,
        must be below the plane defined by S.

    Computationally, these two cases can be reduced to:
        A) for three *consecutive* collinear points, ...
        B) for a simplex (S) with a *single* interior point and no 'extra'
        boundary points (i.e., only boundary points are vertices)...

    In practice, this is calculated by:
        A) iterating over all pairs of distinct points (p0,p2) and keeping
        those whose difference has GCD 2. Such a segment contains exactly one
        further point, its midpoint p1, so impose 2*p1<=p0+p2
        B) iterating over all subsets of three distinct points (p0,p1,p2) with
        2x-area 3. This is either (N_bdry,N_int)=(3,1) or (5,0). We want the
        (3,1) case, so check for the (5,0) case (i.e., a non-primitive edge)
        and skip if so. Now that we have the (3,1) case, calculate the interior
        point, p3, as the centroid and impose restriction 3*p3<=p0+p1+p2

    **Arguments:**
    - `secondary_dim`: The dimension of the secondary-cone space (i.e., the
        number of points in the polytope)

    **Returns:**
    Each row is an inwards-facing hyperplane normal... represents a CPL
    inequality.
    """
    pts = np.asarray(self.points(optimal=True), dtype=np.int64)
    N_pts = len(pts)
    if N_pts < 3:
        return matrix.csr_dicts([], ambient_dim)

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

    return matrix.csr_dicts(rows, ambient_dim)


PolytopeFace._2d_frt_subfan_ineqs = _2d_frt_subfan_ineqs


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
    """
    **Description:**
    For an input set of 2-face triangulations, generate the cone whose strict
    interior gives height vectors leading to the corresponding FRTs of its
    2-faces.

    This is akin to the 'expanded secondary cone' except we allow enforcing a
    subset of 2-faces. I.e., leaving some 2-faces free. This is why the more
    generic function name is used.

    **Arguments:**
    - `triangs` The triangulation(s) for the specified 2-face(s).
    - `npts`: The number of points in the 4D polytope. Defines the ambient
        dimension of the cone.
    - `poly`: The ambient polyope. Used only if require_star=True.
    - `require_star`: Whether to calculate the extra hyperplanes which enforce
        starness of the resultant triangulation. Usually NOT RECOMMENDED, as
        triangulations can be modified to become star simply by lowering the
        height of the origin. Only recommended if the cone (or related ones,
        like the Kahler cone/Kcup) are of independent interest.
    - `dense`: Whether to use dense hyperplanes.
    - `big_ints`: Whether to use 64bit integers.
    - `as_cone`: Whether to return a formal Cone object.
    - `verbosity`: The verbosity level.

    **Returns:**
    The expanded secondary cone, either as hyperplanes or as a formal Cone
    object.
    """
    if require_star and (poly is None):
        raise ValueError("If `require_star=True`, then `poly` must be specified")

    blocks = []

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
        blocks.append(_2d_frt_cone_ineqs(face_triang, npts,
                                         verbosity=verbosity-1))
        if require_star:
            if (verbosity >= 2):
                print("The star inequalities...")
            blocks.append(_2d_s_cone_ineqs(face_triang, poly, npts,
                                           verbosity=verbosity-1))

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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate the hyperplanes defining the (support of the)
    'expanded-secondary' subfan.

    That is, the cone of all height vectors which define FRTs of the 2-faces
    (emphasis on fine). Equivalently, the union of all (expanded) secondary
    cones.

    This is the 'expanded' (not 'plain') secondary fan because only 2-face
    information is used.

    **Arguments:**
    - `dense`: Whether to use dense hyperplanes.
    - `big_ints`: Whether to use 64bit integers.
    - `as_cone`: Whether to return a formal Cone object.

    **Returns:**
    The expanded secondary subfan, either as hyperplanes or as a formal Cone
    object.
    """
    ambient_dim = len(self.labels)

    # iterate over face triangulations
    ineqs = matrix.csr_stack([f._2d_frt_subfan_ineqs(ambient_dim)
                        for f in self.faces(2)], ambient_dim)

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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Given a list of 2-face triangulations, construct an FR(S)T that reduces to
    said triangulations.

    You can decide to not specify some of the 2-face triangulations. For this,
    just leave the associated element in triangs as None or just skip them.

    (basically just a wrapper for cone_of_permissible_heights)

    **Arguments:**
    - `triangs`: The 2-face triangulations. Elements can be None, in which
        case said 2-face is free.
    - `make_star`: Whether to convert the FRT to an FRST (i.e., make it star).
    - `check_heights`: Whether to check the heights used in the Triangulation.
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FR(S)T obeying the specified 2-face triangulations.
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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Given a list of 2-face triangulations, construct an FRST that reduces to
    said triangulations.

    You can decide to not specify some of the 2-face triangulations. For this,
    just leave the associated element in triangs as None or just skip them.

    (just a wrapper for triangfaces_to_frt)

    **Arguments:**
    - `triangs`: The 2-face triangulations. Elements can be None, in which
        case said 2-face is free.
    - `check_heights`: Whether to check the heights used in the Triangulation.
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FRST obeying the specified 2-face triangulations.
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
    """
    **Description:**
    Calculate the 2-face FRTs and their associated inequalities for this
    polytope.

    **Arguments:**
    - `face_triangs`: The FRTs of the 2-faces. Automatically calculated if not
        provided.
    - `require_star`: Whether to calculate the inequalities to ensure starness.
    - `max_npts`: The maximum number of points of 2-faces for which we try to
        enumerate all FRTs (if face_triangs=None). For 2-faces with more
        points, we only look to sample FRTs.
    - `N_face_triangs`: For each face with |points|>max_npts, look to sample
        only #N_face_triangs FRTs (if face_triangs=None).
    - `triang_method`: For each face with |points|>max_npts, sample FRTs using
        the specified method (if face_triangs=None). Allowed options are
        listed in Polytope.face_triangs. Currently, they are "fast", "fair",
        "grow2d", and "dualgnn".
    - `return_triangs`: Whether to return the 2-face triangulation objects in
        addition to the inequalities. Only relevant if face_triangs=None.
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    List of faces. For each face, a list of (the hyperplanes of) each
    expanded-secondary cone.
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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate the hyperplane normals defining each expanded secondary cone of
    this polytope (i.e., each NTFE). Subsampling is allowed.

    **Arguments:**
    - `require_star`: Whether to also generate the hyperplane inequalities
        enforcing star-ness. Not recommended unless such cones are of direct
        interest. If one only cares about NTFE FRSTs, it's more efficient to
        just enforce starness when making the triangulation by lowering the
        height of the origin, using Triangulation(..., make_star=True, ...).
    - `N`: The number of expanded secondary cones (i.e., of NTFEs) to generate.
        If not set, then *all* expanded secondary cones are generated.
    - `seed`: If only generating a subset of the expanded secondary cones, use
        this as the random seed for selecting the subset. If not provided, the
        current time is used.
    - `face_ineqs`: The cpl-inequalities associate to each 2-face
        triangulation. Automatically calculated if not provided.
    - `face_triangs`: The 2-face triangulation objects. Used if
        face_ineqs=None. Automatically calculated if not provided.
    - `max_npts`: The maximum number of points of 2-faces for which we try to
        enumerate all FRTs (if face_triangs=None). For 2-faces with more
        points, we only look to sample FRTs.
    - `N_face_triangs`: For each face with |points|>max_npts, look to sample
        only #N_face_triangs FRTs (if face_triangs=None).
    - `triang_method`: For each face with |points|>max_npts, sample FRTs using
        the specified method (if face_triangs=None). Allowed options are
        listed in Polytope.face_triangs. Currently, they are "fast", "fair",
        "grow2d", and "dualgnn".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `separate_boring`: Whether, when iterating over NTFEs, to group the
        inequalities associated to each 2-face with only 1 FRT. Only changes
        the ordering of outputs (may have effects on random sampling).
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The (hyperplanes of the) expanded secondary cones.
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
        if seed is None:
            seed = time.time_ns() % (2**32)
        np.random.seed(seed)

        # choose the hypers
        while len(chosen) < N:
            choice = tuple(np.random.choice(x) for x in choices)
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
    """
    **Description:**
    See/cite https://arxiv.org/abs/2309.10855

    Generate (some of) the expanded-secondary cones for this polytope.

    **Arguments:**
    - `hypers`: The hyperplanes defining the cones. If no hyperplanes are
        input, these are automatically calculated.
    - `require_star`: Whether to also generate the hyperplane inequalities
        enforcing star-ness. Not recommended unless such cones are of direct
        interest. If one only cares about NTFE FRSTs, it's more efficient to
        just enforce starness when making the triangulation by lowering the
        height of the origin, using Triangulation(..., make_star=True, ...).
    - `N`: The number of expanded secondary cones (i.e., of NTFEs) to generate.
        If not set, then *all* expanded secondary cones are generated.
    - `seed`: If only generating a subset of the expanded secondary cones, use
        this as the random seed for selecting the subset. If not provided, it
        is initialized either as the system time or using hardware-based
        random sources.
    - `face_ineqs`: The cpl-inequalities associate to each 2-face
        triangulation. Automatically calculated if not provided.
    - `face_triangs`: The 2-face triangulation objects. Used if
        face_ineqs=None. Automatically calculated if not provided.
    - `max_npts`: The maximum number of points of 2-faces for which we try to
        enumerate all FRTs (if face_triangs=None). For 2-faces with more
        points, we only look to sample FRTs.
    - `N_face_triangs`: For each face with |points|>max_npts, look to sample
        only #N_face_triangs FRTs (if face_triangs=None).
    - `triang_method`: For each face with |points|>max_npts, sample FRTs using
        the specified method (if face_triangs=None). Allowed options are
        listed in Polytope.face_triangs. Currently, they are "fast", "fair",
        "grow2d", and "dualgnn".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `separate_boring`: Whether, when iterating over NTFEs, to group the
        inequalities associated to each 2-face with only 1 FRT. Only changes
        the ordering of outputs (may have effects on random sampling).
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The expanded-secondary cones.
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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate (some of) the NTFE FR(S)Ts for this polytope.

    If `heights_only=True`, skip building Triangulation objects and just
    return the height vectors realising each NTFE. Much faster when you
    only need to count or post-process the heights.

    **Arguments:**
    - `cones`: The expanded secondary cones corresponding to the NTFEs. If no
        cones are input, these are automatically calculated.
    - `hypers`: The hyperplanes defining the expanded secondary cones. Only
        used if cones=None. If no hyperplanes are input, these are
        automatically calculated.
    - `make_star`: Whether to convert the NTFE FRTs into FRSTs (i.e., to make
        them star).
    - `N`: The number of expanded secondary cones (i.e., of NTFEs) to generate.
        If not set, then *all* expanded secondary cones are generated (by
        depth-first search over the 2-face FRT choices, pruning infeasible
        prefixes on a warm incremental LP -- much faster than checking
        every combination in the product of choices, with identical
        results).
    - `seed`: If only generating a subset of the expanded secondary cones, use
        this as the random seed for selecting the subset. If not provided, it
        is initialized either as the system time or using hardware-based
        random sources.
    - `face_ineqs`: The cpl-inequalities associate to each 2-face
        triangulation. Automatically calculated if not provided.
    - `face_triangs`: The 2-face triangulation objects. Used if
        face_ineqs=None. Automatically calculated if not provided.
    - `max_npts`: The maximum number of points of 2-faces for which we try to
        enumerate all FRTs (if face_triangs=None). For 2-faces with more
        points, we only look to sample FRTs.
    - `N_face_triangs`: For each face with |points|>max_npts, look to sample
        only #N_face_triangs FRTs (if face_triangs=None).
    - `triang_method`: For each face with |points|>max_npts, sample FRTs using
        the specified method (if face_triangs=None). Allowed options are
        listed in Polytope.face_triangs. Currently, they are "fast", "fair",
        "grow2d", and "dualgnn".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `n_jobs`: The number of parallel workers used to find a height vector in
        each cone. Defaults to 1 (sequential); -1 uses all cores. Only worth
        raising when there are many cones, as each worker pays pickling
        overhead.
    - `verbosity`: Verbosity level. Higher means more verbose.

    **Returns:**
    The FRTs
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
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate (some of) the NTFE FRSTs for this polytope.

    If `heights_only=True`, skip building Triangulation objects and just
    return the height vectors realising each NTFE.

    **Arguments:**
    - `hypers`: The expanded secondary cones corresponding to the NTFEs. If no
        cones are input, these are automatically calculated.
    - `hypers`: The hyperplanes defining the expanded secondary cones. Only
        used if cones=None. If no hyperplanes are input, these are
        automatically calculated.
    - `N`: The number of expanded secondary cones (i.e., of NTFEs) to generate.
        If not set, then *all* expanded secondary cones are generated.
    - `seed`: If only generating a subset of the expanded secondary cones, use
        this as the random seed for selecting the subset. If not provided, it
        is initialized either as the system time or using hardware-based
        random sources.
    - `face_ineqs`: The cpl-inequalities associate to each 2-face
        triangulation. Automatically calculated if not provided.
    - `face_triangs`: The 2-face triangulation objects. Used if
        face_ineqs=None. Automatically calculated if not provided.
    - `max_npts`: The maximum number of points of 2-faces for which we try to
        enumerate all FRTs (if face_triangs=None). For 2-faces with more
        points, we only look to sample FRTs.
    - `N_face_triangs`: For each face with |points|>max_npts, look to sample
        only #N_face_triangs FRTs (if face_triangs=None).
    - `triang_method`: For each face with |points|>max_npts, sample FRTs using
        the specified method (if face_triangs=None). Allowed options are
        listed in Polytope.face_triangs. Currently, they are "fast", "fair",
        "grow2d", and "dualgnn".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `separate_boring`: Whether, when iterating over NTFEs, to group the
        inequalities associated to each 2-face with only 1 FRT. Only changes
        the ordering of outputs (may have effects on random sampling).
    - `n_jobs`: The number of parallel workers used to find a height vector in
        each cone. Defaults to 1 (sequential); -1 uses all cores.
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FRTs
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
