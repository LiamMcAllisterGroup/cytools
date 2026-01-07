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
import numpy as np
from tqdm import tqdm

# CYTools imports
from cytools.cone import Cone
from cytools.polytope import Polytope
from cytools.polytopeface import PolytopeFace
from cytools.triangulation import Triangulation
from cytools.helpers import basic_geometry, matrix, misc
from . import face_triangulations

# typing
from numpy.typing import ArrayLike
from typing import Generator, Union


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


def _2d_frt_cone_ineqs(self, ambient_dim: int, verbosity: int=0) -> matrix.LIL:
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
    # the output variable (doesn't need to be LIL object, but that is nice...)
    ineqs = matrix.LIL(dtype=np.int16, width=ambient_dim)

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
        ineq = None
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
        ineqs.new_row()
        if ineq[0] != 0:
            ineqs[-1, n_s[0]] = ineq[0]
        if ineq[1] != 0:
            ineqs[-1, n_s[1]] = ineq[1]
        if ineq[2] != 0:
            ineqs[-1, s[0]] = ineq[2]
        if ineq[3] != 0:
            ineqs[-1, s[1]] = ineq[3]

    return ineqs


Triangulation._2d_frt_cone_ineqs = _2d_frt_cone_ineqs


def _2d_s_cone_ineqs(self,
    poly,
    ambient_dim: int,
    verbosity: int=0) -> matrix.LIL:
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
    # the output variable (doesn't need to be LIL object, but that is nice...)
    ineqs = matrix.LIL(dtype=np.int16, width=ambient_dim)

    # get the homogenized points (for later use)
    npts = len(poly.points())
    pts_homog = np.vstack([poly.points().T, np.ones(npts,dtype=int)])

    # find each facet containing each 2d simplex
    containing_facets = collections.defaultdict(list)
    for s in self.simplices(2):
        for f in poly.faces(3):
            if set(s).issubset(set(f.labels)):
                containing_facets[tuple(s)].append(f)

    # For each 2d simplex s, enforce that it (with origin) appears for each
    # 4d circuit
    o = poly.label_origin
    simps   = self.simplices(2)
    N_simps = len(simps)
    for i,s in enumerate(simps):
        if verbosity >= 1:
            print(f"Constructing inequalities associated to 2-simplex {i+1}/{N_simps}")
        s = s.tolist()
        for f1, f2 in itertools.combinations(containing_facets[tuple(s)], 2):
            f1_only = set(f1.labels_bdry) - set(f2.labels_bdry) - set(s)
            f2_only = set(f2.labels_bdry) - set(f1.labels_bdry) - set(s)
            for p1, p2 in itertools.product(f1_only, f2_only):
                # calculate the not-shared points
                n_s = [p1, p2]

                # check if n_s + s contains any other points
                # ------------------------------------------
                # if so, it can't be flipped so we can ignore
                pts_circ = np.vstack([poly.points(which=n_s + s).T, np.ones(5,dtype=int)])

                other_mask = np.ones(npts,dtype=bool)
                other_mask[n_s+s] = False
                other_mask[0]     = False

                # check if any other point can be written as a convex combination of n_s+s
                lambdas   = np.linalg.inv(pts_circ)@pts_homog[:,other_mask]
                nonneg    = lambdas>-1e-4
                contained = np.all(nonneg,axis=0)
                bad       = np.any(contained)
                if bad:
                    continue

                # passes check!

                # find the dependency
                # -------------------
                M = poly.points(which=n_s + s + [o], optimal=True).T

                # Grab/calculate the nullspace
                null = flint.fmpz_mat(M.tolist() + [[1]*M.shape[1]]).nullspace()
                null = null[0].transpose().tolist()[0]

                # ensure the not-shared points have positive coordinates
                if null[0] < 0:
                    ineq = [-int(x) for x in null]
                else:
                    ineq = [int(x) for x in null]

                # define the associated hyperplane normal
                ineqs.new_row()
                if ineq[0] != 0:
                    ineqs[-1, p1] = ineq[0]
                if ineq[1] != 0:
                    ineqs[-1, p2] = ineq[1]
                if ineq[2] != 0:
                    ineqs[-1, s[0]] = ineq[2]
                if ineq[3] != 0:
                    ineqs[-1, s[1]] = ineq[3]
                if ineq[4] != 0:
                    ineqs[-1, s[2]] = ineq[4]
                if ineq[5] != 0:
                    ineqs[-1, o] = ineq[5]

    return ineqs


Triangulation._2d_s_cone_ineqs = _2d_s_cone_ineqs


def _2d_frt_subfan_ineqs(self, ambient_dim: int) -> matrix.LIL:
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
        1) iterating over all subsets of three distinct points (p0,p1,p2),
        2) calculating the area of the triangle defined by them,
        3) match area
            -) case 0: check that the points are consecutive. If so, add
            restriction 2*p1<=p0+p2 (denoting end points as p0, p2)
            -) case 3/2: either (N_bdry,N_int)=(3,1) or (5,0). We want (3,1)
            case, so check for (5,0) case and skip if so. Now that we have
            (3,1) case, calculate interior point, p3, as centroid and impose
            restriction 3*p3<=p0+p1+p2
            -) case other: skip

    **Arguments:**
    - `secondary_dim`: The dimension of the secondary-cone space (i.e., the
        number of points in the polytope)

    **Returns:**
    Each row is an inwards-facing hyperplane normal... represents a CPL
    inequality.
    """
    # the output variable (doesn't need to be LIL object, but that is nice...)
    ineqs = matrix.LIL(dtype=np.int16, width=ambient_dim)

    # iterate over triples
    # This could be done more intelligently... some pairs of points will be
    # disallowed... e.g., (1,0) and (4,0). Don't try any triple with said
    # tuple...
    # Maybe can check once you find area=0... maybe use sparse array for quick indexing...
    pts = self.points(optimal=True)
    pts_to_inds = {tuple(pt): i for i, pt in enumerate(pts)}
    pts_inds = list(range(len(self.labels)))

    for inds in itertools.combinations(pts_inds, 3):
        pts_triple = pts[list(inds)]
        area_2x = basic_geometry.triangle_area_2x(pts_triple)

        if area_2x == 0:
            # collinear
            seg = basic_geometry.check_3consecutive_sites(pts_triple)

            if seg is not None:
                # add associated inequalities
                ineqs.new_row()
                ineqs[-1, self.labels[inds[seg[0]]]] = 1
                ineqs[-1, self.labels[inds[seg[1]]]] = -2
                ineqs[-1, self.labels[inds[seg[2]]]] = 1
        elif area_2x == 1:
            continue  # unimodular triangle... skip!
        elif area_2x == 3:
            # either (N_bdry,N_int)=(3,1) or (N_bdry,N_int)=(5,0)
            if (
                (not basic_geometry.is_primitive(pts_triple[1] - pts_triple[0]))
                or (
                    not basic_geometry.is_primitive(
                        pts_triple[2] - pts_triple[0]
                    )
                )
                or (
                    not basic_geometry.is_primitive(
                        pts_triple[2] - pts_triple[1]
                    )
                )
            ):
                continue  # bad case, (N_bdry,N_int)=(5,0)

            # good case, (N_bdry,N_int)=(3,1)
            # centroid is interior lattice point (math stackexchange 124553)
            centroid = np.sum(pts_triple, axis=0) // 3
            i_centroid = pts_to_inds[tuple(centroid)]

            ineqs.new_row()
            ineqs[-1, self.labels[inds[0]]] = 1
            ineqs[-1, self.labels[inds[1]]] = 1
            ineqs[-1, self.labels[inds[2]]] = 1
            ineqs[-1, self.labels[i_centroid]] = -3

    return ineqs


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
) -> "matrix.LIL | Cone":
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

    # the output variable (doesn't need to be LIL object, but that is nice...)
    ineqs = matrix.LIL(dtype=np.int16, width=npts)

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
        face_ineqs = _2d_frt_cone_ineqs(face_triang, npts, verbosity=verbosity-1)
        if require_star:
            if (verbosity >= 2):
                print("The star inequalities...")
            face_ineqs.append(_2d_s_cone_ineqs(face_triang, poly, npts, verbosity=verbosity-1))

        ineqs.append(face_ineqs, tocopy=False)

    # delete duplicate rows
    ineqs.unique_rows()

    # densify
    if dense:
        ineqs = ineqs.dense()
        if big_ints:
            ineqs = ineqs.astype(int)

    # return
    if as_cone:
        return Cone(hyperplanes=ineqs, ambient_dim=npts, parse_inputs=(len(ineqs)==0))
    else:
        return ineqs


def expanded_secondary_fan(
    self, dense: bool = False, big_ints: bool = False, as_cone: bool = True
) -> "matrix.LIL | Cone":
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

    # the output variable (doesn't need to be LIL object, but that is nice...)
    ineqs = matrix.LIL(dtype=np.int16, width=ambient_dim)

    # iterate over face triangulations
    for f in self.faces(2):
        ineqs.append(f._2d_frt_subfan_ineqs(ambient_dim), tocopy=False)

    if dense:
        ineqs = ineqs.dense()
        if big_ints:
            ineqs = ineqs.astype(int)
    if as_cone:
        return Cone(hyperplanes=ineqs, ambient_dim=ambient_dim, parse_inputs=(len(ineqs)==0))
    else:
        return ineqs


Polytope.expanded_secondary_fan = expanded_secondary_fan
Polytope.gerald = expanded_secondary_fan


# extend face-triangulations to FR(S)T
# ------------------------------------
def triangfaces_to_frt(
    self,
    triangs: [Triangulation],
    make_star: bool = False,
    check_heights: bool = False,
    backend: str = None,
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
    - `backend`: The backend to use for cone calculations. Options are
        enumerated in the Cone.find_interior_points docstring. Currently, they
        are "glop", "scip", "cpsat", "mosek", "osqp", and "cvxopt".
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FR(S)T obeying the specified 2-face triangulations.
    """
    npts = len(self.labels)

    c = cone_of_permissible_heights(
        triangs, poly=self, npts=npts
    )
    h = c.find_interior_point(backend=backend, verbose=verbosity > 1)

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
    backend: str = None,
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
    - `backend`: The backend to use for cone calculations. Options are
        enumerated in the Cone.find_interior_points docstring. Currently, they
        are "glop", "scip", "cpsat", "mosek", "osqp", and "cvxopt".
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FRST obeying the specified 2-face triangulations.
    """
    return self.triangfaces_to_frt(
        triangs=triangs,
        make_star=True,
        check_heights=check_heights,
        backend=backend,
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
) -> [[matrix.LIL]]:
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
        and "grow2d".
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
) -> Union[Generator["matrix.LIL_stack", None, None], list["matrix.LIL_stack"]]:
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
        and "grow2d".
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
            ineqs_boring = sum(ineqs_boring[1:], ineqs_boring[0])
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
                yield matrix.LIL_stack(ineqs_array, choice, choices_counts)

        return gen()

    else:
        if verbosity >= 1:
            hypers = [
                matrix.LIL_stack(ineqs_array, choice, choices_counts)
                for choice in tqdm(chosen)
            ]
        else:
            hypers = [
                matrix.LIL_stack(ineqs_array, choice, choices_counts)
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
        and "grow2d".
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
        # set dim
        dim = None
        if isinstance(hypers[0], matrix.LIL_stack):
            if not hypers[0].is_empty:
                dim = len(hypers[0][0])
        elif len(hypers[0]):
            dim = len(hypers[0][0])

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
    triang_method: str = "fast",
    as_generator: bool = False,
    backend: str = None,
    verbosity: int = 0,
):
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate (some of) the NTFE FR(S)Ts for this polytope

    **Arguments:**
    - `cones`: The expanded secondary cones corresponding to the NTFEs. If no
        cones are input, these are automatically calculated.
    - `hypers`: The hyperplanes defining the expanded secondary cones. Only
        used if cones=None. If no hyperplanes are input, these are
        automatically calculated.
    - `make_star`: Whether to convert the NTFE FRTs into FRSTs (i.e., to make
        them star).
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
        and "grow2d".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `separate_boring`: Whether, when iterating over NTFEs, to group the
        inequalities associated to each 2-face with only 1 FRT. Only changes
        the ordering of outputs (may have effects on random sampling).
    - `backend`: The backend to use for cone calculations.
    - `verbosity: Verbosity level. Higher means more verbose.

    **Returns:**
    The FRTs
    """
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
    if N is not None:
        random.seed(seed2)
        random.shuffle(data)
        data = data[:N]

    # if returning a generator, just do so here
    if as_generator:
        def gen():
            for datum in data:
                c = Cone(hyperplanes=datum, parse_inputs=(len(datum)==0))
                h = c.find_interior_point(
                    backend=backend, verbose=verbosity > 1
                )

                frst = self.triangulate(heights=h, make_star=make_star)
                if (frst is None) or (not frst):
                    continue
                else:
                    yield frst

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
        c = Cone(hyperplanes=datum, parse_inputs=(len(datum)==0))
        h = c.find_interior_point(backend=backend, verbose=verbosity > 1)

        return self.triangulate(heights=h, make_star=make_star)

    # check the selected rays
    results = joblib.Parallel()(
        joblib.delayed(func)(datum)
        for datum in data
    )

    for frst in results:
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
    triang_method: str = "fast",
    as_generator: bool = False,
    backend: str = None,
    verbosity: int = 0,
):
    """
    **Description:**
    See https://arxiv.org/abs/2309.10855

    Generate (some of) the NTFE FR(S)Ts for this polytope

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
        and "grow2d".
    - `as_generator`: Whether to return a generator which iterates over (the
        hyperplanes of) expanded secondary cones. If False, then a list of all
        such cones is returned. Use generators if memory is a concern.
    - `separate_boring`: Whether, when iterating over NTFEs, to group the
        inequalities associated to each 2-face with only 1 FRT. Only changes
        the ordering of outputs (may have effects on random sampling).
    - `backend`: The backend to use for cone calculations.
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
        backend=backend,
        verbosity=verbosity,
    )


Polytope.ntfe_frsts = ntfe_frsts
