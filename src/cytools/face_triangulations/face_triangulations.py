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
# Description:  Methods for generating/sampling face triangulations
# -----------------------------------------------------------------------------

# 'standard' imports
import math
import time

# 3rd party imports
import numpy as np

# CYTools imports
from cytools.polytope import Polytope, Triangulation
from cytools.helpers import basic_geometry

# typing
from typing import Iterable


def face_triangs(
    self,
    dim: int = 2,
    which: list = None,
    only_regular: bool = True,
    max_npts: int = None,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    seed: int = None,
    verbosity: int = 0,
):
    """
    **Description:**
    For each dim-face of a given polytope, generate the list of all fine
    dim-face triangulations of it.

    **Arguments:**
    - `poly`: Polytope of interset.
    - `dim`: The dimension of the faces to generate triangulations for.
    - `only_regular`: Whether to only generate the regular 2-face
        triangulations.
    - `max_npts`: The maximum number of points in a 2-face s.t. we try to get
        all triangulations.
    - `N_face_triangs`: If we aren't trying to get all triangulations, how
        many to grab per face (upper bound).
    - `triang_method`: The method to generate random triangulations. Allowed
        are "fast", "fair", and "grow2d".
    - `seed`: Random seed if grabbing only some triangulations.
    - `verbosity`: Verbosity level.

    **Returns:**
    List of faces. Each face is a list of Triangulation. May have different
    order than in the input (i.e., faces.)
    """

    # check input for generating triangulations
    allowed_methods = ["fast", "fair", "grow2d"]
    if triang_method not in allowed_methods:
        raise ValueError(
            f"triang_method={triang_method} was not an "
            f"allowed method... Allowed are {allowed_methods}."
        )
    # output variable
    triangs = []

    # the faces
    if isinstance(which, int):
        which = [which]

    faces = self.faces(dim)
    if which is not None:
        faces = [faces[i] for i in which]

    # iterate over faces
    ind = 0
    for face in faces:
        p = face.as_poly()  # convert to Polytope to get all triangulations

        if (max_npts is not None) and (len(p.points()) > max_npts):
            if verbosity >= 1:
                print(
                    f"face_triangs: 2face #{ind} has {len(p.points())} points! "
                    f"This is >={max_npts} (user-set limit). "
                    f"Will only request <={N_face_triangs} random "
                    "samples."
                )

            
            if triang_method == "fast":
                triangs.append(
                    p.random_triangulations_fast(
                        N=N_face_triangs,
                        as_list=True,
                        make_star=False,
                        include_points_interior_to_facets=True,
                        seed=seed,
                    )
                )
            elif triang_method == "fair":
                if verbosity >= 1:
                    print("face_triangs: warning... fair never worked well...")
                triangs.append(
                    p.random_triangulations_fair(
                        N=N_face_triangs,
                        as_list=True,
                        make_star=False,
                        include_points_interior_to_facets=True,
                        seed=seed,
                    )
                )
            elif triang_method == "grow2d":
                triangs.append(list(p.grow_frt(N=N_face_triangs, seed=seed)))

            if verbosity >= 2:
                print(f"face_triangs: found {len(triangs[-1])} triangs...")
                if len(triangs[-1]) < N_face_triangs:
                    print(
                        f"face_triangs: This is < {N_face_triangs}... "
                        "maybe you got them all? (not guaranteed)"
                    )
        else:
            if verbosity >= 1:
                print("face_triangs: computing all face triangulations...")
            triangs.append(
                p.all_triangulations(
                    only_fine=True,
                    only_star=False,
                    only_regular=only_regular,
                    include_points_interior_to_facets=True,
                    as_list=True,
                )
            )

        # increment ind (only used for warning message)
        ind += 1

    return triangs


Polytope.face_triangs = face_triangs


def n_2face_triangs(self, only_regular: bool = True) -> int:
    """
    **Description:**
    Return the count of all 2-face triangulations of the input polytope

    **Arguments:**
    - `only_regular`: Whether to restrict to just regular triangulations.

    **Returns:**
    *(integer)* The count of distinct sets of 2-face triangulations
    """
    triangs = self.face_triangs(dim=2, only_regular=only_regular)
    return math.prod([len(f) for f in triangs])


Polytope.n_2face_triangs = n_2face_triangs
Polytope.num_2face_triangs = n_2face_triangs


def grow_ft(
    self,
    bdry: Iterable[Iterable[int]] = None,
    seed: int = None,
    verbosity: int = 0,
) -> "Triangulation":
    """
    **Description:**
    Grow a fine triangulation (FT) of a polygon

    **Arguments:**
    - `bdry`: The collection of boundary edges. Calculated if not provided
    - `seed`: The seed for the random number generator
    - `verbosity`: Verbosity level.

    **Returns:**
    The fine triangulation of poly.
    """
    if seed is None:
        seed = time.time_ns() % (2**32)
    t0 = time.perf_counter()

    # get random number generator
    rand_gen = np.random.Generator(np.random.PCG64(seed=seed))

    # dimension checking
    if self.dim() != 2:
        raise NotImplementedError

    if self.ambient_dim() != 2:
        # find a 2D representation
        if verbosity >= 1:
            print("grow_frt: Finding a 2D representation!")
        poly = Polytope(self.points(optimal=True))
    else:
        poly = self

    # basic point info...
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Grabbing basic point info...")

    pts, pts_i = poly.points(), poly.points(as_indices=True)

    # choose starting simplex
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Calculating starting simplex...")
    while True:
        # grab random triples of points until they define a triangle with no
        # other interior/boundary points
        start = rand_gen.choice(pts_i, 3, replace=False)
        if basic_geometry.triangle_area_2x(pts[start]) == 1:
            start = sorted(start)
            break

    simps = {tuple(start)}

    # set 'choosable' edges
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Calculating 'choosable' edges...")

    edges = {
        frozenset((start[0], start[1])),
        frozenset((start[0], start[2])),
        frozenset((start[1], start[2])),
    }

    if bdry is None:
        if verbosity >= 1:
            print(time.perf_counter() - t0, ": Calculating boundary points...")
        bdry = basic_geometry.get_bdry(poly)

    choosable = edges - bdry

    # get the bounding box of known edges. Used for intersection checking
    edges_bounds = dict()
    for i in range(2):
        for j in range(i + 1, 3):
            edges_bounds[frozenset((start[i], start[j]))] = [
                [
                    min(pts[start[i]][0], pts[start[j]][0]),
                    min(pts[start[i]][1], pts[start[j]][1]),
                ],
                [
                    max(pts[start[i]][0], pts[start[j]][0]),
                    max(pts[start[i]][1], pts[start[j]][1]),
                ],
            ]

    # grow new simplices
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Growing new simplices...")
    while len(choosable):
        # randomly choose an edge
        edge = rand_gen.choice(list(choosable))
        edge_lis = list(edge)
        to_try = [i for i in pts_i if i not in edge]
        rand_gen.shuffle(to_try)

        if verbosity >= 2:
            if verbosity >= 3:
                print(f"\nBuilding off of edge={edge}...")
                print("----------------------------------")
            else:
                print(f"Building off of edge={edge}...")

        while True:
            # try a new vertex
            if len(to_try):
                i = to_try.pop()
            else:
                print("Failed! Returning known simplices...")
                return simps

            if verbosity >= 3:
                print(
                    f"Trying new vertex {i:03d}... ({len(to_try):03d} left)"
                    " -> ",
                    end="",
                )

            # check if this is an existing simplex
            if tuple(sorted([*edge, i])) in simps:
                if verbosity >= 3:
                    print("This gives an existing simplex!")
                continue

            # check for collinearity
            area_2x = basic_geometry.triangle_area_2x(pts[[*edge, i]])
            if area_2x != 1:
                if verbosity >= 3:
                    print(f"Area must be 1/2... we found {area_2x}/2!")
                continue

            # write the proposed edges
            edges_new = [frozenset((e, i)) for e in edge_lis]

            # get the bounding boxes of the proposed edges
            edges_new_bounds = [edges_bounds.get(e, None) for e in edges_new]

            for j in range(2):
                if edges_new_bounds[j] is None:
                    # said bounding box wasn't yet calculated... do so now
                    edges_pts = np.take(pts, [*edges_new[j]], axis=0)
                    edges_bounds[edges_new[j]] = [
                        np.min(edges_pts, axis=0).tolist(),
                        np.max(edges_pts, axis=0).tolist(),
                    ]

                    edges_new_bounds[j] = edges_bounds[edges_new[j]]

            p0i_min, p0i_max = edges_new_bounds[0]
            p1i_min, p1i_max = edges_new_bounds[1]

            # check if proposed edges intersect old ones
            any_intersect = False
            for other in edges:
                other_lis = list(other)
                po_min, po_max = edges_bounds[other]

                # check for intersection b/t (edge_lis[0], i) and 'other'
                if (
                    po_max[0] < p0i_min[0]
                    or po_max[1] < p0i_min[1]
                    or p0i_max[0] < po_min[0]
                    or p0i_max[1] < po_min[1]
                ):
                    pass
                elif basic_geometry.intersect(
                    pts[edge_lis[0]],
                    pts[i],
                    pts[other_lis[0]],
                    pts[other_lis[1]],
                ):
                    if verbosity >= 3:
                        print(
                            f"New edge ({edge_lis[0]}, {i}) intersects"
                            f" ({other_lis[0]}, {other_lis[1]})"
                        )
                    any_intersect = True
                    break

                # check for intersection b/t (edge_lis[1], i) and 'other'
                if (
                    po_max[0] < p1i_min[0]
                    or po_max[1] < p1i_min[1]
                    or p1i_max[0] < po_min[0]
                    or p1i_max[1] < po_min[1]
                ):
                    pass
                elif basic_geometry.intersect(
                    pts[edge_lis[1]],
                    pts[i],
                    pts[other_lis[0]],
                    pts[other_lis[1]],
                ):
                    if verbosity >= 3:
                        print(
                            f"New edge ({edge_lis[1]}, {i}) intersects"
                            f" ({other_lis[0]}, {other_lis[1]})"
                        )
                    any_intersect = True
                    break

            # new point is good
            if not any_intersect:
                if verbosity >= 3:
                    print("adding", edge, i)

                simps.add(tuple(sorted([*edge, i])))
                choosable.remove(edge)
                edges = edges.union(edges_new)
                choosable ^= set(edges_new) - bdry
                break

    # return triangulation
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Done!")

    return poly.triangulate(
        simplices=np.asarray(sorted(simps)), check_input_simplices=False
    )


Polytope.grow_ft = grow_ft


def grow_frt(
    self,
    N: int = 1,
    max_N_tries: int = None,
    bdry: Iterable[Iterable[int]] = None,
    seed: int = None,
    backend: str = None,
    verbosity: int = 0,
) -> "Triangulation":
    """
    **Description:**
    Grow a fine, regular triangulation of a polygon

    **Arguments:**
    - `poly`: The polygon of interest. Assumed 2d.
    - `bdry`: The collection of boundary edges. Calculated if not provided
    - `seed`: The seed for the random number generator
    - `verbosity`: Verbosity level.

    **Returns:**
    The FRT of poly.
    """
    # input checking
    if self.dim() != 2:
        raise NotImplementedError

    if self.ambient_dim() != 2:
        # find a 2D representation
        if verbosity >= 1:
            print("grow_frt: Finding a 2D representation!")
        poly = Polytope(self.points(optimal=True))
    else:
        poly = self

    if max_N_tries is None:
        max_N_tries = 100 * N

    if seed is None:
        seed = time.time_ns() % (2**32)

    frts = set()

    for N_attempt in range(max_N_tries):
        if verbosity >= 1:
            print(f"Attempt #{N_attempt}. Have #{len(frts)} FRTs")
        while True:
            t = poly.grow_ft(bdry=bdry, seed=seed, verbosity=verbosity)
            seed += 1  # update the seed for next time

            if t.is_regular(backend=backend):
                frts.add(t)
                break

        if len(frts) == N:
            break
    else:
        if verbosity >= 1:
            print(
                f"Couldn't construct {N} FRTs before hitting attempt limit {max_N_tries}!"
            )

    if len(frts) == 1:
        return next(iter(frts))
    else:
        return frts


Polytope.grow_frt = grow_frt
