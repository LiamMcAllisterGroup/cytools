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

"""Generation and sampling of triangulations of polytope faces.

This module provides the functions that produce the 2-face triangulations used
by the NTFE machinery in [`cytools.ntfe`][cytools.ntfe]:

- [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs] enumerates
  (or, for large faces, samples) the fine triangulations of each face of a
  polytope.
- [`n_2face_triangs`][cytools.ntfe.face_triangulations.n_2face_triangs] counts
  the combinations of 2-face triangulations.
- [`grow_ft`][cytools.ntfe.face_triangulations.grow_ft] and
  [`grow_frt`][cytools.ntfe.face_triangulations.grow_frt] randomly grow fine
  (regular) triangulations of a polygon, one unimodular triangle at a time.

Importing this module (which happens on `import cytools`) attaches these
functions to [`Polytope`][cytools.polytope.Polytope] as methods of the same
name (`n_2face_triangs` is also available as `num_2face_triangs`), which is how
they are normally called. In each of them, `self` is the polytope.
"""

# 'standard' imports
import importlib
import math
import subprocess
import sys
import time
import warnings

# 3rd party imports
import numpy as np

# CYTools imports
from cytools.polytope import Polytope, Triangulation
from cytools.helpers import basic_geometry

# typing
from typing import Iterable, Union

_DUALGNN_HINT = (
    "The optional dualgnn package is required for GNN-based sampling "
    "(triang_method='dualgnn'). Install it with `pip install cytools[gnn]`, "
    "or with `conda env update -f environment-gnn.yml` in conda environments."
)


def _is_interactive() -> bool:
    """Return whether a human is plausibly at the controls.

    Returns:
        bool: Whether the session is interactive (a REPL, a Jupyter kernel, or
            a terminal attached to standard input).
    """
    return (
        hasattr(sys, "ps1")
        or "ipykernel" in sys.modules
        or (hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
    )


def _import_dualgnn():
    """Import the optional `dualgnn` package.

    If the package is missing and the session is interactive, the user is asked
    whether to install it with pip. It is never installed without explicit
    consent.

    Returns:
        tuple: The `dualgnn.sample_frts` function and the
            `dualgnn.model.DualGNN` class.

    Raises:
        ImportError: If `dualgnn` is not installed and either the session is not
            interactive or the user declines to install it.
    """
    try:
        from dualgnn import sample_frts
        from dualgnn.model import DualGNN

        return sample_frts, DualGNN
    except ImportError as e:
        if not _is_interactive():
            raise ImportError(_DUALGNN_HINT) from e

    try:
        import torch  # noqa: F401

        size_note = ""
    except ImportError:
        size_note = (
            "\n(This will also install PyTorch, a multi-GB download. Conda "
            "users may prefer `conda env update -f environment-gnn.yml`.)"
        )
    try:
        answer = input(
            "GNN-based sampling requires the optional dualgnn package."
            f"{size_note}\nInstall it now via pip? [y/N] "
        )
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise ImportError(_DUALGNN_HINT)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "dualgnn"])
    importlib.invalidate_caches()
    from dualgnn import sample_frts
    from dualgnn.model import DualGNN

    return sample_frts, DualGNN


def _dualgnn_face_triangs(
    face,
    face_poly: "Polytope",
    N: int,
    model=None,
    seed: int = None,
    verbosity: int = 0,
) -> list:
    """Sample up to `N` distinct FRTs of a 2-face with the dualGNN sampler.

    The sampler is described in
    [arXiv:2605.27770](https://arxiv.org/abs/2605.27770). This is a helper for
    [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs].

    Args:
        face (PolytopeFace): The 2-face of interest.
        face_poly (Polytope): The 2-face as a polytope, i.e. `face.as_poly()`.
        N (int): The number of triangulations to sample. Sampling is done with
            replacement, so fewer may be returned after deduplication and
            regularity filtering.
        model (DualGNN | str | None): A `dualgnn.model.DualGNN` instance, or a
            path to a model checkpoint. If not specified, the default
            checkpoint shipped with the `dualgnn` package (and cached by it) is
            used.
        seed (int | None): Random seed for the GNN sampler.
        verbosity (int): The verbosity level. The sampler is verbose if it is
            at least 2.

    Returns:
        list[Triangulation]: The distinct fine, regular triangulations of
            `face_poly` that were found.

    Raises:
        ImportError: If the `dualgnn` package is not available (see
            `_import_dualgnn`).
    """
    sample_frts, DualGNN = _import_dualgnn()

    # resolve a checkpoint path to a model (None -> dualgnn's cached default)
    if (model is not None) and (not isinstance(model, DualGNN)):
        model = DualGNN.from_ckpt(model)

    # sample unique regular FRTs as point-index triples
    # (row i of face.points(optimal=True) <-> face.labels[i])
    pts2d = np.asarray(face.points(optimal=True), dtype=np.int64)
    simps = sample_frts(
        pts2d,
        N,
        model=model,
        only_regular=True,
        seed=seed,
        verbose=verbosity >= 2,
    )

    # map point indices to labels
    labels = np.asarray(face.labels)
    return [
        # include_points_interior_to_facets=True: 2-faces that are themselves
        # reflexive would otherwise drop edge-interior points
        face_poly.triangulate(
            simplices=labels[s].tolist(),
            include_points_interior_to_facets=True,
        )
        for s in simps
    ]


def face_triangs(
    self,
    dim: int = 2,
    which: list = None,
    only_regular: bool = True,
    max_npts: int = None,
    N_face_triangs: int = 1000,
    triang_method: str = "grow2d",
    dualgnn_model=None,
    seed: int = None,
    verbosity: int = 0,
):
    """Return the fine triangulations of each `dim`-dimensional face.

    For each `dim`-face of the polytope, all of its fine triangulations are
    enumerated with TOPCOM (see
    [`Polytope.all_triangulations`][cytools.polytope.Polytope.all_triangulations]).
    If `max_npts` is specified, faces with more than `max_npts` lattice points
    are instead sampled: up to `N_face_triangs` fine, regular triangulations
    are generated with the method given by `triang_method`. All lattice points
    of each face are used, and the triangulations are never made star.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `face_triangs`, where `self` is the polytope.

    Args:
        dim (int): The dimension of the faces to triangulate. The `"grow2d"`
            and `"dualgnn"` sampling methods only work for 2-faces.
        which (int | list[int] | None): The index, or list of indices, of the
            faces to triangulate, as positions in `self.faces(dim)`. If not
            specified, all `dim`-faces are triangulated.
        only_regular (bool): Whether to only enumerate regular triangulations.
            This only affects faces that are fully enumerated; sampled
            triangulations are always regular.
        max_npts (int | None): The maximum number of lattice points of a face
            for which all triangulations are enumerated. Faces with more points
            are sampled instead. If not specified, all triangulations of every
            face are enumerated.
        N_face_triangs (int): The maximum number of triangulations to sample
            for each face with more than `max_npts` points. Fewer may be
            returned.
        triang_method (str): The method used to sample triangulations of faces
            with more than `max_npts` points. The options are `"fast"` and
            `"fair"` (which use the
            [fast][cytools.polytope.Polytope.random_triangulations_fast] and
            [fair][cytools.polytope.Polytope.random_triangulations_fair]
            random triangulation samplers of `Polytope`), `"grow2d"` (see
            [`grow_frt`][cytools.ntfe.face_triangulations.grow_frt]), and
            `"dualgnn"`, which samples FRTs with a graph neural network
            ([arXiv:2605.27770](https://arxiv.org/abs/2605.27770)) and requires
            the optional `dualgnn` package.
        dualgnn_model (DualGNN | str | None): Only used if
            `triang_method="dualgnn"`. A `dualgnn.model.DualGNN` instance, or a
            path to a model checkpoint. If not specified, the default
            checkpoint shipped with the `dualgnn` package is used.
        seed (int | None): The random seed used when sampling triangulations.
        verbosity (int): The verbosity level.

    Returns:
        list[list[Triangulation]]: One list of triangulations per face, in the
            same order as `self.faces(dim)` (or as `which`, if specified).

    Raises:
        ValueError: If `triang_method` is not one of the allowed options.

    Examples:
        We find the fine, regular triangulations of each 2-face of a polytope
        and count them. Most of its 2-faces have a unique FRT.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> ft = p.face_triangs()
        >>> [len(f) for f in ft]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1]
        >>> ft[19][0].simplices()  # one of the FRTs of a 2-face with 3 FRTs
        array([[ 3,  7,  9],
               [ 3,  9, 12],
               [ 4,  9, 12]], dtype=uint8)
    """

    # check input for generating triangulations
    allowed_methods = ["fast", "fair", "grow2d", "dualgnn"]
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
                frts = p.grow_frt(N=N_face_triangs, seed=seed)
                if isinstance(frts, Triangulation):
                    # grow_frt returns a bare Triangulation when it finds
                    # exactly one
                    frts = {frts}
                triangs.append(list(frts))
            elif triang_method == "dualgnn":
                triangs.append(
                    _dualgnn_face_triangs(
                        face,
                        p,
                        N_face_triangs,
                        model=dualgnn_model,
                        seed=seed,
                        verbosity=verbosity,
                    )
                )

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
    """Return the number of combinations of fine 2-face triangulations.

    This is the product, over the 2-faces of the polytope, of the number of
    fine triangulations of each 2-face. It is an upper bound on the number of
    NTFEs, since not every combination extends to a triangulation of the full
    polytope (see [`ntfe_frts`][cytools.ntfe.ntfe.ntfe_frts]). All
    triangulations of every 2-face are enumerated, which can be slow for large
    2-faces.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    methods `n_2face_triangs` and `num_2face_triangs`, where `self` is the
    polytope.

    Args:
        only_regular (bool): Whether to only count regular 2-face
            triangulations.

    Returns:
        int: The number of combinations of fine 2-face triangulations.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> p.n_2face_triangs()
        36
    """
    triangs = self.face_triangs(dim=2, only_regular=only_regular)
    return math.prod([len(f) for f in triangs])


Polytope.n_2face_triangs = n_2face_triangs
Polytope.num_2face_triangs = n_2face_triangs


def _as_2d_poly(poly: "Polytope", verbosity: int = 0) -> "Polytope":
    r"""Return a representation of a 2D polytope in a 2D ambient lattice.

    The point labels are preserved. This matters because the labels of 2-face
    triangulations are used as column indices into the height space of the
    ambient (4D) polytope; see `cytools.ntfe.ntfe._2d_frt_cone_ineqs`.

    Args:
        poly (Polytope): The polytope of interest. It is assumed to be
            2-dimensional.
        verbosity (int): The verbosity level.

    Returns:
        Polytope: `poly` itself if its ambient lattice is already 2D.
            Otherwise, an equivalent polytope in $\mathbb{Z}^2$ (built from
            the optimal coordinates of the points) carrying the same labels.
    """
    if poly.ambient_dim() == 2:
        return poly

    if verbosity >= 1:
        print("Finding a 2D representation!")

    # NOTE: labels=poly.labels is essential. Without it the new Polytope gets
    # fresh 0..n-1 labels of a re-sorted point list, which silently breaks
    # every downstream consumer that treats labels as ambient-polytope indices.
    return Polytope(poly.points(optimal=True), labels=poly.labels)


def grow_ft(
    self,
    bdry: Iterable[Iterable[int]] = None,
    seed: int = None,
    verbosity: int = 0,
) -> "Triangulation":
    r"""Randomly grow a fine triangulation (FT) of a polygon.

    Starting from a random unimodular triangle (one with no lattice points
    other than its vertices), unimodular triangles are repeatedly glued onto
    random non-boundary edges until no such edges are left. Since every
    triangle is unimodular, the result is fine. It need not be regular; see
    [`grow_frt`][cytools.ntfe.face_triangulations.grow_frt] for that.

    If the polytope does not live in a 2D ambient lattice, the triangulation is
    built on an equivalent polytope in $\mathbb{Z}^2$ with the same point
    labels.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `grow_ft`, where `self` is the polygon.

    Args:
        bdry (Iterable[Iterable[int]] | None): The boundary edges of the
            polygon, each given as an iterable of the labels of its two
            endpoints (as returned by `Polytope.get_bdry`). If not specified,
            they are computed.
        seed (int | None): The seed for the random number generator. If not
            specified, it is derived from the current time.
        verbosity (int): The verbosity level.

    Returns:
        Triangulation | None: The fine triangulation, or `None` (with a
            warning) if the growth got stuck.

    Raises:
        NotImplementedError: If the polytope is not 2-dimensional.

    Examples:
        We grow a fine triangulation of one of the 2-faces of a polytope.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> f = p.faces(2)[19].as_poly()
        >>> t = f.grow_ft(seed=0)
        >>> t.simplices()
        array([[ 3,  7, 12],
               [ 4,  9, 12],
               [ 7,  9, 12]], dtype=uint8)
        >>> t.is_fine()
        True
    """
    if seed is None:
        seed = time.time_ns() % (2**32)
    t0 = time.perf_counter()

    # get random number generator
    rand_gen = np.random.Generator(np.random.PCG64(seed=seed))

    # dimension checking
    if self.dim() != 2:
        raise NotImplementedError

    poly = _as_2d_poly(self, verbosity=verbosity)

    # basic point info...
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Grabbing basic point info...")

    # work with point *indices* internally (they index `pts` directly), then
    # translate back to labels at the very end
    pts = poly.points()
    labels = list(poly.labels)
    label2ind = {label: i for i, label in enumerate(labels)}
    pts_i = list(range(len(pts)))

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

    # `bdry` is expressed in labels; the loop below works in point indices
    bdry = {frozenset(label2ind[label] for label in edge) for edge in bdry}

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
                # growth got stuck; the partial set of simplices is not a
                # triangulation, so there is nothing meaningful to return
                warnings.warn(
                    "grow_ft: growth got stuck (no valid vertex to build off "
                    f"of edge {sorted(edge)}). Returning None."
                )
                return None

            if verbosity >= 3:
                print(
                    f"Trying new vertex {i:03d}... ({len(to_try):03d} left) -> ",
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

    # map point indices back to labels
    simps_labeled = [[labels[i] for i in s] for s in sorted(simps)]

    return poly.triangulate(
        simplices=simps_labeled,
        include_points_interior_to_facets=True,
        check_input_simplices=False,
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
) -> Union["Triangulation", set]:
    """Randomly grow fine, regular triangulations (FRTs) of a polygon.

    Fine triangulations are repeatedly grown with
    [`grow_ft`][cytools.ntfe.face_triangulations.grow_ft] (incrementing the
    seed after each attempt), and the regular ones are kept, until `N`
    distinct FRTs are found or `max_N_tries` attempts have been made.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as the
    method `grow_frt`, where `self` is the polygon.

    Args:
        N (int): The number of distinct FRTs to grow.
        max_N_tries (int | None): The maximum number of growth attempts. If not
            specified, `100*N` is used.
        bdry (Iterable[Iterable[int]] | None): The boundary edges of the
            polygon, each given as an iterable of the labels of its two
            endpoints (as returned by `Polytope.get_bdry`). If not specified,
            they are computed once and reused across attempts.
        seed (int | None): The seed for the random number generator. If not
            specified, it is derived from the current time.
        backend (str | None): The backend used to check regularity. See
            [`is_regular`][cytools.triangulation.Triangulation.is_regular].
        verbosity (int): The verbosity level.

    Returns:
        Triangulation | set[Triangulation]: If exactly one distinct FRT was
            found, that triangulation. Otherwise, the set of FRTs that were
            found, which contains at most `N` elements and may be empty.

    Raises:
        NotImplementedError: If the polytope is not 2-dimensional.

    Examples:
        We grow three distinct FRTs of one of the 2-faces of a polytope.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
        ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
        >>> f = p.faces(2)[19].as_poly()
        >>> frts = f.grow_frt(N=3, seed=0)
        >>> len(frts)
        3
    """
    # input checking
    if self.dim() != 2:
        raise NotImplementedError

    poly = _as_2d_poly(self, verbosity=verbosity)

    if max_N_tries is None:
        max_N_tries = 100 * N

    if seed is None:
        seed = time.time_ns() % (2**32)

    # compute the boundary once, rather than once per grow_ft attempt:
    # get_bdry triangulates the polygon and then pairs up its edges in
    # O(E^2), and grow_ft can be called up to 100*N times
    if bdry is None:
        bdry = basic_geometry.get_bdry(poly)

    frts = set()

    for N_attempt in range(max_N_tries):
        if verbosity >= 1:
            print(f"Attempt #{N_attempt}. Have #{len(frts)} FRTs")

        t = poly.grow_ft(bdry=bdry, seed=seed, verbosity=verbosity)
        seed += 1  # update the seed for next time

        if (t is not None) and t.is_regular(backend=backend):
            frts.add(t)

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
