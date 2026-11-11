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
# Description:  This module contains tools designed to perform polytope face
#               computations.
# -----------------------------------------------------------------------------

"""Faces of lattice polytopes.

This module defines the [`PolytopeFace`][cytools.polytopeface.PolytopeFace]
class, which describes a face of a lattice polytope. Faces are normally obtained
from [`Polytope.faces`][cytools.polytope.Polytope.faces] rather than constructed
directly.
"""

# 'standard' imports

# 3rd party imports
import numpy as np
from numpy.typing import ArrayLike

# CYTools imports
from cytools.triangulation import Triangulation
from cytools.utils import lll_reduce

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cytools.polytope import Polytope



class PolytopeFace:
    """A face of a lattice polytope.

    A face is described by its ambient polytope, the labels of its vertices, and
    the set of defining inequalities of the ambient polytope that it saturates.
    Its lattice points are the points of the ambient polytope that saturate all
    of these inequalities, and they keep the labels they have in the ambient
    polytope.

    Warning:
        Generally, objects of this class should not be constructed directly by
        the user. Instead, they should be obtained from
        [`Polytope.faces`][cytools.polytope.Polytope.faces].

    Args:
        ambient_poly (Polytope): The ambient polytope.
        vert_labels (list): The vertices of the face, specified by their labels
            in `ambient_poly`.
        saturated_ineqs (frozenset): The indices of the inequalities of
            `ambient_poly` (as returned by
            [`Polytope.inequalities`][cytools.polytope.Polytope.inequalities])
            that this face saturates.
        dim (int | None): The dimension of the face. If not specified, it is
            computed from the vertices.

    Examples:
        Since objects of this class should not be directly created by the end
        user, we demonstrate how to obtain them using
        [`Polytope.faces`][cytools.polytope.Polytope.faces].

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        >>> faces_3 = p.faces(3)  # find the 3-dimensional faces
        >>> print(faces_3[0])  # print the first 3-face
        A 3-dimensional face of a 4-dimensional polytope in ZZ^4
    """

    def __init__(
        self,
        ambient_poly: "Polytope",
        vert_labels: list,
        saturated_ineqs: frozenset,
        dim: int = None,
    ) -> None:
        # initialize attributes
        # ---------------------
        self.clear_cache()

        # process the inputs
        # ------------------
        self._ambient_poly = ambient_poly
        self._labels_vertices = vert_labels
        self._saturated_ineqs = saturated_ineqs

        # grab/compute optional inputs
        if dim is not None:
            self._dim = dim
        else:
            verts = ambient_poly.points(which=vert_labels)
            self._dim = np.linalg.matrix_rank([list(pt) + [1] for pt in verts]) - 1

    # defaults
    # ========
    def __repr__(self) -> str:
        """Return a string describing the face.

        Returns:
            str: A string describing the face.

        Examples:
            This function can be used to convert the face to a string or to
            print information about the face.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]
            >>> face_info = str(f)  # converts to string
            >>> print(f)  # prints face info
            A 3-dimensional face of a 4-dimensional polytope in ZZ^4
        """
        return (
            f"A {self.dim()}-dimensional face of a "
            f"{self.ambient_poly.dim()}-dimensional polytope in "
            f"ZZ^{self.ambient_dim()}"
        )

    # caching
    # =======
    def clear_cache(self) -> None:
        """Clear the cached results of any previous computation.

        This clears the cached point labels, the face as a
        [`Polytope`][cytools.polytope.Polytope], the dual face, and the
        subfaces. The ambient polytope, vertices, saturated inequalities, and
        dimension are kept.

        Examples:
            We construct a face object and find its lattice points, then we
            clear the cache and compute the points again.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> pts = f.points()  # find the lattice points
            >>> f.clear_cache()  # clears the results of any previous computation
            >>> pts = f.points()  # find the lattice points again
        """
        self._labels = None
        self._saturating = None
        self._labels_int = None
        self._labels_bdry = None
        self._polytope = None
        self._dual_face = None
        self._faces = None

    # getters
    # =======
    # (all methods here should be @property)
    @property
    def ambient_poly(self) -> "Polytope":
        """Return the ambient polytope.

        Returns:
            Polytope: The polytope that this is a face of.
        """
        return self._ambient_poly

    def ambient_polytope(self):
        """Return the ambient polytope.

        This is the method form of the `ambient_poly` property.

        Returns:
            Polytope: The polytope that this is a face of.
        """
        return self.ambient_poly

    @property
    def labels(self) -> tuple:
        """Return the labels of the lattice points of the face.

        The labels are those of the ambient polytope, and they are ordered as
        in the ambient polytope. The result is computed on first access and
        cached.

        Returns:
            tuple: The labels of the lattice points of the face.
        """
        if self._labels is None:
            self._process_points()

        return self._labels

    @property
    def labels_bdry(self) -> tuple:
        """Return the labels of the boundary lattice points of the face.

        These are the points of the face that lie on one of its proper faces
        (i.e. that saturate more inequalities of the ambient polytope than the
        face does).

        Returns:
            tuple: The labels of the boundary lattice points of the face.
        """
        if self._labels_bdry is None:
            self._process_points()

        return self._labels_bdry

    @property
    def labels_int(self) -> tuple:
        """Return the labels of the relative-interior lattice points of the face.

        These are the points of the face that saturate exactly the same
        inequalities of the ambient polytope as the face itself.

        Returns:
            tuple: The labels of the interior lattice points of the face.
        """
        if self._labels_int is None:
            self._process_points()

        return self._labels_int

    @property
    def labels_vertices(self) -> tuple:
        """Return the labels of the vertices of the face.

        Returns:
            list: The labels of the vertices of the face, as passed to the
                constructor.
        """
        return self._labels_vertices

    def dimension(self) -> int:
        """Return the dimension of the face.

        Also available as `dim`.

        Returns:
            int: The dimension of the face.
        """
        return self._dim

    # aliases
    dim = dimension

    def ambient_dimension(self) -> int:
        """Return the dimension of the ambient lattice.

        This is the ambient dimension of the ambient polytope, not the
        dimension of the ambient polytope itself.

        Also available as `ambient_dim`.

        Returns:
            int: The dimension of the ambient lattice.
        """
        return self.ambient_poly.ambient_dimension()

    # aliases
    ambient_dim = ambient_dimension

    # points
    # ======
    def _process_points(self) -> None:
        """Compute and cache the labels of the lattice points of the face.

        A point of the ambient polytope belongs to the face if it saturates
        every inequality that the face saturates. It is an interior point of
        the face if it saturates no others, and a boundary point otherwise. The
        results are stored in `_labels`, `_labels_int`, and `_labels_bdry`.
        """
        self._labels = []
        _saturating = []

        # inherit the calculation from the ambient polytope
        for label in self.ambient_poly.labels:
            saturating = self.ambient_poly._pts_saturating[label]

            if self._saturated_ineqs.issubset(saturating):
                self._labels.append(label)
                _saturating.append(saturating)

        # save it!
        self._labels = tuple(self._labels)

        # get interior, boundary points
        self._labels_int = []
        self._labels_bdry = []

        for label, sat in zip(self.labels, _saturating):
            if len(sat) == len(self._saturated_ineqs):
                self._labels_int.append(label)
            elif len(sat) > len(self._saturated_ineqs):
                self._labels_bdry.append(label)

        self._labels_int = tuple(self._labels_int)
        self._labels_bdry = tuple(self._labels_bdry)

    def points(
        self, which=None, optimal: bool = False, as_indices: bool = False
    ) -> np.ndarray:
        """Return the lattice points of the face.

        The points are ordered as in the ambient polytope (see
        [`Polytope.points`][cytools.polytope.Polytope.points]).

        Also available as `pts`.

        Args:
            which (list | None): Which points to return, specified by a list of
                labels (not indices). The labels must be labels of points of
                the face. If not specified, all points of the face are
                returned.
            optimal (bool): Whether to return the points in optimal
                coordinates. If the face is not full-dimensional in the ambient
                lattice, the points are translated so that the first returned
                point is at the origin and are then LLL-reduced to coordinates
                in a lattice of the same dimension as the face. Otherwise, the
                optimal coordinates of the ambient polytope are used.
            as_indices (bool): Whether to return the indices of the points in
                the full list of points of the ambient polytope instead of
                their coordinates. Takes precedence over `optimal`.

        Returns:
            np.ndarray | list[int]: The lattice points of the face, one per
                row, or a list of their indices if `as_indices` is true.

        Raises:
            ValueError: If `which` contains labels that are not labels of
                points of the face.

        Examples:
            We construct a face object and find its lattice points.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> f.points()
            array([[-1, -1, -1, -1],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0]])
            >>> f.points(optimal=True)
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]])
        """
        # get the labels of the relevant points
        if which is None:
            # use all points in the face
            which = self.labels
        else:
            # check if the input labels
            if not set(which).issubset(self.labels):
                raise ValueError(
                    f"Specified labels ({which}) aren't subset "
                    f"of the face labels ({self.labels})..."
                )

        # return
        if optimal and (not as_indices):
            dim_diff = self.ambient_dim() - self.dim()
            if dim_diff > 0:
                # asking for optimal points, where the optimal value may
                # differ from the entire polytope
                pts = self.points(which=which)
                return lll_reduce(pts - pts[0])[:, dim_diff:]

        # normal case
        return self.ambient_poly.points(
            which=which, optimal=optimal, as_indices=as_indices
        )

    # aliases
    pts = points

    def interior_points(self, as_indices: bool = False) -> ArrayLike:
        """Return the relative-interior lattice points of the face.

        See `labels_int` for the definition.

        Also available as `interior_pts`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points of the ambient polytope instead of
                their coordinates.

        Returns:
            np.ndarray | list[int]: The interior lattice points of the face, one
                per row, or a list of their indices if `as_indices` is true.

        Examples:
            We construct a face object and find its interior lattice points.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.faces(3)[3]  # pick one of the 3-faces
            >>> f.interior_points()
            array([[ 0,  0, -1, -2],
                   [ 0,  0,  0, -1]])
        """
        return self.ambient_poly.points(which=self.labels_int, as_indices=as_indices)

    # aliases
    interior_pts = interior_points

    def boundary_points(self, as_indices: bool = False) -> ArrayLike:
        """Return the boundary lattice points of the face.

        See `labels_bdry` for the definition.

        Also available as `boundary_pts`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points of the ambient polytope instead of
                their coordinates.

        Returns:
            np.ndarray | list[int]: The boundary lattice points of the face, one
                per row, or a list of their indices if `as_indices` is true.

        Examples:
            We construct a face object and find its boundary lattice points.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> f.boundary_points()
            array([[-1, -1, -1, -1],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0]])
        """
        return self.ambient_poly.points(which=self.labels_bdry, as_indices=as_indices)

    # aliases
    boundary_pts = boundary_points

    def vertices(self, as_indices: bool = False) -> ArrayLike:
        """Return the vertices of the face.

        Args:
            as_indices (bool): Whether to return the indices of the vertices in
                the full list of points of the ambient polytope instead of
                their coordinates.

        Returns:
            np.ndarray | list[int]: The vertices of the face, one per row, or a
                list of their indices if `as_indices` is true.

        Examples:
            We construct a face from a polytope and find its vertices.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(2)[0]  # pick one of the 2-faces
            >>> f.vertices()
            array([[-1, -1, -1, -1],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0]])
        """
        return self.ambient_poly.points(
            which=self._labels_vertices, as_indices=as_indices
        )

    # polytope
    # ========
    def as_polytope(self) -> "Polytope":
        """Return the face as a [`Polytope`][cytools.polytope.Polytope] object.

        The new polytope is built from the lattice points of the face (in the
        coordinates of the ambient lattice), keeps the labels that the points
        have in the ambient polytope, and uses the same backend as the ambient
        polytope. The result is cached.

        Also available as `as_poly`.

        Returns:
            Polytope: The polytope corresponding to the face.

        Examples:
            We construct a face object and then convert it into a
            [`Polytope`][cytools.polytope.Polytope] object.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> f_poly = f.as_polytope()
            >>> print(f_poly)
            A 3-dimensional lattice polytope in ZZ^4
        """
        if self._polytope is None:
            from cytools.polytope import Polytope

            self._polytope = Polytope(
                self.points(),
                labels=self.labels,
                backend=self.ambient_poly.backend,
            )
        return self._polytope

    # alias
    as_poly = as_polytope

    # dual
    # ====
    def dual_face(self) -> "PolytopeFace":
        """Return the dual face of the dual polytope.

        For a $k$-dimensional face of a $d$-dimensional reflexive polytope,
        the dual face is the $(d-k-1)$-dimensional face of the dual polytope
        whose vertices are the normals of the inequalities saturated by this
        face. The result is cached, and the dual face's own dual face is set
        to this face.

        Also available as `dual`.

        Returns:
            PolytopeFace: The dual face.

        Raises:
            NotImplementedError: If the ambient polytope is not reflexive.

        Note:
            This duality is only implemented for reflexive polytopes.

        Examples:
            We construct a face object from a polytope, then find the dual face
            in the dual polytope.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(2)[0]  # pick one of the 2-faces
            >>> f_dual = f.dual_face()
            >>> print(f_dual)
            A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        """
        # return answer, if known
        if self._dual_face is not None:
            return self._dual_face

        # check if calculation makes sense
        if not self.ambient_poly.is_reflexive():
            raise NotImplementedError("Ambient polytope is not reflexive.")

        # perform the calculation
        dual_poly = self.ambient_poly.dual()

        dual_vert = self.ambient_poly.inequalities()[list(self._saturated_ineqs), :-1]
        # index the dual inequalities by their coefficients, so that the
        # lookup below is O(V+F) rather than O(V*F)
        dual_ineq_inds = {}
        for i, v in enumerate(dual_poly.inequalities()[:, :-1].tolist()):
            dual_ineq_inds.setdefault(tuple(v), i)
        dual_saturated_ineqs = frozenset(
            [dual_ineq_inds[tuple(v)] for v in self.vertices().tolist()]
        )
        dual_face_dim = self.ambient_poly._dim - self._dim - 1
        self._dual_face = PolytopeFace(
            dual_poly,
            dual_poly.points_to_labels(dual_vert),
            dual_saturated_ineqs,
            dim=dual_face_dim,
        )
        self._dual_face._dual_face = self

        # return
        return self.dual_face()

    # aliases
    dual = dual_face

    # faces
    # =====
    def faces(self, d: int = None) -> tuple:
        """Return the faces of the face.

        The faces are obtained from
        [`Polytope.faces`][cytools.polytope.Polytope.faces] of the ambient
        polytope, so they are faces of the ambient polytope that are contained
        in this face (including the face itself). The result is cached.

        Args:
            d (int | None): The dimension of the desired faces. If not
                specified, the faces of all dimensions are returned.

        Returns:
            tuple: A tuple of [`PolytopeFace`][cytools.polytopeface.PolytopeFace]
                objects of dimension `d` if `d` is specified. Otherwise, a
                tuple of such tuples, where the $i$-th entry contains the
                $i$-dimensional faces, for $i$ from 0 to the dimension of the
                face.

        Raises:
            ValueError: If `d` is not between 0 and the dimension of the face.

        Examples:
            We construct a face from a polytope and find its 2-faces.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> print(f.faces(2)[0])  # print one of its 2-faces
            A 2-dimensional face of a 4-dimensional polytope in ZZ^4
            >>> [len(ff) for ff in f.faces()]  # number of faces of each dimension
            [4, 6, 4, 1]
        """
        # input checking
        if (d is not None) and (d not in range(self._dim + 1)):
            raise ValueError(f"There are no faces of dimension {d}")

        # return answer if known
        if self._faces is not None:
            return self._faces[d] if d is not None else self._faces

        # calculate the answer
        faces = [tuple(f for f in self.ambient_poly.faces(dd)
                       if self._saturated_ineqs.issubset(f._saturated_ineqs))
                 for dd in range(self._dim + 1)]
        self._faces = tuple(faces)

        # return
        return self.faces(d)

    # triangulating
    # =============
    def triangulate(
        self,
        heights: list = None,
        simplices: ArrayLike = None,
        check_input_simplices: bool = True,
        backend: str = "cgal",
        verbosity=0,
        seed: int = None,
    ) -> "Triangulation":
        """Return a single regular triangulation of the face.

        This is a simple wrapper around the
        [`Triangulation`][cytools.triangulation.Triangulation] constructor that
        triangulates all lattice points of the face (including points interior
        to facets of the ambient polytope) and never makes it star. See also
        [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate].

        Args:
            heights (list | None): The heights specifying the regular
                triangulation, one per lattice point of the face, in the order
                of `labels`. If not specified, the Delaunay triangulation is
                returned when using CGAL, a triangulation obtained from random
                heights near the Delaunay heights when using QHull, or the
                placing triangulation when using TOPCOM. Heights can only be
                specified when using CGAL or QHull as the backend.
            simplices (ArrayLike | None): The simplices specifying the
                triangulation, each given as a list of point labels. This is
                useful when a triangulation was previously computed and needs
                to be used again.
            check_input_simplices (bool): Whether to check if the input
                simplices define a valid triangulation.
            backend (str): The backend used to compute the triangulation. The
                options are `"cgal"`, `"qhull"`, and `"topcom"`. CGAL is the
                default as it is very fast and robust.
            verbosity (int): The verbosity level.
            seed (int | None): Seed for the height perturbation used by the
                QHull backend. Only that backend perturbs its heights, so this
                has no effect for CGAL or TOPCOM. If not specified, the
                perturbation is drawn afresh each time and the triangulation is
                not reproducible.

        Returns:
            Triangulation: A triangulation of the lattice points of the face.

        Examples:
            We triangulate one of the 3-faces of a polytope.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces(3)[0]  # pick one of the 3-faces
            >>> t = f.triangulate()
            >>> t.simplices()
            array([[1, 2, 3, 4]], dtype=uint8)
        """
        return Triangulation(
            self.ambient_poly,
            self.labels,
            make_star=False,
            heights=heights,
            simplices=simplices,
            check_input_simplices=check_input_simplices,
            backend=backend,
            verbosity=verbosity,
            seed=seed,
        )
