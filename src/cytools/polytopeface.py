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

# 'standard' imports
import copy

# 3rd party imports
import numpy as np
from numpy.typing import ArrayLike

# CYTools imports
from cytools.triangulation import Triangulation
from cytools.utils import lll_reduce


class PolytopeFace:
    """
    This class handles all computations relating to faces of lattice polytopes.

    :::important
    Generally, objects of this class should not be constructed directly by the
    user. Instead, they should be created by the [`faces`](./polytope#faces)
    function of the [`Polytope`](./polytope) class.
    :::

    ## Constructor

    ### `cytools.polytopeface.PolytopeFace`

    **Description:**
    Constructs a `PolytopeFace` object describing a face of a lattice polytope.
    This is handled by the hidden [`__init__`](#__init__) function.

    **Arguments:**
    - `ambient_poly` *(Polytope)*: The ambient polytope.
    - `vertices` *(array_like)*: The list of vertices.
    - `saturated_ineqs` *(frozenset)*: A frozenset containing the indices of
        the inequalities that this face saturates.
    - `dim` *(int, optional)*: The dimension of the face. If it is not given
        then it is computed.

    **Example:**
    Since objects of this class should not be directly created by the end user,
    we demonstrate how to construct these objects using the
    [`faces`](./polytope#faces) function of the [`Polytope`](./polytope) class.
    ```python {3}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    faces_3 = p.faces(3) # Find the 3-dimensional faces
    print(faces_3[0]) # Print the first 3-face
    # A 3-dimensional face of a 4-dimensional polytope in ZZ^4
    ```
    """

    def __init__(
        self,
        ambient_poly: "Polytope",
        vert_labels: list,
        saturated_ineqs: frozenset,
        dim: int = None,
    ) -> None:
        """
        **Description:**
        Initializes a `PolytopeFace` object.

        **Arguments:**
        - `ambient_poly`: The ambient polytope.
        - `vert_labels`: The vertices, specified by labels in ambient_poly.
        - `saturated_ineqs`: Indices of inequalities that this face saturates.
        - `dim`: The dimension of this face. If not given, then it's computed.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new
        `PolytopeFace` object. Thus, it is used in the following example.
        ```python {3}
        from cytools import Polytope
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        faces_3 = p.faces(3) # Find the 3-dimensional faces
        print(faces_3[0]) # Print the first 3-face
        # A 3-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
        """
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
        """
        **Description:**
        Returns a string describing the face.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the face.

        **Example:**
        This function can be used to convert the face to a string or to print
        information about the face.
        ```python {3,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0]
        face_info = str(f) # Converts to string
        print(f) # Prints face info
        ```
        """
        return (
            f"A {self.dim()}-dimensional face of a "
            f"{self.ambient_poly.dim()}-dimensional polytope in "
            f"ZZ^{self.ambient_dim()}"
        )

    # caching
    # =======
    def clear_cache(self) -> None:
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We construct a face object and find its lattice points, then we clear
        the cache and compute the points again.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        pts = f.points() # Find the lattice points
        f.clear_cache() # Clears the results of any previous computation
        pts = f.points() # Find the lattice points again
        ```
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
        """
        **Description:**
        Returns the ambient polytope.

        **Arguments:**
        None.

        **Returns:**
        The ambient polytope.
        """
        return self._ambient_poly

    ambient_polytope = lambda self: self.ambient_poly

    @property
    def labels(self) -> tuple:
        """
        **Description:**
        Returns the labels of lattice points in the face.

        **Arguments:**
        None.

        **Returns:**
        The labels of lattice points in the face.
        """
        if self._labels is None:
            self._process_points()

        return self._labels

    @property
    def labels_bdry(self) -> tuple:
        """
        **Description:**
        Returns the labels of boundary lattice points in the face.

        **Arguments:**
        None.

        **Returns:**
        The labels of boundary lattice points in the face.
        """
        if self._labels_bdry is None:
            self._process_points()

        return self._labels_bdry

    @property
    def labels_int(self) -> tuple:
        """
        **Description:**
        Returns the labels of interior lattice points in the face.

        **Arguments:**
        None.

        **Returns:**
        The labels of interior lattice points in the face.
        """
        if self._labels_int is None:
            self._process_points()

        return self._labels_int

    @property
    def labels_vertices(self) -> tuple:
        """
        **Description:**
        Returns the labels of vertices in the face.

        **Arguments:**
        None.

        **Returns:**
        The labels of vertices in the face.
        """
        return self._labels_vertices

    def dimension(self) -> int:
        """
        **Description:**
        Returns the dimension of the face.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the face.

        **Aliases:**
        `dim`.
        """
        return self._dim

    # aliases
    dim = dimension

    def ambient_dimension(self) -> int:
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        The dimension of the ambient lattice.

        **Aliases:**
        `ambient_dim`.
        """
        return self.ambient_poly.ambient_dimension()

    # aliases
    ambient_dim = ambient_dimension

    # points
    # ======
    def _process_points(self) -> None:
        """
        **Description:**
        Grabs the labels of the lattice points of the face along with the
        indices of the hyperplane inequalities that they saturate.

        **Arguments:**
        None.

        **Returns:**
        Nothing.
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
        """
        **Description:**
        Returns the lattice points of the face.

        **Arguments:**
        - `as_indices`: Return the points as indices of the full list of
            points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of lattice points of the face.

        **Aliases:**
        `pts`.

        **Example:**
        We construct a face object and find its lattice points.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        f.points()
        # array([[-1, -1, -1, -1],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0]])
        ```
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
        """
        **Description:**
        Returns the interior lattice points of the face.

        **Arguments:**
        - `as_indices`: Return the points as indices of the full list of
            points of the polytope.

        **Returns:**
        The list of interior lattice points of the face.

        **Aliases:**
        `interior_pts`.

        **Example:**
        We construct a face object and find its interior lattice points.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        f = p.faces(3)[2] # Pick one of the 3-faces
        f.interior_points()
        # array([[ 0,  0, -1, -2],
        #        [ 0,  0,  0, -1]])
        ```
        """
        return self.ambient_poly.points(which=self.labels_int, as_indices=as_indices)

    # aliases
    interior_pts = interior_points

    def boundary_points(self, as_indices: bool = False) -> ArrayLike:
        """
        **Description:**
        Returns the boundary lattice points of the face.

        **Arguments:**
        - `as_indices`: Return the points as indices of the full list of
            points of the polytope.

        **Returns:**
        The list of boundary lattice points of the face.

        **Aliases:**
        `boundary_pts`.

        **Example:**
        We construct a face object and find its boundary lattice points.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        f.boundary_points()
        # array([[-1, -1, -1, -1],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0]])
        ```
        """
        return self.ambient_poly.points(which=self.labels_bdry, as_indices=as_indices)

    # aliases
    boundary_pts = boundary_points

    def vertices(self, as_indices: bool = False) -> ArrayLike:
        """
        **Description:**
        Returns the vertices of the face.

        **Arguments:**
        None.

        **Returns:**
        The list of vertices of the face.

        **Example:**
        We construct a face from a polytope and find its vertices.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(2)[0] # Pick one of the 2-faces
        f.vertices()
        # array([[-1, -1, -1, -1],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0]])
        ```
        """
        return self.ambient_poly.points(
            which=self._labels_vertices, as_indices=as_indices
        )

    # polytope
    # ========
    def as_polytope(self) -> "Polytope":
        """
        **Description:**
        Returns the face as a Polytope object.

        **Arguments:**
        None.

        **Returns:**
        The [`Polytope`](./polytope) corresponding to the face.

        **Example:**
        We construct a face object and then convert it into a
        [`Polytope`](./polytope) object.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        f_poly = f.as_polytope()
        print(f_poly)
        # A 3-dimensional lattice polytope in ZZ^4
        ```
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
        """
        **Description:**
        Returns the dual face of the dual polytope.

        :::note
        This duality is only implemented for reflexive polytopes. An exception
        is raised if the polytope is not reflexive.
        :::

        **Arguments:**
        None.

        **Returns:**
        The dual face.

        **Aliases:**
        `dual`.

        **Example:**
        We construct a face object from a polytope, then find the dual face in
        the dual polytope.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(2)[0] # Pick one of the 2-faces
        f_dual = f.dual_face()
        print(f_dual)
        # A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
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
        dual_ineqs = dual_poly.inequalities()[:, :-1].tolist()
        dual_saturated_ineqs = frozenset(
            [dual_ineqs.index(v) for v in self.vertices().tolist()]
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
        """
        **Description:**
        Computes the faces of the face.

        **Arguments:**
        - `d`: Optional parameter that specifies the dimension of the desired
            faces.

        **Returns:**
        A tuple of [`PolytopeFace`](./polytopeface) objects of dimension d, if
        specified. Otherwise, a tuple of tuples of
        [`PolytopeFace`](./polytopeface) objects organized in ascending
        dimension.

        **Example:**
        We construct a face from a polytope and find its vertices.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        print(f.faces(2)[0]) # Print one of its 2-faces
        # A 2-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
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
    ) -> "Triangulation":
        """
        **Description:**
        Returns a single regular triangulation of the face.

        Just a simple wrapper for the Triangulation constructor.

        Also see Polytope.triangulate

        **Arguments:**
        - `heights`: A list of heights specifying the regular triangulation.
            When not specified, it will return the Delaunay triangulation when
            using CGAL, a triangulation obtained from random heights near the
            Delaunay when using QHull, or the placing triangulation when using
            TOPCOM. Heights can only be specified when using CGAL or QHull as
            the backend.
        - `simplices`: A list of simplices specifying the triangulation. This
            is useful when a triangulation was previously computed and it
            needs to be used again. Note that the order of the points needs to
            be consistent with the order that the `Polytope` class uses.
        - `check_input_simplices`: Flag that specifies whether to check if the
            input simplices define a valid triangulation.
        - `backend`: Specifies the backend used to compute the triangulation.
            The available options are "qhull", "cgal", and "topcom". CGAL is
            the default one as it is very fast and robust.
        - `verbosity`: The verbosity level.

        **Returns:**
        A [`Triangulation`](./triangulation) object describing a triangulation
        of the polytope.
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
        )
