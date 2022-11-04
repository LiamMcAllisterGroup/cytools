# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CYTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CYTools.  If not, see <https://www.gnu.org/licenses/>.

"""
This module contains tools designed to perform polytope face computations.
"""

# Standard imports
import copy
# Third party imports
import numpy as np



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
    Constructs a `PolytopeFace` object describing a face of a lattice
    polytope. This is handled by the hidden [`__init__`](#__init__)
    function.

    **Arguments:**
    - `ambient_poly` *(Polytope)*: The ambient polytope.
    - `vertices` *(array_like)*: The list of vertices.
    - `saturated_ineqs` *(frozenset)*: A frozenset containing the indices
      of the inequalities that this face saturates.
    - `dim` *(int, optional)*: The dimension of the face. If it is not
      given then it is computed.

    **Example:**
    Since objects of this class should not be directly created by the end user,
    we demostrate how to construct these objects using the
    [`faces`](./polytope#faces) function of the
    [`Polytope`](./polytope) class.
    ```python {3}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    faces_3 = p.faces(3) # Find the 3-dimensional faces
    print(faces_3[0]) # Print the first 3-face
    # A 3-dimensional face of a 4-dimensional polytope in ZZ^4
    ```
    """

    def __init__(self, ambient_poly, vertices, saturated_ineqs, dim=None):
        """
        **Description:**
        Initializes a `PolytopeFace` object.

        **Arguments:**
        - `ambient_poly` *(Polytope)*: The ambient polytope.
        - `vertices` *(array_like)*: The list of vertices.
        - `saturated_ineqs` *(frozenset)*: A frozenset containing the
          indices of the inequalities that this face saturates.
        - `dim` *(int, optional)*: The dimension of the face. If it is
          not given then it is computed.

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
        self._ambient_poly = ambient_poly
        self._vertices = np.array(vertices)
        self._ambient_dim = self._ambient_poly.ambient_dim()
        self._saturated_ineqs = saturated_ineqs
        if dim is not None:
            self._dim = dim
        else:
            vert_ext = np.empty((self._vertices.shape[0],
                                 self._vertices.shape[1]+1), dtype=int)
            vert_ext[:,:-1] = self._vertices
            vert_ext[:,-1] = 1
            self._dim = np.linalg.matrix_rank(vert_ext)-1
        # Initialize remaining hidden attributes
        self._points_sat = None
        self._points = None
        self._interior_points = None
        self._boundary_points = None
        self._polytope = None
        self._dual_face = None
        self._faces = None

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We construct a face object and find its lattice points, then we
        clear the cache and compute the points again.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        pts = f.points() # Find the lattice points
        f.clear_cache() # Clears the results of any previos computation
        pts = f.points() # Find the lattice points again
        ```
        """
        self._points_sat = None
        self._points = None
        self._interior_points = None
        self._boundary_points = None
        self._polytope = None
        self._dual_face = None
        self._faces = None

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the face.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the face.

        **Example:**
        This function can be used to convert the face to a string or to
        print information about the face.
        ```python {3,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0]
        face_info = str(f) # Converts to string
        print(f) # Prints face info
        ```
        """
        return (f"A {self._dim}-dimensional face of a "
                f"{self._ambient_poly._dim}-dimensional polytope in "
                f"ZZ^{self._ambient_dim}")

    def _points_saturated(self):
        """
        **Description:**
        Computes the lattice points of the face along with the indices of
        the hyperplane inequalities that they saturate.

        :::note notes
        - Points are sorted in the same way as for the
          [`_points_saturated`](./polytope#_points_saturated) function of the
          [`Polytope`](./polytope) class.
        - Typically this function should not be called by the user. Instead, it
          is called by various other functions in the PolytopeFace class.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(list)* A list of tuples. The first component of each tuple is the list
        of coordinates of the point and the second component is a
        `frozenset` of the hyperplane inequalities that it saturates.

        **Example:**
        We construct a face and compute the lattice points along with the
        inequalities that they saturate. We print the second point and the
        inequalities that it saturates.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0]
        pts_sat = f._points_saturated()
        print(pts_sat[1])
        # ((0, 0, 0, 1), frozenset({0, 1, 2, 4}))
        p.inequalities()[list(pts_sat[1][1])]
        # array([[ 4, -1, -1, -1,  1],
        #        [-1,  4, -1, -1,  1],
        #        [-1, -1,  4, -1,  1],
        #        [-1, -1, -1,  4,  1]])
        ```
        """
        if self._points_sat is None:
            self._points_sat = [
                            pt for pt in self._ambient_poly._points_saturated()
                            if self._saturated_ineqs.issubset(pt[1])]
        return copy.copy(self._points_sat)

    def points(self, as_indices=False):
        """
        **Description:**
        Returns the lattice points of the face.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

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
        if self._points is None:
            self._points = np.array([pt[0] for pt in self._points_saturated()])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._points)
        return np.array(self._points)
    # Aliases
    pts = points

    def interior_points(self, as_indices=False):
        """
        **Description:**
        Returns the interior lattice points of the face.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of interior lattice points of the face.

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
        if self._interior_points is None:
            self._interior_points = np.array(
                                    [pt[0] for pt in self._points_saturated()
                                    if len(pt[1])==len(self._saturated_ineqs)])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._interior_points)
        return np.array(self._interior_points)
    # Aliases
    interior_pts = interior_points

    def boundary_points(self, as_indices=False):
        """
        **Description:**
        Returns the boundary lattice points of the face.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of boundary lattice points of the face.

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
        if self._boundary_points is None:
            self._boundary_points = np.array(
                                    [pt[0] for pt in self._points_saturated()
                                    if len(pt[1])>len(self._saturated_ineqs)])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._boundary_points)
        return np.array(self._boundary_points)
    # Aliases
    boundary_pts = boundary_points

    def as_polytope(self):
        """
        **Description:**
        Returns the face as a Polytope object.

        **Arguments:**
        None.

        **Returns:**
        *(Polytope)* The [`Polytope`](./polytope) corresponding to the
        face.

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
            self._polytope = Polytope(self._vertices,
                                      backend=self._ambient_poly._backend)
        return self._polytope

    def ambient_polytope(self):
        """
        **Description:**
        Returns the ambient polytope of the face.

        **Arguments:**
        None.

        **Returns:**
        *(Polytope)* The ambient polytope.

        **Example:**
        We construct a face object from a polytope, then find the ambient
        polytope and verify that it is the starting polytope.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        ambient_poly = f.ambient_polytope()
        ambient_poly is p
        # True
        ```
        """
        return self._ambient_poly

    def dual_face(self):
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
        *(PolytopeFace)* The dual face.

        **Aliases:**
        `dual`.

        **Example:**
        We construct a face object from a polytope, then find the dual face
        in the dual polytope.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(2)[0] # Pick one of the 2-faces
        f_dual = f.dual_face()
        print(f_dual)
        # A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
        """
        if self._dual_face is not None:
            return self._dual_face
        if not self._ambient_poly.is_reflexive():
            raise NotImplementedError("Ambient polytope is not reflexive.")
        dual_vert = self._ambient_poly._input_ineqs[
                                            list(self._saturated_ineqs),:-1]
        dual_poly = self._ambient_poly.dual()
        dual_ineqs = dual_poly._input_ineqs[:,:-1].tolist()
        dual_saturated_ineqs = frozenset([dual_ineqs.index(v)
                                            for v in self._vertices.tolist()])
        dual_face_dim = self._ambient_poly._dim - self._dim - 1
        self._dual_face = PolytopeFace(dual_poly, dual_vert,
                                       dual_saturated_ineqs, dim=dual_face_dim)
        self._dual_face._dual_face = self
        return self._dual_face
    # Aliases
    dual = dual_face

    def vertices(self):
        """
        **Description:**
        Returns the vertices of the face.

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The list of vertices of the face.

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
        return np.array(self._vertices)

    def faces(self, d=None):
        """
        **Description:**
        Computes the faces of the face.

        **Arguments:**
        - `d` *(int, optional)*: Optional parameter that specifies the
          dimension of the desired faces.

        **Returns:**
        *(tuple)* A tuple of [`PolytopeFace`](./polytopeface) objects of
        dimension d, if specified. Otherwise, a tuple of tuples of
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
        if d is not None and d not in range(self._dim + 1):
            raise ValueError(f"There are no faces of dimension {d}")
        if self._faces is not None:
            return (self._faces[d] if d is not None else self._faces)
        faces = []
        for dd in range(self._dim + 1):
            faces.append(tuple(f for f in self._ambient_poly.faces(dd)
                        if self._saturated_ineqs.issubset(f._saturated_ineqs)))
        self._faces = tuple(faces)
        return (self._faces[d] if d is not None else self._faces)

    def dimension(self):
        """
        **Description:**
        Returns the dimension of the face.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the face.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a face from a polytope and print its dimension.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        f.dimension()
        # 3
        ```
        """
        return self._dim
    # Aliases
    dim = dimension

    def ambient_dimension(self):
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the ambient lattice.

        **Aliases:**
        `ambient_dim`.

        **Example:**
        We construct a face from a polytope and print its ambient dimension.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        f = p.faces(3)[0] # Pick one of the 3-faces
        f.ambient_dimension()
        # 4
        ```
        """
        return self._ambient_dim
    # Aliases
    ambient_dim = ambient_dimension
