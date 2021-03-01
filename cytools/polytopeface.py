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
    user. Instead, they should be created by the [faces](./polytope#faces)
    function of the [Polytope](./polytope) class.
    :::

    ## Constructor

    ### ```cytools.polytope.PolytopeFace```

    **Description:**
    Constructs a ```PolytopeFace``` object describing a face of a lattice
    polytope. This is handled by the hidden [```__init__```](#__init__)
    function.

    **Arguments:**
    - ```ambient_poly``` (Polytope): The ambient polytope.
    - ```vertices``` (list): The list of vertices.
    - ```saturated_ineqs``` (frozenset): A frozenset containing the indices of
      the inequalities that this face saturates.
    - ```dim``` (integer, optional): The dimension of the face. If it is not
      given then it is computed.
    """

    def __init__(self, ambient_poly, vertices, saturated_ineqs, dim=None):
        """
        **Description:**
        Initializes a ```PolytopeFace``` object.

        **Arguments:**
        - ```ambient_poly``` (Polytope): The ambient polytope.
        - ```vertices``` (list): The list of vertices.
        - ```saturated_ineqs``` (frozenset): A frozenset containing the indices
          of the inequalities that this face saturates.
        - ```dim``` (integer, optional): The dimension of the face. If it is
          not given then it is computed.

        **Returns:**
        Nothing.
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
        (string) A string describing the face.
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
          [_points_saturated](./polytope#_points_saturated) function of the
          [Polytope](./polytope) class.
        - Typically this function should not be called by the user. Instead, it
          is called by various other functions in the PolytopeFace class.
        :::

        **Arguments:**
        None.

        **Returns:**
        (list) A list of tuples. The first component of each tuple is the list
        of coordinates of the point and the second component is a
        ```frozenset``` of the hyperplane inequalities that it saturates.
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
        - ```as_indices``` (boolean): Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        (list) The list of lattice points of the face.
        """
        if self._points is None:
            self._points = np.array([pt[0] for pt in self._points_saturated()])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._points)
        return np.array(self._points)

    def interior_points(self, as_indices=False):
        """
        **Description:**
        Returns the interior lattice points of the face.

        **Arguments:**
        - ```as_indices``` (boolean): Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        (list) The list of interior lattice points of the face.
        """
        if self._interior_points is None:
            self._interior_points = np.array(
                                    [pt[0] for pt in self._points_saturated()
                                    if len(pt[1])==len(self._saturated_ineqs)])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._interior_points)
        return np.array(self._interior_points)

    def boundary_points(self, as_indices=False):
        """
        **Description:**
        Returns the boundary lattice points of the face.

        **Arguments:**
        - ```as_indices``` (boolean): Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        (list) The list of boundary lattice points of the face.
         """
        if self._boundary_points is None:
            self._boundary_points = np.array(
                                    [pt[0] for pt in self._points_saturated()
                                    if len(pt[1])>len(self._saturated_ineqs)])
        if as_indices:
            return self._ambient_poly.points_to_indices(self._boundary_points)
        return np.array(self._boundary_points)

    def as_polytope(self):
        """
        **Description:**
        Returns the face as a Polytope object.

        **Arguments:**
        None.

        **Returns:**
        (Polytope) The [Polytope](./polytope) corresponding to the face.
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
        (Polytope) The ambient polytope.
        """
        return self._ambient_poly

    def dual(self):
        """
        **Description:**
        Returns the dual face in the ambient polytope.

        :::note
        This duality is only implemented for reflexive polytopes. An exception
        is raised if the polytope is not reflexive.
        :::

        **Arguments:**
        None.

        **Returns:**
        (PolytopeFace) The dual face.
        """
        if self._dual_face is not None:
            return self._dual_face
        if not self._ambient_poly.is_reflexive():
            raise Exception("Ambient polytope is not reflexive.")
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

    def vertices(self):
        """
        **Description:**
        Returns the vertices of the face.

        **Arguments:**
        None.

        **Returns:**
        (list) The list of vertices of the face.
        """
        return np.array(self._vertices)

    def faces(self, d=None):
        """
        **Description:**
        Computes the faces of the face.

        **Arguments:**
        - ```d``` (integer, optional): Optional parameter that specifies the
          dimension of the desired faces.

        **Returns:**
        (list) A list of [```PolytopeFace```](./polytopeface) objects of
        dimension d, if specified. Otherwise, a list of lists of
        [```PolytopeFace```](./polytopeface) objects organized in ascending
        dimension.
        """
        if d is not None and d not in range(self._dim + 1):
            raise Exception(f"There are no faces of dimension {d}")
        if self._faces is not None:
            return copy.copy(self._faces[d] if d is not None else
                                [copy.copy(ff) for ff in self._faces])
        faces = []
        for dd in range(self._dim + 1):
            faces.append([f for f in self._ambient_poly.faces(dd)
                        if self._saturated_ineqs.issubset(f._saturated_ineqs)])
        self._faces = faces
        return copy.copy(self._faces[d] if d is not None else
                            [copy.copy(ff) for ff in self._faces])

    def dim(self):
        """
        **Description:**
        Returns the dimension of the face.

        **Arguments:**
        None.

        **Returns:**
        (integer) The dimension of the face.
        """
        return self._dim

    def ambient_dim(self):
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        (integer) The dimension of the ambient lattice.
        """
        return self._ambient_dim
