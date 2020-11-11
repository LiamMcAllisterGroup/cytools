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

import numpy as np


class PolytopeFace:
    """A class that handles computations of faces of lattice polytopes."""

    def __init__(self, ambient_poly, vertices, saturated_ineqs, dim=None):
        """
        Creates a PolytopeFace object. These objects should not be directly
        constructed by the user.  Instead, they should be obtained from a
        Polytope object.

        Args:
            ambient_poly (Polytope): The ambient polytope.
            vertices (list): The list of vertices.
            saturated_ineqs (frozenset): A frozenset containing the indices of
                the inequalities that this face saturates.
            dim (int, optional): The dimension of the face. If it is not given then it is
                computed.
        """
        self._ambient_poly = ambient_poly
        self._vertices = np.array(vertices)
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
        """Clears the cached results of any previous computation."""
        self._points_sat = None
        self._points = None
        self._interior_points = None
        self._boundary_points = None
        self._polytope = None
        self._dual_face = None
        self._faces = None

    def __repr__(self):
        """Returns a string describing the face of the face."""
        return (f"A {self._dim}-dimensional face of a "
                f"{self._ambient_poly._dim}-dimensional polytope in "
                f"ZZ^{self._ambient_poly._ambient_dim}")

    def _points_saturated(self):
        if self._points_sat is not None:
            return np.array(self._points_sat)
        self._points_sat = [pt for pt in self._ambient_poly._points_saturated()
                            if self._saturated_ineqs.issubset(pt[1])]
        return np.array(self._points_sat)

    def points(self):
        """Returns the lattice points of the face."""
        if self._points is not None:
            return np.array(self._points)
        self._points = np.array(self._points_saturated()[:,0].tolist())
        return np.array(self._points)

    def interior_points(self):
        """Returns the lattice points in the relative interior of the face."""
        if self._interior_points is not None:
            return np.array(self._interior_points)
        self._interior_points = [pt[0] for pt in self._points_saturated()
                                 if len(pt[1]) == len(self._saturated_ineqs)]
        return np.array(self._interior_points)

    def boundary_points(self):
        """Returns the lattice points in the boundary of the face."""
        if self._boundary_points is not None:
            return np.array(self._boundary_points)
        self._boundary_points = [pt[0] for pt in self._points_saturated()
                                 if len(pt[1]) > len(self._saturated_ineqs)]
        return np.array(self._boundary_points)

    def as_polytope(self):
        """Returns the face as a Polytope object."""
        if self._polytope:
            return self._polytope
        from cytools.polytope import Polytope
        self._polytope = Polytope(self._vertices,
                                  backend=self._ambient_poly._backend)
        return self._polytope

    def ambient_polytope(self):
        """Returns the ambient polytope of the face."""
        return self._ambient_poly

    def dual(self):
        """Returns the dual face as a PolytopeFace object."""
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
        return self._dual_face

    def vertices(self):
        """Returns the vertices of the face."""
        return np.array(self._vertices)

    def faces(self, d=None):
        """Returns the faces of the face."""
        if self._faces is not None:
            if d is not None:
                return np.array(self._faces[d], dtype=object)
            return np.array(self._faces, dtype=object)
        faces = []
        for dd in range(self._dim + 1):
            faces.append([f for f in self._ambient_poly.faces(dd)
                        if self._saturated_ineqs.issubset(f._saturated_ineqs)])
        self._faces = faces
        if d is not None:
            return np.array(self._faces[d], dtype=object)
        return np.array(self._faces, dtype=object)

    def dim(self):
        """Returns the dimension of the face."""
        return self._dim

    def ambient_dim(self):
        """Returns the dimension of the ambient lattice."""
        return self._ambient_poly._ambient_dim
