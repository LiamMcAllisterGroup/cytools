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
This module contains tools designed to perform polytope computations.
"""

from cytools.triangulation import *
from cytools.polytopeface import PolytopeFace
from cytools.utils import gcd_list
from cytools.utils import remove_duplicate_triangulations
from cytools.cone import Cone
from scipy.spatial import ConvexHull
from collections import defaultdict
from itertools import permutations
from flint import fmpz_mat
import numpy as np
import copy
import math
import random
import sys
# PPL is an optional dependency. QHull provides only slightly reduced
# functionality, but it is faster.
try:
    import ppl
    HAVE_PPL = True
except:
    HAVE_PPL = False


class Polytope:
    """Class that handles lattice polytope computations."""

    def __init__(self, points, backend=None):
        """
        Creates a Polytope object.

        Args:
            points (list): A list of points. The polytope is defined by their
                convex hull.
            backend (string, optional): A string that specifies the backend
                used to construct the convex hull.  The available options are
                "ppl" or "qhull".  When not specified, it uses PPL if available
                and QHull otherwise.
        """
        # Convert points to numpy array and compute dimension
        self._input_pts = np.array(points, dtype=int)
        self._ambient_dim = self._input_pts.shape[1]
        pts_ext = np.empty((self._input_pts.shape[0],
                            self._input_pts.shape[1] + 1), dtype=int)
        pts_ext[:,:-1] = self._input_pts
        pts_ext[:,-1] = 1
        self._dim = np.linalg.matrix_rank(pts_ext) - 1
        self._dim_diff = self._ambient_dim - self._dim
        # Select backend for the computation of the convex hull
        backends = ["ppl", "qhull", None]
        if backend not in backends:
            raise Exception(f"Invalid backend. Options are {backends}.")
        if backend is None:
            if HAVE_PPL:
                backend = "ppl"
            else:
                backend = "qhull"
        self._backend = backend
        # Find optimal form of the polytope by performing LLL reduction.
        # If the polytope is not full-dimensional it constructs an
        # affinely-equivalent polytope in a lattice of matching dimension.
        # Internally it uses the optimal form for computations, but it outputs
        # everything in the same form as the input
        if self._dim == self._ambient_dim:
            pts_mat = fmpz_mat(self._input_pts.T.tolist())
            optimal_pts, transf_matrix = pts_mat.lll(transform=True)
            self._optimal_pts = np.array(optimal_pts.table(), dtype=int).T
        else:
            self._transl_vector = self._input_pts[0]
            tmp_pts = np.array(self._input_pts)
            for i in range(self._input_pts.shape[0]):
                tmp_pts[i] -= self._transl_vector
            pts_mat = fmpz_mat(tmp_pts.T.tolist())
            optimal_pts, transf_matrix = pts_mat.lll(transform=True)
            optimal_pts = np.array(optimal_pts.table(), dtype=int).T
            self._optimal_pts = optimal_pts[:, self._dim_diff:]
        self._transf_matrix = np.array(transf_matrix.table(), dtype=int)
        inv_tranf_matrix = transf_matrix.inv(integer=True)
        self._inv_transf_matrix = np.array(inv_tranf_matrix.table(), dtype=int)
        # Flint sometimes returns an inverse that is missing a factor of -1
        check_inverse = self._inv_transf_matrix.dot(self._transf_matrix)
        id_mat = np.eye(self._ambient_dim, dtype=int)
        if all((check_inverse == id_mat).flatten()):
            pass
        elif all((check_inverse == -id_mat).flatten()):
            self._inv_transf_matrix = -self._inv_transf_matrix
        else:
            raise Exception("Problem finding inverse matrix")
        # Construct convex hull and find the hyperplane representation with the
        # appropriate backend. The equations are in the form
        # c_0 * x_0 + ... + c_{d-1} * x_{d-1} + c_d >= 0
        if backend == "ppl":
            gs = ppl.Generator_System()
            vrs = [ppl.Variable(i) for i in range(self._dim)]
            for pt in self._optimal_pts:
                gs.insert(ppl.point(sum(pt[i]*vrs[i]
                                        for i in range(self._dim))))
            self._optimal_poly = ppl.C_Polyhedron(gs)
            optimal_ineqs = []
            for ineq in self._optimal_poly.minimized_constraints():
                optimal_ineqs.append(list(ineq.coefficients())
                                     + [ineq.inhomogeneous_term()])
            self._optimal_ineqs = np.array(optimal_ineqs, dtype=int)
        else:
            if self._dim == 0: # qhull cannot handle 0-dimensional polytopes
                self._optimal_ineqs = np.array([[0]])
            elif self._dim == 1: # qhull cannot handle 1-dimensional polytopes
                self._optimal_ineqs = np.array([[1, -min(self._optimal_pts)]
                                                ,[-1, max(self._optimal_pts)]])
            else:
                self._optimal_poly = ConvexHull(self._optimal_pts)
                tmp_ineqs = set()
                for eq in self._optimal_poly.equations:
                    g = abs(gcd_list(eq))
                    tmp_ineqs.add(tuple(-int(round(i/g)) for i in eq))
                self._optimal_ineqs = np.array(list(tmp_ineqs), dtype=int)
        if self._ambient_dim > self._dim:
            shape = (self._optimal_ineqs.shape[0],
                     self._optimal_ineqs.shape[1] + self._dim_diff)
            self._optimal_ineqs_ext = np.empty(shape, dtype=int)
            self._optimal_ineqs_ext[:,self._dim_diff:] = self._optimal_ineqs
            self._optimal_ineqs_ext[:,:self._dim_diff] = 0
            self._input_ineqs = np.empty(shape, dtype=int)
            self._input_ineqs[:,:-1] = self._transf_matrix.T.dot(
                                            self._optimal_ineqs_ext[:,:-1].T).T
            self._input_ineqs[:,-1] = [self._optimal_ineqs[i,-1]
                                       - v[:-1].dot(self._transl_vector)
                                       for i,v in enumerate(self._input_ineqs)]
        else:
            self._input_ineqs = np.empty(self._optimal_ineqs.shape, dtype=int)
            self._input_ineqs[:,:-1] = self._transf_matrix.T.dot(
                                                self._optimal_ineqs[:,:-1].T).T
            self._input_ineqs[:,-1] = self._optimal_ineqs[:,-1]
        # Initialize remaining hidden attributes
        self._pts_dict = None
        self._points_sat = None
        self._points = None
        self._interior_points = None
        self._boundary_points = None
        self._points_interior_to_facets = None
        self._boundary_points_not_interior_to_facets = None
        self._points_not_interior_to_facets = None
        self._is_reflexive = None
        self._h11 = None
        self._h21 = None
        self._chi = None
        self._faces = None
        self._vertices = None
        self._dual = None
        self._is_favorable = None
        self._volume = None
        self._normal_form = None
        self._autos = [None]*2
        self._glsm_charge_matrix = [None]*2
        self._glsm_linrels = [None]*2
        self._glsm_basis = [None]*4

    def clear_cache(self):
        """Clears the cached results of any previous computation."""
        self._pts_dict = None
        self._points_sat = None
        self._points = None
        self._interior_points = None
        self._boundary_points = None
        self._points_interior_to_facets = None
        self._boundary_points_not_interior_to_facets = None
        self._points_not_interior_to_facets = None
        self._is_reflexive = None
        self._h11 = None
        self._h21 = None
        self._chi = None
        self._faces = None
        self._vertices = None
        self._dual = None
        self._is_favorable = None
        self._volume = None
        self._normal_form = None
        self._autos = [None]*2
        self._glsm_charge_matrix = [None]*2
        self._glsm_linrels = [None]*2
        self._glsm_basis = [None]*4

    def __repr__(self):
        """Returns a string describing the polytope."""
        return (f"A {self._dim}-dimensional "
                f"{('reflexive ' if self.is_reflexive() else '')}"
                f"lattice polytope in ZZ^{self._ambient_dim}")

    def ambient_dim(self):
        """Returns the dimension of the ambient lattice."""
        return self._ambient_dim

    def dim(self):
        """Returns the dimension of the polytope."""
        return self._dim

    def is_full_dimensional(self):
        """Returns True if the polytope is full-dimensional."""
        return self._ambient_dim == self._dim

    def _points_saturated(self):
        """
        Computes the lattice points of the polytope along with the indices of
        the hyperplane inequalities that they saturate.

        Points are sorted so that interior points are first, and then the rest
        are arranged by decreasing number of saturated inequalities.
        For reflexive polytopes this is useful since the origin will be at
        index 0 and boundary points not interior to facets will be last.

        Typically this function should not be called by the user. Instead, it
        is called by various other functions in the polytope class.
        """
        # This function is based on code by Volker Braun, and is redistributed
        # under the GNU General Public License version 3.
        # The original code can be found at
        # https://github.com/sagemath/sage/blob/master/src/sage/geometry/
        #   integral_points.pyx
        if self._points_sat is not None:
            return np.array(self._points_sat, dtype=object)
        d = self._dim
        # Find bounding box and sort by decreasing dimension size
        box_min = np.array([min(self._optimal_pts[:,i]) for i in range(d)])
        box_max = np.array([max(self._optimal_pts[:,i]) for i in range(d)])
        box_diff = box_max - box_min
        diameter_index = np.argsort(box_diff)[::-1]
        # Construct the inverse permutation
        orig_dict = {j:i for i,j in enumerate(diameter_index)}
        orig_perm = [orig_dict[i] for i in range(d)]
        # Sort box bounds
        box_min = box_min[diameter_index]
        box_max = box_max[diameter_index]
        # Inequalities must also have their coordinates permuted
        ineqs = np.array(self._optimal_ineqs)
        ineqs[:,:-1] = self._optimal_ineqs[:,diameter_index]
        # Find all lattice points and apply the inverse permutation
        points = []
        facet_ind = []
        p = np.array(box_min)
        while True:
            tmp_v = ineqs[:,1:-1].dot(p[1:]) + ineqs[:,-1]
            i_min = box_min[0]
            i_max = box_max[0]
            # Find the lower bound for the allowed region
            while i_min <= i_max:
                if all(i_min*ineqs[i,0] + tmp_v[i] >= 0
                       for i in range(len(tmp_v))):
                    break
                i_min += 1
            # Find the upper bound for the allowed region
            while i_min <= i_max:
                if all(i_max*ineqs[i,0] + tmp_v[i] >= 0
                       for i in range(len(tmp_v))):
                    break
                i_max -= 1
            # The points i_min .. i_max are contained in the polytope
            i = i_min
            while i <= i_max:
                p[0] = i
                saturated = frozenset(j for j in range(len(tmp_v))
                                      if i*ineqs[j,0] + tmp_v[j] == 0)
                points.append(np.array(p)[orig_perm])
                facet_ind.append(saturated)
                i += 1
            # Increment the other entries in p to move on to next loop
            inc = 1
            if d == 1:
                break
            break_loop = False
            while True:
                if p[inc] == box_max[inc]:
                    p[inc] = box_min[inc]
                    inc += 1
                    if inc == d:
                        break_loop = True
                        break
                else:
                    p[inc] += 1
                    break
            if break_loop:
                break
        if self._ambient_dim > self._dim:
            points_mat = np.empty((len(points), self._ambient_dim), dtype=int)
            points_mat[:,self._dim_diff:] = points
            points_mat[:,:self._dim_diff] = 0
        else:
            points_mat = np.array(points, dtype=int)
        points_mat = self._inv_transf_matrix.dot(points_mat.T).T
        if self._ambient_dim > self._dim:
            for i in range(points_mat.shape[0]):
                points_mat[i,:] += self._transl_vector
        # Organize the points as explained above.
        self._points_sat = sorted(
                            [(tuple(points_mat[i]), facet_ind[i])
                                for i in range(len(points))],
                            key=(lambda p:
                                    ((len(p[1]) if len(p[1]) > 0 else 1e9),)
                                    + tuple(p[0])),
                            reverse=True)
        self._points_sat = np.array(self._points_sat, dtype=object)
        self._pts_dict = {ii:i for i,ii in enumerate(self._points_sat[:,0])}
        return np.array(self._points_sat, dtype=object)

    def points(self):
        """
        Returns the lattice points of the polytope.

        Points are sorted so that interior points are first, and then the rest
        are arranged by decreasing number of saturated inequalities.
        For reflexive polytopes this is useful since the origin will be at
        index 0 and boundary points not interior to facets will be last.
        """
        if self._points is not None:
            return np.array(self._points)
        self._points = np.array(self._points_saturated()[:,0].tolist())
        return np.array(self._points)

    def interior_points(self):
        """Returns the interior lattice points of the polytope."""
        if self._interior_points is not None:
            return np.array(self._interior_points)
        self._interior_points = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) == 0])
        return np.array(self._interior_points)

    def boundary_points(self):
        """Returns the boundary lattice points of the polytope."""
        if self._boundary_points is not None:
            return np.array(self._boundary_points)
        self._boundary_points = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) > 0])
        return np.array(self._boundary_points)

    def points_interior_to_facets(self):
        """Returns the lattice points interior to facets."""
        if self._points_interior_to_facets is not None:
            return np.array(self._points_interior_to_facets)
        self._points_interior_to_facets = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) == 1])
        return np.array(self._points_interior_to_facets)

    def boundary_points_not_interior_to_facets(self):
        """Returns the boundary lattice points not interior to facets."""
        if self._boundary_points_not_interior_to_facets is not None:
            return np.array(self._boundary_points_not_interior_to_facets)
        self._boundary_points_not_interior_to_facets = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) > 1])
        return np.array(self._boundary_points_not_interior_to_facets)

    def points_not_interior_to_facets(self):
        """Returns the lattice points not interior to facets."""
        if self._points_not_interior_to_facets is not None:
            return np.array(self._points_not_interior_to_facets)
        self._points_not_interior_to_facets = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) != 1])
        return np.array(self._points_not_interior_to_facets)

    def is_reflexive(self):
        """Returns True if the polytopes is reflexive."""
        if self._is_reflexive is not None:
            return self._is_reflexive
        self._is_reflexive = (self.is_full_dimensional()
                            and all(c == 1 for c in self._optimal_ineqs[:,-1]))
        return self._is_reflexive

    def _compute_h11(self):
        """
        Returns the Hodge number h^{1,1} of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        Equivalently, returns the Hodge number h^{2,1} of the Calabi-Yau
        obtained as the anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        It is recommended to use the functions:
        Polytope.h11(lattice='N')
        Polytope.h21(lattice='M')
        to avoid confusion.
        """
        if self._h11 is not None:
            return self._h11
        if not self.is_reflexive() or self._dim != 4:
            raise Exception("Not a reflexive 4D polytope.")
        facesc1 = self.faces(3)
        facesc2 = self.faces(2)
        h_11 = len(self.points())-5
        for f in facesc1:
            h_11 -= len(f.interior_points())
        for f in facesc2:
            h_11 += len(f.interior_points())*len(f.dual().interior_points())
        self._h11 = h_11
        return self._h11

    def _compute_h21(self):
        """
        Returns the Hodge number h^{2,1} of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        Equivalently, returns the Hodge number h^{1,1} of the Calabi-Yau
        obtained as the anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        It is recommended to use the functions:
        Polytope.h21(lattice='N')
        Polytope.h11(lattice='M')
        to avoid confusion.
        """
        if self._h21 is not None:
            return self._h21
        if not self.is_reflexive() or self._dim != 4:
            raise Exception("Not a reflexive 4D polytope.")
        facesc1 = self.dual().faces(3)
        facesc2 = self.dual().faces(2)
        h_21 = len(self.dual().points()) - 5
        for f in facesc1:
            h_21 -= len(f.interior_points())
        for f in facesc2:
            h_21 += len(f.interior_points())*len(f.dual().interior_points())
        self._h21 = h_21
        return self._h21

    def _compute_chi(self):
        """
        Returns the Euler characteristic of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        Equivalently, returns -1*(the Euler characteristic) of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        It is recommended to use the function:
        Polytope.chi(lattice='N')
        to avoid confusion.
        """
        if self._chi is not None:
            return self._chi
        self._chi = 2*(self._compute_h11() - self._compute_h21())
        return self._chi

    def h11(self, lattice):
        """
        Returns the Hodge number h^{1,1} associated with the polytope.

        If lattice='N', this is the h^{1,1} of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        If lattice='M', this is the h^{1,1} of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        Args:
            lattice (string): Specifies the lattice on which the polytope is
            defined. Options are 'N' and 'M'.
        """
        if lattice=='N':
            return self._compute_h11()
        elif lattice=='M':
            return self._compute_h21()
        else:
            raise Exception("Lattice has to be specified when h11, h21 or chi are specified."
                " The options are: 'N' and 'M'.")

    def h21(self, lattice):
        """
        Returns the Hodge number h^{2,1} associated with the polytope.

        If lattice='N', this is the h^{2,1} of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        If lattice='M', this is the h^{2,1} of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        Args:
            lattice (string): Specifies the lattice on which the polytope is
            defined. Options are 'N' and 'M'.
        """
        if lattice=='N':
            return self._compute_h21()
        elif lattice=='M':
            return self._compute_h11()
        else:
            raise Exception("Lattice has to be specified when h11, h21 or chi are specified."
                " The options are: 'N' and 'M'.")

    def chi(self, lattice):
        """
        Returns the Euler characteristic associated with the polytope.

        If lattice='N', this is the Euler characteristic of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the face fan of the polytope.

        If lattice='M', this is the Euler characteristic of the Calabi-Yau obtained
        as the anticanonical hypersurface in the toric variety given by a
        desingularization of the normal fan of the polytope.

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.

        Args:
            lattice (string): Specifies the lattice on which the polytope is
            defined. Options are 'N' and 'M'.
        """
        if lattice=='N':
            return self._compute_chi()
        elif lattice=='M':
            return -self._compute_chi()
        else:
            raise Exception("Lattice has to be specified when h11, h21 or chi are specified."
                " The options are: 'N' and 'M'.")

    def _faces4d(self):
        """
        Computes the faces of a 4D polytope.

        This function is a slightly more optimized version of the faces()
        function present in this class.  Typically the user should not call
        this function directly.  Instead, it is only called by faces() when the
        polytope is 4-dimensional.
        """
        pts_sat = self._points_saturated()
        vert = [tuple(pt) for pt in self.vertices()]
        vert_sat = [tuple(pt) for pt in pts_sat if pt[0] in vert]
        facets = defaultdict(set)
        # First, create facets
        for pt in vert_sat:
            for f in pt[1]:
                facets[frozenset([f])].add(pt)
        # Then find 2-faces
        facets_list = list(facets.keys())
        n_facets = len(facets_list)
        twofaces = defaultdict(set)
        for i in range(n_facets):
            for j in range(i+1, n_facets):
                f1 = facets_list[i]
                f2 = facets_list[j]
                inter = facets[f1] & facets[f2]
                # These intersections are 2D iff there are at least 3 vertices.
                if len(inter) >= 3:
                    f3 = f1 | f2
                    twofaces[f3] = inter
        # Finally find 1-faces
        twofaces_list = list(twofaces.keys())
        n_twofaces = len(twofaces_list)
        onefaces = defaultdict(set)
        for i in range(n_twofaces):
            for j in range(i+1, n_twofaces):
                f1 = twofaces_list[i]
                f2 = twofaces_list[j]
                inter = twofaces[f1] & twofaces[f2]
                inter_list = list(inter)
                # These intersections are 1D iff there are exactly 2 vertices.
                if len(inter) == 2:
                    f3 = inter_list[0][1] & inter_list[1][1]
                    if f3 not in onefaces.keys():
                        onefaces[f3] = inter
        # Now construct all face objects
        fourface_obj_list = [PolytopeFace(self, vert, frozenset(), dim=4)]
        facets_obj_list = []
        for f in facets.keys():
            tmp_vert = [pt[0] for pt in vert_sat if f.issubset(pt[1])]
            facets_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=3))
        twofaces_obj_list = []
        for f in twofaces.keys():
            tmp_vert = [pt[0] for pt in vert_sat if f.issubset(pt[1])]
            twofaces_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=2))
        onefaces_obj_list = []
        for f in onefaces.keys():
            tmp_vert = [pt[0] for pt in vert_sat if f.issubset(pt[1])]
            onefaces_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=1))
        zerofaces_obj_list = [PolytopeFace(self, [pt[0]], pt[1], dim=0)
                              for pt in vert_sat]
        organized_faces = [zerofaces_obj_list, onefaces_obj_list,
                           twofaces_obj_list, facets_obj_list,
                           fourface_obj_list]
        return np.array(organized_faces, dtype=object)

    def faces(self, d=None):
        """
        Computes the faces of a polytope.

        When the polytope is 4-dimensional it calls the slightly more optimized
        function _faces4d().

        Args:
            d (integer, optional): Optional parameter that specifies the
                dimension of the desired faces.

        Returns:
            list: A list of PolytopeFace objects of dimension d, if
                specified. Otherwise it is a list of lists of
                PolytopeFace objects organized in increasing dimension.
        """
        if d is not None and d not in range(self._dim + 1):
            raise Exception(f"Polytope does not have faces of dimension {d}")
        if self._faces is not None:
            if d is not None:
                return np.array(self._faces[d], dtype=object)
            return np.array(self._faces, dtype=object)
        if self._dim == 4:
            self._faces = self._faces4d()
            if d is not None:
                return np.array(self._faces[d])
            return np.array(self._faces)
        pts_sat = self._points_saturated()
        vert = [tuple(pt) for pt in self.vertices()]
        vert_sat = [tuple(pt) for pt in pts_sat if pt[0] in vert]
        organized_faces = [] # The list where all face obejcts will be stored
        # First construct trivial full-dimensional face
        organized_faces.append([PolytopeFace(self, vert, frozenset(),
                                             dim=self._dim)])
        # If thee polytope is zero-dimensional, finish the computation
        if self._dim == 0:
            self._faces = organized_faces
            if d is not None:
                return np.array(self._faces[d])
            return np.array(self._faces)
        # Now construct the facets
        tmp_facets = []
        for j in range(len(self._input_ineqs)):
            tmp_vert = [pt[0] for pt in vert_sat if j in pt[1]]
            tmp_facets.append(PolytopeFace(self, tmp_vert,
                                           frozenset([j]), dim=self._dim-1))
        organized_faces.append(tmp_facets)
        # Then iteratively construct lower-dimensional faces
        previous_faces = defaultdict(set)
        for pt in vert_sat:
            for f in pt[1]:
                previous_faces[frozenset([f])].add(pt)
        previous_faces_list = list(previous_faces.keys())
        n_previous_faces = len(previous_faces_list)
        for dd in range(self._dim-2, 0, -1):
            current_faces = defaultdict(set)
            for i in range(n_previous_faces):
                for j in range(i+1, n_previous_faces):
                    f1 = previous_faces_list[i]
                    f2 = previous_faces_list[j]
                    inter = previous_faces[f1] & previous_faces[f2]
                    # Check if it has the right dimension
                    if np.linalg.matrix_rank([tuple(pt[0])+(1,)
                                              for pt in inter])-1 != dd:
                        continue
                    # Find saturated inequalities
                    f3 = frozenset.intersection(*[pt[1] for pt in inter])
                    current_faces[f3] = inter
            # Add current faces to the list, and reset for next loop
            tmp_faces = []
            for f in current_faces.keys():
                tmp_vert = [pt[0] for pt in vert_sat if f.issubset(pt[1])]
                tmp_faces.append(PolytopeFace(self, tmp_vert, f, dim=dd))
            organized_faces.append(tmp_faces)
            previous_faces = current_faces
            previous_faces_list = list(previous_faces.keys())
            n_previous_faces = len(previous_faces_list)
        # Finally add vertices
        organized_faces.append([PolytopeFace(self, [pt[0]], pt[1], dim=0)
                                for pt in vert_sat])
        self._faces = organized_faces[::-1]
        if d is not None:
            return np.array(self._faces[d], dtype=object)
        return np.array(self._faces, dtype=object)

    def facets(self):
        """Returns the facets (dimension dim(poly)-1 faces) of the polytope."""
        return self.faces(self._dim-1)

    def vertices(self):
        """Returns the vertices of the polytope."""
        if self._vertices is not None:
            return np.array(self._vertices)
        if self._dim == 0:
            self._vertices = self._input_pts
        elif self._backend == "ppl":
            points_mat = np.array([tuple(int(i) for i in pt.coefficients())
                          for pt in self._optimal_poly.minimized_generators()])
            if self._ambient_dim > self._dim:
                pts_mat_tmp = np.empty((points_mat.shape[0],self._ambient_dim),
                                       dtype=int)
                pts_mat_tmp[:,:self._dim_diff] = 0
                pts_mat_tmp[:,self._dim_diff:] = points_mat.reshape(-1,
                                                                    self._dim)
                points_mat = pts_mat_tmp
            points_mat = self._inv_transf_matrix.dot(points_mat.T).T
            if self._ambient_dim > self._dim:
                points_mat = [pt + self._transl_vector for pt in points_mat]
            tmp_vert = [tuple(pt) for pt in points_mat]
            self._vertices = np.array([list(pt) for pt in self._input_pts
                                       if tuple(pt) in tmp_vert])
        else:
            if self._dim == 1: # QHull cannot handle 1D polytopes
                self._vertices = np.array([
                                    pt[0] for pt in self._points_saturated()
                                    if len(pt[1]) == 1])
            else:
                self._vertices = self._input_pts[self._optimal_poly.vertices]
        return np.array(self._vertices)

    def dual(self):
        """
        Returns the dual polytope (also called polar polytope).  Only lattice
        polytopes are currently supported, so only duals of reflexive polytopes
        can be computed.

        If L is a lattice polytope, the dual polytope of L is
        ConvexHull({y\\in \\mathbb{Z}^n | x\\dot y \\geq -1 for all x\\in L}).
        A lattice polytope is reflexive if its dual is also a lattice polytope.
        """
        if self._dual is not None:
            return self._dual
        if not self.is_reflexive():
            raise Exception("Duality of non-reflexive polytopes not "
                            "supported.")
        pts = np.array(self._input_ineqs[:,:-1])
        self._dual = Polytope(pts, backend=self._backend)
        return self._dual

    def polar(self):
        """Alias for dual().  See dual() function for details."""
        return self.polar()

    def is_favorable(self):
        """
        Returns True if the Calabi-Yau manifold arising from this polytope
        is favorable (i.e. all Kahler forms descend from Kahler forms on the
        ambient toric variety).

        An exception is raised if it is not a 4-dimensional reflexive
        polytope.
        """
        if self._is_favorable is not None:
            return self._is_favorable
        if not self.is_reflexive() or self._ambient_dim != 4:
            raise Exception("Not a reflexive 4D polytope.")
        isfavorable = True
        for face in self.faces(2):
            if len(face.interior_points()) != 0:
                if len(face.dual().interior_points()) != 0:
                    isfavorable = False
                    break
        self._is_favorable = isfavorable
        return self._is_favorable

    def glsm_charge_matrix(self, exclude_origin=False, use_all_points=False,
                           n_retries = 100):
        """
        Compute the GLSM charge matrix of the theory resulting from this
        polytope.

        Args:
            exclude_origin (boolean, optional, default=False): Indicates
                whether to use the origin in the calculation.  This corresponds
                to the inclusion of the canonical divisor.
            use_all_points (boolean, optional, default=False): By default only
                boundary points not interior to facets are used. If this flag
                is set to true then points interior to facets are also used.
            n_retries (int, optional, default=100): Flint sometimes fails to
                find the kernel of a matrix. This flag specifies the number of
                times the points will be suffled and the computation retried.

        Returns: The GLSM charge matrix
        """
        if not self.is_reflexive():
            raise Exception("The GLSM charge matrix can only be computed for"
                            "reflexive polytopes.")
        args_id = 1*use_all_points
        if self._glsm_charge_matrix[args_id] is not None:
            if exclude_origin:
                return np.array(self._glsm_charge_matrix[args_id][:,1:])
            return np.array(self._glsm_charge_matrix[args_id])
        # Set up the list of points that will be used. We always include the
        # origin and discard it at the end if necessary.
        if use_all_points:
            pts = np.array([tuple(pt)+(1,) for pt in self.points()])
        else:
            pts = np.array([tuple(pt)+(1,) for pt in
                                self.points_not_interior_to_facets()])
        # We reverse the order of the points because flint returns a nicer
        # matrix in this way
        pts = pts[::-1,:]
        # Find and check GLSM charge matrix
        ker_np = None
        ctr = -1
        while ((ker_np is None or any(ker_np.dot(pts).flatten()))
                and ctr <= n_retries):
            ctr += 1
            indices = np.array(list(range(len(pts))))
            if ctr > 0:
                np.random.shuffle(indices[:-1])
            ker_dict = {ii:i for i,ii in enumerate(indices)}
            pts_rand = pts[indices,:]
            try:
                ker = fmpz_mat(pts_rand.T.tolist()).nullspace()[0]
            except:
                continue
            ker_np = np.array([v for v in ker.transpose().table() if any(v)],
                              dtype=int)
            ker_np = np.array([v//int(round(gcd_list(v))) for v in ker_np],
                              dtype=int)
            # Check if the last column only has one entry. This should be the
            # case, but if not we have to do row reduction so that the charge
            # matrix without the origin is contained as a submatrix.
            if sum(c != 0 for c in ker_np[:,-1]) != 1 and ker_np[-1,-1] != 0:
                try:
                    ker = fmpz_mat(ker_np[::-1,::-1].tolist()).rref()[0]
                except:
                    continue
                ker_np = np.array(ker.table(), dtype=int)
                ker_np = np.array([v//int(round(gcd_list(v))) for v in ker_np],
                                  dtype=int)[::-1,::-1]
            ker_np = np.array(ker_np[:,[ker_dict[i]
                                        for i in range(ker_np.shape[1])]])
        if ker_np is None:
            raise Exception("Error computing GLSM charge matrix")
        # Reflect the matrix to get the original order
        ker_np = ker_np[::-1,::-1]
        self._glsm_charge_matrix[args_id] = ker_np
        if exclude_origin:
            return np.array(self._glsm_charge_matrix[args_id][:,1:])
        return np.array(self._glsm_charge_matrix[args_id])

    def glsm_linear_relations(self, exclude_origin=False, use_all_points=False,
                              n_retries=100):
        """
        Compute the linear relations of the GLSM charge matrix.

        INPUT:

        exclude_origin (boolean, optional, default=False): Indicates whether to
            use the origin in the calculation.  This corresponds to the
            inclusion of the canonical divisor.
        use_all_points (boolean, optional, default=False): By default only
            boundary points not interior to facets are used. If this flag is
            set to true then points interior to facets are also used.
        n_retries (int, optional, default=100): Flint sometimes fails to find
            the kernel of a matrix. This flag specifies the number of times the
            points will be suffled and the computation retried.

        Returns: A matrix of linear relations of the columns of the GLSM
            charge matrix.
        """
        args_id = 1*use_all_points
        if self._glsm_linrels[args_id] is not None:
            if exclude_origin:
                return np.array(self._glsm_linrels[args_id][1:,1:])
            return np.array(self._glsm_linrels[args_id])
        linrel_np = None
        ctr = -1
        ker_np = self.glsm_charge_matrix(use_all_points=use_all_points,
                                         n_retries=n_retries)
        # We reverse the order of the columns because flint returns a nicer
        # matrix in this way
        ker_np = ker_np[:,::-1]
        while ((linrel_np is None or any(ker_np.dot(linrel_np.T).flatten()))
               and ctr <= n_retries):
            ctr += 1
            indices = np.array(list(range(ker_np.shape[1])))
            if ctr > 0:
                np.random.shuffle(indices[:-1])
            linrel_dict = {ii:i for i,ii in enumerate(indices)}
            linrel_rand = ker_np[:,indices]
            try:
                linrel = fmpz_mat(linrel_rand.tolist()
                                  ).nullspace()[0].transpose()
            except:
                continue
            linrel_np = np.array([v for v in linrel.table() if any(v)],
                                 dtype=int)
            linrel_np = np.array([v//int(round(abs(gcd_list(v))))
                                    for v in linrel_np], dtype=int)
            # Check if the last column only has one entry. This should be the
            # case, but if not we have to do row reduction so that the charge
            # matrix without the origin is contained as a submatrix.
            if (sum(c != 0 for c in linrel_np[:,-1]) != 1
                    and linrel_np[-1,-1] != 0):
                try:
                    linrel = fmpz_mat(linrel_np[::-1,::-1].tolist()).rref()[0]
                except:
                    continue
                linrel_np = np.array(linrel.table(), dtype=int)
                linrel_np = np.array([v//int(round(gcd_list(v)))
                                        for v in linrel_np],
                                    dtype=int)[::-1,::-1]
            linrel_np = np.array(linrel_np[:,[linrel_dict[i]
                                          for i in range(linrel_np.shape[1])]])
        if linrel_np is None or any(ker_np.dot(linrel_np.T).flatten()):
            raise Exception("Error computing linear relations")
        # Reflect the matrix to get the original order
        linrel_np = linrel_np[::-1,::-1]
        self._glsm_linrels[args_id] = linrel_np
        if exclude_origin:
            return np.array(self._glsm_linrels[args_id][1:,1:])
        return np.array(self._glsm_linrels[args_id])

    def glsm_basis(self, exclude_origin=False, use_all_points=False,
                   integral=False, n_retries=100):
        """
        Compute a basis for the column span of the GLSM charge matrix.

        Args:
            exclude_origin (boolean, optional, default=False): Indicates
                whether to use the origin in the calculation.  This corresponds
                to the inclusion of the canonical divisor.
            use_all_points (boolean, optional, default=False): By default only
                boundary points not interior to facets are used. If this flag
                is set to true then points interior to facets are also used.
            integral (boolean, optional, default=False): Indicates
                whether to try to find an integer basis for the columns of the
                GLSM charge matrix. (i.e. so that remaining columns can be
                written as an integer linear combination of the basis.)
            n_retries (int, optional, default=100): Flint sometimes fails to
                find the kernel of a matrix. This flag specifies the number of
                times the points will be suffled and the computation retried.

        Returns: A list of column indices that form a basis
        """
        args_id = 1*use_all_points + 2*integral
        if self._glsm_basis[args_id] is not None:
            if exclude_origin:
                return np.array(self._glsm_basis[args_id]) - 1
            return np.array(self._glsm_basis[args_id])
        linrel_np = self.glsm_linear_relations(use_all_points=use_all_points,
                                               n_retries=n_retries)
        for ctr in range(n_retries):
            found_good_basis=True
            indices = np.array(list(range(linrel_np.shape[1])))
            if ctr > 0:
                np.random.shuffle(indices[1:])
            linrel_rand = np.array(linrel_np[:,indices])
            linrel_dict = {ii:i for i,ii in enumerate(indices)}
            try:
                linrel = fmpz_mat(linrel_rand.tolist()).rref()[0]
            except:
                continue
            linrel_rand = np.array(linrel.table(), dtype=int)
            linrel_rand = np.array([v//int(round(abs(gcd_list(v))))
                                    for v in linrel_rand], dtype=int)
            basis_exc = []
            for v in linrel_rand:
                for i,ii in enumerate(v):
                    if ii != 0:
                        if integral:
                            if abs(ii) == 1:
                                v *= ii
                            else:
                                found_good_basis = False
                        basis_exc.append(i)
                        break
                if not found_good_basis:
                    break
            if found_good_basis:
                break
        linrel_np = np.array(linrel_rand[:,[linrel_dict[i]
                                    for i in range(linrel_rand.shape[1])]])
        basis_ind = np.array(sorted([i for i in range(len(linrel_np[0]))
                                     if linrel_dict[i] not in basis_exc]),
                                        dtype=int)
        ker_np = self.glsm_charge_matrix(use_all_points=use_all_points)
        if (np.linalg.matrix_rank(ker_np) != len(basis_ind)
                or any(ker_np.dot(linrel_np.T).flatten())):
            raise Exception("Error finding basis")
        if integral:
            self._glsm_linrels[1*use_all_points] = linrel_np
            self._glsm_basis[1*use_all_points] = basis_ind
        self._glsm_basis[args_id] = basis_ind
        if exclude_origin:
            return np.array(self._glsm_basis[args_id]) - 1
        return np.array(self._glsm_basis[args_id])

    def volume(self):
        """
        Returns the volume of the polytope.  By convention, the standard
        simplex has unit volume.
        """
        if self._volume is not None:
            return self._volume
        if self._dim == 0:
            self._volume = 0
        elif self._dim == 1:
            self._volume = max(self._optimal_pts) - min(self._optimal_pts)
        else:
            self._volume = int(round(ConvexHull(self._optimal_pts).volume
                               * math.factorial(self._dim)))
        return self._volume

    def points_to_indices(self, pts):
        """Returns the list of indices corresponding to the given points."""
        if self._pts_dict is not None:
            self._points_saturated()
        return np.array([self._pts_dict[tuple(pt)] for pt in pts])

    def normal_form(self):
        """
        Returns the normal form of the polytope as defined by Kreuzer-Skarke.
        """
        # This function is based on code by Andrey Novoseltsev, Samuel Gonshaw,
        # and others, and is redistributed under the GNU General Public License
        # version 3.
        # The original code can be found at:
        # https://github.com/sagemath/sage/blob/develop/src/sage/geometry/
        #   lattice_polytope.py
        # https://trac.sagemath.org/ticket/13525
        if self._normal_form is not None:
            return np.array(self._normal_form)
        # Define function that constructs permutation matrices
        def PGE(n, u, v):
            tmp_m = np.eye(n, dtype=int)
            if u == v:
                return tmp_m
            tmp_m[u-1,u-1] = 0
            tmp_m[v-1,v-1] = 0
            tmp_m[u-1,v-1] = 1
            tmp_m[v-1,u-1] = 1
            return tmp_m
        V = self.vertices()
        n_v = len(V)
        n_f = len(self._input_ineqs)
        PM = np.array([n[:-1].dot(V.T) + n[-1] for n in self._input_ineqs])
        n_s = 1
        prm = {0 : [np.eye(n_f, dtype=int), np.eye(n_v, dtype=int)]}
        for j in range(n_v):
            m = np.argmax([PM[0,prm[0][1].dot(range(n_v))][i]
                           for i in range(j, n_v)])
            if m > 0:
                prm[0][1] = PGE(n_v, j+1, m+j+1).dot(prm[0][1])
        first_row = list(PM[0])
        # Arrange other rows one by one and compare with first row
        for k in range(1, n_f):
            # Error for k == 1 already!
            prm[n_s] = [np.eye(n_f, dtype=int), np.eye(n_v, dtype=int)]
            m = np.argmax(PM[:,prm[n_s][1].dot(range(n_v))][k])
            if m > 0:
                prm[n_s][1] = PGE(n_v, 1, m+1).dot(prm[n_s][1])
            d = (PM[k,prm[n_s][1].dot(range(n_v))][0]
                 - prm[0][1].dot(first_row)[0])
            if d < 0:
                # The largest elt of this row is smaller than largest elt
                # in 1st row, so nothing to do
                continue
            # otherwise:
            for i in range(1, n_v):
                m = np.argmax([PM[k,prm[n_s][1].dot(range(n_v))][j]
                               for j in range(i, n_v)])
                if m > 0:
                    prm[n_s][1] = PGE(n_v, i+1, m+i+1).dot(prm[n_s][1])
                if d == 0:
                    d = (PM[k,prm[n_s][1].dot(range(n_v))][i]
                         - prm[0][1].dot(first_row)[i])
                    if d < 0:
                        break
            if d < 0:
                # This row is smaller than 1st row, so nothing to do
                del prm[n_s]
                continue
            prm[n_s][0] =  PGE(n_f, 1, k+1).dot(prm[n_s][0])
            if d == 0:
                # This row is the same, so we have a symmetry!
                n_s += 1
            else:
                # This row is larger, so it becomes the first row and
                # the symmetries reset.
                first_row = list(PM[k])
                prm = {0: prm[n_s]}
                n_s = 1
        prm = {k:prm[k] for k in prm if k < n_s}
        b = PM[prm[0][0].dot(range(n_f)),:][:,prm[0][1].dot(range(n_v))][0]
        # Work out the restrictions the current permutations
        # place on other permutations as a automorphisms
        # of the first row
        # The array is such that:
        # S = [i, 1, ..., 1 (ith), j, i+1, ..., i+1 (jth), k ... ]
        # describes the "symmetry blocks"
        S = list(range(1, n_v+1))
        for i in range(1, n_v):
            if b[i-1] == b[i]:
                S[i] = S[i-1]
                S[S[i]-1] += 1
            else:
                S[i] = i + 1
        # We determine the other rows of PM_max in turn by use of perms and
        # aut on previous rows.
        for l in range(1, n_f-1):
            n_s = len(prm)
            n_s_bar = n_s
            cf = 0
            l_r = [0]*n_v
            # Search for possible local permutations based off previous
            # global permutations.
            for k in range(n_s_bar-1, -1, -1):
                # number of local permutations associated with current global
                n_p = 0
                ccf = cf
                prmb = {0: copy.copy(prm[k])}
                # We look for the line with the maximal entry in the first
                # subsymmetry block, i.e. we are allowed to swap elements
                # between 0 and S(0)
                for s in range(l, n_f):
                    for j in range(1, S[0]):
                        v = PM[prmb[n_p][0].dot(range(n_f)),
                               :][:,prmb[n_p][1].dot(range(n_v))][s]
                        if v[0] < v[j]:
                            prmb[n_p][1] = PGE(n_v, 1, j+1).dot(prmb[n_p][1])
                    if ccf == 0:
                        l_r[0] = PM[prmb[n_p][0].dot(range(n_f)),
                                    :][:,prmb[n_p][1].dot(range(n_v))][s,0]
                        prmb[n_p][0] = PGE(n_f, l+1, s+1).dot(prmb[n_p][0])
                        n_p += 1
                        ccf = 1
                        prmb[n_p] = copy.copy(prm[k])
                    else:
                        d1 = PM[prmb[n_p][0].dot(range(n_f)),
                                :][:,prmb[n_p][1].dot(range(n_v))][s,0]
                        d = d1 - l_r[0]
                        if d < 0:
                            # We move to the next line
                            continue
                        elif d == 0:
                            # Maximal values agree, so possible symmetry
                            prmb[n_p][0] = PGE(n_f, l+1, s+1).dot(prmb[n_p][0])
                            n_p += 1
                            prmb[n_p] = copy.copy(prm[k])
                        else:
                            # We found a greater maximal value for first entry.
                            # It becomes our new reference:
                            l_r[0] = d1
                            prmb[n_p][0] = PGE(n_f, l+1, s+1).dot(prmb[n_p][0])
                            # Forget previous work done
                            cf = 0
                            prmb = {0:copy.copy(prmb[n_p])}
                            n_p = 1
                            prmb[n_p] = copy.copy(prm[k])
                            n_s = k + 1
                # Check if the permutations found just now work
                # with other elements
                for c in range(1, n_v):
                    h = S[c]
                    ccf = cf
                    # Now let us find out where the end of the
                    # next symmetry block is:
                    if  h < c+1:
                        h = S[h-1]
                    s = n_p
                    # Check through this block for each possible permutation
                    while s > 0:
                        s -= 1
                        # Find the largest value in this symmetry block
                        for j in range(c+1, h):
                            v = PM[prmb[s][0].dot(range(n_f)),
                                   :][:,prmb[s][1].dot(range(n_v))][l]
                            if v[c] < v[j]:
                                prmb[s][1] = PGE(n_v, c+1, j+1).dot(prmb[s][1])
                        if ccf == 0:
                            # Set reference and carry on to next permutation
                            l_r[c] = PM[prmb[s][0].dot(range(n_f)),
                                        :][:,prmb[s][1].dot(range(n_v))][l,c]
                            ccf = 1
                        else:
                            d1 = PM[prmb[s][0].dot(range(n_f)),
                                    :][:,prmb[s][1].dot(range(n_v))][l,c]
                            d = d1 - l_r[c]
                            if d < 0:
                                n_p -= 1
                                if s < n_p:
                                    prmb[s] = copy.copy(prmb[n_p])
                            elif d > 0:
                                # The current case leads to a smaller matrix,
                                # hence this case becomes our new reference
                                l_r[c] = d1
                                cf = 0
                                n_p = s + 1
                                n_s = k + 1
                # Update permutations
                if n_s-1 > k:
                    prm[k] = copy.copy(prm[n_s-1])
                n_s -= 1
                for s in range(n_p):
                    prm[n_s] = copy.copy(prmb[s])
                    n_s += 1
                cf = n_s
            prm = {k:prm[k] for k in prm if k < n_s}
            # If the automorphisms are not already completely restricted,
            # update them
            if S != list(range(1, n_v+1)):
                # Take the old automorphisms and update by
                # the restrictions the last worked out
                # row imposes.
                c = 0
                M = PM[prm[0][0].dot(range(n_f)),
                       :][:,prm[0][1].dot(range(n_v))][l]
                while c < n_v:
                    s = S[c] + 1
                    S[c] = c + 1
                    c += 1
                    while c < s-1:
                        if M[c] == M[c-1]:
                            S[c] = S[c-1]
                            S[S[c]-1] += 1
                        else:
                            S[c] = c + 1
                        c += 1
        # Now we have the perms, we construct PM_max using one of them
        PM_max = PM[prm[0][0].dot(range(n_f)),:][:,prm[0][1].dot(range(n_v))]
        # Finally arrange the points the the canonical order
        p_c = np.eye(n_v, dtype=int)
        M_max = [max([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        S_max = [sum([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        for i in range(n_v):
            k = i
            for j in range(i+1, n_v):
                if (M_max[j] < M_max[k]
                    or (M_max[j] == M_max[k] and S_max[j] < S_max[k])):
                    k = j
            if not k == i:
                M_max[i], M_max[k] = M_max[k], M_max[i]
                S_max[i], S_max[k] = S_max[k], S_max[i]
                p_c = PGE(n_v, 1+i, 1+k).dot(p_c)
        # Create array of possible NFs.
        prm = [p_c.dot(l[1]) for l in prm.values()]
        Vs = [np.array(fmpz_mat(V.T[:,sig.dot(range(n_f))].tolist()
                                ).hnf().table(), dtype=int).tolist()
                                                                for sig in prm]
        Vmin = min(Vs)
        self._normal_form  = np.array(Vmin).T
        return np.array(self._normal_form)

    def triangulate(self, heights=None, make_star=None, only_pts=None,
                    simplices=None, check_input_simplices=True, backend="cgal",
                    backend_dir=None):
        """
        Returns a single regular triangulation of the polytope.

        When reflexive polytopes are used it defaults to returning a random
        fine, regular, star triangulation.

        Args:
            heights (list, optional): A list of heights specifying the regular
                triangulation.  When not secified, it will return the Delaunay
                triangulation when using CGAL, a triangulation obtained from
                random heights near the Delaunay when using QHull, or the
                placing triangulation when using TOPCOM.  Heights can only be
                specified when using CGAL or QHull as the backend.
            make_star (boolean, optional): Whether to return star triangulations only,
                where the center is the origin. If not specified, defaults to
                True for reflexive polytopes, False for non-reflexive polytopes.
            only_pts (list or string, optional): A list of the indices of the
                points to be included in the triangulation.  If not specified,
                it uses points not interior to facets when the polytope is
                reflexive, and all of them otherwise.  When it is desired to
                force the inclusion of all points, it can be set to "all".
            simplices (list, optional): A list of simplices specifying the
                triangulation.  This is useful when a triangulation was
                previously computed and it needs to be inputted again. Note
                that the order of the points needs to be consistent.
            backend (string, optional, default=cgal): Specifies the backend
                used to compute the triangulation.  The available options are
                "qhull", "cgal", and "topcom".
            backend_dir (string, optional): This can be used to specify the
                location of CGAL or TOPCOM binaries when they are not in PATH.

        Returns:
            Triangulation: A triangulation of the polytope.
        """
        if self._ambient_dim > self._dim:
            raise Exception("Only triangulations of full-dimensional polytopes"
                            "are supported.")
        if only_pts is not None and not isinstance(only_pts, str):
            pts = self.points()
            triang_pts = [tuple(pts[i]) for i in only_pts]
        elif (only_pts is not None and isinstance(only_pts, str)
                and only_pts == "all"):
            triang_pts = self.points()
        elif self.is_reflexive():
            triang_pts = self.points_not_interior_to_facets()
        else:
            triang_pts = self.points()
        if make_star is None:
            if self.is_reflexive():
                make_star = True
            else:
                make_star = False
        if (0,)*self._dim not in triang_pts:
            make_star = False
        return Triangulation(triang_pts, poly=self, heights=heights,
                             make_star=make_star, simplices=simplices,
                             check_input_simplices=check_input_simplices,
                             backend=backend, backend_dir=backend_dir)

    def random_triangulations_fast(self, N=None, c=0.01, max_retries=500,
                                make_star=True, only_fine=True, only_pts=None,
                                backend="cgal", backend_dir=None,
                                as_list=False, progress_bar=False):
        """
        Constructs pseudorandom regular (optionally fine and star)
        triangulations of a given point set. This is done by picking random
        heights around the Delaunay heights from a Gaussian distribution.

        Args:
            N (int, optional): Number of desired unique triangulations. If not
                specified, it will return as many triangulations as it can find
                until it has to retry more than max_retries times to obtain a new
                triangulation.
            c (float, optional, default=0.01): A contant used as a coefficient of
                the Gaussian distribution used to pick the heights. A larger c
                results in a wider range of possible triangulations, but with a
                larger fraction of them being non-fine.
            max_retries (int, optional, default=500): Maximum number of attempts
                to obtain a new triangulation before the process is terminated.
            make_star (boolean, optional): Whether to return star triangulations only,
                where the center is the origin. If not specified, defaults to
                True for reflexive polytopes, False for non-reflexive polytopes.
            only_fine (boolean, optional, default=True): Whether to find only
                fine triangulations.
            only_pts (list or string, optional): A list of the indices of the
                points to be included in the triangulation.  If not specified,
                it uses points not interior to facets when the polytope is
                reflexive, and all of them otherwise.  When it is desired to
                force the inclusion of all points, it can be set to "all".
            backend (string, optional, default="cgal"): Specifies the backend
                used to compute the triangulation.  The available options are
                "cgal" and "qhull".
            backend_dir (string, optional): This can be used to specify the
                location of CGAL or TOPCOM binaries when they are not in PATH.
            progress_bar (boolean, optional, default=True): Shows number of
                triangulations obtained and progress bar. This is ignored when
                returning a generator.

        Returns:
            A list or a generator of Triangulation objects.
        """
        if self._ambient_dim > self._dim:
            raise Exception("Only triangulations of full-dimensional polytopes"
                            "are supported.")
        if N is None and as_list:
            raise Exception("Number of triangulations must be specified when "
                            "returning a list.")
        if only_pts is not None and not isinstance(only_pts, str):
            pts = self.points()
            triang_pts = [tuple(pts[i]) for i in only_pts]
        elif (only_pts is not None and isinstance(only_pts, str)
                and only_pts == "all"):
            triang_pts = self.points()
        elif self.is_reflexive():
            triang_pts = self.points_not_interior_to_facets()
        else:
            triang_pts = self.points()
        triang_pts = [tuple(pt) for pt in triang_pts]
        if make_star is None:
            if self.is_reflexive():
                make_star = True
            else:
                make_star = False
        if (0,)*self._dim not in triang_pts:
            make_star = False
        g = random_triangulations_fast_generator(triang_pts, N=N, c=c,
                max_retries=max_retries, make_star=make_star,
                only_fine=only_fine, backend=backend, backend_dir=backend_dir,
                poly=self)
        if not as_list:
            return g
        if progress_bar:
            from tqdm import tqdm
            pbar = tqdm(total=N, file=sys.stdout)
        triangs_list = []
        for i in range(N):
            try:
                triangs_list.append(next(g))
                if progress_bar:
                    pbar.update(len(triangs_list)-pbar.n)
            except StopIteration:
                if progress_bar:
                    pbar.update(N-pbar.n)
                break
        return triangs_list

    def random_triangulations_fair(self, N=None, n_walk=None, n_flip=None,
                        initial_walk_steps=None, walk_step_size=1e-2,
                        max_steps_to_wall=10, fine_tune_steps=8, max_retries=50,
                        make_star=None, only_pts=None, backend="cgal",
                        backend_dir=None, as_list=False, progress_bar=False):
        """
        Returns a pseudorandom list of regular triangulations of a given point set.
        Implements Algorithm #3 from:

                    Bounding the Kreuzer-Skarke Landscape
                Mehmet Demirtas, Liam McAllister, Andres Rios-Tascon
                        https://arxiv.org/abs/2008.01730

        This is a Markov chain Monte Carlo algorithm that involves
        * taking random walks inside the subset of the secondary fan
             that result in fine triangulations.
        * performing random flips.
        For details, please see Section 4.1 in the paper.

        Args:
            N (int, optional): Number of desired unique triangulations.
            n_walk (int, optional): Number of hit-and-run steps per triangulation.
            n_flip (int, optional): Number of random flips performed per triangulation.
            initial_walk_steps (int, optional): Number of
                hit-and-run steps to take before starting to record triangulations.
                Small values may result in a bias towards Delaunay-like triangulations.
            walk_step_size (float, optional, default=1e-2): Determines size of random steps taken
                in the secondary fan. Algorithm may stall if too small.
            max_steps_to_wall (int, optional, default=10): Maximum Number of steps to
                take towards a wall of the subset of the secondary fan that correspond to
                fine triangulations. If a wall is not found, a new random direction is
                selected. Setting this to be very large (>100) reduces performance.
                If this, or walk_step_size, is set to be too low, the algorithm may stall.
            fine_tune_steps (int, optional, default=8): Number of steps to determine the
                location of a wall. Decreasing improves performance, but might result in
                biased samples.
            max_retries (int, optional, default=500): Maximum number of attempts
                to obtain a new triangulation before the process is terminated.
            make_star (boolean, optional): Whether to return star triangulations only,
                where the center is the origin. If not specified, defaults to
                True for reflexive polytopes, False for non-reflexive polytopes.
            only_pts (list or string, optional): A list of the indices of the
                points to be included in the triangulation.  If not specified,
                it uses points not interior to facets when the polytope is
                reflexive, and all of them otherwise.  When it is desired to
                force the inclusion of all points, it can be set to "all".
            backend (string, optional, default=cgal): Specifies the backend
                used to compute the triangulation.  The available options are
                "qhull", "cgal", and "topcom".
            backend_dir (string, optional): This can be used to specify the
                location of CGAL or TOPCOM binaries when they are not in PATH.
            progress_bar (boolean, optional, default=False): Shows number of
                triangulations obtained and progress bar.

        Returns:
            list: A list of Triangulation objects.

        """
        if self._ambient_dim > self._dim:
            raise Exception("Only triangulations of full-dimensional polytopes"
                            "are supported.")
        if N is None and as_list:
            raise Exception("Number of triangulations must be specified when "
                            "returning a list.")
        if only_pts is not None and not isinstance(only_pts, str):
            pts = self.points()
            triang_pts = [tuple(pts[i]) for i in only_pts]
        elif (only_pts is not None and isinstance(only_pts, str)
                and only_pts == "all"):
            triang_pts = self.points()
        elif self.is_reflexive():
            triang_pts = self.points_not_interior_to_facets()
        else:
            triang_pts = self.points()
        triang_pts = [tuple(pt) for pt in triang_pts]
        if make_star is None:
            if self.is_reflexive():
                make_star = True
            else:
                make_star = False
        if (0,)*self._dim not in triang_pts:
            make_star = False
        if n_walk is None:
            n_walk = len(self.points())//10 + 1
        if n_flip is None:
            n_flip = len(self.points())//10 + 1
        if initial_walk_steps is None:
            initial_walk_steps = 2*len(self.points())//10 + 1
        g = random_triangulations_fair_generator(triang_pts, N=N, n_walk=n_walk,
                n_flip=n_flip, initial_walk_steps=initial_walk_steps,
                walk_step_size=walk_step_size, max_steps_to_wall=max_steps_to_wall,
                fine_tune_steps=fine_tune_steps, max_retries=max_retries,
                make_star=make_star, backend=backend, backend_dir=backend_dir,
                poly=self)
        if not as_list:
            return g
        if progress_bar:
            from tqdm import tqdm
            pbar = tqdm(total=N, file=sys.stdout)
        triangs_list = []
        for i in range(N):
            try:
                triangs_list.append(next(g))
                if progress_bar:
                    pbar.update(len(triangs_list)-pbar.n)
            except StopIteration:
                if progress_bar:
                    pbar.update(N-pbar.n)
                break
        return triangs_list

    def all_triangulations(self, only_pts=None, only_fine=True,
                           only_regular=True, only_star=True, star_origin=None,
                           topcom_dir=None):
        """
        Computes all triangulations of the polytop using TOPCOM.

        Args:
            only_pts (list or string, optional): A list of the indices of the
                points to be included in the triangulation.  If not specified,
                it uses points not interior to facets when the polytope is
                reflexive, and all of them otherwise.  When it is desired to
                force the inclusion of all points, it can be set to "all".
            only_fine (boolean, optional, default=True): Restricts to only
                fine triangulations.
            only_regular (boolean, optional, default=True): Restricts to only
                regular triangulations.
            only_star (boolean, optional, default=True): Restricts to only
                star triangulations.
            star_origin (int, optional): The index of the point that will be
                used as the star origin. If the polytope is reflexive this
                is set to 0, but otherwise it must be specified.
            topcom_dir (string, optional): This can be used to specify the
                location of the TOPCOM binaries when they are not in PATH.

        Returns:
            list: A list of all triangulations of the polytope with the
                specified properties.
        """
        if star_origin is None:
            if self.is_reflexive():
                star_origin = 0
            else:
                raise Exception("The star_origin parameter must be specified "
                                "when finding star triangulations of "
                                "non-reflexive polytopes.")
        if only_pts is not None and not isinstance(only_pts, str):
            pts = self.points()
            triang_pts = [tuple(pts[i]) for i in only_pts]
        elif (only_pts is not None and isinstance(only_pts, str)
                and only_pts == "all"):
            triang_pts = self.points()
        elif self.is_reflexive():
            triang_pts = self.points_not_interior_to_facets()
        else:
            triang_pts = self.points()
        triangs = all_triangulations(triang_pts, only_fine=only_fine, only_regular=only_regular,
                                     only_star=only_star, star_origin=star_origin,
                                     topcom_dir=topcom_dir)
        return [Triangulation(triang_pts, poly=self, simplices=simps,
                              check_input_simplices=False, backend_dir=topcom_dir)
                    for simps in triangs]

    def automorphisms(self, square_to_one=False):
        """
        Returns the GL(d,Z) matrices that leave the polytope invariant.  These
        matrices act on the points by multiplication on the right.

        Args:
            square_to_one (boolean, optional, default=False): Flag that
                restricts to only matrices that square to the identity.

        Returns:
            list: A list of automorphism matrices.
        """
        args_id = 1*square_to_one
        if self._autos[args_id] is not None:
            return self._autos[args_id]
        vert_set = set(tuple(pt) for pt in self.vertices())
        f_min = None
        for f in self.facets():
            if f_min is None or len(f.vertices()) < len(f_min.vertices()):
                f_min = f
        f_min_vert_rref = np.array(fmpz_mat(f_min.vertices().T.tolist()
                                            ).rref()[0].table(), dtype=int)
        pivots = []
        for v in f_min_vert_rref:
            if any(v):
                for i,ii in enumerate(v):
                    if ii != 0:
                        pivots.append(i)
                        break
        basis = [f_min.vertices()[i].tolist() for i in pivots]
        basis_inverse = fmpz_mat(basis).inv()
        images = []
        for f in self.facets():
            if len(f_min.vertices()) == len(f.vertices()):
                f_vert = [pt.tolist() for pt in f.vertices()]
                images.extend(permutations(f_vert, r=int(self.dim())))
        autos = []
        autos2 = []
        for im in images:
            image = fmpz_mat(im)
            m = basis_inverse*image
            if not all(abs(c.q) == 1 for c in np.array(m.table()).flatten()):
                continue
            m = np.array([[round(int(c.p)/int(c.q)) for c in r]
                           for r in np.array(m.table())], dtype=int)
            if set(tuple(pt) for pt in np.dot(self.vertices(), m)) != vert_set:
                continue
            autos.append(m)
            if all((np.dot(m,m) == np.eye(self.dim(), dtype=int)).flatten()):
                autos2.append(m)
            self._autos = [autos, autos2]
        return self._autos[args_id]
