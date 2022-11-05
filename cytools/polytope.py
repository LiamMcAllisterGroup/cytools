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

# Standard imports
from collections import defaultdict
from itertools import permutations
import subprocess
import warnings
import copy
import math
# Third party imports
from scipy.spatial import ConvexHull
from flint import fmpz_mat, fmpq_mat
from tqdm import tqdm
import numpy as np
import ppl
# CYTools imports
from cytools.triangulation import (Triangulation, all_triangulations,
                                   random_triangulations_fast_generator,
                                   random_triangulations_fair_generator)
from cytools.polytopeface import PolytopeFace
from cytools.utils import gcd_list
from cytools import config



class Polytope:
    """
    This class handles all computations relating to lattice polytopes, such as
    the computation of lattice points and faces. When using reflexive
    polytopes, it also allows the computation of topological properties of the
    arising Calabi-Yau hypersurfaces that only depend on the polytope.

    ## Constructor

    ### `cytools.polytope.Polytope`

    **Description:**
    Constructs a `Polytope` object describing a lattice polytope. This is
    handled by the hidden [`__init__`](#__init__) function.

    :::note notes
    - CYTools only supports lattice polytopes, so any floating point numbers
      will be truncated to integers.
    - The Polytope class is also imported to the root of the CYTools package,
      so it can be imported from `cytools.polytope` or from `cytools`.
    :::

    **Arguments:**
    - `points` *(array_like)*: A list of lattice points defining the
      polytope as their convex hull.
    - `backend` *(string, optional)*: A string that specifies the backend
      used to construct the convex hull. The available options are "ppl",
      "qhull", or "palp". When not specified, it uses PPL for dimensions up to
      four, and palp otherwise.

    **Example:**
    We construct two polytopes from lists of points.
    ```python {2,5}
    from cytools import Polytope
    p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    print(p1)
    # A 4-dimensional reflexive lattice polytope in ZZ^4
    p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
    print(p2)
    # A 3-dimensional lattice polytope in ZZ^4
    ```
    """

    def __init__(self, points, backend=None):
        """
        **Description:**
        Initializes a `Polytope` object describing a lattice polytope.

        :::note
        CYTools only supports lattice polytopes, so any floating point numbers
        will be truncated to integers.
        :::

        **Arguments:**
        - `points` *(array_like)*: A list of lattice points defining the
          polytope as their convex hull.
        - `backend` *(string, optional)*: A string that specifies the
          backend used to construct the convex hull. The available options are
          "ppl", "qhull", or "palp". When not specified, it uses PPL for
          dimensions up to four, and palp otherwise.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new `Polytope`
        object. Thus, it is used in the following example.
        ```python {2,5}
        from cytools import Polytope
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        print(p1)
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        print(p2)
        # A 3-dimensional lattice polytope in ZZ^4
        ```
        """
        # Convert points to numpy array and compute dimension
        self._input_pts = np.array(points, dtype=int)
        self._ambient_dim = self._input_pts.shape[1]
        in_pts_shape = self._input_pts.shape
        pts_ext = np.empty((in_pts_shape[0], in_pts_shape[1]+1), dtype=int)
        pts_ext[:,:-1] = self._input_pts
        pts_ext[:,-1] = 1
        self._dim = np.linalg.matrix_rank(pts_ext) - 1
        self._dim_diff = self._ambient_dim - self._dim
        # Select backend for the computation of the convex hull
        backends = ["ppl", "qhull", "palp", None]
        if backend not in backends:
            raise ValueError(f"Invalid backend. Options are {backends}.")
        if backend is None:
            if self._dim <= 4:
                backend = "ppl"
            else:
                backend = "palp"
        if self._dim == 0: # 0-dimensional polytopes are finicky
            backend = "palp"
        self._backend = backend
        # Find the optimal form of the polytope by performing LLL reduction.
        # If the polytope is not full-dimensional it constructs an
        # affinely-equivalent polytope in a lattice of matching dimension.
        # Internally it uses the optimal form for computations, but it outputs
        # everything in the same form as the input
        if self._dim == self._ambient_dim:
            pts_mat = fmpz_mat(self._input_pts.T.tolist())
            optimal_pts, transf_matrix = pts_mat.lll(transform=True)
            self._optimal_pts = np.array(optimal_pts.tolist(), dtype=int).T
        else:
            self._transl_vector = self._input_pts[0]
            tmp_pts = np.array(self._input_pts)
            for i in range(self._input_pts.shape[0]):
                tmp_pts[i] -= self._transl_vector
            pts_mat = fmpz_mat(tmp_pts.T.tolist())
            optimal_pts, transf_matrix = pts_mat.lll(transform=True)
            optimal_pts = np.array(optimal_pts.tolist(), dtype=int).T
            self._optimal_pts = optimal_pts[:, self._dim_diff:]
        self._transf_matrix = np.array(transf_matrix.tolist(), dtype=int)
        inv_tranf_matrix = transf_matrix.inv(integer=True)
        self._inv_transf_matrix = np.array(inv_tranf_matrix.tolist(),dtype=int)
        # Flint sometimes returns an inverse that is missing a factor of -1
        check_inverse = self._inv_transf_matrix.dot(self._transf_matrix)
        id_mat = np.eye(self._ambient_dim, dtype=int)
        if all((check_inverse == id_mat).flatten()):
            pass
        elif all((check_inverse == -id_mat).flatten()):
            self._inv_transf_matrix = -self._inv_transf_matrix
        else:
            raise RuntimeError("Problem finding inverse matrix")
        # Construct convex hull and find the hyperplane representation with the
        # appropriate backend. The equations are in the form
        # c_0 * x_0 + ... + c_{d-1} * x_{d-1} + c_d >= 0
        if backend == "ppl":
            gs = ppl.Generator_System()
            vrs = [ppl.Variable(i) for i in range(self._dim)]
            for pt in self._optimal_pts:
                ppl_pt = ppl.point(sum(pt[i]*vrs[i] for i in range(self._dim)))
                gs.insert(ppl_pt)
            self._optimal_poly = ppl.C_Polyhedron(gs)
            optimal_ineqs = [list(ineq.coefficients())
                             + [ineq.inhomogeneous_term()]
                             for ineq in self._optimal_poly.minimized_constraints()]
            self._optimal_ineqs = np.array(optimal_ineqs, dtype=int)
        elif backend == "qhull":
            if self._dim == 0: # qhull cannot handle 0-dimensional polytopes
                self._optimal_poly = None
                self._optimal_ineqs = np.array([[0]])
            elif self._dim == 1: # qhull cannot handle 1-dimensional polytopes
                self._optimal_poly = None
                min_pt, max_pt = min(self._optimal_pts), max(self._optimal_pts)
                self._optimal_ineqs = np.array([[1,-min_pt],[-1,max_pt]])
            else:
                self._optimal_poly = ConvexHull(self._optimal_pts)
                tmp_ineqs = set()
                for eq in self._optimal_poly.equations:
                    g = abs(gcd_list(eq))
                    tmp_ineqs.add(tuple(-int(round(i/g)) for i in eq))
                self._optimal_ineqs = np.array(list(tmp_ineqs), dtype=int)
        else: # Backend is PALP
            self._optimal_poly = None
            if self._dim == 0: # PALP cannot handle 0-dimensional polytopes
                self._optimal_ineqs = np.array([[0]])
            else:
                palp = subprocess.Popen((config.palp_path + "poly.x", "-e"),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True)
                pt_list = ""
                optimal_pts = {tuple(pt) for pt in self._optimal_pts}
                for pt in optimal_pts:
                    pt_str = str(pt).replace("(","").replace(")","")
                    pt_str = pt_str.replace(","," ")
                    pt_list += pt_str + "\n"
                palp_in = f"{len(optimal_pts)} {self._dim}\n{pt_list}\n"
                palp_out = palp.communicate(input=palp_in)[0]
                if "Equations" not in palp_out:
                    raise RuntimeError(f"PALP error. Full output: {palp_out}")
                palp_out = palp_out.split("\n")
                for i,line in enumerate(palp_out):
                    if "Equations" not in line:
                        continue
                    self._is_reflexive = "Vertices" in line
                    ineqs_shape = [int(c) for c in line.split()[:2]]
                    tmp_ineqs = [[int(c) for c in palp_out[i+j+1].split()]
                                 for j in range(ineqs_shape[0])]
                    break
                if ineqs_shape[0] < ineqs_shape[1]: # Check if transposed
                    tmp_ineqs = np.array(tmp_ineqs).T
                else:
                    tmp_ineqs = np.array(tmp_ineqs)
                if self._is_reflexive:
                    ineqs_shape = tmp_ineqs.shape
                    tmp_ineqs2 = np.empty((ineqs_shape[0], ineqs_shape[1]+1), dtype=int)
                    tmp_ineqs2[:,:-1] = tmp_ineqs
                    tmp_ineqs2[:,-1] = 1
                    tmp_ineqs = tmp_ineqs2
                self._optimal_ineqs = tmp_ineqs
        if self._ambient_dim > self._dim:
            shape = (self._optimal_ineqs.shape[0],
                     self._optimal_ineqs.shape[1] + self._dim_diff)
            self._optimal_ineqs_ext = np.empty(shape, dtype=int)
            self._optimal_ineqs_ext[:,self._dim_diff:] = self._optimal_ineqs
            self._optimal_ineqs_ext[:,:self._dim_diff] = 0
            self._input_ineqs = np.empty(shape, dtype=int)
            self._input_ineqs[:,:-1] = self._transf_matrix.T.dot(self._optimal_ineqs_ext[:,:-1].T).T
            self._input_ineqs[:,-1] = [self._optimal_ineqs[i,-1]
                                       - v[:-1].dot(self._transl_vector)
                                       for i,v in enumerate(self._input_ineqs)]
        else:
            self._input_ineqs = np.empty(self._optimal_ineqs.shape, dtype=int)
            self._input_ineqs[:,:-1] = self._transf_matrix.T.dot(self._optimal_ineqs[:,:-1].T).T
            self._input_ineqs[:,-1] = self._optimal_ineqs[:,-1]
        # Initialize remaining hidden attributes
        self._hash = None
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
        self._h13 = None
        self._h22 = None
        self._chi = None
        self._faces = None
        self._vertices = None
        self._dual = None
        self._is_favorable = None
        self._volume = None
        self._normal_form = [None]*3
        self._autos = [None]*4
        self._nef_parts = dict()
        self._glsm_charge_matrix = dict()
        self._glsm_linrels = dict()
        self._glsm_basis = dict()

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We compute the lattice points of a large polytope.
        ```python {4}
        p = Polytope([[-1,-1,-1,-1,-1],[3611,-1,-1,-1,-1],[-1,42,-1,-1,-1],[-1,-1,6,-1,-1],[-1,-1,-1,2,-1],[-1,-1,-1,-1,1]])
        pts = p.points() # Takes a few seconds
        pts = p.points() # It runs instantly because the result is cached
        p.clear_cache() # Clears the results of any previos computation
        pts = p.points() # Again it takes a few seconds since the chache was cleared
        ```
        """
        self._hash = None
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
        self._h13 = None
        self._h22 = None
        self._chi = None
        self._faces = None
        self._vertices = None
        self._dual = None
        self._is_favorable = None
        self._volume = None
        self._normal_form = [None]*3
        self._autos = [None]*4
        self._nef_parts = dict()
        self._glsm_charge_matrix = dict()
        self._glsm_linrels = dict()
        self._glsm_basis = dict()

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the polytope.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the polytope.

        **Example:**
        This function can be used to convert the polytope to a string or to
        print information about the polytope.
        ```python {2,3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        poly_info = str(p) # Converts to string
        print(p) # Prints polytope info
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        ```
        """
        return (f"A {self._dim}-dimensional "
                f"{('reflexive ' if self.is_reflexive() else '')}"
                f"lattice polytope in ZZ^{self._ambient_dim}")

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of polytopes with ==.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope that is being compared.

        **Returns:**
        *(bool)* The truth value of the polytopes being equal.

        **Example:**
        We construct two polytopes and compare them.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p1 == p2
        # True
        ```
        """
        if not isinstance(other, Polytope):
            return NotImplemented
        return sorted(self.vertices().tolist()) == sorted(other.vertices().tolist())

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of polytopes with !=.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope that is being compared.

        **Returns:**
        *(bool)* The truth value of the polytopes being different.

        **Example:**
        We construct two polytopes and compare them.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p1 != p2
        # False
        ```
        """
        if not isinstance(other, Polytope):
            return NotImplemented
        return not sorted(self.vertices().tolist()) == sorted(other.vertices().tolist())

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from polytopes.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The hash value of the polytope.

        **Example:**
        We compute the hash value of a polytope. Also, we construct a set and a
        dictionary with a polytope, which make use of the hash function.
        ```python {2,3,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        h = hash(p) # Obtain hash value
        d = {p: 1} # Create dictionary with polytope keys
        s = {p} # Create a set of polytopes
        ```
        """
        if self._hash is None:
            self._hash = hash(tuple(sorted(tuple(v) for v in self.vertices())))
        return self._hash

    def __add__(self, other):
        """
        **Description:**
        Implements addition of polytopes with the
        [`minkowski_sum`](#minkowski_sum) function.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope used for the Minkowski
          sum.

        **Returns:**
        *(Polytope)* The Minkowski sum.

        **Example:**
        We construct two polytops and compute their Minkowski sum.
        ```python {3}
        p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
        p2 = Polytope([[0,0,1],[0,0,-1]])
        p1 + p2
        # A 3-dimensional reflexive lattice polytope in ZZ^3
        ```
        """
        if not isinstance(other, Polytope):
            return NotImplemented
        return self.minkowski_sum(other)

    def is_linearly_equivalent(self, other, backend="palp"):
        """
        **Description:**
        Returns True if the polytopes can be transformed into each other by an
        $SL^{\pm}(d,\mathbb{Z})$ transformation.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope being compared.
        - `backend` *(string, optional, default="palp")*: Selects which
          backend to use to compute the normal form. Options are "native",
          which uses native python code, or "palp", which uses PALP for the
          computation.

        **Returns:**
        *(bool)* The truth value of the polytopes being linearly equivalent.

        **Example:**
        We construct two polytopes and check if they are linearly equivalent.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[1,1,1,1]])
        p1.is_linearly_equivalent(p2)
        # True
        ```
        """
        return (self.normal_form(affine_transform=False, backend=backend).tolist()
                == other.normal_form(affine_transform=False, backend=backend).tolist())

    def is_affinely_equivalent(self, other, backend="palp"):
        """
        **Description:**
        Returns True if the polytopes can be transformed into each other by an
        integral affine transformation.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope being compared.
        - `backend` *(string, optional, default="palp")*: Selects which
          backend to use to compute the normal form. Options are "native",
          which uses native python code, or "palp", which uses PALP for the
          computation.

        **Returns:**
        *(bool)* The truth value of the polytopes being affinely equivalent.

        **Example:**
        We construct two polytopes and check if they are affinely equivalent.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,2],[-1,-1,-1,0]])
        p1.is_affinely_equivalent(p2)
        # True
        ```
        """
        return (self.normal_form(affine_transform=True, backend=backend).tolist()
                == other.normal_form(affine_transform=True, backend=backend).tolist())

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
        We construct a polytope and check the dimension of the ambient lattice.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.ambient_dimension()
        # 4
        ```
        """
        return self._ambient_dim
    # Aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """
        **Description:**
        Returns the dimension of the polytope.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the polytope.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a polytope and check its dimension.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.dimension()
        # 3
        ```
        """
        return self._dim
    # Aliases
    dim = dimension

    def is_solid(self):
        """
        **Description:**
        Returns True if the polytope is solid (i.e. full-dimensional) and False
        otherwise.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the polytope being full-dimensional.

        **Example:**
        We construct a polytope and check if it is solid.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.is_solid()
        # False
        ```
        """
        return self._ambient_dim == self._dim

    def inequalities(self):
        """
        **Description:**
        Returns the inequalities giving the hyperplane representation of the
        polytope. The inequalities are given in the form
        $c_0x_0 + \cdots + c_{d-1}x_{d-1} + c_d \geq 0$.
        Note, however, that equalities are not included.

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The inequalities defining the polytope.

        **Example:**
        We construct a polytope and find the defining inequalities.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p.inequalities()
        # array([[ 4, -1, -1, -1,  1],
        #        [-1,  4, -1, -1,  1],
        #        [-1, -1,  4, -1,  1],
        #        [-1, -1, -1,  4,  1],
        #        [-1, -1, -1, -1,  1]])
        ```
        """
        return np.array(self._input_ineqs)

    def _points_saturated(self):
        """
        **Description:**
        Computes the lattice points of the polytope along with the indices of
        the hyperplane inequalities that they saturate.

        :::note notes
        - Points are sorted so that interior points are first, and then the
          rest are arranged by decreasing number of saturated inequalities and
          lexicographically. For reflexive polytopes this is useful since the
          origin will be at index 0 and boundary points interior to facets will
          be last.
        - Typically this function should not be called by the user. Instead, it
          is called by various other functions in the Polytope class.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(list)* A list of tuples. The first component of each tuple is the list
        of coordinates of the point and the second component is a
        `frozenset` of the hyperplane inequalities that it saturates.

        **Example:**
        We construct a polytope and compute the lattice points along with the
        inequalities that they saturate. We print the second point and the
        inequalities that it saturates.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        pts_sat = p._points_saturated()
        print(pts_sat[1])
        # ((-1, -1, -1, -1), frozenset({0, 1, 2, 3}))
        p.inequalities()[list(pts_sat[1][1])]
        # array([[ 4, -1, -1, -1,  1],
        #        [-1,  4, -1, -1,  1],
        #        [-1, -1,  4, -1,  1],
        #        [-1, -1, -1,  4,  1]])
        ```
        """
        # This function is based on code by Volker Braun, and is redistributed
        # under the GNU General Public License version 2+.
        # The original code can be found at
        # https://github.com/sagemath/sage/blob/master/src/sage/geometry/integral_points.pyx
        if self._points_sat is not None:
            return copy.copy(self._points_sat)
        d = self._dim
        # When using PALP as the backend we use it to compute all lattice
        # points in the polytope.
        if self._backend == "palp":
            if self._dim == 0: # PALP cannot handle 0-dimensional polytopes
                points = [self._optimal_pts[0]]
                facet_ind = [frozenset([0])]
            else:
                palp = subprocess.Popen((config.palp_path + "poly.x", "-p"),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True)
                pt_list = ""
                optimal_pts = {tuple(pt) for pt in self._optimal_pts}
                for pt in optimal_pts:
                    pt_str = str(pt).replace("(","").replace(")","")
                    pt_str = pt_str.replace(","," ")
                    pt_list += pt_str + "\n"
                palp_in = f"{len(optimal_pts)} {self._dim}\n{pt_list}\n"
                palp_out = palp.communicate(input=palp_in)[0]
                if "Points of P" not in palp_out:
                    raise RuntimeError(f"PALP error. Full output: {palp_out}")
                palp_out = palp_out.split("\n")
                for i,line in enumerate(palp_out):
                    if "Points of P" not in line:
                        continue
                    pts_shape = [int(c) for c in line.split()[:2]]
                    tmp_pts = np.empty(pts_shape, dtype=int)
                    for j in range(pts_shape[0]):
                        tmp_pts[j,:] = [int(c) for c in palp_out[i+j+1].split()]
                    break
                if pts_shape[0] < pts_shape[1]: # Check if transposed
                    points = tmp_pts.T
                else:
                    points = tmp_pts
                # Now we find which inequialities each point saturates
                ineqs = self._optimal_ineqs
                facet_ind = [frozenset(i for i,ii in enumerate(ineqs) if ii[:-1].dot(pt) + ii[-1] == 0)
                             for pt in points]
        # Otherwise we use the algorithm by Volker Braun.
        else:
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
            ineqs = np.array(self._optimal_ineqs) # We need a new copy
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
                    if all(i_min*ineqs[i,0] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
                        break
                    i_min += 1
                # Find the upper bound for the allowed region
                while i_min <= i_max:
                    if all(i_max*ineqs[i,0] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
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
        # The points and saturated inequalities have now been computed.
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
        self._points_sat = sorted([(tuple(points_mat[i]), facet_ind[i]) for i in range(len(points))],
                                  key=(lambda p: (-(len(p[1]) if len(p[1]) > 0 else 1e9),) + tuple(p[0])))
        self._pts_dict = {ii[0]:i for i,ii in enumerate(self._points_sat)}
        return copy.copy(self._points_sat)

    def points(self, as_indices=False):
        """
        **Description:**
        Returns the lattice points of the polytope.

        :::note
        Points are sorted so that interior points are first, and then the
        rest are arranged by decreasing number of saturated inequalities and
        lexicographically. For reflexive polytopes this is useful since the
        origin will be at index 0 and boundary points interior to facets will
        be last.
        :::

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of lattice points of the polytope.

        **Aliases:**
        `pts`.

        **Example:**
        We construct a polytope and compute the lattice points. One can verify
        that the first point is the only interior point, and the last three
        points are the ones interior to facets. Thus it follows the
        aforementioned ordering.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points()
        # array([[ 0,  0,  0,  0],
        #        [-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3],
        #        [ 0,  0, -1, -2],
        #        [ 0,  0, -1, -1],
        #        [ 0,  0,  0, -1]])
        ```
        """
        if self._points is None:
            self._points = np.array([pt[0] for pt in self._points_saturated()])
        if as_indices:
            return self.points_to_indices(self._points)
        return np.array(self._points)
    # Aliases
    pts = points

    def interior_points(self, as_indices=False):
        """
        **Description:**
        Returns the interior lattice points of the polytope.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of interior lattice points of the polytope.

        **Aliases:**
        `interior_pts`.

        **Example:**
        We construct a polytope and compute the interior lattice points.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.interior_points()
        # array([[ 0,  0,  0,  0]])
        ```
        """
        if self._interior_points is None:
            self._interior_points = np.array([pt[0] for pt in self._points_saturated() if len(pt[1]) == 0])
        if as_indices:
            return self.points_to_indices(self._interior_points)
        return np.array(self._interior_points)
    # Aliases
    interior_pts = interior_points

    def boundary_points(self, as_indices=False):
        """
        **Description:**
        Returns the boundary lattice points of the polytope.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of boundary lattice points of the polytope.

        **Aliases:**
        `boundary_pts`.

        **Example:**
        We construct a polytope and compute the boundary lattice points.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.interior_points()
        # array([[-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3],
        #        [ 0,  0, -1, -2],
        #        [ 0,  0, -1, -1],
        #        [ 0,  0,  0, -1]])
        ```
        """
        if self._boundary_points is None:
            self._boundary_points = np.array([pt[0] for pt in self._points_saturated() if len(pt[1]) > 0])
        if as_indices:
            return self.points_to_indices(self._boundary_points)
        return np.array(self._boundary_points)
    # Aliases
    boundary_pts = boundary_points

    def points_interior_to_facets(self, as_indices=False):
        """
        **Description:**
        Returns the lattice points interior to facets.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of lattice points interior to facets of the
        polytope.

        **Aliases:**
        `pts_interior_to_facets`.

        **Example:**
        We construct a polytope and compute the lattice points interior to
        facets.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points_interior_to_facets()
        # array([[ 0,  0, -1, -2],
        #        [ 0,  0, -1, -1],
        #        [ 0,  0,  0, -1]])
        ```
        """
        if self._points_interior_to_facets is None:
            self._points_interior_to_facets = np.array([pt[0] for pt in self._points_saturated() if len(pt[1]) == 1])
        if as_indices:
            return self.points_to_indices(self._points_interior_to_facets)
        return np.array(self._points_interior_to_facets)
    # Aliases
    pts_interior_to_facets = points_interior_to_facets

    def boundary_points_not_interior_to_facets(self, as_indices=False):
        """
        **Description:**
        Returns the boundary lattice points not interior to facets.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of boundary lattice points not interior to
        facets of the polytope.

        **Aliases:**
        `boundary_pts_not_interior_to_facets`.

        **Example:**
        We construct a polytope and compute the boundary lattice points not
        interior to facets.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.boundary_points_not_interior_to_facets()
        # array([[-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3]])
        ```
        """
        if self._boundary_points_not_interior_to_facets is None:
            self._boundary_points_not_interior_to_facets = np.array([pt[0] for pt in self._points_saturated() if len(pt[1]) > 1])
        if as_indices:
            return self.points_to_indices(
                                self._boundary_points_not_interior_to_facets)
        return np.array(self._boundary_points_not_interior_to_facets)
    # Aliases
    boundary_pts_not_interior_to_facets = boundary_points_not_interior_to_facets

    def points_not_interior_to_facets(self, as_indices=False):
        """
        **Description:**
        Returns the lattice points not interior to facets.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of lattice points not interior to facets of
        the polytope.

        **Aliases:**
        `pts_not_interior_to_facets`.

        **Example:**
        We construct a polytope and compute the lattice points not interior to
        facets.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points_not_interior_to_facets()
        # array([[ 0,  0,  0,  0],
        #        [-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3]])
        ```
        """
        if self._points_not_interior_to_facets is None:
            self._points_not_interior_to_facets = np.array([pt[0] for pt in self._points_saturated() if len(pt[1]) != 1])
        if as_indices:
            return self.points_to_indices(self._points_not_interior_to_facets)
        return np.array(self._points_not_interior_to_facets)
    # Aliases
    pts_not_interior_to_facets = points_not_interior_to_facets

    def is_reflexive(self):
        """
        **Description:**
        Returns True if the polytope is reflexive and False otherwise.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the polytope being reflexive.

        **Example:**
        We construct a polytope and check if it is reflexive.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.is_reflexive()
        # True
        ```
        """
        if self._is_reflexive is not None:
            return self._is_reflexive
        self._is_reflexive = self.is_solid() and all(c == 1 for c in self._input_ineqs[:,-1])
        return self._is_reflexive

    def hpq(self, p, q, lattice):
        """
        **Description:**
        Returns the Hodge number $h^{p,q}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note notes
        - Only reflexive polytopes of dimension 2-5 are currently supported.
        - This function always computes Hodge numbers from scratch. The
          functions [`h11`](#h11), [`h21`](#h21), [`h12`](#h12),
          [`h13`](#h13), and [`h22`](#h22) cache the results so they
          offer improved performance.
        :::

        **Arguments:**
        - `p` *(int)*: The holomorphic index of the Dolbeault cohomology
          of interest.
        - `q` *(int)*: The anti-holomorphic index of the Dolbeault
          cohomology of interest.
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Hodge number $h^{p,q}$ of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and check some Hodge numbers of the associated
        hypersurfaces.
        ```python {2,4,6,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.hpq(0,0,lattice="N")
        # 1
        p.hpq(0,1,lattice="N")
        # 0
        p.hpq(1,1,lattice="N")
        # 2
        p.hpq(1,2,lattice="N")
        # 272
        ```
        """
        d = self.dim()
        if not self.is_reflexive() or d not in (2,3,4,5):
            raise ValueError("Only reflexive polytopes of dimension 2-5 are "
                             "currently supported.")
        if lattice == "M":
            p = d-p-1
        elif lattice != "N":
            raise ValueError("Lattice must be specified. "
                             "Options are: \"N\" or \"M\".")
        if p > q:
            p,q = q,p
        if p > d-1 or p > d-1 or p < 0 or q < 0 or p+q > d-1:
            return 0
        if p in (0,d-1) or q in (0,d-1):
            if p == q or (p,q) == (0,d-1):
                return 1
            return 0
        if p >= math.ceil((d-1)/2):
            tmp_p = p
            p = d-q-1
            q = d-tmp_p-1
        hpq = 0
        if p == 1:
            faces_cqp1 = self.faces(d-q-1)
            for f in faces_cqp1:
                hpq += len(f.interior_points())*len(f.dual().interior_points())
            if q == 1:
                hpq += len(self.points_not_interior_to_facets()) - d - 1
            if q == d-2:
                hpq += len(self.dual().points_not_interior_to_facets()) - d - 1
            return hpq
        if p == 2:
            hpq = 44 + 4*self.h11(lattice="N") - 2*self.h12(lattice="N") + 4*self.h13(lattice="N")
            return hpq
        raise RuntimeError("Error computing Hodge numbers.")

    def h11(self, lattice):
        """
        **Description:**
        Returns the Hodge number $h^{1,1}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Hodge number $h^{1,1}$ of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and compute $h^{1,1}$ of the associated
        hypersurfaces.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.h11(lattice="N")
        # 2
        p.h11(lattice="M")
        # 272
        ```
        """
        if lattice == "N":
            if self._h11 is None:
                self._h11 = self.hpq(1,1,lattice="N")
            return self._h11
        if lattice == "M":
            return self.dual().h11(lattice="N")
        raise ValueError("Lattice must be specified. "
                         "Options are: \"N\" or \"M\".")

    def h12(self, lattice):
        """
        **Description:**
        Returns the Hodge number $h^{1,2}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Hodge number $h^{1,2}$ of the arising Calabi-Yau manifold.

        **Aliases:**
        `h21`.

        **Example:**
        We construct a polytope and compute $h^{1,2}$ of the associated
        hypersurfaces.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.h12(lattice="N")
        # 272
        p.h12(lattice="M")
        # 2
        ```
        """
        if lattice == "N":
            if self._h21 is None:
                self._h21 = self.hpq(1,2,lattice="N")
            return self._h21
        if lattice == "M":
            return self.dual().h12(lattice="N")
        raise ValueError("Lattice must be specified. "
                         "Options are: \"N\" or \"M\".")
    # Aliases
    h21 = h12

    def h13(self, lattice):
        """
        **Description:**
        Returns the Hodge number $h^{1,3}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Hodge number $h^{1,3}$ of the arising Calabi-Yau manifold.

        **Aliases:**
        `h31`.

        **Example:**
        We construct a polytope and compute $h^{1,3}$ of the associated
        hypersurfaces.
        ```python {2,4}
        p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-6,-9,-18]])
        p.h13(lattice="N")
        # 2966
        p.h13(lattice="M")
        # 8
        ```
        """
        if lattice == "N":
            if self._h13 is None:
                self._h13 = self.hpq(1,3,lattice="N")
            return self._h13
        if lattice == "M":
            return self.dual().h13(lattice="N")
        raise ValueError("Lattice must be specified. "
                         "Options are: \"N\" or \"M\".")
    # Aliases
    h31 = h13

    def h22(self, lattice):
        """
        **Description:**
        Returns the Hodge number $h^{2,2}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Hodge number $h^{2,2}$ of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and compute $h^{2,2}$ of the associated
        hypersurfaces.
        ```python {2}
        p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-6,-9,-18]])
        p.h22(lattice="N")
        # 11940
        ```
        """
        if lattice == "N":
            if self._h22 is None:
                self._h22 = self.hpq(2,2,lattice="N")
            return self._h22
        if lattice == "M":
            return self.dual().h22(lattice="N")
        raise ValueError("Lattice must be specified. "
                         "Options are: \"N\" or \"M\".")

    def chi(self, lattice):
        """
        **Description:**
        Computes the Euler characteristic of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        **Returns:**
        *(int)* The Euler characteristic of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and compute the Euler characteristic of the
        associated hypersurfaces.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.chi(lattice="N")
        # -540
        p.chi(lattice="M")
        # 540
        ```
        """
        if not self.is_reflexive() or self.dim() not in (2,3,4,5):
           raise NotImplementedError("Not a reflexive polytope of dimension 2-5.")
        if lattice not in ("N","M"):
            raise ValueError("Lattice must be specified. Options are: 'N' or 'M'.")
        if lattice == "M":
            return self.dual().chi(lattice="N")
        if self._chi is not None:
            return self._chi
        if self.dim() == 2:
            self._chi = 0
        elif self.dim() == 3:
            self._chi = self.h11(lattice=lattice) + 4
        elif self.dim() == 4:
            self._chi = 2*(self.h11(lattice=lattice)-self.h21(lattice=lattice))
        elif self.dim() == 5:
            self._chi = 48 + 6*(self.h11(lattice=lattice) - self.h12(lattice=lattice) + self.h13(lattice=lattice))
        return self._chi

    def _faces4d(self):
        """
        **Description:**
        Computes the faces of a 4D polytope.

        :::note
        This function is a slightly more optimized version of the
        [`faces`](#faces) function. Typically the user should not call this
        function directly. Instead, it is only called by [`faces`](#faces) when
        the polytope is 4-dimensional.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(tuple)* A tuple of tuples of faces organized in ascending dimension.

        **Example:**
        We construct a 4D polytope and compute its faces. Since this function
        generally should not be directly used, we do this with the
        [`faces`](#faces) function.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        allfaces = p.faces() # the _faces4d function is used since it is a 4d polytope
        print(allfaces[1][0]) # Print the first face in the tuple of 1-dimensional faces
        # A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
        """
        pts_sat = self._points_saturated()
        vert = [tuple(pt) for pt in self.vertices()]
        vert_sat = [tuple(pt) for pt in pts_sat if pt[0] in vert]
        facets = defaultdict(set)
        # First create facets
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
        organized_faces = (tuple(zerofaces_obj_list), tuple(onefaces_obj_list),
                           tuple(twofaces_obj_list), tuple(facets_obj_list),
                           tuple(fourface_obj_list))
        return organized_faces

    def faces(self, d=None):
        """
        **Description:**
        Computes the faces of a polytope.

        :::note
        When the polytope is 4-dimensional it calls the slightly more optimized
        [`_faces4d()`](#_faces4d) function.
        :::

        **Arguments:**
        - `d` *(int, optional)*: Optional parameter that specifies the
          dimension of the desired faces.

        **Returns:**
        *(tuple)* A tuple of [`PolytopeFace`](./polytopeface) objects of
        dimension d, if specified. Otherwise, a tuple of tuples of
        [`PolytopeFace`](./polytopeface) objects organized in ascending
        dimension.

        **Example:**
        We show that this function returns a tuple of 2-faces if `d`
        is set to 2. Otherwise, the function returns all faces in tuples
        organized in ascending dimension. We verify that the first element in
        the tuple of 2-faces is the same as the first element in the
        corresponding subtuple in the tuple of all faces.
        ```python {2,3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        faces2d = p.faces(2)
        allfaces = p.faces()
        print(faces2d[0]) # Print first face in tuple of 2-faces
        # A 2-dimensional face of a 4-dimensional polytope in ZZ^4
        faces2d[0] is allfaces[2][0]
        # True
        ```
        """
        if d is not None and d not in range(self._dim + 1):
            raise ValueError(f"Polytope does not have faces of dimension {d}")
        if self._faces is not None:
            return (self._faces[d] if d is not None else self._faces)
        if self._dual is not None and self._dual._faces is not None:
            self._faces = (tuple(tuple(f.dual() for f in ff) for ff in self._dual._faces[::-1][1:])
                           + ((PolytopeFace(self, self.vertices(), frozenset(), dim=self._dim),),))
            return (self._faces[d] if d is not None else self._faces)
        if self._dim == 4:
            self._faces = self._faces4d()
            return (self._faces[d] if d is not None else self._faces)
        pts_sat = self._points_saturated()
        vert = [tuple(pt) for pt in self.vertices()]
        vert_sat = [tuple(pt) for pt in pts_sat if pt[0] in vert]
        organized_faces = [] # The list where all face obejcts will be stored
        # First construct trivial full-dimensional face
        organized_faces.append([PolytopeFace(self, vert, frozenset(), dim=self._dim)])
        # If thee polytope is zero-dimensional, finish the computation
        if self._dim == 0:
            self._faces = organized_faces
            return np.array(self._faces[d] if d is not None else [np.array(ff) for ff in self._faces])
        # Now construct the facets
        tmp_facets = []
        for j in range(len(self._input_ineqs)):
            tmp_vert = [pt[0] for pt in vert_sat if j in pt[1]]
            tmp_facets.append(PolytopeFace(self, tmp_vert, frozenset([j]), dim=self._dim-1))
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
                    if np.linalg.matrix_rank([tuple(pt[0])+(1,) for pt in inter])-1 != dd:
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
        organized_faces.append([PolytopeFace(self, [pt[0]], pt[1], dim=0) for pt in vert_sat])
        self._faces = tuple(tuple(ff) for ff in organized_faces[::-1])
        return (self._faces[d] if d is not None else self._faces)

    def facets(self):
        """
        **Description:**
        Returns the facets (codimension-1 faces) of the polytope.

        **Arguments:**
        None.

        **Returns:**
        *(tuple)* A list of [`PolytopeFace`](./polytopeface) objects of
        codimension 1.

        **Example:**
        We construct a polytope and find its facets.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        facets = p.facets()
        ```
        """
        return self.faces(self._dim-1)

    def vertices(self, as_indices=False):
        """
        **Description:**
        Returns the vertices of the polytope.

        **Arguments:**
        - `as_indices` *(bool)*: Return the points as indices of the full
          list of points of the polytope.

        **Returns:**
        *(numpy.ndarray)* The list of vertices of the polytope.

        **Example:**
        We construct a polytope and find its vertices. We can see that they
        match the points that we used to construct the polytope.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p.vertices()
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [ 0,  0,  0,  1],
        #        [-1, -1, -1, -1]])
        ```
        """
        if self._vertices is not None:
            if as_indices:
                return self.points_to_indices(self._vertices)
            return np.array(self._vertices)
        if self._dim == 0:
            self._vertices = np.array([self._input_pts[0]])
        elif self._backend == "ppl":
            points_mat = np.array([tuple(int(i) for i in pt.coefficients())
                                   for pt in self._optimal_poly.minimized_generators()])
            if self._ambient_dim > self._dim:
                pts_mat_tmp = np.empty((points_mat.shape[0],self._ambient_dim), dtype=int)
                pts_mat_tmp[:,:self._dim_diff] = 0
                pts_mat_tmp[:,self._dim_diff:] = points_mat.reshape(-1, self._dim)
                points_mat = pts_mat_tmp
            points_mat = self._inv_transf_matrix.dot(points_mat.T).T
            if self._ambient_dim > self._dim:
                points_mat = [pt + self._transl_vector for pt in points_mat]
            tmp_vert = [tuple(pt) for pt in points_mat]
            input_pts = []
            for pt in self._input_pts:
                pt_tup = tuple(pt)
                if pt_tup not in input_pts:
                    input_pts.append(pt_tup)
            self._vertices = np.array([list(pt) for pt in input_pts if pt in tmp_vert])
        elif self._backend == "qhull":
            if self._dim == 1: # QHull cannot handle 1D polytopes
                tmp_vert = [tuple(pt[0]) for pt in self._points_saturated() if len(pt[1]) == 1]
                self._vertices = np.array([list(pt) for pt in self._input_pts if tuple(pt) in tmp_vert])
            else:
                self._vertices = self._input_pts[self._optimal_poly.vertices]
        else: # Backend is PALP
            palp = subprocess.Popen((config.palp_path + "poly.x", "-v"),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
            pt_list = ""
            optimal_pts = {tuple(pt) for pt in self._optimal_pts}
            for pt in optimal_pts:
                pt_list += (str(pt).replace("(","").replace(")","").replace(","," ") + "\n")
            palp_out = palp.communicate(input=f"{len(optimal_pts)} {self._dim}\n" + pt_list + "\n")[0]
            if "Vertices of P" not in palp_out:
                raise RuntimeError(f"PALP error. Full output: {palp_out}")
            palp_out = palp_out.split("\n")
            for i,line in enumerate(palp_out):
                if "Vertices of P" not in line:
                    continue
                pts_shape = [int(c) for c in line.split()[:2]]
                tmp_pts = np.empty(pts_shape, dtype=int)
                for j in range(pts_shape[0]):
                    tmp_pts[j,:] = [int(c) for c in palp_out[i+j+1].split()]
                break
            points = (tmp_pts.T if pts_shape[0] < pts_shape[1] else tmp_pts)
            if self._ambient_dim > self._dim:
                points_mat = np.empty((len(points),self._ambient_dim), dtype=int)
                points_mat[:,self._dim_diff:] = points
                points_mat[:,:self._dim_diff] = 0
            else:
                points_mat = np.array(points, dtype=int)
            points_mat = self._inv_transf_matrix.dot(points_mat.T).T
            if self._ambient_dim > self._dim:
                for i in range(points_mat.shape[0]):
                    points_mat[i,:] += self._transl_vector
            tmp_vert = [tuple(pt) for pt in points_mat]
            input_pts = []
            for pt in self._input_pts:
                pt_tup = tuple(pt)
                if pt_tup not in input_pts:
                    input_pts.append(pt_tup)
            self._vertices = np.array([list(pt) for pt in input_pts if pt in tmp_vert])
        if as_indices:
            return self.points_to_indices(self._vertices)
        return np.array(self._vertices)

    def dual_polytope(self):
        """
        **Description:**
        Returns the dual polytope (also called polar polytope).  Only lattice
        polytopes are currently supported, so only duals of reflexive polytopes
        can be computed.

        :::note
        If $L$ is a lattice polytope, the dual polytope of $L$ is
        $ConvexHull(\{y\in \mathbb{Z}^n | x\cdot y \geq -1 \text{ for all } x \in L\})$.
        A lattice polytope is reflexive if its dual is also a lattice polytope.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(Polytope)* The dual polytope.

        **Aliases:**
        `dual`, `polar_polytope`, `polar`.

        **Example:**
        We construct a reflexive polytope and find its dual. We then verify
        that the dual of the dual is the original polytope.
        ```python {2,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p_dual = p.dual_polytope()
        print(p_dual)
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        p_dual_dual = p_dual.dual_polytope()
        p_dual_dual is p
        # True
        ```
        """
        if self._dual is not None:
            return self._dual
        if not self.is_reflexive():
            raise NotImplementedError("Duality of non-reflexive polytopes not supported.")
        pts = np.array(self._input_ineqs[:,:-1])
        self._dual = Polytope(pts, backend=self._backend)
        self._dual._dual = self
        return self._dual
    # Aliases
    dual = dual_polytope
    polar_polytope = dual_polytope
    polar = dual_polytope

    def is_favorable(self, lattice):
        """
        **Description:**
        Returns True if the Calabi-Yau hypersurface arising from this polytope
        is favorable (i.e. all Kahler forms descend from Kahler forms on the
        ambient toric variety) and False otherwise.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice` *(str)*: Specifies the lattice on which the polytope
          is defined. Options are "N" and "M".

        *(bool)* The truth value of the polytope being favorable.

        **Example:**
        We construct two reflexive polytopes and find whether they are
        favorable when considered in the N lattice.
        ```python {3,5}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-3,-6]])
        p1.is_favorable(lattice="N")
        # True
        p2.is_favorable(lattice="N")
        # False
        ```
        """
        if lattice=="N":
            if self._is_favorable is None:
                self._is_favorable = (len(self.points_not_interior_to_facets())
                                      == self.h11(lattice="N")+self.dim()+1)
            return self._is_favorable
        if lattice=='M':
            return self.dual().is_favorable(lattice="N")
        raise ValueError("Lattice must be specified. "
                        "Options are: \"N\" or \"M\".")

    def glsm_charge_matrix(self, include_origin=True,
                           include_points_interior_to_facets=False,
                           points=None, integral=True):
        """
        **Description:**
        Computes the GLSM charge matrix of the theory resulting from this
        polytope.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.
        - `include_points_interior_to_facets` *(bool, optional,
          default=False)*: By default only boundary points not interior to
          facets are used. If this flag is set to true then points interior to
          facets are also used.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameters `include_origin` and
          `include_points_interior_to_facets` are ignored.
        - `integral` *(bool, optional, default=True)*: Indicates whether
          to find an integral basis for the columns of the GLSM charge matrix.
          (i.e. so that remaining columns can be written as an integer linear
          combination of the basis elements.)

        **Returns:**
        *(numpy.ndarray)* The GLSM charge matrix.

        **Example:**
        We construct a polytope and find the GLSM charge matrix with different
        parameters.
        ```python {2,5,8,11}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_charge_matrix()
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        p.glsm_charge_matrix().dot(p.points_not_interior_to_facets()) # By definition this product must be zero
        # array([[0, 0, 0, 0],
        #        [0, 0, 0, 0]])
        p.glsm_charge_matrix(include_origin=False) # Excludes the canonical divisor
        # array([[1, 9, 6, 1, 1, 0],
        #        [0, 3, 2, 0, 0, 1]])
        p.glsm_charge_matrix(include_points_interior_to_facets=True) # Includes points interior to facets
        # array([[-18,   1,   9,   6,   1,   1,   0,   0,   0,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1,   0,   0,   0],
        #        [ -4,   0,   2,   1,   0,   0,   0,   1,   0,   0],
        #        [ -3,   0,   1,   1,   0,   0,   0,   0,   1,   0],
        #        [ -2,   0,   1,   0,   0,   0,   0,   0,   0,   1]])
        ```
        """
        if not self.is_reflexive():
            raise ValueError("The GLSM charge matrix can only be computed for "
                             "reflexive polytopes.")
        # Set up the list of points that will be used.
        if points is not None:
            # We always add the origin, but remove it later if necessary
            pts_ind = tuple(set(list(points)+[0]))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = tuple(range(self.points().shape[0]))
        else:
            pts_ind = tuple(range(self.points_not_interior_to_facets().shape[0]))
        if (pts_ind,integral) in self._glsm_charge_matrix:
            if not include_origin and points is None:
                return np.array(self._glsm_charge_matrix[(pts_ind,integral)][:,1:])
            return np.array(self._glsm_charge_matrix[(pts_ind,integral)])
        # If the result is not cached we do the computation
        # We start by finding a basis of columns
        if integral:
            linrel = self.points()[list(pts_ind)].T
            sublat_ind =  int(round(np.linalg.det(np.array(fmpz_mat(linrel.tolist()).snf().tolist(), dtype=int)[:,:linrel.shape[0]])))
            norms = [np.linalg.norm(p,1) for p in linrel.T]
            linrel = np.insert(linrel, 0, np.ones(linrel.shape[1], dtype=int), axis=0)
            good_exclusions = 0
            basis_exc = []
            indices = np.argsort(norms)
            indices[:linrel.shape[0]] = np.sort(indices[:linrel.shape[0]])
            for n_try in range(14):
                if n_try == 1:
                    indices[:] = np.array(range(linrel.shape[1]))
                elif n_try == 2:
                    pts_lll = np.array(fmpz_mat(linrel[1:,:].tolist()).lll().tolist(), dtype=int)
                    norms = [np.linalg.norm(p,1) for p in pts_lll.T]
                    indices = np.argsort(norms)
                    indices[:linrel.shape[0]] = np.sort(indices[:linrel.shape[0]])
                elif n_try == 3:
                    indices[:] = np.array([0] + list(range(1,linrel.shape[1]))[::-1])
                    indices[:linrel.shape[0]] = np.sort(indices[:linrel.shape[0]])
                elif n_try > 3:
                    if n_try == 4:
                        np.random.seed(1337)
                    np.random.shuffle(indices[1:])
                    indices[:linrel.shape[0]] = np.sort(indices[:linrel.shape[0]])
                for ctr in range(np.prod(linrel.shape)+1):
                    found_good_basis=True
                    ctr += 1
                    if ctr > 0:
                        st = max([good_exclusions,1])
                        indices[st:] = np.roll(indices[st:], -1)
                        indices[:linrel.shape[0]] = np.sort(indices[:linrel.shape[0]])
                    linrel_rand = np.array(linrel[:,indices])
                    try:
                        linrel_hnf = fmpz_mat(linrel_rand.tolist()).hnf()
                    except:
                        continue
                    linrel_rand = np.array(linrel_hnf.tolist(), dtype=int)
                    good_exclusions = 0
                    basis_exc = []
                    tmp_sublat_ind = 1
                    for v in linrel_rand:
                        for i,ii in enumerate(v):
                            if ii != 0:
                                tmp_sublat_ind *= abs(ii)
                                if sublat_ind % tmp_sublat_ind == 0:
                                    v *= ii//abs(ii)
                                    good_exclusions += 1
                                else:
                                    found_good_basis = False
                                basis_exc.append(i)
                                break
                        if not found_good_basis:
                            break
                    if found_good_basis:
                        break
                if found_good_basis:
                    break
            if not found_good_basis:
                warnings.warn("An integral basis could not be found. "
                              "A non-integral one will be computed. However, this "
                              "will not be usable as a basis of divisors for the "
                              "ToricVariety or CalabiYau classes.")
                if pts_ind == tuple(self.points_not_interior_to_facets(as_indices=True)):
                    warnings.warn("Please let the developers know about the "
                                  "polytope that caused this issue. "
                                  "Here are the vertices of the polytope: "
                                  f"{self.vertices().tolist()}")
                return self.glsm_charge_matrix(include_origin=include_origin,
                                               include_points_interior_to_facets=include_points_interior_to_facets,
                                               points=points, integral=False)
            linrel_dict = {ii:i for i,ii in enumerate(indices)}
            linrel = np.array(linrel_rand[:,[linrel_dict[i] for i in range(linrel_rand.shape[1])]])
            basis_ind = np.array([i for i in range(linrel.shape[1]) if linrel_dict[i] not in basis_exc], dtype=int)
            basis_exc = np.array([indices[i] for i in basis_exc])
            glsm = np.zeros((linrel.shape[1]-linrel.shape[0],linrel.shape[1]), dtype=int)
            glsm[:,basis_ind] = np.eye(len(basis_ind), dtype=int)
            for nb in basis_exc[::-1]:
                tup = [(k,kk) for k,kk in enumerate(linrel[:,nb]) if kk]
                if sublat_ind % tup[-1][1] != 0:
                    raise RuntimeError("Problem with linear relations")
                i,ii = tup[-1]
                if integral:
                    glsm[:,nb] = -glsm.dot(linrel[i])//ii
                else:
                    glsm[i,:] *= ii
                    glsm[:,nb] = -glsm.dot(linrel[i])
        else: # Non-integral basis
            pts = self.points()[list(pts_ind)[1:]] # Exclude the origin
            pts_norms = [np.linalg.norm(p,1) for p in pts]
            pts_order = np.argsort(pts_norms)
            # Find good lattice basis
            good_lattice_basis = pts_order[:1]
            current_rank = 1
            for p in pts_order:
                tmp = pts[np.append(good_lattice_basis, p)]
                rank = np.linalg.matrix_rank(np.dot(tmp.T,tmp))
                if rank>current_rank:
                    good_lattice_basis = np.append(good_lattice_basis, p)
                    current_rank = rank
                    if rank==self._dim:
                        break
            good_lattice_basis = np.sort(good_lattice_basis)
            glsm_basis = [i for i in range(len(pts)) if i not in good_lattice_basis]
            M = fmpq_mat(pts[good_lattice_basis].T.tolist())
            M_inv = np.array(M.inv().tolist())
            extra_pts = -1*np.dot(M_inv,pts[glsm_basis].T)
            row_scalings = np.array([np.lcm.reduce([int(ii.q) for ii in i]) for i in extra_pts])
            column_scalings = np.array([np.lcm.reduce([int(ii.q) for ii in i]) for i in extra_pts.T])
            extra_rows = np.multiply(extra_pts, row_scalings[:, None])
            extra_rows = np.array([[int(ii.p) for ii in i] for i in extra_rows])
            extra_columns = np.multiply(extra_pts.T, column_scalings[:, None]).T
            extra_columns = np.array([[int(ii.p) for ii in i] for i in extra_columns])
            glsm = np.diag(column_scalings)
            for p,pp in enumerate(good_lattice_basis):
                glsm = np.insert(glsm, pp, extra_columns[p], axis=1)
            origin_column = -np.dot(glsm,np.ones(len(glsm[0])))
            glsm = np.insert(glsm, 0, origin_column, axis=1)
            linear_relations = extra_rows
            extra_linear_relation_columns = -1*np.diag(row_scalings)
            for p,pp in enumerate(good_lattice_basis):
                linear_relations = np.insert(linear_relations, pp, extra_linear_relation_columns[p], axis=1)
            linear_relations = np.insert(linear_relations, 0, np.ones(len(pts)), axis=0)
            linear_relations = np.insert(linear_relations, 0, np.zeros(self._dim+1), axis=1)
            linear_relations[0][0] = 1
            linrel = linear_relations
            basis_ind = glsm_basis
        # Check that everything was computed correctly
        if (np.linalg.matrix_rank(glsm[:,basis_ind]) != len(basis_ind)
                or any(glsm.dot(linrel.T).flat)
                or any(glsm.dot(self.points()[list(pts_ind)]).flat)):
            raise RuntimeError("Error finding basis")
        # We now cache the results
        if integral:
            self._glsm_charge_matrix[(pts_ind,integral)] = glsm
            self._glsm_linrels[(pts_ind,integral)] = linrel
            self._glsm_basis[(pts_ind,integral)] = basis_ind
        self._glsm_charge_matrix[(pts_ind,False)] = glsm
        self._glsm_linrels[(pts_ind,False)] = linrel
        self._glsm_basis[(pts_ind,False)] = basis_ind
        # Finally return a copy of the result
        if not include_origin and points is None:
            return np.array(self._glsm_charge_matrix[(pts_ind,integral)][:,1:])
        return np.array(self._glsm_charge_matrix[(pts_ind,integral)])

    def glsm_linear_relations(self, include_origin=True,
                              include_points_interior_to_facets=False,
                              points=None, integral=True):
        """
        **Description:**
        Computes the linear relations of the GLSM charge matrix.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.
        - `include_points_interior_to_facets` *(bool, optional,
          default=False)*: By default
          only boundary points not interior to facets are used. If this flag is
          set to true then points interior to facets are also used.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameters `include_origin` and
          `include_points_interior_to_facets` are ignored.
        - `integral` *(bool, optional, default=True)*: Indicates whether
          to find an integral basis for the columns of the GLSM charge matrix.
          (i.e. so that remaining columns can be written as an integer linear
          combination of the basis elements.)

        **Returns:**
        *(numpy.ndarray)* A matrix of linear relations of the columns of the
        GLSM charge matrix.

        **Example:**
        We construct a polytope and find its GLSM charge matrix and linear
        relations with different parameters.
        ```python {2,8,14,19}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_linear_relations()
        # array([[ 1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3],
        #        [ 0,  6,  0, -1,  0,  0,  2],
        #        [ 0,  1,  0,  0, -1,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0]])
        p.glsm_linear_relations().dot(p.glsm_charge_matrix().T) # By definition this product must be zero
        # array([[0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0]])
        p.glsm_linear_relations(include_origin=False) # Excludes the canonical divisor
        # array([[ 9, -1,  0,  0,  0,  3],
        #        [ 6,  0, -1,  0,  0,  2],
        #        [ 1,  0,  0, -1,  0,  0],
        #        [ 1,  0,  0,  0, -1,  0]])
        p.glsm_linear_relations(include_points_interior_to_facets=True) # Includes points interior to facets
        # array([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3,  2,  1,  1],
        #        [ 0,  6,  0, -1,  0,  0,  2,  1,  1,  0],
        #        [ 0,  1,  0,  0, -1,  0,  0,  0,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0,  0,  0,  0]])
        ```
        """
        if points is not None:
            pts_ind = tuple(set(list(points)+[0]))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = tuple(range(self.points().shape[0]))
        else:
            pts_ind = tuple(range(self.points_not_interior_to_facets().shape[0]))
        if (pts_ind,integral) in self._glsm_linrels:
            if not include_origin and points is None:
                return np.array(self._glsm_linrels[(pts_ind,integral)][1:,1:])
            return np.array(self._glsm_linrels[(pts_ind,integral)])
        # If linear relations are not cached we just call the GLSM charge
        # matrix function since they are computed there
        self.glsm_charge_matrix(include_origin=True,
                                include_points_interior_to_facets=include_points_interior_to_facets,
                                points=points, integral=integral)
        if not include_origin and points is None:
            return np.array(self._glsm_linrels[(pts_ind,integral)][1:,1:])
        return np.array(self._glsm_linrels[(pts_ind,integral)])

    def glsm_basis(self, include_origin=True,
                   include_points_interior_to_facets=False, points=None,
                   integral=True):
        """
        **Description:**
        Computes a basis of columns of the GLSM charge matrix.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.
        - `include_points_interior_to_facets` *(bool, optional,
          default=False)*: By default
          only boundary points not interior to facets are used. If this flag is
          set to true then points interior to facets are also used.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameters `include_origin` and
          `include_points_interior_to_facets` are ignored. Also, note that
          the indices returned here will be the indices of the sorted list
          of points.
        - `integral` *(bool, optional, default=True)*: Indicates whether
          to find an integral basis for the columns of the GLSM charge matrix.
          (i.e. so that remaining columns can be written as an integer linear
          combination of the basis elements.)

        **Returns:**
        *(numpy.ndarray)* A list of column indices that form a basis.

        **Example:**
        We construct a polytope, find its GLSM charge matrix and a basis of
        columns.
        ```python {3,6}
        import numpy as np
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_basis()
        # array([1, 6])
        glsm = p.glsm_charge_matrix()
        np.linalg.matrix_rank(glsm) == np.linalg.matrix_rank(glsm[:,p.glsm_basis()]) # This shows that the columns form a basis
        # True
        ```
        """
        if points is not None:
            pts_ind = tuple(set(list(points)+[0]))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = tuple(range(self.points().shape[0]))
        else:
            pts_ind = tuple(range(self.points_not_interior_to_facets().shape[0]))
        if (pts_ind,integral) in self._glsm_basis:
            if not include_origin and points is None:
                return np.array(self._glsm_basis[(pts_ind,integral)]) - 1
            return np.array(self._glsm_basis[(pts_ind,integral)])
        # If basis is not cached we just call the GLSM charge matrix function
        # since it is computed there
        self.glsm_charge_matrix(include_origin=True,
                                include_points_interior_to_facets=include_points_interior_to_facets,
                                points=points, integral=integral)
        if not include_origin and points is None:
            return np.array(self._glsm_basis[(pts_ind,integral)]) - 1
        return np.array(self._glsm_basis[(pts_ind,integral)])

    def volume(self):
        """
        **Description:**
        Returns the volume of the polytope.

        :::important
        By convention, the standard simplex has unit volume. To get the more
        typical Euclidean volume it must be multiplied by $d!$.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The volume of the polytope.

        **Example:**
        We construct a standard simplex and a cube, and find their volumes.
        ```python {3,5}
        p1 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        p2 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
        p1.volume()
        # 1
        p2.volume()
        # 6
        ```
        """
        if self._volume is not None:
            return self._volume
        if self._dim == 0:
            self._volume = 0
        elif self._dim == 1:
            self._volume = max(self._optimal_pts) - min(self._optimal_pts)
        else:
            self._volume = int(round(ConvexHull(self._optimal_pts).volume * math.factorial(self._dim)))
        return self._volume

    def points_to_indices(self, points):
        """
        **Description:**
        Returns the list of indices corresponding to the given points. It also
        accepts a single point, in which case it returns the corresponding
        index.

        **Arguments:**
        - `points` *(array_like)*: A point or a list of points.

        **Returns:**
        *(numpy.ndarray or int)* The list of indices corresponding to the given
        points, or the index of the point if only one is given.

        **Example:**
        We construct a polytope and find the indices of some of its points.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points_to_indices([-1,-1,-6,-9]) # We input a single point, so a single index is returned
        # 1
        p.points_to_indices([[-1,-1,-6,-9],[0,0,0,0],[0,0,1,0]]) # We input a list of points, so a list of indices is returned
        # array([1, 0, 3])
        ```
        """
        if self._pts_dict is None:
            self._points_saturated()
        if len(np.array(points).shape) == 1:
            if np.array(points).shape[0] == 0:
                return np.zeros(0, dtype=int)
            return self._pts_dict[tuple(points)]
        return np.array([self._pts_dict[tuple(pt)] for pt in points])

    def normal_form(self, affine_transform=False, backend="palp"):
        """
        **Description:**
        Returns the normal form of the polytope as defined by Kreuzer-Skarke.

        **Arguments:**
        - `affine_transform` *(bool, optional, default=False)*: Flag that
          determines whether to only use $SL^{\pm}(d,\mathbb{Z})$
          transformations or also allow translations.
        - `backend` *(str, optional, default="palp")*: Selects which
          backend to use. Options are "native", which uses native python code,
          or "palp", which uses PALP for the computation. There is a different
          convention for affine normal forms between the native algorithm and
          PALP, and PALP generally works better.

        **Returns:**
        *(numpy.ndarray)* The list of vertices in normal form.

        **Example:**
        We construct a polytope, and find its linear and affine normal forms.
        ```python {2,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.normal_form()
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [ 0,  0,  0,  1],
        #        [-9, -6, -1, -1]])
        p.normal_form(affine_transform=True)
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [12, 17, 17, 18],
        #        [ 0,  0,  0,  0]])
        ```
        """
        # This function is based on code by Andrey Novoseltsev, Samuel Gonshaw,
        # Jan Keitel, and others, and is redistributed under the GNU General
        # Public License version 2+.
        # The original code can be found at:
        # https://github.com/sagemath/sage/blob/develop/src/sage/geometry/lattice_polytope.py
        # https://trac.sagemath.org/ticket/13525
        if backend not in ("native", "palp"):
            raise ValueError("Error: options for backend are \"native\" and "
                             "\"palp\".")
        args_id = 1*affine_transform + affine_transform*(backend=="native")*1
        if self._normal_form[args_id] is not None:
            return np.array(self._normal_form[args_id])
        if backend == "palp":
            if not self.is_solid():
                warnings.warn("PALP doesn't support polytopes that are not "
                              "full-dimensional. Using native backend.")
                backend = "native"
        if backend == "palp":
            palp = subprocess.Popen((config.palp_path + "poly.x", ("-A" if affine_transform else "-N")),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
            pt_list = ""
            optimal_pts = {tuple(pt) for pt in self._optimal_pts}
            for pt in optimal_pts:
                pt_str = str(pt).replace("(","").replace(")","")
                pt_str = pt_str.replace(","," ")
                pt_list += pt_str + "\n"
            palp_in = f"{len(optimal_pts)} {self._dim}\n{pt_list}\n"
            palp_out = palp.communicate(input=palp_in)[0]
            if "ormal form" not in palp_out:
                raise RuntimeError(f"PALP error. Full output: {palp_out}")
            palp_out = palp_out.split("\n")
            for i in range(len(palp_out)):
                if "ormal form" not in palp_out[i]:
                    continue
                pts_shape = [int(c) for c in palp_out[i].split()[:2]]
                tmp_pts = np.empty(pts_shape, dtype=int)
                for j in range(pts_shape[0]):
                    tmp_pts[j,:] = [int(c) for c in palp_out[i+j+1].split()]
                break
            points = (tmp_pts.T if pts_shape[0] < pts_shape[1] else tmp_pts)
            self._normal_form[args_id] = points
            return np.array(self._normal_form[args_id])
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
            m = np.argmax([PM[0,prm[0][1].dot(range(n_v))][i] for i in range(j, n_v)])
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
            d = PM[k,prm[n_s][1].dot(range(n_v))][0] - prm[0][1].dot(first_row)[0]
            if d < 0:
                # The largest elt of this row is smaller than largest elt
                # in 1st row, so nothing to do
                continue
            # otherwise:
            for i in range(1, n_v):
                m = np.argmax([PM[k,prm[n_s][1].dot(range(n_v))][j] for j in range(i, n_v)])
                if m > 0:
                    prm[n_s][1] = PGE(n_v, i+1, m+i+1).dot(prm[n_s][1])
                if d == 0:
                    d = PM[k,prm[n_s][1].dot(range(n_v))][i] - prm[0][1].dot(first_row)[i]
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
                        v = PM[prmb[n_p][0].dot(range(n_f)),:][:,prmb[n_p][1].dot(range(n_v))][s]
                        if v[0] < v[j]:
                            prmb[n_p][1] = PGE(n_v, 1, j+1).dot(prmb[n_p][1])
                    if ccf == 0:
                        l_r[0] = PM[prmb[n_p][0].dot(range(n_f)),:][:,prmb[n_p][1].dot(range(n_v))][s,0]
                        prmb[n_p][0] = PGE(n_f, l+1, s+1).dot(prmb[n_p][0])
                        n_p += 1
                        ccf = 1
                        prmb[n_p] = copy.copy(prm[k])
                    else:
                        d1 = PM[prmb[n_p][0].dot(range(n_f)),:][:,prmb[n_p][1].dot(range(n_v))][s,0]
                        d = d1 - l_r[0]
                        if d < 0:
                            # We move to the next line
                            continue
                        if d == 0:
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
                            v = PM[prmb[s][0].dot(range(n_f)),:][:,prmb[s][1].dot(range(n_v))][l]
                            if v[c] < v[j]:
                                prmb[s][1] = PGE(n_v, c+1, j+1).dot(prmb[s][1])
                        if ccf == 0:
                            # Set reference and carry on to next permutation
                            l_r[c] = PM[prmb[s][0].dot(range(n_f)),:][:,prmb[s][1].dot(range(n_v))][l,c]
                            ccf = 1
                        else:
                            d1 = PM[prmb[s][0].dot(range(n_f)),:][:,prmb[s][1].dot(range(n_v))][l,c]
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
                M = PM[prm[0][0].dot(range(n_f)),:][:,prm[0][1].dot(range(n_v))][l]
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
        # Perform a translation if necessary
        if affine_transform:
            v0 = copy.copy(V[0])
            for i in range(n_v):
                V[i] -= v0
        # Finally arrange the points the the canonical order
        p_c = np.eye(n_v, dtype=int)
        M_max = [max([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        S_max = [sum([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        for i in range(n_v):
            k = i
            for j in range(i+1, n_v):
                if M_max[j] < M_max[k] or (M_max[j] == M_max[k] and S_max[j] < S_max[k]):
                    k = j
            if not k == i:
                M_max[i], M_max[k] = M_max[k], M_max[i]
                S_max[i], S_max[k] = S_max[k], S_max[i]
                p_c = PGE(n_v, 1+i, 1+k).dot(p_c)
        # Create array of possible NFs.
        prm = [p_c.dot(l[1]) for l in prm.values()]
        Vs = [np.array(fmpz_mat(V.T[:,sig.dot(range(n_v))].tolist()).hnf().tolist(), dtype=int).tolist() for sig in prm]
        Vmin = min(Vs)
        if affine_transform:
            self._normal_form[args_id] = np.array(Vmin).T[:,:self._dim]
        else:
            self._normal_form[args_id] = np.array(Vmin).T
        return np.array(self._normal_form[args_id])

    def automorphisms(self, square_to_one=False, action="right", as_dictionary=False):
        """
        **Description:**
        Returns the $SL^{\pm}(d,\mathbb{Z})$ matrices that leave the polytope
        invariant. These matrices act on the points by multiplication on the
        right.

        **Arguments:**
        - `square_to_one` *(bool, optional, default=False)*: Flag that
          restricts to only matrices that square to the identity.
        - `action` *(str, optional, default="right")*: Flag that specifies
          whether the returned matrices act on the left or the right. This
          option is ignored when `as_dictionary` is set to True.
        - `as_dictionary` *(bool, optional, default=False)*: Return each
          automphism as a dictionary that describes the action on the
          indices of the points.

        **Returns:**
        *(numpy.ndarray or dict)* A list of automorphism matrices or dictionaries.

        **Example:**
        We construct a polytope, and find its automorphisms. We also check that
        one of the non-trivial automorphisms is indeed an automorphism by
        checking that it acts as a permutation on the vertices. We also show
        how to get matrices that act on the left, which are simply the transpose
        matrices, and we show how to get dictionaries that describe how the
        indices of the points transform.
        ```python {2,20,31,37}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        autos = p.automorphisms()
        a = autos[1]
        print(a)
        # [[ 1,  0,  0,  0],
        #  [-1, -1, -6, -9],
        #  [ 0,  0,  1,  0],
        #  [ 0,  0,  0,  1]])
        print(f"{p.vertices()}\n{p.vertices().dot(a)}") # Print vertices before and after applying the automorphism
        # [[ 1  0  0  0]
        #  [ 0  1  0  0]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]
        #  [-1 -1 -6 -9]]
        # [[ 1  0  0  0]
        #  [-1 -1 -6 -9]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]
        #  [ 0  1  0  0]]
        autos2 = p.automorphisms(square_to_one=True)
        a2 = autos2[1]
        print(f"{a2}\n{a2.dot(a2)}") # Print the automorphism and its square
        # [[ 1  0  0  0]
        #  [-1 -1 -6 -9]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]]
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]
        #  [0 0 0 1]]
        autos_left = p.automorphisms(square_to_one=True, action="left")
        print(autos_left[1].dot(p.vertices().T)) # The vertices are now columns
        # [[ 1 -1  0  0  0]
        # [ 0 -1  0  0  1]
        # [ 0 -6  1  0  0]
        # [ 0 -9  0  1  0]]
        autos_dict = p.automorphisms(as_dictionary=True)
        print(autos_dict)
        # [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 4, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 4, 2: 2, 3: 3, 4: 5, 5: 1, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 5, 2: 2, 3: 3, 4: 1, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6, 7: 7, 8: 8, 9: 9}]
        ```
        """
        if self.dim() != self.ambient_dim():
            raise NotImplementedError("Automorphisms can only be computed for full-dimensional polytopes.")
        if action not in ("right", "left"):
            raise ValueError("Options for action are \"right\" or \"left\".")
        args_id = 1*square_to_one + 2*as_dictionary
        if self._autos[args_id] is not None:
            if as_dictionary:
                return copy.deepcopy(self._autos[args_id])
            if action == "left":
                return np.array([a.T for a in self._autos[args_id]])
            return np.array(self._autos[args_id])
        if self._autos[0] is None:
            vert_set = set(tuple(pt) for pt in self.vertices())
            f_min = None
            for f in self.facets():
                if f_min is None or len(f.vertices()) < len(f_min.vertices()):
                    f_min = f
            f_min_vert_rref = np.array(fmpz_mat(f_min.vertices().T.tolist()).hnf().tolist(), dtype=int)
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
                if not all(abs(c.q) == 1 for c in np.array(m.tolist()).flatten()):
                    continue
                m = np.array([[int(c.p)//int(c.q) for c in r] # just in case c.q==-1 by some weird reason
                            for r in np.array(m.tolist())], dtype=int)
                if set(tuple(pt) for pt in np.dot(self.vertices(), m)) != vert_set:
                    continue
                autos.append(m)
                if all((np.dot(m,m) == np.eye(self.dim(), dtype=int)).flatten()):
                    autos2.append(m)
            self._autos[0] = np.array(autos)
            self._autos[1] = np.array(autos2)
        if as_dictionary and self._autos[2] is None:
            autos_dict = []
            autos2_dict = []
            pts_tup = [tuple(pt) for pt in self.points()]
            for a in self._autos[0]:
                new_pts_tup = [tuple(pt) for pt in self.points().dot(a)]
                autos_dict.append({i:new_pts_tup.index(ii) for i,ii in enumerate(pts_tup)})
            for a in self._autos[1]:
                new_pts_tup = [tuple(pt) for pt in self.points().dot(a)]
                autos2_dict.append({i:new_pts_tup.index(ii) for i,ii in enumerate(pts_tup)})
            self._autos[2] = autos_dict
            self._autos[3] = autos2_dict
        if as_dictionary:
            return copy.deepcopy(self._autos[args_id])
        if action == "left":
            return np.array([a.T for a in self._autos[args_id]])
        return np.array(self._autos[args_id])

    def find_2d_reflexive_subpolytopes(self):
        """
        **Description:**
        Use the algorithm by Huang and Taylor described in
        [1907.09482](https://arxiv.org/abs/1907.09482) to find 2D reflexive
        subpolytopes in 4D polytopes.

        **Arguments:**
        None.

        **Returns:**
        *(list)* The list of 2D reflexive subpolytopes.

        **Example:**
        We construct a polytope and find its 2D reflexive subpolytopes.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.find_2d_reflexive_subpolytopes()
        # [A 2-dimensional lattice polytope in ZZ^4]
        ```
        """
        if not self.is_reflexive() or self.dim() != 4:
            raise NotImplementedError("Only 4D reflexive polytopes are supported.")
        pts = self.points()
        dual_vert = self.dual().vertices()
        # Construct the sets S_i by finding the maximum dot product with dual vertices
        S_i = [[]]*3
        for p in pts:
            m = max(p.dot(v) for v in dual_vert)
            if m in (1,2,3):
                S_i[m-1].append(tuple(p))
        # Check each of the three conditions
        gen_pts = []
        for i in range(len(S_i[0])):
            if tuple(-np.array(S_i[0][i])) in S_i[0]:
                for j in range(i+1,len(S_i[0])):
                    if (tuple(-np.array(S_i[0][j])) in S_i[0]
                            and tuple(-np.array(S_i[0][i]))!=S_i[0][j]):
                        gen_pts.append((S_i[0][i],S_i[0][j]))
        for i in range(len(S_i[1])):
            for j in range(i+1,len(S_i[1])):
                p = tuple(-np.array(S_i[1][i])-np.array(S_i[1][j]))
                if p in S_i[0] or p in S_i[1]:
                    gen_pts.append((S_i[1][i],S_i[1][j]))
        for i in range(len(S_i[2])):
            for j in range(i+1,len(S_i[2])):
                p = -np.array(S_i[2][i])-np.array(S_i[2][j])
                if all(c%2 == 0 for c in p) and tuple(p//2) in S_i[0]:
                    gen_pts.append((S_i[2][i],S_i[2][j]))
        polys_2d = set()
        for p1,p2 in gen_pts:
            pts_2d = set()
            for p in pts:
                if np.linalg.matrix_rank((p1,p2,p)) == 2:
                    pts_2d.add(tuple(p))
            if np.linalg.matrix_rank(list(pts_2d)) == 2:
                polys_2d.add(tuple(sorted(pts_2d)))
        return [Polytope(pp) for pp in polys_2d]

    def minkowski_sum(self, other):
        """
        **Description:**
        Returns the Minkowski sum of the two polytopes.

        **Arguments:**
        - `other` *(Polytope)*: The other polytope used for the Minkowski
          sum.

        **Returns:**
        *(Polytope)* The Minkowski sum.

        **Example:**
        We construct two polytops and compute their Minkowski sum.
        ```python {3}
        p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
        p2 = Polytope([[0,0,1],[0,0,-1]])
        p1.minkowski_sum(p2)
        # A 3-dimensional reflexive lattice polytope in ZZ^3
        ```
        """
        points = []
        for p1 in self.vertices():
            for p2 in other.vertices():
                points.append(p1+p2)
        return Polytope(points)

    def nef_partitions(self, keep_symmetric=False, keep_products=False,
                       keep_projections=False, codim=2,
                       compute_hodge_numbers=True,
                       return_hodge_numbers=False):
        """
        **Description:**
        Computes the nef partitions of the polytope using PALP.

        :::note
        This is currently an experimental feature and may change significantly
        in future versions.
        :::

        **Arguments:**
        - `keep_symmetric` *(bool, optional, default=False)*: Keep
          symmetric partitions related by lattice automorphisms.
        - `keep_products` *(bool, optional, default=False)*: Keep
          product partitions corresponding to complete intersections being
          direct products.
        - `keep_projections` *(bool, optional, default=False)*: Keep
          projection partitions, i.e. partitions where one of the parts
          consists of a single vertex.
        - `codim` *(int, optional, default=2)*: The number of parts in the
          partition or, equivalently, the codimension of the complete
          intersection Calabi-Yau.
        - `compute_hodge_numbers` *(bool, optional, default=True)*:
          Indicates whether Hodge numbers of the CICY are computed.
        - `return_hodge_numbers` *(bool, optional, default=True)*:
          Indicates whether to return the Hodge numbers along with the nef
          partitions. They are returned in a separate tuple and they are
          ordered as in the Hodge diamond from top to bottom and left to right.

        **Returns:**
        *(tuple)* The nef partitions of the polytope. If return_hodge_numbers
        is set to True then two tuples are returned, one with the nef
        partitions and one with the corresponding Hodge numbers.

        **Example:**
        We construct a tesseract and find the 2- and 3-part nef partitions.
        ```python {2,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
        nef_part_2 = p.nef_partitions() # Default codimension is 2
        print(nef_part_2[0]) # Print the first of the nef partitions
        # ((5, 2, 3, 4), (8, 7, 6, 1))
        nef_part_3 = p.nef_partitions(codim=3) # Codimension 3
        print(nef_part_3[0]) # Print the first of the nef partitions
        # ((6, 5, 3), (2, 4), (8, 7, 1))
        ```
        """
        if not config._exp_features_enabled:
            raise Exception("The experimental features must be enabled to "
                            "compute nef partitions.")
        if return_hodge_numbers:
            compute_hodge_numbers = True
        args_id = (keep_symmetric,keep_products,keep_projections,codim,
                   compute_hodge_numbers)
        if self._nef_parts.get(args_id,None) is not None:
            return (self._nef_parts.get(args_id) if return_hodge_numbers
                                                    or not compute_hodge_numbers
                                                else self._nef_parts.get(args_id)[0])
        if not self.is_reflexive():
            raise ValueError("The polytope must be reflexive")
        flags = ("-N", "-V", "-p", f"-c{codim}")
        if keep_symmetric:
            flags += ("-s",)
        if keep_products:
            flags += ("-D",)
        if keep_projections:
            flags += ("-P",)
        palp = subprocess.Popen((config.palp_path + "nef-11d.x",)+flags,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        vert_str = ""
        vert = [tuple(pt) for pt in self.vertices()]
        for pt in vert:
            vert_str += (str(pt).replace("(","").replace(")","").replace(","," ") + "\n")
        palp_out = palp.communicate(input=f"{len(vert)} {self._dim}\n" + vert_str + "\n")[0]
        if "Vertices of P" not in palp_out:
            raise RuntimeError(f"PALP error. Full output: {palp_out}")
        palp_out = palp_out.split("\n")
        n_parts = 0
        # Read number of nef partitions and vertices to make sure it looks right
        for i,line in enumerate(palp_out):
            if "#part" in line:
                n_parts = int(line.split("=")[-1])
            if "Vertices of P" not in line:
                continue
            pts_shape = [int(c) for c in line.split()[:2]]
            tmp_pts = np.empty(pts_shape, dtype=int)
            for j in range(pts_shape[0]):
                tmp_pts[j,:] = [int(c) for c in palp_out[i+j+1].split()]
            nef_part_start = i+j+2
            break
        pts_out = (tmp_pts.T if pts_shape[0] < pts_shape[1] else tmp_pts)
        nef_parts = []
        for n in range(n_parts):
            if "V" not in palp_out[nef_part_start+n]:
                break
            tmp_partition = []
            for nn in range(codim-1):
                tmp_part = []
                start = palp_out[nef_part_start+n].find(f"V{nn if codim>2 else ''}:")
                end = palp_out[nef_part_start+n][start:].find("  ")+start
                for s in palp_out[nef_part_start+n][start+(2 if codim==2 else 3):end].split():
                    if "V" in s or "D" in s or "P" in s or "sec" in s:
                        break
                    tmp_part.append(int(s))
                tmp_partition.append(tmp_part)
            tmp_part = [i for i in range(len(vert)) if not any(i in part for part in tmp_partition)]
            tmp_partition.append(tmp_part)
            # We have to reindex to match polytope indices
            nef_parts.append(tuple(tuple(self.points_to_indices(pts_out[part])) for part in tmp_partition))
        if compute_hodge_numbers:
            flags = ("-N", "-V", "-H", f"-c{codim}")
            if keep_symmetric:
                flags += ("-s",)
            if keep_products:
                flags += ("-D",)
            if keep_projections:
                flags += ("-P",)
            cy_dim = self._dim - codim
            palp = subprocess.Popen((config.palp_path + "nef-11d.x",)+flags,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
            palp_out = palp.communicate(input=f"{len(vert)} {self._dim}\n" + vert_str + "\n")[0]
            data = palp_out.split(f"h {cy_dim} {cy_dim}")[1:]
            hodge_nums = []
            for i in range(len(data)):
                hodge_nums.append(tuple(int(h) for h in data[i].split()[:(cy_dim+1)**2]))
            if len(hodge_nums) != len(nef_parts):
                raise RuntimeError("Unexpected length mismatch.")
            nef_parts = (tuple(nef_parts),tuple(hodge_nums))
        self._nef_parts[args_id] = tuple(nef_parts)
        return (self._nef_parts.get(args_id) if return_hodge_numbers
                                                or not compute_hodge_numbers
                                            else self._nef_parts.get(args_id)[0])

    def _triang_pt_inds(self, include_points_interior_to_facets=None, points=None):
        """
        **Description:**
        Constructs the list of indices of the points that will be used in a
        triangulation.

        :::note
        Typically this function should not be called by the user. Instead, it
        is called by various other functions in the Polytope class.
        :::

        **Arguments:**
        - `include_points_interior_to_facets` *(bool, optional)*: Whether
          to include points interior to facets from the triangulation. If not
          specified, it is set to False for reflexive polytopes and True
          otherwise.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameter `include_points_interior_to_facets` is ignored.

        **Returns:**
        *(tuple)* A tuple of the indices of the points that will be included in
        a triangulation

        **Example:**
        We construct triangulations in various ways. We use the
        [`triangulate`](#triangulate) function instead of using this function
        directly.
        ```python {2,5,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t1 = p.triangulate()
        print(t1)
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 7 points in ZZ^4
        t2 = p.triangulate(include_points_interior_to_facets=True)
        print(t2)
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 10 points in ZZ^4
        t3 = p.triangulate(points=[1,2,3,4,5])
        print(t3)
        # A fine, regular, non-star triangulation of a 4-dimensional point configuration with 5 points in ZZ^4
        ```
        """
        if points is not None:
            pts_ind = tuple(set(points))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
        elif include_points_interior_to_facets is None:
            pts_ind = (tuple(self.points_not_interior_to_facets(as_indices=True))
                        if self.is_reflexive()
                        else tuple(self.points(as_indices=True)))
        else:
            pts_ind = (tuple(self.points(as_indices=True))
                        if include_points_interior_to_facets
                        else tuple(self.points_not_interior_to_facets(as_indices=True)))
        pts_ind = tuple(sorted(pts_ind))
        return pts_ind

    def triangulate(self, heights=None, make_star=None,
                    include_points_interior_to_facets=None, points=None,
                    simplices=None, check_input_simplices=True, backend="cgal"):
        """
        **Description:**
        Returns a single regular triangulation of the polytope.

        :::note
        When reflexive polytopes are used, it defaults to returning a fine,
        regular, star triangulation.
        :::

        **Arguments:**
        - `heights` *(array_like, optional)*: A list of heights specifying
          the regular triangulation. When not specified, it will return the
          Delaunay triangulation when using CGAL, a triangulation obtained from
          random heights near the Delaunay when using QHull, or the placing
          triangulation when using TOPCOM. Heights can only be specified when
          using CGAL or QHull as the backend.
        - `make_star` *(bool, optional)*: Indicates whether to
          turn the triangulation into a star triangulation by deleting
          internal lines and connecting all points to the origin, or
          equivalently by decreasing the height of the origin to be much lower
          than the rest. By default, this flag is set to true if the polytope
          is reflexive and neither heights or simplices are inputted.
          Otherwise, it is set to False.
        - `include_points_interior_to_facets` *(bool, optional)*: Whether
          to include points interior to facets from the triangulation. If not
          specified, it is set to False for reflexive polytopes and True
          otherwise.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameter `include_points_interior_to_facets` is ignored.
        - `simplices` *(array_like, optional)*: A list of simplices
          specifying the triangulation. This is useful when a triangulation was
          previously computed and it needs to be used again. Note that the
          order of the points needs to be consistent with the order that the
          `Polytope` class uses.
        - `check_input_simplices` *(bool, optional, default=True)*: Flag
          that specifies whether to check if the input simplices define a valid
          triangulation.
        - `backend` *(str, optional, default="cgal")*: Specifies the
          backend used to compute the triangulation. The available options are
          "qhull", "cgal", and "topcom". CGAL is the default one as it is very
          fast and robust.

        **Returns:**
        *(Triangulation)* A [`Triangulation`](./triangulation) object
        describing a triangulation of the polytope.

        **Example:**
        We construct a triangulation of a reflexive polytope and check that by
        default it is a fine, regular, star triangulation. We also try
        constructing triangulations with heights, input simplices, and using
        the other backends.
        ```python {2,4,6,8,10}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
        p.triangulate()
        # A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
        p.triangulate(heights=[-30,5,5,24,-19,-14,29])
        # A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
        p.triangulate(simplices=[[0,1,2,3,4],[0,1,2,3,5],[0,1,2,4,6],[0,1,2,5,6],[0,1,3,4,5],[0,1,4,5,6],[0,2,3,4,5],[0,2,4,5,6]])
        # A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
        p.triangulate(backend="qhull")
        # A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
        p.triangulate(backend="topcom")
        # A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
        ```
        """
        pts_ind = self._triang_pt_inds(include_points_interior_to_facets, points)
        triang_pts = self.points()[list(pts_ind)]
        if (heights is not None) and (len(heights) == len(self.points())):
            triang_heights = np.array(heights)[list(pts_ind)]
        else:
            triang_heights = heights
        if make_star is None:
            if heights is None and simplices is None:
                make_star = self.is_reflexive()
            else:
                make_star = False
        if not self.is_reflexive() and (0,)*self._dim not in [tuple(pt) for pt in triang_pts]:
            make_star = False
        return Triangulation(triang_pts, poly=self, heights=triang_heights,
                             make_star=make_star, simplices=simplices,
                             check_input_simplices=check_input_simplices,
                             backend=backend)

    def random_triangulations_fast(self, N=None, c=0.2, max_retries=500,
                                   make_star=True, only_fine=True,
                                   include_points_interior_to_facets=None,
                                   points=None, backend="cgal", as_list=False,
                                   progress_bar=True, seed=None):
        """
        **Description:**
        Constructs pseudorandom regular (optionally fine and star)
        triangulations of a given point set. This is done by picking random
        heights around the Delaunay heights from a Gaussian distribution.

        :::caution important
        This function produces random triangulations very quickly, but it does
        not produce a fair sample. When a fair sampling is required the
        [`random_triangulations_fair`](#random_triangulations_fair)
        function should be used.
        :::

        **Arguments:**
        - `N` *(int, optional)*: Number of desired unique triangulations.
          If not specified, it will generate as many triangulations as it can
          find until it has to retry more than `max_retries` times to
          obtain a new triangulation. This parameter is required when setting
          `as_list` to True.
        - `c` *(float, optional, default=0.2)*: A contant used as the
          standard deviation of the Gaussian distribution used to pick the
          heights. A larger `c` results in a wider range of possible
          triangulations, but with a larger fraction of them being non-fine,
          which slows down the process when `only_fine` is set to True.
        - `max_retries` *(int, optional, default=50)*: Maximum number of
          attempts to obtain a new triangulation before the process is
          terminated.
        - `make_star` *(bool, optional)*: Converts the obtained
          triangulations into star triangulations. If not specified, defaults
          to True for reflexive polytopes, and False for other polytopes.
        - `only_fine` *(bool, optional, default=True)*: Restricts to fine
          triangulations.
        - `include_points_interior_to_facets` *(bool, optional)*: Whether
          to include points interior to facets from the triangulation. If not
          specified, it is set to False for reflexive polytopes and True
          otherwise.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameter `include_points_interior_to_facets` is ignored.
        - `backend` *(str, optional, default="cgal")*: Specifies the
          backend used to compute the triangulation. The available options are
          "cgal" and "qhull".
        - `as_list` *(bool, optional, default=False)*: By default this
          function returns a generator object, which is usually desired for
          efficiency. However, this flag can be set to True so that it returns
          the full list of triangulations at once.
        - `progress_bar` *(bool, optional, default=True)*: Shows the number
          of triangulations obtained and progress bar. Note that this option is
          only available when returning a list instead of a generator.
        - `seed` *(int, optional)*: A seed for the random number generator.
          This can be used to obtain reproducible results.

        **Returns:**
        *(generator or list)* A generator of
        [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set
        to True.

        **Example:**
        We construct a polytope and find some random triangulations. The
        triangulations are obtained very quicly, but they are not a fair sample
        of the space of triangulations. For a fair sample, the
        [`random_triangulations_fair`](#random_triangulations_fair)
        function should be used.
        ```python {2,7}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        g = p.random_triangulations_fast()
        next(g) # Runs very quickly
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        next(g) # Keeps producing triangulations until it has trouble finding more
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        rand_triangs = p.random_triangulations_fast(N=10, as_list=True) # Produces the list of 10 triangulations very quickly
        ```
        """
        if self._ambient_dim > self._dim:
            raise NotImplementedError("Only triangulations of full-dimensional polytopes"
                                      "are supported.")
        if N is None and as_list:
            raise ValueError("Number of triangulations must be specified when "
                             "returning a list.")
        pts_ind = self._triang_pt_inds(include_points_interior_to_facets, points)
        triang_pts = [tuple(pt) for pt in self.points()[list(pts_ind)]]
        if make_star is None:
            make_star = self.is_reflexive()
        if (0,)*self._dim not in triang_pts:
            make_star = False
        g = random_triangulations_fast_generator(triang_pts, N=N, c=c,
                max_retries=max_retries, make_star=make_star,
                only_fine=only_fine, backend=backend, poly=self, seed=seed)
        if not as_list:
            return g
        if progress_bar:
            pbar = tqdm(total=N)
        triangs_list = []
        while len(triangs_list) < N:
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
                                   max_steps_to_wall=25, fine_tune_steps=8,
                                   max_retries=50, make_star=None,
                                   include_points_interior_to_facets=None,
                                   points=None, backend="cgal", as_list=False,
                                   progress_bar=True, seed=None):
        """
        **Description:**
        Constructs pseudorandom regular (optionally star)
        triangulations of a given point set. Implements Algorithm \#3 from the
        paper
        *Bounding the Kreuzer-Skarke Landscape*
        by Mehmet Demirtas, Liam McAllister, and Andres Rios-Tascon.
        [arXiv:2008.01730](https://arxiv.org/abs/2008.01730)

        This is a Markov chain Monte Carlo algorithm that involves taking
        random walks inside the subset of the secondary fan corresponding to
        fine triangulations and performing random flips. For details, please
        see Section 4.1 in the paper.

        :::note notes
        - By default this function tries to guess reasonable parameters for the
          polytope that is being used. However, it may take some
          trial-and-error to find optimal parameters that produce a good
          sampling and prevent the algorithm from getting stuck.
        - This function is designed mainly for large polytopes where sampling
          triangulations is challenging. When small polytopes are used it is
          likely to get stuck or not produce an optimal set.
        :::

        **Arguments:**
        - `N` *(int, optional)*: Number of desired unique triangulations.
          If not specified, it will generate as many triangulations as it can
          find until it has to retry more than `max_retries` times to
          obtain a new triangulation. This parameter is required when setting
          `as_list` to True.
        - `n_walk` *(int, optional, default=n_points//10+10)*: Number of
          hit-and-run steps per triangulation.
        - `n_flip` *(int, optional, default=n_points//10+10)*: Number of
          random flips performed per triangulation.
        - `initial_walk_steps` *(int, optional, default=2*n_pts//10+10)*:
          Number of hit-and-run steps to take before starting to record
          triangulations. Small values may result in a bias towards
          Delaunay-like triangulations.
        - `walk_step_size` *(float, optional, default=1e-2)*: Determines
          the size of random steps taken in the secondary fan. The algorithm
          may stall if too small.
        - `max_steps_to_wall` *(int, optional, default=25)*: Maximum
          number of steps to take towards a wall of the subset of the secondary
          fan that correspond to fine triangulations. If a wall is not found, a
          new random direction is selected. Setting this to be very large
          (>100) reduces performance. If this is set to be too low, the
          algorithm may stall.
        - `fine_tune_steps` *(int, optional, default=8)*: Number of steps
          to determine the location of a wall. Decreasing improves performance,
          but might result in biased samples.
        - `max_retries` *(int, optional, default=50)*: Maximum number of
          attempts to obtain a new triangulation before the process is
          terminated.
        - `make_star` *(bool, optional)*: Converts the obtained
          triangulations into star triangulations. If not specified, defaults
          to True for reflexive polytopes, and False for other polytopes.
        - `include_points_interior_to_facets` *(bool, optional)*: Whether
          to include points interior to facets from the triangulation. If not
          specified, it is set to False for reflexive polytopes and True
          otherwise.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameter `include_points_interior_to_facets` is ignored.
        - `backend` *(str, optional, default="cgal")*: Specifies the
          backend used to compute the triangulation. The available options are
          "cgal" and "qhull".
        - `as_list` *(bool, optional, default=False)*: By default this
          function returns a generator object, which is usually desired for
          efficiency. However, this flag can be set to True so that it returns
          the full list of triangulations at once.
        - `progress_bar` *(bool, optional, default=True)*: Shows number of
          triangulations obtained and progress bar. Note that this option is
          only available when returning a list instead of a generator.
        - `seed` *(int, optional)*: A seed for the random number generator.
          This can be used to obtain reproducible results.

        **Returns:**
        *(generator or list)* A generator of
        [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set
        to True.

        **Example:**
        We construct a polytope and find some random triangulations. The
        computation takes considerable time, but they should be a fair sample
        from the full set of triangulations (if the parameters are chosen
        correctly). For (some) machine learning purposes or when the fairness
        of the sample is not crucial, the
        [`random_triangulations_fast`](#random_triangulations_fast)
        function should be used instead.
        ```python {2,7}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        g = p.random_triangulations_fast()
        next(g) # Takes a long time (around a minute)
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        next(g) # Takes slightly shorter (still around a minute)
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        rand_triangs = p.random_triangulations_fair(N=10, as_list=True) # Produces the list of 10 triangulations, but takes a long time (around 10 minutes)
        ```
        It is worth noting that the time it takes to obtain each triangulation
        varies very significantly on the parameters used. The function tries to
        guess reasonable parameters, but it is better to adjust them to your
        desired balance between speed and fairness of the sampling.
        """
        if self._ambient_dim > self._dim:
            raise NotImplementedError("Only triangulations of full-dimensional polytopes"
                                      "are supported.")
        if N is None and as_list:
            raise ValueError("Number of triangulations must be specified when "
                             "returning a list.")
        pts_ind = self._triang_pt_inds(include_points_interior_to_facets, points)
        triang_pts = [tuple(pt) for pt in self.points()[list(pts_ind)]]
        if make_star is None:
            make_star =  self.is_reflexive()
        if (0,)*self._dim not in triang_pts:
            make_star = False
        if n_walk is None:
            n_walk = len(self.points())//10 + 10
        if n_flip is None:
            n_flip = len(self.points())//10 + 10
        if initial_walk_steps is None:
            initial_walk_steps = 2*len(self.points())//10 + 10
        g = random_triangulations_fair_generator(
                triang_pts, N=N, n_walk=n_walk, n_flip=n_flip,
                initial_walk_steps=initial_walk_steps,
                walk_step_size=walk_step_size,
                max_steps_to_wall=max_steps_to_wall,
                fine_tune_steps=fine_tune_steps, max_retries=max_retries,
                make_star=make_star, backend=backend, poly=self, seed=seed)
        if not as_list:
            return g
        if progress_bar:
            pbar = tqdm(total=N)
        triangs_list = []
        while len(triangs_list) < N:
            try:
                triangs_list.append(next(g))
                if progress_bar:
                    pbar.update(len(triangs_list)-pbar.n)
            except StopIteration:
                if progress_bar:
                    pbar.update(N-pbar.n)
                break
        return triangs_list

    def all_triangulations(self, only_fine=True, only_regular=True,
                           only_star=None, star_origin=None,
                           include_points_interior_to_facets=None,
                           points=None, backend=None, as_list=False,
                           raw_output=False):
        """
        **Description:**
        Computes all triangulations of the polytope using TOPCOM. There is the
        option to only compute fine, regular or fine triangulations.

        :::caution warning
        Polytopes with more than around 15 points usually have too many
        triangulations, so this function may take too long or run out of
        memory.
        :::

        **Arguments:**
        - `only_fine` *(bool, optional, default=True)*: Restricts to only
          fine triangulations.
        - `only_regular` *(bool, optional, default=True)*: Restricts to
          only regular triangulations.
        - `only_star` *(bool, optional)*: Restricts to only
            star triangulations. When not specified it defaults to True for
            reflexive polytopes and False otherwise.
        - `star_origin` *(int, optional)*: The index of the point that
          will be used as the star origin. If the polytope is reflexive this
          is set to 0, but otherwise it must be specified.
        - `include_points_interior_to_facets` *(bool, optional)*: Whether
          to include points interior to facets from the triangulation. If not
          specified, it is set to False for reflexive polytopes and True
          otherwise.
        - `points` *(array_like, optional)*: The list of indices of the
          points that will be used. Note that if this option is used then the
          parameter `include_points_interior_to_facets` is ignored.
        - `backend` *(str, optional)*: The optimizer used to check
          regularity computation. The available options are the backends of the
          [`is_solid`](./cone#is_solid) function of the
          [`Cone`](./cone) class. If not specified, it will be picked
          automatically. Note that TOPCOM is not used to check regularity since
          it is much slower.
        - `as_list` *(bool, optional, default=False)*: By default this
          function returns a generator object, which is usually desired for
          efficiency. However, this flag can be set to True so that it returns
          the full list of triangulations at once.
        - `raw_output` *(bool, optional, default=False)*: Return the
          triangulations as lists of simplices instead of as Triangulation
          objects.

        **Returns:**
        *(generator or list)* A generator of
        [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set
        to True.

        **Example:**
        We construct a polytope and find all of its triangulations. We try
        picking different restrictions and see how the number of triangulations
        changes.
        ```python {2,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
        g = p.all_triangulations()
        next(g) # Takes some time while TOPCOM finds all the triangulations
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 7 points in ZZ^4
        next(g) # Produces the next triangulation immediately
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 7 points in ZZ^4
        len(p.all_triangulations(as_list=True)) # Number of fine, regular, star triangulations
        # 2
        len(p.all_triangulations(only_regular=False, only_star=False, only_fine=False, as_list=True) )# Number of triangularions, no matter if fine, regular, or star
        # 6
        ```
        """
        if only_star is None:
            only_star = self.is_reflexive()
        if only_star and star_origin is None:
            if self.is_reflexive():
                star_origin = 0
            else:
                raise ValueError("The star_origin parameter must be specified "
                                 "when finding star triangulations of "
                                 "non-reflexive polytopes.")
        pts_ind = self._triang_pt_inds(include_points_interior_to_facets, points)
        triang_pts = [tuple(pt) for pt in self.points()[list(pts_ind)]]
        if len(triang_pts) >= 17:
            warnings.warn("Polytopes with more than around 17 points usually "
                          "have too many triangulations, so this function may take "
                          "too long or run out of memory.")
        triangs = all_triangulations(
                    triang_pts, only_fine=only_fine, only_regular=only_regular,
                    only_star=only_star, star_origin=star_origin,
                    backend=backend, poly=self, raw_output=raw_output)
        if as_list:
            return list(triangs)
        return triangs
