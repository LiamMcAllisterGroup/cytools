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
This module contains tools designed to Calabi-Yau hypersurface computations.
"""

from cytools.utils import *
from cytools.cone import Cone
from collections import defaultdict
from itertools import combinations
from scipy.sparse import dok_matrix, csr_matrix
from flint import fmpz_mat, fmpq_mat, fmpz, fmpq
from collections import Counter
from math import factorial
import numpy as np



class CalabiYau:
    """A class that handles computations of Calabi-Yau hypersurfaces."""

    def __init__(self, frst):
        """
        Creates a CalabiYau object.

        Args:
            frst (Triangulation): A fine, regular, star triangularion of a
                reflexive polytope.
        """
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        if not frst._allow_cy or not frst.is_star() or not frst.is_fine():
            raise Exception("The inputted triangulation is not suitable for "
                            "constructing a Calabi-Yau.")
        if not frst._poly.is_favorable():
            raise Exception("Only favorable CYs are currently supported.")
        self._frst = frst
        # Initialize remaining hidden attributes
        self._divisor_basis = None
        self._mori_cone = [None]*3
        self._ambient_intersection_numbers = [None]*3
        self._intersection_numbers = [None]*3

    def clear_cache(self, recursive=True, only_in_basis=True):
        """Clears the cached results of any previous computation."""
        self._mori_cone[2] = None
        self._ambient_intersection_numbers[2] = None
        self._intersection_numbers[2] = None
        if not only_in_basis:
            self._divisor_basis = None
            self._mori_cone = [None]*3
            self._ambient_intersection_numbers = [None]*3
            self._intersection_numbers = [None]*3
            if recursive:
                self._frst.clear_cache(recursive=True)

    def __repr__(self):
        """Returns a string describing the Calabi-Yau hypersurface."""
        d = self.dim()
        return (f"A Calabi-Yau {d}-fold hypersurface "
                + (f"with h11={self.h11()} and h21={self.h21()} "
                    if d == 3 else "")
                + f"in a {d+1}-dimensional toric variety.")

    def frst(self):
        """Returns the FRST giving rise to the ambient toric variety."""
        return self._frst

    def polytope(self):
        """
        Returns the polytope whose triangulation gives rise to the ambient
        toric variety.
        """
        return self._frst.polytope()

    def ambient_dim(self):
        """Returns the complex dimension of the ambient toric variety."""
        return self.frst().dim()

    def dim(self):
        """Returns the complex dimension of the Calabi-Yau hypersurface."""
        return self.frst().dim() - 1

    def h11(self):
        """
        Returns the Hodge number h^{1,1} of the Calabi-Yau.  This is only
        implemented for 3-folds.
        """
        if self.dim() != 3:
            raise Exception("Hodge number can only be computed for 3-folds.")
        return self.polytope()._compute_h11()

    def h21(self):
        """
        Returns the Hodge number h^{2,1} of the Calabi-Yau.  This is only
        implemented for 3-folds
        """
        if self.dim() != 3:
            raise Exception("Hodge number can only be computed for 3-folds.")
        return self.polytope()._compute_h21()

    def chi(self):
        """
        Returns the Euler characteristic of the Calabi-Yau.  This is only
        implemented for 3-folds
        """
        if self.dim() != 3:
            raise Exception("Euler characteristic can only be computed for "
                            "3-folds.")
        return self.polytope()._compute_chi()

    def sr_ideal(self):
        """Returns the Stanley–Reisner ideal of the ambient toric variety."""
        return self.frst().sr_ideal()

    def glsm_charge_matrix(self, exclude_origin=False, n_retries = 100):
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
        return self.polytope().glsm_charge_matrix(
                                    exclude_origin=exclude_origin,
                                    use_all_points=self.frst()._all_poly_pts,
                                    n_retries=n_retries)

    def glsm_linear_relations(self, exclude_origin=False, n_retries=100):
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
        return self.polytope().glsm_linear_relations(
                                    exclude_origin=exclude_origin,
                                    use_all_points=self.frst()._all_poly_pts,
                                    n_retries=n_retries)

    def divisor_basis(self, integral_basis=False, exclude_origin=False,
                   n_retries=100):
        """
        Return the current basis of divisors of the ambient toric Variety.

        Args:
            exclude_origin (boolean, optional, default=False): Indicates
                whether to use the origin in the calculation.  This corresponds
                to the inclusion of the canonical divisor.
            use_all_points (boolean, optional, default=False): By default only
                boundary points not interior to facets are used. If this flag
                is set to true then points interior to facets are also used.
            integral_basis (boolean, optional, default=False): Indicates
                whether to try to find an integer basis for the columns of the
                GLSM charge matrix. (i.e. so that remaining columns can be
                written as an integer linear combination of the basis.)
            n_retries (int, optional, default=100): Flint sometimes fails to
                find the kernel of a matrix. This flag specifies the number of
                times the points will be suffled and the computation retried.

        Returns: A list of column indices that form a basis
        """
        if self._divisor_basis is None or integral_basis:
            self._divisor_basis = self.polytope().glsm_basis(
                                                integral_basis=integral_basis,
                                                exclude_origin=exclude_origin,
                                                n_retries=n_retries)
            self.clear_cache(only_in_basis=True)
        return np.array(self._divisor_basis)

    def set_divisor_basis(self, basis, exclude_origin=False,
                          exact_arithmetic=False):
        """
        Specifies a basis of divisors of the ambient toric variety.  This can
        be done with a vector specifying the indices of the prime toric
        divisors or with a matrix specifying a general basis as a linear
        combination of the h11+4 prime toric divisors, or the canonical
        divisor plus the h11+4 prime toric divisors.

        64-bit integer arithmetic is used when specifying indices of toric
        divisors, and 64-bit floating-point arithmetic is used for generic
        bases unless exact_arithmetic=True is specified.

        Args:
            basis (list): Vector or matrix specifying a basis. When a vector is
                used, the entries will be taken as the indices of points of the
                polytope, which correspond to divisors. When a matrix is used,
                the rows are taken as linear combinations of the divisors
                corresponding to the points of the polytope.
            exclude_origin (boolean, optional, default=False): Whether to
                take the indexing specified by the input vector as excluding
                the origin.
            exact_arithmetic (boolean, optional, default=False): Whether to use
                exact rational arithmetic instead of floats when using a
                generic basis.
        """
        b = np.array(basis)
        glsm_cm = self.polytope().glsm_charge_matrix(
                                    exclude_origin=False,
                                    use_all_points=self.frst()._all_poly_pts)
        glsm_rnk = np.linalg.matrix_rank(glsm_cm)
        # Check if the input is a vector
        if len(b.shape) == 1:
            if b.dtype != int:
                raise Exception("Input vector must contain integer entries.")
            if exclude_origin:
                b += 1
            # Check if it is a valid basis
            if min(b) < 0 or max(b) >= glsm_cm.shape[1]:
                raise Exception("Indices are not in appropriate range.")
            if glsm_rnk != np.linalg.matrix_rank(glsm_cm[:,b]):
                raise Exception("Input divisors do not form a basis.")
            self._divisor_basis = b
        # Else if input is a matrix
        elif len(b.shape) == 2:
            # We start by converting the matrix into a common data type
            t = type(b[0,0])
            if t in [np.int64, np.float64]:
                tmp_b = b
            elif t == fmpz:
                tmp_b = np.array(b, dtype=int)
            elif t == fmpq:
                tmp_b = np.empty(b.shape, dtype=float)
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        tmp_b[i,j] = int(b[i,j].p)/int(b[i,j].q)
            else:
                raise Exception("Unsupported data type.")
            if np.linalg.matrix_rank(tmp_b) != glsm_rnk:
                raise Exception("Input matrix has incorrect rank.")
            if b.shape == (glsm_rnk, glsm_cm.shape[1]):
                new_b = b
                tmp_new_b = tmp_b
            elif b.shape == (glsm_rnk, glsm_cm.shape[1]-1):
                new_b = np.empty(glsm_cm.shape, dtype=t)
                new_b[:,1:] = b
                new_b[:,0] = t(0)
                tmp_new_b = np.zeros(glsm_cm.shape, dtype=tmp_b.dtype)
                tmp_new_b[:,1:] = tmp_b
            else:
                raise Exception("Input matrix has incorrect shape.")
            new_glsm_cm = tmp_new_b.dot(glsm_cm.T).T
            if np.linalg.matrix_rank(new_glsm_cm) != glsm_rnk:
                raise Exception("Input divisors do not form a basis.")
            if exact_arithmetic and t == np.int64:
                new_b = np_int_to_fmpz(new_b)
            elif exact_arithmetic and t == np.float64:
                new_b = np_float_to_fmpq(new_b)
            elif t == np.int64:
                new_b = np.array(new_b, dtype=float)
            self._divisor_basis = new_b
        else:
            raise Exception("Input must be either a vector or a matrix.")
        # Clear the cache of all in-basis computations
        self.clear_cache(only_in_basis=True)

    def set_dual_divisor_basis(self, basis, exclude_origin=False,
                          exact_arithmetic=False):
        """
        Specifies a dual basis of divisors (i.e. a basis of curves) of the
        ambient toric variety. This can be done with a vector specifying the
        indices of the prime toric divisors or with a matrix specifying a
        general basis as a linear combination of the h11+4 prime toric
        divisors, or the canonical divisor plus the h11+4 prime toric divisors.

        64-bit integer arithmetic is used when specifying indices of toric
        divisors, and 64-bit floating-point arithmetic is used for generic
        bases unless exact_arithmetic=True is specified.

        Args:
            basis (list): Vector or matrix specifying a basis. When a vector is
                used, the entries will be taken as the indices of points of the
                polytope, which correspond to divisors. When a matrix is used,
                the rows are taken as linear combinations of the divisors
                corresponding to the points of the polytope.
            exclude_origin (boolean, optional, default=False): Whether to
                take the indexing specified by the input vector as excluding
                the origin.
            exact_arithmetic (boolean, optional, default=False): Whether to use
                exact rational arithmetic instead of floats when using a
                generic basis.
        """
        b = np.array(basis)
        glsm_cm = self.polytope().glsm_charge_matrix(
                                    exclude_origin=False,
                                    use_all_points=self.frst()._all_poly_pts)
        glsm_rnk = np.linalg.matrix_rank(glsm_cm)
        # Check if the input is a vector
        if len(b.shape) == 1:
            if b.dtype != int:
                raise Exception("Input vector must contain integer entries.")
            if exclude_origin:
                b += 1
            # Check if it is a valid basis
            if min(b) < 0 or max(b) >= glsm_cm.shape[1]:
                raise Exception("Indices are not in appropriate range.")
            new_b = np.zeros(glsm_cm.shape, dtype=int)
            for i,ii in enumerate(b):
                new_b[i,ii] = 1
            if glsm_rnk != np.linalg.matrix_rank(new_b):
                raise Exception("Input does not form a basis.")
        # Else if input is a matrix
        elif len(b.shape) == 2:
            t = type(b[0,0])
            if t in [np.int64, np.float64]:
                tmp_b = b
            elif t == fmpz:
                exact_arithmetic = True
                tmp_b = np.array(b, dtype=int)
            elif t == fmpq:
                exact_arithmetic = True
                tmp_b = np.empty(b.shape, dtype=float)
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        tmp_b[i,j] = int(b[i,j].p)/int(b[i,j].q)
            else:
                raise Exception("Unsupported data type.")
            if np.linalg.matrix_rank(tmp_b) != glsm_rnk:
                raise Exception("Input matrix has incorrect rank.")
            if b.shape == (glsm_rnk, glsm_cm.shape[1]):
                new_b = b
            elif b.shape == (glsm_rnk, glsm_cm.shape[1]-1):
                new_b = np.empty(glsm_cm.shape, dtype=t)
                new_b[:,1:] = b
                new_b[:,0] = t(0)
            else:
                raise Exception("Input matrix has incorrect shape.")
            # Now we convert to exact rationals or integers if necessary
            if exact_arithmetic and t == np.int64:
                new_b = np_int_to_fmpz(new_b)
            elif exact_arithmetic and t == np.float64:
                new_b = np_float_to_fmpq(new_b)
        else:
            raise Exception("Input must be either a vector or a matrix.")
        # Now we compute the pseudo-inverse that defines a divisors basis.
        if exact_arithmetic:
            # Flint doesn't have a pseudo-inverse computation so we do this
            # by first extending the matrix and finding the inverse, and
            # then we truncate the result.
            ctr = 0
            while ctr <= 10:
                ctr += 1
                b_ext = np.concatenate(
                    (new_b, np.random.randint(-1, 1,
                     size=(glsm_cm.shape[1]-glsm_rnk,glsm_cm.shape[1]))),
                    axis=0)
                if np.linalg.matrix_rank(np_fmpq_to_float(np.array(fmpq_mat(
                            b_ext.tolist()).table()))) == glsm_cm.shape[1]:
                    break
            if ctr > 10:
                raise Exception("There was a problem finding the inverse "
                                "matrix")
            b_ext_inv = np.array(fmpz_mat(b_ext.tolist()).inv().table())
            b_inv = b_ext_inv[:,:glsm_rnk].T
        else:
            b_inv = np.linalg.pinv(new_b).T
        self.set_divisor_basis(b_inv, exact_arithmetic=exact_arithmetic)

    def mori_cone(self, in_basis=False, exclude_origin=False):
        """Returns the Mori cone of the ambient toric variety."""
        if self._mori_cone[0] is None:
            if self._ambient_intersection_numbers[0] is not None:
                rays = self._compute_mori_rays_from_ambient_intersections()
                self._mori_cone[0] = Cone(rays)
            else:
                self._mori_cone[0] = self.frst().cpl_cone().dual()
        # 0: All divs, 1: No origin, 2: In basis
        args_id = (exclude_origin*1 if not in_basis else 0) + in_basis*2
        if self._mori_cone[args_id] is not None:
            return self._mori_cone[args_id]
        rays = self.frst().cpl_cone().hyperplanes()
        if not exclude_origin and not in_basis:
            new_rays = rays
        elif exclude_origin and not in_basis:
            new_rays = rays[:,1:]
        else:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                new_rays = rays.dot(basis.T)
            else:
                new_rays = rays[:,basis]
        c = Cone(new_rays, check=len(basis.shape)==2)
        self._mori_cone[args_id] = c
        return self._mori_cone[args_id]

    def _compute_mori_rays_from_ambient_intersections(self):
        """
        Computes the Mori cone rays of the ambient variety using intersection
        numbers.  This function should generally not be called by the user.
        Instead, this is called by the mori_cone() function when it detects
        that ambient intersection numbers were already computed so as to save
        some time.

        This function currently only supports CY 3-folds.
        """
        ambient_intnums = self.ambient_intersection_numbers(in_basis=False)
        int_nums = [[ii[0],ii[1],ii[2],ii[3]]+[ii[4]]
                    for ii in ambient_intnums]
        num_divs = int(max([ii[3] for ii in ambient_intnums])) + 1
        curve_dict = {}
        curve_ctr = 0
        curve_sparse_list = []
        for ii in int_nums:
            if ii[0] == 0:
                continue
            if ii[0] == ii[1] == ii[2] == ii[3]:
                continue
            elif ii[0] == ii[1] == ii[2]:
                continue
            elif ii[1] == ii[2] == ii[3]:
                continue
            elif ii[0] == ii[1] and ii[2] == ii[3]:
                continue
            elif ii[0] == ii[1]:
                if (ii[0],ii[2],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[2],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[1],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[1],ii[2],ii[3])],
                                              ii[0],ii[-1]])
            elif ii[1] == ii[2]:
                if (ii[0],ii[1],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[2],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[3])],
                                              ii[2],ii[-1]])
            elif ii[2] == ii[3]:
                if (ii[0],ii[1],ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[3],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[2])],
                                              ii[3],ii[-1]])
            else:
                if (ii[0],ii[1],ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[3],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[2])],
                                              ii[3],ii[-1]])
                if (ii[0],ii[1],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[2],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[3])],
                                              ii[2],ii[-1]])
                if (ii[0],ii[2],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[2],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[1],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[2],ii[3])],
                                              ii[1],ii[-1]])
                if (ii[1],ii[2],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[1],ii[2],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[0],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[1],ii[2],ii[3])],
                                              ii[0],ii[-1]])
        row_list = [[] for i in range(curve_ctr)]
        # Remove zeros
        for ii in curve_sparse_list:
            if ii[2]!=0:
                row_list[ii[0]].append([ii[1],ii[2]])
        # Normalize
        for row in row_list:
            g = abs(gcd_list([ii[1] for ii in row]))
            for ii in row:
                ii[1] = int(round(ii[1]/g))
        row_list = set(tuple(tuple(tuple(ii) for ii in sorted(row))
                            for row in row_list))
        mori_rays = np.zeros((len(row_list),num_divs), dtype=int)
        for i,row in enumerate(row_list):
            for ii in row:
                mori_rays[i,int(round(ii[0]))] = round(ii[1])
        # Compute column corresponding to the origin
        mori_rays[:,0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def kahler_cone(self):
        """Returns the Kahler cone of the ambient toric variety."""
        return self.mori_cone(in_basis=True).dual()

    def ambient_intersection_numbers(self, in_basis=False,
                                     zero_as_anticanonical=False, backend="all",
                                     check=True, backend_error_tol=1e-4,
                                     round_to_zero_treshold=1e-8,
                                     round_to_integer_error_tol=1e-4,
                                     verbose=0):
        """
        Returns the intersection numbers of the ambient toric variety.

        Args:
            in_basis (boolean, optional, default=False): Whether to only return
                the intersection numbers of a basis of divisors.
            zero_as_anticanonical (boolean, optional, default=False): Whether
                to treat the zeroth index as corresponding to the anticanonical
                divisor instead of the canonical divisor.
            backend (string, optional, default="all"): The sparse linear solver
                to use.  Options are "all", "sksparse" and "scipy".  When set
                to "all" every backend is tried in order until one succeeds.
            check (boolean, optional, default=True): Whether to explicitly
                check the solution to the linear system.
            backend_error_tol (float, optional, default=1e-4): Error tolerance
                for the solution of the linear system.
            round_to_zero_treshold (float, optional, default=5e-2):
                Intersection numbers with magnitude smaller than this treshold
                are rounded to zero.
            round_to_integer_error_tol (float, optional, default=5e-2): All
                intersection numbers must be integers up to errors less than
                this value. Otherwise, an Exception is raised.
            verbose (int, optional, default=0): Verbosity level:
                - verbose = 0: Do not print anything.
                - verbose = 1: Print linear backend warnings.

        Returns:
            np.array: A matrix containing nonzero intersection numbers, in the
                format: [[A,B,C,D,Kappa_ABCD], ...], where A,B,C,D are indices
                of divisors and Kappa_ABCD is the intersection number.

        Notes:
            The intersection numbers are computed as floating point numbers,
            not rationals.  This does not make a difference when the
            intersection numbers are integers.
            This function currently only supports 4D toric varieties.
        """
        if self.dim() != 3:
            raise Exception("Only 4D varieties are currently supported.")
        # 0: With canon, 1: With anticanon, 2: In basis
        args_id = (1*zero_as_anticanonical if not in_basis else 0)+2*in_basis
        if self._ambient_intersection_numbers[args_id] is not None:
            return np.array(self._ambient_intersection_numbers[args_id])
        if self._ambient_intersection_numbers[0] is None:
            backends = ["all", "sksparse", "scipy"]
            if backend not in backends:
                raise Exception("Invalid linear system backend. "
                                f"The options are: {backends}.")
            # Prepare Data
            # Origin is at index 0
            rays = [tuple(r) for r in self.frst().points()[1:]]
            pts_ext = np.empty((self.frst().points().shape[0],
                                self.frst().points().shape[1]+1),
                                    dtype=int)
            pts_ext[:,:-1] = self.frst().points()
            pts_ext[:,-1] = 1
            linear_relations = self.glsm_linear_relations(exclude_origin=True)
            # First compute the distict intersection numbers
            distintnum_array = sorted([
                simp[1:].tolist()
                + [1/abs(np.linalg.det([pts_ext[p] for p in simp]))]
                    for simp in self.frst().simplices()])
            frst = [[c for c in s if c != 0] for s in self.frst().simplices()]
            distintnum_array = [simp[:-1] + [simp[-1]]
                                    for simp in distintnum_array]
            simp_2 = set([j for i in [list(combinations(f,2)) for f in frst]
                          for j in i])
            simp_3 = set([j for i in [list(combinations(f,3)) for f in frst]
                          for j in i])
            # We construct and solve the linear system M*x + C = 0, where M is
            # a rectangular mxn matrix and C is a vector.
            ###################################################################
            ### Define dictionaries, to be used to construct the linear system
            ###################################################################
            ## Dictionary of variables
            # Most intersection numbers are trivially zero, find the possibly
            # nonzero intersection numbers.
            variable_array_1 = [tuple(j) for i in [[[s[0],s[0],s[1],s[2]],
                                                    [s[0],s[1],s[1],s[2]],
                                                    [s[0],s[1],s[2],s[2]]]
                                                    for s in simp_3]
                                                        for j in i]
            variable_array_2 = [tuple(j) for i in [[[s[0],s[0],s[1],s[1]],
                                                    [s[0],s[0],s[0],s[1]],
                                                    [s[0],s[1],s[1],s[1]]]
                                                    for s in simp_2]
                                                        for j in i]
            variable_array_3 = [(i,i,i,i) for i in range(1, len(rays)+1)]
            variable_array = sorted(variable_array_1 + variable_array_2
                                    + variable_array_3)
            variable_dict = {vv:v for v,vv in enumerate(variable_array)}
            ## Dictionary to construct C
            # C is constructed by adding/subtracting distinct intersection
            # numbers.
            c_dict = {s:[] for s in simp_3}
            for d in distintnum_array:
                c_dict[(d[0],d[1],d[2])] += [[d[3],d[4]]]
                c_dict[(d[0],d[1],d[3])] += [[d[2],d[4]]]
                c_dict[(d[0],d[2],d[3])] += [[d[1],d[4]]]
                c_dict[(d[1],d[2],d[3])] += [[d[0],d[4]]]
            ## Dictionary to construct M
            eqn_array_1 = [tuple(s) for s in simp_3]
            eqn_array_2 = [tuple(j) for i in [[[s[0],s[0],s[1]],
                                               [s[0],s[1],s[1]]]
                                               for s in simp_2] for j in i]
            eqn_array_3 = [(i,i,i) for i in range(1, len(rays)+1)]
            eqn_array = sorted(eqn_array_1 + eqn_array_2 + eqn_array_3)
            eqn_dict = {eq:[] for eq in eqn_array}
            for v in variable_array:
                if v[0]==v[3]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                elif v[0]==v[2]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[0],v[1],v[3])] += [[v[2],variable_dict[v]]]
                elif v[0]==v[1] and v[2]==v[3]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[0],v[2],v[3])] += [[v[1],variable_dict[v]]]
                elif v[0]==v[1]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[0],v[1],v[3])] += [[v[2],variable_dict[v]]]
                    eqn_dict[(v[0],v[2],v[3])] += [[v[1],variable_dict[v]]]
                elif v[1]==v[3]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[1],v[2],v[3])] += [[v[0],variable_dict[v]]]
                elif v[1]==v[2]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[0],v[1],v[3])] += [[v[2],variable_dict[v]]]
                    eqn_dict[(v[1],v[2],v[3])] += [[v[0],variable_dict[v]]]
                elif v[2]==v[3]:
                    eqn_dict[(v[0],v[1],v[2])] += [[v[3],variable_dict[v]]]
                    eqn_dict[(v[0],v[2],v[3])] += [[v[1],variable_dict[v]]]
                    eqn_dict[(v[1],v[2],v[3])] += [[v[0],variable_dict[v]]]
                else:
                    raise Exception("Failed to construct linear system.")
            # Construct Linear System
            num_cols = len(variable_array)
            num_rows = len(linear_relations)*len(eqn_array)
            C = np.array([0.0]*num_rows)
            M_row = []
            M_col = []
            M_val = []
            row_ctr = 0
            for eqn in eqn_array:
                for lin in linear_relations:
                    if eqn[0]!=eqn[1] and eqn[1]!=eqn[2]:
                        c_temp = c_dict[eqn]
                        C[row_ctr] = sum([lin[cc[0]-1]*cc[1] for cc in c_temp])
                    eqn_temp = eqn_dict[eqn]
                    for e in eqn_temp:
                        M_row.append(row_ctr)
                        M_col.append(e[1])
                        M_val.append(lin[e[0]-1])
                    row_ctr+=1
            Mat = csr_matrix((M_val,(M_row,M_col)), dtype=np.float64)
            # The system to be solved is Mat*x + C = 0. This is an
            # overdetermined but consistent linear system.
            # There is a unique solution to this system. We solve it by
            # defining MM = Mat.transpose()*Mat and CC = - Mat.transpose()*C,
            # and solve
            # MM*x = CC
            # Since MM is a positive definite full rank matrix, this system can
            # be solved using via a Cholesky decomposition.
            solution = solve_linear_system(Mat, C, backend=backend, check=check,
                                           backend_error_tol=backend_error_tol,
                                           verbose=verbose)
            #return solution
            if solution is None:
                raise Exception("Linear system solution failed.")
            ambient_intnum = distintnum_array + [list(ii) + [solution[i]]
                                         for i,ii in enumerate(variable_array)]
            ambient_intnum = [ii for ii in ambient_intnum
                              if abs(ii[-1]) > round_to_zero_treshold]
            # Add intersections with canonical divisor
            # First we only compute intersection numbers with a single index 0
            # This is because precision errors add up significantly for
            # intersection numbers with self-intersections of the canonical
            # divisor
            canon_intnum = defaultdict(lambda: 0)
            for ii in ambient_intnum:
                s012 = tuple(sorted([ii[0],ii[1],ii[2]]))
                s013 = tuple(sorted([ii[0],ii[1],ii[3]]))
                s023 = tuple(sorted([ii[0],ii[2],ii[3]]))
                s123 = tuple(sorted([ii[1],ii[2],ii[3]]))
                canon_intnum[(0,)+s012] -= ii[-1]
                if s013 != s012:
                    canon_intnum[(0,)+s013] -= ii[-1]
                if s023 not in (s012, s013):
                    canon_intnum[(0,)+s023] -= ii[-1]
                if s123 not in (s012, s013, s023):
                    canon_intnum[(0,)+s123] -= ii[-1]
            # Now we round all intersection numbers of the form K_0ijk with
            # i,j,k != 0 to integers
            for ii in list(canon_intnum.keys()):
                val = canon_intnum[ii]
                round_val = round(val)
                if abs(val-round_val) > round_to_integer_error_tol:
                    raise Exception("Non-integer intersection numbers "
                                    "detected. Is the Calabi Yau hypersurface "
                                    "singular?")
                if round_val != 0:
                    canon_intnum[ii] = round_val
                else:
                    canon_intnum.pop(ii)
            # Now we compute K_00ij, K_000i, K_000
            for ii in list(canon_intnum.keys()):
                val = canon_intnum[ii]
                # 4x
                ndiff = len(set(ii[1:]))
                if ndiff == 1:
                    fact = 1
                elif ndiff == 2:
                    fact = 3
                else:
                    fact = 6
                canon_intnum[(0,0,0,0)] -= val*fact
                # 3x
                canon_intnum[(0,0,0,ii[1])] += val*(1+(ii[2]!=ii[3]))
                if ii[2] != ii[1]:
                    canon_intnum[(0,0,0,ii[2])] += val*(1+(ii[1]!=ii[3]))
                if ii[3] not in (ii[1], ii[2]):
                    canon_intnum[(0,0,0,ii[3])] += val*(1+(ii[1]!=ii[2]))
                # 2x
                canon_intnum[(0,0,ii[1],ii[2])] -= val
                if ii[3] != ii[2]:
                    canon_intnum[(0,0,ii[1],ii[3])] -= val
                if (ii[2],ii[3]) not in ((ii[1],ii[2]),(ii[1],ii[3])):
                    canon_intnum[(0,0,ii[2],ii[3])] -= val
            ambient_intnum.extend([list(ii)+[canon_intnum[ii]]
                            for ii in canon_intnum
                            if abs(canon_intnum[ii]) > round_to_zero_treshold])
            ambient_intnum = sorted([[ii[0],ii[1],ii[2],ii[3],
                                              ii[4]] for ii in ambient_intnum])
            self._ambient_intersection_numbers[0] = np.array(ambient_intnum)
        # Now ambient intersection numbers have been computed
        if zero_as_anticanonical and not in_basis:
            self._ambient_intersection_numbers[args_id] = np.array([
                            ii[0:4].tolist()
                            + [ii[4]*(-1 if sum(ii[0:4] == 0)%2 == 1 else 1)]
                            for ii in ambient_intnum])
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                self._ambient_intersection_numbers[2] = (
                    symmetric_sparse_to_dense_in_basis(
                        self._ambient_intersection_numbers[0], basis))
            else:
                self._ambient_intersection_numbers[1] = filter_tensor_indices(
                            self._ambient_intersection_numbers[0], basis)
        return np.array(self._ambient_intersection_numbers[args_id])

    def intersection_numbers(self, in_basis=False, zero_as_anticanonical=False,
                             backend="all", check=True,
                             backend_error_tol=1e-2,
                             round_to_zero_treshold=1e-8,
                             round_to_integer_error_tol=1e-2, verbose=0):
        """
        Calculates the intersection numbers of the (smooth) CY hypersurface.
        The intersection numbers are computed as floating point numbers, not
        rationals.  Only 3-folds are currently supported.

        Warning: If the CY hypersurface is not smooth, the results cannot be
        trusted.

        Args:
            in_basis (boolean, optional, default=True): Whether to only return
                the intersection numbers of a basis of divisors.
            zero_as_anticanonical (boolean, optional, default=False): Whether
                to treat the zeroth index as corresponding to the anticanonical
                divisor instead of the canonical divisor.
            backend (string, optional, default="all"): The sparse linear solver
                to use.  Options are "all", "sksparse" and "scipy".  When set
                to "all" every backend is tried in order until one succeeds.
            check (boolean, optional, default=True): Whether to explicitly
                check the solution to the linear system.
            backend_error_tol (float, optional, default=1e-2): Error tolerance
                for the solution of the linear system.
            round_to_zero_treshold (float, optional, default=1e-8):
                Intersection numbers with magnitude smaller than this treshold
                are rounded to zero.
            round_to_integer_error_tol (float, optional, default=1e-2): All
                intersection numbers must be integers up to errors less than
                this value. Otherwise, an Exception is raised.
            verbose (int, optional, default=0): Verbosity level:
                - verbose = 0: Do not print anything.
                - verbose = 1: Print linear backend warnings.

        Returns:
            np.array: A matrix containing nonzero intersection numbers, in the
                format: [[A,B,C,Kappa_ABC], ...], where A,B,C are indices of
                divisors and Kappa_ABCD is the intersection number.
        """
        if self.dim() != 3:
            raise Exception("Only CY 3-folds are currently supported.")
        # 0: With canon, 1: With anticanon, 2: In basis
        args_id = (1*zero_as_anticanonical if not in_basis else 0)+2*in_basis
        if self._intersection_numbers[args_id] is not None:
            return np.array(self._intersection_numbers[args_id])
        if self._intersection_numbers[0] is None:
            ambient_intnums = self.ambient_intersection_numbers(in_basis=False,
                        backend=backend, check=check,
                        backend_error_tol=backend_error_tol,
                        round_to_zero_treshold=round_to_zero_treshold,
                        round_to_integer_error_tol=round_to_integer_error_tol,
                        verbose=verbose)
            intnum_tmp = np.array([ii[1:4].tolist()+[-ii[4]]
                                for ii in ambient_intnums if ii[0]==0])
            intnum_fin = sorted([[ii[0],ii[1],ii[2],int(round(ii[3]))]
                                 for ii in intnum_tmp if int(round(ii[3]))!=0])
            self._intersection_numbers[0] = np.array(intnum_fin, dtype=int)
        # Now intersection numbers have been computed
        if zero_as_anticanonical and not in_basis:
            self._intersection_numbers[args_id] = np.array([
                            ii[0:3].tolist()
                            + [ii[3]*(-1 if sum(ii[0:3] == 0)%2 == 1 else 1)]
                            for ii in self._intersection_numbers[0]])
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                self._intersection_numbers[args_id] = (
                    symmetric_sparse_to_dense_in_basis(
                        self._intersection_numbers[0], basis))
            else:
                self._intersection_numbers[args_id] = filter_tensor_indices(
                            self._intersection_numbers[0], basis)
        return np.array(self._intersection_numbers[args_id])

    def second_chern_class(self, in_basis=True):
        """
        Computes the second Chern class of the CY hypersurface.
        Returns the integral of the second Chern class over
        the prime effective divisors.

        Args:
            in_basis (boolean, optional, default=True): Whether to only return
                the integrals over a basis of divisors.

        Returns:
            np.array: A vector containing the integrals.
        """
        int_nums = [[ii[0]-1, ii[1]-1, ii[2]-1, ii[3]] for ii in
                     self.intersection_numbers(in_basis=False) if min(ii[:3])!=0]
        c2 = np.zeros(self.h11()+4, dtype=int)
        for ii in int_nums:
            if ii[0]==ii[1]==ii[2]:
                continue
            elif ii[0]==ii[1]:
                c2[ii[0]]+=ii[3]
            elif ii[0]==ii[2]:
                c2[ii[0]]+=ii[3]
            elif ii[1]==ii[2]:
                c2[ii[1]]+=ii[3]
            else:
                c2[ii[0]]+=ii[3]
                c2[ii[1]]+=ii[3]
                c2[ii[2]]+=ii[3]
        if in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                return c2.dot(basis.T)
            return c2[basis]
        return c2

    def compute_cy_volume(self, tloc):
        """
        Takes Kahler parameters as input and calculates the volume of the CY.
        """
        intnums = self.intersection_numbers(in_basis=True)
        xvol = 0
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[0,0,0], fmpz):
                intnums = np_fmpz_to_int(intnums)
            elif isinstance(intnums[0,0,0], fmpq):
                intnums = np_fmpq_to_float(intnums)
            tmp1 = np.tensordot(intnums, tloc, axes=[[2],[0]])
            tmp2 = np.tensordot(tmp1, tloc, axes=[[1],[0]])
            xvol = np.tensordot(tmp2, tloc, axes=[[0],[0]])/6
        else:
            for ii in intnums:
                mult = Counter(ii[:3]).most_common(1)[0][1]
                xvol += (ii[3]*tloc[ii[0]]*tloc[ii[1]]*tloc[ii[2]]
                         /factorial(mult))
        return xvol

    def compute_divisor_volumes(self, tloc):
        """
        Takes Kahler parameters as input and calculates volumes of the basis
        4-cycles.
        """
        intnums = self.intersection_numbers(in_basis=True)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[0,0,0], fmpz):
                intnums = np_fmpz_to_int(intnums)
            elif isinstance(intnums[0,0,0], fmpq):
                intnums = np_fmpq_to_float(intnums)
            tmp1 = np.tensordot(intnums, tloc, axes=[[2],[0]])
            tau = np.tensordot(tmp1, tloc, axes=[[1],[0]])/2
        else:
            tau = np.array([0.]*len(tloc))
            for ii in intnums:
                ii_list = Counter(ii[:3]).most_common(3)
                if len(ii_list)==1:
                    tau[ii_list[0][0]] += ii[3]*tloc[ii_list[0][0]]**2/2
                elif len(ii_list)==2:
                    tau[ii_list[0][0]] += (ii[3]*tloc[ii_list[0][0]]
                                           *tloc[ii_list[1][0]])
                    tau[ii_list[1][0]] += ii[3]*tloc[ii_list[0][0]]**2/2
                elif len(ii_list)==3:
                    tau[ii_list[0][0]] += (ii[3]*tloc[ii_list[1][0]]
                                           *tloc[ii_list[2][0]])
                    tau[ii_list[1][0]] += (ii[3]*tloc[ii_list[0][0]]
                                           *tloc[ii_list[2][0]])
                    tau[ii_list[2][0]] += (ii[3]*tloc[ii_list[0][0]]
                                           *tloc[ii_list[1][0]])
                else:
                    raise Exception("Inconsistent intersection numbers.")
        return np.array(tau)

    def compute_AA(self, tloc):
        """
        Takes Kahler parameters as input and calculates the matrix
        kappa^ijk*t_k.
        """
        intnums = self.intersection_numbers(in_basis=True)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[0,0,0], fmpz):
                intnums = np_fmpz_to_int(intnums)
            elif isinstance(intnums[0,0,0], fmpq):
                intnums = np_fmpq_to_float(intnums)
            AA = np.tensordot(intnums, tloc, axes=[[2],[0]])
            return AA
        AA = [[0 for c in range(len(tloc))] for r in range(len(tloc))]
        for ii in intnums:
            ii_list = Counter(ii[:3]).most_common(3)
            if len(ii_list)==1:
                AA[ii_list[0][0]][ii_list[0][0]] += ii[3]*tloc[ii_list[0][0]]
            elif len(ii_list)==2:
                AA[ii_list[0][0]][ii_list[0][0]] += ii[3]*tloc[ii_list[1][0]]
                AA[ii_list[0][0]][ii_list[1][0]] += ii[3]*tloc[ii_list[0][0]]
                AA[ii_list[1][0]][ii_list[0][0]] += ii[3]*tloc[ii_list[0][0]]
            elif len(ii_list)==3:
                AA[ii_list[0][0]][ii_list[1][0]] += ii[3]*tloc[ii_list[2][0]]
                AA[ii_list[1][0]][ii_list[0][0]] += ii[3]*tloc[ii_list[2][0]]
                AA[ii_list[0][0]][ii_list[2][0]] += ii[3]*tloc[ii_list[1][0]]
                AA[ii_list[2][0]][ii_list[0][0]] += ii[3]*tloc[ii_list[1][0]]
                AA[ii_list[1][0]][ii_list[2][0]] += ii[3]*tloc[ii_list[0][0]]
                AA[ii_list[2][0]][ii_list[1][0]] += ii[3]*tloc[ii_list[0][0]]
            else:
                raise Exception("Error: Inconsistent intersection numbers.")
        return np.array(AA)

    def compute_Kinv(self, tloc):
        """
        Takes Kahler parameters as input and calculates the inverse Kahler
        metric.
        """
        xvol = self.compute_cy_volume(tloc)
        Tau = self.compute_divisor_volumes(tloc)
        AA = self.compute_AA(tloc)
        Kinv = 4*(np.outer(Tau,Tau) - AA*xvol)
        return Kinv
