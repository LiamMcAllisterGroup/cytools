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
This module contains tools designed for Calabi-Yau hypersurface computations.
"""

#Standard imports
from collections import Counter, defaultdict
from itertools import combinations
from math import factorial
# Third party imports
from flint import fmpz_mat, fmpq_mat, fmpz, fmpq
from scipy.sparse import csr_matrix
import numpy as np
# CYTools imports
from cytools.utils import (gcd_list, solve_linear_system, array_fmpz_to_int,
                           array_int_to_fmpz, array_float_to_fmpq,
                           array_fmpq_to_float, filter_tensor_indices,
                           symmetric_sparse_to_dense_in_basis, float_to_fmpq,
                           fmpq_to_float)
from cytools.cone import Cone
from cytools import config



class CalabiYau:
    """
    This class handles various computations relating to the Calabi-Yau manifold
    itself. It can be used to compute intersection numbers, the Kähler cone of
    the ambient variety, among other things.

    :::important
    Generally, objects of this class should not be constructed directly by the
    end user. Instead, they should be created by the
    [get_cy](./triangulation#get_cy) function of the
    [Triangulation](./triangulation) class.
    :::

    :::tip experimental feature
    This package is focused on computations on Calabi-Yau 3-folds, but there is
    experimental support for Calabi-Yaus of other dimensions. See
    [Experimental Features](./experimental) for more details.
    :::

    ## Constructor

    ### ```cytools.calabiyau.CalabiYau```

    **Description:**
    Constructs a ```CalabiYau``` object. This is handled by the hidden
    [```__init__```](#__init__) function.

    **Arguments:**
    - ```frst``` (Triangulation): A fine, regular, star triangularion of a
    reflexive polytope.

    **Example:**
    We construct a Calabi-Yau from an fine, regular, star triangulation of a
    polytope. Since this class is not intended to by initialized by the end
    user, we create it via the
    [```get_cy```](./triangulation#get_cy) function of the
    [Triangulation](./triangulation) class. In this example we obtain the
    quintic hypersurface in $\mathbb{P}^4$.
    ```python {4}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    t = p.triangulate()
    cy = t.get_cy()
    # Prints: A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 in a 4-dimensional toric variety.
    ```
    """

    def __init__(self, frst):
        """
        **Description:**
        Initializes a ```CalabiYau``` object.

        **Arguments:**
        - ```frst``` (Triangulation): A fine, regular, star triangularion of a
        reflexive polytope.

        **Returns:**
        Nothing.
        """
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        if not frst._allow_cy or not frst.is_star() or not frst.is_fine():
            raise Exception("The input triangulation is not suitable for "
                            "constructing a Calabi-Yau.")
        if not frst._poly.is_favorable(lattice="N"):
            raise Exception("Only favorable CYs are currently supported.")
        if frst._all_poly_pts and not config.enable_experimental_features:
            raise Exception("Triangulations that include points interior to "
                            "facets are currently experimental.")
        self._frst = frst
        # Initialize remaining hidden attributes
        self._hash = None
        self._divisor_basis = None
        self._mori_cone = [None]*3
        self._ambient_intersection_numbers = [None]*7
        self._intersection_numbers = [None]*7
        self._second_chern_class = None
        self._is_smooth = None
        self._ambient_eff_gens = None
        self._ambient_eff_cone = None

    def clear_cache(self, recursive=False, only_in_basis=False):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        - ```recursive``` (boolean, optional, default=True): Whether to also
          clear the cache of the defining triangulation and polytope.
          This is ignored when only_in_basis=True.
        - ```only_in_basis``` (boolean, optional, default=False): Only clears
          the cache of computations that depend on a choice of basis.

        **Returns:**
        Nothing.
        """
        self._mori_cone[2] = None
        self._ambient_intersection_numbers[2] = None
        self._ambient_intersection_numbers[6] = None
        self._intersection_numbers[2] = None
        self._intersection_numbers[6] = None
        self._ambient_eff_gens = None
        self._ambient_eff_cone = None
        if not only_in_basis:
            self._hash = None
            self._divisor_basis = None
            self._mori_cone = [None]*3
            self._ambient_intersection_numbers = [None]*7
            self._intersection_numbers = [None]*7
            self._second_chern_class = None
            self._is_smooth = None
            if recursive:
                self._frst.clear_cache(recursive=True)

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the Calabi-Yau hypersurface.

        **Arguments:**
        None.

        **Returns:**
        (string) A string describing the Calabi-Yau hypersurface.
        """
        d = self.dim()
        if d == 2:
            out_str = (f"A K3 hypersurface with h11={self.h11()} in a "
                       + "3-dimensional toric variety.")
        elif d == 3:
            out_str = ("A Calabi-Yau 3-fold hypersurface with "
                       + f"h11={self.h11()} and h21={self.h21()} in a "
                       + "4-dimensional toric variety.")
        elif d == 4:
            out_str = ("A Calabi-Yau 4-fold hypersurface with "
                       + f"h11={self.h11()}, h12={self.h12()}, "
                       + f"h13={self.h13()}, and h22={self.h22()} in a "
                       + "5-dimensional toric variety.")
        else:
            out_str = (f"A Calabi-Yau {d}-fold hypersurface in a "
                       + f"{d+1}-dimensional toric variety.")
        return out_str

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of Calabi-Yaus with ==.

        :::important
        This function provides only a very trivial comparison using the
        [is_trivially_equivalent](#is_trivially_equivalent) function. It is not
        recommended to compare CYs with ==, and it a warning will be printed
        every time it evaluates to False. This is only implemented so that sets
        and dictionaries of CYs can be created.
        The [is_trivially_equivalent](#is_trivially_equivalent) function should
        be used to avoid confusion.
        :::

        **Arguments:**
        - ```other``` (CalabiYau): The other CY that is being compared.

        **Returns:**
        (boolean) The truth value of the CYs being equal.
        """
        if not isinstance(other, CalabiYau):
            return NotImplemented
        is_triv_equiv =  self.is_trivially_equivalent(other)
        if is_triv_equiv:
            return True
        print("Warning: Comparison of CYs should not be done with ==. "
              "Please use the is_trivially_equivalent function.")
        return False

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of Calabi-Yaus with !=.

        :::important
        This function provides only a very trivial comparison using the
        [is_trivially_equivalent](#is_trivially_equivalent) function. It is not
        recommended to compare CYs with !=, and it a warning will be printed
        every time it evaluates to True. This is only implemented so that sets
        and dictionaries of CYs can be created.
        The [is_trivially_equivalent](#is_trivially_equivalent) function should
        be used to avoid confusion.
        :::

        **Arguments:**
        - ```other``` (Polytope): The other CY that is being compared.

        **Returns:**
        (boolean) The truth value of the CYs being different.
        """
        if not isinstance(other, CalabiYau):
            return NotImplemented
        return not (self == other)

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from Calabi-Yaus.

        **Arguments:**
        None.

        **Returns:**
        (integer) The hash value of the CY.
        """
        if self._hash is not None:
            return self._hash
        dim = self.frst().dim()
        codim2_faces = [self.frst().points_to_indices(f.points())
                            for f in self.polytope().faces(dim-2)]
        restr_triang = set()
        for f in codim2_faces:
            face_pts = set(f)
            for s in self.frst().simplices():
                inters = face_pts & set(s)
                if len(inters) == dim-1:
                    ss = tuple(sorted(inters))
                    restr_triang.add(ss)
        self._hash = hash((hash(self.frst().polytope()),) +
                          tuple(sorted(restr_triang)))
        return self._hash

    def is_trivially_equivalent(self, other):
        """
        **Description:**
        Checks if the Calabi-Yaus are trivially equivalent by checking if the
        restrictions of the triangulations to codimension-2 faces are the same.

        :::important
        This function only provides a very trivial equivalence check. When this
        function returns False, there is still the possibility of the
        Calabi-Yaus being equivalent, but is only made evident with a change of
        basis. The full equivalence check is generically very difficult, so it
        is not implemented.
        :::

        **Arguments:**
        - ```other``` (CalabiYau): The other CY that is being compared.

        **Returns:**
        (boolean) The truth value of the CYs being equal.
        """
        if self.frst().polytope() != other.frst().polytope():
            return False
        dim = self.frst().dim()
        codim2_faces = [self.frst().points_to_indices(f.points())
                             for f in self.polytope().faces(dim-2)]
        restr_triang_self = set()
        restr_triang_other = set()
        for f in codim2_faces:
            face_pts = set(f)
            for s in self.frst().simplices():
                inters = face_pts & set(s)
                if len(inters) == dim-1:
                    ss = tuple(sorted(inters))
                    restr_triang_self.add(ss)
            for s in other.frst().simplices():
                inters = face_pts & set(s)
                if len(inters) == dim-1:
                    ss = tuple(sorted(inters))
                    restr_triang_other.add(ss)
        return restr_triang_self == restr_triang_other

    def frst(self):
        """
        **Description:**
        Returns the FRST giving rise to the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        (Triangulation) The FRST giving rise to the ambient toric variety.
        """
        return self._frst

    def polytope(self):
        """
        **Description:**
        Returns the polytope whose triangulation gives rise to the ambient
        toric variety.

        **Arguments:**
        None.

        **Returns:**
        (Polytope) The polytope whose triangulation gives rise to the ambient
        toric variety.
        """
        return self._frst.polytope()

    def ambient_dim(self):
        """
        **Description:**
        Returns the complex dimension of the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        (integer) The complex dimension of the ambient toric variety.
        """
        return self.frst().dim()

    def dim(self):
        """
        **Description:**
        Returns the complex dimension of the Calabi-Yau hypersurface.

        **Arguments:**
        None.

        **Returns:**
        (integer) The complex dimension of the Calabi-Yau hypersurface.
        """
        return self.frst().dim() - 1

    def hpq(self, p, q):
        """
        **Description:**
        Returns the Hodge number h^{p,q} of the Calabi-Yau.

        :::note notes
        - Only Calabi-Yaus of dimension 2-4 are currently supported.
        - This function always computes Hodge numbers from scratch. The
          functions [```h11```](#h11), [```h21```](#h21), [```h12```](#h12),
          [```h13```](#h13), and [```h22```](#h22) cache the results so they
          offer improved performance.
        :::

        **Arguments:**
        - ```p``` (integer): The holomorphic index of the Dolbeault cohomology
          of interest.
        - ```q``` (integer): The anti-holomorphic index of the Dolbeault
          cohomology of interest.

        **Returns:**
        (integer) The Hodge number h^{p,q} of the arising Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().hpq(p,q,lattice="N")

    def h11(self):
        """
        **Description:**
        Returns the Hodge number h^{1,1} of the Calabi-Yau.

        :::note
        Only Calabi-Yaus of dimension 2-4 are currently supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{1,1} of Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().h11(lattice="N")

    def h12(self):
        """
        **Description:**
        Returns the Hodge number h^{1,2} of the Calabi-Yau.

        :::note
        Only Calabi-Yaus of dimension 2-4 are currently supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{1,2} of Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().h12(lattice="N")

    def h13(self):
        """
        **Description:**
        Returns the Hodge number h^{1,3} of the Calabi-Yau.

        :::note
        Only Calabi-Yaus of dimension 2-4 are currently supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{1,3} of Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().h13(lattice="N")

    def h21(self):
        """
        **Description:**
        Alias for the [```h12```](#h12) function.

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{2,1} of Calabi-Yau manifold.
        """
        return self.h12()

    def h22(self):
        """
        **Description:**
        Returns the Hodge number h^{2,2} of the Calabi-Yau.

        :::note
        Only Calabi-Yaus of dimension 2-4 are currently supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{2,2} of Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().h22(lattice="N")

    def h31(self):
        """
        **Description:**
        Alias for the [```h13```](#h13) function.

        **Arguments:**
        None.

        **Returns:**
        (integer) The Hodge number h^{3,1} of Calabi-Yau manifold.
        """
        return self.h13()

    def chi(self):
        """
        **Description:**
        Computes the Euler characteristic of the Calabi-Yau.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        (integer) The Euler characteristic of the Calabi-Yau manifold.
        """
        if self.dim() not in (2,3,4):
            raise Exception("Only Calabi-Yaus of dimension 2-4 are currently "
                            "supported.")
        return self.polytope().chi(lattice="N")

    def sr_ideal(self):
        """
        **Description:**
        Returns the Stanley–Reisner ideal of the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        (list) The Stanley–Reisner ideal of the ambient toric variety.
        """
        return self.frst().sr_ideal()

    def glsm_charge_matrix(self, exclude_origin=False):
        """
        **Description:**
        Computes the GLSM charge matrix of the theory resulting from this
        polytope.

        **Arguments:**
        - ```exclude_origin``` (boolean, optional, default=False): Indicates
          whether to use the origin in the calculation. This corresponds to the
          exclusion of the canonical divisor.

        **Returns:**
        (list) The GLSM charge matrix.
        """
        return self.polytope().glsm_charge_matrix(
                                    exclude_origin=exclude_origin,
                                    use_all_points=self.frst()._all_poly_pts)

    def glsm_linear_relations(self, exclude_origin=False):
        """
        **Description:**
        Computes the linear relations of the GLSM charge matrix.

        **Arguments:**
        - ```exclude_origin``` (boolean, optional, default=False): Indicates
          whether to use the origin in the calculation. This corresponds to the
          exclusion of the canonical divisor.
        - ```use_all_points``` (boolean, optional, default=False): By default
          only boundary points not interior to facets are used. If this flag is
          set to true then points interior to facets are also used.

        **Returns:**
        (list) A matrix of linear relations of the columns of the GLSM charge
        matrix.
        """
        return self.polytope().glsm_linear_relations(
                                    exclude_origin=exclude_origin,
                                    use_all_points=self.frst()._all_poly_pts)

    def divisor_basis(self, integral=True, exclude_origin=False):
        """
        **Description:**
        Returns the current basis of divisors of the ambient toric variety.

        **Arguments:**
        - ```exclude_origin``` (boolean, optional, default=False): Indicates
          whether to use the origin in the calculation. This corresponds to the
          exclusion of the canonical divisor.
        - ```integral``` (boolean, optional, default=True): Indicates whether
          to try to find an integral basis for the columns of the GLSM charge
          matrix.(i.e. so that remaining columns can be written as an integer
          linear combination of the basis.)

        **Returns:**
        (list) A list of column indices that form a basis. If a more generic
        basis has been specified with the
        [```set_divisor_basis```](#set_divisor_basis) or
        [```set_curve_basis```](#set_curve_basis) functions then it returns a
        matrix where the rows are the basis elements specified as a linear
        combination of the canonical divisor and the prime toric divisors.

        **Example:** We consider the hypersurface in $\mathbb{P}(1,1,1,6,9)$
        which has $h^{1,1}=2$. If no basis has been specified, then this
        function finds one. If a basis has been specified, then this
        function returns it.
        ```python {5,8}
        from cytools import Polytope
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.divisor_basis()          # We haven't specified any basis before
        # Prints: array([5, 6])
        cy.set_divisor_basis([1,3]) # Here we specify a basis
        cy.divisor_basis()          # We have specified a basis
        # Prints: array([1, 3])
        ```
        """
        if self._divisor_basis is None or integral:
            self._divisor_basis = self.polytope().glsm_basis(
                                    integral=integral,
                                    exclude_origin=exclude_origin,
                                    use_all_points=self.frst()._all_poly_pts)
            self.clear_cache(only_in_basis=True)
        return np.array(self._divisor_basis)

    def set_divisor_basis(self, basis, exclude_origin=False,
                          exact_arithmetic=False):
        """
        **Description:**
        Specifies a basis of divisors of the ambient toric variety. This can
        be done with a vector specifying the indices of the prime toric
        divisors.

        :::tip experimental feature
        There is also the option of setting a generic basis with a matrix that
        specifies basis elements as a linear combination of the h11+4 prime
        toric divisors, or the canonical divisor plus the h11+4 prime toric
        divisors. When using this kind of bases, 64-bit floating-point
        arithmetic is used for things such as intersection numbers since
        numbers can be very large and overflow 64-bit integers. There is the
        option of using exact rational arithmetic by setting
        exact_arithmetic=True, but performance is significantly affected.
        :::

        **Arguments:**
        - ```basis``` (list): Vector or matrix specifying a basis. When a
          vector is used, the entries will be taken as the indices of points of
          the polytope or prime divisors of the ambient toric variety. When a
          matrix is used, the rows are taken as linear combinations of the
          aforementioned divisors.
        - ```exclude_origin``` (boolean, optional, default=False): Whether to
          interpret the indexing specified by the input vector as excluding the
          origin.
        - ```exact_arithmetic``` (boolean, optional, default=False): Whether to
          use exact rational arithmetic instead of floats when using a generic
          basis.

        **Returns:**
        Nothing.

        **Example:**
        See the example in [```divisor_basis```](#divisor_basis) or in
        [Experimental Features](./experimental).
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
            if not config.enable_experimental_features:
                raise Exception("Using generic bases is currently an "
                                "experimental feature and must be enabled in "
                                "the configuration.")
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
                new_b = array_int_to_fmpz(new_b)
            elif exact_arithmetic and t == np.float64:
                new_b = array_float_to_fmpq(new_b)
            elif t == np.int64:
                new_b = np.array(new_b, dtype=float)
            self._divisor_basis = new_b
        else:
            raise Exception("Input must be either a vector or a matrix.")
        # Clear the cache of all in-basis computations
        self.clear_cache(recursive=False, only_in_basis=True)

    def set_curve_basis(self, basis, exclude_origin=False,
                        exact_arithmetic=False):
        """
        **Description:**
        Specifies a basis of curves of the ambient toric variety, which in turn
        induces a basis of divisors. This can be done with a vector specifying
        the indices of the standard basis of the lattice dual to the lattice of
        prime toric divisors. Note that this case is equivalent to using the
        same vector in the [```set_divisor_basis```](#set_divisor_basis)
        function.

        :::tip experimental feature
        There is also the option of setting a generic basis with a matrix that
        specifies basis elements as a linear combination of the dual lattice of
        prome toric divisors. When using this kind of bases, 64-bit
        floating-point arithmetic is used for things such as intersection
        numbers since numbers can be very large and overflow 64-bit integers.
        There is the option of using exact rational arithmetic by setting
        exact_arithmetic=True, but performance is significantly affected.
        :::

        **Arguments:**
        - ```basis``` (list): Vector or matrix specifying a basis. When a
          vector is used, the entries will be taken as indices of the standard
          basis of the dual to the lattice of prime toric divisors. When a
          matrix is used, the rows are taken as linear combinations of the
          aforementioned elements.
        - ```exclude_origin``` (boolean, optional, default=False): Whether to
          interpret the indexing specified by the input vector as excluding the
          origin.
        - ```exact_arithmetic``` (boolean, optional, default=False): Whether to
          use exact rational arithmetic instead of floats when using a generic
          basis.

        **Returns:**
        Nothing.

        **Example:**
        See the analogous example in [```divisor_basis```](#divisor_basis) or a
        more detailed example in [Experimental Features](./experimental).
        """
        b = np.array(basis)
        # Check if the input is a vector
        if len(b.shape) == 1:
            self.set_divisor_basis(b, exclude_origin=exclude_origin,
                                   exact_arithmetic=exact_arithmetic)
            return
        if len(b.shape) != 2:
            raise Exception("Input must be either a vector or a matrix.")
        # Else input is a matrix
        if not config.enable_experimental_features:
            raise Exception("Using generic bases is currently an "
                            "experimental feature and must be enabled in "
                            "the configuration.")
        glsm_cm = self.polytope().glsm_charge_matrix(
                                    exclude_origin=False,
                                    use_all_points=self.frst()._all_poly_pts)
        glsm_rnk = np.linalg.matrix_rank(glsm_cm)
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
            new_b = array_int_to_fmpz(new_b)
        elif exact_arithmetic and t == np.float64:
            new_b = array_float_to_fmpq(new_b)
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
                if np.linalg.matrix_rank(array_fmpq_to_float(np.array(fmpq_mat(
                            b_ext.tolist()).tolist()))) == glsm_cm.shape[1]:
                    break
            if ctr > 10:
                raise Exception("There was a problem finding the inverse "
                                "matrix")
            b_ext_inv = np.array(fmpz_mat(b_ext.tolist()).inv().tolist())
            b_inv = b_ext_inv[:,:glsm_rnk].T
        else:
            b_inv = np.linalg.pinv(new_b).T
        self.set_divisor_basis(b_inv, exact_arithmetic=exact_arithmetic)

    def ambient_mori_cone(self, in_basis=False, exclude_origin=False,
                          from_intersection_numbers=False):
        """
        **Description:**
        Returns the Mori cone of the ambient toric variety.

        **Arguments:**
        - ```in_basis``` (boolean, optional, default=False): Use the current
          basis of curves, which is dual to what the basis returned by the
          [```divisor_basis```](#divisor_basis) function.
        - ```exclude_origin``` (boolean, optional, default=False): Excludes the
          origin of the polytope in the computation, which corresponds to the
          canonical divisor.
        - ```from_intersection_numbers``` (boolean, optional, default=False):
          Compute the rays of the Mori cone using the intersection numbers of
          the ambient variety. This can be faster if they are already computed.
          The set of rays may be different, but they define the same cone.

        **Returns:**
        (Cone) The Mori cone of the ambient toric variety.
        """
        if self._mori_cone[0] is None:
            if from_intersection_numbers:
                rays = (self._compute_ambient_mori_rays_from_intersections_4d()
                        if self.dim() == 3 else
                        self._compute_ambient_mori_rays_from_intersections())
                self._mori_cone[0] = Cone(rays)
            else:
                self._mori_cone[0] = self.frst().cpl_cone().dual()
        # 0: All divs, 1: No origin, 2: In basis
        args_id = (exclude_origin*1 if not in_basis else 0) + in_basis*2
        if self._mori_cone[args_id] is not None:
            return self._mori_cone[args_id]
        rays = self.frst().cpl_cone().hyperplanes()
        basis = self.divisor_basis()
        if not exclude_origin and not in_basis:
            new_rays = rays
        elif exclude_origin and not in_basis:
            new_rays = rays[:,1:]
        else:
            if len(basis.shape) == 2: # If basis is matrix
                new_rays = rays.dot(basis.T)
            else:
                new_rays = rays[:,basis]
        c = Cone(new_rays, check=len(basis.shape)==2)
        self._mori_cone[args_id] = c
        return self._mori_cone[args_id]

    def _compute_ambient_mori_rays_from_intersections(self):
        """
        **Description:**
        Computes the Mori cone rays of the ambient variety using intersection
        numbers.

        :::note
        This function should generally not be called by the user. Instead, it
        is called by the [ambient_mori_cone](#ambient_mori_cone) function when
        the user wants to save some time if the ambient intersection numbers
        were already computed.
        :::

        **Arguments:**
        None.

        **Returns:**
        (list) The list of generating rays of the Mori cone of the ambient
        toric variety.
        """
        ambient_intnums = self.ambient_intersection_numbers(in_basis=False)
        dim = self.dim()
        num_divs = self.h11() + dim + 2
        curve_dict = defaultdict(lambda: [[],[]])
        for ii in ambient_intnums:
            if 0 in ii[:-1]:
                continue
            ctr = Counter(ii[:-1])
            if len(ctr) < dim:
                continue
            for comb in set(combinations(ctr.keys(),dim)):
                crv = tuple(sorted(comb))
                curve_dict[crv][0].append(int(sum([i*(ctr[i]-(i in crv))
                                                for i in ctr])))
                curve_dict[crv][1].append(ii[-1])
        row_set = set()
        for crv in curve_dict:
            g = gcd_list(curve_dict[crv][1])
            row = np.zeros(num_divs, dtype=int)
            for j,jj in enumerate(curve_dict[crv][0]):
                row[jj] = int(round(curve_dict[crv][1][j]/g))
            row_set.add(tuple(row))
        mori_rays = np.array(list(row_set), dtype=int)
        # Compute column corresponding to the origin
        mori_rays[:,0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def _compute_ambient_mori_rays_from_intersections_4d(self):
        """
        **Description:**
        Computes the Mori cone rays of the ambient variety using intersection
        numbers.

        :::note notes
        - This function should generally not be called by the user. Instead,
        this is called by the [ambient_mori_cone](#ambient_mori_cone) function
        when when the user wants to save some time if the ambient intersection
        numbers were already computed.
        - This function is a more optimized version for 4D toric varieties.
        :::

        **Arguments:**
        None.

        **Returns:**
        (list) The list of generating rays of the Mori cone of the ambient
        toric variety.
        """
        ambient_intnums = self.ambient_intersection_numbers(in_basis=False)
        num_divs = int(max([ii[3] for ii in ambient_intnums])) + 1
        curve_dict = {}
        curve_ctr = 0
        curve_sparse_list = []
        for ii in ambient_intnums:
            if ii[0] == 0:
                continue
            if ii[0] == ii[1] == ii[2] == ii[3]:
                continue
            if ii[0] == ii[1] == ii[2]:
                continue
            if ii[1] == ii[2] == ii[3]:
                continue
            if ii[0] == ii[1] and ii[2] == ii[3]:
                continue
            if ii[0] == ii[1]:
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

    def ambient_kahler_cone(self):
        """
        **Description:**
        Returns the Kähler cone of the ambient toric variety in the current
        basis of divisors.

        **Arguments:**
        None.

        **Returns:**
        (Cone) The Kähler cone of the ambient toric variety.
        """
        return self.ambient_mori_cone(in_basis=True).dual()

    def _construct_intnum_equations_4d(self):
        """
        **Description:**
        Auxiliary function used to compute the intersection numbers of the
        ambient toric variety. This function is optimized for 4D varieties.

        **Arguments:**
        None.

        **Returns:**
        (tuple) A tuple where the first compotent is a sparse matrix M, the
        second is a vector C, which are used to solve the system M*X=C, the
        third is the list of intersection numbers not including
        self-intersections, and the fourth is the list of intersection numbers
        that are used as variables in the equation.
        """
        # Origin is at index 0
        pts_ext = np.empty((self.frst().points().shape[0],
                            self.frst().points().shape[1]+1),
                                dtype=int)
        pts_ext[:,:-1] = self.frst().points()
        pts_ext[:,-1] = 1
        linear_relations = self.glsm_linear_relations(exclude_origin=True)
        # First compute the distict intersection numbers
        distintnum_array = sorted([
            [c for c in simp if c!=0]
            + [1/abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in self.frst().simplices()])
        frst = [[c for c in s if c != 0] for s in self.frst().simplices()]
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
        variable_array_3 = [(i,i,i,i) for i in range(1, len(pts_ext))]
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
        eqn_array_3 = [(i,i,i) for i in range(1, len(pts_ext))]
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
        return Mat, C, distintnum_array, variable_array

    def _construct_intnum_equations(self):
        """
        **Description:**
        Auxiliary function used to compute the intersection numbers of the
        ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        (tuple) A tuple where the first compotent is a sparse matrix M, the
        second is a vector C, which are used to solve the system M*X=C, the
        third is the list of intersection numbers not including
        self-intersections, and the fourth is the list of intersection numbers
        that are used as variables in the equation.
        """
        dim = self.dim()
        pts_ext = np.empty((self.frst().points().shape[0],dim+2), dtype=int)
        pts_ext[:,:-1] = self.frst().points()
        pts_ext[:,-1] = 1
        linear_relations = self.glsm_linear_relations(exclude_origin=True)
        # First compute the distict intersection numbers
        distintnum_array = sorted([
            [c for c in simp if c!=0]
            + [1/abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in self.frst().simplices()])
        frst = [[c for c in s if c != 0] for s in self.frst().simplices()]
        simp_n = [set([j for i in [list(combinations(f,n)) for f in frst]
                      for j in i]) for n in range(2,dim+1)]
        simp_n = [[np.array(c) for c in simp_n[n]] for n in range(len(simp_n))]
        # We construct and solve the linear system M*x + C = 0, where M is
        # a rectangular mxn matrix and C is a vector.
        ###################################################################
        ### Define dictionaries, to be used to construct the linear system
        ###################################################################
        ## Dictionary of variables
        # Most intersection numbers are trivially zero, find the possibly
        # nonzero intersection numbers.
        choices_n = []
        for n in range(2,dim+1):
            comb = list(combinations(range(dim),dim+1-n))
            choices = np.empty((len(comb),dim+1), dtype=int)
            choices[:,0] = 0
            for k,c in enumerate(comb):
                for i in range(1,dim+1):
                    choices[k,i] = choices[k,i-1] + (0 if i-1 in c else 1)
            choices_n.append(choices)
        variable_array_1 = [(i,)*(dim+1) for i in range(1,len(pts_ext))]
        variable_array_n = [tuple(s[ch]) for n in range(len(simp_n))
                            for s in simp_n[n] for ch in choices_n[n]]
        variable_array = variable_array_1 + variable_array_n
        variable_dict = {vv:v for v,vv in enumerate(variable_array)}
        ## Dictionary to construct C
        # C is constructed by adding/subtracting distinct intersection
        # numbers.
        c_dict = defaultdict(lambda: [])
        for d in distintnum_array:
            for i in range(len(d)-1):
                c_dict[tuple(c for j,c in enumerate(d[:-1]) if j!= i)
                        ] += [(d[i],d[-1])]
        ## Dictionary to construct M
        eqn_array_1 = [tuple(s) for s in simp_n[-1]]
        eqn_array_2 = [(i,)*dim for i in range(1, len(pts_ext))]
        choices_n = []
        for n in range(2,dim):
            comb = list(combinations(range(dim-1),dim-n))
            choices = np.empty((len(comb),dim), dtype=int)
            choices[:,0] = 0
            for k,c in enumerate(comb):
                for i in range(1,dim):
                    choices[k,i] = choices[k,i-1] + (0 if i-1 in c else 1)
            choices_n.append(choices)
        eqn_array_n = [tuple(s[ch]) for n in range(len(choices_n))
                            for s in simp_n[n] for ch in choices_n[n]]
        eqn_array = eqn_array_1 + eqn_array_2 + eqn_array_n
        eqn_dict = defaultdict(lambda: [])
        for v in variable_array:
            for c in set(combinations(v,dim)):
                k = None
                for i in range(dim+1):
                    if i == dim or v[i] != c[i]:
                        k = i
                        break
                eqn_dict[c] += [(v[k],variable_dict[v])]
        # Construct Linear System
        num_rows = len(linear_relations)*len(eqn_array)
        C = np.zeros(num_rows, dtype=float)
        M_row = []
        M_col = []
        M_val = []
        row_ctr = 0
        for eqn in eqn_array:
            for lin in linear_relations:
                if len(set(eqn)) == dim:
                    c_temp = c_dict[eqn]
                    C[row_ctr] = sum([lin[cc[0]-1]*cc[1] for cc in c_temp])
                eqn_temp = eqn_dict[eqn]
                for e in eqn_temp:
                    M_row.append(row_ctr)
                    M_col.append(e[1])
                    M_val.append(lin[e[0]-1])
                row_ctr+=1
        Mat = csr_matrix((M_val,(M_row,M_col)), dtype=np.float64)
        return Mat, C, distintnum_array, variable_array

    def ambient_intersection_numbers(self, in_basis=False,
                                     zero_as_anticanonical=False, backend="all",
                                     check=True, backend_error_tol=1e-3,
                                     round_to_zero_treshold=1e-3,
                                     round_to_integer_error_tol=1e-3,
                                     verbose=0, exact_arithmetic=False):
        """
        **Description:**
        Returns the intersection numbers of the ambient toric variety.

        :::tip experimental feature
        The intersection numbers are computed as floating-point numbers by
        default, but there is the option to turn them into rationals. The
        process is fairly quick, but verifying that they are correct becomes
        very slow at large $h^{1,1}$.
        :::

        **Arguments:**
        - ```in_basis``` (boolean, optional, default=False): Return the
          intersection numbers in the current basis of divisors.
        - ```zero_as_anticanonical``` (boolean, optional, default=False): Treat
          the zeroth index as corresponding to the anticanonical divisor
          instead of the canonical divisor.
        - ```backend``` (string, optional, default="all"): The sparse linear
          solver to use. Options are "all", "sksparse" and "scipy". When set
          to "all" every solver is tried in order until one succeeds.
        - ```check``` (boolean, optional, default=True): Whether to explicitly
          check the solution to the linear system.
        - ```backend_error_tol``` (float, optional, default=1e-3): Error
          tolerance for the solution of the linear system.
        - ```round_to_zero_treshold``` (float, optional, default=1e-3):
          Intersection numbers with magnitude smaller than this treshold are
          rounded to zero.
        - ```round_to_integer_error_tol``` (float, optional, default=1e-3): All
          intersection numbers of the Calabi-Yau hypersurface must be integers
          up to errors less than this value, when the CY is smooth.
        - ```verbose``` (integer, optional, default=0): The verbosity level.
          - verbose = 0: Do not print anything.
          - verbose = 1: Print linear backend warnings.
        - ```exact_arithmetic``` (boolean, optional, default=False): Converts
          the intersection numbers into exact rational fractions.

        Returns:
        (list) A matrix containing nonzero intersection numbers, in the format:
        [[A,B,C,...,D,Kappa_ABC...D], ...], where A,B,C,...,D are indices of
        divisors and Kappa_ABC...D is the intersection number.
        """
        # 0: (canon,float), 1: (anticanon, float), 2: (basis, float)
        # 4: (canon,fmpq), 5: (anticanon, fmpq), 6: (basis, fmpq)
        args_id = ((1*zero_as_anticanonical if not in_basis else 0)
                    + 2*in_basis + 4*exact_arithmetic)
        if self._ambient_intersection_numbers[args_id] is not None:
            return np.array(self._ambient_intersection_numbers[args_id])
        if (self._ambient_intersection_numbers[0] is None
                or (self._ambient_intersection_numbers[4] is None
                    and exact_arithmetic)):
            backends = ["all", "sksparse", "scipy"]
            if backend not in backends:
                raise Exception("Invalid linear system backend. "
                                f"The options are: {backends}.")
            if exact_arithmetic and not config.enable_experimental_features:
                raise Exception("Using exact arithmetic is an experimental "
                                "feature and must be enabled in the "
                                "configuration.")
            # Construct the linear equations
            Mat, C, distintnum_array, variable_array = (
                                        self._construct_intnum_equations_4d()
                                        if self.dim() == 30 else
                                        self._construct_intnum_equations())
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
            if solution is None:
                raise Exception("Linear system solution failed.")
            if exact_arithmetic:
                solution_fmpq = fmpq_mat([array_float_to_fmpq(solution)
                                            .tolist()]).transpose()
                if check:
                    Mat_fmpq = fmpq_mat(Mat.shape[0],Mat.shape[1])
                    Mat_dok = Mat.todok()
                    for k in Mat_dok.keys():
                        Mat_fmpq[k] = float_to_fmpq(Mat_dok[k])
                    C_fmpq = fmpq_mat([array_float_to_fmpq(C)
                                        .tolist()]).transpose()
                    res = Mat_fmpq*solution_fmpq + C_fmpq
                    if any(np.array(res.tolist()).flat):
                        raise Exception("Conversion to rationals failed.")
            if exact_arithmetic:
                ambient_intnum = ([ii[:-1]+[float_to_fmpq(ii[-1])]
                                        for ii in distintnum_array]
                                    + [list(ii) + [solution_fmpq[i,0]]
                                        for i,ii in enumerate(variable_array)])
            else:
                ambient_intnum = (distintnum_array
                                  +[list(ii) + [solution[i]]
                                   for i,ii in enumerate(variable_array)
                                   if abs(solution[i])>round_to_zero_treshold])
            # Add intersections with canonical divisor
            # First we only compute intersection numbers with a single index 0
            # This is because precision errors add up significantly for
            # intersection numbers with self-intersections of the canonical
            # divisor
            dim = self.dim()
            canon_intnum = defaultdict(lambda: 0)
            for ii in ambient_intnum:
                choices = set(tuple(c for i,c in enumerate(ii[:-1]) if i!=j)
                                for j in range(dim+1))
                for c in choices:
                    canon_intnum[(0,)+c] -= ii[-1]
            # Now we round all intersection numbers of the form K_0i...j to
            # integers if the CY is smooth. Otherwise, we only remove the zero
            # elements
            if self.is_smooth() and not exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    val = canon_intnum[ii]
                    round_val = int(round(val))
                    if abs(val-round_val) > round_to_integer_error_tol:
                        raise Exception("Non-integer intersection numbers "
                                        "detected in a smooth CY.")
                    if round_val != 0:
                        canon_intnum[ii] = round_val
                    else:
                        canon_intnum.pop(ii)
            elif not exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    if abs(canon_intnum[ii]) < round_to_zero_treshold:
                        canon_intnum.pop(ii)
            # Now we compute remaining intersection numbers
            canon_intnum_n = [canon_intnum]
            for n in range(2,dim+2):
                tmp_intnum = defaultdict(lambda: 0)
                for ii,ii_val in canon_intnum_n[-1].items():
                    choices = set(tuple(c for i,c in enumerate(ii[n-1:])
                                        if i!=j) for j in range(dim+2-n))
                    for c in choices:
                        tmp_intnum[(0,)*n+c] -= ii_val
                if not exact_arithmetic:
                    for ii in list(tmp_intnum.keys()):
                        if abs(tmp_intnum[ii]) < round_to_zero_treshold:
                            tmp_intnum.pop(ii)
                canon_intnum_n.append(tmp_intnum)
            for i in range(len(canon_intnum_n)):
                ambient_intnum.extend([list(ii)+[canon_intnum_n[i][ii]]
                                            for ii in canon_intnum_n[i]])
            ambient_intnum = sorted(ambient_intnum)
            if exact_arithmetic:
                self._ambient_intersection_numbers[4]= np.array(ambient_intnum)
                self._ambient_intersection_numbers[0]= [
                                            ii[:-1] + [fmpq_to_float(ii[-1])]
                                                    for ii in ambient_intnum]
            else:
                self._ambient_intersection_numbers[0]= np.array(ambient_intnum)
        # Now ambient intersection numbers have been computed
        if zero_as_anticanonical and not in_basis:
            self._ambient_intersection_numbers[args_id] = np.array([
                ii[:-1].tolist()
                + [ii[-1]*(-1 if sum(ii[:-1] == 0)%2 == 1 else 1)] for ii in
                self._ambient_intersection_numbers[4*exact_arithmetic]])
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                if basis.dtype == float and exact_arithmetic:
                    basis = array_float_to_fmpq(basis)
                self._ambient_intersection_numbers[2+4*exact_arithmetic] = (
                    symmetric_sparse_to_dense_in_basis(
                        self._ambient_intersection_numbers[4*exact_arithmetic],
                        basis))
            else:
                self._ambient_intersection_numbers[2+4*exact_arithmetic] = (
                    filter_tensor_indices(
                        self._ambient_intersection_numbers[4*exact_arithmetic],
                        basis))
        return np.array(self._ambient_intersection_numbers[args_id])

    def intersection_numbers(self, in_basis=False, zero_as_anticanonical=False,
                             backend="all", check=True,
                             backend_error_tol=1e-3,
                             round_to_zero_treshold=1e-3,
                             round_to_integer_error_tol=1e-3, verbose=0,
                             exact_arithmetic=False):
        """
        **Description:**
        Returns the intersection numbers of the Calabi-Yau hypersurface.

        :::tip experimental feature
        The intersection numbers are computed as integers when it is smooth and
        using a subset of prime toric divisors as a basis, and otherwise they
        are computed as floating-point numbers. There is the option to turn
        them into rationals. The process is fairly quick, but verifying that
        they are correct becomes very slow at large $h^{1,1}$.
        :::

        **Arguments:**
        - ```in_basis``` (boolean, optional, default=False): Return the
          intersection numbers in the current basis of divisors.
        - ```zero_as_anticanonical``` (boolean, optional, default=False): Treat
          the zeroth index as corresponding to the anticanonical divisor
          instead of the canonical divisor.
        - ```backend``` (string, optional, default="all"): The sparse linear
          solver to use. Options are "all", "sksparse" and "scipy". When set
          to "all" every solver is tried in order until one succeeds.
        - ```check``` (boolean, optional, default=True): Whether to explicitly
          check the solution to the linear system.
        - ```backend_error_tol``` (float, optional, default=1e-3): Error
          tolerance for the solution of the linear system.
        - ```round_to_zero_treshold``` (float, optional, default=1e-3):
          Intersection numbers with magnitude smaller than this treshold are
          rounded to zero.
        - ```round_to_integer_error_tol``` (float, optional, default=1e-3): All
          intersection numbers of the Calabi-Yau hypersurface must be integers
          up to errors less than this value, when the CY is smooth.
        - ```verbose``` (integer, optional, default=0): The verbosity level.
          - verbose = 0: Do not print anything.
          - verbose = 1: Print linear backend warnings.
        - ```exact_arithmetic``` (boolean, optional, default=False): Converts
          the intersection numbers into exact rational fractions.

        Returns:
        (list) A matrix containing nonzero intersection numbers, in the format:
        [[A,B,C,...,D,Kappa_ABC...D], ...], where A,B,C,...,D are indices of
        divisors and Kappa_ABC...D is the intersection number.
        """
        if exact_arithmetic and not config.enable_experimental_features:
            raise Exception("Using exact arithmetic is an experimental "
                            "feature and must be enabled in the "
                            "configuration.")
        # 0: (canon,float), 1: (anticanon, float), 2: (basis, float)
        # 4: (canon,fmpq), 5: (anticanon, fmpq), 6: (basis, fmpq)
        args_id = ((1*zero_as_anticanonical if not in_basis else 0)
                    + 2*in_basis + 4*exact_arithmetic)
        if self._intersection_numbers[args_id] is not None:
            return np.array(self._intersection_numbers[args_id])
        if ((self._intersection_numbers[0] is None and not exact_arithmetic)
            or (self._intersection_numbers[4] is None and exact_arithmetic)):
            ambient_intnums = self.ambient_intersection_numbers(in_basis=False,
                        backend=backend, check=check,
                        backend_error_tol=backend_error_tol,
                        round_to_zero_treshold=round_to_zero_treshold,
                        round_to_integer_error_tol=round_to_integer_error_tol,
                        verbose=verbose, exact_arithmetic=exact_arithmetic)
            intnum = np.array([[int(c) for c in ii[1:-1]]+[-ii[-1]]
                                   for ii in ambient_intnums if ii[0]==0],
                              dtype=(None if exact_arithmetic
                                        or not self.is_smooth() else int))
            self._intersection_numbers[4*exact_arithmetic] = intnum
        # Now intersection numbers have been computed
        if zero_as_anticanonical and not in_basis:
            self._intersection_numbers[args_id] = np.array([
                    ii[:-1].tolist()
                    + [ii[-1]*(-1 if sum(ii[:-1] == 0)%2 == 1 else 1)]
                    for ii in self._intersection_numbers[4*exact_arithmetic]])
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                if basis.dtype == float and exact_arithmetic:
                    basis = array_float_to_fmpq(basis)
                self._intersection_numbers[args_id] = (
                        symmetric_sparse_to_dense_in_basis(
                            self._intersection_numbers[4*exact_arithmetic],
                            basis))
            else:
                self._intersection_numbers[args_id] = filter_tensor_indices(
                        self._intersection_numbers[4*exact_arithmetic], basis)
        return np.array(self._intersection_numbers[args_id])

    def second_chern_class(self, in_basis=True):
        """
        **Description:**
        Computes the second Chern class of the CY hypersurface. Returns the
        integral of the second Chern class over the prime effective divisors.

        :::note
        This function currently only supports CY 3-folds.
        :::

        **Arguments:**
        - ```in_basis``` (boolean, optional, default=True): Only return the
          integrals over a basis of divisors.

        **Returns:**
        (list) A vector containing the integrals.
        """
        if self.dim() != 3:
            raise Exception("This function currently only supports 3-folds.")
        if self._second_chern_class is None:
            int_nums = [[c-1 for c in ii[:-1]] +[ii[-1]] for ii in
                        self.intersection_numbers(in_basis=False)
                        if min(ii[:-1]) != 0]
            c2 = np.zeros(self.h11()+self.dim()+1, dtype=int)
            for ii in int_nums:
                if ii[0] == ii[1] == ii[2]:
                    continue
                elif ii[0] == ii[1]:
                    c2[ii[0]] += ii[3]
                elif ii[0] == ii[2]:
                    c2[ii[0]] += ii[3]
                elif ii[1] == ii[2]:
                    c2[ii[1]] += ii[3]
                else:
                    c2[ii[0]] += ii[3]
                    c2[ii[1]] += ii[3]
                    c2[ii[2]] += ii[3]
        if in_basis:
            basis = self.divisor_basis(exclude_origin=True)
            if len(basis.shape) == 2: # If basis is matrix
                return c2.dot(basis.T)
            return c2[basis]
        return c2

    def is_smooth(self):
        """
        **Description:**
        Returns True if the Calabi-Yau is smooth.

        **Arguments:**
        None.

        **Returns:**
        (boolean) The truth value of the CY being smooth.
        """
        if self._is_smooth is not None:
            return self._is_smooth
        sm = (True if self.dim() <= 3 else
                all(c.is_smooth() for c in self.frst().fan_cones(
                                                     self.dim(),self.dim()-1)))
        self._is_smooth = sm
        return self._is_smooth

    def ambient_effective_generators(self):
        """
        **Description:**
        Returns the rays that generate the effective cone of the ambient
        variety.

        **Arguments:**
        None.

        **Returns:**
        (list) The rays that generate the effective cone of the ambient
        variety.
        """
        if self._ambient_eff_gens is not None:
            return np.array(self._ambient_eff_gens)
        rays = np.eye(self.h11(), dtype=float).tolist()
        glsm_cm = self.glsm_charge_matrix()
        linrels = self.glsm_linear_relations()
        basis = self.divisor_basis()
        if len(basis.shape) != 1:
            raise Exception("Generic bases are not yet supported.")
        no_basis = [i for i in range(self.h11()+self.dim()+1)
                    if i not in basis]
        linrels_reord = linrels[:,no_basis+basis.tolist()]
        linrels_rref = np.array(
                        fmpz_mat(linrels_reord.tolist()).rref()[0].tolist(),
                        dtype=int)
        for i in range(linrels_rref.shape[0]):
            linrels_rref[i,:] //= int(round(gcd_list(linrels_rref[i,:])))
        for i,ii in enumerate(no_basis):
            linrels_reord[:,ii] = linrels_rref[:,i]
        for i,ii in enumerate(basis,len(no_basis)):
            linrels_reord[:,ii] = linrels_rref[:,i]
        for l in linrels_reord:
            if l[0] != 0:
                continue
            for i in no_basis:
                if l[i] != 0:
                    r = [0]*self.h11()
                    for j,jj in enumerate(basis):
                        r[j] = l[jj]/(-l[i])
                    for j in no_basis:
                        if l[j] != 0 and j != i:
                            raise Exception("An unexpected error occured.")
                    rays.append(r)
        self._ambient_eff_gens = np.array(rays, dtype=float)
        return np.array(self._ambient_eff_gens)

    def ambient_effective_cone(self):
        """
        **Description:**
        Returns the effective cone of the ambient variety.

        **Arguments:**
        None.

        **Returns:**
        (Cone) The effective cone of the ambient variety.
        """
        if self._ambient_eff_cone is not None:
            return self._ambient_eff_cone
        self._ambient_eff_cone = Cone(self.ambient_effective_generators())
        return self._ambient_eff_cone

    def compute_cy_volume(self, tloc):
        """
        **Description:**
        Computes the volume of the Calabi-Yau at a location in the Kähler cone.

        **Arguments:**
        - ```tloc``` (list): A vector specifying a location in the Kähler cone.

        **Returns:**
        (float) The volume of the Calabi-Yau at the specified location.
        """
        intnums = self.intersection_numbers(in_basis=True,
                                            exact_arithmetic=False)
        xvol = 0
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[(0,)*self.dim()], fmpz):
                intnums = array_fmpz_to_int(intnums)
            elif isinstance(intnums[(0,)*len(intnums.shape)], fmpq):
                intnums = array_fmpq_to_float(intnums)
            tmp = np.array(intnums)
            for i in range(self.dim()):
                tmp = np.tensordot(tmp, tloc, axes=[[self.dim()-1-i],[0]])
            xvol = tmp/factorial(self.dim())
        else:
            for ii in intnums:
                mult = np.prod([factorial(c)
                                   for c in Counter(ii[:self.dim()]).values()])
                xvol += (ii[-1]*np.prod([tloc[int(j)] for j in ii[:-1]])
                         /mult)
        return xvol

    def compute_divisor_volumes(self, tloc):
        """
        **Description:**
        Computes the volume of the basis toric divisors at a location in the
        Kähler cone.

        **Arguments:**
        - ```tloc``` (list): A vector specifying a location in the Kähler cone.

        **Returns:**
        (list) The list of volumes of the basis divisors at the specified
        location.
        """
        intnums = self.intersection_numbers(in_basis=True,
                                            exact_arithmetic=False)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[(0,)*self.dim()], fmpz):
                intnums = array_fmpz_to_int(intnums)
            elif isinstance(intnums[(0,)*self.dim()], fmpq):
                intnums = array_fmpq_to_float(intnums)
            tmp = np.array(intnums)
            for i in range(1,self.dim()):
                tmp = np.tensordot(tmp, tloc, axes=[[self.dim()-1-i],[0]])
            tau = tmp/factorial(self.dim()-1)
        else:
            tau = np.zeros(len(basis), dtype=float)
            for ii in intnums:
                c = Counter(ii[:-1])
                for j in c.keys():
                    tau[j] += ii[-1] * np.prod(
                                [tloc[k]**(c[k]-(j==k))/factorial(c[k]-(j==k))
                                    for k in c.keys()])
        return np.array(tau)

    def compute_AA(self, tloc):
        """
        **Description:**
        Computes the matrix $\kappa_{ijk}t^k$ at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - ```tloc``` (list): A vector specifying a location in the Kähler cone.

        **Returns:**
        (list) The matrix $\kappa_{ijk}t^k$ at the specified location.
        """
        if self.dim() != 3:
            raise Exception("This function only supports Calabi-Yau 3-folds.")
        intnums = self.intersection_numbers(in_basis=True)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            if isinstance(intnums[(0,)*self.dim()], fmpz):
                intnums = array_fmpz_to_int(intnums)
            elif isinstance(intnums[(0,)*self.dim()], fmpq):
                intnums = array_fmpq_to_float(intnums)
            AA = np.tensordot(intnums, tloc, axes=[[2],[0]])
            return AA
        AA = np.zeros((len(basis),)*2, dtype=float)
        for ii in intnums:
            ii_list = Counter(ii[:-1]).most_common(3)
            if len(ii_list)==1:
                AA[ii_list[0][0],ii_list[0][0]] += ii[-1]*tloc[ii_list[0][0]]
            elif len(ii_list)==2:
                AA[ii_list[0][0],ii_list[0][0]] += ii[-1]*tloc[ii_list[1][0]]
                AA[ii_list[0][0],ii_list[1][0]] += ii[-1]*tloc[ii_list[0][0]]
                AA[ii_list[1][0],ii_list[0][0]] += ii[-1]*tloc[ii_list[0][0]]
            elif len(ii_list)==3:
                AA[ii_list[0][0],ii_list[1][0]] += ii[-1]*tloc[ii_list[2][0]]
                AA[ii_list[1][0],ii_list[0][0]] += ii[-1]*tloc[ii_list[2][0]]
                AA[ii_list[0][0],ii_list[2][0]] += ii[-1]*tloc[ii_list[1][0]]
                AA[ii_list[2][0],ii_list[0][0]] += ii[-1]*tloc[ii_list[1][0]]
                AA[ii_list[1][0],ii_list[2][0]] += ii[-1]*tloc[ii_list[0][0]]
                AA[ii_list[2][0],ii_list[1][0]] += ii[-1]*tloc[ii_list[0][0]]
            else:
                raise Exception("Error: Inconsistent intersection numbers.")
        return AA

    def compute_Kinv(self, tloc):
        """
        **Description:**
        Computes the inverse Kähler metric at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - ```tloc``` (list): A vector specifying a location in the Kähler cone.

        **Returns:**
        (list) The inverse Kähler metric at the specified location.
        """
        if self.dim() != 3:
            raise Exception("This function only supports Calabi-Yau 3-folds.")
        xvol = self.compute_cy_volume(tloc)
        Tau = self.compute_divisor_volumes(tloc)
        AA = self.compute_AA(tloc)
        Kinv = 4*(np.outer(Tau,Tau) - AA*xvol)
        return Kinv
