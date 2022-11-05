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
This module contains tools designed for toric variety computations.
"""

#Standard imports
from collections import Counter, defaultdict
from itertools import combinations
import copy
# Third party imports
from flint import fmpz_mat, fmpq_mat, fmpz, fmpq
from scipy.sparse import csr_matrix
import numpy as np
# CYTools imports
from cytools.utils import (gcd_list, solve_linear_system, array_fmpz_to_int,
                           array_int_to_fmpz, array_float_to_fmpq,
                           array_fmpq_to_float, filter_tensor_indices,
                           symmetric_sparse_to_dense, float_to_fmpq,
                           symmetric_dense_to_sparse, fmpq_to_float,
                           set_divisor_basis, set_curve_basis)
from cytools.calabiyau import CalabiYau
from cytools.cone import Cone
from cytools import config



class ToricVariety:
    """
    This class handles various computations relating to toric varieties.
    It can be used to compute intersection numbers and the Kähler cone, among
    other things.

    :::important
    Generally, objects of this class should not be constructed directly by the
    end user. Instead, they should be created by the
    [`get_toric_variety`](./triangulation#get_toric_variety) function of the
    [`Triangulation`](./triangulation) class.
    :::

    :::tip experimental features
    Only star triangulations of reflexive polytopes are fully supported. There
    is experimental support for other kinds of triangulations, but they may not
    always work. See [experimental features](./experimental) for more details.
    :::

    ## Constructor

    ### `cytools.toricvariety.ToricVariety`

    **Description:**
    Constructs a `ToricVariety` object. This is handled by the hidden
    [`__init__`](#__init__) function.

    **Arguments:**
    - `triang` *(Triangulation)*: A star triangulation.

    **Example:**
    We construct a ToricVariety from a regular, star triangulation of a
    reflexive polytope. Since this class is not intended to be initialized by
    the end user, we create it via the
    [`get_toric_variety`](./triangulation#get_toric_variety) function of
    the [`Triangulation`](./triangulation) class. In this example we obtain
    $\mathbb{P}^4$.
    ```python {4}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    t = p.triangulate()
    v = t.get_toric_variety()
    # A smooth compact 4-dimensional toric variety with 5 affine patches
    ```
    """

    def __init__(self, triang):
        """
        **Description:**
        Initializes a `ToricVariety` object.

        **Arguments:**
        - `triang` *(Triangulation)*: A star triangulation.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new
        `ToricVariety` object. We construct a ToricVariety from a regular,
        star triangulation of a reflexive polytope. Since this class is not
        intended to be initialized by the end user, we create it via the
        [`get_toric_variety`](./triangulation#get_toric_variety) function
        of the [`Triangulation`](./triangulation) class. In this example we
        obtain $\mathbb{P}^4$.
        ```python {4}
        from cytools import Polytope
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        # A smooth compact 4-dimensional toric variety with 5 affine patches
        ```
        """
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        if not triang.is_star():
            raise ValueError("The input triangulation must be star.")
        if not triang.polytope().is_reflexive() and not config._exp_features_enabled:
            raise Exception("The experimental features must be enabled to "
                            "construct toric varieties from triangulations "
                            "that are not from reflexive polytopes.")
        self._triang = triang
        # Initialize remaining hidden attributes
        self._hash = None
        self._glsm_charge_matrix = None
        self._glsm_linrels = None
        self._divisor_basis = None
        self._divisor_basis_mat = None
        self._curve_basis = None
        self._curve_basis_mat = None
        self._mori_cone = [None]*3
        self._intersection_numbers = dict()
        self._prime_divs = None
        self._is_compact = None
        self._is_smooth = None
        self._canon_div_is_smooth = None
        self._eff_cone = None
        self._fan_cones = dict()
        self._nef_part = None
        self._cy = None
        if not self.is_compact() and not config._exp_features_enabled:
            raise Exception("The experimental features must be enabled to "
                            "construct non-compact varieties.")

    def clear_cache(self, recursive=False, only_in_basis=False):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        - `recursive` *(bool, optional, default=False)*: Whether to also
          clear the cache of the defining triangulation and polytope. This is
          ignored when only_in_basis=True.
        - `only_in_basis` *(bool, optional, default=False)*: Only clears
          the cache of computations that depend on a choice of basis.

        **Returns:**
        Nothing.

        **Example:**
        We construct a toric variety, compute its Mori cone, clear the cache
        and then compute it again.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.mori_cone()
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        v.clear_cache() # Clears the cached result
        v.mori_cone() # The Mori cone is recomputed
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        ```
        """
        self._mori_cone[2] = None
        self._eff_cone = None
        for k in list(self._intersection_numbers.keys()):
            if k[1]:
                self._intersection_numbers.pop(k)
        if not only_in_basis:
            self._hash = None
            self._glsm_charge_matrix = None
            self._glsm_linrels = None
            self._divisor_basis = None
            self._divisor_basis_mat = None
            self._curve_basis = None
            self._curve_basis_mat = None
            self._mori_cone = [None]*3
            self._intersection_numbers = dict()
            self._prime_divs = None
            self._is_compact = None
            self._is_smooth = None
            self._canon_div_is_smooth = None
            self._fan_cones = dict()
            self._nef_part = None
            self._cy = None
            if recursive:
                self._triang.clear_cache(recursive=True)

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the toric variety.

        **Example:**
        This function can be used to convert the toric variety to a string or
        to print information about the toric variety.
        ```python {4,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        var_info = str(v) # Converts to string
        print(v) # Prints toric variety info
        # A smooth compact 4-dimensional toric variety with 5 affine patches
        ```
        """
        out_str = (f"A {'smooth' if self.is_smooth() else 'simplicial'} "
                    f"{'' if self.is_compact() else 'non-'}compact {self.dim()}"
                    f"-dimensional toric variety with {len(self.triangulation().simplices())}"
                    f" affine patches")
        return out_str

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of toric varieties with ==.

        **Arguments:**
        - `other` *(ToricVariety)*: The other toric variety that is being
          compared.

        **Returns:**
        *(bool)* The truth value of the toric varieties being equal.

        **Example:**
        We construct two toric varieties and compare them.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        v1 = t1.get_toric_variety()
        t2 = p.triangulate(backend="topcom")
        v2 = t2.get_toric_variety()
        v1 == v2
        # True
        ```
        """
        if not isinstance(other, ToricVariety):
            return NotImplemented
        return self.triangulation() == self.triangulation()

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of toric varieties with !=.

        **Arguments:**
        - `other` *(ToricVariety)*: The other toric variety that is being
          compared.

        **Returns:**
        *(bool)* The truth value of the toric varieties being different.

        **Example:**
        We construct two toric varieties and compare them.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        v1 = t1.get_toric_variety()
        t2 = p.triangulate(backend="topcom")
        v2 = t2.get_toric_variety()
        v1 != v2
        # False
        ```
        """
        if not isinstance(other, ToricVariety):
            return NotImplemented
        return not (self == other)

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from toric varieties.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The hash value of the toric variety.

        **Example:**
        We compute the hash value of a toric variety. Also, we construct a set
        and a dictionary with a toric variety, which make use of the hash
        function.
        ```python {4,5,6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        h = hash(v) # Obtain hash value
        d = {v: 1} # Create dictionary with toric variety keys
        s = {v} # Create a set of toric varieties
        ```
        """
        if self._hash is not None:
            return self._hash
        self._hash = hash((1,hash(self.triangulation())))
        return self._hash

    def is_compact(self):
        """
        **Description:**
        Returns True if the variety is compact and False otherwise.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the variety being compact.

        **Example:**
        We construct a toric variety and check if it is compact.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.is_compact()
        # True
        ```
        """
        if self._is_compact is not None:
            return self._is_compact
        self._is_compact = (0,)*self.dim() in [tuple(pt) for pt in self.polytope().interior_points()]
        return self._is_compact

    def triangulation(self):
        """
        **Description:**
        Returns the triangulation giving rise to the toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(Triangulation)* The triangulation giving rise to the toric variety.

        **Example:**
        We construct a toric variety and check that the triangulation that this
        function returns is the same as the one we used to construct it.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.triangulation() is t
        # True
        ```
        """
        return self._triang

    def polytope(self):
        """
        **Description:**
        Returns the polytope whose triangulation gives rise to the toric
        variety.

        **Arguments:**
        None.

        **Returns:**
        *(Polytope)* The polytope whose triangulation gives rise to the toric
        variety.

        **Example:**
        We construct a toric variety and check that the polytope that this
        function returns is the same as the one we used to construct it.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.polytope() is p
        # True
        ```
        """
        return self._triang.polytope()

    def dimension(self):
        """
        **Description:**
        Returns the complex dimension of the toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The complex dimension of the toric variety.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a toric variety and find its dimension.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.dimension()
        # 4
        ```
        """
        return self.triangulation().dim()
    # Aliases
    dim = dimension

    def sr_ideal(self):
        """
        **Description:**
        Returns the Stanley–Reisner ideal of the toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(tuple)* The Stanley–Reisner ideal of the toric variety.

        **Example:**
        We construct a toric variety and find its SR ideal.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.sr_ideal()
        # array([[1, 4, 5],
        #        [2, 3, 6]])
        ```
        """
        return self.triangulation().sr_ideal()

    def glsm_charge_matrix(self, include_origin=True):
        """
        **Description:**
        Computes the GLSM charge matrix of the theory resulting from this
        toric variety.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.

        **Returns:**
        *(numpy.ndarray)* The GLSM charge matrix.

        **Example:**
        We construct a toric variety and find the GLSM charge matrix.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = p.get_toric_variety()
        v.glsm_charge_matrix()
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        ```
        """
        if self._glsm_charge_matrix is not None:
            return np.array(self._glsm_charge_matrix)[:,(0 if include_origin else 1):]
        self._glsm_charge_matrix = self.polytope().glsm_charge_matrix(
                                            include_origin=True,
                                            points=self.polytope().points_to_indices(self.triangulation().points()))
        return np.array(self._glsm_charge_matrix)[:,(0 if include_origin else 1):]

    def glsm_linear_relations(self, include_origin=True):
        """
        **Description:**
        Computes the linear relations of the GLSM charge matrix.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.

        **Returns:**
        *(numpy.ndarray)* A matrix of linear relations of the columns of the
        GLSM charge matrix.

        **Example:**
        We construct a toric variety and find its GLSM charge matrix and linear
        relations.
        ```python {4,10}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = p.get_toric_variety()
        v.glsm_linear_relations()
        # array([[ 1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3],
        #        [ 0,  6,  0, -1,  0,  0,  2],
        #        [ 0,  1,  0,  0, -1,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0]])
        v.glsm_linear_relations().dot(p.glsm_charge_matrix().T) # By definition this product must be zero
        # array([[0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0]])
        ```
        """
        if self._glsm_linrels is not None:
            return np.array(self._glsm_linrels)[(0 if include_origin else 1):,(0 if include_origin else 1):]
        self._glsm_linrels = self.polytope().glsm_linear_relations(
                                include_origin=True,
                                points=self.polytope().points_to_indices(self.triangulation().points()))
        return np.array(self._glsm_linrels)[(0 if include_origin else 1):,(0 if include_origin else 1):]

    def divisor_basis(self, include_origin=True, as_matrix=False):
        """
        **Description:**
        Returns the current basis of divisors of the toric variety.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Whether to
          include the origin in the indexing of the vector, or in the basis
          matrix.
        - `as_matrix` *(bool, optional, default=False)*: Indicates whether
          to return the basis as a matrix intead of a list of indices of prime
          toric divisors. Note that if a matrix basis was specified, then it
          will always be returned as a matrix.

        **Returns:**
        *(numpy.ndarray)* A list of column indices that form a basis. If a more
        generic basis has been specified with the
        [`set_divisor_basis`](#set_divisor_basis) or
        [`set_curve_basis`](#set_curve_basis) functions then it returns a
        matrix where the rows are the basis elements specified as a linear
        combination of the canonical divisor and the prime toric divisors.

        **Example:**
        We consider a simple toric variety with two independent divisors. If
        no basis has been set, then this function finds one. If a basis has
        been set, then this function returns it.
        ```python {4,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.divisor_basis() # We haven't set any basis
        # array([1, 6])
        v.set_divisor_basis([5,6]) # Here we set a basis
        v.divisor_basis() # We get the basis we set
        # array([5, 6])
        v.divisor_basis(as_matrix=True) # We get the basis in matrix form
        # array([[0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 1]])
        ```
        """
        if self._divisor_basis is None:
            self.set_divisor_basis(self.polytope().glsm_basis(
                                    integral=True,
                                    include_origin=True,
                                    points=self.polytope().points_to_indices(self.triangulation().points()))
                                    )
        if len(self._divisor_basis.shape) == 1:
            if 0 in self._divisor_basis and not include_origin:
                raise Exception("The basis was requested not including the "
                                "origin, but it is included in the current basis.")
            if as_matrix:
                return np.array(self._divisor_basis_mat[:,(0 if include_origin else 1):])
            return np.array(self._divisor_basis) - (0 if include_origin else 1)
        return np.array(self._divisor_basis)

    def set_divisor_basis(self, basis, include_origin=True):
        """
        **Description:**
        Specifies a basis of divisors of the toric variety. This can
        be done with a vector specifying the indices of the prime toric
        divisors.

        :::note
        Only integral bases are supported by CYTools, meaning that all prime
        toric divisors must be able to be written as an integral linear
        combination of the basis divisors.
        :::

        **Arguments:**
        - `basis` *(array_like)*: Vector or matrix specifying a basis. When
          a vector is used, the entries will be taken as the indices of points
          of the polytope or prime divisors of the toric variety. When a
          matrix is used, the rows are taken as linear combinations of the
          aforementioned divisors.
        - `include_origin` *(bool, optional, default=True)*: Whether to
          interpret the indexing specified by the input vector as including the
          origin.

        **Returns:**
        Nothing.

        **Example:**
        We consider a simple toric variety with two independent divisors. We
        first find the default basis it picks and then we set a basis of our
        choice.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.divisor_basis() # We haven't set any basis
        # array([1, 6])
        v.set_divisor_basis([5,6]) # Here we set a basis
        v.divisor_basis() # We get the basis we set
        # array([5, 6])
        v.divisor_basis(as_matrix=True) # We get the basis in matrix form
        # array([[0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 1]])
        ```
        An example for more generic basis choices can be found in the
        [experimental features](./experimental) section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the CalabiYau class.
        set_divisor_basis(self, basis, include_origin=include_origin)

    def curve_basis(self, include_origin=True, as_matrix=False):
        """
        **Description:**
        Returns the current basis of curves of the toric variety.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Whether to
          include the origin in the indexing of the vector, or in the basis
          matrix.
        - `as_matrix` *(bool, optional, default=False)*: Indicates whether
          to return the basis as a matrix intead of a list of indices of prime
          toric divisors. Note that if a matrix basis was specified, then it
          will always be returned as a matrix.

        **Returns:**
        *(numpy.ndarray)* A list of column indices that form a basis. If a more
        generic basis has been specified with the
        [`set_divisor_basis`](#set_divisor_basis) or
        [`set_curve_basis`](#set_curve_basis) functions then it returns a
        matrix where the rows are the basis elements specified as a linear
        combination of the canonical divisor and the prime toric divisors.

        **Example:**
        We consider a simple toric variety with two independent curves. If
        no basis has been set, then this function finds one. If a basis has
        been set, then this function returns it.
        ```python {4,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.curve_basis() # We haven't set any basis
        # array([1, 6])
        v.set_curve_basis([5,6]) # Here we set a basis
        v.curve_basis() # We get the basis we set
        # array([5, 6])
        v.curve_basis(as_matrix=True) # We get the basis in matrix form
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        ```
        """
        if self._curve_basis is None:
            self.set_divisor_basis(self.polytope().glsm_basis(
                                    integral=True,
                                    include_origin=True,
                                    points=self.polytope().points_to_indices(self.triangulation().points()))
                                    )
        if len(self._curve_basis.shape) == 1:
            if 0 in self._curve_basis and not include_origin:
                raise Exception("The basis was requested not including the "
                                "origin, but it is included in the current basis.")
            if as_matrix:
                return np.array(self._curve_basis_mat[:,(0 if include_origin else 1):])
            return np.array(self._curve_basis) - (0 if include_origin else 1)
        return np.array(self._curve_basis)

    def set_curve_basis(self, basis, include_origin=True):
        """
        **Description:**
        Specifies a basis of curves of the toric variety, which in turn
        induces a basis of divisors. This can be done with a vector specifying
        the indices of the standard basis of the lattice dual to the lattice of
        prime toric divisors. Note that this case is equivalent to using the
        same vector in the [`set_divisor_basis`](#set_divisor_basis)
        function.

        :::note
        Only integral bases are supported by CYTools, meaning that all toric
        curves must be able to be written as an integral linear combination of
        the basis curves.
        :::

        **Arguments:**
        - `basis` *(array_like)*: Vector or matrix specifying a basis. When
          a vector is used, the entries will be taken as indices of the
          standard basis of the dual to the lattice of prime toric divisors.
          When a matrix is used, the rows are taken as linear combinations of
          the aforementioned elements.
        - `include_origin` *(bool, optional, default=True)*: Whether to
          interpret the indexing specified by the input vector as including the
          origin.

        **Returns:**
        Nothing.

        **Example:**
        We consider a simple toric variety with two independent curves. We
        first find the default basis of curves it picks and then set a basis of
        our choice.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.curve_basis() # We haven't set any basis
        # array([1, 6])
        v.set_curve_basis([5,6]) # Here we set a basis
        v.curve_basis() # We get the basis we set
        # array([5, 6])
        v.curve_basis(as_matrix=True) # We get the basis in matrix form
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        ```
        Note that when setting a curve basis in this way, the function behaves
        exactly the same as [`set_divisor_basis`](#set_divisor_basis). For
        a more advanced example involving generic bases these two functions
        differ. An example can be found in the
        [experimental features](./experimental) section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the CalabiYau class.
        set_curve_basis(self, basis, include_origin=include_origin)

    def mori_cone(self, in_basis=False, include_origin=True,
                  from_intersection_numbers=False):
        """
        **Description:**
        Returns the Mori cone of the toric variety.

        **Arguments:**
        - `in_basis` *(bool, optional, default=False)*: Use the current
          basis of curves, which is dual to the basis returned by the
          [`divisor_basis`](#divisor_basis) function.
        - `include_origin` *(bool, optional, default=True)*: Includes the
          origin of the polytope in the computation, which corresponds to the
          canonical divisor.
        - `from_intersection_numbers` *(bool, optional, default=False)*:
          Compute the rays of the Mori cone using the intersection numbers of
          the variety. This can be faster if they are already computed.
          The set of rays may be different, but they define the same cone.

        **Returns:**
        *(Cone)* The Mori cone of the toric variety.

        **Example:**
        We construct a toric variety and find its Mori cone in an $h^{1,1}+d+1$
        dimensional lattice (i.e. without a particular choice of basis) and
        in an $h^{1,1}$ dimensional lattice (i.e. after picking a basis of
        curves).
        ```python {4,6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.mori_cone() # By default it does not use a basis of curves.
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        v.mori_cone(in_basis=True) # It uses the dual basis of curves to the current divisor basis
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        ```
        """
        if self._mori_cone[0] is None:
            if from_intersection_numbers:
                rays = (self._compute_mori_rays_from_intersections_4d()
                        if self.dim() == 4 else
                        self._compute_mori_rays_from_intersections())
                self._mori_cone[0] = Cone(rays)
            else:
                self._mori_cone[0] = self.triangulation().secondary_cone().dual()
        # 0: All divs, 1: No origin, 2: In basis
        args_id = ((not include_origin)*1 if not in_basis else 0) + in_basis*2
        if self._mori_cone[args_id] is not None:
            return self._mori_cone[args_id]
        rays = self._mori_cone[0].rays()
        basis = self.divisor_basis()
        if include_origin and not in_basis:
            new_rays = rays
        elif not include_origin and not in_basis:
            new_rays = rays[:,1:]
        else:
            if len(basis.shape) == 2: # If basis is matrix
                new_rays = rays.dot(basis.T)
            else:
                new_rays = rays[:,basis]
        c = Cone(new_rays, check=len(basis.shape)==2)
        self._mori_cone[args_id] = c
        return self._mori_cone[args_id]

    def _compute_mori_rays_from_intersections(self):
        """
        **Description:**
        Computes the Mori cone rays of the variety using intersection numbers.

        :::note
        This function should generally not be called by the user. Instead, it
        is called by the [`mori_cone`](#mori_cone) function when
        the user wants to save some time if the intersection numbers
        were already computed.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The list of generating rays of the Mori cone of the
        toric variety.

        **Example:**
        This function is not intended to be directly used, but it is used in
        the following example. We construct a toric variety and compute the
        Mori cone using its intersection numbers.
        ```python {4}
        p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-6,-9,-18]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.mori_cone(from_intersection_numbers=True)
        # A 5-dimensional rational polyhedral cone in RR^11 generated by 14 rays
        ```
        """
        intnums = self.intersection_numbers(in_basis=False)
        dim = self.dim()
        num_divs = self.glsm_charge_matrix().shape[1]
        curve_dict = defaultdict(lambda: [[],[]])
        for ii in intnums:
            if 0 in ii:
                continue
            ctr = Counter(ii)
            if len(ctr) < dim-1:
                continue
            for comb in set(combinations(ctr.keys(),dim-1)):
                crv = tuple(sorted(comb))
                curve_dict[crv][0].append(int(sum([i*(ctr[i]-(i in crv)) for i in ctr])))
                curve_dict[crv][1].append(intnums[ii])
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

    def _compute_mori_rays_from_intersections_4d(self):
        """
        **Description:**
        Computes the Mori cone rays of the variety using intersection numbers.

        :::note notes
        - This function should generally not be called by the user. Instead,
        this is called by the [`mori_cone`](#mori_cone) function
        when when the user wants to save some time if the intersection
        numbers were already computed.
        - This function is a more optimized version for 4D toric varieties.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The list of generating rays of the Mori cone of the
        toric variety.

        **Example:**
        This function is not intended to be directly used, but it is used in
        the following example. We construct a toric variety and compute the
        Mori cone using its intersection numbers.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.mori_cone(from_intersection_numbers=True)
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        ```
        """
        intnums = self.intersection_numbers(in_basis=False)
        num_divs = self.glsm_charge_matrix().shape[1]
        curve_dict = {}
        curve_ctr = 0
        curve_sparse_list = []
        for ii in intnums:
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
                    curve_sparse_list.append([curve_ctr,ii[1],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[1],ii[2],ii[3])],ii[0],intnums[ii]])
            elif ii[1] == ii[2]:
                if (ii[0],ii[1],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[2],ii[-1]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[3])],ii[2],intnums[ii]])
            elif ii[2] == ii[3]:
                if (ii[0],ii[1],ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[3],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[2])],ii[3],intnums[ii]])
            else:
                if (ii[0],ii[1],ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[3],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[2])],ii[3],intnums[ii]])
                if (ii[0],ii[1],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[1],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[2],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[1],ii[3])],ii[2],intnums[ii]])
                if (ii[0],ii[2],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0],ii[2],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[1],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[0],ii[2],ii[3])],ii[1],intnums[ii]])
                if (ii[1],ii[2],ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[1],ii[2],ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr,ii[0],intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append([curve_dict[(ii[1],ii[2],ii[3])],ii[0],intnums[ii]])
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
        row_list = set(tuple(tuple(tuple(ii) for ii in sorted(row)) for row in row_list))
        mori_rays = np.zeros((len(row_list),num_divs), dtype=int)
        for i,row in enumerate(row_list):
            for ii in row:
                mori_rays[i,int(round(ii[0]))] = round(ii[1])
        # Compute column corresponding to the origin
        mori_rays[:,0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def kahler_cone(self):
        """
        **Description:**
        Returns the Kähler cone of the toric variety in the current basis of
        divisors.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The Kähler cone of the toric variety.

        **Example:**
        We construct a toric variety and find its Kahler cone.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.kahler_cone()
        # A rational polyhedral cone in RR^2 defined by 3 hyperplanes normals
        ```
        """
        return self.mori_cone(in_basis=True).dual()

    def _construct_intnum_equations_4d(self):
        """
        **Description:**
        Auxiliary function used to compute the intersection numbers of the
        toric variety. This function is optimized for 4D varieties.

        **Arguments:**
        None.

        **Returns:**
        *(tuple)* A tuple where the first component is a sparse matrix M, the
        second is a vector C, which are used to solve the system M*X=C, the
        third is the list of intersection numbers not including
        self-intersections, and the fourth is the list of intersection numbers
        that are used as variables in the equation.

        **Example:**
        This function is not intended to be directly used, but it is used in
        the following example. We construct a toric variety and compute its
        intersection numbers.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        intnums = v.intersection_numbers()
        ```
        """
        # Origin is at index 0
        pts_ext = np.empty((self.triangulation().points().shape[0],
                            self.triangulation().points().shape[1]+1),
                                dtype=int)
        pts_ext[:,:-1] = self.triangulation().points()
        pts_ext[:,-1] = 1
        linear_relations = self.glsm_linear_relations(include_origin=False)
        # First compute the distict intersection numbers
        distintnum_array = sorted([
            [c for c in simp if c!=0]
            + [1/abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in self.triangulation().simplices()])
        frst = [[c for c in s if c != 0] for s in self.triangulation().simplices()]
        simp_2 = set([j for i in [list(combinations(f,2)) for f in frst] for j in i])
        simp_3 = set([j for i in [list(combinations(f,3)) for f in frst] for j in i])
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
                                                for s in simp_3] for j in i]
        variable_array_2 = [tuple(j) for i in [[[s[0],s[0],s[1],s[1]],
                                                [s[0],s[0],s[0],s[1]],
                                                [s[0],s[1],s[1],s[1]]]
                                                for s in simp_2] for j in i]
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
                raise RuntimeError("Failed to construct linear system.")
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
        toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(tuple)* A tuple where the first component is a sparse matrix M, the
        second is a vector C, which are used to solve the system M*X=C, the
        third is the list of intersection numbers not including
        self-intersections, and the fourth is the list of intersection numbers
        that are used as variables in the equation.

        **Example:**
        This function is not intended to be directly used, but it is used in
        the following example. We construct a toric variety and compute its
        intersection numbers.
        ```python {4}
        p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-6,-9,-18]])
        t = p.triangulate()
        v = t.get_toric_variety()
        intnums = v.intersection_numbers()
        ```
        """
        dim = self.dim()
        pts_ext = np.empty((self.triangulation().points().shape[0],dim+1), dtype=int)
        pts_ext[:,:-1] = self.triangulation().points()
        pts_ext[:,-1] = 1
        linear_relations = self.glsm_linear_relations(include_origin=False)
        # First compute the distict intersection numbers
        distintnum_array = sorted([
            [c for c in simp if c!=0]
            + [1/abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in self.triangulation().simplices()])
        frst = [[c for c in s if c != 0] for s in self.triangulation().simplices()]
        simp_n = [set([j for i in [list(combinations(f,n)) for f in frst]
                      for j in i]) for n in range(2,dim)]
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
        for n in range(2,dim):
            comb = list(combinations(range(dim-1),dim-n))
            choices = np.empty((len(comb),dim), dtype=int)
            choices[:,0] = 0
            for k,c in enumerate(comb):
                for i in range(1,dim):
                    choices[k,i] = choices[k,i-1] + (0 if i-1 in c else 1)
            choices_n.append(choices)
        variable_array_1 = [(i,)*dim for i in range(1,len(pts_ext))]
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
        eqn_array_2 = [(i,)*(dim-1) for i in range(1, len(pts_ext))]
        choices_n = []
        for n in range(2,dim-1):
            comb = list(combinations(range(dim-2),dim-1-n))
            choices = np.empty((len(comb),dim-1), dtype=int)
            choices[:,0] = 0
            for k,c in enumerate(comb):
                for i in range(1,dim-1):
                    choices[k,i] = choices[k,i-1] + (0 if i-1 in c else 1)
            choices_n.append(choices)
        eqn_array_n = [tuple(s[ch]) for n in range(len(choices_n))
                            for s in simp_n[n] for ch in choices_n[n]]
        eqn_array = eqn_array_1 + eqn_array_2 + eqn_array_n
        eqn_dict = defaultdict(lambda: [])
        for v in variable_array:
            for c in set(combinations(v,dim-1)):
                k = None
                for i in range(dim):
                    if i == dim-1 or v[i] != c[i]:
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
                if len(set(eqn)) == dim-1:
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

    def intersection_numbers(self, in_basis=False, format="dok",
                             zero_as_anticanonical=False, backend="all",
                             check=True, backend_error_tol=1e-3,
                             round_to_zero_threshold=1e-3,
                             round_to_integer_error_tol=5e-2,
                             verbose=0, exact_arithmetic=False):
        """
        **Description:**
        Returns the intersection numbers of the toric variety.

        :::tip experimental feature
        The intersection numbers are computed as floating-point numbers by
        default, but there is the option to turn them into rationals. The
        process is fairly quick, but it is unreliable at large $h^{1,1}$.
        Furthermore, verifying that they are correct becomes very slow at large
        $h^{1,1}$.
        :::

        **Arguments:**
        - `in_basis` *(bool, optional, default=False)*: Return the
          intersection numbers in the current basis of divisors.
        - `format` *(str, optional, default="dok")*: The output format of the
          intersection numbers. The options are "dok", "coo", and "dense". When
          set to "dok" (Dictionary Of Keys), it returns a dictionary where the
          keys are divisor indices in ascending order and the corresponding
          value is their intersection number. When set to "coo" (COOrdinate
          format), it returns a numpy array in the format
          [[a,b,...,c,K_ab...c],...], i.e. all but the last entry of each row
          correspond to divisor indices in ascending order, with the last entry
          of the row being their intersection number. Lastly, when set to
          "dense", it returns the full dense array of intersection numbers.
        - `zero_as_anticanonical` *(bool, optional, default=False)*: Treat
          the zeroth index as corresponding to the anticanonical divisor
          instead of the canonical divisor.
        - `backend` *(str, optional, default="all")*: The sparse linear
          solver to use. Options are "all", "sksparse" and "scipy". When set
          to "all" every solver is tried in order until one succeeds.
        - `check` *(bool, optional, default=True)*: Whether to explicitly
          check the solution to the linear system.
        - `backend_error_tol` *(float, optional, default=1e-3)*: Error
          tolerance for the solution of the linear system.
        - `round_to_zero_threshold` *(float, optional, default=1e-3)*:
          Intersection numbers with magnitude smaller than this threshold are
          rounded to zero.
        - `round_to_integer_error_tol` *(float, optional, default=5e-2)*:
          All intersection numbers of the Calabi-Yau hypersurface must be
          integers up to errors less than this value, when the CY is smooth.
        - `verbose` *(int, optional, default=0)*: The verbosity level.
          - verbose = 0: Do not print anything.
          - verbose = 1: Print linear backend warnings.
        - `exact_arithmetic` *(bool, optional, default=False)*: Converts
          the intersection numbers into exact rational fractions.

        Returns:
        *(dict or numpy.array)* When `format` is set to "dok" (Dictionary Of
        Keys), it returns a dictionary where the keys are divisor indices in
        ascending order and the corresponding value is their intersection
        number. When `format` is set to "coo" (COOrdinate format), it returns
        a numpy array in the format [[a,b,...,c,K_ab...c],...], i.e. all but
        the last entry of each row correspond to divisor indices in ascending
        order, with the last entry of the row being their intersection number.
        Lastly, when set to "dense", it returns the full dense array of
        intersection numbers.

        **Example:**
        We construct a toric variety and compute its intersection numbers We
        demonstrate the usage of the `in_basis` flag and the different
        available output formats.
        ```python {5,15,23,29}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        # By default this function computes the intersection numbers of the canonical and prime toric divisors
        intnum_nobasis = v.intersection_numbers()
        # Let's print the output and see how to interpret it
        print(intnum_nobasis)
        # {(1, 2, 3, 4): 1.0, (1, 2, 3, 5): 1.0, (1, 2, 4, 6): 0.5, (1, 2, 5, 6): 0.5, [the output is too long so we truncate it]
        # The above output means that the intersection number of divisors 1, 2, 3, 4  is 1, and so on
        # Let us now compute the intersection numbers in a given basis of divisors
        # First, let's check the current basis of divisors
        v.divisor_basis()
        # array([1, 6])
        # Now, setting in_basis=True we only compute the intersection numbers of divisors 1 and 6
        intnum_basis = v.intersection_numbers(in_basis=True)
        # Let's print the output and see how to interpret it
        print(intnum_basis)
        # {(0, 0, 1, 1): 0.16666666666667923, (0, 1, 1, 1): -1.0000000000000335, (1, 1, 1, 1): 4.500000000000089}
        # Here, the indices correspond to indices of the basis divisors
        # So the intersection of 1, 1, 6, 6 is 0.1666, and so on
        # Now, let's look at the different output formats. The default one is the "dok" (Dictionary Of Keys) format shown above
        # There is also the "coo" (COOrdinate format)
        print(v.intersection_numbers(in_basis=True, format="coo"))
        # [[ 0.          0.          1.          1.          0.16666667]
        #  [ 0.          1.          1.          1.         -1.        ]
        #  [ 1.          1.          1.          1.          4.5       ]]
        # In this format, all but the last entry of each row are the indices and the last entry of the row is the intersection number
        # Lastrly, there is the "dense" format where it outputs the full dense array
        print(v.intersection_numbers(in_basis=True, format="dense"))
        # [[[[ 0.          0.        ]
        #    [ 0.          0.16666667]]
        #
        #   [[ 0.          0.16666667]
        #    [ 0.16666667 -1.        ]]]
        #
        #
        #  [[[ 0.          0.16666667]
        #    [ 0.16666667 -1.        ]]
        #
        #   [[ 0.16666667 -1.        ]
        #    [-1.          4.5       ]]]]
        ```
        """
        if format not in ("dok", "coo", "dense"):
            raise ValueError("Options for format are \"dok\", \"coo\", \"dense\".")
        if in_basis:
            zero_as_anticanonical = False
        args_id = (zero_as_anticanonical, in_basis, exact_arithmetic, format)
        if args_id in self._intersection_numbers:
            return copy.copy(self._intersection_numbers[args_id])
        if ((False,False,False,"dok") not in self._intersection_numbers
                or ((False,False,True,"dok") not in self._intersection_numbers
                    and exact_arithmetic)):
            backends = ["all", "sksparse", "scipy"]
            if backend not in backends:
                raise ValueError("Invalid linear system backend. "
                                 f"The options are: {backends}.")
            if exact_arithmetic and not config._exp_features_enabled:
                raise ValueError("The experimental features must be enabled to "
                                 "use exact arithmetic.")
            # Construct the linear equations
            # Note that self.dim gives the dimension of the CY not the of the
            # variety
            Mat, C, distintnum_array, variable_array = (self._construct_intnum_equations_4d()
                                                        if self.dim() == 4 else
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
                raise RuntimeError("Linear system solution failed.")
            if exact_arithmetic:
                solution_fmpq = fmpq_mat([array_float_to_fmpq(solution).tolist()]).transpose()
                if check:
                    Mat_fmpq = fmpq_mat(Mat.shape[0],Mat.shape[1])
                    Mat_dok = Mat.todok()
                    for k in Mat_dok.keys():
                        Mat_fmpq[k] = float_to_fmpq(Mat_dok[k])
                    C_fmpq = fmpq_mat([array_float_to_fmpq(C).tolist()]).transpose()
                    res = Mat_fmpq*solution_fmpq + C_fmpq
                    if any(np.array(res.tolist()).flat):
                        raise RuntimeError("Failed to convert to rational numbers.")
            intnums = dict()
            if exact_arithmetic:
                for ii in distintnum_array:
                    intnums[tuple(int(round(j)) for j in ii[:-1])] = float_to_fmpq(ii[-1])
                for i,ii in enumerate(variable_array):
                    if abs(solution[i]) < round_to_zero_threshold:
                        continue
                    intnums[tuple(ii)] = float_to_fmpq(solution[i])
            else:
                for ii in distintnum_array:
                    intnums[tuple(int(round(j)) for j in ii[:-1])] = ii[-1]
                for i,ii in enumerate(variable_array):
                    if abs(solution[i]) < round_to_zero_threshold:
                        continue
                    intnums[tuple(ii)] = solution[i]
            if self.is_smooth():
                if exact_arithmetic:
                    for ii in intnums:
                        c = intnums[ii]
                        if c.q != 1:
                            raise RuntimeError("Non-integer intersection numbers "
                                               "detected in a smooth toric variety.")
                else:
                    for ii in intnums:
                        c = intnums[ii]
                        if abs(round(c)-c) > round_to_integer_error_tol:
                            raise RuntimeError("Non-integer intersection numbers "
                                               "detected in a smooth toric variety.")
                        intnums[ii] = int(round(c))
            # Add intersections with canonical divisor
            # First we only compute intersection numbers with a single index 0
            # This is because precision errors add up significantly for
            # intersection numbers with self-intersections of the canonical
            # divisor
            dim = self.dim()
            canon_intnum = defaultdict(lambda: 0)
            for ii in intnums:
                choices = set(tuple(c for i,c in enumerate(ii) if i!=j) for j in range(dim))
                for c in choices:
                    canon_intnum[(0,)+c] -= intnums[ii]
            # Now we round all intersection numbers of the form K_0i...j to
            # integers if the CY is smooth. Otherwise, we only remove the zero
            # elements
            if self.canonical_divisor_is_smooth() and exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    val = canon_intnum[ii]
                    if val.q != 1:
                        raise RuntimeError(f"Non-integer intersection numbers "
                                           f"detected in a smooth CY. {ii}:{val}")
                    if val != 0:
                        canon_intnum[ii] = val
                    else:
                        canon_intnum.pop(ii)
            elif self.canonical_divisor_is_smooth() and not exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    val = canon_intnum[ii]
                    round_val = int(round(val))
                    if abs(val-round_val) > round_to_integer_error_tol:
                        raise RuntimeError(f"Non-integer intersection numbers "
                                           f"detected in a smooth CY. {ii}:{val}")
                    if round_val != 0:
                        canon_intnum[ii] = round_val
                    else:
                        canon_intnum.pop(ii)
            elif exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    if canon_intnum[ii] == 0:
                        canon_intnum.pop(ii)
            else:
                for ii in list(canon_intnum.keys()):
                    if abs(canon_intnum[ii]) < round_to_zero_threshold:
                        canon_intnum.pop(ii)
            # Now we compute remaining intersection numbers
            canon_intnum_n = [canon_intnum]
            for n in range(2,dim+1):
                tmp_intnum = defaultdict(lambda: 0)
                for ii,ii_val in canon_intnum_n[-1].items():
                    choices = set(tuple(c for i,c in enumerate(ii[n-1:]) if i!=j) for j in range(dim+1-n))
                    for c in choices:
                        tmp_intnum[(0,)*n+c] -= ii_val
                if exact_arithmetic:
                    for ii in list(tmp_intnum.keys()):
                        if tmp_intnum[ii] == 0:
                            tmp_intnum.pop(ii)
                else:
                    for ii in list(tmp_intnum.keys()):
                        if abs(tmp_intnum[ii]) < round_to_zero_threshold:
                            tmp_intnum.pop(ii)
                canon_intnum_n.append(tmp_intnum)
            for i in range(len(canon_intnum_n)):
                for ii in canon_intnum_n[i]:
                    intnums[ii] = canon_intnum_n[i][ii]
            if exact_arithmetic:
                self._intersection_numbers[(False,False,True,"dok")] = intnums
                self._intersection_numbers[(False,False,False,"dok")] = {ii:(int(intnums[ii].p) if intnums[ii].q==1
                                                                            else fmpq_to_float(intnums[ii])) for ii in intnums}
            else:
                self._intersection_numbers[(False,False,False,"dok")]= intnums
        # Now intersection numbers have been computed
        # We now compute the intersection numbers of the basis if necessary
        if zero_as_anticanonical and not in_basis:
            self._intersection_numbers[args_id] = self._intersection_numbers[(False,False,exact_arithmetic,"dok")]
            for ii in self._intersection_numbers[args_id]:
                if 0 not in ii:
                    continue
                self._intersection_numbers[args_id][ii] *= (-1 if sum(ii == 0)%2 == 1 else 1)
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                self._intersection_numbers[(False,True,exact_arithmetic,"dense")] = (
                    symmetric_sparse_to_dense(self._intersection_numbers[(False,False,exact_arithmetic,"dok")], basis))
                self._intersection_numbers[(False,True,exact_arithmetic,"dok")] = (
                    symmetric_dense_to_sparse(self._intersection_numbers[(False,True,exact_arithmetic,"dense")]))
            else:
                self._intersection_numbers[(False,True,exact_arithmetic,"dok")] = filter_tensor_indices(
                    self._intersection_numbers[(False,False,exact_arithmetic,"dok")], basis)
        # Intersection numbers of the basis are now done
        # Finally, we convert into the desired format
        if format == "coo":
            tmpintnums = self._intersection_numbers[(zero_as_anticanonical,in_basis,exact_arithmetic,"dok")]
            self._intersection_numbers[args_id] = np.array([list(ii)+[tmpintnums[ii]] for ii in tmpintnums])
        elif format == "dense":
            self._intersection_numbers[args_id] = (
                symmetric_sparse_to_dense(self._intersection_numbers[(zero_as_anticanonical,in_basis,exact_arithmetic,"dok")]))
        return copy.copy(self._intersection_numbers[args_id])

    def prime_toric_divisors(self):
        """
        **Description:**
        Returns the list of point indices corresponding to prime toric
        divisors. This list simply corresponds to the indices of the boundary
        points that are used in the triangulation.

        **Arguments:**
        None

        **Returns:**
        *(tuple)* The point indices corresponding to prime toric divisors.

        **Example:**
        We construct a toric variety and find the list of prime toric
        divisors.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.prime_toric_divisors()
        # (1, 2, 3, 4, 5, 6)
        ```
        """
        if self._prime_divs is None:
            tri_ind = list(set.union(*[set(s) for s in self.triangulation().simplices()]))
            divs = self.triangulation().triangulation_to_polytope_indices(tri_ind)
            self._prime_divs = tuple(i for i in divs if i)
        return self._prime_divs

    def is_smooth(self):
        """
        **Description:**
        Returns True if the toric variety is smooth.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the toric variety being smooth.

        **Example:**
        We construct two toric varieties and check if they are smooth.
        ```python {4,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t1 = p.triangulate()
        v1 = t.get_toric_variety()
        v1.is_smooth()
        # False
        t2 = p.triangulate(include_points_interior_to_facets=True)
        v2 = t.get_toric_variety()
        v2.is_smooth()
        # True
        ```
        """
        if self._is_smooth is not None:
            return self._is_smooth
        pts = self.triangulation().points()
        pts = np.insert(pts, 0, np.ones(len(pts), dtype=int), axis=1)
        simp = self.triangulation().simplices()
        self._is_smooth = all(abs(int(round(np.linalg.det(pts[s]))))==1 for s in simp)
        return self._is_smooth

    def canonical_divisor_is_smooth(self):
        """
        **Description:**
        Returns True if the canonical divisor is smooth.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the canonical divisor being smooth.

        **Example:**
        We construct a toric variety and check if its canonical divisor is
        smooth.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.canonical_divisor_is_smooth()
        # True
        ```
        """
        if self._canon_div_is_smooth is not None:
            return self._canon_div_is_smooth
        pts_mpcp = {tuple(pt) for pt in self.polytope().points_not_interior_to_facets()}
        ind_triang = list(set.union(*[set(s) for s in self._triang.simplices()]))
        pts_triang = {tuple(pt) for pt in self._triang.points()[ind_triang]}
        sm = (pts_mpcp.issubset(pts_triang) and
                (True if self.dim() <= 4 else
                all(c.is_smooth() for c in self.fan_cones(self.dim()-1,self.dim()-2))))
        self._canon_div_is_smooth = sm
        return self._canon_div_is_smooth

    def effective_cone(self):
        """
        **Description:**
        Returns the cone of effective divisors, aka the effective cone, of the
        toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The effective cone of the toric variety.

        **Example:**
        We construct a toric variety and find its effective cone.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.effective_cone()
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 6 rays
        ```
        """
        if self._eff_cone is not None:
            return self._eff_cone
        self._eff_cone = Cone(self.curve_basis(include_origin=False,as_matrix=True).T)
        return self._eff_cone

    def fan_cones(self, d=None, face_dim=None):
        """
        **Description:**
        It returns the cones forming a fan defined by a star triangulation of a
        reflexive polytope. The dimension of the desired cones can be
        specified, and one can also restrict to cones that lie in faces of a
        particular dimension.

        **Arguments:**
        - `d` *(int, optional)*: The dimension of the desired cones. If
          not specified, it returns the full-dimensional cones.
        - `face_dim` *(int, optional)*: Restricts to cones that lie on
          faces of the polytope of a particular dimension. If not specified,
          then no restriction is imposed.

        **Returns:**
        *(tuple)* The tuple of cones with the specified properties defined by
        the star triangulation.

        **Example:**
        We construct a toric variety and find the maximal and 2-dimensional
        cones of the defining fan..
        ```python {4,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        max_cones = v.fan_cones() # By default it returns the maximal cones
        cones_2d = v.fan_cones(d=2) # We can select cones of a specific dimension
        ```
        """
        if d is None:
            d = (self.dim() if face_dim is None else face_dim)
        if d not in range(1,self.dim()+1):
            raise ValueError("Only cones of dimension 1 through d are "
                             "supported.")
        if (d,face_dim) in self._fan_cones:
            return self._fan_cones[(d,face_dim)]
        pts = self.triangulation().points()
        cones = set()
        triang_pts_tup =  [tuple(pt) for pt in self.triangulation().points()]
        faces = ([self.triangulation().points_to_indices([tuple(pt) for pt in f.points() if tuple(pt) in triang_pts_tup])
                 for f in self.triangulation()._poly.faces(face_dim)] if face_dim is not None else None)
        for s in self.triangulation().simplices():
            for c in combinations(s,d):
                if (0 not in c and (faces is None or any(all(cc in f for cc in c) for f in faces))):
                    cones.add(tuple(sorted(c)))
        self._fan_cones[(d,face_dim)] = tuple(Cone(pts[list(c)]) for c in cones)
        return self._fan_cones[(d,face_dim)]

    def get_cy(self, nef_partition=None):
        """
        **Description:**
        Returns a CalabiYau object corresponding to the anti-canonical
        hypersurface on the toric variety defined by the fine, star, regular
        triangulation. If a nef-partition is specified then it returns the
        complete intersection Calabi-Yau that it specifies.

        :::note
        Only Calabi-Yau 3-fold hypersurfaces are fully supported. Other
        dimensions and CICYs require enabling the experimental features of
        CYTools. See [experimental features](./experimental) for more details.
        :::

        **Arguments:**
        - `nef_partition` *(list, optional)*: A list of tuples of indices
          specifying a nef-partition of the polytope, which correspondingly
          defines a complete intersection Calabi-Yau.

        **Returns:**
        *(CalabiYau)* The Calabi-Yau arising from the triangulation.

        **Example:**
        We construct a toric variety and obtain its Calabi-Yau hypersurface.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        v = t.get_toric_variety()
        v.get_cy()
        # A Calabi-Yau 3-fold hypersurface with h11=2 and h21=272 in a 4-dimensional toric variety
        ```
        """
        # First check if the nef partition is the same as the cached one.
        # If not, it resets the cached CY.
        if nef_partition != self._nef_part:
            self._cy = None
        if self._cy is not None:
            return self._cy
        if nef_partition is not None:
            if not config._exp_features_enabled:
                raise Exception("The experimental features must be enabled to "
                                "construct CICYs.")
            self._cy = CalabiYau(self, nef_partition)
            self._nef_part = nef_partition
        else:
            if not self.triangulation().is_fine():
                raise ValueError("Triangulation is non-fine.")
            if ((self.dim() != 4 or not self.triangulation().polytope().is_favorable(lattice="N"))
                    and not config._exp_features_enabled):
                raise Exception("The experimental features must be enabled to "
                                "construct non-favorable CYs or CYs with "
                                "dimension other than 3.")
            if not ((self.triangulation().points().shape == self.triangulation().polytope().points_not_interior_to_facets().shape
                     and all((self.triangulation().points() == self.triangulation().polytope().points_not_interior_to_facets()).flat))
                    or (self.triangulation().points().shape == self.triangulation().polytope().points().shape
                        and all((self.triangulation().points() == self.triangulation().polytope().points()).flat))):
                raise ValueError("Calabi-Yau hypersurfaces must be constructed either from points not interior to facets or using all points.")
            self._cy = CalabiYau(self)
            self._nef_part = nef_partition
        return self._cy
