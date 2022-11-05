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
import warnings
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
from cytools.cone import Cone
from cytools import config



class CalabiYau:
    """
    This class handles various computations relating to the Calabi-Yau manifold
    itself. It can be used to compute intersection numbers and the toric Mori and
    Kähler cones, among other things.

    :::important
    Generally, objects of this class should not be constructed directly by the
    end user. Instead, they should be created by the
    [`get_cy`](./toricvariety#get_cy) function of the
    [`ToricVariety`](./toricvariety) class or the
    [`get_cy`](./triangulation#get_cy) function of the
    [`Triangulation`](./triangulation) class.
    :::

    :::tip experimental feature
    This package is focused on computations on Calabi-Yau 3-fold hypersurfaces,
    but there is experimental support for Calabi-Yaus of other dimensions and
    complete intersections. See [experimental features](./experimental) for more
    details.
    :::

    ## Constructor

    ### `cytools.calabiyau.CalabiYau`

    **Description:**
    Constructs a `CalabiYau` object. This is handled by the hidden
    [`__init__`](#__init__) function.

    **Arguments:**
    - `toric_var` *(ToricVariety)*: The ambient toric variety of the
      Calabi-Yau.
    - `nef_partition` *(list, optional)*: A list of tuples of indices
      specifying a nef-partition of the polytope, which correspondingly
      defines a complete intersection Calabi-Yau.

    **Example:**
    We construct a Calabi-Yau from a fine, regular, star triangulation of a
    polytope. Since this class is not intended to be initialized by the end
    user, we create it via the
    [`get_cy`](./triangulation#get_cy) function of the
    [`Triangulation`](./triangulation) class. In this example we obtain the
    quintic hypersurface in $\mathbb{P}^4$.
    ```python {4}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    t = p.triangulate()
    t.get_cy()
    # A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 in a 4-dimensional toric variety
    ```
    """

    def __init__(self, toric_var, nef_partition=None):
        """
        **Description:**
        Initializes a `CalabiYau` object.

        **Arguments:**
        - `toric_var` *(ToricVariety)*: The ambient toric variety of the
          Calabi-Yau.
        - `nef_partition` *(list, optional)*: A list of tuples of indices
          specifying a nef-partition of the polytope, which correspondingly
          defines a complete intersection Calabi-Yau.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new
        `ToricVariety` object. We construct a Calabi-Yau from a fine,
        regular, star triangulation of a polytope. Since this class is not
        intended to be initialized by the end user, we create it via the
        [`get_cy`](./triangulation#get_cy) function of the
        [`Triangulation`](./triangulation) class. In this example we obtain the
        quintic hypersurface in $\mathbb{P}^4$.
        ```python {4}
        from cytools import Polytope
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.get_cy()
        # A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 in a 4-dimensional toric variety
        ```
        """
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        if nef_partition is not None:
            if not config._exp_features_enabled:
                raise Exception("The experimental features must be enabled to "
                                "construct CICYs.")
            # Verify that the input defines a nef-partition
            from cytools import Polytope
            pts = toric_var.polytope().points()
            convpoly = Polytope(pts[list(set.union(*[set(ii) for ii in nef_partition]))])
            if convpoly != toric_var.polytope():
                raise ValueError("Input data does not define a nef partition")
            polys = [Polytope(pts[[0]+list(ii)]) for ii in nef_partition]
            sumpoly = polys[0]
            for i in range(1,len(polys)):
                sumpoly = sumpoly.minkowski_sum(polys[i])
            if not sumpoly.is_reflexive():
                raise ValueError("Input data does not define a nef partition")
            triang = toric_var.triangulation()
            triangpts = [tuple(pt) for pt in triang.points()]
            parts = [tuple(triang.points_to_indices(pt) for pt in pp.points() if any(pt) and tuple(pt) in triangpts)
                        for pp in polys]
            self._nef_part = parts
        else:
            self._nef_part = None
            if not toric_var.triangulation().is_fine():
                raise ValueError("Triangulation is non-fine.")
            if ((toric_var.dim() != 4 or not toric_var.triangulation().polytope().is_favorable(lattice="N"))
                    and not config._exp_features_enabled):
                raise Exception("The experimental features must be enabled to "
                                "construct non-favorable CYs or CYs with "
                                "dimension other than 3.")
            if not ((toric_var.triangulation().points().shape == toric_var.triangulation().polytope().points_not_interior_to_facets().shape
                     and all((toric_var.triangulation().points() == toric_var.triangulation().polytope().points_not_interior_to_facets()).flat))
                    or (toric_var.triangulation().points().shape == toric_var.triangulation().polytope().points().shape
                        and all((toric_var.triangulation().points() == toric_var.triangulation().polytope().points()).flat))):
                raise ValueError("Calabi-Yau hypersurfaces must be constructed either points not interior to facets or all points.")
        self._ambient_var = toric_var
        self._optimal_ambient_var = None
        self._is_hypersurface = nef_partition is None or len(nef_partition) == 1
        # Initialize remaining hidden attributes
        self._hodge_nums = None
        self._dim = None
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
        self._second_chern_class = None
        self._is_smooth = None
        self._eff_cone = None
        if not self._is_hypersurface:
            self._compute_cicy_hodge_numbers(only_from_cache=True)

    def clear_cache(self, recursive=False, only_in_basis=False):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        - `recursive` *(bool, optional, default=True)*: Whether to also
          clear the cache of the ambient toric variety, defining triangulation,
          and polytope. This is ignored when only_in_basis=True.
        - `only_in_basis` *(bool, optional, default=False)*: Only clears
          the cache of computations that depend on a choice of basis.

        **Returns:**
        Nothing.

        **Example:**
        We construct a CY hypersurface, compute its toric Mori cone, clear the
        cache and then compute it again.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.toric_mori_cone()
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        cy.clear_cache() # Clears the cached result
        cy.toric_mori_cone() # The Mori cone is recomputed
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        ```
        """
        self._mori_cone[2] = None
        self._eff_cone = None
        for k in list(self._intersection_numbers.keys()):
            if k[1]:
                self._intersection_numbers.pop(k)
        if not only_in_basis:
            self._dim = None
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
            self._second_chern_class = None
            self._is_smooth = None
            self._hodge_nums = None
            if recursive:
                self.ambient_variety().clear_cache(recursive=True)

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the Calabi-Yau manifold.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the Calabi-Yau manifold.

        **Example:**
        This function can be used to convert the Calabi-Yau to a string or
        to print information about the Calabi-Yau.
        ```python {4,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        cy_info = str(cy) # Converts to string
        print(cy) # Prints Calabi-Yau info
        # A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 in a 4-dimensional toric variety
        ```
        """
        d = self.dim()
        if self._is_hypersurface:
            if d == 2:
                out_str = (f"A K3 hypersurface with h11={self.h11()} in a "
                           "3-dimensional toric variety")
            elif d == 3:
                out_str = ("A Calabi-Yau 3-fold hypersurface with "
                           f"h11={self.h11()} and h21={self.h21()} in a "
                           "4-dimensional toric variety")
            elif d == 4:
                out_str = ("A Calabi-Yau 4-fold hypersurface with "
                           f"h11={self.h11()}, h12={self.h12()}, "
                           f"h13={self.h13()}, and h22={self.h22()} in a "
                           "5-dimensional toric variety")
            else:
                out_str = (f"A Calabi-Yau {d}-fold hypersurface in a "
                           f"{d+1}-dimensional toric variety")
        else:
            dd = self.ambient_variety().dim()
            if self._hodge_nums is None or d not in (2,3,4):
                out_str = (f"A complete intersection Calabi-Yau {d}-fold in a "
                           f"{dd}-dimensional toric variety")
            elif d == 2:
                out_str = (f"A complete intersection K3 surface with "
                           f"h11={self.h11()} in a "
                           f"{dd}-dimensional toric variety")
            elif d == 3:
                out_str = (f"A complete intersection Calabi-Yau 3-fold with "
                           f"h11={self.h11()} h21={self.h21()} in a "
                           + f"{dd}-dimensional toric variety")
            elif d == 4:
                out_str = (f"A complete intersection Calabi-Yau 4-fold with "
                           f"h11={self.h11()}, h12={self.h12()}, "
                           f"h13={self.h13()}, and h22={self.h22()} in a "
                           f"{dd}-dimensional toric variety")
        return out_str

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of Calabi-Yaus with ==.

        :::important
        This function provides only a fairly trivial comparison using the
        [`is_trivially_equivalent`](#is_trivially_equivalent) function. It is
        not recommended to compare CYs with ==, and a warning will be
        printed every time it evaluates to False. This is only implemented so
        that sets and dictionaries of CYs can be created.
        The [`is_trivially_equivalent`](#is_trivially_equivalent) function
        should be used to avoid confusion.
        :::

        **Arguments:**
        - `other` *(CalabiYau)*: The other CY that is being compared.

        **Returns:**
        *(bool)* The truth value of the CYs being equal.

        **Example:**
        We construct two Calabi-Yaus and compare them. We use the
        [`is_trivially_equivalent`](#is_trivially_equivalent) instead of
        this function, since it is recommended to avoid confusion.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        cy1 = t1.get_cy()
        t2 = p.triangulate(backend="topcom")
        cy2 = t2.get_cy()
        cy1.is_trivially_equivalent(cy2)
        # True
        ```
        """
        if not isinstance(other, CalabiYau):
            return NotImplemented
        is_triv_equiv =  self.is_trivially_equivalent(other)
        if is_triv_equiv:
            return True
        warnings.warn("The comparison of CYs should not be done with ==. "
                      "Please use the is_trivially_equivalent function.")
        return False

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of Calabi-Yaus with !=.

        :::important
        This function provides only a fairly trivial comparison using the
        [`is_trivially_equivalent`](#is_trivially_equivalent) function. It is
        not recommended to compare CYs with !=, and a warning will be
        printed every time it evaluates to False. This is only implemented so
        that sets and dictionaries of CYs can be created.
        The [`is_trivially_equivalent`](#is_trivially_equivalent) function
        should be used to avoid confusion.
        :::

        **Arguments:**
        - `other` *(Polytope)*: The other CY that is being compared.

        **Returns:**
        *(bool)* The truth value of the CYs being different.

        **Example:**
        We construct two Calabi-Yaus and compare them. We use the
        [`is_trivially_equivalent`](#is_trivially_equivalent) instead of
        this function, since it is recommended to avoid confusion.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        cy1 = t1.get_cy()
        t2 = p.triangulate(backend="topcom")
        cy2 = t2.get_cy()
        cy1.is_trivially_equivalent(cy2)
        # True
        ```
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
        *(int)* The hash value of the CY.

        **Example:**
        We compute the hash value of a Calabi-Yau. Also, we construct a set
        and a dictionary with a Calabi-Yau, which make use of the hash
        function.
        ```python {4,5,6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        h = hash(cy) # Obtain hash value
        d = {cy: 1} # Create dictionary with Calabi-Yau keys
        s = {cy} # Create a set of Calabi-Yaus
        ```
        """
        if self._hash is not None:
            return self._hash
        if self._is_hypersurface:
            self_orbit = self.triangulation().automorphism_orbit(on_faces_codim=2)
            self_orbit = tuple(tuple(tuple(s) for s in t) for t in self_orbit)
            return hash((hash(self.triangulation().polytope()),hash(self_orbit)))
        else:
            self._hash = hash((hash(self.ambient_variety()),hash(self._nef_part)))
        return self._hash

    def is_trivially_equivalent(self, other):
        """
        **Description:**
        Checks if the Calabi-Yaus are trivially equivalent by checking if the
        restrictions of the triangulations to codimension-2 faces are the same.
        Polytope automorphisms are also taken into account. This function is
        only implemented for CY hypersurfaces.

        :::important
        This function only provides a fairly trivial equivalence check. When this
        function returns False, there is still the possibility of the
        Calabi-Yaus being equivalent, but is only made evident with a change of
        basis. The full equivalence check is generically very difficult, so it
        is not implemented.
        :::

        **Arguments:**
        - `other` (CalabiYau): The other CY that is being compared.

        **Returns:**
        (boolean) The truth value of the CYs being trivially equivalent.

        **Example:**
        We construct two Calabi-Yaus and compare them. We also show how
        to get the set of Calabi-Yaus that are not trivially equivalent. As
        previously mentioned, if two CYs are not trivially equivalent it
        does not mean that they are actually inequivalent as there might
        exist some more complicated basis transformation that relates
        them.
        ```python {5,7}
        p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
        triangs = p.all_triangulations(as_list=True)
        cy0 = triangs[0].get_cy()
        cy1 = triangs[1].get_cy()
        print(cy0.is_trivially_equivalent(cy1))
        # False
        cys_not_triv_eq = {t.get_cy() for t in triangs} # Not trivially equivalent, but not necessarily inequivalent
        print(len(triangs),len(cys_not_triv_eq))        # We see that many CYs from these triangulations can be trivially equated
        # 102 5
        ```
        """
        if not self._is_hypersurface or not other._is_hypersurface:
            return NotImplemented
        if self.polytope() != other.polytope():
            return False
        self_orbit = self.triangulation().automorphism_orbit(on_faces_codim=2)
        other_orbit = other.triangulation().automorphism_orbit(on_faces_codim=2)
        return self_orbit.shape == other_orbit.shape and all((self_orbit == other_orbit).flat)

    def ambient_variety(self):
        """
        **Description:**
        Returns the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(ToricVariety)* The ambient toric variety.

        **Example:**
        We construct a Calabi-Yau hypersurface in a toric variety and check
        that this function returns the ambient variety.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        v = t.get_toric_variety()
        cy = v.get_cy()
        cy.ambient_variety() is v
        # True
        ```
        """
        return self._ambient_var

    def triangulation(self):
        """
        **Description:**
        Returns the triangulation giving rise to the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(Triangulation)* The triangulation giving rise to the ambient toric
        variety.

        **Example:**
        We construct a Calabi-Yau and check that the triangulation that this
        function returns is the same as the one we used to construct it.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.triangulation() is t
        # True
        ```
        """
        return self.ambient_variety().triangulation()

    def polytope(self):
        """
        **Description:**
        Returns the polytope whose triangulation gives rise to the ambient
        toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(Polytope)* The polytope whose triangulation gives rise to the ambient
        toric variety.

        **Example:**
        We construct a Calabi-Yau and check that the polytope that this
        function returns is the same as the one we used to construct it.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.polytope() is p
        # True
        ```
        """
        return self.ambient_variety().polytope()

    def ambient_dimension(self):
        """
        **Description:**
        Returns the complex dimension of the ambient toric variety.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The complex dimension of the ambient toric variety.

        **Aliases:**
        `ambient_dim`.

        **Example:**
        We construct a Calabi-Yau and find the dimension of its ambient
        variety.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.ambient_dimension()
        # 4
        ```
        """
        return self.ambient_variety().dim()
    # Aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """
        **Description:**
        Returns the complex dimension of the Calabi-Yau hypersurface.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The complex dimension of the Calabi-Yau hypersurface.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a Calabi-Yau and find its dimension.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.dimension()
        # 3
        ```
        """
        if self._dim is not None:
            return self._dim
        if self._is_hypersurface:
            self._dim = self.ambient_variety().dim() - 1
        else:
            self._dim = self.ambient_variety().triangulation().dim() - len(self._nef_part)
        return self._dim
    # Aliases
    dim = dimension

    def hpq(self, p, q):
        """
        **Description:**
        Returns the Hodge number $h^{p,q}$ of the Calabi-Yau.

        :::note notes
        - Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
          supported. Hodge numbers of CICYs are computed with PALP.
        - This function always computes Hodge numbers from scratch, unless
          they were computed with PALP. The functions [`h11`](#h11),
          [`h21`](#h21), [`h12`](#h12), [`h13`](#h13), and [`h22`](#h22) cache
          the results so they offer improved performance.
        :::

        **Arguments:**
        - `p` *(int)*: The holomorphic index of the Dolbeault cohomology
          of interest.
        - `q` *(int)*: The anti-holomorphic index of the Dolbeault
          cohomology of interest.

        **Returns:**
        *(int)* The Hodge number $h^{p,q}$ of the arising Calabi-Yau manifold.

        **Example:**
        We construct a Calabi-Yau and check some of its Hodge numbers.
        ```python {4,6,8,10}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.hpq(0,0)
        # 1
        cy.hpq(0,1)
        # 0
        cy.hpq(1,1)
        # 2
        cy.hpq(1,2)
        # 272
        ```
        """
        if not self._is_hypersurface:
            if self._hodge_nums is None:
                self._compute_cicy_hodge_numbers()
            return self._hodge_nums.get((p,q),0)
        if self.dim() not in (2,3,4) and p!=1 and q!=1:
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().hpq(p,q,lattice="N")

    def h11(self):
        """
        **Description:**
        Returns the Hodge number $h^{1,1}$ of the Calabi-Yau.

        :::note
        Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
        supported. Hodge numbers of CICYs are computed with PALP.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The Hodge number $h^{1,1}$ of Calabi-Yau manifold.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its $h^{1,1}$.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.h11()
        # 2
        ```
        """
        if not self._is_hypersurface:
            return self.hpq(1,1)
        if self.dim() not in (2,3,4):
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().h11(lattice="N")

    def h12(self):
        """
        **Description:**
        Returns the Hodge number $h^{1,2}$ of the Calabi-Yau.

        :::note
        Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
        supported. Hodge numbers of CICYs are computed with PALP.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The Hodge number $h^{1,2}$ of Calabi-Yau manifold.

        **Aliases:**
        `h21`.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its $h^{1,2}$.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.h12()
        # 272
        ```
        """
        if not self._is_hypersurface:
            return self.hpq(1,2)
        if self.dim() not in (2,3,4):
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().h12(lattice="N")
    # Aliases
    h21 = h12

    def h13(self):
        """
        **Description:**
        Returns the Hodge number $h^{1,3}$ of the Calabi-Yau.

        :::note
        Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
        supported. Hodge numbers of CICYs are computed with PALP.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The Hodge number $h^{1,3}$ of Calabi-Yau manifold.

        **Aliases:**
        `h31`.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its $h^{1,3}$.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.h13()
        # 0
        ```
        """
        if not self._is_hypersurface:
            return self.hpq(1,3)
        if self.dim() not in (2,3,4):
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().h13(lattice="N")
    # Aliases
    h31 = h13

    def h22(self):
        """
        **Description:**
        Returns the Hodge number $h^{2,2}$ of the Calabi-Yau.

        :::note
        Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
        supported. Hodge numbers of CICYs are computed with PALP.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The Hodge number $h^{2,2}$ of Calabi-Yau manifold.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its $h^{2,2}$.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.h22()
        # 0
        ```
        """
        if not self._is_hypersurface:
            return self.hpq(2,2)
        if self.dim() not in (2,3,4):
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().h22(lattice="N")

    def chi(self):
        """
        **Description:**
        Computes the Euler characteristic of the Calabi-Yau.

        :::note
        Only Calabi-Yau hypersurfaces of dimension 2-4 are currently
        supported. Hodge numbers of CICYs are computed with PALP.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The Euler characteristic of the Calabi-Yau manifold.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its Euler
        characteristic.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.h31()
        # -540
        ```
        """
        if not self._is_hypersurface:
            if self._hodge_nums is None:
                self._compute_cicy_hodge_numbers()
            if "chi" in self._hodge_nums:
                return self._hodge_nums["chi"]
            chi = 0
            for i in range(2*self.dim()+1):
                ii = min(i,self.dim())
                jj = i - ii
                while True:
                    chi += (-1 if i%2 else 1)*self._hodge_nums[(ii,jj)]
                    ii -= 1
                    jj += 1
                    if ii < 0 or jj > self.dim():
                        break
            self._hodge_nums["chi"] = chi
            return self._hodge_nums["chi"]
        if self.dim() not in (2,3,4):
            raise NotImplementedError("Only Calabi-Yaus of dimension 2-4 are currently "
                                      "supported.")
        return self.polytope().chi(lattice="N")

    def glsm_charge_matrix(self, include_origin=True):
        """
        **Description:**
        Computes the GLSM charge matrix of the theory.

        **Arguments:**
        - `include_origin` *(bool, optional, default=True)*: Indicates
          whether to use the origin in the calculation. This corresponds to the
          inclusion of the canonical divisor.

        **Returns:**
        *(numpy.ndarray)* The GLSM charge matrix.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its GLSM charge
        matrix.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.glsm_charge_matrix()
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        ```
        """
        if self._glsm_charge_matrix is not None:
            return np.array(self._glsm_charge_matrix)[:,(0 if include_origin else 1):]
        pts = [0]+list(self.prime_toric_divisors())
        self._glsm_charge_matrix = self.polytope().glsm_charge_matrix(
                                            include_origin=True,
                                            points=pts)
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
        We construct a Calabi-Yau hypersurface and compute the GLSM linear
        relations.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.glsm_linear_relations()
        # array([[ 1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3],
        #        [ 0,  6,  0, -1,  0,  0,  2],
        #        [ 0,  1,  0,  0, -1,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0]])
        ```
        """
        if self._glsm_linrels is not None:
            return np.array(self._glsm_linrels)[(0 if include_origin else 1):,(0 if include_origin else 1):]
        pts = [0]+list(self.prime_toric_divisors())
        self._glsm_linrels = self.polytope().glsm_linear_relations(
                                include_origin=True,
                                points=pts)
        return np.array(self._glsm_linrels)[(0 if include_origin else 1):,(0 if include_origin else 1):]

    def divisor_basis(self, include_origin=True, as_matrix=False):
        """
        **Description:**
        Returns the current basis of divisors of the Calabi-Yau.

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
        We consider a simple Calabi-Yau with two independent divisors. If
        no basis has been set, then this function finds one. If a basis has
        been set, then this function returns it.
        ```python {4,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.divisor_basis() # We haven't set any basis
        # array([1, 6])
        cy.set_divisor_basis([5,6]) # Here we set a basis
        cy.divisor_basis() # We get the basis we set
        # array([5, 6])
        cy.divisor_basis(as_matrix=True) # We get the basis in matrix form
        # array([[0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 1]])
        ```
        """
        if self._divisor_basis is None:
            pts = [0]+list(self.prime_toric_divisors())
            self.set_divisor_basis(self.polytope().glsm_basis(
                                   integral=True,
                                   include_origin=True,
                                   points=pts))
        if len(self._divisor_basis.shape) == 1:
            if 0 in self._divisor_basis and not include_origin:
                raise ValueError("The basis was requested not including the "
                                 "origin, but it is included in the current basis.")
            if as_matrix:
                return np.array(self._divisor_basis_mat[:,(0 if include_origin else 1):])
            return np.array(self._divisor_basis) - (0 if include_origin else 1)
        return np.array(self._divisor_basis[:,(0 if include_origin else 1):])

    def set_divisor_basis(self, basis, include_origin=True):
        """
        **Description:**
        Specifies a basis of divisors of the Calabi-Yau. This can
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
        We consider a simple Calabi-Yau with two independent divisors. We
        first find the default basis it picks and then we set a basis of our
        choice.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.divisor_basis() # We haven't set any basis
        # array([1, 6])
        cy.set_divisor_basis([5,6]) # Here we set a basis
        cy.divisor_basis() # We get the basis we set
        # array([5, 6])
        cy.divisor_basis(as_matrix=True) # We get the basis in matrix form
        # array([[0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 1]])
        ```
        An example for more generic basis choices can be found in the
        [experimental features](./experimental) section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the ToricVariety class.
        set_divisor_basis(self, basis, include_origin=include_origin)

    def curve_basis(self, include_origin=True, as_matrix=False):
        """
        **Description:**
        Returns the current basis of curves of the Calabi-Yau.

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
        We consider a simple Calabi-Yau with two independent curves. If
        no basis has been set, then this function finds one. If a basis has
        been set, then this function returns it.
        ```python {4,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.curve_basis() # We haven't set any basis
        # array([1, 6])
        cy.set_curve_basis([5,6]) # Here we set a basis
        cy.curve_basis() # We get the basis we set
        # array([5, 6])
        cy.curve_basis(as_matrix=True) # We get the basis in matrix form
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
                raise ValueError("The basis was requested not including the "
                                 "origin, but it is included in the current basis.")
            if as_matrix:
                return np.array(self._curve_basis_mat[:,(0 if include_origin else 1):])
            return np.array(self._curve_basis) - (0 if include_origin else 1)
        return np.array(self._curve_basis[:,(0 if include_origin else 1):])

    def set_curve_basis(self, basis, include_origin=True):
        """
        **Description:**
        Specifies a basis of curves of the Calabi-Yau, which in turn
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
        We consider a simple Calabi-Yau with two independent curves. We
        first find the default basis of curves it picks and then set a basis of
        our choice.
        ```python {6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.curve_basis() # We haven't set any basis
        # array([1, 6])
        cy.set_curve_basis([5,6]) # Here we set a basis
        cy.curve_basis() # We get the basis we set
        # array([5, 6])
        cy.curve_basis(as_matrix=True) # We get the basis in matrix form
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
        # shared with the ToricVariety class.
        set_curve_basis(self, basis, include_origin=include_origin)

    def intersection_numbers(self, in_basis=False, format="dok",
                             zero_as_anticanonical=False, backend="all",
                             check=True, backend_error_tol=1e-3,
                             round_to_zero_threshold=1e-3,
                             round_to_integer_error_tol=5e-2, verbose=0,
                             exact_arithmetic=False):
        """
        **Description:**
        Returns the intersection numbers of the Calabi-Yau manifold.

        :::tip experimental feature
        The intersection numbers are computed as integers when the Calabi-Yau
        is smooth, and a subset of the prime toric divisors is used as the
        basis. Otherwise, they are computed as floating-point numbers. There
        is the option to turn them into rationals. The process is fairly
        quick, but it is unreliable at large $h^{1,1}$. Furthermore,
        verifying that they are correct becomes very slow at large $h^{1,1}$.
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
        cy = t.get_cy()
        # By default this function computes the intersection numbers of the canonical and prime toric divisors
        intnum_nobasis = cy.intersection_numbers()
        # Let's print the output and see how to interpret it
        print(intnum_nobasis)
        # {(1, 2, 3): 18, (2, 3, 4): 18, (1, 3, 4): 2, (1, 2, 4): 3, (1, 2, 5): 3, (2, 3, 5): 18, [the output is too long so we truncate it]
        # The above output means that the intersection number of divisors 1, 2, 3  is 18, and so on
        # Let us now compute the intersection numbers in a given basis of divisors
        # First, let's check the current basis of divisors
        cy.divisor_basis()
        # array([1, 6])
        # Now, setting in_basis=True we only compute the intersection numbers of divisors 1 and 6
        intnum_basis = cy.intersection_numbers(in_basis=True)
        # Let's print the output and see how to interpret it
        print(intnum_basis)
        # {(0, 0, 1): 1, (0, 1, 1): -3, (1, 1, 1): 9}
        # Here, the indices correspond to indices of the basis divisors
        # So the intersection of 1, 1, 6 is 1, and so on
        # Now, let's look at the different output formats. The default one is the "dok" (Dictionary Of Keys) format shown above
        # There is also the "coo" (COOrdinate format)
        print(cy.intersection_numbers(in_basis=True, format="coo"))
        # [[ 0  0  1  1]
        #  [ 0  1  1 -3]
        #  [ 1  1  1  9]]
        # In this format, all but the last entry of each row are the indices and the last entry of the row is the intersection number
        # Lastrly, there is the "dense" format where it outputs the full dense array
        print(cy.intersection_numbers(in_basis=True, format="dense"))
        # [[[ 0  1]
        #   [ 1 -3]]
        #
        #  [[ 1 -3]
        #   [-3  9]]]
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
            ambient_intnums = self.ambient_variety().intersection_numbers(
                        in_basis=False, format="dok", backend=backend,
                        check=check, backend_error_tol=backend_error_tol,
                        round_to_zero_threshold=round_to_zero_threshold,
                        round_to_integer_error_tol=round_to_integer_error_tol,
                        verbose=verbose, exact_arithmetic=exact_arithmetic)
            if self._is_hypersurface:
                intnums_cy = {ii[1:]:-ambient_intnums[ii] for ii in ambient_intnums if 0 in ii}
            else:
                triang_pts = [tuple(pt) for pt in self.ambient_variety().triangulation().points()]
                parts = self._nef_part
                ambient_dim = self.ambient_dim()
                intnums_dict = ambient_intnums
                for dd in range(len(parts)):
                    intnums_dict_tmp = defaultdict(lambda: 0)
                    for ii in intnums_dict:
                        choices = set(tuple(sorted(c for i,c in enumerate(ii) if i!=j)) for j in range(ambient_dim-dd) if ii[j] in parts[dd])
                        for c in choices:
                            intnums_dict_tmp[c] += intnums_dict[ii]
                    intnums_dict = {ii:intnums_dict_tmp[ii] for ii in intnums_dict_tmp if abs(intnums_dict_tmp[ii]) > round_to_zero_threshold}
                intnums_cy = intnums_dict
                if all(abs(round(intnums_cy[ii])-intnums_cy[ii]) < round_to_integer_error_tol for ii in intnums_cy):
                    self._is_smooth = True
                    for ii in intnums_cy:
                        intnums_cy[ii] = int(round(intnums_cy[ii]))
                else:
                    self._is_smooth = False
                # Now we find the prime toric divisors and reindex accordingly
                intnum_ind = set.union(*[set(ii) for ii in intnums_cy])
                triang_inds = sorted(intnum_ind)
                self._prime_divs = tuple(self.triangulation().triangulation_to_polytope_indices([i for i in triang_inds if i]))
                divs_dict = {ii:i for i,ii in enumerate(self._prime_divs,1)}
                divs_dict[0] = 0
                intnums_cy = {tuple(divs_dict[i] for i in ii):intnums_cy[ii] for ii in intnums_cy}
                # If there are some non-intersecting divisors we construct a better
                # toric variety in the background
                if len(self._prime_divs) == self.ambient_variety().triangulation().points().shape[0]-1:
                    self._optimal_ambient_var = self._ambient_var
                else:
                    heights = self.triangulation().heights()[triang_inds]
                    try:
                        self._optimal_ambient_var = self.polytope().triangulate(heights=heights, points=self.polytope().points_to_indices(
                                            self.polytope().points()[[0]+list(self._prime_divs)])).get_toric_variety()
                    except:
                        raise NotImplementedError("This type of complete intersection is not supported.")
            self._intersection_numbers[(False,False,exact_arithmetic,"dok")] = intnums_cy
        # Now intersection numbers have been computed
        # We now compute the intersection numbers of the basis if necessary
        if zero_as_anticanonical and not in_basis:
            self._intersection_numbers[(True,False,exact_arithmetic,"dok")] = self._intersection_numbers[(False,False,exact_arithmetic,"dok")]
            for ii in self._intersection_numbers[(True,False,exact_arithmetic,"dok")]:
                if 0 not in ii:
                    continue
                self._intersection_numbers[(True,False,exact_arithmetic,"dok")][ii] *= (-1 if sum(jj == 0 for jj in ii)%2 == 1 else 1)
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
        Returns the list of inherited prime toric divisors. Due to the sorting
        of points in the polytope class, this list is trivial for
        hypersurfaces, but may be non-trivial for CICYs. The indices in the
        returned tuple correspond to indices of the corresponding points of the
        polytope (i.e. if $n$ is in the tuple, then the $n$th point in
        `p.points()` is a prime toric divisor that intersects the CY).

        **Arguments:**
        None

        **Returns:**
        *(tuple)* A list of indices indicating the points in the polytope whose
        corresponding prime toric divisor intersects the CY.

        **Example:**
        We construct a Calabi-Yau hypersurface and find the list of prime toric
        divisors.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.prime_toric_divisors()
        # (1, 2, 3, 4, 5, 6)
        ```
        """
        if self._prime_divs is not None:
            return self._prime_divs
        if self._is_hypersurface:
            self._prime_divs = tuple(range(1,self.polytope().points_not_interior_to_facets().shape[0]))
            self._optimal_ambient_var = self._ambient_var
        else:
            # For CICYs we have to compute intersection numbers and the
            # variables are set during the computation
            self.intersection_numbers()
        return self._prime_divs

    def second_chern_class(self, in_basis=False, include_origin=True):
        """
        **Description:**
        Computes the second Chern class of the CY hypersurface. Returns the
        integral of the second Chern class over the prime effective divisors.

        :::note
        This function currently only supports CY 3-folds.
        :::

        **Arguments:**
        - `in_basis` *(bool, optional, default=False)*: Only return the
          integrals over a basis of divisors.
        - `include_origin` *(bool, optional, default=True)*: Include the origin
          in the vector, which corresponds to the canonical divisor.

        **Returns:**
        *(numpy.ndarray)* A vector containing the integrals.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute its second Chern
        class.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.second_chern_class()
        # array([-612,   36,  306,  204,   36,   36,   -6])
        ```
        """
        if self.dim() != 3:
            raise NotImplementedError("This function currently only supports 3-folds.")
        if self._second_chern_class is None:
            c2 = np.zeros(len(self.prime_toric_divisors())+1, dtype=int)
            intnums = self.intersection_numbers(in_basis=False)
            for ii in intnums:
                if ii[0] == 0:
                    continue
                if ii[0] == ii[1] == ii[2]:
                    continue
                elif ii[0] == ii[1]:
                    c2[ii[0]] += intnums[ii]
                elif ii[0] == ii[2]:
                    c2[ii[0]] += intnums[ii]
                elif ii[1] == ii[2]:
                    c2[ii[1]] += intnums[ii]
                else:
                    c2[ii[0]] += intnums[ii]
                    c2[ii[1]] += intnums[ii]
                    c2[ii[2]] += intnums[ii]
            c2[0] = -np.sum(c2)
            self._second_chern_class = c2
        if in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2: # If basis is matrix
                return self._second_chern_class.dot(basis.T)
            return np.array(self._second_chern_class[basis])
        return np.array(self._second_chern_class)[(0 if include_origin else 1):]

    def is_smooth(self):
        """
        **Description:**
        Returns True if the Calabi-Yau is smooth.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the CY being smooth.

        **Example:**
        We construct a Calabi-Yau hypersurface and check if it is smooth.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.is_smooth()
        # True
        ```
        """
        if self._is_smooth is not None:
            return self._is_smooth
        if self._is_hypersurface:
            self._is_smooth = self.ambient_variety().canonical_divisor_is_smooth()
        else:
            self.intersection_numbers() # The variable is set while computing intersection numbers
        return self._is_smooth

    def toric_mori_cone(self, in_basis=False, include_origin=True,
                          from_intersection_numbers=False):
        """
        **Description:**
        Returns the Mori cone inferred from toric geometry.

        **Arguments:**
        - `in_basis` *(bool, optional, default=False)*: Use the current
          basis of curves, which is dual to what the basis returned by the
          [`divisor_basis`](#divisor_basis) function.
        - `include_origin` *(bool, optional, default=True)*: Includes the
          origin of the polytope in the computation, which corresponds to the
          canonical divisor.
        - `from_intersection_numbers` *(bool, optional, default=False)*:
          Compute the rays of the Mori cone using the intersection numbers of
          the variety. This can be faster if they are already computed.
          The set of rays may be different, but they define the same cone.

        **Returns:**
        *(Cone)* The Mori cone inferred from toric geometry.

        **Example:**
        We construct a Calabi-Yau hypersurface and find its Mori cone in an
        $h^{1,1}+d+1$ dimensional lattice (i.e. without a particular choice of
        basis) and in an $h^{1,1}$ dimensional lattice (i.e. after picking a
        basis of curves).
        ```python {4,6}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.toric_mori_cone() # By default it does not use a basis of curves.
        # A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        cy.toric_mori_cone(in_basis=True) # It uses the dual basis of curves to the current divisor basis
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        ```
        """
        if self._mori_cone[0] is None:
            if self._optimal_ambient_var is None: # Make sure self._optimal_ambient_var is set
                self.prime_toric_divisors()
            self._mori_cone[0] = self._optimal_ambient_var.mori_cone()
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

    def toric_kahler_cone(self):
        """
        **Description:**
        Returns the Kähler cone inferred from toric geometry in the current
        basis of divisors.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The Kähler cone inferred from toric geometry.

        **Example:**
        We construct a Calabi-Yau hypersurface and find its Kähler cone.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.toric_kahler_cone()
        # A rational polyhedral cone in RR^2 defined by 3 hyperplanes normals
        ```
        """
        return self.toric_mori_cone(in_basis=True).dual()

    def toric_effective_cone(self):
        """
        **Description:**
        Returns the cone of effective divisors, aka the effective cone,
        inferred from toric geometry.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The toric effective cone.

        **Example:**
        We construct a Calabi-Yau hypersurface and find its toric effective
        cone.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        cy.toric_effective_cone()
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 6 rays
        ```
        """
        if self._eff_cone is not None:
            return self._eff_cone
        self._eff_cone = Cone(self.curve_basis(include_origin=False,as_matrix=True).T)
        return self._eff_cone

    def compute_cy_volume(self, tloc):
        """
        **Description:**
        Computes the volume of the Calabi-Yau at a location in the Kähler cone.

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.

        **Returns:**
        *(float)* The volume of the Calabi-Yau at the specified location.

        **Example:**
        We construct a Calabi-Yau hypersurface and find its volume at the tip
        of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_cy_volume(tip)
        # 3.4999999988856496
        ```
        """
        intnums = self.intersection_numbers(in_basis=True,
                                            exact_arithmetic=False)
        xvol = 0
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            tmp = np.array(intnums)
            for i in range(self.dim()):
                tmp = np.tensordot(tmp, tloc, axes=[[self.dim()-1-i],[0]])
            xvol = tmp/factorial(self.dim())
        else:
            for ii in intnums:
                mult = np.prod([factorial(c)
                                   for c in Counter(ii).values()])
                xvol += intnums[ii]*np.prod([tloc[int(j)] for j in ii])/mult
        return xvol

    def compute_divisor_volumes(self, tloc, in_basis=False):
        """
        **Description:**
        Computes the volume of the basis divisors at a location in the Kähler
        cone.

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.
        - `in_basis` *(bool, optional, default=False)*: When set to True, the
          volumes of the current basis of divisors are computed. Otherwise, the
          volumes of all prime toric divisors are computed.

        **Returns:**
        *(numpy.ndarray)* The list of volumes of the prime toric divisors or of
        the basis divisors at the specified location.

        **Example:**
        We construct a Calabi-Yau hypersurface and find the volumes of the
        prime toric divisors at the tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_divisor_volumes(tip)
        # array([ 2.5       , 23.99999999, 16.        ,  2.5       ,  2.5       ,
        #         0.5       ])
        ```
        """
        if not in_basis:
            tloc_new = np.array(tloc).dot(self.divisor_basis(as_matrix=True, include_origin=False))
            intnums = self.intersection_numbers(in_basis=False, exact_arithmetic=False)
            tau = np.zeros(len(self.prime_toric_divisors()), dtype=float)
            for ii in intnums:
                if 0 in ii:
                    continue
                c = Counter(ii)
                for j in c.keys():
                    tau[j-1] += intnums[ii] * np.prod(
                                [tloc_new[k-1]**(c[k]-(j==k))/factorial(c[k]-(j==k))
                                    for k in c.keys()])
            return np.array(tau)
        intnums = self.intersection_numbers(in_basis=True, exact_arithmetic=False)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            tmp = np.array(intnums)
            for i in range(1,self.dim()):
                tmp = np.tensordot(tmp, tloc, axes=[[self.dim()-1-i],[0]])
            tau = tmp/factorial(self.dim()-1)
        else:
            tau = np.zeros(len(basis), dtype=float)
            for ii in intnums:
                c = Counter(ii)
                for j in c.keys():
                    tau[j] += intnums[ii] * np.prod(
                                [tloc[k]**(c[k]-(j==k))/factorial(c[k]-(j==k))
                                    for k in c.keys()])
        return np.array(tau)

    def compute_curve_volumes(self, tloc, only_extremal=False):
        """
        **Description:**
        Computes the volume of the curves corresponding to (not necessarily
        minimal) generators of the Mori cone inferred from toric geometry (i.e.
        the cone obtained with the [`toric_mori_cone`](#toric_mori_cone)
        function).

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.
        - `only_extremal` *(bool, optional, default=False)*: Use only the
          extremal rays of the Mori cone.

        **Returns:**
        *(numpy.ndarray)* The list of volumes of the curves.

        **Example:**
        We construct a Calabi-Yau hypersurface and find the volumes of the
        generators of the Mori cone at the tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_curve_volumes(tip)
        # array([0.99997511, 3.99992091, 0.99998193])
        ```
        As expected, all generators of the Mori cone have volumes greater than
        or equal to 1 (up to rounding errors) at the tip of the stretched
        Kähler cone.
        """
        c = self.toric_mori_cone(in_basis=True)
        if only_extremal:
            return c.extremal_rays().dot(tloc)
        return c.rays().dot(tloc)

    def compute_kappa_matrix(self, tloc):
        """
        **Description:**
        Computes the matrix $\kappa_{ijk}t^k$ at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.

        **Returns:**
        *(numpy.ndarray)* The matrix $\kappa_{ijk}t^k$ at the specified
        location.

        **Aliases:**
        `compute_AA`.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute this matrix at the
        tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_kappa_matrix(tip)
        # array([[ 1.,  1.],
        #        [ 1., -3.]])
        ```
        """
        if self.dim() != 3:
            raise NotImplementedError("This function only supports Calabi-Yau 3-folds.")
        intnums = self.intersection_numbers(in_basis=True, exact_arithmetic=False)
        basis = self.divisor_basis()
        if len(basis.shape) == 2: # If basis is matrix
            AA = np.tensordot(intnums, tloc, axes=[[2],[0]])
            return AA
        AA = np.zeros((len(basis),)*2, dtype=float)
        for ii in intnums:
            ii_list = Counter(ii).most_common(3)
            if len(ii_list)==1:
                AA[ii_list[0][0],ii_list[0][0]] += intnums[ii]*tloc[ii_list[0][0]]
            elif len(ii_list)==2:
                AA[ii_list[0][0],ii_list[0][0]] += intnums[ii]*tloc[ii_list[1][0]]
                AA[ii_list[0][0],ii_list[1][0]] += intnums[ii]*tloc[ii_list[0][0]]
                AA[ii_list[1][0],ii_list[0][0]] += intnums[ii]*tloc[ii_list[0][0]]
            elif len(ii_list)==3:
                AA[ii_list[0][0],ii_list[1][0]] += intnums[ii]*tloc[ii_list[2][0]]
                AA[ii_list[1][0],ii_list[0][0]] += intnums[ii]*tloc[ii_list[2][0]]
                AA[ii_list[0][0],ii_list[2][0]] += intnums[ii]*tloc[ii_list[1][0]]
                AA[ii_list[2][0],ii_list[0][0]] += intnums[ii]*tloc[ii_list[1][0]]
                AA[ii_list[1][0],ii_list[2][0]] += intnums[ii]*tloc[ii_list[0][0]]
                AA[ii_list[2][0],ii_list[1][0]] += intnums[ii]*tloc[ii_list[0][0]]
            else:
                raise Exception("Error: Inconsistent intersection numbers.")
        return AA
    # Aliases
    compute_AA = compute_kappa_matrix

    def compute_kappa_vector(self, tloc):
        """
        **Description:**
        Computes the vector $\kappa_{ijk} t^j t^k$ at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.

        **Returns:**
        *(numpy.ndarray)* The vector $\kappa_{ijk} t^j t^k$ at the specified
        location.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute this vector at the
        tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_kappa_vector(tip)
        # array([5., 1.])
        ```
        """
        AA = self.compute_kappa_matrix(tloc)
        return AA.dot(tloc)

    def compute_inverse_kahler_metric(self, tloc):
        """
        **Description:**
        Computes the inverse Kähler metric at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.

        **Returns:**
        *(numpy.ndarray)* The inverse Kähler metric at the specified location.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute the inverse Kähler
        metric at the tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_inverse_kahler_metric(tip)
        # array([[11., -9.],
        #        [-9., 43.]])
        ```
        """
        if self.dim() != 3:
            raise NotImplementedError("This function only supports Calabi-Yau 3-folds.")
        xvol = self.compute_cy_volume(tloc)
        Tau = self.compute_divisor_volumes(tloc, in_basis=True)
        AA = self.compute_AA(tloc)
        Kinv = 4*(np.outer(Tau,Tau) - AA*xvol)
        return Kinv

    def compute_kahler_metric(self, tloc):
        """
        **Description:**
        Computes the Kähler metric at a location in the Kähler cone.

        :::note
        This function only supports Calabi-Yau 3-folds.
        :::

        **Arguments:**
        - `tloc` *(array_like)*: A vector specifying a location in the
          Kähler cone.

        **Returns:**
        *(numpy.ndarray)* The Kähler metric at the specified location.

        **Example:**
        We construct a Calabi-Yau hypersurface and compute the Kähler
        metric at the tip of the stretched Kähler cone.
        ```python {5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        cy = t.get_cy()
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy.compute_kahler_metric(tip)
        # array([[0.10969388, 0.02295918],
                 [0.02295918, 0.02806122]])
        ```
        """
        return np.linalg.inv(self.compute_inverse_kahler_metric(tloc))

    def _compute_cicy_hodge_numbers(self, only_from_cache=False):
        """
        **Description:**
        Computes the Hodge numbers of a CICY using PALP. The results are stored
        in a hidden dictionary.

        :::note
        This function should generally not be called by the user. Instead, it
        is called by [`hpq`](#hpq) and other Hodge number functions when
        necessary.
        :::

        **Arguments:**
        - `only_from_cache` *(bool, optional, default=False)*: Check if the
          Hodge numbers of the CICY were previously computed and are stored in
          the cache of the polytope object. Only if this flag is false and the
          Hodge numbers are not cached, then PALP is used to compute them.

        **Returns:**
        Nothing.

        **Example:**
        This function is not intended to be directly used, but it is used in
        the following example. We construct a CICY and compute some of its
        Hodge numbers.
        ```python {5}
        p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[-1,-1,-6,-9,0],[0,0,0,0,1],[0,0,0,0,-1]])
        nef_part = p.nef_partitions(compute_hodge_numbers=False)
        t = p.triangulate(include_points_interior_to_facets=True)
        cy = t.get_cy(nef_part[0])
        cy.h11() # The function is called here since the Hodge numbers have not been computed
        # 4
        cy.h21() # It is not called here because the Hodge numbers are already cached
        # 544
        ```
        """
        if self._is_hypersurface:
            raise NotImplementedError("This function should only be used for codim > 2 CICYs.")
        if self._hodge_nums is not None:
           return
        codim = self.ambient_variety().dim() - self.dim()
        poly = self.ambient_variety().polytope()
        vert_ind = poly.points_to_indices(poly.vertices())
        nef_part_fs = frozenset(frozenset(i for i in part if i in vert_ind) for part in self._nef_part)
        matched_hodge_nums = ()
        def search_in_cache():
            for args_id in poly._nef_parts:
                # Search only the ones with correct codim, computed hodge numbers, and without products or projections
                if args_id[-2] != codim or not args_id[-1] or args_id[1] or args_id[2]:
                    continue
                for i,nef_part in enumerate(poly._nef_parts[args_id][0]):
                    tmp_fs = frozenset(frozenset(ii for ii in part if ii in vert_ind) for part in nef_part)
                    if tmp_fs == nef_part_fs:
                        return poly._nef_parts[args_id][1][i]
            return ()
        matched_hodge_nums = search_in_cache()
        if not len(matched_hodge_nums) and not only_from_cache:
            poly.nef_partitions()
            matched_hodge_nums = search_in_cache()
        if not len(matched_hodge_nums) and not only_from_cache:
            poly.nef_partitions(keep_symmetric=True)
            matched_hodge_nums = search_in_cache()
        if not len(matched_hodge_nums) and not only_from_cache:
            raise NotImplementedError("This type of complete intersection is not supported.")
        if len(matched_hodge_nums):
            self._hodge_nums = dict()
            n = 0
            for i in range(2*self.dim()+1):
                ii = min(i,self.dim())
                jj = i - ii
                while True:
                    self._hodge_nums[(ii,jj)] = matched_hodge_nums[n]
                    n += 1
                    ii -= 1
                    jj += 1
                    if ii < 0 or jj > self.dim():
                        break
