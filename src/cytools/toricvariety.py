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
# Description:  This module contains tools designed for toric variety
#               computations.
# -----------------------------------------------------------------------------

"""Toric varieties defined by star triangulations of polytopes.

This module provides the [`ToricVariety`][cytools.toricvariety.ToricVariety]
class, which handles computations such as GLSM charge matrices, bases of divisors
and curves, intersection numbers, and Mori and Kähler cones.
"""

# 'standard' imports
from collections import Counter, defaultdict
import copy
from itertools import combinations

# 3rd party imports
import flint
import numpy as np
from scipy.sparse import csr_matrix

# CYTools imports
from cytools import config
from cytools.calabiyau import CalabiYau
from cytools.cone import Cone
from cytools.utils import (
    gcd_list,
    solve_linear_system,
    array_float_to_fmpq,
    filter_tensor_indices,
    symmetric_sparse_to_dense,
    float_to_fmpq,
    symmetric_dense_to_sparse,
    fmpq_to_float,
    set_divisor_basis,
    set_curve_basis,
)


class ToricVariety:
    r"""A toric variety defined by a star triangulation of a polytope.

    This class handles various computations relating to toric varieties. It can
    be used to compute intersection numbers and the Kähler cone, among other
    things.

    Warning:
        Generally, objects of this class should not be constructed directly by
        the end user. Instead, they should be created with the
        [`get_toric_variety`][cytools.triangulation.Triangulation.get_toric_variety]
        method of [`Triangulation`][cytools.triangulation.Triangulation].

    Tip:
        Only star triangulations of reflexive polytopes are fully supported.
        There is experimental support for other kinds of triangulations, but
        they may not always work. See
        [experimental features](experimental.md) for more details.

    Args:
        triang (Triangulation): A star triangulation. Regularity is not
            checked.

    Raises:
        ValueError: If the triangulation is not star.
        Exception: If the polytope is not reflexive, or if the resulting
            variety is not compact, and the experimental features are not
            enabled.

    Examples:
        We construct a toric variety from a regular, star triangulation of a
        reflexive polytope. Since this class is not intended to be initialized
        by the end user, we create it with the
        [`get_toric_variety`][cytools.triangulation.Triangulation.get_toric_variety]
        method of [`Triangulation`][cytools.triangulation.Triangulation]. In this
        example we obtain $\mathbb{P}^4$.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        >>> t = p.triangulate()
        >>> t.get_toric_variety()
        A smooth compact 4-dimensional toric variety with 5 affine patches
    """

    def __init__(self, triang):
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        if not triang.is_star():
            raise ValueError("The input triangulation must be star.")
        if (not triang.polytope().is_reflexive(allow_translations=False)) and (
            not config._exp_features_enabled
        ):
            raise Exception(
                "The experimental features must be enabled to "
                "construct toric varieties from triangulations "
                "that are not from reflexive polytopes."
            )
        self._triang = triang
        # Initialize remaining hidden attributes
        self._hash = None
        self._glsm_charge_matrix = None
        self._glsm_linrels = None
        self._divisor_basis = None
        self._divisor_basis_mat = None
        self._curve_basis = None
        self._curve_basis_mat = None
        self._mori_cone = [None] * 3
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
            raise Exception(
                "The experimental features must be enabled to "
                "construct non-compact varieties."
            )

    def clear_cache(self, recursive=False, only_in_basis=False):
        """Clear the cached results of any previous computation.

        Note that a full cache clear (`only_in_basis=False`) also discards any
        basis of divisors or curves that was previously set, so the default
        basis is used afterwards.

        Args:
            recursive (bool): Whether to also clear the cache of the defining
                triangulation and polytope. This is ignored when
                `only_in_basis=True`.
            only_in_basis (bool): Whether to only clear the cache of
                computations that depend on the choice of basis (the Mori cone
                in the current basis, the effective cone, and the intersection
                numbers in the current basis).

        Examples:
            We construct a toric variety, compute its Mori cone, clear the cache
            and then compute it again.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.mori_cone()
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
            >>> v.clear_cache()  # Clears the cached result
            >>> v.mori_cone()  # The Mori cone is recomputed
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
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
            self._mori_cone = [None] * 3
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
        """Return a string describing the toric variety.

        The string states whether the variety is smooth or only simplicial,
        whether it is compact, its dimension, and its number of affine patches
        (i.e. the number of simplices in the triangulation).

        Returns:
            str: A string describing the toric variety.

        Examples:
            This function can be used to convert the toric variety to a string
            or to print information about the toric variety.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> var_info = str(v)  # Converts to string
            >>> print(v)  # Prints toric variety info
            A smooth compact 4-dimensional toric variety with 5 affine patches
        """
        out_str = (
            f"A {'smooth' if self.is_smooth() else 'simplicial'} "
            f"{'' if self.is_compact() else 'non-'}compact {self.dim()}"
            f"-dimensional toric variety with {len(self.triangulation().simplices())}"
            f" affine patches"
        )
        return out_str

    def __eq__(self, other):
        """Compare two toric varieties with `==`.

        Two toric varieties are considered equal if their defining
        triangulations are equal.

        Args:
            other (ToricVariety): The other toric variety that is being
                compared.

        Returns:
            bool: Whether the toric varieties are equal. If `other` is not a
                `ToricVariety`, `NotImplemented` is returned instead.

        Examples:
            We construct two toric varieties and compare them.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t1 = p.triangulate(backend="qhull")
            >>> v1 = t1.get_toric_variety()
            >>> t2 = p.triangulate(backend="topcom")
            >>> v2 = t2.get_toric_variety()
            >>> v1 == v2
            True
        """
        if not isinstance(other, ToricVariety):
            return NotImplemented
        return self.triangulation() == other.triangulation()

    def __ne__(self, other):
        """Compare two toric varieties with `!=`.

        Args:
            other (ToricVariety): The other toric variety that is being
                compared.

        Returns:
            bool: Whether the toric varieties are different. If `other` is not
                a `ToricVariety`, `NotImplemented` is returned instead.

        Examples:
            We construct two toric varieties and compare them.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t1 = p.triangulate(backend="qhull")
            >>> v1 = t1.get_toric_variety()
            >>> t2 = p.triangulate(backend="topcom")
            >>> v2 = t2.get_toric_variety()
            >>> v1 != v2
            False
        """
        if not isinstance(other, ToricVariety):
            return NotImplemented
        return not (self == other)

    def __hash__(self):
        """Return the hash value of the toric variety.

        The hash is computed from the hash of the defining triangulation and is
        cached.

        Returns:
            int: The hash value of the toric variety.

        Examples:
            We compute the hash value of a toric variety. Also, we construct a
            set and a dictionary with a toric variety, which make use of the
            hash function.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> h = hash(v)  # Obtain hash value
            >>> d = {v: 1}  # Create dictionary with toric variety keys
            >>> s = {v}  # Create a set of toric varieties
        """
        if self._hash is not None:
            return self._hash
        self._hash = hash((1, hash(self.triangulation())))
        return self._hash

    def is_compact(self):
        """Return whether the toric variety is compact.

        The variety is compact when the origin is an interior point of the
        polytope, so that the fan defined by the star triangulation is
        complete.

        Returns:
            bool: Whether the toric variety is compact.

        Examples:
            We construct a toric variety and check if it is compact.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.is_compact()
            True
        """
        if self._is_compact is not None:
            return self._is_compact
        self._is_compact = (0,) * self.dim() in [
            tuple(pt) for pt in self.polytope().interior_points()
        ]
        return self._is_compact

    def triangulation(self):
        """Return the triangulation giving rise to the toric variety.

        Returns:
            Triangulation: The triangulation giving rise to the toric variety.

        Examples:
            We construct a toric variety and check that the triangulation that
            this function returns is the same as the one we used to construct
            it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.triangulation() is t
            True
        """
        return self._triang

    def polytope(self):
        """Return the polytope whose triangulation gives rise to the toric variety.

        Returns:
            Polytope: The polytope whose triangulation gives rise to the toric
                variety.

        Examples:
            We construct a toric variety and check that the polytope that this
            function returns is the same as the one we used to construct it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.polytope() is p
            True
        """
        return self._triang.polytope()

    def dimension(self):
        """Return the complex dimension of the toric variety.

        Also available as `dim`.

        Returns:
            int: The complex dimension of the toric variety.

        Examples:
            We construct a toric variety and find its dimension.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> print(v.dimension())
            4
        """
        return self.triangulation().dim()

    # aliases
    dim = dimension

    def sr_ideal(self):
        """Return the generators of the Stanley–Reisner ideal of the toric variety.

        This is computed by
        [`Triangulation.sr_ideal`][cytools.triangulation.Triangulation.sr_ideal].
        Each generator is a set of prime toric divisors (given by the labels of
        the corresponding points) that do not all lie in a common simplex of the
        triangulation, and that is minimal with this property.

        Returns:
            tuple[tuple[int, ...], ...]: The generators of the Stanley–Reisner
                ideal, each given as a sorted tuple of point labels. The
                generators are sorted by length and then lexicographically.

        Examples:
            We construct a toric variety and find its SR ideal. Here we convert
            the labels to plain integers for readability.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> [tuple(int(i) for i in gen) for gen in v.sr_ideal()]
            [(1, 4, 5), (2, 3, 6)]
        """
        return self.triangulation().sr_ideal()

    def glsm_charge_matrix(self, include_origin=True):
        """Compute the GLSM charge matrix of the theory resulting from this variety.

        The columns correspond to the points of the triangulation, in the same
        order as
        [`Triangulation.points`][cytools.triangulation.Triangulation.points].
        The first column corresponds to the origin (i.e. the canonical divisor)
        and the remaining ones to the prime toric divisors. The result is
        cached, and a copy is returned.

        Args:
            include_origin (bool): Whether to include the column corresponding
                to the origin (i.e. the canonical divisor).

        Returns:
            np.ndarray: The GLSM charge matrix, with one row per independent
                $U(1)$ charge.

        Examples:
            We construct a toric variety and find the GLSM charge matrix.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.glsm_charge_matrix()
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])
        """
        if self._glsm_charge_matrix is not None:
            return np.array(self._glsm_charge_matrix)[:, (0 if include_origin else 1) :]
        self._glsm_charge_matrix = self.polytope().glsm_charge_matrix(
            include_origin=True,
            points=self.polytope().points_to_indices(self.triangulation().points()),
        )
        return np.array(self._glsm_charge_matrix)[:, (0 if include_origin else 1) :]

    def glsm_linear_relations(self, include_origin=True):
        r"""Compute the linear relations among the columns of the GLSM charge matrix.

        Each row is a vector of coefficients $r$ such that
        $\sum_j r_j Q_{\cdot j} = 0$, where $Q$ is the GLSM charge matrix
        returned by
        [`glsm_charge_matrix`][cytools.toricvariety.ToricVariety.glsm_charge_matrix].
        The columns follow the same ordering as the columns of the charge
        matrix. The result is cached, and a copy is returned.

        Args:
            include_origin (bool): Whether to include the origin (i.e. the
                canonical divisor). If false, the first row and the first
                column of the matrix are removed.

        Returns:
            np.ndarray: A matrix whose rows are linear relations among the
                columns of the GLSM charge matrix.

        Examples:
            We construct a toric variety and find the linear relations of its
            GLSM charge matrix. By definition, their product with the transposed
            charge matrix must vanish.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.glsm_linear_relations()
            array([[ 1,  0,  0,  0,  0, 18,  6],
                   [ 0,  1,  0,  0,  0, -1,  0],
                   [ 0,  0,  1,  0,  0, -9, -3],
                   [ 0,  0,  0,  1,  0, -6, -2],
                   [ 0,  0,  0,  0,  1, -1,  0]])
            >>> v.glsm_linear_relations().dot(v.glsm_charge_matrix().T)
            array([[0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0]])
        """
        if self._glsm_linrels is not None:
            return np.array(self._glsm_linrels)[
                (0 if include_origin else 1) :, (0 if include_origin else 1) :
            ]
        self._glsm_linrels = self.polytope().glsm_linear_relations(
            include_origin=True,
            points=self.polytope().points_to_indices(self.triangulation().points()),
        )
        return np.array(self._glsm_linrels)[
            (0 if include_origin else 1) :, (0 if include_origin else 1) :
        ]

    def divisor_basis(self, include_origin=True, as_matrix=False):
        """Return the current basis of divisors of the toric variety.

        If no basis has been set, a default integral basis of prime toric
        divisors is chosen and set. A different basis can be set with
        [`set_divisor_basis`][cytools.toricvariety.ToricVariety.set_divisor_basis]
        or [`set_curve_basis`][cytools.toricvariety.ToricVariety.set_curve_basis].

        Args:
            include_origin (bool): Whether the indexing of the returned vector
                (or the columns of the returned matrix) includes the origin,
                i.e. the canonical divisor at index 0.
            as_matrix (bool): Whether to return the basis as a matrix instead of
                a vector of indices of prime toric divisors. If a generic
                (matrix) basis was set, then it is always returned as a matrix.

        Returns:
            np.ndarray: A vector of the indices of the prime toric divisors
                that form the basis. If `as_matrix` is true, or if a generic
                (matrix) basis has been set, a matrix is returned instead, where
                each row is a basis element expressed as a linear combination
                of the canonical divisor and the prime toric divisors.

        Raises:
            Exception: If `include_origin=False` but the current basis includes
                the canonical divisor.

        Examples:
            We consider a simple toric variety with two independent divisors. If
            no basis has been set, then this function finds one. If a basis has
            been set, then this function returns it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.divisor_basis()  # We haven't set any basis
            array([5, 6])
            >>> v.set_divisor_basis([5,6])  # Here we set a basis
            >>> v.divisor_basis()  # We get the basis we set
            array([5, 6])
            >>> v.divisor_basis(as_matrix=True)  # We get the basis in matrix form
            array([[0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1]])
        """
        if self._divisor_basis is None:
            self.set_divisor_basis(
                self.polytope().glsm_basis(
                    integral=True,
                    include_origin=True,
                    points=self.polytope().points_to_indices(
                        self.triangulation().points()
                    ),
                )
            )
        if len(self._divisor_basis.shape) == 1:
            if 0 in self._divisor_basis and not include_origin:
                raise Exception(
                    "The basis was requested not including the "
                    "origin, but it is included in the current basis."
                )
            if as_matrix:
                return np.array(
                    self._divisor_basis_mat[:, (0 if include_origin else 1) :]
                )
            return np.array(self._divisor_basis) - (0 if include_origin else 1)
        return np.array(self._divisor_basis[:, (0 if include_origin else 1) :])

    def set_divisor_basis(self, basis, include_origin=True):
        """Set a basis of divisors of the toric variety.

        This also sets the dual basis of curves. The basis can be specified
        with a vector of indices of prime toric divisors or, if the experimental
        features are enabled, with a matrix whose rows are linear combinations
        of the canonical divisor and the prime toric divisors. Cached results
        that depend on the choice of basis are cleared.

        Note:
            Only integral bases are supported by CYTools, meaning that all prime
            toric divisors must be able to be written as an integral linear
            combination of the basis divisors.

        Args:
            basis (ArrayLike): Vector or matrix specifying a basis. When a
                vector is used, its entries are taken as the indices of the
                prime toric divisors (i.e. of the points of the triangulation,
                with the origin, corresponding to the canonical divisor, at
                index 0); the indices are sorted before being stored. When a
                matrix is used, its rows are taken as linear combinations of
                the aforementioned divisors. The matrix may omit the column of
                the canonical divisor, in which case its coefficient is set to
                zero.
            include_origin (bool): Whether to interpret the indices in the
                input vector as including the origin at index 0. This is only
                used when `basis` is a vector.

        Raises:
            ValueError: If the input does not specify a valid integral basis.
            Exception: If a matrix is given and the experimental features are
                not enabled.

        Examples:
            We consider a simple toric variety with two independent divisors. We
            first find the default basis it picks and then we set a basis of our
            choice.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.divisor_basis()  # We haven't set any basis
            array([5, 6])
            >>> v.set_divisor_basis([5,6])  # Here we set a basis
            >>> v.divisor_basis()  # We get the basis we set
            array([5, 6])
            >>> v.divisor_basis(as_matrix=True)  # We get the basis in matrix form
            array([[0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1]])

            An example with more generic basis choices can be found in the
            [experimental
            features](experimental.md#generic-bases-for-divisors-and-curves)
            section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the CalabiYau class.
        set_divisor_basis(self, basis, include_origin=include_origin)

    def curve_basis(self, include_origin=True, as_matrix=False):
        """Return the current basis of curves of the toric variety.

        If no basis has been set, a default integral basis of divisors is
        chosen and set, and its dual basis of curves is returned. A different
        basis can be set with
        [`set_curve_basis`][cytools.toricvariety.ToricVariety.set_curve_basis]
        or
        [`set_divisor_basis`][cytools.toricvariety.ToricVariety.set_divisor_basis].

        Args:
            include_origin (bool): Whether the indexing of the returned vector
                (or the columns of the returned matrix) includes the origin,
                i.e. the canonical divisor at index 0.
            as_matrix (bool): Whether to return the basis as a matrix instead of
                a vector of indices. If a generic (matrix) basis was set, then
                it is always returned as a matrix.

        Returns:
            np.ndarray: A vector of indices of the standard basis of the lattice
                dual to the lattice of prime toric divisors that form the basis
                of curves. If `as_matrix` is true, or if a generic (matrix)
                basis has been set, a matrix is returned instead, where each
                row gives the intersection numbers of a basis curve with the
                canonical divisor and the prime toric divisors.

        Raises:
            Exception: If `include_origin=False` but the current basis includes
                the index of the canonical divisor.

        Examples:
            We consider a simple toric variety with two independent curves. If
            no basis has been set, then this function finds one. If a basis has
            been set, then this function returns it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.curve_basis()  # We haven't set any basis
            array([5, 6])
            >>> v.set_curve_basis([5,6])  # Here we set a basis
            >>> v.curve_basis()  # We get the basis we set
            array([5, 6])
            >>> v.curve_basis(as_matrix=True)  # We get the basis in matrix form
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])
        """
        if self._curve_basis is None:
            self.set_divisor_basis(
                self.polytope().glsm_basis(
                    integral=True,
                    include_origin=True,
                    points=self.polytope().points_to_indices(
                        self.triangulation().points()
                    ),
                )
            )
        if len(self._curve_basis.shape) == 1:
            if 0 in self._curve_basis and not include_origin:
                raise Exception(
                    "The basis was requested not including the "
                    "origin, but it is included in the current basis."
                )
            if as_matrix:
                return np.array(
                    self._curve_basis_mat[:, (0 if include_origin else 1) :]
                )
            return np.array(self._curve_basis) - (0 if include_origin else 1)
        return np.array(self._curve_basis[:, (0 if include_origin else 1) :])

    def set_curve_basis(self, basis, include_origin=True):
        """Set a basis of curves of the toric variety.

        This in turn induces the dual basis of divisors. The basis can be
        specified with a vector of indices of the standard basis of the lattice
        dual to the lattice of prime toric divisors, which is equivalent to
        passing the same vector to
        [`set_divisor_basis`][cytools.toricvariety.ToricVariety.set_divisor_basis].
        If the experimental features are enabled, it can also be specified
        with a matrix whose rows give the intersection numbers of each basis
        curve with the canonical divisor and the prime toric divisors. Cached
        results that depend on the choice of basis are cleared.

        Note:
            Only integral bases are supported by CYTools, meaning that all toric
            curves must be able to be written as an integral linear combination
            of the basis curves.

        Args:
            basis (ArrayLike): Vector or matrix specifying a basis. When a
                vector is used, its entries are taken as indices of the standard
                basis of the lattice dual to the lattice of prime toric divisors
                (with index 0 corresponding to the canonical divisor). When a
                matrix is used, its rows are taken as linear combinations of the
                aforementioned elements. The matrix may omit the column of the
                canonical divisor, in which case it is computed as minus the sum
                of the other entries of the row.
            include_origin (bool): Whether to interpret the indices in the
                input vector as including the origin at index 0. This is only
                used when `basis` is a vector.

        Raises:
            ValueError: If the input does not specify a valid integral basis.
            Exception: If a matrix is given and the experimental features are
                not enabled.

        Examples:
            We consider a simple toric variety with two independent curves. We
            first find the default basis of curves it picks and then set a basis
            of our choice.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.curve_basis()  # We haven't set any basis
            array([5, 6])
            >>> v.set_curve_basis([5,6])  # Here we set a basis
            >>> v.curve_basis()  # We get the basis we set
            array([5, 6])
            >>> v.curve_basis(as_matrix=True)  # We get the basis in matrix form
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])

            Note that when setting a curve basis with a vector, this function
            behaves exactly the same as
            [`set_divisor_basis`][cytools.toricvariety.ToricVariety.set_divisor_basis].
            The two functions differ for generic (matrix) bases. An example can
            be found in the [experimental
            features](experimental.md#generic-bases-for-divisors-and-curves)
            section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the CalabiYau class.
        set_curve_basis(self, basis, include_origin=include_origin)

    def mori_cone(
        self,
        in_basis=False,
        include_origin=True,
        from_intersection_numbers=False,
    ):
        """Return the Mori cone of the toric variety.

        By default, the Mori cone is computed as the dual of the secondary cone
        of the triangulation. Without a choice of basis, each ray is given by
        the intersection numbers of a curve with the canonical divisor and the
        prime toric divisors. The results are cached.

        Args:
            in_basis (bool): Whether to use the current basis of curves, which
                is dual to the basis returned by
                [`divisor_basis`][cytools.toricvariety.ToricVariety.divisor_basis].
            include_origin (bool): Whether to include the coordinate
                corresponding to the origin of the polytope, i.e. the canonical
                divisor. This parameter is ignored when `in_basis=True`.
            from_intersection_numbers (bool): Whether to compute the rays of the
                Mori cone using the intersection numbers of the variety. This
                can be faster if they are already computed. The set of rays may
                be different, but they define the same cone. This parameter only
                has an effect the first time the Mori cone is computed, since
                the result is cached.

        Returns:
            Cone: The Mori cone of the toric variety.

        Examples:
            We construct a toric variety and find its Mori cone in an
            $h^{1,1}+d+1$ dimensional lattice (i.e. without a particular choice
            of basis) and in an $h^{1,1}$ dimensional lattice (i.e. after
            picking a basis of curves).

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.mori_cone()  # By default it does not use a basis of curves
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
            >>> v.mori_cone(in_basis=True)  # Uses the dual of the divisor basis
            A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        """
        if self._mori_cone[0] is None:
            if from_intersection_numbers:
                rays = (
                    self._compute_mori_rays_from_intersections_4d()
                    if self.dim() == 4
                    else self._compute_mori_rays_from_intersections()
                )
                self._mori_cone[0] = Cone(rays)
            else:
                self._mori_cone[0] = self.triangulation().secondary_cone().dual()
        # 0: All divs, 1: No origin, 2: In basis
        args_id = ((not include_origin) * 1 if not in_basis else 0) + in_basis * 2
        if self._mori_cone[args_id] is not None:
            return self._mori_cone[args_id]
        rays = self._mori_cone[0].rays()
        basis = self.divisor_basis()
        if include_origin and not in_basis:
            new_rays = rays
        elif not include_origin and not in_basis:
            new_rays = rays[:, 1:]
        else:
            if len(basis.shape) == 2:  # If basis is matrix
                new_rays = rays.dot(basis.T)
            else:
                new_rays = rays[:, basis]
        c = Cone(new_rays, check=len(basis.shape) == 2)
        self._mori_cone[args_id] = c
        return self._mori_cone[args_id]

    def _compute_mori_rays_from_intersections(self):
        """Compute the Mori cone rays of the variety using intersection numbers.

        Each torus-invariant curve, given by the intersection of $d-1$ distinct
        prime toric divisors (where $d$ is the dimension of the variety),
        yields a ray whose entries are the intersection numbers of the curve
        with the prime toric divisors, divided by their greatest common
        divisor. The entry corresponding to the canonical divisor is fixed to
        minus the sum of the other entries.

        Note:
            This function should generally not be called by the user. Instead,
            it is called by
            [`mori_cone`][cytools.toricvariety.ToricVariety.mori_cone] when
            `from_intersection_numbers=True` and the variety is not
            4-dimensional.

        Returns:
            np.ndarray: The generating rays of the Mori cone of the toric
                variety, one per row, including the coordinate corresponding to
                the canonical divisor. The set of rays is not necessarily
                minimal.

        Examples:
            This function is not intended to be directly used, but it is used
            in the following example. We construct a 5-dimensional toric variety
            and compute the Mori cone using its intersection numbers.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-6,-9,-18]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.mori_cone(from_intersection_numbers=True)
            A 5-dimensional rational polyhedral cone in RR^11 generated by 14 rays
        """
        intnums = self.intersection_numbers(in_basis=False)
        dim = self.dim()
        num_divs = self.glsm_charge_matrix().shape[1]
        curve_dict = defaultdict(lambda: [[], []])
        for ii in intnums:
            if 0 in ii:
                continue
            ctr = Counter(ii)
            if len(ctr) < dim - 1:
                continue
            for comb in set(combinations(ctr.keys(), dim - 1)):
                crv = tuple(sorted(comb))
                curve_dict[crv][0].append(
                    int(sum([i * (ctr[i] - (i in crv)) for i in ctr]))
                )
                curve_dict[crv][1].append(intnums[ii])
        row_set = set()
        for crv in curve_dict:
            g = gcd_list(curve_dict[crv][1])
            row = np.zeros(num_divs, dtype=int)
            for j, jj in enumerate(curve_dict[crv][0]):
                row[jj] = int(round(curve_dict[crv][1][j] / g))
            row_set.add(tuple(row))
        mori_rays = np.array(list(row_set), dtype=int)
        # Compute column corresponding to the origin
        mori_rays[:, 0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def _compute_mori_rays_from_intersections_4d(self):
        """Compute the Mori cone rays of a 4D variety using intersection numbers.

        This is an optimized version of `_compute_mori_rays_from_intersections`
        for 4-dimensional toric varieties. Each torus-invariant curve, given by
        the intersection of three distinct prime toric divisors, yields a ray
        whose entries are the intersection numbers of the curve with the prime
        toric divisors, divided by the absolute value of their greatest common
        divisor. The entry corresponding to the canonical divisor is fixed to
        minus the sum of the other entries.

        Note:
            This function should generally not be called by the user. Instead,
            it is called by
            [`mori_cone`][cytools.toricvariety.ToricVariety.mori_cone] when
            `from_intersection_numbers=True` and the variety is 4-dimensional.

        Returns:
            np.ndarray: The generating rays of the Mori cone of the toric
                variety, one per row, including the coordinate corresponding to
                the canonical divisor. The set of rays is not necessarily
                minimal.

        Examples:
            This function is not intended to be directly used, but it is used
            in the following example. We construct a toric variety and compute
            the Mori cone using its intersection numbers.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.mori_cone(from_intersection_numbers=True)
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
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
                if (ii[0], ii[2], ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[2], ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[1], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[1], ii[2], ii[3])], ii[0], intnums[ii]]
                    )
            elif ii[1] == ii[2]:
                if (ii[0], ii[1], ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[1], ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[2], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[0], ii[1], ii[3])], ii[2], intnums[ii]]
                    )
            elif ii[2] == ii[3]:
                if (ii[0], ii[1], ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[1], ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[3], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[0], ii[1], ii[2])], ii[3], intnums[ii]]
                    )
            else:
                if (ii[0], ii[1], ii[2]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[1], ii[2])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[3], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[0], ii[1], ii[2])], ii[3], intnums[ii]]
                    )
                if (ii[0], ii[1], ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[1], ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[2], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[0], ii[1], ii[3])], ii[2], intnums[ii]]
                    )
                if (ii[0], ii[2], ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[0], ii[2], ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[1], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[0], ii[2], ii[3])], ii[1], intnums[ii]]
                    )
                if (ii[1], ii[2], ii[3]) not in curve_dict.keys():
                    curve_dict[(ii[1], ii[2], ii[3])] = curve_ctr
                    curve_sparse_list.append([curve_ctr, ii[0], intnums[ii]])
                    curve_ctr += 1
                else:
                    curve_sparse_list.append(
                        [curve_dict[(ii[1], ii[2], ii[3])], ii[0], intnums[ii]]
                    )
        row_list = [[] for i in range(curve_ctr)]
        # Remove zeros
        for ii in curve_sparse_list:
            if ii[2] != 0:
                row_list[ii[0]].append([ii[1], ii[2]])
        # Normalize
        for row in row_list:
            g = abs(gcd_list([ii[1] for ii in row]))
            for ii in row:
                ii[1] = int(round(ii[1] / g))
        row_list = set(tuple(tuple(ii) for ii in sorted(row)) for row in row_list)
        mori_rays = np.zeros((len(row_list), num_divs), dtype=int)
        for i, row in enumerate(row_list):
            for ii in row:
                mori_rays[i, int(round(ii[0]))] = round(ii[1])
        # Compute column corresponding to the origin
        mori_rays[:, 0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def kahler_cone(self):
        """Return the Kähler cone of the toric variety in the current divisor basis.

        It is computed as the dual of the Mori cone in the current basis of
        curves (see [`mori_cone`][cytools.toricvariety.ToricVariety.mori_cone]).

        Returns:
            Cone: The Kähler cone of the toric variety.

        Examples:
            We construct a toric variety and find its Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.kahler_cone()
            A rational polyhedral cone in RR^2 defined by 3 hyperplanes
        """
        return self.mori_cone(in_basis=True).dual()

    def _construct_intnum_equations_4d(self):
        """Construct the linear system for the intersection numbers of a 4D variety.

        This is an auxiliary function used by
        [`intersection_numbers`][cytools.toricvariety.ToricVariety.intersection_numbers]
        and is optimized for 4-dimensional varieties. The intersection numbers of
        distinct prime toric divisors that span a simplex of the triangulation
        are computed directly from the simplex volumes. The remaining possibly
        nonzero intersection numbers (those with repeated indices) are the
        unknowns of the linear system $M x + C = 0$, which is obtained from the
        linear relations among the divisors.

        Returns:
            tuple[csr_matrix, np.ndarray, list, list]: A tuple containing the
                sparse matrix $M$, the vector $C$, the list of intersection
                numbers of distinct divisors (each given as a list of four
                divisor indices followed by the intersection number), and the
                list of index tuples of the intersection numbers that are used
                as variables of the linear system.

        Examples:
            This function is not intended to be directly used, but it is used
            in the following example. We construct a toric variety and compute
            its intersection numbers.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> intnums = v.intersection_numbers()
        """
        points = self.triangulation().points()
        simps = self.triangulation().simplices(as_indices=True)

        # Origin is at index 0
        pts_ext = np.empty((points.shape[0], points.shape[1] + 1), dtype=int)
        pts_ext[:, :-1] = points
        pts_ext[:, -1] = 1

        linear_relations = self.glsm_linear_relations(include_origin=False)

        # First compute the distinct intersection numbers
        distintnum_array = sorted(
            [
                [c for c in simp if c != 0]
                + [1 / abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in simps
            ]
        )

        frst = [[c for c in s if c != 0] for s in simps]
        simp_2 = {j for f in frst for j in combinations(f, 2)}
        simp_3 = {j for f in frst for j in combinations(f, 3)}

        # We construct and solve the linear system M*x + C = 0, where M is
        # a rectangular mxn matrix and C is a vector.
        ###################################################################
        ### Define dictionaries, to be used to construct the linear system
        ###################################################################
        ## Dictionary of variables
        # Most intersection numbers are trivially zero, find the possibly
        # nonzero intersection numbers.
        variable_array_1 = [
            tuple(j)
            for i in [
                [
                    [s[0], s[0], s[1], s[2]],
                    [s[0], s[1], s[1], s[2]],
                    [s[0], s[1], s[2], s[2]],
                ]
                for s in simp_3
            ]
            for j in i
        ]
        variable_array_2 = [
            tuple(j)
            for i in [
                [
                    [s[0], s[0], s[1], s[1]],
                    [s[0], s[0], s[0], s[1]],
                    [s[0], s[1], s[1], s[1]],
                ]
                for s in simp_2
            ]
            for j in i
        ]
        variable_array_3 = [(i, i, i, i) for i in range(1, len(pts_ext))]
        variable_array = sorted(variable_array_1 + variable_array_2 + variable_array_3)
        variable_dict = {vv: v for v, vv in enumerate(variable_array)}

        ## Dictionary to construct C
        # C is constructed by adding/subtracting distinct intersection
        # numbers.
        c_dict = {s: [] for s in simp_3}
        for d in distintnum_array:
            c_dict[(d[0], d[1], d[2])] += [[d[3], d[4]]]
            c_dict[(d[0], d[1], d[3])] += [[d[2], d[4]]]
            c_dict[(d[0], d[2], d[3])] += [[d[1], d[4]]]
            c_dict[(d[1], d[2], d[3])] += [[d[0], d[4]]]

        ## Dictionary to construct M
        eqn_array_1 = [tuple(s) for s in simp_3]
        eqn_array_2 = [
            tuple(j)
            for i in [[[s[0], s[0], s[1]], [s[0], s[1], s[1]]] for s in simp_2]
            for j in i
        ]
        eqn_array_3 = [(i, i, i) for i in range(1, len(pts_ext))]
        eqn_array = sorted(eqn_array_1 + eqn_array_2 + eqn_array_3)
        eqn_dict = {eq: [] for eq in eqn_array}
        for v in variable_array:
            if v[0] == v[3]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
            elif v[0] == v[2]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[0], v[1], v[3])] += [[v[2], variable_dict[v]]]
            elif v[0] == v[1] and v[2] == v[3]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[0], v[2], v[3])] += [[v[1], variable_dict[v]]]
            elif v[0] == v[1]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[0], v[1], v[3])] += [[v[2], variable_dict[v]]]
                eqn_dict[(v[0], v[2], v[3])] += [[v[1], variable_dict[v]]]
            elif v[1] == v[3]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[1], v[2], v[3])] += [[v[0], variable_dict[v]]]
            elif v[1] == v[2]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[0], v[1], v[3])] += [[v[2], variable_dict[v]]]
                eqn_dict[(v[1], v[2], v[3])] += [[v[0], variable_dict[v]]]
            elif v[2] == v[3]:
                eqn_dict[(v[0], v[1], v[2])] += [[v[3], variable_dict[v]]]
                eqn_dict[(v[0], v[2], v[3])] += [[v[1], variable_dict[v]]]
                eqn_dict[(v[1], v[2], v[3])] += [[v[0], variable_dict[v]]]
            else:
                raise RuntimeError("Failed to construct linear system.")

        # Construct Linear System
        num_rows = len(linear_relations) * len(eqn_array)
        C = np.array([0.0] * num_rows)
        M_row = []
        M_col = []
        M_val = []
        row_ctr = 0
        for eqn in eqn_array:
            for lin in linear_relations:
                if eqn[0] != eqn[1] and eqn[1] != eqn[2]:
                    c_temp = c_dict[eqn]
                    C[row_ctr] = sum([lin[cc[0] - 1] * cc[1] for cc in c_temp])
                eqn_temp = eqn_dict[eqn]
                for e in eqn_temp:
                    M_row.append(row_ctr)
                    M_col.append(e[1])
                    M_val.append(lin[e[0] - 1])
                row_ctr += 1
        Mat = csr_matrix((M_val, (M_row, M_col)), dtype=np.float64)
        return Mat, C, distintnum_array, variable_array

    def _construct_intnum_equations(self):
        """Construct the linear system for the intersection numbers of the variety.

        This is an auxiliary function used by
        [`intersection_numbers`][cytools.toricvariety.ToricVariety.intersection_numbers]
        for varieties of any dimension (a more optimized version is used for
        4-dimensional varieties). The intersection numbers of distinct prime
        toric divisors that span a simplex of the triangulation are computed
        directly from the simplex volumes. The remaining possibly nonzero
        intersection numbers (those with repeated indices) are the unknowns of
        the linear system $M x + C = 0$, which is obtained from the linear
        relations among the divisors.

        Returns:
            tuple[csr_matrix, np.ndarray, list, list]: A tuple containing the
                sparse matrix $M$, the vector $C$, the list of intersection
                numbers of distinct divisors (each given as a list of $d$
                divisor indices followed by the intersection number), and the
                list of index tuples of the intersection numbers that are used
                as variables of the linear system.

        Examples:
            This function is not intended to be directly used, but it is used
            in the following example. We construct a 5-dimensional toric
            variety and compute its intersection numbers.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-6,-9,-18]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> intnums = v.intersection_numbers()
        """
        dim = self.dim()
        pts_ext = np.empty((self.triangulation().points().shape[0], dim + 1), dtype=int)
        pts_ext[:, :-1] = self.triangulation().points()
        pts_ext[:, -1] = 1
        linear_relations = self.glsm_linear_relations(include_origin=False)
        # First compute the distinct intersection numbers
        distintnum_array = sorted(
            [
                [c for c in simp if c != 0]
                + [1 / abs(np.linalg.det([pts_ext[p] for p in simp]))]
                for simp in self.triangulation().simplices(as_indices=True)
            ]
        )
        frst = [
            [c for c in s if c != 0]
            for s in self.triangulation().simplices(as_indices=True)
        ]
        simp_n = [{j for f in frst for j in combinations(f, n)} for n in range(2, dim)]
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
        for n in range(2, dim):
            comb = list(combinations(range(dim - 1), dim - n))
            choices = np.empty((len(comb), dim), dtype=int)
            choices[:, 0] = 0
            for k, c in enumerate(comb):
                for i in range(1, dim):
                    choices[k, i] = choices[k, i - 1] + (0 if i - 1 in c else 1)
            choices_n.append(choices)
        variable_array_1 = [(i,) * dim for i in range(1, len(pts_ext))]
        variable_array_n = [
            tuple(s[ch])
            for n in range(len(simp_n))
            for s in simp_n[n]
            for ch in choices_n[n]
        ]
        variable_array = variable_array_1 + variable_array_n
        variable_dict = {vv: v for v, vv in enumerate(variable_array)}
        ## Dictionary to construct C
        # C is constructed by adding/subtracting distinct intersection
        # numbers.
        c_dict = defaultdict(lambda: [])
        for d in distintnum_array:
            for i in range(len(d) - 1):
                c_dict[tuple(c for j, c in enumerate(d[:-1]) if j != i)] += [
                    (d[i], d[-1])
                ]
        ## Dictionary to construct M
        eqn_array_1 = [tuple(s) for s in simp_n[-1]]
        eqn_array_2 = [(i,) * (dim - 1) for i in range(1, len(pts_ext))]
        choices_n = []
        for n in range(2, dim - 1):
            comb = list(combinations(range(dim - 2), dim - 1 - n))
            choices = np.empty((len(comb), dim - 1), dtype=int)
            choices[:, 0] = 0
            for k, c in enumerate(comb):
                for i in range(1, dim - 1):
                    choices[k, i] = choices[k, i - 1] + (0 if i - 1 in c else 1)
            choices_n.append(choices)
        eqn_array_n = [
            tuple(s[ch])
            for n in range(len(choices_n))
            for s in simp_n[n]
            for ch in choices_n[n]
        ]
        eqn_array = eqn_array_1 + eqn_array_2 + eqn_array_n
        eqn_dict = defaultdict(lambda: [])
        for v in variable_array:
            for c in set(combinations(v, dim - 1)):
                k = None
                for i in range(dim):
                    if i == dim - 1 or v[i] != c[i]:
                        k = i
                        break
                eqn_dict[c] += [(v[k], variable_dict[v])]
        # Construct Linear System
        num_rows = len(linear_relations) * len(eqn_array)
        C = np.zeros(num_rows, dtype=float)
        M_row = []
        M_col = []
        M_val = []
        row_ctr = 0
        for eqn in eqn_array:
            for lin in linear_relations:
                if len(set(eqn)) == dim - 1:
                    c_temp = c_dict[eqn]
                    C[row_ctr] = sum([lin[cc[0] - 1] * cc[1] for cc in c_temp])
                eqn_temp = eqn_dict[eqn]
                for e in eqn_temp:
                    M_row.append(row_ctr)
                    M_col.append(e[1])
                    M_val.append(lin[e[0] - 1])
                row_ctr += 1
        Mat = csr_matrix((M_val, (M_row, M_col)), dtype=np.float64)
        return Mat, C, distintnum_array, variable_array

    def intersection_numbers(
        self,
        in_basis=False,
        format="dok",
        zero_as_anticanonical=False,
        backend="all",
        check=True,
        backend_error_tol=1e-3,
        round_to_zero_threshold=1e-3,
        round_to_integer_error_tol=5e-2,
        verbose=0,
        exact_arithmetic=False,
    ):
        """Return the intersection numbers of the toric variety.

        The intersection numbers of distinct prime toric divisors spanning a
        simplex of the triangulation are computed from the simplex volumes, and
        the rest are found by solving a sparse linear system obtained from the
        linear relations among the divisors. Intersection numbers involving the
        canonical divisor (index 0) are then computed from these. If the
        variety is smooth, all intersection numbers are rounded to integers,
        and if its canonical divisor is smooth (see
        `canonical_divisor_is_smooth`), the ones involving the canonical divisor
        exactly once are rounded to integers. Otherwise, they are floating-point
        numbers.

        Only the nonzero intersection numbers are stored, and the results are
        cached for each combination of `in_basis`, `format`,
        `zero_as_anticanonical`, and `exact_arithmetic`. The remaining
        parameters only affect the first computation.

        Tip:
            The intersection numbers are computed as floating-point numbers by
            default, but there is the experimental option to turn them into
            exact rationals with `exact_arithmetic=True`. The process is fairly
            quick, but it is unreliable at large $h^{1,1}$. Furthermore,
            verifying that they are correct becomes very slow at large
            $h^{1,1}$.

        Args:
            in_basis (bool): Whether to return the intersection numbers in the
                current basis of divisors (see
                [`divisor_basis`][cytools.toricvariety.ToricVariety.divisor_basis]).
                In this case, the indices refer to the basis elements.
            format (str): The output format of the intersection numbers. The
                options are `"dok"`, `"coo"`, and `"dense"`. When set to `"dok"`
                (Dictionary Of Keys), a dictionary is returned where the keys
                are tuples of divisor indices in ascending order and the values
                are the corresponding intersection numbers. When set to `"coo"`
                (COOrdinate format), a numpy array is returned with rows of the
                form `[a, b, ..., c, K_ab...c]`, i.e. all but the last entry of
                each row are divisor indices in ascending order, and the last
                entry is their intersection number. When set to `"dense"`, the
                full dense array of intersection numbers is returned.
            zero_as_anticanonical (bool): Whether to treat the zeroth index as
                corresponding to the anticanonical divisor instead of the
                canonical divisor. This is ignored when `in_basis=True`.
            backend (str): The sparse linear solver to use. Options are
                `"all"`, `"sksparse"`, and `"scipy"`. When set to `"all"`, every
                solver is tried in order until one succeeds.
            check (bool): Whether to explicitly check the solution to the linear
                system.
            backend_error_tol (float): Error tolerance for the solution of the
                linear system.
            round_to_zero_threshold (float): Intersection numbers with magnitude
                smaller than this threshold are rounded to zero (i.e. dropped).
            round_to_integer_error_tol (float): Maximum allowed distance from an
                integer for intersection numbers that are expected to be
                integral, i.e. all of them when the variety is smooth, and those
                involving the canonical divisor exactly once when the canonical
                divisor is smooth.
            verbose (int): The verbosity level. If 0, nothing is printed. If 1
                or higher, warnings from the linear backends are printed.
            exact_arithmetic (bool): Whether to convert the intersection
                numbers into exact rational numbers (`flint.fmpq`). This
                requires the experimental features to be enabled.

        Returns:
            dict[tuple[int, ...], float] | np.ndarray: The intersection numbers
                in the requested format: a dictionary for `"dok"`, a 2D array
                for `"coo"`, or a dense array of rank equal to the dimension of
                the variety for `"dense"`. A shallow copy of the cached result
                is returned.

        Raises:
            ValueError: If `format` or `backend` is not a valid option, or if
                `exact_arithmetic=True` and the experimental features are not
                enabled.
            RuntimeError: If the linear system could not be solved, if the
                conversion to rational numbers failed, or if non-integer
                intersection numbers are found where integers are expected.

        Examples:
            We construct a toric variety and compute its intersection numbers.
            By default, the intersection numbers of the canonical divisor (index
            0) and the prime toric divisors are returned as a dictionary whose
            keys are the divisor indices in ascending order.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> intnums = v.intersection_numbers()
            >>> print(intnums[(1, 2, 3, 4)])
            1.0
            >>> print(intnums[(1, 2, 4, 6)])
            0.5
            >>> print(intnums[(0, 1, 2, 3)])
            -18

            We can also compute the intersection numbers in the current basis
            of divisors. In this case, the indices refer to the basis elements,
            so the key `(0, 0, 1, 1)` corresponds to the intersection of the
            divisors 5, 5, 6, and 6.

            >>> v.divisor_basis()
            array([5, 6])
            >>> intnums_basis = v.intersection_numbers(in_basis=True)
            >>> print(round(intnums_basis[(0, 0, 1, 1)], 6))
            0.166667

            The intersection numbers can also be obtained in the `"coo"` and
            `"dense"` formats.

            >>> print(v.intersection_numbers(in_basis=True, format="coo"))
            [[ 0.          0.          1.          1.          0.16666667]
             [ 0.          1.          1.          1.         -1.        ]
             [ 1.          1.          1.          1.          4.5       ]]
            >>> v.intersection_numbers(in_basis=True, format="dense").shape
            (2, 2, 2, 2)
        """
        if format not in ("dok", "coo", "dense"):
            raise ValueError('Options for format are "dok", "coo", "dense".')
        if in_basis:
            zero_as_anticanonical = False
        args_id = (zero_as_anticanonical, in_basis, exact_arithmetic, format)
        if args_id in self._intersection_numbers:
            return copy.copy(self._intersection_numbers[args_id])
        if (False, False, False, "dok") not in self._intersection_numbers or (
            (False, False, True, "dok") not in self._intersection_numbers
            and exact_arithmetic
        ):
            backends = ["all", "sksparse", "scipy"]
            if backend not in backends:
                raise ValueError(
                    f"Invalid linear system backend. The options are: {backends}."
                )
            if exact_arithmetic and not config._exp_features_enabled:
                raise ValueError(
                    "The experimental features must be enabled to use exact arithmetic."
                )
            # Construct the linear equations
            # Note that self.dim gives the dimension of the CY not the of the
            # variety
            Mat, C, distintnum_array, variable_array = (
                self._construct_intnum_equations_4d()
                if self.dim() == 4
                else self._construct_intnum_equations()
            )
            # The system to be solved is Mat*x + C = 0. This is an
            # overdetermined but consistent linear system.
            # There is a unique solution to this system. We solve it by
            # defining MM = Mat.transpose()*Mat and CC = - Mat.transpose()*C,
            # and solve
            # MM*x = CC
            # Since MM is a positive definite full rank matrix, this system can
            # be solved using via a Cholesky decomposition.
            solution = solve_linear_system(
                Mat,
                C,
                backend=backend,
                check=check,
                backend_error_tol=backend_error_tol,
                verbosity=verbose,
            )
            if solution is None:
                raise RuntimeError("Linear system solution failed.")
            if exact_arithmetic:
                solution_fmpq = flint.fmpq_mat(
                    [array_float_to_fmpq(solution).tolist()]
                ).transpose()
                if check:
                    Mat_fmpq = flint.fmpq_mat(Mat.shape[0], Mat.shape[1])
                    Mat_dok = Mat.todok()
                    for k in Mat_dok.keys():
                        Mat_fmpq[k] = float_to_fmpq(Mat_dok[k])
                    C_fmpq = flint.fmpq_mat(
                        [array_float_to_fmpq(C).tolist()]
                    ).transpose()
                    res = Mat_fmpq * solution_fmpq + C_fmpq
                    if any(np.array(res.tolist()).flat):
                        raise RuntimeError("Failed to convert to rational numbers.")
            intnums = dict()
            if exact_arithmetic:
                for ii in distintnum_array:
                    intnums[tuple(int(round(j)) for j in ii[:-1])] = float_to_fmpq(
                        ii[-1]
                    )
                for i, ii in enumerate(variable_array):
                    if abs(solution[i]) < round_to_zero_threshold:
                        continue
                    intnums[tuple(ii)] = float_to_fmpq(solution[i])
            else:
                for ii in distintnum_array:
                    intnums[tuple(int(round(j)) for j in ii[:-1])] = ii[-1]
                for i, ii in enumerate(variable_array):
                    if abs(solution[i]) < round_to_zero_threshold:
                        continue
                    intnums[tuple(ii)] = solution[i]
            if self.is_smooth():
                if exact_arithmetic:
                    for ii in intnums:
                        c = intnums[ii]
                        if c.q != 1:
                            raise RuntimeError(
                                "Non-integer intersection numbers "
                                "detected in a smooth toric variety."
                            )
                else:
                    for ii in intnums:
                        c = intnums[ii]
                        if abs(round(c) - c) > round_to_integer_error_tol:
                            raise RuntimeError(
                                "Non-integer intersection numbers "
                                "detected in a smooth toric variety."
                            )
                        intnums[ii] = int(round(c))
            # Add intersections with canonical divisor
            # First we only compute intersection numbers with a single index 0
            # This is because precision errors add up significantly for
            # intersection numbers with self-intersections of the canonical
            # divisor
            dim = self.dim()
            canon_intnum = defaultdict(lambda: 0)
            for ii in intnums:
                choices = set(
                    tuple(c for i, c in enumerate(ii) if i != j) for j in range(dim)
                )
                for c in choices:
                    canon_intnum[(0,) + c] -= intnums[ii]
            # Now we round all intersection numbers of the form K_0i...j to
            # integers if the CY is smooth. Otherwise, we only remove the zero
            # elements
            if self.canonical_divisor_is_smooth() and exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    val = canon_intnum[ii]
                    if val.q != 1:
                        raise RuntimeError(
                            f"Non-integer intersection numbers "
                            f"detected in a smooth CY. {ii}:{val}"
                        )
                    if val != 0:
                        canon_intnum[ii] = val
                    else:
                        canon_intnum.pop(ii)
            elif self.canonical_divisor_is_smooth() and not exact_arithmetic:
                for ii in list(canon_intnum.keys()):
                    val = canon_intnum[ii]
                    round_val = int(round(val))
                    if abs(val - round_val) > round_to_integer_error_tol:
                        raise RuntimeError(
                            f"Non-integer intersection numbers "
                            f"detected in a smooth CY. {ii}:{val}"
                        )
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
            for n in range(2, dim + 1):
                tmp_intnum = defaultdict(lambda: 0)
                for ii, ii_val in canon_intnum_n[-1].items():
                    choices = set(
                        tuple(c for i, c in enumerate(ii[n - 1 :]) if i != j)
                        for j in range(dim + 1 - n)
                    )
                    for c in choices:
                        tmp_intnum[(0,) * n + c] -= ii_val
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
                self._intersection_numbers[(False, False, True, "dok")] = intnums
                self._intersection_numbers[(False, False, False, "dok")] = {
                    ii: (
                        int(intnums[ii].p)
                        if intnums[ii].q == 1
                        else fmpq_to_float(intnums[ii])
                    )
                    for ii in intnums
                }
            else:
                self._intersection_numbers[(False, False, False, "dok")] = intnums
        # Now intersection numbers have been computed
        # We now compute the intersection numbers of the basis if necessary
        if zero_as_anticanonical and not in_basis:
            base_intnums = self._intersection_numbers[
                (False, False, exact_arithmetic, "dok")
            ]
            # NOTE: this must be stored under the "dok" key rather than under
            # args_id. The format conversion below looks the value up as
            # (zero_as_anticanonical, in_basis, exact_arithmetic, "dok"), so
            # storing it under a format-carrying args_id raises a KeyError when
            # a non-default format is requested first.
            self._intersection_numbers[(True, False, exact_arithmetic, "dok")] = {
                ii: (
                    val * (-1 if sum(jj == 0 for jj in ii) % 2 == 1 else 1)
                    if 0 in ii
                    else val
                )
                for ii, val in base_intnums.items()
            }
        elif in_basis:
            basis = self.divisor_basis()
            if len(basis.shape) == 2:  # If basis is matrix
                self._intersection_numbers[(False, True, exact_arithmetic, "dense")] = (
                    symmetric_sparse_to_dense(
                        self._intersection_numbers[
                            (False, False, exact_arithmetic, "dok")
                        ],
                        basis,
                    )
                )
                self._intersection_numbers[(False, True, exact_arithmetic, "dok")] = (
                    symmetric_dense_to_sparse(
                        self._intersection_numbers[
                            (False, True, exact_arithmetic, "dense")
                        ]
                    )
                )
            else:
                self._intersection_numbers[(False, True, exact_arithmetic, "dok")] = (
                    filter_tensor_indices(
                        self._intersection_numbers[
                            (False, False, exact_arithmetic, "dok")
                        ],
                        basis,
                    )
                )
        # Intersection numbers of the basis are now done
        # Finally, we convert into the desired format
        if format == "coo":
            tmpintnums = self._intersection_numbers[
                (zero_as_anticanonical, in_basis, exact_arithmetic, "dok")
            ]
            self._intersection_numbers[args_id] = np.array(
                [list(ii) + [tmpintnums[ii]] for ii in tmpintnums]
            )
        elif format == "dense":
            self._intersection_numbers[args_id] = symmetric_sparse_to_dense(
                self._intersection_numbers[
                    (zero_as_anticanonical, in_basis, exact_arithmetic, "dok")
                ]
            )
        return copy.copy(self._intersection_numbers[args_id])

    def prime_toric_divisors(self):
        """Return the point indices corresponding to the prime toric divisors.

        These are the indices (with respect to the points of the polytope) of
        the points used in the triangulation, excluding the origin at index 0.
        The result is cached.

        Returns:
            tuple[int, ...]: The point indices corresponding to prime toric
                divisors.

        Examples:
            We construct a toric variety and find the list of prime toric
            divisors.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.prime_toric_divisors()
            (1, 2, 3, 4, 5, 6)
        """
        if self._prime_divs is None:
            tri_ind = list(
                set.union(
                    *[set(s) for s in self.triangulation().simplices(as_indices=True)]
                )
            )
            divs = self.triangulation().triangulation_to_polytope_indices(tri_ind)
            self._prime_divs = tuple(i for i in divs if i)
        return self._prime_divs

    def is_smooth(self):
        """Return whether the toric variety is smooth.

        The variety is smooth if every simplex of the triangulation (i.e. every
        maximal cone of the fan) is unimodular. The result is cached.

        Returns:
            bool: Whether the toric variety is smooth.

        Examples:
            We construct two toric varieties and check if they are smooth.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t1 = p.triangulate()
            >>> v1 = t1.get_toric_variety()
            >>> v1.is_smooth()
            False
            >>> t2 = p.triangulate(include_points_interior_to_facets=True)
            >>> v2 = t2.get_toric_variety()
            >>> v2.is_smooth()
            True
        """
        if self._is_smooth is not None:
            return self._is_smooth
        pts = self.triangulation().points()
        pts = np.insert(pts, 0, np.ones(len(pts), dtype=int), axis=1)
        simp = self.triangulation().simplices(as_indices=True)
        self._is_smooth = all(abs(int(round(np.linalg.det(pts[s])))) == 1 for s in simp)
        return self._is_smooth

    def canonical_divisor_is_smooth(self):
        """Return whether the canonical divisor is smooth.

        More precisely, this checks whether a generic anticanonical
        hypersurface (i.e. the Calabi-Yau hypersurface) is smooth. This is
        done by verifying that all lattice points of the polytope not interior
        to facets are used in the triangulation and, for varieties of
        dimension greater than 4, that all $(d-1)$-dimensional cones of the fan
        lying in $(d-2)$-dimensional faces of the polytope are smooth, where
        $d$ is the dimension of the variety. The result is cached.

        Returns:
            bool: Whether the canonical divisor is smooth.

        Examples:
            We construct a toric variety and check if its canonical divisor is
            smooth.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.canonical_divisor_is_smooth()
            True
        """
        if self._canon_div_is_smooth is not None:
            return self._canon_div_is_smooth
        pts_mpcp = {tuple(pt) for pt in self.polytope().points_not_interior_to_facets()}
        ind_triang = list(
            set.union(*[set(s) for s in self._triang.simplices(as_indices=True)])
        )
        pts_triang = {tuple(pt) for pt in self._triang.points()[ind_triang]}
        sm = pts_mpcp.issubset(pts_triang) and (
            True
            if self.dim() <= 4
            else all(
                c.is_smooth() for c in self.fan_cones(self.dim() - 1, self.dim() - 2)
            )
        )
        self._canon_div_is_smooth = sm
        return self._canon_div_is_smooth

    def effective_cone(self):
        """Return the cone of effective divisors of the toric variety.

        The effective cone is generated by the classes of the prime toric
        divisors, expressed in the current basis of divisors. The result is
        cached.

        Returns:
            Cone: The effective cone of the toric variety.

        Examples:
            We construct a toric variety and find its effective cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.effective_cone()
            A 2-dimensional rational polyhedral cone in RR^2 generated by 6 rays
        """
        if self._eff_cone is not None:
            return self._eff_cone
        self._eff_cone = Cone(self.curve_basis(include_origin=False, as_matrix=True).T)
        return self._eff_cone

    def fan_cones(self, d=None, face_dim=None):
        """Return the cones of the fan defined by the star triangulation.

        The dimension of the desired cones can be specified, and one can also
        restrict to cones that lie in faces of the polytope of a particular
        dimension. The cones are generated by the points of the triangulation
        (the origin is never a generator), and the results are cached.

        Args:
            d (int | None): The dimension of the desired cones. If not
                specified, it defaults to `face_dim` if that is given, and
                otherwise to the dimension of the variety (i.e. the maximal
                cones are returned).
            face_dim (int | None): Restricts to cones whose generators lie in a
                common face of the polytope of this dimension. If not
                specified, then no restriction is imposed.

        Returns:
            tuple[Cone, ...]: The cones of the fan with the specified
                properties, in no particular order.

        Raises:
            ValueError: If `d` is not between 1 and the dimension of the
                variety.

        Examples:
            We construct a toric variety and find the maximal and 2-dimensional
            cones of the defining fan.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> max_cones = v.fan_cones()  # By default it returns the maximal cones
            >>> len(max_cones)
            9
            >>> cones_2d = v.fan_cones(d=2)  # We can select cones of a given dimension
            >>> len(cones_2d)
            15
        """
        if d is None:
            d = self.dim() if face_dim is None else face_dim
        if d not in range(1, self.dim() + 1):
            raise ValueError("Only cones of dimension 1 through d are supported.")
        if (d, face_dim) in self._fan_cones:
            return self._fan_cones[(d, face_dim)]
        pts = self.triangulation().points()
        cones = set()
        triang_pts_tup = [tuple(pt) for pt in self.triangulation().points()]
        faces = (
            [
                self.triangulation().points_to_indices(
                    [tuple(pt) for pt in f.points() if tuple(pt) in triang_pts_tup]
                )
                for f in self.triangulation()._poly.faces(face_dim)
            ]
            if face_dim is not None
            else None
        )
        for s in self.triangulation().simplices(as_indices=True):
            for c in combinations(s, d):
                if 0 not in c and (
                    faces is None or any(all(cc in f for cc in c) for f in faces)
                ):
                    cones.add(tuple(sorted(c)))
        self._fan_cones[(d, face_dim)] = tuple(Cone(pts[list(c)]) for c in cones)
        return self._fan_cones[(d, face_dim)]

    def get_cy(self, nef_partition=None):
        """Return the Calabi-Yau hypersurface or complete intersection.

        By default, this returns a [`CalabiYau`][cytools.calabiyau.CalabiYau]
        object corresponding to the anticanonical hypersurface in the toric
        variety, which requires the triangulation to be fine. If a
        nef-partition is specified, then it returns the complete intersection
        Calabi-Yau that it defines. The result is cached as long as the same
        nef-partition is used.

        Note:
            Only Calabi-Yau 3-fold hypersurfaces are fully supported. Other
            dimensions, non-favorable polytopes, and CICYs require enabling the
            experimental features of CYTools. See
            [experimental features](experimental.md) for more details.

        Args:
            nef_partition (list[tuple[int, ...]] | None): A list of tuples of
                point indices specifying a nef-partition of the polytope, which
                defines a complete intersection Calabi-Yau. If not specified,
                the anticanonical hypersurface is returned.

        Returns:
            CalabiYau: The Calabi-Yau arising from the triangulation.

        Raises:
            ValueError: If the triangulation is not fine, if it uses neither
                exactly the points not interior to facets nor all the points
                of the polytope, or if the nef-partition is invalid.
            Exception: If the experimental features are required but not
                enabled.

        Examples:
            We construct a toric variety and obtain its Calabi-Yau hypersurface.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> v.get_cy()
            A Calabi-Yau 3-fold hypersurface with h11=2 and h21=272 in a ...
        """
        # First check if the nef partition is the same as the cached one.
        # If not, it resets the cached CY.
        if nef_partition != self._nef_part:
            self._cy = None
        if self._cy is not None:
            return self._cy
        if nef_partition is not None:
            if not config._exp_features_enabled:
                raise Exception(
                    "The experimental features must be enabled to construct CICYs."
                )
            self._cy = CalabiYau(self, nef_partition)
            self._nef_part = nef_partition
        else:
            if not self.triangulation().is_fine():
                raise ValueError("Triangulation is non-fine.")
            if not config._exp_features_enabled:
                if self.dim() != 4:
                    raise Exception(
                        "The experimental features must be enabled to "
                        "construct CYs with dimension other than 3... "
                        f"observed dimension = {self.dim()}"
                    )
                if not self.triangulation().polytope().is_favorable(lattice="N"):
                    raise Exception(
                        "The experimental features must be enabled to "
                        "construct non-favorable CYs..."
                    )

            # check that we have sensical points
            triang = self.triangulation()
            poly = triang.polytope()

            if sorted(triang.labels) == sorted(poly.labels_not_facet):
                pass
            elif sorted(triang.labels) == sorted(poly.labels):
                pass
            else:
                error_msg = "Calabi-Yau hypersurfaces must be constructed either from points not interior to facets or using all points.\n"
                error_msg += f"Triangulation points = {triang.points().tolist()} (labels = {triang.labels})\n"
                error_msg += f"Polytope points = {poly.points().tolist()} (labels = {poly.labels})\n"
                raise ValueError(error_msg)

            self._cy = CalabiYau(self)
            self._nef_part = nef_partition

        return self._cy
