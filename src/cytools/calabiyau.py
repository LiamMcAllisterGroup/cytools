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
# Description:  This module contains tools designed for Calabi-Yau hypersurface
#               computations.
# -----------------------------------------------------------------------------

"""Calabi-Yau hypersurfaces and complete intersections in toric varieties.

This module provides the [`CalabiYau`][cytools.calabiyau.CalabiYau] class, which
handles computations such as Hodge numbers, intersection numbers, the second
Chern class, toric Mori and Kähler cones, volumes, and Gopakumar-Vafa and
Gromov-Witten invariants. It also provides the
[`Invariants`][cytools.calabiyau.Invariants] class, a container for the
computed invariants.
"""

# 'standard' imports
from collections import defaultdict
import copy
import warnings

# 3rd party imports
import cygv
import numpy as np

# CYTools imports
from cytools import config
from cytools.cone import Cone
from cytools.utils import (
    filter_tensor_indices,
    symmetric_sparse_to_dense,
    symmetric_dense_to_sparse,
    set_divisor_basis,
    set_curve_basis,
)

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from numpy.typing import ArrayLike



class CalabiYau:
    r"""A Calabi-Yau manifold in a toric variety.

    This class handles various computations relating to the Calabi-Yau manifold
    itself. It can be used to compute intersection numbers and the toric Mori
    and Kähler cones, among other things. The Calabi-Yau is either the
    anticanonical hypersurface of its ambient toric variety or, if a
    nef partition is specified, the corresponding complete intersection.

    Warning:
        Generally, objects of this class should not be constructed directly by
        the end user. Instead, they should be created with the
        [`get_cy`][cytools.toricvariety.ToricVariety.get_cy] method of
        [`ToricVariety`][cytools.toricvariety.ToricVariety] or the
        [`get_cy`][cytools.triangulation.Triangulation.get_cy] method of
        [`Triangulation`][cytools.triangulation.Triangulation].

    Tip:
        This package is focused on computations on Calabi-Yau 3-fold
        hypersurfaces, but there is experimental support for Calabi-Yaus of
        other dimensions, for non-favorable ones, and for complete
        intersections. See [experimental features](experimental.md) for more
        details.

    Args:
        toric_var (ToricVariety): The ambient toric variety of the Calabi-Yau.
            For hypersurfaces, its defining triangulation must be fine and use
            either all the points of the polytope not interior to facets, or
            all of its points.
        nef_partition (list[tuple[int, ...]] | None): A nef partition of the
            polytope, given as a list of tuples of indices of points of the
            polytope, which defines a complete intersection Calabi-Yau. If not
            specified, the anticanonical hypersurface is constructed.

    Raises:
        ValueError: If `nef_partition` does not define a nef partition, if the
            triangulation is not fine (for hypersurfaces), or if the
            triangulation does not use a valid set of points (for
            hypersurfaces).
        Exception: If the experimental features are not enabled and a nef
            partition is given, the Calabi-Yau is not a 3-fold, or the polytope
            is not favorable.

    Examples:
        We construct a Calabi-Yau from a fine, regular, star triangulation of a
        polytope. Since this class is not intended to be initialized by the end
        user, we create it with the
        [`get_cy`][cytools.triangulation.Triangulation.get_cy] method of
        [`Triangulation`][cytools.triangulation.Triangulation]. In this example
        we obtain the quintic hypersurface in $\mathbb{P}^4$.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        >>> t = p.triangulate()
        >>> t.get_cy()
        A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 ...
    """

    def __init__(self, toric_var, nef_partition=None):
        # We first make sure that the input triangulation is appropriate.
        # Regularity is not checked since it is generally slow.
        triang = toric_var.triangulation()

        if nef_partition is not None:
            if not config._exp_features_enabled:
                raise Exception(
                    "The experimental features must be enabled to " "construct CICYs."
                )
            # Verify that the input defines a nef-partition
            from cytools import Polytope

            pts = toric_var.polytope().points()
            convpoly = Polytope(
                pts[list(set.union(*[set(ii) for ii in nef_partition]))]
            )
            if convpoly != toric_var.polytope():
                raise ValueError("Input data does not define a nef partition")
            polys = [Polytope(pts[[0] + list(ii)]) for ii in nef_partition]
            sumpoly = polys[0]
            for i in range(1, len(polys)):
                sumpoly = sumpoly.minkowski_sum(polys[i])
            if not sumpoly.is_reflexive():
                raise ValueError("Input data does not define a nef partition")

            triangpts = [tuple(pt) for pt in triang.points()]
            parts = [
                tuple(
                    triang.points_to_indices(pt)
                    for pt in pp.points()
                    if any(pt) and tuple(pt) in triangpts
                )
                for pp in polys
            ]
            # NOTE: this is stored as a tuple (of tuples) so that the CY
            # remains hashable
            self._nef_part = tuple(parts)
        else:
            self._nef_part = None
            if not triang.is_fine():
                raise ValueError("Triangulation is non-fine.")
            if (
                toric_var.dim() != 4 or not triang.polytope().is_favorable(lattice="N")
            ) and not config._exp_features_enabled:
                raise Exception(
                    "The experimental features must be enabled to "
                    "construct non-favorable CYs or CYs with "
                    "dimension other than 3."
                )

            # check that we have sensical points
            poly = triang.polytope()

            if sorted(triang.labels) == sorted(poly.labels_not_facet):
                pass
            elif sorted(triang.labels) == sorted(poly.labels):
                pass
            else:
                error_msg = "Calabi-Yau hypersurfaces must be constructed either from points not interior to facets or using all points.\n"
                error_msg += f"Triangulation points = {self.triangulation().points().tolist()} (labels = {self.triangulation().labels})\n"
                error_msg += f"Polytope points = {self.triangulation().polytope().points().tolist()} (labels = {self.triangulation().polytope().labels})\n"
                raise ValueError(error_msg)

        self._ambient_var = toric_var
        self._optimal_ambient_var = None
        self._fan = None
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
        self._mori_cone = [None] * 3
        self._intersection_numbers = dict()
        self._basis_intnums = None
        self._prime_divs = None
        self._second_chern_class = None
        self._is_smooth = None
        self._eff_cone = None
        if not self._is_hypersurface:
            self._compute_cicy_hodge_numbers(only_from_cache=True)

    def clear_cache(self, recursive=False, only_in_basis=False):
        """Clear the cached results of any previous computation.

        Note that clearing the full cache also resets the bases of divisors and
        curves.

        Args:
            recursive (bool): Whether to also clear the cache of the ambient toric
                variety, defining triangulation, and polytope. This is ignored when
                `only_in_basis=True`.
            only_in_basis (bool): Whether to only clear the cache of computations
                that depend on the choice of basis.

        Examples:
            We construct a Calabi-Yau hypersurface, compute its toric Mori cone,
            clear the cache, and then compute it again.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.toric_mori_cone()
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
            >>> cy.clear_cache()  # Clears the cached result
            >>> cy.toric_mori_cone()  # The Mori cone is recomputed
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
        """
        if only_in_basis:
            self._mori_cone[2] = None
            self._eff_cone = None
            self._basis_intnums = None
            for k in list(self._intersection_numbers.keys()):
                if k[1]:
                    self._intersection_numbers.pop(k)
        else:
            self._dim = None
            self._hash = None
            self._glsm_charge_matrix = None
            self._glsm_linrels = None
            self._divisor_basis = None
            self._divisor_basis_mat = None
            self._curve_basis = None
            self._curve_basis_mat = None
            self._mori_cone = [None] * 3
            self._intersection_numbers = dict()
            self._basis_intnums = None
            self._prime_divs = None
            self._second_chern_class = None
            self._is_smooth = None
            self._hodge_nums = None
            self._eff_cone = None
            self._optimal_ambient_var = None
            self._fan = None
            if recursive:
                self.ambient_variety().clear_cache(recursive=True)

    def __repr__(self):
        """Return a string describing the Calabi-Yau manifold.

        The string includes the dimension and, when available, the Hodge numbers
        of the Calabi-Yau, as well as the dimension of the ambient toric variety.

        Returns:
            str: A string describing the Calabi-Yau manifold.

        Examples:
            This function can be used to convert the Calabi-Yau to a string or to
            print information about the Calabi-Yau.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy_info = str(cy)  # Converts to string
            >>> print(cy)  # Prints Calabi-Yau info
            A Calabi-Yau 3-fold hypersurface with h11=1 and h21=101 ...
        """
        d = self.dim()
        if self._is_hypersurface:
            if d == 2:
                out_str = (
                    f"A K3 hypersurface with h11={self.h11()} in a "
                    "3-dimensional toric variety"
                )
            elif d == 3:
                out_str = (
                    "A Calabi-Yau 3-fold hypersurface with "
                    f"h11={self.h11()} and h21={self.h21()} in a "
                    "4-dimensional toric variety"
                )
            elif d == 4:
                out_str = (
                    "A Calabi-Yau 4-fold hypersurface with "
                    f"h11={self.h11()}, h12={self.h12()}, "
                    f"h13={self.h13()}, and h22={self.h22()} in a "
                    "5-dimensional toric variety"
                )
            else:
                out_str = (
                    f"A Calabi-Yau {d}-fold hypersurface in a "
                    f"{d+1}-dimensional toric variety"
                )
        else:
            dd = self.ambient_variety().dim()
            if self._hodge_nums is None or d not in (2, 3, 4):
                out_str = (
                    f"A complete intersection Calabi-Yau {d}-fold in a "
                    f"{dd}-dimensional toric variety"
                )
            elif d == 2:
                out_str = (
                    f"A complete intersection K3 surface with "
                    f"h11={self.h11()} in a "
                    f"{dd}-dimensional toric variety"
                )
            elif d == 3:
                out_str = (
                    f"A complete intersection Calabi-Yau 3-fold with "
                    f"h11={self.h11()} and h21={self.h21()} in a "
                    + f"{dd}-dimensional toric variety"
                )
            elif d == 4:
                out_str = (
                    f"A complete intersection Calabi-Yau 4-fold with "
                    f"h11={self.h11()}, h12={self.h12()}, "
                    f"h13={self.h13()}, and h22={self.h22()} in a "
                    f"{dd}-dimensional toric variety"
                )
        return out_str

    def __eq__(self, other):
        """Compare two Calabi-Yaus with `==`.

        Warning:
            This function only provides a fairly trivial comparison, checking for
            [trivial equivalence][cytools.calabiyau.CalabiYau.is_trivially_equivalent].
            It is not recommended to compare CYs with `==`, and a warning is issued
            every time it evaluates to `False`. It is only implemented so that sets
            and dictionaries of CYs can be created. The `is_trivially_equivalent`
            function should be used instead to avoid confusion.

        Args:
            other (CalabiYau): The other CY that is being compared.

        Returns:
            bool: Whether the two CYs are trivially equivalent. `NotImplemented`
                is returned if `other` is not a `CalabiYau` or if either of them is
                a complete intersection.

        Examples:
            We construct two Calabi-Yaus from triangulations obtained with
            different backends and compare them. We use `is_trivially_equivalent`
            instead of `==`, as recommended.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> cy1 = p.triangulate(backend="qhull").get_cy()
            >>> cy2 = p.triangulate(backend="topcom").get_cy()
            >>> cy1.is_trivially_equivalent(cy2)
            True
        """
        if not isinstance(other, CalabiYau):
            return NotImplemented
        is_triv_equiv = self.is_trivially_equivalent(other)
        if is_triv_equiv is NotImplemented:
            return NotImplemented
        if is_triv_equiv:
            return True
        warnings.warn(
            "The comparison of CYs should not be done with ==. "
            "Please use the is_trivially_equivalent function."
        )
        return False

    def __ne__(self, other):
        """Compare two Calabi-Yaus with `!=`.

        Warning:
            This function only provides a fairly trivial comparison, checking for
            [trivial equivalence][cytools.calabiyau.CalabiYau.is_trivially_equivalent].
            It is not recommended to compare CYs with `!=`, and a warning is issued
            every time the CYs are found not to be trivially equivalent (i.e. when
            this evaluates to `True`). It is only implemented so that sets and
            dictionaries of CYs can be created. The `is_trivially_equivalent`
            function should be used instead to avoid confusion.

        Args:
            other (CalabiYau): The other CY that is being compared.

        Returns:
            bool: Whether the two CYs are not trivially equivalent.
                `NotImplemented` is returned if `other` is not a `CalabiYau`.

        Examples:
            We construct two Calabi-Yaus from triangulations obtained with
            different backends and compare them. We use `is_trivially_equivalent`
            instead of `!=`, as recommended.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> cy1 = p.triangulate(backend="qhull").get_cy()
            >>> cy2 = p.triangulate(backend="topcom").get_cy()
            >>> not cy1.is_trivially_equivalent(cy2)
            False
        """
        if not isinstance(other, CalabiYau):
            return NotImplemented
        return not (self == other)

    def __hash__(self):
        """Return the hash value of the Calabi-Yau.

        For hypersurfaces, the hash is computed from the polytope and from the
        orbit, under the polytope automorphisms, of the restriction of the
        triangulation to codimension-2 faces, so that trivially equivalent CYs
        have the same hash. For complete intersections, it is computed from the
        ambient variety and the nef partition. The result is cached.

        Returns:
            int: The hash value of the CY.

        Examples:
            We compute the hash value of a Calabi-Yau. We also construct a set and
            a dictionary with a Calabi-Yau, which make use of the hash function.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> h = hash(cy)  # Obtain hash value
            >>> d = {cy: 1}  # Create dictionary with Calabi-Yau keys
            >>> s = {cy}  # Create a set of Calabi-Yaus
        """
        if self._hash is not None:
            return self._hash
        if self._is_hypersurface:
            self_orbit = self.triangulation().automorphism_orbit(on_faces_codim=2)
            self_orbit = tuple(tuple(tuple(s) for s in t) for t in self_orbit)
            self._hash = hash((hash(self.triangulation().polytope()), hash(self_orbit)))
        else:
            self._hash = hash((hash(self.ambient_variety()), hash(self._nef_part)))
        return self._hash

    def is_trivially_equivalent(self, other):
        """Check if the Calabi-Yaus are trivially equivalent.

        Two Calabi-Yau hypersurfaces are trivially equivalent if they come from
        the same polytope and the restrictions of their triangulations to
        codimension-2 faces are the same, up to automorphisms of the polytope.
        This function is only implemented for CY hypersurfaces.

        Warning:
            This function only provides a fairly trivial equivalence check. When
            it returns `False`, the Calabi-Yaus may still be equivalent, which
            could, for example, only become evident after a change of basis. The
            full equivalence check is generically very difficult, so it is not
            implemented.

        Args:
            other (CalabiYau): The other CY that is being compared.

        Returns:
            bool: Whether the CYs are trivially equivalent. `NotImplemented` is
                returned if either of them is a complete intersection.

        Examples:
            We construct two Calabi-Yaus and compare them. We also show how to get
            the set of Calabi-Yaus that are not trivially equivalent. As previously
            mentioned, if two CYs are not trivially equivalent it does not mean
            that they are actually inequivalent, as there might exist some more
            complicated transformation that relates them.

            >>> from cytools import Polytope
            >>> p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],
            ...               [2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
            >>> triangs = p.all_triangulations(as_list=True)
            >>> cy0 = triangs[0].get_cy()
            >>> cy1 = triangs[1].get_cy()
            >>> cy0.is_trivially_equivalent(cy1)  # Not necessarily inequivalent
            False
            >>> cys_not_triv_eq = {t.get_cy() for t in triangs}
            >>> len(triangs), len(cys_not_triv_eq)  # Many are trivially equivalent
            (102, 5)
        """
        if not self._is_hypersurface or not other._is_hypersurface:
            return NotImplemented
        if self.polytope() != other.polytope():
            return False
        self_orbit = self.triangulation().automorphism_orbit(on_faces_codim=2)
        other_orbit = other.triangulation().automorphism_orbit(on_faces_codim=2)
        return self_orbit.shape == other_orbit.shape and all(
            (self_orbit == other_orbit).flat
        )

    def ambient_variety(self):
        """Return the ambient toric variety.

        Returns:
            ToricVariety: The ambient toric variety.

        Examples:
            We construct a Calabi-Yau hypersurface in a toric variety and check
            that this function returns the ambient variety.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> v = t.get_toric_variety()
            >>> cy = v.get_cy()
            >>> cy.ambient_variety() is v
            True
        """
        return self._ambient_var

    def triangulation(self):
        """Return the triangulation giving rise to the ambient toric variety.

        Returns:
            Triangulation: The triangulation giving rise to the ambient toric
                variety.

        Examples:
            We construct a Calabi-Yau and check that the triangulation that this
            function returns is the same as the one we used to construct it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.triangulation() is t
            True
        """
        return self.ambient_variety().triangulation()

    def polytope(self):
        """Return the polytope whose triangulation gives rise to the ambient variety.

        Returns:
            Polytope: The polytope whose triangulation gives rise to the ambient
                toric variety.

        Examples:
            We construct a Calabi-Yau and check that the polytope that this
            function returns is the same as the one we used to construct it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.polytope() is p
            True
        """
        return self.ambient_variety().polytope()

    def ambient_dimension(self):
        """Return the complex dimension of the ambient toric variety.

        Also available as `ambient_dim`.

        Returns:
            int: The complex dimension of the ambient toric variety.

        Examples:
            We construct a Calabi-Yau and find the dimension of its ambient variety.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.ambient_dimension()
            np.int64(4)
        """
        return self.ambient_variety().dim()

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """Return the complex dimension of the Calabi-Yau.

        For hypersurfaces, this is one less than the dimension of the ambient
        variety. For complete intersections, it is the dimension of the ambient
        variety minus the number of parts of the nef partition. The result is
        cached.

        Also available as `dim`.

        Returns:
            int: The complex dimension of the Calabi-Yau.

        Examples:
            We construct a Calabi-Yau and find its dimension.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.dimension()
            np.int64(3)
        """
        if self._dim is not None:
            return self._dim

        tv = self.ambient_variety()

        if self._is_hypersurface:
            self._dim = tv.dim() - 1
        else:
            self._dim = tv.triangulation().dim() - len(self._nef_part)

        return self._dim

    # aliases
    dim = dimension

    def hpq(self, p, q):
        """Return the Hodge number $h^{p,q}$ of the Calabi-Yau.

        Hodge numbers with indices outside the Hodge diamond are returned as 0.

        Note:
            For hypersurfaces, the Hodge numbers are computed from the polytope
            (see [`Polytope.hpq`][cytools.polytope.Polytope.hpq]), and only
            Calabi-Yaus of dimension 2-4 are supported. For complete
            intersections, they are computed with PALP.

        Warning:
            For hypersurfaces, only the Hodge numbers with $p+q$ not greater than
            the dimension of the Calabi-Yau are currently computed correctly. The
            remaining ones are returned as 0, even though they are related to the
            others by $h^{p,q}=h^{n-p,n-q}$, where $n$ is the dimension.

        Args:
            p (int): The holomorphic index of the Dolbeault cohomology of interest.
            q (int): The anti-holomorphic index of the Dolbeault cohomology of
                interest.

        Returns:
            int: The Hodge number $h^{p,q}$ of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau and check some of its Hodge numbers.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.hpq(0,0)
            1
            >>> cy.hpq(0,1)
            0
            >>> cy.hpq(1,1)
            2
            >>> cy.hpq(1,2)
            272
        """
        if not self._is_hypersurface:
            if self._hodge_nums is None:
                self._compute_cicy_hodge_numbers()
            return self._hodge_nums.get((p, q), 0)
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are currently supported."
            )
        return self.polytope().hpq(p, q, lattice="N")

    def h11(self):
        """Return the Hodge number $h^{1,1}$ of the Calabi-Yau.

        Note:
            For hypersurfaces, the Hodge numbers are computed from the polytope
            (see [`Polytope.hpq`][cytools.polytope.Polytope.hpq]), and only
            Calabi-Yaus of dimension 2-4 are supported. For complete intersections,
            they are computed with PALP.

        Returns:
            int: The Hodge number $h^{1,1}$ of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its $h^{1,1}$.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.h11()
            2
        """
        if not self._is_hypersurface:
            return self.hpq(1, 1)
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are " "currently supported."
            )
        return self.polytope().h11(lattice="N")

    def h12(self):
        """Return the Hodge number $h^{1,2}$ of the Calabi-Yau.

        Also available as `h21`.

        Note:
            For hypersurfaces, the Hodge numbers are computed from the polytope
            (see [`Polytope.hpq`][cytools.polytope.Polytope.hpq]), and only
            Calabi-Yaus of dimension 2-4 are supported. For complete intersections,
            they are computed with PALP.

        Returns:
            int: The Hodge number $h^{1,2}$ of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its $h^{1,2}$.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.h12()
            272
        """
        if not self._is_hypersurface:
            return self.hpq(1, 2)
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are " "currently supported."
            )
        return self.polytope().h12(lattice="N")

    # aliases
    h21 = h12

    def h13(self):
        """Return the Hodge number $h^{1,3}$ of the Calabi-Yau.

        Also available as `h31`.

        Note:
            For hypersurfaces, the Hodge numbers are computed from the polytope
            (see [`Polytope.hpq`][cytools.polytope.Polytope.hpq]), and only
            Calabi-Yaus of dimension 2-4 are supported. For complete intersections,
            they are computed with PALP.

        Returns:
            int: The Hodge number $h^{1,3}$ of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau 4-fold hypersurface, which requires enabling
            the experimental features, and compute its $h^{1,3}$.

            >>> from cytools import Polytope, config
            >>> config.enable_experimental_features()
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-1,-1,-1]])
            >>> cy = p.triangulate().get_cy()
            >>> cy.h13()
            426
        """
        if not self._is_hypersurface:
            return self.hpq(1, 3)
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are " "currently supported."
            )
        return self.polytope().h13(lattice="N")

    # aliases
    h31 = h13

    def h22(self):
        """Return the Hodge number $h^{2,2}$ of the Calabi-Yau.

        This is mainly of interest for Calabi-Yau 4-folds. For 3-folds it
        is currently returned as 0 (see the warning in
        [`hpq`][cytools.calabiyau.CalabiYau.hpq]).

        Note:
            For hypersurfaces, the Hodge numbers are computed from the polytope
            (see [`Polytope.hpq`][cytools.polytope.Polytope.hpq]), and only
            Calabi-Yaus of dimension 2-4 are supported. For complete intersections,
            they are computed with PALP.

        Returns:
            int: The Hodge number $h^{2,2}$ of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau 4-fold hypersurface, which requires enabling
            the experimental features, and compute its $h^{2,2}$.

            >>> from cytools import Polytope, config
            >>> config.enable_experimental_features()
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-1,-1,-1]])
            >>> cy = p.triangulate().get_cy()
            >>> cy.h22()
            1752
        """
        if not self._is_hypersurface:
            return self.hpq(2, 2)
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are " "currently supported."
            )
        return self.polytope().h22(lattice="N")

    def chi(self):
        """Compute the Euler characteristic of the Calabi-Yau.

        For hypersurfaces, it is computed from the polytope (see
        [`Polytope.chi`][cytools.polytope.Polytope.chi]). For complete
        intersections, it is computed as the alternating sum of the Hodge numbers,
        which are computed with PALP.

        Note:
            For hypersurfaces, only Calabi-Yaus of dimension 2-4 are supported.

        Returns:
            int: The Euler characteristic of the Calabi-Yau.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface whose
                dimension is not 2, 3, or 4, or if it is a complete intersection
                whose Hodge numbers cannot be computed.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its Euler
            characteristic.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.chi()
            -540
        """
        if not self._is_hypersurface:
            if self._hodge_nums is None:
                self._compute_cicy_hodge_numbers()
            if "chi" in self._hodge_nums:
                return self._hodge_nums["chi"]
            chi = 0
            for i in range(2 * self.dim() + 1):
                ii = min(i, self.dim())
                jj = i - ii
                while True:
                    chi += (-1 if i % 2 else 1) * self._hodge_nums[(ii, jj)]
                    ii -= 1
                    jj += 1
                    if ii < 0 or jj > self.dim():
                        break
            self._hodge_nums["chi"] = chi
            return self._hodge_nums["chi"]
        if self.dim() not in (2, 3, 4):
            raise NotImplementedError(
                "Only Calabi-Yaus of dimension 2-4 are " "currently supported."
            )
        return self.polytope().chi(lattice="N")

    def glsm_charge_matrix(self, include_origin=True):
        """Compute the GLSM charge matrix of the Calabi-Yau.

        The columns correspond to the canonical divisor (the origin of the
        polytope) followed by the prime toric divisors that intersect the
        Calabi-Yau (see
        [`prime_toric_divisors`][cytools.calabiyau.CalabiYau.prime_toric_divisors]).
        Each row is a linear relation among the corresponding points of the
        polytope, with one row per independent $U(1)$ charge. The result is cached,
        and a copy is returned.

        Args:
            include_origin (bool): Whether to include the column corresponding to
                the origin, i.e. the canonical divisor.

        Returns:
            np.ndarray: The GLSM charge matrix.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its GLSM charge
            matrix.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.glsm_charge_matrix()
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])
        """
        if self._glsm_charge_matrix is not None:
            return np.array(self._glsm_charge_matrix)[:, (0 if include_origin else 1) :]
        pts = [0] + list(self.prime_toric_divisors())
        self._glsm_charge_matrix = self.polytope().glsm_charge_matrix(
            include_origin=True, points=pts
        )
        return np.array(self._glsm_charge_matrix)[:, (0 if include_origin else 1) :]

    def glsm_linear_relations(self, include_origin=True):
        r"""Compute the linear relations among the columns of the GLSM charge matrix.

        Each row is a vector of coefficients $r$ such that
        $\sum_j r_j Q_{\cdot j} = 0$, where $Q$ is the GLSM charge matrix returned
        by [`glsm_charge_matrix`][cytools.calabiyau.CalabiYau.glsm_charge_matrix].
        The columns follow the same ordering as the columns of the charge matrix.
        The result is cached, and a copy is returned.

        Args:
            include_origin (bool): Whether to include the origin (i.e. the
                canonical divisor). If false, the first row and the first column of
                the matrix are removed.

        Returns:
            np.ndarray: A matrix whose rows are linear relations among the columns
                of the GLSM charge matrix.

        Examples:
            We construct a Calabi-Yau hypersurface and compute the GLSM linear
            relations. By definition, their product with the transposed charge
            matrix must vanish.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.glsm_linear_relations()
            array([[ 1,  0,  0,  0,  0, 18,  6],
                   [ 0,  1,  0,  0,  0, -1,  0],
                   [ 0,  0,  1,  0,  0, -9, -3],
                   [ 0,  0,  0,  1,  0, -6, -2],
                   [ 0,  0,  0,  0,  1, -1,  0]])
            >>> cy.glsm_linear_relations().dot(cy.glsm_charge_matrix().T)
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
        pts = [0] + list(self.prime_toric_divisors())
        self._glsm_linrels = self.polytope().glsm_linear_relations(
            include_origin=True, points=pts
        )
        return np.array(self._glsm_linrels)[
            (0 if include_origin else 1) :, (0 if include_origin else 1) :
        ]

    def divisor_basis(self, include_origin=True, as_matrix=False):
        """Return the current basis of divisors of the Calabi-Yau.

        If no basis has been set, a default integral basis of prime toric divisors
        is chosen and set. A different basis can be set with
        [`set_divisor_basis`][cytools.calabiyau.CalabiYau.set_divisor_basis] or
        [`set_curve_basis`][cytools.calabiyau.CalabiYau.set_curve_basis].

        Args:
            include_origin (bool): Whether the indexing of the returned vector (or
                the columns of the returned matrix) includes the origin, i.e. the
                canonical divisor at index 0.
            as_matrix (bool): Whether to return the basis as a matrix instead of a
                vector of indices of prime toric divisors. If a generic (matrix)
                basis was set, then it is always returned as a matrix.

        Returns:
            np.ndarray: A vector of the indices of the prime toric divisors that
                form the basis. If `as_matrix` is true, or if a generic (matrix)
                basis has been set, a matrix is returned instead, where each row is
                a basis element expressed as a linear combination of the canonical
                divisor and the prime toric divisors.

        Raises:
            ValueError: If `include_origin=False` but the current basis includes
                the canonical divisor.

        Examples:
            We consider a simple Calabi-Yau with two independent divisors. If no
            basis has been set, then this function finds one. If a basis has been
            set, then this function returns it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.divisor_basis()  # We haven't set any basis
            array([5, 6])
            >>> cy.set_divisor_basis([5,6])  # Here we set a basis
            >>> cy.divisor_basis()  # We get the basis we set
            array([5, 6])
            >>> cy.divisor_basis(as_matrix=True)  # We get the basis in matrix form
            array([[0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1]])
        """
        if self._divisor_basis is None:
            pts = [0] + list(self.prime_toric_divisors())
            self.set_divisor_basis(
                self.polytope().glsm_basis(
                    integral=True, include_origin=True, points=pts
                )
            )
        if len(self._divisor_basis.shape) == 1:
            if 0 in self._divisor_basis and not include_origin:
                raise ValueError(
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
        """Set a basis of divisors of the Calabi-Yau.

        This also sets the dual basis of curves. The basis can be specified with a
        vector of indices of prime toric divisors or, if the experimental features
        are enabled, with a matrix whose rows are linear combinations of the
        canonical divisor and the prime toric divisors. Cached results that depend
        on the choice of basis are cleared.

        Note:
            Only integral bases are supported by CYTools, meaning that all prime
            toric divisors must be able to be written as an integral linear
            combination of the basis divisors.

        Args:
            basis (ArrayLike): Vector or matrix specifying a basis. When a vector
                is used, its entries are taken as the indices of the columns of the
                GLSM charge matrix (i.e. the canonical divisor at index 0 followed by
                the prime toric divisors); the indices are sorted before being
                stored. When a matrix is used, its rows are taken as linear
                combinations of the aforementioned divisors. The matrix may omit the
                column of the canonical divisor, in which case its coefficient is set
                to zero.
            include_origin (bool): Whether to interpret the indices in the input
                vector as including the origin at index 0. This is only used when
                `basis` is a vector.

        Raises:
            ValueError: If the input does not specify a valid integral basis.
            Exception: If a matrix is given and the experimental features are not
                enabled.

        Examples:
            We consider a simple Calabi-Yau with two independent divisors. We first
            find the default basis it picks and then we set a basis of our choice.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.divisor_basis()  # We haven't set any basis
            array([5, 6])
            >>> cy.set_divisor_basis([5,6])  # Here we set a basis
            >>> cy.divisor_basis()  # We get the basis we set
            array([5, 6])
            >>> cy.divisor_basis(as_matrix=True)  # We get the basis in matrix form
            array([[0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1]])

            An example with more generic basis choices can be found in the
            [experimental
            features](experimental.md#generic-bases-for-divisors-and-curves)
            section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the ToricVariety class.
        set_divisor_basis(self, basis, include_origin=include_origin)

    def curve_basis(self, include_origin=True, as_matrix=False):
        """Return the current basis of curves of the Calabi-Yau.

        If no basis has been set, a default integral basis of divisors is chosen
        and set, and its dual basis of curves is returned.

        Args:
            include_origin (bool): Whether the indexing of the returned vector (or
                the columns of the returned matrix) includes the origin, i.e. the
                canonical divisor at index 0.
            as_matrix (bool): Whether to return the basis as a matrix instead of a
                vector of indices. If a generic (matrix) basis was set, then it is
                always returned as a matrix.

        Returns:
            np.ndarray: A vector of indices of the standard basis of the lattice
                dual to the lattice of prime toric divisors that form the basis of
                curves. If `as_matrix` is true, or if a generic basis has been set
                with `set_divisor_basis` or `set_curve_basis`, a matrix is returned
                instead, where each row gives the intersection numbers of a basis
                curve with the canonical divisor and the prime toric divisors.

        Raises:
            ValueError: If `include_origin=False` but the current basis includes
                the index of the canonical divisor.

        Examples:
            We consider a simple Calabi-Yau with two independent curves. If no
            basis has been set, then this function finds one. If a basis has been
            set, then this function returns it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.curve_basis()  # We haven't set any basis
            array([5, 6])
            >>> cy.set_curve_basis([5,6])  # Here we set a basis
            >>> cy.curve_basis()  # We get the basis we set
            array([5, 6])
            >>> cy.curve_basis(as_matrix=True)  # We get the basis in matrix form
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])
        """
        if self._curve_basis is None:
            # NOTE: the point set must match the one used by divisor_basis
            # (and by glsm_charge_matrix), namely the origin together with the
            # prime toric divisors. Using the triangulation point set instead
            # gives indices in the wrong range whenever the CY is constructed
            # from a triangulation of all the points of the polytope.
            pts = [0] + list(self.prime_toric_divisors())
            self.set_divisor_basis(
                self.polytope().glsm_basis(
                    integral=True, include_origin=True, points=pts
                )
            )
        if len(self._curve_basis.shape) == 1:
            if 0 in self._curve_basis and not include_origin:
                raise ValueError(
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
        """Set a basis of curves of the Calabi-Yau.

        This in turn induces the dual basis of divisors. The basis can be
        specified with a vector of indices of the standard basis of the lattice
        dual to the lattice of prime toric divisors, which is equivalent to passing
        the same vector to
        [`set_divisor_basis`][cytools.calabiyau.CalabiYau.set_divisor_basis]. If
        the experimental features are enabled, it can also be specified with a
        matrix whose rows give the intersection numbers of each basis curve with
        the canonical divisor and the prime toric divisors. Cached results that
        depend on the choice of basis are cleared.

        Note:
            Only integral bases are supported by CYTools, meaning that all toric
            curves must be able to be written as an integral linear combination of
            the basis curves.

        Args:
            basis (ArrayLike): Vector or matrix specifying a basis. When a vector
                is used, its entries are taken as indices of the standard basis of
                the lattice dual to the lattice of prime toric divisors (with index
                0 corresponding to the canonical divisor). When a matrix is used,
                its rows are taken as linear combinations of the aforementioned
                elements. The matrix may omit the column of the canonical divisor,
                in which case it is computed as minus the sum of the other entries
                of the row.
            include_origin (bool): Whether to interpret the indices in the input
                vector as including the origin at index 0. This is only used when
                `basis` is a vector.

        Raises:
            ValueError: If the input does not specify a valid integral basis.
            Exception: If a matrix is given and the experimental features are not
                enabled.

        Examples:
            We consider a simple Calabi-Yau with two independent curves. We first
            find the default basis of curves it picks and then set a basis of our
            choice.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.curve_basis()  # We haven't set any basis
            array([5, 6])
            >>> cy.set_curve_basis([5,6])  # Here we set a basis
            >>> cy.curve_basis()  # We get the basis we set
            array([5, 6])
            >>> cy.curve_basis(as_matrix=True)  # We get the basis in matrix form
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])

            Note that when setting a curve basis with a vector, this function
            behaves exactly the same as
            [`set_divisor_basis`][cytools.calabiyau.CalabiYau.set_divisor_basis].
            The two functions differ for generic (matrix) bases. An example can be
            found in the [experimental
            features](experimental.md#generic-bases-for-divisors-and-curves)
            section.
        """
        # This is handled by a function in utils since the functionality is
        # shared with the ToricVariety class.
        set_curve_basis(self, basis, include_origin=include_origin)

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
        r"""Return the intersection numbers of the Calabi-Yau.

        The indices refer to the canonical divisor of the ambient variety (index
        0) and the prime toric divisors that intersect the Calabi-Yau (indices
        1, 2, ..., in the order given by
        [`prime_toric_divisors`][cytools.calabiyau.CalabiYau.prime_toric_divisors]),
        all restricted to the Calabi-Yau. They are computed from the
        [intersection numbers][cytools.toricvariety.ToricVariety.intersection_numbers]
        of the ambient variety. For hypersurfaces,
        $\kappa_{ijk\cdots} = -\kappa^{V}_{0ijk\cdots}$, where $\kappa^{V}$ are
        the intersection numbers of the ambient variety. For
        complete intersections, the ambient intersection numbers are successively
        restricted to each part of the nef partition.

        Only the nonzero intersection numbers are stored, and the results are
        cached for each combination of `in_basis`, `format`,
        `zero_as_anticanonical`, and `exact_arithmetic`. The remaining parameters
        only affect the first computation.

        Tip:
            The intersection numbers are integers when the Calabi-Yau is smooth.
            Otherwise, they are computed as floating-point numbers, but there is
            the experimental option to turn them into exact rationals with
            `exact_arithmetic=True`. The process is fairly quick, but it is
            unreliable at large $h^{1,1}$. Furthermore, verifying that they are
            correct becomes very slow at large $h^{1,1}$.

        Args:
            in_basis (bool): Whether to return the intersection numbers in the
                current basis of divisors (see
                [`divisor_basis`][cytools.calabiyau.CalabiYau.divisor_basis]). In
                this case, the indices refer to the basis elements.
            format (str): The output format of the intersection numbers. The
                options are `"dok"`, `"coo"`, and `"dense"`. When set to `"dok"`
                (Dictionary Of Keys), a dictionary is returned where the keys are
                tuples of divisor indices in ascending order and the values are the
                corresponding intersection numbers. When set to `"coo"` (COOrdinate
                format), a numpy array is returned with rows of the form
                `[a, b, ..., c, K_ab...c]`, i.e. all but the last entry of each row
                are divisor indices in ascending order, and the last entry is their
                intersection number. When set to `"dense"`, the full dense array of
                intersection numbers is returned.
            zero_as_anticanonical (bool): Whether to treat the zeroth index as
                corresponding to the anticanonical divisor instead of the canonical
                divisor. This is ignored when `in_basis=True`.
            backend (str): The sparse linear solver to use. Options are `"all"`,
                `"sksparse"`, and `"scipy"`. When set to `"all"`, every solver is
                tried in order until one succeeds.
            check (bool): Whether to explicitly check the solution to the linear
                system.
            backend_error_tol (float): Error tolerance for the solution of the
                linear system.
            round_to_zero_threshold (float): Intersection numbers with magnitude
                smaller than this threshold are rounded to zero (i.e. dropped).
            round_to_integer_error_tol (float): Maximum allowed distance from an
                integer for intersection numbers that are expected to be integral.
                For complete intersections, the Calabi-Yau is considered smooth if
                all of its intersection numbers are within this distance of an
                integer.
            verbose (int): The verbosity level. If 0, nothing is printed. If 1 or
                higher, warnings from the linear backends are printed.
            exact_arithmetic (bool): Whether to convert the intersection numbers
                into exact rational numbers (`flint.fmpq`). This requires the
                experimental features to be enabled.

        Returns:
            dict[tuple[int, ...], int | float] | np.ndarray: The intersection
                numbers in the requested format: a dictionary for `"dok"`, a 2D
                array for `"coo"`, or a dense array of rank equal to the dimension
                of the Calabi-Yau for `"dense"`. A shallow copy of the cached result
                is returned.

        Raises:
            ValueError: If `format` or `backend` is not a valid option, or if
                `exact_arithmetic=True` and the experimental features are not
                enabled.
            RuntimeError: If the intersection numbers of the ambient variety could
                not be computed.
            NotImplementedError: If the Calabi-Yau is a complete intersection of an
                unsupported type.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its intersection
            numbers. By default, the intersection numbers of the canonical divisor
            (index 0) and the prime toric divisors are returned as a dictionary
            whose keys are the divisor indices in ascending order.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> intnums = cy.intersection_numbers()
            >>> intnums[(1, 2, 3)]
            18
            >>> intnums[(0, 0, 0)]
            -1944

            We can also compute the intersection numbers in the current basis of
            divisors. In this case, the indices refer to the basis elements, so
            `(0, 0, 1)` corresponds to the intersection of divisors 5, 5, and 6.

            >>> cy.divisor_basis()
            array([5, 6])
            >>> cy.intersection_numbers(in_basis=True)
            {(0, 1, 1): -3, (1, 1, 1): 9, (0, 0, 1): 1}

            Other output formats are also available. In the `"coo"` format, all but
            the last entry of each row are the indices and the last entry is the
            intersection number. The `"dense"` format gives the full array.

            >>> cy.intersection_numbers(in_basis=True, format="coo")
            array([[ 0,  1,  1, -3],
                   [ 1,  1,  1,  9],
                   [ 0,  0,  1,  1]])
            >>> cy.intersection_numbers(in_basis=True, format="dense").tolist()
            [[[0, 1], [1, -3]], [[1, -3], [-3, 9]]]
        """
        # input checking
        formats = ["dok", "coo", "dense"]
        if format not in formats:
            raise ValueError(f"Options for format are {formats}.")

        #
        if in_basis:
            zero_as_anticanonical = False
        args_id = (zero_as_anticanonical, in_basis, exact_arithmetic, format)
        if args_id in self._intersection_numbers:
            return copy.copy(self._intersection_numbers[args_id])

        # do the calculation
        if (False, False, False, "dok") not in self._intersection_numbers or (
            (False, False, True, "dok") not in self._intersection_numbers
            and exact_arithmetic
        ):
            ambient_intnums = self.ambient_variety().intersection_numbers(
                in_basis=False,
                format="dok",
                backend=backend,
                check=check,
                backend_error_tol=backend_error_tol,
                round_to_zero_threshold=round_to_zero_threshold,
                round_to_integer_error_tol=round_to_integer_error_tol,
                verbose=verbose,
                exact_arithmetic=exact_arithmetic,
            )
            if self._is_hypersurface:
                intnums_cy = {
                    ii[1:]: -ambient_intnums[ii] for ii in ambient_intnums if 0 in ii
                }
            else:
                parts = self._nef_part
                ambient_dim = self.ambient_dim()
                intnums_dict = ambient_intnums
                for dd in range(len(parts)):
                    intnums_dict_tmp = defaultdict(lambda: 0)
                    for ii in intnums_dict:
                        choices = {
                            tuple(sorted(c for i, c in enumerate(ii) if i != j))
                            for j in range(ambient_dim - dd)
                            if ii[j] in parts[dd]
                        }
                        for c in choices:
                            intnums_dict_tmp[c] += intnums_dict[ii]
                    intnums_dict = {
                        ii: intnums_dict_tmp[ii]
                        for ii in intnums_dict_tmp
                        if abs(intnums_dict_tmp[ii]) > round_to_zero_threshold
                    }
                intnums_cy = intnums_dict
                if all(
                    abs(round(intnums_cy[ii]) - intnums_cy[ii])
                    < round_to_integer_error_tol
                    for ii in intnums_cy
                ):
                    self._is_smooth = True
                    for ii in intnums_cy:
                        intnums_cy[ii] = int(round(intnums_cy[ii]))
                else:
                    self._is_smooth = False
                # Now we find the prime toric divisors and reindex accordingly
                intnum_ind = set.union(*[set(ii) for ii in intnums_cy])
                triang_inds = sorted(intnum_ind)
                nonzero_triang_inds = [i for i in triang_inds if i]
                self._prime_divs = tuple(
                    self.triangulation().triangulation_to_polytope_indices(
                        nonzero_triang_inds
                    )
                )
                # NOTE: the keys of intnums_cy are *triangulation* indices, so
                # the relabeling map must be keyed by those. self._prime_divs
                # holds the corresponding *polytope* indices, which in general
                # are different (they only coincide when the triangulation uses
                # every point of the polytope).
                divs_dict = {ii: i for i, ii in enumerate(nonzero_triang_inds, 1)}
                divs_dict[0] = 0
                intnums_cy = {
                    tuple(divs_dict[i] for i in ii): intnums_cy[ii] for ii in intnums_cy
                }
                # If there are some non-intersecting divisors we construct a better
                # toric variety in the background
                if (
                    len(self._prime_divs)
                    == self.ambient_variety().triangulation().points().shape[0] - 1
                ):
                    self._optimal_ambient_var = self._ambient_var
                else:
                    heights = self.triangulation().heights()[triang_inds]
                    try:
                        self._optimal_ambient_var = (
                            self.polytope()
                            .triangulate(
                                heights=heights,
                                points=self.polytope().points_to_indices(
                                    self.polytope().points()[
                                        [0] + list(self._prime_divs)
                                    ]
                                ),
                            )
                            .get_toric_variety()
                        )
                    except Exception:
                        raise NotImplementedError(
                            "This type of complete intersection is not supported."
                        )
            self._intersection_numbers[(False, False, exact_arithmetic, "dok")] = (
                intnums_cy
            )
        # Now intersection numbers have been computed
        # We now compute the intersection numbers of the basis if necessary
        if zero_as_anticanonical and not in_basis:
            base_intnums = self._intersection_numbers[
                (False, False, exact_arithmetic, "dok")
            ]
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
        """Return the indices of the prime toric divisors that intersect the Calabi-Yau.

        The indices correspond to points of the polytope, i.e. if $n$ is in the
        returned tuple, then the prime toric divisor corresponding to the $n$-th
        point in `p.points()` intersects the Calabi-Yau. Due to the ordering of the
        points in the [`Polytope`][cytools.polytope.Polytope] class, this is simply
        `(1, 2, ..., k)` for hypersurfaces, where `k` is the number of points not
        interior to facets, excluding the origin. For complete intersections, it
        may be non-trivial, and it is found by computing the intersection numbers.
        The result is cached.

        Returns:
            tuple[int, ...]: The indices of the points of the polytope whose
                corresponding prime toric divisors intersect the Calabi-Yau.

        Examples:
            We construct a Calabi-Yau hypersurface and find its prime toric
            divisors.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.prime_toric_divisors()
            (1, 2, 3, 4, 5, 6)
        """
        if self._prime_divs is not None:
            return self._prime_divs
        if self._is_hypersurface:
            self._prime_divs = tuple(
                range(1, self.polytope().points_not_interior_to_facets().shape[0])
            )
            self._optimal_ambient_var = self._ambient_var
        else:
            # For CICYs we have to compute intersection numbers and the
            # variables are set during the computation
            self.intersection_numbers()
        return self._prime_divs

    def second_chern_class(self, in_basis=False, include_origin=True):
        r"""Compute the second Chern class of the Calabi-Yau.

        The second Chern class is returned as the vector of its integrals over the
        canonical divisor and the prime toric divisors, i.e.
        $\int_X c_2 \wedge D_k = \sum_{i<j} \kappa_{ijk}$, where the sum runs over
        pairs of distinct prime toric divisors. The entry corresponding to the
        canonical divisor is minus the sum of all the others. The result is cached.

        Note:
            This function currently only supports Calabi-Yau 3-folds, and the
            formula it uses is only valid for hypersurfaces.

        Args:
            in_basis (bool): Whether to only return the integrals over the current
                basis of divisors. If true, `include_origin` is ignored.
            include_origin (bool): Whether to include the integral over the
                canonical divisor as the first entry of the vector.

        Returns:
            np.ndarray: A vector containing the integrals. It has integer entries if
                the intersection numbers are integral, and floating-point entries
                otherwise.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its second Chern
            class.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.second_chern_class()
            array([-612,   36,  306,  204,   36,   36,   -6])
            >>> cy.second_chern_class(in_basis=True)
            array([36, -6])
        """
        if self.dim() != 3:
            raise NotImplementedError("This function currently only supports 3-folds.")
        if self._second_chern_class is None:
            intnums = self.intersection_numbers(in_basis=False)
            # NOTE: for non-smooth CYs the intersection numbers are not
            # integral. Accumulating them into an integer array would silently
            # truncate them, so the dtype is chosen based on the actual values.
            dtype = (
                int if all(float(v).is_integer() for v in intnums.values()) else float
            )
            c2 = np.zeros(len(self.prime_toric_divisors()) + 1, dtype=dtype)
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
            if len(basis.shape) == 2:  # If basis is matrix
                return self._second_chern_class.dot(basis.T)
            return np.array(self._second_chern_class[basis])
        return np.array(self._second_chern_class)[(0 if include_origin else 1) :]

    def is_smooth(self):
        """Return whether the Calabi-Yau is smooth.

        For hypersurfaces, this is determined by whether the canonical divisor of
        the ambient variety is
        [smooth][cytools.toricvariety.ToricVariety.canonical_divisor_is_smooth].
        For complete intersections, it is determined while computing the
        intersection numbers, by checking whether they are all integral. The result
        is cached.

        Returns:
            bool: Whether the Calabi-Yau is smooth.

        Examples:
            We construct a Calabi-Yau hypersurface and check if it is smooth.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.is_smooth()
            True
        """
        if self._is_smooth is not None:
            return self._is_smooth
        if self._is_hypersurface:
            self._is_smooth = self.ambient_variety().canonical_divisor_is_smooth()
        else:
            self.intersection_numbers()  # The variable is set while computing intersection numbers
        return self._is_smooth

    def toric_mori_cone(self, in_basis=False, include_origin=True):
        """Return the Mori cone inferred from toric geometry.

        This is the Mori cone of the ambient toric variety (or, for complete
        intersections, of a toric variety constructed from only the prime toric
        divisors that intersect the Calabi-Yau). It contains the Mori cone of the
        Calabi-Yau. Each ray is a curve class, given by its intersection numbers
        with the canonical divisor and the prime toric divisors, or with the basis
        divisors if `in_basis=True`. The result is cached.

        Args:
            in_basis (bool): Whether to use the current basis of curves, which is
                dual to the basis returned by
                [`divisor_basis`][cytools.calabiyau.CalabiYau.divisor_basis]. If
                true, `include_origin` is ignored.
            include_origin (bool): Whether to include the coordinate corresponding
                to the origin of the polytope, i.e. the canonical divisor.

        Returns:
            Cone: The Mori cone inferred from toric geometry.

        Examples:
            We construct a Calabi-Yau hypersurface and find its Mori cone in an
            $(h^{1,1}+d+1)$-dimensional lattice, where $d$ is the dimension of the
            ambient variety (i.e. without a particular choice of basis), and in an
            $h^{1,1}$-dimensional lattice (i.e. after picking a basis of curves).

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.toric_mori_cone()  # By default it does not use a basis of curves
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
            >>> cy.toric_mori_cone(in_basis=True)  # Dual to the current divisor basis
            A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        """
        if self._mori_cone[0] is None:
            if (
                self._optimal_ambient_var is None
            ):  # Make sure self._optimal_ambient_var is set
                self.prime_toric_divisors()
            self._mori_cone[0] = self._optimal_ambient_var.mori_cone()
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

    def toric_kahler_cone(self):
        """Return the Kähler cone inferred from toric geometry.

        This is the dual of the toric Mori cone (see
        [`toric_mori_cone`][cytools.calabiyau.CalabiYau.toric_mori_cone]) in the
        current basis of divisors. It is contained in the Kähler cone of the
        Calabi-Yau.

        Returns:
            Cone: The Kähler cone inferred from toric geometry, in the current basis
                of divisors.

        Examples:
            We construct a Calabi-Yau hypersurface and find its Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.toric_kahler_cone()
            A rational polyhedral cone in RR^2 defined by 3 hyperplanes
        """
        return self.toric_mori_cone(in_basis=True).dual()

    def toric_effective_cone(self):
        """Return the effective cone inferred from toric geometry.

        This is the cone generated by the classes of the prime toric divisors, in
        the current basis of divisors. The result is cached.

        Returns:
            Cone: The toric effective cone.

        Examples:
            We construct a Calabi-Yau hypersurface and find its toric effective
            cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.toric_effective_cone()
            A 2-dimensional rational polyhedral cone in RR^2 generated by 6 rays
        """
        if self._eff_cone is not None:
            return self._eff_cone
        self._eff_cone = Cone(self.curve_basis(include_origin=False, as_matrix=True).T)
        return self._eff_cone

    def _basis_intersection_numbers(self):
        r"""Return the intersection numbers of the Calabi-Yau in the current basis.

        The fan of the defining triangulation can compute the in-basis
        intersection numbers much faster than
        [`intersection_numbers`][cytools.calabiyau.CalabiYau.intersection_numbers],
        but it only knows about the polytope's own (Gale) basis. It is therefore
        only used when that basis agrees with the basis currently set on this
        Calabi-Yau. Otherwise, this falls back to
        [`intersection_numbers`][cytools.calabiyau.CalabiYau.intersection_numbers],
        which honors bases set with
        [`set_divisor_basis`][cytools.calabiyau.CalabiYau.set_divisor_basis] and
        [`set_curve_basis`][cytools.calabiyau.CalabiYau.set_curve_basis].

        Note:
            This function should not be called by the user. It is a helper for the
            volume and Kähler metric functions. The returned array is cached and
            must be treated as read-only.

        Returns:
            np.ndarray: The dense, symmetric array of intersection numbers
                $\kappa_{ijk\cdots}$ in the current basis of divisors.
        """
        if self._basis_intnums is not None:
            return self._basis_intnums

        basis = self.divisor_basis()
        if len(basis.shape) == 1:
            # An index basis may coincide with the one used by the fan
            if self._fan is None:
                self._fan = self.triangulation().fan()
            fan_basis = self._fan.vc.divisor_basis
            if fan_basis is not None and np.array_equal(
                np.asarray(fan_basis), np.asarray(basis)
            ):
                self._basis_intnums = self._fan.intersection_numbers(
                    pushed_down=True,
                    in_basis=True,
                    as_np_array=True,
                    copy=False,
                )
                return self._basis_intnums

        self._basis_intnums = np.asarray(
            self.intersection_numbers(
                in_basis=True, format="dense", exact_arithmetic=False
            )
        )
        return self._basis_intnums

    def compute_cy_volume(self, tloc):
        r"""Compute the volume of the Calabi-Yau at a location in the Kähler cone.

        The volume is given by $\mathcal{V} = \frac{1}{6}\kappa_{ijk}t^i t^j t^k$,
        where $\kappa_{ijk}$ are the intersection numbers in the current basis of
        divisors.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.

        Returns:
            float: The volume of the Calabi-Yau at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and find its volume at the tip
            of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_cy_volume(tip)
            np.float64(3.4999999...)
        """
        if self.dim() != 3:
            raise NotImplementedError(
                "This function only supports Calabi-Yau 3-folds."
            )
        intnums = self._basis_intersection_numbers()
        # the 3D x 1D first contraction goes via tensordot because `@`
        # dispatches it as a stack of 2D gemvs and is ~2x slower here
        K = np.tensordot(intnums, tloc, axes=([-1], [0]))
        return (K @ tloc) @ tloc / 6

    def compute_divisor_volumes(self, tloc, in_basis=False):
        r"""Compute the volumes of divisors at a location in the Kähler cone.

        The volume of the $i$-th basis divisor is given by
        $\tau_i = \frac{1}{2}\kappa_{ijk}t^j t^k$, where $\kappa_{ijk}$ are the
        intersection numbers in the current basis of divisors. The volumes of the
        prime toric divisors are obtained by expressing them in terms of the basis
        divisors.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.
            in_basis (bool): Whether to compute the volumes of the current basis of
                divisors. Otherwise, the volumes of all the prime toric divisors are
                computed.

        Returns:
            np.ndarray: The volumes of the prime toric divisors or of the basis
                divisors at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and find the volumes of the
            prime toric divisors and of the basis divisors at the tip of the
            stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_divisor_volumes(tip)
            array([ 2.5, 24. , 16. ,  2.5,  2.5,  0.5])
            >>> cy.compute_divisor_volumes(tip, in_basis=True)
            array([2.5, 0.5])
        """
        if self.dim() != 3:
            raise NotImplementedError(
                "This function only supports Calabi-Yau 3-folds."
            )
        intnums = self._basis_intersection_numbers()
        # the 3D x 1D first contraction goes via tensordot because `@`
        # dispatches it as a stack of 2D gemvs and is ~2x slower here
        K = np.tensordot(intnums, tloc, axes=([-1], [0]))
        tau_basis = (K @ tloc) / 2
        if in_basis:
            return np.asarray(tau_basis)
        # Convert the basis-form volumes to out-of-basis form. Each prime toric
        # divisor is written in the current basis as D_a = sum_i N_{ia} D_i,
        # where N is the (dual) curve basis matrix. For the default basis N is
        # the GLSM charge matrix, but for a general basis it is not.
        curve_basis = self.curve_basis(as_matrix=True, include_origin=False)
        return np.asarray(curve_basis.T @ tau_basis)

    def compute_curve_volumes(self, tloc, only_extremal=False):
        """Compute the volumes of the generators of the toric Mori cone.

        The curves correspond to the (not necessarily minimal) generators of the
        Mori cone inferred from toric geometry in the current basis of curves
        (i.e. the rays of `toric_mori_cone(in_basis=True)`, see
        [`toric_mori_cone`][cytools.calabiyau.CalabiYau.toric_mori_cone]). The
        volume of a curve $C$ is $t^i C_i$.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.
            only_extremal (bool): Whether to only use the extremal rays of the Mori
                cone.

        Returns:
            np.ndarray: The volumes of the curves, in the same order as the rays of
                the cone.

        Examples:
            We construct a Calabi-Yau hypersurface and find the volumes of the
            generators of the Mori cone at the tip of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_curve_volumes(tip)
            array([1., 4., 1.])

            As expected, all generators of the Mori cone have volumes greater than
            or equal to 1 (up to rounding errors) at the tip of the stretched
            Kähler cone.
        """
        c = self.toric_mori_cone(in_basis=True)
        if only_extremal:
            return c.extremal_rays().dot(tloc)
        return c.rays().dot(tloc)

    def compute_kappa_matrix(self, tloc):
        r"""Compute the matrix $\kappa_{ijk}t^k$ at a location in the Kähler cone.

        Here $\kappa_{ijk}$ are the intersection numbers in the current basis of
        divisors.

        Also available as `compute_AA`.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.

        Returns:
            np.ndarray: The matrix $\kappa_{ijk}t^k$ at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and compute this matrix at the
            tip of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_kappa_matrix(tip)
            array([[ 1.,  1.],
                   [ 1., -3.]])
        """
        if self.dim() != 3:
            raise NotImplementedError(
                "This function only supports Calabi-Yau 3-folds."
            )
        intnums = self._basis_intersection_numbers()
        return np.tensordot(intnums, tloc, axes=[[-1], [0]])

    # aliases
    compute_AA = compute_kappa_matrix

    def compute_kappa_vector(self, tloc):
        r"""Compute the vector $\kappa_{ijk}t^j t^k$ at a location in the Kähler cone.

        Here $\kappa_{ijk}$ are the intersection numbers in the current basis of
        divisors.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.

        Returns:
            np.ndarray: The vector $\kappa_{ijk}t^j t^k$ at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and compute this vector at the
            tip of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_kappa_vector(tip)
            array([5., 1.])
        """
        AA = self.compute_kappa_matrix(tloc)
        return AA.dot(tloc)

    def compute_inverse_kahler_metric(self, tloc):
        r"""Compute the inverse Kähler metric at a location in the Kähler cone.

        It is computed as
        $K^{ij} = 4\left(\tau_i\tau_j - \mathcal{V}\kappa_{ijk}t^k\right)$, where
        $\mathcal{V}$ is the volume of the Calabi-Yau, $\tau_i$ are the volumes of
        the basis divisors, and $\kappa_{ijk}$ are the intersection numbers in the
        current basis of divisors.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.

        Returns:
            np.ndarray: The inverse Kähler metric at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and compute the inverse Kähler
            metric at the tip of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_inverse_kahler_metric(tip)
            array([[11., -9.],
                   [-9., 43.]])
        """
        if self.dim() != 3:
            raise NotImplementedError("This function only supports Calabi-Yau 3-folds.")
        xvol = self.compute_cy_volume(tloc)
        Tau = self.compute_divisor_volumes(tloc, in_basis=True)
        AA = self.compute_AA(tloc)
        Kinv = 4 * (np.outer(Tau, Tau) - AA * xvol)
        return Kinv

    def compute_kahler_metric(self, tloc):
        """Compute the Kähler metric at a location in the Kähler cone.

        It is computed by inverting the result of `compute_inverse_kahler_metric`.

        Note:
            This function only supports Calabi-Yau 3-folds.

        Args:
            tloc (ArrayLike): A vector specifying a location in the Kähler cone, in
                the current basis of divisors.

        Returns:
            np.ndarray: The Kähler metric at the specified location.

        Raises:
            NotImplementedError: If the Calabi-Yau is not a 3-fold.

        Examples:
            We construct a Calabi-Yau hypersurface and compute the Kähler metric at
            the tip of the stretched Kähler cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
            >>> cy.compute_kahler_metric(tip)
            array([[0.10969388, 0.02295918],
                   [0.02295918, 0.02806122]])
        """
        return np.linalg.inv(self.compute_inverse_kahler_metric(tloc))

    def _compute_cicy_hodge_numbers(self, only_from_cache=False):
        """Compute the Hodge numbers of a complete intersection Calabi-Yau.

        The Hodge numbers are looked up in the cache of nef partitions of the
        polytope (see
        [`Polytope.nef_partitions`][cytools.polytope.Polytope.nef_partitions]),
        matching the nef partition of this Calabi-Yau restricted to the vertices
        of the polytope. If they are not found there and `only_from_cache` is
        false, the nef partitions (first without, and then with, symmetric ones)
        are computed with PALP and the search is repeated. The results are stored
        in a hidden dictionary keyed by `(p, q)`.

        Note:
            This function should generally not be called by the user. Instead, it
            is called by [`hpq`][cytools.calabiyau.CalabiYau.hpq] and the other
            Hodge number functions when necessary.

        Args:
            only_from_cache (bool): Whether to only look up the Hodge numbers in
                the cache of the polytope, without computing them with PALP if they
                are not found. In that case, nothing is stored if they are not
                found.

        Raises:
            NotImplementedError: If the Calabi-Yau is a hypersurface, or if the
                Hodge numbers could not be found and `only_from_cache` is false.
        """
        if self._is_hypersurface:
            raise NotImplementedError(
                "This function should only be used for codim > 2 CICYs."
            )
        if self._hodge_nums is not None:
            return
        codim = self.ambient_variety().dim() - self.dim()
        poly = self.ambient_variety().polytope()
        vert_ind = poly.points_to_indices(poly.vertices())
        nef_part_fs = frozenset(
            frozenset(i for i in part if i in vert_ind) for part in self._nef_part
        )
        matched_hodge_nums = ()

        def search_in_cache():
            for args_id in poly._nef_parts:
                # Search only the ones with correct codim, computed hodge numbers, and without products or projections
                if args_id[-2] != codim or not args_id[-1] or args_id[1] or args_id[2]:
                    continue
                for i, nef_part in enumerate(poly._nef_parts[args_id][0]):
                    tmp_fs = frozenset(
                        frozenset(ii for ii in part if ii in vert_ind)
                        for part in nef_part
                    )
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
            raise NotImplementedError(
                "This type of complete intersection is not supported."
            )
        if len(matched_hodge_nums):
            self._hodge_nums = dict()
            n = 0
            for i in range(2 * self.dim() + 1):
                ii = min(i, self.dim())
                jj = i - ii
                while True:
                    self._hodge_nums[(ii, jj)] = matched_hodge_nums[n]
                    n += 1
                    ii -= 1
                    jj += 1
                    if ii < 0 or jj > self.dim():
                        break

    # GVs
    # ---
    def _compute_gvs_gws(
        self,
        gv_or_gw: str,
        mcap_generators: "ArrayLike" = None,
        grading_vec: "ArrayLike" = None,
        max_deg: int = None,
        min_points: int = None,
        target_points: "ArrayLike" = None,
        basis: "ArrayLike" = None,
        format: str = None,
    ):
        r"""Compute Gopakumar-Vafa or Gromov-Witten invariants using cygv.

        This is the shared implementation of
        [`compute_gvs`][cytools.calabiyau.CalabiYau.compute_gvs] and
        [`compute_gws`][cytools.calabiyau.CalabiYau.compute_gws], and it is not
        intended to be called directly. The charges are expressed in the current
        basis of curves (see
        [`curve_basis`][cytools.calabiyau.CalabiYau.curve_basis]).

        Args:
            gv_or_gw (str): Which invariants to compute: `"gv"` for Gopakumar-Vafa
                invariants, or `"gw"` for Gromov-Witten invariants.
            mcap_generators (ArrayLike | None): Generators of the cone of charges
                to consider (typically the Mori cone cap), in the current basis of
                curves, which are passed to cygv. If none of `max_deg`, `min_points`,
                and `target_points` is specified, the invariants are computed only
                for these charges, which must then include the origin. If not
                specified, the rays of
                [`mori_cone_cap`][cytools.calabiyau.CalabiYau.mori_cone_cap] (in
                the current basis) together with at least $100\,h^{1,1}$ of its
                lattice points are used.
            grading_vec (ArrayLike | None): The grading vector $g$, which defines
                the degree of a charge $q$ as $q \cdot g$. It must have a positive
                dot product with all the generators of the Mori cone cap. If not
                specified, one is found with
                [`Cone.find_grading_vector`][cytools.cone.Cone.find_grading_vector].
            max_deg (int | None): The maximum degree up to which the invariants are
                computed.
            min_points (int | None): The minimum number of invariants to compute.
                The maximum degree is increased until at least this many are found.
            target_points (ArrayLike | None): Charges, in the current basis of
                curves, for which to compute the invariants. Invariants of other
                charges may be computed as well.
            basis (np.ndarray | None): A matrix used to change the basis of the
                charges in the returned
                [`Invariants`][cytools.calabiyau.Invariants] object, where each
                charge $q$ is replaced by `basis @ q`. It is ignored if `format` is
                specified.
            format (str | None): The output format. If `"dok"`, a dictionary
                mapping charges to invariants is returned. If `"coo"`, a list of
                rows of the form `[q_1, ..., q_n, invariant]` is returned. If not
                specified, an [`Invariants`][cytools.calabiyau.Invariants] object
                is returned.

        Returns:
            Invariants | dict[tuple[int, ...], int] | list[list[int]]: The computed
                invariants, in the format specified by `format`.

        Raises:
            ValueError: If more than one of `max_deg`, `min_points`, and
                `target_points` is specified, or if none of them is specified and
                `mcap_generators` is not specified or does not include the
                origin.
        """
        # input checking
        if (max_deg is not None) and (min_points is not None):
            raise ValueError("Either max_deg or min_points can be set, not both")

        if (max_deg is not None) and (target_points is not None):
            raise ValueError("Either max_deg or target_points can be set, not both")

        if (min_points is not None) and (target_points is not None):
            raise ValueError("Either min_points or target_points can be set, not both")

        if (max_deg is None) and (min_points is None) and (target_points is None):
            # computing GVs by input points
            if mcap_generators is None:
                raise ValueError("If neither max_deg, min_points, nor target_points is set, you must set mcap_generators")

            # (in this case, we definitely still need the origin to be input)
            if not np.any(np.all(np.array(mcap_generators) == 0, axis=1)):
                raise ValueError("Origin is not in mcap_generators... you should pass points like " +\
                                 "m_cap.find_lattice_points(min_points=100)...")

        # get basics
        mori = self.mori_cone_cap(in_basis=True)
        if mcap_generators is None:
            R = mori.rays()
            lattice_pts = mori.find_lattice_points(min_points=100*self.h11())
            mcap_generators = np.vstack([R,lattice_pts])

        # compute a grading vector if none is provided
        if grading_vec is None:
            grading_vec = mori.find_grading_vector()

        # compute the GVs
        if gv_or_gw == "gv":
            fct = cygv.compute_gv
        else:
            fct = cygv.compute_gw
        invariants = fct(
            generators=mcap_generators,
            grading_vector=grading_vec,
            q=self.curve_basis(include_origin=False, as_matrix=True),
            intnums=self.intersection_numbers(in_basis=True, format="dok"),
            max_deg=max_deg,
            min_points=min_points,
            target_points=target_points,
        )

        # format/return the GVs
        if format == "coo":
            return [list(r[0]) + [r[1]] for r in invariants]

        # convert to dok for subsequent processing
        invariants = dict(invariants)

        if format == "dok":
            return invariants

        # return in Invariant class
        invariants = Invariants(
            gv_or_gw,
            invariants,
            grading_vec,
            max_deg,
            calabiyau=self,
            basis=basis,
        )

        return invariants

    def compute_gvs(
        self,
        mcap_generators: "ArrayLike" = None,
        grading_vec: "ArrayLike" = None,
        max_deg: int = None,
        min_points: int = None,
        target_points: "ArrayLike" = None,
        basis: "ArrayLike" = None,
        format: str = None,
    ):
        r"""Compute the genus-zero Gopakumar-Vafa invariants of the Calabi-Yau.

        The computation is performed with
        [cygv](https://github.com/ariostas/cygv). The charges of the curves are
        expressed in the current basis of curves (see
        [`curve_basis`][cytools.calabiyau.CalabiYau.curve_basis]). At most one of
        `max_deg`, `min_points`, and `target_points` can be specified. If none of
        them is, the invariants are computed only for the charges in
        `mcap_generators`.

        Also available as `compute_gv`.

        Args:
            mcap_generators (ArrayLike | None): Generators of the cone of charges
                to consider (typically the Mori cone cap), in the current basis of
                curves, which are passed to cygv. If none of `max_deg`, `min_points`,
                and `target_points` is specified, the invariants are computed only
                for these charges, which must then include the origin. If not
                specified, the rays of
                [`mori_cone_cap`][cytools.calabiyau.CalabiYau.mori_cone_cap] (in
                the current basis) together with at least $100\,h^{1,1}$ of its
                lattice points are used.
            grading_vec (ArrayLike | None): The grading vector $g$, which defines
                the degree of a charge $q$ as $q \cdot g$. It must have a positive
                dot product with all the generators of the Mori cone cap. If not
                specified, one is found with
                [`Cone.find_grading_vector`][cytools.cone.Cone.find_grading_vector].
            max_deg (int | None): The maximum degree up to which the invariants are
                computed.
            min_points (int | None): The minimum number of invariants to compute.
                The maximum degree is increased until at least this many are found.
            target_points (ArrayLike | None): Charges, in the current basis of
                curves, for which to compute the invariants. Invariants of other
                charges may be computed as well.
            basis (np.ndarray | None): A matrix used to change the basis of the
                charges in the returned
                [`Invariants`][cytools.calabiyau.Invariants] object, where each
                charge $q$ is replaced by `basis @ q`. It is ignored if `format` is
                specified.
            format (str | None): The output format. If `"dok"`, a dictionary
                mapping charges to invariants is returned. If `"coo"`, a list of
                rows of the form `[q_1, ..., q_n, invariant]` is returned. If not
                specified, an [`Invariants`][cytools.calabiyau.Invariants] object
                is returned.

        Returns:
            Invariants | dict[tuple[int, ...], int] | list[list[int]]: The GV
                invariants, in the format specified by `format`.

        Raises:
            ValueError: If more than one of `max_deg`, `min_points`, and
                `target_points` is specified, or if none of them is specified and
                `mcap_generators` is not specified or does not include the
                origin.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its GV invariants
            up to degree 2.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> gvs = cy.compute_gvs(max_deg=2)
            >>> gvs
            N=5 gv-invariants calculated for a Calabi-Yau 3-fold hypersurface ...
            >>> gvs.gv((0, 1))
            540
            >>> sorted(cy.compute_gvs(max_deg=1, format="dok").items())
            [((0, 1), 540), ((1, -3), 3)]
        """
        return self._compute_gvs_gws(
            gv_or_gw="gv",
            mcap_generators=mcap_generators,
            grading_vec=grading_vec,
            max_deg=max_deg,
            min_points=min_points,
            target_points=target_points,
            basis=basis,
            format=format
        )

    compute_gv = compute_gvs

    def compute_gws(
        self,
        mcap_generators: "ArrayLike" = None,
        grading_vec: "ArrayLike" = None,
        max_deg: int = None,
        min_points: int = None,
        target_points: "ArrayLike" = None,
        basis: "ArrayLike" = None,
        format: str = None,
    ):
        r"""Compute the genus-zero Gromov-Witten invariants of the Calabi-Yau.

        The computation is performed with
        [cygv](https://github.com/ariostas/cygv). The charges of the curves are
        expressed in the current basis of curves (see
        [`curve_basis`][cytools.calabiyau.CalabiYau.curve_basis]). At most one of
        `max_deg`, `min_points`, and `target_points` can be specified. If none of
        them is, the invariants are computed only for the charges in
        `mcap_generators`. The invariants are rational numbers, given as
        `fractions.Fraction` objects.

        Also available as `compute_gw`.

        Args:
            mcap_generators (ArrayLike | None): Generators of the cone of charges
                to consider (typically the Mori cone cap), in the current basis of
                curves, which are passed to cygv. If none of `max_deg`, `min_points`,
                and `target_points` is specified, the invariants are computed only
                for these charges, which must then include the origin. If not
                specified, the rays of
                [`mori_cone_cap`][cytools.calabiyau.CalabiYau.mori_cone_cap] (in
                the current basis) together with at least $100\,h^{1,1}$ of its
                lattice points are used.
            grading_vec (ArrayLike | None): The grading vector $g$, which defines
                the degree of a charge $q$ as $q \cdot g$. It must have a positive
                dot product with all the generators of the Mori cone cap. If not
                specified, one is found with
                [`Cone.find_grading_vector`][cytools.cone.Cone.find_grading_vector].
            max_deg (int | None): The maximum degree up to which the invariants are
                computed.
            min_points (int | None): The minimum number of invariants to compute.
                The maximum degree is increased until at least this many are found.
            target_points (ArrayLike | None): Charges, in the current basis of
                curves, for which to compute the invariants. Invariants of other
                charges may be computed as well.
            basis (np.ndarray | None): A matrix used to change the basis of the
                charges in the returned
                [`Invariants`][cytools.calabiyau.Invariants] object, where each
                charge $q$ is replaced by `basis @ q`. It is ignored if `format` is
                specified.
            format (str | None): The output format. If `"dok"`, a dictionary
                mapping charges to invariants is returned. If `"coo"`, a list of
                rows of the form `[q_1, ..., q_n, invariant]` is returned. If not
                specified, an [`Invariants`][cytools.calabiyau.Invariants] object
                is returned.

        Returns:
            Invariants | dict[tuple[int, ...], Fraction] | list[list]: The GW
                invariants, in the format specified by `format`.

        Raises:
            ValueError: If more than one of `max_deg`, `min_points`, and
                `target_points` is specified, or if none of them is specified and
                `mcap_generators` is not specified or does not include the
                origin.

        Examples:
            We construct a Calabi-Yau hypersurface and compute its GW invariants
            up to degree 2.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> gws = cy.compute_gws(max_deg=2)
            >>> gws.gw((0, 2))
            Fraction(1215, 2)
        """
        return self._compute_gvs_gws(
            gv_or_gw="gw",
            mcap_generators=mcap_generators,
            grading_vec=grading_vec,
            max_deg=max_deg,
            min_points=min_points,
            target_points=target_points,
            basis=basis,
            format=format
        )

    compute_gw = compute_gws


    # =================
    # TEMPORARY METHODS
    # =================
    def mori_cone_cap(self, in_basis: bool = False,
                      exclude_origin: bool = False,
                      format: str = None,
                      verbosity: int = 0):
        """Return the Mori cone cap of the Calabi-Yau.

        This is the intersection of the toric Mori cones of all the regular star
        triangulations of the polytope that restrict to the same triangulations of
        its 2-faces as the defining triangulation. Since all of these
        triangulations give rise to the same Calabi-Yau hypersurface, this cone
        contains the Mori cone of the Calabi-Yau, and it is generally smaller than
        the one returned by
        [`toric_mori_cone`][cytools.calabiyau.CalabiYau.toric_mori_cone]. Without
        a basis, the coordinates correspond to the points of the triangulation,
        with the origin (i.e. the canonical divisor) first.

        Note:
            This method is provisional and may change in future versions.

        Args:
            in_basis (bool): Whether to express the rays in the current basis of
                curves (see [`curve_basis`][cytools.calabiyau.CalabiYau.curve_basis]).
                If true, `exclude_origin` is ignored.
            exclude_origin (bool): Whether to drop the coordinate corresponding to
                the origin.
            format (str | None): If `"sparse"`, the matrix of rays is returned as a
                sparse matrix instead of a [`Cone`][cytools.cone.Cone]. This is
                useful at large $h^{1,1}$, where constructing a `Cone` requires
                densifying the rays. Otherwise, a `Cone` is returned.
            verbosity (int): The verbosity level.

        Returns:
            Cone | scipy.sparse.csr_matrix: The Mori cone cap, or the sparse matrix
                whose rows are its rays if `format="sparse"`.

        Examples:
            We construct a Calabi-Yau hypersurface and find its Mori cone cap. In
            this case it agrees with the toric Mori cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> cy.mori_cone_cap(in_basis=True)
            A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
            >>> cy.mori_cone_cap(in_basis=True).rays()
            array([[ 1, -3],
                   [ 1,  0],
                   [ 0,  1]])
        """
        from cytools.ntfe.ntfe import cone_of_permissible_heights

        triang = self.ambient_variety().triangulation()
        rays = cone_of_permissible_heights(
            triang.restrict(restrict_dim=2, as_poly=True),
            npts=len(triang.labels),
            poly=self.polytope(),
            require_star=True,
            as_cone=False,
            verbosity=verbosity,
        )
        # the CPL blocks are int16; a basis change would overflow that
        rays = rays.astype(int)

        if in_basis:
            basis = self.divisor_basis()
            rays = rays.dot(basis.T) if np.ndim(basis) == 2 else rays[:, basis]
        elif exclude_origin:
            rays = rays[:, 1:]

        if format == "sparse":
            return rays
        return Cone(rays.toarray(), check=False)




class Invariants:
    """A container for Gopakumar-Vafa (GV) or Gromov-Witten (GW) invariants.

    Objects of this class are generally obtained with
    [`CalabiYau.compute_gvs`][cytools.calabiyau.CalabiYau.compute_gvs] or
    [`CalabiYau.compute_gws`][cytools.calabiyau.CalabiYau.compute_gws]. They
    store a map from curve charges to the corresponding invariants, along with
    information about how they were computed.

    Args:
        invariant_type (str): The type of invariants, either `"gv"` or `"gw"`.
        charge2invariant (dict[tuple[int, ...], int] | ArrayLike): The
            invariants, either as a dictionary mapping charges to invariants
            (`"dok"` format) or as a list of rows of the form
            `[q_1, ..., q_n, invariant]` (`"coo"` format).
        grading_vec (ArrayLike | None): The grading vector used in the
            computation, which defines the degree of a charge.
        cutoff (int | None): The maximum degree used in the computation, if
            any. All invariants of charges up to this degree are assumed to be
            included.
        calabiyau (CalabiYau | None): The Calabi-Yau that the invariants
            correspond to.
        basis (np.ndarray | None): A matrix used to change the basis of the
            charges, where each charge $q$ is replaced by `basis @ q`. The
            grading vector is not transformed.

    Raises:
        ValueError: If `invariant_type` is not `"gv"` or `"gw"`.

    Examples:
        We compute the GV invariants of a Calabi-Yau hypersurface up to degree
        2 and look at some of them.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> t = p.triangulate()
        >>> cy = t.get_cy()
        >>> gvs = cy.compute_gvs(max_deg=2)
        >>> gvs.size
        5
        >>> gvs.gv((1, -3))
        3
        >>> gvs.gv((5, 5)) is None  # Not computed
        True
    """

    def __init__(
        self,
        invariant_type: str,
        charge2invariant,
        grading_vec: "ArrayLike" = None,
        cutoff: int = None,
        calabiyau: "CalabiYau" = None,
        basis: "ArrayLike" = None,
    ):
        if invariant_type not in ["gv", "gw"]:
            raise ValueError(f"Invariant type '{invariant_type}' not recognized")
        self._type = invariant_type

        if isinstance(charge2invariant, dict):
            self._charge2invariant = charge2invariant
        else:
            # assume charge2invariant is of coo format
            self._charge2invariant = {tuple(r[:-1]): r[-1] for r in charge2invariant}

        self._grading_vec = grading_vec
        self._cutoff = cutoff
        self._cy = calabiyau
        self._basis = basis

        if basis is not None:
            charges = self._charge2invariant.keys()
            invariants = self._charge2invariant.values()
            charges = np.array(list(charges)) @ basis.T
            self._charge2invariant = {
                tuple(r): _gv for r, _gv in zip(charges, invariants)
            }

    # standard methods
    # ----------------
    def __str__(self):
        """Return a string describing the invariants.

        The string includes the number and type of invariants, the Calabi-Yau (if
        known), the grading vector, and the degree cutoff.

        Returns:
            str: A string describing the invariants.
        """
        out_str = f"N={self.size} {self._type}-invariants calculated for a "

        # CY
        if self._cy is not None:
            out_str += str(self._cy)[2:]
        else:
            out_str += "unknown CY"

        # grading vector
        out_str += " using grading_vec="

        if self._grading_vec is not None:
            out_str += str(self._grading_vec)
        else:
            out_str += "?"

        # cutoff
        out_str += " and cutoff="

        if self._cutoff is not None:
            out_str += str(self._cutoff)
        else:
            out_str += "?"

        return out_str

    def __repr__(self):
        """Return a string describing the invariants.

        This is the same as `str(self)`.

        Returns:
            str: A string describing the invariants.
        """
        return str(self)

    # getters
    # -------
    @property
    def grading_vec(self):
        """Return a copy of the grading vector used in the computation.

        Returns:
            np.ndarray | None: The grading vector, or `None` if it is not known.
        """
        if self._grading_vec is None:
            return None
        else:
            return self._grading_vec.copy()

    @property
    def cutoff(self):
        """Return the degree cutoff used in the computation.

        Returns:
            int | None: The degree cutoff, or `None` if it is not known (e.g. if
                the invariants were computed with `min_points` or `target_points`).
        """
        return self._cutoff

    def charges(self, by_deg=False, as_np_arr=False):
        """Return the charges of the stored invariants.

        Args:
            by_deg (bool): Whether to group the charges by degree, as determined by
                the grading vector.
            as_np_arr (bool): Whether to return the charges as numpy arrays instead
                of tuples.

        Returns:
            list[tuple[int, ...]] | np.ndarray | dict: If `by_deg` is false, a list
                of charges (or a 2D array if `as_np_arr` is true), in the same
                order as the invariants. If `by_deg` is true, a dictionary mapping
                each degree, in ascending order, to a set of charges (or a 2D array
                if `as_np_arr` is true).

        Raises:
            ValueError: If `by_deg` is true and the grading vector is not known.

        Examples:
            We compute some GV invariants and group their charges by degree.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> gvs = cy.compute_gvs(max_deg=2)
            >>> {int(d): len(c) for d, c in gvs.charges(by_deg=True).items()}
            {1: 2, 2: 3}
            >>> gvs.charges(by_deg=True)[1] == {(0, 1), (1, -3)}
            True
        """
        out = None
        if not by_deg:
            out = list(self._charge2invariant.keys())
            if as_np_arr:
                return np.asarray(out)
            else:
                return out

        # group by degree
        if self._grading_vec is None:
            raise ValueError(
                "Can't group charges by degree if no grading vector is known"
            )
        return _group_by_deg(
            self._charge2invariant.keys(),
            self._grading_vec,
            as_np_arr=as_np_arr,
        )

    def cone(self):
        """Return the cone generated by the charges of the stored invariants.

        For GV invariants, this is the "GV cone", i.e. the cone generated by the
        charges of the computed invariants.

        Returns:
            Cone: The cone generated by the charges.

        Examples:
            We compute some GV invariants and find the cone generated by their
            charges.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> gvs = cy.compute_gvs(max_deg=2)
            >>> gvs.cone()
            A 2-dimensional rational polyhedral cone in RR^2 generated by 5 rays
        """
        return Cone(self.charges(as_np_arr=True))

    @property
    def gvs(self) -> list:
        """Return the GV invariants.

        Returns:
            list | None: The GV invariants, in the same order as
                [`charges`][cytools.calabiyau.Invariants.charges], or `None` if the
                stored invariants are GW invariants.
        """
        if self._type == "gv":
            return list(self._charge2invariant.values())
        else:
            return None

    @property
    def gws(self) -> list:
        """Return the GW invariants.

        Returns:
            list | None: The GW invariants, in the same order as
                [`charges`][cytools.calabiyau.Invariants.charges], or `None` if the
                stored invariants are GV invariants.
        """
        if self._type == "gw":
            return list(self._charge2invariant.values())
        else:
            return None

    def invariant(self, charge, check_deg=True):
        """Return the invariant for a specific charge.

        Args:
            charge (ArrayLike): The charge of interest.
            check_deg (bool): Whether to return 0 for charges that are not stored
                but whose degree is not greater than the cutoff. This is only done
                if the cutoff and the grading vector are known.

        Returns:
            int | Fraction | None: The invariant. If the charge is not stored, 0 is
                returned if `check_deg` is true and the degree of the charge is not
                greater than the cutoff, and `None` otherwise.

        Examples:
            We compute some GV invariants and look them up.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> cy = t.get_cy()
            >>> gvs = cy.compute_gvs(max_deg=2)
            >>> gvs.invariant((0, 1))
            540
            >>> gvs.invariant((2, -7))  # Not stored, but degree 1 <= 2
            0
            >>> gvs.invariant((2, -5)) is None  # Not stored, and degree 3 > 2
            True
        """
        out = self._charge2invariant.get(tuple(charge), None)

        # NOTE: the degree check is only meaningful when a degree cutoff (and a
        # grading vector) is known. This is not the case when the invariants
        # were computed via min_points/target_points/mcap_generators, in which
        # case we simply cannot tell whether the invariant vanishes.
        if (
            check_deg
            and (out is None)
            and (self._cutoff is not None)
            and (self._grading_vec is not None)
            and np.dot(charge, self._grading_vec) <= self._cutoff
        ):
            out = 0  # deg<=cutoff but not in dict -> 0 GV

        return out

    def gv(self, charge, check_deg=True):
        """Return the GV invariant for a specific charge.

        This is the same as [`invariant`][cytools.calabiyau.Invariants.invariant],
        but it returns `None` if the stored invariants are GW invariants.

        Args:
            charge (ArrayLike): The charge of interest.
            check_deg (bool): Whether to return 0 for charges that are not stored
                but whose degree is not greater than the cutoff. This is only done
                if the cutoff and the grading vector are known.

        Returns:
            int | None: The GV invariant, or `None` if it is not known.
        """
        if self._type == "gv":
            return self.invariant(charge, check_deg)
        else:
            return None

    def gw(self, charge, check_deg=True):
        """Return the GW invariant for a specific charge.

        This is the same as [`invariant`][cytools.calabiyau.Invariants.invariant],
        but it returns `None` if the stored invariants are GV invariants.

        Args:
            charge (ArrayLike): The charge of interest.
            check_deg (bool): Whether to return 0 for charges that are not stored
                but whose degree is not greater than the cutoff. This is only done
                if the cutoff and the grading vector are known.

        Returns:
            Fraction | None: The GW invariant, or `None` if it is not known.
        """
        if self._type == "gw":
            return self.invariant(charge, check_deg)
        else:
            return None

    # old formats
    # -----------
    @property
    def dok(self):
        """Return the invariants in the "dok" (Dictionary Of Keys) format.

        Returns:
            dict[tuple[int, ...], int | Fraction]: A copy of the dictionary mapping
                charges to invariants.
        """
        return self._charge2invariant.copy()

    @property
    def coo(self):
        """Return the invariants in the "coo" (COOrdinate) format.

        Returns:
            np.ndarray: An array with rows of the form `[q_1, ..., q_n, invariant]`.
        """
        return np.array([list(k) + [v] for k, v in self._charge2invariant.items()])

    # misc others
    # -----------
    @property
    def size(self):
        """Return the number of stored invariants.

        Returns:
            int: The number of stored charges and invariants.
        """
        return len(self._charge2invariant)


def _group_by_deg(
    charges: "Iterable", grading_vec: "ArrayLike", as_np_arr: bool = False
):
    r"""Group charges by their degrees.

    Args:
        charges (Iterable): The charges.
        grading_vec (ArrayLike): The grading vector $g$. The degree of a charge
            $q$ is $q \cdot g$.
        as_np_arr (bool): Whether to collect the charges of each degree into a
            2D numpy array instead of a set of tuples.

    Returns:
        dict: A dictionary mapping each degree, in ascending order, to the
            charges of that degree.
    """
    charges = np.asarray(sorted(charges))
    degs = charges @ grading_vec

    # sort degrees
    sort_inds = np.argsort(degs)
    charges, degs = charges[sort_inds], degs[sort_inds]

    # organize as dict
    charges_by_deg = dict()
    if as_np_arr:
        for charge, deg in zip(charges, degs):
            charges_by_deg[deg] = charges_by_deg.get(deg, []) + [charge]

        # map to numpy arrays
        for k, v in charges_by_deg.items():
            charges_by_deg[k] = np.asarray(v, dtype=int)
    else:
        for charge, deg in zip(charges, degs):
            charges_by_deg[deg] = charges_by_deg.get(deg, set())
            charges_by_deg[deg].add(tuple(charge))

    return charges_by_deg
