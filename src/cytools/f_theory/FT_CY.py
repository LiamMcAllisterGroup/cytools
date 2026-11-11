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


"""Orientifolds of Calabi-Yau hypersurfaces and their F-theory uplifts.

This module provides the [`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold]
class, which describes O3/O7 orientifolds of Calabi-Yau hypersurfaces in toric
varieties, and the [`F_Theory_Uplift`][cytools.f_theory.FT_CY.F_Theory_Uplift] class,
which describes their F-theory uplifts as toric complete intersections. It also provides
functions to generate them from the polytope databases and to load a precomputed dataset
of uplifts.
"""

# 'standard' imports
import warnings

# 3rd party imports
import numpy as np
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# CYTools imports
from cytools import Polytope, fetch_polytopes
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan
from cytools.f_theory import Uplift_functions as UF

__all__ = [
    "CY_orientifold",
    "F_Theory_Uplift",
    "fetch_orientifolds",
    "fetch_F_Theory_uplifts",
    "fetch_nef_partition_uplifts",
]


class CY_orientifold:
    r"""O3/O7 orientifold of a Calabi-Yau hypersurface in a toric variety.

    The orientifold involution $\sigma$ is the toric $\mathbb{Z}_2$ action given by the
    torus element $\xi/2$, where $\xi$ is an integer vector in the lattice $N$ of the
    ambient fan. The quotient of the ambient variety by this action is a toric orbifold,
    whose rays are the images of the original rays in the refined lattice
    $N + \mathbb{Z}\,\xi/2$, rescaled to be primitive. The orientifold is described by
    this orbifold together with the O3/O7 line bundle $L$ on it, whose generic section
    cuts out the quotient $B = X/\sigma$ of the Calabi-Yau hypersurface $X$, i.e. the
    base of the F-theory uplift (see
    [`F_Theory_Uplift`][cytools.f_theory.FT_CY.F_Theory_Uplift]).

    Toric rays are referred to by one-indexed labels, following the CYTools convention.
    Note that the orbifold rays need not be in the same order as the rays of the
    Calabi-Yau ambient fan.

    Args:
        fan_polytope_or_points (Polytope | VectorConfiguration | Fan | ArrayLike): The
            toric data of the ambient variety of the Calabi-Yau hypersurface. If a
            `Polytope` is given, its nonzero lattice points (including the points
            interior to facets) are used as the rays. A `VectorConfiguration`, an array,
            or a list of lists specifies the rays directly. If a `Fan` is given, its
            rays and cones define the ambient fan; otherwise, the fan is obtained by
            triangulating the rays (see `CY_ambient_toric_fan`).
        xi (ArrayLike | None): The integer vector $\xi$ defining the $\mathbb{Z}_2$
            action. It can only be omitted if the input is a trilayer polytope, in which
            case it is set to minus the distinguished vertex whose GLSM charges are half
            of the anticanonical class.
        resolve_A1_singularities (bool): Whether to resolve the A1 singularities of the
            orbifold along codimension-two fixed loci by adding blowup rays. This only
            has an effect when the input is a `Fan` and the orbifold fan was not
            replaced by the nef-decomposition refinement.
        construct_nef_decomposition (bool): Whether to attempt, for regular
            orientifolds, to replace the orbifold fan by a refinement of the normal fan
            of the Newton polytopes of $L$ and $6(-K - L)$, where $-K$ is the
            anticanonical divisor of the orbifold, so that these divisors become Cartier
            and nef. See `yields_nef_decomposition`.

    Raises:
        ValueError: If `xi` is not specified and the input is not a trilayer polytope.
        TypeError: If `fan_polytope_or_points` has an unsupported type.

    Examples:
        We construct the orientifold of the degree-8 hypersurface in
        $\mathbb{P}_{11114}$. Its polytope is trilayer, so `xi` can be inferred.

        >>> from cytools import Polytope
        >>> from cytools.f_theory import CY_orientifold
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-4]])
        >>> o = CY_orientifold(p)
        >>> o.xi()
        array([ 0,  0,  0, -1])
        >>> o.is_regular()
        True
        >>> o.line_bundle()
        array([0, 8, 0, 0, 0, 1])
        >>> sorted(o.o_planes())
        [(2,), (6,)]
    """

    def __init__(
        self,
        fan_polytope_or_points=None,
        xi=None,
        resolve_A1_singularities=True,
        construct_nef_decomposition=True,
    ):
        self.__p = None
        self.__toric_fan = None
        self.__xi = None
        self.__orbifold_toric_fan = None
        self.__orbifold_line_bundle = None
        self.__yields_nef_decomposition = None
        self.__regular = None
        self.__orbifold_pts = None
        self.__CY_ambient_pts = None
        self.__intersection_numbers_orbifold = None
        self.__normal_fan = None
        self.__Newton_Polytope = None
        self.__NHC_labels = None
        self.ambient_triangulation = False
        self._multiplier = 6
        self.__o_planes = None
        match fan_polytope_or_points:
            case Polytope():
                if xi is None:
                    if fan_polytope_or_points.is_trilayer():
                        xi = -UF.find_trilayer_vertex_polytope(fan_polytope_or_points)
                    else:
                        raise ValueError(
                            "If polytope is not trilayer, xi has to be defined"
                        )
                self.__p = fan_polytope_or_points
                ind = UF.get_index(
                    fan_polytope_or_points.points(),
                    np.zeros(len(fan_polytope_or_points.vertices()[0])),
                )
                if len(ind) > 0:
                    pts = np.delete(fan_polytope_or_points.points(), ind[0], axis=0)
                else:
                    pts = fan_polytope_or_points.points()
            case VectorConfiguration():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
            case Fan():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
                self.__toric_fan = fan_polytope_or_points
                self.ambient_triangulation = True
            case np.ndarray():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points
            case list():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = np.array(fan_polytope_or_points, dtype=int)
            case _:
                raise TypeError("Unsupported input type")
        self.__CY_ambient_pts = pts
        self.__dim = np.linalg.matrix_rank(pts, tol=1e-6)
        self.__xi = xi

        orbifold_pts, rescalings = UF.toric_orbifold(pts_CY_ambient=pts, q=xi)
        orbifold_line_bundle = UF.O3O7_line_bundle(
            pts_CY_ambient=pts, q=xi, rescalings=rescalings
        )
        if orbifold_line_bundle is None:
            self.__regular = False
        else:
            self.__regular = not bool(
                UF.generic_section_factorizes(orbifold_pts, orbifold_line_bundle)
            )

        if self.__regular:
            if construct_nef_decomposition:
                self.__yields_nef_decomposition = False
                KB_multiplier = UF.Newton_Polytope(
                    orbifold_pts,
                    self._multiplier
                    * (
                        np.ones(len(orbifold_line_bundle), dtype=int)
                        - orbifold_line_bundle
                    ),
                )
                if len(KB_multiplier.points()) > 0:
                    NP_orbifold = UF.Newton_Polytope(orbifold_pts, orbifold_line_bundle)
                    self.__Newton_Polytope = NP_orbifold
                    if NP_orbifold.minkowski_sum(KB_multiplier).dim() == self.__dim:
                        orbifold_pts_refined, line_bundles, normal_fan = UF.normal_fan(
                            [NP_orbifold, KB_multiplier],
                            np.array([self._multiplier, 1, self._multiplier]),
                            maximal_refinement=True,
                            triangulate_refinement=False,
                            return_unrefined_fan=True,
                        )
                        if orbifold_pts_refined is not None:
                            self.__yields_nef_decomposition = True
                            self.__normal_fan = normal_fan
                            orbifold_pts = orbifold_pts_refined
                            orbifold_line_bundle = line_bundles[:, 0]
                            self.ambient_triangulation = False
                            self.__NHC_labels = (
                                np.where(
                                    self._multiplier * (1 - orbifold_line_bundle)
                                    - line_bundles[:, 1]
                                    == 3
                                )[0]
                                + 1
                            )

        else:
            self.__yields_nef_decomposition = False

        self.__orbifold_pts = orbifold_pts
        self.__orbifold_line_bundle = orbifold_line_bundle
        if resolve_A1_singularities:
            if self.ambient_triangulation:
                self.__resolve_A1_singularities()

    def __repr__(self):
        """Return a short description of the object.

        Returns:
            str: The string `"A Calabi-Yau orientifold"`.
        """
        return "A Calabi-Yau orientifold"

    def __resolve_A1_singularities(self):
        r"""Resolve the orbifold A1 singularities along codimension-two fixed loci.

        Finds the two-dimensional cones of the Calabi-Yau ambient fan whose toric strata
        are fixed by the $\mathbb{Z}_2$ action. For each such cone with rays labeled $a$
        and $b$, the primitive vector along the sum of the orbifold rays $a$ and $b$ is
        added as a new ray, and the orbifold fan is refined accordingly. The line bundle
        is extended by the coefficients $(L_a + L_b - 1)/2$ of the exceptional divisors.
        The cached orbifold rays, line bundle, and fan are updated in place.
        """

        singular_two_cones = UF.Z2_fixed_locus(
            self.CY_ambient_toric_fan(), self.xi(), cone_dimension=2, denominator=2
        )
        orbifold_blowups = []
        if len(singular_two_cones) > 0:
            for c in singular_two_cones:
                vec = np.sum(self.vectors_orbifold(c), axis=0)
                vec_add = np.rint(vec / np.gcd.reduce(vec)).astype(int)
                orbifold_blowups.append(vec_add.tolist())
            orbifold_blowups = np.array(orbifold_blowups)
            fan = UF.refine_fan(self.orbifold_toric_fan(), orbifold_blowups)
            self.__orbifold_pts = fan.vectors()
            blow_up_weights = np.rint(
                np.array(
                    [
                        sum(self.line_bundle()[np.array(c) - 1]) - 1
                        for c in singular_two_cones
                    ]
                )
                / 2
            ).astype(int)
            self.__orbifold_line_bundle = np.concatenate(
                (self.line_bundle(), blow_up_weights), axis=0
            )
            self.__orbifold_toric_fan = fan
        return None

    def orbifold_toric_fan(self):
        """Return the toric fan of the orbifold ambient variety.

        If the orientifold was constructed from a `Fan` (and the nef-decomposition
        refinement was not applied), or if it does not yield a nef decomposition, the
        cones of the Calabi-Yau ambient fan are used on the orbifold rays. Otherwise,
        the fan is obtained from the normal fan (see `normal_fan`), which is made
        simplicial and refined to include all orbifold rays. If A1 singularities were
        resolved at construction, the refined fan is returned. The fan is computed on
        the first call and cached.

        Returns:
            Fan: The fan of the orbifold ambient variety, whose rays are
                `vectors_orbifold()`.
        """
        if self.__orbifold_toric_fan is None:
            if self.ambient_triangulation:
                self.__orbifold_toric_fan = Fan(
                    vc=VectorConfiguration(self.vectors_orbifold()),
                    cones=self.CY_ambient_toric_fan().cones(),
                )
            else:
                if self.yields_nef_decomposition():
                    nf = UF.make_simplicial(self.normal_fan())
                    VC = VectorConfiguration(self.vectors_orbifold())
                    inds = VC.vectors_to_labels(nf.vectors())
                    new_cones = set()
                    for c in nf.cones():
                        nc = tuple(sorted([inds[x - 1] for x in c]))
                        new_cones.add(nc)
                    new_fan = Fan(vc=VC, cones=new_cones)
                    self.__orbifold_toric_fan = UF.refine_fan(new_fan)
                else:
                    self.__orbifold_toric_fan = Fan(
                        VectorConfiguration(self.vectors_orbifold()),
                        cones=self.CY_ambient_toric_fan().cones(),
                    )
        return self.__orbifold_toric_fan

    def CY_ambient_toric_fan(self):
        """Return the toric fan of the Calabi-Yau ambient variety, before orbifolding.

        If a `Fan` was given at construction, it is returned. Otherwise, the fan is
        obtained on the first call by triangulating the vector configuration of the
        ambient rays with the default settings of
        [`VectorConfiguration.triangulate`][cytools.vector_config.vectorconfiguration.VectorConfiguration.triangulate],
        and cached.

        Returns:
            Fan: The fan of the Calabi-Yau ambient variety.
        """
        if self.__toric_fan is None:
            self.__toric_fan = VectorConfiguration(self.__CY_ambient_pts).triangulate()
        return self.__toric_fan

    def vectors_orbifold(self, c=None):
        """Return the rays of the orbifold ambient fan.

        These are the images of the Calabi-Yau ambient rays in the refined lattice, made
        primitive. If the nef-decomposition refinement was applied, they are instead the
        rays of the refined fan, and if A1 singularities were resolved, the blowup rays
        are appended at the end.

        Args:
            c (Iterable[int] | None): One-indexed labels of the rays to return. If not
                specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        if c is not None:
            return self.__orbifold_pts[np.array(c) - 1]
        else:
            return self.__orbifold_pts

    def vectors_CY_ambient(self, c=None):
        """Return the rays of the Calabi-Yau ambient fan, before orbifolding.

        Args:
            c (Iterable[int] | None): One-indexed labels of the rays to return. If not
                specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        if c is not None:
            return self.__CY_ambient_pts[np.array(c) - 1]
        else:
            return self.__CY_ambient_pts

    def xi(self):
        r"""Return the vector $\xi$ defining the $\mathbb{Z}_2$ action.

        Returns:
            np.ndarray | list: The vector $\xi$, as given at construction or inferred
                from a trilayer polytope. The involution is the torus element $\xi/2$.
        """
        return self.__xi

    def is_regular(self):
        r"""Return whether the orientifold is regular.

        The orientifold is regular if the Newton polytope of the Calabi-Yau hypersurface
        contains a monomial $m$ with $\langle m, \xi\rangle$ odd, so that the O3/O7 line
        bundle $L$ is defined, and the generic section of $L$ on the orbifold does not
        factorize, i.e. no toric coordinate divides all the sections of $L$.

        Returns:
            bool: Whether the orientifold is regular.
        """
        return self.__regular

    def line_bundle(self):
        """Return the O3/O7 line bundle $L$ on the orbifold ambient variety.

        The generic section of $L$ cuts out the quotient of the Calabi-Yau hypersurface
        by the involution, i.e. the base of the F-theory uplift.

        Returns:
            np.ndarray | None: The coefficients of $L$ in the basis of prime toric
                divisors, one per orbifold ray (see `vectors_orbifold`). `None` if the
                line bundle could not be defined (see `is_regular`).
        """
        return self.__orbifold_line_bundle

    def dim(self):
        """Return the dimension of the Calabi-Yau ambient variety.

        This is computed as the rank of the matrix of ambient rays, and it is also the
        dimension of the orbifold ambient variety.

        Returns:
            int: The dimension of the ambient variety.
        """
        return self.__dim

    def normal_fan(self):
        """Return the normal fan used for the nef-decomposition construction.

        This is the normal fan of the Minkowski sum of the Newton polytopes of $L$ and
        $6(-K - L)$, where $-K$ is the anticanonical divisor of the orbifold. If the
        nef-decomposition refinement was applied at construction, the unrefined normal
        fan computed there is returned. Otherwise, it is computed on the first call and
        cached.

        Returns:
            Fan: The normal fan.
        """
        if self.__normal_fan is None:
            self.__normal_fan = UF.normal_fan(
                [
                    self.Newton_Polytope(),
                    UF.Newton_Polytope(
                        self.vectors_orbifold(),
                        self._multiplier * (1 - self.line_bundle()),
                    ),
                ]
            )[0]
        return self.__normal_fan

    def ambient_dim(self):
        """Return the dimension of the lattice containing the orbifold rays.

        Returns:
            int: The number of coordinates of each orbifold ray.
        """
        return len(self.vectors_orbifold()[0])

    def Newton_Polytope(self):
        """Return the Newton polytope of the O3/O7 line bundle $L$.

        This is the polytope of monomial sections of $L$ on the orbifold ambient
        variety. It is computed on first use and cached.

        Returns:
            HPolytope: The Newton polytope of $L$.
        """
        if self.__Newton_Polytope is None:
            self.__Newton_Polytope = UF.Newton_Polytope(
                self.vectors_orbifold(), self.line_bundle()
            )
        return self.__Newton_Polytope

    def yields_nef_decomposition(self):
        r"""Return whether the orientifold data yields a Cartier and nef decomposition.

        This is always false for non-regular orientifolds. If the nef-decomposition
        refinement was attempted at construction, this returns whether it succeeded.
        Otherwise, the result is computed on the first call and cached: if the
        orientifold was constructed from a `Fan`, it checks that $L$ and
        $6(-K - L) - 3\sum_{i} D_i$, where the sum runs over the non-Higgsable-cluster
        divisors (see `NHC`), are both Cartier and nef on the orbifold fan; otherwise,
        it checks that all rays of `normal_fan` are among the orbifold rays.

        Returns:
            bool: Whether the orientifold yields a nef decomposition.
        """
        if self.__yields_nef_decomposition is None:
            if self.ambient_triangulation:
                NHClb = np.zeros(len(self.line_bundle()))
                NHC_inds = self.NHC(as_labels=True) - 1
                NHClb[NHC_inds] = 1
                lb = self._multiplier * (1 - self.line_bundle()) - np.rint(
                    (self._multiplier / 2) * NHClb
                ).astype(int)
                self.__yields_nef_decomposition = (
                    UF.is_Cartier(self.orbifold_toric_fan(), lb)[0]
                    and UF.is_nef(self.orbifold_toric_fan(), lb)
                    and UF.is_Cartier(self.orbifold_toric_fan(), self.line_bundle())[0]
                    and UF.is_nef(self.orbifold_toric_fan(), self.line_bundle())
                )
            else:
                self.__yields_nef_decomposition = UF.contains_rows(
                    self.vectors_orbifold(), self.normal_fan().vectors()
                )
        return self.__yields_nef_decomposition

    def polytope(self):
        """Return the polytope used to construct the orientifold.

        Returns:
            Polytope | None: The polytope given at construction, or `None` if the
                orientifold was constructed from a fan, a vector configuration, an
                array, or a list of points.
        """
        return self.__p

    def intersection_numbers_orbifold(self):
        """Return the intersection numbers of the orbifold ambient variety.

        They are computed with
        [`Fan.intersection_numbers`][cytools.vector_config.fan.Fan.intersection_numbers]
        on the first call and cached.

        Returns:
            dict[tuple[int, ...], float]: The intersection numbers, keyed by sorted
                tuples of divisor labels. Labels are one-indexed orbifold ray labels,
                and the label 0 refers to the canonical divisor.
        """
        if self.__intersection_numbers_orbifold is None:
            self.__intersection_numbers_orbifold = (
                self.orbifold_toric_fan().intersection_numbers()
            )
        return self.__intersection_numbers_orbifold

    def NHC(self, as_labels=False):
        """Return the toric divisors that support non-Higgsable clusters.

        These are the orbifold toric divisors along which the generic section of the
        divisor with coefficients $2(1 - L_i)$, i.e. $2(-K - L)$, vanishes to order
        exactly one. If the nef-decomposition refinement was applied at construction,
        they are instead determined there, as the divisors along which the generic
        section of $6(-K - L)$ vanishes to order exactly three. The result is computed
        on first use and cached.

        Args:
            as_labels (bool): Whether to return the one-indexed labels of the orbifold
                rays instead of the rays themselves.

        Returns:
            np.ndarray: The labels of the non-Higgsable-cluster divisors, or the
                corresponding orbifold rays, one per row. Empty if there are none.
        """
        if self.__NHC_labels is None:
            sections_NP2 = UF.sections(
                self.vectors_orbifold(), 2 * (1 - self.line_bundle())
            )
            if len(sections_NP2) > 0:
                self.__NHC_labels = np.where(np.min(sections_NP2, axis=1) == 1)[0] + 1
            else:
                self.__NHC_labels = np.array([])
        if as_labels:
            return self.__NHC_labels
        else:
            if len(self.__NHC_labels) == 0:
                return np.array([])
            return self.vectors_orbifold(self.__NHC_labels)

    def o_planes(self):
        r"""Return the ambient cones whose toric strata are fixed by $\sigma$.

        A cone of the Calabi-Yau ambient fan that is not maximal is included if the sum
        of its rays is congruent to $\xi$ modulo 2. The corresponding toric strata are
        pointwise fixed by the $\mathbb{Z}_2$ action, and their intersections with the
        Calabi-Yau hypersurface give the O-planes. The result is computed on the first
        call and cached.

        Returns:
            list[tuple[int, ...]]: The fixed cones, as tuples of one-indexed ray labels
                of the Calabi-Yau ambient fan, in no particular order.
        """
        if self.__o_planes is None:
            self.__o_planes = UF.Z2_fixed_locus(self.CY_ambient_toric_fan(), self.__xi)
        return self.__o_planes


class F_Theory_Uplift:
    r"""F-theory uplift of an O3/O7 Calabi-Yau orientifold.

    The uplift is an elliptic fibration over the base $B = X/\sigma$ of a
    [`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold], described as a complete
    intersection of two equations in a toric variety whose dimension is two more than
    that of the orbifold ambient variety: the base equation, a section of the O3/O7 line
    bundle $L$, and the Weierstrass equation in the $\mathbb{P}^2_{231}$ fiber, of class
    $3D_x$. The singular uplift ambient fan has the rays $(v_i, 0, 1 - L_i)$ for the
    orbifold rays $v_i$, together with the fiber rays $x = (0, 3, 1)$,
    $y = (0, -2, -1)$, and $z = (0, 0, 1)$, so that the fiber is twisted by the
    anticanonical class of the base. The fibration is then resolved over the
    non-Higgsable clusters by adding two blowup rays for each non-Higgsable-cluster
    divisor (see `blowups`).

    The class gives access to the singular and resolved uplift ambient fans, the base
    and Weierstrass divisors and, when these form a nef partition, the associated
    Batyrev-Borisov data: Cayley polytopes, the dual nef partition, and the Hodge
    numbers of the Calabi-Yau fourfold. Methods with the suffix `_N` refer to the nef
    partition on the resolved uplift ambient variety (in the N lattice), and methods
    with the suffix `_M` refer to the dual data in the M lattice. Toric rays are
    referred to by one-indexed labels.

    Args:
        orientifold_or_points (CY_orientifold | Polytope | Fan | ArrayLike): The
            orientifold to uplift, or toric data from which to construct one. Any input
            accepted by [`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold]
            (including a `VectorConfiguration`) is allowed.
        xi (ArrayLike | None): The integer vector defining the $\mathbb{Z}_2$ action.
            Only used when toric data are given, in which case it can only be omitted
            for trilayer polytopes.
        resolve_A1_singularities (bool): Whether to resolve A1 singularities when
            constructing the orientifold from toric data. Note that, unlike for
            [`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold], this is disabled
            by default.
        construct_nef_decomposition (bool): Whether to attempt to construct a Cartier
            and nef decomposition when constructing the orientifold from toric data.

    Raises:
        ValueError: If toric data are given without `xi` and they are not a trilayer
            polytope.
        TypeError: If `orientifold_or_points` has an unsupported type.

    Examples:
        We construct the F-theory uplift of the orientifold of the degree-8
        hypersurface in $\mathbb{P}_{11114}$, which is an elliptic fibration over
        $\mathbb{P}^3$, and compute its Hodge numbers.

        >>> from cytools import Polytope
        >>> from cytools.f_theory import F_Theory_Uplift
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-4]])
        >>> F = F_Theory_Uplift(p)
        >>> F.is_nef_partition()
        True
        >>> F.h11(), F.h21(), F.h31()
        (2, 0, 3878)
        >>> F.chi()
        23328
    """

    def __init__(
        self,
        orientifold_or_points=None,
        xi=None,
        resolve_A1_singularities=False,
        construct_nef_decomposition=True,
    ):
        self.__blowups = None
        self.__Cayley_M = None
        self.__Cayley_N = None
        self.__pol_N_sum = None
        self.__pol_N_conv = None
        self.__pol_M_conv = None
        self.__pol_M_sum = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = None
        self.__pol_B_M = None
        self.__pol_W_M = None
        self.__pol_W_N = None
        self.__pol_B_N = None
        self.__NHC_labels = None
        self.__M_conv_toric_fan = None
        self.__intersection_numbers_M_conv = None
        self.__intersection_numbers_singular_uplift = None
        self.__intersection_numbers_smooth_uplift = None
        self.__LBW_N = None
        self.__LBB_N = None
        self.__LBW_M = None
        self.__LBB_M = None
        self.__basis_homology_M = None
        self.__basis_homology_N = None
        self.__normal_fan = None
        self.__singular_uplift_toric_fan = None
        self.__pts_singular_uplift = None
        self.__pts_smooth_uplift = None
        self.__smooth_uplift_toric_fan = None
        self.x = None
        self.y = None
        self.z = None
        self.__h11 = None
        self.__h31 = None
        self.__h21 = None
        self.__chi = None
        self.__is_nef_partition = None
        self.__is_partition = None
        self.__is_nef_decomposition = None
        self.__CY_orientifold = None

        match orientifold_or_points:
            case CY_orientifold():
                self.__CY_orientifold = orientifold_or_points
            case _:
                self.__CY_orientifold = CY_orientifold(
                    orientifold_or_points,
                    xi=xi,
                    resolve_A1_singularities=resolve_A1_singularities,
                    construct_nef_decomposition=construct_nef_decomposition,
                )

        if not self.is_regular():
            self.__is_partition = False
            self.__is_nef_partition = False

    def __set_divisor_representations(self):
        r"""Compute the base and Weierstrass divisors of the resolved uplift.

        The base divisor has the coefficients $L_i$ on the orbifold rays and, for each
        non-Higgsable-cluster divisor $D$ with $L_D \neq 0$, the coefficients $L_D$ and
        $2L_D$ on the two blowup rays over $D$. The Weierstrass divisor is $3D_x$. The
        method then looks for linearly equivalent representatives whose coefficients are
        0 or 1 and which together partition the rays, i.e. a partition of the
        anticanonical divisor. If they exist, they are stored. Otherwise, the base
        divisor is kept and the Weierstrass divisor is replaced by the anticanonical
        divisor minus the base divisor. Whether a partition was found is stored for
        `is_partition`.
        """
        n_sing = len(self.vectors_orbifold()) + 3
        n_smooth = len(self.vectors_smooth_uplift_ambient())
        LBB_N = np.zeros(n_smooth, dtype=int)
        LBW_N = np.zeros(n_smooth, dtype=int)
        LBB_N[: n_sing - 3] = self.line_bundle_orbifold()
        LBW_N[n_sing - 3] = 3
        NHC = self.NHC(as_labels=True) - 1
        if len(NHC) > 0:
            cont = np.where(self.line_bundle_orbifold() != 0)[0]
            nhc_cont = np.where(np.isin(NHC, cont))[0]
            for n in nhc_cont:
                contribution_n = self.line_bundle_orbifold()[NHC[n]]
                LBB_N[2 * n + n_sing] = 1 * contribution_n
                LBB_N[2 * n + 1 + n_sing] = 2 * contribution_n
        is_part = UF.compute_partition(
            [LBB_N, LBW_N], self.vectors_smooth_uplift_ambient()
        )
        if is_part[0]:
            self.__LBB_N = is_part[1][0]
            self.__LBW_N = is_part[1][1]
        else:
            sta = UF.sums_to_anticanonical(
                self.vectors_smooth_uplift_ambient(), LBB_N, LBW_N
            )
            self.__LBB_N = LBB_N
            self.__LBW_N = self.vectors_smooth_uplift_ambient() @ sta[1] + LBW_N
        self.__is_partition = is_part[0]

    def __repr__(self):
        """Return a short description of the object.

        Returns:
            str: The string `"An F-theory uplift of a Calabi-Yau orientifold"`.
        """

        return "An F-theory uplift of a Calabi-Yau orientifold"

    def orientifold(self):
        """Return the orientifold from which the F-theory uplift is constructed.

        Returns:
            CY_orientifold: The underlying orientifold.
        """

        return self.__CY_orientifold

    def is_nef_partition(self):
        """Return whether the base and Weierstrass divisors form a nef partition.

        This is the case if they form a partition of the anticanonical divisor (see
        `is_partition`) and a nef decomposition (see `is_nef_decomposition`). Uplifts of
        non-regular orientifolds are never nef partitions. The result is computed on the
        first call and cached.

        Returns:
            bool: Whether the uplift defines a nef partition.
        """
        if self.__is_nef_partition is None:
            if self.is_nef_decomposition():
                if self.is_partition():
                    self.__is_nef_partition = True
                    return True
            self.__is_nef_partition = False
        return self.__is_nef_partition

    def is_partition(self):
        r"""Return whether the base and Weierstrass divisors form a partition.

        This is the case if they are linearly equivalent to divisors of the form
        $\sum_{i \in I_B} D_i$ and $\sum_{i \in I_W} D_i$, where $I_B$ and $I_W$ are
        disjoint sets whose union contains all rays of the resolved uplift ambient fan,
        so that the two divisors sum to the anticanonical divisor. Uplifts of
        non-regular orientifolds are never partitions. The result is computed on the
        first call and cached.

        Returns:
            bool: Whether the uplift defines a partition.
        """
        if self.__is_partition is None:
            self.__set_divisor_representations()
        return self.__is_partition

    def smooth_uplift_ambient_toric_fan(self):
        """Return the toric fan of the resolved uplift ambient variety.

        Starting from the singular uplift ambient fan (see
        `singular_uplift_ambient_toric_fan`), for each non-Higgsable-cluster divisor
        $D$, every maximal cone that contains the rays of $D$, $x$, and $y$ is replaced
        by five cones involving the two blowup rays over $D$ (see `blowups`). The fan is
        computed on the first call and cached.

        Returns:
            Fan: The fan of the resolved uplift ambient variety, whose rays are
                `vectors_smooth_uplift_ambient()`.
        """

        if self.__smooth_uplift_toric_fan is None:
            n_sing_uplift = len(self.vectors_singular_uplift_ambient())
            xlabel = n_sing_uplift - 2
            ylabel = n_sing_uplift - 1
            ct = 0
            vc_smooth = VectorConfiguration(self.vectors_smooth_uplift_ambient())
            new_fan = Fan(
                vc=vc_smooth, cones=self.singular_uplift_ambient_toric_fan().cones()
            )

            for xxx in self.NHC_singular_uplift(as_labels=True):
                all_cones = set(new_fan.cones())
                for link in new_fan.link((xxx, xlabel, ylabel)):
                    all_cones.discard(tuple(sorted((xxx, xlabel, ylabel) + link)))
                    all_cones.add(
                        tuple(
                            sorted(
                                link
                                + (
                                    xxx,
                                    n_sing_uplift + 2 * ct + 1,
                                    n_sing_uplift + 2 * ct + 2,
                                )
                            )
                        )
                    )
                    all_cones.add(
                        tuple(sorted(link + (xxx, xlabel, n_sing_uplift + 2 * ct + 2)))
                    )
                    all_cones.add(
                        tuple(sorted(link + (xxx, ylabel, n_sing_uplift + 2 * ct + 1)))
                    )
                    all_cones.add(
                        tuple(
                            sorted(link + (xlabel, ylabel, n_sing_uplift + 2 * ct + 1))
                        )
                    )
                    all_cones.add(
                        tuple(
                            sorted(
                                link
                                + (
                                    xlabel,
                                    n_sing_uplift + 2 * ct + 1,
                                    n_sing_uplift + 2 * ct + 2,
                                )
                            )
                        )
                    )
                ct += 1
                new_fan = Fan(vc=vc_smooth, cones=all_cones)
            self.__smooth_uplift_toric_fan = new_fan
        return self.__smooth_uplift_toric_fan

    def singular_uplift_ambient_toric_fan(self):
        r"""Return the toric fan of the singular uplift ambient variety.

        Each maximal cone of the orbifold fan is combined with each of the three
        two-dimensional cones $\{x, y\}$, $\{x, z\}$, and $\{y, z\}$ of the fan of the
        $\mathbb{P}^2_{231}$ fiber. The fan is computed on the first call and cached.

        Returns:
            Fan: The fan of the singular uplift ambient variety, whose rays are
                `vectors_singular_uplift_ambient()`.
        """
        if self.__singular_uplift_toric_fan is None:
            cones6 = set()
            x_pts6 = len(self.vectors_orbifold()) + 1
            y_pts6 = x_pts6 + 1
            z_pts6 = x_pts6 + 2

            for c in self.orbifold_toric_fan().cones():
                cones6.add(c + (x_pts6, y_pts6))
                cones6.add(c + (x_pts6, z_pts6))
                cones6.add(c + (y_pts6, z_pts6))
            self.__singular_uplift_toric_fan = Fan(
                VectorConfiguration(self.vectors_singular_uplift_ambient()),
                cones=cones6,
            )
        return self.__singular_uplift_toric_fan

    def Cayley_M(self):
        """Return the Cayley polytope of the Weierstrass and base Newton polytopes.

        This is the convex hull of the points $(m, 1, 0)$ for $m$ in `pol_W_M()` and
        $(m, 0, 1)$ for $m$ in `pol_B_M()`. It is computed on the first call and cached.
        A warning is issued if the uplift is not a nef partition.

        Returns:
            Polytope: The Cayley polytope in the M lattice.
        """
        if self.__Cayley_M is None:
            self.__Cayley_M = Polytope(
                np.concatenate(
                    (
                        np.column_stack(
                            [
                                self.pol_W_M().points(),
                                np.ones(len(self.pol_W_M().points()), dtype=int),
                                np.zeros(len(self.pol_W_M().points()), dtype=int),
                            ]
                        ),
                        np.column_stack(
                            [
                                self.pol_B_M().points(),
                                np.zeros(len(self.pol_B_M().points()), dtype=int),
                                np.ones(len(self.pol_B_M().points()), dtype=int),
                            ]
                        ),
                    ),
                    axis=0,
                )
            )
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__Cayley_M

    def Cayley_N(self):
        """Return the Cayley polytope of the N-lattice polytopes of the partition.

        This is the convex hull of the points $(v, 1, 0)$ for $v$ in `pol_W_N()` and
        $(v, 0, 1)$ for $v$ in `pol_B_N()`. It is computed on the first call and cached.
        A warning is issued if the uplift is not a nef partition.

        Returns:
            Polytope: The Cayley polytope in the N lattice.

        Raises:
            ValueError: If the uplift is not a partition.
        """

        if self.is_partition():
            if self.__Cayley_N is None:
                self.__Cayley_N = Polytope(
                    np.concatenate(
                        (
                            np.column_stack(
                                [
                                    self.pol_W_N().points(),
                                    np.ones(len(self.pol_W_N().points()), dtype=int),
                                    np.zeros(len(self.pol_W_N().points()), dtype=int),
                                ]
                            ),
                            np.column_stack(
                                [
                                    self.pol_B_N().points(),
                                    np.zeros(len(self.pol_B_N().points()), dtype=int),
                                    np.ones(len(self.pol_B_N().points()), dtype=int),
                                ]
                            ),
                        ),
                        axis=0,
                    )
                )
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__Cayley_N

    def pol_W_M(self):
        """Return the Newton polytope of the Weierstrass divisor.

        This is the polytope of monomial sections of the Weierstrass divisor (see
        `line_bundle_weierstrass_N`) on the resolved uplift ambient variety. It is
        computed on the first call and cached.

        Returns:
            HPolytope: The Newton polytope in the M lattice.
        """
        if self.__pol_W_M is None:
            self.__pol_W_M = UF.Newton_Polytope(
                self.vectors_smooth_uplift_ambient(), self.line_bundle_weierstrass_N()
            )
        return self.__pol_W_M

    def pol_B_M(self):
        """Return the Newton polytope of the base divisor.

        This is the polytope of monomial sections of the base divisor (see
        `line_bundle_base_N`) on the resolved uplift ambient variety. It is computed on
        the first call and cached.

        Returns:
            HPolytope: The Newton polytope in the M lattice.
        """
        if self.__pol_B_M is None:
            self.__pol_B_M = UF.Newton_Polytope(
                self.vectors_smooth_uplift_ambient(), self.line_bundle_base_N()
            )
        return self.__pol_B_M

    def pol_M_conv(self):
        """Return the convex hull of the base and Weierstrass Newton polytopes.

        For a nef partition, this is the dual of the Minkowski sum of the N-lattice
        polytopes of the partition, and its lattice points give the rays of the fan
        `M_conv_toric_fan`. It is computed on the first call and cached.

        Returns:
            Polytope: The convex hull of `pol_B_M()` and `pol_W_M()`.
        """
        if self.__pol_M_conv is None:
            self.__pol_M_conv = Polytope(
                np.unique(
                    np.concatenate(
                        (self.pol_B_M().vertices(), self.pol_W_M().vertices()), axis=0
                    ),
                    axis=0,
                )
            )
        return self.__pol_M_conv

    def pol_M_sum(self):
        """Return the Minkowski sum of the base and Weierstrass Newton polytopes.

        It is computed on the first call and cached.

        Returns:
            Polytope: The Minkowski sum of `pol_B_M()` and `pol_W_M()`.
        """
        if self.__pol_M_sum is None:
            self.__pol_M_sum = self.pol_B_M().minkowski_sum(self.pol_W_M())
        return self.__pol_M_sum

    def pol_W_N(self):
        """Return the N-lattice polytope of the Weierstrass part of the partition.

        This is the convex hull of the origin and the rays of the resolved uplift
        ambient fan whose coefficient in the Weierstrass divisor is 1. It is computed on
        the first call and cached. A warning is issued if the uplift is not a nef
        partition.

        Returns:
            Polytope: The Weierstrass polytope in the N lattice.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        if self.is_partition():
            if self.__pol_W_N is None:
                self.__pol_W_N = Polytope(
                    np.concatenate(
                        (
                            np.zeros((1, self.ambient_dim_base() + 2), dtype=int),
                            self.vectors_smooth_uplift_ambient()[
                                np.where(self.line_bundle_weierstrass_N() == 1)[0]
                            ],
                        ),
                        axis=0,
                    )
                )
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_W_N

    def pol_B_N(self):
        """Return the N-lattice polytope of the base part of the partition.

        This is the convex hull of the origin and the rays of the resolved uplift
        ambient fan whose coefficient in the base divisor is 1. It is computed on the
        first call and cached. A warning is issued if the uplift is not a nef partition.

        Returns:
            Polytope: The base polytope in the N lattice.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        if self.is_partition():
            if self.__pol_B_N is None:
                self.__pol_B_N = Polytope(
                    np.concatenate(
                        (
                            np.zeros((1, self.ambient_dim_base() + 2), dtype=int),
                            self.vectors_smooth_uplift_ambient()[
                                np.where(self.line_bundle_base_N() == 1)[0]
                            ],
                        ),
                        axis=0,
                    )
                )
        else:
            raise ValueError("Uplift does not yield a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_B_N

    def pol_N_conv(self):
        """Return the convex hull of the rays of the resolved uplift ambient fan.

        For a partition, this is also the convex hull of `pol_B_N()` and `pol_W_N()`. It
        is computed on the first call and cached.

        Returns:
            Polytope: The convex hull of `vectors_smooth_uplift_ambient()`.
        """
        if self.__pol_N_conv is None:
            self.__pol_N_conv = Polytope(self.vectors_smooth_uplift_ambient())
        return self.__pol_N_conv

    def pol_N_sum(self):
        """Return the Minkowski sum of the N-lattice polytopes of the partition.

        It is computed on the first call and cached. A warning is issued if the uplift
        is not a nef partition.

        Returns:
            Polytope: The Minkowski sum of `pol_B_N()` and `pol_W_N()`.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        if self.is_partition():
            if self.__pol_N_sum is None:
                self.__pol_N_sum = self.pol_B_N().minkowski_sum(self.pol_W_N())
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_N_sum

    def vectors_singular_uplift_ambient(self, labels=None):
        """Return the rays of the singular uplift ambient fan.

        These are the lifted orbifold rays $(v_i, 0, 1 - L_i)$, in the same order as
        `vectors_orbifold()`, followed by the fiber rays $x = (0, 3, 1)$,
        $y = (0, -2, -1)$, and $z = (0, 0, 1)$. They are computed on the first call and
        cached, and the fiber rays are also stored in the attributes `x`, `y`, and `z`.

        Args:
            labels (Iterable[int] | None): One-indexed labels of the rays to return. If
                not specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        if self.__pts_singular_uplift is None:
            self.x = np.concatenate(
                (np.zeros(self.ambient_dim_base(), dtype=int), np.array([3, 1]))
            )
            self.y = np.concatenate(
                (np.zeros(self.ambient_dim_base(), dtype=int), np.array([-2, -1]))
            )
            self.z = np.concatenate(
                (np.zeros(self.ambient_dim_base(), dtype=int), np.array([0, 1]))
            )
            coordinates231 = (1 - self.line_bundle_orbifold())[:, None] @ self.z[-2:][
                None, :
            ]
            pts6 = np.concatenate((self.vectors_orbifold(), coordinates231), axis=1)
            pts6 = np.concatenate((pts6, np.array([self.x, self.y, self.z])), axis=0)
            self.__pts_singular_uplift = pts6
        if labels is None:
            return self.__pts_singular_uplift
        return self.__pts_singular_uplift[np.array(labels) - 1]

    def blowups(self, as_labels=False):
        """Return the blowup rays over the non-Higgsable clusters.

        For each non-Higgsable-cluster divisor with ray $u$ in the singular uplift
        ambient fan, the two rays $u + x + 2y$ and $2u + 2x + 3y$ are added, where $x$
        and $y$ are fiber rays (see `vectors_singular_uplift_ambient`). The rays are
        computed on the first call and cached.

        Args:
            as_labels (bool): Whether to return the one-indexed labels of the blowup
                rays in the resolved uplift ambient fan instead of the rays themselves.

        Returns:
            list[np.ndarray] | np.ndarray: A list with one array per
                non-Higgsable-cluster divisor, containing its two blowup rays as rows.
                If `as_labels` is true, a 1D array of labels instead.
        """
        if self.__blowups is None:
            bus = []
            for xxx in self.NHC_singular_uplift(as_labels=True):
                bus.append(
                    np.array(
                        [
                            self.vectors_singular_uplift_ambient()[xxx - 1]
                            + self.x
                            + 2 * self.y,
                            2 * self.vectors_singular_uplift_ambient()[xxx - 1]
                            + 2 * self.x
                            + 3 * self.y,
                        ]
                    )
                )
            self.__blowups = bus
        if as_labels:
            return np.arange(
                len(self.vectors_singular_uplift_ambient()) + 1,
                len(self.vectors_smooth_uplift_ambient()) + 1,
            )
        return self.__blowups

    def points_not_interior_to_codim_1_and_2_face_M(self):
        """Return the points of `pol_M_conv()` not interior to facets or codim-2 faces.

        The points interior to facets and to codimension-two faces are removed. The
        remaining nonzero points are the rays of `M_conv_toric_fan`. The result is
        computed on the first call and cached.

        Returns:
            np.ndarray: The selected lattice points, one per row, including the origin.
        """
        if self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = (
                UF.points_not_interior_to_facets_and_codim2_faces(self.pol_M_conv())
            )
        return self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv

    def points_not_interior_to_codim_1_and_2_face_N(self):
        """Return the points of `pol_N_conv()` not interior to facets or codim-2 faces.

        The points interior to facets and to codimension-two faces are removed. The
        result is computed on the first call and cached.

        Returns:
            np.ndarray: The selected lattice points, one per row, including the origin.
        """
        if self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = (
                UF.points_not_interior_to_facets_and_codim2_faces(self.pol_N_conv())
            )
        return self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv

    def polytope(self):
        """Return the polytope used to construct the underlying orientifold.

        Returns:
            Polytope | None: The polytope, or `None` if the orientifold was not
                constructed from a polytope.
        """
        return self.orientifold().polytope()

    def intersection_numbers_M_conv(self, check=True):
        """Return the intersection numbers of the toric variety of `M_conv_toric_fan`.

        They are computed on the first call and cached.

        Args:
            check (bool): Unused. Kept for backwards compatibility.

        Returns:
            dict[tuple[int, ...], float]: The intersection numbers, keyed by sorted
                tuples of divisor labels. Labels are one-indexed ray labels, and the
                label 0 refers to the canonical divisor.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        if self.__intersection_numbers_M_conv is None:
            self.__intersection_numbers_M_conv = (
                self.M_conv_toric_fan().intersection_numbers()
            )
        return self.__intersection_numbers_M_conv

    def intersection_numbers_smooth_uplift_ambient(self):
        """Return the intersection numbers of the resolved uplift ambient variety.

        They are computed on the first call and cached.

        Returns:
            dict[tuple[int, ...], float]: The intersection numbers, keyed by sorted
                tuples of divisor labels. Labels are one-indexed ray labels, and the
                label 0 refers to the canonical divisor.
        """
        if self.__intersection_numbers_smooth_uplift is None:
            self.__intersection_numbers_smooth_uplift = (
                self.smooth_uplift_ambient_toric_fan().intersection_numbers()
            )
        return self.__intersection_numbers_smooth_uplift

    def intersection_numbers_singular_uplift_ambient(self, check=True):
        """Return the intersection numbers of the singular uplift ambient variety.

        They are computed on the first call and cached.

        Args:
            check (bool): Unused. Kept for backwards compatibility.

        Returns:
            dict[tuple[int, ...], float]: The intersection numbers, keyed by sorted
                tuples of divisor labels. Labels are one-indexed ray labels, and the
                label 0 refers to the canonical divisor.
        """
        if self.__intersection_numbers_singular_uplift is None:
            self.__intersection_numbers_singular_uplift = (
                self.singular_uplift_ambient_toric_fan().intersection_numbers()
            )
        return self.__intersection_numbers_singular_uplift

    def h11(self):
        """Return the Hodge number $h^{1,1}$ of the Calabi-Yau fourfold.

        It is computed with the Batyrev-Borisov formula for two-part nef partitions,
        using `Cayley_M` and `Cayley_N`, on the first call and cached.

        Returns:
            int: The Hodge number $h^{1,1}$.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            if self.__h11 is None:
                self.__h11 = UF.h11_2_part(self.Cayley_M(), self.Cayley_N())
            return self.__h11
        else:
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")

    def h31(self):
        """Return the Hodge number $h^{3,1}$ of the Calabi-Yau fourfold.

        It is computed with the Batyrev-Borisov formula for two-part nef partitions,
        with the roles of `Cayley_M` and `Cayley_N` exchanged relative to `h11`, on the
        first call and cached.

        Returns:
            int: The Hodge number $h^{3,1}$.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            if self.__h31 is None:
                self.__h31 = UF.h11_2_part(self.Cayley_N(), self.Cayley_M())
            return self.__h31
        else:
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")

    def h31_trivial(self):
        """Return the leading term of the formula for $h^{3,1}$.

        This is the number of lattice points of `Cayley_M()` minus the dimension of
        `Cayley_N()` minus one, i.e. the formula used by `h31` without its correction
        terms. It is not the Hodge number itself.

        Returns:
            int: The leading term of $h^{3,1}$.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        return len(self.Cayley_M().points()) - 1 - (self.Cayley_N().dim())

    def line_bundle_weierstrass_M(self):
        """Return the Weierstrass part of the dual nef partition in the M lattice.

        This is the coefficient vector with one entry per ray of `M_conv_toric_fan`,
        which is 1 for the rays that are lattice points of `pol_W_M()` and 0 otherwise.
        It is computed on the first call and cached.

        Returns:
            np.ndarray: The coefficients of the Weierstrass part.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            if self.__LBW_M is None:
                LB_W = np.zeros(
                    len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int
                )
                LB_W[
                    UF.get_indices(
                        self.points_not_interior_to_codim_1_and_2_face_M(),
                        self.pol_W_M().points(),
                    )
                ] = 1
                self.__LBW_M = LB_W[1:]
        else:
            raise ValueError("Uplift is not a nef-partition")
        return self.__LBW_M

    def line_bundle_base_M(self):
        """Return the base part of the dual nef partition in the M lattice.

        This is the coefficient vector with one entry per ray of `M_conv_toric_fan`,
        which is 1 for the rays that are lattice points of `pol_B_M()` and 0 otherwise.
        It is computed on the first call and cached.

        Returns:
            np.ndarray: The coefficients of the base part.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            if self.__LBB_M is None:
                LB_B = np.zeros(
                    len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int
                )
                LB_B[
                    UF.get_indices(
                        self.points_not_interior_to_codim_1_and_2_face_M(),
                        self.pol_B_M().points(),
                    )
                ] = 1
                self.__LBB_M = LB_B[1:]
        else:
            raise ValueError("Uplift is not a nef-partition")
        return self.__LBB_M

    def line_bundle_weierstrass_N(self):
        """Return the Weierstrass divisor on the resolved uplift ambient variety.

        If the uplift is a partition (see `is_partition`), this is the representative
        with coefficients 0 or 1 that, together with the base divisor, partitions the
        rays. Otherwise, it is the anticanonical divisor minus the base divisor, which
        is linearly equivalent to $3D_x$. It is computed on the first call and cached.

        Returns:
            np.ndarray: The coefficients of the divisor, one per ray of
                `vectors_smooth_uplift_ambient()`.
        """
        if self.__LBW_N is None:
            self.__set_divisor_representations()
        return self.__LBW_N

    def line_bundle_base_N(self):
        """Return the base divisor on the resolved uplift ambient variety.

        If the uplift is a partition (see `is_partition`), this is the representative
        with coefficients 0 or 1 that, together with the Weierstrass divisor, partitions
        the rays. Otherwise, it has the coefficients $L_i$ on the orbifold rays and the
        coefficients $L_D$ and $2L_D$ on the two blowup rays over each
        non-Higgsable-cluster divisor $D$. It is computed on the first call and cached.

        Returns:
            np.ndarray: The coefficients of the divisor, one per ray of
                `vectors_smooth_uplift_ambient()`.
        """
        if self.__LBB_N is None:
            self.__set_divisor_representations()
        return self.__LBB_N

    def line_bundle_orbifold(self):
        """Return the O3/O7 line bundle of the underlying orientifold.

        See
        [`CY_orientifold.line_bundle`][cytools.f_theory.FT_CY.CY_orientifold.line_bundle].

        Returns:
            np.ndarray | None: The coefficients of the line bundle, one per orbifold
                ray.
        """
        return self.orientifold().line_bundle()

    def M_conv_toric_fan(self):
        """Return the toric fan over the M-lattice convex hull polytope.

        Its rays are the nonzero points of
        `points_not_interior_to_codim_1_and_2_face_M`, and it is obtained by
        triangulating their vector configuration with the default settings. It is
        computed on the first call and cached.

        Returns:
            Fan: The fan associated with `pol_M_conv()`.

        Raises:
            ValueError: If the uplift is not a partition.
        """
        if not self.is_partition():
            raise ValueError("Uplift is not a partition")
        if self.__M_conv_toric_fan is None:
            index0 = UF.get_index(
                self.points_not_interior_to_codim_1_and_2_face_M(),
                np.zeros(
                    len(self.points_not_interior_to_codim_1_and_2_face_M()[0]),
                    dtype=int,
                ),
            )[0]
            pts = np.delete(
                self.points_not_interior_to_codim_1_and_2_face_M(), index0, axis=0
            )
            self.__M_conv_toric_fan = VectorConfiguration(pts).triangulate()
        return self.__M_conv_toric_fan

    def divisor_intersection_M(self, as_LLL=True):
        r"""Return the classes of curves cut out by the dual nef partition.

        For each cone $\sigma$ of `M_conv_toric_fan` of codimension three, the curve
        obtained by intersecting the toric stratum of $\sigma$ with the base and
        Weierstrass divisors of the dual nef partition (see `line_bundle_base_M` and
        `line_bundle_weierstrass_M`) is recorded through its intersection numbers with
        the basis divisors given by `basis_homology_M`.

        Args:
            as_LLL (bool): Whether to return an LLL-reduced basis of the lattice
                generated by these curve classes, with zero rows removed, instead of the
                classes of all the curves.

        Returns:
            np.ndarray: The curve classes, one per row, with one column per basis
                divisor.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if not self.is_nef_partition():
            raise ValueError("Uplift is not a nef-partition")
        return UF.divisor_intersections(
            fan=self.M_conv_toric_fan(),
            divisors=[self.line_bundle_base_M(), self.line_bundle_weierstrass_M()],
            intersection_dict=self.intersection_numbers_M_conv(),
            basis_set=set(self.basis_homology_M()),
            as_LLL=as_LLL,
        )

    def divisor_intersection_N(self, as_LLL=True):
        r"""Return the classes of curves cut out by the base and Weierstrass divisors.

        For each cone $\sigma$ of `smooth_uplift_ambient_toric_fan` of codimension
        three, the curve obtained by intersecting the toric stratum of $\sigma$ with the
        base and Weierstrass divisors (see `line_bundle_base_N` and
        `line_bundle_weierstrass_N`) is recorded through its intersection numbers with
        the basis divisors given by `basis_homology_N`.

        Args:
            as_LLL (bool): Whether to return an LLL-reduced basis of the lattice
                generated by these curve classes, with zero rows removed, instead of the
                classes of all the curves.

        Returns:
            np.ndarray: The curve classes, one per row, with one column per basis
                divisor.
        """
        return UF.divisor_intersections(
            fan=self.smooth_uplift_ambient_toric_fan(),
            divisors=[self.line_bundle_base_N(), self.line_bundle_weierstrass_N()],
            intersection_dict=self.intersection_numbers_smooth_uplift_ambient(),
            basis_set=set(self.basis_homology_N()),
            as_LLL=as_LLL,
        )

    def NHC(self, as_labels=False):
        """Return the orbifold toric divisors that support non-Higgsable clusters.

        See [`CY_orientifold.NHC`][cytools.f_theory.FT_CY.CY_orientifold.NHC].

        Args:
            as_labels (bool): Whether to return the one-indexed labels of the orbifold
                rays instead of the rays themselves.

        Returns:
            np.ndarray: The labels of the non-Higgsable-cluster divisors, or the
                corresponding orbifold rays, one per row.
        """
        return self.orientifold().NHC(as_labels)

    def NHC_singular_uplift(self, as_labels=False):
        """Return the singular uplift rays of the non-Higgsable-cluster divisors.

        Since the lifted orbifold rays come first in the singular uplift ambient fan,
        their labels coincide with those returned by `NHC`.

        Args:
            as_labels (bool): Whether to return the one-indexed labels of the rays
                instead of the rays themselves.

        Returns:
            np.ndarray | None: The labels of the rays, or the rays themselves, one per
                row. If `as_labels` is false and there are no non-Higgsable clusters,
                `None` is returned.
        """
        if as_labels:
            return self.NHC(as_labels=True)
        else:
            labels = self.NHC(as_labels=True)
            if len(labels) > 0:
                return self.vectors_singular_uplift_ambient(labels)

    def vectors_orbifold(self, labels=None):
        """Return the rays of the orbifold ambient fan of the underlying orientifold.

        See
        [`CY_orientifold.vectors_orbifold`][cytools.f_theory.FT_CY.CY_orientifold.vectors_orbifold].

        Args:
            labels (Iterable[int] | None): One-indexed labels of the rays to return. If
                not specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        return self.orientifold().vectors_orbifold(labels)

    def vectors_CY_ambient(self, labels=None):
        """Return the rays of the Calabi-Yau ambient fan of the underlying orientifold.

        Args:
            labels (Iterable[int] | None): One-indexed labels of the rays to return. If
                not specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        return self.orientifold().vectors_CY_ambient(labels)

    def dim_base(self):
        """Return the dimension of the ambient variety of the underlying orientifold.

        This is also the dimension of the orbifold ambient variety containing the base.

        Returns:
            int: The dimension of the orientifold ambient variety.
        """
        return self.orientifold().dim()

    def ambient_dim_base(self):
        """Return the dimension of the lattice containing the orbifold rays.

        Returns:
            int: The number of coordinates of each orbifold ray.
        """
        return self.orientifold().ambient_dim()

    def CY_ambient_toric_fan(self):
        """Return the toric fan of the Calabi-Yau ambient variety, before orbifolding.

        See
        [`CY_orientifold.CY_ambient_toric_fan`][cytools.f_theory.FT_CY.CY_orientifold.CY_ambient_toric_fan].

        Returns:
            Fan: The fan of the Calabi-Yau ambient variety.
        """
        return self.orientifold().CY_ambient_toric_fan()

    def orbifold_toric_fan(self):
        """Return the toric fan of the orbifold ambient variety.

        See
        [`CY_orientifold.orbifold_toric_fan`][cytools.f_theory.FT_CY.CY_orientifold.orbifold_toric_fan].

        Returns:
            Fan: The fan of the orbifold ambient variety.
        """
        return self.orientifold().orbifold_toric_fan()

    def vectors_smooth_uplift_ambient(self, labels=None):
        """Return the rays of the resolved uplift ambient fan.

        These are the rays of the singular uplift ambient fan (see
        `vectors_singular_uplift_ambient`), followed by the two blowup rays over each
        non-Higgsable-cluster divisor (see `blowups`). They are computed on the first
        call and cached.

        Args:
            labels (Iterable[int] | None): One-indexed labels of the rays to return. If
                not specified, all rays are returned.

        Returns:
            np.ndarray: The selected rays, one per row.
        """
        if self.__pts_smooth_uplift is None:
            if len(self.NHC()) > 0:
                self.__pts_smooth_uplift = np.concatenate(
                    (self.vectors_singular_uplift_ambient(), np.vstack(self.blowups())),
                    axis=0,
                )
            else:
                self.__pts_smooth_uplift = self.vectors_singular_uplift_ambient()
        if labels is None:
            return self.__pts_smooth_uplift
        return self.__pts_smooth_uplift[np.array(labels) - 1]

    def xi(self):
        r"""Return the vector $\xi$ defining the orientifold $\mathbb{Z}_2$ action.

        Returns:
            np.ndarray | list: The vector $\xi$. The involution is the torus element
                $\xi/2$.
        """
        return self.__CY_orientifold.xi()

    def h21(self):
        """Return the Hodge number $h^{2,1}$ of the Calabi-Yau fourfold.

        It is computed with the Batyrev-Borisov formula for two-part nef partitions,
        using `Cayley_M` and `Cayley_N`, on the first call and cached.

        Returns:
            int: The Hodge number $h^{2,1}$.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.__h21 is None:
            if self.is_nef_partition():
                self.__h21 = UF.h21_2_part(self.Cayley_N(), self.Cayley_M())
            else:
                raise ValueError("Uplift is not a nef-partition")
        return self.__h21

    def chi(self):
        r"""Return the Euler characteristic of the Calabi-Yau fourfold.

        It is computed from the Hodge numbers as
        $\chi = 48 + 6(h^{1,1} + h^{3,1} - h^{2,1})$ on the first call and cached.

        Returns:
            int: The Euler characteristic.

        Raises:
            ValueError: If the uplift is not a nef partition.
        """
        if self.__chi is None:
            self.__chi = 48 + 6 * (self.h11() + self.h31() - self.h21())
        return self.__chi

    def is_regular(self):
        """Return whether the underlying orientifold is regular.

        See
        [`CY_orientifold.is_regular`][cytools.f_theory.FT_CY.CY_orientifold.is_regular].

        Returns:
            bool: Whether the orientifold is regular.
        """
        return self.orientifold().is_regular()

    def normal_fan(self):
        """Return the normal fan used in the nef-decomposition construction.

        See
        [`CY_orientifold.normal_fan`][cytools.f_theory.FT_CY.CY_orientifold.normal_fan].

        Returns:
            Fan: The normal fan.
        """
        return self.orientifold().normal_fan()

    def basis_homology_M(self, as_LLL=True):
        """Return a basis of toric divisors of `M_conv_toric_fan`.

        The basis consists of the prime toric divisors whose rays are not in the first
        smooth maximal cone found. Curve classes are expressed through their
        intersection numbers with these divisors (see `divisor_intersection_M`). The
        result is computed on the first call and cached.

        Args:
            as_LLL (bool): Unused. Kept for backwards compatibility.

        Returns:
            np.ndarray: The one-indexed labels of the basis divisors.

        Raises:
            ValueError: If the uplift is not a partition, or if the fan has no smooth
                maximal cone.
        """
        if self.__basis_homology_M is None:
            self.__basis_homology_M = UF.basis_H2_toric_fan(self.M_conv_toric_fan())
        return self.__basis_homology_M

    def basis_homology_N(self, as_LLL=True):
        """Return a basis of toric divisors of the resolved uplift ambient fan.

        The basis consists of the prime toric divisors whose rays are not in the first
        smooth maximal cone found. Curve classes are expressed through their
        intersection numbers with these divisors (see `divisor_intersection_N`). The
        result is computed on the first call and cached.

        Args:
            as_LLL (bool): Unused. Kept for backwards compatibility.

        Returns:
            np.ndarray: The one-indexed labels of the basis divisors.

        Raises:
            ValueError: If the fan has no smooth maximal cone.
        """
        if self.__basis_homology_N is None:
            self.__basis_homology_N = UF.basis_H2_toric_fan(
                self.smooth_uplift_ambient_toric_fan()
            )
        return self.__basis_homology_N

    def nef_partition_N(self):
        """Return the nef partition of the rays of the resolved uplift ambient fan.

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]]: The one-indexed labels of the rays
                in the base part and in the Weierstrass part, respectively. Both tuples
                are empty if the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            return (
                tuple(np.where(self.line_bundle_base_N() == 1)[0] + 1),
                tuple(np.where(self.line_bundle_weierstrass_N() == 1)[0] + 1),
            )
        else:
            return ((), ())

    def nef_partition_M(self):
        """Return the dual nef partition of the rays of `M_conv_toric_fan`.

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]]: The one-indexed labels of the rays
                in the base part and in the Weierstrass part, respectively. Both tuples
                are empty if the uplift is not a nef partition.
        """
        if self.is_nef_partition():
            return (
                tuple(np.where(self.line_bundle_base_M() == 1)[0] + 1),
                tuple(np.where(self.line_bundle_weierstrass_M() == 1)[0] + 1),
            )
        else:
            return ((), ())

    def is_nef_decomposition(self):
        """Return whether the base and Weierstrass divisors are Cartier and nef.

        If the underlying orientifold was constructed from a `Fan` (and the
        nef-decomposition refinement was not applied), this checks that both divisors
        are Cartier and nef on the resolved uplift ambient fan. Otherwise, it returns
        [`CY_orientifold.yields_nef_decomposition`][cytools.f_theory.FT_CY.CY_orientifold.yields_nef_decomposition].
        The result is computed on the first call and cached.

        Returns:
            bool: Whether the uplift defines a nef decomposition.
        """
        if self.__is_nef_decomposition is None:
            if self.orientifold().ambient_triangulation:
                self.__is_nef_decomposition = np.all(
                    [
                        UF.is_Cartier(
                            self.smooth_uplift_ambient_toric_fan(),
                            self.line_bundle_base_N(),
                        )[0],
                        UF.is_Cartier(
                            self.smooth_uplift_ambient_toric_fan(),
                            self.line_bundle_weierstrass_N(),
                        )[0],
                        UF.is_nef(
                            self.smooth_uplift_ambient_toric_fan(),
                            self.line_bundle_base_N(),
                        ),
                        UF.is_nef(
                            self.smooth_uplift_ambient_toric_fan(),
                            self.line_bundle_weierstrass_N(),
                        ),
                    ]
                )
            else:
                self.__is_nef_decomposition = (
                    self.orientifold().yields_nef_decomposition()
                )
        return self.__is_nef_decomposition

    def intersection_numbers_orbifold(self):
        """Return the intersection numbers of the orbifold ambient variety.

        See
        [`CY_orientifold.intersection_numbers_orbifold`][cytools.f_theory.FT_CY.CY_orientifold.intersection_numbers_orbifold].

        Returns:
            dict[tuple[int, ...], float]: The intersection numbers, keyed by sorted
                tuples of divisor labels.
        """
        return self.orientifold().intersection_numbers_orbifold()


def fetch_orientifolds(
    only_regular: bool = True,
    only_nef_decomposition: bool = False,
    h11: int = None,
    h12: int = None,
    h13: int = None,
    h21: int = None,
    h22: int = None,
    h31: int = None,
    chi: int = None,
    lattice: str = "N",
    dim: int = 4,
    n_points: int = None,
    n_vertices: int = None,
    n_dual_points: int = None,
    n_facets: int = None,
    limit: int = 1000,
    samples: int = None,
    sample_seed: int = None,
    timeout: int = 60,
    as_list: bool = True,
    backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0,
):
    r"""Yield orientifolds of Calabi-Yau hypersurfaces from the polytope databases.

    For each polytope returned by [`fetch_polytopes`][cytools.utils.fetch_polytopes],
    the $\mathbb{Z}_2$ actions $\xi/2$, with $\xi$ a nonzero vector with entries 0 or 1,
    are enumerated up to the automorphisms of the polytope, and the corresponding
    [`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold] objects are constructed
    with the default settings and yielded, optionally filtered.

    Args:
        only_regular (bool): Whether to yield only regular orientifolds (see
            [`CY_orientifold.is_regular`][cytools.f_theory.FT_CY.CY_orientifold.is_regular]).
        only_nef_decomposition (bool): Whether to yield only orientifolds that yield a
            nef decomposition (see
            [`CY_orientifold.yields_nef_decomposition`][cytools.f_theory.FT_CY.CY_orientifold.yields_nef_decomposition]).
            These are always regular, so `only_regular` is ignored in this case.
        h11 (int | None): The Hodge number $h^{1,1}$ of the Calabi-Yau hypersurface.
        h12 (int | None): The Hodge number $h^{1,2}$ of the Calabi-Yau hypersurface.
        h13 (int | None): The Hodge number $h^{1,3}$ of the Calabi-Yau hypersurface.
        h21 (int | None): The Hodge number $h^{2,1}$ of the Calabi-Yau hypersurface.
        h22 (int | None): The Hodge number $h^{2,2}$ of the Calabi-Yau hypersurface.
        h31 (int | None): The Hodge number $h^{3,1}$ of the Calabi-Yau hypersurface.
        chi (int | None): The Euler characteristic of the Calabi-Yau hypersurface.
        lattice (str): The lattice in which the polytopes are defined, either `"N"` or
            `"M"`.
        dim (int): The dimension of the polytopes.
        n_points (int | None): The number of lattice points of the polytopes.
        n_vertices (int | None): The number of vertices of the polytopes.
        n_dual_points (int | None): The number of lattice points of the dual polytopes.
        n_facets (int | None): The number of facets of the polytopes.
        limit (int): The maximum number of polytopes to fetch. Each polytope can give
            several orientifolds.
        samples (int | None): The number of polytopes to randomly sample from the
            fetched ones.
        sample_seed (int | None): The random seed used for sampling.
        timeout (int): The maximum number of seconds to wait for the server.
        as_list (bool): Whether to fetch all polytopes before constructing any
            orientifolds, instead of fetching them lazily. Orientifolds are yielded one
            at a time in either case.
        backend (str | None): The backend used for the
            [`Polytope`][cytools.polytope.Polytope] objects.
        deterministic_glsm_basis (bool): Whether to fix the GLSM bases of the polytopes
            in a deterministic manner.
        dualize (bool): Whether to dualize the polytopes before using them.
        favorable (bool | None): If true or false, only use favorable or non-favorable
            polytopes, respectively. If not specified, both are used.
        verbosity (int): The verbosity level.

    Yields:
        CY_orientifold: The orientifolds satisfying the requested conditions.

    Examples:
        We iterate over the regular orientifolds of polytopes with $h^{1,1}=2$.

        >>> from cytools.f_theory import fetch_orientifolds
        >>> gen = fetch_orientifolds(h11=2, limit=5)
        >>> for o in gen:  # doctest: +SKIP
        ...     print(o.xi(), o.line_bundle())
    """

    if only_nef_decomposition:
        for p in fetch_polytopes(
            h11,
            h12,
            h13,
            h21,
            h22,
            h31,
            chi,
            lattice,
            dim,
            n_points,
            n_vertices,
            n_dual_points,
            n_facets,
            limit,
            samples,
            sample_seed,
            timeout,
            as_list,
            backend,
            deterministic_glsm_basis,
            dualize,
            favorable,
            verbosity,
        ):
            for xi in UF.inequivalent_Z2_actions(p.automorphisms(action="left")):
                orientifold = CY_orientifold(p, xi)
                if orientifold.yields_nef_decomposition():
                    yield orientifold
    else:
        for p in fetch_polytopes(
            h11,
            h12,
            h13,
            h21,
            h22,
            h31,
            chi,
            lattice,
            dim,
            n_points,
            n_vertices,
            n_dual_points,
            n_facets,
            limit,
            samples,
            sample_seed,
            timeout,
            as_list,
            backend,
            deterministic_glsm_basis,
            dualize,
            favorable,
            verbosity,
        ):
            for xi in UF.inequivalent_Z2_actions(p.automorphisms(action="left")):
                orientifold = CY_orientifold(p, xi)
                if only_regular:
                    if orientifold.is_regular():
                        yield orientifold
                else:
                    yield orientifold


def fetch_F_Theory_uplifts(
    only_regular: bool = True,
    only_nef_partition: bool = False,
    only_nef_decomposition: bool = False,
    h11: int = None,
    h12: int = None,
    h13: int = None,
    h21: int = None,
    h22: int = None,
    h31: int = None,
    chi: int = None,
    lattice: str = "N",
    dim: int = 4,
    n_points: int = None,
    n_vertices: int = None,
    n_dual_points: int = None,
    n_facets: int = None,
    limit: int = 1000,
    samples: int = None,
    sample_seed: int = None,
    timeout: int = 60,
    as_list: bool = True,
    backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0,
):
    """Yield F-theory uplifts of orientifolds of polytopes from the databases.

    The orientifolds are obtained with
    [`fetch_orientifolds`][cytools.f_theory.FT_CY.fetch_orientifolds], and the
    corresponding [`F_Theory_Uplift`][cytools.f_theory.FT_CY.F_Theory_Uplift] objects
    are yielded, optionally filtered.

    Args:
        only_regular (bool): Whether to only use regular orientifolds. Ignored if
            `only_nef_partition` or `only_nef_decomposition` is true, since these imply
            regularity.
        only_nef_partition (bool): Whether to yield only uplifts that define a nef
            partition (see
            [`F_Theory_Uplift.is_nef_partition`][cytools.f_theory.FT_CY.F_Theory_Uplift.is_nef_partition]).
        only_nef_decomposition (bool): Whether to only use orientifolds that yield a nef
            decomposition. Ignored if `only_nef_partition` is true, since it is then
            implied.
        h11 (int | None): The Hodge number $h^{1,1}$ of the Calabi-Yau hypersurface.
        h12 (int | None): The Hodge number $h^{1,2}$ of the Calabi-Yau hypersurface.
        h13 (int | None): The Hodge number $h^{1,3}$ of the Calabi-Yau hypersurface.
        h21 (int | None): The Hodge number $h^{2,1}$ of the Calabi-Yau hypersurface.
        h22 (int | None): The Hodge number $h^{2,2}$ of the Calabi-Yau hypersurface.
        h31 (int | None): The Hodge number $h^{3,1}$ of the Calabi-Yau hypersurface.
        chi (int | None): The Euler characteristic of the Calabi-Yau hypersurface.
        lattice (str): The lattice in which the polytopes are defined, either `"N"` or
            `"M"`.
        dim (int): The dimension of the polytopes.
        n_points (int | None): The number of lattice points of the polytopes.
        n_vertices (int | None): The number of vertices of the polytopes.
        n_dual_points (int | None): The number of lattice points of the dual polytopes.
        n_facets (int | None): The number of facets of the polytopes.
        limit (int): The maximum number of polytopes to fetch. Each polytope can give
            several uplifts.
        samples (int | None): The number of polytopes to randomly sample from the
            fetched ones.
        sample_seed (int | None): The random seed used for sampling.
        timeout (int): The maximum number of seconds to wait for the server.
        as_list (bool): Whether to fetch all polytopes before constructing any uplifts,
            instead of fetching them lazily. Uplifts are yielded one at a time in either
            case.
        backend (str | None): The backend used for the
            [`Polytope`][cytools.polytope.Polytope] objects.
        deterministic_glsm_basis (bool): Whether to fix the GLSM bases of the polytopes
            in a deterministic manner.
        dualize (bool): Whether to dualize the polytopes before using them.
        favorable (bool | None): If true or false, only use favorable or non-favorable
            polytopes, respectively. If not specified, both are used.
        verbosity (int): The verbosity level.

    Yields:
        F_Theory_Uplift: The F-theory uplifts satisfying the requested conditions.

    Examples:
        We compute the Hodge numbers of the uplifts with nef partitions obtained from
        polytopes with $h^{1,1}=2$.

        >>> from cytools.f_theory import fetch_F_Theory_uplifts
        >>> gen = fetch_F_Theory_uplifts(h11=2, limit=5, only_nef_partition=True)
        >>> for F in gen:  # doctest: +SKIP
        ...     print(F.h11(), F.h21(), F.h31())
    """

    if only_nef_partition:
        for orientifold in fetch_orientifolds(
            only_nef_decomposition=True,
            h11=h11,
            h12=h12,
            h13=h13,
            h21=h21,
            h22=h22,
            h31=h31,
            chi=chi,
            lattice=lattice,
            dim=dim,
            n_points=n_points,
            n_vertices=n_vertices,
            n_dual_points=n_dual_points,
            n_facets=n_facets,
            limit=limit,
            samples=samples,
            sample_seed=sample_seed,
            timeout=timeout,
            as_list=as_list,
            backend=backend,
            deterministic_glsm_basis=deterministic_glsm_basis,
            dualize=dualize,
            favorable=favorable,
            verbosity=verbosity,
        ):
            F = F_Theory_Uplift(orientifold)
            if F.is_nef_partition():
                yield F
    elif only_nef_decomposition:
        for orientifold in fetch_orientifolds(
            only_nef_decomposition=True,
            h11=h11,
            h12=h12,
            h13=h13,
            h21=h21,
            h22=h22,
            h31=h31,
            chi=chi,
            lattice=lattice,
            dim=dim,
            n_points=n_points,
            n_vertices=n_vertices,
            n_dual_points=n_dual_points,
            n_facets=n_facets,
            limit=limit,
            samples=samples,
            sample_seed=sample_seed,
            timeout=timeout,
            as_list=as_list,
            backend=backend,
            deterministic_glsm_basis=deterministic_glsm_basis,
            dualize=dualize,
            favorable=favorable,
            verbosity=verbosity,
        ):
            yield F_Theory_Uplift(orientifold)
    else:
        for orientifold in fetch_orientifolds(
            only_regular=only_regular,
            only_nef_decomposition=False,
            h11=h11,
            h12=h12,
            h13=h13,
            h21=h21,
            h22=h22,
            h31=h31,
            chi=chi,
            lattice=lattice,
            dim=dim,
            n_points=n_points,
            n_vertices=n_vertices,
            n_dual_points=n_dual_points,
            n_facets=n_facets,
            limit=limit,
            samples=samples,
            sample_seed=sample_seed,
            timeout=timeout,
            as_list=as_list,
            backend=backend,
            deterministic_glsm_basis=deterministic_glsm_basis,
            dualize=dualize,
            favorable=favorable,
            verbosity=verbosity,
        ):
            yield F_Theory_Uplift(orientifold)


def fetch_nef_partition_uplifts(
    h11=None,
    h21=None,
    limit=100,
    repo_id="jakobmoritz/F-theory_Uplifts_Nef-partitions",
    index_path=None,
    local_base=None,
    columns=None,
    as_pandas=True,
):
    """Load precomputed F-theory uplifts that define nef partitions.

    The data are read from an indexed parquet dataset, by default
    [jakobmoritz/F-theory_Uplifts_Nef-partitions](https://huggingface.co/datasets/jakobmoritz/F-theory_Uplifts_Nef-partitions)
    on the Hugging Face Hub, or from a local copy of it. The small index file
    `index/hodge_index.parquet` is used to find the parquet files and row groups
    containing the rows with the requested Hodge numbers, so that only those are read.
    The rows are returned in order of increasing `h11` and `h21`. Reading from the
    Hugging Face Hub requires network access and the `huggingface_hub` package.

    Args:
        h11 (int | None): The value of the `h11` column (the Hodge number $h^{1,1}$) of
            the rows to return. If not specified, it is not restricted.
        h21 (int | None): The value of the `h21` column (the Hodge number $h^{2,1}$) of
            the rows to return. If not specified, it is not restricted.
        limit (int | None): The maximum number of rows to return. If `None`, all
            matching rows are returned.
        repo_id (str): The Hugging Face dataset repository to read from. Only used if
            `local_base` is not specified.
        index_path (str | Path | None): The path to the index file. If not specified, it
            is `index/hodge_index.parquet` inside the dataset.
        local_base (str | Path | None): A local directory containing the dataset, with
            the index file and the parquet files at the paths recorded in the index. If
            not specified, the data are read from the Hugging Face Hub.
        columns (list[str] | None): The columns to read. If not specified, all columns
            are read.
        as_pandas (bool): Whether to return a pandas `DataFrame` instead of a pyarrow
            `Table`.

    Returns:
        pd.DataFrame | pa.Table: The matching rows. If there are none, or if `limit` is
            not positive, an empty table with no columns is returned.

    Raises:
        ValueError: If neither `h11` nor `h21` is specified, or if the index has no
            `row_group` column.

    Examples:
        We load the first ten uplifts with $h^{1,1}=2$.

        >>> from cytools.f_theory import fetch_nef_partition_uplifts
        >>> df = fetch_nef_partition_uplifts(h11=2, limit=10)  # doctest: +SKIP
    """

    if h11 is None and h21 is None:
        raise ValueError("At least one of h11 or h21 must be provided.")

    if limit is not None and limit <= 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    if local_base is None:
        base = f"hf://datasets/{repo_id}"
        if index_path is None:
            index_path = f"{base}/index/hodge_index.parquet"
    else:
        base = Path(local_base)
        if index_path is None:
            index_path = base / "index" / "hodge_index.parquet"

    # Load small index file.
    index = pd.read_parquet(index_path)

    # Restrict index.
    mask = pd.Series(True, index=index.index)

    if h11 is not None:
        mask &= index["h11"] == int(h11)

    if h21 is not None:
        mask &= index["h21"] == int(h21)

    subindex = index.loc[mask].copy()

    if len(subindex) == 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    # Preserve natural Hodge ordering.
    sort_cols = []
    if "h11" in subindex.columns:
        sort_cols.append("h11")
    if "h21" in subindex.columns:
        sort_cols.append("h21")
    if "row_group" in subindex.columns:
        sort_cols.append("row_group")

    if sort_cols:
        subindex = subindex.sort_values(sort_cols)

    tables = []
    rows_collected = 0

    # Group by file to avoid reopening unnecessarily.
    for file, group in subindex.groupby("file", sort=False):
        if limit is not None and rows_collected >= limit:
            break

        if local_base is None:
            parquet_path = f"{base}/{file}"
        else:
            parquet_path = base / file

        pf = pq.ParquetFile(parquet_path)

        if "row_group" not in group.columns:
            raise ValueError(
                "The index does not contain a 'row_group' column. "
                "For efficient limited access, recreate hodge_index.parquet "
                "with row-group information."
            )

        row_groups = group["row_group"].astype(int).tolist()

        for rg in row_groups:
            if limit is not None and rows_collected >= limit:
                break

            table = pf.read_row_group(rg, columns=columns)

            # This should usually be unnecessary if the index is correct,
            # but keep it as a safety check.
            if h11 is not None or h21 is not None:
                df_check = table.select(
                    [c for c in ["h11", "h21"] if c in table.column_names]
                ).to_pandas()

                keep = pd.Series(True, index=df_check.index)

                if h11 is not None:
                    keep &= df_check["h11"] == int(h11)

                if h21 is not None:
                    keep &= df_check["h21"] == int(h21)

                if not keep.all():
                    table = table.filter(pa.array(keep.to_numpy()))

            if limit is not None:
                remaining = limit - rows_collected
                if table.num_rows > remaining:
                    table = table.slice(0, remaining)

            tables.append(table)
            rows_collected += table.num_rows

    if len(tables) == 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    result = pa.concat_tables(tables, promote_options="default")

    return result.to_pandas() if as_pandas else result
