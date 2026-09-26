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
# Description:  This module contains tools designed to perform vector
#               configuration computations.
# -----------------------------------------------------------------------------

"""Vector configurations.

This module defines the
[`VectorConfiguration`][cytools.vector_config.vectorconfiguration.VectorConfiguration]
class. Importing it also adds a `vc` method to
[`Polytope`][cytools.polytope.Polytope] and [`Cone`][cytools.cone.Cone] that
constructs the associated vector configuration.
"""

# external imports
from collections.abc import Iterable
import numpy as np
import regfans
from typing import Union

# vector configuration imports
from .fan import Fan

# core CYTools imports
from cytools import Cone, Polytope

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike



class VectorConfiguration(regfans.VectorConfiguration):
    r"""A configuration of integral vectors.

    This is the analogue of [`Polytope`][cytools.polytope.Polytope] (which describes
    a point configuration) for vector configurations. A vector configuration can be
    triangulated, giving a simplicial [`Fan`][cytools.vector_config.fan.Fan].

    This class extends `VectorConfiguration` from the
    [regfans](https://github.com/natemacfadden/regfans) package, and all arguments are
    passed to its constructor. These are, in order:

    - `vectors` (`ArrayLike`): The vectors, one per row. They must be integral (up
      to the tolerance `eps`). Zero vectors are removed with a warning.
    - `labels` (`Iterable[int] | None`): Integer labels for the vectors. If not
      specified, the vectors are labelled $1, \dots, n$ (label 0 is reserved for the
      origin).
    - `eps` (`float`): The tolerance used to check that the vectors are integral.
      Defaults to `1e-4`.
    - `gale_basis` (`Iterable[int] | None`): The labels of the vectors defining the
      basis of the Gale transform, i.e. such that the rows of
      [`gale`][cytools.vector_config.vectorconfiguration.VectorConfiguration.gale]
      corresponding to these labels form the identity matrix. They must index a
      unimodular submatrix. If not specified and the convex hull of the vectors is
      reflexive, the GLSM basis of the convex hull is used (see
      [`is_reflexive`][cytools.vector_config.vectorconfiguration.VectorConfiguration.is_reflexive]).

    Typically, a vector configuration is constructed from a reflexive polytope with
    `Polytope.vc`, in which case its vectors are the lattice points of the polytope
    (other than the origin) with the same labels.

    Args:
        *args (Any): Positional arguments passed to the regfans constructor (see
            above).
        **kwargs (Any): Keyword arguments passed to the regfans constructor (see
            above).

    Raises:
        ValueError: If the vectors are not integral or the labels are not integers.

    Examples:
        We construct a vector configuration, compute its Gale transform, and
        triangulate it.

        >>> from cytools.vector_config import VectorConfiguration
        >>> vc = VectorConfiguration([[1,0],[0,1],[-1,-1]])
        >>> vc.labels
        (1, 2, 3)
        >>> vc.gale()
        array([[1],
               [1],
               [1]])
        >>> vc.triangulate().cones()
        ((1, 2), (1, 3), (2, 3))
    """
    def __init__(self, *args, **kwargs):
        # call regfans' initializer
        super().__init__(*args, **kwargs)

        # Polytope/toric info is computed lazily (see `is_reflexive`); many
        # VCs never need the convex-hull view, and the common construction
        # path (Polytope.vc) hands us a ready-made Polytope afterwards
        self._is_reflexive = None
        self._poly = dict()

    # hulls
    # -----
    def conical_hull(self, which: Union[int, Iterable[int]] = None) -> Cone:
        """Compute the conical (positive) hull of some or all of the vectors.

        This is the natural hull to take for a vector configuration, as opposed to the
        convex hull.

        Also available as `positive_hull`, `pos`, `coni`, and `cone`.

        Args:
            which (Iterable[int] | None): The labels of the vectors to use. If not
                specified, all vectors are used, giving the support of the
                configuration.

        Returns:
            Cone: The cone generated by the vectors.

        Examples:
            >>> from cytools.vector_config import VectorConfiguration
            >>> vc = VectorConfiguration([[1,0],[0,1],[-1,-1]])
            >>> vc.conical_hull([1,2])
            A 2-dimensional rational polyhedral cone in RR^2 generated by 2 rays
        """
        return Cone(rays=self.vectors(which=which))

    # aliases
    positive_hull = conical_hull
    pos = conical_hull
    coni = conical_hull
    cone = conical_hull

    def convex_hull(self, which: Union[int, Iterable[int]] = None) -> Polytope:
        """Compute the convex hull of some or all of the vectors.

        This hull is not very natural from the point of view of vector configurations,
        but it connects them to polytopes and point configurations. In the resulting
        [`Polytope`][cytools.polytope.Polytope], the chosen vectors keep their labels,
        while its other lattice points (such as the origin) are assigned new labels.
        The polytope is cached for each choice of `which`.

        Also available as `conv` and `polytope`.

        Args:
            which (Iterable[int] | None): The labels of the vectors to use. If not
                specified, all vectors are used.

        Returns:
            Polytope: The convex hull of the vectors, with the vectors keeping their
                labels.

        Examples:
            >>> from cytools.vector_config import VectorConfiguration
            >>> vc = VectorConfiguration([[1,0],[0,1],[-1,-1]])
            >>> vc.convex_hull().is_reflexive()
            True
        """
        if which is None:
            which = self.labels

        # cache computed polytopes
        which = tuple(which)
        if which not in self._poly:
            self._poly[which] = Polytope(self.vectors(which), labels=which)

        return self._poly[which]

    # aliases
    conv = convex_hull
    polytope = convex_hull

    # properties of the PC
    # --------------------
    @property
    def is_reflexive(self) -> bool:
        """Return whether the convex hull of the vectors is reflexive.

        Translations are not allowed, i.e. the origin must be the unique interior
        lattice point. The result is cached. If the convex hull is reflexive and no Gale
        basis was specified, this also sets the Gale basis (and thus the
        [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis])
        to the GLSM basis of the convex hull, computed without points interior to
        facets.

        Returns:
            bool: Whether the convex hull is reflexive.
        """
        if self._is_reflexive is None:
            p = self.convex_hull()
            self._is_reflexive = p.is_reflexive(allow_translations=False)

            # some toric info
            if self._is_reflexive and (self._gale_basis is None):
                self._gale_basis = p.glsm_basis(
                    include_points_interior_to_facets=False)

        return self._is_reflexive

    @property
    def divisor_basis(self) -> Iterable[int]:
        """Return the basis of divisors, as labels.

        This is the basis in which the Gale transform is written, which for a vector
        configuration obtained from a polytope agrees with
        [`Polytope.glsm_basis`][cytools.polytope.Polytope.glsm_basis].

        Returns:
            Iterable[int]: The labels of the vectors whose divisors form the basis.

        Raises:
            NotImplementedError: If the convex hull of the vectors is not reflexive.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.vc().divisor_basis
            array([5, 6])
        """
        if not self.is_reflexive:
            raise NotImplementedError(
                "A divisor basis is only defined when the convex hull of the "
                "vector configuration is reflexive."
            )

        return self._gale_basis

    @property
    def divisor_basis_inds(self) -> Iterable[int]:
        r"""Return the basis of divisors, as indices.

        This is
        [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis]
        with every label shifted down by one, so it assumes that the vectors are
        labelled $1, \dots, n$ (as is the case for configurations constructed from
        polytopes).

        Returns:
            np.ndarray: The indices of the vectors whose divisors form the basis.

        Raises:
            NotImplementedError: If the convex hull of the vectors is not reflexive.
        """
        # map labels to inds
        # (this assumes the labels are the contiguous range 1, ..., n, which
        #  is the convention used everywhere here - including when points
        #  interior to facets are excluded, since the excluded points are
        #  relabelled away rather than leaving gaps)
        return self.divisor_basis - 1

    # misc regularity methods
    # -----------------------
    def central_fan(self) -> "Fan":
        """Compute the central fan of the vector configuration.

        This is the regular subdivision obtained by lifting every vector to height 1.
        When the origin is in the interior of the convex hull of the vectors, its
        cones are the cones over the proper faces of the convex hull. It need not be
        simplicial or fine.

        Returns:
            Fan: The central fan.

        Examples:
            >>> from cytools.vector_config import VectorConfiguration
            >>> vc = VectorConfiguration([[1,1],[1,-1],[-1,1],[-1,-1],[1,0]])
            >>> vc.central_fan().cones()
            ((1, 2, 5), (1, 3), (2, 4), (3, 4))
        """
        return self.subdivide(heights=[1 for _ in self.labels])

    def gale(self, set_basis: bool = None, **kwargs) -> "ArrayLike":
        r"""Compute the Gale transform of the vector configuration.

        This is a basis of the integral linear relations among the vectors, i.e. of the
        kernel of the matrix whose columns are the vectors. It is returned as an
        $n \times (n-d)$ matrix, where $n$ is the number of vectors and $d$ is the
        dimension, whose $i$-th row is the Gale vector (the GLSM charges) of the $i$-th
        vector. When it is written in a basis, the rows corresponding to the labels in
        [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis]
        form the identity matrix. The result is cached.

        Args:
            set_basis (bool | None): Whether to write the Gale transform in the chosen
                basis. If not specified, it is written in the basis if and only if the
                convex hull of the vectors is reflexive.
            **kwargs (Any): Additional keyword arguments passed to the regfans method.

        Returns:
            np.ndarray: The Gale transform.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.vc().gale()
            array([[1, 0],
                   [9, 3],
                   [6, 2],
                   [1, 0],
                   [1, 0],
                   [0, 1]])
        """
        if set_basis is None:
            set_basis = self.is_reflexive

        return super().gale(set_basis=set_basis, **kwargs)

    def moving_cone(self, 
                    pushed_down: bool = False,
                    verbosity: int = 0) -> Cone:
        """Compute the moving cone of the vector configuration.

        This is the intersection, over all vectors, of the cones generated by the Gale
        vectors of all other vectors. It is the support of the part of the secondary fan
        corresponding to fine regular triangulations.

        Args:
            pushed_down (bool): Whether to return the cone in the space of the Gale
                transform, of dimension $n-d$ ($h^{1,1}$ for a configuration from a
                favorable reflexive 4D polytope), instead of the $n$-dimensional space
                of heights. In the latter case, the cone contains the linear height
                functions as a lineality space.
            verbosity (int): The verbosity level.

        Returns:
            Cone: The moving cone.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.vc().moving_cone(pushed_down=True).rays()
            array([[3, 1],
                   [1, 0]])
        """
        glsm = self.gale().T

        hyps = []
        for i in range(glsm.shape[1]):
            if verbosity >= 1:
                msg  =  "Computing the cone corresponding to deleting "
                msg += f"i={i}/{glsm.shape[1]}..."
                print(msg)
            hyps.append(Cone(rays=np.delete(glsm, i, axis=1).T).hyperplanes())
        hyps = np.vstack(hyps)

        # pull up
        if not pushed_down:
            hyps = hyps @ glsm

        # map to cone, return
        return Cone(hyperplanes=hyps)

    # override lifting to give CYTools Fan object
    # -------------------------------------------
    def triangulate(self, *args, **kwargs):
        """Subdivide the vector configuration, either by heights or by specified cells.

        This calls the regfans method of the same name and converts the result into a
        CYTools [`Fan`][cytools.vector_config.fan.Fan]. If neither heights nor cells are
        given, a fine triangulation is computed from Delaunay-like heights.

        Also available as `subdivide`.

        Args:
            *args (Any): Positional arguments passed to the regfans method.
            **kwargs (Any): Keyword arguments passed to the regfans method. These are:
                `heights` (`ArrayLike | None`), the heights to lift the vectors by;
                `cells` (`ArrayLike | None`), the cones of the fan, each given by the
                labels of its vectors; `tol` (`float`), the numerical tolerance used
                when curing heights; `backend` (`str | None`), the lifting backend,
                either `"cgal"` or `"ppl"` (the default is CGAL when no heights are
                given and PPL otherwise); `make_fine` (`bool | None`), whether to make
                the triangulation fine (the default is true only when no heights are
                given); `check_heights` (`bool`), whether to check that the heights lie
                in the secondary cone of the output; `cure_heights` (`bool`), whether to
                try to cure heights that do not; and `verbosity` (`int`), the verbosity
                level.

        Returns:
            Fan: The resulting subdivision.

        Examples:
            >>> from cytools.vector_config import VectorConfiguration
            >>> vc = VectorConfiguration([[1,1],[1,-1],[-1,1],[-1,-1],[1,0]])
            >>> vc.triangulate().cones()
            ((1, 3), (1, 5), (2, 4), (2, 5), (3, 4))
            >>> vc.triangulate(heights=[1,1,1,1,2]).cones()
            ((1, 2), (1, 3), (2, 4), (3, 4))
        """
        fan = super().subdivide(*args, **kwargs)
        fan = Fan.from_regfans(fan) # cast to CYTools type
        return fan

    subdivide = triangulate

# misc
# ----
# give Polytope a method to directly generate its VC
def vc(self,
    include_points_interior_to_facets: bool = False) -> "VectorConfiguration":
    """Construct the vector configuration associated to the polytope.

    Its vectors are the lattice points of the polytope other than the origin, with
    the same labels, and its Gale transform is written in the GLSM basis of the
    polytope (see [`Polytope.glsm_basis`][cytools.polytope.Polytope.glsm_basis]). The
    polytope is expected to be reflexive. The result is cached, and its convex hull
    is set to the polytope itself.

    This function is attached to [`Polytope`][cytools.polytope.Polytope] as
    `Polytope.vc` when `cytools.vector_config` is imported.

    Args:
        include_points_interior_to_facets (bool): Whether to include the points
            interior to facets.

    Returns:
        VectorConfiguration: The associated vector configuration.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> vc = p.vc()
        >>> vc.labels
        (1, 2, 3, 4, 5, 6)
        >>> vc.vectors()
        array([[-1, -1, -6, -9],
               [ 0,  0,  0,  1],
               [ 0,  0,  1,  0],
               [ 0,  1,  0,  0],
               [ 1,  0,  0,  0],
               [ 0,  0, -2, -3]])
    """
    # see if we already know the answer
    if include_points_interior_to_facets:
        if hasattr(self, '_vc_yesfacet'):
            return self._vc_yesfacet
    else:
        if hasattr(self, '_vc_nofacet'):
            return self._vc_nofacet

    # determine which points set to use
    if include_points_interior_to_facets:
        poly_labels = self.labels
    else:
        poly_labels = self.labels_not_facet

    # get the associated lattice points
    label_origin = self.label_origin
    vc_labels = tuple(sorted(
        [label for label in poly_labels if label != label_origin]
    ))

    # save the VC (for caching purposes)
    vc = VectorConfiguration(
        self.points(which=vc_labels), labels=vc_labels,
        gale_basis = self.glsm_basis(include_points_interior_to_facets=False)
    )

    vc._poly = {vc.labels: self}

    if include_points_interior_to_facets:
        self._vc_yesfacet = vc
    else:
        self._vc_nofacet = vc

    return vc
Polytope.vc = vc

# give Cone a method to directly generate its VC
def vc(self):
    r"""Construct the vector configuration given by the rays of the cone.

    This function is attached to [`Cone`][cytools.cone.Cone] as `Cone.vc` when
    `cytools.vector_config` is imported.

    Returns:
        VectorConfiguration: The vector configuration whose vectors are the rays of
            the cone, labelled $1, \dots, n$.

    Examples:
        >>> from cytools import Cone
        >>> Cone([[1,0],[0,1]]).vc().vectors()
        array([[1, 0],
               [0, 1]])
    """
    return VectorConfiguration(self.rays())
Cone.vc = vc
