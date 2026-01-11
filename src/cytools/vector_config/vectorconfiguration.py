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

# external imports
from collections.abc import Iterable
import numpy as np
import regfans
from typing import Union

# vector configuration imports
from .fan import Fan

# core CYTools imports
from cytools import Cone, Polytope


class VectorConfiguration(regfans.VectorConfiguration):
    """
    This class handles definition/operations on vector configurations. It is
    analogous to the Polytope class. This object can be triangulated,
    making a simplicial fan.

    **Description:**
    Constructs a `VectorConfiguration` object describing a lattice vector
    configuration. This is handled by the hidden [`__init__`](#__init__)
    function.

    **Arguments:**
    - `vectors`:    The vectors defining the VC.
    - `labels`:     A list of labels for the vectors. Only integral labels are
                    allowed.
    - `eps`:        Threshold for checking for non-integral vectors.
    - `gale_basis`: An optional basis for the gale transform. If provided, then
                    the gale transform will be put a basis such that the
                    submatrix given by these labels equals the identity.

    **Returns:**
    Nothing.
    """
    def __init__(self, *args, **kwargs):
        """
        **Description:**
        Initializes a `VectorConfiguration` object.

        **Arguments:**
        - `vectors`:    The vectors defining the VC.
        - `labels`:     A list of integer labels for the vectors. Only integral
                        labels are allowed.
        - `eps`:        Threshold for checking for non-integral vectors.
        - `gale_basis`: An optional basis for the gale transform. If provided,
                        then the gale transform will be put a basis such that
                        the submatrix given by these labels equals the identity.

        **Returns:**
        Nothing.
        """
        # call regfans' initializer
        super().__init__(*args, **kwargs)

        # some Polytope info
        p = Polytope(self.vectors(), labels=self.labels)
        self._is_reflexive = p.is_reflexive()
        self._poly = {self.labels: p}

        # some toric info
        if self._is_reflexive:
            self._divisor_basis = p.glsm_basis(include_points_interior_to_facets=False)
            # this is what's internally used by regfans.VectorConfiguration
            self._gale_basis    = self._divisor_basis 
    
    # hulls
    # -----
    def conical_hull(self, which: Union[int, Iterable[int]] = None) -> Cone:
        """
        **Description:**
        Compute the positive/conical hull of (some) vectors of the VC.

        If which = None, then the support over the entire VC is calculated.

        This is the most natural hull (i.e., not the convex hull) to take.

        **Arguments:**
        - `which`: Either a single label, for which the single corresponding
            vector will be returned, or a list of labels.

        **Returns:**
        The associated conical hull.
        """
        return Cone(rays=self.vectors(which=which))

    # aliases
    positive_hull = conical_hull
    pos = conical_hull
    coni = conical_hull
    cone = conical_hull

    def convex_hull(self, which: Union[int, Iterable[int]] = None) -> Polytope:
        """
        **Description:**
        Compute the convex hull of (some) vectors of the VC.

        If which = None, then the support over the entire VC is calculated.

        This hull is not very natural from a VC perspective... mainly used to
        connect to polytopes/point configurations.

        **Arguments:**
        - `which`: Either a single label, for which the single corresponding
            vector will be returned, or a list of labels.

        **Returns:**
        The associated convex hull.
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
        """
        **Description:**
        Return whether or not the convex hull is reflexive.

        **Arguments:**
        None.

        **Returns:**
        True if the convex hull is reflexive. False otherwise.
        """
        return self._is_reflexive

    @property
    def divisor_basis(self) -> Iterable[int]:
        """
        **Description:**
        Return the divisor basis corresponding to the Polytope class.

        As labels.

        **Arguments:**
        None.

        **Returns:**
        The divisor basis, as labels.
        """
        if self._is_reflexive:
            return self._divisor_basis

    @property
    def divisor_basis_inds(self) -> Iterable[int]:
        """
        **Description:**
        Return the divisor basis corresponding to the Polytope class.

        As labels.

        **Arguments:**
        None.

        **Returns:**
        The divisor basis, as indices.
        """
        if self._is_reflexive:
            # map labels to inds
            return self.divisor_basis-1

    # misc regularity methods
    # -----------------------
    def central_fan(self) -> "Fan":
        """
        **Description:**
        Generate the central fan of the vector configuration. Can be defined
        as lifting each vector by a height of 1.

        **Arguments:**
        None.

        **Returns:**
        The central fan.
        """
        return self.subdivide(heights=[1 for _ in self.labels])

    def gale(self) -> "ArrayLike":
        """
        **Description:**
        Compute the gale transform of the config.

        I.e., a basis of the null-space of the vectors.

        Will automatically be put in the divisor basis iff the associated
        polytope is reflexive.

        **Arguments:**
        None.

        **Returns:**
        The gale transform.
        """
        if self.is_reflexive:
            return super().gale(set_basis=True)
        else:
            return super().gale(set_basis=False)
        
    def moving_cone(self, 
                    pushed_down: bool = False,
                    verbosity: int = 0) -> Cone:
        """
        **Description:**
        Compute the moving cone of the vector configuration.

        Equiv to the support of the subfan of fine, regular triangulations.

        **Arguments:**
        - `pushed_down`: Whether to give the moving cone in h11-dim space or
        (h11+4)-dim space.
        - `verbosity`: The verbosity level.

        **Returns:**
        The moving cone.
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
        """
        **Description:**
        Subdivide the vector configuration either by specified cells/simplices
        or by heights.

        **Arguments:**
        - `heights`:   The heights to lift the vectors by.
        - `cells`:     The cells to use in the triangulation.
        - `backend`:   The lifting backend. Use 'qhull'.
        - `tol`:       Numerical tolerance used.
        - `verbosity`: The verbosity level. Higher is more verbose

        **Returns:**
        The resultant subdivision.
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
    """
    **Description:**
    Construct the VectorConfiguration associated to the polytope.

    **Arguments:**
    - `include_points_interior_to_facets`: Whether to include points interior
        to facets

    **Returns:**
    The associated VectorConfiguration.
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
        self.points(which=vc_labels), labels=vc_labels
    )

    if include_points_interior_to_facets:
        self._vc_yesfacet = vc
    else:
        self._vc_nofacet = vc
    
    return self.vc(include_points_interior_to_facets=include_points_interior_to_facets)
Polytope.vc = vc

# give Cone a method to directly generate its VC
def vc(self):
    """
    **Description:**
    Construct the VectorConfiguration associated to the cone.

    **Arguments:**
    None.

    **Returns:**
    The associated VectorConfiguration.
    """
    return regfans.VectorConfiguration(self.rays())
Cone.vc = vc
