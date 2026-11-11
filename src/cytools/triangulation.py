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
# Description:  This module contains tools designed to compute triangulations.
# -----------------------------------------------------------------------------

"""Triangulations of lattice point configurations.

This module defines the [`Triangulation`][cytools.triangulation.Triangulation]
class, along with helper functions to compute triangulations with different
backends and to generate all triangulations or random triangulations of a point
configuration. Triangulations are normally obtained from
[`Polytope.triangulate`][cytools.polytope.Polytope.triangulate] and related
functions rather than constructed directly.
"""

# 'standard' imports
from collections.abc import Iterable
import copy
import itertools
import math
import warnings

# 3rd party imports
import flint
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
import triangulumancer

# CYTools imports
from cytools.cone import Cone
from cytools.toricvariety import ToricVariety
from cytools.utils import gcd_list, lll_reduce

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from cytools.calabiyau import CalabiYau
    from cytools.polytope import Polytope
    from cytools.polytopeface import PolytopeFace


class Triangulation:
    """A triangulation of a lattice point configuration.

    This class handles triangulations of point configurations made of lattice
    points of a polytope. It can compute various properties of the
    triangulation, such as its simplices, secondary cone, and GKZ vector, and
    it can construct a [`ToricVariety`][cytools.toricvariety.ToricVariety] or
    [`CalabiYau`][cytools.calabiyau.CalabiYau] object if the triangulation is
    suitable.

    The triangulation is either given explicitly by a list of simplices, or it
    is computed with one of the backends, usually as the regular triangulation
    induced by a height vector (see the `heights` argument). The points
    are specified by their labels in the ambient polytope, and their order in
    `pts` determines the indexing of the points of the triangulation (e.g. in
    the output of [`points`][cytools.triangulation.Triangulation.points] and
    in the `as_indices` options).

    Warning:
        Generally, objects of this class should not be constructed directly by
        the end user. Instead, they should be created by functions such as
        [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate], which
        pick the points and ordering conventions appropriately.

    Args:
        poly (Polytope): The ambient polytope of the points to be triangulated.
        pts (ArrayLike): The points to be triangulated, specified by their
            labels in `poly`.
        make_star (bool): Whether to turn the triangulation into a star
            triangulation with respect to the origin. With the CGAL backend
            this is done by decreasing the height of the origin until it is in
            every simplex. Otherwise it is done by deleting the simplices
            containing interior lines and connecting the origin to the
            boundary simplices, which requires the polytope to be reflexive.
            This flag is ignored if the origin is not one of the points.
        simplices (ArrayLike | None): The simplices specifying the
            triangulation, each given as a list of point labels. This is useful
            when a triangulation was previously computed and needs to be used
            again. If not specified, the triangulation is computed from
            `heights` with the chosen backend.
        check_input_simplices (bool): Whether to check that the input
            simplices define a valid triangulation.
        heights (list | None): The heights specifying the regular
            triangulation. Only used if `simplices` is not given, and only
            supported by the CGAL and QHull backends. If not specified, the
            triangulation is constructed based on the backend: CGAL uses the
            Delaunay triangulation, QHull uses random heights near the Delaunay
            heights, and TOPCOM computes a fine triangulation.
        check_heights (bool): Whether to check that the heights define a unique
            triangulation, i.e. that they do not lie on a wall of the secondary
            cone. If they do, they are discarded and recomputed when
            [`heights`][cytools.triangulation.Triangulation.heights] is called.
        backend (str): The backend used to compute the triangulation. Options
            are `"cgal"`, `"qhull"`, and `"topcom"`. CGAL is the default as it
            is very fast and robust.
        verbosity (int): The verbosity level.
        seed (int | None): Seed for the height perturbation used by the QHull
            backend. Only that backend perturbs its heights, so this has no
            effect for CGAL or TOPCOM. If not specified, the perturbation is
            drawn afresh each time and the triangulation is not reproducible.

    Raises:
        ValueError: If `pts` is empty or contains labels that are not in
            `poly`, if the backend is invalid, if the number of heights does
            not match the number of points, or if the input simplices are
            empty, have the wrong dimension, contain unknown labels, or do not
            form a valid triangulation.

    Examples:
        We construct a triangulation of a polytope. Since this class is not
        intended to be initialized by the end user, we create it via
        [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate]. In
        this example the polytope is reflexive, so by default the triangulation
        is fine, regular, and star. Also, since the polytope is reflexive, by
        default only the lattice points not interior to facets are included in
        the triangulation.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> t = p.triangulate()
        >>> print(t)
        A fine, regular, star triangulation of a 4-dimensional point configuration...
    """

    def __init__(
        self,
        poly: "Polytope",
        pts: ArrayLike,
        make_star: bool = False,
        simplices: ArrayLike = None,
        check_input_simplices: bool = True,
        heights: list = None,
        check_heights: bool = True,
        backend: str = "cgal",
        verbosity: int = 1,
        seed: int = None,
    ) -> None:
        # input checking
        # --------------
        # points
        pts_set = set(pts)
        if len(pts_set) == 0:
            raise ValueError("Need at least 1 point.")
        elif not pts_set.issubset(poly.labels):
            raise ValueError("All point labels must exist in the polytope.")

        # backend
        backend = backend.lower()
        if backend not in ["qhull", "cgal", "topcom", None]:
            raise ValueError(
                f"Invalid backend, {backend}. "
                + f"Options: {['qhull', 'cgal', 'topcom', None]}."
            )

        # initialize attributes
        # ---------------------
        self.clear_cache()

        # process the inputs
        # ------------------
        # backend
        self._backend = backend

        # polytope
        self._poly = poly

        # points
        # (ordered to match poly.label ordering...)
        self._labels = pts
        self._labels2inds = None
        self._labels2optPts = None

        # dimension
        self._dim_ambient = poly.ambient_dim()
        self._dim = (
            np.linalg.matrix_rank([list(pt) + [1] for pt in poly.points(which=pts)]) - 1
        )
        self._is_fulldim = self._dim == self._dim_ambient

        # simplices
        if simplices is not None:
            self._simplices = sorted([sorted(s) for s in simplices])
            if len(self._simplices) == 0:
                raise ValueError(
                    "The list of simplices is empty. A triangulation must "
                    + "contain at least one simplex."
                )
            self._simplices = np.asarray(self._simplices)
        else:
            self._simplices = None

        # Parse points
        # ------------
        # find index of origin
        if self.poly._label_origin in self.labels:
            self._origin_index = list(self.labels).index(self.poly._label_origin)
        else:
            # triangulation doesn't include origin
            self._origin_index = -1
            make_star = False

        # Save input triangulation, or construct it
        # -----------------------------------------
        heights = copy.deepcopy(heights)
        if self._simplices is not None:
            self._heights = None

            # check dimension
            if self._simplices.shape[1] != self._dim + 1:
                simp_dim = self._simplices.shape[1] - 1
                error_msg = (
                    f"Dimension of simplices, ({simp_dim}), "
                    + "doesn't match polytope dimension (+1), "
                    + f"{self._dim} (+1)..."
                )
                raise ValueError(error_msg)

            if check_input_simplices:
                # only basic checks here. Most are in self.is_valid()
                # (do some checks here b/c otherwise self.is_valid() hits
                # errors)
                simp_labels = set(self._simplices.flatten())
                if any([(lab not in self._labels) for lab in simp_labels]):
                    unknown = simp_labels.difference(self._labels)
                    error_msg = (
                        f"Simplices had labels {simp_labels}; "
                        + f"triangulation has labels {self._labels}. "
                        + f"Labels {unknown} are not recognized..."
                    )
                    raise ValueError(error_msg)
                # if min(simp_inds)<0:
                #    error_msg = f"A simplex had index, {min(simp_inds)}, " +\
                #                f"out of range [0,{len(self.points())-1}]"
                #    raise ValueError(error_msg)
                # elif max(simp_inds)>=len(self.points()):
                #    error_msg = f"A simplex had index, {max(simp_inds)}, " +\
                #                f"out of range [0,{len(self.points())-1}]"
                #    raise ValueError(error_msg)

            # convert simplices to star
            if make_star:
                _to_star(self)

            # ensure simplices define valid triangulation
            if check_input_simplices:
                if not self.is_valid(verbosity=verbosity - 1):
                    msg = "Simplices don't form valid triangulation."
                    raise ValueError(msg)
        else:
            # self._simplices==None... construct simplices from heights

            self._is_regular = None if (backend == "qhull") else True
            self._is_valid = True

            default_triang = heights is None
            if default_triang:
                # construct the heights
                if backend is None:
                    raise ValueError(
                        "Simplices must be specified when workingwithout a backend"
                    )

                # Heights need to be perturbed around the Delaunay heights for
                # QHull or the triangulation might not be regular. If using
                # CGAL then they are not perturbed.
                if backend == "qhull":
                    # a local generator: reading the global stream made this
                    # reproducible only via a global seed, which is invisible
                    # from here. Pass `seed` instead.
                    rng = np.random.RandomState(seed)
                    heights = [
                        np.dot(p, p) + rng.normal(0, 0.05) for p in self.points()
                    ]
                elif backend == "cgal":
                    heights = [np.dot(p, p) for p in self.points()]
                else:  # TOPCOM
                    heights = None
            else:
                # check the heights
                if len(heights) != len(pts):
                    raise ValueError("Need same number of heights as points.")

            if heights is None:
                self._heights = None
            else:
                self._heights = np.asarray(heights)

            # Now run the appropriate triangulation function
            triang_pts = self.points(optimal=not self._is_fulldim)
            if backend == "qhull":
                self._simplices = _qhull_triangulate(triang_pts, self._heights)

                # map to labels
                self._simplices = [
                    [self._labels[i] for i in s] for s in self._simplices
                ]

                # convert to star
                if make_star:
                    pre_star = {frozenset(s) for s in self._simplices}
                    _to_star(self)
                    if pre_star != {frozenset(s) for s in self._simplices}:
                        # the star-ification changed the triangulation, so the
                        # heights no longer generate what we store in
                        # self._simplices. Drop them (they get recomputed via
                        # an LP whenever heights() is called).
                        self._heights = None
            elif backend == "cgal":
                self._simplices = _cgal_triangulate(triang_pts, self._heights)

                # can obtain star more quickly than in QHull by setting height
                # of origin to be much lower than others
                # (can't do this in QHull since it sometimes causes errors...)
                if make_star:
                    # get max/min of all heights other than the origin...
                    origin_mask = np.zeros(self._heights.size, dtype=bool)
                    origin_mask[self._origin_index] = True
                    heights_masked = np.ma.array(self._heights, mask=origin_mask)

                    # (don't assume that the origin sits at index 0!)
                    other_heights = heights_masked.compressed()
                    if len(other_heights):
                        origin_step = max(10, other_heights.max() - other_heights.min())
                    else:
                        origin_step = 10

                    # reduce height of origin until it's in all simplices
                    while any(self._origin_index not in s for s in self._simplices):
                        self._heights[self._origin_index] -= origin_step
                        self._simplices = _cgal_triangulate(triang_pts, self._heights)
                # map to labels
                self._simplices = [
                    [self._labels[i] for i in s] for s in self._simplices
                ]

            else:  # Use TOPCOM
                self._simplices = _topcom_triangulate(triang_pts)

                # map to labels
                self._simplices = [
                    [self._labels[i] for i in s] for s in self._simplices
                ]

                # convert to star
                if make_star:
                    _to_star(self)

            # check that the heights uniquely define this triangulation
            if check_heights and (self._heights is not None):
                self.check_heights(verbosity - default_triang)

        # Make sure that the simplices are sorted
        self._simplices = sorted([sorted(s) for s in self._simplices])

        # select the data-type carefully, as the simplices are some of the
        # biggest data stored in this class...
        max_label = max(self.poly.labels)
        if isinstance(max_label, int):
            if max_label < 2**8:
                dtype = np.uint8
            elif max_label < 2**16:
                dtype = np.uint16
            elif max_label < 2**32:
                dtype = np.uint32
            else:
                dtype = np.uint64

            self._simplices = np.array(self._simplices, dtype=dtype)
        else:
            self._simplices = np.array(self._simplices)

        # also set the heights data structure
        if self._heights is not None:
            self._heights = _normalize_heights(self._heights)

    # defaults
    # ========
    def __repr__(self) -> str:
        """Return a string describing the triangulation.

        The description states whether the triangulation is fine, regular (if
        known), and star, along with the dimension of the point configuration,
        its number of points, and the dimension of the ambient lattice.

        Returns:
            str: A string describing the triangulation.

        Examples:
            This function can be used to convert the triangulation to a string
            or to print information about the triangulation.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> poly_info = str(t)  # converts to string
            >>> print(t)  # prints triangulation info
            A fine, regular, star triangulation of a 4-dimensional point...
        """
        fine_str = "fine" if self.is_fine() else "non-fine"

        regular_str = ""
        if self._is_regular is not None:
            regular_str = ", "
            regular_str += "regular" if self._is_regular else "irregular"

        # star_str = ""
        # if self.polytope().is_reflexive():
        star_str = ", "
        star_str += "star" if self.is_star() else "non-star"

        return (
            "A "
            + fine_str
            + regular_str
            + star_str
            + f" triangulation of a {self.dim()}-dimensional "
            + f"point configuration with {len(self.labels)} points "
            + f"in ZZ^{self.ambient_dim()}"
        )

    def __eq__(self, other: "Triangulation") -> bool:
        """Return whether two triangulations are equal.

        Two triangulations are equal if they have the same ambient polytope and
        the same simplices.

        Args:
            other (Triangulation): The other triangulation that is being
                compared.

        Returns:
            bool: Whether the triangulations are equal. `False` is returned if
                `other` is not a `Triangulation`.

        Examples:
            We construct two triangulations and compare them.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t1 = p.triangulate(backend="qhull")
            >>> t2 = p.triangulate(backend="topcom")
            >>> t1 == t2
            True
        """
        # check if other is even a triangulation
        if not isinstance(other, Triangulation):
            return False

        # check that we have the same polytope and simplices
        our_simps = sorted(self.simplices().tolist())
        other_simps = sorted(other.simplices().tolist())

        return self.polytope() == other.polytope() and our_simps == other_simps

    def __ne__(self, other: "Triangulation") -> bool:
        """Return whether two triangulations are different.

        Args:
            other (Triangulation): The other triangulation that is being
                compared.

        Returns:
            bool: Whether the triangulations are different.

        Examples:
            We construct two triangulations and compare them.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t1 = p.triangulate(backend="qhull")
            >>> t2 = p.triangulate(backend="topcom")
            >>> t1 != t2
            False
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Return the hash value of the triangulation.

        The hash is computed from the points and the simplices of the
        triangulation, and it is cached.

        Returns:
            int: The hash value of the triangulation.

        Examples:
            We compute the hash value of a triangulation. Also, we construct a
            set and a dictionary with a triangulation, which make use of the
            hash function.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> h = hash(t)  # obtain hash value
            >>> d = {t: 1}  # create dictionary with triangulation keys
            >>> s = {t}  # create a set of triangulations
        """
        if self._hash is None:
            pt_hash = hash(tuple(sorted(tuple(v) for v in self.points())))
            simps = tuple(sorted(tuple(sorted(s)) for s in self.simplices()))
            self._hash = hash((pt_hash,) + simps)

        return self._hash

    # caching
    # =======
    def _dump(self) -> dict:
        """Return every instance variable of the triangulation.

        No copying is done, so be careful with mutability!

        Returns:
            dict: The instance's `__dict__`, mapping variable names to values.
        """
        return self.__dict__

    def clear_cache(self, recursive: bool = False) -> None:
        """Clear the cached results of any previous computation.

        Args:
            recursive (bool): Whether to also clear the cache of the ambient
                polytope.

        Examples:
            We construct a triangulation, compute its GKZ vector, clear the
            cache and then compute it again.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> gkz_phi = t.gkz_phi()  # computes the GKZ vector
            >>> t.clear_cache()
            >>> gkz_phi = t.gkz_phi()  # the GKZ vector is recomputed
        """
        self._automorphism_orbit = dict()
        self._gkz_phi = None
        self._hash = None
        self._is_fine = None
        self._is_regular = None
        self._is_star = None
        self._is_valid = None
        self._secondary_cone = [None] * 2
        self._sr_ideal = None
        self._toricvariety = None

        self._restricted_simplices = dict()

        if recursive:
            self.poly.clear_cache()

    # getters
    # =======
    @property
    def poly(self):
        """Return the ambient polytope of the triangulated points.

        Returns:
            Polytope: The ambient polytope.
        """
        return self._poly

    def polytope(self):
        """Return the ambient polytope of the triangulated points.

        This is the method form of the `poly` property.

        Returns:
            Polytope: The ambient polytope.

        Examples:
            We construct a triangulation and obtain its ambient polytope.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.polytope() == p
            True
        """
        return self.poly

    @property
    def labels(self):
        """Return the labels of the points in the triangulation.

        The order of the labels defines the indexing of the points of the
        triangulation.

        Returns:
            list | tuple: The labels of the points in the triangulation, in the
                order they were given at construction.

        Examples:
            We construct a triangulation and find the labels of its points.
            Since the polytope is reflexive, the points interior to facets
            (labels 7, 8, and 9 here) are not included by default.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.labels
            [0, 1, 2, 3, 4, 5, 6]
        """
        return self._labels

    def dimension(self) -> int:
        """Return the dimension of the triangulated point configuration.

        This is the dimension of the affine span of the points, which can be
        smaller than the dimension of the ambient lattice.

        Also available as `dim`.

        Returns:
            int: The dimension of the triangulation.

        Examples:
            We construct a triangulation and find its dimension.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> print(t.dimension())
            4
        """
        return self._dim

    # aliases
    dim = dimension

    def ambient_dimension(self) -> int:
        """Return the dimension of the ambient lattice.

        Also available as `ambient_dim`.

        Returns:
            int: The dimension of the ambient lattice.

        Examples:
            We construct a triangulation and find its ambient dimension.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.ambient_dim()
            4
        """
        return self.poly.ambient_dim()

    # aliases
    ambient_dim = ambient_dimension

    def is_fine(self) -> bool:
        """Return whether the triangulation is fine.

        A triangulation is fine if all the points are used as vertices of some
        simplex. Note that this only checks if it is fine with respect to the
        point configuration, not with respect to the full set of lattice points
        of the polytope.

        Returns:
            bool: Whether the triangulation is fine.

        Examples:
            We construct a triangulation and check if it is fine.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.is_fine()
            True
        """
        # check if we know the answer
        if self._is_fine is not None:
            return self._is_fine

        # calculate the answer
        N_used_pts = len(set.union(*[set(s) for s in self._simplices]))
        self._is_fine = N_used_pts == len(self.labels)

        # return
        return self._is_fine

    def is_star(self, star_origin=None) -> bool:
        """Return whether the triangulation is star.

        A triangulation is star with respect to a point if that point is a
        vertex of every simplex. By default the star origin is taken to be the
        origin, so for point configurations that don't contain the origin this
        function always returns `False` unless `star_origin` is specified.

        Args:
            star_origin (int | None): The label of the point to use as the
                star origin. If not specified, the origin is used.

        Returns:
            bool: Whether the triangulation is star.

        Examples:
            We construct a triangulation and check if it is star.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.is_star()
            True
        """
        # check if we know the answer
        if star_origin is not None:
            return all(star_origin in s for s in self._simplices)

        # calculate the answer
        if self._is_star is None:
            # if self._origin_index != -1:
            if self.poly._label_origin in self.labels:
                self._is_star = all(
                    self.poly._label_origin in s for s in self._simplices
                )
            else:
                self._is_star = False

        # return
        return self._is_star

    def get_toric_variety(self) -> ToricVariety:
        """Return the toric variety defined by the fan of the triangulation.

        The fan is obtained by taking the cones over the simplices of the
        triangulation. The result is cached.

        Returns:
            ToricVariety: The toric variety arising from the triangulation.

        Raises:
            NotImplementedError: If the triangulation is not full-dimensional or
                not star.

        Examples:
            We construct a triangulation and obtain the resulting toric
            variety.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.get_toric_variety()
            A simplicial compact 4-dimensional toric variety with 9 affine patches
        """
        # check if we know the answer
        if self._toricvariety is not None:
            return self._toricvariety

        # check if the question makes sense
        if not self._is_fulldim:
            raise NotImplementedError(
                "Toric varieties can only be "
                + "constructed from full-dimensional triangulations."
            )
        if not self.is_star():
            raise NotImplementedError(
                "Toric varieties can only be " + "constructed from star triangulations."
            )

        # initialize the answer
        self._toricvariety = ToricVariety(self)

        # return
        return self._toricvariety

    def get_cy(self, nef_partition: list[tuple[int]] = None) -> "CalabiYau":
        """Return the Calabi-Yau defined by the triangulation.

        By default this is the anti-canonical hypersurface in the toric variety
        defined by the fine, regular, star triangulation. If a nef-partition is
        specified then it returns the complete intersection Calabi-Yau that it
        specifies. This is a shortcut for calling
        [`ToricVariety.get_cy`][cytools.toricvariety.ToricVariety.get_cy] on the
        toric variety returned by
        [`get_toric_variety`][cytools.triangulation.Triangulation.get_toric_variety].

        Also available as `compute_cy` and `cy`.

        Note:
            Only Calabi-Yau 3-fold hypersurfaces are fully supported. Other
            dimensions and CICYs require enabling the
            [experimental features][cytools.config.enable_experimental_features]
            of CYTools.

        Args:
            nef_partition (list[tuple[int]] | None): A list of tuples of indices
                specifying a nef-partition of the polytope, which
                correspondingly defines a complete intersection Calabi-Yau. If
                not specified, the anti-canonical hypersurface is used.

        Returns:
            CalabiYau: The Calabi-Yau arising from the triangulation.

        Raises:
            NotImplementedError: If the triangulation is not full-dimensional or
                not star.

        Examples:
            We construct a triangulation and obtain the Calabi-Yau hypersurface
            in the resulting toric variety.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.get_cy()
            A Calabi-Yau 3-fold hypersurface with h11=2 and h21=272 in a...
        """
        return self.get_toric_variety().get_cy(nef_partition)

    # aliases
    compute_cy = get_cy
    cy = get_cy

    # points
    # ======
    def points(
        self,
        which=None,
        optimal: bool = False,
        as_poly_indices: bool = False,
        as_triang_indices: bool = False,
        check_labels: bool = True,
    ) -> np.ndarray:
        """Return the points of the triangulation.

        Note that these are not necessarily equal to the lattice points of the
        polytope they define. The points are returned in the order of
        [`labels`][cytools.triangulation.Triangulation.labels] (or in the order
        given by `which`).

        Also available as `pts`.

        Args:
            which (int | Iterable | None): Which points to return, specified by
                a single label or a list of labels (not indices). If not
                specified, all points of the triangulation are returned.
            optimal (bool): Whether to return the points in their optimal
                coordinates. For triangulations that are not full-dimensional,
                the points are expressed in an LLL-reduced lattice basis of
                their affine span, so they have `dim()` coordinates.
            as_poly_indices (bool): Whether to return the indices of the points
                in the full list of points of the polytope instead of their
                coordinates.
            as_triang_indices (bool): Whether to return the indices of the
                points in the list of points of the triangulation instead of
                their coordinates.
            check_labels (bool): Whether to check that the labels in `which`
                belong to the triangulation.

        Returns:
            np.ndarray | list[int]: The points, one per row. If
                `as_triang_indices` is true, a 1D array of indices is returned
                instead, and if `as_poly_indices` is true, a list of indices is
                returned.

        Raises:
            ValueError: If both `as_poly_indices` and `as_triang_indices` are
                true, or if `check_labels` is true and some of the labels in
                `which` are not labels of the triangulation.

        Examples:
            We construct a triangulation and print the points in the point
            configuration. Note that since the polytope is reflexive, by
            default only the lattice points not interior to facets are used.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.points()
            array([[ 0,  0,  0,  0],
                   [-1, -1, -6, -9],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0],
                   [ 1,  0,  0,  0],
                   [ 0,  0, -2, -3]])
            >>> t.points(which=[1, 6])
            array([[-1, -1, -6, -9],
                   [ 0,  0, -2, -3]])

            If the triangulation does not include all the points of the
            polytope, the indices in the triangulation and in the polytope can
            differ.

            >>> t2 = p.triangulate(points=[0,1,2,3,4,5,6,8])
            >>> t2.points(which=[8], as_triang_indices=True)
            array([7])
            >>> t2.points(which=[8], as_poly_indices=True)
            [8]
        """
        if as_poly_indices and as_triang_indices:
            raise ValueError(
                "Both as_poly_indices and as_triang_indices " + "can't be set to True."
            )

        # get the labels of the relevant points
        if which is None:
            # use all points in the face
            which = self.labels
        else:
            # if which is a single label, wrap it in an iterable
            if (not isinstance(which, Iterable)) and (which in self.labels):
                which = [which]

            # check if the input labels
            if check_labels:
                try:
                    if not set(which).issubset(self.labels):
                        raise ValueError(
                            f"Specified labels ({which}) aren't "
                            "subset of triangulation labels "
                            f"({self.labels})..."
                        )
                except Exception:
                    # print(f"Specified labels, {which}, likely aren't hashable.")
                    raise

        # return
        if as_triang_indices:
            if self._labels2inds is None:
                self._labels2inds = {v: i for i, v in enumerate(self._labels)}
            return np.array([self._labels2inds[label] for label in which])
        else:
            if optimal and (not as_poly_indices):
                dim_diff = self.ambient_dim() - self.dim()
                if dim_diff > 0:
                    # asking for optimal points, where the optimal value may
                    # differ from the entire polytope

                    # calculate the map from labels to optimal points
                    if self._labels2optPts is None:
                        pts_opt = self.points()
                        pts_opt = lll_reduce(pts_opt - pts_opt[0])[:, dim_diff:]

                        self._labels2optPts = dict()
                        for label, pt in zip(self.labels, pts_opt):
                            self._labels2optPts[label] = tuple(pt)

                    # return the relevant points
                    return np.array([self._labels2optPts[lab] for lab in which])

            # normal case
            return self.poly.points(
                which=which, optimal=optimal, as_indices=as_poly_indices
            )

    # aliases
    pts = points

    def points_to_labels(
        self, points: ArrayLike, is_optimal: bool = False
    ) -> "list | None":
        """Return the labels corresponding to the given points.

        It also accepts a single point, in which case it returns the
        corresponding label. This simply calls
        [`Polytope.points_to_labels`][cytools.polytope.Polytope.points_to_labels]
        on the ambient polytope.

        Args:
            points (ArrayLike): A point or a list of points.
            is_optimal (bool): Whether the points are given in the optimal
                coordinates of the polytope (`True`) or in the input
                coordinates (`False`).

        Returns:
            list | int: The list of labels corresponding to the given points,
                or the label of the point if only one is given.

        Examples:
            We construct a triangulation and find the labels of some points.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.points_to_labels([[-1,-1,-6,-9],[0,0,0,0]])
            [1, 0]
        """
        return self.poly.points_to_labels(points, is_optimal=is_optimal)

    def points_to_indices(
        self,
        points: ArrayLike,
        is_optimal: bool = False,
        as_poly_indices: bool = False,
    ) -> "np.ndarray | int":
        """Return the indices corresponding to the given points.

        It also accepts a single point, in which case it returns the
        corresponding index. By default, the indices refer to the list of
        points of the triangulation.

        Args:
            points (ArrayLike): A point or a list of points.
            is_optimal (bool): Whether the points are given in the optimal
                coordinates of the polytope (`True`) or in the input
                coordinates (`False`).
            as_poly_indices (bool): Whether to return the indices in the full
                list of points of the polytope instead of in the list of points
                of the triangulation.

        Returns:
            np.ndarray | list[int] | int: The indices corresponding to the given
                points, or the index of the point if only one is given. The
                indices are returned as an array, or as a list if
                `as_poly_indices` is true.

        Examples:
            We construct a triangulation and find the indices of some of its
            points.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.points_to_indices([-1,-1,-6,-9])  # a single point
            np.int64(1)
            >>> t.points_to_indices([[-1,-1,-6,-9],[0,0,0,0],[0,0,1,0]])
            array([1, 0, 3])
        """
        # check for empty input
        if len(points) == 0:
            return np.asarray([], dtype=int)

        # map single-point input into list case
        single_pt = len(np.array(points).shape) == 1
        if single_pt:
            points = [points]

        # grab labels, and then map to indices
        labels = self.points_to_labels(points, is_optimal=is_optimal)
        inds = self.points(
            which=labels,
            as_poly_indices=as_poly_indices,
            as_triang_indices=not as_poly_indices,
        )

        # get/return the indices
        if single_pt and len(inds):
            return inds[0]  # just return the single index
        else:
            return inds  # return a list of indices

    def triangulation_to_polytope_indices(self, inds) -> "np.ndarray | int":
        """Map indices of points of the triangulation to indices in the polytope.

        Takes a list of indices of points in the triangulation (i.e. positions
        in [`labels`][cytools.triangulation.Triangulation.labels]) and returns
        the indices of the same points in the full list of points of the
        polytope.

        Also available as `points_to_poly_indices`.

        Args:
            inds (Iterable[int]): The indices of points in the triangulation.

        Returns:
            list[int]: The corresponding indices in the list of points of the
                polytope.

        Examples:
            We construct a triangulation that excludes some points of the
            polytope, and map triangulation indices to polytope indices.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate(points=[0,1,2,3,4,5,6,8])
            >>> t.triangulation_to_polytope_indices([1, 7])
            [1, 8]
        """
        # (NM: remove this... use labels instead...)
        return self.points(which=[self._labels[i] for i in inds], as_poly_indices=True)

    points_to_poly_indices = triangulation_to_polytope_indices

    # triangulation
    # =============
    # sanity checks
    # -------------
    def is_valid(self, backend: str = None, verbosity: int = 0) -> bool:
        """Return whether the simplices form a valid triangulation.

        The simplices must cover the full volume of the convex hull of the
        points, and they cannot intersect at full-dimensional regions. If the
        triangulation is regular, this is checked by verifying that a height
        vector in its secondary cone generates the same triangulation.
        Otherwise, the volumes and pairwise intersections of the simplices are
        checked directly, which can be quite slow for large polytopes. The
        result is cached.

        Args:
            backend (str | None): The optimizer used to check regularity. The
                available options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically.
            verbosity (int): The verbosity level.

        Returns:
            bool: Whether the triangulation is valid.

        Examples:
            This function is useful when constructing a triangulation from a
            given set of simplices. Note that, unless
            `check_input_simplices=False` is passed, the validity of the input
            simplices is already checked when the triangulation is constructed.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> simps = [[0,1,2,3,4],[0,1,2,3,5],[0,1,2,4,5],[0,1,3,4,5],[0,2,3,4,5]]
            >>> t = p.triangulate(simplices=simps, check_input_simplices=False)
            >>> t.is_valid()
            True
        """
        # check if we know the answer
        if self._is_valid is not None:
            return self._is_valid

        # calculate the answer
        simps = self.simplices()
        # simps = np.array([self.points(s, as_triang_indices=True) for s in simps])

        if simps.shape[1] != self.dim() + 1:
            self._is_valid = False
            return self._is_valid

        # if simps.shape[0] == 1:
        #    # triangulation is trivial
        #    self._is_valid = True
        #    return self._is_valid

        # If the triangulation is presumably regular, then we can check if
        # heights inside the secondary cone yield the same triangulation.
        if self.is_regular(backend=backend):
            tmp_triang = Triangulation(
                self.polytope(),
                self.labels,
                heights=self.heights(),
                make_star=False,
            )

            self._is_valid = self == tmp_triang
            if verbosity >= 1:
                msg = f"By regularity, returning valid={self._is_valid}"
                print(msg)
            return self._is_valid

        # If it is not regular, then we check this using the definition of a
        # triangulation. This can be quite slow for large polytopes.

        # append a 1 to each point
        pts = dict(zip(self.labels, self.points(optimal=True)))
        pts_ext = {
            lab: list(pts[lab])
            + [
                1,
            ]
            for lab in self.labels
        }
        pts_all = np.array(list(pts.values()))
        # pts_ext = np.array([list(pt)+[1,] for pt in pts])

        # We first check if the volumes add up to the volume of the polytope
        v = 0
        for s in simps:
            tmp_v = abs(int(round(np.linalg.det([pts_ext[lab] for lab in s]))))

            if tmp_v == 0:
                self._is_valid = False
                if verbosity >= 1:
                    msg = f"By simp volume, returning valid={self._is_valid}"
                    print(msg)
                return self._is_valid

            v += tmp_v

        poly_vol = int(round(ConvexHull(pts_all).volume * math.factorial(self._dim)))
        if v != poly_vol:
            self._is_valid = False
            if verbosity >= 1:
                msg = (
                    f"Simp volume ({v}) != poly volume ({poly_vol})... "
                    + f"returning valid={self._is_valid}"
                )
                print(msg)
            return self._is_valid

        # Finally, check if simplices have full-dimensional intersections
        for i, s1 in enumerate(simps):
            pts_1 = [pts_ext[i] for i in s1]

            for s2 in simps[i + 1 :]:
                pts_2 = [pts_ext[i] for i in s2]

                inters = Cone(pts_1).intersection(Cone(pts_2))

                if inters.is_solid():
                    self._is_valid = False
                    return self._is_valid

        # return
        self._is_valid = True
        return self._is_valid

    def check_heights(self, verbosity: int = 1) -> bool:
        """Check whether the stored heights uniquely define the triangulation.

        The heights uniquely define the triangulation if they do not lie on (or
        within $10^{-6}$ of) a wall of its secondary cone. If they do not, the
        stored heights are deleted so that they are recomputed the next time
        [`heights`][cytools.triangulation.Triangulation.heights] is called. The
        simplices are kept unchanged.

        Args:
            verbosity (int): The verbosity level. A message is printed when the
                heights are discarded if this is greater than 1.

        Returns:
            bool: `True` if the heights uniquely define the triangulation, and
                `False` if they are within $10^{-6}$ of a wall of the secondary
                cone.
        """
        # find hyperplanes that we are within eps of wall
        eps = 1e-6
        hyp_dist = np.matmul(self.secondary_cone().hyperplanes(), self._heights)
        not_interior = hyp_dist < eps

        # if we are close to any walls
        if not_interior.any():
            self._heights = None

            if verbosity > 1:
                print(
                    f"Triangulation: height-vector is within {eps}"
                    + " of a wall of the secondary cone... heights "
                    + "likely don't define a unique triangulation "
                    + "(common for Delaunay)..."
                )
                print(
                    "Will recalculate more appropriate heights "
                    + "for the (semi-arbitrarily chosen) "
                    + "triangulation..."
                )
            return False

        return True

    # main method
    # -----------
    def simplices(
        self,
        on_faces_dim: int = None,
        on_faces_codim: int = None,
        split_by_face: bool = False,
        as_np_array: bool = True,
        as_indices: bool = False,
    ) -> "set | np.ndarray":
        """Return the simplices of the triangulation.

        It also has the option of restricting the simplices to faces of the
        polytope of a particular dimension or codimension. This restriction is
        useful for checking CY equivalences from different triangulations. The
        simplices are specified by the labels of their vertices unless
        `as_indices` is true.

        Also available as `simps`.

        Args:
            on_faces_dim (int | None): Restrict the simplices to faces of the
                polytope of the given dimension.
            on_faces_codim (int | None): Restrict the simplices to faces of the
                polytope of the given codimension (relative to the dimension of
                the triangulation). Ignored if `on_faces_dim` is specified.
            split_by_face (bool): Whether to return the restricted simplices
                separately for each face, in the order of
                [`Polytope.faces`][cytools.polytope.Polytope.faces], instead of
                merging them. Only used when restricting to faces.
            as_np_array (bool): Whether to return the simplices as a numpy
                array. Otherwise, they are returned as a set of frozensets.
            as_indices (bool): Whether to specify the simplices by the indices
                of their vertices in the list of points of the triangulation
                instead of by their labels.

        Returns:
            np.ndarray | set[frozenset] | list: The simplices of the
                triangulation, either as a 2D array with one simplex per row
                (sorted) or as a set of frozensets. If `split_by_face` is true,
                a list with one such array or set per face is returned.

        Raises:
            ValueError: If the requested face dimension is negative or larger
                than the dimension of the triangulation.

        Examples:
            We construct a triangulation and find its simplices. We also find
            the simplices that lie on 2-faces of the polytope.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.simplices()
            array([[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 5],
                   [0, 1, 2, 4, 5],
                   [0, 1, 3, 4, 5],
                   [0, 2, 3, 4, 5]], dtype=uint8)
            >>> t.simplices(on_faces_dim=2)
            array([[1, 2, 3],
                   [1, 2, 4],
                   [1, 2, 5],
                   [1, 3, 4],
                   [1, 3, 5],
                   [1, 4, 5],
                   [2, 3, 4],
                   [2, 3, 5],
                   [2, 4, 5],
                   [3, 4, 5]])
        """
        # input parsing
        if (on_faces_dim is None) and (on_faces_codim is None):
            out = self._simplices

            # cast to indices
            if as_indices:
                l2i = {lab: i for i, lab in enumerate(self.labels)}
                out = [[l2i[lab] for lab in s] for s in out]

            if as_np_array:
                return np.array(out)
            else:
                return {frozenset(simp) for simp in out}
        elif on_faces_dim is not None:
            faces_dim = on_faces_dim
        else:
            faces_dim = self.dim() - on_faces_codim

        if faces_dim < 0 or faces_dim > self.dim():
            raise ValueError("Invalid face dimension.")

        # restrict simplices to faces
        if faces_dim not in self._restricted_simplices:
            full_simp = [frozenset(s) for s in self._simplices]

            # get face indices
            face_labels = [
                frozenset(face.labels) for face in self.polytope().faces(faces_dim)
            ]

            # actually restrict
            restricted = []
            for f in face_labels:
                restricted.append(set())
                for s in full_simp:
                    inters = f & s
                    if len(inters) == faces_dim + 1:
                        restricted[-1].add(inters)

            self._restricted_simplices[faces_dim] = restricted

        # return
        out = self._restricted_simplices[faces_dim]

        # cast to indices
        if as_indices:
            l2i = {lab: i for i, lab in enumerate(self.labels)}
            out = [[frozenset([l2i[lab] for lab in s]) for s in face] for face in out]

        if split_by_face:
            if as_np_array:
                out = [np.array(sorted(sorted(s) for s in face)) for face in out]
            elif out is self._restricted_simplices[faces_dim]:
                # don't hand out the cached (mutable) sets themselves
                out = [set(face) for face in out]
        else:
            out = set().union(*out)
            if as_np_array:
                return np.array(sorted(sorted(s) for s in out))

        return out

    # aliases
    simps = simplices

    def restrict(
        self,
        restrict_to: ["PolytopeFace"] = None,
        restrict_dim: int = 2,
        as_poly: bool = False,
        verbosity: int = 0,
    ):
        """Restrict the triangulation to one or more faces of the polytope.

        The restriction of the triangulation to a face consists of the
        intersections of the simplices with the face that are full-dimensional
        simplices of the face.

        Args:
            restrict_to (PolytopeFace | Iterable[PolytopeFace] | None): The
                face or faces to restrict to. If not specified, the
                triangulation is restricted to each face of dimension
                `restrict_dim`.
            restrict_dim (int): The dimension of the faces to restrict to. Only
                used if `restrict_to` is not specified.
            as_poly (bool): Whether to return each restriction as a
                `Triangulation` object instead of as a list of simplices.
            verbosity (int): The verbosity level.

        Returns:
            list | Triangulation: If a single face is given, the restriction to
                that face, as a sorted list of simplices (each a sorted list of
                point labels) or as a `Triangulation` if `as_poly` is true.
                Otherwise, a list with the restriction to each face.

        Examples:
            We construct a triangulation and restrict it to the 2-faces of the
            polytope.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.restrict()[:3]
            [[[1, 2, 3]], [[1, 2, 4]], [[1, 2, 5]]]
            >>> t.restrict(as_poly=True)[0]
            A fine, regular, non-star triangulation of a 2-dimensional point...
        """
        # determine what to restrict to
        if restrict_to is None:
            # default as each dim-face
            restrict_to = self.polytope().faces(restrict_dim)

        if isinstance(restrict_to, Iterable):
            # recursivesly call for each face given
            return [
                self.restrict(
                    restrict_to=f,
                    as_poly=as_poly,
                    verbosity=verbosity,
                )
                for f in restrict_to
            ]

        # if above checks failed, then single face was input...

        # output variable
        face_simps = set()

        # grab basic information
        dim = restrict_to.dim()

        # find the labels in our face
        label_set = set(restrict_to.labels)

        # find restriction of simplices to our faces
        for simp in self.simplices():
            restricted = label_set.intersection(simp)
            restricted = sorted(restricted)

            if len(restricted) == (dim + 1):
                # full dimension restriction
                face_simps.add(tuple(restricted))

        if as_poly:
            restricted = restrict_to.triangulate(
                simplices=face_simps, verbosity=verbosity - 1
            )
            restricted._ambient_triangulation = self
            return restricted
        else:
            return sorted([sorted([int(i) for i in simp]) for simp in face_simps])

    # regularity
    # ----------
    def secondary_cone(
        self,
        backend: str = None,
        include_points_not_in_triangulation: bool = True,
        as_cone: bool = True,
        on_faces_dim: int = None,
        use_cache: bool = True,
    ) -> Cone:
        r"""Compute the secondary cone of the triangulation.

        This cone is also called the cone of strictly convex piecewise linear
        functions (CPL cone). It lives in the space of height vectors, with one
        coordinate per point of the triangulation. The triangulation is regular
        if and only if this cone is solid (i.e. full-dimensional), in which case
        the points in its strict interior are exactly the heights that give
        rise to the triangulation.

        This function also allows the computation of the secondary cone of the
        $N$-skeleton, with $N$ set by `on_faces_dim`. This cone is defined as
        either of the following equivalent descriptions:

        1. the intersection of the secondary cones of the restrictions of this
           triangulation to all $N$-dimensional faces of the polytope, or
        2. the union of the secondary cones of all triangulations with the same
           restrictions to $N$-dimensional faces.

        Any point in the strict interior of this cone corresponds to heights
        that generate a triangulation with the same restrictions to the
        $N$-faces. Some cases of interest are:

        - $N$ equal to the dimension of the triangulation, which gives the
          secondary cone of the triangulation.
        - $N=2$, which gives the union of the secondary cones of all
          triangulations with the same 2-face restrictions (for FRSTs, these
          give rise to equivalent Calabi-Yau hypersurfaces). This is akin to
          $\mathcal{K}_\cup$.

        The secondary cone of the triangulation is cached, while the
        secondary cones of skeletons are always recomputed.

        Also available as `cpl_cone`.

        Args:
            backend (str | None): The backend to use. Options are `"native"`,
                which uses a native implementation of an algorithm by Berglund,
                Katz and Klemm, and `"topcom"`, which uses differences of GKZ
                vectors of neighboring triangulations. If not specified,
                `"native"` is used when possible, i.e. when the triangulation is
                fine or `include_points_not_in_triangulation` is false, and
                `"topcom"` otherwise. Only the native backend is supported for
                skeletons.
            include_points_not_in_triangulation (bool): Whether to include the
                points of the point configuration that are not vertices of any
                simplex. Excluding them can be done to check regularity faster,
                but it cannot be used if the actual cone in the secondary fan
                is needed.
            as_cone (bool): Whether to return a
                [`Cone`][cytools.cone.Cone] object. Otherwise, the list of
                hyperplane normals defining the cone is returned.
            on_faces_dim (int | None): Compute the secondary cone of the
                restriction to each face of this dimension and return their
                intersection. This has the interpretation of fixing the
                triangulations of these faces, while being agnostic about the
                rest of the structure. If not specified, the full secondary
                cone of the triangulation is computed.
            use_cache (bool): Whether to use the cached secondary cone, if
                available. The newly computed cone is cached either way.

        Returns:
            Cone | list: The secondary cone, or the list of hyperplane normals
                defining it if `as_cone` is false.

        Raises:
            ValueError: If an invalid backend is specified.

        Examples:
            We construct a triangulation and find its secondary cone.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.secondary_cone()
            A rational polyhedral cone in RR^7 defined by 3 hyperplanes
        """
        # input sanitization
        # ------------------
        args_id = int(include_points_not_in_triangulation)

        # set backend
        backends = (None, "native", "topcom")
        if backend not in backends:
            raise ValueError(f"Options for backend are: {backends}")

        if backend is None:
            if self.is_fine() or (not include_points_not_in_triangulation):
                backend = "native"
            else:
                backend = "topcom"

        if (
            backend == "native"
            and (not self.is_fine())
            and include_points_not_in_triangulation
        ):
            warnings.warn(
                "Native backend is not supported when including "
                "points that are not in the triangulation. Using "
                "TOPCOM..."
            )
            backend = "topcom"

        # set on_faces_dim
        if on_faces_dim is None:
            on_faces_dim = self.dim()

        # compute the cone
        # ----------------
        # want the secondary cone of the N-skeleton for N<dim
        if on_faces_dim < self.dim():
            if backend == "topcom":
                print("Only native backend is currently supported for skeletons")
                backend = "native"

            # intersect the hyperplanes for each N-face
            hyps = []
            for face in self.restrict(restrict_dim=on_faces_dim, as_poly=True):
                hyps += face.secondary_cone(
                    backend=backend,
                    include_points_not_in_triangulation=include_points_not_in_triangulation,
                    as_cone=False,
                    use_cache=False,
                )

            # restrict to unique hyperplanes
            hyps = sorted(set(hyps))

            # return in the desired format
            if as_cone:
                # clean up empty hyperplanes
                if len(hyps) == 0:
                    hyps = np.zeros((0, len(self.labels)), dtype=int)

                return Cone(hyperplanes=hyps, check=False)
            else:
                return hyps

        # want the secondary cone of the triangulation
        if use_cache and (self._secondary_cone[args_id] is not None):
            # we already know the answer!
            hyps = self._secondary_cone[args_id]

            # return in the desired format
            if as_cone:
                # clean up empty hyperplanes
                if len(hyps) == 0:
                    hyps = np.zeros((0, len(self.labels)), dtype=int)

                return Cone(hyperplanes=hyps, check=False)
            else:
                return self._secondary_cone[args_id]

        # we don't yet know the answer... calculate it now!
        if backend == "topcom":
            # calculate hyperplanes via GKZ-vectors from neighboring
            # triangulations
            hyps = []
            for t in self.neighbor_triangulations(
                only_fine=False, only_regular=False, only_star=False
            ):
                hyps.append(t.gkz_phi() - self.gkz_phi())

        elif backend == "native":
            # calculate hyperplanes via the null-space of a homogenization of
            # the points in adjacent simplcies

            # get the ambient labels
            if hasattr(self, "_ambient_triangulation"):
                ambient_labels = self._ambient_triangulation.labels
            else:
                ambient_labels = self.labels

            # get the ambient dimension and a map from labels to indices
            ambient_dim = len(ambient_labels)
            labels2inds = {v: i for i, v in enumerate(ambient_labels)}

            # get the simplices and the actual dimension
            simps = [set(s) for s in self.simplices()]
            dim = self.dim()

            # container for matrix
            m = np.zeros((dim + 1, dim + 2), dtype=int)
            m[-1, :] = 1  # (homogenization)

            # container for hyperplane normal
            full_v = np.zeros(ambient_dim, dtype=int)

            # small optimization
            # (star triangulations all share 0th point, the origin, so it can
            #  be removed from consideration, effectively reducing dimension)
            if self.is_star():
                simps = [s - {self.poly._label_origin} for s in simps]
                dim -= 1

                m[:-1, -1] = self.points(which=self.poly._label_origin, optimal=True)

            # find adjacent simplices (those sharing a facet) via a
            # facet->simplices incidence map, rather than scanning all O(n^2)
            # pairs of simplices
            facet_to_simps = {}
            for idx, s in enumerate(simps):
                for pt in s:
                    facet = frozenset(s - {pt})
                    facet_to_simps.setdefault(facet, []).append(idx)

            # cache the optimal coordinates of every point once, rather than
            # re-fetching them for each adjacent pair
            all_labels = list(set().union(*simps))
            all_opt = self.points(which=all_labels, optimal=True)
            opt_pts = {lbl: all_opt[i] for i, lbl in enumerate(all_labels)}

            # calculate the hyperplanes
            null_vecs = set()
            for idxs in facet_to_simps.values():
                # an interior facet joins exactly two simplices; a boundary
                # facet joins one (and is not flippable)
                if len(idxs) != 2:
                    continue
                s1 = simps[idxs[0]]
                s2 = simps[idxs[1]]

                # the two opposite vertices and the shared facet
                diff_pts = list(s1 ^ s2)
                comm_pts = list(s1 & s2)

                for c, pt in enumerate(diff_pts):
                    m[:-1, c] = opt_pts[pt]
                for c, pt in enumerate(comm_pts):
                    m[:-1, 2 + c] = opt_pts[pt]

                # calculate nullspace/hyperplane ineq
                # (quicker than the determinant method)
                v = flint.fmpz_mat(m.tolist()).nullspace()[0]
                v = np.array(v.transpose().tolist()[0], dtype=int)

                # ensure the sign is correct
                if v[0] < 0:
                    v *= -1

                # Reduce the vector
                g = math.gcd(*v.tolist())
                if g != 1:
                    v //= g

                # Construct the full vector (including all points)
                for k, pt in enumerate(diff_pts):
                    full_v[labels2inds[pt]] = v[k]
                for k, pt in enumerate(comm_pts):
                    full_v[labels2inds[pt]] = v[k + 2]

                if self.is_star():
                    full_v[labels2inds[self.poly._label_origin]] = v[-1]

                null_vecs.add(tuple(full_v))

                # clear full_v
                for pt in diff_pts:
                    full_v[labels2inds[pt]] = 0
                for pt in comm_pts:
                    full_v[labels2inds[pt]] = 0

            # organize the hyperplanes
            hyps = list(null_vecs)

        # save the answer
        self._secondary_cone[args_id] = hyps

        # return
        # (do via calling the same function to use above cached answer)
        return self.secondary_cone(
            backend=backend,
            include_points_not_in_triangulation=include_points_not_in_triangulation,
            as_cone=as_cone,
        )

    # aliases
    cpl_cone = secondary_cone

    def is_regular(self, backend: str = None) -> bool:
        """Return whether the triangulation is regular.

        A triangulation is regular if it is induced by some height vector,
        which happens if and only if its secondary cone is solid. The result is
        cached.

        Args:
            backend (str | None): The optimizer used for the computation. The
                available options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically.

        Returns:
            bool: Whether the triangulation is regular.

        Examples:
            We construct a triangulation and check if it is regular.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.is_regular()
            True
        """
        # check if we know the answer
        if self._is_regular is not None:
            return self._is_regular

        # calculate the answer
        if self.simplices().shape[0] == 1:
            self._is_regular = True
        else:
            C = self.secondary_cone(include_points_not_in_triangulation=False)
            self._is_regular = C.is_solid(backend=backend)

        # return
        return self._is_regular

    def heights(self, integral: bool = False, backend: str = None) -> np.ndarray:
        """Return a height vector that generates the triangulation.

        If the triangulation was constructed from heights (and they uniquely
        define it), those heights are returned, normalized to be non-negative
        with minimum 0. Otherwise, a point in the strict interior of the
        secondary cone is found with an optimizer and cached. Heights found
        this way are not normalized. An exception is raised if no height
        vector can be found, either because the optimizer failed or because
        the triangulation is not regular.

        Args:
            integral (bool): Whether to find an integral height vector. If the
                stored heights are non-integral, they are rescaled and rounded,
                which may introduce rounding errors (a warning is issued).
            backend (str | None): The optimizer used for the computation. The
                available options are the backends of
                [`Cone.find_interior_point`][cytools.cone.Cone.find_interior_point].
                If not specified, it is picked automatically.

        Returns:
            np.ndarray: A height vector giving rise to the triangulation, with
                one entry per point of the triangulation. Integral heights are
                returned as a *signed* integer array (so that differences of
                heights behave correctly), while non-integral heights are
                returned as `float64`.

        Examples:
            We construct a triangulation and find a height vector that
            generates it.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> t = p.triangulate()
            >>> t.heights()
            array([0, 4, 1, 1, 1, 1], dtype=int8)
        """
        # check if we already know the heights...
        if self._heights is not None:
            heights_out = self._heights.copy()

            # convert to integral heights, if desired
            if integral and (heights_out.dtype == float):
                warnings.warn(
                    "Potential rounding errors... better to"
                    + "solve LP problem in this case..."
                )
                heights_out = np.rint(heights_out / gcd_list(heights_out))

            if integral:
                return heights_out.astype(int)
            else:
                return heights_out

        # need to calculate the heights
        Npts = len(self.labels)
        if (self._simplices.shape[0] == 1) and (self._simplices.shape[1] == Npts):
            # If the triangulation is trivial we just return a vector of zeros
            self._heights = np.zeros(Npts, dtype=(int if integral else float))
        else:
            # Otherwise we find a point in the secondary cone
            C = self.secondary_cone(include_points_not_in_triangulation=True)
            self._heights = C.find_interior_point(integral=integral, backend=backend)

        return self._heights.copy()

    # symmetries
    # ==========
    def automorphism_orbit(
        self,
        automorphism: "int | ArrayLike" = None,
        on_faces_dim: int = None,
        on_faces_codim: int = None,
    ) -> np.ndarray:
        """Return the orbit of the triangulation under polytope automorphisms.

        This returns all of the triangulations of the point configuration that
        can be obtained by applying one or more polytope automorphisms to the
        triangulation. It also has the option of restricting the simplices to
        faces of the polytope of a particular dimension or codimension. This
        restriction is useful for checking CY equivalences from different
        triangulations. The result is cached.

        Note:
            Depending on how the point configuration was constructed, it may be
            the case that the automorphism group of the point configuration is
            larger or smaller than the one of the polytope. This function only
            uses the subset of automorphisms of the polytope that are also
            automorphisms of the point configuration.

        Args:
            automorphism (int | ArrayLike | None): The index or list of indices
                of the polytope automorphisms to use, as ordered by
                [`Polytope.automorphisms`][cytools.polytope.Polytope.automorphisms].
                The orbit under the group generated by these automorphisms is
                computed. If not specified, all automorphisms are used.
            on_faces_dim (int | None): Restrict the simplices to faces of the
                polytope of the given dimension.
            on_faces_codim (int | None): Restrict the simplices to faces of the
                polytope of the given codimension. Ignored if `on_faces_dim` is
                specified.

        Returns:
            np.ndarray: The triangulations in the orbit, as a 3D array. Each
                entry is a sorted array of (possibly restricted) simplices, each
                specified by the sorted indices of its vertices in the list of
                points of the triangulation.

        Raises:
            NotImplementedError: If the point configuration is not
                full-dimensional.

        Examples:
            We construct a triangulation and find some of its automorphism
            orbits.

            >>> from cytools import Polytope
            >>> p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],
            ...               [2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],
            ...               [-1,0,1,1]])
            >>> t = p.triangulate()
            >>> orbit_all_autos = t.automorphism_orbit()
            >>> len(orbit_all_autos)
            36
            >>> orbit_all_autos_2faces = t.automorphism_orbit(on_faces_dim=2)
            >>> len(orbit_all_autos_2faces)
            36
            >>> orbit_sixth_auto = t.automorphism_orbit(automorphism=5)
            >>> len(orbit_sixth_auto)
            3
            >>> orbit_list_autos = t.automorphism_orbit(automorphism=[5,6,9])
            >>> len(orbit_list_autos)
            12
        """
        # sanity checks
        if not self._is_fulldim:
            raise NotImplementedError(
                "Automorphisms can only be computed "
                + "for full-dimensional point configurations."
            )

        # input parsing
        if (on_faces_dim is None) and (on_faces_codim is None):
            faces_dim = self.dim()
        elif on_faces_dim is not None:
            faces_dim = on_faces_dim
        else:
            faces_dim = self.dim() - on_faces_codim

        if automorphism is None:
            orbit_id = (None, faces_dim)
        else:
            if isinstance(automorphism, Iterable):
                orbit_id = (tuple(automorphism), faces_dim)
            else:
                orbit_id = ((automorphism,), faces_dim)

        # return answer if known
        if orbit_id in self._automorphism_orbit:
            return np.array(self._automorphism_orbit[orbit_id])

        # calculate orbit
        simps = self.simplices(as_indices=True, on_faces_dim=faces_dim)
        autos = self.polytope().automorphisms(as_dictionary=True)

        # collect automorphisms of polytope that are also automorphisms of
        # the triangulation point configuration
        good_autos = []
        for i in range(len(autos)):
            for j, k in autos[i].items():
                # user-input 'allowed' automorphims
                if (orbit_id[0] is not None) and (i > 0) and (i not in orbit_id[0]):
                    # non-trivial automorphism that isn't specifically allowed
                    break

                # check if jth and kth points are either both in the
                # triangulation, or both not in it
                if (self.poly.labels[j] in self.labels) != (
                    self.poly.labels[k] in self.labels
                ):
                    # oops! One pt is in the triangulation while other isn't!
                    break
            else:
                good_autos.append(i)

        # Finally, we
        #   1) reindex the 'good' automorphisms so that the indices match the
        #      indices of the point configuration and
        #   2) remove the bad automorphisms to make sure they are not used (we
        #      just replace them with None
        for i in range(len(autos)):
            # check if it is a 'bad' automorphism
            if i not in good_autos:
                autos[i] = None
                continue

            # it's a 'good' automorphism
            temp = {}
            for j, jj in autos[i].items():
                if (self.poly.labels[j] in self.labels) and (
                    self.poly.labels[jj] in self.labels
                ):
                    tmp_labels = [self.poly.labels[j], self.poly.labels[jj]]
                    idx_j, idx_jj = self.points(
                        tmp_labels, as_triang_indices=True, check_labels=False
                    )
                    temp[idx_j] = idx_jj
            autos[i] = temp

        # define helper function
        def apply_auto(auto):
            return tuple(sorted(tuple(sorted([auto[i] for i in s])) for s in simps))

        # apply the automorphisms
        orbit = set()
        for j, a in enumerate(autos):
            # skip if it is a 'bad' automorphism
            if a is None:
                continue

            # it's a 'good' automorphism
            orbit.add(apply_auto(a))

        if automorphism is not None:
            # keep applying autos until we stop getting new triangulations
            while True:
                new_triangs = []

                for simps in orbit:
                    for i in orbit_id[0]:
                        if autos[i] is None:
                            # 'bad' automorphism (doesn't preserve the point
                            # configuration), so it was filtered out above
                            continue
                        new_triang = apply_auto(autos[i])

                        if new_triang not in orbit:
                            new_triangs.append(new_triang)

                if len(new_triangs):
                    for t in new_triangs:
                        orbit.add(t)
                else:
                    break

        self._automorphism_orbit[orbit_id] = np.array(sorted(orbit))
        return np.array(self._automorphism_orbit[orbit_id])

    def is_equivalent(
        self,
        other: "Triangulation",
        use_automorphisms: bool = True,
        on_faces_dim: int = None,
        on_faces_codim: int = None,
    ) -> bool:
        """Return whether two triangulations are equivalent.

        The triangulations can be compared with or without allowing
        automorphism transformations, and with the option of restricting to
        faces of a particular dimension or codimension. Triangulations of
        different polytopes are never equivalent.

        Args:
            other (Triangulation): The other triangulation that is being
                compared.
            use_automorphisms (bool): Whether to check the equivalence using
                automorphism transformations. This flag is ignored for point
                configurations that are not full-dimensional.
            on_faces_dim (int | None): Restrict the simplices to faces of the
                polytope of the given dimension.
            on_faces_codim (int | None): Restrict the simplices to faces of the
                polytope of the given codimension. Ignored if `on_faces_dim` is
                specified.

        Returns:
            bool: Whether the triangulations are equivalent under the specified
                parameters.

        Examples:
            We construct two triangulations and check whether they are
            equivalent under various conditions.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,1,1,0],
            ...               [0,-1,-1,0],[0,0,0,1],[1,-2,1,1],[-2,2,1,-1],
            ...               [1,1,-1,-1]])
            >>> triangs_gen = p.all_triangulations()
            >>> t1 = next(triangs_gen)
            >>> t2 = next(triangs_gen)
            >>> t1.is_equivalent(t2)
            False
            >>> t1.is_equivalent(t2, on_faces_dim=2)
            True
            >>> t1.is_equivalent(t2, on_faces_dim=2, use_automorphisms=False)
            True
        """
        if self.polytope() != other.polytope():
            return False

        # check via automorphisms
        if self._is_fulldim and use_automorphisms:
            orbit1 = self.automorphism_orbit(
                on_faces_dim=on_faces_dim, on_faces_codim=on_faces_codim
            )
            orbit2 = other.automorphism_orbit(
                on_faces_dim=on_faces_dim, on_faces_codim=on_faces_codim
            )

            return (orbit1.shape == orbit2.shape) and all((orbit1 == orbit2).flat)

        # check via simplices
        simp1 = self.simplices(on_faces_dim=on_faces_dim, on_faces_codim=on_faces_codim)
        simp2 = other.simplices(
            on_faces_dim=on_faces_dim, on_faces_codim=on_faces_codim
        )

        return (simp1.shape == simp2.shape) and all((simp1 == simp2).flat)

    # flips
    # =====
    def neighbor_triangulations(
        self,
        only_fine: bool = False,
        only_regular: bool = False,
        only_star: bool = False,
        two_neighbors: bool = False,
        two_neighbors_track_flips: bool = False,
        backend: str = None,
        verbose: bool = False,
    ) -> list["Triangulation"]:
        """Return the triangulations that differ by one bistellar flip.

        The flips are computed with TOPCOM (through triangulumancer). There is
        the option of limiting the flips to fine, regular, and star
        triangulations. Regularity is checked with a separate optimizer, since
        checking it with TOPCOM is very slow for large polytopes. For fine
        2-dimensional triangulations with `only_fine=True`, a faster native
        implementation is used. A trivial triangulation (with a single
        simplex) has no neighbors computed, and an empty list is returned with
        a warning.

        Also available as `neighbors`.

        Args:
            only_fine (bool): Whether to restrict to fine triangulations.
            only_regular (bool): Whether to restrict to regular triangulations.
            only_star (bool): Whether to restrict to star triangulations.
            two_neighbors (bool): Whether to return the 2-neighbors instead.
                FRSTs with the same restrictions to 2-faces give equivalent
                Calabi-Yau hypersurfaces, so each FRST represents an
                equivalence class. The 2-neighbors are representatives of the
                classes that differ by a single diagonal flip of one 2-face,
                which gives the neighboring Calabi-Yaus more directly. The
                2-neighbors are fine and regular by construction, so
                `only_fine` and `only_regular` are ignored. They are star if
                `only_star` is true or if this triangulation is star.
            two_neighbors_track_flips (bool): Only valid with `two_neighbors`.
                If true, return `(triangulation, face_index, circuit)` tuples
                instead of bare triangulations. `face_index` indexes
                `polytope.faces(2)` and `circuit` is the sorted tuple of the
                four point labels spanning the flipped quadrilateral.
            backend (str | None): The backend used to check regularity. The
                options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically. When `two_neighbors` is
                true, it is instead passed on as the triangulation backend of
                [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate].
            verbose (bool): Currently unused.

        Returns:
            list: The triangulations that differ by one bistellar flip from the
                current triangulation, or the 2-neighbors if `two_neighbors` is
                true. If `two_neighbors_track_flips` is also true, the entries
                are `(triangulation, face_index, circuit)` tuples.

        Raises:
            ValueError: If `two_neighbors_track_flips` is true but
                `two_neighbors` is not.

        Examples:
            We construct a triangulation and find its neighbor triangulations.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [-1,-1,-1,-1]]).dual()
            >>> t = p.triangulate()
            >>> triangs = t.neighbor_triangulations()
            >>> len(triangs)  # print how many triangulations it found
            263

            We now find the 2-neighbors of an FRST, keeping track of the
            flipped 2-faces.

            >>> p = Polytope([[0,0,1,0],[-2,-2,-1,-2],[0,0,1,2],[-1,0,1,0],
            ...               [1,2,-2,-1],[-1,0,0,-1],[0,1,0,0],[1,0,0,0]])
            >>> t = p.triangulate()
            >>> nbrs = t.neighbor_triangulations(two_neighbors=True,
            ...                                  two_neighbors_track_flips=True)
            >>> [(face_index, circuit) for _, face_index, circuit in nbrs]
            [(0, (1, 6, 7, 9)), (21, (4, 8, 10, 12))]
        """
        # check for topcom bug
        if len(self.simplices()) == 1:
            warnings.warn(
                "Triangulation.neighbor_triangulations called "
                + "for trivial triangulation (1 simplex)... "
                + "Returning []! Fix TOPCOM!"
            )
            return []

        # the 2-neighbors are fine and regular FRSTs by construction, so
        # only_fine/only_regular need no handling here. only_star=False (the
        # default) is passed on as None so that _two_neighbors falls back to
        # its documented default of self.is_star().
        if two_neighbors:
            return self._two_neighbors(
                make_star=(True if only_star else None),
                backend=backend,
                two_neighbors_track_flips=two_neighbors_track_flips,
            )

        if two_neighbors_track_flips:
            raise ValueError(
                "two_neighbors_track_flips is only supported with two_neighbors=True."
            )

        # optimized method for 2D fine neighbors
        if self.is_fine() and (self.dim() == 2) and only_fine:
            return self._fine_neighbors_2d(only_regular=only_regular, backend=backend)

        # triangulumancer indexes points by their position in the point
        # configuration (i.e., row of self.points(...)), while self._simplices
        # is expressed in point *labels*. These agree only when the labels
        # happen to be 0..n-1, so translate explicitly in both directions.
        labels = list(self.labels)
        label2ind = {label: i for i, label in enumerate(labels)}
        simps_inds = [[label2ind[label] for label in s] for s in self._simplices]

        pc = triangulumancer.PointConfiguration(self.points(optimal=True))
        t = triangulumancer.Triangulation(pc, simps_inds)
        triangs_list = t.neighbors()

        # parse the outputs
        triangs = []
        for t in triangs_list:
            # TODO: Is this needed
            if not all(len(s) == self.dim() + 1 for s in t.simplices):
                continue

            # construct and check triangulation
            tri = Triangulation(
                self.poly,
                self.labels,
                simplices=[[labels[i] for i in s] for s in t.simplices],
                check_input_simplices=False,
            )
            if only_fine and (not tri.is_fine()):
                continue
            if only_star and (not tri.is_star()):
                continue
            if only_regular and (not tri.is_regular(backend=backend)):
                continue

            # keep it :)
            triangs.append(tri)
        return triangs

    # aliases
    neighbors = neighbor_triangulations

    def random_flips(
        self,
        N: int,
        only_fine: bool = None,
        only_regular: bool = None,
        only_star: bool = None,
        backend: str = None,
        seed: int = None,
    ) -> "Triangulation":
        """Return a triangulation obtained by performing random bistellar flips.

        At each step, a random neighbor triangulation (see
        [`neighbors`][cytools.triangulation.Triangulation.neighbor_triangulations])
        satisfying the requested restrictions is chosen. There is the option of
        limiting the flips to fine, regular, and star triangulations.
        Regularity is checked with a separate optimizer, since checking it with
        TOPCOM is very slow for large polytopes.

        Args:
            N (int): The number of bistellar flips to perform.
            only_fine (bool | None): Whether to restrict the flips to fine
                triangulations. If not specified, it is set to whether this
                triangulation is fine.
            only_regular (bool | None): Whether to restrict the flips to
                regular triangulations. If not specified, it is set to whether
                this triangulation is regular.
            only_star (bool | None): Whether to restrict the flips to star
                triangulations. If not specified, it is set to whether this
                triangulation is star.
            backend (str | None): The backend used to check regularity. The
                options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically.
            seed (int | None): A seed for the random number generator. This can
                be used to obtain reproducible results.

        Returns:
            Triangulation: A new triangulation obtained by performing `N`
                random flips.

        Raises:
            RuntimeError: If at some step no neighbor satisfies the requested
                restrictions.

        Examples:
            We construct a triangulation and perform 5 random flips to find a
            new triangulation.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [-1,-1,-1,-1]]).dual()
            >>> t = p.triangulate()
            >>> t.random_flips(5)  # takes a few seconds
            A fine, regular, star triangulation of a 4-dimensional point...
        """
        # parse inputs
        # (use a local generator so that we don't perturb NumPy's global RNG)
        rng = np.random.RandomState(seed)

        # unspecified restrictions default to the properties of this
        # triangulation (as documented above)
        if only_fine is None:
            only_fine = self.is_fine()
        if only_star is None:
            only_star = self.is_star()
        if only_regular is None:
            only_regular = self.is_regular(backend=backend)

        # take random flips
        curr_triang = self
        for n in range(N):
            neighbors = curr_triang.neighbor_triangulations(
                only_fine=False, only_regular=False, only_star=False
            )
            rng.shuffle(neighbors)

            for t in neighbors:
                # check that the triangulation meets the requirements
                if only_fine and (not t.is_fine()):
                    continue
                if only_star and (not t.is_star()):
                    continue
                if only_regular and (not t.is_regular(backend=backend)):
                    continue

                # accept the triangulation
                curr_triang = t
                break
            else:
                raise RuntimeError("There was an error in the random walk.")

        return curr_triang

    def _fine_neighbors_2d(
        self,
        only_regular: bool = False,
        backend: str = None,
    ) -> list["Triangulation"]:
        """Return the fine neighbors of a fine 2-dimensional triangulation.

        This is an optimized variant of
        [`neighbors`][cytools.triangulation.Triangulation.neighbor_triangulations]
        for triangulations that are 2-dimensional (the ambient dimension does
        not matter) and fine, when only fine neighbors are desired. In this
        case, a flip preserves fineness exactly when the two triangles sharing
        the flipped edge form a parallelogram, which makes this much quicker
        than the corresponding TOPCOM computation.

        Args:
            only_regular (bool): Whether to restrict to regular triangulations.
            backend (str | None): The backend used to check regularity. The
                options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically.

        Returns:
            list[Triangulation]: The triangulations that differ by one diagonal
                flip from the current triangulation.
        """
        simps_set = [set(s) for s in self._simplices]

        triangs = []
        for i, s1 in enumerate(simps_set):
            for _j, s2 in enumerate(simps_set[i + 1 :]):
                j = i + 1 + _j

                # the two simplices must share an edge (form a quadrilateral)
                inter = s1 & s2
                if len(inter) != 2:
                    continue

                # ...and the quadrilateral must be a parallelogram, so that
                # swapping the diagonal gives a fine triangulation
                other = s1.union(s2) - inter
                pts_inter = self.points(inter, check_labels=False)
                pts_other = self.points(other, check_labels=False)
                if (sum(pts_inter) != sum(pts_other)).any():
                    continue

                # flip the inner diagonal
                flipped = list(map(lambda p: sorted(other | {p}), inter))
                new_simps = self._simplices.copy()
                new_simps[i] = flipped[0]
                new_simps[j] = flipped[1]

                tri = Triangulation(
                    self.poly,
                    self.labels,
                    simplices=new_simps,
                    check_input_simplices=False,
                )
                if only_regular and (not tri.is_regular(backend=backend)):
                    continue

                triangs.append(tri)

        return triangs

    def _two_neighbors(
        self,
        make_star: bool = None,
        backend: str = None,
        two_neighbors_track_flips: bool = False,
    ) -> list["Triangulation"]:
        """Return the 2-neighbors of the triangulation.

        The 2-neighbors are the FRSTs reachable by a single 2D diagonal flip of
        one 2-face, extended back to a full triangulation via NTFE
        ([arXiv:2309.10855](https://arxiv.org/abs/2309.10855)). Flips that
        cannot be extended are skipped. The computation is parallelized
        according to `config.n_threads`. This is a private helper for
        `neighbor_triangulations(two_neighbors=True)`.

        Args:
            make_star (bool | None): Whether to produce star triangulations. If
                not specified, it is set to whether the current triangulation
                is star.
            backend (str | None): The triangulation backend passed to
                [`Polytope.triangulate`][cytools.polytope.Polytope.triangulate]
                when extending. If not specified, its default is used.
            two_neighbors_track_flips (bool): Whether to return
                `(triangulation, face_index, circuit)` tuples instead of bare
                triangulations. `face_index` indexes `polytope.faces(2)` and
                `circuit` is the sorted tuple of the four point labels spanning
                the flipped quadrilateral.

        Returns:
            list: The 2-neighbor triangulations, or tuples as described above
                if `two_neighbors_track_flips` is true. Each triangulation
                differs from the current one by a single 2-face diagonal flip
                and is fine and regular (and star if `make_star`); they give
                the neighboring Calabi-Yaus.

        Examples:
            We construct an FRST and find its 2-neighbors.

            >>> from cytools import Polytope
            >>> p = Polytope([[0,0,1,0],[-2,-2,-1,-2],[0,0,1,2],[-1,0,1,0],
            ...               [1,2,-2,-1],[-1,0,0,-1],[0,1,0,0],[1,0,0,0]])
            >>> t = p.triangulate()
            >>> neighbors = t.neighbor_triangulations(two_neighbors=True)
            >>> len(neighbors)  # print how many 2-neighbors it found
            2
        """
        from cytools.ntfe.ntfe import _2d_frt_cone_ineqs, _IncrementalLP

        if make_star is None:
            make_star = self.is_star()

        poly = self.polytope()
        npts = len(poly.labels)
        # the per-2-face triangulations, aligned to poly.faces(2)
        face_triangs = self.restrict(as_poly=True)
        n_faces = len(face_triangs)

        def ineqs(ft):
            return np.asarray(_2d_frt_cone_ineqs(ft, npts).toarray(), dtype=np.float64)

        # A single 2-face flip changes only that one face's inequalities, so
        # the other faces' blocks are stacked once into a warm incremental LP
        # and reused across all of a face's flips; only the flipped block is
        # pushed/popped per neighbor. The feasible heights are then extended to
        # a full triangulation.
        fixed = [ineqs(ft) for ft in face_triangs]

        # extend a chunk of (face index, flipped 2-face triangulation) pairs
        # to full triangulations. Each (face, flip) is a distinct 2-face
        # restriction, so it is its own dedup key.
        # `fixed` is passed as an argument (not captured) so it is not
        # re-cloudpickled with the closure into every worker
        def extend_chunk(chunk, fixed):
            out = []
            seen = set()
            lp = None
            cur = None
            for i, flipped, circuit in chunk:
                key = (
                    i,
                    tuple(
                        sorted(
                            tuple(sorted(int(x) for x in s))
                            for s in flipped.simplices()
                        )
                    ),
                )
                if key in seen:
                    continue
                if cur != i:
                    other = [
                        fixed[j] for j in range(n_faces) if j != i and len(fixed[j])
                    ]
                    base = np.vstack(other) if other else np.zeros((0, npts))
                    lp = _IncrementalLP(npts)
                    lp.push(base)
                    cur = i
                # skip flips that cannot be extended (no full triangulation
                # has that 2-face restriction); push() rolls back on its own
                if not lp.push(ineqs(flipped)):
                    continue
                heights = lp.witness()
                lp.pop()
                seen.add(key)
                tri = poly.triangulate(
                    heights=np.delete(heights, poly.labels_facet),
                    include_points_interior_to_facets=False,
                    make_star=make_star,
                    check_heights=False,
                    **({} if backend is None else {"backend": backend}),
                )
                out.append((tri, i, circuit) if two_neighbors_track_flips else tri)
            return out

        def _flip_circuit(pre, post):
            # the diagonal swap rewrites the 2-face's triangulation; the
            # symmetric difference of the two spans exactly the quadrilateral
            before = {tuple(sorted(int(v) for v in s)) for s in pre.simplices()}
            after = {tuple(sorted(int(v) for v in s)) for s in post.simplices()}
            return tuple(sorted({v for s in before ^ after for v in s}))

        # the (face, flip) tasks, grouped by face so each chunk reuses its
        # warm LP for a face's flips
        tasks = [
            (
                i,
                flipped,
                _flip_circuit(ft, flipped) if two_neighbors_track_flips else None,
            )
            for i, ft in enumerate(face_triangs)
            for flipped in ft._fine_neighbors_2d()
        ]

        # parallelization follows CYTools' global config.n_threads
        from multiprocessing import cpu_count

        from cytools import config

        n_threads = config.n_threads
        if n_threads is None:
            n_threads = 1 if len(tasks) < 32 else cpu_count()

        if n_threads == 1 or len(tasks) <= 1:
            return extend_chunk(tasks, fixed)

        # split the grouped tasks into contiguous chunks, one per worker
        import joblib

        n = joblib.effective_n_jobs(n_threads)
        size = -(-len(tasks) // n)
        chunks = [tasks[k * size : (k + 1) * size] for k in range(n)]
        chunks = [c for c in chunks if c]
        batches = joblib.Parallel(n_jobs=n_threads)(
            joblib.delayed(extend_chunk)(c, fixed) for c in chunks
        )
        return [t for batch in batches for t in batch]

    # misc
    # ====
    def gkz_phi(self) -> np.ndarray:
        r"""Return the GKZ phi vector of the triangulation.

        The $i$-th component is defined to be the sum of the normalized volumes
        of the simplices that have the $i$-th point as a vertex:

        $$
        \varphi^i = \sum_{\sigma\in\mathcal{T} \,:\, p_i\in\text{vert}(\sigma)}
        \text{vol}(\sigma).
        $$

        The components are ordered as the points of the triangulation. The
        result is cached.

        Returns:
            np.ndarray: The GKZ phi vector of the triangulation.

        Examples:
            We construct a triangulation and find its GKZ vector.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.gkz_phi()
            array([18, 12,  9, 12, 12, 12, 15])
        """
        # check if we know the answer
        if self._gkz_phi is not None:
            return np.array(self._gkz_phi)

        # calculate the answer
        pts_ext = {
            lab: list(pt)
            + [
                1,
            ]
            for lab, pt in zip(self.labels, self.points(optimal=True))
        }
        l2i = {lab: i for i, lab in enumerate(self.labels)}
        phi = np.zeros(len(pts_ext), dtype=int)

        for s in self._simplices:
            simp_vol = int(round(abs(np.linalg.det([pts_ext[lab] for lab in s]))))
            for lab in s:
                phi[l2i[lab]] += simp_vol

        # return
        self._gkz_phi = phi
        return np.array(self._gkz_phi)

    def sr_ideal(self) -> tuple:
        r"""Return the Stanley-Reisner ideal of the triangulation.

        The Stanley-Reisner (SR) ideal is generated by the monomials
        corresponding to the sets of points (other than the origin) that are
        not all contained in any single simplex. This function returns the
        minimal such sets, i.e. the *generators* of the ideal, and not their
        multiples. The result is cached.

        For example, the simplicial complex with simplices $\{1,2,3\}$,
        $\{1,2,4\}$, $\{2,3,4\}$, and $\{1,2,5\}$ has the sets
        $\{3,5\}$, $\{4,5\}$, $\{1,3,4\}$, $\{1,3,5\}$, $\{1,4,5\}$,
        $\{2,3,5\}$, $\{2,4,5\}$, and $\{3,4,5\}$ of points that do not
        appear together in a simplex. The SR ideal is generated by
        $x_3x_5$, $x_4x_5$, and $x_1x_3x_4$, while monomials such as
        $x_2x_3x_5$ are not included since they are multiples of these.

        Returns:
            tuple[tuple[int, ...], ...]: The generators of the SR ideal, each
                given as a sorted tuple of point labels. They are sorted by
                length and then lexicographically.

        Raises:
            NotImplementedError: If the triangulation is not star or not
                full-dimensional.

        Examples:
            We construct a triangulation and find its SR ideal.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.sr_ideal()
            ((np.uint8(1), np.uint8(4), 5), (np.uint8(2), np.uint8(3), 6))
        """
        # check if we know the answer
        if self._sr_ideal is not None:
            return self._sr_ideal

        # check if we can answer
        if not self.is_star() or not self._is_fulldim:
            raise NotImplementedError(
                "SR ideals can only be computed for "
                + "full-dimensional star triangulations."
            )

        # prep-work
        labels = set(self.labels) - {self.poly._label_origin}
        simplices = [labels.intersection(s) for s in self.simplices()]

        simplex_tuples = []
        for dd in range(1, self.dim() + 1):
            simplex_tuples.append(set())

            for s in simplices:
                simplex_tuples[-1].update(
                    frozenset(tup) for tup in itertools.combinations(s, dd)
                )

        # calculate the SR ideal
        SR_ideal, checked = set(), set()

        for i in range(len(simplex_tuples) - 1):
            for tup in simplex_tuples[i]:
                for j in labels:
                    k = tup.union((j,))

                    # skip if already checked
                    if (k in checked) or (len(k) != len(tup) + 1):
                        continue
                    else:
                        checked.add(k)

                    if k in simplex_tuples[i + 1]:
                        continue

                    # check it
                    for order in range(1, i + 1):
                        for t in itertools.combinations(tup, order):
                            if frozenset(t + (j,)) in SR_ideal:
                                break
                        else:
                            # frozenset(t+(j,)) was not in SR_ideal for any t
                            # at this order
                            continue

                        # there was a t at this order such that
                        # frozenset(t+(j,)) was in SR_ideal for some
                        break
                    else:
                        # frozenset(t+(j,)) was not in SR_ideal for any order
                        SR_ideal.add(k)

        # return
        self._sr_ideal = [tuple(sorted(s)) for s in SR_ideal]
        self._sr_ideal = tuple(sorted(self._sr_ideal, key=lambda x: (len(x), x)))
        return self._sr_ideal


def _normalize_heights(heights: ArrayLike) -> np.ndarray:
    """Normalize a height vector for storage in a `Triangulation` object.

    Heights are only defined up to an overall positive rescaling and an overall
    constant shift (both leave the induced regular subdivision unchanged), so
    they are normalized to be non-negative with minimum 0, and integral heights
    are also divided by their GCD. The normalization is *lossless*: the
    returned heights always generate the same triangulation as the input ones.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by the [`Triangulation`][cytools.triangulation.Triangulation]
        class when needed.

    Args:
        heights (ArrayLike): The height vector to normalize.

    Returns:
        np.ndarray: The normalized heights. Integral input (including floats
            with integral values) is returned as a *signed* integer array using
            the smallest dtype that fits the normalized values, while
            non-integral input is returned as `float64`.
    """
    heights = np.asarray(heights)

    is_integral = np.issubdtype(heights.dtype, np.integer) or (
        np.issubdtype(heights.dtype, np.floating)
        and bool(np.all(heights == np.rint(heights)))
    )

    if not is_integral:
        # Non-integral heights (e.g. the perturbed Delaunay heights used by
        # QHull). Dividing by a float "gcd" and rounding to integers is lossy
        # and can silently produce heights that generate a *different*
        # triangulation, so we keep the full float precision instead.
        heights = np.asarray(heights, dtype=np.float64)
        if heights.size:
            heights = heights - heights.min()
        return heights

    # integral heights: divide out the gcd and shift so the minimum is 0
    if np.issubdtype(heights.dtype, np.integer):
        heights = heights.astype(np.int64)
    else:
        heights = np.rint(heights).astype(np.int64)
    gcd = np.gcd.reduce(np.abs(heights)) if heights.size else 0
    if gcd > 1:
        heights = heights // gcd
    if heights.size:
        heights = heights - heights.min()

    # Pick the smallest dtype that fits. This *must* be done after the values
    # are final (rounding first, then choosing the dtype), and the dtype must
    # be signed -- callers routinely take differences of heights, and unsigned
    # arithmetic wraps around silently.
    max_h = int(np.abs(heights).max()) if heights.size else 0
    if max_h < 2**7:
        dtype = np.int8
    elif max_h < 2**15:
        dtype = np.int16
    elif max_h < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    return heights.astype(dtype)


def _to_star(triang: Triangulation) -> None:
    """Turn a triangulation into a star triangulation in place.

    This is done by deleting the simplices that contain interior lines and
    connecting the origin to the simplices on the boundary of the polytope.
    Only the boundary points of each facet are used, since points interior to
    facets are normally not included in the triangulation. The simplices of
    `triang` are replaced in place.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by the [`Triangulation`][cytools.triangulation.Triangulation]
        class when needed.

    Warning:
        This function only works for triangulations of reflexive polytopes.
        An `AssertionError` is raised if the polytope is not reflexive.

    Args:
        triang (Triangulation): The triangulation to convert to star.
    """
    # reflexivity check
    assert triang.poly.is_reflexive()

    # preliminary
    # (use boundary pts b/c pts interior to facets aren't normally included)
    facets = [f.labels_bdry for f in triang.poly.facets()]
    dim = len(triang._simplices[0]) - 1

    # map the simplices to being star
    star_triang = []

    for facet in facets:
        for simp in np.array(triang._simplices):
            overlap = simp[np.isin(simp, facet)].tolist()
            if len(overlap) == dim:
                star_triang.append([triang.poly._label_origin] + overlap)

    # update triang
    triang._simplices = np.array(sorted([sorted(s) for s in star_triang]))


def _qhull_triangulate(points: ArrayLike, heights: ArrayLike) -> np.ndarray:
    """Compute a regular triangulation using QHull.

    The points are lifted using the heights, and the triangulation is given by
    the projection of the lower facets of the convex hull of the lifted points
    (discarding the ones that project to degenerate simplices).

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by the [`Triangulation`][cytools.triangulation.Triangulation]
        class when using QHull as the backend.

    Args:
        points (ArrayLike): The points to triangulate, one per row. They must
            be full-dimensional.
        heights (ArrayLike): The heights defining the regular triangulation,
            one per point.

    Returns:
        np.ndarray: The simplices of the regular triangulation, one per row,
            specified by the indices of their vertices in `points`.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a triangulation using QHull.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> p.triangulate(backend="qhull")
        A fine, star triangulation of a 4-dimensional point configuration with 7...
    """
    lifted_points = [tuple(points[i]) + (heights[i],) for i in range(len(points))]
    hull = ConvexHull(lifted_points)

    # We first pick the lower facets of the convex hull
    low_fac = [
        hull.simplices[n] for n, nn in enumerate(hull.equations) if nn[-2] < 0
    ]  # The -2 component is the lifting dimension

    # Then we only take the faces that project to full-dimensional simplices
    # in the original point configuration
    lifted_points = [pt[:-1] + (1,) for pt in lifted_points]
    simp = [
        s
        for s in low_fac
        if int(round(np.linalg.det([lifted_points[i] for i in s]))) != 0
    ]

    return np.array(sorted([sorted(s) for s in simp]))


def _cgal_triangulate(points: ArrayLike, heights: ArrayLike) -> np.ndarray:
    """Compute a regular triangulation using CGAL.

    The computation is done through triangulumancer.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by the [`Triangulation`][cytools.triangulation.Triangulation]
        class when using CGAL as the backend.

    Args:
        points (ArrayLike): The points to triangulate, one per row.
        heights (ArrayLike): The heights defining the regular triangulation,
            one per point.

    Returns:
        np.ndarray: The simplices of the regular triangulation, one per row,
            specified by the indices of their vertices in `points`.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a triangulation using CGAL.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> p.triangulate(backend="cgal")
        A fine, regular, star triangulation of a 4-dimensional point...
    """

    pc = triangulumancer.PointConfiguration(points)
    simp = pc.triangulate_with_heights(heights).simplices

    return np.array(sorted([sorted(s) for s in simp]))


def _topcom_triangulate(points: ArrayLike) -> np.ndarray:
    """Compute a fine triangulation using TOPCOM.

    The computation is done through triangulumancer.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by the [`Triangulation`][cytools.triangulation.Triangulation]
        class when using TOPCOM as the backend.

    Args:
        points (ArrayLike): The points to triangulate, one per row.

    Returns:
        np.ndarray: The simplices of the triangulation, one per row, specified
            by the indices of their vertices in `points`.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a triangulation using TOPCOM.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> p.triangulate(backend="topcom")
        A fine, regular, star triangulation of a 4-dimensional point...
    """
    pc = triangulumancer.PointConfiguration(points)
    simp = pc.fine_triangulation().simplices

    return np.array(sorted([sorted(s) for s in simp]))


def all_triangulations(
    poly: "Polytope",
    pts: ArrayLike,
    only_fine: bool = False,
    only_regular: bool = False,
    only_star: bool = False,
    star_origin: int = None,
    backend: str = None,
    raw_output: bool = False,
) -> "Iterator[Triangulation]":
    """Compute all triangulations of a point configuration using TOPCOM.

    There is the option to only compute fine, regular, or star triangulations.
    All the triangulations are computed with TOPCOM (through triangulumancer)
    when the generator is first advanced, and they are then yielded one at a
    time. Regularity is checked for each triangulation as it is yielded.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by
        [`Polytope.all_triangulations`][cytools.polytope.Polytope.all_triangulations].

    Args:
        poly (Polytope): The ambient polytope.
        pts (ArrayLike): The points to be triangulated, specified by their
            labels in `poly`.
        only_fine (bool): Whether to restrict to fine triangulations.
        only_regular (bool): Whether to restrict to regular triangulations.
            Ignored if `raw_output` is true.
        only_star (bool): Whether to restrict to star triangulations.
        star_origin (int | None): The label of the point used as the star
            origin. It must be specified if `only_star` is true.
        backend (str | None): The optimizer used to check regularity. The
            available options are the backends of
            [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not specified, it
            is picked automatically.
        raw_output (bool): Whether to yield the triangulations as arrays of
            simplices (each specified by point labels) instead of as
            `Triangulation` objects. In this case regularity is not checked.

    Yields:
        Triangulation | np.ndarray: The triangulations with the specified
            properties, or their arrays of simplices if `raw_output` is true.

    Raises:
        ValueError: If `pts` is empty, or if `only_star` is true but
            `star_origin` is not specified.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a polytope and find all of its
        triangulations. We try picking different restrictions and see how the
        number of triangulations changes.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        ...               [-1,-2,-1,-1],[-2,-1,-1,-1]])
        >>> g = p.all_triangulations()
        >>> next(g)  # takes some time while TOPCOM finds all the triangulations
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> next(g)  # produces the next triangulation immediately
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> len(p.all_triangulations(as_list=True))  # number of FRSTs
        2
        >>> len(p.all_triangulations(only_regular=False, only_star=False,
        ...                          only_fine=False, as_list=True))
        6
    """
    # input checking/parsing
    if len(pts) == 0:
        raise ValueError("List of points cannot be empty.")

    if only_star and star_origin is None:
        raise ValueError(
            "The star_origin parameter must be specified when "
            "restricting to star triangulations."
        )

    # ensure points are appropriately sorted (for Triangulation inputs)
    triang_pts = poly.points(which=pts)
    if raw_output:
        backend = "topcom"

    # if not full-dimenstional, find better representation
    # (only performs affine transformation, so can treat the new points as if
    # they were the original ones)
    dim = np.linalg.matrix_rank([tuple(pt) + (1,) for pt in triang_pts]) - 1
    if dim == triang_pts.shape[1]:
        optimal_pts = triang_pts
    else:
        optimal_pts = lll_reduce([pt - triang_pts[0] for pt in triang_pts])[:, -dim:]

    pc = triangulumancer.PointConfiguration(optimal_pts)
    triangs = pc.all_triangulations(only_fine=only_fine)

    # map the triangulations to labels
    triangs = [[[pts[x] for x in i] for i in t.simplices] for t in triangs]

    # sort the triangs
    srt_triangs = [
        np.array(sorted([sorted(s) for s in t]))
        for t in triangs
        if (not only_star or all(star_origin in ss for ss in t))
    ]

    # return the output
    for t in srt_triangs:
        if raw_output:
            yield t
            continue
        tri = Triangulation(
            poly,
            pts,
            simplices=t,
            make_star=False,
            check_input_simplices=False,
        )
        if not only_regular or tri.is_regular(backend=backend):
            yield tri


def random_triangulations_fast_generator(
    poly: "Polytope",
    pts: ArrayLike,
    N: int = None,
    c: float = 0.2,
    max_retries: int = 500,
    make_star: bool = False,
    only_fine: bool = True,
    backend: str = "cgal",
    seed: int = None,
    verbosity: int = 0,
) -> "Iterator[Triangulation]":
    """Generate pseudorandom regular triangulations of a point configuration.

    The triangulations are regular, and optionally fine and star. They are
    obtained by picking random heights around the Delaunay heights $|p|^2$ from
    a Gaussian distribution. Duplicate triangulations are skipped. The
    generator stops once `N` triangulations have been produced, or once
    `max_retries` consecutive attempts fail to produce a new triangulation.

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by `Polytope.random_triangulations_fast` (see the
        [fast sampler][cytools.polytope.Polytope.random_triangulations_fast]).

    Warning:
        This function produces random triangulations very quickly, but it does
        not produce a fair sample. When a fair sampling is required, the
        [fair sampler][cytools.polytope.Polytope.random_triangulations_fair]
        (`Polytope.random_triangulations_fair`) should be used.

    Args:
        poly (Polytope): The ambient polytope.
        pts (ArrayLike): The points to be triangulated, specified by their
            labels in `poly`.
        N (int | None): Number of desired unique triangulations. If not
            specified, it generates as many triangulations as it can find until
            it has to retry `max_retries` times in a row to obtain a new
            triangulation.
        c (float): The standard deviation of the Gaussian distribution used to
            pick the heights. A larger `c` results in a wider range of possible
            triangulations, but with a larger fraction of them being non-fine,
            which slows down the process when `only_fine` is true.
        max_retries (int): Maximum number of consecutive attempts to obtain a
            new triangulation before the process is terminated.
        make_star (bool): Whether to convert the obtained triangulations into
            star triangulations.
        only_fine (bool): Whether to restrict to fine triangulations.
        backend (str): The backend used to compute the triangulations. The
            available options are `"cgal"` and `"qhull"`.
        seed (int | None): A seed for the random number generator. This can be
            used to obtain reproducible results.
        verbosity (int): The verbosity level.

    Yields:
        Triangulation: The random triangulations with the specified properties.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a polytope and find some random
        triangulations. The triangulations are obtained very quickly, but they
        are not a fair sample of the space of triangulations.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        ...               [-1,-1,-1,-1]]).dual()
        >>> g = p.random_triangulations_fast()
        >>> next(g)  # runs very quickly
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> next(g)  # keeps producing triangulations until it has trouble
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> rand_triangs = p.random_triangulations_fast(N=10, as_list=True)
    """
    # parse inputs
    # (use a local generator so that we don't perturb NumPy's global RNG)
    rng = np.random.RandomState(seed)

    triang_hashes = set()
    n_retries = 0
    while True:
        if n_retries >= max_retries:
            if verbosity > 0:
                print(
                    "random_triangulations_fast_generator: Hit max_retries... returning"
                )
            return
        if (N is not None) and (len(triang_hashes) >= N):
            if verbosity > 1:
                print(
                    "random_triangulations_fast_generator: Generated enough triangulations... returning"
                )
            return

        # generate random heights, make the triangulation
        heights = [pt.dot(pt) + rng.normal(0, c) for pt in poly.points(which=pts)]
        t = Triangulation(
            poly,
            pts,
            heights=heights,
            make_star=make_star,
            backend=backend,
            check_heights=False,
        )

        # check if it's good
        if only_fine and (not t.is_fine()):
            n_retries += 1
            continue

        # check that the heights aren't on a wall of the secondary cone
        t.check_heights(verbosity - 1)

        h = hash(t)
        if h in triang_hashes:
            n_retries += 1
            continue

        # it is good!
        triang_hashes.add(h)
        n_retries = 0
        yield t


def random_triangulations_fair_generator(
    poly: "Polytope",
    pts: ArrayLike,
    N: int = None,
    n_walk: int = 10,
    n_flip: int = 10,
    initial_walk_steps: int = 20,
    walk_step_size: float = 1e-2,
    max_steps_to_wall: int = 10,
    fine_tune_steps: int = 8,
    max_retries: int = 50,
    make_star: bool = False,
    backend: str = "cgal",
    seed: int = None,
) -> "Iterator[Triangulation]":
    r"""Generate a fair sample of random regular triangulations.

    The triangulations are fine and regular, and optionally star. This
    implements Algorithm \#3 from the paper *Bounding the Kreuzer-Skarke
    Landscape* by Mehmet Demirtas, Liam McAllister, and Andres Rios-Tascon
    ([arXiv:2008.01730](https://arxiv.org/abs/2008.01730)).

    This is a Markov chain Monte Carlo algorithm that involves taking random
    walks inside the subset of the secondary fan corresponding to fine
    triangulations and performing random flips. For details, please see
    Section 4.1 in the paper. The generator stops once `N` triangulations have
    been produced, once `max_retries` consecutive attempts fail to produce a
    new triangulation, or if no wall of the fine subset of the secondary fan
    can be found (in which case a message is printed).

    Note:
        This function is not intended to be called by the end user. Instead,
        it is used by `Polytope.random_triangulations_fair` (see the
        [fair sampler][cytools.polytope.Polytope.random_triangulations_fair]).

    Note:
        This function is designed mainly for large polytopes where sampling
        triangulations is challenging. When small polytopes are used it is
        likely to get stuck.

    Args:
        poly (Polytope): The ambient polytope.
        pts (ArrayLike): The points to be triangulated, specified by their
            labels in `poly`. They must be full-dimensional.
        N (int | None): Number of desired unique triangulations. If not
            specified, it generates as many triangulations as it can find until
            it has to retry `max_retries` times in a row to obtain a new
            triangulation.
        n_walk (int): Number of hit-and-run steps per triangulation.
        n_flip (int): Number of random flips performed per triangulation.
        initial_walk_steps (int): Number of hit-and-run steps to take before
            starting to record triangulations. Small values may result in a
            bias towards Delaunay-like triangulations.
        walk_step_size (float): Determines the size of random steps taken in
            the secondary fan. The algorithm may stall if it is too small.
        max_steps_to_wall (int): Maximum number of steps to take towards a wall
            of the subset of the secondary fan that corresponds to fine
            triangulations. If a wall is not found, a new random direction is
            selected. Setting this to be very large (>100) reduces performance.
            If this, or `walk_step_size`, is set too low, the algorithm may
            stall.
        fine_tune_steps (int): Number of steps to determine the location of a
            wall. Decreasing it improves performance, but might result in
            biased samples.
        max_retries (int): Maximum number of consecutive attempts to obtain a
            new triangulation (or to find a wall) before the process is
            terminated.
        make_star (bool): Whether to convert the obtained triangulations into
            star triangulations.
        backend (str): The backend used to compute the triangulations. The
            available options are `"cgal"` and `"qhull"`.
        seed (int | None): A seed for the random number generator. This can be
            used to obtain reproducible results.

    Yields:
        Triangulation: The random triangulations with the specified properties.

    Raises:
        Exception: If the point configuration is not full-dimensional.

    Examples:
        This function is not intended to be directly used, but it is used in
        the following example. We construct a polytope and find some random
        triangulations. The computation takes considerable time, but they
        should be a fair sample from the full set of triangulations (if the
        parameters are chosen correctly). For (some) machine learning purposes
        or when the fairness of the sample is not crucial, the
        [fast sampler][cytools.polytope.Polytope.random_triangulations_fast]
        (`Polytope.random_triangulations_fast`) should be used instead.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        ...               [-1,-1,-1,-1]]).dual()
        >>> g = p.random_triangulations_fair()
        >>> next(g)  # takes a long time (a few minutes)
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> next(g)  # also takes a few minutes
        A fine, regular, star triangulation of a 4-dimensional point...
        >>> rand_triangs = p.random_triangulations_fair(N=10, as_list=True)

        It is worth noting that the time it takes to obtain each triangulation
        varies very significantly with the parameters used. The function tries
        to guess reasonable parameters, but it is better to adjust them to your
        desired balance between speed and fairness of the sampling.
    """
    # input parsing
    triang_pts = poly.points(which=pts)
    num_points = len(pts)

    dim = np.linalg.matrix_rank([tuple(pt) + (1,) for pt in triang_pts]) - 1
    if dim != triang_pts.shape[1]:
        raise Exception("Point configuration must be full-dimensional.")

    # (use a local generator so that we don't perturb NumPy's global RNG)
    rng = np.random.RandomState(seed)

    # Obtain random Delaunay triangulation by picking random point as origin
    rand_ind = rng.randint(0, len(pts))
    points_shifted = [p - triang_pts[rand_ind] for p in triang_pts]

    delaunay_heights = [walk_step_size * (np.dot(p, p)) for p in points_shifted]
    start_pt = delaunay_heights
    old_pt = start_pt

    step_size = walk_step_size * np.mean(delaunay_heights)

    # initialize for MCMC
    step_ctr = 0  # total # of steps taken
    step_per_tri_ctr = 0  # # of steps taken for given triangulation

    n_retries = 0
    triang_hashes = set()

    # do the work
    while True:
        # check if we're done
        if n_retries >= max_retries:
            break
        if (N is not None) and (len(triang_hashes) >= N):
            break

        # find a wall
        # (move along random direction until we hit a non-fine triangulation)
        outside_bounds = False
        for _ in range(max_retries):
            in_pt = old_pt

            # find random direction
            random_dir = rng.normal(size=num_points)
            random_dir = random_dir / np.linalg.norm(random_dir)

            # take steps
            for __ in range(max_steps_to_wall):
                new_pt = in_pt + random_dir * step_size
                temp_tri = Triangulation(
                    poly,
                    pts,
                    heights=new_pt,
                    make_star=False,
                    backend=backend,
                    verbosity=0,
                )

                # check triang
                if temp_tri.is_fine():
                    in_pt = new_pt
                else:
                    out_pt = new_pt
                    outside_bounds = True
                    break

            if outside_bounds:
                break

        # break loop it it can't find any new wall after max_retries
        if not outside_bounds:
            print("Couldn't find wall.")
            break

        # Find the location of the boundary
        fine_tune_ctr = 0
        in_pt_found = False
        while (fine_tune_ctr < fine_tune_steps) or (not in_pt_found):
            new_pt = (in_pt + out_pt) / 2
            temp_tri = Triangulation(
                poly,
                pts,
                heights=new_pt,
                make_star=False,
                backend=backend,
                verbosity=0,
            )

            # check triang
            if temp_tri.is_fine():
                in_pt = new_pt
                in_pt_found = True
            else:
                out_pt = new_pt

            fine_tune_ctr += 1

        # Take a random walk step
        in_pt = in_pt / np.linalg.norm(in_pt)
        random_coef = rng.uniform(0, 1)
        new_pt = random_coef * np.array(old_pt) + (1 - random_coef) * np.array(in_pt)

        # after enough steps are taken, move on to random flips
        if (step_ctr > initial_walk_steps) and (step_per_tri_ctr >= n_walk):
            flip_seed_tri = Triangulation(
                poly,
                pts,
                heights=new_pt,
                make_star=make_star,
                backend=backend,
                verbosity=0,
            )

            # take flips
            if n_flip > 0:
                temp_tri = flip_seed_tri.random_flips(
                    n_flip,
                    only_fine=True,
                    only_regular=True,
                    only_star=True,
                    # derive the sub-seed from our local generator so that the
                    # walk stays reproducible for a given `seed`
                    seed=int(rng.randint(2**32)),
                )
            else:
                temp_tri = flip_seed_tri

            # record triangulation
            h = hash(temp_tri)
            if h in triang_hashes:
                n_retries += 1
                continue
            triang_hashes.add(h)

            n_retries = 0
            step_per_tri_ctr = 0

            yield temp_tri

        # update old point
        old_pt = new_pt / np.linalg.norm(new_pt)

        # update counters
        step_ctr += 1
        step_per_tri_ctr += 1
