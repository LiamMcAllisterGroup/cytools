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

# 'standard' imports
import ast
from collections.abc import Iterable
import copy
import itertools
import math
import re
import subprocess
import warnings

# 3rd party imports
import flint
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
import triangulumancer

# CYTools imports
from cytools import config
from cytools.cone import Cone
from cytools.toricvariety import ToricVariety
from cytools.utils import gcd_list, lll_reduce


class Triangulation:
    """
    This class handles triangulations of lattice polytopes. It can compute
    various properties of the triangulation, as well as construct a
    ToricVariety or CalabiYau object if the triangulation is suitable.

    :::important
    Generally, objects of this class should not be constructed directly by the
    end user. Instead, they should be created by various functions of the
    [`Polytope`](./polytope) class.
    :::

    ## Constructor

    ### `cytools.triangulation.Triangulation`

    **Description:**
    Constructs a `Triangulation` object describing a triangulation of a lattice
    polytope. This is handled by the hidden [`__init__`](#__init__) function.

    :::note
    If you construct a triangulation object directly by inputting a list of
    points they may be reordered to match the ordering of the points from the
    [`Polytope`](./polytope) class. This is to ensure that computations of
    toric varieties and Calabi-Yau manifolds are correct. To avoid this
    subtlety we discourage users from constructing Triangulation objects
    directly, and instead use the triangulation functions in the
    [`Polytope`](./polytope) class.
    :::

    **Arguments:**
    - `poly`: The ambient polytope of the points to be triangulated. If not
        specified, it's constructed as the convex hull of the given points.
    - `pts`: The list of points to be triangulated. Specified by labels.
    - `heights`: The heights specifying the regular triangulation. When not
        specified, construct based off of the backend:
            - (CGAL) Delaunay triangulation,
            - (QHULL) triangulation from random heights near Delaunay, or
            - (TOPCOM) placing triangulation.
        Heights can only be specified when using CGAL or QHull as the backend.
    - `make_star`: Whether to turn the triangulation into a star triangulation
        by deleting internal lines and connecting all points to the origin, or
        equivalently, by decreasing the height of the origin until it is much
        lower than all other heights.
    - `simplices`: Array-like of simplices specifying the triangulation. Each
        simplex is a list of point labels. This is useful when a triangulation
        was previously computed and it needs to be used again. Note that the
        ordering of the points needs to be consistent.
    - `check_input_simplices`: Whether to check if the input simplices define a
        valid triangulation.
    - `backend`: The backend used to compute the triangulation. Options are
        "qhull", "cgal", and "topcom". CGAL is the default as it is very
        fast and robust.
    - `verbosity`: The verbosity level.

    **Example:**
    We construct a triangulation of a polytope. Since this class is not
    intended to by initialized by the end user, we create it via the
    [`triangulate`](./polytope#triangulate) function of the
    [`Polytope`](./polytope) class. In this example the polytope is reflexive,
    so by default the triangulation is fine, regular, and star. Also, since the
    polytope is reflexive then by default only the lattice points not interior
    to facets are included in the triangulation.
    ```python {3}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    t = p.triangulate()
    print(t)
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 7 points in ZZ^4
    ```
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
    ) -> None:
        """
        **Description:**
        Initializes a `Triangulation` object.

        **Arguments:**
        - `poly`: The ambient polytope of the points to be triangulated.
        - `pts`: The list of points to be triangulated. Specified by labels.
        - `make_star`: Whether to turn the triangulation into a star
            triangulation by deleting internal lines and connecting all points
            to the origin, or equivalently, by decreasing the height of the
            origin until it is much lower than all other heights.
        - `simplices`: Array-like of simplices specifying the triangulation.
            Each simplex is a list of point labels. This is useful when a
            triangulation was previously computed and it needs to be used
            again. Note that the ordering of the points needs to be consistent.
        - `check_input_simplices`: Whether to check if the input simplices
            define a valid triangulation.
        - `heights`: The heights specifying the regular triangulation. When not
            specified, construct based off of the backend:
                - (CGAL) a Delaunay triangulation,
                - (QHULL) triangulation from random heights near Delaunay, or
                - (TOPCOM) placing triangulation.
            Heights can only be specified when using CGAL or QHull as the
            backend.
        - `check_heights`: Whether to check if the input/default heights define
            a valid/unique triangulation.
        - `backend`: The backend used to compute the triangulation. Options are
            "qhull", "cgal", and "topcom". CGAL is the default as it is very
            fast and robust.
        - `verbosity`: The verbosity level.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new
        `Triangulation` object. In this example the polytope is reflexive, so
        by default the triangulation is fine, regular, and star. Also, since
        the polytope is reflexive then by default only the lattice points not
        interior to facets are included in the triangulation.
        ```python {3}
        from cytools import Polytope
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        print(t)
        # A fine, regular, star triangulation of a 4-dimensional point
        # configuration with 7 points in ZZ^4
        ```
        """
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
                if any([(l not in self._labels) for l in simp_labels]):
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
                        "Simplices must be specified when working" "without a backend"
                    )

                # Heights need to be perturbed around the Delaunay heights for
                # QHull or the triangulation might not be regular. If using
                # CGAL then they are not perturbed.
                if backend == "qhull":
                    heights = [
                        np.dot(p, p) + np.random.normal(0, 0.05) for p in self.points()
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
                    _to_star(self)
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

                    origin_step = max(
                        10, (max(heights_masked[1:]) - min(heights_masked[1:]))
                    )

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
            if not np.all(self._heights == 0):
                self._heights = self._heights / gcd_list(self._heights)
            self._heights -= min(self._heights)

            max_h = max(abs(self._heights))
            if max_h < 2**8:
                dtype = np.uint8
            elif max_h < 2**16:
                dtype = np.uint16
            elif max_h < 2**32:
                dtype = np.uint32
            else:
                dtype = np.uint64
            self._heights = self._heights.round().astype(dtype)

    # defaults
    # ========
    def __repr__(self) -> str:
        """
        **Description:**
        Returns a string describing the triangulation.

        **Arguments:**
        None.

        **Returns:**
        A string describing the triangulation.

        **Example:**
        This function can be used to convert the triangulation to a string or
        to print information about the triangulation.
        ```python {3,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        poly_info = str(t) # Converts to string
        print(t) # Prints triangulation info
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 7 points in ZZ^4
        ```
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
            f"A "
            + fine_str
            + regular_str
            + star_str
            + f" triangulation of a {self.dim()}-dimensional "
            + f"point configuration with {len(self.labels)} points "
            + f"in ZZ^{self.ambient_dim()}"
        )

    def __eq__(self, other: "Triangulation") -> bool:
        """
        **Description:**
        Implements comparison of triangulations with ==.

        **Arguments:**
        - `other`: The other triangulation that is being compared.

        **Returns:**
        The truth value of the triangulations being equal.

        **Example:**
        We construct two triangulations and compare them.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        t2 = p.triangulate(backend="topcom")
        t1 == t2
        # True
        ```
        """
        # check if other is even a triangulation
        if not isinstance(other, Triangulation):
            return False

        # check that we have the same polytope and simplices
        our_simps = sorted(self.simplices().tolist())
        other_simps = sorted(other.simplices().tolist())

        return self.polytope() == other.polytope() and our_simps == other_simps

    def __ne__(self, other: "Triangulation") -> bool:
        """
        **Description:**
        Implements comparison of triangulations with !=.

        **Arguments:**
        - `other`: The other triangulation that is being compared.

        **Returns:**
        The truth value of the triangulations being different.

        **Example:**
        We construct two triangulations and compare them.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t1 = p.triangulate(backend="qhull")
        t2 = p.triangulate(backend="topcom")
        t1 != t2
        # False
        ```
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        **Description:**
        Implements the ability to obtain hash values from triangulations.

        **Arguments:**
        None.

        **Returns:**
        The hash value of the triangulation.

        **Example:**
        We compute the hash value of a triangulation. Also, we construct a set
        and a dictionary with a triangulation, which make use of the hash
        function.
        ```python {3,4,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        h = hash(t) # Obtain hash value
        d = {t: 1} # Create dictionary with triangulation keys
        s = {t} # Create a set of triangulations
        ```
        """
        if self._hash is None:
            pt_hash = hash(tuple(sorted(tuple(v) for v in self.points())))
            simps = tuple(sorted(tuple(sorted(s)) for s in self.simplices()))
            self._hash = hash((pt_hash,) + simps)

        return self._hash

    # caching
    # =======
    def _dump(self) -> dict:
        """
        **Description:**
        Get every class variable.

        No copying is done, so be careful with mutability!

        **Arguments:**
        None.

        **Returns:**
        Dictionary mapping variable name to value.
        """
        return self.__dict__

    def clear_cache(self, recursive: bool = False) -> None:
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        - `recursive`: Whether to also clear the cache of the ambient polytope.

        **Returns:**
        Nothing.

        **Example:**
        We construct a triangulation, compute its GKZ vector, clear the cache
        and then compute it again.
        ```python {4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        gkz_phi = t.gkz_phi() # Computes the GKZ vector
        t.clear_cache()
        gkz_phi = t.gkz_phi() # The GKZ vector is recomputed
        ```
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
        """
        **Description:**
        Returns the polytope being triangulated.

        **Arguments:**
        None.

        **Returns:**
        The ambient polytope.
        """
        return self._poly

    polytope = lambda self: self.poly

    @property
    def labels(self):
        """
        **Description:**
        Returns the labels of lattice points in the triangulation.

        **Arguments:**
        None.

        **Returns:**
        The labels of lattice points in the triangulation.
        """
        return self._labels

    def dimension(self) -> int:
        """
        **Description:**
        Returns the dimension of the triangulated point configuration.

        **Arguments:**
        None.

        **Returns:**
        The dimension of the triangulation.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a triangulation and find its dimension.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.dimension()
        # 4
        ```
        """
        return self._dim

    # aliases
    dim = dimension

    def ambient_dimension(self) -> int:
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        The dimension of the ambient lattice.

        **Aliases:**
        `ambient_dim`.

        **Example:**
        We construct a triangulation and find its ambient dimension.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.ambient_dim()
        # 4
        ```
        """
        return self.poly.ambient_dim()

    # aliases
    ambient_dim = ambient_dimension

    def is_fine(self) -> bool:
        """
        **Description:**
        Returns True if the triangulation is fine (all the points are used), and
        False otherwise. Note that this only checks if it is fine with respect
        to the point configuration, not with respect to the full set of lattice
        points of the polytope.

        **Arguments:**
        None.

        **Returns:**
        The truth value of the triangulation being fine.

        **Example:**
        We construct a triangulation and check if it is fine.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.is_fine()
        # True
        ```
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
        """
        **Description:**
        Returns True if the triangulation is star and False otherwise. The star
        origin is assumed to be the origin, so for polytopes that don't contain
        the origin this function will always return False unless `star_origin`
        is specified.

        **Arguments:**
        - `star_origin`: The index of the origin of the star triangulation

        **Returns:**
        The truth value of the triangulation being star.

        **Example:**
        We construct a triangulation and check if it is star.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.is_star()
        # True
        ```
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
        """
        **Description:**
        Returns a ToricVariety object corresponding to the fan defined by the
        triangulation.

        **Arguments:**
        None.

        **Returns:**
        The toric variety arising from the triangulation.

        **Example:**
        We construct a triangulation and obtain the resulting toric variety.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.get_toric_variety()
        # A simplicial compact 4-dimensional toric variety with 9 affine patches
        ```
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
        - `nef_partition`: A list of tuples of indices specifying a
            nef-partition of the polytope, which correspondingly defines a
            complete intersection Calabi-Yau.

        **Returns:**
        The Calabi-Yau arising from the triangulation.

        **Example:**
        We construct a triangulation and obtain the Calabi-Yau hypersurface in
        the resulting toric variety.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.get_cy()
        # A Calabi-Yau 3-fold hypersurface with h11=2 and h21=272 in a 4-dimensional toric variety
        ```
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
        """
        **Description:**
        Returns the points of the triangulation. Note that these are not
        necessarily equal to the lattice points of the polytope they define.

        **Arguments:**
        - `which`: Which points to return. Specified by a (list of) labels.
            NOT INDICES!!!
        - `optimal`: Whether to return the points in their optimal coordinates.
        - `as_indices`: Return the points as indices of the full list of points
            of the polytope.

        **Returns:**
        The points of the triangulation.

        **Aliases:**
        `pts`.

        **Example:**
        We construct a triangulation and print the points in the point
        configuration. Note that since the polytope is reflexive, then by
        default only the lattice points not interior to facets were used.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.points()
        # array([[ 0,  0,  0,  0],
        #        [-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3]])
        ```
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
                except Exception as e:
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
                    return np.array([self._labels2optPts[l] for l in which])

            # normal case
            return self.poly.points(
                which=which, optimal=optimal, as_indices=as_poly_indices
            )

    # aliases
    pts = points

    def points_to_labels(
        self, points: ArrayLike, is_optimal: bool = False
    ) -> "list | None":
        """
        **Description:**
        Returns the list of labels corresponding to the given points. It also
        accepts a single point, in which case it returns the corresponding
        label.

        **Arguments:**
        - `points`: A point or a list of points.
        - `is_optimal`: Whether the points argument represents points in the
            optimal (True) or input (False) basis

        **Returns:**
        The list of labels corresponding to the given points, or the label of
        the point if only one is given.
        """
        return self.poly.points_to_labels(points, is_optimal=is_optimal)

    def points_to_indices(
        self,
        points: ArrayLike,
        is_optimal: bool = False,
        as_poly_indices: bool = False,
    ) -> "np.ndarray | int":
        """
        **Description:**
        Returns the list of indices corresponding to the given points. It also
        accepts a single point, in which case it returns the corresponding
        index.

        **Arguments:**
        - `points`: A point or a list of points.
        - `is_optimal`: Whether the points argument represents points in the
            optimal (True) or input (False) basis

        **Returns:**
        The list of indices corresponding to the given points, or the index of
        the point if only one is given.

        **Example:**
        We construct a polytope and find the indices of some of its points.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points_to_indices([-1,-1,-6,-9]) # We input a single point, so a single index is returned
        # 1
        p.points_to_indices([[-1,-1,-6,-9],[0,0,0,0],[0,0,1,0]]) # We input a list of points, so a list of indices is returned
        # array([1, 0, 3])
        ```
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
        """
        **Description:**
        Takes a list of indices of points of the triangulation and it returns
        the corresponding indices of the polytope. It also accepts a single
        entry, in which case it returns the corresponding index.

        **Arguments:**
        - `points`: A list of indices of points.

        **Returns:**
        The list of indices corresponding to the given points. Or the index of
        the point if only one is given.
        """
        # (NM: remove this... use labels instead...)
        return self.points(which=[self._labels[i] for i in inds], as_poly_indices=True)

    points_to_poly_indices = triangulation_to_polytope_indices

    # triangulation
    # =============
    # sanity checks
    # -------------
    def is_valid(self, backend: str = None, verbosity: int = 0) -> bool:
        """
        **Description:**
        Returns True if the presumed triangulation meets all requirements to be
        a triangulation. The simplices must cover the full volume of the convex
        hull, and they cannot intersect at full-dimensional regions.

        **Arguments:**
        - `backend`: The optimizer used for the computation. The available
            options are the backends of the [`is_solid`](./cone#is_solid)
            function of the [`Cone`](./cone) class. If not specified, it will
            be picked automatically.
        - `verbosity`: The verbosity level.

        **Returns:**
        The truth value of the triangulation being valid.

        **Example:**
        This function is useful when constructing a triangulation from a given
        set of simplices. We show this in this example.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate(simplices=[[0,1,2,3,4],[0,1,2,3,5],[0,1,2,4,5],[0,1,3,4,5],[0,2,3,4,5]]) # It is already used here unless using check_input_simplices=False
        t.is_valid()
        # True
        ```
        """
        # check if we know the answer
        if self._is_valid is not None:
            return self._is_valid

        # calculate the answer
        simps = self.simplices()
        # simps = np.array([self.points(s, as_triang_indices=True) for s in simps])

        if simps.shape[1] != self.dim()+1:
            self._is_valid = False
            return self._is_valid

        #if simps.shape[0] == 1:
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
            l: list(pts[l])
            + [
                1,
            ]
            for l in self.labels
        }
        pts_all = np.array(list(pts.values()))
        # pts_ext = np.array([list(pt)+[1,] for pt in pts])

        # We first check if the volumes add up to the volume of the polytope
        v = 0
        for s in simps:
            tmp_v = abs(int(round(np.linalg.det([pts_ext[l] for l in s]))))

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
        """
        **Description:**
        Check if the heights uniquely define a triangulation. That is, if they
        do not lie on a wall of the generated secondary cone.

        If heights don't uniquely correspond to a triangulation, delete the
        heights so they are re-calculated (keep triangulation, though...)

        **Arguments:**
        - `verbosity`: The verbosity level.

        **Returns:**
        Nothing.
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
        """
        **Description:**
        Returns the simplices of the triangulation. It also has the option of
        restricting the simplices to faces of the polytope of a particular
        dimension or codimension. This restriction is useful for checking CY
        equivalences from different triangulations.

        **Arguments:**
        - `on_faces_dim: Restrict the simplices to faces of the polytope of a
            given dimension.
        - `on_faces_codim`: Restrict the simplices to faces of the polytope of
            a given codimension.
        - `split_by_face`: Return the simplices for each face. Don't merge the
            collection.
        - `as_np_array`: Return the simplices as a numpy array. Otherwise,
            they are returned as a set of frozensets.
        - `as_indices`: Whether to map the simplices from labels to point
            indices (in the triangulations).

        **Returns:**
        The simplices of the triangulation.

        **Example:**
        We construct a triangulation and find its simplices. We also find the
        simplices the lie on 2-faces of the polytope.
        ```python {3,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.simplices()
        # array([[0, 1, 2, 3, 4],
        #        [0, 1, 2, 3, 5],
        #        [0, 1, 2, 4, 5],
        #        [0, 1, 3, 4, 5],
        #        [0, 2, 3, 4, 5]])
        t.simplices(on_faces_dim=2)
        # [[1 2 3]
        #  [1 2 4]
        #  [1 2 5]
        #  [1 3 4]
        #  [1 3 5]
        #  [1 4 5]
        #  [2 3 4]
        #  [2 3 5]
        #  [2 4 5]
        #  [3 4 5]]
        ```
        """
        # input parsing
        if (on_faces_dim is None) and (on_faces_codim is None):
            out = self._simplices

            # cast to indices
            if as_indices:
                l2i = {l: i for i, l in enumerate(self.labels)}
                out = [[l2i[l] for l in s] for s in out]

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
            face_labels = [frozenset(face.labels)
                           for face in self.polytope().faces(faces_dim)]

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
            l2i = {l: i for i, l in enumerate(self.labels)}
            out = [[frozenset([l2i[l] for l in s]) for s in face] for face in out]

        if split_by_face:
            if as_np_array:
                out = [np.array(sorted(sorted(s) for s in face)) for face in out]
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
        """
        **Description:**
        Restrict the triangulation to some face(s).

        **Arguments:**
        - `restrict_to`: The face(s) to restrict to.  If none provided, gives
            restriction to each dim-face (see next argument).
        - `restrict_dim`: If restrict_to is None, sets the dimension of the
            faces to restrict to. *** Only used if restrict_to == None. ***
        - `as_poly`: Construct the formal Triangulation objects.
        - `verbosity`: The verbosity level.

        **Returns:**
        The restrictions
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
        """
        **Description:**
        Computes the (hyperplanes defining the) secondary cone of the
        triangulation. This cone is also called the 'cone of strictly convex
        piecewise linear functions'.

        The triangulation is regular if and only if this cone is solid (i.e.
        full-dimensional), in which case the points in the strict interior
        correspond to heights that give rise to the triangulation.


        Also, allow calculation of the 'secondary cone of the N-skeleton'. This
        cone is defined as
            1) the intersection of the secondary cones of all N-dim faces of
               this triangulation or, equivalently,
            2) the union of all secondary cones arising from triangulations
               with the same N-face restrictions.
        Any point in the strict interior of this cone correspond to heights
        which generate a triangulation with the imposed N-face restrictions.
        Set N via argument `on_faces_dim`.

        Some cases of interest:
            1) N=self.dim() -> return the secondary cone of the triangulation
            2) N=2          -> return the union of all secondary cones arising
                               from triangulations with the same 2-face
                               restrictions. Akin to Kcup.

        **Arguments:**
        - `backend`: The backend to use. Options are "native", which uses a
            native implementation of an algorithm by Berglund, Katz and Klemm,
            or "topcom" which uses differences of GKZ vectors.computation.
        - `include_points_not_in_triangulation`: This flag allows the exclusion
            of points that are not part of the triangulation. This can be done
            to check regularity faster, but this cannot be used if the actual
            cone in the secondary fan is needed.
        - `as_cone`: Return a cone or just the defining hyperplanes.
        - `on_faces_dim`: Compute the secondary cone for each face with this
            dimension and then take their intersection. This has the
            interpretation of enforcing the triangulations on each of these
            faces, but being agnostic to the rest of the structure.
        - `use_cache`: Whether to use cached values of the secondary cone.

        **Returns:**
        The secondary cone.

        **Aliases:**
        `cpl_cone`.

        **Example:**
        We construct a triangulation and find its secondary cone.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.secondary_cone()
        # A rational polyhedral cone in RR^7 defined by 3 hyperplanes normals
        ```
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

            # calculate the hyperplanes
            null_vecs = set()
            for i, s1 in enumerate(simps):
                for s2 in simps[i + 1 :]:
                    # ensure that the simps have a large enough intersection
                    comm_pts = s1 & s2
                    if len(comm_pts) != dim:
                        continue

                    # organize the diff
                    diff_pts = list(s1 ^ s2)
                    comm_pts = list(comm_pts)

                    m[:-1, :2] = self.points(which=diff_pts, optimal=True).T
                    m[:-1, 2 : (2 + dim)] = self.points(which=comm_pts, optimal=True).T

                    # calculate nullspace/hyperplane ineq
                    v = flint.fmpz_mat(m.tolist()).nullspace()[0]
                    v = np.array(v.transpose().tolist()[0], dtype=int)

                    # ensure the sign is correct
                    if v[0] < 0:
                        v *= -1

                    # Reduce the vector
                    g = gcd_list(v)
                    if g != 1:
                        v //= g

                    # Construct the full vector (including all points)
                    # (could get some more performance by allowing sparse vectors as Cone argument...)
                    for i, pt in enumerate(diff_pts):
                        full_v[labels2inds[pt]] = v[i]
                    for i, pt in enumerate(comm_pts):
                        full_v[labels2inds[pt]] = v[i + 2]

                    if self.is_star():
                        full_v[labels2inds[self.poly._label_origin]] = v[-1]

                    null_vecs.add(tuple(full_v))

                    # clear full_v
                    for i, pt in enumerate(diff_pts):
                        full_v[labels2inds[pt]] = 0
                    for i, pt in enumerate(comm_pts):
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
        """
        **Description:**
        Returns True if the triangulation is regular and False otherwise.

        **Arguments:**
        - `backend`: The optimizer used for the computation. The available
            options are the backends of the [`is_solid`](./cone#is_solid)
            function of the [`Cone`](./cone) class. If not specified, it will
            be picked automatically.

        **Returns:**
        The truth value of the triangulation being regular.

        **Example:**
        We construct a triangulation and check if it is regular.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.is_regular()
        # True
        ```
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
        """
        **Description:**
        Returns the a height vector if the triangulation is regular. An
        exception is raised if a height vector could not be found either
        because the optimizer failed or because the triangulation is not
        regular.

        **Arguments:**
        - `integral`: Whether to find an integral height vector.
        - `backend`: The optimizer used for the computation. The available
            options are the backends of the
            [`find_interior_point`](./cone#find_interior_point) function of the
            [`Cone`](./cone) class. If not specified, it will be picked
            automatically.

        **Returns:**
        A height vector giving rise to the triangulation.

        **Example:**
        We construct a triangulation and find a height vector that generates it.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        t = p.triangulate()
        t.heights()
        # array([0., 0., 0., 0., 0., 1.])
        ```
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
        """
        **Description:**
        Returns all of the triangulations of the polytope that can be obtained
        by applying one or more polytope automorphisms to the triangulation.

        It also has the option of restricting the simplices to faces of the
        polytope of a particular dimension or codimension. This restriction is
        useful for checking CY equivalences from different triangulations.

        :::note
        Depending on how the point configuration was constructed, it may be the
        case that the automorphism group of the point configuration is larger
        or smaller than the one from the polytope. This function only uses the
        subset of automorphisms of the polytope that are also automorphisms of
        the point configuration.
        :::

        **Arguments:**
        - `automorphism`: The index or list of indices of the polytope
            automorphisms to use. If not specified it uses all automorphisms.
        - `on_faces_dim`: Restrict the simplices to faces of the polytope of a
            given dimension.
        - `on_faces_codim`: Restrict the simplices to faces of the polytope of
            a given codimension.

        **Returns:**
        The list of triangulations obtained by performing automorphisms
        transformations.

        **Example:**
        We construct a triangulation and find some of its automorphism orbits.
        ```python {3,9}
        p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
        t = p.triangulate()
        orbit_all_autos = t.automorphism_orbit()
        print(len(orbit_all_autos))
        # 36
        orbit_all_autos_2faces = t.automorphism_orbit(on_faces_dim=2)
        print(len(orbit_all_autos_2faces))
        # 36
        orbit_sixth_auto = t.automorphism_orbit(automorphism=5)
        print(len(orbit_sixth_auto))
        # 2
        orbit_list_autos = t.automorphism_orbit(automorphism=[5,6,9])
        print(len(orbit_list_autos))
        # 12
        ```
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
        apply_auto = lambda auto: tuple(
            sorted(tuple(sorted([auto[i] for i in s])) for s in simps)
        )

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
        """
        **Description:**
        Compares two triangulations with or without allowing automorphism
        transformations and with the option of restricting to faces of a
        particular dimension.

        **Arguments:**
        - `other`: The other triangulation that is being compared.
        - `use_automorphisms`: Whether to check the equivalence using
            automorphism transformations. This flag is ignored for point
            configurations that are not full dimensional.
        - `on_faces_dim`: Restrict the simplices to faces of the polytope of a
            given dimension.
        - `on_faces_codim`: Restrict the simplices to faces of the polytope of
            a given codimension.

        **Returns:**
        The truth value of the triangulations being equivalent under the
        specified parameters.

        **Example:**
        We construct two triangulations and check whether they are equivalent
        under various conditions.
        ```python {5,7,9}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,1,1,0],[0,-1,-1,0],[0,0,0,1],[1,-2,1,1],[-2,2,1,-1],[1,1,-1,-1]])
        triangs_gen = p.all_triangulations()
        t1 = next(triangs_gen)
        t2 = next(triangs_gen)
        t1.is_equivalent(t2)
        # False
        t1.is_equivalent(t2, on_faces_dim=2)
        # True
        t1.is_equivalent(t2, on_faces_dim=2, use_automorphisms=False)
        # True
        ```
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
        backend: str = None,
        verbose: bool = False,
    ) -> list["Triangulation"]:
        """
        **Description:**
        Returns the list of triangulations that differ by one bistellar flip
        from the current triangulation. The computation is performed with a
        modified version of TOPCOM. There is the option of limiting the flips
        to fine, regular, and star triangulations. An additional backend is
        used to check regularity, as checking this with TOPCOM is very slow for
        large polytopes.

        **Arguments:**
        - `only_fine`: Restricts to fine triangulations.
        - `only_regular`: Restricts the to regular triangulations.
        - `only_star`: Restricts to star triangulations.
        - `backend`: The backend used to check regularity. The options are any
            backend available for the [`is_solid`](./cone#is_solid) function of
            the [`Cone`](./cone) class. If not specified, it will be picked
            automatically.
        - `verbose`: Whether to print extra info from the TOPCOM command.

        **Returns:**
        The list of triangulations that differ by one bistellar flip from the
        current triangulation.

        **Example:**
        We construct a triangulation and find its neighbor triangulations.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        t = p.triangulate()
        triangs = t.neighbor_triangulations()
        len(triangs) # Print how many triangulations it found
        # 263
        ```
        """
        # check for topcom bug
        if len(self.simplices()) == 1:
            warnings.warn(
                "Triangulation.neighbor_triangulations called "
                + "for trivial triangulation (1 simplex)... "
                + "Returning []! Fix TOPCOM!"
            )
            return []

        # optimized method for 2D fine neighbors
        if self.is_fine() and (self.dim() == 2) and only_fine:
            return self._fine_neighbors_2d()

        pc = triangulumancer.PointConfiguration(self.points(optimal=True))
        t = triangulumancer.Triangulation(pc, self._simplices) # TODO: Need to implement this
        triangs_list = t.neighbors()

        # parse the outputs
        triangs = []
        for t in triangs_list:
            # TODO: Is this needed
            if not all(len(s) == self.dim() + 1 for s in t.simplices()):
                continue

            # construct and check triangulation
            tri = Triangulation(
                self.poly,
                self.labels,
                simplices=[[self.labels[i] for i in s] for s in t.simplices()],
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
        """
        **Description:**
        Returns a triangulation obtained by performing N random bistellar
        flips. The computation is performed with a modified version of TOPCOM.
        There is the option of limiting the flips to fine, regular, and star
        triangulations. An additional backend is used to check regularity, as
        checking this with TOPCOM is very slow for large polytopes.

        **Arguments:**
        - `N`: The number of bistellar flips to perform.
        - `only_fine`: Restricts to flips to fine triangulations. If not
            specified, it is set to True if the triangulation is fine, and
            False otherwise.
        - `only_regular`: Restricts the flips to regular triangulations. If not
            specified, it is set to True if the triangulation is regular, and
            False otherwise.
        - `only_star`: Restricts the flips to star triangulations. If not
            specified, it is set to True if the triangulation is star, and
            False otherwise.
        - `backend`: The backend used to check regularity. The options are any
            backend available for the [`is_solid`](./cone#is_solid) function of
            the [`Cone`](./cone) class. If not specified, it will be picked
            automatically.
        - `seed`: A seed for the random number generator. This can be used to
            obtain reproducible results.

        **Returns:**
        A new triangulation obtained by performing N random flips.

        **Example:**
        We construct a triangulation and perform 5 random flips to find a new
        triangulation.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        t = p.triangulate()
        t.random_flips(5) # Takes a few seconds
        # A fine, star triangulation of a 4-dimensional point configuration
        # with 106 points in ZZ^4
        ```
        """
        # parse inputs
        if seed is not None:
            np.random.seed(seed)

        # take random flips
        curr_triang = self
        for n in range(N):
            neighbors = curr_triang.neighbor_triangulations(
                only_fine=False, only_regular=False, only_star=False
            )
            np.random.shuffle(neighbors)

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
        only_star: bool = False,
        backend: str = None,
    ) -> list["Triangulation"]:
        """
        **Description:**
        An optimized variant of neighbor_triangulations for triangulations that
        are:
            1) 2D (in dimension... ambient dimension doesn't matter)
            2) fine
            3) fine neighbors are desired
        In this case, _fine_neighbors_2d runs much quicker than the
        corresponding TOPCOM calculation

        **Arguments:**
        - `only_regular`: Restricts the to regular triangulations.
        - `only_star`: Restricts to star triangulations.
        - `backend`: The backend used to check regularity. The options are any
            backend available for the [`is_solid`](./cone#is_solid) function of
            the [`Cone`](./cone) class. If not specified, it will be picked
            automatically.

        **Returns:**
        The list of triangulations that differ by one diagonal flip from the
        current triangulation.
        """
        simps_set = [set(s) for s in self._simplices]
        triangs = []

        # for each pair of simplices
        for i, s1 in enumerate(simps_set):
            for _j, s2 in enumerate(simps_set[i + 1 :]):
                j = i + 1 + _j

                # check if they form a quadrilateral
                # (i.e., if they intersect along an edge)
                inter = s1 & s2
                if len(inter) != 2:
                    continue

                # (and if the edge is 'internal')
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

                # construct the triangulation
                tri = Triangulation(
                    self.poly,
                    self.labels,
                    simplices=new_simps,
                    check_input_simplices=False,
                )

                # check the triangulation
                if only_star and (not tri.is_star()):
                    continue
                if only_regular and (not tri.is_regular(backend=backend)):
                    continue

                # keep it :)
                triangs.append(tri)
        return triangs

    # misc
    # ====
    def gkz_phi(self) -> np.ndarray:
        r"""
        **Description:**
        Returns the GKZ phi vector of the triangulation. The $i$-th component is
        defined to be the sum of the volumes of the simplices that have the
        $i$-th point as a vertex.
        $\varphi^i=\sum_{\sigma\in\mathcal{T}|p_i\in\text{vert}(\sigma)}\text{vol}(\sigma)$

        **Arguments:**
        None.

        **Returns:**
        The GKZ phi vector of the triangulation.

        **Example:**
        We construct a triangulation and find its GKZ vector.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.gkz_phi()
        # array([18, 12,  9, 12, 12, 12, 15])
        ```
        """
        # check if we know the answer
        if self._gkz_phi is not None:
            return np.array(self._gkz_phi)

        # calculate the answer
        pts_ext = {
            l: list(pt)
            + [
                1,
            ]
            for l, pt in zip(self.labels, self.points(optimal=True))
        }
        l2i = {l: i for i, l in enumerate(self.labels)}
        phi = np.zeros(len(pts_ext), dtype=int)

        for s in self._simplices:
            simp_vol = int(round(abs(np.linalg.det([pts_ext[l] for l in s]))))
            for l in s:
                phi[l2i[l]] += simp_vol

        # return
        self._gkz_phi = phi
        return np.array(self._gkz_phi)

    def sr_ideal(self) -> tuple:
        """
        **Description:**
        Returns the Stanley-Reisner ideal if the triangulation is star.

        That is, find all sets of points which are not subsets of any simplex.
        The SR-ideal is generated by such subsets.

        N.B.: This function returns the *generators* of the ideal. I.e., we
        don't return multiples of the generators.

        E.g., take the simplicial complex
            [[1,2,3],[1,2,4],[2,3,4],[1,2,5]]
        It has the following sets of points not appearing together
            (3,5), (4,5), (1,3,4),(1,3,5),(1,4,5), (2,3,5), (2,4,5), (3,4,5)
        So, the SR ideal is generated by
            x3x5, x4x5, x1x3x4
        Don't include things like x2x3x5 since it is a multiple of x3x5.

        **Arguments:**
        None.

        **Returns:**
        The Stanley-Reisner ideal of the triangulation.

        **Example:**
        We construct a triangulation and find its SR ideal.
        ```python {3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t = p.triangulate()
        t.sr_ideal()
        # array([[1, 4, 5],
        #        [2, 3, 6]])
        ```
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
                    in_SR = False
                    for order in range(1, i + 1):
                        for t in itertools.combinations(tup, order):
                            if frozenset(t + (j,)) in SR_ideal:
                                in_SR = True
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


def _to_star(triang: Triangulation) -> np.ndarray:
    """
    **Description:**
    Turns a triangulation into a star triangulation by deleting internal lines
    and connecting all points to the origin.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`Triangulation`](./triangulation) class when needed.
    :::

    :::important
    This function is only reliable for triangulations of reflexive polytopes
    and may produce invalid triangulations for other polytopes.
    :::

    **Arguments:**
    - `triang`: The triangulation to convert to star.

    **Returns:**
    Nothing.
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
    """
    **Description:**
    Computes a regular triangulation using QHull.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`Triangulation`](./triangulation) class when using QHull as
    the backend.
    :::

    **Arguments:**
    - `points`: A list of points.
    - `heights`: A list of heights defining the regular triangulation.

    **Returns:**
    A list of simplices defining a regular triangulation.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a triangulation using QHull.
    ```python {2}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    p.triangulate(backend="qhull")
    # A fine, star triangulation of a 4-dimensional point configuration with 7
    # points in ZZ^4
    ```
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
    """
    **Description:**
    Computes a regular triangulation using CGAL.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`Triangulation`](./triangulation) class when using CGAL as the
    backend.
    :::

    **Arguments:**
    - `points`: A list of points.
    - `heights`: A list of heights defining the regular triangulation.

    **Returns:**
    A list of simplices defining a regular triangulation.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a triangulation using CGAL.
    ```python {2}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    p.triangulate(backend="cgal")
    # A fine, star triangulation of a 4-dimensional point configuration with 7
    # points in ZZ^4
    ```
    """
    
    pc = triangulumancer.PointConfiguration(points)
    simp = pc.triangulate_with_heights(heights).simplices()

    return np.array(sorted([sorted(s) for s in simp]))


def _topcom_triangulate(points: ArrayLike) -> np.ndarray:
    """
    **Description:**
    Computes the placing/pushing triangulation using TOPCOM.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`Triangulation`](./triangulation) class when using TOPCOM as
    the backend.
    :::

    **Arguments:**
    - `points`: A list of points.

    **Returns:**
    A list of simplices defining a triangulation.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a triangulation using TOMCOM.
    ```python {2}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    p.triangulate(backend="topcom")
    # A fine, star triangulation of a 4-dimensional point configuration with 7
    # points in ZZ^4
    ```
    """
    
    pc = triangulumancer.PointConfiguration(points)
    simp = pc.fine_triangulation().simplices()

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
) -> "generator[Triangulation]":
    """
    **Description:**
    Computes all triangulations of the input point configuration using TOPCOM.
    There is the option to only compute fine, regular or fine triangulations.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`all_triangulations`](./polytope#all_triangulations) function
    of the [`Polytope`](./polytope) class.
    :::

    **Arguments:**
    - `poly`: The ambient polytope.
    - `pts`: The list of points to be triangulated. Specified by labels.
    - `only_fine`: Restricts to only fine triangulations.
    - `only_regular`: Restricts to only regular triangulations.
    - `only_star`: Restricts to only star triangulations.
    - `star_origin`: The index of the point that will be used as the star
        origin. If the polytope is reflexive this is set to 0, but otherwise it
        must be specified.
    - `backend`: The optimizer used to check regularity computation. The
        available options are "topcom" and the backends of the
        [`is_solid`](./cone#is_solid) function of the [`Cone`](./cone) class.
        If not specified, it will be picked automatically. Note that using
        TOPCOM to check regularity is slower.
    - `raw_output`: Return the triangulations as lists of simplices instead of
        as Triangulation objects.

    **Returns:**
    A generator of [`Triangulation`](./triangulation) objects with the
    specified properties. If `raw_output` is set to True then numpy arrays are
    used instead.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a polytope and find all of its
    triangulations. We try picking different restrictions and see how the number
    of triangulations changes.
    ```python {2,7,9}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
    g = p.all_triangulations()
    next(g) # Takes some time while TOPCOM finds all the triangulations
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 7 points in ZZ^4
    next(g) # Produces the next triangulation immediately
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 7 points in ZZ^4
    len(p.all_triangulations(as_list=True)) # Number of fine, regular, star triangulations
    # 2
    len(p.all_triangulations(only_regular=False, only_star=False, only_fine=False, as_list=True) )# Number of triangularions, no matter if fine, regular, or star
    # 6
    ```
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
    triangs = [[[pts[x] for x in i] for i in t.simplices()] for t in triangs]

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
) -> "generator[Triangulation]":
    """
    Constructs pseudorandom regular (optionally fine and star) triangulations
    of a given point set. This is done by picking random heights around the
    Delaunay heights from a Gaussian distribution.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the
    [`random_triangulations_fast`](./polytope#random_triangulations_fast)
    function of the [`Polytope`](./polytope) class.
    :::

    :::caution important
    This function produces random triangulations very quickly, but it does not
    produce a fair sample. When a fair sampling is required the
    [`random_triangulations_fair`](./polytope#random_triangulations_fair)
    function should be used.
    :::

    **Arguments:**
    - `poly`: The ambient polytope.
    - `pts`: The list of points to be triangulated. Specified by labels.
    - `N`: Number of desired unique triangulations. If not specified, it will
        generate as many triangulations as it can find until it has to retry
        more than max_retries times to obtain a new triangulation.
    - `c`: A constant used as the standard deviation of the Gaussian
        distribution used to pick the heights. A larger c results in a wider
        range of possible triangulations, but with a larger fraction of them
        being non-fine, which slows down the process when using only_fine=True.
    - `max_retries`: Maximum number of attempts to obtain a new triangulation
        before the process is terminated.
    - `make_star`: Converts the obtained triangulations into star
        triangulations.
    - `only_fine`: Restricts to fine triangulations.
    - `backend`: Specifies the backend used to compute the triangulation. The
        available options are "cgal" and "qhull".
    - `seed`: A seed for the random number generator. This can be used to
        obtain reproducible results.
    - `verbosity`: The verbosity level.

    **Returns:**
    A generator of [`Triangulation`](./triangulation) objects with the
    specified properties.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a polytope and find some random
    triangulations. The triangulations are obtained very quickly, but they are
    not a fair sample of the space of triangulations. For a fair sample, the
    [`random_triangulations_fair`](./polytope#random_triangulations_fair)
    function should be used.
    ```python {2,7}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
    g = p.random_triangulations_fast()
    next(g) # Runs very quickly
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 106 points in ZZ^4
    next(g) # Keeps producing triangulations until it has trouble finding more
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 106 points in ZZ^4
    rand_triangs = p.random_triangulations_fast(N=10, as_list=True) # Produces the list of 10 triangulations very quickly
    ```
    """
    # parse inputs
    if seed is not None:
        np.random.seed(seed)

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
        heights = [pt.dot(pt) + np.random.normal(0, c) for pt in poly.points(which=pts)]
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
) -> "generator[Triangulation]":
    r"""
    **Description:**
    Constructs pseudorandom regular (optionally star) triangulations of a given
    point set. Implements Algorithm \#3 from the paper
    *Bounding the Kreuzer-Skarke Landscape* by Mehmet Demirtas, Liam
    McAllister, and Andres Rios-Tascon.
    [arXiv:2008.01730](https://arxiv.org/abs/2008.01730)

    This is a Markov chain Monte Carlo algorithm that involves taking random
    walks inside the subset of the secondary fan corresponding to fine
    triangulations and performing random flips. For details, please see Section
    4.1 in the paper.

    :::note notes
    - This function is not intended to be called by the end user. Instead, it
        is used by the
        [`random_triangulations_fair`](./polytope#random_triangulations_fair)
        function of the [`Polytope`](./polytope) class.
    - This function is designed mainly for large polytopes where sampling
        triangulations is challenging. When small polytopes are used it is
        likely to get stuck.
    :::

    **Arguments:**
    - `poly`: The ambient polytope.
    - `pts`: The list of points to be triangulated. Specified by labels.
    - `N`: Number of desired unique triangulations. If not specified, it will
        generate as many triangulations as it can find until it has to retry
        more than max_retries times to obtain a new triangulation.
    - `n_walk`: Number of hit-and-run steps per triangulation.
    - `n_flip`: Number of random flips performed per triangulation.
    - `initial_walk_steps`: Number of hit-and-run steps to take before starting
        to record triangulations. Small values may result in a bias towards
        Delaunay-like triangulations.
    - `walk_step_size`: Determines size of random steps taken in the secondary
        fan. Algorithm may stall if too small.
    - `max_steps_to_wall`: Maximum number of steps to take towards a wall of
        the subset of the secondary fan that correspond to fine triangulations.
        If a wall is not found, a new random direction is selected. Setting
        this to be very large (>100) reduces performance. If this, or
        walk_step_size, is set to be too low, the algorithm may stall.
    - `fine_tune_steps`: Number of steps to determine the location of a wall.
        Decreasing improves performance, but might result in biased samples.
    - `max_retries`: Maximum number of attempts to obtain a new triangulation
        before the process is terminated.
    - `make_star`: Converts the obtained triangulations into star
        triangulations.
    - `backend`: Specifies the backend used to compute the triangulation. The
        available options are "cgal" and "qhull".
    - `seed`: A seed for the random number generator. This can be used to
        obtain reproducible results.

    **Returns:**
    A generator of [`Triangulation`](./triangulation) objects with the
    specified properties.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We construct a polytope and find some random
    triangulations. The computation takes considerable time, but they should be
    a fair sample from the full set of triangulations (if the parameters are
    chosen correctly). For (some) machine learning purposes or when the
    fairness of the sample is not crucial, the
    [`random_triangulations_fast`](./polytope#random_triangulations_fast)
    function should be used instead.
    ```python {2,7}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
    g = p.random_triangulations_fast()
    next(g) # Takes a long time (around a minute)
    # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
    next(g) # Takes slightly shorter (still around a minute)
    # A fine, regular, star triangulation of a 4-dimensional point
    # configuration with 106 points in ZZ^4
    rand_triangs = p.random_triangulations_fair(N=10, as_list=True) # Produces the list of 10 triangulations, but takes a long time (around 10 minutes)
    ```
    It is worth noting that the time it takes to obtain each triangulation
    varies very significantly on the parameters used. The function tries to
    guess reasonable parameters, but it is better to adjust them to your
    desired balance between speed and fairness of the sampling.
    """
    # input parsing
    traing_pts = poly.points(which=pts)
    num_points = len(pts)

    dim = np.linalg.matrix_rank([tuple(pt) + (1,) for pt in traing_pts]) - 1
    if dim != traing_pts.shape[1]:
        raise Exception("Point configuration must be full-dimensional.")

    if seed is not None:
        np.random.seed(seed)

    # Obtain random Delaunay triangulation by picking random point as origin
    rand_ind = np.random.randint(0, len(pts))
    points_shifted = [p - traing_pts[rand_ind] for p in traing_pts]

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
        if (N is not None) and (len(triang_hashes) > N):
            break

        # find a wall
        # (move along random direction until we hit a non-fine triangulation)
        outside_bounds = False
        for _ in range(max_retries):
            in_pt = old_pt

            # find random direction
            random_dir = np.random.normal(size=num_points)
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
        random_coef = np.random.uniform(0, 1)
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
                    n_flip, only_fine=True, only_regular=True, only_star=True
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
