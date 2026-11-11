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
# Description:  This module contains tools designed to perform polytope
#               computations.
# -----------------------------------------------------------------------------

"""Lattice polytopes and related computations.

This module provides the [`Polytope`][cytools.polytope.Polytope] class, which handles
lattice points, faces, duality, symmetries, normal forms, triangulations, GLSM
charge matrices, and the Hodge numbers of Calabi-Yau hypersurfaces arising from
reflexive polytopes, as well as a few standalone helper functions.
"""

# 'standard' imports
from collections import defaultdict
import copy
import itertools
import math
import time
import warnings

# 3rd party imports
from flint import fmpz_mat, fmpq_mat
import numpy as np
from numpy.typing import ArrayLike
from cytools._ppl import ppl  # ppl, with its FPU rounding side effect undone

# imported for annotations only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from scipy.spatial import ConvexHull
from tqdm import tqdm
import pypalp
import latticepts

# CYTools imports
from cytools import config
from cytools.polytopeface import PolytopeFace
from cytools.triangulation import (
    Triangulation,
    all_triangulations,
    random_triangulations_fast_generator,
    random_triangulations_fair_generator,
)
from cytools.utils import gcd_list, lll_reduce, instanced_lru_cache, integral_nullspace


class Polytope:
    """A lattice polytope.

    This class handles all computations relating to lattice polytopes, such as
    the computation of lattice points and faces. When using reflexive polytopes,
    it also allows the computation of topological properties of the arising
    Calabi-Yau hypersurfaces that only depend on the polytope.

    The polytope is the convex hull of the input points. All lattice points of
    the polytope are computed on construction and are labelled. Input points
    keep the labels given in `labels` (if any), and every other lattice point
    receives an integer label that does not clash with the given ones. Most
    methods accept and return labels, and the helpers
    [`points_to_labels`][cytools.polytope.Polytope.points_to_labels] and
    [`points_to_indices`][cytools.polytope.Polytope.points_to_indices] convert
    between points, labels, and indices.

    The `Polytope` class is also exported at the root of the package, so it can
    be imported from either `cytools.polytope` or `cytools`.

    Args:
        points (ArrayLike): The lattice points whose convex hull defines the
            polytope. CYTools only supports lattice polytopes, so the points
            must have integer coordinates. They must all be distinct, but they
            do not need to be vertices.
        labels (ArrayLike | None): Labels for the input points, i.e.
            `points[i]` is labelled `labels[i]`. The labels must be distinct
            and hashable, and there must be one per input point. If not
            specified, all points are given integer labels following the
            default point ordering (see
            [`points`][cytools.polytope.Polytope.points]).
        backend (str | None): The backend used to construct the convex hull.
            The options are `"ppl"`, `"qhull"`, or `"palp"`. If not specified,
            PPL is used for polytopes of dimension 1 to 4, and PALP otherwise.
            0-dimensional polytopes always use PALP.
        deterministic_glsm_basis (bool): Whether to choose the GLSM (and thus
            divisor) basis in a deterministic manner, so that it is consistent
            across different machines. The basis chosen with this flag may
            differ from the default one.

    Raises:
        ValueError: If `points` is a string, if the points or labels are not
            unique, if the number of labels does not match the number of
            points, or if `backend` is not one of the allowed options.

    Examples:
        We construct two polytopes from lists of points.

        >>> from cytools import Polytope
        >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        >>> print(p1)
        A 4-dimensional reflexive lattice polytope in ZZ^4
        >>> p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        >>> print(p2)
        A 3-dimensional reflexive lattice polytope in ZZ^4

        Custom labels can be given to the input points.

        >>> p3 = Polytope([[1,0],[0,1],[-1,-1]], labels=["a","b","c"])
        >>> p3.labels
        (0, 'c', 'b', 'a')
    """

    def __init__(
        self, points: ArrayLike, labels: ArrayLike = None, backend: str = None,
        deterministic_glsm_basis: bool = False
    ) -> None:
        # input checking
        # --------------
        # reject a string: it would be iterated as single-character "points"
        # and fail with a confusing uniqueness error
        if isinstance(points, str):
            raise ValueError(
                "`points` must be a list of lattice points, not a string "
                f"(got {points!r}). Pass coordinates, e.g. "
                "[[1,0,0,0],[0,1,0,0],...]."
            )

        # check that points are unique
        N_input_pts = len(points)
        N_unique_pts = len({tuple(pt) for pt in points})
        if N_input_pts != N_unique_pts:
            msg = f"Points must all be unique! Out of {N_input_pts} points, "
            msg += f"only {N_unique_pts} of them were unique...\n"
            msg += f"Points = {points}..."

            raise ValueError(msg)

        # check that labels are unique and match point counts
        if labels is not None:
            N_labels = len(labels)
            N_unique_labels = len(set(labels))

            if N_labels != N_unique_labels:
                raise ValueError(
                    "Labels must all be unique! "
                    + f"There were {N_labels}, {N_unique_labels} "
                    + "of them were unique..."
                )

            if N_labels != N_input_pts:
                raise ValueError(
                    f"Count of labels, {N_labels}, must match "
                    + f"the count of points, {N_input_pts}"
                )

        # check that backend is allowed
        backends = ["ppl", "qhull", "palp", None]
        if backend not in backends:
            raise ValueError(
                f"Invalid backend, {backend}." + f" Options are {backends}."
            )

        # initialize attributes
        # ---------------------
        self._deterministic_glsm_basis = deterministic_glsm_basis
        self.clear_cache()

        # process the inputs
        # ------------------
        # dimension
        if N_input_pts == 0:
            self._dim_ambient = None
            self._dim = -1
            self._dim_diff = None
        else:
            self._dim_ambient = len(points[0])
            self._dim = int(
                np.linalg.matrix_rank([list(pt) + [1] for pt in points]) - 1
            )
            self._dim_diff = self.ambient_dim() - self.dim()

        # backend
        if backend is None:
            if 1 <= self.dim() <= 4:
                backend = "ppl"
            else:
                backend = "palp"

        if self.dim() == 0:  # 0-dimensional polytopes are finicky
            backend = "palp"

        self._backend = backend

        # set point information (better basis, H-representation)
        self._process_points(points, labels)

    # defaults
    # ========
    def __repr__(self) -> str:
        """Return an unambiguous string describing the polytope.

        Unlike `__str__`, the string also lists all lattice points of the
        polytope along with their labels.

        Returns:
            str: A string describing the polytope.

        Examples:
            >>> p = Polytope([[1,0],[0,1],[-1,-1]])
            >>> print(repr(p))
            A 2-dimensional reflexive lattice polytope in ZZ^2 with points [(0, 0), ...
        """
        msg = f"A {self.dim()}-dimensional "
        if self.dim() >= 0:
            msg += f"{('reflexive ' if self.is_reflexive() else '')}"
            msg += f"lattice polytope in ZZ^{self.ambient_dim()} "
        else:
            msg += "polytope "
        msg += f"with points {list(self._inputpts2labels.keys())} "
        msg += f"which are labelled {list(self._inputpts2labels.values())}"
        
        return msg

    def __str__(self) -> str:
        """Return a human-readable string describing the polytope.

        Returns:
            str: A string describing the polytope.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> print(p)
            A 4-dimensional reflexive lattice polytope in ZZ^4
        """
        msg = f"A {self.dim()}-dimensional "
        if self.dim() >= 0:
            msg += f"{('reflexive ' if self.is_reflexive() else '')}"
            msg += f"lattice polytope in ZZ^{self.ambient_dim()} "
        else:
            msg += "polytope"
        
        return msg

    def __getstate__(self):
        """Return the state of the instance, for pickling.

        The instance cache used by `instanced_lru_cache` is dropped, since it
        cannot be pickled.

        Returns:
            dict: A shallow copy of the instance's `__dict__`, with the cache
                removed.
        """
        state = self.__dict__.copy()
        # delete instanced_lru_cache since it doesn't play nicely with pickle
        state["_cache"] = None
        return state

    def __setstate__(self, state: dict):
        """Set the state of the instance, for unpickling.

        The instance cache used by `instanced_lru_cache` is re-initialized as an
        empty dictionary.

        Args:
            state (dict): The dictionary of the instance state, read from the
                pickle.
        """
        self.__dict__.update(state)
        # re-initialize the instanced_lru_cache
        # (needed since we check if self has _cache, which it is now None)
        self._cache = {}

    def __eq__(self, other: "Polytope") -> bool:
        """Implement comparison of polytopes with `==`.

        Two polytopes are considered equal if they have the same set of
        vertices. Labels and backends are ignored.

        Args:
            other (Polytope): The other polytope that is being compared.

        Returns:
            bool: `True` if the polytopes are equal, `False` otherwise. If
                `other` is not a `Polytope`, `False` is returned.

        Examples:
            We construct two polytopes and compare them.

            >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p1 == p2
            True
        """
        if not isinstance(other, Polytope):
            return False

        our_verts = self.vertices().tolist()
        other_verts = other.vertices().tolist()
        return sorted(our_verts) == sorted(other_verts)

    def __ne__(self, other: "Polytope") -> bool:
        """Implement comparison of polytopes with `!=`.

        Args:
            other (Polytope): The other polytope that is being compared.

        Returns:
            bool: `True` if the polytopes are different, `False` otherwise.

        Examples:
            We construct two polytopes and compare them.

            >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p1 != p2
            False
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Return the hash value of the polytope.

        The hash is computed from the sorted vertices, so it is consistent with
        `__eq__`. It is cached after the first call.

        Returns:
            int: The hash value of the polytope.

        Examples:
            We compute the hash value of a polytope. We also construct a set
            and a dictionary with a polytope, which make use of the hash
            function.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> h = hash(p)  # obtain the hash value
            >>> d = {p: 1}  # create a dictionary with polytope keys
            >>> s = {p}  # create a set of polytopes
        """
        if self._hash is None:
            self._hash = hash(tuple(sorted(tuple(v) for v in self.vertices())))

        return self._hash

    def __add__(self, other: "Polytope") -> "Polytope":
        """Implement addition of polytopes as their Minkowski sum.

        See [`minkowski_sum`][cytools.polytope.Polytope.minkowski_sum].

        Args:
            other (Polytope): The other polytope used for the Minkowski sum.

        Returns:
            Polytope: The Minkowski sum.

        Raises:
            ValueError: If `other` is not a `Polytope`.

        Examples:
            We construct two polytopes and compute their Minkowski sum.

            >>> p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
            >>> p2 = Polytope([[0,0,1],[0,0,-1]])
            >>> print(p1 + p2)
            A 3-dimensional reflexive lattice polytope in ZZ^3
        """
        if not isinstance(other, Polytope):
            raise ValueError

        return self.minkowski_sum(other)

    # caching
    # =======
    def _dump(self) -> dict:
        """Return all instance attributes.

        No copying is done, so be careful with mutability.

        Returns:
            dict: The instance's `__dict__`, mapping attribute names to values.
        """
        return self.__dict__

    def clear_cache(self) -> None:
        """Clear the cached results of any previous computation.

        This clears cached quantities such as the vertices, faces, dual
        polytope, automorphisms, normal forms, Euler characteristic,
        favorability, GLSM data, and nef partitions. The results of
        [`hpq`][cytools.polytope.Polytope.hpq] are cached separately and are
        not cleared. The lattice points, their labels, and the hyperplane
        representation are computed on construction and are not cleared
        either.

        Examples:
            We compute the faces of a polytope, which are cached, and then
            clear the cache.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> f = p.faces()  # computed and cached
            >>> f = p.faces()  # returned from the cache
            >>> p.clear_cache()  # clear the results of previous computations
            >>> f = p.faces()  # computed again
        """
        # basics
        # ------
        # inputs                (DON'T CLEAR! Set in init...)
        # self._backend

        # simple properties
        self._hash = None
        self._volume = None

        # points
        # ------
        # LLL-reduction         (DON'T CLEAR! Set in init...)
        # self._transf_mat_inv
        # self._transl_vector

        # H-rep                 (DON'T CLEAR! Set in init...)
        # self._ineqs_input
        # self._ineqs_optimal
        # self._poly_optimal
        self._is_reflexive = dict()

        # input, optimal points (DON'T CLEAR! Set in init...)
        # self._labels2inputPts
        # self._labels2optPts
        # self._pts_saturating

        # self._pts_order

        # self._inputpts2labels
        # self._optimalpts2labels
        # self._labels2inds

        # groups of points
        # self._label_origin      = None
        # self._labels_int        = None
        # self._labels_facet      = None
        # self._labels_bdry       = None
        # self._labels_codim2     = None
        # self._labels_not_facet = None
        self._labels_vertices = None

        # others
        # ------
        # dimension             (DON'T CLEAR! Set in init...)
        # self._dim
        # self._dim_ambient
        # self._dim_diff

        # dual, faces, H-rep
        self._dual = None
        self._faces = None

        # symmetries
        self._autos = [None] * 4
        self._Z2_actions = None

        # hodge
        self._chi = None
        self._is_favorable = None

        # glsm
        self._glsm_basis = dict()
        self._glsm_charge_matrix = dict()
        self._glsm_linrels = dict()

        # misc
        self._nef_parts = dict()
        self._normal_form = [None] * 3

    # getters
    # =======
    # (all methods here should be @property)
    @property
    def backend(self) -> str:
        """Return the backend used to compute the convex hull.

        Returns:
            str: The backend, one of `"ppl"`, `"qhull"`, or `"palp"`.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p.backend
            'ppl'
        """
        return self._backend

    def ambient_dimension(self) -> int:
        """Return the dimension of the ambient lattice.

        Also available as `ambient_dim`.

        Returns:
            int | None: The dimension of the ambient lattice, or `None` if the
                polytope was constructed from an empty list of points.

        Examples:
            We construct a polytope and check the dimension of the ambient
            lattice.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
            >>> p.ambient_dimension()
            4
        """
        return self._dim_ambient

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self) -> int:
        """Return the dimension of the polytope.

        Also available as `dim`.

        Returns:
            int: The dimension of the polytope. It is -1 for the empty
                polytope.

        Examples:
            We construct a polytope and check its dimension.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
            >>> p.dimension()
            3
        """
        return self._dim

    # aliases
    dim = dimension

    def is_solid(self) -> bool:
        """Return whether the polytope is solid (i.e. full-dimensional).

        Returns:
            bool: `True` if the dimension of the polytope equals the dimension of
                the ambient lattice, `False` otherwise.

        Examples:
            We construct a polytope and check if it is solid.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
            >>> p.is_solid()
            False
        """
        return self.ambient_dim() == self.dim()

    @property
    def labels(self):
        """Return the labels of all lattice points, in order.

        The order matches the order of the points returned by
        [`points`][cytools.polytope.Polytope.points].

        Returns:
            tuple: The labels of the lattice points.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.labels
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        """
        return self._pts_order

    @property
    def label_origin(self):
        """Return the label of the origin.

        Returns:
            Hashable | None: The label of the origin, or `None` if the origin is
                not a lattice point of the polytope.
        """
        return self._label_origin

    @property
    def labels_vertices(self):
        """Return the labels of the vertices, sorted.

        The vertices are computed (and cached) if needed.

        Returns:
            tuple: The sorted labels of the vertices of the polytope.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.labels_vertices
            (1, 2, 3, 4, 5)
        """
        if self._labels_vertices is None:
            # generate the labels
            self.vertices()
        return self._labels_vertices

    @property
    def labels_int(self):
        """Return the labels of the interior points.

        Interior points are those that do not saturate any inequality of the
        polytope, i.e. the ones in the relative interior.

        Returns:
            tuple: The labels of the interior points.
        """
        return self._labels_int

    @property
    def labels_facet(self):
        """Return the labels of the points interior to facets.

        These are the boundary points that saturate exactly one inequality.

        Returns:
            tuple: The labels of the points interior to facets.
        """
        return self._labels_facet

    @property
    def labels_bdry(self):
        """Return the labels of the boundary points.

        Returns:
            tuple: The labels of the boundary points, ordered by decreasing
                number of saturated inequalities.
        """
        return self._labels_bdry

    @property
    def labels_codim2(self):
        """Return the labels of the boundary points not interior to facets.

        These are the points that saturate at least two inequalities, i.e.
        the ones lying in faces of codimension 2 or higher.

        Returns:
            tuple: The labels of the boundary points not interior to facets.
        """
        return self._labels_codim2

    @property
    def labels_not_facet(self):
        """Return the labels of the points not interior to facets.

        These are the interior points followed by the boundary points not
        interior to facets.

        Returns:
            tuple: The labels of the points not interior to facets.
        """
        return self._labels_not_facet

    def inequalities(self) -> np.ndarray:
        r"""Return the inequalities giving the hyperplane representation.

        Each row $(c_0, \ldots, c_{d-1}, c_d)$ encodes the inequality

        $$c_0x_0 + \cdots + c_{d-1}x_{d-1} + c_d \geq 0.$$

        For polytopes that are not full-dimensional, the equalities defining
        the affine hull of the polytope are not included, so the inequalities
        only define the polytope when restricted to its affine hull.

        Returns:
            np.ndarray: The inequalities defining the polytope, one per row, as
                an integer array of shape `(n_ineqs, ambient_dim + 1)`.

        Raises:
            NotImplementedError: If the polytope is empty.

        Examples:
            We construct a polytope and find the defining inequalities.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p.inequalities()
            array([[ 4, -1, -1, -1,  1],
                   [-1,  4, -1, -1,  1],
                   [-1, -1,  4, -1,  1],
                   [-1, -1, -1,  4,  1],
                   [-1, -1, -1, -1,  1]])
        """
        if self._dim < 0:
            raise NotImplementedError
        return np.array(self._ineqs_input)

    # points
    # ======
    # internal/prep
    # -------------
    def _process_points(self, pts_input: ArrayLike, labels: ArrayLike = None) -> None:
        """Process the input points.

        Internal function that should only be called once, from the
        constructor. It computes the "optimal" representation of the points
        (translated so that the affine hull passes through the origin, if the
        polytope is not full-dimensional, and LLL-reduced), the hyperplane
        representation, all lattice points and their saturated inequalities,
        and the point labels and orderings.

        It sets the attributes `_transl_vector`, `_transf_mat_inv`,
        `_poly_optimal`, `_ineqs_optimal`, `_ineqs_input`, `_labels2optPts`,
        `_labels2inputPts`, `_pts_saturating`, `_pts_order`, `_inputpts2labels`,
        `_optimalpts2labels`, `_labels2inds`, `_label_origin`, `_labels_int`,
        `_labels_facet`, `_labels_bdry`, `_labels_codim2`, and
        `_labels_not_facet`.

        Args:
            pts_input (ArrayLike): The points input by the user.
            labels (ArrayLike | None): The point labels input by the user. If not
                specified, default integer labels are used.
        """
        if len(pts_input) == 0:
            self._transl_vector     = None
            self._transf_mat_inv    = None
            self._poly_optimal      = None
            self._ineqs_optimal     = None
            self._labels2optPts     = dict()
            self._labels2inputPts   = dict()
            self._pts_saturating    = dict()
            self._pts_order         = tuple()
            self._inputpts2labels   = dict()
            self._optimalpts2labels = dict()
            self._labels2inds       = dict()
            self._label_origin      = None
            self._labels_int        = tuple()
            self._labels_facet      = tuple()
            self._labels_bdry       = tuple()
            self._labels_codim2     = tuple()
            self._labels_not_facet  = tuple()

            return

        # Find 'optimal' representation
        # -----------------------------
        # translate if not full-dim (allows LLL-reduction)
        if self.is_solid():
            self._transl_vector = np.zeros(self.ambient_dim(), dtype=int)
        else:
            self._transl_vector = pts_input[0]
        pts_optimal = np.array(pts_input) - self._transl_vector

        # 0D is a special case
        if self._dim == 0:
            # no LLL reduction; optimal space is 0-dimensional
            self._transf_mat_inv = np.eye(self.ambient_dim(), dtype=int)

            # no non-trivial inequalities on a single point
            self._ineqs_optimal = np.zeros((0, 1), dtype=int)
            self._poly_optimal  = None
            self._ineqs_input   = np.zeros((0, self.ambient_dim() + 1), dtype=int)

            # the one point (origin in optimal coords) and its data
            label = labels[0] if labels is not None else 0
            opt_pt   = ()
            input_pt = tuple(int(x) for x in pts_input[0])

            self._labels2optPts   = {label: opt_pt}
            self._pts_saturating  = {label: frozenset()}

            self._pts_order         = (label,)
            self._labels2inputPts   = {label: input_pt}
            self._inputpts2labels   = {input_pt: label}
            self._optimalpts2labels = {opt_pt: label}
            self._labels2inds       = {label: 0}

            # origin handling: the single point is "interior" (no saturations)
            origin = (0,) * self.ambient_dim()
            self._label_origin = self._inputpts2labels.get(origin, None)

            self._labels_int       = (label,)
            self._labels_facet     = tuple()
            self._labels_bdry      = tuple()
            self._labels_codim2    = tuple()
            self._labels_not_facet = (label,)

            return

        # >0D below...
        # ============
        # LLL-reduction (allows reduction in dimension)
        pts_optimal, transf = lll_reduce(pts_optimal, transform=True)
        pts_optimal = pts_optimal[:, self._dim_diff :]
        transf_mat, self._transf_mat_inv = transf

        # Calculate the polytope, inequalities
        # ------------------------------------
        out = poly_v_to_h(pts_optimal, self._backend)
        self._ineqs_optimal, self._poly_optimal = out

        # convert to input representation
        shape_opt = self._ineqs_optimal.shape
        shape = (shape_opt[0], shape_opt[1] + self._dim_diff)

        self._ineqs_input = np.zeros(shape, dtype=int)
        self._ineqs_input[:, self._dim_diff :] = self._ineqs_optimal
        self._ineqs_input[:, :-1] = transf_mat.T.dot(self._ineqs_input[:, :-1].T).T

        if self.is_solid():
            self._ineqs_input[:, -1] = self._ineqs_optimal[:, -1]
        else:
            # this method is always correct, just a bit slower
            for i, v in enumerate(self._ineqs_input):
                self._ineqs_input[i, -1] = self._ineqs_optimal[i, -1] - v[:-1].dot(
                    self._transl_vector
                )

        # Get the lattice points and their saturated inequalities
        # -------------------------------------------------------
        pts_optimal = [tuple(pt) for pt in pts_optimal]
        pts_optimal_all, saturating = saturating_lattice_pts(
            pts_optimal, self._ineqs_optimal, self.dim(), self._backend
        )

        # undo LLL transformation, to get points in original basis
        pts_input_all = self._optimal_to_input(pts_optimal_all)

        # Assign labels, organize points by saturated inequalities
        # --------------------------------------------------------
        # the sorting function
        def sort_fct(ind):
            # order:
            #   -) interior points first
            #   -) rest by decreasing # of saturated inequalities
            # ties (i.e., 2+ points with the same # of saturated inequalities)
            # are broken by lexicographical ordering on the (input)
            # coordinates.
            out = []

            # the number of saturated inequalities
            if len(saturating[ind]) > 0:
                out.append(-len(saturating[ind]))
            else:
                out.append(-float("inf"))

            # the coordinates
            out += list(pts_input_all[ind])
            return out

        # sort it!
        inds_sort = sorted(range(len(pts_input_all)), key=sort_fct)

        # save info to useful variables/dictionaries
        self._labels2optPts = dict()
        self._pts_saturating = dict()
        nSat_to_labels = [[] for _ in range(len(self._ineqs_optimal) + 1)]

        if labels is None:
            labels = []

        # map each user-supplied point to its label, and keep the labels in a
        # set. Doing this up-front keeps the loop below linear in the number of
        # lattice points (rather than doing list scans per point).
        # N.B.: setdefault reproduces list.index() semantics, i.e. in case of
        # duplicated input points the first label wins.
        pt_to_label = {}
        for pt, label in zip(pts_optimal, labels):
            pt_to_label.setdefault(pt, label)
        labels_set = set(labels)

        last_default_label = -1
        for i in inds_sort:
            pt = tuple(pts_optimal_all[i])

            # find the label to use
            if pt in pt_to_label:
                label = pt_to_label[pt]
            else:
                label = last_default_label + 1

                while (label in self._labels2optPts) or (label in labels_set):
                    label += 1
                last_default_label = label

            # save it!
            self._labels2optPts[label] = pt
            self._pts_saturating[label] = saturating[i]
            nSat_to_labels[len(saturating[i])].append(label)

        # save order of labels
        self._pts_order = list(
            itertools.chain(nSat_to_labels[0], *nSat_to_labels[1:][::-1])
        )
        # if hasattr(self._pts_order[0], 'item'):
        #    # convert numpy types to ordinary ones
        #    self._pts_order = tuple([i.item() for i in self._pts_order])
        # else:
        #    self._pts_order = tuple([i for i in self._pts_order])
        self._pts_order = tuple(
            [i.item() if hasattr(i, "item") else i for i in self._pts_order]
        )

        # dictionary from labels to input coordinates
        pts_input_all = self._optimal_to_input(self.points(optimal=True))
        self._labels2inputPts = {
            label: tuple(map(int, pt))
            for label, pt in zip(self._pts_order, pts_input_all)
        }

        # reverse dictionaries
        self._inputpts2labels = {v: k for k, v in self._labels2inputPts.items()}
        self._optimalpts2labels = {v: k for k, v in self._labels2optPts.items()}

        self._labels2inds = {v: i for i, v in enumerate(self._pts_order)}

        # common sets of labels
        # ---------------------
        origin = (0,) * self.ambient_dim()
        if origin in self._inputpts2labels:
            self._label_origin = self._inputpts2labels[origin]
        else:
            self._label_origin = None

        self._labels_int = nSat_to_labels[0]
        self._labels_facet = nSat_to_labels[1]

        self._labels_bdry = list(itertools.chain(*nSat_to_labels[1:][::-1]))
        self._labels_codim2 = list(itertools.chain(*nSat_to_labels[2:][::-1]))

        self._labels_not_facet = self._labels_int + self._labels_codim2

        # store as tuples
        self._labels_int = tuple(self._labels_int)
        self._labels_facet = tuple(self._labels_facet)
        self._labels_bdry = tuple(self._labels_bdry)
        self._labels_codim2 = tuple(self._labels_codim2)
        self._labels_not_facet = tuple(self._labels_not_facet)

    def _optimal_to_input(self, pts_opt: ArrayLike) -> np.array:
        """Map points from the optimal representation to the input one.

        Points are stored internally in an "optimal" representation
        (translated and LLL-reduced), which eases computations, but answers
        are always returned in the original representation. This function
        performs the mapping.

        This map is somewhat costly so, whenever possible, use the
        `_optimalpts2labels` and `_labels2inputPts` dictionaries instead.

        Args:
            pts_opt (ArrayLike): The points in the optimal representation, one
                per row.

        Returns:
            np.ndarray: The points in the original representation, one per row.
        """

        # *** could be sped up by using pre-calculated dicts ***

        # pad points with 0s, to make width match original dim
        points_orig = np.empty((len(pts_opt), self.ambient_dim()), dtype=int)
        points_orig[:, self._dim_diff :] = pts_opt
        points_orig[:, : self._dim_diff] = 0

        # undo the LLL-reduction
        points_orig = self._transf_mat_inv.dot(points_orig.T).T

        # undo the translation, if applicable
        if not self.is_solid():
            for i in range(len(points_orig)):
                points_orig[i, :] += self._transl_vector

        return points_orig

    # main
    # ----
    def points(
        self, which=None, optimal: bool = False, as_indices: bool = False
    ) -> np.ndarray:
        """Return the lattice points of the polytope.

        Points are sorted so that interior points come first, and then the
        rest are arranged by decreasing number of saturated inequalities, with
        ties broken lexicographically by their coordinates. For reflexive
        polytopes this is useful since the origin will be at index 0 and the
        boundary points interior to facets will be last.

        Also available as `pts`.

        Args:
            which (Hashable | list | np.ndarray | None): The label, or list of
                labels, of the points to return. Note that these are labels and
                not indices. If not specified, all points are returned.
            optimal (bool): Whether to return the points in their optimal
                coordinates (i.e. translated and LLL-reduced, and with the
                dimension reduced if the polytope is not full-dimensional).
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The lattice points, one per row. If
                `as_indices` is true, a list of indices is returned instead.

        Examples:
            We construct a polytope and compute the lattice points. One can
            verify that the first point is the only interior point, and the
            last three points are the ones interior to facets. Thus it follows
            the aforementioned ordering.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.points()
            array([[ 0,  0,  0,  0],
                   [-1, -1, -6, -9],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0],
                   [ 1,  0,  0,  0],
                   [ 0,  0, -2, -3],
                   [ 0,  0, -1, -2],
                   [ 0,  0, -1, -1],
                   [ 0,  0,  0, -1]])

            Specific points can be selected by their labels.

            >>> p.points(which=[3,1])
            array([[ 0,  0,  1,  0],
                   [-1, -1, -6, -9]])
        """
        # get the labels of the points to return
        if which is None:
            which = self._pts_order
        elif (not isinstance(which, np.ndarray)) and (which in self._pts_order):
            which = [which]

        # return the answer in the desired format
        if as_indices:
            return [self._labels2inds[label] for label in which]
        else:
            # set pts to be optimal/input depending on 'optimal' parameter
            if optimal:
                pts = self._labels2optPts
            else:
                pts = self._labels2inputPts

            # return
            return np.array([pts[label] for label in which])

    # aliases
    pts = points

    # common point grabbers
    # ---------------------
    def pts_int(self, as_indices=False):
        """Return the interior lattice points of the polytope.

        These are the points in the relative interior of the polytope. The
        ordering is the same as in [`points`][cytools.polytope.Polytope.points].

        Also available as `interior_points` and `interior_pts`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The interior points, one per row, or their
                indices if `as_indices` is true.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.interior_points()
            array([[0, 0, 0, 0]])
        """
        return self.pts(which=self._labels_int, as_indices=as_indices)

    def pts_bdry(self, as_indices=False):
        """Return the boundary lattice points of the polytope.

        The points are ordered by decreasing number of saturated inequalities,
        as in [`points`][cytools.polytope.Polytope.points].

        Also available as `boundary_points` and `bdry_points`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The boundary points, one per row, or their
                indices if `as_indices` is true.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.boundary_points(as_indices=True)
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        return self.pts(which=self._labels_bdry, as_indices=as_indices)

    def pts_facet(self, as_indices=False):
        """Return the lattice points interior to facets.

        These are the boundary points that saturate exactly one inequality.

        Also available as `points_interior_to_facets` and
        `pts_interior_to_facets`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The points interior to facets, one per row,
                or their indices if `as_indices` is true.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.points_interior_to_facets()
            array([[ 0,  0, -1, -2],
                   [ 0,  0, -1, -1],
                   [ 0,  0,  0, -1]])
        """
        return self.pts(which=self._labels_facet, as_indices=as_indices)

    def pts_codim2(self, as_indices=False):
        """Return the boundary lattice points not interior to facets.

        These are the points that saturate at least two inequalities, i.e.
        those that lie in faces of codimension 2 or higher.

        Also available as `boundary_points_not_interior_to_facets` and
        `boundary_pts_not_interior_to_facets`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The boundary points not interior to facets,
                one per row, or their indices if `as_indices` is true.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.boundary_points_not_interior_to_facets(as_indices=True)
            [1, 2, 3, 4, 5, 6]
        """
        return self.pts(which=self._labels_codim2, as_indices=as_indices)

    def pts_not_facets(self, as_indices=False):
        """Return the lattice points not interior to facets.

        These are the interior points followed by the boundary points not
        interior to facets. For reflexive polytopes, these are the points
        typically used to construct triangulations, since the points interior
        to facets correspond to divisors that do not intersect a generic
        Calabi-Yau hypersurface.

        Also available as `points_not_interior_to_facets` and
        `pts_not_interior_to_facets`.

        Args:
            as_indices (bool): Whether to return the indices of the points in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The points not interior to facets, one per
                row, or their indices if `as_indices` is true.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.points_not_interior_to_facets(as_indices=True)
            [0, 1, 2, 3, 4, 5, 6]
        """
        return self.pts(which=self._labels_not_facet, as_indices=as_indices)

    # aliases
    interior_points = pts_int
    interior_pts = pts_int
    boundary_points = pts_bdry
    bdry_points = pts_bdry
    points_interior_to_facets = pts_facet
    pts_interior_to_facets = pts_facet

    boundary_points_not_interior_to_facets = pts_codim2
    boundary_pts_not_interior_to_facets = pts_codim2

    points_not_interior_to_facets = pts_not_facets
    pts_not_interior_to_facets = pts_not_facets

    def points_to_labels(
        self, points: ArrayLike, is_optimal: bool = False
    ) -> "list | None":
        """Return the labels corresponding to the given points.

        It also accepts a single point, in which case it returns the
        corresponding label.

        Args:
            points (ArrayLike): A point or a list of points.
            is_optimal (bool): Whether the points are given in the optimal
                (`True`) or input (`False`) coordinates.

        Returns:
            list | Hashable: The list of labels corresponding to the given
                points, or the label of the point if only one is given. An
                empty list is returned if `points` is empty.

        Raises:
            KeyError: If any of the points is not a lattice point of the
                polytope.

        Examples:
            We construct a polytope with custom labels and find the labels of
            some of its points.

            >>> p = Polytope([[1,0],[0,1],[-1,-1]], labels=["a","b","c"])
            >>> p.points_to_labels([0,1])
            'b'
            >>> p.points_to_labels([[0,0],[-1,-1]])
            [0, 'c']
        """
        # check for empty input
        if len(points) == 0:
            return []

        # map single-point input into list case
        single_pt = len(np.array(points).shape) == 1
        if single_pt:
            points = [points]

        # get relevant dictionary
        if is_optimal:
            relevant_map = self._optimalpts2labels
        else:
            relevant_map = self._inputpts2labels

        # get/return the indices
        labels = [relevant_map[tuple(pt)] for pt in points]
        if single_pt and len(labels):
            return labels[0]  # just return the single label
        else:
            return labels  # return a list of labels

    def points_to_indices(
        self, points: ArrayLike, is_optimal: bool = False
    ) -> "np.ndarray | int":
        """Return the indices corresponding to the given points.

        The indices refer to the list of points returned by
        [`points`][cytools.polytope.Polytope.points]. It also accepts a single
        point, in which case it returns the corresponding index.

        Args:
            points (ArrayLike): A point or a list of points.
            is_optimal (bool): Whether the points are given in the optimal
                (`True`) or input (`False`) coordinates.

        Returns:
            list[int] | int: The list of indices corresponding to the given
                points, or the index of the point if only one is given. An
                empty integer array is returned if `points` is empty.

        Raises:
            KeyError: If any of the points is not a lattice point of the
                polytope.

        Examples:
            We construct a polytope and find the indices of some of its
            points. For a single point, a single index is returned.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.points_to_indices([-1,-1,-6,-9])
            1

            For a list of points, a list of indices is returned.

            >>> p.points_to_indices([[-1,-1,-6,-9],[0,0,0,0],[0,0,1,0]])
            [1, 0, 3]
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
        inds = self.points(which=labels, as_indices=True)

        # get/return the indices
        if single_pt and len(inds):
            return inds[0]  # just return the single index
        else:
            return inds  # return a list of indices

    def vertices(self, optimal: bool = False, as_indices: bool = False) -> np.ndarray:
        """Return the vertices of the polytope.

        The vertices are sorted by their labels, and the result is cached.

        Args:
            optimal (bool): Whether to return the vertices in their optimal
                coordinates.
            as_indices (bool): Whether to return the indices of the vertices in
                the full list of points instead of their coordinates.

        Returns:
            np.ndarray | list[int]: The vertices of the polytope, one per row,
                or their indices if `as_indices` is true.

        Examples:
            We construct a polytope and find its vertices. We can see that they
            match the points that we used to construct the polytope.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p.vertices()
            array([[-1, -1, -1, -1],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0],
                   [ 1,  0,  0,  0]])
        """
        # return the answer if known
        if self._labels_vertices is not None:
            return self.pts(
                which=self._labels_vertices,
                optimal=optimal,
                as_indices=as_indices,
            )

        # calculate the answer
        if self.dim() <= 0:
            # 0D... trivial
            self._labels_vertices = self._pts_order

        elif self._backend == "qhull":
            if self.dim() == 1:  # QHull cannot handle 1D polytopes
                self._labels_vertices = self._labels_facet
            else:
                verts = self._poly_optimal.points[self._poly_optimal.vertices]
        else:
            # get the vertices
            if self._backend == "ppl":
                verts = []
                for pt in self._poly_optimal.minimized_generators():
                    verts.append(pt.coefficients())
                verts = np.array(verts, dtype=int)

            else:  # Backend is PALP
                p = pypalp.Polytope(self.points(optimal=True))
                verts = p.vertices()

        # for either ppl/PALP, map points to original representation
        if self._labels_vertices is None:
            self._labels_vertices = self.points_to_labels(verts, is_optimal=True)

        # sort, map to tuple
        self._labels_vertices = tuple(sorted(self._labels_vertices))

        # return
        return self.vertices(optimal=optimal, as_indices=as_indices)

    # faces
    # =====
    def faces(self, d: int = None) -> tuple:
        """Compute the faces of the polytope.

        The faces of each dimension are sorted lexicographically by their
        labels, and the result is cached. If the faces of the dual polytope
        have already been computed, they are obtained by duality. Otherwise,
        when the polytope is 4-dimensional, the slightly more optimized
        `_faces4d` function is used.

        Args:
            d (int | None): The dimension of the desired faces. If not
                specified, the faces of all dimensions are returned.

        Returns:
            tuple: If `d` is specified, a tuple of
                [`PolytopeFace`][cytools.polytopeface.PolytopeFace] objects of
                dimension `d`. Otherwise, a tuple of tuples of
                [`PolytopeFace`][cytools.polytopeface.PolytopeFace] objects
                organized in ascending dimension, so that the $i$-th entry
                contains the $i$-dimensional faces. The last entry contains the
                polytope itself as a face.

        Raises:
            NotImplementedError: If the polytope is empty.
            ValueError: If `d` is not in the range from 0 to the dimension of
                the polytope.

        Examples:
            We show that this function returns a tuple of 2-faces if `d` is set
            to 2. Otherwise, the function returns all faces in tuples organized
            in ascending dimension. We verify that the first element in the
            tuple of 2-faces is the same as the first element in the
            corresponding subtuple in the tuple of all faces.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> faces2d = p.faces(2)
            >>> allfaces = p.faces()
            >>> print(faces2d[0])  # print the first 2-face
            A 2-dimensional face of a 4-dimensional polytope in ZZ^4
            >>> faces2d[0] is allfaces[2][0]
            True
        """
        if self.dim() < 0:
            raise NotImplementedError

        # input checking
        if (d is not None) and (d not in range(self.dim() + 1)):
            raise ValueError(f"Polytope does not have faces of dimension {d}")

        # return answer if known
        if self._faces is not None:
            return self._faces[d] if (d is not None) else self._faces

        # calculate the answer
        # ====================
        # see if we can punt the problem
        # ------------------------------
        if (self._dual is not None) and (self._dual._faces is not None):
            # can convert answer from dual-polytope!
            self._faces = []

            # codim>0 faces in increasing order of dimension
            for dim_faces in self._dual._faces[::-1][1:]:
                self._faces.append(tuple(f.dual() for f in dim_faces))
            # full-dim face
            self._faces.append(
                (PolytopeFace(self, self.labels_vertices, frozenset(), dim=self._dim),)
            )

            # sort each collection of faces lexicographically by their used labels
            for i in range(len(self._faces)):
                self._faces[i] = tuple(sorted(self._faces[i], key=lambda f: f.labels))

            # cast to tuple
            self._faces = tuple(self._faces)

        elif self.dim() == 4:
            # can use optimized method for 4d polytopes
            self._faces = self._faces4d()

            # sort each collection of faces lexicographically by their used labels
            self._faces = list(self._faces)
            for i in range(len(self._faces)):
                self._faces[i] = tuple(sorted(self._faces[i], key= lambda f:f.labels))
            self._faces = tuple(self._faces)

        # return, if we just figured it out
        if self._faces is not None:
            return self.faces(d)

        # have to calculate from scratch...
        # ---------------------------------
        # get vertices, along with their saturated inequalities
        verts = [tuple(pt) for pt in self.vertices()]
        vert_sat = [self._pts_saturating[label] for label in self.labels_vertices]
        vert_legacy = list(zip(verts, vert_sat))

        # construct faces in reverse order (decreasing dim)
        self._faces = []

        # full-dim face
        self._faces.append(
            (PolytopeFace(self, self.labels_vertices, frozenset(), dim=self.dim()),)
        )
        # if polytope is 0-dimensional, we're done!
        if self.dim() == 0:
            self._faces = tuple(self._faces)
            return self.faces(d)

        # not done... construct the codim>0 faces
        #
        # do so in decreasing order of dimension, by constructing a map,
        # ineq2pts, from size-dd sets of inequalities (indicating faces of
        # dim-dd) to the points saturating them
        #
        # then, to get dim-(dd-1) faces, just take intersections of dim-dd ones
        for dd in range(self.dim() - 1, 0, -1):
            # map from inequalities to points saturating them
            ineq2pts = defaultdict(set)

            if dd == self.dim() - 1:
                # facets... for f-th facet, just collect all points saturating
                # said inequality
                for pt in vert_legacy:
                    for f in pt[1]:
                        ineq2pts[frozenset([f])].add(pt)
            else:
                # codim>1 faces... take intersections of higher-dim faces
                for f1, f2 in itertools.combinations(ineq2pts_prev.values(), 2):  # noqa: F821  (bound below, on the previous iteration)
                    # check if their intersection has the right dimension
                    inter = f1 & f2
                    dim = np.linalg.matrix_rank([pt[0] + (1,) for pt in inter]) - 1
                    if dim != dd:
                        continue

                    # it does! grab saturated inequalities
                    ineqs = frozenset.intersection(*[pt[1] for pt in inter])
                    ineq2pts[ineqs] = inter

            # save dim-dd faces to self._faces
            dd_faces = []
            for f in ineq2pts.keys():
                tmp_vert = [pt[0] for pt in vert_legacy if f.issubset(pt[1])]
                dd_faces.append(
                    PolytopeFace(self, self.points_to_labels(tmp_vert), f, dim=dd)
                )

            self._faces.append(dd_faces)

            # store for next iteration
            # (read at the top of the next pass; ruff reads the loop body as
            #  straight-line code, so it sees a store with no load here and a
            #  load with no binding there -- hence the F821 noqa above too)
            ineq2pts_prev = ineq2pts  # noqa: F841

        # Finally add vertices
        self._faces.append(
            [
                PolytopeFace(self, self.points_to_labels([pt[0]]), pt[1], dim=0)
                for pt in vert_legacy
            ]
        )

        # reverse order (to increasing with dimension)
        self._faces = tuple(tuple(ff) for ff in self._faces[::-1])

        # sort each collection of faces lexicographically by their used labels
        self._faces = list(self._faces)
        for i in range(len(self._faces)):
            self._faces[i] = tuple(sorted(self._faces[i], key= lambda f:f.labels))
        self._faces = tuple(self._faces)

        return self.faces(d)

    def _faces4d(self) -> tuple:
        """Compute the faces of a 4-dimensional polytope.

        This function is a slightly more optimized version of
        [`faces`][cytools.polytope.Polytope.faces] for 4-dimensional polytopes.
        It should typically not be called directly; it is called by
        [`faces`][cytools.polytope.Polytope.faces] when the polytope is
        4-dimensional. Unlike [`faces`][cytools.polytope.Polytope.faces], it
        neither sorts nor caches the result.

        Returns:
            tuple: A tuple of tuples of
                [`PolytopeFace`][cytools.polytopeface.PolytopeFace] objects
                organized in ascending dimension.

        Examples:
            We construct a 4D polytope and compute its faces. Since this
            function generally should not be used directly, we do this with
            [`faces`][cytools.polytope.Polytope.faces].

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> allfaces = p.faces()  # _faces4d is used since p is 4-dimensional
            >>> print(allfaces[1][0])  # print the first 1-dimensional face
            A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        """
        assert self.dim() == 4

        # get vertices, along with their saturated inequalities
        verts = [tuple(pt) for pt in self.vertices()]
        vert_sat = [self._pts_saturating[label] for label in self.labels_vertices]
        vert_legacy = list(zip(verts, vert_sat))

        # facets
        facets = defaultdict(set)
        for pt in vert_legacy:
            for f in pt[1]:
                facets[frozenset([f])].add(pt)

        # 2-faces
        twofaces = defaultdict(set)
        for ineqs1, ineqs2 in itertools.combinations(facets.keys(), 2):
            inter = facets[ineqs1] & facets[ineqs2]

            # These intersections are 2D iff there are at least 3 vertices.
            if len(inter) >= 3:
                ineqs3 = ineqs1 | ineqs2
                twofaces[ineqs3] = inter

        # 1-faces
        onefaces = defaultdict(set)
        for f1, f2 in itertools.combinations(twofaces.values(), 2):
            inter = f1 & f2

            # These intersections are 1D iff there are exactly 2 vertices.
            if len(inter) == 2:
                inter_list = list(inter)
                f3 = inter_list[0][1] & inter_list[1][1]

                if f3 not in onefaces.keys():
                    onefaces[f3] = inter

        # now, construct all formal face objects
        fourface_obj_list = [
            PolytopeFace(self, self.labels_vertices, frozenset(), dim=4)
        ]

        facets_obj_list = []
        for f in facets.keys():
            tmp_vert = self.points_to_labels(
                [pt[0] for pt in vert_legacy if f.issubset(pt[1])]
            )
            facets_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=3))

        twofaces_obj_list = []
        for f in twofaces.keys():
            tmp_vert = self.points_to_labels(
                [pt[0] for pt in vert_legacy if f.issubset(pt[1])]
            )
            twofaces_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=2))

        onefaces_obj_list = []
        for f in onefaces.keys():
            tmp_vert = self.points_to_labels(
                [pt[0] for pt in vert_legacy if f.issubset(pt[1])]
            )
            onefaces_obj_list.append(PolytopeFace(self, tmp_vert, f, dim=1))

        zerofaces_obj_list = [
            PolytopeFace(self, self.points_to_labels([pt[0]]), pt[1], dim=0)
            for pt in vert_legacy
        ]

        # organize in tuple and return
        organized_faces = (
            tuple(zerofaces_obj_list),
            tuple(onefaces_obj_list),
            tuple(twofaces_obj_list),
            tuple(facets_obj_list),
            tuple(fourface_obj_list),
        )
        return organized_faces

    def facets(self) -> tuple[PolytopeFace]:
        """Return the facets (codimension-1 faces) of the polytope.

        Returns:
            tuple[PolytopeFace, ...]: The
                [`PolytopeFace`][cytools.polytopeface.PolytopeFace] objects of
                codimension 1.

        Examples:
            We construct a polytope and find its facets.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> facets = p.facets()
            >>> len(facets)
            5
        """
        return self.faces(self.dim() - 1)

    # H-rep, dual
    # ===========
    def dual_polytope(self) -> "Polytope":
        r"""Return the dual polytope (also called polar polytope).

        The dual polytope of a polytope $\Delta \subset \mathbb{R}^n$ is

        $$\Delta^\circ = \{y\in \mathbb{R}^n \mid x\cdot y \geq -1
        \text{ for all } x \in \Delta\}.$$

        A lattice polytope is reflexive if its dual is also a lattice polytope.
        Only lattice polytopes are currently supported, so only duals of
        reflexive polytopes (with the origin as their unique interior point)
        can be computed. The vertices of the dual polytope are the normal
        vectors of the inequalities returned by
        [`inequalities`][cytools.polytope.Polytope.inequalities].

        The result is cached, and the dual of the dual polytope is the
        original `Polytope` object.

        Also available as `dual`, `polar_polytope`, and `polar`.

        Returns:
            Polytope: The dual polytope.

        Raises:
            NotImplementedError: If the polytope is empty, or if it is not
                reflexive without translations (see
                [`is_reflexive`][cytools.polytope.Polytope.is_reflexive]).

        Examples:
            We construct a reflexive polytope and find its dual. We then verify
            that the dual of the dual is the original polytope.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p_dual = p.dual_polytope()
            >>> print(p_dual)
            A 4-dimensional reflexive lattice polytope in ZZ^4
            >>> p_dual_dual = p_dual.dual_polytope()
            >>> p_dual_dual is p
            True
        """
        if self.dim() < 0:
            raise NotImplementedError

        # return answer if known
        if self._dual is not None:
            return self._dual

        # calculate the answer
        if not self.is_reflexive(allow_translations=False):
            raise NotImplementedError(
                "Duality of non-reflexive polytopes " + "is not supported."
            )

        pts = np.array(self._ineqs_input[:, :-1])
        self._dual = Polytope(
            pts,
            backend=self._backend,
            deterministic_glsm_basis=self._deterministic_glsm_basis,
        )
        self._dual._dual = self
        return self._dual

    # aliases
    dual = dual_polytope
    polar_polytope = dual_polytope
    polar = dual_polytope

    def is_reflexive(self, allow_translations=True) -> bool:
        """Return whether the polytope is reflexive.

        A lattice polytope is reflexive if it has a unique interior lattice
        point and every facet is at lattice distance 1 from it. Polytopes that
        are not full-dimensional are checked relative to the lattice of their
        affine hull. Polytopes of dimension 0 or lower are never considered
        reflexive.

        The result is cached.

        Args:
            allow_translations (bool): Whether to allow the polytope to be
                translated. If `True`, the polytope is checked for reflexivity
                after translating its unique interior point to the origin. If
                `False`, the origin must be the unique interior point (and, for
                polytopes that are not full-dimensional, the affine hull must
                contain the origin).

        Returns:
            bool: `True` if the polytope is reflexive, `False` otherwise.

        Examples:
            We construct a polytope and check if it is reflexive.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.is_reflexive()
            True

            A translated reflexive polytope is only reflexive if translations
            are allowed.

            >>> p = Polytope([[2,1],[1,2],[0,0]])
            >>> p.is_reflexive()
            True
            >>> p.is_reflexive(allow_translations=False)
            False
        """
        if self.dim() <= 0:
            return False

        # check if we know the answer
        if allow_translations in self._is_reflexive:
            return self._is_reflexive[allow_translations]

        # calculate the answer
        if allow_translations:
            # reflexive after translating v to the origin iff every facet
            # then sits at distance 1. optimal frame, so non-solid works too
            interior = self.points(which=self._labels_int, optimal=True)
            if len(interior) != 1:
                self._is_reflexive[allow_translations] = False
            else:
                v = interior[0]
                self._is_reflexive[allow_translations] = bool(
                    np.all(
                        self._ineqs_optimal[:, :-1] @ v
                        + self._ineqs_optimal[:, -1]
                        == 1
                    )
                )
        else:
            if self.is_solid():
                self._is_reflexive[allow_translations] = all(
                    c == 1 for c in self._ineqs_input[:, -1]
                )
            else:
                # integral_nullspace returns the null vectors as *columns*, so
                # the codimension is the number of columns, not of rows.
                codim = integral_nullspace(self.points()).shape[1]
                dim = self.ambient_dim()-codim
                if dim!=self.dim():
                    self._is_reflexive[allow_translations] = False
                else:
                    p = Polytope(
                        lll_reduce(self.points())[:, -self.dim() :],
                        backend=self._backend,
                        deterministic_glsm_basis=self._deterministic_glsm_basis,
                    )
                    self._is_reflexive[allow_translations] = p.is_reflexive(
                        allow_translations=False
                    )

        # return
        return self._is_reflexive[allow_translations]

    # symmetries
    # ==========
    def automorphisms(
        self,
        square_to_one: bool = False,
        action: str = "right",
        as_dictionary: bool = False,
    ) -> "np.ndarray | dict":
        r"""Return the $GL(d,\mathbb{Z})$ matrices that leave the polytope invariant.

        These are the integer matrices with determinant $\pm 1$ (i.e. elements
        of $SL^{\pm}(d,\mathbb{Z})$) that map the polytope to itself. By
        default, they act on the points (as row vectors) by multiplication on
        the right. The results are cached.

        Args:
            square_to_one (bool): Whether to restrict to the matrices that
                square to the identity.
            action (str): Whether the returned matrices act on the `"left"` or
                on the `"right"`. The matrices that act on the left are the
                transposes of the ones that act on the right. This option is
                ignored when `as_dictionary` is true.
            as_dictionary (bool): Whether to return each automorphism as a
                dictionary that describes its action on the indices of the
                points. For an automorphism matrix `a` (acting on the right),
                the dictionary `d` satisfies
                `points()[d[i]] @ a == points()[i]` for every index `i`, i.e.
                it maps each index to the index of its preimage.

        Returns:
            np.ndarray | list[dict[int, int]]: The automorphism matrices, as an
                array of shape `(n_autos, d, d)`, or a list of dictionaries if
                `as_dictionary` is true. In both cases the automorphisms are
                given in the same order.

        Raises:
            NotImplementedError: If the polytope is not full-dimensional.
            ValueError: If `action` is not `"left"` or `"right"`.

        Examples:
            We construct a polytope, and find its automorphisms. We also check
            that one of the non-trivial automorphisms is indeed an automorphism
            by checking that it acts as a permutation on the vertices.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> autos = p.automorphisms()
            >>> a = autos[1]
            >>> a
            array([[ 1,  0,  0,  0],
                   [-1, -1, -6, -9],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])
            >>> p.vertices()
            array([[-1, -1, -6, -9],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [ 0,  1,  0,  0],
                   [ 1,  0,  0,  0]])
            >>> p.vertices() @ a
            array([[ 0,  1,  0,  0],
                   [ 0,  0,  0,  1],
                   [ 0,  0,  1,  0],
                   [-1, -1, -6, -9],
                   [ 1,  0,  0,  0]])

            We can restrict to automorphisms that square to the identity.

            >>> autos2 = p.automorphisms(square_to_one=True)
            >>> a2 = autos2[1]
            >>> a2 @ a2
            array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

            We can also get matrices that act on the left, which are simply the
            transposes, so that the points are now columns.

            >>> autos_left = p.automorphisms(square_to_one=True, action="left")
            >>> autos_left[1] @ p.vertices().T
            array([[ 0,  0,  0, -1,  1],
                   [ 1,  0,  0, -1,  0],
                   [ 0,  0,  1, -6,  0],
                   [ 0,  1,  0, -9,  0]])

            Finally, we get dictionaries that describe how the indices of the
            points transform.

            >>> autos_dict = p.automorphisms(as_dictionary=True)
            >>> autos_dict[1]
            {0: 0, 1: 4, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        """
        # check that this is sensical
        if self.dim() != self.ambient_dim():
            raise NotImplementedError(
                "Automorphisms can only be computed "
                + "for full-dimensional polytopes."
            )

        if action not in ("right", "left"):
            raise ValueError('Options for action are "right" or "left".')

        # check if we know the answer
        args_id = 1 * square_to_one + 2 * as_dictionary
        if self._autos[args_id] is not None:
            if as_dictionary:
                return copy.deepcopy(self._autos[args_id])
            elif action == "left":
                return np.array([a.T for a in self._autos[args_id]])
            else:
                return np.array(self._autos[args_id])

        # calculate the answer
        if self._autos[0] is None:
            vert_set = {tuple(pt) for pt in self.vertices()}

            # get the facet with minimum number of vertices
            f_min = min(self.facets(), key=lambda f: len(f.vertices()))
            f_min_vert_rref = np.array(
                fmpz_mat(f_min.vertices().T.tolist()).hnf().tolist(), dtype=int
            )

            pivots = []
            for v in f_min_vert_rref:
                if not any(v):
                    continue

                for i, ii in enumerate(v):
                    if ii != 0:
                        pivots.append(i)
                        break

            basis = [f_min.vertices()[i].tolist() for i in pivots]
            basis_inverse = fmpz_mat(basis).inv()
            images = []
            for f in self.facets():
                if len(f_min.vertices()) == len(f.vertices()):
                    f_vert = [pt.tolist() for pt in f.vertices()]
                    images.extend(itertools.permutations(f_vert, r=int(self.dim())))
            autos = []
            autos2 = []
            for im in images:
                image = fmpz_mat(im)
                m = basis_inverse * image
                if not all(abs(c.q) == 1 for c in np.array(m.tolist()).flatten()):
                    continue
                m = np.array(
                    [
                        [
                            int(c.p) // int(c.q) for c in r
                        ]  # just in case c.q==-1 by some weird reason
                        for r in np.array(m.tolist())
                    ],
                    dtype=int,
                )
                if {tuple(pt) for pt in np.dot(self.vertices(), m)} != vert_set:
                    continue
                autos.append(m)
                if all((np.dot(m, m) == np.eye(self.dim(), dtype=int)).flatten()):
                    autos2.append(m)
            self._autos[0] = np.array(autos)
            self._autos[1] = np.array(autos2)
        if as_dictionary and self._autos[2] is None:
            autos_dict = []
            autos2_dict = []
            pts_tup = [tuple(pt) for pt in self.points()]

            def _auto_dict(a):
                # index the image points by coordinate, so that the lookup
                # below is O(n) rather than O(n^2)
                new_inds = {}
                for j, pt in enumerate(self.points().dot(a)):
                    new_inds.setdefault(tuple(pt), j)
                return {i: new_inds[ii] for i, ii in enumerate(pts_tup)}

            for a in self._autos[0]:
                autos_dict.append(_auto_dict(a))
            for a in self._autos[1]:
                autos2_dict.append(_auto_dict(a))
            self._autos[2] = autos_dict
            self._autos[3] = autos2_dict

        # return
        if as_dictionary:
            return copy.deepcopy(self._autos[args_id])
        elif action == "left":
            return np.array([a.T for a in self._autos[args_id]])
        else:
            return np.array(self._autos[args_id])

    def inequivalent_Z2_actions(self) -> "np.ndarray":
        r"""Enumerate the inequivalent toric $\mathbb{Z}_2$ actions.

        A toric $\mathbb{Z}_2$ action is specified by a half-integral point
        $q/2$, where $q \in \{0,1\}^d$ is non-zero. Two such actions are
        considered equivalent if their vectors $q$ are related (modulo 2) by a
        lattice automorphism of the polytope (see
        [`automorphisms`][cytools.polytope.Polytope.automorphisms]). The result
        is cached.

        Returns:
            np.ndarray: The inequivalent representatives $q$, one per row,
                sorted lexicographically. Each one is the lexicographically
                smallest vector in its equivalence class.

        Raises:
            NotImplementedError: If the polytope is not full-dimensional.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p.inequivalent_Z2_actions()
            array([[0, 0, 0, 1],
                   [0, 0, 1, 1]])
        """
        if self._Z2_actions is not None:
            return copy.deepcopy(self._Z2_actions)

        autos = self.automorphisms(action="left")
        d = self.ambient_dim()

        # all q in {0,1}^d except the trivial q = 0
        candidates = [q for q in itertools.product((0, 1), repeat=d) if any(q)]

        # group candidates into orbits under q -> (s @ q) mod 2, keeping one
        # canonical (lexicographically smallest) representative per orbit
        seen, reps = set(), []
        for q in candidates:
            if q in seen:
                continue
            orbit = {tuple(np.mod(s @ q, 2)) for s in autos}
            seen |= orbit
            reps.append(min(orbit))

        self._Z2_actions = np.array(sorted(reps), dtype=int)
        return copy.deepcopy(self._Z2_actions)

    def normal_form(
        self, affine_transform: bool = False, backend: str = "palp"
    ) -> np.ndarray:
        r"""Return the normal form of the polytope as defined by Kreuzer-Skarke.

        The normal form is a canonical choice of the vertices such that two
        polytopes have the same normal form if and only if they are related by
        a $GL(d,\mathbb{Z})$ transformation (or also a translation, if
        `affine_transform` is true). The results are cached.

        Args:
            affine_transform (bool): Whether to also allow translations, in
                addition to $GL(d,\mathbb{Z})$ transformations.
            backend (str): The backend to use. The options are `"native"`,
                which uses native Python code, or `"palp"`, which uses PALP
                for the computation. The native algorithm and PALP use
                different conventions for affine normal forms, and PALP
                generally works better. PALP only supports full-dimensional
                polytopes, so for other polytopes a warning is issued and the
                native backend is used instead.

        Returns:
            np.ndarray: The vertices in normal form, one per row.

        Raises:
            NotImplementedError: If the polytope is empty.
            ValueError: If `backend` is not `"native"` or `"palp"`.

        Examples:
            We construct a polytope, and find its linear and affine normal
            forms.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.normal_form()
            array([[ 1,  0,  0,  0],
                   [ 0,  1,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1],
                   [-9, -6, -1, -1]])
            >>> p.normal_form(affine_transform=True)
            array([[ 1,  0,  0,  0],
                   [ 0,  1,  0,  0],
                   [ 0,  0,  1,  0],
                   [12, 17, 17, 18],
                   [ 0,  0,  0,  0]])
        """
        if self.dim() < 0:
            raise NotImplementedError

        # This function is based on code by Andrey Novoseltsev, Samuel Gonshaw,
        # Jan Keitel, and others, and is redistributed under the GNU General
        # Public License version 2+.
        #
        # The original code can be found at:
        # https://github.com/sagemath/sage/blob/develop/src/sage/geometry/lattice_polytope.py
        # https://trac.sagemath.org/ticket/13525

        # process backend
        if backend not in ("native", "palp"):
            raise ValueError('Error: options for backend are "native" and ' '"palp".')

        if backend == "palp":
            if not self.is_solid():
                warnings.warn(
                    "PALP doesn't support polytopes that are not "
                    "full-dimensional. Using native backend."
                )
                backend = "native"

        # check if we know the answer
        args_id = 1 * affine_transform + affine_transform * (backend == "native") * 1
        if self._normal_form[args_id] is not None:
            return np.array(self._normal_form[args_id])

        # PALP backend
        # ------------
        if backend == "palp":
            p = pypalp.Polytope(self.points(optimal=True))
            points = p.normal_form(affine=affine_transform)

            # cache it and return
            self._normal_form[args_id] = points
            return np.array(self._normal_form[args_id])

        # native backend
        # --------------
        # Define function that constructs permutation matrices
        def PGE(n, u, v):
            tmp_m = np.eye(n, dtype=int)
            if u == v:
                return tmp_m
            tmp_m[u - 1, u - 1] = 0
            tmp_m[v - 1, v - 1] = 0
            tmp_m[u - 1, v - 1] = 1
            tmp_m[v - 1, u - 1] = 1
            return tmp_m

        V = self.vertices()
        n_v = len(V)
        n_f = len(self._ineqs_input)
        PM = np.array([n[:-1].dot(V.T) + n[-1] for n in self._ineqs_input])
        n_s = 1
        prm = {0: [np.eye(n_f, dtype=int), np.eye(n_v, dtype=int)]}
        for j in range(n_v):
            m = np.argmax([PM[0, prm[0][1].dot(range(n_v))][i] for i in range(j, n_v)])
            if m > 0:
                prm[0][1] = PGE(n_v, j + 1, m + j + 1).dot(prm[0][1])
        first_row = list(PM[0])
        # Arrange other rows one by one and compare with first row
        for k in range(1, n_f):
            # Error for k == 1 already!
            prm[n_s] = [np.eye(n_f, dtype=int), np.eye(n_v, dtype=int)]
            m = np.argmax(PM[:, prm[n_s][1].dot(range(n_v))][k])
            if m > 0:
                prm[n_s][1] = PGE(n_v, 1, m + 1).dot(prm[n_s][1])
            d = PM[k, prm[n_s][1].dot(range(n_v))][0] - prm[0][1].dot(first_row)[0]
            if d < 0:
                # The largest elt of this row is smaller than largest elt
                # in 1st row, so nothing to do
                continue
            # otherwise:
            for i in range(1, n_v):
                m = np.argmax(
                    [PM[k, prm[n_s][1].dot(range(n_v))][j] for j in range(i, n_v)]
                )
                if m > 0:
                    prm[n_s][1] = PGE(n_v, i + 1, m + i + 1).dot(prm[n_s][1])
                if d == 0:
                    d = (
                        PM[k, prm[n_s][1].dot(range(n_v))][i]
                        - prm[0][1].dot(first_row)[i]
                    )
                    if d < 0:
                        break
            if d < 0:
                # This row is smaller than 1st row, so nothing to do
                del prm[n_s]
                continue
            prm[n_s][0] = PGE(n_f, 1, k + 1).dot(prm[n_s][0])
            if d == 0:
                # This row is the same, so we have a symmetry!
                n_s += 1
            else:
                # This row is larger, so it becomes the first row and
                # the symmetries reset.
                first_row = list(PM[k])
                prm = {0: prm[n_s]}
                n_s = 1
        prm = {k: prm[k] for k in prm if k < n_s}
        b = PM[prm[0][0].dot(range(n_f)), :][:, prm[0][1].dot(range(n_v))][0]
        # Work out the restrictions the current permutations
        # place on other permutations as a automorphisms
        # of the first row
        # The array is such that:
        # S = [i, 1, ..., 1 (ith), j, i+1, ..., i+1 (jth), k ... ]
        # describes the "symmetry blocks"
        S = list(range(1, n_v + 1))
        for i in range(1, n_v):
            if b[i - 1] == b[i]:
                S[i] = S[i - 1]
                S[S[i] - 1] += 1
            else:
                S[i] = i + 1
        # We determine the other rows of PM_max in turn by use of perms and
        # aut on previous rows.
        for row in range(1, n_f - 1):
            n_s = len(prm)
            n_s_bar = n_s
            cf = 0
            l_r = [0] * n_v
            # Search for possible local permutations based off previous
            # global permutations.
            for k in range(n_s_bar - 1, -1, -1):
                # number of local permutations associated with current global
                n_p = 0
                ccf = cf
                prmb = {0: copy.copy(prm[k])}
                # We look for the line with the maximal entry in the first
                # subsymmetry block, i.e. we are allowed to swap elements
                # between 0 and S(0)
                for s in range(row, n_f):
                    for j in range(1, S[0]):
                        v = PM[prmb[n_p][0].dot(range(n_f)), :][
                            :, prmb[n_p][1].dot(range(n_v))
                        ][s]
                        if v[0] < v[j]:
                            prmb[n_p][1] = PGE(n_v, 1, j + 1).dot(prmb[n_p][1])
                    if ccf == 0:
                        l_r[0] = PM[prmb[n_p][0].dot(range(n_f)), :][
                            :, prmb[n_p][1].dot(range(n_v))
                        ][s, 0]
                        prmb[n_p][0] = PGE(n_f, row + 1, s + 1).dot(prmb[n_p][0])
                        n_p += 1
                        ccf = 1
                        prmb[n_p] = copy.copy(prm[k])
                    else:
                        d1 = PM[prmb[n_p][0].dot(range(n_f)), :][
                            :, prmb[n_p][1].dot(range(n_v))
                        ][s, 0]
                        d = d1 - l_r[0]
                        if d < 0:
                            # We move to the next line
                            continue
                        if d == 0:
                            # Maximal values agree, so possible symmetry
                            prmb[n_p][0] = PGE(n_f, row + 1, s + 1).dot(prmb[n_p][0])
                            n_p += 1
                            prmb[n_p] = copy.copy(prm[k])
                        else:
                            # We found a greater maximal value for first entry.
                            # It becomes our new reference:
                            l_r[0] = d1
                            prmb[n_p][0] = PGE(n_f, row + 1, s + 1).dot(prmb[n_p][0])
                            # Forget previous work done
                            cf = 0
                            prmb = {0: copy.copy(prmb[n_p])}
                            n_p = 1
                            prmb[n_p] = copy.copy(prm[k])
                            n_s = k + 1
                # Check if the permutations found just now work
                # with other elements
                for c in range(1, n_v):
                    h = S[c]
                    ccf = cf
                    # Now let us find out where the end of the
                    # next symmetry block is:
                    if h < c + 1:
                        h = S[h - 1]
                    s = n_p
                    # Check through this block for each possible permutation
                    while s > 0:
                        s -= 1
                        # Find the largest value in this symmetry block
                        for j in range(c + 1, h):
                            v = PM[prmb[s][0].dot(range(n_f)), :][
                                :, prmb[s][1].dot(range(n_v))
                            ][row]
                            if v[c] < v[j]:
                                prmb[s][1] = PGE(n_v, c + 1, j + 1).dot(prmb[s][1])
                        if ccf == 0:
                            # Set reference and carry on to next permutation
                            l_r[c] = PM[prmb[s][0].dot(range(n_f)), :][
                                :, prmb[s][1].dot(range(n_v))
                            ][row, c]
                            ccf = 1
                        else:
                            d1 = PM[prmb[s][0].dot(range(n_f)), :][
                                :, prmb[s][1].dot(range(n_v))
                            ][row, c]
                            d = d1 - l_r[c]
                            if d < 0:
                                n_p -= 1
                                if s < n_p:
                                    prmb[s] = copy.copy(prmb[n_p])
                            elif d > 0:
                                # The current case leads to a smaller matrix,
                                # hence this case becomes our new reference
                                l_r[c] = d1
                                cf = 0
                                n_p = s + 1
                                n_s = k + 1
                # Update permutations
                if n_s - 1 > k:
                    prm[k] = copy.copy(prm[n_s - 1])
                n_s -= 1
                for s in range(n_p):
                    prm[n_s] = copy.copy(prmb[s])
                    n_s += 1
                cf = n_s
            prm = {k: prm[k] for k in prm if k < n_s}
            # If the automorphisms are not already completely restricted,
            # update them
            if S != list(range(1, n_v + 1)):
                # Take the old automorphisms and update by
                # the restrictions the last worked out
                # row imposes.
                c = 0
                M = PM[prm[0][0].dot(range(n_f)), :][:, prm[0][1].dot(range(n_v))][row]
                while c < n_v:
                    s = S[c] + 1
                    S[c] = c + 1
                    c += 1
                    while c < s - 1:
                        if M[c] == M[c - 1]:
                            S[c] = S[c - 1]
                            S[S[c] - 1] += 1
                        else:
                            S[c] = c + 1
                        c += 1
        # Now we have the perms, we construct PM_max using one of them
        PM_max = PM[prm[0][0].dot(range(n_f)), :][:, prm[0][1].dot(range(n_v))]
        # Perform a translation if necessary
        if affine_transform:
            v0 = copy.copy(V[0])
            for i in range(n_v):
                V[i] -= v0
        # Finally arrange the points the the canonical order
        p_c = np.eye(n_v, dtype=int)
        M_max = [max([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        S_max = [sum([PM_max[i][j] for i in range(n_f)]) for j in range(n_v)]
        for i in range(n_v):
            k = i
            for j in range(i + 1, n_v):
                if M_max[j] < M_max[k] or (
                    M_max[j] == M_max[k] and S_max[j] < S_max[k]
                ):
                    k = j
            if not k == i:
                M_max[i], M_max[k] = M_max[k], M_max[i]
                S_max[i], S_max[k] = S_max[k], S_max[i]
                p_c = PGE(n_v, 1 + i, 1 + k).dot(p_c)
        # Create array of possible NFs.
        prm = [p_c.dot(perm[1]) for perm in prm.values()]
        Vs = [
            np.array(
                fmpz_mat(V.T[:, sig.dot(range(n_v))].tolist()).hnf().tolist(),
                dtype=int,
            ).tolist()
            for sig in prm
        ]
        Vmin = min(Vs)
        if affine_transform:
            self._normal_form[args_id] = np.array(Vmin).T[:, : self.dim()]
        else:
            self._normal_form[args_id] = np.array(Vmin).T
        return np.array(self._normal_form[args_id])

    def is_linearly_equivalent(self, other: "Polytope", backend: str = "palp") -> bool:
        r"""Return whether the polytopes are related by a $GL(d,\mathbb{Z})$ map.

        Two polytopes are linearly equivalent if they can be transformed into
        each other by a $GL(d,\mathbb{Z})$ (i.e. $SL^{\pm}(d,\mathbb{Z})$)
        transformation. This is checked by comparing their normal forms (see
        [`normal_form`][cytools.polytope.Polytope.normal_form]).

        Args:
            other (Polytope): The other polytope being compared.
            backend (str): The backend used to compute the normal form. The
                options are `"native"`, which uses native Python code, or
                `"palp"`, which uses PALP for the computation.

        Returns:
            bool: `True` if the polytopes are linearly equivalent, `False`
                otherwise.

        Examples:
            We construct two polytopes and check if they are linearly
            equivalent.

            >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p2 = Polytope([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[1,1,1,1]])
            >>> p1.is_linearly_equivalent(p2)
            True
        """
        our_normal_form = self.normal_form(
            affine_transform=False, backend=backend
        ).tolist()
        other_normal_form = other.normal_form(
            affine_transform=False, backend=backend
        ).tolist()

        return our_normal_form == other_normal_form

    def is_affinely_equivalent(self, other: "Polytope", backend: str = "palp") -> bool:
        r"""Return whether the polytopes are related by an integral affine map.

        Two polytopes are affinely equivalent if they can be transformed into
        each other by a $GL(d,\mathbb{Z})$ transformation followed by an
        integral translation. This is checked by comparing their affine normal
        forms (see [`normal_form`][cytools.polytope.Polytope.normal_form]).

        Args:
            other (Polytope): The other polytope being compared.
            backend (str): The backend used to compute the normal form. The
                options are `"native"`, which uses native Python code, or
                `"palp"`, which uses PALP for the computation.

        Returns:
            bool: `True` if the polytopes are affinely equivalent, `False`
                otherwise.

        Examples:
            We construct two polytopes and check if they are affinely
            equivalent.

            >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p2 = Polytope([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,2],[-1,-1,-1,0]])
            >>> p1.is_affinely_equivalent(p2)
            True
        """
        our_normal_form = self.normal_form(
            affine_transform=True, backend=backend
        ).tolist()
        other_normal_form = other.normal_form(
            affine_transform=True, backend=backend
        ).tolist()

        return our_normal_form == other_normal_form

    # triangulating
    # =============
    def _triang_labels(
        self, include_points_interior_to_facets: bool = None
    ) -> tuple[int]:
        """Return the labels of the points to be used in a triangulation.

        Typically this function should not be called by the user. Instead, it
        is called by the triangulation functions of this class.

        Args:
            include_points_interior_to_facets (bool | None): Whether to include
                points interior to facets in the triangulation. If not
                specified, it is set to `False` for reflexive polytopes and
                `True` otherwise.

        Returns:
            tuple: The labels of the points that will be included in a
                triangulation.

        Examples:
            For a reflexive polytope, points interior to facets are excluded
            by default.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p._triang_labels()
            (0, 1, 2, 3, 4, 5, 6)
            >>> p._triang_labels(include_points_interior_to_facets=True)
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        """
        if include_points_interior_to_facets is None:
            include_points_interior_to_facets = not self.is_reflexive()

        if include_points_interior_to_facets:
            return self.labels
        else:
            return self.labels_not_facet

    def triangulate(
        self, *, # enforce all arguments are keyword
        include_points_interior_to_facets: bool = None,
        points: "ArrayLike" = None,
        make_star: bool = None,
        simplices: "ArrayLike" = None,
        check_input_simplices: bool = True,
        heights: "ArrayLike" = None,
        check_heights: bool = True,
        backend: str = "cgal",
        verbosity: int = 1,
        seed: int = None,
    ) -> Triangulation:
        """Return a single regular triangulation of the polytope.

        When reflexive polytopes are used, it defaults to returning a fine,
        regular, star triangulation. The points of the triangulation are
        ordered by their labels. All arguments are keyword-only.

        Args:
            include_points_interior_to_facets (bool | None): Whether to include
                points interior to facets in the triangulation. If not
                specified, it is set to `False` for reflexive polytopes and
                `True` otherwise.
            points (ArrayLike | None): The labels of the points to be used.
                If specified, `include_points_interior_to_facets` is ignored.
            make_star (bool | None): Whether to turn the triangulation into a
                star triangulation with respect to the origin, either by
                deleting internal lines and connecting all points to the
                origin, or equivalently by decreasing the height of the origin
                to be much lower than the rest. If not specified, it is set to
                `True` if the polytope is reflexive (without translations) and
                neither `heights` nor `simplices` are given, and to `False`
                otherwise. It is always set to `False` if the polytope is not
                reflexive and the origin is not one of the points.
            simplices (ArrayLike | None): The simplices specifying the
                triangulation, each given as a list of point labels. This is
                useful when a triangulation was previously computed and needs
                to be used again.
            check_input_simplices (bool): Whether to check that the input
                simplices define a valid triangulation.
            heights (ArrayLike | None): The heights specifying the regular
                triangulation. They can be given either for all the points of
                the polytope, in which case only the heights of the points used
                in the triangulation are kept, or for the points used in the
                triangulation, ordered by their labels. Heights can only be
                specified when using CGAL or QHull as the backend. If not
                specified, the triangulation depends on the backend: CGAL
                constructs the Delaunay triangulation, QHull uses random
                heights near the Delaunay heights, and TOPCOM constructs a
                placing triangulation.
            check_heights (bool): Whether to check that the input or default
                heights define a unique triangulation.
            backend (str): The backend used to compute the triangulation. The
                options are `"cgal"`, `"qhull"`, and `"topcom"`. CGAL is the
                default as it is very fast and robust.
            verbosity (int): The verbosity level.
            seed (int | None): Seed for the height perturbation used by the
                QHull backend. Only that backend perturbs its heights, so this
                has no effect for CGAL or TOPCOM. If not specified, the
                perturbation is drawn afresh each time and the triangulation is
                not reproducible.

        Returns:
            Triangulation: A
                [`Triangulation`][cytools.triangulation.Triangulation] object
                describing a triangulation of the polytope.

        Raises:
            ValueError: If the input simplices use more points than the ones
                selected for the triangulation. Further errors may be raised
                by the [`Triangulation`][cytools.triangulation.Triangulation]
                constructor.

        Examples:
            We construct a triangulation of a reflexive polytope and check that
            by default it is a fine, regular, star triangulation.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.is_fine(), t.is_regular(), t.is_star()
            (True, True, True)
            >>> len(t.points())
            7

            By default the points interior to facets are not included, but we
            can include them, or specify the points to use.

            >>> len(p.triangulate(include_points_interior_to_facets=True).points())
            10
            >>> p.triangulate(points=[1,2,3,4,5]).is_star()
            False

            We also construct triangulations with heights, input simplices,
            and using the other backends.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [-1,-2,-1,-1],[-2,-1,-1,-1]])
            >>> t = p.triangulate(heights=[-30,5,5,24,-19,-14,29])
            >>> t.simplices()
            array([[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 5],
                   [0, 1, 2, 4, 6],
                   [0, 1, 2, 5, 6],
                   [0, 1, 3, 4, 5],
                   [0, 1, 4, 5, 6],
                   [0, 2, 3, 4, 5],
                   [0, 2, 4, 5, 6]], dtype=uint8)
            >>> t2 = p.triangulate(simplices=t.simplices())
            >>> t2 == t
            True
            >>> t3 = p.triangulate(backend="qhull")
            >>> t4 = p.triangulate(backend="topcom")
        """
        # set include_points_interior_to_facets
        if include_points_interior_to_facets is None:
            use_pts_in_facets = not self.is_reflexive()
        else:
            use_pts_in_facets = include_points_interior_to_facets

        # get indices of relevant points
        if points is not None:
            points = tuple(sorted(set(points)))
        else:
            points = self._triang_labels(use_pts_in_facets)
            points = sorted(points)

        # if simplices are provided, check if they span the relevant points
        if simplices is not None:
            simps_inds = tuple(sorted({i for simp in simplices for i in simp}))

            # index mismatch... Raise error
            if len(simps_inds) > len(points):
                error_msg = (
                    f"Simplices spanned {simps_inds}, which is "
                    + f"longer than length of points, {points}. "
                    + "Check include_points_interior_to_facets... it "
                    + f"was set to {include_points_interior_to_facets}"
                )

                if include_points_interior_to_facets is None:
                    error_msg += (
                        f" and then to {use_pts_in_facets} b/c "
                        + "the polytope is "
                        + ("" if self.is_reflexive() else "not ")
                        + "reflexive."
                    )
                raise ValueError(error_msg)

        # if heights are provided for all points, trim them
        if (heights is not None) and (len(heights) == len(self.labels)):
            pts_inds = self.points(which=points, as_indices=True)
            triang_heights = np.array(heights)[list(pts_inds)]
        else:
            triang_heights = heights

        # set make_star
        if make_star is None:
            if heights is None and simplices is None:
                make_star = self.is_reflexive(allow_translations=False)
            else:
                make_star = False

        if (not self.is_reflexive(allow_translations=False)) and (self._label_origin not in points):
            make_star = False

        # return triangulation
        return Triangulation(
            self,
            points,
            make_star=make_star,
            simplices=simplices,
            check_input_simplices=check_input_simplices,
            heights=triang_heights,
            check_heights=check_heights,
            backend=backend,
            verbosity=verbosity,
            seed=seed,
        )

    def random_triangulations_fast(
        self,
        N: int = None,
        c: float = 0.2,
        max_retries: int = 500,
        make_star: bool = True,
        only_fine: bool = True,
        include_points_interior_to_facets: bool = None,
        points: ArrayLike = None,
        backend: str = "cgal",
        as_list: bool = False,
        progress_bar: bool = True,
        seed: int = None,
    ) -> "Iterator | list":
        """Construct pseudorandom regular triangulations of the polytope.

        The triangulations are regular, and optionally fine and star. They are
        obtained by picking random heights around the Delaunay heights from a
        Gaussian distribution. Only distinct triangulations are returned.

        Warning:
            This function produces random triangulations very quickly, but it
            does not produce a fair sample. When a fair sampling is required,
            [`random_triangulations_fair`]
            [cytools.polytope.Polytope.random_triangulations_fair]
            should be used.

        Args:
            N (int | None): The number of desired distinct triangulations. If
                not specified, it will generate as many triangulations as it
                can find until it fails to obtain a new triangulation
                `max_retries` consecutive times. This parameter is required when
                `as_list` is true.
            c (float): The standard deviation of the Gaussian distribution used
                to pick the heights. A larger `c` results in a wider range of
                possible triangulations, but with a larger fraction of them
                being non-fine, which slows down the process when `only_fine`
                is true.
            max_retries (int): The maximum number of consecutive failed
                attempts to obtain a new triangulation before the process is
                terminated.
            make_star (bool | None): Whether to convert the obtained
                triangulations into star triangulations. If set to `None`, it
                is set to `True` for reflexive polytopes (without translations)
                and `False` otherwise. It is always set to `False` if the
                origin is not one of the points.
            only_fine (bool): Whether to restrict to fine triangulations.
            include_points_interior_to_facets (bool | None): Whether to include
                points interior to facets in the triangulation. If not
                specified, it is set to `False` for reflexive polytopes and
                `True` otherwise.
            points (ArrayLike | None): The labels of the points to be used.
                If specified, `include_points_interior_to_facets` is ignored.
            backend (str): The backend used to compute the triangulations. The
                options are `"cgal"` and `"qhull"`.
            as_list (bool): Whether to return the full list of triangulations
                at once instead of a generator. By default a generator is
                returned, which is usually preferred for efficiency.
            progress_bar (bool): Whether to show a progress bar with the number
                of triangulations obtained. This option only has an effect when
                `as_list` is true.
            seed (int | None): A seed for the random number generator. This can
                be used to obtain reproducible results.

        Returns:
            Iterator[Triangulation] | list[Triangulation]: A generator of
                [`Triangulation`][cytools.triangulation.Triangulation] objects,
                or a list of them if `as_list` is true.

        Raises:
            ValueError: If `as_list` is true but `N` is not specified.

        Examples:
            We construct a polytope and find some random triangulations. The
            triangulations are obtained very quickly, but they are not a fair
            sample of the space of triangulations. For a fair sample,
            [`random_triangulations_fair`]
            [cytools.polytope.Polytope.random_triangulations_fair]
            should be used.

            >>> p = Polytope(
            ...     [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]
            ... ).dual()
            >>> g = p.random_triangulations_fast()
            >>> next(g)  # runs very quickly
            A fine, regular, star triangulation of a 4-dimensional point...
            >>> next(g)  # keeps producing triangulations until it can't find more
            A fine, regular, star triangulation of a 4-dimensional point...

            We can also get a list of 10 triangulations, which is produced very
            quickly.

            >>> rand_triangs = p.random_triangulations_fast(N=10, as_list=True)
        """
        # if self.ambient_dim() > self.dim():
        #    raise NotImplementedError("Only triangulations of "
        #                              "full-dimensional polytopes are "
        #                              "supported.")
        if (N is None) and as_list:
            raise ValueError(
                "Number of triangulations must be specified when " "returning a list."
            )

        if points is not None:
            points = tuple(sorted(set(points)))
        else:
            points = self._triang_labels(include_points_interior_to_facets)

        if make_star is None:
            make_star = self.is_reflexive(allow_translations=False)
        if self._label_origin not in points:
            make_star = False
        g = random_triangulations_fast_generator(
            self,
            points,
            N=N,
            c=c,
            max_retries=max_retries,
            make_star=make_star,
            only_fine=only_fine,
            backend=backend,
            seed=seed,
        )
        if not as_list:
            return g
        if progress_bar:
            pbar = tqdm(total=N)
        triangs_list = []
        while len(triangs_list) < N:
            try:
                triangs_list.append(next(g))
                if progress_bar:
                    pbar.update(len(triangs_list) - pbar.n)
            except StopIteration:
                if progress_bar:
                    pbar.update(N - pbar.n)
                break
        return triangs_list

    def random_triangulations_fair(
        self,
        N: int = None,
        n_walk: int = None,
        n_flip: int = None,
        initial_walk_steps: int = None,
        walk_step_size: float = 1e-2,
        max_steps_to_wall: int = 25,
        fine_tune_steps: int = 8,
        max_retries: int = 50,
        make_star: bool = None,
        include_points_interior_to_facets: bool = None,
        points: ArrayLike = None,
        backend: str = "cgal",
        as_list: bool = False,
        progress_bar: bool = True,
        seed: int = None,
    ) -> "Iterator | list":
        r"""Construct pseudorandom fine, regular triangulations of the polytope.

        The triangulations are fine and regular, and optionally star. This
        function implements Algorithm \#3 from the paper
        *Bounding the Kreuzer-Skarke Landscape* by Mehmet Demirtas, Liam
        McAllister, and Andres Rios-Tascon
        ([arXiv:2008.01730](https://arxiv.org/abs/2008.01730)).

        This is a Markov chain Monte Carlo algorithm that involves taking
        random walks inside the subset of the secondary fan corresponding to
        fine triangulations and performing random flips. For details, please
        see Section 4.1 in the paper. Only distinct triangulations are
        returned.

        Note:
            - By default this function tries to guess reasonable parameters for
              the polytope that is being used. However, it may take some
              trial-and-error to find optimal parameters that produce a good
              sampling and prevent the algorithm from getting stuck.
            - This function is designed mainly for large polytopes where
              sampling triangulations is challenging. When small polytopes are
              used it is likely to get stuck or not produce an optimal set.
            - The time it takes to obtain each triangulation varies very
              significantly with the parameters used, so it is best to adjust
              them to the desired balance between speed and fairness of the
              sampling.

        Args:
            N (int | None): The number of desired distinct triangulations. If
                not specified, it will generate as many triangulations as it
                can find until it fails to obtain a new triangulation
                `max_retries` consecutive times. This parameter is required when
                `as_list` is true.
            n_walk (int | None): The number of hit-and-run steps per
                triangulation. If not specified, it is set to
                `n_points//10 + 10`, where `n_points` is the total number of
                lattice points of the polytope.
            n_flip (int | None): The number of random flips performed per
                triangulation. If not specified, it is set to
                `n_points//10 + 10`.
            initial_walk_steps (int | None): The number of hit-and-run steps to
                take before starting to record triangulations. Small values may
                result in a bias towards Delaunay-like triangulations. If not
                specified, it is set to `2*n_points//10 + 10`.
            walk_step_size (float): The size of the random steps taken in the
                secondary fan. The algorithm may stall if it is too small.
            max_steps_to_wall (int): The maximum number of steps to take
                towards a wall of the subset of the secondary fan that
                corresponds to fine triangulations. If a wall is not found, a
                new random direction is selected. Setting this to be very large
                (>100) reduces performance. If it is too low, the algorithm may
                stall.
            fine_tune_steps (int): The number of steps used to determine the
                location of a wall. Decreasing it improves performance, but
                might result in biased samples.
            max_retries (int): The maximum number of consecutive failed
                attempts to obtain a new triangulation before the process is
                terminated. It is also the maximum number of random directions
                tried when searching for a wall.
            make_star (bool | None): Whether to convert the obtained
                triangulations into star triangulations. If not specified, it
                is set to `True` for reflexive polytopes (without translations)
                and `False` otherwise. It is always set to `False` if the
                origin is not one of the points.
            include_points_interior_to_facets (bool | None): Whether to include
                points interior to facets in the triangulation. If not
                specified, it is set to `False` for reflexive polytopes and
                `True` otherwise.
            points (ArrayLike | None): The labels of the points to be used.
                If specified, `include_points_interior_to_facets` is ignored.
            backend (str): The backend used to compute the triangulations. The
                options are `"cgal"` and `"qhull"`.
            as_list (bool): Whether to return the full list of triangulations
                at once instead of a generator. By default a generator is
                returned, which is usually preferred for efficiency.
            progress_bar (bool): Whether to show a progress bar with the number
                of triangulations obtained. This option only has an effect when
                `as_list` is true.
            seed (int | None): A seed for the random number generator. This can
                be used to obtain reproducible results.

        Returns:
            Iterator[Triangulation] | list[Triangulation]: A generator of
                [`Triangulation`][cytools.triangulation.Triangulation] objects,
                or a list of them if `as_list` is true.

        Raises:
            NotImplementedError: If the polytope is not full-dimensional.
            ValueError: If `as_list` is true but `N` is not specified.

        Examples:
            We construct a polytope and find some random triangulations. The
            computation takes considerable time, but they should be a fair
            sample from the full set of triangulations (if the parameters are
            chosen correctly). For (some) machine learning purposes or when the
            fairness of the sample is not crucial,
            [`random_triangulations_fast`]
            [cytools.polytope.Polytope.random_triangulations_fast] should be
            used instead.

            >>> p = Polytope(
            ...     [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]
            ... ).dual()
            >>> g = p.random_triangulations_fair()
            >>> next(g)  # takes a long time
            A fine, regular, star triangulation of a 4-dimensional point...
            >>> next(g)  # takes a similar amount of time
            A fine, regular, star triangulation of a 4-dimensional point...

            We can also get a list of 10 triangulations, which takes a while.

            >>> rand_triangs = p.random_triangulations_fair(N=10, as_list=True)
        """
        if self.ambient_dim() > self.dim():
            raise NotImplementedError(
                "Only triangulations of full-dimensional polytopes" "are supported."
            )
        if N is None and as_list:
            raise ValueError(
                "Number of triangulations must be specified when " "returning a list."
            )
        if points is not None:
            points = tuple(sorted(set(points)))
        else:
            points = self._triang_labels(include_points_interior_to_facets)
        if make_star is None:
            make_star = self.is_reflexive(allow_translations=False)
        if self._label_origin not in points:
            make_star = False
        if n_walk is None:
            n_walk = len(self.points()) // 10 + 10
        if n_flip is None:
            n_flip = len(self.points()) // 10 + 10
        if initial_walk_steps is None:
            initial_walk_steps = 2 * len(self.points()) // 10 + 10
        g = random_triangulations_fair_generator(
            self,
            points,
            N=N,
            n_walk=n_walk,
            n_flip=n_flip,
            initial_walk_steps=initial_walk_steps,
            walk_step_size=walk_step_size,
            max_steps_to_wall=max_steps_to_wall,
            fine_tune_steps=fine_tune_steps,
            max_retries=max_retries,
            make_star=make_star,
            backend=backend,
            seed=seed,
        )
        if not as_list:
            return g
        if progress_bar:
            pbar = tqdm(total=N)
        triangs_list = []
        while len(triangs_list) < N:
            try:
                triangs_list.append(next(g))
                if progress_bar:
                    pbar.update(len(triangs_list) - pbar.n)
            except StopIteration:
                if progress_bar:
                    pbar.update(N - pbar.n)
                break
        return triangs_list

    def random_triangulations_gnn(
        self,
        N: int,
        make_star: bool = True,
        max_npts: int = 0,
        N_face_triangs: int = 1000,
        as_heights: bool = False,
        as_generator: bool = False,
        seed: int = None,
        verbosity: int = 0,
    ) -> "list | Iterator":
        """Construct random NTFE FR(S)Ts of a reflexive 4-dimensional polytope.

        The fine, regular (and optionally star) triangulations (FR(S)Ts) are
        built by sampling fine, regular triangulations (FRTs) of the 2-faces
        with the dualGNN graph neural network
        ([arXiv:2605.27770](https://arxiv.org/abs/2605.27770)) and extending
        them via the NTFE algorithm
        ([arXiv:2309.10855](https://arxiv.org/abs/2309.10855)). Each attempt
        draws one FRT per 2-face uniformly from the sampled pools, and it is
        accepted if the combination extends to a triangulation of the full
        polytope (i.e. if its expanded secondary cone is solid). Only
        distinct NTFEs are returned.

        This function provides a convenient interface for this use case.
        When more control is needed (e.g. to provide precomputed face
        triangulations, possibly sampled with a custom GNN checkpoint),
        [`ntfe_frts`][cytools.ntfe.ntfe.ntfe_frts] with
        `triang_method="dualgnn"` can be used directly.

        This sampler is much closer to uniform over the set of FRSTs than
        [`random_triangulations_fast`]
        [cytools.polytope.Polytope.random_triangulations_fast], while being
        far faster than
        [`random_triangulations_fair`]
        [cytools.polytope.Polytope.random_triangulations_fair] at large
        $h^{1,1}$.

        Note:
            This function requires the optional `dualgnn` package (which
            depends on PyTorch). It can be installed with
            `pip install cytools[gnn]` or `pip install dualgnn`.

        Args:
            N (int): The number of NTFEs to sample. Since not every
                combination of 2-face FRTs glues into an NTFE (acceptance can
                be ~1% for polytopes with large 2-faces, e.g. the $h^{1,1}=86$
                benchmark of arXiv:2605.27770), draws are retried until `N`
                distinct NTFEs are found. Fewer are returned only if the
                attempt budget (`1000*N` draws) is exhausted or every
                combination has been checked.
            make_star (bool): Whether to convert the NTFE FRTs into FRSTs
                (i.e. to make them star).
            max_npts (int): The maximum number of points of 2-faces for which
                all FRTs are enumerated. The GNN samples the FRTs of 2-faces
                with more points. The default of 0 means that the GNN samples
                every 2-face, since enumerating the FRTs of a face can be slow
                even when only a handful of triangulations are wanted.
            N_face_triangs (int): The number of FRTs the GNN samples per
                (large) 2-face. Every sampled NTFE is assembled from these
                per-face pools, so the pool size sets the support of the
                sampler: small pools restrict (and so bias) which NTFEs can
                appear, while the cost of building the pools is proportional
                to their size. Lower it (e.g. to 50) for quick exploratory
                draws on polytopes with large 2-faces, and raise it when
                coverage of the NTFE space matters.
            as_heights (bool): Whether to return the height vectors realizing
                each NTFE instead of
                [`Triangulation`][cytools.triangulation.Triangulation]
                objects. This is cheaper when the `Triangulation` objects are
                not needed.
            as_generator (bool): Whether to return a generator instead of a
                list. Use generators if memory is a concern.
            seed (int | None): A seed for the random number generator. This
                can be used to obtain reproducible results.
            verbosity (int): The verbosity level. Higher means more verbose.

        Returns:
            list | Iterator: A list of
                [`Triangulation`][cytools.triangulation.Triangulation] objects
                (or of height vectors if `as_heights` is true), or a generator
                of them if `as_generator` is true.

        Raises:
            NotImplementedError: If the polytope is not a reflexive
                4-dimensional polytope.
            ImportError: If the optional `dualgnn` package is not installed.

        Examples:
            We construct a polytope and sample some random triangulations.
            This polytope has a single NTFE, so a single triangulation is
            returned regardless of `N`.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [0,0,0,-1],[-1,-1,-6,-9]])
            >>> p.random_triangulations_gnn(N=4, seed=0)
            [A fine, regular, star triangulation of a 4-dimensional point...]

            For large polytopes, the GNN-sampled 2-face FRT pools can be built
            once with `face_triangs` and reused across NTFE draws with
            [`ntfe_frts`][cytools.ntfe.ntfe.ntfe_frts]. Since only ~1% of the
            sampled expanded secondary cones may be solid, request many more
            than the desired count.

            >>> verts = [[-1,-1,-1,-1],[-1,-1,-1,3],[-1,-1,3,-1],[-1,3,-1,-1],
            ...          [ 1,-1,-1,-1],[ 1,-1,-1,3],[ 1,-1,3,-1],[ 1,3,-1,-1]]
            >>> p = Polytope(verts)  # h11=86
            >>> pools = p.face_triangs(
            ...     dim=2, triang_method="dualgnn", max_npts=0, N_face_triangs=50,
            ...     seed=0,
            ... )  # the expensive step (~10s)
            >>> frsts = p.ntfe_frts(N=300, face_triangs=pools, make_star=True,
            ...                     seed=0)  # cheap (~2s)
        """
        if self.dim() != 4 or not self.is_reflexive():
            raise NotImplementedError(
                "GNN-based sampling is only implemented for reflexive "
                "4-dimensional polytopes."
            )

        # sample the 2-face FRT pools once (the expensive, GNN step) and
        # derive each FRT's secondary-cone inequality block
        if verbosity >= 1:
            print(
                f"Sampling up to {N_face_triangs} FRTs per 2-face with "
                "the GNN (the expensive step; its cost is proportional "
                "to N_face_triangs)..."
            )
        t0 = time.time()
        pools = self.face_triangs(
            dim=2,
            triang_method="dualgnn",
            max_npts=max_npts,
            N_face_triangs=N_face_triangs,
            seed=seed,
            verbosity=verbosity - 1,
        )
        if verbosity >= 1:
            print(
                f"Built pools of sizes {[len(f) for f in pools]} "
                f"in {time.time() - t0:.1f}s. Drawing NTFEs..."
            )
        ineqs = self.triangface_ineqs(
            face_triangs=pools, verbosity=verbosity - 1
        )
        from cytools.ntfe.ntfe import _adjacency_order, _IncrementalLP

        dense = [[np.asarray(t.toarray(), dtype=np.float64) for t in f]
                 for f in ineqs]
        # 2-faces conflict only through shared points, so checking
        # adjacent ones early aborts rejected draws at shallower depth
        dense = [dense[j] for j in _adjacency_order(self)]
        counts = np.array([len(f) for f in dense])
        n_combos = math.prod(counts.tolist())

        def gen():
            rng = np.random.default_rng(seed)
            lp = _IncrementalLP(len(self.labels))
            # remember drawn combinations so no NTFE is yielded twice; if
            # the space is too large to exhaust (redraws are then also
            # vanishingly rare), only remember the accepted ones
            remember_rejects = n_combos <= 1_000_000
            seen = set()
            n_out = 0
            pbar = tqdm(total=N, desc="NTFEs") if verbosity >= 1 else None
            try:
                # each attempt is an independent uniform draw of one FRT
                # per 2-face, accepted iff the stacked inequalities are
                # feasible -- checked face-by-face on one warm LP, so most
                # rejections cost only a few solves instead of a
                # full-stack one
                for _ in range(1000 * N):
                    if n_out >= N or len(seen) >= n_combos:
                        return
                    combo = tuple(rng.integers(counts).tolist())
                    if combo in seen:
                        continue
                    ok, pushed = True, 0
                    for d, k in enumerate(combo):
                        if lp.push(dense[d][k]):
                            pushed += 1
                        else:
                            ok = False
                            break
                    if ok:
                        h = lp.witness()
                    for _ in range(pushed):
                        lp.pop()
                    if ok or remember_rejects:
                        seen.add(combo)
                    if not ok:
                        continue
                    n_out += 1
                    if pbar is not None:
                        pbar.update(1)
                    yield (h if as_heights else
                           self.triangulate(heights=h,
                                            make_star=make_star))
            finally:
                if pbar is not None:
                    pbar.close()

        if as_generator:
            return gen()
        if (not as_heights) and N >= 1000:
            print(
                f"Warning: a list of {N} Triangulation objects can take "
                "several GB of RAM. Consider as_generator=True (process "
                "them one at a time) or as_heights=True (height vectors "
                "are far smaller)."
            )
        return list(gen())

    def all_triangulations(
        self,
        points: ArrayLike = None,
        only_fine: bool = True,
        only_regular: bool = True,
        only_star: bool = None,
        star_origin: int = None,
        include_points_interior_to_facets: bool = None,
        backend: str = None,
        as_list: bool = False,
        raw_output: bool = False,
    ) -> "Iterator | list":
        """Compute all triangulations of the polytope using TOPCOM.

        There is the option to only compute fine, regular, or star
        triangulations. Regularity is checked separately for each
        triangulation, since checking it with TOPCOM is much slower.

        If the polytope is a simplex, its unique triangulation is returned
        and the restriction flags are ignored.

        Warning:
            Polytopes with more than around 15 points usually have too many
            triangulations, so this function may take too long or run out of
            memory. A warning is issued when 17 or more points are used.

        Args:
            points (ArrayLike | None): The labels of the points to be used.
                If specified, `include_points_interior_to_facets` is ignored.
            only_fine (bool): Whether to restrict to fine triangulations.
            only_regular (bool): Whether to restrict to regular
                triangulations. This is ignored if `raw_output` is true.
            only_star (bool | None): Whether to restrict to star
                triangulations. If not specified, it is set to `True` for
                reflexive polytopes (without translations) and `False`
                otherwise.
            star_origin (Hashable | None): The label of the point that will be
                used as the star origin. If not specified and the polytope is
                reflexive (without translations), the origin is used.
                Otherwise, it must be specified when `only_star` is true.
            include_points_interior_to_facets (bool | None): Whether to include
                points interior to facets in the triangulation. If not
                specified, it is set to `False` for reflexive polytopes and
                `True` otherwise.
            backend (str | None): The optimizer used to check regularity. The
                available options are the backends of
                [`Cone.is_solid`][cytools.cone.Cone.is_solid]. If not
                specified, it is picked automatically.
            as_list (bool): Whether to return the full list of triangulations
                at once instead of a generator. By default a generator is
                returned, which is usually preferred for efficiency.
            raw_output (bool): Whether to return each triangulation as an array
                of simplices, each specified by point labels, instead of as a
                [`Triangulation`][cytools.triangulation.Triangulation] object.
                In this case regularity is not checked.

        Returns:
            Iterator | list: A generator of
                [`Triangulation`][cytools.triangulation.Triangulation] objects
                (or of arrays of simplices if `raw_output` is true), or a list
                of them if `as_list` is true.

        Raises:
            ValueError: If `only_star` is true, `star_origin` is not specified,
                and the polytope is not reflexive (without translations).

        Examples:
            We construct a polytope and find all of its triangulations. We try
            picking different restrictions and see how the number of
            triangulations changes.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [-1,-2,-1,-1],[-2,-1,-1,-1]])
            >>> g = p.all_triangulations()
            >>> next(g)  # takes some time while TOPCOM finds all triangulations
            A fine, regular, star triangulation of a 4-dimensional point...
            >>> next(g)  # produces the next triangulation immediately
            A fine, regular, star triangulation of a 4-dimensional point...
            >>> len(p.all_triangulations(as_list=True))  # number of FRSTs
            2

            The number of triangulations, no matter if they are fine, regular,
            or star, is larger.

            >>> len(p.all_triangulations(only_regular=False, only_star=False,
            ...                          only_fine=False, as_list=True))
            6
        """
        if len(self.points()) == self.dim() + 1:
            # simplex... trivial
            triangs = None
            if raw_output:
                # N.B.: points(as_indices=True) returns a list, so it must be
                # cast before it can be indexed with a tuple.
                triangs = [np.asarray(self.points(as_indices=True))[None, :]]
            else:
                triangs = [Triangulation(self, self.labels)]

            if as_list:
                return triangs

            def gen():
                yield from triangs

            return gen()

        if only_star is None:
            only_star = self.is_reflexive(allow_translations=False)
        if only_star and star_origin is None:
            if self.is_reflexive(allow_translations=False):
                star_origin = self._label_origin
            else:
                raise ValueError(
                    "The star_origin parameter must be specified "
                    "when finding star triangulations of "
                    "non-reflexive polytopes."
                )
        if points is not None:
            points = tuple(sorted(set(points)))
        else:
            points = self._triang_labels(include_points_interior_to_facets)

        if len(points) >= 17:
            warnings.warn(
                "Polytopes with more than around 17 points usually "
                "have too many triangulations, so this function may "
                "take too long or run out of memory."
            )

        triangs = all_triangulations(
            self,
            points,
            only_fine=only_fine,
            only_regular=only_regular,
            only_star=only_star,
            star_origin=star_origin,
            backend=backend,
            raw_output=raw_output,
        )
        if as_list:
            return list(triangs)
        return triangs

    # hodge
    # =====
    @instanced_lru_cache(maxsize=None)
    def hpq(self, p: int, q: int, lattice: str="N") -> int:
        r"""Return the Hodge number $h^{p,q}$ of the associated Calabi-Yau.

        The Calabi-Yau is the anticanonical hypersurface in the toric variety
        given by a desingularization of the face fan (if `lattice="N"`) or the
        normal fan (if `lattice="M"`) of the polytope. The Hodge numbers are
        computed combinatorially (Batyrev's formulas), and the results are
        cached.

        Note:
            Only reflexive polytopes of dimension 2 to 5 are currently
            supported.

        Warning:
            Hodge numbers $h^{p,q}$ with $p+q$ larger than the complex
            dimension of the Calabi-Yau (after mapping to the $N$ lattice with
            $p \to d-1-p$ when `lattice="M"`) are currently returned as 0,
            even when they are not zero (e.g. $h^{2,2}=h^{1,1}$ for Calabi-Yau
            threefolds). In particular, `h12(lattice="M")` for 4- and
            5-dimensional polytopes, `h13(lattice="M")` for 5-dimensional
            polytopes, and `h22` for 3- and 4-dimensional polytopes are
            affected. Using the dual polytope with `lattice="N"` gives the
            correct values for the `"M"` lattice.

        Args:
            p (int): The holomorphic index of the Dolbeault cohomology of
                interest.
            q (int): The anti-holomorphic index of the Dolbeault cohomology
                of interest.
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Hodge number $h^{p,q}$ of the arising Calabi-Yau
                manifold.

        Raises:
            ValueError: If the polytope is not reflexive or its dimension is
                not between 2 and 5, or if `lattice` is not `"N"` or `"M"`.

        Examples:
            We construct a polytope and check some Hodge numbers of the
            associated hypersurface.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.hpq(0,0,lattice="N")
            1
            >>> p.hpq(0,1,lattice="N")
            0
            >>> p.hpq(1,1,lattice="N")
            2
            >>> p.hpq(1,2,lattice="N")
            272
        """
        # check that we support hodge-number calculations for this polytope
        d = self.dim()
        if not self.is_reflexive() or d not in (2, 3, 4, 5):
            raise ValueError(
                "Only reflexive polytopes of dimension 2-5 are " "currently supported."
            )

        # check lattice/configure p accordingly
        if lattice == "M":
            p = d - p - 1
        elif lattice != "N":
            raise ValueError(f'Unknown lattice {lattice!r}. Options are: "N" or "M".')

        # assume p,q ordered such that q>p
        if p > q:
            p, q = q, p

        # easy answers
        if (p > d - 1) or (q > d - 1) or (p < 0) or (q < 0) or (p + q > d - 1):
            return 0
        elif (p in (0, d - 1)) or (q in (0, d - 1)):
            if (p == q) or ((p, q) == (0, d - 1)):
                return 1
            return 0

        #
        if p >= d // 2:
            tmp_p = p
            p = d - q - 1
            q = d - tmp_p - 1

        # calculate hpq
        hpq = 0
        if p == 1:
            for f in self.faces(d - q - 1):
                hpq += len(f.interior_points()) * len(f.dual().interior_points())
            if q == 1:
                hpq += len(self.points_not_interior_to_facets()) - d - 1
            if q == d - 2:
                hpq += len(self.dual().points_not_interior_to_facets()) - d - 1
            return hpq
        elif p == 2:
            hpq = (
                44
                + 4 * self.h11(lattice="N")
                - 2 * self.h12(lattice="N")
                + 4 * self.h13(lattice="N")
            )
            return hpq
        raise RuntimeError("Error computing Hodge numbers.")

    def h11(self, lattice="N"):
        r"""Return the Hodge number $h^{1,1}$ of the associated Calabi-Yau.

        This is a shortcut for
        [`hpq(1, 1, lattice)`][cytools.polytope.Polytope.hpq], where the
        conventions are explained.

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Hodge number $h^{1,1}$ of the arising Calabi-Yau
                manifold.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.h11(lattice="N")
            2
            >>> p.h11(lattice="M")
            272
        """
        return self.hpq(1, 1, lattice=lattice)

    def h12(self, lattice="N"):
        r"""Return the Hodge number $h^{1,2}$ of the associated Calabi-Yau.

        This is a shortcut for
        [`hpq(1, 2, lattice)`][cytools.polytope.Polytope.hpq], where the
        conventions are explained. Note that `lattice="M"` currently gives
        incorrect results for 4- and 5-dimensional polytopes (see the warning
        in [`hpq`][cytools.polytope.Polytope.hpq]).

        Also available as `h21`.

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Hodge number $h^{1,2}$ of the arising Calabi-Yau
                manifold.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.h12(lattice="N")
            272
        """
        return self.hpq(1, 2, lattice=lattice)

    h21 = h12

    def h13(self, lattice="N"):
        r"""Return the Hodge number $h^{1,3}$ of the associated Calabi-Yau.

        This is a shortcut for
        [`hpq(1, 3, lattice)`][cytools.polytope.Polytope.hpq], where the
        conventions are explained. It is only non-zero for Calabi-Yau
        fourfolds, i.e. for 5-dimensional polytopes. Note that `lattice="M"`
        currently gives incorrect results for 5-dimensional polytopes (see the
        warning in [`hpq`][cytools.polytope.Polytope.hpq]).

        Also available as `h31`.

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Hodge number $h^{1,3}$ of the arising Calabi-Yau
                manifold.

        Examples:
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-1,-1,-1]])
            >>> p.h13(lattice="N")
            426
        """
        return self.hpq(1, 3, lattice=lattice)

    h31 = h13

    def h22(self, lattice="N"):
        r"""Return the Hodge number $h^{2,2}$ of the associated Calabi-Yau.

        This is a shortcut for
        [`hpq(2, 2, lattice)`][cytools.polytope.Polytope.hpq], where the
        conventions are explained. Note that it is currently only computed
        correctly for Calabi-Yau fourfolds, i.e. for 5-dimensional polytopes
        (see the warning in [`hpq`][cytools.polytope.Polytope.hpq]).

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Hodge number $h^{2,2}$ of the arising Calabi-Yau
                manifold.

        Examples:
            >>> p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],
            ...               [0,0,0,0,1],[-1,-1,-1,-1,-1]])
            >>> p.h22(lattice="N")
            1752
        """
        return self.hpq(2, 2, lattice=lattice)

    def chi(self, lattice: str) -> int:
        """Compute the Euler characteristic of the associated Calabi-Yau.

        The Calabi-Yau is the anticanonical hypersurface in the toric variety
        given by a desingularization of the face fan (if `lattice="N"`) or the
        normal fan (if `lattice="M"`) of the polytope. The Euler
        characteristic is computed from the Hodge numbers, and for the `"M"`
        lattice it is computed from the dual polytope. The result is cached.

        Note:
            Only reflexive polytopes of dimension 2 to 5 are currently
            supported.

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            int: The Euler characteristic of the arising Calabi-Yau manifold.

        Raises:
            ValueError: If the polytope is not reflexive or its dimension is
                not between 2 and 5, or if `lattice` is not `"N"` or `"M"`.

        Examples:
            We construct a polytope and compute the Euler characteristic of
            the associated hypersurfaces.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.chi(lattice="N")
            -540
            >>> p.chi(lattice="M")
            540
        """
        # check that we support hodge-number calculations for this polytope
        if not self.is_reflexive() or self.dim() not in (2, 3, 4, 5):
            raise ValueError(
                "Only reflexive polytopes of dimension 2-5 are " "currently supported."
            )

        # input checking
        if lattice not in ("N", "M"):
            raise ValueError("Lattice must be specified. " 'Options are: "N" or "M".')

        # punt the answer if lattice "M"
        if lattice == "M":
            return self.dual().chi(lattice="N")

        # check if we know the answer
        if self._chi is not None:
            return self._chi

        # calculate the answer
        if self.dim() == 2:
            self._chi = 0
        elif self.dim() == 3:
            self._chi = self.h11(lattice=lattice) + 4
        elif self.dim() == 4:
            self._chi = 2 * (self.h11(lattice=lattice) - self.h21(lattice=lattice))
        elif self.dim() == 5:
            self._chi = 48 + 6 * (
                self.h11(lattice=lattice)
                - self.h12(lattice=lattice)
                + self.h13(lattice=lattice)
            )

        # return
        return self._chi

    def is_favorable(self, lattice: str) -> bool:
        r"""Return whether the associated Calabi-Yau hypersurface is favorable.

        A Calabi-Yau hypersurface is favorable if all of its Kähler forms
        descend from Kähler forms on the ambient toric variety. This is checked
        by verifying that the number of lattice points not interior to facets
        equals $h^{1,1} + d + 1$. For the `"M"` lattice, the dual polytope is
        used. The result is cached.

        Note:
            Only reflexive polytopes of dimension 2 to 5 are currently
            supported.

        Args:
            lattice (str): The lattice on which the polytope is defined. The
                options are `"N"` and `"M"`.

        Returns:
            bool: `True` if the hypersurface is favorable, `False` otherwise.

        Raises:
            ValueError: If `lattice` is not `"N"` or `"M"`, or if the polytope
                is not reflexive or its dimension is not between 2 and 5.

        Examples:
            We construct two reflexive polytopes and find whether they are
            favorable when considered in the N lattice.

            >>> p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
            >>> p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-3,-6]])
            >>> p1.is_favorable(lattice="N")
            True
            >>> p2.is_favorable(lattice="N")
            False
        """
        if lattice == "N":
            if self._is_favorable is None:
                self._is_favorable = (
                    len(self.points_not_interior_to_facets())
                    == self.h11(lattice="N") + self.dim() + 1
                )
            return self._is_favorable
        elif lattice == "M":
            return self.dual().is_favorable(lattice="N")

        raise ValueError("Lattice must be specified. " 'Options are: "N" or "M".')

    # glsm
    # ====
    def glsm_charge_matrix(
        self,
        include_origin: bool = True,
        include_points_interior_to_facets: bool = False,
        points: ArrayLike = None,
        integral: bool = True,
    ) -> np.ndarray:
        """Compute the GLSM charge matrix of the theory arising from the polytope.

        Each column of the GLSM charge matrix corresponds to a lattice point of
        the polytope (i.e. to a prime toric divisor), and the rows span the
        integer linear relations among the points, so that the product of the
        charge matrix with the matrix of points (one per row) vanishes. The
        column of the origin is minus the sum of the other columns, so that
        each row sums to zero. The matrix is chosen such that the columns
        given by [`glsm_basis`][cytools.polytope.Polytope.glsm_basis] form an
        identity matrix (or a diagonal matrix, if `integral` is false). The
        results are cached.

        Args:
            include_origin (bool): Whether to include the origin in the
                calculation. This corresponds to the inclusion of the canonical
                divisor.
            include_points_interior_to_facets (bool): Whether to also use the
                points interior to facets. By default only the origin and the
                boundary points not interior to facets are used.
            points (ArrayLike | None): The indices of the points that will be
                used. If specified, `include_origin` and
                `include_points_interior_to_facets` are ignored, and the origin
                (index 0) is always included as the first column.
            integral (bool): Whether to find an integral basis for the columns
                of the GLSM charge matrix, i.e. a basis such that the remaining
                columns can be written as integer linear combinations of the
                basis columns. If an integral basis cannot be found, a warning
                is issued and a non-integral one is computed.

        Returns:
            np.ndarray: The GLSM charge matrix, with one column per point.

        Raises:
            ValueError: If the polytope is not reflexive, or if any of the
                indices in `points` is out of range.

        Examples:
            We construct a polytope and find the GLSM charge matrix with
            different parameters.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.glsm_charge_matrix()
            array([[-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])

            By definition, this product must be zero.

            >>> p.glsm_charge_matrix().dot(p.points_not_interior_to_facets())
            array([[0, 0, 0, 0],
                   [0, 0, 0, 0]])

            We can exclude the origin (i.e. the canonical divisor), or include
            points interior to facets.

            >>> p.glsm_charge_matrix(include_origin=False)
            array([[1, 9, 6, 1, 1, 0],
                   [0, 3, 2, 0, 0, 1]])
            >>> p.glsm_charge_matrix(include_points_interior_to_facets=True)
            array([[ 0,  1,  0,  6,  1,  1,  0,  0,  0, -9],
                   [-2,  0,  1,  0,  0,  0,  0,  0,  0,  1],
                   [ 0,  0,  0,  2,  0,  0,  1,  0,  0, -3],
                   [ 0,  0,  0,  1,  0,  0,  0,  1,  0, -2],
                   [-1,  0,  0,  1,  0,  0,  0,  0,  1, -1]])
        """
        # check that this makes sense
        if not self.is_reflexive():
            raise ValueError(
                "The GLSM charge matrix can only be computed for "
                "reflexive polytopes."
            )

        # Set up the list of points that will be used.
        if points is not None:
            # We always add the origin, but remove it later if necessary
            pts_ind = set(list(points) + [0])
            if (min(pts_ind) < 0) or (max(pts_ind) >= self.points().shape[0]):
                raise ValueError("An index is out of the allowed range.")

            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = range(self.points().shape[0])
        else:
            pts_ind = range(self.points_not_interior_to_facets().shape[0])
        pts_ind = tuple(pts_ind)

        # check if we know the answer
        if (pts_ind, integral) in self._glsm_charge_matrix:
            out = np.array(self._glsm_charge_matrix[(pts_ind, integral)])

            if (not include_origin) and (points is None):
                return out[:, 1:]
            else:
                return out

        # actually have to do the work...
        # -------------------------------
        # find a basis of columns
        if integral:
            linrel = self.points()[list(pts_ind)].T
            sublat_ind = int(
                round(
                    np.linalg.det(
                        np.array(fmpz_mat(linrel.tolist()).snf().tolist(), dtype=int)[
                            :, : linrel.shape[0]
                        ]
                    )
                )
            )
            norms = [np.linalg.norm(p, 1) for p in linrel.T]
            if self._deterministic_glsm_basis:
                norms = [[n,i] for i,n in enumerate(norms)]
            linrel = np.insert(linrel, 0, np.ones(linrel.shape[1], dtype=int), axis=0)
            good_exclusions = 0
            basis_exc = []
            if self._deterministic_glsm_basis:
                indices = np.array(sorted(range(len(norms)), key=lambda i:norms[i]))
            else:
                indices = np.argsort(norms)
            indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
            for n_try in range(14):
                if n_try == 1:
                    indices[:] = np.array(range(linrel.shape[1]))
                elif n_try == 2:
                    pts_lll = np.array(
                        fmpz_mat(linrel[1:, :].tolist()).lll().tolist(),
                        dtype=int,
                    )
                    norms = [np.linalg.norm(p, 1) for p in pts_lll.T]
                    if self._deterministic_glsm_basis:
                        norms = [[n,i] for i,n in enumerate(norms)]
                    if self._deterministic_glsm_basis:
                        indices = np.array(
                            sorted(range(len(norms)), key=lambda i: norms[i])
                        )
                    else:
                        indices = np.argsort(norms)
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                elif n_try == 3:
                    indices[:] = np.array([0] + list(range(1, linrel.shape[1]))[::-1])
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                elif n_try > 3:
                    if n_try == 4:
                        # use a local generator so that this fallback stays
                        # deterministic without touching NumPy's global RNG
                        rng = np.random.RandomState(1337)
                    rng.shuffle(indices[1:])
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])

                for ctr in range(np.prod(linrel.shape) + 1):
                    found_good_basis = True
                    # DO NOT REMOVE. this makes `ctr > 0` true on the first
                    # pass, so the norm-sorted order is rolled before it is
                    # ever tried. that skip has fixed the default basis since
                    # 2021 (e.g. [5, 6] for P(1,1,1,6,9)); dropping it as dead
                    # code silently changes every default basis, and with it
                    # in-basis intersection numbers and Kahler cone data
                    ctr += 1
                    if ctr > 0:
                        st = max([good_exclusions, 1])
                        indices[st:] = np.roll(indices[st:], -1)
                        indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                    linrel_rand = np.array(linrel[:, indices])
                    try:
                        linrel_hnf = fmpz_mat(linrel_rand.tolist()).hnf()
                    except ValueError:
                        # flint rejected the matrix; try the next ordering
                        continue
                    linrel_rand = np.array(linrel_hnf.tolist(), dtype=int)
                    good_exclusions = 0
                    basis_exc = []
                    tmp_sublat_ind = 1
                    for v in linrel_rand:
                        for i, ii in enumerate(v):
                            if ii == 0:
                                continue

                            tmp_sublat_ind *= abs(ii)
                            if sublat_ind % tmp_sublat_ind == 0:
                                v *= ii // abs(ii)
                                good_exclusions += 1
                            else:
                                found_good_basis = False
                            basis_exc.append(i)
                            break
                        if not found_good_basis:
                            break
                    if found_good_basis:
                        break
                if found_good_basis:
                    break

            if not found_good_basis:
                warnings.warn(
                    "An integral basis could not be found. "
                    "A non-integral one will be computed. However, "
                    "this will not be usable as a basis of divisors "
                    "for the ToricVariety or CalabiYau classes."
                )
                if pts_ind == tuple(
                    self.points_not_interior_to_facets(as_indices=True)
                ):
                    warnings.warn(
                        "Please let the developers know about the "
                        "polytope that caused this issue. "
                        "Here are the vertices of the polytope: "
                        f"{self.vertices().tolist()}"
                    )
                return self.glsm_charge_matrix(
                    include_origin=include_origin,
                    include_points_interior_to_facets=include_points_interior_to_facets,
                    points=points,
                    integral=False,
                )
            linrel_dict = {ii: i for i, ii in enumerate(indices)}
            linrel = np.array(
                linrel_rand[:, [linrel_dict[i] for i in range(linrel_rand.shape[1])]]
            )
            basis_ind = np.array(
                [i for i in range(linrel.shape[1]) if linrel_dict[i] not in basis_exc],
                dtype=int,
            )
            basis_exc = np.array([indices[i] for i in basis_exc])
            glsm = np.zeros(
                (linrel.shape[1] - linrel.shape[0], linrel.shape[1]), dtype=int
            )
            glsm[:, basis_ind] = np.eye(len(basis_ind), dtype=int)
            for nb in basis_exc[::-1]:
                tup = [(k, kk) for k, kk in enumerate(linrel[:, nb]) if kk]
                if sublat_ind % tup[-1][1] != 0:
                    raise RuntimeError("Problem with linear relations")
                i, ii = tup[-1]
                # this whole block is inside the `integral` branch
                glsm[:, nb] = -glsm.dot(linrel[i]) // ii
        else:  # Non-integral basis
            pts = self.points()[list(pts_ind)[1:]]  # Exclude the origin
            pts_norms = [np.linalg.norm(p, 1) for p in pts]
            if self._deterministic_glsm_basis:
                pts_norms = [[n,i] for i,n in enumerate(pts_norms)]
            if self._deterministic_glsm_basis:
                pts_order = np.array(
                    sorted(range(len(pts_norms)), key=lambda i: pts_norms[i])
                )
            else:
                pts_order = np.argsort(pts_norms)
            # Find good lattice basis
            good_lattice_basis = pts_order[:1]
            current_rank = 1
            for p in pts_order:
                tmp = pts[np.append(good_lattice_basis, p)]
                rank = np.linalg.matrix_rank(np.dot(tmp.T, tmp))
                if rank > current_rank:
                    good_lattice_basis = np.append(good_lattice_basis, p)
                    current_rank = rank
                    if rank == self.dim():
                        break
            good_lattice_basis = np.sort(good_lattice_basis)
            glsm_basis = [i for i in range(len(pts)) if i not in good_lattice_basis]
            M = fmpq_mat(pts[good_lattice_basis].T.tolist())
            M_inv = np.array(M.inv().tolist())
            extra_pts = -1 * np.dot(M_inv, pts[glsm_basis].T)
            row_scalings = np.array(
                [np.lcm.reduce([int(ii.q) for ii in i]) for i in extra_pts]
            )
            column_scalings = np.array(
                [np.lcm.reduce([int(ii.q) for ii in i]) for i in extra_pts.T]
            )
            extra_rows = np.multiply(extra_pts, row_scalings[:, None])
            extra_rows = np.array([[int(ii.p) for ii in i] for i in extra_rows])
            extra_columns = np.multiply(extra_pts.T, column_scalings[:, None]).T
            extra_columns = np.array([[int(ii.p) for ii in i] for i in extra_columns])
            glsm = np.diag(column_scalings)
            for p, pp in enumerate(good_lattice_basis):
                glsm = np.insert(glsm, pp, extra_columns[p], axis=1)
            origin_column = -np.dot(glsm, np.ones(len(glsm[0])))
            glsm = np.insert(glsm, 0, origin_column, axis=1)
            linear_relations = extra_rows
            extra_linear_relation_columns = -1 * np.diag(row_scalings)
            for p, pp in enumerate(good_lattice_basis):
                linear_relations = np.insert(
                    linear_relations,
                    pp,
                    extra_linear_relation_columns[p],
                    axis=1,
                )
            linear_relations = np.insert(linear_relations, 0, np.ones(len(pts)), axis=0)
            linear_relations = np.insert(
                linear_relations, 0, np.zeros(self.dim() + 1), axis=1
            )
            linear_relations[0][0] = 1
            linrel = linear_relations
            basis_ind = glsm_basis

        # check that everything was computed correctly
        if (
            np.linalg.matrix_rank(glsm[:, basis_ind]) != len(basis_ind)
            or any(glsm.dot(linrel.T).flat)
            or any(glsm.dot(self.points()[list(pts_ind)]).flat)
        ):
            raise RuntimeError("Error finding basis")

        # cache the results
        if integral:
            self._glsm_charge_matrix[(pts_ind, integral)] = glsm
            self._glsm_linrels[(pts_ind, integral)] = linrel
            self._glsm_basis[(pts_ind, integral)] = basis_ind

        self._glsm_charge_matrix[(pts_ind, False)] = glsm
        self._glsm_linrels[(pts_ind, False)] = linrel
        self._glsm_basis[(pts_ind, False)] = basis_ind

        # return
        if (not include_origin) and (points is None):
            return np.array(self._glsm_charge_matrix[(pts_ind, integral)][:, 1:])
        else:
            return np.array(self._glsm_charge_matrix[(pts_ind, integral)])

    def glsm_linear_relations(
        self,
        include_origin: bool = True,
        include_points_interior_to_facets: bool = False,
        points: ArrayLike = None,
        integral: bool = True,
    ) -> np.ndarray:
        """Compute the linear relations of the columns of the GLSM charge matrix.

        The rows of the returned matrix span the linear relations among the
        columns of the GLSM charge matrix, so that the product of the two
        (with one of them transposed) vanishes. When the origin is included,
        one of the relations states that the rows of the charge matrix sum to
        zero. They are computed (and cached) together with the charge matrix
        by [`glsm_charge_matrix`][cytools.polytope.Polytope.glsm_charge_matrix].

        Args:
            include_origin (bool): Whether to include the origin in the
                calculation. This corresponds to the inclusion of the canonical
                divisor. If `False`, the first row and column are dropped.
            include_points_interior_to_facets (bool): Whether to also use the
                points interior to facets. By default only the origin and the
                boundary points not interior to facets are used.
            points (ArrayLike | None): The indices of the points that will be
                used. If specified, `include_origin` and
                `include_points_interior_to_facets` are ignored, and the origin
                (index 0) is always included.
            integral (bool): Whether to find an integral basis for the columns
                of the GLSM charge matrix, i.e. a basis such that the remaining
                columns can be written as integer linear combinations of the
                basis columns.

        Returns:
            np.ndarray: The matrix of linear relations of the columns of the
                GLSM charge matrix, one per row.

        Raises:
            ValueError: If any of the indices in `points` is out of range, or
                if the polytope is not reflexive.

        Examples:
            We construct a polytope and find its linear relations with
            different parameters.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.glsm_linear_relations()
            array([[ 1,  0,  0,  0,  0, 18,  6],
                   [ 0,  1,  0,  0,  0, -1,  0],
                   [ 0,  0,  1,  0,  0, -9, -3],
                   [ 0,  0,  0,  1,  0, -6, -2],
                   [ 0,  0,  0,  0,  1, -1,  0]])

            By definition, this product must be zero.

            >>> p.glsm_linear_relations().dot(p.glsm_charge_matrix().T)
            array([[0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0]])

            We can exclude the origin (i.e. the canonical divisor).

            >>> p.glsm_linear_relations(include_origin=False)
            array([[ 1,  0,  0,  0, -1,  0],
                   [ 0,  1,  0,  0, -9, -3],
                   [ 0,  0,  1,  0, -6, -2],
                   [ 0,  0,  0,  1, -1,  0]])
        """
        if points is not None:
            pts_ind = tuple(set(list(points) + [0]))
            if min(pts_ind) < 0 or max(pts_ind) >= self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = tuple(range(self.points().shape[0]))
        else:
            pts_ind = tuple(range(self.points_not_interior_to_facets().shape[0]))
        if (pts_ind, integral) in self._glsm_linrels:
            if not include_origin and points is None:
                return np.array(self._glsm_linrels[(pts_ind, integral)][1:, 1:])
            return np.array(self._glsm_linrels[(pts_ind, integral)])
        # If linear relations are not cached we just call the GLSM charge
        # matrix function since they are computed there
        self.glsm_charge_matrix(
            include_origin=True,
            include_points_interior_to_facets=include_points_interior_to_facets,
            points=points,
            integral=integral,
        )
        if not include_origin and points is None:
            return np.array(self._glsm_linrels[(pts_ind, integral)][1:, 1:])
        return np.array(self._glsm_linrels[(pts_ind, integral)])

    def glsm_basis(
        self,
        include_origin: bool = True,
        include_points_interior_to_facets: bool = False,
        points: ArrayLike = None,
        integral: bool = True,
    ) -> np.ndarray:
        """Compute a basis of columns of the GLSM charge matrix.

        The basis is computed (and cached) together with the charge matrix by
        [`glsm_charge_matrix`][cytools.polytope.Polytope.glsm_charge_matrix],
        and the corresponding columns of the charge matrix form an identity
        matrix (or a diagonal matrix, if `integral` is false). The chosen
        basis can be made machine-independent with the
        `deterministic_glsm_basis` option of the constructor.

        Args:
            include_origin (bool): Whether to include the origin in the
                calculation. This corresponds to the inclusion of the canonical
                divisor. If `False`, the returned indices are shifted so that
                they refer to the charge matrix without the origin column.
            include_points_interior_to_facets (bool): Whether to also use the
                points interior to facets. By default only the origin and the
                boundary points not interior to facets are used.
            points (ArrayLike | None): The indices of the points that will be
                used. If specified, `include_origin` and
                `include_points_interior_to_facets` are ignored, and the origin
                (index 0) is always included. Note that the returned indices
                then refer to the columns of the corresponding charge matrix,
                i.e. to the positions in the sorted list of point indices.
            integral (bool): Whether to find an integral basis for the columns
                of the GLSM charge matrix, i.e. a basis such that the remaining
                columns can be written as integer linear combinations of the
                basis columns.

        Returns:
            np.ndarray: The indices of the columns that form a basis.

        Raises:
            ValueError: If any of the indices in `points` is out of range, or
                if the polytope is not reflexive.

        Examples:
            We construct a polytope, find its GLSM charge matrix and a basis of
            columns.

            >>> import numpy as np
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.glsm_basis()
            array([5, 6])
            >>> glsm = p.glsm_charge_matrix()
            >>> glsm[:,p.glsm_basis()]  # the basis columns form an identity matrix
            array([[1, 0],
                   [0, 1]])
        """
        if points is not None:
            pts_ind = tuple(set(list(points) + [0]))
            if min(pts_ind) < 0 or max(pts_ind) >= self.points().shape[0]:
                raise ValueError("An index is out of the allowed range.")
            include_origin = 0 in points
        elif include_points_interior_to_facets:
            pts_ind = tuple(range(self.points().shape[0]))
        else:
            pts_ind = tuple(range(self.points_not_interior_to_facets().shape[0]))
        if (pts_ind, integral) in self._glsm_basis:
            if not include_origin and points is None:
                return np.array(self._glsm_basis[(pts_ind, integral)]) - 1
            return np.array(self._glsm_basis[(pts_ind, integral)])
        # If basis is not cached we just call the GLSM charge matrix function
        # since it is computed there
        self.glsm_charge_matrix(
            include_origin=True,
            include_points_interior_to_facets=include_points_interior_to_facets,
            points=points,
            integral=integral,
        )
        if not include_origin and points is None:
            return np.array(self._glsm_basis[(pts_ind, integral)]) - 1
        return np.array(self._glsm_basis[(pts_ind, integral)])

    # misc
    # ====
    def minkowski_sum(self, other: "Polytope") -> "Polytope":
        """Return the Minkowski sum of the two polytopes.

        The sum is computed as the convex hull of all pairwise sums of the
        vertices of the two polytopes, so both must lie in lattices of the
        same dimension. The resulting polytope uses the default backend.

        Also available through the `+` operator.

        Args:
            other (Polytope): The other polytope used for the Minkowski sum.

        Returns:
            Polytope: The Minkowski sum.

        Examples:
            We construct two polytopes and compute their Minkowski sum.

            >>> p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
            >>> p2 = Polytope([[0,0,1],[0,0,-1]])
            >>> print(p1.minkowski_sum(p2))
            A 3-dimensional reflexive lattice polytope in ZZ^3
        """
        points = {tuple(sum(verts))
                  for verts in itertools.product(self.vertices(),
                                                 other.vertices())}
        return Polytope(
            list(points),
            deterministic_glsm_basis=self._deterministic_glsm_basis,
        )

    def volume(self) -> int:
        """Return the normalized volume of the polytope.

        The volume is computed relative to the lattice of the affine hull of
        the polytope, so it is well defined for polytopes that are not
        full-dimensional. The result is cached.

        Warning:
            By convention, the standard simplex has unit volume. To get the
            more typical Euclidean volume it must be divided by $d!$.

        Returns:
            int: The normalized volume of the polytope. It is 0 for
                0-dimensional polytopes.

        Examples:
            We construct a standard simplex and a cube, and find their volumes.

            >>> p1 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
            >>> p2 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0],
            ...                [0,1,1],[1,0,1],[1,1,0],[1,1,1]])
            >>> p1.volume()
            1
            >>> p2.volume()
            6
        """
        # calculate the answer if not known
        if self._volume is None:
            if self.dim() == 0:
                self._volume = 0
            elif self.dim() == 1:
                # points(optimal=True) is (n,1)-shaped here, so flatten it to
                # get an int rather than a length-1 array
                pts_1d = self.points(optimal=True).flatten()
                self._volume = int(max(pts_1d) - min(pts_1d))
            else:
                self._volume = ConvexHull(self.points(optimal=True)).volume
                self._volume *= math.factorial(self.dim())
                self._volume = int(round(self._volume))

        # return
        return self._volume

    def _huang_taylor_sets(self) -> list[list[tuple]]:
        """Construct the sets $S_1$, $S_2$, $S_3$ used by the Huang-Taylor algorithm.

        See [arXiv:1907.09482](https://arxiv.org/abs/1907.09482). $S_i$ holds
        the lattice points whose maximum dot product with the vertices of the
        dual polytope equals $i$. These sets are disjoint by construction.

        Returns:
            list[list[tuple]]: The three sets $S_1$, $S_2$, $S_3$, each given
                as a list of point tuples.

        Raises:
            NotImplementedError: If the dual polytope cannot be computed, i.e.
                if the polytope is not reflexive (without translations).
        """
        dual_vert = self.dual().vertices()

        S_i = [[], [], []]
        for p in self.points():
            m = max(p.dot(v) for v in dual_vert)
            if m in (1, 2, 3):
                S_i[m - 1].append(tuple(p))

        return S_i

    def find_2d_reflexive_subpolytopes(self) -> list["Polytope"]:
        """Find the 2-dimensional reflexive subpolytopes of a 4D polytope.

        This uses the algorithm by Huang and Taylor described in
        [arXiv:1907.09482](https://arxiv.org/abs/1907.09482). Each subpolytope
        is the intersection of the polytope with a 2-dimensional plane through
        the origin, and it is returned as a new `Polytope` (with default
        labels) in the ambient lattice of this polytope. The subpolytopes are
        not returned in any particular order.

        Returns:
            list[Polytope]: The 2-dimensional reflexive subpolytopes.

        Raises:
            NotImplementedError: If the polytope is not a reflexive
                4-dimensional polytope.

        Examples:
            We construct a polytope and find its 2D reflexive subpolytopes.

            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> subpolys = p.find_2d_reflexive_subpolytopes()
            >>> len(subpolys)
            1
            >>> print(subpolys[0])
            A 2-dimensional reflexive lattice polytope in ZZ^4
        """
        if not self.is_reflexive() or self.dim() != 4:
            raise NotImplementedError("Only 4D reflexive polytopes are supported.")
        pts = self.points()
        S_i = self._huang_taylor_sets()
        # Check each of the three conditions
        gen_pts = []
        for i in range(len(S_i[0])):
            if tuple(-np.array(S_i[0][i])) in S_i[0]:
                for j in range(i + 1, len(S_i[0])):
                    if (
                        tuple(-np.array(S_i[0][j])) in S_i[0]
                        and tuple(-np.array(S_i[0][i])) != S_i[0][j]
                    ):
                        gen_pts.append((S_i[0][i], S_i[0][j]))
        for i in range(len(S_i[1])):
            for j in range(i + 1, len(S_i[1])):
                p = tuple(-np.array(S_i[1][i]) - np.array(S_i[1][j]))
                if p in S_i[0] or p in S_i[1]:
                    gen_pts.append((S_i[1][i], S_i[1][j]))
        for i in range(len(S_i[2])):
            for j in range(i + 1, len(S_i[2])):
                p = -np.array(S_i[2][i]) - np.array(S_i[2][j])
                if all(c % 2 == 0 for c in p) and tuple(p // 2) in S_i[0]:
                    gen_pts.append((S_i[2][i], S_i[2][j]))
        polys_2d = set()
        for p1, p2 in gen_pts:
            pts_2d = set()
            for p in pts:
                if np.linalg.matrix_rank((p1, p2, p)) == 2:
                    pts_2d.add(tuple(p))
            if np.linalg.matrix_rank(list(pts_2d)) == 2:
                polys_2d.add(tuple(sorted(pts_2d)))
        return [Polytope(pp) for pp in polys_2d]

    def nef_partitions(
        self,
        keep_symmetric: bool = False,
        keep_products: bool = False,
        keep_projections: bool = False,
        codim: int = 2,
        compute_hodge_numbers: bool = True,
        return_hodge_numbers: bool = False,
    ) -> tuple:
        r"""Compute the nef partitions of the polytope using PALP.

        A nef partition is a partition of the vertices of the polytope into
        `codim` parts, which defines a complete intersection Calabi-Yau (CICY).
        Each part is given as a tuple of vertex indices, where the vertices
        are ordered as by PALP. For polytopes with default labels, this
        matches the ordering of
        [`vertices`][cytools.polytope.Polytope.vertices]. The results are
        cached.

        Note:
            This is currently an experimental feature and may change
            significantly in future versions. It requires the experimental
            features to be enabled with
            [`enable_experimental_features`]
            [cytools.config.enable_experimental_features].

        Args:
            keep_symmetric (bool): Whether to keep symmetric partitions related
                by lattice automorphisms.
            keep_products (bool): Whether to keep product partitions,
                corresponding to complete intersections that are direct
                products.
            keep_projections (bool): Whether to keep projection partitions,
                i.e. partitions where one of the parts consists of a single
                vertex.
            codim (int): The number of parts in the partition or, equivalently,
                the codimension of the complete intersection Calabi-Yau.
            compute_hodge_numbers (bool): Whether to compute the Hodge numbers
                of the CICYs. This is always done if `return_hodge_numbers` is
                true.
            return_hodge_numbers (bool): Whether to return the Hodge numbers
                along with the nef partitions. For each nef partition, the Hodge
                numbers are given as a tuple of tuples whose $(p,q)$ entry is
                $h^{p,q}$.

        Returns:
            list | tuple[list, list]: The list of nef partitions of the
                polytope. If `return_hodge_numbers` is true, a tuple with the
                list of nef partitions and the list of the corresponding Hodge
                numbers is returned instead.

        Raises:
            Exception: If the experimental features are not enabled.
            ValueError: If the polytope is not reflexive.

        Examples:
            We construct a tesseract and find the 2- and 3-part nef partitions.

            >>> from cytools import config
            >>> config.enable_experimental_features()
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            ...               [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
            >>> nef_part_2 = p.nef_partitions()  # default codimension is 2
            >>> nef_part_2[0]  # the first of the nef partitions
            ((0, 1, 2, 5), (3, 4, 6, 7))
            >>> nef_part_3 = p.nef_partitions(codim=3)
            >>> nef_part_3[0]
            ((0, 1, 6), (2, 3, 4), (5, 7))

            We can also get the Hodge numbers of the CICYs, which in this case
            are K3 surfaces.

            >>> nef_parts, hodge_nums = p.nef_partitions(return_hodge_numbers=True)
            >>> hodge_nums[0]
            ((1, 0, 1), (0, 20, 0), (1, 0, 1))
        """
        if not config._exp_features_enabled:
            raise Exception(
                "The experimental features must be enabled to "
                "compute nef partitions."
            )
        if return_hodge_numbers:
            compute_hodge_numbers = True
        args_id = (
            keep_symmetric,
            keep_products,
            keep_projections,
            codim,
            compute_hodge_numbers,
        )
        if self._nef_parts.get(args_id, None) is not None:
            return (
                self._nef_parts.get(args_id)
                if return_hodge_numbers or not compute_hodge_numbers
                else self._nef_parts.get(args_id)[0]
            )
        if not self.is_reflexive():
            raise ValueError("The polytope must be reflexive")
            
        p = pypalp.Polytope(self.points(optimal=True))
        results = p.nef_partitions(
            codim=codim,
            keep_symmetric=keep_symmetric,
            keep_products=keep_products,
            keep_projections=keep_projections,
            with_hodge_numbers=compute_hodge_numbers
        )
        
        nef_parts = [tuple(
            tuple(part)
            for part in partition[0]
        ) for partition in results]
        
        if compute_hodge_numbers:
            hodge_nums = [
                tuple(tuple(part) for part in partition[1]) for partition in results
            ]
            nef_parts = (nef_parts, hodge_nums)
            
        self._nef_parts[args_id] = nef_parts
        return (
            self._nef_parts.get(args_id)
            if return_hodge_numbers or not compute_hodge_numbers
            else self._nef_parts.get(args_id)[0]
        )

    def is_trilayer(self, return_anticanon=False):
        """Return whether the polytope is "trilayer".

        This computes the GLSM charges of the vertices (a basis of the integer
        linear relations among the vertices) and the anticanonical class,
        given by the sum of the charges of all vertices. The polytope is
        considered trilayer if the anticanonical class is divisible by 2 and
        some vertex has charges equal to half of it. The polytope is assumed to
        be full-dimensional.

        Args:
            return_anticanon (bool): Whether to also return the charges of the
                anticanonical class.

        Returns:
            bool | tuple[bool, np.ndarray]: `True` if the polytope is trilayer,
                `False` otherwise. If `return_anticanon` is true, a tuple with
                this value and the charges of the anticanonical class is
                returned instead.

        Examples:
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.is_trilayer()
            True
            >>> p.is_trilayer(return_anticanon=True)
            (True, array([18]))
        """
        glsm_vert = np.array(
            fmpz_mat(self.vertices().T.tolist()).nullspace()[0].transpose().tolist(),
            dtype=int,
        )[: -self.dim()]
        anticanon = np.sum(glsm_vert, axis=1)

        # compute if the Polytope is trilayer
        is_tri = False
        if all(c%2==0 for c in anticanon):
            half_anticanon = anticanon//2
            is_tri = any(all((v == half_anticanon).flat) for v in glsm_vert.T)
        
        # return
        if return_anticanon:
            return is_tri, anticanon
        else:
            return is_tri


# utils
# -----
def poly_v_to_h(pts: ArrayLike, backend: str) -> (ArrayLike, None):
    r"""Compute the H-representation of a polytope from its V-representation.

    That is, map the points (or vertices) to the hyperplane inequalities
    defining their convex hull. Each row $(c_0, \ldots, c_{d-1}, c_d)$ of the
    output encodes the inequality

    $$c_0x_0 + \cdots + c_{d-1}x_{d-1} + c_d \geq 0.$$

    The points are expected to span a full-dimensional polytope. Otherwise,
    the output may contain equalities (PPL) or the computation may fail.

    Args:
        pts (ArrayLike): The input points, one per row.
        backend (str): The backend to use. The options are `"ppl"`, `"qhull"`,
            and `"palp"`.

    Returns:
        tuple[np.ndarray, object]: The integer array of inequalities, one per
            row, and the convex hull object computed by the backend. The
            latter is a `ppl.C_Polyhedron` for PPL, a
            `scipy.spatial.ConvexHull` for QHull (or `None` for 1-dimensional
            polytopes), and `None` for PALP.

    Raises:
        ValueError: If `backend` is not one of the allowed options.

    Examples:
        >>> from cytools.polytope import poly_v_to_h
        >>> ineqs, _ = poly_v_to_h([[1,0],[0,1],[-1,-1]], backend="palp")
        >>> ineqs
        array([[ 2, -1,  1],
               [-1,  2,  1],
               [-1, -1,  1]])
    """
    # preliminary
    dim = len(pts[0])

    # do the work, depending on backend
    if backend == "ppl":
        gs = ppl.Generator_System()
        vrs = np.array([ppl.Variable(i) for i in range(dim)])

        # insert points to generator system
        for linexp in pts@vrs:
            gs.insert(ppl.point(linexp))

        # find polytope, hyperplanes
        poly = ppl.C_Polyhedron(gs)
        ineqs = []
        for ineq in poly.minimized_constraints():
            ineqs.append(list(ineq.coefficients()) + [ineq.inhomogeneous_term()])
        ineqs = np.array(ineqs, dtype=int) # the data should automatically be integer

    elif backend == "qhull":
        if dim == 1:
            # qhull cannot handle 1-dimensional polytopes
            poly = None
            ineqs = np.array([[1, -np.min(pts)], [-1, np.max(pts)]], dtype=int)

        else:
            poly = ConvexHull(pts)

            # get the ineqs, ensure right sign and gcd
            ineqs = set()
            for eq in poly.equations:
                g = abs(gcd_list(eq))
                ineqs.add(tuple(-int(round(i / g)) for i in eq))
            ineqs = np.array(list(ineqs), dtype=int)

    elif backend == "palp":
        poly = None
        if dim == 0:
            # PALP cannot handle 0-dimensional polytopes
            ineqs = np.array([[0]])
        else:
            # prepare the command
            p = pypalp.Polytope(pts)
            ineqs = p.equations()
    else:
        raise ValueError(f"Unrecognized backend '{backend}'...")

    return ineqs, poly


def saturating_lattice_pts(
    pts_in: [tuple],
    ineqs: ArrayLike = None,
    dim: int = None,
    backend: str = None,
) -> (ArrayLike, [frozenset]):
    """Compute the lattice points of a polytope and the inequalities they saturate.

    The lattice points contained in the convex hull of `pts_in` are
    enumerated within a bounding box, and for each of them the indices of the
    hyperplane inequalities that it saturates are found. The points are
    expected to span a full-dimensional polytope.

    Args:
        pts_in (list[tuple]): The points whose convex hull defines the
            polytope.
        ineqs (ArrayLike | None): The hyperplane inequalities defining the
            polytope, in the same format as the output of
            [`poly_v_to_h`][cytools.polytope.poly_v_to_h]. If not specified,
            they are computed with `backend`.
        dim (int | None): The dimension of the polytope. If not specified, it
            is computed from the points.
        backend (str | None): The backend used to compute the inequalities if
            they are not given. The options are `"ppl"`, `"qhull"`, and
            `"palp"`. If not specified, PPL is used for polytopes of dimension
            1 to 4, and PALP otherwise. 0-dimensional polytopes always use
            PALP.

    Returns:
        tuple[np.ndarray, list[frozenset[int]]]: The lattice points, one per
            row, and, for each point, the set of indices of the inequalities
            that it saturates. The points are not in any particular order.

    Raises:
        ValueError: If the inequalities have coefficients that do not fit in a
            32-bit integer.

    Examples:
        >>> from cytools.polytope import saturating_lattice_pts
        >>> pts, sat = saturating_lattice_pts([(1,0),(0,1),(-1,-1)])
        >>> pts
        array([[-1, -1],
               [ 0,  0],
               [ 1,  0],
               [ 0,  1]], dtype=int32)
        >>> sat
        [frozenset({0, 1}), frozenset(), frozenset({1, 2}), frozenset({0, 2})]
    """
    # check/clean points
    if isinstance(pts_in, list) and isinstance(pts_in[0], tuple):
        pts = pts_in
    else:
        pts = [tuple(pt) for pt in pts_in]

    # check/clean ineqs
    if dim is None:
        dim = np.linalg.matrix_rank([list(pt) + [1] for pt in pts]) - 1
    
    if backend is None:
        if 1 <= dim <= 4:
            backend = "ppl"
        else:
            backend = "palp"
    if dim == 0:
        backend = "palp"
    if ineqs is None:
        ineqs, _ = poly_v_to_h(pts, backend)

    ineqs = np.array(ineqs)

    # the lattice-point kernel operates on int32 data, so check that the H-rep
    # actually fits before casting (otherwise the values wrap silently)
    int32_info = np.iinfo(np.int32)
    if ineqs.size and (
        (ineqs.min() < int32_info.min) or (ineqs.max() > int32_info.max)
    ):
        raise ValueError(
            "The H-representation of the polytope has coefficients that do "
            "not fit in a 32-bit integer, which the lattice point enumerator "
            "requires."
        )

    H     = np.ascontiguousarray(ineqs[:,:-1]).astype(np.int32)
    rhs   = -np.ascontiguousarray(ineqs[:,-1]).astype(np.int32)

    # compute the lattice ponts
    max_N_out = 1_000_000
    while True:
        pts, status, __ = latticepts.box_enum(
            B=np.max(np.abs(pts_in)),
            H=H,
            rhs=rhs,
            max_N_out=max_N_out)
        if len(pts) == max_N_out:
            max_N_out *= 10
            if max_N_out >= 1e9:
                print("your polytope has >=1e9 lattice points...")
        else:
            break
    pts = np.array(pts)

    # compute the saturation (in int64, so that the products can't overflow)
    satd = (pts.astype(np.int64) @ H.T.astype(np.int64)) == rhs.reshape(1, -1)
    satd = [
        frozenset([int(x) for x in np.where(satd_i)[0]])
        for satd_i in satd
    ]
    
    return pts, satd


def is_reflexive_barebones(points: "ArrayLike",
                           backend: str = 'qhull') -> bool:
    """Check whether the convex hull of the points is reflexive.

    This is a minimal and fast check that avoids constructing a
    [`Polytope`][cytools.polytope.Polytope] object. The polytope is considered
    reflexive if it is full-dimensional and all of its facets are at lattice
    distance 1 from the origin. Translations are not allowed, so the origin
    must be the unique interior point.

    Args:
        points (ArrayLike): The points whose convex hull defines the polytope.
        backend (str): The backend used to compute the inequalities. The
            options are `"ppl"`, `"qhull"`, and `"palp"` (see
            [`poly_v_to_h`][cytools.polytope.poly_v_to_h]).

    Returns:
        bool: `True` if the convex hull of the points is reflexive, `False`
            otherwise.

    Examples:
        >>> from cytools.polytope import is_reflexive_barebones
        >>> is_reflexive_barebones([[1,0],[0,1],[-1,-1]])
        True
        >>> is_reflexive_barebones([[2,1],[1,2],[0,0]])
        False
    """
    # check if the convex hull is solid
    ambient_dim = len(points[0])
    dim = np.linalg.matrix_rank([list(pt) + [1] for pt in points]) - 1
    if dim != ambient_dim:
        return False

    # check the distance for each inequality
    ineqs, _ = poly_v_to_h(points, backend=backend)
    for ineq in ineqs:
        if ineq[-1] != 1:
            return False

    # all checks passed
    return True
