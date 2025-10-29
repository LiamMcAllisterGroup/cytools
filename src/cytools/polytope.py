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

# 'standard' imports
from collections import defaultdict
import copy
import itertools
import math
import subprocess
import warnings

# 3rd party imports
from flint import fmpz_mat, fmpq_mat
import numpy as np
from numpy.typing import ArrayLike
import ppl
from scipy.spatial import ConvexHull
from tqdm import tqdm
import pypalp

# CYTools imports
from cytools import config
from cytools.polytopeface import PolytopeFace
from cytools.triangulation import (
    Triangulation,
    all_triangulations,
    random_triangulations_fast_generator,
    random_triangulations_fair_generator,
)
from cytools.utils import gcd_list, lll_reduce, instanced_lru_cache


class Polytope:
    """
    This class handles all computations relating to lattice polytopes, such as
    the computation of lattice points and faces. When using reflexive
    polytopes, it also allows the computation of topological properties of the
    arising Calabi-Yau hypersurfaces that only depend on the polytope.

    ## Constructor

    ### `cytools.polytope.Polytope`

    **Description:**
    Constructs a `Polytope` object describing a lattice polytope. This is
    handled by the hidden [`__init__`](#__init__) function.

    :::note notes
    - CYTools only supports lattice polytopes, so any floating point numbers
        will be truncated to integers.
    - The Polytope class is also imported to the root of the CYTools package,
        so it can be imported from `cytools.polytope` or from `cytools`.
    :::

    **Arguments:**
    - `points`: A list of lattice points defining the polytope as their
        convex hull.
    - `labels`: A list of labels to specify the points. I.e., points[i] is
        labelled/accessed as labels[i]. If no labels are provided, then the
        points are given semi-arbitrary default labels.
    - `backend`: A string that specifies the backend used to construct the
        convex hull. The available options are "ppl", "qhull", or "palp".
        When not specified, it uses PPL for dimensions up to four, and palp
        otherwise.

    **Example:**
    We construct two polytopes from lists of points.
    ```python {2,5}
    from cytools import Polytope
    p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    print(p1)
    # A 4-dimensional reflexive lattice polytope in ZZ^4
    p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
    print(p2)
    # A 3-dimensional lattice polytope in ZZ^4
    ```
    """

    def __init__(
        self, points: ArrayLike, labels: ArrayLike = None, backend: str = None
    ) -> None:
        """
        **Description:**
        Initializes a `Polytope` object describing a lattice polytope.

        :::note
        CYTools only supports lattice polytopes, so any floating point numbers
        will be truncated to integers.
        :::

        **Arguments:**
        - `points`: A list of lattice points defining the polytope as their
            convex hull.
        - `labels`: A list of labels to specify the points. I.e., points[i] is
            labelled/accessed as labels[i]. If no labels are provided, then the
            points are given semi-arbitrary default labels.
        - `backend`: A string that specifies the backend used to construct the
            convex hull. The available options are "ppl", "qhull", or "palp".
            When not specified, it uses PPL for dimensions up to four, and palp
            otherwise.

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new `Polytope`
        object. Thus, it is used in the following example.
        ```python {2,5}
        from cytools import Polytope
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        print(p1)
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        print(p2)
        # A 3-dimensional lattice polytope in ZZ^4
        ```
        """
        # input checking
        # --------------
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
        self.clear_cache()

        # process the inputs
        # ------------------
        # dimension
        self._dim_ambient = len(points[0])
        self._dim = int(np.linalg.matrix_rank([list(pt) + [1] for pt in points]) - 1)
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
        """
        **Description:**
        Returns an umabiguous string describing the polytope.

        **Arguments:**
        None.

        **Returns:**
        A string describing the polytope.

        **Example:**
        This function can be used to convert the polytope to a string or to
        print information about the polytope.
        ```python {2,3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        print(repr(p)) # Prints polytope info
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        ```
        """
        return (
            f"A {self.dim()}-dimensional "
            f"{('reflexive ' if self.is_reflexive() else '')}"
            f"lattice polytope in ZZ^{self.ambient_dim()} "
            f"with points {list(self._inputpts2labels.keys())} "
            f"which are labelled {list(self._inputpts2labels.values())}"
        )

    def __str__(self) -> str:
        """
        **Description:**
        Returns a human-readable string describing the polytope.

        **Arguments:**
        None.

        **Returns:**
        A string describing the polytope.

        **Example:**
        This function can be used to convert the polytope to a string or to
        print information about the polytope.
        ```python {2,3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        poly_info = str(p) # Converts to string
        print(p) # Prints polytope info
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        ```
        """
        return (
            f"A {self.dim()}-dimensional "
            f"{('reflexive ' if self.is_reflexive() else '')}"
            f"lattice polytope in ZZ^{self.ambient_dim()}"
        )

    def __getstate__(self):
        """
        **Description:**
        Gets the state of the class instance, for pickling.

        **Arguments:**
        None

        **Returns:**
        Nothing.
        """
        state = self.__dict__.copy()
        # delete instanced_lru_cache since it doesn't play nicely with pickle
        state["_cache"] = None
        return state

    def __setstate__(self, state: dict):
        """
        **Description:**
        Gets the state of the class instance, for pickling.

        **Arguments:**
        - `state`: The dictionary of the instance state, read from pickle.

        **Returns:**
        Nothing.
        """
        self.__dict__.update(state)
        # re-initialize the instanced_lru_cache
        # (needed since we check if self has _cache, which it is now None)
        self._cache = {}

    def __eq__(self, other: "Polytope") -> bool:
        """
        **Description:**
        Implements comparison of polytopes with ==.

        **Arguments:**
        - `other`: The other polytope that is being compared.

        **Returns:**
        The truth value of the polytopes being equal.

        **Example:**
        We construct two polytopes and compare them.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p1 == p2
        # True
        ```
        """
        if not isinstance(other, Polytope):
            return False

        our_verts = self.vertices().tolist()
        other_verts = other.vertices().tolist()
        return sorted(our_verts) == sorted(other_verts)

    def __ne__(self, other: "Polytope") -> bool:
        """
        **Description:**
        Implements comparison of polytopes with !=.

        **Arguments:**
        - `other`: The other polytope that is being compared.

        **Returns:**
        The truth value of the polytopes being different.

        **Example:**
        We construct two polytopes and compare them.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p1 != p2
        # False
        ```
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        **Description:**
        Implements the ability to obtain hash values from polytopes.

        **Arguments:**
        None.

        **Returns:**
        The hash value of the polytope.

        **Example:**
        We compute the hash value of a polytope. Also, we construct a set and a
        dictionary with a polytope, which make use of the hash function.
        ```python {2,3,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        h = hash(p) # Obtain hash value
        d = {p: 1} # Create dictionary with polytope keys
        s = {p} # Create a set of polytopes
        ```
        """
        if self._hash is None:
            self._hash = hash(tuple(sorted(tuple(v) for v in self.vertices())))

        return self._hash

    def __add__(self, other: "Polytope") -> "Polytope":
        """
        **Description:**
        Implements addition of polytopes with the
        [`minkowski_sum`](#minkowski_sum) function.

        **Arguments:**
        - `other`: The other polytope used for the Minkowski sum.

        **Returns:**
        The Minkowski sum.

        **Example:**
        We construct two polytopes and compute their Minkowski sum.
        ```python {3}
        p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
        p2 = Polytope([[0,0,1],[0,0,-1]])
        p1 + p2
        # A 3-dimensional reflexive lattice polytope in ZZ^3
        ```
        """
        if not isinstance(other, Polytope):
            raise ValueError

        return self.minkowski_sum(other)

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

    def clear_cache(self) -> None:
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We compute the lattice points of a large polytope.
        ```python {4}
        p = Polytope([[-1,-1,-1,-1,-1],[3611,-1,-1,-1,-1],[-1,42,-1,-1,-1],[-1,-1,6,-1,-1],[-1,-1,-1,2,-1],[-1,-1,-1,-1,1]])
        pts = p.points() # Takes a few seconds
        pts = p.points() # It runs instantly because the result is cached
        p.clear_cache() # Clears the results of any previous computation
        pts = p.points() # Again it takes a few seconds since the cache was cleared
        ```
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
        self._is_reflexive = None

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
        """
        **Description:**
        Returns the backend.

        **Arguments:**
        None.

        **Returns:**
        The computational backend
        """
        return self._backend

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
        We construct a polytope and check the dimension of the ambient lattice.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.ambient_dimension()
        # 4
        ```
        """
        return self._dim_ambient

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self) -> int:
        """
        **Description:**
        Returns the dimension of the polytope.

        **Arguments:**
        None.

        **Returns:**
        The dimension of the polytope.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a polytope and check its dimension.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.dimension()
        # 3
        ```
        """
        return self._dim

    # aliases
    dim = dimension

    def is_solid(self) -> bool:
        """
        **Description:**
        Returns True if the polytope is solid (i.e. full-dimensional) and False
        otherwise.

        **Arguments:**
        None.

        **Returns:**
        The truth value of the polytope being full-dimensional.

        **Example:**
        We construct a polytope and check if it is solid.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-1,-1,-1,0]])
        p.is_solid()
        # False
        ```
        """
        return self.ambient_dim() == self.dim()

    @property
    def labels(self):
        """
        **Description:**
        Returns the point labels, in order.
        """
        return self._pts_order

    @property
    def label_origin(self):
        return self._label_origin

    @property
    def labels_vertices(self):
        if self._labels_vertices is None:
            # generate the labels
            self.vertices()
        return self._labels_vertices

    @property
    def labels_int(self):
        return self._labels_int

    @property
    def labels_facet(self):
        return self._labels_facet

    @property
    def labels_bdry(self):
        return self._labels_bdry

    @property
    def labels_codim2(self):
        return self._labels_codim2

    @property
    def labels_not_facet(self):
        return self._labels_not_facet

    def inequalities(self) -> np.ndarray:
        r"""
        **Description:**
        Returns the inequalities giving the hyperplane representation of the
        polytope. The inequalities are given in the form

        $c_0x_0 + \cdots + c_{d-1}x_{d-1} + c_d \geq 0$.

        Note, however, that equalities are not included.

        **Arguments:**
        None.

        **Returns:**
        The inequalities defining the polytope.

        **Example:**
        We construct a polytope and find the defining inequalities.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p.inequalities()
        # array([[ 4, -1, -1, -1,  1],
        #        [-1,  4, -1, -1,  1],
        #        [-1, -1,  4, -1,  1],
        #        [-1, -1, -1,  4,  1],
        #        [-1, -1, -1, -1,  1]])
        ```
        """
        return np.array(self._ineqs_input)

    # points
    # ======
    # internal/prep
    # -------------
    def _process_points(self, pts_input: ArrayLike, labels: ArrayLike = None) -> None:
        """
        **Description:**
        Internal function for processing input points. Should only be called
        once (in the initializer). Abstracted here to clarify logic.

        Sets:
            self._transl_vector, self._transf_mat_inv,
            self._poly_optimal,
            self._ineqs_optimal, self._ineqs_input,
            self._labels2optPts, self._labels2inputPts,
            self._pts_saturating,
            self._pts_order, and self._pts_indices

        **Arguments:**
        - `pts_input`: The points input from the user.
        - `labels`: The point labels input from the user.

        **Returns:**
        Nothing.
        """
        # Find 'optimal' representation
        # -----------------------------
        # translate if not full-dim (allows LLL-reduction)
        if self.is_solid():
            self._transl_vector = np.zeros(self.ambient_dim(), dtype=int)
        else:
            self._transl_vector = pts_input[0]
        pts_optimal = np.array(pts_input) - self._transl_vector

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

        last_default_label = -1
        for i in inds_sort:
            pt = tuple(pts_optimal_all[i])

            # find the label to use
            if (labels != []) and (pt in pts_optimal):
                label = labels[pts_optimal.index(pt)]
            else:
                label = last_default_label + 1

                while (label in self._labels2optPts) or (label in labels):
                    label += 1
                last_default_label = label

            # save it!
            self._labels2optPts[label] = pt
            self._pts_saturating[label] = saturating[i]
            nSat_to_labels[len(saturating[i])].append(label)

        # save order of labels
        self._pts_order = sum(nSat_to_labels[1:][::-1], nSat_to_labels[0])
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
            label: tuple(map(int, pt)) for label, pt in zip(self._pts_order, pts_input_all)
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

        self._labels_bdry = sum(nSat_to_labels[1:][::-1], [])
        self._labels_codim2 = sum(nSat_to_labels[2:][::-1], [])

        self._labels_not_facet = self._labels_int + self._labels_codim2

        # store as tuples
        self._labels_int = tuple(self._labels_int)
        self._labels_facet = tuple(self._labels_facet)
        self._labels_bdry = tuple(self._labels_bdry)
        self._labels_codim2 = tuple(self._labels_codim2)
        self._labels_not_facet = tuple(self._labels_not_facet)

    def _optimal_to_input(self, pts_opt: ArrayLike) -> np.array:
        """
        **Description:**
        We internally store the points in an 'optimal' representation
        (translated, LLL-reduced, ...). This eases computations. We will
        always want to return answers in original representation. This
        function performs the mapping.

        This is a kind of costly map so, whenever possible, use
            _optimalpts2labels
        and
            _labels2inputPts

        **Arguments:**
        - `pts_opt`: The points in the optimal representation.

        **Returns:**
        The points in the original representation.
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
        """
        **Description:**
        Returns the lattice points of the polytope.

        :::note
        Points are sorted so that interior points are first, and then the rest
        are arranged by decreasing number of saturated inequalities and
        lexicographically. For reflexive polytopes this is useful since the
        origin will be at index 0 and boundary points interior to facets will
        be last.
        :::

        **Arguments:**
        - `which`: Which points to return. Specified by a (list of) labels.
            NOT INDICES!!!
        - `optimal`: Whether to return the points in their optimal coordinates.
        - `as_indices`: Return the points as indices of the full list of points
            of the polytope.

        **Returns:**
        The list of lattice points of the polytope.

        **Aliases:**
        `pts`.

        **Example:**
        We construct a polytope and compute the lattice points. One can verify
        that the first point is the only interior point, and the last three
        points are the ones interior to facets. Thus it follows the
        aforementioned ordering.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.points()
        # array([[ 0,  0,  0,  0],
        #        [-1, -1, -6, -9],
        #        [ 0,  0,  0,  1],
        #        [ 0,  0,  1,  0],
        #        [ 0,  1,  0,  0],
        #        [ 1,  0,  0,  0],
        #        [ 0,  0, -2, -3],
        #        [ 0,  0, -1, -2],
        #        [ 0,  0, -1, -1],
        #        [ 0,  0,  0, -1]])
        ```
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
    pts_int = lambda self, as_indices=False: self.pts(
        which=self._labels_int, as_indices=as_indices
    )
    pts_bdry = lambda self, as_indices=False: self.pts(
        which=self._labels_bdry, as_indices=as_indices
    )
    pts_facet = lambda self, as_indices=False: self.pts(
        which=self._labels_facet, as_indices=as_indices
    )
    pts_codim2 = lambda self, as_indices=False: self.pts(
        which=self._labels_codim2, as_indices=as_indices
    )
    pts_not_facets = lambda self, as_indices=False: self.pts(
        which=self._labels_not_facet, as_indices=as_indices
    )

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
        inds = self.points(which=labels, as_indices=True)

        # get/return the indices
        if single_pt and len(inds):
            return inds[0]  # just return the single index
        else:
            return inds  # return a list of indices

    def vertices(self, optimal: bool = False, as_indices: bool = False) -> np.ndarray:
        """
        **Description:**
        Returns the vertices of the polytope.

        **Arguments:**
        - `optimal`: Whether to return the points in their optimal coordinates.
        - `as_indices`: Return the points as indices of the full list
            of points of the polytope.

        **Returns:**
        The list of vertices of the polytope.

        **Example:**
        We construct a polytope and find its vertices. We can see that they
        match the points that we used to construct the polytope.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p.vertices()
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [ 0,  0,  0,  1],
        #        [-1, -1, -1, -1]])
        ```
        """
        # return the answer if known
        if self._labels_vertices is not None:
            return self.pts(
                which=self._labels_vertices,
                optimal=optimal,
                as_indices=as_indices,
            )

        # calculate the answer
        if self.dim() == 0:
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
        """
        **Description:**
        Computes the faces of a polytope.

        :::note
        When the polytope is 4-dimensional it calls the slightly more optimized
        [`_faces4d()`](#_faces4d) function.
        :::

        **Arguments:**
        - `d`: Optional parameter that specifies the dimension of the desired
            faces.

        **Returns:**
        A tuple of [`PolytopeFace`](./polytopeface) objects of dimension d, if
        specified. Otherwise, a tuple of tuples of
        [`PolytopeFace`](./polytopeface) objects organized in ascending
        dimension.

        **Example:**
        We show that this function returns a tuple of 2-faces if `d` is set to
        2. Otherwise, the function returns all faces in tuples organized in
        ascending dimension. We verify that the first element in the tuple of
        2-faces is the same as the first element in the corresponding subtuple
        in the tuple of all faces.
        ```python {2,3}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        faces2d = p.faces(2)
        allfaces = p.faces()
        print(faces2d[0]) # Print first face in tuple of 2-faces
        # A 2-dimensional face of a 4-dimensional polytope in ZZ^4
        faces2d[0] is allfaces[2][0]
        # True
        ```
        """
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
                (PolytopeFace(self, self._labels_vertices, frozenset(), dim=self._dim),)
            )

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
                for f1, f2 in itertools.combinations(ineq2pts_prev.values(), 2):
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
            ineq2pts_prev = ineq2pts

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
        """
        **Description:**
        Computes the faces of a 4D polytope.

        :::note
        This function is a slightly more optimized version of the
        [`faces`](#faces) function. Typically the user should not call this
        function directly. Instead, it is only called by [`faces`](#faces) when
        the polytope is 4-dimensional.
        :::

        **Arguments:**
        None.

        **Returns:**
        A tuple of tuples of faces organized in ascending dimension.

        **Example:**
        We construct a 4D polytope and compute its faces. Since this function
        generally should not be directly used, we do this with the
        [`faces`](#faces) function.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        allfaces = p.faces() # the _faces4d function is used since it is a 4d polytope
        print(allfaces[1][0]) # Print the first face in the tuple of 1-dimensional faces
        # A 1-dimensional face of a 4-dimensional polytope in ZZ^4
        ```
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
        """
        **Description:**
        Returns the facets (codimension-1 faces) of the polytope.

        **Arguments:**
        None.

        **Returns:**
        A list of [`PolytopeFace`](./polytopeface) objects of codimension 1.

        **Example:**
        We construct a polytope and find its facets.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        facets = p.facets()
        ```
        """
        return self.faces(self.dim() - 1)

    # H-rep, dual
    # ===========
    def dual_polytope(self) -> "Polytope":
        r"""
        **Description:**
        Returns the dual polytope (also called polar polytope).  Only lattice
        polytopes are currently supported, so only duals of reflexive polytopes
        can be computed.

        :::note
        If $L$ is a lattice polytope, the dual polytope of $L$ is
        $ConvexHull(\{y\in \mathbb{Z}^n | x\cdot y \geq -1 \text{ for all } x \in L\})$.
        A lattice polytope is reflexive if its dual is also a lattice polytope.
        :::

        **Arguments:**
        None.

        **Returns:**
        The dual polytope.

        **Aliases:**
        `dual`, `polar_polytope`, `polar`.

        **Example:**
        We construct a reflexive polytope and find its dual. We then verify
        that the dual of the dual is the original polytope.
        ```python {2,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p_dual = p.dual_polytope()
        print(p_dual)
        # A 4-dimensional reflexive lattice polytope in ZZ^4
        p_dual_dual = p_dual.dual_polytope()
        p_dual_dual is p
        # True
        ```
        """
        # return answer if known
        if self._dual is not None:
            return self._dual

        # calculate the answer
        if not self.is_reflexive():
            raise NotImplementedError(
                "Duality of non-reflexive polytopes " + "is not supported."
            )

        pts = np.array(self._ineqs_input[:, :-1])
        self._dual = Polytope(pts, backend=self._backend)
        self._dual._dual = self
        return self._dual

    # aliases
    dual = dual_polytope
    polar_polytope = dual_polytope
    polar = dual_polytope

    def is_reflexive(self, allow_translations=True) -> bool:
        """
        **Description:**
        Returns True if the polytope is reflexive and False otherwise.

        **Arguments:**
        - `allow_translations`: Whether to allow the polytope to be translated.

        **Returns:**
        The truth value of the polytope being reflexive.

        **Example:**
        We construct a polytope and check if it is reflexive.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.is_reflexive()
        # True
        ```
        """
        # check if we know the answer
        if self._is_reflexive is not None:
            return self._is_reflexive

        # calculate the answer
        if self.is_solid():
            self._is_reflexive = all(
                c == 1 for c in self._ineqs_input[:, -1]
            )
        else:
            if allow_translations:
                p = Polytope(self.points(optimal=True))
            else:
                p = Polytope(lll_reduce(self.points())[:,-self.dim():])
            self._is_reflexive = p.is_reflexive()

        # return
        return self._is_reflexive

    # symmetries
    # ==========
    def automorphisms(
        self,
        square_to_one: bool = False,
        action: str = "right",
        as_dictionary: bool = False,
    ) -> "np.ndarray | dict":
        r"""
        **Description:**
        Returns the $SL^{\pm}(d,\mathbb{Z})$ matrices that leave the polytope
        invariant. These matrices act on the points by multiplication on the
        right.

        **Arguments:**
        - `square_to_one`: Flag that restricts to only matrices that square to
            the identity.
        - `action`: Flag that specifies whether the returned matrices act on
            the left or the right. This option is ignored when `as_dictionary`
            is set to True.
        - `as_dictionary`: Return each automphism as a dictionary that
            describes the action on the indices of the points.

        **Returns:**
        A list of automorphism matrices or dictionaries.

        **Example:**
        We construct a polytope, and find its automorphisms. We also check that
        one of the non-trivial automorphisms is indeed an automorphism by
        checking that it acts as a permutation on the vertices. We also show
        how to get matrices that act on the left, which are simply the
        transpose matrices, and we show how to get dictionaries that describe
        how the indices of the points transform.
        ```python {2,20,31,37}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        autos = p.automorphisms()
        a = autos[1]
        print(a)
        # [[ 1,  0,  0,  0],
        #  [-1, -1, -6, -9],
        #  [ 0,  0,  1,  0],
        #  [ 0,  0,  0,  1]])
        print(f"{p.vertices()}\n{p.vertices().dot(a)}") # Print vertices before and after applying the automorphism
        # [[ 1  0  0  0]
        #  [ 0  1  0  0]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]
        #  [-1 -1 -6 -9]]
        # [[ 1  0  0  0]
        #  [-1 -1 -6 -9]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]
        #  [ 0  1  0  0]]
        autos2 = p.automorphisms(square_to_one=True)
        a2 = autos2[1]
        print(f"{a2}\n{a2.dot(a2)}") # Print the automorphism and its square
        # [[ 1  0  0  0]
        #  [-1 -1 -6 -9]
        #  [ 0  0  1  0]
        #  [ 0  0  0  1]]
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]
        #  [0 0 0 1]]
        autos_left = p.automorphisms(square_to_one=True, action="left")
        print(autos_left[1].dot(p.vertices().T)) # The vertices are now columns
        # [[ 1 -1  0  0  0]
        # [ 0 -1  0  0  1]
        # [ 0 -6  1  0  0]
        # [ 0 -9  0  1  0]]
        autos_dict = p.automorphisms(as_dictionary=True)
        print(autos_dict)
        # [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 4, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 4, 2: 2, 3: 3, 4: 5, 5: 1, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 5, 2: 2, 3: 3, 4: 1, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9},
        #  {0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6, 7: 7, 8: 8, 9: 9}]
        ```
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
            for a in self._autos[0]:
                new_pts_tup = [tuple(pt) for pt in self.points().dot(a)]
                autos_dict.append(
                    {i: new_pts_tup.index(ii) for i, ii in enumerate(pts_tup)}
                )
            for a in self._autos[1]:
                new_pts_tup = [tuple(pt) for pt in self.points().dot(a)]
                autos2_dict.append(
                    {i: new_pts_tup.index(ii) for i, ii in enumerate(pts_tup)}
                )
            self._autos[2] = autos_dict
            self._autos[3] = autos2_dict

        # return
        if as_dictionary:
            return copy.deepcopy(self._autos[args_id])
        elif action == "left":
            return np.array([a.T for a in self._autos[args_id]])
        else:
            return np.array(self._autos[args_id])

    def normal_form(
        self, affine_transform: bool = False, backend: str = "palp"
    ) -> np.ndarray:
        r"""
        **Description:**
        Returns the normal form of the polytope as defined by Kreuzer-Skarke.

        **Arguments:**
        - `affine_transform`: Flag that determines whether to only use
            $SL^{\pm}(d,\mathbb{Z})$ transformations or also allow
            translations.
        - `backend`: Selects which backend to use. Options are "native", which
            uses native python code, or "palp", which uses PALP for the
            computation. There is a different convention for affine normal
            forms between the native algorithm and PALP, and PALP generally
            works better.

        **Returns:**
        The list of vertices in normal form.

        **Example:**
        We construct a polytope, and find its linear and affine normal forms.
        ```python {2,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.normal_form()
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [ 0,  0,  0,  1],
        #        [-9, -6, -1, -1]])
        p.normal_form(affine_transform=True)
        # array([[ 1,  0,  0,  0],
        #        [ 0,  1,  0,  0],
        #        [ 0,  0,  1,  0],
        #        [12, 17, 17, 18],
        #        [ 0,  0,  0,  0]])
        ```
        """
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
        for l in range(1, n_f - 1):
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
                for s in range(l, n_f):
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
                        prmb[n_p][0] = PGE(n_f, l + 1, s + 1).dot(prmb[n_p][0])
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
                            prmb[n_p][0] = PGE(n_f, l + 1, s + 1).dot(prmb[n_p][0])
                            n_p += 1
                            prmb[n_p] = copy.copy(prm[k])
                        else:
                            # We found a greater maximal value for first entry.
                            # It becomes our new reference:
                            l_r[0] = d1
                            prmb[n_p][0] = PGE(n_f, l + 1, s + 1).dot(prmb[n_p][0])
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
                            ][l]
                            if v[c] < v[j]:
                                prmb[s][1] = PGE(n_v, c + 1, j + 1).dot(prmb[s][1])
                        if ccf == 0:
                            # Set reference and carry on to next permutation
                            l_r[c] = PM[prmb[s][0].dot(range(n_f)), :][
                                :, prmb[s][1].dot(range(n_v))
                            ][l, c]
                            ccf = 1
                        else:
                            d1 = PM[prmb[s][0].dot(range(n_f)), :][
                                :, prmb[s][1].dot(range(n_v))
                            ][l, c]
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
                M = PM[prm[0][0].dot(range(n_f)), :][:, prm[0][1].dot(range(n_v))][l]
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
        prm = [p_c.dot(l[1]) for l in prm.values()]
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
        r"""
        **Description:**
        Returns True if the polytopes can be transformed into each other by an
        $SL^{\pm}(d,\mathbb{Z})$ transformation.

        **Arguments:**
        - `other`: The other polytope being compared.
        - `backend`: Selects which backend to use to compute the normal form.
            Options are "native", which uses native python code, or "palp",
            which uses PALP for the computation.

        **Returns:**
        The truth value of the polytopes being linearly equivalent.

        **Example:**
        We construct two polytopes and check if they are linearly equivalent.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[1,1,1,1]])
        p1.is_linearly_equivalent(p2)
        # True
        ```
        """
        our_normal_form = self.normal_form(
            affine_transform=False, backend=backend
        ).tolist()
        other_normal_form = other.normal_form(
            affine_transform=False, backend=backend
        ).tolist()

        return our_normal_form == other_normal_form

    def is_affinely_equivalent(self, other: "Polytope", backend: str = "palp") -> bool:
        """
        **Description:**
        Returns True if the polytopes can be transformed into each other by an
        integral affine transformation.

        **Arguments:**
        - `other`: The other polytope being compared.
        - `backend`: Selects which backend to use to compute the normal form.
            Options are "native", which uses native python code, or "palp",
            which uses PALP for the computation.

        **Returns:**
        The truth value of the polytopes being affinely equivalent.

        **Example:**
        We construct two polytopes and check if they are affinely equivalent.
        ```python {3}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,2],[-1,-1,-1,0]])
        p1.is_affinely_equivalent(p2)
        # True
        ```
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
        """
        **Description:**
        Constructs the list of point labels of the points that will be used in
        a triangulation.

        :::note
        Typically this function should not be called by the user. Instead, it
        is called by various other functions in the Polytope class.
        :::

        **Arguments:**
        - `include_points_interior_to_facets`: Whether to include points
            interior to facets from the triangulation. If not specified, it is
            set to False for reflexive polytopes and True otherwise.

        **Returns:**
        A tuple of the indices of the points that will be included in a
        triangulation

        **Example:**
        We construct triangulations in various ways. We use the
        [`triangulate`](#triangulate) function instead of using this function
        directly.
        ```python {2,5,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        t1 = p.triangulate()
        print(t1)
        # A fine, regular, star triangulation of a 4-dimensional point
        # configuration with 7 points in ZZ^4
        t2 = p.triangulate(include_points_interior_to_facets=True)
        print(t2)
        # A fine, regular, star triangulation of a 4-dimensional point
        # configuration with 10 points in ZZ^4
        t3 = p.triangulate(points=[1,2,3,4,5])
        print(t3)
        # A fine, regular, non-star triangulation of a 4-dimensional point
        # configuration with 5 points in ZZ^4
        ```
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
    ) -> Triangulation:
        """
        **Description:**
        Returns a single regular triangulation of the polytope.

        :::note
        When reflexive polytopes are used, it defaults to returning a fine,
        regular, star triangulation.
        :::

        **Arguments:**
        - `include_points_interior_to_facets`: Whether to include points
            interior to facets from the triangulation. If not specified, it is
            set to False for reflexive polytopes and True otherwise.
        - `points`: List of point labels that will be used. Note that if this
            option is used then the parameter
            `include_points_interior_to_facets` is ignored.
        - `make_star`: Indicates whether to turn the triangulation into a star
            triangulation by deleting internal lines and connecting all points
            to the origin, or equivalently by decreasing the height of the
            origin to be much lower than the rest. By default, this flag is set
            to true if the polytope is reflexive and neither heights or
            simplices are inputted. Otherwise, it is set to False.
        - `simplices`: A list of simplices specifying the triangulation. This
            is useful when a triangulation was previously computed and it needs
            to be used again. Note that the order of the points needs to be
            consistent with the order that the `Polytope` class uses.
        - `check_input_simplices`: Flag that specifies whether to check if the
            input simplices define a valid triangulation.
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
        A [`Triangulation`](./triangulation) object describing a triangulation
        of the polytope.

        **Example:**
        We construct a triangulation of a reflexive polytope and check that by
        default it is a fine, regular, star triangulation. We also try
        constructing triangulations with heights, input simplices, and using
        the other backends.
        ```python {2,4,6,8,10}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
        p.triangulate()
        # A fine, regular, star triangulation of a 4-dimensional polytope in
        # ZZ^4
        p.triangulate(heights=[-30,5,5,24,-19,-14,29])
        # A fine, regular, star triangulation of a 4-dimensional polytope in
        # ZZ^4
        p.triangulate(simplices=[[0,1,2,3,4],[0,1,2,3,5],[0,1,2,4,6],[0,1,2,5,6],[0,1,3,4,5],[0,1,4,5,6],[0,2,3,4,5],[0,2,4,5,6]])
        # A fine, regular, star triangulation of a 4-dimensional polytope in
        # ZZ^4
        p.triangulate(backend="qhull")
        # A fine, regular, star triangulation of a 4-dimensional polytope in
        # ZZ^4
        p.triangulate(backend="topcom")
        # A fine, regular, star triangulation of a 4-dimensional polytope in
        # ZZ^4
        ```
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
                make_star = self.is_reflexive()
            else:
                make_star = False

        if (not self.is_reflexive()) and (self._label_origin not in points):
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
    ) -> "generator | list":
        """
        **Description:**
        Constructs pseudorandom regular (optionally fine and star)
        triangulations of a given point set. This is done by picking random
        heights around the Delaunay heights from a Gaussian distribution.

        :::caution important
        This function produces random triangulations very quickly, but it does
        not produce a fair sample. When a fair sampling is required the
        [`random_triangulations_fair`](#random_triangulations_fair)
        function should be used.
        :::

        **Arguments:**
        - `N`: Number of desired unique triangulations. If not specified, it
            will generate as many triangulations as it can find until it has to
            retry more than `max_retries` times to obtain a new triangulation.
            This parameter is required when setting `as_list` to True.
        - `c`: A constant used as the standard deviation of the Gaussian
            distribution used to pick the heights. A larger `c` results in a
            wider range of possible triangulations, but with a larger fraction
            of them being non-fine, which slows down the process when
            `only_fine` is set to True.
        - `max_retries`: Maximum number of attempts to obtain a new
            triangulation before the process is terminated.
        - `make_star`: Converts the obtained triangulations into star
            triangulations. If not specified, defaults to True for reflexive
            polytopes, and False for other polytopes.
        - `only_fine`: Restricts to fine triangulations.
        - `include_points_interior_to_facets`: Whether to include points
            interior to facets from the triangulation. If not specified, it is
            set to False for reflexive polytopes and True otherwise.
        - `points`: List of point labels that will be used. Note that if this
            option is used then the parameter
            `include_points_interior_to_facets` is ignored.
        - `backend`: Specifies the backend used to compute the triangulation.
            The available options are "cgal" and "qhull".
        - `as_list`: By default this function returns a generator object, which
            is usually desired for efficiency. However, this flag can be set to
            True so that it returns the full list of triangulations at once.
        - `progress_bar`: Shows the number of triangulations obtained and
            progress bar. Note that this option is only available when
            returning a list instead of a generator.
        - `seed`: A seed for the random number generator. This can be used to
            obtain reproducible results.

        **Returns:**
        A generator of [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set to True.

        **Example:**
        We construct a polytope and find some random triangulations. The
        triangulations are obtained very quickly, but they are not a fair sample
        of the space of triangulations. For a fair sample, the
        [`random_triangulations_fair`](#random_triangulations_fair) function
        should be used.
        ```python {2,7}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        g = p.random_triangulations_fast()
        next(g) # Runs very quickly
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        next(g) # Keeps producing triangulations until it has trouble finding more
        # A fine, regular, star triangulation of a 4-dimensional point configuration with 106 points in ZZ^4
        rand_triangs = p.random_triangulations_fast(N=10, as_list=True) # Produces the list of 10 triangulations very quickly
        ```
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
            make_star = self.is_reflexive()
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
    ) -> "generator | list":
        r"""
        **Description:**
        Constructs pseudorandom regular (optionally star) triangulations of a
        given point set. Implements Algorithm \#3 from the paper
        *Bounding the Kreuzer-Skarke Landscape* by Mehmet Demirtas, Liam
        McAllister, and Andres Rios-Tascon.
        [arXiv:2008.01730](https://arxiv.org/abs/2008.01730)

        This is a Markov chain Monte Carlo algorithm that involves taking
        random walks inside the subset of the secondary fan corresponding to
        fine triangulations and performing random flips. For details, please
        see Section 4.1 in the paper.

        :::note notes
        - By default this function tries to guess reasonable parameters for the
            polytope that is being used. However, it may take some
            trial-and-error to find optimal parameters that produce a good
            sampling and prevent the algorithm from getting stuck.
        - This function is designed mainly for large polytopes where sampling
            triangulations is challenging. When small polytopes are used it is
            likely to get stuck or not produce an optimal set.
        :::

        **Arguments:**
        - `N`: Number of desired unique triangulations. If not specified, it
            will generate as many triangulations as it can find until it has to
            retry more than `max_retries` times to obtain a new triangulation.
            This parameter is required when setting `as_list` to True.
        - `n_walk`: Number of hit-and-run steps per triangulation. Default is
            n_points//10+10.
        - `n_flip`: Number of random flips performed per triangulation.
            Default is n_points//10+10.
        - `initial_walk_steps`: Number of hit-and-run steps to take before
            starting to record triangulations. Small values may result in a
            bias towards Delaunay-like triangulations. Default is
            2*n_pts//10+10.
        - `walk_step_size`: Determines the size of random steps taken in the
            secondary fan. The algorithm may stall if too small.
        - `max_steps_to_wall`: Maximum number of steps to take towards a wall
            of the subset of the secondary fan that correspond to fine
            triangulations. If a wall is not found, a new random direction is
            selected. Setting this to be very large (>100) reduces performance.
            If this is set to be too low, the algorithm may stall.
        - `fine_tune_steps`: Number of steps to determine the location of a
            wall. Decreasing improves performance, but might result in biased
            samples.
        - `max_retries`: Maximum number of attempts to obtain a new
            triangulation before the process is terminated.
        - `make_star`: Converts the obtained triangulations into star
            triangulations. If not specified, defaults to True for reflexive
            polytopes, and False for other polytopes.
        - `include_points_interior_to_facets`: Whether to include points
            interior to facets from the triangulation. If not specified, it is
            set to False for reflexive polytopes and True otherwise.
        - `points`: List of point labels that will be used. Note that if this
            option is used then the parameter
            `include_points_interior_to_facets` is ignored.
        - `backend`: Specifies the backend used to compute the triangulation.
            The available options are "cgal" and "qhull".
        - `as_list`: By default this function returns a generator object, which
            is usually desired for efficiency. However, this flag can be set
            to True so that it returns the full list of triangulations at once.
        - `progress_bar`: Shows number of triangulations obtained and progress
            bar. Note that this option is only available when returning a list
            instead of a generator.
        - `seed`: A seed for the random number generator. This can be used to
            obtain reproducible results.

        **Returns:**
        A generator of [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set to True.

        **Example:**
        We construct a polytope and find some random triangulations. The
        computation takes considerable time, but they should be a fair sample
        from the full set of triangulations (if the parameters are chosen
        correctly). For (some) machine learning purposes or when the fairness
        of the sample is not crucial, the
        [`random_triangulations_fast`](#random_triangulations_fast) function
        should be used instead.
        ```python {2,7}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
        g = p.random_triangulations_fast()
        next(g) # Takes a long time (around a minute)
        # A fine, regular, star triangulation of a 4-dimensional point
        # configuration with 106 points in ZZ^4
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
            make_star = self.is_reflexive()
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
    ) -> "generator | list":
        """
        **Description:**
        Computes all triangulations of the polytope using TOPCOM. There is the
        option to only compute fine, regular or fine triangulations.

        :::caution warning
        Polytopes with more than around 15 points usually have too many
        triangulations, so this function may take too long or run out of
        memory.
        :::

        **Arguments:**
        - `only_fine`: Restricts to only fine triangulations.
        - `only_regular`: Restricts to only regular triangulations.
        - `only_star`: Restricts to only star triangulations. When not
            specified it defaults to True for reflexive polytopes and False
            otherwise.
        - `star_origin`: The index of the point that will be used as the star
            origin. If the polytope is reflexive this is set to 0, but
            otherwise it must be specified.
        - `include_points_interior_to_facets`: Whether to include points
            interior to facets from the triangulation.
        - `points`: List of point labels that will be used. Note that if this
            option is used then the parameter
            `include_points_interior_to_facets` is ignored.
        - `backend`: The optimizer used to check regularity computation. The
            available options are the backends of the
            [`is_solid`](./cone#is_solid) function of the [`Cone`](./cone)
            class. If not specified, it will be picked automatically. Note that
            TOPCOM is not used to check regularity since it is much slower.
        - `as_list`: By default this function returns a generator object, which
            is usually desired for efficiency. However, this flag can be set to
            True so that it returns the full list of triangulations at once.
        - `raw_output`: Return the triangulations as lists of simplices instead
            of as Triangulation objects.

        **Returns:**
        A generator of [`Triangulation`](./triangulation) objects, or a list of
        [`Triangulation`](./triangulation) objects if `as_list` is set to True.

        **Example:**
        We construct a polytope and find all of its triangulations. We try
        picking different restrictions and see how the number of triangulations
        changes.
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
        if len(self.points()) == self.dim() + 1:
            # simplex... trivial
            triangs = None
            if raw_output:
                triangs = [self.points(as_indices=True)[None, :]]
            else:
                triangs = [Triangulation(self, self.labels)]

            if as_list:
                return triangs

            def gen():
                yield from triangs

            return gen()

        if only_star is None:
            only_star = self.is_reflexive()
        if only_star and star_origin is None:
            if self.is_reflexive():
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
    def hpq(self, p: int, q: int, lattice: str) -> int:
        """
        **Description:**
        Returns the Hodge number $h^{p,q}$ of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note notes
        - Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `p`: The holomorphic index of the Dolbeault cohomology of interest.
        - `q`: The anti-holomorphic index of the Dolbeault cohomology of
            interest.
        - `lattice`: Specifies the lattice on which the polytope is defined.
            Options are "N" and "M".

        **Returns:**
        The Hodge number $h^{p,q}$ of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and check some Hodge numbers of the associated
        hypersurfaces.
        ```python {2,4,6,8}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.hpq(0,0,lattice="N")
        # 1
        p.hpq(0,1,lattice="N")
        # 0
        p.hpq(1,1,lattice="N")
        # 2
        p.hpq(1,2,lattice="N")
        # 272
        ```
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
            raise ValueError("Lattice must be specified. " 'Options are: "N" or "M".')

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

    h11 = lambda self, lattice: self.hpq(1, 1, lattice=lattice)
    h12 = lambda self, lattice: self.hpq(1, 2, lattice=lattice)
    h21 = h12
    h13 = lambda self, lattice: self.hpq(1, 3, lattice=lattice)
    h31 = h13
    h22 = lambda self, lattice: self.hpq(2, 2, lattice=lattice)

    def chi(self, lattice: str) -> int:
        """
        **Description:**
        Computes the Euler characteristic of the Calabi-Yau obtained as the
        anticanonical hypersurface in the toric variety given by a
        desingularization of the face or normal fan of the polytope when the
        lattice is specified as "N" or "M", respectively.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice`: Specifies the lattice on which the polytope is defined.
            Options are "N" and "M".

        **Returns:**
        The Euler characteristic of the arising Calabi-Yau manifold.

        **Example:**
        We construct a polytope and compute the Euler characteristic of the
        associated hypersurfaces.
        ```python {2,4}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.chi(lattice="N")
        # -540
        p.chi(lattice="M")
        # 540
        ```
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
        """
        **Description:**
        Returns True if the Calabi-Yau hypersurface arising from this polytope
        is favorable (i.e. all Kahler forms descend from Kahler forms on the
        ambient toric variety) and False otherwise.

        :::note
        Only reflexive polytopes of dimension 2-5 are currently supported.
        :::

        **Arguments:**
        - `lattice`: Specifies the lattice on which the polytope is
            defined. Options are "N" and "M".

        The truth value of the polytope being favorable.

        **Example:**
        We construct two reflexive polytopes and find whether they are
        favorable when considered in the N lattice.
        ```python {3,5}
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-3,-6]])
        p1.is_favorable(lattice="N")
        # True
        p2.is_favorable(lattice="N")
        # False
        ```
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
        """
        **Description:**
        Computes the GLSM charge matrix of the theory resulting from this
        polytope.

        **Arguments:**
        - `include_origin`: Indicates whether to use the origin in the
            calculation. This corresponds to the inclusion of the canonical
            divisor.
        - `include_points_interior_to_facets`: By default only boundary points
            not interior to facets are used. If this flag is set to true then
            points interior to facets are also used.
        - `points`: The list of indices of the points that will be used. Note
            that if this option is used then the parameters `include_origin`
            and `include_points_interior_to_facets` are ignored.
        - `integral`: Indicates whether to find an integral basis for the
            columns of the GLSM charge matrix. (i.e. so that remaining columns
            can be written as an integer linear combination of the basis
            elements.)

        **Returns:**
        The GLSM charge matrix.

        **Example:**
        We construct a polytope and find the GLSM charge matrix with different
        parameters.
        ```python {2,5,8,11}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_charge_matrix()
        # array([[-18,   1,   9,   6,   1,   1,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1]])
        p.glsm_charge_matrix().dot(p.points_not_interior_to_facets()) # By definition this product must be zero
        # array([[0, 0, 0, 0],
        #        [0, 0, 0, 0]])
        p.glsm_charge_matrix(include_origin=False) # Excludes the canonical divisor
        # array([[1, 9, 6, 1, 1, 0],
        #        [0, 3, 2, 0, 0, 1]])
        p.glsm_charge_matrix(include_points_interior_to_facets=True) # Includes points interior to facets
        # array([[-18,   1,   9,   6,   1,   1,   0,   0,   0,   0],
        #        [ -6,   0,   3,   2,   0,   0,   1,   0,   0,   0],
        #        [ -4,   0,   2,   1,   0,   0,   0,   1,   0,   0],
        #        [ -3,   0,   1,   1,   0,   0,   0,   0,   1,   0],
        #        [ -2,   0,   1,   0,   0,   0,   0,   0,   0,   1]])
        ```
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
            if (min(pts_ind) < 0) or (max(pts_ind) > self.points().shape[0]):
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
            linrel = np.insert(linrel, 0, np.ones(linrel.shape[1], dtype=int), axis=0)
            good_exclusions = 0
            basis_exc = []
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
                    indices = np.argsort(norms)
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                elif n_try == 3:
                    indices[:] = np.array([0] + list(range(1, linrel.shape[1]))[::-1])
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                elif n_try > 3:
                    if n_try == 4:
                        np.random.seed(1337)
                    np.random.shuffle(indices[1:])
                    indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])

                for ctr in range(np.prod(linrel.shape) + 1):
                    found_good_basis = True
                    ctr += 1
                    if ctr > 0:
                        st = max([good_exclusions, 1])
                        indices[st:] = np.roll(indices[st:], -1)
                        indices[: linrel.shape[0]] = np.sort(indices[: linrel.shape[0]])
                    linrel_rand = np.array(linrel[:, indices])
                    try:
                        linrel_hnf = fmpz_mat(linrel_rand.tolist()).hnf()
                    except:
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
                if integral:
                    glsm[:, nb] = -glsm.dot(linrel[i]) // ii
                else:
                    glsm[i, :] *= ii
                    glsm[:, nb] = -glsm.dot(linrel[i])
        else:  # Non-integral basis
            pts = self.points()[list(pts_ind)[1:]]  # Exclude the origin
            pts_norms = [np.linalg.norm(p, 1) for p in pts]
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
        """
        **Description:**
        Computes the linear relations of the GLSM charge matrix.

        **Arguments:**
        - `include_origin`: Indicates whether to use the origin in the
            calculation. This corresponds to the inclusion of the canonical
            divisor.
        - `include_points_interior_to_facets`: By default only boundary points
            not interior to facets are used. If this flag is set to true then
            points interior to facets are also used.
        - `points`: The list of indices of the points that will be used. Note
            that if this option is used then the parameters `include_origin`
            and `include_points_interior_to_facets` are ignored.
        - `integral`: Indicates whether to find an integral basis for the
            columns of the GLSM charge matrix. (i.e. so that remaining columns
            can be written as an integer linear combination of the basis
            elements.)

        **Returns:**
        A matrix of linear relations of the columns of the GLSM charge matrix.

        **Example:**
        We construct a polytope and find its GLSM charge matrix and linear
        relations with different parameters.
        ```python {2,8,14,19}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_linear_relations()
        # array([[ 1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3],
        #        [ 0,  6,  0, -1,  0,  0,  2],
        #        [ 0,  1,  0,  0, -1,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0]])
        p.glsm_linear_relations().dot(p.glsm_charge_matrix().T) # By definition this product must be zero
        # array([[0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0],
        #        [0, 0]])
        p.glsm_linear_relations(include_origin=False) # Excludes the canonical divisor
        # array([[ 9, -1,  0,  0,  0,  3],
        #        [ 6,  0, -1,  0,  0,  2],
        #        [ 1,  0,  0, -1,  0,  0],
        #        [ 1,  0,  0,  0, -1,  0]])
        p.glsm_linear_relations(include_points_interior_to_facets=True) # Includes points interior to facets
        # array([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #        [ 0,  9, -1,  0,  0,  0,  3,  2,  1,  1],
        #        [ 0,  6,  0, -1,  0,  0,  2,  1,  1,  0],
        #        [ 0,  1,  0,  0, -1,  0,  0,  0,  0,  0],
        #        [ 0,  1,  0,  0,  0, -1,  0,  0,  0,  0]])
        ```
        """
        if points is not None:
            pts_ind = tuple(set(list(points) + [0]))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
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
        """
        **Description:**
        Computes a basis of columns of the GLSM charge matrix.

        **Arguments:**
        - `include_origin`: Indicates whether to use the origin in the
            calculation. This corresponds to the inclusion of the canonical
            divisor.
        - `include_points_interior_to_facets`: By default only boundary points
            not interior to facets are used. If this flag is set to true then
            points interior to facets are also used.
        - `points`: The list of indices of the points that will be used. Note
            that if this option is used then the parameters `include_origin`
            and `include_points_interior_to_facets` are ignored. Also, note
            that the indices returned here will be the indices of the sorted
            list of points.
        - `integral`: Indicates whether to find an integral basis for the
            columns of the GLSM charge matrix. (i.e. so that remaining columns
            can be written as an integer linear combination of the basis
            elements.)

        **Returns:**
        A list of column indices that form a basis.

        **Example:**
        We construct a polytope, find its GLSM charge matrix and a basis of
        columns.
        ```python {3,6}
        import numpy as np
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.glsm_basis()
        # array([1, 6])
        glsm = p.glsm_charge_matrix()
        np.linalg.matrix_rank(glsm) == np.linalg.matrix_rank(glsm[:,p.glsm_basis()]) # This shows that the columns form a basis
        # True
        ```
        """
        if points is not None:
            pts_ind = tuple(set(list(points) + [0]))
            if min(pts_ind) < 0 or max(pts_ind) > self.points().shape[0]:
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
        """
        **Description:**
        Returns the Minkowski sum of the two polytopes.

        **Arguments:**
        - `other`: The other polytope used for the Minkowski sum.

        **Returns:**
        The Minkowski sum.

        **Example:**
        We construct two polytopes and compute their Minkowski sum.
        ```python {3}
        p1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
        p2 = Polytope([[0,0,1],[0,0,-1]])
        p1.minkowski_sum(p2)
        # A 3-dimensional reflexive lattice polytope in ZZ^3
        ```
        """
        points = {tuple(sum(verts))
                  for verts in itertools.product(self.vertices(),
                                                 other.vertices())}
        return Polytope(list(points))

    def volume(self) -> int:
        """
        **Description:**
        Returns the volume of the polytope.

        :::important
        By convention, the standard simplex has unit volume. To get the more
        typical Euclidean volume it must be multiplied by $d!$.
        :::

        **Arguments:**
        None.

        **Returns:**
        The volume of the polytope.

        **Example:**
        We construct a standard simplex and a cube, and find their volumes.
        ```python {3,5}
        p1 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        p2 = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
        p1.volume()
        # 1
        p2.volume()
        # 6
        ```
        """
        # calculate the answer if not known
        if self._volume is None:
            if self.dim() == 0:
                self._volume = 0
            elif self.dim() == 1:
                self._volume = max(self.points(optimal=True)) - min(
                    self.points(optimal=True)
                )
            else:
                self._volume = ConvexHull(self.points(optimal=True)).volume
                self._volume *= math.factorial(self.dim())
                self._volume = int(round(self._volume))

        # return
        return self._volume

    def find_2d_reflexive_subpolytopes(self) -> list["Polytope"]:
        """
        **Description:**
        Use the algorithm by Huang and Taylor described in
        [1907.09482](https://arxiv.org/abs/1907.09482) to find 2D reflexive
        subpolytopes in 4D polytopes.

        **Arguments:**
        None.

        **Returns:**
        The list of 2D reflexive subpolytopes.

        **Example:**
        We construct a polytope and find its 2D reflexive subpolytopes.
        ```python {2}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        p.find_2d_reflexive_subpolytopes()
        # [A 2-dimensional lattice polytope in ZZ^4]
        ```
        """
        if not self.is_reflexive() or self.dim() != 4:
            raise NotImplementedError("Only 4D reflexive polytopes are supported.")
        pts = self.points()
        dual_vert = self.dual().vertices()
        # Construct the sets S_i by finding the maximum dot product with dual
        # vertices
        S_i = [[]] * 3
        for p in pts:
            m = max(p.dot(v) for v in dual_vert)
            if m in (1, 2, 3):
                S_i[m - 1].append(tuple(p))
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
        """
        **Description:**
        Computes the nef partitions of the polytope using PALP.

        :::note
        This is currently an experimental feature and may change significantly
        in future versions.
        :::

        **Arguments:**
        - `keep_symmetric`: Keep symmetric partitions related by lattice
            automorphisms.
        - `keep_products`: Keep product partitions corresponding to complete
            intersections being direct products.
        - `keep_projections`: Keep projection partitions, i.e. partitions where
            one of the parts consists of a single vertex.
        - `codim`: The number of parts in the partition or, equivalently, the
            codimension of the complete intersection Calabi-Yau.
        - `compute_hodge_numbers`: Indicates whether Hodge numbers of the CICY
            are computed.
        - `return_hodge_numbers`: Indicates whether to return the Hodge numbers
            along with the nef partitions. They are returned in a separate
            tuple and they are ordered as in the Hodge diamond from top to
            bottom and left to right.

        **Returns:**
        The nef partitions of the polytope. If return_hodge_numbers is set to
        True then two tuples are returned, one with the nef partitions and one
        with the corresponding Hodge numbers.

        **Example:**
        We construct a tesseract and find the 2- and 3-part nef partitions.
        ```python {2,5}
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
        nef_part_2 = p.nef_partitions() # Default codimension is 2
        print(nef_part_2[0]) # Print the first of the nef partitions
        # ((5, 2, 3, 4), (8, 7, 6, 1))
        nef_part_3 = p.nef_partitions(codim=3) # Codimension 3
        print(nef_part_3[0]) # Print the first of the nef partitions
        # ((6, 5, 3), (2, 4), (8, 7, 1))
        ```
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
            hodge_nums = [tuple(tuple(part) for part in partition[1]) for partition in results]
            nef_parts = (nef_parts, hodge_nums)
            
        self._nef_parts[args_id] = nef_parts
        return (
            self._nef_parts.get(args_id)
            if return_hodge_numbers or not compute_hodge_numbers
            else self._nef_parts.get(args_id)[0]
        )

    def is_trilayer(self, return_anticanon=False):
        """
        Check if a polytope is 'trilayer'.
        """
        glsm_vert = np.array(fmpz_mat(self.vertices().T.tolist()).nullspace()[0].transpose().tolist(), dtype=int)[:-4]
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
    """
    **Description:**
    Generate the H-representation of a polytope, given the V-representation.
    I.e., map points/vertices to hyperplanes inequalities.

    The o inequalities are in the form
        c_0 * x_0 + ... + c_{d-1} * x_{d-1} + c_d >= 0

    **Arguments:**
    - `pts`: The input points. Each row is a point.
    - `backend`: The backend to use. Currently, support "ppl", "qhull", and
        "palp".

    **Returns:**
    The hyperplane inequalities in the form
        c_0 * x_0 + ... + c_{d-1} * x_{d-1} + c_d >= 0
    and, depending on backend/dimension, the formal convex hull of the points.
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
    """
    **Description:**
    Computes the lattice points contained in conv(pts), along with the indices
    of the hyperplane inequalities that they saturate.

    **Arguments:**
    - `pts`: A list of points spanning the hull.
    - `ineqs`: Hyperplane inqualities defining the hull. Same format as
        output by poly_v_to_h
    - `dim`: The dimension of the hull.
    - `backend`: The backend to use. Either "palp" or defaults to native.

    **Returns:**
    An array of all lattice points (the rows).
    A list of sets of all inequalities each lattice point saturates.
    """
    # check inputs
    if isinstance(pts_in, list) and isinstance(pts_in[0], tuple):
        pts = pts_in
    else:
        pts = [tuple(pt) for pt in pts_in]

    # fill in missing inputs
    if dim is None:
        dim = np.linalg.matrix_rank([list(pt) + [1] for pt in pts]) - 1

    if backend is None:
        if 1 <= dim <= 4:
            backend = "ppl"
        else:
            backend = "palp"

    if dim == 0:  # 0-dimensional polytopes are finicky
        backend = "palp"

    if ineqs is None:
        ineqs, _ = poly_v_to_h(pts, backend)

    # split computation by backend
    if backend == "palp":
        if dim == 0:
            # PALP cannot handle 0-dimensional polytopes
            pts_all = [pts[0]]
            facet_ind = [frozenset([0])]
        else:
            p = pypalp.Polytope(pts)
            pts_all = p.points()

            # find inequialities each point saturates
            facet_ind = [
                frozenset(
                    i for i, ii in enumerate(ineqs) if ii[:-1].dot(pt) + ii[-1] == 0
                )
                for pt in pts_all
            ]

    # Otherwise use the algorithm by Volker Braun.
    # This is redistributed under GNU General Public License version 2+.
    #
    # The original code can be found at
    # https://github.com/sagemath/sage/blob/master/src/sage/geometry/integral_points.pxi
    else:
        # Find bounding box and sort by decreasing dimension size
        box_min = np.min(pts, axis=0)
        box_max = np.max(pts, axis=0)

        # Sort box bounds
        diameter_index = np.argsort(box_max - box_min)[::-1]
        box_min = box_min[diameter_index]
        box_max = box_max[diameter_index]

        # Construct the inverse permutation
        orig_dict = {j: i for i, j in enumerate(diameter_index)}
        orig_perm = [orig_dict[i] for i in range(dim)]

        # Inequalities must also have their coordinates permuted
        ineqs = ineqs.copy()
        ineqs[:, :-1] = ineqs[:, diameter_index]

        # Find all lattice points and apply the inverse permutation
        pts_all = []
        facet_ind = []
        p = np.array(box_min)

        while True:
            tmp_v = ineqs[:, 1:-1].dot(p[1:]) + ineqs[:, -1]

            # Find the lower bound for the allowed region
            for i_min in range(box_min[0], box_max[0] + 1, 1):
                if all(i_min * ineqs[:, 0] + tmp_v >= 0):
                    break

            # Find the upper bound for the allowed region
            for i_max in range(box_max[0], i_min - 1, -1):
                if all(i_max * ineqs[:, 0] + tmp_v >= 0):
                    break
            else:
                i_max -= 1

            # The points i_min .. i_max are contained in the polytope
            for i in range(i_min, i_max + 1):
                p[0] = i
                pts_all.append(np.array(p)[orig_perm])

                saturated = frozenset(
                    j for j in range(len(tmp_v)) if i * ineqs[j, 0] + tmp_v[j] == 0
                )
                facet_ind.append(saturated)

            # Increment the other entries in p to move on to next loop
            inc = 1
            if dim == 1:
                break

            while p[inc] == box_max[inc]:
                p[inc] = box_min[inc]
                inc += 1
                if inc == dim:
                    break
            else:
                p[inc] += 1
                continue
            break

    # return
    return pts_all, facet_ind



def is_reflexive_barebones(points: "ArrayLike",
                           backend: str = 'qhull') -> bool:
    """
    **Description:**
    Minimal code to check if conv(points) is reflexive.

    **Arguments:**
    - `points`: The points defining the hull.
    - `backend`: The backend to use. See poly_v_to_h.

    **Returns:**
    Whether conv(points) is reflexive
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
