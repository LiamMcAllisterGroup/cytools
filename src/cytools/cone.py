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
# Description:  This module contains tools designed to perform cone
#               computations.
# -----------------------------------------------------------------------------
"""Rational polyhedral cones.

This module provides the [`Cone`][cytools.cone.Cone] class, which handles
computations with rational polyhedral cones such as duality, extremal rays,
faces, interior and lattice points, and Hilbert bases, as well as a few
standalone helper functions used by it.
"""

from __future__ import annotations

# 'standard' imports
from ast import literal_eval
from collections.abc import Iterable
from copy import deepcopy
from fractions import Fraction
import functools
import importlib
import itertools
import joblib
from multiprocessing import cpu_count
import os
import random
import shutil
import string
import subprocess
import warnings

# 3rd party imports
from flint import fmpz_mat, fmpz, fmpq
import numpy as np
import highspy
from cytools._ppl import ppl  # ppl, with its FPU rounding side effect undone

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# the default extremal-ray backend. Optional: extremal_rays falls back to the
# legacy method, with a warning, when it is missing
try:
    import extremalrays as _extremalrays

    _HAS_EXTREMALRAYS = True
except ImportError:
    _extremalrays = None
    _HAS_EXTREMALRAYS = False
import qpsolvers
from scipy import sparse
from scipy.optimize import linprog, nnls
import latticepts

# CYTools imports
from cytools import config
from cytools import utils


def _rank(arr) -> int:
    """Return the rank of a set of rays.

    Unlike `np.linalg.matrix_rank`, this also accepts an empty `(0, d)` array,
    which is how a cone with no rays is represented, and returns 0 for it.

    Args:
        arr (ArrayLike): The rays, one per row.

    Returns:
        int: The dimension of the linear span of the rays.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    return int(np.linalg.matrix_rank(arr))


class _LazyModule:
    """A stand-in for a module that is only imported when it is first used.

    It forwards every attribute access to the real module, so `mod.attr` works
    exactly as it would after a plain `import`.

    This is used for the OR-Tools solvers, which are only needed by a handful
    of methods but which are expensive to import (`ortools.sat` alone pulls in
    pandas, costing ~1 s of `import cytools` time).

    Args:
        name (str): The fully-qualified name of the module to import lazily.
    """

    def __init__(self, name: str):
        self._name = name
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)

    def __repr__(self) -> str:
        state = "loaded" if self._module is not None else "not loaded"
        return f"<lazy module {self._name!r} ({state})>"


cp_model = _LazyModule("ortools.sat.python.cp_model")
pywraplp = _LazyModule("ortools.linear_solver.pywraplp")


class Cone:
    r"""Rational polyhedral cone.

    This class handles all computations relating to rational polyhedral cones,
    such as cone duality and extremal ray computations. It is mainly used for
    the study of Kähler and Mori cones. A cone can be specified either by a set
    of generating rays $r_i$, in which case it is the set of non-negative
    linear combinations $\sum_i \lambda_i r_i$ with $\lambda_i\geq 0$, or by a
    set of inward-pointing hyperplane normals $h_j$, in which case it is the
    set of points $x$ with $h_j\cdot x\geq 0$ for all $j$. The other
    description is computed on demand.

    The input rays or hyperplane normals are stored as integer vectors.
    Floating-point input is converted by dividing each row by the (numerical)
    GCD of its entries and rounding. This class is also available at the root
    of the package, so it can be imported with `from cytools import Cone`.

    Args:
        rays (ArrayLike | None): The rays that generate the cone, one per row.
            If it is not specified, then `hyperplanes` must be specified.
        hyperplanes (ArrayLike | None): The inward-pointing normals to the
            hyperplanes that define the cone, one per row. If it is not
            specified, then `rays` must be specified. An empty list defines the
            whole ambient space.
        parse_inputs (bool): Whether to parse and normalize the input. If
            false, `hyperplanes` must be a 2D NumPy array of integers, and it is
            stored as is. This is only supported when constructing a cone from
            hyperplanes.
        check (bool): Whether to divide each integer input row by the GCD of
            its entries, dropping rows of zeros (with a warning). Floating-point
            input is always normalized this way. Recommended when constructing
            a cone directly.
        copy (bool): Whether to copy the input data. If false, the input
            array may be used directly when no conversion is needed.
            Recommended.
        ambient_dim (int | None): The dimension of the ambient space. It is
            only used when `hyperplanes` is empty, in which case it must be
            specified unless it can be inferred from the shape of the input.

    Raises:
        ValueError: If not exactly one of `rays` or `hyperplanes` is specified,
            if the input is not a 2D matrix, if the ambient dimension is zero,
            or if `hyperplanes` is empty and `ambient_dim` is missing or
            inconsistent with its shape.
        NotImplementedError: If `parse_inputs` is false and `rays` are given,
            or if the input has an unsupported data type.
        Exception: If arbitrary-precision (`fmpz` or `fmpq`) input is given
            without enabling the experimental features.

    Warning:
        This class is primarily tailored to pointed (i.e. strongly convex)
        cones. There are a few computations, such as finding extremal rays,
        that may produce some unexpected results when working with non-pointed
        cones.

    Examples:
        We construct a cone in two different ways, first from a list of rays and
        then from a list of hyperplane normals, and verify that the two inputs
        result in the same cone.

        >>> from cytools import Cone
        >>> c1 = Cone([[0,1],[1,1]])  # equivalently, Cone(rays=[[0,1],[1,1]])
        >>> c2 = Cone(hyperplanes=[[1,0],[-1,1]])
        >>> c1 == c2
        True
    """

    def __init__(
        self,
        rays: "ArrayLike" = None,
        hyperplanes: "ArrayLike" = None,
        parse_inputs: bool = True,
        check: bool = True,
        copy: bool = True,
        ambient_dim: int = None,
    ):
        # check whether rays or hyperplanes were input
        if not ((rays is None) ^ (hyperplanes is None)):
            raise ValueError(
                'Exactly one of "rays" and "hyperplanes" must be specified.'
            )

        # parse empty hyperplanes
        if (rays is None) and (len(hyperplanes) == 0):
            # the input may be a plain (empty) list, so promote it to an array
            # before inspecting its shape
            hyperplanes = np.asarray(hyperplanes)

            # check if ambient dim is inferrable from hyperplanes
            if (len(hyperplanes.shape) > 1) and (hyperplanes.shape[1] != 0):
                # yes inferrable - ensure no conflicts in specification
                if (ambient_dim is not None) and (ambient_dim != hyperplanes.shape[1]):
                    raise ValueError(
                        f"Specified ambient dim = {ambient_dim} doesn't match inferrable shape from hyperplanes = {hyperplanes.shape[1]}..."
                    )

                ambient_dim = hyperplanes.shape[1]
            else:
                if ambient_dim is None:
                    raise ValueError(
                        "Must specify ambient dimension if len(hyperplanes)=0."
                    )

            # move to a ray representation
            hyperplanes = None
            rays = []
            for i in range(ambient_dim):
                # add e_i and -e_i
                rays.append([int(i == j) for j in range(ambient_dim)])
                rays.append([-int(i == j) for j in range(ambient_dim)])

        # minimal work if we don't parse the data
        if not parse_inputs:
            if rays is None:
                data_name = "hyperplane(s)"
                self._rays_were_input = False
                self._rays = None
                data = hyperplanes
            else:
                raise NotImplementedError(
                    "Currently, parse_inputs is required if rays are input..."
                )

            # initialize other variables
            self.clear_cache()
            self._ambient_dim = data.shape[1]
            self._dim = None

            if self._rays_were_input:
                self._rays = data
            else:
                self._hyperplanes = data
            return

        # standard case
        if rays is None:
            data_name = "hyperplane(s)"
            self._rays_were_input = False
            self._rays = None
            if copy:
                data = np.array(hyperplanes)
            else:
                data = np.asarray(hyperplanes)
        else:
            data_name = "ray(s)"
            self._rays_were_input = True
            self._hyperplanes = None
            if copy:
                data = np.array(rays)
            else:
                data = np.asarray(rays)

        # initialize other variables
        self.clear_cache()

        # basic data-checking
        if len(data.shape) != 2:
            raise ValueError(f"Input {data_name} must be a 2D matrix.")
        elif data.shape[1] < 1:
            raise ValueError("Zero-dimensional cones are not supported.")
        # elif data.shape[0]<1:
        #    raise ValueError(f"At least one {data_name} is required.")

        self._ambient_dim = data.shape[1]

        if len(data):
            # check size of coordinates
            if np.min(data) <= -100000000000000:
                warnings.warn(
                    f"Extremely small coordinate, {np.min(data)}, "
                    f"found in {data_name}. Computations may be incorrect."
                )
            if np.max(data) >= +100000000000000:
                warnings.warn(
                    f"Extremely large coordinate, {np.max(data)}, "
                    f"found in {data_name}. Computations may be incorrect."
                )

            # parse input according to data type
            t = type(data[0, 0])
            if t in (fmpz, fmpq):
                if not config._exp_features_enabled:
                    raise Exception(
                        "Arbitrary precision data types only have "
                        "experimental support, so experimental "
                        "features must be enabled in configuration."
                    )
                if t == fmpz:
                    data = utils.array_fmpz_to_int(data)
                else:
                    data = utils.array_fmpq_to_float(data)
            elif np.issubdtype(t, np.integer):
                # rest of calculations assume ints are 64-bit? convert...
                if t != np.int64:
                    data = data.astype(np.int64)
                t = np.int64
            elif np.issubdtype(t, np.floating):
                # rest of calculations assume floats are 64-bit? convert...
                if t != np.float64:
                    data = data.astype(np.float64)
                t = np.float64
            else:
                raise NotImplementedError("Unsupported data type.")

            # reduce by GCD
            if check or t in (fmpz, np.float64):
                # get GCDs
                if t == np.int64:
                    gcds = np.gcd.reduce(data, axis=1)
                else:
                    gcds = np.asarray([utils.gcd_list(v) for v in data])

                # reduce by them
                if t == np.int64:
                    mask = gcds > 0
                    if False in mask:
                        warnings.warn("0 gcd found (row of zeros)... Skipping it!")
                    data = data[mask] // gcds[mask].reshape(-1, 1).astype(int)
                else:
                    mask = gcds >= 1e-5
                    if False in mask:
                        warnings.warn(
                            "Extremely small gcd found... "
                            "Computations may be incorrect!"
                        )
                    data = np.rint(data[mask] / gcds[mask].reshape(-1, 1)).astype(int)
            else:
                data = data.astype(int)

        # put data in correct variable
        if self._rays_were_input:
            self._rays = np.asarray(data)
            self._dim = _rank(self._rays)
        else:
            self._hyperplanes = np.asarray(data)
            self._dim = None

    def clear_cache(self):
        """Clear the cached results of any previous computation.

        The input data (rays or hyperplane normals, whichever was used to
        construct the cone) is kept, while everything that was derived from it,
        including the other description of the cone, is discarded.

        Examples:
            We construct a cone, compute its extremal rays, clear the cache and
            then compute them again.

            >>> c = Cone([[1,0],[1,1],[0,1]])
            >>> c.extremal_rays()
            array([[1, 0],
                   [0, 1]])
            >>> c.clear_cache()  # clears the cached result
            >>> c.extremal_rays()  # the extremal rays are recomputed
            array([[1, 0],
                   [0, 1]])
        """
        self._hash = None
        self._dual = None
        self._ext_rays = [None, None]
        self._is_solid = None
        self._is_pointed = None
        self._is_simplicial = None
        self._is_smooth = None
        self._hilbert_basis = None
        self._faces = None
        if self._rays_were_input:
            self._hyperplanes = None
        else:
            self._rays = None

    def __repr__(self):
        """Return a string describing the cone.

        If the rays of the cone are known, the string states the dimension of
        the cone and the number of generating rays. Otherwise, it states the
        number of defining hyperplanes.

        Returns:
            str: A string describing the cone.

        Examples:
            This function can be used to convert the cone to a string or to print
            information about the cone.

            >>> c = Cone([[1,0],[1,1],[0,1]])
            >>> cone_info = str(c)  # converts to string
            >>> print(c)  # prints cone info
            A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
            >>> print(Cone(hyperplanes=[[1,0],[-1,1]]))
            A rational polyhedral cone in RR^2 defined by 2 hyperplanes
        """
        if self._rays is not None:
            return (
                f"A {self._dim}-dimensional rational polyhedral cone in "
                f"RR^{self._ambient_dim} generated by {len(self._rays)} "
                f"rays"
            )
        return (
            f"A rational polyhedral cone in RR^{self._ambient_dim} "
            f"defined by {len(self._hyperplanes)} hyperplanes"
        )

    def __eq__(self, other):
        """Compare two cones with `==`.

        Two cones are equal if they define the same set of points. The cones are
        first compared by their input data. If that is inconclusive, they are
        compared by their extremal rays if they are pointed, or by the extremal
        rays of their duals if the duals are pointed. As a last resort, it is
        checked whether the rays of each cone are contained in the other.

        Args:
            other (Cone): The other cone that is being compared.

        Returns:
            bool: The truth value of the cones being equal. If `other` is not a
                `Cone`, `NotImplemented` is returned.

        Examples:
            We construct two cones and compare them.

            >>> c1 = Cone([[0,1],[1,1]])
            >>> c2 = Cone(hyperplanes=[[1,0],[-1,1]])
            >>> c1 == c2
            True
        """
        if not isinstance(other, Cone):
            return NotImplemented

        if (
            self._rays is not None
            and other._rays is not None
            and sorted(self._rays.tolist()) == sorted(other._rays.tolist())
        ):
            # rays trivially match
            # N.B.: doesn't check for non-trivial equivalence. E.g.,
            # self._rays  = {e_1, -e_1, e2, -e_2}
            # other._rays = {e_1+e_2, -(e_1+e_2), e_1-e_2, -(e_1-e_2)}
            return True
        if (
            self._hyperplanes is not None
            and other._hyperplanes is not None
            and sorted(self._hyperplanes.tolist())
            == sorted(other._hyperplanes.tolist())
        ):
            # hyperplanes trivially match
            # N.B.: doesn't check for non-trivial equivalence. Same as above
            return True
        if self.is_pointed() ^ other.is_pointed():
            return False
        if self.is_pointed() and other.is_pointed():
            return sorted(self.extremal_rays().tolist()) == sorted(
                other.extremal_rays().tolist()
            )
        if self.dual().is_pointed() ^ other.dual().is_pointed():
            return False
        if self.dual().is_pointed() and other.dual().is_pointed():
            return sorted(self.dual().extremal_rays().tolist()) == sorted(
                other.dual().extremal_rays().tolist()
            )

        # ugly method... check if each ray self is contained in other
        # (and vice-versa)
        self_contained_in_other = np.all(
            other.hyperplanes() @ self.rays().transpose() >= 0
        )
        other_contained_in_self = np.all(
            self.hyperplanes() @ other.rays().transpose() >= 0
        )
        return self_contained_in_other and other_contained_in_self

    def __ne__(self, other):
        """Compare two cones with `!=`.

        This is the negation of `==`.

        Args:
            other (Cone): The other cone that is being compared.

        Returns:
            bool: The truth value of the cones being different. If `other` is not
                a `Cone`, `NotImplemented` is returned.

        Examples:
            We construct two cones and compare them.

            >>> c1 = Cone([[0,1],[1,1]])
            >>> c2 = Cone(hyperplanes=[[1,0],[-1,1]])
            >>> c1 != c2
            False
        """
        if not isinstance(other, Cone):
            return NotImplemented
        return not self == other

    def __hash__(self):
        """Return the hash value of the cone.

        For pointed cones, the hash is computed from the extremal rays. For
        cones whose dual is pointed, it is computed from the extremal rays of
        the dual cone (with a sign flip, so that a cone and its dual do not
        share the same hash). The result is cached.

        Returns:
            int: The hash value of the cone.

        Warning:
            Cones that are not pointed and whose duals are also not pointed are
            all assigned a hash value of 0, and a warning is issued.

        Examples:
            We compute the hash value of a cone. Also, we construct a set and a
            dictionary with a cone, which make use of the hash function.

            >>> c = Cone([[0,1],[1,1]])
            >>> h = hash(c)  # obtain hash value
            >>> d = {c: 1}  # create dictionary with cone keys
            >>> s = {c}  # create a set of cones
        """
        if self._hash is not None:
            return self._hash
        if self.is_pointed():
            self._hash = hash(tuple(sorted(tuple(v) for v in self.extremal_rays())))
            return self._hash
        if self.dual().is_pointed():
            # Note: The minus sign is important because otherwise the dual cone
            # would have the same hash.
            self._hash = -hash(
                tuple(sorted(tuple(v) for v in self.dual().extremal_rays()))
            )
            return self._hash

        warnings.warn(
            "Cones that are not pointed and whose duals are also "
            "not pointed are assigned a hash value of 0."
        )
        return 0

    def ambient_dimension(self):
        """Return the dimension of the ambient lattice.

        Also available as `ambient_dim`.

        Returns:
            int: The dimension of the ambient lattice.

        Examples:
            We construct a cone and find the dimension of the ambient lattice.

            >>> c = Cone([[0,1,0],[1,1,0]])
            >>> c.ambient_dimension()
            3
        """
        return self._ambient_dim

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """Return the dimension of the cone.

        This is the dimension of the linear span of the cone. If the cone was
        constructed from hyperplanes and is not solid, its rays are computed
        first, which can be expensive. The result is cached.

        Also available as `dim`.

        Returns:
            int: The dimension of the cone.

        Examples:
            We construct a cone and find its dimension.

            >>> c = Cone([[0,1,0],[1,1,0]])
            >>> c.dimension()
            2
        """
        if self._dim is not None:
            return self._dim

        if self._rays is not None:
            # know the rays... semi simple computation
            self._dim = _rank(self._rays)
        else:
            # don't know the rays... still simple if the cone is solid...
            if self.is_solid():
                self._dim = self.ambient_dim()
            else:
                # yikes need to compute the rays
                self._dim = _rank(self.rays())
        return self._dim

    # aliases
    dim = dimension

    def rays(self, use_extremal_hyperplanes: bool = False, verbosity: int = 0):
        r"""Return the (not necessarily extremal) rays that generate the cone.

        If the cone was constructed from rays, these are returned (after the
        normalization done at construction). Otherwise, they are computed from
        the hyperplane normals with PPL and cached. In that case, a lineality
        space is represented by pairs of opposite rays $\pm v$. This can be
        slow for cones in more than about 12 dimensions.

        Args:
            use_extremal_hyperplanes (bool): Whether to first compute the
                extremal hyperplanes and use them in this computation, instead of
                all the defining hyperplanes. Only relevant if the rays are not
                already known.
            verbosity (int): The verbosity level.

        Returns:
            np.ndarray: A copy of the rays that generate the cone, one per row.

        Examples:
            We construct two cones and find their generating rays.

            >>> c1 = Cone([[0,1],[1,1]])
            >>> c2 = Cone(hyperplanes=[[0,1],[1,1]])
            >>> c1.rays()
            array([[0, 1],
                   [1, 1]])
            >>> c2.rays()
            array([[ 1,  0],
                   [-1,  1]])
        """
        if self._rays is not None:
            return np.array(self._rays)
        if self._ambient_dim >= 12 and len(self._hyperplanes) != self._ambient_dim:
            warnings.warn(
                "This operation might take a while for d > ~12 "
                "and is likely impossible for d > ~18."
            )

        # select the hyperplanes
        if use_extremal_hyperplanes:
            H = self.extremal_hyperplanes()
        else:
            H = self.hyperplanes()

        # compute the rays
        rays = dualize(H, verbosity=verbosity)

        # save/return
        if verbosity >= 1:
            print("Saving the rays & computing dimension...", flush=True)
        self._rays = np.asarray(rays, dtype=int)
        if len(self._rays) == 0:
            # matrix_rank raises on a (0, d) array, and the span is trivial
            self._rays = np.zeros((0, self._ambient_dim), dtype=int)
        self._dim = _rank(self._rays)
        return np.array(self._rays)

    def hyperplanes(self, use_extremal_rays: bool = False, verbosity: int = 0):
        r"""Return the inward-pointing normals to the hyperplanes that define the cone.

        The cone is the set of points $x$ such that $h\cdot x\geq 0$ for every
        returned normal $h$. If the cone was constructed from hyperplane
        normals, these are returned (after the normalization done at
        construction). Otherwise, they are computed from the rays with PPL and
        cached. In that case, if the cone is not solid, the linear subspace it
        spans is cut out by pairs of opposite normals $\pm h$. This can be slow
        for cones in more than about 12 dimensions.

        Args:
            use_extremal_rays (bool): Whether to first compute the extremal rays
                and use them in this computation, instead of all the generating
                rays. Only relevant if the hyperplanes are not already known.
            verbosity (int): The verbosity level.

        Returns:
            np.ndarray: A copy of the inward-pointing hyperplane normals, one per
                row.

        Examples:
            We construct two cones and find their hyperplane normals.

            >>> c1 = Cone([[0,1],[1,1]])
            >>> c2 = Cone(hyperplanes=[[0,1],[1,1]])
            >>> c1.hyperplanes()
            array([[ 1,  0],
                   [-1,  1]])
            >>> c2.hyperplanes()
            array([[0, 1],
                   [1, 1]])
        """
        if self._hyperplanes is not None:
            return np.array(self._hyperplanes)
        if self._ambient_dim >= 12 and len(self.rays()) != self._ambient_dim:
            warnings.warn(
                "This operation might take a while for d > ~12 "
                "and is likely impossible for d > ~18."
            )

        # select the rays
        if use_extremal_rays:
            R = self.extremal_rays()
        else:
            R = self.rays()

        # compute the hyperplanes
        H = dualize(R, verbosity=verbosity)

        # save/return
        if verbosity >= 1:
            print("Saving the hyperplanes...", flush=True)
        self._hyperplanes = np.asarray(H, dtype=int)
        if len(self._hyperplanes) == 0:
            self._hyperplanes = np.zeros((0, self._ambient_dim), dtype=int)
        return np.array(self._hyperplanes)

    def contains(self, other, eps: float = 0) -> bool:
        r"""Check whether points, or another cone, are contained in the cone.

        A point $x$ is considered to be contained if $h\cdot x\geq \epsilon$ for
        every hyperplane normal $h$ returned by
        [`hyperplanes`][cytools.cone.Cone.hyperplanes]. With the default
        $\epsilon=0$ this checks membership in the (closed) cone, while a
        positive $\epsilon$ can be used to check whether a point is in the
        strict interior. A cone is contained if all of its generating rays are.

        Args:
            other (ArrayLike | Cone): The object to check. A 1D array is treated
                as a single point, a 2D array as a list of points (one per row),
                and a `Cone` is checked through its rays.
            eps (float): The minimum value that $h\cdot x$ must attain for
                every hyperplane normal $h$.

        Returns:
            bool | tuple[bool, ...]: Whether the point or the cone is contained.
                If a 2D array is given, a tuple with one truth value per point is
                returned.

        Examples:
            We construct a cone and check whether some points and a cone are
            contained in it.

            >>> c = Cone([[1,0],[0,1]])
            >>> bool(c.contains([1,1]))
            True
            >>> [bool(b) for b in c.contains([[1,1],[-1,0]])]
            [True, False]
            >>> bool(c.contains([1,0], eps=1))  # on the boundary
            False
            >>> c.contains(Cone([[1,1],[1,2]]))
            True
        """
        if isinstance(other, Cone):
            # just check if we contain all of other's rays...
            return all(self.contains(other.rays(), eps=eps))

        # other was a point(s)
        H = self.hyperplanes()
        pt = np.array(other)

        # cast to 2D array, transpose
        if len(pt.shape) == 1:
            pt = pt.reshape(-1, 1)
            return_list = False
        else:
            # transpose so columns are points
            pt = pt.transpose()
            return_list = True

        # compute which points are in the cone
        if len(H):
            contained = np.all(H @ pt >= eps, axis=0)
        else:
            contained = [True for _ in range(pt.shape[1])]

        # return
        if return_list:
            return tuple(contained)
        else:
            return contained[0]

    def dual_cone(self):
        r"""Return the dual cone.

        The dual cone is the set of points $y$ such that $y\cdot x\geq 0$ for
        all points $x$ in the cone. It is constructed by using the rays of this
        cone as the hyperplane normals of the dual (or vice versa), and it is
        cached. The dual of the returned cone is this cone.

        Also available as `dual`.

        Returns:
            Cone: The dual cone.

        Examples:
            We construct a cone and find its dual cone.

            >>> c = Cone([[0,1],[1,1]])
            >>> c.dual_cone()
            A rational polyhedral cone in RR^2 defined by 2 hyperplanes
            >>> c.dual_cone().rays()
            array([[ 1,  0],
                   [-1,  1]])
        """
        if self._dual is None:
            if self._rays is not None:
                self._dual = Cone(hyperplanes=self.rays(), check=False)
            else:
                self._dual = Cone(rays=self.hyperplanes(), check=False)
            self._dual._dual = self
        return self._dual

    # aliases
    dual = dual_cone

    def extremal_rays(
        self,
        tol: float = 1e-4,
        minimal: bool = True,
        method: str = "extremalrays",
        verbose: bool = False,
    ) -> "ArrayLike":
        """Return the extremal rays of the cone.

        A ray is extremal if it cannot be written as a non-negative linear
        combination of the other rays. For pointed cones, the extremal rays form
        the unique minimal generating set of the cone. For non-pointed cones
        there is no such unique set, and the behaviour depends on `minimal`.

        Duplicate rays are removed first, and the order of the returned rays
        need not match the order of the input rays. The result is cached
        separately for each value of `minimal` (the cache does not depend on
        `method` or `tol`). If the cone was constructed from hyperplanes, the
        extremal rays are also stored as its generating rays.

        Args:
            tol (float): The tolerance for deciding whether a ray is extremal.
                Only used if `method` is `"nnls"`.
            minimal (bool): Whether to return a minimal generating set of rays.
                This only matters for non-pointed cones. If true, the cone is
                decomposed into its lineality space and its pointed part (see
                [`lineality_space`][cytools.cone.Cone.lineality_space] and
                [`pointed_space`][cytools.cone.Cone.pointed_space]), and the
                extremal rays of both are returned together, so the result need
                not be a subset of the generating rays. If false, the generating
                rays that are not non-negative combinations of the others are
                returned, which need not form a minimal generating set.
            method (str): The backend used to prune the rays. The options are:
                `"extremalrays"`, which uses the `extremalrays` package and falls
                back to `"legacy"` if the package is not installed (with a warning)
                or if the cone is not pointed; `"legacy"` (or its synonym `"lp"`),
                which checks each ray by solving a linear program; and `"nnls"`,
                which checks each ray by solving a non-negative least-squares
                problem.
            verbose (bool): Whether to print the progress of the computation.

        Returns:
            np.ndarray: The extremal rays of the cone, one per row.

        Raises:
            ValueError: If `method` is not one of the allowed options.
            RuntimeError: If the check of a ray fails repeatedly with the
                `"legacy"` or `"nnls"` backends.

        Note:
            The `"legacy"` and `"nnls"` backends check the rays in parallel for
            pointed cones with at least 32 rays, using as many CPU threads as are
            available. To fix the number of threads, you can set the
            `n_threads` variable in the `cytools.config` module.

        Examples:
            We construct a cone and find its extremal rays.

            >>> c = Cone([[0,1],[1,1],[1,0]])
            >>> c.extremal_rays()
            array([[0, 1],
                   [1, 0]])
        """
        if self._ext_rays[minimal] is not None:
            return np.array(self._ext_rays[minimal])

        # non-pointed cones are tricky
        # A ray r of the ray set (i.e., generating matrix) R is extremal if it
        # cannot be written as a non-negative combination of the other rays
        #
        # For pointed cones, there is a unique collection of extremal rays
        # defining a cone. For non-pointed cones, this is not true.
        #
        # Furthermore, for non-pointed cones, every ray r of R may be extremal
        # with respect to R, but there might be a smaller set of rays R'
        # defining the same region.
        #
        # For simplicity, we return minimal (in terms of ray count) generating
        # matrices by analyzing the lineality space and the pointed bit of the
        # cone separately
        if minimal and (not self.is_pointed()):
            self._ext_rays[minimal] = np.vstack(
                [
                    self.lineality_space().extremal_rays(),
                    self.pointed_space().extremal_rays(),
                ]
            )

            return np.array(self._ext_rays[minimal])

        # It is important to delete duplicates
        rays = np.array(list({tuple(r) for r in self.rays()}))

        # a cone with no rays (e.g. the pointed part of a whole linear space)
        # has none that are extremal; np.array([]) is 1d, so restore the shape
        if rays.shape[0] == 0:
            self._ext_rays[minimal] = rays.reshape(0, self.ambient_dim())
            return np.array(self._ext_rays[minimal])

        # if only 1 ray, this is trivial
        if rays.shape[0] == 1:
            self._ext_rays[minimal] = rays
            if self._rays is None:
                self._rays = np.array(self._ext_rays[minimal])

            return np.array(self._ext_rays[minimal])

        # "lp" named the per-ray backend before extremalrays became default
        if method == "lp":
            method = "legacy"
        if method not in ("extremalrays", "legacy", "nnls"):
            raise ValueError(
                f"Unknown method '{method}'; expected "
                "'extremalrays', 'legacy' or 'nnls'."
            )

        if method == "extremalrays":
            if not _HAS_EXTREMALRAYS:
                warnings.warn(
                    "The extremalrays package is not installed, so the slower "
                    "legacy backend is being used. Install it with "
                    "'pip install extremalrays', or pass method='legacy' to "
                    "silence this warning."
                )
                method = "legacy"
            elif not self.is_pointed():
                # the sweep needs a pointed cone; one only gets here with
                # minimal=False, since minimal=True is decomposed above
                method = "legacy"

        if method == "extremalrays":
            if verbose:
                print(
                    f"Computing extremal rays for a cone with {len(rays)} "
                    "rays using extremalrays..."
                )
            # returns indices; sort to match the legacy output order. Its own
            # tol is a separation tolerance, unrelated to the nnls tol above
            keep = np.sort(
                np.asarray(
                    _extremalrays.exhaustive(rays, verbosity=1 if verbose else 0)
                )
            )
            self._ext_rays[minimal] = rays[keep]
            if self._rays is None:
                self._rays = np.array(self._ext_rays[minimal])

            return np.array(self._ext_rays[minimal])

        # the per-ray backend. is_extremal takes "lp" or "nnls"
        check_method = "nnls" if method == "nnls" else "lp"

        # configure threads
        n_threads = config.n_threads
        if n_threads is None:
            if rays.shape[0] < 32 or not self.is_pointed():
                n_threads = 1
            else:
                n_threads = cpu_count()
        elif n_threads > 1 and not self.is_pointed():
            warnings.warn(
                "When finding the extremal rays of a non-pointed "
                "cone in parallel, there can be conflicts that end up "
                "producing erroneous results. It is highly recommended to "
                "use a single thread."
            )

        # compute the extremal rays
        ext_rays = [True for _ in range(len(rays))]
        to_check = list(range(len(rays)))

        if verbose:
            print(
                f"Computing extremal rays for a cone with {len(rays)} using {n_threads} threads..."
            )

        # bound the retries; unbounded ones spin forever on a ray that
        # deterministically fails
        max_attempts = 3
        attempts = {}

        # one ray per worker per round: ext_rays is only pruned between
        # rounds, so bigger batches re-check known-redundant rays
        # (4*n_threads took >10 min on a 908x125 cone, vs 7 s here)
        batch_size = n_threads

        def learn(results, checking):
            """Record verdicts; queue failures for a bounded retry."""
            for i, extremalQ, err in results:
                if err is None:
                    ext_rays[i] = extremalQ
                    continue
                attempts[i] = attempts.get(i, 0) + 1
                if attempts[i] >= max_attempts:
                    raise RuntimeError(
                        f"Failed to check whether ray #{i} was extremal after "
                        f"{max_attempts} attempts. (Last error was: {err})"
                    )
                to_check.append(i)
                if verbose:
                    print(f"Failed to check whether ray #{i} was extremal")
                    print(f"(Error was: {err})")
                    print("(Putting it at the end and retrying later...)")

        if n_threads == 1:
            # skip the pool; with one worker the serialization is pure
            # overhead
            while len(to_check):
                checking = to_check[:batch_size]
                to_check = to_check[batch_size:]
                learn(
                    [
                        is_extremal(rays, i, ext_rays, method=check_method, tol=tol)
                        for i in checking
                    ],
                    checking,
                )
        else:
            # one context for the whole loop: joblib auto-memmaps args
            # over 1 MB, so a new context per round re-dumped the rays each
            # round; held open it dumps once and keeps the workers warm
            with joblib.Parallel(n_jobs=n_threads) as parallel:
                while len(to_check):
                    checking = to_check[:batch_size]
                    to_check = to_check[batch_size:]
                    learn(
                        parallel(
                            joblib.delayed(is_extremal)(
                                rays, i, ext_rays, method=check_method, tol=tol
                            )
                            for i in checking
                        ),
                        checking,
                    )

        # save the answer
        self._ext_rays[minimal] = rays[list(ext_rays)]
        if self._rays is None:
            self._rays = np.array(self._ext_rays[minimal])

        return np.array(self._ext_rays[minimal])

    def extremal_hyperplanes(
        self,
        tol: float = 1e-4,
        minimal=True,
        method="extremalrays",
        verbose: bool = False,
    ) -> "ArrayLike":
        """Return the extremal hyperplanes of the cone.

        These are the extremal rays of the dual cone, i.e. a set of
        inward-pointing hyperplane normals that define the cone and are not
        redundant. This is computed with
        [`extremal_rays`][cytools.cone.Cone.extremal_rays] on the dual cone, so
        see its documentation for details.

        Args:
            tol (float): The tolerance for deciding whether a hyperplane is
                extremal. Only used if `method` is `"nnls"`.
            minimal (bool): Whether to return a minimal set of hyperplanes. This
                only matters when the dual cone is not pointed, i.e. when the cone
                is not solid.
            method (str): The backend used to prune the hyperplanes. The options
                are `"extremalrays"`, `"legacy"` (or its synonym `"lp"`), and
                `"nnls"`.
            verbose (bool): Whether to print the progress of the computation.

        Returns:
            np.ndarray: The extremal hyperplane normals of the cone, one per row.

        Examples:
            We construct a cone and find its extremal hyperplanes.

            >>> c = Cone([[0,1],[1,1],[1,0]])
            >>> c.extremal_hyperplanes()
            array([[0, 1],
                   [1, 0]])
        """
        return self.dual().extremal_rays(
            tol=tol, minimal=minimal, method=method, verbose=verbose
        )

    def faces(
        self, codim: int = None, include_self: bool = False, verbosity: int = 0
    ) -> tuple[tuple[Cone]] | tuple[Cone]:
        """Return the faces of the cone, organized by codimension.

        The faces are found from the incidences between the extremal rays and
        the extremal hyperplanes of the cone. Each face of positive dimension is
        returned as a `Cone` generated by its extremal rays, while the
        zero-dimensional face (the origin) is returned as a `Cone` defined by
        hyperplanes. The result is cached.

        Only pointed cones are currently supported. This computation can be
        expensive for high-dimensional cones, and it is slower than
        [`facets`][cytools.cone.Cone.facets] if only the facets are needed.

        Args:
            codim (int | None): The codimension of the desired faces. If not
                specified, the faces of all codimensions are returned. When set to
                0, the cone itself is returned.
            include_self (bool): Whether to include the cone itself (the
                codimension-0 face) when returning the faces of all codimensions.
            verbosity (int): The verbosity level.

        Returns:
            tuple[Cone, ...] | tuple[tuple[Cone, ...], ...]: If `codim` is
                specified, the faces of that codimension. Otherwise, a tuple with
                one tuple of faces per codimension, in increasing order, from
                codimension 1 (or 0 if `include_self` is true) up to the dimension
                of the cone.

        Raises:
            ValueError: If `codim` is negative or larger than the dimension of the
                cone.
            NotImplementedError: If the cone is not pointed.

        Examples:
            We construct a simplicial cone and find the number of faces of each
            codimension, as well as the rays of one of its facets.

            >>> c = Cone([[1,0,0],[0,1,0],[0,0,1]])
            >>> [len(f) for f in c.faces()]
            [3, 3, 1]
            >>> c.faces(codim=1)[0].rays()
            array([[1, 0, 0],
                   [0, 0, 1]])
        """
        dim = self.dim()

        # input guard
        if (codim is not None) and ((codim < 0) or (codim > dim)):
            raise ValueError(f"Cone does not have faces of codimension {codim}")

        if dim > 20:
            warnings.warn("Getting the faces of high-dim cones is expensive")

        # easy answers
        if codim == 0:
            return (self,)
        if codim == dim:
            if self.is_pointed():
                # return cone spanned by 0 rays
                eye = np.eye(self.ambient_dim(), dtype=int)
                return (Cone(hyperplanes=np.vstack([eye, -eye])),)
            else:
                # return empty list... no 0D faces here
                return tuple()

        # fast track if cached
        if self._faces is not None:
            return (
                self._faces[codim]
                if codim is not None
                else (self._faces if include_self else self._faces[1:])
            )

        if not self.is_pointed():
            raise NotImplementedError(
                "Cone.faces() currently supports only pointed cones."
            )

        if verbosity >= 1:
            print(
                "Computing cone faces via extremal ray/hyperplane incidence..."
            )

        # expensive work vvv
        R = self.extremal_rays()
        H = self.extremal_hyperplanes()
        # expensive work ^^^

        # compute the incidences
        if self.is_solid():
            can_saturate = H
        else:
            can_saturate = np.array(
                [h for h in H if not self.dual().contains(-h)],
                dtype=int,
            )

        facet_ray_sets = set()
        for h in can_saturate:
            ray_inds = frozenset(
                i
                for i, r in enumerate(R)
                if sum(int(a) * int(b) for a, b in zip(h, r)) == 0
            )
            if ray_inds:
                facet_ray_sets.add(ray_inds)

        seen = set(facet_ray_sets)
        frontier = list(facet_ray_sets)
        while frontier:
            current = frontier.pop()
            for facet in facet_ray_sets:
                inter = current & facet
                if inter and inter not in seen:
                    seen.add(inter)
                    frontier.append(inter)

        face_sets = [[] for _ in range(dim)]
        face_objects = {}
        for ray_inds in sorted(seen, key=lambda inds: tuple(sorted(inds))):
            face_rays = R[list(ray_inds)]
            face_dim = np.linalg.matrix_rank(face_rays)
            if face_dim <= 0:
                continue

            face_codim = dim - face_dim
            if face_codim in (0, dim):
                continue

            face_sets[face_codim].append(ray_inds)
            face_objects[ray_inds] = Cone(rays=face_rays, check=False)

        faces = [(self,)]
        for face_codim in range(1, dim):
            codim_faces = tuple(
                face_objects[ray_inds]
                for ray_inds in sorted(
                    face_sets[face_codim], key=lambda inds: tuple(sorted(inds))
                )
            )
            faces.append(codim_faces)

        # add the 0D cone if this is pointed
        if self.is_pointed():
            eye = np.eye(self.ambient_dim(), dtype=int)
            faces.append((Cone(hyperplanes=np.vstack([eye, -eye])),))

        # cache and return
        self._faces = tuple(faces)
        return (
            self._faces[codim]
            if codim is not None
            else (self._faces if include_self else self._faces[1:])
        )

    def facets(self, verbosity: int = 0):
        """Return the facets of the cone.

        The facets are the faces of codimension 1. If the cone is simplicial,
        they are the cones generated by each subset of all but one of the
        extremal rays. Otherwise, they are computed by saturating, one at a time,
        each extremal hyperplane that does not vanish on the whole cone, which
        requires computing the extremal hyperplanes. In this case, the facets are
        returned as cones defined by hyperplanes. The result is not cached.

        Args:
            verbosity (int): The verbosity level.

        Returns:
            list[Cone]: The facets of the cone.

        Examples:
            We construct a non-simplicial cone and find its facets.

            >>> c = Cone([[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]])
            >>> f = c.facets()
            >>> len(f)
            4
            >>> f[0].extremal_rays()
            array([[-1,  0,  1],
                   [ 0,  1,  1]])
        """
        # ray-based computation
        if self.is_simplicial():
            if verbosity >= 1:
                print("Cone is simplicial! Easy computation...")
            R = self.extremal_rays()

            dim = len(R)

            if dim == 1:
                eye = np.eye(self.ambient_dim(), dtype=int)
                return [
                    Cone(hyperplanes=np.vstack([eye, -eye])),
                ]
            elif dim == 0:
                return []

            ray_inds = list(range(dim))

            # facets are defined by collections of #(dim-1) rays
            return [
                Cone(rays=R[list(inds)])
                for inds in itertools.combinations(ray_inds, dim - 1)
            ]

        # hyperplane based-computation
        if verbosity >= 1:
            print("Computing facets via extremal hyperplanes...")
        H = self.extremal_hyperplanes()

        if self.is_solid():
            # still pretty easy
            can_saturate = H
        else:
            # this means that the cone contains both h and -h as hyperplanes...
            # i.e., h is already saturated by definition...
            # need to skip these when looking to saturate hyperplanes
            can_saturate = [h for h in H if not self.dual().contains(-h)]

        return [Cone(hyperplanes=np.vstack((H, -h)), check=False) for h in can_saturate]

    def tip_of_stretched_cone(
        self,
        c=1,
        backend=None,
        check=True,
        constraint_error_tol=5e-2,
        max_iter=10**6,
        show_hints=True,
        verbose=False,
    ):
        r"""Find the tip of the stretched cone.

        The stretched cone is the region of points $x$ in the cone such that
        $h\cdot x\geq c$ for every hyperplane normal $h$ returned by
        [`hyperplanes`][cytools.cone.Cone.hyperplanes]. Its tip is the point in
        this region with the smallest (Euclidean) norm. Note that the normals
        are integral vectors that are not normalized, so $c$ is not literally a
        distance to the hyperplanes.

        If the cone has no defining hyperplanes (i.e. it is the whole space), a
        vector of ones is returned.

        Args:
            c (float): A positive number specifying the stretching of the cone.
            backend (str | None): The optimizer to use. The options are
                `"mosek"`, `"osqp"`, and `"cvxopt"`, which solve the quadratic
                program, and `"highs"` and `"glop"`, which solve a linear program
                whose solution is only an approximation of the tip (see
                [`find_interior_point`][cytools.cone.Cone.find_interior_point]).
                If not specified, `"osqp"` is used for ambient dimension $d<25$.
                For $d\geq 25$, `"mosek"` is used if it is activated, and
                `"highs"` otherwise.
            check (bool): Whether to check that the output of the optimizer
                satisfies the constraints within `constraint_error_tol`.
            constraint_error_tol (float): The error tolerance for the linear
                constraints.
            max_iter (int): The maximum number of iterations allowed for the
                quadratic programming backends. If this function is returning
                `None`, then increasing this parameter (up to $2^{31}-1$) might
                resolve the issue. It is ignored by the `"highs"` and `"glop"`
                backends.
            show_hints (bool): Whether to print hints about possible reasons for
                failure when no solution is found.
            verbose (bool): Whether to print extra diagnostic information.

        Returns:
            np.ndarray | None: The location of the tip. If it could not be found,
                or if it does not pass the check, then `None` is returned.

        Raises:
            ValueError: If an invalid backend is specified.
            Exception: If `"mosek"` is specified but Mosek is not activated.

        Note:
            Finding the tip requires quadratic programming, since the norm of a
            vector is being minimized. For dimensions up to around 25, this can
            easily be done with open-source solvers like OSQP or CVXOPT. For
            higher dimensions this becomes a difficult task that only the Mosek
            optimizer handles well. However, Mosek is closed-source and requires a
            license. For this reason, when Mosek is not activated, the default for
            high dimensions is to solve a linear problem with HiGHS, which finds a
            good approximation of the tip.

        Examples:
            We construct two cones and find the locations of the tips of the
            stretched cones.

            >>> c1 = Cone([[1,0],[0,1]])
            >>> c2 = Cone([[3,2],[5,3]])
            >>> c1.tip_of_stretched_cone(1)
            array([1., 1.])
            >>> c2.tip_of_stretched_cone(1)
            array([8., 5.])
        """
        # set the backend
        backends = (None, "mosek", "osqp", "cvxopt", "highs", "glop")
        if backend not in backends:
            raise ValueError(f"Invalid backend. The options are: {backends}.")

        if backend is None:
            if self.ambient_dim() < 25:
                backend = "osqp"
            else:
                backend = (
                    "mosek"
                    if config.mosek_is_activated() and self.ambient_dim() >= 25
                    else "highs"
                )
        elif backend == "mosek" and not config.mosek_is_activated():
            raise Exception(
                "Mosek is not activated. See the advanced usage "
                "page on our website to see how to activate it."
            )

        # check backend
        if (self.ambient_dim() >= 25) and (backend != "mosek") and verbose:
            print(f"The backend {backend} may not work given the large ")
            print(f"dimension ({self.ambient_dim()}) of the problem...")

        # find the tip of the stretched cone
        if len(self.hyperplanes()) == 0:
            # trivial
            return np.ones(self._ambient_dim)

        if backend in ("glop", "highs"):
            solution = self.find_interior_point(c, backend=backend, verbose=verbose)
            G = -1 * sparse.csc_matrix(self.hyperplanes(), dtype=float)
        else:
            hp = self._hyperplanes
            # The problem is defined as:
            # Minimize (1/2) x.P.x + q.x
            # Subject to G.x <= h
            P = 2 * sparse.identity(hp.shape[1], dtype=float, format="csc")
            q = np.zeros(hp.shape[1], dtype=float)
            h = np.full(hp.shape[0], -c, dtype=float)
            G = -1 * sparse.csc_matrix(hp, dtype=float)
            settings_dict = (
                {
                    "scaling": 50,
                    "eps_abs": 1e-4,
                    "eps_rel": 1e-4,
                    "polishing": True,
                }
                if backend == "osqp"
                else dict()
            )
            solution = qpsolvers.solve_qp(
                P,
                q,
                G,
                h,
                solver=backend,
                max_iter=max_iter,
                verbose=verbose,
                **settings_dict,
            )

        # parse solution
        if solution is None:
            if show_hints:
                print("Calculated 'solution' was None...", end=" ")
                print("some potential reasons why:")

                # max_iter
                if backend not in ("glop", "highs"):
                    print(f"-) maybe max_iter={max_iter} was too low?")

                # bad solver
                if (self.ambient_dim() >= 25) and (backend != "mosek"):
                    print(
                        f"-) given the high dimension, {self.ambient_dim()}",
                        end=" ",
                    )
                    print(
                        f"and backend={backend}, this could be a numerical",
                        end=" ",
                    )
                    print("issue. Try Mosek...")

                # scaling
                print(
                    f"-) if the cone is narrow, try decreasing c from {c}",
                    end=" ",
                )
                print(
                    "(you can then scale up the tip to hit the desired stretching...)"
                )

                print("For more info, re-run with verbose=True")
            return
        if check:
            res = max(G.dot(solution)) + c
            if res > constraint_error_tol:
                warnings.warn(
                    f"The solution that was found is invalid: {res} > {constraint_error_tol}"
                )
                return
        return solution

    def find_grading_vector(self, backend=None):
        r"""Find a grading vector for the cone.

        A grading vector is a vector $v$ such that $v\cdot p>0$ for every
        non-zero point $p$ in the cone. Thus, it must be in the strict interior
        of the dual cone, so it only exists for pointed cones. This function
        returns an integral grading vector, found with
        [`find_interior_point`][cytools.cone.Cone.find_interior_point] on the
        dual cone.

        Args:
            backend (str | None): The optimizer to use. The options are the same
                as for [`find_interior_point`][cytools.cone.Cone.find_interior_point].
                If not specified, the default of that function is used.

        Returns:
            np.ndarray | None: An integral grading vector. If it could not be
                found, then `None` is returned.

        Raises:
            Exception: If the cone is not pointed.

        Examples:
            We construct a cone and find a grading vector.

            >>> c = Cone([[3,2],[5,3]])
            >>> c.find_grading_vector()
            array([-1,  2])
        """
        if not self.is_pointed():
            raise Exception("Grading vectors are only defined for pointed cones.")
        return self.dual().find_interior_point(backend=backend, integral=True)

    def find_interior_point(
        self,
        c: float = 1,
        lower: float = None,
        integral: bool = False,
        backend: str = None,
        check: bool = True,
        show_hints: bool = False,
        verbose: bool = False,
    ) -> "ArrayLike":
        r"""Find a point in the strict interior of the cone.

        If the rays of the cone are known and neither `backend` nor `lower` is
        specified, this simply returns the sum of the rays divided by their GCD
        (and, unless `integral` is true, divided by the number of rays), or
        `None` if the cone is not solid. In this case, `c` and `check` are
        ignored.

        Otherwise, an optimization problem is solved to find a point $x$ with
        $h\cdot x\geq c$ for every hyperplane normal $h$ returned by
        [`hyperplanes`][cytools.cone.Cone.hyperplanes]. The LP and integer
        programming backends minimize the average of the $h\cdot x$ (see
        [`feasibility`][cytools.cone.feasibility]), while the quadratic
        programming backends find the tip of the stretched cone (see
        [`tip_of_stretched_cone`][cytools.cone.Cone.tip_of_stretched_cone]).

        Args:
            c (float): A positive number specifying the stretching of the cone,
                i.e. the minimum value of $h\cdot x$ for each hyperplane normal
                $h$. Not used if the point is obtained from the rays.
            lower (float | None): A lower bound on each of the coordinates of
                the point. It is not supported by the `"mosek"`, `"osqp"`, and
                `"cvxopt"` backends.
            integral (bool): Whether the point should have integral coordinates.
                If the optimizer returns a non-integral point, its multiples are
                rounded until one that is strictly interior is found.
            backend (str | None): The optimizer to use. The options are `"highs"`
                (an LP via HiGHS), `"glop"` (an LP via OR-Tools), `"scip"` (an
                integer program via OR-Tools), `"cpsat"` (an integer program via
                OR-Tools CP-SAT), and `"mosek"`, `"osqp"`, and `"cvxopt"` (quadratic
                programs). If not specified and the rays are not known, `"mosek"`
                is used if it is activated and the ambient dimension is at least
                25, and `"highs"` otherwise.
            check (bool): Whether to verify that the point found by the optimizer
                is in the strict interior of the cone.
            show_hints (bool): Whether to print hints about possible reasons for
                failure of the quadratic programming backends.
            verbose (bool): Whether to print extra diagnostic information.

        Returns:
            np.ndarray | None: A point in the strict interior of the cone. If no
                point is found, then `None` is returned.

        Raises:
            ValueError: If an invalid backend is specified, or if `lower` is
                specified with a backend that does not support it.

        Examples:
            We construct a cone and find some interior points.

            >>> c = Cone([[3,2],[5,3]])
            >>> c.find_interior_point()
            array([4. , 2.5])
            >>> c.find_interior_point(integral=True)
            array([8, 5])
        """
        backends = (None, "highs", "glop", "scip", "cpsat", "mosek", "osqp", "cvxopt")
        if backend not in backends:
            raise ValueError(f"Invalid backend. The options are: {backends}.")

        # If the rays are already computed then this is a simple task
        if (self._rays is not None) and (backend is None) and (lower is None):
            if _rank(self._rays) != self._ambient_dim:
                return None

            point = self._rays.sum(axis=0)

            if max(abs(point)) > 1e-3:
                point //= utils.gcd_list(point)
            else:
                # looks like the point is all zeros
                if np.prod(self.hyperplanes().shape) == 0:
                    # trivial cone... all space
                    point = [0 for _ in range(self._ambient_dim)]
                    point[0] = 1
                    return np.asarray(point)
                else:
                    raise Exception(
                        f"Unexpected error in finding point in cone with rays = {self._rays}"
                    )

            if not integral:
                point = point / len(self._rays)

            return point

        # Otherwise we need to do a harder computation...
        H = self.hyperplanes()

        if backend is None:
            if config.mosek_is_activated() and (self.ambient_dim() >= 25):
                backend = "mosek"
            else:
                backend = "highs"

        if backend in ("highs", "glop", "scip", "cpsat"):
            solution = feasibility(
                hyperplanes=H,
                c=c,
                ambient_dim=self._ambient_dim,
                backend=backend,
                lower_bound=lower,
                verbose=verbose,
            )
        else:
            if lower is not None:
                raise ValueError(
                    f"Cannot set custom lower bound for backend = {backend}"
                )
            solution = self.tip_of_stretched_cone(
                c, backend=backend, show_hints=show_hints, verbose=verbose
            )
        if solution is None:
            return None

        # function to take dot products
        if isinstance(H, (list, np.ndarray)):

            def dot(hp, x):
                return hp.dot(x)

        else:

            def dot(hp, x):
                return sum([val * x[ind] for ind, val in hp.items()])

        # Make sure that the solution is valid
        if check and any(dot(v, solution) <= 0 for v in H):
            warnings.warn("The solution that was found is invalid.")
            return None

        # Finally, round to an integer if necessary
        if integral:
            n_tries = 1000
            for i in range(1, n_tries):
                int_sol = np.array([int(round(x)) for x in i * solution])
                if all(dot(v, int_sol) > 0 for v in H):
                    break
                if i == n_tries - 1:
                    return None
            solution = int_sol

        return solution

    def find_lattice_points(
        self,
        min_points=None,
        max_deg=None,
        grading_vector=None,
        c=0,
        max_coord=None,
        deg_window=0,
        filter_function=None,
        process_function=None,
        fast_mode=True,
        max_B=10000,
        verbose=False,
    ):
        r"""Find lattice points in the cone.

        The lattice points $x$ that are found satisfy $h\cdot x\geq c$ for every
        hyperplane normal $h$ returned by
        [`hyperplanes`][cytools.cone.Cone.hyperplanes], so with the default
        `c=0` they are the lattice points of the cone. Either the minimum number
        of points to find or the maximum degree of the points must be
        specified. There are two algorithms:

        - A fast mode, which is used when `fast_mode` is true, `min_points` is
          specified, the ambient dimension is at most 10, and none of `max_deg`,
          `grading_vector`, `filter_function`, or `process_function` are given.
          It enumerates the lattice points in boxes of increasing size (i.e.
          bounded by the $L^\infty$ norm) using the `latticepts` package, and it
          returns at least `min_points` points, unless the box size reaches
          `max_B`. In this mode, `max_coord` and `deg_window` are ignored, and
          `c` is converted to integers.
        - The general algorithm, which requires the cone to be pointed. It uses
          OR-Tools CP-SAT to find the points whose degree, i.e. their dot product
          with the grading vector, is at most `max_deg`. If `max_deg` is not
          given, it instead searches in windows of `deg_window + 1` consecutive
          degrees, starting from degree 0, until at least `min_points` points
          are found. All the points in the last window are kept, so more than
          `min_points` points may be returned. If both `max_deg` and
          `min_points` are given, `min_points` is ignored. The points are
          returned sorted by degree, and lexicographically within each degree.

        Args:
            min_points (int | None): The minimum number of points to find. Note
                that this count does not take `filter_function` into account.
            max_deg (int | None): The maximum degree of the points to find. This
                is useful when working with a preferred grading vector.
            grading_vector (ArrayLike | None): The grading vector that will be
                used. If not specified, then it is computed with
                [`find_grading_vector`][cytools.cone.Cone.find_grading_vector].
            c (float | ArrayLike): The minimum allowed stretching. It can be a
                single number, or one number per hyperplane, in the order of
                [`hyperplanes`][cytools.cone.Cone.hyperplanes].
            max_coord (int | None): The maximum absolute value of the coordinates
                of the points in the general algorithm. If not specified, the
                largest bound supported by CP-SAT is used.
            deg_window (int): The number of additional degrees that are searched
                at a time when using `min_points` in the general algorithm. The
                searched windows are the degrees from $n(w+1)$ to $n(w+1)+w$, for
                $n=0,1,2,\dots$ and $w$ equal to `deg_window`.
            filter_function (Callable | None): A function that takes a point (as
                a tuple of integers) and returns a boolean indicating whether to
                keep it.
            process_function (Callable | None): A function that is called on each
                point (as a tuple of integers) as it is found. This is useful to
                avoid first constructing a large list of points and then
                processing it. When it is specified, nothing is returned.
            fast_mode (bool): Whether to allow the fast mode described above.
            max_B (int): The maximum box size (i.e. $L^\infty$ norm) allowed in
                the fast mode.
            verbose (bool): Whether to print extra diagnostic information.

        Returns:
            np.ndarray | None: The lattice points, one per row. `None` is returned
                if `process_function` is specified, or if the search with `max_deg`
                was unsuccessful (in which case a message is printed).

        Raises:
            Exception: If neither `min_points` nor `max_deg` is specified, if both
                `filter_function` and `process_function` are specified, if the
                general algorithm is used on a cone that is not pointed, or if the
                grading vector is not valid.
            ValueError: If `max_coord` is specified and not enough points can be
                found within the corresponding box.

        Examples:
            We construct a cone and find at least 20 lattice points in it.

            >>> c = Cone([[3,2],[5,3]])
            >>> pts = c.find_lattice_points(min_points=20)
            >>> len(pts)
            20

            We can also find all the points up to a given degree. Here the grading
            vector is computed automatically.

            >>> c.find_lattice_points(max_deg=2)
            array([[ 0,  0],
                   [ 3,  2],
                   [ 5,  3],
                   [ 6,  4],
                   [ 8,  5],
                   [10,  6]])

            Let's also give an example where we use a function to apply some
            filtering. This can be something very complicated, but here we just
            pick the points where all coordinates are odd.

            >>> def odd_coords(pt):
            ...     return all(c%2 for c in pt)
            >>> pts = c.find_lattice_points(min_points=20, filter_function=odd_coords)
            >>> len(pts)  # now we get only 6 points
            6

            Finally, let's give an example where we process the data as it comes
            instead of first constructing a list. In this simple example we just
            print each point with odd coordinates, but in general it can be a
            complex algorithm.

            >>> def process_function(pt):
            ...     if all(c%2 for c in pt):
            ...         print(f"Processing point {pt}")
            >>> c.find_lattice_points(min_points=20, process_function=process_function)
            Processing point (5, 3)
            Processing point (11, 7)
            Processing point (15, 9)
            Processing point (17, 11)
            Processing point (21, 13)
            Processing point (25, 15)
        """
        # initial checks
        if max_deg is None and min_points is None:
            raise Exception(
                "Either the maximum degree or the minimum number of points must be specified."
            )

        if process_function is not None and filter_function is not None:
            raise Exception(
                "Only one of filter_function or process_function can be specified."
            )

        # Shortcut if min_points is set and dim is low.
        #
        # The shortcut enumerates by Linf norm rather than by degree, and it
        # returns the points in bulk, so it cannot honor max_deg,
        # grading_vector, filter_function, or process_function. Rather than
        # silently ignoring them, we simply fall back to the general algorithm
        # whenever any of them is given.
        can_use_fast_mode = all(
            arg is None
            for arg in (max_deg, grading_vector, filter_function, process_function)
        )
        if (
            fast_mode
            and (min_points is not None)
            and (self.ambient_dim() <= 10)
            and can_use_fast_mode
        ):
            return np.array(
                latticepts.enum_lattice_points(
                    H=self.hyperplanes(),
                    rhs=c,
                    min_N_pts=min_points,
                    max_B=max_B,
                )
            )

        if not self.is_pointed():
            raise Exception("Only pointed cones are currently supported.")

        if grading_vector is None:
            grading_vector = self.find_grading_vector()
        finite_max_coord = max_coord is not None
        if max_coord is None:
            max_coord = cp_model.INT32_MAX - 1

        hp = self.hyperplanes()

        # We start by defining a class that will store the points we find
        class SolutionStorage(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables, filter_function=None, process_function=None):
                super().__init__()
                self._variables = variables
                self._solutions = set()
                self._filter_function = filter_function
                self._process_function = process_function
                self._n_sol = 0

        # We now define various versions of the on_solution_callback method for
        # the different scenarios. The reason for having multiple functions
        # instead of having various if statements in a single function is that,
        # since it will be run many times, it is very inefficient to keep
        # checking the conditions even though they will never change. This
        # first method is for when we want to check that it is a pointed cone
        # with a good grading vector
        class MoreThanOneSolution(Exception):
            pass

        def on_soln_callback_single_pt(self):
            self._n_sol += 1
            if self._n_sol > 1:
                raise MoreThanOneSolution

        # This one is the standard one that will be used
        def on_soln_callback_default(self):
            self._n_sol += 1
            self._solutions.add(tuple(self.Value(v) for v in self._variables))

        # This one will be used when a custom filtering is specified
        def on_soln_callback_filter(self):
            self._n_sol += 1
            point = tuple(self.Value(v) for v in self._variables)
            if self._filter_function(point):
                self._solutions.add(point)

        def on_soln_callback_process(self):
            self._n_sol += 1
            process_function(tuple(self.Value(v) for v in self._variables))

        # If pointed cone, first check that we have a good grading vector
        if self.is_pointed():
            solver = cp_model.CpSolver()
            model = cp_model.CpModel()

            # define variables
            var = [
                model.NewIntVar(-max_coord, max_coord, f"x_{i}")
                for i in range(hp.shape[1])
            ]

            # define constraints
            for v in hp:
                model.Add(sum(ii * var[i] for i, ii in enumerate(v)) >= 0)
            model.Add(sum(ii * var[i] for i, ii in enumerate(grading_vector)) <= 0)

            SolutionStorage.on_solution_callback = on_soln_callback_single_pt
            solution_storage = SolutionStorage(var, filter_function, process_function)

            try:
                status = solver.SearchForAllSolutions(model, solution_storage)
            except MoreThanOneSolution:
                raise Exception(
                    "More than one solution was found. The grading"
                    " vector must be wrong."
                )

        # Now, construct the solution storage that will hold the points we find
        if filter_function is not None:
            SolutionStorage.on_solution_callback = on_soln_callback_filter
        elif process_function is not None:
            SolutionStorage.on_solution_callback = on_soln_callback_process
        else:
            SolutionStorage.on_solution_callback = on_soln_callback_default

        if not isinstance(c, Iterable):
            c_vals = [c] * len(hp)
        else:
            c_vals = list(c)

        def make_lattice_model():
            # define the model
            solver = cp_model.CpSolver()
            model = cp_model.CpModel()

            # define variables
            var = [
                model.NewIntVar(-max_coord, max_coord, f"x_{i}")
                for i in range(hp.shape[1])
            ]

            # define constraints
            for h, cc in zip(hp, c_vals):
                # clear the denominator
                cc_rat = Fraction(cc).limit_denominator()
                denom = cc_rat.denominator
                numer = cc_rat.numerator

                # add the constraint
                model.Add(sum(ii * var[i] * denom for i, ii in enumerate(h)) >= numer)

            soln_deg = sum(ii * var[i] for i, ii in enumerate(grading_vector))
            return solver, model, var, soln_deg

        solver, model, var, soln_deg = make_lattice_model()
        solution_storage = SolutionStorage(var, filter_function, process_function)

        finite_coord_degree_cap = None
        if finite_max_coord and max_deg is None:
            cap_solver, cap_model, _, cap_soln_deg = make_lattice_model()
            cap_model.Maximize(cap_soln_deg)
            cap_status = cap_solver.Solve(cap_model)
            if cap_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                finite_coord_degree_cap = int(cap_solver.ObjectiveValue())
            else:
                raise ValueError(
                    f"Could not find lattice points within the finite "
                    f"max_coord={max_coord} search box. Increase max_coord "
                    "or leave it unset to use the automatic CP-SAT bound."
                )

        # solve according to whether max_deg or min_points was specified
        if max_deg is not None:
            # If the maximum degree is specified, we use it as a constraint
            model.Add(soln_deg <= max_deg)

            # solve and check status
            status = solver.SearchForAllSolutions(model, solution_storage)
            if status != cp_model.OPTIMAL:
                print(
                    "There was a problem finding the points. Status code: "
                    f"{solver.StatusName(status)}"
                )
                return
        else:
            # Else, add points until the minimum number is reached
            deg = 0
            while solution_storage._n_sol < min_points:
                if (
                    finite_coord_degree_cap is not None
                    and deg > finite_coord_degree_cap
                ):
                    raise ValueError(
                        f"Could not find {min_points} lattice points within "
                        f"the finite max_coord={max_coord} search box; found "
                        f"{solution_storage._n_sol}. Increase max_coord or "
                        "leave it unset to use the automatic CP-SAT bound."
                    )

                # define model with windowed degree constraints
                window_model = deepcopy(model)
                window_model.Add(deg <= soln_deg)
                window_model.Add(soln_deg <= deg + deg_window)

                # solve and check status
                status = solver.SearchForAllSolutions(window_model, solution_storage)
                if verbose and status != cp_model.OPTIMAL:
                    print(
                        "There was a problem finding the points b/t degrees "
                        f"{deg} and {deg + deg_window}. "
                        f"Status code: {solver.StatusName(status)}"
                    )

                deg += deg_window + 1

        # parse solutions
        if process_function is not None:
            return
        pts = np.array(list(solution_storage._solutions), dtype=int)

        # provide uniform sorting of points
        degs = pts @ grading_vector

        out = []
        for deg in sorted(set(degs)):
            out.append(sorted(pts[degs == deg].tolist()))

        return np.vstack(out)

    def is_solid(self, backend=None):
        """Return whether the cone is solid, i.e. whether it is full-dimensional.

        If the generating rays are known, this is simply checked by computing
        the dimension of the linear space that they span, and `backend` is
        ignored. When only the hyperplane inequalities are known this can be a
        more difficult problem. When using PPL as the backend, the cone is
        explicitly constructed and its dimension is checked. The other backends
        try to find a point in the strict interior of the cone with
        [`find_interior_point`][cytools.cone.Cone.find_interior_point], which
        fails if the cone is not solid. The latter approach is much faster, but
        there could be extremely narrow cones where the optimization fails and
        this function returns a false negative. The result is cached.

        Also available as `is_full_dimensional`.

        Args:
            backend (str | None): The backend to use when the rays are not
                known. The options are `"ppl"`, and any of the backends of
                [`find_interior_point`][cytools.cone.Cone.find_interior_point]. If
                not specified, the default backend of
                [`find_interior_point`][cytools.cone.Cone.find_interior_point] is
                used.

        Returns:
            bool: The truth value of the cone being solid.

        Raises:
            ValueError: If an invalid backend is specified.

        Examples:
            We construct two cones and check if they are solid.

            >>> c1 = Cone([[1,0],[0,1]])
            >>> c2 = Cone([[1,0,0],[0,1,0]])
            >>> c1.is_solid()
            True
            >>> c2.is_solid()
            False
        """
        # check for cached answer or if we have rays... makes calc is easy
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            self._is_solid = bool(_rank(self._rays) == self._ambient_dim)
            return self._is_solid

        # we just have hyperplanes... a bit harder
        backends = (
            None,
            "ppl",
            "highs",
            "glop",
            "scip",
            "cpsat",
            "mosek",
            "osqp",
            "cvxopt",
        )
        if backend not in backends:
            raise ValueError(f"Invalid backend. The options are: {backends}.")

        # solve according to backend
        if backend == "ppl":
            cs = ppl.Constraint_System()

            vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
            for h in self._hyperplanes:
                cs.insert(sum(h[i] * vrs[i] for i in range(self._ambient_dim)) >= 0)
            cone = ppl.C_Polyhedron(cs)

            self._is_solid = cone.affine_dimension() == self._ambient_dim
        else:
            # Otherwise we check this by trying to find an interior point
            interior_point = self.find_interior_point(show_hints=False, backend=backend)
            self._is_solid = interior_point is not None

        return self._is_solid

    # aliases
    is_full_dimensional = is_solid

    def is_pointed(self, backend: str = "dual", tol: float = 1e-7) -> bool:
        r"""Return whether the cone is pointed (i.e. strongly convex).

        A cone is pointed if it contains no line, i.e. if there is no non-zero
        $x$ such that both $x$ and $-x$ are in the cone. The available backends
        are:

        - `"dual"`: checks whether the dual cone is solid (see
          [`is_solid`][cytools.cone.Cone.is_solid]). This works for any cone.
        - `"null"`: checks whether the matrix of hyperplane normals $H$ has full
          rank, since otherwise any non-zero $x$ with $Hx=0$ spans a line in the
          cone. This requires the hyperplanes to be known.
        - `"lp"` and `"nnls"`: check whether there are non-negative coefficients
          $\lambda\neq 0$ such that $\sum_i\lambda_i r_i=0$ for the generating
          rays $r_i$, using a linear program or non-negative least squares,
          respectively. These can only be used when the hyperplanes are not
          known.

        The result is cached, regardless of the backend used.

        Also available as `is_strongly_convex`.

        Args:
            backend (str): The backend to use. The options are `"dual"`, `"null"`,
                `"lp"`, and `"nnls"`, as described above.
            tol (float): The tolerance for determining when a linear subspace is
                found. This is only used by the `"nnls"` backend.

        Returns:
            bool: The truth value of the cone being pointed.

        Raises:
            ValueError: If the backend is unknown, or if it cannot be used with
                the known description of the cone.

        Examples:
            We construct two cones and check if they are pointed.

            >>> c1 = Cone([[1,0],[0,1]])
            >>> c2 = Cone([[1,0],[0,1],[-1,0]])
            >>> c1.is_pointed()
            True
            >>> c2.is_pointed()
            False
        """
        if self._is_pointed is not None:
            return self._is_pointed

        # duality based check
        if backend.lower() == "dual":
            self._is_pointed = self.dual().is_solid()

        # ray-based analysis (only if we have no hyperplanes)
        elif self._hyperplanes is None:
            # check if some lmbda!=0 exists such that R.T@lmbda==0
            # N.B.: this is equiv to [R; 1]@lmbda=[0; 1] for any lmbda>=0
            #       (akin to homogenization...)
            R = self._rays.copy().T
            R = np.vstack((R, np.ones((1, R.shape[1]), dtype=int)))
            b = [0] * (R.shape[0] - 1) + [1]

            # allow different backends
            if backend.lower() == "nnls":
                self._is_pointed = nnls(R, b)[1] > tol
            elif backend.lower() == "lp":
                res = linprog(
                    # no objective (one variable per ray, i.e. per column of R)
                    c=np.zeros(R.shape[1], dtype=int),
                    A_eq=R,
                    b_eq=b,  # [R; 1] lmbda = [0;1]
                    bounds=[(0, None)],  # lmbda >= 0
                    method="highs",
                )
                self._is_pointed = not res.success
            else:
                raise ValueError(f"backend '{backend.lower()}' not known for V-cones")

        # hyperplane-based analyis (likely easiest...)
        else:
            if backend.lower() == "null":
                H_rank = np.linalg.matrix_rank(self.hyperplanes())
                self._is_pointed = bool(H_rank == self.ambient_dim())
            else:
                raise ValueError(f"backend '{backend.lower()}' not known for H-cones")

        return self._is_pointed

    # aliases
    is_strongly_convex = is_pointed

    def is_simplicial(self):
        """Return whether the cone is simplicial.

        A cone is simplicial if its extremal rays are linearly independent, i.e.
        if the number of extremal rays equals the dimension of the cone. If the
        rays are not known and the cone is solid, it is instead checked whether
        the number of extremal hyperplanes equals the dimension, which is
        equivalent in that case. The result is cached.

        Returns:
            bool: The truth value of the cone being simplicial.

        Examples:
            We construct two cones and check if they are simplicial.

            >>> c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
            >>> c2 = Cone([[1,0,0],[0,1,0],[0,0,1],[1,1,-1]])
            >>> c1.is_simplicial()
            True
            >>> c2.is_simplicial()
            False
        """
        if self._is_simplicial is not None:
            return self._is_simplicial

        # split analysis by whether we know rays or not
        if (self._rays is None) and (self.is_solid()):
            self._is_simplicial = len(self.extremal_hyperplanes()) == self.dim()
        else:
            self._is_simplicial = len(self.extremal_rays()) == self.dim()

        return self._is_simplicial

    def is_degenerate(
        self,
        use_extremal_hyperplanes: bool = True,
        M: int = None,
        certificate: bool = False,
        verbosity: int = 0,
    ):
        r"""Check whether the cone is degenerate.

        For a cone $\{x : Hx\geq 0\}$ in $d$ ambient dimensions, this checks
        whether some non-zero $x$ in the cone saturates (i.e. satisfies
        $h\cdot x=0$) at least $d+1$ of the hyperplanes $h$ simultaneously. If
        so, the cone is considered degenerate. It is more difficult to compute
        the rays of a degenerate cone.

        This is representation-sensitive: a cone may be degenerate for a certain
        matrix of hyperplane normals $H$ but not for another one that defines
        the same cone. It is therefore best to use the extremal hyperplanes.

        The check first tries $x=(1,\ldots,1)$, and otherwise searches for such
        an $x$ among the integer vectors with OR-Tools CP-SAT.

        Args:
            use_extremal_hyperplanes (bool): Whether to check the extremal
                hyperplanes for degeneracy. If false, the hyperplanes returned by
                [`hyperplanes`][cytools.cone.Cone.hyperplanes] are used.
            M (int | None): The bound on the absolute value of the coordinates of
                $x$ in the search. If not specified, the largest bound supported by
                CP-SAT is used.
            certificate (bool): Whether to also return the point $x$ that was
                found and the hyperplanes that the solver claims it saturates.
            verbosity (int): The verbosity level.

        Returns:
            bool | tuple | None: Whether the cone is degenerate, or `None` if
                the solver failed. If `certificate` is true, a tuple
                `(degenerate, (x, z))` is returned instead, where `x` is the
                point that was found and `z` is an array of 0s and 1s indicating
                which hyperplanes it saturates (both can be `None` if no point
                was found).

        Examples:
            We construct a cone over a square and a cone over a pyramid with a
            pentagonal base, and check whether they are degenerate. The apex of the
            pyramid lies on five facets.

            >>> c1 = Cone([[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]])
            >>> bool(c1.is_degenerate())
            False
            >>> c2 = Cone([[1,0,0,1],[0,1,0,1],[-1,1,0,1],[-1,0,0,1],[0,-1,0,1],
            ...            [0,0,1,1]])
            >>> bool(c2.is_degenerate())
            True
        """
        if use_extremal_hyperplanes:
            H = self.extremal_hyperplanes()
        else:
            H = self.hyperplanes()

        # try a common representative of degeneracy
        xtest = np.ones(self.ambient_dim(), dtype=int)
        dists = H @ xtest
        z = dists == 0
        if sum(z) >= self.ambient_dim() + 1:
            degen, x, z = True, xtest, z
        else:
            # _is_degenerate is an minimal, non-Cone method doing the check
            out = _is_degenerate(H=H, M=M, certificate=certificate, verbosity=verbosity)

            if certificate:
                degen, (x, z) = out
            else:
                degen = out

        # return
        if certificate:
            return degen, (x, z)
        else:
            return degen

    def is_smooth(self):
        """Return whether the cone is smooth.

        A cone is smooth if it is simplicial and its extremal rays either form a
        basis of the ambient lattice, or they can be extended into one. The
        result is cached.

        Returns:
            bool: The truth value of the cone being smooth.

        Examples:
            We construct two cones and check if they are smooth.

            >>> c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
            >>> c2 = Cone([[2,0,1],[0,1,0],[1,0,2]])
            >>> c1.is_smooth()
            np.True_
            >>> c2.is_smooth()
            np.False_
        """
        if self._is_smooth is not None:
            return self._is_smooth
        if not self.is_simplicial():
            self._is_smooth = False
            return self._is_smooth
        if self.is_solid():
            self._is_smooth = abs(abs(np.linalg.det(self.extremal_rays())) - 1) < 1e-4
            return self._is_smooth
        snf = np.array(
            fmpz_mat(self.extremal_rays().tolist()).snf().tolist(), dtype=int
        )
        self._is_smooth = abs(np.prod([snf[i, i] for i in range(len(snf))])) == 1
        return self._is_smooth

    def lineality_space(self):
        """Return the lineality space of the cone as a `Cone` object.

        The lineality space is the largest linear subspace contained in the
        cone, i.e. the set of $x$ such that $Hx=0$, where $H$ is the matrix of
        hyperplane normals. Since it is a linear subspace, packaging it as a
        cone is somewhat formal: it is returned as the cone generated by an
        integral basis of the subspace together with minus the sum of the basis
        vectors, and these rays are also set as its extremal rays. For a pointed
        cone, the lineality space is the origin.

        Returns:
            Cone: The lineality space of the cone.

        Examples:
            We construct a cone that is a half-plane and find its lineality space.

            >>> c = Cone([[1,0],[0,1],[-1,0]])
            >>> c.lineality_space().rays()
            array([[ 1,  0],
                   [-1,  0]])
        """
        H = self.hyperplanes()

        # the lineality space is defined by the x such that H@x==0
        # (the following definition is extremely redundant, so it's only listed
        #  for pedagogical purposes. It's better to define the cone via rays
        #  and then compute the hyperplanes via DDM since there will only be 6
        #  rays, since lineality space should typically be 5D)
        # lin = Cone(hyperplanes = np.vstack([H,-H]))

        # linearly spanning vectors are given by null(H)
        R = utils.integral_nullspace(H).T

        # to map to positively spanning rays, add in the ray r=np.sum(axis=0)
        r = -np.sum(R, axis=0)
        r = r // utils.gcd_list(r)
        R = np.vstack([R, [r]])

        lin = Cone(rays=R)

        # save the extremal rays manually
        # (this is split into two saves since _ext_rays stores both the naive
        #  extremal rays [i.e., a subset of _rays] at index 0 and the minimal
        #  extremal rays  at index 1)
        lin._ext_rays[0] = R.copy()
        lin._ext_rays[1] = R.copy()

        return lin

    def pointed_space(self):
        """Return the pointed part of the cone.

        A cone can be decomposed as the sum of its lineality space (see
        [`lineality_space`][cytools.cone.Cone.lineality_space]) and a pointed
        cone. The pointed part is obtained by intersecting the cone with the
        orthogonal complement of the lineality space, and it is returned as a
        cone defined by hyperplanes.

        Returns:
            Cone: The pointed part of the cone.

        Examples:
            We construct a cone that is a half-plane and find its pointed part.

            >>> c = Cone([[1,0],[0,1],[-1,0]])
            >>> c.pointed_space().rays()
            array([[0, 1]])
        """
        H = self.hyperplanes()

        # linearly spanning vectors of the lineality space
        # (don't need to add -\\sum_i r_i since we're dealing with linear spans)
        R = utils.integral_nullspace(H).T

        # The hyperplanes defining the orthogonal complement are just [R, -R].
        # This is because
        # R@x==0 <=> y@R@x==0 (for all y)
        #        <=> r.x==0   (for all r in the rowspan of R... lineality space)

        # the pointed part is just the intersection with these hyperplanes
        pointed = Cone(hyperplanes=np.vstack([H, R, -R]))
        return pointed

    def hilbert_basis(self):
        """Return the Hilbert basis of the cone.

        The Hilbert basis is the minimal set of lattice points that generate
        all the lattice points in the cone as non-negative integer
        combinations. [Normaliz](https://www.normaliz.uni-osnabrueck.de/) is
        used for the computation, so its `normaliz` executable must be installed
        and available on the `PATH`. The result is cached.

        Returns:
            np.ndarray: The vectors forming the Hilbert basis, one per row.

        Raises:
            RuntimeError: If the `normaliz` executable is not found.

        Examples:
            We compute the Hilbert basis of a two-dimensional cone.

            >>> c = Cone([[1,3],[2,1]])
            >>> c.hilbert_basis()
            array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [2, 1]])
        """
        if self._hilbert_basis is not None:
            return np.array(self._hilbert_basis)
        if shutil.which("normaliz") is None:
            raise RuntimeError(
                "Hilbert basis computation requires the external 'normaliz' "
                "executable to be installed and available on PATH."
            )
        # Generate a random project name so that it doesn't conflict with
        # other computations
        letters = string.ascii_lowercase
        proj_name = "cytools_" + "".join(random.choice(letters) for i in range(10))

        rays = self.rays()
        with open(f"/tmp/{proj_name}.in", "w+") as f:
            f.write(f"amb_space {rays.shape[1]}\ncone {rays.shape[0]}\n")
            f.write(
                str(rays.tolist())
                .replace("],", "\n")
                .replace(",", "")
                .replace("[", "")
                .replace("]", "")
                + "\n"
            )

        normaliz = subprocess.Popen(
            ("normaliz", f"/tmp/{proj_name}.in"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        normaliz.communicate()
        with open(f"/tmp/{proj_name}.out") as f:
            data = f.readlines()
        os.remove(f"/tmp/{proj_name}.in")
        os.remove(f"/tmp/{proj_name}.out")
        rays = []
        found_stars = False
        l_n = 0
        while True:
            if l_n >= len(data):
                break
            line = data[l_n]
            if "******" in line:
                found_stars = True
                l_n += 1
                continue
            if not found_stars:
                l_n += 1
                continue
            if "lattice points in polytope" in line or "Hilbert basis elements" in line:
                n_rays = literal_eval(line.split()[0])
                for i in range(n_rays):
                    rays.append([literal_eval(c) for c in data[l_n + 1 + i].split()])
                l_n += n_rays + 1
                continue
            if "further Hilbert basis elements" in line:
                n_rays = literal_eval(line.split()[0])
                for i in range(n_rays):
                    rays.append([literal_eval(c) for c in data[l_n + 1 + i].split()])
                l_n += n_rays + 1
                continue
            l_n += 1
            continue
        self._hilbert_basis = np.array(rays)
        return np.array(self._hilbert_basis)

    def intersection(self, other):
        """Compute the intersection with another cone, or with a list of cones.

        The resulting cone is defined by the combined hyperplane normals of all
        the cones.

        Args:
            other (Cone | Iterable[Cone]): The other cone that is being
                intersected, or a list of cones to intersect with.

        Returns:
            Cone: The cone that results from the intersection.

        Raises:
            ValueError: If `other` is a list containing objects that are not
                cones, or cones with a different ambient dimension.

        Examples:
            We construct two cones and find their intersection.

            >>> c1 = Cone([[1,0],[1,2]])
            >>> c2 = Cone([[0,1],[2,1]])
            >>> c3 = c1.intersection(c2)
            >>> c3.rays()
            array([[2, 1],
                   [1, 2]])
        """
        if isinstance(other, Cone):
            return Cone(
                hyperplanes=self.hyperplanes().tolist() + other.hyperplanes().tolist()
            )

        hyperplanes = self.hyperplanes().tolist()
        for c in other:
            if not isinstance(c, Cone):
                raise ValueError("Elements of the list must be Cone objects.")
            if c.ambient_dim() != self.ambient_dim():
                raise ValueError("Ambient lattices must have the samedimension.")
            hyperplanes.extend(c.hyperplanes().tolist())
        return Cone(hyperplanes=hyperplanes)


def dualize(M, verbosity=0):
    r"""Convert between the hyperplane and ray descriptions of a cone.

    The matrix `M` can be interpreted as the inward-pointing hyperplane
    normals of the cone $\{x : Mx\geq 0\}$, in which case the returned
    matrix $R$ contains rays that generate it, i.e. the cone is
    $\{R^T\lambda : \lambda\geq 0\}$. Equivalently, by cone duality, `M`
    can be interpreted as the generating rays of the cone
    $\{M^T\lambda : \lambda\geq 0\}$, in which case the returned matrix
    contains the hyperplane normals that define it. The computation is done
    with PPL, treating `M` as hyperplanes.

    The output is a minimized description, but it is not guaranteed to
    consist of extremal rays. In particular, a lineality space is
    represented by pairs of opposite vectors $\pm v$.

    Args:
        M (ArrayLike): The matrix defining the cone, one vector per row.
        verbosity (int): The verbosity level.

    Returns:
        np.ndarray: The dual description of the cone, one vector per row. If
            there are no vectors, an empty array of shape `(0, M.shape[1])` is
            returned.

    Examples:
        We find the rays of the cone defined by $x_1\geq 0$ and
        $x_1+x_2\geq 0$.

        >>> from cytools.cone import dualize
        >>> dualize([[1,0],[1,1]])
        array([[ 0,  1],
               [ 1, -1]])
    """
    M = np.asarray(M)

    # define the cone in PPL
    if verbosity >= 1:
        print("Defining the cone in PPL...", flush=True)

    cone = ppl.C_Polyhedron(M.shape[1])

    for row in M:
        ineq = ppl.Linear_Expression(row.tolist(), 0)
        cone.add_constraint(ppl.Constraint(ineq >= 0))

    # grab the dual description (in this perspective, the rays)
    if verbosity >= 1:
        print("Computing the rays...", flush=True)
    rays = []
    for gen_i, gen in enumerate(cone.minimized_generators()):
        if verbosity >= 2:
            print(f"ray #{gen_i}...", end="\r")

        if gen.is_ray():
            rays.append(tuple(int(c) for c in gen.coefficients()))
        elif gen.is_line():
            # lineality space... add both signs
            rays.append(tuple(int(c) for c in gen.coefficients()))
            rays.append(tuple(-int(c) for c in gen.coefficients()))

    # return
    # (np.array([]) would have shape (0,), which breaks downstream matrix
    #  operations, so build the empty answer with the correct 2nd dimension)
    if len(rays) == 0:
        return np.zeros((0, M.shape[1]), dtype=int)
    return np.array(rays, dtype=int)


def is_extremal(
    R: "ArrayLike",
    i: int,
    extFlags: list[bool] = None,
    method: str = "lp",
    tol: float = 1e-4,
) -> (int, bool, "Exception"):
    """Check whether a ray is extremal among a set of rays.

    This is an auxiliary function used by
    [`Cone.extremal_rays`][cytools.cone.Cone.extremal_rays]. The ray `R[i]`
    is extremal if it cannot be written as a non-negative linear combination
    of the other rays. The output includes the index of the ray and any
    exception that was raised, so that it can be used when parallelizing.

    Args:
        R (ArrayLike): A matrix whose rows are the rays of the cone.
        i (int): The index of the ray to check for extremality.
        extFlags (list[bool] | None): Flags indicating which rays are possibly
            extremal. Rays whose flag is false are known to be non-extremal
            and are excluded from the comparison. If not specified, all the
            other rays are used.
        method (str): The method used to check extremality. The options are
            `"lp"` (a linear program, recommended) and `"nnls"` (non-negative
            least squares).
        tol (float): The tolerance for determining whether a ray is extremal.
            Only used if `method` is `"nnls"`.

    Returns:
        tuple[int, bool | None, Exception | None]: The index `i`, the truth
            value of the ray being extremal, and `None`. If an exception was
            raised during the computation, the truth value is `None` and the
            exception is returned as the third element.

    Examples:
        This function is not meant to be directly used by the end user. We
        check whether two rays are extremal.

        >>> import numpy as np
        >>> from cytools.cone import is_extremal
        >>> R = np.array([[0,1],[1,1],[1,0]])
        >>> is_extremal(R, 0)
        (0, True, None)
        >>> is_extremal(R, 1)
        (1, False, None)
    """
    try:
        # the ray to check if it's extremal
        r = R[i]

        # get the other rays (trim by those which are known non-extremal)
        if extFlags is None:
            R = np.delete(R, i, axis=0)
        else:
            R = np.delete(R, i, axis=0)[np.delete(extFlags, i)]

        # check if it's extremal
        if method.lower() == "lp":
            res = linprog(
                c=np.zeros(R.shape[0], dtype=int),  # no objective
                A_eq=R.T,
                b_eq=r,  # (R\r) lmbda = r
                bounds=[(0, None)],  # lmbda >= 0
                method="highs",
            )
            return (i, not res.success, None)

        elif method.lower() == "nnls":
            v = nnls(R.T, r)
            return (i, abs(v[1]) > tol, None)
    except Exception as e:
        return (i, None, e)


def feasibility(
    hyperplanes: "ArrayLike",
    c: float,
    ambient_dim: int,
    backend: str,
    lower_bound: float = None,
    verbose: bool = False,
):
    r"""Solve the feasibility problem $Ax\geq c$.

    A point $x$ is found such that $a\cdot x\geq c$ for each row $a$ of
    $A$, minimizing the average of the $a\cdot x$ (so that the point tends
    to lie close to the tip of the cone), and optionally subject to a lower
    bound on each coordinate.

    Args:
        hyperplanes (ArrayLike): The constraining hyperplanes, $A$, one per
            row. The rows can also be sparse, given as objects whose `items()`
            method yields `(index, value)` pairs.
        c (float): The "stretching", i.e. the minimum value of $a\cdot x$ for
            each row $a$.
        ambient_dim (int): The ambient dimension, i.e. the number of columns
            of $A$.
        backend (str): The backend to use. The options are `"highs"` (an LP
            via HiGHS), `"glop"` (an LP via OR-Tools), `"scip"` (an integer
            program via OR-Tools), and `"cpsat"` (an integer program via
            OR-Tools CP-SAT).
        lower_bound (float | None): A lower bound on each of the variables. If
            not specified, the variables are unbounded below.
        verbose (bool): Whether to print extra diagnostic information.

    Returns:
        np.ndarray | None: A feasible point, if one was found, or `None`
            otherwise. If there are no hyperplanes, a vector of ones is
            returned.

    Examples:
        We find a point satisfying $x_1\geq 1$ and $-x_1+x_2\geq 1$.

        >>> from cytools.cone import feasibility
        >>> feasibility([[1,0],[-1,1]], c=1, ambient_dim=2, backend="highs")
        array([1., 2.])
    """
    if isinstance(hyperplanes, (list, np.ndarray)):
        hyperplanes = np.asarray(hyperplanes)
        hp_iter = enumerate
    else:

        def hp_iter(hp):
            return hp.items()

    # accommodate trivial hyperplanes
    if len(hyperplanes) == 0:
        return np.ones(ambient_dim)

    def col_sums():
        """Column sums of the hyperplanes. Works for dense and sparse rows."""
        if isinstance(hyperplanes, np.ndarray):
            return hyperplanes.sum(axis=0)

        out = np.zeros(ambient_dim)
        for v in hyperplanes:
            for ind, val in hp_iter(v):
                out[int(ind)] += float(val)
        return out

    if backend == "highs":
        # LP feasibility via HiGHS
        n = ambient_dim
        if isinstance(hyperplanes, np.ndarray):
            # dense: assemble the CSR structure with numpy instead of a Python
            # double loop over every (row, column) entry
            m = len(hyperplanes)
            grading = hyperplanes.sum(axis=0) / m
            starts = np.arange(m, dtype=np.int32) * n
            index = np.tile(np.arange(n, dtype=np.int32), m)
            value = hyperplanes.ravel().astype(float)
        else:
            # sparse rows (e.g. LIL): iterate only the stored entries
            starts, index, value = [], [], []
            grading = np.zeros(n)
            for v in hyperplanes:
                starts.append(len(index))
                for ind, val in hp_iter(v):
                    index.append(int(ind))
                    value.append(float(val))
                    grading[int(ind)] += float(val)

            grading /= len(starts)
            starts = np.asarray(starts, dtype=np.int32)
            index = np.asarray(index, dtype=np.int32)
            value = np.asarray(value, dtype=float)

        inf = highspy.kHighsInf
        lb = -inf if lower_bound is None else float(lower_bound)
        h = highspy.Highs()
        if not verbose:
            h.silent()
        h.addVars(n, np.full(n, lb), np.full(n, inf))
        h.changeColsCost(n, np.arange(n, dtype=np.int32), grading)
        h.addRows(
            len(starts),
            np.full(len(starts), float(c)),
            np.full(len(starts), inf),
            len(index),
            starts,
            index,
            value,
        )
        h.run()
        status = h.getModelStatus()
        if status == highspy.HighsModelStatus.kOptimal:
            return np.asarray(h.getSolution().col_value, dtype=float)
        if verbose and status != highspy.HighsModelStatus.kInfeasible:
            warnings.warn(f"Solver returned status {status!r}.")
        return None

    if backend in ("glop", "scip"):
        solver = pywraplp.Solver.CreateSolver(backend.upper())

        if verbose:
            # enable solver to print output
            solver.EnableOutput()

        # define variables
        var_type = solver.NumVar if backend == "glop" else solver.IntVar
        if lower_bound is None:
            lower = -solver.infinity()
        else:
            lower = lower_bound
        var = [
            (var_type)(lower, solver.infinity(), f"x_{i}") for i in range(ambient_dim)
        ]

        # define constraints
        cons_list = []
        for v in hyperplanes:
            cons_list.append(solver.Constraint(c, solver.infinity()))
            for ind, val in hp_iter(v):
                cons_list[-1].SetCoefficient(var[ind], float(val))

        # define objective
        obj = solver.Objective()
        obj.SetMinimization()

        obj_vec = col_sums() / len(hyperplanes)
        for i in range(ambient_dim):
            obj.SetCoefficient(var[i], float(obj_vec[i]))

        # solve and parse solution
        status = solver.Solve()
        if status in (solver.FEASIBLE, solver.OPTIMAL):
            solution = np.array([x.solution_value() for x in var])
        elif status == solver.INFEASIBLE:
            if verbose:
                warnings.warn("Solver returned status INFEASIBLE.")
            return None
        else:
            status_list = [
                "OPTIMAL",
                "FEASIBLE",
                "INFEASIBLE",
                "UNBOUNDED",
                "ABNORMAL",
                "MODEL_INVALID",
                "NOT_SOLVED",
            ]
            warnings.warn(f"Solver returned status {status_list[status]}.")
            return None

    elif backend == "cpsat":
        solver = cp_model.CpSolver()
        model = cp_model.CpModel()

        # define variables
        var = []
        if lower_bound is None:
            lower = cp_model.INT32_MIN
        else:
            # the variables are integral, so x >= lower_bound <=> x >= ceil(..)
            lower = int(np.ceil(lower_bound))
        for i in range(ambient_dim):
            var.append(model.NewIntVar(lower, cp_model.INT32_MAX, f"x_{i}"))

        # define constraints
        for v in hyperplanes:
            model.Add(sum(val * var[int(ind)] for ind, val in hp_iter(v)) >= c)

        # define objective
        obj_vec = np.rint(col_sums()).astype(np.int64)
        obj_gcd = utils.gcd_list(obj_vec)
        if obj_gcd:
            obj_vec //= obj_gcd

        obj = 0
        for i in range(ambient_dim):
            obj += var[i] * obj_vec[i]

        model.Minimize(obj)

        # solve and parse solution
        status = solver.Solve(model)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            solution = np.array([solver.Value(x) for x in var])
        elif status == cp_model.INFEASIBLE:
            if verbose:
                warnings.warn("Solver returned status INFEASIBLE.")
            return None
        else:
            warnings.warn(f"Solver returned status {solver.StatusName(status)}.")
            return None

    return solution


# cone degeneracy
# ---------------
@functools.cache
def _early_stop_callback_cls():
    """Build (and cache) a CP-SAT callback that stops the search early.

    The callback stops the search once the objective reaches a given
    threshold. It is defined in a function rather than at module scope
    because its base class lives in `ortools.sat`, which is imported
    lazily.

    Returns:
        type: The `EarlyStopCallback` class, whose constructor takes the
            objective threshold and the solver.
    """

    class EarlyStopCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, threshold, solver):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._threshold = threshold
            self._solver = solver

        def on_solution_callback(self):
            current_value = int(self.ObjectiveValue())
            if current_value >= self._threshold:
                self.StopSearch()

    return EarlyStopCallback


def __getattr__(name):
    """Return module attributes that are built on demand.

    Args:
        name (str): The name of the requested attribute.

    Returns:
        type: The `EarlyStopCallback` class, if `name` is `"EarlyStopCallback"`.

    Raises:
        AttributeError: If the module has no attribute `name`.
    """
    # keeps `cytools.cone.EarlyStopCallback` working now that the class is
    # only built on demand (see PEP 562)
    if name == "EarlyStopCallback":
        return _early_stop_callback_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_degenerate(
    H: "ArrayLike",
    M: int = None,
    certificate: bool = False,
    verbosity: int = 0,
) -> bool:
    r"""Check whether the cone $\{x : Hx\geq 0\}$ is degenerate.

    The cone is considered degenerate if some non-zero integer $x$ in the
    cone saturates (i.e. satisfies $h\cdot x=0$) at least $d+1$ of the
    hyperplanes $h$ simultaneously, where $d$ is the ambient dimension. The
    search is done with OR-Tools CP-SAT, and it stops as soon as such an $x$
    is found. See [`Cone.is_degenerate`][cytools.cone.Cone.is_degenerate]
    for more details.

    Args:
        H (ArrayLike): The inward-pointing hyperplane normals defining the
            cone, one per row.
        M (int | None): The bound on the absolute value of the coordinates of
            $x$ in the search. If not specified, the largest bound supported by
            CP-SAT is used.
        certificate (bool): Whether to also return the point $x$ that was
            found and the hyperplanes that the solver claims it saturates.
        verbosity (int): The verbosity level.

    Returns:
        bool | tuple | None: Whether the cone is degenerate, or `None` if the
            solver failed. If `certificate` is true, a tuple
            `(degenerate, (x, z))` is returned instead, where `x` is the point
            that was found and `z` is an array of 0s and 1s indicating which
            hyperplanes it saturates (both can be `None`).
    """
    H = np.asarray(H)

    # accommodate trivial hyperplanes
    if 0 in H.shape:
        if certificate:
            return False, (None, None)
        else:
            return False

    # create the solver/model
    solver = cp_model.CpSolver()
    model = cp_model.CpModel()

    if verbosity >= 2:
        solver.parameters.log_search_progress = True
        solver.parameters.num_search_workers = 1

    # define variables
    # ----------------
    # variable bounds
    if M is None:
        lower = cp_model.INT32_MIN
        upper = cp_model.INT32_MAX
    else:
        lower, upper = -int(M), int(M)

    # actual variables
    x = [model.NewIntVar(lower, upper, f"x_{j}") for j in range(H.shape[1])]
    xnz = [model.NewBoolVar(f"nz_{j}") for j in range(H.shape[1])]

    satd = [model.NewBoolVar(f"z_{i}") for i in range(H.shape[0])]

    # define constraints
    # ------------------
    # count the nonzeros
    for j in range(H.shape[1]):
        model.Add(x[j] != 0).OnlyEnforceIf(xnz[j])
        model.Add(x[j] == 0).OnlyEnforceIf(xnz[j].Not())

    # enforce nonzeros
    model.Add(sum(xnz) >= 1)

    # enforce cone constraints
    for i, v in enumerate(H):
        dist = sum(_x * _v for _x, _v in zip(x, v))

        # enforce that dists are non-negative (cone hyperplane constraint)
        model.Add(dist >= 0)

        # saturate the hyperplane if the indicator variable is True.
        ct = model.Add(dist == 0)
        ct.OnlyEnforceIf(satd[i])

    # define objective
    # ----------------
    model.Maximize(sum(satd))

    # implement early-stop callback
    # -----------------------------
    cb = _early_stop_callback_cls()(H.shape[1] + 1, solver)

    # solve and parse solution
    status = solver.Solve(model, cb)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        x = np.array([solver.Value(_x) for _x in x])
        z = np.array([solver.Value(z) for z in satd])
        degen = sum(z) >= H.shape[1] + (1 - 1e-4)

        if verbosity >= 1:
            print(f"Found x={x} saturating the indicated hyperplanes={z}...")
    elif status == cp_model.INFEASIBLE:
        if verbosity >= 1:
            warnings.warn("Solver returned status INFEASIBLE.")
        degen, z, x = False, None, None
    else:
        status_list = [
            "OPTIMAL",
            "FEASIBLE",
            "INFEASIBLE",
            "UNBOUNDED",
            "ABNORMAL",
            "MODEL_INVALID",
            "NOT_SOLVED",
        ]
        warnings.warn(f"Solver returned status {status_list[status]}.")
        degen, z, x = None, None, None

    if certificate:
        return degen, (x, z)
    else:
        return degen
