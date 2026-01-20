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

# 'standard' imports
from ast import literal_eval
from collections.abc import Iterable
from copy import deepcopy
import contextlib
from fractions import Fraction
import joblib
from multiprocessing import cpu_count
import os
import random
import string
import subprocess
import warnings

# 3rd party imports
from flint import fmpz_mat, fmpz, fmpq
import itertools
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import ppl
import qpsolvers
from scipy import sparse
from scipy.optimize import linprog, nnls

# CYTools imports
from cytools import config
from cytools import utils


class Cone:
    """
    This class handles all computations relating to rational polyhedral cones,
    such cone duality and extremal ray computations. It is mainly used for the
    study of Kähler and Mori cones.

    :::important warning
    This class is primarily tailored to pointed (i.e. strongly convex) cones.
    There are a few computations, such as finding extremal rays, that may
    produce some unexpected results when working with non-pointed cones.
    :::

    ## Constructor

    ### `cytools.cone.Cone`

    **Description:**
    Constructs a `Cone` object. This is handled by the hidden
    [`__init__`](#__init__) function.

    **Arguments:**
    - `rays`: A list of rays that generates the cone. If it is not specified then the hyperplane normals must be specified.
    - `hyperplanes` *(array_like, optional)*: A list of inward-pointing
        hyperplane normals that define the cone. If it is not specified then the
        generating rays must be specified.
    - `check` *(bool, optional, default=True)*: Whether to check the input.
        Recommended if constructing a cone directly.

    :::note
    Exactly one of `rays` or `hyperplanes` must be specified. Otherwise an
    exception is raised.
    :::

    **Example:**
    We construct a cone in two different ways. First from a list of rays then
    from a list of hyperplane normals. We verify that the two inputs result in
    the same cone.
    ```python {2,3}
    from cytools import Cone
    c1 = Cone([[0,1],[1,1]]) # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
    c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Create a cone using hyperplane normals.
    c1 == c2 # We verify that the two cones are the same.
    # True
    ```
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
        """
        **Description:**
        Initializes a `Cone` object.

        **Arguments:**
        - `rays`: A list of rays that generates the cone. If it is not
            specified then the hyperplane normals must be specified.
        - `hyperplanes`: A list of inward-pointing hyperplane normals that
            define the cone. If it is not specified then the generating rays
            must be specified.
        - `check`: Whether to check the input. Recommended if constructing a
            cone directly.
        - `copy`: Whether to ensure we copy the input rays/hyperplanes.
            ecommended.
        - `ambient_dim`: The ambient dimension of the cone, if not inferrable.

        :::note
        Exactly one of `rays` or `hyperplanes` must be specified. Otherwise, an
        exception is raised.
        :::

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new `Cone` object.
        We construct a cone in two different ways. First from a list of rays
        then from a list of hyperplane normals. We verify that the two inputs
        result in the same cone.
        ```python {2,3}
        from cytools import Cone
        c1 = Cone([[0,1],[1,1]]) # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Create a cone using hyperplane normals.
        c1 == c2 # We verify that the two cones are the same.
        # True
        ```
        """
        # check whether rays or hyperplanes were input
        if not ((rays is None) ^ (hyperplanes is None)):
            raise ValueError(
                'Exactly one of "rays" and "hyperplanes" ' "must be specified."
            )

        # parse empty hyperplanes
        if (rays is None) and (len(hyperplanes) == 0):

            # check if ambient dim is inferrable from hyperplanes
            if (len(hyperplanes.shape)>1) and (hyperplanes.shape[1]!=0):
                # yes inferrable - ensure no conflicts in specification
                if (ambient_dim is not None) and (ambient_dim != hyperplanes.shape[1]):
                    raise ValueError(f"Specified ambient dim = {ambient_dim} doesn't match inferrable shape from hyperplanes = {hyperplanes.shape[1]}...")

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
                rays.append([int(i==j) for j in range(ambient_dim)])
                rays.append([-int(i==j) for j in range(ambient_dim)])


        # minimal work if we don't parse the data
        if not parse_inputs:
            if rays is None:
                data_name = "hyperplane(s)"
                self._rays_were_input = False
                self._rays = None
                data = hyperplanes
            else:
                raise NotImplementedError(
                    "Currently, parse_inputs is required " "if rays are input..."
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
            elif t == np.int8:
                # rest of calculations assume ints are 64-bit? convert...
                data = data.astype(np.int64)
                t = np.int64
            elif t not in (np.int64, np.float64):
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
                        warnings.warn("0 gcd found (row of zeros)... " "Skipping it!")
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
            self._dim = np.linalg.matrix_rank(self._rays)
        else:
            self._hyperplanes = np.asarray(data)
            self._dim = None

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We construct a cone, compute its extremal rays, clear the cache and
        then compute them again.
        ```python {5}
        c = Cone([[1,0],[1,1],[0,1]])
        c.extremal_rays()
        # array([[0, 1],
        #        [1, 0]])
        c.clear_cache() # Clears the cached result
        c.extremal_rays() # The extremal rays recomputed
        # array([[0, 1],
        #        [1, 0]])
        ```
        """
        self._hash = None
        self._dual = None
        self._ext_rays = [None, None]
        self._is_solid = None
        self._is_pointed = None
        self._is_simplicial = None
        self._is_smooth = None
        self._hilbert_basis = None
        if self._rays_were_input:
            self._hyperplanes = None
        else:
            self._rays = None

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the polytope.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the polytope.

        **Example:**
        This function can be used to convert the Cone to a string or to print
        information about the cone.
        ```python {2,3}
        c = Cone([[1,0],[1,1],[0,1]])
        cone_info = str(c) # Converts to string
        print(c) # Prints cone info
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        ```
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
        """
        **Description:**
        Implements comparison of cones with ==.

        :::note
        The comparison of cones that are not pointed, and whose duals are also
        not pointed, is not supported.
        :::

        **Arguments:**
        - `other` *(Cone)*: The other cone that is being compared.

        **Returns:**
        *(bool)* The truth value of the cones being equal.

        **Example:**
        We construct two cones and compare them.
        ```python {3}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]])
        c1 == c2
        # True
        ```
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
        """
        **Description:**
        Implements comparison of cones with !=.

        :::note
        The comparison of cones that are not pointed, and whose duals are also
        not pointed, is not supported.
        :::

        **Arguments:**
        - `other` *(Cone)*: The other cone that is being compared.

        **Returns:**
        *(bool)* The truth value of the cones being different.

        **Example:**
        We construct two cones and compare them.
        ```python {3}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]])
        c1 != c2
        # False
        ```
        """
        if not isinstance(other, Cone):
            return NotImplemented
        return not self == other

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from cones.

        :::note
        Cones that are not pointed, and whose duals are also not pointed, are
        not supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The hash value of the cone.

        **Example:**
        We compute the hash value of a cone. Also, we construct a set and a
        dictionary with a cone, which make use of the hash function.
        ```python {2,3,4}
        c = Cone([[0,1],[1,1]])
        h = hash(c) # Obtain hash value
        d = {c: 1} # Create dictionary with cone keys
        s = {c} # Create a set of cones
        ```
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
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the ambient lattice.

        **Aliases:**
        `ambient_dim`.

        **Example:**
        We construct a cone and find the dimension of the ambient lattice.
        ```python {2}
        c = Cone([[0,1,0],[1,1,0]])
        c.ambient_dimension()
        # 3
        ```
        """
        return self._ambient_dim

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """
        **Description:**
        Returns the dimension of the cone.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the cone.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a cone and find its dimension.
        ```python {2}
        c = Cone([[0,1,0],[1,1,0]])
        c.dimension()
        # 2
        ```
        """
        if self._dim is not None:
            return self._dim

        if self._rays is not None:
            # know the rays... semi simple computation
            self._dim = np.linalg.matrix_rank(self._rays)
        else:
            # don't know the rays... still simple if the cone is solid...
            if self.is_solid():
                self._dim = self.ambient_dim()
            else:
                # yikes need to compute the rays
                self._dim = np.linalg.matrix_rank(self.rays())
        return self._dim

    # aliases
    dim = dimension

    def rays(self, use_extremal_hyperplanes: bool=False, verbosity: int=0):
        """
        **Description:**
        Returns the (not necessarily extremal) rays that generate the cone.

        **Arguments:**
        - `use_extremal_hyperplanes`: Whether to use extremal hyperplanes in
            this computation, or just any hyperplanes.
        - `verbosity`: The verbosity level.

        **Returns:**
        *(numpy.ndarray)* The list of rays that generate the cone.

        **Example:**
        We construct two cones and find their generating rays.
        ```python {3,6}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[0,1],[1,1]])
        c1.rays()
        # array([[0, 1],
        #        [1, 1]])
        c2.rays()
        # array([[ 1,  0],
        #        [-1,  1]])
        ```
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
        self._dim = np.linalg.matrix_rank(self._rays)
        return np.array(self._rays)

    def hyperplanes(self, use_extremal_rays: bool=False, verbosity: int=0):
        """
        **Description:**
        Returns the inward-pointing normals to the hyperplanes that define the
        cone.

        **Arguments:**
        - `use_extremal_rays` :Whether to use extremal rays in this
            computation, or just any rays.
        - `verbosity`: The verbosity level.

        **Returns:**
        *(numpy.ndarray)* The list of inward-pointing normals to the
        hyperplanes that define the cone.

        **Example:**
        We construct two cones and find their hyperplane normals.
        ```python {3,6}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[0,1],[1,1]])
        c1.hyperplanes()
        # array([[ 1,  0],
        #        [-1,  1]])
        c2.hyperplanes()
        # array([[0, 1],
        #        [1, 1]])
        ```
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
        """
        **Description:**
        Checks if a point is in the (strict) interior.

        **Arguments:**
        - `other`: The object to check containment of. Can be a 1D array, which
            is treated as a point. Can be a 2D array, which is treated as a
            list of points. Can be a Cone.
        - `eps`: Check H@pt >= eps.

        **Returns:**
        Whether pt is in the (strict) interior.
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
        """
        **Description:**
        Returns the dual cone.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The dual cone.

        **Aliases:**
        `dual`.

        **Example:**
        We construct a cone and find its dual cone.
        ```python {2,4}
        c = Cone([[0,1],[1,1]])
        c.dual_cone()
        # A rational polyhedral cone in RR^2 defined by 2 hyperplanes normals
        c.dual_cone().rays()
        # array([[ 1,  0],
        #        [-1,  1]])
        ```
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

    def extremal_rays(self,
        tol: float=1e-4,
        minimal: bool=True,
        method: str="lp",
        verbose: bool=False) -> "ArrayLike":
        """
        **Description:**
        Returns the extremal rays of the cone.

        :::note
        By default, this function will use as many CPU threads as there are
        available. To fix the number of threads, you can set the `n_threads`
        variable in the `config` submodule.
        :::

        **Arguments:**
        - `tol`: Specifies the tolerance for deciding whether a ray is extremal
            or not. Only used if method=="nnls".
        - `minimal`: Whether to return a minimal generating set of rays. For
            pointed cones, there is a unique minimal generating set -- the
            extremal rays. For non-pointed cones, one can have a collection of
            extremal rays generating the cone that is not minimal with respect
            to ray count.
        - `method`: If calling `is_extremal`, this sets the method used for
            extremality checking. Can be "lp" or "nnls". Recommendation is "lp".
        - verbose: When set to True it show the progress while finding the
            extremal rays.

        **Returns:**
        The list of extremal rays of the cone.

        **Example:**
        We construct a cone and find its extremal_rays.
        ```python {2}
        c = Cone([[0,1],[1,1],[1,0]])
        c.extremal_rays()
        # array([[0, 1],
        #        [1, 0]])
        ```
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
            self._ext_rays[minimal] = np.vstack([
                self.lineality_space().extremal_rays(),
                self.pointed_space().extremal_rays()
            ])

            return self._ext_rays[minimal]

        # It is important to delete duplicates
        rays = np.array(list({tuple(r) for r in self.rays()}))

        # if only 1 ray, this is trivial
        if rays.shape[0] == 1:
            self._ext_rays[minimal] = rays
            if self._rays is None:
                self._rays = self._ext_rays[minimal]

            return self._ext_rays[minimal]

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
            print(f"Computing extremal rays for a cone with {len(rays)} using {n_threads} threads...")

        while len(to_check):
            # pull off n_threads rays to check
            checking = to_check[:n_threads]
            to_check = to_check[n_threads:]

            # check the selected rays
            results = joblib.Parallel(n_jobs=n_threads)(
                joblib.delayed(is_extremal)(rays, i, ext_rays, method=method, tol=tol)
                for i in checking
            )

            # learn from the results
            for i, extremalQ, err in results:
                if err is None:
                    ext_rays[i] = extremalQ
                else:
                    to_check.append(i)
                    if verbose:
                        print(f"Failed to check whether ray #{i} was extremal")
                        print(f"(Error was: {err})")
                        print( "(Putting it at the end and retrying later...)")

        # save the answer
        self._ext_rays[minimal] = rays[list(ext_rays)]
        if self._rays is None:
            self._rays = self._ext_rays[minimal]

        return self._ext_rays[minimal]

    def extremal_hyperplanes(self,
        tol: float=1e-4,
        minimal=True,
        method="lp",
        verbose: bool=False) -> "ArrayLike":
        """
        **Description:**
        Returns the extremal hyperplanes of the cone.

        **Arguments:**
        - `tol`: Specifies the tolerance for deciding whether a hyperplane is
            extremal or not. Only used if method=="nnls".
        - `minimal`: Whether to return a minimal generating set of hyperplane.
            For duals of pointed cones, there is a unique minimal generating
            set -- the extremal hyperplanes. For non-pointed cones, one can
            have a collection of extremal hyperplanes defining the cone that is
            not minimal with respect to hyperplane count.
        - `method`: If calling `is_extremal`, this sets the method used for
            extremality checking. Can be "lp" or "nnls". Recommendation is "lp".
        - verbose: When set to True it show the progress while finding the
            extremal hyperplanes.

        **Returns:**
        The list of extremal hyperplanes of the cone.
        """
        return self.dual().extremal_rays(
            tol=tol,
            minimal=minimal,
            method=method,
            verbose=verbose
        )

    def facets(self, verbosity: int = 0):
        """
        **Description:**
        Get the facets of the cone.

        This is easy if:
            -) the cone is simplicial OR
            -) the cone is solid and the extremal hyperplanes can be computed.
        Otherwise, the computation uses both rays and hyperplanes... this is
        semi-expensive to compute...

        **Arguments:**
        - `verbosity`: The verbosity level.

        **Returns:**
        The facets of the cone.
        """
        # ray-based computation
        if self.is_simplicial():
            if verbosity >= 1:
                print("Cone is simplicial! Easy computation...")
            R = self.extremal_rays()

            dim = len(R)
            ray_inds = list(range(dim))

            # facets are defined by collections of #(dim-1) rays
            return [Cone(rays=R[list(inds)]) for inds in\
                                        itertools.combinations(ray_inds,dim-1)]

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

        return [Cone(hyperplanes=np.vstack((H, -h)), check=False) for h in\
                                                                can_saturate]

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
        r"""
        **Description:**
        Finds the tip of the stretched cone. The stretched cone is defined as
        the convex polyhedral region inside the cone that is at least a
        distance `c` from any of its defining hyperplanes. Its tip is defined
        as the point in this region with the smallest norm.

        :::note
        This is a problem that requires quadratic programming since the norm of
        a vector is being minimized. For dimensions up to around 50, this can
        easily be done with open-source solvers like OSQP or CVXOPT, however
        for higher dimensions this becomes a difficult task that only the Mosek
        optimizer is able to handle. However, Mosek is closed-source and
        requires a license. For this reason we preferentially use ORTools,
        which is open-source, to solve a linear problem and find a good
        approximation of the tip. Nevertheless, if Mosek is activated then it
        uses Mosek as it is faster and more accurate.
        :::

        **Arguments:**
        - `c` *(float)*: A real positive number specifying the stretching of
            the cone (i.e. the minimum distance to the defining hyperplanes).
        - `backend` *(str, optional, default=None)*: String that specifies the
            optimizer to use. Options are "mosek", "osqp", "cvxopt", and
            "glop". If it is not specified then for $d<50$ it uses "osqp" by
            default. For $d\geq50$ it uses "mosek" if it is activated, or
            "glop" otherwise.
        - `check` *(bool, optional, default=True)*: Flag that specifies whether
            to check if the output of the optimizer is consistent and satisfies
            `constraint_error_tol`.
        - `constraint_error_tol` *(float, optional, default=1e-2)*: Error
            tolerance for the linear constraints.
        - `max_iter` *(int, optional, default=10**6)*: The maximum number of
            iterations allowed for the non-GLOP backends. If this function is
            returning None, then increasing this parameter (maximum
            permissible value: 2**31-1) might resolve the issue. For
            backend=="glop", this does nothing.
        - `show_hints`: Whether to show hints about odd backend behavior.
        - `verbose` *(boolean, optional)*: Whether to print extra diagnostic
            information (True) or not (False).

        **Returns:**
        *(numpy.ndarray)* The vector specifying the location of the tip. If it
            could not be found then None is returned.

        **Example:**
        We construct two cones and find the locations of the tips of the
        stretched cones.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[3,2],[5,3]])
        c1.tip_of_stretched_cone(1)
        # array([1., 1.])
        c2.tip_of_stretched_cone(1)
        # array([8., 5.])
        ```
        """
        # set the backend
        backends = (None, "mosek", "osqp", "cvxopt", "glop")
        if backend not in backends:
            raise ValueError("Invalid backend. " f"The options are: {backends}.")

        if backend is None:
            if self.ambient_dim() < 25:
                backend = "osqp"
            else:
                backend = (
                    "mosek"
                    if config.mosek_is_activated() and self.ambient_dim() >= 25
                    else "glop"
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

        if backend == "glop":
            solution = self.find_interior_point(c, backend="glop", verbose=verbose)
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
                    "polish": True,
                }
                if backend == "osqp"
                else dict()
            )
            solution = qpsolvers.solve_qp(
                P, q, G, h,
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
                if backend != "glop":
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
        r"""
        **Description:**
        Finds a grading vector for the cone, i.e. a vector $\mathbf{v}$ such
        that any non-zero point in the cone $\mathbf{p}$ has a positive dot
        product $\mathbf{v}\cdot\mathbf{p}>0$. Thus, the grading vector must be
        strictly interior to the dual cone, so it is only defined for pointed
        cones. This function returns an integer grading vector.

        **Arguments:**
        - `backend` *(str, optional, default=None)*: String that specifies the
            optimizer to use. The options are the same as for the
            [`find_interior_point`](#find_interior_point) function.

        **Returns:**
        *(numpy.ndarray)* A grading vector. If it could not be found then None
            is returned.

        **Example:**
        We construct a cone and find a grading vector.
        ```python {2}
        c = Cone([[3,2],[5,3]])
        c.find_grading_vector()
        # array([-1,  2])
        ```
        """
        if not self.is_pointed():
            raise Exception("Grading vectors are only defined for pointed " "cones.")
        return self.dual().find_interior_point(backend=backend, integral=True)

    def find_interior_point(
        self,
        c: float=1,
        lower: float=None,
        integral: bool=False,
        backend: str=None,
        check: bool=True,
        show_hints: bool=False,
        verbose: bool=False,
    ) -> "ArrayLike":
        r"""
        **Description:**
        Finds a point in the strict interior of the cone. If no point is found
        then None is returned.

        **Arguments:**
        - `c`: A real positive number specifying the stretching of the cone
            (i.e. the minimum distance to the defining hyperplanes). Only used
            if rays are not known.
        - `lower`: A lower bound on the components of the interior point.
        - `integral`: A flag that specifies whether the point should have
            integral coordinates.
        - `backend`: String that specifies the optimizer to use. Options are
            "glop", "scip", "cpsat", "mosek", "osqp", and "cvxopt". If it is
            not specified then "glop" is used by default. For $d\geq50$ it uses
            "mosek" if it is activated.
        - `check`: Whether to verify that the point is inside the cone.
        - `show_hints`: Whether to show hints about odd backend behavior.
        - `verbose`: Whether to print diagnostic information.

        **Returns:**
        A point in the strict interior of the cone. If no point is found then
        None is returned.

        **Example:**
        We construct a cone and find some interior points.
        ```python {2,4}
        c = Cone([[3,2],[5,3]])
        c.find_interior_point()
        # array([4. , 2.5])
        c.find_interior_point(integral=True)
        # array([8, 5])
        ```
        """
        backends = (None, "glop", "scip", "cpsat", "mosek", "osqp", "cvxopt")
        if backend not in backends:
            raise ValueError("Invalid backend. " f"The options are: {backends}.")

        # If the rays are already computed then this is a simple task
        if (self._rays is not None) and (backend is None) and (lower is None):
            if np.linalg.matrix_rank(self._rays) != self._ambient_dim:
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
                backend = "glop"

        if backend in ("glop", "scip", "cpsat"):
            solution = feasibility(
                hyperplanes=H,
                c=c,
                ambient_dim=self._ambient_dim,
                backend=backend,
                lower_bound=lower,
                verbose=verbose,
            )
        else:
            if not (lower is None):
                raise ValueError(f"Cannot set custom lower bound for backend = {backend}")
            solution = self.tip_of_stretched_cone(
                c, backend=backend, show_hints=show_hints, verbose=verbose
            )
        if solution is None:
            return None

        # function to take dot products
        if isinstance(H, (list, np.ndarray)):
            dot = lambda hp, x: hp.dot(x)
        else:
            dot = lambda hp, x: sum([val * x[ind] for ind, val in hp.items()])

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
        max_coord=1000,
        deg_window=0,
        filter_function=None,
        process_function=None,
        verbose=False,
    ):
        """
        **Description:**
        Finds lattice points in the cone. The points are found in the region
        bounded by the cone, and by a cutoff surface given by the grading
        vector. Note that this requires the cone to be pointed. The minimum
        number of points to find can be specified, or if working with a
        preferred grading vector it is possible to specify the maximum degree.

        **Arguments:**
        - `min_point` *(int, optional)*: Specifies the minimum number of points
            to find. The degree will be increased until this minimum number is
            achieved.
        - `max_deg` *(int, optional)*: The maximum degree of the points to
            find. This is useful when working with a preferred grading.
        - `grading_vector` *(array_like, optional)*: The grading vector that
            will be used. If it is not specified then it is computed.
        - `c` *(numeric or array_like, optional)*: The minimum allowed
            stretching. Can be a single number or a stretching per each
            hyperplane (applied in the order of self.hyperplanes()).
        - `max_coord` *(int, optional, default=1000)*: The maximum magnitude of
            the coordinates of the points.
        - `deg_window` *(int, optional)*: If using min_points, search for
            lattice points with degrees in range [n*(deg_window+1),
            n*(deg_window+1)+deg_window] for 0<=n
        - `filter_function` *(function, optional)*: A function to use as a
            filter of the points that will be kept. It should return a boolean
            indicating whether to keep the point. Note that `min_points` does
            not take the filtering into account.
        - `process_function` *(function, optional)*: A function to process the
            points as they are found. This is useful to avoid first constructing
            a large list of points and then processing it.
        - `verbose` *(boolean, optional)*: Whether to print extra diagnostic
            information (True) or not (False).

        **Returns:**
        *(numpy.ndarray)* The list of points.

        **Example:**
        We construct a cone and find at least 20 lattice points in it.
        ```python {2}
        c = Cone([[3,2],[5,3]])
        pts = c.find_lattice_points(min_points=20)
        print(len(pts)) # We see that it found 21 points
        # 21
        ```
        Let's also give an example where we use a function to apply some
        filtering. This can be something very complicated, but here we just
        pick the points where all coordinates are odd.
        ```python {5}
        def filter_function(pt):
            return all(c%2 for c in pt)

        c = Cone([[3,2],[5,3]])
        pts = c.find_lattice_points(min_points=20, filter_function=filter_function)
        print(len(pts)) # Now we get only 6 points instead of 21
        # 6
        ```
        Finally, let's give an example where we process the data as it comes
        instead of first constructing a list. In this simple example we just
        print each point with odd coordinates, but in general it can be a
        complex algorithm.
        ```python {6}
        def process_function(pt):
            if all(c%2 for c in pt):
                print(f"Processing point {pt}")

        c = Cone([[3,2],[5,3]])
        c.find_lattice_points(min_points=20, process_function=process_function)
        # Processing point (5, 3)
        # Processing point (11, 7)
        # Processing point (15, 9)
        # Processing point (17, 11)
        # Processing point (21, 13)
        # Processing point (25, 15)
        ```
        """
        # initial checks
        if max_deg is None and min_points is None:
            raise Exception(
                "Either the maximum degree or the minimum number of points must be specified."
            )

        if not self.is_pointed():
            raise Exception("Only pointed cones are currently supported.")

        if process_function is not None and filter_function is not None:
            raise Exception(
                "Only one of filter_function or process_function " "can be specified."
            )
        if grading_vector is None:
            grading_vector = self.find_grading_vector()
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

        solution_storage = SolutionStorage(var, filter_function, process_function)

        # define the model
        solver = cp_model.CpSolver()
        model = cp_model.CpModel()

        # define variables
        var = [
            model.NewIntVar(-max_coord, max_coord, f"x_{i}") for i in range(hp.shape[1])
        ]

        # define constraints
        if not isinstance(c, Iterable):
            c = [c]*len(hp)

        for h,cc in zip(hp,c):
            # clear the denominator
            cc_rat = Fraction(cc).limit_denominator()
            denom = cc_rat.denominator
            numer = cc_rat.numerator

            # add the constraint
            model.Add(
                sum(ii * var[i] * denom for i, ii in enumerate(h)) >= numer
            )

        soln_deg = sum(ii * var[i] for i, ii in enumerate(grading_vector))

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
                # define model with windowed degree constraints
                window_model   = deepcopy(model)
                deg_constr_low = window_model.Add(deg <= soln_deg)
                deg_constr_up  = window_model.Add(soln_deg <= deg + deg_window)

                # solve and check status
                status = solver.SearchForAllSolutions(window_model, solution_storage)
                if verbose and status != cp_model.OPTIMAL:
                    print(
                        "There was a problem finding the points b/t degrees "
                        f"{deg} and {deg+deg_window}. "
                        f"Status code: {solver.StatusName(status)}"
                    )

                deg += deg_window + 1

        # parse solutions
        if process_function is not None:
            return
        pts  = np.array(list(solution_storage._solutions), dtype=int)

        # provide uniform sorting of points
        degs = pts@grading_vector

        out = []
        for deg in sorted(set(degs)):
            out.append(sorted(pts[degs==deg].tolist()))

        return np.vstack(out)

    def is_solid(self, backend=None):
        """
        **Description:**
        Returns True if the cone is solid, i.e. if it is full-dimensional.

        :::note
        If the generating rays are known then this can simply be checked by
        computing the dimension of the linear space that they span. However,
        when only the hyperplane inequalities are known this can be a difficult
        problem. When using PPL as the backend, the convex hull is explicitly
        constructed and checked. The other backends try to find a point in the
        strict interior of the cone, which fails if the cone is not solid. The
        latter approach is much faster, but there could be extremely narrow
        cones where the optimization fails and this function returns a false
        negative.
        :::

        **Arguments:**
        - `backend` *(str, optional)*: Specifies which backend to use. Available
            options are "ppl", and any backends available for the
            [`find_interior_point`](#find_interior_point) function. If not
            specified, it uses the default backend of the
            [`find_interior_point`](#find_interior_point) function.

        **Returns:**
        *(bool)* The truth value of the cone being solid.

        **Aliases:**
        `is_full_dimensional`.

        **Example:**
        We construct two cones and check if they are solid.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[1,0,0],[0,1,0]])
        c1.is_solid()
        # True
        c2.is_solid()
        # False
        ```
        """
        # check for cached answer or if we have rays... makes calc is easy
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            return bool(np.linalg.matrix_rank(self._rays) == self._ambient_dim)

        # we just have hyperplanes... a bit harder
        backends = (
            None,
            "ppl",
            "glop",
            "scip",
            "cpsat",
            "mosek",
            "osqp",
            "cvxopt",
        )
        if backend not in backends:
            raise ValueError("Invalid backend. " f"The options are: {backends}.")

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

    def is_pointed(self, backend: str="dual", tol: float=1e-7) -> bool:
        """
        **Description:**
        Returns True if the cone is pointed (i.e. strongly convex). A cone is
        pointed if no x exists such that both x and -x are in the cone.

        If one has hyperplanes, this check is as simple as `not full_rank(H)`
        since, if H is not full rank, then some x has H@x==0. I.e., H@(+x)>=0
        and H@(-x)>=0.

        If one has rays, this check can be done either via
            1) finding some psi such that psi.r > 0 for all rays r
            2) checking if some lmbda!=0 exist such that R.T@lmbda = 0

        
        The backends are, in order of preference,
            1) (backend='dual') check if dual is solid
            2) (backend='null') hyperplane rank
            3) (backend='lp')   rays@lmbda=0 via LP
            4) (backend='nnls') rays@lmbda=0 via nnls

        **Arguments:**
        - `backend`: Specifies which backend to use. Available options are
            "dual", "null", "lp", and "nnls".
        - `tol`: The tolerance for determining when a linear subspace is found.
            This is only used for the NNLS backend.

        **Returns:**
        The truth value of the cone being pointed.

        **Aliases:**
        `is_strongly_convex`.

        **Example:**
        We construct two cones and check if they are pointed.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[1,0],[0,1],[-1,0]])
        c1.is_pointed()
        # True
        c2.is_pointed()
        # False
        ```
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
            R = np.vstack((
                R,
                np.ones((1,R.shape[1]), dtype=int)
            ))
            b = [0] * (R.shape[0]-1) + [1]

            # allow different backends
            if backend.lower() == "nnls":
                self._is_pointed = nnls(A, b)[1] > tol
            elif backend.lower() == "lp":
                res = linprog(
                    c=np.zeros(R.shape[0], dtype=int),  # no objective
                    A_eq=R, b_eq=b,                     # [R; 1] lmbda = [0;1]
                    bounds=[(0, None)],                 # lmbda >= 0
                    method="highs"
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
        """
        **Description:**
        Returns True if the cone is simplicial.

        N.B.: if c is solid, then c is simplicial <=> c.dual() is simplicial.

        A sometimes-simpler check if c is solid, then, is to check if
        #(extremal hyperplanes) = dim.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the cone being simplicial.

        **Example:**
        We construct two cones and check if they are simplicial.
        ```python {3,5}
        c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
        c2 = Cone([[1,0,0],[0,1,0],[0,0,1],[1,1,-1]])
        c1.is_simplicial()
        # True
        c2.is_simplicial()
        # False
        ```
        """
        if self._is_simplicial is not None:
            return self._is_simplicial

        # split analysis by whether we know rays or not
        if (self._rays is None) and (self.is_solid()):
            self._is_simplicial = len(self.extremal_hyperplanes()) == self.dim()
        else:
            self._is_simplicial = len(self.extremal_rays()) == self.dim()

        return self._is_simplicial

    def is_degenerate(self,
        use_extremal_hyperplanes: bool = True,
        M: int = None,
        certificate: bool=False,
        verbosity: int = 0,):
        """
        **Description:**
        Checks if a cone {x : H@x>=0} is degenerate. I.e., does any x in this
        cone saturate >=d+1 hyperplanes simultaneously, for d the ambient dim?
        If so, the cone is degenerate.

        This is representation-sensitive. Just because the cone is degenerate
        for a certain representation matrix, H, doesn't mean that it's
        degenerate for all representation matrices. Probably best to use H as
        the *extremal hyperplanes*.


        Application: It is more difficult to compute the (extremal or not) rays
        of a degenerate cone.

        **Arguments:**
        - `use_extremal_hyperplanes`: Whether the check the extremal hyperplanes
            for degeneracy. If False, the naive self.hyperplanes() will be used.
        - `M`: The (absolute value of the) bounds on variables considered.
        - `certificate`: Whether to return a certificate x as well as the
            hyperplanes the solver claims it saturates
        - `verbosity`: The verbosity level.

        **Returns:**
        The maximum number of hyperplanes that a single x can saturate
        simultaneously.

        If certificate==True, also return (x,z)
        """
        if use_extremal_hyperplanes:
            H = self.extremal_hyperplanes()
        else:
            H = self.hyperplanes()

        # try a common representative of degeneracy
        xtest = np.ones(self.ambient_dim(), dtype=int)
        dists = H@xtest
        z = dists==0
        if sum(z) >= self.ambient_dim()+1:
            degen, x, z = True, xtest, z
        else:
            # _is_degenerate is an minimal, non-Cone method doing the check
            out = _is_degenerate(
                H=H,
                M=M,
                certificate=certificate,
                verbosity=verbosity
            )

            if certificate:
                degen, (x,z) = out

        # return
        if certificate:
            return degen, (x,z)
        else:
            return degen

    def is_smooth(self):
        """
        **Description:**
        Returns True if the cone is smooth, i.e. its extremal rays either form a
        basis of the ambient lattice, or they can be extended into one.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the cone being smooth.

        **Example:**
        We construct two cones and check if they are smooth.
        ```python {3,5}
        c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
        c2 = Cone([[2,0,1],[0,1,0],[1,0,2]])
        c1.is_smooth()
        # True
        c2.is_smooth()
        # False
        ```
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
        """
        **Description:**
        Returns the lineality space as a formal cone object.

        This Cone object a bit odd since, by definition, the lineality space is
        the largest *linear subspace* in the cone, so it allows coefficients of
        any sign. Regardless, it's convenient to package this as a Cone

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* A cone defining the lineality space.
        """
        H = self.hyperplanes()

        # the lineality space is defined by the x such that H@x==0
        # (the following definition is extremely redundant, so it's only listed
        #  for pedagogical purposes. It's better to define the cone via rays
        #  and then compute the hyperplanes via DDM since there will only be 6
        #  rays, since lineality space should typically be 5D)
        #lin = Cone(hyperplanes = np.vstack([H,-H]))

        # linearly spanning vectors are given by null(H)
        R = utils.integral_nullspace(H).T

        # to map to positively spanning rays, add in the ray r=np.sum(axis=0)
        r = -np.sum(R,axis=0)
        r = r//utils.gcd_list(r)
        R = np.vstack( [R,[r]] )

        lin = Cone(rays=R)

        # save the extremal rays manually
        # (this is split into two saves since _ext_rays stores both the naive
        #  extremal rays [i.e., a subset of _rays] at index 0 and the minimal
        #  extremal rays  at index 1)
        lin._ext_rays[0] = R.copy()
        lin._ext_rays[1] = R.copy()

        return lin

    def pointed_space(self):
        """
        **Description:**
        A cone can be decomposed into its lineality space and its pointed
        component.

        The pointed component is obtained by intersection of the cone with the
        orthogonal complement of the lineality space. I.e., want to impose
        H@x=0 for any x in the lineality space.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The pointed part of the cone.
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
        pointed = Cone(hyperplanes=np.vstack( [H, R, -R] ))
        return pointed

    def hilbert_basis(self):
        """
        **Description:**
        Returns the Hilbert basis of the cone. Normaliz is used for the
        computation.

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The list of vectors forming the Hilbert basis.

        **Example:**
        We compute the Hilbert basis of a two-dimensional cone.
        ```python {2}
        c = Cone([[1,3],[2,1]])
        c.hilbert_basis()
        # array([[1, 1],
        #        [1, 2],
        #        [1, 3],
        #        [2, 1]])
        ```
        """
        if self._hilbert_basis is not None:
            return np.array(self._hilbert_basis)
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
        normaliz_out = normaliz.communicate()
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
            l = data[l_n]
            if "******" in l:
                found_stars = True
                l_n += 1
                continue
            if not found_stars:
                l_n += 1
                continue
            if "lattice points in polytope" in l or "Hilbert basis elements" in l:
                n_rays = literal_eval(l.split()[0])
                for i in range(n_rays):
                    rays.append([literal_eval(c) for c in data[l_n + 1 + i].split()])
                l_n += n_rays + 1
                continue
            if "further Hilbert basis elements" in l:
                n_rays = literal_eval(l.split()[0])
                for i in range(n_rays):
                    rays.append([literal_eval(c) for c in data[l_n + 1 + i].split()])
                l_n += n_rays + 1
                continue
            l_n += 1
            continue
        self._hilbert_basis = np.array(rays)
        return np.array(self._hilbert_basis)

    def intersection(self, other):
        """
        **Description:**
        Computes the intersection with another cone, or with a list of cones.

        **Arguments:**
        - `other` *(Cone or array_like)*: The other cone that is being
            intersected, or a list of cones to intersect with.

        **Returns:**
        *(Cone)* The cone that results from the intersection.

        **Example:**
        We construct two cones and find their intersection.
        ```python {3}
        c1 = Cone([[1,0],[1,2]])
        c2 = Cone([[0,1],[2,1]])
        c3 = c1.intersection(c2)
        c3.rays()
        # array([[2, 1],
        #        [1, 2]])
        ```
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
                raise ValueError("Ambient lattices must have the same" "dimension.")
            hyperplanes.extend(c.hyperplanes().tolist())
        return Cone(hyperplanes=hyperplanes)

def dualize(M, verbosity=0):
    """
    **Description:**
    Converts between hyperplanes and rays of a cone. Output isn't guaranteed to
    be extremal.

    Internal to this function, we treat M as the hyperplanes since that seems
    to be faster.

    **Arguments:**
    - `M`: The matrix defining the cone.
        Can be thought of as the hyperplanes cone = {x: M@x>=0} in which case we
        return the rays cone = {dualize(M).T@lmbda: lmbda>=0}.
        Can also be thought of as the rays cone = {M.T@lmbda: lmbda>=0} in
        which case we return the hypeplanes cone = {x: dualize(M)@x>=0}.
    - `verbosity`: The verbosity level.

    **Returns:**
    The dual description
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
            print(f"ray #{gen_i}...", end='\r')

        if gen.is_ray():
            rays.append(tuple(int(c) for c in gen.coefficients()))
        elif gen.is_line():
            # lineality space... add both signs
            rays.append(tuple(int(c) for c in gen.coefficients()))
            rays.append(tuple(-int(c) for c in gen.coefficients()))

    # return
    return np.array(rays, dtype=int)

def is_extremal(
    R: "ArrayLike",
    i: int,
    extFlags: list[bool] = None,
    method: str = "lp",
    tol: float=1e-4) -> (int, bool, "Exception"):
    """
    **Description:**
    Auxiliary function that is used to find the extremal rays of cones. Returns
    True if the ray is extremal and False otherwise. It has additional
    parameters that are used when parallelizing.

    **Arguments:**
    - `R`: A matrix whose rows are the rays of the cone.
    - `i`: The index of the ray to check for extremality.
    - `extFlags`: A list of flags indicating if the rays r in R are possibly
        extremal. If a ray is known non-extremal, delete it.
    - `method`: The method to check extremality. Can be "lp" or "nnls".
        Reccomendation is "lp".
    - `tol`: The tolerance for determining whether a ray is extremal.

    **Returns:**
    *(bool or None)* The truth value of the ray being extremal.

    **Example:**
    This function is not meant to be directly used by the end user. Instead it
    is used by the [`extremal_rays`](#extremal_rays) function. We construct a
    cone and find its extremal_rays.
    ```python {2}
    c = Cone([[0,1],[1,1],[1,0]])
    c.extremal_rays()
    # array([[0, 1],
    #        [1, 0]])
    ```
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
                A_eq=R.T, b_eq=r,                   # (R\r) lmbda = r
                bounds=[(0, None)],                 # lmbda >= 0
                method="highs"
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
    """
    **Description:**
    Solve a feasibility problem Ax>=c.

    **Arguments:**
    - `hyperplanes`: The constraining hyperplanes, A.
    - `c`: The 'stretching'.
    - `ambient_dim`: The ambient dimension... A.shape[1].
    - `backend`: The backend to use. Options are "glop", "scip", or "cpsat".
    - `verbose`: Whether to print extra diagnostic info.

    **Returns:**
    A feasible point, if it exists. Else, None.
    """
    if isinstance(hyperplanes, (list, np.ndarray)):
        hyperplanes = np.asarray(hyperplanes)
        hp_iter = enumerate
    else:
        hp_iter = lambda hp: hp.items()

    # accommodate trivial hyperplanes
    if len(hyperplanes) == 0:
        return np.ones(ambient_dim)

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
        var = [(var_type)(lower, solver.infinity(), f"x_{i}")
               for i in range(ambient_dim)]

        # define constraints
        cons_list = []
        for v in hyperplanes:
            cons_list.append(solver.Constraint(c, solver.infinity()))
            for ind, val in hp_iter(v):
                cons_list[-1].SetCoefficient(var[ind], float(val))

        # define objective
        obj = solver.Objective()
        obj.SetMinimization()

        obj_vec = hyperplanes.sum(axis=0) / len(hyperplanes)
        for i in range(ambient_dim):
            obj.SetCoefficient(var[i], obj_vec[i])

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
            lower = lower_bound
        for i in range(ambient_dim):
            var.append(
                model.NewIntVar(cp_model.INT32_MIN, cp_model.INT32_MAX, f"x_{i}")
            )

        # define constraints
        for v in hyperplanes:
            model.Add(sum(ii * var[i] for i, ii in enumerate(v)) >= c)

        # define objective
        obj_vec = hyperplanes.sum(axis=0)
        obj_vec //= utils.gcd_list(obj_vec)

        obj = 0
        for i in range(ambient_dim):
            obj += var[i] * obj_vec[i]

        model.Minimize(obj)

        # solve and parse solution
        status = solver.Solve(model)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            solution = np.array([solver.Value(x) for x in var])
        elif status == cp_model.INFEASIBLE:
            return None
        else:
            warnings.warn("Solver returned status " f"{solver.StatusName(status)}.")

    return solution

# cone degeneracy
# ---------------
class EarlyStopCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, threshold, solver):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._threshold = threshold
        self._solver = solver

    def on_solution_callback(self):
        current_value = int(self.ObjectiveValue())
        if current_value >= self._threshold:
            self.StopSearch()

def _is_degenerate(
    H: "ArrayLike",
    M: int = None,
    certificate: bool=False,
    verbosity: int = 0,
) -> bool:
    """
    **Description:**
    Checks if a cone {x : H@x>=0} is degenerate. I.e., does any x in this cone
    saturate >=d+1 hyperplanes simultaneously, for d the ambient dim? If so, the
    cone is degenerate.

    This is representation-sensitive. Just because the cone is degenerate for a
    certain representation matrix, H, doesn't mean that it's degenerate for all
    representation matrices. Probably best to use H as the *extremal
    hyperplanes*.

    Uses CP-SAT from OR-Tools.


    Application: It is more difficult to compute the (extremal or not) rays of
    a degenerate cone.

    **Arguments:**
    - `H`: The inwards-facing hyperplanes defining the cone.
    - `M`: The (absolute value of the) bounds on variables considered.
    - `certificate`: Whether to return a certificate x as well as the
        hyperplanes the solver claims it saturates
    - `verbosity`: The verbosity level.

    **Returns:**
    Whether the cone {x : H@x>=0} is degenerate.

    If certificate==True, also return (x,z)
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
    x   = [
        model.NewIntVar(lower, upper, f"x_{j}")
        for j in range(H.shape[1])]
    xnz = [
        model.NewBoolVar(f"nz_{j}")
        for j in range(H.shape[1])
    ]

    satd = [
        model.NewBoolVar(f"z_{i}")
        for i in range(H.shape[0])
    ]

    # define constraints
    # ------------------
    # count the nonzeros
    for j in range(H.shape[1]):
        model.Add(x[j] != 0).OnlyEnforceIf(xnz[j])
        model.Add(x[j] == 0).OnlyEnforceIf(xnz[j].Not())

    # enforce nonzeros
    model.Add(sum(xnz) >= 1)

    # enforce cone constraints
    for i,v in enumerate(H):
        dist = sum(_x*_v for _x, _v in zip(x,v))

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
    cb = EarlyStopCallback(H.shape[1]+1, solver)

    # solve and parse solution
    status = solver.SolveWithSolutionCallback(model, cb)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        x = np.array([solver.Value(_x) for _x in x])
        z = np.array([solver.Value( z) for  z in satd])
        degen = sum(z)>=H.shape[1]+(1-1e-4)
        
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
