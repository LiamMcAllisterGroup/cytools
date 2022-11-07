# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CYTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CYTools.  If not, see <https://www.gnu.org/licenses/>.

"""
This module contains tools designed to perform cone computations.
"""

# Standard imports
from multiprocessing import Process, Queue, cpu_count
from ast import literal_eval
import subprocess
import warnings
import random
import string
import os
# Third party imports
from flint import fmpz_mat, fmpz, fmpq
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from scipy.optimize import nnls
from scipy import sparse
import numpy as np
import qpsolvers
import ppl
# CYTools imports
from cytools.utils import gcd_list, array_fmpz_to_int, array_fmpq_to_float
from cytools import config



class Cone:
    """
    This class handles all computations relating to rational polyhedral cones,
    such cone duality and extremal ray computations. It is mainly used for the
    study of Kähler and Mori cones.

    :::important warning
    This class is primarily tailored to pointed (i.e. strongly convex) cones.
    There are a few computations, such as finding extremal rays, that may
    produce some unexpected results when working with non-pointed cones.
    Additionally, cones that are not pointed, and whose dual is also not
    pointed, are not supported since they are uncommon and difficult to deal
    with.
    :::

    ## Constructor

    ### `cytools.cone.Cone`

    **Description:**
    Constructs a `Cone` object. This is handled by the hidden
    [`__init__`](#__init__) function.

    **Arguments:**
    - `rays` *(array_like, optional)*: A list of rays that generates
      the cone. If it is not specified then the hyperplane normals must be
      specified.
    - `hyperplanes` *(array_like, optional)*: A list of inward-pointing
      hyperplane normals that define the cone. If it is not specified then the
      generating rays must be specified.
    - `check` *(bool, optional, default=True)*: Whether to check the input.
      Recommended if constructing a cone directly.

    :::note
    Exactly one of `rays` or `hyperplanes` must be specified. Otherwise
    an exception is raised.
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

    def __init__(self, rays=None, hyperplanes=None, check=True):
        """
        **Description:**
        Initializes a `Cone` object.

        **Arguments:**
        - `rays` *(array_like, optional)*: A list of rays that generates
          the cone. If it is not specified then the hyperplane normals must be
          specified.
        - `hyperplanes` *(array_like, optional)*: A list of inward-pointing
          hyperplane normals that define the cone. If it is not specified then
          the generating rays must be specified.
        - `check` *(bool, optional, default=True)*: Whether to check the
          input. Recommended if constructing a cone directly.

        :::note
        Exactly one of `rays` or `hyperplanes` must be specified.
        Otherwise an exception is raised.
        :::

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new
        `Cone` object. We construct a cone in two different ways. First
        from a list of rays then from a list of hyperplane normals. We verify
        that the two inputs result in the same cone.
        ```python {2,3}
        from cytools import Cone
        c1 = Cone([[0,1],[1,1]]) # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Create a cone using hyperplane normals.
        c1 == c2 # We verify that the two cones are the same.
        # True
        ```
        """
        if not ((rays is None) ^ (hyperplanes is None)):
            raise ValueError("Exactly one of \"rays\" and \"hyperplanes\" "
                            "must be specified.")
        if rays is not None:
            tmp_rays = np.array(rays)
            if any(not i for i in tmp_rays.shape):
                raise NotImplementedError("Zero-dimensional cones are not supported.")
            if len(tmp_rays.shape) != 2:
                raise ValueError("Input must be a matrix.")
            t = type(tmp_rays[0,0])
            if t == fmpz:
                if not config._exp_features_enabled:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                rays = array_fmpz_to_int(tmp_rays)
            elif t == fmpq:
                if not config._exp_features_enabled:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                rays = array_fmpq_to_float(tmp_rays)
            elif t not in (np.int64, np.float64):
                raise NotImplementedError("Unsupported data type.")
            if check or t in (fmpz, np.float64):
                tmp_rays = []
                if len(rays) < 1:
                    raise ValueError("At least one rays is required.")
                for r in rays:
                    g = gcd_list(r)
                    if g == 0:
                        continue
                    if g < 1e-5:
                        warnings.warn("Extremely small gcd found. "
                                      "Computations may be incorrect.")
                    tmp_rays.append([int(round(c/g)) for c in r])
                self._rays = np.array(tmp_rays, dtype=int)
            else:
                self._rays = np.array(rays, dtype=int)
            self._hyperplanes = None
            self._rays_were_input = True
        if hyperplanes is not None:
            tmp_hp = np.array(hyperplanes)
            if any(not i for i in tmp_hp.shape):
                raise NotImplementedError("Cones that cover the entire space are not supported.")
            if len(tmp_hp.shape) != 2:
                raise ValueError("Input must be a matrix.")
            t = type(tmp_hp[0,0])
            if t == fmpz:
                if not config._exp_features_enabled:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                hyperplanes = array_fmpz_to_int(tmp_hp)
            elif t == fmpq:
                if not config._exp_features_enabled:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                hyperplanes = array_fmpq_to_float(tmp_hp)
            elif t not in (np.int64, np.float64):
                raise NotImplementedError("Unsupported data type.")
            if check or t in (fmpz, np.float64):
                tmp_hp = []
                if len(hyperplanes) < 1:
                    raise ValueError("At least one hyperplane is required.")
                for r in hyperplanes:
                    g = gcd_list(r)
                    if g == 0:
                        continue
                    if g < 1e-5:
                        warnings.warn("Extremely small gcd found. "
                                      "Computations may be incorrect.")
                    tmp_hp.append([int(round(c/g)) for c in r])
                self._hyperplanes = np.array(tmp_hp, dtype=int)
            else:
                self._hyperplanes = np.array(hyperplanes, dtype=int)
            self._rays = None
            self._rays_were_input = False
        self._ambient_dim = (self._rays.shape[1] if rays is not None else
                             self._hyperplanes.shape[1])
        if rays is not None:
            self._dim = np.linalg.matrix_rank(self._rays)
        else:
            self._dim = None
        # Initialize remaining hidden attributes
        self._hash = None
        self._dual = None
        self._ext_rays = None
        self._is_solid = None
        self._is_pointed = None
        self._is_simplicial = None
        self._is_smooth = None
        self._hilbert_basis = None

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We construct a cone, compute its extremal rays, clear the cache
        and then compute them again.
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
        self._ext_rays = None
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
            return (f"A {self._dim}-dimensional rational polyhedral cone in "
                    f"RR^{self._ambient_dim} generated by {len(self._rays)} "
                    f"rays")
        return (f"A rational polyhedral cone in RR^{self._ambient_dim} "
                f"defined by {len(self._hyperplanes)} hyperplanes")

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
        if (self._rays is not None and other._rays is not None and
            sorted(self._rays.tolist()) == sorted(other._rays.tolist())):
            return True
        if (self._hyperplanes is not None and
                other._hyperplanes is not None and
                sorted(self._hyperplanes.tolist()) ==
                    sorted(other._hyperplanes.tolist())):
            return True
        if self.is_pointed() ^ other.is_pointed():
            return False
        if self.is_pointed() and other.is_pointed():
            return (sorted(self.extremal_rays().tolist())
                    == sorted(other.extremal_rays().tolist()))
        if self.dual().is_pointed() ^ other.dual().is_pointed():
            return False
        if self.dual().is_pointed() and other.dual().is_pointed():
            return (sorted(self.dual().extremal_rays().tolist())
                    == sorted(other.dual().extremal_rays().tolist()))
        warnings.warn("The comparison of cones that are not pointed, and "
                      "whose duals are also not pointed, is not supported.")
        return NotImplemented

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
            self._hash = hash(tuple(sorted(tuple(v)
                                           for v in self.extremal_rays())))
            return self._hash
        if self.dual().is_pointed():
            # Note: The minus sign is important because otherwise the dual
            # cone would have the same hash.
            self._hash = -hash(tuple(sorted(tuple(v)
                                        for v in self.dual().extremal_rays())))
            return self._hash
        warnings.warn("Cones that are not pointed and whose duals are also "
                      "not pointed are assigned a hash value of 0.")
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
    # Aliases
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
            self._dim = np.linalg.matrix_rank(self._rays)
            return self._dim
        self._dim = np.linalg.matrix_rank(self.rays())
        return self._dim
    # Aliases
    dim = dimension

    def rays(self):
        """
        **Description:**
        Returns the (not necessarily extremal) rays that generate the cone.

        **Arguments:**
        None.

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
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        if self._rays is not None:
            return np.array(self._rays)
        if (self._ambient_dim >= 12
                and len(self._hyperplanes) != self._ambient_dim):
            warnings.warn("This operation might take a while for d > ~12 "
                          "and is likely impossible for d > ~18.")
        cs = ppl.Constraint_System()
        vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
        for h in self.dual().extremal_rays():
            cs.insert(
                sum(h[i]*vrs[i] for i in range(self._ambient_dim)) >= 0 )
        cone = ppl.C_Polyhedron(cs)
        rays = []
        for gen in cone.minimized_generators():
            if gen.is_ray():
                rays.append(tuple(int(c) for c in gen.coefficients()))
            elif gen.is_line():
                rays.append(tuple(int(c) for c in gen.coefficients()))
                rays.append(tuple(-int(c) for c in gen.coefficients()))
        self._rays = np.array(rays, dtype=int)
        self._dim = np.linalg.matrix_rank(self._rays)
        return np.array(self._rays)

    def hyperplanes(self):
        """
        **Description:**
        Returns the inward-pointing normals to the hyperplanes that define the
        cone.

        **Arguments:**
        None.

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
            warnings.warn("This operation might take a while for d > ~12 "
                          "and is likely impossible for d > ~18.")
        gs = ppl.Generator_System()
        vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
        gs.insert(ppl.point(0))
        for r in self.extremal_rays():
            gs.insert(ppl.ray(sum(r[i]*vrs[i]
                                  for i in range(self._ambient_dim))))
        cone = ppl.C_Polyhedron(gs)
        hyperplanes = []
        for cstr in cone.minimized_constraints():
            hyperplanes.append(tuple(int(c) for c in cstr.coefficients()))
            if cstr.is_equality():
                hyperplanes.append(tuple(-int(c) for c in cstr.coefficients()))
        self._hyperplanes = np.array(hyperplanes, dtype=int)
        return np.array(self._hyperplanes)

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
    # Aliases
    dual = dual_cone

    def extremal_rays(self, tol=1e-4, verbose=False):
        """
        **Description:**
        Returns the extremal rays of the cone.

        :::note
        By default, this function will use as many CPU threads as there are
        available. To fix the number of threads, you can set the `n_threads`
        variable in the `config` submodule.
        :::

        **Arguments:**
        - `tol` *(float, optional, default=1e-4)*: Specifies the tolerance
          for deciding whether a ray is extremal or not.
        - verbose *(bool, optional, default=False)*: When set to True it show
          the progress while finding the extremal rays.

        **Returns:**
        *(numpy.ndarray)* The list of extremal rays of the cone.

        **Example:**
        We construct a cone and find its extremal_rays.
        ```python {2}
        c = Cone([[0,1],[1,1],[1,0]])
        c.extremal_rays()
        # array([[0, 1],
        #        [1, 0]])
        ```
        """
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        # It is important to delete duplicates
        rays = np.array(list({tuple(r) for r in self.rays()}))
        n_threads = config.n_threads
        if n_threads is None:
            if rays.shape[0] < 32 or not self.is_pointed():
                n_threads = 1
            else:
                n_threads = cpu_count()
        elif n_threads > 1 and not self.is_pointed():
            warnings.warn("When finding the extremal rays of a non-pointed "
                          "cone in parallel there can be conflicts that end up "
                          "producing erroneous results. It is highly recommended to "
                          "use a single thread.")
        current_rays = set(range(rays.shape[0]))
        ext_rays = set()
        error_rays = set()
        rechecking_rays = False
        failed_after_rechecking = False
        while True:
            checking = []
            for i in current_rays:
                if (i not in ext_rays
                        and (i not in error_rays or rechecking_rays)):
                    checking.append(i)
                if len(checking) >= n_threads:
                    break
            if len(checking) == 0:
                if rechecking_rays:
                    break
                rechecking_rays = True
            As = [np.array([rays[j] for j in current_rays if j!=k],dtype=int).T
                    for k in checking]
            bs = [rays[k] for k in checking]
            q = Queue()
            procs = [Process(target=is_extremal,
                     args=(As[k],bs[k],k,q,tol)) for k in range(len(checking))]
            for t in procs:
                t.start()
            for t in procs:
                t.join()
            results = [q.get() for j in range(len(checking))]
            for res in results:
                if res[1] is None:
                    error_rays.add(checking[res[0]])
                    if rechecking_rays:
                        failed_after_rechecking = True
                        ext_rays.add(checking[res[0]])
                    elif verbose:
                        print("Minimization failed. "
                              "Ray will be rechecked later...")
                elif not res[1]:
                    current_rays.remove(checking[res[0]])
                else:
                    ext_rays.add(checking[res[0]])
                if rechecking_rays:
                    error_rays.remove(checking[res[0]])
            if verbose:
                print(f"Eliminated {sum(not r[1] for r in results)}. "
                      f"Current number of rays: {len(current_rays)}")
        if failed_after_rechecking:
            warnings.warn("Minimization failed after multiple attempts. "
                          "Some rays may not be extremal.")
        self._ext_rays = rays[list(ext_rays),:]
        return self._ext_rays

    def tip_of_stretched_cone(self, c, backend=None, check=True,
                              constraint_error_tol=5e-2):
        """
        **Description:**
        Finds the tip of the stretched cone. The stretched cone is defined
        as the convex polyhedral region inside the cone that is at least a
        distance `c` from any of its defining hyperplanes. Its tip is defined
        as the point in this region with the smallest norm.

        :::note
        This is a problem that requires quadratic programming since the norm
        of a vector is being minimized. For dimensions up to around 50, this
        can easily be done with open-source solvers like OSQP or CVXOPT,
        however for higher dimensions this becomes a difficult task that only
        the Mosek optimizer is able to handle. However, Mosek is closed-source
        and requires a license. For this reason we preferentially use ORTools,
        which is open-source, to solve a linear problem and find a good
        approximation of the tip. Nevertheless, if Mosek is activated then
        it uses Mosek as it is faster and more accurate.
        :::

        **Arguments:**
        - `c` *(float)*: A real positive number specifying the stretching
          of the cone (i.e. the minimum distance to the defining hyperplanes).
        - `backend` *(str, optional, default=None)*: String that
          specifies the optimizer to use. Options are "mosek", "osqp",
          "cvxopt", and "glop". If it is not specified then for $d<50$
          it uses "osqp" by default. For $d\geq50$ it uses "mosek" if it is
          activated, or "glop" otherwise.
        - `check` *(bool, optional, default=True)*: Flag that specifies
          whether to check if the output of the optimizer is consistent and
          satisfies `constraint_error_tol`.
        - `constraint_error_tol` *(float, optional, default=1e-2)*: Error
          tolerance for the linear constraints.

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
        backends = (None, "mosek", "osqp", "cvxopt", "glop")
        if backend not in backends:
            raise ValueError("Invalid backend. "
                             f"The options are: {backends}.")
        if backend is None:
            if self.ambient_dim() < 25:
                backend = "osqp"
            else:
                backend = ("mosek" if config.mosek_is_activated() and self.ambient_dim() >= 25
                           else "glop")
        if backend == "mosek" and not config.mosek_is_activated():
            raise Exception("Mosek is not activated. See the advanced usage page on our website to see how to activate it.")
        if backend == "glop":
            solution = self.find_interior_point(c, backend="glop")
            G = -1*sparse.csc_matrix(self.hyperplanes(), dtype=float)
        else:
            hp = self.hyperplanes()
            # The problem is defined as:
            # Minimize (1/2) x.P.x + q.x
            # Subject to G.x <= h
            P = 2*sparse.identity(hp.shape[1], dtype=float, format="csc")
            q = np.zeros(hp.shape[1], dtype=float)
            h = np.full(hp.shape[0], -c, dtype=float)
            G = -1*sparse.csc_matrix(hp, dtype=float)
            settings_dict = ({"max_iter":100000, "scaling":50} if backend=="osqp"
                                else dict())
            solution = qpsolvers.solve_qp(P,q,G,h,solver=backend, **settings_dict)
        if solution is None:
            return
        if check:
            res = max(G.dot(solution)) + c
            if res > constraint_error_tol:
                warnings.warn(f"The solution that was found is invalid: {res} > {constraint_error_tol}")
                return
        return solution

    def find_grading_vector(self, backend=None):
        """
        **Description:**
        Finds a grading vector for the cone, i.e. a vector $\mathbf{v}$ such
        that any non-zero point in the cone $\mathbf{p}$ has a positive dot
        product $\mathbf{v}\cdot\mathbf{p}>0$. Thus, the grading vector must
        be strictly interior to the dual cone, so it is only defined for
        pointed cones. This function returns an integer grading vector.

        **Arguments:**
        - `backend` *(str, optional, default=None)*: String that
          specifies the optimizer to use. The options are the same as for the
          [`find_interior_point`](#find_interior_point) function.

        **Returns:**
        *(numpy.ndarray)* A grading vector. If it could not be found then
        None is returned.

        **Example:**
        We construct a cone and find a grading vector.
        ```python {2}
        c = Cone([[3,2],[5,3]])
        c.find_grading_vector()
        # array([-1,  2])
        ```
        """
        if not self.is_pointed():
            raise Exception("Grading vectors are only defined for pointed cones.")
        return self.dual().find_interior_point(backend=backend, integral=True)

    def find_interior_point(self, c=1, integral=False, backend=None):
        """
        **Description:**
        Finds a point in the strict interior of the cone. If no point is found
        then None is returned.

        **Arguments:**
        - `c` *(float, optional, default=1)*: A real positive number specifying the stretching
          of the cone (i.e. the minimum distance to the defining hyperplanes).
        - `backend` *(str, optional, default=None)*: String that
          specifies the optimizer to use. Options are "glop", "scip", "cpsat",
          "mosek", "osqp", and "cvxopt". If it is not specified then for $d<50$
          it uses "glop" by if `integral` is False or "scip" if it is True. For
          $d\geq50$ it uses "mosek" if it is activated, or "glop" otherwise.
        - `integral` *(bool, optional, default=False)*: A flag that specifies
          whether the point should have integral coordinates.

        **Returns:**
        *(numpy.ndarray)* A point in the strict interior of the cone. If no
        point is found then None is returned.

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
            raise ValueError(f"Backend must be one of {backends}.")
        # If the rays are already computed then this is a simple task
        if self._rays is not None and backend is None:
            if np.linalg.matrix_rank(self._rays) != self._ambient_dim:
                return None
            point = np.sum(self._rays, axis=0)
            point //= gcd_list(point)
            if not integral:
                point = point/len(self._rays)
            return point
        # Otherwise we need to do a harder computation to find an interior point
        if integral and backend is None and self.ambient_dim() < 25:
            backend = "scip"
        if backend is None:
            backend = ("mosek" if config.mosek_is_activated() and self.ambient_dim() >= 25 else "glop")
        if backend in ("glop", "scip"):
            hp = self.hyperplanes().tolist()
            obj_vec = np.sum(hp, axis=0)/len(hp)
            solver = pywraplp.Solver.CreateSolver(backend.upper())
            obj = solver.Objective()
            var = []
            for i in range(self._ambient_dim):
                var.append((solver.NumVar if backend=="glop" else solver.IntVar)(-solver.infinity(), solver.infinity(), f"x_{i}"))
                obj.SetCoefficient(var[-1], obj_vec[i])
            obj.SetMinimization()
            cons_list = []
            for v in hp:
                cons_list.append(solver.Constraint(c, solver.infinity()))
                for j in range(self._ambient_dim):
                    cons_list[-1].SetCoefficient(var[j], v[j])
            status = solver.Solve()
            if status in (solver.FEASIBLE, solver.OPTIMAL):
                solution = np.array([x.solution_value() for x in var])
            elif status == solver.INFEASIBLE:
                return None
            else:
                warnings.warn(f"Solver returned status {status}.")
        elif backend == "cpsat":
            hp = self.hyperplanes().tolist()
            obj_vec = np.sum(hp, axis=0)
            obj_vec //= gcd_list(obj_vec)
            model = cp_model.CpModel()
            obj = 0
            var = []
            for i in range(self._ambient_dim):
                var.append(model.NewIntVar(cp_model.INT32_MIN, cp_model.INT32_MAX, f"x_{i}"))
                obj += var[-1]*obj_vec[i]
            model.Minimize(obj)
            for v in hp:
                model.Add(sum(ii*var[i] for i,ii in enumerate(v)) >= c)
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                solution = np.array([solver.Value(x) for x in var])
            elif status == cp_model.INFEASIBLE:
                return None
            else:
                warnings.warn(f"Solver returned status {status}.")
        else:
            solution = self.tip_of_stretched_cone(c, backend=backend)
            if solution is None:
                return None
        # Make sure that the solution is valid
        hp = self.hyperplanes()
        if any(v.dot(solution) <= 0 for v in hp):
            warnings.warn("The solution that was found is invalid.")
            return None
        # Finally, round to an integer if necessary
        if integral:
            n_tries = 1000
            for i in range(1,n_tries):
                int_sol = np.array([int(round(x)) for x in i*solution])
                if all(int_sol.dot(v) > 0 for v in hp):
                    break
                if i == n_tries-1:
                    return None
            solution = int_sol
        return solution

    def find_lattice_points(self, min_points=None, max_deg=None,
                            grading_vector=None, max_coord=1000,
                            filter_function=None, process_function=None):
        """
        **Description:**
        Finds lattice points in the cone. The points are found in the region
        bounded by the cone, and by a cutoff surface given by the grading
        vector. Note that this requires the cone to be pointed. The minimum
        number of points to find can be specified, or if working with a
        preferred grading vector it is possible to specify the maximum degree.

        **Arguments:**
        - `min_point` *(int, optional)*: Specifies the minimum number of
          points to find. The degree will be increased until this minimum
          number is achieved.
        - `max_deg` *(int, optional)*: The maximum degree of the points to
          find. This is useful when working with a preferred grading.
        - `grading_vector` *(array_like, optional)*: The grading vector that
          will be used. If it is not specified then it is computed.
        - `max_coord` *(int, optional, default=1000)*: The maximum magnitude
          of the coordinates of the points.
        - `filter_function` *(function, optional)*: A function to use as a
          filter of the points that will be kept. It should return a boolean
          indicating whether to keep the point. Note that `min_points` does
          not take the filtering into account.
        - `process_function` *(function, optional)*: A function to process the
          points as the are found. This is useful to avoid first constructing
          a large list of points and then processing it.

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
        if max_deg is None and min_points is None:
            raise Exception("Either the maximum degree or the minimum number of points must be specified.")
        if not self.is_pointed():
            raise Exception("Only pointed cones are currently supported.")
        if process_function is not None and filter_function is not None:
            raise Exception("Only one of filter_function or process_function can be specified.")
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
        # We now define various versions of the on_solution_callback method for the different scenarios
        # The reason for having multiple functions instead of having various if statements in a single
        # function is that, since it will be run many times, it is very inefficient to keep checking
        # the conditions even though they will never change.
        # This first method is for when we want to check that it is a pointed cone with a good grading vector
        class MoreThanOneSolution(Exception):
            pass
        def on_solution_callback_single_point(self):
            self._n_sol += 1
            if self._n_sol > 1:
                raise MoreThanOneSolution
        # This one is the standard one that will be used
        def on_solution_callback_default(self):
            self._n_sol += 1
            self._solutions.add(tuple(self.Value(v) for v in self._variables))
        # This one is the one that will be used when a custom filtering is specified
        def on_solution_callback_filter(self):
            self._n_sol += 1
            point = tuple(self.Value(v) for v in self._variables)
            if self._filter_function(point):
                self._solutions.add(point)
        def on_solution_callback_process(self):
            self._n_sol += 1
            process_function(tuple(self.Value(v) for v in self._variables))
        # If it is a pointed cone we first check that we have a good grading vector
        if self.is_pointed():
            model = cp_model.CpModel()
            var = [model.NewIntVar(-max_coord, max_coord, f"x_{i}") for i in range(hp.shape[1])]
            for v in hp:
                model.Add(sum(ii*var[i] for i,ii in enumerate(v)) >= 0)
            model.Add(sum(ii*var[i] for i,ii in enumerate(grading_vector)) <= 0)
            solver = cp_model.CpSolver()
            SolutionStorage.on_solution_callback = on_solution_callback_single_point
            solution_storage = SolutionStorage(var, filter_function, process_function)
            try:
                status = solver.SearchForAllSolutions(model, solution_storage)
            except MoreThanOneSolution:
                raise Exception("More than one solution was found. The grading vector must be wrong.")
        # Now we construct the solution storage that will hold the points we find
        if filter_function is not None:
            SolutionStorage.on_solution_callback = on_solution_callback_filter
        elif process_function is not None:
            SolutionStorage.on_solution_callback = on_solution_callback_process
        else:
            SolutionStorage.on_solution_callback = on_solution_callback_default
        solution_storage = SolutionStorage(var, filter_function, process_function)
        # If the maximum degree is specified, we use it as a constraint
        if max_deg is not None:
            model = cp_model.CpModel()
            var = [model.NewIntVar(-max_coord, max_coord, f"x_{i}")
                        for i in range(hp.shape[1])]
            for v in hp:
                model.Add(sum(ii*var[i] for i,ii in enumerate(v)) >= 0)
            model.Add(sum(ii*var[i] for i,ii in enumerate(grading_vector)) <= max_deg)
            solver = cp_model.CpSolver()
            status = solver.SearchForAllSolutions(model, solution_storage)
            if status != cp_model.OPTIMAL:
                raise Exception(f"There was a problem finding the points. Status code: {status}")
        else: # Else, we're going to add points until the minimum number is reached
            deg = 0
            while True:
                model = cp_model.CpModel()
                var = [model.NewIntVar(-max_coord, max_coord, f"x_{i}")
                            for i in range(hp.shape[1])]
                for v in hp:
                    model.Add(sum(ii*var[i] for i,ii in enumerate(v)) >= 0)
                model.Add(sum(ii*var[i] for i,ii in enumerate(grading_vector)) == deg)
                solver = cp_model.CpSolver()
                status = solver.SearchForAllSolutions(model, solution_storage)
                if status != cp_model.OPTIMAL:
                    raise Exception(f"There was a problem finding the points. Status code: {status}")
                deg += 1
                if solution_storage._n_sol >= min_points:
                    break
        if process_function is not None:
            return
        pts = np.array(list(solution_storage._solutions), dtype=int)
        return pts

    def is_solid(self, backend=None):
        """
        **Description:**
        Returns True if the cone is solid, i.e. if it is full-dimensional.

        :::note
        If the generating rays are known then this can simply be checked by
        computing the dimension of the linear space that they span. However,
        when only the hyperplane inequalities are known this can be a
        difficult problem. When using PPL as the backend, the convex hull is
        explicitly constructed and checked. The other backends try to find a
        point in the strict interior of the cone, which fails if the cone
        is not solid. The latter approach is much faster, but there could be
        extremely narrow cones where the optimization fails and this function
        returns a false negative.
        :::

        **Arguments:**
        - `backend` *(str, optional)*: Specifies which backend to use.
          Available options are "ppl", and any backends available
          for the [`find_interior_point`](#find_interior_point)
          function. If not specified, it uses the default backend of
          the [`find_interior_point`](#find_interior_point) function.

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
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            return np.linalg.matrix_rank(self._rays) == self._ambient_dim
        backends = (None, "ppl", "glop", "scip", "cpsat", "mosek", "osqp", "cvxopt")
        if backend not in backends:
            raise ValueError(f"Backend must be one of {backends}.")
        if backend == "ppl":
            cs = ppl.Constraint_System()
            vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
            for h in self._hyperplanes:
                cs.insert(sum(h[i]*vrs[i] for i in range(self._ambient_dim)) >= 0 )
            cone = ppl.C_Polyhedron(cs)
            self._is_solid = cone.affine_dimension() == self._ambient_dim
            return self._is_solid
        # Otherwise we check this by trying to find an interior point
        interior_point = self.find_interior_point(backend=backend)
        self._is_solid = interior_point is not None
        return self._is_solid
    # Aliases
    is_full_dimensional = is_solid

    def is_pointed(self, backend=None, tol=1e-7):
        """
        **Description:**
        Returns True if the cone is pointed (i.e. strongly convex).

        :::note
        There are two available methods to perform the computation. Using NNLS
        it directly checks if it can find a linear subspace. Alternatively,
        it can check if the dual cone is solid. For extremely wide cones the
        second approach is more reliable, so that is the default one.
        :::

        **Arguments:**
        - `backend` *(str, optional)*: Specifies which backend to use.
          Available options are "nnls", and any backends available for the
          [`is_solid`](#is_solid) function. If not specified, it uses
          the default backend for the [`is_solid`](#is_solid) function.
        - `tol` *(float, optional, default=1e-7)*: The tolerance for
          determining when a linear subspace is found. This is only used for
          the NNLS backend.

        **Returns:**
        *(bool)* The truth value of the cone being pointed.

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
        if backend == "nnls" and self._rays is not None:
            # If the cone is defined in term of hyperplanes we still don't use
            # nnls since we first would have to find the rays.
            A = np.empty((self._rays.shape[1]+1,self._rays.shape[0]),dtype=int)
            A[:-1,:] = self._rays.T
            A[-1,:] = 1
            b = [0]*self._rays.shape[1] + [1]
            self._is_pointed = nnls(A,b)[1] > tol
            return self._is_pointed
        self._is_pointed = self.dual().is_solid(backend=backend)
        return self._is_pointed
    # Aliases
    is_strongly_convex = is_pointed

    def is_simplicial(self):
        """
        **Description:**
        Returns True if the cone is simplicial.

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
        self._is_simplicial = len(self.extremal_rays()) == self.dim()
        return self._is_simplicial

    def is_smooth(self):
        """
        **Description:**
        Returns True if the cone is smooth, i.e. its extremal rays either form
        a basis of the ambient lattice, or they can be extended into one.

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
            self._is_smooth = (abs(abs(np.linalg.det(self.extremal_rays()))-1)
                               < 1e-4)
            return self._is_smooth
        snf = np.array(fmpz_mat(self.extremal_rays().tolist()).snf().tolist(),
                       dtype=int)
        self._is_smooth = (abs(np.prod([snf[i,i] for i in range(len(snf))]))
                            == 1)
        return self._is_smooth

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
        with open(f"/dev/shm/{proj_name}.in", "w+") as f:
            f.write(f"amb_space {rays.shape[1]}\ncone {rays.shape[0]}\n")
            f.write(str(rays.tolist()).replace("],","\n").replace(",","").replace("[","").replace("]","")+"\n")
        normaliz = subprocess.Popen(("normaliz", f"/dev/shm/{proj_name}.in"),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        normaliz_out = normaliz.communicate()
        with open(f"/dev/shm/{proj_name}.out", "r") as f:
            data = f.readlines()
        os.remove(f"/dev/shm/{proj_name}.in")
        os.remove(f"/dev/shm/{proj_name}.out")
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
                    rays.append([literal_eval(c) for c in data[l_n+1+i].split()])
                l_n += n_rays+1
                continue
            if "further Hilbert basis elements" in l:
                n_rays = literal_eval(l.split()[0])
                for i in range(n_rays):
                    rays.append([literal_eval(c) for c in data[l_n+1+i].split()])
                l_n += n_rays+1
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
            return Cone(hyperplanes=self.hyperplanes().tolist()
                            + other.hyperplanes().tolist()
                        )
        hyperplanes = self.hyperplanes().tolist()
        for c in other:
            if not isinstance(c, Cone):
                raise ValueError("Elements of the list must be Cone objects.")
            if c.ambient_dim() != self.ambient_dim():
                raise ValueError("Ambient lattices must have the same"
                                 "dimension.")
            hyperplanes.extend(c.hyperplanes().tolist())
        return Cone(hyperplanes=hyperplanes)


def is_extremal(A, b, i=None, q=None, tol=1e-4):
    """
    **Description:**
    Auxiliary function that is used to find the extremal rays of cones. Returns
    True if the ray is extremal and False otherwise. It has additional
    parameters that are used when parallelizing.

    **Arguments:**
    - `A` *(array_like)*: A matrix where the columns are rays (excluding
      b).
    - `b` *(array_like)*: The vector that will be checked if it can be
      expressed as a positive linear combination of the columns of A.
    - `i` *(int, optional)*: An id number that is used when parallelizing.
    - `q` *(multiprocessing.Queue, optional)*: A queue that is used when
      parallelizing.
    - `tol` *(float, optional, default=1e-4)*: The tolerance for
      determining whether a ray is extremal.

    **Returns:**
    *(bool or None)* The truth value of the ray being extremal. If the process
    fails then it returns nothing.

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
        v = nnls(A,b)
        is_ext = abs(v[1]) > tol
        if q is not None:
            q.put((i, is_ext))
        return is_ext
    except:
        if q is not None:
            q.put((i,None))
        return
