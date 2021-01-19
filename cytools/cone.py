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
# Third party imports
from flint import fmpz_mat, fmpz, fmpq
from ortools.linear_solver import pywraplp
from scipy.optimize import nnls
import numpy as np
import cvxopt
import mosek
import ppl
# CYTools imports
from cytools.utils import gcd_list, array_fmpz_to_int, array_fmpq_to_float
from cytools import config



class Cone:
    """
    This class handles all computations relating to rational polyhedral cones,
    such cone duality and extremal ray computations. It is mainly used for the
    study of Kähler and Mori cones.

    :::note
    This class is primarily taylored to pointed (i.e. strongly convex) cones.
    There are a few computations, such as finding extremal rays, that may
    produce some unexpected results when working with non-pointed cones.
    :::

    ## Constructor

    ### ```cytools.cone.Cone```

    **Description:**
    Constructs a ```Cone``` object. This is handled by the hidden
    [```__init__```](#__init__) function.

    **Arguments:**
    - ```rays``` (list, optional): A list of rays that generates the cone. If
      it is not specified then the hyperplane normals must be specified.
    - ```hyperplanes``` (list, optional): A list of inward-pointing hyperplane
      normals that define the cone. If it is not specified then the generating
      rays must be specified.
    - ```check``` (boolean, optional): Whether to check the input. Recommended
      if constructing a cone directly.

    :::note
    Exactly one of ```rays``` or ```hyperplanes``` must be specified. Otherwise
    an exception is raised.
    :::

    **Example:**
    We construct a cone in two different ways. First from a list of rays then
    from a list of hyperplane normals. We verify that the two inputs result in
    the same cone.
    ```python {2,3}
    from cytools import Cone
    c1 = Cone([[0,1],[1,1]])              # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
    c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Ceate a cone using hyperplane normals.
    c1 == c2                              # We verify that the two cones are the same.
    # Prints: True
    ```
    """

    def __init__(self, rays=None, hyperplanes=None, check=True):
        """
        **Description:**
        Initializes a ```Cone``` object.

        **Arguments:**
        - ```rays``` (list, optional): A list of rays that generates the cone.
          If it is not specified then the hyperplane normals must be specified.
        - ```hyperplanes``` (list, optional): A list of inward-pointing
          hyperplane normals that define the cone. If it is not specified then
          the generating rays must be specified.
        - ```check``` (boolean, optional): Whether to check the input.
          Recommended if constructing a cone directly.

        :::note
        Exactly one of ```rays``` or ```hyperplanes``` must be specified.
        Otherwise an exception is raised.
        :::

        **Returns:**
        Nothing.
        """
        if not ((rays is None) ^ (hyperplanes is None)):
            raise Exception("Exactly one of \"rays\" and \"hyperplanes\" "
                            "must be specified.")
        if rays is not None:
            tmp_rays = np.array(rays)
            if len(tmp_rays.shape) != 2:
                raise Exception("Input must be a matrix.")
            t = type(tmp_rays[0,0])
            if t == fmpz:
                if not config.enable_experimental_features:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                rays = array_fmpz_to_int(tmp_rays)
            elif t == fmpq:
                if not config.enable_experimental_features:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                rays = array_fmpq_to_float(tmp_rays)
            elif t not in (np.int64, np.float64):
                raise Exception("Unsupported data type.")
            if check or t in (fmpz, np.float64):
                tmp_rays = []
                if len(rays) < 1:
                    raise Exception("At least one rays is required.")
                for r in rays:
                    g = gcd_list(r)
                    if g < 1e-5:
                        print("Warning: Extremely small gcd found. "
                              "Computations may be incorrect.")
                    tmp_rays.append([int(round(c/g)) for c in r])
                self._rays = np.array(tmp_rays, dtype=int)
            else:
                self._rays = np.array(rays, dtype=int)
            self._hyperplanes = None
            self._rays_were_input = True
        if hyperplanes is not None:
            tmp_hp = np.array(hyperplanes)
            if len(tmp_hp.shape) != 2:
                raise Exception("Input must be a matrix.")
            t = type(tmp_hp[0,0])
            if t == fmpz:
                if not config.enable_experimental_features:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                hyperplanes = array_fmpz_to_int(tmp_hp)
            elif t == fmpq:
                if not config.enable_experimental_features:
                    print("Arbitrary precision data types only have "
                          "experimental support, so experimental features "
                          "must be enabled in the configuration.")
                hyperplanes = array_fmpq_to_float(tmp_hp)
            elif t not in (np.int64, np.float64):
                raise Exception("Unsupported data type.")
            if check or t in (fmpz, np.float64):
                tmp_hp = []
                if len(hyperplanes) < 1:
                    raise Exception("At least one hyperplane is required.")
                for r in hyperplanes:
                    g = gcd_list(r)
                    if g < 1e-5:
                        print("Warning: Extremely small gcd found. "
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

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.
        """
        self._hash = None
        self._dual = None
        self._ext_rays = None
        self._is_solid = None
        self._is_pointed = None
        self._is_simplicial = None
        self._is_smooth = None
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
        (string) A string describing the polytope.
        """
        if self._rays is not None:
            return (f"A {self._dim}-dimensional rational polyhedral cone in "
                    f"RR^{self._ambient_dim} generated by {len(self._rays)} "
                    f"rays.")
        return (f"A rational polyhedral cone in RR^{self._ambient_dim} "
                f"defined by {len(self._hyperplanes)} hyperplanes normals.")

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of cones with ==.

        :::note
        The comparison of cones that are not pointed, and whose duals are also
        not pointed, is not supported.
        :::

        **Arguments:**
        - ```other``` (Cone): The other cone that is being compared.

        **Returns:**
        (boolean) The truth value of the cones being equal.
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
        print("Warning: The comparison of cones that are not pointed, and "
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
        - ```other``` (Cone): The other cone that is being compared.

        **Returns:**
        (boolean) The truth value of the cones being different.
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
        (integer) The hash value of the cone.
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
        print("Warning: Cones that are not pointed and whose duals are also "
              "not pointed are assigned a hash value of 0.")
        return 0

    def ambient_dim(self):
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        (integer) The dimension of the ambient lattice.
        """
        return self._ambient_dim

    def dim(self):
        """
        **Description:**
        Returns the dimension of the cone.

        **Arguments:**
        None.

        **Returns:**
        (integer) The dimension of the cone.
        """
        if self._dim is not None:
            return self._dim
        if self._rays is not None:
            self._dim = np.linalg.matrix_rank(self._rays)
            return self._dim
        self._dim = np.linalg.matrix_rank(self.rays())
        return self._dim

    def rays(self):
        """
        **Description:**
        Returns the (not necessarily extremal) rays that generate the cone.

        **Arguments:**
        None.

        **Returns:**
        (list) The list of rays that generate the cone.
        """
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        if self._rays is not None:
            return np.array(self._rays)
        if (self._ambient_dim >= 12
                and len(self._hyperplanes) != self._ambient_dim):
            print("Warning: This operation might take a while for d > ~12 "
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
        (list) The list of inward-pointing normals to the hyperplanes that
        define the cone.
        """
        if self._hyperplanes is not None:
            return np.array(self._hyperplanes)
        if self._ambient_dim >= 12 and len(self.rays()) != self._ambient_dim:
            print("Warning: This operation might take a while for d > ~12 "
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

    def dual(self):
        """
        **Description:**
        Returns the dual cone.

        **Arguments:**
        None.

        **Returns:**
        (Cone) The dual cone.
        """
        if self._dual is None:
            if self._rays is not None:
                self._dual = Cone(hyperplanes=self.rays(), check=False)
            else:
                self._dual = Cone(rays=self.hyperplanes(), check=False)
            self._dual._dual = self
        return self._dual

    def extremal_rays(self, tol=1e-4, n_threads=None, verbose=False):
        """
        **Description:**
        Returns the extremal rays of the cone.

        **Arguments:**
        - ```tol``` (float, optional, default=1e-4): Specifies the tolerance
          for deciding whether a ray is extremal or not.
        - ```n_threads``` (integer, optional): Specifies the number of CPU
          threads to be used in the computation. Using multiple threads can
          greatly speed up the computation. If not specified, it is set to the
          number of available CPU threads.
        - verbose (boolean, optional, default=False): When set to True it show
          the progress in finding the extremal rays.

        **Returns:**
        (list) The list of extremal rays of the cone.
        """
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        rays = np.array(list({tuple(r) for r in self.rays()}))
        if n_threads is None:
            n_threads = cpu_count()
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
            print("Warning: Minimization failed after multiple attempts. "
                  "Some rays may not be extremal.")
        self._ext_rays = rays[list(ext_rays),:]
        return self._ext_rays

    def tip_of_stretched_cone(self, c, backend="all", check=True,
                              constraint_error_tol=1e-4, verbose=0):
        """
        **Description:**
        Finds the tip of the stretched cone via quadratic programming. The
        stretched cone is defined as the convex polyhedral region inside the
        cone that is at least a distance ```c``` from any of its defining
        hyperplanes.

        **Arguments:**
        - ```c``` (float): A real positive number specifying the stretching of
          the cone (i.e. the minimum distance to the defining hyperplanes).
        - ```backend``` (string, optional, default="all"): String that
          specifies the optimizer to use. Options are "all", "mosek" and
          "cvxopt".
        - ```checks``` (boolean, optional, default=True): Flag that specifies
          whether to check if the output of the optimizer is consistent and
          satisfies constraint_error_tol.
        - ```constraint_error_tol``` (float, optional, default=1e-4): Error
          tolerence for the linear constraints.
        - ```verbose``` (integer, optional, default=0): The verbosity level.
          - verbose = 0: Do not print anything.
          - verbose = 1: Print warnings when optimizers fail.

        **Returns:**
        (tuple) A tuple with two components. The first one is a float that
        specifies the distance to the tip of the stretched cone, and the second
        one is a vector specifying the location.
        """
        backends = ['all', 'mosek', 'cvxopt']
        if backend not in backends:
            raise Exception("Invalid backend. "
                        f"The options are: {backends}.")
        hp = self.hyperplanes()
        ## The problem is defined as:
        ## Minimize (1/2) x.Q.x + p.x
        ## Subject to G.x <= h
        Q = 2*np.identity(hp.shape[1], dtype=float)
        p = np.zeros(hp.shape[1], dtype=float)
        h = np.full(hp.shape[0], (-c,), dtype=float)
        G = -1*hp.astype(dtype=float)
        Q_cvxopt = cvxopt.matrix(Q)
        p_cvxopt = cvxopt.matrix(p)
        h_cvxopt = cvxopt.matrix(h)
        G_cvxopt = cvxopt.matrix(G)
        optimization_done = False
        if backend == "all":
            for b in backends[1:]:
                if b == "mosek" and not config.mosek_is_activated:
                    continue
                solution = self.tip_of_stretched_cone(c,backend=b, check=check,
                    constraint_error_tol=constraint_error_tol, verbose=verbose)
                if solution is not None:
                    return solution
            raise Exception("All available quadratic programming backends "
                            "have failed.")
        if backend == "mosek":
            cvxopt.solvers.options["mosek"] = {mosek.iparam.num_threads: 1,
                                               mosek.iparam.log: 0}
            try:
                solution = cvxopt.solvers.qp(Q_cvxopt, p_cvxopt, G_cvxopt,
                                             h_cvxopt, solver="mosek")
                assert solution["status"] == "optimal"
            except:
                if verbose >= 1:
                    print("Quadratic programming error: mosek failed. "
                          f"Returned status: {solution['status']}")
            else:
                optimization_done = True
                solution_x = [x[0] for x in np.array(solution["x"]).tolist()]
                solution_val = solution["primal objective"]
        elif backend == "cvxopt":
            cvxopt.solvers.options["abstol"] = 1e-4
            cvxopt.solvers.options["reltol"] = 1e-4
            cvxopt.solvers.options["feastol"] = 1e-2
            cvxopt.solvers.options["maxiters"] = 1000
            cvxopt.solvers.options["show_progress"] = False
            try:
                solution = cvxopt.solvers.qp(Q_cvxopt, p_cvxopt, G_cvxopt,
                                             h_cvxopt)
                assert solution["status"] == "optimal"
            except:
                if verbose >= 1:
                    print("Quadratic programming error: cvxopt failed. "
                          f"Returned status: {solution['status']}")
            else:
                optimization_done = True
                solution_x = [x[0] for x in np.array(solution["x"]).tolist()]
                solution_val = solution["primal objective"]
        if optimization_done and check:
            res = max(np.dot(G, solution_x)) + c
            if res > constraint_error_tol or solution_val < 0:
                optimization_done = False
                raise Exception("Quadratic programming error: "
                                "Large numerical error. Try raising "
                                "constraint_error_tol, or using a different "
                                "backend")
        if optimization_done:
            return (np.sqrt(solution_val), np.array(solution_x))

    def is_solid(self, backend=None, c=0.01):
        """
        **Description:**
        Returns True if the cone is solid, i.e. if it is full-dimensional.

        :::note
        If the generating rays are known then this can simply be checked by
        computing the dimension of the linear space that they span. However,
        when only the hyperplane inequalities are known this can be a difficult
        problem. When using PPL as the backend, the convex hull is explicitly
        constructed and checked. The other backends try to solve an
        optimization problem inside the stretched cone, which fails if the cone
        is not solid. The latter approach is much faster, but there can be
        extremely narrow cones where the optimization fails and this function
        returns a false negative. Mosek is recommended when using such
        extremely narrow cones.
        :::

        **Arguments:**
        - ```backend``` (string, optional): Specifies which backend to use.
          Available options are "ppl", "ortools", and any backends available
          for the [```tip_of_stretched_cone```](#tip_of_stretched_cone)
          function. If not specified, it uses Mosek if it is activated, or
          ORTools otherwise.
        - ```c``` (float, optional, default=0.01): A number used to create the
          stretched cone and try to find the tip. This is ignored when using
          PPL as the backend.

        **Returns:**
        (boolean) The truth value of the cone being solid.
        """
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            return np.linalg.matrix_rank(self._rays) == self._ambient_dim
        if backend is None:
            backend = ("all" if config.mosek_is_activated else "ortools")
        if backend == "ppl":
            cs = ppl.Constraint_System()
            vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
            for h in self._hyperplanes:
                cs.insert(sum(h[i]*vrs[i] for i in range(self._ambient_dim)
                              ) >= 0 )
            cone = ppl.C_Polyhedron(cs)
            self._is_solid = cone.affine_dimension() == self._ambient_dim
            return self._is_solid
        if backend == "ortools":
            hp = self._hyperplanes.tolist()
            solve_ctr = 0
            while solve_ctr < 10:
                obj_vec = np.dot(np.random.random(size=len(hp)), hp)
                solver = pywraplp.Solver("find_pt",
                                pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
                obj = solver.Objective()
                var = []
                for i in range(self._ambient_dim):
                    var.append(solver.NumVar(-solver.infinity(),
                                             solver.infinity(),
                                             f"x_{i}"))
                    obj.SetCoefficient(var[-1], obj_vec[i])
                obj.SetMinimization()
                cons_list = []
                for v in hp:
                    cons_list.append(solver.Constraint(c, solver.infinity()))
                    for j in range(self._ambient_dim):
                        cons_list[-1].SetCoefficient(var[j], v[j])
                status = solver.Solve()
                if status in (solver.FEASIBLE, solver.OPTIMAL):
                    self._is_solid = True
                    return self._is_solid
                elif status == solver.INFEASIBLE:
                    self._is_solid = False
                    return self._is_solid
                else:
                    print(f"Solver returned status: {status}. Trying again.")
                    solve_ctr += 1
            print("Linear solver failed too many times. "
                  "Assuming problem infeasible.")
            self._is_solid = False
            return self._is_solid
        if backend in ("all", "mosek", "cvxopt"):
            opt_res = None
            try:
                opt_res = self.tip_of_stretched_cone(c, backend=backend)
            except:
                pass
            self._is_solid = opt_res is not None
            return self._is_solid
        else:
            backends = ["ppl", "ortools", "all", "mosek", "cvxopt"]
            raise Exception(f"Available options for backends are: {backends}")

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
        - ```backend``` (string, optional): Specifies which backend to use.
          Available options are "nnls", and any backends available for the
          [```is_solid```](#is_solid) function. If not specified, it uses
          the default backend for the [```is_solid```](#is_solid) function.
        - ```tol``` (float, optional, default=1e-7): The tolerance for
          determining when a linear subspace is found. This is only used for
          the NNLS backend.

        **Returns:**
        (boolean) The truth value of the cone being pointed.
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

    def is_simplicial(self):
        """
        **Description:**
        Returns True if the cone is simplicial.

        **Arguments:**
        None.

        **Returns:**
        (boolean) The truth value of the cone being simplicial.
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
        (boolean) The truth value of the cone being smooth.
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

    def intersection(self, other):
        """
        **Description:**
        Computes the intersection with another cone.

        **Arguments:**
        - ```other``` (Cone or list): The other cone that is being intersected,
          or a list of cones to intersect with.

        **Returns:**
        (Cone) The cone that results from the intersection.
        """
        if isinstance(other, Cone):
            return Cone(self.dual().rays().tolist()
                            + other.dual().rays().tolist()
                        ).dual()
        dual_rays = self.dual().rays().tolist()
        for c in other:
            if not isinstance(c, Cone):
                raise Exception("Elements of the list must be Cone objects.")
            if c.ambient_dim() != self.ambient_dim():
                raise Exception("Ambient lattices must have the same"
                                "dimension.")
            dual_rays.extend(c.dual().rays().tolist())
        return Cone(dual_rays).dual()


def is_extremal(A, b, i=None, q=None, tol=1e-4):
    """
    **Description:**
    Auxiliary function that is used to find the extremal rays of cones. Returns
    True if the ray is extremal and False otherwise. It has additional
    parameters that are used when parallelizing.

    **Arguments:**
    - ```A``` (list): A matrix where the columns are rays (excluding b).
    - ```b``` (list): The vector that will be checked if it can be expressed as
      a positive linear combination of the columns of A.
    - ```i``` (integer, optional): An id number that is used when
      parallelizing.
    - ```q``` (Queue, optional): A queue that is used when parallelizing.
    - ```tol``` (float, optional, default=1e-4): The tolerance for determining
      whether a ray is extremal.

    **Returns:**
    (boolean or None) The truth value of the ray baing extremal. If the process
    fails then it returns nothing.
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
