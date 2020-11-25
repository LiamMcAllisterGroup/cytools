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

from cytools.utils import *
from scipy.spatial import ConvexHull
from flint import fmpz_mat, fmpq_mat, fmpz, fmpq
import numpy as np
from scipy.optimize import nnls
from ortools.linear_solver import pywraplp
import cvxopt
import mosek
from scipy.sparse import dok_matrix
# PPL is an optional dependency. Most functions work without it.
try:
    import ppl
    HAVE_PPL = True
except:
    HAVE_PPL = False


class Cone:
    """Class that handles lattice polytope computations."""

    def __init__(self, rays=None, hyperplanes=None, check=True):
        """
        Creates a Cone object.  It can be specified by a list of rays, or by a
        list of hyperplane normals.

        Args:
            rays (list, optional): A list of rays that generates the cone.  If
                it is not specified then the hyperplane normals must be
                specified.
            hyperplanes (list, optional): A list of inward hyperplane normals
                that define the cone.  If it is not specified then the
                generating rays must be specified.
            check (boolean, optional): Whether to check the input. Recommended
                if constructing a cone directly.
        """
        if ((rays is None and hyperplanes is None)
                or (rays is not None and hyperplanes is not None)):
            raise Exception("Exactly one of \"rays\" and \"hyperplanes\" "
                            "must be specified.")
        if rays is not None:
            tmp_rays = np.array(rays)
            if len(tmp_rays.shape) != 2:
                raise Exception("Input must be a matrix.")
            t = type(tmp_rays[0,0])
            if t == fmpz:
                rays = np.array(tmp_rays, dtype=int)
            elif t == fmpq:
                rays = np_fmpq_to_float(tmp_rays)
            elif t not in [np.int64, np.float64]:
                raise Exception("Unsupported data type.")
            if check:
                tmp_rays = []
                if len(rays) < 1:
                    raise Exception("At least one rays is required.")
                for r in rays:
                    g = gcd_list(r)
                    if g < 1e-6:
                        continue
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
                hyperplanes = np.array(tmp_hp, dtype=int)
            elif t == fmpq:
                hyperplanes = np_fmpq_to_float(tmp_hp)
            elif t not in [np.int64, np.float64]:
                raise Exception("Unsupported data type.")
            if check:
                tmp_hp = []
                if len(hyperplanes) < 1:
                    raise Exception("At least one hyperplane is required.")
                for r in hyperplanes:
                    g = gcd_list(r)
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
            self.dim = None
        # Initialize remaining hidden attributes
        self._dual = None
        self._ext_rays = None
        self._is_solid = None

    def clear_cache(self):
        """Clears the cached results of any previous computation."""
        self._dual = None
        self._ext_rays = None
        self._is_solid = None
        if self._rays_were_input:
            self._hyperplanes = None
        else:
            self._rays = None

    def __repr__(self):
        """Returns a string describing the cone."""
        if self._rays is not None:
            return (f"A {self._dim}-dimensional rational polyhedral cone in "
                    f"RR^{self._ambient_dim} generated by {len(self._rays)} "
                    f"rays")
        return (f"A rational polyhedral cone in RR^{self._ambient_dim} "
                f"defined by {len(self._hyperplanes)} hyperplanes")

    def rays(self):
        """
        Returns the (not necessarily extremal) rays that generate the cone.
        """
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        elif self._rays is not None:
            return np.array(self._rays)
        if not HAVE_PPL:
            raise Exception("PPL is necessary for this computation.")
        if self._ambient_dim >= 12:
            print("Warning: This operation might take a while for d > ~12 "
                  "and is likely impossible for d > ~18.")
        cs = ppl.Constraint_System()
        vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
        for h in self._hyperplanes:
            cs.insert(sum(h[i]*vrs[i] for i in range(self._ambient_dim)
                          ) >= 0 )
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
        Returns the inward normals to the hyperplanes that define the cone.
        """
        if self._hyperplanes is not None:
            return np.array(self._hyperplanes)
        if not HAVE_PPL:
            raise Exception("PPL is necessary for this computation.")
        if self._ambient_dim >= 12:
            print("Warning: This operation might take a while for d > ~12 "
                  "and is likely impossible for d > ~18.")
        ext_rays = self.extremal_rays()
        gs = ppl.Generator_System()
        vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
        gs.insert(ppl.point(0))
        for r in ext_rays:
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
        """Returns the dual cone."""
        if self._dual is None:
            if self._rays is not None:
                self._dual = Cone(hyperplanes=self.rays(), check=False)
            else:
                self._dual = Cone(rays=self.hyperplanes(), check=False)
            self._dual._dual = self
        return self._dual

    def extremal_rays(self, tol=1e-4, n_threads=1, verbose=False):
        """
        Returns the extremal rays of the polytope.

        Args:
            tol (float, optional, default=1e-5): Optional parameter specifying
                the allowed rounding error.

        Returns:
            list: The list of extremal rays.
        """
        if self._ext_rays is not None:
            return np.array(self._ext_rays)
        ext_rays = find_extremal(self.rays(), tol=tol, n_threads=n_threads,
                             verbose=verbose)
        self._ext_rays = np.array(ext_rays, dtype=int)
        return np.array(self._ext_rays)

    def dmin(self, c, backend="all", check=True, constraint_error_tol=1e-4, verbose=0):
        """
        Finds the tip of the stretched cone via quadratic programming.

        Args:
            c (float): A real positive number specifying the stretching of the
                cone.
            backend (string, optional, default="all"): String that specifies the optimizer to
                use. Options are "all", "mosek" and "cvxopt".
            checks (boolean, optional, default=True): Flag that specifies
                whether to check if the output of the optimizer is consistent
                and satisfies constraint_error_tol.
            constraint_error_tol (float, optional, default=1e-4): Error tolerence
                for the linear constraints.
            verbose (int, optional, default=0): Verbosity level:
            - verbose = 0: Do not print anything.
            - verbose = 1: Print warnings when optimizers fail.

        Returns:
            tuple: A tuple containing the distance to the tip of the stretched
                cone (dmin), as well as the vector specifying the location.
        """
        backends = ['all', 'mosek', 'cvxopt']
        if backend not in backends:
            raise Exception("Invalid backend. "
                        f"The options are: {backends}.")

        hp = self.hyperplanes()
        ## The problem is defined as:
        ## Minimize (1/2) x.Q.x + p.x
        ## Subject to G.x <= h
        Q = 2*np.identity(hp.shape[1],dtype=np.double)
        p = np.zeros(hp.shape[1],dtype=np.double)
        h = np.full(hp.shape[0],(-c,),dtype=np.double)
        G = -1*hp.astype(dtype=np.double)
        Q_cvxopt = cvxopt.matrix(Q)
        p_cvxopt = cvxopt.matrix(p)
        h_cvxopt = cvxopt.matrix(h)
        G_cvxopt = cvxopt.matrix(G)

        optimization_done = False
        if backend == "all":
            for b in backends[1:]:
                solution = self.dmin(c, backend=b, check=check,
                    constraint_error_tol=constraint_error_tol, verbose=verbose)
                if solution is not None:
                    return solution
            raise Exception("""All available quadratic programming backends have failed.""")

        elif backend == "mosek":
            cvxopt.solvers.options['mosek'] = {mosek.iparam.num_threads: 1, mosek.iparam.log: 0}
            try:
                solution = cvxopt.solvers.qp(Q_cvxopt,p_cvxopt,G_cvxopt,h_cvxopt,solver='mosek')
                assert solution['status']=='optimal'
            except:
                if verbose >= 1:
                    print("Quadratic programming error: mosek failed. Returned status:"
                                , solution['status'])
            else:
                optimization_done = True
                solution_x = [x[0] for x in np.array(solution['x']).tolist()]
                solution_val = solution['primal objective']
        elif backend == "cvxopt":
            cvxopt.solvers.options['abstol'] = 1e-4
            cvxopt.solvers.options['reltol'] = 1e-4
            cvxopt.solvers.options['feastol'] = 1e-2
            cvxopt.solvers.options['maxiters'] = 1000
            cvxopt.solvers.options['show_progress'] = False
            try:
                solution = cvxopt.solvers.qp(Q_cvxopt,p_cvxopt,G_cvxopt,h_cvxopt)
                assert solution['status']=='optimal'
            except:
                if verbose >= 1:
                    print("Quadratic programming error: cvxopt failed. Returned status:"
                               , solution['status'])
            else:
                optimization_done = True
                solution_x = [x[0] for x in np.array(solution['x']).tolist()]
                solution_val = solution['primal objective']
        if optimization_done and check:
            res = max(np.dot(G, solution_x)) + c
            if res>constraint_error_tol or solution_val<0:
                optimization_done = False
                raise Exception("Quadratic programming error: Large numerical error."
                             " Try raising constraint_error_tol, or try a different backend")
        if optimization_done:
            return (np.sqrt(solution_val), np.array(solution_x))
        else:
            return None

    def is_solid(self, backend="ortools", c=0.01):
        """
        Returns True if the cone is solid, i.e. if it is full-dimensional.

        If the generating rays are known then this can simply be checked by
        computing the dimension of the linear space that they span.  However,
        when only the hyperplane inequalities are known this can be a difficult
        problem.  There are three available backends for this computation.  PPL
        explicitly constructs the convex hull defined by the inequalities and
        thus only works for low dimensions.  ORTools and Mosek work by solving
        an optimization problem and work even at high dimensions.

        Args:
            backend (string, optional, default="ortools"): Specifies which
                backend to use. Available options are "ppl", "ortools", and
                "mosek".
            c (float, optional, default=0.01): A number used to create a
                "stretched cone" and try to find the tip. This is used for all
                backends except for PPL.

        Returns:
            boolean: The truth value of whether the cone is solid.
        """
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            return np.linalg.matrix_rank(self._rays) == self._ambient_dim
        if backend == "ppl":
            cs = ppl.Constraint_System()
            vrs = [ppl.Variable(i) for i in range(self._ambient_dim)]
            for h in self._hyperplanes:
                cs.insert(sum(h[i]*vrs[i] for i in range(self._ambient_dim)
                              ) >= 0 )
            cone = ppl.C_Polyhedron(cs)
            self._is_solid = cone.affine_dimension() == self._ambient_dim
            return self._is_solid
        elif backend == "ortools":
            hp = self._hyperplanes.tolist()
            solve_ctr = 0
            while solve_ctr < 10:
                obj_vec = np.dot(np.random.random(size=len(hp)), hp)
                solver = pywraplp.Solver('find_pt',
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
        elif backend in ["mosek", "gekko"]:
            opt_res = None
            try:
                opt_res = self.dmin(c, backend=backend)
            except:
                pass
            self._is_solid = opt_res is not None
            return self._is_solid
        else:
            backends = ["ppl", "ortools", "mosek"]
            raise Exception("Available options for optimizers are ortools or mosek.")
