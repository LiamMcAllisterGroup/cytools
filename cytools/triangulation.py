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
This module contains tools designed to compute triangulations.
"""

from cytools.calabiyau import CalabiYau
from cytools.cone import Cone
from cytools.utils import gcd_list
from scipy.spatial import ConvexHull
import numpy as np
import subprocess
from itertools import combinations
from scipy.linalg import null_space
from scipy.optimize import nnls
from flint import fmpz_mat
import re
import copy


class Triangulation:
    """A class that handles triangulations of point configurations."""

    def __init__(self, triang_pts, poly=None, heights=None, make_star=False,
                 simplices=None, backend="qhull", backend_dir=None):
        """
        Creates a Triangulation object.

        Typically a triangulation object should not be created directly by the
        user and instead should be created by some of the other functions in
        CYTools. Some checks on the input are always performed, as it is
        nesessary to have all the data organized properly so that the results
        obtained with the Calabi-Yau class are correct. In particular, the
        ordering of the points needs to be consistent with what the Polytope
        class returns. For this reason, this function first attempts to fix
        discrepancies, and if it fails then it disallows the creation of a
        Calabi-Yau object.

        Args:
            triang_pts (list): The list of points to be triangulated.
            poly (Polytope, optional): The ambient polytope on which the points
                sit.  If not specified, it is constructed as the convex hull
                of the given points.
            heights (list, optional): A list of heights specifying the regular
                triangulation.  When not secified, it will return the Delaunay
                triangulation when using CGAL, a triangulation obtained from
                random heights near the Delaunay when using QHull, or the
                placing triangulation when using TOPCOM.  Heights can only be
                specified when using CGAL or QHull as the backend.
            make_star (boolean, optional, default=False): Indicates whether to
                turn the triangulation into a star triangulation by delering
                internal lines and connecting all points to the origin.
            exclude_pts (list, optional): A list of points to be excluded in
                the triangulation.  If not specified, it excludes points
                interior to facets when the polytope is reflexive.  When it is
                desired to include all points an empty list should be given.
            simplices (list, optional): A list of simplices specifying the
                triangulation.  Each simplex is a list of point indices. This
                is useful when a triangulation was previously computed and it
                needs to be inputted again.  Note that the ordering of the
                points needs to be consistent.
            backend (string, optional, default="qhull"): Specifies the backend
                used to compute the triangulation.  The available options are
                "qhull", "cgal", and "topcom".  QHull is the default one as it
                is included in scipy, but it has some problems when there is
                degeneracy with the heights and when polytopes are too large.
                CGAL is far better in general, but it requires additional
                software to be installed.
            backend_dir (string, optional): This can be used to specify the
                location of CGAL or TOPCOM binaries when they are not in PATH.
        """
        self._triang_pts = np.array(triang_pts, dtype=int)
        triang_pts_tup = [tuple(pt) for pt in self._triang_pts]
        heights = copy.deepcopy(heights)
        if poly is None:
            from cytools.polytope import Polytope
            self._poly = Polytope(self._triang_pts)
        else:
            self._poly = poly
        if not self._poly.is_full_dimensional():
            raise Exception("Only triangulations of full-dimensional point "
                            "configurations are supported.")
        self._allow_cy = True
        # Find out if all points are being used and check the consistency of
        # the input
        poly_pts_mpcp = [tuple(pt) for pt in
                           self._poly.points_not_interior_to_facets()]
        if triang_pts_tup == poly_pts_mpcp:
            self._all_poly_pts = False
        else:
            poly_pts = [tuple(pt) for pt in self._poly.points()]
            if triang_pts_tup == poly_pts:
                self._all_poly_pts = True
            else:
                # Here we attempt to fix the ordering or else we disallow the
                # creation of a Calabi-Yau object
                print("Warning: Inconsistency found in the input. "
                      "Attempting to fix...")
                if (set(triang_pts_tup) == set(poly_pts_mpcp)
                        and set(triang_pts_tup) == set(poly_pts_mpcp)):
                    ind_dict = {i:poly_pts_mpcp.index(triang_pts_tup[i])
                                for i in range(len(triang_pts_tup))}
                    triang_pts_tup = poly_pts_mpcp
                    self._triang_pts = np.array(triang_pts_tup)
                    if heights is not None:
                        heights = [heights[ind_dict[i]]
                                    for i in range(len(heights))]
                    if simplices is not None:
                        simplices = [[ind_dict[i] for i in s]
                                        for s in simplices]
                    print("Sucessfully fixed the input.")
                elif (set(triang_pts_tup) == set(poly_pts_mpcp)
                        and set(triang_pts_tup) != set(poly_pts_mpcp)):
                    print("Ignoring inputted heights and simplices...")
                    triang_pts_tup = poly_pts_mpcp
                    self._triang_pts = np.array(triang_pts_tup)
                    heights = None
                    simplices = None
                    print("Sucessfully fixed the input.")
                elif (set(triang_pts_tup) == set(poly_pts)
                        and set(triang_pts_tup) == set(poly_pts)):
                    ind_dict = {i:poly_pts.index(triang_pts_tup[i])
                                for i in range(len(triang_pts_tup))}
                    triang_pts_tup = poly_pts
                    self._triang_pts = np.array(triang_pts_tup)
                    if heights is not None:
                        heights = [heights[ind_dict[i]]
                                    for i in range(len(heights))]
                    if simplices is not None:
                        simplices = [[ind_dict[i] for i in s]
                                        for s in simplices]
                    print("Sucessfully fixed the input.")
                elif (set(triang_pts_tup) == set(poly_pts)
                        and set(triang_pts_tup) != set(poly_pts)):
                    print("Ignoring inputted heights and simplices...")
                    triang_pts_tup = poly_pts
                    self._triang_pts = np.array(triang_pts_tup)
                    heights = None
                    simplices = None
                    print("Sucessfully fixed the input.")
                else:
                    self._allow_cy = False
                    print("Warning: Failed to fix the input. Creation of a "
                          "Calabi-Yau object will be disallowed. Other "
                          "functions may also return incorrect results.")
        # Try to find the index of the origin.
        try:
            self._origin_index = triang_pts_tup.index((0,)*self._poly.dim())
        except:
            self._origin_index = -1
            make_star = False
        self._pts_dict = {ii:i for i,ii in enumerate(triang_pts_tup)}
        self._backend = backend
        self._backend_dir = backend_dir
        # Now save input triangulation or construct it
        if simplices is not None:
            self._simplices = np.array(simplices)
            if self._simplices.shape[1] != self._triang_pts.shape[1]+1:
                raise Exception("Input simplices have wrong dimension.")
            self._heights = None
        else:
            backends = ["qhull", "cgal", "topcom", None]
            if heights is None:
                # Heights need to be perturbed around the Delaunay heights for
                # QHull or the triangulation might not be regular. If using
                # CGAL then they are not perturbed.
                if backend is None:
                    raise Exception("The simplices of the triangularion must "
                                    "be specified when working without a "
                                    "backend")
                elif backend == "qhull":
                    heights = [np.dot(p,p) + 0.001*np.random.random()
                                for p in self._triang_pts]
                elif backend == "cgal":
                    heights = [np.dot(p,p) for p in self._triang_pts]
                elif backend == "topcom":
                    heights = None
                else:
                    raise Exception("Invalid backend. "
                                    f"Options are: {backends}.")
            self._heights = heights
            # Now run the appropriate triangulation function
            if backend == "qhull":
                self._simplices = qhull_triangulate(self._triang_pts,
                                                    heights)
                if make_star:
                    facets_ind = [[self._pts_dict[tuple(pt)]
                                    for pt in f.boundary_points()]
                                        for f in self._poly.facets()]
                    self._simplices = convert_to_star(self._simplices,
                                                      facets_ind,
                                                      self._origin_index)
            # With CGAL we can obtain a star triangulation more quickly by
            # setting the height of the origin to be much lower than the
            # others. In theory this can also be done with QHull, but one
            # sometimes runs into errors.
            elif backend == "cgal":
                if make_star:
                    origin_offset = 1e6
                    heights[self._origin_index] = min(heights) - origin_offset
                self._simplices = cgal_triangulate(self._triang_pts,
                                                   heights,
                                                   cgal_dir=backend_dir)
            else: # Use TOPCOM
                self._simplices = topcom_triangulate(self._triang_pts,
                                                     topcom_dir=backend_dir)
                if make_star:
                    facets_ind = [[self._pts_dict[tuple(pt)]
                                    for pt in f.boundary_points()]
                                        for f in self._poly.facets()]
                    self._simplices = convert_to_star(self._simplices,
                                                      facets_ind,
                                                      self._origin_index)
        # Initialize remaining hidden attributes
        self._is_fine = None
        self._is_regular = None
        self._is_star = None
        self._is_valid = None
        self._gkz_phi = None
        self._sr_ideal = None
        self._cpl_cone = None
        self._cy = None

    def clear_cache(self, recursive=True):
        """Clears the cached results of any previous computation."""
        self._is_fine = None
        self._is_regular = None
        self._is_star = None
        self._is_valid = None
        self._gkz_phi = None
        self._sr_ideal = None
        self._cpl_cone = None
        self._cy = None
        if recursive:
            self._poly.clear_cache()

    def __repr__(self):
        """Returns a string describing the triangulation."""
        return (f"A {('fine' if self.is_fine() else 'non-fine')}, "
                + (("regular, " if self._is_regular else "irregular, ")
                    if self._is_regular is not None else "") +
                f"{('star' if self.is_star() else 'non-star')} "
                f"triangulation of a {self.dim()}-dimensional "
                f"polytope in ZZ^{self.dim()}")

    def polytope(self):
        """Returns the polytope being triangulated."""
        return self._poly

    def points(self):
        """Returns the points of the triangulation."""
        return np.array(self._triang_pts)

    def points_to_indices(self, points):
        """Returns the list of indices corresponding to the given points."""
        return np.array([self._pts_dict[tuple(pt)] for pt in points])

    def simplices(self):
        """Returns the simplices of the triangulation."""
        return np.array(self._simplices)

    def dim(self):
        """Returns the dimension of the point configuration."""
        return self._poly.dim()

    def is_fine(self):
        """
        Returns True if the triangulation is fine (all the points are used).
        """
        if self._is_fine is not None:
            return self._is_fine
        self._is_fine = (len(set.union(*[set(s) for s in self._simplices]))
                        == len(self._triang_pts))
        return self._is_fine

    def is_regular(self, optimizer="ortools"):
        """Returns True if the triangulation is regular."""
        if self._is_regular is not None:
            return self._is_regular
        self._is_regular = self.cpl_cone().is_solid()
        return self._is_regular

    def is_star(self):
        """Returns True if the triangulation is star."""
        if self._is_star is not None:
            return self._is_star
        self._is_star =  (
                len(set.intersection(*[set(s) for s in self._simplices])) > 0)
        return self._is_star

    def is_valid(self):
        """
        Returns True if the presumed triangulation meets all requirements to be
        a triangularion.  The simplices must cover the full volume of the
        convex hull, and they cannot intersect at full-dimensional regions.
        """
        if self._is_valid is not None:
            return self._is_valid
        simps = self.simplices()
        pts = self.points()
        pts_ext = np.empty((pts.shape[0],pts.shape[1]+1), dtype=int)
        pts_ext[:,:-1] = pts
        pts_ext[:,-1] = 1
        # First check if the dimensions of the simplices and polytope match up
        if simps.shape[1] != self.dim()+1:
            self._is_valid = False
            return self._is_valid
        # Then check if the volumes add up to the volume of the polytope
        v = 0
        for s in simps:
            tmp_v = abs(int(round(np.linalg.det([pts_ext[i] for i in s]))))
            if tmp_v == 0:
                self._is_valid = False
                return self._is_valid
            v += tmp_v
        if v != self._poly.volume():
            self._is_valid = False
            return self._is_valid
        # Finally, check if simplices have full-dimensional intersections
        pts_ext = pts_ext.tolist()
        for i in range(len(simps)):
            for j in range(i+1,len(simps)):
                s1 = simps[i]
                s2 = simps[j]
                # Compute rays of the dual cones
                d1 = []
                for ii in s1:
                    m = fmpz_mat([pts_ext[jj] for jj in s1 if jj != ii])
                    r = np.array(m.nullspace()[0].transpose().table()[0],
                                 dtype=int)
                    if r.dot(pts_ext[ii]) < 0:
                        r *= -1
                    d1.append(r)
                d2 = []
                for ii in s2:
                    m = fmpz_mat([pts_ext[jj] for jj in s2 if jj != ii])
                    r = np.array(m.nullspace()[0].transpose().table()[0],
                                 dtype=int)
                    if r.dot(pts_ext[ii]) < 0:
                        r *= -1
                    d2.append(r)
                # Merge the rays and check if there is a linear subspace
                A = np.concatenate((d1,d2), axis=0).T
                A_ext = np.empty((A.shape[0]+1,A.shape[1]), dtype=int)
                A_ext[:-1,:] = A
                A_ext[-1,:] = 1
                b = [0]*(len(s1)) + [1]
                res = nnls(A_ext,b)[1]
                if res > 1e-4:
                    self._is_valid = False
                    return self._is_valid
        self._is_valid = True
        return self._is_valid

    def gkz_phi(self):
        """Returns the GKZ phi vector of the triangulation."""
        if self._gkz_phi is not None:
            return np.array(self._gkz_phi)
        ext_pts = [tuple(pt)+(1,) for pt in self._triang_pts]
        phi = np.array([0]*len(ext_pts))
        for s in self._simplices:
            simp_vol = int(round(abs(np.linalg.det([ext_pts[i] for i in s]))))
            for i in s:
                phi[i] += simp_vol
        self._gkz_phi = phi
        return np.array(self._gkz_phi)

    def random_flips(self, N, only_regular=True, topcom_dir=None):
        """
        Returns a list of triangulations obtained after performing N random
        bistellar flips.

        Args:
            N (int): The number of flips to perform.
            only_regular (boolean, optional, default=True): Whether to restrict
                to only regular triangulations.
            topcom_dir (string, optional): The directory of the TOPCOM
                installation in case the binaries are not in PATH.
        """
        pts_str = str([list(pt)+[1] for pt in self._triang_pts])
        triang_str = str([list(s) for s in self._simplices]
                         ).replace("[","{").replace("]","}")
        flips_str = "(" + str((1 if only_regular else -1)*N) + ")"
        topcom_input = pts_str + "[]" + triang_str + flips_str
        topcom_bin = ((topcom_dir + "/" if topcom_dir is not None else "")
                      +"topcom-randomWalk")
        topcom = subprocess.Popen((topcom_bin,), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        topcom_res, topcom_err = topcom.communicate(input=topcom_input)
        if len(topcom_res)==0:
            return []
        triangs_list = [eval(r) for r in topcom_res.strip().split("\n")]
        triangs = []
        for t in triangs_list:
            triangs.append(Triangulation(self._triang_pts, self._poly,
                                         simplices=t))
        return triangs

    def neighbor_triangs(self, only_regular=True, topcom_dir=None):
        """Returns triangulations that differ by one bistellar flip."""
        return self.random_flips(1, only_regular, topcom_dir=topcom_dir)

    def sr_ideal(self):
        """Returns the Stanley–Reisner ideal if the triangulation is star."""
        if self._sr_ideal is not None:
            return np.array(self._sr_ideal)
        if not self.is_star():
            raise Exception("Triangulation is not star.")
        points = (frozenset(range(len(self._triang_pts)))
                  - frozenset([self._origin_index]))
        simplices = [[i for i in s if i != self._origin_index]
                      for s in self.simplices()]
        simplex_tuples = tuple(frozenset(frozenset(ss)
                                         for s in simplices
                                            for ss in combinations(s, dd))
                                                for dd in range(1,
                                                    self._poly._ambient_dim+1))
        SR_ideal = set()
        checked = set()
        for i in range(len(simplex_tuples)-1):
            for tup in simplex_tuples[i]:
                for j in points:
                    k = tup.union((j,))
                    if k in checked or len(k) != len(tup)+1:
                        continue
                    checked.add(k)
                    in_SR = False
                    for order in range(1, i+1):
                        for t in combinations(tup, order):
                            if frozenset(t+(j,)) in SR_ideal:
                                in_SR = True
                    if k not in simplex_tuples[i+1] and not in_SR:
                        SR_ideal.add(k)
        self._sr_ideal = sorted([sorted(s)for s in SR_ideal],
                                                     key=lambda x: (len(x),x))
        return np.array(self._sr_ideal, dtype=object)

    def cpl_cone(self):
        """
        Compute the cone of strictly convex piecewise linear functions
        defining the triangulation.  It is computed by finding the defining
        hyperplane equations.  The triangularion is regular if and only if this
        cone is solid (i.e. full-dimensional).
        """
        if self._cpl_cone is not None:
            return self._cpl_cone
        pts = [tuple(pt) + (1,) for pt in self._triang_pts]
        dim = self.dim()
        # Construct sets and pairs sharing a wall
        simps = [set(s) for s in self._simplices]
        walls = []
        for i in range(len(simps)):
            for j in range(i, len(simps)):
                if len(simps[i] & simps[j]) == dim:
                    walls.append((simps[i], simps[j]))
        # We construct a matrix where the first two columns are the two points
        # that are not shared. Then we find the unique linear relation between
        # them
        null_vecs_all = set()
        null_vecs = []
        for w in walls:
            m = np.zeros((dim+1, dim+2), dtype=int)
            diff_pts = list(w[0] ^ w[1])
            comm_pts = list(w[0] & w[1])
            for i in range(len(diff_pts)):
                m[:,i] = pts[diff_pts[i]]
            for i in range(len(comm_pts)):
                m[:,i+2] = pts[comm_pts[i]]
            v = np.array(fmpz_mat(m.tolist()
                                  ).nullspace()[0].transpose().table()[0],
                         dtype=int)
            if v[0] < 0:
                v = -v
            # Reduce the vector
            g = gcd_list(v)
            if g != 1:
                v //= g
            # Construct the full vector (including all points)
            full_v = [0]*len(pts)
            for i in range(len(diff_pts)):
                full_v[diff_pts[i]] = v[i]
            for i in range(len(comm_pts)):
                full_v[comm_pts[i]] = v[i+2]
            full_v = tuple(full_v)
            if full_v not in null_vecs_all:
                null_vecs_all.add(full_v)
                null_vecs.append(full_v)
        self._cpl_cone = Cone(hyperplanes=null_vecs, check=False)
        return self._cpl_cone

    def get_cy(self):
        """
        Returns a CalabiYau object corresponding to the anti-canonical
        hypersurface on the toric variety defined by the fine, star, regular
        triangulation.
        """
        if self._cy is not None:
            return self._cy
        if not self._allow_cy:
            raise Exception("There is a problem with the data of the "
                            "triangulation. Constructing a CY is disallowed "
                            "since computations will likely be erroneous.")
        if not self._poly.is_favorable():
            raise Exception("Only favorable CYs are currently supported.")
        if not self.is_star() or not self.is_fine():
            raise Exception("Triangulation is non-fine or non-star.")
        self._cy = CalabiYau(self)
        return self._cy

def convert_to_star(simplices, facets, star_origin):
    """
    Turns a triangulation into a star triangulation by deleting internal lines
    and connecting all points to the origin.  This is only reliable for
    reflexive polytopes and may produce invalid tiangulations for other
    polytopes.

    Args:
        simplices (list): List of simplices of the triangulation. Each simplex
            consists of the indices of the points forming its vertices.
        facets (iterable): The list of facets of the polytope. Each facet
            consists of the indices of the points in the facet.
        star_origin (integer): The index of the point that is used as the star
            origin.

    Returns:
        np.array: A list of simplices forming a star triangulation.
    """
    star_triang = []
    triang = np.array(simplices)
    dim = triang.shape[1] - 1
    for facet in facets:
        for simp in triang:
            overlap = simp[np.isin(simp, facet)].tolist()
            if len(overlap) == dim:
                star_triang.append([star_origin] + overlap)
    return np.array(star_triang)


def qhull_triangulate(points, heights):
    """
    Computes a regular triangulation using QHull.  See the Triangulation class
    for more details.
    """
    lifted_points = [tuple(points[i]) + (heights[i],)
                        for i in range(len(points))]
    hull = ConvexHull(lifted_points)
    # We first pick the lower facets of the convex hull
    low_fac = [hull.simplices[n] for n,nn in enumerate(hull.equations)
                if nn[-2] < 0] # The -2 component is the lifting dimension
    # Then we only take the faces that project to full-dimensional simplices
    # in the original point configuration
    lifted_points = [pt[:-1] + (1,) for pt in lifted_points]
    simp = [s for s in low_fac
                if int(round(np.linalg.det([lifted_points[i] for i in s])))
                    != 0]
    return np.array(sorted([sorted(s) for s in simp]))


def cgal_triangulate(points, heights, cgal_dir=None):
    """
    Computes a regular triangulation using CGAL.  See the Triangulation class
    for more details.
    """
    cgal_bin = ((cgal_dir + "/" if cgal_dir is not None else "")
                + "cgal-triangulate")
    cgal = subprocess.Popen((cgal_bin,), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    cgal_res, cgal_err = cgal.communicate(
                            input=str([list(pt) for pt in points])
                                  + str(list(heights)
                                        ).replace("[", "(").replace("]", ")"))
    if cgal_err != "":
        raise Exception(f"CGAL error: {cgal_err}")
    try:
        simp = eval(cgal_res)
    except:
        raise Exception("Error: Failed to parse CGAL output.")
    return np.array(sorted([sorted(s) for s in simp]))


def topcom_triangulate(points, topcom_dir=None):
    """
    Computes a regular triangulation using TOPCOM.  See the Triangulation class
    for more details.
    """
    topcom_bin = ((topcom_dir + "/" if topcom_dir is not None else "")
                + "topcom-points2finetriang")
    topcom = subprocess.Popen((topcom_bin, "--regular"), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    topcom_res, topcom_err = topcom.communicate(
                                input=str([list(pt)+[1] for pt in points])
                                        + "[]")
    try:
        simp = eval(topcom_res.replace("{", "[").replace("}", "]"))
    except:
        raise Exception("Error: Failed to parse TOPCOM output. "
                        f"stderr: {topcom_err}")
    return np.array(sorted([sorted(s) for s in simp]))


def all_triangulations(points, only_fine=True, only_regular=True,
                       only_star=True, star_origin=None, topcom_dir=None):
    """
    Computes all triangulations of the inputted point configuration using
    TOPCOM.  There is the option to only compute regular or fine
    triangulations.

    Args:
        points (list): The list of points to be triangulated.
        only_regular (boolean, optional, default=True): Whether to restrict to
            regular triangulations.
        only_fine (boolean, optional, default=True): Whether to restrict to
            fine triangulations.
        only_star (boolean, optional, default=True): Whether to restrict to
            star triangulations.
        star_origin (int, optional): The index of the point used as the star
            origin. It needs to be specified if only_star=True.
        topcom_dir (string, optional): This can be used to specify the
            location of the TOPCOM binaries when they are not in PATH.
    """
    if only_star and star_origin is None:
        raise Exception("The star_origin parameter must be specified when "
                        "restricting to star triangulations.")
    topcom_bin = ((topcom_dir + "/" if topcom_dir is not None else "")
                + ("topcom-points2finetriangs" if only_fine
                    else "topcom-points2triangs"))
    topcom = subprocess.Popen(
                        (topcom_bin, ("--regular" if only_regular else "")),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True)
    topcom_res, topcom_err = topcom.communicate(
                                input=str([list(pt)+[1] for pt in points])
                                        + "[]")
    try:
        triangs = [eval("[" + t.replace("{", "[").replace("}", "]") + "]")
                    for t in re.findall(r"\{([^\:]*)\}", topcom_res)]
    except:
        raise Exception("Error: Failed to parse TOPCOM output. "
                        f"stderr: {topcom_err}")
    return [np.array(sorted([sorted(s) for s in t])) for t in triangs
                if (not only_star or all(star_origin in ss for ss in t))]
