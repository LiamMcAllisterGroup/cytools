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

# Standard imports
from itertools import combinations
from ast import literal_eval
import subprocess
import copy
import re
# Third party imports
from scipy.spatial import ConvexHull
from flint import fmpz_mat
import numpy as np
# CYTools imports
from cytools.calabiyau import CalabiYau
from cytools.cone import Cone
from cytools.utils import gcd_list
from cytools import config



class Triangulation:
    """
    This class handles triangulations of lattice polytopes. It can analyze
    various aspects of the triangulation, as well as construct a CalabiYau
    object if the triangulation is suitable.

    :::important
    Generally, objects of this class should not be constructed directly by the
    end user. Instead, they should be created by various functions of the
    [Polytope](./polytope) class.
    :::

    ## Constructor

    ### ```cytools.triangulation.Triangulation```

    **Description:**
    Constructs a ```Triangulation``` object describing a triangulation of a
    lattice polytope. This is handled by the hidden
    [```__init__```](#__init__) function.

    :::note
    Some checks on the input are always performed, as it is nesessary to have
    all the data organized properly so that the results obtained with the
    Calabi-Yau class are correct. In particular, the ordering of the points
    needs to be consistent with what the the ordering the
    [Polytope](./polytope) class uses. For this reason, this function first
    attempts to fix discrepancies, and if it fails then it disallows the
    creation of a Calabi-Yau object.
    :::

    **Arguments:**
    - ```triang_pts``` (list): The list of points to be triangulated.
    - ```poly``` (Polytope, optional): The ambient polytope of the points to be
      triangulated. If not specified, it is constructed as the convex hull of
      the given points.
    - ```heights``` (list, optional): A list of heights specifying the regular
      triangulation. When not secified, it will return the Delaunay
      triangulation when using CGAL, a triangulation obtained from random
      heights near the Delaunay when using QHull, or the placing triangulation
      when using TOPCOM. Heights can only be specified when using CGAL or QHull
      as the backend.
    - ```make_star``` (boolean, optional, default=False): Indicates whether to
      turn the triangulation into a star triangulation by deleting internal
      lines and connecting all points to the origin, or equivalently, by
      decreasing the height of the origin until it is much lower than all other
      heights.
    - ```simplices``` (list, optional): A list of simplices specifying the
      triangulation. Each simplex is a list of point indices. This is useful
      when a triangulation was previously computed and it needs to be used
      again. Note that the ordering of the points needs to be consistent.
    - ```check_input_simplices``` (boolean, optional, default=True): Flag
      that specifies whether to check if the input simplices define a valid
      triangulation.
    - ```backend``` (string, optional, default="cgal"): Specifies the backend
      used to compute the triangulation.  The available options are "qhull",
      "cgal", and "topcom". CGAL is the default one as it is very
      fast and robust.

    **Example:**
    We construct a triangulation of a polytope. Since this class is not
    intended to by initialized by the end user, we create it via the
    [```triangulate```](./polytope#triangulate) function of the
    [Polytope](./polytope) class. In this example the polytope is reflexive, so
    by default the triangulation is fine, regular, and star.
    ```python {3}
    from cytools import Polytope
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    t = p.triangulate()
    t
    # Prints: A fine, regular, star triangulation of a 4-dimensional polytope in ZZ^4
    ```
    """

    def __init__(self, triang_pts, poly=None, heights=None, make_star=False,
                 simplices=None, check_input_simplices=True, backend="cgal"):
        """
        **Description:**
        Initializes a ```Triangulation``` object.

        **Arguments:**
        - ```triang_pts``` (list): The list of points to be triangulated.
        - ```poly``` (Polytope, optional): The ambient polytope of the points
          to be triangulated. If not specified, it is constructed as the convex
          hull of the given points.
        - ```heights``` (list, optional): A list of heights specifying the
          regular triangulation. When not secified, it will return the Delaunay
          triangulation when using CGAL, a triangulation obtained from random
          heights near the Delaunay when using QHull, or the placing
          triangulation when using TOPCOM. Heights can only be specified when
          using CGAL or QHull as the backend.
        - ```make_star``` (boolean, optional, default=False): Indicates whether
          to turn the triangulation into a star triangulation by deleting
          internal lines and connecting all points to the origin, or
          equivalently, by decreasing the height of the origin until it is much
          lower than all other heights.
        - ```simplices``` (list, optional): A list of simplices specifying the
          triangulation. Each simplex is a list of point indices. This is
          useful when a triangulation was previously computed and it needs to
          be used again. Note that the ordering of the points needs to be
          consistent.
        - ```check_input_simplices``` (boolean, optional, default=True): Flag
          that specifies whether to check if the input simplices define a valid
          triangulation.
        - ```backend``` (string, optional, default="cgal"): Specifies the
          backend used to compute the triangulation.  The available options are
          "qhull", "cgal", and "topcom". CGAL is the default one as it is very
          fast and robust.

        **Returns:**
        Nothing.
        """
        self._triang_pts = np.array(triang_pts, dtype=int)
        triang_pts_tup = [tuple(pt) for pt in self._triang_pts]
        heights = copy.deepcopy(heights)
        if poly is None:
            from cytools.polytope import Polytope
            self._poly = Polytope(self._triang_pts)
        else:
            self._poly = poly
        if not self._poly.is_solid():
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
                print("Warning: Unsupported point configuration found. "
                      "Attempting to fix...")
                if set(triang_pts_tup) == set(poly_pts_mpcp):
                    self._all_poly_pts = False
                    ind_dict = {i:poly_pts_mpcp.index(triang_pts_tup[i])
                                for i in range(len(triang_pts_tup))}
                    triang_pts_tup = poly_pts_mpcp
                    self._triang_pts = np.array(triang_pts_tup)
                    if heights is not None:
                        tmp_heights = [0]*len(heights)
                        for i,j in ind_dict.items():
                            tmp_heights[j] = heights[i]
                        heights = tmp_heights
                    if simplices is not None:
                        simplices = [[ind_dict[i] for i in s]
                                        for s in simplices]
                    print("Sucessfully fixed the input.")
                elif set(triang_pts_tup) == set(poly_pts):
                    self._all_poly_pts = True
                    ind_dict = {i:poly_pts.index(triang_pts_tup[i])
                                for i in range(len(triang_pts_tup))}
                    triang_pts_tup = poly_pts
                    self._triang_pts = np.array(triang_pts_tup)
                    if heights is not None:
                        tmp_heights = [0]*len(heights)
                        for i,j in ind_dict.items():
                            tmp_heights[j] = heights[i]
                        heights = tmp_heights
                    if simplices is not None:
                        simplices = [[ind_dict[i] for i in s]
                                        for s in simplices]
                    print("Sucessfully fixed the input.")
                else:
                    self._all_poly_pts = False
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
        # Initialize hidden attributes
        self._hash = None
        self._is_regular = None
        self._is_valid = None
        self._is_fine = None
        self._is_star = None
        self._gkz_phi = None
        self._sr_ideal = None
        self._cpl_cone = [None]*2
        self._fan_cones = dict()
        self._cy = None
        # Now save input triangulation or construct it
        if simplices is not None:
            self._simplices = np.array(sorted([sorted(s) for s in simplices]))
            if self._simplices.shape[1] != self._triang_pts.shape[1]+1:
                raise Exception("Input simplices have wrong dimension.")
            self._heights = None
            if check_input_simplices and not self.is_valid():
                raise Exception("Input simplices do not form a valid "
                                "triangulation.")
        else:
            backends = ["qhull", "cgal", "topcom", None]
            self._is_regular = (None if backend == "qhull" else True)
            self._is_valid = True
            if heights is None:
                # Heights need to be perturbed around the Delaunay heights for
                # QHull or the triangulation might not be regular. If using
                # CGAL then they are not perturbed.
                if backend is None:
                    raise Exception("The simplices of the triangulation must "
                                    "be specified when working without a "
                                    "backend")
                if backend == "qhull":
                    heights = [np.dot(p,p) + np.random.normal(0,0.05)
                                for p in self._triang_pts]
                elif backend == "cgal":
                    heights = [np.dot(p,p) for p in self._triang_pts]
                elif backend == "topcom":
                    heights = None
                else:
                    raise Exception("Invalid backend. "
                                    f"Options are: {backends}.")
            else:
                if len(heights) != len(triang_pts):
                    raise Exception("Lists of heights and points must have the"
                                    " same number of elements.")
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
                self._simplices = cgal_triangulate(self._triang_pts, heights)
            else: # Use TOPCOM
                self._simplices = topcom_triangulate(self._triang_pts)
                if make_star:
                    facets_ind = [[self._pts_dict[tuple(pt)]
                                    for pt in f.boundary_points()]
                                        for f in self._poly.facets()]
                    self._simplices = convert_to_star(self._simplices,
                                                      facets_ind,
                                                      self._origin_index)
        # Make sure that the simplices are sorted
        self._simplices = np.array(
                                sorted([sorted(s) for s in self._simplices]))

    def clear_cache(self, recursive=False):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        - ```recursive``` (boolean, optional, default=False): Whether to also
          clear the cache of the ambient polytope.

        **Returns:**
        Nothing.
        """
        self._hash = None
        self._is_fine = None
        self._is_regular = None
        self._is_star = None
        self._is_valid = None
        self._gkz_phi = None
        self._sr_ideal = None
        self._cpl_cone = [None]*2
        self._fan_cones = dict()
        self._cy = None
        if recursive:
            self._poly.clear_cache()

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the triangulation.

        **Arguments:**
        None.

        **Returns:**
        (string) A string describing the triangulation.
        """
        return (f"A {('fine' if self.is_fine() else 'non-fine')}, "
                + (("regular, " if self._is_regular else "irregular, ")
                    if self._is_regular is not None else "") +
                f"{('star' if self.is_star() else 'non-star')} "
                f"triangulation of a {self.dim()}-dimensional "
                f"polytope in ZZ^{self.dim()}")

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of triangulations with ==.

        **Arguments:**
        - ```other``` (Triangulation): The other triangulation that is being
          compared.

        **Returns:**
        (boolean) The truth value of the triangulations being equal.
        """
        if not isinstance(other, Triangulation):
            return NotImplemented
        return (self.polytope() == other.polytope() and
                sorted(self.simplices().tolist())
                    == sorted(other.simplices().tolist()))

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of triangulations with !=.

        **Arguments:**
        - ```other``` (Triangulation): The other triangulation that is being
          compared.

        **Returns:**
        (boolean) The truth value of the triangulations being different.
        """
        if not isinstance(other, Triangulation):
            return NotImplemented
        return not (self.polytope() == other.polytope() and
                    sorted(self.simplices().tolist())
                        == sorted(other.simplices().tolist()))

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from triangulations.

        **Arguments:**
        None.

        **Returns:**
        (integer) The hash value of the triangulation.
        """
        if self._hash is None:
            self._hash = hash((hash(self.polytope()),) +
                            tuple(sorted(tuple(s) for s in self.simplices())))
        return self._hash

    def polytope(self):
        """
        **Description:**
        Returns the polytope being triangulated.

        **Arguments:**
        None.

        **Returns:**
        (Polytope) The ambient polytope.
        """
        return self._poly

    def points(self):
        """
        **Description:**
        Returns the points of the triangulation.

        **Arguments:**
        None.

        **Returns:**
        (list) The points of the triangulation.
        """
        return np.array(self._triang_pts)

    def points_to_indices(self, points):
        """
        **Description:**
        Returns the list of indices corresponding to the given points. It also
        accepts a single point, in which case it returns the corresponding
        index.

        **Arguments:**
        - ```points``` (list): A list of points.

        **Returns:**
        (list or integer) The list of indices corresponding to the given
        points. Or the index of the point if only one is given.
        """
        if len(np.array(points).shape) == 1:
            if np.array(points).shape[0] == 0:
                return np.zeros(0, dtype=int)
            return self._pts_dict[tuple(points)]
        return np.array([self._pts_dict[tuple(pt)] for pt in points])

    def simplices(self):
        """
        **Description:**
        Returns the simplices of the triangulation.

        **Arguments:**
        None.

        **Returns:**
        (list) The simplices of the triangulation.
        """
        return np.array(self._simplices)

    def dim(self):
        """
        **Description:**
        Returns the dimension of the triangulation.

        **Arguments:**
        None.

        **Returns:**
        (integer) The dimension of the triangulation.
        """
        return self._poly.dim()

    def is_fine(self):
        """
        **Description:**
        Returns True if the triangulation is fine (all the points are used),
        and False otherwise.

        **Arguments:**
        None.

        **Returns:**
        (boolean) The truth value of the triangulation being fine.
        """
        if self._is_fine is not None:
            return self._is_fine
        self._is_fine = (len(set.union(*[set(s) for s in self._simplices]))
                        == len(self._triang_pts))
        return self._is_fine

    def is_regular(self, backend=None):
        """
        **Description:**
        Returns True if the triangulation is regular and False otherwise.

        **Arguments:**
        - ```backend``` (string, optional): The optimizer used for the
          computation. The available options are the backends of the
          [```is_solid```](./cone#is_solid) function of the
          [```Cone```](./cone) class. If not specified, it will be picked
          automatically.

        **Returns:**
        (boolean) The truth value of the triangulation being regular.
        """
        if self._is_regular is not None:
            return self._is_regular
        self._is_regular = (True if self.simplices().shape[0] == 1 else
                            self.cpl_cone(
                                exclude_points_not_in_triangulation=True
                                ).is_solid(backend=backend))
        return self._is_regular

    def is_star(self, star_origin=0):
        """
        **Description:**
        Returns True if the triangulation is star and False otherwise.

        **Arguments:**
        - ```star_origin``` (integer, optional, default=0): The index of the
          origin of the star triangulation

        **Returns:**
        (boolean) The truth value of the triangulation being star.
        """
        if self._is_star is not None:
            return self._is_star
        self._is_star = all(star_origin in s for s in self._simplices)
        return self._is_star

    def is_valid(self):
        """
        **Description:**
        Returns True if the presumed triangulation meets all requirements to be
        a triangulation. The simplices must cover the full volume of the
        convex hull, and they cannot intersect at full-dimensional regions.

        **Arguments:**
        None.

        **Returns:**
        (boolean) The truth value of the triangulation being valid.
        """
        if self._is_valid is not None:
            return self._is_valid
        simps = self.simplices()
        pts = self.points()
        pts_ext = np.empty((pts.shape[0],pts.shape[1]+1), dtype=int)
        pts_ext[:,:-1] = pts
        pts_ext[:,-1] = 1
        # Check if the dimensions of the simplices and polytope match up
        if simps.shape[1] != self.dim()+1:
            self._is_valid = False
            return self._is_valid
        # If the triangulation is presumably regular, then we can check if
        # heights inside the CPL cone yield the same triangulation.
        if self.is_regular():
            cpl = self.cpl_cone()
            heights = cpl.tip_of_stretched_cone(0.1)[1]
            tmp_triang = Triangulation(self.points(), self.polytope(),
                                        heights=heights, make_star=False)
            self._is_valid = (
                    sorted(sorted(s) for s in self.simplices().tolist()) ==
                    sorted(sorted(s) for s in tmp_triang.simplices().tolist()))
            return self._is_valid
        # If it is not regular, then we check this using the definition of a
        # triangulation. This can be quite slow for large polytopes.
        # We first check if the volumes add up to the volume of the polytope.
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
        # Finally, check if simplices have full-dimensional intersections.
        pts_ext = pts_ext.tolist()
        for i,s1 in enumerate(simps):
            for s2 in simps[i+1:]:
                inters = Cone(s1).intersection(Cone(s2))
                if inters.is_solid():
                    self._is_valid = False
                    return self._is_valid
        self._is_valid = True
        return self._is_valid

    def gkz_phi(self):
        """
        **Description:**
        Returns the GKZ phi vector of the triangulation.

        **Arguments:**
        None.

        **Returns:**
        (list) The GKZ phi vector of the triangulation.
        """
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

    def random_flips(self, N, only_fine=None, only_regular=None,
                     only_star=None, backend=None):
        """
        **Description:**
        Returns a triangulation obtained by performing N random bistellar
        flips. The computation is performed with a modified version of TOPCOM.
        There is the option of limiting the flips to fine, regular, and star
        triangulations. An additional backend is used to check regularity, as
        checking this with TOPCOM is very slow for large polytopes.

        **Arguments:**
        - ```N``` (integer): The number of bistellar flips to perform.
        - ```only_fine``` (boolean, optional): Restricts to flips to fine
          triangulations. If not specified, it is set to True if the
          triangulation is fine, and False otherwise.
        - ```only_regular``` (boolean, optional): Restricts the flips to
          regular triangulations. If not specified, it is set to True if the
          triangulation is regular, and False otherwise.
        - ```only_star``` (boolean, optional): Restricts the flips to star
          triangulations. If not specified, it is set to True if the
          triangulation is star, and False otherwise.
        - ```backend``` (string, optional, default=None): The backend used to
          check regularity. The options are any backend available for the
          [```is_solid```](./cone#is_solid) function of the
          [```Cone```](./cone) class.

        **Returns:**
        (Triangulation) A new triangulation obtained by performing N random
        flips.
        """
        current_triang = self
        for n in range(N):
            neighbors = current_triang.neighbor_triangulations()
            np.random.shuffle(neighbors)
            good_pick = False
            for t in neighbors:
                if only_fine and not t.is_fine():
                    continue
                if only_star and not t.is_star():
                    continue
                if only_regular and not t.is_regular():
                    continue
                good_pick = True
                current_triang = t
                break
            if not good_pick:
                raise Exception("There was an error in the random walk.")
        return current_triang

    def neighbor_triangulations(self, only_fine=False, only_regular=False,
                         only_star=False, backend=None):
        """
        **Description:**
        Returns the list of triangulations that differ by one bistellar flip
        from the current triangulation. The computation is performed with a
        modified version of TOPCOM. There is the option of limiting the flips
        to fine, regular, and star triangulations. An additional backend is
        used to check regularity, as checking this with TOPCOM is very slow for
        large polytopes.

        **Arguments:**
        - ```only_fine``` (boolean, optional, default=False): Restricts to fine
          triangulations.
        - ```only_regular``` (boolean, optional, default=False): Restricts the
          to regular triangulations.
        - ```only_star``` (boolean, optional, default=False): Restricts to star
          triangulations.
        - ```backend``` (string, optional, default=None): The backend used to
          check regularity. The options are any backend available for the
          [```is_solid```](./cone#is_solid) function of the
          [```Cone```](./cone) class.

        **Returns:**
        (list) The list of triangulations that differ by one bistellar flip
        from the current triangulation.
        """
        pts_str = str([list(pt)+[1] for pt in self._triang_pts])
        triang_str = str([list(s) for s in self._simplices]
                         ).replace("[","{").replace("]","}")
        flips_str = "(-1)"
        topcom_input = pts_str + "[]" + triang_str + flips_str
        topcom_bin = config.topcom_path + "topcom-randomWalk"
        topcom = subprocess.Popen((topcom_bin,), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        topcom_res, topcom_err = topcom.communicate(input=topcom_input)
        if len(topcom_res)==0:
            return []
        triangs_list = [literal_eval(r) for r in topcom_res.strip().split("\n")]
        triangs = []
        for t in triangs_list:
            tri = Triangulation(self._triang_pts, self._poly, simplices=t,
                                check_input_simplices=False)
            if only_fine and not tri.is_fine():
                continue
            if only_star and not tri.is_star():
                continue
            if only_regular and not tri.is_regular():
                continue
            triangs.append(tri)
        return triangs

    def sr_ideal(self):
        """
        **Description:**
        Returns the Stanley-Reisner ideal if the triangulation is star.

        **Arguments:**
        None.

        **Returns:**
        (list) The Stanley-Reisner ideal of the triangulation.
        """
        if self._sr_ideal is not None:
            return copy.deepcopy(self._sr_ideal)
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
                                                        self.dim() + 1))
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
        return copy.deepcopy(self._sr_ideal)

    def cpl_cone(self, backend=None,
                 exclude_points_not_in_triangulation=False):
        """
        **Description:**
        Computes the cone of strictly convex piecewise linear functions
        defining the triangulation. It is computed by finding the defining
        hyperplane equations. The triangulation is regular if and only if this
        cone is solid (i.e. full-dimensional).

        **Arguments:**
        - ```backend``` (string, optional): Specifies how the cone is computed.
          Options are "native", which uses a native implementation of an
          algorithm by Berglund, Katz and Klemm, or "topcom" which uses
          differences of GKZ vectors for the computation.
        - ```exclude_points_not_in_triangulation``` (boolean, optional,
          default=False): This flag allows the exclusion of points that are
          not part of the triangulation. This can be done to check regularity
          faster, but this cannot be used if the actual cone in the secondary
          fan is needed.

        **Returns:**
        (Cone) The CPL cone.
        """
        backends = (None, "native", "topcom")
        if backend not in backends:
            raise Exception(f"Options for backend are: {backends}")
        if backend is None:
            backend = ("native" if self.is_fine()
                                    or exclude_points_not_in_triangulation
                                else "topcom")
        if (backend == "native" and not self.is_fine()
                and not exclude_points_not_in_triangulation):
            print("Warning: Native backend is not supported when not excluding"
                  "points that are not in the triangulation. Using TOPCOM...")
            backend = "topcom"
        args_id = 1*exclude_points_not_in_triangulation
        if self._cpl_cone[args_id] is not None:
            return self._cpl_cone[args_id]
        if backend == "native":
            pts_ext = [tuple(pt) + (1,) for pt in self._triang_pts]
            dim = self.dim()
            simps = [set(s) for s in self._simplices]
            m = np.zeros((dim+1, dim+2), dtype=int)
            null_vecs = set()
            for i,s1 in enumerate(simps):
                for s2 in simps[i+1:]:
                    if len(s1 & s2) != dim:
                        continue
                    diff_pts = list(s1 ^ s2)
                    comm_pts = list(s1 & s2)
                    for j,pt in enumerate(diff_pts):
                        m[:,j] = pts_ext[pt]
                    for j,pt in enumerate(comm_pts):
                        m[:,j+2] = pts_ext[pt]
                    v = np.array(fmpz_mat(m.tolist()).nullspace()[0]
                                    .transpose().tolist()[0], dtype=int)
                    if v[0] < 0:
                        v = -v
                    # Reduce the vector
                    g = gcd_list(v)
                    if g != 1:
                        v //= g
                    # Construct the full vector (including all points)
                    full_v = np.zeros(len(pts_ext), dtype=int)
                    for i,pt in enumerate(diff_pts):
                        full_v[pt] = v[i]
                    for i,pt in enumerate(comm_pts):
                        full_v[pt] = v[i+2]
                    full_v = tuple(full_v)
                    if full_v not in null_vecs:
                        null_vecs.add(full_v)
            self._cpl_cone[args_id] = Cone(hyperplanes=list(null_vecs),
                                           check=False)
            return self._cpl_cone[args_id]
        # Otherwise we compute this cone by using differences of GKZ vectors.
        gkz_phi = self.gkz_phi()
        diffs = []
        for t in self.neighbor_triangulations(only_fine=False,
                                              only_regular=False,
                                              only_star=False):
            diffs.append(t.gkz_phi()-gkz_phi)
        self._cpl_cone[args_id] = Cone(hyperplanes=diffs)
        return self._cpl_cone[args_id]

    def get_cy(self):
        """
        **Description:**
        Returns a CalabiYau object corresponding to the anti-canonical
        hypersurface on the toric variety defined by the fine, star, regular
        triangulation.

        :::note
        Only Calabi-Yau 3-fold hypersurfaces are fully supported. Other
        dimensions require enabling the experimetal features of CYTools in the
        [configuration](./configuration).
        :::

        **Arguments:**
        None.

        **Returns:**
        (CalabiYau) The Calabi-Yau arising from the triangulation.
        """
        if self._cy is not None:
            return self._cy
        if not self._allow_cy:
            raise Exception("There is a problem with the data of the "
                            "triangulation. Constructing a CY is disallowed "
                            "since computations might produce wrong results.")
        if not self._poly.is_favorable(lattice="N"):
            raise Exception("Only favorable CYs are currently supported.")
        if not self.is_star() or not self.is_fine():
            raise Exception("Triangulation is non-fine or non-star.")
        if self._poly.dim() != 4 and not config.enable_experimental_features:
            raise Exception("Constructing Calabi-Yaus of dimensions other "
                            "than 3 is experimental and must be enabled in "
                            "the configuration.")
        self._cy = CalabiYau(self)
        return self._cy

    def fan_cones(self, d=None, face_dim=None):
        """
        **Description:**
        It returns the cones forming a fan defined by a star triangulation of a
        reflexive polytope. The dimension of the desired cones can be
        specified, and one can also restrict to cones that lie in faces of a
        particular dimension.

        **Arguments:**
        - ```d``` (integer, optional): The dimension of the desired cones. If
          not specified, it returns the full-dimensional cones.
        - ```face_dim``` (integer, optional): Restricts to cones that lie on
          faces of the polytope of a particular dimension. If not specified,
          then no restriction is imposed.

        **Returns:**
        (list) The list of cones with the specified properties defined by the
        star triangulation.
        """
        if d is None:
            d = (self.dim() if face_dim is None else face_dim)
        if d not in range(1,self.dim()+1):
            raise Exception("Only cones of dimension 1 through d are "
                            "supported.")
        if (d,face_dim) in self._fan_cones:
            return self._fan_cones[(d,face_dim)]
        if not self.is_star() or not self._poly.is_reflexive():
            raise Exception("Cones can only be obtained from star "
                            "triangulations of reflexive polytopes.")
        pts = self.points()
        cones = set()
        faces = ([self.points_to_indices(f.points())
                        for f in self._poly.faces(face_dim)]
                 if face_dim is not None else None)
        for s in self.simplices():
            for c in combinations(s,d):
                if (0 not in c
                    and (faces is None
                         or any(all(cc in f for cc in c) for f in faces))):
                    cones.add(tuple(sorted(c)))
        self._fan_cones[(d,face_dim)] = [Cone(pts[list(c)]) for c in cones]
        return self._fan_cones[(d,face_dim)]


def convert_to_star(simplices, facets, star_origin):
    """
    **Description:**
    Turns a triangulation into a star triangulation by deleting internal lines
    and connecting all points to the origin.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [Triangulation](./triangulation) class when needed.
    :::

    :::important
    This function is only reliable for triangulations of reflexive polytopes
    and may produce invalid tiangulations for other polytopes.
    :::

    **Arguments:**
    - ```simplices``` (list): The list of simplices of the triangulation. Each
      simplex consists of the list of indices of the points forming its
      vertices.
    - ```facets``` (list): The list of facets of the polytope. Each facet
      consists of the indices of the points in the facet.
    - ```star_origin``` (integer): The index of the point that is used as the
      star origin.

    **Returns:**
    (list) A list of simplices forming a star triangulation.
    """
    star_triang = []
    triang = np.array(simplices)
    dim = triang.shape[1] - 1
    for facet in facets:
        for simp in triang:
            overlap = simp[np.isin(simp, facet)].tolist()
            if len(overlap) == dim:
                star_triang.append([star_origin] + overlap)
    return np.array(sorted([sorted(s) for s in star_triang]))


def qhull_triangulate(points, heights):
    """
    **Description:**
    Computes a regular triangulation using QHull.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [Triangulation](./triangulation) class when using QHull as the
    backend.
    :::

    **Arguments:**
    - ```points``` (list): A list of points.
    - ```heights``` (list): A list of heights defining the regular
      triangulation.

    **Returns:**
    (list) A list of simplices defining a regular triangulation.
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
            if int(round(np.linalg.det([lifted_points[i] for i in s]))) != 0]
    return np.array(sorted([sorted(s) for s in simp]))


def cgal_triangulate(points, heights):
    """
    **Description:**
    Computes a regular triangulation using CGAL.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [Triangulation](./triangulation) class when using CGAL as the
    backend.
    :::

    **Arguments:**
    - ```points``` (list): A list of points.
    - ```heights``` (list): A list of heights defining the regular
      triangulation.

    **Returns:**
    (list) A list of simplices defining a regular triangulation.
    """
    dim = points.shape[1]
    if dim > 10:
        raise Exception("CGAL code is only compiled up to d=10.")
    cgal_bin = config.cgal_path + f"/cgal-triangulate-{dim}d"
    cgal = subprocess.Popen((cgal_bin,), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    pts_str = str([list(pt) for pt in points])
    heights_str = str(list(heights)).replace("[", "(").replace("]", ")")
    cgal_res, cgal_err = cgal.communicate(input=pts_str+heights_str)
    if cgal_err != "":
        raise Exception(f"CGAL error: {cgal_err}")
    try:
        simp = literal_eval(cgal_res)
    except:
        raise Exception("Error: Failed to parse CGAL output.")
    return np.array(sorted([sorted(s) for s in simp]))


def topcom_triangulate(points):
    """
    **Description:**
    Computes the placing/pushing triangulation using TOPCOM.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [Triangulation](./triangulation) class when using TOPCOM as the
    backend.
    :::

    **Arguments:**
    - ```points``` (list): A list of points.

    **Returns:**
    (list) A list of simplices defining a triangulation.
    """
    topcom_bin = config.topcom_path + "/topcom-points2finetriang"
    topcom = subprocess.Popen((topcom_bin, "--regular"), stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
    pts_str = str([list(pt)+[1] for pt in points])
    topcom_res, topcom_err = topcom.communicate(input=pts_str+"[]")
    try:
        simp = literal_eval(topcom_res.replace("{", "[").replace("}", "]"))
    except:
        raise Exception("Error: Failed to parse TOPCOM output. "
                        f"\nstdout: {topcom_res} \nstderr: {topcom_err}")
    return np.array(sorted([sorted(s) for s in simp]))


def all_triangulations(points, only_fine=False, only_regular=False,
                       only_star=False, star_origin=None, backend=None,
                       poly=None):
    """
    **Description:**
    Computes all triangulations of the input point configuration using TOPCOM.
    There is the option to only compute fire, regular or fine triangulations.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [all_triangulations](./polytope#all_triangulations) function of
    the [Polytope](./polytope) class.
    :::

    **Arguments:**
    - ```points``` (list): The list of points to be triangulated.
    - ```only_fine``` (boolean, optional, default=False): Restricts to fine
      triangulations.
    - ```only_regular``` (boolean, optional, default=False): Restricts to
      regular triangulations.
    - ```only_star``` (boolean, optional, default=False): Restricts to star
      triangulations.
    - ```star_origin``` (int, optional): The index of the point used as the
      star origin. It needs to be specified if only_star=True.
    - ```backend``` (string, optional): The optimizer used to check
      regularity computation. The available options are the backends of the
      [```is_solid```](./cone#is_solid) function of the
      [```Cone```](./cone) class. If not specified, it will be picked
      automatically. Note that TOPCOM is not used to check regularity since
      it is slower.
    - ```poly``` (Polytope, optional): The ambient polytope. It is constructed
      if not specified.

    **Returns:**
    (generator) a generator of [Triangulation](./triangulation) objects with
    the specified properties.
    """
    if only_star and star_origin is None:
        raise Exception("The star_origin parameter must be specified when "
                        "restricting to star triangulations.")
    if poly is None:
        from cytools.polytope import Polytope
        poly = Polytope(points)
    topcom_bin = (config.topcom_path
                  + ("topcom-points2finetriangs" if only_fine
                     else "topcom-points2triangs"))
    topcom = subprocess.Popen(
                        (topcom_bin,),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True)
    pts_str = str([list(pt)+[1] for pt in points])
    topcom_res, topcom_err = topcom.communicate(input=pts_str+"[]")
    try:
        triangs = [literal_eval("["+ t.replace("{","[").replace("}","]") + "]")
                    for t in re.findall(r"\{([^\:]*)\}", topcom_res)]
    except:
        raise Exception("Error: Failed to parse TOPCOM output. "
                        f"\nstdout: {topcom_res} \nstderr: {topcom_err}")
    srt_triangs = [np.array(sorted([sorted(s) for s in t])) for t in triangs
                    if (not only_star or all(star_origin in ss for ss in t))]
    for t in srt_triangs:
        tri = Triangulation(points, poly=poly, simplices=t, make_star=False,
                            check_input_simplices=False)
        if not only_regular or tri.is_regular():
            yield tri


def random_triangulations_fast_generator(triang_pts, N=None, c=0.2,
                            max_retries=500, make_star=False, only_fine=True,
                            backend="cgal", poly=None):
    """
    Constructs pseudorandom regular (optionally fine and star) triangulations
    of a given point set. This is done by picking random heights around the
    Delaunay heights from a Gaussian distribution.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the
    [random_triangulations_fast](./polytope#random_triangulations_fast)
    function of the [Polytope](./polytope) class.
    :::

    :::caution important
    This function produces random triangulations very quickly, but it does not
    produce a fair sample. When a fair sampling is required the
    [random_triangulations_fair](./polytope#random_triangulations_fair)
    function should be used.
    :::

    **Arguments:**
    - ```triang_pts``` (list): The list of points to be triangulated.
    - ```N``` (integer, optional): Number of desired unique triangulations. If
      not specified, it will generate as many triangulations as it can find
      until it has to retry more than max_retries times to obtain a new
      triangulation.
    - ```c``` (float, optional, default=0.2): A contant used as the standard
      deviation of the Gaussian distribution used to pick the heights. A
      larger c results in a wider range of possible triangulations, but with a
      larger fraction of them being non-fine, which slows down the process when
      using only_fine=True.
    - ```max_retries``` (integer, optional, default=50): Maximum number of
      attempts to obtain a new triangulation before the process is terminated.
    - ```make_star``` (boolean, optional, default=False): Converts the obtained
      triangulations into star triangulations.
    - ```only_fine``` (boolean, optional, default=True): Restricts to fine
      triangulations.
    - ```backend``` (string, optional, default="cgal"): Specifies the
      backend used to compute the triangulation. The available options are
      "cgal" and "qhull".

    **Returns:**
    (generator) A generator of [```Triangulation```](./triangulation) objects
    with the specified properties.
    """
    triang_pts = np.array(triang_pts)
    triang_hashes = set()
    n_retries = 0
    while True:
        if (n_retries >= max_retries
                or (N is not None and len(triang_hashes) >= N)):
            break
        heights= [pt.dot(pt) + np.random.normal(0,c) for pt in triang_pts]
        t = Triangulation(triang_pts, poly=poly, heights=heights,
                          make_star=make_star, backend=backend)
        if only_fine and not t.is_fine():
            n_retries += 1
            continue
        h = hash(t)
        if h in triang_hashes:
            n_retries += 1
            continue
        triang_hashes.add(h)
        n_retries = 0
        yield t


def random_triangulations_fair_generator(triang_pts, N=None, n_walk=10, n_flip=10,
                    initial_walk_steps=20, walk_step_size=1e-2,
                    max_steps_to_wall=10, fine_tune_steps=8, max_retries=50,
                    make_star=False, backend="cgal", poly=None):
    """
    **Description:**
    Returns a pseudorandom list of regular triangulations of a given point set.
    Implements Algorithm \#3 from the paper
    *Bounding the Kreuzer-Skarke Landscape*
    by Mehmet Demirtas, Liam McAllister, and Andres Rios-Tascon.
    [arXiv:2008.01730](https://arxiv.org/abs/2008.01730)

    This is a Markov chain Monte Carlo algorithm that involves taking random
    walks inside the subset of the secondary fan corresponding to fine
    triangulations and performing random flips. For details, please see
    Section 4.1 in the paper.

    :::note notes
    - This function is not intended to be called by the end user. Instead, it
      is used by the
      [random_triangulations_fast](./polytope#random_triangulations_fast)
      function of the [Polytope](./polytope) class.
    - This function is designed mainly for large polytopes where sampling
      triangulations is challenging. When small polytopes are used it is likely
      to get stuck.
    :::

    **Arguments:**
    - ```triang_pts``` (list): The list of points to be triangulated.
    - ```N``` (integer, optional): Number of desired unique triangulations. If
      not specified, it will generate as many triangulations as it can find
      until it has to retry more than max_retries times to obtain a new
      triangulation.
    - ```n_walk``` (integer, optional, default=10): Number of hit-and-run steps
      per triangulation.
    - ```n_flip``` (integer, optional, default=10): Number of random flips
      performed per triangulation.
    - ```initial_walk_steps``` (integer, optional, default=20): Number of
      hit-and-run steps to take before starting to record triangulations. Small
      values may result in a bias towards Delaunay-like triangulations.
    - ```walk_step_size``` (float, optional, default=1e-2): Determines size of
      random steps taken in the secondary fan. Algorithm may stall if too
      small.
    - ```max_steps_to_wall``` (integer, optional, default=10): Maximum number
      of steps to take towards a wall of the subset of the secondary fan that
      correspond to fine triangulations. If a wall is not found, a new random
      direction is selected. Setting this to be very large (>100) reduces
      performance. If this, or walk_step_size, is set to be too low, the
      algorithm may stall.
    - ```fine_tune_steps``` (integer, optional, default=8): Number of steps to
      determine the location of a wall. Decreasing improves performance, but
      might result in biased samples.
    - ```max_retries``` (integer, optional, default=50): Maximum number of
      attempts to obtain a new triangulation before the process is terminated.
    - ```make_star``` (boolean, optional, default=False): Converts the obtained
      triangulations into star triangulations.
    - ```backend``` (string, optional, default="cgal"): Specifies the
      backend used to compute the triangulation. The available options are
      "cgal" and "qhull".

    **Returns:**
    (generator) A generator of [```Triangulation```](./triangulation) objects
    with the specified properties.
    """
    triang_hashes = set()
    triang_pts = np.array(triang_pts)
    num_points = len(triang_pts)
    n_retries = 0

    # Obtain a random Delaunay triangulation by picking a random point as the
    # origin.
    rand_ind = np.random.randint(0,len(triang_pts))
    points_shifted = [p-triang_pts[rand_ind] for p in triang_pts]
    delaunay_heights = [walk_step_size*(np.dot(p,p)) for p in points_shifted]
    start_pt = delaunay_heights
    step_size = walk_step_size*np.mean(delaunay_heights)
    old_pt = start_pt
    # Pick a random direction, and move until a non-fine
    # or non-star triangulation is found.
    step_ctr = 0 # Total number of random walk steps taken.
    step_per_tri_ctr = 0 # Number of random walk steps
                         # taken for the given triangulation.
    while True:
        if n_retries>=max_retries or (N is not None and len(triang_hashes)>N):
            break
        outside_bounds = False
        n_retries_out = 0
        while not outside_bounds and n_retries_out < max_retries:
            n_retries_out += 1
            in_pt = old_pt
            random_dir = np.random.normal(size=num_points)
            random_dir = random_dir / np.linalg.norm(random_dir)
            steps_to_wall = 0 # Number of steps taken towards a wall
            while not outside_bounds and steps_to_wall < max_steps_to_wall:
                new_pt = in_pt + random_dir*step_size
                temp_tri = Triangulation(triang_pts, poly=poly, heights=new_pt,
                            make_star=False, backend=backend)
                if temp_tri.is_fine():
                    in_pt = new_pt
                else:
                    out_pt = new_pt
                    outside_bounds = True
                steps_to_wall += 1
        if not outside_bounds:
            print("Couldn't find wall.")
            break # break loop it it can't find any new wall after max_retries
        # Find the location of the boundary
        fine_tune_ctr = 0
        in_pt_found = False
        while fine_tune_ctr < fine_tune_steps or not in_pt_found:
            new_pt = (in_pt + out_pt)/2
            temp_tri = Triangulation(triang_pts, poly=poly, heights=new_pt,
                            make_star=False, backend=backend)
            if temp_tri.is_fine():
                in_pt = new_pt
                in_pt_found = True
            else:
                out_pt = new_pt
            fine_tune_ctr += 1
        # Take a random walk step
        in_pt = in_pt/np.linalg.norm(in_pt)
        random_coef = np.random.uniform(0,1)
        new_pt = (random_coef*np.array(old_pt)
                 + (1-random_coef)*np.array(in_pt))
        # After initial walk steps are done and n_walk steps
        # are taken, move on to random flips
        if step_ctr > initial_walk_steps and step_per_tri_ctr >= n_walk:
            flip_seed_tri = Triangulation(triang_pts, poly=poly, heights=new_pt,
                                make_star=make_star, backend=backend)
            if n_flip > 0:
                temp_tri = flip_seed_tri.random_flips(n_flip, only_fine=True,
                                                      only_regular=True,
                                                      only_star=True)
            else:
                temp_tri = flip_seed_tri
            # Random walks and random flips complete. Record triangulation.
            h = hash(temp_tri)
            if h in triang_hashes:
                n_retries += 1
                continue
            triang_hashes.add(h)
            n_retries = 0
            step_per_tri_ctr = 0
            yield temp_tri

        step_ctr += 1
        step_per_tri_ctr += 1
        old_pt = new_pt/np.linalg.norm(new_pt)
