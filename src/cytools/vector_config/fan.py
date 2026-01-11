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
# Description:  This module contains tools designed to describe vector
#               configurations and their triangulations.
# -----------------------------------------------------------------------------

# external imports
import collections
import itertools
from numba import njit
import numpy as np
import regfans
from typing import Union

# core CYTools imports
from cytools import Cone, utils
from cytools.triangulation import Triangulation


class Fan(regfans.fan.Fan):
    """
    This class handles definition/operations on fans. It is analogous to the
    Triangulation class. It also has toric methods, like intersection numbers.

    **Description:**
    Constructs a `Fan` object describing a fan over a lattice vector
    configuration. This is handled by the hidden [`__init__`](#__init__)
    function.

    NOTE: This is not intended to be called directly. This is instead meant to
    be constructed from `VectorConfiguration.subdivide`.

    *Arguments:**
    - `vc`:      The ambient vector configuration that this fan is over.
    - `cones`:   The cones defining the fan. Each cone is a collection of
                 integer labels.
    - `heights`: The heights defining the fan, if it is regular. Can be
                 computed later.

    **Returns:**
    Nothing.
    """
    # read from regfans type
    @classmethod
    def from_regfans(cls, fan: "regfans.Fan") -> "Fan":
        """
        **Description:**
        Convert a `regfans.Fans` object to a object of the CYTools Fan class.
        This is used when calling the CYTools VectorConfiguration.subdivide
        method.

        **Arguments:**
        - `fan`: The `regfans.Fan` object.

        **Returns:**
        The CYTools Fan object.
        """
        obj = cls.__new__(cls)            # allocate new instance
        obj.__dict__ = fan.__dict__.copy()  # copy state
        return obj

    # cones/simplices
    # ---------------
    def cones(self,
        formal: bool = False,
        as_inds: bool = False,
        ind_offset: int = 0) -> Union[ tuple[tuple[int]], list["Cone"] ]:
        """
        **Description:**
        Returns the cones in the fan, each cone specified either by
            - (default) a tuple of labels
            - (formal=True) as a formal Cone object.
            - (as_inds=True) as a tuple of indices

        **Arguments:**
        - `formal`:     Whether to return the cones as formal Cone objects.
        - `as_inds`:    Whether to return the cones as indices (not labels).
        - `ind_offset`: Additive offset to the indices

        **Returns:**
        The full-dimensional cones in the fan.
        """
        if formal:
            return tuple([self.vc.cone(c) for c in self._cones])
        else:
            if as_inds:
                cones = tuple([tuple([self.vc.label_to_ind(i)+ind_offset for \
                                        i in simp]) for simp in self._cones])
            else:
                cones = self._cones

            return cones

    # aliases
    simplices = cones
    simps = cones


    def restricted_simps(self,
        to_dim: int = 2,
        padded: bool = False,
        as_face_inds: bool = False):
        """
        **Description:**
        Restrict input simplices to the to_dim-faces of the underlying polytope.

        **Arguments:**
        - `to_dim`:       The dimension of the faces to restrict to.
        - `padded`:       Whether to pad simplices that have <=to_dim points in
                          the restriction
        - `as_face_inds`: Whether to return the simplices as face-inds, not
                          labels.

        **Returns:**
        The restricted simplices
        """
        p = self.vc.conv()
        simps = self.simps()

        # restrict to each to_dim-face
        restricted = []
        for face in p.faces(to_dim):
            # get the face labels
            face_labels = frozenset(face.labels)

            # restrict the simplices
            face_simps = {face_labels.intersection(simp) for simp in simps}

            # sort by decreasing length
            face_simps = sorted(face_simps, key=lambda x: len(x), reverse=True)

            # only keep sets that aren't subsets of others
            face_simps_reduced = []
            for simp in face_simps:
                for simp_test in face_simps_reduced:
                    if simp.issubset(simp_test):
                        break
                else:
                    if len(simp) > 1:
                        face_simps_reduced.append(simp)

            # map to face indices
            if as_face_inds:
                face_simps_reduced = [
                    [face.labels.index(i) for i in simp] for simp in\
                                                            face_simps_reduced
                ]

            # pad the simplices
            if padded:
                for i, simp in enumerate(face_simps_reduced):
                    if len(simp) == 2:
                        face_simps_reduced[i] = simp + [simp[-1]]

            # save it!
            face_simps_reduced = sorted([sorted(simp) for simp in\
                                                            face_simps_reduced])
            restricted.append(face_simps_reduced)

        return restricted

    # VC <-> PC
    # ---------
    def get_pc_triangulation(self,
        check: bool = True,
        verbosity: int = 0) -> "Triangulation":
        """
        **Description:**
        If the fan respects the point configuration, get the associated point
        configuration triangulation.

        **Arguments:**
        - `check`:     Whether the fan respects the point configuration.
        - `verbosity`: The verbosity level. Higher is more verbose.

        **Returns:**
        The restricted simplices
        """
        # check if this makes sense
        if check:
            if not self.respects_ptconfig():
                raise ValueError("Not triangulation of PC")

        # super easy for regular triangulations. Lift by heights >=0, with origin at 0
        if self.is_regular():
            p = self.vc.conv(which=self.labels)
            pts_in_facets = (self.labels == p.labels[1:])
            h = self.heights()
            if not np.all(h >= 0):
                msg =   "Heights are assumed non-negative here... "
                msg += f"your heights={h}..."
                raise ValueError(msg)
            return p.triangulate(heights=[0]+h.tolist(),
                                 include_points_interior_to_facets=pts_in_facets,
                                 check_heights=check)

        # get/return the triangulation
        p = self.vc.conv(self.labels)

        simps = tuple([(0,) + c for c in self.simps()])
        t = p.triangulate(
            simplices=simps,
            include_points_interior_to_facets=True,
            check_input_simplices=False,
        )

        return t

    triang = get_pc_triangulation

    # reformatting regfans.Fans outputs
    # ---------------------------------
    def secondary_cone(self,
        via_circuits: bool = False,
        project_lineality: bool = False,
        verbosity: int = 0):
        """
        **Description:**
        Compute the hyperplanes of the secondary cone associated to this fan.
        This cone has the interpretation:
            for a regular fan, a height h generates the fan iff it is in the
            relative interior of the secondary cone.

        Irregular fans do not have heights generating them and thus do not have
        secondary cones. One way to check regularity of a simplicial fan (i.e.,
        a triangulation) is to attempt to construct the secondary cone. This
        should be solid (i.e., full-dimensional). If the output cone is
        non-solid, then the fan is irregular.

        IRREGULARITY CHECKING ONLY WORKS IF `via_circuits=False`. WHEN
        ATTEMPTING TO COMPUTE THE SECONDARY CONE OF AN IRREGULAR FAN USING
        CIRCUITS, ONE CAN GET A FULL-DIMENSIONAL CONE!!!

        **Arguments:**
        - `via_circuits`:      Whether to use circuits to compute the secondary
                               cone. Should always be correct if the fan is
                               regular but dangerous/not correct for checking
                               irregularity... Alternative is local folding.
        - `project_lineality`: Secondary cones have linear subspaces ('lineality
                               spaces'). These can be projected out without
                               loss of information, giving a cone in the
                               chamber complex.
        - `verbosity`:         The verbosity level. Higher is more verbose.

        **Returns:**
        The secondary/chamber cone.
        """
        H = super().secondary_cone_hyperplanes(
            via_circuits=via_circuits,
            verbosity=verbosity)
        cone = Cone(hyperplanes=H)

        if project_lineality:
            return Cone(rays=cone.rays()@(self.vc.gale()))
        else:
            return cone

    # toric stuff
    # -----------
    def intersection_numbers(
        self,
        pushed_down: bool = False,
        in_basis: bool = False,
        symmetrize: bool = False,
        as_np_array: bool = False,
        eps: float = 1e-4,
        digits: int = 10,
        verbosity: int = 0,
    ) -> Union[dict, "ArrayLike"]:
        """
        **Description:**
        Compute the intersection numbers of the toric variety defined by the
        cones
            cones = [Cone(p.points(which=simp)) for simp in simps]

        **Arguments:**
        - `pushed_down`: Whether to push down the intersection numbers.
        - `in_basis`:    Whether to put the intersection numbers in basis.
        - `symmetrize`:  Whether to give all intersection numbers, using the
                         symmetry of the intersection numbers. Otherwise, just
                         give components kappa[i,j,k] for i<=j<=k.
        - `as_np_array`: Whether to format the intersection numbers as a NumPy
                         array.
        - `eps`:         Tolerance for rejecting 0 intersection numbers.
        - `digits`:      How many digits to round to during the computations.
        - `verbosity`:   The verbosity level. Higher is more verbose.

        **Returns:**
        The intersection numbers.
        """
        if not hasattr(self, '_kappa'):
            self._kappa = collections.defaultdict(float)
            self._kappa_dok = None
            self._kappa_known_labels = set()

        if not self.is_triangulation():
            raise ValueError("Fan is not a triangulation!")

        if as_np_array:
            symmetrize = True

        # get relevant labels
        # -------------------
        relevant_labels = self.used_labels

        # given a basis
        # -------------
        # push down the intersection numbers and represent them in said basis
        if pushed_down or in_basis:
            # get the ambient intersection numbers
            kappa = self.intersection_numbers(symmetrize=False,
                                              digits=None)
            arr_size = self.vc.size

            if pushed_down:
                # push down the intersection numbers
                # (i.e., sum over one index... symmetric so doesn't matter
                #  which)
                # (equiv to taking kappa[i,j,k] = -kappa[0,i,j,k] b/c
                #  anticanonical)
                kappa_pushdown = dict()
                for k,v in kappa.items():
                    if (k[0] == 0) and (0 not in k[1:]):
                        kappa_pushdown[k[1:]] = -v

                kappa = kappa_pushdown

            # write the intersection numbers in the basis
            if in_basis:
                basis = self.vc.divisor_basis

                # ensure basis is list
                basis = list(basis)
                arr_size = len(basis)

                non_basis = [0]+[i for i in self.used_labels if i not in basis]
                kappa = {
                    tuple([basis.index(i) for i in k]): kappa[k]
                    for k in kappa
                    if len(set(k).intersection(non_basis)) == 0
                }

            # symmetrize
            if symmetrize:
                keys = list(kappa.keys())
                for k in keys:
                    for k_perm in itertools.permutations(k):
                        kappa[tuple(k_perm)] = kappa[k]

            # map to numpy array
            if as_np_array:
                kappa_np = np.zeros((self.ambient_dim-pushed_down)*(arr_size,))
                if in_basis:
                    for k,v in kappa.items():
                        kappa_np[k] = v
                else:
                    # map labels to indices...
                    for k,v in kappa.items():
                        kappa_np[*[ki-1 for ki in k]] = v
                return kappa_np
            
            return kappa

        # not given a basis
        # -----------------
        # check if we know the answer
        if (self._kappa_known_labels == set(self.labels)):
            kappa = dict(self._kappa)

            # symmetrize
            if symmetrize:
                keys = list(kappa.keys())
                for k in keys:
                    for k_perm in itertools.permutations(k):
                        kappa[tuple(k_perm)] = kappa[k]

            if as_np_array:
                kappa_np = np.zeros(self.ambient_dim*(self.vc.size+1,))
                for k,v in kappa.items():
                    kappa_np[k] = v
                return kappa_np

            return kappa

        # don't know the answer... need to calculate it...
        # setup
        # -----
        # helpers
        dim = self.dim

        # (formally add the origin to ease indexing)
        vecs  = np.vstack([ np.zeros((1,dim),dtype=int), self.vectors() ])
        simps = self.cones(as_inds=True, ind_offset=1)
        simps_np = np.array(simps)

        # face-to-neighbor map for each codim
        # (a neighbor of a face `f` is any point `i` s.t. $f cup {i}$ is a face)
        # (really, this is an encoding of the face lattice...)
        neighbors = {i: collections.defaultdict(set) for i in range(1, dim)}

        for r in range(1, dim):
            # map each r-dim face to the neighboring points
            for simp in simps:
                for s in itertools.combinations(simp, r):
                    neighbors[r][tuple(sorted(s))].update(tuple(simp))

            # delete the indices corresponding to the face itself
            for face in neighbors[r]:
                neighbors[r][face] -= set(face)

        # compute the intersection numbers
        # --------------------------------
        # for distinct indices, just 1/vol of the cone
        vals = 1/np.abs(np.linalg.det(vecs[simps_np])) # NumPy broadcasting
        self._kappa.update(zip(simps, map(float,vals)))

        if verbosity >= 1:
            msg =  "After computing the intersection numbers associated to "
            msg += "solid cones, we have:"
            print(msg)
            print({k: v for k, v in self._kappa.items()})

        # helpers
        if dim==4:
            def insert_sorted(t, x):
                a, b, c = t
                if x <= a:
                    return (x, a, b, c)
                elif x <= b:
                    return (a, x, b, c)
                elif x <= c:
                    return (a, b, x, c)
                else:
                    return (a, b, c, x)
        else:
            def insert_sorted(t, x):
                return tuple(sorted(t + [x]))

        # for duplicated indices, these can be determined by r-dim faces for
        # dim-1 >= r >= 1
        # (i.e., intersection numbers with duplicated indices)
        for r in range(dim-1, 0, -1): # dim-1, dim-2, ..., 1
            if r == 3:
                kappa_solver = kappa_solve_3x3
            elif r==2:
                kappa_solver = kappa_solve_2x2
            elif r==1:
                kappa_solver = kappa_solve_1x1
            else:
                kappa_solver = lambda pts, known: kappa_solve_nxn(pts, known, r)

            # iterate over each r-face
            for face, neighbs in neighbors[r].items():
                neighbs = sorted(neighbs)
                inds = list(face) + neighbs # analogous to the star of the face
                pts  = vecs[inds].astype(float)

                if len(set(inds)-self._kappa_known_labels) == 0:
                    # already know intersection numbers to these labels... skip
                    continue

                if verbosity >= 1:
                    msg =   "Computing the intersection numbers associated to "
                    msg += f"face = {face}"
                    print(msg)

                # solve for new intersection numbers (with more duplicated
                # indices) by using that `linrels @ kappa_{..i} = 0` for linrels
                # the linear relations of the divisors i
                #
                # the linear relations are given by the coordinates of the rays
                #
                # if we know all intersection numbers with n-duplicates, then
                # we can make a prefactor `pre` with n duplicates so that the
                # above problem is solving `pre+[i]` for `i` iterating over
                # the star
                #
                # this will include cases where i is off the face, so `pre+[i]`
                # will only have n duplicates. This will also include cases
                # where i is on the face, so `pre+[i]` will have n+1 duplicates
                #
                # the former intersection numbers are known at this stage, the
                # latter are not but are deducible via least squares
                for duplicates in itertools.combinations_with_replacement(
                    face, dim - r - 1
                ):
                    # the known intersection numbers will have indices of the
                    # form `face+duplicates+[i]` for i in neighbors
                    prefactor = sorted(face + duplicates)
                    # the insert_sorted and the key lookup are the slowest bits
                    # of what follows
                    known_vals = np.array([ 
                        self._kappa[insert_sorted(prefactor, i)] for i in\
                                                                        neighbs
                        ]).reshape(-1,1)

                    # the unknown intersection numbers will have indices of the
                    # form `face+duplicates+[i]` for i in face

                    # solve for the unknowns
                    # (i.e., find x such that pts.T@kappa_{..., [x,known]} == 0)
                    # (i.e., find x such that pts.T[:,:N_unknown]@kappa_{...,x} == -pts.T[:,N_unknown:]@kappa_{..., known})           
                    sol = kappa_solver(pts, known_vals)

                    # save it
                    for i, val in zip(face, sol):
                        self._kappa[insert_sorted(prefactor, i)] = float(val)

        # intersection numbers w/ 0
        # -------------------------
        kappa_nzeros = [self._kappa]

        for num_zeros in range(1,dim+1):
            # make container for kappa w/ n zeros
            kappa_nzeros.append(collections.defaultdict(float))

            # iterate over values with 1 fewer 0
            for k,v in kappa_nzeros[num_zeros-1].items():

                for kprime in set(itertools.combinations(k[num_zeros-1:],\
                                                            r=dim-num_zeros)):
                    kappa_nzeros[num_zeros][num_zeros*(0,)+kprime] -= v
        
        for kappa_n in kappa_nzeros[1:]:
            self._kappa.update(kappa_n)

        # clean kappa
        # -----------
        # (i.e., trim 0s, round values)
        keys = list(self._kappa.keys())
        for k in keys:
            v = self._kappa[k]

            # delete 0 entries
            if np.abs(v) < eps:
                del self._kappa[k]
            else:
                # round the floating point numbers
                if digits is not None:
                    v_round = round(v, digits)
                else:
                    v_round = v

                # optionally, symmetrize
                if symmetrize:
                    for k_perm in itertools.permutations(k):
                        self._kappa[tuple(k_perm)] = v_round
                else:
                    self._kappa[k] = v_round

        # save kappa labels (for caching)
        self._kappa_known_labels = set(self.labels)

        return self.intersection_numbers(
            pushed_down=pushed_down,
            in_basis=in_basis,
            symmetrize=symmetrize,
            as_np_array=as_np_array,
            eps=eps,
            digits=digits,
            verbosity=0,
        )

    int_nums = intersection_numbers
    kappa    = intersection_numbers

    def c2(self, eps: float = 1e-4, digits: int = 4) -> "ArrayLike":
        """
        **Description:**
        Compute the second chern class associated to the fan.

        **Arguments:**
        - `eps`:    The tolerance for rejecting 0 intersection numbers in the
                    `Fan.intersection_numbers` method.
        - `digits`: The number of digits to use in the intersection number
                    computations.

        **Returns:**
        The second chern class.
        """
        # get the 2-cones
        max_ind = -1
        two_cones = set()
        for s in self.cones():
            max_ind = max(max_ind, max(s))

            for two_cone in itertools.combinations(s, 2):
                two_cones.add(two_cone)

        # get the intersection numbers
        kappa = self.intersection_numbers(
            pushed_down=True, symmetrize=False, eps=eps, digits=4
        )

        # compute c2
        out = []
        for a in range(1, max_ind + 1):
            out.append(
                round(sum(kappa.get(tuple(sorted(c + (a,))), 0) for c in two_cones))
            )

        return out

    def mori_rays(self) -> "ArrayLike":
        """
        **Description:**
        Compute the rays of the Mori cone of the toric variety defined by the
        fan.

        **Arguments:**
        None.

        **Returns:**
        The rays defining the Mori cone.
        """
        intnums = self.intersection_numbers(in_basis=False)
        dim = self.dim
        num_divs = self.vc.gale().shape[0]+1
        
        # COPIED FROM THE ToricVariety CLASS
        curve_dict = collections.defaultdict(lambda: [[], []])
        for ii in intnums:
            if 0 in ii:
                continue
            ctr = collections.Counter(ii)
            if len(ctr) < dim - 1:
                continue
            for comb in set(itertools.combinations(ctr.keys(), dim - 1)):
                crv = tuple(sorted(comb))
                curve_dict[crv][0].append(
                    int(sum([i * (ctr[i] - (i in crv)) for i in ctr]))
                )
                curve_dict[crv][1].append(intnums[ii])
        row_set = set()
        for crv in curve_dict:
            g = utils.gcd_list(curve_dict[crv][1])
            row = np.zeros(num_divs, dtype=int)
            for j, jj in enumerate(curve_dict[crv][0]):
                row[jj] = int(round(curve_dict[crv][1][j] / g))
            row_set.add(tuple(row))
        mori_rays = np.array(list(row_set), dtype=int)
        # Compute column corresponding to the origin
        mori_rays[:, 0] = -np.sum(mori_rays, axis=1)
        return mori_rays

    def mori_cone(self,
        pushed_down: bool = False,
        in_basis: bool = False,
        verbosity: int = 0) -> "Cone":
        """
        **Description:**
        Compute the Mori cone of the toric variety defined by the fan.

        **Arguments:**
        - `pushed_down`: Whether to push down the Mori cone.
        - `in_basis`:    Whether to put the Mori cone in basis.
        - `verbosity`:   The verbosity level. Higher is more verbose.

        **Returns:**
        The Mori cone.
        """
        include_origin = (pushed_down==False)
        
        rays = self.mori_rays()
        basis = self.vc.divisor_basis
        if include_origin and not in_basis:
            new_rays = rays
        elif not include_origin and not in_basis:
            new_rays = rays[:, 1:]
        else:
            if len(basis.shape) == 2:  # If basis is matrix
                new_rays = rays.dot(basis.T)
            else:
                new_rays = rays[:, basis]
        c = Cone(new_rays, check=len(basis.shape) == 2)
        return c
        # (vvv OLD... MAYBE PREFERABLE BUT NOT WORKING??? vvv)
        """
        dim = self.dim
        
        

        if in_basis:
            basis = self.vc.divisor_basis_inds
        else:
            basis = None

        # push down if basis provided
        #if False:
        #    if in_basis:
        #        print("basis requested so push down!")
        #        pushed_down = True

        # compute the facets
        if verbosity >= 1:
            print("Computing the facets...")
        facets = {
            facet
            for simp in self.cones()
            for facet in itertools.combinations(simp, r=dim - 1)
        }

        # get intersection numbers
        if verbosity >= 1:
            print("Computing the intersection numbers...")
        if False:
            kappa = self.intersection_numbers(
                pushed_down=pushed_down, in_basis=in_basis, verbosity=verbosity - 1
            )
        else:
            kappa = self.intersection_numbers(
                pushed_down = False,
                in_basis = False,
                verbosity=verbosity - 1
            )

        # get the rays of the Mori cone
        if verbosity >= 1:
            print("Computing the rays...")
        ray_iter = (
            basis
            if basis is not None
            else [i for i in range(0, self.vc.size + 1)]
        )
        rays = []
        for facet in facets:
            
            for extra_zeros in range(0,1+0*len(facet)+1):
                for fake_facet in itertools.combinations(facet, r=len(facet)-extra_zeros):
                    fake_facet = extra_zeros*(0,) + fake_facet

                    r = np.array([kappa.get(tuple(sorted(fake_facet + (i,))), 0) for i in ray_iter])
                    r = float_to_int_vec(r)
                    rays.append(r)
        rays = np.array(rays)
        if verbosity >= 1:
            print(f"Found rays = {rays}")

        # check for trivial cones
        if (len(rays) == 0) or max(np.linalg.norm(rays, axis=1)) < eps:
            size = dim if in_basis else len(self.vc.divisor_basis_inds)
            H = np.vstack(
                (+np.identity(size, dtype=int), -np.identity(size, dtype=int))
            )
            return Cone(hyperplanes=H)

        return Cone(rays=rays)
        """

    def kahler_cone(self, pushed_down=False, in_basis=False, verbosity=0):
        """
        **Description:**
        Compute the Kahler cone of the toric variety defined by the fan.

        **Arguments:**
        - `pushed_down`: Whether to push down the Mori cone.
        - `in_basis`:    Whether to put the Mori cone in basis.
        - `verbosity`:   The verbosity level. Higher is more verbose.

        **Returns:**
        The Kahler cone.
        """
        return self.mori_cone(
            pushed_down=pushed_down, in_basis=in_basis, verbosity=verbosity
        ).dual()

    def newton_polytope(self, divisor, dilate=False):
        """
        **Description:**
        Returns the Newton polytope associated to a divisor on a toric variety
        with 1-cones $Sigma(1)$ given by the points (not interior to facets) of
        a polytope, expressed as a linear combination of the prime toric
        divisors inherited by the CY hypersurface.

        **Arguments:**
        - `divisor`: A list of indices with length equal to the Picard group of
                     the CY hypersurface (assuming favorability), or the number
                     of points not interior to facets of the polytope. This can
                     be interpreted as a torus-invariant divisor, the set of
                     which overparameterize the Class group. Alternatively,
                     this is the obvious multidegree of a monomial in the Cox
                     ring, mapped to the Class group multidegree by multiplying
                     by the GLSM charge matrix.
        - `dilate`:  Whether to dilate the Newton polytope into a normally
                     equivalent integer polytope or not. If False, then just
                     compute the integer hull (convex hull of contained lattice
                     points) instead.


        **Returns:**
        The Newton polytope for the passed divisor
        """

        return HPolytope(
            np.array(
                [
                    list(point) + [divisor[n]]
                    for n, point in enumerate(self.vectors(which=self.used_labels))
                ]
            ),
            verbosity=0,
            dilate=dilate,
        )

    def is_gorestein_fano(self):
        # idea: self.conv().is_reflexive() should be equivalent
        return self.newton_polytope([1] * len(self.used_labels)).is_reflexive()

    def h21_cy(self):
        """
        Makes assumption that for CYs in Gorestein Fano four-folds, h21(CY) = h11(CY) for dual polytope.
        """
        if not self.is_gorestein_fano():
            raise NotImplementedError()

        if p.labels_not_facet[1:] != self.used_labels:
            print(
                f"This function may not hold! Polytope labels are {p.labels_not_facet} and VC labels are {self.used_labels}"
            )

        return len(self.newton_polytope([1] * len(self.used_labels)).labels_not_facet)

    # generalize flip_linear
    # ----------------------
    def flop_linear(self, *args, **kwargs):
        # define the hooks
        def hook_init(fan):
            fan.kappa =  fan.intersection_numbers(
                pushed_down=True,
                in_basis=True,
                as_np_array=True)

        def hook_flip(fanpre, fanpost, circ):
            fanpost.kappa = flop(fanpre, fanpre.kappa, circ)

        # call flip_linear with the hooks
        return self.flip_linear(
            *args,
            hook_init=hook_init,
            hook_flip=hook_flip,
            **kwargs)

# intersection numbers
# --------------------
# kernels
# (See Fan.intersection_numbers)
def kappa_solve_nxn(pts, known, r):
    # alternative methods (slower)
    # -------------------
    #solutions, res, *_ = np.linalg.lstsq(
    #    pts[:, 0:N_unknown], -pts[:, N_unknown:] @ known,
    #    rcond=None
    #)
    #
    #solutions, res, *_ = sp.linalg.lstsq(
    #    pts[:, 0:N_unknown], -pts[:, N_unknown:] @ known,
    #    check_finite=False,lapack_driver='gelsy'
    #)
    M = pts[0:r,:].T
    b = -pts[r:,:].T @ known
    return np.linalg.solve(M.T @ M, M.T @ b).flatten()

@njit
def kappa_solve_1x1(pts,known):
    # solve M@x=b
    # solution x = a/b
    b = -pts[1:,:].T @ known
    M = pts[0:1,:].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M
    x0 = b[0,0] * (1/M[0,0])

    # hardcoded 1x1 solve
    return [x0]

@njit
def kappa_solve_2x2(pts,known):
    # solve M@x=b
    b = -pts[2:,:].T @ known
    M = pts[0:2,:].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M

    # hardcoded 2x2 solve
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if det == 0:
        raise ValueError("Singular matrix")

    x0 = (b[0,0]*M[1,1] - M[0, 1]*b[1,0]) / det
    x1 = (M[0,0]*b[1,0] - b[0,0]*M[1, 0]) / det

    # return
    return [x0,x1]

@njit
def kappa_solve_3x3(pts,known):
    # solve M@x=b
    b = -pts[3:,:].T @ known
    M = pts[0:3,:].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M

    # hardcoded 3x3 solve
    det = (
        M[0,0]*(M[1,1]*M[2,2] - M[1,2]*M[2,1])
        - M[0,1]*(M[1,0]*M[2,2] - M[1,2]*M[2,0])
        + M[0,2]*(M[1,0]*M[2,1] - M[1,1]*M[2,0])
    )
    if det == 0:
        raise ValueError("Singular matrix")

    # inverse via cofactor / adjugate
    inv = np.empty((3,3), dtype=np.float64)
    inv[0,0] =  (M[1,1]*M[2,2] - M[1,2]*M[2,1]) / det
    inv[0,1] = -(M[0,1]*M[2,2] - M[0,2]*M[2,1]) / det
    inv[0,2] =  (M[0,1]*M[1,2] - M[0,2]*M[1,1]) / det
    inv[1,0] = -(M[1,0]*M[2,2] - M[1,2]*M[2,0]) / det
    inv[1,1] =  (M[0,0]*M[2,2] - M[0,2]*M[2,0]) / det
    inv[1,2] = -(M[0,0]*M[1,2] - M[0,2]*M[1,0]) / det
    inv[2,0] =  (M[1,0]*M[2,1] - M[1,1]*M[2,0]) / det
    inv[2,1] = -(M[0,0]*M[2,1] - M[0,1]*M[2,0]) / det
    inv[2,2] =  (M[0,0]*M[1,1] - M[0,1]*M[1,0]) / det

    # multiply inverse with b
    x0 = inv[0,0]*b[0,0] + inv[0,1]*b[1,0] + inv[0,2]*b[2,0]
    x1 = inv[1,0]*b[0,0] + inv[1,1]*b[1,0] + inv[1,2]*b[2,0]
    x2 = inv[2,0]*b[0,0] + inv[2,1]*b[1,0] + inv[2,2]*b[2,0]

    return [x0, x1, x2]

# flops
def minface_dim(config, labels):
    # get dim (face must have at least this dimension)
    p = config.conv()
    A = config.vectors(labels)
    dim = np.linalg.matrix_rank([pt.tolist()+[1] for pt in A])-1
    
    # try increasing dimensions
    for facedim in range(dim,config.dim+1):
        for face in p.faces(facedim):
            if labels.issubset(face.labels):
                return facedim
    
def curve_to_gv(fan, kappa, circ, verbosity=0):
    config = fan.vc
    
    # check that the signature is of the right type
    if min(circ.signature) != 2:
        raise ValueError
    
    unsigned = set(circ.Z)
    
    # get the minface_dim of the circuit
    dim = minface_dim(config, unsigned)
    if max(circ.signature) == 3:
        if dim == 4:
            if verbosity >= 1:
                print(f"(3,2), minface dim=4 -> gv=1")
            return 1
        else:
            if verbosity >= 1:
                print(f"(3,2), minface dim<4 -> gv=0")
            return 0
    elif max(circ.signature) == 2:
        if dim == 4:
            raise ValueError("Dim=4 minface???")
        else:
            # get a 3-cone on this side
            c = sorted(list(circ.Zneg) + [circ.Zpos[0]])
            
            if set(c).issubset(fan.vc.divisor_basis):
                c_inds = fan.vc.labels_to_inds(c, ambient_labels=fan.vc.divisor_basis)
                kappa_c = kappa[*c_inds]
                if False:
                    A = kappa_c

                    c_inds = fan.vc.labels_to_inds(c)
                    divisors = fan.vc.gale()[c_inds,:].T
                    B = ((kappa@divisors[:,0])@divisors[:,1])@divisors[:,2]
                    if abs(A-B)>1e-4:
                        print(A,B)
                        print(c, fan.vc.labels_to_inds(c, ambient_labels=fan.vc.divisor_basis), fan.vc.labels_to_inds(c), fan.vc.labels_to_inds(c, ambient_labels=fan.labels))
                        print(fan.vc.divisor_basis)
                        print(divisors)
                        adsasa()
            else:
                c_inds = fan.vc.labels_to_inds(c)
                divisors = fan.vc.gale()[c_inds,:].T
                kappa_c = ((kappa@divisors[:,0])@divisors[:,1])@divisors[:,2]
            
            if verbosity >= 1:
                print(f"(2,2), minface dim<4 -> gv=kappa[cone={c}]={kappa_c}")
            
            return np.round(kappa_c)
    else:
        raise ValueError

def flop(fan, kappa, circ, verbosity=0):
    # get GV
    gv = curve_to_gv(fan, kappa, circ, verbosity-1)
    if verbosity >= 1:
        print(f"GV = {gv}")

    if gv == 0:
        return kappa.copy()

    n = np.array(circ.normal)
    n = n[fan.vc.divisor_basis_inds]
    
    if False:
        if False:
            outer = np.einsum('i,j,k->ijk', n, n, n)
        else:
            outer = n[:, None, None] * n[None, :, None] * n[None, None, :]
        return kappa - gv*outer
    else:
        kappa_flopped = kappa.copy()
        n_nonzero = np.where(n)[0]

        for i,j,k in itertools.product(n_nonzero,repeat=3):
            kappa_flopped[i,j,k] = kappa_flopped[i,j,k] - gv*n[i]*n[j]*n[k]

        return kappa_flopped

# misc
# ----
# give Triangulation a method to directly generate VCs/Fans
def vc(self, include_points_interior_to_facets=None):
    """
    **Description:**
    Construct the VectorConfiguration associated to the triangulation.

    **Arguments:**
    - `include_points_interior_to_facets`: Whether to include points interior
        to facets

    **Returns:**
    The associated VectorConfiguration.
    """
    # set include_points_interior_to_facets
    if include_points_interior_to_facets is None:
        include_points_interior_to_facets = tuple(self.labels) == self.polytope().labels

    # get the vc
    vc = self.polytope().vc(include_points_interior_to_facets=include_points_interior_to_facets)

    return vc
Triangulation.vc = vc

def fan(self, include_points_interior_to_facets=None):
    """
    **Description:**
    Construct the Fan associated to the triangulation.

    **Arguments:**
    - `include_points_interior_to_facets`: Whether to include points interior
        to facets

    **Returns:**
    The associated Fan.
    """
    # set include_points_interior_to_facets
    if include_points_interior_to_facets is None:
        include_points_interior_to_facets = tuple(self.labels) == self.polytope().labels

    # get the vc
    vc = self.polytope().vc(include_points_interior_to_facets=include_points_interior_to_facets)

    # get/return the fan
    fan = vc.subdivide(cells=self.simplices()[:,1:])
    return fan
Triangulation.fan = fan
