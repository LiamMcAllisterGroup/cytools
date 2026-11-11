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

"""Fans over vector configurations.

This module defines the [`Fan`][cytools.vector_config.fan.Fan] class, which
describes a subdivision of a vector configuration into cones, together with
helper functions to compute intersection numbers and to track them through flops.
Importing it also adds `vc` and `fan` methods to
[`Triangulation`][cytools.triangulation.Triangulation].
"""

# external imports
import collections
import itertools
from numba import njit
import numpy as np
import regfans
from typing import Union

# core CYTools imports
from cytools import Cone, HPolytope, utils
from cytools.triangulation import Triangulation

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


_INTERSECTION_NUMBERS_DEFAULT_EPS = 1e-4
_INTERSECTION_NUMBERS_DEFAULT_DIGITS = 10


class Fan(regfans.fan.Fan):
    """A fan over a vector configuration.

    This is the analogue of [`Triangulation`][cytools.triangulation.Triangulation]
    for vector configurations. Besides the combinatorial methods inherited from the
    `Fan` class of the [regfans](https://github.com/natemacfadden/regfans) package, it
    provides toric methods such as intersection numbers and Mori and Kähler cones.
    Most of these assume that the fan is a triangulation (i.e. simplicial) and, for
    the methods involving a basis of divisors or the Calabi-Yau hypersurface, that
    the convex hull of the vectors is a reflexive 4D polytope.

    This class is not intended to be constructed directly. Instead, use
    [`VectorConfiguration.triangulate`][cytools.vector_config.vectorconfiguration.VectorConfiguration.triangulate]
    (also available as `subdivide`) or `Triangulation.fan` (see
    [`fan`][cytools.vector_config.fan.fan]). The constructor is inherited from
    regfans and takes the following arguments:

    - `vc` (`VectorConfiguration`): The vector configuration that the fan is over.
    - `cones` (`list[list[int]]`): The maximal cones of the fan, each given by the
      labels of its vectors.
    - `heights` (`ArrayLike | None`): Heights defining the fan, if it is regular.
      If not specified, they are computed when needed.

    Examples:
        We construct a fan from a triangulation of a reflexive polytope and find its
        cones.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> f = p.triangulate().fan()
        >>> len(f.cones())
        9
        >>> f.cones()[:3]
        ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 6))
    """

    # read from regfans type
    @classmethod
    def from_regfans(cls, fan: "regfans.Fan") -> "Fan":
        """Convert a regfans `Fan` object into a CYTools `Fan` object.

        The new object shares the state of the input (a shallow copy of its attributes
        is made). This is used by
        [`VectorConfiguration.triangulate`][cytools.vector_config.vectorconfiguration.VectorConfiguration.triangulate].

        Args:
            fan (regfans.Fan): The regfans fan.

        Returns:
            Fan: The CYTools fan.
        """
        obj = cls.__new__(cls)  # allocate new instance
        obj.__dict__ = fan.__dict__.copy()  # copy state
        return obj

    # cones/simplices
    # ---------------
    def cones(
        self,
        dim: int = None,
        formal: bool = False,
        as_hyps: bool = False,
        as_inds: bool = False,
        ind_offset: int = 0,
    ) -> Union[tuple[tuple[int]], list["Cone"]]:
        """Return the cones of the fan.

        By default, the maximal cones are returned, each as a sorted tuple of labels. If
        `dim` is set, the `dim`-dimensional faces of the maximal cones are returned
        instead, in no particular order.

        Also available as `simplices` and `simps`.

        Args:
            dim (int | None): If set, return the `dim`-dimensional cones (faces of the
                maximal cones). Only implemented for simplicial fans.
            formal (bool): Whether to return the cones as
                [`Cone`][cytools.cone.Cone] objects. If true, the remaining format
                options are ignored.
            as_hyps (bool): Whether to return each cone by its hyperplane inequalities.
            as_inds (bool): Whether to return each cone by the indices of its vectors
                instead of their labels.
            ind_offset (int): An offset added to the indices when `as_inds` is true.

        Returns:
            tuple[tuple[int, ...], ...] | list: The cones, as tuples of labels,
                tuples of indices, lists of hyperplane inequalities, or
                [`Cone`][cytools.cone.Cone] objects (the latter in a tuple).

        Raises:
            ValueError: If more than one of `as_hyps` and `as_inds` is set, or if `dim`
                is set for a non-simplicial fan.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> f.cones()[:2]
            ((1, 2, 3, 4), (1, 2, 3, 5))
            >>> f.cones(as_inds=True)[:2]
            ((0, 1, 2, 3), (0, 1, 2, 4))
            >>> f.cones(formal=True)[0]
            A 4-dimensional rational polyhedral cone in RR^4 generated by 4 rays
        """
        if formal:
            return tuple([self.vc.cone(c) for c in super(Fan, self).cones(dim=dim)])
        else:
            return super(Fan, self).cones(
                dim=dim, as_hyps=as_hyps, as_inds=as_inds, ind_offset=ind_offset
            )

    # aliases
    simplices = cones
    simps = cones

    def restricted_simps(
        self, to_dim: int = 2, padded: bool = False, as_face_inds: bool = False
    ):
        """Restrict the cones of the fan to the faces of the polytope.

        For each `to_dim`-dimensional face of the convex hull of the vectors, each
        maximal cone is intersected with the face (as sets of labels), and the maximal
        such intersections with at least two elements are kept. For a fine star
        triangulation of a polytope, this gives the restriction of the triangulation to
        each face.

        Args:
            to_dim (int): The dimension of the faces to restrict to.
            padded (bool): Whether to pad restricted simplices consisting of two points
                by repeating their last point, so that they have three entries.
            as_face_inds (bool): Whether to specify the points by their indices within
                the labels of each face instead of by their labels.

        Returns:
            list[list[list[int]]]: For each `to_dim`-dimensional face (in the
                order given by [`Polytope.faces`][cytools.polytope.Polytope.faces]),
                the sorted list of restricted simplices, each a sorted list of labels
                (or of face indices).

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> f.restricted_simps()[5]
            [[1, 4, 6], [1, 5, 6], [4, 5, 6]]
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
            # (build the label->index map once per face; `.index` would be
            #  O(#labels) per lookup)
            if as_face_inds:
                label_to_ind = {lbl: i for i, lbl in enumerate(face.labels)}
                face_simps_reduced = [
                    sorted(label_to_ind[i] for i in simp) for simp in face_simps_reduced
                ]
            else:
                face_simps_reduced = [sorted(simp) for simp in face_simps_reduced]

            # pad the simplices
            # (the entries are sorted lists by now, so this works regardless of
            #  `as_face_inds`)
            if padded:
                for i, simp in enumerate(face_simps_reduced):
                    if len(simp) == 2:
                        face_simps_reduced[i] = simp + [simp[-1]]

            # save it!
            face_simps_reduced = sorted(face_simps_reduced)
            restricted.append(face_simps_reduced)

        return restricted

    # VC <-> PC
    # ---------
    def get_pc_triangulation(
        self, check: bool = True, verbosity: int = 0
    ) -> "Triangulation":
        """Return the triangulation of the point configuration associated to the fan.

        This requires that the fan respects the point configuration, i.e. that it is the
        fan over a star triangulation of the convex hull of the vectors (with the origin
        labelled 0). For regular fans, the triangulation is obtained by lifting the
        points by the heights of the fan, with the origin at height 0; otherwise, it is
        constructed from the cones of the fan, with the origin added to each of them.

        Also available as `triang`.

        Args:
            check (bool): Whether to check that the fan respects the point configuration
                and, for regular fans, that the heights generate the triangulation.
            verbosity (int): The verbosity level. Currently unused.

        Returns:
            Triangulation: The corresponding star triangulation.

        Raises:
            ValueError: If `check` is true and the fan does not respect the point
                configuration, or if the heights of a regular fan are negative.
            NotImplementedError: If `check` is true and the fan is not regular.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> t = p.triangulate()
            >>> t.fan().get_pc_triangulation() == t
            True
        """
        # check if this makes sense
        if check:
            if not self.respects_ptconfig():
                raise ValueError("Not triangulation of PC")

        # super easy for regular triangulations. Lift by heights >=0, with origin at 0
        if self.is_regular():
            p = self.vc.conv(which=self.labels)
            pts_in_facets = self.labels == p.labels[1:]
            h = self.heights()
            if not np.all(h >= 0):
                msg = "Heights are assumed non-negative here... "
                msg += f"your heights={h}..."
                raise ValueError(msg)
            return p.triangulate(
                heights=[0] + h.tolist(),
                include_points_interior_to_facets=pts_in_facets,
                check_heights=check,
            )

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
    def secondary_cone(
        self,
        via_circuits: bool = False,
        project_lineality: bool = False,
        verbosity: int = 0,
    ):
        """Compute the secondary cone of the fan.

        For a regular fan, a height vector generates the fan if and only if it lies in
        the relative interior of the secondary cone. Irregular fans are not generated by
        any heights, so they do not have secondary cones. When computed by local folding
        (`via_circuits=False`), the resulting cone is solid (full-dimensional) if and
        only if the fan is regular, so this can be used to check regularity. The result
        is cached separately for each value of `via_circuits`.

        Args:
            via_circuits (bool): Whether to compute the cone from the circuits of the
                fan instead of by local folding. This is correct for regular fans, but
                it can give a full-dimensional cone for an irregular fan, so it must not
                be used to check regularity.
            project_lineality (bool): Whether to project out the lineality space (the
                linear height functions) by mapping the rays with the Gale transform.
                This gives the corresponding cone in the chamber complex, of dimension
                $n-d$. For a fine regular star triangulation of a reflexive polytope,
                this is the Kähler cone of the toric variety.
            verbosity (int): The verbosity level.

        Returns:
            Cone: The secondary cone, in the $n$-dimensional space of heights, or its
                projection if `project_lineality` is true.

        Warning:
            Irregularity can only be detected with `via_circuits=False`.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> f.secondary_cone()
            A rational polyhedral cone in RR^6 defined by 18 hyperplanes
            >>> f.secondary_cone(project_lineality=True).rays()
            array([[3, 1],
                   [1, 0]])
        """
        # the circuits-based and the folding-based cones can differ (see the
        # warning above), so they must be cached separately
        if not hasattr(self, "_secondary_cone"):
            self._secondary_cone = dict()

        cache_key = bool(via_circuits)
        if cache_key not in self._secondary_cone:
            H = super().secondary_cone_hyperplanes(
                via_circuits=via_circuits, verbosity=verbosity
            )
            self._secondary_cone[cache_key] = Cone(hyperplanes=H)

        cone = self._secondary_cone[cache_key]

        if project_lineality:
            return Cone(rays=cone.rays() @ (self.vc.gale()))
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
        eps: float = _INTERSECTION_NUMBERS_DEFAULT_EPS,
        digits: int = _INTERSECTION_NUMBERS_DEFAULT_DIGITS,
        copy: bool = True,
        verbosity: int = 0,
    ) -> Union[dict, "ArrayLike"]:
        r"""Compute the intersection numbers of the toric variety defined by the fan.

        The intersection numbers are returned in a sparse format: a dictionary whose
        keys are sorted tuples of indices $(i_1 \le \dots \le i_d)$ and whose values are
        the corresponding intersection numbers $D_{i_1} \cdots D_{i_d}$, where
        $D_1, \dots, D_n$ are the prime toric divisors (index $i$ corresponding to the
        vector with label $i$) and $D_0 = K = -\sum_i D_i$ is the canonical divisor. The
        numbers are computed as floating-point values. With `pushed_down`, they are the
        intersection numbers of the anticanonical hypersurface (the Calabi-Yau for a
        reflexive 4D polytope), $\kappa_{ijk} = -D_0 D_i D_j D_k$.

        The full set of ambient intersection numbers is computed once and cached, as are
        the formatted outputs for `pushed_down` or `in_basis`.

        Also available as `int_nums` and `kappa`.

        Args:
            pushed_down (bool): Whether to return the intersection numbers of the
                anticanonical hypersurface instead of those of the toric variety.
            in_basis (bool): Whether to only return the intersection numbers of the
                divisors in
                [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis].
                Indices then refer to positions in the basis.
            symmetrize (bool): Whether to include all permutations of the indices,
                instead of only sorted ones.
            as_np_array (bool): Whether to return the intersection numbers as a dense
                NumPy array. This implies `symmetrize=True`. Without `in_basis`, the
                array is indexed by the labels, together with 0 for the canonical
                divisor, when `pushed_down` is false, and by the labels minus one when
                it is true.
            eps (float): Entries with absolute value below this are dropped. Only used
                when neither `pushed_down` nor `in_basis` is set, and it does not affect
                the cached values.
            digits (int | None): The number of decimal digits the values are rounded to
                (`None` for no rounding). Only used when neither `pushed_down` nor
                `in_basis` is set, and it does not affect the cached values.
            copy (bool): Whether to return a copy of the cached output when
                `pushed_down` or `in_basis` is set. Set it to false only if the result
                will not be modified, to avoid copying it on every call.
            verbosity (int): The verbosity level.

        Returns:
            dict[tuple[int, ...], float] | np.ndarray: The intersection numbers.

        Raises:
            ValueError: If the fan is not a triangulation.

        Examples:
            We compute the intersection numbers of a Calabi-Yau hypersurface, which
            agree with those computed by
            [`CalabiYau.intersection_numbers`][cytools.calabiyau.CalabiYau.intersection_numbers]
            up to rounding.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> kappa = f.intersection_numbers(pushed_down=True, in_basis=True)
            >>> {k: round(v) for k, v in kappa.items()}
            {(0, 0, 1): 1, (0, 1, 1): -3, (1, 1, 1): 9}
        """
        if not hasattr(self, "_kappa"):
            # a plain dict (not a defaultdict) so that a missing key is an
            # error rather than a silent 0.0; the few genuinely-optional
            # lookups below use .get(..., 0.0) explicitly
            self._kappa = dict()
            self._kappa_known_labels = set()
        if not hasattr(self, "_kappa_default"):
            self._kappa_default = None
        if not hasattr(self, "_kappa_view_cache"):
            # formatted views keyed by (pushed_down, in_basis, symmetrize, as_np_array)
            # invalidated alongside _kappa when labels change
            self._kappa_view_cache = {}

        if not self.is_triangulation():
            raise ValueError("Fan is not a triangulation!")

        if as_np_array:
            symmetrize = True

        def format_kappa(
            raw_kappa,
            *,
            eps_value,
            digits_value,
            symmetrize_output,
            as_np_array_output,
        ):
            if (
                eps_value == _INTERSECTION_NUMBERS_DEFAULT_EPS
                and digits_value == _INTERSECTION_NUMBERS_DEFAULT_DIGITS
                and not symmetrize_output
                and not as_np_array_output
                and self._kappa_default is not None
            ):
                return dict(self._kappa_default)

            kappa = dict(raw_kappa)

            for k in list(kappa.keys()):
                v = kappa[k]

                if np.abs(v) < eps_value:
                    del kappa[k]
                    continue

                if digits_value is not None:
                    kappa[k] = round(v, digits_value)

            if symmetrize_output:
                keys = list(kappa.keys())
                for k in keys:
                    for k_perm in itertools.permutations(k):
                        kappa[tuple(k_perm)] = kappa[k]

            if as_np_array_output:
                kappa_np = np.zeros(self.ambient_dim * (self.vc.size + 1,))
                for k, v in kappa.items():
                    kappa_np[k] = v
                return kappa_np

            return kappa

        # given a basis
        # -------------
        # push down the intersection numbers and represent them in said basis
        if pushed_down or in_basis:
            # view cache: these flag combos are pure functions of _kappa
            view_key = (
                bool(pushed_down),
                bool(in_basis),
                bool(symmetrize),
                bool(as_np_array),
            )
            if self._kappa_known_labels == set(self.labels):
                cached = self._kappa_view_cache.get(view_key)
                if cached is not None:
                    if not copy:
                        return cached
                    return (
                        cached.copy()
                        if isinstance(cached, np.ndarray)
                        else dict(cached)
                    )

            # get the ambient intersection numbers
            kappa = self.intersection_numbers(symmetrize=False, digits=None)
            arr_size = self.vc.size

            if verbosity >= 10:
                tmp = sorted([k + (round(v),) for k, v in kappa.items()])
                print(f"ambient intersection numbers were (in COO format) {tmp}")

            if pushed_down:
                # push down the intersection numbers
                # (i.e., sum over one index... symmetric so doesn't matter
                #  which)
                # (equiv to taking kappa[i,j,k] = -kappa[0,i,j,k] b/c
                #  anticanonical)
                kappa_pushdown = dict()
                for k, v in kappa.items():
                    if (k[0] == 0) and (0 not in k[1:]):
                        kappa_pushdown[k[1:]] = -v

                kappa = kappa_pushdown

                if verbosity >= 10:
                    tmp = sorted([k + (round(v),) for k, v in kappa.items()])
                    print(f"after pushing kappa down, it is {tmp}")

            # write the intersection numbers in the basis
            if in_basis:
                basis = self.vc.divisor_basis

                # ensure basis is list
                basis = list(basis)
                arr_size = len(basis)

                # set/dict lookups in place of list scans (faster for in_basis=True)
                basis_set = set(basis)
                basis_idx = {x: i for i, x in enumerate(basis)}
                non_basis_set = {
                    0,
                    *(i for i in self.used_labels if i not in basis_set),
                }
                kappa = {
                    tuple(basis_idx[i] for i in k): kappa[k]
                    for k in kappa
                    if not any(i in non_basis_set for i in k)
                }

                if verbosity >= 10:
                    tmp = sorted([k + (round(v),) for k, v in kappa.items()])
                    print(f"after putting kappa in-basis, it is {tmp}")

            # symmetrize
            if symmetrize:
                keys = list(kappa.keys())
                for k in keys:
                    for k_perm in itertools.permutations(k):
                        kappa[tuple(k_perm)] = kappa[k]

                if verbosity >= 10:
                    tmp = sorted([k + (round(v),) for k, v in kappa.items()])
                    print(f"after symmetrizing kappa, it is {tmp}")

            # map to numpy array
            if as_np_array:
                kappa_np = np.zeros((self.ambient_dim - pushed_down) * (arr_size,))
                if in_basis:
                    for k, v in kappa.items():
                        kappa_np[k] = v
                else:
                    # map labels to indices...
                    for k, v in kappa.items():
                        kappa_np[tuple(ki - 1 for ki in k)] = v
                result = kappa_np
            else:
                result = kappa

            # cache the freshly-built result; hand back canonical or a copy
            # (the check is redone here since `_kappa` may have just been
            #  computed by the recursive call above)
            if self._kappa_known_labels == set(self.labels):
                self._kappa_view_cache[view_key] = result
            if not copy:
                return result
            return result.copy() if isinstance(result, np.ndarray) else dict(result)

        # not given a basis
        # -----------------
        # check if we know the answer
        if self._kappa_known_labels == set(self.labels):
            if self._kappa_default is None:
                self._kappa_default = format_kappa(
                    self._kappa,
                    eps_value=_INTERSECTION_NUMBERS_DEFAULT_EPS,
                    digits_value=_INTERSECTION_NUMBERS_DEFAULT_DIGITS,
                    symmetrize_output=False,
                    as_np_array_output=False,
                )

            return format_kappa(
                self._kappa,
                eps_value=eps,
                digits_value=digits,
                symmetrize_output=symmetrize,
                as_np_array_output=as_np_array,
            )

        # don't know the answer... need to calculate it...
        # (any previously cached values were computed for a different label
        #  set, i.e. for a different fan. They must be purged rather than
        #  updated, else stale keys leak into every subsequent output.)
        self._kappa = dict()
        self._kappa_known_labels = set()
        self._kappa_view_cache = {}
        self._kappa_default = None

        # setup
        # -----
        # helpers
        dim = self.dim

        # (formally add the origin to ease indexing)
        vecs = np.vstack([np.zeros((1, dim), dtype=int), self.vectors()])
        simps = self.cones(as_inds=True, ind_offset=1)
        simps_np = np.array(simps)

        # face-to-neighbor map for each codim
        # (a neighbor of a face `f` is any point `i` s.t. $f cup {i}$ is a face)
        # (really, this is an encoding of the face lattice...)
        neighbors = {i: dict() for i in range(1, dim)}

        for r in range(1, dim):
            rdict = neighbors[r]

            # map each r-dim face to the neighboring points
            for simp in simps:
                for s in itertools.combinations(simp, r):
                    if s in rdict:
                        rdict[s].update(simp)
                    else:
                        rdict[s] = set(simp)

            # delete the indices corresponding to the face itself
            for face in neighbors[r]:
                rdict[face] -= set(face)

        # compute the intersection numbers
        # --------------------------------
        # for distinct indices, just 1/vol of the cone
        vals = 1 / np.abs(np.linalg.det(vecs[simps_np]))  # NumPy broadcasting
        kappa_dict = self._kappa  # local aliases (the loop below is hot)
        kappa_get = kappa_dict.get
        kappa_dict.update(zip(simps, map(float, vals)))

        if verbosity >= 1:
            msg = "After computing the intersection numbers associated to "
            msg += "solid cones, we have:"
            print(msg)
            print({k: v for k, v in self._kappa.items()})

        # helpers
        if dim == 4:

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
        for r in range(dim - 1, 0, -1):  # dim-1, dim-2, ..., 1
            if r == 3:
                kappa_solver = kappa_solve_3x3
            elif r == 2:
                kappa_solver = kappa_solve_2x2
            elif r == 1:
                kappa_solver = kappa_solve_1x1
            else:

                def kappa_solver(pts, known):
                    return kappa_solve_nxn(pts, known, r)

            # iterate over each r-face
            for face, neighbs in neighbors[r].items():
                neighbs = sorted(neighbs)
                inds = list(face) + neighbs  # analogous to the star of the face
                pts = vecs[inds].astype(float)

                if verbosity >= 1:
                    msg = "Computing the intersection numbers associated to "
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
                    #
                    # a key can legitimately be absent (the corresponding
                    # indices don't span a cone of the fan), in which case the
                    # intersection number vanishes
                    known_vals = np.array(
                        [kappa_get(insert_sorted(prefactor, i), 0.0) for i in neighbs]
                    ).reshape(-1, 1)

                    # the unknown intersection numbers will have indices of the
                    # form `face+duplicates+[i]` for i in face

                    # solve for the unknowns
                    # (i.e., find x such that pts.T@kappa_{..., [x,known]} == 0)
                    # (i.e., find x such that pts.T[:,:N_unknown]@kappa_{...,x} == -pts.T[:,N_unknown:]@kappa_{..., known})
                    sol = kappa_solver(pts, known_vals)

                    # save it
                    for i, val in zip(face, sol):
                        kappa_dict[insert_sorted(prefactor, i)] = float(val)

        # intersection numbers w/ 0
        # -------------------------
        kappa_nzeros = [kappa_dict]

        for num_zeros in range(1, dim + 1):
            # make container for kappa w/ n zeros
            kappa_nzeros.append(collections.defaultdict(float))

            # iterate over values with 1 fewer 0
            for k, v in kappa_nzeros[num_zeros - 1].items():
                for kprime in set(
                    itertools.combinations(k[num_zeros - 1 :], r=dim - num_zeros)
                ):
                    kappa_nzeros[num_zeros][num_zeros * (0,) + kprime] -= v

        for kappa_n in kappa_nzeros[1:]:
            self._kappa.update(kappa_n)

        # save kappa labels (for caching); drop stale views since _kappa just changed
        self._kappa_known_labels = set(self.labels)
        self._kappa_view_cache = {}
        self._kappa_default = format_kappa(
            self._kappa,
            eps_value=_INTERSECTION_NUMBERS_DEFAULT_EPS,
            digits_value=_INTERSECTION_NUMBERS_DEFAULT_DIGITS,
            symmetrize_output=False,
            as_np_array_output=False,
        )

        return self.intersection_numbers(
            pushed_down=pushed_down,
            in_basis=in_basis,
            symmetrize=symmetrize,
            as_np_array=as_np_array,
            eps=eps,
            digits=digits,
            copy=copy,
            verbosity=0,
        )

    int_nums = intersection_numbers
    kappa = intersection_numbers

    def c2(self, eps: float = 1e-4, digits: int = 4) -> "ArrayLike":
        r"""Compute the second Chern class of the anticanonical hypersurface.

        The second Chern class is returned as its integrals $\int c_2 \wedge D_a$
        over the prime toric divisors, computed as $\sum_{\{i,j\}} \kappa_{ija}$ where
        the sum runs over the 2-dimensional cones of the fan. This assumes a
        4-dimensional fan, so that the hypersurface is a Calabi-Yau threefold.

        Args:
            eps (float): Passed to
                [`intersection_numbers`][cytools.vector_config.fan.Fan.intersection_numbers].
                Currently it has no effect, since the pushed-down intersection numbers
                are used.
            digits (int): Passed to
                [`intersection_numbers`][cytools.vector_config.fan.Fan.intersection_numbers].
                Currently it has no effect, since the pushed-down intersection numbers
                are used.

        Returns:
            list[int]: The integrals of $c_2$ over the divisors with labels
                $1, 2, \dots$, up to the largest label used by the fan.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.triangulate().fan().c2()
            [36, 306, 204, 36, 36, -6]
        """
        # get the 2-cones
        max_ind = -1
        two_cones = set()
        for s in self.cones():
            max_ind = max(max_ind, max(s))

            for two_cone in itertools.combinations(s, 2):
                two_cones.add(two_cone)
        two_cones = np.array(list(two_cones)) - 1

        # get the intersection numbers
        kappa = self.intersection_numbers(
            pushed_down=True, as_np_array=True, eps=eps, digits=digits
        )

        # compute c2
        out = []
        for a in range(max_ind):
            vals = kappa[two_cones[:, 0], two_cones[:, 1], a]
            out.append(round(vals.sum()))

        return out

    def mori_rays(self) -> "ArrayLike":
        """Compute the generators of the Mori cone of the toric variety.

        The generators are the torus-invariant curves corresponding to the
        $(d-1)$-dimensional cones of the fan, each given by its intersection numbers
        with the divisors. The first column corresponds to the origin (the canonical
        divisor) and column $i$ to the prime toric divisor with label $i$.

        Returns:
            np.ndarray: The generators of the Mori cone, one per row.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.triangulate().fan().mori_rays()
            array([[  0,   1,   0,   0,   1,   1,  -3],
                   [-18,   1,   9,   6,   1,   1,   0],
                   [ -6,   0,   3,   2,   0,   0,   1]])
        """
        intnums = self.intersection_numbers(in_basis=False)
        dim = self.dim
        num_divs = self.vc.gale().shape[0] + 1

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

    def mori_cone(
        self, pushed_down: bool = False, in_basis: bool = False, verbosity: int = 0
    ) -> "Cone":
        """Compute the Mori cone of the toric variety.

        The rays of the cone are given by
        [`mori_rays`][cytools.vector_config.fan.Fan.mori_rays].

        Args:
            pushed_down (bool): Whether to drop the coordinate corresponding to the
                origin (the canonical divisor). Ignored if `in_basis` is true.
            in_basis (bool): Whether to use the basis of curves dual to
                [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis].
            verbosity (int): The verbosity level. Currently unused.

        Returns:
            Cone: The Mori cone.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> f.mori_cone()
            A 2-dimensional rational polyhedral cone in RR^7 generated by 3 rays
            >>> f.mori_cone(in_basis=True).rays()
            array([[ 1, -3],
                   [ 1,  0],
                   [ 0,  1]])
        """
        include_origin = not pushed_down

        rays = self.mori_rays()

        # the divisor basis is only needed (and only defined) when the Mori
        # cone is requested in a basis
        if not in_basis:
            new_rays = rays if include_origin else rays[:, 1:]
            return Cone(new_rays, check=False)

        basis = self.vc.divisor_basis
        if len(basis.shape) == 2:  # If basis is matrix
            new_rays = rays.dot(basis.T)
        else:
            new_rays = rays[:, basis]
        return Cone(new_rays, check=len(basis.shape) == 2)

    def kahler_cone(self, pushed_down=False, in_basis=False, verbosity=0):
        """Compute the Kähler cone of the toric variety.

        This is the dual of the
        [`mori_cone`][cytools.vector_config.fan.Fan.mori_cone].

        Args:
            pushed_down (bool): Whether to drop the coordinate corresponding to the
                origin (the canonical divisor). Ignored if `in_basis` is true.
            in_basis (bool): Whether to use
                [`divisor_basis`][cytools.vector_config.vectorconfiguration.VectorConfiguration.divisor_basis].
            verbosity (int): The verbosity level. Currently unused.

        Returns:
            Cone: The Kähler cone.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.triangulate().fan().kahler_cone(in_basis=True).rays()
            array([[3, 1],
                   [1, 0]])
        """
        return self.mori_cone(
            pushed_down=pushed_down, in_basis=in_basis, verbosity=verbosity
        ).dual()

    def newton_polytope(self, divisor, dilate=False):
        r"""Compute the polytope of a torus-invariant divisor.

        For the divisor $D = \sum_\rho a_\rho D_\rho$, where $D_\rho$ are the prime
        toric divisors corresponding to the rays $u_\rho$ of the fan, this is the
        polytope
        $P_D = \{m : \langle m, u_\rho \rangle + a_\rho \geq 0 \ \forall \rho\}$,
        whose lattice points correspond to the monomials of the Cox ring of degree $[D]$
        (the multidegree being obtained by multiplying the coefficients by the GLSM
        charge matrix). If $P_D$ is not a lattice polytope, it is converted into one as
        described in [`HPolytope`][cytools.h_polytope.h_polytope.HPolytope].

        Args:
            divisor (list[int]): The coefficients $a_\rho$ of the divisor, one for each
                ray of the fan, in the order of `used_labels`.
            dilate (bool): Whether to rescale $P_D$ so that it becomes a lattice
                polytope, instead of taking its integer hull.

        Returns:
            HPolytope: The polytope of the divisor.

        Examples:
            The polytope of the anticanonical divisor is the polar dual of the convex
            hull of the rays.

            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> f = p.triangulate().fan()
            >>> f.newton_polytope([1]*6).is_reflexive()
            True
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

    def is_gorenstein_fano(self):
        r"""Return whether the polytope of the anticanonical divisor is reflexive.

        The anticanonical polytope is
        [`newton_polytope`][cytools.vector_config.fan.Fan.newton_polytope] of
        $-K = \sum_\rho D_\rho$. Its reflexivity is necessary for the toric variety to
        be Gorenstein Fano, and sufficient when the fan is the face fan of the convex
        hull of its rays. Note that the fan itself is not checked.

        Returns:
            bool: Whether the anticanonical polytope is reflexive.
        """
        # idea: self.conv().is_reflexive() should be equivalent
        return self.newton_polytope([1] * len(self.used_labels)).is_reflexive()

    def h21_cy(self):
        """Return $h^{2,1}$ of the anticanonical Calabi-Yau hypersurface.

        The Hodge numbers are determined by the polytope alone (they are birational
        invariants), so they are the same for every fine regular star triangulation of
        it, and this returns $h^{2,1}$ of the convex hull of the vectors.

        Returns:
            int: The Hodge number $h^{2,1}$.

        Raises:
            NotImplementedError: If
                [`is_gorenstein_fano`][cytools.vector_config.fan.Fan.is_gorenstein_fano]
                is false.

        Examples:
            >>> from cytools import Polytope
            >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            >>> p.triangulate().fan().h21_cy()
            272
        """
        if not self.is_gorenstein_fano():
            raise NotImplementedError()

        return self.vc.conv().h21()

    # generalize flip_linear
    # ----------------------
    def flop_linear(self, *args, **kwargs):
        """Flip along a linear path of heights, tracking the intersection numbers.

        This calls regfans' `Fan.flip_linear` (which performs all the flips along the
        linear homotopy of heights from this fan to a target), with hooks that keep
        track of the intersection numbers of the Calabi-Yau hypersurface. The initial
        fan and every fan reached after a flip get an attribute `_kappa_np` holding the
        pushed-down intersection numbers in the divisor basis, as a NumPy array. These
        are computed directly for the initial fan and then updated through each flop
        with [`flop`][cytools.vector_config.fan.flop], instead of being recomputed.

        Args:
            *args (Any): Positional arguments passed to `flip_linear`.
            **kwargs (Any): Keyword arguments passed to `flip_linear`, such as
                `h_target`, `direction`, `h_init`, `max_N_flips`, and `verbosity`.
                They must not include `hook_init` or `hook_flip`.

        Returns:
            list: The output of `flip_linear`: the status of the homotopy (1 on success,
                or the exception raised), the final heights, the final fan, the
                hyperplanes of its secondary cone, and the number of flips (plus any
                requested records).
        """
        # define the hooks
        # (the intersection numbers are stashed as `_kappa_np`; using `kappa`
        #  would shadow the `kappa`/`intersection_numbers` method on these
        #  instances)
        def hook_init(fan):
            fan._kappa_np = fan.intersection_numbers(
                pushed_down=True, in_basis=True, as_np_array=True
            )

        def hook_flip(fanpre, fanpost, circ):
            fanpost._kappa_np = flop(fanpre, fanpre._kappa_np, circ)

        # call flip_linear with the hooks
        return self.flip_linear(
            *args, hook_init=hook_init, hook_flip=hook_flip, **kwargs
        )


# intersection numbers
# --------------------
# kernels
# (See Fan.intersection_numbers)
def kappa_solve_nxn(pts, known, r):
    """Solve for unknown intersection numbers from linear relations.

    Helper for
    [`Fan.intersection_numbers`][cytools.vector_config.fan.Fan.intersection_numbers].
    Finds the least-squares solution $x$ of `pts.T @ [x, known] = 0`, i.e. of
    $M x = b$ with `M = pts[:r].T` and `b = -pts[r:].T @ known`, via the normal
    equations.

    Args:
        pts (np.ndarray): The vectors, one per row. The first `r` rows correspond to
            the unknowns and the rest to the known values.
        known (np.ndarray): The known intersection numbers, as a column vector.
        r (int): The number of unknowns.

    Returns:
        np.ndarray: The `r` unknown intersection numbers.
    """
    # alternative methods (slower)
    # -------------------
    # solutions, res, *_ = np.linalg.lstsq(
    #    pts[:, 0:N_unknown], -pts[:, N_unknown:] @ known,
    #    rcond=None
    # )
    #
    # solutions, res, *_ = sp.linalg.lstsq(
    #    pts[:, 0:N_unknown], -pts[:, N_unknown:] @ known,
    #    check_finite=False,lapack_driver='gelsy'
    # )
    M = pts[0:r, :].T
    b = -pts[r:, :].T @ known
    return np.linalg.solve(M.T @ M, M.T @ b).flatten()


@njit
def kappa_solve_1x1(pts, known):
    """Solve for one unknown intersection number from linear relations.

    Specialization of [`kappa_solve_nxn`][cytools.vector_config.fan.kappa_solve_nxn]
    for `r=1`, compiled with Numba.

    Args:
        pts (np.ndarray): The vectors, one per row. The first row corresponds to the
            unknown and the rest to the known values.
        known (np.ndarray): The known intersection numbers, as a column vector.

    Returns:
        list[float]: The unknown intersection number.

    Raises:
        ValueError: If the normal equations are singular.
    """
    # solve M@x=b
    # solution x = a/b
    b = -pts[1:, :].T @ known
    M = pts[0:1, :].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M
    # hardcoded 1x1 solve
    if M[0, 0] == 0:
        raise ValueError("Singular matrix")
    x0 = b[0, 0] * (1 / M[0, 0])
    return [x0]


@njit
def kappa_solve_2x2(pts, known):
    """Solve for two unknown intersection numbers from linear relations.

    Specialization of [`kappa_solve_nxn`][cytools.vector_config.fan.kappa_solve_nxn]
    for `r=2`, compiled with Numba.

    Args:
        pts (np.ndarray): The vectors, one per row. The first two rows correspond to
            the unknowns and the rest to the known values.
        known (np.ndarray): The known intersection numbers, as a column vector.

    Returns:
        list[float]: The two unknown intersection numbers.

    Raises:
        ValueError: If the normal equations are singular.
    """
    # solve M@x=b
    b = -pts[2:, :].T @ known
    M = pts[0:2, :].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M

    # hardcoded 2x2 solve
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if det == 0:
        raise ValueError("Singular matrix")

    x0 = (b[0, 0] * M[1, 1] - M[0, 1] * b[1, 0]) / det
    x1 = (M[0, 0] * b[1, 0] - b[0, 0] * M[1, 0]) / det

    # return
    return [x0, x1]


@njit
def kappa_solve_3x3(pts, known):
    """Solve for three unknown intersection numbers from linear relations.

    Specialization of [`kappa_solve_nxn`][cytools.vector_config.fan.kappa_solve_nxn]
    for `r=3`, compiled with Numba.

    Args:
        pts (np.ndarray): The vectors, one per row. The first three rows correspond
            to the unknowns and the rest to the known values.
        known (np.ndarray): The known intersection numbers, as a column vector.

    Returns:
        list[float]: The three unknown intersection numbers.

    Raises:
        ValueError: If the normal equations are singular.
    """
    # solve M@x=b
    b = -pts[3:, :].T @ known
    M = pts[0:3, :].T
    M = np.ascontiguousarray(M)

    # normal equation
    b = M.T @ b
    M = M.T @ M

    # hardcoded 3x3 solve
    det = (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )
    if det == 0:
        raise ValueError("Singular matrix")

    # inverse via cofactor / adjugate
    inv = np.empty((3, 3), dtype=np.float64)
    inv[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) / det
    inv[0, 1] = -(M[0, 1] * M[2, 2] - M[0, 2] * M[2, 1]) / det
    inv[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) / det
    inv[1, 0] = -(M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) / det
    inv[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) / det
    inv[1, 2] = -(M[0, 0] * M[1, 2] - M[0, 2] * M[1, 0]) / det
    inv[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) / det
    inv[2, 1] = -(M[0, 0] * M[2, 1] - M[0, 1] * M[2, 0]) / det
    inv[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) / det

    # multiply inverse with b
    x0 = inv[0, 0] * b[0, 0] + inv[0, 1] * b[1, 0] + inv[0, 2] * b[2, 0]
    x1 = inv[1, 0] * b[0, 0] + inv[1, 1] * b[1, 0] + inv[1, 2] * b[2, 0]
    x2 = inv[2, 0] * b[0, 0] + inv[2, 1] * b[1, 0] + inv[2, 2] * b[2, 0]

    return [x0, x1, x2]


# flops
def minface_dim(config, labels):
    """Return the dimension of the smallest face of the polytope containing some points.

    The polytope is the convex hull of the vector configuration.

    Args:
        config (VectorConfiguration): The vector configuration.
        labels (set[int]): The labels of the points.

    Returns:
        int: The dimension of the smallest face of the convex hull of `config` whose
            labels include all of `labels`.
    """
    # get dim (face must have at least this dimension)
    p = config.conv()
    A = config.vectors(labels)
    dim = np.linalg.matrix_rank([pt.tolist() + [1] for pt in A]) - 1

    # try increasing dimensions
    for facedim in range(dim, config.dim + 1):
        for face in p.faces(facedim):
            if labels.issubset(face.labels):
                return facedim


def curve_to_gv(fan, kappa, circ, verbosity=0):
    """Compute the GV invariant of the curve contracted in a flop.

    The circuit describes a flop between two triangulations of a 4D reflexive
    polytope, and the returned value is the genus-zero Gopakumar-Vafa invariant of
    the flopping curve class of the Calabi-Yau hypersurface. It is determined by the
    signature of the circuit and the dimension of the smallest face of the polytope
    containing it: for signature $(3,2)$ it is 1 if that face is the whole polytope
    and 0 otherwise, while for signature $(2,2)$ it is the triple intersection number
    of the three divisors spanning a cone on one side of the circuit.

    Args:
        fan (Fan): The fan before the flop.
        kappa (np.ndarray): The pushed-down intersection numbers of `fan` in the
            divisor basis, as a dense array.
        circ (regfans.circuits.Circuit): The flopped circuit.
        verbosity (int): The verbosity level.

    Returns:
        int | float: The GV invariant.

    Raises:
        ValueError: If the circuit does not describe a flop (i.e. both sides of its
            signature are not at least 2), or if the flop is not of a supported
            type.
    """
    config = fan.vc

    # check that the signature is of the right type
    if min(circ.signature) != 2:
        raise ValueError(
            f"curve_to_gv needs a flop circuit (min signature 2); "
            f"got {circ.signature}, which has no transport rule for kappa"
        )

    unsigned = set(circ.Z)

    # get the minface_dim of the circuit
    dim = minface_dim(config, unsigned)
    if max(circ.signature) == 3:
        if dim == 4:
            if verbosity >= 1:
                print("(3,2), minface dim=4 -> gv=1")
            return 1
        else:
            if verbosity >= 1:
                print("(3,2), minface dim<4 -> gv=0")
            return 0
    elif max(circ.signature) == 2:
        if dim == 4:
            raise ValueError("Dim=4 minface???")
        else:
            # get a 3-cone on this side
            c = sorted(list(circ.Zneg) + [circ.Zpos[0]])

            if set(c).issubset(fan.vc.divisor_basis):
                c_inds = fan.vc.labels_to_inds(c, ambient_labels=fan.vc.divisor_basis)
                kappa_c = kappa[tuple(c_inds)]
            else:
                c_inds = fan.vc.labels_to_inds(c)
                divisors = fan.vc.gale()[c_inds, :].T
                kappa_c = ((kappa @ divisors[:, 0]) @ divisors[:, 1]) @ divisors[:, 2]

            if verbosity >= 1:
                print(f"(2,2), minface dim<4 -> gv=kappa[cone={c}]={kappa_c}")

            return np.round(kappa_c)
    else:
        raise ValueError


def flop(fan, kappa, circ, verbosity=0):
    r"""Compute the intersection numbers after flopping a circuit.

    Under the flop of a curve class $C$ with GV invariant $n_C$ (see
    [`curve_to_gv`][cytools.vector_config.fan.curve_to_gv]), the intersection
    numbers transform as $\kappa_{ijk} \to \kappa_{ijk} - n_C\, q_i q_j q_k$, where
    $q$ is the normal vector of the circuit restricted to the divisor basis.

    Args:
        fan (Fan): The fan before the flop.
        kappa (np.ndarray): The pushed-down intersection numbers of `fan` in the
            divisor basis, as a dense array.
        circ (regfans.circuits.Circuit): The flopped circuit.
        verbosity (int): The verbosity level.

    Returns:
        np.ndarray: The intersection numbers after the flop, as a new array.
    """
    # get GV
    gv = curve_to_gv(fan, kappa, circ, verbosity - 1)
    if verbosity >= 1:
        print(f"GV = {gv}")

    if gv == 0:
        return kappa.copy()

    n = np.array(circ.normal)
    n = n[fan.vc.divisor_basis_inds]

    # kappa -> kappa - gv*(n outer n outer n)
    # (only the nonzero entries of n contribute, and n is typically very
    #  sparse, so this beats forming the full outer product)
    kappa_flopped = kappa.copy()
    n_nonzero = np.where(n)[0]

    for i, j, k in itertools.product(n_nonzero, repeat=3):
        kappa_flopped[i, j, k] = kappa_flopped[i, j, k] - gv * n[i] * n[j] * n[k]

    return kappa_flopped


# misc
# ----
# give Triangulation a method to directly generate VCs/Fans
def vc(self, include_points_interior_to_facets=None):
    """Return the vector configuration associated to the triangulation.

    This is the (cached) vector configuration returned by `Polytope.vc` for the
    polytope of the triangulation.

    This function is attached to [`Triangulation`][cytools.triangulation.Triangulation]
    as `Triangulation.vc` when `cytools.vector_config` is imported.

    Args:
        include_points_interior_to_facets (bool | None): Whether to include the
            points interior to facets. If not specified, they are included if and
            only if the triangulation uses any of them.

    Returns:
        VectorConfiguration: The associated vector configuration.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> p.triangulate().vc().labels
        (1, 2, 3, 4, 5, 6)
    """
    # default to the vc that contains every point the triangulation uses
    if include_points_interior_to_facets is None:
        include_points_interior_to_facets = not set(self.labels).issubset(
            self.polytope().labels_not_facet
        )

    # get the vc
    vc = self.polytope().vc(
        include_points_interior_to_facets=include_points_interior_to_facets
    )

    return vc


Triangulation.vc = vc


def fan(self, include_points_interior_to_facets=None):
    """Return the fan associated to the triangulation.

    The fan consists of the cones over the simplices of the triangulation that
    contain the origin (i.e. the star of the origin), as a fan over the vector
    configuration of the polytope. For a star triangulation, this is the complete
    fan defining the toric variety.

    This function is attached to [`Triangulation`][cytools.triangulation.Triangulation]
    as `Triangulation.fan` when `cytools.vector_config` is imported.

    Args:
        include_points_interior_to_facets (bool | None): Whether to include the
            points interior to facets in the vector configuration. If not specified,
            they are included if and only if the triangulation uses any of them.

    Returns:
        Fan: The associated fan.

    Raises:
        ValueError: If the origin is not a lattice point of the polytope, if the
            triangulation does not use the origin or has no simplex containing it,
            or if the triangulation uses points interior to facets but
            `include_points_interior_to_facets` is false.

    Examples:
        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> f = p.triangulate().fan()
        >>> f.is_fine(), f.is_regular()
        (True, True)
        >>> f.cones()[:3]
        ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 6))
    """
    # the fan is the star of the origin, so the triangulation must contain it.
    # note that triangulating without the origin is legal (it just silently
    # turns off make_star), so this is the first place it can be rejected
    origin = self.polytope().label_origin
    if origin is None:
        raise ValueError(
            "Cannot construct a fan: the origin is not a lattice point of the "
            "polytope, so the star of the origin is undefined."
        )
    if origin not in self.labels:
        raise ValueError(
            f"Cannot construct a fan: the triangulation omits the origin "
            f"(label {origin}), so the star of the origin is empty. If the "
            f"triangulation was built with an explicit `points` argument, "
            f"include label {origin} in it."
        )

    # default to the vc that contains every point the triangulation uses
    if include_points_interior_to_facets is None:
        include_points_interior_to_facets = not set(self.labels).issubset(
            self.polytope().labels_not_facet
        )

    # get the vc
    vc = self.polytope().vc(
        include_points_interior_to_facets=include_points_interior_to_facets
    )

    # the cells index the vc by label, so the vc must contain each of them
    missing = sorted(set(self.labels) - {origin} - set(vc.labels))
    if missing:
        raise ValueError(
            f"Cannot construct a fan: the triangulation uses points {missing}, "
            f"which are interior to facets and so are absent from the vector "
            f"configuration. Pass include_points_interior_to_facets=True."
        )

    # build the star fan: keep only simplices containing the origin and drop
    # the origin label from each. the vc above already carries these same
    # labels (minus the origin), so the cells index it directly (no remap).
    cells = [
        sorted(x for x in simp if x != origin)
        for simp in self.simplices().tolist()
        if origin in simp
    ]
    if len(cells) == 0:
        raise ValueError(
            f"Cannot construct a fan: no simplex of the triangulation contains "
            f"the origin (label {origin}), so the star of the origin is empty. "
            f"This happens for non-fine triangulations that leave the origin "
            f"unused."
        )

    fan = vc.subdivide(cells=cells)
    return fan


Triangulation.fan = fan
