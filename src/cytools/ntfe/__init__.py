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

r"""Non-two-face-equivalent (NTFE) triangulations.

Two fine, regular, star triangulations (FRSTs) of a 4D reflexive polytope are
*two-face equivalent* if they restrict to the same triangulation on every
2-face of the polytope. The Calabi-Yau threefold hypersurfaces obtained from
two-face-equivalent FRSTs are diffeomorphic (by Wall's theorem), so for many
purposes it suffices to construct a single FRST for each class of
non-two-face-equivalent (NTFE) triangulations. This subpackage implements the
algorithm of [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) for doing so
without enumerating all FRSTs.

Every fine, regular triangulation of the polytope is induced by a height
vector, with one entry per lattice point. The triangulation it induces on a
2-face is determined by the heights of the points of that 2-face, and it is a
given fine, regular triangulation (FRT) of the 2-face exactly when the heights
lie in the (lifted) secondary cone of that FRT. Choosing one FRT for each
2-face and intersecting the corresponding cones gives an *expanded secondary
cone*. The choices whose expanded secondary cone is solid are precisely the
NTFEs, and any height vector in the interior of such a cone gives a
representative FRT, which can be made star by lowering the height of the
origin.

The functions fit together as follows:

1. [`face_triangs`][cytools.ntfe.face_triangulations.face_triangs] enumerates
   (or, for large 2-faces, samples) the FRTs of each 2-face.
2. [`triangface_ineqs`][cytools.ntfe.ntfe.triangface_ineqs] computes the
   inequalities of the secondary cone of each 2-face FRT.
3. [`ntfe_hypers`][cytools.ntfe.ntfe.ntfe_hypers] stacks these inequalities
   for each choice of 2-face FRTs, and
   [`ntfe_cones`][cytools.ntfe.ntfe.ntfe_cones] wraps them into
   [`Cone`][cytools.cone.Cone] objects.
4. [`ntfe_frts`][cytools.ntfe.ntfe.ntfe_frts] and
   [`ntfe_frsts`][cytools.ntfe.ntfe.ntfe_frsts] find a height vector in each
   solid cone and return the corresponding FR(S)Ts. When all NTFEs are
   requested, they do so with a depth-first search that prunes infeasible
   partial choices early.

For a single choice of 2-face triangulations (possibly leaving some 2-faces
free),
[`cone_of_permissible_heights`][cytools.ntfe.ntfe.cone_of_permissible_heights]
returns the corresponding cone, and
[`triangfaces_to_frt`][cytools.ntfe.ntfe.triangfaces_to_frt] and
[`triangfaces_to_frst`][cytools.ntfe.ntfe.triangfaces_to_frst] return a
triangulation that realizes it.
[`expanded_secondary_fan`][cytools.ntfe.ntfe.expanded_secondary_fan] returns
the union of all expanded secondary cones, i.e. the cone of heights that induce
FRTs of every 2-face.

Importing `cytools` attaches all of these functions except
`cone_of_permissible_heights` to [`Polytope`][cytools.polytope.Polytope] as
methods of the same name, which is how they are normally used.

Warning:
    Height vectors are indexed by point label, so these functions assume that
    the polytope uses its default labels $0, 1, \ldots, n-1$ (as is the case
    unless custom labels are passed to the constructor).

Examples:
    We find all NTFE FRSTs of a polytope whose 2-faces have 36 combinations of
    FRTs, of which 30 extend to a triangulation of the full polytope.

    >>> from cytools import Polytope
    >>> p = Polytope([[1,-1,-2,1],[1,-1,0,-1],[-1,0,1,1],[-1,2,1,-1],[1,0,0,0],
    ...               [-1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,1,-1]])
    >>> [len(ft) for ft in p.face_triangs()]  # number of FRTs of each 2-face
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1]
    >>> p.n_2face_triangs()
    36
    >>> frsts = p.ntfe_frsts()
    >>> len(frsts)
    30
"""

from cytools.ntfe.ntfe import (
    cone_of_permissible_heights,
    expanded_secondary_fan,
    ntfe_cones,
    ntfe_frsts,
    ntfe_frts,
    ntfe_hypers,
    triangface_ineqs,
    triangfaces_to_frst,
    triangfaces_to_frt,
)

__all__ = [
    "cone_of_permissible_heights",
    "expanded_secondary_fan",
    "ntfe_cones",
    "ntfe_frsts",
    "ntfe_frts",
    "ntfe_hypers",
    "triangface_ineqs",
    "triangfaces_to_frst",
    "triangfaces_to_frt",
]

# imported for its side effect: attaches methods to Polytope
from cytools.ntfe import face_triangulations as face_triangulations
