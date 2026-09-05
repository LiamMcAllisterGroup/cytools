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
