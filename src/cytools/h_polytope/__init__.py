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

"""Polytopes defined by hyperplane inequalities.

This subpackage provides [`HPolytope`][cytools.h_polytope.h_polytope.HPolytope], which
constructs a lattice polytope from an H-representation (a list of linear
inequalities), either as the integer hull of the possibly rational polytope or by
rescaling it into a lattice polytope.
"""

from cytools.h_polytope.h_polytope import HPolytope

__all__ = ["HPolytope"]
