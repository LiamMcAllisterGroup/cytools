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

"""F-theory uplifts of Calabi-Yau orientifolds.

This subpackage constructs O3/O7 orientifolds of Calabi-Yau hypersurfaces in toric
varieties ([`CY_orientifold`][cytools.f_theory.FT_CY.CY_orientifold]) and their
F-theory uplifts, described as toric complete intersections
([`F_Theory_Uplift`][cytools.f_theory.FT_CY.F_Theory_Uplift]). It also provides
functions to generate orientifolds and uplifts from the Kreuzer-Skarke database
([`fetch_orientifolds`][cytools.f_theory.FT_CY.fetch_orientifolds],
[`fetch_F_Theory_uplifts`][cytools.f_theory.FT_CY.fetch_F_Theory_uplifts]) and to
read a precomputed dataset of nef-partition uplifts
([`fetch_nef_partition_uplifts`][cytools.f_theory.FT_CY.fetch_nef_partition_uplifts]).
"""

from cytools.f_theory.FT_CY import (
    CY_orientifold,
    F_Theory_Uplift,
    fetch_F_Theory_uplifts,
    fetch_nef_partition_uplifts,
    fetch_orientifolds,
)

__all__ = [
    "CY_orientifold",
    "F_Theory_Uplift",
    "fetch_F_Theory_uplifts",
    "fetch_nef_partition_uplifts",
    "fetch_orientifolds",
]
