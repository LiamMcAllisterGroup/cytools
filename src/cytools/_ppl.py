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
# CYTools.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Imports ppl and undoes its side effect on the FPU.
# -----------------------------------------------------------------------------

# ppl sets the FPU rounding mode when it loads, which silently changes the
# result of unrelated floating-point work elsewhere in the process. Importing it
# here, and resetting the mode immediately, confines that to one place: modules
# that need ppl import this instead, so the reset always runs directly after the
# load and before any later import gets to do arithmetic.

import ctypes

import ppl  # noqa: F401  (re-exported; callers do `from cytools._ppl import ppl`)

ctypes.CDLL(None).fesetround(0)  # FE_TONEAREST
