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

"""Vector configurations and fans.

This subpackage provides
[`VectorConfiguration`][cytools.vector_config.vectorconfiguration.VectorConfiguration],
a configuration of integral vectors (analogous to
[`Polytope`][cytools.polytope.Polytope] for point configurations), and
[`Fan`][cytools.vector_config.fan.Fan], a subdivision of a vector configuration into
cones (analogous to [`Triangulation`][cytools.triangulation.Triangulation]), with
toric methods such as intersection numbers and Mori and Kähler cones. Both classes
build on the [regfans](https://github.com/natemacfadden/regfans) package.

Importing this subpackage also adds `vc` methods to
[`Polytope`][cytools.polytope.Polytope], [`Cone`][cytools.cone.Cone] and
[`Triangulation`][cytools.triangulation.Triangulation], and a `fan` method to
[`Triangulation`][cytools.triangulation.Triangulation].
"""

from cytools.vector_config.fan import Fan
from cytools.vector_config.vectorconfiguration import VectorConfiguration

__all__ = ["Fan", "VectorConfiguration"]
