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
This module contains various configuration variables for experimental features
and custom installations.
"""

import os

# Paths to external software in the Docker image. These can be modified when
# using a custom installation.
cgal_path = "/usr/local/bin/"
topcom_path = "/usr/bin/"
palp_path = "/cytools-install/external/palp/"

# Mosek license
mosek_license = "/cytools-install/external/mosek/mosek.lic"
def check_mosek_license():
    os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license
    global mosek_is_activated
    try:
        import mosek
        mosek.Env().Task(0,0).optimize()
        mosek_is_activated = True
    except:
        print("Info: Mosek is not activated. "
              "An alternative optimizer will be used.")
        mosek_is_activated = False
check_mosek_license()

# Lock experimental features unless enabled by the user.
enable_experimental_features = False
