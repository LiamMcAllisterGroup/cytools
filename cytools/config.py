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
palp_path = "/usr/bin/"

# Mosek license
mosek_license = "/opt/cytools/external/mosek/mosek.lic"
def check_mosek_license():
    """
    **Description:**
    Checks if the Mosek license is valid. If it is not, it prints the reason.

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    The Mosek license should be automatically checked, but it can also be
    checked as follows.
    ```python {2}
    import cytools
    cytools.config.check_mosek_license()
    # It will print an error if it is not working, and if nothing is printed then it is working correctly
    ```
    """
    os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license
    global mosek_is_activated
    try:
        import mosek
        mosek.Env().Task(0,0).optimize()
        mosek_is_activated = True
    except mosek.Error as e:
        print("\nInfo: Mosek is not activated. "
              "An alternative optimizer will be used.\n"
              f"Error encountered: {e}\n")
        mosek_is_activated = False
    except:
        print("\nInfo: There was a problem with Mosek. "
              "An alternative optimizer will be used.\n")
        mosek_is_activated = False
check_mosek_license()

def set_mosek_path(path):
    """
    **Description:**
    Sets a custom path to the Mosek license. This is useful if the Docker image
    was built without the license, but it is stored somewhere in your computer.
    The license will be checked after the new path is set.

    **Arguments:**
    - `path` *(str)*: The path to the Mosek license. Note that the mounted
      directory on the Docker container is `/home/cytools/mounted_volume/`.

    **Returns:**
    Nothing.

    **Example:**
    We set the path to the original one for illustation purposes, and show how
    to set the path to a directory on the host computer.
    ```python {2,3}
    import cytools
    cytools.config.set_mosek_path("/opt/cytools/external/mosek/mosek.lic") # Original path
    cytools.config.set_mosek_path("/home/cytools/mounted_volume/[path-to-license]") # If not in the Docker image
    ```
    """
    global mosek_license
    mosek_license = path
    check_mosek_license()

# Lock experimental features by default.
_exp_features_enabled = False

def enable_experimental_features():
    """
    **Description:**
    Enables the experimental features of CYTools. For more information read the
    [experimental features page](./experimental).

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    We enable the experimetal features.
    ```python {2}
    import cytools
    cytools.config.enable_experimental_features()
    ```
    """
    global _exp_features_enabled
    _exp_features_enabled = True
    print("\n**************************************************************\n"
          "Warning: You have enabled experimental features of CYTools.\n"
          "Some of these features may be broken or not fully tested,\n"
          "and they may undergo significant changes in future versions.\n"
          "**************************************************************\n")
