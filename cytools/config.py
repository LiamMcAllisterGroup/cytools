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

import warnings
import os

# The number of CPU threads to use in some computations, such as finding the extremal rays of a cone.
# When set to None, then it uses all available threads.
n_threads = None

# Paths to external software in the Docker image. These can be modified when
# using a custom installation.
cgal_path = "/usr/local/bin/"
topcom_path = "/usr/bin/"
palp_path = "/usr/bin/"

# Mosek license
_mosek_license = f"/home/{'root' if os.geteuid()==0 else 'cytools'}/mounted_volume/mosek/mosek.lic"
_mosek_is_activated = None
_mosek_error = ""
def check_mosek_license(silent=False):
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
    global _mosek_license
    os.environ["MOSEKLM_LICENSE_FILE"] = _mosek_license
    global _mosek_error
    global _mosek_is_activated
    try:
        import mosek
        mosek.Env().Task(0,0).optimize()
        _mosek_is_activated = True
        if not silent:
            print("Mosek was successfully activated.")
    except mosek.Error as e:
        _mosek_error = ("Info: Mosek is not activated. "
                        "An alternative optimizer will be used.\n"
                        f"Error encountered: {e}")
        _mosek_is_activated = False
    except:
        _mosek_error = ("Info: There was a problem with Mosek. "
                        "An alternative optimizer will be used.")
        _mosek_is_activated = False
    if not silent:
        print(_mosek_error)

def mosek_is_activated():
    global _mosek_error
    global _mosek_is_activated
    global _printed_mosek_error
    if _mosek_is_activated is None:
        check_mosek_license(silent=True)
    return _mosek_is_activated

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
    global _mosek_license
    _mosek_license = path
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
    We enable the experimental features.
    ```python {2}
    import cytools
    cytools.config.enable_experimental_features()
    ```
    """
    global _exp_features_enabled
    _exp_features_enabled = True
    warnings.warn("\n**************************************************************\n"
                  "Warning: You have enabled experimental features of CYTools.\n"
                  "Some of these features may be broken or not fully tested,\n"
                  "and they may undergo significant changes in future versions.\n"
                  "**************************************************************\n")
