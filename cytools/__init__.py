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

# Make the main classes and function accessible from the root of CYTools.
from cytools.polytope import Polytope
from cytools.cone import Cone
from cytools.utils import read_polytopes, fetch_polytopes

# Latest version
version = "1.0.0"
versions_with_serious_bugs = []

# Check for more recent versions of CYTools
def check_for_updates():
    """
    **Description:**
    Checks for updates of CYTools. It prints a message if a new version is
    avaiable, and displays a warning if the current version has a serious bug.

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    We check for updates of CYTools. This is done automatically, so there is
    usually no need to do this.
    ```python {2}
    import cytools
    cytools.check_for_updates()
    ```
    """
    from ast import literal_eval
    import requests
    try:
        p = requests.get("https://raw.githubusercontent.com/"
                         + "LiamMcAllisterGroup/cytools/main/cytools/"
                         + "__init__.py",
                         timeout=2)
        checked_version = False
        checked_bugs = False
        for l in p.text.split("\n"):
            if not checked_version and "version =" in l:
                checked_version = True
                latest_ver = tuple(int(c) for c in l.split("\"")[1].split("."))
                ver = tuple(int(c) for c in version.split("."))
                if latest_ver <= ver:
                    continue
                print("\nInfo: A more recent version of CYTools is available: "
                      f"v{ver[0]}.{ver[1]}.{ver[2]} -> "
                      f"v{latest_ver[0]}.{latest_ver[1]}.{latest_ver[2]}.\n"
                      "We recommend upgrading before continuing.\n"
                      "On Linux and macOS you can update CYTools by running 'cytools --update'\n"
                      "and on Windows you can do this by running the updater tool.\n")
            elif not checked_bugs and "versions_with_serious_bugs =" in l:
                checked_bugs = True
                bad_versions = literal_eval(l.split("=")[1].strip())
                if version in bad_versions:
                    print("\n****************************\n"
                          "Warning: This version of CYTools contains a serious"
                          " bug. Please upgrade to the latest version.\n"
                          "****************************\n")
            if checked_version and checked_bugs:
                break
    except:
       pass
check_for_updates()
