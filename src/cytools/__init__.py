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

"""CYTools: a software package for analyzing Calabi-Yau manifolds.

The most commonly used classes and functions, such as
[`Polytope`][cytools.polytope.Polytope] and
[`fetch_polytopes`][cytools.utils.fetch_polytopes], can be imported directly
from the package root.
"""

# Make the main classes and function accessible from the root of CYTools.
import os
import threading

from cytools.polytope import Polytope
from cytools.h_polytope import HPolytope
from cytools.cone import Cone
from cytools.utils import read_polytopes, fetch_polytopes
from . import ntfe, vector_config

# the public surface of the package root; these are re-exports, so listing them
# here is what marks them as intentionally imported-but-unused
__all__ = [
    "Polytope",
    "HPolytope",
    "Cone",
    "read_polytopes",
    "fetch_polytopes",
    "ntfe",
    "vector_config",
]

# Latest version
version = "1.4.12"
versions_with_serious_bugs = []

# Check for more recent versions of CYTools
def check_for_updates():
    """Check whether a newer version of CYTools is available.

    The version number on the `main` branch of the CYTools GitHub repository
    is compared with the installed one. A message is printed if a
    newer version is available, and a warning is printed if the installed
    version is known to contain a serious bug. Nothing is printed otherwise.

    This performs a network request, so it blocks until the request completes
    or times out (after 2 seconds). On import, CYTools instead runs it in a
    background daemon thread (see
    [`start_update_check`][cytools.start_update_check]), so that
    `import cytools` never waits on the network. Any failure (no network, DNS
    failure, timeout, ...) is silently ignored.

    Examples:
        We check for updates of CYTools. This is done automatically on import,
        so there is usually no need to do this.

        >>> import cytools
        >>> cytools.check_for_updates()
    """
    from ast import literal_eval
    import requests

    checked_version = False
    checked_bugs = False

    try:
        # get updated __init__ from github
        p = requests.get(
            "https://raw.githubusercontent.com/"
            + "LiamMcAllisterGroup/cytools/main/src/cytools/"
            + "__init__.py",
            timeout=2,
        )

        # find/check the version in this file
        for line in p.text.split("\n"):
            if (not checked_version) and ("version =" in line):
                checked_version = True

                # parse version
                latest_ver = tuple(int(c) for c in line.split('"')[1].split("."))
                ver = tuple(int(c) for c in version.split("."))

                # check
                if latest_ver <= ver:
                    continue

                # local version is old -> print warning
                print(
                    "\nInfo: A more recent version of CYTools is available: "
                    f"v{ver[0]}.{ver[1]}.{ver[2]} -> "
                    f"v{latest_ver[0]}.{latest_ver[1]}.{latest_ver[2]}.\n"
                    "We recommend upgrading before continuing.\n"
                    "On Linux and macOS you can update CYTools by running "
                    "'cytools --update'\n"
                    "and on Windows you can do this by running the updater "
                    "tool.\n"
                )

            elif (not checked_bugs) and ("versions_with_serious_bugs =" in line):
                checked_bugs = True
                bad_versions = literal_eval(line.split("=")[1].strip())
                if version in bad_versions:
                    print(
                        "\n****************************\n"
                        "Warning: This version of CYTools contains a serious"
                        " bug. Please upgrade to the latest version.\n"
                        "****************************\n"
                    )

            if checked_version and checked_bugs:
                break
    except Exception:
        pass



def start_update_check():
    """Start an update check in a background thread.

    This runs [`check_for_updates`][cytools.check_for_updates] in a background
    daemon thread and returns immediately. It is what runs on import, so that
    `import cytools` never blocks on the network. Since the thread is a daemon,
    it never keeps the interpreter alive, and any error it encounters is
    silently ignored.

    The check is skipped entirely if the `CYTOOLS_NO_UPDATE_CHECK` environment
    variable is set to a non-empty value, which is useful on clusters and
    air-gapped machines.

    Returns:
        threading.Thread | None: The thread performing the check, or `None` if
            the check is disabled.

    Examples:
        We start a background update check. This is done automatically on
        import, so there is usually no need to do this.

        >>> import cytools
        >>> thread = cytools.start_update_check()
    """
    if os.environ.get("CYTOOLS_NO_UPDATE_CHECK"):
        return None

    thread = threading.Thread(
        target=check_for_updates,
        name="cytools-update-check",
        daemon=True,
    )
    thread.start()
    return thread


# Automatically check for updates on import. This runs in a background daemon
# thread so that it can never block or slow down the import, and it can be
# disabled altogether via the CYTOOLS_NO_UPDATE_CHECK environment variable
# (e.g. for HPC/air-gapped use).
_update_check_thread = start_update_check()
