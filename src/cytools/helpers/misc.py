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
#
# -----------------------------------------------------------------------------
# Description:  Miscellaneous utilities not needed for mainline CYTools.
# -----------------------------------------------------------------------------

# 'standard' imports
import gzip
import os

# 3rd party imports
import pickle
from platformdirs import user_cache_dir

# typing
from typing import Any


# numbers
# -------
def to_base10(c: list[int], B: list[int]) -> int:
    """
    **Description:**
    Converts a number given in components w.r.t. some bases to an integer base
    10.

    **Arguments:**
    - `c`: A list of the components.
    - `B`: A list of the bases.

    **Returns:**
    The integer in base-10.
    """
    result = 0
    multiplier = 1
    for c_i, B_i in zip(reversed(c), reversed(B)):
        result += int(c_i) * multiplier
        multiplier *= B_i
    return result


def from_base10(n: int, B: list[int]) -> list[int]:
    """
    **Description:**
    Split an integer in base 10 to components components w.r.t. some bases.

    **Arguments:**
    - `n`: The integer in base 10.
    - `B`: A list of the bases.

    **Returns:**
    The bases
    """
    c = []
    for B_i in reversed(B):
        c.append(n % B_i)
        n //= B_i
    return list(reversed(c))


# loading/saving zipped pickle files
# ----------------------------------
# default directory to save to
cache_dir = user_cache_dir("CYTools", "CYTools")
os.makedirs(cache_dir, exist_ok=True)


# saving/loading functions
def load_zipped_pickle(fname, path=cache_dir):
    """
    **Description:**
    Loads zipped pickle files.

    Custom/atypical classes may fail to load.

    **Arguments:**
    - `fname`: Filename.
    - `path`: Path to file.

    **Returns:**
    Data from file.
    """
    if "." not in fname:
        fname += ".p"
    file = os.path.join(path, fname)

    if os.path.isfile(file):
        try:
            with gzip.open(file, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: cache {file} is broken ({e}), removing it...")
            os.remove(file)
            return None
    else:
        return None


def save_zipped_pickle(
    obj: Any,
    fname: str,
    path: str = cache_dir,
    protocol: int = pickle.DEFAULT_PROTOCOL,
):
    """
    **Description:**
    Saves zipped pickle files.

    **Arguments:**
    - `obj`: The object to save.
    - `fname`: Filename.
    - `path`: Path to file.
    - `protocol`: Protocol to use for saving the file. Defaults to -1.

    **Returns:**
    Nothing.
    """
    if "." not in fname:
        fname += ".p"
    file = os.path.join(path, fname)

    with gzip.open(file, "wb") as f:
        pickle.dump(obj, f, protocol)
