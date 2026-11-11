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

"""Miscellaneous utilities that are not needed for mainline CYTools.

This includes mixed-radix integer conversions and the saving and loading of
compressed pickle files (by default in the CYTools cache directory).
"""

# 'standard' imports
import gzip
import os
import zlib

# 3rd party imports
import pickle
from platformdirs import user_cache_dir

# typing
from typing import Any


# numbers
# -------
def to_base10(c: list[int], B: list[int]) -> int:
    """Convert a number from a mixed-radix representation to an integer.

    The digits are ordered from most to least significant, and digit `c[i]` is
    expected to lie in `range(B[i])`. The first radix `B[0]` is not needed for the
    conversion. This is the inverse of
    [`from_base10`][cytools.helpers.misc.from_base10].

    Args:
        c (list[int]): The digits.
        B (list[int]): The radix of each digit.

    Returns:
        int: The integer.

    Examples:
        >>> from cytools.helpers.misc import to_base10
        >>> to_base10([1,0,1], [2,2,2])
        5
        >>> to_base10([1,2], [3,4])
        6
    """
    result = 0
    multiplier = 1
    for c_i, B_i in zip(reversed(c), reversed(B)):
        result += int(c_i) * multiplier
        multiplier *= B_i
    return result


def from_base10(n: int, B: list[int]) -> list[int]:
    """Convert an integer to a mixed-radix representation.

    The digits are ordered from most to least significant, with digit `i` in
    `range(B[i])`. If the integer is not smaller than the product of the radices, the
    most significant digit wraps around. This is the inverse of
    [`to_base10`][cytools.helpers.misc.to_base10].

    Args:
        n (int): The non-negative integer.
        B (list[int]): The radix of each digit.

    Returns:
        list[int]: The digits.

    Examples:
        >>> from cytools.helpers.misc import from_base10
        >>> from_base10(5, [2,2,2])
        [1, 0, 1]
        >>> from_base10(6, [3,4])
        [1, 2]
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
    """Load an object from a gzip-compressed pickle file.

    If the file is broken (not valid gzip, truncated, or not unpicklable), a warning
    is printed, the file is deleted, and `None` is returned. Objects of custom
    classes may fail to load.

    Args:
        fname (str): The file name. If it has no extension, `.p` is appended.
        path (str): The directory containing the file. By default, the CYTools cache
            directory is used.

    Returns:
        Any: The loaded object, or `None` if the file does not exist or is broken.
    """
    if "." not in fname:
        fname += ".p"
    file = os.path.join(path, fname)

    if os.path.isfile(file):
        try:
            with gzip.open(file, "rb") as f:
                return pickle.load(f)
        except (
            EOFError,
            pickle.UnpicklingError,
            # a file that isn't valid gzip at all raises BadGzipFile (an
            # OSError), and a truncated/corrupted deflate stream raises
            # zlib.error; both are 'broken cache', not fatal errors
            gzip.BadGzipFile,
            zlib.error,
        ) as e:
            print(f"Warning: cache {file} is broken ({e}), removing it...")
            try:
                os.remove(file)
            except OSError:
                pass
            return None
    else:
        return None


def save_zipped_pickle(
    obj: Any,
    fname: str,
    path: str = cache_dir,
    protocol: int = pickle.DEFAULT_PROTOCOL,
):
    """Save an object to a gzip-compressed pickle file.

    The object is first written to a temporary file which then replaces the target
    file atomically, so concurrent readers never see a partially written file.

    Args:
        obj (Any): The object to save.
        fname (str): The file name. If it has no extension, `.p` is appended.
        path (str): The directory to save the file in. By default, the CYTools cache
            directory is used.
        protocol (int): The pickle protocol to use.
    """
    if "." not in fname:
        fname += ".p"
    file = os.path.join(path, fname)

    # Write to a per-process temp file, then atomically rename into place. This
    # keeps concurrent writers (e.g. a 128-way array sweep all sharing one cache)
    # and processes killed mid-write from ever leaving a truncated/zero-padded
    # file: readers always see either the old or the new complete file.
    tmp = f"{file}.tmp.{os.getpid()}"
    try:
        with gzip.open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol)
        os.replace(tmp, file)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
