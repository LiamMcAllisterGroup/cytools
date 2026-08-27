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
# Description:  This module contains a stack of sparse blocks along with
#               some helpers.
# -----------------------------------------------------------------------------

# 'standard' imports
# 3rd party imports
import numpy as np
import scipy.sparse as sp

# CYTools imports
from cytools.helpers import misc

# typing
from numpy.typing import ArrayLike
from typing import Union

numeric = Union[int, float, np.number]
# helpers
# -------
def flatten_top(
    arr: ArrayLike, as_list: bool = True, N: int = 1
) -> "list or np.array":
    """
    **Description:**
    Flatten the top level (axis=0) of an array.

    **Arguments:**
    - `arr`: The array to flatten. Can be ragged/have unequal depths.
    - `as_list`: Whether to return a list of elements (True) or a numpy array
        (False).
    - `N`: How many levels to flatten, from the top.

    **Returns:**
    *(list or np.array)* lis, but with the top level flattened.

    **Examples:**
    >>> A = np.asarray(range(2**3)).reshape(2,2,2)
    >>> flatten_top(A)
    flatten_top: You really should use .reshape instead...
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    >>> flatten_top(A.tolist())
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    >>> flatten_top(A.tolist(), N=2)
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    if N > 1:
        return flatten_top(
            flatten_top(arr, as_list=as_list, N=1), as_list=as_list, N=N - 1
        )
    else:
        if isinstance(arr, np.ndarray):
            print("flatten_top: You really should use .reshape instead...")

        # we convert elements to lists if they are np arrays
        flattened = [
            ele.tolist() if isinstance(ele, np.ndarray) else ele
            for row in arr
            for ele in row
        ]
        if as_list:
            return flattened
        else:
            return np.asarray(flattened)


# Secondary cone hyperplanes are sparse: a row has <= d+2 nonzeros over an
# ambient dimension (# of lattice points) that can reach the hundreds, so
# blocks are held as CSR. Dense does not fit at scale.
def csr_rows(cols, vals, width, dtype=np.int16):
    """
    **Description:**
    Build a CSR block from equal-shaped 2D arrays of column indices and
    values, one row each. Zero values are dropped.

    **Arguments:**
    - `cols`: Column index of every entry.
    - `vals`: The corresponding values.
    - `width`: The ambient dimension.
    - `dtype`: The stored value type.

    **Returns:**
    The block, as a CSR matrix.
    """
    cols = np.asarray(cols)
    vals = np.asarray(vals)
    # len, not size: a (k, 0) input is k rows that happen to have no entries,
    # and dropping them would silently lose rows
    if len(vals) == 0:
        return sp.csr_matrix((0, width), dtype=dtype)

    keep = vals != 0
    indptr = np.zeros(len(vals) + 1, dtype=np.int64)
    np.cumsum(keep.sum(axis=1), out=indptr[1:])
    return sp.csr_matrix((vals[keep].astype(dtype), cols[keep], indptr),
                         shape=(len(vals), width))


def csr_dicts(rows, width, dtype=np.int16):
    """
    **Description:**
    Build a CSR block from a list of {column: value} rows.

    **Arguments:**
    - `rows`: The rows.
    - `width`: The ambient dimension.
    - `dtype`: The stored value type.

    **Returns:**
    The block, as a CSR matrix.
    """
    if not len(rows):
        return sp.csr_matrix((0, width), dtype=dtype)

    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=indptr[1:])
    cols = np.fromiter((c for r in rows for c in r), dtype=np.int64,
                       count=int(indptr[-1]))
    vals = np.fromiter((v for r in rows for v in r.values()), dtype=dtype,
                       count=int(indptr[-1]))
    return sp.csr_matrix((vals, cols, indptr), shape=(len(rows), width))


def csr_stack(blocks, width, dtype=np.int16):
    """
    **Description:**
    Concatenate CSR blocks, skipping empty ones.

    **Arguments:**
    - `blocks`: The blocks.
    - `width`: The ambient dimension, used when nothing is left to stack.
    - `dtype`: The stored value type.

    **Returns:**
    The stacked block.
    """
    blocks = [b for b in blocks if b.shape[0]]
    if not blocks:
        return sp.csr_matrix((0, width), dtype=dtype)
    return sp.vstack(blocks, format="csr")


def csr_unique_rows(mat):
    """
    **Description:**
    Drop duplicate rows, keeping the first occurrence of each.

    **Arguments:**
    - `mat`: A CSR matrix.

    **Returns:**
    The matrix with duplicate rows removed.
    """
    if mat.shape[0] < 2:
        return mat

    mat.sum_duplicates()
    keys = [(tuple(mat.indices[a:b]), tuple(mat.data[a:b]))
            for a, b in zip(mat.indptr[:-1], mat.indptr[1:])]
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return mat[keep] if len(keep) < mat.shape[0] else mat


class CSR_stack:
    """
    This class describes a stack of sparse blocks, organized as a list of
    options together with a choice of one option per position:
        options = [ [top_block_option1, top_block_option2, ...],
                    ...
                    [bot_block_option1, bot_block_option2, ...]]
        choices = [i_top_block, ..., i_bot_block]
    Nothing is concatenated until it has to be, which is the point: the
    enumeration builds one stack per candidate and discards most of them.

    **Arguments:**
    - `options`: The possible blocks. options[i] lists every block that can
        sit in the ith position.
    - `choices`: Which block to take from each position.
    - `choice_bounds`: The number of options at each position.
    - `iter_densely`: Whether iteration yields dense rows or sparse ones.
    """

    def __init__(
        self,
        options: "[[sp.csr_matrix]]",
        choices: "[int]",
        choice_bounds: "[int]",
        iter_densely: bool = False,
    ) -> None:
        self._options = options
        if isinstance(choices, int):
            self._choices = choices
        else:
            self._choices = misc.to_base10(choices, choice_bounds)
        self._choice_bounds = choice_bounds
        self.iter_densely = iter_densely

    def __repr__(self) -> str:
        return f"CSR_stack(shape={tuple(self.shape)})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def choices(self) -> "list[int]":
        return misc.from_base10(self._choices, self._choice_bounds)

    def _blocks(self):
        for i, opts in zip(self.choices, self._options):
            yield opts[i]

    @property
    def dtype(self) -> np.dtype:
        return self._options[0][0].dtype

    @property
    def width(self) -> int:
        return self._options[0][0].shape[1]

    def __len__(self) -> int:
        if not hasattr(self, "_len"):
            self._len = sum(b.shape[0] for b in self._blocks())
        return self._len

    @property
    def shape(self) -> tuple:
        return (len(self), self.width)

    @property
    def is_empty(self) -> bool:
        if not hasattr(self, "_is_empty"):
            self._is_empty = not any(b.shape[0] for b in self._blocks())
        return self._is_empty

    def __getitem__(self, idx):
        # a single row comes back as {column: value}, as the LIL stack did
        if isinstance(idx, tuple):
            row = self[idx[0]]
            return row.get(idx[1], 0)

        if idx < 0:
            raise IndexError("CSR_stack: negative indexing not allowed")
        for block in self._blocks():
            n = block.shape[0]
            if idx < n:
                lo, hi = block.indptr[idx], block.indptr[idx + 1]
                return dict(zip(block.indices[lo:hi].tolist(),
                                block.data[lo:hi].tolist()))
            idx -= n
        raise IndexError("CSR_stack: list index out of range")

    def __iter__(self):
        if self.iter_densely:
            return iter(self.dense())
        return (self[i] for i in range(len(self)))

    def __array__(self, dtype: np.dtype = None, copy: bool = None) -> np.array:
        # the order and defaults are fixed by the numpy protocol, which calls
        # this as __array__(dtype, copy)
        return np.array(self.dense(), dtype=dtype, copy=copy)

    def tocsr(self) -> "sp.csr_matrix":
        """
        **Description:**
        Concatenate the chosen blocks into one CSR matrix.

        **Arguments:**
        None.

        **Returns:**
        The stacked block.
        """
        blocks = [b for b in self._blocks() if b.shape[0]]
        if not blocks:
            return sp.csr_matrix((0, self.width), dtype=self.dtype)
        return sp.vstack(blocks, format="csr")

    def dense(self, tocopy: bool = False) -> ArrayLike:
        """
        **Description:**
        Return a dense version of the stack, with duplicate rows removed.

        **Arguments:**
        - `tocopy`: Whether to return a copy.

        **Returns:**
        The dense array.
        """
        if not hasattr(self, "_arr_dense"):
            mat = self.tocsr()
            mat.sum_duplicates()
            keys, seen, keep = [], set(), []
            for a, b in zip(mat.indptr[:-1], mat.indptr[1:]):
                keys.append((tuple(mat.indices[a:b]), tuple(mat.data[a:b])))
            for i, k in enumerate(keys):
                if k not in seen:
                    seen.add(k)
                    keep.append(i)
            self._arr_dense = mat[keep].toarray()
        return self._arr_dense.copy() if tocopy else self._arr_dense
