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

"""Sparse matrix helpers.

This module contains helpers to build and combine sparse (CSR) matrices, such as
the hyperplanes of secondary cones, and a class describing a lazily concatenated
stack of sparse blocks.
"""

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
    """Flatten the top level(s) of a nested list or array.

    Each level of flattening concatenates the elements of the top-level entries.
    Entries may be ragged or have unequal depths. NumPy arrays are accepted, but a
    message suggesting `reshape` is printed, and array elements are converted to
    lists.

    Args:
        arr (ArrayLike): The nested list or array to flatten.
        as_list (bool): Whether to return a list (instead of a NumPy array).
        N (int): The number of levels to flatten, starting from the top.

    Returns:
        list | np.ndarray: The input with its top `N` levels flattened.

    Examples:
        >>> import numpy as np
        >>> from cytools.helpers.matrix import flatten_top
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
def _check_range(vals, dtype):
    """Check that some values fit in an integer type.

    NumPy's `astype` silently wraps out-of-range values, which would corrupt the
    coefficients instead of failing.

    Args:
        vals (np.ndarray): The values.
        dtype (np.dtype): The integer type.

    Raises:
        OverflowError: If some value does not fit in `dtype`.
    """
    if not len(vals):
        return
    info = np.iinfo(dtype)
    lo, hi = int(np.min(vals)), int(np.max(vals))
    if lo < info.min or hi > info.max:
        raise OverflowError(
            f"entries [{lo}, {hi}] do not fit in {np.dtype(dtype).name} "
            f"[{info.min}, {info.max}]; pass a wider dtype")


def _index_dtype(nnz, width):
    """Choose the index type for building a CSR matrix.

    SciPy rescans `int64` indices to downcast them, which costs more than these
    arrays are worth, so the narrower `int32` is chosen up front when it fits.

    Args:
        nnz (int): The number of stored entries.
        width (int): The number of columns.

    Returns:
        type: `np.int32` if both `nnz` and `width` fit in it, otherwise `np.int64`.
    """
    return np.int32 if max(int(nnz), int(width)) < 2**31 else np.int64


def csr_rows(cols, vals, width, dtype=np.int16):
    """Build a CSR matrix from arrays of column indices and values.

    Row `i` of the output has value `vals[i][j]` in column `cols[i][j]` for every
    `j`. Zero values are dropped.

    Args:
        cols (ArrayLike): A 2D array with the column index of each entry.
        vals (ArrayLike): A 2D array, of the same shape as `cols`, with the value of
            each entry.
        width (int): The number of columns.
        dtype (np.dtype): The type of the stored values.

    Returns:
        sp.csr_matrix: The matrix, with one row per row of `cols`.

    Raises:
        OverflowError: If some value does not fit in `dtype`.

    Examples:
        >>> from cytools.helpers.matrix import csr_rows
        >>> csr_rows([[0,2],[1,1]], [[1,-1],[0,3]], width=4).toarray()
        array([[ 1,  0, -1,  0],
               [ 0,  3,  0,  0]], dtype=int16)
    """
    cols = np.asarray(cols)
    vals = np.asarray(vals)
    # len, not size: a (k, 0) input is k rows that happen to have no entries,
    # and dropping them would silently lose rows
    if len(vals) == 0:
        return sp.csr_matrix((0, width), dtype=dtype)

    keep = vals != 0
    idx = _index_dtype(keep.sum(), width)
    indptr = np.zeros(len(vals) + 1, dtype=idx)
    indptr[1:] = np.cumsum(keep.sum(axis=1))
    _check_range(vals[keep], dtype)
    return sp.csr_matrix((vals[keep].astype(dtype),
                          cols[keep].astype(idx, copy=False), indptr),
                         shape=(len(vals), width))


def csr_dicts(rows, width, dtype=np.int16):
    """Build a CSR matrix from a list of rows given as dictionaries.

    Each row is a dictionary mapping column indices to values. Unlike
    [`csr_rows`][cytools.helpers.matrix.csr_rows], explicit zero values are kept as
    stored entries.

    Args:
        rows (list[dict[int, int]]): The rows, as `{column: value}` dictionaries.
        width (int): The number of columns.
        dtype (np.dtype): The type of the stored values.

    Returns:
        sp.csr_matrix: The matrix, with one row per dictionary.

    Raises:
        OverflowError: If some value does not fit in `dtype`.

    Examples:
        >>> from cytools.helpers.matrix import csr_dicts
        >>> csr_dicts([{0: 1, 3: 2}, {}], width=4).toarray()
        array([[1, 0, 0, 2],
               [0, 0, 0, 0]], dtype=int16)
    """
    if not len(rows):
        return sp.csr_matrix((0, width), dtype=dtype)

    lens = [len(r) for r in rows]
    nnz = int(sum(lens))
    idx = _index_dtype(nnz, width)
    indptr = np.zeros(len(rows) + 1, dtype=idx)
    indptr[1:] = np.cumsum(lens)
    cols = np.fromiter((c for r in rows for c in r), dtype=idx, count=nnz)
    raw = np.fromiter((v for r in rows for v in r.values()), dtype=np.int64,
                      count=nnz)
    _check_range(raw, dtype)
    vals = raw.astype(dtype)
    return sp.csr_matrix((vals, cols, indptr), shape=(len(rows), width))


def csr_stack(blocks, width, dtype=np.int16):
    """Stack CSR matrices vertically, skipping empty ones.

    Args:
        blocks (Iterable[sp.csr_matrix]): The matrices to stack.
        width (int): The number of columns, used when there is nothing to stack.
        dtype (np.dtype): The value type, used when there is nothing to stack.

    Returns:
        sp.csr_matrix: The stacked matrix.
    """
    blocks = [b for b in blocks if b.shape[0]]
    if not blocks:
        return sp.csr_matrix((0, width), dtype=dtype)
    return sp.vstack(blocks, format="csr")


def csr_unique_rows(mat):
    """Remove duplicate rows from a CSR matrix, keeping the first occurrence of each.

    Two rows are considered equal if they have the same stored entries. The input
    matrix is canonicalized in place (duplicate entries are summed and the indices
    sorted), but explicitly stored zeros are not removed.

    Args:
        mat (sp.csr_matrix): The matrix.

    Returns:
        sp.csr_matrix: The matrix without duplicate rows. If there are no
            duplicates, the input matrix itself is returned.
    """
    if mat.shape[0] < 2:
        return mat

    mat.sum_duplicates()
    # key each row by the raw bytes of its indices and values. Slicing one
    # bytes buffer beats building a tuple per row, which boxed every entry
    ind = np.ascontiguousarray(mat.indices)
    dat = np.ascontiguousarray(mat.data)
    ind_b, dat_b = ind.tobytes(), dat.tobytes()
    iw, dw = ind.itemsize, dat.itemsize
    ptr = mat.indptr.tolist()
    keys = [ind_b[ptr[i] * iw:ptr[i + 1] * iw]
            + dat_b[ptr[i] * dw:ptr[i + 1] * dw]
            for i in range(len(ptr) - 1)]
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return mat[keep] if len(keep) < mat.shape[0] else mat


class CSR_stack:
    """A lazily concatenated stack of sparse blocks.

    The stack is organized as a list of options for each position, together with a
    choice of one option per position:

    ```python
    options = [[top_block_option1, top_block_option2, ...],
               ...,
               [bot_block_option1, bot_block_option2, ...]]
    choices = [i_top_block, ..., i_bot_block]
    ```

    The chosen blocks are not concatenated until needed, which is useful when many
    candidate stacks are built and most of them are discarded. Rows are accessed as
    `{column: value}` dictionaries.

    Args:
        options (list[list[sp.csr_matrix]]): The possible blocks, where `options[i]`
            lists the blocks that can be placed at position `i`. All blocks must have
            the same number of columns and value type.
        choices (list[int] | int): The index of the chosen block at each position,
            or these indices already encoded as a single integer with
            [`to_base10`][cytools.helpers.misc.to_base10].
        choice_bounds (list[int]): The number of options at each position.
        iter_densely (bool): Whether iterating over the stack yields the rows of
            [`dense`][cytools.helpers.matrix.CSR_stack.dense] (as arrays, with
            duplicates removed) instead of all rows as dictionaries.

    Examples:
        >>> from cytools.helpers.matrix import CSR_stack, csr_dicts
        >>> b1 = csr_dicts([{0: 1}], width=3)
        >>> b2 = csr_dicts([{1: 2}, {2: 3}], width=3)
        >>> s = CSR_stack([[b1, b2], [b1]], choices=[1, 0], choice_bounds=[2, 1])
        >>> s.shape
        (3, 3)
        >>> s[1]
        {2: 3}
        >>> s.dense()
        array([[0, 2, 0],
               [0, 0, 3],
               [1, 0, 0]], dtype=int16)
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
        """Return a short description of the stack, including its shape.

        Returns:
            str: The description.
        """
        return f"CSR_stack(shape={tuple(self.shape)})"

    def __str__(self) -> str:
        """Return a short description of the stack, including its shape.

        Returns:
            str: The description.
        """
        return self.__repr__()

    @property
    def choices(self) -> "list[int]":
        """Return the index of the chosen block at each position.

        Returns:
            list[int]: The chosen indices.
        """
        return misc.from_base10(self._choices, self._choice_bounds)

    def _blocks(self):
        """Iterate over the chosen blocks, from top to bottom.

        Yields:
            sp.csr_matrix: The chosen blocks.
        """
        for i, opts in zip(self.choices, self._options):
            yield opts[i]

    @property
    def dtype(self) -> np.dtype:
        """Return the type of the stored values.

        Returns:
            np.dtype: The value type of the blocks.
        """
        return self._options[0][0].dtype

    @property
    def width(self) -> int:
        """Return the number of columns.

        Returns:
            int: The number of columns of the blocks.
        """
        return self._options[0][0].shape[1]

    def __len__(self) -> int:
        """Return the number of rows of the stack, including duplicates.

        Returns:
            int: The number of rows.
        """
        if not hasattr(self, "_len"):
            self._len = sum(b.shape[0] for b in self._blocks())
        return self._len

    @property
    def shape(self) -> tuple:
        """Return the shape of the stack, including duplicate rows.

        Returns:
            tuple[int, int]: The number of rows and columns.
        """
        return (len(self), self.width)

    @property
    def is_empty(self) -> bool:
        """Return whether the stack has no rows.

        Returns:
            bool: Whether all of the chosen blocks are empty.
        """
        if not hasattr(self, "_is_empty"):
            self._is_empty = not any(b.shape[0] for b in self._blocks())
        return self._is_empty

    def __getitem__(self, idx):
        """Return a row, or a single entry, of the stack.

        Args:
            idx (int | tuple[int, int]): A non-negative row index, or a pair of row and
                column indices.

        Returns:
            dict[int, int] | int: The row, as a `{column: value}` dictionary of its
                stored entries, or the value of the entry (0 if it is not stored).

        Raises:
            IndexError: If the row index is negative or out of range.
        """
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
        """Iterate over the rows of the stack.

        If `iter_densely` is true, the rows of
        [`dense`][cytools.helpers.matrix.CSR_stack.dense] (with duplicates removed) are
        yielded as arrays. Otherwise, all rows are yielded as `{column: value}`
        dictionaries.

        Returns:
            Iterator: An iterator over the rows.
        """
        if self.iter_densely:
            return iter(self.dense())
        return (self[i] for i in range(len(self)))

    def __array__(self, dtype: np.dtype = None, copy: bool = None) -> np.array:
        """Convert the stack into a NumPy array.

        This supports `np.asarray(stack)`, and gives the rows of
        [`dense`][cytools.helpers.matrix.CSR_stack.dense] (with duplicates removed).

        Args:
            dtype (np.dtype | None): The type of the output array.
            copy (bool | None): Passed to `np.array`.

        Returns:
            np.ndarray: The dense array.
        """
        # the order and defaults are fixed by the numpy protocol, which calls
        # this as __array__(dtype, copy)
        return np.array(self.dense(), dtype=dtype, copy=copy)

    def tocsr(self) -> "sp.csr_matrix":
        """Concatenate the chosen blocks into a single CSR matrix.

        Returns:
            sp.csr_matrix: The stacked matrix.
        """
        blocks = [b for b in self._blocks() if b.shape[0]]
        if not blocks:
            return sp.csr_matrix((0, self.width), dtype=self.dtype)
        return sp.vstack(blocks, format="csr")

    def dense(self, tocopy: bool = False) -> ArrayLike:
        """Return a dense version of the stack, with duplicate rows removed.

        The first occurrence of each row is kept. The result is cached.

        Args:
            tocopy (bool): Whether to return a copy of the cached array. If false, the
                cached array itself is returned, so it must not be modified.

        Returns:
            np.ndarray: The dense array.
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
