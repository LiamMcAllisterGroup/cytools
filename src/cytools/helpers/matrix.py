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
# Description:  This module contains a basic LIL class along with some helpers.
# -----------------------------------------------------------------------------

# 'standard' imports
import copy

# 3rd party imports
import numpy as np

# CYTools imports
from cytools.helpers import misc

# typing
from numpy.typing import ArrayLike
from typing import Callable, Iterator, Union

numeric = Union[int, float, np.number]


class LazyTuple:
    """
    A tuple class whose components are only lazily calculated

    **Arguments:**
    - `data`: Tuple elements (or functions to calculate them).
    """

    def __init__(self, *data: Union[object, Callable[[], object]]) -> None:
        self._data = tuple(data)

    def __repr__(self) -> str:
        # piggy-back printing from tuple
        return self._data.__repr__()

    def __str__(self) -> str:
        # piggy-back string conversion from tuple
        return self._data.__str__()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: int) -> numeric:
        item = self._data[key]

        if callable(item):
            self._data = list(self._data)
            self._data[key] = item()
            self._data = tuple(self._data)
            item = self._data[key]

        return item


class LIL:
    """
    This class describes a 2D lists of lists (LIL) sparse matrix. This has the
    same/less functionality as scipy.sparse.lil_array, but it is sometimes
    (much) quicker.

    **Arguments:**
    - `dtype`: The data type to use for when converting to a dense matrix.
    - `width`: The width of the matrix, used when converting to a dense matrix.
        A minimum necessary width can be inferred if this is not provided.
    - `iter_densely`: When iterating over the matrix, to iterate over the sparse
        representation (rows are dictionaries mapping column index to value) or
        over the dense representation.
    """

    def __init__(
        self,
        dtype: Union[np.dtype, str],
        width: int = None,
        iter_densely: bool = False,
    ) -> None:
        # data container
        self.arr = []
        self.arr_dense = None  # dense representation of `arr`

        # data type
        self.dtype = dtype

        # data shape
        self._len = None
        self.width = width

        # default value for unspecified indices
        self.default_val = 0

        # configuration on how to iterate over the matrix
        self.iter_densely = iter_densely

        # various sums
        self._sum_all = None
        self._sum_0 = None
        self._sum_0_dense = None
        self._sum_1 = None

    # basic interface
    # ---------------
    def __repr__(self) -> str:
        # piggy-back printing from list
        return self.arr.__repr__()

    def __str__(self) -> str:
        # piggy-back string conversion from list
        return self.arr.__str__()

    def __iter__(self) -> Iterator:
        # iterator
        if self.iter_densely:
            return iter(self.dense())
        else:
            return iter(self.arr)

    def __setitem__(self, idx: tuple, value: numeric) -> None:
        # item assignment
        if not isinstance(idx, tuple):
            raise ValueError(f"Index must be tuple but was {type(idx)}...")

        self.arr[idx[0]][idx[1]] = value

    def __getitem__(self, idx: tuple) -> numeric:
        # indexing
        if isinstance(idx, tuple):
            # get element self.arr[i][j]
            if self.width is None:
                print("LIL: Width not set. Inferring from non-zero values...")
                self.width = self.infer_width()

            if idx[1] >= self.width:
                raise IndexError("LIL: list index out of range")
            else:
                return self.arr[idx[0]].get(idx[1], 0)
        else:
            # get element self.arr[i]
            return self.arr[idx]

    def __len__(self) -> int:
        # length
        return len(self.arr)

    def __array__(self, copy=False, dtype: Union[np.dtype, str] = None) -> np.array:
        # What is called upon running np.array on this object
        return np.array(self.dense(), copy=copy, dtype=dtype)

    @property
    def shape(self) -> tuple[int]:
        return (len(self), self.width)

    def __add__(self, other: "LIL") -> "LIL":
        # Addition. Acts like lists (appends)
        out = LIL(dtype=self.dtype, width=self.width)
        out.arr = self.arr + other.arr
        return out

    # basic methods
    # --------------
    def infer_width(self) -> int:
        """
        **Description:**
        Find the minimum width necessary to hold array

        **Arguments:**
        None.

        **Returns:**
        Nothing
        """
        return 1 + max([max(row.keys()) for row in self.arr])

    def new_row(self) -> None:
        """
        **Description:**
        Append an empty row to the dict.

        **Arguments:**
        None.

        **Returns:**
        Nothing
        """
        self.arr.append(dict())

    def append(self, toadd: "dict or LIL", tocopy: bool = True) -> "LIL":
        """
        **Description:**
        Append (a) row(s) to the array.

        **Arguments:**
        - `toadd`: Row(s) to add.
        - `tocopy`: Whether to append a copy of toadd.

        **Returns:**
        Itself.
        """
        if len(toadd) == 0:
            return self

        # convert to list of dicts
        if isinstance(toadd, dict):
            toadd = [toadd]
        elif isinstance(toadd[0], type(self)):
            toadd = flatten_top(toadd)

        if tocopy:
            self.arr += copy.copy(toadd)
        else:
            self.arr += toadd

        # reset length
        self._len = None

        return self

    def col_inds(self) -> set:
        return set().union(*[r.keys() for r in self.arr])

    def reindex(self, f: dict = None) -> None:
        """
        **Description:**
        Reindex the ith column to be the f(i)-th one.

        **Arguments:**
        - `f`: Dictionary mapping old column indices to new ones.

        **Returns:**
        Nothing
        """
        self.arr_dense = None

        # default map is contiguous from 0 to N_cols-1
        if f is None:
            f = {v: k for (k, v) in enumerate(self.col_inds())}

        for i, row in enumerate(self.arr):
            self.arr[i] = {(f[j] if j in f else j): v for j, v in row.items()}

    def unique_rows(self) -> None:
        """
        **Description:**
        Delete repeated rows. Maybe re-orders rows...

        **Arguments:**
        None.

        **Returns:**
        Nothing
        """
        self.arr_dense = None
        self.arr = [dict(t) for t in {tuple(sorted(d.items())) for d in self.arr}]

    def dense(self, tocopy: bool = False) -> np.array:
        """
        **Description:**
        Return a dense version of the array

        **Arguments:**
        - `copy`: Whether to return a copy of self.arr_dense.

        **Returns:**
        The dense array
        """
        if self.arr_dense is None:
            # delete duplicated rows
            self.unique_rows()

            # build empty dense array
            if self.default_val == 0:
                self.arr_dense = np.zeros(
                    self.shape, dtype=self.dtype
                )
            else:
                self.arr_dense = self.default_val * np.ones(
                    self.shape, dtype=self.dtype
                )

            # fill in output
            for i, row in enumerate(self.arr):
                for j, v in row.items():
                    self.arr_dense[i, j] = v

        # return
        if tocopy:
            return self.arr_dense.copy()
        else:
            return self.arr_dense

    def tolist(self) -> list:
        return self.dense().tolist()

    def sum(
        self, axis: int = None, dense: bool = True
    ) -> Union[numeric, ArrayLike]:
        if axis is None:
            if self._sum_all is None:
                self._sum_all = np.sum(self.sum(axis=1))
            return self._sum_all
        elif axis == 1:
            if self._sum_1 is None:
                self._sum_1 = np.asarray([sum(r.values()) for r in self.arr])
            return self._sum_1
        elif axis == 0:
            if dense:
                if self._sum_0_dense is None:
                    self._sum_0_dense = np.asarray(
                        [
                            sum(r.get(i, 0) for r in self.arr)
                            for i in range(self.width)
                        ]
                    )
                return self._sum_0_dense
            else:
                if self._sum_0 is None:
                    self._sum_0 = {
                        i: sum(r.get(i, 0) for r in self.arr)
                        for i in self.col_inds()
                    }
                return self._sum_0


class LIL_stack:
    """
    This class describes a stack of LIL objects. One could just manually stack
    the rows but this implementation is quicker.

    The stack is organized as a list of options,
        options = [ [top_block_option1, top_block_option2, ...],
                    [next_block_option1, next_block_option2, ...],
                    ...
                    [bot_block_option1, bot_block_option2, ...]]
    and a list of choices
        choices = [i_top_block, i_next_block, ..., i_bot_block]
    E.g., if choices were [7,2,...,6], the stack would look like:
        stack = [top_block_option7;
                 next_block_option2;
                 ...
                 bot_block_option6]

    **Arguments:**
    - `options`: The possible arrays to stack. The entry options[i] is a list
        of all possible blocks you can put in the ith entry
    - `choices`: The selection of which blocks (from options) to stack.
    - `choice_bounds`: The number of possible choices for each block. I.e.,
        [len(opts) for opts in options]
    - `iter_densely`: Whether to iterate densely over the array or sparsely.
    """

    def __init__(
        self,
        options: [[ArrayLike]],
        choices: [int],
        choice_bounds: [int],
        iter_densely: bool = False,
    ) -> None:
        self._options = options
        if isinstance(choices, int):
            self._choices = choices
        else:
            self._choices = misc.to_base10(choices, choice_bounds)
        self._choice_bounds = choice_bounds
        self.iter_densely = iter_densely

    # basic interfaces
    def __repr__(self) -> str:
        # piggy-back printing from list
        return self.arr.__repr__()

    def __str__(self) -> str:
        # piggy-back string conversion from list
        return self.arr.__str__()

    def __getitem__(self, idx: tuple) -> numeric:
        if isinstance(idx, tuple):
            # get element self.arr[i][j]

            if idx[0] >= len(self):
                raise IndexError("LIL_stack: 0th list index out of range")
            elif idx[0] < 0:
                raise IndexError(
                    "LIL_stack: negative indexing not currently allowed"
                )

            for block in self._blocks():
                L = len(block)
                if idx[0] < L:
                    return block[idx]
                else:
                    idx = (idx[0] - L, idx[1])
        else:
            # get element self.arr[i]
            if idx >= len(self):
                raise IndexError("LIL_stack: list index out of range")
            elif idx < 0:
                raise IndexError(
                    "LIL_stack: negative indexing not currently allowed"
                )

            for block in self._blocks():
                L = len(block)
                if idx < L:
                    return block[idx]
                else:
                    idx -= L

    def __len__(self) -> int:
        # length
        return len(self.arr)

    def __iter__(self) -> Iterator:
        # iterator
        return self._rows(self.iter_densely)

    def __array__(self, copy=False, dtype: np.dtype = None) -> np.array:
        # What is called upon running np.array on this object
        return np.array(self.dense(), copy=copy, dtype=dtype)

    # properties
    @property
    def choices(self) -> list[int]:
        return misc.from_base10(self._choices, self._choice_bounds)

    @property
    def dtype(self) -> np.dtype:
        return self._options[0][0].dtype

    @property
    def width(self) -> int:
        return self._options[0][0].width

    @property
    def shape(self) -> "LazyTuple":
        if not hasattr(self, "_shape"):
            # self._shape = (len(self),self.width) # slow
            self._shape = LazyTuple(self.__len__, self.width)
        return self._shape

    @property
    def is_empty(self) -> bool:
        if not hasattr(self, "_is_empty"):
            self._is_empty = True

            for block in self._blocks():
                if len(block) > 0:
                    self._is_empty = False
                    break

        return self._is_empty

    def _blocks(self) -> Iterator["LIL"]:
        for i, opts in zip(self.choices, self._options):
            yield opts[i]

    def _rows(self, dense: bool = True) -> Iterator:
        if dense:

            def row_iter(r: "LIL") -> np.array:
                return r.dense()

        else:

            def row_iter(r: "LIL") -> "LIL":
                return r

        for block in self._blocks():
            yield from row_iter(block)

    # getter
    @property
    def arr(self) -> ArrayLike:
        if not hasattr(self, "_arr"):
            self._arr = [row for row in self._rows(False)]

        return self._arr

    @arr.setter
    def arr(self, value):
        self._arr = value

    def dense(self, tocopy: bool = False) -> ArrayLike:
        """
        **Description:**
        Return a dense version of the array

        **Arguments:**
        - `copy`: Whether to return a copy of self.arr_dense.

        **Returns:**
        *(np.array)* The dense array
        """
        if not hasattr(self, "_arr_dense"):
            # delete duplicated rows
            self.unique_rows()
            
            # build empty dense array
            self._arr_dense = np.zeros(self.shape, dtype=self.dtype)

            # fill in output
            for i, row in enumerate(self.arr):
                for j, v in row.items():
                    self._arr_dense[i, j] = v

        # return
        if tocopy:
            return self._arr_dense.copy()
        else:
            return self._arr_dense

    def unique_rows(self) -> None:
        """
        **Description:**
        Delete repeated rows. Maybe re-orders rows...

        **Arguments:**
        None.

        **Returns:**
        Nothing
        """
        self.arr_dense = None
        self.arr = [dict(t) for t in {tuple(d.items()) for d in self.arr}]

    def tolist(self) -> list:
        return self.dense().tolist()

    # basic methods
    def sum(
        self, axis: int = None, dense: bool = True
    ) -> Union[numeric, ArrayLike]:
        if axis is None:
            if not hasattr(self, "_sum_all"):
                self._sum_all = np.sum(self.sum(axis=1))
            return self._sum_all
        elif axis == 1:
            if not hasattr(self, "_sum_1"):
                self._sum_1 = flatten_top(
                    [M.sum(axis=1) for M in self._blocks()], as_list=False
                )
            return self._sum_1
        elif axis == 0:
            if dense:
                if not hasattr(self, "_sum_0_dense"):
                    self._sum_0_dense = np.sum(
                        M.sum(axis=0, dense=True) for M in self._blocks()
                    )
                return self._sum_0_dense
            else:
                raise NotImplementedError("sparse sum not yet implemented")


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
