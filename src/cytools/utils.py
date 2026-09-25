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
# Description:  This module contains various common functions that are used in
#               CYTools.
# -----------------------------------------------------------------------------

"""Miscellaneous utility functions used throughout CYTools.

This module contains the functions to read polytopes from files or strings
([`read_polytopes`][cytools.utils.read_polytopes]) and to fetch them from the
Kreuzer-Skarke and Schöller-Skarke databases
([`fetch_polytopes`][cytools.utils.fetch_polytopes]), both of which can also be
imported from the package root. It also contains various helpers for exact
integer linear algebra, conversions to and from FLINT number types, sparse
tensors, and linear systems, which are mainly used internally.
"""

# 'standard' imports
import fractions
import functools
import itertools
import re
import requests
from typing import Generator

# 3rd party imports
import flint
import numpy as np
from numpy.typing import ArrayLike
import pypalp
import scipy.sparse as sp

# `scipy.sparse` does not necessarily bring in its `linalg` subpackage, so
# import it explicitly; it is used below as `sp.linalg.spsolve`
import scipy.sparse.linalg  # noqa: F401

# CYTools imports
from cytools import config

# imported for annotations only; the signatures quote these, so they
# are never needed at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cytools.polytope import Polytope
    from collections.abc import Generator
    from cytools.calabiyau import CalabiYau
    from cytools.toricvariety import ToricVariety


# custom decorators
# -----------------
# class instance caching
# (lru_cache persists for all class instances... that is not desired...)
def instanced_lru_cache(maxsize=128):
    """Return a decorator that caches the results of a method per instance.

    This works like `functools.lru_cache`, but each instance gets its own cache,
    stored in its `_cache` dictionary (which is created if needed), so that cached
    results are never shared between instances. Resetting `_cache` to an empty
    dictionary clears the cache of the instance. The arguments of the decorated
    method must be hashable.

    Args:
        maxsize (int | None): The maximum number of results cached per instance
            and method, as in `functools.lru_cache`. If `None`, the cache can grow
            without bound.

    Returns:
        Callable: The decorator.

    Examples:
        We define a class with a cached method. The result is only computed the
        first time it is requested.

        >>> from cytools.utils import instanced_lru_cache
        >>> class A:
        ...     @instanced_lru_cache(maxsize=None)
        ...     def square(self, x):
        ...         print("computing")
        ...         return x**2
        >>> a = A()
        >>> a.square(3)
        computing
        9
        >>> a.square(3)
        9
    """
    # implement lru_cache, stored in self._cache
    def decorator(func):
        @functools.wraps(func)  # copy func's metadata
        def wrapper(self, *args, **kwargs):
            # make class cache if it doesn't exist
            if not hasattr(self, "_cache"):
                self._cache = {}

            # store function cache in class cache
            fname = func.__name__
            if fname not in self._cache:
                self._cache[fname] = functools.lru_cache(maxsize=maxsize)(func)

            # use cached result
            return self._cache[fname](self, *args, **kwargs)

        return wrapper

    return decorator


# basic math
# ----------
def gcd_float(a: float, b: float, tol: float = 1e-5) -> float:
    """Return the greatest common divisor of two floats.

    The gcd is computed with the Euclidean algorithm, where a remainder smaller
    than `tol` in absolute value is treated as zero. The result is thus, up to
    rounding errors, the largest number that divides both `a` and `b` a whole
    number of times. This is useful since the standard gcd functions do not accept
    non-integer values.

    Args:
        a (float): The first number.
        b (float): The second number.
        tol (float): The tolerance below which a remainder is considered to be
            zero.

    Returns:
        float: The (non-negative) gcd of `a` and `b`.

    Warning:
        If `b` is already smaller than `tol` in absolute value, then `abs(a)` is
        returned. For example, `gcd_float(100, 0.1, 0.2)` returns `100`.

    Examples:
        We compute the gcd of two floats. The exact answer is 0.1, but there are
        small rounding errors.

        >>> from cytools.utils import gcd_float
        >>> gcd_float(0.2, 0.5)
        0.09999999999999998
    """
    if abs(b) < tol:
        return abs(a)
    return gcd_float(b, a % b, tol)


# variant that computes gcd over all elements in arr
def gcd_list(arr):
    """Return the greatest common divisor of a list of numbers.

    The gcd is computed pairwise with [`gcd_float`][cytools.utils.gcd_float],
    using its default tolerance, so the entries do not need to be integers.

    Args:
        arr (Iterable[float]): The numbers. It must not be empty.

    Returns:
        float: The gcd of all the numbers.

    Examples:
        We compute the gcd of a few integers.

        >>> from cytools.utils import gcd_list
        >>> gcd_list([4, 6, 10])
        2
    """
    return functools.reduce(gcd_float, arr)


# linear algebra
# --------------
def integral_nullspace(M, reduce_by_gcd=True):
    r"""Return a basis of the integral nullspace of an integer matrix.

    The returned vectors form a basis of the lattice
    $\{x \in \mathbb{Z}^n : Mx = 0\}$, where $n$ is the number of columns of
    `M`. This lattice is saturated: if $kx$ lies in it for some nonzero integer
    $k$, then so does $x$. As a consequence, the basis can always be completed to
    a basis of $\mathbb{Z}^n$. Note that dividing the vectors of a rational
    nullspace basis by their gcds is not enough to obtain such a basis; in general
    they only span a sublattice of finite index. For example, for
    `M = [[9, 4, 7]]` the kernel vector `[1, -4, 1]` might not be an integer
    combination of them.

    The basis is computed from the Hermite normal form $H = UM^T$, where $U$ is
    unimodular. The rows of $U$ corresponding to zero rows of $H$ span the kernel,
    and since $U$ is unimodular they extend to a basis of $\mathbb{Z}^n$.

    Args:
        M (ArrayLike): The integer matrix whose nullspace is computed.
        reduce_by_gcd (bool): Ignored. It is only accepted for backwards
            compatibility, since the returned basis vectors are always primitive.

    Returns:
        np.ndarray: The basis vectors, as the columns of an array of shape
            `(n, k)`, where `k` is the dimension of the nullspace. The entries have
            `int64` dtype when they fit, and are Python integers (`object` dtype)
            otherwise.

    Examples:
        We compute the integral nullspace of a $1\times 3$ matrix. The kernel
        vector `[1, -4, 1]` is 5 times the first column plus the second one.

        >>> from cytools.utils import integral_nullspace
        >>> integral_nullspace([[9, 4, 7]])
        array([[ -4,  21],
               [  9, -49],
               [  0,   1]])
    """
    # a row-less M gives flint a 0x0 matrix, losing the column count; its
    # kernel is the whole ambient space
    M = np.asarray(M)
    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=int)

    A = flint.fmpz_mat([[int(x) for x in row] for row in M.T.tolist()])
    H, U = A.hnf(transform=True)

    nrows, ncols = H.nrows(), H.ncols()
    kernel_rows = [i for i in range(nrows)
                   if all(H[i, j] == 0 for j in range(ncols))]
    if not kernel_rows:
        return np.zeros((M.shape[1], 0), dtype=int)

    # kernel entries can exceed int64 (a 20x24 matrix of single-digit
    # integers reached ~70 bits), so keep exact Python ints when they do
    cols = [[int(U[i, j]) for i in kernel_rows] for j in range(U.ncols())]
    try:
        return np.array(cols, dtype=np.int64)
    except OverflowError:
        return np.array(cols, dtype=object)


def adjugate(mat: ArrayLike) -> "tuple[np.ndarray, int]":
    r"""Return the adjugate and the determinant of a square integer matrix.

    The adjugate is the transpose of the cofactor matrix. It satisfies
    $A \operatorname{adj}(A) = \operatorname{adj}(A) A = \det(A) I$, and it is
    defined even when $A$ is singular. All computations are done exactly with
    Python integers, so the result is correct at any magnitude.

    Args:
        mat (ArrayLike): A square integer matrix.

    Returns:
        tuple[np.ndarray, int]: The adjugate and the determinant. The adjugate has
            `int64` dtype when its entries fit, and `object` dtype (Python
            integers) otherwise.

    Raises:
        ValueError: If the matrix is not square.

    Examples:
        We compute the adjugate and the determinant of a $2\times 2$ matrix.

        >>> from cytools.utils import adjugate
        >>> adjugate([[1, 2], [3, 4]])
        (array([[ 4, -2],
               [-3,  1]]), -2)
    """
    S = [[int(x) for x in row] for row in np.asarray(mat)]
    n = len(S)
    if any(len(row) != n for row in S):
        raise ValueError(f"expected a square matrix, got shape {np.shape(mat)}")

    def minor(i, j):
        return [[S[r][c] for c in range(n) if c != j] for r in range(n) if r != i]

    cof = [[(-1) ** (i + j) * _det(minor(i, j)) for j in range(n)] for i in range(n)]
    det = sum(S[0][j] * cof[0][j] for j in range(n))

    adj = [[cof[j][i] for j in range(n)] for i in range(n)]
    try:
        out = np.array(adj, dtype=np.int64)
    except OverflowError:
        out = np.array(adj, dtype=object)
    return out, det


def _det(m: list) -> int:
    """Return the exact determinant of a square integer matrix.

    Matrices up to 3x3 are expanded explicitly and 4x4 matrices by cofactors,
    while larger ones are handed to FLINT, which is faster at that size. The
    determinant of the 0x0 matrix is 1 (the empty product).

    Args:
        m (list[list[int]]): A square integer matrix, as a list of rows.

    Returns:
        int: The determinant.
    """
    n = len(m)
    if n == 0:
        return 1  # the empty product; makes adjugate of a 1x1 be [[1]]
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if n == 3:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
    if n >= 5:
        return int(flint.fmpz_mat(m).det())
    return sum(
        (-1) ** j * m[0][j] * _det([[r[c] for c in range(n) if c != j] for r in m[1:]])
        for j in range(n)
        if m[0][j]
    )


def lattice_index(mat: ArrayLike) -> int:
    r"""Return the index of the lattice generated by the rows of an integer matrix.

    The index is computed as the product of the invariant factors (the diagonal
    entries of the Smith normal form) of `mat`. For a square matrix of full rank,
    this is $|\det(\text{mat})|$, the index of the generated sublattice in
    $\mathbb{Z}^n$. More generally, if the rank of `mat` equals its number of rows
    or of columns, this is the index of the generated lattice in its saturation
    (the set of integer points in its real span). Otherwise, the result is 0.

    Args:
        mat (ArrayLike): An integer matrix whose rows generate the lattice.

    Returns:
        int: The lattice index.

    Examples:
        The vectors $(1,1)$ and $(1,-1)$ generate the sublattice of index 2 of
        $\mathbb{Z}^2$ consisting of the points with even coordinate sum.

        >>> from cytools.utils import lattice_index
        >>> lattice_index([[1, 1], [1, -1]])
        2
    """
    S = flint.fmpz_mat(np.asarray(mat, dtype=int).tolist()).snf()
    index = 1
    for i in range(min(S.nrows(), S.ncols())):
        index *= int(S[i, i])
    return index


# flint conversion
# ----------------
def float_to_fmpq(c: float) -> flint.fmpq:
    r"""Convert a float to an `fmpq`, FLINT's rational number type.

    The float is approximated by the closest rational number with denominator at
    most $10^6$ (see `fractions.Fraction.limit_denominator`), which recovers simple
    fractions despite floating-point rounding errors.

    Args:
        c (float): The number to convert.

    Returns:
        flint.fmpq: The rational approximation of `c`.

    Examples:
        We convert a few floats to rational numbers.

        >>> from cytools.utils import float_to_fmpq
        >>> float_to_fmpq(0.1), float_to_fmpq(0.333333333333), float_to_fmpq(2.45)
        (1/10, 1/3, 49/20)
    """
    f = fractions.Fraction(c).limit_denominator()
    return flint.fmpq(f.numerator, f.denominator)


def fmpq_to_float(c: flint.fmpq) -> float:
    """Convert an `fmpq`, FLINT's rational number type, to a float.

    Args:
        c (flint.fmpq): The rational number to convert.

    Returns:
        float: The number as a float.

    Examples:
        We convert a few rational numbers to floats.

        >>> from flint import fmpq
        >>> from cytools.utils import fmpq_to_float
        >>> [fmpq_to_float(fmpq(p, q)) for p, q in [(1, 2), (1, 3), (49, 20)]]
        [0.5, 0.3333333333333333, 2.45]
    """
    return int(c.p) / int(c.q)


def array_to_flint(arr: np.ndarray, t: "int | float" = None) -> np.ndarray:
    """Convert a numpy array to an array of FLINT numbers.

    Depending on `t`, each entry is converted either to an `fmpz` (FLINT's
    integer type) or to an `fmpq` (FLINT's rational type), the latter using
    [`float_to_fmpq`][cytools.utils.float_to_fmpq]. The functions
    [`array_int_to_fmpz`][cytools.utils.array_int_to_fmpz] and
    [`array_float_to_fmpq`][cytools.utils.array_float_to_fmpq] are shortcuts for
    the two cases.

    Args:
        arr (np.ndarray): The array to convert.
        t (type | None): The type of the entries of `arr`. If it is `int`, the
            entries are converted to `fmpz`, and otherwise to `fmpq`. If not
            specified, `arr.dtype` is used. Since a numpy dtype is never `int`,
            this always results in a conversion to `fmpq`.

    Returns:
        np.ndarray: An array of the same shape, with `object` dtype and `fmpz` or
            `fmpq` entries.

    Examples:
        We convert an integer array and a float array.

        >>> import numpy as np
        >>> from cytools.utils import array_to_flint
        >>> array_to_flint(np.array([[1, 0], [0, 2]]), t=int)
        array([[1, 0],
               [0, 2]], dtype=object)
        >>> array_to_flint(np.array([0.5, 0.25]))
        array([1/2, 1/4], dtype=object)
    """
    # type conversion function
    if t is None:
        t = arr.dtype

    if t is int:

        def f(n):
            return flint.fmpz(int(n))
    else:
        f = float_to_fmpq

    return np.vectorize(f)(arr).astype(object)


# some type-specific aliases
def array_int_to_fmpz(arr):
    """Convert an integer array to an array of `fmpz`, FLINT's integer type.

    This is a shortcut for [`array_to_flint`][cytools.utils.array_to_flint] with
    `t=int`.

    Args:
        arr (ArrayLike): An array with integer entries.

    Returns:
        np.ndarray: An array of the same shape, with `object` dtype and `fmpz`
            entries.

    Examples:
        We convert an integer array to an `fmpz` array.

        >>> from cytools.utils import array_int_to_fmpz
        >>> array_int_to_fmpz([[1, 0], [0, 2]])
        array([[1, 0],
               [0, 2]], dtype=object)
    """
    return array_to_flint(arr, t=int)


def array_float_to_fmpq(arr):
    """Convert a float array to an array of `fmpq`, FLINT's rational type.

    This is a shortcut for [`array_to_flint`][cytools.utils.array_to_flint] with
    `t=float`. Each entry is converted with
    [`float_to_fmpq`][cytools.utils.float_to_fmpq].

    Args:
        arr (ArrayLike): An array with float entries.

    Returns:
        np.ndarray: An array of the same shape, with `object` dtype and `fmpq`
            entries.

    Examples:
        We convert a float array to an `fmpq` array.

        >>> from cytools.utils import array_float_to_fmpq
        >>> array_float_to_fmpq([0.5, 0.25])
        array([1/2, 1/4], dtype=object)
    """
    return array_to_flint(arr, t=float)


def array_from_flint(arr: np.ndarray, t=None) -> np.ndarray:
    """Convert an array of FLINT numbers to an array of integers or floats.

    Entries of type `fmpz` (FLINT's integer type) are converted to integers, and
    entries of type `fmpq` (FLINT's rational type) to floats.

    Args:
        arr (np.ndarray): An array with `fmpz` or `fmpq` entries.
        t (type | None): The type of the entries, either `flint.fmpz` or
            `flint.fmpq`. If not specified, it is taken to be the type of the first
            entry of `arr`.

    Returns:
        np.ndarray: An array of the same shape, with `int` or `float` dtype.

    Raises:
        ValueError: If the entries are not of type `fmpz` or `fmpq`.

    Examples:
        We convert an `fmpq` array to a float array.

        >>> import numpy as np
        >>> from flint import fmpq
        >>> from cytools.utils import array_from_flint
        >>> array_from_flint(np.array([fmpq(1, 2), fmpq(1, 3)]))
        array([0.5       , 0.33333333])
    """
    # get the type of arr
    if t is None:
        t = type(next(iter(arr.flatten())))

    # convert
    if t == flint.fmpz:
        return np.array(arr, dtype=int)
    elif t == flint.fmpq:
        return np.vectorize(fmpq_to_float)(arr).astype(float)
    else:
        raise ValueError(
            f"Input array had element of type {t}!" + "This is not a flint type!"
        )


# some type-specific aliases
def array_fmpz_to_int(arr):
    """Convert an array of `fmpz`, FLINT's integer type, to an integer array.

    This is a shortcut for [`array_from_flint`][cytools.utils.array_from_flint]
    with `t=flint.fmpz`.

    Args:
        arr (np.ndarray): An array with `fmpz` entries.

    Returns:
        np.ndarray: An array of the same shape, with `int` dtype.
    """
    return array_from_flint(arr, t=flint.fmpz)


def array_fmpq_to_float(arr):
    """Convert an array of `fmpq`, FLINT's rational type, to a float array.

    This is a shortcut for [`array_from_flint`][cytools.utils.array_from_flint]
    with `t=flint.fmpq`.

    Args:
        arr (np.ndarray): An array with `fmpq` entries.

    Returns:
        np.ndarray: An array of the same shape, with `float` dtype.
    """
    return array_from_flint(arr, t=flint.fmpq)


# sparse conversions
# ------------------
def to_sparse(
    arr: "dict | list", sparse_type: str = "dok"
) -> "sp.dok_matrix | sp.csr_matrix":
    """Convert a matrix given by its nonzero entries to a SciPy sparse matrix.

    The input matrix can be given either as a list of triples
    `[[a, b, M_ab], ...]` or as a dictionary `{(a, b): M_ab, ...}`, where `a` and
    `b` are the row and column indices of an entry. The shape of the output is
    inferred from the largest row and column indices, and its entries are
    floats.

    Args:
        arr (dict | list): The matrix, either as a list of the form
            `[[a, b, M_ab], ...]` or as a dictionary of the form
            `{(a, b): M_ab, ...}`.
        sparse_type (str): The format of the output matrix. The options are
            `"dok"` (a `scipy.sparse.dok_matrix`) and `"csr"` (a
            `scipy.sparse.csr_matrix`).

    Returns:
        sp.dok_matrix | sp.csr_matrix: The sparse matrix.

    Raises:
        ValueError: If `sparse_type` is not one of the allowed options.

    Examples:
        We convert the 5x5 identity matrix into a sparse matrix, in the two
        available formats.

        >>> from cytools.utils import to_sparse
        >>> id_array = [[0,0,1],[1,1,1],[2,2,1],[3,3,1],[4,4,1]]
        >>> M = to_sparse(id_array)
        >>> M.format, M.shape, M.nnz
        ('dok', (5, 5), 5)
        >>> to_sparse(id_array, sparse_type="csr").format
        'csr'

        The matrix can also be given as a dictionary.

        >>> to_sparse({(0, 1): 2, (2, 0): 1}).toarray()
        array([[0., 2.],
               [0., 0.],
               [1., 0.]])
    """
    #  input checking
    if sparse_type not in ("dok", "csr"):
        raise ValueError('sparse_type must be either "dok" or "csr".')

    # map all inputs to list case
    if isinstance(arr, dict):
        arr = [list(ind) + [val] for ind, val in arr.items()]

    # map to numpy array
    arr = np.asarray(arr)

    # form empty sparse matrix with appropriate dimensions
    sp_mat = sp.dok_matrix(tuple(1 + arr.max(axis=0)[:2]))

    # fill in matrix
    for r in arr:
        sp_mat[r[0], r[1]] = r[2]

    # return in appropriate format
    if sparse_type == "dok":
        return sp_mat
    else:
        return sp.csr_matrix(sp_mat)


def symmetric_sparse_to_dense(tensor: dict, basis: ArrayLike = None) -> np.ndarray:
    r"""Convert a symmetric sparse tensor to a dense array.

    The sparse tensor is a dictionary `{(a, b, ..., c): M_ab...c, ...}`, where each
    entry only needs to be given for one ordering of its indices; the dense tensor
    is filled in for all their permutations. Optionally, a change of basis is then
    applied, so that the entries of the output are
    $$
    T'_{ij\cdots k} = \sum_{a,b,\ldots,c} B_{ia} B_{jb} \cdots B_{kc}\,
    T_{ab\cdots c}\,,
    $$
    where $B$ is the `basis` matrix.

    Args:
        tensor (dict): The symmetric sparse tensor, as a dictionary of the form
            `{(a, b, ..., c): M_ab...c, ...}`.
        basis (ArrayLike | None): A matrix whose rows are the new basis elements,
            written in terms of the original ones. It must have at least as many
            columns as the largest index in `tensor` plus one. If not specified, no
            change of basis is applied.

    Returns:
        np.ndarray: The dense tensor. Without a change of basis, its length along
            each axis is the largest index in `tensor` plus one; with a change of
            basis, it is the number of rows of `basis`. The entries have the type
            of the first value in `tensor` (before any change of basis).

    Examples:
        We convert a sparse tensor to a dense array. We then do the same in the
        basis consisting of the elements 1 and 3, which gives the same result as
        [`filter_tensor_indices`][cytools.utils.filter_tensor_indices] with
        indices `[1, 3]`.

        >>> from cytools.utils import symmetric_sparse_to_dense
        >>> tensor = {(0,1):0, (1,1):1, (1,2):2, (1,3):3, (2,3):4}
        >>> symmetric_sparse_to_dense(tensor)
        array([[0, 0, 0, 0],
               [0, 1, 2, 3],
               [0, 2, 0, 4],
               [0, 3, 4, 0]])
        >>> basis = [[0,1,0,0],[0,0,0,1]]
        >>> symmetric_sparse_to_dense(tensor, basis)
        array([[1, 3],
               [3, 0]])
    """
    # build empty output object
    if basis is not None:
        dim = np.asarray(basis).shape[1]
    else:
        dim = 1 + max(set.union(*[set(inds) for inds in tensor.keys()]))

    rank = len(next(iter(tensor.keys())))
    t = type(next(iter(tensor.values())))
    out = np.zeros((dim,) * rank, dtype=t)

    # fill dense tensor
    for inds, val in tensor.items():
        for c in itertools.permutations(inds):
            out[c] = val

    # apply basis transformation
    if basis is not None:
        for i in reversed(range(rank)):
            out = np.tensordot(out, basis, axes=[[i], [1]])

    return out


def symmetric_dense_to_sparse(tensor: ArrayLike, basis: ArrayLike = None) -> dict:
    r"""Convert a dense symmetric tensor to a sparse tensor.

    The output is a dictionary `{(a, b, ..., c): M_ab...c, ...}` containing only
    the nonzero entries with sorted indices $a \le b \le \cdots \le c$.
    Optionally, a change of basis is applied first, in the same way as in
    [`symmetric_sparse_to_dense`][cytools.utils.symmetric_sparse_to_dense].

    Args:
        tensor (ArrayLike): A dense symmetric tensor, with the same length along
            all of its axes.
        basis (ArrayLike | None): A matrix whose rows are the new basis elements,
            written in terms of the original ones. Currently, only square matrices
            are handled correctly. If not specified, no change of basis is
            applied.

    Returns:
        dict: The sparse tensor, as a dictionary of the form
            `{(a, b, ..., c): M_ab...c, ...}`. The values are numpy scalars.

    Raises:
        ValueError: If the axes of `tensor` do not all have the same length.

    Examples:
        We convert a dense symmetric matrix to a sparse one.

        >>> from cytools.utils import symmetric_dense_to_sparse
        >>> symmetric_dense_to_sparse([[1,2],[2,3]])
        {(0, 0): np.int64(1), (0, 1): np.int64(2), (1, 1): np.int64(3)}
    """
    out = {}

    # grab dense tensor
    tensor = np.array(tensor)

    rank = len(tensor.shape)
    dim = set(tensor.shape)
    if len(dim) != 1:
        raise ValueError("All dimensions must have the same length")
    dim = next(iter(dim))

    # apply basis transformation
    if basis is not None:
        for i in reversed(range(rank)):
            tensor = np.tensordot(tensor, basis, axes=[[i], [1]])

    # iterate over increasing indices, filling sparse tensor
    for ind in itertools.combinations_with_replacement(range(dim), rank):
        if tensor[ind] != 0:
            out[ind] = tensor[ind]

    return out


# other tensor operations
# -----------------------
def filter_tensor_indices(tensor: dict, indices: list[int]) -> dict:
    """Restrict a sparse tensor to a subset of its indices.

    Only the entries whose indices all belong to `indices` are kept, and they are
    reindexed so that `indices[i]` becomes `i`. The indices of each resulting key
    are sorted in ascending order. This can be used, for example, to obtain the
    intersection numbers in a basis of divisors that is a subset of the prime
    toric divisors.

    Args:
        tensor (dict): A symmetric sparse tensor, as a dictionary of the form
            `{(a, b, ..., c): M_ab...c, ...}`.
        indices (list[int]): The indices to keep. Their order determines the new
            indexing.

    Returns:
        dict: The filtered and reindexed sparse tensor, in the same format as the
            input.

    Examples:
        We construct a simple tensor and keep only the indices 1 and 3.

        >>> from cytools.utils import filter_tensor_indices
        >>> tensor = {(0,1):0, (1,1):1, (1,2):2, (1,3):3, (2,3):4}
        >>> filter_tensor_indices(tensor, [1,3])
        {(0, 0): 1, (0, 1): 3}

        The intersection numbers of a toric variety in its divisor basis can be
        obtained in this way from those of all the prime toric divisors.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> v = p.triangulate().get_toric_variety()
        >>> intnums_all = v.intersection_numbers(in_basis=False)
        >>> intnums_basis = v.intersection_numbers(in_basis=True)
        >>> intnums_basis == filter_tensor_indices(intnums_all, v.divisor_basis())
        True
    """
    # map from index to its count in indices object
    reindex = {ind: i for i, ind in enumerate(indices)}

    # only keep entries whose indices match those in indices
    # (membership is tested against a set; `indices` may be large and these
    # tensors can hold O(10^6) entries)
    indices_set = set(indices)
    filtered = {
        key: val for key, val in tensor.items() if all(c in indices_set for c in key)
    }

    # return reindexed tensor (order defined by indices input)
    return {
        tuple(sorted(reindex[c] for c in key)): val for key, val in filtered.items()
    }


# solve systems
# -------------
def solve_linear_system(
    M: sp.csr_matrix,
    C: list[float],
    backend: str = "all",
    check: bool = True,
    backend_error_tol: float = 1e-4,
    verbosity: int = 0,
) -> np.ndarray:
    """Solve the sparse linear system $Mx + C = 0$.

    The system is solved in the least-squares sense, through the normal equations
    $M^T M x = -M^T C$. The `"sksparse"` backend uses a sparse Cholesky
    factorization from scikit-sparse (an optional dependency), and the `"scipy"`
    backend uses `scipy.sparse.linalg.spsolve`. If `check` is true, a solution is
    only accepted if every entry of the residual $Mx + C$ is at most
    `backend_error_tol` in absolute value.

    Args:
        M (sp.csr_matrix): The coefficient matrix, as a SciPy sparse matrix.
        C (list[float]): The constant term.
        backend (str): The solver to use. The options are `"all"`, `"sksparse"`,
            and `"scipy"`. With `"all"`, the `"sksparse"` and `"scipy"` backends
            are tried in that order until one of them finds a solution.
        check (bool): Whether to check the solution by computing the residual.
        backend_error_tol (float): The largest absolute value of the residual
            entries for which the solution is accepted. Only used if `check` is
            true.
        verbosity (int): The verbosity level. If it is 1 or larger, a message is
            printed whenever a backend fails.

    Returns:
        list[float] | np.ndarray | None: The solution, or `None` if no acceptable
            solution was found. The `"scipy"` backend returns a list, and the
            `"sksparse"` backend returns an array.

    Raises:
        ValueError: If `backend` is not one of the allowed options.
        RuntimeError: If `backend` is `"all"` and every backend raised an error,
            rather than just failing to find a solution.

    Examples:
        We solve a very simple linear system.

        >>> from cytools.utils import to_sparse, solve_linear_system
        >>> id_array = [[0,0,1],[1,1,1],[2,2,1],[3,3,1],[4,4,1]]
        >>> M = to_sparse(id_array, sparse_type="csr")
        >>> C = [1,1,1,1,1]
        >>> solve_linear_system(M, C)
        [-1.0, -1.0, -1.0, -1.0, -1.0]
    """
    # input checking
    backends = ["all", "sksparse", "scipy"]
    if backend not in backends:
        raise ValueError(f"Invalid backend... options are {backends}.")

    # solve the system
    solution = None

    if backend == "all":
        # a backend that fails in an unanticipated way must not break the
        # chain, so the fallback is guarded here rather than by widening the
        # per-backend excepts below
        errors = []
        for s in backends[1:]:
            try:
                solution = solve_linear_system(
                    M,
                    C,
                    backend=s,
                    check=check,
                    backend_error_tol=backend_error_tol,
                    verbosity=verbosity,
                )
            except Exception as e:
                errors.append((s, e))
                if verbosity >= 1:
                    print(f"Linear backend error: {s} raised {e!r}.")
                continue
            if solution is not None:
                return solution

        # every backend raised: surface the underlying error rather than
        # reporting it as "no solution"
        if len(errors) == len(backends) - 1:
            raise RuntimeError(
                f"All linear system backends failed: {errors}"
            ) from errors[-1][1]

    elif backend == "sksparse":
        try:
            from sksparse.cholmod import CholmodError, cholesky_AAt
        except ImportError:
            # optional dependency; not an error worth propagating
            if verbosity >= 1:
                print("Linear backend error: sksparse is not installed.")
        else:
            try:
                factor = cholesky_AAt(M.transpose())
                solution = factor(-M.transpose() * C)
            except (CholmodError, ArithmeticError, ValueError, RuntimeError) as e:
                # only numerical/shape failures are treated as "no solution";
                # anything else is a genuine bug and must not be swallowed
                if verbosity >= 1:
                    print(f"Linear backend error: sksparse failed ({e}).")

    elif backend == "scipy":
        try:
            solution = sp.linalg.spsolve(M.transpose() * M, -M.transpose() * C).tolist()
        except (ArithmeticError, ValueError, RuntimeError) as e:
            # only numerical/shape failures are treated as "no solution";
            # anything else is a genuine bug and must not be swallowed
            if verbosity >= 1:
                print(f"Linear backend error: scipy failed ({e}).")

    # check/return solution
    if solution is None:
        return None

    if check:
        res = M.dot(solution) + C
        max_error = max(abs(s) for s in res.flat)

        # NaNs/infs (e.g. from a singular system) are failures, but they do not
        # compare greater than the tolerance, so check for them explicitly
        if (not np.isfinite(max_error)) or (max_error > backend_error_tol):
            if verbosity >= 1:
                print("Linear backend error: numerical error.")
            solution = None

    return solution


# set algebraic geometric bases
# -----------------------------
def set_divisor_basis(
    tv_or_cy: "ToricVariety | CalabiYau",
    basis: ArrayLike,
    include_origin: bool = True,
):
    """Set the basis of divisors of a toric variety or Calabi-Yau.

    Specifying a basis of divisors also determines the dual basis of curves. The
    basis can be given either as a vector of indices of prime toric divisors, or
    as a matrix whose rows are linear combinations of prime toric divisors. All
    cached quantities that depend on the basis are cleared.

    This function is not meant to be called directly. Instead, it is called by
    the methods
    [`set_divisor_basis`][cytools.toricvariety.ToricVariety.set_divisor_basis] of
    `ToricVariety` and
    [`set_divisor_basis`][cytools.calabiyau.CalabiYau.set_divisor_basis] of
    `CalabiYau`. Generic bases given by matrices are an experimental feature,
    which must first be enabled with
    [`enable_experimental_features`][cytools.config.enable_experimental_features].

    Args:
        tv_or_cy (ToricVariety | CalabiYau): The toric variety or Calabi-Yau whose
            basis is set.
        basis (ArrayLike): The basis. If it is a vector, its entries are the
            indices of the prime toric divisors (i.e. of the corresponding points
            of the polytope) that form the basis. The indices are sorted, so their
            order does not matter. If it is a matrix, each row is a basis divisor
            written as a linear combination of the prime toric divisors, with or
            without a first column for the origin (i.e. the canonical divisor), as
            inferred from its shape.
        include_origin (bool): Whether the indices of a vector basis count the
            origin as index 0. If false, the indices are shifted by one. Ignored
            for matrix bases.

    Raises:
        ValueError: If the indices are out of range, if the matrix has the wrong
            rank or shape, or if the given divisors do not form an integral basis.
        Exception: If a matrix basis is given without enabling the experimental
            features.

    Note:
        Only integral bases are supported, meaning that every prime toric divisor
        must be an integral linear combination of the basis divisors.

    Examples:
        This function is not meant to be used directly, but it is called in the
        following example. We consider a toric variety with two independent
        divisors, find the default basis, and then set a basis of our choice.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> t = p.triangulate()
        >>> v = t.get_toric_variety()
        >>> v.divisor_basis()  # the default basis
        array([5, 6])
        >>> v.set_divisor_basis([1,6])  # here we set a basis
        >>> v.divisor_basis()  # we get the basis we set
        array([1, 6])
        >>> v.divisor_basis(as_matrix=True)  # the basis in matrix form
        array([[0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1]])
    """
    self = tv_or_cy

    # grab GLSM information
    glsm_cm = self.glsm_charge_matrix(include_origin=True)
    glsm_rnk = np.linalg.matrix_rank(glsm_cm)

    # grab basis information
    b = np.array(basis, dtype=int)  # (only integer bases are supported)

    if len(b.shape) == 1:
        # input is a vector
        b = np.array(sorted(basis)) + (not include_origin)

        # check that it is valid
        if (min(b) < 0) or (max(b) >= glsm_cm.shape[1]):
            raise ValueError("Indices are not in appropriate range.")

        if (glsm_rnk != np.linalg.matrix_rank(glsm_cm[:, b])) or (glsm_rnk != len(b)):
            raise ValueError("Input divisors do not form a basis.")

        if abs(int(round(np.linalg.det(glsm_cm[:, b])))) != 1:
            raise ValueError("Only integer bases are supported.")

        # Save divisor basis
        self._divisor_basis = b
        self._divisor_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._divisor_basis_mat[:, b] = np.eye(glsm_rnk, dtype=int)

        # Construct dual basis of curves
        self._curve_basis = b
        nobasis = np.array([i for i in range(glsm_cm.shape[1]) if i not in b])

        linrels = self.glsm_linear_relations()

        linrels_tmp = np.empty(linrels.shape, dtype=int)
        linrels_tmp[:, : len(nobasis)] = linrels[:, nobasis]
        linrels_tmp[:, len(nobasis) :] = linrels[:, b]

        linrels_tmp = flint.fmpz_mat(linrels_tmp.tolist()).hnf()
        linrels_tmp = np.array(linrels_tmp.tolist(), dtype=int)

        linrels_new = np.empty(linrels.shape, dtype=int)
        linrels_new[:, nobasis] = linrels_tmp[:, : len(nobasis)]
        linrels_new[:, b] = linrels_tmp[:, len(nobasis) :]

        self._curve_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._curve_basis_mat[:, b] = np.eye(len(b), dtype=int)
        sublat_ind = int(
            round(
                np.linalg.det(
                    np.array(
                        flint.fmpz_mat(linrels.tolist()).snf().tolist(),
                        dtype=int,
                    )[:, : linrels.shape[0]]
                )
            )
        )
        for nb in nobasis[::-1]:
            tup = [(k, kk) for k, kk in enumerate(linrels_new[:, nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i, ii = tup[-1]
            self._curve_basis_mat[:, nb] = (
                -self._curve_basis_mat.dot(linrels_new[i]) // ii
            )

    elif len(b.shape) == 2:
        # input is a matrix
        if not config._exp_features_enabled:
            raise Exception(
                "The experimental features must be enabled to use generic bases."
            )

        # We start by checking if the input matrix looks right
        if np.linalg.matrix_rank(b) != glsm_rnk:
            raise ValueError("Input matrix has incorrect rank.")

        if b.shape == (glsm_rnk, glsm_cm.shape[1]):
            new_b = b
        elif b.shape == (glsm_rnk, glsm_cm.shape[1] - 1):
            new_b = np.empty(glsm_cm.shape, dtype=int)
            new_b[:, 1:] = b
            new_b[:, 0] = 0
        else:
            raise ValueError("Input matrix has incorrect shape.")

        new_glsm_cm = new_b.dot(glsm_cm.T).T
        if np.linalg.matrix_rank(new_glsm_cm) != glsm_rnk:
            raise ValueError("Input divisors do not form a basis.")
        if (
            abs(
                int(
                    round(
                        np.linalg.det(
                            np.array(
                                flint.fmpz_mat(new_glsm_cm.tolist()).snf().tolist(),
                                dtype=int,
                            )[:glsm_rnk, :glsm_rnk]
                        )
                    )
                )
            )
            != 1
        ):
            raise ValueError("Input divisors do not form an integral basis.")
        self._divisor_basis = np.array(new_b)
        # Now we store a more convenient form of the matrix where we use the
        # linear relations to express them in terms of the default prime toric
        # divisors
        standard_basis = self.polytope().glsm_basis(
            integral=True,
            include_origin=True,
            points=self.prime_toric_divisors(),
        )
        linrels = self.polytope().glsm_linear_relations(
            include_origin=True, points=self.prime_toric_divisors()
        )
        self._divisor_basis_mat = np.array(new_b)
        nobasis = np.array(
            [i for i in range(glsm_cm.shape[1]) if i not in standard_basis]
        )
        sublat_ind = int(
            round(
                np.linalg.det(
                    np.array(
                        flint.fmpz_mat(linrels.tolist()).snf().tolist(),
                        dtype=int,
                    )[:, : linrels.shape[0]]
                )
            )
        )
        for nb in nobasis[::-1]:
            tup = [(k, kk) for k, kk in enumerate(linrels[:, nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i, ii = tup[-1]
            for j in range(self._divisor_basis_mat.shape[0]):
                self._divisor_basis_mat[j] -= (
                    self._divisor_basis_mat[j, nb] * linrels[i]
                )
        # Finally, we invert the matrix and construct the dual curve basis
        if (
            abs(int(round(np.linalg.det(self._divisor_basis_mat[:, standard_basis]))))
            != 1
        ):
            raise ValueError("Input divisors do not form an integral basis.")
        inv_mat = flint.fmpz_mat(
            self._divisor_basis_mat[:, standard_basis].tolist()
        ).inv(integer=True)
        inv_mat = np.array(inv_mat.tolist(), dtype=int)
        # flint sometimes returns the negative inverse
        if inv_mat.dot(self._divisor_basis_mat[:, standard_basis])[0, 0] == -1:
            inv_mat *= -1
        self._curve_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._curve_basis_mat[:, standard_basis] = np.array(inv_mat).T
        for nb in nobasis[::-1]:
            tup = [(k, kk) for k, kk in enumerate(linrels[:, nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i, ii = tup[-1]
            self._curve_basis_mat[:, nb] = -self._curve_basis_mat.dot(linrels[i]) // ii
        self._curve_basis = np.array(self._curve_basis_mat)
    else:
        raise ValueError("Input must be either a vector or a matrix.")
    # Clear the cache of all in-basis computations
    self.clear_cache(recursive=False, only_in_basis=True)


def set_curve_basis(
    tv_or_cy: "ToricVariety | CalabiYau",
    basis: ArrayLike,
    include_origin: bool = True,
):
    """Set the basis of curves of a toric variety or Calabi-Yau.

    Specifying a basis of curves also determines the dual basis of divisors. The
    basis can be given either as a vector of indices, or as a matrix whose rows
    are the basis curves, specified by their intersection numbers with the prime
    toric divisors. When a vector is used, this is equivalent to calling
    [`set_divisor_basis`][cytools.utils.set_divisor_basis] with the same vector.
    All cached quantities that depend on the basis are cleared.

    This function is not meant to be called directly. Instead, it is called by
    the methods
    [`set_curve_basis`][cytools.toricvariety.ToricVariety.set_curve_basis] of
    `ToricVariety` and
    [`set_curve_basis`][cytools.calabiyau.CalabiYau.set_curve_basis] of
    `CalabiYau`. Generic bases given by matrices are an experimental feature,
    which must first be enabled with
    [`enable_experimental_features`][cytools.config.enable_experimental_features].

    Args:
        tv_or_cy (ToricVariety | CalabiYau): The toric variety or Calabi-Yau whose
            basis is set.
        basis (ArrayLike): The basis. If it is a vector, its entries are the
            indices of the elements of the standard basis of the dual to the
            lattice of prime toric divisors, i.e. the curve basis dual to the
            divisor basis with the same indices. If it is a matrix, each row is a
            basis curve, given by its intersection numbers with the prime toric
            divisors, with or without a first column for the origin (i.e. the
            canonical divisor), as inferred from its shape. If the first column is
            omitted, it is set to minus the sum of the other entries of the row.
        include_origin (bool): Whether the indices of a vector basis count the
            origin as index 0. If false, the indices are shifted by one. Ignored
            for matrix bases.

    Raises:
        ValueError: If the vector basis is invalid (see
            [`set_divisor_basis`][cytools.utils.set_divisor_basis]), if the matrix
            has the wrong rank or shape, if its rows are not valid curve charge
            vectors (i.e. the corresponding combinations of the points of the
            polytope, each with a 1 appended, do not vanish), or if they do not
            form an integral basis.
        Exception: If a matrix basis is given without enabling the experimental
            features.

    Note:
        Only integral bases are supported, meaning that every toric curve must be
        an integral linear combination of the basis curves.

    Examples:
        This function is not meant to be used directly, but it is called in the
        following example. We consider a toric variety with two independent
        curves, find the default basis of curves, and then set a basis of our
        choice.

        >>> from cytools import Polytope
        >>> p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
        >>> t = p.triangulate()
        >>> v = t.get_toric_variety()
        >>> v.curve_basis()  # the default basis
        array([5, 6])
        >>> v.set_curve_basis([1,6])  # here we set a basis
        >>> v.curve_basis()  # we get the basis we set
        array([1, 6])
        >>> v.curve_basis(as_matrix=True)  # the basis in matrix form
        array([[-18,   1,   9,   6,   1,   1,   0],
               [ -6,   0,   3,   2,   0,   0,   1]])

        Since a vector is used, this is the same as calling
        [`set_divisor_basis`][cytools.utils.set_divisor_basis] with the same
        vector. The two functions only differ for generic bases given by matrices.
    """
    self = tv_or_cy  # More convenient to work with

    # parse basis
    b = np.array(basis, dtype=int)

    if len(b.shape) == 1:
        # input is a vector
        set_divisor_basis(self, b, include_origin=include_origin)
        return

    if len(b.shape) != 2:
        raise ValueError("Input must be either a vector or a matrix.")

    # Else input is a matrix
    if not config._exp_features_enabled:
        raise Exception(
            "The experimental features must be enabled to use generic bases."
        )

    # grab GLSM information
    glsm_cm = self.glsm_charge_matrix(include_origin=True)
    glsm_rnk = np.linalg.matrix_rank(glsm_cm)

    if np.linalg.matrix_rank(b) != glsm_rnk:
        raise ValueError("Input matrix has incorrect rank.")
    if b.shape == (glsm_rnk, glsm_cm.shape[1]):
        new_b = b
    elif b.shape == (glsm_rnk, glsm_cm.shape[1] - 1):
        new_b = np.empty(glsm_cm.shape, dtype=int)
        new_b[:, 1:] = b
        new_b[:, 0] = -np.sum(b, axis=1)
    else:
        raise ValueError("Input matrix has incorrect shape.")
    pts = [
        tuple(pt) + (1,)
        for pt in self.polytope().points()[[0] + list(self.prime_toric_divisors())]
    ]
    if any(new_b.dot(pts).flat):
        raise ValueError("Input curves do not form a valid basis.")
    if (
        abs(
            int(
                round(
                    np.linalg.det(
                        np.array(
                            flint.fmpz_mat(new_b.tolist()).snf().tolist(),
                            dtype=int,
                        )[:glsm_rnk, :glsm_rnk]
                    )
                )
            )
        )
        != 1
    ):
        raise ValueError("Input divisors do not form an integral basis.")
    standard_basis = self.polytope().glsm_basis(
        integral=True, include_origin=True, points=self.prime_toric_divisors()
    )
    if abs(int(round(np.linalg.det(new_b[:, standard_basis])))) != 1:
        raise ValueError("Input divisors do not form an integral basis.")
    inv_mat = flint.fmpz_mat(new_b[:, standard_basis].tolist()).inv(integer=True)
    inv_mat = np.array(inv_mat.tolist(), dtype=int)

    # flint sometimes returns the negative inverse
    if inv_mat.dot(new_b[:, standard_basis])[0, 0] == -1:
        inv_mat *= -1

    self._divisor_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
    self._divisor_basis_mat[:, standard_basis] = np.array(inv_mat).T
    self._divisor_basis = np.array(self._divisor_basis_mat)
    self._curve_basis = np.array(new_b)
    self._curve_basis_mat = np.array(new_b)

    # Clear the cache of all in-basis computations
    self.clear_cache(recursive=False, only_in_basis=True)


# polytope grabbing
# -----------------
def polytope_generator(
    input: str,
    input_type: str = "file",
    format: str = "ks",
    backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    lattice: str = None,
    limit: int = None,
) -> Generator["Polytope", None, None]:
    """Read polytopes from a file or a string, yielding them one at a time.

    The polytopes can be specified by their vertices, in the format used in the
    Kreuzer-Skarke database, or by weight systems.

    In the `"ks"` format, each polytope consists of a header line containing
    `M:`, whose first two numbers are the dimensions `n m` of the matrix that
    follows, and `n` lines with `m` integers each. The vertices are the columns of
    this matrix if it has more columns than rows, and its rows otherwise. In the
    `"ws"` format, each non-empty line is a weight system, such as `5 1 1 1 1 1`
    (the degree followed by the weights), and the polytope is the one that PALP
    associates with it.

    This function is not meant to be called directly. Instead, it is used by
    [`read_polytopes`][cytools.utils.read_polytopes] and
    [`fetch_polytopes`][cytools.utils.fetch_polytopes].

    Args:
        input (str): The name of the file to read, or the string containing the
            polytopes, depending on `input_type`.
        input_type (str): Whether `input` is the name of a file or a string
            containing the data. The options are `"file"` and `"str"`.
        format (str): The format of the data. The options are `"ks"`, the format
            used in the Kreuzer-Skarke database, where the polytopes are specified
            by their vertices, and `"ws"`, where each line is a weight system.
        backend (str | None): The backend used to construct the convex hull of
            each [`Polytope`][cytools.polytope.Polytope]. The options are `"ppl"`,
            `"qhull"`, and `"palp"`. If not specified, it is chosen automatically.
        deterministic_glsm_basis (bool): Whether to choose the GLSM (and thus
            divisor) basis of each polytope in a deterministic manner, so that it
            is consistent across different machines. The basis chosen with this
            flag may differ from the default one.
        dualize (bool): Whether to return the dual of each polytope instead of
            the polytope itself.
        favorable (bool | None): If true, only favorable polytopes are returned,
            and if false, only non-favorable ones. Favorability is checked before
            dualizing. If not specified, all polytopes are returned.
        lattice (str | None): The lattice, `"N"` or `"M"`, with respect to which
            favorability is checked. It must be specified if `favorable` is.
        limit (int | None): The maximum number of polytopes to return. If not
            specified, all of them are returned.

    Yields:
        Polytope: The polytopes that were read.

    Raises:
        ValueError: If `favorable` is specified without `lattice`, if `input_type`
            or `format` is not one of the allowed options, or if the dimensions of
            a vertex matrix do not match its header. Since this is a generator,
            these errors are raised when it is first advanced.

    Examples:
        We read a polytope from a string taken from the Kreuzer-Skarke database.

        >>> from cytools.utils import polytope_generator
        >>> poly_data = '''4 5  M:10 5 N:376 5 H:272,2 [540]
        ...    1    0    0    0   -9
        ...    0    1    0    0   -6
        ...    0    0    1    0   -1
        ...    0    0    0    1   -1
        ... '''
        >>> g = polytope_generator(poly_data, input_type="str")
        >>> print(next(g))
        A 4-dimensional reflexive lattice polytope in ZZ^4
    """
    from cytools.polytope import Polytope

    # input checking
    if favorable is not None and lattice is None:
        raise ValueError('Lattice must be specified. Options are "M" and "N".')

    if input_type not in ["file", "str"]:
        raise ValueError('"input_type" must be either "file" or "str"')

    if format not in ["ks", "ws"]:
        raise ValueError('Unsupported format. Options are "ks" and "ws".')

    # read data
    n_yielded = 0

    if input_type == "file":
        in_file = open(input)
        line = in_file.readline()
    else:
        in_string = input.split("\n")
        line = in_string.pop(0)

    if format == "ws":
        # read the polytopes as weight systems, one per line
        #
        # N.B.: the lines must be taken from the same source as for the "ks"
        # format (i.e. the *contents* of the file, when reading a file), and
        # blank lines must never be handed to PALP: an empty weight system
        # trips a C-level assertion that aborts the entire interpreter
        try:
            while (limit is None) or (n_yielded < limit):
                if line.strip():
                    # pass line to PALP
                    p = pypalp.Polytope(line)
                    vert = p.vertices()

                    # ensure reasonable shape
                    if len(vert.shape) == 0:
                        break
                    if vert.shape[0] < vert.shape[1]:
                        vert = vert.T

                    # build the Polytope
                    p = Polytope(
                        vert,
                        backend=backend,
                        deterministic_glsm_basis=deterministic_glsm_basis,
                    )

                    if (favorable is None) or (
                        p.is_favorable(lattice=lattice) == favorable
                    ):
                        n_yielded += 1
                        yield (p.dual() if dualize else p)

                # get next line ("" signals EOF; blank lines are just "\n")
                if input_type == "file":
                    line = in_file.readline()
                    if line == "":
                        break
                else:
                    if len(in_string) > 0:
                        line = in_string.pop(0)
                    else:
                        break
        finally:
            if input_type == "file":
                in_file.close()

        return

    # format is "ks"
    #
    # N.B.: like the "ws" branch above, the file must be closed on every
    # exit path, including an abandoned generator
    try:
        while limit is None or n_yielded < limit:
            if "M:" in line:
                h = line.split()
                n, m = int(h[0]), int(h[1])

                # add vertices
                vert = []
                for i in range(n):
                    if input_type == "file":
                        vert.append([int(c) for c in in_file.readline().split()])
                    else:
                        vert.append([int(c) for c in in_string.pop(0).split()])

                vert = np.asarray(vert)

                # ensure reasonable shape
                if vert.shape != (n, m):
                    raise ValueError("Dimensions of array do not match")
                if m > n:
                    vert = vert.T

                # build the Polytope
                p = Polytope(
                    vert,
                    backend=backend,
                    deterministic_glsm_basis=deterministic_glsm_basis,
                )
                if (favorable is None) or (
                    p.is_favorable(lattice=lattice) == favorable
                ):
                    n_yielded += 1
                    yield (p.dual() if dualize else p)

            # get next line
            if input_type == "file":
                line = in_file.readline()

                for i in range(5):
                    if line != "":
                        break
                    line = in_file.readline()
                else:
                    in_file.close()
                    break
            else:
                if len(in_string) > 0:
                    line = in_string.pop(0)
                else:
                    break
    finally:
        if input_type == "file":
            in_file.close()


def read_polytopes(
    input: str,
    input_type: str = "file",
    format: str = "ks",
    backend: str = None,
    deterministic_glsm_basis: bool = False,
    as_list: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    lattice: str = None,
    limit: int = None,
) -> 'Generator["Polytope", None, None] | list["Polytope"]':
    """Read polytopes from a file or a string.

    The polytopes can be specified by their vertices, in the format used in the
    Kreuzer-Skarke database, or by weight systems.

    In the `"ks"` format, each polytope consists of a header line containing
    `M:`, whose first two numbers are the dimensions `n m` of the matrix that
    follows, and `n` lines with `m` integers each. The vertices are the columns of
    this matrix if it has more columns than rows, and its rows otherwise. In the
    `"ws"` format, each non-empty line is a weight system, such as `5 1 1 1 1 1`
    (the degree followed by the weights), and the polytope is the one that PALP
    associates with it.

    This function can also be imported from the package root.

    Args:
        input (str): The name of the file to read, or the string containing the
            polytopes, depending on `input_type`.
        input_type (str): Whether `input` is the name of a file or a string
            containing the data. The options are `"file"` and `"str"`.
        format (str): The format of the data. The options are `"ks"`, the format
            used in the Kreuzer-Skarke database, where the polytopes are specified
            by their vertices, and `"ws"`, where each line is a weight system.
        backend (str | None): The backend used to construct the convex hull of
            each [`Polytope`][cytools.polytope.Polytope]. The options are `"ppl"`,
            `"qhull"`, and `"palp"`. If not specified, it is chosen automatically.
        deterministic_glsm_basis (bool): Whether to choose the GLSM (and thus
            divisor) basis of each polytope in a deterministic manner, so that it
            is consistent across different machines. The basis chosen with this
            flag may differ from the default one.
        as_list (bool): Whether to return a list of polytopes instead of a
            generator.
        dualize (bool): Whether to return the dual of each polytope instead of
            the polytope itself.
        favorable (bool | None): If true, only favorable polytopes are returned,
            and if false, only non-favorable ones. Favorability is checked before
            dualizing. If not specified, all polytopes are returned.
        lattice (str | None): The lattice, `"N"` or `"M"`, with respect to which
            favorability is checked. It must be specified if `favorable` is.
        limit (int | None): The maximum number of polytopes to return. If not
            specified, all of them are returned.

    Returns:
        Generator[Polytope, None, None] | list[Polytope]: A generator of
            [`Polytope`][cytools.polytope.Polytope] objects, or a list of them if
            `as_list` is true.

    Raises:
        ValueError: If `favorable` is specified without `lattice`, if `input_type`
            or `format` is not one of the allowed options, or if the dimensions of
            a vertex matrix do not match its header. If a generator is returned,
            these errors are raised when it is first advanced.

    Examples:
        We read a polytope from a string taken from the Kreuzer-Skarke database.

        >>> from cytools import read_polytopes
        >>> poly_data = '''4 5  M:10 5 N:376 5 H:272,2 [540]
        ...    1    0    0    0   -9
        ...    0    1    0    0   -6
        ...    0    0    1    0   -1
        ...    0    0    0    1   -1
        ... '''
        >>> polys = read_polytopes(poly_data, input_type="str", as_list=True)
        >>> len(polys)
        1
        >>> print(polys[0])
        A 4-dimensional reflexive lattice polytope in ZZ^4
        >>> polys[0].h11(lattice="N"), polys[0].h21(lattice="N")
        (2, 272)

        We can also construct polytopes from weight systems. Here we use the
        weight system of the quintic, and take the dual of the resulting polytope.

        >>> ws_polys = read_polytopes("5 1 1 1 1 1", input_type="str", format="ws",
        ...                           dualize=True, as_list=True)
        >>> ws_polys[0].h11(lattice="N")
        1
    """
    g = polytope_generator(
        input,
        input_type=input_type,
        format=format,
        backend=backend,
        deterministic_glsm_basis=deterministic_glsm_basis,
        dualize=dualize,
        favorable=favorable,
        lattice=lattice,
        limit=limit,
    )

    if as_list:
        return list(g)
    else:
        return g


def fetch_polytopes(
    h11: int = None,
    h12: int = None,
    h13: int = None,
    h21: int = None,
    h22: int = None,
    h31: int = None,
    chi: int = None,
    lattice: str = "N",
    dim: int = 4,
    n_points: int = None,
    n_vertices: int = None,
    n_dual_points: int = None,
    n_facets: int = None,
    limit: int = 1000,
    samples: int = None,
    sample_seed: int = None,
    timeout: int = 60,
    as_list: bool = True,
    backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0,
) -> 'Generator["Polytope", None, None] | list["Polytope"]':
    """Fetch reflexive polytopes from the Kreuzer-Skarke or Schöller-Skarke database.

    Four-dimensional polytopes are fetched from the
    [Kreuzer-Skarke database](http://hep.itp.tuwien.ac.at/~kreuzer/CY/), and
    five-dimensional ones from the
    [Schöller-Skarke database](http://rgc.itp.tuwien.ac.at/fourfolds/). The
    polytopes can be filtered by the Hodge numbers and Euler characteristic of the
    associated Calabi-Yau hypersurfaces, and, for four-dimensional polytopes, by
    their numbers of points, vertices, and facets.

    This function can also be imported from the package root.

    Args:
        h11 (int | None): The Hodge number $h^{1,1}$ of the Calabi-Yau
            hypersurface.
        h12 (int | None): The Hodge number $h^{1,2}$ of the Calabi-Yau
            hypersurface.
        h13 (int | None): The Hodge number $h^{1,3}$ of the Calabi-Yau
            hypersurface. Only used for five-dimensional polytopes.
        h21 (int | None): The Hodge number $h^{2,1}$ of the Calabi-Yau
            hypersurface. This is equivalent to `h12`.
        h22 (int | None): The Hodge number $h^{2,2}$ of the Calabi-Yau
            hypersurface. Only used for five-dimensional polytopes.
        h31 (int | None): The Hodge number $h^{3,1}$ of the Calabi-Yau
            hypersurface. This is equivalent to `h13`.
        chi (int | None): The Euler characteristic of the Calabi-Yau
            hypersurface.
        lattice (str | None): The lattice in which the polytopes are defined,
            `"N"` or `"M"`. The Hodge numbers and the Euler characteristic refer
            to the Calabi-Yau hypersurface associated with polytopes in this
            lattice, and favorability is checked with respect to it. It must be
            specified if any Hodge number, the Euler characteristic, or
            `favorable` is specified.
        dim (int): The dimension of the polytopes. The options are 4 and 5.
        n_points (int | None): The number of lattice points of the polytopes.
            Only used for four-dimensional polytopes.
        n_vertices (int | None): The number of vertices of the polytopes. Only
            used for four-dimensional polytopes.
        n_dual_points (int | None): The number of lattice points of the dual
            polytopes. Only used for four-dimensional polytopes.
        n_facets (int | None): The number of facets of the polytopes. Only used
            for four-dimensional polytopes.
        limit (int): The maximum number of polytopes to return.
        samples (int | None): If specified, this many polytopes are chosen at
            random, without replacement, among the (at most `limit`) fetched ones.
            It must not be larger than `limit`.
        sample_seed (int | None): The seed of the random number generator used for
            sampling. If not specified, a different sample is drawn every time.
        timeout (int): The maximum number of seconds to wait for the server to
            return the data.
        as_list (bool): Whether to return a list of polytopes instead of a
            generator.
        backend (str | None): The backend used to construct the convex hull of
            each [`Polytope`][cytools.polytope.Polytope]. The options are `"ppl"`,
            `"qhull"`, and `"palp"`. If not specified, it is chosen automatically.
        deterministic_glsm_basis (bool): Whether to choose the GLSM (and thus
            divisor) basis of each polytope in a deterministic manner, so that it
            is consistent across different machines. The basis chosen with this
            flag may differ from the default one.
        dualize (bool): Whether to return the dual of each polytope instead of
            the polytope itself.
        favorable (bool | None): If true, only favorable polytopes are returned,
            and if false, only non-favorable ones. Favorability is checked before
            dualizing. If not specified, all polytopes are returned.
        verbosity (int): The verbosity level. If it is 1 or larger, the URL of
            the request is printed, and if it is 2 or larger, the received data is
            printed as well.

    Returns:
        Generator[Polytope, None, None] | list[Polytope]: A list of
            [`Polytope`][cytools.polytope.Polytope] objects, or a generator of
            them if `as_list` is false.

    Raises:
        ValueError: If `dim` or `lattice` is not one of the allowed options, if
            `lattice` is not specified when required, if `samples` is larger than
            `limit`, if both `h12` and `h21` (or `h13` and `h31`) are given with
            different values, or if the given Hodge numbers and Euler
            characteristic are inconsistent.

    Note:
        The Kreuzer-Skarke database does not store favorability data. Thus, when
        `favorable` is specified, additional polytopes are fetched so that the
        requested limit can be saturated after filtering them. However, it may
        happen that fewer polytopes than requested are returned even though more
        exist. To verify that no more polytopes with the requested properties
        exist, one can increase the limit significantly and check whether more
        polytopes are returned.

    Examples:
        We fetch polytopes from the Kreuzer-Skarke and Schöller-Skarke databases
        with a few different parameters.

        >>> from cytools import fetch_polytopes
        >>> g = fetch_polytopes(h11=27, as_list=False)  # a generator of polytopes
        >>> print(next(g))
        A 4-dimensional reflexive lattice polytope in ZZ^4
        >>> polys = fetch_polytopes(h11=27, limit=100)  # a list of polytopes
        >>> len(polys)
        100
        >>> g_5d = fetch_polytopes(h11=1000, dim=5, limit=100, as_list=False)
        >>> print(next(g_5d))
        A 5-dimensional reflexive lattice polytope in ZZ^5
    """
    # input checking
    # --------------
    if dim not in (4, 5):
        raise ValueError("Only polytopes of dimension 4 or 5 are available.")

    if lattice not in ("N", "M", None):
        raise ValueError("Options for lattice are 'N' and 'M'.")

    if favorable is not None:
        if lattice is None:
            raise ValueError("Must specify lattice when checking favorability.")

        fetch_limit = (5 if favorable else 10) * limit + 100
    else:
        fetch_limit = limit

    if samples is not None:
        if (limit is None) or (samples > limit):
            raise ValueError("The number of samples must be <= limit.")

    # hodge numbers
    if (h12 is not None) and (h21 is not None) and (h12 != h21):
        raise ValueError("Only one of h12 or h21 should be specified.")

    if (h13 is not None) and (h31 is not None) and (h13 != h31):
        raise ValueError("Only one of h13 or h31 should be specified.")

    if (h12 is None) and (h21 is not None):
        h12 = h21
    if (h13 is None) and (h31 is not None):
        h13 = h31

    # grab the polytopes
    # ------------------
    if dim == 4:
        # further input checking...
        if h13 is not None or h22 is not None:
            print("Ignoring inputs for h13 and h22.")

        if (lattice is None) and (
            (h11 is not None) or (h12 is not None) or (chi is not None)
        ):
            raise ValueError(
                "Lattice must be specified when Hodge numbers "
                "or Euler characteristic are given."
            )
        if lattice == "N":
            h11, h12 = h12, h11
            chi = -chi if chi is not None else None

        if (
            (chi is not None)
            and (h11 is not None)
            and (h12 is not None)
            # h11/h12/chi have been converted to the M-lattice convention
            # above, in which chi = 2*(h11 - h21) and h21 == h12
            and (chi != 2 * (h11 - h12))
        ):
            raise ValueError("Inconsistent Euler characteristic input.")

        # build/send a request
        variables = [
            h11,
            h12,
            n_points,
            n_vertices,
            n_dual_points,
            n_facets,
            chi,
            fetch_limit,
        ]
        names = ["h11", "h12", "M", "V", "N", "F", "chi", "L"]

        parameters = {
            name: str(var) for name, var in zip(names, variables) if var is not None
        }

        r = requests.get(
            "http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi",
            params=parameters,
            timeout=timeout,
        )
    else:
        # further input checking...
        if (lattice is None) and ((h11 is not None) or (h13 is not None)):
            raise ValueError("Lattice must be specified when h11 or h13 are given.")

        if lattice == "N":
            h11, h13 = h13, h11

        if (
            (chi is not None)
            and (h11 is not None)
            and (h12 is not None)
            and (h13 is not None)
            and (chi != 48 + 6 * (h11 - h12 + h13))
        ):
            raise ValueError("Inconsistent Euler characteristic input.")

        if (
            (h22 is not None)
            and (h11 is not None)
            and (h12 is not None)
            and (h13 is not None)
            and (h22 != 44 + 6 * h11 - 2 * h12 + 4 * h13)
        ):
            raise ValueError("Inconsistent h22 input.")

        # build/send a request
        variables = [h11, h12, h13, h22, chi, fetch_limit]
        names = ["h11", "h12", "h13", "h22", "chi", "limit"]

        url = "http://rgc.itp.tuwien.ac.at/fourfolds/db/5d_reflexive"
        for i, vr in enumerate(variables):
            if vr is not None:
                url += f",{names[i]}={vr}"
        url += ".txt"

        r = requests.get(url, timeout=timeout)

    data_str = r.text

    # verbosity
    if verbosity >= 1:
        print(f"Fetched from URL = '{r.url}'...")
        if verbosity >= 2:
            print(f"and received {data_str}")

    # sample
    if samples is not None:
        if dim == 4:
            result_text = re.split(r"<b>Result:</b>\n", data_str, maxsplit=1)[1]
            items = re.split(r"\n(?=\d+ \d+\s+M:)", result_text)
            items = [item.strip() for item in items if item.strip()]
        else:
            items = data_str.split("\n")

        # check if we actually need to sample (otherwise, just use full list)
        if len(items) > samples:
            # a local generator: seeding the global one would reset the
            # caller's stream, and with sample_seed=None it reseeded from
            # entropy. RandomState (not default_rng) keeps the draws that
            # existing sample_seed values already produce.
            rng = np.random.RandomState(sample_seed)
            sampled = rng.choice(items, size=samples, replace=False)
        else:
            sampled = items

        # rejoin in the previous format
        data_str = "\n".join(sampled) + "\n"
        limit = samples

    # return the generator/list based off of output of request
    return read_polytopes(
        data_str,
        input_type="str",
        format=("ks" if dim == 4 else "ws"),
        backend=backend,
        deterministic_glsm_basis=deterministic_glsm_basis,
        as_list=as_list,
        dualize=dualize,
        favorable=favorable,
        lattice=lattice,
        limit=limit,
    )


# point manipulations
# -------------------
def lll_reduce(pts_in: ArrayLike, transform: bool = False):
    r"""Apply LLL reduction to a set of lattice points.

    The points are the rows of `pts_in`. The LLL algorithm is applied to the rows
    of `pts_in.T`, i.e. the vectors of coordinates of the points. This amounts to a
    unimodular change of lattice coordinates $p \mapsto Ap$, with
    $A \in GL(n, \mathbb{Z})$, that makes these vectors short and nearly
    orthogonal. If the points span a sublattice of dimension $d < n$, the
    coordinates that vanish for all points come first, so that the last $d$
    columns of the result give the coordinates of the points in the sublattice.

    Args:
        pts_in (ArrayLike): The integer points, one per row.
        transform (bool): Whether to also return the transformation matrix and its
            inverse.

    Returns:
        np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]: The reduced
            points, one per row. If `transform` is true, a tuple
            `(pts_red, (A, Ainv))` is returned instead, where `A` is the
            unimodular matrix satisfying `pts_red.T == A @ pts_in.T` and `Ainv` is
            its inverse.

    Examples:
        We reduce a set of points spanning a 2-dimensional sublattice of
        $\mathbb{Z}^3$. The first coordinate of all the reduced points vanishes.

        >>> from cytools.utils import lll_reduce
        >>> lll_reduce([[1,1,0],[0,1,1],[1,2,1],[0,0,0]])
        array([[0, 1, 0],
               [0, 0, 1],
               [0, 1, 1],
               [0, 0, 0]])
    """
    pts = np.array(pts_in)

    # lll-reduction
    # given input M, this solves for a T,L such that T@M = L
    # with T unimodular. Thus T@M represents an equivalent polytope if
    # the **columns** of M are the points
    pts = pts.T

    if transform is True:
        pts_red, transf = flint.fmpz_mat(pts.tolist()).lll(transform=True)
    else:
        pts_red = flint.fmpz_mat(pts.tolist()).lll(transform=False)

    pts_red = pts_red.transpose()  # map points back to rows

    # convert to numpy
    pts_red = np.array(pts_red.tolist(), dtype=int)

    if transform is True:
        A = np.array(transf.tolist(), dtype=int)
        Ainv = np.array(transf.inv(integer=True).tolist(), dtype=int)

        # check that Ainv is indeed an inverse
        # (sometimes it's off by a sign)
        check_inverse = Ainv.dot(A)
        id_mat = np.eye(len(A), dtype=int)

        if all((check_inverse == id_mat).flatten()):
            pass
        elif all((check_inverse == -id_mat).flatten()):
            Ainv *= -1
        else:
            raise RuntimeError("Problem finding inverse matrix")

        return pts_red, (A, Ainv)
    else:
        return pts_red


def find_new_affinely_independent_points(pts: ArrayLike) -> np.ndarray:
    r"""Find new points that make a set of points full-dimensional.

    If the input points affinely span a $d$-dimensional subspace of
    $\mathbb{R}^n$, this function returns $n-d$ integer points such that the input
    points together with the new ones affinely span all of $\mathbb{R}^n$. This is
    useful to turn a point configuration that is not full-dimensional into one
    that is.

    Args:
        pts (ArrayLike): The points, one per row. There must be at least one.

    Returns:
        np.ndarray: The $n-d$ new points, one per row.

    Raises:
        ValueError: If the list of points is empty.

    Examples:
        We construct three points spanning a plane in 3 dimensions, and find a
        new point off that plane.

        >>> from cytools.utils import find_new_affinely_independent_points
        >>> find_new_affinely_independent_points([[1,0,1],[0,0,1],[0,1,1]])
        array([[1, 0, 2]])
    """
    # input checking
    if len(pts) == 0:
        raise ValueError("List of points cannot be empty.")

    # cast to numpy array
    # (copy, so that the caller's array is never modified in-place below)
    pts = np.array(pts)
    shape = pts.shape

    # translate, append unit vector
    translation = pts[0].copy()
    pts -= translation

    if shape[0] == 1:
        pts = np.append(pts, [[1] + [0] * (shape[1] - 1)], axis=0)

    dim = np.linalg.matrix_rank(pts)

    # make basis of points
    basis = []
    basis_dim = 0

    for pt in pts:
        basis.append(pt.tolist())

        new_rank = np.linalg.matrix_rank(basis)
        if basis_dim < new_rank:
            basis_dim = new_rank
        else:
            basis.pop()

        if basis_dim == dim:
            break

    # find independent points
    k, n_k = flint.fmpz_mat(basis).nullspace()
    new_pts = np.array(k.transpose().tolist(), dtype=int)[:n_k, :]
    if shape[0] == 1:
        new_pts = np.append(new_pts, [[1] + [0] * (shape[1] - 1)], axis=0)

    return new_pts + translation


# heights to/from kahlers
# -----------------------
# TEMPORARY
def project_heights_to_kahler(poly, heights_in, prime_divisors=None):
    r"""Project a height vector onto the basis of divisors.

    Given heights $h_0, h_1, \ldots$ for the origin and the prime toric divisors
    $D_1, D_2, \ldots$ of the polytope, this computes the divisor class
    $\sum_i (h_i - h_0) D_i$, written in the basis of divisors. The prime toric
    divisors that are not in the basis are eliminated using `prime_divisors`,
    which gives each of them as a linear combination of the basis divisors.

    This is one of a few temporary helpers for converting between heights and
    Kähler parameters. It assumes that `poly` is a four-dimensional reflexive
    polytope whose prime toric divisors are the $h^{1,1}+4$ nonzero lattice
    points not interior to facets (i.e. that `poly` is favorable), and it uses
    the basis of divisors given by
    [`Polytope.glsm_basis`][cytools.polytope.Polytope.glsm_basis].

    Args:
        poly (Polytope): The polytope.
        heights_in (ArrayLike): The heights, a vector of length $h^{1,1}+5$ with
            the height of the origin first, followed by those of the prime toric
            divisors.
        prime_divisors (ArrayLike | None): The prime toric divisors that are not
            in the basis, in order, written as linear combinations of the basis
            divisors, one per row. If not specified, they are read off from the
            rays of the toric effective cone of a Calabi-Yau hypersurface obtained
            from a triangulation of `poly`.

    Returns:
        np.ndarray: A vector of length $h^{1,1}+5$, indexed like `heights_in`,
            whose entries at the positions of the basis divisors are the
            coefficients of the divisor class, and whose other entries are zero.

    Warning:
        The rays of the toric effective cone are primitive, so the default
        `prime_divisors` are only correct when the classes of the non-basis prime
        toric divisors are primitive vectors in the basis.
    """
    basis = [i - 1 for i in poly.glsm_basis(include_origin=True)]
    if prime_divisors is None:
        prime_divisors = np.array(
            [
                rr
                for r, rr in enumerate(
                    poly.triangulate(verbosity=0).get_cy().toric_effective_cone().rays()
                )
                if r not in basis
            ],
            dtype=float,
        )
    extra_divs = [i for i in range(poly.h11(lattice="N") + 4) if i not in basis]
    origin_height = heights_in[0]
    kahler_parameters = np.array([i - origin_height for i in heights_in[1:]])
    for e, ee in enumerate(prime_divisors):
        prime_ind = extra_divs[e]
        prime_height = kahler_parameters[prime_ind]
        lin_rel = np.zeros(poly.h11(lattice="N") + 4)
        lin_rel[basis] = ee
        lin_rel[prime_ind] = -1
        corr = prime_height * lin_rel
        kahler_parameters = np.array(kahler_parameters) + np.array(corr)
    return np.concatenate(([0], kahler_parameters))


def heights_to_kahler(poly, heights_in, prime_divisors=None):
    r"""Convert a height vector to the corresponding divisor class in the basis.

    This is the same as
    [`project_heights_to_kahler`][cytools.utils.project_heights_to_kahler], but
    only the $h^{1,1}$ coefficients of the divisor class in the basis of divisors
    are returned.

    This is one of a few temporary helpers for converting between heights and
    Kähler parameters. It assumes that `poly` is a four-dimensional reflexive
    polytope whose prime toric divisors are the $h^{1,1}+4$ nonzero lattice
    points not interior to facets (i.e. that `poly` is favorable), and it uses
    the basis of divisors given by
    [`Polytope.glsm_basis`][cytools.polytope.Polytope.glsm_basis].

    Args:
        poly (Polytope): The polytope.
        heights_in (ArrayLike): The heights, a vector of length $h^{1,1}+5$ with
            the height of the origin first, followed by those of the prime toric
            divisors.
        prime_divisors (ArrayLike | None): The prime toric divisors that are not
            in the basis, written as linear combinations of the basis divisors.
            See [`project_heights_to_kahler`][cytools.utils.project_heights_to_kahler]
            for details.

    Returns:
        np.ndarray: The $h^{1,1}$ coefficients of the divisor class
            $\sum_i (h_i - h_0) D_i$ in the basis of divisors.
    """
    basis = poly.glsm_basis()
    return project_heights_to_kahler(poly, heights_in, prime_divisors)[basis]


def kahler_to_heights(poly, kahler_in):
    r"""Convert a divisor class in the basis to a height vector.

    The heights of the basis divisors are set to the given coefficients, and all
    the other heights, including that of the origin, are set to zero. This is a
    right inverse of [`heights_to_kahler`][cytools.utils.heights_to_kahler].

    This is one of a few temporary helpers for converting between heights and
    Kähler parameters. It assumes that `poly` is a four-dimensional reflexive
    polytope whose prime toric divisors are the $h^{1,1}+4$ nonzero lattice
    points not interior to facets (i.e. that `poly` is favorable), and it uses
    the basis of divisors given by
    [`Polytope.glsm_basis`][cytools.polytope.Polytope.glsm_basis].

    Args:
        poly (Polytope): The polytope.
        kahler_in (ArrayLike): The $h^{1,1}$ coefficients of the divisor class in
            the basis of divisors.

    Returns:
        np.ndarray: The heights, a vector of length $h^{1,1}+5$ with the height of
            the origin first, followed by those of the prime toric divisors.
    """
    basis = list(poly.glsm_basis(include_origin=True))
    kahler_gen = iter(kahler_in)
    return np.array(
        [
            next(kahler_gen) if i in basis else 0
            for i in range(poly.h11(lattice="N") + 5)
        ]
    )
