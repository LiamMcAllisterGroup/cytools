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

# 'standard' imports
import ast
import fractions
import functools
import itertools
import math
import requests
import subprocess
from typing import Generator

# 3rd party imports
import flint
import numpy as np
from numpy.typing import ArrayLike
import pypalp
import scipy.sparse as sp

# CYTools imports
from cytools import config


# custom decorators
# -----------------
# class instance caching
# (lru_cache persists for all class instances... that is not desired...)
def instanced_lru_cache(maxsize=128):
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
    """
    **Description:**
    Compute the greatest common (floating-point) divisor of a and b. This is
    simply the largest floating point number that divides a and b. Uses the
    Euclidean algorithm.

    Warning - unexpected/buggy behavior can occur if b starts tiny. E.g.,
    gcd_float(100,0.1,0.2) returns 100.

    This only seems to be a risk if b *starts* below tol.

    **Arguments:**
    - `a`: The first number.
    - `b`: The second number.
    - `tol`: The tolerance for rounding.

    **Returns:**
    The gcd of a and b.

    **Example:**
    We compute the gcd of two floats. This function is useful since the
    standard gcd functions raise an error for non-integer values.
    ```python {2}
    from cytools.utils import gcd_float
    gcd_float(0.2, 0.5)
    # Should be 0.1, but there are small rounding errors
    # 0.09999999999999998
    ```
    """
    if abs(b) < tol:
        return abs(a)
    return gcd_float(b, a % b, tol)

# variant that computes gcd over all elements in arr
gcd_list = lambda arr: functools.reduce(gcd_float, arr)

# linear algebra
# --------------
def integral_nullspace(M, reduce_by_gcd=True):
    """
    Returns the integral nullspace as column vectors
    """
    null, nullity = flint.fmpz_mat(M.tolist()).nullspace()
    
    # trim extra columns
    null = np.array(null.tolist(), dtype=int)[:,:nullity]
    
    # reduce by gcd
    if reduce_by_gcd:
        gcds = np.array([math.gcd(*c) for c in null.T])
        null = null//gcds
    
    return null

# flint conversion
# ----------------
def float_to_fmpq(c: float) -> flint.fmpq:
    """
    **Description:**
    Converts a float to an fmpq (Flint's rational number class).

    See https://flintlib.org/doc/fmpq.html

    **Arguments:**
    - `c`: The input number.

    **Returns:**
    The rational number that most reasonably approximates the input.

    **Example:**
    We convert a few floats to rational numbers.
    ```python {2}
    from cytools.utils import float_to_fmpq
    float_to_fmpq(0.1), float_to_fmpq(0.333333333333), float_to_fmpq(2.45)
    # (1/10, 1/3, 49/20)
    ```
    """
    f = fractions.Fraction(c).limit_denominator()
    return flint.fmpq(f.numerator, f.denominator)

def fmpq_to_float(c: flint.fmpq) -> float:
    """
    **Description:**
    Converts an fmpq (Flint's rational number class) to a float.

    See https://flintlib.org/doc/fmpq.html

    **Arguments:**
    - `c`: The input rational number.

    **Returns:**
    The number as a float.

    **Example:**
    We convert a few rational numbers to floats.
    ```python {3}
    from cytools.utils import fmpq_to_float
    from flint import fmpq
    fmpq_to_float(fmpq(1,2)), fmpq_to_float(fmpq(1,3)),\
                                                    fmpq_to_float(fmpq(49,20))
    # (0.5, 0.3333333333333333, 2.45)
    ```
    """
    return int(c.p) / int(c.q)

def array_to_flint(arr: np.ndarray, t: "int | float" = None) -> np.ndarray:
    """
    **Description:**
    Converts a numpy array with either:
        1) 64-bit integer entries or
        2) float entries
    to Flint type (fmpz or fmpq for integer or rational numbers, respectively).

    See https://flintlib.org/doc/fmpz.html and
        https://flintlib.org/doc/fmpq.html

    **Arguments:**
    - `arr`: A numpy array with either 64-bit integer or float entries.

    **Returns:**
    A numpy array with either fmpz or fmpq entries.

    **Example:**
    We convert an integer array to an fmpz array.
    ```python {3}
    from cytools.utils import array_int_to_fmpz
    arr = [[1,0,0],[0,2,0],[0,0,3]]
    array_int_to_fmpz(arr)
    # array([[1, 0, 0],
    #        [0, 2, 0],
    #        [0, 0, 3]], dtype=object)
    ```
    """
    # type conversion function
    if t is None:
        t = arr.dtype

    if t == int:
        f = lambda n: flint.fmpz(int(n))
    else:
        f = float_to_fmpq

    return np.vectorize(f)(arr).astype(object)

# some type-specific aliases
array_int_to_fmpz = lambda arr: array_to_flint(arr, t=int)
array_float_to_fmpq = lambda arr: array_to_flint(arr, t=float)

def array_from_flint(arr: np.ndarray, t=None) -> np.ndarray:
    """
    **Description:**
    Converts a numpy array with fmpz/fmpq (Flint's integer/float number class)
    entries to 64-bit integer/float entries.

    **Arguments:**
    - `arr`: A numpy array with fmpz/fmpq entries.

    **Returns:**
    A numpy array with 64-bit integer/float entries.
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
array_fmpz_to_int = lambda arr: array_from_flint(arr, t=flint.fmpz)
array_fmpq_to_float = lambda arr: array_from_flint(arr, t=flint.fmpq)

# sparse conversions
# ------------------
def to_sparse(
    arr: "dict | list", sparse_type: str = "dok"
) -> "sp.dok_matrix | sp.csr_matrix":
    """
    **Description:**
    Converts a (manually implemented) sparse matrix of the form
    [[a,b,M_ab], ...] or a dictionary of the form {(a,b):M_ab, ...} to a formal
    dok_matrix or to a csr_matrix.

    **Arguments:**
    - `arr`: A list of the form [[a,b,M_ab],...] or a dictionary of the
        form [(a,b):M_ab,...].
    - `sparse_type`: The type of sparse matrix to return. The options are "dok"
        and "csr".

    **Returns:**
    The sparse matrix in dok_matrix or csr_matrix format.

    **Example:**
    We convert the 5x5 identity matrix into a sparse matrix.
    ```python {3,6}
    from cytools.utils import to_sparse
    id_array = [[0,0,1],[1,1,1],[2,2,1],[3,3,1],[4,4,1]]
    to_sparse(id_array)
    # <5x5 sparse matrix of type '<class 'numpy.float64'>'
    #        with 5 stored elements in Dictionary Of Keys format>
    to_sparse(id_array, sparse_type="csr")
    # <5x5 sparse matrix of type '<class 'numpy.float64'>'
    #        with 5 stored elements in Compressed Sparse Row format>
    ```
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
    """
    **Description:**
    Converts a symmetric sparse tensor of the form {(a,b,...,c): M_ab...c, ...}
    to a dense tensor.

    Optionally, it then applies a basis transformation.

    **Arguments:**
    - `tensor`: A sparse tensor of the form {(a,b,...,c):M_ab...c, ...}.
    - `basis`: A matrix where the rows are the basis elements.

    **Returns:**
    A dense tensor.

    **Example:**
    We construct a simple tensor and then convert it to a dense array. We
    consider the same example as for the
    [`filter_tensor_indices`](#filter_tensor_indices) function, but now we have
    to specify the basis in matrix form.
    ```python {4}
    from cytools.utils import symmetric_sparse_to_dense_in_basis
    tensor = {(0,1):0, (1,1):1, (1,2):2, (1,3):3, (2,3):4}
    basis = [[0,1,0,0],[0,0,0,1]]
    symmetric_sparse_to_dense(tensor, basis)
    # array([[1, 3],
    #        [3, 0]])
    ```
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
    """
    **Description:**
    Converts a dense symmetric tensor to a sparse tensor of the form
    {(a,b,...,c):M_ab...c, ...}.

    The upper triangular indices are used. That is, a<=b<=...<=c.

    Optionally, it applies a basis transformation.

    **Arguments:**
    - `tensor`: A dense symmetric tensor.
    - `basis`: A matrix where the rows are the basis elements.

    **Returns:**
    A sparse tensor of the form {(a,b,...,c):M_ab...c, ...}.

    **Example:**
    We construct a simple tensor and then convert it to a dense array. We
    consider the same example as for the
    [`filter_tensor_indices`](#filter_tensor_indices) function, but now we have
    to specify the basis in matrix form.
    ```python {3}
    from cytools.utils import symmetric_dense_to_sparse
    tensor = [[1,2],[2,3]]
    symmetric_dense_to_sparse(tensor)
    # {(0, 0): 1, (0, 1): 2, (1, 1): 3}
    ```
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
    """
    **Description:**
    Selects a specific subset of indices from a tensor.

    The tensor is reindexed so that indices are in the range 0..len(indices)
    with the ordering specified by the input indices. This function can be used
    to convert the tensor of intersection numbers to a given basis.

    **Arguments:**
    - `tensor`: The input symmetric sparse tensor of the form of a dictionary
        {(a,b,...,c):M_ab...c, ...}.
    - `indices`: The list of indices that will be preserved.

    **Returns:**
    A dictionary describing a tensor in the same format as the input, but only with the desired indices.

    **Example:**
    We construct a simple tensor and then filter some of the indices. We also
    give a concrete example of when this is used for intersection numbers.
    ```python {3,10}
    from cytools.utils import filter_tensor_indices
    tensor = {(0,1):0, (1,1):1, (1,2):2, (1,3):3, (2,3):4}
    filter_tensor_indices(tensor, [1,3])
    # {(0, 0): 1, (0, 1): 3}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    t = p.triangulate()
    v = t.get_toric_variety()
    intnums_nobasis = v.intersection_numbers(in_basis=False)
    intnums_inbasis = v.intersection_numbers(in_basis=True)
    intnums_inbasis == filter_tensor_indices(intnums_nobasis, v.divisor_basis())
    # True
    ```
    """
    # map from index to its count in indices object
    reindex = {ind: i for i, ind in enumerate(indices)}

    # only keep entries whose indices match those in indices
    filtered = {
        key: val for key, val in tensor.items() if all(c in indices for c in key)
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
    """
    **Description:**
    Solves the sparse linear system M*x + C = 0.

    **Arguments:**
    - `M`: The matrix.
    - `C`: The constant term.
    - `backend`: The solver to use. Options are "all", "sksparse" and "scipy".
        When set to "all" it tries all available backends.
    - `check`: Whether to explicitly check the solution to the linear system.
    - `backend_error_tol`: Error tolerance for the solution.
    - `verbosity`: The verbosity level.
        - verbosity = 0: Do not print anything.
        - verbosity = 1: Print warnings when backends fail.

    **Returns:**
    Floating-point solution to the linear system.

    **Example:**
    We solve a very simple linear equation.
    ```python {5}
    from cytools.utils import to_sparse, solve_linear_system
    id_array = [[0,0,1],[1,1,1],[2,2,1],[3,3,1],[4,4,1]]
    M = to_sparse(id_array, sparse_type="csr")
    C = [1,1,1,1,1]
    solve_linear_system(M, C)
    # array([-1., -1., -1., -1., -1.])
    ```
    """
    # input checking
    backends = ["all", "sksparse", "scipy"]
    if backend not in backends:
        raise ValueError(f"Invalid backend... options are {backends}.")

    # solve the system
    solution = None

    if backend == "all":
        for s in backends[1:]:
            solution = solve_linear_system(
                M,
                C,
                backend=s,
                check=check,
                backend_error_tol=backend_error_tol,
                verbosity=verbosity,
            )
            if solution is not None:
                return solution

    elif backend == "sksparse":
        try:
            from sksparse.cholmod import cholesky_AAt

            factor = cholesky_AAt(M.transpose())
            solution = factor(-M.transpose() * C)
        except:
            if verbosity >= 1:
                print("Linear backend error: sksparse failed.")

    elif backend == "scipy":
        try:
            solution = sp.linalg.spsolve(M.transpose() * M, -M.transpose() * C).tolist()
        except:
            if verbosity >= 1:
                print("Linear backend error: scipy failed.")

    # check/return solution
    if solution is None:
        return None

    if check:
        res = M.dot(solution) + C
        max_error = max(abs(s) for s in res.flat)

        if max_error > backend_error_tol:
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
    """
    **Description:**
    Specifies a basis of divisors for the toric variety or Calabi-Yau manifold,
    which in turn specifies a dual basis of curves. This can be done with a
    vector specifying the indices of the prime toric divisors, or as a matrix
    where each row is a linear combination of prime toric divisors.

    :::important
    This function should generally not be called directly by the user. Instead,
    it is called by the [`set_divisor_basis`](./toricvariety#set_divisor_basis)
    function of the [`ToricVariety`](./toricvariety) class, or the
    [`set_divisor_basis`](./calabiyau#set_divisor_basis) function of the
    [`CalabiYau`](./calabiyau) class.
    :::

    :::note
    Only integral bases are supported by CYTools, meaning that all prime toric
    divisors must be able to be written as an integral linear combination of
    the basis divisors.
    :::

    **Arguments:**
    - `tv_or_cy`: The toric variety or Calabi-Yau whose basis will be set.
    - `basis`: Vector or matrix specifying a basis. When a vector is used, the
        entries will be taken as the indices of points of the polytope or prime
        divisors of the toric variety. When a matrix is used, the rows are
        taken as linear combinations of the aforementioned divisors.
    - `include_origin`: Whether to interpret the indexing specified by the
        input vector as including the origin.

    **Returns:**
    Nothing.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We consider a simple toric variety with two independent
    divisors. We first find the default basis it picks and then we set a basis
    of our choice.
    ```python {6}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    t = p.triangulate()
    v = t.get_toric_variety()
    v.divisor_basis() # We haven't set any basis
    # array([1, 6])
    v.set_divisor_basis([5,6]) # Here we set a basis
    v.divisor_basis() # We get the basis we set
    # array([5, 6])
    v.divisor_basis(as_matrix=True) # We get the basis in matrix form
    # array([[0, 0, 0, 0, 0, 1, 0],
    #        [0, 0, 0, 0, 0, 0, 1]])
    ```
    An example for more generic basis choices can be found in the
    [experimental features](./experimental) section.
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
                "The experimental features must be enabled to " "use generic bases."
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
    """
    **Description:**
    Specifies a basis of curves of the toric variety, which in turn specifies a
    dual basis of divisors. This can be done with a vector specifying the
    indices of the dual prime toric divisors or as a matrix with the rows being
    the basis curves, and the entries are the intersection numbers with the
    prime toric divisors. Note that when using a vector it is equivalent to
    using the same vector in the [`set_divisor_basis`](#set_divisor_basis)
    function.

    :::important
    This function should generally not be called directly by the user. Instead,
    it is called by the [`set_curve_basis`](./toricvariety#set_curve_basis)
    function of the [`ToricVariety`](./toricvariety) class, or the
    [`set_curve_basis`](./calabiyau#set_curve_basis) function of the
    [`CalabiYau`](./calabiyau) class.
    :::

    :::note
    Only integral bases are supported by CYTools, meaning that all toric curves
    must be able to be written as an integral linear combination of the basis
    curves.
    :::

    **Arguments:**
    - `tv_or_cy`: The toric variety or Calabi-Yau whose basis will be set.
    - `basis`: Vector or matrix specifying a basis. When a vector is used, the
        entries will be taken as indices of the standard basis of the dual to
        the lattice of prime toric divisors. When a matrix is used, the rows
        are taken as linear combinations of the aforementioned elements.
    - `include_origin`: Whether to interpret the indexing specified by the
        input vector as including the origin.

    **Returns:**
    Nothing.

    **Example:**
    This function is not intended to be directly used, but it is used in the
    following example. We consider a simple toric variety with two independent
    curves. We first find the default basis of curves it picks and then set a
    basis of our choice.
    ```python {6}
    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
    t = p.triangulate()
    v = t.get_toric_variety()
    v.curve_basis() # We haven't set any basis
    # array([1, 6])
    v.set_curve_basis([5,6]) # Here we set a basis
    v.curve_basis() # We get the basis we set
    # array([5, 6])
    v.curve_basis(as_matrix=True) # We get the basis in matrix form
    # array([[-18,   1,   9,   6,   1,   1,   0],
    #        [ -6,   0,   3,   2,   0,   0,   1]])
    ```
    Note that when setting a curve basis in this way, the function behaves
    exactly the same as [`set_divisor_basis`](#set_divisor_basis). For a more
    advanced example involving generic bases these two functions differ. An
    example can be found in the [experimental features](./experimental) section.
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
            "The experimental features must be enabled to " "use generic bases."
        )

    # grab GLSM information
    glsm_cm = self.glsm_charge_matrix(include_origin=True)
    glsm_rnk = np.linalg.matrix_rank(glsm_cm)

    if np.linalg.matrix_rank(b) != glsm_rnk:
        raise ValueError("Input matrix has incorrect rank.")
    if b.shape == (glsm_rnk, glsm_cm.shape[1]):
        new_b = b
    elif b.shape == (glsm_rnk, glsm_cm.shape[1] - 1):
        new_b = np.empty(glsm_cm.shape, dtype=t)
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
    dualize: bool = False,
    favorable: bool = None,
    lattice: str = None,
    limit: int = None,
) -> Generator["Polytope", None, None]:
    """
    **Description:**
    Reads polytopes from a file or a string. The polytopes can be specified
    with their vertices, as used in the Kreuzer-Skarke database, or from a
    weight system.

    :::note
    This function is not intended to be called by the end user. Instead, it is
    used by the [`read_polytopes`](#read_polytopes) and
    [`fetch_polytopes`](#fetch_polytopes) functions.
    :::

    **Arguments:**
    - `input`: Specifies the name of the file to read or the string containing
        the polytopes.
    - `input_type`: Specifies whether to read from a file or from the input
        string. Options are "file" or "str".
    - `format`: Specifies the format to read. The options are "ks", which is
        the format used in the KS database, and "ws", if the polytopes should
        be constructed from weight systems.
    - `backend`: A string that specifies the backend used for the
        [`Polytope`](./polytope) class.
    - `dualize`: Flag that indicates whether to dualize all the polytopes
        before yielding them.
    - `favorable`: Yield only polytopes that are favorable when set to True, or
        non-favorable when set to False. If not specified then it yields both
        favorable and non-favorable polytopes.
    - `lattice`: The lattice to use when checking favorability. This parameter
        is only required when `favorable` is set. Options are "M" and "N".
    - `limit`: Sets a maximum numbers of polytopes to yield.

    **Returns:**
    A generator of [`Polytope`](./polytope) objects.

    **Example:**
    Since this function should not be used directly, we show an example of it
    being used with the [`read_polytopes`](#read_polytopes) function. We take a
    string obtained from the KS database and read the polytope it specifies.
    ```python {8}
    from cytools import read_polytopes # Note - it cannot be imported from root
    poly_data = '''4 5  M:10 5 N:376 5 H:272,2 [540]
                    1    0    0    0   -9
                    0    1    0    0   -6
                    0    0    1    0   -1
                    0    0    0    1   -1
                '''
    read_polytopes(poly_data, input_type="str", as_list=True)
    # [A 4-dimensional reflexive lattice polytope in ZZ^4]
    ```
    """
    from cytools.polytope import Polytope

    # input checking
    if favorable is not None and lattice is None:
        raise ValueError('Lattice must be specified. Options are "M" and "N".')

    if input_type not in ["file", "str"]:
        raise ValueError('"input_type" must be either "file" or "str"')

    # read data
    n_yielded = 0

    if input_type == "file":
        in_file = open(input)
        l = in_file.readline()
    else:
        in_string = input.split("\n")
        l = in_string.pop(0)

    if format == "ws":
        # read the polytopes as weight systems
        while (limit is None) or (n_yielded < limit):
            # pass line to PALP
            p    = pypalp.Polytope(input)
            vert = p.vertices()

            # ensure reasonable shape
            if len(vert.shape) == 0:
                break
            if vert.shape[0] < vert.shape[1]:
                vert = vert.T

            # build the Polytope
            p = Polytope(vert, backend=backend)

            if (favorable is None) or (p.is_favorable(lattice=lattice) == favorable):
                n_yielded += 1
                yield (p.dual() if dualize else p)

            # get next line
            if input_type == "file":
                l = in_file.readline()

                for i in range(5):
                    if l != "":
                        break
                    l = in_file.readline()
                else:
                    in_file.close()
                    break
            else:
                if len(in_string) > 0:
                    l = in_string.pop(0)
                else:
                    break
    elif format != "ks":
        raise ValueError('Unsupported format. Options are "ks" and "ws".')

    # format is "ks"
    while limit is None or n_yielded < limit:
        if "M:" in l:
            h = l.split()
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
            p = Polytope(vert, backend=backend)
            if (favorable is None) or (p.is_favorable(lattice=lattice) == favorable):
                n_yielded += 1
                yield (p.dual() if dualize else p)

        # get next line
        if input_type == "file":
            l = in_file.readline()

            for i in range(5):
                if l != "":
                    break
                l = in_file.readline()
            else:
                in_file.close()
                break
        else:
            if len(in_string) > 0:
                l = in_string.pop(0)
            else:
                break


def read_polytopes(
    input: str,
    input_type: str = "file",
    format: str = "ks",
    backend: str = None,
    as_list: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    lattice: str = None,
    limit: int = None,
) -> 'Generator["Polytope", None, None] | \
                                                            list["Polytope"]':
    """
    **Description:**
    Reads polytopes from a file or a string. The polytopes can be specified
    with their vertices, as used in the Kreuzer-Skarke database, or from a
    weight system.

    **Arguments:**
    - `input`: Specifies the name of the file to read or the string containing
        the polytopes.
    - `input_type`: Specifies whether to read from a file or from the input
        string. Options are "file" or "str".
    - `format`: Specifies the format to read. The options are "ks", which is
        the format used in the KS database, and "ws", if the polytopes should
        be constructed from weight systems.
    - `backend`: A string that specifies the backend used for the
        [`Polytope`](./polytope) class.
    - `as_list`: Return the list of polytopes instead of a generator.
    - `dualize`: Flag that indicates whether to dualize all the polytopes
        before yielding them.
    - `favorable`: Yield or return only polytopes that are favorable when set
        to True, or non-favorable when set to False. If not specified then it
        yields both favorable and non-favorable polytopes.
    - `lattice`: The lattice to use when checking favorability. This parameter
        is only required when `favorable` is set. Options are "M" and "N".
    - `limit`: Sets a maximum numbers of polytopes to yield.

    **Returns:**
    A generator of [`Polytope`](./polytope) objects, or the full list when
    `as_list` is set to True.

    **Example:**
    We take a string obtained from the KS database and read the polytope it
    specifies.
    ```python {8}
    from cytools import read_polytopes # Note that it can directly be imported from the root
    poly_data = '''4 5  M:10 5 N:376 5 H:272,2 [540]
                    1    0    0    0   -9
                    0    1    0    0   -6
                    0    0    1    0   -1
                    0    0    0    1   -1
                '''
    read_polytopes(poly_data, input_type="str", as_list=True)
    # [A 4-dimensional reflexive lattice polytope in ZZ^4]
    ```
    """
    g = polytope_generator(
        input,
        input_type=input_type,
        format=format,
        backend=backend,
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
    timeout: int = 60,
    as_list: bool = True,
    backend: str = None,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0,
) -> 'Generator["Polytope", None, None] | list["Polytope"]':
    """
    **Description:**
    Fetches reflexive polytopes from the Kreuzer-Skarke database or from the
    Schöller-Skarke database. The data is fetched from the websites
    http://hep.itp.tuwien.ac.at/~kreuzer/CY/ and
    http://rgc.itp.tuwien.ac.at/fourfolds/ respectively.

    :::note
    The Kreuzer-Skarke database does not store favorability data. Thus, when
    setting favorable to True or False it fetches additional polytopes so that
    after filtering by favorability it can saturate the requested limit.
    However, it may happen that fewer polytopes than requested are returned
    even though more exist. To verify that no more polytopes with the requested
    conditions exist one can increase the limit significantly and check if more
    polytopes are returned.
    :::

    **Arguments:**
    - `h11`: The Hodge number $h^{1,1}$ of the Calabi-Yau hypersurface.
    - `h12`: The Hodge number $h^{1,2}$ of the Calabi-Yau hypersurface.
    - `h13`: The Hodge number $h^{1,3}$ of the Calabi-Yau hypersurface.
    - `h21`: The Hodge number $h^{2,1}$ of the Calabi-Yau hypersurface. This is
        equivalent to the h12 parameter.
    - `h22`: The Hodge number $h^{2,2}$ of the Calabi-Yau hypersurface.
    - `h31`: The Hodge number $h^{3,1}$ of the Calabi-Yau hypersurface. This is
        equivalent to the h13 parameter.
    - `chi`: The Euler characteristic of the Calabi-Yau hypersurface.
    - `lattice`: The lattice on which the polytope is defined. Options are "N"
        and "M". Has to be specified if the Hodge numbers or the Euler
        characteristic is specified.
    - `dim`: The dimension of the polytope. Only available options are 4 and 5.
    - `n_points`: The number of lattice points of the desired polytopes.
    - `n_vertices`: The number of vertices of the desired polytopes.
    - `n_dual_points`: The number of points of the dual polytopes of the
        desired polytopes.
    - `n_facets`: The number of facets of the desired polytopes.
    - `limit`: The maximum number of fetched polytopes.
    - `timeout`: The maximum number of seconds to wait for the server to return
        the data.
    - `as_list`: Return the list of polytopes instead of a generator.
    - `backend`: A string that specifies the backend used for the
        [`Polytope`](./polytope) class.
    - `dualize`: Flag that indicates whether to dualize all the polytopes
        before yielding them.
    - `favorable`: Yield or return only polytopes that are favorable when set
        to True, or non-favorable when set to False. If not specified then it
        yields both favorable and non-favorable polytopes.
    - `verbostiy`: The verbosity level.

    **Returns:**
    A generator of [`Polytope`](./polytope) objects, or the full list when
    `as_list` is set to True.

    **Example:**
    We fetch polytopes from the Kreuzer-Skarke and Schöller-Skarke databases
    with a few different parameters.
    ```python {2,5,8}
    from cytools import fetch_polytopes # Note that it can directly be imported from the root
    g = fetch_polytopes(h11=27, as_list=False) # Constructs a generator of polytopes
    next(g)
    # A 4-dimensional reflexive lattice polytope in ZZ^4
    l = fetch_polytopes(h11=27, limit=100) # Constructs a list of polytopes
    print(f"Fetched {len(l)} polytopes")
    # Fetched 100 polytopes
    g_5d = fetch_polytopes(h11=1000, as_list=False, dim=5, limit=100) # Generator of 5D polytopes
    next(g_5d)
    # A 5-dimensional reflexive lattice polytope in ZZ^5
    ```
    """
    # input checking
    # --------------
    if dim not in (4, 5):
        raise ValueError("Only polytopes of dimension 4 or 5 are available.")

    if lattice not in ("N", "M", None):
        raise ValueError("Options for lattice are 'N' and 'M'.")

    if favorable is not None:
        if lattice is None:
            raise ValueError("Must specify lattice when checking " "favorability.")

        fetch_limit = (5 if favorable else 10) * limit + 100
    else:
        fetch_limit = limit

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
            and (chi != 2 * (h11 - h21))
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
            raise ValueError("Lattice must be specified when h11 or h13 " "are given.")

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

    # verbosity
    if verbosity >= 1:
        print(f"Fetched from URL = '{r.url}'...")

    # return the generator based off of output of request
    return read_polytopes(
        r.text,
        input_type="str",
        format=("ks" if dim == 4 else "ws"),
        backend=backend,
        as_list=as_list,
        dualize=dualize,
        favorable=favorable,
        lattice=lattice,
        limit=limit,
    )


# point manipulations
# -------------------
def lll_reduce(pts_in: ArrayLike, transform: bool = False) -> "misc":
    """
    Apply lll-reduction to the input points (the rows).

    **Arguments:**
    - `pts`: A list of points.

    **Returns:**
    The reduced points (pts_red; as rows of a numpy array).
    If transform==True, also return the transformation matrix/inverse
        (A, Ainv) s.t. pts_red.T = A*pts_in.T. As numpy arrays.
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
    """
    **Description:**
    Finds new points that are affinely independent to the input list of points.

    This is useful when one wants to turn a polytope that is not
    full-dimensional into one that is, without affecting the structure of the
    triangulations.

    **Arguments:**
    - `pts`: A list of points.

    **Returns:**
    A list of affinely independent points with respect to the ones inputted.

    **Example:**
    We construct a list of points and then find a set of affinely independent
    points.
    ```python {2}
    pts = [[1,0,1],[0,0,1],[0,1,1]]
    find_new_affinely_independent_points(pts)
    array([[1, 0, 2]])
    ```
    """
    # input checking
    if len(pts) == 0:
        raise ValueError("List of points cannot be empty.")

    # cast to numpy array
    pts = np.asarray(pts)
    shape = pts.shape

    # translate, append unit vector
    translation = pts[0].copy()
    pts -= translation

    if shape[0] == 1:
        pts = np.append(pts_trans, [[1] + [0] * (shape[1] - 1)], axis=0)

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
    """
    Given an h11+5 dimensional height vector,
    returns an h11+5 dimensional vector that corresponds to point in the kahler cone.
    """
    basis = [i-1 for i in poly.glsm_basis(include_origin=True)]
    if prime_divisors is None:
        prime_divisors = np.array([rr for r,rr in enumerate(poly.triangulate(verbosity=0).get_cy().toric_effective_cone().rays()) if r not in basis], dtype=float)
    extra_divs = [i for i in range(poly.h11(lattice='N')+4) if i not in basis]
    origin_height = heights_in[0]
    kahler_parameters = np.array([i-origin_height for i in heights_in[1:]])
    for e,ee in enumerate(prime_divisors):
        prime_ind = extra_divs[e]
        prime_height = kahler_parameters[prime_ind]
        lin_rel = np.zeros(poly.h11(lattice='N')+4)
        lin_rel[basis] = ee
        lin_rel[prime_ind] = -1
        corr = prime_height*lin_rel
        kahler_parameters = np.array(kahler_parameters) + np.array(corr)
    return np.concatenate(([0],kahler_parameters))

def heights_to_kahler(poly, heights_in, prime_divisors=None):
    """
    Given an h11+5 dimensional height vector,
    returns an h11 dimensional vector that corresponds to point in the kahler cone.
    """
    basis = poly.glsm_basis()
    return project_heights_to_kahler(poly, heights_in, prime_divisors)[basis]

def kahler_to_heights(poly, kahler_in):
    """
    Given an h11 dimensional vector hat corresponds to point in the kahler cone,
    returns an h11+5 dimensional height vector.
    """
    basis = [i for i in poly.glsm_basis(include_origin=True)]
    kahler_gen = iter(kahler_in)
    return np.array([next(kahler_gen) if i in basis else 0 for i in range(poly.h11(lattice='N')+5)])
