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

"""This module contains various common functions that are used in CYTools."""

# Standard imports
from itertools import permutations
from fractions import Fraction
from functools import reduce
from ast import literal_eval
import subprocess
import requests
# Third party imports
from flint import fmpz_mat, fmpq, fmpz
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
# CYTools imports
from cytools import config



def gcd_float(a, b, tol=1e-5):
    """
    **Description:**
    Compute the greatest common (floating-point) divisor of a and b.

    **Arguments:**
    - `a` *(float)*: The first number.
    - `b` *(float)*: The second number.
    - `tol` *(float, optional, default=1e-5)*: The tolerance for rounding.

    **Returns:**
    *(float)* The gcd of a and b.

    **Example:**
    We compute the gcd of two floats. This function is useful since the
    standard gcd functions raise an error for non-integer values.
    ```python {2}
    from cytools.utils import gcd_float
    gcd_float(0.2, 0.5) # Should be 0.1, but there are small rounding errors
    # 0.09999999999999998
    ```
    """
    if abs(b) < tol:
        return abs(a)
    return gcd_float(b,a%b,tol)


def gcd_list(arr):
    """
    **Description:**
    Compute the greatest common divisor of the elements in an array.

    **Arguments:**
    - `arr` *(array_like)*: A list of floating-point numbers.

    **Returns:**
    *(float)* The gcd of all the elements in the input list.

    **Example:**
    We compute the gcd of a list of floats This function is useful since the
    standard gcd functions raise an error for non-integer values.
    ```python {2}
    from cytools.utils import gcd_list
    gcd_list([0.2, 0.5, 1.4, 6.05, 3.45]) # Should be 0.05, but there are small rounding errors
    # 0.04999999999999882
    ```
    """
    return reduce(gcd_float,arr)


def to_sparse(rule_arr_in, sparse_type="dok"):
    """
    **Description:**
    Converts an matrix of the form [[a,b,M_ab],...] or a dictionary of the
    form [(a,b):M_ab, ...] to a dok_matrix or to a csr_matrix.

    **Arguments:**
    - `rule_arr_in` *(dict or array_like)*: A list of the form
      [[a,b,M_ab],...] or a dictionary of the form [(a,b):M_ab,...].
    - `sparse_type` *(str, optional, default="dok")*: The type of sparse
      matrix to return. The options are "dok" and "csr".

    **Returns:**
    *(scipy.sparse.dok_matrix or scipy.sparse.csr_matrix)* The sparse matrix in
    dok_matrix or csr_matrix format.

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
    if isinstance(rule_arr_in, dict):
        rule_arr_in = [list(ii)+[rule_arr_in[ii]] for ii in rule_arr_in]
    rule_arr = np.array(rule_arr_in)
    if sparse_type not in ("dok", "csr"):
        raise ValueError("sparse_type must be either \"dok\" or \"csr\".")
    dim_0 = max(rule_arr[:,0]+1)
    dim_1 = max(rule_arr[:,1]+1)
    sp_mat = dok_matrix((dim_0,dim_1))
    for r in rule_arr:
        sp_mat[r[0],r[1]] = r[2]
    return (sp_mat if sparse_type == "dok" else csr_matrix(sp_mat))


def solve_linear_system(M, C, backend="all", check=True,
                        backend_error_tol=1e-4, verbose=0):
    """
    **Description:**
    Solves the sparse linear system M*x+C=0.

    **Arguments:**
    - `M` *(scipy.sparse.csr_matrix)*: A scipy csr_matrix.
    - `C` *(array_like)*: A vector of floats.
    - `backend` *(str, optional, default="all")*: The sparse linear solver
      to use. Options are "all", "sksparse" and "scipy". When set to "all" it
      tries all available backends.
    - `check` *(bool, optional, default=True)*: Whether to explicitly
      check the solution to the linear system.
    - `backend_error_tol` *(float, optional, default=1e-4)*: Error
      tolerance for the solution of the linear system.
    - `verbose` *(int, optional, default=0)*: The verbosity level.
      - verbose = 0: Do not print anything.
      - verbose = 1: Print warnings when backends fail.

    **Returns:**
    *(numpy.ndarray)* Floating-point solution to the linear system.

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
    backends = ["all", "sksparse", "scipy"]
    if backend not in backends:
        raise ValueError("Invalid linear system backend. "
                         f"The options are: {backends}.")
    system_solved = False
    if backend == "all":
        for s in backends[1:]:
            solution = solve_linear_system(M, C, backend=s, check=check,
                                           backend_error_tol=backend_error_tol,
                                           verbose=verbose)
            if solution is not None:
                return solution
    elif backend == "sksparse":
        try:
            from sksparse.cholmod import cholesky_AAt
            factor = cholesky_AAt(M.transpose())
            CC = -M.transpose()*C
            solution = factor(CC)
            system_solved = True
        except:
            if verbose >= 1:
                print("Linear backend error: sksparse failed.")
            system_solved = False
    elif backend == "scipy":
        try:
            from scipy.sparse.linalg import spsolve
            MM = M.transpose()*M
            CC = -M.transpose()*C
            solution = spsolve(MM, CC).tolist()
            system_solved = True
        except:
            if verbose >= 1:
                print("Linear backend error: scipy failed.")
            system_solved = False
    if system_solved and check:
        res = M.dot(solution) + C
        max_error = max(abs(s) for s in res.flat)
        if max_error > backend_error_tol:
            system_solved = False
            if verbose >= 1:
                print("Linear backend error: Large numerical error.")
    if system_solved:
        return solution
    else:
        return None


def filter_tensor_indices(tensor, indices):
    """
    **Description:**
    Selects a specific subset of indices from a tensor. The tensor is
    reindexed so that indices are in the range 0..len(indices) with the
    ordering specified by the input indices. This function can be used to
    convert the tensor of intersection numbers to a given basis.

    **Arguments:**
    - `tensor` *(dict)*: The input symmetric sparse tensor of the form
      of a dictionary {(a,b,...,c):M_ab...c, ...}.
    - `indices` *(array_like)*: The list of indices that will be preserved.

    **Returns:**
    *(dict)* A dictionary describing a tensor in the same format as the input,
    but only with the desired indices.

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
    tensor_filtered = {k:tensor[k] for k in tensor if all(c in indices for c in k)}
    indices_dict = {vv:v for v,vv in enumerate(indices)}
    tensor_reindexed = {tuple(sorted(indices_dict[c] for c in k)):tensor_filtered[k] for k in tensor_filtered}
    return tensor_reindexed


def symmetric_sparse_to_dense(tensor, basis=None):
    """
    **Description:**
    Converts a symmetric sparse tensor of the form {(a,b,...,c):M_ab...c, ...}
    to a dense tensor. Optionally, it then applies a basis transformation.

    **Arguments:**
    - `tensor` *(dict)*: A sparse tensor of the form
      {(a,b,...,c):M_ab...c, ...}.
    - `basis` *(array_like, optional)*: A matrix where the rows are the basis
      elements.

    **Returns:**
    *(numpy.ndarray)* A dense tensor.

    **Example:**
    We construct a simple tensor and then convert it to a dense array. We
    consider the same example as for the
    [`filter_tensor_indices`](#filter_tensor_indices) function, but now we
    have to specify the basis in matrix form.
    ```python {4}
    from cytools.utils import symmetric_sparse_to_dense_in_basis
    tensor = {(0,1):0, (1,1):1, (1,2):2, (1,3):3, (2,3):4}
    basis = [[0,1,0,0],[0,0,0,1]]
    symmetric_sparse_to_dense(tensor, basis)
    # array([[1, 3],
    #        [3, 0]])
    ```
    """
    l = (np.array(basis).shape[1] if basis is not None else
            max(set.union(*[set(ii) for ii in tensor.keys()]))+1)
    dense_tensor = np.zeros((l,)*(len(list(tensor.items())[0][0])),
                            dtype=type(list(tensor.items())[0][1]))
    for ii in tensor:
        for c in permutations(ii):
            dense_tensor[c] = tensor[ii]
    dense_result = np.array(dense_tensor)
    if basis is not None:
        for i in list(range(len(list(tensor.items())[0][0])))[::-1]:
            dense_result = np.tensordot(dense_result, basis, axes=[[i],[1]])
    return dense_result


def symmetric_dense_to_sparse(tensor, basis=None):
    """
    **Description:**
    Converts a dense symmetric tensor to a sparse tensor of the form
    {(a,b,...,c):M_ab...c,...}. Optionally, it applies a basis transformation.

    **Arguments:**
    - `tensor` *(array_like)*: A dense symmetric tensor.
    - `basis` *(array_like, optional)*: A matrix where the rows are the basis
      elements.

    **Returns:**
    *(dict)* A sparse tensor of the form {(a,b,...,c):M_ab...c,...}.

    **Example:**
    We construct a simple tensor and then convert it to a dense array. We
    consider the same example as for the
    [`filter_tensor_indices`](#filter_tensor_indices) function, but now we
    have to specify the basis in matrix form.
    ```python {3}
    from cytools.utils import symmetric_dense_to_sparse
    tensor = [[1,2],[2,3]]
    symmetric_dense_to_sparse(tensor)
    # {(0, 0): 1, (0, 1): 2, (1, 1): 3}
    ```
    """
    dense_tensor = np.array(tensor)
    if basis is not None:
        for i in list(range(len(list(tensor.items())[0][0])))[::-1]:
            dense_result = np.tensordot(dense_result, basis, axes=[[i],[1]])
    sparse_tensor = dict()
    s = set(dense_tensor.shape)
    d = len(dense_tensor.shape)
    if len(s) != 1:
        raise ValueError("All dimensions must have the same length")
    s = list(s)[0]
    ind = [0]*d
    inc = d-1
    while True:
        if dense_tensor[tuple(ind)] != 0:
            sparse_tensor[tuple(ind)] = dense_tensor[tuple(ind)]
        inc = d-1
        if d == 1:
            break
        break_loop = False
        while True:
            if ind[inc] == s-1:
                ind[inc] = 0
                inc -= 1
                if inc == -1:
                    break_loop = True
                    break
            else:
                ind[inc] += 1
                for inc2 in range(inc+1,d):
                    ind[inc2] = ind[inc]
                break
        if break_loop:
            break
    return sparse_tensor


def float_to_fmpq(c):
    """
    **Description:**
    Converts a float to an fmpq.

    **Arguments:**
    - `c` *(float)*: The input number.

    **Returns:**
    *(flint.fmpq)* The rational number that most reasonably approximates the
    input.

    **Example:**
    We convert a few floats to rational numbers.
    ```python {2}
    from cytools.utils import float_to_fmpq
    float_to_fmpq(0.1), float_to_fmpq(0.333333333333), float_to_fmpq(2.45)
    # (1/10, 1/3, 49/20)
    ```
    """
    f = Fraction(c).limit_denominator()
    return fmpq(f.numerator, f.denominator)


def fmpq_to_float(c):
    """
    **Description:**
    Converts an fmpq to a float.

    **Arguments:**
    - `c` *(flint.fmpq)*: The input rational number.

    **Returns:**
    *(float)* The number as a float.

    **Example:**
    We convert a few rational numbers to floats.
    ```python {3}
    from cytools.utils import fmpq_to_float
    from flint import fmpq
    fmpq_to_float(fmpq(1,2)), fmpq_to_float(fmpq(1,3)), fmpq_to_float(fmpq(49,20))
    # (0.5, 0.3333333333333333, 2.45)
    ```
    """
    return int(c.p)/int(c.q)


def array_int_to_fmpz(arr):
    """
    **Description:**
    Converts a numpy array with 64-bit integer entries to fmpz entries.

    **Arguments:**
    - `arr` *(array_like)*: A numpy array with 64-bit integer entries.

    **Returns:**
    *(numpy.ndarray)* A numpy array with fmpz entries.

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
    in_arr = np.array(arr, dtype=int)
    out_arr = np.empty(in_arr.shape, dtype=object)
    for i in range(len(in_arr.flat)):
        out_arr.flat[i] = fmpz(int(in_arr.flat[i]))
    return out_arr


def array_float_to_fmpq(arr):
    """
    **Description:**
    Converts a numpy array with floating-point entries to fmpq entries.

    **Arguments:**
    - `arr` *(array_like)*: A numpy array with floating-point entries.

    **Returns:**
    *(numpy.ndarray)* A numpy array with fmpq entries.

    **Example:**
    We convert an float array to an fmpz array.
    ```python {3}
    from cytools.utils import array_float_to_fmpq
    arr = [[1.1,0,0],[0,0.5,0],[0,0,0.3333333333]]
    array_float_to_fmpq(arr)
    # array([[11/10, 0, 0],
    #        [0, 1/2, 0],
    #        [0, 0, 1/3]], dtype=object)
    ```
    """
    in_arr = np.array(arr, dtype=float)
    out_arr = np.empty(in_arr.shape, dtype=object)
    for i in range(len(in_arr.flat)):
        out_arr.flat[i] = float_to_fmpq(in_arr.flat[i])
    return out_arr


def array_fmpz_to_int(arr):
    """
    **Description:**
    Converts a numpy array with fmpz entries to 64-bit integer entries.

    **Arguments:**
    - `arr` *(array_like)*: A numpy array with fmpz entries.

    **Returns:**
    *(numpy.ndarray)* A numpy array with 64-bit integer entries.

    **Example:**
    We convert an fmpz array to an int array.
    ```python {4}
    from cytools.utils import array_fmpz_to_int
    from flint import fmpz
    arr = [fmpz(1), fmpz(2), fmpz(3)]
    array_fmpz_to_int(arr)
    # array([1, 2, 3])
    ```
    """
    return np.array(arr, dtype=int)


def array_fmpq_to_float(arr):
    """
    **Description:**
    Converts a numpy array with fmpq entries to floating-point entries.

    **Arguments:**
    - `arr` *(array_like)*: A numpy array with fmpq entries.

    **Returns:**
    *(numpy.ndarray)* A numpy array with floating-point entries.

    **Example:**
    We convert an fmpq array to a float array.
    ```python {4}
    from cytools.utils import array_fmpq_to_float
    from flint import fmpq
    arr = [fmpq(1,2), fmpq(2,5), fmpq(1,3)]
    array_fmpq_to_float(arr)
    # array([0.5       , 0.4       , 0.33333333])
    ```
    """
    in_arr = np.array(arr, dtype=object)
    out_arr = np.empty(in_arr.shape, dtype=float)
    for i in range(len(in_arr.flat)):
        out_arr.flat[i] = fmpq_to_float(in_arr.flat[i])
    return out_arr

def set_divisor_basis(tv_or_cy, basis, include_origin=True):
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
    [`set_divisor_basis`](./calabiyau#set_divisor_basis)
    function of the [`CalabiYau`](./calabiyau) class.
    :::

    :::note
    Only integral bases are supported by CYTools, meaning that all prime toric
    divisors must be able to be written as an integral linear combination of
    the basis divisors.
    :::

    **Arguments:**
    - `tv_or_cy` *(ToricVariety or CalabiYau)*: The toric variety or
      Calabi-Yau whose basis will be set.
    - `basis` *(array_like)*: Vector or matrix specifying a basis. When
      a vector is used, the entries will be taken as the indices of points
      of the polytope or prime divisors of the toric variety. When a
      matrix is used, the rows are taken as linear combinations of the
      aforementioned divisors.
    - `include_origin` *(bool, optional, default=True)*: Whether to
      interpret the indexing specified by the input vector as including the
      origin.

    **Returns:**
    Nothing.

    **Example:**
    This function is not intended to be directly used, but it is used in
    the following example. We consider a simple toric variety with two
    independent divisors. We first find the default basis it picks and then we
    set a basis of our choice.
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
    self = tv_or_cy # More conveninent to work with
    b = np.array(basis, dtype=int) # It must be an array of integers or they will be truncated
    glsm_cm = self.glsm_charge_matrix(include_origin=True)
    glsm_rnk = np.linalg.matrix_rank(glsm_cm)
    # Check if the input is a vector
    if len(b.shape) == 1:
        b = np.array(sorted(basis))
        if not include_origin:
            b += 1
        # Check if it is a valid basis
        if min(b) < 0 or max(b) >= glsm_cm.shape[1]:
            raise ValueError("Indices are not in appropriate range.")
        if (glsm_rnk != np.linalg.matrix_rank(glsm_cm[:,b])
                or glsm_rnk != len(b)):
            raise ValueError("Input divisors do not form a basis.")
        if abs(int(round(np.linalg.det(glsm_cm[:,b])))) != 1:
            raise ValueError("Only integer bases are supported.")
        # Save divisor basis
        self._divisor_basis = b
        self._divisor_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._divisor_basis_mat[:,b] = np.eye(glsm_rnk, dtype=int)
        # Construct dual basis of curves
        self._curve_basis = b
        nobasis = np.array([i for i in range(glsm_cm.shape[1]) if i not in b])
        linrels = self.glsm_linear_relations()
        linrels_tmp = np.empty(linrels.shape, dtype=int)
        linrels_tmp[:,:len(nobasis)] = linrels[:,nobasis]
        linrels_tmp[:,len(nobasis):] = linrels[:,b]
        linrels_tmp = fmpz_mat(linrels_tmp.tolist()).hnf()
        linrels_tmp = np.array(linrels_tmp.tolist(), dtype=int)
        linrels_new = np.empty(linrels.shape, dtype=int)
        linrels_new[:,nobasis] = linrels_tmp[:,:len(nobasis)]
        linrels_new[:,b] = linrels_tmp[:,len(nobasis):]
        self._curve_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._curve_basis_mat[:,b] = np.eye(len(b),dtype=int)
        sublat_ind =  int(round(np.linalg.det(np.array(fmpz_mat(linrels.tolist()).snf().tolist(), dtype=int)[:,:linrels.shape[0]])))
        for nb in nobasis[::-1]:
            tup = [(k,kk) for k,kk in enumerate(linrels_new[:,nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i,ii = tup[-1]
            self._curve_basis_mat[:,nb] = -self._curve_basis_mat.dot(linrels_new[i])//ii
    # Else if input is a matrix
    elif len(b.shape) == 2:
        if not config._exp_features_enabled:
            raise Exception("The experimental features must be enabled to "
                            "use generic bases.")
        # We start by checking if the input matrix looks right
        if np.linalg.matrix_rank(b) != glsm_rnk:
            raise ValueError("Input matrix has incorrect rank.")
        if b.shape == (glsm_rnk, glsm_cm.shape[1]):
            new_b = b
        elif b.shape == (glsm_rnk, glsm_cm.shape[1]-1):
            new_b = np.empty(glsm_cm.shape, dtype=int)
            new_b[:,1:] = b
            new_b[:,0] = 0
        else:
            raise ValueError("Input matrix has incorrect shape.")
        new_glsm_cm = new_b.dot(glsm_cm.T).T
        if np.linalg.matrix_rank(new_glsm_cm) != glsm_rnk:
            raise ValueError("Input divisors do not form a basis.")
        if abs(int(round(np.linalg.det(np.array(fmpz_mat(new_glsm_cm.tolist()).snf().tolist(),dtype=int)[:glsm_rnk,:glsm_rnk])))) != 1:
            raise ValueError("Input divisors do not form an integral basis.")
        self._divisor_basis = np.array(new_b)
        # Now we store a more convenient form of the matrix where we use the
        # linear relations to express them in terms of the default prime toric
        # divisors
        standard_basis = self.polytope().glsm_basis(
                                integral=True,
                                include_origin=True,
                                points=self.prime_toric_divisors())
        linrels = self.polytope().glsm_linear_relations(
                                include_origin=True,
                                points=self.prime_toric_divisors())
        self._divisor_basis_mat = np.array(new_b)
        nobasis = np.array([i for i in range(glsm_cm.shape[1]) if i not in standard_basis])
        sublat_ind =  int(round(np.linalg.det(np.array(fmpz_mat(linrels.tolist()).snf().tolist(), dtype=int)[:,:linrels.shape[0]])))
        for nb in nobasis[::-1]:
            tup = [(k,kk) for k,kk in enumerate(linrels[:,nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i,ii = tup[-1]
            for j in range(self._divisor_basis_mat.shape[0]):
                self._divisor_basis_mat[j] -= self._divisor_basis_mat[j,nb]*linrels[i]
        # Finally, we invert the matrix and construct the dual curve basis
        if abs(int(round(np.linalg.det(self._divisor_basis_mat[:,standard_basis])))) != 1:
            raise ValueError("Input divisors do not form an integral basis.")
        inv_mat = fmpz_mat(self._divisor_basis_mat[:,standard_basis].tolist()).inv(integer=True)
        inv_mat = np.array(inv_mat.tolist(), dtype=int)
        # flint sometimes returns the negative inverse
        if inv_mat.dot(self._divisor_basis_mat[:,standard_basis])[0,0] == -1:
            inv_mat *= -1;
        self._curve_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
        self._curve_basis_mat[:,standard_basis] = np.array(inv_mat).T
        for nb in nobasis[::-1]:
            tup = [(k,kk) for k,kk in enumerate(linrels[:,nb]) if kk]
            if sublat_ind % tup[-1][1] != 0:
                raise RuntimeError("Problem with linear relations")
            i,ii = tup[-1]
            self._curve_basis_mat[:,nb] = -self._curve_basis_mat.dot(linrels[i])//ii
        self._curve_basis = np.array(self._curve_basis_mat)
    else:
        raise ValueError("Input must be either a vector or a matrix.")
    # Clear the cache of all in-basis computations
    self.clear_cache(recursive=False, only_in_basis=True)


def set_curve_basis(tv_or_cy, basis, include_origin=True):
    """
    **Description:**
    Specifies a basis of curves of the toric variety, which in turn
    specifies a dual basis of divisors. This can be done with a vector
    specifying the indices of the dual prime toric divisors or as a matrix
    with the rows being the basis curves, and the entries are the intersection
    numbers with the prime toric divisors. Note that when using a vector it is
    equivalent to using the same vector in the
    [`set_divisor_basis`](#set_divisor_basis) function.

    :::important
    This function should generally not be called directly by the user. Instead,
    it is called by the [`set_curve_basis`](./toricvariety#set_curve_basis)
    function of the [`ToricVariety`](./toricvariety) class, or the
    [`set_curve_basis`](./calabiyau#set_curve_basis)
    function of the [`CalabiYau`](./calabiyau) class.
    :::

    :::note
    Only integral bases are supported by CYTools, meaning that all toric
    curves must be able to be written as an integral linear combination of
    the basis curves.
    :::

    **Arguments:**
    - `tv_or_cy` *(ToricVariety or CalabiYau)*: The toric variety or
      Calabi-Yau whose basis will be set.
    - `basis` *(array_like)*: Vector or matrix specifying a basis. When
      a vector is used, the entries will be taken as indices of the
      standard basis of the dual to the lattice of prime toric divisors.
      When a matrix is used, the rows are taken as linear combinations of
      the aforementioned elements.
    - `include_origin` *(bool, optional, default=True)*: Whether to
      interpret the indexing specified by the input vector as including the
      origin.

    **Returns:**
    Nothing.

    **Example:**
    This function is not intended to be directly used, but it is used in
    the following example. We consider a simple toric variety with two
    independent curves. We first find the default basis of curves it picks and
    then set a basis of our choice.
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
    exactly the same as [`set_divisor_basis`](#set_divisor_basis). For
    a more advanced example involving generic bases these two functions
    differ. An example can be found in the
    [experimental features](./experimental) section.
    """
    self = tv_or_cy # More conveninent to work with
    b = np.array(basis, dtype=int)
    # Check if the input is a vector
    if len(b.shape) == 1:
        set_divisor_basis(self, b, include_origin=include_origin)
        return
    if len(b.shape) != 2:
        raise ValueError("Input must be either a vector or a matrix.")
    # Else input is a matrix
    if not config._exp_features_enabled:
        raise Exception("The experimental features must be enabled to "
                        "use generic bases.")
    glsm_cm = self.glsm_charge_matrix(include_origin=True)
    glsm_rnk = np.linalg.matrix_rank(glsm_cm)
    if np.linalg.matrix_rank(b) != glsm_rnk:
        raise ValueError("Input matrix has incorrect rank.")
    if b.shape == (glsm_rnk, glsm_cm.shape[1]):
        new_b = b
    elif b.shape == (glsm_rnk, glsm_cm.shape[1]-1):
        new_b = np.empty(glsm_cm.shape, dtype=t)
        new_b[:,1:] = b
        new_b[:,0] = -np.sum(b, axis=1)
    else:
        raise ValueError("Input matrix has incorrect shape.")
    pts = [tuple(pt)+(1,) for pt in self.polytope().points()[[0]+list(self.prime_toric_divisors())]]
    if any(new_b.dot(pts).flat):
        raise ValueError("Input curves do not form a valid basis.")
    if abs(int(round(np.linalg.det(np.array(fmpz_mat(new_b.tolist()).snf().tolist(),dtype=int)[:glsm_rnk,:glsm_rnk])))) != 1:
        raise ValueError("Input divisors do not form an integral basis.")
    standard_basis = self.polytope().glsm_basis(
                            integral=True,
                            include_origin=True,
                            points=self.prime_toric_divisors())
    if abs(int(round(np.linalg.det(new_b[:,standard_basis])))) != 1:
        raise ValueError("Input divisors do not form an integral basis.")
    inv_mat = fmpz_mat(new_b[:,standard_basis].tolist()).inv(integer=True)
    inv_mat = np.array(inv_mat.tolist(), dtype=int)
    # flint sometimes returns the negative inverse
    if inv_mat.dot(new_b[:,standard_basis])[0,0] == -1:
        inv_mat *= -1;
    self._divisor_basis_mat = np.zeros(glsm_cm.shape, dtype=int)
    self._divisor_basis_mat[:,standard_basis] = np.array(inv_mat).T
    self._divisor_basis = np.array(self._divisor_basis_mat)
    self._curve_basis = np.array(new_b)
    self._curve_basis_mat = np.array(new_b)
    # Clear the cache of all in-basis computations
    self.clear_cache(recursive=False, only_in_basis=True)


def polytope_generator(input, input_type="file", format="ks", backend=None,
                       dualize=False, favorable=None, lattice=None,
                       limit=None):
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
    - `input` *(str)*: Specifies the name of the file to read or the
      string containing the polytopes.
    - `input_type` *(str, optional, default="file")*: Specifies whether to
      read from a file or from the input string. Options are "file" or
      "str".
    - `format` *(str, optional, default="ks")*: Specifies the format to
      read. The options are "ks", which is the format used in the KS database,
      and "ws", if the polytopes should be constructed from weight systems.
    - `backend` *(str, optional)*: A string that specifies the backend
      used for the [`Polytope`](./polytope) class.
    - `dualize` *(bool, optional, default=False)*: Flag that indicates
      whether to dualize all the polytopes before yielding them.
    - `favorable` *(bool, optional)*: Yield only polytopes that are
      favorable when set to True, or non-favorable when set to False. If not
      specified then it yields both favorable and non-favorable polytopes.
    - `lattice` *(str, optional)*: The lattice to use when checking
      favorability. This parameter is only required when `favorable` is
      specified. Options are "M" and "N".
    - `limit` *(int, optional)*: Sets a maximum numbers of polytopes to yield.

    **Returns:**
    *(generator)* A generator of [`Polytope`](./polytope) objects.

    **Example:**
    Since this function should not be used directly, we show an example of it
    being used with the [`read_polytopes`](#read_polytopes) function. We
    take a string obtained from the KS database and read the polytope it
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
    from cytools import Polytope
    if favorable is not None and lattice is None:
        raise ValueError("Lattice must be specified. Options are \"M\" and \"N\".")
    if input_type not in ["file", "str"]:
        raise ValueError("\"input_type\" must be either \"file\" or \"str\"")
    if input_type == "file":
        in_file = open(input, "r")
        l = in_file.readline()
    else:
        in_string = input.split("\n")
        l = in_string[0]
        in_string.pop(0)
    n_yielded = 0
    if format == "ws":
        while limit is None or n_yielded < limit:
            palp = subprocess.Popen((config.palp_path + "/poly.x", "-v"),
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
            palp_res, palp_err = palp.communicate(input=l+"\n")
            palp_res = palp_res.split("\n")
            vert = []
            i = 0
            while i < len(palp_res):
                if "Vertices" in palp_res[i]:
                    for j in range(literal_eval(palp_res[i].split()[0])):
                        i += 1
                        vert.append([int(c) for c in palp_res[i].split()])
                i += 1
            vert = np.array(vert)
            if len(vert.shape) == 0:
                break
            if vert.shape[0] < vert.shape[1]:
                vert = vert.T
            p = Polytope(vert, backend=backend)
            if favorable is None or p.is_favorable(lattice=lattice) == favorable:
                n_yielded += 1
                yield (p.dual() if dualize else p)
            if input_type == "file":
                l = in_file.readline()
                reached_end = True
                for i in range(5):
                    if l != "":
                        reached_end = False
                        break
                    l = in_file.readline()
                if reached_end:
                    in_file.close()
                    break
            else:
                if len(in_string) > 0:
                    l = in_string[0]
                    in_string.pop(0)
                else:
                    break
    elif format != "ks":
        raise ValueError("Unsupported format. Options are \"ks\" and \"ws\".")
    while limit is None or n_yielded < limit:
        if "M:" in l:
            h = l.split()
            n, m = int(h[0]), int(h[1])
            vert = []
            for i in range(n):
                if input_type == "file":
                    vert.append([int(c) for c in in_file.readline().split()])
                else:
                    vert.append([int(c) for c in in_string[0].split()])
                    in_string.pop(0)
            vert = np.array(vert)
            if vert.shape != (n, m):
                raise ValueError("Dimensions of array do not match")
            if m > n:
                vert = vert.T
            p = Polytope(vert, backend=backend)
            if favorable is None or p.is_favorable(lattice=lattice) == favorable:
                n_yielded += 1
                yield (p.dual() if dualize else p)
        if input_type == "file":
            l = in_file.readline()
            reached_end = True
            for i in range(5):
                if l != "":
                    reached_end = False
                    break
                l = in_file.readline()
            if reached_end:
                in_file.close()
                break
        else:
            if len(in_string) > 0:
                l = in_string[0]
                in_string.pop(0)
            else:
                break


def read_polytopes(input, input_type="file", format="ks", backend=None,
                   as_list=False, dualize=False, favorable=None, lattice=None,
                   limit=None):
    """
    **Description:**
    Reads polytopes from a file or a string. The polytopes can be specified
    with their vertices, as used in the Kreuzer-Skarke database, or from a
    weight system.

    **Arguments:**
    - `input` *(str)*: Specifies the name of the file to read or the
      string containing the polytopes.
    - `input_type` *(str, optional, default="file")*: Specifies whether to
      read from a file or from the input string. Options are "file" or
      "str".
    - `format` *(str, optional, default="ks")*: Specifies the format to
      read. The options are "ks", which is the format used in the KS database,
      and "ws", if the polytopes should be constructed from weight systems.
    - `backend` *(str, optional)*: A string that specifies the backend
      used for the [`Polytope`](./polytope) class.
    - `as_list` *(bool, optional, default=False)*: Return the list of
      polytopes instead of a generator.
    - `dualize` *(bool, optional, default=False)*: Flag that indicates
      whether to dualize all the polytopes before yielding them.
    - `favorable` *(bool, optional)*: Yield or return only polytopes that are
      favorable when set to True, or non-favorable when set to False. If not
      specified then it yields both favorable and non-favorable polytopes.
    - `lattice` *(str, optional)*: The lattice to use when checking
      favorability. This parameter is only required when `favorable` is
      specified. Options are "M" and "N".
    - `limit` *(int, optional)*: Sets a maximum numbers of polytopes to yield.

    **Returns:**
    *(generator or list)* A generator of [`Polytope`](./polytope) objects,
    or the full list when `as_list` is set to True.

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
    g = polytope_generator(input, input_type=input_type, format=format,
                           backend=backend, dualize=dualize,
                           favorable=favorable, lattice=lattice, limit=limit)
    if as_list:
        return list(g)
    return g


def fetch_polytopes(h11=None, h12=None, h13=None, h21=None, h22=None, h31=None,
                    chi=None, lattice=None, dim=4, n_points=None,
                    n_vertices=None, n_dual_points=None, n_facets=None,
                    limit=1000, timeout=60, as_list=False, backend=None,
                    dualize=False, favorable=None):
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
    conditions exist one can increase the limit significantly and check if
    more polytopes are returned.
    :::

    **Arguments:**
    - `h11` *(int, optional)*: Specifies the Hodge number $h^{1,1}$ of
      the Calabi-Yau hypersurface.
    - `h12` *(int, optional)*: Specifies the Hodge number $h^{1,2}$ of
      the Calabi-Yau hypersurface.
    - `h13` *(int, optional)*: Specifies the Hodge number $h^{1,3}$ of
      the Calabi-Yau hypersurface.
    - `h21` *(int, optional)*: Specifies the Hodge number $h^{2,1}$ of
      the Calabi-Yau hypersurface. This is equivalent to the h12 parameter.
    - `h22` *(int, optional)*: Specifies the Hodge number $h^{2,2}$ of
      the Calabi-Yau hypersurface.
    - `h31` *(int, optional)*: Specifies the Hodge number $h^{3,1}$ of
      the Calabi-Yau hypersurface. This is equivalent to the h13 parameter.
    - `chi` *(int, optional)*: Specifies the Euler characteristic of the
      Calabi-Yau hypersurface.
    - `lattice` *(str, optional)*: Specifies the lattice on which the
      polytope is defined. Options are "N" and "M". Has to be specified if the
      Hodge numbers or the Euler characteristic is specified.
    - `dim` *(int, optional, default=4)*: The dimension of the polytope.
      The only available options are 4 and 5.
    - `n_points` *(int, optional)*: Specifies the number of lattice
      points of the desired polytopes.
    - `n_vertices` *(int, optional)*: Specifies the number of vertices of
      the desired polytopes.
    - `n_dual_points` *(int, optional)*: Specifies the number of points
      of the dual polytopes of the desired polytopes.
    - `n_facets` *(int, optional)*: Specifies the number of facets of the
      desired polytopes.
    - `limit` *(int, optional, default=1000)*: Specifies the maximum
      number of fetched polytopes.
    - `timeout` *(int, optional, default=60)*: Specifies the maximum
      number of seconds to wait for the server to return the data.
    - `as_list` *(bool, optional, default=False)*: Return the list of
      polytopes instead of a generator.
    - `backend` *(str, optional)*: A string that specifies the backend
      used for the [`Polytope`](./polytope) class.
    - `dualize` *(bool, optional, default=False)*: Flag that indicates
      whether to dualize all the polytopes before yielding them.
    - `favorable` *(bool, optional)*: Yield or return only polytopes that are
      favorable when set to True, or non-favorable when set to False. If not
      specified then it yields both favorable and non-favorable polytopes.

    **Returns:**
    *(generator or list)* A generator of [`Polytope`](./polytope) objects,
    or the full list when `as_list` is set to True.

    **Example:**
    We fetch polytopes from the Kreuzer-Skarke and Schöller-Skarke databases
    with a few different parameters.
    ```python {2,5,8}
    from cytools import fetch_polytopes # Note that it can directly be imported from the root
    g = fetch_polytopes(h11=27, lattice="N") # Constructs a generator of polytopes
    next(g)
    # A 4-dimensional reflexive lattice polytope in ZZ^4
    l = fetch_polytopes(h11=27, lattice="N", as_list=True, limit=100) # Constructs a generator of polytopes
    print(f"Fetched {len(l)} polytopes")
    # Fetched 100 polytopes
    g_5d = fetch_polytopes(h11=1000, lattice="N", dim=5, limit=100) # Generator of 5D polytopes
    next(g_5d)
    # A 5-dimensional reflexive lattice polytope in ZZ^5
    ```
    """
    if dim not in (4,5):
        raise ValueError("Only polytopes of dimension 4 or 5 are available.")
    if lattice not in ("N", "M", None):
        raise ValueError("Options for lattice are 'N' and 'M'.")
    if favorable is not None and lattice is None:
        raise ValueError("Lattice must be specified when checking favorability.")
    if h12 is not None and h21 is not None and h12 != h21:
        raise ValueError("Only one of h12 or h21 should be specified.")
    if h12 is None and h21 is not None:
        h12 = h21
    if h13 is not None and h31 is not None and h13 != h31:
        raise ValueError("Only one of h13 or h31 should be specified.")
    if h13 is None and h31 is not None:
        h13 = h31
    fetch_limit = limit
    # if favorable is set to True or False we fetch extra polytopes
    if favorable is not None:
         fetch_limit = (5 if favorable else 10)*fetch_limit + 100
    if dim == 4:
        if h13 is not None or h22 is not None:
            print("Ignoring inputs for h13 and h22.")
        if (lattice is None
                and (h11 is not None or h12 is not None or chi is not None)):
            raise ValueError("Lattice must be specified when Hodge numbers "
                             "or Euler characteristic are given.")
        if lattice == "N":
            h11, h12 = h12, h11
            chi = (-chi if chi is not None else None)
        if (chi is not None and h11 is not None and h12 is not None
                and chi != 2*(h11-h21)):
            raise ValueError("Inconsistent Euler characteristic input.")
        variables = [h11, h12, n_points, n_vertices, n_dual_points, n_facets,
                     chi, fetch_limit]
        names = ["h11", "h12", "M", "V", "N", "F", "chi", "L"]
        parameters = {n:str(v) for n, v in zip(names, variables)
                        if v is not None}
        r = requests.get("http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi",
                         params=parameters, timeout=timeout)
    else:
        if lattice is None and (h11 is not None or h13 is not None):
            raise ValueError("Lattice must be specified when h11 or h13 "
                             "are given.")
        if lattice == "N":
            h11, h13 = h13, h11
        if (chi is not None and h11 is not None and h12 is not None
                and h13 is not None and chi != 48+6*(h11-h12+h13)):
            raise ValueError("Inconsistent Euler characteristic input.")
        if (h22 is not None and h11 is not None and h12 is not None
                and h13 is not None and h22 != 44+6*h11-2*h12+4*h13):
            raise ValueError("Inconsistent h22 input.")
        variables = [h11, h12, h13, h22, chi, fetch_limit]
        names = ["h11", "h12", "h13", "h22", "chi", "limit"]
        url = "http://rgc.itp.tuwien.ac.at/fourfolds/db/5d_reflexive"
        for i,vr in enumerate(variables):
            if vr is not None:
                url += f",{names[i]}={vr}"
        url += ".txt"
        r = requests.get(url, timeout=timeout)
    g = polytope_generator(r.text, input_type="str", dualize=dualize,
                           format=("ks" if dim==4 else "ws"), backend=backend,
                           favorable=favorable, lattice=lattice, limit=limit)
    if as_list:
        return list(g)
    return g


def find_new_affinely_independent_points(points):
    """
    **Description:**
    Finds new points that are affinely independent to the input list of
    points. This is useful when one wants to turn a polytope that is not
    full-dimensional into one that is, without affecting the structure of
    the triangulations.

    **Arguments:**
    - `points` *(array_like)*: A list of points.

    **Returns:**
    *(numpy.ndarray)* A list of affinely independent points with respect to
    the ones inputted.

    **Example:**
    We construct a list of points and then find a set of affinely independent
    points.
    ```python {2}
    pts = [[1,0,1],[0,0,1],[0,1,1]]
    find_new_affinely_independent_points(pts)
    array([[1, 0, 2]])
    ```
    """
    if len(points) == 0:
        raise ValueError("List of points cannot be empty.")
    pts = np.array(points)
    pts_trans = np.array([pt-pts[0] for pt in pts])
    if len(pts) == 1:
        pts_trans = np.array(pts_trans.tolist()+[[1]+[0]*(pts.shape[1]-1)])
    dim = np.linalg.matrix_rank(pts_trans)
    basis_dim = 0
    basis_pts = []
    for pt in pts_trans:
        new_rank = np.linalg.matrix_rank(basis_pts+[pt])
        if new_rank > basis_dim:
            basis_pts.append(pt)
            basis_dim = new_rank
        if basis_dim == dim:
            break
    basis_pts = np.array(basis_pts)
    k, n_k = fmpz_mat(basis_pts.tolist()).nullspace()
    new_pts = np.array(k.transpose().tolist(), dtype=int)[:n_k,:]
    if len(pts) == 1:
        new_pts = np.array(new_pts.tolist() + [[1]+[0]*(pts.shape[1]-1)])
    new_pts = np.array([pt+pts[0] for pt in new_pts])
    return new_pts