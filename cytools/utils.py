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

from functools import reduce
import numpy as np
from scipy.sparse import dok_matrix
from itertools import permutations
from fractions import Fraction
from flint import fmpz_mat, fmpq_mat, fmpz, fmpq
from multiprocessing import Process, Queue
from scipy.optimize import nnls

def gcd_float(a, b, tol=1e-5):
    """Compute the greatest common (floating point) divisor of a and b.

    Args:
        a (float): The first number.
        b (float): The second number.
        tol (float, optional, default=1e-5): The tolerance for rounding.

    Returns:
        float: The gcd of a and b.
    """
    if abs(b) < tol:
        return abs(a)
    return gcd_float(b,a%b,tol)


def gcd_list(arr):
    """Compute the greatest common divisor of the elements in an array.

    Args:
        arr (list): A list of floating point numbers.

    Returns:
        float: The gcd of all the elements in the input list.
    """
    return reduce(gcd_float,arr)


def to_sparse(rule_arr_in):
    """Converts an matrix of the form [[a,b, M_ab]] to a dok_matrix.

    Args:
        rule_arr_in (list): A list containing the rules.

    Returns:
        scipy dok_matrix: The sparse dok_matrix. 

    """
    rule_arr = np.array(rule_arr_in)
    dim_0 = max(rule_arr[:,0])
    dim_1 = max(rule_arr[:,1])
    sp_mat = dok_matrix((dim_0,dim_1))
    for r in rule_arr:
        sp_mat[r[0],r[1]] = r[2]
    return sp_mat


def solve_linear_system(M, C, backend="all", check=True,
                        backend_error_tol=1e-4, verbose=0):
    """
    Solves the sparse linear system M*x=C.

    Args:
        M (dok_matrix): A scipy dok_matrix.
        C (list): A vector of floats.
        backend (string, optional, default="all"): The sparse linear solver to
            use. Options are "all", "sksparse" and "scipy". When set to "all" 
            tries all available backends.
        check (boolean, optional, default=True): Whether to explicitly check
            the solution to the linear system.
        backend_error_tol (float, optional, default=1e-4): Error tolerance for
            the solution of the linear system.
        verbose (int, optional, default=0): Verbosity level:
            - verbose = 0: Do not print anything.
            - verbose = 1: Print warnings when backends fail.

    Returns:
        list: Floating point solution to the linear system.
    """
    backends = ["all", "sksparse", "scipy"]
    if backend not in backends:
        raise Exception("Invalid linear system backend. "
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
            from scipy.sparse.linalg import dsolve
            MM = M.transpose()*M
            CC = -M.transpose()*C
            solution = dsolve.spsolve(MM, CC).tolist()
            system_solved = True
        except:
            if verbose >= 1:
                print("Linear backend error: scipy failed.")
            system_solved = False
    if system_solved and check:
        res = np.dot(M.todense(), solution) + C
        max_error = max(abs(s) for s in res.flatten().tolist()[0])
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
    Selects a specific subset of indices from a tensor.  The tensor is
    reindexed so that indices are in the range 0..len(indices) with the
    ordering specified by the input indices.

    This function can be used to convert the tensor of triple intersection
    numbers to a given basis.

    Args:
        tensor (list): A list describing a n-tensor. Each element of the list
            is a list of length n+1, with the first n elements being indices
            and the last entry being the value at the corresponding position.
        indices (list): The list of indices that will be preserved.

    Returns:
        np.array: A matrix describing a tensor in the same format as the input,
            but only with the desired indices.
    """
    dim = len(tensor[0]) - 1
    tensor_filtered = [c for c in tensor
                        if all(c[i] in indices for i in range(dim))]
    indices_dict = {vv:v for v,vv in enumerate(indices)}
    tensor_reindexed = sorted([[indices_dict[jj] for jj in ii[:-1]] + [ii[-1]]
                               for ii in tensor_filtered])
    return np.array(tensor_reindexed)


def symmetric_sparse_to_dense_in_basis(tensor, basis, check=True):
    """
    Converts a symmetric sparse tensor of the form [[a,b, ... , M_ab...]] to a
    dense tensor and then applies the basis transformation.

    The inverse transformation is used to make sure that the change of basis
    succeeded, and if not the computation is performed again using arbitrary
    precision integers instead of 64-bit integers.
    """
    dense_tensor = np.zeros((basis.shape[1],)*(tensor.shape[1]-1),
                            dtype=tensor.dtype)
    for ii in tensor:
        for c in permutations([int(round(cc)) for cc in ii[:-1]]):
            dense_tensor[c] = ii[-1]
    dense_result = np.array(dense_tensor)
    for i in list(range(tensor.shape[1]-1))[::-1]:
        dense_result = np.tensordot(dense_result, basis, axes=[[i],[1]])
    return dense_result


def is_extremal(A, b, i=None, q=None, tol=10e-4):
    """
    Returns True if the ray is extremal and False otherwise. It has additional
    parameters that are used when parallelizing.

    Args:
        A (list): A matrix where the columns are rays (excluding b).
        b (list): The vector that will be checked if it can be expressed as a
            positive linear combination of the columns of A.
        i (int): An id number that is used when parallelizing.
        q (Queue): A queue that is used when parallelizing.
        tol (float): The tolerance for determining whether a ray is extremal.

    Returns:
        bool: True if the ray is extremal and False otherwise.
    """
    try:
        v = nnls(A,b)
        is_ext = abs(v[1]) > tol
        if q is not None:
            q.put((i, is_ext))
        return is_ext
    except:
        if q is not None:
            q.put((i,None))
        return


def find_extremal(in_rays, tol=1e-4, n_threads=1, verbose=False):
    rays = np.array(in_rays, dtype=np.int)
    current_rays = set(range(rays.shape[0]))
    ext_rays = set()
    error_rays = set()
    rechecking_rays = False
    failed_after_rechecking = False
    while True:
        checking = []
        for i in current_rays:
            if i not in ext_rays and (i not in error_rays or rechecking_rays):
                checking.append(i)
            if len(checking) >= n_threads:
                break
        if len(checking) == 0:
            if rechecking_rays:
                break
            else:
                rechecking_rays = True
        As = [np.array([rays[j] for j in current_rays if j != k], dtype=int).T
                for k in checking]
        bs = [rays[k] for k in checking]
        q = Queue()
        procs = [Process(target=is_extremal,
                 args=(As[k], bs[k], k, q, tol)) for k in range(len(checking))]
        for t in procs:
            t.start()
        for t in procs:
            t.join()
        results = [q.get() for j in range(len(checking))]
        for res in results:
            if res[1] is None:
                error_rays.add(checking[res[0]])
                if rechecking_rays:
                    failed_after_rechecking = True
                    ext_rays.add(checking[res[0]])
                elif verbose:
                    print("Minimizatio failed. Ray will be rechecked later...")
            elif not res[1]:
                current_rays.remove(checking[res[0]])
            else:
                ext_rays.add(checking[res[0]])
            if rechecking_rays:
                error_rays.remove(checking[res[0]])
        if verbose:
            print(f"Eliminated {sum(not r[1] for r in results)}. "
                  f"Current number of rays: {len(current_rays)}")
    if failed_after_rechecking:
        print("Warning: Minimization failed after multiple attempts. "
              "Some rays may not be extremal.")
    return rays[list(ext_rays),:]


def np_int_to_fmpz(mat):
    """
    Converts a numpy array with integer entries to fmpz entries.
    """
    m = np.array(mat, dtype=int)
    return np.array(fmpz_mat(m.tolist()).table())


def np_float_to_fmpq(mat):
    """
    Converts a numpy array with floating-point entries to fmpq entries.
    """
    in_flat = mat.flatten()
    fmpq_flat = np.empty(in_flat.shape[0], dtype=fmpq)
    for i in range(in_flat.shape[0]):
        f = Fraction(in_flat[i]).limit_denominator()
        fmpq_flat[i] = fmpq(f.numerator, f.denominator)
    return fmpq_flat.reshape(mat.shape)


def np_fmpz_to_int(mat):
    """
    Converts a numpy array with fmpq entries to integer entries.
    """
    return np.array(mat, dtype=int)


def np_fmpq_to_float(mat):
    """
    Converts a numpy array with fmpq entries to floating-point entries.
    """
    in_flat = mat.flatten()
    float_flat = np.empty(in_flat.shape[0], dtype=float)
    for i in range(in_flat.shape[0]):
        float_flat[i] = int(in_flat[i].p)/int(in_flat[i].q)
    return float_flat.reshape(mat.shape)

def unique(sequence):
    """
    Finds indices of the unique elements of a tuple.

    Args:
        sequence(tuple): A tuple.

    Returns:
        tuple: A tuple containing indices of the unique elements.
    """
    seen = set()
    return [i for i,x in enumerate(sequence) if not (x in seen or seen.add(x))]

def remove_duplicate_triangulations(triangulation_list):
    """
    Removes duplicate triangulations.

    Args:
        triangulation_list (iterable): A list Triangulation objects.

    Returns:
        list: A list containing unique Triangulation objects.
    """
    simplices_list = (tri.simplices() for tri in triangulation_list)
    indices_list = unique(tuple(tuple(ii) for ii in i) for i in simplices_list)
    return [triangulation_list[i] for i in indices_list]
