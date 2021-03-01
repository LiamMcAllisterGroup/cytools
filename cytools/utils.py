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
from flint import fmpz_mat, fmpq
from scipy.sparse import dok_matrix
import numpy as np
# CYTools imports
from cytools import config



def gcd_float(a, b, tol=1e-5):
    """
    **Description:**
    Compute the greatest common (floating-point) divisor of a and b.

    **Arguments:**
    - ```a``` (float): The first number.
    - ```b``` (float): The second number.
    - ```tol``` (float, optional, default=1e-5): The tolerance for rounding.

    **Returns:**
    (float) The gcd of a and b.
    """
    if abs(b) < tol:
        return abs(a)
    return gcd_float(b,a%b,tol)


def gcd_list(arr):
    """
    **Description:**
    Compute the greatest common divisor of the elements in an array.

    **Arguments:**
    - ```arr``` (list): A list of floating-point numbers.

    **Returns:**
    (float) The gcd of all the elements in the input list.
    """
    return reduce(gcd_float,arr)


def to_sparse(rule_arr_in):
    """
    **Description:**
    Converts an matrix of the form [[a,b, M_ab],...] to a dok_matrix.

    **Arguments:**
    - ```rule_arr_in``` (list): A list of the form [[a,b, M_ab],...].

    **Returns:**
    (dok_matrix) The sparse dok_matrix.
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
    **Description:**
    Solves the sparse linear system M*x=C.

    **Arguments:**
    - ```M``` (dok_matrix): A scipy dok_matrix.
    - ```C``` (list): A vector of floats.
    - ```backend``` (string, optional, default="all"): The sparse linear solver
      to use. Options are "all", "sksparse" and "scipy". When set to "all" it
      tries all available backends.
    - ```check``` (boolean, optional, default=True): Whether to explicitly
      check the solution to the linear system.
    - ```backend_error_tol``` (float, optional, default=1e-4): Error tolerance
      for the solution of the linear system.
    - ```verbose``` (integer, optional, default=0): The verbosity level.
      - verbose = 0: Do not print anything.
      - verbose = 1: Print warnings when backends fail.

    **Returns:**
    (list) Floating-point solution to the linear system.
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
    - ```tensor``` (list): The input symmetric sparse tensor of the form
      [[a,b,...,c,M_ab...c]].
    - ```indices``` (list): The list of indices that will be preserved.

    **Returns:**
    (list) A matrix describing a tensor in the same format as the input, but
    only with the desired indices.
    """
    dim = len(tensor[0]) - 1
    tensor_filtered = [c for c in tensor
                        if all(c[i] in indices for i in range(dim))]
    indices_dict = {vv:v for v,vv in enumerate(indices)}
    tensor_reindexed = sorted([sorted([indices_dict[jj] for jj in ii[:-1]])
                              + [ii[-1]] for ii in tensor_filtered])
    return np.array(tensor_reindexed)


def symmetric_sparse_to_dense_in_basis(tensor, basis):
    """
    **Description:**
    Converts a symmetric sparse tensor of the form [[a,...,b,M_a...b]] to a
    dense tensor and then applies the basis transformation.

    **Arguments:**
    - ```tensor``` (list): A sparse tensor of the form [[a,...,b,M_a...b]].
    - ```basis``` (list): A matrix where the rows are the basis elements.

    **Returns:**
    (list) A dense tensor in the chosen basis.
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


def float_to_fmpq(c):
    """
    **Description:**
    Converts a float to an fmpq.

    **Arguments:**
    - ```c``` (float): The input number.

    **Returns:**
    (fmpq) The rational number that most reasonably approximates the input.
    """
    f = Fraction(c).limit_denominator()
    return fmpq(f.numerator, f.denominator)


def fmpq_to_float(c):
    """
    **Description:**
    Converts an fmpq to a float.

    **Arguments:**
    - ```c``` (fmpq): The input number.

    **Returns:**
    (float) The input number as a float.
    """
    return int(c.p)/int(c.q)


def array_int_to_fmpz(arr):
    """
    **Description:**
    Converts a numpy array with 64-bit integer entries to fmpz entries.

    **Arguments:**
    - ```arr``` (list): A numpy array with 64-bit integer entries.

    **Returns:**
    A numpy array with fmpz entries.
    """
    in_arr = np.array(arr, dtype=int)
    return np.array(fmpz_mat(in_arr.tolist()).tolist())


def array_float_to_fmpq(arr):
    """
    **Description:**
    Converts a numpy array with floating-point entries to fmpq entries.

    **Arguments:**
    - ```arr``` (list): A numpy array with floating-point entries.

    **Returns:**
    A numpy array with fmpq entries.
    """
    in_arr = np.array(arr, dtype=float)
    fmpq_flat = np.empty(len(in_arr.flat), dtype=fmpq)
    for i,c in enumerate(in_arr.flat):
        f = Fraction(c).limit_denominator()
        fmpq_flat[i] = fmpq(f.numerator, f.denominator)
    return fmpq_flat.reshape(in_arr.shape)


def array_fmpz_to_int(arr):
    """
    **Description:**
    Converts a numpy array with fmpz entries to 64-bit integer entries.

    **Arguments:**
    - ```arr``` (list): A numpy array with fmpz entries.

    **Returns:**
    A numpy array with 64-bit integer entries.
    """
    return np.array(arr, dtype=int)


def array_fmpq_to_float(arr):
    """
    **Description:**
    Converts a numpy array with fmpq entries to floating-point entries.

    **Arguments:**
    - ```arr``` (list): A numpy array with fmpq entries.

    **Returns:**
    A numpy array with floating-point entries.
    """
    in_arr = np.array(arr, dtype=fmpq)
    float_flat = np.empty(len(in_arr.flat), dtype=float)
    for i,c in enumerate(in_arr.flat):
        float_flat[i] = int(c.p)/int(c.q)
    return float_flat.reshape(in_arr.shape)


def polytope_generator(input, input_type="file", format="ks", backend=None,
                       dualize=False):
    """
    **Description:**
    Reads the polytopes from a file or a string containing a list of polytopes
    in the format used in the Kreuzer-Skarke database, or a list of weight
    systems.

    **Arguments:**
    - ```input``` (string): Specifies the name of the file to read or the
      string containing the polytopes.
    - ```input_type``` (string, optional, default="file"): Specifies whether to
      read from a file or from the input string.  Options are "file" or
      "string".
    - ```format``` (string, optional, default="ks"): Specifies the format to
      read. The options are "ks", which is the format used in the KS database,
      and "ws", if the polytopes should be constructed from weight systems.
    - ```backend``` (string, optional): A string that specifies the backend
      used for the [```Polytope```](./polytope) class.
    - ```dualize``` (boolean, optional, default=False): Flag that indicates
      whether to dualize all the polytopes before returning them.

    **Returns:**
    (generator) A generator of [```Polytope```](./polytope) objects.
    """
    from cytools import Polytope
    if input_type not in ["file", "string"]:
        raise Exception("\"input_type\" must be either \"file\" or \"string\"")
    if input_type == "file":
        in_file = open(input, "r")
        l = in_file.readline()
    else:
        in_string = input.split("\n")
        l = in_string[0]
        in_string.pop(0)
    if format == "ws":
        while True:
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
            if vert.shape[0] < vert.shape[1]:
                vert = vert.T
            p = Polytope(vert, backend=backend)
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
        raise Exception("Unsupported format. Options are \"ks\" and \"ws\".")
    while format == "ks":
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
                raise Exception("Error: Dimensions of array do not match")
            if m > n:
                vert = vert.T
            p = Polytope(vert, backend=backend)
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
                   dualize=False, as_list=False):
    """
    **Description:**
    Reads the polytopes from a file or a string containing a list of polytopes
    in the format used in the Kreuzer-Skarke database, or a list of weight
    systems.

    **Arguments:**
    - ```input``` (string): Specifies the name of the file to read or the
      string containing the polytopes.
    - ```input_type``` (string, optional, default="file"): Specifies whether to
      read from a file or from the input string.  Options are "file" or
      "string".
    - ```format``` (string, optional, default="ks"): Specifies the format to
      read. The options are "ks", which is the format used in the KS database,
      and "ws", if the polytopes should be constructed from weight systems.
    - ```backend``` (string, optional): A string that specifies the backend
      used for the [```Polytope```](./polytope) class.
    - ```dualize``` (boolean, optional, default=False): Flag that indicates
      whether to dualize all the polytopes before returning them.
    - ```as_list``` (boolean, optional, default=False): Return the list of
      polytopes instead of a generator.

    **Returns:**
    (generator or list) A generator of [```Polytope```](./polytope) objects, or
    the full list when setting as_list=True.
    """
    g = polytope_generator(input, input_type=input_type, format=format,
                           backend=backend, dualize=dualize)
    if as_list:
        return list(g)
    return g


def fetch_polytopes(h11=None, h12=None, h13=None, h21=None, h22=None, h31=None,
                    chi=None, lattice=None, dim=4, n_points=None,
                    n_vertices=None, n_dual_points=None, n_facets=None,
                    limit=1000, timeout=60, dualize=False, as_list=False):
    """
    **Description:**
    Fetch reflexive polytopes from the Kreuzer-Skarke database or from the
    Schöller-Skarke database. The data is fetched from websites
    http://hep.itp.tuwien.ac.at/~kreuzer/CY/ and
    http://rgc.itp.tuwien.ac.at/fourfolds/ respectively.

    **Arguments:**
    - ```h11``` (integer, optional): Specifies the Hodge number $h^{1,1}$ of
      the Calabi-Yau hypersurface.
    - ```h12``` (integer, optional): Specifies the Hodge number $h^{1,2}$ of
      the Calabi-Yau hypersurface.
    - ```h13``` (integer, optional): Specifies the Hodge number $h^{1,3}$ of
      the Calabi-Yau hypersurface.
    - ```h21``` (integer, optional): Specifies the Hodge number $h^{2,1}$ of
      the Calabi-Yau hypersurface. This is equivalent to the h12 parameter.
    - ```h22``` (integer, optional): Specifies the Hodge number $h^{2,2}$ of
      the Calabi-Yau hypersurface.
    - ```h31``` (integer, optional): Specifies the Hodge number $h^{3,1}$ of
      the Calabi-Yau hypersurface. This is equivalent to the h13 parameter.
    - ```chi``` (integer, optional): Specifies the Euler characteristic of the
      Calabi-Yau hypersurface.
    - ```lattice``` (string, optional): Specifies the lattice on which the
      polytope is defined. Options are 'N' and 'M'. Has to be specified if the
      Hodge numbers or the Euler characteristic is specified.
    - ```dim``` (integer, optional, default=4): The dimension of the polytope.
      The only available options are 4 and 5.
    - ```n_points``` (integer, optional): Specifies the number of lattice
      points of the desired polytopes.
    - ```n_vertices``` (integer, optional): Specifies the number of vertices of
      the desired polytopes.
    - ```n_dual_points``` (integer, optional): Specifies the number of points
      of the dual polytopes of the desired polytopes.
    - ```n_facets``` (integer, optional): Specifies the number of facets of the
      desired polytopes.
    - ```limit``` (integer, optional, default=1000): Specifies the maximum
      number of fetched polytopes.
    - ```timeout``` (integer, optional, default=60): Specifies the maximum
      number of seconds to wait for the server to return the data.
    - ```dualize``` (boolean, optional, default=False): Flag that indicates
      whether to dualize all the polytopes before returning them.
    - ```as_list``` (boolean, optional, default=False): Return the list of
      polytopes instead of a generator.

    **Returns:**
    (generator or list) A generator of [```Polytope```](./polytope) objects, or
    the full list when as_list=True.
    """
    if dim not in (4,5):
        raise Exception("Only polytopes of dimension 4 or 5 are available.")
    if lattice not in ("N", "M", None):
        raise Exception("Options for lattice are 'N' and 'M'.")
    if h12 is not None and h21 is not None and h12 != h21:
        raise Exception("Only one of h12 or h21 should be specified.")
    if h12 is None and h21 is not None:
        h12 = h21
    if h13 is not None and h31 is not None and h13 != h31:
        raise Exception("Only one of h13 or h31 should be specified.")
    if h13 is None and h31 is not None:
        h13 = h31
    if dim == 4:
        if h13 is not None or h22 is not None:
            print("Ignoring inputs for h13 and h22.")
        if (lattice is None
                and (h11 is not None or h12 is not None or chi is not None)):
            raise Exception("Lattice must be specified when Hodge numbers "
                            "or Euler characteristic are given.")
        if lattice == "N":
            h11, h12 = h12, h11
            chi = (-chi if chi is not None else None)
        if (chi is not None and h11 is not None and h12 is not None
                and chi != 2*(h11-h21)):
            raise Exception("Inconsistent Euler characteristic input.")
        variables = [h11, h12, n_points, n_vertices, n_dual_points, n_facets,
                     chi, limit]
        names = ["h11", "h12", "M", "V", "N", "F", "chi", "L"]
        parameters = {n:str(v) for n, v in zip(names, variables)
                        if v is not None}
        r = requests.get("http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi",
                         params=parameters, timeout=timeout)
    else:
        if lattice is None and (h11 is not None or h13 is not None):
            raise Exception("Lattice must be specified when h11 or h13 "
                            "are given.")
        if lattice == "N":
            h11, h13 = h13, h11
            if (chi is not None and h11 is not None and h12 is not None
                    and h13 is not None and chi != 48+6*(h11-h12+h13)):
                raise Exception("Inconsistent Euler characteristic input.")
            if (h22 is not None and h11 is not None and h12 is not None
                    and h13 is not None and h22 != 44+6*h11-2*h12+4*h13):
                raise Exception("Inconsistent h22 input.")
        variables = [h11, h12, h13, h22, chi, limit]
        names = ["h11", "h12", "h13", "h22", "chi", "limit"]
        url = "http://rgc.itp.tuwien.ac.at/fourfolds/db/5d_reflexive"
        for i,vr in enumerate(variables):
            if vr is not None:
                url += f",{names[i]}={vr}"
        url += ".txt"
        r = requests.get(url, timeout=timeout)
    g = polytope_generator(r.text, input_type="string", dualize=dualize,
                           format=("ks" if dim==4 else "ws"))
    if as_list:
        return list(g)
    return g
