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
# Description:  This module contains tools designed for Calabi-Yau hypersurface
#               computations.
# -----------------------------------------------------------------------------


# 'standard' imports
from itertools import combinations, combinations_with_replacement, permutations, product

# 3rd party imports
import numpy as np
import math
from flint import fmpz_mat
from sympy import Matrix, ZZ, lcm, fraction
from sympy.matrices.normalforms import smith_normal_decomp
from sympy.matrices.normalforms import smith_normal_form
from scipy.optimize import nnls
from scipy.optimize import milp, LinearConstraint

# CYTools imports
from cytools import Polytope, h_polytope, Cone
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration


def compute_partition(divisors,rays):

    """

    Attempts to represent divisors as a partition of the anticanonical class.

    **Description:**

    Uses linear equivalence to shift a collection of toric divisors into a representation whose coefficients form a partition of the anticanonical divisor, if such a representation exists.

    **Arguments:**

    - `divisors (array-like)`: The divisor coefficient vectors.
    - `rays (numpy.ndarray)`: The primitive ray generators of the toric fan.

    **Returns:**

    - `tuple`: A pair `(exists, shifted_divisors)`, where `exists` is a boolean and `shifted_divisors` is the shifted divisor array if it exists, otherwise `None`.

    """

    null = np.zeros([rays.shape[0],rays.shape[1]],dtype=int)
    linear1 = np.tensordot(np.identity(2,dtype=int),rays,axes=0)
    linear1 = linear1.transpose(0, 2, 1, 3).reshape(len(divisors)*rays.shape[0], len(divisors)*rays.shape[1])
    linear2 = np.hstack([rays]*len(divisors))

    affine1 = np.concatenate(divisors)
    affine2 = sum(divisors)-1

    lower1 = -affine1
    upper1 = np.full_like(affine1,np.inf,dtype=float)
    lower2 = -affine2
    upper2 = -affine2

    linear = np.vstack([linear1,linear2])
    lower = np.concatenate([lower1,lower2])
    upper = np.concatenate([upper1,upper2])

    c = np.zeros(linear.shape[1])
    integrality = np.ones_like(c)

    constraints = LinearConstraint(linear, lower, upper)
    res = milp(c=np.zeros(linear.shape[1]),
               constraints=constraints,
               integrality=np.ones(linear.shape[1]),
               bounds=(-np.inf, np.inf))
    
    if not res.success or res.x is None:
        return (False, None)

    sol = np.rint(res.x.reshape(len(divisors),len(rays[0]))@(rays.T)+np.array(divisors)).astype(int)
    return (True,sol)
    
def contains_row(arr: np.array, target: np.array):
    """

    Checks whether an array contains a given row.

    **Description:**

    Determines whether the one-dimensional array `target` occurs as a row of the two-dimensional array `arr`.

    **Arguments:**

    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `target (numpy.ndarray)`: The one-dimensional row to find.

    **Returns:**

    - `bool`: `True` if `target` is a row of `arr`, otherwise `False`.

    """
    return np.any(np.all(arr == target, axis=1))

def contains_rows(arr: np.array,targets: np.array):
    """

    Checks whether an array contains all target rows.

    **Description:**

    Determines whether every row of `targets` occurs as a row of `arr`.

    **Arguments:**

    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `targets (numpy.ndarray)`: The target rows.

    **Returns:**

    - `bool`: `True` if every target row occurs in `arr`, otherwise `False`.

    """
    return np.any((targets[:, None, :] == arr[None, :, :]).all(axis=2), axis=1).all()

def get_same_rows(A: np.array, B: np.array):
    """

    Returns rows shared by two arrays.

    **Description:**

    Computes the rows of `A` that also occur as rows of `B`.

    **Arguments:**

    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.

    **Returns:**

    - `numpy.ndarray`: The rows of `A` that also occur in `B`.

    """
    return A[np.where((A[:, None, :] == B[None, :, :]).all(axis=2))[0]]

def same_rows(A, B):
    """

    Checks whether two arrays have the same rows.

    **Description:**

    Determines whether `A` and `B` contain the same rows with the same multiplicities, independent of row order.

    **Arguments:**

    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.

    **Returns:**

    - `bool`: `True` if `A` and `B` have the same rows with the same multiplicities, otherwise `False`.

    """
    rowsA, countsA = np.unique(A, axis=0, return_counts=True)
    rowsB, countsB = np.unique(B, axis=0, return_counts=True)
    return np.array_equal(rowsA, rowsB) and np.array_equal(countsA, countsB)

def dual_face_Cayley_polytope(Cdvert: np.array,f):
    """

    Returns the dual face of a face of a Cayley polytope.

    **Description:**

    Computes the face of the dual Cayley polytope whose vertices pair trivially with all vertices of the input face `f`.

    **Arguments:**

    - `Cdvert (numpy.ndarray)`: Vertices of the dual Cayley polytope.
    - `f`: A face of the Cayley polytope.

    **Returns:**

    - `Polytope`: The dual face.

    """
    return Polytope(Cdvert[np.all(Cdvert@f.vertices().T == 0, axis=1)])
    
def h11_2_part(Cay: Polytope,Cayd: Polytope,det=False):
    """

    Computes h11 for a two-part nef partition.

    **Description:**

    Computes the Hodge number `h^{1,1}` of a complete intersection Calabi-Yau described by a two-part nef partition using the associated Cayley polytope and its dual.

    **Arguments:**

    - `Cay (Polytope)`: The Cayley polytope.
    - `Cayd (Polytope)`: The dual Cayley polytope.
    - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.

    **Returns:**

    - `int`: The Hodge number `h^{1,1}`.

    """
    Cdvert=Cayd.vertices()
    n=Cay.dim()-1
    h11_ret=len(Cayd.points())-n-2
    
    if det:
        print("Trivial term: ",h11_ret)
        
    for f in Cayd.faces(n):
        h11_ret=h11_ret-len(Polytope(2*(f.vertices())).interior_points())
        
    if det:
        print("After 2*dual facets: ",h11_ret)
        
    for f in Cayd.faces(n-1):
        h11_ret=h11_ret+len(f.interior_points())
        
    if det:
        print("After dual codim-2: ",h11_ret)
        
    for f in Cay.faces(1):
        k=len(f.interior_points())
        if k>0:
            h11_ret=h11_ret+(k*(len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())))
        
    if det: 
        print("After 1-face/2*codim-2 dual face: ",h11_ret)
        
    for f in Cay.faces(2):
        k=len(Polytope(2*(f.vertices())).interior_points())
        if k>0:
            h11_ret=h11_ret-(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
        
    if det:
        print("After 2*(2-face)/codim-3 dual face: ",h11_ret)
        
    for f in Cay.faces(3):
        k=len(Polytope(2*(f.vertices())).interior_points())
        if k>0:
            h11_ret=h11_ret+(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(2):
                h11_ret=h11_ret-(len(g.interior_points())*k)
        
    return h11_ret

def h21_2_part(Cay: Polytope,Cayd: Polytope,det=False):
    """

    Computes h21 for a two-part nef partition.

    **Description:**

    Computes the Hodge number `h^{2,1}` of a complete intersection Calabi-Yau described by a two-part nef partition in a six-dimensional ambient variety.

    **Arguments:**

    - `Cay (Polytope)`: The Cayley polytope.
    - `Cayd (Polytope)`: The dual Cayley polytope.
    - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.

    **Returns:**

    - `int`: The Hodge number `h^{2,1}`.

    """
    Cdvert=Cayd.vertices()
    n=Cay.dim()-1
    h21_ret=0
    for f in Cay.faces(2):
        h21_ret=h21_ret+len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())*len(f.interior_points())
    if det:
        print(h21_ret)
    for f in Cay.faces(4):
        h21_ret=h21_ret+len(Polytope(2*(f.vertices())).interior_points())*len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
    if det:
        print(h21_ret)
    for f in Cay.faces(3):
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(2):
                h21_ret=h21_ret-len(g.interior_points())*k
    if det: 
        print(h21_ret)
    for f in Cay.faces(4):
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(3):
                h21_ret=h21_ret-len(g.interior_points())*k
    return h21_ret

def get_indices(arr: np.array,targets: np.array):
    """

    Returns indices of target rows.

    **Description:**

    Finds the indices of rows of `arr` that also occur among the rows of `targets`.

    **Arguments:**

    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `targets (numpy.ndarray)`: The target row or rows.

    **Returns:**

    - `numpy.ndarray`: The indices of matching rows in `arr`.

    """
    return  np.where(np.any((arr[:, None, :] == targets).all(axis=2), axis=1))[0]

def get_index(arr: np.array,target: np.array):
    """

    Returns the index of a target row.

    **Description:**

    Finds the indices of rows of `arr` equal to the one-dimensional array `target`.

    **Arguments:**

    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `target (numpy.ndarray)`: The target row.

    **Returns:**

    - `numpy.ndarray`: The indices of rows of `arr` equal to `target`.

    """
    return  np.where(np.all(arr == target, axis=1))[0]

def glsm_from_points(pts):
    """

    Computes a GLSM charge matrix from toric points.

    **Description:**

    Computes an integral basis of linear relations among the input toric points using Smith normal form.

    **Arguments:**

    - `pts (array-like)`: The toric point configuration.

    **Returns:**

    - `numpy.ndarray`: A GLSM charge matrix for the point configuration.

    """
    a,s,t=smith_normal_decomp(Matrix(np.array(pts).T),domain=ZZ)
    aa=np.array(a,dtype=int)
    ss=np.array(s,dtype=int)
    tt=np.array(t,dtype=int)
    rank_a=np.linalg.matrix_rank(aa)
    return tt.T[rank_a:]

def points_from_glsm(glsm):
    """

    Computes toric points from a GLSM charge matrix.

    **Description:**

    Computes an integral point configuration whose relations are described by the input GLSM charge matrix.

    **Arguments:**

    - `glsm (array-like)`: The GLSM charge matrix.

    **Returns:**

    - `numpy.ndarray`: A toric point configuration.

    """
    D,U,V=smith_normal_decomp(Matrix(np.array(glsm).T),domain=ZZ)
    DD=np.array(D,dtype=int)
    UU=np.array(U,dtype=int)
    VV=np.array(V,dtype=int)
    rank_D=np.linalg.matrix_rank(DD)
    return (UU[rank_D:].astype(int)).T

def find_trilayer_vertex_polytope(p,as_index=False):
    """

    Finds the distinguished vertex of a trilayer polytope.

    **Description:**

    Uses the GLSM charge matrix of the vertices to identify the vertex corresponding to half the anticanonical class.

    **Arguments:**

    - `p (Polytope)`: The trilayer polytope.

    - `as_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.

    **Returns:**

    - `numpy.ndarray` or `int`: The distinguished vertex, or its index if `as_index=True`.

    """
    glsm_vert=glsm_from_points(p.vertices())
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_index:
        return index
    else:
        return p.vertices()[index]

def find_trilayer_vertex_vertices(V,as_vertex_index=False):

    """

    Finds the distinguished vertex of a trilayer vertex set.

    **Description:**

    Uses the GLSM charge matrix of the vertex set to identify the vertex corresponding to half the anticanonical class.

    **Arguments:**

    - `V (array-like)`: The vertices of a trilayer polytope.
    - `as_vertex_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.

    **Returns:**

    - `numpy.ndarray` or `int`: The distinguished vertex, or its index if `as_vertex_index=True`.

    """
    
    glsm_vert=glsm_from_points(V)
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_vertex_index:
        return index
    else:
        return V[index]



   
        
def trilayer_normal_form(p):
    """

    Computes a normal form for a trilayer polytope.

    **Description:**

    Applies an integral change of basis that moves the distinguished trilayer vertex into a standard position.

    **Arguments:**

    - `p (Polytope)`: The trilayer polytope.

    **Returns:**

    - `Polytope`: The polytope in trilayer normal form.

    """
    verts=p.vertices()
    index=find_trilayer_vertex_vertices(verts,as_vertex_index=True)
    verts[[0,index]]=verts[[index,0]]
    aa,ss,tt=smith_normal_decomp(Matrix(verts),domain=ZZ)
    a=np.array(aa,dtype=int)
    s=np.array(ss,dtype=int)
    t=np.array(tt,dtype=int)
    b=np.ones(len(verts),dtype=int)
    b[0]=-1
    c=s@b
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(len(y)):
        if a[ii][ii]!=0:
            y[ii]=c[ii]/a[ii][ii]
    r=t@y
    aa2,ss2,tt2=smith_normal_decomp(Matrix(r[:,None]),domain=ZZ)
    a2=np.array(aa2,dtype=int)
    s2=np.array(ss2,dtype=int)
    t2=np.array(tt2,dtype=int)
    U0=np.round(np.linalg.inv(s2).T).astype(int)
    M=U0@verts.T
    for i in range(1,p.ambient_dim()):
        U0[i]=U0[i]+M[i,0]*U0[0]
    return Polytope((U0@verts.T).T)

def Newton_Polytope(pts,weights):
    """

    Computes the Newton polytope of a toric divisor.

    **Description:**

    Constructs the Newton polytope associated with a toric divisor with coefficient vector `weights` on the fan with rays `pts`.

    **Arguments:**

    - `pts (numpy.ndarray)`: Matrix whose rows are the rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `Polytope`: The Newton polytope of the divisor.

    """
    return h_polytope.HPolytope(np.column_stack([pts, weights]).astype(int))

def row_difference(A: np.array, B: np.array):
    """

    Returns rows of one array not contained in another.

    **Description:**

    Computes all rows of `A` that do not occur as rows of `B`.

    **Arguments:**

    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.

    **Returns:**

    - `numpy.ndarray`: Rows of `A` that are not rows of `B`.

    """
    return A[~np.any((A[:, None, :] == B[None, :, :]).all(axis=2), axis=1)]

def points_not_interior_to_facets_and_codim2_faces(p: Polytope):
    """

    Returns points not interior to facets or codimension-two faces.

    **Description:**

    Computes the lattice points of a polytope after removing points interior to facets and codimension-two faces.

    **Arguments:**

    - `p (Polytope)`: The polytope.

    **Returns:**

    - `numpy.ndarray`: The selected lattice points.

    """
    pts=p.points_not_interior_to_facets()
    for f in p.faces(p.dim()-2):
        if len(f.interior_points())>0:
            pts=np.delete(pts,get_indices(pts,f.interior_points()),axis=0)
    return pts
    


def get_lower_dimensional_cones(cones,d):
    """

    Returns lower-dimensional cones of a fan.

    **Description:**

    Computes all `d`-element faces of the given maximal cones.

    **Arguments:**

    - `cones (iterable)`: Collection of cones, each represented by a tuple of one-indexed ray labels.
    - `d (int)`: Number of rays in the lower-dimensional cones to extract.

    **Returns:**

    - `list`: The list of distinct `d`-ray cones.

    """
    return list({combo for row in cones for combo in combinations(row, d)})

def lattice_refinement(q, denominator = 2):

    """

    Computes a lattice refinement map.

    **Description:**

    Returns the smallest integral embedding of the unit lattice into a refined lattice in which `q/denominator` becomes integral.

    **Arguments:**

    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.

    **Returns:**

    - `numpy.ndarray`: The lattice refinement map.

    """
    
    lattice_basis = denominator*np.identity(len(q)).astype(int)
    lattice_basis = np.vstack([denominator*np.identity(len(q)).astype(int),[q]])
    A = fmpz_mat(lattice_basis.tolist())
    A_lll = A.lll()
    scaled_up_basis = np.array(A_lll.tolist()).astype(int)
    vanishing_pos = np.where(np.all(scaled_up_basis == 0,axis=1))[0]
    scaled_up_basis = np.delete(scaled_up_basis,vanishing_pos,axis=0)
    Lambda = np.linalg.inv(scaled_up_basis.T)
    
    return np.rint(Lambda*denominator).astype(int)

def toric_orbifold(pts_CY_ambient,q,denominator=2):

    """

    Constructs the toric data of a lattice orbifold.

    **Description:**

    Applies the lattice refinement defined by `q/denominator` to the ambient toric rays and returns the primitive orbifold rays together with the ray rescalings.

    **Arguments:**

    - `pts_CY_ambient (numpy.ndarray)`: Rays of the original Calabi-Yau ambient toric fan.
    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.

    **Returns:**

    - `tuple`: A pair `(orbifold_points, rescalings)` consisting of primitive orbifold rays and the corresponding edge rescalings.

    """
            
    Lambda = lattice_refinement(q,denominator)
    orbifold_points = pts_CY_ambient@(Lambda.T)
    rescalings = np.array([math.gcd(*list(i)) for i in orbifold_points])
    orbifold_points = np.rint((orbifold_points.T/rescalings).T).astype(int)
    return (orbifold_points,rescalings)

def O3O7_line_bundle(pts_CY_ambient,q,rescalings):

    """

    Computes the O3/O7 line bundle on the toric orbifold.

    **Description:**

    Determines the divisor coefficients of the orientifold line bundle by selecting a projected-in monomial of the Calabi-Yau hypersurface Newton polytope and rescaling divisor classes under the orbifold map.

    **Arguments:**

    - `pts_CY_ambient (numpy.ndarray)`: Rays of the original Calabi-Yau ambient toric fan.
    - `q (array-like)`: Integer vector defining the `Z_2` action.
    - `rescalings (array-like)`: Rescalings of toric divisor classes under the orbifold map.

    **Returns:**

    - `numpy.ndarray` or `None`: The O3/O7 line-bundle coefficients, or `None` if no projected-in monomial is found.

    """
            
    CY3_equation_newton_polytope = Newton_Polytope(pts_CY_ambient,[1]*len(pts_CY_ambient))
    projected_in_monomial_indices = np.where(np.mod(CY3_equation_newton_polytope.points()@q,2)==1)[0]
    if len(projected_in_monomial_indices)==0:
        return None
    arbitrary_monomial_point = CY3_equation_newton_polytope.points()[projected_in_monomial_indices[0]]
    line_bundle_weights_CYhypersurface = pts_CY_ambient@arbitrary_monomial_point+1
    line_bundle_weights_FtheoryBase = np.rint(line_bundle_weights_CYhypersurface/rescalings).astype(int)

    return line_bundle_weights_FtheoryBase


def Z2_fixed_locus(vc_triangulation,q,cone_dimension=None,denominator=2):

    """

    Computes fixed toric strata of a `Z_2` orbifold action.

    **Description:**

    Finds cones whose associated toric strata are fixed by the lattice refinement defined by `q/denominator`. Optionally restricts to cones of a specified dimension.

    **Arguments:**

    - `vc_triangulation (Fan)`: The toric fan of the ambient variety.
    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `cone_dimension (int or None)`: If specified, only cones with this number of rays are considered. Defaults to `None`.

    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.

    **Returns:**

    - `list`: Fixed-locus cones, represented as tuples of one-indexed ray labels.

    """
    if type(cone_dimension)==type(None):
        all_cones = {j for c in vc_triangulation.cones() for i in range(1,len(c))  for j in combinations(c,i)}
        all_cones = [c for c in all_cones]
    else:
        all_cones = list(get_lower_dimensional_cones(vc_triangulation.cones(),cone_dimension))
    fixed_locus_cones = [all_cones[i] for i in np.where([np.all(np.mod(sum(vc_triangulation.vectors()[np.array(c)-1])+q,denominator)==0) for c in all_cones])[0]]

    return fixed_locus_cones


def inequivalent_Z2_actions(lattice_symmetries):

    """

    Enumerates inequivalent toric `Z_2` actions.

    **Description:**

    Enumerates half-integer lattice points defining `Z_2` torus actions modulo the action of the supplied lattice symmetry group.

    **Arguments:**

    - `lattice_symmetries (array-like)`: List or array of square integer matrices acting from the left.

    **Returns:**

    - `numpy.ndarray`: Inequivalent integer representatives `q` such that `q/2` defines a `Z_2` action.

    """
    
    dim = lattice_symmetries[0].shape[0]
    t_possibilities = {t for t0 in combinations_with_replacement([0,1],dim) for t in permutations(t0)}
    t_possibilities = [t for t in t_possibilities]
    t_possibilities = np.delete(t_possibilities,t_possibilities.index(tuple([0]*dim)),0)

    inequivalent_t_possibilities = {frozenset([tuple(y) for y in x]) 
                                    for x in  np.transpose(np.array([np.mod(s@(t_possibilities.T),2) 
                                                                     for s in lattice_symmetries]),[2,0,1])}
    inequivalent_t_possibilities = np.array([[y for y in x][0] 
                                             for x in inequivalent_t_possibilities])
    
    return inequivalent_t_possibilities

    

def linebundle_weights_from_Newton_Polytope(vectors,Newton_polytope: Polytope):
    """

    Computes divisor coefficients from a Newton polytope.

    **Description:**

    Recovers the toric divisor coefficients whose Newton polytope is `Newton_polytope` by maximizing the corresponding inequalities over its lattice points.

    **Arguments:**

    - `vectors (numpy.ndarray)`: Rays of the toric fan.
    - `Newton_polytope (Polytope)`: The Newton polytope.

    **Returns:**

    - `numpy.ndarray`: The divisor coefficient vector.

    """
    return np.max(-(vectors@Newton_polytope.points().T),axis=1)

def is_Gorenstein(cone):
    """

    Checks whether a cone is Gorenstein.

    **Description:**

    Determines whether there exists an integral linear functional taking value one on all extremal rays of the cone, and returns that functional when it exists.

    **Arguments:**

    - `cone (Cone)`: The cone to test.

    **Returns:**

    - `tuple`: A pair `(is_gorenstein, n)`, where `n` is the Gorenstein functional if it exists, otherwise `None`.

    """
    
    M=cone.extremal_rays()
    ones=np.ones(M.shape[0],dtype=int)
    S,U,V=smith_normal_decomp(Matrix(M,domain=ZZ))
    s=np.array(S,dtype=int)
    u=np.array(U,dtype=int)
    v=np.array(V,dtype=int)
    c=u@ones
    y=np.zeros(s.shape[1],dtype=int)
    for ii in range(np.min(s.shape)):
        if s[ii][ii]!=0:
            y[ii]=c[ii]/s[ii][ii]
        else:
            return (False,None)
    n=v@y
    return (True,n)

def is_reflexive_Gorenstein(cone):
    """

    Checks whether a cone is reflexive Gorenstein.

    **Description:**

    Determines whether both the cone and its dual are Gorenstein.

    **Arguments:**

    - `cone (Cone)`: The cone to test.

    **Returns:**

    - `bool`: `True` if the cone is reflexive Gorenstein, otherwise `False`.

    """
    if is_Gorenstein(cone)[0]:
        dual_cone=cone.dual()
        if is_Gorenstein(dual_cone)[0]:
            return True
    return False

def Gorenstein_index(cone):
    """

    Computes the Gorenstein index of a reflexive Gorenstein cone.

    **Description:**

    Computes the pairing of the Gorenstein generators of a reflexive Gorenstein cone and its dual.

    **Arguments:**

    - `cone (Cone)`: The cone to test.

    **Raises:**

    - `ValueError`: Raised if the cone is not reflexive Gorenstein.

    **Returns:**

    - `int`: The Gorenstein index.

    """
    if is_reflexive_Gorenstein(cone):
        dual_cone=cone.dual()
        return is_Gorenstein(cone)[1]@is_Gorenstein(dual_cone)[1]
    raise ValueError("Cone is not reflexive Gorenstein")


def Cartier_index(toric_fan,weights):
    
    """

    Computes the Cartier index of a toric Weil divisor.

    **Description:**

    Computes the smallest positive integer multiple of a Q-Cartier toric Weil divisor that is Cartier. Returns `None` if the divisor is not Q-Cartier.

    **Arguments:**

    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `int` or `None`: The Cartier index, or `None` if the divisor is not Q-Cartier.

    """
    
    weights = np.array(weights)
    index_data=[]
    for c in toric_fan.cones():
        arr = toric_fan.vectors(c)
        cone_gen_weights= weights[np.array(c)-1]
        least_sq = np.linalg.lstsq(arr,cone_gen_weights)
        y=-least_sq[0]
        res = sum(least_sq[1])
        if res>1e-10:
            return None
        cone_index = 1
        while np.all(np.isclose(cone_index*y, np.round(cone_index*y)))==False:
            cone_index+=1
        index_data.append(cone_index)
    return np.lcm.reduce(np.array(index_data))
    

def is_Cartier(toric_fan,weights,return_Q_Cartier_data=False,decimals=10):
    """

    Checks whether a toric Weil divisor is Cartier.

    **Description:**

    Solves for local Cartier data on each maximal cone and determines whether all local data are integral. Optionally returns rational approximations to Q-Cartier data.

    **Arguments:**

    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    - `return_Q_Cartier_data (bool)`: Whether to return local data also for Q-Cartier divisors. Defaults to `False`.

    - `decimals (int)`: Number of decimals used when storing approximate Q-Cartier data. Defaults to `10`.

    **Returns:**

    - `tuple`: A pair `(is_cartier, cartier_data)`, where `cartier_data` is a list of local Cartier data or `None`.

    """
    weights = np.array(weights)
    cartier_data=[]
    is_cartier = True
    for c in toric_fan.cones():
        arr = toric_fan.vectors(c)
        cone_gen_weights= weights[np.array(c)-1]
        least_sq = np.linalg.lstsq(arr,cone_gen_weights)
        y=-least_sq[0]
        res = sum(least_sq[1])
        if np.all(np.isclose(y, np.round(y))) and res<1e-10:
            cartier_data.append(np.round(y).astype(int))
        else:
            is_cartier = False
            if return_Q_Cartier_data:
                if res<1e-10:
                    cartier_data.append(np.round(y,decimals=decimals))
                else:
                    cartier_data.append(None)
            else:
                return (False,None)
                
    return (is_cartier,cartier_data)

def is_nef(toric_fan,weights):
    """

    Checks whether a toric divisor is nef.

    **Description:**

    Tests whether the divisor coefficient vector lies in the nef cone by pairing with the secondary-cone hyperplanes.

    **Arguments:**

    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `bool`: `True` if the divisor is nef, otherwise `False`.

    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>=0)

def is_ample(toric_fan,weights):
    """

    Checks whether a toric divisor is ample.

    **Description:**

    Tests whether the divisor coefficient vector lies in the interior of the nef cone by strict pairing with the secondary-cone hyperplanes.

    **Arguments:**

    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `bool`: `True` if the divisor is ample, otherwise `False`.

    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>0)

def is_effective(points,weights):

    """

    Checks whether a toric divisor is effective.

    **Description:**

    Determines whether the Newton polytope of a divisor has at least one lattice point.

    **Arguments:**

    - `points (numpy.ndarray)`: Rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `bool`: `True` if the divisor has a nonzero section, otherwise `False`.

    """

    
    try:
        NP=Newton_Polytope(points,weights)
        if len(NP.points())>0:
            return True
        return False
    except ValueError:
        return False


def integer_kernel_basis(A):
    """

    Computes an integral kernel basis.

    **Description:**

    Computes a `Z`-basis for the integer kernel of the matrix `A` using Hermite normal form.

    **Arguments:**

    - `A (array-like)`: Integer matrix of shape `(m, n)`.

    **Returns:**

    - `numpy.ndarray`: Rows forming an integral basis of `ker(A : Z^n -> Z^m)`.

    WARNING: Written by chatGPT, but seems to work.
    """
    if isinstance(A, np.ndarray):
        A = A.tolist()

    AT = fmpz_mat(A).transpose()      
    H, T = AT.hnf(transform=True)     

    rank = sum(
        any(int(H[i, j]) != 0 for j in range(H.ncols()))
        for i in range(H.nrows())
    )

    basis = []
    for i in range(rank, T.nrows()):
        v = np.array([int(T[i, j]) for j in range(T.ncols())], dtype=object)
        basis.append(v)

    return np.array(basis).astype(int)



def LLL_wrapper(A):
    """

    LLL-reduces an integer matrix.

    **Description:**

    Applies FLINT's LLL reduction to the rows of an integer matrix and removes zero rows.

    **Arguments:**

    - `A (array-like)`: Integer matrix.

    **Returns:**

    - `numpy.ndarray`: The LLL-reduced nonzero rows.

    """
    
    if isinstance(A, np.ndarray):
        A = A.tolist()
    L = np.array(fmpz_mat(A).lll().tolist()).astype(int)
    zeropos = np.where(sum(abs(L.T))==0)[0]
    L = np.delete(L,zeropos,0)
    return L


def integer_rowspan_basis(A):
    """

    Computes an integral row-span basis.

    **Description:**

    Computes a `Z`-basis for the lattice generated by the rows of `A` using Hermite normal form.

    **Arguments:**

    - `A (array-like)`: Integer matrix.

    **Returns:**

    - `numpy.ndarray`: Rows forming a `Z`-basis of the row span.

    """
    if isinstance(A, np.ndarray):
        A = A.tolist()

    H = fmpz_mat(A).hnf()  
    H = np.array(H.tolist()).astype(int)
    zero_pos = np.where(sum(abs(H.T))==0)[0]
    basis = np.delete(H,zero_pos,0)

    return basis



def moving_cone(toric_variety):
    """

    Computes the moving cone of a toric variety.

    **Description:**

    Computes the moving cone from the GLSM charges by intersecting the cones obtained after deleting each toric ray.

    **Arguments:**

    - `toric_variety (Fan)`: The toric fan or toric variety object.

    **Returns:**

    - `Cone`: The moving cone.

    """
    
    rays = toric_variety.vectors()
    glsm = integer_kernel_basis(rays.T)
    h_planes = np.array([h for i in range(len(rays)) for h in Cone(np.delete(glsm.T,i,0)).hyperplanes()])@glsm
    mov = Cone(hyperplanes = h_planes)
    
    return mov

def generic_section_factorizes(points,linebundle_weights):
    """

    Checks whether a generic section factorizes.

    **Description:**

    Determines whether the generic section of a toric divisor factorizes by testing whether every toric coordinate appears nontrivially in some section.

    **Arguments:**

    - `points (array-like)`: Rays of the toric fan.

    - `linebundle_weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `bool`: `True` if the generic section factorizes, otherwise `False`.

    """
    
    try:
        NP=Newton_Polytope(points,linebundle_weights)
    except ValueError as e:
        return True
    return ~np.all(np.any(points@NP.points().T+linebundle_weights[:,None]!=0,axis=1))

def attempt_to_make_nef(toric_variety,line_bundle,epsilon=1e-5):

    """

    Attempts to find a triangulation where a divisor is nef.

    **Description:**

    Perturbs the triangulation heights in the direction of the given line bundle in order to find a triangulation for which the line bundle is nef.

    **Arguments:**

    - `toric_variety`: A triangulated vector configuration or toric fan with a vector configuration.
    - `line_bundle (array-like)`: Divisor coefficients in the toric prime divisor basis.
    - `epsilon (float)`: Magnitude of the perturbation used to obtain a triangulation rather than a subdivision. Defaults to `1e-5`.

    **Returns:**

    - `Fan`: A triangulation of the vector configuration.

    """

    line_bundle = np.array(line_bundle)
    hts0 = toric_variety.heights()+epsilon*np.array([(np.sin(i+1)+1) for i in range(len(line_bundle))])
    hts1 = line_bundle
    
    return toric_variety.vc.triangulate(heights=hts1/epsilon+hts0)

def basis(points):

    """

    Finds a linearly independent row basis.

    **Description:**

    Returns indices of rows of `points` forming a basis for the row span.

    **Arguments:**

    - `points (numpy.ndarray)`: Matrix whose rows are candidate basis vectors.

    **Raises:**

    - `ValueError`: Raised if no basis can be found.

    **Returns:**

    - `list`: Indices of basis rows.

    """
    
    n=points.shape[0]
    d=np.linalg.matrix_rank(points)
    basis_indices = []
    for i in range(n):
        test_indices = basis_indices + [i]
        if np.linalg.matrix_rank(points[test_indices]) == len(test_indices):
            basis_indices.append(i)
        if len(basis_indices) == d:
            return basis_indices
    raise ValueError("No basis could be found")


def sums_to_anticanonical(pts,L1,L2):
    """

    Checks whether two divisors sum to the anticanonical class.

    **Description:**

    Determines whether `L1 + L2` is linearly equivalent to the anticanonical divisor and, if so, returns the character implementing the equivalence.

    **Arguments:**

    - `pts (numpy.ndarray)`: Rays of the toric fan.
    - `L1 (array-like)`: First divisor coefficient vector.
    - `L2 (array-like)`: Second divisor coefficient vector.

    **Returns:**

    - `tuple`: A pair `(sums_to_anticanonical, character)`, where `character` is the linear-equivalence shift if it exists, otherwise `None`.

    """
    pts_float = np.array(pts, dtype=float)
    b_float = (1 - np.array(L1) - np.array(L2)).astype(float)
    
    try:
        x_float, residuals, rank, s = np.linalg.lstsq(pts_float, b_float, rcond=None)
        
        x_int = np.round(x_float).astype(int)
        
        pts_exact = np.array(pts, dtype=object)
        b_exact = 1 - np.array(L1) - np.array(L2)
        
        if np.array_equal(pts_exact @ x_int, b_exact):
            return True, x_int
            
    except np.linalg.LinAlgError:
        pass 
        
    return False, None

def is_partition(points, L1,L2):
    """

    Checks whether two divisors define a partition.

    **Description:**

    Determines whether two toric divisors can be shifted by principal divisors so that their coefficients are in `{0,1}` and their sum is the anticanonical divisor.

    **Arguments:**

    - `points (numpy.ndarray)`: Rays of the toric fan.
    - `L1 (array-like)`: First divisor coefficient vector.
    - `L2 (array-like)`: Second divisor coefficient vector.

    **Returns:**

    - `tuple`: A tuple `(is_partition, sums_to_anticanonical, shift_L1, shift_L2)`.

    """
    
    sta=sums_to_anticanonical(points,L1,L2)
    if sta[0]==False:
        return (False,False,np.zeros(points.shape[1],dtype=int),np.zeros(points.shape[1],dtype=int))
        
    basis_indices=basis(points)
    
    basis_vectors = points[basis_indices]
    basis_L2 = L2[basis_indices]
    possible_targets = [[-w, 1 - w] for w in basis_L2]
    if len(basis_indices)==points.shape[1]:
        for target_comb in product(*possible_targets):
            target_vec = np.array(target_comb)
            
            m2_float = np.linalg.solve(basis_vectors, target_vec)
            
            m2_int = np.round(m2_float).astype(int)
            if not np.allclose(m2_float, m2_int):
                continue
                
            dot_products = points @ m2_int
            
            valid_lower = (dot_products == -L2)
            valid_upper = (dot_products == 1 - L2)
            
            if np.all(valid_lower | valid_upper):
                return (True,True,sta[1]-m2_int,m2_int)
    else:
        basis_vecs_pseudo=basis_vectors@basis_vectors.T
        for target_comb in product(*possible_targets):
            target_vec = np.array(target_comb)
            
            m2_float = basis_vectors.T@np.linalg.solve(basis_vecs_pseudo, target_vec)
            
            m2_int = np.round(m2_float).astype(int)
            if not np.allclose(m2_float, m2_int):
                continue
                
            dot_products = points @ m2_int
            
            valid_lower = (dot_products == -L2)
            valid_upper = (dot_products == 1 - L2)
            
            if np.all(valid_lower | valid_upper):
                return (True,True,sta[1]-m2_int,m2_int)
            
    return (False,True,np.zeros(points.shape[1],dtype=int),sta[1])
    
def attempt_to_make_Cartier(tri,D):
    """

    Attempts to make a divisor Cartier by refining the fan.

    **Description:**

    If the divisor is not Cartier on the given fan, this function adds rays from the Newton-polytope inequalities and attempts to triangulate the refined configuration so that the divisor becomes Cartier and nef.

    **Arguments:**

    - `tri (Fan)`: The initial toric fan.
    - `D (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `tuple`: A pair `(new_fan, new_D)` consisting of the refined fan and the updated divisor coefficients.

    """
    if is_Cartier(tri,D)[0]:
        return (tri,D)
    else:
        NP=Newton_Polytope(tri.vectors(),D)
        inequalities = NP.inequalities()
        new_points = row_difference(inequalities[:,:-1],tri.vectors())
        indices_new_points = get_indices(inequalities[:,:-1],new_points)
        new_vectors = np.concatenate((tri.vectors(),new_points),axis=0)
        new_vc=VectorConfiguration(new_vectors)
        new_D=np.concatenate((D,inequalities[:,-1][indices_new_points]))
        tri_Cartier = attempt_to_make_nef(new_vc.triangulate(),new_D)
        return (tri_Cartier,new_D)


def BL(fan,lb):
    """

    Computes the base locus of a toric divisor.

    **Description:**

    Computes sections of the divisor with weights `lb` and returns the corresponding base locus in the given fan.

    **Arguments:**

    - `fan (Fan)`: The toric fan.
    - `lb (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `list`: Cones defining the base locus.

    """
    return base_locus(sections(fan.vectors(),lb),cones=fan.cones())
        
def base_locus(sections,cones=None,dim=4):
    """

    Computes the base locus from section exponents.

    **Description:**

    Computes minimal coordinate strata on which all sections vanish. If cones are provided, the search is restricted to strata of the corresponding toric fan.

    **Arguments:**

    - `sections (numpy.ndarray)`: Section exponent matrix.
    - `cones (iterable or None)`: Cones of the toric fan. If `None`, all coordinate strata up to dimension `dim` are considered. Defaults to `None`.
    - `dim (int)`: Dimension used when `cones=None`. Defaults to `4`.

    **Returns:**

    - `list`: Cones defining the base locus.

    """
    num_coords,num_sections=sections.shape
    B=sections > 0 
    minimal_hitting_sets=[]
    if type(cones)==type(None):
        for codim in range(1,dim + 1):
            for combo in combinations(range(num_coords), codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[list(combo), :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [tuple(x + 1 for x in mhs) for mhs in minimal_hitting_sets]
    else:
        for codim in range(1,len(cones[0])+1):
            for combo in get_lower_dimensional_cones(cones,codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[np.array(combo)-1, :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [mhs for mhs in minimal_hitting_sets]

def normal_fan(polytopes,inequalities=None,maximal_refinement=False,triangulate_refinement=False,return_unrefined_fan=False):

    """

    Computes the normal fan of a polytope or Minkowski sum.

    **Description:**

    Constructs the normal fan of a lattice polytope, or of the Minkowski sum of a list of lattice polytopes. Optionally constructs a maximal refinement subject to the specified inequalities.

    **Arguments:**

    - `polytopes (Polytope or list[Polytope])`: A lattice polytope, or a list of lattice polytopes whose Minkowski sum is used.
    - `inequalities (array-like or None)`: Inequality data used for maximal refinement. Required if `maximal_refinement=True`. Defaults to `None`.
    - `maximal_refinement (bool)`: Whether to construct the maximal refinement. Defaults to `False`.
    - `triangulate_refinement (bool)`: Whether to triangulate the refined vector configuration. Defaults to `False`.
    - `return_unrefined_fan (bool)`: Whether to also return the unrefined normal fan. Defaults to `False`.

    **Raises:**

    - `Exception`: Raised if `maximal_refinement=True` but no inequalities are provided.

    **Returns:**

    - `tuple`: The normal fan or refined vector data, together with line-bundle weights and optionally the unrefined normal fan.

    """
    
    if type(polytopes)==type([]):
        msum_vertices = nested_sum([p.vertices() for p in polytopes])
        p = Polytope(np.unique(flatten(msum_vertices,len(polytopes)-1),axis=0))
        vertex_split = np.array([np.array(np.where(np.all(np.array(msum_vertices)-v==0,axis=-1))).T[0] for v in p.vertices()])
    else:
        p = polytopes
        weights = p.inequalities().T[-1]

    hyperplane_saturations = [p.inequalities()[np.where(x==0)[0]] for x in (np.vstack([p.vertices().T,[1]*len(p.vertices())]).T@(p.inequalities().T))]
    normal_fan_edges = np.delete(p.inequalities().T,-1,0).T
    normal_fan_vc = VectorConfiguration(normal_fan_edges)
    cones = [[int(np.where(np.all(normal_fan_edges-x==0,axis=1))[0][0])+1 for x in np.delete(s.T,-1,0).T] for s in hyperplane_saturations]
    n_fan = Fan(vc=normal_fan_vc,cones=cones)

    if type(polytopes)==type([]):
        vertices_to_vertices_map = [vertex_split[np.where([i in c for c in cones])[0][0]] for i in range(1,len(normal_fan_edges)+1)]
        weights = np.array([-np.array([polytopes[j].vertices()[x] for j,x in enumerate(pointers)])@(normal_fan_edges[i]) for i,pointers in enumerate(vertices_to_vertices_map)])

    if not maximal_refinement:
        return (normal_fan_vc.triangulate(cells=cones),weights,cones)

    if type(inequalities)==type(None):
        raise Exception('Inequalities must be given to construct maximal refinement')

    inequalities=np.array(inequalities)
    
    n_vectors=n_fan.vectors()
    if np.max((1-weights[:,0])*inequalities[0]-weights[:,1])>=inequalities[-1]:
        if return_unrefined_fan:
            return (None,None,None)
        else:
            return (None,None)
    if np.min((1-weights[:,0])*inequalities[0]-weights[:,1])<0:
        if return_unrefined_fan:
            return (None,None,None)
        else:
            return (None,None)

    maximal_blow_ups = [h_polytope.HPolytope(np.vstack([[np.concatenate([np.delete(inequalities,-1,0)@np.array([polytopes[j].vertices()[x] 
                for j,x in enumerate(vertex_split[i])]),[inequalities[-1]]])],np.vstack([(p.vertices()-m).T, [0]*len(p.vertices())]).T ])).points() 
                        for i,m in enumerate(p.vertices())]

    maximal_blow_ups = [np.unique([np.rint(x/np.gcd.reduce(x)).astype(int) 
                                       for x in np.delete(b,np.where(np.all(b==0,axis=1))[0][0],0)],axis=0) 
                            for b in maximal_blow_ups]

    all_vectors = np.unique(np.array([y for x in maximal_blow_ups for y in x]),axis=0)
    all_weights = np.array([-np.array([(pol.vertices()[vertex_split[np.where([np.any(np.all(y-x==0,axis=-1)) for y in maximal_blow_ups])[0][0]][j]])@x for x in all_vectors]) 
                        for j,pol in enumerate(polytopes)])

    old_indices = np.where([type(n_fan.vc.vectors_to_labels(v))!=type(None) for v in all_vectors])[0]
    blow_up_weights = np.delete(all_weights.T,old_indices,0)
    blow_up_vectors = np.delete(all_vectors,old_indices,0)
    
    all_vectors = np.vstack([n_fan.vectors(),blow_up_vectors])
    all_weights = np.vstack([weights,blow_up_weights])

    if not triangulate_refinement:
        if return_unrefined_fan:
            return (all_vectors,all_weights,n_fan)
        else:
            return (all_vectors,all_weights)

        
    if return_unrefined_fan:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights,n_fan)
    else:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights)




def nested_sum(lists, depth=0, acc=0):

    """

    Computes nested sums of entries from several lists.

    **Description:**

    Recursively forms all sums obtained by choosing one element from each list in `lists`.

    **Arguments:**

    - `lists (list)`: A list of lists or arrays whose elements are to be summed.
    - `depth (int)`: Recursion depth. Defaults to `0`.
    - `acc`: Accumulated partial sum. Defaults to `0`.

    **Returns:**

    - `list`: The nested list of sums.

    """
    
    if depth == len(lists):
        return acc
    return [
        nested_sum(lists, depth + 1, acc + x)
        for x in lists[depth]
    ]
def flatten(lst, depth):

    """

    Flattens a nested list to a specified depth.

    **Description:**

    Recursively flattens a nested list by the specified number of levels.

    **Arguments:**

    - `lst (list)`: The nested list to flatten.
    - `depth (int)`: Number of nesting levels to flatten.

    **Returns:**

    - `list`: The flattened list.

    """
    
    if depth == 0:
        return lst
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(flatten(x, depth - 1))
        else:
            result.append(x)
    return result

def O7_cones(vc_orbifold,O7_labels,d):
    """

    Returns cones supported on O7 divisors.

    **Description:**

    Computes the `d`-ray cones of a toric fan whose rays are all contained in the set of O7-plane labels.

    **Arguments:**

    - `vc_orbifold (Fan)`: The toric fan of the orbifold.
    - `O7_labels (array-like)`: One-indexed labels of the O7 divisors.
    - `d (int)`: Number of rays in the cones to consider.

    **Returns:**

    - `list`: Cones whose rays are contained in `O7_labels`.

    """
    d_cones= get_lower_dimensional_cones(vc_orbifold.cones(),d)
    relevant_d_cones = [t for t in d_cones if set(t).issubset(O7_labels)]
    return relevant_d_cones


def basis_H2_toric_fan(toric_fan):
    """

    Computes a toric curve-homology basis.

    **Description:**

    Finds a smooth maximal cone and returns the complementary ray labels, giving a convenient GLSM or curve-homology basis.

    **Arguments:**

    - `toric_fan (Fan)`: The toric fan.

    **Raises:**

    - `ValueError`: Raised if no smooth maximal cone is found.

    **Returns:**

    - `numpy.ndarray`: One-indexed ray labels forming the basis.

    """
    for c in toric_fan.cones():
        if Cone(toric_fan.vectors(c)).is_smooth():
            mask = np.ones(len(toric_fan.vectors()),dtype=bool)
            mask[np.array(c)-1]=False
            basis = np.arange(1,len(mask)+1)[mask]
            return basis
    raise ValueError("No basis could be found")

def trilayer_5d_Ftheory_uplift(p,verbosity=1):

    """

    Computes a five-dimensional F-theory uplift polytope from a trilayer polytope.

    **Description:**

    Constructs the five-dimensional polytope associated with the F-theory uplift of a trilayer orientifold in the limit where all mid-layer divisors are blown down.

    **Arguments:**

    - `p (Polytope)`: The reflexive trilayer polytope.
    - `verbosity (int)`: Verbosity level controlling printed blowdown information. Defaults to `1`.

    **Raises:**

    - `Exception`: Raised if `p` is not trilayer.

    **Returns:**

    - `Polytope`: The five-dimensional F-theory uplift polytope.

    """
    
    if not p.is_trilayer():
        raise Exception("Polytope is not trilayer.")
        
    p = trilayer_normal_form(p)
    mid_layer_points = np.where(p.points().T[0]==0)[0]
    mid_layer_points = np.array([i for i in set(mid_layer_points).intersection(set(p.points_to_indices(p.points_not_interior_to_facets())))])
    if len(mid_layer_points)>1 and verbosity>0:
        print(f"{len(mid_layer_points)-1} exceptional divisors have been blown down.")
    
    p_dim = p.dim()

    p3 = Polytope(np.delete(p.points()[np.where(p.points().T[0]==1)[0]].T,0,0).T)

    p2KB = Newton_Polytope(p3.points(),len(p3.points())*[2])

    n_fan,wts,cns = normal_fan(p2KB)

    cone_hyperplanes = [Cone(n_fan.vectors(c)).dual().rays() for c in cns]
    blown_up = [h_polytope.HPolytope(np.vstack([np.vstack([h.T,[0]*len(h)]).T,[np.concatenate([p2KB.vertices()[i],[2]]),np.concatenate([-p2KB.vertices()[i],[-1]])]])).points()  
                for i,h in enumerate(cone_hyperplanes)]
    blown_up = [[np.rint(x/np.gcd.reduce(x)).astype(int) for x in b] for b in blown_up]
    blown_up = [np.unique(b,axis=0) for b in blown_up]
    all_vecs = np.unique([i for b in blown_up for i in b],axis=0)
    full_vc = VectorConfiguration(all_vecs)

    monomials = Newton_Polytope(full_vc.vectors(),[2]*len(full_vc.vectors())).points()@(full_vc.vectors().T)+np.array([2]*len(full_vc.vectors()))
    O7pos = np.where(np.array([min(m) for m in monomials.T])==1)[0]

    vx = np.concatenate([[0]*(p_dim-1),[3,1]])
    vy = np.concatenate([[0]*(p_dim-1),[-2,-1]])
    vz = np.concatenate([[0]*(p_dim-1),[0,1]])
    uplift_vecs_singular = np.vstack([np.vstack([full_vc.vectors().T,[[0]*len(full_vc.vectors()),[1]*len(full_vc.vectors())]]).T,[vx,vy,vz]])

    D4res1 = uplift_vecs_singular[O7pos]+vx+2*vy
    D4res2 = 2*uplift_vecs_singular[O7pos]+2*vx+3*vy
    uplift_vecs = np.vstack([uplift_vecs_singular,D4res1,D4res2])

    p5 = Polytope(uplift_vecs)
    
    return p5

def sections(points,weights):
    """

    Computes the sections of a toric divisor.

    **Description:**

    Computes the exponent vectors of all monomial sections of the divisor with coefficient vector `weights` on the fan with rays `points`.

    **Arguments:**

    - `points (numpy.ndarray)`: Rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.

    **Returns:**

    - `numpy.ndarray`: Matrix of section exponents, or an empty array if there are no sections.

    """
    NP=Newton_Polytope(points,weights)
    if len(NP.points())==0:
        return np.array([])
    else:
        return points@NP.points().T+weights[:,None]

def solve_over_integers(M,b):
    """

    Solves a linear system over the integers.

    **Description:**

    Uses Smith normal form to determine whether the equation `Mx=b` has an integral solution, and returns one if it exists.

    **Arguments:**

    - `M (array-like)`: Integer matrix.
    - `b (array-like)`: Integer inhomogeneous term.

    **Returns:**

    - `tuple`: A pair `(has_solution, x)`, where `x` is an integral solution if one exists, otherwise `None`.

    """
    A,S,T = smith_normal_decomp(Matrix(M),domain=ZZ)
    a=np.array(A,dtype=int)
    s=np.array(S,dtype=int)
    t=np.array(T,dtype=int)
    c=-s@b
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(np.min(a.shape)):
        if a[ii,ii]!=0:
            y[ii]=c[ii]/a[ii,ii]
        else:
            if c[ii]!=0:
                print("PROBLEM")
    if np.all(a@y==c):
        return (True,t@y)
    else:
        return (False,None)

def make_simplicial(fan):

    """

    Refines non-simplicial cones of a fan.

    **Description:**

    Replaces non-simplicial cones by cones obtained from a fine triangulation of the corresponding vector configuration, leaving simplicial cones unchanged.

    **Arguments:**

    - `fan (Fan)`: The toric fan to refine.

    **Returns:**

    - `Fan`: A simplicial refinement of the fan.

    """
    
    new_cones=set(fan.cones())
    dim=fan.dim
    for c in fan.cones():
        if len(c)>dim:
            new_cones.discard(c)
            carr=np.array(c)
            vc_tri=VectorConfiguration(fan.vectors(c)).triangulate(make_fine=True)
            for co in vc_tri.cones():
                new_cones.add(tuple(carr[np.array(co)-1]))
    return Fan(vc=fan.vc,cones=new_cones)

def refine_fan(fan,blowups_or_all_vectors=None):

    """

    Refines a toric fan by inserting rays.

    **Description:**

    Adds new rays to a fan and star-subdivides the cones containing them. If no new vectors are given, the vector configuration of the fan is used to detect rays not already present in the fan.

    **Arguments:**

    - `fan (Fan)`: The toric fan to refine.
    - `blowups_or_all_vectors (numpy.ndarray or None)`: Blowup rays or a full vector configuration containing the old rays and new rays. Defaults to `None`.

    **Returns:**

    - `Fan`: The refined toric fan.

    """
    
    if blowups_or_all_vectors is not None:
        blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
        all_vectors=np.concatenate((fan.vc.vectors(),blowups),axis=0)
        vc_all=VectorConfiguration(all_vectors)
        new_fan=Fan(vc_all,cones=fan.cones())
    else: 
        new_fan=fan
    vec_diff=row_difference(new_fan.vc.vectors(),new_fan.vectors())
    to_be_refined = set(new_fan.vc.vectors_to_labels(vec_diff))
    if fan.dim==len(new_fan.vc.vectors()[0]):
        for label in to_be_refined:
            all_cones=set(new_fan.cones())
            link_base=tuple(find_cone(new_fan.vc.vectors(label),all_cones,new_fan.vc.vectors()))
            link_base_len=len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan=Fan(vc=new_fan.vc,cones=all_cones)
        return new_fan
    else:
        for label in to_be_refined:
            all_cones=set(new_fan.cones())
            link_base=tuple(find_cone_general(new_fan.vc.vectors(label),all_cones,all_vectors))
            link_base_len=len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan=Fan(vc=new_fan.vc,cones=all_cones)
        return new_fan

def find_cone_general(new_ray, current_cones, all_vectors):
    """

    Finds the carrier face of a new ray by nonnegative least squares.

    **Description:**

    Searches the current cones for a minimal set of one-indexed ray labels whose strictly positive linear combination gives `new_ray`.

    **Arguments:**

    - `new_ray (numpy.ndarray)`: The ray to locate.
    - `current_cones (iterable)`: Current cones, represented by tuples of one-indexed ray labels.
    - `all_vectors (numpy.ndarray)`: Full ray matrix.

    **Returns:**

    - `frozenset` or `None`: The carrier face labels, or `None` if no carrier cone is found.

    """
    for cone in current_cones:
        cone_list = list(cone)
        numpy_indices = [idx - 1 for idx in cone_list]
        
        A = all_vectors[numpy_indices].T
        
        x, residual = nnls(A, new_ray)
        
        if residual < 1e-10:
            carrier_face = frozenset(
                cone_list[i] for i, coeff in enumerate(x) if coeff > 1e-10
            )
            return carrier_face
            
    return None

def array_to_latex(arr):
    """

    Converts a NumPy array to LaTeX matrix form.

    **Description:**

    Converts a two-dimensional NumPy array into a LaTeX `pmatrix` string.

    **Arguments:**

    - `arr (numpy.ndarray)`: The two-dimensional array to convert.

    **Raises:**

    - `ValueError`: Raised if `arr` is not two-dimensional.

    **Returns:**

    - `str`: A LaTeX `pmatrix` representation of the array.

    """
    if len(arr.shape) > 2:
        raise ValueError("Only 2D matrices are supported.")
    
    lines = [
        "  " + " & ".join(map(str, row)) + " \\\\"
        for row in arr
    ]
    return "\\begin{pmatrix}\n" + "\n".join(lines) + "\n\\end{pmatrix}"

def lattice_index(mat):
    """

    Computes the lattice index of a row-generated sublattice.

    **Description:**

    Uses Smith normal form to compute the index of the sublattice generated by the rows of an integer matrix.

    **Arguments:**

    - `mat (array-like)`: Integer matrix whose rows generate the sublattice.

    **Returns:**

    - `int`: The lattice index.

    """
    A=Matrix(mat,domain=ZZ)
    snf = smith_normal_form(A)
    
    s=np.array(snf,dtype=int)
    l_ind=np.prod([s[i][i] for i in range(np.min(s.shape))])
    
    return l_ind

def integral_gale_transform(points):
    """

    Computes the integral Gale transform of a point configuration.

    **Description:**

    Lifts the input points by appending a column of ones, computes the rational nullspace exactly using SymPy, and clears denominators to obtain an integral Gale transform.

    **Arguments:**

    - `points (array-like)`: Point configuration with shape `(n, d)`.

    **Raises:**

    - `ValueError`: Raised if the number of points is not greater than `d+1`.

    **Returns:**

    - `numpy.ndarray`: The integral Gale transform.

    """
    points = np.array(points)
    n, d = points.shape

    if n <= d + 1:
        raise ValueError(f"Need strictly more points (n={n}) than dimensions + 1 (d+1={d+1}).")

    lifted_points = np.hstack((points, np.ones((n, 1))))
    
    A = Matrix(lifted_points.T)
    
    null_basis_vectors = A.nullspace()
    
    if not null_basis_vectors:
        return np.array([])
        
    B = Matrix.hstack(*null_basis_vectors)
    
    for j in range(B.cols):
        LCM = lcm([fraction(B[i, j])[1] for i in range(B.rows)])
        B[:, j] = B[:, j] * LCM
        
    gale_points = np.array(B.T).astype(int)
    
    return gale_points

def find_cone(new_ray, current_cones, all_vectors, tol=1e-10):
    """

    Computes the integral Gale transform of a point configuration.

    **Description:**

    Lifts the input points by appending a column of ones, computes the rational nullspace exactly using SymPy, and clears denominators to obtain an integral Gale transform.

    **Arguments:**

    - `points (array-like)`: Point configuration with shape `(n, d)`.

    **Raises:**

    - `ValueError`: Raised if the number of points is not greater than `d+1`.

    **Returns:**

    - `numpy.ndarray`: The integral Gale transform.

    """
    for cone in current_cones:
        numpy_indices = np.array(cone) - 1

        A = all_vectors[numpy_indices].T

        coeffs = np.linalg.solve(A, new_ray)

        if np.all(coeffs >= -tol):
            return frozenset(
                idx for idx, coeff in zip(cone, coeffs)
                if coeff > tol
            )

    return None

def primitive_rows(A):
    """

    Makes integer matrix rows primitive.

    **Description:**

    Divides each row of an integer matrix by the greatest common divisor of its entries.

    **Arguments:**

    - `A (array-like)`: Integer matrix.

    **Returns:**

    - `numpy.ndarray`: The row-primitive integer matrix.

    """
    A = np.asarray(A, dtype=int)

    row_gcds = np.gcd.reduce(np.abs(A), axis=1, keepdims=True)
    row_gcds[row_gcds == 0] = 1  

    return A // row_gcds


def divisor_intersections(fan, intersection_dict,divisors, basis_set,as_LLL=True):

    """

    Computes divisor-intersection curve classes.

    **Description:**

    Computes the curve classes obtained by intersecting a list of divisors with toric strata, expressed in a chosen basis of curve homology.

    **Arguments:**

    - `fan (Fan)`: The toric fan.
    - `intersection_dict (dict)`: Dictionary of toric intersection numbers.
    - `divisors (list)`: List of divisor coefficient vectors.
    - `basis_set (set)`: Set of one-indexed ray labels used as the homology basis.
    - `as_LLL (bool)`: Whether to LLL-reduce the resulting lattice basis. Defaults to `True`.

    **Returns:**

    - `numpy.ndarray`: The divisor-intersection curve classes, optionally LLL-reduced.

    """
            
    codim_cicy=len(divisors)
    simplices = get_lower_dimensional_cones(fan.cones(), fan.dim - codim_cicy-1)
    divisor_nonvanishing_sets = []
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div != 0)[0] + 1))
    
    curves_homology_in_basis = np.zeros((len(simplices), len(basis_set)), dtype=int)
    
    basis_idx_map = {b: idx for idx, b in enumerate(basis_set)}

    for s_idx, s in enumerate(simplices):
        star_s=fan.star(s)
        link_rays = {item for sub_tuple in star_s for item in sub_tuple}
        valid_intersections = []
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(link_rays))
        valid_i = basis_set.intersection(link_rays)
        
        if not (all(valid_intersections) and valid_i):
            continue
        for i in valid_i:
            i_idx = basis_idx_map.get(i)
            total_intersection = 0
            
            for ts in product(*valid_intersections):
                key = tuple(sorted(s + ts+ (i,)))
                coefficient = np.prod([divisors[a][ray - 1]for a, ray in enumerate(ts)])
                total_intersection += coefficient * intersection_dict.get(key, 0)
            
            curves_homology_in_basis[s_idx, i_idx] = total_intersection

    if as_LLL:
        return LLL_wrapper(curves_homology_in_basis)
    return curves_homology_in_basis