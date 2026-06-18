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
from cytools import Polytope, h_polytope, Cone, fetch_polytopes
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration


def compute_partition(divisors,rays):

    """
    **Description:**
    Uses linear equivalence to bring a set of divisors in a toric variety into the form of a partition of the anti-canonical, if it exists.
    
    **Arguments:**
    - `divisors` *(np array or list)*: The input divisors
    - `rays` *(np array)*: The primitive generators of the one-skeleton of the toric fan.
    **Returns:**
    (Boolean: whether partition exists, np array or None: output divisors if they exist, otherwise None) 
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
    **Description:**
    Computes whether a 1d np array is the row of a 2d np array 
    
    **Arguments:**
    - `arr` *(np array)*: The 2d np array
    - `target` *(np array)*: The 1d np array
    **Returns:**
    Boolean: target in arr
    """
    return np.any(np.all(arr == target, axis=1))

def contains_rows(arr: np.array,targets: np.array):
    """
    **Description:**
    Computes whether all rows of a 2d np array targets are rows of a 2d np array arr
    
    **Arguments:**
    - `arr` *(np array)*: The 2d np array
    - `targets` *(np array)*: The 2d np array targets

    **Returns:**
    Boolean: All targets in arr
    """
    return np.any((targets[:, None, :] == arr[None, :, :]).all(axis=2), axis=1).all()

def get_same_rows(A: np.array, B: np.array):
    """
    **Description:**
    Computes the rows of A that are also in B
    
    **Arguments:**
    - `A` *(np array)*: 2d np.array
    - `B` *(np array)*: 2d np.array

    **Returns:**
    2d np.array: Rows that are both in A and B
    """
    return A[np.where((A[:, None, :] == B[None, :, :]).all(axis=2))[0]]

def same_rows(A, B):
    """
    **Description:**
    Computes wether A and B have the same rows
    
    **Arguments:**
    - `A` *(np array)*: 2d np.array
    - `B` *(np array)*: 2d np.array

    **Returns:**
    Bool: A and B have the same rows
    """
    rowsA, countsA = np.unique(A, axis=0, return_counts=True)
    rowsB, countsB = np.unique(B, axis=0, return_counts=True)
    return np.array_equal(rowsA, rowsB) and np.array_equal(countsA, countsB)

def dual_face_Cayley_polytope(Cdvert: np.array,f):
    """
    **Description:**
    Computes the dual face of a face f of a Cayley polytope
    
    **Arguments:**
    - `Cdvert` *(np array)*: Vertices of the dual Cayley polytope

    **Returns:**
    Polytope: Dual face
    """
    return Polytope(Cdvert[np.all(Cdvert@f.vertices().T == 0, axis=1)])
    
def h11_2_part(Cay: Polytope,Cayd: Polytope,det=False):
    """
    **Description:**
    Computes the Hodge number h11 of a CICY realized as a 2-part nef partition
    
    **Arguments:**
    - `Cay` *Polytope*: Cayley polytope
    - `Cayd` *Polytope*: Dual Cayley polytope

    **Returns:**
    int: h11 of the CICY
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
    **Description:**
    Computes the Hodge number h21 of a CICY realized as a 2-part nef partition in a 6d variety
    
    **Arguments:**
    - `Cay` *Polytope*: Cayley polytope
    - `Cayd` *Polytope*: Dual Cayley polytope

    **Returns:**
    int: h21 of the CICY
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
    **Description:**
    Returns the indices of the target rows inside arr
    
    **Arguments:**
    - `arr` *np array*: 2d array
    - `targets` *np array*: array or targets

    **Returns:**
    array: indices of targets in arr
    """
    return  np.where(np.any((arr[:, None, :] == targets).all(axis=2), axis=1))[0]

def get_index(arr: np.array,target: np.array):
    """
    **Description:**
    Returns the index of the target array inside arr
    
    **Arguments:**
    - `arr` *np array*: 2d array
    - `target` *np array*: 1d np array, the target

    **Returns:**
    int: index of target in arr
    """
    return  np.where(np.all(arr == target, axis=1))[0]

def glsm_from_points(pts):
    """
    **Description:**
    Computes the glsm of a set of points
    
    **Arguments:**
    - `pts` *np array*: List of points

    **Returns:**
    np array: The GLSM
    """
    a,s,t=smith_normal_decomp(Matrix(np.array(pts).T),domain=ZZ)
    aa=np.array(a,dtype=int)
    ss=np.array(s,dtype=int)
    tt=np.array(t,dtype=int)
    rank_a=np.linalg.matrix_rank(aa)
    return tt.T[rank_a:]

def points_from_glsm(glsm):
    """
    **Description:**
    Computes a set of points corresponding to a glsm charge matrix
    
    **Arguments:**
    - `glsm` *np array*: GLSM matrix

    **Returns:**
    np array: The points
    """
    D,U,V=smith_normal_decomp(Matrix(np.array(glsm).T),domain=ZZ)
    DD=np.array(D,dtype=int)
    UU=np.array(U,dtype=int)
    VV=np.array(V,dtype=int)
    rank_D=np.linalg.matrix_rank(DD)
    return (UU[rank_D:].astype(int)).T

def find_trilayer_vertex_polytope(p,as_index=False):
    """
    **Description:**
    Finds the special vertex of a trilayer polytope
    
    **Arguments:**
    - `p` *Polytope*: Trilayer polytope
    - `as_index` (optional): If true, returns the index of the vertex in the polytope vertices

    **Returns:**
    np array: The points
    """
    glsm_vert=glsm_from_points(p.vertices())
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_index:
        return index
    else:
        return p.vertices()[index]

def find_trilayer_vertex_vertices(V,as_vertex_index=False):
    glsm_vert=glsm_from_points(V)
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_vertex_index:
        return index
    else:
        return V[index]

def glsm_Weierstrass(glsm,line_bundle_weights):
    """
    Deprecated
    """
    xyz_weights=np.zeros((glsm.shape[0],3))
    for i in range(len(glsm)):
        sum_charges=np.sum(glsm[i])
        for j in range(len(line_bundle_weights)):
            sum_charges=sum_charges-line_bundle_weights[j]*glsm[i,j]
        xyz_weights[i,0]=2*sum_charges
        xyz_weights[i,1]=3*sum_charges
    glsm_6=np.concatenate((glsm,xyz_weights),axis=1)
    glsm_fiber=np.zeros((1,glsm_6.shape[1]))
    glsm_fiber[0][-3]=2
    glsm_fiber[0][-2]=3
    glsm_fiber[0][-1]=1
    return np.concatenate((glsm_6,glsm_fiber),axis=0).astype(int)

def is_trilayer(p,get_index=False):
    """
    Deprecated
    """
    glsm_vert=glsm_from_points(p.vertices())
    anticanon = np.sum(glsm_vert, axis=1)
    is_int = all(c%2==0 for c in anticanon)
    is_tri = False
    if is_int:
        half_anticanon = anticanon//2
        index=get_indices(glsm_vert.T,np.array([half_anticanon]))
        is_tri = any(all((v == half_anticanon).flat) for v in glsm_vert.T)
    if get_index:
        return is_tri,index
    return is_tri    
        
def trilayer_normal_form(p):
    """
    Computes trilayer normal form of a polytope p
    
    - `p`: *Polytope*: Trilayer polytope

    **Returns:**
    Polytope: Trilayer normal form of p
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
    Computes the Newton Polytope of a divisor with weights weights in a toric fan with rays in pts
    
    - `pts`: *np.array*: Matrix of points of a fan
    - `weights`: *np.array*: Weights of a divisor

    **Returns:**
    Polytope: The Newton polytope of D
    """
    return h_polytope.HPolytope(np.column_stack([pts, weights]).astype(int))

def row_difference(A: np.array, B: np.array):
    """
    Returns all rows of A that are not in B
    - `A`: *np.array*: Numpy matrix A
    - `B`: *np.array*: Numpy matrix B

    **Returns:**
    np Matrix: Rows in A that are not in B
    """
    return A[~np.any((A[:, None, :] == B[None, :, :]).all(axis=2), axis=1)]

def points_not_interior_to_facets_and_codim2_faces(p: Polytope):
    """
    **Description:**
    Computes the points not interior to facets or codim-2 faces of a polytope

    **Arguments:**
    - `p` *Polytope*: The polytope

    **Returns:**
    np array: List of points
    """
    pts=p.points_not_interior_to_facets()
    for f in p.faces(p.dim()-2):
        if len(f.interior_points())>0:
            pts=np.delete(pts,get_indices(pts,f.interior_points()),axis=0)
    return pts
    
def rotate_points(pts,pts6):
    """
    Deprecated
    """
    dimB=pts.shape[1]
    B1=np.zeros((dimB,dimB))
    ct=0
    for i in range(len(pts)):
        B1[ct]=pts[i]
        if np.linalg.matrix_rank(B1)==ct+1:
            ct=ct+1
        if ct==dimB:
            break
    B2=np.zeros((dimB,dimB))
    ct=0
    for i in range(len(pts6)):
        B2[ct]=pts6[i][:dimB]
        if np.linalg.matrix_rank(B2)==ct+1:
            ct=ct+1
        if ct==dimB:
            break
    M6=np.block([[np.round(B1.T@np.linalg.inv(B2.T)).astype(int),np.zeros((dimB,2),dtype=int)],[np.zeros((2,dimB),dtype=int),np.eye(pts6.shape[1]-dimB)]]).astype(int)
    return (M6@pts6.T).T
def glsm_uplift_toric_base(glsm):
    """
    Deprecated
    """
    W3=np.zeros((glsm.shape[0],3),dtype=int)
    for i in range(glsm.shape[0]):
        s=np.sum(glsm[i])
        W3[i][0]=2*s
        W3[i][1]=3*s
    W231=np.zeros((1,glsm.shape[1]+3),dtype=int)
    W231[0][-3]=2
    W231[0][-2]=3
    W231[0][-1]=1
    glsm5=np.concatenate((glsm,W3),axis=1)
    return np.concatenate((glsm5,W231),axis=0)
def get_lower_dimensional_cones(cones,d):
    """
    **Description:**
    Computes the lower dimensional cones of a toric fan

    **Arguments:**
    - `cones` *(list or tuple of tuples)*: The set/list/tuple of cones
    - `d` *int*: dimension of the lower dimensional cone

    **Returns:**
    A list of all d-dimensional cones
    """
    return list({combo for row in cones for combo in combinations(row, d)})

def lattice_refinement(q, denominator = 2):

    """
    **Description:**
    Returns smallest embedding of unit lattice into larger lattice such that q/denominator is integral.

    **Arguments:**
    - `q` *(list of integers or numpy array)*: scaled up input vector 
    - `denominator` *(int, optional, default=2)*: denominator of q/denominator

    **Returns:**
    A lattice map.
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
    **Description:**
    Constructs toric orbifold defined via lattice refinement q/denominator

    **Arguments:**
    - `vc_triangulation` *(triangulation of vector configuration)*: a simplicial toric variety
    - `q` *(list of integers or numpy array)*: scaled up co-prime lattice refinement vector
    - `denominator` *(int, optional, default=2)*: denominator of q/denominator

    **Returns:**
    tuple of (toric variety, array of rescalings of edges of toric fan)
    """
            
    Lambda = lattice_refinement(q,denominator)
    orbifold_points = pts_CY_ambient@(Lambda.T)
    rescalings = np.array([math.gcd(*list(i)) for i in orbifold_points])
    orbifold_points = np.rint((orbifold_points.T/rescalings).T).astype(int)
    return (orbifold_points,rescalings)

def O3O7_line_bundle(pts_CY_ambient,q,rescalings):

    """
    **Description:**
    Computes the line bundle weigths associated with O3/O7 type hypersurface

    **Arguments:**
    - `vc_triangulation` *(triangulation of vector configuration)*: a simplicial toric variety
    - `q` *(list of integers or numpy array)*: scaled up co-prime lattice refinement vector
    - `rescalings` *(list of integers or numpy array)*: rescalings of divisor classes in toric orbifold

    **Returns:**
    numpy array of line bundle weights
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
    **Description:**
    Computes the fixed locus of a toric orbifold defined by the lattice refinement q/denominator

    **Arguments:**
    - `vc_triangulation` *(triangulation of vector configuration)*: a simplicial toric variety
    - `q` *(list of integers or numpy array)*: scaled up co-prime lattice refinement vector
    - `cone_dimension` *(int or None, optional, default=None)*: only considers subvarieties of this codimension
    - `denominator` *(int, optional, default=2)*: denominator of q/denominator

    **Returns:**
    list of cones, each cone defined as tuple of indices. Each index represents an edge of the toric fan.
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
    **Description:**
    Enumerates all conjugacy classes of Z2 subgroups of algebraic torus, parameterized by half-integer lattice points, given some set of lattice symmetries

    **Arguments:**
    - `lattice_symmetries` *(list or numpy array of square integer matrices)*: lattice symmetries, acting from the left

    **Returns:**
    numpy array of inequivalent integer lattice points q such that q/2 generates a Z2 symmetry.
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

    
def get_fixed_loci(TF,q):
    # DEPRECATED
    for i in range(1,1+TF.dim):
        for cone in get_lower_dimensional_cones(TF.cones(),i):
            if np.gcd.reduce(np.sum(TF.vectors()[np.array(cone)-1],axis=0)+q)==2:
                return i
    return None

def linebundle_weights_from_Newton_Polytope(vectors,Newton_polytope: Polytope):
    """
    **Description:**
    Computes the weights of a linebundle given the Newton Polytope
    
    **Arguments:**
    - `vectors` *(np array)*: The vectors of the toric fan
    - `Newton_polytope` *(Polytope)*: The Newton Polytope

    **Returns:**
    np array: (boolean: Is reflexive Gorenstein, np array: n vector)
    """
    return np.max(-(vectors@Newton_polytope.points().T),axis=1)

def is_Gorenstein(cone):
    """
    **Description:**
    Computes whether a cone is Gorenstein and if so determines the vector n
    
    **Arguments:**
    - `cone` *(Cone)*: The cone in consideration

    **Returns:**
    Tuple: (boolean: Is reflexive Gorenstein, np array: n vector)
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
    **Description:**
    Computes whether a cone is reflexive Gorenstein
    
    **Arguments:**
    - `cone` *(Cone)*: The cone in consideration

    **Returns:**
    Boolean: Is reflexive Gorenstein
    """
    if is_Gorenstein(cone)[0]:
        dual_cone=cone.dual()
        if is_Gorenstein(dual_cone)[0]:
            return True
    return False

def Gorenstein_index(cone):
    """
    **Description:**
    Computes the Gorenstein Index for a Gorenstein cone
    
    **Arguments:**
    - `cone` *(Cone)*: The cone in consideration

    **Returns:**
    Integer: Gorenstein Index
    """
    if is_reflexive_Gorenstein(cone):
        dual_cone=cone.dual()
        return is_Gorenstein(cone)[1]@is_Gorenstein(dual_cone)[1]
    raise ValueError("Cone is not reflexive Gorenstein")


def Cartier_index(toric_fan,weights):
    
    """
    **Description:**
    Computes Cartier index of a Weil divisor for a toric variety, if divisor is Q-Cartier. Returns None if divisor is not Q-Cartier.
    
    **Arguments:**
    - `toric_fan` *(triangulation of vector configuration)*: toric fan
    - `weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors

    **Returns:**
    integer Cartier index, or None.
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
     **Description**
    Determines whether a divisor is Cartier and computes the Cartier data

     **Arguments:**
    - `toric_fan` *(vector triangulation)*: toric fan
    - `weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors
    - `return_Q_Cartier_data` *(optional)*: if True, returns the Cartier data also for Q-Cartier divisors

    **Returns:**
    Tuple (boolean: Is Cartier, list of np arrays: Cartier Data)
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
     **Description**
    Determines whether a divisor is nef

     **Arguments:**
    - `toric_fan` *(vector triangulation)*: toric fan
    - `weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors

    **Returns:**
    Boolean: the Weil divisor is nef
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>=0)

def is_ample(toric_fan,weights):
    """
    **Description**
    Determines whether a divisor is ample

     **Arguments:**
    - `toric_fan` *(vector triangulation)*: toric fan
    - `weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors

    **Returns:**
    Boolean: the Weil divisor is ample
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>0)

def is_effective(points,weights):
    try:
        NP=Newton_Polytope(points,weights)
        if len(NP.points())>0:
            return True
        return False
    except ValueError:
        return False


def integer_kernel_basis(A):
    """
    Return an integral basis of ker(A : Z^n -> Z^m).

    Input:
        A : 2D numpy array or nested list of ints, shape (m, n)

    Output:
        basis : list of numpy arrays in Z^n forming a Z-basis of ker(A)

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
    '''
    Wrapper that performs LLL reduction using flint
    '''
    
    if isinstance(A, np.ndarray):
        A = A.tolist()
    L = np.array(fmpz_mat(A).lll().tolist()).astype(int)
    zeropos = np.where(sum(abs(L.T))==0)[0]
    L = np.delete(L,zeropos,0)
    return L


def integer_rowspan_basis(A):
    """
    Return a Z-basis of the Z-span of the rows of A.

    Input:
        A : 2D numpy array or nested list of ints, shape (m, n)

    Output:
        basis : list of numpy arrays in Z^n forming a Z-basis
                of the lattice generated by the rows of A
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
    **Description:**
    Computes the moving cone of a toric variety.
    
    **Arguments:**
    - `toric_fan` *(triangulation of vector configuration)*: toric fan

    **Returns:**
    Moving Cone (Cone)
    """
    
    rays = toric_variety.vectors()
    glsm = integer_kernel_basis(rays.T)
    h_planes = np.array([h for i in range(len(rays)) for h in Cone(np.delete(glsm.T,i,0)).hyperplanes()])@glsm
    mov = Cone(hyperplanes = h_planes)
    
    return mov

def generic_section_factorizes(points,linebundle_weights):
    """
    **Description:**
    Determines whether the generic section of a Divisor factorizes
    
    **Arguments:**
    - `points` *(numpy array or list)*: vectors of the toric fan
    - `linebundle_weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors

    **Returns:**
    Boolean: Generic section factorizes
    """
    
    try:
        NP=Newton_Polytope(points,linebundle_weights)
    except ValueError as e:
        return True
    return ~np.all(np.any(points@NP.points().T+linebundle_weights[:,None]!=0,axis=1))

def attempt_to_make_nef(toric_variety,line_bundle,epsilon=1e-5):

    """
    **Description:**
    Attempts to find a triangulation of a vector configuration such that the given line bundle is nef. Terminates once it has been found, or an exterior wall of
    the moving cone is encountered
    
    **Arguments:**
    - `toric_fan` *(triangulation of vector configuration)*: toric fan
    - `weights` *(numpy array or list)*: weights of Weil divisor as integer linear combination of prime divisors
    - `epsilon` (optional) *(float)*: magnitude of perturbation used to obtain triangulation instead of subdivision

    **Returns:**
    Toric variety, as triangulation of vector triangulation
    """

    line_bundle = np.array(line_bundle)
    hts0 = toric_variety.heights()+epsilon*np.array([(np.sin(i+1)+1) for i in range(len(line_bundle))])
    hts1 = line_bundle
    
    return toric_variety.vc.triangulate(heights=hts1/epsilon+hts0)

def basis(points):
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
    **Description:**
    Checks if two divisors L1,L2 sum to the anticanonical divisor
    
    **Arguments:**
    - `pts` *np.array*: Points of the fan
    - `L1` *np.array*: Divisor1 in terms of weights of the prime torics
    - `L2` *np.array*: Divisor2 in terms of weights of the prime torics

    **Returns:**
    (bool,np.array): (True is sums to anticanonical, False if not, character that is needed such that L1+L2=1)
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
    **Description:**
    Determines whether two divisors L1 = sum_i l1_i D_i,L2 = sum_i l2_i D_i determine a partition of the anticanonical divisor of the ambient variety. By a partition, a representation of L1,L2 is meant such that l1_i,l2_i in {0,1}.
    
    **Arguments:**
    - `points` *(numpy array)*: points of the toric fan
    - `L1` *(numpy array)*: representation of L1 in terms of l1
    - `L2` *(numpy array)*: representation of L2 in terms of l2

    **Returns:**
    Tuple (bool,bool,principle_div,principle_div) given by (is_partition,sums_to_anticanonical,principal divisor L1 needs to be shifted, principal divisor L2 needs to be shifted)
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
    **Description:**
    Changes triangulation such that a givien divisor D becomes Cartier in this triangulation (if possible)
    
    **Arguments:**
    - `tri` *Fan*: triangulation of a vector configuration
    - `D` *np.array*: Divisor D in terms of weights of the prime torics

    **Returns:**
    (Fan,np.array): New fan and new representation of D
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
    **Description:**
    Computes the base locus of a divisor. 
    
    **Arguments:**
    - `fan` *Fan*: Toric fan
    - `lb` *array-like*(float)*: Divisor weights

    **Returns:**
    List of cones that define the base locus of the line bundle
    """
    return base_locus(sections(fan.vectors(),lb),cones=fan.cones())
        
def base_locus(sections,cones=None,dim=4):
    """
    **Description:**
    Computes the base locus of a line bundle. 
    
    **Arguments:**
    - `sections` *(numpy array)*: sections of the line bundle
    - `cones` *(list or tuple of tuples)*: cones of the relevant toric fan. If left None, all base loci independent of the toric fan will be outputed. Not recommended for large h^1,1 varieties.
    - `dim` (optional) *(float)*: If the cones are not specified, the system needs to know the dimension of the variety. 4 is the default

    **Returns:**
    List of cones that define the base locus of the line bundle
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
    **Description:**
    Computes normal fan associated to lattice polytope p. If list of polytopes is given, the normal fan of the Minkowski sum is constructed. 
    Optionally, a maximal refinement of the normal fan is constructed. This requires specifying inequalities

    **Arguments:**
    - `p` *(polytope object)*: a lattice polytope, or list of lattice polytopes
    - `inequalities` *(list or array of length k+1 where k is the number of polytopes given)*: for maximal refinement, the inequalities <v,n_1 m_1 +...+ n_k m_k> + n_{k+1} >= 0 are imposed,
        where n is the vector of inequalities, and m_1 to m_k are the vertices of the polytopes p1,...,pk that sum to a given vertex of the Minkowski sum. Weights [r,1,r] are appropriate for
        interpreting the second polytope as the Newton polytope of the line bundle r*(anti-canonical-D_1).

    **Returns:**
    tuple of (toric variety, weights of line bundle, maximal cones in same order as the vertices of the Newton polytope)
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

def normal_fan_OLD(polytopes,inequalities=None,maximal_refinement=False,triangulate_refinement=False,return_unrefined_fan=False):

    """
    **Description:**
    Computes normal fan associated to lattice polytope p. If list of polytopes is given, the normal fan of the Minkowski sum is constructed. 
    Optionally, a maximal refinement of the normal fan is constructed. This requires specifying inequalities

    **Arguments:**
    - `p` *(polytope object)*: a lattice polytope, or list of lattice polytopes
    - `inequalities` *(list or array of length k+1 where k is the number of polytopes given)*: for maximal refinement, the inequalities <v,n_1 m_1 +...+ n_k m_k> + n_{k+1} >= 0 are imposed,
        where n is the vector of inequalities, and m_1 to m_k are the vertices of the polytopes p1,...,pk that sum to a given vertex of the Minkowski sum. Weights [r,1,r] are appropriate for
        interpreting the second polytope as the Newton polytope of the line bundle r*(anti-canonical-D_1).

    **Returns:**
    tuple of (toric variety, weights of line bundle, maximal cones in same order as the vertices of the Newton polytope)
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
    problem=False
    for vec_index in range(len(n_fan.vectors())):
        for c_index in range(len(cones)):
            if vec_index+1 in cones[c_index]:
                break
        tot=0
        for i in range(len(polytopes)):
            tot+=inequalities[i]*n_vectors[vec_index]@(polytopes[i].vertices()[vertex_split[c_index][i]])
        if tot>=0:
            print("XXX")
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

    vc_total = VectorConfiguration(all_vectors)

    if not triangulate_refinement:
        if return_unrefined_fan:
            return (vc_total,all_weights,n_fan)
        else:
            return (vc_total,all_weights)

        
    if return_unrefined_fan:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights,n_fan)
    else:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights)


def nested_sum(lists, depth=0, acc=0):
    if depth == len(lists):
        return acc
    return [
        nested_sum(lists, depth + 1, acc + x)
        for x in lists[depth]
    ]
def flatten(lst, depth):
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
    **Description:**
    Computes the d dimensional cones containing rigid O7 divisors
    
    **Arguments:**
    - `vc_orbifold` *VectorConfiguration or Fan*: Configuration of the toric variety
    - `O7_labels` *(list)*: The vector lables of the rigid O7 planes
    - `d` *(float)*: Dimension of the cones in question

    **Returns:**
    List of cones that contain O7 divisors
    """
    d_cones= get_lower_dimensional_cones(vc_orbifold.cones(),d)
    relevant_d_cones = [t for t in d_cones if set(t).issubset(O7_labels)]
    return relevant_d_cones


def basis_H2_toric_fan(toric_fan):
    """
    **Description:**
    Computes a basis for the GLSM of a toric fan
    
    **Arguments:**
    - `toric_fan` *Fan*: Toric fan

    **Returns:**
    A basis of the GLSM in terms of lables of vectors
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
    **Description:**
    Computes 5d polytope associated with uplift of trilayer orientifold in limit where all mid-layer divisors are blown down.
    
    **Arguments:**
    - `p` *(Polytope object)*: The reflexive trilayer polytope

    **Returns:**
    Polytope object: F-theory uplift as polytope in one higher dimension
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
    **Description:**
    Computes the exponents of all sections of a divisor.
    
    **Arguments:**
    - `points` *(np array)*: points of the toric fan of consideration
    - `weights` *np array*: The weights of the divisor

    **Returns:**
    np array: Exponents of all sections of the divisor
    """
    NP=Newton_Polytope(points,weights)
    if len(NP.points())==0:
        return np.array([])
    else:
        return points@NP.points().T+weights[:,None]

def solve_over_integers(M,b):
    """
    **Description:**
    Computes the solution to a linear equation Mx=b over the integers exactly
    
    **Arguments:**
    - `M` *np.array Matrix*: Matrix
    - `b` *np.array*: Inhomogeneous term

    **Returns:**
    np.array: solution x, and None if no solution exists
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
    Finds the minimal set of 1-based ray indices (the carrier face) whose 
    strictly positive linear combination yields the new_ray.
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
    Converts a NumPy array to a LaTeX matrix format.
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
    Computes the lattice index of the sublattice generated by the rows of mat.
    """
    A=Matrix(mat,domain=ZZ)
    snf = smith_normal_form(A)
    
    s=np.array(snf,dtype=int)
    l_ind=np.prod([s[i][i] for i in range(np.min(s.shape))])
    
    return l_ind

def integral_gale_transform(points):
    """
    Computes the exact, integer Gale transform using SymPy rational arithmetic.
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
    Finds the minimal set of 1-based ray indices whose strictly positive
    linear combination yields new_ray.

    Assumptions:
    - each cone in current_cones is an ordered tuple of 1-based integers;
    - current_cones are full-dimensional simplicial cones;
    - the fan is complete;
    - new_ray and all_vectors are integer NumPy arrays.
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
    Simplifies the rows of integer matrix A by dividing each row by the GCD of its entries.
    """
    A = np.asarray(A, dtype=int)

    row_gcds = np.gcd.reduce(np.abs(A), axis=1, keepdims=True)
    row_gcds[row_gcds == 0] = 1  

    return A // row_gcds


def divisor_intersections(fan, intersection_dict,divisors, basis_set,as_LLL=True):
            
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