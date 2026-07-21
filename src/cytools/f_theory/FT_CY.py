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
import warnings

# 3rd party imports
import numpy as np
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# CYTools imports
from cytools import Polytope, fetch_polytopes
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan
from cytools.f_theory import Uplift_functions as UF

__all__ = [
    "CY_orientifold",
    "F_Theory_Uplift",
    "fetch_orientifolds",
    "fetch_F_Theory_uplifts",
    "fetch_nef_partition_uplifts",
]


class CY_orientifold():
    """
    This class describes an O3/O7 orientifold of a Calabi-Yau hypersurface in
    a toric ambient variety.
    """

    
    def __init__(self, fan_polytope_or_points=None, xi=None, resolve_A1_singularities=True, construct_nef_decomposition=True):
        """
        **Description:**
    
        Constructs a `CY_orientifold` object describing an O3/O7 orientifold of a Calabi-Yau hypersurface in a toric ambient variety. The input toric data are converted into a vector configuration, orbifolded by the `Z_2` action `xi`, and used to determine the associated orientifold line bundle. Optionally, the construction resolves A1 singularities and attempts to refine the toric data so that the relevant divisors become Cartier and nef.
    
        **Arguments:**
    
        - `fan_polytope_or_points`: The input toric data. This can be a `Polytope`, `VectorConfiguration`, `Fan`, `numpy.ndarray`, or list of lattice points.
        - `xi`: The orientifold action/vector. If the input is a trilayer `Polytope`, this may be inferred automatically.
        - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the toric orbifold. Defaults to `True`.
        - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition by refining the orbifold data. Defaults to `True`.
    
        **Raises:**
    
        - `ValueError`: Raised if `xi` is required but not provided.
        - `TypeError`: Raised if the input toric data have an unsupported type.
    
        **Returns:**
    
        Nothing.
    
        """
        self.__p = None
        self.__toric_fan = None
        self.__xi = None
        self.__orbifold_toric_fan = None
        self.__orbifold_line_bundle = None
        self.__yields_nef_decomposition = None
        self.__regular = None
        self.__orbifold_pts = None
        self.__CY_ambient_pts=None
        self.__intersection_numbers_orbifold = None
        self.__normal_fan = None
        self.__Newton_Polytope = None
        self.__NHC_labels = None
        self.ambient_triangulation = False
        self._multiplier = 6
        self.__o_planes = None
        match fan_polytope_or_points:
            case Polytope():
                if xi is None:
                    if fan_polytope_or_points.is_trilayer():
                        xi = -UF.find_trilayer_vertex_polytope(fan_polytope_or_points)
                    else:
                        raise ValueError("If polytope is not trilayer, xi has to be defined")
                self.__p = fan_polytope_or_points
                ind = UF.get_index(fan_polytope_or_points.points(), np.zeros(len(fan_polytope_or_points.vertices()[0])))
                if len(ind) > 0:
                    pts = np.delete(fan_polytope_or_points.points(), ind[0], axis=0)
                else:
                    pts = fan_polytope_or_points.points()
            case VectorConfiguration():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
            case Fan():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
                self.__toric_fan = fan_polytope_or_points
                self.ambient_triangulation = True
            case np.ndarray():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points
            case list():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = np.array(fan_polytope_or_points, dtype=int)
            case _:
                raise TypeError("Unsupported input type")
        self.__CY_ambient_pts = pts
        self.__dim = np.linalg.matrix_rank(pts,tol=1e-6)
        self.__xi = xi          
        
        orbifold_pts, rescalings = UF.toric_orbifold(pts_CY_ambient=pts, q=xi)
        orbifold_line_bundle = UF.O3O7_line_bundle(pts_CY_ambient=pts, q=xi, rescalings=rescalings)
        if orbifold_line_bundle is None:
            self.__regular = False
        else:
            self.__regular = not bool(UF.generic_section_factorizes(orbifold_pts, orbifold_line_bundle))
        
        if self.__regular:
            if construct_nef_decomposition:
                self.__yields_nef_decomposition=False
                KB_multiplier = UF.Newton_Polytope(orbifold_pts, self._multiplier * (np.ones(len(orbifold_line_bundle), dtype=int) - orbifold_line_bundle))
                if len(KB_multiplier.points()) > 0:
                    NP_orbifold = UF.Newton_Polytope(orbifold_pts, orbifold_line_bundle)
                    self.__Newton_Polytope = NP_orbifold
                    if NP_orbifold.minkowski_sum(KB_multiplier).dim() == self.__dim:
                        orbifold_pts_refined, line_bundles, normal_fan = UF.normal_fan([NP_orbifold, KB_multiplier], np.array([self._multiplier, 1, self._multiplier]), maximal_refinement=True, triangulate_refinement=False, return_unrefined_fan=True)
                        if orbifold_pts_refined is not None:
                            self.__yields_nef_decomposition = True
                            self.__normal_fan = normal_fan
                            orbifold_pts = orbifold_pts_refined
                            orbifold_line_bundle = line_bundles[:, 0]
                            self.ambient_triangulation=False
                            self.__NHC_labels = np.where(self._multiplier*(1-orbifold_line_bundle)-line_bundles[:, 1]==3)[0]+1
                            
            
        else:
            self.__yields_nef_decomposition=False

        self.__orbifold_pts = orbifold_pts
        self.__orbifold_line_bundle = orbifold_line_bundle
        if resolve_A1_singularities:
            if self.ambient_triangulation:
                self.__resolve_A1_singularities()

    def __repr__(self):
        """
        **Description:**
    
        Produces a short human-readable representation of the object.
    
        **Returns:**
    
        - `str`: A description of the object.
    
        """ 
        return f"A Calabi-Yau orientifold"

    def __resolve_A1_singularities(self):

        """
        **Description:**
    
        Finds codimension-two `Z_2` fixed loci in the Calabi-Yau ambient toric fan and refines the orbifold fan by adding the corresponding blowup rays. The orbifold line bundle is updated by adding the induced weights of the exceptional divisors.
    
        **Returns:**
    
        Nothing.
    
        """

        singular_two_cones = UF.Z2_fixed_locus(self.CY_ambient_toric_fan(),self.xi(),cone_dimension=2,denominator=2)
        orbifold_blowups=[]
        if len(singular_two_cones)>0:
            for c in singular_two_cones:
                vec=np.sum(self.vectors_orbifold(c),axis=0)
                vec_add=np.rint(vec/np.gcd.reduce(vec)).astype(int)
                orbifold_blowups.append(vec_add.tolist())
            orbifold_blowups=np.array(orbifold_blowups)
            fan=UF.refine_fan(self.orbifold_toric_fan(),orbifold_blowups)
            self.__orbifold_pts=fan.vectors()
            blow_up_weights = np.rint(np.array([sum(self.line_bundle()[np.array(c)-1])-1 for c in singular_two_cones])/2).astype(int)
            self.__orbifold_line_bundle=np.concatenate((self.line_bundle(),blow_up_weights),axis=0)
            self.__orbifold_toric_fan=fan
        return None

    def orbifold_toric_fan(self):
        """
        **Description:**
    
        Constructs the toric fan associated with the orbifold vector configuration if it has not already been cached. If a nef decomposition was constructed, the fan is obtained from the corresponding normal fan refinement. Otherwise, it uses the cone structure inherited from the Calabi-Yau ambient toric fan.
    
        **Returns:**
    
        - `Fan`: The toric fan of the orbifold ambient variety.
    
        """
        if self.__orbifold_toric_fan is None:
            if self.ambient_triangulation:
                self.__orbifold_toric_fan=Fan(vc=VectorConfiguration(self.vectors_orbifold()),cones=self.CY_ambient_toric_fan().cones())
            else:
                if self.yields_nef_decomposition():
                    nf=UF.make_simplicial(self.normal_fan())
                    VC=VectorConfiguration(self.vectors_orbifold())
                    inds=VC.vectors_to_labels(nf.vectors())
                    new_cones=set()
                    for c in nf.cones():
                        nc=tuple(sorted([inds[x-1] for x in c]))
                        new_cones.add(nc)
                    new_fan=Fan(vc=VC,cones=new_cones)
                    self.__orbifold_toric_fan = UF.refine_fan(new_fan)
                else:
                    self.__orbifold_toric_fan = Fan(VectorConfiguration(self.vectors_orbifold()),cones=self.CY_ambient_toric_fan().cones())
        return self.__orbifold_toric_fan

    def CY_ambient_toric_fan(self):
        """
        **Description:**
    
        Returns the toric fan before orbifolding. If no fan was provided at initialization, one is constructed by triangulating the ambient vector configuration.
    
        **Returns:**
    
        - `Fan`: The toric fan of the original Calabi-Yau ambient variety.
    
        """
        if self.__toric_fan is None:
            self.__toric_fan = VectorConfiguration(self.__CY_ambient_pts).triangulate()
        return self.__toric_fan

    def vectors_orbifold(self, c=None):
        """
        **Description:**
    
        Returns all orbifold rays, or only those with labels specified by `c`. Labels are one-indexed, following the CYTools convention for toric rays.
    
        **Arguments:**
    
        - `c`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected orbifold vectors.
    
        """
        if c is not None:
            return self.__orbifold_pts[np.array(c)-1]
        else:
            return self.__orbifold_pts

    def vectors_CY_ambient(self, c=None):
        """
        **Description:**
    
        Returns all rays of the toric ambient space before orbifolding, or only those with labels specified by `c`. Labels are one-indexed.
    
        **Arguments:**
    
        - `c`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected ambient-space vectors.
    
        """
        if c is not None:
            return self.__CY_ambient_pts[np.array(c)-1]
        else:
            return self.__CY_ambient_pts

    def xi(self):
        """
        **Description:**
    
        Returns the vector `xi` specifying the toric `Z_2` orientifold action.
    
        **Returns:**
    
        - `numpy.ndarray` or list: The orientifold action/vector.
    
        """
        return self.__xi

    def is_regular(self):
        """
        **Description:**
    
        Returns whether the generic invariant Calabi-Yau hypersurface descends regularly to the toric orbifold, i.e. whether the corresponding generic section does not factorize in the orbifold description.
    
        **Returns:**
    
        - `bool`: `True` if the orientifold is regular, otherwise `False`.
    
        """
        return self.__regular

    def line_bundle(self):
        """
        **Description:**
    
        Returns the toric divisor coefficients of the O3/O7 line bundle on the orbifold ambient space.
    
        **Returns:**
    
        - `numpy.ndarray`: The orbifold line bundle coefficients.
    
        """
        return self.__orbifold_line_bundle

    def dim(self):
        """
        **Description:**
    
        Computes the rank of the ambient vector configuration used to define the Calabi-Yau orientifold.
    
        **Returns:**
    
        - `int`: The dimension of the toric ambient space.
    
        """
        return self.__dim

    def normal_fan(self):
        """
        **Description:**
    
        Computes and caches the normal fan associated with the relevant Newton polytopes if it has not already been constructed. This fan is used to test or implement the Cartier/nef refinement of the orientifold data.
    
        **Returns:**
    
        - `Fan`: The normal fan associated with the Newton-polytope data.
    
        """
        if self.__normal_fan is None:
            self.__normal_fan = UF.normal_fan([self.Newton_Polytope(), UF.Newton_Polytope(self.vectors_orbifold(), self._multiplier*(1-self.line_bundle()))])[0]
        return self.__normal_fan

    def ambient_dim(self):
        """
        **Description:**
    
        Returns the number of coordinates of the orbifold vectors.
    
        **Returns:**
    
        - `int`: The ambient lattice dimension.
    
        """
        return len(self.vectors_orbifold()[0])

    def Newton_Polytope(self):
        """
        **Description:**
    
        Computes and caches the Newton polytope associated with the orbifold vector configuration and orientifold line bundle.
    
        **Returns:**
    
        - `Polytope`: The Newton polytope of the orientifold line bundle.
    
        """
        if self.__Newton_Polytope is None:
            self.__Newton_Polytope = UF.Newton_Polytope(self.vectors_orbifold(), self.line_bundle())
        return self.__Newton_Polytope

    def yields_nef_decomposition(self):
        """
        **Description:**
    
        Determines whether the relevant line bundles define a Cartier and nef decomposition. If an ambient triangulation was provided, this is tested directly on the orbifold toric fan. Otherwise, it is checked through the normal-fan refinement data.
    
        **Returns:**
    
        - `bool`: `True` if the construction yields a nef decomposition, otherwise `False`.
    
        """
        if self.__yields_nef_decomposition is None:
            if self.ambient_triangulation:
                NHClb=np.zeros(len(self.line_bundle()))
                NHC_inds=self.NHC(as_labels=True)-1
                NHClb[NHC_inds]=1
                lb=self._multiplier*(1-self.line_bundle())-np.rint((self._multiplier/2)*NHClb).astype(int)
                self.__yields_nef_decomposition = UF.is_Cartier(self.orbifold_toric_fan(),lb)[0] and UF.is_nef(self.orbifold_toric_fan(),lb) and UF.is_Cartier(self.orbifold_toric_fan(),self.line_bundle())[0] and UF.is_nef(self.orbifold_toric_fan(),self.line_bundle())
            else:
                self.__yields_nef_decomposition = UF.contains_rows(self.vectors_orbifold(),self.normal_fan().vectors())
        return self.__yields_nef_decomposition

    def polytope(self):
        """
        **Description:**
    
        Returns the `Polytope` object used to initialize the orientifold. If the orientifold was initialized from a fan, vector configuration, array, or list of points, this returns `None`.
    
        **Returns:**
    
        - `Polytope` or `None`: The original polytope, if one was provided.
    
        """
        return self.__p

    def intersection_numbers_orbifold(self):
        """
        **Description:**
    
        Computes and caches the intersection numbers of the orbifold toric fan.
    
        **Returns:**
    
        - `dict`: The intersection numbers of the orbifold ambient toric variety.
    
        """
        if self.__intersection_numbers_orbifold is None:
            self.__intersection_numbers_orbifold = self.orbifold_toric_fan().intersection_numbers()
        return self.__intersection_numbers_orbifold

    def NHC(self,as_labels=False):
        """
        **Description:**
    
        Determines the toric divisors supporting non-Higgsable clusters by checking the sections of the appropriate line bundle. The result may be returned either as one-indexed ray labels or as lattice vectors.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed ray labels. If `False`, return the corresponding lattice vectors. Defaults to `False`.
    
        **Returns:**
    
        - `numpy.ndarray`: The NHC labels or corresponding toric rays.
    
        """
        if self.__NHC_labels is None:
            sections_NP2 = UF.sections(self.vectors_orbifold(), 2 * (1 - self.line_bundle()))
            if len(sections_NP2) > 0:
                self.__NHC_labels = np.where(np.min(sections_NP2, axis=1) == 1)[0] + 1
            else:
                self.__NHC_labels = np.array([])
        if as_labels:
            return self.__NHC_labels
        else:
            if len(self.__NHC_labels)==0:
                return np.array([])
            return self.vectors_orbifold(self.__NHC_labels)

    def o_planes(self):
        """
        **Description:**
    
        Computes and caches the fixed loci of the `Z_2` orientifold action in the Calabi-Yau ambient toric fan.
    
        **Returns:**
    
        - `list`: The toric strata fixed by the orientifold action.
    
        """
        if self.__o_planes is None:
            self.__o_planes = UF.Z2_fixed_locus(self.CY_ambient_toric_fan(),self.__xi)
        return self.__o_planes
            

class F_Theory_Uplift():
    """
    **Description:**

    Constructs and stores the toric data associated with an F-theory uplift of a `CY_orientifold`. The uplift is described as a toric complete intersection with base and Weierstrass line bundles, together with optional blowups over non-Higgsable-cluster loci. The class provides access to the associated singular and resolved ambient fans, Cayley polytopes, nef-partition data, intersection numbers, homology bases, and Hodge numbers when available.

    **Arguments:**

    - `orientifold_or_points`: Either a `CY_orientifold` object or raw toric input data from which a `CY_orientifold` should first be constructed.
    - `xi`: The orientifold action/vector. Required if raw toric data are provided instead of a `CY_orientifold`.
    - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the underlying orientifold. Defaults to `False`.
    - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition for the underlying orientifold. Defaults to `True`.

    **Returns:**

    Nothing.

    """
    
    def __init__(self, orientifold_or_points=None, xi=None, resolve_A1_singularities=False, construct_nef_decomposition=True):
        """
        **Description:**
    
        Constructs an `F_Theory_Uplift` object from either an existing `CY_orientifold` or from raw toric input data. The object stores the underlying orientifold and initializes caches for the singular uplift, resolved uplift, nef-partition data, divisor representations, Hodge numbers, and intersection-theoretic data.
    
        **Arguments:**
    
        - `orientifold_or_points`: Either a `CY_orientifold` object or raw toric input data accepted by `CY_orientifold`.
        - `xi`: The orientifold action/vector. Required when raw toric data are provided.
        - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the underlying orientifold. Defaults to `False`.
        - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition for the underlying orientifold. Defaults to `True`.
    
        **Returns:**
    
        Nothing.
    
        """
        self.__blowups = None
        self.__Cayley_M = None
        self.__Cayley_N = None
        self.__pol_N_sum = None
        self.__pol_N_conv = None
        self.__pol_M_conv = None
        self.__pol_M_sum = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = None
        self.__pol_B_M = None
        self.__pol_W_M = None
        self.__pol_W_N = None
        self.__pol_B_N = None
        self.__NHC_labels = None
        self.__M_conv_toric_fan = None
        self.__intersection_numbers_M_conv = None
        self.__intersection_numbers_singular_uplift = None
        self.__intersection_numbers_smooth_uplift = None
        self.__LBW_N = None
        self.__LBB_N = None
        self.__LBW_M = None
        self.__LBB_M = None
        self.__basis_homology_M = None
        self.__basis_homology_N = None
        self.__normal_fan = None
        self.__singular_uplift_toric_fan = None
        self.__pts_singular_uplift=None
        self.__pts_smooth_uplift=None
        self.__smooth_uplift_toric_fan = None
        self.x = None
        self.y = None
        self.z = None
        self.__h11 = None
        self.__h31 = None
        self.__h21 = None
        self.__chi = None
        self.__is_nef_partition = None
        self.__is_partition = None
        self.__is_nef_decomposition = None
        self.__CY_orientifold = None

        match orientifold_or_points:
            case CY_orientifold():
                self.__CY_orientifold = orientifold_or_points
            case _:
                self.__CY_orientifold = CY_orientifold(orientifold_or_points, xi=xi, resolve_A1_singularities=resolve_A1_singularities, construct_nef_decomposition=construct_nef_decomposition)

        if not self.is_regular():
            self.__is_partition=False
            self.__is_nef_partition=False

    def __set_divisor_representations(self):
        """
        **Description:**
    
        Constructs the toric divisor coefficient vectors representing the base and Weierstrass line bundles on the resolved uplift ambient space. The method also checks whether these divisors define a valid partition of the anticanonical class and stores the resulting line bundle data.
    
        **Returns:**
    
        Nothing.
    
        """
        n_sing = len(self.vectors_orbifold()) + 3
        n_smooth = len(self.vectors_smooth_uplift_ambient())
        LBB_N = np.zeros(n_smooth, dtype=int)
        LBW_N = np.zeros(n_smooth, dtype=int)
        LBB_N[:n_sing-3] = self.line_bundle_orbifold()
        LBW_N[n_sing-3] = 3
        NHC = self.NHC(as_labels=True) - 1
        if len(NHC) > 0:
            cont = np.where(self.line_bundle_orbifold() != 0)[0]
            nhc_cont = np.where(np.isin(NHC, cont))[0]
            for n in nhc_cont:
                contribution_n = self.line_bundle_orbifold()[NHC[n]]
                LBB_N[2*n + n_sing] = 1 * contribution_n
                LBB_N[2*n + 1 + n_sing] = 2 * contribution_n
        is_part=UF.compute_partition([LBB_N, LBW_N],self.vectors_smooth_uplift_ambient())
        if is_part[0]:
            self.__LBB_N = is_part[1][0]
            self.__LBW_N = is_part[1][1]
        else:
            sta=UF.sums_to_anticanonical(self.vectors_smooth_uplift_ambient(),LBB_N,LBW_N)
            self.__LBB_N = LBB_N
            self.__LBW_N = self.vectors_smooth_uplift_ambient()@sta[1]+LBW_N
        self.__is_partition = is_part[0]

    def __repr__(self):

        """
        **Description:**
    
        Produces a short human-readable representation of the object.
    
        **Returns:**
    
        - `str`: A description of the object.
    
        """
        
        return f"An F-theory uplift of a Calabi-Yau orientifold"
        
    def orientifold(self):

        """
        **Description:**
    
        Returns the `CY_orientifold` object from which the F-theory uplift is constructed.
    
        **Returns:**
    
        - `CY_orientifold`: The underlying orientifold.
    
        """
        
        return self.__CY_orientifold

    def is_nef_partition(self):
        """
        **Description:**
    
        Returns whether the resolved F-theory uplift defines both a valid partition and a nef decomposition. The value is computed once and then cached.
    
        **Returns:**
    
        - `bool`: `True` if the uplift defines a nef partition, otherwise `False`.
    
        """
        if self.__is_nef_partition is None:
            if self.is_nef_decomposition():
                if self.is_partition():
                    self.__is_nef_partition = True
                    return True
            self.__is_nef_partition = False        
        return self.__is_nef_partition
        
    def is_partition(self):
        """
        **Description:**
    
        Returns whether the base and Weierstrass divisor representatives sum to the anticanonical class of the resolved uplift ambient space.
    
        **Returns:**
    
        - `bool`: `True` if the uplift defines a partition, otherwise `False`.
    
        """
        if self.__is_partition is None:
            self.__set_divisor_representations()
        return self.__is_partition

    def smooth_uplift_ambient_toric_fan(self):
        """
        **Description:**
    
        Constructs and caches the toric fan of the resolved F-theory uplift ambient space. Starting from the singular uplift fan, the method inserts the blowup rays associated with non-Higgsable-cluster divisors and performs the corresponding local cone refinements.
    
        **Returns:**
    
        - `Fan`: The toric fan of the resolved uplift ambient space.
    
        """
        
        if self.__smooth_uplift_toric_fan is None:
            n_sing_uplift = len(self.vectors_singular_uplift_ambient())
            xlabel = n_sing_uplift - 2
            ylabel = n_sing_uplift - 1
            ct = 0
            vc_smooth = VectorConfiguration(self.vectors_smooth_uplift_ambient())
            new_fan = Fan(vc=vc_smooth, cones=self.singular_uplift_ambient_toric_fan().cones())
        
            for xxx in self.NHC_singular_uplift(as_labels=True):
                all_cones = set(new_fan.cones())
                for link in new_fan.link((xxx, xlabel, ylabel)):
                    all_cones.discard(tuple(sorted((xxx, xlabel, ylabel) + link)))
                    all_cones.add(tuple(sorted(link + (xxx, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
                    all_cones.add(tuple(sorted(link + (xxx, xlabel, n_sing_uplift + 2*ct + 2))))
                    all_cones.add(tuple(sorted(link + (xxx, ylabel, n_sing_uplift + 2*ct + 1))))
                    all_cones.add(tuple(sorted(link + (xlabel, ylabel, n_sing_uplift + 2*ct + 1))))
                    all_cones.add(tuple(sorted(link + (xlabel, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
                ct += 1
                new_fan = Fan(vc=vc_smooth, cones=all_cones)
            self.__smooth_uplift_toric_fan = new_fan
        return self.__smooth_uplift_toric_fan

    def singular_uplift_ambient_toric_fan(self):            
        """
        **Description:**
    
        Constructs and caches the toric fan obtained by adding the toric `P^2_[2,3,1]` fiber rays to the orbifold base fan, before resolving non-Higgsable-cluster singularities.
    
        **Returns:**
    
        - `Fan`: The toric fan of the singular uplift ambient space.
    
        """
        if self.__singular_uplift_toric_fan is None:
            cones6 = set()
            x_pts6 = len(self.vectors_orbifold()) + 1
            y_pts6 = x_pts6 + 1
            z_pts6 = x_pts6 + 2
            
            for c in self.orbifold_toric_fan().cones():
                cones6.add(c + (x_pts6, y_pts6))
                cones6.add(c + (x_pts6, z_pts6))
                cones6.add(c + (y_pts6, z_pts6))
            self.__singular_uplift_toric_fan = Fan(VectorConfiguration(self.vectors_singular_uplift_ambient()),cones=cones6)
        return self.__singular_uplift_toric_fan
        
    def Cayley_M(self):
        """
        **Description:**
    
        Constructs and caches the Cayley polytope associated with the base and Weierstrass Newton polytopes in the M-lattice. If the uplift is not a nef partition, a warning is raised.
    
        **Returns:**
    
        - `Polytope`: The M-lattice Cayley polytope.
    
        """
        if self.__Cayley_M is None:
            self.__Cayley_M = Polytope(np.concatenate((
                np.column_stack([self.pol_W_M().points(), np.ones(len(self.pol_W_M().points()), dtype=int), np.zeros(len(self.pol_W_M().points()), dtype=int)]),
                np.column_stack([self.pol_B_M().points(), np.zeros(len(self.pol_B_M().points()), dtype=int), np.ones(len(self.pol_B_M().points()), dtype=int)])
            ), axis=0))
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__Cayley_M
        
    def Cayley_N(self):
        """
        **Description:**
    
        Constructs and caches the Cayley polytope associated with the dual N-lattice nef partition. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The N-lattice Cayley polytope.
    
        """
        
        if self.is_partition():
            if self.__Cayley_N is None:
                self.__Cayley_N = Polytope(np.concatenate((
                    np.column_stack([self.pol_W_N().points(), np.ones(len(self.pol_W_N().points()), dtype=int), np.zeros(len(self.pol_W_N().points()), dtype=int)]),
                    np.column_stack([self.pol_B_N().points(), np.zeros(len(self.pol_B_N().points()), dtype=int), np.ones(len(self.pol_B_N().points()), dtype=int)])
                ), axis=0))
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__Cayley_N

    def pol_W_M(self):
        """
        **Description:**
    
        Computes and caches the Newton polytope associated with the Weierstrass line bundle on the resolved uplift ambient space.
    
        **Returns:**
    
        - `Polytope`: The Weierstrass Newton polytope in the M-lattice.
    
        """
        if self.__pol_W_M is None:
            self.__pol_W_M = UF.Newton_Polytope(self.vectors_smooth_uplift_ambient(), self.line_bundle_weierstrass_N())
        return self.__pol_W_M
        
    def pol_B_M(self):
        """
        **Description:**
    
        Computes and caches the Newton polytope associated with the base divisor line bundle on the resolved uplift ambient space.
    
        **Returns:**
    
        - `Polytope`: The base Newton polytope in the M-lattice.
    
        """
        if self.__pol_B_M is None:
            self.__pol_B_M = UF.Newton_Polytope(self.vectors_smooth_uplift_ambient(), self.line_bundle_base_N())
        return self.__pol_B_M

    def pol_M_conv(self):
        """
        **Description:**
    
        Constructs and caches the convex hull of the union of the base and Weierstrass Newton polytopes in the M-lattice.
    
        **Returns:**
    
        - `Polytope`: The convex hull polytope in the M-lattice.
    
        """
        if self.__pol_M_conv is None:
            self.__pol_M_conv = Polytope(np.unique(np.concatenate((self.pol_B_M().vertices(), self.pol_W_M().vertices()), axis=0), axis=0))
        return self.__pol_M_conv
        
    def pol_M_sum(self):
        """
        **Description:**
    
        Computes and caches the Minkowski sum of the base and Weierstrass Newton polytopes in the M-lattice.
    
        **Returns:**
    
        - `Polytope`: The Minkowski sum of the M-lattice Newton polytopes.
    
        """
        if self.__pol_M_sum is None:
            self.__pol_M_sum = self.pol_B_M().minkowski_sum(self.pol_W_M())
        return self.__pol_M_sum

    def pol_W_N(self):
        """
        **Description:**
    
        Constructs and caches the N-lattice polytope corresponding to the Weierstrass part of the dual nef partition. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The Weierstrass polytope in the N-lattice.
    
        """
        if self.is_partition():
            if self.__pol_W_N is None:
                self.__pol_W_N = Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.vectors_smooth_uplift_ambient()[np.where(self.line_bundle_weierstrass_N()==1)[0]]), axis=0))
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_W_N

    def pol_B_N(self):
        """
        **Description:**
    
        Constructs and caches the N-lattice polytope corresponding to the base part of the dual nef partition. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The base polytope in the N-lattice.
    
        """
        if self.is_partition():
            if self.__pol_B_N is None:
                self.__pol_B_N = Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.vectors_smooth_uplift_ambient()[np.where(self.line_bundle_base_N()==1)[0]]), axis=0))
        else:
            raise ValueError("Uplift does not yield a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_B_N
    
    def pol_N_conv(self):
        """
        **Description:**
    
        Constructs and caches the convex hull of the lattice points defining the resolved uplift ambient space.
    
        **Returns:**
    
        - `Polytope`: The convex N-lattice uplift polytope.
    
        """
        if self.__pol_N_conv is None:
            self.__pol_N_conv = Polytope(self.vectors_smooth_uplift_ambient())
        return self.__pol_N_conv
        
    def pol_N_sum(self):
        """
        **Description:**
    
        Computes and caches the Minkowski sum of the base and Weierstrass N-lattice polytopes. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The Minkowski sum of the N-lattice partition polytopes.
    
        """
        if self.is_partition():
            if self.__pol_N_sum is None:
                self.__pol_N_sum = self.pol_B_N().minkowski_sum(self.pol_W_N())
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_N_sum
        
    def vectors_singular_uplift_ambient(self, labels=None):
        """
        **Description:**
    
        Constructs and caches the rays of the singular F-theory uplift ambient fan. These consist of the orbifold base rays together with the three toric `P^2_[2,3,1]` fiber rays. If `labels` are provided, only the corresponding one-indexed rays are returned.
    
        **Arguments:**
    
        - `labels`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected singular-uplift ambient vectors.
    
        """
        if self.__pts_singular_uplift is None:
            self.x = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([3, 1])))
            self.y = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([-2, -1])))
            self.z = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([0, 1])))
            coordinates231 = (1 - self.line_bundle_orbifold())[:, None] @ self.z[-2:][None, :]
            pts6 = np.concatenate((self.vectors_orbifold(), coordinates231), axis=1)
            pts6 = np.concatenate((pts6, np.array([self.x, self.y, self.z])), axis=0)
            self.__pts_singular_uplift=pts6
        if labels is None:
            return self.__pts_singular_uplift
        return self.__pts_singular_uplift[np.array(labels)-1]

    def blowups(self, as_labels=False):
        """
        **Description:**
    
        Constructs and caches the blowup rays inserted over the non-Higgsable clusters of the singular uplift. The result can be returned either as lattice vectors or as one-indexed ray labels in the resolved uplift fan.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed labels of the blowup rays. If `False`, return the corresponding lattice vectors. Defaults to `False`.
    
        **Returns:**
    
        - `list` or `numpy.ndarray`: The blowup vectors or their ray labels.
    
        """
        if self.__blowups is None:
            bus = []
            for xxx in self.NHC_singular_uplift(as_labels=True):
                bus.append(np.array([
                    self.vectors_singular_uplift_ambient()[xxx-1] + self.x + 2*self.y,
                    2*self.vectors_singular_uplift_ambient()[xxx-1] + 2*self.x + 3*self.y
                ]))
            self.__blowups = bus
        if as_labels:
            return np.arange(len(self.vectors_singular_uplift_ambient())+1,len(self.vectors_smooth_uplift_ambient())+1)
        return self.__blowups
        
    def points_not_interior_to_codim_1_and_2_face_M(self):
        """
        **Description:**
    
        Returns the points of the M-lattice convex hull polytope that are not interior to facets or codimension-two faces. These are the points retained for the corresponding toric ambient fan construction.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected M-lattice points.
    
        """
        if self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = UF.points_not_interior_to_facets_and_codim2_faces(self.pol_M_conv())
        return self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv

    def points_not_interior_to_codim_1_and_2_face_N(self):
        """
        **Description:**
    
        Returns the points of the N-lattice convex hull polytope that are not interior to facets or codimension-two faces.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected N-lattice points.
    
        """
        if self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = UF.points_not_interior_to_facets_and_codim2_faces(self.pol_N_conv())
        return self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv
        
    def polytope(self):
        """
        **Description:**
    
        Returns the polytope underlying the associated `CY_orientifold`. If the orientifold was not initialized from a polytope, this returns `None`.
    
        **Returns:**
    
        - `Polytope` or `None`: The original polytope, if one was provided.
    
        """
        return self.orientifold().polytope()

    def intersection_numbers_M_conv(self, check=True):
        """
        **Description:**
    
        Computes and caches the intersection numbers of the toric fan associated with the M-lattice convex hull polytope.
    
        **Arguments:**
    
        - `check (bool)`: Included for interface compatibility. Defaults to `True`.
    
        **Returns:**
    
        - `dict`: The intersection numbers of the M-lattice toric fan.
    
        """
        if self.__intersection_numbers_M_conv is None:
            self.__intersection_numbers_M_conv = self.M_conv_toric_fan().intersection_numbers()
        return self.__intersection_numbers_M_conv
        
    def intersection_numbers_smooth_uplift_ambient(self):
        """
        **Description:**
    
        Computes and caches the intersection numbers of the resolved F-theory uplift ambient toric fan.
    
        **Returns:**
    
        - `dict`: The intersection numbers of the resolved uplift ambient fan.
    
        """
        if self.__intersection_numbers_smooth_uplift is None:
            self.__intersection_numbers_smooth_uplift = self.smooth_uplift_ambient_toric_fan().intersection_numbers()
        return self.__intersection_numbers_smooth_uplift
        
    def intersection_numbers_singular_uplift_ambient(self, check=True):
        """
        **Description:**
    
        Computes and caches the intersection numbers of the singular F-theory uplift ambient toric fan.
    
        **Arguments:**
    
        - `check (bool)`: Included for interface compatibility. Defaults to `True`.
    
        **Returns:**
    
        - `dict`: The intersection numbers of the singular uplift ambient fan.
    
        """
        if self.__intersection_numbers_singular_uplift is None:
            self.__intersection_numbers_singular_uplift = self.singular_uplift_ambient_toric_fan().intersection_numbers()
        return self.__intersection_numbers_singular_uplift
        
    def h11(self):
        """
        **Description:**
    
        Computes and caches `h^{1,1}` using the Batyrev-Borisov formulas for two-part nef partitions. This is only available when the uplift defines a nef partition.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `int`: The Hodge number `h^{1,1}`.
    
        """
        if self.is_nef_partition():
            if self.__h11 is None:
                self.__h11 = UF.h11_2_part(self.Cayley_M(), self.Cayley_N())
            return self.__h11 
        else: 
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")
        
    def h31(self):
        """
        **Description:**
    
        Computes and caches `h^{3,1}` using the Batyrev-Borisov formulas for two-part nef partitions. This is only available when the uplift defines a nef partition.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `int`: The Hodge number `h^{3,1}`.
    
        """
        if self.is_nef_partition():
            if self.__h31 is None:
                self.__h31 = UF.h11_2_part(self.Cayley_N(), self.Cayley_M())
            return self.__h31
        else: 
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")
            
    def h31_trivial(self):
        """
        **Description:**
    
        Computes a fast point-counting expression based on the M-lattice Cayley polytope. This is not the full Batyrev-Borisov Hodge-number formula.
    
        **Returns:**
    
        - `int`: The point-counting estimate.
    
        """
        return len(self.Cayley_M().points()) - 1 - (self.Cayley_N().dim())

    def line_bundle_weierstrass_M(self):
        """
        **Description:**
    
        Computes and caches the divisor coefficient vector corresponding to the Weierstrass component of the dual nef partition in the M-lattice.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `numpy.ndarray`: The M-lattice Weierstrass line bundle coefficients.
    
        """
        if self.is_nef_partition():
            if self.__LBW_M is None:
                LB_W = np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int)
                LB_W[UF.get_indices(self.points_not_interior_to_codim_1_and_2_face_M(), self.pol_W_M().points())] = 1
                self.__LBW_M = LB_W[1:]
        else:
            raise ValueError("Uplift is not a nef-partition")
        return self.__LBW_M

    def line_bundle_base_M(self):
        """
        **Description:**
    
        Computes and caches the divisor coefficient vector corresponding to the base component of the dual nef partition in the M-lattice.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `numpy.ndarray`: The M-lattice base line bundle coefficients.
    
        """
        if self.is_nef_partition():
            if self.__LBB_M is None:
                LB_B = np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int)
                LB_B[UF.get_indices(self.points_not_interior_to_codim_1_and_2_face_M(), self.pol_B_M().points())] = 1
                self.__LBB_M = LB_B[1:]
        else:
            raise ValueError("Uplift is not a nef-partition")
        return self.__LBB_M

    def line_bundle_weierstrass_N(self):
        """
        **Description:**
    
        Returns the divisor coefficient vector representing the Weierstrass component of the partition on the resolved uplift ambient space.
    
        **Returns:**
    
        - `numpy.ndarray`: The N-lattice Weierstrass line bundle coefficients.
    
        """
        if self.__LBW_N is None:
            self.__set_divisor_representations()
        return self.__LBW_N

    def line_bundle_base_N(self):
        """
        **Description:**
    
        Returns the divisor coefficient vector representing the base component of the partition on the resolved uplift ambient space.
    
        **Returns:**
    
        - `numpy.ndarray`: The N-lattice base line bundle coefficients.
    
        """
        if self.__LBB_N is None:
            self.__set_divisor_representations()
        return self.__LBB_N

    def line_bundle_orbifold(self):
        """
        **Description:**
    
        Returns the line bundle of the underlying Calabi-Yau orientifold.
    
        **Returns:**
    
        - `numpy.ndarray`: The orientifold line bundle coefficients.
    
        """
        return self.orientifold().line_bundle()
        
    def M_conv_toric_fan(self):
        """
        **Description:**
    
        Constructs and caches a triangulation of the relevant primitive rays of the M-lattice convex-hull polytope. This fan is only defined when the uplift is a partition.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Fan`: The toric fan associated with the M-lattice convex hull.
    
        """
        if not self.is_partition():
            raise ValueError("Uplift is not a partition")
        if self.__M_conv_toric_fan is None:
            index0 = UF.get_index(self.points_not_interior_to_codim_1_and_2_face_M(),np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()[0]),dtype=int))[0]
            pts=np.delete(self.points_not_interior_to_codim_1_and_2_face_M(),index0,axis=0)
            self.__M_conv_toric_fan = VectorConfiguration(pts).triangulate()
        return self.__M_conv_toric_fan

    def divisor_intersection_M(self, as_LLL=True):
        """
        **Description:**
    
        Computes the curve classes obtained by intersecting the base and Weierstrass divisors in the M-lattice toric fan.
    
        **Arguments:**
    
        - `as_LLL (bool)`: Whether to return the intersections in an LLL-reduced basis. Defaults to `True`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - Object returned by `UF.divisor_intersections`: The divisor-intersection data in the chosen homology basis.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Uplift is not a nef-partition")
        return(UF.divisor_intersections(fan=self.M_conv_toric_fan(),divisors=[self.line_bundle_base_M(),self.line_bundle_weierstrass_M()],intersection_dict=self.intersection_numbers_M_conv(),basis_set=set(self.basis_homology_M()),as_LLL=as_LLL))

    def divisor_intersection_N(self, as_LLL=True):
        """
        **Description:**
    
        Computes the curve classes obtained by intersecting the base and Weierstrass divisors in the resolved uplift ambient toric fan.
    
        **Arguments:**
    
        - `as_LLL (bool)`: Whether to return the intersections in an LLL-reduced basis. Defaults to `True`.
    
        **Returns:**
    
        - Object returned by `UF.divisor_intersections`: The divisor-intersection data in the chosen homology basis.
    
        """
        return(UF.divisor_intersections(fan=self.smooth_uplift_ambient_toric_fan(),divisors=[self.line_bundle_base_N(),self.line_bundle_weierstrass_N()],intersection_dict=self.intersection_numbers_smooth_uplift_ambient(),basis_set=set(self.basis_homology_N()),as_LLL=as_LLL))
    
    def NHC(self, as_labels=False):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the divisors supporting non-Higgsable clusters, either as one-indexed labels or as lattice vectors.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed ray labels. If `False`, return the corresponding lattice vectors. Defaults to `False`.
    
        **Returns:**
    
        - `numpy.ndarray`: The NHC labels or corresponding toric rays.
    
        """
        return self.orientifold().NHC(as_labels)

    def NHC_singular_uplift(self, as_labels=False):
        """
        **Description:**
    
        Returns the rays of the singular uplift ambient fan corresponding to the base non-Higgsable-cluster divisors. The result may be returned as one-indexed labels or as vectors.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed ray labels. If `False`, return the corresponding lattice vectors. Defaults to `False`.
    
        **Returns:**
    
        - `numpy.ndarray`: The NHC labels or corresponding singular-uplift rays.
    
        """
        if as_labels:
            return self.NHC(as_labels=True)
        else:
            labels = self.NHC(as_labels=True)
            if len(labels) > 0:
                return self.vectors_singular_uplift_ambient(labels)
        
    def vectors_orbifold(self, labels=None):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns either all orbifold base rays or the one-indexed rays specified by `labels`.
    
        **Arguments:**
    
        - `labels`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected orbifold base vectors.
    
        """
        return self.orientifold().vectors_orbifold(labels)
    
    def vectors_CY_ambient(self, labels=None):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns either all original ambient rays or the one-indexed rays specified by `labels`.
    
        **Arguments:**
    
        - `labels`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected ambient vectors.
    
        """
        return self.orientifold().vectors_CY_ambient(labels)
    
    def dim_base(self):
        """
        **Description:**
    
        Returns the dimension of the toric ambient space of the underlying orientifold.
    
        **Returns:**
    
        - `int`: The base dimension.
    
        """
        return self.orientifold().dim()
        
    def ambient_dim_base(self):
        """
        **Description:**
    
        Returns the number of coordinates of the base orbifold rays.
    
        **Returns:**
    
        - `int`: The ambient lattice dimension of the base.
    
        """
        return self.orientifold().ambient_dim()

    def CY_ambient_toric_fan(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the toric fan before orbifolding.
    
        **Returns:**
    
        - `Fan`: The original Calabi-Yau ambient toric fan.
    
        """
        return self.orientifold().CY_ambient_toric_fan()

    def orbifold_toric_fan(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the toric fan of the orbifold base.
    
        **Returns:**
    
        - `Fan`: The orbifold base toric fan.
    
        """
        return self.orientifold().orbifold_toric_fan()

    def vectors_smooth_uplift_ambient(self, labels=None):
        """
        **Description:**
    
        Constructs and caches the rays of the resolved F-theory uplift ambient fan. These consist of the singular uplift rays together with the blowup rays over non-Higgsable clusters. If `labels` are provided, only the corresponding one-indexed rays are returned.
    
        **Arguments:**
    
        - `labels`: Optional collection of one-indexed ray labels.
    
        **Returns:**
    
        - `numpy.ndarray`: The selected resolved-uplift ambient vectors.
    
        """
        if self.__pts_smooth_uplift is None:
            if len(self.NHC()) > 0:
                self.__pts_smooth_uplift = np.concatenate((self.vectors_singular_uplift_ambient(), np.vstack(self.blowups())), axis=0)
            else: 
                self.__pts_smooth_uplift = self.vectors_singular_uplift_ambient()
        if labels is None:
            return self.__pts_smooth_uplift
        return self.__pts_smooth_uplift[np.array(labels)-1]

    def xi(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the vector `xi` specifying the toric `Z_2` action.
    
        **Returns:**
    
        - `numpy.ndarray` or list: The orientifold action/vector.
    
        """
        return self.__CY_orientifold.xi()

    def h21(self):
        """
        **Description:**
    
        Computes and caches `h^{2,1}` using the Batyrev-Borisov formulas for two-part nef partitions. This is only available when the uplift defines a nef partition.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `int`: The Hodge number `h^{2,1}`.
    
        """
        if self.__h21 is None:
            if self.is_nef_partition():
                self.__h21 = UF.h21_2_part(self.Cayley_N(), self.Cayley_M())
            else:
                raise ValueError("Uplift is not a nef-partition")
        return self.__h21

    def chi(self):
        """
        **Description:**
    
        Computes and caches the Euler characteristic using the fourfold relation `chi = 48 + 6*(h11 + h31 - h21)`.
    
        **Returns:**
    
        - `int`: The Euler characteristic.
    
        """
        if self.__chi is None:
            self.__chi = 48 + 6 * (self.h11() + self.h31() - self.h21())
        return self.__chi
        
    def is_regular(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns whether the orientifold construction is regular.
    
        **Returns:**
    
        - `bool`: `True` if the orientifold is regular, otherwise `False`.
    
        """
        return self.orientifold().is_regular()

    def normal_fan(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the normal fan used in the nef-decomposition construction.
    
        **Returns:**
    
        - `Fan`: The normal fan associated with the orientifold data.
    
        """
        return self.orientifold().normal_fan()
    
    def basis_homology_M(self, as_LLL=True):
        """
        **Description:**
    
        Computes and caches a basis of `H_2` for the toric fan associated with the M-lattice convex hull.
    
        **Arguments:**
    
        - `as_LLL (bool)`: Included for interface compatibility. Defaults to `True`.
    
        **Returns:**
    
        - Object returned by `UF.basis_H2_toric_fan`: A basis for the relevant toric curve homology.
    
        """
        if self.__basis_homology_M is None:
            self.__basis_homology_M = UF.basis_H2_toric_fan(self.M_conv_toric_fan())
        return self.__basis_homology_M
    
    def basis_homology_N(self, as_LLL=True):
        """
        **Description:**
    
        Computes and caches a basis of `H_2` for the resolved F-theory uplift ambient toric fan.
    
        **Arguments:**
    
        - `as_LLL (bool)`: Included for interface compatibility. Defaults to `True`.
    
        **Returns:**
    
        - Object returned by `UF.basis_H2_toric_fan`: A basis for the relevant toric curve homology.
    
        """
        if self.__basis_homology_N is None:
            self.__basis_homology_N = UF.basis_H2_toric_fan(self.smooth_uplift_ambient_toric_fan())
        return self.__basis_homology_N
    
    def nef_partition_N(self):
        """
        **Description:**
    
        Returns the one-indexed labels of the base and Weierstrass components of the N-lattice nef partition. If the uplift is not a nef partition, returns empty tuples.
    
        **Returns:**
    
        - `tuple`: A pair of tuples `(base_labels, weierstrass_labels)`.
    
        """
        if self.is_nef_partition():
            return (tuple(np.where(self.line_bundle_base_N()==1)[0]+1),tuple(np.where(self.line_bundle_weierstrass_N()==1)[0]+1))
        else:
            return ((),())

    def nef_partition_M(self):
        """
        **Description:**
    
        Returns the one-indexed labels of the base and Weierstrass components of the M-lattice nef partition. If the uplift is not a nef partition, returns empty tuples.
    
        **Returns:**
    
        - `tuple`: A pair of tuples `(base_labels, weierstrass_labels)`.
    
        """
        if self.is_nef_partition():
            return (tuple(np.where(self.line_bundle_base_M()==1)[0]+1),tuple(np.where(self.line_bundle_weierstrass_M()==1)[0]+1))
        else:
            return ((),())

            
    def is_nef_decomposition(self):
        """
        **Description:**
    
        Determines whether the base and Weierstrass line bundles define Cartier and nef divisors on the resolved uplift ambient fan. If the underlying orientifold was not constructed from an ambient triangulation, this delegates to the orientifold nef-decomposition check.
    
        **Returns:**
    
        - `bool`: `True` if the uplift defines a nef decomposition, otherwise `False`.
    
        """
        if self.__is_nef_decomposition is None:
            if self.orientifold().ambient_triangulation:
                self.__is_nef_decomposition = np.all([UF.is_Cartier(self.smooth_uplift_ambient_toric_fan(),self.line_bundle_base_N())[0],UF.is_Cartier(self.smooth_uplift_ambient_toric_fan(),self.line_bundle_weierstrass_N())[0],UF.is_nef(self.smooth_uplift_ambient_toric_fan(),self.line_bundle_base_N()),UF.is_nef(self.smooth_uplift_ambient_toric_fan(),self.line_bundle_weierstrass_N())])
            else:
                self.__is_nef_decomposition = self.orientifold().yields_nef_decomposition()
        return self.__is_nef_decomposition
            
    def intersection_numbers_orbifold(self):
        """
        **Description:**
    
        Delegates to the underlying orientifold and returns the intersection numbers of the orbifold toric fan.
    
        **Returns:**
    
        - `dict`: The intersection numbers of the orbifold base.
    
        """
        return self.orientifold().intersection_numbers_orbifold()

def fetch_orientifolds(only_regular: bool=True, only_nef_decomposition: bool=False,h11: int = None,h12: int = None,h13: int = None,
    h21: int = None,h22: int = None,h31: int = None,chi: int = None,
    lattice: str = 'N',dim: int = 4,n_points: int = None,n_vertices: int = None,
    n_dual_points: int = None,n_facets: int = None,limit: int = 1000,
    samples: int = None,sample_seed: int = None,timeout: int = 60,
    as_list: bool = True,backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0):

    
    """
    **Description:**

    Iterates over polytopes returned by `fetch_polytopes`, computes inequivalent toric `Z_2` actions, and yields the corresponding `CY_orientifold` objects. The output may optionally be restricted to regular orientifolds or to orientifolds yielding a nef decomposition.

    **Arguments:**

    - `only_regular (bool)`: Whether to yield only regular orientifolds. Defaults to `True`.
    - `only_nef_decomposition (bool)`: Whether to yield only orientifolds that yield a nef decomposition. Defaults to `False`.
    - `h11, h12, h13, h21, h22, h31, chi`: Optional Hodge-number and Euler characteristic filters passed to `fetch_polytopes`.
    - `lattice (str)`: Lattice used by `fetch_polytopes`. Defaults to `"N"`.
    - `dim (int)`: Polytope dimension. Defaults to `4`.
    - `n_points (int)`: Optional filter on the number of lattice points.
    - `n_vertices (int)`: Optional filter on the number of vertices.
    - `n_dual_points (int)`: Optional filter on the number of dual points.
    - `n_facets (int)`: Optional filter on the number of facets.
    - `limit (int)`: Maximum number of polytopes fetched. Defaults to `1000`.
    - `samples (int)`: Optional number of random samples.
    - `sample_seed (int)`: Optional random seed for sampling.
    - `timeout (int)`: Timeout passed to `fetch_polytopes`. Defaults to `60`.
    - `as_list (bool)`: Whether `fetch_polytopes` should return a list. Defaults to `True`.
    - `backend (str)`: Optional backend passed to `fetch_polytopes`.
    - `deterministic_glsm_basis (bool)`: Passed to `fetch_polytopes`. Defaults to `False`.
    - `dualize (bool)`: Passed to `fetch_polytopes`. Defaults to `False`.
    - `favorable (bool)`: Optional favorability filter.
    - `verbosity (int)`: Verbosity level. Defaults to `0`.

    **Yields:**

    - `CY_orientifold`: A Calabi-Yau orientifold satisfying the requested filters.

    """
    
    if only_nef_decomposition:
        for p in fetch_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,n_dual_points,n_facets,limit,samples,sample_seed,timeout,as_list,backend,deterministic_glsm_basis,dualize,favorable,verbosity):
            for xi in UF.inequivalent_Z2_actions(p.automorphisms(action="left")):
                O=CY_orientifold(p,xi)
                if O.yields_nef_decomposition():
                    yield O
    else:
        for p in fetch_polytopes(h11,h12,h13,h21,h22,h31,chi,lattice,dim,n_points,n_vertices,n_dual_points,n_facets,limit,samples,sample_seed,timeout,as_list,backend,deterministic_glsm_basis,dualize,favorable,verbosity):
            for xi in UF.inequivalent_Z2_actions(p.automorphisms(action="left")):
                O=CY_orientifold(p,xi)
                if only_regular:
                    if O.is_regular():
                        yield O 
                else:
                    yield O

def fetch_F_Theory_uplifts(only_regular: bool = True, only_nef_partition:bool=False,only_nef_decomposition: bool=False,h11: int = None,h12: int = None,h13: int = None,
    h21: int = None,h22: int = None,h31: int = None,chi: int = None,
    lattice: str = 'N',dim: int = 4,n_points: int = None,n_vertices: int = None,
    n_dual_points: int = None,n_facets: int = None,limit: int = 1000,
    samples: int = None,sample_seed: int = None,timeout: int = 60,
    as_list: bool = True,backend: str = None,
    deterministic_glsm_basis: bool = False,
    dualize: bool = False,
    favorable: bool = None,
    verbosity: int = 0):

    
    """
    **Description:**

    Iterates over Calabi-Yau orientifolds obtained from `fetch_orientifolds` and yields the associated `F_Theory_Uplift` objects. The output may optionally be restricted to regular orientifolds, orientifolds yielding a nef decomposition, or uplifts defining a nef partition.

    **Arguments:**

    - `only_regular (bool)`: Whether to use only regular orientifolds. Defaults to `True`.
    - `only_nef_partition (bool)`: Whether to yield only uplifts defining a nef partition. Defaults to `False`.
    - `only_nef_decomposition (bool)`: Whether to use only orientifolds yielding a nef decomposition. Defaults to `False`.
    - `h11, h12, h13, h21, h22, h31, chi`: Optional Hodge-number and Euler characteristic filters passed to `fetch_polytopes`.
    - `lattice (str)`: Lattice used by `fetch_polytopes`. Defaults to `"N"`.
    - `dim (int)`: Polytope dimension. Defaults to `4`.
    - `n_points (int)`: Optional filter on the number of lattice points.
    - `n_vertices (int)`: Optional filter on the number of vertices.
    - `n_dual_points (int)`: Optional filter on the number of dual points.
    - `n_facets (int)`: Optional filter on the number of facets.
    - `limit (int)`: Maximum number of polytopes fetched. Defaults to `1000`.
    - `samples (int)`: Optional number of random samples.
    - `sample_seed (int)`: Optional random seed for sampling.
    - `timeout (int)`: Timeout passed to `fetch_polytopes`. Defaults to `60`.
    - `as_list (bool)`: Whether `fetch_polytopes` should return a list. Defaults to `True`.
    - `backend (str)`: Optional backend passed to `fetch_polytopes`.
    - `deterministic_glsm_basis (bool)`: Passed to `fetch_polytopes`. Defaults to `False`.
    - `dualize (bool)`: Passed to `fetch_polytopes`. Defaults to `False`.
    - `favorable (bool)`: Optional favorability filter.
    - `verbosity (int)`: Verbosity level. Defaults to `0`.

    **Yields:**

    - `F_Theory_Uplift`: An F-theory uplift satisfying the requested filters.
    """
    
    if only_nef_partition:
        for O in fetch_orientifolds(only_nef_decomposition=True,h11=h11,h12=h12,h13=h13,h21=h21,h22=h22,h31=h31,chi=chi,lattice=lattice,dim=dim,n_points=n_points,n_vertices=n_vertices,n_dual_points=n_dual_points,n_facets=n_facets,limit=limit,samples=samples,sample_seed=sample_seed,timeout=timeout,as_list=as_list,backend=backend,deterministic_glsm_basis=deterministic_glsm_basis,dualize=dualize,favorable=favorable,verbosity=verbosity):
            F=F_Theory_Uplift(O)
            if F.is_nef_partition():
                yield F
    elif only_nef_decomposition:
        for O in fetch_orientifolds(only_nef_decomposition=True,h11=h11,h12=h12,h13=h13,h21=h21,h22=h22,h31=h31,chi=chi,lattice=lattice,dim=dim,n_points=n_points,n_vertices=n_vertices,n_dual_points=n_dual_points,n_facets=n_facets,limit=limit,samples=samples,sample_seed=sample_seed,timeout=timeout,as_list=as_list,backend=backend,deterministic_glsm_basis=deterministic_glsm_basis,dualize=dualize,favorable=favorable,verbosity=verbosity):
            yield F_Theory_Uplift(O)
    else:
        for O in fetch_orientifolds(only_regular=only_regular, only_nef_decomposition=False,h11=h11,h12=h12,h13=h13,h21=h21,h22=h22,h31=h31,chi=chi,lattice=lattice,dim=dim,n_points=n_points,n_vertices=n_vertices,n_dual_points=n_dual_points,n_facets=n_facets,limit=limit,samples=samples,sample_seed=sample_seed,timeout=timeout,as_list=as_list,backend=backend,deterministic_glsm_basis=deterministic_glsm_basis,dualize=dualize,favorable=favorable,verbosity=verbosity):
            yield F_Theory_Uplift(O)



def fetch_nef_partition_uplifts(
    h11=None,
    h21=None,
    limit=100,
    repo_id="jakobmoritz/F-theory_Uplifts_Nef-partitions",
    index_path=None,
    local_base=None,
    columns=None,
    as_pandas=True,
):
    """
    **Description:**

    Reads an indexed parquet dataset of F-theory uplifts and returns rows matching the specified Hodge-number filters. Data may be read either from the Hugging Face Hub or from a local dataset directory.

    **Arguments:**

    - `h11 (int or None)`: Desired value of `h^{1,1}`. If `None`, no `h11` restriction is applied. Defaults to `None`.
    - `h21 (int or None)`: Desired value of `h^{2,1}`. If `None`, no `h21` restriction is applied. Defaults to `None`.
    - `limit (int or None)`: Maximum number of rows to return. If `None`, all matching rows are returned. Defaults to `100`.
    - `repo_id (str)`: Hugging Face dataset repository ID. Defaults to `"jakobmoritz/F-theory_Uplifts_Nef-partitions"`.
    - `index_path (str or Path or None)`: Path to `hodge_index.parquet`. If `None`, the default index path is inferred from `repo_id` or `local_base`. Defaults to `None`.
    - `local_base (str or Path or None)`: Local dataset directory containing `data_by_h11/` and `index/`. If `None`, data are read from Hugging Face. Defaults to `None`.
    - `columns (list[str] or None)`: Columns to read from the parquet files. If `None`, all columns are read. Defaults to `None`.
    - `as_pandas (bool)`: Whether to return a pandas `DataFrame`. If `False`, returns a pyarrow `Table`. Defaults to `True`.

    **Raises:**

    - `ValueError`: Raised if neither `h11` nor `h21` is specified, or if the index is missing required row-group information.

    **Returns:**

    - `pandas.DataFrame` or `pyarrow.Table`: The matching dataset rows.

    """

    if h11 is None and h21 is None:
        raise ValueError("At least one of h11 or h21 must be provided.")

    if limit is not None and limit <= 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    if local_base is None:
        base = f"hf://datasets/{repo_id}"
        if index_path is None:
            index_path = f"{base}/index/hodge_index.parquet"
    else:
        base = Path(local_base)
        if index_path is None:
            index_path = base / "index" / "hodge_index.parquet"

    # Load small index file.
    index = pd.read_parquet(index_path)

    # Restrict index.
    mask = pd.Series(True, index=index.index)

    if h11 is not None:
        mask &= index["h11"] == int(h11)

    if h21 is not None:
        mask &= index["h21"] == int(h21)

    subindex = index.loc[mask].copy()

    if len(subindex) == 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    # Preserve natural Hodge ordering.
    sort_cols = []
    if "h11" in subindex.columns:
        sort_cols.append("h11")
    if "h21" in subindex.columns:
        sort_cols.append("h21")
    if "row_group" in subindex.columns:
        sort_cols.append("row_group")

    if sort_cols:
        subindex = subindex.sort_values(sort_cols)

    tables = []
    rows_collected = 0

    # Group by file to avoid reopening unnecessarily.
    for file, group in subindex.groupby("file", sort=False):
        if limit is not None and rows_collected >= limit:
            break

        if local_base is None:
            parquet_path = f"{base}/{file}"
        else:
            parquet_path = base / file

        pf = pq.ParquetFile(parquet_path)

        if "row_group" not in group.columns:
            raise ValueError(
                "The index does not contain a 'row_group' column. "
                "For efficient limited access, recreate hodge_index.parquet "
                "with row-group information."
            )

        row_groups = group["row_group"].astype(int).tolist()

        for rg in row_groups:
            if limit is not None and rows_collected >= limit:
                break

            table = pf.read_row_group(rg, columns=columns)

            # This should usually be unnecessary if the index is correct,
            # but keep it as a safety check.
            if h11 is not None or h21 is not None:
                df_check = table.select(
                    [c for c in ["h11", "h21"] if c in table.column_names]
                ).to_pandas()

                keep = pd.Series(True, index=df_check.index)

                if h11 is not None:
                    keep &= df_check["h11"] == int(h11)

                if h21 is not None:
                    keep &= df_check["h21"] == int(h21)

                if not keep.all():
                    table = table.filter(pa.array(keep.to_numpy()))

            if limit is not None:
                remaining = limit - rows_collected
                if table.num_rows > remaining:
                    table = table.slice(0, remaining)

            tables.append(table)
            rows_collected += table.num_rows

    if len(tables) == 0:
        empty = pa.table({})
        return empty.to_pandas() if as_pandas else empty

    result = pa.concat_tables(tables, promote_options="default")

    return result.to_pandas() if as_pandas else result