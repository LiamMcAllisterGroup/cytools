import unittest
import os
import json
from scipy.sparse import dok_matrix
import cytools as cyt
import numpy as np

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/cy_data.json')

class Intersection_Number_Sage_Comparison_Test(unittest.TestCase):
    def setUp(self):
        with open(test_data_path,'r') as fp:
            self.test_data = json.load(fp)

    def test_intersection_numbers(self):
        print("Testing: Intersection numbers...")
        try:
            from tqdm import tqdm
            data_iter = tqdm(self.test_data)
        except:
            data_iter = self.test_data
        for doc in data_iter:
            vertices = eval(doc['vertices'])
            sage_points = [list(i) for i in eval(doc['rays'])]
            sage_favorable = eval(doc['N_favorable'])
            if sage_favorable:
                poly = cyt.Polytope(vertices)
                cyt_points = poly.boundary_points_not_interior_to_facets()
                frst = poly.triangulate(make_star=True)
                cy = frst.get_cy()
                cyt_int_nums_origin_raw = cy.intersection_numbers(in_basis=False,
                 zero_as_anticanonical=True)
                cyt_int_nums_raw = [[ii[0]-1, ii[1]-1, ii[2]-1] + [cyt_int_nums_origin_raw[ii]]
                 for ii in cyt_int_nums_origin_raw if ii[0]!=0 and ii[1]!=0 and ii[2]!=0]
                sage_int_nums_raw = eval(doc['intersection_numbers'])
                sage_index = {tuple(pp):p for p,pp in enumerate(sage_points)}
                cyt_index = {tuple(pp):p for p,pp in enumerate(cyt_points)}
                transform_dict_1 = {sage_index[tuple(pp)]:cyt_index[tuple(pp)] for pp in cyt_points}
                sage_int_nums = sorted([sorted(
                    [transform_dict_1[ii[0]],transform_dict_1[ii[1]],transform_dict_1[ii[2]]])
                     + [ii[3]] for ii in sage_int_nums_raw])
                cyt_int_nums = sorted([ii for ii in cyt_int_nums_raw])
                self.assertTrue(sage_int_nums==cyt_int_nums,
                        msg="Intersection numbers don't match. \n Sage: \n"
                            + str(sage_int_nums) + "\n Cytools: \n" + str(cyt_int_nums))

    def test_sr_ideal(self):
        print("Testing: Stanley-Reisner ideal...")
        try:
            from tqdm import tqdm
            data_iter = tqdm(self.test_data[5:])
        except:
            data_iter = self.test_data[5:]
        for doc in data_iter:
            vertices = eval(doc['vertices'])
            sage_points = [list(i) for i in eval(doc['rays'])]
            sage_favorable = eval(doc['N_favorable'])
            if sage_favorable:
                poly = cyt.Polytope(vertices)
                cyt_points = poly.boundary_points_not_interior_to_facets()
                frst = poly.triangulate(make_star=True)
                cy = frst.get_cy()
                cyt_sr_ideal_raw = frst.sr_ideal()
                sage_sr_ideal_raw = eval(doc['sr_ideal'])
                sage_index = {tuple(pp):p for p,pp in enumerate(sage_points)}
                cyt_index = {tuple(pp):p for p,pp in enumerate(cyt_points)}
                transform_dict_1 = {sage_index[tuple(pp)]:cyt_index[tuple(pp)] for pp in cyt_points}
                sage_sr_ideal = sorted([sorted([transform_dict_1[ss]+1 for ss in s]) for s in sage_sr_ideal_raw])
                cyt_sr_ideal = sorted([sorted(s) for s in cyt_sr_ideal_raw])
                self.assertTrue(sage_sr_ideal==cyt_sr_ideal,
                        msg="Stanley-Reisner ideals don't match. \n Sage: \n"
                            + str(sage_sr_ideal) + "\n Cytools: \n" + str(cyt_sr_ideal))

    def test_second_chern_class(self):
        print("Testing: Second Chern class...")
        try:
            from tqdm import tqdm
            data_iter = tqdm(self.test_data)
        except:
            data_iter = self.test_data
        for doc in data_iter:
            vertices = eval(doc['vertices'])
            sage_points = [list(i) for i in eval(doc['rays'])]
            sage_favorable = eval(doc['N_favorable'])
            if sage_favorable:
                poly = cyt.Polytope(vertices)
                cyt_points = poly.boundary_points_not_interior_to_facets()
                frst = poly.triangulate(make_star=True)
                cy = frst.get_cy()
                sage_c2_raw = eval(doc['c2'])
                cyt_c2 = cy.second_chern_class(in_basis=False, include_origin=False).tolist()
                sage_index = {tuple(pp):p for p,pp in enumerate(sage_points)}
                cyt_index = {tuple(pp):p for p,pp in enumerate(cyt_points)}
                transform_dict = {cyt_index[tuple(pp)]:sage_index[tuple(pp)] for pp in cyt_points}
                sage_c2 = [sage_c2_raw[transform_dict[c]] for c in range(len(sage_c2_raw))]
                self.assertTrue(sage_c2==cyt_c2,
                        msg="Second Chern class doesn't match. \n Sage: \n"
                            + str(sage_c2) + "\n Cytools: \n" + str(cyt_c2))
                
if __name__ == '__main__':
    unittest.main()