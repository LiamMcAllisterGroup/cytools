import unittest
import os
import json
from scipy.sparse import dok_matrix
import cytools as cyt
import numpy as np

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/polytope_data.json')

class Polytope_Sage_Comparison_Test(unittest.TestCase):
    def setUp(self):
        with open(test_data_path,'r') as fp:
            self.test_data = json.load(fp)

    def test_polytope_data(self):
        print("Testing: Polytope data...")
        def points_to_indices(points_in, points):
            pts_dict = {tuple(ii):i for i,ii in enumerate(points)}
            return [pts_dict[tuple(pt)] for pt in points_in]
        try:
            from tqdm import tqdm
            data_iter = tqdm(self.test_data)
        except:
            data_iter = self.test_data
        for doc in data_iter:
            vertices = eval(doc['vertices'])
            poly = cyt.Polytope(vertices)
            dual_poly = poly.dual()
            points = poly.points()
            dual_points = dual_poly.points()

            sage_points = [list(i) for i in eval(doc['N_points'])]
            sage_dual_points = [list(i) for i in eval(doc['M_points'])]
            sage_favorable = eval(doc['N_favorable'])
            sage_dual_favorable = eval(doc['M_favorable'])

            sage_faces_0_pts = [[sage_points[ii] for ii in i] for i in eval(doc['N_faces_0'])]
            sage_faces_1_pts = [[sage_points[ii] for ii in i] for i in eval(doc['N_faces_1'])]
            sage_faces_2_pts = [[sage_points[ii] for ii in i] for i in eval(doc['N_faces_2'])]
            sage_faces_3_pts = [[sage_points[ii] for ii in i] for i in eval(doc['N_faces_3'])]
            sage_dual_faces_0_pts = [[sage_dual_points[ii] for ii in i] for i in eval(doc['M_faces_0'])]
            sage_dual_faces_1_pts = [[sage_dual_points[ii] for ii in i] for i in eval(doc['M_faces_1'])]
            sage_dual_faces_2_pts = [[sage_dual_points[ii] for ii in i] for i in eval(doc['M_faces_2'])]
            sage_dual_faces_3_pts = [[sage_dual_points[ii] for ii in i] for i in eval(doc['M_faces_3'])]

            sage_faces_0 = sorted([sorted(points_to_indices(f, points)) for f in sage_faces_0_pts])
            sage_faces_1 = sorted([sorted(points_to_indices(f, points)) for f in sage_faces_1_pts])
            sage_faces_2 = sorted([sorted(points_to_indices(f, points)) for f in sage_faces_2_pts])
            sage_faces_3 = sorted([sorted(points_to_indices(f, points)) for f in sage_faces_3_pts])
            sage_dual_faces_0 = sorted([sorted(points_to_indices(f, dual_points)) for f in sage_dual_faces_0_pts])
            sage_dual_faces_1 = sorted([sorted(points_to_indices(f, dual_points)) for f in sage_dual_faces_1_pts])
            sage_dual_faces_2 = sorted([sorted(points_to_indices(f, dual_points)) for f in sage_dual_faces_2_pts])
            sage_dual_faces_3 = sorted([sorted(points_to_indices(f, dual_points)) for f in sage_dual_faces_3_pts])
           
            faces_0 = sorted([sorted(poly.points_to_indices(f.points())) for f in poly.faces(0)])
            faces_1 = sorted([sorted(poly.points_to_indices(f.points())) for f in poly.faces(1)])
            faces_2 = sorted([sorted(poly.points_to_indices(f.points())) for f in poly.faces(2)])
            faces_3 = sorted([sorted(poly.points_to_indices(f.points())) for f in poly.faces(3)])            
            dual_faces_0 = sorted([sorted(dual_poly.points_to_indices(f.dual().points())) for f in poly.faces(0)])
            dual_faces_1 = sorted([sorted(dual_poly.points_to_indices(f.dual().points())) for f in poly.faces(1)])
            dual_faces_2 = sorted([sorted(dual_poly.points_to_indices(f.dual().points())) for f in poly.faces(2)])
            dual_faces_3 = sorted([sorted(dual_poly.points_to_indices(f.dual().points())) for f in poly.faces(3)])
            favorable = poly.is_favorable(lattice='N')
            dual_favorable = dual_poly.is_favorable(lattice='N')

            self.assertTrue(sorted(points.tolist())==sorted(sage_points),
                msg="Polytope points don't match. \n Sage: \n" + str(sage_points)
                     + "\n Cytools: \n" + str(points))
            self.assertTrue(sorted(dual_points.tolist())==sorted(sage_dual_points),
                msg="Dual polytope points don't match. \n Sage: \n" + str(sage_dual_points)
                     + "\n Cytools: \n" + str(dual_points))
            self.assertTrue(faces_0==sage_faces_0,
                msg="Vertices don't match. \n Sage: \n" + str(sage_faces_0)
                 + "\n Cytools: \n" + str(faces_0))
            self.assertTrue(faces_1==sage_faces_1,
                msg="1-faces don't match. \n Sage: \n" + str(sage_faces_1)
                     + "\n Cytools: \n" + str(faces_1))
            self.assertTrue(faces_2==sage_faces_2,
                msg="2-faces don't match. \n Sage: \n" + str(sage_faces_2)
                     + "\n Cytools: \n" + str(faces_2))
            self.assertTrue(faces_3==sage_faces_3,
                msg="3-faces don't match. \n Sage: \n" + str(sage_faces_3)
                     + "\n Cytools: \n" + str(faces_3))
            self.assertTrue(dual_faces_0==sage_dual_faces_0,
                msg="Dual Vertices don't match. \n Sage: \n" + str(sage_dual_faces_0)
                     + "\n Cytools: \n" + str(dual_faces_0))
            self.assertTrue(dual_faces_1==sage_dual_faces_1,
                msg="Dual 1-faces don't match. \n Sage: \n" + str(sage_dual_faces_1)
                     + "\n Cytools: \n" + str(dual_faces_1))
            self.assertTrue(dual_faces_2==sage_dual_faces_2,
                msg="Dual 2-faces don't match. \n Sage: \n" + str(sage_dual_faces_2)
                     + "\n Cytools: \n" + str(dual_faces_2))
            self.assertTrue(dual_faces_3==sage_dual_faces_3,
                msg="Dual 3-faces don't match. \n Sage: \n" + str(sage_dual_faces_3)
                     + "\n Cytools: \n" + str(dual_faces_3))
            self.assertTrue(favorable==sage_favorable,
                msg="Polytope favorability doesn't match. \n Sage: \n" + str(sage_favorable)
                     + "\n Cytools: \n" + str(favorable))
            self.assertTrue(dual_favorable==sage_dual_favorable,
                msg="Dual polytope favorability doesn't match. \n Sage: \n" + str(sage_dual_favorable)
                     + "\n Cytools: \n" + str(dual_favorable))

if __name__ == '__main__':
    unittest.main()