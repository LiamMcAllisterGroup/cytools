import unittest
import os
import json
import cytools as cyt
from cytools import config
import numpy as np

test_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/polytope_data.json"
)


def compute_positivity_data(poly, triangulation_backend, intnum_backend):
    frst = poly.triangulate(backend=triangulation_backend, make_star=True)
    cy = frst.get_cy()
    cy.intersection_numbers(backend=intnum_backend)
    kahler_cone = kahler_cone = cy.toric_kahler_cone()
    kcone_tip = kahler_cone.tip_of_stretched_cone(1)
    XVol = cy.compute_cy_volume(kcone_tip)
    div_vols = cy.compute_divisor_volumes(kcone_tip, in_basis=False)
    Kinv = cy.compute_inverse_kahler_metric(kcone_tip)
    Kinv_eigvals = np.linalg.eigvals(Kinv)
    K_eigvals = sorted([1 / eig for eig in Kinv_eigvals])
    return {
        "CY_Volume": XVol,
        "Divisor_Volumes": div_vols,
        "Kahler_eigenvalues": K_eigvals,
    }


class Kahler_Positivity_Test(unittest.TestCase):
    def setUp(self):
        with open(test_data_path, "r") as fp:
            self.test_data = json.load(fp)
            self.test_data = (
                self.test_data[:40] + self.test_data[40:-2:20] + self.test_data[-2:]
            )

    def test_positivity_cgal(self):
        try:
            from tqdm import tqdm

            data_iter = tqdm(self.test_data)
        except:
            data_iter = self.test_data
        for doc in data_iter:
            if doc["N_favorable"] == "True":
                vertices = eval(doc["vertices"])
                poly = cyt.Polytope(vertices)
                options = (
                    (p, pp)
                    for p in ("qhull", "cgal", "topcom")
                    for pp in ("all", "sksparse", "scipy")
                )
                for opt in options:
                    positivity_data = compute_positivity_data(poly, opt[0], opt[1])
                    self.assertTrue(
                        positivity_data["CY_Volume"] > 0,
                        msg="Negative CY Volume. Options:"
                        + str(opt)
                        + ": \n"
                        + str(positivity_data["CY_Volume"]),
                    )
                    self.assertTrue(
                        all(i > 0 for i in positivity_data["Divisor_Volumes"]),
                        msg="Negative divisor volumes. Options:"
                        + str(opt)
                        + ": \n"
                        + str(positivity_data["Divisor_Volumes"]),
                    )
                    self.assertTrue(
                        all(i > 0 for i in positivity_data["Kahler_eigenvalues"]),
                        msg="Negative Kahler metric eigenvalues. Options:"
                        + str(opt)
                        + ": \n"
                        + str(positivity_data["Kahler_eigenvalues"]),
                    )


if __name__ == "__main__":
    unittest.main()
