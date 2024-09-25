import unittest
import os
import json
import numpy as np
import cytools as cyt
import itertools

test_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/polytope_data.json"
)


def compute_glsm_data(
    poly,
    dual_poly,
    include_origin,
    include_points_interior_to_facets,
    integral,
):
    glsm = poly.glsm_charge_matrix(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
    )
    basis = poly.glsm_basis(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
        integral=integral,
    )
    lin_rels = poly.glsm_linear_relations(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
    )
    dual_glsm = dual_poly.glsm_charge_matrix(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
    )
    dual_basis = dual_poly.glsm_basis(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
        integral=integral,
    )
    dual_lin_rels = dual_poly.glsm_linear_relations(
        include_origin=include_origin,
        include_points_interior_to_facets=include_points_interior_to_facets,
    )
    return {
        "glsm": glsm,
        "basis": basis,
        "linear_relations": lin_rels,
        "dual_glsm": dual_glsm,
        "dual_basis": dual_basis,
        "dual_linear_relations": dual_lin_rels,
    }


def compute_point_data(
    poly, dual_poly, include_origin, include_points_interior_to_facets
):
    if include_origin and include_points_interior_to_facets:
        poly_points = poly.points()
        dual_poly_points = dual_poly.points()
    elif include_origin and not include_points_interior_to_facets:
        poly_points = poly.points_not_interior_to_facets()
        dual_poly_points = dual_poly.points_not_interior_to_facets()
    elif not include_origin and include_points_interior_to_facets:
        poly_points = poly.boundary_points()
        dual_poly_points = dual_poly.boundary_points()
    elif not include_origin and not include_points_interior_to_facets:
        poly_points = poly.boundary_points_not_interior_to_facets()
        dual_poly_points = dual_poly.boundary_points_not_interior_to_facets()
    else:
        raise
    return {"poly_points": poly_points, "dual_poly_points": dual_poly_points}


class GLSM_Consistency_Test(unittest.TestCase):
    def setUp(self):
        with open(test_data_path, "r") as fp:
            self.test_data = json.load(fp)
            self.test_data = (
                self.test_data[:40] + self.test_data[40:-5:20] + self.test_data[-5:]
            )

    def test_glsm(self):
        print("Testing: GLSM charge matrix...")
        try:
            from tqdm import tqdm

            data_iter = tqdm(self.test_data)
        except:
            data_iter = self.test_data
        for doc in data_iter:
            vertices = eval(doc["vertices"])
            poly = cyt.Polytope(vertices)
            dual_poly = poly.dual()
            for opt in itertools.product((True, False), repeat=3):
                glsm_data = compute_glsm_data(
                    poly,
                    dual_poly,
                    include_origin=opt[0],
                    include_points_interior_to_facets=opt[1],
                    integral=opt[2],
                )
                point_data = compute_point_data(
                    poly,
                    dual_poly,
                    include_origin=opt[0],
                    include_points_interior_to_facets=opt[1],
                )
                self.assertTrue(
                    np.all(np.dot(glsm_data["glsm"], point_data["poly_points"]) == 0),
                    msg="Incorrect GLSM charge matrix. Parameters:"
                    + str(opt)
                    + "\n Vertices of the polytope:"
                    + str(vertices),
                )
                self.assertTrue(
                    np.linalg.slogdet(glsm_data["glsm"].T[glsm_data["basis"]].T)[1]
                    > -10,
                    msg="Inconsistent GLSM basis. Parameters:"
                    + str(opt)
                    + "\n Vertices of the polytope:"
                    + str(vertices)
                    + "basis:"
                    + str(glsm_data["basis"]),
                )
                self.assertTrue(
                    np.all(
                        np.dot(glsm_data["linear_relations"], glsm_data["glsm"].T) == 0
                    ),
                    msg="Incorrect linear relations. Parameters:"
                    + str(opt)
                    + "\n Vertices of the polytope:"
                    + str(vertices),
                )
                self.assertTrue(
                    np.all(
                        np.dot(
                            glsm_data["dual_glsm"],
                            point_data["dual_poly_points"],
                        )
                        == 0
                    ),
                    msg="Incorrect GLSM charge matrix of dual polytope. Parameters:"
                    + str(opt)
                    + "\n Vertices of the original polytope:"
                    + str(vertices),
                )
                self.assertTrue(
                    np.linalg.slogdet(
                        glsm_data["dual_glsm"].T[glsm_data["dual_basis"]].T
                    )[1]
                    > -10,
                    msg="Inconsistent GLSM basis of dual polytope. Parameters:"
                    + str(opt)
                    + "\n Vertices of the original polytope:"
                    + str(vertices),
                )
                self.assertTrue(
                    np.all(
                        np.dot(
                            glsm_data["dual_linear_relations"],
                            glsm_data["dual_glsm"].T,
                        )
                        == 0
                    ),
                    msg="Incorrect linear relations of dual polytope. Parameters:"
                    + str(opt)
                    + "\n Vertices of the original polytope:"
                    + str(vertices),
                )


if __name__ == "__main__":
    unittest.main()
