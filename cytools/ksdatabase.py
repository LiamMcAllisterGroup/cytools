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

"""
This module contains tools designed to extract data from the Kreuzer-Skarke
(KS) database.
"""

from cytools.polytope import Polytope
import requests
import numpy as np
import re


def read_polytopes(input, input_type="file", dualize=False):
    """
    Read the polytopes from a file or a string containing a list of polytopes
    in the format used in the Kreuzer-Skarke database.

    Args:
        input (string): Specifies the name of the file to read or the string
            containing the polytopes.
        input_type (string, optional, default="file"): Specifies whether to
            read from a file or from the input string.  Options are "file" or
            "string".
        dualize (boolean, optional, default=False): Flag that indicates whether
            to dualize all the polytopes before returning them.

    Returns:
        generator: A generator of Polytope objects
    """
    if input_type not in ["file", "string"]:
        raise Exception("\"input_type\" must be either \"file\" or \"string\"")
    if input_type == "file":
        in_file = open(input, "r")
        l = in_file.readline()
    else:
        in_string = input.split("\n")
        l = in_string[0]
        in_string.pop(0)
    while True:
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
            if m != 4:
                vert = vert.T
                h[1] = h[0]
                h[1] = "4"
            yield (Polytope(vert).dual() if dualize else Polytope(vert))
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


def fetch_polytopes(h11=None, h21=None, chi=None, lattice=None, n_points=None,
                    n_vertices=None, n_dual_points=None, n_facets=None,
                    limit=1000, timeout=60, dualize=False):
    """
    Fetch reflexive polytopes from the Kreuzer-Skarke database.  The data is
    pulled from website http://hep.itp.tuwien.ac.at/~kreuzer/CY/.

    Args:
        h11 (int, optional): Specifies the Hodge number h^{1,1} of the Calabi-Yau hypersurface.
        h21 (int, optional): Specifies the Hodge number h^{2,1} of the Calabi-Yau hypersurface.
        chi (int, optional): Specifies the Euler characteristic of the Calabi-Yau hypersurface.
        lattice (string, optional): Specifies the lattice on which the polytope is
            defined. Options are 'N' and 'M'. Has to be specified if h11, h21 or chi
            are specified.
            If lattice = 'N': the Hodge numbers h11 and h21, and the Euler characteristic
            chi are those of the Calabi-Yau obtained as the anticanonical hypersurface in
            the toric variety given by a desingularization of the face fan of the polytope.
            If lattice = 'M': the Hodge numbers h11 and h21, and the Euler characteristic
            chi are those of the Calabi-Yau obtained as the anticanonical hypersurface in
            the toric variety given by a desingularization of the normal fan of the polytope.
        n_points (int, optional): Specifies the number of lattice points of the
            desired polytopes.
        n_vertices (int, optional): Specifies the number of vertices of the
            desired polytopes.
        n_dual_points (int, optional): Specifies the number of points of the
            dual polytopes of the desired polytopes.
        n_facets (int, optional): Specifies the number of facets of the desired
            polytopes.
        limit (int, optional, default=1000): Specifies the maximum number of
            fetched polytopes.
        timeout (int, optional, default=60): Specifies the maximum number of
            seconds to wait for the server to return the data.
        dualize (boolean, optional, default=False): Flag that indicates whether
            to dualize all the polytopes before returning them.

    Returns:
        generator: A generator of Polytope objects
    """
    # Process input
    h11_ks = None
    h21_ks = None
    chi_ks = None
    if lattice=='M':
        h11_ks = h11
        h21_ks = h21
        chi_ks = chi
    elif lattice=='N':
        h11_ks = h21
        h21_ks = h11
        if chi is not None:
            chi_ks = -chi
    elif lattice is None:
        if (h11 is not None or
            h21 is not None or
            chi is not None):
            raise Exception("Lattice has to be specified when h11, h21 or chi are specified."
                " The options are: 'N' and 'M'.")
    else:
        raise Exception("Invalid lattice. The options are: 'N' and 'M'.")

    # Check input consistency
    if (chi_ks is not None and
        h11_ks is not None and
        h21_ks is not None and
        chi_ks != 2*(h11_ks-h21_ks)):
        raise Exception("Inconsistent Euler characteristic setting.")
    variables = [h11_ks, h21_ks, n_points, n_vertices, n_dual_points, n_facets,
                 chi_ks, limit]
    names = ["h11", "h12", "M", "V", "N", "F", "chi", "L"]
    parameters = {n:str(v) for n, v in zip(names, variables) if v is not None}
    r = requests.get("http://quark.itp.tuwien.ac.at/cgi-bin/cy/cydata.cgi",
                     params=parameters, timeout=timeout)
    return read_polytopes(r.text, input_type="string", dualize=dualize)
