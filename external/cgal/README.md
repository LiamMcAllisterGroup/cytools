# CGAL triangulations

This folder contains some code that uses the Computer Geometry Algorithms Library (CGAL) to produce regular triangulations from given heights or weights (or Delaunay triangulations in the absence thereof). The full documentation of the library can be found [here](https://doc.cgal.org/latest/Triangulation/index.html#Chapter_Triangulations), along with installation instructions. The easiest way to get this code working is by building the ```cytools``` Docker image.

## Usage

After compiling, a triangulation can be obtained by running ```triangulate```, while passing a list of points and heights/weights. The input standard format of an array of points in square brackets, followed by an array of heights/weights in parenthesis.

Note that the dimension of the points is hardcoded into the code, so it will be needed to recompile for dimensions different to four. Additionally, it is also hardcoded whether it interprets the second array as heights or weights. Since the heights are typically easier to work with, this is the default.

This is an example input file ```input.dat``` of a polytope with $h^{1,1}=5$.
```
[[-1, 0, 0, 0], [1, -1, 1, 0], [-1, 0, -1, 0], [-1, 0, -1, -1], [-1, 0, 0, -1], [1, 0, -1, 2], [-1, 1, -1, -1], [0, 0, 1, -1], [0, 0, -1, 1], [0, 0, 0, 0]]
(4.1, 5.2, -5.3, 2.4, 0.1, 1.2, 1.0, 0.0, 0.4, 2.3)
```
The triangulation is obtained by running
```bash
./triangulate < input.dat 1>output.dat
```
