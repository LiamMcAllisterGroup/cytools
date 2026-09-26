---
title: Overview
---

# Documentation

CYTools provides six main Python classes containing various functions specific
to the objects they describe. These classes are the following:

- [`Polytope`](polytope.md): Handles all computations relating to lattice
  polytopes, such as the computation of lattice points and faces. When using
  reflexive polytopes, it also allows the computation of topological properties
  of the arising Calabi-Yau hypersurfaces that only depend on the polytope.
- [`PolytopeFace`](polytopeface.md): Handles all computations relating to faces
  of lattice polytopes.
- [`Triangulation`](triangulation.md): Handles triangulations of lattice
  polytopes. It can compute various properties of the triangulation, as well as
  construct a `ToricVariety` or `CalabiYau` object if the triangulation is
  suitable.
- [`ToricVariety`](toricvariety.md): Handles various computations relating to
  toric varieties. It can be used to compute intersection numbers and the Kähler
  cone, among other things.
- [`CalabiYau`](calabiyau.md): Handles various computations relating to the
  Calabi-Yau manifold itself. It can be used to compute intersection numbers and
  the toric Mori and Kähler cones, among other things.
- [`Cone`](cone.md): Handles all computations relating to rational polyhedral
  cones, such as cone duality and extremal ray computations. It is mainly used
  for the study of Kähler and Mori cones.

There are also a few more specialized classes and modules:

- [`HPolytope`](hpolytope.md): Polytopes specified by a list of inequalities
  (their H-representation) instead of a list of points.
- [`VectorConfiguration` and `Fan`](vector_config.md): Vector configurations and
  the (simplicial) fans obtained from their triangulations.
- [NTFE enumeration](ntfe.md): Enumeration of triangulations that are not
  two-face equivalent.
- [F-theory and orientifolds](f_theory.md): Datasets of orientifolds and
  F-theory uplifts.

Apart from the above, there are other miscellaneous functions that are
documented in the misc functions page. There are also a few configuration
options that can be found in the configuration page, and some experimental
features documented in the experimental features page.

<div class="cy-buttons" markdown>
[Misc functions](other.md){ .md-button .md-button--primary }
[Configuration](config.md){ .md-button .md-button--primary }
[Experimental features](experimental.md){ .md-button .md-button--primary }
</div>

CYTools is open-source software distributed under the
[GNU GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.txt). See the license
page for more details.

<div class="cy-buttons" markdown>
[License](license.md){ .md-button .md-button--primary }
</div>
