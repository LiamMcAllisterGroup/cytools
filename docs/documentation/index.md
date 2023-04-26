---
id: overview
title: Overview
slug: /documentation/
---

import Link from '@docusaurus/Link';
import styles from './index.module.css';

CYTools provides six main Python classes containing various functions specific to the object they describe. These classes are the following:

* [`Polytope`](./polytope) This class handles all computations relating to lattice polytopes, such as
the computation of lattice points and faces. When using reflexive
polytopes, it also allows the computation of topological properties of the
arising Calabi-Yau hypersurfaces that only depend on the polytope.

* [`PolytopeFace`](./polytopeface) This class handles all computations relating to faces of lattice polytopes.

* [`Triangulation`](./triangulation) This class handles triangulations of lattice polytopes. It can compute
various properties of the triangulation, as well as construct a
ToricVariety or CalabiYau object if the triangulation is suitable.

* [`ToricVariety`](./toricvariety) This class handles various computations relating to toric varieties.
It can be used to compute intersection numbers and the Kähler cone, among
other things.

* [`CalabiYau`](./calabiyau) This class handles various computations relating to the Calabi-Yau manifold
itself. It can be used to compute intersection numbers and the toric Mori and
Kähler cones, among other things.

* [`Cone`](./cone) This class handles all computations relating to rational polyhedral cones,
such cone duality and extremal ray computations. It is mainly used for the
study of Kähler and Mori cones.

Apart from the above classes there are other miscellaneous functions that we document in the misc functions page. There are also a few configuration options that can be found in the configuration page, and some experimental features documented in the experimental features page.


<div className={styles.buttons2}>
  <centered>
    <Link
    className="button button--primary"
    to="./other">
    Misc functions
    </Link>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <Link
    className="button button--primary"
    to="./config">
    Configuration
    </Link>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <Link
    className="button button--primary"
    to="./experimental">
    Experimental features
    </Link>
  </centered>
</div>
<br></br>

CYTools is open-source software distributed under the [GNU GPL3 license](https://www.gnu.org/licenses/gpl-3.0.txt). See the license page for more details.

<div className={styles.buttons2}>
  <centered>
    <Link
    className="button button--primary"
    to="./license">
    License
    </Link>
  </centered>
</div>
