---
id: advanced
title: Advanced usage
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

We have compiled some guidelines for people that want to get the best performance out of CYTools or want to perform computationally intensive tasks such as a large scan over the Kreuzer-Skarke database. We discuss some performance tips and describe how to tinker with the functionality of CYTools.

## Performance tips

### Caching

CYTools classes cache most of the hard computations so that results don't have to be recomputed if they are needed again. While this gives a significant speedup during typical usage, one should also be aware of possible complications.

The cached data is stored as hidden attributes of the classes, which are never exposed by any function, and they should not be directly accessed unless you know what you are doing. Accidentally modifying one of these hidden attributes may result in following computations being wrong.

There is also the issue of memory management. If a large number of polytopes or triangulations need to be used, then it is good to make sure that not everything is kept in memory at the same time. For instance, let's look at the following example.

```python
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100, as_list=True)

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.toric_mori_cone()
    # Do some other computations
```

In the above example we have a list of 100 triangulations that we use to construct the corresponding Calabi-Yau hypersurfaces and do some computations. Each Triangulation object caches its corresponding CalabiYau object, which in turn caches its intersection numbers and whatever else we compute. So if our list of triangulations is too big then we might end up running out of memory. We could solve this in two ways.

1. We could use the ```clear_cache``` function that all main classes have to clear the results of any previous computation.
``` python {12}
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100, as_list=True)

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.ambient_mori_cone()
    # Do some other computations
    t.clear_cache()
```

2. (Preferable alternative) We could rewrite our code so that only one triangulation is kept in memory at a time. This is exactly why by default various functions in CYTools return generator objects instead of lists! The example that we considered can simply be fixed by removing the `as_list=True` argument.
``` python {5}
import numpy as np
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100) # Now triangs is a generator instead of a list

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.ambient_mori_cone()
    # Do some other computations
```

### Identify equivalent Calabi-Yau hypersurfaces

It is common for multiple triangulations of a polytope to result in Calabi-Yau hypersurfaces that are equivalent. CYTools has functions designed to identify these cases when the equivalence is simple to check. In particular, it can check whether the restriction of the triangulations to codimension-2 faces of the polytope are equivalent or related by a polytope automorphism. The function to check for this equivalence is the following.
```python
cy1.is_trivially_equivalent(cy2)
# returns a boolean value
```
If the function returns `True` then the two hypersurfaces are certainly equivalent, but if it returns `False` it could still be that they are equivalent in a more complicated way.

When scanning a large set of Calabi-Yaus it is useful to check for this equivalence to make sure that you don't waste computing power on repeated computations. The set of Calabi-Yaus that are not trivially equivalent can be found simply by constructing a set of them in Python. Here is an example of how to do this.
```python {8,9}
from cytools import Polytope

p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
triangs = p.all_triangulations(as_list=True) # Remember that as_list=True is not always a good idea, but it is convenient for this example

all_cys = [t.get_cy() for t in triangs]

cys_not_triv_eq = set(all_cys)
cys_not_triv_eq = {t.get_cy() for t in triangs} # This is another way to construct the set

print(len(all_cys),len(cys_not_triv_eq))
# 102 5
```

In this example we can see that the set of Calabi-Yaus was significantly reduced. If we had iterated over all of the Calabi-Yaus we would have wasted a lot of time since we would have repeated the same computation over and over.

### Mosek

There are various computations that require finding a point in a cone. This task is particularly difficult when it is a high-dimensional problem and the cone is very narrow. The CYTools Docker image bundles some good tools to perform these computations, such as [OR-Tools](https://developers.google.com/optimization). However, for high-dimensional cases, the [Mosek](https://www.mosek.com/) optimizer is significantly faster. Therefore, if you plan to scan over a large number of Calabi-Yaus with large $h^{1,1}$, it is worth using Mosek.

Unfortunately, Mosek is proprietary software and requires a license to use it. Fortunately, it is easy to obtain and activate a free academic license and we have listed the instructions to do so here.

1. You can find the form to request a free academic license at [this link](https://www.mosek.com/products/academic-licenses/).

2. Follow the instructions to request a personal academic license. Make sure to use your university email to have the request approved quickly.

3. You will receive an email with a `mosek.lic` file. Simply copy this license file into the following path (you will need to create the `mosek` directory).

<Tabs>
<TabItem value="linux" label="Linux" default>

```
/home/YOUR_USER_NAME/mosek/mosek.lic
```

</TabItem>
<TabItem value="macos" label="macOS">

```
/Users/YOUR_USER_NAME/mosek/mosek.lic
```

</TabItem>
<TabItem value="windows" label="Windows">

```
C:\Users\YOUR_USER_NAME\mosek\mosek.lic
```

</TabItem>
</Tabs>

:::note
If you prefer to place the license file in a separate directory or if you run CYTools with a non-standard mounted location you will need to configure CYTools to look for the Mosek license at the appropriate place. To do this look at the [configuration page](../documentation/config.md).
:::

In order to check if your Mosek license is properly activated you can run the following commands.

```python
from cytools import config
config.check_mosek_license()
config.mosek_is_activated()
```
If the last function returns `True` then you are all set. Also, if you have the Mosek license in the recommended path then it is not necessary to do this check every time, as the `check_mosek_license` function is run every time that CYTools is started.
