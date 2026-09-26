---
title: Advanced usage
---

# Advanced usage

We have compiled some guidelines for people who want to get the best performance out of CYTools, or who want to perform computationally intensive tasks such as a large scan over the Kreuzer-Skarke database.

## Performance tips

### Caching

CYTools classes cache the results of most expensive computations, so that they don't have to be recomputed if they are needed again. While this gives a significant speedup during typical usage, one should also be aware of possible complications.

The cached data is stored in private attributes of the objects (the ones whose names start with an underscore), which should not be accessed directly unless you know what you are doing. Accidentally modifying one of these attributes may result in subsequent computations being wrong.

There is also the issue of memory management. If a large number of polytopes or triangulations need to be used, then it is good to make sure that not everything is kept in memory at the same time. For instance, let's look at the following example.

```python
from cytools import Polytope

p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
triangs = p.random_triangulations_fast(N=100, as_list=True)

for t in triangs:
    cy = t.get_cy()
    intnums = cy.intersection_numbers()
    m = cy.toric_mori_cone()
    # Do some other computations
```

In the above example, we have a list of 100 triangulations that we use to construct the corresponding Calabi-Yau hypersurfaces and do some computations. Each `Triangulation` object caches its corresponding `CalabiYau` object, which in turn caches its intersection numbers and whatever else we compute. So if our list of triangulations is too big, then we might end up running out of memory. We could solve this in two ways.

1. We could use the `clear_cache` function that all main classes have to clear the results of any previous computation.

    ```python hl_lines="11"
    from cytools import Polytope

    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
    triangs = p.random_triangulations_fast(N=100, as_list=True)

    for t in triangs:
        cy = t.get_cy()
        intnums = cy.intersection_numbers()
        m = cy.toric_mori_cone()
        # Do some other computations
        t.clear_cache()
    ```

2. (Preferable alternative) We could rewrite our code so that only one triangulation is kept in memory at a time. This is exactly why various functions in CYTools return generators instead of lists by default! The example that we considered can simply be fixed by removing the `as_list=True` argument.

    ```python hl_lines="4"
    from cytools import Polytope

    p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]).dual()
    triangs = p.random_triangulations_fast(N=100) # Now triangs is a generator instead of a list

    for t in triangs:
        cy = t.get_cy()
        intnums = cy.intersection_numbers()
        m = cy.toric_mori_cone()
        # Do some other computations
    ```

!!! note

    Not every function returns a generator by default. For example, [`fetch_polytopes`][cytools.utils.fetch_polytopes] returns a list unless `as_list=False` is passed. Check the documentation of each function to see what it returns.

### Identify equivalent Calabi-Yau hypersurfaces

It is common for multiple triangulations of a polytope to result in Calabi-Yau hypersurfaces that are equivalent. CYTools has functions designed to identify these cases when the equivalence is simple to check. In particular, it can check whether the restrictions of the triangulations to the codimension-2 faces of the polytope are the same, or related by a polytope automorphism. The function to check for this equivalence is the following.

```python
cy1.is_trivially_equivalent(cy2)
# returns a boolean value
```

If the function returns `True`, then the two hypersurfaces are certainly equivalent, but if it returns `False` they could still be equivalent in a more complicated way.

When scanning over a large set of Calabi-Yaus, it is useful to check for this equivalence to make sure that you don't waste computing power on repeated computations. The Calabi-Yaus that are not trivially equivalent to each other can be found simply by constructing a Python set of them, since two Calabi-Yaus are considered equal when they are trivially equivalent. Here is an example of how to do this.

```python hl_lines="8 9"
from cytools import Polytope

p = Polytope([[-1,0,0,0],[-1,1,0,0],[-1,0,1,0],[2,-1,0,-1],[2,0,-1,-1],[2,-1,-1,-1],[-1,0,0,1],[-1,1,0,1],[-1,0,1,1]])
triangs = p.all_triangulations(as_list=True) # Remember that as_list=True is not always a good idea, but it is convenient for this example

all_cys = [t.get_cy() for t in triangs]

cys_not_triv_eq = set(all_cys)
cys_not_triv_eq = {t.get_cy() for t in triangs} # This is another way to construct the set

print(len(all_cys), len(cys_not_triv_eq))
# 102 5
```

In this example, we can see that the set of Calabi-Yaus was significantly reduced. If we had iterated over all of the Calabi-Yaus, we would have wasted a lot of time repeating the same computations over and over.

### Mosek

Various computations require finding a point in a cone, such as the tip of the stretched Kähler cone. This task is particularly difficult when the problem is high-dimensional and the cone is very narrow. By default, CYTools uses open-source optimizers such as [OSQP](https://osqp.org/) and [HiGHS](https://highs.dev/) for these computations. However, for high-dimensional cases, the [Mosek](https://www.mosek.com/) optimizer is significantly faster, and CYTools automatically uses it for cones in 25 or more dimensions when it is available. Therefore, if you plan to scan over a large number of Calabi-Yaus with large $h^{1,1}$, it is worth using Mosek.

Mosek is included in the conda environment created by the installation instructions. If you installed CYTools with pip, you can install it with `pip install mosek`.

Mosek is proprietary software and requires a license to use it. Fortunately, it is easy to obtain and activate a free academic license. The steps to do so are the following.

1. You can find the form to request a free academic license at [this link](https://www.mosek.com/products/academic-licenses/).

2. Follow the instructions to request a personal academic license. Make sure to use your university email address so that the request is approved quickly.

3. You will receive an email with a `mosek.lic` file. Simply copy this license file into the following path (you will need to create the `mosek` directory).

    === "Linux"

        ```
        /home/YOUR_USER_NAME/mosek/mosek.lic
        ```

    === "macOS"

        ```
        /Users/YOUR_USER_NAME/mosek/mosek.lic
        ```

    === "Windows (WSL)"

        ```
        /home/YOUR_WSL_USER_NAME/mosek/mosek.lic
        ```

        This is the home directory inside WSL, not your Windows user directory.
        You can open it in the Windows File Explorer by running `explorer.exe ~`
        in the WSL terminal.

!!! note

    If you prefer to place the license file in a different location, you will need to tell CYTools where to find it. See the [configuration page](../documentation/config.md#mosek-license) for details.

In order to check whether your Mosek license is properly activated, you can run the following commands.

```python
from cytools import config
config.check_mosek_license()
config.mosek_is_activated()
```

If the last function returns `True`, then you are all set. It is not necessary to do this check every time, as CYTools checks the license automatically the first time that Mosek might be used.
