---
id: experimental
title: Experimental Features
---

There are a few experimental features that are locked by default since they haven't been through enough testing. They can be enabled as follows.
```python
import cytools
cytools.config.enable_experimental_features()
```

## Calabi-Yau hypersurfaces of dimensions other than 3

There is experimental support for Calabi-Yau manifolds of dimensions other than 3. They can be constructed in the analogous way by starting with reflexive polytopes in dimensions other than 4. As the simplest example, we can construct the Calabi-Yau hypersurface in $\mathbb{P}^5$.
```python
p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-1,-1,-1]])
t = p.triangulate()
t.get_cy()
# A Calabi-Yau 4-fold hypersurface with h11=1, h12=0, h13=426, and h22=1752 in a 5-dimensional toric variety
```
Most of the functions such as the toric Mori cone or the intersection numbers should work well, but we can't guarantee that there won't be any problems.

## Toric Complete Intersection Calabi-Yaus (toric CICYs)

There is also experimental support for a much more general class of Calabi-Yau manifolds obtained as complete intersections in toric varieties. They are constructed by specifying a nef partition of the polytope. In the following example we construct a polytope, find some nef partitions, and construct the corresponding CICYs.
```python
p = Polytope([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,0,0,0,0],[0,-1,0,0,0],[0,0,-1,0,0],[0,0,0,-1,0],[0,0,0,0,-1]])
nef_parts = p.nef_partitions() # Takes a few seconds
print(nef_parts[0]) # We print the first nef partition
# ((7, 6, 3, 4, 5), (10, 9, 8, 1, 2))
print(len(nef_parts)) # We print the number of nef partitions
# 8
t = p.triangulate()
cy0 = t.get_cy(nef_parts[0]) # Construct CY using first nef partition
print(cy0) # Print info
# A complete intersection Calabi-Yau 3-fold with h11=19 h21=19 in a 5-dimensional toric variety
cy1 = t.get_cy(nef_parts[1]) # Construct CY using second nef partition
print(cy1) # Print info
# A complete intersection Calabi-Yau 3-fold with h11=5 h21=37 in a 5-dimensional toric variety
```
Again, most of the functions such as the toric Mori cone or the intersection numbers should work well, but we can't guarantee that there won't be any problems.

## Generic bases for divisors and curves

The only kind of bases that are fully supported are those formed from a subset of prime toric divisors and such that the remaining prime toric divisors can be written as an integral linear combination. However, there is experimental support for generic bases that are specified with a matrix where each row is a linear combination of the canonical divisor, and the prime effective divisors (or the canonical divisor can be left out). There is still the requirement that all prime toric divisors must be able to be written as an integral linear combination of the basis divisors. We can see this in the following example.
```python
p = Polytope([[-1,3,-2,-1],[1,-1,0,0],[-1,0,0,1],[-1,0,0,0],[-1,0,1,1],[-1,0,2,0]])
t = p.triangulate()
cy = t.get_cy()
cy.divisor_basis() # The default basis of divisors
# Prints: array([1, 6, 7])
cy.divisor_basis(as_matrix=True) # Divisor basis in matrix form
# array([[0, 1, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 0, 0, 1]])
cy.curve_basis() # The default basis of curves
# Prints: array([1, 6, 7])
cy.curve_basis(as_matrix=True) # Curves basis in matrix form
# array([[-6,  1,  3,  1, -1,  2,  0,  0],
#        [ 0,  0,  0, -1,  2, -2,  1,  0],
#        [ 0,  0,  0, -1,  1, -1,  0,  1]])
cy.divisor_basis(as_matrix=True).dot(cy.curve_basis(as_matrix=True).T) # Product is always the identity since they are dual bases
# array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
new_div_basis = [[0, 0, 0, 1, 0, 0, 0, 0], # Define a new basis
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
cy.set_divisor_basis(new_div_basis)
cy.divisor_basis() # Now it returns the basis we set
# array([[0, 0, 0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 0, 0, 1]])
cy.curve_basis() # The curve basis is also changed
# array([[-6,  1,  3,  1, -1,  2,  0,  0],
#        [-6,  1,  3,  0,  1,  0,  1,  0],
#        [-6,  1,  3,  0,  0,  1,  0,  1]])
cy.divisor_basis(as_matrix=True).dot(cy.curve_basis(as_matrix=True).T) # Product remains the identity
# array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
new_curve_basis = cy.toric_mori_cone().extremal_rays() # Sometimes the Mori cone is simplicial and smooth, so we can use the extremal rays as a curve basis
cy.set_curve_basis(new_curve_basis)
cy.curve_basis() # Returns the new curve basis
# array([[-6,  1,  3, -1,  1,  0,  0,  2],
#        [ 0,  0,  0,  0,  1, -1,  1, -1],
#        [ 0,  0,  0,  1, -1,  1,  0, -1]])
cy.divisor_basis() # The divisor basis is changed
# array([[ 0,  1,  0,  0,  0,  0,  0,  0],
#        [ 0,  0,  0,  0,  0,  0,  1,  0],
#        [ 0,  2,  0,  0,  0,  0, -1, -1]])
cy.divisor_basis(as_matrix=True).dot(cy.curve_basis(as_matrix=True).T) # Product remains the identity
# array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
cy.toric_mori_cone(in_basis=True).extremal_rays() # The Mori cone is now the first orthant
# array([[1, 0, 0],
#        [0, 0, 1],
#        [0, 1, 0]])
```
One important thing to keep in mind when setting generic bases is that intersection numbers will be computed as a dense array since for a generic basis they are not sparse.

## Rational intersection numbers of singular varieties or Calabi-Yaus

By default, the intersection numbers of the ambient varieties are computed as floating-point numbers, and so are the intersection numbers of singular CYs. There is the option of transforming them into exact rational numbers. The conversion doesn't take too long, but verifying it was successful can be very slow or even run out of memory. Furthermore, the success rate and speed of the conversion decreases quickly as the number of divisors increases. Here we look at a simple example.
```python
p = Polytope([[-1,3,-2,-1],[1,-1,0,0],[-1,0,0,1],[-1,0,0,0],[-1,0,1,1],[-1,0,2,0]])
t = p.triangulate()
v = t.get_toric_variety()
intnums = v.intersection_numbers(exact_arithmetic=True)
print(intnums.get((1,2,3,4),0)) # Print intersection number D_1 \cap D_2 \cap D_3 \cap D_4. They are flint.fmpq objects
# 1/2
```
