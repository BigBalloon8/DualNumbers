# Dual Numbers

## Basics

The Dual number class works like the `float` type **treat it like it is just a number** (e.g. `5*0.5 + 2**6`). First wrap your given number with the Dual Class 
```python
>>> from duals import Dual
>>> print(Dual(2))
2 + ε
>>> print(Dual(4) + 0.5)
4.5 + ε
>>> print((Dual(6)/4)**2)
2.25 + 0.75ε
```

we can use the dual number to get the derivatives of any functions

```python
>>> def f(x):
...    return (x+1)**2/(x+2)
... 
>>> a = Dual(4)
>>> out = f(a)
>>> print(out)
4.166666666666667 + 0.9722222222222222ε
>>> print(out.re)
4.166666666666667
>>> print(out.get_dual())
0.9722222222222222
```
We can get the real part of f(a) with the `Dual.re` property and use the `Dual.get_dual()` method to return the derivative. **Warning:** `Dual.re` is not a function but `Dual.get_dual()` is.

The Dual numbers class works with the `math` library
```python
>>> import math
>>> pi = Dual(math.pi)
>>> print(math.cos(pi))
1.0 - 1.2246467991473532e-16ε
>>> def g(x):
...    return math.sin(x)*math.cos(x)**2
... 
>>> b = Dual(5)
>>> print(g(b))
-0.0771591086265054 - 0.4988503882783093ε
```
notice that for `math.cos(Dual(math.pi))` the output of the derivate is `1.2246467991473532e-16` where it should be 0 this is due to an inaccuracy with how computers do maths, notice that it is very close though so it doesn't really matter.

## Multi-Variable

To use multiple variable we use the `tag` to tell the Dual class which variable the derivative belongs too

```python
>>> a = duals.Dual(math.pi, tag="x")
>>> b = duals.Dual(math.pi, tag="y")
>>> def h(x, y):
...     return math.sin(x)/(math.cos(y)+x**2)
... 
>>> out = h(a, b)
>>> print(out)
1.3807231346157255e-17 - 0.11274459995951801ε_x + 1.9063963744630746e-34ε_y
>>> print(out.get_dual("x"))
0.11274459995951801
```

## Sharp Bits (warnings)
=======================================

Dont use duals inplace for example 
```python
>>> p = Dual(0)
>>> for i in range(10):
...     p += 2
...
ERROR 
```
instead do
```python
>>> p = Dual(0)
>>> for i in range(10):
...     p = p + 2
...
>>> print(p)
20 + ε
```
=======================================

Duals dont work with numpy so dont use numpy


