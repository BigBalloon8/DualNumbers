import math
from functools import wraps


def dual_wrapper(f):
    # for functions that take 1 number
    @wraps(f)
    def helper(*args, **kwargs):
        x = args[0]
        if len(args) == 1:
            if isinstance(x, Dual):
                return getattr(x, f.__name__)(**kwargs)
            else:
                return f(x)
        else:
            if isinstance(x, Dual):
                return getattr(x, f.__name__)(*args[1:], **kwargs)
            else:
                return f(x)
    return helper

def _overwrite_math():
    math.sin = dual_wrapper(math.sin)
    math.cos = dual_wrapper(math.cos)
    math.tan = dual_wrapper(math.tan)
    math.log = dual_wrapper(math.log)
    math.exp = dual_wrapper(math.exp)
    math.sqrt = dual_wrapper(math.sqrt)
    math.import_tracker = True

_overwrite_math()


class DualDict:
    """Class used to store multiple variable dual numbers"""
    def __init__(self, dual_dict):
        # Each value in the dict represents the dual number of a separate variable
        self.duals = dual_dict
    
    def _all_keys(self, other_duals):
        """Returns a list of the variables from both dual dicts"""
        keys = []
        for k in self.duals.keys():
            if k not in keys:
                keys.append(k)
        for k in other_duals.keys():
            if k not in keys:
                keys.append(k)
        return keys
    
    def _fill_blanks(self, other_duals):
        """Adds variable dual to the dict if they dont exist"""
        for k in self._all_keys(other_duals):
            if k not in self.duals.keys():
                self.duals[k] = 0
            if k not in other_duals.keys():
                other_duals[k] = 0

    def __add__(self, other_duals):
        """add 2 sets of duals together"""
        self._fill_blanks(other_duals)
        new_dict = dict()
        for k in self.duals.keys():
            new_dict[k] = self[k] + other_duals[k]
        return DualDict(new_dict)

    def __radd__(self, other_duals):
        return self.__add__(other_duals)
    
    def __sub__(self, other_duals):
        return self.__add__(-other_duals)

    def __rsub__(self, other_duals):
        return other_duals.__sub__(self)

    def __mul__(self, val):
        """multiply every dual by a constant"""
        new_dict = dict()
        for k in self.duals.keys():
            new_dict[k] = val*self[k]
        return DualDict(new_dict)

    def __rmul__(self, val):
        return self.__mul__(val)
    
    def __truediv__(self, val):
        new_dict = dict()
        for k in self.duals.keys():
            new_dict[k] = self[k]/val
        return DualDict(new_dict)
    
    def __neg__(self):
        """negate all duals"""
        new_dict = dict()
        for k in self.duals.keys():
            new_dict[k] = -self[k]
        return DualDict(new_dict)

    def __getitem__(self, name):
        """get a dual from the dict"""
        return self.duals[name]

    def __setitem__(self, name, value):
        """set the value of a dual in the dict"""
        self.duals[name] = value
    
    def keys(self):
        return self.duals.keys()
    
    def __repr__(self):
        """helper to format the output of duals nicely"""
        base = ""
        for k in self.duals.keys():
            if self.duals[k] == 0:
                continue
            sign = '+' if self.duals[k]>=0 else '-'
            val = abs(self.duals[k]) if abs(self.duals[k] != 1) else ''
            tag = f"_{k}" if k != "" else ""
            base += f"{sign} {val}ε{tag} "
        return base
    
    def zero(self):
        """Set all duals to 0, useful for functions with derivative 0"""
        new_dict = dict()
        for k in self.duals.keys():
            new_dict[k] = 0
        return DualDict(new_dict)

    @classmethod
    def divide(cls, r1, d1, r2, d2):
        """Returns the dual numbers after a division operation [(r1 +d1ε)/(r2+d2ε)]"""
        d1._fill_blanks(d2)
        new_dict = dict()
        for k in d1.keys():
            new_dict[k] = (r2*d1[k] - r1*d2[k])/r2**2
        return cls(new_dict)
    
    def __eq__(self, other_duals):
        self._fill_blanks(other_duals)
        for k in self.duals.keys():
            if self[k] != self.duals.keys[k]:
                return False
        return True


class Dual:
    """
    Attributes
    ----------
    re: Num\\
        The real part of the dual number
    dual: DualDict[Any, Num]\\
        A dictionary containing an epsilon for each variable. If no variables tags are given in the default tag 
        "" will be used
    
    Example 1
    ---------
    ```python
    >>> f = lambda x: math.sin(x)*math.cos(x)**2
    >>> x = duals.Dual(4)
    >>> print(f(x))
    -0.3233438533270908 + 0.46947956383346595ε 
    ```

    Example 2
    ---------
    Multi-Variable
    ```python
    >>> f = lambda x, y: math.sin(x)/(math.cos(y)+x**2)
    >>> x = duals.Dual(math.pi, tag="x")
    >>> y = duals.Dual(math.pi, tag="y")
    >>> print(f(x,y))
    1.3807231346157255e-17 - 0.11274459995951801ε_x + 1.9063963744630746e-34ε_y
    >>> z = f(x,y)
    >>> print(z.dual.duals)
    ```
    """
    def __init__(self, real=0, dual=1, tag=""):
        self.re = real
        if isinstance(dual, DualDict):
            self.dual = dual
        elif isinstance(dual, dict):
            self.dual = DualDict(dual)
        else:
            self.dual = DualDict({tag: dual})
        if not hasattr(math, "import_tracker"):
            _overwrite_math()
        
    def __abs__(self):
        if self.re == 0:
            raise ValueError("Gradient does not exist when at 0")
        if self.re>0:
            return Dual(self.re, self.dual)
        else:
            return Dual(-self.re, -self.dual)
    
    def __add__(self, value):
        if isinstance(value, Dual):
            return Dual(self.re+value.re, self.dual+value.dual)
        else:
            return Dual(self.re+value, self.dual)
    
    def __radd__(self, value): return self.__add__(value)

    def __iadd__(self, value):
        if isinstance(value, Dual):
            self.re += value.re
            self.dual = self.dual + value.dual
        else:
            self.re += value
        return self
    
    def __sub__(self, value):
        return self.__add__(-value)
    
    def __rsub__(self, value): return value + -self

    def __iadd__(self, value):
        if isinstance(value, Dual):
            self.re -= value.re
            self.dual = self.dual - value.dual
        else:
            self.re -= value
        return self

    def __ceil__(self):
        return Dual(math.ceil(self.re), self.dual.zero())
    
    def __floor__(self):
        return Dual(math.floor(self.re), self.dual.zero())

    def __mod__(self, value):
        return Dual(self.re%value, self.dual)
    
    def __mul__(self, value):
        if isinstance(value, Dual):
            return Dual(
                self.re*value.re,
                self.re*value.dual + value.re*self.dual
            )
        else:
            return Dual(value * self.re, value* self.dual)
    
    def __rmul__(self, value): return self.__mul__(value)

    def __imul__(self,value):
        if isinstance(value, Dual):
            self.re *= value.re
            self.dual = self.re*value.dual + value.re*self.dual
        else:
            self.re *= value
            self.dual = value*self.dual
        return self
    
    def __truediv__(self, value):
        if isinstance(value, Dual):
            return Dual(
                self.re/value.re,
                DualDict.divide(self.re, self.dual, value.re, value.dual)
            )
        else:
            return Dual(self.re/value, self.dual/value)
    
    def __rtruediv__(self, value):
        if isinstance(value, Dual):
            return value.__truediv__(self)
        else:
            return value * pow(self, -1)
    
    def __itruediv__(self, value):
        if isinstance(value, Dual):
            self.re /= value.re
            self.dual = DualDict.divide(self.re, self.dual, value.re, value.dual)
        else:
            self.re= self.re/value
            self.dual = self.dual/value 
        return self

    def __neg__(self):
        return Dual(-self.re, -self.dual)

    def __pow__(self, value):
        if isinstance(value, Dual):
            raise NotImplementedError()
        else:
            return Dual(self.re**value, self.dual*value*self.re**(value-1))
        
    def __rpow__(self, value):
        if isinstance(value, Dual):
            raise NotImplementedError()
        else:
            return Dual(value**self.re, self.dual*math.log(value)*value**(self.re))
    
    def __repr__(self):
        return f"{self.re} {self.dual.__repr__()}"

    def __round__(self):
        return Dual(round(self.re), self.dual.zero())
    
    def __eq__(self, value):
        if isinstance(value, Dual):
            return self.re == value.re and self.dual == value.dual
        else:
            return self.re == value

    def get_dual(self, tag=""):
        try:
            return self.dual[tag]
        except KeyError:
            if tag == "":
                raise KeyError("Pls provide dual tag")
            else:
                raise KeyError(f"Dual Tag [{tag}] does not exist")
    
    def sin(self):
        return Dual(math.sin(self.re), self.dual*math.cos(self.re))
    
    def cos(self):
        return Dual(math.cos(self.re), -self.dual*math.sin(self.re))
    
    def tan(self):
        return Dual(math.tan(self.re), self.dual/math.cos(self.re)**2)
    
    def log(self, base=None):
        if base is None:
            return Dual(math.log(self.re), self.dual/self.re)
        else:
            return Dual(math.log(self.re), self.dual/(math.log(base)*self.re))
    
    def exp(self):
        return Dual(math.exp(self.re), self.dual*math.exp(self.re))

    def sqrt(self):
        return Dual(math.sqrt(self.re), self.dual/(2*math.sqrt(self.re)))

    def __getitem__(self, key=None):
        return self.get_dual(key)


class DualVec:
    def __init__(self, vals, duals=None, tag=None):
        if isinstance(vals[0], Dual):
            self.vec = vals
            return
        
        if tag is None:
            raise ValueError("Please give a tag")
        
        if duals is None:
            duals = [1]*len(vals)
        if duals is isinstance(duals, (int, float, complex)) and not isinstance(duals, bool):
            duals = [duals]*len(vals)

        self.vec = []

        for i, (r,d) in enumerate(zip(vals, duals)):
            self.vec.append(Dual(r, d, f"{tag}_{i}"))
    
    def __add__(self, value):
        out_vec =[]
        if isinstance(value, DualVec):
            for (i, j) in zip(self.vec, value.vec):
                out_vec.append(i + j)
        elif isinstance(value, list):
            for (i, j) in zip(self.vec, value):
                out_vec.append(i + j)
        else:
            for i in self.vec:
                out_vec.append(i+value)
        return DualVec(out_vec)
    
    __addr__ = __add__

    def __sub__(self, value):
        out_vec =[]
        if isinstance(value, DualVec):
            for (i, j) in zip(self.vec, value.vec):
                out_vec.append(i - j)
        elif isinstance(value, list):
            for (i, j) in zip(self.vec, value):
                out_vec.append(i - j)
        else:
            for i in self.vec:
                out_vec.append(i-value)
        return DualVec(out_vec)
    
    def __subr__(self, value):
        return self.__add__(-self, value)

    def __neg__(self):
        out_vec = []
        for i in self.vec:
            out_vec.append(-i)
        return DualVec(out_vec)

    def __mul__(self, value):
        out_vec =[]
        if isinstance(value, DualVec):
            for (i, j) in zip(self.vec, value.vec):
                out_vec.append(i * j)
        elif isinstance(value, list):
            for (i, j) in zip(self.vec, value):
                out_vec.append(i * j)
        else:
            for i in self.vec:
                out_vec.append(i*value)
        return DualVec(out_vec)
    
    __mulr__ = __mul__

    def __truediv__(self, value):
        out_vec = []
        for i in self.vec:
            out_vec.append(i/value)
        return DualVec(out_vec)

    def dot(self, value):
        total = 0.0
        if isinstance(value, DualVec):
            for (i, j) in zip(self.vec, value.vec):
                total = total + i*j
        elif isinstance(value, list):
            for (i, j) in zip(self.vec, value):
                total = total + i*j
        return total

    def mag(self):
        total = 0.0
        for i in self.vec:
            total = total + i**2
        return math.sqrt(total)
    
    def __repr__(self):
        return self.vec.__repr__()

    __abs__ = mag