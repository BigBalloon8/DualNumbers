import math
import duals


def test1():
    f = lambda x: math.sin(x)*math.cos(x)**2
    x = duals.Dual(4)
    print(f(x))

def test2():
    f = lambda x, y: math.sin(x)/(math.cos(y)+x**2)
    x = duals.Dual(math.pi, tag="x")
    y = duals.Dual(math.pi, tag="y")
    print(f(x,y))

def test3():
    f = lambda x, y: math.sin(math.exp(x))/math.tan(math.sqrt(y)) + math.log(x, 10)
    x0 = duals.Dual(1, tag="x")
    y0 = duals.Dual(2, tag="y")
    print(f(x0,y0))

def test4():
    vec = [1,2,3,4]
    vec2 = [0,-2,-3,1]
    v = duals.DualVec(vec, tag="v")
    u = duals.DualVec(vec2, tag="u")
    print((v.dot(u)))


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()