"""Compare quadrature convergence and a Gaussian rule."""
import math
import numpy as np

exact = math.e-1
for n in [4, 8, 16, 32, 64]:
    x = np.linspace(0, 1, n+1)
    y = np.exp(x)
    trapezoid = (y[0]/2+y[-1]/2+y[1:-1].sum())/n
    simpson = (y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())/(3*n)
    print(f"n={n:2d}: trapezoid error={abs(trapezoid-exact):.6e}, Simpson error={abs(simpson-exact):.6e}")
nodes = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
quad = float(np.sum(nodes**2))
print(f"two-point Gauss integral of x^2 on [-1,1]: {quad:.12f}")
assert abs(quad-2/3)<1e-14
for h in [0.2, 0.1, 0.05]:
    d1 = (math.exp(h)-math.exp(-h))/(2*h)
    d2 = (math.exp(h/2)-math.exp(-h/2))/h
    richardson = (4*d2-d1)/3
    print(f"h={h:.2f}: central error={abs(d1-1):.6e}, extrapolated error={abs(richardson-1):.6e}")
