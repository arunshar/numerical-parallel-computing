"""Solve a scalar root and a small nonlinear system."""
import math
import numpy as np

lo, hi = 1., 2.
for _ in range(30):
    mid = (lo+hi)/2
    if mid*mid < 2:
        lo = mid
    else:
        hi = mid
print(f"bisection error bound: {(hi-lo)/2:.6e}")
print(f"bisection root: {(lo+hi)/2:.12f}")
x = 1.
for k in range(5):
    x -= (x*x-2)/(2*x)
    print(f"Newton step {k+1}: x={x:.12f}, error={abs(x-math.sqrt(2)):.6e}")
z = np.array([1.2, 0.8])
for _ in range(6):
    f = np.array([z[0]**2+z[1]**2-2, z[0]-z[1]])
    J = np.array([[2*z[0], 2*z[1]], [1., -1.]])
    z += np.linalg.solve(J, -f)
print("system root:", np.round(z, 10))
assert np.allclose(z, [1., 1.])
