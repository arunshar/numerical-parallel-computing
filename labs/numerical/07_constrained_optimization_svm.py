"""Solve an equality-constrained quadratic through its KKT system."""
import numpy as np

H = np.eye(2)
A = np.ones((1, 2))
K = np.block([[H, A.T], [A, np.zeros((1, 1))]])
rhs = np.array([0., 0., 1.])
solution = np.linalg.solve(K, rhs)
x, multiplier = solution[:2], solution[2]
print("primal solution:", x)
print(f"Lagrange multiplier: {multiplier:.6f}")
print(f"constraint residual: {float((A@x)[0]-1):.6e}")
print(f"stationarity residual: {np.linalg.norm(H@x+A.ravel()*multiplier):.6e}")
assert np.allclose(x, [0.5, 0.5])
u = np.array([-2., 3.])
projection = np.maximum(u, 0.)
print("projection onto nonnegative orthant:", projection)
X = np.array([[-2.], [-1.], [1.], [2.]])
y = np.array([-1., -1., 1., 1.])
w, bias = np.array([1.]), 0.
print("hard-margin signed scores:", y*(X@w+bias))
print(f"geometric margin: {1/np.linalg.norm(w):.6f}")
