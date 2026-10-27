"""Measure residuals and perturbation sensitivity in two linear systems."""
import numpy as np

A = np.array([[2., 1.], [1., 3.]])
b = np.array([1., 2.])
x = np.linalg.solve(A, b)
print("solution:", np.array2string(x, precision=6))
print(f"residual norm: {np.linalg.norm(b-A@x):.6e}")
print(f"condition number: {np.linalg.cond(A):.6f}")
assert np.allclose(x, [0.2, 0.6])
B = np.array([[1., 1.], [1., 1.+1e-6]])
x_true = np.ones(2)
c = B @ x_true
delta = np.array([1e-7, -1e-7])
y = np.linalg.solve(B, c+delta)
print(f"sensitive-system condition number: {np.linalg.cond(B):.6e}")
print(f"relative data change: {np.linalg.norm(delta)/np.linalg.norm(c):.6e}")
print(f"relative solution change: {np.linalg.norm(y-x_true)/np.linalg.norm(x_true):.6e}")
print(f"perturbed residual: {np.linalg.norm(c+delta-B@y):.6e}")
