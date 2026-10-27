"""Compare power iteration and a symmetric eigensolver."""
import numpy as np

A = np.array([[2., 1.], [1., 2.]])
x = np.array([1., 0.])
for k in range(1, 9):
    x = A@x
    x /= np.linalg.norm(x)
    rayleigh = x@A@x
    residual = np.linalg.norm(A@x-rayleigh*x)
    print(f"iteration {k}: lambda={rayleigh:.9f}, residual={residual:.6e}")
values, vectors = np.linalg.eigh(A)
print("reference eigenvalues:", values)
assert np.allclose(values, [1., 3.])
B = np.array([[1., 1.], [0., 1.]])
print("defective-matrix eigenvalues:", np.linalg.eigvals(B))
print("rank of B-I:", np.linalg.matrix_rank(B-np.eye(2)))
