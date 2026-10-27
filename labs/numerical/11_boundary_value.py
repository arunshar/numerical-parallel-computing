"""Solve a two-point boundary problem by centered finite differences."""
import numpy as np

errors = []
for n in [9, 19, 39, 79]:
    h = 1/(n+1)
    x = h*np.arange(1, n+1)
    A = (2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1))/h**2
    rhs = np.pi**2*np.sin(np.pi*x)
    u = np.linalg.solve(A, rhs)
    error = np.max(np.abs(u-np.sin(np.pi*x)))
    errors.append(error)
    assert np.linalg.norm(A@u-rhs) / np.linalg.norm(rhs) < 1e-10
    print(f"interior nodes={n:2d}, max error={error:.6e}, residual={np.linalg.norm(A@u-rhs):.6e}")
assert all(3.8 < coarse/fine < 4.2 for coarse, fine in zip(errors[:-1], errors[1:]))
# The exact shooting solution for u''=0 is u(x)=a+b*x.
a, target = 2., 5.
slope = target-a
print(f"linear shooting slope for u(0)=2 and u(1)=5: {slope:.6f}")
assert slope == 3.
