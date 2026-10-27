"""Compare stationary iteration and conjugate gradients on an SPD system."""
import numpy as np

n = 12
A = 2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
b = np.ones(n)
x = np.zeros(n)
for _ in range(100):
    x += (b-A@x)/np.diag(A)
print(f"Jacobi residual after 100 steps: {np.linalg.norm(b-A@x):.6e}")
x = np.zeros(n);r = b.copy();p = r.copy();rr = r@r
for k in range(n):
    Ap = A@p
    alpha = rr/(p@Ap)
    x += alpha*p;r -= alpha*Ap
    newrr = r@r
    if np.sqrt(newrr)<1e-10:
        break
    p = r+(newrr/rr)*p;rr = newrr
print(f"CG iterations: {k+1}")
print(f"CG residual: {np.linalg.norm(b-A@x):.6e}")
print(f"solution error: {np.linalg.norm(x-np.linalg.solve(A,b)):.6e}")
assert np.allclose(A@x,b)
