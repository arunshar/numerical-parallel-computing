"""Fit a line and compare rank decisions with singular values."""
import numpy as np

x = np.array([0., 1., 2.])
y = np.array([1., 2., 2.])
A = np.column_stack([np.ones_like(x), x])
Q, R = np.linalg.qr(A, mode="reduced")
beta = np.linalg.solve(R, Q.T@y)
r = y-A@beta
print("intercept and slope:", np.round(beta, 6))
print("residual:", np.round(r, 6))
print(f"normal-equation residual: {np.linalg.norm(A.T@r):.6e}")
assert np.allclose(beta, [7/6, 1/2])
B = np.column_stack([np.ones(5), np.ones(5)+1e-7*np.arange(5)])
s = np.linalg.svd(B, compute_uv=False)
print("singular values:", np.array2string(s, precision=6))
print(f"condition of B: {np.linalg.cond(B):.6e}")
print(f"condition of B.T@B: {np.linalg.cond(B.T@B):.6e}")
for cutoff in [1e-8, 1e-6]:
    print(f"relative cutoff {cutoff:.0e}: numerical rank {np.count_nonzero(s>cutoff*s[0])}")
