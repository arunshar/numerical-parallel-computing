"""Compare stable gradient steps with an exact quadratic Newton step."""
import numpy as np

H = np.diag([1., 20.])
x0 = np.array([2., 2.])
def objective(x):
    return float(0.5*x@H@x)
for alpha in [0.05, 0.09, 0.11]:
    x = x0.copy()
    for _ in range(25):
        x -= alpha*(H@x)
    print(f"alpha={alpha:.2f}, objective after 25 steps={objective(x):.6e}")
xn = x0-np.linalg.solve(H, H@x0)
print(f"Newton objective after one step: {objective(xn):.6e}")
assert np.allclose(xn, 0.)
x = x0.copy()
for _ in range(10):
    g = H@x
    alpha = float((g@g)/(g@H@g))
    x -= alpha*g
print(f"exact-line-search objective after 10 steps: {objective(x):.6e}")
