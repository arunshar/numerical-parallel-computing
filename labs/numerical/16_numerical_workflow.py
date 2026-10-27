"""Check a multivariate Taylor model and bilinear interpolation."""
import numpy as np

f = lambda z: np.exp(z[0])+z[0]*z[1]+z[1]**2
base = np.array([0.,1.])
grad = np.array([2.,2.])
hessian = np.array([[1.,1.],[1.,2.]])
for scale in [0.2,0.1,0.05]:
    step = scale*np.array([1.,-1.])
    first = f(base)+grad@step
    second = first+0.5*step@hessian@step
    print(f"scale={scale:.2f}: first-order error={abs(f(base+step)-first):.6e}, second-order error={abs(f(base+step)-second):.6e}")
u,v = 0.25,0.75
corners = np.array([[0.,1.],[2.,3.]])
weights = np.array([[(1-u)*(1-v), (1-u)*v], [u*(1-v),u*v]])
print(f"bilinear value: {np.sum(weights*corners):.6f}")
print(f"sum of weights: {weights.sum():.6f}")
assert abs(weights.sum()-1)<1e-15
print("elementwise square:", np.array([1.,2.,3.])**2)
