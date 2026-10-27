"""Estimate an integral and contrast random with low-discrepancy points."""
import numpy as np

def radical_inverse(k, base=2):
    value, factor = 0., 1/base
    while k:
        value += factor*(k % base)
        k //= base
        factor /= base
    return value

rng = np.random.default_rng(2026)
for n in [64,256,1024,4096]:
    u = rng.random(n)
    values = u*u
    estimate = values.mean()
    se = values.std(ddof=1)/np.sqrt(n)
    quasi = np.array([radical_inverse(k) for k in range(1,n+1)])
    print(f"n={n:4d}: random error={abs(estimate-1/3):.6e}, estimated SE={se:.6e}, quasi error={abs(np.mean(quasi**2)-1/3):.6e}")
u = rng.random(10000)
exponential = -np.log1p(-u)/2
print(f"exponential sample mean: {exponential.mean():.6f}; exact mean: 0.500000")
assert np.all(exponential>=0)
