"""Test the heat-equation stencil and its amplification factors."""
import numpy as np

n = 40
x = np.arange(n)/n
u0 = np.sin(2*np.pi*x)
amplitudes = {}
for r in [0.4, 0.6]:
    u = u0.copy()
    # A tiny high-frequency component makes instability visible.
    u += 1e-8*(-1.)**np.arange(n)
    for _ in range(100):
        u = u+r*(np.roll(u, -1)-2*u+np.roll(u, 1))
    amplitudes[r] = np.max(np.abs(u))
    print(f"r={r:.1f}: maximum absolute value after 100 steps={np.max(np.abs(u)):.6e}")
    print(f"r={r:.1f}: highest-frequency amplification={1-4*r:.6f}")
assert amplitudes[0.4] < np.max(np.abs(u0))
assert amplitudes[0.6] > 1e4
r = 0.4
u = u0+r*(np.roll(u0, -1)-2*u0+np.roll(u0, 1))
print(f"periodic mass change: {abs(u.sum()-u0.sum()):.6e}")
assert abs(u.sum()-u0.sum())<1e-12
