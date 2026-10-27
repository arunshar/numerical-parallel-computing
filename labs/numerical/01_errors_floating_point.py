"""Compare stable and unstable formulas on deterministic toy inputs."""
import math
import numpy as np

x = 1e-12
naive = math.sqrt(1.0 + x) - 1.0
stable = x / (math.sqrt(1.0 + x) + 1.0)
print(f"binary64 spacing at 1: {np.spacing(1.0):.6e}")
print(f"naive cancellation result: {naive:.16e}")
print(f"rationalized result: {stable:.16e}")
print(f"relative disagreement: {abs(naive-stable)/abs(stable):.6e}")
terms = [1e16, 1.0, -1e16]
sequential = 0.0
for term in terms:
    sequential += term
print(f"left-to-right sum: {sequential:.1f}")
assert sequential == 0.0
print(f"compensated sum: {math.fsum(terms):.1f}")
assert math.fsum(terms) == 1.0
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    derivative = (math.sin(1+h)-math.sin(1-h))/(2*h)
    print(f"h={h:.0e}, central difference error={abs(derivative-math.cos(1)):.6e}")
