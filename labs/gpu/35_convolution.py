"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
x = np.array([2., 1., 4., 3., 5.])
h = np.array([1., 2., -1.])
r, tile = 1, 3
padded = np.pad(x, (r, r))
direct = np.array([sum(h[k] * padded[i+k] for k in range(len(h))) for i in range(len(x))])
tiled = np.zeros_like(x)
for start in range(0, len(x), tile):
    count = min(tile, len(x) - start)
    local = padded[start:start + count + 2*r]
    for t in range(count):
        tiled[start+t] = np.dot(local[t:t+len(h)], h)
assert np.array_equal(direct, tiled)
print("zero_padded_cross_correlation=", direct.tolist())
for t in [8, 16, 32]:
    print(f"output_width={t}, halo_load_ratio={(t+2*r)**2/t**2:.6f}")
