"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
rng = np.random.default_rng(7)
a = rng.normal(size=(7, 5))
b = rng.normal(size=(5, 6))
c = np.zeros((7, 6))
tile = 4
for row in range(0, a.shape[0], tile):
    for col in range(0, b.shape[1], tile):
        for k in range(0, a.shape[1], tile):
            aa = a[row:row+tile, k:k+tile]
            bb = b[k:k+tile, col:col+tile]
            c[row:row+aa.shape[0], col:col+bb.shape[1]] += aa @ bb
error = float(np.max(np.abs(c - a @ b)))
assert np.allclose(c, a @ b)
print(f"rectangular_product_max_error={error:.3e}")
for t in [4, 8, 16, 32]:
    print(f"tile={t}, input_only_intensity={t/4:.1f} FLOP/byte")
for stride in [32, 33]:
    banks = {(lane * stride) % 32 for lane in range(32)}
    print(f"word_stride={stride}, distinct_banks={len(banks)}")
