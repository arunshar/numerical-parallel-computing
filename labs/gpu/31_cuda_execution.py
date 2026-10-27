"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
n, block = 10, 4
grid = (n + block - 1) // block
ids = [b * block + t for b in range(grid) for t in range(block)]
valid = [i for i in ids if i < n]
a = np.arange(n, dtype=np.float32)
b = np.full(n, 3, dtype=np.float32)
c = np.empty_like(a)
for i in valid:
    c[i] = a[i] + b[i]
assert valid == list(range(n))
assert np.array_equal(c, a + b)
height, width = 3, 5
addresses = [r * width + col for r in range(height) for col in range(width)]
assert addresses == list(range(height * width))
print(f"elements={n}, blocks={grid}, launched_threads={len(ids)}")
print(f"valid_threads={len(valid)}, skipped_threads={len(ids)-len(valid)}")
print("vector_sum=", c.astype(int).tolist())
print(f"rectangular_address_count={len(addresses)}")
