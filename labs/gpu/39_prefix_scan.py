"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
x = np.array([3,1,4,1,5,9,2,6], dtype=np.int64)
a = x.copy(); doubling_adds = 0
for distance in [1,2,4]:
    old = a.copy()
    a[distance:] = old[distance:] + old[:-distance]
    doubling_adds += len(x)-distance
assert np.array_equal(a, np.cumsum(x))
b = x.copy(); adds = 0; n = len(b)
stride = 2
while stride <= n:
    for end in range(stride-1,n,stride):
        b[end] += b[end-stride//2]; adds += 1
    stride *= 2
b[-1] = 0
stride = n
while stride >= 2:
    for end in range(stride-1,n,stride):
        left = end-stride//2
        saved = b[left]; b[left] = b[end]; b[end] += saved; adds += 1
    stride //= 2
assert np.array_equal(b, np.r_[0,np.cumsum(x)[:-1]])
flags = (x % 2 == 0).astype(int)
offsets = np.r_[0,np.cumsum(flags)[:-1]]
out = np.empty(int(flags.sum()), dtype=x.dtype)
for i in range(n):
    if flags[i]: out[offsets[i]] = x[i]
print("inclusive=", a.tolist())
print("exclusive=", b.tolist())
print(f"doubling_additions={doubling_adds}, tree_additions={adds}")
print("compacted_even_values=", out.tolist())
