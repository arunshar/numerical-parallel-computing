"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
values = np.arange(1, 10, dtype=np.float64)
level = values.copy()
levels = 0
while len(level) > 1:
    if len(level) % 2:
        level = np.append(level, 0.0)
    level = level[0::2] + level[1::2]
    levels += 1
assert level[0] == 45
x = np.array([1e8, 1., -1e8, 1.], dtype=np.float32)
left = np.float32(0)
for value in x:
    left = np.float32(left + value)
reordered = np.float32(np.float32(x[0] + x[2]) + np.float32(x[1] + x[3]))
print(f"odd_length_sum={level[0]:.0f}, tree_levels={levels}")
print(f"float32_left_order={left:.0f}, reordered={reordered:.0f}, exact_real_sum=2")
