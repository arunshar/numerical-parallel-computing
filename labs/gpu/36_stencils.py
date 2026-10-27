"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
u = np.zeros(9)
u[4] = 1.0
lam = 0.25
for step in range(4):
    new = u.copy()
    new[1:-1] = u[1:-1] + lam * (u[:-2] - 2*u[1:-1] + u[2:])
    new[0] = new[-1] = 0.0
    u = new
assert np.allclose(u, u[::-1])
assert np.min(u) >= 0
print("four_step_solution=", u.tolist())
print(f"sum={u.sum():.8f}, maximum={u.max():.8f}")
for dimension in [1, 2, 3]:
    print(f"dimension={dimension}, tile_width=8, storage_ratio={(10/8)**dimension:.6f}")
