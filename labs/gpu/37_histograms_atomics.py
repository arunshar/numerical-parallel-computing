"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
values = np.array([0,0,0,0,1,1,2,3,0,0,1,1,2,2,3,3])
bins, chunk = 4, 8
direct = np.bincount(values, minlength=bins)
private = np.zeros(bins, dtype=np.int64)
merge_updates = 0
for start in range(0, len(values), chunk):
    local = np.bincount(values[start:start+chunk], minlength=bins)
    merge_updates += np.count_nonzero(local)
    private += local
assert np.array_equal(direct, private)
print("histogram=", private.tolist())
print(f"direct_global_updates={len(values)}, private_nonzero_merges={merge_updates}")
skewed = np.zeros(16, dtype=int)
print(f"one_bin_private_merges={sum(np.count_nonzero(np.bincount(skewed[s:s+chunk], minlength=bins)) for s in range(0,16,chunk))}")
