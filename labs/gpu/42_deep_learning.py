"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
x = np.arange(25,dtype=np.float64).reshape(5,5)
filters = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]],
                    [[0,1,0],[1,2,1],[0,1,0]]],dtype=float)
patches = np.column_stack([x[r:r+3,c:c+3].reshape(-1) for r in range(3) for c in range(3)])
y_gemm = (filters.reshape(2,9) @ patches).reshape(2,3,3)
y_direct = np.zeros((2,3,3))
for f in range(2):
    for r in range(3):
        for c in range(3):
            y_direct[f,r,c] = np.sum(filters[f]*x[r:r+3,c:c+3])
assert np.array_equal(y_gemm,y_direct)
print(f"input_values={x.size}, patch_values={patches.size}, expansion={patches.size/x.size:.2f}")
print(f"gemm_shape=(2,9)@(9,9), nominal_flops={2*2*9*9}")
print("first_output_channel=",y_gemm[0].astype(int).tolist())
