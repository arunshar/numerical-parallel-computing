"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
rng=np.random.default_rng(11)
a=rng.integers(-3,4,size=(4,3,3)).astype(np.float64)
b=rng.integers(-3,4,size=(5,3,3)).astype(np.float64)
direct=sum((aa@bb for aa in a for bb in b),start=np.zeros((3,3)))
factored=a.sum(axis=0)@b.sum(axis=0)
assert np.array_equal(direct,factored)
print("integer_case_equal=",bool(np.array_equal(direct,factored)))
print("result=",direct.astype(int).tolist())
af=rng.normal(size=(17,3,3)).astype(np.float32)
bf=rng.normal(size=(19,3,3)).astype(np.float32)
df=np.zeros((3,3),dtype=np.float32)
for aa in af:
    for bb in bf: df += aa@bb
ff=af.sum(axis=0)@bf.sum(axis=0)
error=float(np.max(np.abs(df-ff)))
assert np.allclose(df,ff,rtol=1e-4,atol=1e-4)
print(f"float32_reassociation_max_difference={error:.8e}")
na,nb,d=512,512,4
print(f"direct_matrix_products={na*nb}, factored_matrix_products=1")
print(f"matrix_entries_to_sum={(na+nb)*d*d}")
