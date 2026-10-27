import json, hashlib
import numpy as np
rng=np.random.default_rng(5451)
A=rng.normal(size=(7,5));B=rng.normal(size=(5,6));C=np.zeros((7,6))
tile=3
for i in range(0,7,tile):
 for j in range(0,6,tile):
  for k in range(0,5,tile):
   C[i:i+tile,j:j+tile]+=A[i:i+tile,k:k+tile]@B[k:k+tile,j:j+tile]
err=float(np.max(np.abs(C-A@B)))
assert err<1e-12
print(json.dumps({'seed':5451,'A_shape':list(A.shape),'B_shape':list(B.shape),
 'dtype':str(A.dtype),'tile':tile,'max_abs_error':err,
 'input_sha256':hashlib.sha256(A.tobytes()+B.tobytes()).hexdigest(),
 'runtime_measured':False,'claim':'The blocked result matches the reference within1e-12.'},indent=2))
