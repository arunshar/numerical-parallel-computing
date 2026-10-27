import json
import numpy as np
A=np.arange(16,dtype=np.int64).reshape(4,4)
B=(np.arange(16,dtype=np.int64).reshape(4,4)+1)
q,b=2,2
C=np.zeros((4,4),dtype=np.int64)
trace=[]
for i in range(q):
 for j in range(q):
  block=np.zeros((b,b),dtype=np.int64)
  ks=[]
  for stage in range(q):
   k=(i+j+stage)%q
   block += A[i*b:(i+1)*b,k*b:(k+1)*b] @ B[k*b:(k+1)*b,j*b:(j+1)*b]
   ks.append(k)
  C[i*b:(i+1)*b,j*b:(j+1)*b]=block
  trace.append({'rank':[i,j],'inner_blocks':ks})
assert np.array_equal(C,A@B)
print(json.dumps({'trace':trace,'C':C.tolist(),'matches_serial':True},indent=2))
