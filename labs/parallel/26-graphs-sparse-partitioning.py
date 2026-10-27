import json
import numpy as np
A=np.array([[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]],dtype=float)
rowptr=[0];cols=[];vals=[]
for row in A:
 for j,v in enumerate(row):
  if v!=0:cols.append(j);vals.append(v)
 rowptr.append(len(vals))
x=np.array([1,2,4,8.]);y=np.zeros(4)
for i in range(4):
 for k in range(rowptr[i],rowptr[i+1]):y[i]+=vals[k]*x[cols[k]]
assert np.allclose(y,A@x) and np.allclose(A@np.ones(4),0)
print(json.dumps({'rowptr':rowptr,'columns':cols,'y':y.tolist(),
                  'laplacian_constant_residual':float(np.linalg.norm(A@np.ones(4)))},indent=2))
