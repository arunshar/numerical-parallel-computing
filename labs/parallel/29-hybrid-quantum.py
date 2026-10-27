import json
import numpy as np
H=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
H2=np.kron(H,H)
start=np.array([1,0,0,0],dtype=complex)
results={}
for name,phases in [('constant',[1,1,1,1]),('balanced',[1,-1,-1,1])]:
 state=H2@(np.array(phases)*(H2@start))
 probs=np.abs(state)**2
 assert np.isclose(probs.sum(),1)
 results[name]=np.round(probs,12).tolist()
bell=np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
assert np.isclose(results['constant'][0],1) and np.isclose(results['balanced'][0],0)
print(json.dumps({'deutsch_jozsa_probabilities':results,
                  'bell_probabilities':(np.abs(bell)**2).tolist()},indent=2))
