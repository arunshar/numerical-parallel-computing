import json
import numpy as np
def fft(x):
 x=np.asarray(x,dtype=complex);n=len(x)
 if n==1:return x.copy()
 assert n%2==0
 e,o=fft(x[::2]),fft(x[1::2])
 w=np.exp(-2j*np.pi*np.arange(n//2)/n)*o
 return np.concatenate([e+w,e-w])
x=np.array([1,2,0,-1,3,0,2,1.])
y=fft(x)
error=float(np.max(np.abs(y-np.fft.fft(x))))
assert error<1e-12
n,p=16,4;local=n//p;stages=[]
for bit in range(4):
 pairs=[(i,i^(1<<bit)) for i in range(n) if i<(i^(1<<bit))]
 remote=sum(a//local!=b//local for a,b in pairs)
 stages.append({'partner_bit':bit,'remote_pairs':remote})
assert [s['remote_pairs'] for s in stages]==[0,0,8,8]
print(json.dumps({'max_reference_error':error,'reconstruction_ok':bool(np.allclose(np.fft.ifft(y),x)),
                  'ownership_stages':stages},indent=2))
