"""Draw an original figure that separates numerical evidence from a cost model."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ns=np.array([7,15,31,63]);errors=[]
for n in ns:
 h=1/(n+1);x=np.arange(1,n+1)*h
 A=(2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1))/h**2
 u=np.linalg.solve(A,np.pi**2*np.sin(np.pi*x))
 errors.append(np.max(np.abs(u-np.sin(np.pi*x))))
p=np.array([1,2,4,8,16,32,64,128,256]);t=2e-3/p+np.where(p>1,10e-6+16/10e9,0)
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].loglog(1/(ns+1),errors,'o-',label='Computed solution error')
ax[0].set(xlabel='Grid spacing h',ylabel='Maximum absolute error',title='A numerical check measures error.')
ax[0].grid(alpha=.25);ax[0].legend()
ax[1].semilogx(p,2e-3/t,'o-',label='Predicted speedup')
ax[1].semilogx(p,p,'--',label='Ideal reference')
ax[1].set(xlabel='Modeled worker count',ylabel='Speedup',title='A cost model predicts scaling.')
ax[1].grid(alpha=.25);ax[1].legend()
fig.suptitle('The left panel uses a CPU solve. The right panel uses invented cost parameters.',fontsize=12)
fig.savefig(ROOT/'docs/_static/accuracy-and-scaling.svg')
