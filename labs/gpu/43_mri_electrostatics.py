"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
n = 8
k = np.arange(n)
f = np.exp(-2j*np.pi*np.outer(k,k)/n)/np.sqrt(n)
x_true = np.array([0,0,1,2,1,0,0,0],dtype=complex)
y = f@x_true
x = np.zeros(n,dtype=complex); lam=0.1; eta=0.5
objective = lambda x: 0.5*np.linalg.norm(f@x-y)**2 + 0.5*lam*np.linalg.norm(x)**2
initial = objective(x)
for _ in range(30): x -= eta*(f.conj().T@(f@x-y)+lam*x)
z = np.arange(n)+1j*np.arange(n)[::-1]
adjoint_error = abs(np.vdot(f@x_true,z)-np.vdot(x_true,f.conj().T@z))
assert objective(x)<initial and adjoint_error<1e-12
sources=np.array([[0.,0.],[1.,0.]])
charges=np.array([1.,-0.5]);targets=np.array([[0.,1.],[1.,1.],[2.,1.]])
softening=0.1
gather=np.array([sum(q/np.sqrt(np.sum((r-s)**2)+softening**2) for s,q in zip(sources,charges)) for r in targets])
scatter=np.zeros(len(targets))
for s,q in zip(sources,charges):
    scatter += q/np.sqrt(np.sum((targets-s)**2,axis=1)+softening**2)
assert np.allclose(gather,scatter)
print(f"objective_initial={initial:.6f}, objective_final={objective(x):.6f}")
print(f"adjoint_identity_error={adjoint_error:.3e}")
print("softened_potential=",np.round(gather,6).tolist())
