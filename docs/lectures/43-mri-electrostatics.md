# 43. MRI reconstruction and electrostatic potential maps

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`42-deep-learning`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will write a forward operator and its adjoint for a small Fourier inverse problem.
- You will separate a numerical reconstruction method from its expensive kernels.
- You will compare gather, scatter, and spatial cutoff designs for a potential map. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/43_mri_electrostatics.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## Applications begin with an operator

An application kernel should come from a mathematical model. Magnetic resonance imaging measures samples related to the Fourier transform of a spatial signal. A simplified discrete model is $y=Fx+\epsilon$, where $x$ contains unknown image values, $F$ maps those values to measurements, and $\epsilon$ represents noise or modeling error. Real MRI adds coil sensitivities, acquisition geometry, and physical effects that this toy model omits.

A reconstruction can minimize

$$\phi(x)=\tfrac12\|Fx-y\|_2^2+\tfrac{\lambda}{2}\|x\|_2^2.$$

The gradient is $F^H(Fx-y)+\lambda x$, where $F^H$ is the conjugate transpose, or **adjoint**, of the forward operator. The adjoint is not generally an inverse. Confusing these two objects produces an incorrect iterative method even if each GPU kernel is fast.

A gradient step is $x_{k+1}=x_k-\eta\nabla\phi(x_k)$. The step size depends on the operator’s spectrum. More sophisticated methods use conjugate gradients on suitable normal equations, proximal operators for nonsmooth regularizers, or other problem-specific algorithms. The reconstruction method should be selected before optimizing its individual kernels.

## The adjoint identity is a strong test

A forward-adjoint pair should satisfy $\langle Fx,z\rangle=\langle x,F^Hz\rangle$ under the chosen complex inner-product convention. This **adjoint test** catches indexing, conjugation, scaling, and interpolation errors. It is especially valuable when a nonuniform Fourier operator is implemented through gridding or interpolation.

Uniformly spaced Fourier samples can use FFT algorithms. Nonuniform samples may require direct evaluation or a nonuniform FFT approximation. A direct kernel computes complex exponentials and sums contributions, often with substantial reuse of coordinates. A faster approximation introduces parameters that control accuracy and work. The performance report must retain that approximation setting.

A GPU implementation can accelerate repeated forward and adjoint applications, but moving the image and measurements between host and device every iteration can erase much of the gain. Keeping the iterative state resident on the device often matters as much as optimizing one arithmetic kernel.

## A potential map is a sum over sources

An electrostatic potential at target position $r$ has a simplified form

$$V(r)=\sum_j\frac{q_j}{\|r-r_j\|}.$$

The physical constant is omitted here because the lesson focuses on computation. A target coinciding with a point source creates a singularity. A finite-size source model, exclusion rule, or softened denominator must be stated explicitly. Adding a small number to the denominator changes the model and is not merely a numerical implementation detail.

A **gather** kernel assigns one target to a thread and loops over sources. The output has one owner, so no atomic output updates are needed. Threads can cooperate to stage source positions and charges because many targets use the same source batch. The resulting structure resembles tiled matrix multiplication, although the interaction formula is different.

A **scatter** kernel assigns sources to workers and adds their contributions to targets. This approach can expose useful source-centered locality but creates competing writers. Atomic updates, private accumulators, or later reductions are needed. Gather and scatter compute the same sum only when they use the same interaction set and arithmetic model.

## Cutoffs reduce work by changing the interaction set

A spatial cutoff ignores sources farther than a chosen radius. A grid of spatial bins can identify nearby sources efficiently. The target then visits only neighboring bins whose geometry intersects its cutoff region. The work depends on source density and the radius, so an uneven distribution can create load imbalance.

A cutoff is justified only when the model or an error-controlled approximation permits it. Long-range electrostatic interactions do not vanish automatically. Methods such as multipole expansions or particle-mesh techniques handle long-range effects with different approximations. This lecture surveys that design choice without presenting a complete electrostatics solver.

Thread coarsening can give one thread several nearby targets. A loaded source can then contribute to several accumulators. This increases reuse but uses more registers. A structure-of-arrays layout for coordinates can improve coalesced source loads when neighboring lanes load neighboring sources. The geometry and the chosen loop nesting determine the actual address pattern.

## Numerical and performance validation belong together

An MRI test should check the forward-adjoint identity, objective decrease under a justified step size, and reconstruction error on a known synthetic signal. A potential-map test should compare gather and scatter on a small source-target set and exercise the chosen singularity policy. Both applications need tolerance-based comparisons because parallel accumulation can change rounding order.

The reference uses a tiny unitary discrete Fourier matrix and a regularized objective. It also computes a softened toy potential by gather and scatter. The softening parameter is part of the displayed model. These experiments explain application structure, but they are not clinical MRI validation or a physically calibrated molecular simulation.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`43_mri_electrostatics.py <../../labs/gpu/43_mri_electrostatics.py>`.

```python
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
```

The recorded run produced the following output.

```text
objective_initial=3.000000, objective_final=0.272727
adjoint_identity_error=0.000e+00
softened_potential= [0.642364, 0.207827, 0.094094]
```

**The reconstruction objective decreases, and the forward-adjoint identity holds within the stated tolerance.** The potential implementations agree because they use the same softened interaction. The reported values belong only to these synthetic inputs.

:::{admonition} Try it 1 :class: exercise

Remove half of the Fourier measurements and compare the reconstruction with and without regularization. Explain why missing data changes the inverse problem even when the forward kernel remains correct. :::

:::{admonition} Try it 2 :class: exercise

Place a target exactly on a source. Compare a stated exclusion policy with the softened model and explain why the results answer different physical questions. :::

## Summary

| Decision | What you should remember |
|---|---|
| MRI operator | The forward map predicts measurements, while its adjoint supports optimization. |
| Gather and scatter | The ownership pattern determines whether output updates conflict. |
| Approximation | A cutoff or softening parameter changes the mathematical model. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Is an adjoint always an inverse?",
    "options": [
      "Yes, the words mean the same operation.",
      "No, the adjoint is defined by an inner-product identity."
    ],
    "answer": 1,
    "explain": "An inverse solves the forward equation, while an adjoint transposes and conjugates the linear map."
  },
  {
    "q": "Which potential design has one owner per target?",
    "options": [
      "Gather has one owner per target.",
      "Scatter always has one owner per target."
    ],
    "answer": 0,
    "explain": "A gather worker accumulates all sources for its own target."
  },
  {
    "q": "What does softening do here?",
    "options": [
      "It changes the interaction formula near a source.",
      "It only changes the file format."
    ],
    "answer": 0,
    "explain": "The denominator includes the stated softening parameter."
  },
  {
    "q": "What should a valid gradient step usually reduce in this test?",
    "options": [
      "It should reduce the stated objective.",
      "It should increase the number of measurements."
    ],
    "answer": 0,
    "explain": "The experiment checks optimization progress for its chosen step size."
  },
  {
    "q": "Can a cutoff ignore long-range interactions without justification?",
    "options": [
      "Yes, distant interactions are always zero.",
      "No, the physical or approximation model must justify it."
    ],
    "answer": 1,
    "explain": "The omitted interactions can contribute materially to the potential."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 17, 18. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
