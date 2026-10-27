# 12. Partial differential equations and sparse spatial operators

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`11-boundary-value` discretized a differential equation along one spatial axis. Partial differential equations describe variation across several independent coordinates, often space and time together. Their numerical methods combine calculus, linear algebra, stability analysis, and parallel data movement. We will study a heat-equation stencil and then survey the major spatial discretization choices.

:::{admonition} Learning goals
- You will distinguish elliptic, parabolic, and hyperbolic model problems.
- You will derive and test a heat-equation finite-difference update.
- You will explain the role of boundary conditions and stability limits.
- You will compare finite differences, finite volumes, finite elements, and spectral methods.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## The PDE type predicts important behavior

An **elliptic equation** often describes an equilibrium field, such as a steady potential governed by Poisson's equation. Its solution depends globally on boundary information. A **parabolic equation** describes diffusion, such as heat spreading over time. A **hyperbolic equation** describes propagation with finite characteristic speeds, such as a wave or transport equation.

These labels summarize useful model classes, but real systems can mix behaviors or change type. Boundary and initial conditions must match the equation's structure. Prescribing too many conditions can make a problem inconsistent, while too few can leave it undetermined.

The one-dimensional heat equation sets the time derivative equal to diffusivity times the second spatial derivative. Diffusivity has units of length squared per time. This dimensional fact already suggests that a grid spacing and time step will interact through a ratio containing h squared.

$$
u_t=\alpha u_{xx}.
$$

## A stencil turns a differential operator into local arithmetic

Combining forward Euler in time with a centered second difference in space gives a three-point update. The nondimensional ratio r equals diffusivity times the time step divided by the square of the spatial spacing.

$$
u_i^{n+1}=u_i^n+r(u_{i-1}^n-2u_i^n+u_{i+1}^n),\qquad r=\alpha\Delta t/\Delta x^2.
$$

The update reads neighboring values from the old time level. Updating the array in place from left to right would mix old and new states and implement a different method. Parallel implementations also need a clear separation between read and write states or a mathematically justified synchronization scheme.

The lab uses periodic boundaries implemented by array shifts. That choice models a ring, not a rod with fixed endpoint temperatures. The sum of the periodic second differences is zero, so the update preserves the discrete total mass up to rounding.

## Stability comes from amplification of Fourier modes

A Fourier mode is an eigenvector of the periodic difference operator. Substitution into the update gives amplification one minus four r times the square of sine of half the grid phase. The highest-frequency mode has amplification one minus four r.

$$
G(\theta)=1-4r\sin^2(\theta/2).
$$

Keeping the magnitude of every amplification factor at most one requires r between zero and one-half for this one-dimensional explicit scheme. The lab compares r equal to 0.4 with r equal to 0.6. A tiny alternating perturbation represents the highest-frequency component and reveals the unstable case after repeated steps.

The bound depends on the discretization and dimension. A two-dimensional update has contributions from both coordinate directions and therefore a different combined restriction. A **CFL condition** expresses a relationship between numerical dependence and physical propagation, but the exact bound must be derived for the actual method.

:::{admonition} Try it
:class: exercise
You can remove the tiny alternating perturbation from the initial condition. Explain why an unstable method can then look acceptable for a while on a smooth input. Restore the perturbation and compare the difference after one hundred steps.
:::

## Implicit time steps produce sparse linear systems

Backward Euler for diffusion moves the spatial operator to the new time level. Each time step solves a linear system involving identity minus the time step times the spatial operator. The matrix is sparse because the stencil connects only neighboring nodes.

An implicit scheme can remove a severe explicit stability limit, but a large time step still reduces temporal accuracy. Crank-Nicolson averages old and new spatial derivatives and has second-order temporal accuracy for smooth solutions. Its weaker damping of stiff modes can cause oscillatory artifacts when the initial data or forcing are not sufficiently smooth.

The **method of lines** discretizes space first and treats the resulting system as an ODE. This lets an adaptive ODE solver control time integration. The spatial operator can make the ODE stiff as the mesh is refined, so the time solver and linear algebra must be selected together.

## Conservative methods track fluxes through boundaries

A **finite-volume method** integrates the PDE over each cell and updates the cell average from fluxes through its faces. Neighboring cells share a face flux with opposite signs, which supports exact discrete conservation. The method is particularly useful for transport and conservation laws.

An upwind method uses information from the direction characteristics travel. It can stabilize transport but introduces numerical diffusion. Higher-order reconstructions reduce diffusion, while limiters reduce nonphysical oscillations near sharp changes. A method that preserves a smooth wave well can still behave poorly at a discontinuity.

Finite differences approximate derivatives directly at grid points. Finite volumes approximate balance over cells. Finite elements approximate a weak formulation using local basis functions. These approaches can produce similar matrices in simple settings, but their interpretations and extensions differ.

## Geometry and smoothness guide the spatial representation

**Finite elements** handle irregular geometry through a mesh of elements and local basis functions. Mesh quality, boundary approximation, quadrature, and function-space choice all affect accuracy. Adaptive refinement can concentrate effort near layers, corners, or large estimated error.

**Spectral methods** use global basis functions such as Fourier modes or Chebyshev polynomials. Smooth solutions can converge rapidly as the basis size grows. Discontinuities can cause Gibbs oscillations, and nonlinear products can introduce aliasing unless the calculation uses an appropriate treatment.

**Domain decomposition** partitions a large spatial problem into subdomains. Each subdomain solves local work and exchanges boundary or interface information. The communication pattern depends on the stencil and decomposition. Strong numerical coupling across interfaces can limit convergence even when the workload is well balanced.

## Sparse solvers become part of the PDE method

A sparse direct solver can provide a reliable answer but may create fill-in that consumes much more memory than the original stencil. Iterative methods use matrix-vector products and preconditioners. Conjugate gradients suit symmetric positive definite systems, while nonsymmetric transport operators need other methods.

**Multigrid** uses a hierarchy of grids. A smoother reduces some error components on a fine grid, and a coarse-grid correction addresses components that appear smooth there. Restriction transfers residual information to a coarse grid, and prolongation transfers a correction back. The idea can be geometric or algebraic.

Verification should separate spatial error, temporal error, algebraic solver error, and modeling error. A manufactured solution helps isolate discretization accuracy. Conservation checks and symmetry checks reveal different classes of bugs. A plausible-looking image is weak evidence because unstable or inconsistent schemes can still create smooth pictures.

:::{admonition} Try it
:class: exercise
You can verify the periodic mass before and after every step for the stable case. Then change one boundary update deliberately and observe how the conservation check responds. State which property that check establishes and which accuracy questions it leaves unanswered.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/12_pde.py
```

```python
"""Test the heat-equation stencil and its amplification factors."""
import numpy as np

n = 40
x = np.arange(n)/n
u0 = np.sin(2*np.pi*x)
amplitudes = {}
for r in [0.4, 0.6]:
    u = u0.copy()
    # A tiny high-frequency component makes instability visible.
    u += 1e-8*(-1.)**np.arange(n)
    for _ in range(100):
        u = u+r*(np.roll(u, -1)-2*u+np.roll(u, 1))
    amplitudes[r] = np.max(np.abs(u))
    print(f"r={r:.1f}: maximum absolute value after 100 steps={np.max(np.abs(u)):.6e}")
    print(f"r={r:.1f}: highest-frequency amplification={1-4*r:.6f}")
assert amplitudes[0.4] < np.max(np.abs(u0))
assert amplitudes[0.6] > 1e4
r = 0.4
u = u0+r*(np.roll(u0, -1)-2*u0+np.roll(u0, 1))
print(f"periodic mass change: {abs(u.sum()-u0.sum()):.6e}")
assert abs(u.sum()-u0.sum())<1e-12
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
r=0.4: maximum absolute value after 100 steps=3.716453e-01
r=0.4: highest-frequency amplification=-0.600000
r=0.6: maximum absolute value after 100 steps=4.100186e+06
r=0.6: highest-frequency amplification=-1.400000
periodic mass change: 5.551115e-17
```

The stable ratio 0.4 yields a decaying solution with maximum magnitude about 0.372 after one hundred steps. The ratio 0.6 amplifies a tiny alternating perturbation to a magnitude above four million. The periodic mass change is near rounding precision, which checks a separate conservation property.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| PDE structure | Equilibrium, diffusion, and propagation need different numerical treatments. |
| Stencil stability | The heat update requires a suitable time-to-space ratio. |
| Solver composition | Spatial, temporal, algebraic, and modeling errors need separate checks. |

The next lecture, {doc}`13-fourier-fft`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the one-dimensional explicit heat stability limit in this lecture?",
    "options": [
      "The ratio r must be at most one-half and nonnegative.",
      "Any positive r is stable."
    ],
    "answer": 0,
    "explain": "The highest-frequency amplification is one minus four r."
  },
  {
    "q": "Why does the periodic stencil preserve total mass?",
    "options": [
      "Every individual point remains unchanged.",
      "Its second-difference contributions sum to zero."
    ],
    "answer": 1,
    "explain": "Neighbor flux contributions cancel in the periodic sum."
  },
  {
    "q": "What does finite volume emphasize?",
    "options": [
      "It emphasizes balance of fluxes across cell faces.",
      "It ignores conservation laws."
    ],
    "answer": 0,
    "explain": "Shared face fluxes cancel between adjacent cells."
  },
  {
    "q": "Why can an unstable method initially look acceptable?",
    "options": [
      "Stability depends only on the plotted color scale.",
      "The initial data may contain little of the unstable mode."
    ],
    "answer": 1,
    "explain": "Small high-frequency errors can take time to grow visibly."
  },
  {
    "q": "What roles do smoothing and coarse correction play in multigrid?",
    "options": [
      "They target complementary error components.",
      "They both only repeat the same fine-grid operation."
    ],
    "answer": 0,
    "explain": "The coarse grid addresses error that a fine-grid smoother reduces slowly."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 11 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap11.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
