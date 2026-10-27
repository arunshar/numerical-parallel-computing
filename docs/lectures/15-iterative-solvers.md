# 15. Iterative linear solvers and preconditioning

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`14-random-simulation` controlled error through sample count. Iterative linear solvers control algebraic error through repeated corrections, which can be much cheaper than a direct factorization for a large sparse matrix. The main challenge is to make each correction useful. We will compare Jacobi iteration with conjugate gradients and connect convergence to matrix structure.

:::{admonition} Learning goals
- You will derive stationary iterations from a matrix splitting.
- You will explain the Krylov subspace behind conjugate gradients and GMRES.
- You will identify when a preconditioner preserves the assumptions of a solver.
- You will choose stopping criteria that account for scale and downstream accuracy.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A matrix-vector product can be enough

A sparse matrix stores only its nonzero entries, while a **matrix-free operator** computes A times x without storing the matrix explicitly. Iterative methods can use either representation. This is valuable for PDE stencils, large graphs, and operators whose entries would be expensive to materialize.

A direct factorization often gives strong reliability and makes repeated right-hand sides inexpensive after setup. An iterative method can use less memory and avoid fill-in, but its iteration count depends on conditioning and the preconditioner. The best choice depends on matrix structure, accuracy needs, reuse, and available hardware.

The lab uses a tridiagonal matrix with twos on the diagonal and minus ones next to it. This is the one-dimensional discrete negative second derivative up to a scale factor. It is symmetric positive definite with zero Dirichlet boundaries, making it suitable for conjugate gradients.

## Stationary iteration applies the same correction rule repeatedly

A matrix splitting writes A as M minus N, where solving with M is easy. The iteration solves M times the new x equal to N times the old x plus b. Its error propagation matrix is M inverse times N.

$$
x_{k+1}=M^{-1}Nx_k+M^{-1}b.
$$

The iteration converges for every starting vector exactly when the spectral radius of the error propagation matrix is below one in the finite-dimensional linear setting. The spectral radius is the largest eigenvalue magnitude. A small residual in one iteration does not establish this property for every right-hand side.

**Jacobi iteration** chooses M as the diagonal of A. Each new coordinate uses only old coordinates, so the update is naturally parallel. **Gauss-Seidel iteration** uses the lower triangular part including the diagonal, so later coordinates use newly updated earlier coordinates. Its sequential dependence complicates parallel execution but can improve convergence per iteration.

Successive over-relaxation blends a Gauss-Seidel correction with the previous state through a relaxation parameter. A suitable parameter can accelerate convergence, while a poor choice can destabilize it. The best parameter depends on the operator and ordering, so a single remembered value is not universally valid.

## Slowly varying error survives simple smoothing

On many elliptic discretizations, Jacobi or Gauss-Seidel reduces rapidly oscillating error more effectively than slowly varying error. Refining the grid makes some slow components even harder to remove, so the iteration count can grow substantially.

This behavior explains their role as **smoothers** in multigrid. A coarse grid represents a slowly varying fine-grid error with fewer unknowns, allowing a cheap global correction. The smoother and coarse correction solve complementary parts of the error problem.

The lab reports the Jacobi residual after one hundred iterations. That result shows that a cheap step can require many repetitions. The comparison is about convergence on a fixed toy problem, not measured wall-clock performance or a claim about optimized parallel implementations.

## Conjugate gradients chooses directions with memory

**Conjugate gradients** solves a symmetric positive definite system by minimizing the associated quadratic objective over expanding Krylov subspaces. Its search directions are conjugate under the A inner product, which prevents the method from undoing earlier progress in exact arithmetic.

The residual starts as b minus A x. Each iteration computes a matrix-vector product, updates the solution and residual, and constructs the next direction from the new residual and the previous direction. The lab exposes these operations directly rather than hiding them inside a solver call.

For an n-dimensional problem in exact arithmetic, CG terminates in at most n iterations. The actual number can be smaller when the right-hand side excites fewer distinct eigenvalue components. Finite precision changes that exact statement, so practical solvers use a residual tolerance and an iteration limit.

A common error bound involves the square root of the condition number. Eigenvalue clustering can make practical convergence better than a worst-case bound suggests. This is one reason a preconditioner that clusters eigenvalues can help even when its effect is not summarized perfectly by one scalar condition number.

:::{admonition} Try it
:class: exercise
You can change the system size from twelve to twenty-four and compare Jacobi's residual after the same number of steps. Then record CG's iterations. Explain why the observed CG count can be below the dimension for this symmetric right-hand side.
:::

## Nonsymmetric matrices need different Krylov methods

**GMRES** builds an orthonormal basis of a Krylov subspace using Arnoldi iteration and chooses the approximation with the smallest residual norm in that subspace. Its basis storage and orthogonalization work grow with iteration count. Restarted GMRES limits memory by periodically discarding most of the basis, but restarting can slow or stall convergence.

BiCG and BiCGSTAB use shorter recurrences for nonsymmetric problems. Their residual behavior can be less smooth, and breakdown conditions need attention. MINRES handles symmetric indefinite systems by minimizing a residual over a Krylov subspace while preserving symmetry assumptions.

Choosing CG simply because a matrix is square is incorrect. Symmetry and positive definiteness are essential to its standard guarantees. A matrix that is almost symmetric because of an assembly bug should be diagnosed rather than passed silently to a symmetric solver.

## Preconditioning changes the geometry of the solve

A **preconditioner** is an operator that approximates the difficult inverse action cheaply. Left preconditioning solves M inverse A x equals M inverse b. Right preconditioning changes variables through x equal to M inverse y. The exact solution is unchanged when the transformations are valid, but residual interpretation can differ.

A diagonal preconditioner rescales coordinates. Incomplete LU or incomplete Cholesky retains only part of the factorization structure. Block methods group strongly coupled variables. Multigrid provides an approximate solve across scales. A useful preconditioner balances setup cost, application cost, memory, and iteration reduction.

Preconditioned CG needs a compatible symmetric positive definite preconditioner and the correct transformed inner-product structure. Applying an arbitrary nonsymmetric inverse approximation can destroy its assumptions. Flexible Krylov methods allow a changing preconditioner when the algorithm accounts for that change.

## Stopping should match the larger calculation

A practical rule compares the residual norm with an absolute tolerance and a relative tolerance times the right-hand-side norm. A scale-aware backward error can be more informative when matrix and solution scales vary widely. The recursive residual inside an iteration can drift from the true residual, so periodic recomputation can help.

A tiny residual can coexist with a substantial solution error when the matrix is ill conditioned. A PDE calculation also has discretization error, and an optimization method has an outer convergence target. The inner linear solve should be accurate enough for those goals, with its assumptions and tolerances recorded.

:::{admonition} Try it
:class: exercise
You can replace the right-hand side with a seeded random vector and compare the CG iteration count. Then calculate the true residual from the final solution rather than trusting the recursively updated residual. Explain what agreement between those two quantities verifies.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/15_iterative_solvers.py
```

```python
"""Compare stationary iteration and conjugate gradients on an SPD system."""
import numpy as np

n = 12
A = 2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
b = np.ones(n)
x = np.zeros(n)
for _ in range(100):
    x += (b-A@x)/np.diag(A)
print(f"Jacobi residual after 100 steps: {np.linalg.norm(b-A@x):.6e}")
x = np.zeros(n);r = b.copy();p = r.copy();rr = r@r
for k in range(n):
    Ap = A@p
    alpha = rr/(p@Ap)
    x += alpha*p;r -= alpha*Ap
    newrr = r@r
    if np.sqrt(newrr)<1e-10:
        break
    p = r+(newrr/rr)*p;rr = newrr
print(f"CG iterations: {k+1}")
print(f"CG residual: {np.linalg.norm(b-A@x):.6e}")
print(f"solution error: {np.linalg.norm(x-np.linalg.solve(A,b)):.6e}")
assert np.allclose(A@x,b)
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
Jacobi residual after 100 steps: 1.692648e-01
CG iterations: 6
CG residual: 0.000000e+00
solution error: 1.776357e-15
```

Jacobi retains a residual near 0.169 after one hundred iterations. CG reaches a zero displayed residual in six iterations for this symmetric toy right-hand side. The solution agrees with the direct solve to about 1.78e-15. The six-step result is specific to the excited spectral components and should not be generalized to every twelve-variable system.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Stationary methods | A fixed error-propagation matrix controls convergence. |
| Krylov methods | An expanding subspace reuses information from earlier products. |
| Preconditioning | An approximate inverse action can improve the geometry at an added cost. |

The next lecture, {doc}`16-numerical-workflow`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Which assumptions support standard conjugate gradients?",
    "options": [
      "The matrix is symmetric positive definite.",
      "The matrix is any square matrix."
    ],
    "answer": 0,
    "explain": "The positive quadratic geometry supports conjugacy and the standard convergence analysis."
  },
  {
    "q": "What is the convergence condition for a stationary linear iteration?",
    "options": [
      "Its diagonal has at least one positive entry.",
      "Its error-propagation spectral radius is below one."
    ],
    "answer": 1,
    "explain": "All error modes must decay under repeated application."
  },
  {
    "q": "Why can restarted GMRES converge more slowly?",
    "options": [
      "Restarting discards useful subspace information.",
      "Restarting computes an exact inverse."
    ],
    "answer": 0,
    "explain": "The bounded memory comes at a possible convergence cost."
  },
  {
    "q": "What does a preconditioner need to balance?",
    "options": [
      "It must always equal the exact inverse.",
      "It balances setup and application costs against iteration savings."
    ],
    "answer": 1,
    "explain": "An exact inverse can be too expensive, so useful approximations target the difficult modes cheaply."
  },
  {
    "q": "Why should a solver occasionally recompute the true residual?",
    "options": [
      "The recursive residual can drift through rounding.",
      "The original matrix changes automatically at every step."
    ],
    "answer": 0,
    "explain": "Direct evaluation of b minus A x checks the accumulated recurrence error."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 11 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap11.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
