# 2. Linear systems and direct solvers

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`01-errors-floating-point` separated problem sensitivity from arithmetic error. A linear system gives those ideas a concrete form because we can measure the equation residual and perturb the right-hand side. We will solve a small system, explain elimination, and construct a nearly singular example whose answer changes sharply under tiny data changes.

:::{admonition} Learning goals
- You will interpret a matrix equation as a set of simultaneous constraints.
- You will explain elimination, pivoting, and reusable factorizations.
- You will compare residual size with solution error.
- You will choose a direct method that respects matrix structure.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A matrix equation describes shared unknowns

The equation A x equals b collects linear equations into one object. The matrix stores coefficients, the vector x stores unknowns, and b stores observed or prescribed values. A square system has a unique solution exactly when its matrix is nonsingular. A singular system can have no solution or many solutions, depending on whether b belongs to the column space.

For the toy system with rows two x plus y equals one and x plus three y equals two, the exact solution is x equal to 0.2 and y equal to 0.6. The determinant is five, which confirms nonsingularity for this two-by-two example. Determinants are a poor numerical test of singularity for large problems because their magnitude changes dramatically under scaling.

A rectangular system asks a different question. Extra equations can conflict, which leads to least squares. Too few independent equations leave free directions, which require an additional choice such as the minimum-norm solution. You should identify that mathematical problem before selecting a library routine.

## Elimination converts a hard system into two easy systems

**Gaussian elimination** uses row operations to remove entries below each pivot. A lower triangular matrix records the multipliers, and an upper triangular matrix stores the reduced system. With row permutations, the factorization is P A equals L U.

$$
PA=LU,\qquad Ly=Pb,\qquad Ux=y.
$$

Forward substitution solves the lower triangular system from top to bottom. Back substitution solves the upper triangular system from bottom to top. Each substitution requires roughly quadratic work for a dense matrix. Dense factorization requires roughly two-thirds n cubed arithmetic operations, so repeated right-hand sides should reuse the factorization.

You do not usually form A inverse just to solve A x equals b. An inverse needs more work, uses extra storage, and adds opportunities for rounding error. Solving the triangular systems exposes the operations the application actually needs. Explicit inverse entries can still be useful when the scientific question specifically requires them.

## Pivoting controls dangerous divisions

A tiny pivot can produce huge elimination multipliers. Those multipliers can magnify rounding even when the original matrix is well conditioned. **Partial pivoting** selects the largest available magnitude in the current column and exchanges rows before elimination. Complete pivoting also searches columns, but it requires more comparisons and bookkeeping.

Partial pivoting works well for many dense problems, but it is not an unconditional guarantee of small intermediate growth. Special matrices can create a large growth factor. The practical conclusion is to use a mature solver, examine diagnostics, and avoid claiming that pivoting eliminates every numerical risk.

A scaled problem can behave better because its rows and columns have comparable magnitudes. Row scaling changes the units of equations without changing their exact solution. Column scaling changes the units of unknowns and requires transforming the answer back. Scaling should reflect the physical quantities, not merely make one reported residual look small.

## Norms turn vectors and matrices into comparable sizes

A vector norm measures size under a chosen convention. The two-norm measures Euclidean length. The infinity norm measures the largest absolute component. An induced matrix norm bounds how much a matrix can enlarge a vector in the corresponding norm.

The condition number of a nonsingular matrix is the product of the matrix norm and the inverse norm. In the two-norm, it equals the ratio of the largest singular value to the smallest singular value.

$$
\kappa(A)=\|A\|\,\|A^{-1}\|.
$$

A large value identifies a direction that the matrix compresses strongly compared with another direction. Recovering the original vector then amplifies errors along that compressed direction. The relevant sensitivity also depends on the right-hand side, so a worst-case condition number is a bound rather than a prediction for every input.

## A residual asks whether the equations hold

The **residual** is b minus A times the computed solution. A residual close to zero means the computed values nearly satisfy the stated equations. The solution error compares those values with the exact x, which is usually unavailable. A badly conditioned matrix can map a substantial solution error into a very small residual.

A scale-aware backward-error indicator divides the residual norm by the sum of matrix norm times solution norm and right-hand-side norm. This avoids treating a residual of 0.001 as equally good for quantities near one and quantities near a billion.

$$
\eta=\frac{\|b-A\hat{x}\|}{\|A\|\|\hat{x}\|+\|b\|}.
$$

The lab makes two rows almost equal. The resulting matrix has a condition number near four million. A tiny opposing change in the two right-hand-side entries causes a much larger relative change in the solution, even though the perturbed system has a tiny residual.

:::{admonition} Try it
:class: exercise
You can replace the diagonal perturbation of 10 to the power minus 6 with 10 to the power minus 3. Predict the change in condition number and solution sensitivity before running the experiment. Keep the right-hand-side perturbation fixed so the comparison isolates the matrix.
:::

## Structure changes the best solver

A symmetric positive definite matrix admits **Cholesky factorization**, A equals L times L transpose. Cholesky uses about half the factorization work of general dense LU and stores only one triangle. Positive definiteness means every nonzero vector has positive quadratic energy x transpose A x.

A symmetric indefinite system needs a different factorization, often an L D L transpose form with careful pivoting and small diagonal blocks. A banded matrix has nonzeros near its diagonal, so a banded solver avoids much of the work and storage of a dense solver. Sparse elimination can create **fill-in**, which introduces nonzeros absent from the original matrix. Reordering seeks to reduce that fill.

BLAS kernels organize vector, matrix-vector, and matrix-matrix operations. LAPACK assembles dense numerical algorithms from these kernels. Blocked elimination uses matrix-matrix work to improve data reuse, connecting the numerical algorithm to the parallel computing lectures. A low arithmetic count does not by itself guarantee low runtime when memory movement dominates.

## Refinement uses the residual to improve an answer

**Iterative refinement** computes a residual, solves for a correction with the existing factors, and adds that correction to the current solution. Higher precision for the residual can make the correction more informative. The method can improve a solution from lower-precision factors when conditioning and rounding satisfy suitable bounds.

Refinement cannot rescue an arbitrary singular or severely unstable factorization. You should monitor whether the residual and correction decrease. Rank-revealing methods become appropriate when the system is effectively singular at the precision or data uncertainty of the application.

:::{admonition} Try it
:class: exercise
You can calculate the scale-aware backward-error indicator for both systems in the lab. Compare its conclusion with the forward-error measurement. Explain why the two conclusions can both be correct.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/02_linear_systems.py
```

```python
"""Measure residuals and perturbation sensitivity in two linear systems."""
import numpy as np

A = np.array([[2., 1.], [1., 3.]])
b = np.array([1., 2.])
x = np.linalg.solve(A, b)
print("solution:", np.array2string(x, precision=6))
print(f"residual norm: {np.linalg.norm(b-A@x):.6e}")
print(f"condition number: {np.linalg.cond(A):.6f}")
assert np.allclose(x, [0.2, 0.6])
B = np.array([[1., 1.], [1., 1.+1e-6]])
x_true = np.ones(2)
c = B @ x_true
delta = np.array([1e-7, -1e-7])
y = np.linalg.solve(B, c+delta)
print(f"sensitive-system condition number: {np.linalg.cond(B):.6e}")
print(f"relative data change: {np.linalg.norm(delta)/np.linalg.norm(c):.6e}")
print(f"relative solution change: {np.linalg.norm(y-x_true)/np.linalg.norm(x_true):.6e}")
print(f"perturbed residual: {np.linalg.norm(c+delta-B@y):.6e}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
solution: [0.2 0.6]
residual norm: 0.000000e+00
condition number: 2.618034
sensitive-system condition number: 4.000002e+06
relative data change: 4.999999e-08
relative solution change: 2.000000e-01
perturbed residual: 0.000000e+00
```

The first solve returns [0.2, 0.6] with a zero residual at the displayed precision. The sensitive system amplifies a relative right-hand-side change near 5e-8 into a relative solution change of 0.2. Its perturbed residual is still zero at the displayed precision. **A tiny residual does not guarantee a small solution error.**

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Factorization | LU separates expensive setup from cheaper repeated solves. |
| Conditioning | Nearly dependent equations can amplify tiny data changes. |
| Structure | Positive definite, banded, and sparse systems justify specialized methods. |

The next lecture, {doc}`03-least-squares-svd`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Why should repeated right-hand sides reuse a factorization?",
    "options": [
      "The factorization costs much more than triangular substitutions.",
      "The matrix inverse is exact once computed."
    ],
    "answer": 0,
    "explain": "Dense factorization has cubic work, while each triangular solve has quadratic work."
  },
  {
    "q": "Which matrix property supports Cholesky factorization?",
    "options": [
      "The matrix is merely square.",
      "The matrix is symmetric positive definite."
    ],
    "answer": 1,
    "explain": "Positive definiteness provides positive pivots for a standard Cholesky factorization."
  },
  {
    "q": "What does the residual b-Ax measure?",
    "options": [
      "It measures how well the computed values satisfy the equations.",
      "It directly measures distance to the unknown exact solution."
    ],
    "answer": 0,
    "explain": "A matrix can compress a large solution error into a small residual."
  },
  {
    "q": "Why does partial pivoting exchange rows?",
    "options": [
      "It guarantees that every matrix is well conditioned.",
      "It avoids unnecessarily small pivots in the current column."
    ],
    "answer": 1,
    "explain": "Pivoting controls algorithmic growth, while conditioning belongs to the input problem."
  },
  {
    "q": "What does iterative refinement solve for?",
    "options": [
      "It solves for a correction driven by the residual.",
      "It changes the original physical model."
    ],
    "answer": 0,
    "explain": "The correction equation reuses the existing factors and targets remaining algebraic error."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 2 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap02.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
