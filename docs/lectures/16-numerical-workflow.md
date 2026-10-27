# 16. Numerical workflows, multivariate models, and cross-language practice

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`15-iterative-solvers` emphasized that a solver sits inside a larger calculation. A reliable numerical workflow joins modeling, approximation, implementation, and validation without confusing their error sources. This lecture consolidates the CSCI 5302 handouts on Taylor expansion and two-dimensional interpolation, then connects MATLAB-style notation with NumPy and the parallel course.

:::{admonition} Learning goals
- You will construct and check a multivariate Taylor approximation.
- You will explain bilinear and bicubic interpolation on a structured grid.
- You will translate key MATLAB operations into NumPy without changing their meaning.
- You will design a validation record that separates mathematical and implementation claims.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A numerical result needs a chain of explicit choices

A scientific question first becomes a mathematical model. The model then becomes a finite approximation, which becomes an algorithm and a program. Each stage introduces assumptions. A software test at the last stage cannot establish that the earlier model represents the physical system.

A useful workflow begins by writing the quantity of interest, units, input uncertainty, and acceptable output error. The next step identifies conditioning and the likely dominant error sources. The implementation then chooses precision, data representation, a solver, and stopping rules that fit that budget.

Verification checks whether the code solves the stated mathematical problem. Validation checks whether that problem adequately represents the intended real phenomenon. A manufactured solution supports verification. Agreement with independent measurements can support validation, but only within the measurement conditions and uncertainty.

## A gradient is the first local map of a scalar function

For a scalar function of several variables, the **gradient** stores first partial derivatives. Its dot product with a small displacement predicts the first-order change. The **Hessian** stores second partial derivatives and predicts local curvature.

$$
f(x+s)\approx f(x)+\nabla f(x)^T s+\frac12 s^T\nabla^2 f(x)s.
$$

The linear term depends on direction through a dot product. The quadratic term includes cross terms that describe interactions between coordinates. A diagonal-only approximation drops those interactions and can miss important behavior even when each coordinate's individual curvature is known.

The lab uses f of x and y equal to exp of x plus x y plus y squared at the point zero, one. Its gradient is two, two, and its Hessian has rows one, one and one, two. Along the displacement proportional to one, minus one, the first-order term vanishes, making second-order behavior especially visible.

## Taylor remainders provide a checkable prediction

A first-order model has a remainder proportional to the square of the step under bounded second derivatives. A second-order model has a remainder proportional to the cube of the step under suitable third-derivative bounds. These are local statements, and their constants depend on the path and derivatives.

The lab halves the displacement scale and prints errors for both models. The first-order error should shrink by a factor near four, while the second-order error should shrink by a factor near eight in the asymptotic regime. This test can detect a missing cross derivative or an incorrect gradient component.

A **directional derivative test** compares the predicted gradient dot direction with a finite-difference estimate along that direction. A Hessian-vector test can compare changes in gradients. Several seeded directions provide broader coverage than one coordinate direction, but no finite set of tests proves correctness for every input.

:::{admonition} Try it
:class: exercise
You can set the Hessian's off-diagonal entries to zero while leaving the diagonal unchanged. Predict the change in the second-order error rate. Then restore the cross terms and explain why the corrected model captures the interaction between x and y.
:::

## Bilinear interpolation combines four corners

A rectangular grid cell has local coordinates u and v between zero and one. Bilinear interpolation weights its four corner values with products of one-dimensional linear weights. The four weights sum to one, so the method reproduces a constant field exactly.

$$
p(u,v)=(1-u)(1-v)f_{00}+(1-u)v f_{01}+u(1-v)f_{10}+uv f_{11}.
$$

The lab evaluates a cell with corner values zero, one, two, and three at u equal to one-quarter and v equal to three-quarters. The resulting value is 1.25. The method is linear in either coordinate when the other is fixed, but the mixed product means it is not generally a plane in both coordinates together.

Coordinate conventions matter. Array rows can represent y while columns represent x, or a codebase can use the opposite convention. A transposed grid can produce plausible values while placing them at the wrong physical locations. A test field such as x plus two y exposes that mistake more clearly than a symmetric field.

## Bicubic interpolation adds smoothness through more information

A bicubic polynomial has sixteen coefficients because it includes powers zero through three in each coordinate. A common construction uses function values, first derivatives in each direction, and mixed derivatives at the four corners. These sixteen constraints determine the coefficients under a consistent coordinate scaling.

Derivatives expressed in physical coordinates must be scaled when converting to local cell coordinates. Forgetting a grid-spacing factor can leave values correct at corners while distorting the interior. Neighboring cells also need compatible derivative information if the interpolated field should be smooth across their shared edge.

A higher-order interpolant can overshoot observed values and violate positivity or monotonicity. That may be unacceptable for concentrations or probabilities. The numerical method should be chosen for the required physical properties as well as visual smoothness.

## MATLAB and NumPy share ideas but differ in syntax

Both environments support array operations and numerical linear algebra, but similar-looking expressions can mean different things. NumPy's ordinary arrays use zero-based indexing and distinguish elementwise multiplication from matrix multiplication with separate operators.

| Mathematical task | MATLAB expression | NumPy expression |
|---|---|---|
| Elementwise multiplication multiplies corresponding entries. | `A .* B` | `A * B` |
| Matrix multiplication contracts an inner dimension. | `A * B` | `A @ B` |
| Elementwise powers apply the power to every entry. | `A .^ 2` | `A ** 2` |
| A square linear solve finds x from A x equal to b. | `A \ b` | `np.linalg.solve(A, b)` |
| A least-squares solve minimizes a residual. | `A \ b` for suitable rectangular A | `np.linalg.lstsq(A, b, rcond=None)` |
| A transpose exchanges axes without conjugation. | `A.'` | `A.T` |
| A conjugate transpose also conjugates complex entries. | `A'` | `A.conj().T` |

A one-dimensional NumPy array has no separate row or column orientation. Its transpose has the same shape. Explicit two-dimensional shapes are useful when a derivation depends on row and column distinctions. Broadcasting can also expand dimensions implicitly, so shape assertions help detect unintended operations.

## Special functions require numerically suitable formulas

A mathematically correct formula can be a poor implementation for an extreme argument. Functions such as `log1p` and `expm1` preserve accuracy for small changes that ordinary subtraction can lose. Gamma, Bessel, and error functions often require different approximations in different argument regimes.

A production special-function library combines series, asymptotic expansions, recurrence relations, rational approximations, and domain-specific transformations. A short Taylor series around zero is not a reliable implementation over the entire real line. This lecture surveys that strategy rather than reproducing a special-function library.

Integral equations provide another example of a larger workflow. Quadrature can convert an integral operator into a matrix, but that matrix inherits properties of the continuous problem. First-kind equations can be severely ill posed, so a tiny discrete residual can accompany an unstable recovered function. Regularization and data uncertainty must be considered together.

## Numerical and parallel decisions interact

Parallel reductions change summation order, which can change floating-point results. Blocking changes data reuse and can alter the order of operations. Mixed precision can improve throughput but requires an error analysis that connects low-precision stages with higher-precision corrections.

A complete report records problem size, precision, input construction, solver settings, software versions, and validation metrics. Performance measurements also need hardware scope and a clear timing boundary. A CPU toy simulation of a parallel algorithm demonstrates logic, while a compiled distributed or GPU run establishes behavior on that execution platform.

:::{admonition} Try it
:class: exercise
You can create a small nonsymmetric two-dimensional test field and evaluate it with both coordinate conventions. Identify a query point where the transposition error is obvious. Then write a shape and value assertion that would catch the mistake before a larger simulation runs.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/16_numerical_workflow.py
```

```python
"""Check a multivariate Taylor model and bilinear interpolation."""
import numpy as np

f = lambda z: np.exp(z[0])+z[0]*z[1]+z[1]**2
base = np.array([0.,1.])
grad = np.array([2.,2.])
hessian = np.array([[1.,1.],[1.,2.]])
for scale in [0.2,0.1,0.05]:
    step = scale*np.array([1.,-1.])
    first = f(base)+grad@step
    second = first+0.5*step@hessian@step
    print(f"scale={scale:.2f}: first-order error={abs(f(base+step)-first):.6e}, second-order error={abs(f(base+step)-second):.6e}")
u,v = 0.25,0.75
corners = np.array([[0.,1.],[2.,3.]])
weights = np.array([[(1-u)*(1-v), (1-u)*v], [u*(1-v),u*v]])
print(f"bilinear value: {np.sum(weights*corners):.6f}")
print(f"sum of weights: {weights.sum():.6f}")
assert abs(weights.sum()-1)<1e-15
print("elementwise square:", np.array([1.,2.,3.])**2)
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
scale=0.20: first-order error=2.140276e-02, second-order error=1.402758e-03
scale=0.10: first-order error=5.170918e-03, second-order error=1.709181e-04
scale=0.05: first-order error=1.271096e-03, second-order error=2.109638e-05
bilinear value: 1.250000
sum of weights: 1.000000
elementwise square: [1. 4. 9.]
```

The first-order Taylor error decreases by about four when the step halves, while the second-order error decreases by about eight. The bilinear weights sum to one and produce the expected value 1.25. These are synthetic verification checks for formulas and indexing, not validation of a physical interpolation model.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Local models | Gradients and Hessians predict changes at different orders. |
| Coordinates | Interpolation and array layouts need explicit axis conventions. |
| Workflow | Verification, validation, and performance evidence establish different claims. |

The parallel-computing lectures apply these numerical choices to threads, distributed systems, and GPUs.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What does the Hessian add to a Taylor model?",
    "options": [
      "It adds quadratic curvature and cross-coordinate interactions.",
      "It only changes the constant term."
    ],
    "answer": 0,
    "explain": "The quadratic form includes both diagonal and mixed second derivatives."
  },
  {
    "q": "Why do bilinear weights sum to one?",
    "options": [
      "It guarantees exact interpolation of every nonlinear field.",
      "That property preserves constant fields."
    ],
    "answer": 1,
    "explain": "A weighted sum of equal corner values should reproduce the same constant."
  },
  {
    "q": "How many coefficients does a bicubic polynomial have?",
    "options": [
      "It has sixteen coefficients.",
      "It has exactly four coefficients."
    ],
    "answer": 0,
    "explain": "There are four powers in each of two coordinates, giving four times four terms."
  },
  {
    "q": "Which NumPy operator represents matrix multiplication?",
    "options": [
      "The * operator always represents matrix multiplication.",
      "The @ operator represents matrix multiplication."
    ],
    "answer": 1,
    "explain": "Ordinary NumPy arrays use * for elementwise multiplication and @ for matrix products."
  },
  {
    "q": "What does a manufactured-solution test establish?",
    "options": [
      "It supports verification of the stated numerical problem.",
      "It proves that a physical model matches every real system."
    ],
    "answer": 0,
    "explain": "A known synthetic answer checks implementation and discretization without establishing real-world validity."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 1 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap01.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
