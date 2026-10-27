# 3. Least squares, orthogonality, and singular values

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`02-linear-systems` solved compatible square systems. Measurements usually create more equations than unknowns, and noise prevents a line or surface from passing through every observation. Least squares chooses the model whose residual has the smallest squared length. This lecture connects that choice to projections and shows why different algebraic routes have different numerical behavior.

:::{admonition} Learning goals
- You will derive the normal equations from an optimization objective.
- You will explain QR, Householder transformations, and Givens rotations.
- You will use singular values to diagnose rank and sensitivity.
- You will distinguish interpolation, discrete fitting, and continuous approximation.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A fit chooses coefficients within a model family

Suppose three observations have input values zero, one, and two, with responses one, two, and two. We choose the line a plus b x. The design matrix has a column of ones and a column of input values. Its two columns span all prediction vectors that this line family can produce at the observed inputs.

The least-squares objective is the squared two-norm of A beta minus y. Differentiating with respect to beta gives the normal equations. The factor of two disappears when the derivative is set to zero.

$$
\min_\beta\|A\beta-y\|_2^2,\qquad A^T A\beta=A^T y.
$$

For this toy data, the fitted intercept is seven-sixths and the slope is one-half. The residual is negative one-sixth, one-third, and negative one-sixth. Its components do not all vanish, but its dot product with each column of A does. That orthogonality is the defining geometry of the least-squares projection.

## Orthogonality explains what the fit leaves behind

An **orthogonal projector** maps a vector onto a subspace while leaving a perpendicular residual. If Q has orthonormal columns spanning the model space, the fitted vector is Q times Q transpose times y. The projector is symmetric and idempotent, so applying it twice has the same effect as applying it once.

A fit with full column rank has unique coefficients. A rank-deficient model can have many coefficient vectors with the same fitted values. Two identical columns, for example, let one coefficient increase while the other decreases without changing any prediction. A small residual therefore does not prove that every model parameter is identifiable.

Weighted least squares changes the meaning of residual size. Independent measurements with different standard deviations can be scaled by their uncertainties before fitting. Correlated errors require a covariance-aware transformation. An arbitrary weight choice can change the scientific question, so the weights should have a stated interpretation.

## Normal equations are compact but can worsen conditioning

The normal equations turn a rectangular problem into a square system. This seems convenient, but forming A transpose A can lose information through rounding. Its two-norm condition number is the square of the condition number of A when A has full column rank.

$$
\kappa_2(A^TA)=\kappa_2(A)^2.
$$

A design matrix with condition number one million can therefore lead to a normal-equation matrix with condition number one trillion. The lab constructs nearly identical columns and compares those values. Floating-point evaluation can blur the exact squared relationship when the product is already close to singular.

Normal equations can still be useful when conditioning is mild, the matrix has suitable structure, or the application accepts the accuracy tradeoff. A mature solver may use them deliberately in a carefully analyzed algorithm. You should avoid treating the shortest mathematical expression as the safest default implementation.

## QR preserves the geometry

A **QR factorization** expresses A as an orthonormal basis Q times an upper triangular coefficient matrix R. Multiplying the objective by an orthogonal transformation preserves Euclidean length. The least-squares coefficients then satisfy R beta equals Q transpose y.

A **Householder transformation** reflects a vector so that all but one of its components become zero. Applying successive reflections eliminates entries below the diagonal while preserving lengths. A **Givens rotation** rotates two coordinates to eliminate one chosen entry. Givens rotations are useful when data arrive incrementally or when sparse structure makes local changes attractive.

Classical Gram-Schmidt projects a new column against earlier columns and subtracts all projections. Modified Gram-Schmidt updates the vector after each projection and usually maintains orthogonality better. Reorthogonalization can improve either method. Householder QR remains a standard reliable choice for dense least-squares problems.

:::{admonition} Try it
:class: exercise
You can compare the coefficients from QR with coefficients from the normal equations on the nearly dependent matrix. Use a right-hand side with a small controlled perturbation. Compare the fitted vectors as well as the coefficients because parameter error and prediction error can tell different stories.
:::

## Singular values expose the important directions

The **singular value decomposition** writes A as U Sigma V transpose. The columns of V describe input directions, the diagonal entries of Sigma describe stretching, and the columns of U describe the corresponding output directions. Singular values are nonnegative and can be ordered from largest to smallest.

$$
A=U\Sigma V^T,\qquad A^+=V\Sigma^+U^T.
$$

The pseudoinverse replaces nonzero singular values with their reciprocals. It gives the minimum-norm least-squares solution when a unique ordinary solution does not exist. A tiny singular value makes its reciprocal huge, so the numerical definition of zero must reflect arithmetic precision and data uncertainty.

**Truncated SVD** discards directions below a chosen threshold. **Tikhonov regularization** instead penalizes large coefficients and smoothly reduces amplification in weakly identified directions. These methods introduce bias in exchange for lower sensitivity. The regularization strength is part of the statistical or scientific model, not merely a hidden solver parameter.

The SVD also constructs a best rank-k approximation in the two-norm and Frobenius norm by retaining the largest k singular values. This result explains dimensionality reduction and low-rank compression. A low-rank approximation can preserve dominant energy while losing small features that still matter to an application.

## Best approximation extends beyond observed data

Discrete least squares minimizes error at a finite set of observations. Continuous least squares minimizes an integral of squared error, possibly with a weight function. Orthogonal polynomial bases simplify the continuous problem because their cross terms vanish under the chosen inner product.

Legendre polynomials are orthogonal under a uniform weight on minus one to one. Chebyshev polynomials use a different weight and have strong connections to uniform approximation. Changing the interval requires a coordinate transformation. Changing the weight changes which regions receive the most attention.

A **minimax approximation** minimizes the largest absolute error over an interval. Its optimal polynomial typically has an equioscillating error at enough alternating extreme points. The Remez algorithm searches for those points and updates the polynomial. This lecture surveys that algorithm rather than implementing a complete robust Remez solver.

Interpolation forces zero error at selected nodes. Least squares balances aggregate error. Minimax controls the worst error. Those objectives can produce different approximations even when they use the same degree and basis. The application should determine which objective matters.

:::{admonition} Try it
:class: exercise
You can add one observation with a large response to the line-fitting data. Compare ordinary least squares with a manually chosen lower weight for that observation. Explain which assumption about measurement reliability the weighted fit makes.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/03_least_squares_svd.py
```

```python
"""Fit a line and compare rank decisions with singular values."""
import numpy as np

x = np.array([0., 1., 2.])
y = np.array([1., 2., 2.])
A = np.column_stack([np.ones_like(x), x])
Q, R = np.linalg.qr(A, mode="reduced")
beta = np.linalg.solve(R, Q.T@y)
r = y-A@beta
print("intercept and slope:", np.round(beta, 6))
print("residual:", np.round(r, 6))
print(f"normal-equation residual: {np.linalg.norm(A.T@r):.6e}")
assert np.allclose(beta, [7/6, 1/2])
B = np.column_stack([np.ones(5), np.ones(5)+1e-7*np.arange(5)])
s = np.linalg.svd(B, compute_uv=False)
print("singular values:", np.array2string(s, precision=6))
print(f"condition of B: {np.linalg.cond(B):.6e}")
print(f"condition of B.T@B: {np.linalg.cond(B.T@B):.6e}")
for cutoff in [1e-8, 1e-6]:
    print(f"relative cutoff {cutoff:.0e}: numerical rank {np.count_nonzero(s>cutoff*s[0])}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
intercept and slope: [1.166667 0.5     ]
residual: [-0.166667  0.333333 -0.166667]
normal-equation residual: 4.440892e-16
singular values: [3.162278e+00 2.236068e-07]
condition of B: 1.414214e+07
condition of B.T@B: 2.009920e+14
relative cutoff 1e-08: numerical rank 2
relative cutoff 1e-06: numerical rank 1
```

The fit has intercept about 1.166667 and slope 0.5. The normal-equation residual is about 4.44e-16, confirming orthogonality of the fitted residual to the model columns. The nearly dependent design has condition number about 1.41e7, while its normal-equation matrix is near 2.01e14. Changing the relative singular-value cutoff changes its numerical rank from two to one.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Projection | Least squares leaves a residual orthogonal to the model space. |
| QR | Orthogonal transformations preserve lengths without squaring conditioning. |
| SVD | Singular values expose weak directions and numerical rank. |

The next lecture, {doc}`04-eigenvalues`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is orthogonal to the design columns at a least-squares solution?",
    "options": [
      "The residual is orthogonal to the columns.",
      "The response vector must be orthogonal to the columns."
    ],
    "answer": 0,
    "explain": "The normal equations state A-transpose times the residual equals zero."
  },
  {
    "q": "What happens to the two-norm condition number when forming normal equations?",
    "options": [
      "It is always halved.",
      "It is squared in exact arithmetic for full column rank."
    ],
    "answer": 1,
    "explain": "The singular values of A-transpose A are the squared singular values of A."
  },
  {
    "q": "What does a numerical-rank threshold depend on?",
    "options": [
      "It depends on precision, scaling, and the meaningful data uncertainty.",
      "It is always exactly zero."
    ],
    "answer": 0,
    "explain": "Very small singular values can represent directions that the available data cannot identify reliably."
  },
  {
    "q": "Which objective controls the largest approximation error?",
    "options": [
      "Ordinary least squares controls the maximum error directly.",
      "Minimax approximation controls the worst absolute error."
    ],
    "answer": 1,
    "explain": "Least squares controls a sum or integral of squared errors, while minimax controls the supremum."
  },
  {
    "q": "What does a Householder transformation preserve?",
    "options": [
      "It preserves Euclidean lengths.",
      "It preserves every individual coordinate."
    ],
    "answer": 0,
    "explain": "An orthogonal reflection changes coordinates while preserving their two-norm."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 3 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap03.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
