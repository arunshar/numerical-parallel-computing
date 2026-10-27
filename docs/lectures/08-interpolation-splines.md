# 8. Interpolation, splines, and approximation choices

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`07-constrained-optimization-svm` chose a model subject to exact constraints. Interpolation also imposes exact constraints, requiring an approximating function to match specified data values. Exact agreement at the nodes can still produce poor behavior between them. We will compare node choices and then explain why piecewise methods can be safer.

:::{admonition} Learning goals
- You will distinguish interpolation from least-squares and minimax approximation.
- You will explain monomial, Lagrange, Newton, and barycentric representations.
- You will interpret Runge behavior and the role of node placement.
- You will compare piecewise linear, Hermite, cubic-spline, and B-spline models.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## The interpolation problem chooses a function space

Given n distinct nodes and n values, there is one polynomial of degree at most n minus one that matches them. Distinctness matters because repeated value constraints at the same node are either redundant or inconsistent. Repeated nodes can instead represent derivative constraints in Hermite interpolation.

Uniqueness does not imply good numerical conditioning. The node positions, chosen function space, and evaluation location all influence sensitivity. Extrapolation outside the data range can be especially fragile because the constraints provide little control there.

The monomial representation writes a polynomial as a sum of coefficients times powers of x. Substituting the nodes creates a **Vandermonde matrix**. Solving that system is mathematically valid, but the matrix can be badly conditioned, particularly for high degrees or poorly scaled nodes.

## Several formulas represent the same polynomial

The **Lagrange representation** uses basis polynomials that equal one at one node and zero at all others. The interpolant is the sum of each data value times its corresponding basis. This representation makes the matching property obvious.

The **Newton representation** uses products of successive factors x minus node and coefficients called divided differences. It supports incremental addition of nodes more naturally than rebuilding a monomial system. A nested evaluation analogous to Horner's rule computes the polynomial efficiently.

The **barycentric formula** rewrites Lagrange interpolation as a ratio of weighted sums. Each weight depends only on the nodes, so many data vectors on the same nodes can reuse the weights.

$$
p(x)=\frac{\sum_j w_j y_j/(x-x_j)}{\sum_j w_j/(x-x_j)},\qquad
w_j=\frac{1}{\prod_{k\ne j}(x_j-x_k)}.
$$

The formula needs special handling when x equals a node because its individual terms contain division by zero. The correct value there is simply the stored data value. The lab includes that branch. Its direct weight construction is suitable for small demonstrations, while large problems need scaling and more careful algorithms.

## More nodes can make an interpolant worse

For a sufficiently smooth function, the interpolation error includes a high derivative multiplied by a product of distances from the nodes. That product depends strongly on the node placement. Equally spaced high-degree nodes can create large oscillations near the interval ends.

$$
f(x)-p(x)=\frac{f^{(n)}(\xi)}{n!}\prod_{j=0}^{n-1}(x-x_j).
$$

The unknown point xi depends on x, so this formula is an analysis tool rather than a directly computable error estimate. The **Runge phenomenon** illustrates how increasing degree on equally spaced nodes can fail to give uniform convergence for an otherwise smooth function.

Chebyshev nodes cluster toward the interval ends and control the relevant polynomial growth more effectively. The lab compares fifteen equally spaced nodes with fifteen Chebyshev-Lobatto nodes on minus one to one. It uses the smooth function one divided by one plus twenty-five x squared.

The **Lebesgue constant** measures how much the interpolation operator can amplify errors in the data values under the maximum norm. A good numerical evaluation formula reduces arithmetic problems, but it does not change the inherent conditioning of the chosen interpolation problem.

:::{admonition} Try it
:class: exercise
You can increase the number of nodes from fifteen to twenty-five. Predict which error curve improves and which can worsen. Then add tiny alternating perturbations to the data values and compare their amplification between the two node sets.
:::

## Piecewise polynomials keep the degree local

**Piecewise linear interpolation** joins neighboring data points with straight segments. It is local, inexpensive, and easy to explain. Its first derivative jumps at the nodes, which can be unsuitable when the interpolated quantity feeds a derivative-based model.

**Hermite cubic interpolation** uses values and slopes at the endpoints of each interval. Four constraints determine a cubic on each interval. Choosing slopes carefully can preserve monotonicity and reduce overshoot. A generic cubic interpolation method does not automatically preserve positivity, monotonicity, or physical bounds.

A **cubic spline** joins cubic pieces while matching first and second derivatives at interior knots. Additional boundary conditions complete the system. Natural boundary conditions set the second derivative to zero at the endpoints. Clamped conditions specify endpoint slopes. Periodic conditions connect both ends when the underlying phenomenon is periodic.

The spline equations often form a banded or tridiagonal system. That structure makes them much cheaper than a generic dense solve. The boundary choice can noticeably influence behavior near the ends, so it should be tied to the application rather than treated as decoration.

## B-splines provide a local basis

A **B-spline basis** has compact support, so changing one coefficient affects only a limited region. The knot sequence controls where pieces meet and how smooth they are. Repeating a knot reduces continuity and can represent a deliberate corner or derivative discontinuity.

B-splines separate the basis from the fitting objective. The same basis can support interpolation, least squares, or a smoothing penalty. A **smoothing spline** balances data agreement against roughness, often measured by an integral of a squared derivative. The smoothing parameter decides how much variation is treated as signal rather than noise.

Local support also leads to sparse matrices and efficient updates. The advantage is computational as well as interpretive. A coefficient attached to a local basis function has a more localized effect than a high-degree monomial coefficient.

## Approximation objectives should match the question

Interpolation is appropriate when exact node agreement is meaningful and data are sufficiently reliable. Least squares is appropriate when errors should be averaged across observations or a continuous interval. Minimax approximation is appropriate when the worst error must be controlled. These objectives can use the same basis while selecting different coefficients.

Orthogonal polynomials can improve approximation calculations by reducing coupling between coefficients under a chosen inner product. Chebyshev expansions are useful for smooth functions on finite intervals, and their coefficient decay can indicate how much degree is needed. Rational approximations can represent poles or sharp behavior more efficiently than polynomials, but their denominators introduce new stability and domain concerns.

Two-dimensional interpolation extends these decisions. Bilinear interpolation uses four corners of a rectangular cell and is linear in each coordinate separately. Bicubic interpolation uses a cubic in each coordinate and needs sixteen coefficients, determined from values and derivative-related information or an equivalent data arrangement. A tensor-product construction assumes a structured grid and does not directly solve interpolation on arbitrary scattered points.

:::{admonition} Try it
:class: exercise
You can compare the high-degree interpolants with piecewise linear interpolation on the same nodes. Inspect the maximum error and the behavior near the endpoints. Explain why a lower-degree local model can outperform a globally exact high-degree fit between its nodes.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/08_interpolation_splines.py
```

```python
"""Compare polynomial node choices and piecewise linear interpolation."""
import numpy as np

def barycentric(nodes, values, query):
    weights = np.array([1/np.prod(nodes[j]-np.delete(nodes, j)) for j in range(len(nodes))])
    output = []
    for t in query:
        hit = np.flatnonzero(np.isclose(t, nodes, atol=1e-14, rtol=0))
        if len(hit):
            output.append(values[hit[0]])
        else:
            terms = weights/(t-nodes)
            output.append((terms@values)/terms.sum())
    return np.array(output)

f = lambda x: 1/(1+25*x*x)
grid = np.linspace(-1, 1, 2001)
n = 15
for label, nodes in [("equally spaced", np.linspace(-1, 1, n)),
                     ("Chebyshev", np.cos(np.arange(n)*np.pi/(n-1)))]:
    error = np.max(np.abs(barycentric(nodes, f(nodes), grid)-f(grid)))
    print(f"{label} maximum error: {error:.6e}")
nodes = np.linspace(-1, 1, n)
linear = np.interp(grid, nodes, f(nodes))
print(f"piecewise linear maximum error: {np.max(np.abs(linear-f(grid))):.6e}")
assert np.allclose(barycentric(nodes, f(nodes), nodes), f(nodes))
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
equally spaced maximum error: 7.194881e+00
Chebyshev maximum error: 5.350804e-02
piecewise linear maximum error: 5.975961e-02
```

The equally spaced degree-fourteen interpolant has a maximum sampled error near 7.19 on the Runge example. The Chebyshev-node error is about 0.0535, and piecewise linear interpolation is about 0.0598. Exact node agreement therefore does not establish useful behavior between nodes.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Representation | Equivalent polynomial formulas can differ in arithmetic stability. |
| Node placement | The nodes control inherent interpolation sensitivity. |
| Local models | Splines and piecewise polynomials avoid one global high-degree fit. |

The next lecture, {doc}`09-quadrature-differentiation`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "How many distinct values determine a polynomial of degree at most n minus one?",
    "options": [
      "There are n distinct interpolation nodes.",
      "There is always only one node."
    ],
    "answer": 0,
    "explain": "The polynomial space has n coefficients, and distinct nodes give independent value constraints."
  },
  {
    "q": "What should a barycentric evaluator return at an exact node?",
    "options": [
      "It should perform the singular division directly.",
      "It should return the stored data value."
    ],
    "answer": 1,
    "explain": "The apparent singularity is removable, and the interpolation constraint gives the value."
  },
  {
    "q": "Why can equally spaced high-degree interpolation oscillate?",
    "options": [
      "The interpolation problem can amplify endpoint behavior and data errors.",
      "The uniqueness theorem has failed."
    ],
    "answer": 0,
    "explain": "Uniqueness and approximation quality are different properties."
  },
  {
    "q": "What does a natural cubic spline specify at the endpoints?",
    "options": [
      "Its endpoint values must both be zero.",
      "Its endpoint second derivatives are zero."
    ],
    "answer": 1,
    "explain": "Natural conditions complete the spline system by setting curvature conditions."
  },
  {
    "q": "What is a principal benefit of a B-spline basis?",
    "options": [
      "Its basis functions have local support.",
      "Every coefficient changes the entire interval equally."
    ],
    "answer": 0,
    "explain": "Local support limits interactions and supports sparse calculations."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 7 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap07.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
