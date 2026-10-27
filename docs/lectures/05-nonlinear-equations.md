# 5. Nonlinear equations and safeguarded iteration

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`04-eigenvalues` used iteration to find a matrix property. Nonlinear equations require the same discipline of defining a target, choosing an update, and checking whether progress is meaningful. We will solve a scalar equation with two different guarantees and then solve a coupled system using a Jacobian.

:::{admonition} Learning goals
- You will compare bracketing methods with derivative-based methods.
- You will distinguish convergence rate from practical reliability.
- You will construct a Newton step for a nonlinear system.
- You will explain safeguarding, stopping criteria, and secant updates.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A root makes a residual vanish

A **root** of f is a point where f equals zero. Finding that root can represent force balance, calibration, a steady state, or a parameter that matches an observation. Multiple roots can exist, and the desired one may require a physical constraint or a bracket.

The intermediate value theorem guarantees a root between two points when a continuous scalar function has opposite signs there. It does not guarantee uniqueness. A positive derivative throughout that interval would provide monotonicity and therefore uniqueness for a sign-changing interval.

The sensitivity of a simple root depends on the derivative at the root. A small residual perturbation produces an approximate root change equal to minus the perturbation divided by f-prime. A nearly flat crossing can therefore make the root highly sensitive. A repeated root has zero derivative and requires a different local analysis.

## Bisection exchanges speed for a strong enclosure

**Bisection** repeatedly halves a sign-changing interval and keeps the half that still brackets a root. After k halvings, its interval width is the original width divided by two to the power k. The midpoint error is at most half that width when the assumptions hold.

$$
|x_k-x_*|\leq\frac{b_0-a_0}{2^{k+1}}.
$$

This bound gives a predictable iteration budget. The method needs continuity and a valid initial bracket, but it does not need derivatives. A discontinuity can imitate a sign change without containing a root, so a sign test alone is not sufficient mathematical evidence.

Floating-point arithmetic eventually prevents the midpoint from differing from either endpoint. A practical implementation checks interval size and stagnation. It also uses an absolute-plus-relative tolerance so that the stopping rule remains sensible near zero and at large magnitudes.

## Newton's method follows a local tangent

A first-order Taylor model replaces f near the current estimate by a tangent line. Setting that line to zero gives the Newton update. The method takes a residual divided by a local slope as its correction.

$$
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.
$$

For x squared minus two, the update averages x with two divided by x. Starting at one produces 1.5, then about 1.4167, and then about 1.4142. Near a simple root with sufficient smoothness, the error is approximately proportional to the square of the previous error. That **quadratic convergence** explains the rapid local improvement.

Newton's method can also leave the useful domain, divide by a tiny derivative, or cycle. Its fast local behavior does not guarantee global success. Repeated roots usually reduce ordinary Newton convergence to linear unless the update accounts for multiplicity.

## Fixed-point iteration exposes the convergence condition

A root problem can often be rewritten as x equals g of x. **Fixed-point iteration** repeatedly evaluates g. A contraction maps nearby points closer together, and a derivative magnitude below one in a suitable invariant interval is a useful sufficient condition.

Different rearrangements of the same equation can have very different convergence. For example, solving x squared equals two by the map two divided by x can oscillate, while Newton's averaged map converges locally. An algebraic identity does not by itself identify a good iteration.

The observed ratio of successive errors helps distinguish linear and superlinear behavior when an accurate reference root is available. In real applications the reference is unknown, so residuals and step sizes are used as proxies. Both proxies require interpretation through sensitivity and scaling.

## Secant methods avoid an explicit derivative

The **secant method** estimates the derivative from two previous function values. It typically has superlinear convergence near a simple root, with order about 1.618 under standard assumptions. It needs one new function evaluation per iteration after initialization, which can make it attractive when derivatives are expensive.

Inverse interpolation fits x as a function of f and evaluates that approximation at zero. Inverse quadratic interpolation uses three recent points. Rational or linear-fractional interpolation can model a different local shape. These methods can produce excellent proposals, but their denominators and extrapolated steps can become unreliable.

A **safeguarded method** combines a fast proposal with a reliable bracket. It accepts interpolation or Newton steps only when they meet progress and location conditions. Otherwise it takes a conservative step such as bisection. This combination explains why a production scalar root solver is often more elaborate than one textbook update formula.

:::{admonition} Try it
:class: exercise
You can start Newton's method for x squared minus two at a very small positive number. Predict the first step before running it. Then reject any Newton proposal outside the current bracket and use a midpoint step instead. Compare the behavior with unguarded Newton iteration.
:::

## Coupled equations need a Jacobian

A vector-valued residual F has one component for each equation. Its **Jacobian** stores partial derivatives, with row i containing derivatives of residual component i. The linearized correction solves J times s equals minus F.

$$
J(x_k)s_k=-F(x_k),\qquad x_{k+1}=x_k+s_k.
$$

The lab solves a unit-circle-related equation and an equality constraint together. Its system is x squared plus y squared equals two and x equals y. Starting near the positive solution leads to the root one, one. The negative root also exists, so the initial point matters.

An analytic Jacobian can be exact at the mathematical level but still contain a programming error. Finite differences approximate derivatives using additional function evaluations. Automatic differentiation applies derivative rules to the program, which avoids finite-difference step selection but still differentiates the program actually written. A directional derivative check is a useful independent diagnostic.

## Broyden updates reuse information

**Broyden's method** updates an approximate Jacobian so that it satisfies the newest secant relation. A common rank-one update changes the matrix as little as possible under a chosen norm while matching the observed residual change along the latest step.

$$
B_{k+1}=B_k+\frac{(y_k-B_k s_k)s_k^T}{s_k^T s_k},\qquad y_k=F(x_{k+1})-F(x_k).
$$

The update saves derivative work, but inaccurate information can accumulate. A robust implementation can rebuild the Jacobian, damp a step, or use a trust region. A line search often evaluates the squared residual norm as a merit function. Convergence of that merit function alone does not prove a root exists, so the final residual remains essential.

Polynomial root finding adds concerns about scaling, clustered roots, and coefficient sensitivity. Companion-matrix eigenvalues provide one route, while specialized polynomial methods exploit additional structure. Deflation by dividing out an approximate root can propagate error, particularly for clustered roots.

:::{admonition} Try it
:class: exercise
You can start the two-variable lab near the negative solution. Then scale one equation by a factor of one thousand. Compare the mathematical roots with the residual norms and discuss why equation scaling changes stopping diagnostics without changing the exact solution set.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/05_nonlinear_equations.py
```

```python
"""Solve a scalar root and a small nonlinear system."""
import math
import numpy as np

lo, hi = 1., 2.
for _ in range(30):
    mid = (lo+hi)/2
    if mid*mid < 2:
        lo = mid
    else:
        hi = mid
print(f"bisection error bound: {(hi-lo)/2:.6e}")
print(f"bisection root: {(lo+hi)/2:.12f}")
x = 1.
for k in range(5):
    x -= (x*x-2)/(2*x)
    print(f"Newton step {k+1}: x={x:.12f}, error={abs(x-math.sqrt(2)):.6e}")
z = np.array([1.2, 0.8])
for _ in range(6):
    f = np.array([z[0]**2+z[1]**2-2, z[0]-z[1]])
    J = np.array([[2*z[0], 2*z[1]], [1., -1.]])
    z += np.linalg.solve(J, -f)
print("system root:", np.round(z, 10))
assert np.allclose(z, [1., 1.])
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
bisection error bound: 4.656613e-10
bisection root: 1.414213561919
Newton step 1: x=1.500000000000, error=8.578644e-02
Newton step 2: x=1.416666666667, error=2.453104e-03
Newton step 3: x=1.414215686275, error=2.123901e-06
Newton step 4: x=1.414213562375, error=1.594724e-12
Newton step 5: x=1.414213562373, error=0.000000e+00
system root: [1. 1.]
```

Thirty bisections reduce the midpoint error bound to about 4.66e-10. Newton reaches a displayed error near 1.59e-12 after four steps on this favorable scalar problem. The coupled solve returns [1,1]. These results illustrate strong local Newton convergence alongside a separate bracket-based guarantee.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Bracketing | A continuous sign-changing interval gives a reliable enclosure. |
| Newton | A derivative model gives fast local convergence near a regular root. |
| Safeguarding | A reliable solver can reject unsafe fast proposals. |

The next lecture, {doc}`06-unconstrained-optimization`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What assumption makes a sign-changing bracket useful?",
    "options": [
      "The scalar function is continuous on the interval.",
      "The function may have any discontinuity."
    ],
    "answer": 0,
    "explain": "A discontinuity can change sign without passing through zero."
  },
  {
    "q": "What is the Newton correction for a system?",
    "options": [
      "It multiplies F by J without a solve.",
      "It solves J s equals minus F."
    ],
    "answer": 1,
    "explain": "The linearized residual F plus J s is set to zero."
  },
  {
    "q": "When is ordinary Newton convergence typically quadratic?",
    "options": [
      "The root is simple and the start is sufficiently close under smoothness assumptions.",
      "Every starting point has a quadratic guarantee."
    ],
    "answer": 0,
    "explain": "Quadratic convergence is a local statement and requires a nonsingular derivative or Jacobian."
  },
  {
    "q": "What information does the secant method reuse?",
    "options": [
      "It requires an exact Hessian.",
      "It uses two recent function values to estimate a derivative."
    ],
    "answer": 1,
    "explain": "A divided difference replaces the explicit scalar derivative."
  },
  {
    "q": "Why should stopping inspect both residual and step size?",
    "options": [
      "A small step can occur without a small residual.",
      "The two quantities are always identical."
    ],
    "answer": 0,
    "explain": "Stagnation or poor scaling can produce tiny steps while the equations remain unsatisfied."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 5 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap05.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
