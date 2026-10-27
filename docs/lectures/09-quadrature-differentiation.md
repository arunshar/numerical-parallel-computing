# 9. Numerical integration and differentiation

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`08-interpolation-splines` built functions that match sampled values. Integrating or differentiating those approximations produces practical formulas for calculus on a computer. Integration usually smooths local noise, while differentiation magnifies it. We will derive familiar quadrature rules, compare their convergence, and explain when refinement needs more than a smaller step.

:::{admonition} Learning goals
- You will derive quadrature from polynomial approximation.
- You will compare composite, adaptive, Gaussian, and extrapolation methods.
- You will explain the conditioning difference between integration and differentiation.
- You will separate discretization error from floating-point and measurement error.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## Quadrature replaces a continuum with weighted samples

A **quadrature rule** approximates an integral by a weighted sum of function values. The nodes specify where to sample, and the weights specify how much each sample contributes. Integrating the linear interpolant on one interval gives the trapezoid rule. Integrating a quadratic interpolant through an interval's endpoints and midpoint gives Simpson's rule.

A composite rule applies a low-degree rule across many subintervals. This keeps the local approximation simple and lets the total number of samples grow. High-degree Newton-Cotes rules on equally spaced nodes can have troublesome weights and stability, so a higher polynomial degree is not automatically the best route to accuracy.

For a sufficiently smooth function on a fixed interval, the composite trapezoid error is typically proportional to the square of the spacing. Composite Simpson error is typically proportional to the fourth power. Halving the spacing should therefore reduce their errors by factors near four and sixteen once the asymptotic regime is reached.

## A convergence table is a practical diagnostic

The lab integrates exp of x from zero to one, whose exact integral is e minus one. It compares trapezoid and Simpson approximations on grids with four through sixty-four subintervals. The reference is analytic, so the displayed discrepancies are actual errors for this toy example.

A convergence table can reveal an implementation mistake when the observed rate disagrees with theory. It can also reveal that the theorem's assumptions are missing. A function with a corner, a discontinuity, or a singular derivative may converge more slowly. Rounding eventually limits the improvement from refinement.

The number of subintervals must meet the rule's layout requirements. Composite Simpson uses an even number so neighboring pairs of subintervals form quadratic panels. Applying the coefficients mechanically to an incompatible grid does not preserve the rule's accuracy claim.

## Gaussian nodes spend samples more efficiently

A Gaussian rule chooses both nodes and weights so that n samples integrate polynomials through degree two n minus one exactly under a specified weight. Gauss-Legendre quadrature uses the uniform weight on minus one to one. Other orthogonal polynomial families match other weights and domains.

The two-point Gauss-Legendre rule uses nodes at plus and minus one divided by square-root of three, each with weight one. It integrates x squared exactly because the two squared node values sum to two-thirds. The lab verifies this identity numerically.

$$
\int_{-1}^{1}f(x)\,dx\approx f(-1/\sqrt{3})+f(1/\sqrt{3}).
$$

A change of variables maps a finite interval to the standard interval and scales the weights. Gaussian exactness refers to a polynomial class, not to every smooth function. Endpoint singularities or oscillations can require a different transformation, subdivision, or specialized rule.

**Clenshaw-Curtis quadrature** uses cosine-spaced nodes and connects naturally to Chebyshev expansions. Nested node choices can reuse function evaluations during refinement. Gauss-Kronrod extensions add nodes to a Gaussian rule and provide paired estimates that support adaptive error control.

:::{admonition} Try it
:class: exercise
You can replace exp of x with the absolute value of x minus 0.3. Compare the apparent convergence orders with the smooth example. Then split the interval exactly at 0.3 and explain why the same local rule behaves better on each smooth piece.
:::

## Adaptivity allocates work where the function is difficult

An **adaptive integrator** compares approximations at different resolutions or orders and refines where the estimated error is large. A simple strategy compares one panel with two half-panels. The difference estimates local error only under assumptions about the leading error term.

Absolute and relative tolerances work together because an integral can be close to zero even when its positive and negative contributions are large. A relative tolerance alone becomes unreasonable near zero. An error estimate is also not an infallible certificate because a narrow feature can lie between all sampled points.

A robust implementation limits subdivisions, detects nonfinite function values, and reports whether it met the requested tolerance. You should inspect warnings and integration domains. A routine that returns a number after exhausting its budget has not necessarily achieved the requested accuracy.

## Extrapolation cancels a known leading error

Suppose an approximation has error C times h to the power p plus higher-order terms. Combining values at h and h divided by two can cancel that leading term. **Richardson extrapolation** uses the known order p to form the combination.

$$
A_{\mathrm{extra}}=\frac{2^p A(h/2)-A(h)}{2^p-1}.
$$

Romberg integration applies this idea repeatedly to trapezoid estimates. The method relies on an appropriate smooth error expansion. It can perform poorly when nonsmoothness invalidates that expansion or when arithmetic noise dominates the difference between estimates.

The lab also applies Richardson extrapolation to centered differences, where p equals two. The resulting derivative estimate has fourth-order truncation error under sufficient smoothness. Its subtraction still amplifies rounding at small step sizes, so extrapolation does not eliminate the finite-precision limit.

## Differentiation amplifies noise

A difference quotient divides a change in measured values by a small distance. If each value has independent uncertainty on the order of epsilon, the derivative's noise contribution is on the order of epsilon divided by h. This is a problem-conditioning issue when the input consists of noisy samples, not merely a floating-point implementation issue.

Forward differences have first-order truncation error, while centered differences have second-order truncation error for smooth functions. Higher-order formulas use more samples and can amplify noise through larger or alternating coefficients. Fitting a local polynomial and differentiating that fit can smooth noise, but it introduces a window-size choice and modeling assumptions.

Complex-step differentiation evaluates a suitable analytic function at a small imaginary perturbation and uses the imaginary component divided by the step. It avoids the subtractive cancellation of ordinary finite differences. The technique does not apply unchanged to nonanalytic operations such as absolute value, branch-dependent code, or functions that discard imaginary parts.

## Domains and dimensions change the method

Improper integrals can have infinite limits or endpoint singularities. A transformation can map an infinite domain to a finite one, but it may create sharp endpoint behavior. Oscillatory integrals need enough information about frequency, and cancellation can make relative accuracy difficult.

Tabulated data provide only existing samples, so an adaptive method cannot request new evaluations. The numerical task then depends on an interpolation or stochastic model between samples. Repeated quadrature in multiple dimensions suffers from rapidly growing sample counts, which motivates sparse-grid and Monte Carlo alternatives.

Integral equations ask for a function that appears inside an integral, rather than merely one integral value. Nyström methods replace the integral with quadrature and solve for values at nodes. Projection methods approximate the unknown in a basis. These topics connect quadrature to linear systems and require separate analysis of discretization and conditioning.

:::{admonition} Try it
:class: exercise
You can add a fixed small perturbation to each sampled function value before differentiating. Compare the error as the step shrinks. State whether the resulting failure comes from the derivative formula, the data uncertainty, or both.
:::

## Automatic differentiation follows the program's operations

**Automatic differentiation** applies the chain rule to the elementary operations in a program. Forward mode propagates directional derivatives alongside values. Reverse mode propagates sensitivities backward from an output and is especially useful when a scalar objective depends on many parameters. The resulting derivative has arithmetic error, but it does not have the finite-difference truncation error from an arbitrary step size.

Automatic differentiation differentiates the implemented program, including its chosen operations and control flow. Nondifferentiable points, branch changes, custom kernels, and invalid domains still need interpretation. A derivative can be correct for the code while the code represents the wrong mathematical model. Directional finite-difference checks remain useful as an independent diagnostic at carefully chosen smooth points.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/09_quadrature_differentiation.py
```

```python
"""Compare quadrature convergence and a Gaussian rule."""
import math
import numpy as np

exact = math.e-1
for n in [4, 8, 16, 32, 64]:
    x = np.linspace(0, 1, n+1)
    y = np.exp(x)
    trapezoid = (y[0]/2+y[-1]/2+y[1:-1].sum())/n
    simpson = (y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())/(3*n)
    print(f"n={n:2d}: trapezoid error={abs(trapezoid-exact):.6e}, Simpson error={abs(simpson-exact):.6e}")
nodes = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])
quad = float(np.sum(nodes**2))
print(f"two-point Gauss integral of x^2 on [-1,1]: {quad:.12f}")
assert abs(quad-2/3)<1e-14
for h in [0.2, 0.1, 0.05]:
    d1 = (math.exp(h)-math.exp(-h))/(2*h)
    d2 = (math.exp(h/2)-math.exp(-h/2))/h
    richardson = (4*d2-d1)/3
    print(f"h={h:.2f}: central error={abs(d1-1):.6e}, extrapolated error={abs(richardson-1):.6e}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
n= 4: trapezoid error=8.940076e-03, Simpson error=3.701346e-05
n= 8: trapezoid error=2.236764e-03, Simpson error=2.326241e-06
n=16: trapezoid error=5.593001e-04, Simpson error=1.455928e-07
n=32: trapezoid error=1.398319e-04, Simpson error=9.102726e-09
n=64: trapezoid error=3.495839e-05, Simpson error=5.689702e-10
two-point Gauss integral of x^2 on [-1,1]: 0.666666666667
h=0.20: central error=6.680013e-03, extrapolated error=3.337304e-06
h=0.10: central error=1.667500e-03, extrapolated error=2.083953e-07
h=0.05: central error=4.167188e-04, extrapolated error=1.302180e-08
```

The trapezoid error decreases by about four under each grid halving, while Simpson error decreases by about sixteen in this smooth example. The two-point Gaussian rule returns two-thirds for the quadratic integral. Richardson extrapolation sharply reduces the centered-difference truncation error before rounding becomes important.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Quadrature | Weighted samples approximate an integral. |
| Adaptivity | Local error estimates guide where extra samples are spent. |
| Differentiation | Small differences amplify both measurement noise and arithmetic error. |

The next lecture, {doc}`10-ode-initial-value`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "How many polynomial degrees does an n-point Gaussian rule integrate exactly?",
    "options": [
      "It integrates through degree two n minus one under its weight.",
      "It integrates every function exactly."
    ],
    "answer": 0,
    "explain": "The nodes and weights satisfy polynomial moment conditions, which do not cover arbitrary functions."
  },
  {
    "q": "What reduction is expected for second-order error when h is halved?",
    "options": [
      "The error is approximately doubled.",
      "The error is approximately divided by four."
    ],
    "answer": 1,
    "explain": "The leading term is proportional to h squared."
  },
  {
    "q": "What does Richardson extrapolation require?",
    "options": [
      "It requires an appropriate leading error expansion and order.",
      "It works equally well for every discontinuous function."
    ],
    "answer": 0,
    "explain": "The combination cancels a specific power of h."
  },
  {
    "q": "Why does a relative tolerance need an absolute companion?",
    "options": [
      "Relative error never uses the integral magnitude.",
      "The integral can be near zero."
    ],
    "answer": 1,
    "explain": "A near-zero reference makes a relative-only target unreasonable or undefined."
  },
  {
    "q": "What does automatic differentiation differentiate?",
    "options": [
      "It differentiates the implemented operations through the chain rule.",
      "It proves that the physical model is correct."
    ],
    "answer": 0,
    "explain": "A correct program derivative still depends on whether the program expresses the intended model."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 8 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap08.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
