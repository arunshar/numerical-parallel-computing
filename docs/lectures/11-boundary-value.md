# 11. Boundary-value problems and spatial discretization

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`10-ode-initial-value` advanced a state from one starting condition. A boundary-value problem constrains a solution at different locations, so the missing information must be determined globally. We will solve a one-dimensional Poisson problem with finite differences and compare that approach with shooting, collocation, and finite elements.

:::{admonition} Learning goals
- You will distinguish initial-value and boundary-value formulations.
- You will derive a centered finite-difference linear system.
- You will compare shooting with global discretization methods.
- You will separate boundary conditions, consistency, and stability.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## Boundary conditions complete the differential equation

A second-order ODE needs two independent conditions, but their locations can differ. A temperature profile along a rod may specify temperature at both ends. A beam model may specify displacement and slope at selected endpoints. These are boundary conditions rather than an initial state for a time-marching problem.

A **Dirichlet condition** specifies the solution value. A **Neumann condition** specifies a derivative or flux. A **Robin condition** specifies a combination of value and derivative. The right choice comes from the physical model, and different choices can change existence and uniqueness.

The toy problem is minus u-double-prime equal to pi squared times sine of pi x on zero to one, with zero values at both ends. Its exact solution is sine of pi x. This manufactured solution gives a direct way to check a discretization without using real measurements.

$$
-u''(x)=\pi^2\sin(\pi x),\qquad u(0)=u(1)=0.
$$

## Shooting turns the boundary condition into a root problem

**Shooting** guesses the missing initial derivative, solves an initial-value problem, and checks the final boundary mismatch. A root solver then adjusts the guess. This approach reuses ODE integrators and nonlinear equation methods already introduced in earlier lectures.

For u-double-prime equal to zero, with u of zero equal to two and u of one equal to five, the exact initial slope is three. A shot with slope two arrives at four, and a shot with slope four arrives at six. The endpoint mismatch is a simple linear function of the guessed slope.

Shooting can become sensitive when a tiny initial-slope error excites a rapidly growing mode. The boundary problem itself may be well behaved even when the associated initial-value propagation is unstable. **Multiple shooting** divides the interval into shorter pieces and solves for matching conditions between them, reducing long-distance amplification.

## Finite differences solve for all grid values together

A centered second difference approximates the second derivative using a point and its two neighbors. Taylor expansion shows second-order truncation error when the solution has enough derivatives.

$$
u''(x_i)\approx\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}.
$$

Substituting this formula into the toy equation creates a tridiagonal system with two on the main diagonal and minus one on each neighboring diagonal, scaled by one divided by h squared. The boundary values enter the right-hand side when they are nonzero.

The lab uses nine, nineteen, thirty-nine, and seventy-nine interior nodes. Each refinement halves h. The maximum solution error should decrease by roughly a factor of four. The residual of the discrete linear system can be near machine precision even while the continuum discretization error is much larger.

That difference is central. Solving a discretized problem accurately is not the same as accurately representing the differential equation. A solver tolerance far below the discretization error may waste work, while a loose solver tolerance can obscure the method's expected convergence rate.

## Consistency needs stability to imply convergence

A consistent discretization approximates the differential operator as the grid is refined. Stability limits how much perturbations in the discrete equations can change the discrete solution. Convergence requires both ideas in an appropriate framework.

A small local truncation error does not automatically yield a small global solution error. The inverse discrete operator can amplify local defects. Boundary treatment can also reduce the overall order even when the interior stencil is high order.

The condition number of the one-dimensional discrete negative second derivative grows as the grid is refined. This does not mean refinement is wrong. It means the algebraic solve becomes more demanding at the same time that the differential approximation improves. Preconditioning and multigrid address that growing algebraic difficulty.

:::{admonition} Try it
:class: exercise
You can replace the zero boundary values with values from a different known smooth solution. Derive the extra right-hand-side terms before changing the code. Check whether the observed convergence rate remains second order.
:::

## Collocation enforces the equation at selected points

A **collocation method** approximates the unknown function in a basis and requires the differential equation to hold at selected collocation points. Polynomial or spline bases provide common choices. The boundary conditions add further equations for the basis coefficients.

Global polynomial collocation can converge very quickly for smooth solutions, but the differentiation matrices are often dense and can be poorly conditioned. Piecewise collocation localizes the representation and allows adaptive refinement. The location of collocation points matters for both interpolation quality and operator conditioning.

Nonlinear boundary problems produce nonlinear algebraic equations after discretization. Newton's method then solves a sequence of linearized systems. A poor initial guess can converge to a different branch or fail, particularly when multiple boundary solutions exist.

## Finite elements weaken the derivative requirement

A **finite-element method** starts with a weak form. Multiplying the differential equation by a test function and integrating by parts moves one derivative from the unknown onto the test function. This allows a broader class of approximate functions and makes boundary fluxes appear naturally.

For the toy problem with zero endpoint values, the weak form seeks u such that the integral of u-prime times v-prime equals the integral of f times v for admissible test functions. Piecewise linear basis functions lead to a sparse stiffness matrix.

$$
\int_0^1 u'(x)v'(x)\,dx=\int_0^1 f(x)v(x)\,dx.
$$

The basis functions live on small elements, so each coefficient interacts mainly with nearby coefficients. Higher-order elements, curved geometry, and adaptive meshes extend the idea. A complete finite-element solver also needs quadrature, boundary handling, assembly, and error estimation.

## Boundary nullspaces need explicit treatment

A pure Neumann problem can determine a solution only up to an additive constant. Its forcing must satisfy a compatibility condition related to the specified flux. The discrete matrix then has a nullspace that a generic nonsingular solve cannot ignore.

One can fix a reference value, impose a mean-zero condition, or solve in a subspace that removes the null mode, depending on the mathematical formulation. Adding a tiny diagonal merely to make the solver run changes the problem and should not be hidden.

Eigenvalue boundary problems seek parameter values for which nontrivial solutions satisfy the boundaries. They connect differential operators to matrix eigenvalue problems after discretization. Mesh refinement must then assess eigenvalue and eigenfunction errors, not only a linear-system residual.

:::{admonition} Try it
:class: exercise
You can record both the maximum continuum error and the discrete residual on every grid. Explain why the residual can remain tiny while the continuum error decreases by a factor near four. Then estimate whether one more grid refinement is more useful than tightening the linear solver tolerance.
:::

## Galerkin methods control the residual in a test space

A **Galerkin method** requires the residual to be orthogonal to a selected test space. When the trial and test spaces use the same basis family, the method is often called a standard Galerkin method. Finite elements provide a local piecewise basis, while a spectral Galerkin method can use global functions. The weak formulation specifies the inner product and boundary terms, so orthogonality here has a precise problem-dependent meaning.

A Petrov-Galerkin method uses different trial and test spaces. That flexibility can improve stability for transport-dominated problems, where a naive symmetric-looking discretization may oscillate. The choice needs a stability argument tied to the operator. Merely increasing the number of basis functions does not guarantee a stable approximation.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/11_boundary_value.py
```

```python
"""Solve a two-point boundary problem by centered finite differences."""
import numpy as np

errors = []
for n in [9, 19, 39, 79]:
    h = 1/(n+1)
    x = h*np.arange(1, n+1)
    A = (2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1))/h**2
    rhs = np.pi**2*np.sin(np.pi*x)
    u = np.linalg.solve(A, rhs)
    error = np.max(np.abs(u-np.sin(np.pi*x)))
    errors.append(error)
    assert np.linalg.norm(A@u-rhs) / np.linalg.norm(rhs) < 1e-10
    print(f"interior nodes={n:2d}, max error={error:.6e}, residual={np.linalg.norm(A@u-rhs):.6e}")
assert all(3.8 < coarse/fine < 4.2 for coarse, fine in zip(errors[:-1], errors[1:]))
# The exact shooting solution for u''=0 is u(x)=a+b*x.
a, target = 2., 5.
slope = target-a
print(f"linear shooting slope for u(0)=2 and u(1)=5: {slope:.6f}")
assert slope == 3.
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
interior nodes= 9, max error=8.265417e-03, residual=5.085161e-14
interior nodes=19, max error=2.058707e-03, residual=2.051279e-13
interior nodes=39, max error=5.142005e-04, residual=1.191366e-12
interior nodes=79, max error=1.285204e-04, residual=6.038032e-12
linear shooting slope for u(0)=2 and u(1)=5: 3.000000
```

The maximum continuum error falls from about 8.27e-3 to 1.29e-4 across three grid halvings, consistent with second-order convergence. The discrete residual stays tiny on every grid. The experiment therefore checks both the linear solve and the distinct spatial approximation error.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Boundary data | Conditions at different locations require a global consistency calculation. |
| Discretization | A small algebraic residual is not the same as a small continuum error. |
| Weak forms | Finite elements and Galerkin methods control residuals through test functions. |

The next lecture, {doc}`12-pde`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What does a Dirichlet condition prescribe?",
    "options": [
      "It prescribes the solution value.",
      "It always prescribes the second derivative."
    ],
    "answer": 0,
    "explain": "Dirichlet data fix values, while Neumann data fix derivatives or fluxes."
  },
  {
    "q": "What unknown does simple shooting often adjust?",
    "options": [
      "It adjusts only the floating-point exponent.",
      "It adjusts a missing initial derivative."
    ],
    "answer": 1,
    "explain": "The initial-value solve maps that guessed derivative to an endpoint mismatch."
  },
  {
    "q": "What sparsity pattern does the centered one-dimensional second derivative produce?",
    "options": [
      "It produces a tridiagonal coupling.",
      "It necessarily produces a fully dense matrix."
    ],
    "answer": 0,
    "explain": "Each equation uses a node and its immediate neighbors."
  },
  {
    "q": "Why can a pure Neumann problem be singular?",
    "options": [
      "Every Neumann condition is inconsistent.",
      "An additive constant can remain undetermined."
    ],
    "answer": 1,
    "explain": "Derivatives do not fix a constant offset, and compatibility also needs checking."
  },
  {
    "q": "What does a Galerkin condition enforce?",
    "options": [
      "It makes the residual orthogonal to a chosen test space.",
      "It makes the residual zero at every physical point."
    ],
    "answer": 0,
    "explain": "The condition is integral or inner-product based rather than pointwise exactness."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 10 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap10.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
