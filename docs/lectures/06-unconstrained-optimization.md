# 6. Unconstrained optimization and nonlinear least squares

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`05-nonlinear-equations` searched for zeros of a residual. Optimization searches for a point with a small objective, and a zero gradient is only one part of that task. We will examine local optimality, compare step directions, and show why a step size can make an otherwise sensible method diverge.

:::{admonition} Learning goals
- You will distinguish stationary points, local minima, and global minima.
- You will explain gradient descent, Newton steps, and line searches.
- You will interpret curvature and conditioning in convergence.
- You will relate Gauss-Newton and Levenberg-Marquardt to least squares.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## An objective needs a domain and a scale

An **objective function** assigns a value to each candidate decision. The optimization problem also needs a domain because a formula can have poles, invalid logarithms, or physically impossible inputs. Existence of a minimum requires assumptions. A continuous function on a compact set attains a minimum, while a coercive continuous function on Euclidean space also has an attained minimum.

A coercive function grows without bound as the variable norm grows without bound. This prevents the best values from escaping toward infinity. The function exp of minus x has an infimum of zero but never reaches it at any finite x, which shows why existence cannot be assumed.

An unconstrained differentiable local minimum has zero gradient. Its Hessian must be positive semidefinite under standard second-order necessary conditions. A zero gradient and positive definite Hessian are sufficient for a strict local minimum. A zero gradient alone can describe a saddle point or a maximum.

## Convexity turns local information into a global claim

A function is **convex** when its value on a line segment is no greater than the corresponding linear interpolation of endpoint values. For differentiable convex functions, every stationary point is a global minimizer. Strict convexity gives at most one minimizer, although existence still needs attention.

For a quadratic objective, positive definiteness of the Hessian creates an elliptical bowl. Its eigenvectors determine the principal directions, and its eigenvalues determine curvature along those directions. A large ratio between largest and smallest curvature creates a narrow valley that can slow simple gradient steps.

The toy objective is one-half x transpose H x with H diagonal entries one and twenty. Its unique minimizer is the origin. The second coordinate is twenty times as curved as the first, so a step that looks modest in one direction can overshoot badly in the other.

## Gradient descent trades local slope for progress

The **negative gradient** is the direction of steepest local decrease under the Euclidean norm. A gradient step subtracts alpha times the gradient. The word steepest depends on the norm and variable scaling, so changing units changes the geometry of a naive gradient step.

$$
x_{k+1}=x_k-\alpha_k\nabla f(x_k).
$$

For a positive definite quadratic with constant step size, convergence requires alpha between zero and two divided by the largest eigenvalue. In the lab, that upper bound is 0.1. The step 0.11 amplifies the high-curvature component, while 0.05 and 0.09 remain inside the stability interval.

An exact line search minimizes the objective along a selected direction. For the quadratic and the negative gradient direction, the minimizing step can be written from gradient and Hessian products. The best point on one line still need not be the global minimizer, because the next useful direction can change.

## Line searches and trust regions control local models

A practical **line search** seeks sufficient decrease rather than an exact one-dimensional optimum. The Armijo condition compares actual decrease with a fraction of the decrease predicted by the directional derivative. Curvature conditions, such as Wolfe conditions, also prevent steps that are unnecessarily short.

A **trust-region method** limits the distance over which a local model is trusted. It compares actual reduction with predicted reduction and adjusts the region size. This is useful when the quadratic model suggests a very large step or when the Hessian is indefinite. Line searches regulate distance along a chosen direction, while trust regions can change the direction itself.

Derivative-free methods also have a place. Golden-section search shrinks a bracket for a unimodal scalar objective using reusable function evaluations. Parabolic interpolation can accelerate a smooth local search. Direct-search methods explore points without explicit derivatives, but they can become expensive in high dimensions and need their own convergence assumptions.

:::{admonition} Try it
:class: exercise
You can change the larger Hessian entry from twenty to one hundred. Predict a safe constant step range and rerun the experiment. Compare progress in the two coordinates rather than only the final objective.
:::

## Newton uses curvature to change the scale of the step

A second-order Taylor model gives the Newton direction by solving the Hessian system. The method corrects for curvature rather than treating every coordinate equally.

$$
\nabla^2 f(x_k)s_k=-\nabla f(x_k),\qquad x_{k+1}=x_k+s_k.
$$

For the lab's exact quadratic, one Newton step reaches the minimizer up to arithmetic error. For a nonlinear function, the Hessian and gradient change between steps, so the local model is only approximate. A positive definite Hessian gives a descent direction, but an indefinite Hessian can point toward a saddle or increase the objective.

Modified Newton methods alter the Hessian or direction to secure useful descent. Inexact Newton methods solve the linear system only to an appropriate accuracy. Truncated Newton methods often use a Krylov solver and Hessian-vector products, avoiding storage of the full Hessian. Solving the inner problem too accurately can waste work when the outer iterate is still far from the solution.

## Quasi-Newton methods learn curvature from steps

A **quasi-Newton method** updates a Hessian or inverse-Hessian approximation from changes in position and gradient. The secant condition asks the model to reproduce the observed gradient change along the latest step. BFGS maintains a symmetric approximation and can preserve positive definiteness when its curvature condition holds.

Limited-memory BFGS stores a small history of step pairs rather than a dense matrix. This reduces storage from quadratic to roughly linear in the number of variables times the history length. The method still needs sensible scaling, line-search safeguards, and accurate enough gradients.

Nonlinear conjugate-gradient methods combine a new negative gradient with part of the previous direction. They use little memory, but their behavior depends on the update formula and line search. They should not be confused with linear conjugate gradients, whose strongest finite-termination statement assumes a fixed symmetric positive definite linear system in exact arithmetic.

## Nonlinear least squares has extra structure

A nonlinear least-squares objective is one-half the sum of squared residuals. Its gradient is J transpose r. Its Hessian is J transpose J plus a sum of residual-weighted second derivatives. **Gauss-Newton** drops that second term and solves a linear least-squares correction.

This approximation is effective when residuals are small near the solution or the neglected term is otherwise modest. Large residuals or strong nonlinearity can make it inaccurate. **Levenberg-Marquardt** adds damping to the approximate normal equations, combining a more cautious direction with the Gauss-Newton direction as the damping changes.

A small gradient can result from cancellation among residual contributions even when the fit remains poor. You should inspect residual patterns, parameter identifiability, and physical plausibility. An optimizer's termination message reports its stopping rule, not automatic confirmation of a globally best scientific model.

:::{admonition} Try it
:class: exercise
You can apply gradient descent to the same quadratic after rescaling its second variable. Compare convergence before and after scaling, and explain how this resembles preconditioning a linear solve. Then state which optimum remains unchanged under the coordinate transformation.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/06_unconstrained_optimization.py
```

```python
"""Compare stable gradient steps with an exact quadratic Newton step."""
import numpy as np

H = np.diag([1., 20.])
x0 = np.array([2., 2.])
def objective(x):
    return float(0.5*x@H@x)
for alpha in [0.05, 0.09, 0.11]:
    x = x0.copy()
    for _ in range(25):
        x -= alpha*(H@x)
    print(f"alpha={alpha:.2f}, objective after 25 steps={objective(x):.6e}")
xn = x0-np.linalg.solve(H, H@x0)
print(f"Newton objective after one step: {objective(xn):.6e}")
assert np.allclose(xn, 0.)
x = x0.copy()
for _ in range(10):
    g = H@x
    alpha = float((g@g)/(g@H@g))
    x -= alpha*g
print(f"exact-line-search objective after 10 steps: {objective(x):.6e}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
alpha=0.05, objective after 25 steps=1.538900e-01
alpha=0.09, objective after 25 steps=1.848107e-02
alpha=0.11, objective after 25 steps=3.640175e+05
Newton objective after one step: 0.000000e+00
exact-line-search objective after 10 steps: 9.015413e-13
```

The step size 0.11 lies above the quadratic stability threshold 0.1 and produces an objective near 3.64e5 after twenty-five steps. The stable step sizes reduce the objective, and the exact quadratic Newton step reaches zero in one update. **Fast local convergence still requires a suitable model and step policy.**

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Optimality | A stationary point needs additional information to be called a minimum. |
| Step size | Curvature limits useful gradient steps. |
| Structured models | Newton and Gauss-Newton exploit second-order or residual structure. |

The next lecture, {doc}`07-constrained-optimization-svm`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does a zero gradient alone prove a minimum?",
    "options": [
      "It can also identify a saddle or maximum.",
      "It always proves a global minimum."
    ],
    "answer": 0,
    "explain": "Second-order information and convexity determine stronger conclusions."
  },
  {
    "q": "What is the stable constant-step interval for the lab quadratic?",
    "options": [
      "Every positive step is stable.",
      "The step must lie between zero and 0.1."
    ],
    "answer": 1,
    "explain": "The largest Hessian eigenvalue is twenty, so the upper limit is two divided by twenty."
  },
  {
    "q": "Why does Newton solve the toy quadratic in one step?",
    "options": [
      "Its quadratic local model is the exact objective.",
      "Newton always solves every objective in one step."
    ],
    "answer": 0,
    "explain": "The Hessian and gradient exactly describe this fixed quadratic."
  },
  {
    "q": "What term does Gauss-Newton omit from the least-squares Hessian?",
    "options": [
      "It omits J-transpose J.",
      "It omits the residual-weighted second-derivative term."
    ],
    "answer": 1,
    "explain": "The remaining approximation is J-transpose J, which is useful when the omitted term is small."
  },
  {
    "q": "What does a trust region control?",
    "options": [
      "It controls the region where the local model is used.",
      "It guarantees that every model prediction is exact."
    ],
    "answer": 0,
    "explain": "Actual and predicted decrease determine whether the model is trustworthy enough."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 6 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap06.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
