# 7. Constraints, quadratic programs, and support vector machines

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`06-unconstrained-optimization` allowed every point in the domain. Real decisions often obey conservation laws, bounds, budgets, or classification requirements. A constrained optimum can have a nonzero objective gradient because every improving direction may leave the feasible set. This lecture explains that geometry through a small quadratic program and connects it to support vector machines.

:::{admonition} Learning goals
- You will identify feasible directions and active constraints.
- You will derive and check the KKT conditions for a small problem.
- You will compare projection, penalty, barrier, and SQP methods.
- You will interpret the margin and slack variables of a support vector machine.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## Constraints define which decisions are allowed

An equality constraint requires h of x to equal zero. An inequality constraint requires g of x to be at most zero under the convention used here. A **feasible point** satisfies every constraint. An **active inequality** holds at equality at the current point, while an inactive inequality has slack.

Consider minimizing one-half times x squared plus y squared while requiring x plus y to equal one. The unconstrained optimum is the origin, but it violates the equality. The feasible set is a line, and the constrained solution is the point on that line closest to the origin. Symmetry suggests x and y should each equal one-half.

An equality removes allowable directions. At a smooth feasible point, a tangent direction must be orthogonal to the constraint gradient. The objective cannot decrease along any allowable tangent direction at a local optimum. This explains why the objective gradient can be balanced by a combination of constraint gradients.

## Multipliers express the balance of gradients

The **Lagrangian** adds the objective to a weighted combination of constraints. Equality multipliers can have either sign. Inequality multipliers must be nonnegative when inequalities use the convention g at most zero.

$$
L(x,\lambda,\mu)=f(x)+\lambda^T h(x)+\mu^T g(x).
$$

The **Karush-Kuhn-Tucker conditions** combine stationarity, primal feasibility, dual feasibility, and complementary slackness. Complementary slackness requires each inequality multiplier times its constraint value to equal zero. An inactive constraint therefore has zero multiplier at a KKT point.

These conditions are necessary at a regular local optimum under suitable constraint qualifications. They are not automatically necessary for every degenerate formulation. For a convex problem with affine equalities and appropriate regularity, they also provide a global optimality certificate.

The sign and interpretation of a multiplier depend on the constraint convention. A multiplier often describes a local sensitivity of the optimal objective to a constraint value, but the sign must be derived from the actual Lagrangian definition. It is unsafe to assign an economic interpretation from a remembered sign rule alone.

## A quadratic program gives a linear KKT system

An equality-constrained quadratic objective with Hessian H and constraint A x equals b leads to a block linear system. Its top block enforces stationarity, and its bottom block enforces feasibility.

$$
\begin{bmatrix}H&A^T\\A&0\end{bmatrix}
\begin{bmatrix}x\\\lambda\end{bmatrix}
=\begin{bmatrix}-c\\b\end{bmatrix}.
$$

The lab uses H equal to the identity, c equal to zero, and A equal to the row one, one. It recovers x equal to one-half, one-half and a multiplier of minus one-half. The KKT matrix is symmetric but indefinite, even though the objective is strictly convex. Cholesky on the full KKT matrix would therefore be inappropriate.

A **quadratic program** has a quadratic objective and linear constraints. A **linear program** has a linear objective and linear constraints. Linear programs often attain an optimum at a vertex when an optimum exists, but multiple optimal points and unbounded problems are possible. Simplex methods traverse a basis structure, while interior-point methods approach optimality from an interior path.

:::{admonition} Try it
:class: exercise
You can change the equality from x plus y equals one to x plus y equals two. Predict the new primal solution, objective, and multiplier. Check both stationarity and feasibility after solving the modified block system.
:::

## Projection is simple when the feasible set is simple

A **projection** finds the feasible point closest to a proposed point under a chosen norm. Projection onto the nonnegative orthant clips each negative coordinate to zero. Projection onto a box clips to lower and upper bounds. Projection onto a simplex requires a shared threshold because coordinates must also sum to a fixed total.

Projected gradient descent takes an ordinary gradient step and then projects it. The method works well when projection is inexpensive and the objective and feasible set have suitable convexity properties. A difficult projection problem can erase the apparent simplicity of the outer algorithm.

An **active-set method** guesses which inequalities bind at the solution, solves the corresponding equality problem, and updates that guess. This can be efficient when only a few constraints are active and the active set changes little between related solves. Poor guesses can cause many changes or numerical trouble near degeneracy.

## Penalties and barriers change the objective

A **penalty method** adds a cost for constraint violation. A quadratic equality penalty adds a large coefficient times the squared constraint residual. Larger coefficients enforce feasibility more strongly but can make the optimization problem badly conditioned. A finite penalty parameter usually leaves some equality violation unless a special exact-penalty formulation applies.

A **barrier method** adds a term that diverges near an inequality boundary from inside the feasible region. A logarithmic barrier for g below zero uses minus the logarithm of minus g. The barrier parameter gradually decreases as the iterates approach the constrained optimum. The method requires an interior treatment and careful handling of feasible starts.

An augmented Lagrangian combines multiplier updates with a penalty term. It can enforce constraints without requiring the penalty parameter to become as extreme as a pure penalty method. **Sequential quadratic programming** builds a local quadratic objective model and linearized constraints, solves that subproblem, and globalizes the proposed step. These are algorithm families, so practical details determine robustness.

## An SVM turns classification into constrained geometry

A linear **support vector machine** chooses a separating hyperplane w transpose x plus b. The hard-margin formulation minimizes one-half the squared norm of w subject to each labeled score being at least one. The scaling fixes an otherwise arbitrary multiplication of w and b.

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad y_i(w^T x_i+b)\geq1.
$$

The distance from the boundary to the nearest margin plane is one divided by the norm of w. The full distance between the two margin planes is twice that amount. The toy one-dimensional points at minus two, minus one, one, and two are separated by w equal to one and b equal to zero. The points at minus one and one lie on the margin.

Soft-margin SVMs add nonnegative slack variables to allow violations and penalize their total. The penalty coefficient controls the tradeoff between margin width and training violations. The dual formulation expresses the classifier through selected training points and inner products. A positive semidefinite kernel replaces those inner products with a valid implicit feature-space inner product.

A maximum margin in the chosen feature space does not guarantee good predictions on every future distribution. Feature scaling, label noise, and distribution shift remain modeling concerns. The numerical optimizer solves the selected mathematical objective, which is only one part of the learning problem.

:::{admonition} Try it
:class: exercise
You can multiply the toy feature values by ten and recompute a valid hard-margin weight. Predict how the weight norm and geometric margin change. Explain why feature units affect the numerical meaning of margin.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/07_constrained_optimization_svm.py
```

```python
"""Solve an equality-constrained quadratic through its KKT system."""
import numpy as np

H = np.eye(2)
A = np.ones((1, 2))
K = np.block([[H, A.T], [A, np.zeros((1, 1))]])
rhs = np.array([0., 0., 1.])
solution = np.linalg.solve(K, rhs)
x, multiplier = solution[:2], solution[2]
print("primal solution:", x)
print(f"Lagrange multiplier: {multiplier:.6f}")
print(f"constraint residual: {float((A@x)[0]-1):.6e}")
print(f"stationarity residual: {np.linalg.norm(H@x+A.ravel()*multiplier):.6e}")
assert np.allclose(x, [0.5, 0.5])
u = np.array([-2., 3.])
projection = np.maximum(u, 0.)
print("projection onto nonnegative orthant:", projection)
X = np.array([[-2.], [-1.], [1.], [2.]])
y = np.array([-1., -1., 1., 1.])
w, bias = np.array([1.]), 0.
print("hard-margin signed scores:", y*(X@w+bias))
print(f"geometric margin: {1/np.linalg.norm(w):.6f}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
primal solution: [0.5 0.5]
Lagrange multiplier: -0.500000
constraint residual: 0.000000e+00
stationarity residual: 0.000000e+00
projection onto nonnegative orthant: [0. 3.]
hard-margin signed scores: [2. 1. 1. 2.]
geometric margin: 1.000000
```

The KKT solve returns [0.5,0.5] with multiplier -0.5 and zero displayed feasibility and stationarity residuals. The toy SVM scores are [2,1,1,2], so the inner two observations meet the margin constraint exactly. The geometric distance from the boundary to each margin plane is one.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Feasibility | Constraints define allowable decisions. |
| KKT | Stationarity and feasibility must be checked with multiplier conditions. |
| SVM geometry | A norm penalty and signed-score constraints define the margin. |

The next lecture, {doc}`08-interpolation-splines`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Can a constrained optimum have a nonzero objective gradient?",
    "options": [
      "Constraint gradients can balance it.",
      "Every constrained optimum requires a zero objective gradient."
    ],
    "answer": 0,
    "explain": "Only feasible directions matter, so the objective gradient can be normal to the feasible set."
  },
  {
    "q": "What does complementary slackness imply for an inactive inequality?",
    "options": [
      "Its multiplier must be strictly positive.",
      "Its multiplier is zero at a KKT point."
    ],
    "answer": 1,
    "explain": "The constraint value is nonzero, so its product with the multiplier can vanish only when the multiplier is zero."
  },
  {
    "q": "Is the equality-QP KKT matrix positive definite in general?",
    "options": [
      "It is generally indefinite.",
      "A convex objective makes the entire block matrix positive definite."
    ],
    "answer": 0,
    "explain": "The constraint block changes the spectrum even when the objective Hessian is positive definite."
  },
  {
    "q": "What difficulty can a very large quadratic penalty create?",
    "options": [
      "It eliminates all numerical error.",
      "It can worsen conditioning."
    ],
    "answer": 1,
    "explain": "Strong penalties create sharply different curvature scales."
  },
  {
    "q": "What is the hard-margin distance from the decision boundary to one margin plane?",
    "options": [
      "It is one divided by the norm of w.",
      "It is the squared norm of w."
    ],
    "answer": 0,
    "explain": "The signed functional margin is fixed at one, and Euclidean distance divides by the weight norm."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 6 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap06.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
