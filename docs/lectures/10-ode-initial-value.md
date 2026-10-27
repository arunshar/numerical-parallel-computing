# 10. Initial-value problems for ordinary differential equations

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`09-quadrature-differentiation` approximated calculus operations on a known function. An ordinary differential equation specifies how an unknown state changes, so the numerical method must build the trajectory itself. We will compare Euler and Runge-Kutta integration and distinguish a small truncation error from a stable time step.

:::{admonition} Learning goals
- You will identify the state, derivative function, and initial condition of an ODE.
- You will distinguish local truncation error from global error.
- You will analyze absolute stability with the scalar test equation.
- You will compare explicit, implicit, multistep, and adaptive methods.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## An initial condition selects a trajectory

An **initial-value problem** specifies y-prime equal to f of t and y, together with y at the starting time. The state can contain one scalar or many coupled variables. A second-order mechanical equation can become a first-order system by treating position and velocity as separate state components.

$$
y'(t)=f(t,y(t)),\qquad y(t_0)=y_0.
$$

Continuity of f supports existence under suitable conditions. A local Lipschitz condition in the state gives local uniqueness in a standard theorem. A solution can still grow without bound in finite time, so local existence does not guarantee a finite trajectory over any requested interval.

The numerical method should also respect the model's domain. A population state may need to remain nonnegative, or a constrained mechanical state may need to remain on a manifold. A generic integrator does not automatically preserve these properties.

## Euler follows the current slope

**Forward Euler** advances by the current derivative times a time step. It uses one derivative evaluation and no implicit solve.

$$
y_{n+1}=y_n+h f(t_n,y_n).
$$

The exact solution has additional Taylor terms. Starting from the exact state, one Euler step has error on the order of h squared under sufficient smoothness. Over a fixed interval, accumulated error is typically on the order of h. This distinction separates local truncation error from global error.

The lab integrates y-prime equals minus y with initial value one over one unit of time. Its exact answer is exp of minus one. Doubling the number of Euler steps should approximately halve the global error once the asymptotic regime is reached.

## Runge-Kutta methods sample slopes inside the step

A **Runge-Kutta method** combines several derivative evaluations at selected intermediate states. Classical RK4 uses four slopes with weights one, two, two, and one after appropriate scaling. Its global order is four for sufficiently smooth problems.

The intermediate states are numerical constructions, not exact trajectory points. Their coefficients satisfy algebraic order conditions that cancel low-order terms in the local expansion. More stages can support higher order or a larger stability region, but stage count alone does not determine either property.

The lab compares Euler with RK4 on the same decay equation. The reference answer makes the error visible. These tiny problems demonstrate convergence, not a universal efficiency ranking, because a derivative evaluation in a real model may be very expensive.

:::{admonition} Try it
:class: exercise
You can compute the ratio of errors between successive grid sizes. Compare the observed ratios with two for Euler and sixteen for RK4. Explain why the ratio may become unreliable when errors approach floating-point precision.
:::

## Absolute stability controls unwanted amplification

The scalar test equation y-prime equals lambda y isolates how a numerical step amplifies a mode. Its exact step multiplies the state by exp of h lambda. A numerical method instead uses an amplification function R of h lambda.

$$
y'=\lambda y,\qquad y_{n+1}=R(h\lambda)y_n.
$$

Forward Euler has R of z equal to one plus z. For a negative real lambda, its magnitude is at most one only when h times the magnitude of lambda is at most two. A stable continuous decay can therefore become an exploding numerical sequence when the time step is too large.

The lab uses lambda equal to minus one hundred and h equal to 0.03. Forward Euler has amplification minus two, so it alternates sign and doubles magnitude. The exact amplification is approximately 0.0498, which strongly decays. This is a stability failure rather than a small error in a stable trajectory.

## Implicit methods can handle fast decaying modes

**Backward Euler** evaluates the derivative at the unknown next state. Each step requires solving an equation, usually with Newton or another nonlinear method.

$$
y_{n+1}=y_n+h f(t_{n+1},y_{n+1}).
$$

For the scalar linear test equation, its amplification is one divided by one minus z. It remains stable throughout the left half-plane and strongly damps very stiff modes. The lab's amplification is 0.25, which decays even though the step is too large for forward Euler.

That result does not make backward Euler accurate at every large step. Its stable value 0.25 still differs substantially from the exact factor 0.0498. Stability prevents catastrophic growth, while accuracy controls closeness to the true trajectory.

A **stiff problem** contains dynamics that force an explicit method to use steps much smaller than the accuracy needs of the slowly varying solution. Several time scales often create stiffness, but a large coefficient alone is not a complete definition. Implicit methods exchange restrictive stability limits for more expensive steps.

## Multistep methods reuse history

An **Adams-Bashforth method** uses derivatives from previous steps to form an explicit update. An **Adams-Moulton method** includes the next derivative and is implicit. **Backward differentiation formulas** approximate the derivative using several recent state values and are useful for many stiff problems.

Multistep methods need starting values, usually from a one-step method. Variable step sizes change their coefficients and history management. A method must satisfy consistency and zero-stability conditions for a standard convergence result. Order alone is insufficient because an unstable history recurrence can amplify perturbations.

A-stability means the entire left half-plane lies in the stability region. L-stability additionally damps arbitrarily stiff modes strongly. The trapezoidal rule is A-stable but does not have that strong damping limit. These distinctions matter when a fast physical mode should disappear from the resolved trajectory.

## Adaptive integration makes several decisions

An embedded Runge-Kutta pair produces two approximations of different order from shared stages. Their difference estimates local error. A controller adjusts the next step using that estimate, tolerance scales, and safety factors. The tolerances usually combine absolute and relative terms component by component.

Event detection locates a crossing such as impact, threshold exceedance, or a sign change in a diagnostic. Dense output interpolates within accepted steps, but it has its own accuracy properties. An event can be missed if the detection assumptions do not match the event's behavior.

Long-time integration adds concerns beyond pointwise local error. Symplectic methods preserve a discrete geometric structure for Hamiltonian systems and can show better long-term energy behavior than a generic method. They do not necessarily conserve energy exactly at each step. Differential-algebraic equations add constraints and need methods that account for their index and consistency conditions.

:::{admonition} Try it
:class: exercise
You can decrease the stiff example's step to 0.01 and compare forward Euler, backward Euler, and the exact amplification. Then increase it to 0.1. State separately whether each method is stable and whether it resolves the decay accurately.
:::

## Multivalue representations organize changing step sizes

A **multivalue method** carries several pieces of solution information between steps, which can include recent values or scaled derivative-like quantities. A Nordsieck representation stores a vector resembling coefficients of a local Taylor polynomial. Changing the step size then rescales and transforms that history. Predictor-corrector methods can also combine an explicit prediction with an implicit correction.

The stored history is part of the numerical state. Restarting an integration, crossing a discontinuity, or changing order requires updating that state consistently. These implementation details explain why a production variable-order integrator contains much more logic than its core update formula. This section surveys the representation and does not implement a variable-order solver.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/10_ode_initial_value.py
```

```python
"""Separate integration accuracy from absolute stability."""
import math

for n in [10, 20, 40, 80]:
    h = 1/n
    euler, rk4 = 1., 1.
    for _ in range(n):
        euler += h*(-euler)
        k1 = -rk4
        k2 = -(rk4+h*k1/2)
        k3 = -(rk4+h*k2/2)
        k4 = -(rk4+h*k3)
        rk4 += h*(k1+2*k2+2*k3+k4)/6
    print(f"n={n:2d}: Euler error={abs(euler-math.exp(-1)):.6e}, RK4 error={abs(rk4-math.exp(-1)):.6e}")
rate, h = -100., 0.03
forward_factor = 1+h*rate
backward_factor = 1/(1-h*rate)
print(f"forward Euler amplification: {forward_factor:.6f}")
print(f"backward Euler amplification: {backward_factor:.6f}")
print(f"exact amplification: {math.exp(h*rate):.6f}")
assert abs(forward_factor)>1 and abs(backward_factor)<1
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
n=10: Euler error=1.920100e-02, RK4 error=3.332411e-07
n=20: Euler error=9.393519e-03, RK4 error=1.997610e-08
n=40: Euler error=4.647001e-03, RK4 error=1.222742e-09
n=80: Euler error=2.311297e-03, RK4 error=7.562884e-11
forward Euler amplification: -2.000000
backward Euler amplification: 0.250000
exact amplification: 0.049787
```

RK4 reaches much smaller errors than Euler at the same small step count for the smooth decay example. In the stiff test, forward Euler multiplies by -2 while backward Euler multiplies by 0.25. The exact factor is about 0.0498, so the stable backward step still has visible accuracy error.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Order | Local and global error orders describe different quantities. |
| Stability | An integrator must not amplify a decaying mode spuriously. |
| Stiffness | Implicit solves can trade computation for a less restrictive stable step. |

The next lecture, {doc}`11-boundary-value`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is forward Euler global order on a smooth regular problem?",
    "options": [
      "Its global order is one.",
      "Its global order is always four."
    ],
    "answer": 0,
    "explain": "The local one-step error is second order, and accumulation gives first-order global error."
  },
  {
    "q": "Is a stable time step necessarily accurate?",
    "options": [
      "Every stable step reproduces the exact trajectory.",
      "Stability and accuracy are separate requirements."
    ],
    "answer": 1,
    "explain": "Backward Euler is stable in the lab but its amplification differs from the exact factor."
  },
  {
    "q": "What does an implicit step usually require?",
    "options": [
      "It requires solving an equation for the next state.",
      "It requires only evaluating the old derivative once."
    ],
    "answer": 0,
    "explain": "The unknown next state appears inside the derivative function."
  },
  {
    "q": "What do embedded Runge-Kutta pairs estimate?",
    "options": [
      "They directly reveal all physical modeling error.",
      "They estimate local error from two related approximations."
    ],
    "answer": 1,
    "explain": "The paired formulas assess numerical integration error under their assumptions."
  },
  {
    "q": "What can make an explicit method inefficient on a stiff problem?",
    "options": [
      "Stability can require steps much smaller than accuracy needs.",
      "Every derivative evaluation must be random."
    ],
    "answer": 0,
    "explain": "Fast decaying modes can limit the step even when the interesting trajectory changes slowly."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 9 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap09.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
