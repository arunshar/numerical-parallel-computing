# 1. Errors and floating-point arithmetic

Prepared for Arun Sharma. Codex drafted this study edition.

A computer stores a finite description of a number. Numerical computing studies how that limitation interacts with approximations, uncertain measurements, and the algorithms we choose. A faster answer has little value when rounding destroys the digits that the application needs. This lecture builds a vocabulary for separating those problems and tests two formulas that are equal in exact arithmetic.

:::{admonition} Learning goals
- You will distinguish input error, approximation error, and rounding error.
- You will explain conditioning, stability, and forward versus backward error.
- You will diagnose cancellation and choose a more accurate formula.
- You will measure the tradeoff between step size and differentiation error.
:::

## Setup

The labs use Python 3 and NumPy on a CPU. Each numerical lecture includes a standalone script, its captured output, and assertions for a mathematical property. You can copy its Python block into a notebook or run the named file from the repository root. The examples use synthetic inputs chosen to reveal a mechanism. They are not benchmarks of a production solver.

| Shared choice | Meaning |
|---|---|
| Python and NumPy provide the runtime. | The experiments use ordinary binary64 arrays unless a page says otherwise. |
| Small synthetic problems provide the data. | Every input is visible, so you can reproduce the calculation. |
| Residuals and known answers provide checks. | A successful run checks mathematical behavior rather than execution alone. |
| Saved outputs provide evidence. | Last-bit differences can occur across machines and numerical libraries. |

## Error has several sources

**Absolute error** measures the distance between an approximation and a reference. **Relative error** divides that distance by the reference magnitude. If a temperature estimate is 20.1 rather than 20 degrees, its absolute error is 0.1 degree. Relative error requires care near zero, where a small absolute discrepancy can produce a huge ratio.

A physical model introduces modeling error when it leaves out a relevant effect. Measured inputs introduce data error. Replacing an infinite series or a continuous derivative with finitely many operations introduces **truncation error**. Finite arithmetic introduces **rounding error**. These errors can interact, so decreasing one contribution does not guarantee a better final answer.

A useful error budget assigns an acceptable scale to each contribution. A sensor with one percent uncertainty rarely benefits from a calculation that spends hours resolving its result to fourteen digits. An accurate numerical algorithm also cannot repair a wrong physical model. You should state the reference against which every reported error is measured.

## Conditioning belongs to the problem

A **condition number** measures how much a small input perturbation can change the exact answer. For a scalar function, the local relative condition number is approximately the magnitude of x times f-prime of x divided by f of x.

$$
\kappa_f(x)=\left|\frac{x f'(x)}{f(x)}\right|.
$$

The derivative measures local sensitivity, and the scaling converts absolute changes into relative changes. The formula requires nonzero denominators and is a local approximation. A large condition number means the problem can amplify input uncertainty even if the algorithm performs every operation perfectly.

For example, subtracting two nearly equal measured quantities can be sensitive because their difference is much smaller than either measurement. If both measurements have an uncertainty of 0.01 and their difference is 0.001, the difference is poorly determined. More arithmetic precision cannot recover information that the measurements never contained.

## Stability belongs to the algorithm

**Forward error** compares the computed answer with the exact answer for the stated input. **Backward error** asks how much the input would need to change for the computed answer to become exact. A backward-stable method produces the exact answer to a nearby problem. That statement becomes useful when the nearby problem is also close in its answer.

The usual relationship is that forward error is bounded approximately by conditioning times backward error. A backward-stable solve of a badly conditioned system can therefore have substantial forward error. A tiny residual is reassuring about the equations the answer satisfies, but it does not alone establish accuracy in the unknown solution.

An algorithm can be unstable even for a well-conditioned problem. Computing a small quantity by subtracting two rounded large quantities can throw away most significant digits. The computation has introduced avoidable error beyond the sensitivity already present in the mathematical question.

## Floating-point values have uneven spacing

A normal floating-point value stores a sign, a significand, and an exponent. Larger magnitudes have larger absolute gaps between adjacent representable numbers. Binary64 has 53 bits of significand precision, including the implicit leading bit. Its spacing just above one is approximately 2.22 times 10 to the power minus 16.

The common round-to-nearest arithmetic model treats one basic operation as its exact value multiplied by one plus a small relative perturbation. The unit roundoff is half the spacing above one for binary64. This model excludes some exceptional situations, including overflow and the smallest subnormal results.

**Subnormal numbers** fill the range between zero and the smallest normal magnitude with constant absolute spacing. They support gradual underflow, but their relative precision decreases near zero. Overflow can produce infinity. Invalid operations can produce NaN. NaN compares unequal to itself, so a numerical pipeline should check finiteness explicitly rather than trust ordinary comparisons.

Decimal fractions such as 0.1 generally have no finite binary expansion. Their stored approximation is not a defect in Python. Decimal arithmetic can be appropriate for decimal accounting rules, while arbitrary precision can help resolve difficult numerical problems. Neither option removes the need to analyze conditioning and truncation error.

## Algebra can rescue a small result

The expression square-root of one plus x minus one loses accuracy when x is tiny. Both the square root and one are close to one, so their subtraction exposes rounding that was harmless at the larger scale. Multiplying numerator and denominator by the conjugate gives an equivalent expression.

$$
\sqrt{1+x}-1=\frac{x}{\sqrt{1+x}+1}.
$$

The denominator on the right is close to two. The right-hand expression retains the small input directly rather than reconstructing it from a cancellation. The lab compares both forms at x equal to 10 to the power minus 12. Its disagreement measures an algorithmic effect on a synthetic input.

Summation also depends on order because floating-point addition is not associative. The toy sequence 10 to the power 16, one, and minus 10 to the power 16 loses the one under ordinary left-to-right addition. Compensated summation tracks information lost to rounding. Pairwise summation reduces the length of the accumulation chain and maps naturally to parallel reduction trees.

:::{admonition} Try it
:class: exercise
You can replace the small input with powers from 10 to the power minus 4 through 10 to the power minus 18. Predict when the subtraction becomes zero, then compare the two formulas. Explain why a zero computed difference does not imply that the exact difference is zero.
:::

## Smaller steps eventually hurt differentiation

A centered difference estimates a derivative from two nearby function values. Its truncation error is proportional to the square of the step under suitable smoothness. Rounding in the function values is divided by the step, so making the step extremely small can magnify arithmetic noise.

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h},\qquad
\text{error}\approx C_1 h^2+C_2 u/h.
$$

The constants depend on the function and the implementation. Balancing these terms suggests a step proportional to the cube root of unit roundoff after sensible scaling. This is a guide, not a universal optimal value. The lab exposes the falling and rising error regimes without requiring a timing measurement.

:::{admonition} Try it
:class: exercise
You can change the evaluation point from one to zero or ten. Compare absolute and relative error against the analytic cosine derivative. Explain why relative error becomes hard to interpret near a zero derivative.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/01_errors_floating_point.py
```

```python
"""Compare stable and unstable formulas on deterministic toy inputs."""
import math
import numpy as np

x = 1e-12
naive = math.sqrt(1.0 + x) - 1.0
stable = x / (math.sqrt(1.0 + x) + 1.0)
print(f"binary64 spacing at 1: {np.spacing(1.0):.6e}")
print(f"naive cancellation result: {naive:.16e}")
print(f"rationalized result: {stable:.16e}")
print(f"relative disagreement: {abs(naive-stable)/abs(stable):.6e}")
terms = [1e16, 1.0, -1e16]
sequential = 0.0
for term in terms:
    sequential += term
print(f"left-to-right sum: {sequential:.1f}")
assert sequential == 0.0
print(f"compensated sum: {math.fsum(terms):.1f}")
assert math.fsum(terms) == 1.0
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    derivative = (math.sin(1+h)-math.sin(1-h))/(2*h)
    print(f"h={h:.0e}, central difference error={abs(derivative-math.cos(1)):.6e}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
binary64 spacing at 1: 2.220446e-16
naive cancellation result: 5.0004445029117051e-13
rationalized result: 4.9999999999987498e-13
relative disagreement: 8.890058e-05
left-to-right sum: 0.0
compensated sum: 1.0
h=1e-02, central difference error=9.004993e-06
h=1e-04, central difference error=9.004295e-10
h=1e-06, central difference error=2.771694e-11
h=1e-08, central difference error=2.969885e-09
h=1e-10, central difference error=5.848104e-08
h=1e-12, central difference error=1.227093e-05
```

The stable formula retains the small quantity directly. The saved run shows a relative disagreement of about 8.89e-5 between the two formulas, despite their exact algebraic equivalence. The explicit left-to-right loop loses the unit term, while math.fsum preserves it. Python versions can change the implementation of built-in sum, so this experiment specifies the accumulation order explicitly.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Conditioning | The problem determines how input uncertainty is amplified. |
| Stability | The algorithm determines whether avoidable arithmetic error grows. |
| Step selection | A smaller difference step eventually magnifies rounding. |

The next lecture, {doc}`02-linear-systems`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "A backward-stable algorithm solves which problem accurately?",
    "options": [
      "It solves a nearby input problem.",
      "It always solves the original problem with tiny forward error."
    ],
    "answer": 0,
    "explain": "Backward stability controls input perturbation, while conditioning controls its effect on the answer."
  },
  {
    "q": "Why does rationalizing the square-root expression help?",
    "options": [
      "It increases the number of significant bits in the hardware.",
      "It avoids subtracting nearly equal rounded values."
    ],
    "answer": 1,
    "explain": "The revised formula carries the small input in its numerator rather than reconstructing it by subtraction."
  },
  {
    "q": "What happens to relative error when the reference approaches zero?",
    "options": [
      "It can become uninformative or undefined.",
      "It always approaches zero."
    ],
    "answer": 0,
    "explain": "The denominator in relative error becomes small, so absolute error also needs consideration."
  },
  {
    "q": "Why can an extremely small differentiation step hurt?",
    "options": [
      "A smaller step always increases truncation error.",
      "Rounding errors in function values are divided by the small step."
    ],
    "answer": 1,
    "explain": "The centered formula has decreasing truncation error but increasing arithmetic-noise amplification."
  },
  {
    "q": "What does a condition number describe?",
    "options": [
      "It describes sensitivity of the mathematical problem.",
      "It describes only the speed of a particular implementation."
    ],
    "answer": 0,
    "explain": "Conditioning is separate from the algorithm and machine runtime."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 1 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap01.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
