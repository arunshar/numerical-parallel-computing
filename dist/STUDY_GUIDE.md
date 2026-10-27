# Numerical and Parallel Computing

This document combines original study lessons for CSCI 5302 and CSCI 5451. The course website and repository contain the same lessons, runnable labs, source maps, and verification evidence.

The notes explain the public course topics and the major topics in all four associated textbook outlines. Some advanced topics are surveys. The coverage ledger records those distinctions. This document does not reproduce or replace the textbooks.

Prepared for Arun Sharma. Codex drafted this edition. Claude refinement is pending.


---

# 0. Getting ready

Prepared for Arun Sharma. Codex drafted this study edition.

You can read these lessons with basic programming, first-year calculus, and introductory linear algebra. You do not need to remember every theorem. You do need to distinguish a mathematical problem from the algorithm and machine used to solve it.

A matrix equation can have a unique exact solution while a computer returns a poor approximation. A parallel program can produce the right answer and still run more slowly than a serial program. These are different failures, and the course gives you tools to diagnose each one.

**Learning goals.**
- You will distinguish the mathematical problem, numerical algorithm, and implementation.
- You will read array shapes, derivatives, and asymptotic costs without confusing their roles.
- You will identify whether an experiment measures numerical accuracy or execution performance.
- You will select the correct runtime for a Python, C, MPI, or CUDA example.

## Setup

You should follow [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html) for the Python environment. The first numerical lessons run on a CPU. Native parallel examples have additional setup instructions in their own lessons.

## A problem, an algorithm, and an implementation answer different questions.

The **problem** specifies the desired result. For example, the equation `Ax = b` asks for a vector `x` that satisfies a linear relationship. The **algorithm** specifies a procedure, such as Gaussian elimination. The **implementation** expresses that procedure in code and chooses data types, memory layouts, and processor work assignments.

You should preserve these distinctions when diagnosing an error. If two rows of `A` are identical, the problem may not have a unique solution. If the algorithm divides by a zero pivot without exchanging rows, the procedure may fail on a solvable problem. If the code indexes outside an array, the implementation fails independently of the mathematical method.

A solver has at least two useful outputs. It produces a candidate answer, and it should provide evidence about that answer. A residual, convergence history, or error estimate is often more useful than another printed digit.

## Array shapes make algebra concrete.

A vector with three entries can represent three temperatures. A matrix with four rows and three columns can map those temperatures into four sensor readings. The product has four entries because each matrix row produces one dot product.

If `A` has shape `(m, k)` and `B` has shape `(k, n)`, the product `AB` has shape `(m, n)`. The matching inner dimension `k` counts the products summed in each output entry. A straightforward implementation performs approximately `2mkn` floating-point operations when a multiply and an addition count separately.

The order of multiplication matters. Matrix multiplication generally does not commute. The product `AB` may exist while `BA` does not. You should write the shapes beside a new formula before implementing it.

## Derivatives describe local changes.

A scalar derivative describes the first-order change in one output when one input changes. A **gradient** collects the derivatives of a scalar output with respect to several inputs. A **Jacobian** collects the derivatives of several outputs with respect to several inputs. A **Hessian** collects second derivatives of a scalar output.

For the made-up function `f(x, y) = x² + 3y²`, the gradient is `(2x, 6y)` and the Hessian is the diagonal matrix with entries `2` and `6`. The gradient tells us the local uphill direction. The Hessian tells us that the function curves more sharply in the `y` direction.

A Taylor approximation combines these objects. A first-order model uses a function value and a derivative. A second-order model adds curvature. Numerical algorithms repeatedly replace a difficult problem with a local model that is easier to solve.

**Try it.**

You should compute the function, gradient, and Hessian at `(1, 2)`. You should then predict the first-order change when `x` increases by `0.01`. Compare that prediction with the exact change in the polynomial.

## Error and cost have different units.

An **absolute error** has the same units as the answer. A **relative error** divides the error magnitude by a reference magnitude and has no units. A time measurement has seconds as its unit. A rate might have operations per second or bytes per second.

You cannot compare these quantities by their numeric values alone. An error of `0.01` might be tiny for a distance measured in kilometers and large for a quantity near zero. A computation that takes fewer arithmetic operations can still run slowly if it moves more data through memory.

Asymptotic notation describes how a cost grows as the problem size grows. It does not remove constant factors, hardware limits, or communication. A method with cost `O(n log n)` can lose to an `O(n²)` method on a sufficiently small input.

## A tiny experiment shows why execution order matters.

The following experiment uses binary64 floating-point numbers. The large magnitude makes a unit-sized change too small to preserve in one of the additions.

```python
values = [1e16, 1.0, -1e16]
left = (values[0] + values[1]) + values[2]
right = values[0] + (values[1] + values[2])
reordered = (values[0] + values[2]) + values[1]
print(left, right, reordered)
```

The usual binary64 output is `0.0 0.0 1.0`. Exact real arithmetic gives `1` for all three expressions. The computer preserves only a finite number of significant bits, so it can lose a small term before later cancellation exposes that loss.

A parallel reduction changes the addition tree. That change may alter low-order bits even when every thread follows its instructions correctly. You should choose a correctness tolerance that reflects the numerical problem and validate the reduction separately from its speed.

## C and Python expose different responsibilities.

Python arrays carry shape and type information that many library calls inspect. C pointers generally do not carry an array length. A C function that accepts a pointer needs a separate contract for dimensions, allocation, ownership, and valid indices.

CUDA adds another distinction. The host code runs on the CPU, while a kernel runs on the device. A pointer's meaning depends on its allocation and the API being used. MPI adds another boundary because each rank usually has a separate address space.

You should write down who owns each array, who writes each output, and which synchronization makes a result visible. These questions prevent many bugs before a compiler or profiler becomes relevant.

**Try it.**

You should sketch the ownership of a 4-by-4 matrix divided among four MPI ranks. Then sketch the same matrix processed by four threads in one shared array. Explain which version needs messages to obtain a neighbor's data.

## Summary

| Concept | What you should remember |
|---|---|
| The problem defines the desired answer. | Its conditioning can limit the accuracy any algorithm can achieve. |
| The algorithm defines the mathematical procedure. | Its stability and convergence determine how it handles approximation. |
| The implementation defines execution details. | Its indexing, synchronization, and data movement affect correctness and speed. |
| A validation record defines observed evidence. | Its claims should stay within the tested inputs and runtime. |

The numerical track begins by making error and floating-point arithmetic precise. The parallel track later asks which parts of an algorithm can execute together.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. A has shape (4, 3), and B has shape (3, 2). What is the shape of AB?**

1. The shape is (4, 2).
2. The shape is (3, 3).
3. The product is undefined.

**Answer.** The shape is (4, 2). Each of four rows combines with each of two columns, using three products per output entry.

**Question 2. Does a faster execution prove that an algorithm is numerically accurate?**

1. Yes, because faster algorithms have fewer errors.
2. No, because accuracy requires a separate check.

**Answer.** No, because accuracy requires a separate check. Execution time measures performance. A residual or comparison with a reference checks numerical behavior.

**Question 3. What does a Jacobian describe?**

1. It describes first derivatives of several outputs with respect to several inputs.
2. It describes only one matrix eigenvalue.

**Answer.** It describes first derivatives of several outputs with respect to several inputs. The Jacobian is the matrix of partial derivatives for a vector-valued function.

**Question 4. Can changing an addition tree change a floating-point result?**

1. Yes, because rounding occurs after intermediate operations.
2. No, because addition is always associative on a computer.

**Answer.** Yes, because rounding occurs after intermediate operations. Floating-point addition can discard small terms at different intermediate steps.

**Question 5. Does a CPU model of a CUDA kernel measure GPU throughput?**

1. Yes, if it computes the same numbers.
2. No, because it executes on a different machine model.

**Answer.** No, because it executes on a different machine model. A CPU reference can check arithmetic or indexing, but device performance requires device execution.



---

# 1. Errors and floating-point arithmetic

Prepared for Arun Sharma. Codex drafted this study edition.

A computer stores a finite description of a number. Numerical computing studies how that limitation interacts with approximations, uncertain measurements, and the algorithms we choose. A faster answer has little value when rounding destroys the digits that the application needs. This lecture builds a vocabulary for separating those problems and tests two formulas that are equal in exact arithmetic.

**Learning goals.**
- You will distinguish input error, approximation error, and rounding error.
- You will explain conditioning, stability, and forward versus backward error.
- You will diagnose cancellation and choose a more accurate formula.
- You will measure the tradeoff between step size and differentiation error.

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

**Try it.**

You can replace the small input with powers from 10 to the power minus 4 through 10 to the power minus 18. Predict when the subtraction becomes zero, then compare the two formulas. Explain why a zero computed difference does not imply that the exact difference is zero.

## Smaller steps eventually hurt differentiation

A centered difference estimates a derivative from two nearby function values. Its truncation error is proportional to the square of the step under suitable smoothness. Rounding in the function values is divided by the step, so making the step extremely small can magnify arithmetic noise.

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h},\qquad
\text{error}\approx C_1 h^2+C_2 u/h.
$$

The constants depend on the function and the implementation. Balancing these terms suggests a step proportional to the cube root of unit roundoff after sensible scaling. This is a guide, not a universal optimal value. The lab exposes the falling and rising error regimes without requiring a timing measurement.

**Try it.**

You can change the evaluation point from one to zero or ten. Compare absolute and relative error against the analytic cosine derivative. Explain why relative error becomes hard to interpret near a zero derivative.

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

The next lecture, [02-linear-systems](https://arunshar.github.io/numerical-parallel-computing/lectures/02-linear-systems.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. A backward-stable algorithm solves which problem accurately?**

1. It solves a nearby input problem.
2. It always solves the original problem with tiny forward error.

**Answer.** It solves a nearby input problem. Backward stability controls input perturbation, while conditioning controls its effect on the answer.

**Question 2. Why does rationalizing the square-root expression help?**

1. It increases the number of significant bits in the hardware.
2. It avoids subtracting nearly equal rounded values.

**Answer.** It avoids subtracting nearly equal rounded values. The revised formula carries the small input in its numerator rather than reconstructing it by subtraction.

**Question 3. What happens to relative error when the reference approaches zero?**

1. It can become uninformative or undefined.
2. It always approaches zero.

**Answer.** It can become uninformative or undefined. The denominator in relative error becomes small, so absolute error also needs consideration.

**Question 4. Why can an extremely small differentiation step hurt?**

1. A smaller step always increases truncation error.
2. Rounding errors in function values are divided by the small step.

**Answer.** Rounding errors in function values are divided by the small step. The centered formula has decreasing truncation error but increasing arithmetic-noise amplification.

**Question 5. What does a condition number describe?**

1. It describes sensitivity of the mathematical problem.
2. It describes only the speed of a particular implementation.

**Answer.** It describes sensitivity of the mathematical problem. Conditioning is separate from the algorithm and machine runtime.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 1 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap01.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 2. Linear systems and direct solvers

Prepared for Arun Sharma. Codex drafted this study edition.

[01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html) separated problem sensitivity from arithmetic error. A linear system gives those ideas a concrete form because we can measure the equation residual and perturb the right-hand side. We will solve a small system, explain elimination, and construct a nearly singular example whose answer changes sharply under tiny data changes.

**Learning goals.**
- You will interpret a matrix equation as a set of simultaneous constraints.
- You will explain elimination, pivoting, and reusable factorizations.
- You will compare residual size with solution error.
- You will choose a direct method that respects matrix structure.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## A matrix equation describes shared unknowns

The equation A x equals b collects linear equations into one object. The matrix stores coefficients, the vector x stores unknowns, and b stores observed or prescribed values. A square system has a unique solution exactly when its matrix is nonsingular. A singular system can have no solution or many solutions, depending on whether b belongs to the column space.

For the toy system with rows two x plus y equals one and x plus three y equals two, the exact solution is x equal to 0.2 and y equal to 0.6. The determinant is five, which confirms nonsingularity for this two-by-two example. Determinants are a poor numerical test of singularity for large problems because their magnitude changes dramatically under scaling.

A rectangular system asks a different question. Extra equations can conflict, which leads to least squares. Too few independent equations leave free directions, which require an additional choice such as the minimum-norm solution. You should identify that mathematical problem before selecting a library routine.

## Elimination converts a hard system into two easy systems

**Gaussian elimination** uses row operations to remove entries below each pivot. A lower triangular matrix records the multipliers, and an upper triangular matrix stores the reduced system. With row permutations, the factorization is P A equals L U.

$$
PA=LU,\qquad Ly=Pb,\qquad Ux=y.
$$

Forward substitution solves the lower triangular system from top to bottom. Back substitution solves the upper triangular system from bottom to top. Each substitution requires roughly quadratic work for a dense matrix. Dense factorization requires roughly two-thirds n cubed arithmetic operations, so repeated right-hand sides should reuse the factorization.

You do not usually form A inverse just to solve A x equals b. An inverse needs more work, uses extra storage, and adds opportunities for rounding error. Solving the triangular systems exposes the operations the application actually needs. Explicit inverse entries can still be useful when the scientific question specifically requires them.

## Pivoting controls dangerous divisions

A tiny pivot can produce huge elimination multipliers. Those multipliers can magnify rounding even when the original matrix is well conditioned. **Partial pivoting** selects the largest available magnitude in the current column and exchanges rows before elimination. Complete pivoting also searches columns, but it requires more comparisons and bookkeeping.

Partial pivoting works well for many dense problems, but it is not an unconditional guarantee of small intermediate growth. Special matrices can create a large growth factor. The practical conclusion is to use a mature solver, examine diagnostics, and avoid claiming that pivoting eliminates every numerical risk.

A scaled problem can behave better because its rows and columns have comparable magnitudes. Row scaling changes the units of equations without changing their exact solution. Column scaling changes the units of unknowns and requires transforming the answer back. Scaling should reflect the physical quantities, not merely make one reported residual look small.

## Norms turn vectors and matrices into comparable sizes

A vector norm measures size under a chosen convention. The two-norm measures Euclidean length. The infinity norm measures the largest absolute component. An induced matrix norm bounds how much a matrix can enlarge a vector in the corresponding norm.

The condition number of a nonsingular matrix is the product of the matrix norm and the inverse norm. In the two-norm, it equals the ratio of the largest singular value to the smallest singular value.

$$
\kappa(A)=\|A\|\,\|A^{-1}\|.
$$

A large value identifies a direction that the matrix compresses strongly compared with another direction. Recovering the original vector then amplifies errors along that compressed direction. The relevant sensitivity also depends on the right-hand side, so a worst-case condition number is a bound rather than a prediction for every input.

## A residual asks whether the equations hold

The **residual** is b minus A times the computed solution. A residual close to zero means the computed values nearly satisfy the stated equations. The solution error compares those values with the exact x, which is usually unavailable. A badly conditioned matrix can map a substantial solution error into a very small residual.

A scale-aware backward-error indicator divides the residual norm by the sum of matrix norm times solution norm and right-hand-side norm. This avoids treating a residual of 0.001 as equally good for quantities near one and quantities near a billion.

$$
\eta=\frac{\|b-A\hat{x}\|}{\|A\|\|\hat{x}\|+\|b\|}.
$$

The lab makes two rows almost equal. The resulting matrix has a condition number near four million. A tiny opposing change in the two right-hand-side entries causes a much larger relative change in the solution, even though the perturbed system has a tiny residual.

**Try it.**

You can replace the diagonal perturbation of 10 to the power minus 6 with 10 to the power minus 3. Predict the change in condition number and solution sensitivity before running the experiment. Keep the right-hand-side perturbation fixed so the comparison isolates the matrix.

## Structure changes the best solver

A symmetric positive definite matrix admits **Cholesky factorization**, A equals L times L transpose. Cholesky uses about half the factorization work of general dense LU and stores only one triangle. Positive definiteness means every nonzero vector has positive quadratic energy x transpose A x.

A symmetric indefinite system needs a different factorization, often an L D L transpose form with careful pivoting and small diagonal blocks. A banded matrix has nonzeros near its diagonal, so a banded solver avoids much of the work and storage of a dense solver. Sparse elimination can create **fill-in**, which introduces nonzeros absent from the original matrix. Reordering seeks to reduce that fill.

BLAS kernels organize vector, matrix-vector, and matrix-matrix operations. LAPACK assembles dense numerical algorithms from these kernels. Blocked elimination uses matrix-matrix work to improve data reuse, connecting the numerical algorithm to the parallel computing lectures. A low arithmetic count does not by itself guarantee low runtime when memory movement dominates.

## Refinement uses the residual to improve an answer

**Iterative refinement** computes a residual, solves for a correction with the existing factors, and adds that correction to the current solution. Higher precision for the residual can make the correction more informative. The method can improve a solution from lower-precision factors when conditioning and rounding satisfy suitable bounds.

Refinement cannot rescue an arbitrary singular or severely unstable factorization. You should monitor whether the residual and correction decrease. Rank-revealing methods become appropriate when the system is effectively singular at the precision or data uncertainty of the application.

**Try it.**

You can calculate the scale-aware backward-error indicator for both systems in the lab. Compare its conclusion with the forward-error measurement. Explain why the two conclusions can both be correct.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/02_linear_systems.py
```

```python
"""Measure residuals and perturbation sensitivity in two linear systems."""
import numpy as np

A = np.array([[2., 1.], [1., 3.]])
b = np.array([1., 2.])
x = np.linalg.solve(A, b)
print("solution:", np.array2string(x, precision=6))
print(f"residual norm: {np.linalg.norm(b-A@x):.6e}")
print(f"condition number: {np.linalg.cond(A):.6f}")
assert np.allclose(x, [0.2, 0.6])
B = np.array([[1., 1.], [1., 1.+1e-6]])
x_true = np.ones(2)
c = B @ x_true
delta = np.array([1e-7, -1e-7])
y = np.linalg.solve(B, c+delta)
print(f"sensitive-system condition number: {np.linalg.cond(B):.6e}")
print(f"relative data change: {np.linalg.norm(delta)/np.linalg.norm(c):.6e}")
print(f"relative solution change: {np.linalg.norm(y-x_true)/np.linalg.norm(x_true):.6e}")
print(f"perturbed residual: {np.linalg.norm(c+delta-B@y):.6e}")
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
solution: [0.2 0.6]
residual norm: 0.000000e+00
condition number: 2.618034
sensitive-system condition number: 4.000002e+06
relative data change: 4.999999e-08
relative solution change: 2.000000e-01
perturbed residual: 0.000000e+00
```

The first solve returns [0.2, 0.6] with a zero residual at the displayed precision. The sensitive system amplifies a relative right-hand-side change near 5e-8 into a relative solution change of 0.2. Its perturbed residual is still zero at the displayed precision. **A tiny residual does not guarantee a small solution error.**

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Factorization | LU separates expensive setup from cheaper repeated solves. |
| Conditioning | Nearly dependent equations can amplify tiny data changes. |
| Structure | Positive definite, banded, and sparse systems justify specialized methods. |

The next lecture, [03-least-squares-svd](https://arunshar.github.io/numerical-parallel-computing/lectures/03-least-squares-svd.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Why should repeated right-hand sides reuse a factorization?**

1. The factorization costs much more than triangular substitutions.
2. The matrix inverse is exact once computed.

**Answer.** The factorization costs much more than triangular substitutions. Dense factorization has cubic work, while each triangular solve has quadratic work.

**Question 2. Which matrix property supports Cholesky factorization?**

1. The matrix is merely square.
2. The matrix is symmetric positive definite.

**Answer.** The matrix is symmetric positive definite. Positive definiteness provides positive pivots for a standard Cholesky factorization.

**Question 3. What does the residual b-Ax measure?**

1. It measures how well the computed values satisfy the equations.
2. It directly measures distance to the unknown exact solution.

**Answer.** It measures how well the computed values satisfy the equations. A matrix can compress a large solution error into a small residual.

**Question 4. Why does partial pivoting exchange rows?**

1. It guarantees that every matrix is well conditioned.
2. It avoids unnecessarily small pivots in the current column.

**Answer.** It avoids unnecessarily small pivots in the current column. Pivoting controls algorithmic growth, while conditioning belongs to the input problem.

**Question 5. What does iterative refinement solve for?**

1. It solves for a correction driven by the residual.
2. It changes the original physical model.

**Answer.** It solves for a correction driven by the residual. The correction equation reuses the existing factors and targets remaining algebraic error.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 2 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap02.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 3. Least squares, orthogonality, and singular values

Prepared for Arun Sharma. Codex drafted this study edition.

[02-linear-systems](https://arunshar.github.io/numerical-parallel-computing/lectures/02-linear-systems.html) solved compatible square systems. Measurements usually create more equations than unknowns, and noise prevents a line or surface from passing through every observation. Least squares chooses the model whose residual has the smallest squared length. This lecture connects that choice to projections and shows why different algebraic routes have different numerical behavior.

**Learning goals.**
- You will derive the normal equations from an optimization objective.
- You will explain QR, Householder transformations, and Givens rotations.
- You will use singular values to diagnose rank and sensitivity.
- You will distinguish interpolation, discrete fitting, and continuous approximation.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can compare the coefficients from QR with coefficients from the normal equations on the nearly dependent matrix. Use a right-hand side with a small controlled perturbation. Compare the fitted vectors as well as the coefficients because parameter error and prediction error can tell different stories.

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

**Try it.**

You can add one observation with a large response to the line-fitting data. Compare ordinary least squares with a manually chosen lower weight for that observation. Explain which assumption about measurement reliability the weighted fit makes.

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

The next lecture, [04-eigenvalues](https://arunshar.github.io/numerical-parallel-computing/lectures/04-eigenvalues.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is orthogonal to the design columns at a least-squares solution?**

1. The residual is orthogonal to the columns.
2. The response vector must be orthogonal to the columns.

**Answer.** The residual is orthogonal to the columns. The normal equations state A-transpose times the residual equals zero.

**Question 2. What happens to the two-norm condition number when forming normal equations?**

1. It is always halved.
2. It is squared in exact arithmetic for full column rank.

**Answer.** It is squared in exact arithmetic for full column rank. The singular values of A-transpose A are the squared singular values of A.

**Question 3. What does a numerical-rank threshold depend on?**

1. It depends on precision, scaling, and the meaningful data uncertainty.
2. It is always exactly zero.

**Answer.** It depends on precision, scaling, and the meaningful data uncertainty. Very small singular values can represent directions that the available data cannot identify reliably.

**Question 4. Which objective controls the largest approximation error?**

1. Ordinary least squares controls the maximum error directly.
2. Minimax approximation controls the worst absolute error.

**Answer.** Minimax approximation controls the worst absolute error. Least squares controls a sum or integral of squared errors, while minimax controls the supremum.

**Question 5. What does a Householder transformation preserve?**

1. It preserves Euclidean lengths.
2. It preserves every individual coordinate.

**Answer.** It preserves Euclidean lengths. An orthogonal reflection changes coordinates while preserving their two-norm.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 3 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap03.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 4. Eigenvalues, invariant subspaces, and matrix transformations

Prepared for Arun Sharma. Codex drafted this study edition.

[03-least-squares-svd](https://arunshar.github.io/numerical-parallel-computing/lectures/03-least-squares-svd.html) described how a matrix stretches input directions. Eigenvalues ask which directions remain on their own line after multiplication. These directions explain repeated matrix application, vibration modes, stability, and many iterative algorithms. We will compute a dominant eigenpair and use its residual to assess progress.

**Learning goals.**
- You will interpret eigenpairs and invariant subspaces.
- You will explain power, inverse, Rayleigh quotient, and QR iteration.
- You will distinguish symmetric problems from general nonsymmetric problems.
- You will use residuals without confusing them with guaranteed eigenvector accuracy.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## An eigenvector keeps its direction

An **eigenpair** consists of a nonzero vector v and a scalar lambda satisfying A v equals lambda v. Multiplying by A changes the vector's scale but keeps it in the same one-dimensional subspace. A negative real eigenvalue reverses its orientation, while a complex eigenvalue can describe rotation and scaling in a real two-dimensional representation.

$$
Av=\lambda v.
$$

For the matrix with rows two, one and one, two, the vectors one, one and one, minus one are eigenvectors. Their eigenvalues are three and one. Repeated multiplication magnifies the first direction three times per application and leaves the second direction unchanged. A generic starting vector therefore becomes increasingly aligned with the first direction.

The characteristic polynomial provides a formal definition through the determinant of A minus lambda I. Computing that polynomial and then finding its roots is usually a poor numerical strategy. Its coefficients can be extremely sensitive, and it discards useful matrix structure.

## Multiplicity does not guarantee enough eigenvectors

A repeated eigenvalue has an algebraic multiplicity in the characteristic polynomial. Its geometric multiplicity is the dimension of its eigenspace. A matrix is **defective** when it lacks a full basis of eigenvectors. The two-by-two matrix with ones on the diagonal and one above the diagonal has only one independent eigenvector.

A defective or nearly defective matrix can have very sensitive eigenvectors and eigenvalues. A small matrix perturbation can change the spectrum much more than a comparable perturbation would change a symmetric matrix. The condition of the eigenvector basis matters for a diagonalizable nonsymmetric matrix.

An **invariant subspace** is a collection of directions that multiplication keeps within that collection. A cluster of nearby eigenvalues can define a stable subspace even when its individual eigenvectors are difficult to identify. Many practical algorithms target the subspace rather than insist on a particular basis inside it.

## Symmetry gives strong guarantees

A real symmetric matrix has real eigenvalues and an orthonormal eigenvector basis. Orthogonal changes of coordinates preserve Euclidean distances, which makes symmetric eigenproblems numerically attractive. A general real matrix can instead require complex eigenpairs or a real block representation.

The **Schur decomposition** writes a square matrix as Q T Q conjugate-transpose with unitary Q and upper triangular T in the complex case. The diagonal of T gives the eigenvalues. Real arithmetic can use two-by-two diagonal blocks for conjugate pairs. Schur form exists even when an eigenvector basis does not.

Gershgorin disks provide inexpensive eigenvalue location bounds. Each disk is centered at a diagonal entry, with radius equal to the sum of absolute off-diagonal entries in that row. The bounds can be loose, but they connect diagonal dominance to spectral location without a complete decomposition.

## Power iteration follows the strongest direction

**Power iteration** repeatedly multiplies a vector by A and normalizes it. If one eigenvalue has strictly largest magnitude and the starting vector has a component in its direction, that component can dominate. For a diagonalizable matrix, the asymptotic direction error depends on the magnitude ratio between the second and first eigenvalues.

The Rayleigh quotient estimates the eigenvalue associated with the current vector. For a unit vector, it is x transpose A x. The eigenpair residual is A x minus the quotient times x.

$$
\rho(x)=\frac{x^T A x}{x^T x},\qquad r=Ax-\rho(x)x.
$$

The lab's eigenvalues are three and one, so the direction error contracts at roughly one-third per iteration after the relevant assumptions hold. The quotient converges faster for this symmetric problem because its first-order variation vanishes near an eigenvector. The printed residual gives a direct check that the eigenpair equation is becoming more accurate.

**Try it.**

You can change the diagonal entries to make the two eigenvalues closer in magnitude. Predict how the iteration count changes. Then choose the starting vector exactly perpendicular to the dominant eigenvector and explain why power iteration misses that eigenpair.

## Inverse iteration targets a chosen region

**Inverse iteration** applies the inverse of A minus a shift times I to the current vector. The method solves a linear system at each step rather than forming an inverse. Eigenvalues close to the shift become dominant after inversion. A fixed shift lets the algorithm reuse one factorization.

**Rayleigh quotient iteration** updates the shift from the current quotient. It can converge very quickly near a simple eigenpair of a symmetric matrix. The shifted solve becomes nearly singular near convergence, so implementation requires care. Rapid local convergence does not imply reliable convergence from every starting vector.

Deflation removes or separates a converged component so the algorithm can find another one. Simultaneous iteration advances a block of vectors and repeatedly orthogonalizes them. A block method handles an invariant subspace more naturally than a single-vector method when several eigenvalues are clustered.

## QR iteration transforms the whole matrix

An unshifted QR step factors the current matrix into Q R and forms R Q. The new matrix is orthogonally similar to the old one, so the eigenvalues remain unchanged. Shifts accelerate convergence. Practical dense solvers first reduce a general matrix to Hessenberg form or a symmetric matrix to tridiagonal form.

That reduction makes later iterations far cheaper than repeatedly factoring an arbitrary dense matrix. A nearly zero subdiagonal entry allows the problem to split, which is also called deflation. Symmetric tridiagonal problems support specialized methods such as bisection, divide-and-conquer, and relatively robust representations. This lecture names their purpose without reproducing their full implementation machinery.

## Large problems need selected spectral information

A **Krylov subspace** contains a starting vector and its repeated matrix products. Arnoldi iteration builds an orthonormal basis for a general matrix. Lanczos iteration exploits symmetry to obtain a shorter recurrence. These methods project a large problem into a smaller one whose eigenpairs approximate selected eigenpairs of the original.

Loss of orthogonality can create duplicate or misleading approximations, so robust implementations monitor it. The matrix can be represented by a product routine rather than an explicit dense array. Generalized problems, A v equals lambda B v, add another matrix and require attention to whether B is positive definite, singular, or indefinite.

**Try it.**

You can calculate residuals for every reference eigenpair returned by the symmetric solver. Then perturb the defective matrix's lower-left entry by a tiny positive number. Compare the eigenvalue change with the perturbation size and explain why this is a sensitivity experiment rather than a timing benchmark.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/04_eigenvalues.py
```

```python
"""Compare power iteration and a symmetric eigensolver."""
import numpy as np

A = np.array([[2., 1.], [1., 2.]])
x = np.array([1., 0.])
for k in range(1, 9):
    x = A@x
    x /= np.linalg.norm(x)
    rayleigh = x@A@x
    residual = np.linalg.norm(A@x-rayleigh*x)
    print(f"iteration {k}: lambda={rayleigh:.9f}, residual={residual:.6e}")
values, vectors = np.linalg.eigh(A)
print("reference eigenvalues:", values)
assert np.allclose(values, [1., 3.])
B = np.array([[1., 1.], [0., 1.]])
print("defective-matrix eigenvalues:", np.linalg.eigvals(B))
print("rank of B-I:", np.linalg.matrix_rank(B-np.eye(2)))
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
iteration 1: lambda=2.800000000, residual=6.000000e-01
iteration 2: lambda=2.975609756, residual=2.195122e-01
iteration 3: lambda=2.997260274, residual=7.397260e-02
iteration 4: lambda=2.999695215, residual=2.468760e-02
iteration 5: lambda=2.999966130, residual=8.230313e-03
iteration 6: lambda=2.999996237, residual=2.743479e-03
iteration 7: lambda=2.999999582, residual=9.144946e-04
iteration 8: lambda=2.999999954, residual=3.048316e-04
reference eigenvalues: [1. 3.]
defective-matrix eigenvalues: [1. 1.]
rank of B-I: 1
```

The Rayleigh estimate approaches the dominant eigenvalue three, and the eigenpair residual falls from 0.6 to about 3.05e-4 across eight iterations. The defective example has two repeated eigenvalues but only a one-dimensional eigenspace because B-I has rank one. These checks separate an eigenvalue list from a complete eigenvector basis.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Power iteration | A dominant magnitude and a suitable starting component support convergence. |
| Schur form | A unitary transformation remains available when diagonalization fails. |
| Krylov methods | Selected eigeninformation can be approximated through matrix-vector products. |

The next lecture, [05-nonlinear-equations](https://arunshar.github.io/numerical-parallel-computing/lectures/05-nonlinear-equations.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Which eigenvalue does ordinary power iteration usually target?**

1. It targets the eigenvalue of largest magnitude under suitable assumptions.
2. It always targets the smallest algebraic eigenvalue.

**Answer.** It targets the eigenvalue of largest magnitude under suitable assumptions. Repeated multiplication amplifies the component with greatest eigenvalue magnitude.

**Question 2. Why can power iteration miss the dominant eigenvector?**

1. Normalization always removes the dominant component.
2. The starting vector can have zero component in that direction.

**Answer.** The starting vector can have zero component in that direction. The diagonalizable exact-arithmetic iteration cannot create a missing eigenvector component.

**Question 3. What does a small eigenpair residual check?**

1. It checks the equation A v approximately equals lambda v.
2. It proves the eigenvector is insensitive to all perturbations.

**Answer.** It checks the equation A v approximately equals lambda v. Eigenvector sensitivity also depends on separation and matrix structure.

**Question 4. Which decomposition exists even for a defective square matrix?**

1. A complete eigenvector diagonalization always exists.
2. A Schur decomposition exists.

**Answer.** A Schur decomposition exists. Schur form uses a triangular or real block-triangular representation rather than requiring independent eigenvectors.

**Question 5. Why can inverse iteration target a chosen eigenvalue?**

1. A shift near that eigenvalue makes its inverse-transformed magnitude large.
2. Inversion makes all eigenvalues equal.

**Answer.** A shift near that eigenvalue makes its inverse-transformed magnitude large. The transformed eigenvalues are reciprocals of the shifted eigenvalues.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 4 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap04.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 5. Nonlinear equations and safeguarded iteration

Prepared for Arun Sharma. Codex drafted this study edition.

[04-eigenvalues](https://arunshar.github.io/numerical-parallel-computing/lectures/04-eigenvalues.html) used iteration to find a matrix property. Nonlinear equations require the same discipline of defining a target, choosing an update, and checking whether progress is meaningful. We will solve a scalar equation with two different guarantees and then solve a coupled system using a Jacobian.

**Learning goals.**
- You will compare bracketing methods with derivative-based methods.
- You will distinguish convergence rate from practical reliability.
- You will construct a Newton step for a nonlinear system.
- You will explain safeguarding, stopping criteria, and secant updates.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can start Newton's method for x squared minus two at a very small positive number. Predict the first step before running it. Then reject any Newton proposal outside the current bracket and use a midpoint step instead. Compare the behavior with unguarded Newton iteration.

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

**Try it.**

You can start the two-variable lab near the negative solution. Then scale one equation by a factor of one thousand. Compare the mathematical roots with the residual norms and discuss why equation scaling changes stopping diagnostics without changing the exact solution set.

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

The next lecture, [06-unconstrained-optimization](https://arunshar.github.io/numerical-parallel-computing/lectures/06-unconstrained-optimization.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What assumption makes a sign-changing bracket useful?**

1. The scalar function is continuous on the interval.
2. The function may have any discontinuity.

**Answer.** The scalar function is continuous on the interval. A discontinuity can change sign without passing through zero.

**Question 2. What is the Newton correction for a system?**

1. It multiplies F by J without a solve.
2. It solves J s equals minus F.

**Answer.** It solves J s equals minus F. The linearized residual F plus J s is set to zero.

**Question 3. When is ordinary Newton convergence typically quadratic?**

1. The root is simple and the start is sufficiently close under smoothness assumptions.
2. Every starting point has a quadratic guarantee.

**Answer.** The root is simple and the start is sufficiently close under smoothness assumptions. Quadratic convergence is a local statement and requires a nonsingular derivative or Jacobian.

**Question 4. What information does the secant method reuse?**

1. It requires an exact Hessian.
2. It uses two recent function values to estimate a derivative.

**Answer.** It uses two recent function values to estimate a derivative. A divided difference replaces the explicit scalar derivative.

**Question 5. Why should stopping inspect both residual and step size?**

1. A small step can occur without a small residual.
2. The two quantities are always identical.

**Answer.** A small step can occur without a small residual. Stagnation or poor scaling can produce tiny steps while the equations remain unsatisfied.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 5 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap05.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 6. Unconstrained optimization and nonlinear least squares

Prepared for Arun Sharma. Codex drafted this study edition.

[05-nonlinear-equations](https://arunshar.github.io/numerical-parallel-computing/lectures/05-nonlinear-equations.html) searched for zeros of a residual. Optimization searches for a point with a small objective, and a zero gradient is only one part of that task. We will examine local optimality, compare step directions, and show why a step size can make an otherwise sensible method diverge.

**Learning goals.**
- You will distinguish stationary points, local minima, and global minima.
- You will explain gradient descent, Newton steps, and line searches.
- You will interpret curvature and conditioning in convergence.
- You will relate Gauss-Newton and Levenberg-Marquardt to least squares.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can change the larger Hessian entry from twenty to one hundred. Predict a safe constant step range and rerun the experiment. Compare progress in the two coordinates rather than only the final objective.

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

**Try it.**

You can apply gradient descent to the same quadratic after rescaling its second variable. Compare convergence before and after scaling, and explain how this resembles preconditioning a linear solve. Then state which optimum remains unchanged under the coordinate transformation.

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

The next lecture, [07-constrained-optimization-svm](https://arunshar.github.io/numerical-parallel-computing/lectures/07-constrained-optimization-svm.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does a zero gradient alone prove a minimum?**

1. It can also identify a saddle or maximum.
2. It always proves a global minimum.

**Answer.** It can also identify a saddle or maximum. Second-order information and convexity determine stronger conclusions.

**Question 2. What is the stable constant-step interval for the lab quadratic?**

1. Every positive step is stable.
2. The step must lie between zero and 0.1.

**Answer.** The step must lie between zero and 0.1. The largest Hessian eigenvalue is twenty, so the upper limit is two divided by twenty.

**Question 3. Why does Newton solve the toy quadratic in one step?**

1. Its quadratic local model is the exact objective.
2. Newton always solves every objective in one step.

**Answer.** Its quadratic local model is the exact objective. The Hessian and gradient exactly describe this fixed quadratic.

**Question 4. What term does Gauss-Newton omit from the least-squares Hessian?**

1. It omits J-transpose J.
2. It omits the residual-weighted second-derivative term.

**Answer.** It omits the residual-weighted second-derivative term. The remaining approximation is J-transpose J, which is useful when the omitted term is small.

**Question 5. What does a trust region control?**

1. It controls the region where the local model is used.
2. It guarantees that every model prediction is exact.

**Answer.** It controls the region where the local model is used. Actual and predicted decrease determine whether the model is trustworthy enough.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 6 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap06.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 7. Constraints, quadratic programs, and support vector machines

Prepared for Arun Sharma. Codex drafted this study edition.

[06-unconstrained-optimization](https://arunshar.github.io/numerical-parallel-computing/lectures/06-unconstrained-optimization.html) allowed every point in the domain. Real decisions often obey conservation laws, bounds, budgets, or classification requirements. A constrained optimum can have a nonzero objective gradient because every improving direction may leave the feasible set. This lecture explains that geometry through a small quadratic program and connects it to support vector machines.

**Learning goals.**
- You will identify feasible directions and active constraints.
- You will derive and check the KKT conditions for a small problem.
- You will compare projection, penalty, barrier, and SQP methods.
- You will interpret the margin and slack variables of a support vector machine.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can change the equality from x plus y equals one to x plus y equals two. Predict the new primal solution, objective, and multiplier. Check both stationarity and feasibility after solving the modified block system.

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

**Try it.**

You can multiply the toy feature values by ten and recompute a valid hard-margin weight. Predict how the weight norm and geometric margin change. Explain why feature units affect the numerical meaning of margin.

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

The next lecture, [08-interpolation-splines](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Can a constrained optimum have a nonzero objective gradient?**

1. Constraint gradients can balance it.
2. Every constrained optimum requires a zero objective gradient.

**Answer.** Constraint gradients can balance it. Only feasible directions matter, so the objective gradient can be normal to the feasible set.

**Question 2. What does complementary slackness imply for an inactive inequality?**

1. Its multiplier must be strictly positive.
2. Its multiplier is zero at a KKT point.

**Answer.** Its multiplier is zero at a KKT point. The constraint value is nonzero, so its product with the multiplier can vanish only when the multiplier is zero.

**Question 3. Is the equality-QP KKT matrix positive definite in general?**

1. It is generally indefinite.
2. A convex objective makes the entire block matrix positive definite.

**Answer.** It is generally indefinite. The constraint block changes the spectrum even when the objective Hessian is positive definite.

**Question 4. What difficulty can a very large quadratic penalty create?**

1. It eliminates all numerical error.
2. It can worsen conditioning.

**Answer.** It can worsen conditioning. Strong penalties create sharply different curvature scales.

**Question 5. What is the hard-margin distance from the decision boundary to one margin plane?**

1. It is one divided by the norm of w.
2. It is the squared norm of w.

**Answer.** It is one divided by the norm of w. The signed functional margin is fixed at one, and Euclidean distance divides by the weight norm.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 6 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap06.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 8. Interpolation, splines, and approximation choices

Prepared for Arun Sharma. Codex drafted this study edition.

[07-constrained-optimization-svm](https://arunshar.github.io/numerical-parallel-computing/lectures/07-constrained-optimization-svm.html) chose a model subject to exact constraints. Interpolation also imposes exact constraints, requiring an approximating function to match specified data values. Exact agreement at the nodes can still produce poor behavior between them. We will compare node choices and then explain why piecewise methods can be safer.

**Learning goals.**
- You will distinguish interpolation from least-squares and minimax approximation.
- You will explain monomial, Lagrange, Newton, and barycentric representations.
- You will interpret Runge behavior and the role of node placement.
- You will compare piecewise linear, Hermite, cubic-spline, and B-spline models.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can increase the number of nodes from fifteen to twenty-five. Predict which error curve improves and which can worsen. Then add tiny alternating perturbations to the data values and compare their amplification between the two node sets.

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

**Try it.**

You can compare the high-degree interpolants with piecewise linear interpolation on the same nodes. Inspect the maximum error and the behavior near the endpoints. Explain why a lower-degree local model can outperform a globally exact high-degree fit between its nodes.

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

The next lecture, [09-quadrature-differentiation](https://arunshar.github.io/numerical-parallel-computing/lectures/09-quadrature-differentiation.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. How many distinct values determine a polynomial of degree at most n minus one?**

1. There are n distinct interpolation nodes.
2. There is always only one node.

**Answer.** There are n distinct interpolation nodes. The polynomial space has n coefficients, and distinct nodes give independent value constraints.

**Question 2. What should a barycentric evaluator return at an exact node?**

1. It should perform the singular division directly.
2. It should return the stored data value.

**Answer.** It should return the stored data value. The apparent singularity is removable, and the interpolation constraint gives the value.

**Question 3. Why can equally spaced high-degree interpolation oscillate?**

1. The interpolation problem can amplify endpoint behavior and data errors.
2. The uniqueness theorem has failed.

**Answer.** The interpolation problem can amplify endpoint behavior and data errors. Uniqueness and approximation quality are different properties.

**Question 4. What does a natural cubic spline specify at the endpoints?**

1. Its endpoint values must both be zero.
2. Its endpoint second derivatives are zero.

**Answer.** Its endpoint second derivatives are zero. Natural conditions complete the spline system by setting curvature conditions.

**Question 5. What is a principal benefit of a B-spline basis?**

1. Its basis functions have local support.
2. Every coefficient changes the entire interval equally.

**Answer.** Its basis functions have local support. Local support limits interactions and supports sparse calculations.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 7 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap07.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 9. Numerical integration and differentiation

Prepared for Arun Sharma. Codex drafted this study edition.

[08-interpolation-splines](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html) built functions that match sampled values. Integrating or differentiating those approximations produces practical formulas for calculus on a computer. Integration usually smooths local noise, while differentiation magnifies it. We will derive familiar quadrature rules, compare their convergence, and explain when refinement needs more than a smaller step.

**Learning goals.**
- You will derive quadrature from polynomial approximation.
- You will compare composite, adaptive, Gaussian, and extrapolation methods.
- You will explain the conditioning difference between integration and differentiation.
- You will separate discretization error from floating-point and measurement error.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can replace exp of x with the absolute value of x minus 0.3. Compare the apparent convergence orders with the smooth example. Then split the interval exactly at 0.3 and explain why the same local rule behaves better on each smooth piece.

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

**Try it.**

You can add a fixed small perturbation to each sampled function value before differentiating. Compare the error as the step shrinks. State whether the resulting failure comes from the derivative formula, the data uncertainty, or both.

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

The next lecture, [10-ode-initial-value](https://arunshar.github.io/numerical-parallel-computing/lectures/10-ode-initial-value.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. How many polynomial degrees does an n-point Gaussian rule integrate exactly?**

1. It integrates through degree two n minus one under its weight.
2. It integrates every function exactly.

**Answer.** It integrates through degree two n minus one under its weight. The nodes and weights satisfy polynomial moment conditions, which do not cover arbitrary functions.

**Question 2. What reduction is expected for second-order error when h is halved?**

1. The error is approximately doubled.
2. The error is approximately divided by four.

**Answer.** The error is approximately divided by four. The leading term is proportional to h squared.

**Question 3. What does Richardson extrapolation require?**

1. It requires an appropriate leading error expansion and order.
2. It works equally well for every discontinuous function.

**Answer.** It requires an appropriate leading error expansion and order. The combination cancels a specific power of h.

**Question 4. Why does a relative tolerance need an absolute companion?**

1. Relative error never uses the integral magnitude.
2. The integral can be near zero.

**Answer.** The integral can be near zero. A near-zero reference makes a relative-only target unreasonable or undefined.

**Question 5. What does automatic differentiation differentiate?**

1. It differentiates the implemented operations through the chain rule.
2. It proves that the physical model is correct.

**Answer.** It differentiates the implemented operations through the chain rule. A correct program derivative still depends on whether the program expresses the intended model.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 8 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap08.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 10. Initial-value problems for ordinary differential equations

Prepared for Arun Sharma. Codex drafted this study edition.

[09-quadrature-differentiation](https://arunshar.github.io/numerical-parallel-computing/lectures/09-quadrature-differentiation.html) approximated calculus operations on a known function. An ordinary differential equation specifies how an unknown state changes, so the numerical method must build the trajectory itself. We will compare Euler and Runge-Kutta integration and distinguish a small truncation error from a stable time step.

**Learning goals.**
- You will identify the state, derivative function, and initial condition of an ODE.
- You will distinguish local truncation error from global error.
- You will analyze absolute stability with the scalar test equation.
- You will compare explicit, implicit, multistep, and adaptive methods.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can compute the ratio of errors between successive grid sizes. Compare the observed ratios with two for Euler and sixteen for RK4. Explain why the ratio may become unreliable when errors approach floating-point precision.

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

**Try it.**

You can decrease the stiff example's step to 0.01 and compare forward Euler, backward Euler, and the exact amplification. Then increase it to 0.1. State separately whether each method is stable and whether it resolves the decay accurately.

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

The next lecture, [11-boundary-value](https://arunshar.github.io/numerical-parallel-computing/lectures/11-boundary-value.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is forward Euler global order on a smooth regular problem?**

1. Its global order is one.
2. Its global order is always four.

**Answer.** Its global order is one. The local one-step error is second order, and accumulation gives first-order global error.

**Question 2. Is a stable time step necessarily accurate?**

1. Every stable step reproduces the exact trajectory.
2. Stability and accuracy are separate requirements.

**Answer.** Stability and accuracy are separate requirements. Backward Euler is stable in the lab but its amplification differs from the exact factor.

**Question 3. What does an implicit step usually require?**

1. It requires solving an equation for the next state.
2. It requires only evaluating the old derivative once.

**Answer.** It requires solving an equation for the next state. The unknown next state appears inside the derivative function.

**Question 4. What do embedded Runge-Kutta pairs estimate?**

1. They directly reveal all physical modeling error.
2. They estimate local error from two related approximations.

**Answer.** They estimate local error from two related approximations. The paired formulas assess numerical integration error under their assumptions.

**Question 5. What can make an explicit method inefficient on a stiff problem?**

1. Stability can require steps much smaller than accuracy needs.
2. Every derivative evaluation must be random.

**Answer.** Stability can require steps much smaller than accuracy needs. Fast decaying modes can limit the step even when the interesting trajectory changes slowly.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 9 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap09.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 11. Boundary-value problems and spatial discretization

Prepared for Arun Sharma. Codex drafted this study edition.

[10-ode-initial-value](https://arunshar.github.io/numerical-parallel-computing/lectures/10-ode-initial-value.html) advanced a state from one starting condition. A boundary-value problem constrains a solution at different locations, so the missing information must be determined globally. We will solve a one-dimensional Poisson problem with finite differences and compare that approach with shooting, collocation, and finite elements.

**Learning goals.**
- You will distinguish initial-value and boundary-value formulations.
- You will derive a centered finite-difference linear system.
- You will compare shooting with global discretization methods.
- You will separate boundary conditions, consistency, and stability.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can replace the zero boundary values with values from a different known smooth solution. Derive the extra right-hand-side terms before changing the code. Check whether the observed convergence rate remains second order.

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

**Try it.**

You can record both the maximum continuum error and the discrete residual on every grid. Explain why the residual can remain tiny while the continuum error decreases by a factor near four. Then estimate whether one more grid refinement is more useful than tightening the linear solver tolerance.

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

The next lecture, [12-pde](https://arunshar.github.io/numerical-parallel-computing/lectures/12-pde.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What does a Dirichlet condition prescribe?**

1. It prescribes the solution value.
2. It always prescribes the second derivative.

**Answer.** It prescribes the solution value. Dirichlet data fix values, while Neumann data fix derivatives or fluxes.

**Question 2. What unknown does simple shooting often adjust?**

1. It adjusts only the floating-point exponent.
2. It adjusts a missing initial derivative.

**Answer.** It adjusts a missing initial derivative. The initial-value solve maps that guessed derivative to an endpoint mismatch.

**Question 3. What sparsity pattern does the centered one-dimensional second derivative produce?**

1. It produces a tridiagonal coupling.
2. It necessarily produces a fully dense matrix.

**Answer.** It produces a tridiagonal coupling. Each equation uses a node and its immediate neighbors.

**Question 4. Why can a pure Neumann problem be singular?**

1. Every Neumann condition is inconsistent.
2. An additive constant can remain undetermined.

**Answer.** An additive constant can remain undetermined. Derivatives do not fix a constant offset, and compatibility also needs checking.

**Question 5. What does a Galerkin condition enforce?**

1. It makes the residual orthogonal to a chosen test space.
2. It makes the residual zero at every physical point.

**Answer.** It makes the residual orthogonal to a chosen test space. The condition is integral or inner-product based rather than pointwise exactness.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 10 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap10.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 12. Partial differential equations and sparse spatial operators

Prepared for Arun Sharma. Codex drafted this study edition.

[11-boundary-value](https://arunshar.github.io/numerical-parallel-computing/lectures/11-boundary-value.html) discretized a differential equation along one spatial axis. Partial differential equations describe variation across several independent coordinates, often space and time together. Their numerical methods combine calculus, linear algebra, stability analysis, and parallel data movement. We will study a heat-equation stencil and then survey the major spatial discretization choices.

**Learning goals.**
- You will distinguish elliptic, parabolic, and hyperbolic model problems.
- You will derive and test a heat-equation finite-difference update.
- You will explain the role of boundary conditions and stability limits.
- You will compare finite differences, finite volumes, finite elements, and spectral methods.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## The PDE type predicts important behavior

An **elliptic equation** often describes an equilibrium field, such as a steady potential governed by Poisson's equation. Its solution depends globally on boundary information. A **parabolic equation** describes diffusion, such as heat spreading over time. A **hyperbolic equation** describes propagation with finite characteristic speeds, such as a wave or transport equation.

These labels summarize useful model classes, but real systems can mix behaviors or change type. Boundary and initial conditions must match the equation's structure. Prescribing too many conditions can make a problem inconsistent, while too few can leave it undetermined.

The one-dimensional heat equation sets the time derivative equal to diffusivity times the second spatial derivative. Diffusivity has units of length squared per time. This dimensional fact already suggests that a grid spacing and time step will interact through a ratio containing h squared.

$$
u_t=\alpha u_{xx}.
$$

## A stencil turns a differential operator into local arithmetic

Combining forward Euler in time with a centered second difference in space gives a three-point update. The nondimensional ratio r equals diffusivity times the time step divided by the square of the spatial spacing.

$$
u_i^{n+1}=u_i^n+r(u_{i-1}^n-2u_i^n+u_{i+1}^n),\qquad r=\alpha\Delta t/\Delta x^2.
$$

The update reads neighboring values from the old time level. Updating the array in place from left to right would mix old and new states and implement a different method. Parallel implementations also need a clear separation between read and write states or a mathematically justified synchronization scheme.

The lab uses periodic boundaries implemented by array shifts. That choice models a ring, not a rod with fixed endpoint temperatures. The sum of the periodic second differences is zero, so the update preserves the discrete total mass up to rounding.

## Stability comes from amplification of Fourier modes

A Fourier mode is an eigenvector of the periodic difference operator. Substitution into the update gives amplification one minus four r times the square of sine of half the grid phase. The highest-frequency mode has amplification one minus four r.

$$
G(\theta)=1-4r\sin^2(\theta/2).
$$

Keeping the magnitude of every amplification factor at most one requires r between zero and one-half for this one-dimensional explicit scheme. The lab compares r equal to 0.4 with r equal to 0.6. A tiny alternating perturbation represents the highest-frequency component and reveals the unstable case after repeated steps.

The bound depends on the discretization and dimension. A two-dimensional update has contributions from both coordinate directions and therefore a different combined restriction. A **CFL condition** expresses a relationship between numerical dependence and physical propagation, but the exact bound must be derived for the actual method.

**Try it.**

You can remove the tiny alternating perturbation from the initial condition. Explain why an unstable method can then look acceptable for a while on a smooth input. Restore the perturbation and compare the difference after one hundred steps.

## Implicit time steps produce sparse linear systems

Backward Euler for diffusion moves the spatial operator to the new time level. Each time step solves a linear system involving identity minus the time step times the spatial operator. The matrix is sparse because the stencil connects only neighboring nodes.

An implicit scheme can remove a severe explicit stability limit, but a large time step still reduces temporal accuracy. Crank-Nicolson averages old and new spatial derivatives and has second-order temporal accuracy for smooth solutions. Its weaker damping of stiff modes can cause oscillatory artifacts when the initial data or forcing are not sufficiently smooth.

The **method of lines** discretizes space first and treats the resulting system as an ODE. This lets an adaptive ODE solver control time integration. The spatial operator can make the ODE stiff as the mesh is refined, so the time solver and linear algebra must be selected together.

## Conservative methods track fluxes through boundaries

A **finite-volume method** integrates the PDE over each cell and updates the cell average from fluxes through its faces. Neighboring cells share a face flux with opposite signs, which supports exact discrete conservation. The method is particularly useful for transport and conservation laws.

An upwind method uses information from the direction characteristics travel. It can stabilize transport but introduces numerical diffusion. Higher-order reconstructions reduce diffusion, while limiters reduce nonphysical oscillations near sharp changes. A method that preserves a smooth wave well can still behave poorly at a discontinuity.

Finite differences approximate derivatives directly at grid points. Finite volumes approximate balance over cells. Finite elements approximate a weak formulation using local basis functions. These approaches can produce similar matrices in simple settings, but their interpretations and extensions differ.

## Geometry and smoothness guide the spatial representation

**Finite elements** handle irregular geometry through a mesh of elements and local basis functions. Mesh quality, boundary approximation, quadrature, and function-space choice all affect accuracy. Adaptive refinement can concentrate effort near layers, corners, or large estimated error.

**Spectral methods** use global basis functions such as Fourier modes or Chebyshev polynomials. Smooth solutions can converge rapidly as the basis size grows. Discontinuities can cause Gibbs oscillations, and nonlinear products can introduce aliasing unless the calculation uses an appropriate treatment.

**Domain decomposition** partitions a large spatial problem into subdomains. Each subdomain solves local work and exchanges boundary or interface information. The communication pattern depends on the stencil and decomposition. Strong numerical coupling across interfaces can limit convergence even when the workload is well balanced.

## Sparse solvers become part of the PDE method

A sparse direct solver can provide a reliable answer but may create fill-in that consumes much more memory than the original stencil. Iterative methods use matrix-vector products and preconditioners. Conjugate gradients suit symmetric positive definite systems, while nonsymmetric transport operators need other methods.

**Multigrid** uses a hierarchy of grids. A smoother reduces some error components on a fine grid, and a coarse-grid correction addresses components that appear smooth there. Restriction transfers residual information to a coarse grid, and prolongation transfers a correction back. The idea can be geometric or algebraic.

Verification should separate spatial error, temporal error, algebraic solver error, and modeling error. A manufactured solution helps isolate discretization accuracy. Conservation checks and symmetry checks reveal different classes of bugs. A plausible-looking image is weak evidence because unstable or inconsistent schemes can still create smooth pictures.

**Try it.**

You can verify the periodic mass before and after every step for the stable case. Then change one boundary update deliberately and observe how the conservation check responds. State which property that check establishes and which accuracy questions it leaves unanswered.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/12_pde.py
```

```python
"""Test the heat-equation stencil and its amplification factors."""
import numpy as np

n = 40
x = np.arange(n)/n
u0 = np.sin(2*np.pi*x)
amplitudes = {}
for r in [0.4, 0.6]:
    u = u0.copy()
    # A tiny high-frequency component makes instability visible.
    u += 1e-8*(-1.)**np.arange(n)
    for _ in range(100):
        u = u+r*(np.roll(u, -1)-2*u+np.roll(u, 1))
    amplitudes[r] = np.max(np.abs(u))
    print(f"r={r:.1f}: maximum absolute value after 100 steps={np.max(np.abs(u)):.6e}")
    print(f"r={r:.1f}: highest-frequency amplification={1-4*r:.6f}")
assert amplitudes[0.4] < np.max(np.abs(u0))
assert amplitudes[0.6] > 1e4
r = 0.4
u = u0+r*(np.roll(u0, -1)-2*u0+np.roll(u0, 1))
print(f"periodic mass change: {abs(u.sum()-u0.sum()):.6e}")
assert abs(u.sum()-u0.sum())<1e-12
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
r=0.4: maximum absolute value after 100 steps=3.716453e-01
r=0.4: highest-frequency amplification=-0.600000
r=0.6: maximum absolute value after 100 steps=4.100186e+06
r=0.6: highest-frequency amplification=-1.400000
periodic mass change: 5.551115e-17
```

The stable ratio 0.4 yields a decaying solution with maximum magnitude about 0.372 after one hundred steps. The ratio 0.6 amplifies a tiny alternating perturbation to a magnitude above four million. The periodic mass change is near rounding precision, which checks a separate conservation property.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| PDE structure | Equilibrium, diffusion, and propagation need different numerical treatments. |
| Stencil stability | The heat update requires a suitable time-to-space ratio. |
| Solver composition | Spatial, temporal, algebraic, and modeling errors need separate checks. |

The next lecture, [13-fourier-fft](https://arunshar.github.io/numerical-parallel-computing/lectures/13-fourier-fft.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the one-dimensional explicit heat stability limit in this lecture?**

1. The ratio r must be at most one-half and nonnegative.
2. Any positive r is stable.

**Answer.** The ratio r must be at most one-half and nonnegative. The highest-frequency amplification is one minus four r.

**Question 2. Why does the periodic stencil preserve total mass?**

1. Every individual point remains unchanged.
2. Its second-difference contributions sum to zero.

**Answer.** Its second-difference contributions sum to zero. Neighbor flux contributions cancel in the periodic sum.

**Question 3. What does finite volume emphasize?**

1. It emphasizes balance of fluxes across cell faces.
2. It ignores conservation laws.

**Answer.** It emphasizes balance of fluxes across cell faces. Shared face fluxes cancel between adjacent cells.

**Question 4. Why can an unstable method initially look acceptable?**

1. Stability depends only on the plotted color scale.
2. The initial data may contain little of the unstable mode.

**Answer.** The initial data may contain little of the unstable mode. Small high-frequency errors can take time to grow visibly.

**Question 5. What roles do smoothing and coarse correction play in multigrid?**

1. They target complementary error components.
2. They both only repeat the same fine-grid operation.

**Answer.** They target complementary error components. The coarse grid addresses error that a fine-grid smoother reduces slowly.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 11 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap11.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 13. Fourier transforms, fast algorithms, and convolution

Prepared for Arun Sharma. Codex drafted this study edition.

[12-pde](https://arunshar.github.io/numerical-parallel-computing/lectures/12-pde.html) used Fourier modes to explain the stability of a periodic stencil. The Fourier transform makes those modes into a practical coordinate system for signals and operators. We will compare a direct discrete transform with an FFT, demonstrate aliasing, and compute a linear convolution without accidentally wrapping its ends together.

**Learning goals.**
- You will explain the discrete Fourier transform as a change of basis.
- You will distinguish the DFT from the FFT algorithm that computes it.
- You will interpret frequency bins, aliasing, and normalization conventions.
- You will use zero-padding to compute linear convolution with an FFT.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## The DFT measures agreement with periodic patterns

A **discrete Fourier transform** expresses a vector as a combination of complex sinusoidal basis vectors. Each coefficient measures how strongly the samples align with one periodic pattern. The common forward convention uses a negative sign in the exponential, while the inverse uses a positive sign and divides by the number of samples.

$$
X_k=\sum_{j=0}^{N-1}x_j e^{-2\pi i jk/N},\qquad
x_j=\frac1N\sum_{k=0}^{N-1}X_k e^{2\pi i jk/N}.
$$

A different normalization convention can split the scaling between forward and inverse transforms. Both conventions are valid when used consistently. A transform library's definition determines how you interpret coefficient magnitudes, energy identities, and convolution scaling.

The complex exponential combines cosine and sine. Real input creates conjugate symmetry, so negative-frequency information is redundant with positive-frequency information. A real-input FFT can exploit this symmetry to reduce storage and work. The zero-frequency coefficient is the sum of the samples under the displayed convention.

## Frequency indices become physical frequencies through sampling

If the sampling interval is delta t, the total record duration is N times delta t. The spacing between DFT frequency bins is one divided by that duration. The bin index alone is not a frequency in hertz until the sampling interval is specified.

The DFT treats the finite record as one period of a periodic sequence. A mismatch between the first and last parts of a record can create an apparent discontinuity at the periodic join. **Spectral leakage** distributes energy across bins when a sinusoid does not align with a bin or when the record boundary is discontinuous.

A window tapers the record to reduce some leakage, but it changes amplitude scaling and frequency resolution. Zero-padding samples the discrete-time Fourier transform on a denser frequency grid. It can make a spectrum easier to interpolate visually, but it does not create new measurement information or improve the fundamental resolution set by the record duration.

## Aliasing makes different continuous signals look identical

Sampling cannot distinguish frequencies separated by integer multiples of the sampling frequency. Real cosines also identify positive and negative frequency counterparts. The lab samples sixteen points and compares cosine bins three and thirteen. They produce the same real samples because thirteen is equivalent to minus three modulo sixteen.

The **Nyquist limit** describes the highest recoverable frequency for a band-limited signal under ideal uniform sampling. An antialiasing filter removes higher-frequency content before sampling. Filtering after aliasing cannot generally recover which original high frequency produced the observed samples.

This distinction matters in spatial grids as well as time series. A fine oscillation in a simulated field can appear as a different coarse oscillation after downsampling. A numerical method that moves information between grids should therefore consider frequency content, not merely array shape.

**Try it.**

You can sample cosine bins two and fourteen on the same sixteen-point grid. Then sample a frequency of 2.5 cycles per record and inspect which bins receive energy. Explain why the second experiment concerns leakage while the first demonstrates aliasing.

## An FFT reuses structure in the DFT sum

A direct DFT evaluates N sums with N terms each, giving quadratic arithmetic work. A **fast Fourier transform** reorganizes the same calculation to reuse repeated subproblems. A radix-two Cooley-Tukey decomposition splits the input into even and odd indices.

The two smaller transforms combine through twiddle factors, which are roots of unity. Each recursion level performs work proportional to N, and there are logarithmically many levels for powers of two. The resulting work is proportional to N log N.

$$
T(N)=2T(N/2)+O(N)=O(N\log N).
$$

The FFT is an algorithm family, while the DFT is the mathematical transform. Mixed-radix algorithms handle lengths with other small factors. Prime-size methods use other identities, including reductions to convolution. Practical performance depends on transform length, memory access, planning, vectorization, and communication.

The lab constructs the dense DFT matrix and compares its product with NumPy's FFT. Their answers agree to rounding error. That check establishes the chosen sign and normalization conventions for a small example. It is not a performance benchmark, and the dense matrix approach should not be used for large production transforms.

## Multiplication in frequency corresponds to circular convolution

The DFT convolution theorem relates pointwise multiplication of transforms to **circular convolution** of sequences. Circular convolution wraps indices modulo the transform length. This is appropriate for periodic signals, but ordinary finite sequences usually require linear convolution.

Two sequences of lengths m and n have a linear convolution of length m plus n minus one. Padding both sequences to at least that length prevents the nonzero result from wrapping onto itself. The lab multiplies the transforms of the sequences one, two, three and four, five, then inverts the product.

The result is four, thirteen, twenty-two, fifteen. A shorter transform would fold the tail into earlier entries. Numerical roundoff can leave tiny imaginary parts even when the mathematical result is real, so the lab takes the real part only after using a real-input problem with a known identity.

FFT convolution supports polynomial multiplication, signal filtering, and some integral operators. For very short filters, direct convolution can be faster because FFT setup and extra memory movement outweigh the asymptotic saving. Overlap-add and overlap-save methods process long streams in blocks while handling the convolution boundaries carefully.

## Fourier methods connect calculus and numerical models

Differentiating a complex Fourier mode multiplies it by its frequency times the imaginary unit. A spectral derivative therefore transforms a periodic smooth function, multiplies coefficients by frequency factors, and transforms back. High frequencies receive larger multipliers, so differentiation still amplifies noise.

Fast Poisson solvers can diagonalize certain regular-grid operators with Fourier, sine, or cosine transforms depending on the boundary conditions. The transform choice must match those conditions. Using a periodic FFT for a nonperiodic boundary problem solves a different mathematical problem.

**Wavelets** provide another basis that localizes information in both scale and position. A wavelet transform can represent abrupt localized changes more compactly than a global Fourier expansion. The simplest Haar transform combines neighboring averages and differences. Wavelet methods require their own boundary, scaling, and reconstruction conventions.

**Try it.**

You can repeat the convolution with a transform length of three and compare it with the length-four result. Identify exactly which output terms wrap together. Then increase the padding beyond four and confirm that the meaningful linear-convolution entries stay unchanged up to rounding.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/13_fourier_fft.py
```

```python
"""Check a DFT convention, aliasing, and convolution padding."""
import numpy as np

n = 16
j = np.arange(n)
signal = np.cos(2*np.pi*3*j/n)
F = np.exp(-2j*np.pi*np.outer(j,j)/n)
direct = F@signal
fast = np.fft.fft(signal)
print(f"DFT versus FFT difference: {np.max(np.abs(direct-fast)):.6e}")
print("nonzero frequency bins:", np.flatnonzero(np.abs(fast)>1e-10))
print(f"inverse error: {np.max(np.abs(np.fft.ifft(fast)-signal)):.6e}")
aliased = np.cos(2*np.pi*13*j/n)
print(f"sample difference for cosine bins 3 and 13: {np.max(np.abs(signal-aliased)):.6e}")
a = np.array([1.,2.,3.]);b = np.array([4.,5.])
length = len(a)+len(b)-1
conv = np.fft.ifft(np.fft.fft(a,length)*np.fft.fft(b,length)).real
print("padded FFT convolution:", np.round(conv, 6))
assert np.allclose(conv, np.convolve(a,b))
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
DFT versus FFT difference: 1.317398e-14
nonzero frequency bins: [ 3 13]
inverse error: 1.359740e-16
sample difference for cosine bins 3 and 13: 7.660539e-15
padded FFT convolution: [ 4. 13. 22. 15.]
```

The dense DFT and FFT differ by about 1.32e-14 on the toy vector. The nonzero bins are three and thirteen, and the inverse reconstructs the input near floating-point precision. Padding to four samples gives the linear convolution [4,13,22,15].

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Transform | The DFT specifies a coordinate change and normalization. |
| Algorithm | An FFT reduces the work needed to compute that transform. |
| Sampling | Aliasing and leakage arise from different sampling and record effects. |

The next lecture, [14-random-simulation](https://arunshar.github.io/numerical-parallel-computing/lectures/14-random-simulation.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the difference between DFT and FFT?**

1. The DFT is a transform, and an FFT is a fast algorithm for it.
2. They necessarily compute different mathematical quantities.

**Answer.** The DFT is a transform, and an FFT is a fast algorithm for it. The fast algorithm reorganizes the same sums.

**Question 2. What does zero-padding do to spectral information?**

1. It restores frequencies already lost to aliasing.
2. It samples the spectrum more densely without creating new measurements.

**Answer.** It samples the spectrum more densely without creating new measurements. Padding changes interpolation in frequency, not the original sampling information.

**Question 3. What convolution does a length-N DFT product naturally produce?**

1. It produces circular convolution modulo N.
2. It always produces unwrapped linear convolution.

**Answer.** It produces circular convolution modulo N. The transform treats the sequence as periodic.

**Question 4. What padding length suffices for sequences of lengths m and n?**

1. A length of one always suffices.
2. A length of at least m plus n minus one suffices.

**Answer.** A length of at least m plus n minus one suffices. The full linear convolution has m plus n minus one nonzero output positions at most.

**Question 5. Why do cosine bins three and thirteen match on sixteen samples?**

1. Thirteen is equivalent to negative three modulo sixteen.
2. The cosine function has no frequency dependence.

**Answer.** Thirteen is equivalent to negative three modulo sixteen. Real cosine identifies the positive and negative frequency pair under this sampling.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 12 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap12.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 14. Random numbers, simulation, and uncertainty

Prepared for Arun Sharma. Codex drafted this study edition.

[13-fourier-fft](https://arunshar.github.io/numerical-parallel-computing/lectures/13-fourier-fft.html) exploited deterministic algebraic structure. Some numerical problems are easier to solve by sampling, especially when the dimension is high or the quantity itself is an expectation. Random simulation replaces a deterministic approximation error with a statistical uncertainty that must be measured and reported. We will estimate an integral and compare random samples with a simple low-discrepancy sequence.

**Learning goals.**
- You will distinguish pseudorandom generation from independent ideal randomness.
- You will estimate Monte Carlo uncertainty and interpret a standard error.
- You will transform uniform samples into other distributions.
- You will compare variance reduction with quasi-Monte Carlo sampling.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## A generator produces a reproducible sequence

A **pseudorandom number generator** evolves a finite internal state through a deterministic rule. A seed initializes that state. The output is reproducible under the same generator and relevant implementation conditions, which is valuable for debugging and comparing algorithms.

Reproducibility does not make the sequence truly independent or suitable for every use. A finite-state generator eventually repeats. Period length alone does not measure quality because correlations or lattice structure can harm a simulation long before the period is exhausted. Scientific simulation and cryptographic security also require different properties.

A linear congruential generator updates an integer state by multiplying, adding, and taking a modulus. Its simplicity makes the mechanism easy to study, but poor parameters can create severe patterns. Lagged Fibonacci generators combine older sequence values. Modern libraries provide better-designed generators and APIs for managing independent streams.

Parallel simulation should not assign seeds casually and assume independence. A supported stream-splitting or counter-based scheme can make the relationship between workers explicit. The reproducibility record should include the generator family, seed, library version, and sampling design.

## Monte Carlo turns an integral into an expectation

If U is uniform on zero to one, the expectation of f of U equals the integral of f over that interval. The sample mean of independent evaluations estimates that expectation. For the toy function U squared, the exact integral is one-third.

$$
\hat{I}_N=\frac1N\sum_{i=1}^{N}f(U_i).
$$

The estimator is unbiased when the sampling assumptions hold and the expectation exists. Its variance is the variance of one sample divided by N. Its standard error therefore decreases proportionally to one divided by square-root of N.

$$
\operatorname{SE}(\hat{I}_N)=\frac{\sigma_f}{\sqrt{N}}.
$$

Reducing the standard error by a factor of ten requires about one hundred times as many independent samples. This slow convergence is a real cost. Its rate does not directly worsen with dimension in the same way as a full tensor-product grid, which makes Monte Carlo attractive for high-dimensional integrals.

## A standard error is an uncertainty estimate

The sample standard deviation divided by square-root of N estimates the standard error when observations are independent and the variance is finite. A normal approximation can support an approximate confidence interval for sufficiently regular large samples. The interval is a statement about the sampling procedure, not a guarantee that one realized estimate lies near the truth.

The lab prints the absolute error against the known one-third reference and the estimated standard error. These are different quantities. The actual error is available only because the toy integral is known, while the standard error is what a real simulation would estimate from its samples.

An individual run need not improve monotonically as sample count grows. A lucky small run can beat an unlucky larger run. Comparing methods requires multiple replications or an appropriate paired design when the claim concerns typical statistical performance. The fixed seed in this tutorial makes the demonstration reproducible rather than proving a universal ranking.

## Transformations create useful distributions

The **inverse transform method** applies an inverse cumulative distribution function to a uniform sample. For an exponential distribution with rate lambda, the transform is minus the logarithm of one minus U divided by lambda.

$$
X=-\frac{\log(1-U)}{\lambda}.
$$

The lab uses `log1p` to evaluate the logarithm accurately when its argument is near one. The expected exponential value is one divided by the rate, so a rate of two gives mean one-half. The simulated mean fluctuates around that value.

Rejection sampling draws from an easier proposal distribution and accepts points according to a bounded density ratio. A poor proposal can reject most samples. Box-Muller transforms two uniform samples into normal samples through polar geometry. Correlated normal vectors can be formed by applying a covariance factor, provided the covariance is positive semidefinite and factored appropriately.

**Try it.**

You can change the exponential rate from two to four. Predict the mean before rerunning the sample. Then compare the median with the mean and explain why a skewed distribution need not place half its mass on each side of its mean.

## Variance reduction spends samples more carefully

**Stratified sampling** divides the domain into regions and samples within each region. It reduces the chance that a finite sample misses a whole important part of the domain. **Antithetic sampling** pairs related samples whose errors tend to cancel, such as U and one minus U for some monotone integrands.

A **control variate** subtracts a correlated quantity with known expectation and adds that expectation back exactly. The method can reduce variance without changing the desired mean when applied correctly. The control coefficient and its estimation affect the practical benefit.

**Importance sampling** draws more often from important regions and corrects for the changed distribution through weights. The proposal must cover the target contribution. Extremely variable weights can produce a poor estimator even when every formula is formally correct. Effective sample size and weight diagnostics help expose this failure.

Markov chain Monte Carlo produces correlated samples from a target distribution through a transition rule. The ordinary independent-sample standard-error formula is then inappropriate. Autocorrelation, convergence diagnostics, and effective sample size become part of the numerical analysis.

## Low-discrepancy points replace randomness with coverage

A **quasi-Monte Carlo sequence** tries to cover the domain evenly instead of mimicking independent randomness. The lab uses the base-two radical inverse, also called a one-dimensional van der Corput sequence. Reversing the binary digits of an integer places points at fractions that progressively fill gaps.

For some smooth low-dimensional integrands, low-discrepancy sampling can converge faster than ordinary random sampling. Its deterministic error does not have the same standard-error interpretation as independent Monte Carlo. Randomized quasi-Monte Carlo can combine useful coverage with repeated randomizations for uncertainty estimation.

The benefit depends on dimension, effective dimension, smoothness, coordinate ordering, and the specific integrand. The lab's simple one-dimensional polynomial is favorable and should not be treated as evidence that one sequence wins for every problem. Discontinuities or poorly aligned high-dimensional structure can reduce the advantage.

**Try it.**

You can replace the squared function with an indicator that is one below 0.37 and zero elsewhere. Its exact integral is 0.37. Compare random and low-discrepancy estimates, and explain which smoothness assumption changed.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/14_random_simulation.py
```

```python
"""Estimate an integral and contrast random with low-discrepancy points."""
import numpy as np

def radical_inverse(k, base=2):
    value, factor = 0., 1/base
    while k:
        value += factor*(k % base)
        k //= base
        factor /= base
    return value

rng = np.random.default_rng(2026)
for n in [64,256,1024,4096]:
    u = rng.random(n)
    values = u*u
    estimate = values.mean()
    se = values.std(ddof=1)/np.sqrt(n)
    quasi = np.array([radical_inverse(k) for k in range(1,n+1)])
    print(f"n={n:4d}: random error={abs(estimate-1/3):.6e}, estimated SE={se:.6e}, quasi error={abs(np.mean(quasi**2)-1/3):.6e}")
u = rng.random(10000)
exponential = -np.log1p(-u)/2
print(f"exponential sample mean: {exponential.mean():.6f}; exact mean: 0.500000")
assert np.all(exponential>=0)
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
n=  64: random error=3.252338e-02, estimated SE=3.114520e-02, quasi error=7.770856e-03
n= 256: random error=4.356464e-03, estimated SE=1.883435e-02, quasi error=1.950567e-03
n=1024: random error=2.493155e-02, estimated SE=9.457994e-03, quasi error=4.881221e-04
n=4096: random error=1.885137e-03, estimated SE=4.649878e-03, quasi error=1.220604e-04
exponential sample mean: 0.503652; exact mean: 0.500000
```

The fixed-seed Monte Carlo errors are not monotonic as sample count grows, even though the estimated standard error generally decreases. The low-discrepancy sequence performs well on this smooth one-dimensional example. The exponential sample mean is about 0.504, close to its exact mean 0.5, without being required to equal it exactly.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Random generation | A seed controls reproducibility but does not prove statistical independence. |
| Uncertainty | Monte Carlo standard error typically decreases as the inverse square root of samples. |
| Coverage | Low-discrepancy designs target even coverage with a different error interpretation. |

The next lecture, [15-iterative-solvers](https://arunshar.github.io/numerical-parallel-computing/lectures/15-iterative-solvers.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. How many samples are typically needed to reduce Monte Carlo standard error tenfold?**

1. About one hundred times as many independent samples are needed.
2. Only twice as many samples are needed.

**Answer.** About one hundred times as many independent samples are needed. Standard error scales as one divided by square-root of N.

**Question 2. Does a fixed seed establish that a generator is statistically adequate?**

1. It proves independence of every generated value.
2. It establishes reproducibility, not adequacy by itself.

**Answer.** It establishes reproducibility, not adequacy by itself. Generator quality and the sampling design require additional analysis.

**Question 3. Why can actual error increase in one larger random run?**

1. Sampling fluctuations need not improve monotonically.
2. The law of large numbers forbids every finite fluctuation.

**Answer.** Sampling fluctuations need not improve monotonically. Asymptotic convergence does not impose monotonic error on individual realizations.

**Question 4. What is the mean of the exponential distribution with rate two?**

1. Its mean is two.
2. Its mean is one-half.

**Answer.** Its mean is one-half. The exponential mean is the reciprocal of its rate.

**Question 5. Can the independent Monte Carlo standard-error formula be used unchanged for a deterministic low-discrepancy sequence?**

1. It needs a different error or randomization analysis.
2. It always applies without modification.

**Answer.** It needs a different error or randomization analysis. The sequence does not provide independent identically distributed random samples.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 13 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap13.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 15. Iterative linear solvers and preconditioning

Prepared for Arun Sharma. Codex drafted this study edition.

[14-random-simulation](https://arunshar.github.io/numerical-parallel-computing/lectures/14-random-simulation.html) controlled error through sample count. Iterative linear solvers control algebraic error through repeated corrections, which can be much cheaper than a direct factorization for a large sparse matrix. The main challenge is to make each correction useful. We will compare Jacobi iteration with conjugate gradients and connect convergence to matrix structure.

**Learning goals.**
- You will derive stationary iterations from a matrix splitting.
- You will explain the Krylov subspace behind conjugate gradients and GMRES.
- You will identify when a preconditioner preserves the assumptions of a solver.
- You will choose stopping criteria that account for scale and downstream accuracy.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

## A matrix-vector product can be enough

A sparse matrix stores only its nonzero entries, while a **matrix-free operator** computes A times x without storing the matrix explicitly. Iterative methods can use either representation. This is valuable for PDE stencils, large graphs, and operators whose entries would be expensive to materialize.

A direct factorization often gives strong reliability and makes repeated right-hand sides inexpensive after setup. An iterative method can use less memory and avoid fill-in, but its iteration count depends on conditioning and the preconditioner. The best choice depends on matrix structure, accuracy needs, reuse, and available hardware.

The lab uses a tridiagonal matrix with twos on the diagonal and minus ones next to it. This is the one-dimensional discrete negative second derivative up to a scale factor. It is symmetric positive definite with zero Dirichlet boundaries, making it suitable for conjugate gradients.

## Stationary iteration applies the same correction rule repeatedly

A matrix splitting writes A as M minus N, where solving with M is easy. The iteration solves M times the new x equal to N times the old x plus b. Its error propagation matrix is M inverse times N.

$$
x_{k+1}=M^{-1}Nx_k+M^{-1}b.
$$

The iteration converges for every starting vector exactly when the spectral radius of the error propagation matrix is below one in the finite-dimensional linear setting. The spectral radius is the largest eigenvalue magnitude. A small residual in one iteration does not establish this property for every right-hand side.

**Jacobi iteration** chooses M as the diagonal of A. Each new coordinate uses only old coordinates, so the update is naturally parallel. **Gauss-Seidel iteration** uses the lower triangular part including the diagonal, so later coordinates use newly updated earlier coordinates. Its sequential dependence complicates parallel execution but can improve convergence per iteration.

Successive over-relaxation blends a Gauss-Seidel correction with the previous state through a relaxation parameter. A suitable parameter can accelerate convergence, while a poor choice can destabilize it. The best parameter depends on the operator and ordering, so a single remembered value is not universally valid.

## Slowly varying error survives simple smoothing

On many elliptic discretizations, Jacobi or Gauss-Seidel reduces rapidly oscillating error more effectively than slowly varying error. Refining the grid makes some slow components even harder to remove, so the iteration count can grow substantially.

This behavior explains their role as **smoothers** in multigrid. A coarse grid represents a slowly varying fine-grid error with fewer unknowns, allowing a cheap global correction. The smoother and coarse correction solve complementary parts of the error problem.

The lab reports the Jacobi residual after one hundred iterations. That result shows that a cheap step can require many repetitions. The comparison is about convergence on a fixed toy problem, not measured wall-clock performance or a claim about optimized parallel implementations.

## Conjugate gradients chooses directions with memory

**Conjugate gradients** solves a symmetric positive definite system by minimizing the associated quadratic objective over expanding Krylov subspaces. Its search directions are conjugate under the A inner product, which prevents the method from undoing earlier progress in exact arithmetic.

The residual starts as b minus A x. Each iteration computes a matrix-vector product, updates the solution and residual, and constructs the next direction from the new residual and the previous direction. The lab exposes these operations directly rather than hiding them inside a solver call.

For an n-dimensional problem in exact arithmetic, CG terminates in at most n iterations. The actual number can be smaller when the right-hand side excites fewer distinct eigenvalue components. Finite precision changes that exact statement, so practical solvers use a residual tolerance and an iteration limit.

A common error bound involves the square root of the condition number. Eigenvalue clustering can make practical convergence better than a worst-case bound suggests. This is one reason a preconditioner that clusters eigenvalues can help even when its effect is not summarized perfectly by one scalar condition number.

**Try it.**

You can change the system size from twelve to twenty-four and compare Jacobi's residual after the same number of steps. Then record CG's iterations. Explain why the observed CG count can be below the dimension for this symmetric right-hand side.

## Nonsymmetric matrices need different Krylov methods

**GMRES** builds an orthonormal basis of a Krylov subspace using Arnoldi iteration and chooses the approximation with the smallest residual norm in that subspace. Its basis storage and orthogonalization work grow with iteration count. Restarted GMRES limits memory by periodically discarding most of the basis, but restarting can slow or stall convergence.

BiCG and BiCGSTAB use shorter recurrences for nonsymmetric problems. Their residual behavior can be less smooth, and breakdown conditions need attention. MINRES handles symmetric indefinite systems by minimizing a residual over a Krylov subspace while preserving symmetry assumptions.

Choosing CG simply because a matrix is square is incorrect. Symmetry and positive definiteness are essential to its standard guarantees. A matrix that is almost symmetric because of an assembly bug should be diagnosed rather than passed silently to a symmetric solver.

## Preconditioning changes the geometry of the solve

A **preconditioner** is an operator that approximates the difficult inverse action cheaply. Left preconditioning solves M inverse A x equals M inverse b. Right preconditioning changes variables through x equal to M inverse y. The exact solution is unchanged when the transformations are valid, but residual interpretation can differ.

A diagonal preconditioner rescales coordinates. Incomplete LU or incomplete Cholesky retains only part of the factorization structure. Block methods group strongly coupled variables. Multigrid provides an approximate solve across scales. A useful preconditioner balances setup cost, application cost, memory, and iteration reduction.

Preconditioned CG needs a compatible symmetric positive definite preconditioner and the correct transformed inner-product structure. Applying an arbitrary nonsymmetric inverse approximation can destroy its assumptions. Flexible Krylov methods allow a changing preconditioner when the algorithm accounts for that change.

## Stopping should match the larger calculation

A practical rule compares the residual norm with an absolute tolerance and a relative tolerance times the right-hand-side norm. A scale-aware backward error can be more informative when matrix and solution scales vary widely. The recursive residual inside an iteration can drift from the true residual, so periodic recomputation can help.

A tiny residual can coexist with a substantial solution error when the matrix is ill conditioned. A PDE calculation also has discretization error, and an optimization method has an outer convergence target. The inner linear solve should be accurate enough for those goals, with its assumptions and tolerances recorded.

**Try it.**

You can replace the right-hand side with a seeded random vector and compare the CG iteration count. Then calculate the true residual from the final solution rather than trusting the recursively updated residual. Explain what agreement between those two quantities verifies.

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/15_iterative_solvers.py
```

```python
"""Compare stationary iteration and conjugate gradients on an SPD system."""
import numpy as np

n = 12
A = 2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
b = np.ones(n)
x = np.zeros(n)
for _ in range(100):
    x += (b-A@x)/np.diag(A)
print(f"Jacobi residual after 100 steps: {np.linalg.norm(b-A@x):.6e}")
x = np.zeros(n);r = b.copy();p = r.copy();rr = r@r
for k in range(n):
    Ap = A@p
    alpha = rr/(p@Ap)
    x += alpha*p;r -= alpha*Ap
    newrr = r@r
    if np.sqrt(newrr)<1e-10:
        break
    p = r+(newrr/rr)*p;rr = newrr
print(f"CG iterations: {k+1}")
print(f"CG residual: {np.linalg.norm(b-A@x):.6e}")
print(f"solution error: {np.linalg.norm(x-np.linalg.solve(A,b)):.6e}")
assert np.allclose(A@x,b)
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
Jacobi residual after 100 steps: 1.692648e-01
CG iterations: 6
CG residual: 0.000000e+00
solution error: 1.776357e-15
```

Jacobi retains a residual near 0.169 after one hundred iterations. CG reaches a zero displayed residual in six iterations for this symmetric toy right-hand side. The solution agrees with the direct solve to about 1.78e-15. The six-step result is specific to the excited spectral components and should not be generalized to every twelve-variable system.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Stationary methods | A fixed error-propagation matrix controls convergence. |
| Krylov methods | An expanding subspace reuses information from earlier products. |
| Preconditioning | An approximate inverse action can improve the geometry at an added cost. |

The next lecture, [16-numerical-workflow](https://arunshar.github.io/numerical-parallel-computing/lectures/16-numerical-workflow.html), extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Which assumptions support standard conjugate gradients?**

1. The matrix is symmetric positive definite.
2. The matrix is any square matrix.

**Answer.** The matrix is symmetric positive definite. The positive quadratic geometry supports conjugacy and the standard convergence analysis.

**Question 2. What is the convergence condition for a stationary linear iteration?**

1. Its diagonal has at least one positive entry.
2. Its error-propagation spectral radius is below one.

**Answer.** Its error-propagation spectral radius is below one. All error modes must decay under repeated application.

**Question 3. Why can restarted GMRES converge more slowly?**

1. Restarting discards useful subspace information.
2. Restarting computes an exact inverse.

**Answer.** Restarting discards useful subspace information. The bounded memory comes at a possible convergence cost.

**Question 4. What does a preconditioner need to balance?**

1. It must always equal the exact inverse.
2. It balances setup and application costs against iteration savings.

**Answer.** It balances setup and application costs against iteration savings. An exact inverse can be too expensive, so useful approximations target the difficult modes cheaply.

**Question 5. Why should a solver occasionally recompute the true residual?**

1. The recursive residual can drift through rounding.
2. The original matrix changes automatically at every step.

**Answer.** The recursive residual can drift through rounding. Direct evaluation of b minus A x checks the accumulated recurrence error.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 11 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap11.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 16. Numerical workflows, multivariate models, and cross-language practice

Prepared for Arun Sharma. Codex drafted this study edition.

[15-iterative-solvers](https://arunshar.github.io/numerical-parallel-computing/lectures/15-iterative-solvers.html) emphasized that a solver sits inside a larger calculation. A reliable numerical workflow joins modeling, approximation, implementation, and validation without confusing their error sources. This lecture consolidates the CSCI 5302 handouts on Taylor expansion and two-dimensional interpolation, then connects MATLAB-style notation with NumPy and the parallel course.

**Learning goals.**
- You will construct and check a multivariate Taylor approximation.
- You will explain bilinear and bicubic interpolation on a structured grid.
- You will translate key MATLAB operations into NumPy without changing their meaning.
- You will design a validation record that separates mathematical and implementation claims.

## Setup

This lecture uses the CPU setup from [01-errors-floating-point](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html). The following standalone experiment needs only Python and NumPy.

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

**Try it.**

You can set the Hessian's off-diagonal entries to zero while leaving the diagonal unchanged. Predict the change in the second-order error rate. Then restore the cross terms and explain why the corrected model captures the interaction between x and y.

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

**Try it.**

You can create a small nonsymmetric two-dimensional test field and evaluate it with both coordinate conventions. Identify a query point where the transposition error is obvious. Then write a shape and value assertion that would catch the mistake before a larger simulation runs.

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

### Review questions and answers

**Question 1. What does the Hessian add to a Taylor model?**

1. It adds quadratic curvature and cross-coordinate interactions.
2. It only changes the constant term.

**Answer.** It adds quadratic curvature and cross-coordinate interactions. The quadratic form includes both diagonal and mixed second derivatives.

**Question 2. Why do bilinear weights sum to one?**

1. It guarantees exact interpolation of every nonlinear field.
2. That property preserves constant fields.

**Answer.** That property preserves constant fields. A weighted sum of equal corner values should reproduce the same constant.

**Question 3. How many coefficients does a bicubic polynomial have?**

1. It has sixteen coefficients.
2. It has exactly four coefficients.

**Answer.** It has sixteen coefficients. There are four powers in each of two coordinates, giving four times four terms.

**Question 4. Which NumPy operator represents matrix multiplication?**

1. The * operator always represents matrix multiplication.
2. The @ operator represents matrix multiplication.

**Answer.** The @ operator represents matrix multiplication. Ordinary NumPy arrays use * for elementwise multiplication and @ for matrix products.

**Question 5. What does a manufactured-solution test establish?**

1. It supports verification of the stated numerical problem.
2. It proves that a physical model matches every real system.

**Answer.** It supports verification of the stated numerical problem. A known synthetic answer checks implementation and discretization without establishing real-world validity.


## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 1 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap01.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.


---

# 17. Parallel architectures and memory locality

Prepared for Arun Sharma. Codex drafted this study edition.

Numerical methods describe the arithmetic that produces an answer. Parallel architecture determines how many operations can proceed together and how quickly their operands arrive. We begin the parallel track by separating instruction execution, data movement, and synchronization. This distinction explains why doubling the number of cores rarely doubles application speed.

**Learning goals.**
- You can distinguish SIMD, MIMD, shared memory, and distributed memory.
- You can calculate cache-line traffic for a simple access pattern.
- You can explain why locality and false sharing matter before adding threads.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/17-architecture-cache.py`.

## A processor has several sources of parallelism

**Instruction-level parallelism** overlaps independent instructions from one instruction stream. A pipeline divides instruction execution into stages. Several instructions occupy different stages at the same time, much as several packages occupy different stations in a factory. A deeper pipeline can improve throughput without reducing the latency of one dependent operation. A dependency such as `x = f(x)` still forces the next invocation to wait for the previous result.

A superscalar processor can issue multiple suitable instructions during one cycle. Out-of-order execution searches for ready instructions while older instructions wait for operands. The processor must preserve the observable behavior required by the instruction set. These mechanisms do not remove true data dependencies. Branch prediction also matters because the machine may discard speculative work after learning that a branch prediction was wrong.

**SIMD** applies one operation to several data lanes. A vector addition can add several pairs of floating-point numbers with one vector instruction. Vector width describes capacity per instruction, not the number of independent programs. Tail elements, misaligned access, reductions, and irregular branches can limit useful lane occupancy. AVX is an x86 vector-instruction family. An Apple Silicon machine uses a different instruction set, so an AVX example cannot simply be executed on every laptop.

**MIMD** processors execute different instruction streams on different data. Multiple CPU cores and separate cluster nodes fit this model. A machine can combine MIMD across cores and SIMD within each core. Flynn's taxonomy therefore gives useful categories without forcing a modern system into only one practical behavior. GPUs add their own execution and scheduling model, which later lectures examine directly.

## A memory hierarchy trades capacity for access cost

A register holds a small amount of data near an execution unit. Caches hold larger blocks, and main memory holds much more. Storage adds another level with different access properties. **Latency** measures the delay before an individual request is satisfied. **Bandwidth** measures the sustained volume transferred per unit time. A system can have high bandwidth and still have substantial latency. Many independent outstanding requests can hide latency when the application exposes enough useful work.

A cache transfers a **cache line**, not one isolated scalar. Suppose a model uses 64-byte lines and 8-byte numbers. One line contains eight adjacent numbers. A sequential traversal consumes those eight numbers after one line fill. A stride of eight numbers touches one useful value from each line. The arithmetic stays the same, but useful bytes per fetched line drop from 64 to 8. These are model parameters, not claims about the line size of the computer running this page.

Temporal locality means that a program reuses data soon. Spatial locality means that nearby addresses are accessed close together in time. Matrix loop order affects both. A row-major matrix stores a row contiguously. Traversing columns can create large address jumps. Blocking partitions a matrix into tiles that fit a chosen cache level so that each fetched tile supports many arithmetic operations. A tile must leave space for other live data and cannot be sized only from the advertised cache capacity.

## Shared addresses do not imply uniform access

A shared-memory program gives threads access to one address space. Hardware coherence keeps cached copies consistent according to the machine's coherence rules. **NUMA** systems attach memory to different processor domains. A remote NUMA access can have a different cost from a local access even though the pointer syntax is identical. First-touch allocation and thread placement can affect locality. A placement policy must be measured on the actual system because operating systems and runtimes can move work or pages.

A **false-sharing** pattern occurs when independent variables occupy the same coherence unit and different cores repeatedly write them. The variables are logically separate, but their line moves between ownership states. Padding can separate counters, although excessive padding wastes memory and cache capacity. A better algorithm may keep a private counter in a register and combine results once. This reduces both sharing traffic and the number of stores.

Distributed-memory systems give processes separate address spaces. A process cannot dereference another process's ordinary pointer. It must communicate data through a supported mechanism such as MPI. Shared-memory and distributed-memory models describe visibility and communication, while physical deployment describes where processes run. Multiple MPI ranks can run on one laptop, and a hybrid MPI-plus-OpenMP application can combine both models.

## A cost model guides the first optimization

A simple roofline estimate compares arithmetic work with transferred bytes. Arithmetic intensity is `operations / bytes`. If a kernel performs two operations while streaming three 8-byte arrays, its simplified intensity is $2/24=1/12$ operations per byte. A hypothetical bandwidth of 120 GB/s would then support at most 10 billion operations per second under that traffic model. Cache reuse, write allocation, fused operations, and data types change the actual accounting. The estimate is useful because it exposes which assumption must improve.

A parallel random-access machine, or **PRAM**, abstracts away these physical costs and lets processors access a shared memory through idealized rules. EREW, CREW, and CRCW variants differ in whether simultaneous reads and writes are allowed. PRAM algorithms reveal dependencies and asymptotic opportunities. They do not predict a real machine's memory traffic without a mapping step. We will keep algorithmic work, communication cost, and measured runtime as separate quantities throughout the course.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
N, LINE, ITEM = 64, 64, 8
rows = []
for stride in (1, 2, 4, 8):
    addresses = [i * stride * ITEM for i in range(N)]
    lines = {a // LINE for a in addresses}
    rows.append({'stride_elements': stride, 'useful_bytes': N * ITEM,
                 'line_bytes': len(lines) * LINE,
                 'utilization': N * ITEM / (len(lines) * LINE)})
assert [r['line_bytes'] for r in rows] == [512, 1024, 2048, 4096]
print(json.dumps(rows, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "stride_elements": 1,
    "useful_bytes": 512,
    "line_bytes": 512,
    "utilization": 1.0
  },
  {
    "stride_elements": 2,
    "useful_bytes": 512,
    "line_bytes": 1024,
    "utilization": 0.5
  },
  {
    "stride_elements": 4,
    "useful_bytes": 512,
    "line_bytes": 2048,
    "utilization": 0.25
  },
  {
    "stride_elements": 8,
    "useful_bytes": 512,
    "line_bytes": 4096,
    "utilization": 0.125
  }
]
```

The sequential model fetches **512 bytes**, while stride eight fetches **4, 096 bytes** for the same 512 useful bytes. This experiment counts compulsory line traffic in an idealized address trace. It does not measure a hardware cache, model prefetching, or predict an eightfold runtime change. Its purpose is to make the wasted transfer visible before you benchmark.

**Try it.**

You can change LINE to 128 and predict the result for every stride before rerunning the trace. Explain why a larger transfer unit can help sequential access and hurt sparse access.

**Try it.**

You can model four thread counters placed 8 bytes apart, then 128 bytes apart. Count shared lines and explain why this count is only a warning sign, not a runtime measurement.

## Summary

| Idea | What the lesson established |
|---|---|
| Instruction parallelism | The processor overlaps instructions that do not depend on each other. |
| Data locality | The program gets more useful work from each transferred cache line. |
| Memory model | Shared memory exposes addresses, while distributed memory requires explicit data exchange. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does SIMD require a separate program for each lane?**

1. No, one operation applies across lanes.
2. Yes, each lane runs an unrelated program.

**Answer.** No, one operation applies across lanes. SIMD shares an instruction across data lanes.

**Question 2. What does bandwidth measure?**

1. Bytes transferred per unit time.
2. The delay of one request.

**Answer.** Bytes transferred per unit time. Latency measures individual delay, while bandwidth measures sustained transfer.

**Question 3. How many bytes does the stride-eight model fetch?**

1. 512 bytes.
2. 4, 096 bytes.

**Answer.** 4, 096 bytes. Each of the 64 reads touches a distinct 64-byte line.

**Question 4. Can independent variables cause coherence traffic?**

1. Yes, if they share a cache line.
2. No, distinct variables never interact.

**Answer.** Yes, if they share a cache line. False sharing concerns physical transfer units rather than logical variables.

**Question 5. Does a PRAM model include realistic network contention?**

1. No, it is an abstraction.
2. Yes, it models every packet.

**Answer.** No, it is an abstraction. A mapping and cost model are needed for physical performance.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 18. Decomposition, dependency graphs, and load balance

Prepared for Arun Sharma. Codex drafted this study edition.

[17-architecture-cache](https://arunshar.github.io/numerical-parallel-computing/lectures/17-architecture-cache.html) explained the hardware resources that can execute work together. This lecture asks whether an algorithm exposes enough independent work to use those resources. We represent an algorithm as tasks and dependencies, then decide how to group and place those tasks.

**Learning goals.**
- You can compute work and span for a dependency graph.
- You can compare data, recursive, exploratory, and speculative decomposition.
- You can explain the tradeoff between load balance and locality.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/18-decomposition-task-graphs.py`.

## A dependency graph states what must happen first

A **task** is a unit of computation that the scheduler can assign. A directed edge from task A to task B means that B needs something A produces. A valid schedule respects every edge. Task size is a design choice. One multiplication per task may expose enormous theoretical parallelism but require more scheduling work than useful computation. One entire application per task hides almost all internal parallelism.

**Work**, written $T_1$, is the total execution cost of all tasks on one processor under the chosen cost model. **Span**, written $T_\infty$, is the cost of the longest dependency path. An execution on $p$ identical processors must take at least $\max(T_1/p,T_\infty)$. The ratio $T_1/T_\infty$ is average parallelism. This bound assumes the task costs are comparable across processors and ignores communication unless communication appears explicitly in the graph.

Consider two independent tasks costing 4 and 2 units. Each feeds a separate 3-unit task, and a final 1-unit task combines their results. Total work is 13 units. The longer path costs $4+3+1=8$ units, so unlimited processors cannot finish in fewer than eight units. Two processors can achieve eight units for this graph. Four processors cannot improve the critical path merely by waiting beside it.

## Decomposition follows the problem's structure

**Data decomposition** splits an input, output, or intermediate array. A row decomposition of matrix multiplication assigns output rows to tasks. Output decomposition often avoids conflicting writes because each task owns its result region. Input decomposition can be useful when input blocks are naturally distributed, but it may require a reduction when several tasks contribute to the same output. Intermediate-data decomposition creates reusable partial results and must account for their storage and movement.

**Recursive decomposition** splits a problem into smaller instances until each leaf is large enough to compute efficiently. Merge sort and divide-and-conquer matrix algorithms provide familiar examples. The recursion creates a tree of tasks, but the merge stage can become the bottleneck. A task scheduler benefits from a cutoff that switches small subproblems to a sequential implementation. The cutoff depends on task overhead, cache behavior, and the operation inside each leaf.

**Exploratory decomposition** gives workers different branches of a search space. The amount of work is often unknown until a branch is explored. **Speculative decomposition** computes alternatives before knowing which result will be needed. Speculation trades extra work for a shorter dependency chain. It is useful only when the saved waiting time exceeds the cost of discarded computation. Search, branch prediction, and optimistic execution use related ideas under different correctness rules.

## Task interactions affect the placement policy

Two tasks may exchange data in a regular nearest-neighbor pattern, a global collective, or an irregular graph. Interaction may be static and known before execution, or dynamic and discovered from the input. The amount of data and the frequency of interaction both matter. Sending a tiny message a million times can cost more than sending the same total bytes in a few batches because each message has startup overhead.

**Static scheduling** assigns work before execution. Block scheduling assigns contiguous ranges, which often preserves locality. Cyclic scheduling spreads nearby iteration indices across workers, which can balance a predictable cost trend. Block-cyclic scheduling uses small contiguous blocks and distributes those blocks cyclically. A block size of one is pure cyclic distribution, while one block per worker approaches block distribution. This family gives a useful knob between locality and balance.

**Dynamic scheduling** assigns new work to idle workers. A central queue is simple but may become a contention point. Distributed queues and work stealing reduce central pressure, although stolen tasks may move away from their data. Work stealing commonly removes a large pending subtree so the stealing overhead supports substantial computation. The correct policy depends on whether uncertainty in task cost is more expensive than queue access and lost locality.

## An imbalance model makes the tradeoff concrete

Suppose task durations are 1, 2, 3, 4, 5, 6, 7, 8 units and four workers are available. A contiguous assignment of two tasks per worker produces loads 3, 7, 11, 15. The makespan is 15. A cyclic assignment produces loads 6, 8, 10, 12, so its makespan is 12. A greedy assignment that sorts tasks from largest to smallest and repeatedly uses the least-loaded worker achieves 9 in this case. Total work is 36, so 9 is the work lower bound.

The greedy result is a property of this small list, not proof that greedy scheduling is always optimal. It also assumes every task is ready at time zero and moving a task has no communication cost. A graph with dependencies can prevent the largest task from being ready. A matrix partition can make a balanced assignment slower if it duplicates data transfers. A realistic placement objective must include computation, communication, and memory capacity.

The **owner-computes rule** assigns an operation to the worker that owns its output or dominant input. It can reduce transfers, but it does not automatically balance load. Graph partitioning addresses a related objective by balancing vertex or edge work while reducing communication across partitions. Later lectures apply that idea to sparse matrices. The design process starts with a dependency graph, estimates task costs, records interactions, and only then chooses a mapping.


## Asymptotic notation compares growth rates

The textbook appendix uses big-O, big-Omega, and big-Theta to separate algorithm growth from one machine's clock rate. The statement $f(n)=O(g(n))$ means that fixed positive constants eventually bound $f$ above by a multiple of $g$. Big-Omega provides a corresponding eventual lower bound, and big-Theta means both bounds hold. These definitions concern sufficiently large inputs. They do not say that the hidden constant is small or that the asymptotically better method wins at every practical size.

For example, $3n^2+20n+7$ is $\Theta(n^2)$. It is also $O(n^3)$, but that upper bound is less informative. A logarithm's base changes a constant factor when the bases are fixed, so order notation commonly omits the base. The number of processors is another independent variable in a parallel algorithm. A claim such as $T_p(n,p)=O(n/p+\log p)$ must state the valid range of $p$ and account for available concurrency. Substituting more processors than useful tasks does not create fractional indivisible work.

Work and span describe two different functions of input size. A balanced reduction has linear work and logarithmic span. A sequential recurrence can have linear work and linear span. Both algorithms can therefore have the same sequential big-Theta complexity while offering very different parallel speedup. This is why a parallel analysis reports total work, critical path, and communication separately.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
costs = list(range(1, 9))
p = 4
block = [sum(costs[2*i:2*i+2]) for i in range(p)]
cyclic = [sum(costs[i::p]) for i in range(p)]
greedy = [0] * p
for cost in sorted(costs, reverse=True):
    i = min(range(p), key=lambda k: greedy[k])
    greedy[i] += cost
out = {name: {'loads': loads, 'makespan': max(loads)}
       for name, loads in [('block', block), ('cyclic', cyclic), ('greedy', greedy)]}
assert max(block) == 15 and max(cyclic) == 12 and max(greedy) == 9
print(json.dumps(out, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "block": {
    "loads": [
      3,
      7,
      11,
      15
    ],
    "makespan": 15
  },
  "cyclic": {
    "loads": [
      6,
      8,
      10,
      12
    ],
    "makespan": 12
  },
  "greedy": {
    "loads": [
      9,
      9,
      9,
      9
    ],
    "makespan": 9
  }
}
```

The modeled makespans are **15, 12,and 9 cost units** for block,cyclic,and greedy assignment. The best schedule matches the work lower bound for this particular independent-task list. The experiment does not execute tasks concurrently. It verifies the arithmetic of an assignment model, which you can later compare with a real runtime.

**Try it.**

You can add a two-unit migration cost to each task assigned away from its initial block. Determine whether the greedy assignment still wins after that transfer cost.

**Try it.**

You can change the costs to eight identical values, then explain why a dynamic scheduler has little useful imbalance to fix. Consider its queue overhead separately.

## Summary

| Idea | What the lesson established |
|---|---|
| Work and span | Work limits total capacity, and span limits dependency speedup. |
| Static mapping | A predictable workload can benefit from cheap scheduling and good locality. |
| Dynamic mapping | An irregular workload can benefit from reassignment despite scheduling costs. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the lower bound on runtime for work 36 and four processors?**

1. 9 cost units.
2. 4 cost units.

**Answer.** 9 cost units. The work bound is 36 divided by 4.

**Question 2. Can more processors shorten a fixed dependency chain?**

1. No, the chain still sets the span.
2. Yes, every additional processor halves it.

**Answer.** No, the chain still sets the span. Dependent tasks must preserve their order.

**Question 3. Which decomposition explores separate search branches?**

1. Exploratory decomposition.
2. Only output-array decomposition.

**Answer.** Exploratory decomposition. Search branches are tasks whose sizes may not be known in advance.

**Question 4. Does perfect load balance guarantee low runtime?**

1. No, communication can dominate.
2. Yes, placement never matters.

**Answer.** No, communication can dominate. Balance is only one part of the cost.

**Question 5. What does a task cutoff avoid?**

1. Excessive overhead on tiny tasks.
2. All numerical error.

**Answer.** Excessive overhead on tiny tasks. Small recursive tasks may cost more to schedule than to execute.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 19. Threads, synchronization, and OpenMP

Prepared for Arun Sharma. Codex drafted this study edition.

[18-decomposition-task-graphs](https://arunshar.github.io/numerical-parallel-computing/lectures/18-decomposition-task-graphs.html) separated useful parallelism from scheduling choices. Shared-memory threads turn that plan into an executable program. This lecture explains what threads share, how a race appears, and why a reduction is usually better than a lock around every arithmetic operation.

**Learning goals.**
- You can identify shared and private state in a threaded loop.
- You can explain mutexes, condition variables, barriers, and atomics.
- You can compile a small pthread and OpenMP reduction and check its answer.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/19-pthreads-openmp.py`.

## A thread shares an address space with its peers

A process owns an address space and operating-system resources. Its threads have separate execution state and stacks, but they can access shared heap objects and global variables. Passing a pointer to a worker is therefore enough to expose an array. The pointer must remain valid until the worker finishes. A common bug passes the address of a loop variable that changes before the new thread reads it. A stable argument structure per worker removes that ambiguity.

POSIX threads expose creation, joining, mutual exclusion, and condition variables through a library API. `pthread_create` starts a thread, and `pthread_join` waits for a joinable thread to finish. A program should inspect return codes and clean up successfully created threads if later creation fails. Joining also provides a clear point at which the parent can safely consume a worker's completed result.

A **data race** occurs when concurrent accesses conflict, at least one is a write, and the required synchronization is missing. In C, a data race is not merely a guarantee of a slightly wrong count. It gives undefined behavior under the language memory model. Marking a variable `volatile` does not provide mutual exclusion or turn an increment into an atomic operation. A useful review asks who owns each writable object and which synchronization establishes visibility.

## A lost update has a precise mechanism

The expression `counter +=1` includes a read, arithmetic, and a write. Two workers can both read zero, both compute one, and both write one. Two increments then produce one. A **mutex** makes a protected region exclusive. A thread locks the mutex before reading the shared state, updates that state, and unlocks the mutex afterward. All threads that access that protected state must follow the same discipline.

An **atomic operation** provides an indivisible update with specified memory-order behavior. It can avoid a general-purpose lock for a supported operation. Atomicity alone does not preserve an invariant involving several separate variables. For example, updating a balance and a transaction count may require one larger synchronization protocol. A **barrier** makes participating threads wait until all participants reach a phase boundary. A barrier does not choose which thread owns an output element before that boundary.

A condition variable lets a thread sleep until another thread changes a predicate. The waiting thread tests the predicate while holding its mutex and waits in a loop. The loop is necessary because wakeups and competing consumers can invalidate the condition before the awakened thread proceeds. The signal is a notification to recheck state, not a stored replacement for the state itself. Lock ordering also matters because two threads that acquire the same locks in opposite orders can deadlock.

## Reductions reduce shared updates

A **reduction** combines many local contributions using an operation such as addition, minimum, or maximum. Each worker accumulates a private partial sum, and the partial sums are combined after the workers finish. If a loop has one million iterations and four workers, a lock per iteration introduces one million synchronized updates. Private accumulation needs only four partial results at the final combination point.

Associativity permits a reduction tree in exact arithmetic. Floating-point addition is not exactly associative, so changing the tree can change low-order bits. A correctness test should use a justified tolerance for floating-point output and an exact comparison for small integer output. A reproducible summation tree can improve repeatability, but it may restrict scheduling or cost additional work. The numerical and parallel tracks meet directly at this issue.

## OpenMP expresses common patterns with directives

OpenMP's fork-join model creates a team for a parallel region. A worksharing loop distributes iterations among that team. `parallel for` combines these concepts. A `reduction(+:sum)` clause gives each participating thread a private reduction value and combines the values using the named operation. `default(none)` makes data-sharing decisions explicit and is useful while learning. Variables introduced inside a parallel region are often private, but the exact rule depends on their construct and scope.

A `critical` region protects a named or unnamed critical section. An `atomic` directive protects a supported memory operation. A `single` region selects one thread to execute a region, and its default implicit barrier differs from using `nowait`. A `master` or `masked` region has its own participation rules and should not be treated as a general replacement for `single`. Tasks let a thread describe deferred work, while task dependencies and task groups describe ordering and completion.

Static scheduling has low overhead and works well for uniform iterations. Dynamic scheduling hands chunks to workers as they become idle. Guided scheduling begins with larger chunks and decreases them. Nested parallel regions and libraries that create their own threads can oversubscribe the machine. Thread affinity and false sharing can also dominate a logically correct loop. You should first verify the result, then inspect the number of active threads, and only afterward interpret timing.

## The native programs use a deterministic correctness target

The native examples sum integers from 1 through 1000, whose exact result is 500500. The pthread version uses four independent argument structures and four private partial sums. The parent joins all workers and combines those sums. The OpenMP version uses a reduction and requests four threads. The output confirms a result and a completed local execution. It does not establish a speedup because such a tiny loop is dominated by startup and measurement noise.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
values = list(range(1, 1001))
partials = [sum(values[i::4]) for i in range(4)]
wrong = 0
read_a, read_b = wrong, wrong
wrong = read_a + 1
wrong = read_b + 1
assert sum(partials) == 500500 and wrong == 1
print(json.dumps({'partial_sums': partials, 'correct_sum': sum(partials),
                  'modeled_lost_update': wrong}, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "partial_sums": [
    124750,
    125000,
    125250,
    125500
  ],
  "correct_sum": 500500,
  "modeled_lost_update": 1
}
```

The private partial sums combine to **500500**, while the explicit lost-update interleaving produces **1 after two increments**. The Python program models the interleaving and is not an attempt to force a Python race. The accompanying `pthread_sum.c` and `openmp_sum.c` exercise real native concurrency. Their compiler commands and captured output appear in `evidence/parallel/native.json`.

**Try it.**

You can replace the integer input by values with very different magnitudes. Compare serial and pairwise sums, and explain why a valid reduction can change low-order floating-point bits.

**Try it.**

You can change the native pthread worker count from four to two. Verify every input still belongs to exactly one worker before interpreting any performance difference.

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The pthread example uses the native API

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { int rank; long partial; } Worker;

static void *sum_part(void *pointer) {
    Worker *worker = pointer;
    worker->partial = 0;
    for (int i = worker->rank + 1; i <= 1000; i += 4)
        worker->partial += i;
    return NULL;
}

int main(void) {
    pthread_t threads[4];
    Worker workers[4];
    int created = 0;
    for (int i = 0; i < 4; ++i) {
        workers[i] = (Worker){i, 0};
        if (pthread_create(&threads[i], NULL, sum_part, &workers[i]) != 0) {
            for (int j = 0; j < created; ++j) pthread_join(threads[j], NULL);
            fprintf(stderr, "Thread creation failed.\n");
            return EXIT_FAILURE;
        }
        ++created;
    }
    long total = 0;
    for (int i = 0; i < 4; ++i) {
        if (pthread_join(threads[i], NULL) != 0) return EXIT_FAILURE;
        total += workers[i].partial;
    }
    printf("pthread sum = %ld\n", total);
    return total == 500500 ? EXIT_SUCCESS : EXIT_FAILURE;
}

```

### The OpenMP example uses the native API

```c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    long sum = 0;
    int actual_threads = 0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(4) default(none) shared(actual_threads) reduction(+:sum)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 1; i <= 1000; ++i) sum += i;
    }
    printf("OpenMP sum = %ld, threads = %d\n", sum, actual_threads);
    return sum == 500500 ? EXIT_SUCCESS : EXIT_FAILURE;
}

```

## Summary

| Idea | What the lesson established |
|---|---|
| Ownership | Each writable location needs a clear owner or synchronization rule. |
| Reduction | Private accumulation minimizes synchronized shared updates. |
| Validation | A correct small result is a correctness check rather than a speedup claim. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does volatile make a C increment atomic?**

1. No.
2. Yes.

**Answer.** No. Volatile does not provide synchronization.

**Question 2. Why does a condition wait use a loop?**

1. The predicate must be rechecked after waking.
2. The mutex is unnecessary.

**Answer.** The predicate must be rechecked after waking. Another thread or a spurious wakeup can leave the predicate false.

**Question 3. What is the exact sum from 1 through 1000?**

1. 500500.
2. 1000000.

**Answer.** 500500. The formula is 1000 times 1001 divided by 2.

**Question 4. Does a barrier assign exclusive ownership of an array element?**

1. No.
2. Yes.

**Answer.** No. A barrier synchronizes phases rather than individual updates.

**Question 5. Can a floating-point reduction change low-order bits?**

1. Yes, its addition order can change.
2. No, addition is exactly associative.

**Answer.** Yes, its addition order can change. Floating-point rounding depends on the evaluation order.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.


---

# 20. Interconnection networks and communication costs

Prepared for Arun Sharma. Codex drafted this study edition.

[19-pthreads-openmp](https://arunshar.github.io/numerical-parallel-computing/lectures/19-pthreads-openmp.html) described coordination inside one address space. A distributed machine moves data across an interconnect. This lecture builds a cost model for that movement and explains why the communication pattern should influence the algorithm and its processor mapping.

**Learning goals.**
- You can compare rings, meshes, hypercubes, and tree networks.
- You can use a latency-bandwidth model with explicit units.
- You can explain contention, routing, and the difference between broadcast and personalized exchange.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/20-communication-networks.py`.

## A network is a graph with capacity constraints

A network node represents a processor, a router, or a switching element. A link carries data between adjacent nodes. A **static network** exposes a fixed direct-neighbor structure, while a switched network routes traffic through intermediate switching elements. Either organization has finite link bandwidth and finite buffers. An algorithm's logical communication graph must be embedded in that physical network.

A ring gives each node two neighbors and provides simple cyclic shifts. Its degree is small, but the distance between far-apart nodes grows with the number of nodes. A two-dimensional mesh connects interior nodes to four neighbors. A square mesh with $p$ nodes has side length about $\sqrt p$, so its largest route has order $\sqrt p$ hops. A torus adds wraparound links to shorten some routes. Boundary conditions in an application and wraparound links in a network are separate concepts.

A $d$-dimensional hypercube has $p=2^d$ nodes. Each node has one neighbor for each bit of its binary address, and the neighbor differs in that bit. The distance between two nodes is their Hamming distance, which is at most $d=\log_2p$. Hypercubes provide useful recursive communication patterns, although increasing degree complicates physical construction. A fat tree increases aggregate capacity toward the root to avoid the severe upper-level bottleneck of a thin tree.

The **bisection bandwidth** measures the capacity across a cut dividing the machine into two equally sized groups under the chosen definition of the cut. An all-to-all exchange can stress that cut because many messages must cross simultaneously. Small diameter alone does not guarantee high aggregate bandwidth. A star has short routes but can overload its center. Network quality therefore needs both distance and capacity measures.

## Startup and per-byte cost describe a first approximation

A common point-to-point model is $T(m)=\alpha+\beta m$, where $m$ is message size in bytes, $\alpha$ is startup time, and $\beta$ is time per byte. The reciprocal $1/\beta$ is the modeled bandwidth. Units prevent common errors. If $\alpha=2$ microseconds and bandwidth is 10 GB/s, a 1000-byte message costs approximately 2.1 microseconds in this simple model. Decimal GB/s means one billion bytes per second.

The model ignores queueing, software progress, protocol transitions, and contention. A route may add hop-dependent delay. **Store-and-forward** routing receives a whole packet before forwarding it. Cut-through and wormhole styles can pipeline portions through a path. Packetization means the cost of a long logical message can differ from multiplying a one-packet cost. These details motivate measurement, but the simple model remains useful for comparing message counts and volumes.

Batching ten small messages can save nine startups if the receiver can use one combined message. Batching also delays the first result and requires buffering. An algorithm that needs each reply before sending the next request cannot always batch freely. The dependency graph therefore limits the communication optimization just as it limits arithmetic parallelism.

## Collective patterns have different data semantics

A **broadcast** sends one root's data to every participant. A binary-tree broadcast reaches twice as many processes in each round, requiring $\log_2p$ rounds when $p$ is a power of two. A **reduction** reverses the pattern by combining inputs toward a root. An **all-reduce** leaves the combined value on every process. A reduction followed by a broadcast is a valid conceptual implementation, but actual libraries often choose better algorithms for particular sizes.

A **scatter** distributes distinct pieces from a root, and a **gather** collects distinct pieces at a root. An **all-gather** gives every participant every piece. An **all-to-all personalized exchange** sends a different destination-specific block from each process to every other process. This is fundamentally different from repeatedly broadcasting one shared block. A prefix operation, or scan, combines values in rank order and gives each process its prefix result.

A ring all-reduce can use a reduce-scatter phase followed by an all-gather phase. It uses more rounds than a tree but partitions a large vector so each link carries a useful stream of chunks. A tree can favor small messages where startup dominates. A ring can favor large messages where bandwidth dominates. The crossover depends on the network and implementation, so a model comparison is a hypothesis for benchmarking.

## Mapping can turn logical neighbors into distant machines

A stencil communicates mostly with neighboring spatial partitions. Placing those partitions on nearby network nodes can reduce link contention and route length. A poor mapping can route every local-looking exchange across the same central cut. **Embedding** asks how one graph can be represented on another. Dilation measures how long an embedded edge becomes, while congestion counts how many logical edges compete for a physical link under the mapping.

Topology-aware placement can help, but application balance still matters. A rank with twice the computation delays every synchronized phase regardless of excellent placement. Modern interconnects also use adaptive routing, so the route may not be a fixed function of rank numbers. The safest design starts with a clear communication pattern and a portable algorithm, then tests placement changes on the target machine with recorded conditions.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, math
p, alpha, bandwidth = 8, 2e-6, 10e9
rows=[]
for m in (1024, 1024*1024):
    beta=1/bandwidth
    tree=2*math.log2(p)*(alpha+beta*m)
    ring=2*(p-1)*alpha+2*(p-1)/p*beta*m
    rows.append({'bytes':m,'tree_allreduce_us':round(tree*1e6,3),
                 'ring_allreduce_us':round(ring*1e6,3)})
assert rows[0]['tree_allreduce_us'] < rows[0]['ring_allreduce_us']
assert rows[1]['ring_allreduce_us'] < rows[1]['tree_allreduce_us']
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "bytes": 1024,
    "tree_allreduce_us": 12.614,
    "ring_allreduce_us": 28.179
  },
  {
    "bytes": 1048576,
    "tree_allreduce_us": 641.146,
    "ring_allreduce_us": 211.501
  }
]
```

The model favors a tree for the 1, 024-byte message and a ring for the1 MiB message. This result follows from assumed startup and bandwidth parameters. It does not establish which algorithm an MPI library selects or which algorithm wins on your network. The point is to compare **round count and byte volume separately**.

**Try it.**

You can increase startup to 20 microseconds while keeping bandwidth fixed. Predict which method becomes more attractive for medium messages.

**Try it.**

You can draw an eight-node ring and an eight-node hypercube. Find the route from node 0 to node 7 in each graph and explain why hop count does not fully determine runtime.

## Summary

| Idea | What the lesson established |
|---|---|
| Topology | A network has both route-length and shared-capacity constraints. |
| Cost model | Startup penalizes many small messages, and bandwidth penalizes large byte volume. |
| Collective choice | Different message sizes can favor different communication schedules. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the degree of a three-dimensional hypercube?**

1. 3.
2. 8.

**Answer.** 3. Each node has one neighbor per address bit.

**Question 2. What are the units of beta in alpha plus beta times bytes?**

1. Seconds per byte.
2. Bytes per second.

**Answer.** Seconds per byte. Multiplying beta by bytes must produce seconds.

**Question 3. Does all-to-all send the same data to every destination?**

1. Not in a personalized all-to-all.
2. Always.

**Answer.** Not in a personalized all-to-all. Each destination can receive a distinct block from each sender.

**Question 4. What does bisection bandwidth describe?**

1. Capacity across a balanced network cut.
2. The number of instructions per core.

**Answer.** Capacity across a balanced network cut. Many simultaneous routes may compete for that cut.

**Question 5. Does the model prove ring all-reduce always wins?**

1. No, the result depends on size and assumptions.
2. Yes.

**Answer.** No, the result depends on size and assumptions. The small-message example favors the tree.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 21. MPI point-to-point communication and correctness

Prepared for Arun Sharma. Codex drafted this study edition.

[20-communication-networks](https://arunshar.github.io/numerical-parallel-computing/lectures/20-communication-networks.html) modeled the cost of moving data. MPI gives a portable interface for expressing those transfers between processes. This lecture focuses on message matching, buffer ownership, deadlock, and completion, because a fast communication pattern is useless when it can hang or read unfinished data.

**Learning goals.**
- You can identify the source, destination, tag, communicator, and datatype of a message.
- You can explain why blocking send does not always mean receiver completion.
- You can write a safe neighbor exchange with MPI_Sendrecv or nonblocking requests.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/21-mpi-point-to-point.py`.

## MPI processes communicate within a context

An MPI program commonly follows the single-program, multiple-data model. Every rank starts the same executable, and rank-specific branches assign different work. `MPI_Init` initializes MPI, `MPI_Comm_rank` identifies the rank within a communicator, and `MPI_Comm_size` reports that communicator's size. `MPI_Finalize` ends MPI use after outstanding communication and collective obligations have been satisfied. A rank is a communicator-relative identifier rather than a permanent hardware-core number.

A **communicator** combines a process group with a communication context. The context keeps unrelated message streams from interfering. Two libraries can use matching rank numbers and tags without receiving each other's messages if they use distinct communicators. The envelope of a point-to-point message includes its source, destination, tag, and communicator. A receive specifies matching constraints, including optional wildcards for source or tag.

The message payload has a datatype and count. `MPI_INT` describes a C integer according to the implementation's MPI binding. A count specifies elements of that datatype, not bytes unless the datatype is `MPI_BYTE`. A receive buffer must hold enough matching elements. A larger receive count can be valid because it specifies capacity. `MPI_Get_count` lets the receiver inspect how many elements arrived. A too-small receive can produce truncation rather than a partially valid algorithm.

## Completion describes buffer reuse precisely

A blocking standard-mode `MPI_Send` returns when the send buffer can be safely reused. It does not universally guarantee that the receiving program has completed its receive. MPI may copy data into an internal buffer or may wait for a matching receive, depending on message size and protocol. A program that works only because a small message was buffered is therefore incorrect as a portable design.

`MPI_Ssend` uses synchronous mode, which requires a matching receive to have started before the send completes. Buffered mode uses application-provided buffer space with its own management rules. Ready mode assumes the matching receive has already been posted. These modes are semantic contracts, not interchangeable performance switches. A sender must satisfy the required precondition rather than infer it from a test that happened to finish.

A blocking receive returns when the receive buffer holds the received data and is available for access. A nonblocking operation returns a request before completion. `MPI_Isend` does not let the program overwrite its send buffer immediately. `MPI_Irecv` does not let the program read unfinished received elements. `MPI_Wait`, `MPI_Waitall`, or a successful test establishes completion of the corresponding request. A request handle tracks an operation, not a copy of the user's array.

## A deadlock can appear when buffering changes

Consider two ranks that both call a blocking send before either posts a receive. Small messages may complete through internal buffering. Larger messages may require a rendezvous with the receiver, causing both ranks to wait forever. Changing the machine, message size, or MPI implementation can expose the bug. The successful small-message run never proved that the ordering was safe.

One solution gives the ranks complementary orderings. Another uses `MPI_Sendrecv`, which combines one send and one receive and avoids the circular blocking pattern for a suitable exchange. A third posts receives and sends nonblockingly, then waits for all requests. The best choice depends on the application's dependency structure. A halo exchange often posts receives first so incoming data already has a destination when neighbors start sending.

Message ordering also needs care. MPI preserves certain ordering guarantees for matching operations from a single execution thread, but wildcard receives and concurrent threads can introduce nondeterminism. Tags can distinguish iteration phases or message roles, although they should remain within the supported tag range. Communicator separation is often cleaner than allocating large informal tag spaces across independent modules.

## Derived datatypes describe noncontiguous memory

A matrix column in row-major storage is not contiguous. You can pack it into a temporary contiguous buffer or describe its stride with an MPI datatype. `MPI_Type_vector` constructs repeated blocks with a stride measured in units of the old datatype. `MPI_Type_commit` makes a newly constructed type available for communication, and `MPI_Type_free` releases the handle when it is no longer needed. Datatype extent controls spacing between repeated datatype instances, which can surprise you when holes or resized types are involved.

A derived datatype expresses the memory layout. It does not promise that the implementation avoids internal packing or uses a particular hardware transfer. A fair performance comparison includes datatype construction where appropriate and separates one-time setup from repeated exchanges. Explicit packing can make irregular data easier to inspect, while datatypes can reduce manual indexing errors in stable layouts.

## Nonblocking calls create an opportunity for overlap

An application can initiate communication, compute on independent interior data, then wait before using halo data. This overlap is valid only if the interior computation does not touch active communication buffers in a conflicting way. MPI progress behavior and hardware support determine whether communication advances while computation runs. A nonblocking call alone is not evidence of overlap. You should measure a timeline or compare carefully defined elapsed intervals on the target implementation.

The small native program in this lecture performs a ring exchange with `MPI_Sendrecv`. Each rank sends its rank number to the right neighbor and receives from the left. The last rank sends to rank zero, so the communicator forms a logical ring even when all processes run on one host. The validation checks received values, which establishes message-routing correctness without claiming network performance.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
p=4
rows=[{'rank':r,'send_to':(r+1)%p,'receive_from':(r-1)%p,
       'received_value':(r-1)%p} for r in range(p)]
assert [r['received_value'] for r in rows]==[3,0,1,2]
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "rank": 0,
    "send_to": 1,
    "receive_from": 3,
    "received_value": 3
  },
  {
    "rank": 1,
    "send_to": 2,
    "receive_from": 0,
    "received_value": 0
  },
  {
    "rank": 2,
    "send_to": 3,
    "receive_from": 1,
    "received_value": 1
  },
  {
    "rank": 3,
    "send_to": 0,
    "receive_from": 2,
    "received_value": 2
  }
]
```

The model predicts received values **3, 0, 1, 2** at ranks 0 through 3. The native `mpi_exchange.c` checks the same mapping with real MPI processes. The saved native log records its result. Both tests concern correctness. Local rank-to-rank communication does not characterize communication across a cluster interconnect.

**Try it.**

You can reverse the direction of the ring and update the expected receive rank. Explain how source and destination change without changing the communicator.

**Try it.**

You can sketch a nonblocking halo exchange with distinct send and receive buffers. Mark the earliest point at which each buffer may be reused.

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The MPI ring exchange example uses the native API

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int received = -1;
    MPI_Sendrecv(&rank, 1, MPI_INT, right, 10,
                 &received, 1, MPI_INT, left, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int ok = received == left, all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI ring exchange: ranks = %d, all_correct = %d\n", size, all_ok);
    MPI_Finalize();
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

```

## Summary

| Idea | What the lesson established |
|---|---|
| Matching | A communicator and message envelope determine which receive can match a send. |
| Completion | A completed request establishes when its buffer may be accessed or reused. |
| Deadlock avoidance | A portable exchange must not depend on unspecified internal buffering. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Can MPI_Send return before the receiver finishes its receive?**

1. Yes, standard mode can use buffering.
2. No, it always waits for receiver completion.

**Answer.** Yes, standard mode can use buffering. Send completion guarantees safe send-buffer reuse.

**Question 2. Can a program overwrite an active MPI_Isend buffer?**

1. No, it must establish completion first.
2. Yes, the call always copied it.

**Answer.** No, it must establish completion first. Nonblocking initiation does not imply a completed copy.

**Question 3. What does a communicator provide besides ranks?**

1. A communication context.
2. A promise of one rank per physical core.

**Answer.** A communication context. The context separates independent communication streams.

**Question 4. What can happen when a receive buffer is too small?**

1. A truncation error can occur.
2. MPI always allocates a larger user buffer.

**Answer.** A truncation error can occur. The receiver must provide adequate capacity.

**Question 5. Does nonblocking initiation prove useful overlap?**

1. No, progress and independent work must be verified.
2. Yes.

**Answer.** No, progress and independent work must be verified. The API exposes an opportunity rather than a timing guarantee.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.


---

# 22. MPI collectives, groups, and virtual topologies

Prepared for Arun Sharma. Codex drafted this study edition.

[21-mpi-point-to-point](https://arunshar.github.io/numerical-parallel-computing/lectures/21-mpi-point-to-point.html) explained how individual messages match and complete. Many algorithms need a group-wide operation instead of a hand-written collection of sends. This lecture turns those operations into clear data-layout contracts and explains how communicators describe smaller teams and logical neighborhoods.

**Learning goals.**
- You can choose between broadcast, scatter, gather, reduce, all-reduce, and scan.
- You can explain collective ordering and communicator participation.
- You can map a two-dimensional rank grid to a matrix algorithm.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/22-mpi-collectives-topology.py`.

## A collective has a data contract

A broadcast gives every rank a copy of the root's input. The root argument identifies a rank within the participating communicator. A scatter distributes separate pieces from the root, and a gather collects separate pieces at the root. An all-gather collects everyone's contribution at every rank. A reduction combines corresponding elements using an operation, while an all-reduce also distributes the combined result to every rank.

These operations differ in who owns valid input and output buffers. In a root-based reduction, the final output is meaningful at the root. A non-root process should not read an arbitrary receive buffer and assume it contains the reduced result. An all-reduce provides that result to every participant. A scan gives rank $r$ the combination of ranks 0 through $r$, while an exclusive scan omits rank $r$ itself. The exclusive scan's rank-zero result needs special handling according to the API contract.

Suppose four ranks hold sums 3, 7, 11, 15 from disjoint blocks. A sum reduction produces 36 at one root. An all-reduce produces 36 everywhere. An inclusive scan produces 3, 10, 21, 36. The same inputs therefore support different useful outputs. A distributed offset calculation often needs an exclusive scan, while a convergence test often needs an all-reduce of a maximum residual or a logical condition.

## Every participant must follow a compatible order

Collectives coordinate processes in a communicator. If one rank calls a broadcast while another calls a reduction at the corresponding position, the program is invalid even if both calls involve the same data size. Control flow must keep the collective sequence compatible across participants. A condition based only on a local value can break that agreement. A preceding collective can establish a shared decision when the algorithm requires one.

A collective is not automatically a barrier. Some calls can return on one rank while another rank is still completing its part. Programs must rely on the specified data and completion guarantees rather than adding an imaginary global synchronization. `MPI_Barrier` provides an explicit participation boundary when that boundary is required. Extra barriers can hide load imbalance or distort measurements, so they should serve a stated purpose.

Nonblocking collectives return a request and follow similar buffer-lifetime rules to nonblocking point-to-point operations. They can support overlap with independent work. Their initiation order still matters. Persistent collectives separate repeated setup from activation for a stable operation signature. These advanced forms should follow a correct blocking implementation, because debugging a changing buffer layout and asynchronous completion at the same time is difficult.

## Irregular partitions need explicit counts and offsets

Uniform scatter and gather assume regular contributions. Their variable-count variants support uneven partitions through counts and displacements. For ten elements on four ranks, a balanced contiguous partition can use counts 3, 3, 2, 2 and displacements 0, 3, 6, 8. A displacement is measured according to the relevant datatype's extent, so it is not universally a byte offset. Every count and boundary should be checked against the allocated buffer.

An all-to-all exchange often appears when changing data layout. Each rank first groups items by destination, exchanges destination counts, computes offsets, and then moves the variable-sized payloads. A missing count exchange can leave receivers with incorrect capacity. The operation may also require substantial temporary storage. An algorithm that reduces arithmetic but doubles peak communication buffers can fail because of memory capacity before it reaches a timing bottleneck.

## Communicators define independent teams

`MPI_Comm_split` partitions an existing communicator by a color and orders the members by a key. Ranks with the same color form a new communicator. A key controls their relative order but does not determine their physical placement. A process can participate in several communicators and have a different rank in each. Code must therefore keep communicator-specific rank variables separate.

Groups describe ordered process collections, while communicators add communication context. Duplicating a communicator can isolate a library's message traffic from its caller. Freeing a communicator releases that MPI object when the application no longer needs it. These operations describe logical membership. They do not create a new operating-system process or move a process to another node.

A Cartesian communicator attaches dimensions, periodicity, and coordinates to a process group. A 2-by 2 grid contains four ranks and gives each rank a row and column coordinate. `MPI_Cart_shift` identifies neighbors along one dimension, with `MPI_PROC_NULL` representing a missing neighbor at a nonperiodic boundary. The null-process behavior simplifies boundary exchange because the same call structure can handle interior and edge ranks.

## A virtual grid supports matrix communication

Dense matrix multiplication often uses a two-dimensional process grid. Ranks in one row share a row communicator, and ranks in one column share a column communicator. A panel of A can be broadcast along its process row while a panel of B is broadcast down its process column. Each rank multiplies its received panels into its local C block. This is a central idea behind SUMMA-style algorithms.

The virtual topology gives a clear naming system and may help an implementation optimize mapping when reordering is permitted. It does not guarantee that adjacent Cartesian ranks occupy adjacent physical switches. A portable program should obtain coordinates and neighbors through MPI rather than assuming a fixed arithmetic rank formula after allowing reordering. The native example tests a fixed 2-by 2 grid without reordering to keep the first experiment transparent.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
counts=[3,3,2,2]
offsets=[sum(counts[:i]) for i in range(4)]
values=list(range(10))
chunks=[values[d:d+c] for d,c in zip(offsets,counts)]
assert sum(chunks,[])==values
local=[3,7,11,15]
prefix=[sum(local[:i+1]) for i in range(4)]
assert prefix==[3,10,21,36]
print(json.dumps({'counts':counts,'displacements':offsets,'chunks':chunks,
                  'inclusive_scan':prefix,'allreduce_sum':sum(local)},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "counts": [
    3,
    3,
    2,
    2
  ],
  "displacements": [
    0,
    3,
    6,
    8
  ],
  "chunks": [
    [
      0,
      1,
      2
    ],
    [
      3,
      4,
      5
    ],
    [
      6,
      7
    ],
    [
      8,
      9
    ]
  ],
  "inclusive_scan": [
    3,
    10,
    21,
    36
  ],
  "allreduce_sum": 36
}
```

The model partitions ten elements without overlap or omission and produces the expected inclusive scan. The native `mpi_collectives.c` adds a real four-rank scatter, reduction, all-reduce, and Cartesian-neighbor check. Its integer results are deterministic. The MPI implementation remains free to choose its internal communication algorithm.

**Try it.**

You can distribute eleven elements across four ranks and compute balanced counts and displacements. Verify that the reconstructed sequence is exactly the input.

**Try it.**

You can change a Cartesian dimension from periodic to nonperiodic. Predict which boundary neighbors become MPI_PROC_NULL before running a native variation.

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The MPI collectives example uses the native API

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "This example requires four ranks.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int values[4] = {3, 7, 11, 15}, local = 0, root_sum = 0, all_sum = 0;
    MPI_Scatter(values, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local, &root_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &all_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int dims[2] = {2, 2}, periods[2] = {1, 0};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2], source, destination;
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Cart_shift(grid, 1, 1, &source, &destination);
    int expected_left = coords[1] == 0 ? MPI_PROC_NULL : rank - 1;
    int expected_right = coords[1] == 1 ? MPI_PROC_NULL : rank + 1;
    int ok = all_sum == 36 && source == expected_left && destination == expected_right;
    if (rank == 0) ok = ok && root_sum == 36;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI collectives: sum = %d, topology_correct = %d\n", all_sum, all_ok);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

```

## Summary

| Idea | What the lesson established |
|---|---|
| Collective semantics | Each operation specifies a different input-to-output distribution. |
| Participation | Every communicator member must follow a compatible collective sequence. |
| Virtual topology | Coordinates simplify neighbor logic without promising physical placement. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Which operation places a reduced result on every rank?**

1. MPI_Allreduce.
2. MPI_Reduce alone.

**Answer.** MPI_Allreduce. Reduce produces the result at its root.

**Question 2. Is every collective a barrier?**

1. No.
2. Yes.

**Answer.** No. Programs must use each operation's specified completion semantics.

**Question 3. What are balanced counts for ten items on four ranks?**

1. 3, 3, 2, 2.
2. 4, 4, 4, 4.

**Answer.** 3, 3, 2, 2. The counts sum to ten and differ by at most one.

**Question 4. Does MPI_Comm_split create new operating-system processes?**

1. No, it creates a communicator.
2. Yes.

**Answer.** No, it creates a communicator. It reorganizes existing participants logically.

**Question 5. What can represent a missing Cartesian neighbor?**

1. MPI_PROC_NULL.
2. An arbitrary rank outside the communicator.

**Answer.** MPI_PROC_NULL. MPI defines null-process communication behavior for this purpose.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.


---

# 23. Speedup, efficiency, scaling, and isoefficiency

Prepared for Arun Sharma. Codex drafted this study edition.

[22-mpi-collectives-topology](https://arunshar.github.io/numerical-parallel-computing/lectures/22-mpi-collectives-topology.html) showed how group operations organize communication. We now ask when the parallel algorithm is worth using. A useful performance claim states the baseline, problem size, resource count, data movement, and measured interval instead of reporting one unexplained speedup number.

**Learning goals.**
- You can calculate speedup, efficiency, cost, and overhead.
- You can distinguish strong scaling, weak scaling, and scaled speedup.
- You can derive a simple isoefficiency condition and identify its assumptions.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/23-performance-models.py`.

## A speedup needs a well-defined baseline

Let $T_s(n)$ be the runtime of a chosen sequential algorithm on a specified machine and $T_p(n,p)$ the runtime of a parallel implementation using $p$ processors. **Speedup** is $S=T_s/T_p$, and **efficiency** is $E=S/p$. A speedup of 6 on 8 processors gives efficiency 0.75. This does not mean that each processor is idle exactly 25% of the time, because different forms of overhead and baseline differences can produce the same ratio.

The **parallel cost** is $pT_p$. An implementation is cost-optimal in the asymptotic sense when this cost has the same order as the best relevant sequential work. A method can run quickly while wasting enough resources to be poor for a throughput-limited service. Conversely, a modest speedup can still be valuable when the sequential job misses a deadline. The objective determines whether latency, total core-hours, memory capacity, or energy matters most.

Define overhead as $T_o=pT_p-T_s$. It can include communication, redundant work, waiting, scheduling, and other costs. This definition compares entire runtimes, so it can become negative when the parallel execution improves cache behavior relative to the baseline. Such a result needs explanation rather than an automatic claim that all standard limits were defeated. A different baseline algorithm or a different memory regime may be responsible.

## Amdahl's law holds the workload fixed

Suppose a fraction $f$ of sequential time is inherently serial, while the remaining fraction scales ideally. The model gives

$$S(p)=\frac{1}{f+(1-f)/p}.$$

For $f=0.1$ and $p=4$, speedup is about 3.077. As $p$ increases without bound, the limit is 10. This is a model for a fixed workload with an unchanged serial fraction. Communication overhead would lower the prediction further. Optimizing the serial phase changes $f$ and therefore changes the bound.

The serial fraction must be defined from the baseline workload. A measured parallel execution can spend a large fraction of its wall time in a short serial phase because the parallel phase has become faster. Substituting that observed parallel fraction directly into the fixed-work Amdahl formula mixes two denominators. Derivations should write down which runtime each fraction uses before manipulating it.

## Gustafson's model changes the workload

A scaled workload can grow as resources increase. If a parallel execution spends fraction $\alpha$ of its time in a serial part and fraction $1-\alpha$ using $p$ processors, a scaled-speedup model is $S_G=p-\alpha(p-1)$. With $\alpha=0.1$ and $p=4$, this gives 3.7. This does not contradict the Amdahl value 3.077 because the experiments hold different things fixed.

**Strong scaling** fixes the global problem size and adds resources. Each processor gets less useful work, while communication and synchronization eventually dominate. **Weak scaling** keeps some measure of work per processor fixed while growing the global problem. Ideal weak-scaling time is constant under that definition. A two-dimensional domain can grow in area while each partition's boundary-to-interior ratio changes, so the exact scaling rule must be stated.

Memory capacity is another reason to scale. A job may become possible only when aggregate memory grows. Comparing its parallel runtime with a hypothetical sequential run that cannot fit in memory requires care. You can report throughput, time-to-solution, or a model-based baseline, but you should label a modeled baseline clearly. A speedup computed from a measurement that never occurred is not an empirical speedup.

## Isoefficiency asks how fast the problem must grow

From the overhead definition, efficiency is $E=T_s/(T_s+T_o)$. Holding a target efficiency fixed requires sequential work to grow proportionally to overhead. This relation defines an **isoefficiency function** when the overhead can be expressed in terms of problem size and processor count. The result describes scalability of an algorithm-machine model, not a universal hardware score.

Consider a model with work $W$ and runtime $T_p=W/p+\log_2p$ in normalized units. Its overhead is $p\log_2p$. To maintain efficiency 0.8, the relation $W/(W+p\log_2p)=0.8$ gives $W=4p\log_2p$. A workload that grows only proportionally to $p$ does not maintain the same efficiency in this model. Concurrency limits and memory capacity can impose additional growth requirements, so the largest relevant requirement controls the useful regime.

The target efficiency 0.8 is inapplicable when $p=1$. The overhead is zero, so every positive workload has ideal efficiency 1. The output uses `null` for this case because no positive work attains exactly 0.8.

A work-span bound adds another limit. If the critical path grows almost as quickly as total work, adding processors cannot create enough independent tasks. A communication-volume model may reveal a different limit even when the graph has abundant parallelism. A complete analysis checks work, span, communication, and memory together.

## A measurement protocol makes the claim reproducible

A trustworthy benchmark records input dimensions, data type, compiler flags, library versions, process and thread counts, affinity, warmup, repetition count, and the timed region. The result should include variability. MPI jobs often report the maximum rank time for a phase because the slowest rank determines completion. Summing rank times answers a different resource-use question. A barrier before timing can align a start, but it also changes the experiment and should be documented.

A performance model is useful before timing because it predicts which curve should bend and why. After timing, a mismatch is evidence about an assumption. It may reveal memory bandwidth saturation, a hidden serial section, contention, or a weak baseline. It is not a reason to silently redefine the model until the graph looks right. The experiment below uses exact formulas and makes no claim about measured machine speed.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, math
rows=[]
for p in (1,2,4,8,16):
    amdahl=1/(0.1+0.9/p)
    gustafson=p-0.1*(p-1)
    required=4*p*math.log2(p) if p>1 else None
    rows.append({'processors':p,'amdahl_speedup':round(amdahl,4),
                 'gustafson_scaled_speedup':round(gustafson,4),
                 'work_for_efficiency_0_8':required})
assert rows[0]['work_for_efficiency_0_8'] is None
assert abs(rows[2]['amdahl_speedup']-3.0769)<1e-4
for row in rows[1:]:
    p=row['processors']
    work=row['work_for_efficiency_0_8']
    efficiency=work/(work+p*math.log2(p))
    assert abs(efficiency-0.8)<1e-12
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "processors": 1,
    "amdahl_speedup": 1.0,
    "gustafson_scaled_speedup": 1.0,
    "work_for_efficiency_0_8": null
  },
  {
    "processors": 2,
    "amdahl_speedup": 1.8182,
    "gustafson_scaled_speedup": 1.9,
    "work_for_efficiency_0_8": 8.0
  },
  {
    "processors": 4,
    "amdahl_speedup": 3.0769,
    "gustafson_scaled_speedup": 3.7,
    "work_for_efficiency_0_8": 32.0
  },
  {
    "processors": 8,
    "amdahl_speedup": 4.7059,
    "gustafson_scaled_speedup": 7.3,
    "work_for_efficiency_0_8": 96.0
  },
  {
    "processors": 16,
    "amdahl_speedup": 6.4,
    "gustafson_scaled_speedup": 14.5,
    "work_for_efficiency_0_8": 256.0
  }
]
```

The four-processor row gives **3.0769 fixed-work speedup** and **3.7 scaled speedup** under different assumptions. The table is a calculation, not a benchmark. For the isoefficiency example, the required work at four processors is 32 normalized units, which gives 0.8 efficiency in the stated runtime model.

**Try it.**

You can reduce the serial fraction from 0.1 to 0.01. Calculate the asymptotic limit and compare the benefit with doubling processors from 8 to 16.

**Try it.**

You can add a communication term proportional to the square root of the work. Derive the new condition for fixed efficiency instead of guessing it from a plot.

## Summary

| Idea | What the lesson established |
|---|---|
| Strong scaling | The global problem stays fixed while resources increase. |
| Weak scaling | The chosen work-per-resource quantity stays fixed while the global problem grows. |
| Isoefficiency | The workload must grow fast enough to keep overhead from dominating. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is efficiency for speedup 6 on 8 processors?**

1. 0.75.
2. 1.33.

**Answer.** 0.75. Efficiency divides speedup by processor count.

**Question 2. What is the Amdahl limit for serial fraction 0.1?**

1. 10.
2. 100.

**Answer.** 10. The ideal asymptotic limit is 1 divided by the serial fraction.

**Question 3. Do Amdahl and Gustafson hold the same workload fixed?**

1. No.
2. Yes.

**Answer.** No. Their different assumptions explain their different results.

**Question 4. Which rank duration determines a synchronized phase's completion?**

1. The maximum relevant rank duration.
2. The minimum rank duration.

**Answer.** The maximum relevant rank duration. The phase cannot finish before its slowest required participant.

**Question 5. Is a formula table a measured speedup result?**

1. No, it is a model calculation.
2. Yes.

**Answer.** No, it is a model calculation. An empirical claim requires saved measured runtime evidence.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 24. Dense matrix algorithms and two-dimensional layouts

Prepared for Arun Sharma. Codex drafted this study edition.

[23-performance-models](https://arunshar.github.io/numerical-parallel-computing/lectures/23-performance-models.html) explained why communication and work must be counted separately. Dense matrix multiplication offers enough arithmetic reuse to make that distinction concrete. This lecture derives blocked multiplication, Cannon’s algorithm, and panel-based alternatives, then connects the same layout ideas to solving linear systems.

**Learning goals.**
- You can explain matrix-vector and matrix-matrix communication patterns.
- You can trace Cannon’s initial skew and cyclic shifts.
- You can distinguish block, cyclic, and block-cyclic matrix distributions.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/24-dense-matrix-cannon.py`.

## Matrix multiplication exposes reusable operands

For $C=AB$, entry $C_{ij}$ is $\sum_k A_{ik}B_{kj}$. A straightforward square multiplication performs about $2n^3$ floating-point operations under the conventional multiply-plus-add count. Its inputs and output contain order $n^2$ values. The ratio suggests substantial reuse, but a poor loop order can fetch the same values repeatedly from a slower memory level. Blocking makes the intended reuse explicit.

The BLAS separates common operations into levels. Level 1 operates mainly on vectors, level 2 includes matrix-vector operations, and level 3 includes matrix-matrix operations. A dense matrix-vector product performs order $n^2$ work while reading order $n^2$ matrix data, so its arithmetic intensity is limited. A matrix-matrix product can perform order $b^3$ work from tiles containing order $b^2$ data. This difference explains why high-performance solvers reorganize work around level 3 kernels where possible.

A one-dimensional row distribution assigns each rank a group of output rows. For matrix-vector multiplication, each rank needs its rows of A and the relevant entries of x. Replicating x can be reasonable when x is small relative to A. For matrix multiplication, a row-distributed rank may need all of B, which becomes expensive as matrices grow. A two-dimensional process grid reduces the amount each rank must store and exchange.

## Cannon's algorithm aligns the inner index

Assume a square matrix is divided into $q$-by $q$ blocks and $p=q^2$ ranks form a periodic $q$-by $q$ process grid. Rank $(i,j)$ initially owns $A_{ij}$ and $B_{ij}$. Cannon’s initial alignment shifts row $i$ of A left by $i$ block positions and column $j$ of B up by $j$ positions. Rank $(i,j)$ then holds an A block and B block with the same inner block index $k=(i+j)\bmod q$.

Each rank multiplies those local blocks into its C block. The algorithm then shifts A left by one and B up by one. Repeating $q$ stages visits every inner block index once. The initial skew is therefore a correctness step, not a performance decoration. Without it, some ranks would multiply block pairs whose inner dimensions refer to different global regions.

For $q=2$, rank $(0,1)$ first computes $A_{01}B_{11}$ and then $A_{00}B_{01}$. Their sum is exactly $C_{01}$. Rank $(1,0)$ starts with $A_{11}B_{10}$ and then computes $A_{10}B_{00}$. A small trace makes the wraparound indexing much easier to trust than a large run with plausible-looking numbers.

The simplified per-rank work is order $n^3/p$. Each stage exchanges two blocks with $n^2/p$ elements, and there are $\sqrt p$ stages. The per-rank communication volume is therefore order $n^2/\sqrt p$, with order $\sqrt p$ startup events. This analysis assumes regular blocks and a suitable network mapping. The setup skew and final data distribution also need accounting in a complete timed region.

## SUMMA and block-cyclic distributions offer flexibility

A SUMMA-style multiplication iterates over panels. A rank receives an A panel broadcast along its process row and a B panel broadcast along its process column, then updates its local C block. This pattern supports rectangular process grids and more flexible panel sizes. The tradeoff includes collective startup costs, panel buffering, and the implementation's ability to overlap communication with computation.

A **block distribution** assigns a contiguous region to each rank. A **cyclic distribution** spreads individual rows or columns round-robin. A **block-cyclic distribution** spreads tiles round-robin over a process grid. If a tile has width $b$, the owner of global tile coordinate $(I,J)$ in a $p_r$-by $p_c$ grid is $(I\bmod p_r,J\bmod p_c)$. The in-rank tile indices are integer quotients. This separation between global coordinates, owner coordinates, and local indices prevents many indexing bugs.

Block-cyclic layouts help algorithms whose active work shrinks during execution. A pure block layout can leave early ranks idle as an elimination front moves. Spreading tiles shares later work more evenly. The layout also creates more communication relationships and complicates local indexing. Tile size affects cache reuse, load balance, communication count, and the amount of parallelism exposed to local kernels.

## Linear-system solvers add ordering and stability

Gaussian elimination factors a matrix into triangular factors with pivoting when needed. A distributed LU factorization performs panel work, communicates pivot decisions, and updates the trailing matrix. The trailing update maps well to matrix multiplication. The panel can form a serial or communication-sensitive bottleneck. Parallelizing the update alone therefore does not imply that the entire factorization scales equally well.

Cholesky factorization applies to symmetric positive definite matrices and avoids general pivoting under that mathematical assumption. QR factorization uses orthogonal transformations and is often attractive for numerical stability, but its communication pattern and arithmetic cost differ from LU. A parallel method must preserve the mathematical assumptions of its sequential counterpart. A fast factorization that silently drops required pivoting is a different algorithm with different failure cases.

Tridiagonal systems illustrate another choice. The Thomas algorithm performs linear work but has a sequential recurrence. Cyclic reduction removes alternating unknowns, creating logarithmic-depth stages with additional work and storage. The best method depends on whether one long system or many independent short systems are available. Parallelism across independent systems can be simpler and cheaper than parallelism within each system.

The experiment below simulates Cannon's block ownership with NumPy arrays. It checks every output against a serial matrix multiplication. The simulation validates the indexing and stage schedule. It does not run distributed matrix multiplication or measure a communication speedup.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
A=np.arange(16,dtype=np.int64).reshape(4,4)
B=(np.arange(16,dtype=np.int64).reshape(4,4)+1)
q,b=2,2
C=np.zeros((4,4),dtype=np.int64)
trace=[]
for i in range(q):
 for j in range(q):
  block=np.zeros((b,b),dtype=np.int64)
  ks=[]
  for stage in range(q):
   k=(i+j+stage)%q
   block += A[i*b:(i+1)*b,k*b:(k+1)*b] @ B[k*b:(k+1)*b,j*b:(j+1)*b]
   ks.append(k)
  C[i*b:(i+1)*b,j*b:(j+1)*b]=block
  trace.append({'rank':[i,j],'inner_blocks':ks})
assert np.array_equal(C,A@B)
print(json.dumps({'trace':trace,'C':C.tolist(),'matches_serial':True},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "trace": [
    {
      "rank": [
        0,
        0
      ],
      "inner_blocks": [
        0,
        1
      ]
    },
    {
      "rank": [
        0,
        1
      ],
      "inner_blocks": [
        1,
        0
      ]
    },
    {
      "rank": [
        1,
        0
      ],
      "inner_blocks": [
        1,
        0
      ]
    },
    {
      "rank": [
        1,
        1
      ],
      "inner_blocks": [
        0,
        1
      ]
    }
  ],
  "C": [
    [
      62,
      68,
      74,
      80
    ],
    [
      174,
      196,
      218,
      240
    ],
    [
      286,
      324,
      362,
      400
    ],
    [
      398,
      452,
      506,
      560
    ]
  ],
  "matches_serial": true
}
```

Every rank visits both inner blocks exactly once, and the simulated result matches `A @ B` exactly for the integer example. Rank(0, 1) visits inner indices **1 then 0**, which exposes the purpose of the initial skew. Floating-point variants should use a stated tolerance because the accumulation order can differ.

**Try it.**

You can use a 6-by 6 matrix with a 3-by 3 process grid and 2-by 2 blocks. Trace rank(2, 1) and verify that it visits three distinct inner indices.

**Try it.**

You can distribute a 7-by 5 matrix in 2-by 2 tiles over a 2-by 3 process grid. List each tile owner and identify the partial tiles at the boundaries.

## Summary

| Idea | What the lesson established |
|---|---|
| Blocking | Tiles make operand reuse explicit. |
| Cannon alignment | The initial skew matches the inner block index before multiplication. |
| Solver design | A scalable update kernel does not remove panel dependencies or stability requirements. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. How many ranks does a q-by-q Cannon grid use?**

1. q squared.
2. q.

**Answer.** q squared. Each process-grid coordinate owns a block pair.

**Question 2. Why is the initial skew needed?**

1. It aligns the A and B inner block indices.
2. It changes the mathematical product.

**Answer.** It aligns the A and B inner block indices. Each local multiplication must use matching inner indices.

**Question 3. What does a block-cyclic owner formula use?**

1. Tile coordinates modulo grid dimensions.
2. Only the matrix value.

**Answer.** Tile coordinates modulo grid dimensions. Ownership is determined by position and tile size.

**Question 4. Which BLAS level contains matrix-matrix multiplication?**

1. Level 3.
2. Level 1.

**Answer.** Level 3. Level 3 kernels offer substantial data reuse.

**Question 5. Does the NumPy simulation establish MPI performance?**

1. No.
2. Yes.

**Answer.** No. It checks indexing and arithmetic rather than distributed runtime.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 25. Parallel sorting, merging, and selection

Prepared for Arun Sharma. Codex drafted this study edition.

[24-dense-matrix-cannon](https://arunshar.github.io/numerical-parallel-computing/lectures/24-dense-matrix-cannon.html) used regular blocks with predictable communication. Sorting adds data-dependent destinations and an ordering requirement. This lecture compares fixed sorting networks with partition-based methods and explains why communication and load balance can dominate the comparisons.

**Learning goals.**
- You can trace odd-even transposition and a bitonic compare-exchange stage.
- You can explain parallel quicksort, sample sort, radix sort, and co-ranked merge.
- You can verify ordering and permutation preservation separately.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/25-sorting-merging.py`.

## Sorting has two correctness obligations

A sorting algorithm must put keys in the required order and preserve the input multiset. Checking only adjacent order can miss a program that silently drops difficult elements. Checking only a checksum can miss collisions or duplicate errors. A small test can compare the full result with a trusted serial sort. Larger tests should combine ordering, counts, and an appropriate permutation check. A stable sort also preserves the input order of equal keys when that order matters.

A **comparison sort** learns order by comparing keys. Its sequential comparison lower bound is order $n\log n$ for arbitrary distinct inputs. Parallel time depends on how many comparisons can occur at once and how data moves between workers. Integer-key methods such as radix sort use key structure and are not restricted to the same comparison model. A complexity claim should name that model before comparing algorithms.

## Sorting networks use a fixed communication schedule

A comparator receives two keys and emits their minimum and maximum in prescribed positions. A **sorting network** fixes the comparator graph independently of the input values. Comparators in one stage can execute in parallel when they use disjoint positions. The network depth gives a critical-path bound, while the comparator count gives work. Hardware and SIMD implementations can benefit from a predictable schedule, but a fixed network can perform unnecessary comparisons on nearly sorted data.

Odd-even transposition alternates comparisons of positions(0, 1),(2, 3),and so on with positions(1, 2),(3, 4),and so on. Repeating enough phases moves small values left and large values right. With one key per processor, the communication is local, but the number of phases is linear in the number of keys. A block version first sorts local blocks and then uses compare-split operations with neighbors, retaining the smaller or larger half as appropriate.

A **bitonic sequence** rises and then falls under a cyclic interpretation. A bitonic merge compares positions separated by half the sequence length, places the smaller values in one half, and recursively merges each half. Building bitonic sequences from smaller sorted runs produces a network with order $\log^2n$ depth and order $n\log^2n$ work for power-of-two input sizes. Padding needs a sentinel that preserves the requested order and does not conflict with valid data.

Odd-even mergesort recursively merges even-indexed and odd-indexed subsequences before a cleanup compare-exchange stage. Its name resembles odd-even transposition, but the algorithms have different structures and complexity. Keeping the full algorithm name in notes prevents that common confusion. Network sorting is particularly useful for small fixed-size blocks inside a larger algorithm.

## Partition-based sorting follows the data

Parallel quicksort partitions values around a pivot and recursively sorts the two sides. A balanced pivot exposes useful work, while a poor pivot can create one almost-empty branch and one nearly unchanged branch. Distributed quicksort must also move values to the ranks responsible for each partition. Choosing a pivot therefore affects both computation and communication. Randomization or sampling can reduce the chance of severe imbalance without eliminating every difficult input.

**Sample sort** first sorts local data, selects samples, chooses splitters, partitions each local array into buckets, exchanges destination buckets, and merges or sorts the received data. The splitters define global key ranges. Duplicate-heavy data can overload one bucket unless the algorithm defines how equal keys are distributed. Oversampling improves splitter quality at additional sample-processing cost. The all-to-all exchange can require a large temporary buffer, so peak memory belongs in the design.

**Bucket sort** is especially attractive when the key distribution or range is known. Bad distribution estimates can concentrate work in one bucket. **Radix sort** processes digits of integer or encoded keys. A pass counts digit frequencies, scans the counts to obtain offsets, and scatters items into output positions. A stable least-significant-digit pass preserves the ordering produced by earlier passes. Increasing the radix reduces pass count but increases histogram and offset storage.

## Parallel merging divides output positions

Two sorted arrays can be merged sequentially in linear work. A naive parallelization that gives each worker the same number of elements from each input can produce unequal output work because the keys may interleave unevenly. **Co-ranking** instead chooses a target output position $k$ and finds input prefixes of lengths $i$ and $j$ with $i+j=k$ whose elements exactly form that output prefix.

The boundary conditions compare the last included elements with the first excluded elements. A binary search finds a valid partition in logarithmic time. Workers compute boundaries for disjoint output intervals and merge only the corresponding input slices. Each worker then writes a distinct output region, avoiding locks. Tie-breaking rules determine stability when equal keys occur in both inputs.

Parallel selection finds an order statistic rather than fully sorting everything. Partitioning around candidate pivots can discard irrelevant key ranges. This can save work when only a median or top-k set is needed. A full sort is often the wrong baseline for a task that asks only for one threshold. The algorithm should match the output requirement before its implementation is parallelized.

## A phase trace reveals both progress and limits

The experiment sorts eight integers with odd-even transposition. Each phase uses disjoint adjacent comparisons, and the code records the array after every phase. The final comparison with Python's sorted result checks both ordering and multiset preservation for this small input. The simulation makes the phase structure visible but performs its comparisons sequentially in Python. It therefore demonstrates algorithmic parallelism rather than measured speedup.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
source=[7,2,6,3,8,1,5,4]
a=source.copy();phases=[]
for phase in range(len(a)):
 for i in range(phase%2,len(a)-1,2):
  if a[i]>a[i+1]:a[i],a[i+1]=a[i+1],a[i]
 phases.append(a.copy())
assert a==sorted(source)
print(json.dumps({'phases':phases,'sorted':a,'preserves_multiset':True},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "phases": [
    [
      2,
      7,
      3,
      6,
      1,
      8,
      4,
      5
    ],
    [
      2,
      3,
      7,
      1,
      6,
      4,
      8,
      5
    ],
    [
      2,
      3,
      1,
      7,
      4,
      6,
      5,
      8
    ],
    [
      2,
      1,
      3,
      4,
      7,
      5,
      6,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      7,
      6,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ]
  ],
  "sorted": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
  ],
  "preserves_multiset": true
}
```

The final array is **1 through 8 in increasing order**. The trace shows that one phase does not solve the entire problem even though several comparisons are independent. Each phase depends on the previous phase’s outputs, which creates the algorithm’s span.

**Try it.**

You can add repeated keys and label each key with its original index. Determine whether your comparison rule preserves stable order.

**Try it.**

You can change the input to descending order and compare the trace with an already sorted input. Explain why a fixed network can still perform the same comparisons.

## Summary

| Idea | What the lesson established |
|---|---|
| Networks | A fixed comparator graph exposes predictable stages. |
| Partition methods | Good splitters balance work and data movement. |
| Validation | Sorted order and multiset preservation are separate obligations. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Is adjacent sorted order enough to prove no values were lost?**

1. No.
2. Yes.

**Answer.** No. An algorithm can return an ordered but incomplete output.

**Question 2. What does a sorting network fix in advance?**

1. The comparator schedule.
2. The input values.

**Answer.** The comparator schedule. Its structure does not depend on comparison outcomes.

**Question 3. Why can sample sort become imbalanced?**

1. Splitters or duplicate handling can create uneven buckets.
2. Every bucket always has equal size.

**Answer.** Splitters or duplicate handling can create uneven buckets. Data distribution controls bucket sizes.

**Question 4. What does co-ranking partition?**

1. Positions in the merged output.
2. Only equal-sized input prefixes.

**Answer.** Positions in the merged output. Output boundaries define balanced merge tasks.

**Question 5. Does radix sort use only pairwise key comparisons?**

1. No, it uses digit structure.
2. Yes.

**Answer.** No, it uses digit structure. Its complexity model differs from comparison sorting.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 26. Graph algorithms, sparse matrices, and partitioning

Prepared for Arun Sharma. Codex drafted this study edition.

[25-sorting-merging](https://arunshar.github.io/numerical-parallel-computing/lectures/25-sorting-merging.html) showed how irregular data can unbalance a parallel algorithm. Graphs make that irregularity central. This lecture connects graph traversal, shortest paths, sparse matrix multiplication, and graph partitioning through one common question about which information must cross a partition boundary.

**Learning goals.**
- You can distinguish BFS, Dijkstra, Prim, and Floyd-Warshall.
- You can represent a sparse matrix in CSR form and compute a matrix-vector product.
- You can explain graph Laplacians, edge cuts, and load-balance constraints.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/26-graphs-sparse-partitioning.py`.

## A representation determines the work you touch

A graph has vertices and edges, possibly with direction and weights. An adjacency matrix stores a value for every possible pair, while an adjacency list stores only present neighbors. A dense matrix makes constant-time edge lookup simple but uses order $n^2$ space. A sparse representation uses space proportional to vertices plus stored edges. The best representation depends on density and the operations the algorithm performs.

A sparse matrix often represents a graph or a discretized operator. **COO** stores row indices, column indices, and values for each nonzero. **CSR** groups entries by row and stores a row-pointer array, column indices, and values. Row $i$ occupies the interval from `rowptr[i]` to `rowptr[i+1]`. The last row pointer equals the stored nonzero count. Empty rows have identical consecutive pointers and must still produce a valid output value.

For sparse matrix-vector multiplication, each output is $y_i=\sum_j A_{ij}x_j$ over stored entries. Assigning rows to workers gives exclusive ownership of output entries. Equal row counts can be poorly balanced when row lengths differ. Balancing nonzeros improves arithmetic balance but may split rows and require reductions. The access to x is indirect, so regular storage of A does not guarantee regular memory access overall.

## Traversals have different correctness rules

**Breadth-first search** explores an unweighted graph in distance layers. A frontier contains vertices discovered at the current layer. Expanding the frontier finds unvisited neighbors for the next layer. Multiple parents can discover the same vertex, so a parallel implementation needs a safe claim operation or a deduplication phase. A race-free visited flag is necessary, but the algorithm must also preserve the intended distance level.

A top-down BFS visits outgoing edges from the current frontier. A bottom-up step scans unvisited vertices and asks whether any neighbor belongs to the frontier. Bottom-up work can be attractive when the frontier is large, but it can waste work when the frontier is small. Direction-optimizing traversal switches based on a model or measured condition. Directed graphs require care because incoming and outgoing adjacency serve different roles.

**Dijkstra's algorithm** selects the unsettled vertex with minimum tentative distance and relaxes its edges. Its classical correctness argument requires nonnegative edge weights. Parallelizing the neighbor relaxations is straightforward, but selecting the next global minimum and maintaining ordering can limit scalability. Bucketed approaches such as delta-stepping relax the strict global order while preserving a different correctness argument. Arbitrary out-of-order relaxation is not automatically equivalent to Dijkstra.

**Prim's algorithm** also chooses a minimum candidate, but its state records the cheapest edge connecting the current tree to an outside vertex. Dijkstra's state records distance from a source. Replacing one update formula with the other changes the problem. A minimum spanning tree minimizes total tree-edge weight, while a shortest-path tree minimizes each root-to-vertex path distance. A graph can have very different trees for these two objectives.

## All-pairs and connectivity algorithms expose bulk phases

Floyd-Warshall considers each vertex $k$ as an allowed intermediate and updates $D_{ij}=\min(D_{ij},D_{ik}+D_{kj})$. For a fixed $k$, many pairs $(i,j)$ can be updated in parallel when the required pivot row and column are available. The phase order over $k$ remains important. Distributed layouts communicate the pivot information and can block the computation for locality.

The algorithm supports negative edge weights when negative cycles are absent. A negative diagonal after closure signals a negative cycle. **Transitive closure** uses a related recurrence with Boolean reachability and logical operations. **Connected components** can use label propagation, hooking, or union-find-based methods, but parallel updates need a consistent way to merge component identities. A graph algorithm's arithmetic may be simple while its ordering and synchronization are the difficult part.

## A graph partition balances work and communication

Partitioning assigns vertices or rows to workers. An **edge cut** counts edges crossing partition boundaries, possibly with weights. A balanced partition constrains the amount of work or storage assigned to each worker. Minimizing cuts without balance is trivial if every vertex goes to one worker, so the two requirements must be stated together. In sparse matrix-vector multiplication, a cut can estimate which x values need remote access.

The graph Laplacian is $L=D-A$ for an undirected weighted graph, where D contains weighted vertex degrees. Its quadratic form is a sum of weighted squared differences across edges. A vector with slowly varying values across strongly connected regions can reveal a useful division. Spectral partitioning often uses an eigenvector associated with the second-smallest Laplacian eigenvalue. The eigenvector is a heuristic guide to a constrained partitioning problem, not a guarantee of the best balanced cut.

Multilevel partitioning coarsens the graph, partitions a smaller representation, then refines the division while expanding it. Domain decomposition applies similar ideas to physical meshes. A subdomain's interior can be computed locally, while halo values cross its boundary. A compact partition often reduces boundary traffic, but anisotropic work or irregular connectivity can change the best shape.

## Sparse correctness should be checked against a dense reference

The experiment constructs a four-vertex path Laplacian, stores it in CSR form, and multiplies it by a small vector. A dense reference validates the CSR indexing. The row sums are zero, so multiplying by a constant vector gives zero. These checks exercise different invariants. The first checks storage and arithmetic, while the second checks the Laplacian construction.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
A=np.array([[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]],dtype=float)
rowptr=[0];cols=[];vals=[]
for row in A:
 for j,v in enumerate(row):
  if v!=0:cols.append(j);vals.append(v)
 rowptr.append(len(vals))
x=np.array([1,2,4,8.]);y=np.zeros(4)
for i in range(4):
 for k in range(rowptr[i],rowptr[i+1]):y[i]+=vals[k]*x[cols[k]]
assert np.allclose(y,A@x) and np.allclose(A@np.ones(4),0)
print(json.dumps({'rowptr':rowptr,'columns':cols,'y':y.tolist(),
                  'laplacian_constant_residual':float(np.linalg.norm(A@np.ones(4)))},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "rowptr": [
    0,
    2,
    5,
    8,
    10
  ],
  "columns": [
    0,
    1,
    0,
    1,
    2,
    1,
    2,
    3,
    2,
    3
  ],
  "y": [
    -1.0,
    -1.0,
    -2.0,
    4.0
  ],
  "laplacian_constant_residual": 0.0
}
```

The product is **[-1,-1,-2, 4]**, and the constant-vector residual is zero in this small exact construction. The experiment checks a sparse representation and its graph meaning. It does not measure distributed communication or solve the graph partitioning optimization problem.

**Try it.**

You can insert an empty row in a sparse matrix and verify that its CSR row pointers are equal. Check that the corresponding output is zero.

**Try it.**

You can partition the four-vertex path into {0, 1} and {2, 3}, then into {0, 2} and {1, 3}. Count cut edges and explain the communication implication.

## Summary

| Idea | What the lesson established |
|---|---|
| Graph algorithm | The state and ordering rules distinguish shortest paths, spanning trees, and traversal. |
| Sparse storage | CSR exposes row ownership but can hide irregular row cost. |
| Partitioning | A useful partition balances load while limiting cross-partition traffic. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Can classical Dijkstra safely assume negative edge weights?**

1. No.
2. Yes.

**Answer.** No. Its settled-distance argument requires nonnegative weights.

**Question 2. Does Prim optimize source-to-vertex distance?**

1. No, it builds a minimum spanning tree.
2. Yes.

**Answer.** No, it builds a minimum spanning tree. Its candidate value is a connecting-edge cost.

**Question 3. What does an empty CSR row look like?**

1. Two equal consecutive row pointers.
2. A missing row pointer.

**Answer.** Two equal consecutive row pointers. Every row needs an interval, including an empty interval.

**Question 4. What does a graph Laplacian do to a constant vector?**

1. It produces zero for this undirected construction.
2. It always doubles the vector.

**Answer.** It produces zero for this undirected construction. Each row sums to zero.

**Question 5. Why must a cut objective include balance?**

1. Otherwise all vertices can go to one worker.
2. Balance has no effect.

**Answer.** Otherwise all vertices can go to one worker. Communication alone admits a useless trivial placement.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 27. Parallel search and dynamic programming

Prepared for Arun Sharma. Codex drafted this study edition.

[26-graphs-sparse-partitioning](https://arunshar.github.io/numerical-parallel-computing/lectures/26-graphs-sparse-partitioning.html) used graph structure that is present before execution. Search often discovers its graph while running, and dynamic programming compresses repeated subproblems into a dependency structure. This lecture explains the different sources of parallelism and why search speedup can be difficult to interpret.

**Learning goals.**
- You can explain branch-and-bound pruning and parallel search overhead.
- You can identify wavefront parallelism in a dynamic program.
- You can distinguish monadic and polyadic dependencies in textbook terminology.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/27-search-dynamic-programming.py`.

## Search discovers work as it runs

A search state describes a partial candidate solution and the choices still available. A search tree connects a state to its extensions. Depth-first search uses a stack and explores one branch deeply before returning. Breadth-first search expands a level at a time, which exposes many tasks but can consume large memory. Best-first search prioritizes states using a score, often a lower bound or estimated remaining cost.

In a discrete optimization problem, the goal is to find the best feasible solution. **Branch and bound** maintains an incumbent feasible solution and a bound on the best possible result below each partial state. For minimization, a state whose lower bound is no better than the incumbent can be pruned. The bound must be valid. An optimistic-looking estimate that sometimes exceeds the true best descendant can remove the optimal solution and invalidate the algorithm.

Parallel workers can explore separate subtrees. A good incumbent found by one worker helps others prune more work, so sharing incumbents can improve total work. Delayed sharing can make workers explore branches that would soon become unnecessary. Extremely frequent sharing adds synchronization and communication. The correct update protocol must also protect the incumbent's objective and corresponding solution from becoming inconsistent.

## Search overhead can be positive or negative

Parallel search may expand a different set of states from serial search because it visits branches in a different order. A **search overhead factor** compares parallel expanded work with the work of a specified sequential search. A factor above one means more states were explored. A factor below one can occur when a parallel worker finds a strong incumbent early and enables pruning that the sequential order discovered much later.

This effect can produce apparent superlinear speedup without violating a fixed-work bound, because the work was not fixed. A fair report records both runtime and expanded states. It also specifies the tie-breaking rule, random seed, incumbent-sharing policy, and stopping condition. Comparing a carefully tuned parallel search with a deliberately poor serial search order would exaggerate the value of parallel execution.

Parallel depth-first search often uses work stealing. An idle worker requests a portion of another worker's pending search stack or a subtree root. Coarse transfers reduce communication overhead, while very coarse transfers can leave the final few workers imbalanced. Termination detection is harder than checking that one local queue is empty. Work can still be executing or traveling in messages. A correct distributed protocol must account for all possible sources of future work.

Parallel best-first search can use a shared priority queue or several local queues. A shared queue preserves a stronger global ordering but can become a contention bottleneck. Local queues relax that ordering and improve throughput at the cost of extra exploration. Algorithms such as A* also need their heuristic assumptions stated precisely. Admissibility and consistency have different roles, and a parallel implementation must preserve the correctness conditions of its chosen variant.

## Dynamic programming stores repeated subproblems

A **dynamic program** defines states, a recurrence, base cases, and an evaluation order. Memoization evaluates states on demand and caches results. Tabulation evaluates a planned order. Parallelism comes from states whose prerequisites are already complete. The fact that a table has many entries does not mean all entries can be filled simultaneously.

Edit distance gives a clear example. State $D_{ij}$ is the minimum cost of transforming the first $i$ symbols of one string into the first $j$ symbols of another. Its recurrence takes the minimum of deletion, insertion, and substitution candidates from $D_{i-1,j}$,$D_{i,j-1}$,and $D_{i-1,j-1}$. The first row and column encode edits involving an empty prefix. States on the same anti-diagonal have no direct dependency on one another, so an anti-diagonal forms a **wavefront**.

For an $m$-by $n$ table, the usual recurrence performs order $mn$ work and has order $m+n$ wavefront steps when cells have constant cost. A real implementation groups cells into tiles to amortize scheduling and communication. Tile dependencies follow the same north, west, and northwest pattern. The wavefront is short near the corners and widest near the middle, so available parallelism changes during execution.

## Textbook dependency classes describe different recurrences

In the textbook's terminology, a monadic formulation builds a candidate from one predecessor state, while a polyadic formulation combines multiple subproblem values in one candidate. Serial formulations use a staged dependency structure, while nonserial formulations admit dependencies that do not fit a simple sequence of fixed stages. These labels describe recurrence structure. They do not mean that a serially classified formulation has no parallel implementation.

Matrix-chain parenthesization is a useful polyadic example. To compute the best cost for a matrix interval, each candidate split adds the best cost of its left interval, the best cost of its right interval, and the scalar multiplication cost of combining them. All shorter intervals must be ready. Intervals of the same length can be computed together, while candidate splits within an interval can also be reduced in parallel. These are two different levels of parallelism.

A dependency graph can be denser than the original input representation. Recomputing values may reduce communication at the cost of extra arithmetic, while storing the entire table may exceed memory capacity. Checkpointing, banding, and divide-and-conquer reconstruction address different tradeoffs. A banded edit-distance algorithm assumes the relevant path remains within its band, so it must not be presented as an exact unrestricted solver without a valid bound.

## The wavefront experiment checks the recurrence

The experiment computes edit distance between two short strings by anti-diagonal order and verifies a known result. Every cell is updated only after its prerequisites. The implementation runs sequential Python loops to expose the schedule, so it does not measure parallel execution. Its purpose is to provide a correct reference that can later support a tiled OpenMP or MPI version.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
s,t='kitten','sitting'
m,n=len(s),len(t)
d=[[0]*(n+1) for _ in range(m+1)]
for i in range(m+1):d[i][0]=i
for j in range(n+1):d[0][j]=j
widths=[]
for diagonal in range(2,m+n+1):
 cells=[]
 for i in range(1,m+1):
  j=diagonal-i
  if 1<=j<=n:
   d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(s[i-1]!=t[j-1]))
   cells.append((i,j))
 widths.append(len(cells))
assert d[m][n]==3
print(json.dumps({'edit_distance':d[m][n],'wavefront_widths':widths,
                  'cells_updated':sum(widths)},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "edit_distance": 3,
  "wavefront_widths": [
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    5,
    4,
    3,
    2,
    1
  ],
  "cells_updated": 42
}
```

The computed edit distance is **3**, and the program updates 42 non-boundary cells. The wavefront widths grow and then shrink, showing why a fixed processor team cannot stay equally busy throughout this small problem. The result validates the recurrence and schedule, not a parallel speedup.

**Try it.**

You can change one input to the empty string and verify the base cases. Explain why no interior wavefront is needed.

**Try it.**

You can sketch a branch-and-bound search with two branches where the second branch contains a strong incumbent. Compare work when serial and parallel searches discover it at different times.

## Summary

| Idea | What the lesson established |
|---|---|
| Search | Execution order can change the amount of explored work. |
| Pruning | A valid bound is a correctness requirement. |
| Dynamic programming | The recurrence determines which states can run together. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Can an invalid lower bound prune the optimum?**

1. Yes.
2. No.

**Answer.** Yes. A bound that overestimates the best descendant can remove a necessary branch.

**Question 2. Does an empty local queue prove global termination?**

1. No.
2. Yes.

**Answer.** No. Other workers or messages can still create work.

**Question 3. Which edit-distance cells can share a wavefront?**

1. Cells on one anti-diagonal.
2. All cells in the table.

**Answer.** Cells on one anti-diagonal. Their required predecessors lie on earlier anti-diagonals.

**Question 4. What is the computed distance from kitten to sitting?**

1. 3.
2. 1.

**Answer.** 3. Three unit-cost edits suffice and are necessary.

**Question 5. Can parallel search execute less work than a serial search?**

1. Yes, a different discovery order can improve pruning.
2. No, the explored tree is always fixed.

**Answer.** Yes, a different discovery order can improve pruning. An earlier incumbent can eliminate additional subtrees.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 28. Parallel Fourier transforms and data redistribution

Prepared for Arun Sharma. Codex drafted this study edition.

[27-search-dynamic-programming](https://arunshar.github.io/numerical-parallel-computing/lectures/27-search-dynamic-programming.html) exposed parallelism by following a recurrence. The fast Fourier transform also follows a structured recurrence, but its data exchanges form regular butterfly patterns. This lecture connects the numerical FFT to binary-exchange and transpose-based parallel algorithms.

**Learning goals.**
- You can derive an FFT butterfly from even and odd input subsequences.
- You can identify which butterfly stages require communication under a block layout.
- You can explain the role of transposes in distributed multidimensional transforms.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/28-parallel-fft.py`.

## The transform separates a signal into frequencies

The discrete Fourier transform of $n$ complex samples is $X_k=\sum_{j=0}^{n-1}x_j\exp(-2\pi i jk/n)$. The sign and normalization are conventions that must match the inverse transform. A direct implementation evaluates $n$ sums of $n$ terms and performs order $n^2$ arithmetic. The FFT is a family of algorithms that computes the same transform more efficiently by reusing structure in the roots of unity.

For even $n$, split the input into even and odd indices. Their transforms E and O have length $n/2$. The output satisfies $X_k=E_k+\omega_n^kO_k$ and $X_{k+n/2}=E_k-\omega_n^kO_k$, where $\omega_n=\exp(-2\pi i/n)$. This pair of updates is a **butterfly**. Recursing to small transforms produces order $n\log n$ work. The recursion changes the evaluation order, so a floating-point FFT need not match direct summation bit for bit.

Radix-2 assumes a power-of-two factorization. Mixed-radix methods use other factors, and algorithms such as Bluestein handle awkward lengths through convolution. Padding a signal to a convenient length changes the sampled frequency grid and can change the application semantics. Padding is appropriate for some convolution calculations when enough zeros are added, but it is not a universal way to replace an arbitrary transform with an equivalent one.

## An iterative FFT has stages with different partners

An iterative radix-2 implementation commonly begins with a bit-reversal permutation and then doubles the butterfly width at each stage. For eight samples, the stages combine groups of 2, 4, and 8. Within one stage, butterflies are independent. Between stages, outputs become inputs for the next stage. A stage therefore offers data parallelism, while the stage sequence creates the span.

Bit reversal maps the binary representation of an index to its reversed bit order. With three bits, index 3 is 011 and maps to 110, which is 6. The permutation can be performed explicitly or absorbed into another layout convention. An implementation must document whether its input or output is in natural order. A numerically plausible but permuted spectrum is still an incorrect result for a caller expecting natural order.

Twiddle factors are the roots of unity used by each butterfly. Precomputing them saves repeated trigonometric evaluation but consumes memory. Recurrence-based generation can reduce storage while accumulating additional rounding error. Libraries choose among many decompositions and small kernels because the best arithmetic count alone does not determine cache behavior or vectorization efficiency.

## Binary exchange follows the butterfly graph

Suppose $n$ samples are distributed in contiguous blocks across $p=2^q$ ranks, with $n/p$ samples per rank. Some butterfly stages pair values inside one rank's block, while later stages pair values on different ranks. A binary-exchange algorithm communicates with partners whose rank identifiers differ in one selected bit. The pattern mirrors a hypercube even when the physical network is not a hypercube.

For $n=16$ and $p=4$, each rank owns four consecutive positions in the simplified layout. A butterfly pairing indices that differ in bit 0 or bit 1 remains local. Pairings that differ in bit 2 or bit 3 cross rank boundaries. The implementation may choose a different input permutation or cyclic layout, which changes which stages are local. The important method is to derive ownership for both endpoints of every butterfly rather than memorize a fixed stage rule.

Each rank performs order $(n/p)\log n$ arithmetic. A simple block-based exchange model has order $\log p$ communicating stages with order $n/p$ data per stage. This gives a useful first approximation, but exact volume depends on packing, duplication, and the chosen FFT decomposition. Network contention and the cost of permutations can dominate at high process counts.

## Transpose algorithms move whole pencils of data

A multidimensional transform can be computed through one-dimensional transforms along each axis. A rank must have the relevant line of data locally or communicate during that transform. A **slab decomposition** distributes whole planes or thick slabs. It can limit the number of useful ranks when one dimension is small. A **pencil decomposition** distributes thinner two-dimensional partitions and supports a larger process grid, but it needs more redistribution stages.

A distributed transpose changes which dimension is contiguous and locally owned. The application first performs local transforms along a resident axis, redistributes data, then performs transforms along the next axis. All-to-all or subgroup all-to-all operations often implement that redistribution. Packing into destination-contiguous buffers can improve transfer efficiency but adds memory traffic and temporary storage.

A transpose-based one-dimensional FFT can reinterpret a long transform through a matrix-shaped decomposition, local transforms, twiddle multiplication, and transposes. The algebra must account for the chosen index factorization. Simply taking a matrix FFT without the required twiddle terms does not reproduce an arbitrary long one-dimensional DFT. The decomposition is valuable because bulk transposes can replace many fine-grained exchanges.

## Transform applications impose extra checks

Circular convolution corresponds to pointwise multiplication in the Fourier domain. Linear convolution requires enough zero padding to prevent wraparound. Real-input transforms exploit conjugate symmetry and store fewer complex coefficients, but the packed layout must be understood before combining it with another kernel. Frequency bins also depend on sample spacing, so a peak index alone is not a physical frequency.

The experiment compares an original radix-2 implementation with NumPy's FFT for eight samples and checks the inverse reconstruction. A separate ownership count classifies butterfly pairs as local or remote for a four-rank block layout. These are algorithmic checks. The script does not call distributed FFT software or measure interconnect traffic.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
def fft(x):
 x=np.asarray(x,dtype=complex);n=len(x)
 if n==1:return x.copy()
 assert n%2==0
 e,o=fft(x[::2]),fft(x[1::2])
 w=np.exp(-2j*np.pi*np.arange(n//2)/n)*o
 return np.concatenate([e+w,e-w])
x=np.array([1,2,0,-1,3,0,2,1.])
y=fft(x)
error=float(np.max(np.abs(y-np.fft.fft(x))))
assert error<1e-12
n,p=16,4;local=n//p;stages=[]
for bit in range(4):
 pairs=[(i,i^(1<<bit)) for i in range(n) if i<(i^(1<<bit))]
 remote=sum(a//local!=b//local for a,b in pairs)
 stages.append({'partner_bit':bit,'remote_pairs':remote})
assert [s['remote_pairs'] for s in stages]==[0,0,8,8]
print(json.dumps({'max_reference_error':error,'reconstruction_ok':bool(np.allclose(np.fft.ifft(y),x)),
                  'ownership_stages':stages},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "max_reference_error": 6.280369834735101e-16,
  "reconstruction_ok": true,
  "ownership_stages": [
    {
      "partner_bit": 0,
      "remote_pairs": 0
    },
    {
      "partner_bit": 1,
      "remote_pairs": 0
    },
    {
      "partner_bit": 2,
      "remote_pairs": 8
    },
    {
      "partner_bit": 3,
      "remote_pairs": 8
    }
  ]
}
```

The reference error stays below **1e-12** for this small example, and the inverse reconstructs the input within floating-point tolerance. The ownership model has zero remote pairs in the first two partner-bit stages and eight in each of the next two. That count depends on the stated contiguous-block layout.

**Try it.**

You can change the ownership rule to cyclic assignment `owner = index % p`. Predict which partner bits become remote and verify the count.

**Try it.**

You can convolve two length-four signals directly and through FFTs. Compare transform lengths 4 and 8 and explain why only adequate padding gives the full linear convolution.

## Summary

| Idea | What the lesson established |
|---|---|
| FFT structure | Even-odd decomposition reuses smaller transforms through butterflies. |
| Distributed ownership | A butterfly requires communication when its endpoints have different owners. |
| Transpose design | Bulk redistribution can make the next transform axis local. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What does the FFT compute?**

1. The same DFT through a faster factorization.
2. A different approximate frequency definition.

**Answer.** The same DFT through a faster factorization. Rounding differs, but the mathematical transform is the same.

**Question 2. What is radix-2 FFT work for n samples?**

1. Order n log n.
2. Order n cubed.

**Answer.** Order n log n. There are logarithmically many stages with linear work per stage.

**Question 3. Why can an FFT need a distributed transpose?**

1. The next transform axis may be split across ranks.
2. A transpose changes addition into multiplication.

**Answer.** The next transform axis may be split across ranks. Redistribution changes local ownership.

**Question 4. Does zero padding always preserve the original transform samples?**

1. No, it changes the frequency grid.
2. Yes.

**Answer.** No, it changes the frequency grid. Padding has application-specific meaning.

**Question 5. What is the modeled remote-pair count in partner bit 2?**

1. 8.
2. 0.

**Answer.** 8. Bit 2 changes the owner under four-element contiguous blocks.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 29. Hybrid programming and an introduction to quantum computation

Prepared for Arun Sharma. Codex drafted this study edition.

[28-parallel-fft](https://arunshar.github.io/numerical-parallel-computing/lectures/28-parallel-fft.html) combined local arithmetic with distributed communication. Hybrid programs deliberately combine several execution models. Saad’s final notes also introduce quantum computation, which uses a different mathematical state model. This lecture treats those topics separately so that ordinary parallel speedup is not confused with quantum interference.

**Learning goals.**
- You can explain MPI thread-support levels and a safe hybrid design.
- You can describe CPU-GPU communication ownership and completion.
- You can calculate a small quantum-state evolution and explain the Deutsch-Jozsa promise.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/29-hybrid-quantum.py`.

## Hybrid programs use different models at different scales

A common cluster design uses one or a few MPI ranks per node and several OpenMP threads inside each rank. MPI handles data exchange between address spaces, while threads share local arrays. This can reduce the number of MPI endpoints and copies of rank-private data. It can also introduce thread synchronization, NUMA placement issues, and competition between communication progress and computation. More software layers do not automatically improve performance.

The application should decide which level owns each loop and each data partition. A node might own one spatial subdomain, with threads splitting its interior. The rank exchanges halo data after the required local computation. If a library also creates threads, the total concurrency can exceed available cores. A configuration of four ranks times four threads means up to 16 application threads before counting runtime helpers. Resource accounting must include nested libraries and progress threads.

`MPI_Init_thread` requests a thread-support level and returns the level actually provided. `MPI_THREAD_SINGLE` assumes one thread. `MPI_THREAD_FUNNELED` permits multiple threads but requires the thread that initialized MPI to make MPI calls. `MPI_THREAD_SERIALIZED` permits calls from multiple threads as long as they do not overlap. `MPI_THREAD_MULTIPLE` permits concurrent MPI calls, subject to MPI's remaining rules. A program must inspect the provided level and choose a compatible path.

A funnelled design often keeps MPI calls outside OpenMP parallel regions or uses a carefully controlled main-thread section. A serialized design protects MPI calls with a protocol that prevents overlap. Multiple support can simplify some designs, but its overhead and progress behavior depend on the implementation. It should be requested because the algorithm needs it rather than assumed to be free.

## GPU participation adds another completion boundary

An MPI-plus-CUDA program can compute a local subproblem on a GPU and exchange boundary data through MPI. A host-staging design copies data from device to host, performs MPI communication, then copies incoming data to the device. A CUDA-aware MPI implementation may accept device pointers directly for supported operations. This removes some explicit staging code, but it does not imply that every datatype, stream interaction, or transport path is supported equally.

CUDA kernels and copies can be asynchronous with respect to the host. MPI must not read a device send buffer before the producer operation is complete under the supported interoperability contract. A GPU consumer must likewise wait until incoming data is ready. Events, stream synchronization, and library-specific interoperability features help express these dependencies. A successful run with an implicit synchronization does not prove that a later asynchronous version is safe.

The hybrid performance model should separate local CPU work, GPU work, transfers, communication, and synchronization. Overlap can reduce the critical path when operations are independent and the hardware supports concurrent progress. Adding their isolated runtimes gives a serialized estimate. Taking their maximum gives an ideal overlap estimate when stages can progress independently without resource contention. Neither is a measured pipeline without a timeline or elapsed-time evidence.

## A qubit is a normalized complex state

A **qubit** has a state $\alpha|0\rangle+\beta|1\rangle$ with $|\alpha|^2+|\beta|^2=1$. Measurement in the computational basis returns 0 or 1 with those respective probabilities. The amplitudes can have phases, and those phases influence later interference. A global phase does not change measurement probabilities, while relative phase can change the result of a later gate.

An $n$-qubit pure state has $2^n$ complex amplitudes in a classical state-vector representation. This does not let a user read all those amplitudes from one measurement. Measurement returns an outcome with $n$ classical bits. A quantum algorithm must arrange interference so that the measurement distribution reveals useful information. Calling the state a collection of simultaneously readable classical answers is misleading.

A quantum gate acts through a unitary transformation in the ideal circuit model. The X gate exchanges the 0 and 1 amplitudes. The Hadamard gate maps $|0\rangle$ to an equal superposition and maps relative phase into different outcomes when applied again. A controlled-NOT flips its target conditional on its control. Applying H to one qubit followed by controlled-NOT can create an entangled Bell state. Entanglement means the joint state cannot be factored into independent single-qubit states.

## Deutsch-Jozsa relies on a promised oracle

The Deutsch-Jozsa problem assumes a Boolean function is promised to be either constant or balanced. Balanced means it returns one on exactly half its inputs. The algorithm does not solve arbitrary function classification outside that promise. A phase-oracle form maps $|x\rangle$ to $(-1)^{f(x)}|x\rangle$. Hadamard gates prepare a uniform superposition before the oracle and transform it again afterward.

The amplitude of the all-zero output after the final Hadamards is proportional to $\sum_x(-1)^{f(x)}$. A constant function makes all terms have the same sign, while a balanced function makes them cancel. In the ideal promised setting, measurement distinguishes these cases with one oracle query. The query model treats the oracle as available. Building the oracle and implementing error correction are separate costs, so this result does not prove an exponential end-to-end speedup for arbitrary real applications.

The experiment constructs the ideal state vector for two input qubits and evaluates a constant function and a balanced parity function. It also creates a Bell state and prints its probabilities. This is classical simulation of four amplitudes. Its cost grows exponentially with the number of simulated qubits, and it does not demonstrate quantum hardware performance. The value of the exercise is to expose normalization, phase, and interference with numbers you can inspect.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
H=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
H2=np.kron(H,H)
start=np.array([1,0,0,0],dtype=complex)
results={}
for name,phases in [('constant',[1,1,1,1]),('balanced',[1,-1,-1,1])]:
 state=H2@(np.array(phases)*(H2@start))
 probs=np.abs(state)**2
 assert np.isclose(probs.sum(),1)
 results[name]=np.round(probs,12).tolist()
bell=np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
assert np.isclose(results['constant'][0],1) and np.isclose(results['balanced'][0],0)
print(json.dumps({'deutsch_jozsa_probabilities':results,
                  'bell_probabilities':(np.abs(bell)**2).tolist()},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "deutsch_jozsa_probabilities": {
    "constant": [
      1.0,
      0.0,
      0.0,
      0.0
    ],
    "balanced": [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  },
  "bell_probabilities": [
    0.4999999999999999,
    0.0,
    0.0,
    0.4999999999999999
  ]
}
```

The constant oracle gives all-zero output probability **1**, while the balanced parity oracle gives probability **0** for that output. The Bell state gives equal probability to 00 and 11. These results follow from ideal unitary arithmetic in a tiny classical simulator. They are not observations from a quantum processor.

**Try it.**

You can replace the balanced oracle by another balanced truth table. Verify that the all-zero probability stays zero even when the other output probabilities change.

**Try it.**

You can draw the dependency chain for a GPU-produced send buffer, MPI exchange, and GPU consumer. Mark which event or completion check protects each transition.

## Summary

| Idea | What the lesson established |
|---|---|
| Hybrid execution | Each model needs explicit ownership and completion boundaries. |
| Quantum state | Amplitudes determine probabilities through interference and measurement. |
| Quantum claim | A query-model advantage does not establish an arbitrary application speedup. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. May a funnelled MPI program call MPI from any worker thread?**

1. No, only the initializing thread may call MPI.
2. Yes.

**Answer.** No, only the initializing thread may call MPI. MPI_THREAD_FUNNELED has a specific calling-thread restriction.

**Question 2. Does requesting MPI_THREAD_MULTIPLE guarantee it was provided?**

1. No, the returned level must be checked.
2. Yes.

**Answer.** No, the returned level must be checked. The implementation reports the available support.

**Question 3. Can one measurement reveal all 2^n amplitudes?**

1. No.
2. Yes.

**Answer.** No. Measurement yields an outcome rather than a complete amplitude table.

**Question 4. What promise does Deutsch-Jozsa require?**

1. The function is constant or balanced.
2. The function is any real-valued program.

**Answer.** The function is constant or balanced. The distinction depends on the promised Boolean-function class.

**Question 5. What creates the balanced-case zero amplitude?**

1. Cancellation of positive and negative phase contributions.
2. Reading every branch as a classical answer.

**Answer.** Cancellation of positive and negative phase contributions. Interference produces the useful measurement pattern.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts. Saad’s sets 18 and 19 provide the hybrid-programming and quantum-computing source topics.


---

# 30. From course exercises to a reproducible parallel project

Prepared for Arun Sharma. Codex drafted this study edition.

[29-hybrid-quantum](https://arunshar.github.io/numerical-parallel-computing/lectures/29-hybrid-quantum.html) separated several execution models and their evidence limits. This lecture combines the numerical and parallel habits into a practical project workflow. The goal is a result that another reader can rebuild, check, and improve without guessing which algorithm, input, or measurement produced it.

**Learning goals.**
- You can turn an assignment into a testable problem contract.
- You can design correctness tests and a performance experiment separately.
- You can prepare a source-linked handoff that distinguishes tested and untested behavior.

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/30-course-project-workflow.py`.

## A problem contract prevents accidental scope changes

A problem contract defines inputs, outputs, valid shapes, data types, boundary behavior, tolerances, and resource limits. A matrix multiplication routine may accept rectangular matrices and arbitrary sizes, or it may assume fixed 4-by 4 tiles. These are different contracts. A kernel that works on one square size should not be advertised as a general matrix multiplication implementation. A tutorial can simplify a contract, but it must say exactly what it simplifies.

The contract also identifies the mathematical operation. Pairwise batched matrix products followed by a sum differ from one elementwise product or one ordinary matrix product. A convolution needs a definition of padding, stride, filter orientation, and output shape. A histogram needs a bin rule and out-of-range behavior. A reduction needs an operation and an identity. Many parallel bugs begin as disagreements about these details rather than mistakes in thread syntax.

An input generator should create ordinary cases and cases near the contract boundaries. A matrix size that is not divisible by a tile width tests edge handling. An empty partition tests collective counts and pointer logic. Repeated keys test sorting stability or bucket handling. A graph with an isolated vertex tests traversal initialization. A floating-point cancellation example tests whether the chosen tolerance reflects the operation's conditioning.

## A serial reference gives a clear comparison

A small serial implementation is often the easiest correctness oracle. It should express the mathematical definition directly and use an independent implementation path when practical. Comparing a parallel kernel with the same buggy indexing code executed serially is a weak check. NumPy or another established library can provide a reference for suitable operations, but you must still match data types, normalization, and layout conventions.

Exact integer results permit exact comparison when overflow is excluded. Floating-point comparison usually needs absolute and relative tolerances. A common criterion is $|x-y|\leq a+r|y|$, where $a$ protects near-zero values and $r$ scales with the reference magnitude. The tolerances should follow the data type, accumulation length, conditioning, and application requirements. A large tolerance chosen after a failure can hide a real algorithmic error.

Invariant checks complement a reference. A sort preserves the multiset, a probability vector sums to one, a Laplacian annihilates constants, and a partition covers every item exactly once. Metamorphic tests transform an input in a way that predicts a related output. For example, linearity predicts how a matrix-vector product changes when the input is scaled. Such checks can reveal errors on cases where a full reference is expensive.

## A performance hypothesis should name its mechanism

A useful hypothesis says that a specific change reduces a specific cost. Blocking may reduce memory traffic. A reduction may replace repeated atomic updates with private accumulation. A nonblocking exchange may overlap communication with interior computation. A new process grid may reduce per-rank communication volume. Each hypothesis suggests an observable quantity beyond total runtime, which makes a negative result informative.

The baseline should be correct and reasonably configured. A speedup over an intentionally poor implementation can teach one mechanism, but it should be labeled as that comparison. A strong application claim needs a relevant baseline, such as an optimized library or a well-configured previous method. All compared versions should use the same mathematical contract and a justified precision setting.

A benchmark separates setup, transfer, computation, and cleanup when that separation serves the question. An end-to-end measurement should include the costs a real caller pays. A kernel-only measurement is useful for kernel analysis but should not be relabeled as application latency. Warmup, repeated trials, and synchronization boundaries affect results. GPU timing needs completion-aware events or a documented synchronization method because host launch time is not kernel runtime.

## The CSCI 5451 homework sequence teaches several mechanisms

The early shared-memory assignment develops thread-level decomposition and testing. The distributed matrix assignments add communication, process-grid indexing, general dimensions, and load balance. The CUDA assignments then emphasize memory hierarchy, convolution, and small-matrix reuse. These connections explain why a correct serial model remains useful across the whole course. The same reference can validate several execution strategies without pretending their runtime costs are identical.

This study repository uses new toy examples and original code to teach those mechanisms. The instructor's assignment statements, starter files, and grading rules remain at their original links. A study example is not automatically a drop-in homework submission. The actual assignment can restrict libraries, precision, file edits, or execution environment, and those constraints belong to its own contract.

The team project should choose a bounded operation with a meaningful baseline and enough data to expose a bottleneck. A good first milestone reproduces correctness on small inputs. The second milestone establishes a serial or simple parallel baseline. The third tests one optimization hypothesis. A final extension can explore scale or a second hardware configuration after the core result is stable. This sequence makes a useful result possible even when an ambitious final optimization does not work.

## A handoff preserves evidence and uncertainty

A reproducible repository records exact commands, dependency versions, input-generation seeds, source hashes, and raw outputs. A result table should point to the log or data file that supports it. A machine description should name the relevant processor or accelerator, process count, and software stack without exposing credentials or private infrastructure details. Results from a CPU model must remain labeled as models when a CUDA implementation has not run.

A refinement handoff should list completed sections, known mathematical or implementation limitations, untested platforms, and the next concrete checks. It should separate source-derived facts from original explanations and measured results. A reviewer can then improve a chapter without accidentally strengthening a claim. A green documentation build proves that the site renders under the tested toolchain. It does not prove that every algorithm is correct or every explanation is exhaustive.

The experiment below demonstrates an evidence record for a deterministic local result. It computes a blocked matrix product, verifies it against a reference, and records dimensions, data type, maximum error, and a content hash of the inputs. It intentionally contains no speedup field because it does not measure runtime. Adding a runtime later should require an actual benchmark record rather than an estimated number.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, hashlib
import numpy as np
rng=np.random.default_rng(5451)
A=rng.normal(size=(7,5));B=rng.normal(size=(5,6));C=np.zeros((7,6))
tile=3
for i in range(0,7,tile):
 for j in range(0,6,tile):
  for k in range(0,5,tile):
   C[i:i+tile,j:j+tile]+=A[i:i+tile,k:k+tile]@B[k:k+tile,j:j+tile]
err=float(np.max(np.abs(C-A@B)))
assert err<1e-12
print(json.dumps({'seed':5451,'A_shape':list(A.shape),'B_shape':list(B.shape),
 'dtype':str(A.dtype),'tile':tile,'max_abs_error':err,
 'input_sha256':hashlib.sha256(A.tobytes()+B.tobytes()).hexdigest(),
 'runtime_measured':False,'claim':'The blocked result matches the reference within1e-12.'},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "seed": 5451,
  "A_shape": [
    7,
    5
  ],
  "B_shape": [
    5,
    6
  ],
  "dtype": "float64",
  "tile": 3,
  "max_abs_error": 8.881784197001252e-16,
  "input_sha256": "5b80f8064e32ef337056be992e149bf709cc9da22481df492edc98a461d8d420",
  "runtime_measured": false,
  "claim": "The blocked result matches the reference within1e-12."
}
```

The 7-by 5 and 5-by 6 inputs force partial tiles, and the blocked result matches the reference within **1e-12**. The saved record describes exactly what was checked. Its `runtime_measured` field is false, so it supports a correctness claim and no performance claim.

**Try it.**

You can change the tile width to 1, 2, 4,and 8. Verify the same contract across widths and explain why partial tiles are an important test.

**Try it.**

You can design a timing record with separate setup and compute intervals. State which interval answers your project question before collecting any numbers.

## Summary

| Idea | What the lesson established |
|---|---|
| Contract | Inputs, outputs, precision, and boundary behavior define the actual problem. |
| Evidence | Correctness and performance need different checks and records. |
| Handoff | A useful refinement packet preserves both results and unresolved limits. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does a documentation build prove algorithmic correctness?**

1. No.
2. Yes.

**Answer.** No. Rendering checks and algorithm checks operate at different layers.

**Question 2. Why test dimensions that are not multiples of the tile width?**

1. They exercise partial boundary tiles.
2. They guarantee maximum speedup.

**Answer.** They exercise partial boundary tiles. Boundary indexing often differs from the full-tile path.

**Question 3. Can a CPU model establish CUDA throughput?**

1. No.
2. Yes.

**Answer.** No. The model does not execute GPU hardware.

**Question 4. What should support a measured speedup?**

1. Saved runtimes with a specified baseline and setup.
2. An expected ratio from the algorithm name.

**Answer.** Saved runtimes with a specified baseline and setup. A measurement claim requires measurement evidence.

**Question 5. What does the example's false runtime_measured field mean?**

1. The record supports correctness but no timing claim.
2. The algorithm failed.

**Answer.** The record supports correctness but no timing claim. The experiment deliberately validates arithmetic without benchmarking.


## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.


---

# 31. CUDA execution, threads, and multidimensional data

Prepared for Arun Sharma. Codex drafted this study edition.

This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will map one output to one thread without writing outside an array.
- You will separate host allocation, device allocation, transfers, and execution.
- You will explain why parallel work can still produce a slow application. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/31_cuda_execution.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A kernel gives many workers the same recipe

A **kernel** is a function that many GPU threads execute. Each thread receives an identity and uses it to choose data. A vector-addition kernel has a simple ownership rule. Thread $i$ writes $c_i=a_i+b_i$. The addition is easy. The important design work establishes that every output has exactly one writer and that every read addresses a valid element.

A **host** is the CPU-side program that prepares work. A **device** is the GPU that executes kernels. An application typically allocates host arrays, allocates device arrays, copies inputs, launches a kernel, copies results back, and releases resources. A kernel launch normally returns before the GPU finishes. A later synchronization or a suitable blocking transfer establishes when the host can consume the result. Error checks must cover both launch configuration errors and errors that surface only after execution.

The CPU and GPU have different strengths. A CPU devotes substantial resources to low-latency execution of a small number of instruction streams. A GPU devotes resources to high-throughput execution of many ready threads. A complicated branch-heavy task with very little data can favor the CPU. A large regular calculation with enough independent work can favor the GPU. Neither statement eliminates the need to measure the complete application.

## The hierarchy defines coordinates

CUDA organizes threads into **blocks**, and blocks into a **grid**. A one-dimensional launch computes the global index as

$$i=bB+t,$$

where $b$ is the block index, $B$ is the number of threads per block, and $t$ is the thread index within its block. The grid requires $G=\lceil N/B\rceil$ blocks to cover $N$ elements. The final block can contain threads without valid elements. Those threads must skip array accesses using the condition $i<N$.

A ten-element array with four threads per block needs three blocks. The launch creates twelve threads, but only ten may read or write the arrays. A guard is part of the algorithm, not a performance afterthought. Testing only array lengths divisible by the block size hides this error.

Two-dimensional data needs two coordinates. A row-major image maps pixel $(r,c)$ to address $rW+c$, where $W$ is the row width. A two-dimensional block computes the row from its y coordinates and the column from its x coordinates. The x dimension usually follows adjacent columns because adjacent threads then access nearby words. A pitched allocation can add padding between rows. Its pitch is measured in bytes, so a kernel must not silently substitute the logical width for the physical row stride.

## Ownership determines the first safe implementation

A grayscale conversion gives one thread one output pixel. The thread combines the input color channels with fixed weights. An image blur gives one thread one output pixel but reads a neighborhood. A matrix product gives one thread one output element and makes that thread compute a dot product. These mappings differ in reuse, but all begin with the same clear writer rule.

A first implementation should make dependencies obvious. A thread can read immutable inputs without synchronization. Threads cannot safely read outputs that another block may not have produced yet. Ordinary blocks may run in any order, and only some blocks may be resident simultaneously. A grid-wide software barrier built from spinning blocks can deadlock when waiting blocks occupy the resources needed by blocks that have not started.

The next optimization follows the data. A blur rereads neighboring pixels, and a matrix product rereads rows and columns. Shared-memory tiling can reduce repeated global loads, but it introduces cooperation and barriers. A correct independent-output kernel therefore provides a useful reference before cooperation makes debugging harder.

## Transfers can dominate the work

Suppose a toy operation reads two float32 arrays and writes one float32 array. It moves at least $12N$ bytes within the device memory system when caching and extra traffic are ignored. A host-driven implementation also transfers inputs to the device and results back. A tiny kernel can finish quickly while its launch and transfers dominate the total duration.

An application can often keep intermediate arrays on the GPU across several operations. That change removes repeated transfer costs without changing arithmetic. Kernel fusion can also remove intermediate arrays, but larger kernels may use more registers or reduce scheduling flexibility. A useful performance report therefore records both kernel time and end-to-end time, including allocation and transfers when those costs belong to the application.

The reference experiment enumerates the launched thread identities and checks that the guard gives exact coverage. It also verifies a rectangular row-major mapping. This experiment proves the indexing rule for the chosen cases. It does not exercise a CUDA runtime or model GPU scheduling.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [31_cuda_execution.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/31_cuda_execution.py).

```python
import numpy as np
n, block = 10, 4
grid = (n + block - 1) // block
ids = [b * block + t for b in range(grid) for t in range(block)]
valid = [i for i in ids if i < n]
a = np.arange(n, dtype=np.float32)
b = np.full(n, 3, dtype=np.float32)
c = np.empty_like(a)
for i in valid:
    c[i] = a[i] + b[i]
assert valid == list(range(n))
assert np.array_equal(c, a + b)
height, width = 3, 5
addresses = [r * width + col for r in range(height) for col in range(width)]
assert addresses == list(range(height * width))
print(f"elements={n}, blocks={grid}, launched_threads={len(ids)}")
print(f"valid_threads={len(valid)}, skipped_threads={len(ids)-len(valid)}")
print("vector_sum=", c.astype(int).tolist())
print(f"rectangular_address_count={len(addresses)}")
```

The recorded run produced the following output.

```text
elements=10, blocks=3, launched_threads=12
valid_threads=10, skipped_threads=2
vector_sum= [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
rectangular_address_count=15
```

**The launch needs twelve threads to cover ten elements.** The two extra threads are harmless because the guard excludes them. The rectangular check confirms that fifteen distinct coordinates map to fifteen consecutive addresses.

**Try it 1 :class: exercise.**

Set the array length to 1, 4, 5, and 17. Predict the grid size and number of skipped threads before each run. Explain why a zero-length array needs separate host-side launch handling. :::

**Try it 2 :class: exercise.**

Replace the rectangular address formula with `col * height + r`. Explain why this remains a valid permutation but changes the layout interpretation. Identify what goes wrong if the producer uses one convention and the consumer uses the other. :::

## Summary

| Decision | What you should remember |
|---|---|
| Thread mapping | Each valid output should have a clear owner. |
| Bounds | A ceiling-divided launch requires a bounds guard. |
| Application cost | Transfers and launches can dominate a small kernel. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. How many blocks cover ten elements with four threads per block?**

1. Two blocks cover them.
2. Three blocks cover them.
3. Ten blocks cover them.

**Answer.** Three blocks cover them. The ceiling of ten divided by four is three.

**Question 2. What does thread i own in the reference vector addition?**

1. It owns output c[i].
2. It owns every output.
3. It owns a whole block.

**Answer.** It owns output c[i]. The one-writer rule assigns exactly one output to each valid thread.

**Question 3. Can ordinary blocks rely on their execution order?**

1. Yes, block indices determine time order.
2. No, their execution order is not guaranteed.

**Answer.** No, their execution order is not guaranteed. The scheduler can execute blocks in any order.

**Question 4. What is the row-major index for row 2, column 3, and width 5?**

1. It is 13.
2. It is 11.
3. It is 10.

**Answer.** It is 13. The index is 2 times 5 plus 3.

**Question 5. Does this Python experiment establish GPU speed?**

1. Yes, its output gives a GPU benchmark.
2. No, it checks indexing on a CPU.

**Answer.** No, it checks indexing on a CPU. The program never launches CUDA code.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 1, 2, 3. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 32. Warps, scheduling, divergence, and occupancy

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [31-cuda-execution](https://arunshar.github.io/numerical-parallel-computing/lectures/31-cuda-execution.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will distinguish a software thread block from a hardware execution resource.
- You will calculate a resource-limited occupancy model.
- You will explain why maximum occupancy does not guarantee minimum runtime. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/32_scheduling_occupancy.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Ready work hides waiting

A GPU thread often waits for data or for an earlier instruction. The scheduler can issue work from another ready **warp** while the first warp waits. A CUDA warp currently groups 32 threads. The warp size is an architectural programming fact, while the number of schedulers, issue paths, and resident warps depends on the GPU. A course diagram showing one particular streaming multiprocessor should not become a universal specification.

A **streaming multiprocessor**, or SM, hosts blocks and supplies their registers, shared memory, and execution resources. A block is assigned to one SM for its execution. Several blocks can reside on that SM if resources permit. A large grid provides work for later scheduling waves. A grid with only a few blocks can leave a large GPU partly unused even when each resident block is efficient.

**Latency tolerance** requires ready instructions. More resident warps increase the chance that some warp can proceed, but extra warps do not remove a memory-bandwidth limit. A kernel can already saturate memory bandwidth at moderate occupancy. An instruction with a long dependency chain can also limit issue even when the grid contains many threads.

## Resource constraints choose the resident block count

A simplified resource model takes the minimum across several limits:

$$B_{resident}=\min\left(B_{max},\left\lfloor T_{max}/T_b\right\rfloor,\left\lfloor R_{max}/(R_tT_b)\right\rfloor,\left\lfloor S_{max}/S_b\right\rfloor\right).$$

Here $T_b$ is the number of threads per block, $R_t$ is registers per thread, and $S_b$ is shared-memory bytes per block. The SM provides thread, register, and shared-memory budgets $T_{max}$, $R_{max}$, and $S_{max}$. The block-count limit is $B_{max}$. Actual hardware rounds allocations and imposes additional restrictions, so this equation is a teaching model rather than an occupancy calculator for an unnamed GPU.

A toy SM has 2,048 thread slots and 65,536 registers. A 256-thread block using 32 registers per thread consumes 8,192 registers. Both thread and register budgets allow eight blocks. Increasing usage to 64 registers per thread reduces the register limit to four blocks. That change cuts modeled occupancy from 100% to 50% even though the launch geometry is unchanged.

Register use can still be worthwhile. A thread that holds several output accumulators can reuse loaded operands and issue independent arithmetic. The kernel may perform less memory traffic while admitting fewer resident warps. Conversely, forcing a low register cap can cause spills into device-backed local memory. The register count becomes smaller while traffic and runtime become worse.

## Divergence changes useful work per issued instruction

Threads in a warp can follow different control paths. The hardware must execute the paths needed by participating threads, with inactive lanes masked where appropriate. A branch that sends half the lanes down a long path and half down another can waste issue capacity. A branch whose condition is uniform across the warp does not have the same problem.

Independent thread scheduling on newer architectures does not make arbitrary divergent code free. It also removes excuses for relying on undocumented lockstep behavior. A correct implementation uses the synchronization primitive required by the data exchange. Warp-level collectives require participation masks that match the active threads. A mask copied from a full-warp example can be wrong inside a partially active branch.

A boundary guard creates divergence only near the edge of a sufficiently large array. Eliminating that guard by launching too few threads loses data. Padding or specialized tail handling can be useful, but the common case should first remain correct. Large algorithmic divergence, such as one lane traversing a long linked list while its neighbors finish quickly, deserves a different work distribution.

## Barriers belong to a dependency

A block barrier coordinates threads that share data. Threads load a tile, wait until the tile is ready, compute with it, and wait before overwriting it for the next iteration. The second barrier prevents a fast thread from destroying values that a slower thread still needs.

Every participating thread must reach a block barrier consistently. A thread outside the logical image can still load a neutral value and join the barrier. Returning early from that thread before a barrier is a common error in tiled kernels. A barrier does not synchronize different ordinary blocks, and an atomic update does not automatically complete a larger multi-step protocol.

The experiment varies register and shared-memory demand in a deliberately specified toy SM. Its output answers a narrow question about resource arithmetic. A real optimization must additionally inspect achieved occupancy, issue utilization, memory traffic, and elapsed time on the target GPU.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [32_scheduling_occupancy.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/32_scheduling_occupancy.py).

```python
import math
max_threads, registers, shared, max_blocks = 2048, 65536, 65536, 16
threads = 256
for regs_per_thread, shared_per_block in [(32, 8192), (64, 8192), (32, 24576)]:
    resident = min(max_blocks, max_threads // threads,
                   registers // (regs_per_thread * threads),
                   shared // shared_per_block)
    warps = resident * math.ceil(threads / 32)
    occupancy = resident * threads / max_threads
    print(f"registers/thread={regs_per_thread}, shared/block={shared_per_block}: "
          f"blocks={resident}, warps={warps}, occupancy={occupancy:.3f}")
assert 65536 // (64 * 256) == 4
```

The recorded run produced the following output.

```text
registers/thread=32, shared/block=8192: blocks=8, warps=64, occupancy=1.000
registers/thread=64, shared/block=8192: blocks=4, warps=32, occupancy=0.500
registers/thread=32, shared/block=24576: blocks=2, warps=16, occupancy=0.250
```

**Resource demand changes the number of resident blocks.** The three modeled cases admit eight, four, and two blocks. The result is not a prediction of their runtime, because the model contains no instruction or memory-throughput information.

**Try it 1 :class: exercise.**

Change the block size to 128 and recompute each limit. Explain why the same occupied thread count can coexist with a different number of blocks and barriers. :::

**Try it 2 :class: exercise.**

Construct a hypothetical kernel that doubles register use but halves global-memory loads. State the measurements you would need to decide whether its lower occupancy is worthwhile. :::

## Summary

| Decision | What you should remember |
|---|---|
| Occupancy | Occupancy describes resident warps relative to a hardware limit. |
| Divergence | Different lane paths can reduce useful work per issued instruction. |
| Barriers | A barrier is needed when cooperating threads depend on one another’s data. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What limits resident blocks in the 64-register toy case?**

1. The register budget limits them.
2. The number of input arrays limits them.
3. The output value limits them.

**Answer.** The register budget limits them. Each block needs 16,384 registers, so four blocks consume 65,536.

**Question 2. Does higher occupancy always mean lower runtime?**

1. Yes, it always determines runtime.
2. No, throughput and data reuse also matter.

**Answer.** No, throughput and data reuse also matter. A lower-occupancy kernel can move less data or expose more useful instruction-level parallelism.

**Question 3. How many threads are in a CUDA warp?**

1. There are 16 threads.
2. There are 32 threads.
3. There are 256 threads.

**Answer.** There are 32 threads. The CUDA execution model uses 32-thread warps.

**Question 4. Can a block barrier synchronize two ordinary blocks?**

1. Yes, every barrier is global.
2. No, it is scoped to one block.

**Answer.** No, it is scoped to one block. Cross-block dependencies need an appropriate global coordination mechanism.

**Question 5. Why can register caps hurt performance?**

1. They can cause spills to device-backed local memory.
2. They always increase shared-memory capacity.

**Answer.** They can cause spills to device-backed local memory. A spill replaces register storage with memory traffic.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 4. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 33. Memory spaces, data reuse, and tiled matrix multiplication

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [32-scheduling-occupancy](https://arunshar.github.io/numerical-parallel-computing/lectures/32-scheduling-occupancy.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will derive the global-load reduction of a square matrix tile.
- You will separate global coalescing from shared-memory bank behavior.
- You will explain the two barriers in a tiled matrix product. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/33_memory_tiling.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Reusing data changes the cost of arithmetic

A naive matrix product computes $C_{ij}=\sum_k A_{ik}B_{kj}$. One thread can own each output. That mapping performs roughly $2N^3$ floating-point operations for square matrices, counting a multiplication and an addition separately. Without reuse, the threads request about $2N^3$ float32 input values, or $8N^3$ bytes. The simplified input-only arithmetic intensity is therefore about $0.25$ FLOP per byte.

A **tile** is a small block of data that cooperating threads reuse. A block computing a $T\times T$ output tile needs one $T\times T$ tile of $A$ and one tile of $B$ for each stage along the inner dimension. The two input tiles require $2T^2$ loads. They support $T^3$ multiply-add pairs. Each staged value can contribute to several outputs before the block loads the next pair of tiles.

The input-only arithmetic intensity becomes $2T^3/(8T^2)=T/4$ FLOP per byte. A tile width of 16 gives four FLOP per input byte in this model. The factor-of-16 improvement compares requested input loads, not elapsed time. Cache reuse, output stores, edge tiles, synchronization, and instruction overhead change the real traffic and runtime.

## Memory spaces express ownership and lifetime

**Registers** hold thread-private scalars when the compiler can keep them there. **Shared memory** holds block-shared values with an explicitly managed lifetime. **Global memory** stores arrays that persist beyond one block or one kernel. **Local memory** is private in addressability but backed by device memory when registers cannot hold a thread’s data. The word local therefore does not promise low latency.

**Constant memory** can be effective when a warp reads the same coefficient. The constant cache can serve a uniform address efficiently. Threads reading many distinct addresses have different behavior. A read-only data path can improve some cached reads, but its details depend on architecture and compiler choices. Correct code should express read-only intent and measure the result rather than assuming a particular physical cache arrangement.

Shared memory is a manually managed reuse mechanism. It does not automatically make a bad global access pattern good. Threads must first load the tile. A strided loading pattern can waste global transactions even if later shared-memory accesses are efficient. A tiled transpose often uses one mapping for coalesced loading and another mapping for consuming the tile.

## Two address systems produce two different problems

**Coalescing** concerns how a warp’s global addresses combine into memory transactions. Adjacent float32 values are usually a useful starting point. Alignment and the active-lane pattern affect the exact transaction count. A layout that sends consecutive lanes to widely separated rows can make each lane request a different memory segment.

**Bank conflicts** concern simultaneous shared-memory accesses that compete for a bank under the device’s bank mapping. A classic teaching example uses 32 banks and 4-byte words. A column of a row-major $32\times 32$ float tile has a stride of 32 words, so distinct addresses land in the same bank. Padding the row length to 33 words changes the bank pattern. This example is a model of a common configuration, not a universal rule for every access size or instruction.

Broadcast behavior matters. Threads reading the same address are different from threads reading different addresses in one bank. Vectorized instructions and newer hardware capabilities can also change the details. A profiler’s shared-memory conflict metrics are a better guide than counting banks without considering the actual instruction.

## A staged tile needs a complete lifetime

Each stage first loads values into shared memory. Threads whose input coordinates lie outside the matrix store zero into their tile positions. The whole block then reaches a barrier. Every output thread uses the completed tiles to update its private accumulator. A second barrier ensures that all readers have finished before any thread writes the next stage.

A zero-filled edge tile preserves the multiplication formula. Omitting the fill leaves stale values from an earlier stage. Returning early for an out-of-range output thread can also be wrong because that thread may own a load needed by another output. The block’s cooperative load responsibility and each thread’s final output responsibility are separate decisions.

Register tiling extends this idea. One thread holds several output accumulators and reuses a loaded operand across them. A larger register tile reduces load instructions but increases register demand. Larger shared-memory tiles can similarly reduce global traffic while reducing resident blocks. The best tile is therefore a measured compromise among reuse, occupancy, scheduling, and matrix shape.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [33_memory_tiling.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/33_memory_tiling.py).

```python
import numpy as np
rng = np.random.default_rng(7)
a = rng.normal(size=(7, 5))
b = rng.normal(size=(5, 6))
c = np.zeros((7, 6))
tile = 4
for row in range(0, a.shape[0], tile):
    for col in range(0, b.shape[1], tile):
        for k in range(0, a.shape[1], tile):
            aa = a[row:row+tile, k:k+tile]
            bb = b[k:k+tile, col:col+tile]
            c[row:row+aa.shape[0], col:col+bb.shape[1]] += aa @ bb
error = float(np.max(np.abs(c - a @ b)))
assert np.allclose(c, a @ b)
print(f"rectangular_product_max_error={error:.3e}")
for t in [4, 8, 16, 32]:
    print(f"tile={t}, input_only_intensity={t/4:.1f} FLOP/byte")
for stride in [32, 33]:
    banks = {(lane * stride) % 32 for lane in range(32)}
    print(f"word_stride={stride}, distinct_banks={len(banks)}")
```

The recorded run produced the following output.

```text
rectangular_product_max_error=1.110e-16
tile=4, input_only_intensity=1.0 FLOP/byte
tile=8, input_only_intensity=2.0 FLOP/byte
tile=16, input_only_intensity=4.0 FLOP/byte
tile=32, input_only_intensity=8.0 FLOP/byte
word_stride=32, distinct_banks=1
word_stride=33, distinct_banks=32
```

**The tiled CPU reference agrees with the rectangular matrix product within floating-point tolerance.** The intensity values are algebraic estimates, and the bank counts use the explicitly stated 32-bank model. Neither result measures a CUDA kernel.

**Try it 1 :class: exercise.**

Change the matrix shapes to 1 by 9 and 9 by 3. Explain which tiles are partial and why slicing remains correct in the CPU reference. Describe the explicit zero padding that a fixed-size CUDA tile would need. :::

**Try it 2 :class: exercise.**

Add an output-store term to the arithmetic-intensity estimate for a complete square product. Derive how its importance changes as the inner dimension grows. :::

## Summary

| Decision | What you should remember |
|---|---|
| Tiling | A block loads a tile once and reuses it across several outputs. |
| Synchronization | The block waits after loading and before replacing a tile. |
| Layout | Global coalescing and shared-memory bank behavior require separate reasoning. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the input-only intensity of a tile with width 16?**

1. It is 0.25 FLOP/byte.
2. It is 4 FLOP/byte.
3. It is 16 FLOP/byte.

**Answer.** It is 4 FLOP/byte. The model gives T/4, which equals four.

**Question 2. Why is the second tile barrier needed?**

1. It protects readers before the tile is overwritten.
2. It copies the output to the host.

**Answer.** It protects readers before the tile is overwritten. A fast loader must wait until every thread has finished reading the old stage.

**Question 3. Does local memory always mean on-chip memory?**

1. Yes, local means physically close.
2. No, CUDA local memory can be device-backed.

**Answer.** No, CUDA local memory can be device-backed. Local describes thread-private addressability.

**Question 4. What should an out-of-range tile load contribute?**

1. It should contribute zero.
2. It should contribute the old tile value.

**Answer.** It should contribute zero. Zero preserves the dot product over the valid range.

**Question 5. What does padding a modeled 32-word row to 33 words change?**

1. It changes shared-memory bank mapping.
2. It changes the mathematical product.

**Answer.** It changes shared-memory bank mapping. Padding affects layout without changing the logical data.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 5. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 34. Coalescing, roofline models, and optimization experiments

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [33-memory-tiling](https://arunshar.github.io/numerical-parallel-computing/lectures/33-memory-tiling.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will estimate a roofline bound with consistent units.
- You will count memory segments for a simple warp access pattern.
- You will design a measurement that separates kernel speed from application speed. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/34_gpu_performance.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A bottleneck is a resource with insufficient capacity

A kernel requires arithmetic, memory traffic, instruction issue, and coordination. A useful first model asks which resource could finish last. The **roofline model** combines a compute ceiling $P$ with a memory-bandwidth ceiling $B$ and arithmetic intensity $I$:

$$P_{attainable}\leq\min(P,BI).$$

The units must agree. A bandwidth of 100 GB/s multiplied by two FLOP per byte gives 200 GFLOP/s under decimal units. A toy processor with a 1,000 GFLOP/s compute ceiling remains bandwidth-limited at that intensity. It reaches the model’s ridge point at $I=P/B=10$ FLOP per byte. The values in this lesson are invented teaching inputs, not specifications of a particular GPU.

The roofline is a bound, not a promise. A kernel can lie far below both ceilings because its warps lack ready instructions, its memory requests are inefficient, its occupancy is low, or its workload is too small. The model is useful because it rejects impossible targets and suggests which measurements to inspect next.

## Transactions reveal wasted movement

A warp can request a small useful amount of data while causing much larger traffic. Consider 32 lanes reading one float32 each. The useful payload is 128 bytes. In a simplified aligned model with 32-byte memory segments, consecutive words span four segments. A stride of eight words places the lanes in 32 separate segments. The useful payload is unchanged, but the segment model charges eight times as much movement.

This calculation assumes a cold access and does not model cache hits or the full memory hierarchy. A repeated strided read can hit in cache, and an unaligned contiguous read can touch an extra segment. The lesson is to count requested addresses before assuming that adjacent logical work means adjacent physical data.

An array-of-structures layout places fields of one object together. A structure-of-arrays layout places one field from many objects together. If every thread needs only one field, structure-of-arrays often improves coalescing. If every thread needs every field, the choice is less obvious. Alignment, vector loads, cache behavior, and the compiler’s generated instructions all matter.

## Coarsening trades parallelism for reuse

**Thread coarsening** assigns several logical outputs to one thread. The thread can reuse address calculations or loaded operands and can maintain several independent accumulators. That extra work can reduce overhead and expose instruction-level parallelism. It also increases register use and may reduce the number of resident warps.

A naive implementation gives each thread a consecutive chunk. Neighboring lanes may then access widely separated chunks at the same instruction. A coalesced implementation instead lets neighboring lanes process adjacent elements on each iteration. The loop index advances by the block or grid width. The two algorithms perform the same arithmetic but present different address patterns to the memory system.

Coarsening can also reduce the number of blocks until the grid becomes too small. A configuration that helps a large square matrix may hurt a short vector or a skinny matrix. A tuning report should therefore include several shapes and should not generalize one winning tile to every input.

## A useful worklog changes one mechanism at a time

A baseline establishes correctness and a repeatable measurement. The next version changes one design choice, such as coalescing. A later version adds shared-memory reuse, and another version introduces register tiling. Each change should explain what resource cost it reduces and what cost it may increase. A profiler then tests the explanation.

A timer must bracket completed GPU work. Host wall-clock timing around an asynchronous launch can measure only enqueue overhead. CUDA events can measure work on a stream when they are placed and synchronized correctly. End-to-end timing still needs a host-side interval that includes all relevant work. Warmup, allocation, initialization, repeated trials, and output verification must be specified.

A speedup requires a fair baseline. The baseline and optimized version should use the same mathematical operation, input shape, precision, and result tolerance. Tensor-core matrix multiplication can use different input formats or accumulation behavior, so its arithmetic mode belongs in the report. A comparison against a library also needs a clear description of layout conversions and whether their cost is included.

## A profiler answers a sequence of questions

A system-level trace first shows whether the application is transferring data, waiting on the CPU, or running kernels. A kernel-level profiler then examines memory efficiency, issue behavior, occupancy, and dependencies. The two views answer different questions. Optimizing a kernel that contributes one percent of application time cannot yield a large end-to-end improvement.

A practical optimization loop keeps a correctness test next to every benchmark. It records the GPU, driver, toolkit, compiler flags, input shapes, precision, and timing method. It also records failed ideas. An apparent speedup that disappears when the result is checked is a bug, not an optimization.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [34_gpu_performance.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/34_gpu_performance.py).

```python
lanes = list(range(32))
for stride in [1, 2, 8]:
    addresses = [4 * stride * lane for lane in lanes]
    segments = {a // 32 for a in addresses}
    useful, charged = 128, 32 * len(segments)
    print(f"word_stride={stride}, segments={len(segments)}, useful_fraction={useful/charged:.3f}")
peak_gflops, bandwidth_gbs = 1000.0, 100.0
for intensity in [0.25, 2.0, 10.0, 20.0]:
    bound = min(peak_gflops, bandwidth_gbs * intensity)
    print(f"intensity={intensity:.2f}, roofline_bound={bound:.1f} GFLOP/s")
assert min(peak_gflops, bandwidth_gbs * 2) == 200
```

The recorded run produced the following output.

```text
word_stride=1, segments=4, useful_fraction=1.000
word_stride=2, segments=8, useful_fraction=0.500
word_stride=8, segments=32, useful_fraction=0.125
intensity=0.25, roofline_bound=25.0 GFLOP/s
intensity=2.00, roofline_bound=200.0 GFLOP/s
intensity=10.00, roofline_bound=1000.0 GFLOP/s
intensity=20.00, roofline_bound=1000.0 GFLOP/s
```

**The segment model distinguishes useful bytes from charged bytes.** The contiguous case uses four segments, while the stride-eight case uses 32. The roofline values are upper bounds for a toy machine and must not be reported as measured GPU performance.

**Try it 1 :class: exercise.**

Offset every address by four bytes. Count the segments again and explain why alignment changes the contiguous case even though the stride is unchanged. :::

**Try it 2 :class: exercise.**

Suppose a kernel is 20% of total runtime and becomes four times faster. Calculate the application speedup and explain why it is smaller than four. :::

## Summary

| Decision | What you should remember |
|---|---|
| Roofline | The minimum of compute capacity and bandwidth times intensity gives a first upper bound. |
| Coalescing | Physical addresses determine transaction efficiency. |
| Measurement | A timing interval must include completed work and a stated scope. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the toy ridge-point intensity?**

1. It is 10 FLOP/byte.
2. It is 100 FLOP/byte.
3. It is 0.1 FLOP/byte.

**Answer.** It is 10 FLOP/byte. The ratio 1000 GFLOP/s divided by 100 GB/s is 10 FLOP/byte.

**Question 2. Does the roofline bound guarantee achieved throughput?**

1. Yes, every kernel reaches it.
2. No, other limits can reduce throughput.

**Answer.** No, other limits can reduce throughput. Instruction dependencies and other costs are absent from the model.

**Question 3. How many 32-byte segments does the aligned contiguous warp model touch?**

1. It touches four segments.
2. It touches 32 segments.

**Answer.** It touches four segments. Its 128-byte payload spans four aligned segments.

**Question 4. Why can timing only the host launch be misleading?**

1. The launch can return before GPU completion.
2. The host always executes the kernel itself.

**Answer.** The launch can return before GPU completion. Asynchronous enqueue time is different from completed execution time.

**Question 5. What must accompany a claimed GPU speedup?**

1. Only the fastest number is needed.
2. A comparable baseline, correctness check, and measurement scope are needed.

**Answer.** A comparable baseline, correctness check, and measurement scope are needed. The claim depends on what was computed and what the timer included.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 6. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [CUDA matrix-multiplication worklog](https://siboehm.com/articles/22/CUDA-MMM) supplies a concrete optimization sequence. Its reported measurements belong to its own setup, and this lesson does not reproduce those measurements.


---

# 35. Convolution, constant memory, and halo tiles

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [34-gpu-performance](https://arunshar.github.io/numerical-parallel-computing/lectures/34-gpu-performance.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will state the boundary convention before implementing a convolution.
- You will derive the input region needed by an output tile.
- You will compare direct loads, shared-memory tiles, and cached halo reads. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/35_convolution.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A small filter describes a local operation

A **convolution** combines nearby input values with filter coefficients. A one-dimensional mathematical convolution can be written as $y_i=\sum_{k=-r}^{r}h_kx_{i-k}$. Many image and machine-learning implementations instead compute cross-correlation, $y_i=\sum_{k=-r}^{r}h_kx_{i+k}$. The formulas agree for a symmetric filter, but they differ for an asymmetric filter. A correctness test should include an asymmetric filter so that an accidental reversal becomes visible.

This lesson computes cross-correlation and names that choice explicitly. The input outside the array is zero. A length-five signal and a radius-one filter therefore produce five outputs under same-size padding. A valid-only convention would produce three outputs. A periodic convention would wrap the indices, and a reflected convention would mirror the signal. These are different mathematical problems, not interchangeable implementation details.

A two-dimensional filter has a neighborhood in both row and column coordinates. A radius-$r$ square filter contains $(2r+1)^2$ coefficients. A direct kernel gives one thread one output pixel. The thread loops over the filter and accumulates valid input-coefficient products. Its writer ownership is simple, but adjacent threads reread many of the same input pixels.

## An output tile needs a larger input tile

An output tile with width $T$ and radius $r$ needs an input span of $T+2r$ in one dimension. The extra values form a **halo**. In two dimensions, the required input tile has $(T+2r)^2$ entries, while the block produces $T^2$ outputs. The loading overhead relative to the output count is

$$\rho=\frac{(T+2r)^2}{T^2}.$$

For $T=8$ and $r=1$, the block loads 100 input entries for 64 outputs, giving a ratio of 1.5625. For $T=16$, it loads 324 entries for 256 outputs, giving 1.265625. Larger tiles reduce the relative halo overhead, but they require more threads, shared memory, or per-thread work.

An input-tile mapping launches a thread for every loaded position. Some threads load only halo values and do not compute outputs. An output-tile mapping launches a thread for every output and assigns extra halo loads through a loop. Both mappings can be correct. Their efficiency depends on loading balance, occupancy, and the amount of unused work.

The barrier belongs after all cooperative loads. Threads that correspond to padded input positions store zero and still participate. A thread must not return before the barrier merely because its final output lies outside the image. The tile’s loading work and the output bounds check are separate responsibilities.

## Constant coefficients have a useful access pattern

All threads often use the same filter coefficient at the same loop iteration. **Constant memory** is well suited to that uniform read pattern when the coefficient array is small enough and the access behavior matches the device. A constant-memory declaration does not make every arbitrary coefficient pattern fast. Divergent coefficient addresses can serialize requests through the constant path.

The input image has a different access pattern. Neighboring output threads read overlapping neighborhoods, so caches can already capture some reuse. A shared-memory version adds explicit loads and barriers to guarantee block-local reuse. Its benefit can shrink when the direct kernel’s cache behavior is already good. The filter radius, image size, and target hardware determine the result.

A cached-halo design places only the central input tile in shared memory and obtains halo values through global reads. Neighboring blocks may make those values available in cache. This design reduces shared-memory demand and loader complexity, but its traffic depends more on cache behavior. It should be measured separately from a fully staged halo design.

## Separable filters change the algorithm

Some two-dimensional filters factor as an outer product $H=uv^T$. Their convolution can be implemented as a horizontal pass and a vertical pass. A direct square filter uses roughly $(2r+1)^2$ products per output, while the two one-dimensional passes use roughly $2(2r+1)$. The second approach also writes and rereads an intermediate image unless the implementation fuses or tiles the passes.

A filter must actually be separable before this transformation is valid. An arbitrary filter can sometimes be approximated by a low-rank decomposition, but that changes the result and introduces an accuracy question. A useful optimization report states whether it preserves the exact mathematical operator or approximates it.

The CPU reference compares a direct loop with a tiled implementation under zero padding. The asymmetric filter exposes its cross-correlation convention. The test checks arithmetic equivalence and partial final tiles. It does not settle whether tiling or constant memory would be faster on a GPU.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [35_convolution.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/35_convolution.py).

```python
import numpy as np
x = np.array([2., 1., 4., 3., 5.])
h = np.array([1., 2., -1.])
r, tile = 1, 3
padded = np.pad(x, (r, r))
direct = np.array([sum(h[k] * padded[i+k] for k in range(len(h))) for i in range(len(x))])
tiled = np.zeros_like(x)
for start in range(0, len(x), tile):
    count = min(tile, len(x) - start)
    local = padded[start:start + count + 2*r]
    for t in range(count):
        tiled[start+t] = np.dot(local[t:t+len(h)], h)
assert np.array_equal(direct, tiled)
print("zero_padded_cross_correlation=", direct.tolist())
for t in [8, 16, 32]:
    print(f"output_width={t}, halo_load_ratio={(t+2*r)**2/t**2:.6f}")
```

The recorded run produced the following output.

```text
zero_padded_cross_correlation= [3.0, 0.0, 6.0, 5.0, 13.0]
output_width=8, halo_load_ratio=1.562500
output_width=16, halo_load_ratio=1.265625
output_width=32, halo_load_ratio=1.128906
```

**The direct and tiled references produce the same five outputs.** The halo ratio falls as tile width grows, but this count omits occupancy, cache reuse, and synchronization costs.

**Try it 1 :class: exercise.**

Reverse the asymmetric filter and compare the outputs. Explain the difference between convolution and cross-correlation using the value at the first pixel. :::

**Try it 2 :class: exercise.**

Replace zero padding with periodic padding. Derive the expected first and last outputs before changing the implementation. :::

## Summary

| Decision | What you should remember |
|---|---|
| Boundary convention | Padding defines the mathematical operation at the edges. |
| Halo | An output tile requires neighboring input values beyond its output region. |
| Coefficient reuse | Uniform coefficient reads can suit constant memory. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What input width supports an output width 8 and radius 1?**

1. The input width is 8.
2. The input width is 10.
3. The input width is 16.

**Answer.** The input width is 10. The input tile adds one halo value on each side.

**Question 2. Do asymmetric convolution and cross-correlation always agree?**

1. Yes, the filter direction never matters.
2. No, reversing an asymmetric filter changes the result.

**Answer.** No, reversing an asymmetric filter changes the result. Convolution and cross-correlation use opposite filter directions.

**Question 3. What should padded loader threads do before a block barrier?**

1. They should write the neutral value and participate.
2. They should always return immediately.

**Answer.** They should write the neutral value and participate. The cooperative tile must be complete before computation.

**Question 4. When is a two-pass separable implementation exact?**

1. It is exact when the filter factors into the two one-dimensional filters.
2. It is exact for every arbitrary filter.

**Answer.** It is exact when the filter factors into the two one-dimensional filters. The outer-product factorization defines separability.

**Question 5. Does a smaller halo ratio prove a faster GPU kernel?**

1. Yes, it contains every cost.
2. No, it omits other hardware costs.

**Answer.** No, it omits other hardware costs. Tile size also affects occupancy and synchronization.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 7. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 36. Stencil computations and space-time dependencies

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [35-convolution](https://arunshar.github.io/numerical-parallel-computing/lectures/35-convolution.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will derive the dependency neighborhood of a stencil update.
- You will explain why separate input and output buffers preserve a time step.
- You will compare spatial tiling, register rolling, and temporal blocking. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/36_stencils.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A stencil applies the same neighborhood rule repeatedly

A **stencil** updates each point from a fixed pattern of nearby points. A one-dimensional diffusion update uses the center and its two neighbors. A two-dimensional five-point stencil uses the center and four axial neighbors. A three-dimensional seven-point stencil adds the two neighbors along the third axis. The pattern describes dependencies, while the coefficients describe the mathematical model.

For the heat equation $u_t=\alpha u_{xx}$, a forward-Euler time step and centered spatial difference give

$$u_i^{n+1}=u_i^n+\lambda(u_{i-1}^n-2u_i^n+u_{i+1}^n),\qquad\lambda=\alpha\Delta t/\Delta x^2.$$

The superscript denotes time, not a power. Every value on the right belongs to time step $n$. Updating the array in place from left to right changes that dependency. Later positions would use some already updated neighbors and some old neighbors. The result would be a different method whose properties need separate analysis.

The reference uses two buffers and swaps their roles after each complete step. Fixed zero boundary values represent a rod whose ends are held at zero. Periodic boundaries, prescribed fluxes, and time-varying boundary data require different indexing and physical interpretations.

## Numerical stability is part of the kernel contract

The one-dimensional explicit diffusion scheme has a familiar stability restriction $0\leq\lambda\leq1/2$ under the standard uniform-grid setting. The GPU cannot repair a time step that violates the mathematical stability condition. A fast kernel can produce an unstable solution more quickly.

For a two-dimensional equal-spacing diffusion scheme, the corresponding coefficient restriction becomes more severe because each point exchanges information with more neighbors. The precise condition depends on the operator and discretization. The correct workflow derives or consults the stability condition, then chooses the kernel implementation.

A stencil is often bandwidth-sensitive because it performs few operations per value. Neighboring outputs reuse input values, so caching and tiling can reduce memory traffic. The useful quantity is the traffic reaching each memory level, not merely the number of source-level array references. A compiler or cache can already remove some repeated traffic.

## A tile has a surface and an interior

A tile of side $T$ with a one-cell halo stores $(T+2)^d$ values in $d$ dimensions and updates $T^d$ interior values. The halo-to-interior ratio worsens for small tiles. In three dimensions, a full shared-memory cube can become too large before the surface overhead becomes attractive.

**Register tiling** can avoid storing an entire cube. A thread processes several positions along one axis while keeping a small rolling window in registers. The next iteration shifts the window and loads one new plane or value. Neighboring threads still cooperate across the other dimensions. This arrangement trades registers and control logic for fewer loads.

A register window must preserve the correct time level. Reusing a register is safe only after every dependent output has consumed its value. A rolling implementation is therefore easiest to derive from a dependency diagram. The diagram shows the current output, the values it reads, and the last use of each input.

## Temporal blocking expands the dependency cone

**Temporal blocking** performs several time steps while a tile remains close to the processor. One time step needs a radius-one halo. Two time steps need information from two cells away, and $k$ steps need a halo wide enough for the $k$-step dependency cone. A block can compute several steps on a progressively shrinking valid interior.

The technique can reduce global traffic, but it performs redundant work near overlapping tile boundaries and consumes more local storage. A larger time block is not automatically better. The design balances reuse against redundant computation, synchronization, and occupancy. Distributed-memory stencils add another constraint because halos must be exchanged between ranks.

The straightforward multi-kernel implementation uses one kernel per time step. The kernel boundary supplies a global ordering point for the next step in the same stream. A single ordinary kernel cannot replace that ordering with a block barrier. Cooperative launches or specialized algorithms can provide broader coordination, but they introduce requirements that must be checked explicitly.

## Conservation and symmetry make useful tests

A symmetric initial impulse should remain symmetric under a symmetric stencil and symmetric boundaries. Periodic diffusion should preserve the sum up to rounding for the matching discrete operator. Fixed-value boundaries can exchange heat with the domain, so the sum need not remain constant. These properties test the mathematics more strongly than comparing one printed element.

The experiment checks symmetry, nonnegativity, and the result of four small stable steps. It uses an exact binary coefficient, which keeps this particular toy arithmetic easy to inspect. A production solver still needs convergence tests as the spatial and temporal grids are refined.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [36_stencils.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/36_stencils.py).

```python
import numpy as np
u = np.zeros(9)
u[4] = 1.0
lam = 0.25
for step in range(4):
    new = u.copy()
    new[1:-1] = u[1:-1] + lam * (u[:-2] - 2*u[1:-1] + u[2:])
    new[0] = new[-1] = 0.0
    u = new
assert np.allclose(u, u[::-1])
assert np.min(u) >= 0
print("four_step_solution=", u.tolist())
print(f"sum={u.sum():.8f}, maximum={u.max():.8f}")
for dimension in [1, 2, 3]:
    print(f"dimension={dimension}, tile_width=8, storage_ratio={(10/8)**dimension:.6f}")
```

The recorded run produced the following output.

```text
four_step_solution= [0.0, 0.03125, 0.109375, 0.21875, 0.2734375, 0.21875, 0.109375, 0.03125, 0.0]
sum=0.99218750, maximum=0.27343750
dimension=1, tile_width=8, storage_ratio=1.250000
dimension=2, tile_width=8, storage_ratio=1.562500
dimension=3, tile_width=8, storage_ratio=1.953125
```

**The stable update preserves symmetry and nonnegativity in this example.** The total decreases slightly because the fixed boundaries remove heat. The storage ratios quantify halo overhead without claiming a GPU speedup.

**Try it 1 :class: exercise.**

Use a periodic boundary rule and verify conservation of the sum over 20 steps. Explain why the original fixed-boundary conservation expectation would be wrong. :::

**Try it 2 :class: exercise.**

Increase lambda above one half and observe a longer run. Describe the mathematical instability separately from an indexing or synchronization bug. :::

## Summary

| Decision | What you should remember |
|---|---|
| Time levels | Double buffering preserves the chosen explicit method. |
| Stability | The time step must satisfy the discretization’s requirements. |
| Temporal reuse | Several local time steps require a wider dependency halo. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Why does this explicit update use two buffers?**

1. It keeps every right-hand-side value at the old time level.
2. It doubles the mathematical time step.

**Answer.** It keeps every right-hand-side value at the old time level. In-place updates would mix time levels.

**Question 2. What is the standard1D diffusion bound in this lesson?**

1. Lambda must be at most one half.
2. Lambda must be at least two.

**Answer.** Lambda must be at most one half. The centered explicit scheme is stable for lambda between zero and one half.

**Question 3. How wide is the dependency radius after k radius-one steps?**

1. It can grow to k cells.
2. It remains one cell.

**Answer.** It can grow to k cells. Each step propagates dependence by one additional cell.

**Question 4. Does a block barrier order the whole grid?**

1. Yes, it orders every block.
2. No, it coordinates one block.

**Answer.** No, it coordinates one block. A multi-step stencil needs a valid global ordering mechanism.

**Question 5. Should fixed-zero boundaries preserve total heat?**

1. They always preserve it.
2. They can remove heat from the domain.

**Answer.** They can remove heat from the domain. The boundary acts as an external reservoir.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 8. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 37. Histograms, atomic updates, and privatization

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [36-stencils](https://arunshar.github.io/numerical-parallel-computing/lectures/36-stencils.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will explain the lost-update race in a histogram.
- You will derive when block-private bins reduce global updates.
- You will distinguish atomicity from ordering a larger computation. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/37_histograms_atomics.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Counting creates competing writers

A **histogram** counts how many observations fall in each bin. The serial operation is simple. The program reads a value, computes its bin, and increments that bin’s count. A parallel implementation creates a conflict because many threads can choose the same bin.

An increment is normally a read-modify-write sequence. Two threads can both read the old count 7, both compute 8, and both write 8. The correct result 9 is lost. An **atomic addition** makes the update indivisible with respect to competing updates at its specified scope. It prevents this lost update, but it does not guarantee that the overall algorithm is fast.

An atomic update also does not establish arbitrary program-wide ordering. A thread that increments a completion counter may need explicit memory-ordering semantics before another thread can safely consume separate data. A histogram of integer counts needs only the atomic bin updates and proper ordering between initialization, counting, and later use. More complicated producer-consumer protocols require a separate memory-model argument.

## Contention depends on the distribution

A uniform distribution over many bins spreads updates. A highly skewed distribution sends many threads to the same address. Both inputs contain the same number of observations, but the second creates much more contention. A benchmark using only uniformly random input can conceal the worst case.

Suppose eight observations map to bins $[0,0,0,0,1,1,2,3]$. A direct implementation performs eight global atomic updates. The count for bin 0 receives four competing contributions. A one-bin input is an extreme stress test, while a one-observation-per-bin input stresses a different resource pattern.

Integer counter width matters. A 32-bit unsigned bin overflows after sufficiently many contributions. An implementation must choose a type that can represent the maximum count, not just the typical count. Input validation also matters because an invalid bin index becomes an out-of-bounds write.

## Privatization moves the contention

**Privatization** gives each block its own temporary histogram. Threads within the block update local bins, often in shared memory. The block then contributes its local counts to the global result. If a block processes $M$ observations into $K$ bins, the merge can require at most $K$ nonzero global updates instead of $M$ updates.

The optimization works best when each block processes many observations relative to the number of bins. It also has costs. The block must initialize its private bins, perform its local updates, synchronize, and merge. A large bin count can exceed convenient shared-memory capacity or make initialization expensive. Several levels of privatization can trade local storage against contention.

A warp-private histogram can reduce contention between warps but replicates the bins more times. A per-thread histogram eliminates update conflicts within a thread but uses much more storage. The correct granularity depends on the input distribution, bin count, and resource budget.

## Coarsening and aggregation reduce update count differently

Thread coarsening assigns multiple observations to one thread. The input access pattern should remain coalesced across neighboring lanes on each iteration. Giving each lane a large contiguous chunk can create strided warp accesses when all lanes process their first element.

**Aggregation** combines repeated contributions before issuing an atomic operation. A thread that sees a run of the same bin can accumulate a local count and perform one update for the run. A warp can also combine lanes targeting the same bin with suitable grouping primitives. The benefit depends on repetition. Aggregation adds work when almost every target is different.

Privatization reduces the scope of contention, while aggregation reduces the number of updates. They can be combined. A block-private histogram can receive aggregated thread or warp contributions and later merge to global memory. Each level should have an explicit correctness argument and a cost model.

## Initialization and merging are algorithm stages

The global histogram must begin at zero unless accumulation across batches is intentional. A kernel launch in a stream must follow the initialization operation in a valid order. Inside a block-private kernel, all bins must be initialized before any thread updates them. A barrier after initialization establishes that dependency.

Another barrier is needed before the block reads its completed private histogram for merging. Threads that have no input work still participate in initialization and synchronization. A block whose final input chunk is partial must not let inactive workers skip a required barrier.

The reference compares direct counting and block-private counting. It also counts the number of nonzero global contributions in the private design. The arithmetic result is exact for these small integer counts. The contribution count is a structural property, not a timing result for shared-memory or global-memory atomics.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [37_histograms_atomics.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/37_histograms_atomics.py).

```python
import numpy as np
values = np.array([0,0,0,0,1,1,2,3,0,0,1,1,2,2,3,3])
bins, chunk = 4, 8
direct = np.bincount(values, minlength=bins)
private = np.zeros(bins, dtype=np.int64)
merge_updates = 0
for start in range(0, len(values), chunk):
    local = np.bincount(values[start:start+chunk], minlength=bins)
    merge_updates += np.count_nonzero(local)
    private += local
assert np.array_equal(direct, private)
print("histogram=", private.tolist())
print(f"direct_global_updates={len(values)}, private_nonzero_merges={merge_updates}")
skewed = np.zeros(16, dtype=int)
print(f"one_bin_private_merges={sum(np.count_nonzero(np.bincount(skewed[s:s+chunk], minlength=bins)) for s in range(0,16,chunk))}")
```

The recorded run produced the following output.

```text
histogram= [6, 4, 3, 3]
direct_global_updates=16, private_nonzero_merges=8
one_bin_private_merges=2
```

**The private design preserves the histogram while reducing the counted global contributions from 16 to 8.** The one-bin example needs only two merge contributions. These counts do not include initialization or local atomic costs.

**Try it 1 :class: exercise.**

Increase the number of bins until each observation occupies a different bin. Explain why privatization may then save few global updates while retaining its setup costs. :::

**Try it 2 :class: exercise.**

Change the chunk size to 4 and 16. Predict the number of private histograms and nonzero merges for both the mixed and one-bin inputs. :::

## Summary

| Decision | What you should remember |
|---|---|
| Atomic addition | An atomic bin update prevents lost increments. |
| Privatization | Local histograms reduce the number of global contributions when bins are reused. |
| Distribution | Skew changes contention and aggregation opportunities. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What can two unsynchronized increments of 7 produce?**

1. They can incorrectly produce 8.
2. They must produce 9.

**Answer.** They can incorrectly produce 8. Both threads can read 7 before either writes 8.

**Question 2. How many global contributions does the private mixed example count?**

1. It counts eight.
2. It counts sixteen.
3. It counts one.

**Answer.** It counts eight. Each of the two chunks contributes to four bins.

**Question 3. Does atomic addition make every multi-step protocol correct?**

1. Yes, one atomic operation orders everything.
2. No, separate data can require memory-ordering reasoning.

**Answer.** No, separate data can require memory-ordering reasoning. Atomicity of one location is narrower than a complete producer-consumer protocol.

**Question 4. Which input creates extreme single-bin contention?**

1. An input whose values all select one bin does.
2. An input with a distinct bin for every value does.

**Answer.** An input whose values all select one bin does. All updates compete for the same location.

**Question 5. Why can very many bins make privatization unattractive?**

1. The local storage and initialization cost grow.
2. The count values stop being integers.

**Answer.** The local storage and initialization cost grow. Every private histogram requires storage and setup.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 9. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 38. Parallel reductions and numerical order

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [37-histograms-atomics](https://arunshar.github.io/numerical-parallel-computing/lectures/37-histograms-atomics.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will derive the work and depth of a reduction tree.
- You will handle arbitrary input lengths with an identity element.
- You will explain why floating-point reduction order affects reproducibility. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/38_reductions.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A reduction replaces a collection with one result

A **reduction** combines many inputs with an operation such as sum, maximum, or logical conjunction. The operation ideally has an identity and is associative. Zero is the identity for addition, negative infinity is the identity for maximum over real values, and true is the identity for logical conjunction. The choice of identity makes missing elements in a partial tile harmless.

A serial sum of $N$ values uses $N-1$ additions and has a dependency chain of length $N-1$. A balanced binary tree also uses $N-1$ additions but has depth approximately $\lceil\log_2N\rceil$. The tree exposes parallelism because many additions on the same level are independent.

The tree does not imply unlimited speedup. Workers must load inputs, exchange partial results, synchronize, and eventually finish with very little remaining parallel work. A large GPU reduction therefore combines local work with a hierarchy of blocks and kernel stages.

## The thread mapping changes inactive work

An interleaved reduction can leave active threads scattered across the block. A contiguous mapping keeps the remaining active threads together as the tree shrinks. Both compute the same abstract tree, but they can create different divergence and memory-access patterns.

A common block reduction first loads values into shared memory. Each thread can load and combine several input values before storing one partial sum. The block then reduces the partial sums. This coarsening reduces the number of active blocks and the amount of shared-memory work, but excessive coarsening can reduce available parallelism.

A thread must not read beyond the final partial chunk. It can initialize its accumulator to zero and add only valid elements. The same principle applies to a max reduction with its corresponding identity. A wrong identity can silently corrupt the answer. Using zero for a maximum of negative numbers is an example.

## The global hierarchy needs an ordering mechanism

Each block writes one partial result to a separate output location. A second kernel reduces those partial results. Further stages continue until one result remains. A kernel boundary in the same stream provides the required ordering between stages. The host can also choose a small final CPU reduction, but that introduces a transfer and a different cost model.

An atomic global accumulation is another design. It avoids storing and rereading a partial array, but it introduces contention and makes the accumulation order scheduling-dependent. A floating-point atomic sum can therefore vary in its final low-order bits across runs. The operation can be race-free while its numerical output is not bitwise deterministic.

A single-kernel global barrier requires stronger execution guarantees than ordinary blocks provide. A last-block completion protocol can be implemented with careful memory ordering and counters, but it is not equivalent to adding a block barrier. A tutorial should begin with the explicit multi-stage version because its dependency structure is easier to verify.

## Floating-point addition is only approximately associative

Real-number addition satisfies $(a+b)+c=a+(b+c)$. Floating-point addition rounds intermediate results. With sufficiently different magnitudes, one order can lose a small contribution while another retains it. A parallel tree changes the order, so it can change the final bits even when every thread and memory access is correct.

Pairwise summation often has a better error-growth bound than a long serial chain because its depth is logarithmic. That observation does not mean every tree is more accurate for every individual input. Compensated methods, wider accumulation, and reproducible summation methods address different accuracy and cost goals.

A test should compare against a reference using an explicit absolute and relative tolerance. A near-zero expected result needs a meaningful absolute tolerance because relative error can become enormous or undefined. A deterministic tree helps reproducibility, but compiler transformations, fused operations, and precision choices can still affect cross-platform bitwise equality.

## Reduction is a reusable pattern

A dot product first multiplies pairs and then reduces the products. A norm reduces squares and applies a square root, although a stable norm implementation must avoid overflow and underflow. An argmax reduces value-index pairs with a stated tie-breaking rule. A log-sum-exp computation uses a maximum followed by a shifted exponential sum to improve numerical stability.

The reference implements a binary sum tree for an odd-length array and separately demonstrates order sensitivity in float32. The example is intentionally small enough to inspect. The tree-level count is an algorithmic depth count, not a GPU clock count.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [38_reductions.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/38_reductions.py).

```python
import numpy as np
values = np.arange(1, 10, dtype=np.float64)
level = values.copy()
levels = 0
while len(level) > 1:
    if len(level) % 2:
        level = np.append(level, 0.0)
    level = level[0::2] + level[1::2]
    levels += 1
assert level[0] == 45
x = np.array([1e8, 1., -1e8, 1.], dtype=np.float32)
left = np.float32(0)
for value in x:
    left = np.float32(left + value)
reordered = np.float32(np.float32(x[0] + x[2]) + np.float32(x[1] + x[3]))
print(f"odd_length_sum={level[0]:.0f}, tree_levels={levels}")
print(f"float32_left_order={left:.0f}, reordered={reordered:.0f}, exact_real_sum=2")
```

The recorded run produced the following output.

```text
odd_length_sum=45, tree_levels=4
float32_left_order=1, reordered=2, exact_real_sum=2
```

**The odd-length reduction returns 45, and its tree takes four levels.** The float32 example returns different answers for different addition orders. That difference demonstrates rounding behavior and is not evidence of a data race.

**Try it 1 :class: exercise.**

Replace addition with maximum and use an all-negative input. Choose the correct identity and explain why zero would fail. :::

**Try it 2 :class: exercise.**

Compare a serial sum, pairwise sum, and `math.fsum` on a long cancellation-heavy input. Report absolute error and explain the cost of a more accurate reference. :::

## Summary

| Decision | What you should remember |
|---|---|
| Tree structure | A balanced reduction exposes logarithmic dependency depth. |
| Boundary handling | An identity element handles partial input groups. |
| Numerical order | Race freedom does not imply bitwise reproducibility. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the exact sum of integers 1 through 9?**

1. It is 45.
2. It is 36.
3. It is 81.

**Answer.** It is 45. The arithmetic-series formula gives 9 times 10 divided by 2.

**Question 2. What identity pads an addition reduction?**

1. The identity is zero.
2. The identity is one.

**Answer.** The identity is zero. Adding zero leaves a partial sum unchanged.

**Question 3. Is floating-point addition exactly associative?**

1. Yes, it behaves like real addition in every case.
2. No, rounding can change the result.

**Answer.** No, rounding can change the result. Intermediate rounding depends on the order.

**Question 4. What does a second kernel supply in a hierarchical reduction?**

1. It supplies a valid stage boundary and reduces block partials.
2. It makes ordinary block barriers global.

**Answer.** It supplies a valid stage boundary and reduces block partials. The stream orders the kernels, while barriers remain block-scoped.

**Question 5. Can a race-free atomic sum vary in low-order bits?**

1. Yes, update order can vary.
2. No, atomicity fixes the arithmetic order.

**Answer.** Yes, update order can vary. Atomicity prevents lost updates but does not choose one fixed addition sequence.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 10. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 39. Prefix sums and work-efficient coordination

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [38-reductions](https://arunshar.github.io/numerical-parallel-computing/lectures/38-reductions.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will distinguish inclusive and exclusive prefix sums.
- You will compare logarithmic-depth scan algorithms by total work.
- You will use a scan to allocate output positions without a global counter per item. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/39_prefix_scan.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A scan preserves every partial result

A **prefix sum**, or scan, returns a cumulative result at every position. An inclusive sum returns $s_i=\sum_{j=0}^{i}x_j$. An exclusive sum returns $e_i=\sum_{j=0}^{i-1}x_j$, with $e_0=0$. For the input $[3,1,4,1]$, the inclusive result is $[3,4,8,9]$, and the exclusive result is $[0,3,4,8]$.

A reduction keeps only the final aggregate. A scan retains the earlier aggregates as well. That extra information lets threads agree on where to write variable amounts of output. If thread $i$ will emit $c_i$ elements, an exclusive scan of the counts gives its starting offset. The total output length is the final offset plus the final count.

A filter uses the same idea with counts of zero or one. Each thread evaluates a predicate, the scan computes positions, and retained elements write to those positions. This process is called stream compaction. It preserves input order when the predicate and offsets follow the original order.

## A short critical path can still do too much work

The Kogge-Stone or iterative-doubling style of scan performs stages with distances 1,2,4, and so on. At a stage with distance $d$, position $i$ combines its previous value with position $i-d$ when that predecessor exists. A separate old-stage view is necessary because the stage is logically parallel. An in-place left-to-right CPU loop would accidentally reuse updates from the current stage.

The depth is logarithmic, but many positions perform an addition at every stage. The total work is $O(N\log N)$. This extra work can be acceptable for a short warp-sized sequence where low latency and simple communication matter. It becomes more expensive for a large array.

A **work-efficient scan** uses $O(N)$ additions and $O(\log N)$ depth. A common Blelloch construction builds a reduction tree in an up-sweep, sets the root to the identity, and distributes prefixes in a down-sweep. Brent-Kung-style prefix networks similarly reduce the number of active combine operations through a tree structure. Their exact wiring and boundary handling differ, so names should not substitute for a concrete dependency diagram.

## The down-sweep carries a prefix into each subtree

At an internal tree node, the parent’s prefix is the aggregate of every input before that subtree. The left child receives the same prefix. The right child receives that prefix combined with the total of the left subtree. This rule explains the exclusive result without treating the algorithm as a mysterious sequence of swaps.

For eight inputs, the up-sweep uses seven additions. The down-sweep uses seven more additions. The fourteen additions are linear in the input length, although the scan computes eight outputs. Padding to a power of two with the identity simplifies the tree but adds work for non-power-of-two arrays.

The operation need not be addition. It must support a suitable associative combination and identity. Operand order matters for associative but noncommutative operations. A scan implementation that swaps the order while moving values through the tree can be wrong even though addition tests pass.

## Large arrays need a hierarchy

A block can scan a tile and write its total. Another stage scans the block totals. A final stage adds each block’s offset to its local result. The local scan, block-total scan, and offset addition form an easy-to-verify hierarchy. The extra passes move data more than once, which can become bandwidth-expensive.

More advanced single-pass scans use carefully designed inter-block communication protocols to obtain prior prefixes without a separate full-array adjustment pass. Such protocols require precise memory ordering and progress arguments. They should be implemented from a well-tested library or a fully specified algorithm, not improvised from a spinning global counter.

Thread coarsening can scan a small local sequence per thread. The block then scans thread totals and adds the resulting offsets to the local sequences. Coarsening reduces inter-thread communication but increases register demand. A coalesced load layout may differ from the convenient per-thread scan layout, requiring a deliberate transpose or indexing strategy.

## Segments mark independent sequences

A **segmented scan** performs separate scans within one packed array. A flag marks the beginning of each segment. The combine operation carries both a value and a flag so that a new segment resets the accumulated prefix. This pattern appears in grouped reductions, sparse data structures, and variable-length sequences.

A segmented scan requires clear handling of an empty segment and of a segment crossing a block boundary. The flags are part of the mathematical input. A local block cannot simply reset its first value unless that position is a real segment boundary.

The experiment compares an inclusive doubling scan with an exclusive Blelloch scan. It then uses exclusive positions to compact even values. The tests establish the toy algorithms’ outputs and work counts. They do not demonstrate a GPU scan implementation or a single-pass memory-ordering protocol.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [39_prefix_scan.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/39_prefix_scan.py).

```python
import numpy as np
x = np.array([3,1,4,1,5,9,2,6], dtype=np.int64)
a = x.copy(); doubling_adds = 0
for distance in [1,2,4]:
    old = a.copy()
    a[distance:] = old[distance:] + old[:-distance]
    doubling_adds += len(x)-distance
assert np.array_equal(a, np.cumsum(x))
b = x.copy(); adds = 0; n = len(b)
stride = 2
while stride <= n:
    for end in range(stride-1,n,stride):
        b[end] += b[end-stride//2]; adds += 1
    stride *= 2
b[-1] = 0
stride = n
while stride >= 2:
    for end in range(stride-1,n,stride):
        left = end-stride//2
        saved = b[left]; b[left] = b[end]; b[end] += saved; adds += 1
    stride //= 2
assert np.array_equal(b, np.r_[0,np.cumsum(x)[:-1]])
flags = (x % 2 == 0).astype(int)
offsets = np.r_[0,np.cumsum(flags)[:-1]]
out = np.empty(int(flags.sum()), dtype=x.dtype)
for i in range(n):
    if flags[i]: out[offsets[i]] = x[i]
print("inclusive=", a.tolist())
print("exclusive=", b.tolist())
print(f"doubling_additions={doubling_adds}, tree_additions={adds}")
print("compacted_even_values=", out.tolist())
```

The recorded run produced the following output.

```text
inclusive= [3, 4, 8, 9, 14, 23, 25, 31]
exclusive= [0, 3, 4, 8, 9, 14, 23, 25]
doubling_additions=17, tree_additions=14
compacted_even_values= [4, 2, 6]
```

**Both scans produce the expected prefixes, but their work counts differ.** The eight-element doubling scan uses 17 additions, while the tree uses 14. The asymptotic difference becomes more important as the array grows.

**Try it 1 :class: exercise.**

Extend the input to 16 elements and compare addition counts. Derive the doubling count by summing the number of active positions at each distance. :::

**Try it 2 :class: exercise.**

Replace the even-value predicate with a count from zero to three per item. Use an exclusive scan to assign disjoint ranges and verify that the ranges exactly fill the output. :::

## Summary

| Decision | What you should remember |
|---|---|
| Inclusive scan | Each result includes its current input. |
| Exclusive scan | Each result gives the aggregate before its current input. |
| Work efficiency | A shallow algorithm can still perform too many total operations. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the first exclusive sum?**

1. It is the identity zero.
2. It is the first input value.

**Answer.** It is the identity zero. No input precedes position zero.

**Question 2. What work does iterative doubling require asymptotically?**

1. It requires O(N log N) work.
2. It requires O(1) work.

**Answer.** It requires O(N log N) work. Many positions combine at each logarithmic stage.

**Question 3. How many additions does the eight-element tree perform here?**

1. It performs 14 additions.
2. It performs 64 additions.

**Answer.** It performs 14 additions. The up-sweep and down-sweep each use seven additions.

**Question 4. What does a compaction scan compute?**

1. It computes output positions for retained items.
2. It sorts all values by magnitude.

**Answer.** It computes output positions for retained items. The prefix of keep flags assigns each retained item a unique stable position.

**Question 5. Can a block boundary automatically reset a segmented scan?**

1. Yes, every block begins a new mathematical segment.
2. No, only a real segment flag permits the reset.

**Answer.** No, only a real segment flag permits the reset. A segment can cross a block boundary.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 11. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 40. Parallel merge, co-ranking, and sorting patterns

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [39-prefix-scan](https://arunshar.github.io/numerical-parallel-computing/lectures/39-prefix-scan.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will partition a merge by output positions using co-ranks.
- You will explain stable sorting and tie rules.
- You will compare radix sorting with comparison-based sorting. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/40_merge_sort.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Output ownership makes merge parallel

A **merge** combines two sorted arrays into one sorted array. The serial algorithm repeatedly compares the next unused elements and emits the smaller one. Its output is easy to define, but a single pair of moving pointers creates a serial dependency.

A parallel design divides the output into ranges. For each range boundary $k$, it finds how many items come from the first input and how many come from the second. If those counts are $i$ and $j$, then $i+j=k$. The pair is the boundary’s **co-rank**. Once adjacent co-ranks are known, a worker can merge the corresponding slices independently.

This design balances output volume directly. Splitting only one input into equal pieces can leave workers with very different amounts from the other input. Output partitioning assigns nearly the same number of produced elements even when one input contains mostly small values.

## A co-rank is a binary-search condition

Suppose ties should take values from array $A$ before equal values from array $B$. A valid boundary satisfies the conditions

$$i=0\ \text{or}\ j=|B|\ \text{or}\ A_{i-1}\leq B_j,$$

$$j=0\ \text{or}\ i=|A|\ \text{or}\ B_{j-1}<A_i.$$

The strict inequality in the second condition encodes the tie rule. The counts range from $\max(0,k-|B|)$ to $\min(k,|A|)$ for $i$, with $j=k-i$. Binary search adjusts $i$ until both conditions hold.

A duplicated value is therefore a useful correctness test. An implementation can return numerically sorted values while violating stability. **Stability** means that equal-key elements retain their original relative order under the stated input ordering. It matters when records carry payloads or when multiple stable sorting passes are composed.

## Tiling improves the access pattern after partitioning

Co-ranking determines ownership, but it does not automatically make loads efficient. A tiled merge loads pieces of the input into shared memory and lets threads merge from the local tile. The tile boundary must account for how many values each input contributes. A circular buffer can preserve unused local values while loading the next input portion, reducing repeated movement.

Thread coarsening gives one thread several output values. It can reduce partitioning overhead and use a short serial merge within the assigned range. The tradeoff is less fine-grained parallelism and more per-thread state. A useful implementation chooses a granularity large enough to amortize binary searches but small enough to distribute irregular work.

A merge sort repeatedly merges sorted runs. The first runs can be tiny per-thread or per-block sorts, and later passes merge larger runs. Each pass reads and writes much of the array, so global traffic becomes a major cost. A comparison-based merge sort applies to a broad class of ordered keys, including compound comparison rules.

## Radix sorting uses key digits

A **radix sort** groups keys by a fixed-width digit. A least-significant-digit version processes low digits first and requires each pass to be stable. A later pass preserves the ordering already established by earlier digits within equal current digits.

A digit pass typically computes bucket counts, scans them to get bucket offsets, determines each key’s local rank within its bucket, and scatters the keys. The histogram, scan, and scatter patterns therefore become building blocks. The algorithm avoids arbitrary key comparisons but depends on a representation whose digit ordering matches the desired key ordering.

A wider radix reduces the number of passes but increases the number of buckets and local counters. A narrow radix increases passes and global traffic. Signed integers and floating-point keys require transformations or special handling so that bit order matches numerical order. NaNs and signed zero require an explicit ordering policy rather than an accidental consequence of their bit patterns.

## Sorting networks trade generality for regularity

A bitonic sorting network performs a fixed sequence of compare-exchange operations. Its structure is regular and can suit small blocks. It uses more comparisons than the best comparison sorts for large inputs, but it avoids data-dependent control flow in its network structure. Odd-even methods provide another regular comparison pattern.

A hybrid implementation can sort small tiles with a network and merge them globally. The fastest method depends on input length, key type, payload size, and whether stability is required. A library sort should be the practical baseline because a hand-written implementation must handle these edge cases as well as the main kernel.

The reference verifies every co-rank against a stable merged prefix and uses the boundaries to form two output chunks. Its small least-significant-digit radix example sorts nonnegative integers. It intentionally does not pretend to solve signed or floating-point ordering.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [40_merge_sort.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/40_merge_sort.py).

```python
a, b = [1,3,3,8], [2,3,7,9]
def co_rank(k, a, b):
    low, high = max(0,k-len(b)), min(k,len(a))
    while low <= high:
        i = (low+high)//2; j = k-i
        if i > 0 and j < len(b) and a[i-1] > b[j]: high = i-1
        elif j > 0 and i < len(a) and b[j-1] >= a[i]: low = i+1
        else: return i,j
    raise AssertionError("No valid co-rank")
merged = sorted([(v,0,i) for i,v in enumerate(a)] + [(v,1,i) for i,v in enumerate(b)])
for k in range(len(a)+len(b)+1):
    i,j = co_rank(k,a,b)
    assert i == sum(origin==0 for _,origin,_ in merged[:k])
    assert i+j == k
print("middle_co_rank=", co_rank(4,a,b))
print("merged_values=", [v for v,_,_ in merged])
values = [13,2,9,0,15,7]
for shift in [0,2]:
    buckets = [[] for _ in range(4)]
    for value in values: buckets[(value >> shift)&3].append(value)
    values = [value for bucket in buckets for value in bucket]
assert values == [0,2,7,9,13,15]
print("two_pass_radix_result=", values)
```

The recorded run produced the following output.

```text
middle_co_rank= (3, 1)
merged_values= [1, 2, 3, 3, 3, 7, 8, 9]
two_pass_radix_result= [0, 2, 7, 9, 13, 15]
```

**The middle co-rank splits the output after four values while respecting the A-before-B tie rule.** The radix example uses two stable two-bit passes to sort four-bit nonnegative keys.

**Try it 1 :class: exercise.**

Reverse the tie convention and derive which inequality must become strict. Test the result with payloads that identify the original array and position. :::

**Try it 2 :class: exercise.**

Compare one four-bit radix pass with two two-bit passes. Count buckets and full-array passes, then explain why neither count alone determines runtime. :::

## Summary

| Decision | What you should remember |
|---|---|
| Co-ranking | Output boundaries determine matching input prefixes. |
| Stability | Equal-key ordering must be preserved when the contract requires it. |
| Radix choice | More buckets reduce passes but increase local bookkeeping. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What identity links a co-rank to output position k?**

1. The counts satisfy i+j=k.
2. The counts satisfy i*j=k.

**Answer.** The counts satisfy i+j=k. The first k outputs contain i elements from A and j from B.

**Question 2. Why does the tie rule use one strict inequality?**

1. It chooses which input’s equal elements come first.
2. It changes the numerical key values.

**Answer.** It chooses which input’s equal elements come first. The boundary must agree with the stable merge convention.

**Question 3. What must a least-significant-digit radix pass preserve?**

1. It must preserve stability.
2. It must reverse equal keys.

**Answer.** It must preserve stability. Later passes rely on ordering established by earlier digits.

**Question 4. Does the reference handle arbitrary floating-point keys?**

1. Yes, raw float bits always sort numerically.
2. No, it sorts a restricted nonnegative integer example.

**Answer.** No, it sorts a restricted nonnegative integer example. Signed values and floating-point special cases need additional rules.

**Question 5. Why can a sorting network help small tiles?**

1. Its compare-exchange structure is regular.
2. It removes the need to compare values.

**Answer.** Its compare-exchange structure is regular. The network fixes the operation pattern while still comparing keys.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 12, 13. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 41. Sparse matrices and graph traversal on GPUs

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [40-merge-sort](https://arunshar.github.io/numerical-parallel-computing/lectures/40-merge-sort.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will interpret COO, CSR, ELL, hybrid, and JDS storage choices.
- You will derive a sparse matrix-vector product from row offsets.
- You will compare push and pull graph traversal using frontiers. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/41_sparse_graphs.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Sparse data saves storage but introduces irregularity

A **sparse matrix** stores only a small fraction of its entries explicitly. The usual motivation is to avoid spending memory and arithmetic on zeros. A dense $m\times n$ matrix stores $mn$ values. A sparse representation stores the nonzero values plus metadata that identifies their locations. The metadata and irregular access can dominate when each stored value supports only a small amount of arithmetic.

The sparse matrix-vector product computes $y_i=\sum_j A_{ij}x_j$. Each nonzero contributes one multiplication and one addition. A float32 value, a 32-bit column index, and a float32 vector read already suggest substantial traffic for two operations. Cache reuse can reduce vector traffic, but the index traffic remains part of the representation.

**COO**, or coordinate storage, keeps row indices, column indices, and values. It is convenient for construction and streaming individual nonzeros. Multiple nonzeros from one row can require a reduction or atomic updates to the output. Sorting or grouping coordinates can improve locality but costs preprocessing.

## Row formats trade balance against padding

**CSR**, or compressed sparse row storage, keeps a row-offset array, column indices, and values. Row $i$ occupies positions `row_ptr[i]` through `row_ptr[i+1]-1`. A one-thread-per-row kernel has simple ownership but can suffer when row lengths vary widely. A one-warp-per-row kernel distributes a long row across lanes, while short rows can leave many lanes idle.

**ELL** stores the same number of slots for every row, padding short rows to the longest stored width. Its regular layout can improve coalescing, but one exceptionally long row can waste much of the storage. A hybrid ELL-COO format keeps a regular prefix in ELL and sends the overflow to COO.

**JDS**, or jagged diagonal storage, sorts rows by length and groups their first nonzeros, second nonzeros, and later nonzeros into jagged diagonals. This layout reduces padding relative to a fixed rectangular ELL layout while grouping similar work. It requires a permutation to restore original row ordering. No format is best independently of the matrix’s row-length distribution and the operation being performed.

The preprocessing cost belongs in the application model. A format conversion can pay off when the same matrix is multiplied by many vectors. It can be wasteful when the sparsity pattern changes every iteration or the matrix is used only once.

## A graph turns matrix structure into neighbors

A graph’s adjacency matrix has a nonzero wherever an edge exists. CSR therefore also stores outgoing adjacency lists. A **breadth-first search**, or BFS, assigns each reachable vertex its shortest unweighted edge distance from a source. The algorithm advances one level at a time through a **frontier** of newly reached vertices.

A push traversal visits outgoing edges of the current frontier. It tries to claim each neighbor that has not yet been visited. Several frontier vertices can discover the same neighbor, so the visited-state update needs a race-safe mechanism. A vertex can be enqueued once through an atomic claim, or duplicates can be generated and removed later. The choice changes contention and memory traffic.

A pull traversal examines unvisited vertices and asks whether any predecessor lies in the current frontier. It can stop as soon as it finds one. Pull is attractive when the frontier is large, while push is attractive when the frontier is small. Directed graphs may need both outgoing and incoming adjacency structures to support both directions efficiently.

## Work assignment must follow edge volume

A one-thread-per-vertex mapping is simple but can be badly imbalanced. A hub vertex may have thousands of edges while most vertices have only a few. Edge-centric work distributes edges more evenly but needs a way to determine each edge’s source and its frontier membership. A hybrid design assigns short adjacency lists to threads and long lists to warps or blocks.

A frontier can be stored as a compact list or as a bitmap. A list makes iteration over a small frontier efficient. A bitmap makes membership tests cheap and can suit pull traversal. Converting representations adds work, so a direction-switching policy should include the conversion cost.

Privatized output queues can reduce pressure on one global queue counter. A block gathers discoveries locally and reserves one larger global range. It then copies the local discoveries into that range. This technique connects graph traversal to the histogram and scan lessons, but it still requires correct visited-state handling.

## Sparse correctness tests need structural cases

A sparse matrix test should include empty rows, one-entry rows, duplicate coordinates under a stated combine rule, and highly uneven row lengths. A graph test should include disconnected vertices, self-loops, duplicate edges, and multiple paths to the same vertex. A random graph alone rarely exercises every important edge case.

The experiment compares CSR multiplication against a dense reference and performs BFS on a small graph. It also calculates ELL padding from row lengths. These checks establish mathematical agreement for the toy data and expose the storage tradeoff. They do not benchmark a GPU graph library or a sparse kernel.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [41_sparse_graphs.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/41_sparse_graphs.py).

```python
import numpy as np
row_ptr = np.array([0,2,2,5,6])
cols = np.array([0,3,0,1,3,2])
vals = np.array([2.,1.,-1.,3.,2.,4.])
x = np.array([1.,2.,3.,4.])
y = np.zeros(4); dense = np.zeros((4,4))
for row in range(4):
    for k in range(row_ptr[row],row_ptr[row+1]):
        y[row] += vals[k]*x[cols[k]]
        dense[row,cols[k]] += vals[k]
assert np.allclose(y,dense@x)
lengths = np.diff(row_ptr)
print("csr_product=", y.tolist())
print(f"nonzeros={len(vals)}, ell_slots={len(lengths)*max(lengths)}, padding={len(lengths)*max(lengths)-len(vals)}")
graph = [[1,2],[3],[3,4],[5],[5],[],[]]
distance = [-1]*len(graph); distance[0]=0; frontier=[0]
while frontier:
    following=[]
    for vertex in frontier:
        for neighbor in graph[vertex]:
            if distance[neighbor]<0:
                distance[neighbor]=distance[vertex]+1;following.append(neighbor)
    frontier=following
assert distance == [0,1,1,2,2,3,-1]
print("bfs_distances=",distance)
```

The recorded run produced the following output.

```text
csr_product= [6.0, 0.0, 13.0, 12.0]
nonzeros=6, ell_slots=12, padding=6
bfs_distances= [0, 1, 1, 2, 2, 3, -1]
```

**CSR produces the same vector as the dense reference while storing six nonzeros.** The modeled ELL layout allocates 12 slots and wastes six. BFS leaves the disconnected final vertex at distance minus one.

**Try it 1 :class: exercise.**

Add one row with 20 entries to a larger toy matrix. Compare ELL padding with a hybrid format that stores only four entries per row in ELL. :::

**Try it 2 :class: exercise.**

Construct a graph with one high-degree hub and many degree-one vertices. Estimate how a vertex-per-thread mapping and an edge-based mapping distribute work. :::

## Summary

| Decision | What you should remember |
|---|---|
| Sparse format | Metadata, padding, and row balance determine the useful representation. |
| Frontier | BFS processes newly discovered vertices one level at a time. |
| Traversal direction | Push and pull favor different frontier sizes and access patterns. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What does CSR row_ptr[i+1]-row_ptr[i] give?**

1. It gives the number of stored entries in row i.
2. It gives the total number of columns.

**Answer.** It gives the number of stored entries in row i. The offsets delimit the row’s nonzero interval.

**Question 2. How many padding slots does the toy ELL format use?**

1. It uses six padding slots.
2. It uses zero padding slots.

**Answer.** It uses six padding slots. Four rows times maximum length three gives 12 slots for six values.

**Question 3. Why can push BFS require atomic claims?**

1. Several parents can discover the same neighbor.
2. Every graph has negative weights.

**Answer.** Several parents can discover the same neighbor. The visited-state transition must handle competing discoveries.

**Question 4. What does distance minus one mean in the reference?**

1. The vertex is unreachable from the source.
2. The vertex has a negative-length path.

**Answer.** The vertex is unreachable from the source. The unweighted BFS leaves undiscovered vertices marked minus one.

**Question 5. When can preprocessing a sparse format pay off?**

1. It can pay off when the converted matrix is reused many times.
2. It always pays off even for one tiny operation.

**Answer.** It can pay off when the converted matrix is reused many times. Repeated use can amortize conversion cost.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 14, 15. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 42. Convolutional networks, GEMM, and library choices

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [41-sparse-graphs](https://arunshar.github.io/numerical-parallel-computing/lectures/41-sparse-graphs.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will map a convolutional layer to a matrix multiplication.
- You will account for the memory cost of an explicit patch matrix.
- You will separate numerical precision choices from an operation-count comparison. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/42_deep_learning.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## A layer is a structured collection of dot products

A convolutional layer takes an input tensor, a bank of filters, and often a bias. Each output channel applies one filter across the input’s channel and spatial dimensions. A batch dimension repeats the operation for several examples. The output shape depends on stride, padding, dilation, filter size, and input layout.

For one spatial dimension, the output length is

$$L_{out}=\left\lfloor\frac{L_{in}+2p-d(k-1)-1}{s}\right\rfloor+1,$$

where $p$ is padding, $d$ is dilation, $k$ is filter width, and $s$ is stride. The same formula applies separately to height and width. A kernel must implement the same convention as its reference framework, including whether the operation reverses the filter or uses cross-correlation.

A direct implementation assigns an output element to a thread and loops over channels and filter positions. This ownership rule is correct and easy to test. It can reread many input values and coefficients, which motivates reuse, tiling, and matrix-multiplication formulations.

## The patch matrix exposes GEMM

An **im2col** transformation turns each input patch into a column. If each patch has $K=C_{in}R S$ values and there are $P$ output positions across the batch, the patch matrix has shape $K\times P$. Flattened filters form a matrix with shape $C_{out}\times K$. Their product has shape $C_{out}\times P$ and can be reshaped into the output tensor.

The arithmetic is approximately $2C_{out}KP$ FLOP. A one-channel three-by-three filter gives $K=9$. A large image repeats many input values across neighboring patch columns. The explicit patch matrix can therefore be much larger than the original input.

An **implicit GEMM** implementation computes the patch addresses as needed and avoids materializing the complete patch matrix. It retains the matrix-multiplication structure while trading extra address calculations for less intermediate traffic. The best implementation depends on the convolution shape and the hardware’s favored tile structure.

## Other algorithms change the cost profile

A pointwise one-by-one convolution has no spatial neighborhood and often maps especially naturally to GEMM. A depthwise convolution processes each input channel separately and has much less cross-channel reuse. A kernel optimized for dense multi-channel convolution may therefore perform poorly on a depthwise layer.

FFT-based convolution transforms the input and filter, multiplies in the frequency domain, and transforms back. It can reduce arithmetic for sufficiently large filters, but it introduces transforms, padding, temporary storage, and numerical differences. Winograd-style algorithms reduce multiplication counts for selected small filters while increasing transformation work and sensitivity. A lower multiplication count is not enough to establish lower runtime or acceptable accuracy.

A high-quality library selects among implementations using the actual tensor shape, layout, workspace limit, precision, and target GPU. cuDNN is the primary CUDA deep-learning primitive library for such operations. A hand-written teaching kernel should be compared with an appropriate library baseline rather than treated as the default production implementation.

## Fusion removes intermediate traffic

A layer can apply a bias and activation after its convolution. Separate kernels write the convolution output, reread it for the bias, write it again, and reread it for the activation. A fused implementation can apply these operations while the value is still in a register and write only the final result.

Fusion can also enlarge the kernel and increase register demand. An operation that needs a global reduction, such as some normalization schemes, may not fit a simple local fusion. The dependency graph determines which intermediate values can remain local without changing the computation.

Training adds backward passes for inputs and weights. The weight-gradient calculation accumulates contributions across examples and output positions. This accumulation creates different parallelism and numerical-order issues from the forward pass. A complete training benchmark must include the actual forward, backward, optimizer, and communication work that belongs to the workload.

## Precision is a contract

Tensor-core paths can use specialized input and accumulation formats. The precise supported modes depend on the GPU and library version. A throughput comparison must report these formats and the allowed error. A faster low-precision path is useful only when it satisfies the application’s accuracy requirements.

A stable test uses more than one random tensor. It includes zeros, large and small magnitudes, unusual channel counts, nonmultiple tile shapes, and asymmetric filters. It checks both absolute and relative error and records the reference precision. The same shape and arithmetic convention must be used for every implementation being compared.

The reference constructs a small patch matrix and compares its product with direct cross-correlation. It reports the patch-storage expansion and the number of multiplication-addition operations. It does not claim a cuDNN or tensor-core benchmark.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [42_deep_learning.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/42_deep_learning.py).

```python
import numpy as np
x = np.arange(25,dtype=np.float64).reshape(5,5)
filters = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]],
                    [[0,1,0],[1,2,1],[0,1,0]]],dtype=float)
patches = np.column_stack([x[r:r+3,c:c+3].reshape(-1) for r in range(3) for c in range(3)])
y_gemm = (filters.reshape(2,9) @ patches).reshape(2,3,3)
y_direct = np.zeros((2,3,3))
for f in range(2):
    for r in range(3):
        for c in range(3):
            y_direct[f,r,c] = np.sum(filters[f]*x[r:r+3,c:c+3])
assert np.array_equal(y_gemm,y_direct)
print(f"input_values={x.size}, patch_values={patches.size}, expansion={patches.size/x.size:.2f}")
print(f"gemm_shape=(2,9)@(9,9), nominal_flops={2*2*9*9}")
print("first_output_channel=",y_gemm[0].astype(int).tolist())
```

The recorded run produced the following output.

```text
input_values=25, patch_values=81, expansion=3.24
gemm_shape=(2,9)@(9,9), nominal_flops=324
first_output_channel= [[-6, -6, -6], [-6, -6, -6], [-6, -6, -6]]
```

**The patch-matrix product exactly matches the direct calculation for these integer-valued inputs.** The patch matrix stores 81 values for a 25-value input, demonstrating the cost that implicit GEMM seeks to avoid.

**Try it 1 :class: exercise.**

Add stride two and derive the output dimensions before editing the patch extraction. Check that direct and GEMM formulations still agree. :::

**Try it 2 :class: exercise.**

Replace the filter bank with a one-by-one convolution over several channels. Explain which duplication disappears and which GEMM dimensions remain. :::

## Summary

| Decision | What you should remember |
|---|---|
| Convolution mapping | Each output is a dot product over a channel-spatial patch. |
| im2col | An explicit patch matrix exposes GEMM but duplicates input values. |
| Libraries | A production comparison should use the same shape, precision, and operation contract. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What does one im2col column represent?**

1. It represents one input patch.
2. It represents the entire training dataset.

**Answer.** It represents one input patch. Each column contains the values needed for one output position.

**Question 2. How many values are in the toy patch matrix?**

1. There are 81 values.
2. There are 25 values.

**Answer.** There are 81 values. Nine patches each contain nine values.

**Question 3. What does implicit GEMM avoid?**

1. It avoids materializing the full patch matrix.
2. It avoids every input load.

**Answer.** It avoids materializing the full patch matrix. The kernel generates patch addresses while retaining GEMM-like computation.

**Question 4. Does fewer multiplications always imply faster convolution?**

1. Yes, no other cost matters.
2. No, transformations and memory traffic also matter.

**Answer.** No, transformations and memory traffic also matter. Alternative algorithms can trade multiplication count for other work.

**Question 5. Why must precision be stated in a benchmark?**

1. Different formats can change both speed and numerical error.
2. Precision affects only file naming.

**Answer.** Different formats can change both speed and numerical error. The arithmetic contract determines whether two results are comparable.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 16. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/) describes the library interface and supported execution paths. The lesson’s NumPy experiment does not execute cuDNN.


---

# 43. MRI reconstruction and electrostatic potential maps

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [42-deep-learning](https://arunshar.github.io/numerical-parallel-computing/lectures/42-deep-learning.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will write a forward operator and its adjoint for a small Fourier inverse problem.
- You will separate a numerical reconstruction method from its expensive kernels.
- You will compare gather, scatter, and spatial cutoff designs for a potential map. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/43_mri_electrostatics.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Applications begin with an operator

An application kernel should come from a mathematical model. Magnetic resonance imaging measures samples related to the Fourier transform of a spatial signal. A simplified discrete model is $y=Fx+\epsilon$, where $x$ contains unknown image values, $F$ maps those values to measurements, and $\epsilon$ represents noise or modeling error. Real MRI adds coil sensitivities, acquisition geometry, and physical effects that this toy model omits.

A reconstruction can minimize

$$\phi(x)=\tfrac12\|Fx-y\|_2^2+\tfrac{\lambda}{2}\|x\|_2^2.$$

The gradient is $F^H(Fx-y)+\lambda x$, where $F^H$ is the conjugate transpose, or **adjoint**, of the forward operator. The adjoint is not generally an inverse. Confusing these two objects produces an incorrect iterative method even if each GPU kernel is fast.

A gradient step is $x_{k+1}=x_k-\eta\nabla\phi(x_k)$. The step size depends on the operator’s spectrum. More sophisticated methods use conjugate gradients on suitable normal equations, proximal operators for nonsmooth regularizers, or other problem-specific algorithms. The reconstruction method should be selected before optimizing its individual kernels.

## The adjoint identity is a strong test

A forward-adjoint pair should satisfy $\langle Fx,z\rangle=\langle x,F^Hz\rangle$ under the chosen complex inner-product convention. This **adjoint test** catches indexing, conjugation, scaling, and interpolation errors. It is especially valuable when a nonuniform Fourier operator is implemented through gridding or interpolation.

Uniformly spaced Fourier samples can use FFT algorithms. Nonuniform samples may require direct evaluation or a nonuniform FFT approximation. A direct kernel computes complex exponentials and sums contributions, often with substantial reuse of coordinates. A faster approximation introduces parameters that control accuracy and work. The performance report must retain that approximation setting.

A GPU implementation can accelerate repeated forward and adjoint applications, but moving the image and measurements between host and device every iteration can erase much of the gain. Keeping the iterative state resident on the device often matters as much as optimizing one arithmetic kernel.

## A potential map is a sum over sources

An electrostatic potential at target position $r$ has a simplified form

$$V(r)=\sum_j\frac{q_j}{\|r-r_j\|}.$$

The physical constant is omitted here because the lesson focuses on computation. A target coinciding with a point source creates a singularity. A finite-size source model, exclusion rule, or softened denominator must be stated explicitly. Adding a small number to the denominator changes the model and is not merely a numerical implementation detail.

A **gather** kernel assigns one target to a thread and loops over sources. The output has one owner, so no atomic output updates are needed. Threads can cooperate to stage source positions and charges because many targets use the same source batch. The resulting structure resembles tiled matrix multiplication, although the interaction formula is different.

A **scatter** kernel assigns sources to workers and adds their contributions to targets. This approach can expose useful source-centered locality but creates competing writers. Atomic updates, private accumulators, or later reductions are needed. Gather and scatter compute the same sum only when they use the same interaction set and arithmetic model.

## Cutoffs reduce work by changing the interaction set

A spatial cutoff ignores sources farther than a chosen radius. A grid of spatial bins can identify nearby sources efficiently. The target then visits only neighboring bins whose geometry intersects its cutoff region. The work depends on source density and the radius, so an uneven distribution can create load imbalance.

A cutoff is justified only when the model or an error-controlled approximation permits it. Long-range electrostatic interactions do not vanish automatically. Methods such as multipole expansions or particle-mesh techniques handle long-range effects with different approximations. This lecture surveys that design choice without presenting a complete electrostatics solver.

Thread coarsening can give one thread several nearby targets. A loaded source can then contribute to several accumulators. This increases reuse but uses more registers. A structure-of-arrays layout for coordinates can improve coalesced source loads when neighboring lanes load neighboring sources. The geometry and the chosen loop nesting determine the actual address pattern.

## Numerical and performance validation belong together

An MRI test should check the forward-adjoint identity, objective decrease under a justified step size, and reconstruction error on a known synthetic signal. A potential-map test should compare gather and scatter on a small source-target set and exercise the chosen singularity policy. Both applications need tolerance-based comparisons because parallel accumulation can change rounding order.

The reference uses a tiny unitary discrete Fourier matrix and a regularized objective. It also computes a softened toy potential by gather and scatter. The softening parameter is part of the displayed model. These experiments explain application structure, but they are not clinical MRI validation or a physically calibrated molecular simulation.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [43_mri_electrostatics.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/43_mri_electrostatics.py).

```python
import numpy as np
n = 8
k = np.arange(n)
f = np.exp(-2j*np.pi*np.outer(k,k)/n)/np.sqrt(n)
x_true = np.array([0,0,1,2,1,0,0,0],dtype=complex)
y = f@x_true
x = np.zeros(n,dtype=complex); lam=0.1; eta=0.5
objective = lambda x: 0.5*np.linalg.norm(f@x-y)**2 + 0.5*lam*np.linalg.norm(x)**2
initial = objective(x)
for _ in range(30): x -= eta*(f.conj().T@(f@x-y)+lam*x)
z = np.arange(n)+1j*np.arange(n)[::-1]
adjoint_error = abs(np.vdot(f@x_true,z)-np.vdot(x_true,f.conj().T@z))
assert objective(x)<initial and adjoint_error<1e-12
sources=np.array([[0.,0.],[1.,0.]])
charges=np.array([1.,-0.5]);targets=np.array([[0.,1.],[1.,1.],[2.,1.]])
softening=0.1
gather=np.array([sum(q/np.sqrt(np.sum((r-s)**2)+softening**2) for s,q in zip(sources,charges)) for r in targets])
scatter=np.zeros(len(targets))
for s,q in zip(sources,charges):
    scatter += q/np.sqrt(np.sum((targets-s)**2,axis=1)+softening**2)
assert np.allclose(gather,scatter)
print(f"objective_initial={initial:.6f}, objective_final={objective(x):.6f}")
print(f"adjoint_identity_error={adjoint_error:.3e}")
print("softened_potential=",np.round(gather,6).tolist())
```

The recorded run produced the following output.

```text
objective_initial=3.000000, objective_final=0.272727
adjoint_identity_error=0.000e+00
softened_potential= [0.642364, 0.207827, 0.094094]
```

**The reconstruction objective decreases, and the forward-adjoint identity holds within the stated tolerance.** The potential implementations agree because they use the same softened interaction. The reported values belong only to these synthetic inputs.

**Try it 1 :class: exercise.**

Remove half of the Fourier measurements and compare the reconstruction with and without regularization. Explain why missing data changes the inverse problem even when the forward kernel remains correct. :::

**Try it 2 :class: exercise.**

Place a target exactly on a source. Compare a stated exclusion policy with the softened model and explain why the results answer different physical questions. :::

## Summary

| Decision | What you should remember |
|---|---|
| MRI operator | The forward map predicts measurements, while its adjoint supports optimization. |
| Gather and scatter | The ownership pattern determines whether output updates conflict. |
| Approximation | A cutoff or softening parameter changes the mathematical model. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Is an adjoint always an inverse?**

1. Yes, the words mean the same operation.
2. No, the adjoint is defined by an inner-product identity.

**Answer.** No, the adjoint is defined by an inner-product identity. An inverse solves the forward equation, while an adjoint transposes and conjugates the linear map.

**Question 2. Which potential design has one owner per target?**

1. Gather has one owner per target.
2. Scatter always has one owner per target.

**Answer.** Gather has one owner per target. A gather worker accumulates all sources for its own target.

**Question 3. What does softening do here?**

1. It changes the interaction formula near a source.
2. It only changes the file format.

**Answer.** It changes the interaction formula near a source. The denominator includes the stated softening parameter.

**Question 4. What should a valid gradient step usually reduce in this test?**

1. It should reduce the stated objective.
2. It should increase the number of measurements.

**Answer.** It should reduce the stated objective. The experiment checks optimization progress for its chosen step size.

**Question 5. Can a cutoff ignore long-range interactions without justification?**

1. Yes, distant interactions are always zero.
2. No, the physical or approximation model must justify it.

**Answer.** No, the physical or approximation model must justify it. The omitted interactions can contribute materially to the potential.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 17, 18. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.


---

# 44. CUDA streams, communication overlap, and GPU clusters

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [43-mri-electrostatics](https://arunshar.github.io/numerical-parallel-computing/lectures/43-mri-electrostatics.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will draw dependencies between transfers, kernels, and communication.
- You will derive the ideal throughput of a three-stage chunk pipeline.
- You will explain what CUDA-aware MPI does and what it does not guarantee. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/44_streams_clusters.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Computational thinking starts before a kernel

A parallel application begins with the required result, the data representation, and the dependencies among operations. An algorithm that minimizes arithmetic may still move too much data. A representation that makes one kernel fast may require expensive conversion elsewhere. **Computational thinking** means choosing the problem decomposition and algorithm with the complete application in view.

A GPU cluster adds several levels of parallelism. Threads cooperate inside a block, blocks execute on a GPU, GPUs communicate within a node, and nodes exchange messages. The cost and synchronization rules differ at each level. A block barrier cannot replace a message completion, and an MPI barrier does not automatically synchronize unrelated GPU streams.

The cleanest design first identifies ownership. Each rank owns a subdomain or tensor shard. Each GPU kernel owns specific outputs within that shard. Halo exchanges, reductions, or all-to-all operations then connect the ownership regions. The program can overlap independent work only after these dependencies are explicit.

## A stream is an ordered sequence

A **CUDA stream** orders the operations submitted to it. Operations in different streams may overlap when dependencies and hardware resources permit. The word may matters. Separate streams do not create extra bandwidth, copy engines, or compute units.

A chunk pipeline can copy chunk $k+1$ to the GPU while computing chunk $k$ and copying result $k-1$ back. Buffers must remain valid until their operations finish. A program that overwrites a host buffer immediately after an asynchronous copy can corrupt the transfer. Events can connect a producer operation in one stream to a consumer in another without blocking the CPU unnecessarily.

Pinned host memory is normally needed for genuinely asynchronous host-device transfers that can overlap with kernels in the usual CUDA pipeline design. Pinned memory has allocation and system-resource costs, so a reusable buffer pool is often better than repeatedly pinning small arrays. The exact overlap behavior also depends on the device and transfer path.

## A simple pipeline model exposes the bottleneck

Let upload, compute, and download times per chunk be $U$, $C$, and $D$. Processing $M$ chunks sequentially costs $M(U+C+D)$. An ideal three-stage pipeline with independent resources costs approximately

$$T_{pipe}=U+C+D+(M-1)\max(U,C,D).$$

The first chunk fills the pipeline, and later chunks emerge at the slowest-stage rate. A toy system with $U=2$ ms, $C=5$ ms, and $D=2$ ms needs 72 ms for eight sequential chunks. The ideal pipeline takes 44 ms. These values are modeled examples, not measurements.

The model assumes full overlap, equal chunk sizes, negligible launch overhead, and independent stage resources. Real transfers may share a link or engine, and a memory-bound kernel may compete with copy traffic. Smaller chunks improve pipeline granularity but increase launch and message overhead. Larger chunks amortize overhead but reduce overlap opportunities.

## MPI needs a completed producer and a valid consumer

A CUDA-aware MPI implementation accepts device buffers for supported communication operations. This feature can remove explicit application staging through host memory. It does not imply that every operation uses a direct GPU-to-GPU path or that every outstanding CUDA stream is automatically ordered with MPI.

A send must not read a device buffer before its producing kernel has completed under the MPI implementation’s documented integration rules. A receive buffer must not be consumed by a kernel before the receive is complete and visible to the device. Some environments provide stream-aware extensions, while a portable baseline uses explicit completion points. The program must follow the actual implementation’s contract.

Nonblocking MPI allows the application to initiate a transfer and perform independent work before waiting. It does not guarantee unlimited background progress. Progress can depend on the MPI implementation, network, runtime calls, and configuration. A measured overlap experiment must therefore compare a real trace or completed timings with the intended dependency graph.

## Domain and tensor decompositions create different messages

A stencil decomposition exchanges boundary halos. A distributed matrix product broadcasts panels or shifts tiles. Data-parallel neural-network training replicates model parameters and combines gradients. Tensor parallelism shards an operation and communicates intermediate tensors. Pipeline parallelism assigns different layers or stages to devices, while context parallelism partitions sequence-related work and its required communication.

These names describe ownership and dependency choices, not automatic speedups. Data parallelism can become communication-limited as local batches shrink. Pipeline parallelism has fill and drain bubbles. Tensor parallelism can require frequent collectives on a latency-sensitive path. A hybrid design combines the techniques only when the workload and network justify the added complexity.

The reference evaluates the ideal chunk-pipeline formula and a serial-versus-overlapped halo model. It intentionally uses declared stage times. The output teaches scheduling arithmetic and establishes no claim about local CUDA or MPI overlap.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [44_streams_clusters.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/44_streams_clusters.py).

```python
upload, compute, download = 2.0, 5.0, 2.0
for chunks in [1,2,4,8]:
    serial=chunks*(upload+compute+download)
    ideal=upload+compute+download+(chunks-1)*max(upload,compute,download)
    print(f"chunks={chunks}, sequential={serial:.1f} ms, ideal_pipeline={ideal:.1f} ms, speedup_bound={serial/ideal:.3f}")
interior, halo_exchange, boundary = 6.0,4.0,2.0
serial=interior+halo_exchange+boundary
overlap=max(interior,halo_exchange)+boundary
assert overlap==8.0
print(f"halo_sequential={serial:.1f} ms, halo_ideal_overlap={overlap:.1f} ms")
```

The recorded run produced the following output.

```text
chunks=1, sequential=9.0 ms, ideal_pipeline=9.0 ms, speedup_bound=1.000
chunks=2, sequential=18.0 ms, ideal_pipeline=14.0 ms, speedup_bound=1.286
chunks=4, sequential=36.0 ms, ideal_pipeline=24.0 ms, speedup_bound=1.500
chunks=8, sequential=72.0 ms, ideal_pipeline=44.0 ms, speedup_bound=1.636
halo_sequential=12.0 ms, halo_ideal_overlap=8.0 ms
```

**The eight-chunk ideal pipeline takes 44 ms under the stated independent-stage assumptions.** The halo example reduces its modeled critical path from 12 to 8 ms because interior work does not depend on the incoming halo.

**Try it 1 :class: exercise.**

Give uploads and downloads one shared copy resource. Draw a new schedule and explain why the three-independent-stage formula may be too optimistic. :::

**Try it 2 :class: exercise.**

Increase the number of pipeline stages in a layer-partitioned model while holding the number of microbatches fixed. Explain how fill and drain bubbles affect utilization. :::

## Summary

| Decision | What you should remember |
|---|---|
| Streams | A stream orders its own operations, and cross-stream dependencies need explicit coordination. |
| Overlap | Only independent work can safely overlap communication. |
| Cluster design | The decomposition determines message volume, latency sensitivity, and ownership. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What is the ideal eight-chunk time in the toy model?**

1. It is 44 ms.
2. It is 72 ms.
3. It is 9 ms.

**Answer.** It is 44 ms. The first chunk costs 9 ms, and seven later intervals cost 5 ms each.

**Question 2. Do two streams guarantee concurrent execution?**

1. Yes, streams create unlimited resources.
2. No, dependencies and hardware resources still limit overlap.

**Answer.** No, dependencies and hardware resources still limit overlap. Concurrency is permitted but not guaranteed.

**Question 3. What does CUDA-aware MPI primarily allow?**

1. It allows supported MPI operations to accept device buffers.
2. It guarantees that every kernel is automatically synchronized.

**Answer.** It allows supported MPI operations to accept device buffers. Buffer support does not replace the implementation’s ordering contract.

**Question 4. Which stencil work can overlap an incoming halo?**

1. Interior work that does not read that halo can overlap.
2. Every boundary computation can always overlap.

**Answer.** Interior work that does not read that halo can overlap. Boundary work must wait when it depends on the new halo.

**Question 5. Does nonblocking MPI guarantee independent background progress?**

1. Yes, the name guarantees it.
2. No, progress depends on the implementation and environment.

**Answer.** No, progress depends on the implementation and environment. A real overlap claim needs measurement and the runtime’s documented behavior.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 19, 20. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) describes GPU collective communication. NCCL and MPI provide different interfaces, and neither removes the application’s dependency requirements.


---

# 45. Dynamic parallelism, libraries, and accelerator evolution

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [44-streams-clusters](https://arunshar.github.io/numerical-parallel-computing/lectures/44-streams-clusters.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will explain when device-launched child work can express an irregular problem.
- You will compare dynamic launches with queues and precomputed work lists.
- You will choose a library, compiler layer, and profiler for a stated question. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/45_dynamic_parallelism_future.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## Irregular work can discover more work

Some computations do not know their full task graph at launch time. An adaptive subdivision can split a region only when its error estimate is too large. A tree traversal can discover children based on the current node. **CUDA dynamic parallelism** allows eligible device code to launch child grids, subject to the supported runtime model and target platform.

A child launch expresses nested work, but it has overhead and resource limits. Launching one child for every tiny task can cost more than processing those tasks in a loop or queue. The useful question is whether each child contains enough parallel work to justify a launch and whether dynamic discovery avoids substantial unnecessary computation.

The parent-child execution and synchronization rules have evolved across CUDA versions. Device-side synchronization patterns from an older textbook example must be checked against current documentation and the target architecture. A new implementation should not copy a legacy `cudaDeviceSynchronize` call inside a kernel and assume that it remains supported everywhere.

## Adaptive subdivision needs a stopping rule

A quadtree splits a two-dimensional region into four children. A Bezier-curve tessellation can subdivide a parameter interval until a geometric error estimate falls below a tolerance. Both algorithms need a termination condition, a maximum depth, and a rule for handling work that reaches the depth limit without satisfying the tolerance.

The task tree can grow rapidly. A fully expanded binary tree of depth $d$ has $2^{d+1}-1$ nodes. A quadtree grows even faster. A realistic implementation therefore needs bounded storage and a strategy for handling uneven work. Dynamic parallelism changes how work is launched, but it does not remove the combinatorial growth of the underlying tree.

A CPU reference can verify that the subdivision rule covers the original domain without overlaps or gaps. It can also count leaves, internal nodes, and the number of hypothetical child launches. Those counts reveal launch granularity before GPU execution is attempted.

## A work queue is an alternative

A **persistent kernel** keeps workers active while they obtain tasks from a queue. It can amortize launch costs and support irregular task sizes. The queue introduces atomic coordination, memory-ordering requirements, load-balance questions, and termination detection. A worker that observes an empty queue cannot always conclude that the computation is complete because another worker may still be producing tasks.

A breadth-first work-list approach processes one tree level or frontier per kernel. It has simpler stage boundaries and can use scans to allocate space for the next frontier. It may require more global passes than a dynamic child-launch design. The three approaches therefore trade launch overhead, queue complexity, and global traffic differently.

A practical comparison starts with a serial reference and a workload distribution. It then measures task sizes and queue or launch counts. An average task size can hide a long tail, so the distribution matters. A few very large tasks can dominate completion time even when the mean looks small.

## Libraries encode more engineering than one kernel

cuBLAS provides dense linear algebra, cuFFT provides Fourier transforms, cuSPARSE provides sparse operations, and cuDNN provides deep-learning primitives. CUTLASS supplies composable building blocks for high-performance linear algebra kernels. A library baseline is often the best starting point because it already handles many shapes, data types, layouts, and architecture-specific paths.

A tensor core is a specialized execution resource for supported matrix operations and formats. It is not a general replacement for every scalar floating-point operation. Tile shape, data layout, alignment, accumulation type, and numerical tolerance determine whether the operation fits an accelerated path. A result obtained with a different arithmetic mode should be labeled accordingly.

PTX is an intermediate virtual instruction representation in NVIDIA’s toolchain. SASS is the machine instruction representation for a target architecture. Inspecting generated code can reveal spills, vectorized loads, or unexpected instruction sequences. It should answer a concrete question rather than substitute for profiling. Compiler versions and target flags can change the generated code.

## Tools answer different levels of the performance question

Nsight Systems provides a system timeline that connects CPU activity, transfers, kernels, and communication. Nsight Compute examines individual kernel behavior in greater detail. A system trace can show that the GPU is idle between kernels, while a kernel report can explain poor memory efficiency during a busy interval. These findings call for different fixes.

CUDA Graphs can capture and replay an operation graph to reduce repeated submission overhead for suitable workloads. Kernel fusion reduces intermediate traffic and launches by combining operations. Triton offers a different programming abstraction for writing certain GPU kernels. These tools change developer choices, but they still obey the underlying data dependencies and hardware limits.

Apple MLX and AWS Trainium represent different accelerator ecosystems. Their programming interfaces, memory models, compiler stacks, and supported operations differ from CUDA. This lesson provides a comparison framework rather than claiming that CUDA kernel code runs unchanged on those platforms. The useful transferable skills are ownership, locality, dependency analysis, numerical validation, and measured cost models.

The experiment counts an adaptive binary task tree under a simple size-based stopping rule. It estimates the number of child-launch opportunities and compares leaf sizes. It is a scheduling model, not an implementation or benchmark of dynamic parallelism.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [45_dynamic_parallelism_future.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/45_dynamic_parallelism_future.py).

```python
def subdivide(size, threshold):
    pending=[size];leaves=[];internal=0
    while pending:
        current=pending.pop()
        if current<=threshold:leaves.append(current)
        else:
            internal+=1
            left=current//2;right=current-left
            pending.extend([left,right])
    return leaves,internal
for threshold in [8,32,128]:
    leaves,internal=subdivide(1000,threshold)
    assert sum(leaves)==1000 and max(leaves)<=threshold
    print(f"threshold={threshold}, leaves={len(leaves)}, internal_nodes={internal}, "
          f"leaf_range={min(leaves)}..{max(leaves)}")
```

The recorded run produced the following output.

```text
threshold=8, leaves=128, internal_nodes=127, leaf_range=7..8
threshold=32, leaves=32, internal_nodes=31, leaf_range=31..32
threshold=128, leaves=8, internal_nodes=7, leaf_range=125..125
```

**A smaller stopping threshold creates more tasks and more potential launch overhead.** Every decomposition preserves the total work size 1000, but the task granularity changes. A real adaptive error estimator would produce a less regular tree.

**Try it 1 :class: exercise.**

Replace the size-based split condition with a synthetic error value that varies by region. Compare the largest leaf workload with the mean and discuss load imbalance. :::

**Try it 2 :class: exercise.**

Assign a hypothetical fixed launch cost and a per-element work cost. Find a threshold where child-launch overhead dominates, and clearly label the result as a model. :::

## Summary

| Decision | What you should remember |
|---|---|
| Dynamic launch | A device can discover child work, but launch overhead and supported semantics still matter. |
| Work queues | Persistent workers trade launch costs for queue and termination complexity. |
| Future systems | Ownership, locality, dependencies, and validation transfer across accelerator ecosystems. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does dynamic parallelism remove launch overhead?**

1. Yes, device launches are free.
2. No, child launches still have costs.

**Answer.** No, child launches still have costs. Small child tasks can spend more on launch overhead than useful work.

**Question 2. What must an adaptive subdivision include?**

1. It needs a termination rule and resource bounds.
2. It needs unlimited recursive depth.

**Answer.** It needs a termination rule and resource bounds. The task tree can grow rapidly without bounded stopping rules.

**Question 3. What does a temporarily empty task queue prove?**

1. It always proves global completion.
2. It may prove nothing if active workers can produce more tasks.

**Answer.** It may prove nothing if active workers can produce more tasks. Termination detection must account for active producers.

**Question 4. Which tool is suited to a CPU-GPU timeline?**

1. Nsight Systems is suited to that view.
2. A single arithmetic checksum is suited to that view.

**Answer.** Nsight Systems is suited to that view. The timeline shows transfers, gaps, kernels, and host activity.

**Question 5. Can CUDA code be assumed to run unchanged on every accelerator?**

1. Yes, all accelerators share the same interface.
2. No, platform interfaces and execution rules differ.

**Answer.** No, platform interfaces and execution rules differ. The general design principles transfer more broadly than source compatibility.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 21, 22, 23. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [CUDA dynamic-parallelism documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html) should be checked for the installed toolkit. The [CUTLASS repository](https://github.com/NVIDIA/cutlass), [Triton documentation](https://triton-lang.org/main/index.html), [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html), and [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) describe their own supported environments. This lesson surveys these ecosystems without claiming local execution.


---

# 46. GPU numerical reliability and a complete project workflow

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, [45-dynamic-parallelism-future](https://arunshar.github.io/numerical-parallel-computing/lectures/45-dynamic-parallelism-future.html), established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

**Learning goals.**
- You will derive the exact-real factorization of an all-pairs matrix-product sum.
- You will explain how floating-point reassociation changes the numerical contract.
- You will write a reproducible correctness and performance plan for a GPU project. :::

## Setup

You can use the Python environment described in [run-locally](https://arunshar.github.io/numerical-parallel-computing/getting-started/run-locally.html). The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/46_gpu_numerics_project.py
```

****The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::.**

## The best optimization can change the algorithm

The course’s final homework studies a sum of pairwise small matrix products. Let $A_i$ and $B_j$ be compatible matrices. The requested real-number result is

$$C=\sum_{i=0}^{N_A-1}\sum_{j=0}^{N_B-1} A_iB_j.$$

Distributivity gives

$$C=\left(\sum_iA_i\right)\left(\sum_jB_j\right).$$

This identity preserves the order of multiplication between an $A$ matrix and a $B$ matrix. It does not assume that arbitrary matrices commute. Every product still places $A_i$ on the left and $B_j$ on the right. The transformation groups sums before the matrix multiplication.

For square matrices of order $d$, the direct arithmetic grows roughly as $N_AN_Bd^3$. The factored approach first sums matrix entries and then multiplies two matrices, giving work on the order of $(N_A+N_B)d^2+d^3$. The change can dwarf a low-level tiling improvement. However, an assignment or benchmark may require a particular algorithm, restrict libraries, or judge a specific floating-point result. The mathematical observation does not grant permission to violate those rules.

## Floating-point distributivity is approximate

The factorization changes when rounding occurs. The direct method rounds each small product and accumulates many results. The factored method rounds the two matrix sums before multiplying them. Cancellation in either sum can make the factored result sensitive to accumulation error. The two paths can therefore disagree even when both are correctly implemented.

A **unit roundoff** describes the relative rounding scale for ordinary floating-point operations under a specified model. A common bound for a sequence of $n$ additions uses $\gamma_n=nu/(1-nu)$ when $nu<1$. The bound depends on the magnitudes being summed and can be pessimistic. It helps explain error growth, but it does not replace tests on the application’s data distribution.

Fused multiply-add computes a multiplication and addition with one final rounding under its arithmetic semantics. A separately rounded multiply followed by add can produce a different result. Compiler contraction, fast-math options, tensor-core formats, denormal handling, and accumulation precision therefore belong in the numerical contract.

## Accuracy, stability, and reproducibility are separate goals

**Accuracy** compares the computed result with a sufficiently trustworthy reference. **Stability** describes how an algorithm responds to rounding and perturbations. **Reproducibility** asks whether repeated or cross-platform runs return the same result under a defined scope. An algorithm can be accurate without being bitwise reproducible, and it can be reproducible while returning a consistently inaccurate answer.

A cancellation-heavy sum can have a tiny exact result compared with the magnitudes of its terms. Relative error measured only against that tiny result can be misleading without an absolute scale. A test should report both absolute and relative tolerances and should define what happens near zero. A wider-precision reference or a carefully justified analytical result is more useful than assuming that one arbitrary CPU order is exact.

A pairwise reduction changes the rounding path. Compensated accumulation tracks some lost low-order information at additional cost. A reproducible reduction can impose a fixed accumulation scheme or use more elaborate representations. The right choice depends on whether the application needs a tolerance, deterministic runs, or a stronger bitwise guarantee.

## A project needs a contract before a benchmark

The project specification should state the mathematical operation, supported shapes, layout, data types, permitted approximations, and output tolerance. It should also state which inputs are invalid and how they are handled. An optimized kernel must satisfy that contract before its timing is meaningful.

The first milestone is a trusted reference and a small collection of adversarial tests. The second is a simple correct GPU implementation. The third changes one mechanism at a time and records the hypothesis. The final report compares the winning implementation with the baseline and explains the remaining bottleneck.

Correctness cases should include nonmultiple dimensions, small inputs, zeros, mixed signs, very different magnitudes, and cancellation. Memory checking should inspect out-of-bounds access and synchronization problems on an actual CUDA system. A Python arithmetic reference cannot certify CUDA memory safety or race freedom.

## A benchmark needs a reproducible scope

A result record should identify the GPU model, driver, toolkit, compiler flags, source revision, input generator and seed, shape, arithmetic mode, and timing method. It should say whether transfers, allocations, initialization, and synchronization are included. Warmup and repeated trials should be specified, and the report should retain a distribution or variability summary rather than only one minimum.

A useful report includes a plot or table that varies problem size and shape. A single square matrix can hide tail handling, occupancy changes, or transfer overhead. An optimization that wins only at one large size can still be useful, but its claim should retain that scope.

The CPU experiment compares direct and factored expressions on small integer-valued matrices, then repeats the comparison on floating-point data. The integer-valued case provides an exact small check in float64 because its values remain safely representable. The floating-point comparison demonstrates tolerance-based agreement for one seeded sample, not universal equality.

## The handoff should preserve open work

A handoff for a later reviewer should name the tested CPU scripts, the unexecuted CUDA files, the source chapters, and the unresolved hardware checks. It should never replace an unverified GPU result with a projected speedup. This edition supplies a runnable CPU foundation and a native CUDA teaching suite, while keeping GPU compilation and execution explicitly unverified.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as [46_gpu_numerics_project.py](https://github.com/arunshar/numerical-parallel-computing/blob/main/labs/gpu/46_gpu_numerics_project.py).

```python
import numpy as np
rng=np.random.default_rng(11)
a=rng.integers(-3,4,size=(4,3,3)).astype(np.float64)
b=rng.integers(-3,4,size=(5,3,3)).astype(np.float64)
direct=sum((aa@bb for aa in a for bb in b),start=np.zeros((3,3)))
factored=a.sum(axis=0)@b.sum(axis=0)
assert np.array_equal(direct,factored)
print("integer_case_equal=",bool(np.array_equal(direct,factored)))
print("result=",direct.astype(int).tolist())
af=rng.normal(size=(17,3,3)).astype(np.float32)
bf=rng.normal(size=(19,3,3)).astype(np.float32)
df=np.zeros((3,3),dtype=np.float32)
for aa in af:
    for bb in bf: df += aa@bb
ff=af.sum(axis=0)@bf.sum(axis=0)
error=float(np.max(np.abs(df-ff)))
assert np.allclose(df,ff,rtol=1e-4,atol=1e-4)
print(f"float32_reassociation_max_difference={error:.8e}")
na,nb,d=512,512,4
print(f"direct_matrix_products={na*nb}, factored_matrix_products=1")
print(f"matrix_entries_to_sum={(na+nb)*d*d}")
```

The recorded run produced the following output.

```text
integer_case_equal= True
result= [[-12, -44, -16], [-27, -20, -5], [13, -28, -17]]
float32_reassociation_max_difference=7.62939453e-05
direct_matrix_products=262144, factored_matrix_products=1
matrix_entries_to_sum=16384
```

**The exact small integer case agrees, while the float32 paths can differ in low-order bits.** The operation-count reduction is an algebraic observation. It is not a measured speedup and does not establish whether a particular assignment permits the transformation.

**Try it 1 :class: exercise.**

Create matrices whose entrywise sums nearly cancel. Compare both expressions against a float64 reference and explain why one random normal sample cannot establish a universal error bound. :::

**Try it 2 :class: exercise.**

Write a one-page benchmark contract for this operation. Specify dimensions, precision, allowed transformations, tolerances, included costs, and the evidence needed before claiming a speedup. :::

## Summary

| Decision | What you should remember |
|---|---|
| Algebra | Distributivity can remove many products without assuming matrix commutativity. |
| Numerics | Changing the evaluation order changes rounding and requires validation. |
| Evidence | A GPU performance claim needs executed GPU measurements under a stated contract. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. Does the factorization assume AiBj equals BjAi?**

1. Yes, it assumes matrix commutativity.
2. No, it preserves the multiplication order.

**Answer.** No, it preserves the multiplication order. Distributivity groups the A sum on the left and the B sum on the right.

**Question 2. How many direct matrix products occur for 512 matrices in each list?**

1. There are 262144 products.
2. There are 1024 products.

**Answer.** There are 262144 products. The direct double loop contains 512 times 512 pairs.

**Question 3. Must the two float32 expressions be bitwise identical?**

1. Yes, real algebra guarantees identical rounding.
2. No, reassociation changes intermediate rounding.

**Answer.** No, reassociation changes intermediate rounding. The operations are grouped differently.

**Question 4. Does a passing CPU reference certify a CUDA kernel’s race freedom?**

1. Yes, mathematical agreement proves synchronization.
2. No, CUDA execution and memory checks are separate.

**Answer.** No, CUDA execution and memory checks are separate. The CPU script does not exercise GPU concurrency.

**Question 5. What should a handoff say about an unexecuted CUDA file?**

1. It should label GPU execution UNVERIFIED.
2. It should infer success from the source code’s appearance.

**Answer.** It should label GPU execution UNVERIFIED. An implementation and an executed validation result are different evidence states.


## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters A. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html) explains CUDA numerical behavior. The local `labs/gpu/native_suite.cu` is an original teaching implementation whose CUDA compilation and execution remain **UNVERIFIED**.


---

# 47. A numerical and parallel computing capstone

Prepared for Arun Sharma. Codex drafted this study edition.

The earlier lessons considered numerical accuracy, decomposition, communication, and hardware separately. This capstone puts those decisions into one small example. We solve a steady one-dimensional diffusion problem, check its error, and model how its iterative update could be divided among workers.

A useful solver needs three kinds of evidence. The discrete equation should approximate the intended continuous equation. The algorithm should solve the discrete system accurately. The implementation should perform the required work without invalid data dependencies. A speed claim becomes meaningful only after these conditions are checked.

**Learning goals.**
- You will separate discretization error from algebraic error.
- You will compare a partitioned update with a serial reference.
- You will estimate when communication limits strong scaling.
- You will write a reproducibility record that preserves the scope of each result.

## Setup

This notebook uses NumPy and the shared quiz helper. The helper is loaded locally when the notebook runs inside the repository. A fresh Colab session downloads the same public helper. The experiment runs on a CPU and does not launch MPI or CUDA.

**The partitioned update is a synchronous CPU simulation. It checks the ownership and halo arithmetic. It does not measure a distributed network or a GPU..**


```python
from pathlib import Path
import sys
import urllib.request
import importlib.util
import numpy as np

helper_url = "https://raw.githubusercontent.com/arunshar/numerical-parallel-computing/main/labs/tutorial_helpers.py"
candidates = [Path("../../labs/tutorial_helpers.py"), Path("labs/tutorial_helpers.py")]
helper = next((p.resolve() for p in candidates if p.exists()), None)
if helper is None:
    helper = Path("tutorial_helpers.py").resolve()
    urllib.request.urlretrieve(helper_url, helper)
spec = importlib.util.spec_from_file_location("tutorial_helpers", helper)
th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)
SYSTEM = {"interior_points": [15, 31, 63], "workers": 4, "iterations": 100}
```

## The continuous problem gives us an independent reference.

We solve the boundary-value problem

$$-u''(x)=\pi^2\sin(\pi x),\qquad 0<x<1,\qquad u(0)=u(1)=0.$$

The exact solution is $u(x)=\sin(\pi x)$. We chose this manufactured problem because its answer is known independently of the numerical solver. We can therefore measure an actual solution error, not just the residual of a discrete equation.

A uniform grid with $n$ interior points has spacing $h=1/(n+1)$. The centered difference approximation gives

$$\frac{-u_{i-1}+2u_i-u_{i+1}}{h^2}=\pi^2\sin(\pi x_i).$$

The matrix has a diagonal of $2/h^2$ and adjacent diagonals of $-1/h^2$. The boundary values are zero, so they add no contribution to the right-hand side. A nonzero boundary value would change the first or last equation and must be handled explicitly.

The centered formula has second-order truncation error for a sufficiently smooth solution. Halving the grid spacing should therefore reduce the leading discretization error by about a factor of four. That prediction depends on reaching the asymptotic regime and solving the discrete system accurately enough.

```python
def problem(n):
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    A = (2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
    b = np.pi**2 * np.sin(np.pi*x)
    return h, x, A, b

errors = []
for n in SYSTEM["interior_points"]:
    h, x, A, b = problem(n)
    u = np.linalg.solve(A, b)
    error = np.max(np.abs(u - np.sin(np.pi*x)))
    residual = np.linalg.norm(A @ u - b, ord=np.inf)
    errors.append(error)
    print(f"n={n:2d} h={h:.6f} solution_error={error:.8f} residual={residual:.3e}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print("successive error ratios:", np.round(ratios, 5))
assert np.all((ratios > 3.9) & (ratios < 4.1))
```

```text
n=15 h=0.062500 solution_error=0.00321896 residual=4.086e-14
n=31 h=0.031250 solution_error=0.00080358 residual=3.499e-13
n=63 h=0.015625 solution_error=0.00020082 residual=1.258e-12
successive error ratios: [4.00579 4.00145]

```

The error ratios should be close to **four**. The residual is much smaller than the solution error because the linear solver nearly satisfies the discrete equations. The remaining error comes mainly from replacing the continuous derivative with a finite difference.

A small residual alone would not demonstrate second-order accuracy. We needed the exact solution and a grid-refinement experiment. On a problem without an exact solution, we would need a different independent check, such as a trusted finer-grid result or a manufactured test problem with similar structure.

**Try it.**

You should replace the three grid sizes with 7, 15, and 31. Predict whether the ratios move closer to four as the grid becomes finer. Then change the forcing function without changing the claimed exact solution and observe why the reference check fails.

## A Jacobi update exposes the communication boundary.

A Jacobi iteration computes each new value from the previous iterate. The update is

$$u_i^{(k+1)}=\tfrac12\left(u_{i-1}^{(k)}+u_{i+1}^{(k)}+h^2b_i\right).$$

Each interior point needs two neighbors. A worker that owns a consecutive block already has most of those values. It needs one old value from each adjacent worker at an internal boundary. Those borrowed boundary values are called a **halo**.

The implementation below keeps the old array immutable during an iteration. Each worker writes a disjoint slice of a new array. This separation makes the update mathematically identical to the serial Jacobi step. Updating the old array in place would change the method and introduce an ordering dependency.

```python
n = 32
h, x, A, b = problem(n)
serial = np.zeros(n)
partitioned = np.zeros(n)
blocks = np.array_split(np.arange(n), SYSTEM["workers"])
for step in range(SYSTEM["iterations"]):
    padded = np.pad(serial, (1, 1))
    serial = 0.5 * (padded[:-2] + padded[2:] + h*h*b)
    old = np.pad(partitioned, (1, 1))
    new = np.empty_like(partitioned)
    for ids in blocks:
        start, stop = int(ids[0]), int(ids[-1]) + 1
        new[start:stop] = 0.5*(old[start:stop] + old[start+2:stop+2] + h*h*b[start:stop])
    partitioned = new
print("largest serial/partitioned difference:", np.max(np.abs(serial-partitioned)))
print("Jacobi residual after 100 updates:", np.linalg.norm(A @ partitioned-b, ord=np.inf))
assert np.array_equal(serial, partitioned)
```

```text
largest serial/partitioned difference: 0.0
Jacobi residual after 100 updates: 6.26194253903371

```

The two update paths should agree exactly in this implementation because each output uses the same arithmetic expression and operand order. This agreement is a correctness check for the partition logic. It does not demonstrate parallel execution or a speedup.

The residual after 100 iterations remains nonzero. That result is expected because Jacobi can converge slowly on a fine diffusion grid. A faster halo exchange would not fix its convergence rate. A better iterative method or preconditioner could reduce the number of communication rounds as well as the arithmetic work.

A real MPI implementation would exchange boundary values before computing the boundary cells. It could start nonblocking communication, compute cells that do not need the incoming values, wait for completion, and then compute the boundaries. The overlap depends on MPI progress and the available interior work. A nonblocking call alone does not prove useful overlap.

## A cost model predicts a scaling limit.

Suppose one update costs $c$ seconds per point and a halo exchange costs approximately $2\alpha+2\beta s$. Here $\alpha$ is a startup delay, $\beta$ is seconds per byte, and $s$ is the payload per neighbor. An interior worker's simple estimate is

$$T_p\approx cN/p+2\alpha+2\beta s.$$

This model ignores overlap, endpoint differences, network contention, and reductions used for convergence checks. It is a first diagnostic. The compute term shrinks with worker count, but the startup term does not. Strong scaling eventually runs out of enough work per worker to hide that cost.

The next cell uses invented teaching parameters. Its times are model predictions, not benchmark measurements.

```python
N = 1_000_000
c = 2e-9
alpha = 5e-6
beta = 1 / 10e9
s = 8
baseline = c*N
for workers in [1, 2, 4, 16, 64, 256]:
    communication = 0 if workers == 1 else 2*alpha + 2*beta*s
    estimate = c*N/workers + communication
    print(f"workers={workers:3d} predicted_us={1e6*estimate:9.3f} predicted_speedup={baseline/estimate:8.3f}")
```

```text
workers=  1 predicted_us= 2000.000 predicted_speedup=   1.000
workers=  2 predicted_us= 1010.002 predicted_speedup=   1.980
workers=  4 predicted_us=  510.002 predicted_speedup=   3.922
workers= 16 predicted_us=  135.002 predicted_speedup=  14.815
workers= 64 predicted_us=   41.252 predicted_speedup=  48.483
workers=256 predicted_us=   17.814 predicted_speedup= 112.271

```

The predicted speedup grows less rapidly as the fixed communication term becomes important. The model uses one scalar per neighboring boundary because the problem is one-dimensional. A two-dimensional decomposition has a different surface-to-volume relationship, so its halo term must be derived again.

**Try it.**

You should increase the startup delay by a factor of ten and rerun the model. Predict which worker counts become inefficient. Then derive the halo payload for a square subdomain with one layer of ghost cells and eight-byte values.

## A useful report separates the evidence.

Your report should state the continuous problem, discretization, solver, stopping rule, and implementation. It should record the grid size, precision, processor count, compiler, and hardware. It should then show numerical error and runtime separately.

You should compare methods at a comparable accuracy target. A solver that stops earlier can look faster while solving a different effective problem. You should also include setup costs when they matter to the intended use. A preconditioner that costs seconds to build may be valuable across many right-hand sides and wasteful for one tiny solve.

This notebook establishes a grid-refinement trend, an exact agreement between two CPU update paths, and a communication-cost prediction under invented parameters. Native MPI scaling and CUDA performance remain separate experiments. That separation makes the notebook a useful starting point for a later implementation rather than an unsupported performance claim.

## Summary

| Check | What the check establishes |
|---|---|
| Grid refinement compares numerical solutions with the manufactured exact solution. | It tests the expected second-order discretization trend. |
| The serial and partitioned updates use the same arithmetic. | Their agreement tests the ownership and halo-index logic. |
| The latency-bandwidth formula predicts iteration cost. | It explains a possible strong-scaling limit under stated assumptions. |
| A native benchmark would record hardware and actual elapsed time. | It would provide a different kind of evidence that this CPU model does not supply. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

### Review questions and answers

**Question 1. What does a tiny linear-system residual establish?**

1. It establishes exact agreement with the continuous PDE.
2. It establishes near-satisfaction of the discrete equations.

**Answer.** It establishes near-satisfaction of the discrete equations. Discretization error remains even when the discrete equations are solved accurately.

**Question 2. What error ratio is expected when a second-order method halves its grid spacing?**

1. The ratio is about two.
2. The ratio is about four.
3. The ratio is about eight.

**Answer.** The ratio is about four. The leading error is proportional to h squared, so halving h reduces it by about four.

**Question 3. Why does the Jacobi implementation preserve an old array?**

1. Every new value must use the previous iteration.
2. The array makes the network faster.

**Answer.** Every new value must use the previous iteration. An in-place update would change the algorithm and its dependency order.

**Question 4. Does exact agreement between the two CPU update paths prove MPI speedup?**

1. Yes.
2. No.

**Answer.** No. The paths run on a CPU without MPI communication, so they only check the modeled computation.

**Question 5. Which term prevents ideal strong scaling in the simple model?**

1. The startup cost remains even as local arithmetic shrinks.
2. The grid spacing is always one.

**Answer.** The startup cost remains even as local arithmetic shrinks. The fixed communication delay eventually becomes large compared with each worker's computation.


## Sources and next implementation

The finite-difference and iterative-method ideas connect Heath's boundary-value and PDE chapters with Ascher and Greif's iterative methods. The communication model connects to Grama and colleagues' analytical modeling chapter. The source bibliography appears in [sources](https://arunshar.github.io/numerical-parallel-computing/reference/sources.html).

A next implementation can replace the simulated blocks with MPI ranks or a CUDA stencil. It should preserve the independent numerical checks and add hardware-specific evidence before making a performance claim.

---

# Course and textbook coverage

This ledger maps the public course topics and all chapter outlines from the four associated books to the study lessons. A mapped chapter means that the notes explain its subject and include selected examples. It does not mean that every proof, exercise, historical note, or software variant in the full text has been reproduced.

The full textbooks were not available during the initial build. Public slides, author and publisher chapter outlines, and primary API documentation supplied the verifiable scope. The track-specific JSON ledgers in `coverage/` distinguish explanations from compressed surveys.

## The textbook map covers the complete chapter outlines.

| Book | Chapter | Topic | Lessons |
|---|---|---|---|
| Heath | 1 | Scientific Computing | [1](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html) |
| Heath | 2 | Systems of Linear Equations | [2](https://arunshar.github.io/numerical-parallel-computing/lectures/02-linear-systems.html), [15](https://arunshar.github.io/numerical-parallel-computing/lectures/15-iterative-solvers.html) |
| Heath | 3 | Linear Least Squares | [3](https://arunshar.github.io/numerical-parallel-computing/lectures/03-least-squares-svd.html) |
| Heath | 4 | Eigenvalue Problems | [4](https://arunshar.github.io/numerical-parallel-computing/lectures/04-eigenvalues.html) |
| Heath | 5 | Nonlinear Equations | [5](https://arunshar.github.io/numerical-parallel-computing/lectures/05-nonlinear-equations.html) |
| Heath | 6 | Optimization | [6](https://arunshar.github.io/numerical-parallel-computing/lectures/06-unconstrained-optimization.html), [7](https://arunshar.github.io/numerical-parallel-computing/lectures/07-constrained-optimization-svm.html) |
| Heath | 7 | Interpolation | [8](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html) |
| Heath | 8 | Numerical Integration and Differentiation | [9](https://arunshar.github.io/numerical-parallel-computing/lectures/09-quadrature-differentiation.html) |
| Heath | 9 | Initial Value Problems for Ordinary Differential Equations | [10](https://arunshar.github.io/numerical-parallel-computing/lectures/10-ode-initial-value.html) |
| Heath | 10 | Boundary Value Problems for Ordinary Differential Equations | [11](https://arunshar.github.io/numerical-parallel-computing/lectures/11-boundary-value.html) |
| Heath | 11 | Partial Differential Equations | [12](https://arunshar.github.io/numerical-parallel-computing/lectures/12-pde.html) |
| Heath | 12 | Fast Fourier Transform | [13](https://arunshar.github.io/numerical-parallel-computing/lectures/13-fourier-fft.html) |
| Heath | 13 | Random Numbers and Simulation | [14](https://arunshar.github.io/numerical-parallel-computing/lectures/14-random-simulation.html) |
| Ascher and Greif | 1 | Numerical Algorithms | [1](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html), [16](https://arunshar.github.io/numerical-parallel-computing/lectures/16-numerical-workflow.html) |
| Ascher and Greif | 2 | Roundoff Errors | [1](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html) |
| Ascher and Greif | 3 | Nonlinear Equations in One Variable | [5](https://arunshar.github.io/numerical-parallel-computing/lectures/05-nonlinear-equations.html) |
| Ascher and Greif | 4 | Linear Algebra Background | [2](https://arunshar.github.io/numerical-parallel-computing/lectures/02-linear-systems.html), [3](https://arunshar.github.io/numerical-parallel-computing/lectures/03-least-squares-svd.html), [4](https://arunshar.github.io/numerical-parallel-computing/lectures/04-eigenvalues.html) |
| Ascher and Greif | 5 | Linear Systems: Direct Methods | [2](https://arunshar.github.io/numerical-parallel-computing/lectures/02-linear-systems.html) |
| Ascher and Greif | 6 | Least Squares | [3](https://arunshar.github.io/numerical-parallel-computing/lectures/03-least-squares-svd.html) |
| Ascher and Greif | 7 | Linear Systems: Iterative Methods | [15](https://arunshar.github.io/numerical-parallel-computing/lectures/15-iterative-solvers.html) |
| Ascher and Greif | 8 | Eigenvalues and Singular Values | [4](https://arunshar.github.io/numerical-parallel-computing/lectures/04-eigenvalues.html) |
| Ascher and Greif | 9 | Nonlinear Systems and Optimization | [5](https://arunshar.github.io/numerical-parallel-computing/lectures/05-nonlinear-equations.html), [6](https://arunshar.github.io/numerical-parallel-computing/lectures/06-unconstrained-optimization.html), [7](https://arunshar.github.io/numerical-parallel-computing/lectures/07-constrained-optimization-svm.html) |
| Ascher and Greif | 10 | Polynomial Interpolation | [8](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html) |
| Ascher and Greif | 11 | Piecewise Polynomial Interpolation | [8](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html) |
| Ascher and Greif | 12 | Best Approximation | [8](https://arunshar.github.io/numerical-parallel-computing/lectures/08-interpolation-splines.html), [16](https://arunshar.github.io/numerical-parallel-computing/lectures/16-numerical-workflow.html) |
| Ascher and Greif | 13 | Fourier Transform | [13](https://arunshar.github.io/numerical-parallel-computing/lectures/13-fourier-fft.html) |
| Ascher and Greif | 14 | Numerical Differentiation | [9](https://arunshar.github.io/numerical-parallel-computing/lectures/09-quadrature-differentiation.html) |
| Ascher and Greif | 15 | Numerical Integration | [9](https://arunshar.github.io/numerical-parallel-computing/lectures/09-quadrature-differentiation.html) |
| Ascher and Greif | 16 | Differential Equations | [10](https://arunshar.github.io/numerical-parallel-computing/lectures/10-ode-initial-value.html), [11](https://arunshar.github.io/numerical-parallel-computing/lectures/11-boundary-value.html), [12](https://arunshar.github.io/numerical-parallel-computing/lectures/12-pde.html) |
| Grama and colleagues | 1 | Introduction | [17](https://arunshar.github.io/numerical-parallel-computing/lectures/17-architecture-cache.html) |
| Grama and colleagues | 2 | Parallel Programming Platforms | [17](https://arunshar.github.io/numerical-parallel-computing/lectures/17-architecture-cache.html) |
| Grama and colleagues | 3 | Principles of Parallel Algorithm Design | [18](https://arunshar.github.io/numerical-parallel-computing/lectures/18-decomposition-task-graphs.html) |
| Grama and colleagues | 4 | Basic Communication Operations | [20](https://arunshar.github.io/numerical-parallel-computing/lectures/20-communication-networks.html), [22](https://arunshar.github.io/numerical-parallel-computing/lectures/22-mpi-collectives-topology.html) |
| Grama and colleagues | 5 | Analytical Modeling of Parallel Programs | [23](https://arunshar.github.io/numerical-parallel-computing/lectures/23-performance-models.html) |
| Grama and colleagues | 6 | Programming Using the Message-Passing Paradigm | [21](https://arunshar.github.io/numerical-parallel-computing/lectures/21-mpi-point-to-point.html), [22](https://arunshar.github.io/numerical-parallel-computing/lectures/22-mpi-collectives-topology.html) |
| Grama and colleagues | 7 | Programming Shared Address Space Platforms | [19](https://arunshar.github.io/numerical-parallel-computing/lectures/19-pthreads-openmp.html) |
| Grama and colleagues | 8 | Dense Matrix Algorithms | [24](https://arunshar.github.io/numerical-parallel-computing/lectures/24-dense-matrix-cannon.html) |
| Grama and colleagues | 9 | Sorting | [25](https://arunshar.github.io/numerical-parallel-computing/lectures/25-sorting-merging.html) |
| Grama and colleagues | 10 | Graph Algorithms | [26](https://arunshar.github.io/numerical-parallel-computing/lectures/26-graphs-sparse-partitioning.html) |
| Grama and colleagues | 11 | Search Algorithms for Discrete Optimization Problems | [27](https://arunshar.github.io/numerical-parallel-computing/lectures/27-search-dynamic-programming.html) |
| Grama and colleagues | 12 | Dynamic Programming | [27](https://arunshar.github.io/numerical-parallel-computing/lectures/27-search-dynamic-programming.html) |
| Grama and colleagues | 13 | Fast Fourier Transform | [28](https://arunshar.github.io/numerical-parallel-computing/lectures/28-parallel-fft.html) |
| Grama and colleagues | A | Appendix A: Complexity of Functions and Order Analysis | [18](https://arunshar.github.io/numerical-parallel-computing/lectures/18-decomposition-task-graphs.html), [23](https://arunshar.github.io/numerical-parallel-computing/lectures/23-performance-models.html) |
| Hwu, Kirk, and El Hajj | 1 | Introduction | [31](https://arunshar.github.io/numerical-parallel-computing/lectures/31-cuda-execution.html) |
| Hwu, Kirk, and El Hajj | 2 | Heterogeneous data parallel computing | [31](https://arunshar.github.io/numerical-parallel-computing/lectures/31-cuda-execution.html) |
| Hwu, Kirk, and El Hajj | 3 | Multidimensional grids and data | [31](https://arunshar.github.io/numerical-parallel-computing/lectures/31-cuda-execution.html) |
| Hwu, Kirk, and El Hajj | 4 | Compute architecture and scheduling | [32](https://arunshar.github.io/numerical-parallel-computing/lectures/32-scheduling-occupancy.html) |
| Hwu, Kirk, and El Hajj | 5 | Memory architecture and data locality | [33](https://arunshar.github.io/numerical-parallel-computing/lectures/33-memory-tiling.html) |
| Hwu, Kirk, and El Hajj | 6 | Performance considerations | [34](https://arunshar.github.io/numerical-parallel-computing/lectures/34-gpu-performance.html) |
| Hwu, Kirk, and El Hajj | 7 | Convolution | [35](https://arunshar.github.io/numerical-parallel-computing/lectures/35-convolution.html) |
| Hwu, Kirk, and El Hajj | 8 | Stencil | [36](https://arunshar.github.io/numerical-parallel-computing/lectures/36-stencils.html) |
| Hwu, Kirk, and El Hajj | 9 | Parallel histogram | [37](https://arunshar.github.io/numerical-parallel-computing/lectures/37-histograms-atomics.html) |
| Hwu, Kirk, and El Hajj | 10 | Reduction | [38](https://arunshar.github.io/numerical-parallel-computing/lectures/38-reductions.html) |
| Hwu, Kirk, and El Hajj | 11 | Prefix sum (scan) | [39](https://arunshar.github.io/numerical-parallel-computing/lectures/39-prefix-scan.html) |
| Hwu, Kirk, and El Hajj | 12 | Merge | [40](https://arunshar.github.io/numerical-parallel-computing/lectures/40-merge-sort.html) |
| Hwu, Kirk, and El Hajj | 13 | Sorting | [40](https://arunshar.github.io/numerical-parallel-computing/lectures/40-merge-sort.html) |
| Hwu, Kirk, and El Hajj | 14 | Sparse matrix computation | [41](https://arunshar.github.io/numerical-parallel-computing/lectures/41-sparse-graphs.html) |
| Hwu, Kirk, and El Hajj | 15 | Graph traversal | [41](https://arunshar.github.io/numerical-parallel-computing/lectures/41-sparse-graphs.html) |
| Hwu, Kirk, and El Hajj | 16 | Deep learning | [42](https://arunshar.github.io/numerical-parallel-computing/lectures/42-deep-learning.html) |
| Hwu, Kirk, and El Hajj | 17 | Iterative magnetic resonance imaging reconstruction | [43](https://arunshar.github.io/numerical-parallel-computing/lectures/43-mri-electrostatics.html) |
| Hwu, Kirk, and El Hajj | 18 | Electrostatic potential map | [43](https://arunshar.github.io/numerical-parallel-computing/lectures/43-mri-electrostatics.html) |
| Hwu, Kirk, and El Hajj | 19 | Parallel programming and computational thinking | [44](https://arunshar.github.io/numerical-parallel-computing/lectures/44-streams-clusters.html), [46](https://arunshar.github.io/numerical-parallel-computing/lectures/46-gpu-numerics-project.html) |
| Hwu, Kirk, and El Hajj | 20 | Programming a heterogeneous computing cluster | [44](https://arunshar.github.io/numerical-parallel-computing/lectures/44-streams-clusters.html) |
| Hwu, Kirk, and El Hajj | 21 | CUDA dynamic parallelism | [45](https://arunshar.github.io/numerical-parallel-computing/lectures/45-dynamic-parallelism-future.html) |
| Hwu, Kirk, and El Hajj | 22 | Advanced practices and future evolution | [45](https://arunshar.github.io/numerical-parallel-computing/lectures/45-dynamic-parallelism-future.html) |
| Hwu, Kirk, and El Hajj | 23 | Conclusion and outlook | [45](https://arunshar.github.io/numerical-parallel-computing/lectures/45-dynamic-parallelism-future.html) |
| Hwu, Kirk, and El Hajj | A | Appendix A: Numerical considerations | [46](https://arunshar.github.io/numerical-parallel-computing/lectures/46-gpu-numerics-project.html), [1](https://arunshar.github.io/numerical-parallel-computing/lectures/01-errors-floating-point.html) |

## The course-specific material has its own map.

| Source unit | Study location | Coverage boundary |
|---|---|---|
| The CSCI 5302 syllabus covers errors, equations, interpolation, least squares, optimization, quadrature, and ODEs. | Lessons 1–10 explain the core sequence. | Lessons 11–16 extend the sequence to the remaining book topics. |
| The supplementary Taylor and bicubic notes explain local multivariate models and two-dimensional interpolation. | Lessons 8 and 16 explain those constructions. | The examples use new teaching data. |
| The supplementary QP/LP and SVM materials connect optimization to constrained models. | Lesson 7 explains the objectives, constraints, and optimality conditions. | The notes do not reproduce the instructor's handouts. |
| The MATLAB references compare array operations and programming conventions. | Lesson 16 explains the important MATLAB/Python differences. | The original reference sheets remain external sources. |
| Mooney's 26 lecture decks cover architectures, OpenMP, MPI, modeling, and CUDA. | Lessons 17–46 explain that sequence and its textbook extensions. | The final-lecture hardware and ML topics are clearly marked surveys. |
| HW1 uses shared-memory parallelism and dynamic time warping. | Lessons 19, 27, and 30 explain dependencies, implementation checks, and reporting. | The examples teach the methods without redistributing the assignment package. |
| HW2 and HW3 use distributed matrix multiplication and block-cyclic layouts. | Lessons 21, 22, 24, and 30 explain the ownership and communication. | The new examples have their own data and assertions. |
| HW4 uses CUDA convolution and profiling. | Lessons 34 and 35 explain memory reuse, boundaries, and performance evidence. | Native CUDA performance remains unverified locally. |
| HW5 uses pairwise batched matrix products. | Lesson 46 explains the algebra, numerical consequences, and validation choices. | Algebraic simplification does not establish permission under a particular assignment rubric. |
| Saad's 19 lecture sets add sorting, sparse and graph algorithms, hybrid programming, and quantum concepts. | Lessons 17–29 and 40–41 cover these subjects. | The quantum material is an introductory model. |
| The final project requires an implementation and evidence. | Lessons 30, 46, and 47 explain reproducibility, correctness, and performance reporting. | The capstone is an original teaching problem. |

## Some advanced subjects remain survey-level treatments.

A short section introduces a method's purpose, assumptions, main mechanism, and tradeoffs. It does not establish mastery of that method. The compressed areas include specialized eigensolver variants, finite-element and spectral implementation details, quantum algorithms, MRI reconstruction, dynamic parallelism, and evolving accelerator platforms. The handoff identifies these areas for expansion with authorized textbook access and hardware evidence.

The intended next step is to deepen those sections, not to change their status labels without adding explanations and checks. A future edition should also map textbook subsections and exercises when the actual editions become available.


---

# Sources and attribution

The notes use public course material to identify scope and primary references to check mechanisms. The explanations, worked examples, quizzes, and teaching programs in this repository are original study material. The repository links to instructors' files instead of republishing their PDF collections or homework archives.

## CSCI 5302 provides the numerical course sequence.

[Prof. Daniel Boley's Spring 2025 course](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/) provides the syllabus, weekly sequence, and links to [notes and supplementary material](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The supplementary topics include multivariate Taylor expansion, bicubic interpolation, QP/LP examples, FFTs, SVMs, and MATLAB references.

Michael T. Heath wrote *Scientific Computing: An Introductory Survey*. The course names the second edition, and its SIAM link points to the revised second edition. The [author's table of contents](https://heath.cs.illinois.edu/scicomp/contents/index.html) and [lecture slides](https://heath.cs.illinois.edu/scicomp/notes/) provide the public topic outline. The [SIAM book page](https://epubs.siam.org/doi/book/10.1137/1.9781611975581) identifies the revised edition.

Uri M. Ascher and Chen Greif wrote *A First Course in Numerical Methods* (SIAM, 2011). The [publisher's chapter list](https://epubs.siam.org/doi/book/10.1137/9780898719987) supplies the 16-chapter outline. The full textbook was not downloaded for this study edition.

## CSCI 5451 provides the parallel course sequences.

[James Mooney's Fall 2025 course](https://jimtmooney.github.io/Courses/F25/index.html) provides the dated lecture sequence, 26 PDF decks, five homework packages, and team-project expectations. [Prof. Yousef Saad's Spring 2021 notes](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) add a second sequence with sorting, sparse matrices, graphs, hybrid programming, and introductory quantum computing.

Ananth Grama, Anshul Gupta, George Karypis, and Vipin Kumar wrote *Introduction to Parallel Computing*, second edition (Addison-Wesley, 2003). The [author-hosted contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) provide a preprint chapter outline. That page reverses the final edition's chapters 6 and 7. The author's [chapter 6 slides](https://www.cs.purdue.edu/homes/ayg/book/Slides/chap6_slides.pdf) identify message passing, which agrees with the course's reading references. This edition uses chapter 6 for MPI and chapter 7 for shared-memory programming. This repository does not redistribute the authors' restricted preprint chapters.

Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj wrote *Programming Massively Parallel Processors: A Hands-on Approach*, fourth edition (Morgan Kaufmann, 2022). The [publisher's page](https://www.educate.elsevier.com/book/details/9780323912310) lists 23 chapters and Appendix A. The author list includes El Hajj, whose name was absent from the initial course summary supplied in chat.

[Simon Boehm's CUDA matrix-multiplication worklog](https://siboehm.com/articles/22/CUDA-MMM) is an additional performance case study linked by the course. The new lessons explain the relevant optimization ideas using their own examples. They do not reproduce his article or its figures.

## Primary technical references support the implementation details.

The [NumPy documentation](https://numpy.org/doc/stable/) describes the array and linear algebra functions used in the CPU examples. The [SciPy documentation](https://docs.scipy.org/doc/scipy/) provides additional numerical method references.

The [MPI Forum](https://www.mpi-forum.org/docs/) publishes the MPI standard. [OpenMP](https://www.openmp.org/specifications/) publishes the OpenMP specification. [NVIDIA's CUDA documentation](https://docs.nvidia.com/cuda/) covers execution, memory behavior, synchronization, and native development tools. Individual lessons cite narrower pages where useful.

## Public availability does not transfer copyright.

The MIT license applies to the original code and prose contributed to this repository. Linked books, course slides, assignment files, and articles retain their own copyright and terms. The local archives created before this repository are separate study downloads. They are not included in the public Git history.

## You can fetch the public source files locally.

The following command downloads cataloged source PDFs into `downloaded-sources/`, which Git ignores. The source sites can change or withdraw a file. The script records access failures and compares downloaded bytes with the cataloged hashes.

```bash
python scripts/fetch_public_sources.py --course all --pdf-only
```

You can omit `--pdf-only` to include the other cataloged course files. The script does not download full textbooks, sign into Canvas, or execute downloaded programs. You should keep the downloaded directory outside public commits.


---

# A glossary for both courses

The definitions below use the same terms as the lessons. You should return to the relevant lesson for assumptions, equations, and examples.

## Numerical analysis

Absolute error
: The absolute error is the magnitude of the difference between an approximation and its reference value.

Relative error
: The relative error divides an absolute error by the magnitude of a nonzero reference value.

Conditioning
: Conditioning describes how much the exact answer can change when the problem data change slightly.

Stability
: Numerical stability describes how an algorithm's errors compare with the effect of small changes to the input problem.

Backward error
: The backward error measures how much the original data must change to make the computed answer exact for a nearby problem.

Residual
: A residual measures the discrepancy left when a candidate answer is substituted into the original equation.

Truncation error
: Truncation error arises when a finite approximation replaces an infinite process or an exact mathematical operation.

Roundoff error
: Roundoff error arises when a value is represented or computed with finite precision.

Cancellation
: Cancellation removes leading significant digits when nearly equal values are subtracted, which can expose earlier errors.

Pivoting
: Pivoting changes the order of elimination to avoid unsuitable pivots and improve numerical behavior.

Orthogonality
: Two vectors are orthogonal when their inner product is zero under the chosen inner product.

Singular value
: A singular value describes a stretching factor of a linear map along appropriate orthogonal directions.

Spectral radius
: The spectral radius is the largest magnitude among a matrix's eigenvalues.

Preconditioner
: A preconditioner transforms a system so an iterative method can converge more effectively while preserving the recoverable solution.

Interpolation
: Interpolation constructs an approximation that matches specified data values exactly at the interpolation nodes.

Approximation
: Approximation constructs a simpler representation whose error is controlled according to a chosen criterion.

Quadrature
: Quadrature estimates an integral with a weighted sum of function values.

Stiffness
: Stiffness describes a differential-equation setting in which stability forces some explicit methods to take much smaller steps than accuracy alone would require.

Consistency
: Consistency means that a numerical discretization approaches the target equation in the appropriate small-step limit.

Convergence
: Convergence means that a sequence of approximations approaches the desired mathematical object under stated assumptions.

Convexity
: Convexity means that a function's value on a line segment does not exceed the corresponding linear interpolation of endpoint values.

KKT conditions
: The Karush–Kuhn–Tucker conditions combine stationarity, feasibility, and complementary slackness for constrained optimization under suitable assumptions.

## Parallel computing

Task
: A task is a unit of computational work that the algorithm can schedule subject to its dependencies.

Work
: Work counts the total operations required by a parallel algorithm under a stated cost model.

Span
: Span measures the longest dependency path through the task graph under that model.

Speedup
: Speedup divides a reference execution time by the parallel execution time for the specified comparison.

Efficiency
: Parallel efficiency divides speedup by the number of processors used in the comparison.

Strong scaling
: Strong scaling studies execution time as processor count grows while the total problem size stays fixed.

Weak scaling
: Weak scaling studies execution time while problem size grows with processor count to keep work per processor approximately fixed.

Latency
: Latency is the delay before a requested operation or transfer produces its result.

Bandwidth
: Bandwidth is the rate at which data can move through a specified interface.

Arithmetic intensity
: Arithmetic intensity divides an operation count by bytes moved across a specified memory boundary.

Race condition
: A race condition occurs when the result depends on an uncontrolled ordering of concurrent operations.

Data race
: A data race involves conflicting concurrent memory accesses without the synchronization required by the language or memory model.

Deadlock
: Deadlock occurs when participants wait for conditions that their current waits prevent from occurring.

False sharing
: False sharing occurs when independent variables share a coherence unit and cause unnecessary coherence traffic between processors.

Collective operation
: An MPI collective operation involves the ranks in a communicator according to the operation's matching rules.

Nonblocking operation
: A nonblocking operation returns control before the operation has necessarily completed and requires a later completion check before protected resources can be reused.

Communicator
: An MPI communicator identifies a group of ranks and a communication context.

Load balance
: Load balance describes how evenly useful work is distributed among available workers.

## GPU computing

Kernel
: A CUDA kernel is a device function launched over a configured collection of threads.

Thread block
: A thread block is a group of CUDA threads that can cooperate through block-level synchronization and shared memory.

Warp
: A warp is the group of threads scheduled together under NVIDIA's execution model, subject to the architecture's control-flow rules.

Occupancy
: Occupancy compares active warps with the hardware limit for the relevant multiprocessor configuration.

Coalescing
: Coalescing combines suitable thread memory accesses into fewer memory transactions under the device's access rules.

Shared-memory bank conflict
: A bank conflict occurs when a group of shared-memory accesses requires extra service because its addresses compete for a bank under the hardware's rules.

Tiling
: Tiling groups data and operations into blocks so a loaded value can serve several computations before it is discarded.

Atomic operation
: An atomic operation performs a specified read-modify-write action indivisibly at the scope guaranteed by its API.

Privatization
: Privatization gives workers separate partial state and later combines the results to reduce contention.

Reduction
: A reduction combines many values into fewer values with a specified operation and ordering policy.

Scan
: A scan computes every prefix aggregate of an ordered sequence, using an inclusive or exclusive convention.

Stream
: A CUDA stream orders submitted operations within that stream and interacts with other streams according to explicit and implicit dependency rules.

Pinned memory
: Pinned host memory remains resident so device transfers can use the supported transfer path without ordinary pageable-memory handling.

Dynamic parallelism
: CUDA dynamic parallelism allows eligible device code to launch additional device work under the toolkit and hardware rules.

Roofline model
: A roofline model bounds attainable performance using computational throughput, memory bandwidth, and arithmetic intensity for a chosen memory level.


---

# Claude continuation handoff

Arun requested a detailed study document for CSCI 5302 and CSCI 5451, with simple explanations, worked examples, tutorials, and coverage of the associated textbook topics. He approved the public repository name `arunshar/numerical-parallel-computing`.

## You should start with the sources and evidence.

Read `README.md`, `docs/reference/coverage.md`, `docs/reference/sources.md`, and `docs/reference/validation.md`. Then inspect `coverage/book-chapters.json`, the track-specific coverage JSON files, and `evidence/validation.json`. These files distinguish original explanations, selected experiments, compressed surveys, and hardware validation.

The previous local archive contained the instructors' original PDFs and homework bundles. Those third-party files are excluded from the public repository. `coverage/public-source-inventory.json` retains their source URLs, retrieval times, and hashes. You can obtain the files from the original sources when access permits.

The complete textbooks were not available during this build. The author and publisher outlines establish chapter scope, and public lecture slides establish course topics. Do not claim that every textbook proof, exercise, or subsection has been checked against the full edition.

## The authored sources have a stable layout.

Lessons 1–16 cover numerical methods. Lessons 17–30 cover CPU and distributed computing. Lessons 31–46 cover GPU computing. Lesson 0 prepares the reader, and lesson 47 connects the tracks with a manufactured diffusion problem and a communication model.

The Markdown lectures live in `docs/lectures/`. The capstone notebook is generated from `labs/src/47-integrated-capstone.py`. The portable teaching scripts live in `labs/numerical/`, `labs/parallel/`, and `labs/gpu/`. The native C and CUDA examples are stored within their relevant tracks.

`dist/STUDY_GUIDE.md` is the combined reading edition. It is generated from the lesson sources. You should edit the sources and rebuild the reading edition instead of editing the generated book alone.

## You should preserve the evidence boundaries.

The Python experiments use small synthetic teaching inputs. Their successful execution validates their stated assertions on those inputs. It does not validate all possible inputs, numerical stability in every regime, a distributed cluster, or CUDA hardware performance.

Native pthread, OpenMP, and MPI checks have their own evidence when the required tools were available. Native CUDA execution is UNVERIFIED unless a later record names the hardware, compiler, command, input, and result. A CPU model of a GPU algorithm is not a GPU benchmark.

The study material includes a useful algebraic identity for the pairwise batched matrix-product example. Real-arithmetic equivalence does not guarantee bitwise floating-point equivalence or permission under an instructor's assignment constraints. Keep those distinctions explicit.

## The next refinement pass should deepen the compressed areas.

1. You should compare each lesson with the actual textbook edition when Arun provides authorized access. Expand the chapter ledger into a subsection ledger before calling the coverage exhaustive.
2. You should strengthen the advanced numerical surveys with complete derivations and independent examples. Priority topics include specialized eigensolvers, best approximation, finite elements, spectral discretizations, and integral equations.
3. You should extend the distributed examples to multiple machines only when Arun authorizes that runtime. Record the network and process placement before interpreting scaling.
4. You should compile and test the CUDA examples on a compatible NVIDIA device. Add sanitizer checks, boundary cases, and numerical comparisons before collecting performance data.
5. You should deepen the advanced GPU case studies, including MRI reconstruction, electrostatic cutoff binning, dynamic parallelism, and cluster overlap.
6. You should read the prose as a student. Replace unexplained jargon with definitions and small examples while preserving equations and claims.
7. You should check every quiz answer against its lesson and executable evidence. A plausible-looking distractor is not a substitute for a checked explanation.

## You should run the repository checks before proposing changes.

```bash
python -m pip install -r requirements-build.txt
python scripts/verify_course.py
python labs/build_notebooks.py 47
python scripts/build_inventory.py
python scripts/build_book.py
python scripts/build_reader.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The notebook build uses a local helper inside the repository. A browser notebook can download the helper from the public GitHub source. The helper URL and repository settings must stay aligned if the repository is renamed.

You should work on a branch and open a draft pull request after publication. Arun should review substantive changes before they merge. The public source files contain no credentials and should remain that way.

## The publishing states are separate.

The GitHub repository stores the source and generated reading editions. A GitHub Actions workflow verifies the portable labs, builds the Sphinx site with warnings treated as errors, and deploys GitHub Pages. A successful remote workflow and live page checks establish website publication.

The repository also contains a Read the Docs configuration. A local preflight can validate that configuration, but a live Read the Docs project still requires Arun to import the repository through his account. Do not describe that service as live until its build and URL are verified.


---

# Validation and evidence

The final build records its checks in `evidence/validation.json`. Each portable lab entry contains the source path, source hash, return code, and captured output. The track-specific evidence directories preserve additional numerical and native-runtime checks.

## The recorded checks cover these artifacts.

| Check | Recorded result | Scope |
|---|---|---|
| The structure and quiz checks inspect 48 lessons. | All 240 quiz questions parse, and the required lesson sections are present. | These checks establish structure, not proof correctness. |
| The portable execution check runs 46 numbered Python labs. | All labs exit successfully with their assertions enabled. | The experiments use small synthetic inputs on the recorded CPU environment. |
| The integrated capstone executes as a notebook. | Its numerical and decomposition assertions pass. | Its communication costs are invented model inputs. |
| The native checks compile and run four C programs. | pthreads, OpenMP, an MPI ring, and MPI collectives pass. | The run uses at most four local threads or ranks and records no speed measurements. |
| The documentation preflight uses a fresh Python 3.12 environment. | All 59 source pages pass the strict build and page checks. | The preflight checks canonical links, quizzes, and notebook launch URLs. |
| The external reference check requests the linked source pages. | Two SIAM book pages return HTTP 403 to the automated checker. | These references remain listed as access-limited. The check does not establish full-book access. |
| The CUDA suite has source-level review. | Native compilation and execution remain UNVERIFIED. | The machine has no compatible NVIDIA CUDA runtime. |

The preflight transcript appears in `evidence/documentation-preflight.txt`. The external link report appears in `evidence/external-links.json`.

## The portable checks have a defined scope.

The Python checks execute small teaching cases. They test the assertions stated by each lab. The capstone checks grid-refinement behavior, agreement between two update paths, and a cost model with invented parameters. It does not measure MPI or CUDA performance.

The documentation build uses Sphinx with `-W`, which treats warnings as errors. The structure check verifies lesson sections, quiz schemas, and unresolved placeholders. The combined document is generated from the same authored lessons.

## Native execution needs separate records.

The native examples require their stated compiler and runtime. Successful CPU reference checks do not establish native execution. CUDA compilation and execution remain UNVERIFIED in this edition because a local CUDA compiler and compatible NVIDIA runtime were not available.

The relevant track evidence reports whether pthread, OpenMP, and MPI examples compiled and ran. Those results apply to the recorded local configuration and small test inputs. They do not establish performance on a university cluster.

## Publication requires remote verification.

A local build does not prove that a public website is live. The final publication report separately checks the pushed commit, GitHub Actions status, and the live site. The Read the Docs configuration is provided, but the service requires an account import before its hosted site can be claimed.

## Visual and substantive review have limits.

The initial review checks equations, output agreement, representative content, and navigation. It does not certify every theorem or replace expert review of all 48 lessons. The coverage ledger identifies the advanced survey sections that should receive another technical pass.
