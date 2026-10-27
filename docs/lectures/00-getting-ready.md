# 0. Getting ready

Prepared for Arun Sharma. Codex drafted this study edition.

You can read these lessons with basic programming, first-year calculus, and introductory linear algebra. You do not need to remember every theorem. You do need to distinguish a mathematical problem from the algorithm and machine used to solve it.

A matrix equation can have a unique exact solution while a computer returns a poor approximation. A parallel program can produce the right answer and still run more slowly than a serial program. These are different failures, and the course gives you tools to diagnose each one.

:::{admonition} Learning goals
- You will distinguish the mathematical problem, numerical algorithm, and implementation.
- You will read array shapes, derivatives, and asymptotic costs without confusing their roles.
- You will identify whether an experiment measures numerical accuracy or execution performance.
- You will select the correct runtime for a Python, C, MPI, or CUDA example.
:::

## Setup

You should follow {doc}`../getting-started/run-locally` for the Python environment. The first numerical lessons run on a CPU. Native parallel examples have additional setup instructions in their own lessons.

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

:::{admonition} Try it
:class: exercise
You should compute the function, gradient, and Hessian at `(1, 2)`. You should then predict the first-order change when `x` increases by `0.01`. Compare that prediction with the exact change in the polynomial.
:::

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

:::{admonition} Try it
:class: exercise
You should sketch the ownership of a 4-by-4 matrix divided among four MPI ranks. Then sketch the same matrix processed by four threads in one shared array. Explain which version needs messages to obtain a neighbor's data.
:::

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

```{quiz}
- q: A has shape (4, 3), and B has shape (3, 2). What is the shape of AB?
  options: ["The shape is (4, 2).", "The shape is (3, 3).", "The product is undefined."]
  answer: 0
  explain: Each of four rows combines with each of two columns, using three products per output entry.
- q: Does a faster execution prove that an algorithm is numerically accurate?
  options: ["Yes, because faster algorithms have fewer errors.", "No, because accuracy requires a separate check."]
  answer: 1
  explain: Execution time measures performance. A residual or comparison with a reference checks numerical behavior.
- q: What does a Jacobian describe?
  options: ["It describes first derivatives of several outputs with respect to several inputs.", "It describes only one matrix eigenvalue."]
  answer: 0
  explain: The Jacobian is the matrix of partial derivatives for a vector-valued function.
- q: Can changing an addition tree change a floating-point result?
  options: ["Yes, because rounding occurs after intermediate operations.", "No, because addition is always associative on a computer."]
  answer: 0
  explain: Floating-point addition can discard small terms at different intermediate steps.
- q: Does a CPU model of a CUDA kernel measure GPU throughput?
  options: ["Yes, if it computes the same numbers.", "No, because it executes on a different machine model."]
  answer: 1
  explain: A CPU reference can check arithmetic or indexing, but device performance requires device execution.
```
