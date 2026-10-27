# 46. GPU numerical reliability and a complete project workflow

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`45-dynamic-parallelism-future`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will derive the exact-real factorization of an all-pairs matrix-product sum.
- You will explain how floating-point reassociation changes the numerical contract.
- You will write a reproducible correctness and performance plan for a GPU project. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/46_gpu_numerics_project.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`46_gpu_numerics_project.py <../../labs/gpu/46_gpu_numerics_project.py>`.

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

:::{admonition} Try it 1 :class: exercise

Create matrices whose entrywise sums nearly cancel. Compare both expressions against a float64 reference and explain why one random normal sample cannot establish a universal error bound. :::

:::{admonition} Try it 2 :class: exercise

Write a one-page benchmark contract for this operation. Specify dimensions, precision, allowed transformations, tolerances, included costs, and the evidence needed before claiming a speedup. :::

## Summary

| Decision | What you should remember |
|---|---|
| Algebra | Distributivity can remove many products without assuming matrix commutativity. |
| Numerics | Changing the evaluation order changes rounding and requires validation. |
| Evidence | A GPU performance claim needs executed GPU measurements under a stated contract. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does the factorization assume AiBj equals BjAi?",
    "options": [
      "Yes, it assumes matrix commutativity.",
      "No, it preserves the multiplication order."
    ],
    "answer": 1,
    "explain": "Distributivity groups the A sum on the left and the B sum on the right."
  },
  {
    "q": "How many direct matrix products occur for 512 matrices in each list?",
    "options": [
      "There are 262144 products.",
      "There are 1024 products."
    ],
    "answer": 0,
    "explain": "The direct double loop contains 512 times 512 pairs."
  },
  {
    "q": "Must the two float32 expressions be bitwise identical?",
    "options": [
      "Yes, real algebra guarantees identical rounding.",
      "No, reassociation changes intermediate rounding."
    ],
    "answer": 1,
    "explain": "The operations are grouped differently."
  },
  {
    "q": "Does a passing CPU reference certify a CUDA kernel\u2019s race freedom?",
    "options": [
      "Yes, mathematical agreement proves synchronization.",
      "No, CUDA execution and memory checks are separate."
    ],
    "answer": 1,
    "explain": "The CPU script does not exercise GPU concurrency."
  },
  {
    "q": "What should a handoff say about an unexecuted CUDA file?",
    "options": [
      "It should label GPU execution UNVERIFIED.",
      "It should infer success from the source code\u2019s appearance."
    ],
    "answer": 0,
    "explain": "An implementation and an executed validation result are different evidence states."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters A. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html) explains CUDA numerical behavior. The local `labs/gpu/native_suite.cu` is an original teaching implementation whose CUDA compilation and execution remain **UNVERIFIED**.
