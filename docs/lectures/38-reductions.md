# 38. Parallel reductions and numerical order

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`37-histograms-atomics`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will derive the work and depth of a reduction tree.
- You will handle arbitrary input lengths with an identity element.
- You will explain why floating-point reduction order affects reproducibility. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/38_reductions.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`38_reductions.py <../../labs/gpu/38_reductions.py>`.

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

:::{admonition} Try it 1 :class: exercise

Replace addition with maximum and use an all-negative input. Choose the correct identity and explain why zero would fail. :::

:::{admonition} Try it 2 :class: exercise

Compare a serial sum, pairwise sum, and `math.fsum` on a long cancellation-heavy input. Report absolute error and explain the cost of a more accurate reference. :::

## Summary

| Decision | What you should remember |
|---|---|
| Tree structure | A balanced reduction exposes logarithmic dependency depth. |
| Boundary handling | An identity element handles partial input groups. |
| Numerical order | Race freedom does not imply bitwise reproducibility. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the exact sum of integers 1 through 9?",
    "options": [
      "It is 45.",
      "It is 36.",
      "It is 81."
    ],
    "answer": 0,
    "explain": "The arithmetic-series formula gives 9 times 10 divided by 2."
  },
  {
    "q": "What identity pads an addition reduction?",
    "options": [
      "The identity is zero.",
      "The identity is one."
    ],
    "answer": 0,
    "explain": "Adding zero leaves a partial sum unchanged."
  },
  {
    "q": "Is floating-point addition exactly associative?",
    "options": [
      "Yes, it behaves like real addition in every case.",
      "No, rounding can change the result."
    ],
    "answer": 1,
    "explain": "Intermediate rounding depends on the order."
  },
  {
    "q": "What does a second kernel supply in a hierarchical reduction?",
    "options": [
      "It supplies a valid stage boundary and reduces block partials.",
      "It makes ordinary block barriers global."
    ],
    "answer": 0,
    "explain": "The stream orders the kernels, while barriers remain block-scoped."
  },
  {
    "q": "Can a race-free atomic sum vary in low-order bits?",
    "options": [
      "Yes, update order can vary.",
      "No, atomicity fixes the arithmetic order."
    ],
    "answer": 0,
    "explain": "Atomicity prevents lost updates but does not choose one fixed addition sequence."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 10. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
