# 31. CUDA execution, threads, and multidimensional data

Prepared for Arun Sharma. Codex drafted this study edition.

This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will map one output to one thread without writing outside an array.
- You will separate host allocation, device allocation, transfers, and execution.
- You will explain why parallel work can still produce a slow application. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/31_cuda_execution.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`31_cuda_execution.py <../../labs/gpu/31_cuda_execution.py>`.

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

:::{admonition} Try it 1 :class: exercise

Set the array length to 1, 4, 5, and 17. Predict the grid size and number of skipped threads before each run. Explain why a zero-length array needs separate host-side launch handling. :::

:::{admonition} Try it 2 :class: exercise

Replace the rectangular address formula with `col * height + r`. Explain why this remains a valid permutation but changes the layout interpretation. Identify what goes wrong if the producer uses one convention and the consumer uses the other. :::

## Summary

| Decision | What you should remember |
|---|---|
| Thread mapping | Each valid output should have a clear owner. |
| Bounds | A ceiling-divided launch requires a bounds guard. |
| Application cost | Transfers and launches can dominate a small kernel. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "How many blocks cover ten elements with four threads per block?",
    "options": [
      "Two blocks cover them.",
      "Three blocks cover them.",
      "Ten blocks cover them."
    ],
    "answer": 1,
    "explain": "The ceiling of ten divided by four is three."
  },
  {
    "q": "What does thread i own in the reference vector addition?",
    "options": [
      "It owns output c[i].",
      "It owns every output.",
      "It owns a whole block."
    ],
    "answer": 0,
    "explain": "The one-writer rule assigns exactly one output to each valid thread."
  },
  {
    "q": "Can ordinary blocks rely on their execution order?",
    "options": [
      "Yes, block indices determine time order.",
      "No, their execution order is not guaranteed."
    ],
    "answer": 1,
    "explain": "The scheduler can execute blocks in any order."
  },
  {
    "q": "What is the row-major index for row 2, column 3, and width 5?",
    "options": [
      "It is 13.",
      "It is 11.",
      "It is 10."
    ],
    "answer": 0,
    "explain": "The index is 2 times 5 plus 3."
  },
  {
    "q": "Does this Python experiment establish GPU speed?",
    "options": [
      "Yes, its output gives a GPU benchmark.",
      "No, it checks indexing on a CPU."
    ],
    "answer": 1,
    "explain": "The program never launches CUDA code."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 1, 2, 3. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
