# 45. Dynamic parallelism, libraries, and accelerator evolution

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`44-streams-clusters`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will explain when device-launched child work can express an irregular problem.
- You will compare dynamic launches with queues and precomputed work lists.
- You will choose a library, compiler layer, and profiler for a stated question. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/45_dynamic_parallelism_future.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`45_dynamic_parallelism_future.py <../../labs/gpu/45_dynamic_parallelism_future.py>`.

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

:::{admonition} Try it 1 :class: exercise

Replace the size-based split condition with a synthetic error value that varies by region. Compare the largest leaf workload with the mean and discuss load imbalance. :::

:::{admonition} Try it 2 :class: exercise

Assign a hypothetical fixed launch cost and a per-element work cost. Find a threshold where child-launch overhead dominates, and clearly label the result as a model. :::

## Summary

| Decision | What you should remember |
|---|---|
| Dynamic launch | A device can discover child work, but launch overhead and supported semantics still matter. |
| Work queues | Persistent workers trade launch costs for queue and termination complexity. |
| Future systems | Ownership, locality, dependencies, and validation transfer across accelerator ecosystems. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does dynamic parallelism remove launch overhead?",
    "options": [
      "Yes, device launches are free.",
      "No, child launches still have costs."
    ],
    "answer": 1,
    "explain": "Small child tasks can spend more on launch overhead than useful work."
  },
  {
    "q": "What must an adaptive subdivision include?",
    "options": [
      "It needs a termination rule and resource bounds.",
      "It needs unlimited recursive depth."
    ],
    "answer": 0,
    "explain": "The task tree can grow rapidly without bounded stopping rules."
  },
  {
    "q": "What does a temporarily empty task queue prove?",
    "options": [
      "It always proves global completion.",
      "It may prove nothing if active workers can produce more tasks."
    ],
    "answer": 1,
    "explain": "Termination detection must account for active producers."
  },
  {
    "q": "Which tool is suited to a CPU-GPU timeline?",
    "options": [
      "Nsight Systems is suited to that view.",
      "A single arithmetic checksum is suited to that view."
    ],
    "answer": 0,
    "explain": "The timeline shows transfers, gaps, kernels, and host activity."
  },
  {
    "q": "Can CUDA code be assumed to run unchanged on every accelerator?",
    "options": [
      "Yes, all accelerators share the same interface.",
      "No, platform interfaces and execution rules differ."
    ],
    "answer": 1,
    "explain": "The general design principles transfer more broadly than source compatibility."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 21, 22, 23. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [CUDA dynamic-parallelism documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html) should be checked for the installed toolkit. The [CUTLASS repository](https://github.com/NVIDIA/cutlass), [Triton documentation](https://triton-lang.org/main/index.html), [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html), and [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) describe their own supported environments. This lesson surveys these ecosystems without claiming local execution.
