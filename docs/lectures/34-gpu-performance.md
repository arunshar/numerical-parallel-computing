# 34. Coalescing, roofline models, and optimization experiments

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`33-memory-tiling`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will estimate a roofline bound with consistent units.
- You will count memory segments for a simple warp access pattern.
- You will design a measurement that separates kernel speed from application speed. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/34_gpu_performance.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`34_gpu_performance.py <../../labs/gpu/34_gpu_performance.py>`.

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

:::{admonition} Try it 1 :class: exercise

Offset every address by four bytes. Count the segments again and explain why alignment changes the contiguous case even though the stride is unchanged. :::

:::{admonition} Try it 2 :class: exercise

Suppose a kernel is 20% of total runtime and becomes four times faster. Calculate the application speedup and explain why it is smaller than four. :::

## Summary

| Decision | What you should remember |
|---|---|
| Roofline | The minimum of compute capacity and bandwidth times intensity gives a first upper bound. |
| Coalescing | Physical addresses determine transaction efficiency. |
| Measurement | A timing interval must include completed work and a stated scope. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the toy ridge-point intensity?",
    "options": [
      "It is 10 FLOP/byte.",
      "It is 100 FLOP/byte.",
      "It is 0.1 FLOP/byte."
    ],
    "answer": 0,
    "explain": "The ratio 1000 GFLOP/s divided by 100 GB/s is 10 FLOP/byte."
  },
  {
    "q": "Does the roofline bound guarantee achieved throughput?",
    "options": [
      "Yes, every kernel reaches it.",
      "No, other limits can reduce throughput."
    ],
    "answer": 1,
    "explain": "Instruction dependencies and other costs are absent from the model."
  },
  {
    "q": "How many 32-byte segments does the aligned contiguous warp model touch?",
    "options": [
      "It touches four segments.",
      "It touches 32 segments."
    ],
    "answer": 0,
    "explain": "Its 128-byte payload spans four aligned segments."
  },
  {
    "q": "Why can timing only the host launch be misleading?",
    "options": [
      "The launch can return before GPU completion.",
      "The host always executes the kernel itself."
    ],
    "answer": 0,
    "explain": "Asynchronous enqueue time is different from completed execution time."
  },
  {
    "q": "What must accompany a claimed GPU speedup?",
    "options": [
      "Only the fastest number is needed.",
      "A comparable baseline, correctness check, and measurement scope are needed."
    ],
    "answer": 1,
    "explain": "The claim depends on what was computed and what the timer included."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 6. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [CUDA matrix-multiplication worklog](https://siboehm.com/articles/22/CUDA-MMM) supplies a concrete optimization sequence. Its reported measurements belong to its own setup, and this lesson does not reproduce those measurements.
