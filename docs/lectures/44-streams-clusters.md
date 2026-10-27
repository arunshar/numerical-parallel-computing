# 44. CUDA streams, communication overlap, and GPU clusters

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`43-mri-electrostatics`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will draw dependencies between transfers, kernels, and communication.
- You will derive the ideal throughput of a three-stage chunk pipeline.
- You will explain what CUDA-aware MPI does and what it does not guarantee. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/44_streams_clusters.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`44_streams_clusters.py <../../labs/gpu/44_streams_clusters.py>`.

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

:::{admonition} Try it 1 :class: exercise

Give uploads and downloads one shared copy resource. Draw a new schedule and explain why the three-independent-stage formula may be too optimistic. :::

:::{admonition} Try it 2 :class: exercise

Increase the number of pipeline stages in a layer-partitioned model while holding the number of microbatches fixed. Explain how fill and drain bubbles affect utilization. :::

## Summary

| Decision | What you should remember |
|---|---|
| Streams | A stream orders its own operations, and cross-stream dependencies need explicit coordination. |
| Overlap | Only independent work can safely overlap communication. |
| Cluster design | The decomposition determines message volume, latency sensitivity, and ownership. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the ideal eight-chunk time in the toy model?",
    "options": [
      "It is 44 ms.",
      "It is 72 ms.",
      "It is 9 ms."
    ],
    "answer": 0,
    "explain": "The first chunk costs 9 ms, and seven later intervals cost 5 ms each."
  },
  {
    "q": "Do two streams guarantee concurrent execution?",
    "options": [
      "Yes, streams create unlimited resources.",
      "No, dependencies and hardware resources still limit overlap."
    ],
    "answer": 1,
    "explain": "Concurrency is permitted but not guaranteed."
  },
  {
    "q": "What does CUDA-aware MPI primarily allow?",
    "options": [
      "It allows supported MPI operations to accept device buffers.",
      "It guarantees that every kernel is automatically synchronized."
    ],
    "answer": 0,
    "explain": "Buffer support does not replace the implementation\u2019s ordering contract."
  },
  {
    "q": "Which stencil work can overlap an incoming halo?",
    "options": [
      "Interior work that does not read that halo can overlap.",
      "Every boundary computation can always overlap."
    ],
    "answer": 0,
    "explain": "Boundary work must wait when it depends on the new halo."
  },
  {
    "q": "Does nonblocking MPI guarantee independent background progress?",
    "options": [
      "Yes, the name guarantees it.",
      "No, progress depends on the implementation and environment."
    ],
    "answer": 1,
    "explain": "A real overlap claim needs measurement and the runtime\u2019s documented behavior."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 19, 20. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) describes GPU collective communication. NCCL and MPI provide different interfaces, and neither removes the application’s dependency requirements.
