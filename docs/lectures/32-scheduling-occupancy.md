# 32. Warps, scheduling, divergence, and occupancy

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`31-cuda-execution`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will distinguish a software thread block from a hardware execution resource.
- You will calculate a resource-limited occupancy model.
- You will explain why maximum occupancy does not guarantee minimum runtime. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/32_scheduling_occupancy.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`32_scheduling_occupancy.py <../../labs/gpu/32_scheduling_occupancy.py>`.

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

:::{admonition} Try it 1 :class: exercise

Change the block size to 128 and recompute each limit. Explain why the same occupied thread count can coexist with a different number of blocks and barriers. :::

:::{admonition} Try it 2 :class: exercise

Construct a hypothetical kernel that doubles register use but halves global-memory loads. State the measurements you would need to decide whether its lower occupancy is worthwhile. :::

## Summary

| Decision | What you should remember |
|---|---|
| Occupancy | Occupancy describes resident warps relative to a hardware limit. |
| Divergence | Different lane paths can reduce useful work per issued instruction. |
| Barriers | A barrier is needed when cooperating threads depend on one another’s data. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What limits resident blocks in the 64-register toy case?",
    "options": [
      "The register budget limits them.",
      "The number of input arrays limits them.",
      "The output value limits them."
    ],
    "answer": 0,
    "explain": "Each block needs 16,384 registers, so four blocks consume 65,536."
  },
  {
    "q": "Does higher occupancy always mean lower runtime?",
    "options": [
      "Yes, it always determines runtime.",
      "No, throughput and data reuse also matter."
    ],
    "answer": 1,
    "explain": "A lower-occupancy kernel can move less data or expose more useful instruction-level parallelism."
  },
  {
    "q": "How many threads are in a CUDA warp?",
    "options": [
      "There are 16 threads.",
      "There are 32 threads.",
      "There are 256 threads."
    ],
    "answer": 1,
    "explain": "The CUDA execution model uses 32-thread warps."
  },
  {
    "q": "Can a block barrier synchronize two ordinary blocks?",
    "options": [
      "Yes, every barrier is global.",
      "No, it is scoped to one block."
    ],
    "answer": 1,
    "explain": "Cross-block dependencies need an appropriate global coordination mechanism."
  },
  {
    "q": "Why can register caps hurt performance?",
    "options": [
      "They can cause spills to device-backed local memory.",
      "They always increase shared-memory capacity."
    ],
    "answer": 0,
    "explain": "A spill replaces register storage with memory traffic."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 4. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
