# 33. Memory spaces, data reuse, and tiled matrix multiplication

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`32-scheduling-occupancy`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will derive the global-load reduction of a square matrix tile.
- You will separate global coalescing from shared-memory bank behavior.
- You will explain the two barriers in a tiled matrix product. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/33_memory_tiling.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`33_memory_tiling.py <../../labs/gpu/33_memory_tiling.py>`.

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

:::{admonition} Try it 1 :class: exercise

Change the matrix shapes to 1 by 9 and 9 by 3. Explain which tiles are partial and why slicing remains correct in the CPU reference. Describe the explicit zero padding that a fixed-size CUDA tile would need. :::

:::{admonition} Try it 2 :class: exercise

Add an output-store term to the arithmetic-intensity estimate for a complete square product. Derive how its importance changes as the inner dimension grows. :::

## Summary

| Decision | What you should remember |
|---|---|
| Tiling | A block loads a tile once and reuses it across several outputs. |
| Synchronization | The block waits after loading and before replacing a tile. |
| Layout | Global coalescing and shared-memory bank behavior require separate reasoning. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the input-only intensity of a tile with width 16?",
    "options": [
      "It is 0.25 FLOP/byte.",
      "It is 4 FLOP/byte.",
      "It is 16 FLOP/byte."
    ],
    "answer": 1,
    "explain": "The model gives T/4, which equals four."
  },
  {
    "q": "Why is the second tile barrier needed?",
    "options": [
      "It protects readers before the tile is overwritten.",
      "It copies the output to the host."
    ],
    "answer": 0,
    "explain": "A fast loader must wait until every thread has finished reading the old stage."
  },
  {
    "q": "Does local memory always mean on-chip memory?",
    "options": [
      "Yes, local means physically close.",
      "No, CUDA local memory can be device-backed."
    ],
    "answer": 1,
    "explain": "Local describes thread-private addressability."
  },
  {
    "q": "What should an out-of-range tile load contribute?",
    "options": [
      "It should contribute zero.",
      "It should contribute the old tile value."
    ],
    "answer": 0,
    "explain": "Zero preserves the dot product over the valid range."
  },
  {
    "q": "What does padding a modeled 32-word row to 33 words change?",
    "options": [
      "It changes shared-memory bank mapping.",
      "It changes the mathematical product."
    ],
    "answer": 0,
    "explain": "Padding affects layout without changing the logical data."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 5. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
