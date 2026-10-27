# 35. Convolution, constant memory, and halo tiles

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`34-gpu-performance`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will state the boundary convention before implementing a convolution.
- You will derive the input region needed by an output tile.
- You will compare direct loads, shared-memory tiles, and cached halo reads. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/35_convolution.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

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

The complete implementation is shown below. The source file is also available as {download}`35_convolution.py <../../labs/gpu/35_convolution.py>`.

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

:::{admonition} Try it 1 :class: exercise

Reverse the asymmetric filter and compare the outputs. Explain the difference between convolution and cross-correlation using the value at the first pixel. :::

:::{admonition} Try it 2 :class: exercise

Replace zero padding with periodic padding. Derive the expected first and last outputs before changing the implementation. :::

## Summary

| Decision | What you should remember |
|---|---|
| Boundary convention | Padding defines the mathematical operation at the edges. |
| Halo | An output tile requires neighboring input values beyond its output region. |
| Coefficient reuse | Uniform coefficient reads can suit constant memory. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What input width supports an output width 8 and radius 1?",
    "options": [
      "The input width is 8.",
      "The input width is 10.",
      "The input width is 16."
    ],
    "answer": 1,
    "explain": "The input tile adds one halo value on each side."
  },
  {
    "q": "Do asymmetric convolution and cross-correlation always agree?",
    "options": [
      "Yes, the filter direction never matters.",
      "No, reversing an asymmetric filter changes the result."
    ],
    "answer": 1,
    "explain": "Convolution and cross-correlation use opposite filter directions."
  },
  {
    "q": "What should padded loader threads do before a block barrier?",
    "options": [
      "They should write the neutral value and participate.",
      "They should always return immediately."
    ],
    "answer": 0,
    "explain": "The cooperative tile must be complete before computation."
  },
  {
    "q": "When is a two-pass separable implementation exact?",
    "options": [
      "It is exact when the filter factors into the two one-dimensional filters.",
      "It is exact for every arbitrary filter."
    ],
    "answer": 0,
    "explain": "The outer-product factorization defines separability."
  },
  {
    "q": "Does a smaller halo ratio prove a faster GPU kernel?",
    "options": [
      "Yes, it contains every cost.",
      "No, it omits other hardware costs."
    ],
    "answer": 1,
    "explain": "Tile size also affects occupancy and synchronization."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 7. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
