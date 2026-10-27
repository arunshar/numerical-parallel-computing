# 36. Stencil computations and space-time dependencies

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`35-convolution`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will derive the dependency neighborhood of a stencil update.
- You will explain why separate input and output buffers preserve a time step.
- You will compare spatial tiling, register rolling, and temporal blocking. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/36_stencils.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## A stencil applies the same neighborhood rule repeatedly

A **stencil** updates each point from a fixed pattern of nearby points. A one-dimensional diffusion update uses the center and its two neighbors. A two-dimensional five-point stencil uses the center and four axial neighbors. A three-dimensional seven-point stencil adds the two neighbors along the third axis. The pattern describes dependencies, while the coefficients describe the mathematical model.

For the heat equation $u_t=\alpha u_{xx}$, a forward-Euler time step and centered spatial difference give

$$u_i^{n+1}=u_i^n+\lambda(u_{i-1}^n-2u_i^n+u_{i+1}^n),\qquad\lambda=\alpha\Delta t/\Delta x^2.$$

The superscript denotes time, not a power. Every value on the right belongs to time step $n$. Updating the array in place from left to right changes that dependency. Later positions would use some already updated neighbors and some old neighbors. The result would be a different method whose properties need separate analysis.

The reference uses two buffers and swaps their roles after each complete step. Fixed zero boundary values represent a rod whose ends are held at zero. Periodic boundaries, prescribed fluxes, and time-varying boundary data require different indexing and physical interpretations.

## Numerical stability is part of the kernel contract

The one-dimensional explicit diffusion scheme has a familiar stability restriction $0\leq\lambda\leq1/2$ under the standard uniform-grid setting. The GPU cannot repair a time step that violates the mathematical stability condition. A fast kernel can produce an unstable solution more quickly.

For a two-dimensional equal-spacing diffusion scheme, the corresponding coefficient restriction becomes more severe because each point exchanges information with more neighbors. The precise condition depends on the operator and discretization. The correct workflow derives or consults the stability condition, then chooses the kernel implementation.

A stencil is often bandwidth-sensitive because it performs few operations per value. Neighboring outputs reuse input values, so caching and tiling can reduce memory traffic. The useful quantity is the traffic reaching each memory level, not merely the number of source-level array references. A compiler or cache can already remove some repeated traffic.

## A tile has a surface and an interior

A tile of side $T$ with a one-cell halo stores $(T+2)^d$ values in $d$ dimensions and updates $T^d$ interior values. The halo-to-interior ratio worsens for small tiles. In three dimensions, a full shared-memory cube can become too large before the surface overhead becomes attractive.

**Register tiling** can avoid storing an entire cube. A thread processes several positions along one axis while keeping a small rolling window in registers. The next iteration shifts the window and loads one new plane or value. Neighboring threads still cooperate across the other dimensions. This arrangement trades registers and control logic for fewer loads.

A register window must preserve the correct time level. Reusing a register is safe only after every dependent output has consumed its value. A rolling implementation is therefore easiest to derive from a dependency diagram. The diagram shows the current output, the values it reads, and the last use of each input.

## Temporal blocking expands the dependency cone

**Temporal blocking** performs several time steps while a tile remains close to the processor. One time step needs a radius-one halo. Two time steps need information from two cells away, and $k$ steps need a halo wide enough for the $k$-step dependency cone. A block can compute several steps on a progressively shrinking valid interior.

The technique can reduce global traffic, but it performs redundant work near overlapping tile boundaries and consumes more local storage. A larger time block is not automatically better. The design balances reuse against redundant computation, synchronization, and occupancy. Distributed-memory stencils add another constraint because halos must be exchanged between ranks.

The straightforward multi-kernel implementation uses one kernel per time step. The kernel boundary supplies a global ordering point for the next step in the same stream. A single ordinary kernel cannot replace that ordering with a block barrier. Cooperative launches or specialized algorithms can provide broader coordination, but they introduce requirements that must be checked explicitly.

## Conservation and symmetry make useful tests

A symmetric initial impulse should remain symmetric under a symmetric stencil and symmetric boundaries. Periodic diffusion should preserve the sum up to rounding for the matching discrete operator. Fixed-value boundaries can exchange heat with the domain, so the sum need not remain constant. These properties test the mathematics more strongly than comparing one printed element.

The experiment checks symmetry, nonnegativity, and the result of four small stable steps. It uses an exact binary coefficient, which keeps this particular toy arithmetic easy to inspect. A production solver still needs convergence tests as the spatial and temporal grids are refined.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`36_stencils.py <../../labs/gpu/36_stencils.py>`.

```python
import numpy as np
u = np.zeros(9)
u[4] = 1.0
lam = 0.25
for step in range(4):
    new = u.copy()
    new[1:-1] = u[1:-1] + lam * (u[:-2] - 2*u[1:-1] + u[2:])
    new[0] = new[-1] = 0.0
    u = new
assert np.allclose(u, u[::-1])
assert np.min(u) >= 0
print("four_step_solution=", u.tolist())
print(f"sum={u.sum():.8f}, maximum={u.max():.8f}")
for dimension in [1, 2, 3]:
    print(f"dimension={dimension}, tile_width=8, storage_ratio={(10/8)**dimension:.6f}")
```

The recorded run produced the following output.

```text
four_step_solution= [0.0, 0.03125, 0.109375, 0.21875, 0.2734375, 0.21875, 0.109375, 0.03125, 0.0]
sum=0.99218750, maximum=0.27343750
dimension=1, tile_width=8, storage_ratio=1.250000
dimension=2, tile_width=8, storage_ratio=1.562500
dimension=3, tile_width=8, storage_ratio=1.953125
```

**The stable update preserves symmetry and nonnegativity in this example.** The total decreases slightly because the fixed boundaries remove heat. The storage ratios quantify halo overhead without claiming a GPU speedup.

:::{admonition} Try it 1 :class: exercise

Use a periodic boundary rule and verify conservation of the sum over 20 steps. Explain why the original fixed-boundary conservation expectation would be wrong. :::

:::{admonition} Try it 2 :class: exercise

Increase lambda above one half and observe a longer run. Describe the mathematical instability separately from an indexing or synchronization bug. :::

## Summary

| Decision | What you should remember |
|---|---|
| Time levels | Double buffering preserves the chosen explicit method. |
| Stability | The time step must satisfy the discretization’s requirements. |
| Temporal reuse | Several local time steps require a wider dependency halo. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Why does this explicit update use two buffers?",
    "options": [
      "It keeps every right-hand-side value at the old time level.",
      "It doubles the mathematical time step."
    ],
    "answer": 0,
    "explain": "In-place updates would mix time levels."
  },
  {
    "q": "What is the standard1D diffusion bound in this lesson?",
    "options": [
      "Lambda must be at most one half.",
      "Lambda must be at least two."
    ],
    "answer": 0,
    "explain": "The centered explicit scheme is stable for lambda between zero and one half."
  },
  {
    "q": "How wide is the dependency radius after k radius-one steps?",
    "options": [
      "It can grow to k cells.",
      "It remains one cell."
    ],
    "answer": 0,
    "explain": "Each step propagates dependence by one additional cell."
  },
  {
    "q": "Does a block barrier order the whole grid?",
    "options": [
      "Yes, it orders every block.",
      "No, it coordinates one block."
    ],
    "answer": 1,
    "explain": "A multi-step stencil needs a valid global ordering mechanism."
  },
  {
    "q": "Should fixed-zero boundaries preserve total heat?",
    "options": [
      "They always preserve it.",
      "They can remove heat from the domain."
    ],
    "answer": 1,
    "explain": "The boundary acts as an external reservoir."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 8. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
