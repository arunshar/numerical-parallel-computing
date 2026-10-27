# 39. Prefix sums and work-efficient coordination

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`38-reductions`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will distinguish inclusive and exclusive prefix sums.
- You will compare logarithmic-depth scan algorithms by total work.
- You will use a scan to allocate output positions without a global counter per item. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/39_prefix_scan.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## A scan preserves every partial result

A **prefix sum**, or scan, returns a cumulative result at every position. An inclusive sum returns $s_i=\sum_{j=0}^{i}x_j$. An exclusive sum returns $e_i=\sum_{j=0}^{i-1}x_j$, with $e_0=0$. For the input $[3,1,4,1]$, the inclusive result is $[3,4,8,9]$, and the exclusive result is $[0,3,4,8]$.

A reduction keeps only the final aggregate. A scan retains the earlier aggregates as well. That extra information lets threads agree on where to write variable amounts of output. If thread $i$ will emit $c_i$ elements, an exclusive scan of the counts gives its starting offset. The total output length is the final offset plus the final count.

A filter uses the same idea with counts of zero or one. Each thread evaluates a predicate, the scan computes positions, and retained elements write to those positions. This process is called stream compaction. It preserves input order when the predicate and offsets follow the original order.

## A short critical path can still do too much work

The Kogge-Stone or iterative-doubling style of scan performs stages with distances 1,2,4, and so on. At a stage with distance $d$, position $i$ combines its previous value with position $i-d$ when that predecessor exists. A separate old-stage view is necessary because the stage is logically parallel. An in-place left-to-right CPU loop would accidentally reuse updates from the current stage.

The depth is logarithmic, but many positions perform an addition at every stage. The total work is $O(N\log N)$. This extra work can be acceptable for a short warp-sized sequence where low latency and simple communication matter. It becomes more expensive for a large array.

A **work-efficient scan** uses $O(N)$ additions and $O(\log N)$ depth. A common Blelloch construction builds a reduction tree in an up-sweep, sets the root to the identity, and distributes prefixes in a down-sweep. Brent-Kung-style prefix networks similarly reduce the number of active combine operations through a tree structure. Their exact wiring and boundary handling differ, so names should not substitute for a concrete dependency diagram.

## The down-sweep carries a prefix into each subtree

At an internal tree node, the parent’s prefix is the aggregate of every input before that subtree. The left child receives the same prefix. The right child receives that prefix combined with the total of the left subtree. This rule explains the exclusive result without treating the algorithm as a mysterious sequence of swaps.

For eight inputs, the up-sweep uses seven additions. The down-sweep uses seven more additions. The fourteen additions are linear in the input length, although the scan computes eight outputs. Padding to a power of two with the identity simplifies the tree but adds work for non-power-of-two arrays.

The operation need not be addition. It must support a suitable associative combination and identity. Operand order matters for associative but noncommutative operations. A scan implementation that swaps the order while moving values through the tree can be wrong even though addition tests pass.

## Large arrays need a hierarchy

A block can scan a tile and write its total. Another stage scans the block totals. A final stage adds each block’s offset to its local result. The local scan, block-total scan, and offset addition form an easy-to-verify hierarchy. The extra passes move data more than once, which can become bandwidth-expensive.

More advanced single-pass scans use carefully designed inter-block communication protocols to obtain prior prefixes without a separate full-array adjustment pass. Such protocols require precise memory ordering and progress arguments. They should be implemented from a well-tested library or a fully specified algorithm, not improvised from a spinning global counter.

Thread coarsening can scan a small local sequence per thread. The block then scans thread totals and adds the resulting offsets to the local sequences. Coarsening reduces inter-thread communication but increases register demand. A coalesced load layout may differ from the convenient per-thread scan layout, requiring a deliberate transpose or indexing strategy.

## Segments mark independent sequences

A **segmented scan** performs separate scans within one packed array. A flag marks the beginning of each segment. The combine operation carries both a value and a flag so that a new segment resets the accumulated prefix. This pattern appears in grouped reductions, sparse data structures, and variable-length sequences.

A segmented scan requires clear handling of an empty segment and of a segment crossing a block boundary. The flags are part of the mathematical input. A local block cannot simply reset its first value unless that position is a real segment boundary.

The experiment compares an inclusive doubling scan with an exclusive Blelloch scan. It then uses exclusive positions to compact even values. The tests establish the toy algorithms’ outputs and work counts. They do not demonstrate a GPU scan implementation or a single-pass memory-ordering protocol.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`39_prefix_scan.py <../../labs/gpu/39_prefix_scan.py>`.

```python
import numpy as np
x = np.array([3,1,4,1,5,9,2,6], dtype=np.int64)
a = x.copy(); doubling_adds = 0
for distance in [1,2,4]:
    old = a.copy()
    a[distance:] = old[distance:] + old[:-distance]
    doubling_adds += len(x)-distance
assert np.array_equal(a, np.cumsum(x))
b = x.copy(); adds = 0; n = len(b)
stride = 2
while stride <= n:
    for end in range(stride-1,n,stride):
        b[end] += b[end-stride//2]; adds += 1
    stride *= 2
b[-1] = 0
stride = n
while stride >= 2:
    for end in range(stride-1,n,stride):
        left = end-stride//2
        saved = b[left]; b[left] = b[end]; b[end] += saved; adds += 1
    stride //= 2
assert np.array_equal(b, np.r_[0,np.cumsum(x)[:-1]])
flags = (x % 2 == 0).astype(int)
offsets = np.r_[0,np.cumsum(flags)[:-1]]
out = np.empty(int(flags.sum()), dtype=x.dtype)
for i in range(n):
    if flags[i]: out[offsets[i]] = x[i]
print("inclusive=", a.tolist())
print("exclusive=", b.tolist())
print(f"doubling_additions={doubling_adds}, tree_additions={adds}")
print("compacted_even_values=", out.tolist())
```

The recorded run produced the following output.

```text
inclusive= [3, 4, 8, 9, 14, 23, 25, 31]
exclusive= [0, 3, 4, 8, 9, 14, 23, 25]
doubling_additions=17, tree_additions=14
compacted_even_values= [4, 2, 6]
```

**Both scans produce the expected prefixes, but their work counts differ.** The eight-element doubling scan uses 17 additions, while the tree uses 14. The asymptotic difference becomes more important as the array grows.

:::{admonition} Try it 1 :class: exercise

Extend the input to 16 elements and compare addition counts. Derive the doubling count by summing the number of active positions at each distance. :::

:::{admonition} Try it 2 :class: exercise

Replace the even-value predicate with a count from zero to three per item. Use an exclusive scan to assign disjoint ranges and verify that the ranges exactly fill the output. :::

## Summary

| Decision | What you should remember |
|---|---|
| Inclusive scan | Each result includes its current input. |
| Exclusive scan | Each result gives the aggregate before its current input. |
| Work efficiency | A shallow algorithm can still perform too many total operations. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the first exclusive sum?",
    "options": [
      "It is the identity zero.",
      "It is the first input value."
    ],
    "answer": 0,
    "explain": "No input precedes position zero."
  },
  {
    "q": "What work does iterative doubling require asymptotically?",
    "options": [
      "It requires O(N log N) work.",
      "It requires O(1) work."
    ],
    "answer": 0,
    "explain": "Many positions combine at each logarithmic stage."
  },
  {
    "q": "How many additions does the eight-element tree perform here?",
    "options": [
      "It performs 14 additions.",
      "It performs 64 additions."
    ],
    "answer": 0,
    "explain": "The up-sweep and down-sweep each use seven additions."
  },
  {
    "q": "What does a compaction scan compute?",
    "options": [
      "It computes output positions for retained items.",
      "It sorts all values by magnitude."
    ],
    "answer": 0,
    "explain": "The prefix of keep flags assigns each retained item a unique stable position."
  },
  {
    "q": "Can a block boundary automatically reset a segmented scan?",
    "options": [
      "Yes, every block begins a new mathematical segment.",
      "No, only a real segment flag permits the reset."
    ],
    "answer": 1,
    "explain": "A segment can cross a block boundary."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 11. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
