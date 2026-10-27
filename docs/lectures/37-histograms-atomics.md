# 37. Histograms, atomic updates, and privatization

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`36-stencils`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will explain the lost-update race in a histogram.
- You will derive when block-private bins reduce global updates.
- You will distinguish atomicity from ordering a larger computation. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/37_histograms_atomics.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## Counting creates competing writers

A **histogram** counts how many observations fall in each bin. The serial operation is simple. The program reads a value, computes its bin, and increments that bin’s count. A parallel implementation creates a conflict because many threads can choose the same bin.

An increment is normally a read-modify-write sequence. Two threads can both read the old count 7, both compute 8, and both write 8. The correct result 9 is lost. An **atomic addition** makes the update indivisible with respect to competing updates at its specified scope. It prevents this lost update, but it does not guarantee that the overall algorithm is fast.

An atomic update also does not establish arbitrary program-wide ordering. A thread that increments a completion counter may need explicit memory-ordering semantics before another thread can safely consume separate data. A histogram of integer counts needs only the atomic bin updates and proper ordering between initialization, counting, and later use. More complicated producer-consumer protocols require a separate memory-model argument.

## Contention depends on the distribution

A uniform distribution over many bins spreads updates. A highly skewed distribution sends many threads to the same address. Both inputs contain the same number of observations, but the second creates much more contention. A benchmark using only uniformly random input can conceal the worst case.

Suppose eight observations map to bins $[0,0,0,0,1,1,2,3]$. A direct implementation performs eight global atomic updates. The count for bin 0 receives four competing contributions. A one-bin input is an extreme stress test, while a one-observation-per-bin input stresses a different resource pattern.

Integer counter width matters. A 32-bit unsigned bin overflows after sufficiently many contributions. An implementation must choose a type that can represent the maximum count, not just the typical count. Input validation also matters because an invalid bin index becomes an out-of-bounds write.

## Privatization moves the contention

**Privatization** gives each block its own temporary histogram. Threads within the block update local bins, often in shared memory. The block then contributes its local counts to the global result. If a block processes $M$ observations into $K$ bins, the merge can require at most $K$ nonzero global updates instead of $M$ updates.

The optimization works best when each block processes many observations relative to the number of bins. It also has costs. The block must initialize its private bins, perform its local updates, synchronize, and merge. A large bin count can exceed convenient shared-memory capacity or make initialization expensive. Several levels of privatization can trade local storage against contention.

A warp-private histogram can reduce contention between warps but replicates the bins more times. A per-thread histogram eliminates update conflicts within a thread but uses much more storage. The correct granularity depends on the input distribution, bin count, and resource budget.

## Coarsening and aggregation reduce update count differently

Thread coarsening assigns multiple observations to one thread. The input access pattern should remain coalesced across neighboring lanes on each iteration. Giving each lane a large contiguous chunk can create strided warp accesses when all lanes process their first element.

**Aggregation** combines repeated contributions before issuing an atomic operation. A thread that sees a run of the same bin can accumulate a local count and perform one update for the run. A warp can also combine lanes targeting the same bin with suitable grouping primitives. The benefit depends on repetition. Aggregation adds work when almost every target is different.

Privatization reduces the scope of contention, while aggregation reduces the number of updates. They can be combined. A block-private histogram can receive aggregated thread or warp contributions and later merge to global memory. Each level should have an explicit correctness argument and a cost model.

## Initialization and merging are algorithm stages

The global histogram must begin at zero unless accumulation across batches is intentional. A kernel launch in a stream must follow the initialization operation in a valid order. Inside a block-private kernel, all bins must be initialized before any thread updates them. A barrier after initialization establishes that dependency.

Another barrier is needed before the block reads its completed private histogram for merging. Threads that have no input work still participate in initialization and synchronization. A block whose final input chunk is partial must not let inactive workers skip a required barrier.

The reference compares direct counting and block-private counting. It also counts the number of nonzero global contributions in the private design. The arithmetic result is exact for these small integer counts. The contribution count is a structural property, not a timing result for shared-memory or global-memory atomics.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`37_histograms_atomics.py <../../labs/gpu/37_histograms_atomics.py>`.

```python
import numpy as np
values = np.array([0,0,0,0,1,1,2,3,0,0,1,1,2,2,3,3])
bins, chunk = 4, 8
direct = np.bincount(values, minlength=bins)
private = np.zeros(bins, dtype=np.int64)
merge_updates = 0
for start in range(0, len(values), chunk):
    local = np.bincount(values[start:start+chunk], minlength=bins)
    merge_updates += np.count_nonzero(local)
    private += local
assert np.array_equal(direct, private)
print("histogram=", private.tolist())
print(f"direct_global_updates={len(values)}, private_nonzero_merges={merge_updates}")
skewed = np.zeros(16, dtype=int)
print(f"one_bin_private_merges={sum(np.count_nonzero(np.bincount(skewed[s:s+chunk], minlength=bins)) for s in range(0,16,chunk))}")
```

The recorded run produced the following output.

```text
histogram= [6, 4, 3, 3]
direct_global_updates=16, private_nonzero_merges=8
one_bin_private_merges=2
```

**The private design preserves the histogram while reducing the counted global contributions from 16 to 8.** The one-bin example needs only two merge contributions. These counts do not include initialization or local atomic costs.

:::{admonition} Try it 1 :class: exercise

Increase the number of bins until each observation occupies a different bin. Explain why privatization may then save few global updates while retaining its setup costs. :::

:::{admonition} Try it 2 :class: exercise

Change the chunk size to 4 and 16. Predict the number of private histograms and nonzero merges for both the mixed and one-bin inputs. :::

## Summary

| Decision | What you should remember |
|---|---|
| Atomic addition | An atomic bin update prevents lost increments. |
| Privatization | Local histograms reduce the number of global contributions when bins are reused. |
| Distribution | Skew changes contention and aggregation opportunities. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What can two unsynchronized increments of 7 produce?",
    "options": [
      "They can incorrectly produce 8.",
      "They must produce 9."
    ],
    "answer": 0,
    "explain": "Both threads can read 7 before either writes 8."
  },
  {
    "q": "How many global contributions does the private mixed example count?",
    "options": [
      "It counts eight.",
      "It counts sixteen.",
      "It counts one."
    ],
    "answer": 0,
    "explain": "Each of the two chunks contributes to four bins."
  },
  {
    "q": "Does atomic addition make every multi-step protocol correct?",
    "options": [
      "Yes, one atomic operation orders everything.",
      "No, separate data can require memory-ordering reasoning."
    ],
    "answer": 1,
    "explain": "Atomicity of one location is narrower than a complete producer-consumer protocol."
  },
  {
    "q": "Which input creates extreme single-bin contention?",
    "options": [
      "An input whose values all select one bin does.",
      "An input with a distinct bin for every value does."
    ],
    "answer": 0,
    "explain": "All updates compete for the same location."
  },
  {
    "q": "Why can very many bins make privatization unattractive?",
    "options": [
      "The local storage and initialization cost grow.",
      "The count values stop being integers."
    ],
    "answer": 0,
    "explain": "Every private histogram requires storage and setup."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 9. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
