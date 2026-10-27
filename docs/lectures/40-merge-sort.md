# 40. Parallel merge, co-ranking, and sorting patterns

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`39-prefix-scan`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will partition a merge by output positions using co-ranks.
- You will explain stable sorting and tie rules.
- You will compare radix sorting with comparison-based sorting. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/40_merge_sort.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## Output ownership makes merge parallel

A **merge** combines two sorted arrays into one sorted array. The serial algorithm repeatedly compares the next unused elements and emits the smaller one. Its output is easy to define, but a single pair of moving pointers creates a serial dependency.

A parallel design divides the output into ranges. For each range boundary $k$, it finds how many items come from the first input and how many come from the second. If those counts are $i$ and $j$, then $i+j=k$. The pair is the boundary’s **co-rank**. Once adjacent co-ranks are known, a worker can merge the corresponding slices independently.

This design balances output volume directly. Splitting only one input into equal pieces can leave workers with very different amounts from the other input. Output partitioning assigns nearly the same number of produced elements even when one input contains mostly small values.

## A co-rank is a binary-search condition

Suppose ties should take values from array $A$ before equal values from array $B$. A valid boundary satisfies the conditions

$$i=0\ \text{or}\ j=|B|\ \text{or}\ A_{i-1}\leq B_j,$$

$$j=0\ \text{or}\ i=|A|\ \text{or}\ B_{j-1}<A_i.$$

The strict inequality in the second condition encodes the tie rule. The counts range from $\max(0,k-|B|)$ to $\min(k,|A|)$ for $i$, with $j=k-i$. Binary search adjusts $i$ until both conditions hold.

A duplicated value is therefore a useful correctness test. An implementation can return numerically sorted values while violating stability. **Stability** means that equal-key elements retain their original relative order under the stated input ordering. It matters when records carry payloads or when multiple stable sorting passes are composed.

## Tiling improves the access pattern after partitioning

Co-ranking determines ownership, but it does not automatically make loads efficient. A tiled merge loads pieces of the input into shared memory and lets threads merge from the local tile. The tile boundary must account for how many values each input contributes. A circular buffer can preserve unused local values while loading the next input portion, reducing repeated movement.

Thread coarsening gives one thread several output values. It can reduce partitioning overhead and use a short serial merge within the assigned range. The tradeoff is less fine-grained parallelism and more per-thread state. A useful implementation chooses a granularity large enough to amortize binary searches but small enough to distribute irregular work.

A merge sort repeatedly merges sorted runs. The first runs can be tiny per-thread or per-block sorts, and later passes merge larger runs. Each pass reads and writes much of the array, so global traffic becomes a major cost. A comparison-based merge sort applies to a broad class of ordered keys, including compound comparison rules.

## Radix sorting uses key digits

A **radix sort** groups keys by a fixed-width digit. A least-significant-digit version processes low digits first and requires each pass to be stable. A later pass preserves the ordering already established by earlier digits within equal current digits.

A digit pass typically computes bucket counts, scans them to get bucket offsets, determines each key’s local rank within its bucket, and scatters the keys. The histogram, scan, and scatter patterns therefore become building blocks. The algorithm avoids arbitrary key comparisons but depends on a representation whose digit ordering matches the desired key ordering.

A wider radix reduces the number of passes but increases the number of buckets and local counters. A narrow radix increases passes and global traffic. Signed integers and floating-point keys require transformations or special handling so that bit order matches numerical order. NaNs and signed zero require an explicit ordering policy rather than an accidental consequence of their bit patterns.

## Sorting networks trade generality for regularity

A bitonic sorting network performs a fixed sequence of compare-exchange operations. Its structure is regular and can suit small blocks. It uses more comparisons than the best comparison sorts for large inputs, but it avoids data-dependent control flow in its network structure. Odd-even methods provide another regular comparison pattern.

A hybrid implementation can sort small tiles with a network and merge them globally. The fastest method depends on input length, key type, payload size, and whether stability is required. A library sort should be the practical baseline because a hand-written implementation must handle these edge cases as well as the main kernel.

The reference verifies every co-rank against a stable merged prefix and uses the boundaries to form two output chunks. Its small least-significant-digit radix example sorts nonnegative integers. It intentionally does not pretend to solve signed or floating-point ordering.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`40_merge_sort.py <../../labs/gpu/40_merge_sort.py>`.

```python
a, b = [1,3,3,8], [2,3,7,9]
def co_rank(k, a, b):
    low, high = max(0,k-len(b)), min(k,len(a))
    while low <= high:
        i = (low+high)//2; j = k-i
        if i > 0 and j < len(b) and a[i-1] > b[j]: high = i-1
        elif j > 0 and i < len(a) and b[j-1] >= a[i]: low = i+1
        else: return i,j
    raise AssertionError("No valid co-rank")
merged = sorted([(v,0,i) for i,v in enumerate(a)] + [(v,1,i) for i,v in enumerate(b)])
for k in range(len(a)+len(b)+1):
    i,j = co_rank(k,a,b)
    assert i == sum(origin==0 for _,origin,_ in merged[:k])
    assert i+j == k
print("middle_co_rank=", co_rank(4,a,b))
print("merged_values=", [v for v,_,_ in merged])
values = [13,2,9,0,15,7]
for shift in [0,2]:
    buckets = [[] for _ in range(4)]
    for value in values: buckets[(value >> shift)&3].append(value)
    values = [value for bucket in buckets for value in bucket]
assert values == [0,2,7,9,13,15]
print("two_pass_radix_result=", values)
```

The recorded run produced the following output.

```text
middle_co_rank= (3, 1)
merged_values= [1, 2, 3, 3, 3, 7, 8, 9]
two_pass_radix_result= [0, 2, 7, 9, 13, 15]
```

**The middle co-rank splits the output after four values while respecting the A-before-B tie rule.** The radix example uses two stable two-bit passes to sort four-bit nonnegative keys.

:::{admonition} Try it 1 :class: exercise

Reverse the tie convention and derive which inequality must become strict. Test the result with payloads that identify the original array and position. :::

:::{admonition} Try it 2 :class: exercise

Compare one four-bit radix pass with two two-bit passes. Count buckets and full-array passes, then explain why neither count alone determines runtime. :::

## Summary

| Decision | What you should remember |
|---|---|
| Co-ranking | Output boundaries determine matching input prefixes. |
| Stability | Equal-key ordering must be preserved when the contract requires it. |
| Radix choice | More buckets reduce passes but increase local bookkeeping. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What identity links a co-rank to output position k?",
    "options": [
      "The counts satisfy i+j=k.",
      "The counts satisfy i*j=k."
    ],
    "answer": 0,
    "explain": "The first k outputs contain i elements from A and j from B."
  },
  {
    "q": "Why does the tie rule use one strict inequality?",
    "options": [
      "It chooses which input\u2019s equal elements come first.",
      "It changes the numerical key values."
    ],
    "answer": 0,
    "explain": "The boundary must agree with the stable merge convention."
  },
  {
    "q": "What must a least-significant-digit radix pass preserve?",
    "options": [
      "It must preserve stability.",
      "It must reverse equal keys."
    ],
    "answer": 0,
    "explain": "Later passes rely on ordering established by earlier digits."
  },
  {
    "q": "Does the reference handle arbitrary floating-point keys?",
    "options": [
      "Yes, raw float bits always sort numerically.",
      "No, it sorts a restricted nonnegative integer example."
    ],
    "answer": 1,
    "explain": "Signed values and floating-point special cases need additional rules."
  },
  {
    "q": "Why can a sorting network help small tiles?",
    "options": [
      "Its compare-exchange structure is regular.",
      "It removes the need to compare values."
    ],
    "answer": 0,
    "explain": "The network fixes the operation pattern while still comparing keys."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 12, 13. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
