# 25. Parallel sorting, merging, and selection

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`24-dense-matrix-cannon` used regular blocks with predictable communication. Sorting adds data-dependent destinations and an ordering requirement. This lecture compares fixed sorting networks with partition-based methods and explains why communication and load balance can dominate the comparisons.

:::{admonition} Learning goals
- You can trace odd-even transposition and a bitonic compare-exchange stage.
- You can explain parallel quicksort, sample sort, radix sort, and co-ranked merge.
- You can verify ordering and permutation preservation separately.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/25-sorting-merging.py`.

## Sorting has two correctness obligations

A sorting algorithm must put keys in the required order and preserve the input multiset. Checking only adjacent order can miss a program that silently drops difficult elements. Checking only a checksum can miss collisions or duplicate errors. A small test can compare the full result with a trusted serial sort. Larger tests should combine ordering, counts, and an appropriate permutation check. A stable sort also preserves the input order of equal keys when that order matters.

A **comparison sort** learns order by comparing keys. Its sequential comparison lower bound is order $n\log n$ for arbitrary distinct inputs. Parallel time depends on how many comparisons can occur at once and how data moves between workers. Integer-key methods such as radix sort use key structure and are not restricted to the same comparison model. A complexity claim should name that model before comparing algorithms.

## Sorting networks use a fixed communication schedule

A comparator receives two keys and emits their minimum and maximum in prescribed positions. A **sorting network** fixes the comparator graph independently of the input values. Comparators in one stage can execute in parallel when they use disjoint positions. The network depth gives a critical-path bound, while the comparator count gives work. Hardware and SIMD implementations can benefit from a predictable schedule, but a fixed network can perform unnecessary comparisons on nearly sorted data.

Odd-even transposition alternates comparisons of positions(0, 1),(2, 3),and so on with positions(1, 2),(3, 4),and so on. Repeating enough phases moves small values left and large values right. With one key per processor, the communication is local, but the number of phases is linear in the number of keys. A block version first sorts local blocks and then uses compare-split operations with neighbors, retaining the smaller or larger half as appropriate.

A **bitonic sequence** rises and then falls under a cyclic interpretation. A bitonic merge compares positions separated by half the sequence length, places the smaller values in one half, and recursively merges each half. Building bitonic sequences from smaller sorted runs produces a network with order $\log^2n$ depth and order $n\log^2n$ work for power-of-two input sizes. Padding needs a sentinel that preserves the requested order and does not conflict with valid data.

Odd-even mergesort recursively merges even-indexed and odd-indexed subsequences before a cleanup compare-exchange stage. Its name resembles odd-even transposition, but the algorithms have different structures and complexity. Keeping the full algorithm name in notes prevents that common confusion. Network sorting is particularly useful for small fixed-size blocks inside a larger algorithm.

## Partition-based sorting follows the data

Parallel quicksort partitions values around a pivot and recursively sorts the two sides. A balanced pivot exposes useful work, while a poor pivot can create one almost-empty branch and one nearly unchanged branch. Distributed quicksort must also move values to the ranks responsible for each partition. Choosing a pivot therefore affects both computation and communication. Randomization or sampling can reduce the chance of severe imbalance without eliminating every difficult input.

**Sample sort** first sorts local data, selects samples, chooses splitters, partitions each local array into buckets, exchanges destination buckets, and merges or sorts the received data. The splitters define global key ranges. Duplicate-heavy data can overload one bucket unless the algorithm defines how equal keys are distributed. Oversampling improves splitter quality at additional sample-processing cost. The all-to-all exchange can require a large temporary buffer, so peak memory belongs in the design.

**Bucket sort** is especially attractive when the key distribution or range is known. Bad distribution estimates can concentrate work in one bucket. **Radix sort** processes digits of integer or encoded keys. A pass counts digit frequencies, scans the counts to obtain offsets, and scatters items into output positions. A stable least-significant-digit pass preserves the ordering produced by earlier passes. Increasing the radix reduces pass count but increases histogram and offset storage.

## Parallel merging divides output positions

Two sorted arrays can be merged sequentially in linear work. A naive parallelization that gives each worker the same number of elements from each input can produce unequal output work because the keys may interleave unevenly. **Co-ranking** instead chooses a target output position $k$ and finds input prefixes of lengths $i$ and $j$ with $i+j=k$ whose elements exactly form that output prefix.

The boundary conditions compare the last included elements with the first excluded elements. A binary search finds a valid partition in logarithmic time. Workers compute boundaries for disjoint output intervals and merge only the corresponding input slices. Each worker then writes a distinct output region, avoiding locks. Tie-breaking rules determine stability when equal keys occur in both inputs.

Parallel selection finds an order statistic rather than fully sorting everything. Partitioning around candidate pivots can discard irrelevant key ranges. This can save work when only a median or top-k set is needed. A full sort is often the wrong baseline for a task that asks only for one threshold. The algorithm should match the output requirement before its implementation is parallelized.

## A phase trace reveals both progress and limits

The experiment sorts eight integers with odd-even transposition. Each phase uses disjoint adjacent comparisons, and the code records the array after every phase. The final comparison with Python's sorted result checks both ordering and multiset preservation for this small input. The simulation makes the phase structure visible but performs its comparisons sequentially in Python. It therefore demonstrates algorithmic parallelism rather than measured speedup.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
source=[7,2,6,3,8,1,5,4]
a=source.copy();phases=[]
for phase in range(len(a)):
 for i in range(phase%2,len(a)-1,2):
  if a[i]>a[i+1]:a[i],a[i+1]=a[i+1],a[i]
 phases.append(a.copy())
assert a==sorted(source)
print(json.dumps({'phases':phases,'sorted':a,'preserves_multiset':True},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "phases": [
    [
      2,
      7,
      3,
      6,
      1,
      8,
      4,
      5
    ],
    [
      2,
      3,
      7,
      1,
      6,
      4,
      8,
      5
    ],
    [
      2,
      3,
      1,
      7,
      4,
      6,
      5,
      8
    ],
    [
      2,
      1,
      3,
      4,
      7,
      5,
      6,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      7,
      6,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ]
  ],
  "sorted": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
  ],
  "preserves_multiset": true
}
```

The final array is **1 through 8 in increasing order**. The trace shows that one phase does not solve the entire problem even though several comparisons are independent. Each phase depends on the previous phase’s outputs, which creates the algorithm’s span.

:::{admonition} Try it
:class: exercise
You can add repeated keys and label each key with its original index. Determine whether your comparison rule preserves stable order.
:::

:::{admonition} Try it
:class: exercise
You can change the input to descending order and compare the trace with an already sorted input. Explain why a fixed network can still perform the same comparisons.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Networks | A fixed comparator graph exposes predictable stages. |
| Partition methods | Good splitters balance work and data movement. |
| Validation | Sorted order and multiset preservation are separate obligations. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Is adjacent sorted order enough to prove no values were lost?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "An algorithm can return an ordered but incomplete output."
  },
  {
    "q": "What does a sorting network fix in advance?",
    "options": [
      "The comparator schedule.",
      "The input values."
    ],
    "answer": 0,
    "explain": "Its structure does not depend on comparison outcomes."
  },
  {
    "q": "Why can sample sort become imbalanced?",
    "options": [
      "Splitters or duplicate handling can create uneven buckets.",
      "Every bucket always has equal size."
    ],
    "answer": 0,
    "explain": "Data distribution controls bucket sizes."
  },
  {
    "q": "What does co-ranking partition?",
    "options": [
      "Positions in the merged output.",
      "Only equal-sized input prefixes."
    ],
    "answer": 0,
    "explain": "Output boundaries define balanced merge tasks."
  },
  {
    "q": "Does radix sort use only pairwise key comparisons?",
    "options": [
      "No, it uses digit structure.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Its complexity model differs from comparison sorting."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
