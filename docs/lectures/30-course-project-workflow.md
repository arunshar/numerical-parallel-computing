# 30. From course exercises to a reproducible parallel project

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`29-hybrid-quantum` separated several execution models and their evidence limits. This lecture combines the numerical and parallel habits into a practical project workflow. The goal is a result that another reader can rebuild, check, and improve without guessing which algorithm, input, or measurement produced it.

:::{admonition} Learning goals
- You can turn an assignment into a testable problem contract.
- You can design correctness tests and a performance experiment separately.
- You can prepare a source-linked handoff that distinguishes tested and untested behavior.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/30-course-project-workflow.py`.

## A problem contract prevents accidental scope changes

A problem contract defines inputs, outputs, valid shapes, data types, boundary behavior, tolerances, and resource limits. A matrix multiplication routine may accept rectangular matrices and arbitrary sizes, or it may assume fixed 4-by 4 tiles. These are different contracts. A kernel that works on one square size should not be advertised as a general matrix multiplication implementation. A tutorial can simplify a contract, but it must say exactly what it simplifies.

The contract also identifies the mathematical operation. Pairwise batched matrix products followed by a sum differ from one elementwise product or one ordinary matrix product. A convolution needs a definition of padding, stride, filter orientation, and output shape. A histogram needs a bin rule and out-of-range behavior. A reduction needs an operation and an identity. Many parallel bugs begin as disagreements about these details rather than mistakes in thread syntax.

An input generator should create ordinary cases and cases near the contract boundaries. A matrix size that is not divisible by a tile width tests edge handling. An empty partition tests collective counts and pointer logic. Repeated keys test sorting stability or bucket handling. A graph with an isolated vertex tests traversal initialization. A floating-point cancellation example tests whether the chosen tolerance reflects the operation's conditioning.

## A serial reference gives a clear comparison

A small serial implementation is often the easiest correctness oracle. It should express the mathematical definition directly and use an independent implementation path when practical. Comparing a parallel kernel with the same buggy indexing code executed serially is a weak check. NumPy or another established library can provide a reference for suitable operations, but you must still match data types, normalization, and layout conventions.

Exact integer results permit exact comparison when overflow is excluded. Floating-point comparison usually needs absolute and relative tolerances. A common criterion is $|x-y|\leq a+r|y|$, where $a$ protects near-zero values and $r$ scales with the reference magnitude. The tolerances should follow the data type, accumulation length, conditioning, and application requirements. A large tolerance chosen after a failure can hide a real algorithmic error.

Invariant checks complement a reference. A sort preserves the multiset, a probability vector sums to one, a Laplacian annihilates constants, and a partition covers every item exactly once. Metamorphic tests transform an input in a way that predicts a related output. For example, linearity predicts how a matrix-vector product changes when the input is scaled. Such checks can reveal errors on cases where a full reference is expensive.

## A performance hypothesis should name its mechanism

A useful hypothesis says that a specific change reduces a specific cost. Blocking may reduce memory traffic. A reduction may replace repeated atomic updates with private accumulation. A nonblocking exchange may overlap communication with interior computation. A new process grid may reduce per-rank communication volume. Each hypothesis suggests an observable quantity beyond total runtime, which makes a negative result informative.

The baseline should be correct and reasonably configured. A speedup over an intentionally poor implementation can teach one mechanism, but it should be labeled as that comparison. A strong application claim needs a relevant baseline, such as an optimized library or a well-configured previous method. All compared versions should use the same mathematical contract and a justified precision setting.

A benchmark separates setup, transfer, computation, and cleanup when that separation serves the question. An end-to-end measurement should include the costs a real caller pays. A kernel-only measurement is useful for kernel analysis but should not be relabeled as application latency. Warmup, repeated trials, and synchronization boundaries affect results. GPU timing needs completion-aware events or a documented synchronization method because host launch time is not kernel runtime.

## The CSCI 5451 homework sequence teaches several mechanisms

The early shared-memory assignment develops thread-level decomposition and testing. The distributed matrix assignments add communication, process-grid indexing, general dimensions, and load balance. The CUDA assignments then emphasize memory hierarchy, convolution, and small-matrix reuse. These connections explain why a correct serial model remains useful across the whole course. The same reference can validate several execution strategies without pretending their runtime costs are identical.

This study repository uses new toy examples and original code to teach those mechanisms. The instructor's assignment statements, starter files, and grading rules remain at their original links. A study example is not automatically a drop-in homework submission. The actual assignment can restrict libraries, precision, file edits, or execution environment, and those constraints belong to its own contract.

The team project should choose a bounded operation with a meaningful baseline and enough data to expose a bottleneck. A good first milestone reproduces correctness on small inputs. The second milestone establishes a serial or simple parallel baseline. The third tests one optimization hypothesis. A final extension can explore scale or a second hardware configuration after the core result is stable. This sequence makes a useful result possible even when an ambitious final optimization does not work.

## A handoff preserves evidence and uncertainty

A reproducible repository records exact commands, dependency versions, input-generation seeds, source hashes, and raw outputs. A result table should point to the log or data file that supports it. A machine description should name the relevant processor or accelerator, process count, and software stack without exposing credentials or private infrastructure details. Results from a CPU model must remain labeled as models when a CUDA implementation has not run.

A refinement handoff should list completed sections, known mathematical or implementation limitations, untested platforms, and the next concrete checks. It should separate source-derived facts from original explanations and measured results. A reviewer can then improve a chapter without accidentally strengthening a claim. A green documentation build proves that the site renders under the tested toolchain. It does not prove that every algorithm is correct or every explanation is exhaustive.

The experiment below demonstrates an evidence record for a deterministic local result. It computes a blocked matrix product, verifies it against a reference, and records dimensions, data type, maximum error, and a content hash of the inputs. It intentionally contains no speedup field because it does not measure runtime. Adding a runtime later should require an actual benchmark record rather than an estimated number.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, hashlib
import numpy as np
rng=np.random.default_rng(5451)
A=rng.normal(size=(7,5));B=rng.normal(size=(5,6));C=np.zeros((7,6))
tile=3
for i in range(0,7,tile):
 for j in range(0,6,tile):
  for k in range(0,5,tile):
   C[i:i+tile,j:j+tile]+=A[i:i+tile,k:k+tile]@B[k:k+tile,j:j+tile]
err=float(np.max(np.abs(C-A@B)))
assert err<1e-12
print(json.dumps({'seed':5451,'A_shape':list(A.shape),'B_shape':list(B.shape),
 'dtype':str(A.dtype),'tile':tile,'max_abs_error':err,
 'input_sha256':hashlib.sha256(A.tobytes()+B.tobytes()).hexdigest(),
 'runtime_measured':False,'claim':'The blocked result matches the reference within1e-12.'},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "seed": 5451,
  "A_shape": [
    7,
    5
  ],
  "B_shape": [
    5,
    6
  ],
  "dtype": "float64",
  "tile": 3,
  "max_abs_error": 8.881784197001252e-16,
  "input_sha256": "5b80f8064e32ef337056be992e149bf709cc9da22481df492edc98a461d8d420",
  "runtime_measured": false,
  "claim": "The blocked result matches the reference within1e-12."
}
```

The 7-by 5 and 5-by 6 inputs force partial tiles, and the blocked result matches the reference within **1e-12**. The saved record describes exactly what was checked. Its `runtime_measured` field is false, so it supports a correctness claim and no performance claim.

:::{admonition} Try it
:class: exercise
You can change the tile width to 1, 2, 4,and 8. Verify the same contract across widths and explain why partial tiles are an important test.
:::

:::{admonition} Try it
:class: exercise
You can design a timing record with separate setup and compute intervals. State which interval answers your project question before collecting any numbers.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Contract | Inputs, outputs, precision, and boundary behavior define the actual problem. |
| Evidence | Correctness and performance need different checks and records. |
| Handoff | A useful refinement packet preserves both results and unresolved limits. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does a documentation build prove algorithmic correctness?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Rendering checks and algorithm checks operate at different layers."
  },
  {
    "q": "Why test dimensions that are not multiples of the tile width?",
    "options": [
      "They exercise partial boundary tiles.",
      "They guarantee maximum speedup."
    ],
    "answer": 0,
    "explain": "Boundary indexing often differs from the full-tile path."
  },
  {
    "q": "Can a CPU model establish CUDA throughput?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "The model does not execute GPU hardware."
  },
  {
    "q": "What should support a measured speedup?",
    "options": [
      "Saved runtimes with a specified baseline and setup.",
      "An expected ratio from the algorithm name."
    ],
    "answer": 0,
    "explain": "A measurement claim requires measurement evidence."
  },
  {
    "q": "What does the example's false runtime_measured field mean?",
    "options": [
      "The record supports correctness but no timing claim.",
      "The algorithm failed."
    ],
    "answer": 0,
    "explain": "The experiment deliberately validates arithmetic without benchmarking."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
