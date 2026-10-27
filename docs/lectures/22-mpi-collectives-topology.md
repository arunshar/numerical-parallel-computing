# 22. MPI collectives, groups, and virtual topologies

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`21-mpi-point-to-point` explained how individual messages match and complete. Many algorithms need a group-wide operation instead of a hand-written collection of sends. This lecture turns those operations into clear data-layout contracts and explains how communicators describe smaller teams and logical neighborhoods.

:::{admonition} Learning goals
- You can choose between broadcast, scatter, gather, reduce, all-reduce, and scan.
- You can explain collective ordering and communicator participation.
- You can map a two-dimensional rank grid to a matrix algorithm.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/22-mpi-collectives-topology.py`.

## A collective has a data contract

A broadcast gives every rank a copy of the root's input. The root argument identifies a rank within the participating communicator. A scatter distributes separate pieces from the root, and a gather collects separate pieces at the root. An all-gather collects everyone's contribution at every rank. A reduction combines corresponding elements using an operation, while an all-reduce also distributes the combined result to every rank.

These operations differ in who owns valid input and output buffers. In a root-based reduction, the final output is meaningful at the root. A non-root process should not read an arbitrary receive buffer and assume it contains the reduced result. An all-reduce provides that result to every participant. A scan gives rank $r$ the combination of ranks 0 through $r$, while an exclusive scan omits rank $r$ itself. The exclusive scan's rank-zero result needs special handling according to the API contract.

Suppose four ranks hold sums 3, 7, 11, 15 from disjoint blocks. A sum reduction produces 36 at one root. An all-reduce produces 36 everywhere. An inclusive scan produces 3, 10, 21, 36. The same inputs therefore support different useful outputs. A distributed offset calculation often needs an exclusive scan, while a convergence test often needs an all-reduce of a maximum residual or a logical condition.

## Every participant must follow a compatible order

Collectives coordinate processes in a communicator. If one rank calls a broadcast while another calls a reduction at the corresponding position, the program is invalid even if both calls involve the same data size. Control flow must keep the collective sequence compatible across participants. A condition based only on a local value can break that agreement. A preceding collective can establish a shared decision when the algorithm requires one.

A collective is not automatically a barrier. Some calls can return on one rank while another rank is still completing its part. Programs must rely on the specified data and completion guarantees rather than adding an imaginary global synchronization. `MPI_Barrier` provides an explicit participation boundary when that boundary is required. Extra barriers can hide load imbalance or distort measurements, so they should serve a stated purpose.

Nonblocking collectives return a request and follow similar buffer-lifetime rules to nonblocking point-to-point operations. They can support overlap with independent work. Their initiation order still matters. Persistent collectives separate repeated setup from activation for a stable operation signature. These advanced forms should follow a correct blocking implementation, because debugging a changing buffer layout and asynchronous completion at the same time is difficult.

## Irregular partitions need explicit counts and offsets

Uniform scatter and gather assume regular contributions. Their variable-count variants support uneven partitions through counts and displacements. For ten elements on four ranks, a balanced contiguous partition can use counts 3, 3, 2, 2 and displacements 0, 3, 6, 8. A displacement is measured according to the relevant datatype's extent, so it is not universally a byte offset. Every count and boundary should be checked against the allocated buffer.

An all-to-all exchange often appears when changing data layout. Each rank first groups items by destination, exchanges destination counts, computes offsets, and then moves the variable-sized payloads. A missing count exchange can leave receivers with incorrect capacity. The operation may also require substantial temporary storage. An algorithm that reduces arithmetic but doubles peak communication buffers can fail because of memory capacity before it reaches a timing bottleneck.

## Communicators define independent teams

`MPI_Comm_split` partitions an existing communicator by a color and orders the members by a key. Ranks with the same color form a new communicator. A key controls their relative order but does not determine their physical placement. A process can participate in several communicators and have a different rank in each. Code must therefore keep communicator-specific rank variables separate.

Groups describe ordered process collections, while communicators add communication context. Duplicating a communicator can isolate a library's message traffic from its caller. Freeing a communicator releases that MPI object when the application no longer needs it. These operations describe logical membership. They do not create a new operating-system process or move a process to another node.

A Cartesian communicator attaches dimensions, periodicity, and coordinates to a process group. A 2-by 2 grid contains four ranks and gives each rank a row and column coordinate. `MPI_Cart_shift` identifies neighbors along one dimension, with `MPI_PROC_NULL` representing a missing neighbor at a nonperiodic boundary. The null-process behavior simplifies boundary exchange because the same call structure can handle interior and edge ranks.

## A virtual grid supports matrix communication

Dense matrix multiplication often uses a two-dimensional process grid. Ranks in one row share a row communicator, and ranks in one column share a column communicator. A panel of A can be broadcast along its process row while a panel of B is broadcast down its process column. Each rank multiplies its received panels into its local C block. This is a central idea behind SUMMA-style algorithms.

The virtual topology gives a clear naming system and may help an implementation optimize mapping when reordering is permitted. It does not guarantee that adjacent Cartesian ranks occupy adjacent physical switches. A portable program should obtain coordinates and neighbors through MPI rather than assuming a fixed arithmetic rank formula after allowing reordering. The native example tests a fixed 2-by 2 grid without reordering to keep the first experiment transparent.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
counts=[3,3,2,2]
offsets=[sum(counts[:i]) for i in range(4)]
values=list(range(10))
chunks=[values[d:d+c] for d,c in zip(offsets,counts)]
assert sum(chunks,[])==values
local=[3,7,11,15]
prefix=[sum(local[:i+1]) for i in range(4)]
assert prefix==[3,10,21,36]
print(json.dumps({'counts':counts,'displacements':offsets,'chunks':chunks,
                  'inclusive_scan':prefix,'allreduce_sum':sum(local)},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "counts": [
    3,
    3,
    2,
    2
  ],
  "displacements": [
    0,
    3,
    6,
    8
  ],
  "chunks": [
    [
      0,
      1,
      2
    ],
    [
      3,
      4,
      5
    ],
    [
      6,
      7
    ],
    [
      8,
      9
    ]
  ],
  "inclusive_scan": [
    3,
    10,
    21,
    36
  ],
  "allreduce_sum": 36
}
```

The model partitions ten elements without overlap or omission and produces the expected inclusive scan. The native `mpi_collectives.c` adds a real four-rank scatter, reduction, all-reduce, and Cartesian-neighbor check. Its integer results are deterministic. The MPI implementation remains free to choose its internal communication algorithm.

:::{admonition} Try it
:class: exercise
You can distribute eleven elements across four ranks and compute balanced counts and displacements. Verify that the reconstructed sequence is exactly the input.
:::

:::{admonition} Try it
:class: exercise
You can change a Cartesian dimension from periodic to nonperiodic. Predict which boundary neighbors become MPI_PROC_NULL before running a native variation.
:::

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The MPI collectives example uses the native API

```{literalinclude} ../../labs/parallel/mpi_collectives.c
:language: c
```

## Summary

| Idea | What the lesson established |
|---|---|
| Collective semantics | Each operation specifies a different input-to-output distribution. |
| Participation | Every communicator member must follow a compatible collective sequence. |
| Virtual topology | Coordinates simplify neighbor logic without promising physical placement. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Which operation places a reduced result on every rank?",
    "options": [
      "MPI_Allreduce.",
      "MPI_Reduce alone."
    ],
    "answer": 0,
    "explain": "Reduce produces the result at its root."
  },
  {
    "q": "Is every collective a barrier?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Programs must use each operation's specified completion semantics."
  },
  {
    "q": "What are balanced counts for ten items on four ranks?",
    "options": [
      "3, 3, 2, 2.",
      "4, 4, 4, 4."
    ],
    "answer": 0,
    "explain": "The counts sum to ten and differ by at most one."
  },
  {
    "q": "Does MPI_Comm_split create new operating-system processes?",
    "options": [
      "No, it creates a communicator.",
      "Yes."
    ],
    "answer": 0,
    "explain": "It reorganizes existing participants logically."
  },
  {
    "q": "What can represent a missing Cartesian neighbor?",
    "options": [
      "MPI_PROC_NULL.",
      "An arbitrary rank outside the communicator."
    ],
    "answer": 0,
    "explain": "MPI defines null-process communication behavior for this purpose."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.
