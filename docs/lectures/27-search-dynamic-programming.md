# 27. Parallel search and dynamic programming

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`26-graphs-sparse-partitioning` used graph structure that is present before execution. Search often discovers its graph while running, and dynamic programming compresses repeated subproblems into a dependency structure. This lecture explains the different sources of parallelism and why search speedup can be difficult to interpret.

:::{admonition} Learning goals
- You can explain branch-and-bound pruning and parallel search overhead.
- You can identify wavefront parallelism in a dynamic program.
- You can distinguish monadic and polyadic dependencies in textbook terminology.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/27-search-dynamic-programming.py`.

## Search discovers work as it runs

A search state describes a partial candidate solution and the choices still available. A search tree connects a state to its extensions. Depth-first search uses a stack and explores one branch deeply before returning. Breadth-first search expands a level at a time, which exposes many tasks but can consume large memory. Best-first search prioritizes states using a score, often a lower bound or estimated remaining cost.

In a discrete optimization problem, the goal is to find the best feasible solution. **Branch and bound** maintains an incumbent feasible solution and a bound on the best possible result below each partial state. For minimization, a state whose lower bound is no better than the incumbent can be pruned. The bound must be valid. An optimistic-looking estimate that sometimes exceeds the true best descendant can remove the optimal solution and invalidate the algorithm.

Parallel workers can explore separate subtrees. A good incumbent found by one worker helps others prune more work, so sharing incumbents can improve total work. Delayed sharing can make workers explore branches that would soon become unnecessary. Extremely frequent sharing adds synchronization and communication. The correct update protocol must also protect the incumbent's objective and corresponding solution from becoming inconsistent.

## Search overhead can be positive or negative

Parallel search may expand a different set of states from serial search because it visits branches in a different order. A **search overhead factor** compares parallel expanded work with the work of a specified sequential search. A factor above one means more states were explored. A factor below one can occur when a parallel worker finds a strong incumbent early and enables pruning that the sequential order discovered much later.

This effect can produce apparent superlinear speedup without violating a fixed-work bound, because the work was not fixed. A fair report records both runtime and expanded states. It also specifies the tie-breaking rule, random seed, incumbent-sharing policy, and stopping condition. Comparing a carefully tuned parallel search with a deliberately poor serial search order would exaggerate the value of parallel execution.

Parallel depth-first search often uses work stealing. An idle worker requests a portion of another worker's pending search stack or a subtree root. Coarse transfers reduce communication overhead, while very coarse transfers can leave the final few workers imbalanced. Termination detection is harder than checking that one local queue is empty. Work can still be executing or traveling in messages. A correct distributed protocol must account for all possible sources of future work.

Parallel best-first search can use a shared priority queue or several local queues. A shared queue preserves a stronger global ordering but can become a contention bottleneck. Local queues relax that ordering and improve throughput at the cost of extra exploration. Algorithms such as A* also need their heuristic assumptions stated precisely. Admissibility and consistency have different roles, and a parallel implementation must preserve the correctness conditions of its chosen variant.

## Dynamic programming stores repeated subproblems

A **dynamic program** defines states, a recurrence, base cases, and an evaluation order. Memoization evaluates states on demand and caches results. Tabulation evaluates a planned order. Parallelism comes from states whose prerequisites are already complete. The fact that a table has many entries does not mean all entries can be filled simultaneously.

Edit distance gives a clear example. State $D_{ij}$ is the minimum cost of transforming the first $i$ symbols of one string into the first $j$ symbols of another. Its recurrence takes the minimum of deletion, insertion, and substitution candidates from $D_{i-1,j}$,$D_{i,j-1}$,and $D_{i-1,j-1}$. The first row and column encode edits involving an empty prefix. States on the same anti-diagonal have no direct dependency on one another, so an anti-diagonal forms a **wavefront**.

For an $m$-by $n$ table, the usual recurrence performs order $mn$ work and has order $m+n$ wavefront steps when cells have constant cost. A real implementation groups cells into tiles to amortize scheduling and communication. Tile dependencies follow the same north, west, and northwest pattern. The wavefront is short near the corners and widest near the middle, so available parallelism changes during execution.

## Textbook dependency classes describe different recurrences

In the textbook's terminology, a monadic formulation builds a candidate from one predecessor state, while a polyadic formulation combines multiple subproblem values in one candidate. Serial formulations use a staged dependency structure, while nonserial formulations admit dependencies that do not fit a simple sequence of fixed stages. These labels describe recurrence structure. They do not mean that a serially classified formulation has no parallel implementation.

Matrix-chain parenthesization is a useful polyadic example. To compute the best cost for a matrix interval, each candidate split adds the best cost of its left interval, the best cost of its right interval, and the scalar multiplication cost of combining them. All shorter intervals must be ready. Intervals of the same length can be computed together, while candidate splits within an interval can also be reduced in parallel. These are two different levels of parallelism.

A dependency graph can be denser than the original input representation. Recomputing values may reduce communication at the cost of extra arithmetic, while storing the entire table may exceed memory capacity. Checkpointing, banding, and divide-and-conquer reconstruction address different tradeoffs. A banded edit-distance algorithm assumes the relevant path remains within its band, so it must not be presented as an exact unrestricted solver without a valid bound.

## The wavefront experiment checks the recurrence

The experiment computes edit distance between two short strings by anti-diagonal order and verifies a known result. Every cell is updated only after its prerequisites. The implementation runs sequential Python loops to expose the schedule, so it does not measure parallel execution. Its purpose is to provide a correct reference that can later support a tiled OpenMP or MPI version.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
s,t='kitten','sitting'
m,n=len(s),len(t)
d=[[0]*(n+1) for _ in range(m+1)]
for i in range(m+1):d[i][0]=i
for j in range(n+1):d[0][j]=j
widths=[]
for diagonal in range(2,m+n+1):
 cells=[]
 for i in range(1,m+1):
  j=diagonal-i
  if 1<=j<=n:
   d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(s[i-1]!=t[j-1]))
   cells.append((i,j))
 widths.append(len(cells))
assert d[m][n]==3
print(json.dumps({'edit_distance':d[m][n],'wavefront_widths':widths,
                  'cells_updated':sum(widths)},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "edit_distance": 3,
  "wavefront_widths": [
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    5,
    4,
    3,
    2,
    1
  ],
  "cells_updated": 42
}
```

The computed edit distance is **3**, and the program updates 42 non-boundary cells. The wavefront widths grow and then shrink, showing why a fixed processor team cannot stay equally busy throughout this small problem. The result validates the recurrence and schedule, not a parallel speedup.

:::{admonition} Try it
:class: exercise
You can change one input to the empty string and verify the base cases. Explain why no interior wavefront is needed.
:::

:::{admonition} Try it
:class: exercise
You can sketch a branch-and-bound search with two branches where the second branch contains a strong incumbent. Compare work when serial and parallel searches discover it at different times.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Search | Execution order can change the amount of explored work. |
| Pruning | A valid bound is a correctness requirement. |
| Dynamic programming | The recurrence determines which states can run together. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Can an invalid lower bound prune the optimum?",
    "options": [
      "Yes.",
      "No."
    ],
    "answer": 0,
    "explain": "A bound that overestimates the best descendant can remove a necessary branch."
  },
  {
    "q": "Does an empty local queue prove global termination?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Other workers or messages can still create work."
  },
  {
    "q": "Which edit-distance cells can share a wavefront?",
    "options": [
      "Cells on one anti-diagonal.",
      "All cells in the table."
    ],
    "answer": 0,
    "explain": "Their required predecessors lie on earlier anti-diagonals."
  },
  {
    "q": "What is the computed distance from kitten to sitting?",
    "options": [
      "3.",
      "1."
    ],
    "answer": 0,
    "explain": "Three unit-cost edits suffice and are necessary."
  },
  {
    "q": "Can parallel search execute less work than a serial search?",
    "options": [
      "Yes, a different discovery order can improve pruning.",
      "No, the explored tree is always fixed."
    ],
    "answer": 0,
    "explain": "An earlier incumbent can eliminate additional subtrees."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
