# 41. Sparse matrices and graph traversal on GPUs

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`40-merge-sort`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will interpret COO, CSR, ELL, hybrid, and JDS storage choices.
- You will derive a sparse matrix-vector product from row offsets.
- You will compare push and pull graph traversal using frontiers. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/41_sparse_graphs.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## Sparse data saves storage but introduces irregularity

A **sparse matrix** stores only a small fraction of its entries explicitly. The usual motivation is to avoid spending memory and arithmetic on zeros. A dense $m\times n$ matrix stores $mn$ values. A sparse representation stores the nonzero values plus metadata that identifies their locations. The metadata and irregular access can dominate when each stored value supports only a small amount of arithmetic.

The sparse matrix-vector product computes $y_i=\sum_j A_{ij}x_j$. Each nonzero contributes one multiplication and one addition. A float32 value, a 32-bit column index, and a float32 vector read already suggest substantial traffic for two operations. Cache reuse can reduce vector traffic, but the index traffic remains part of the representation.

**COO**, or coordinate storage, keeps row indices, column indices, and values. It is convenient for construction and streaming individual nonzeros. Multiple nonzeros from one row can require a reduction or atomic updates to the output. Sorting or grouping coordinates can improve locality but costs preprocessing.

## Row formats trade balance against padding

**CSR**, or compressed sparse row storage, keeps a row-offset array, column indices, and values. Row $i$ occupies positions `row_ptr[i]` through `row_ptr[i+1]-1`. A one-thread-per-row kernel has simple ownership but can suffer when row lengths vary widely. A one-warp-per-row kernel distributes a long row across lanes, while short rows can leave many lanes idle.

**ELL** stores the same number of slots for every row, padding short rows to the longest stored width. Its regular layout can improve coalescing, but one exceptionally long row can waste much of the storage. A hybrid ELL-COO format keeps a regular prefix in ELL and sends the overflow to COO.

**JDS**, or jagged diagonal storage, sorts rows by length and groups their first nonzeros, second nonzeros, and later nonzeros into jagged diagonals. This layout reduces padding relative to a fixed rectangular ELL layout while grouping similar work. It requires a permutation to restore original row ordering. No format is best independently of the matrix’s row-length distribution and the operation being performed.

The preprocessing cost belongs in the application model. A format conversion can pay off when the same matrix is multiplied by many vectors. It can be wasteful when the sparsity pattern changes every iteration or the matrix is used only once.

## A graph turns matrix structure into neighbors

A graph’s adjacency matrix has a nonzero wherever an edge exists. CSR therefore also stores outgoing adjacency lists. A **breadth-first search**, or BFS, assigns each reachable vertex its shortest unweighted edge distance from a source. The algorithm advances one level at a time through a **frontier** of newly reached vertices.

A push traversal visits outgoing edges of the current frontier. It tries to claim each neighbor that has not yet been visited. Several frontier vertices can discover the same neighbor, so the visited-state update needs a race-safe mechanism. A vertex can be enqueued once through an atomic claim, or duplicates can be generated and removed later. The choice changes contention and memory traffic.

A pull traversal examines unvisited vertices and asks whether any predecessor lies in the current frontier. It can stop as soon as it finds one. Pull is attractive when the frontier is large, while push is attractive when the frontier is small. Directed graphs may need both outgoing and incoming adjacency structures to support both directions efficiently.

## Work assignment must follow edge volume

A one-thread-per-vertex mapping is simple but can be badly imbalanced. A hub vertex may have thousands of edges while most vertices have only a few. Edge-centric work distributes edges more evenly but needs a way to determine each edge’s source and its frontier membership. A hybrid design assigns short adjacency lists to threads and long lists to warps or blocks.

A frontier can be stored as a compact list or as a bitmap. A list makes iteration over a small frontier efficient. A bitmap makes membership tests cheap and can suit pull traversal. Converting representations adds work, so a direction-switching policy should include the conversion cost.

Privatized output queues can reduce pressure on one global queue counter. A block gathers discoveries locally and reserves one larger global range. It then copies the local discoveries into that range. This technique connects graph traversal to the histogram and scan lessons, but it still requires correct visited-state handling.

## Sparse correctness tests need structural cases

A sparse matrix test should include empty rows, one-entry rows, duplicate coordinates under a stated combine rule, and highly uneven row lengths. A graph test should include disconnected vertices, self-loops, duplicate edges, and multiple paths to the same vertex. A random graph alone rarely exercises every important edge case.

The experiment compares CSR multiplication against a dense reference and performs BFS on a small graph. It also calculates ELL padding from row lengths. These checks establish mathematical agreement for the toy data and expose the storage tradeoff. They do not benchmark a GPU graph library or a sparse kernel.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`41_sparse_graphs.py <../../labs/gpu/41_sparse_graphs.py>`.

```python
import numpy as np
row_ptr = np.array([0,2,2,5,6])
cols = np.array([0,3,0,1,3,2])
vals = np.array([2.,1.,-1.,3.,2.,4.])
x = np.array([1.,2.,3.,4.])
y = np.zeros(4); dense = np.zeros((4,4))
for row in range(4):
    for k in range(row_ptr[row],row_ptr[row+1]):
        y[row] += vals[k]*x[cols[k]]
        dense[row,cols[k]] += vals[k]
assert np.allclose(y,dense@x)
lengths = np.diff(row_ptr)
print("csr_product=", y.tolist())
print(f"nonzeros={len(vals)}, ell_slots={len(lengths)*max(lengths)}, padding={len(lengths)*max(lengths)-len(vals)}")
graph = [[1,2],[3],[3,4],[5],[5],[],[]]
distance = [-1]*len(graph); distance[0]=0; frontier=[0]
while frontier:
    following=[]
    for vertex in frontier:
        for neighbor in graph[vertex]:
            if distance[neighbor]<0:
                distance[neighbor]=distance[vertex]+1;following.append(neighbor)
    frontier=following
assert distance == [0,1,1,2,2,3,-1]
print("bfs_distances=",distance)
```

The recorded run produced the following output.

```text
csr_product= [6.0, 0.0, 13.0, 12.0]
nonzeros=6, ell_slots=12, padding=6
bfs_distances= [0, 1, 1, 2, 2, 3, -1]
```

**CSR produces the same vector as the dense reference while storing six nonzeros.** The modeled ELL layout allocates 12 slots and wastes six. BFS leaves the disconnected final vertex at distance minus one.

:::{admonition} Try it 1 :class: exercise

Add one row with 20 entries to a larger toy matrix. Compare ELL padding with a hybrid format that stores only four entries per row in ELL. :::

:::{admonition} Try it 2 :class: exercise

Construct a graph with one high-degree hub and many degree-one vertices. Estimate how a vertex-per-thread mapping and an edge-based mapping distribute work. :::

## Summary

| Decision | What you should remember |
|---|---|
| Sparse format | Metadata, padding, and row balance determine the useful representation. |
| Frontier | BFS processes newly discovered vertices one level at a time. |
| Traversal direction | Push and pull favor different frontier sizes and access patterns. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What does CSR row_ptr[i+1]-row_ptr[i] give?",
    "options": [
      "It gives the number of stored entries in row i.",
      "It gives the total number of columns."
    ],
    "answer": 0,
    "explain": "The offsets delimit the row\u2019s nonzero interval."
  },
  {
    "q": "How many padding slots does the toy ELL format use?",
    "options": [
      "It uses six padding slots.",
      "It uses zero padding slots."
    ],
    "answer": 0,
    "explain": "Four rows times maximum length three gives 12 slots for six values."
  },
  {
    "q": "Why can push BFS require atomic claims?",
    "options": [
      "Several parents can discover the same neighbor.",
      "Every graph has negative weights."
    ],
    "answer": 0,
    "explain": "The visited-state transition must handle competing discoveries."
  },
  {
    "q": "What does distance minus one mean in the reference?",
    "options": [
      "The vertex is unreachable from the source.",
      "The vertex has a negative-length path."
    ],
    "answer": 0,
    "explain": "The unweighted BFS leaves undiscovered vertices marked minus one."
  },
  {
    "q": "When can preprocessing a sparse format pay off?",
    "options": [
      "It can pay off when the converted matrix is reused many times.",
      "It always pays off even for one tiny operation."
    ],
    "answer": 0,
    "explain": "Repeated use can amortize conversion cost."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 14, 15. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.
