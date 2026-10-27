# 26. Graph algorithms, sparse matrices, and partitioning

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`25-sorting-merging` showed how irregular data can unbalance a parallel algorithm. Graphs make that irregularity central. This lecture connects graph traversal, shortest paths, sparse matrix multiplication, and graph partitioning through one common question about which information must cross a partition boundary.

:::{admonition} Learning goals
- You can distinguish BFS, Dijkstra, Prim, and Floyd-Warshall.
- You can represent a sparse matrix in CSR form and compute a matrix-vector product.
- You can explain graph Laplacians, edge cuts, and load-balance constraints.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/26-graphs-sparse-partitioning.py`.

## A representation determines the work you touch

A graph has vertices and edges, possibly with direction and weights. An adjacency matrix stores a value for every possible pair, while an adjacency list stores only present neighbors. A dense matrix makes constant-time edge lookup simple but uses order $n^2$ space. A sparse representation uses space proportional to vertices plus stored edges. The best representation depends on density and the operations the algorithm performs.

A sparse matrix often represents a graph or a discretized operator. **COO** stores row indices, column indices, and values for each nonzero. **CSR** groups entries by row and stores a row-pointer array, column indices, and values. Row $i$ occupies the interval from `rowptr[i]` to `rowptr[i+1]`. The last row pointer equals the stored nonzero count. Empty rows have identical consecutive pointers and must still produce a valid output value.

For sparse matrix-vector multiplication, each output is $y_i=\sum_j A_{ij}x_j$ over stored entries. Assigning rows to workers gives exclusive ownership of output entries. Equal row counts can be poorly balanced when row lengths differ. Balancing nonzeros improves arithmetic balance but may split rows and require reductions. The access to x is indirect, so regular storage of A does not guarantee regular memory access overall.

## Traversals have different correctness rules

**Breadth-first search** explores an unweighted graph in distance layers. A frontier contains vertices discovered at the current layer. Expanding the frontier finds unvisited neighbors for the next layer. Multiple parents can discover the same vertex, so a parallel implementation needs a safe claim operation or a deduplication phase. A race-free visited flag is necessary, but the algorithm must also preserve the intended distance level.

A top-down BFS visits outgoing edges from the current frontier. A bottom-up step scans unvisited vertices and asks whether any neighbor belongs to the frontier. Bottom-up work can be attractive when the frontier is large, but it can waste work when the frontier is small. Direction-optimizing traversal switches based on a model or measured condition. Directed graphs require care because incoming and outgoing adjacency serve different roles.

**Dijkstra's algorithm** selects the unsettled vertex with minimum tentative distance and relaxes its edges. Its classical correctness argument requires nonnegative edge weights. Parallelizing the neighbor relaxations is straightforward, but selecting the next global minimum and maintaining ordering can limit scalability. Bucketed approaches such as delta-stepping relax the strict global order while preserving a different correctness argument. Arbitrary out-of-order relaxation is not automatically equivalent to Dijkstra.

**Prim's algorithm** also chooses a minimum candidate, but its state records the cheapest edge connecting the current tree to an outside vertex. Dijkstra's state records distance from a source. Replacing one update formula with the other changes the problem. A minimum spanning tree minimizes total tree-edge weight, while a shortest-path tree minimizes each root-to-vertex path distance. A graph can have very different trees for these two objectives.

## All-pairs and connectivity algorithms expose bulk phases

Floyd-Warshall considers each vertex $k$ as an allowed intermediate and updates $D_{ij}=\min(D_{ij},D_{ik}+D_{kj})$. For a fixed $k$, many pairs $(i,j)$ can be updated in parallel when the required pivot row and column are available. The phase order over $k$ remains important. Distributed layouts communicate the pivot information and can block the computation for locality.

The algorithm supports negative edge weights when negative cycles are absent. A negative diagonal after closure signals a negative cycle. **Transitive closure** uses a related recurrence with Boolean reachability and logical operations. **Connected components** can use label propagation, hooking, or union-find-based methods, but parallel updates need a consistent way to merge component identities. A graph algorithm's arithmetic may be simple while its ordering and synchronization are the difficult part.

## A graph partition balances work and communication

Partitioning assigns vertices or rows to workers. An **edge cut** counts edges crossing partition boundaries, possibly with weights. A balanced partition constrains the amount of work or storage assigned to each worker. Minimizing cuts without balance is trivial if every vertex goes to one worker, so the two requirements must be stated together. In sparse matrix-vector multiplication, a cut can estimate which x values need remote access.

The graph Laplacian is $L=D-A$ for an undirected weighted graph, where D contains weighted vertex degrees. Its quadratic form is a sum of weighted squared differences across edges. A vector with slowly varying values across strongly connected regions can reveal a useful division. Spectral partitioning often uses an eigenvector associated with the second-smallest Laplacian eigenvalue. The eigenvector is a heuristic guide to a constrained partitioning problem, not a guarantee of the best balanced cut.

Multilevel partitioning coarsens the graph, partitions a smaller representation, then refines the division while expanding it. Domain decomposition applies similar ideas to physical meshes. A subdomain's interior can be computed locally, while halo values cross its boundary. A compact partition often reduces boundary traffic, but anisotropic work or irregular connectivity can change the best shape.

## Sparse correctness should be checked against a dense reference

The experiment constructs a four-vertex path Laplacian, stores it in CSR form, and multiplies it by a small vector. A dense reference validates the CSR indexing. The row sums are zero, so multiplying by a constant vector gives zero. These checks exercise different invariants. The first checks storage and arithmetic, while the second checks the Laplacian construction.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
A=np.array([[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]],dtype=float)
rowptr=[0];cols=[];vals=[]
for row in A:
 for j,v in enumerate(row):
  if v!=0:cols.append(j);vals.append(v)
 rowptr.append(len(vals))
x=np.array([1,2,4,8.]);y=np.zeros(4)
for i in range(4):
 for k in range(rowptr[i],rowptr[i+1]):y[i]+=vals[k]*x[cols[k]]
assert np.allclose(y,A@x) and np.allclose(A@np.ones(4),0)
print(json.dumps({'rowptr':rowptr,'columns':cols,'y':y.tolist(),
                  'laplacian_constant_residual':float(np.linalg.norm(A@np.ones(4)))},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "rowptr": [
    0,
    2,
    5,
    8,
    10
  ],
  "columns": [
    0,
    1,
    0,
    1,
    2,
    1,
    2,
    3,
    2,
    3
  ],
  "y": [
    -1.0,
    -1.0,
    -2.0,
    4.0
  ],
  "laplacian_constant_residual": 0.0
}
```

The product is **[-1,-1,-2, 4]**, and the constant-vector residual is zero in this small exact construction. The experiment checks a sparse representation and its graph meaning. It does not measure distributed communication or solve the graph partitioning optimization problem.

:::{admonition} Try it
:class: exercise
You can insert an empty row in a sparse matrix and verify that its CSR row pointers are equal. Check that the corresponding output is zero.
:::

:::{admonition} Try it
:class: exercise
You can partition the four-vertex path into {0, 1} and {2, 3}, then into {0, 2} and {1, 3}. Count cut edges and explain the communication implication.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Graph algorithm | The state and ordering rules distinguish shortest paths, spanning trees, and traversal. |
| Sparse storage | CSR exposes row ownership but can hide irregular row cost. |
| Partitioning | A useful partition balances load while limiting cross-partition traffic. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Can classical Dijkstra safely assume negative edge weights?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Its settled-distance argument requires nonnegative weights."
  },
  {
    "q": "Does Prim optimize source-to-vertex distance?",
    "options": [
      "No, it builds a minimum spanning tree.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Its candidate value is a connecting-edge cost."
  },
  {
    "q": "What does an empty CSR row look like?",
    "options": [
      "Two equal consecutive row pointers.",
      "A missing row pointer."
    ],
    "answer": 0,
    "explain": "Every row needs an interval, including an empty interval."
  },
  {
    "q": "What does a graph Laplacian do to a constant vector?",
    "options": [
      "It produces zero for this undirected construction.",
      "It always doubles the vector."
    ],
    "answer": 0,
    "explain": "Each row sums to zero."
  },
  {
    "q": "Why must a cut objective include balance?",
    "options": [
      "Otherwise all vertices can go to one worker.",
      "Balance has no effect."
    ],
    "answer": 0,
    "explain": "Communication alone admits a useless trivial placement."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
