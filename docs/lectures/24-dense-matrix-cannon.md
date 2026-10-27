# 24. Dense matrix algorithms and two-dimensional layouts

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`23-performance-models` explained why communication and work must be counted separately. Dense matrix multiplication offers enough arithmetic reuse to make that distinction concrete. This lecture derives blocked multiplication, Cannon’s algorithm, and panel-based alternatives, then connects the same layout ideas to solving linear systems.

:::{admonition} Learning goals
- You can explain matrix-vector and matrix-matrix communication patterns.
- You can trace Cannon’s initial skew and cyclic shifts.
- You can distinguish block, cyclic, and block-cyclic matrix distributions.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/24-dense-matrix-cannon.py`.

## Matrix multiplication exposes reusable operands

For $C=AB$, entry $C_{ij}$ is $\sum_k A_{ik}B_{kj}$. A straightforward square multiplication performs about $2n^3$ floating-point operations under the conventional multiply-plus-add count. Its inputs and output contain order $n^2$ values. The ratio suggests substantial reuse, but a poor loop order can fetch the same values repeatedly from a slower memory level. Blocking makes the intended reuse explicit.

The BLAS separates common operations into levels. Level 1 operates mainly on vectors, level 2 includes matrix-vector operations, and level 3 includes matrix-matrix operations. A dense matrix-vector product performs order $n^2$ work while reading order $n^2$ matrix data, so its arithmetic intensity is limited. A matrix-matrix product can perform order $b^3$ work from tiles containing order $b^2$ data. This difference explains why high-performance solvers reorganize work around level 3 kernels where possible.

A one-dimensional row distribution assigns each rank a group of output rows. For matrix-vector multiplication, each rank needs its rows of A and the relevant entries of x. Replicating x can be reasonable when x is small relative to A. For matrix multiplication, a row-distributed rank may need all of B, which becomes expensive as matrices grow. A two-dimensional process grid reduces the amount each rank must store and exchange.

## Cannon's algorithm aligns the inner index

Assume a square matrix is divided into $q$-by $q$ blocks and $p=q^2$ ranks form a periodic $q$-by $q$ process grid. Rank $(i,j)$ initially owns $A_{ij}$ and $B_{ij}$. Cannon’s initial alignment shifts row $i$ of A left by $i$ block positions and column $j$ of B up by $j$ positions. Rank $(i,j)$ then holds an A block and B block with the same inner block index $k=(i+j)\bmod q$.

Each rank multiplies those local blocks into its C block. The algorithm then shifts A left by one and B up by one. Repeating $q$ stages visits every inner block index once. The initial skew is therefore a correctness step, not a performance decoration. Without it, some ranks would multiply block pairs whose inner dimensions refer to different global regions.

For $q=2$, rank $(0,1)$ first computes $A_{01}B_{11}$ and then $A_{00}B_{01}$. Their sum is exactly $C_{01}$. Rank $(1,0)$ starts with $A_{11}B_{10}$ and then computes $A_{10}B_{00}$. A small trace makes the wraparound indexing much easier to trust than a large run with plausible-looking numbers.

The simplified per-rank work is order $n^3/p$. Each stage exchanges two blocks with $n^2/p$ elements, and there are $\sqrt p$ stages. The per-rank communication volume is therefore order $n^2/\sqrt p$, with order $\sqrt p$ startup events. This analysis assumes regular blocks and a suitable network mapping. The setup skew and final data distribution also need accounting in a complete timed region.

## SUMMA and block-cyclic distributions offer flexibility

A SUMMA-style multiplication iterates over panels. A rank receives an A panel broadcast along its process row and a B panel broadcast along its process column, then updates its local C block. This pattern supports rectangular process grids and more flexible panel sizes. The tradeoff includes collective startup costs, panel buffering, and the implementation's ability to overlap communication with computation.

A **block distribution** assigns a contiguous region to each rank. A **cyclic distribution** spreads individual rows or columns round-robin. A **block-cyclic distribution** spreads tiles round-robin over a process grid. If a tile has width $b$, the owner of global tile coordinate $(I,J)$ in a $p_r$-by $p_c$ grid is $(I\bmod p_r,J\bmod p_c)$. The in-rank tile indices are integer quotients. This separation between global coordinates, owner coordinates, and local indices prevents many indexing bugs.

Block-cyclic layouts help algorithms whose active work shrinks during execution. A pure block layout can leave early ranks idle as an elimination front moves. Spreading tiles shares later work more evenly. The layout also creates more communication relationships and complicates local indexing. Tile size affects cache reuse, load balance, communication count, and the amount of parallelism exposed to local kernels.

## Linear-system solvers add ordering and stability

Gaussian elimination factors a matrix into triangular factors with pivoting when needed. A distributed LU factorization performs panel work, communicates pivot decisions, and updates the trailing matrix. The trailing update maps well to matrix multiplication. The panel can form a serial or communication-sensitive bottleneck. Parallelizing the update alone therefore does not imply that the entire factorization scales equally well.

Cholesky factorization applies to symmetric positive definite matrices and avoids general pivoting under that mathematical assumption. QR factorization uses orthogonal transformations and is often attractive for numerical stability, but its communication pattern and arithmetic cost differ from LU. A parallel method must preserve the mathematical assumptions of its sequential counterpart. A fast factorization that silently drops required pivoting is a different algorithm with different failure cases.

Tridiagonal systems illustrate another choice. The Thomas algorithm performs linear work but has a sequential recurrence. Cyclic reduction removes alternating unknowns, creating logarithmic-depth stages with additional work and storage. The best method depends on whether one long system or many independent short systems are available. Parallelism across independent systems can be simpler and cheaper than parallelism within each system.

The experiment below simulates Cannon's block ownership with NumPy arrays. It checks every output against a serial matrix multiplication. The simulation validates the indexing and stage schedule. It does not run distributed matrix multiplication or measure a communication speedup.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
A=np.arange(16,dtype=np.int64).reshape(4,4)
B=(np.arange(16,dtype=np.int64).reshape(4,4)+1)
q,b=2,2
C=np.zeros((4,4),dtype=np.int64)
trace=[]
for i in range(q):
 for j in range(q):
  block=np.zeros((b,b),dtype=np.int64)
  ks=[]
  for stage in range(q):
   k=(i+j+stage)%q
   block += A[i*b:(i+1)*b,k*b:(k+1)*b] @ B[k*b:(k+1)*b,j*b:(j+1)*b]
   ks.append(k)
  C[i*b:(i+1)*b,j*b:(j+1)*b]=block
  trace.append({'rank':[i,j],'inner_blocks':ks})
assert np.array_equal(C,A@B)
print(json.dumps({'trace':trace,'C':C.tolist(),'matches_serial':True},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "trace": [
    {
      "rank": [
        0,
        0
      ],
      "inner_blocks": [
        0,
        1
      ]
    },
    {
      "rank": [
        0,
        1
      ],
      "inner_blocks": [
        1,
        0
      ]
    },
    {
      "rank": [
        1,
        0
      ],
      "inner_blocks": [
        1,
        0
      ]
    },
    {
      "rank": [
        1,
        1
      ],
      "inner_blocks": [
        0,
        1
      ]
    }
  ],
  "C": [
    [
      62,
      68,
      74,
      80
    ],
    [
      174,
      196,
      218,
      240
    ],
    [
      286,
      324,
      362,
      400
    ],
    [
      398,
      452,
      506,
      560
    ]
  ],
  "matches_serial": true
}
```

Every rank visits both inner blocks exactly once, and the simulated result matches `A @ B` exactly for the integer example. Rank(0, 1) visits inner indices **1 then 0**, which exposes the purpose of the initial skew. Floating-point variants should use a stated tolerance because the accumulation order can differ.

:::{admonition} Try it
:class: exercise
You can use a 6-by 6 matrix with a 3-by 3 process grid and 2-by 2 blocks. Trace rank(2, 1) and verify that it visits three distinct inner indices.
:::

:::{admonition} Try it
:class: exercise
You can distribute a 7-by 5 matrix in 2-by 2 tiles over a 2-by 3 process grid. List each tile owner and identify the partial tiles at the boundaries.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Blocking | Tiles make operand reuse explicit. |
| Cannon alignment | The initial skew matches the inner block index before multiplication. |
| Solver design | A scalable update kernel does not remove panel dependencies or stability requirements. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "How many ranks does a q-by-q Cannon grid use?",
    "options": [
      "q squared.",
      "q."
    ],
    "answer": 0,
    "explain": "Each process-grid coordinate owns a block pair."
  },
  {
    "q": "Why is the initial skew needed?",
    "options": [
      "It aligns the A and B inner block indices.",
      "It changes the mathematical product."
    ],
    "answer": 0,
    "explain": "Each local multiplication must use matching inner indices."
  },
  {
    "q": "What does a block-cyclic owner formula use?",
    "options": [
      "Tile coordinates modulo grid dimensions.",
      "Only the matrix value."
    ],
    "answer": 0,
    "explain": "Ownership is determined by position and tile size."
  },
  {
    "q": "Which BLAS level contains matrix-matrix multiplication?",
    "options": [
      "Level 3.",
      "Level 1."
    ],
    "answer": 0,
    "explain": "Level 3 kernels offer substantial data reuse."
  },
  {
    "q": "Does the NumPy simulation establish MPI performance?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "It checks indexing and arithmetic rather than distributed runtime."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
