# 20. Interconnection networks and communication costs

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`19-pthreads-openmp` described coordination inside one address space. A distributed machine moves data across an interconnect. This lecture builds a cost model for that movement and explains why the communication pattern should influence the algorithm and its processor mapping.

:::{admonition} Learning goals
- You can compare rings, meshes, hypercubes, and tree networks.
- You can use a latency-bandwidth model with explicit units.
- You can explain contention, routing, and the difference between broadcast and personalized exchange.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/20-communication-networks.py`.

## A network is a graph with capacity constraints

A network node represents a processor, a router, or a switching element. A link carries data between adjacent nodes. A **static network** exposes a fixed direct-neighbor structure, while a switched network routes traffic through intermediate switching elements. Either organization has finite link bandwidth and finite buffers. An algorithm's logical communication graph must be embedded in that physical network.

A ring gives each node two neighbors and provides simple cyclic shifts. Its degree is small, but the distance between far-apart nodes grows with the number of nodes. A two-dimensional mesh connects interior nodes to four neighbors. A square mesh with $p$ nodes has side length about $\sqrt p$, so its largest route has order $\sqrt p$ hops. A torus adds wraparound links to shorten some routes. Boundary conditions in an application and wraparound links in a network are separate concepts.

A $d$-dimensional hypercube has $p=2^d$ nodes. Each node has one neighbor for each bit of its binary address, and the neighbor differs in that bit. The distance between two nodes is their Hamming distance, which is at most $d=\log_2p$. Hypercubes provide useful recursive communication patterns, although increasing degree complicates physical construction. A fat tree increases aggregate capacity toward the root to avoid the severe upper-level bottleneck of a thin tree.

The **bisection bandwidth** measures the capacity across a cut dividing the machine into two equally sized groups under the chosen definition of the cut. An all-to-all exchange can stress that cut because many messages must cross simultaneously. Small diameter alone does not guarantee high aggregate bandwidth. A star has short routes but can overload its center. Network quality therefore needs both distance and capacity measures.

## Startup and per-byte cost describe a first approximation

A common point-to-point model is $T(m)=\alpha+\beta m$, where $m$ is message size in bytes, $\alpha$ is startup time, and $\beta$ is time per byte. The reciprocal $1/\beta$ is the modeled bandwidth. Units prevent common errors. If $\alpha=2$ microseconds and bandwidth is 10 GB/s, a 1000-byte message costs approximately 2.1 microseconds in this simple model. Decimal GB/s means one billion bytes per second.

The model ignores queueing, software progress, protocol transitions, and contention. A route may add hop-dependent delay. **Store-and-forward** routing receives a whole packet before forwarding it. Cut-through and wormhole styles can pipeline portions through a path. Packetization means the cost of a long logical message can differ from multiplying a one-packet cost. These details motivate measurement, but the simple model remains useful for comparing message counts and volumes.

Batching ten small messages can save nine startups if the receiver can use one combined message. Batching also delays the first result and requires buffering. An algorithm that needs each reply before sending the next request cannot always batch freely. The dependency graph therefore limits the communication optimization just as it limits arithmetic parallelism.

## Collective patterns have different data semantics

A **broadcast** sends one root's data to every participant. A binary-tree broadcast reaches twice as many processes in each round, requiring $\log_2p$ rounds when $p$ is a power of two. A **reduction** reverses the pattern by combining inputs toward a root. An **all-reduce** leaves the combined value on every process. A reduction followed by a broadcast is a valid conceptual implementation, but actual libraries often choose better algorithms for particular sizes.

A **scatter** distributes distinct pieces from a root, and a **gather** collects distinct pieces at a root. An **all-gather** gives every participant every piece. An **all-to-all personalized exchange** sends a different destination-specific block from each process to every other process. This is fundamentally different from repeatedly broadcasting one shared block. A prefix operation, or scan, combines values in rank order and gives each process its prefix result.

A ring all-reduce can use a reduce-scatter phase followed by an all-gather phase. It uses more rounds than a tree but partitions a large vector so each link carries a useful stream of chunks. A tree can favor small messages where startup dominates. A ring can favor large messages where bandwidth dominates. The crossover depends on the network and implementation, so a model comparison is a hypothesis for benchmarking.

## Mapping can turn logical neighbors into distant machines

A stencil communicates mostly with neighboring spatial partitions. Placing those partitions on nearby network nodes can reduce link contention and route length. A poor mapping can route every local-looking exchange across the same central cut. **Embedding** asks how one graph can be represented on another. Dilation measures how long an embedded edge becomes, while congestion counts how many logical edges compete for a physical link under the mapping.

Topology-aware placement can help, but application balance still matters. A rank with twice the computation delays every synchronized phase regardless of excellent placement. Modern interconnects also use adaptive routing, so the route may not be a fixed function of rank numbers. The safest design starts with a clear communication pattern and a portable algorithm, then tests placement changes on the target machine with recorded conditions.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, math
p, alpha, bandwidth = 8, 2e-6, 10e9
rows=[]
for m in (1024, 1024*1024):
    beta=1/bandwidth
    tree=2*math.log2(p)*(alpha+beta*m)
    ring=2*(p-1)*alpha+2*(p-1)/p*beta*m
    rows.append({'bytes':m,'tree_allreduce_us':round(tree*1e6,3),
                 'ring_allreduce_us':round(ring*1e6,3)})
assert rows[0]['tree_allreduce_us'] < rows[0]['ring_allreduce_us']
assert rows[1]['ring_allreduce_us'] < rows[1]['tree_allreduce_us']
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "bytes": 1024,
    "tree_allreduce_us": 12.614,
    "ring_allreduce_us": 28.179
  },
  {
    "bytes": 1048576,
    "tree_allreduce_us": 641.146,
    "ring_allreduce_us": 211.501
  }
]
```

The model favors a tree for the 1, 024-byte message and a ring for the1 MiB message. This result follows from assumed startup and bandwidth parameters. It does not establish which algorithm an MPI library selects or which algorithm wins on your network. The point is to compare **round count and byte volume separately**.

:::{admonition} Try it
:class: exercise
You can increase startup to 20 microseconds while keeping bandwidth fixed. Predict which method becomes more attractive for medium messages.
:::

:::{admonition} Try it
:class: exercise
You can draw an eight-node ring and an eight-node hypercube. Find the route from node 0 to node 7 in each graph and explain why hop count does not fully determine runtime.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Topology | A network has both route-length and shared-capacity constraints. |
| Cost model | Startup penalizes many small messages, and bandwidth penalizes large byte volume. |
| Collective choice | Different message sizes can favor different communication schedules. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the degree of a three-dimensional hypercube?",
    "options": [
      "3.",
      "8."
    ],
    "answer": 0,
    "explain": "Each node has one neighbor per address bit."
  },
  {
    "q": "What are the units of beta in alpha plus beta times bytes?",
    "options": [
      "Seconds per byte.",
      "Bytes per second."
    ],
    "answer": 0,
    "explain": "Multiplying beta by bytes must produce seconds."
  },
  {
    "q": "Does all-to-all send the same data to every destination?",
    "options": [
      "Not in a personalized all-to-all.",
      "Always."
    ],
    "answer": 0,
    "explain": "Each destination can receive a distinct block from each sender."
  },
  {
    "q": "What does bisection bandwidth describe?",
    "options": [
      "Capacity across a balanced network cut.",
      "The number of instructions per core."
    ],
    "answer": 0,
    "explain": "Many simultaneous routes may compete for that cut."
  },
  {
    "q": "Does the model prove ring all-reduce always wins?",
    "options": [
      "No, the result depends on size and assumptions.",
      "Yes."
    ],
    "answer": 0,
    "explain": "The small-message example favors the tree."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
