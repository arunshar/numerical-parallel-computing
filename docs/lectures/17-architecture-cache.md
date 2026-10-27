# 17. Parallel architectures and memory locality

Prepared for Arun Sharma. Codex drafted this study edition.

Numerical methods describe the arithmetic that produces an answer. Parallel architecture determines how many operations can proceed together and how quickly their operands arrive. We begin the parallel track by separating instruction execution, data movement, and synchronization. This distinction explains why doubling the number of cores rarely doubles application speed.

:::{admonition} Learning goals
- You can distinguish SIMD, MIMD, shared memory, and distributed memory.
- You can calculate cache-line traffic for a simple access pattern.
- You can explain why locality and false sharing matter before adding threads.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/17-architecture-cache.py`.

## A processor has several sources of parallelism

**Instruction-level parallelism** overlaps independent instructions from one instruction stream. A pipeline divides instruction execution into stages. Several instructions occupy different stages at the same time, much as several packages occupy different stations in a factory. A deeper pipeline can improve throughput without reducing the latency of one dependent operation. A dependency such as `x = f(x)` still forces the next invocation to wait for the previous result.

A superscalar processor can issue multiple suitable instructions during one cycle. Out-of-order execution searches for ready instructions while older instructions wait for operands. The processor must preserve the observable behavior required by the instruction set. These mechanisms do not remove true data dependencies. Branch prediction also matters because the machine may discard speculative work after learning that a branch prediction was wrong.

**SIMD** applies one operation to several data lanes. A vector addition can add several pairs of floating-point numbers with one vector instruction. Vector width describes capacity per instruction, not the number of independent programs. Tail elements, misaligned access, reductions, and irregular branches can limit useful lane occupancy. AVX is an x86 vector-instruction family. An Apple Silicon machine uses a different instruction set, so an AVX example cannot simply be executed on every laptop.

**MIMD** processors execute different instruction streams on different data. Multiple CPU cores and separate cluster nodes fit this model. A machine can combine MIMD across cores and SIMD within each core. Flynn's taxonomy therefore gives useful categories without forcing a modern system into only one practical behavior. GPUs add their own execution and scheduling model, which later lectures examine directly.

## A memory hierarchy trades capacity for access cost

A register holds a small amount of data near an execution unit. Caches hold larger blocks, and main memory holds much more. Storage adds another level with different access properties. **Latency** measures the delay before an individual request is satisfied. **Bandwidth** measures the sustained volume transferred per unit time. A system can have high bandwidth and still have substantial latency. Many independent outstanding requests can hide latency when the application exposes enough useful work.

A cache transfers a **cache line**, not one isolated scalar. Suppose a model uses 64-byte lines and 8-byte numbers. One line contains eight adjacent numbers. A sequential traversal consumes those eight numbers after one line fill. A stride of eight numbers touches one useful value from each line. The arithmetic stays the same, but useful bytes per fetched line drop from 64 to 8. These are model parameters, not claims about the line size of the computer running this page.

Temporal locality means that a program reuses data soon. Spatial locality means that nearby addresses are accessed close together in time. Matrix loop order affects both. A row-major matrix stores a row contiguously. Traversing columns can create large address jumps. Blocking partitions a matrix into tiles that fit a chosen cache level so that each fetched tile supports many arithmetic operations. A tile must leave space for other live data and cannot be sized only from the advertised cache capacity.

## Shared addresses do not imply uniform access

A shared-memory program gives threads access to one address space. Hardware coherence keeps cached copies consistent according to the machine's coherence rules. **NUMA** systems attach memory to different processor domains. A remote NUMA access can have a different cost from a local access even though the pointer syntax is identical. First-touch allocation and thread placement can affect locality. A placement policy must be measured on the actual system because operating systems and runtimes can move work or pages.

A **false-sharing** pattern occurs when independent variables occupy the same coherence unit and different cores repeatedly write them. The variables are logically separate, but their line moves between ownership states. Padding can separate counters, although excessive padding wastes memory and cache capacity. A better algorithm may keep a private counter in a register and combine results once. This reduces both sharing traffic and the number of stores.

Distributed-memory systems give processes separate address spaces. A process cannot dereference another process's ordinary pointer. It must communicate data through a supported mechanism such as MPI. Shared-memory and distributed-memory models describe visibility and communication, while physical deployment describes where processes run. Multiple MPI ranks can run on one laptop, and a hybrid MPI-plus-OpenMP application can combine both models.

## A cost model guides the first optimization

A simple roofline estimate compares arithmetic work with transferred bytes. Arithmetic intensity is `operations / bytes`. If a kernel performs two operations while streaming three 8-byte arrays, its simplified intensity is $2/24=1/12$ operations per byte. A hypothetical bandwidth of 120 GB/s would then support at most 10 billion operations per second under that traffic model. Cache reuse, write allocation, fused operations, and data types change the actual accounting. The estimate is useful because it exposes which assumption must improve.

A parallel random-access machine, or **PRAM**, abstracts away these physical costs and lets processors access a shared memory through idealized rules. EREW, CREW, and CRCW variants differ in whether simultaneous reads and writes are allowed. PRAM algorithms reveal dependencies and asymptotic opportunities. They do not predict a real machine's memory traffic without a mapping step. We will keep algorithmic work, communication cost, and measured runtime as separate quantities throughout the course.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
N, LINE, ITEM = 64, 64, 8
rows = []
for stride in (1, 2, 4, 8):
    addresses = [i * stride * ITEM for i in range(N)]
    lines = {a // LINE for a in addresses}
    rows.append({'stride_elements': stride, 'useful_bytes': N * ITEM,
                 'line_bytes': len(lines) * LINE,
                 'utilization': N * ITEM / (len(lines) * LINE)})
assert [r['line_bytes'] for r in rows] == [512, 1024, 2048, 4096]
print(json.dumps(rows, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "stride_elements": 1,
    "useful_bytes": 512,
    "line_bytes": 512,
    "utilization": 1.0
  },
  {
    "stride_elements": 2,
    "useful_bytes": 512,
    "line_bytes": 1024,
    "utilization": 0.5
  },
  {
    "stride_elements": 4,
    "useful_bytes": 512,
    "line_bytes": 2048,
    "utilization": 0.25
  },
  {
    "stride_elements": 8,
    "useful_bytes": 512,
    "line_bytes": 4096,
    "utilization": 0.125
  }
]
```

The sequential model fetches **512 bytes**, while stride eight fetches **4, 096 bytes** for the same 512 useful bytes. This experiment counts compulsory line traffic in an idealized address trace. It does not measure a hardware cache, model prefetching, or predict an eightfold runtime change. Its purpose is to make the wasted transfer visible before you benchmark.

:::{admonition} Try it
:class: exercise
You can change LINE to 128 and predict the result for every stride before rerunning the trace. Explain why a larger transfer unit can help sequential access and hurt sparse access.
:::

:::{admonition} Try it
:class: exercise
You can model four thread counters placed 8 bytes apart, then 128 bytes apart. Count shared lines and explain why this count is only a warning sign, not a runtime measurement.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Instruction parallelism | The processor overlaps instructions that do not depend on each other. |
| Data locality | The program gets more useful work from each transferred cache line. |
| Memory model | Shared memory exposes addresses, while distributed memory requires explicit data exchange. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does SIMD require a separate program for each lane?",
    "options": [
      "No, one operation applies across lanes.",
      "Yes, each lane runs an unrelated program."
    ],
    "answer": 0,
    "explain": "SIMD shares an instruction across data lanes."
  },
  {
    "q": "What does bandwidth measure?",
    "options": [
      "Bytes transferred per unit time.",
      "The delay of one request."
    ],
    "answer": 0,
    "explain": "Latency measures individual delay, while bandwidth measures sustained transfer."
  },
  {
    "q": "How many bytes does the stride-eight model fetch?",
    "options": [
      "512 bytes.",
      "4, 096 bytes."
    ],
    "answer": 1,
    "explain": "Each of the 64 reads touches a distinct 64-byte line."
  },
  {
    "q": "Can independent variables cause coherence traffic?",
    "options": [
      "Yes, if they share a cache line.",
      "No, distinct variables never interact."
    ],
    "answer": 0,
    "explain": "False sharing concerns physical transfer units rather than logical variables."
  },
  {
    "q": "Does a PRAM model include realistic network contention?",
    "options": [
      "No, it is an abstraction.",
      "Yes, it models every packet."
    ],
    "answer": 0,
    "explain": "A mapping and cost model are needed for physical performance."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
