# 19. Threads, synchronization, and OpenMP

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`18-decomposition-task-graphs` separated useful parallelism from scheduling choices. Shared-memory threads turn that plan into an executable program. This lecture explains what threads share, how a race appears, and why a reduction is usually better than a lock around every arithmetic operation.

:::{admonition} Learning goals
- You can identify shared and private state in a threaded loop.
- You can explain mutexes, condition variables, barriers, and atomics.
- You can compile a small pthread and OpenMP reduction and check its answer.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/19-pthreads-openmp.py`.

## A thread shares an address space with its peers

A process owns an address space and operating-system resources. Its threads have separate execution state and stacks, but they can access shared heap objects and global variables. Passing a pointer to a worker is therefore enough to expose an array. The pointer must remain valid until the worker finishes. A common bug passes the address of a loop variable that changes before the new thread reads it. A stable argument structure per worker removes that ambiguity.

POSIX threads expose creation, joining, mutual exclusion, and condition variables through a library API. `pthread_create` starts a thread, and `pthread_join` waits for a joinable thread to finish. A program should inspect return codes and clean up successfully created threads if later creation fails. Joining also provides a clear point at which the parent can safely consume a worker's completed result.

A **data race** occurs when concurrent accesses conflict, at least one is a write, and the required synchronization is missing. In C, a data race is not merely a guarantee of a slightly wrong count. It gives undefined behavior under the language memory model. Marking a variable `volatile` does not provide mutual exclusion or turn an increment into an atomic operation. A useful review asks who owns each writable object and which synchronization establishes visibility.

## A lost update has a precise mechanism

The expression `counter +=1` includes a read, arithmetic, and a write. Two workers can both read zero, both compute one, and both write one. Two increments then produce one. A **mutex** makes a protected region exclusive. A thread locks the mutex before reading the shared state, updates that state, and unlocks the mutex afterward. All threads that access that protected state must follow the same discipline.

An **atomic operation** provides an indivisible update with specified memory-order behavior. It can avoid a general-purpose lock for a supported operation. Atomicity alone does not preserve an invariant involving several separate variables. For example, updating a balance and a transaction count may require one larger synchronization protocol. A **barrier** makes participating threads wait until all participants reach a phase boundary. A barrier does not choose which thread owns an output element before that boundary.

A condition variable lets a thread sleep until another thread changes a predicate. The waiting thread tests the predicate while holding its mutex and waits in a loop. The loop is necessary because wakeups and competing consumers can invalidate the condition before the awakened thread proceeds. The signal is a notification to recheck state, not a stored replacement for the state itself. Lock ordering also matters because two threads that acquire the same locks in opposite orders can deadlock.

## Reductions reduce shared updates

A **reduction** combines many local contributions using an operation such as addition, minimum, or maximum. Each worker accumulates a private partial sum, and the partial sums are combined after the workers finish. If a loop has one million iterations and four workers, a lock per iteration introduces one million synchronized updates. Private accumulation needs only four partial results at the final combination point.

Associativity permits a reduction tree in exact arithmetic. Floating-point addition is not exactly associative, so changing the tree can change low-order bits. A correctness test should use a justified tolerance for floating-point output and an exact comparison for small integer output. A reproducible summation tree can improve repeatability, but it may restrict scheduling or cost additional work. The numerical and parallel tracks meet directly at this issue.

## OpenMP expresses common patterns with directives

OpenMP's fork-join model creates a team for a parallel region. A worksharing loop distributes iterations among that team. `parallel for` combines these concepts. A `reduction(+:sum)` clause gives each participating thread a private reduction value and combines the values using the named operation. `default(none)` makes data-sharing decisions explicit and is useful while learning. Variables introduced inside a parallel region are often private, but the exact rule depends on their construct and scope.

A `critical` region protects a named or unnamed critical section. An `atomic` directive protects a supported memory operation. A `single` region selects one thread to execute a region, and its default implicit barrier differs from using `nowait`. A `master` or `masked` region has its own participation rules and should not be treated as a general replacement for `single`. Tasks let a thread describe deferred work, while task dependencies and task groups describe ordering and completion.

Static scheduling has low overhead and works well for uniform iterations. Dynamic scheduling hands chunks to workers as they become idle. Guided scheduling begins with larger chunks and decreases them. Nested parallel regions and libraries that create their own threads can oversubscribe the machine. Thread affinity and false sharing can also dominate a logically correct loop. You should first verify the result, then inspect the number of active threads, and only afterward interpret timing.

## The native programs use a deterministic correctness target

The native examples sum integers from 1 through 1000, whose exact result is 500500. The pthread version uses four independent argument structures and four private partial sums. The parent joins all workers and combines those sums. The OpenMP version uses a reduction and requests four threads. The output confirms a result and a completed local execution. It does not establish a speedup because such a tiny loop is dominated by startup and measurement noise.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
values = list(range(1, 1001))
partials = [sum(values[i::4]) for i in range(4)]
wrong = 0
read_a, read_b = wrong, wrong
wrong = read_a + 1
wrong = read_b + 1
assert sum(partials) == 500500 and wrong == 1
print(json.dumps({'partial_sums': partials, 'correct_sum': sum(partials),
                  'modeled_lost_update': wrong}, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "partial_sums": [
    124750,
    125000,
    125250,
    125500
  ],
  "correct_sum": 500500,
  "modeled_lost_update": 1
}
```

The private partial sums combine to **500500**, while the explicit lost-update interleaving produces **1 after two increments**. The Python program models the interleaving and is not an attempt to force a Python race. The accompanying `pthread_sum.c` and `openmp_sum.c` exercise real native concurrency. Their compiler commands and captured output appear in `evidence/parallel/native.json`.

:::{admonition} Try it
:class: exercise
You can replace the integer input by values with very different magnitudes. Compare serial and pairwise sums, and explain why a valid reduction can change low-order floating-point bits.
:::

:::{admonition} Try it
:class: exercise
You can change the native pthread worker count from four to two. Verify every input still belongs to exactly one worker before interpreting any performance difference.
:::

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The pthread example uses the native API

```{literalinclude} ../../labs/parallel/pthread_sum.c
:language: c
```

### The OpenMP example uses the native API

```{literalinclude} ../../labs/parallel/openmp_sum.c
:language: c
```

## Summary

| Idea | What the lesson established |
|---|---|
| Ownership | Each writable location needs a clear owner or synchronization rule. |
| Reduction | Private accumulation minimizes synchronized shared updates. |
| Validation | A correct small result is a correctness check rather than a speedup claim. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Does volatile make a C increment atomic?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Volatile does not provide synchronization."
  },
  {
    "q": "Why does a condition wait use a loop?",
    "options": [
      "The predicate must be rechecked after waking.",
      "The mutex is unnecessary."
    ],
    "answer": 0,
    "explain": "Another thread or a spurious wakeup can leave the predicate false."
  },
  {
    "q": "What is the exact sum from 1 through 1000?",
    "options": [
      "500500.",
      "1000000."
    ],
    "answer": 0,
    "explain": "The formula is 1000 times 1001 divided by 2."
  },
  {
    "q": "Does a barrier assign exclusive ownership of an array element?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "A barrier synchronizes phases rather than individual updates."
  },
  {
    "q": "Can a floating-point reduction change low-order bits?",
    "options": [
      "Yes, its addition order can change.",
      "No, addition is exactly associative."
    ],
    "answer": 0,
    "explain": "Floating-point rounding depends on the evaluation order."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.
